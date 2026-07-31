#ifndef VOLVOXAI_ATTENTION_F32_TILED_H
#define VOLVOXAI_ATTENTION_F32_TILED_H

/*
 * Tiled F32 attention: the uncached path.
 *
 * The portable kernel visits each key once, which is optimal in arithmetic but
 * not in memory traffic: it holds one query and walks the whole memory, so every
 * K and V row is loaded once per query row — an arithmetic intensity of one.
 * That left a full forward at 5.0 GMAC/s against ONNX Runtime's 40, because ORT
 * runs attention as batched GEMM where each K/V load serves many query rows.
 *
 * Here the query axis lives in the vector lanes, sixteen at a time, so one
 * broadcast of a K or V element feeds sixteen lanes of useful work and the
 * softmax bookkeeping — running maximum, denominator, rescale — is vertical
 * arithmetic across queries with no horizontal reduction anywhere.
 *
 * Only Q is repacked, into [d][lane] order, and only once per query tile, so it
 * amortizes over every key tile.  K and V are read in place.  An earlier
 * revision packed K transposed per tile instead and measured 10.5 GMAC/s: the
 * pack was 2560 scattered scalar stores against 2560 FMA instructions of real
 * work.  Pack the small operand, not the large one.
 *
 * ---------------------------------------------------------------------------
 * Row-versus-full bit consistency
 *
 * test_incremental_runtime requires a query row computed alone to match the same
 * row inside a full forward.  Every lane's arithmetic is independent of which
 * lane it is and of how many lanes are occupied, so a query at lane 0 of a
 * one-row call and the same query at lane 5 of a full pass execute the identical
 * sequence of operations.  What remains is to keep the number of steps from
 * depending on the key count:
 *
 *  - Keys are consumed in fixed blocks of VX_ATTN_KB.  A block past the end
 *    reads a clamped (in-range) key row and is masked off, so a two-key tile and
 *    a sixty-key tile run the same instructions rather than falling into a
 *    scalar tail with different rounding.
 *  - Masked keys and unoccupied lanes contribute a probability of exactly zero,
 *    and adding zero is exact in IEEE, so extra blocks and extra tiles change
 *    neither the denominator nor the accumulator.
 *
 * The loops that do choose between a vector body and a masked tail are over
 * head_dim, which is fixed for a given node, so both callers take the same one.
 */

#if VX_SDPA_X86_AVX2

enum {
    VX_ATTN_MR = 16,        /* queries per tile: two ymm of lanes             */
    VX_ATTN_KB = 6,         /* keys per score block: 12 live accumulators     */
    VX_ATTN_KC = 48,        /* keys per tile, a whole number of key blocks    */
    VX_ATTN_KC_PADDED = 48
};

/* Low enough that exp() underflows it to zero, used as the running maximum's
 * initial value and as the stand-in score for a masked key. */
#define VX_ATTN_MASKED (-1.0e30f)

/*
 * One output block: four queries by N channel vectors, reducing over the tile's
 * keys.  N is a literal at every use so the bounds fold away and the
 * accumulators stay in registers.  N=3 gives twelve accumulators against three
 * V loads and four broadcasts — 0.58 loads per FMA, comfortably FMA-bound —
 * and head_dim=40 is covered exactly as 24 + 16.
 */
#define VX_ATTN_OUT_BLOCK(N)                                                   \
    {                                                                          \
        __m256 acc[4][N];                                                      \
        int _q, _c;                                                            \
        for (_q = 0; _q < 4; _q++)                                             \
            for (_c = 0; _c < N; _c++)                                         \
                acc[_q][_c] = _mm256_loadu_ps(                                 \
                    base + (long)_q * head_dim + d + _c * 8);                  \
        for (j = 0; j < kc; j++) {                                             \
            const float* _p = probability + (long)j * VX_ATTN_MR;              \
            const float* _v = v_in + (long)(k0 + j) * qkv_stride +             \
                head_offset + d;                                               \
            __m256 _vv[N];                                                     \
            for (_c = 0; _c < N; _c++) _vv[_c] = _mm256_loadu_ps(_v + _c * 8); \
            for (_q = 0; _q < 4; _q++) {                                       \
                const __m256 _pb = _mm256_broadcast_ss(_p + _q);               \
                for (_c = 0; _c < N; _c++)                                     \
                    acc[_q][_c] = _mm256_fmadd_ps(_pb, _vv[_c], acc[_q][_c]);  \
            }                                                                  \
        }                                                                      \
        for (_q = 0; _q < 4; _q++)                                             \
            for (_c = 0; _c < N; _c++)                                         \
                _mm256_storeu_ps(base + (long)_q * head_dim + d + _c * 8,      \
                                 acc[_q][_c]);                                 \
    }


/*
 * 2^x for the softmax, where x is never positive.
 *
 * exp() was 21% of this kernel — the third-largest cost after the two GEMMs —
 * and most of a general-purpose exp is spent on cases softmax cannot produce.
 * The argument here is always `score - running_maximum`, so it is non-positive
 * by construction, which removes the overflow clamp and both underflow guards:
 * one floor at -126 keeps the reconstructed exponent in [1,127], so no branch,
 * no zero-exponent test, and no final select.
 *
 * Working in base two rather than base e removes the change-of-base multiply as
 * well: log2(e) is folded into the score scale, which the score GEMM was already
 * applying, so it costs nothing.  And because this kernel is checked against an
 * independent double reference with a tolerance rather than against the scalar
 * kernel bit-for-bit, the range reduction can use a plain round-to-nearest
 * instead of reproducing accurate_expf's round-half-away-from-zero.
 *
 * Twelve operations against the previous twenty-one, at the same ~1e-6 relative
 * accuracy: the degree-five minimax of 2^f on [-1/2, 1/2] truncates at a term
 * worth 1.2e-6 there.
 *
 * A masked key must still contribute exactly zero, and 2^-126 is small but not
 * zero.  The callers that can mask already AND the result with a keep mask, and
 * the unmasked path never exponentiates a masked score, so exactness holds.
 */
static inline VX_SDPA_TARGET_AVX2 __m256 vx_attention_exp2_avx2(__m256 x) {
    __m256 n, f, p;
    __m256i e;
    x = _mm256_max_ps(x, _mm256_set1_ps(-126.0f));
    n = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    f = _mm256_sub_ps(x, n);
    p = _mm256_set1_ps(0.0013333558f);                        /* ln2^5/120 */
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.0096181290f)); /* ln2^4/24  */
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.0555041087f)); /* ln2^3/6   */
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.2402265070f)); /* ln2^2/2   */
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.6931471805f)); /* ln2       */
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    e = _mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

static VX_SDPA_TARGET_AVX2 void vx_attention_tiled_avx2(
        const float* q_in, const float* k_in, const float* v_in,
        long qkv_stride, float* output, long out_stride,
        int seq_q, int seq_kv, int num_heads, int head_dim, float scale,
        const int32_t* mask, int mask_mode, int causal) {
    /* [d][lane]: the score GEMM reads sixteen queries with two vector loads. */
    float queries[VX_SDPA_MAX_HEAD_DIM * VX_ATTN_MR];
    /* [key][lane], which is also the layout the output GEMM wants. */
    float scores[VX_ATTN_KC_PADDED * VX_ATTN_MR];
    /* [lane][d]: one contiguous output row per query. */
    float accumulator[VX_ATTN_MR * VX_SDPA_MAX_HEAD_DIM];
    float corrections[VX_ATTN_MR];
    float denominators[VX_ATTN_MR];
    const __m256 lanes_low = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f,
                                           3.0f, 2.0f, 1.0f, 0.0f);
    const __m256 lanes_high = _mm256_set_ps(15.0f, 14.0f, 13.0f, 12.0f,
                                            11.0f, 10.0f, 9.0f, 8.0f);
    const __m256 masked = _mm256_set1_ps(VX_ATTN_MASKED);
    /* Channel tails: head_dim is not always a multiple of eight, and a scalar
     * remainder is ruinous — at head_dim=20 four of twenty channels fell to a
     * scalar loop and the kernel ran three times slower.  A lane mask keeps the
     * remainder vectorized. */
    int32_t tail_lanes[8];

    for (int i = 0; i < 8; i++)
        tail_lanes[i] = i < (head_dim & 7) ? -1 : 0;
    {
        const __m256i tail_mask = _mm256_loadu_si256((const __m256i*)tail_lanes);

    for (int head = 0; head < num_heads; head++) {
        const long head_offset = (long)head * head_dim;
        for (int q0 = 0; q0 < seq_q; q0 += VX_ATTN_MR) {
            const int mr = seq_q - q0 < VX_ATTN_MR ? seq_q - q0 : VX_ATTN_MR;
            /* Under causal masking no key past the tile's last query can reach
             * any lane in it, so the whole tail of the memory is skipped. */
            const int key_limit = causal
                ? (q0 + mr < seq_kv ? q0 + mr : seq_kv)
                : seq_kv;
            /* Lanes beyond the tile's occupancy are masked, never branched
             * around, so the GEMMs always run their full width. */
            const __m256 valid_low = _mm256_cmp_ps(
                lanes_low, _mm256_set1_ps((float)mr), _CMP_LT_OQ);
            const __m256 valid_high = _mm256_cmp_ps(
                lanes_high, _mm256_set1_ps((float)mr), _CMP_LT_OQ);
            /* When every lane is occupied and nothing can be masked the keep
             * mask is all ones for every key, so the select is the identity and
             * the probability loop needs no correction. */
            const int unmasked = mr == VX_ATTN_MR && !causal &&
                (!mask || mask_mode == 0);
            /* A key-only mask is the same for every lane, so a whole block is
             * either kept as computed or replaced wholesale.  Recognizing that
             * turns three vector ops per key into one predictable branch, which
             * matters because this — not the unmasked case — is what the decoder
             * and encoder actually run. */
            const int key_only_mask = mr == VX_ATTN_MR && !causal &&
                mask && mask_mode == 1;
            __m256 max_low = masked, max_high = masked;
            __m256 sum_low = _mm256_setzero_ps();
            __m256 sum_high = _mm256_setzero_ps();
            int r, d, j;

            for (r = 0; r < VX_ATTN_MR; r++) {
                const int query = q0 + r < seq_q ? q0 + r : seq_q - 1;
                const float* source =
                    q_in + (long)query * qkv_stride + head_offset;
                const float valid = r < mr ? 1.0f : 0.0f;
                for (d = 0; d < head_dim; d++)
                    queries[(long)d * VX_ATTN_MR + r] = source[d] * valid;
                for (d = 0; d < head_dim; d++)
                    accumulator[(long)r * head_dim + d] = 0.0f;
            }

            for (int k0 = 0; k0 < key_limit; k0 += VX_ATTN_KC) {
                const int kc = key_limit - k0 < VX_ATTN_KC
                    ? key_limit - k0 : VX_ATTN_KC;
                const int kc_blocked =
                    (kc + VX_ATTN_KB - 1) / VX_ATTN_KB * VX_ATTN_KB;
                __m256 tile_low = masked, tile_high = masked;
                __m256 new_low, new_high;

                /* --- scores: six keys per block, sixteen queries in the lanes,
                 * reduction over head_dim.  Twelve accumulators; two Q loads and
                 * six K broadcasts feed twelve FMAs. --- */
                for (j = 0; j < kc_blocked; j += VX_ATTN_KB) {
                    __m256 a0 = _mm256_setzero_ps(), b0 = _mm256_setzero_ps();
                    __m256 a1 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
                    __m256 a2 = _mm256_setzero_ps(), b2 = _mm256_setzero_ps();
                    __m256 a3 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
                    __m256 a4 = _mm256_setzero_ps(), b4 = _mm256_setzero_ps();
                    __m256 a5 = _mm256_setzero_ps(), b5 = _mm256_setzero_ps();
                    const float* rows[VX_ATTN_KB];
                    int jj;
                    for (jj = 0; jj < VX_ATTN_KB; jj++) {
                        /* Blocks past the tile's end read a clamped row and are
                         * masked off below; this keeps one code path. */
                        const int key = k0 + j + jj < seq_kv
                            ? k0 + j + jj : seq_kv - 1;
                        rows[jj] = k_in + (long)key * qkv_stride + head_offset;
                    }
                    for (d = 0; d < head_dim; d++) {
                        const float* slot = queries + (long)d * VX_ATTN_MR;
                        const __m256 low = _mm256_loadu_ps(slot);
                        const __m256 high = _mm256_loadu_ps(slot + 8);
                        __m256 key;
                        key = _mm256_broadcast_ss(rows[0] + d);
                        a0 = _mm256_fmadd_ps(low, key, a0);
                        b0 = _mm256_fmadd_ps(high, key, b0);
                        key = _mm256_broadcast_ss(rows[1] + d);
                        a1 = _mm256_fmadd_ps(low, key, a1);
                        b1 = _mm256_fmadd_ps(high, key, b1);
                        key = _mm256_broadcast_ss(rows[2] + d);
                        a2 = _mm256_fmadd_ps(low, key, a2);
                        b2 = _mm256_fmadd_ps(high, key, b2);
                        key = _mm256_broadcast_ss(rows[3] + d);
                        a3 = _mm256_fmadd_ps(low, key, a3);
                        b3 = _mm256_fmadd_ps(high, key, b3);
                        key = _mm256_broadcast_ss(rows[4] + d);
                        a4 = _mm256_fmadd_ps(low, key, a4);
                        b4 = _mm256_fmadd_ps(high, key, b4);
                        key = _mm256_broadcast_ss(rows[5] + d);
                        a5 = _mm256_fmadd_ps(low, key, a5);
                        b5 = _mm256_fmadd_ps(high, key, b5);
                    }
                    {
                        /* log2(e) folded in: the softmax below works in base two. */
                        const __m256 scales =
                            _mm256_set1_ps(scale * 1.44269504088896f);
                        __m256 block[VX_ATTN_KB * 2];
                        block[0] = _mm256_mul_ps(a0, scales);
                        block[1] = _mm256_mul_ps(b0, scales);
                        block[2] = _mm256_mul_ps(a1, scales);
                        block[3] = _mm256_mul_ps(b1, scales);
                        block[4] = _mm256_mul_ps(a2, scales);
                        block[5] = _mm256_mul_ps(b2, scales);
                        block[6] = _mm256_mul_ps(a3, scales);
                        block[7] = _mm256_mul_ps(b3, scales);
                        block[8] = _mm256_mul_ps(a4, scales);
                        block[9] = _mm256_mul_ps(b4, scales);
                        block[10] = _mm256_mul_ps(a5, scales);
                        block[11] = _mm256_mul_ps(b5, scales);
                        for (jj = 0; jj < VX_ATTN_KB; jj++) {
                            const int index = j + jj;
                            const int key = k0 + index;
                            __m256 low = block[jj * 2];
                            __m256 high = block[jj * 2 + 1];
                            if (index >= kc) {
                                low = masked;
                                high = masked;
                            } else if (key_only_mask) {
                                if (!mask[key]) {
                                    low = masked;
                                    high = masked;
                                }
                            } else if (!unmasked) {
                                __m256 keep_low = valid_low;
                                __m256 keep_high = valid_high;
                                if (causal) {
                                    const __m256 limit =
                                        _mm256_set1_ps((float)(key - q0));
                                    keep_low = _mm256_and_ps(keep_low,
                                        _mm256_cmp_ps(lanes_low, limit, _CMP_GE_OQ));
                                    keep_high = _mm256_and_ps(keep_high,
                                        _mm256_cmp_ps(lanes_high, limit, _CMP_GE_OQ));
                                }
                                if (mask) {
                                    if (mask_mode == 1) {
                                        if (!mask[key]) {
                                            keep_low = _mm256_setzero_ps();
                                            keep_high = _mm256_setzero_ps();
                                        }
                                    } else if (mask_mode == 2) {
                                        float per_lane[VX_ATTN_MR];
                                        int lane;
                                        for (lane = 0; lane < VX_ATTN_MR; lane++)
                                            per_lane[lane] =
                                                (lane < mr &&
                                                 mask[(long)(q0 + lane) * seq_kv +
                                                      key]) ? -1.0f : 0.0f;
                                        keep_low = _mm256_and_ps(keep_low,
                                            _mm256_loadu_ps(per_lane));
                                        keep_high = _mm256_and_ps(keep_high,
                                            _mm256_loadu_ps(per_lane + 8));
                                    } else if (mask_mode != 0) {
                                        keep_low = _mm256_setzero_ps();
                                        keep_high = _mm256_setzero_ps();
                                    }
                                }
                                low = _mm256_or_ps(_mm256_and_ps(keep_low, low),
                                    _mm256_andnot_ps(keep_low, masked));
                                high = _mm256_or_ps(_mm256_and_ps(keep_high, high),
                                    _mm256_andnot_ps(keep_high, masked));
                            }
                            _mm256_storeu_ps(
                                scores + (long)index * VX_ATTN_MR, low);
                            _mm256_storeu_ps(
                                scores + (long)index * VX_ATTN_MR + 8, high);
                            tile_low = _mm256_max_ps(tile_low, low);
                            tile_high = _mm256_max_ps(tile_high, high);
                        }
                    }
                }

                /* --- rescale the running state, then turn scores into
                 * probabilities.  All of this is vertical across queries. --- */
                new_low = _mm256_max_ps(max_low, tile_low);
                new_high = _mm256_max_ps(max_high, tile_high);
                {
                    const __m256 low = vx_attention_exp2_avx2(
                        _mm256_sub_ps(max_low, new_low));
                    const __m256 high = vx_attention_exp2_avx2(
                        _mm256_sub_ps(max_high, new_high));
                    sum_low = _mm256_mul_ps(sum_low, low);
                    sum_high = _mm256_mul_ps(sum_high, high);
                    _mm256_storeu_ps(corrections, low);
                    _mm256_storeu_ps(corrections + 8, high);
                    /* Every tile after a new maximum has to rescale what earlier
                     * tiles accumulated.  When no lane found a larger score the
                     * factors are all exactly one and the pass is skipped. */
                    if (_mm256_movemask_ps(_mm256_and_ps(
                            _mm256_cmp_ps(low, _mm256_set1_ps(1.0f), _CMP_EQ_OQ),
                            _mm256_cmp_ps(high, _mm256_set1_ps(1.0f),
                                          _CMP_EQ_OQ))) != 0xff) {
                        for (r = 0; r < VX_ATTN_MR; r++) {
                            float* row = accumulator + (long)r * head_dim;
                            const __m256 factor =
                                _mm256_set1_ps(corrections[r]);
                            for (d = 0; d + 8 <= head_dim; d += 8)
                                _mm256_storeu_ps(row + d, _mm256_mul_ps(
                                    _mm256_loadu_ps(row + d), factor));
                            for (; d < head_dim; d++) row[d] *= corrections[r];
                        }
                    }
                }
                max_low = new_low;
                max_high = new_high;
                for (j = 0; j < kc; j++) {
                    float* slot = scores + (long)j * VX_ATTN_MR;
                    const __m256 low = _mm256_loadu_ps(slot);
                    const __m256 high = _mm256_loadu_ps(slot + 8);
                    __m256 plow = vx_attention_exp2_avx2(
                        _mm256_sub_ps(low, new_low));
                    __m256 phigh = vx_attention_exp2_avx2(
                        _mm256_sub_ps(high, new_high));
                    /* A masked score sits at VX_ATTN_MASKED, whose distance
                     * below the maximum underflows exp() to zero on its own —
                     * except when the maximum is itself VX_ATTN_MASKED because
                     * the lane has seen no key yet, which would give one. */
                    if (!unmasked) {
                        plow = _mm256_and_ps(plow,
                            _mm256_cmp_ps(low, masked, _CMP_NEQ_OQ));
                        phigh = _mm256_and_ps(phigh,
                            _mm256_cmp_ps(high, masked, _CMP_NEQ_OQ));
                    }
                    _mm256_storeu_ps(slot, plow);
                    _mm256_storeu_ps(slot + 8, phigh);
                    sum_low = _mm256_add_ps(sum_low, plow);
                    sum_high = _mm256_add_ps(sum_high, phigh);
                }

                /* --- output: four queries by three channel vectors, reduction
                 * over the tile's keys.  Three V loads and four probability
                 * broadcasts feed twelve FMAs.  V is contiguous along d and read
                 * in place; probabilities past the tile are zero. --- */
                /* Query groups beyond the tile's occupancy hold only masked
                 * lanes, whose probabilities are zero, so their contribution is
                 * zero and skipping them changes no occupied lane's result.  A
                 * single-row call is otherwise paying for sixteen. */
                for (r = 0; r < mr; r += 4) {
                    float* base = accumulator + (long)r * head_dim;
                    const float* probability = scores + r;
                    d = 0;
                    for (; d + 24 <= head_dim; d += 24) {
                        VX_ATTN_OUT_BLOCK(3)
                    }
                    for (; d + 16 <= head_dim; d += 16) {
                        VX_ATTN_OUT_BLOCK(2)
                    }
                    for (; d + 8 <= head_dim; d += 8) {
                        VX_ATTN_OUT_BLOCK(1)
                    }
                    if (head_dim & 7) {
                        __m256 c0 = _mm256_maskload_ps(base + 0 * head_dim + d,
                                                       tail_mask);
                        __m256 c1 = _mm256_maskload_ps(base + 1 * head_dim + d,
                                                       tail_mask);
                        __m256 c2 = _mm256_maskload_ps(base + 2 * head_dim + d,
                                                       tail_mask);
                        __m256 c3 = _mm256_maskload_ps(base + 3 * head_dim + d,
                                                       tail_mask);
                        for (j = 0; j < kc; j++) {
                            const float* slot =
                                probability + (long)j * VX_ATTN_MR;
                            const __m256 value = _mm256_maskload_ps(
                                v_in + (long)(k0 + j) * qkv_stride +
                                head_offset + d, tail_mask);
                            c0 = _mm256_fmadd_ps(
                                _mm256_broadcast_ss(slot + 0), value, c0);
                            c1 = _mm256_fmadd_ps(
                                _mm256_broadcast_ss(slot + 1), value, c1);
                            c2 = _mm256_fmadd_ps(
                                _mm256_broadcast_ss(slot + 2), value, c2);
                            c3 = _mm256_fmadd_ps(
                                _mm256_broadcast_ss(slot + 3), value, c3);
                        }
                        _mm256_maskstore_ps(base + 0 * head_dim + d, tail_mask, c0);
                        _mm256_maskstore_ps(base + 1 * head_dim + d, tail_mask, c1);
                        _mm256_maskstore_ps(base + 2 * head_dim + d, tail_mask, c2);
                        _mm256_maskstore_ps(base + 3 * head_dim + d, tail_mask, c3);
                    }
                }
            }

            _mm256_storeu_ps(denominators, sum_low);
            _mm256_storeu_ps(denominators + 8, sum_high);
            for (r = 0; r < mr; r++) {
                float* out = output + (long)(q0 + r) * out_stride + head_offset;
                /* A zero denominator means every key was masked for this query,
                 * which the portable kernel reports as a zero row. */
                if (denominators[r] > 0.0f) {
                    const float inverse = 1.0f / denominators[r];
                    for (d = 0; d < head_dim; d++)
                        out[d] = accumulator[(long)r * head_dim + d] * inverse;
                } else {
                    for (d = 0; d < head_dim; d++) out[d] = 0.0f;
                }
            }
        }
    }
    }
}

#endif /* VX_SDPA_X86_AVX2 */

#endif
