#include "concat_order.h"
#if VOLVOXAI_ENABLE_TRAINING
#include "generated/operator_param_registry_full.h"
#else
#include "generated/operator_param_registry.h"
#endif

/* The dynamically reserved table is a power of two and kept at or below 50%
 * occupancy. Entries store tensor_index + 1 so zero remains the empty
 * sentinel. */

static uint32_t tensor_name_hash(const char* name) {
    uint32_t hash = UINT32_C(2166136261);
    for (const unsigned char* byte = (const unsigned char*)name; *byte; byte++) {
        hash ^= *byte;
        hash *= UINT32_C(16777619);
    }
    return hash;
}

static int tensor_name_index_insert(int tensor_index) {
    if (tensor_index < 0 || tensor_index >= g_nt || !g_t[tensor_index].name[0]) return -1;
    if (!g_tensor_name_index_capacity) return -1;
    size_t slot = (size_t)tensor_name_hash(g_t[tensor_index].name) &
                  (g_tensor_name_index_capacity - 1u);
    for (size_t probe = 0; probe < g_tensor_name_index_capacity; probe++) {
        int encoded = g_tensor_name_index[slot];
        if (!encoded) {
            g_tensor_name_index[slot] = tensor_index + 1;
            return 0;
        }
        /* Preserve the former linear lookup's first-entry behavior if an
         * invalid internal caller constructs duplicate names. */
        int existing = encoded - 1;
        if (existing >= 0 && existing < g_nt &&
            !strcmp(g_t[existing].name, g_t[tensor_index].name)) return 0;
        slot = (slot + 1u) & (g_tensor_name_index_capacity - 1u);
    }
    return -1;
}

void volvoxai_engine_tensor_name_index_rebuild(void) {
    if (g_tensor_name_index_count == g_nt) return;
    if (g_tensor_name_index_capacity)
        memset(g_tensor_name_index, 0,
               g_tensor_name_index_capacity * sizeof(g_tensor_name_index[0]));
    for (int index = 0; index < g_nt; index++) {
        if (tensor_name_index_insert(index) != 0) {
            /* Valid runtime tables cannot reach this at <= 50% occupancy.
             * Leave the count stale so a later lookup retries after the
             * caller repairs an invalid table. */
            g_tensor_name_index_count = -1;
            return;
        }
    }
    g_tensor_name_index_count = g_nt;
}

void volvoxai_engine_tensor_name_index_invalidate(void) {
    g_tensor_name_index_count = -1;
}

void volvoxai_engine_tensor_name_index_add(int tensor_index) {
    /* Appending is the common load/build path.  Keep it O(1) when the old
     * table was current; otherwise defer one full rebuild until lookup. */
    if (tensor_index != g_nt - 1 || g_tensor_name_index_count != tensor_index ||
        tensor_name_index_insert(tensor_index) != 0) {
        volvoxai_engine_tensor_name_index_invalidate();
        return;
    }
    g_tensor_name_index_count = g_nt;
}

T* t_find(const char* name) {
    if (!name || !name[0]) return NULL;
    if (g_tensor_name_index_count == g_nt && g_tensor_name_index_capacity) {
        size_t slot = (size_t)tensor_name_hash(name) &
                      (g_tensor_name_index_capacity - 1u);
        for (size_t probe = 0; probe < g_tensor_name_index_capacity; probe++) {
            int encoded = g_tensor_name_index[slot];
            if (!encoded) return NULL;
            int index = encoded - 1;
            if (index < 0 || index >= g_nt) break;
            if (!strcmp(g_t[index].name, name)) return &g_t[index];
            slot = (slot + 1u) & (g_tensor_name_index_capacity - 1u);
        }
    }

    /* Structural writers invalidate the index while they rename or compact
     * tensors. Lookups can run under either the model or metadata mutex, so a
     * reader must never lazily rebuild shared state under only one of them.
     * The mutation boundary publishes a rebuilt index; until then this
     * read-only fallback preserves the original lookup semantics. */
    for (int index = 0; index < g_nt; index++)
        if (!strcmp(g_t[index].name, name)) return &g_t[index];
    return NULL;
}
static int removed_graph_output_index(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < g_removed_graph_output_count; i++) {
        if (!strcmp(g_removed_graph_outputs[i], name)) return i;
    }
    return -1;
}

static void removed_graph_output_add(const char* name) {
    if (!name || !name[0] || removed_graph_output_index(name) >= 0 ||
        (size_t)g_removed_graph_output_count >= g_tensor_capacity) return;
    strncpy(g_removed_graph_outputs[g_removed_graph_output_count], name, 127);
    g_removed_graph_outputs[g_removed_graph_output_count][127] = 0;
    g_removed_graph_output_count++;
}

static void removed_graph_output_remove(const char* name) {
    int index = removed_graph_output_index(name);
    if (index < 0) return;
    memmove(&g_removed_graph_outputs[index], &g_removed_graph_outputs[index + 1],
            (size_t)(g_removed_graph_output_count - index - 1) * sizeof(g_removed_graph_outputs[0]));
    g_removed_graph_output_count--;
    g_removed_graph_outputs[g_removed_graph_output_count][0] = 0;
}

int volvoxai_engine_tensor_is_removed_output(const char* name) {
    return removed_graph_output_index(name) >= 0;
}
static int t_index(const T* t) {
    if (!t) return -1;
    ptrdiff_t idx = t - g_t;
    return (idx >= 0 && idx < g_nt) ? (int)idx : -1;
}
static long numel_of(const int* shape, int ndim) {
    if (ndim < 0 || ndim > 8 || (ndim > 0 && !shape)) return -1;
    long n = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0 || n > LONG_MAX / shape[i]) return -1;
        n *= shape[i];
    }
    return n;
}

void volvoxai_engine_clear_qlinear_metadata(void) {
    for (size_t i = 0; i < g_node_capacity; i++) {
        free(g_qlinear_meta[i].weight_scales);
        free(g_qlinear_meta[i].weight_zero_points);
        free(g_qlinear_meta[i].bias);
        free(g_qlinear_meta[i].packed_weight);
        memset(&g_qlinear_meta[i], 0, sizeof(g_qlinear_meta[i]));
    }
}

void volvoxai_engine_clear_qconv_metadata(void) {
    for (size_t i = 0; i < g_node_capacity; i++) {
        free(g_qconv_meta[i].weight_scales);
        free(g_qconv_meta[i].weight_zero_points);
        free(g_qconv_meta[i].bias);
        free(g_qconv_meta[i].packed_weight);
        free(g_qconv_meta[i].small_c_packed_weight);
        memset(&g_qconv_meta[i], 0, sizeof(g_qconv_meta[i]));
    }
}

void volvoxai_engine_clear_qembedding_metadata(void) {
    for (size_t i = 0; i < g_node_capacity; i++) {
        free(g_qembedding_meta[i].weight_scales);
        free(g_qembedding_meta[i].weight_zero_points);
        memset(&g_qembedding_meta[i], 0, sizeof(g_qembedding_meta[i]));
    }
}

void materialize_tensor_f32(T* t) {
    widen_f16_tensor_to_f32(t);
}

static size_t dtype_size(int dtype) {
    switch (dtype) {
        case T_I8: return 1;
        case T_U8: return 1;
        case T_F16: return 2;
        case T_I32: return 4;
        case T_F32: return 4;
        default: return 0;
    }
}

/*
 * Bits per element.
 *
 * `dtype_size` answers in whole bytes and returns zero for anything it cannot,
 * which is right for allocation and useless for a 4-bit weight: two of those
 * share a byte, so no whole-byte answer exists. Every address in the decode and
 * paging paths is `count * element size`, and that product is the one place a
 * sub-byte dtype has to be understood -- so it is stated once, here, rather
 * than at each of the twenty-odd sites that multiply by `elem_size`.
 *
 * The sub-byte entries name dtypes the enum already carries and no kernel
 * implements yet. They are here so that adding such a kernel is a change to the
 * kernel, not a hunt for every place a row's extent was assumed to be a whole
 * number of bytes per element.
 */
static size_t dtype_bits(int dtype) {
    switch (dtype) {
        case VX_DTYPE_F4: return 4;
        case VX_DTYPE_F6_E2M3:
        case VX_DTYPE_F6_E3M2: return 6;
        case T_I8:
        case T_U8: return 8;
        case T_F16: return 16;
        case T_I32:
        case T_F32: return 32;
        default: return 0;
    }
}

/*
 * Bytes occupied by `count` contiguous elements of `dtype`, or 0 on failure.
 *
 * Refuses rather than rounds when the run does not end on a byte boundary. A
 * 4-bit row of odd width shares its last byte with the row after it, so a
 * caller handed a rounded answer would name one row and touch two -- and every
 * row here is a unit that gets copied, bound, gathered and scattered on its
 * own. "This row is not separately addressable" is a real answer for such a
 * shape, and the callers already have a host path to fall back to.
 */
static int dtype_span_bytes(int dtype, long count, size_t* out_bytes) {
    size_t bits = dtype_bits(dtype);
    uint64_t total;
    if (!out_bytes || !bits || count < 0) return 0;
    if ((uint64_t)count > (uint64_t)UINT64_MAX / bits) return 0;
    total = (uint64_t)count * bits;
    if (total % 8u) return 0;
    total /= 8u;
    if (total > SIZE_MAX) return 0;
    *out_bytes = (size_t)total;
    return 1;
}

/* The same question asked of a tensor, cross-checked against the whole-byte
 * element size it was allocated with. A tensor whose two answers disagree is
 * one this path cannot address, not one to pick a winner for. */
static int tensor_span_bytes(const T* tensor, long count, size_t* out_bytes) {
    if (!tensor || !dtype_span_bytes(tensor->dtype, count, out_bytes)) return 0;
    if (tensor->elem_size) {
        if ((uint64_t)count > (uint64_t)SIZE_MAX / tensor->elem_size) return 0;
        if ((size_t)count * tensor->elem_size != *out_bytes) return 0;
    }
    return 1;
}

/* A whole tensor's bytes, which is the bound every gather and scatter checks
 * its rows against. */
static int tensor_total_bytes(const T* tensor, size_t* out_bytes) {
    return tensor && tensor_span_bytes(tensor, tensor->numel, out_bytes);
}

static int tensor_dtype_from_safetensor(VxDataType dtype) {
    switch (dtype) {
        case VX_DTYPE_F32:
        case VX_DTYPE_F16:
        case VX_DTYPE_I8:
        case VX_DTYPE_U8:
        case VX_DTYPE_I32:
            return dtype;
        case SAFETENSORS_DTYPE_BOOL:
        case SAFETENSORS_DTYPE_F4:
        case SAFETENSORS_DTYPE_F6_E2M3:
        case SAFETENSORS_DTYPE_F6_E3M2:
        case SAFETENSORS_DTYPE_F8_E5M2:
        case SAFETENSORS_DTYPE_F8_E4M3:
        case SAFETENSORS_DTYPE_F8_E8M0:
        case SAFETENSORS_DTYPE_F8_E4M3FNUZ:
        case SAFETENSORS_DTYPE_F8_E5M2FNUZ:
        case SAFETENSORS_DTYPE_I16:
        case SAFETENSORS_DTYPE_U16:
        case SAFETENSORS_DTYPE_BF16:
        case SAFETENSORS_DTYPE_U32:
        case SAFETENSORS_DTYPE_C64:
        case SAFETENSORS_DTYPE_F64:
        case SAFETENSORS_DTYPE_I64:
        case SAFETENSORS_DTYPE_U64:
        case SAFETENSORS_DTYPE_UNKNOWN:
        default: return -1;
    }
}

static float f16_to_f32_scalar(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                e--;
            }
            mant &= 0x03ffu;
            bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t f16_load_bits(const void* data, long idx) {
    const unsigned char* p = (const unsigned char*)data + idx * 2;
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static float tensor_scalar_f32(const T* t, long idx) {
    if (!t) return 0.0f;
    if (t->dtype == T_F16) return f16_to_f32_scalar(f16_load_bits(t->data, idx));
    if (t->dtype == T_F32) {
        float value;
        memcpy(&value, (const unsigned char*)t->data + (size_t)idx * sizeof(float), sizeof(value));
        return value;
    }
    return 0.0f;
}

static float tensor_scalar_any(const T* t, long idx) {
    if (!t || idx < 0 || idx >= t->numel) return 0.0f;
    if (t->dtype == T_F32 || t->dtype == T_F16) return tensor_scalar_f32(t, idx);
    if (t->dtype == T_I8) return (float)((const int8_t*)t->data)[idx];
    if (t->dtype == T_U8) return (float)((const uint8_t*)t->data)[idx];
    if (t->dtype == T_I32) {
        int32_t value;
        memcpy(&value, (const unsigned char*)t->data + (size_t)idx * sizeof(value), sizeof(value));
        return (float)value;
    }
    return 0.0f;
}

/* CPU fallback may widen an owned F16 activation into a new F32 allocation.
 * Public backends are allowed to retain the old host pointer as an opaque
 * device-storage key until reset/teardown, so defer freeing replaced storage
 * until that lifetime boundary.  A tensor widens at most once. */
static void release_retired_f16_storage(void) {
    for (int index = 0; index < g_retired_f16_storage_count; index++)
        free(g_retired_f16_storage[index]);
    if (g_tensor_capacity)
        memset(g_retired_f16_storage, 0,
               g_tensor_capacity * sizeof(g_retired_f16_storage[0]));
    g_retired_f16_storage_count = 0;
}

static int widen_f16_tensor_to_f32(T* t) {
    if (!t || t->dtype != T_F16) return 0;
    float* dst = (float*)malloc((size_t)t->numel * sizeof(float));
    if (!dst) return -1;
    for (long i = 0; i < t->numel; i++) dst[i] = f16_to_f32_scalar(f16_load_bits(t->data, i));
    if (t->owns) {
        if ((size_t)g_retired_f16_storage_count >= g_tensor_capacity) {
            free(dst);
            return -1;
        }
        g_retired_f16_storage[g_retired_f16_storage_count++] = t->data;
    }
    t->data = dst;
    t->dtype = T_F32;
    t->elem_size = sizeof(float);
    t->owns = 1;
    return 1;
}

static T* t_add_ex(const char* name, const int* shape, int ndim, void* data, int dtype) {
    long numel = numel_of(shape, ndim);
    /*
     * `elem_size` is zero for a sub-byte dtype, and that is the honest answer
     * rather than a failure: two 4-bit values share a byte, so there is no
     * whole-byte size per element. Storage is sized from the total instead,
     * which `dtype_span_bytes` can express and a per-element size cannot.
     *
     * Everything downstream reads the zero the same way -- `tensor_span_bytes`
     * skips its cross-check, and `physical_byte_tensor` still demands one byte
     * per element, so a sub-byte activation reaches no W8A8 kernel until one is
     * written for it. Allocating it correctly is what lets that kernel be the
     * only change when it arrives.
     */
    size_t elem_size = dtype_size(dtype);
    size_t storage_bytes;
    if (!name || !name[0] || strlen(name) >= sizeof(((T*)0)->name) ||
        g_nt < 0 || (size_t)g_nt >= g_tensor_capacity || numel < 0 ||
        !dtype_span_bytes(dtype, numel > 0 ? numel : 1, &storage_bytes) ||
        !storage_bytes) return NULL;
    void* storage = data;
    if (!storage) {
        VxEngineState* state = vx_engine_state_current();
        uint64_t limit = state->shape_policy.max_activation_capacity_bytes;
        if (!g_loaded && limit && (state->bootstrap_activation_bytes > limit ||
            storage_bytes > limit - state->bootstrap_activation_bytes)) {
            state->activation_budget_exceeded = 1;
            return NULL;
        }
        storage = calloc(storage_bytes, 1);
        if (!storage) return NULL;
        if (!g_loaded) state->bootstrap_activation_bytes += storage_bytes;
    }
    T* t = &g_t[g_nt];
    memset(t, 0, sizeof(*t));
    memcpy(t->name, name, strlen(name) + 1);
    t->ndim = ndim; for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    t->numel = numel;
    t->dtype = dtype;
    t->elem_size = elem_size;
    t->owns = data ? 0 : 1;
    t->data = (float*)storage;
    g_nt++;
    volvoxai_engine_tensor_name_index_add(g_nt - 1);
    return t;
}

static T* t_add(const char* name, const int* shape, int ndim, float* data) {
    return t_add_ex(name, shape, ndim, data, T_F32);
}

// Row window [r0, r0+rc) over a tensor whose leading (batch-folded) dim is the sequence.
static void seq_range(long total_rows, int* r0, int* rc) {
    if (g_active_row >= 0) { *r0 = g_active_row; *rc = 1; }
    else {
        *r0 = 0;
        *rc = (g_prefix_rows > 0 &&
               total_rows == g_prefix_row_capacity &&
               g_prefix_rows < total_rows)
            ? g_prefix_rows : (int)total_rows;
    }
}

/*
 * Row narrowing for a declared batch, where `seq_range` cannot reach.
 *
 * `seq_range` answers with a start and a count, which says a step's rows are
 * one contiguous span. That is true of every one-lane step and of no other
 * kind: with `lanes` sequences retained in one `[lanes,S,W]` activation the
 * rows a step writes are `lane * S + position[lane]`, and two lanes are S rows
 * apart even when they sit at the same position. Generalising the *arithmetic*
 * was the wrong move here as it was in WebGPU -- the caller states the geometry
 * it has, and `decode_row_set.c` resolves the addresses for both runtimes.
 *
 * Returns the lane count on success and 0 when this tensor is not the declared
 * batch geometry, which is the caller's signal to leave the operand alone
 * rather than to address it under a second meaning.
 */
static int decode_lane_rows(const T* tensor, long width, int* out_rows) {
    long lane_stride;
    if (g_decode_lanes <= 1 || !tensor || !out_rows || width <= 0) return 0;
    /* The batch axis is a declaration, so it is checked against the shape
     * rather than read from it. A tensor whose leading extent is not the
     * declared lane count is some other operand of the same graph. */
    if (tensor->ndim != 3 || tensor->shape[0] != g_decode_lanes ||
        tensor->shape[1] <= 0 || (long)tensor->shape[2] != width) return 0;
    lane_stride = tensor->shape[1];
    if (vx_decode_row_set_write_rows(&g_decode_rows, (int)lane_stride, 0,
                                     out_rows) != VX_DECODE_ROW_SET_OK) return 0;
    return g_decode_lanes;
}

/*
 * A declared batch's rows as contiguous runs, for a row-local operator.
 *
 * QAdd, QSiLU, QGELU and the norms all write row r from row r and nothing else,
 * so unlike QLinear they need no staging at all: the kernel runs once per run
 * of adjacent lane rows, at that run's own offset. Copying them into scratch
 * first would move the same bytes twice for no arithmetic saving, because an
 * elementwise kernel has nothing to amortise across rows -- which is the whole
 * reason QLinear stages and these do not.
 *
 * Parked lanes are excluded rather than computed and discarded. A released
 * lane's row holds whatever its previous occupant left, and a row-local
 * operator would faithfully recompute a function of another request's bytes
 * and store it. Skipping costs one extra run at worst.
 *
 * Returns the run count, or a negative value when this tensor is not the
 * declared batch geometry.
 */
static int decode_lane_runs(const T* tensor, long width, VxDecodeRowRun* runs) {
    int* rows = g_decode_rows.scratch_indices[0];
    if (!runs || decode_lane_rows(tensor, width, rows) != g_decode_lanes) return -1;
    for (int lane = 0; lane < g_decode_lanes; lane++)
        if (g_decode_rows.parked[lane]) rows[lane] = -1;
    return vx_decode_row_copy_runs(rows, g_decode_lanes, runs,
                                   g_decode_rows.lanes);
}

/* Scratch for one step's staged rows. Grown geometrically and never returned:
 * the size is a function of the model's widths, so it settles after the first
 * step of the first decode. */
static void* decode_stage_bytes(size_t bytes) {
    VxEngineState* state = vx_engine_state_current();
    if (!state || !bytes) return NULL;
    if (state->decode_stage_bytes < bytes) {
        size_t grown = state->decode_stage_bytes ? state->decode_stage_bytes : 1;
        unsigned char* replacement;
        while (grown < bytes) {
            if (grown > SIZE_MAX / 2) return NULL;
            grown *= 2;
        }
        replacement = (unsigned char*)realloc(state->decode_stage, grown);
        if (!replacement) return NULL;
        state->decode_stage = replacement;
        state->decode_stage_bytes = grown;
    }
    return state->decode_stage;
}

/* Incremental attention: one query row attends over cached K/V [0..len).
 *
 * Delegates to the paged kernel with a null page table, which is the identity
 * mapping — the same expression the paged form evaluates for a contiguous
 * cache. Keeping one implementation is what makes "paged KV is numerically
 * identical to contiguous KV" structural instead of a property some later
 * change can quietly break on one of two paths. */
static int sdpa_cached_row(const float* q, const float* kc, const float* vc, int len,
                       float* out, int d_model, int heads, int head_dim, float scale) {
    return vx_sdpa_paged_row_f32(q, kc, vc, NULL, 1, len, NULL, out,
                                 d_model, heads, head_dim, scale);
}


char* read_file(const char* path, long* out_size) {
    if (out_size) *out_size = 0;
    if (!path || !path[0]) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) { vx_engine_log("[engine] cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1);
    if (fread(buf, 1, sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    buf[sz] = 0; fclose(f); if (out_size) *out_size = sz; return buf;
}

// ---- safetensors: load weights; F16 tensors remain half-precision in the blob ----
static int volvoxai_engine_load_weight_files_impl(
        const char* const* paths,
        const SafetensorsFile* immutable_files,
        int count) {
    double t0 = g_debug ? volvoxai_engine_now_ms() : 0.0;
    const int borrowed = immutable_files != NULL;
    if (count < 0 || count > MAX_WEIGHT_FILES ||
        (count > 0 && !paths)) return -1;
    for (int i = 0; i < g_weight_file_count; i++) {
        safetensors_free(&g_weight_files[i]);
        g_weight_paths[i][0] = 0;
    }
    g_weight_file_count = 0;
    g_weight_files_borrowed = borrowed;
    g_blob = NULL;

    if (count == 0) {
        if (g_debug) {
            vx_engine_log("[debug] load_weights files=0 tensors=0 bytes=0 %.3f ms\n",
                    volvoxai_engine_now_ms() - t0);
        }
        return 0;
    }

    int weight_count = 0;
    int override_count = 0;
    int f16_loaded = 0;
    uint64_t total_size = 0;
    for (int f = 0; f < count; f++) {
        if (!paths[f] || !paths[f][0]) return -1;
        SafetensorsFile weights = {0};
        if (borrowed) {
            if (safetensors_borrow_immutable(&immutable_files[f], &weights) != 0)
                return -1;
        } else {
            SafetensorsLoadOptions options = { SAFETENSORS_OPEN_READ_WRITE };
            if (safetensors_load_with_options(paths[f], &options, &weights) != 0)
                return -1;
        }
        int new_tensor_count = 0;
        VxEngineResult reserve_status;
        for (int i = 0; i < weights.tensor_count; i++) {
            if (!t_find(weights.tensors[i].name)) {
                if (new_tensor_count == INT_MAX) {
                    safetensors_free(&weights);
                    return -1;
                }
                new_tensor_count++;
            }
        }
        if (weights.tensor_count < 0 || g_nt > INT_MAX - new_tensor_count) {
            safetensors_free(&weights);
            return -1;
        }
        reserve_status = vx_engine_state_reserve_graph_metadata(
            vx_engine_state_current(), (size_t)(g_nt + new_tensor_count),
            g_node_capacity);
        if (reserve_status != VX_ENGINE_RESULT_OK) {
            safetensors_free(&weights);
            return reserve_status;
        }
        volvoxai_engine_tensor_name_index_rebuild();
        for (int i = 0; i < weights.tensor_count; i++) {
            const SafetensorsTensor* tensor = &weights.tensors[i];
            const VxCompiledCpuWeight* compiled_cpu_weight =
                vx_compiled_cpu_weight_store_find(
                    vx_engine_state_current()->compiled_cpu_weights,
                    tensor->name);
            int dtype = compiled_cpu_weight
                ? T_F32
                : tensor_dtype_from_safetensor(tensor->dtype);
            void* tensor_data = compiled_cpu_weight
                ? (void*)compiled_cpu_weight->data
                : tensor->data;
            if (dtype < 0) {
                vx_engine_log("[engine] unsupported safetensors dtype %s for tensor %s\n",
                        safetensors_dtype_name(tensor->dtype), tensor->name);
                safetensors_free(&weights);
                return -1;
            }
            if (dtype == T_F16) f16_loaded++;
            T* existing = t_find(tensor->name);
            if (existing) {
                if (existing->owns && existing->data) free(existing->data);
                existing->ndim = tensor->ndim;
                for (int d = 0; d < tensor->ndim && d < 8; d++) existing->shape[d] = tensor->shape[d];
                existing->numel = numel_of(existing->shape, existing->ndim);
                existing->dtype = dtype;
                existing->elem_size = dtype_size(dtype);
                existing->owns = 0;
                existing->data = (float*)tensor_data;
                override_count++;
            } else {
                if (!t_add_ex(tensor->name, tensor->shape, tensor->ndim,
                              tensor_data, dtype)) {
                    safetensors_free(&weights);
                    return -1;
                }
            }
            weight_count++;
        }
        if (weights.size < 0 ||
            (uint64_t)weights.size > UINT64_MAX - total_size) {
            safetensors_free(&weights);
            return -1;
        }
        total_size += (uint64_t)weights.size;
        g_weight_files[g_weight_file_count] = weights;
        strncpy(g_weight_paths[g_weight_file_count], paths[f], sizeof(g_weight_paths[g_weight_file_count]) - 1);
        g_weight_paths[g_weight_file_count][sizeof(g_weight_paths[g_weight_file_count]) - 1] = 0;
        if (!g_blob) g_blob = weights.blob;
        g_weight_file_count++;
    }
    if (g_debug) {
        vx_engine_log("[debug] load_weights files=%d tensors=%d overrides=%d f16_loaded=%d bytes=%" PRIu64 " %.3f ms\n",
                count, weight_count, override_count, f16_loaded, total_size, volvoxai_engine_now_ms() - t0);
    }
    return 0;
}

int volvoxai_engine_load_weight_files(const char* const* paths, int count) {
    return volvoxai_engine_load_weight_files_impl(paths, NULL, count);
}

int volvoxai_engine_load_borrowed_weight_files(
        const SafetensorsFile* files,
        const char* const* paths,
        int count) {
    if (!files) return -1;
    return volvoxai_engine_load_weight_files_impl(paths, files, count);
}

static int node_json_i32_value(const cJSON* value, int* output) {
    double number;
    if (!value || !output || !cJSON_IsNumber(value)) return -1;
    number = value->valuedouble;
    if (!isfinite(number) || number < (double)INT32_MIN ||
        number > (double)INT32_MAX || trunc(number) != number) return -1;
    *output = (int)(int32_t)number;
    return 0;
}

static int node_json_u32_value(const cJSON* value, uint32_t minimum,
                               uint32_t maximum, uint32_t* output) {
    double number;
    if (!value || !output || !cJSON_IsNumber(value)) return -1;
    number = value->valuedouble;
    if (!isfinite(number) || trunc(number) != number ||
        number < (double)minimum || number > (double)maximum) return -1;
    *output = (uint32_t)number;
    return 0;
}

static VxNodeParamArraySlot node_param_array_slot(VxNodeParamKey key) {
    switch (key) {
        case VX_NODE_PARAM_STRIDE: return VX_NODE_ARRAY_STRIDE;
        case VX_NODE_PARAM_DILATION: return VX_NODE_ARRAY_DILATION;
        case VX_NODE_PARAM_PADDING: return VX_NODE_ARRAY_PADDING;
        case VX_NODE_PARAM_KERNEL: return VX_NODE_ARRAY_KERNEL;
        case VX_NODE_PARAM_KERNEL_SIZE: return VX_NODE_ARRAY_KERNEL_SIZE;
        case VX_NODE_PARAM_PADS: return VX_NODE_ARRAY_PADS;
        case VX_NODE_PARAM_PERM: return VX_NODE_ARRAY_PERM;
        case VX_NODE_PARAM_STARTS: return VX_NODE_ARRAY_STARTS;
        case VX_NODE_PARAM_ENDS: return VX_NODE_ARRAY_ENDS;
        case VX_NODE_PARAM_AXES: return VX_NODE_ARRAY_AXES;
        case VX_NODE_PARAM_STEPS: return VX_NODE_ARRAY_STEPS;
        case VX_NODE_PARAM_SHAPE: return VX_NODE_ARRAY_SHAPE;
        case VX_NODE_PARAM_SPLIT: return VX_NODE_ARRAY_SPLIT;
        case VX_NODE_PARAM_SIZE: return VX_NODE_ARRAY_SIZE;
        default: return VX_NODE_ARRAY_COUNT;
    }
}

static const VxGeneratedNodeParamDefinition* node_param_spec(
        const char* name) {
#if VOLVOXAI_ENABLE_TRAINING
    return vx_generated_full_node_param_definition_find(name);
#else
    return vx_generated_node_param_definition_find(name);
#endif
}

static int node_param_allowed_for_op(VxOperatorKind op, VxNodeParamKey key) {
#if VOLVOXAI_ENABLE_TRAINING
    return vx_generated_full_node_param_allowed(op, key);
#else
    return vx_generated_node_param_allowed(op, key);
#endif
}

static int node_param_allows_null(const Node* node, VxNodeParamKey key) {
    if (!node) return 0;
#if VOLVOXAI_ENABLE_TRAINING
    return vx_generated_full_node_param_allows_null(node->operator_kind, key);
#else
    return vx_generated_node_param_allows_null(node->operator_kind, key);
#endif
}

static VxNodeParamSymbol node_param_symbol_value(const char* value) {
    return vx_generated_node_param_symbol_from_json(value);
}

static int node_cache_i32(const cJSON* item, VxCachedNodeParam* output) {
    int value;
    if (node_json_i32_value(item, &value) != 0) return -1;
    output->kind = VX_NODE_PARAM_I32;
    output->value.i32 = value;
    return 0;
}

static int node_cache_i32_or_singleton_array(
        const cJSON* item, VxCachedNodeParam* output) {
    if (cJSON_IsArray(item)) {
        if (cJSON_GetArraySize(item) != 1) return -1;
        item = cJSON_GetArrayItem(item, 0);
    }
    return node_cache_i32(item, output);
}

static int node_cache_bool_or_i32_01(const cJSON* item,
                                     VxCachedNodeParam* output) {
    if (cJSON_IsBool(item)) {
        output->kind = VX_NODE_PARAM_BOOL;
        output->value.i32 = cJSON_IsTrue(item) ? 1 : 0;
        return 0;
    }
    if (node_cache_i32(item, output) != 0 ||
        (output->value.i32 != 0 && output->value.i32 != 1)) return -1;
    return 0;
}

static int node_cache_false_or_i32_0(const cJSON* item,
                                     VxCachedNodeParam* output) {
    if (node_cache_bool_or_i32_01(item, output) != 0 || output->value.i32 != 0)
        return -1;
    return 0;
}

static int node_cache_f32(const cJSON* item, VxCachedNodeParam* output) {
    float value;
    if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble)) return -1;
    value = (float)item->valuedouble;
    if (!isfinite(value)) return -1;
    output->kind = VX_NODE_PARAM_F32;
    output->value.f32 = value;
    return 0;
}

static int node_cache_u32(const cJSON* item, VxCachedNodeParam* output) {
    uint32_t value;
    if (node_json_u32_value(item, 0u, UINT32_MAX, &value) != 0) return -1;
    output->kind = VX_NODE_PARAM_U32;
    output->value.u32 = value;
    return 0;
}

static int node_cache_array_storage(VxNodeParams* parsed,
        VxNodeParamArraySlot slot, int count) {
    if (count < 0 || (size_t)count > SIZE_MAX / sizeof(int32_t)) return -1;
    // A present empty array must remain distinguishable from an absent field.
    int32_t* values = calloc(count ? (size_t)count : 1u, sizeof(*values));
    if (!values) return -1;
    free(parsed->arrays[slot]);
    parsed->arrays[slot] = values;
    parsed->array_bytes[slot] = (count ? (size_t)count : 1u) * sizeof(*values);
    return 0;
}

static int node_cache_array(const cJSON* item, VxNodeParams* parsed,
                            const VxGeneratedNodeParamDefinition* spec,
                            int pair) {
    VxCachedNodeParam* output = &parsed->cache[spec->key];
    VxNodeParamArraySlot slot = node_param_array_slot(spec->key);
    if (slot == VX_NODE_ARRAY_COUNT) return -1;
    int scalar_pair = pair && !cJSON_IsArray(item);
    if (!scalar_pair && !cJSON_IsArray(item)) return -1;
    int count = scalar_pair ? 1 : cJSON_GetArraySize(item);
    if (count < (pair ? 1 : 0) || (pair && count > 2) ||
        node_cache_array_storage(parsed, slot, pair ? 2 : count)) return -1;
    int index = 0;
    for (const cJSON* value = scalar_pair ? item : item->child; value && index < count;
         value = value->next, ++index) {
        if (node_json_i32_value(value, &parsed->arrays[slot][index])) return -1;
    }
    if (pair && count == 1) {
        parsed->arrays[slot][1] = parsed->arrays[slot][0];
        count = 2;
    }
    output->kind = VX_NODE_PARAM_I32_ARRAY;
    output->count = count;
    return 0;
}

static int node_cache_shape_array(const cJSON* item, VxNodeParams* parsed,
                                  const VxGeneratedNodeParamDefinition* spec) {
    VxCachedNodeParam* output = &parsed->cache[spec->key];
    VxNodeParamArraySlot slot = node_param_array_slot(spec->key);
    if (slot == VX_NODE_ARRAY_COUNT || !cJSON_IsArray(item)) return -1;
    int count = cJSON_GetArraySize(item), has_symbol = 0, index = 0;
    if (node_cache_array_storage(parsed, slot, count)) return -1;
    for (const cJSON* dimension = item->child; dimension; dimension = dimension->next, ++index) {
        if (cJSON_IsString(dimension) && dimension->valuestring && dimension->valuestring[0])
            has_symbol = 1;
        else if (node_json_i32_value(dimension, &parsed->arrays[slot][index])) return -1;
    }
    output->kind = has_symbol ? VX_NODE_PARAM_SHAPE_ARRAY : VX_NODE_PARAM_I32_ARRAY;
    output->count = count;
    return 0;
}

static int node_param_aliases_valid_for_table(
        const Node* node, const VxGeneratedNodeParamAliasMember* members,
        size_t member_count) {
    if (!node || (member_count && !members)) return -1;
    for (size_t index = 0; index < member_count; ++index) {
        const VxGeneratedNodeParamAliasMember* member = &members[index];
        if (node->operator_kind != member->operator_kind ||
            !vx_node_param_has(node, member->key))
            continue;
        for (size_t other_index = index + 1;
             other_index < member_count; ++other_index) {
            const VxGeneratedNodeParamAliasMember* other =
                &members[other_index];
            if (member->operator_kind == other->operator_kind &&
                !strcmp(member->group_id, other->group_id) &&
                vx_node_param_has(node, other->key))
                return -1;
        }
    }
    return 0;
}

static int node_param_aliases_valid(const Node* node) {
    if (node_param_aliases_valid_for_table(
            node, vx_generated_node_param_alias_members,
            vx_generated_node_param_alias_member_count) != 0)
        return -1;
#if VOLVOXAI_ENABLE_TRAINING
    if (node_param_aliases_valid_for_table(
            node, vx_generated_full_node_param_alias_members,
            vx_generated_full_node_param_alias_member_count) != 0)
        return -1;
#endif
    return 0;
}

static int node_compile_param_cache(Node* node) {
    cJSON* item;
    VxNodeParams* parsed;
    if (!node) return -1;
    parsed = &node->parsed_params;
    if (!node->params || cJSON_IsNull(node->params)) return 0;
    if (!cJSON_IsObject(node->params)) return -1;
    for (item = node->params->child; item; item = item->next) {
        const VxGeneratedNodeParamDefinition* spec =
            node_param_spec(item->string);
        VxCachedNodeParam* output;
        VxNodeParamSymbol symbol;
        if (parsed->parameter_count == UINT16_MAX) return -1;
        parsed->parameter_count++;
        if (!spec) return -1;
        if (!node_param_allowed_for_op(node->operator_kind, spec->key)) return -1;
        if (parsed->cached_parameter_count == UINT16_MAX) return -1;
        parsed->cached_parameter_count++;
        output = &parsed->cache[spec->key];
        if (output->kind != VX_NODE_PARAM_ABSENT) return -1;
        if (cJSON_IsNull(item)) {
            if (!node_param_allows_null(node, spec->key)) return -1;
            output->kind = VX_NODE_PARAM_NULL;
            continue;
        }
        switch (spec->json_form) {
            case VX_NODE_PARAM_JSON_I32:
                if (node_cache_i32(item, output) != 0) return -1;
                break;
            case VX_NODE_PARAM_JSON_I32_ZERO:
                if (node_cache_i32(item, output) != 0 || output->value.i32 != 0)
                    return -1;
                break;
            case VX_NODE_PARAM_JSON_I32_OR_SINGLETON_ARRAY:
                if (node_cache_i32_or_singleton_array(item, output) != 0)
                    return -1;
                break;
            case VX_NODE_PARAM_JSON_BOOL_OR_I32_01:
                if (node_cache_bool_or_i32_01(item, output) != 0) return -1;
                break;
            case VX_NODE_PARAM_JSON_FALSE_OR_I32_0:
                if (node_cache_false_or_i32_0(item, output) != 0) return -1;
                break;
            case VX_NODE_PARAM_JSON_U32:
                if (node_cache_u32(item, output) != 0) return -1;
                break;
            case VX_NODE_PARAM_JSON_F32:
                if (node_cache_f32(item, output) != 0) return -1;
                break;
            case VX_NODE_PARAM_JSON_BOOL:
                if (!cJSON_IsBool(item)) return -1;
                output->kind = VX_NODE_PARAM_BOOL;
                output->value.i32 = cJSON_IsTrue(item) ? 1 : 0;
                break;
            case VX_NODE_PARAM_JSON_SYMBOL:
                if (!cJSON_IsString(item) || !item->valuestring ||
                    (symbol = node_param_symbol_value(item->valuestring)) ==
                        VX_NODE_SYMBOL_INVALID) return -1;
                output->kind = VX_NODE_PARAM_SYMBOL;
                output->symbol = (uint16_t)symbol;
                break;
            case VX_NODE_PARAM_JSON_PAIR:
                if (node_cache_array(item, parsed, spec, 1) != 0) return -1;
                break;
            case VX_NODE_PARAM_JSON_ARRAY:
                if (node_cache_array(item, parsed, spec, 0) != 0) return -1;
                break;
            case VX_NODE_PARAM_JSON_SHAPE_ARRAY:
                if (node_cache_shape_array(item, parsed, spec) != 0) return -1;
                break;
            default:
                return -1;
        }
    }
    return node_param_aliases_valid(node);
}

static int node_cached_pair(const Node* node, VxNodeParamKey key,
                            VxNodeParamArraySlot slot, int fallback,
                            int output[2]) {
    int count = 0;
    const int32_t* values = vx_node_param_array(node, key, slot, &count);
    if (!output) return -1;
    if (!values) {
        output[0] = fallback;
        output[1] = fallback;
        return 0;
    }
    if (count != 2) return -1;
    output[0] = values[0];
    output[1] = values[1];
    return 0;
}

static int node_validate_cached_spatial_padding(const Node* node) {
    int padding[2];
    int pads_count = 0;
    const int32_t* pads;
    if (!node || !vx_node_param_has(node, VX_NODE_PARAM_PADS)) return 0;
    pads = vx_node_param_array(
        node, VX_NODE_PARAM_PADS, VX_NODE_ARRAY_PADS, &pads_count);
    if (!pads || pads_count != 4) return -1;
    if (!vx_node_param_has(node, VX_NODE_PARAM_PADDING)) return 0;
    if (node_cached_pair(node, VX_NODE_PARAM_PADDING,
                         VX_NODE_ARRAY_PADDING, 0, padding) != 0)
        return -1;
    return pads[0] == padding[0] && pads[1] == padding[1] &&
           pads[2] == padding[0] && pads[3] == padding[1] ? 0 : -1;
}

static int node_parse_i32_array(const cJSON* array, int count, int* output) {
    if (!array || !output || !cJSON_IsArray(array) || count < 0 ||
        cJSON_GetArraySize(array) != count) return -1;
    for (int index = 0; index < count; index++) {
        if (node_json_i32_value(cJSON_GetArrayItem(array, index),
                                &output[index]) != 0) return -1;
    }
    return 0;
}

static int node_parse_qgroupnorm_params(const cJSON* params,
                                        VxNodeParams* parsed) {
    const cJSON* item;
    int saw_groups = 0;
    int saw_epsilon = 0;
    int saw_layout = 0;
    uint32_t groups = 0;
    float epsilon = 1.0e-5f;
    if (!params || !parsed || !cJSON_IsObject(params)) return -1;
    for (item = params->child; item; item = item->next) {
        if (!item->string) return -1;
        if (!strcmp(item->string, "num_groups")) {
            if (saw_groups ||
                node_json_u32_value(item, 1u, UINT32_MAX, &groups) != 0)
                return -1;
            saw_groups = 1;
        } else if (!strcmp(item->string, "eps")) {
            float value;
            if (saw_epsilon) return -1;
            saw_epsilon = 1;
            if (cJSON_IsNull(item)) continue;
            if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble) ||
                item->valuedouble <= 0.0) return -1;
            value = (float)item->valuedouble;
            if (!isfinite(value) || value <= 0.0f) return -1;
            epsilon = value;
        } else if (!strcmp(item->string, "data_layout")) {
            if (saw_layout) return -1;
            saw_layout = 1;
            if (cJSON_IsNull(item)) continue;
            if (!cJSON_IsString(item) || !item->valuestring ||
                (vx_generated_node_param_symbol_from_json(item->valuestring) != VX_NODE_SYMBOL_NHWC)) return -1;
        } else {
            return -1;
        }
    }
    if (!saw_groups) return -1;
    parsed->strict_num_groups = groups;
    parsed->eps = epsilon;
    parsed->data_layout = VX_DATA_LAYOUT_NHWC;
    return 0;
}

static int node_parse_qsdpa_params(const cJSON* params, VxNodeParams* parsed) {
    const cJSON* item;
    int saw_heads = 0;
    int saw_causal = 0;
    int saw_scale = 0;
    uint32_t heads = 0;
    if (!params || !parsed || !cJSON_IsObject(params)) return -1;
    parsed->has_scale = 0;
    for (item = params->child; item; item = item->next) {
        if (!item->string) return -1;
        if (!strcmp(item->string, "heads")) {
            if (saw_heads ||
                node_json_u32_value(item, 1u, UINT32_MAX, &heads) != 0)
                return -1;
            saw_heads = 1;
        } else if (!strcmp(item->string, "causal")) {
            if (saw_causal || !cJSON_IsBool(item)) return -1;
            saw_causal = 1;
            parsed->causal = cJSON_IsTrue(item) ? 1 : 0;
        } else if (!strcmp(item->string, "scale")) {
            float value;
            if (saw_scale) return -1;
            saw_scale = 1;
            if (cJSON_IsNull(item)) continue;
            if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble) ||
                item->valuedouble <= 0.0) return -1;
            value = (float)item->valuedouble;
            if (!isfinite(value) || value <= 0.0f) return -1;
            parsed->scale = value;
            parsed->has_scale = 1;
        } else {
            return -1;
        }
    }
    if (!saw_heads || !saw_causal) return -1;
    parsed->strict_heads = heads;
    return 0;
}

static int node_parse_qargmax_params(const cJSON* params, VxNodeParams* parsed) {
    const cJSON* item;
    int saw_axis = 0;
    if (!params || !parsed || !cJSON_IsObject(params)) return -1;
    for (item = params->child; item; item = item->next) {
        if (!item->string || strcmp(item->string, "axis") || saw_axis ||
            node_json_i32_value(item, &parsed->axis) != 0) return -1;
        saw_axis = 1;
    }
    return saw_axis ? 0 : -1;
}

static int node_parse_params(Node* n) {
    VxNodeParams* parsed;
    cJSON* p;
    cJSON* item;
    if (!n) return -1;
    /* Concat2 has the same execution semantics as Concat. The immutable
     * source and semantic plan retain its declared two-port representation. */
    if (n->operator_kind == VX_OP_CONCAT2) n->operator_kind = VX_OP_CONCAT;
    if ((n->operator_kind == VX_OP_CONCAT) && vx_concat_sort(n)) return -1;
    vx_node_clear_param_cache(n);
    parsed = &n->parsed_params;
    p = n->params;

    if (node_compile_param_cache(n) != 0) goto hard_invalid;

    /* Defaults are cached even when params is omitted. */
    parsed->axis = vx_node_param_i32(n, VX_NODE_PARAM_AXIS, 0);
    parsed->keepdims = vx_node_param_i32(n, VX_NODE_PARAM_KEEPDIMS, 1);
    parsed->select_last_index = vx_node_param_i32(
        n, VX_NODE_PARAM_SELECT_LAST_INDEX, 0);
    parsed->min_val = -INFINITY;
    parsed->max_val = INFINITY;
    parsed->min_val_i32 = INT32_MIN;
    parsed->max_val_i32 = INT32_MAX;
    parsed->relu = vx_node_param_i32(n, VX_NODE_PARAM_RELU, 0);
    parsed->groups = vx_node_param_i32(n, VX_NODE_PARAM_GROUPS, 1);
    parsed->num_groups = vx_node_param_i32(
        n, VX_NODE_PARAM_NUM_GROUPS, parsed->groups);
    parsed->d_model = vx_node_param_i32(n, VX_NODE_PARAM_D_MODEL, 0);
    parsed->heads = vx_node_param_i32(n, VX_NODE_PARAM_HEADS, 8);
    parsed->causal = vx_node_param_i32(
        n, VX_NODE_PARAM_CAUSAL, (n->operator_kind == VX_OP_SDPA) ? 1 : 0);
    parsed->top_k = vx_node_param_i32(n, VX_NODE_PARAM_TOP_K, 2);
    parsed->num_experts = vx_node_param_i32(n, VX_NODE_PARAM_NUM_EXPERTS, 0);
    parsed->normalize = vx_node_param_i32(n, VX_NODE_PARAM_NORMALIZE, 1);
    parsed->temperature = vx_node_param_f32(
        n, VX_NODE_PARAM_TEMPERATURE, 1.0f);
    parsed->attention_dropout_probability = vx_node_param_f32(
        n, VX_NODE_PARAM_DROPOUT,
        vx_node_param_f32(n, VX_NODE_PARAM_ATTENTION_DROPOUT, 0.0f));
    parsed->attention_dropout_seed = vx_node_param_u32(
        n, VX_NODE_PARAM_DROPOUT_SEED,
        vx_node_param_u32(n, VX_NODE_PARAM_TRAINING_SEED,
            vx_node_param_u32(n, VX_NODE_PARAM_SEED, 0u)));
    parsed->transB = vx_node_param_i32(n, VX_NODE_PARAM_TRANS_B, 0);
    parsed->alpha = vx_node_param_f32(n, VX_NODE_PARAM_ALPHA, 1.0f);
    parsed->beta = vx_node_param_f32(n, VX_NODE_PARAM_BETA, 1.0f);
    parsed->epsilon = vx_node_param_f32(n, VX_NODE_PARAM_EPSILON, 1.0e-5f);
    parsed->eps = vx_node_param_f32(n, VX_NODE_PARAM_EPS, 1.0e-5f);
    parsed->sigmoid = vx_node_param_i32(n, VX_NODE_PARAM_SIGMOID, 0);
    parsed->weight_only_valid = 1;
    parsed->data_layout = VX_DATA_LAYOUT_NHWC;

    item = p ? cJSON_GetObjectItemCaseSensitive(p, "scale") : NULL;
    if (vx_node_param_has(n, VX_NODE_PARAM_SCALE)) {
        parsed->has_scale = 1;
        parsed->scale = vx_node_param_f32(n, VX_NODE_PARAM_SCALE, 0.0f);
    }

    item = p ? cJSON_GetObjectItemCaseSensitive(p, "min") : NULL;
    if (vx_node_param_has(n, VX_NODE_PARAM_MIN)) {
        parsed->has_min_val = 1;
        parsed->min_val = vx_node_param_f32(n, VX_NODE_PARAM_MIN, 0.0f);
    }
    item = p ? cJSON_GetObjectItemCaseSensitive(p, "max") : NULL;
    if (vx_node_param_has(n, VX_NODE_PARAM_MAX)) {
        parsed->has_max_val = 1;
        parsed->max_val = vx_node_param_f32(n, VX_NODE_PARAM_MAX, 0.0f);
    }

    if (n->operator_kind == VX_OP_TRANSPOSE) {
        parsed->transpose_params_valid = 1;
        for (cJSON* parameter = p ? p->child : NULL;
             parameter; parameter = parameter->next) {
            if (!parameter->string || strcmp(parameter->string, "perm"))
                parsed->transpose_params_valid = 0;
        }
        item = p ? cJSON_GetObjectItem(p, "perm") : NULL;
        if (item) {
            int rank = cJSON_IsArray(item) ? cJSON_GetArraySize(item) : -1;
            if (rank <= 0 || rank > 8 ||
                node_parse_i32_array(item, rank, parsed->perm) != 0) goto invalid;
            for (int index = 0; index < rank; index++) {
                if (parsed->perm[index] < 0 || parsed->perm[index] >= rank)
                    goto invalid;
                for (int prior = 0; prior < index; prior++)
                    if (parsed->perm[prior] == parsed->perm[index]) goto invalid;
            }
            parsed->has_perm = 1;
            parsed->perm_rank = rank;
        }
    }

    if (n->operator_kind == VX_OP_SLICE) {
        const VxCachedNodeParam* axes_param =
            vx_node_param(n, VX_NODE_PARAM_AXES);
        const VxCachedNodeParam* steps_param =
            vx_node_param(n, VX_NODE_PARAM_STEPS);
        int starts_count = 0;
        int ends_count = 0;
        int axes_count = 0;
        int steps_count = 0;
        const int32_t* starts = vx_node_param_array(
            n, VX_NODE_PARAM_STARTS, VX_NODE_ARRAY_STARTS, &starts_count);
        const int32_t* ends = vx_node_param_array(
            n, VX_NODE_PARAM_ENDS, VX_NODE_ARRAY_ENDS, &ends_count);
        const int32_t* axes = vx_node_param_array(
            n, VX_NODE_PARAM_AXES, VX_NODE_ARRAY_AXES, &axes_count);
        const int32_t* steps = vx_node_param_array(
            n, VX_NODE_PARAM_STEPS, VX_NODE_ARRAY_STEPS, &steps_count);
        if (!starts || !ends || starts_count <= 0 || starts_count > 8 ||
            ends_count != starts_count) goto invalid;
        if (axes_param->kind != VX_NODE_PARAM_ABSENT &&
            (!axes || axes_count != starts_count)) goto invalid;
        if (steps_param->kind != VX_NODE_PARAM_ABSENT &&
            (!steps || steps_count != starts_count)) goto invalid;
        for (int index = 0; index < starts_count; index++) {
            parsed->starts[index] = starts[index];
            parsed->ends[index] = ends[index];
            parsed->axes[index] = axes_param->kind == VX_NODE_PARAM_ABSENT
                ? index : axes[index];
            parsed->steps[index] = steps_param->kind == VX_NODE_PARAM_ABSENT
                ? 1 : steps[index];
            if (parsed->steps[index] <= 0) goto invalid;
        }
        parsed->has_slice = 1;
        parsed->slice_rank = starts_count;
    }

    if (((n->operator_kind == VX_OP_CONV_2D) || (n->operator_kind == VX_OP_Q_CONV_2D) ||
         (n->operator_kind == VX_OP_MAX_POOL_2D) ||
         (n->operator_kind == VX_OP_AVERAGE_POOL_2D)) &&
        node_validate_cached_spatial_padding(n) != 0)
        goto hard_invalid;

    parsed->kernel_size[0] = 3;
    parsed->kernel_size[1] = 3;
    parsed->stride[0] = 1;
    parsed->stride[1] = 1;
    parsed->dilation[0] = 1;
    parsed->dilation[1] = 1;
    if ((n->operator_kind == VX_OP_CONV_2D) || (n->operator_kind == VX_OP_Q_CONV_2D)) {
        int padding[2] = {0, 0};
        int has_padding = vx_node_param_has(n, VX_NODE_PARAM_PADDING);
        int has_pads = vx_node_param_has(n, VX_NODE_PARAM_PADS);
        if (node_cached_pair(n, VX_NODE_PARAM_KERNEL_SIZE,
                             VX_NODE_ARRAY_KERNEL_SIZE, 3,
                             parsed->kernel_size) != 0 ||
            node_cached_pair(n, VX_NODE_PARAM_STRIDE,
                             VX_NODE_ARRAY_STRIDE, 1, parsed->stride) != 0 ||
            node_cached_pair(n, VX_NODE_PARAM_DILATION,
                             VX_NODE_ARRAY_DILATION, 1,
                             parsed->dilation) != 0)
            goto invalid;
        if (has_padding) {
            if (node_cached_pair(n, VX_NODE_PARAM_PADDING,
                                 VX_NODE_ARRAY_PADDING, 0, padding) != 0)
                goto invalid;
        } else {
            padding[0] = vx_node_param_i32(
                n, VX_NODE_PARAM_PADDING_Y,
                vx_node_param_i32(n, VX_NODE_PARAM_PAD_Y,
                    vx_node_param_i32(n, VX_NODE_PARAM_PAD, 0)));
            padding[1] = vx_node_param_i32(
                n, VX_NODE_PARAM_PADDING_X,
                vx_node_param_i32(n, VX_NODE_PARAM_PAD_X,
                    vx_node_param_i32(n, VX_NODE_PARAM_PAD, 0)));
        }
        parsed->pads[0] = padding[0];
        parsed->pads[1] = padding[1];
        parsed->pads[2] = padding[0];
        parsed->pads[3] = padding[1];
        if (has_pads) {
            int pad_count = 0;
            const int32_t* cached_pads = vx_node_param_array(
                n, VX_NODE_PARAM_PADS, VX_NODE_ARRAY_PADS, &pad_count);
            if (!cached_pads || pad_count != 4) goto invalid;
            if (has_padding &&
                (cached_pads[0] != padding[0] ||
                 cached_pads[1] != padding[1] ||
                 cached_pads[2] != padding[0] ||
                 cached_pads[3] != padding[1])) goto invalid;
            for (int index = 0; index < 4; index++)
                parsed->pads[index] = cached_pads[index];
            parsed->has_pads = 1;
        }
        switch (vx_node_param_symbol(
                    n, VX_NODE_PARAM_WEIGHT_LAYOUT,
                    (n->operator_kind == VX_OP_Q_CONV_2D) ? VX_NODE_SYMBOL_OHWI
                                               : VX_NODE_SYMBOL_HWIO)) {
            case VX_NODE_SYMBOL_HWIO: parsed->weight_layout = VX_CONV_WEIGHT_LAYOUT_HWIO; break;
            case VX_NODE_SYMBOL_HWCM: parsed->weight_layout = VX_CONV_WEIGHT_LAYOUT_HWCM; break;
            case VX_NODE_SYMBOL_OHWI: parsed->weight_layout = VX_CONV_WEIGHT_LAYOUT_OHWI; break;
            case VX_NODE_SYMBOL_1HWO: parsed->weight_layout = VX_CONV_WEIGHT_LAYOUT_1HWO; break;
            case VX_NODE_SYMBOL_1HWM: parsed->weight_layout = VX_CONV_WEIGHT_LAYOUT_1HWM; break;
            default: parsed->weight_layout = VX_CONV_WEIGHT_LAYOUT_INVALID; break;
        }
        parsed->data_layout = vx_node_param_symbol(
            n, VX_NODE_PARAM_DATA_LAYOUT, VX_NODE_SYMBOL_NHWC) ==
                VX_NODE_SYMBOL_NHWC ? VX_DATA_LAYOUT_NHWC
                                    : VX_DATA_LAYOUT_INVALID;
    }

    if (n->operator_kind == VX_OP_CONV_TRANSPOSE_2D) {
        VxNodeParamSymbol data_layout = vx_node_param_symbol(
            n, VX_NODE_PARAM_DATA_LAYOUT, VX_NODE_SYMBOL_NHWC);
        VxNodeParamSymbol weight_layout = vx_node_param_symbol(
            n, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_HWIO);
        if (data_layout != VX_NODE_SYMBOL_NHWC ||
            weight_layout != VX_NODE_SYMBOL_HWIO) goto hard_invalid;
        parsed->data_layout = VX_DATA_LAYOUT_NHWC;
        parsed->weight_layout = VX_CONV_WEIGHT_LAYOUT_HWIO;
    }

    if (n->operator_kind == VX_OP_Q_CONV_2D) {
        cJSON* weight_only = p ? cJSON_GetObjectItem(p, "weight_only") : NULL;
        if (weight_only && !cJSON_IsNull(weight_only)) {
            if (!cJSON_IsBool(weight_only)) parsed->weight_only_valid = 0;
            else parsed->weight_only = cJSON_IsTrue(weight_only) ? 1 : 0;
        }
        if (parsed->weight_only) {
            const VxCachedNodeParam* groups =
                vx_node_param(n, VX_NODE_PARAM_GROUPS);
            const VxCachedNodeParam* relu =
                vx_node_param(n, VX_NODE_PARAM_RELU);
            if (groups && groups->kind != VX_NODE_PARAM_ABSENT) {
                if (groups->kind != VX_NODE_PARAM_I32 ||
                    groups->value.i32 <= 0)
                    parsed->weight_only_valid = 0;
                else
                    parsed->groups = groups->value.i32;
            }
            if (relu && relu->kind != VX_NODE_PARAM_ABSENT) {
                if (relu->kind != VX_NODE_PARAM_I32 ||
                    relu->value.i32 < 0 || relu->value.i32 > 2)
                    parsed->weight_only_valid = 0;
                else
                    parsed->relu = relu->value.i32;
            }
        }
        if (parsed->weight_only &&
            (parsed->data_layout != VX_DATA_LAYOUT_NHWC ||
             parsed->weight_layout != VX_CONV_WEIGHT_LAYOUT_OHWI ||
             parsed->groups <= 0 || parsed->relu < 0 || parsed->relu > 2 ||
             parsed->stride[0] <= 0 || parsed->stride[1] <= 0 ||
             parsed->dilation[0] <= 0 || parsed->dilation[1] <= 0 ||
             parsed->pads[0] < 0 || parsed->pads[1] < 0 ||
             parsed->pads[2] < 0 || parsed->pads[3] < 0))
            parsed->weight_only_valid = 0;
    }

    if (n->operator_kind == VX_OP_Q_GROUP_NORM)
        parsed->qgroupnorm_params_valid =
            node_parse_qgroupnorm_params(p, parsed) == 0;
    if (n->operator_kind == VX_OP_Q_SDPA)
        parsed->qsdpa_params_valid = node_parse_qsdpa_params(p, parsed) == 0;
    if (n->operator_kind == VX_OP_Q_ARG_MAX)
        parsed->qargmax_params_valid = node_parse_qargmax_params(p, parsed) == 0;

    if (n->operator_kind == VX_OP_CLIP) {
        T* output = t_find(n->out);
        if (output && output->dtype == T_I32) {
            cJSON* minimum = p ? cJSON_GetObjectItemCaseSensitive(p, "min") : NULL;
            cJSON* maximum = p ? cJSON_GetObjectItemCaseSensitive(p, "max") : NULL;
            parsed->has_min_val = minimum ? 1 : 0;
            parsed->has_max_val = maximum ? 1 : 0;
            if ((minimum && node_json_i32_value(minimum, &parsed->min_val_i32) != 0) ||
                (maximum && node_json_i32_value(maximum, &parsed->max_val_i32) != 0) ||
                parsed->min_val_i32 > parsed->max_val_i32) goto hard_invalid;
        }
        if (parsed->min_val > parsed->max_val) goto hard_invalid;
    }

    if ((n->operator_kind == VX_OP_LINEAR) || (n->operator_kind == VX_OP_MATMUL) ||
        (n->operator_kind == VX_OP_GEMM)) {
        VxNodeParamSymbol layout = vx_node_param_symbol(
            n, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_INVALID);
        if ((vx_node_param_has(n, VX_NODE_PARAM_WEIGHT_LAYOUT) &&
             vx_node_param_has(n, VX_NODE_PARAM_TRANS_B)) ||
            (layout != VX_NODE_SYMBOL_INVALID &&
             layout != VX_NODE_SYMBOL_DIN_DOUT &&
             layout != VX_NODE_SYMBOL_DOUT_DIN) ||
            (vx_node_param_has(n, VX_NODE_PARAM_TRANS_B) &&
             parsed->transB != 0 && parsed->transB != 1))
            goto hard_invalid;
    }

    parsed->parsed = 1;
    return 0;
invalid:
    parsed->parsed = -1;
    return -1;
hard_invalid:
    parsed->parsed = -1;
    return -1;
}

int volvoxai_engine_compile_node_params(Node* node) {
    return node_parse_params(node);
}

static T* nin(Node* n, VxPortKind key) {
    if (!n || key == VX_PORT_UNSPECIFIED) return NULL;
    for (int i = 0; i < n->nin; i++) if (n->ins[i].port == key) return t_find(n->ins[i].name);
    return NULL;
}
static T* nout(Node* n, VxPortKind key) {
    if (!n || key == VX_PORT_UNSPECIFIED) return NULL;
    for (int i = 0; i < n->nout; i++) if (n->outs[i].port == key) return t_find(n->outs[i].name);
    return NULL;
}

static int native_qlinear_op(VxOperatorKind op) {
    return op && ((op == VX_OP_Q_LINEAR) || (op == VX_OP_Q_MATMUL) || (op == VX_OP_Q_GEMM));
}

static int native_qconv2d_op(VxOperatorKind op) {
    return op && (op == VX_OP_Q_CONV_2D);
}

static int native_qembedding_op(VxOperatorKind op) {
    return op && (op == VX_OP_Q_EMBEDDING);
}

/* QEmbedding may consume an internal ID tensor only when the graph itself
 * proves the complete range before execution. Public inputs are preflighted
 * from their supplied bytes; the sole internal proof is one canonical I32
 * Clip with inclusive bounds inside the immutable vocabulary. */
static int qembedding_ids_preflight_complete(Node* consumer, T* ids, int vocab) {
    int consumer_index;
    int producer_index = -1;
    Node* producer;
    T* source;
    if (!consumer || !ids || ids->dtype != T_I32 || vocab <= 0) return 0;
    if (ids->is_graph_input) return 1;
    consumer_index = (int)(consumer - g_n);
    if (consumer_index < 0 || consumer_index >= g_nn) return 0;
    for (int node_index = 0; node_index < g_nn; node_index++) {
        Node* candidate = &g_n[node_index];
        for (int output_index = 0; output_index < candidate->nout; output_index++) {
            if (strcmp(candidate->outs[output_index].name, ids->name)) continue;
            if (producer_index != -1) return 0;
            producer_index = node_index;
        }
    }
    if (producer_index < 0 || producer_index >= consumer_index) return 0;
    producer = &g_n[producer_index];
    if ((producer->operator_kind != VX_OP_CLIP) || producer->nin != 1 || producer->nout != 1 ||
        (producer->ins[0].port != VX_PORT_INPUT) || (producer->outs[0].port != VX_PORT_OUT) ||
        producer->parsed_params.parsed != 1 ||
        producer->parsed_params.parameter_count != 2 ||
        !producer->parsed_params.has_min_val ||
        !producer->parsed_params.has_max_val ||
        producer->parsed_params.min_val_i32 < 0 ||
        producer->parsed_params.max_val_i32 <
            producer->parsed_params.min_val_i32 ||
        producer->parsed_params.max_val_i32 >= vocab) return 0;
    source = nin(producer, VX_PORT_INPUT);
    if (!source || source->dtype != T_I32 || source->ndim != ids->ndim ||
        source->numel != ids->numel) return 0;
    for (int dimension = 0; dimension < ids->ndim; dimension++)
        if (source->shape[dimension] != ids->shape[dimension]) return 0;
    return 1;
}

static uint32_t physical_byte_dtype(int dtype) {
    if (dtype == T_I8 || dtype == T_U8) return (uint32_t)dtype;
    return (uint32_t)VX_DTYPE_UNSPECIFIED;
}

typedef struct {
    T* a;
    T* b;
    T* output;
    uint32_t a_dtype;
    uint32_t b_dtype;
    uint32_t output_dtype;
    uint32_t batch_rank;
    uint32_t output_batch_count;
    uint32_t m;
    uint32_t k;
    uint32_t n;
} QBatchMatMulPhysicalDescriptor;

static int qbatch_matmul_tensor_elements(const T* tensor,
                                         uint32_t* elements_out) {
    uint64_t elements = 1u;
    if (!tensor || !elements_out || !tensor->data || tensor->ndim <= 0 ||
        tensor->ndim > 8 || tensor->numel <= 0) return -1;
    for (int dimension = 0; dimension < tensor->ndim; dimension++) {
        uint32_t extent;
        if (tensor->shape[dimension] <= 0) return -1;
        extent = (uint32_t)tensor->shape[dimension];
        if (elements > UINT32_MAX / extent) return -1;
        elements *= extent;
    }
    if (tensor->numel != (long)elements) return -1;
    *elements_out = (uint32_t)elements;
    return 0;
}

static int qbatch_matmul_byte_tensor(const T* tensor, uint32_t* dtype_out,
                                     uint32_t* elements_out) {
    uint32_t dtype;
    if (!tensor || tensor->elem_size != 1 || !tensor->quantization.valid ||
        !isfinite(tensor->quantization.scale) ||
        tensor->quantization.scale <= 0.0f ||
        qbatch_matmul_tensor_elements(tensor, elements_out) != 0)
        return -1;
    dtype = physical_byte_dtype(tensor->dtype);
    if (!dtype ||
        (dtype == VX_DTYPE_I8 &&
         (tensor->quantization.zero_point < -128 ||
          tensor->quantization.zero_point > 127)) ||
        (dtype == VX_DTYPE_U8 &&
         (tensor->quantization.zero_point < 0 ||
          tensor->quantization.zero_point > 255)))
        return -1;
    *dtype_out = dtype;
    return 0;
}

static int qbatch_matmul_ranges_overlap(const T* left, const T* right) {
    uintptr_t left_start;
    uintptr_t right_start;
    size_t left_bytes;
    size_t right_bytes;
    if (!left || !right || !left->data || !right->data ||
        left->numel <= 0 || right->numel <= 0 ||
        (uint64_t)left->numel > SIZE_MAX / left->elem_size ||
        (uint64_t)right->numel > SIZE_MAX / right->elem_size)
        return 1;
    left_start = (uintptr_t)left->data;
    right_start = (uintptr_t)right->data;
    left_bytes = (size_t)left->numel * left->elem_size;
    right_bytes = (size_t)right->numel * right->elem_size;
    if (left_start > UINTPTR_MAX - left_bytes ||
        right_start > UINTPTR_MAX - right_bytes)
        return 1;
    return left_start < right_start + right_bytes &&
        right_start < left_start + left_bytes;
}

static uint64_t qbatch_matmul_centered_magnitude(uint32_t dtype,
                                                 int32_t zero_point) {
    int64_t minimum =
        (int64_t)(dtype == VX_DTYPE_I8 ? -128 : 0) - zero_point;
    int64_t maximum =
        (int64_t)(dtype == VX_DTYPE_I8 ? 127 : 255) - zero_point;
    uint64_t minimum_magnitude =
        (uint64_t)(minimum < 0 ? -minimum : minimum);
    uint64_t maximum_magnitude =
        (uint64_t)(maximum < 0 ? -maximum : maximum);
    return minimum_magnitude > maximum_magnitude
        ? minimum_magnitude : maximum_magnitude;
}

/* Canonical physical-byte ONNX MatMul. Vector promotion belongs to the
 * exporter; native execution accepts rank-2..8 matrices with right-aligned
 * batch broadcasting and one immutable per-tensor descriptor per tensor. */
static int qbatch_matmul_physical_validate(
        Node* node, T* output, QBatchMatMulPhysicalDescriptor* descriptor) {
    QBatchMatMulPhysicalDescriptor parsed = {0};
    uint32_t a_elements;
    uint32_t b_elements;
    uint32_t output_elements;
    uint64_t batch_count = 1u;
    uint64_t expected_output_elements;
    uint64_t a_magnitude;
    uint64_t b_magnitude;
    uint64_t accumulator_bound;
    float multiplier;
    int a_refs = 0;
    int b_refs = 0;
    if (!node || !output || !descriptor ||
        (node->operator_kind != VX_OP_Q_BATCH_MATMUL) ||
        node->nin != 2 || node->nout != 1 ||
        (node->outs[0].port != VX_PORT_OUT) ||
        node->parsed_params.parsed != 1 ||
        node->parsed_params.parameter_count != 0)
        return -1;
    for (int index = 0; index < node->nin; index++) {
        if (node->ins[index].port == VX_PORT_A) a_refs++;
        else if (node->ins[index].port == VX_PORT_B) b_refs++;
        else return -1;
    }
    if (a_refs != 1 || b_refs != 1 || nout(node, VX_PORT_OUT) != output ||
        t_find(node->out) != output)
        return -1;
    parsed.a = nin(node, VX_PORT_A);
    parsed.b = nin(node, VX_PORT_B);
    parsed.output = output;
    if (qbatch_matmul_byte_tensor(
            parsed.a, &parsed.a_dtype, &a_elements) != 0 ||
        qbatch_matmul_byte_tensor(
            parsed.b, &parsed.b_dtype, &b_elements) != 0 ||
        qbatch_matmul_byte_tensor(
            parsed.output, &parsed.output_dtype, &output_elements) != 0 ||
        parsed.a->ndim < 2 || parsed.b->ndim < 2 ||
        parsed.output->ndim !=
            (parsed.a->ndim > parsed.b->ndim
                 ? parsed.a->ndim : parsed.b->ndim))
        return -1;
    parsed.batch_rank = (uint32_t)parsed.output->ndim - 2u;
    parsed.m = (uint32_t)parsed.a->shape[parsed.a->ndim - 2];
    parsed.k = (uint32_t)parsed.a->shape[parsed.a->ndim - 1];
    parsed.n = (uint32_t)parsed.b->shape[parsed.b->ndim - 1];
    if (parsed.a->shape[parsed.a->ndim - 1] !=
            parsed.b->shape[parsed.b->ndim - 2] ||
        parsed.output->shape[parsed.output->ndim - 2] != (int)parsed.m ||
        parsed.output->shape[parsed.output->ndim - 1] != (int)parsed.n)
        return -1;
    for (uint32_t dimension = 0; dimension < parsed.batch_rank; dimension++) {
        int a_dimension =
            (int)dimension - (parsed.output->ndim - parsed.a->ndim);
        int b_dimension =
            (int)dimension - (parsed.output->ndim - parsed.b->ndim);
        int a_extent =
            a_dimension >= 0 ? parsed.a->shape[a_dimension] : 1;
        int b_extent =
            b_dimension >= 0 ? parsed.b->shape[b_dimension] : 1;
        int expected_extent =
            a_extent > b_extent ? a_extent : b_extent;
        if ((a_extent != 1 && b_extent != 1 && a_extent != b_extent) ||
            parsed.output->shape[dimension] != expected_extent ||
            batch_count > UINT32_MAX / (uint32_t)expected_extent)
            return -1;
        batch_count *= (uint32_t)expected_extent;
    }
    expected_output_elements =
        batch_count * (uint64_t)parsed.m * parsed.n;
    if (expected_output_elements != output_elements ||
        (uint64_t)parsed.m * parsed.k > a_elements ||
        (uint64_t)parsed.k * parsed.n > b_elements ||
        qbatch_matmul_ranges_overlap(parsed.output, parsed.a) ||
        qbatch_matmul_ranges_overlap(parsed.output, parsed.b))
        return -1;
    parsed.output_batch_count = (uint32_t)batch_count;
    a_magnitude = qbatch_matmul_centered_magnitude(
        parsed.a_dtype, parsed.a->quantization.zero_point);
    b_magnitude = qbatch_matmul_centered_magnitude(
        parsed.b_dtype, parsed.b->quantization.zero_point);
    accumulator_bound =
        a_magnitude * b_magnitude * (uint64_t)parsed.k;
    if (accumulator_bound > INT32_MAX) return -1;
    multiplier =
        (parsed.a->quantization.scale * parsed.b->quantization.scale) /
        parsed.output->quantization.scale;
    if (!isfinite(multiplier) || multiplier <= 0.0f) return -1;
    *descriptor = parsed;
    return 0;
}

static T* qlinear_node_input(Node* n) {
    T* input = nin(n, VX_PORT_INPUT);
    if (!input) input = nin(n, VX_PORT_X);
    if (!input) input = nin(n, VX_PORT_A);
    return input;
}

static T* qconv2d_node_input(Node* n) {
    T* input = nin(n, VX_PORT_INPUT);
    if (!input) input = nin(n, VX_PORT_X);
    return input;
}

static T* qembedding_node_input(Node* n) {
    return nin(n, VX_PORT_INPUT);
}

static int safetensors_affine_quantization_mode(cJSON* root) {
    cJSON* format;
    cJSON* quantization;
    cJSON* storage_format;
    cJSON* tensors;
    if (!cJSON_IsObject(root)) return -1;
    format = cJSON_GetObjectItemCaseSensitive(root, "format");
    if (!cJSON_IsString(format) || !format->valuestring) return -1;
    if (strcmp(format->valuestring, "volvox-graph/v1")) return -1;
    if (cJSON_GetObjectItemCaseSensitive(root, "weights_quantization") ||
        cJSON_GetObjectItemCaseSensitive(root, "weights_quantization_storage"))
        return -1;
    quantization = cJSON_GetObjectItemCaseSensitive(root, "quantization");
    if (!quantization) return 1;
    if (!cJSON_IsObject(quantization) || cJSON_GetArraySize(quantization) != 2)
        return -1;
    storage_format = cJSON_GetObjectItemCaseSensitive(quantization, "format");
    tensors = cJSON_GetObjectItemCaseSensitive(quantization, "tensors");
    if (!cJSON_IsString(storage_format) || !storage_format->valuestring ||
        strcmp(storage_format->valuestring, "volvox-affine-safetensors/v1") ||
        !cJSON_IsObject(tensors) || !tensors->child) return -1;
    for (cJSON* item = quantization->child; item; item = item->next) {
        if (!item->string || (strcmp(item->string, "format") &&
                              strcmp(item->string, "tensors"))) return -1;
    }
    return 1;
}

static cJSON* safetensors_affine_descriptor(cJSON* root, const char* name) {
    cJSON* quantization;
    cJSON* tensors;
    if (!root || !name || safetensors_affine_quantization_mode(root) != 1)
        return NULL;
    quantization = cJSON_GetObjectItemCaseSensitive(root, "quantization");
    tensors = quantization
        ? cJSON_GetObjectItemCaseSensitive(quantization, "tensors") : NULL;
    return cJSON_IsObject(tensors)
        ? cJSON_GetObjectItemCaseSensitive(tensors, name) : NULL;
}

static int safetensors_affine_descriptor_refs(
        cJSON* descriptor, int* per_axis, int* axis,
        const char** scale_name, const char** zero_name) {
    cJSON* scheme;
    cJSON* axis_item;
    cJSON* scale;
    cJSON* zero;
    int saw_scheme = 0;
    int saw_axis = 0;
    int saw_scale = 0;
    int saw_zero = 0;
    if (!cJSON_IsObject(descriptor) || !per_axis || !axis ||
        !scale_name || !zero_name) return -1;
    scheme = cJSON_GetObjectItemCaseSensitive(descriptor, "scheme");
    axis_item = cJSON_GetObjectItemCaseSensitive(descriptor, "axis");
    scale = cJSON_GetObjectItemCaseSensitive(descriptor, "scale_tensor");
    zero = cJSON_GetObjectItemCaseSensitive(descriptor, "zero_point_tensor");
    if (!cJSON_IsString(scheme) || !scheme->valuestring ||
        !cJSON_IsString(scale) || !scale->valuestring || !scale->valuestring[0] ||
        !cJSON_IsString(zero) || !zero->valuestring || !zero->valuestring[0] ||
        !strcmp(scale->valuestring, zero->valuestring)) return -1;
    if (vx_generated_quantization_kind(scheme->valuestring) == VX_QUANTIZATION_PER_TENSOR) {
        *per_axis = 0;
        *axis = 0;
        if (axis_item) return -1;
    } else if (vx_generated_quantization_kind(scheme->valuestring) == VX_QUANTIZATION_PER_AXIS) {
        *per_axis = 1;
        if (!cJSON_IsNumber(axis_item) ||
            axis_item->valuedouble != trunc(axis_item->valuedouble) ||
            axis_item->valuedouble < INT_MIN || axis_item->valuedouble > INT_MAX)
            return -1;
        *axis = axis_item->valueint;
    } else return -1;
    for (cJSON* item = descriptor->child; item; item = item->next) {
        if (!item->string) return -1;
        if (!strcmp(item->string, "scheme") && !saw_scheme) saw_scheme = 1;
        else if (!strcmp(item->string, "axis") && !saw_axis) saw_axis = 1;
        else if (!strcmp(item->string, "scale_tensor") && !saw_scale) saw_scale = 1;
        else if (!strcmp(item->string, "zero_point_tensor") && !saw_zero) saw_zero = 1;
        else return -1;
    }
    if (!saw_scheme || !saw_scale || !saw_zero || saw_axis != *per_axis)
        return -1;
    *scale_name = scale->valuestring;
    *zero_name = zero->valuestring;
    return 0;
}

static int parse_safetensors_per_tensor_quantization(
        cJSON* root, const char* tensor_name, int dtype,
        TensorQuantization* out) {
    cJSON* descriptor;
    int per_axis;
    int axis;
    const char* scale_name;
    const char* zero_name;
    T* scale;
    T* zero;
    float scale_value;
    float zero_value;
    int minimum;
    int maximum;
    TensorQuantization parsed = {0};
    if (!out || !tensor_name) return -1;
    descriptor = safetensors_affine_descriptor(root, tensor_name);
    if (!descriptor) {
        *out = parsed;
        return 0;
    }
    if (dtype != T_I8 && dtype != T_U8) return -1;
    if (safetensors_affine_descriptor_refs(
            descriptor, &per_axis, &axis, &scale_name, &zero_name) != 0 ||
        per_axis || !strcmp(tensor_name, scale_name) ||
        !strcmp(tensor_name, zero_name)) return -1;
    scale = t_find(scale_name);
    zero = t_find(zero_name);
    if (!scale || !zero || !scale->data || !zero->data ||
        scale->dtype != T_F32 || scale->elem_size != sizeof(float) ||
        scale->ndim != 1 || scale->shape[0] != 1 || scale->numel != 1 ||
        zero->dtype != dtype || zero->elem_size != 1 ||
        zero->ndim != 1 || zero->shape[0] != 1 || zero->numel != 1 ||
        !volvoxai_engine_tensor_is_model_weight_locked(scale_name) ||
        !volvoxai_engine_tensor_is_model_weight_locked(zero_name)) return -1;
    scale_value = tensor_scalar_f32(scale, 0);
    zero_value = tensor_scalar_any(zero, 0);
    minimum = dtype == T_I8 ? -128 : 0;
    maximum = dtype == T_I8 ? 127 : 255;
    if (!isfinite(scale_value) || scale_value <= 0.0f ||
        !isfinite(zero_value) || zero_value != truncf(zero_value) ||
        zero_value < minimum || zero_value > maximum) return -1;
    parsed.valid = 1;
    parsed.scale = scale_value;
    parsed.zero_point = (int)zero_value;
    *out = parsed;
    return 0;
}

static int validate_per_axis_quantization(cJSON* descriptor, const T* tensor);

static int parse_graph_tensor_quantization(
        cJSON* root, const char* tensor_name, cJSON* inline_descriptor,
        int dtype, const int* shape, int rank, TensorQuantization* out) {
    int mode = safetensors_affine_quantization_mode(root);
    if (mode != 1 || inline_descriptor || !out) return -1;
    cJSON* descriptor = safetensors_affine_descriptor(root, tensor_name);
    cJSON* scheme = descriptor
        ? cJSON_GetObjectItemCaseSensitive(descriptor, "scheme") : NULL;
    if (cJSON_IsString(scheme) && (vx_generated_quantization_kind(scheme->valuestring) == VX_QUANTIZATION_PER_AXIS)) {
        T tensor = {0};
        if (!tensor_name || strlen(tensor_name) >= sizeof(tensor.name) ||
            rank < 1 || rank > 8 || !shape) return -1;
        memcpy(tensor.name, tensor_name, strlen(tensor_name) + 1u);
        tensor.dtype = dtype;
        tensor.ndim = rank;
        memcpy(tensor.shape, shape, (size_t)rank * sizeof(*shape));
        if (validate_per_axis_quantization(descriptor, &tensor)) return -1;
        /* Per-axis values stay in immutable graph/weight metadata. Never
         * mark them as scalar quantization for per-tensor arithmetic. */
        memset(out, 0, sizeof(*out));
        return 0;
    }
    return parse_safetensors_per_tensor_quantization(
        root, tensor_name, dtype, out);
}

typedef struct {
    const T* scale;
    const T* zero_point;
    int count;
} PerAxisScaleSource;

int volvoxai_engine_tensor_is_quantization_parameter_locked(const char* name) {
    cJSON* quantization;
    cJSON* descriptors;
    if (!name || safetensors_affine_quantization_mode(g_graph_root) != 1)
        return 0;
    quantization = cJSON_GetObjectItemCaseSensitive(g_graph_root, "quantization");
    descriptors = quantization
        ? cJSON_GetObjectItemCaseSensitive(quantization, "tensors") : NULL;
    for (cJSON* descriptor = cJSON_IsObject(descriptors) ? descriptors->child : NULL;
         descriptor; descriptor = descriptor->next) {
        int per_axis;
        int axis;
        const char* scale_name;
        const char* zero_name;
        if (safetensors_affine_descriptor_refs(
                descriptor, &per_axis, &axis, &scale_name, &zero_name) == 0 &&
            (!strcmp(name, scale_name) || !strcmp(name, zero_name))) return 1;
    }
    return 0;
}

static int per_axis_scale_source(cJSON* descriptor, const T* weight,
                                 int expected_count,
                                 PerAxisScaleSource* out) {
    int per_axis;
    int axis;
    const char* scale_name;
    const char* zero_name;
    if (!descriptor || !weight || !out || expected_count <= 0) return -1;
    memset(out, 0, sizeof(*out));
    if (safetensors_affine_descriptor_refs(
            descriptor, &per_axis, &axis, &scale_name, &zero_name) != 0 ||
        !per_axis || !strcmp(weight->name, scale_name) ||
        !strcmp(weight->name, zero_name)) return -1;
    out->scale = t_find(scale_name);
    out->zero_point = t_find(zero_name);
    if (!out->scale || !out->zero_point || !out->scale->data ||
        !out->zero_point->data || out->scale->dtype != T_F32 ||
        out->scale->elem_size != sizeof(float) || out->scale->ndim != 1 ||
        out->scale->shape[0] != expected_count ||
        out->scale->numel != expected_count ||
        out->zero_point->dtype != weight->dtype ||
        out->zero_point->elem_size != 1 || out->zero_point->ndim != 1 ||
        out->zero_point->shape[0] != expected_count ||
        out->zero_point->numel != expected_count ||
        !volvoxai_engine_tensor_is_model_weight_locked(scale_name) ||
        !volvoxai_engine_tensor_is_model_weight_locked(zero_name)) return -1;
    out->count = expected_count;
    return 0;
}

static int per_axis_scale_at(const PerAxisScaleSource* source, int index,
                             float* out) {
    float value;
    if (!source || !out || index < 0 || index >= source->count) return -1;
    value = tensor_scalar_f32(source->scale, index);
    if (!isfinite(value) || value <= 0.0f) return -1;
    *out = value;
    return 0;
}

static int per_axis_zero_point_at(const PerAxisScaleSource* source, int index,
                                  int dtype, int32_t* out) {
    float value;
    int minimum = dtype == T_I8 ? -128 : 0;
    int maximum = dtype == T_I8 ? 127 : 255;
    if (!source || !out || index < 0 || index >= source->count ||
        (dtype != T_I8 && dtype != T_U8)) return -1;
    value = tensor_scalar_any(source->zero_point, index);
    if (!isfinite(value) || value != truncf(value) ||
        value < minimum || value > maximum) return -1;
    *out = (int32_t)value;
    return 0;
}

/* Per-axis descriptors are consumed by QLinear, QEmbedding, and QConv2D.
 * Per-tensor descriptors on byte constants hydrate the ordinary immutable
 * tensor metadata used by typed operators such as QAdd. */
static int validate_per_axis_quantization(cJSON* descriptor, const T* tensor) {
    cJSON* scheme;
    cJSON* axis_json;
    PerAxisScaleSource scales;
    int axis;
    if (!descriptor || !tensor || !cJSON_IsObject(descriptor) ||
        (tensor->dtype != T_I8 && tensor->dtype != T_U8) || tensor->ndim <= 0) return -1;
    scheme = cJSON_GetObjectItem(descriptor, "scheme");
    axis_json = cJSON_GetObjectItem(descriptor, "axis");
    if (!cJSON_IsString(scheme) || !scheme->valuestring || (vx_generated_quantization_kind(scheme->valuestring) != VX_QUANTIZATION_PER_AXIS) ||
        !cJSON_IsNumber(axis_json) || axis_json->valuedouble != trunc(axis_json->valuedouble) ||
        axis_json->valuedouble < (double)INT_MIN ||
        axis_json->valuedouble > (double)INT_MAX) return -1;
    axis = axis_json->valueint;
    if (axis < 0) axis += tensor->ndim;
    if (axis < 0 || axis >= tensor->ndim || tensor->shape[axis] <= 0 ||
        per_axis_scale_source(descriptor, tensor, tensor->shape[axis], &scales) != 0)
        return -1;
    for (int index = 0; index < tensor->shape[axis]; index++) {
        float scale_value;
        int32_t zero_value;
        if (per_axis_scale_at(&scales, index, &scale_value) != 0 ||
            per_axis_zero_point_at(
                &scales, index, tensor->dtype, &zero_value) != 0) return -1;
    }
    return 0;
}

static int apply_weight_quantization(cJSON* root) {
    cJSON* quantization;
    cJSON* descriptors;
    if (safetensors_affine_quantization_mode(root) != 1) return -1;
    quantization = cJSON_GetObjectItemCaseSensitive(root, "quantization");
    if (!quantization) return 0;
    descriptors = cJSON_GetObjectItemCaseSensitive(quantization, "tensors");
    if (!cJSON_IsObject(descriptors)) return -1;
    for (cJSON* item = descriptors->child; item; item = item->next) {
        T* tensor;
        cJSON* scheme;
        if (!item->string || !item->string[0] || !cJSON_IsObject(item)) return -1;
        tensor = t_find(item->string);
        if (!tensor) continue;
        scheme = cJSON_GetObjectItem(item, "scheme");
        if (!scheme || !cJSON_IsString(scheme) || !scheme->valuestring) return -1;
        if (vx_generated_quantization_kind(scheme->valuestring) == VX_QUANTIZATION_PER_TENSOR) {
            if (parse_safetensors_per_tensor_quantization(
                    root, item->string, tensor->dtype,
                    &tensor->quantization) != 0) return -1;
        } else if (vx_generated_quantization_kind(scheme->valuestring) == VX_QUANTIZATION_PER_AXIS) {
            if (validate_per_axis_quantization(item, tensor) != 0) return -1;
        } else {
            return -1;
        }
    }
    return 0;
}

static int qlinear_metadata_build(int node_index, Node* n) {
    QLinearMetadata parsed = {0};
    T* input;
    T* weight;
    T* bias;
    T* output;
    cJSON* descriptor;
    cJSON* scheme;
    cJSON* axis;
    PerAxisScaleSource scales;
    int d_in, d_out;
    long rows;
    if (node_index < 0 || (size_t)node_index >= g_node_capacity ||
        !n || !native_qlinear_op(n->operator_kind)) return -1;
    input = qlinear_node_input(n);
    weight = nin(n, VX_PORT_WEIGHT);
    bias = nin(n, VX_PORT_BIAS);
    output = t_find(n->out);
    if (!input || !weight || !bias || !output || input->ndim <= 0 ||
        output->ndim != input->ndim || weight->ndim != 2 || bias->ndim != 1 ||
        (input->dtype != T_I8 && input->dtype != T_U8) ||
        (weight->dtype != T_I8 && weight->dtype != T_U8) ||
        (output->dtype != T_I8 && output->dtype != T_U8) || bias->dtype != T_I32 ||
        !input->quantization.valid || !output->quantization.valid ||
        /* Physical QLinear/QMatMul/QGemm execute from the aligned metadata
         * bias copy below, not from bias->data. Only an immutable model
         * tensor can therefore be cached without stale public/producer data. */
        !volvoxai_engine_tensor_is_model_weight_locked(bias->name)) return -1;
    d_in = input->shape[input->ndim - 1];
    d_out = output->shape[output->ndim - 1];
    if (d_in <= 0 || d_out <= 0 || weight->shape[0] != d_out || weight->shape[1] != d_in ||
        weight->numel != (long)d_out * d_in || bias->shape[0] != d_out ||
        bias->numel != d_out) return -1;
    for (int dimension = 0; dimension < input->ndim - 1; dimension++) {
        if (input->shape[dimension] != output->shape[dimension]) return -1;
    }
    rows = input->numel / d_in;
    if (rows <= 0 || input->numel != rows * d_in || output->numel != rows * d_out ||
        (unsigned long)rows > UINT32_MAX || (unsigned)d_in > UINT32_MAX ||
        (unsigned)d_out > UINT32_MAX) return -1;
    descriptor = safetensors_affine_descriptor(g_graph_root, weight->name);
    if (!descriptor || !cJSON_IsObject(descriptor)) return -1;
    scheme = cJSON_GetObjectItem(descriptor, "scheme");
    axis = cJSON_GetObjectItem(descriptor, "axis");
    if (!cJSON_IsString(scheme) || !scheme->valuestring || (vx_generated_quantization_kind(scheme->valuestring) != VX_QUANTIZATION_PER_AXIS) ||
        !cJSON_IsNumber(axis) || axis->valuedouble != 0.0 ||
        per_axis_scale_source(descriptor, weight, d_out, &scales) != 0) return -1;
    if ((size_t)d_out > SIZE_MAX / sizeof(*parsed.weight_scales) ||
        (size_t)d_out > SIZE_MAX / sizeof(*parsed.weight_zero_points) ||
        (size_t)d_out > SIZE_MAX / sizeof(*parsed.bias)) return -1;
    parsed.weight_scales = (float*)malloc((size_t)d_out * sizeof(*parsed.weight_scales));
    parsed.weight_zero_points = (int32_t*)malloc((size_t)d_out * sizeof(*parsed.weight_zero_points));
    parsed.bias = (int32_t*)malloc((size_t)d_out * sizeof(*parsed.bias));
    if (!parsed.weight_scales || !parsed.weight_zero_points || !parsed.bias) goto fail;
    parsed.weight_zero_all_zero = 1;
    for (int channel = 0; channel < d_out; channel++) {
        if (per_axis_scale_at(&scales, channel, &parsed.weight_scales[channel]) != 0)
            goto fail;
        if (per_axis_zero_point_at(
                &scales, channel, weight->dtype,
                &parsed.weight_zero_points[channel]) != 0) goto fail;
        if (parsed.weight_zero_points[channel] != 0)
            parsed.weight_zero_all_zero = 0;
        memcpy(&parsed.bias[channel], (const unsigned char*)bias->data +
               (size_t)channel * sizeof(parsed.bias[channel]), sizeof(parsed.bias[channel]));
    }
    parsed.valid = 1;
    parsed.d_in = d_in;
    parsed.d_out = d_out;
    parsed.input_dtype = input->dtype;
    parsed.weight_dtype = weight->dtype;
    parsed.output_dtype = output->dtype;
    parsed.packed_weight_bytes = vx_packed_q8_weight_size((uint32_t)d_in,
        (uint32_t)d_out);
    if (parsed.packed_weight_bytes &&
        volvoxai_engine_tensor_is_model_weight_locked(weight->name)) {
        parsed.packed_weight = malloc(parsed.packed_weight_bytes);
        if (!parsed.packed_weight || !vx_pack_q8_weight(parsed.packed_weight,
                parsed.packed_weight_bytes, weight->data, (uint32_t)d_in,
                (uint32_t)d_out, physical_byte_dtype(weight->dtype), 1u)) {
            free(parsed.packed_weight);
            parsed.packed_weight = NULL;
            parsed.packed_weight_bytes = 0;
        }
    }
    g_qlinear_meta[node_index] = parsed;
    return 0;
fail:
    free(parsed.weight_scales);
    free(parsed.weight_zero_points);
    free(parsed.bias);
    free(parsed.packed_weight);
    return -1;
}

/* QEmbedding has no activation quantization: IDs are ordinary I32 values and
 * each table row has its own I8/U8 descriptor. Keep this strict parser apart
 * from QLinear because the per-axis dimension is vocabulary, not output
 * channels with an accumulator bias. */
static int qembedding_metadata_build(int node_index, Node* n) {
    QEmbeddingMetadata parsed = {0};
    T* input;
    T* weight;
    T* output;
    cJSON* descriptor;
    cJSON* scheme;
    cJSON* axis;
    PerAxisScaleSource scales;
    uint64_t weight_elements;
    uint64_t output_elements;
    int input_refs = 0;
    int weight_refs = 0;
    if (node_index < 0 || (size_t)node_index >= g_node_capacity ||
        !n || !native_qembedding_op(n->operator_kind) ||
        n->nin != 2 || n->nout != 1 || (n->outs[0].port != VX_PORT_OUT)) return -1;
    for (int input_index = 0; input_index < n->nin; input_index++) {
        if (n->ins[input_index].port == VX_PORT_INPUT) input_refs++;
        else if (n->ins[input_index].port == VX_PORT_WEIGHT) weight_refs++;
        else return -1;
    }
    if (input_refs != 1 || weight_refs != 1) return -1;
    input = qembedding_node_input(n);
    weight = nin(n, VX_PORT_WEIGHT);
    output = t_find(n->out);
    if (!input || !weight || !output || input->dtype != T_I32 || input->ndim <= 0 ||
        input->numel <= 0 || (uint64_t)input->numel > UINT32_MAX ||
        input->elem_size != sizeof(int32_t) || weight->ndim != 2 ||
        (weight->dtype != T_I8 && weight->dtype != T_U8) || weight->elem_size != 1 ||
        output->ndim != input->ndim + 1 ||
        (output->dtype != T_I8 && output->dtype != T_U8) || output->elem_size != 1 ||
        !output->quantization.valid || !isfinite(output->quantization.scale) ||
        output->quantization.scale <= 0.0f ||
        output->quantization.zero_point < (output->dtype == T_I8 ? -128 : 0) ||
        output->quantization.zero_point > (output->dtype == T_I8 ? 127 : 255) ||
        weight->shape[0] <= 0 || weight->shape[1] <= 0 ||
        !qembedding_ids_preflight_complete(n, input, weight->shape[0])) return -1;
    for (int dimension = 0; dimension < input->ndim; dimension++) {
        if (input->shape[dimension] <= 0 || output->shape[dimension] != input->shape[dimension])
            return -1;
    }
    if (output->shape[input->ndim] != weight->shape[1]) return -1;
    weight_elements = (uint64_t)(uint32_t)weight->shape[0] * (uint32_t)weight->shape[1];
    output_elements = (uint64_t)input->numel * (uint32_t)weight->shape[1];
    if (weight_elements > (uint64_t)LONG_MAX || output_elements > (uint64_t)LONG_MAX ||
        weight->numel != (long)weight_elements || output->numel != (long)output_elements)
        return -1;
    descriptor = safetensors_affine_descriptor(g_graph_root, weight->name);
    if (!descriptor || !cJSON_IsObject(descriptor)) return -1;
    scheme = cJSON_GetObjectItem(descriptor, "scheme");
    axis = cJSON_GetObjectItem(descriptor, "axis");
    if (!cJSON_IsString(scheme) || !scheme->valuestring || (vx_generated_quantization_kind(scheme->valuestring) != VX_QUANTIZATION_PER_AXIS) ||
        !cJSON_IsNumber(axis) || axis->valuedouble != 0.0 ||
        per_axis_scale_source(descriptor, weight, weight->shape[0], &scales) != 0)
        return -1;
    if ((size_t)weight->shape[0] > SIZE_MAX / sizeof(*parsed.weight_scales) ||
        (size_t)weight->shape[0] > SIZE_MAX / sizeof(*parsed.weight_zero_points)) return -1;
    parsed.weight_scales = (float*)malloc((size_t)weight->shape[0] * sizeof(*parsed.weight_scales));
    parsed.weight_zero_points = (int32_t*)malloc((size_t)weight->shape[0] *
                                                  sizeof(*parsed.weight_zero_points));
    if (!parsed.weight_scales || !parsed.weight_zero_points) goto fail;
    for (int row = 0; row < weight->shape[0]; row++) {
        if (per_axis_scale_at(&scales, row, &parsed.weight_scales[row]) != 0)
            goto fail;
        if (per_axis_zero_point_at(
                &scales, row, weight->dtype,
                &parsed.weight_zero_points[row]) != 0) goto fail;
    }
    parsed.valid = 1;
    parsed.vocab = (uint32_t)weight->shape[0];
    parsed.hidden = (uint32_t)weight->shape[1];
    parsed.weight_dtype = weight->dtype;
    parsed.output_dtype = output->dtype;
    g_qembedding_meta[node_index] = parsed;
    return 0;
fail:
    free(parsed.weight_scales);
    free(parsed.weight_zero_points);
    return -1;
}

static int qconv2d_product4(const uint32_t dimensions[4], uint64_t* product) {
    uint64_t value = 1;
    if (!dimensions || !product) return -1;
    for (int i = 0; i < 4; i++) {
        if (!dimensions[i] || value > UINT64_MAX / dimensions[i]) return -1;
        value *= dimensions[i];
    }
    *product = value;
    return 0;
}

static int qconv2d_shape4(const T* tensor, uint32_t dimensions[4]) {
    uint64_t elements;
    if (!tensor || !dimensions || tensor->ndim != 4 || tensor->numel <= 0) return -1;
    for (int i = 0; i < 4; i++) {
        if (tensor->shape[i] <= 0) return -1;
        dimensions[i] = (uint32_t)tensor->shape[i];
    }
    if (qconv2d_product4(dimensions, &elements) != 0 || elements > (uint64_t)LONG_MAX ||
        tensor->numel != (long)elements) return -1;
    return 0;
}

static int qconv2d_weight_only_parameter(const Node* node, int* enabled) {
    if (!node || !enabled) return -1;
    if (node->parsed_params.parsed != 1 ||
        !node->parsed_params.weight_only_valid) return -1;
    *enabled = node->parsed_params.weight_only;
    return 0;
}

/* The weight-only spelling owns a persistent dequantized-weight cache, so all
 * quantization operands must be immutable model tensors and the complete
 * NHWC/OHWI convolution contract must be known before the first forward. */
static int qconv2d_weight_only_validate(Node* node) {
    T* input;
    T* weight;
    T* weight_scale;
    T* weight_zero_point;
    T* bias;
    T* output;
    T* declared_output;
    uint32_t input_shape[4], weight_shape[4], output_shape[4];
    uint32_t stride[2], dilation[2], pads[4], groups;
    uint64_t padded_height, padded_width, effective_height, effective_width;
    uint64_t expected_height, expected_width;
    int input_refs = 0, weight_refs = 0, scale_refs = 0;
    int zero_point_refs = 0, bias_refs = 0;
    if (!node || (node->operator_kind != VX_OP_Q_CONV_2D) || node->nout != 1 ||
        (node->outs[0].port != VX_PORT_OUT)) return -1;
    for (int index = 0; index < node->nin; index++) {
        VxPortKind key = node->ins[index].port;
        if (!(key != VX_PORT_INPUT)) input_refs++;
        else if (!(key != VX_PORT_WEIGHT)) weight_refs++;
        else if (!(key != VX_PORT_WEIGHT_SCALE)) scale_refs++;
        else if (!(key != VX_PORT_WEIGHT_ZERO_POINT)) zero_point_refs++;
        else if (!(key != VX_PORT_BIAS)) bias_refs++;
        else return -1;
    }
    if (input_refs != 1 || weight_refs != 1 || scale_refs != 1 ||
        zero_point_refs > 1 || bias_refs > 1) return -1;
    input = nin(node, VX_PORT_INPUT);
    weight = nin(node, VX_PORT_WEIGHT);
    weight_scale = nin(node, VX_PORT_WEIGHT_SCALE);
    weight_zero_point = nin(node, VX_PORT_WEIGHT_ZERO_POINT);
    bias = nin(node, VX_PORT_BIAS);
    output = t_find(node->out);
    declared_output = nout(node, VX_PORT_OUT);
    if (!input || !weight || !weight_scale || !output || !declared_output ||
        !input->data || !weight->data || !weight_scale->data || !output->data ||
        !declared_output->data || input->dtype != T_F32 || output->dtype != T_F32 ||
        declared_output->dtype != T_F32 || input->elem_size != sizeof(float) ||
        output->elem_size != sizeof(float) || declared_output->elem_size != sizeof(float) ||
        (weight->dtype != T_I8 && weight->dtype != T_U8) || weight->elem_size != 1 ||
        weight_scale->dtype != T_F32 || weight_scale->elem_size != sizeof(float) ||
        weight_scale->ndim != 1 || input->quantization.valid ||
        output->quantization.valid || declared_output->quantization.valid ||
        qconv2d_shape4(input, input_shape) != 0 ||
        qconv2d_shape4(weight, weight_shape) != 0 ||
        qconv2d_shape4(output, output_shape) != 0 ||
        !volvoxai_engine_tensor_is_model_weight_locked(weight->name) ||
        !volvoxai_engine_tensor_is_model_weight_locked(weight_scale->name)) return -1;
    if (output != declared_output) {
        uint32_t declared_shape[4];
        if (qconv2d_shape4(declared_output, declared_shape) != 0) return -1;
        for (int dimension = 0; dimension < 4; dimension++)
            if (declared_shape[dimension] != output_shape[dimension]) return -1;
    }
    if (weight_scale->numel != 1 && weight_scale->numel != weight_shape[0]) return -1;
    for (long index = 0; index < weight_scale->numel; index++) {
        float value = tensor_scalar_f32(weight_scale, index);
        if (!isfinite(value) || value <= 0.0f) return -1;
    }
    if (weight_zero_point) {
        int minimum = weight->dtype == T_I8 ? -128 : 0;
        int maximum = weight->dtype == T_I8 ? 127 : 255;
        if (!weight_zero_point->data || weight_zero_point->ndim != 1 ||
            (weight_zero_point->dtype != T_I8 && weight_zero_point->dtype != T_U8 &&
             weight_zero_point->dtype != T_I32) ||
            (weight_zero_point->numel != 1 &&
             weight_zero_point->numel != weight_shape[0]) ||
            !volvoxai_engine_tensor_is_model_weight_locked(weight_zero_point->name)) return -1;
        for (long index = 0; index < weight_zero_point->numel; index++) {
            float value = tensor_scalar_any(weight_zero_point, index);
            if (!isfinite(value) || value != truncf(value) ||
                value < (float)minimum || value > (float)maximum) return -1;
        }
    }
    if (bias) {
        if (!bias->data || bias->dtype != T_F32 || bias->elem_size != sizeof(float) ||
            bias->ndim != 1 || bias->numel != weight_shape[0] ||
            !volvoxai_engine_tensor_is_model_weight_locked(bias->name)) return -1;
        for (long index = 0; index < bias->numel; index++)
            if (!isfinite(tensor_scalar_f32(bias, index))) return -1;
    }
    if (node->parsed_params.parsed != 1 ||
        !node->parsed_params.weight_only_valid ||
        node->parsed_params.data_layout != VX_DATA_LAYOUT_NHWC ||
        node->parsed_params.weight_layout != VX_CONV_WEIGHT_LAYOUT_OHWI)
        return -1;
    stride[0] = (uint32_t)node->parsed_params.stride[0];
    stride[1] = (uint32_t)node->parsed_params.stride[1];
    dilation[0] = (uint32_t)node->parsed_params.dilation[0];
    dilation[1] = (uint32_t)node->parsed_params.dilation[1];
    for (int index = 0; index < 4; index++)
        pads[index] = (uint32_t)node->parsed_params.pads[index];
    groups = (uint32_t)node->parsed_params.groups;
    if (groups > INT_MAX || stride[0] > INT_MAX || stride[1] > INT_MAX ||
        dilation[0] > INT_MAX || dilation[1] > INT_MAX ||
        pads[0] > INT_MAX || pads[1] > INT_MAX || pads[2] > INT_MAX ||
        pads[3] > INT_MAX || output_shape[0] != input_shape[0] ||
        weight_shape[0] != output_shape[3] || input_shape[3] % groups ||
        output_shape[3] % groups ||
        (uint64_t)weight_shape[3] * groups != input_shape[3]) return -1;
    padded_height = (uint64_t)input_shape[1] + pads[0] + pads[2];
    padded_width = (uint64_t)input_shape[2] + pads[1] + pads[3];
    effective_height = (uint64_t)(weight_shape[1] - 1u) * dilation[0] + 1u;
    effective_width = (uint64_t)(weight_shape[2] - 1u) * dilation[1] + 1u;
    if (padded_height < effective_height || padded_width < effective_width) return -1;
    expected_height = (padded_height - effective_height) / stride[0] + 1u;
    expected_width = (padded_width - effective_width) / stride[1] + 1u;
    return expected_height == output_shape[1] && expected_width == output_shape[2] ? 0 : -1;
}

static int qconv2d_weight_quantization_build(const T* weight, uint32_t output_channels,
                                             float** scales_out, int32_t** zero_points_out) {
    cJSON* descriptor;
    cJSON* scheme;
    cJSON* axis;
    PerAxisScaleSource scales;
    float* parsed_scales = NULL;
    int32_t* parsed_zero_points = NULL;
    if (!weight || !scales_out || !zero_points_out || !output_channels ||
        output_channels > (uint32_t)INT_MAX) return -1;
    *scales_out = NULL;
    *zero_points_out = NULL;
    descriptor = safetensors_affine_descriptor(g_graph_root, weight->name);
    if (!descriptor || !cJSON_IsObject(descriptor)) return -1;
    scheme = cJSON_GetObjectItem(descriptor, "scheme");
    axis = cJSON_GetObjectItem(descriptor, "axis");
    if (!cJSON_IsString(scheme) || !scheme->valuestring || (vx_generated_quantization_kind(scheme->valuestring) != VX_QUANTIZATION_PER_AXIS) ||
        !cJSON_IsNumber(axis) || axis->valuedouble != 0.0 ||
        per_axis_scale_source(descriptor, weight, (int)output_channels, &scales) != 0)
        return -1;
    if ((size_t)output_channels > SIZE_MAX / sizeof(*parsed_scales) ||
        (size_t)output_channels > SIZE_MAX / sizeof(*parsed_zero_points)) return -1;
    parsed_scales = (float*)malloc((size_t)output_channels * sizeof(*parsed_scales));
    parsed_zero_points = (int32_t*)malloc((size_t)output_channels * sizeof(*parsed_zero_points));
    if (!parsed_scales || !parsed_zero_points) goto fail;
    for (uint32_t channel = 0; channel < output_channels; channel++) {
        if (per_axis_scale_at(&scales, (int)channel, &parsed_scales[channel]) != 0)
            goto fail;
        if (per_axis_zero_point_at(
                &scales, (int)channel, weight->dtype,
                &parsed_zero_points[channel]) != 0) goto fail;
    }
    *scales_out = parsed_scales;
    *zero_points_out = parsed_zero_points;
    return 0;
fail:
    free(parsed_scales);
    free(parsed_zero_points);
    return -1;
}

/* Returns 1 for a complete physical-byte descriptor, 0 when no physical-byte
 * descriptor is present, and -1 for a malformed typed QConv2D. Callers may
 * separately admit explicit weight-only nodes, but must reject every other
 * zero result rather than inventing activation quantizers. */
static int qconv2d_metadata_build(int node_index, Node* n) {
    QConv2DMetadata parsed = {0};
    T* input;
    T* weight;
    T* bias;
    T* output;
    uint32_t input_shape[4], weight_shape[4], output_shape[4];
    uint32_t stride[2], dilation[2], pads[4];
    uint64_t expected_weight_elements;
    uint64_t padded_height, padded_width, effective_height, effective_width;
    uint64_t expected_height, expected_width;
    int physical_hint;
    int symmetric_i8_weight = 0;
    if (node_index < 0 || (size_t)node_index >= g_node_capacity ||
        !n || !native_qconv2d_op(n->operator_kind)) return -1;
    input = qconv2d_node_input(n);
    weight = nin(n, VX_PORT_WEIGHT);
    bias = nin(n, VX_PORT_BIAS);
    output = t_find(n->out);
    physical_hint = (input && ((input->dtype == T_I8 || input->dtype == T_U8) ||
                               input->quantization.valid)) ||
                    (output && ((output->dtype == T_I8 || output->dtype == T_U8) ||
                                output->quantization.valid));
    if (!physical_hint) return 0;
    if (!input || !weight || !output || !input->data || !weight->data || !output->data ||
        (input->dtype != T_I8 && input->dtype != T_U8) ||
        (weight->dtype != T_I8 && weight->dtype != T_U8) ||
        (output->dtype != T_I8 && output->dtype != T_U8) ||
        input->elem_size != 1 || weight->elem_size != 1 || output->elem_size != 1 ||
        !input->quantization.valid || !output->quantization.valid ||
        qconv2d_shape4(input, input_shape) != 0 ||
        qconv2d_shape4(weight, weight_shape) != 0 ||
        qconv2d_shape4(output, output_shape) != 0) goto fail;
    if (n->parsed_params.parsed != 1 || n->parsed_params.groups <= 0 ||
        n->parsed_params.relu < 0 || n->parsed_params.relu > 2 ||
        n->parsed_params.data_layout != VX_DATA_LAYOUT_NHWC ||
        n->parsed_params.weight_layout != VX_CONV_WEIGHT_LAYOUT_OHWI ||
        n->parsed_params.stride[0] <= 0 || n->parsed_params.stride[1] <= 0 ||
        n->parsed_params.dilation[0] <= 0 || n->parsed_params.dilation[1] <= 0)
        goto fail;
    parsed.groups = (uint32_t)n->parsed_params.groups;
    parsed.relu = (uint32_t)n->parsed_params.relu;
    for (int dimension = 0; dimension < 2; dimension++) {
        stride[dimension] = (uint32_t)n->parsed_params.stride[dimension];
        dilation[dimension] = (uint32_t)n->parsed_params.dilation[dimension];
    }
    for (int dimension = 0; dimension < 4; dimension++) {
        if (n->parsed_params.pads[dimension] < 0) goto fail;
        pads[dimension] = (uint32_t)n->parsed_params.pads[dimension];
    }
    if (n->fuse_relu6) parsed.relu = 2;
    parsed.batch = input_shape[0];
    parsed.input_height = input_shape[1];
    parsed.input_width = input_shape[2];
    parsed.input_channels = input_shape[3];
    parsed.output_height = output_shape[1];
    parsed.output_width = output_shape[2];
    parsed.output_channels = output_shape[3];
    parsed.kernel_height = weight_shape[1];
    parsed.kernel_width = weight_shape[2];
    parsed.input_per_group = weight_shape[3];
    parsed.stride_y = stride[0];
    parsed.stride_x = stride[1];
    parsed.dilation_y = dilation[0];
    parsed.dilation_x = dilation[1];
    parsed.padding_top = pads[0];
    parsed.padding_left = pads[1];
    parsed.padding_bottom = pads[2];
    parsed.padding_right = pads[3];
    if (output_shape[0] != parsed.batch || weight_shape[0] != parsed.output_channels ||
        !parsed.groups || parsed.input_channels % parsed.groups ||
        parsed.output_channels % parsed.groups ||
        (uint64_t)parsed.input_per_group * parsed.groups != parsed.input_channels ||
        qconv2d_product4(weight_shape, &expected_weight_elements) != 0 ||
        expected_weight_elements > (uint64_t)LONG_MAX ||
        weight->numel != (long)expected_weight_elements) goto fail;
    padded_height = (uint64_t)parsed.input_height + parsed.padding_top + parsed.padding_bottom;
    padded_width = (uint64_t)parsed.input_width + parsed.padding_left + parsed.padding_right;
    effective_height = (uint64_t)(parsed.kernel_height - 1u) * parsed.dilation_y + 1u;
    effective_width = (uint64_t)(parsed.kernel_width - 1u) * parsed.dilation_x + 1u;
    if (padded_height < effective_height || padded_width < effective_width) goto fail;
    expected_height = (padded_height - effective_height) / parsed.stride_y + 1u;
    expected_width = (padded_width - effective_width) / parsed.stride_x + 1u;
    if (expected_height != parsed.output_height || expected_width != parsed.output_width) goto fail;
    if (bias && (!bias->data || bias->dtype != T_I32 || bias->ndim != 1 ||
                 bias->shape[0] != (int)parsed.output_channels ||
                 bias->numel != (long)parsed.output_channels ||
                 /* The physical path reads parsed.bias, so a mutable graph
                  * input or producer output would otherwise be silently
                  * frozen at model-build time. */
                 !volvoxai_engine_tensor_is_model_weight_locked(bias->name)))
        goto fail;
    if (qconv2d_weight_quantization_build(weight, parsed.output_channels,
                                          &parsed.weight_scales,
                                          &parsed.weight_zero_points) != 0) goto fail;
    if (weight->dtype == T_I8) {
        symmetric_i8_weight = 1;
        for (uint32_t channel = 0; channel < parsed.output_channels; channel++) {
            if (parsed.weight_zero_points[channel] != 0) {
                symmetric_i8_weight = 0;
                break;
            }
        }
    }
    if (bias) {
        if ((size_t)parsed.output_channels > SIZE_MAX / sizeof(*parsed.bias)) goto fail;
        parsed.bias = (int32_t*)malloc((size_t)parsed.output_channels * sizeof(*parsed.bias));
        if (!parsed.bias) goto fail;
        for (uint32_t channel = 0; channel < parsed.output_channels; channel++) {
            memcpy(&parsed.bias[channel], (const unsigned char*)bias->data +
                   (size_t)channel * sizeof(parsed.bias[channel]), sizeof(parsed.bias[channel]));
        }
    }
    parsed.input_dtype = input->dtype;
    parsed.weight_dtype = weight->dtype;
    parsed.output_dtype = output->dtype;
    if (symmetric_i8_weight && parsed.groups == 1u && parsed.relu == 0u && parsed.bias &&
        parsed.kernel_height == 3u && parsed.kernel_width == 3u &&
        volvoxai_engine_tensor_is_model_weight_locked(weight->name)) {
        const uint64_t d_in = (uint64_t)parsed.kernel_height *
            parsed.kernel_width * parsed.input_per_group;
        if (d_in <= UINT32_MAX) {
            parsed.packed_weight_bytes = vx_packed_q8_weight_size(
                (uint32_t)d_in, parsed.output_channels);
            if (parsed.packed_weight_bytes) {
                parsed.packed_weight = malloc(parsed.packed_weight_bytes);
                if (!parsed.packed_weight || !vx_pack_q8_weight(
                        parsed.packed_weight, parsed.packed_weight_bytes,
                        weight->data, (uint32_t)d_in, parsed.output_channels,
                        physical_byte_dtype(weight->dtype), 1u)) {
                    free(parsed.packed_weight);
                    parsed.packed_weight = NULL;
                    parsed.packed_weight_bytes = 0;
                }
            }
        }
    }
    if (volvoxai_engine_tensor_is_model_weight_locked(weight->name)) {
        /* Narrow-input convolutions (the RGB stem, say) cannot vectorize their
         * channel dot product, so the AVX2 kernel transposes OHWI to put eight
         * output channels under one input load. The bytes never change for an
         * immutable weight, so pay for the transpose once here instead of on
         * every call. A zero size means this geometry or target never takes
         * that path. */
        const size_t small_c_bytes = vx_w8a8_qconv_small_c_pack_size(
            parsed.kernel_height, parsed.kernel_width, parsed.input_per_group,
            parsed.output_channels, parsed.groups);
        if (small_c_bytes) {
            parsed.small_c_packed_weight = malloc(small_c_bytes);
            if (!parsed.small_c_packed_weight || !vx_w8a8_qconv_pack_small_c(
                    parsed.small_c_packed_weight, small_c_bytes, weight->data,
                    parsed.kernel_height, parsed.kernel_width,
                    parsed.input_per_group, parsed.output_channels,
                    parsed.groups)) {
                free(parsed.small_c_packed_weight);
                parsed.small_c_packed_weight = NULL;
            }
        }
    }
    parsed.valid = 1;
    g_qconv_meta[node_index] = parsed;
    return 1;
fail:
    free(parsed.weight_scales);
    free(parsed.weight_zero_points);
    free(parsed.bias);
    free(parsed.packed_weight);
    free(parsed.small_c_packed_weight);
    return -1;
}

static size_t tensor_bytes(const T* t) {
    return t ? (size_t)t->numel * t->elem_size : 0;
}

static int gpu_graph_storage_tensor(const T* t) {
    if (!t) return 0;
    if (t->dtype == T_F32 || t->dtype == T_I32 ||
        ((t->dtype == T_I8 || t->dtype == T_U8) && t->quantization.valid)) return 1;
    return 0;
}

static void vk_mark_host_tensor(T* t) {
    if (gpu_graph_storage_tensor(t))
        vx_runtime_backend_mark_host(
            t->data, tensor_bytes(t), volvoxai_engine_tensor_is_model_weight_locked(t->name));
}

static void vk_mark_node_outputs_host(Node* node, T* actual_output) {
    if (!node) return;
    /* A pure CPU route has no device ownership state to invalidate.  Avoid
     * resolving every declared output and probing model-weight ownership on
     * each incremental token.  CPU fallbacks inside either a built-in graph
     * graph backend still take the coherence path below. */
    if (!vx_runtime_backend_has_graph()) return;
    if (actual_output) {
        vk_mark_host_tensor(actual_output);
    }
    T* primary = t_find(node->out);
    /* A private fusion can write a skipped peer or redirected output while
     * the declared outputs still describe the original graph node. Mark
     * only the actual write in that case.  Ordinary multi-output nodes retain
     * Node.out == outs[0] and need every declared output invalidated. */
    if (actual_output != primary || node->nout <= 0 ||
        strcmp(node->out, node->outs[0].name)) return;
    for (int output = 0; output < node->nout; output++) {
        T* tensor = t_find(node->outs[output].name);
        int duplicate = 0;
        if (!tensor || (actual_output && actual_output->data == tensor->data)) continue;
        for (int prior = 0; prior < output; prior++) {
            T* previous = t_find(node->outs[prior].name);
            if (previous && previous->data == tensor->data) {
                duplicate = 1;
                break;
            }
        }
        if (!duplicate) {
            vk_mark_host_tensor(tensor);
        }
    }
}

/* The graph loop only consumes synchronously readable host storage. */
int vk_sync_host_tensor(T* t) {
    if (!gpu_graph_storage_tensor(t)) return VX_SYNC_OK;
    return vx_runtime_backend_sync_host(
        t->data, tensor_bytes(t),
        volvoxai_engine_tensor_is_model_weight_locked(t->name));
}

static int vk_sync_node_inputs(Node* n) {
    if (!vx_runtime_backend_has_graph() || !n) return 0;
    for (int i = 0; i < n->nin; i++) {
        T* t = t_find(n->ins[i].name);
        if (vk_sync_host_tensor(t) != 0) return -1;
    }
    return 0;
}

static void materialize_node_inputs(Node* n) {
    if (!n) return;
    for (int i = 0; i < n->nin; i++) {
        T* t = t_find(n->ins[i].name);
        materialize_tensor_f32(t);
    }
}

void vk_mark_owned_tensors_host_dirty(void) {
    if (!vx_runtime_backend_has_graph()) return;
    for (int i = 0; i < g_nt; i++) {
        if (g_t[i].owns && gpu_graph_storage_tensor(&g_t[i])) {
            vx_runtime_backend_mark_host(g_t[i].data, tensor_bytes(&g_t[i]), 0);
        }
    }
}

// ---- build graph from graph.json -----------------------------------------
// Fuse ReLU6 (Clip min=0,max=6) into the conv that produces its input. When the conv's
// output feeds only the Clip, we redirect the conv to write the Clip's output tensor with a
// clamped epilogue and skip the Clip node — removing a full read+write pass over the tensor.
static const char* node_input_name(Node* n) {
    for (int i = 0; i < n->nin; i++) if (n->ins[i].port == VX_PORT_INPUT) return n->ins[i].name;
    return n->nin > 0 ? n->ins[0].name : NULL;
}
static int tensor_use_count(const char* name) {
    int uses = 0;
    for (int i = 0; i < g_nn; i++) for (int j = 0; j < g_n[i].nin; j++)
        if (!strcmp(g_n[i].ins[j].name, name)) uses++;
    return uses;
}
static int tensor_same_shape(const T* a, const T* b) {
    if (!a || !b || a->ndim != b->ndim || a->numel != b->numel) return 0;
    for (int i = 0; i < a->ndim; i++) if (a->shape[i] != b->shape[i]) return 0;
    return 1;
}
static int producer_index_for(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < g_nn; i++) if (!g_n[i].skip && !strcmp(g_n[i].out, name)) return i;
    return -1;
}
static int producer_index_any(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < g_nn; i++) if (!strcmp(g_n[i].out, name)) return i;
    return -1;
}
#define VOLVOX_ENGINE_PRIVATE_INCLUDE 1
#include "graph_opt_fusion.inc"
#undef VOLVOX_ENGINE_PRIVATE_INCLUDE
static int build_quant_params_pass(void) {
    volvoxai_engine_clear_qlinear_metadata();
    volvoxai_engine_clear_qconv_metadata();
    volvoxai_engine_clear_qembedding_metadata();
    for (int i = 0; i < g_nn; i++) {
        Node* n = &g_n[i];
        if (n->operator_kind == VX_OP_ADD) {
            T* a = nin(n, VX_PORT_A);
            T* b = nin(n, VX_PORT_B);
            T* output = t_find(n->out);
            int physical_bytes =
                (a && (a->dtype == T_I8 || a->dtype == T_U8 || a->quantization.valid)) ||
                (b && (b->dtype == T_I8 || b->dtype == T_U8 || b->quantization.valid)) ||
                (output && (output->dtype == T_I8 || output->dtype == T_U8 ||
                            output->quantization.valid));
            if (physical_bytes) {
                vx_engine_log(
                        "[engine] generic Add cannot consume or produce I8/U8 tensors; "
                        "use QAdd with explicit quantization metadata\n");
                goto fail;
            }
        }
        if ((n->operator_kind == VX_OP_TRANSPOSE) || (n->operator_kind == VX_OP_EXPAND)) {
            T* input = nin(n, VX_PORT_INPUT);
            if (!input) input = nin(n, VX_PORT_X);
            if (!input) input = nin(n, VX_PORT_DATA);
            T* output = t_find(n->out);
            int physical_bytes =
                (input && (input->dtype == T_I8 || input->dtype == T_U8 ||
                           input->quantization.valid)) ||
                (output && (output->dtype == T_I8 || output->dtype == T_U8 ||
                            output->quantization.valid));
            if (physical_bytes &&
                physical_shape_model_validate(n, output) != 0) {
                vx_engine_log(
                        "[engine] typed shape operator has an invalid "
                        "descriptor-preserving I8/U8 contract\n");
                goto fail;
            }
        }
        if (native_qlinear_op(n->operator_kind)) {
            if (qlinear_metadata_build(i, n) != 0) goto fail;
            continue;
        }
        if (native_qembedding_op(n->operator_kind)) {
            if (qembedding_metadata_build(i, n) != 0) goto fail;
            continue;
        }
        if (n->operator_kind == VX_OP_Q_CONV_2D) {
            int weight_only;
            if (qconv2d_weight_only_parameter(n, &weight_only) != 0) goto fail;
            if (weight_only) {
                if (qconv2d_weight_only_validate(n) != 0) goto fail;
                continue;
            }
            int qconv = qconv2d_metadata_build(i, n);
            if (qconv < 0) goto fail;
            if (!qconv) {
                vx_engine_log(
                        "[engine] QConv2D requires explicit I8/U8 input and output "
                        "quantization metadata or params.weight_only=true\n");
                goto fail;
            }
            continue;
        }
        if (n->operator_kind == VX_OP_Q_BATCH_MATMUL) {
            QBatchMatMulPhysicalDescriptor descriptor;
            if (qbatch_matmul_physical_validate(
                    n, t_find(n->out), &descriptor) != 0) {
                vx_engine_log(
                        "[engine] QBatchMatMul requires exact per-tensor "
                        "I8/U8 {a,b}->{out} descriptors, rank-2..8 ONNX "
                        "broadcasting, and an I32-safe contraction\n");
                goto fail;
            }
            continue;
        }
    }
    return 0;
fail:
    volvoxai_engine_clear_qlinear_metadata();
    volvoxai_engine_clear_qconv_metadata();
    volvoxai_engine_clear_qembedding_metadata();
    return -1;
}

static int graph_declared_dtype(cJSON* item) {
    if (!item) return T_F32;
    if (!cJSON_IsString(item) || !item->valuestring) return -1;
    const char* value = item->valuestring;
    if (!strcmp(value, "float32")) return T_F32;
    if (!strcmp(value, "int32")) return T_I32;
    if (!strcmp(value, "int8")) return T_I8;
    if (!strcmp(value, "uint8")) return T_U8;
    return -1;
}

static int graph_input_dtype(cJSON* descriptor) {
    cJSON* declared = descriptor
        ? cJSON_GetObjectItemCaseSensitive(descriptor, "dtype") : NULL;
    int dtype;
    if (!declared) return -1;
    dtype = graph_declared_dtype(declared);
    return dtype;
}

static int graph_parse_shape(cJSON* item, int required, int* shape, int* ndim, long* numel) {
    if (!item) {
        if (required) return -1;
        if (ndim) *ndim = 0;
        if (numel) *numel = 1;
        return 0;
    }
    if (!cJSON_IsArray(item)) return -1;
    int rank = cJSON_GetArraySize(item);
    if (rank < 0 || rank > 8) return -1;
    int parsed[8] = {0};
    for (int index = 0; index < rank; index++) {
        cJSON* dimension = cJSON_GetArrayItem(item, index);
        if (!cJSON_IsNumber(dimension) || !isfinite(dimension->valuedouble) ||
            dimension->valuedouble != (double)dimension->valueint ||
            dimension->valueint <= 0) return -1;
        parsed[index] = dimension->valueint;
    }
    long count = numel_of(parsed, rank);
    if (count < 0) return -1;
    if (shape) for (int index = 0; index < rank; index++) shape[index] = parsed[index];
    if (ndim) *ndim = rank;
    if (numel) *numel = count;
    return 0;
}

static int graph_json_name_valid(const char* name, size_t capacity) {
    return name && name[0] && strlen(name) < capacity;
}

static int graph_object_has_prior_key(const cJSON* object, const cJSON* stop, const char* key) {
    if (!object || !key) return 0;
    for (const cJSON* item = object->child; item && item != stop; item = item->next) {
        if (item->string && !strcmp(item->string, key)) return 1;
    }
    return 0;
}

static int graph_inputs_declare(const cJSON* inputs, const char* name) {
    if (!inputs || !name) return 0;
    for (const cJSON* item = inputs->child; item; item = item->next) {
        if (item->string && !strcmp(item->string, name)) return 1;
    }
    return 0;
}

static int graph_nodes_declare_before(const cJSON* nodes, const cJSON* stop,
                                      const char* name) {
    if (!nodes || !name) return 0;
    for (const cJSON* node = nodes->child; node && node != stop; node = node->next) {
        cJSON* outputs = cJSON_GetObjectItemCaseSensitive(node, "outputs");
        for (cJSON* output = outputs ? outputs->child : NULL; output; output = output->next) {
            if (cJSON_IsString(output) && output->valuestring &&
                !strcmp(output->valuestring, name)) return 1;
        }
    }
    return 0;
}

static int graph_tensor_declared_before(const cJSON* inputs, const cJSON* nodes,
                                        const cJSON* node, const char* name) {
    return t_find(name) || graph_inputs_declare(inputs, name) ||
           graph_nodes_declare_before(nodes, node, name);
}

static int graph_validate_output_metadata(const cJSON* outputs, const cJSON* metadata) {
    if (!metadata) return 0;
    if (!cJSON_IsObject(metadata)) return -1;
    for (const cJSON* item = metadata->child; item; item = item->next) {
        if (!graph_json_name_valid(item->string, sizeof(((Ref*)0)->key)) ||
            !outputs ||
            !cJSON_GetObjectItemCaseSensitive(outputs, item->string)) return -1;
    }
    return 0;
}

/* Affine values and their association live in safetensors plus the central
 * reference table. Operator parameters may still contain an unrelated
 * semantic scalar such as SDPA.params.scale, but none of the retired
 * activation/weight quantization spellings are graph topology. */
static int graph_value_contains_affine_alias(const cJSON* value) {
    static const char* const forbidden[] = {
        "quantization",
        "input_scale", "input_zero_point",
        "output_scale", "output_zero_point",
        "weight_scale", "weight_zero_point",
        "zero_point",
        "scales", "zero_points",
        "scale_tensor", "zero_point_tensor",
    };
    for (const cJSON* item = value ? value->child : NULL;
         item; item = item->next) {
        for (size_t index = 0;
             item->string && index < sizeof(forbidden) / sizeof(forbidden[0]);
             index++) {
            if (!strcmp(item->string, forbidden[index])) return 1;
        }
        if (graph_value_contains_affine_alias(item)) return 1;
    }
    return 0;
}

static int graph_params_contain_affine_payload(const cJSON* params) {
    if (!params) return 0;
    return !cJSON_IsObject(params) ||
           graph_value_contains_affine_alias(params);
}

static int graph_safetensors_tensor_occurrences(const char* name) {
    int occurrences = 0;
    if (!name) return 0;
    for (int file = 0; file < g_weight_file_count; file++) {
        if (safetensors_find_tensor(&g_weight_files[file], name)) occurrences++;
    }
    return occurrences;
}

static int graph_outputs_name(const cJSON* outputs, const char* name) {
    for (const cJSON* item = cJSON_IsArray(outputs) ? outputs->child : NULL;
         item; item = item->next) {
        if (cJSON_IsString(item) && item->valuestring &&
            !strcmp(item->valuestring, name)) return 1;
    }
    return 0;
}

static int graph_validate_safetensors_affine_quantization(
        cJSON* root, const cJSON* inputs, const cJSON* nodes,
        const cJSON* graph_outputs) {
    cJSON* quantization;
    cJSON* descriptors;
    if (safetensors_affine_quantization_mode(root) != 1) return -1;
    for (int file_index = 0; file_index < g_weight_file_count; file_index++) {
        const SafetensorsFile* file = &g_weight_files[file_index];
        cJSON* metadata = NULL;
        if (!file->has_metadata) continue;
        if (!file->metadata_json ||
            !(metadata = cJSON_Parse(file->metadata_json)) ||
            !cJSON_IsObject(metadata)) {
            cJSON_Delete(metadata);
            return -1;
        }
        if (cJSON_GetObjectItemCaseSensitive(
                metadata, "weights_quantization") ||
            cJSON_GetObjectItemCaseSensitive(
                metadata, "weights_quantization_storage")) {
            cJSON_Delete(metadata);
            return -1;
        }
        cJSON_Delete(metadata);
    }
    quantization = cJSON_GetObjectItemCaseSensitive(root, "quantization");
    if (!quantization) return 0;
    descriptors = cJSON_GetObjectItemCaseSensitive(quantization, "tensors");
    if (!cJSON_IsObject(descriptors) || !descriptors->child) return -1;

    for (cJSON* descriptor = descriptors->child; descriptor;
         descriptor = descriptor->next) {
        int per_axis;
        int axis;
        const char* scale_name;
        const char* zero_name;
        T* target;
        if (!graph_json_name_valid(descriptor->string, sizeof(g_t[0].name)) ||
            safetensors_affine_descriptor_refs(
                descriptor, &per_axis, &axis, &scale_name, &zero_name) != 0 ||
            !graph_json_name_valid(scale_name, sizeof(g_t[0].name)) ||
            !graph_json_name_valid(zero_name, sizeof(g_t[0].name)) ||
            !strcmp(descriptor->string, scale_name) ||
            !strcmp(descriptor->string, zero_name) ||
            graph_safetensors_tensor_occurrences(scale_name) != 1 ||
            graph_safetensors_tensor_occurrences(zero_name) != 1 ||
            !volvoxai_engine_tensor_is_model_weight_locked(scale_name) ||
            !volvoxai_engine_tensor_is_model_weight_locked(zero_name) ||
            graph_inputs_declare(inputs, scale_name) ||
            graph_inputs_declare(inputs, zero_name) ||
            graph_nodes_declare_before(nodes, NULL, scale_name) ||
            graph_nodes_declare_before(nodes, NULL, zero_name) ||
            graph_outputs_name(graph_outputs, scale_name) ||
            graph_outputs_name(graph_outputs, zero_name)) return -1;

        for (cJSON* other = descriptors->child; other; other = other->next) {
            int other_per_axis;
            int other_axis;
            const char* other_scale;
            const char* other_zero;
            if (safetensors_affine_descriptor_refs(
                    other, &other_per_axis, &other_axis,
                    &other_scale, &other_zero) != 0 ||
                !strcmp(descriptor->string, other_scale) ||
                !strcmp(descriptor->string, other_zero)) return -1;
        }

        target = t_find(descriptor->string);
        if (!target && !graph_inputs_declare(inputs, descriptor->string) &&
            !graph_nodes_declare_before(nodes, NULL, descriptor->string)) return -1;
        if (per_axis) {
            /* Inputs/outputs are checked with their declared shape during
             * graph validation, before they have allocated T records. */
            if (target && validate_per_axis_quantization(
                    descriptor, target) != 0) return -1;
        } else if (target) {
            TensorQuantization parsed = {0};
            if (parse_safetensors_per_tensor_quantization(
                    root, descriptor->string, target->dtype, &parsed) != 0)
                return -1;
        }
    }

    for (const cJSON* node = cJSON_IsArray(nodes) ? nodes->child : NULL;
         node; node = node->next) {
        cJSON* op = cJSON_GetObjectItemCaseSensitive(node, "opType");
        int quantize;
        cJSON* node_inputs;
        cJSON* node_outputs;
        cJSON* target_item;
        cJSON* scale_item;
        cJSON* zero_item;
        cJSON* descriptor;
        int per_axis;
        int axis;
        const char* scale_name;
        const char* zero_name;
        if (!cJSON_IsString(op) || !op->valuestring) continue;
        VxOperatorKind kind = vx_operator_kind_from_name(op->valuestring);
        if (kind == VX_OP_QUANTIZE_LINEAR) quantize = 1;
        else if (kind == VX_OP_DEQUANTIZE_LINEAR) quantize = 0;
        else continue;
        node_inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
        node_outputs = cJSON_GetObjectItemCaseSensitive(node, "outputs");
        target_item = quantize
            ? cJSON_GetObjectItemCaseSensitive(node_outputs, "out")
            : cJSON_GetObjectItemCaseSensitive(node_inputs, "input");
        scale_item = cJSON_GetObjectItemCaseSensitive(node_inputs, "scale");
        zero_item = cJSON_GetObjectItemCaseSensitive(node_inputs, "zero_point");
        if (!cJSON_IsString(target_item) || !target_item->valuestring ||
            !cJSON_IsString(scale_item) || !scale_item->valuestring ||
            !cJSON_IsString(zero_item) || !zero_item->valuestring) return -1;
        descriptor = safetensors_affine_descriptor(
            root, target_item->valuestring);
        if (safetensors_affine_descriptor_refs(
                descriptor, &per_axis, &axis, &scale_name, &zero_name) != 0 ||
            per_axis || strcmp(scale_item->valuestring, scale_name) ||
            strcmp(zero_item->valuestring, zero_name)) return -1;
    }
    return 0;
}

static int graph_validate_document(cJSON* root, int* tensor_count_out,
                                   int* node_count_out) {
    if (!cJSON_IsObject(root)) return -1;
    cJSON* format = cJSON_GetObjectItemCaseSensitive(root, "format");
    if (!cJSON_IsString(format) || !format->valuestring ||
        strcmp(format->valuestring, "volvox-graph/v1") ||
        safetensors_affine_quantization_mode(root) != 1) return -1;
    cJSON* inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    cJSON* nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    cJSON* graph_outputs = cJSON_GetObjectItemCaseSensitive(root, "outputs");
    if ((inputs && !cJSON_IsObject(inputs)) || (nodes && !cJSON_IsArray(nodes)) ||
        !cJSON_IsArray(graph_outputs) || !graph_outputs->child) {
        return -1;
    }

    int projected_tensors = g_nt;
    for (cJSON* input = inputs ? inputs->child : NULL; input; input = input->next) {
        int shape[8];
        int ndim;
        long numel;
        int dtype = graph_input_dtype(input);
        TensorQuantization quantization = {0};
        if (!graph_json_name_valid(input->string, sizeof(g_t[0].name)) ||
            !cJSON_IsObject(input) || graph_object_has_prior_key(inputs, input, input->string) ||
            t_find(input->string) ||
            graph_parse_shape(cJSON_GetObjectItemCaseSensitive(input, "shape"), 1,
                              shape, &ndim, &numel) != 0 ||
            dtype < 0 || !dtype_size(dtype) ||
            (size_t)numel > SIZE_MAX / dtype_size(dtype) ||
            parse_graph_tensor_quantization(
                root, input->string,
                cJSON_GetObjectItemCaseSensitive(input, "quantization"),
                dtype, shape, ndim, &quantization) != 0 ||
            projected_tensors == INT_MAX) return -1;
        projected_tensors++;
    }

    int node_count = nodes ? cJSON_GetArraySize(nodes) : 0;
    if (node_count < 0) return -1;
    for (cJSON* node = nodes ? nodes->child : NULL; node; node = node->next) {
        if (!cJSON_IsObject(node)) return -1;
        cJSON* op = cJSON_GetObjectItemCaseSensitive(node, "opType");
        cJSON* node_inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
        cJSON* outputs = cJSON_GetObjectItemCaseSensitive(node, "outputs");
        cJSON* output_shapes =
            cJSON_GetObjectItemCaseSensitive(node, "outputs_shape");
        cJSON* output_dtypes =
            cJSON_GetObjectItemCaseSensitive(node, "outputs_dtype");
        cJSON* output_quantization =
            cJSON_GetObjectItemCaseSensitive(node, "outputs_quantization");
        cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
        if (!cJSON_IsString(op) ||
            vx_operator_kind_from_name(op->valuestring) == VX_OP_UNSPECIFIED ||
            cJSON_GetObjectItemCaseSensitive(node, "op") ||
            (node_inputs && !cJSON_IsObject(node_inputs)) ||
            !cJSON_IsObject(outputs) || !outputs->child ||
            !cJSON_IsObject(output_shapes) ||
            output_quantization ||
            graph_params_contain_affine_payload(params) ||
            graph_validate_output_metadata(outputs, output_shapes) != 0 ||
            graph_validate_output_metadata(outputs, output_dtypes) != 0 ||
            graph_validate_output_metadata(outputs, output_quantization) != 0) {
            return -1;
        }

        for (cJSON* input = node_inputs ? node_inputs->child : NULL; input; input = input->next) {
            if (!graph_json_name_valid(input->string, sizeof(g_n[0].ins[0].key)) ||
                graph_object_has_prior_key(node_inputs, input, input->string) ||
                !cJSON_IsString(input) ||
                !graph_json_name_valid(input->valuestring, sizeof(g_t[0].name)) ||
                !graph_tensor_declared_before(inputs, nodes, node, input->valuestring)) {
                return -1;
            }
        }

        for (cJSON* output = outputs->child; output; output = output->next) {
            int shape[8];
            int ndim;
            long numel;
            if (!graph_json_name_valid(output->string, sizeof(g_n[0].outs[0].key)) ||
                graph_object_has_prior_key(outputs, output, output->string) ||
                !cJSON_IsString(output) ||
                !graph_json_name_valid(output->valuestring, sizeof(g_t[0].name)) ||
                graph_tensor_declared_before(inputs, nodes, node, output->valuestring)) return -1;
            for (cJSON* prior = outputs->child; prior && prior != output; prior = prior->next) {
                if (cJSON_IsString(prior) && prior->valuestring &&
                    !strcmp(prior->valuestring, output->valuestring)) return -1;
            }
            cJSON* shape_item = output_shapes
                ? cJSON_GetObjectItemCaseSensitive(output_shapes,
                                                   output->string) : NULL;
            cJSON* dtype_item = output_dtypes
                ? cJSON_GetObjectItemCaseSensitive(output_dtypes,
                                                   output->string) : NULL;
            int dtype = graph_declared_dtype(dtype_item);
            TensorQuantization quantization = {0};
            if ((output_dtypes && !dtype_item) ||
                graph_parse_shape(shape_item, 1, shape, &ndim, &numel) != 0 ||
                dtype < 0 || !dtype_size(dtype) ||
                (size_t)numel > SIZE_MAX / dtype_size(dtype) ||
                parse_graph_tensor_quantization(
                    root, output->valuestring, NULL,
                    dtype, shape, ndim, &quantization) != 0 ||
                projected_tensors == INT_MAX) return -1;
            projected_tensors++;
        }
    }

    if (cJSON_IsArray(graph_outputs)) {
        for (cJSON* output = graph_outputs->child; output; output = output->next) {
            if (!cJSON_IsString(output) ||
                !graph_json_name_valid(output->valuestring, sizeof(g_t[0].name)) ||
                !graph_tensor_declared_before(inputs, nodes, NULL, output->valuestring)) return -1;
            for (cJSON* prior = graph_outputs->child; prior && prior != output;
                 prior = prior->next) {
                if (cJSON_IsString(prior) && prior->valuestring &&
                    !strcmp(prior->valuestring, output->valuestring)) return -1;
            }
        }
    }
    if (graph_validate_safetensors_affine_quantization(
            root, inputs, nodes, graph_outputs) != 0) return -1;
    if (tensor_count_out) *tensor_count_out = projected_tensors;
    if (node_count_out) *node_count_out = node_count;
    return 0;
}

static void graph_build_rollback(int tensor_count) {
    for (int index = 0; index < g_nn; index++) {
        vx_node_dispose(&g_n[index]);
    }
    for (int index = 0; index < g_nt; index++) {
        if (index >= tensor_count) {
            if (g_t[index].owns) free(g_t[index].data);
            memset(&g_t[index], 0, sizeof(g_t[index]));
        }
    }
    volvoxai_engine_clear_qlinear_metadata();
    volvoxai_engine_clear_qconv_metadata();
    volvoxai_engine_clear_qembedding_metadata();
    g_nt = tensor_count;
    volvoxai_engine_tensor_name_index_invalidate();
    g_nn = 0;
    g_first_input[0] = 0;
    if (g_graph_root) {
        cJSON_Delete(g_graph_root);
        g_graph_root = NULL;
    }
}

/* The graph reaches the engine as bytes on every target. Only naming it by
 * path is target-specific, so the path form is a thin wrapper and this is the
 * whole loader. The caller keeps ownership of the text. */
int build_graph_from_text(const char* graph_text, long size) {
    double t0 = g_debug ? volvoxai_engine_now_ms() : 0.0;
    g_removed_graph_output_count = 0;
    graph_opt_reset_state();
    if (g_graph_root) {
        cJSON_Delete(g_graph_root);
        g_graph_root = NULL;
    }
    if (!graph_text) return -1;
    if (size < 0 || vx_json_text_contains_decoded_nul(
            (const unsigned char*)graph_text, (size_t)size)) {
        vx_engine_log("[engine] invalid graph.json graph string vocabulary\n");
        return -1;
    }
    cJSON* root = cJSON_Parse(graph_text);
    int projected_tensor_count = 0;
    int projected_node_count = 0;
    VxEngineResult reserve_status;
    if (!root || !vx_json_object_keys_unique_recursive(root) ||
        graph_validate_document(root, &projected_tensor_count,
                                &projected_node_count) != 0) {
        vx_engine_log("[engine] invalid graph.json graph contract\n");
        cJSON_Delete(root);
        return -1;
    }
    reserve_status = vx_engine_state_reserve_graph_metadata(
        vx_engine_state_current(), (size_t)projected_tensor_count,
        (size_t)projected_node_count);
    if (reserve_status != VX_ENGINE_RESULT_OK) {
        cJSON_Delete(root);
        return reserve_status;
    }
    volvoxai_engine_tensor_name_index_rebuild();
    int initial_tensor_count = g_nt;
    g_graph_root = root;
    if (apply_weight_quantization(root) != 0) {
        vx_engine_log("[engine] invalid safetensors quantization metadata\n");
        goto fail;
    }

    // Graph inputs (allocated, zero-filled; the caller pokes them before forward()).
    cJSON* inputs = cJSON_GetObjectItem(root, "inputs");
    if (inputs) for (cJSON* it = inputs->child; it; it = it->next) {
        cJSON* shape = cJSON_GetObjectItem(it, "shape");
        int ndim; int sh[8]; long numel;
        int dtype = graph_input_dtype(it);
        if (graph_parse_shape(shape, 1, sh, &ndim, &numel) != 0) goto fail;
        T* input = t_add_ex(it->string, sh, ndim, NULL, dtype);
        if (!input || !input->data) goto fail;
        if (parse_graph_tensor_quantization(
                root, it->string,
                cJSON_GetObjectItemCaseSensitive(it, "quantization"), dtype,
                sh, ndim, &input->quantization) != 0) goto fail;
        input->is_graph_input = 1;
        if (!g_first_input[0]) { strncpy(g_first_input, it->string, 127); g_first_input[127] = 0; }
    }

    // Nodes.
    cJSON* nodes = cJSON_GetObjectItem(root, "nodes");
    for (cJSON* nd = nodes ? nodes->child : NULL; nd; nd = nd->next) {
        Node* n = &g_n[g_nn++];
        vx_node_dispose(n);
        cJSON* op = cJSON_GetObjectItem(nd, "opType");
        n->operator_kind = vx_operator_kind_from_name(op->valuestring);
        if (n->operator_kind == VX_OP_UNSPECIFIED) goto fail;
        n->params = cJSON_GetObjectItem(nd, "params");  // borrowed (g_graph_root kept alive)
        n->nin = 0; n->nout = 0; n->owns_params = 0; n->disabled = 0; n->fuse_relu6 = 0; n->skip = 0;
        cJSON* nins = cJSON_GetObjectItem(nd, "inputs");
        if (vx_node_reserve_refs(n, cJSON_GetArraySize(nins),
                cJSON_GetArraySize(cJSON_GetObjectItem(nd, "outputs")))) goto fail;
        if (nins) for (cJSON* it = nins->child; it; it = it->next) {
            memcpy(n->ins[n->nin].key, it->string, strlen(it->string) + 1);
            n->ins[n->nin].port = vx_port_kind_from_name(it->string);
            memcpy(n->ins[n->nin].name, it->valuestring, strlen(it->valuestring) + 1);
            n->nin++;
        }
        // Create output tensor(s) from outputs_shape; keep the first as the node output.
        cJSON* outs = cJSON_GetObjectItem(nd, "outputs");
        cJSON* oshape = cJSON_GetObjectItem(nd, "outputs_shape");
        cJSON* odtype = cJSON_GetObjectItem(nd, "outputs_dtype");
        n->out[0] = 0;
        if (outs) for (cJSON* it = outs->child; it; it = it->next) {
            const char* tname = it->valuestring;
            cJSON* sh = oshape ? cJSON_GetObjectItem(oshape, it->string) : NULL;
            cJSON* dtype_item = odtype ? cJSON_GetObjectItem(odtype, it->string) : NULL;
            int ndim; int shp[8]; long numel;
            int dtype = graph_declared_dtype(dtype_item);
            if (graph_parse_shape(sh, 1, shp, &ndim, &numel) != 0) goto fail;
            T* output = t_add_ex(tname, shp, ndim, NULL, dtype);
            if (!output || parse_graph_tensor_quantization(
                    root, tname, NULL, dtype,
                    shp, ndim, &output->quantization) != 0) goto fail;
            memcpy(n->outs[n->nout].key, it->string, strlen(it->string) + 1);
            n->outs[n->nout].port = vx_port_kind_from_name(it->string);
            memcpy(n->outs[n->nout].name, tname, strlen(tname) + 1);
            n->nout++;
            if (!n->out[0]) { strncpy(n->out, tname, 127); n->out[127] = 0; }
        }
        if (node_parse_params(n) != 0) goto fail;
    }
    GraphOptStats opt = {0};
    if (!vx_engine_state_current()->shape_policy.retain_activations)
        opt = graph_optimize_operator_fusion();
    for (int i = 0; i < g_nn; i++)
        if (node_parse_params(&g_n[i]) != 0) goto fail;
    if (build_quant_params_pass() != 0) goto fail;
    if (g_debug) {
        vx_engine_log("[debug] build_graph tensors=%d nodes=%d bytes=%ld %.3f ms\n",
                g_nt, g_nn, size, volvoxai_engine_now_ms() - t0);
        vx_engine_log("[debug] graph_opt operator_fusion relu6=%d conv_add=%d depthwise_pointwise=%d chained_add=%d concat_sigmoid=%d alias=%d skipped=%d\n",
                opt.relu6, opt.conv_add, opt.depthwise_pointwise, opt.chained_add,
                opt.concat_sigmoid, opt.alias, opt.skipped);
    }
    volvoxai_engine_finish_tensor_table_mutation();
    return 0;
fail:
    graph_build_rollback(initial_tensor_count);
    return -1;
}

/* Native builds resolve an OS path; WASM resolves a host-mounted VFS path. */
int build_graph(const char* graph_path) {
    long size = 0;
    char* graph_text = read_file(graph_path, &size);
    int status;
    if (!graph_text) return -1;
    status = build_graph_from_text(graph_text, size);
    free(graph_text);
    return status;
}

static const char* volvoxai_engine_dtype_name(int dtype) {
    switch (dtype) {
        case T_F32: return "F32";
        case T_I8: return "I8";
        case T_U8: return "U8";
        case T_I32: return "I32";
        case T_F16: return "F16";
        default: return "UNKNOWN";
    }
}

static cJSON* shape_to_json(const int* shape, int ndim) {
    cJSON* arr = cJSON_CreateArray();
    if (!arr) return NULL;
    for (int i = 0; i < ndim; i++) cJSON_AddItemToArray(arr, cJSON_CreateNumber(shape[i]));
    return arr;
}

static cJSON* tensor_to_json(const T* t) {
    cJSON* obj = cJSON_CreateObject();
    if (!obj) return NULL;
    cJSON_AddStringToObject(obj, "name", t->name);
    cJSON_AddStringToObject(obj, "dtype", volvoxai_engine_dtype_name(t->dtype));
    cJSON_AddNumberToObject(obj, "size_bytes", (double)(t->numel * (long)t->elem_size));
    cJSON_AddBoolToObject(obj, "owns", t->owns ? 1 : 0);
    cJSON* shape = shape_to_json(t->shape, t->ndim);
    if (shape) cJSON_AddItemToObject(obj, "shape", shape);
    return obj;
}

static cJSON* refs_to_json(const Ref* refs, int count) {
    cJSON* obj = cJSON_CreateObject();
    if (!obj) return NULL;
    for (int i = 0; i < count; i++) cJSON_AddStringToObject(obj, refs[i].key, refs[i].name);
    return obj;
}

static cJSON* node_output_shapes_json(const Node* n) {
    cJSON* obj = cJSON_CreateObject();
    if (!obj) return NULL;
    for (int i = 0; i < n->nout; i++) {
        T* t = t_find(n->outs[i].name);
        if (!t) continue;
        cJSON* shape = shape_to_json(t->shape, t->ndim);
        if (shape) cJSON_AddItemToObject(obj, n->outs[i].key, shape);
    }
    return obj;
}

static const char* graph_dtype_name(int dtype) {
    switch (dtype) {
        case T_F32: return "float32";
        case T_I8: return "int8";
        case T_U8: return "uint8";
        case T_I32: return "int32";
        case T_F16: return "float16";
        default: return NULL;
    }
}

static cJSON* node_output_dtypes_json(const Node* n) {
    cJSON* obj = cJSON_CreateObject();
    if (!obj) return NULL;
    for (int i = 0; i < n->nout; i++) {
        T* tensor = t_find(n->outs[i].name);
        const char* dtype = tensor ? graph_dtype_name(tensor->dtype) : NULL;
        if (!dtype) {
            cJSON_Delete(obj);
            return NULL;
        }
        cJSON_AddStringToObject(obj, n->outs[i].key, dtype);
    }
    return obj;
}

static cJSON* node_to_json(const Node* n, int index, int include_params) {
    cJSON* obj = cJSON_CreateObject();
    if (!obj) return NULL;
    cJSON_AddNumberToObject(obj, "index", index);
    cJSON_AddStringToObject(obj, "id", "");
    cJSON_AddStringToObject(obj, "op", vx_operator_kind_name(n->operator_kind));
    cJSON_AddBoolToObject(obj, "disabled", n->disabled ? 1 : 0);
    cJSON* inputs = refs_to_json(n->ins, n->nin);
    cJSON* outputs = refs_to_json(n->outs, n->nout);
    cJSON* output_shapes = node_output_shapes_json(n);
    if (inputs) cJSON_AddItemToObject(obj, "inputs", inputs);
    if (outputs) cJSON_AddItemToObject(obj, "outputs", outputs);
    if (output_shapes) cJSON_AddItemToObject(obj, "output_shapes", output_shapes);
    if (include_params && n->params) {
        cJSON* params = cJSON_Duplicate(n->params, 1);
        if (params) cJSON_AddItemToObject(obj, "params", params);
    }
    return obj;
}

static cJSON* safetensors_file_to_json(int index, const SafetensorsFile* file) {
    cJSON* obj = cJSON_CreateObject();
    cJSON* tensors = cJSON_CreateArray();
    if (!obj || !tensors) {
        cJSON_Delete(obj);
        cJSON_Delete(tensors);
        return NULL;
    }
    cJSON_AddNumberToObject(obj, "index", index);
    cJSON_AddStringToObject(obj, "path", g_weight_paths[index]);
    cJSON_AddNumberToObject(obj, "size_bytes", (double)file->size);
    if (file->has_metadata && file->metadata_json) {
        cJSON* md = cJSON_Parse(file->metadata_json);
        if (md) cJSON_AddItemToObject(obj, "metadata", md);
    }
    for (int i = 0; i < file->tensor_count; i++) {
        const SafetensorsTensor* t = &file->tensors[i];
        cJSON* tj = cJSON_CreateObject();
        if (!tj) continue;
        cJSON_AddStringToObject(tj, "name", t->name);
        cJSON_AddStringToObject(tj, "dtype", safetensors_dtype_name(t->dtype));
        cJSON_AddNumberToObject(tj, "data_start", (double)t->data_start);
        cJSON_AddNumberToObject(tj, "data_end", (double)t->data_end);
        cJSON_AddNumberToObject(tj, "access_flags", (double)t->flags);
        cJSON* shape = shape_to_json(t->shape, t->ndim);
        if (shape) cJSON_AddItemToObject(tj, "shape", shape);
        cJSON_AddItemToArray(tensors, tj);
    }
    cJSON_AddItemToObject(obj, "tensors", tensors);
    return obj;
}

char* volvoxai_engine_inspect_model_json(int include_graph, int include_tensors, int include_weight_files, int include_params) {
    cJSON* root = cJSON_CreateObject();
    if (!root) return NULL;
    int visible_tensor_count = 0;
    for (int i = 0; i < g_nt; i++) {
        if (removed_graph_output_index(g_t[i].name) < 0 &&
            !volvoxai_engine_tensor_is_quantization_parameter_locked(g_t[i].name)) {
            visible_tensor_count++;
        }
    }
    cJSON_AddNumberToObject(root, "num_ops", g_nn);
    cJSON_AddNumberToObject(root, "num_tensors", visible_tensor_count);
    if (include_tensors) {
        cJSON* tensors = cJSON_CreateArray();
        if (tensors) {
            for (int i = 0; i < g_nt; i++) {
                if (removed_graph_output_index(g_t[i].name) >= 0 ||
                    volvoxai_engine_tensor_is_quantization_parameter_locked(g_t[i].name)) continue;
                cJSON* tj = tensor_to_json(&g_t[i]);
                if (tj) cJSON_AddItemToArray(tensors, tj);
            }
            cJSON_AddItemToObject(root, "tensors", tensors);
        }
    }
    if (include_graph) {
        cJSON* nodes = cJSON_CreateArray();
        if (nodes) {
            for (int i = 0; i < g_nn; i++) {
                cJSON* nj = node_to_json(&g_n[i], i, include_params);
                if (nj) cJSON_AddItemToArray(nodes, nj);
            }
            cJSON_AddItemToObject(root, "nodes", nodes);
        }
    }
    if (include_weight_files) {
        cJSON* weights = cJSON_CreateArray();
        if (weights) {
            for (int i = 0; i < g_weight_file_count; i++) {
                cJSON* wj = safetensors_file_to_json(i, &g_weight_files[i]);
                if (wj) cJSON_AddItemToArray(weights, wj);
            }
            cJSON_AddItemToObject(root, "weights", weights);
        }
    }
    char* out = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return out;
}

static int free_node_caches_impl(int training_transition) {
    int preserve_cuda_weights = 0;
#if VOLVOXAI_ENABLE_TRAINING && VOLVOXAI_ENABLE_CUDA
    preserve_cuda_weights = training_transition && g_use_cuda;
#else
    (void)training_transition;
#endif
    /* A full cache reset discards backend slots. Materialize any optimizer-
     * updated model weights first so graph patches, CPU handoff, and cache
     * rebuilds cannot silently restore stale Safetensors bytes. */
    if (!preserve_cuda_weights && g_loaded &&
        volvoxai_engine_sync_model_weights_locked() != 0) return -1;
#if VOLVOXAI_ENABLE_TRAINING
    if (training_transition ? volvoxai_engine_prepare_training_transition() != 0 :
        volvoxai_engine_prepare_tensor_table_mutation() != 0) return -1;
#else
    if (volvoxai_engine_prepare_tensor_table_mutation() != 0) return -1;
#endif
    vx_conv_f32_isa_free_all();
    vx_gemm_f32_cache_free_all(&vx_engine_state_current()->gemm_f32_cache);
#if VOLVOXAI_ENABLE_VULKAN
    vk_free_weight_cache();
#endif
#if VOLVOXAI_ENABLE_OPENGL
    opengl_free_weight_cache();
#endif
    if (preserve_cuda_weights)
        vx_runtime_backend_reset_transients();
    else
        vx_runtime_backend_reset();
    for (size_t i = 0; i < g_node_capacity; i++) {
        free(g_kcache[i]); g_kcache[i] = NULL;
        free(g_vcache[i]); g_vcache[i] = NULL;
        free(g_qwcache[i]); g_qwcache[i] = NULL;
        free(g_f16wcache[i]); g_f16wcache[i] = NULL;
        free(g_f16bcache[i]); g_f16bcache[i] = NULL;
        free(g_conv_wcache[i]); g_conv_wcache[i] = NULL;
        free(g_q8wcache[i]); g_q8wcache[i] = NULL; g_q8wcache_bytes[i] = 0;
    }
    return 0;
}

static int free_node_caches(void) {
    return free_node_caches_impl(0);
}

#if VOLVOXAI_ENABLE_TRAINING
static int free_node_caches_for_training(void) {
    return free_node_caches_impl(1);
}
#endif

int volvoxai_engine_refresh_weight_caches(void) {
    if (vx_dynamic_autograd_active_locked()) return -1;
    if (g_loaded && volvoxai_engine_sync_model_weights_locked() != 0)
        return -1;
    vx_conv_f32_isa_free_all();
    vx_gemm_f32_cache_free_all(&vx_engine_state_current()->gemm_f32_cache);
#if VOLVOXAI_ENABLE_VULKAN
    vk_free_weight_cache();
#endif
#if VOLVOXAI_ENABLE_OPENGL
    opengl_free_weight_cache();
#endif
    vx_runtime_backend_reset();
    for (size_t i = 0; i < g_node_capacity; i++) {
        free(g_qwcache[i]); g_qwcache[i] = NULL;
        free(g_f16wcache[i]); g_f16wcache[i] = NULL;
        free(g_f16bcache[i]); g_f16bcache[i] = NULL;
        free(g_conv_wcache[i]); g_conv_wcache[i] = NULL;
        free(g_q8wcache[i]); g_q8wcache[i] = NULL; g_q8wcache_bytes[i] = 0;
    }
    if (build_quant_params_pass() != 0) return -1;
    if (prepack_cpu_weights() != 0) return -1;
    g_weight_caches_dirty = 0;
    return 0;
}

static int rebuild_after_graph_patch_ex(int reoptimize) {
    if (free_node_caches() != 0) return -1;
    graph_opt_reset_state();
    for (int i = 0; i < g_nn; i++) {
        g_n[i].skip = g_n[i].disabled ? 1 : 0;
        g_n[i].fuse_relu6 = 0;
    }
    /* Patches can mutate params in place. Recompile every typed descriptor
     * before fusion inspects it so no optimization decision observes the
     * pre-patch cache. Fusion may then rewrite node metadata, matching the
     * initial graph-build lifecycle, so compile once more afterward. */
    for (int i = 0; i < g_nn; i++)
        if (node_parse_params(&g_n[i]) != 0) return -1;
    if (reoptimize && !vx_engine_state_current()->shape_policy.retain_activations)
        graph_optimize_operator_fusion();
    for (int i = 0; i < g_nn; i++)
        if (node_parse_params(&g_n[i]) != 0) return -1;
    if (build_quant_params_pass() != 0) return -1;
    volvoxai_engine_finish_tensor_table_mutation();
    if (prepack_cpu_weights() != 0) return -1;
    g_weight_caches_dirty = 0;
    return 0;
}

static void clear_owned_params(Node* n) {
    if (n->owns_params && n->params) cJSON_Delete(n->params);
    n->params = NULL;
    n->owns_params = 0;
    vx_node_clear_param_cache(n);
}

static int shape_from_json(cJSON* arr, int* shape, int* ndim) {
    if (!cJSON_IsArray(arr)) return -1;
    int n = cJSON_GetArraySize(arr);
    if (n < 0 || n > 8) return -1;
    for (int i = 0; i < n; i++) {
        cJSON* d = cJSON_GetArrayItem(arr, i);
        if (!cJSON_IsNumber(d) || d->valueint < 0) return -1;
        shape[i] = d->valueint;
    }
    *ndim = n;
    return 0;
}

static void graph_patch_reuse_begin(void) {
    g_graph_patch_reused_count = 0;
}

static void graph_patch_reuse_rollback(void) {
    for (int i = g_graph_patch_reused_count - 1; i >= 0; i--) {
        int index = g_graph_patch_reused_indices[i];
        if (index < 0 || index >= g_nt) continue;
        if (g_t[index].owns && g_t[index].data &&
            g_t[index].data != g_graph_patch_reused_old_tensors[i].data) {
            free(g_t[index].data);
        }
        g_t[index] = g_graph_patch_reused_old_tensors[i];
    }
    if (g_graph_patch_reused_count > 0)
        volvoxai_engine_tensor_name_index_invalidate();
    g_graph_patch_reused_count = 0;
}

static void graph_patch_reuse_commit(void) {
    for (int i = 0; i < g_graph_patch_reused_count; i++) {
        T* old = &g_graph_patch_reused_old_tensors[i];
        int index = g_graph_patch_reused_indices[i];
        if (old->owns && old->data && index >= 0 && index < g_nt && old->data != g_t[index].data) {
            free(old->data);
        }
    }
    g_graph_patch_reused_count = 0;
}

static int graph_patch_tensor_was_reused(int index) {
    for (int i = 0; i < g_graph_patch_reused_count; i++) {
        if (g_graph_patch_reused_indices[i] == index) return 1;
    }
    return 0;
}

static T* graph_patch_add_output_tensor(const char* name, const int* shape, int ndim) {
    for (int removed = 0; removed < g_removed_graph_output_count; removed++) {
        T* candidate = t_find(g_removed_graph_outputs[removed]);
        if (!candidate || candidate->is_graph_input ||
            volvoxai_engine_tensor_weight_file_index(candidate->name) >= 0) continue;
        int referenced = 0;
        for (int node = 0; node < g_nn && !referenced; node++) {
            for (int input = 0; input < g_n[node].nin; input++) {
                if (!strcmp(g_n[node].ins[input].name, candidate->name)) { referenced = 1; break; }
            }
            for (int output = 0; output < g_n[node].nout && !referenced; output++) {
                if (!strcmp(g_n[node].outs[output].name, candidate->name)) { referenced = 1; break; }
            }
            if (!referenced && !strcmp(g_n[node].out, candidate->name)) referenced = 1;
        }
        if (referenced ||
            (size_t)g_graph_patch_reused_count >= g_tensor_capacity) continue;
        long numel = 1;
        for (int d = 0; d < ndim; d++) numel *= shape[d];
        float* data = (float*)calloc(numel > 0 ? (size_t)numel : 1u, sizeof(float));
        if (!data) return NULL;
        int index = (int)(candidate - g_t);
        g_graph_patch_reused_indices[g_graph_patch_reused_count] = index;
        g_graph_patch_reused_old_tensors[g_graph_patch_reused_count] = *candidate;
        g_graph_patch_reused_count++;
        char old_name[128];
        strncpy(old_name, candidate->name, sizeof(old_name) - 1);
        old_name[sizeof(old_name) - 1] = 0;
        memset(candidate, 0, sizeof(*candidate));
        strncpy(candidate->name, name, sizeof(candidate->name) - 1);
        candidate->ndim = ndim;
        for (int d = 0; d < ndim; d++) candidate->shape[d] = shape[d];
        candidate->numel = numel;
        candidate->data = data;
        candidate->owns = 1;
        candidate->dtype = T_F32;
        candidate->elem_size = sizeof(float);
        volvoxai_engine_tensor_name_index_invalidate();
        removed_graph_output_remove(old_name);
        return candidate;
    }
    if (g_nt == INT_MAX || vx_engine_state_reserve_graph_metadata(
            vx_engine_state_current(), (size_t)g_nt + 1u,
            g_node_capacity) != 0) return NULL;
    return t_add(name, shape, ndim, NULL);
}

static int set_ref(Node* node, int output, const char* key, const char* name) {
    Ref* refs = output ? node->outs : node->ins;
    int* count = output ? &node->nout : &node->nin;
    if (!key || !name) return -1;
    for (int i = 0; i < *count; i++) {
        if (strcmp(refs[i].key, key) == 0) {
            strncpy(refs[i].name, name, sizeof(refs[i].name) - 1);
            refs[i].name[sizeof(refs[i].name) - 1] = 0;
            return 0;
        }
    }
    if (*count == INT_MAX || vx_node_reserve_refs(node,
            node->nin + !output, node->nout + output)) return -1;
    refs = output ? node->outs : node->ins;
    strncpy(refs[*count].key, key, sizeof(refs[*count].key) - 1);
    refs[*count].key[sizeof(refs[*count].key) - 1] = 0;
    refs[*count].port = vx_port_kind_from_name(key);
    strncpy(refs[*count].name, name, sizeof(refs[*count].name) - 1);
    refs[*count].name[sizeof(refs[*count].name) - 1] = 0;
    (*count)++;
    return 0;
}

static int apply_node_patch(Node* n, cJSON* patch, int replace) {
    if (replace) {
        vx_node_dispose(n);
    }
    cJSON* op = cJSON_GetObjectItem(patch, "op");
    if (!op) op = cJSON_GetObjectItem(patch, "opType");
    if (op) {
        if (!cJSON_IsString(op)) return -1;
        VxOperatorKind kind = vx_operator_kind_from_name(op->valuestring);
        if (kind == VX_OP_UNSPECIFIED) return -1;
        n->operator_kind = kind;
    } else if (replace || n->operator_kind == VX_OP_UNSPECIFIED) {
        return -1;
    }

    cJSON* disabled = cJSON_GetObjectItem(patch, "disabled");
    if (cJSON_IsBool(disabled)) n->disabled = cJSON_IsTrue(disabled) ? 1 : 0;

    cJSON* inputs = cJSON_GetObjectItem(patch, "inputs");
    if (inputs) {
        if (!cJSON_IsObject(inputs)) return -1;
        if (replace) n->nin = 0;
        for (cJSON* it = inputs->child; it; it = it->next) {
            if (!cJSON_IsString(it) || !t_find(it->valuestring) ||
                volvoxai_engine_tensor_is_quantization_parameter_locked(
                    it->valuestring)) return -1;
            if (set_ref(n, 0, it->string, it->valuestring) != 0) return -1;
        }
    }

    cJSON* output_shapes = cJSON_GetObjectItem(patch, "output_shapes");
    cJSON* outputs = cJSON_GetObjectItem(patch, "outputs");
    if (outputs) {
        if (!cJSON_IsObject(outputs)) return -1;
        if (replace) {
            n->nout = 0;
            n->out[0] = 0;
        }
        for (cJSON* it = outputs->child; it; it = it->next) {
            if (!cJSON_IsString(it)) return -1;
            T* existing = t_find(it->valuestring);
            if (existing && volvoxai_engine_tensor_is_quantization_parameter_locked(
                                it->valuestring))
                return -1;
            if (!existing) {
                cJSON* sh = output_shapes ? cJSON_GetObjectItem(output_shapes, it->string) : NULL;
                int shape[8], ndim = 0;
                if (shape_from_json(sh, shape, &ndim) != 0) return -1;
                existing = graph_patch_add_output_tensor(it->valuestring, shape, ndim);
                if (!existing) return -1;
            }
            if (set_ref(n, 1, it->string, it->valuestring) != 0) return -1;
            if (!n->out[0]) {
                strncpy(n->out, it->valuestring, sizeof(n->out) - 1);
                n->out[sizeof(n->out) - 1] = 0;
            }
        }
    }

    cJSON* params = cJSON_GetObjectItem(patch, "params");
    cJSON* params_json = cJSON_GetObjectItem(patch, "params_json");
    if (params || params_json) {
        cJSON* next = NULL;
        if (params) {
            if (!cJSON_IsObject(params)) return -1;
            next = cJSON_Duplicate(params, 1);
        } else if (cJSON_IsString(params_json)) {
            if (!params_json->valuestring || vx_json_text_contains_decoded_nul(
                    (const unsigned char*)params_json->valuestring,
                    strlen(params_json->valuestring))) return -1;
            next = cJSON_Parse(params_json->valuestring);
        }
        if (!next) return -1;
        if (!replace && n->params && params) {
            for (cJSON* it = params->child; it; it = it->next) {
                cJSON_DeleteItemFromObject(n->params, it->string);
                cJSON_AddItemToObject(n->params, it->string, cJSON_Duplicate(it, 1));
            }
            cJSON_Delete(next);
        } else {
            clear_owned_params(n);
            n->params = next;
            n->owns_params = 1;
        }
    } else if (!n->params) {
        n->params = cJSON_CreateObject();
        n->owns_params = n->params ? 1 : 0;
    }
    /* MERGE mutates the owned JSON object in place, so pointer equality cannot
     * be used as a cache key. Rebuild reparses every node after fusion. */
    n->parsed_params.parsed = 0;
    return 0;
}

static int volvoxai_engine_patch_node_json_impl(int node_index, const char* patch_json,
                                                VolvoxAIEngineNodePatchMode mode,
                                                int rebuild_execution_plan) {
    if (!patch_json || vx_json_text_contains_decoded_nul(
            (const unsigned char*)patch_json, strlen(patch_json))) return -1;
    cJSON* patch = cJSON_Parse(patch_json);
    if (!patch) return -1;
    int rc = -1;
    size_t planned_tensor_capacity = (size_t)g_nt;
    size_t planned_node_capacity = (size_t)g_nn;
    cJSON* planned_outputs = cJSON_GetObjectItemCaseSensitive(patch, "outputs");
    if (planned_outputs && cJSON_IsObject(planned_outputs)) {
        for (cJSON* output = planned_outputs->child; output;
             output = output->next) {
            if (cJSON_IsString(output) && output->valuestring &&
                !t_find(output->valuestring)) {
                if (planned_tensor_capacity == (size_t)INT_MAX) goto done;
                planned_tensor_capacity++;
            }
        }
    }
    if ((mode == VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_BEFORE ||
         mode == VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_AFTER)) {
        if (planned_node_capacity == (size_t)INT_MAX) goto done;
        planned_node_capacity++;
    }
    /* Plan every metadata move before taking a live Node/T address. A later
     * patch failure may retain the larger allocation, but it cannot leave a
     * partially moved table or invalidate a pointer while apply_node_patch is
     * using it. */
    if (vx_engine_state_reserve_graph_metadata(
            vx_engine_state_current(), planned_tensor_capacity,
            planned_node_capacity) != 0) goto done;
    volvoxai_engine_tensor_name_index_rebuild();
    if (mode == VOLVOXAI_ENGINE_NODE_PATCH_MODE_DELETE) {
        if (node_index >= 0 && node_index < g_nn) {
            vx_node_dispose(&g_n[node_index]);
            memmove(&g_n[node_index], &g_n[node_index + 1],
                    (size_t)(g_nn - node_index - 1) * sizeof(g_n[0]));
            memset(&g_n[g_nn - 1], 0, sizeof(g_n[0]));
            g_nn--;
            rc = 0;
        }
    } else if (mode == VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_BEFORE ||
               mode == VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_AFTER) {
        /* -1 is the builder-friendly append sentinel, including an empty graph. */
        if (((node_index == -1) || (node_index >= 0 && node_index < g_nn)) &&
            g_nn < INT_MAX && vx_engine_state_reserve_graph_metadata(
                vx_engine_state_current(), g_tensor_capacity,
                (size_t)g_nn + 1u) == 0) {
            int pos = node_index == -1
                          ? g_nn
                          : mode == VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_BEFORE
                                ? node_index
                                : node_index + 1;
            memmove(&g_n[pos + 1], &g_n[pos], (size_t)(g_nn - pos) * sizeof(g_n[0]));
            memset(&g_n[pos], 0, sizeof(g_n[pos]));
            g_nn++;
            rc = apply_node_patch(&g_n[pos], patch, 1);
            if (rc != 0) {
                vx_node_dispose(&g_n[pos]);
                memmove(&g_n[pos], &g_n[pos + 1], (size_t)(g_nn - pos - 1) * sizeof(g_n[0]));
                memset(&g_n[g_nn - 1], 0, sizeof(g_n[0]));
                g_nn--;
            }
        }
    } else if (mode == VOLVOXAI_ENGINE_NODE_PATCH_MODE_MERGE ||
               mode == VOLVOXAI_ENGINE_NODE_PATCH_MODE_REPLACE) {
        if (node_index >= 0 && node_index < g_nn) {
            rc = apply_node_patch(&g_n[node_index], patch,
                                  mode == VOLVOXAI_ENGINE_NODE_PATCH_MODE_REPLACE);
        }
    }
done:
    cJSON_Delete(patch);
    (void)rebuild_execution_plan;
    return rc;
}

typedef struct {
    Node* nodes;
    int node_count;
    int tensor_count;
    cJSON* graph_root;
    GraphNodeFusion* fusions;
    GraphOptStats graph_opt_stats;
    int* concat_sigmoid_fuse;
    char (*removed_outputs)[128];
    int removed_output_count;
} GraphPatchSnapshot;

static void graph_patch_snapshot_free(GraphPatchSnapshot* snapshot);

static int graph_node_output_count(const Node* node) {
    if (!node) return 0;
    return node->nout > 0 ? node->nout : (node->out[0] ? 1 : 0);
}

static const char* graph_node_output_name(const Node* node, int index) {
    if (!node || index < 0) return NULL;
    if (node->nout > 0) return index < node->nout ? node->outs[index].name : NULL;
    return index == 0 && node->out[0] ? node->out : NULL;
}

static int graph_nodes_producer(const Node* nodes, int count, const char* name) {
    int producer = -1;
    if (!nodes || !name || !name[0]) return -1;
    for (int i = 0; i < count; i++) {
        if (nodes[i].disabled) continue;
        int outputs = graph_node_output_count(&nodes[i]);
        for (int j = 0; j < outputs; j++) {
            const char* output = graph_node_output_name(&nodes[i], j);
            if (output && !strcmp(output, name)) {
                if (producer >= 0 && producer != i) return -2;
                producer = i;
            }
        }
    }
    return producer;
}

static int graph_patch_snapshot_take(GraphPatchSnapshot* snapshot) {
    if (!snapshot) return -1;
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->node_count = g_nn;
    snapshot->tensor_count = g_nt;
    snapshot->nodes = g_nn > 0 ? (Node*)calloc((size_t)g_nn, sizeof(Node)) : NULL;
    snapshot->fusions = g_nn > 0
        ? (GraphNodeFusion*)malloc((size_t)g_nn * sizeof(g_node_fusion[0])) : NULL;
    snapshot->concat_sigmoid_fuse = g_nn > 0
        ? (int*)malloc((size_t)g_nn * sizeof(g_concat_sigmoid_fuse[0])) : NULL;
    snapshot->removed_outputs = g_removed_graph_output_count > 0
        ? (char (*)[128])malloc((size_t)g_removed_graph_output_count *
                               sizeof(g_removed_graph_outputs[0])) : NULL;
    if ((g_nn > 0 && (!snapshot->nodes || !snapshot->fusions ||
                      !snapshot->concat_sigmoid_fuse)) ||
        (g_removed_graph_output_count > 0 && !snapshot->removed_outputs)) goto fail;
    for (int i = 0; i < g_nn; i++) {
        snapshot->nodes[i] = g_n[i];
        snapshot->nodes[i].ins = NULL; snapshot->nodes[i].outs = NULL;
        snapshot->nodes[i].input_capacity = snapshot->nodes[i].output_capacity = 0;
        memset(&snapshot->nodes[i].parsed_params, 0, sizeof(VxNodeParams));
        snapshot->nodes[i].params = NULL; snapshot->nodes[i].owns_params = 0;
        if (vx_node_reserve_refs(&snapshot->nodes[i], g_n[i].nin, g_n[i].nout)) goto fail;
        if (g_n[i].nin) memcpy(snapshot->nodes[i].ins, g_n[i].ins, (size_t)g_n[i].nin * sizeof(Ref));
        if (g_n[i].nout) memcpy(snapshot->nodes[i].outs, g_n[i].outs, (size_t)g_n[i].nout * sizeof(Ref));
        snapshot->nodes[i].params = g_n[i].params ? cJSON_Duplicate(g_n[i].params, 1) : NULL;
        snapshot->nodes[i].owns_params = snapshot->nodes[i].params ? 1 : 0;
        if (g_n[i].params && !snapshot->nodes[i].params) goto fail;
        if (node_parse_params(&snapshot->nodes[i])) goto fail;
    }
    snapshot->graph_root = g_graph_root ? cJSON_Duplicate(g_graph_root, 1) : NULL;
    if (g_graph_root && !snapshot->graph_root) goto fail;
    if (g_nn > 0)
        memcpy(snapshot->fusions, g_node_fusion,
               (size_t)g_nn * sizeof(g_node_fusion[0]));
    snapshot->graph_opt_stats = g_graph_opt_stats;
    if (g_nn > 0)
        memcpy(snapshot->concat_sigmoid_fuse, g_concat_sigmoid_fuse,
               (size_t)g_nn * sizeof(g_concat_sigmoid_fuse[0]));
    if (g_removed_graph_output_count > 0)
        memcpy(snapshot->removed_outputs, g_removed_graph_outputs,
               (size_t)g_removed_graph_output_count *
               sizeof(g_removed_graph_outputs[0]));
    snapshot->removed_output_count = g_removed_graph_output_count;
    return 0;
fail:
    graph_patch_snapshot_free(snapshot);
    return -1;
}

static void graph_patch_snapshot_free(GraphPatchSnapshot* snapshot) {
    if (!snapshot) return;
    if (snapshot->nodes) {
        for (int i = 0; i < snapshot->node_count; i++) {
            vx_node_dispose(&snapshot->nodes[i]);
        }
    }
    free(snapshot->nodes);
    cJSON_Delete(snapshot->graph_root);
    free(snapshot->fusions);
    free(snapshot->concat_sigmoid_fuse);
    free(snapshot->removed_outputs);
    memset(snapshot, 0, sizeof(*snapshot));
}

static int graph_patch_snapshot_restore(GraphPatchSnapshot* snapshot) {
    if (!snapshot || (snapshot->node_count > 0 &&
                      (!snapshot->fusions || !snapshot->concat_sigmoid_fuse)) ||
        (snapshot->removed_output_count > 0 && !snapshot->removed_outputs)) return -1;
    int rc = free_node_caches();
    graph_patch_reuse_rollback();
    for (int i = 0; i < g_nn; i++) vx_node_dispose(&g_n[i]);
    for (int i = snapshot->tensor_count; i < g_nt; i++) {
        if (g_t[i].owns) free(g_t[i].data);
        memset(&g_t[i], 0, sizeof(g_t[i]));
    }
    g_nt = snapshot->tensor_count;
    volvoxai_engine_tensor_name_index_invalidate();
    if (g_graph_root) cJSON_Delete(g_graph_root);
    g_graph_root = snapshot->graph_root;
    snapshot->graph_root = NULL;
    if (g_node_capacity) memset(g_n, 0, g_node_capacity * sizeof(g_n[0]));
    g_nn = snapshot->node_count;
    for (int i = 0; i < g_nn; i++) {
        g_n[i] = snapshot->nodes[i];
        memset(&snapshot->nodes[i], 0, sizeof(snapshot->nodes[i]));
    }
    if (g_node_capacity) {
        memset(g_node_fusion, 0, g_node_capacity * sizeof(g_node_fusion[0]));
        for (size_t index = 0; index < g_node_capacity; index++)
            g_concat_sigmoid_fuse[index] = -1;
    }
    if (g_nn > 0)
        memcpy(g_node_fusion, snapshot->fusions,
               (size_t)g_nn * sizeof(g_node_fusion[0]));
    g_graph_opt_stats = snapshot->graph_opt_stats;
    if (g_nn > 0)
        memcpy(g_concat_sigmoid_fuse, snapshot->concat_sigmoid_fuse,
               (size_t)g_nn * sizeof(g_concat_sigmoid_fuse[0]));
    if (g_tensor_capacity)
        memset(g_removed_graph_outputs, 0,
               g_tensor_capacity * sizeof(g_removed_graph_outputs[0]));
    if (snapshot->removed_output_count > 0)
        memcpy(g_removed_graph_outputs, snapshot->removed_outputs,
               (size_t)snapshot->removed_output_count *
               sizeof(g_removed_graph_outputs[0]));
    g_removed_graph_output_count = snapshot->removed_output_count;
    if (build_quant_params_pass() != 0) rc = -1;
    volvoxai_engine_finish_tensor_table_mutation();
    if (prepack_cpu_weights() != 0) rc = -1;
    g_weight_caches_dirty = rc == 0 ? 0 : 1;
    graph_patch_snapshot_free(snapshot);
    return rc;
}

static void graph_patch_restore_canonical_nodes(void) {
    for (int producer = 0; producer < g_nn; producer++) {
        const GraphNodeFusion* fusion = &g_node_fusion[producer];
        int peer = fusion->peer_idx;
        if (peer < 0 || peer >= g_nn) continue;
        if (fusion->id == GRAPH_FUSION_CONV_ADD || fusion->id == GRAPH_FUSION_CHAINED_ELEMENTWISE) {
            if (!fusion->source_tensor[0]) continue;
            strncpy(g_n[producer].out, fusion->source_tensor,
                    sizeof(g_n[producer].out) - 1);
            g_n[producer].out[sizeof(g_n[producer].out) - 1] = 0;
        }
    }
    for (int producer = 0; producer < g_nn; producer++) {
        if (!g_n[producer].fuse_relu6) continue;
        for (int consumer = 0; consumer < g_nn; consumer++) {
            if (!g_n[consumer].skip || (g_n[consumer].operator_kind != VX_OP_CLIP) ||
                strcmp(g_n[consumer].out, g_n[producer].out)) continue;
            const char* original = node_input_name(&g_n[consumer]);
            if (original) {
                strncpy(g_n[producer].out, original, sizeof(g_n[producer].out) - 1);
                g_n[producer].out[sizeof(g_n[producer].out) - 1] = 0;
            }
            break;
        }
    }
    graph_opt_reset_state();
    for (int i = 0; i < g_nn; i++) {
        g_n[i].skip = g_n[i].disabled ? 1 : 0;
        g_n[i].fuse_relu6 = 0;
    }
}

static int graph_patch_snapshot_has_output(const GraphPatchSnapshot* snapshot, const char* name) {
    return snapshot && graph_nodes_producer(snapshot->nodes, snapshot->node_count, name) != -1;
}

static int graph_patch_validate(const GraphPatchSnapshot* snapshot) {
    for (int i = 0; i < g_nn; i++) {
        Node* node = &g_n[i];
        if (node->disabled) continue;
        if (!vx_operator_kind_name(node->operator_kind) ||
            graph_node_output_count(node) <= 0) return -1;
        int outputs = graph_node_output_count(node);
        for (int oi = 0; oi < outputs; oi++) {
            const char* output = graph_node_output_name(node, oi);
            T* tensor = t_find(output);
            if (!output || !output[0] || !tensor || graph_nodes_producer(g_n, g_nn, output) == -2) return -1;
            for (int previous = 0; previous < oi; previous++) {
                if (!strcmp(graph_node_output_name(node, previous), output)) return -1;
            }
            int existed = tensor >= g_t && tensor < g_t + snapshot->tensor_count;
            if (existed && !graph_patch_snapshot_has_output(snapshot, output) &&
                removed_graph_output_index(output) < 0 &&
                !graph_patch_tensor_was_reused((int)(tensor - g_t))) return -1;
        }
        for (int ii = 0; ii < node->nin; ii++) {
            const char* input = node->ins[ii].name;
            if (!input[0] || !t_find(input)) return -1;
            int producer = graph_nodes_producer(g_n, g_nn, input);
            if (producer == -2 || producer >= i) return -1;
            if (producer < 0 && (graph_patch_snapshot_has_output(snapshot, input) ||
                                 removed_graph_output_index(input) >= 0)) return -1;
        }
    }
    return 0;
}

static void graph_patch_update_removed_outputs(const GraphPatchSnapshot* snapshot) {
    for (int i = 0; i < snapshot->node_count; i++) {
        int outputs = graph_node_output_count(&snapshot->nodes[i]);
        for (int oi = 0; oi < outputs; oi++) {
            const char* output = graph_node_output_name(&snapshot->nodes[i], oi);
            if (!output || graph_nodes_producer(g_n, g_nn, output) >= 0) continue;
            removed_graph_output_add(output);
            T* tensor = t_find(output);
            if (!tensor || !tensor->owns || !tensor->data) continue;
            int aliased = 0;
            for (int t = 0; t < g_nt; t++) {
                if (&g_t[t] != tensor && g_t[t].data == tensor->data) { aliased = 1; break; }
            }
            if (!aliased) memset(tensor->data, 0, (size_t)tensor->numel * tensor->elem_size);
        }
    }
    for (int i = 0; i < g_nn; i++) {
        int outputs = graph_node_output_count(&g_n[i]);
        for (int oi = 0; oi < outputs; oi++) {
            removed_graph_output_remove(graph_node_output_name(&g_n[i], oi));
        }
    }
}

static int graph_patch_transaction_begin(GraphPatchSnapshot* snapshot) {
    graph_patch_reuse_begin();
    if (graph_patch_snapshot_take(snapshot) != 0) {
        graph_patch_snapshot_free(snapshot);
        return -1;
    }
    if (free_node_caches() != 0) {
        graph_patch_snapshot_restore(snapshot);
        return -1;
    }
    graph_patch_restore_canonical_nodes();
    return 0;
}

static int graph_patch_transaction_finish(GraphPatchSnapshot* snapshot, int mutation_rc,
                                          int reoptimize, const char* persist_graph_path) {
    int rc = mutation_rc;
    if (rc == 0) rc = graph_patch_validate(snapshot);
    if (rc == 0) {
        graph_patch_update_removed_outputs(snapshot);
        rc = rebuild_after_graph_patch_ex(reoptimize ? 1 : 0);
    }
    if (rc == 0 && persist_graph_path &&
        (!persist_graph_path[0] || volvoxai_engine_save_graph(persist_graph_path) != 0)) rc = -1;
    if (rc != 0) {
        graph_patch_snapshot_restore(snapshot);
        return -1;
    }
    graph_patch_reuse_commit();
    graph_patch_snapshot_free(snapshot);
    return 0;
}

static int volvoxai_engine_patch_node_json_transaction(int node_index, const char* patch_json,
                                                       VolvoxAIEngineNodePatchMode mode,
                                                       int reoptimize) {
    GraphPatchSnapshot snapshot;
    if (graph_patch_transaction_begin(&snapshot) != 0) return -1;
    int rc = volvoxai_engine_patch_node_json_impl(node_index, patch_json, mode, 0);
    return graph_patch_transaction_finish(&snapshot, rc, reoptimize, NULL);
}

static int graph_patch_set_declared_outputs(cJSON* declared_outputs) {
    if (!declared_outputs) return 0;
    if (!cJSON_IsArray(declared_outputs)) return -1;
    cJSON* outputs = cJSON_CreateArray();
    if (!outputs) return -1;
    int count = cJSON_GetArraySize(declared_outputs);
    for (int i = 0; i < count; i++) {
        cJSON* name_json = cJSON_GetArrayItem(declared_outputs, i);
        T* tensor = cJSON_IsString(name_json) && name_json->valuestring
            ? t_find(name_json->valuestring) : NULL;
        if (!cJSON_IsString(name_json) || !name_json->valuestring || !name_json->valuestring[0] ||
            !tensor ||
            (!tensor->is_graph_input &&
             graph_nodes_producer(g_n, g_nn, name_json->valuestring) < 0) ||
            removed_graph_output_index(name_json->valuestring) >= 0) {
            cJSON_Delete(outputs);
            return -1;
        }
        for (int j = 0; j < i; j++) {
            cJSON* previous = cJSON_GetArrayItem(declared_outputs, j);
            if (cJSON_IsString(previous) && !strcmp(previous->valuestring, name_json->valuestring)) {
                cJSON_Delete(outputs);
                return -1;
            }
        }
        cJSON* output = cJSON_CreateString(name_json->valuestring);
        if (!output || !cJSON_AddItemToArray(outputs, output)) {
            cJSON_Delete(output);
            cJSON_Delete(outputs);
            return -1;
        }
    }
    if (!g_graph_root) g_graph_root = cJSON_CreateObject();
    if (!g_graph_root) { cJSON_Delete(outputs); return -1; }
    cJSON_DeleteItemFromObject(g_graph_root, "outputs");
    cJSON_AddItemToObject(g_graph_root, "outputs", outputs);
    return 0;
}

int volvoxai_engine_patch_node_json(int node_index, const char* patch_json,
                                    VolvoxAIEngineNodePatchMode mode,
                                    int rebuild_execution_plan) {
    if (volvoxai_engine_model_route_lease_active()) return -1;
    volvoxai_engine_model_lock();
    if (vx_dynamic_autograd_active_locked()) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    if (volvoxai_engine_training_accumulation_pending()) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    volvoxai_engine_metadata_lock();
    /* Structural metadata must never be left stale; rebuilding is mandatory. */
    (void)rebuild_execution_plan;
    int rc = volvoxai_engine_patch_node_json_transaction(node_index, patch_json, mode, 1);
    if (rc == 0) volvoxai_engine_model_generation_advance_locked();
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_engine_patch_graph_json(const char* patches_json, int rebuild_execution_plan,
                                     int reoptimize) {
    if (!patches_json || volvoxai_engine_model_route_lease_active() ||
        vx_json_text_contains_decoded_nul(
            (const unsigned char*)patches_json, strlen(patches_json))) return -1;
    cJSON* envelope = cJSON_Parse(patches_json);
    cJSON* patches = cJSON_IsArray(envelope) ? envelope : cJSON_GetObjectItem(envelope, "patches");
    cJSON* declared_outputs = cJSON_IsObject(envelope)
                                  ? cJSON_GetObjectItem(envelope, "declared_outputs") : NULL;
    cJSON* persist_path_json = cJSON_IsObject(envelope)
                                   ? cJSON_GetObjectItem(envelope, "persist_graph_path") : NULL;
    const char* persist_path = cJSON_IsString(persist_path_json)
                                   ? persist_path_json->valuestring : NULL;
    if (persist_path_json && !persist_path) {
        cJSON_Delete(envelope);
        return -1;
    }
    if (!cJSON_IsArray(patches) ||
        (cJSON_GetArraySize(patches) <= 0 && !declared_outputs)) {
        cJSON_Delete(envelope);
        return -1;
    }
    volvoxai_engine_model_lock();
    if (vx_dynamic_autograd_active_locked()) {
        volvoxai_engine_model_unlock();
        cJSON_Delete(envelope);
        return -1;
    }
    if (volvoxai_engine_training_accumulation_pending()) {
        volvoxai_engine_model_unlock();
        cJSON_Delete(envelope);
        return -1;
    }
    volvoxai_engine_metadata_lock();
    GraphPatchSnapshot snapshot;
    int rc = graph_patch_transaction_begin(&snapshot);
    if (rc == 0) {
        int count = cJSON_GetArraySize(patches);
        for (int i = 0; i < count; i++) {
            cJSON* item = cJSON_GetArrayItem(patches, i);
            cJSON* node_index_json = cJSON_GetObjectItem(item, "node_index");
            cJSON* mode_json = cJSON_GetObjectItem(item, "mode");
            cJSON* patch_json = cJSON_GetObjectItem(item, "patch");
            if (!cJSON_IsObject(item) || !cJSON_IsNumber(node_index_json) ||
                !cJSON_IsNumber(mode_json)) { rc = -1; break; }
            char* patch_text = NULL;
            if (cJSON_IsString(patch_json) || !patch_json) {
                const char* source = cJSON_IsString(patch_json) ? patch_json->valuestring : "{}";
                size_t length = strlen(source);
                patch_text = (char*)malloc(length + 1);
                if (patch_text) memcpy(patch_text, source, length + 1);
            } else {
                patch_text = cJSON_PrintUnformatted(patch_json);
            }
            if (!patch_text) { rc = -1; break; }
            rc = volvoxai_engine_patch_node_json_impl(node_index_json->valueint, patch_text,
                                                       (VolvoxAIEngineNodePatchMode)mode_json->valueint, 0);
            free(patch_text);
            if (rc != 0) break;
        }
        if (rc == 0) rc = graph_patch_set_declared_outputs(declared_outputs);
        rc = graph_patch_transaction_finish(&snapshot, rc, reoptimize, persist_path);
    }
    if (rc == 0) volvoxai_engine_model_generation_advance_locked();
    (void)rebuild_execution_plan;
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    cJSON_Delete(envelope);
    return rc;
}

static cJSON* node_to_graph_json(const Node* n) {
    cJSON* obj = cJSON_CreateObject();
    if (!obj) return NULL;
    cJSON_AddStringToObject(obj, "opType", vx_operator_kind_name(n->operator_kind));
    cJSON* inputs = refs_to_json(n->ins, n->nin);
    cJSON* outputs = refs_to_json(n->outs, n->nout);
    cJSON* output_shapes = node_output_shapes_json(n);
    cJSON* output_dtypes = node_output_dtypes_json(n);
    if (inputs) cJSON_AddItemToObject(obj, "inputs", inputs);
    if (outputs) cJSON_AddItemToObject(obj, "outputs", outputs);
    if (output_shapes) cJSON_AddItemToObject(obj, "outputs_shape", output_shapes);
    if (output_dtypes) cJSON_AddItemToObject(obj, "outputs_dtype", output_dtypes);
    if (n->params) {
        cJSON* params = cJSON_Duplicate(n->params, 1);
        if (params) cJSON_AddItemToObject(obj, "params", params);
    }
    return obj;
}

static int write_text_file_atomic(const char* path, const char* text) {
    if (!path || !path[0] || !text) return -1;
    char tmp_path[4096];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%ld", path, (long)time(NULL));
    FILE* f = fopen(tmp_path, "wb");
    if (!f) return -1;
    size_t n = strlen(text);
    if (n > 0 && fwrite(text, 1, n, f) != n) {
        fclose(f);
        remove(tmp_path);
        return -1;
    }
    if (fclose(f) != 0) {
        remove(tmp_path);
        return -1;
    }
    if (rename(tmp_path, path) != 0) {
        remove(tmp_path);
        return -1;
    }
    return 0;
}

int volvoxai_engine_save_graph(const char* path) {
    if (!g_graph_root || !path || !path[0]) return -1;
    cJSON* root = cJSON_Duplicate(g_graph_root, 1);
    if (!root) return -1;
    cJSON* nodes = cJSON_CreateArray();
    if (!nodes) {
        cJSON_Delete(root);
        return -1;
    }
    for (int i = 0; i < g_nn; i++) {
        if (g_n[i].disabled) continue;
        cJSON* node = node_to_graph_json(&g_n[i]);
        if (!node) {
            cJSON_Delete(nodes);
            cJSON_Delete(root);
            return -1;
        }
        cJSON_AddItemToArray(nodes, node);
    }
    cJSON_DeleteItemFromObject(root, "nodes");
    cJSON_AddItemToObject(root, "nodes", nodes);
    char* text = cJSON_Print(root);
    cJSON_Delete(root);
    if (!text) return -1;
    int rc = write_text_file_atomic(path, text);
    free(text);
    return rc;
}

/*
 * Forward helpers shared by the inference runtime and the guarded training
 * implementation. Backward orchestration itself lives under src/training.
 */
