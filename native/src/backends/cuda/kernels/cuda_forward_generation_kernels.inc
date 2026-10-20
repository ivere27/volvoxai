/* Private seeded generation: parallel score ordering and independent weight
 * calculation, followed by the unchanged score-order F64 sum and CDF. The
 * comparator includes the ID, so ties have one deterministic total order. */
struct VxCudaGenerationCandidate { double score; vx_i32 id; };
static_assert(sizeof(VxCudaGenerationCandidate) == 16, "generation scratch layout");

static VX_CUDA_DEVICE bool vx_generation_candidate_worse(
        VxCudaGenerationCandidate a, VxCudaGenerationCandidate b) {
    return a.score < b.score || (a.score == b.score && a.id > b.id);
}

static VX_CUDA_DEVICE double vx_generation_infinity() {
    union { vx_u64 bits; double value; } infinity;
    infinity.bits = (vx_u64)0x7ff0000000000000ULL;
    return infinity.value;
}

static VX_CUDA_DEVICE VxCudaGenerationCandidate vx_generation_candidate(
        float value, vx_u32 id) {
    VxCudaGenerationCandidate candidate;
    candidate.score = (double)value;
    candidate.id = (vx_i32)id;
    /* A NaN sorts ahead of every real candidate and survives top-k pruning.
     * Distinct negative IDs retain a total order even for several NaNs. The
     * final selection refuses this marker without modifying the ID tensor. */
    if (candidate.score != candidate.score) {
        candidate.score = vx_generation_infinity();
        candidate.id = -1 - (vx_i32)id;
    }
    return candidate;
}

static VX_CUDA_DEVICE void vx_generation_sort_tile_pair(
        VxCudaGenerationCandidate* items, vx_u32 lane,
        vx_u32 width, vx_u32 stride) {
    vx_u32 partner = lane ^ stride;
    if (lane >= partner) return;
    bool descending = (lane & width) == 0u;
    bool swap = descending
        ? vx_generation_candidate_worse(items[lane], items[partner])
        : vx_generation_candidate_worse(items[partner], items[lane]);
    if (swap) {
        VxCudaGenerationCandidate value = items[lane];
        items[lane] = items[partner]; items[partner] = value;
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_generation_sort_tiles_f32(
        const float* logits, VxCudaGenerationCandidate* scratch,
        vx_u32 vocabulary, vx_u32 row, vx_u32 keep) {
    VX_CUDA_SHARED VxCudaGenerationCandidate items[256];
    vx_u32 lane = threadIdx.x;
    vx_u32 index = blockIdx.x * 256u + lane;
    VxCudaGenerationCandidate candidate;
    candidate.score = -vx_generation_infinity(); candidate.id = 2147483647;
    if (index < vocabulary)
        candidate = vx_generation_candidate(logits[(vx_u64)row * vocabulary + index], index);
    items[lane] = candidate;
    vx_sync_threads();
    for (vx_u32 width = 2u; width <= 256u; width *= 2u)
        for (vx_u32 stride = width / 2u; stride; stride /= 2u) {
            /* One lane owns both members of each disjoint pair; no lane
             * reads a value concurrently rewritten by its partner. */
            vx_generation_sort_tile_pair(items, lane, width, stride);
            vx_sync_threads();
        }
    if (index < vocabulary && lane < keep) scratch[index] = items[lane];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_generation_merge_f32(
        const VxCudaGenerationCandidate* input, VxCudaGenerationCandidate* output,
        vx_u32 vocabulary, vx_u32 width, vx_u32 keep) {
    vx_u32 index = vx_global_x();
    if (index >= vocabulary) return;
    vx_u32 base = index / (width * 2u) * (width * 2u);
    vx_u32 right = index - base >= width ? 1u : 0u;
    vx_u32 own_base = base + right * width;
    vx_u32 own_index = index - own_base;
    vx_u32 own_count = vocabulary - own_base;
    if (own_count > width) own_count = width;
    if (own_count > keep) own_count = keep;
    if (own_index >= own_count) return;
    vx_u32 other_base = base + (1u - right) * width;
    vx_u32 other_count = other_base < vocabulary ? vocabulary - other_base : 0u;
    if (other_count > width) other_count = width;
    if (other_count > keep) other_count = keep;
    VxCudaGenerationCandidate candidate = input[index];
    vx_u32 low = 0u, high = other_count;
    while (low < high) {
        vx_u32 middle = low + (high - low) / 2u;
        if (vx_generation_candidate_worse(candidate, input[other_base + middle])) low = middle + 1u;
        else high = middle;
    }
    vx_u32 rank = own_index + low;
    if (rank < keep) output[base + rank] = candidate;
}

/* Preserve the original freestanding nonpositive F64 exponential exactly.
 * Evaluating candidates independently changes neither its arithmetic nor
 * the ordered accumulation used by the seeded distribution. */
static VX_CUDA_DEVICE double vx_generation_exp(double value) {
    if (value < -745.0) return 0.0;
    double scaled = value * 1.4426950408889634073599;
    int exponent = (int)scaled;
    if ((double)exponent > scaled) exponent--;
    double remainder = value - (double)exponent * 0.6931471805599453094172;
    double sum = 1.0, term = 1.0;
    for (int degree = 1; degree <= 18; degree++) { term *= remainder / (double)degree; sum += term; }
    union { vx_u64 bits; double value; } power;
    if (exponent >= -1022) power.bits = (vx_u64)(exponent + 1023) << 52;
    else if (exponent >= -1074) power.bits = (vx_u64)1 << (exponent + 1074);
    else return 0.0;
    return sum * power.value;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_generation_weights_f32(
        const VxCudaGenerationCandidate* input, VxCudaGenerationCandidate* output,
        vx_u32 count, double temperature) {
    vx_u32 index = vx_global_x();
    if (index >= count) return;
    double maximum = input[0].score;
    double infinity = vx_generation_infinity();
    VxCudaGenerationCandidate candidate = input[index];
    /* The maximum remains immutable until every lane has read it. Reusing
     * the other ping-pong buffer avoids a head-write/read race. */
    candidate.score = input[0].id < 0 || maximum == -infinity ? 0.0
        : maximum == infinity ? (candidate.score == infinity ? 1.0 : 0.0)
        : vx_generation_exp((candidate.score - maximum) / temperature);
    output[index] = candidate;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_generation_sample_f32(
        const float* logits, vx_i32* ids, VxCudaGenerationCandidate* scratch,
        vx_i32* selected, vx_u32 vocabulary, vx_u32 row, double temperature,
        vx_u32 top_k, double top_p, double uniform) {
    if (vx_global_x() != 0u) return;
    *selected = -1;
    (void)logits; (void)temperature;
    vx_u32 count = top_k && top_k < vocabulary ? top_k : vocabulary;
    if (scratch[0].id < 0) return;
    double total = 0.0;
    for (vx_u32 candidate = 0u; candidate < count; candidate++) total += scratch[candidate].score;
    if (!(total > 0.0)) return;
    double retained = 0.0, threshold = top_p * total;
    vx_u32 admitted = 0u;
    do { retained += scratch[admitted++].score; } while (admitted < count && retained < threshold);
    double target = uniform * retained, cumulative = 0.0;
    vx_i32 id = scratch[admitted - 1u].id;
    for (vx_u32 candidate = 0u; candidate < admitted; candidate++) {
        cumulative += scratch[candidate].score;
        if (target < cumulative) { id = scratch[candidate].id; break; }
    }
    ids[row] = id;
    *selected = id;
}
