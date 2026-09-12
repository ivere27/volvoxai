/* Declared-dimension and resource-bound helpers.
 *
 * Fragment of public_api.c, included where these definitions used to sit.
 * These are the axis-level helpers retained by native GPU route/capacity
 * checks, the independent-batch proof, and CPU typed-workspace accounting.
 * Canonical graph semantics, including Concat affine relations, are proved by
 * graph_domain.c and are not duplicated here. */

static int vx_dimension_domain_is_dynamic(const VxDeclaredTensor* tensor,
                                          uint32_t axis) {
    int64_t first;
    int64_t last;
    int64_t remainder;
    int64_t adjustment;
    if (!tensor || axis >= tensor->rank ||
        tensor->kinds[axis] != VX_DIMENSION_SYMBOLIC) return 0;
    remainder = tensor->minimums[axis] % tensor->multiples[axis];
    adjustment = remainder ? tensor->multiples[axis] - remainder : 0;
    if (adjustment > tensor->maximums[axis] - tensor->minimums[axis]) return 0;
    first = tensor->minimums[axis] + adjustment;
    last = tensor->maximums[axis] -
        tensor->maximums[axis] % tensor->multiples[axis];
    return first < last;
}

typedef struct VxDimensionProgression {
    int64_t first;
    int64_t last;
    int64_t step;
    uint64_t count;
} VxDimensionProgression;

static const VxDeclaredTensor* vx_declared_tensor_find(
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        const char* name) {
    if (!tensors || !name) return NULL;
    for (size_t index = 0; index < tensor_count; index++)
        if (tensors[index].name && !strcmp(tensors[index].name, name))
            return &tensors[index];
    return NULL;
}

static int vx_concat_axis(const cJSON* params,
                          uint32_t rank,
                          size_t* out_axis) {
    static const char* const fields[] = {"axis"};
    const cJSON* item;
    int64_t raw = 0;
    if (!params || !out_axis || !rank ||
        !vx_json_object_fields(params, fields, 1u, 0u, 1u)) return 0;
    item = cJSON_GetObjectItemCaseSensitive(params, "axis");
    if (item) {
        if (!cJSON_IsNumber(item) ||
            item->valuedouble < -(double)rank ||
            item->valuedouble >= (double)rank) return 0;
        raw = (int64_t)item->valuedouble;
        if ((double)raw != item->valuedouble) return 0;
    }
    if (raw < 0) raw += rank;
    *out_axis = (size_t)raw;
    return 1;
}

static int vx_declared_dimension_fixed_value(
        const VxDeclaredTensor* tensor,
        size_t axis,
        int64_t* out_value) {
    if (!tensor || axis >= tensor->rank || !out_value) return 0;
    if (tensor->kinds[axis] == VX_DIMENSION_FIXED ||
        (tensor->kinds[axis] == VX_DIMENSION_SYMBOLIC &&
         tensor->minimums[axis] == tensor->maximums[axis])) {
        *out_value = tensor->minimums[axis];
        return 1;
    }
    return 0;
}

static int vx_declared_dimensions_provably_equal(
        const VxDeclaredTensor* left,
        size_t left_axis,
        const VxDeclaredTensor* right,
        size_t right_axis) {
    int64_t left_fixed;
    int64_t right_fixed;
    if (!left || !right || left_axis >= left->rank ||
        right_axis >= right->rank) return 0;
    if (left->kinds[left_axis] == VX_DIMENSION_SYMBOLIC &&
        right->kinds[right_axis] == VX_DIMENSION_SYMBOLIC &&
        left->symbols[left_axis] && right->symbols[right_axis] &&
        !strcmp(left->symbols[left_axis], right->symbols[right_axis]))
        return 1;
    return vx_declared_dimension_fixed_value(left, left_axis, &left_fixed) &&
        vx_declared_dimension_fixed_value(right, right_axis, &right_fixed) &&
        left_fixed == right_fixed;
}

static int vx_declared_dimension_progression(
        const VxDeclaredTensor* tensor,
        size_t axis,
        VxDimensionProgression* progression) {
    int64_t remainder;
    int64_t adjustment;
    if (!tensor || axis >= tensor->rank || !progression ||
        tensor->kinds[axis] != VX_DIMENSION_SYMBOLIC ||
        tensor->minimums[axis] <= 0 ||
        tensor->maximums[axis] < tensor->minimums[axis] ||
        tensor->multiples[axis] <= 0) return 0;
    remainder = tensor->minimums[axis] % tensor->multiples[axis];
    adjustment = remainder ? tensor->multiples[axis] - remainder : 0;
    if (adjustment > tensor->maximums[axis] - tensor->minimums[axis])
        return 0;
    progression->first = tensor->minimums[axis] + adjustment;
    progression->last = tensor->maximums[axis] -
        tensor->maximums[axis] % tensor->multiples[axis];
    progression->step = tensor->multiples[axis];
    progression->count =
        (uint64_t)((progression->last - progression->first) /
                   progression->step) + 1u;
    return progression->first <= progression->last;
}

/* The AVX2 QBatch route packs both dynamic operands into one transient buffer:
 *
 *   pairs   = ceil(K / 2)
 *   nblocks = ceil(N / 16)
 *   bytes   = nblocks * pairs * 64 + M * pairs * sizeof(int32_t)
 *
 * The expression is monotone in M/K/N, so evaluating declared maxima is a
 * conservative full-domain bound even when symbols correlate those axes. A
 * node whose operands are not logical activations contributes zero: it then
 * remains on the allocation-free native/portable route unless another already
 * charged shared workspace happens to fit its concrete call. */
static size_t vx_qbatch_cpu_workspace_domain_bound(
        const cJSON* node,
        const VxDeclaredTensor* tensors,
        size_t tensor_count) {
    const cJSON* inputs = node
        ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
    const cJSON* a_ref = cJSON_IsObject(inputs)
        ? cJSON_GetObjectItemCaseSensitive(inputs, "a") : NULL;
    const cJSON* b_ref = cJSON_IsObject(inputs)
        ? cJSON_GetObjectItemCaseSensitive(inputs, "b") : NULL;
    const VxDeclaredTensor* a;
    const VxDeclaredTensor* b;
    int64_t m;
    int64_t k;
    int64_t b_k;
    int64_t n;
    if (!cJSON_IsString(a_ref) || !a_ref->valuestring ||
        !cJSON_IsString(b_ref) || !b_ref->valuestring ||
        !(a = vx_declared_tensor_find(
              tensors, tensor_count, a_ref->valuestring)) ||
        !(b = vx_declared_tensor_find(
              tensors, tensor_count, b_ref->valuestring)) ||
        a->rank < 2u || b->rank < 2u)
        return 0u;
    m = a->maximums[a->rank - 2u];
    k = a->maximums[a->rank - 1u];
    b_k = b->maximums[b->rank - 2u];
    n = b->maximums[b->rank - 1u];
    if (b_k > k) k = b_k;
    if (m <= 0 || k <= 0 || n <= 0 ||
        (uint64_t)m > UINT32_MAX || (uint64_t)k > UINT32_MAX ||
        (uint64_t)n > UINT32_MAX)
        return 0u;
    return vx_qbatch_matmul_i8u8_native_workspace_bytes(
        (uint32_t)m, (uint32_t)k, (uint32_t)n);
}
