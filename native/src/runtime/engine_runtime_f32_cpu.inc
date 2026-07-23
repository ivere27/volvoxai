static int tensor_contiguous_strides(const T* tensor, long strides[8]) {
    if (!tensor || !strides || tensor->ndim <= 0 || tensor->ndim > 8) return -1;
    long stride = 1;
    for (int dimension = tensor->ndim - 1; dimension >= 0; dimension--) {
        if (tensor->shape[dimension] <= 0 || stride > LONG_MAX / tensor->shape[dimension])
            return -1;
        strides[dimension] = stride;
        stride *= tensor->shape[dimension];
    }
    return stride == tensor->numel ? 0 : -1;
}

static int tensor_broadcast_strides(const T* input, const T* output, long strides[8]) {
    if (!input || !output || !strides || output->ndim <= 0 || output->ndim > 8 ||
        input->ndim < 0 || input->ndim > output->ndim) return -1;
    long contiguous[8] = {0};
    if (input->ndim > 0 && tensor_contiguous_strides(input, contiguous) != 0) return -1;
    for (int dimension = 0; dimension < 8; dimension++) strides[dimension] = 0;
    int offset = output->ndim - input->ndim;
    for (int dimension = 0; dimension < output->ndim; dimension++) {
        int source = dimension - offset;
        int input_size = source < 0 ? 1 : input->shape[source];
        int output_size = output->shape[dimension];
        if (output_size <= 0 || (input_size != 1 && input_size != output_size)) return -1;
        if (source >= 0 && input_size != 1) strides[dimension] = contiguous[source];
    }
    return 0;
}

static long tensor_broadcast_index(long output_index, int rank,
                                   const long output_strides[8],
                                   const long input_strides[8]) {
    long remainder = output_index;
    long input_index = 0;
    for (int dimension = 0; dimension < rank; dimension++) {
        long coordinate = remainder / output_strides[dimension];
        remainder %= output_strides[dimension];
        input_index += coordinate * input_strides[dimension];
    }
    return input_index;
}

static int tensor_plain_32(const T* tensor) {
    return tensor && tensor->data && tensor->elem_size == sizeof(uint32_t) &&
        (tensor->dtype == T_F32 || tensor->dtype == T_I32);
}

static int tensor_copy_plain_32(const T* input, T* output) {
    if (!tensor_plain_32(input) || !tensor_plain_32(output) ||
        input->dtype != output->dtype || input->numel != output->numel ||
        input->numel <= 0 || (size_t)input->numel > SIZE_MAX / sizeof(uint32_t))
        return -1;
    if (input->data != output->data)
        memcpy(output->data, input->data,
               (size_t)input->numel * sizeof(uint32_t));
    return 0;
}

static T* node_input_default(Node* n) {
    T* in = nin(n, "input");
    if (!in) in = nin(n, "x");
    if (!in) in = nin(n, "data");
    if (!in && n->nin > 0) in = t_find(n->ins[0].name);
    return in;
}

static int groupnorm_dimensions(const T* in, const T* out, const T* weight,
                                const T* bias, int groups, int* batch, int* height,
                                int* width, int* channels) {
    if (!in || !out || !weight || in->dtype != T_F32 || out->dtype != T_F32 ||
        weight->dtype != T_F32 || in->ndim != 4 || out->ndim != 4 ||
        in->numel != out->numel) return -1;
    int n = in->shape[0], h = in->shape[1], w = in->shape[2], c = in->shape[3];
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || groups <= 0 || c % groups ||
        out->shape[0] != n || out->shape[1] != h || out->shape[2] != w ||
        out->shape[3] != c || weight->numel != c ||
        (bias && (bias->dtype != T_F32 || bias->numel != c))) return -1;
    if (batch) *batch = n;
    if (height) *height = h;
    if (width) *width = w;
    if (channels) *channels = c;
    return 0;
}

static int normalization_dimensions(const T* in, const T* out,
                                    const T* weight, const T* bias,
                                    int width, long* rows) {
    long row_count;
    if (!in || !out || !weight || in->dtype != T_F32 ||
        out->dtype != T_F32 || weight->dtype != T_F32 ||
        in->ndim <= 0 || !tensor_same_shape(in, out) ||
        width <= 0 || in->shape[in->ndim - 1] != width ||
        weight->ndim != 1 || weight->shape[0] != width ||
        weight->numel != width ||
        (bias && (bias->dtype != T_F32 || bias->ndim != 1 ||
                  bias->shape[0] != width || bias->numel != width)))
        return -1;
    row_count = in->numel / width;
    if (row_count <= 0 || row_count > INT32_MAX ||
        in->numel != row_count * width)
        return -1;
    if (rows) *rows = row_count;
    return 0;
}

static int groupnorm_forward_f32(const T* in, const T* weight, const T* bias,
                                 T* out, int groups, float epsilon) {
    int batch, height, width, channels;
    if (!(epsilon > 0.0f) || !isfinite(epsilon) ||
        groupnorm_dimensions(in, out, weight, bias, groups, &batch, &height,
                             &width, &channels) != 0) return -1;
    return groupnorm_f32(in->data, weight->data, bias ? bias->data : NULL,
                         out->data, (uint32_t)batch, (uint32_t)height,
                         (uint32_t)width, (uint32_t)channels,
                         (uint32_t)groups, (double)epsilon) ? 0 : -1;
}

static int last_axis_reduction_dimensions(Node* n, T* out, T** input,
                                          long* rows, int* width, float* scale) {
    T* in = node_input_default(n);
    if (!in) in = nin(n, "data");
    if (!n || !in || !out || in->dtype != T_F32 || out->dtype != T_F32 ||
        in->ndim <= 0) return -1;
    int axis = p_int(n->params, "axis", in->ndim - 1);
    if (axis < 0) axis += in->ndim;
    int d = in->shape[in->ndim - 1];
    long r = d > 0 ? in->numel / d : 0;
    if (axis != in->ndim - 1 || d <= 0 || r <= 0 || in->numel != r * d ||
        out->numel != r) return -1;
    if (input) *input = in;
    if (rows) *rows = r;
    if (width) *width = d;
    if (scale) *scale = !strcmp(n->op, "ReduceMean") ? 1.0f / (float)d : 1.0f;
    return 0;
}

/* ONNX-style F32 ArgMax with an ordinary I32 index output.  The decoder-row
 * case is intentionally narrow: incremental inference currently exposes one
 * active sequence row for B=1 rank-3 logits. */
static int argmax_forward_f32(Node* node, T* output) {
    T* input = node_input_default(node);
    long outer = 1;
    long inner = 1;
    int axis;
    int keepdims;
    int select_last_index;
    if (!node || !input || !output || !input->data || !output->data ||
        input->dtype != T_F32 || output->dtype != T_I32 ||
        output->elem_size != sizeof(int32_t) || input->ndim <= 0 ||
        input->ndim > 8 || input->data == output->data) return -1;
    axis = p_int(node->params, "axis", 0);
    if (axis < 0) axis += input->ndim;
    keepdims = p_int(node->params, "keepdims", 1);
    select_last_index = p_int(node->params, "select_last_index", 0);
    if (axis < 0 || axis >= input->ndim || (keepdims != 0 && keepdims != 1) ||
        select_last_index != 0 ||
        input->shape[axis] <= 0) return -1;
    for (int dimension = 0; dimension < axis; dimension++) {
        if (input->shape[dimension] <= 0 ||
            outer > LONG_MAX / input->shape[dimension]) return -1;
        outer *= input->shape[dimension];
    }
    for (int dimension = axis + 1; dimension < input->ndim; dimension++) {
        if (input->shape[dimension] <= 0 ||
            inner > LONG_MAX / input->shape[dimension]) return -1;
        inner *= input->shape[dimension];
    }
    if (outer <= 0 || inner <= 0 || outer > LONG_MAX / inner ||
        output->numel != outer * inner) return -1;
    if (keepdims) {
        if (output->ndim != input->ndim) return -1;
        for (int dimension = 0; dimension < input->ndim; dimension++) {
            int expected = dimension == axis ? 1 : input->shape[dimension];
            if (output->shape[dimension] != expected) return -1;
        }
    } else {
        if (output->ndim != input->ndim - 1) return -1;
        for (int dimension = 0; dimension < input->ndim; dimension++) {
            if (dimension == axis) continue;
            int output_dimension = dimension < axis ? dimension : dimension - 1;
            if (output->shape[output_dimension] != input->shape[dimension]) return -1;
        }
    }

    long first_outer = 0;
    long outer_count = outer;
    if (g_active_row >= 0) {
        if (input->ndim != 3 || input->shape[0] != 1 || axis != 2 ||
            inner != 1 || output->ndim != (keepdims ? 3 : 2) ||
            g_active_row >= input->shape[1]) return -1;
        first_outer = g_active_row;
        outer_count = 1;
    }
    const int axis_size = input->shape[axis];
    int32_t* indices = (int32_t*)output->data;
    for (long outer_index = first_outer;
         outer_index < first_outer + outer_count; outer_index++) {
        for (long inner_index = 0; inner_index < inner; inner_index++) {
            long base = (outer_index * axis_size * inner) + inner_index;
            float best = input->data[base];
            int best_index = 0;
            for (int axis_index = 1; axis_index < axis_size; axis_index++) {
                float value = input->data[base + (long)axis_index * inner];
                if (value > best) {
                    best = value;
                    best_index = axis_index;
                }
            }
            indices[outer_index * inner + inner_index] = best_index;
        }
    }
    return 0;
}

static int gpu_broadcast_strides(const T* input, const T* output, uint32_t strides[8]) {
    if (!input || !output || output->ndim < 0 || output->ndim > 8 ||
        input->ndim < 0 || input->ndim > output->ndim) return -1;
    for (int i = 0; i < 8; i++) strides[i] = 0;
    long contiguous[8] = {0};
    long stride = 1;
    for (int i = input->ndim - 1; i >= 0; i--) {
        if (input->shape[i] <= 0 || stride > UINT32_MAX) return -1;
        contiguous[i] = stride;
        stride *= input->shape[i];
    }
    int offset = output->ndim - input->ndim;
    for (int dimension = 0; dimension < output->ndim; dimension++) {
        int source = dimension - offset;
        int input_dim = source < 0 ? 1 : input->shape[source];
        int output_dim = output->shape[dimension];
        if (output_dim <= 0 || (input_dim != 1 && input_dim != output_dim)) return -1;
        if (source >= 0 && input_dim != 1) strides[dimension] = (uint32_t)contiguous[source];
    }
    return 0;
}

#if VOLVOXAI_ENABLE_TRAINING
#include "../training/training_forward_dropout.inc"
#include "../training/training_runtime.inc"
#endif

// ---- Shape-aware F32 binary arithmetic (row-windowed) ----------------------
static int binop(Node* n, int operation) {
    T* a = nin(n, "a"); T* b = nin(n, "b"); T* o = t_find(n->out);
    if (!a) a = nin(n, "input");
    if (!a || !b || !o || a->dtype != T_F32 || b->dtype != T_F32 || o->dtype != T_F32 ||
        o->ndim <= 0 || o->ndim > 8 ||
        operation < VX_PORTABLE_BINARY_ADD ||
        operation > VX_PORTABLE_BINARY_DIV) return -1;
    long output_strides[8] = {0}, a_strides[8] = {0}, b_strides[8] = {0};
    if (tensor_contiguous_strides(o, output_strides) != 0 ||
        tensor_broadcast_strides(a, o, a_strides) != 0 ||
        tensor_broadcast_strides(b, o, b_strides) != 0) return -1;
    int D = o->shape[o->ndim - 1];
    long total = D > 0 ? o->numel / D : 0;
    int r0, rc;
    if (D <= 0 || total <= 0 || total > INT_MAX || o->numel != total * D)
        return -1;
    /* seq_range() is a token-row window only for [S,D], [1,S,D], and the
     * sequence-major singleton-batch spelling [S,1,D]. In multi-head
     * attention tensors, the token axis can precede or follow a head/feature
     * axis; treating active_row as one flattened outer row would update only
     * one head. Keep those layouts correct by recomputing this inexpensive
     * elementwise node over its complete retained tensor. */
    if (g_active_row >= 0 &&
        !(o->ndim == 2 ||
          (o->ndim == 3 && (o->shape[0] == 1 || o->shape[1] == 1)))) {
        r0 = 0;
        rc = (int)total;
    } else {
        seq_range(total, &r0, &rc);
    }
    long off = (long)r0 * D; int cnt = rc * D;
    int relu = p_int(n->params, "relu", 0);
    if (a->numel == o->numel && b->numel == o->numel) {
        if (operation == VX_PORTABLE_BINARY_ADD)
            vx_fused_add_relu_f32(a->data + off, b->data + off,
                                  o->data + off, cnt, relu);
        else if (operation == VX_PORTABLE_BINARY_MUL)
            mul_f32(a->data + off, b->data + off, o->data + off, cnt);
        else {
            for (int index = 0; index < cnt; index++) {
                float left = a->data[off + index];
                float right = b->data[off + index];
                o->data[off + index] =
                    operation == VX_PORTABLE_BINARY_SUB
                        ? left - right : left / right;
            }
        }
    } else if (a->ndim > 0 && b->ndim > 0 && off == 0 && cnt == o->numel &&
               o->numel <= UINT32_MAX) {
        uint32_t a_shape[8], b_shape[8], output_shape[8];
        for (int dimension = 0; dimension < a->ndim; dimension++)
            a_shape[dimension] = (uint32_t)a->shape[dimension];
        for (int dimension = 0; dimension < b->ndim; dimension++)
            b_shape[dimension] = (uint32_t)b->shape[dimension];
        for (int dimension = 0; dimension < o->ndim; dimension++)
            output_shape[dimension] = (uint32_t)o->shape[dimension];
        if (!binary_broadcast_f32(
                a->data, b->data, o->data, a_shape, b_shape, output_shape,
                (uint32_t)a->ndim, (uint32_t)b->ndim, (uint32_t)o->ndim,
                (uint32_t)o->numel, (uint32_t)operation))
            return -1;
        if (operation == VX_PORTABLE_BINARY_ADD && relu == 1)
            relu_f32(o->data, o->data, (int)o->numel);
        else if (operation == VX_PORTABLE_BINARY_ADD && relu >= 2)
            clip_f32(o->data, o->data, (int)o->numel, 0.0f, 6.0f);
    } else {
        for (long output_index = off; output_index < off + cnt; output_index++) {
            long a_index = tensor_broadcast_index(output_index, o->ndim,
                                                  output_strides, a_strides);
            long b_index = tensor_broadcast_index(output_index, o->ndim,
                                                  output_strides, b_strides);
            float left = a->data[a_index];
            float right = b->data[b_index];
            float value = operation == VX_PORTABLE_BINARY_ADD ? left + right :
                operation == VX_PORTABLE_BINARY_MUL ? left * right :
                operation == VX_PORTABLE_BINARY_SUB ? left - right :
                left / right;
            if (operation == VX_PORTABLE_BINARY_ADD && relu) {
                if (value < 0.0f) value = 0.0f;
                if (relu >= 2 && value > 6.0f) value = 6.0f;
            }
            o->data[output_index] = value;
        }
    }
    return 0;
}

static int compare_i32_forward(Node* node, int comparison) {
    T* a = nin(node, "a");
    T* b = nin(node, "b");
    T* output = t_find(node->out);
    long output_strides[8] = {0};
    long a_strides[8] = {0};
    long b_strides[8] = {0};
    if (!a || !b || !output || a->dtype != T_I32 || b->dtype != T_I32 ||
        output->dtype != T_I32 || output->ndim <= 0 || output->ndim > 8 ||
        output->numel <= 0 || (comparison != 0 && comparison != 1) ||
        tensor_contiguous_strides(output, output_strides) != 0 ||
        tensor_broadcast_strides(a, output, a_strides) != 0 ||
        tensor_broadcast_strides(b, output, b_strides) != 0)
        return -1;
    const int32_t* left = (const int32_t*)a->data;
    const int32_t* right = (const int32_t*)b->data;
    int32_t* result = (int32_t*)output->data;
    for (long index = 0; index < output->numel; index++) {
        long left_index = tensor_broadcast_index(
            index, output->ndim, output_strides, a_strides);
        long right_index = tensor_broadcast_index(
            index, output->ndim, output_strides, b_strides);
        result[index] = comparison == 0
            ? left[left_index] == right[right_index]
            : left[left_index] >= right[right_index];
    }
    return 0;
}

static int not_i32_forward(Node* node) {
    T* input = node_input_default(node);
    T* output = t_find(node->out);
    if (!input || !output || input->dtype != T_I32 ||
        output->dtype != T_I32 || !tensor_same_shape(input, output))
        return -1;
    const int32_t* source = (const int32_t*)input->data;
    int32_t* destination = (int32_t*)output->data;
    for (long index = 0; index < output->numel; index++)
        destination[index] = source[index] == 0;
    return 0;
}

static int where_plain_32_forward(Node* node) {
    T* condition = nin(node, "condition");
    T* x = nin(node, "x");
    T* y = nin(node, "y");
    T* output = t_find(node->out);
    if (!condition) condition = nin(node, "cond");
    if (!x) x = nin(node, "a");
    if (!y) y = nin(node, "b");
    if (!condition || !x || !y || !output ||
        condition->dtype != T_I32 || !tensor_plain_32(x) ||
        !tensor_plain_32(y) || !tensor_plain_32(output) ||
        x->dtype != y->dtype || output->dtype != x->dtype ||
        !tensor_same_shape(condition, output) ||
        !tensor_same_shape(x, output) || !tensor_same_shape(y, output))
        return -1;
    const int32_t* selected = (const int32_t*)condition->data;
    const uint32_t* left = (const uint32_t*)x->data;
    const uint32_t* right = (const uint32_t*)y->data;
    uint32_t* result = (uint32_t*)output->data;
    for (long index = 0; index < output->numel; index++)
        result[index] = selected[index] != 0 ? left[index] : right[index];
    return 0;
}

static int batch_matmul_f32_forward(Node* node) {
    T* a = nin(node, "a");
    T* b = nin(node, "b");
    T* output = t_find(node->out);
    int coordinates[6] = {0};
    if (!a || !b || !output || a->dtype != T_F32 || b->dtype != T_F32 ||
        output->dtype != T_F32 || a->ndim < 2 || b->ndim < 2 ||
        a->ndim > 8 || b->ndim > 8 ||
        output->ndim != (a->ndim > b->ndim ? a->ndim : b->ndim) ||
        output->ndim < 2 || output->ndim > 8)
        return -1;
    int m = a->shape[a->ndim - 2];
    int k = a->shape[a->ndim - 1];
    int b_k = b->shape[b->ndim - 2];
    int n = b->shape[b->ndim - 1];
    int batch_rank = output->ndim - 2;
    if (m <= 0 || k <= 0 || n <= 0 || b_k != k ||
        output->shape[output->ndim - 2] != m ||
        output->shape[output->ndim - 1] != n)
        return -1;
    long batch_count = 1;
    for (int dimension = 0; dimension < batch_rank; dimension++) {
        int a_source = dimension - (batch_rank - (a->ndim - 2));
        int b_source = dimension - (batch_rank - (b->ndim - 2));
        int a_size = a_source < 0 ? 1 : a->shape[a_source];
        int b_size = b_source < 0 ? 1 : b->shape[b_source];
        int expected = a_size > b_size ? a_size : b_size;
        if (a_size <= 0 || b_size <= 0 ||
            (a_size != 1 && b_size != 1 && a_size != b_size) ||
            output->shape[dimension] != expected ||
            batch_count > LONG_MAX / expected)
            return -1;
        batch_count *= expected;
    }
    long a_batches = a->numel / ((long)m * k);
    long b_batches = b->numel / ((long)k * n);
    if (a->numel != a_batches * (long)m * k ||
        b->numel != b_batches * (long)k * n ||
        output->numel != batch_count * (long)m * n)
        return -1;
    for (long batch = 0; batch < batch_count; batch++) {
        long remainder = batch;
        for (int dimension = batch_rank - 1; dimension >= 0; dimension--) {
            int size = output->shape[dimension];
            coordinates[dimension] = (int)(remainder % size);
            remainder /= size;
        }
        long a_batch = 0;
        int a_offset = batch_rank - (a->ndim - 2);
        for (int dimension = 0; dimension < a->ndim - 2; dimension++) {
            int size = a->shape[dimension];
            int coordinate = size == 1 ? 0 : coordinates[a_offset + dimension];
            a_batch = a_batch * size + coordinate;
        }
        long b_batch = 0;
        int b_offset = batch_rank - (b->ndim - 2);
        for (int dimension = 0; dimension < b->ndim - 2; dimension++) {
            int size = b->shape[dimension];
            int coordinate = size == 1 ? 0 : coordinates[b_offset + dimension];
            b_batch = b_batch * size + coordinate;
        }
        matmul_f32(
            a->data + a_batch * (long)m * k,
            b->data + b_batch * (long)k * n,
            NULL,
            output->data + batch * (long)m * n,
            m, k, n);
    }
    return 0;
}

static int transpose_tensor(const T* in, T* out, const int* perm) {
    uint32_t input_shape[8], permutation[8];
    if (!in || !out || !perm || !tensor_plain_32(in) ||
        !tensor_plain_32(out) || in->dtype != out->dtype ||
        in->ndim <= 0 || in->ndim > 8 || out->ndim != in->ndim ||
        in->numel != out->numel || in->numel <= 0 || in->numel > UINT32_MAX)
        return -1;
    for (int axis = 0; axis < in->ndim; axis++) {
        if (in->shape[axis] <= 0 || perm[axis] < 0 || perm[axis] >= in->ndim ||
            out->shape[axis] != in->shape[perm[axis]])
            return -1;
        input_shape[axis] = (uint32_t)in->shape[axis];
        permutation[axis] = (uint32_t)perm[axis];
    }
    /* Swapping a singleton batch axis with the sequence axis preserves the
     * physical order of every [token, feature] row. During incremental decode
     * only the active row can have changed, so copying that one contiguous row
     * is both exact and substantially cheaper than replaying the full
     * transpose. No other permutation is inferred from shape alone. */
    if (g_active_row >= 0 && in->ndim == 3 && perm[0] == 1 &&
        perm[1] == 0 && perm[2] == 2 && in->shape[2] > 0 &&
        ((in->shape[0] == 1 && out->shape[1] == 1 &&
          out->shape[0] == in->shape[1]) ||
         (in->shape[1] == 1 && out->shape[0] == 1 &&
          out->shape[1] == in->shape[0]))) {
        const long sequence = in->shape[0] == 1
            ? in->shape[1] : in->shape[0];
        const size_t width = (size_t)in->shape[2];
        size_t offset;
        if (g_active_row >= sequence ||
            (size_t)g_active_row > SIZE_MAX / width) return -1;
        offset = (size_t)g_active_row * width;
        if (offset > (size_t)in->numel ||
            width > (size_t)in->numel - offset ||
            offset > SIZE_MAX / sizeof(uint32_t) ||
            width > SIZE_MAX / sizeof(uint32_t) - offset) return -1;
        if (in->data != out->data)
            memmove((uint32_t*)out->data + offset,
                    (const uint32_t*)in->data + offset,
                    width * sizeof(uint32_t));
        return 0;
    }
    int ok = in->dtype == T_F32
        ? vx_transpose_nd_f32_native_validated(
                           in->data, out->data, input_shape, permutation,
                           (uint32_t)in->ndim, (uint32_t)in->numel)
        : transpose_nd_u32((const uint32_t*)in->data,
                           (uint32_t*)out->data, input_shape, permutation,
                           (uint32_t)in->ndim, (uint32_t)in->numel);
    return ok ? 0 : -1;
}

static int concat_tensor(Node* n, T* out, int axis) {
    if (!n || !tensor_plain_32(out) || out->ndim <= 0 ||
        out->ndim > 8 || n->nin <= 0)
        return -1;
    if (axis < 0) axis += out->ndim;
    if (axis < 0 || axis >= out->ndim || out->shape[axis] <= 0) return -1;
    long inner = 1;
    for (int i = axis + 1; i < out->ndim; i++) {
        if (out->shape[i] <= 0 || inner > LONG_MAX / out->shape[i]) return -1;
        inner *= out->shape[i];
    }
    long output_block = (long)out->shape[axis] * inner;
    long outer = output_block > 0 ? out->numel / output_block : 0;
    if (inner <= 0 || outer <= 0 || inner > UINT32_MAX || outer > UINT32_MAX)
        return -1;
    if (out->numel != outer * output_block) return -1;
    int axis_offset = 0;
    for (int input_index = 0; input_index < n->nin; input_index++) {
        T* source = t_find(n->ins[input_index].name);
        if (!tensor_plain_32(source) || source->dtype != out->dtype ||
            source->ndim != out->ndim ||
            source->shape[axis] <= 0 ||
            axis_offset > out->shape[axis] - source->shape[axis])
            return -1;
        for (int dimension = 0; dimension < out->ndim; dimension++) {
            if (dimension != axis && source->shape[dimension] != out->shape[dimension])
                return -1;
        }
        if (source->numel != outer * (long)source->shape[axis] * inner)
            return -1;
        int ok = source->dtype == T_F32
            ? concat_slice_f32(
                source->data, out->data, (uint32_t)outer,
                (uint32_t)source->shape[axis],
                (uint32_t)out->shape[axis], (uint32_t)inner,
                (uint32_t)axis_offset, 0u)
            : concat_slice_u32(
                (const uint32_t*)source->data, (uint32_t*)out->data,
                (uint32_t)outer, (uint32_t)source->shape[axis],
                (uint32_t)out->shape[axis], (uint32_t)inner,
                (uint32_t)axis_offset);
        if (!ok) return -1;
        axis_offset += source->shape[axis];
    }
    return axis_offset == out->shape[axis] ? 0 : -1;
}

static int json_i32_value(const cJSON* value, int32_t* output) {
    double number;
    if (!value || !output || !cJSON_IsNumber(value)) return -1;
    number = value->valuedouble;
    if (!isfinite(number) || number < (double)INT32_MIN ||
        number > (double)INT32_MAX || trunc(number) != number)
        return -1;
    *output = (int32_t)number;
    return 0;
}

static int node_i32_parameter(cJSON* params, const char* key,
                              int32_t default_value, int32_t* output) {
    cJSON* value;
    if (!output) return -1;
    value = params ? cJSON_GetObjectItemCaseSensitive(params, key) : NULL;
    if (!value) {
        *output = default_value;
        return 0;
    }
    return json_i32_value(value, output);
}

static int clip_plain_forward(Node* node) {
    T* input = node_input_default(node);
    T* output = t_find(node->out);
    if (!input || !output || !tensor_plain_32(input) ||
        !tensor_plain_32(output) || input->dtype != output->dtype ||
        !tensor_same_shape(input, output))
        return -1;
    if (input->dtype == T_F32) {
        float minimum = p_flt(node->params, "min", -INFINITY);
        float maximum = p_flt(node->params, "max", INFINITY);
        if (isnan(minimum) || isnan(maximum) || minimum > maximum)
            return -1;
        clip_f32(input->data, output->data, (int)output->numel,
                 minimum, maximum);
        return 0;
    }
    int32_t minimum;
    int32_t maximum;
    if (node_i32_parameter(node->params, "min", INT32_MIN, &minimum) != 0 ||
        node_i32_parameter(node->params, "max", INT32_MAX, &maximum) != 0 ||
        minimum > maximum)
        return -1;
    const int32_t* source = (const int32_t*)input->data;
    int32_t* destination = (int32_t*)output->data;
    for (long index = 0; index < output->numel; index++) {
        int32_t value = source[index];
        if (value < minimum) value = minimum;
        if (value > maximum) value = maximum;
        destination[index] = value;
    }
    return 0;
}

static uint32_t cast_f32_to_u32_mod(float value) {
    union { float value; uint32_t bits; } source = {value};
    uint32_t exponent = (source.bits >> 23u) & 0xffu;
    uint32_t significand;
    uint32_t magnitude;
    int32_t shift;
    if (exponent == 0xffu || exponent < 127u) return 0u;
    significand = (source.bits & 0x7fffffu) | 0x800000u;
    shift = (int32_t)exponent - 127 - 23;
    if (shift >= 32) magnitude = 0u;
    else if (shift >= 0) magnitude = significand << (uint32_t)shift;
    else magnitude = significand >> (uint32_t)(-shift);
    return (source.bits & 0x80000000u) ? 0u - magnitude : magnitude;
}

static int32_t cast_i32_from_u32(uint32_t value) {
    return value <= 0x7fffffffu
        ? (int32_t)value
        : (int32_t)(value - 0x80000000u) - 2147483647 - 1;
}

static int8_t cast_i8_from_u8(uint8_t value) {
    return value <= 127u
        ? (int8_t)value
        : (int8_t)(value - 128u) - 128;
}

static int cast_value_to_i32(const T* input, long index, int32_t* output) {
    if (input->dtype == T_I32) {
        *output = ((const int32_t*)input->data)[index];
        return 0;
    }
    if (input->dtype == T_I8) {
        *output = ((const int8_t*)input->data)[index];
        return 0;
    }
    if (input->dtype == T_U8) {
        *output = ((const uint8_t*)input->data)[index];
        return 0;
    }
    if (input->dtype == T_F32) {
        *output = cast_i32_from_u32(cast_f32_to_u32_mod(input->data[index]));
        return 0;
    }
    return -1;
}

static int cast_forward(Node* node) {
    T* input = node_input_default(node);
    T* output = t_find(node->out);
    cJSON* target = node->params
        ? cJSON_GetObjectItemCaseSensitive(node->params, "to") : NULL;
    int target_dtype;
    if (!input || !output || !input->data || !output->data ||
        input->numel != output->numel || input->numel <= 0 ||
        !target || (target_dtype = graph_declared_dtype(target)) < 0 ||
        target_dtype != output->dtype ||
        (input->dtype != T_F32 && input->dtype != T_I32 &&
         input->dtype != T_I8 && input->dtype != T_U8))
        return -1;
    if (input->dtype == output->dtype) {
        size_t bytes = (size_t)input->numel * input->elem_size;
        if (input->data != output->data) memcpy(output->data, input->data, bytes);
        return 0;
    }
    for (long index = 0; index < output->numel; index++) {
        if (output->dtype == T_F32) {
            int32_t integer;
            if (cast_value_to_i32(input, index, &integer) != 0) return -1;
            output->data[index] = (float)integer;
        } else {
            int32_t integer;
            if (cast_value_to_i32(input, index, &integer) != 0) return -1;
            if (output->dtype == T_I32)
                ((int32_t*)output->data)[index] = integer;
            else if (output->dtype == T_I8)
                ((int8_t*)output->data)[index] =
                    cast_i8_from_u8((uint8_t)integer);
            else if (output->dtype == T_U8)
                ((uint8_t*)output->data)[index] = (uint8_t)integer;
            else
                return -1;
        }
    }
    return 0;
}

static int expand_plain_32_forward(Node* node) {
    T* input = node_input_default(node);
    T* output = t_find(node->out);
    long output_strides[8] = {0};
    long input_strides[8] = {0};
    if (!input || !output || !tensor_plain_32(input) ||
        !tensor_plain_32(output) || input->dtype != output->dtype ||
        output->ndim <= 0 || output->ndim > 8 ||
        tensor_contiguous_strides(output, output_strides) != 0 ||
        tensor_broadcast_strides(input, output, input_strides) != 0)
        return -1;
    const uint32_t* source = (const uint32_t*)input->data;
    uint32_t* destination = (uint32_t*)output->data;
    for (long index = 0; index < output->numel; index++) {
        long input_index = tensor_broadcast_index(
            index, output->ndim, output_strides, input_strides);
        destination[index] = source[input_index];
    }
    return 0;
}

static int gather_f32_i32_forward(Node* node) {
    T* input = node_input_default(node);
    T* indices = nin(node, "indices");
    T* output = t_find(node->out);
    int axis;
    long outer = 1;
    long inner = 1;
    if (!input || !indices || !output || input->dtype != T_F32 ||
        indices->dtype != T_I32 || output->dtype != T_F32 ||
        input->ndim <= 0 || input->ndim > 8 || indices->ndim < 0 ||
        indices->ndim > 8)
        return -1;
    axis = p_int(node->params, "axis", 0);
    if (axis < 0) axis += input->ndim;
    if (axis < 0 || axis >= input->ndim || input->shape[axis] <= 0 ||
        output->ndim != input->ndim - 1 + indices->ndim)
        return -1;
    int output_dimension = 0;
    for (int dimension = 0; dimension < axis; dimension++) {
        if (input->shape[dimension] <= 0 ||
            output->shape[output_dimension++] != input->shape[dimension] ||
            outer > LONG_MAX / input->shape[dimension])
            return -1;
        outer *= input->shape[dimension];
    }
    for (int dimension = 0; dimension < indices->ndim; dimension++) {
        if (indices->shape[dimension] <= 0 ||
            output->shape[output_dimension++] != indices->shape[dimension])
            return -1;
    }
    for (int dimension = axis + 1; dimension < input->ndim; dimension++) {
        if (input->shape[dimension] <= 0 ||
            output->shape[output_dimension++] != input->shape[dimension] ||
            inner > LONG_MAX / input->shape[dimension])
            return -1;
        inner *= input->shape[dimension];
    }
    if (indices->numel <= 0 || outer > LONG_MAX / indices->numel ||
        outer * indices->numel > LONG_MAX / inner ||
        output->numel != outer * indices->numel * inner)
        return -1;
    const int32_t* selections = (const int32_t*)indices->data;
    int axis_size = input->shape[axis];
    for (long index = 0; index < indices->numel; index++) {
        int64_t selected = selections[index];
        if (selected < 0) selected += axis_size;
        if (selected < 0 || selected >= axis_size) return -1;
    }
    for (long outer_index = 0; outer_index < outer; outer_index++) {
        for (long index = 0; index < indices->numel; index++) {
            int64_t selected = selections[index];
            if (selected < 0) selected += axis_size;
            const float* source = input->data +
                ((outer_index * axis_size + selected) * inner);
            float* destination = output->data +
                ((outer_index * indices->numel + index) * inner);
            memcpy(destination, source, (size_t)inner * sizeof(float));
        }
    }
    return 0;
}

static int json_i32_array_item(const cJSON* array, int index, int* output) {
    int32_t parsed;
    if (!array || !cJSON_IsArray(array) ||
        json_i32_value(cJSON_GetArrayItem(array, index), &parsed) != 0)
        return -1;
    *output = (int)parsed;
    return 0;
}

static int slice_plain_32_forward(Node* node) {
    T* input = node_input_default(node);
    T* output = t_find(node->out);
    cJSON* starts_node = node->params
        ? cJSON_GetObjectItemCaseSensitive(node->params, "starts") : NULL;
    cJSON* axes_node = node->params
        ? cJSON_GetObjectItemCaseSensitive(node->params, "axes") : NULL;
    cJSON* steps_node = node->params
        ? cJSON_GetObjectItemCaseSensitive(node->params, "steps") : NULL;
    int starts[8] = {0};
    int steps[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    unsigned seen = 0;
    long input_strides[8] = {0};
    if (!input || !output || !tensor_plain_32(input) ||
        !tensor_plain_32(output) || input->dtype != output->dtype ||
        input->ndim <= 0 || input->ndim > 8 ||
        output->ndim != input->ndim ||
        !cJSON_IsArray(starts_node) || !cJSON_IsArray(axes_node) ||
        !cJSON_IsArray(steps_node))
        return -1;
    int count = cJSON_GetArraySize(starts_node);
    if (count <= 0 || count > input->ndim ||
        cJSON_GetArraySize(axes_node) != count ||
        cJSON_GetArraySize(steps_node) != count ||
        tensor_contiguous_strides(input, input_strides) != 0)
        return -1;
    for (int index = 0; index < count; index++) {
        int start;
        int axis;
        int step;
        if (json_i32_array_item(starts_node, index, &start) != 0 ||
            json_i32_array_item(axes_node, index, &axis) != 0 ||
            json_i32_array_item(steps_node, index, &step) != 0)
            return -1;
        if (axis < 0) axis += input->ndim;
        if (axis < 0 || axis >= input->ndim || step <= 0 || start < 0 ||
            (seen & (1u << (unsigned)axis)))
            return -1;
        starts[axis] = start;
        steps[axis] = step;
        seen |= 1u << (unsigned)axis;
    }
    for (int dimension = 0; dimension < input->ndim; dimension++) {
        if (input->shape[dimension] <= 0 || output->shape[dimension] <= 0)
            return -1;
        if (!(seen & (1u << (unsigned)dimension))) {
            if (output->shape[dimension] != input->shape[dimension])
                return -1;
        } else {
            int64_t last = starts[dimension] +
                (int64_t)(output->shape[dimension] - 1) * steps[dimension];
            if (last < 0 || last >= input->shape[dimension]) return -1;
        }
    }
    const uint32_t* source = (const uint32_t*)input->data;
    uint32_t* destination = (uint32_t*)output->data;
    for (long output_index = 0; output_index < output->numel; output_index++) {
        long remainder = output_index;
        long input_index = 0;
        for (int dimension = input->ndim - 1; dimension >= 0; dimension--) {
            int coordinate = (int)(remainder % output->shape[dimension]);
            remainder /= output->shape[dimension];
            input_index += (starts[dimension] + (long)coordinate * steps[dimension]) *
                input_strides[dimension];
        }
        destination[output_index] = source[input_index];
    }
    return 0;
}

static void concat_tensor_sigmoid(Node* n, T* out, int axis) {
    if (axis < 0) axis += out->ndim;
    long inner = 1;
    for (int i = axis + 1; i < out->ndim; i++) inner *= out->shape[i];
    long outer = out->numel / ((long)out->shape[axis] * inner);
    const float* srcs[MAXIN];
    int axis_lens[MAXIN];
    int nsrc = 0;
    for (int i = 0; i < n->nin && nsrc < MAXIN; i++) {
        T* src = t_find(n->ins[i].name);
        if (!src) continue;
        srcs[nsrc] = src->data;
        axis_lens[nsrc] = src->shape[axis];
        nsrc++;
    }
    vx_fused_concat_sigmoid_f32(srcs, axis_lens, nsrc, out->data, out->shape[axis], outer, inner);
}

static int resize_nearest2d_image(const T* in, T* out) {
    if (!in || !out || in->dtype != T_F32 || out->dtype != T_F32 ||
        in->ndim != 4 || out->ndim != 4 || in->shape[0] != out->shape[0] ||
        in->shape[3] != out->shape[3])
        return -1;
    int n = in->shape[0], h = in->shape[1], w = in->shape[2], c = in->shape[3];
    int oh = out->shape[1], ow = out->shape[2];
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || oh <= 0 || ow <= 0)
        return -1;
    return resize_nearest2d_f32(
        in->data, out->data, (uint32_t)n, (uint32_t)h, (uint32_t)w,
        (uint32_t)c, (uint32_t)oh, (uint32_t)ow) ? 0 : -1;
}

static void maxpool2d_f32_image(const T* in, T* out, int ky, int kx, int sy, int sx, int py, int px) {
    int n = in->shape[0], h = in->shape[1], w = in->shape[2], c = in->shape[3];
    int oh = out->shape[1], ow = out->shape[2];
    vx_maxpool2d_f32(in->data, out->data, n, h, w, c, oh, ow, ky, kx, sy, sx, py, px);
}

static void conv2d_pointwise_f32_image(int node_idx, const T* in, T* out, const T* wt, const T* bias,
                                      const T* add, int relu) {
    int n = in->shape[0], h = in->shape[1], w = in->shape[2], c = in->shape[3];
    int out_c = out->shape[3];
    long pixels = (long)n * h * w;
    vx_conv2d_pointwise_f32(node_idx, in->data, out->data, wt->data,
                                 bias ? bias->data : NULL,
                                 add ? add->data : NULL,
                                 pixels, c, out_c, relu);
}

static int conv2d_depthwise_pointwise_f32_image(int dw_node_idx, int pw_node_idx,
                                               const T* in, T* out,
                                               const T* dw_wt, const T* dw_bias,
                                               const T* pw_wt, const T* pw_bias,
                                               const T* add,
                                               int sy, int sx, const int* pads,
                                               int dw_relu, int pw_relu) {
    if (!in || !out || !dw_wt || !pw_wt || in->ndim != 4 || out->ndim != 4 ||
        dw_wt->ndim < 4 || pw_wt->ndim < 4) return 0;
    return vx_conv2d_depthwise_pointwise_f32(dw_node_idx, pw_node_idx,
                                                  in->data, out->data,
                                                  dw_wt->data, dw_bias ? dw_bias->data : NULL,
                                                  pw_wt->data, pw_bias ? pw_bias->data : NULL,
                                                  add ? add->data : NULL,
                                                  in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                                                  out->shape[1], out->shape[2], out->shape[3],
                                                  dw_wt->shape[0], dw_wt->shape[1], dw_wt->shape[2], dw_wt->shape[3],
                                                  pw_wt->shape[0], pw_wt->shape[1], pw_wt->shape[2],
                                                  sy, sx, pads, dw_relu, pw_relu);
}

static void conv2d_f32_image(int node_idx, const T* in, T* out, const T* wt, const T* bias, const T* add,
                            int sy, int sx, const int* pads, int groups, int dy, int dx, int relu) {
    if (!in || !out || !wt || in->ndim != 4 || out->ndim != 4 || wt->ndim < 4) return;
    if (dy <= 0) dy = 1;
    if (dx <= 0) dx = 1;
    int kh = wt->shape[0], kw = wt->shape[1];
    int n = in->shape[0], h = in->shape[1], w = in->shape[2], c = in->shape[3];
    int oh = out->shape[1], ow = out->shape[2], out_c = out->shape[3];
    if (groups == c && wt->shape[2] == c && out_c == c * wt->shape[3]) {
        vx_conv2d_depthwise_f32(in->data, out->data, wt->data, bias ? bias->data : NULL,
                                     n, h, w, c, oh, ow, out_c,
                                     kh, kw, wt->shape[2], wt->shape[3],
                                     sy, sx, pads, dy, dx, relu);
    } else if (groups == 1 && kh == 1 && kw == 1 && sy == 1 && sx == 1 &&
               pads[0] == 0 && pads[1] == 0 && pads[2] == 0 && pads[3] == 0) {
        conv2d_pointwise_f32_image(node_idx, in, out, wt, bias, add, relu);
    } else {
        vx_conv2d_generic_f32(node_idx, in->data, out->data, wt->data, bias ? bias->data : NULL,
                                  n, h, w, c, oh, ow, out_c, kh, kw, wt->shape[2],
                                  groups, sy, sx, pads, dy, dx, relu);
    }
}

static int quantized_weight_i32(const T* wt, long idx) {
    if (wt->dtype == T_U8) return ((const unsigned char*)wt->data)[idx];
    return ((const signed char*)wt->data)[idx];
}

static float* qconv_weight_only_dequant_cache(int node_idx, const T* wt, const T* wscale,
                                              const T* wzp, const char* layout) {
    if (g_qwcache[node_idx]) return g_qwcache[node_idx];
    float* out = (float*)malloc((size_t)wt->numel * sizeof(float));
    if (!out) return NULL;
    int ohwi = layout && !strcmp(layout, "OHWI");
    int hwo = layout && (!strcmp(layout, "1HWO") || !strcmp(layout, "1HWM"));
    if (ohwi) {
        int out_c = wt->shape[0], kh = wt->shape[1], kw = wt->shape[2], in_c = wt->shape[3];
        for (int oc = 0; oc < out_c; oc++) {
            float ws = tensor_scalar_f32(wscale, wscale->numel == 1 ? 0 : oc);
            int wz = wzp ? (int)tensor_scalar_any(wzp, wzp->numel == 1 ? 0 : oc) : 0;
            for (int ky = 0; ky < kh; ky++) {
                for (int kx = 0; kx < kw; kx++) {
                    for (int ic = 0; ic < in_c; ic++) {
                        long idx = (((long)oc * kh + ky) * kw + kx) * in_c + ic;
                        int wv = quantized_weight_i32(wt, idx) - wz;
                        out[idx] = (float)wv * ws;
                    }
                }
            }
        }
    } else if (hwo) {
        int one = wt->shape[0], kh = wt->shape[1], kw = wt->shape[2], out_c = wt->shape[3];
        if (one != 1) { free(out); return NULL; }
        for (int oc = 0; oc < out_c; oc++) {
            float ws = tensor_scalar_f32(wscale, wscale->numel == 1 ? 0 : oc);
            int wz = wzp ? (int)tensor_scalar_any(wzp, wzp->numel == 1 ? 0 : oc) : 0;
            for (int ky = 0; ky < kh; ky++) {
                for (int kx = 0; kx < kw; kx++) {
                    long idx = (((long)0 * kh + ky) * kw + kx) * out_c + oc;
                    int wv = quantized_weight_i32(wt, idx) - wz;
                    out[idx] = (float)wv * ws;
                }
            }
        }
    } else {
        int out_c = wt->shape[0];
        long per_oc = wt->numel / out_c;
        for (int oc = 0; oc < out_c; oc++) {
            float ws = tensor_scalar_f32(wscale, wscale->numel == 1 ? 0 : oc);
            int wz = wzp ? (int)tensor_scalar_any(wzp, wzp->numel == 1 ? 0 : oc) : 0;
            long base = (long)oc * per_oc;
            for (long i = 0; i < per_oc; i++) {
                long idx = base + i;
                int wv = quantized_weight_i32(wt, idx) - wz;
                out[idx] = (float)wv * ws;
            }
        }
    }
    g_qwcache[node_idx] = out;
    return out;
}

static float* f16_tensor_f32_cache(float** cache, int node_idx, const T* t) {
    if (!t || t->dtype != T_F16 || node_idx < 0 || node_idx >= MAXN) return NULL;
    if (cache[node_idx]) return cache[node_idx];
    float* out = (float*)malloc((size_t)t->numel * sizeof(float));
    if (!out) return NULL;
    for (long i = 0; i < t->numel; i++) out[i] = tensor_scalar_f32(t, i);
    cache[node_idx] = out;
    return out;
}

static int layout_is(const char* layout, const char* want) {
    return layout && want && !strcmp(layout, want);
}

static T* prepare_conv_weight_image(int node_idx, T* wt, cJSON* params, int in_c, int out_c, int groups, T* tmp) {
    if (!wt || wt->ndim != 4 || node_idx < 0 || node_idx >= MAXN) return NULL;
    const char* layout = p_str(params, "weight_layout", groups == in_c ? "1HWO" : "OHWI");

    if (layout_is(layout, "HWIO") || layout_is(layout, "HWCM")) {
        if (wt->dtype == T_F32) return wt;
        if (wt->dtype != T_F16) return NULL;
        float* wf = f16_tensor_f32_cache(g_conv_wcache, node_idx, wt);
        if (!wf) return NULL;
        *tmp = *wt;
        tmp->data = wf;
        tmp->dtype = T_F32;
        tmp->elem_size = sizeof(float);
        return tmp;
    }

    int conv_ohwi = layout_is(layout, "OHWI");
    int dw_1hwo = layout_is(layout, "1HWO") || layout_is(layout, "1HWM");
    if (!conv_ohwi && !dw_1hwo) return NULL;

    if (!g_conv_wcache[node_idx]) {
        float* out = (float*)malloc((size_t)wt->numel * sizeof(float));
        if (!out) return NULL;
        if (conv_ohwi) {
            int oc_n = wt->shape[0], kh = wt->shape[1], kw = wt->shape[2], ic_n = wt->shape[3];
            if (groups != 1 || oc_n != out_c || ic_n != in_c) { free(out); return NULL; }
            for (int oc = 0; oc < oc_n; oc++) {
                for (int ky = 0; ky < kh; ky++) {
                    for (int kx = 0; kx < kw; kx++) {
                        for (int ic = 0; ic < ic_n; ic++) {
                            long src = (((long)oc * kh + ky) * kw + kx) * ic_n + ic;
                            long dst = (((long)ky * kw + kx) * ic_n + ic) * oc_n + oc;
                            out[dst] = tensor_scalar_f32(wt, src);
                        }
                    }
                }
            }
        } else {
            int one = wt->shape[0], kh = wt->shape[1], kw = wt->shape[2], oc_n = wt->shape[3];
            if (one != 1 || groups != in_c || oc_n != out_c || in_c <= 0 || out_c % in_c) {
                free(out);
                return NULL;
            }
            int mult = out_c / in_c;
            for (int ky = 0; ky < kh; ky++) {
                for (int kx = 0; kx < kw; kx++) {
                    for (int ch = 0; ch < in_c; ch++) {
                        for (int m = 0; m < mult; m++) {
                            int oc = ch * mult + m;
                            long src = (((long)0 * kh + ky) * kw + kx) * out_c + oc;
                            long dst = (((long)ky * kw + kx) * in_c + ch) * mult + m;
                            out[dst] = tensor_scalar_f32(wt, src);
                        }
                    }
                }
            }
        }
        g_conv_wcache[node_idx] = out;
    }

    *tmp = *wt;
    tmp->data = g_conv_wcache[node_idx];
    tmp->dtype = T_F32;
    tmp->elem_size = sizeof(float);
    if (conv_ohwi) {
        tmp->shape[0] = wt->shape[1];
        tmp->shape[1] = wt->shape[2];
        tmp->shape[2] = wt->shape[3];
        tmp->shape[3] = wt->shape[0];
    } else {
        tmp->shape[0] = wt->shape[1];
        tmp->shape[1] = wt->shape[2];
        tmp->shape[2] = in_c;
        tmp->shape[3] = out_c / in_c;
    }
    return tmp;
}

int prepack_conv_weights(void) {
    double t0 = g_debug ? volvoxai_engine_now_ms() : 0.0;
    int conv_weights = 0, biases = 0, pointwise_packs = 0, igemm_indirs = 0;
    for (int i = 0; i < g_nn; i++) {
        Node* n = &g_n[i];
        if (strcmp(n->op, "Conv2D")) continue;
        T* w = nin(n, "weight");
        T* b = nin(n, "bias");
        T* in = nin(n, "input");
        T* out = t_find(n->out);
        T tmp;
        int groups = p_int(n->params, "groups", 1);
        T* wp = (w && in && out) ? prepare_conv_weight_image(i, w, n->params, in->shape[3], out->shape[3], groups, &tmp) : NULL;
        if (wp && wp != w) {
            if (!g_conv_wcache[i]) return -1;
            conv_weights++;
        }
        if (wp && in && out && groups == 1 && wp->ndim >= 4 &&
            wp->shape[0] == 1 && wp->shape[1] == 1 &&
            vx_pwf32_pack_cache(i, wp->data, in->shape[3], out->shape[3])) {
            pointwise_packs++;
        } else if (wp && in && out && groups == 1 && wp->ndim >= 4 &&
                   wp->shape[0] * wp->shape[1] > 1 && out->shape[3] >= 16) {
            int py = p_int(n->params, "padding_y", p_int(n->params, "padding", 0));
            int px = p_int(n->params, "padding_x", p_int(n->params, "padding", 0));
            int stride[2] = {1, 1}, dilation[2] = {1, 1}, pads[4] = {py, px, py, px};
            p_iarr(n->params, "stride", stride, 2, 1);
            p_iarr(n->params, "dilation", dilation, 2, 1);
            if (cJSON_GetObjectItem(n->params, "pads")) p_iarr(n->params, "pads", pads, 4, 0);
            if (vx_f32_igemm_indirection_cache(i, in->data,
                                               in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                                               out->shape[1], out->shape[2], wp->shape[0], wp->shape[1],
                                               stride[0], stride[1], pads, dilation[0], dilation[1])) {
                igemm_indirs++;
            }
        }
        if (b && b->dtype == T_F16) {
            if (!f16_tensor_f32_cache(g_f16bcache, i, b)) return -1;
            biases++;
        }
    }
    if (g_debug && (conv_weights || biases || pointwise_packs || igemm_indirs)) {
        fprintf(stderr, "[debug] prepack_conv weights=%d biases=%d pointwise_packs=%d igemm_indirs=%d %.3f ms\n",
                conv_weights, biases, pointwise_packs, igemm_indirs, volvoxai_engine_now_ms() - t0);
    }
    return 0;
}

static int qconv2d_weight_only_f32(int node_idx, const T* in, T* out, const T* wt,
                                   const T* wscale, const T* wzp, const T* bias,
                                   int sy, int sx, const int* pads, int groups,
                                   int dy, int dx, int relu, cJSON* params,
                                   const char* weight_layout) {
    float* w_deq = qconv_weight_only_dequant_cache(
        node_idx, wt, wscale, wzp, weight_layout);
    if (!w_deq) return -1;
    T wraw = *wt;
    wraw.data = w_deq;
    wraw.dtype = T_F32;
    wraw.elem_size = sizeof(float);
    T wtmp;
    T* wp = prepare_conv_weight_image(node_idx, &wraw, params, in->shape[3], out->shape[3], groups, &wtmp);
    if (!wp) return -1;
    conv2d_f32_image(node_idx, in, out, wp, bias, NULL, sy, sx, pads, groups, dy, dx, relu);
    return 0;
}
