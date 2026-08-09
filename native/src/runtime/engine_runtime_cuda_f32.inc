#if VOLVOXAI_ENABLE_CUDA
/* CUDA keeps complete activation allocations resident while transformer
 * prefix/row calls pass interior host pointers as device-buffer views.  Limit
 * this windowing to the canonical [1, sequence, width] layout; every other
 * shape retains the complete-tensor kernel and its existing semantics. */
static int cuda_partial_sequence_rows(const T* tensor, int width,
                                      int* row_start, int* row_count) {
    long rows;
    if (g_active_row < 0 && g_prefix_rows <= 0) return 0;
    if (!tensor || !row_start || !row_count || tensor->ndim != 3 ||
        tensor->shape[0] != 1 || tensor->shape[1] <= 0 || width <= 0 ||
        tensor->shape[2] != width || tensor->numel % width) return -1;
    rows = tensor->numel / width;
    if (rows != tensor->shape[1] || rows > INT_MAX) return -1;
    seq_range(rows, row_start, row_count);
    return *row_start >= 0 && *row_count > 0 &&
           (long)*row_start + *row_count <= rows ? 1 : -1;
}

static int try_cuda_graph_embedding_partial(Node* n, T* in, T* output) {
    T* tokens;
    T* weight;
    const int32_t* ids;
    int d_model;
    int vocab;
    int row_start;
    int row_count;
    if (g_active_row < 0 && g_prefix_rows <= 0) return 0;
    tokens = node_data_input(n, in);
    weight = nin(n, "weight");
    if (!tokens || !weight || !output || tokens->dtype != T_I32 ||
        weight->dtype != T_F32 || output->dtype != T_F32 ||
        tokens->ndim != 2 || tokens->shape[0] != 1 ||
        output->ndim != 3 || output->shape[0] != 1 ||
        output->shape[1] != tokens->shape[1] || weight->ndim <= 0)
        return -1;
    materialize_tensor_f32(weight);
    d_model = weight->shape[weight->ndim - 1];
    vocab = d_model > 0 && weight->numel % d_model == 0
        ? (int)(weight->numel / d_model) : 0;
    if (d_model <= 0 || vocab <= 0 || output->shape[2] != d_model ||
        output->numel != tokens->numel * (long)d_model ||
        tokens->numel <= 0 || tokens->numel > INT_MAX) return -1;
    seq_range(tokens->numel, &row_start, &row_count);
    if (row_start < 0 || row_count <= 0 ||
        (long)row_start + row_count > tokens->numel) return -1;
    ids = (const int32_t*)tokens->data;
    for (int row = row_start; row < row_start + row_count; row++)
        if (ids[row] < 0 || ids[row] >= vocab) return -1;
    return cuda_graph_embedding_f32(ids + row_start, weight->data,
            output->data + (long)row_start * d_model, row_count, d_model,
            vocab) ? 1 : -1;
}

static int try_cuda_graph_sdpa_partial(Node* n, T* in, T* output) {
    T* qkv;
    T* mask;
    int d_model;
    int batch;
    int sequence;
    int heads;
    int head_dim;
    int mask_mode;
    int query_start;
    int query_count;
    int causal;
    float scale;
    if (g_active_row < 0 && g_prefix_rows <= 0) return 0;
#if VOLVOXAI_ENABLE_TRAINING
    if (g_native_training_mode) return 0;
#endif
    qkv = nin(n, "qkv");
    if (!qkv) qkv = node_data_input(n, in);
    mask = nin(n, "mask");
    if (!qkv || !output || (qkv->ndim != 2 && qkv->ndim != 3) ||
        output->ndim != qkv->ndim) return -1;
    materialize_tensor_f32(qkv);
    d_model = output->shape[output->ndim - 1];
    batch = qkv->ndim == 3 ? qkv->shape[0] : 1;
    sequence = qkv->shape[qkv->ndim - 2];
    heads = p_int(n->params, "heads", 8);
    if (qkv->dtype != T_F32 || d_model <= 0 || batch <= 0 ||
        sequence <= 0 || heads <= 0 || d_model % heads ||
        qkv->shape[qkv->ndim - 1] != 3 * d_model ||
        output->shape[output->ndim - 2] != sequence ||
        qkv->numel != (long)batch * sequence * 3 * d_model ||
        output->numel != (long)batch * sequence * d_model) return -1;
    head_dim = d_model / heads;
    mask_mode = attention_mask_mode(mask, batch, sequence, sequence);
    causal = p_int(n->params, "causal", 1) ? 1 : 0;
    if (head_dim <= 0 || head_dim > 64 || mask_mode < 0) return -1;
    /* A non-causal prefix depends on rows that the prefix contract has not
     * produced yet, so retain the correct complete-tensor path for it. */
    if (!causal && g_prefix_rows > 0 && g_prefix_rows < sequence) return 0;
    seq_range(sequence, &query_start, &query_count);
    if (query_start < 0 || query_count <= 0 ||
        query_start + query_count > sequence) return -1;
    scale = p_flt(n->params, "scale", 1.0f / sqrtf((float)head_dim));
    return cuda_graph_sdpa_range_f32(qkv->data,
            mask ? (const int32_t*)mask->data : NULL,
            mask ? mask->numel : 0, output->data, sequence, d_model, heads,
            head_dim, batch, scale, causal, mask_mode, query_start,
            query_count) ? 1 : -1;
}

static int cuda_runtime_output_strides(const T* output,
                                       uint32_t strides[8]) {
    uint64_t stride = 1u;
    if (!output || !strides || output->ndim <= 0 || output->ndim > 8 ||
        output->numel <= 0 || output->numel > UINT32_MAX)
        return 0;
    memset(strides, 0, 8u * sizeof(uint32_t));
    for (int dimension = output->ndim - 1; dimension >= 0; dimension--) {
        if (output->shape[dimension] <= 0 || stride > UINT32_MAX)
            return 0;
        strides[dimension] = (uint32_t)stride;
        stride *= (uint32_t)output->shape[dimension];
        if (stride > UINT32_MAX) return 0;
    }
    return stride == (uint64_t)output->numel;
}

static int cuda_runtime_cast_dtype_supported(int dtype) {
    return dtype == T_F32 || dtype == T_I32 ||
        dtype == T_I8 || dtype == T_U8;
}

static int try_cuda_graph_compare_i32(Node* n, T* in, T* output,
                                      int operation) {
    T* a = nin(n, "a");
    T* b = nin(n, "b");
    uint32_t output_strides[8] = {0};
    uint32_t a_strides[8] = {0};
    uint32_t b_strides[8] = {0};
    if (!a) a = node_data_input(n, in);
    if (!a || !b || !output || a->dtype != T_I32 ||
        b->dtype != T_I32 || output->dtype != T_I32 ||
        gpu_broadcast_strides(a, output, a_strides) != 0 ||
        gpu_broadcast_strides(b, output, b_strides) != 0 ||
        !cuda_runtime_output_strides(output, output_strides))
        return 0;
    return cuda_graph_compare_i32(
        (const int32_t*)a->data, a->numel,
        (const int32_t*)b->data, b->numel,
        (int32_t*)output->data, output->numel,
        output_strides, a_strides, b_strides, output->ndim, operation);
}

static int try_cuda_graph_argmax_f32_i32(Node* n, T* in, T* output) {
    T* input = in ? in : node_data_input(n, NULL);
    uint64_t outer = 1u;
    uint64_t inner = 1u;
    uint64_t output_elements;
    uint64_t input_elements;
    int axis;
    int keepdims;
    if (!n || !input || !output || !input->data || !output->data ||
        input->dtype != T_F32 || output->dtype != T_I32 ||
        output->elem_size != sizeof(int32_t) ||
        input->ndim <= 0 || input->ndim > 8)
        return 0;
    axis = normalize_axis(p_int(n->params, "axis", 0), input->ndim);
    keepdims = p_int(n->params, "keepdims", 1);
    if (axis < 0 || axis >= input->ndim ||
        (keepdims != 0 && keepdims != 1) ||
        p_int(n->params, "select_last_index", 0) != 0 ||
        input->shape[axis] <= 0)
        return 0;
    for (int dimension = 0; dimension < axis; dimension++) {
        uint32_t extent;
        if (input->shape[dimension] <= 0) return 0;
        extent = (uint32_t)input->shape[dimension];
        if (outer > UINT32_MAX / extent) return 0;
        outer *= extent;
    }
    for (int dimension = axis + 1;
         dimension < input->ndim; dimension++) {
        uint32_t extent;
        if (input->shape[dimension] <= 0) return 0;
        extent = (uint32_t)input->shape[dimension];
        if (inner > UINT32_MAX / extent) return 0;
        inner *= extent;
    }
    if (outer > UINT32_MAX / inner) return 0;
    output_elements = outer * inner;
    if ((uint64_t)input->shape[axis] >
        UINT32_MAX / output_elements)
        return 0;
    input_elements =
        output_elements * (uint32_t)input->shape[axis];
    if (input->numel <= 0 ||
        (uint64_t)input->numel != input_elements ||
        output->numel <= 0 ||
        (uint64_t)output->numel != output_elements)
        return 0;
    if (keepdims) {
        if (output->ndim != input->ndim) return 0;
        for (int dimension = 0;
             dimension < input->ndim; dimension++) {
            int expected =
                dimension == axis ? 1 : input->shape[dimension];
            if (output->shape[dimension] != expected) return 0;
        }
    } else {
        if (output->ndim != input->ndim - 1) return 0;
        for (int dimension = 0;
             dimension < input->ndim; dimension++) {
            int output_dimension;
            if (dimension == axis) continue;
            output_dimension =
                dimension < axis ? dimension : dimension - 1;
            if (output->shape[output_dimension] !=
                input->shape[dimension])
                return 0;
        }
    }
    materialize_tensor_f32(input);
    return cuda_graph_argmax_f32(
        input->data, (int32_t*)output->data,
        (uint32_t)outer, (uint32_t)input->shape[axis],
        (uint32_t)inner);
}

static int try_cuda_graph_split_i32(Node* n, T* in) {
    T* input = node_data_input(n, in);
    int axis;
    int inner;
    int offset = 0;
    if (!input || input->dtype != T_I32 || input->ndim <= 0 ||
        input->ndim > 8 || n->nout <= 0)
        return 0;
    axis = normalize_axis(p_int(n->params, "axis", 0), input->ndim);
    inner = tensor_inner_after_axis(input, axis);
    if (axis < 0 || axis >= input->ndim || input->shape[axis] <= 0 ||
        inner <= 0)
        return 0;
    /* n->outs is already in graph declaration order; do not sort keys such as
     * out10 lexically ahead of out2. */
    for (int output_index = 0; output_index < n->nout; output_index++) {
        T* destination = t_find(n->outs[output_index].name);
        if (!destination || destination->dtype != T_I32 ||
            destination->ndim != input->ndim ||
            destination->shape[axis] <= 0 ||
            offset > input->shape[axis] - destination->shape[axis])
            return 0;
        for (int dimension = 0; dimension < input->ndim; dimension++) {
            if (dimension != axis &&
                destination->shape[dimension] != input->shape[dimension])
                return 0;
        }
        offset += destination->shape[axis];
    }
    if (offset != input->shape[axis]) return 0;
    offset = 0;
    for (int output_index = 0; output_index < n->nout; output_index++) {
        T* destination = t_find(n->outs[output_index].name);
        int split_size = destination->shape[axis];
        if (!cuda_graph_split_32(
                input->data, destination->data,
                input->numel, destination->numel, inner,
                split_size, input->shape[axis], offset))
            return 0;
        offset += split_size;
    }
    return 1;
}

static int try_cuda_graph_node(Node* n, int idx, const char** node_backend) {
    T* o;
    T* in;
    cJSON* p;
    const char* op;
    (void)idx;
    if (!n || !node_backend) return 0;
    o = t_find(n->out);
    in = nin(n, "input");
    p = n->params;
    op = n->op;
    if (!o) return 0;

    if (!strcmp(op, "ArgMax")) {
        if (try_cuda_graph_argmax_f32_i32(n, in, o)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "QBatchMatMul")) {
        QBatchMatMulPhysicalDescriptor descriptor;
        if (qbatch_matmul_physical_validate(n, o, &descriptor) != 0)
            return 0;
        if (cuda_graph_qbatch_matmul_i8u8(
                descriptor.a->data,
                descriptor.a->shape, descriptor.a->ndim,
                descriptor.a->quantization.scale,
                descriptor.a->quantization.zero_point,
                descriptor.a_dtype,
                descriptor.b->data,
                descriptor.b->shape, descriptor.b->ndim,
                descriptor.b->quantization.scale,
                descriptor.b->quantization.zero_point,
                descriptor.b_dtype,
                descriptor.output->data,
                descriptor.output->shape, descriptor.output->ndim,
                descriptor.output->quantization.scale,
                descriptor.output->quantization.zero_point,
                descriptor.output_dtype)) {
            *node_backend = "cuda-qbatch-matmul-w8a8";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "BatchMatMul")) {
        T* a = nin(n, "a");
        T* b = nin(n, "b");
        if (!a || !b || a->dtype != T_F32 || b->dtype != T_F32 ||
            o->dtype != T_F32)
            return 0;
        materialize_tensor_f32(a);
        materialize_tensor_f32(b);
        if (cuda_graph_batch_matmul_f32(
                a->data, a->shape, a->ndim,
                b->data, b->shape, b->ndim,
                o->data, o->shape, o->ndim)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Equal") || !strcmp(op, "GreaterOrEqual")) {
        if (try_cuda_graph_compare_i32(
                n, in, o, !strcmp(op, "GreaterOrEqual") ? 1 : 0)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Not")) {
        T* source = node_data_input(n, in);
        if (source && source->dtype == T_I32 && o->dtype == T_I32 &&
            source->numel == o->numel &&
            tensor_same_shape(source, o) &&
            cuda_graph_not_i32(
                (const int32_t*)source->data, (int32_t*)o->data,
                o->numel)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Clip") && o->dtype == T_I32) {
        T* source = node_data_input(n, in);
        if (source && source->dtype == T_I32 &&
            source->numel == o->numel && tensor_same_shape(source, o) &&
            cuda_graph_clip_i32(
                (const int32_t*)source->data, (int32_t*)o->data,
                o->numel, p_int(p, "min", INT_MIN),
                p_int(p, "max", INT_MAX))) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Where")) {
        T* condition = nin(n, "condition");
        T* a = nin(n, "x");
        T* b = nin(n, "y");
        if (!condition) condition = nin(n, "cond");
        if (!a) a = nin(n, "a");
        if (!b) b = nin(n, "b");
        if (condition && a && b && condition->dtype == T_I32 &&
            (o->dtype == T_F32 || o->dtype == T_I32) &&
            a->dtype == o->dtype && b->dtype == o->dtype &&
            tensor_same_shape(condition, o) &&
            tensor_same_shape(a, o) && tensor_same_shape(b, o)) {
            if (o->dtype == T_F32) {
                materialize_tensor_f32(a);
                materialize_tensor_f32(b);
            }
            if (cuda_graph_where_32(
                    (const int32_t*)condition->data,
                    a->data, b->data, o->data, o->numel)) {
                *node_backend = "cuda-graph";
                return 1;
            }
        }
        return 0;
    }

    if ((!strcmp(op, "Expand") || !strcmp(op, "Broadcast")) &&
        (o->dtype == T_F32 || o->dtype == T_I32)) {
        T* source = in ? in : nin(n, "data");
        if (source && source->dtype == o->dtype) {
            if (source->dtype == T_F32) materialize_tensor_f32(source);
            if (cuda_graph_expand_32(
                    source->data, o->data, source->shape, source->ndim,
                    o->shape, o->ndim)) {
                *node_backend = "cuda-graph";
                return 1;
            }
        }
        return 0;
    }

    if (!strcmp(op, "Transpose") && o->dtype == T_I32) {
        T* source = in ? in : node_data_input(n, NULL);
        int permutation[8];
        if (!source || source->dtype != T_I32 ||
            source->ndim <= 0 || source->ndim > 8 ||
            source->numel != o->numel)
            return 0;
        for (int dimension = 0; dimension < source->ndim; dimension++)
            permutation[dimension] = source->ndim - 1 - dimension;
        if (p && cJSON_GetObjectItem(p, "perm"))
            p_iarr(p, "perm", permutation, source->ndim, 0);
        if (cuda_graph_transpose_32(
                source->data, o->data, source->shape,
                permutation, source->ndim)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Cast")) {
        T* source = in ? in : node_data_input(n, NULL);
        int ok = 0;
        if (!source || source->numel != o->numel || o->numel <= 0 ||
            !cuda_runtime_cast_dtype_supported(source->dtype) ||
            !cuda_runtime_cast_dtype_supported(o->dtype))
            return 0;
        if (source->dtype == T_F32) materialize_tensor_f32(source);
        if (source->dtype == o->dtype && source->elem_size == sizeof(uint32_t))
            ok = cuda_graph_copy_32(source->data, o->data, o->numel);
        else
            ok = cuda_graph_cast_typed(
                source->data, source->dtype, o->data, o->dtype,
                o->numel);
        if (ok) {
            *node_backend = source->dtype == o->dtype
                ? "cuda-graph-copy" : "cuda-graph";
            return 1;
        }
        return 0;
    }

    if ((!strcmp(op, "Reshape") || !strcmp(op, "Flatten") ||
         !strcmp(op, "Squeeze") || !strcmp(op, "Unsqueeze") ||
         !strcmp(op, "Identity")) && o->dtype == T_I32) {
        T* source = in ? in : node_data_input(n, NULL);
        if (source && source->dtype == T_I32 &&
            source->numel == o->numel && o->numel > 0 &&
            cuda_graph_copy_32(source->data, o->data, o->numel)) {
            *node_backend = "cuda-graph-copy";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Split") && o->dtype == T_I32) {
        if (try_cuda_graph_split_i32(n, in)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Concat") && o->dtype == T_I32) {
        int axis = p_int(p, "axis", 1);
        const void* inputs[MAXIN];
        long sizes[MAXIN];
        int input_axes[MAXIN];
        long total = 0;
        long summed_axis = 0;
        int inner;
        if (axis < 0) axis += o->ndim;
        if (axis < 0 || axis >= o->ndim || o->shape[axis] <= 0)
            return 0;
        for (int input_index = 0;
             input_index < n->nin; input_index++) {
            T* source = t_find(n->ins[input_index].name);
            if (!source || source->dtype != T_I32 ||
                source->ndim != o->ndim || source->shape[axis] <= 0)
                return 0;
            for (int dimension = 0; dimension < o->ndim; dimension++) {
                if (dimension != axis &&
                    source->shape[dimension] != o->shape[dimension])
                    return 0;
            }
            inputs[input_index] = source->data;
            sizes[input_index] = source->numel;
            input_axes[input_index] = source->shape[axis];
            total += source->numel;
            summed_axis += source->shape[axis];
        }
        inner = tensor_inner_after_axis(o, axis);
        if (total == o->numel && summed_axis == o->shape[axis] &&
            inner > 0 && cuda_graph_concat_32(
                inputs, sizes, input_axes, n->nin, o->data,
                o->shape[axis], inner)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Slice") &&
        (o->dtype == T_F32 || o->dtype == T_I32)) {
        T* source = in ? in : nin(n, "data");
        int raw_starts[8] = {0};
        int raw_steps[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        int raw_axes[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        int starts[8] = {0};
        int steps[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        cJSON* starts_node = p ? cJSON_GetObjectItem(p, "starts") : NULL;
        cJSON* steps_node = p ? cJSON_GetObjectItem(p, "steps") : NULL;
        cJSON* axes_node = p ? cJSON_GetObjectItem(p, "axes") : NULL;
        int axis_count;
        unsigned int seen_axes = 0u;
        if (!source || source->dtype != o->dtype ||
            source->ndim <= 0 || source->ndim > 8 ||
            o->ndim != source->ndim ||
            !starts_node || !cJSON_IsArray(starts_node))
            return 0;
        axis_count = cJSON_GetArraySize(starts_node);
        if (axis_count <= 0 || axis_count > source->ndim ||
            (steps_node && (!cJSON_IsArray(steps_node) ||
                            cJSON_GetArraySize(steps_node) != axis_count)) ||
            (axes_node && (!cJSON_IsArray(axes_node) ||
                           cJSON_GetArraySize(axes_node) != axis_count)))
            return 0;
        p_iarr(p, "starts", raw_starts, axis_count, 0);
        p_iarr(p, "steps", raw_steps, axis_count, 1);
        p_iarr(p, "axes", raw_axes, axis_count, 0);
        for (int index = 0; index < axis_count; index++) {
            int axis = axes_node ? raw_axes[index] : index;
            int start;
            if (axis < 0) axis += source->ndim;
            if (axis < 0 || axis >= source->ndim ||
                raw_steps[index] <= 0 ||
                (seen_axes & (1u << (unsigned int)axis)))
                return 0;
            seen_axes |= 1u << (unsigned int)axis;
            start = raw_starts[index] < 0
                ? raw_starts[index] + source->shape[axis]
                : raw_starts[index];
            if (start < 0) return 0;
            starts[axis] = start;
            steps[axis] = raw_steps[index];
        }
        if (source->dtype == T_F32) materialize_tensor_f32(source);
        if (cuda_graph_slice_32(
                source->data, o->data, source->shape, source->ndim,
                o->shape, o->ndim, starts, steps)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (o->dtype != T_F32) return 0;

    if (!strcmp(op, "Conv2D")) {
        T* weight = nin(n, "weight");
        T* bias = nin(n, "bias");
        const GraphNodeFusion* conv_add_fusion =
            graph_opt_node_fusion(idx, GRAPH_FUSION_CONV_ADD);
        T* residual = NULL;
        int stride_y, stride_x, padding_y, padding_x, dilation_y, dilation_x;
        int pads[4];
        int groups;
        int relu;
        T weight_f32;
        T bias_f32;
        T* prepared_weight;
        T* prepared_bias = bias;
        if (!in || !weight || in->dtype != T_F32 ||
            (weight->dtype != T_F32 && weight->dtype != T_F16) ||
            in->ndim != 4 || weight->ndim != 4 || o->ndim != 4 ||
            in->shape[0] <= 0 || o->shape[0] != in->shape[0])
            return 0;
        p_pair(p, "stride", 1, &stride_y, &stride_x);
        p_pair(p, "padding", 0, &padding_y, &padding_x);
        p_pair(p, "dilation", 1, &dilation_y, &dilation_x);
        pads[0] = padding_y;
        pads[1] = padding_x;
        pads[2] = padding_y;
        pads[3] = padding_x;
        if (p && cJSON_GetObjectItem(p, "pads")) p_iarr(p, "pads", pads, 4, 0);
        groups = p_int(p, "groups", 1);
        relu = n->fuse_relu6 ? 2 : p_int(p, "relu", 0);
        if (pads[0] < 0 || pads[1] < 0 || relu < 0 || relu > 2)
            return 0;
        if (conv_add_fusion) {
            int peer_idx = conv_add_fusion->peer_idx;
            Node* peer = peer_idx > idx && peer_idx < g_nn
                ? &g_n[peer_idx] : NULL;
            int residual_occurrences = 0;
            int conv_occurrences = 0;
            if (!peer || !peer->skip || strcmp(peer->op, "Add") ||
                peer->nin != 2 || strcmp(peer->out, n->out) ||
                p_int(peer->params, "relu", 0) != 0 ||
                !conv_add_fusion->source_tensor[0] ||
                !conv_add_fusion->aux_tensor[0]) return 0;
            for (int input_index = 0; input_index < 2; input_index++) {
                if (!strcmp(peer->ins[input_index].name,
                            conv_add_fusion->aux_tensor))
                    residual_occurrences++;
                if (!strcmp(peer->ins[input_index].name,
                            conv_add_fusion->source_tensor))
                    conv_occurrences++;
            }
            residual = t_find(conv_add_fusion->aux_tensor);
            if (residual_occurrences != 1 || conv_occurrences != 1 ||
                !residual ||
                residual->dtype != T_F32 || residual->numel != o->numel ||
                !tensor_same_shape(residual, o)) return 0;
            int residual_producer =
                producer_index_any(conv_add_fusion->aux_tensor);
            if (residual_producer > idx) return 0;
        }
        prepared_weight = prepare_conv_weight_image(idx, weight, p,
            in->shape[3], o->shape[3], groups, &weight_f32);
        if (!prepared_weight) return 0;
        if (bias && bias->dtype == T_F16) {
            float* values = f16_tensor_f32_cache(g_f16bcache, idx, bias);
            if (!values) return 0;
            bias_f32 = *bias;
            bias_f32.data = values;
            bias_f32.dtype = T_F32;
            bias_f32.elem_size = sizeof(float);
            prepared_bias = &bias_f32;
        }
        if (prepared_bias && prepared_bias->dtype != T_F32)
            return 0;
        int launched = conv_add_fusion
            ? cuda_graph_conv2d_add_f32(in->data, o->data,
                prepared_weight->data,
                prepared_bias ? prepared_bias->data : NULL, residual->data,
                in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                o->shape[3], prepared_weight->shape[0],
                prepared_weight->shape[1], o->shape[1], o->shape[2],
                stride_y, stride_x, pads[0], pads[1], groups, relu,
                dilation_y, dilation_x)
            : cuda_graph_conv2d_f32(in->data, o->data,
                prepared_weight->data,
                prepared_bias ? prepared_bias->data : NULL,
                in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                o->shape[3], prepared_weight->shape[0],
                prepared_weight->shape[1], o->shape[1], o->shape[2],
                stride_y, stride_x, pads[0], pads[1], groups, relu,
                dilation_y, dilation_x);
        if (launched) {
            *node_backend = conv_add_fusion
                ? "cuda-graph+conv-add" : "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Add") || !strcmp(op, "Mul") ||
        !strcmp(op, "Sub") || !strcmp(op, "Div")) {
        if (!strcmp(op, "Add")) {
            T* a = nin(n, "a");
            T* b = nin(n, "b");
            const GraphNodeFusion* add3_fusion =
                graph_opt_node_fusion(idx, GRAPH_FUSION_CHAINED_ELEMENTWISE);
            int relu = p_int(p, "relu", 0);
            if (!a) a = in;
            if (add3_fusion) {
                int peer_idx = add3_fusion->peer_idx;
                Node* peer = peer_idx > idx && peer_idx < g_nn
                    ? &g_n[peer_idx] : NULL;
                T* c = add3_fusion->aux_tensor[0]
                    ? t_find(add3_fusion->aux_tensor) : NULL;
                int c_occurrences = 0;
                int add0_occurrences = 0;
                int peer_relu = peer ? p_int(peer->params, "relu", 0) : -1;
                if (!peer || !peer->skip || strcmp(peer->op, "Add") ||
                    peer->nin != 2 || strcmp(peer->out, n->out) ||
                    n->nin != 2 || !a || !b || !c ||
                    !add3_fusion->source_tensor[0] ||
                    p_int(n->params, "relu", 0) != 0 ||
                    peer_relu < 0 || peer_relu > 2) return 0;
                for (int input_index = 0; input_index < 2; input_index++) {
                    if (!strcmp(peer->ins[input_index].name,
                                add3_fusion->aux_tensor))
                        c_occurrences++;
                    if (!strcmp(peer->ins[input_index].name,
                                add3_fusion->source_tensor))
                        add0_occurrences++;
                }
                if (c_occurrences != 1 || add0_occurrences != 1 ||
                    a->dtype != T_F32 ||
                    b->dtype != T_F32 || c->dtype != T_F32 ||
                    a->numel != b->numel || a->numel != c->numel ||
                    a->numel != o->numel || !tensor_same_shape(a, o) ||
                    !tensor_same_shape(b, o) || !tensor_same_shape(c, o) ||
                    producer_index_any(add3_fusion->aux_tensor) >= idx)
                    return 0;
                int row_start = 0;
                int row_count = 0;
                int partial = cuda_partial_sequence_rows(
                    o, o->ndim > 0 ? o->shape[o->ndim - 1] : 0,
                    &row_start, &row_count);
                if (partial < 0) return 0;
                long offset = partial
                    ? (long)row_start * o->shape[o->ndim - 1] : 0;
                long elements = partial
                    ? (long)row_count * o->shape[o->ndim - 1] : o->numel;
                if (!cuda_graph_add3_relu_f32(a->data + offset,
                        b->data + offset, c->data + offset, o->data + offset,
                        elements, peer_relu)) return 0;
                *node_backend = partial
                    ? "cuda-graph-row+add3" : "cuda-graph+add3";
                return 1;
            }
            if (a && b && a->dtype == T_F32 && b->dtype == T_F32 &&
                a->numel == b->numel && a->numel == o->numel &&
                relu >= 0 && relu <= 2) {
                int row_start = 0;
                int row_count = 0;
                int partial = cuda_partial_sequence_rows(
                    o, o->ndim > 0 ? o->shape[o->ndim - 1] : 0,
                    &row_start, &row_count);
                if (partial >= 0) {
                    long offset = partial
                        ? (long)row_start * o->shape[o->ndim - 1] : 0;
                    long elements = partial
                        ? (long)row_count * o->shape[o->ndim - 1] : o->numel;
                    if (cuda_graph_add_relu_f32(a->data + offset,
                            b->data + offset, o->data + offset, elements,
                            relu)) {
                        *node_backend = partial
                            ? "cuda-graph-row" : "cuda-graph";
                        return 1;
                    }
                }
            }
            if (relu) return 0;
        }
        if (try_gpu_graph_binary(n, in, o, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Sigmoid") || !strcmp(op, "Clip") ||
        !strcmp(op, "ReLU") || !strcmp(op, "GELU") ||
        !strcmp(op, "SiLU") ||
        !strcmp(op, "Tanh") || !strcmp(op, "HardSwish") ||
        !strcmp(op, "HardSigmoid") || !strcmp(op, "LeakyReLU")) {
        int ok = 0;
        if (!in) in = node_data_input(n, NULL);
        if (!in) return 0;
        materialize_tensor_f32(in);
        if (in->dtype != T_F32 || in->numel != o->numel || o->numel <= 0) return 0;
        int row_start = 0;
        int row_count = 0;
        int partial = cuda_partial_sequence_rows(
            o, o->ndim > 0 ? o->shape[o->ndim - 1] : 0,
            &row_start, &row_count);
        if (partial < 0) return 0;
        long offset = partial
            ? (long)row_start * o->shape[o->ndim - 1] : 0;
        long elements = partial
            ? (long)row_count * o->shape[o->ndim - 1] : o->numel;
        const float* input_data = in->data + offset;
        float* output_data = o->data + offset;
        if (!strcmp(op, "Sigmoid"))
            ok = cuda_graph_sigmoid_f32(input_data, output_data, elements);
        else if (!strcmp(op, "Clip")) {
            float minimum = p_flt(p, "min", -INFINITY);
            float maximum = p_flt(p, "max", INFINITY);
            if (isnan(minimum) || isnan(maximum) || minimum > maximum) return 0;
            ok = cuda_graph_clip_f32(input_data, output_data, elements, minimum, maximum);
        } else if (!strcmp(op, "ReLU"))
            ok = cuda_graph_relu_f32(input_data, output_data, elements);
        else if (!strcmp(op, "GELU")) {
            int approximate_tanh = 0;
            if (gelu_approximate_tanh(p, &approximate_tanh) != 0) return 0;
            ok = cuda_graph_gelu_f32(input_data, output_data, elements, approximate_tanh);
        } else if (!strcmp(op, "SiLU"))
            ok = cuda_graph_silu_f32(input_data, output_data, elements);
        else if (!strcmp(op, "Tanh"))
            ok = cuda_graph_tanh_f32(input_data, output_data, elements);
        else if (!strcmp(op, "HardSwish"))
            ok = cuda_graph_hardswish_f32(input_data, output_data, elements);
        else if (!strcmp(op, "HardSigmoid"))
            ok = cuda_graph_hardsigmoid_f32(input_data, output_data, elements);
        else
            ok = cuda_graph_leaky_relu_f32(input_data, output_data, elements,
                                            p_flt(p, "alpha", 0.01f));
        if (ok) {
            *node_backend = partial ? "cuda-graph-row" : "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "PReLU")) {
        T* weight = nin(n, "weight");
        if (!weight) weight = nin(n, "slope");
        int channels;
        if (!in || !weight || !o || in->ndim <= 0) return 0;
        materialize_tensor_f32(in);
        materialize_tensor_f32(weight);
        channels = weight->numel == 1 ? 1 : in->shape[in->ndim - 1];
        if (in->dtype != T_F32 || weight->dtype != T_F32 ||
            o->dtype != T_F32 || channels <= 0 || in->numel != o->numel ||
            (weight->numel != 1 && weight->numel != channels)) return 0;
        if (cuda_graph_prelu_f32(in->data, weight->data, o->data, o->numel,
                                 channels)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "LayerNorm")) {
        T* weight = nin(n, "weight");
        T* bias = nin(n, "bias");
        int width;
        long row_count;
        float epsilon;
        if (!in || !weight || in->ndim <= 0) return 0;
        width = p_int(p, "d_model", in->shape[in->ndim - 1]);
        epsilon = p_flt(p, "eps", 1.0e-6f);
        if (normalization_dimensions(in, o, weight, bias, width,
                                     &row_count) != 0 ||
            !(epsilon > 0.0f) || !isfinite(epsilon)) return 0;
        materialize_tensor_f32(in);
        materialize_tensor_f32(weight);
        if (bias) materialize_tensor_f32(bias);
        int row_start = 0;
        int selected_rows = 0;
        int partial = cuda_partial_sequence_rows(in, width, &row_start,
                                                  &selected_rows);
        if (partial < 0) return 0;
        if (!partial) selected_rows = (int)row_count;
        if (cuda_graph_layernorm_f32(in->data + (long)row_start * width,
                weight->data, bias ? bias->data : NULL,
                o->data + (long)row_start * width,
                selected_rows, width, epsilon)) {
            *node_backend = partial ? "cuda-graph-row" : "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "RMSNorm")) {
        T* weight = nin(n, "weight");
        int width;
        long row_count;
        float epsilon;
        if (!in || !weight || in->ndim <= 0) return 0;
        width = p_int(p, "d_model", in->shape[in->ndim - 1]);
        epsilon = p_flt(p, "eps", 1.0e-6f);
        if (normalization_dimensions(in, o, weight, NULL, width,
                                     &row_count) != 0 ||
            !(epsilon > 0.0f) || !isfinite(epsilon)) return 0;
        materialize_tensor_f32(in);
        materialize_tensor_f32(weight);
        int row_start = 0;
        int selected_rows = 0;
        int partial = cuda_partial_sequence_rows(in, width, &row_start,
                                                  &selected_rows);
        if (partial < 0) return 0;
        if (!partial) selected_rows = (int)row_count;
        if (cuda_graph_rmsnorm_f32(in->data + (long)row_start * width,
                weight->data, o->data + (long)row_start * width,
                selected_rows, width, epsilon)) {
            *node_backend = partial ? "cuda-graph-row" : "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "GroupNorm")) {
        T* weight = nin(n, "weight");
        T* bias = nin(n, "bias");
        int groups = p_int(p, "num_groups", p_int(p, "groups", 1));
        int batch, height, width, channels;
        float epsilon = p_flt(p, "eps", 1.0e-5f);
        if (!weight) weight = nin(n, "scale");
        if (groupnorm_dimensions(in, o, weight, bias, groups, &batch,
                                 &height, &width, &channels) != 0 ||
            !(epsilon > 0.0f) || !isfinite(epsilon)) return 0;
        materialize_tensor_f32(in);
        materialize_tensor_f32(weight);
        if (bias) materialize_tensor_f32(bias);
        if (cuda_graph_groupnorm_f32(in->data, weight->data,
                                     bias ? bias->data : NULL,
                                     o->data, batch, height, width, channels,
                                     groups, epsilon)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Softmax") || !strcmp(op, "LogSoftmax")) {
        int axis;
        int width;
        long row_count;
        int ok;
        if (!in || in->ndim <= 0) return 0;
        axis = p_int(p, "axis", in->ndim - 1);
        if (axis < 0) axis += in->ndim;
        if (axis != in->ndim - 1) return 0;
        materialize_tensor_f32(in);
        width = in->shape[in->ndim - 1];
        row_count = width > 0 ? in->numel / width : 0;
        if (in->dtype != T_F32 || width <= 0 || row_count <= 0 ||
            row_count > INT32_MAX || in->numel != row_count * width ||
            o->numel != in->numel) return 0;
        ok = !strcmp(op, "Softmax")
            ? cuda_graph_softmax_f32(in->data, o->data, (int)row_count, width)
            : cuda_graph_logsoftmax_f32(in->data, o->data, (int)row_count, width);
        if (ok) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "ReduceSum") || !strcmp(op, "ReduceMean")) {
        T* source = NULL;
        long rows = 0;
        int width = 0;
        float scale = 1.0f;
        if (last_axis_reduction_dimensions(n, o, &source, &rows, &width,
                                           &scale) != 0 ||
            rows > INT32_MAX) return 0;
        materialize_tensor_f32(source);
        if (cuda_graph_reduce_f32(source->data, o->data, (int)rows, width,
                                  scale)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "BatchNorm2D")) {
        T* weight = nin(n, "weight");
        T* bias = nin(n, "bias");
        T* mean = nin(n, "running_mean");
        T* variance = nin(n, "running_var");
        int channels;
        float epsilon = p_flt(p, "eps", 1.0e-5f);
        if (!in || !weight || !bias || !mean || !variance ||
            in->ndim != 4 || o->ndim != 4 || !(epsilon > 0.0f) ||
            !isfinite(epsilon)) return 0;
        materialize_tensor_f32(in);
        materialize_tensor_f32(weight);
        materialize_tensor_f32(bias);
        materialize_tensor_f32(mean);
        materialize_tensor_f32(variance);
        channels = in->shape[3];
        if (in->dtype != T_F32 || weight->dtype != T_F32 ||
            bias->dtype != T_F32 || mean->dtype != T_F32 ||
            variance->dtype != T_F32 || o->numel != in->numel ||
            channels <= 0 || weight->numel < channels ||
            bias->numel < channels || mean->numel < channels ||
            variance->numel < channels) return 0;
        if (cuda_graph_batchnorm2d_f32(in->data, weight->data, bias->data,
                mean->data, variance->data, o->data, in->shape[0],
                in->shape[1], in->shape[2], channels, epsilon)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "GlobalAveragePool")) {
        if (!in || in->ndim != 4) return 0;
        materialize_tensor_f32(in);
        if (in->dtype != T_F32 ||
            o->numel != (long)in->shape[0] * in->shape[3]) return 0;
        if (cuda_graph_global_average_pool_f32(in->data, o->data,
                in->shape[0], in->shape[1], in->shape[2], in->shape[3])) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "AveragePool2D")) {
        int kernel_y, kernel_x, stride_y, stride_x, padding_y, padding_x;
        if (!in || in->ndim != 4 || o->ndim != 4 ||
            in->dtype != T_F32) return 0;
        materialize_tensor_f32(in);
        p_pair(p, "kernel", 1, &kernel_y, &kernel_x);
        p_pair(p, "stride", 1, &stride_y, &stride_x);
        p_spatial_pad_origin(p, &padding_y, &padding_x);
        if (cuda_graph_average_pool2d_f32(in->data, o->data, in->shape[0],
                in->shape[1], in->shape[2], in->shape[3], o->shape[1],
                o->shape[2], kernel_y, kernel_x, stride_y, stride_x,
                padding_y, padding_x)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Conv1D")) {
        if (try_gpu_graph_conv1d(n, in, o, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Embedding")) {
        int partial = try_cuda_graph_embedding_partial(n, in, o);
        if (partial != 0) {
            if (partial > 0) {
                *node_backend = "cuda-graph-row";
                return 1;
            }
            return 0;
        }
        if (try_gpu_graph_embedding(n, in, o, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Resize") || !strcmp(op, "ResizeNearest2D") ||
        !strcmp(op, "UpsampleNearest2D")) {
        int mode;
        if (!in || in->dtype != T_F32 || in->ndim != 4 || o->ndim != 4 ||
            in->shape[0] <= 0 || o->shape[0] != in->shape[0] ||
            o->shape[3] != in->shape[3]) return 0;
        mode = (!strcmp(op, "Resize") &&
                strcmp(p_str(p, "mode", "bilinear"), "nearest")) ? 1 : 0;
        if (cuda_graph_resize_f32(in->data, o->data, in->shape[0],
                in->shape[1], in->shape[2], in->shape[3], o->shape[1],
                o->shape[2], mode)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "MaxPool2D")) {
        int kernel_y, kernel_x, stride_y, stride_x, padding_y, padding_x;
        if (!in || in->dtype != T_F32 || in->ndim != 4 || o->ndim != 4 ||
            in->shape[0] <= 0 || o->shape[0] != in->shape[0] ||
            o->shape[3] != in->shape[3]) return 0;
        p_pair(p, "kernel", 1, &kernel_y, &kernel_x);
        p_pair(p, "stride", 1, &stride_y, &stride_x);
        p_spatial_pad_origin(p, &padding_y, &padding_x);
        if (cuda_graph_maxpool2d_f32(in->data, o->data, in->shape[0],
                in->shape[1], in->shape[2], in->shape[3], o->shape[1],
                o->shape[2], kernel_y, kernel_x, stride_y, stride_x,
                padding_y, padding_x)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Transpose")) {
        int permutation[8];
        if (!in || in->ndim <= 0 || in->ndim > 8) return 0;
        materialize_tensor_f32(in);
        for (int dimension = 0; dimension < in->ndim; dimension++)
            permutation[dimension] = in->ndim - 1 - dimension;
        if (p && cJSON_GetObjectItem(p, "perm"))
            p_iarr(p, "perm", permutation, in->ndim, 0);
        if (in->dtype == T_F32 && o->numel == in->numel &&
            cuda_graph_transpose_f32(in->data, o->data, in->shape,
                                     permutation, in->ndim)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Where") || !strcmp(op, "Mask")) {
        T* condition = nin(n, "cond");
        T* a = nin(n, "x");
        T* b = nin(n, "y");
        if (!condition) condition = nin(n, "condition");
        if (!a) a = nin(n, "a");
        if (!b) b = nin(n, "b");
        if (!condition || !a || !b) return 0;
        materialize_tensor_f32(condition);
        materialize_tensor_f32(a);
        materialize_tensor_f32(b);
        if (condition->dtype == T_F32 && a->dtype == T_F32 &&
            b->dtype == T_F32 && condition->numel == o->numel &&
            a->numel == o->numel && b->numel == o->numel &&
            cuda_graph_where_f32(condition->data, a->data, b->data,
                                 o->data, o->numel)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Expand") || !strcmp(op, "Broadcast")) {
        T* source = in ? in : nin(n, "data");
        if (!source) return 0;
        materialize_tensor_f32(source);
        if (source->dtype == T_F32 &&
            cuda_graph_expand_f32(source->data, o->data, source->shape,
                                  source->ndim, o->shape, o->ndim)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Gather")) {
        T* indices = nin(n, "indices");
        int axis;
        int outer;
        int axis_size;
        int inner;
        int indices_elements;
        int output_elements;
        if (!in || !indices || in->ndim <= 0) return 0;
        axis = normalize_axis(p_int(p, "axis", 0), in->ndim);
        materialize_tensor_f32(in);
        if (gather_i32_dimensions(in, indices, o, axis, &outer, &axis_size,
                                  &inner, &indices_elements,
                                  &output_elements) &&
            cuda_graph_gather_i32_f32(in->data,
                                      (const int32_t*)indices->data, o->data,
                                      outer, axis_size, inner,
                                      indices_elements, output_elements)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Pad")) {
        T* source = in ? in : nin(n, "data");
        int pads[8] = {0};
        cJSON* pads_node;
        int pads_length;
        int padding_top;
        int padding_left;
        if (!source || source->ndim <= 0 || source->ndim > 4 ||
            o->ndim <= 0 || o->ndim > 4) return 0;
        materialize_tensor_f32(source);
        p_iarr(p, "pads", pads, 8, 0);
        pads_node = p ? cJSON_GetObjectItem(p, "pads") : NULL;
        pads_length = pads_node && cJSON_IsArray(pads_node)
            ? cJSON_GetArraySize(pads_node) : 0;
        padding_top = pads_length == 8 ? pads[1] : pads[0];
        padding_left = pads_length == 8 ? pads[2] : pads[1];
        if (padding_top < 0 || padding_left < 0) return 0;
        if (source->dtype == T_F32 &&
            cuda_graph_pad4d_f32(source->data, o->data, source->shape,
                source->ndim, o->shape, o->ndim, padding_top, padding_left,
                p_flt(p, "value", 0.0f))) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Slice")) {
        T* source = in ? in : nin(n, "data");
        int starts4[4] = {0, 0, 0, 0};
        int steps4[4] = {1, 1, 1, 1};
        int starts[8] = {0};
        int steps[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        int axes[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        cJSON* axes_node;
        int axis_count;
        int base;
        if (!source || source->ndim <= 0 || source->ndim > 4 ||
            o->ndim <= 0 || o->ndim > 4) return 0;
        materialize_tensor_f32(source);
        p_iarr(p, "starts", starts, 8, 0);
        p_iarr(p, "steps", steps, 8, 1);
        p_iarr(p, "axes", axes, 8, 0);
        axes_node = p ? cJSON_GetObjectItem(p, "axes") : NULL;
        axis_count = axes_node && cJSON_IsArray(axes_node)
            ? cJSON_GetArraySize(axes_node) : source->ndim;
        base = 4 - source->ndim;
        for (int index = 0; index < axis_count && index < 8; index++) {
            int axis = axes_node ? axes[index] : index;
            int start;
            if (axis < 0) axis += source->ndim;
            if (axis < 0 || axis >= source->ndim || steps[index] <= 0)
                return 0;
            start = starts[index] < 0
                ? starts[index] + source->shape[axis] : starts[index];
            if (start < 0) return 0;
            starts4[base + axis] = start;
            steps4[base + axis] = steps[index];
        }
        if (source->dtype == T_F32 &&
            cuda_graph_slice4d_f32(source->data, o->data, source->shape,
                source->ndim, o->shape, o->ndim, starts4, steps4)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "ConvTranspose2D")) {
        T* weight = nin(n, "weight");
        T* bias = nin(n, "bias");
        int kernel_y, kernel_x, stride_y, stride_x, padding_y, padding_x;
        if (!in || !weight || in->ndim != 4 || o->ndim != 4) return 0;
        materialize_tensor_f32(in);
        materialize_tensor_f32(weight);
        if (bias) materialize_tensor_f32(bias);
        p_pair(p, "kernel", 1, &kernel_y, &kernel_x);
        p_pair(p, "stride", 1, &stride_y, &stride_x);
        p_pair(p, "padding", 0, &padding_y, &padding_x);
        if (in->dtype == T_F32 && weight->dtype == T_F32 &&
            (!bias || bias->dtype == T_F32) &&
            cuda_graph_conv_transpose2d_f32(in->data, weight->data,
                bias ? bias->data : NULL, o->data, in->shape[0],
                in->shape[1], in->shape[2], in->shape[3], o->shape[1],
                o->shape[2], o->shape[3], kernel_y, kernel_x, stride_y,
                stride_x, padding_y, padding_x)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "SDPA")) {
        int partial = try_cuda_graph_sdpa_partial(n, in, o);
        if (partial != 0) {
            if (partial > 0) {
                *node_backend = "cuda-graph-row";
                return 1;
            }
            return 0;
        }
        if (try_gpu_graph_sdpa(n, in, o, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "CrossSDPA")) {
        if (try_gpu_graph_cross_sdpa(n, o, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "CrossAttention")) {
        if (try_gpu_graph_cross_attention(n, o, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Split")) {
        if (try_gpu_graph_split(n, in, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "SpatialSoftargmaxY") || !strcmp(op, "ProfileX") ||
        !strcmp(op, "ProfileY") || !strcmp(op, "MeanHeight")) {
        if (try_gpu_graph_profile_op(n, in, o, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "NonMaxSuppression")) {
        if (try_gpu_graph_nms(n, o, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Interpolate1D")) {
        int channels;
        int input_length;
        int output_length;
        if (!in || in->ndim != 2 || o->ndim != 2) return 0;
        materialize_tensor_f32(in);
        channels = in->shape[in->ndim - 2];
        input_length = in->shape[in->ndim - 1];
        output_length = o->shape[o->ndim - 1];
        if (in->dtype == T_F32 && o->shape[0] == channels &&
            cuda_graph_interp1d_f32(in->data, o->data, channels,
                                    input_length, output_length)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Cast") || !strcmp(op, "Reshape") ||
        !strcmp(op, "Flatten") || !strcmp(op, "Squeeze") ||
        !strcmp(op, "Unsqueeze") || !strcmp(op, "Identity") ||
        !strcmp(op, "Dropout")) {
        T* source = in ? in : node_data_input(n, NULL);
#if VOLVOXAI_ENABLE_TRAINING
        if (!strcmp(op, "Dropout") && g_native_training_mode) {
            NativeDropoutParams dropout;
            if (!source || source->dtype != T_F32 ||
                source->numel != o->numel || o->numel <= 0 ||
                native_dropout_params(n, idx, &dropout) != 0) return 0;
            materialize_tensor_f32(source);
            if (cuda_graph_dropout_f32(source->data, o->data, o->numel,
                                       dropout.threshold, dropout.seed,
                                       dropout.counter, dropout.scale)) {
                *node_backend = "cuda-graph";
                return 1;
            }
            return 0;
        }
#endif
        if (!source) return 0;
        materialize_tensor_f32(source);
        if (source->dtype == T_F32 && source->numel == o->numel && o->numel > 0 &&
            cuda_graph_copy_f32(source->data, o->data, o->numel)) {
            *node_backend = "cuda-graph-copy";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Concat")) {
        int axis = p_int(p, "axis", 1);
        const float* inputs[MAXIN];
        long sizes[MAXIN];
        int input_axes[MAXIN];
        long total = 0;
        long summed_axis = 0;
        int inner;
        int sigmoid;
        if (axis < 0) axis += o->ndim;
        if (axis < 0 || axis >= o->ndim || o->shape[axis] <= 0) return 0;
        for (int i = 0; i < n->nin; i++) {
            T* source = t_find(n->ins[i].name);
            if (!source || source->dtype != T_F32 || source->ndim != o->ndim ||
                source->shape[axis] <= 0) return 0;
            for (int dimension = 0; dimension < o->ndim; dimension++)
                if (dimension != axis &&
                    source->shape[dimension] != o->shape[dimension]) return 0;
            materialize_tensor_f32(source);
            inputs[i] = source->data;
            sizes[i] = source->numel;
            input_axes[i] = source->shape[axis];
            total += source->numel;
            summed_axis += source->shape[axis];
        }
        inner = tensor_inner_after_axis(o, axis);
        sigmoid = p_int(p, "sigmoid", 0) ? 1 : 0;
        if (total == o->numel && summed_axis == o->shape[axis] && inner > 0 &&
            cuda_graph_concat_f32(inputs, sizes, input_axes, n->nin, o->data,
                                  o->shape[axis], inner, sigmoid)) {
            *node_backend = sigmoid ? "cuda-graph+sigmoid" : "cuda-graph";
            return 1;
        }
        return 0;
    }

    return 0;
}
#endif
