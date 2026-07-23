/* Lives beside matmul_f32 in broadcast_ops.c so both layouts share one runtime
 * ISA dispatch; it used to be a scalar loop here and was the only Linear
 * fallback left unvectorized. */
void matmul_f32_out_in(const float* input, const float* weight, const float* bias,
                       float* output, int rows, int d_in, int d_out);

static int linear_node_weight_layout(Node* node, const T* input, const T* output,
                                     const T* weight, int* out_in) {
    if (!node || !input || !output || !weight || input->ndim <= 0 || output->ndim <= 0 || weight->ndim != 2) return -1;
    int d_in = input->shape[input->ndim - 1];
    int d_out = output->shape[output->ndim - 1];
    const char* layout = p_str(node->params, "weight_layout", "");
    int transposed = -1;
    if (layout[0]) {
        if (!strcmp(layout, "OUT_IN") || !strcmp(layout, "out_in") || !strcmp(layout, "OI") || !strcmp(layout, "peft")) transposed = 1;
        else if (!strcmp(layout, "IN_OUT") || !strcmp(layout, "in_out") || !strcmp(layout, "IO") || !strcmp(layout, "din_dout")) transposed = 0;
        else return -1;
    }
    if (transposed < 0 && p_int(node->params, "transB", 0)) transposed = 1;
    int canonical = weight->shape[0] == d_in && weight->shape[1] == d_out;
    int out_in_shape = weight->shape[0] == d_out && weight->shape[1] == d_in;
    if (transposed < 0) transposed = out_in_shape && !canonical ? 1 : canonical ? 0 : -1;
    if (transposed < 0 || (!transposed && !canonical) || (transposed && !out_in_shape)) return -1;
    *out_in = transposed;
    return 0;
}

static int route_index_read(const T* indices, long index, int* value) {
    if (!indices || !value || index < 0 || index >= indices->numel) return -1;
    if (indices->dtype == T_I32) {
        *value = ((const int32_t*)indices->data)[index];
        return 0;
    }
    if (indices->dtype == T_F32) {
        float f = indices->data[index];
        int i = (int)f;
        if (!isfinite(f) || f != (float)i) return -1;
        *value = i;
        return 0;
    }
    return -1;
}

static int route_index_write(T* indices, long index, int value) {
    if (!indices || index < 0 || index >= indices->numel) return -1;
    if (indices->dtype == T_I32) {
        ((int32_t*)indices->data)[index] = value;
        return 0;
    }
    if (indices->dtype == T_F32) {
        indices->data[index] = (float)value;
        return 0;
    }
    return -1;
}

static int moe_router_forward(Node* n, int use_cuda) {
    T* in = node_input_default(n);
    T* w = nin(n, "weight");
    T* b = nin(n, "bias");
    T* indices = nout(n, "indices");
    T* weights = nout(n, "weights");
    if (!indices) indices = nout(n, "route_indices");
    if (!weights) weights = nout(n, "route_weights");
    if (!in || !w || !indices || !weights || in->dtype != T_F32 || w->dtype != T_F32 ||
        (indices->dtype != T_F32 && indices->dtype != T_I32) ||
        weights->dtype != T_F32 || in->ndim <= 0 || w->ndim != 2) return -1;
    int d_model = in->shape[in->ndim - 1];
    int num_experts = p_int(n->params, "num_experts", w->shape[1]);
    int top_k = p_int(n->params, "top_k", 1);
    float temperature = p_flt(n->params, "temperature", 1.0f);
    if (d_model <= 0 || num_experts <= 0 || top_k <= 0 || top_k > num_experts ||
        !isfinite(temperature) || temperature <= 0.0f ||
        w->shape[0] != d_model || w->shape[1] != num_experts ||
        (b && (b->dtype != T_F32 || b->numel != num_experts))) return -1;
    long tokens = in->numel / d_model;
    if (in->numel != tokens * d_model || indices->numel != tokens * top_k ||
        weights->numel != tokens * top_k || tokens <= 0 ||
        tokens > INT_MAX || indices->numel > UINT32_MAX ||
        (size_t)indices->numel > SIZE_MAX / sizeof(float) ||
        w->numel != (long)d_model * num_experts) return -1;
#if VOLVOXAI_ENABLE_CUDA
    if (use_cuda) {
        if (indices->dtype != T_F32 ||
            !cuda_graph_moe_router_f32(
                in->data, w->data, b ? b->data : NULL, indices->data,
                weights->data, (int)tokens, d_model, num_experts, top_k,
                temperature, p_int(n->params, "normalize", 1) ? 1 : 0))
            return -1;
        return 0;
    }
#else
    if (use_cuda) return -1;
#endif
    float* kernel_indices = indices->dtype == T_F32 ? indices->data :
        (float*)malloc((size_t)indices->numel * sizeof(float));
    if (!kernel_indices) return -1;
    int normalize = p_int(n->params, "normalize", 1) ? 1 : 0;
    int ok = moe_router_f32(
        in->data, w->data, b ? b->data : NULL, kernel_indices, weights->data,
        (uint32_t)tokens, (uint32_t)d_model, (uint32_t)num_experts,
        (uint32_t)top_k, (double)temperature, (uint32_t)normalize);
    if (ok && indices->dtype == T_I32) {
        for (long index = 0; index < indices->numel; index++) {
            if (route_index_write(indices, index, (int)kernel_indices[index]) != 0) {
                ok = 0;
                break;
            }
        }
    }
    if (kernel_indices != indices->data) free(kernel_indices);
    return ok ? 0 : -1;
}

static int moe_linear_forward(Node* n, T* out, int use_cuda) {
    T* in = node_input_default(n);
    T* ew = nin(n, "expert_weight");
    T* eb = nin(n, "expert_bias");
    T* indices = nin(n, "route_indices");
    T* weights = nin(n, "route_weights");
    if (!indices) indices = nin(n, "indices");
    if (!weights) weights = nin(n, "weights");
    if (!in || !ew || !indices || !weights || !out || in->dtype != T_F32 ||
        ew->dtype != T_F32 || (indices->dtype != T_F32 && indices->dtype != T_I32) ||
        weights->dtype != T_F32 || out->dtype != T_F32 ||
        in->ndim <= 0 || ew->ndim != 3 || out->ndim <= 0) return -1;
    int num_experts = ew->shape[0];
    int d_in = ew->shape[1];
    int d_out = ew->shape[2];
    if (num_experts <= 0 || d_in <= 0 || d_out <= 0 || in->shape[in->ndim - 1] != d_in ||
        out->shape[out->ndim - 1] != d_out || (eb && (eb->dtype != T_F32 ||
        eb->numel != (long)num_experts * d_out))) return -1;
    long tokens = in->numel / d_in;
    if (out->numel != tokens * d_out || tokens <= 0 || indices->numel != weights->numel ||
        indices->numel % tokens) return -1;
    int top_k = (int)(indices->numel / tokens);
    if (top_k <= 0 || top_k > num_experts || tokens > INT_MAX ||
        indices->numel > UINT32_MAX ||
        (size_t)indices->numel > SIZE_MAX / sizeof(float) ||
        ew->numel != (long)num_experts * d_in * d_out) return -1;
#if VOLVOXAI_ENABLE_CUDA
    if (use_cuda) {
        if (indices->dtype != T_F32 ||
            !cuda_graph_moe_linear_f32(
                in->data, ew->data, eb ? eb->data : NULL, indices->data,
                weights->data, out->data, (int)tokens, d_in, d_out,
                num_experts, top_k))
            return -1;
        return 0;
    }
#else
    if (use_cuda) return -1;
#endif
    float* kernel_indices = indices->dtype == T_F32 ? indices->data :
        (float*)malloc((size_t)indices->numel * sizeof(float));
    if (!kernel_indices) return -1;
    if (indices->dtype == T_I32) {
        for (long index = 0; index < indices->numel; index++) {
            int expert;
            if (route_index_read(indices, index, &expert) != 0 || expert < 0 ||
                expert >= num_experts) {
                free(kernel_indices);
                return -1;
            }
            kernel_indices[index] = (float)expert;
        }
    }
    int ok = moe_linear_f32(
        in->data, ew->data, eb ? eb->data : NULL, kernel_indices,
        weights->data, out->data, (uint32_t)tokens, (uint32_t)d_in,
        (uint32_t)d_out, (uint32_t)num_experts, (uint32_t)top_k);
    if (kernel_indices != indices->data) free(kernel_indices);
    return ok ? 0 : -1;
}

/* The native engine has global graph state and serializes forwards with the
 * model lock.  Keep the per-node dispatch facts here rather than widening the
 * public VxBackend ABI with engine-private graph metadata. */
typedef struct {
    const VxW8A8Ops* (*w8a8_ops)(void);
    int (*try_graph_node)(Node* node, int index, const char** node_backend);
    const char* qlinear_label;
    const char* qbatch_matmul_label;
    const char* qembedding_label;
    const char* qconv_label;
    const char* qadd_label;
    const char* qsilu_label;
    const char* qgelu_label;
    const char* qgroupnorm_label;
    const char* qlayernorm_label;
    const char* qsdpa_label;
    const char* qargmax_label;
    const char* qmaskedmean_label;
    const char* requantize_label;
    const char* quantize_label;
    const char* dequantize_label;
    const char* shape_alias_label;
    const char* copy_label;
    const char* transpose_label;
    const char* resize_label;
    const char* maxpool_label;
    const char* concat_label;
} VxDeviceRoute;

/* Static graph replay is disabled while request-specific execution state is
 * active. */
int vx_runtime_full_graph_execution_eligible(void) {
#if VOLVOXAI_ENABLE_TRAINING
    if (g_native_training_mode) return 0;
#endif
    return g_active_row < 0 && g_prefix_rows == 0 && g_execution_row < 0;
}

static int vx_runtime_node_has_physical_gpu_op(const VxRuntimeNodeState* state) {
    return state->native_physical_qlinear ||
        state->native_physical_qbatch_matmul ||
        state->native_physical_qembedding ||
        state->native_physical_qconv || state->native_physical_qadd ||
        state->native_physical_qsilu || state->native_physical_qgelu ||
        state->native_physical_qgroupnorm || state->native_physical_qlayernorm ||
        state->native_physical_qsdpa || state->native_physical_qargmax ||
        state->native_physical_qmaskedmean || state->native_physical_requantize ||
        state->native_physical_quantize || state->native_physical_dequantize ||
        state->native_physical_shape;
}

static int vx_device_route_run(const Node* node, T* input, T* output,
                               const VxDeviceRoute* route) {
    const VxRuntimeNodeState* state = &g_vx_runtime_node;
    const VxW8A8Ops* w8a8;
    Node* mutable_node = (Node*)node;
    (void)input;
    if (!node || !output || !route || !state->active || !state->node_backend)
        return VX_BACKEND_DECLINED;
    w8a8 = route->w8a8_ops ? route->w8a8_ops() : NULL;
    if (state->adapter_linear) return VX_BACKEND_DECLINED;
    if (state->native_physical_qlinear)
        return qlinear_physical_gpu_forward(state->idx, mutable_node, output,
                                             state->is_last, w8a8,
                                             route->qlinear_label, state->node_backend)
            ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
    if (state->native_physical_qbatch_matmul)
        return qbatch_matmul_physical_gpu_forward(
                   mutable_node, output, w8a8, route->qbatch_matmul_label,
                   state->node_backend)
            ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
    if (state->native_physical_qconv)
        return qconv2d_physical_gpu_forward(state->idx, mutable_node, output, w8a8,
                                             route->qconv_label, state->node_backend)
            ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
    if (state->native_physical_qembedding)
        return qembedding_physical_gpu_forward(state->idx, mutable_node, output, w8a8,
                                                route->qembedding_label, state->node_backend)
            ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
    if (state->native_physical_qadd)
        return qadd_physical_gpu_forward(mutable_node, output, w8a8, route->qadd_label,
                                          state->node_backend)
            ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
    if (state->native_physical_qsilu)
        return qsilu_physical_gpu_forward(mutable_node, output, w8a8, route->qsilu_label,
                                           state->node_backend)
            ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
    if (state->native_physical_qgelu)
        return qgelu_physical_gpu_forward(mutable_node, output, w8a8, route->qgelu_label,
                                           state->node_backend)
            ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
    if (state->native_physical_qgroupnorm)
        return qgroupnorm_physical_gpu_forward(mutable_node, output, w8a8,
                                                route->qgroupnorm_label, state->node_backend)
            ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
    if (state->native_physical_qlayernorm)
        return qlayernorm_physical_gpu_forward(mutable_node, output, w8a8,
                                                route->qlayernorm_label, state->node_backend)
            ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
    if (state->native_physical_qsdpa)
        return qsdpa_physical_gpu_forward(mutable_node, output, w8a8, route->qsdpa_label,
                                           state->node_backend)
            ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
    if (state->native_physical_qargmax)
        return qargmax_physical_gpu_forward(mutable_node, output, w8a8,
                                             route->qargmax_label, state->node_backend)
            ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
    if (state->native_physical_qmaskedmean)
        return qmaskedmean_physical_gpu_forward(mutable_node, output, w8a8,
                                                 route->qmaskedmean_label, state->node_backend)
            ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
    if (state->native_physical_requantize)
        return requantize_physical_gpu_forward(mutable_node, output, w8a8,
                                                route->requantize_label, state->node_backend)
            ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
    if (state->native_physical_quantize)
        return quantize_physical_gpu_forward(mutable_node, output, w8a8,
                                              route->quantize_label, state->node_backend)
            ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
    if (state->native_physical_dequantize)
        return dequantize_physical_gpu_forward(mutable_node, output, w8a8,
                                                route->dequantize_label, state->node_backend)
            ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
    if (state->native_physical_shape)
        return physical_shape_gpu_forward(mutable_node, state->idx, output, w8a8,
                                          route->shape_alias_label, route->copy_label,
                                          route->transpose_label, route->resize_label,
                                          route->maxpool_label,
                                          route->concat_label, state->node_backend)
            ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
    if (vx_runtime_node_has_physical_gpu_op(state) || !route->try_graph_node)
        return VX_BACKEND_DECLINED;
    return route->try_graph_node(mutable_node, state->idx, state->node_backend)
        ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
}

#if VOLVOXAI_ENABLE_VULKAN
static const VxDeviceRoute g_vx_vulkan_route = {
    .w8a8_ops = vx_w8a8_vulkan_ops,
    .try_graph_node = try_vulkan_graph_node,
    .qlinear_label = "vulkan-qlinear-w8a8",
    .qbatch_matmul_label = "vulkan-qbatch-matmul-w8a8",
    .qembedding_label = "vulkan-qembedding-w8a8",
    .qconv_label = "vulkan-qconv-w8a8",
    .qadd_label = "vulkan-qadd-w8a8",
    .qsilu_label = "vulkan-qsilu-w8a8",
    .qgelu_label = "vulkan-qgelu-w8a8",
    .qgroupnorm_label = "vulkan-qgroupnorm-w8a8",
    .qlayernorm_label = "vulkan-qlayernorm-w8a8",
    .qsdpa_label = "vulkan-qsdpa-w8a8",
    .qargmax_label = "vulkan-qargmax-w8a8",
    .qmaskedmean_label = "vulkan-qmaskedmean-w8a8",
    .requantize_label = "vulkan-requantize-w8a8",
    .quantize_label = "vulkan-quantize-w8a8",
    .dequantize_label = "vulkan-dequantize-w8a8",
    .shape_alias_label = "vulkan-shape-alias-w8a8",
    .copy_label = "vulkan-copy-w8a8",
    .transpose_label = "vulkan-transpose-w8a8",
    .resize_label = "vulkan-resize-w8a8",
    .maxpool_label = "vulkan-maxpool-w8a8",
    .concat_label = "vulkan-concat-w8a8",
};
#endif

#if VOLVOXAI_ENABLE_OPENGL
static const VxDeviceRoute g_vx_opengl_route = {
    .w8a8_ops = vx_w8a8_opengl_ops,
    .try_graph_node = try_opengl_graph_node,
    .qlinear_label = "opengl-qlinear-w8a8",
    .qbatch_matmul_label = "opengl-qbatch-matmul-w8a8",
    .qembedding_label = "opengl-qembedding-w8a8",
    .qconv_label = "opengl-qconv-w8a8",
    .qadd_label = "opengl-qadd-w8a8",
    .qsilu_label = "opengl-qsilu-w8a8",
    .qgelu_label = "opengl-qgelu-w8a8",
    .qgroupnorm_label = "opengl-qgroupnorm-w8a8",
    .qlayernorm_label = "opengl-qlayernorm-w8a8",
    .qsdpa_label = "opengl-qsdpa-w8a8",
    .qargmax_label = "opengl-qargmax-w8a8",
    .qmaskedmean_label = "opengl-qmaskedmean-w8a8",
    .requantize_label = "opengl-requantize-w8a8",
    .quantize_label = "opengl-quantize-w8a8",
    .dequantize_label = "opengl-dequantize-w8a8",
    .shape_alias_label = "opengl-shape-alias-w8a8",
    .copy_label = "opengl-copy-w8a8",
    .transpose_label = "opengl-transpose-w8a8",
    .resize_label = "opengl-resize-w8a8",
    .maxpool_label = "opengl-maxpool-w8a8",
    .concat_label = "opengl-concat-w8a8",
};
#endif

#if VOLVOXAI_ENABLE_METAL
static const VxDeviceRoute g_vx_metal_route = {
    .w8a8_ops = vx_w8a8_metal_ops,
    .try_graph_node = try_metal_graph_node,
    .qlinear_label = "metal-qlinear-w8a8",
    .qbatch_matmul_label = "metal-qbatch-matmul-w8a8",
    .qembedding_label = "metal-qembedding-w8a8",
    .qconv_label = "metal-qconv-w8a8",
    .qadd_label = "metal-qadd-w8a8",
    .qsilu_label = "metal-qsilu-w8a8",
    .qgelu_label = "metal-qgelu-w8a8",
    .qgroupnorm_label = "metal-qgroupnorm-w8a8",
    .qlayernorm_label = "metal-qlayernorm-w8a8",
    .qsdpa_label = "metal-qsdpa-w8a8",
    .qargmax_label = "metal-qargmax-w8a8",
    .qmaskedmean_label = "metal-qmaskedmean-w8a8",
    .requantize_label = "metal-requantize-w8a8",
    .quantize_label = "metal-quantize-w8a8",
    .dequantize_label = "metal-dequantize-w8a8",
    .shape_alias_label = "metal-shape-alias-w8a8",
    .copy_label = "metal-copy-w8a8",
    .transpose_label = "metal-transpose-w8a8",
    .resize_label = "metal-resize-w8a8",
    .maxpool_label = "metal-maxpool-w8a8",
    .concat_label = "metal-concat-w8a8",
};
#endif

#if VOLVOXAI_ENABLE_CUDA
static const VxDeviceRoute g_vx_cuda_route = {
    .w8a8_ops = vx_w8a8_cuda_ops,
    .try_graph_node = try_cuda_graph_node,
    .qlinear_label = "cuda-qlinear-w8a8",
    .qbatch_matmul_label = "cuda-qbatch-matmul-w8a8",
    .qembedding_label = "cuda-qembedding-w8a8",
    .qconv_label = "cuda-qconv-w8a8",
    .qadd_label = "cuda-qadd-w8a8",
    .qsilu_label = "cuda-qsilu-w8a8",
    .qgelu_label = "cuda-qgelu-w8a8",
    .qgroupnorm_label = "cuda-qgroupnorm-w8a8",
    .qlayernorm_label = "cuda-qlayernorm-w8a8",
    .qsdpa_label = "cuda-qsdpa-w8a8",
    .qargmax_label = "cuda-qargmax-w8a8",
    .qmaskedmean_label = "cuda-qmaskedmean-w8a8",
    .requantize_label = "cuda-requantize-w8a8",
    .quantize_label = "cuda-quantize-w8a8",
    .dequantize_label = "cuda-dequantize-w8a8",
    .shape_alias_label = "cuda-shape-alias-w8a8",
    .copy_label = "cuda-copy-w8a8",
    .transpose_label = "cuda-transpose-w8a8",
    .resize_label = "cuda-resize-w8a8",
    .maxpool_label = "cuda-maxpool-w8a8",
    .concat_label = "cuda-concat-w8a8",
};
#endif

static int vx_backend_attach(void* user_data) { (void)user_data; return VX_BACKEND_INIT_READY; }
static void vx_backend_detach(void* user_data) { (void)user_data; }
#if VOLVOXAI_ENABLE_VULKAN
static int vx_vulkan_graph_supports(void* user_data, const Node* node) {
    (void)user_data;
    return node && g_use_vulkan ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
}
#endif
#if VOLVOXAI_ENABLE_OPENGL
static int vx_opengl_graph_supports(void* user_data, const Node* node) {
    (void)user_data;
    return node && g_use_opengl ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
}
#endif
#if VOLVOXAI_ENABLE_METAL
static int vx_metal_graph_supports(void* user_data, const Node* node) {
    (void)user_data;
    return node && g_use_metal ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
}
#endif
#if VOLVOXAI_ENABLE_CUDA
static int vx_cuda_linear_node(const Node* node) {
    return node && (!strcmp(node->op, "Linear") ||
                    !strcmp(node->op, "MatMul") ||
                    !strcmp(node->op, "Gemm"));
}

static int vx_cuda_graph_supports(void* user_data, const Node* node) {
    (void)user_data;
    return node && g_use_cuda ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
}
#endif
#if VOLVOXAI_ENABLE_VULKAN
static int vx_vulkan_graph_run(void* user_data, const Node* node, T* input, T* output) {
    (void)user_data;
    return vx_device_route_run(node, input, output, &g_vx_vulkan_route);
}
#endif
#if VOLVOXAI_ENABLE_OPENGL
static int vx_opengl_graph_run(void* user_data, const Node* node, T* input, T* output) {
    (void)user_data;
    return vx_device_route_run(node, input, output, &g_vx_opengl_route);
}
#endif
#if VOLVOXAI_ENABLE_METAL
static int vx_metal_graph_run(void* user_data, const Node* node, T* input, T* output) {
    (void)user_data;
    return vx_device_route_run(node, input, output, &g_vx_metal_route);
}
#endif
#if VOLVOXAI_ENABLE_CUDA
static int vx_cuda_graph_run(void* user_data, const Node* node, T* input, T* output) {
    int status;
    (void)user_data;
    if (node && !strcmp(node->op, "MoERouter")) {
        if (moe_router_forward((Node*)node, 1) != 0) {
            fprintf(stderr, "[CUDA] node 'MoERouter' failed in the CUDA graph route\n");
            return VX_BACKEND_ERROR;
        }
        if (g_vx_runtime_node.node_backend)
            *g_vx_runtime_node.node_backend = "cuda-moe-router";
        return VX_BACKEND_HANDLED;
    }
    if (node && !strcmp(node->op, "MoELinear")) {
        if (moe_linear_forward((Node*)node, output, 1) != 0) {
            fprintf(stderr, "[CUDA] node 'MoELinear' failed in the CUDA graph route\n");
            return VX_BACKEND_ERROR;
        }
        if (g_vx_runtime_node.node_backend)
            *g_vx_runtime_node.node_backend = "cuda-moe-linear";
        return VX_BACKEND_HANDLED;
    }
    status = vx_device_route_run(node, input, output, &g_vx_cuda_route);
    if (status != VX_BACKEND_DECLINED) return status;
    /* A decline remains part of the selected built-in CUDA provider's route.
     * The registry may reach its CPU operator entry when operator fallback is
     * allowed; compile/execution route attestation rejects that exact route
     * when fallback is forbidden. */
    return VX_BACKEND_DECLINED;
}
#endif

typedef struct {
    const char* weight_name;
    const float* input;
    const float* weight;
    const float* bias;
    float* output;
    int rows;
    int d_in;
    int d_out;
    int out_in;
    int batch_size;
    long row_start;
    long total_rows;
} VxOneShotLinear;

static int vx_prepare_one_shot_linear(const Node* node, T* input, T* output,
                                      VxOneShotLinear* call,
                                      long minimum_cost,
                                      int full_tensor,
                                      int allow_adapter) {
    const VxRuntimeNodeState* state = &g_vx_runtime_node;
    Node* mutable_node = (Node*)node;
    T* weight;
    T* bias;
    T weight_f32;
    T bias_f32;
    T* prepared_weight;
    T* prepared_bias;
    int out_in = 0;
    int d_in;
    int d_out;
    long total;
    int row_start;
    int row_count;
    long cost;
    if (!node || !input || !output || !call || !state->active ||
        (state->adapter_linear && !allow_adapter) ||
        (strcmp(node->op, "MatMul") && strcmp(node->op, "Gemm") &&
         strcmp(node->op, "Linear"))) return 0;
    materialize_tensor_f32(input);
    weight = nin(mutable_node, "weight");
    bias = nin(mutable_node, "bias");
    if (!weight || input->dtype != T_F32 || output->dtype != T_F32 || input->ndim <= 0 ||
        output->ndim <= 0) return 0;
    d_in = input->shape[input->ndim - 1];
    d_out = output->shape[output->ndim - 1];
    if (d_in <= 0 || d_out <= 0 || linear_node_weight_layout(mutable_node, input, output,
                                                               weight, &out_in) != 0)
        return 0;
    prepared_weight = weight;
    prepared_bias = bias;
    if (weight->dtype == T_F16) {
        float* values = f16_tensor_f32_cache(g_f16wcache, state->idx, weight);
        if (!values) return 0;
        weight_f32 = *weight;
        weight_f32.data = values;
        weight_f32.dtype = T_F32;
        weight_f32.elem_size = sizeof(float);
        prepared_weight = &weight_f32;
    }
    if (bias && bias->dtype == T_F16) {
        float* values = f16_tensor_f32_cache(g_f16bcache, state->idx, bias);
        if (!values) return 0;
        bias_f32 = *bias;
        bias_f32.data = values;
        bias_f32.dtype = T_F32;
        bias_f32.elem_size = sizeof(float);
        prepared_bias = &bias_f32;
    }
    if (prepared_weight->dtype != T_F32 ||
        (prepared_bias && (prepared_bias->dtype != T_F32 || prepared_bias->numel != d_out)))
        return 0;
    total = input->numel / d_in;
    if (total <= 0 || input->numel != total * d_in || output->numel != total * d_out) return 0;
    if (full_tensor) {
        if (total > INT_MAX) return 0;
        row_start = 0;
        row_count = (int)total;
    } else {
        seq_range(total, &row_start, &row_count);
        if (state->is_last && g_active_row < 0 && g_execution_row >= 0 && total > 1) {
            row_start = g_execution_row;
            row_count = 1;
        }
    }
    if (row_start < 0 || row_count <= 0 || (long)row_start + row_count > total) return 0;
    cost = (long)row_count * d_in * d_out;
    if (minimum_cost > 0 && cost < minimum_cost) return 0;
    call->weight_name = weight->name;
    call->input = input->data + (long)row_start * d_in;
    call->weight = prepared_weight->data;
    call->bias = prepared_bias ? prepared_bias->data : NULL;
    call->output = output->data + (long)row_start * d_out;
    call->rows = row_count;
    call->d_in = d_in;
    call->d_out = d_out;
    call->out_in = out_in;
    call->batch_size = input->ndim >= 2 ? input->shape[0] : 1;
    call->row_start = row_start;
    call->total_rows = total;
    return 1;
}

#if VOLVOXAI_ENABLE_NNAPI
static int vx_nnapi_linear_supports(void* user_data, const Node* node) {
    (void)user_data;
    return node && g_use_nnapi ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
}
static int vx_nnapi_linear_run(void* user_data, const Node* node, T* input, T* output) {
    VxOneShotLinear call;
    (void)user_data;
    if (vk_sync_node_inputs((Node*)node) != 0) return VX_BACKEND_ERROR;
    if (!vx_prepare_one_shot_linear(node, input, output, &call,
                                    1L << 20, 0, 0) ||
        call.out_in)
        return VX_BACKEND_DECLINED;
    nnapi_matmul(call.input, call.weight, call.bias, call.output,
                 call.rows, call.d_in, call.d_out);
    if (g_vx_runtime_node.node_backend) *g_vx_runtime_node.node_backend = "nnapi";
    return VX_BACKEND_HANDLED;
}
#endif
#if VOLVOXAI_ENABLE_VULKAN
static int vx_vulkan_linear_supports(void* user_data, const Node* node) {
    (void)user_data;
    return node && g_use_vulkan ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
}
static int vx_vulkan_linear_run(void* user_data, const Node* node, T* input, T* output) {
    VxOneShotLinear call;
    (void)user_data;
    if (vk_sync_node_inputs((Node*)node) != 0) return VX_BACKEND_ERROR;
    if (!vx_prepare_one_shot_linear(node, input, output, &call,
                                    1L << 20, 0, 0) ||
        call.out_in)
        return VX_BACKEND_DECLINED;
    if (!vk_matmul(call.input, call.weight, call.bias, call.output,
                   call.rows, call.d_in, call.d_out)) return VX_BACKEND_DECLINED;
    if (g_vx_runtime_node.node_backend) *g_vx_runtime_node.node_backend = "vulkan";
    return VX_BACKEND_HANDLED;
}
#endif
#if VOLVOXAI_ENABLE_OPENGL
static int vx_opengl_linear_supports(void* user_data, const Node* node) {
    (void)user_data;
    return node && g_use_opengl ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
}
static int vx_opengl_linear_run(void* user_data, const Node* node, T* input, T* output) {
    VxOneShotLinear call;
    (void)user_data;
    if (vk_sync_node_inputs((Node*)node) != 0) return VX_BACKEND_ERROR;
    if (!vx_prepare_one_shot_linear(node, input, output, &call,
                                    1L << 20, 0, 0) ||
        call.out_in)
        return VX_BACKEND_DECLINED;
    if (!opengl_matmul(call.input, call.weight, call.bias, call.output,
                       call.rows, call.d_in, call.d_out)) return VX_BACKEND_DECLINED;
    if (g_vx_runtime_node.node_backend) *g_vx_runtime_node.node_backend = "opengl";
    return VX_BACKEND_HANDLED;
}
#endif
#if VOLVOXAI_ENABLE_CUDA
static int vx_cuda_linear_supports(void* user_data, const Node* node) {
    (void)user_data;
    return g_use_cuda && vx_cuda_linear_node(node)
        ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
}
static int vx_cuda_linear_run(void* user_data, const Node* node, T* input, T* output) {
    VxOneShotLinear call;
    VxAdapterLinearSegment* segments = NULL;
    int segment_count = 0;
    int adapter_linear = g_vx_runtime_node.adapter_linear;
    (void)user_data;
    /* The CUDA graph allocator resolves these host slices as interior views
     * of the full resident tensor.  This keeps prefix/row execution local to
     * the requested sequence window without treating stale host bytes as a
     * second allocation. */
    if (!vx_prepare_one_shot_linear(node, input, output, &call,
                                    0, 0, 1)) {
        return VX_BACKEND_DECLINED;
    }
    if (adapter_linear) {
        segment_count = vx_adapter_run_linear_segments(
            call.weight_name, call.batch_size, call.row_start, call.rows,
            call.total_rows, call.d_in, call.d_out, NULL, 0);
        if (segment_count < 0 ||
            (segment_count > 0 &&
             !(segments = (VxAdapterLinearSegment*)calloc(
                 (size_t)segment_count, sizeof(*segments)))) ||
            (segment_count > 0 && vx_adapter_run_linear_segments(
                call.weight_name, call.batch_size, call.row_start, call.rows,
                call.total_rows, call.d_in, call.d_out, segments,
                segment_count) != segment_count)) {
            free(segments);
            fprintf(stderr,
                    "[CUDA] node '%s' has an invalid routed LoRA binding\n",
                    node && node->op[0] ? node->op : "<invalid>");
            return VX_BACKEND_ERROR;
        }
    }
    if (!cuda_graph_matmul_layout_f32(call.input, call.weight, call.bias,
                                      call.output, call.rows, call.d_in,
                                      call.d_out, call.out_in)) {
        free(segments);
        fprintf(stderr, "[CUDA] node '%s' failed in the CUDA Linear route\n",
                node && node->op[0] ? node->op : "<invalid>");
        return VX_BACKEND_ERROR;
    }
    for (int segment_index = 0; segment_index < segment_count;
         segment_index++) {
        VxAdapterLinearSegment* segment = &segments[segment_index];
        if (!cuda_graph_lora_apply_f32(
                call.input + (long)segment->row_start * call.d_in,
                segment->a, segment->b,
                call.output + (long)segment->row_start * call.d_out,
                segment->row_count, call.d_in, segment->rank,
                call.d_out, segment->scale)) {
            free(segments);
            fprintf(stderr,
                    "[CUDA] node '%s' failed in the CUDA LoRA route\n",
                    node && node->op[0] ? node->op : "<invalid>");
            return VX_BACKEND_ERROR;
        }
    }
    free(segments);
    if (g_vx_runtime_node.node_backend)
        *g_vx_runtime_node.node_backend = adapter_linear
            ? "cuda-linear+lora" : "cuda-linear";
    return VX_BACKEND_HANDLED;
}
#endif

static int vx_cpu_backend_supports(void* user_data, const Node* node) {
    (void)user_data;
    if (!node) return VX_BACKEND_ERROR;
    return vx_kernel_registry_find("native-cpu", node->op)
        ? VX_BACKEND_HANDLED : VX_BACKEND_ERROR;
}
static int vx_cpu_backend_run(void* user_data, const Node* node, T* input, T* output) {
    int result;
    (void)user_data;
    (void)input;
    (void)output;
    if (!node || !g_vx_runtime_node.active || g_vx_cpu_dispatch) return VX_BACKEND_ERROR;
    g_vx_cpu_dispatch = 1;
    result = run_node((Node*)node, g_vx_runtime_node.idx, g_vx_runtime_node.is_last);
    g_vx_cpu_dispatch = 0;
    return result == 0 ? VX_BACKEND_HANDLED : VX_BACKEND_ERROR;
}

/* Incremental-row execution uses this only in a CPU-only session or after a
 * built-in graph backend has relinquished a validated prefix. Routing such a
 * node through the registry merely calls run_node() a second time after
 * repeating its name resolution and immutable candidate checks. Keep this
 * helper private to that ownership contract; ordinary execution continues to
 * exercise the complete backend registry. */
int run_node_cpu_direct(Node* node, int idx, int is_last) {
    int previous = g_vx_cpu_dispatch;
    int result;
    if (!node || previous) return -1;
    g_vx_cpu_dispatch = 1;
    result = run_node(node, idx, is_last);
    g_vx_cpu_dispatch = previous;
    return result;
}

/* Reset is intentionally unconditional. The pre-vtable lifecycle cleared
 * graph slots even after a caller had turned a backend flag off, so stale
 * host pointers cannot survive a graph reload or shutdown. */
#if VOLVOXAI_ENABLE_VULKAN
static void vx_vulkan_graph_reset(void* user_data) { (void)user_data; vk_graph_reset(); }
static void vx_vulkan_graph_begin(void* user_data) { (void)user_data; if (g_use_vulkan) vk_graph_begin_forward(); }
static int vx_vulkan_graph_end(void* user_data) { (void)user_data; return !g_use_vulkan || vk_graph_end_forward() == 0 ? 0 : -1; }
static void vx_vulkan_graph_mark(void* user_data, const void* host, size_t bytes, int is_weight) {
    (void)user_data;
    if (g_use_vulkan) vk_graph_mark_host(host, bytes, is_weight);
}
static int vx_vulkan_graph_sync(void* user_data, const void* host, size_t bytes, int is_weight) {
    (void)user_data;
    return !g_use_vulkan || vk_graph_sync_host(host, bytes, is_weight) ? 1 : 0;
}
#endif
#if VOLVOXAI_ENABLE_OPENGL
static void vx_opengl_graph_reset(void* user_data) { (void)user_data; opengl_graph_reset(); }
static void vx_opengl_graph_begin(void* user_data) { (void)user_data; if (g_use_opengl) opengl_graph_begin_forward(); }
static int vx_opengl_graph_end(void* user_data) { (void)user_data; return !g_use_opengl || opengl_graph_end_forward() == 0 ? 0 : -1; }
static void vx_opengl_graph_mark(void* user_data, const void* host, size_t bytes, int is_weight) {
    (void)user_data;
    if (g_use_opengl) opengl_graph_mark_host(host, bytes, is_weight);
}
static int vx_opengl_graph_sync(void* user_data, const void* host, size_t bytes, int is_weight) {
    (void)user_data;
    return !g_use_opengl || opengl_graph_sync_host(host, bytes, is_weight) ? 1 : 0;
}
#endif
#if VOLVOXAI_ENABLE_METAL
static void vx_metal_graph_reset(void* user_data) { (void)user_data; metal_graph_reset(); }
static void vx_metal_graph_begin(void* user_data) { (void)user_data; if (g_use_metal) metal_graph_begin_forward(); }
static int vx_metal_graph_end(void* user_data) { (void)user_data; return !g_use_metal || metal_graph_end_forward() == 0 ? 0 : -1; }
static void vx_metal_graph_mark(void* user_data, const void* host, size_t bytes, int is_weight) {
    (void)user_data;
    if (g_use_metal) metal_graph_mark_host(host, bytes, is_weight);
}
static int vx_metal_graph_sync(void* user_data, const void* host, size_t bytes, int is_weight) {
    (void)user_data;
    return !g_use_metal || metal_graph_sync_host(host, bytes, is_weight) ? 1 : 0;
}
#endif
#if VOLVOXAI_ENABLE_CUDA
static void vx_cuda_graph_reset(void* user_data) { (void)user_data; cuda_graph_reset(); }
static void vx_cuda_graph_begin(void* user_data) {
    (void)user_data;
    if (g_use_cuda)
        cuda_graph_begin_runtime_forward(g_vx_cuda_replay_eligible,
                                         g_vx_cuda_replay_generation);
}
static int vx_cuda_graph_end(void* user_data) {
    (void)user_data;
    return !g_use_cuda || cuda_graph_end_runtime_forward(g_vx_forward_ok) == 0
        ? 0 : -1;
}
static void vx_cuda_graph_mark(void* user_data, const void* host, size_t bytes,
                               int is_weight) {
    (void)user_data;
    if (g_use_cuda) cuda_graph_mark_host(host, bytes, is_weight);
}
static int vx_cuda_graph_sync(void* user_data, const void* host, size_t bytes,
                              int is_weight) {
    (void)user_data;
    /* A CUDA optimizer slot can remain device-authoritative while policy is
     * handing execution to CPU. Sync must therefore follow slot ownership,
     * not only the currently selected route flag. */
    return cuda_graph_sync_host(host, bytes, is_weight) ? 1 : 0;
}
#endif

#if VOLVOXAI_ENABLE_VULKAN
static const VxBackend g_vx_vulkan_graph_backend = {
    .name = "vulkan-graph", .init = vx_backend_attach,
    .supports = vx_vulkan_graph_supports, .run = vx_vulkan_graph_run,
    .outputs_device_resident = 1,
    .reset = vx_vulkan_graph_reset, .begin_forward = vx_vulkan_graph_begin,
    .end_forward = vx_vulkan_graph_end, .mark_host = vx_vulkan_graph_mark,
    .sync_host = vx_vulkan_graph_sync, .teardown = vx_backend_detach,
};
#endif
#if VOLVOXAI_ENABLE_OPENGL
static const VxBackend g_vx_opengl_graph_backend = {
    .name = "opengl-graph", .init = vx_backend_attach,
    .supports = vx_opengl_graph_supports, .run = vx_opengl_graph_run,
    .outputs_device_resident = 1,
    .reset = vx_opengl_graph_reset, .begin_forward = vx_opengl_graph_begin,
    .end_forward = vx_opengl_graph_end, .mark_host = vx_opengl_graph_mark,
    .sync_host = vx_opengl_graph_sync, .teardown = vx_backend_detach,
};
#endif
#if VOLVOXAI_ENABLE_METAL
static const VxBackend g_vx_metal_graph_backend = {
    .name = "metal-graph", .init = vx_backend_attach,
    .supports = vx_metal_graph_supports, .run = vx_metal_graph_run,
    .outputs_device_resident = 1,
    .reset = vx_metal_graph_reset, .begin_forward = vx_metal_graph_begin,
    .end_forward = vx_metal_graph_end, .mark_host = vx_metal_graph_mark,
    .sync_host = vx_metal_graph_sync, .teardown = vx_backend_detach,
};
#endif
#if VOLVOXAI_ENABLE_CUDA
static const VxBackend g_vx_cuda_graph_backend = {
    .name = "cuda-graph", .init = vx_backend_attach,
    .supports = vx_cuda_graph_supports, .run = vx_cuda_graph_run,
    .outputs_device_resident = 1,
    .reset = vx_cuda_graph_reset, .begin_forward = vx_cuda_graph_begin,
    .end_forward = vx_cuda_graph_end, .mark_host = vx_cuda_graph_mark,
    .sync_host = vx_cuda_graph_sync, .teardown = vx_backend_detach,
};
#endif
#if VOLVOXAI_ENABLE_NNAPI
static const VxBackend g_vx_nnapi_linear_backend = {
    .name = "nnapi-linear", .init = vx_backend_attach,
    .supports = vx_nnapi_linear_supports, .run = vx_nnapi_linear_run,
    .teardown = vx_backend_detach,
};
#endif
#if VOLVOXAI_ENABLE_VULKAN
static const VxBackend g_vx_vulkan_linear_backend = {
    .name = "vulkan-linear", .init = vx_backend_attach,
    .supports = vx_vulkan_linear_supports, .run = vx_vulkan_linear_run,
    .teardown = vx_backend_detach,
};
#endif
#if VOLVOXAI_ENABLE_OPENGL
static const VxBackend g_vx_opengl_linear_backend = {
    .name = "opengl-linear", .init = vx_backend_attach,
    .supports = vx_opengl_linear_supports, .run = vx_opengl_linear_run,
    .teardown = vx_backend_detach,
};
#endif
#if VOLVOXAI_ENABLE_CUDA
static const VxBackend g_vx_cuda_linear_backend = {
    .name = "cuda-linear", .init = vx_backend_attach,
    .supports = vx_cuda_linear_supports, .run = vx_cuda_linear_run,
    .outputs_device_resident = 1,
    .mark_host = vx_cuda_graph_mark, .sync_host = vx_cuda_graph_sync,
    .teardown = vx_backend_detach,
};
#endif
static const VxBackend g_vx_cpu_backend = {
    .name = "cpu", .init = vx_backend_attach,
    .supports = vx_cpu_backend_supports, .run = vx_cpu_backend_run,
    .teardown = vx_backend_detach,
};

static int vx_runtime_backend_registry_init(VxBackendRegistry* registry) {
    /* Graph routes precede the host-resident one-shot Linear routes. Keeping
     * them separate preserves device-first routing and lets an unsupported
     * graph node continue to a matching dense offload or required CPU fallback. */
    const VxBackend* backends[VX_BACKEND_MAX];
    size_t count = 0;
#if VOLVOXAI_ENABLE_VULKAN
    backends[count++] = &g_vx_vulkan_graph_backend;
#endif
#if VOLVOXAI_ENABLE_OPENGL
    backends[count++] = &g_vx_opengl_graph_backend;
#endif
#if VOLVOXAI_ENABLE_METAL
    backends[count++] = &g_vx_metal_graph_backend;
#endif
#if VOLVOXAI_ENABLE_CUDA
    backends[count++] = &g_vx_cuda_graph_backend;
#endif
#if VOLVOXAI_ENABLE_NNAPI
    backends[count++] = &g_vx_nnapi_linear_backend;
#endif
#if VOLVOXAI_ENABLE_VULKAN
    backends[count++] = &g_vx_vulkan_linear_backend;
#endif
#if VOLVOXAI_ENABLE_OPENGL
    backends[count++] = &g_vx_opengl_linear_backend;
#endif
#if VOLVOXAI_ENABLE_CUDA
    backends[count++] = &g_vx_cuda_linear_backend;
#endif
    backends[count++] = &g_vx_cpu_backend;
    return vx_backend_registry_init(registry, backends, count, &g_vx_cpu_backend);
}

static int vx_runtime_backend_registry_ensure(void) {
    return g_vx_backend_registry.ready ? 0 :
        vx_runtime_backend_registry_init(&g_vx_backend_registry);
}

int vx_runtime_backend_stage(void) {
    VxBackendRegistry next = {0};
    if (vx_runtime_backend_registry_init(&next) != 0)
        return -1;
    vx_backend_registry_teardown(&g_vx_backend_registry);
    g_vx_backend_registry = next;
    return 0;
}

#if VOLVOXAI_ENABLE_CUDA
static int vx_cuda_replay_safe_op(const char* op) {
    if (!op || !op[0]) return 0;
    return !strcmp(op, "Linear") || !strcmp(op, "Gemm") ||
        !strcmp(op, "MatMul") || !strcmp(op, "PReLU") ||
        !strcmp(op, "QBatchMatMul") ||
        !strcmp(op, "Conv2D") || !strcmp(op, "Add") ||
        !strcmp(op, "Concat") || !strcmp(op, "MaxPool2D") ||
        !strcmp(op, "ResizeNearest2D") || !strcmp(op, "Reshape") ||
        !strcmp(op, "Transpose") ||
        !strcmp(op, "Sigmoid") || !strcmp(op, "QConv2D") ||
        !strcmp(op, "QAdd") || !strcmp(op, "QuantizeLinear") ||
        !strcmp(op, "DequantizeLinear") ||
        !strcmp(op, "RequantizeLinear");
}
#endif

int vx_runtime_backend_cuda_replay_eligible(void) {
#if !VOLVOXAI_ENABLE_CUDA
    return 0;
#else
    if (!g_use_cuda || g_debug || g_nn <= 0 ||
        !vx_runtime_full_graph_execution_eligible() ||
        volvoxai_engine_adapter_effect_active_locked()) return 0;
#if VOLVOXAI_ENABLE_VULKAN
    if (g_use_vulkan) return 0;
#endif
#if VOLVOXAI_ENABLE_OPENGL
    if (g_use_opengl) return 0;
#endif
#if VOLVOXAI_ENABLE_METAL
    if (g_use_metal) return 0;
#endif
    for (int node = 0; node < g_nn; node++) {
        Node* current = &g_n[node];
        if (current->skip || current->disabled) continue;
        if (!vx_cuda_replay_safe_op(current->op)) return 0;
        for (int input = 0; input < current->nin; input++)
            if (vx_adapter_run_has_target(current->ins[input].name)) return 0;
    }
    return 1;
#endif
}

void vx_runtime_backend_reset(void) {
    if (vx_runtime_backend_registry_ensure() == 0) {
        vx_backend_registry_reset(&g_vx_backend_registry);
        release_retired_f16_storage();
    }
}
void vx_runtime_backend_reset_transients(void) {
#if VOLVOXAI_ENABLE_CUDA
    /* A configured built-in CUDA route is the only graph backend whose slots
     * survive activation-address changes. Other routes retain the full reset
     * contract used before persistent CUDA residency existed. */
    if (g_use_cuda) {
        cuda_graph_release_transients();
        release_retired_f16_storage();
        return;
    }
#endif
    vx_runtime_backend_reset();
}
void vx_runtime_backend_teardown(void) {
    vx_backend_registry_teardown(&g_vx_backend_registry);
    release_retired_f16_storage();
    vx_kernels_shutdown();
}
void vx_runtime_backend_begin_forward(int ordinary_static_replay_eligible,
                                      uint64_t model_generation) {
    g_vx_cuda_replay_eligible = ordinary_static_replay_eligible;
    g_vx_cuda_route_tracking = ordinary_static_replay_eligible;
    g_vx_cuda_replay_generation = model_generation;
    g_vx_forward_ok = 0;
    if (vx_runtime_backend_registry_ensure() == 0)
        vx_backend_registry_begin_forward(&g_vx_backend_registry);
#if VOLVOXAI_ENABLE_CUDA && VOLVOXAI_ENABLE_TRAINING
    if (g_use_cuda && cuda_profile_route_tracking_active())
        g_vx_cuda_route_tracking = 1;
#endif
}
int vx_runtime_backend_prepare_forward(void) {
#if VOLVOXAI_ENABLE_CUDA
    if (g_use_cuda && cuda_graph_prepare_runtime_forward() != 0) return -1;
#endif
    return 0;
}
int vx_runtime_backend_end_forward(int forward_ok) {
    int result;
    g_vx_forward_ok = forward_ok;
    result = vx_runtime_backend_registry_ensure() == 0
        ? vx_backend_registry_end_forward(&g_vx_backend_registry) : -1;
    g_vx_cuda_replay_eligible = 0;
    g_vx_cuda_route_tracking = 0;
    g_vx_cuda_replay_generation = 0;
    g_vx_forward_ok = 0;
    return result;
}
void vx_runtime_backend_mark_host(const void* host, size_t bytes, int is_weight) {
    if (vx_runtime_backend_registry_ensure() == 0)
        vx_backend_registry_mark_host(&g_vx_backend_registry, host, bytes, is_weight);
}
int vx_runtime_backend_sync_host(const void* host, size_t bytes, int is_weight) {
    return vx_runtime_backend_registry_ensure() == 0
        ? vx_backend_registry_sync_host(&g_vx_backend_registry, host, bytes, is_weight) : -1;
}
int vx_runtime_backend_has_graph(void) {
    return
#if VOLVOXAI_ENABLE_VULKAN
        g_use_vulkan ||
#endif
#if VOLVOXAI_ENABLE_OPENGL
        g_use_opengl ||
#endif
#if VOLVOXAI_ENABLE_METAL
        g_use_metal ||
#endif
#if VOLVOXAI_ENABLE_CUDA
        g_use_cuda ||
#endif
        0;
}

static int vx_incremental_row_tensor_3d(const T* tensor, int row) {
    return tensor && tensor->ndim == 3 && tensor->shape[0] == 1 &&
        tensor->shape[1] > row && tensor->shape[2] > 0;
}

/* Compatibility is queried before the hybrid GPU-to-CPU ownership handoff,
 * while g_dirty contains only the changed roots.  Mirror dependency
 * propagation through the nodes preceding this one so a produced right-hand
 * QBatchMatMul operand cannot be mistaken for a retained seed tensor. */
static int vx_incremental_tensor_dirty_before_node(
        const T* target, int node_index) {
    unsigned char dirty[MAXT];
    int target_index = t_index(target);
    if (!target || target_index < 0 || target_index >= g_nt ||
        node_index < 0 || node_index > g_nn || !g_incremental_plan.valid ||
        g_incremental_plan.tensor_count != g_nt ||
        g_incremental_plan.node_count != g_nn)
        return 1;
    memcpy(dirty, g_dirty, (size_t)g_nt);
    for (int node = 0; node < node_index; node++) {
        int should_run = 0;
        if (g_incremental_plan.input_counts[node] > MAXIN ||
            g_incremental_plan.output_counts[node] > MAXIN + 1)
            return 1;
        for (int input = 0;
             input < g_incremental_plan.input_counts[node]; input++) {
            int index = g_incremental_plan.input_indices[node][input];
            if (index < 0 || index >= g_nt) return 1;
            if (dirty[index]) should_run = 1;
        }
        if (!should_run) continue;
        for (int output = 0;
             output < g_incremental_plan.output_counts[node]; output++) {
            int index = g_incremental_plan.output_indices[node][output];
            if (index < 0 || index >= g_nt) return 1;
            dirty[index] = 1;
        }
    }
    return dirty[target_index] != 0;
}

/* A device seed may be handed to the existing CPU row implementation only
 * when every reachable node has the exact canonical physical-byte spelling
 * already covered by that implementation. Keep this preflight beside the
 * strict validators it calls: accepting a new op here is an explicit cache
 * ownership decision, not a generic shape heuristic. */
int vx_runtime_node_incremental_row_compatible(Node* node, int idx, int row) {
    T* output;
    if (!node || row < 0 || node->skip || node->disabled) return 0;
    output = t_find(node->out);
    if (!output) return 0;

    if (native_qlinear_op(node->op)) {
        QLinearMetadata* meta;
        T* input;
        T* weight;
        T* bias;
        if (idx < 0 || idx >= MAXN) return 0;
        meta = &g_qlinear_meta[idx];
        input = qlinear_node_input(node);
        weight = nin(node, "weight");
        bias = nin(node, "bias");
        if (!meta->valid || !vx_incremental_row_tensor_3d(input, row) ||
            !vx_incremental_row_tensor_3d(output, row) ||
            input->shape[1] != output->shape[1] ||
            input->shape[2] != meta->d_in || output->shape[2] != meta->d_out ||
            !weight || !bias || weight->ndim != 2 || bias->ndim != 1 ||
            input->dtype != meta->input_dtype || weight->dtype != meta->weight_dtype ||
            output->dtype != meta->output_dtype || bias->dtype != T_I32 ||
            weight->shape[0] != meta->d_out || weight->shape[1] != meta->d_in ||
            bias->numel != meta->d_out || !input->quantization.valid ||
            !output->quantization.valid || !meta->weight_scales ||
            !meta->weight_zero_points || !meta->bias) return 0;
        return 1;
    }

    if (!strcmp(node->op, "QEmbedding")) {
        QEmbeddingPhysicalDescriptor descriptor;
        return qembedding_physical_validate(idx, node, output, &descriptor) == 0 &&
            descriptor.tokens->ndim == 2 && descriptor.tokens->shape[0] == 1 &&
            descriptor.tokens->shape[1] > row &&
            vx_incremental_row_tensor_3d(output, row);
    }

    if (!strcmp(node->op, "QuantizeLinear")) {
        T* scale = nin(node, "scale");
        T* zero_point = nin(node, "zero_point");
        if (!quantize_physical_row_compatible(node, output, row)) return 0;
        if (g_cache_valid &&
            ((scale && vx_incremental_tensor_dirty_before_node(scale, idx)) ||
             (zero_point &&
              vx_incremental_tensor_dirty_before_node(zero_point, idx)))) return 0;
        return 1;
    }

    if (!strcmp(node->op, "DequantizeLinear")) {
        T* scale = nin(node, "scale");
        T* zero_point = nin(node, "zero_point");
        if (!dequantize_physical_row_compatible(node, output, row)) return 0;
        if (g_cache_valid &&
            ((scale && vx_incremental_tensor_dirty_before_node(scale, idx)) ||
             (zero_point &&
              vx_incremental_tensor_dirty_before_node(zero_point, idx)))) return 0;
        return 1;
    }

    if (!strcmp(node->op, "QBatchMatMul")) {
        QBatchMatMulPhysicalDescriptor descriptor;
        QBatchMatMulPhysicalRowDescriptor row_descriptor;
        int b_dirty;
        /* Model capability probing happens before a seed exists and uses a
         * synthetic dirty root.  The real post-seed handoff is revalidated
         * against the current dependency closure. */
        if (qbatch_matmul_physical_validate(
                node, output, &descriptor) != 0) return 0;
        b_dirty = g_cache_valid
            ? vx_incremental_tensor_dirty_before_node(descriptor.b, idx)
            : 0;
        return qbatch_matmul_physical_row_validate(
                   &descriptor, row, b_dirty, &row_descriptor) == 0;
    }

    if (!strcmp(node->op, "Expand")) {
        PhysicalByteExpandDescriptor descriptor;
        if (physical_shape_expand_validate(node, output, &descriptor) != 0 ||
            descriptor.input_rank != 3 || descriptor.output_rank != 3 ||
            descriptor.input_shape[0] != 1u ||
            descriptor.input_shape[1] != 1u ||
            descriptor.output_shape[0] != 1u ||
            descriptor.output_shape[1] <= (uint32_t)row ||
            descriptor.input_shape[2] != descriptor.output_shape[2] ||
            (g_cache_valid && vx_incremental_tensor_dirty_before_node(
                descriptor.input, idx))) return 0;
        return 1;
    }

    if (!strcmp(node->op, "QAdd")) {
        T* a = nin(node, "a");
        T* b = nin(node, "b");
        uint32_t a_dtype;
        uint32_t b_dtype;
        uint32_t output_dtype;
        uint32_t relu;
        if (!a) a = nin(node, "input");
        if (!a) a = nin(node, "x");
        if (!b) b = nin(node, "y");
        return vx_incremental_row_tensor_3d(a, row) &&
            vx_incremental_row_tensor_3d(b, row) &&
            vx_incremental_row_tensor_3d(output, row) &&
            physical_byte_tensor(a, &a_dtype) == 0 &&
            physical_byte_tensor(b, &b_dtype) == 0 &&
            physical_byte_tensor(output, &output_dtype) == 0 &&
            physical_same_shape(a, b) && physical_same_shape(a, output) &&
            physical_relu_parameter(node->params, &relu) == 0;
    }

    if (!strcmp(node->op, "QGELU")) {
        T* input;
        uint32_t input_dtype;
        uint32_t output_dtype;
        return qgelu_physical_validate(node, output, &input, &input_dtype,
                                        &output_dtype) == 0 &&
            vx_incremental_row_tensor_3d(input, row) &&
            vx_incremental_row_tensor_3d(output, row);
    }

    if (!strcmp(node->op, "QLayerNorm")) {
        QLayerNormPhysicalDescriptor descriptor;
        return qlayernorm_physical_validate(node, output, &descriptor) == 0 &&
            vx_incremental_row_tensor_3d(descriptor.input, row) &&
            vx_incremental_row_tensor_3d(output, row);
    }

    if (!strcmp(node->op, "QSDPA")) {
        QSDPAPhysicalDescriptor descriptor;
        return qsdpa_physical_validate(node, output, &descriptor) == 0 &&
            descriptor.batch == 1 && descriptor.q->ndim == 3 &&
            descriptor.seq_q > (uint32_t)row &&
            (!descriptor.causal || (uint32_t)row + 1u <= descriptor.seq_kv) &&
            vx_incremental_row_tensor_3d(output, row);
    }

    if (!strcmp(node->op, "QArgMax")) {
        QArgMaxPhysicalDescriptor descriptor;
        return qargmax_physical_validate(node, output, &descriptor) == 0 &&
            descriptor.input->ndim == 3 && descriptor.input->shape[0] == 1 &&
            output->ndim == 2 && output->shape[0] == 1 &&
            output->shape[1] > row && descriptor.inner == 1;
    }
    return 0;
}

// ---- run one node ---------------------------------------------------------
int run_node(Node* n, int idx, int is_last) {
    if (idx >= 0 && idx < MAXN)
        vx_engine_state_current()->runtime_route_backend[idx] = NULL;
    if (n->skip || n->disabled) return 0;  // node fused/aliased or disabled by graph patching
    cJSON* p = n->params;
    T* o = t_find(n->out);
    T* in = nin(n, "input");
    const char* op = n->op;
    if (!strcmp(op, "SDPA")) {
        T* qkv = nin(n, "qkv");
        if (!qkv || (qkv->ndim != 2 && qkv->ndim != 3) ||
            (qkv->ndim == 3 && qkv->shape[0] <= 0)) return -1;
    } else if (!strcmp(op, "CrossSDPA")) {
        T* q = nin(n, "q"); T* k = nin(n, "k"); T* v = nin(n, "v");
        if (!q || !k || !v ||
            (q->ndim != 2 && q->ndim != 3) || k->ndim != q->ndim || v->ndim != q->ndim ||
            (q->ndim == 3 && (q->shape[0] <= 0 || k->shape[0] != q->shape[0] ||
                              v->shape[0] != q->shape[0]))) return -1;
    }
    int is_linear = !strcmp(op, "MatMul") || !strcmp(op, "Gemm") || !strcmp(op, "Linear");
    T* adapter_weight = is_linear ? nin(n, "weight") : NULL;
    int adapter_linear = adapter_weight && vx_adapter_run_has_target(adapter_weight->name);
    int native_physical_qlinear = native_qlinear_op(op);
    if (native_physical_qlinear && (idx < 0 || idx >= MAXN || !g_qlinear_meta[idx].valid)) return -1;
    QBatchMatMulPhysicalDescriptor qbatch_matmul_descriptor;
    int native_physical_qbatch_matmul =
        !strcmp(op, "QBatchMatMul") &&
        qbatch_matmul_physical_validate(
            n, o, &qbatch_matmul_descriptor) == 0;
    int native_physical_qembedding = native_qembedding_op(op);
    if (native_physical_qembedding &&
        (idx < 0 || idx >= MAXN || !g_qembedding_meta[idx].valid)) return -1;
    int native_physical_qconv = native_qconv2d_op(op) && idx >= 0 && idx < MAXN &&
        g_qconv_meta[idx].valid;
    int native_physical_qadd = !strcmp(op, "QAdd") && qadd_physical_candidate(n, o);
    int native_physical_qsilu = !strcmp(op, "QSiLU") && qsilu_physical_candidate(n, o);
    int native_physical_qgelu = !strcmp(op, "QGELU") && qgelu_physical_candidate(n, o);
    int native_physical_qgroupnorm = !strcmp(op, "QGroupNorm") &&
        qgroupnorm_physical_candidate(n, o);
    int native_physical_qlayernorm = !strcmp(op, "QLayerNorm") &&
        qlayernorm_physical_candidate(n, o);
    int native_physical_qsdpa = !strcmp(op, "QSDPA") && qsdpa_physical_candidate(n, o);
    int native_physical_qargmax = !strcmp(op, "QArgMax") &&
        qargmax_physical_candidate(n, o);
    int native_physical_qmaskedmean = !strcmp(op, "QMaskedMean") &&
        qmaskedmean_physical_candidate(n, o);
    int native_physical_requantize = !strcmp(op, "RequantizeLinear") &&
        requantize_physical_candidate(n, o);
    int native_physical_quantize = !strcmp(op, "QuantizeLinear") && quantize_physical_candidate(o);
    int native_physical_dequantize = !strcmp(op, "DequantizeLinear") &&
        dequantize_physical_candidate(n);
    int native_physical_shape = physical_shape_candidate(n, o);
    int log_node = g_debug;
    double node_t0 = log_node ? volvoxai_engine_now_ms() : 0.0;
    const char* node_backend = "cpu";

    // Row window for elementwise/shape-preserving ops (in and out share layout).
    int e_D = o ? (o->ndim ? o->shape[o->ndim - 1] : (int)o->numel) : 1;
    long e_total = (o && e_D) ? o->numel / e_D : 1;
    int e_r0, e_rc; seq_range(e_total, &e_r0, &e_rc);
    long e_off = (long)e_r0 * e_D; int e_cnt = e_rc * e_D;

    int ran_gpu_graph = 0;
    if (!g_vx_cpu_dispatch) {
        const VxBackend* selected = NULL;
        int dispatch_result;
        g_vx_runtime_node = (VxRuntimeNodeState){
            .active = 1, .idx = idx, .is_last = is_last,
            .adapter_linear = adapter_linear,
            .native_physical_qlinear = native_physical_qlinear,
            .native_physical_qbatch_matmul =
                native_physical_qbatch_matmul,
            .native_physical_qembedding = native_physical_qembedding,
            .native_physical_qconv = native_physical_qconv,
            .native_physical_qadd = native_physical_qadd,
            .native_physical_qsilu = native_physical_qsilu,
            .native_physical_qgelu = native_physical_qgelu,
            .native_physical_qgroupnorm = native_physical_qgroupnorm,
            .native_physical_qlayernorm = native_physical_qlayernorm,
            .native_physical_qsdpa = native_physical_qsdpa,
            .native_physical_qargmax = native_physical_qargmax,
            .native_physical_qmaskedmean = native_physical_qmaskedmean,
            .native_physical_requantize = native_physical_requantize,
            .native_physical_quantize = native_physical_quantize,
            .native_physical_dequantize = native_physical_dequantize,
            .native_physical_shape = native_physical_shape,
            .node_backend = &node_backend,
        };
        if (vx_runtime_backend_registry_ensure() != 0) {
            g_vx_runtime_node.active = 0;
            return -1;
        }
        dispatch_result = vx_backend_registry_dispatch(&g_vx_backend_registry,
                                                       n, in, o, &selected);
        g_vx_runtime_node.active = 0;
#if VOLVOXAI_ENABLE_TRAINING
        dynamic_autograd_note_forward_backend(selected ? selected->name : NULL);
#endif
#if VOLVOXAI_ENABLE_CUDA
        if (g_vx_cuda_route_tracking) {
            int built_in_cuda = dispatch_result == VX_BACKEND_HANDLED &&
                selected && (selected == &g_vx_cuda_graph_backend ||
                             selected == &g_vx_cuda_linear_backend);
            cuda_graph_note_node_route(built_in_cuda);
        }
#endif
        if (dispatch_result != VX_BACKEND_HANDLED || !selected) return -1;
        if (selected == &g_vx_cpu_backend) {
            if (idx >= 0 && idx < MAXN)
                vx_engine_state_current()->runtime_route_backend[idx] =
                    selected->name;
            return 0;
        }
        ran_gpu_graph = selected->outputs_device_resident;
        goto done;
    }
    /* Hybrid row execution has atomically synchronized the complete clean
     * boundary and every retained output prefix. From that point onward the
     * changed closure is host-owned until the incremental cache is reset, so
     * repeated device-slot probes cannot discover a newer input. */
    if (!vx_incremental_hybrid_row_active_locked() &&
        vk_sync_node_inputs(n) != 0) return -1;
    int sequence_result = vx_sequence_node_run(n, e_off, e_cnt, &node_backend);
    if (sequence_result == VX_SEQUENCE_ERROR) return -1;

    if (sequence_result == VX_SEQUENCE_HANDLED) {
        /* The private sequence runtime owns validation and execution. */
    } else if (native_physical_qlinear) {
        if (qlinear_physical_forward(idx, n, o, is_last) != 0) return -1;
        node_backend = "cpu-qlinear-w8a8";
    } else if (native_physical_qembedding) {
        if (qembedding_physical_forward(idx, n, o) != 0) return -1;
        node_backend = "cpu-qembedding-w8a8";
    } else if (native_physical_qconv) {
        if (qconv2d_physical_forward(idx, n, o) != 0) return -1;
        node_backend = "cpu-qconv-w8a8";
    } else if (native_physical_qadd) {
        if (qadd_physical_forward(n, o) != 0) return -1;
        node_backend = "cpu-qadd-w8a8";
    } else if (native_physical_qsilu) {
        if (qsilu_physical_forward(n, o) != 0) return -1;
        node_backend = "cpu-qsilu-w8a8";
    } else if (native_physical_qgelu) {
        if (qgelu_physical_forward(n, o) != 0) return -1;
        node_backend = "cpu-qgelu-w8a8";
    } else if (native_physical_qgroupnorm) {
        if (qgroupnorm_physical_forward(n, o) != 0) return -1;
        node_backend = "cpu-qgroupnorm-w8a8";
    } else if (native_physical_qlayernorm) {
        if (qlayernorm_physical_forward(n, o) != 0) return -1;
        node_backend = "cpu-qlayernorm-w8a8";
    } else if (native_physical_qsdpa) {
        if (qsdpa_physical_forward(n, o) != 0) return -1;
        node_backend = "cpu-qsdpa-w8a8";
    } else if (native_physical_qargmax) {
        if (qargmax_physical_forward(n, o) != 0) return -1;
        node_backend = "cpu-qargmax-w8a8";
    } else if (native_physical_qmaskedmean) {
        if (qmaskedmean_physical_forward(n, o) != 0) return -1;
        node_backend = "cpu-qmaskedmean-w8a8";
    } else if (native_physical_requantize) {
        if (requantize_physical_forward(n, o) != 0) return -1;
        node_backend = "cpu-requantize-w8a8";
    } else if (native_physical_quantize) {
        if (quantize_physical_forward(n, o) != 0) return -1;
        node_backend = "cpu-quantize-w8a8";
    } else if (native_physical_dequantize) {
        if (dequantize_physical_forward(n, o) != 0) return -1;
        node_backend = "cpu-dequantize-w8a8";
    } else if (native_physical_shape) {
        if (physical_shape_forward(n, idx, o) != 0) return -1;
        node_backend = "cpu-shape-w8a8";
    } else if (!strcmp(op, "MoERouter")) {
        if (moe_router_forward(n, 0) != 0) return -1;
        node_backend = "cpu-moe-router";
    } else if (!strcmp(op, "MoELinear")) {
        if (moe_linear_forward(n, o, 0) != 0) return -1;
        node_backend = "cpu-moe-linear";
    } else if (!strcmp(op, "QConv2D")) {
        T* w = nin(n, "weight"); T* ws = nin(n, "weight_scale"); T* wzp = nin(n, "weight_zero_point"); T* b = nin(n, "bias");
        int sy, sx, py, px, dy, dx; p_pair(p, "stride", 1, &sy, &sx); p_pair(p, "padding", 0, &py, &px); p_pair(p, "dilation", 1, &dy, &dx);
        int pads[4] = {py, px, py, px}; if (p && cJSON_GetObjectItem(p, "pads")) p_iarr(p, "pads", pads, 4, 0);
        int groups = p_int(p, "groups", 1);
        int relu = n->fuse_relu6 ? 2 : p_int(p, "relu", 0);   // 2 = ReLU6, fused from a following Clip(0,6)
        int weight_only;
        if (qconv2d_weight_only_parameter(n, &weight_only) != 0 ||
            !weight_only || qconv2d_weight_only_validate(n) != 0) return -1;
        const char* weight_layout = p_str(p, "weight_layout", "OHWI");
        materialize_tensor_f32(in);
        if (qconv2d_weight_only_f32(idx, in, o, w, ws, wzp, b, sy, sx, pads,
                                    groups, dy, dx, relu, p, weight_layout) != 0) return -1;
        node_backend = "cpu-qconv-weight-only";
    } else if (!strcmp(op, "Conv2D")) {
        materialize_tensor_f32(in);
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        int sy, sx, py, px, dy, dx; p_pair(p, "stride", 1, &sy, &sx); p_pair(p, "padding", 0, &py, &px); p_pair(p, "dilation", 1, &dy, &dx);
        int pads[4] = {py, px, py, px}; if (p && cJSON_GetObjectItem(p, "pads")) p_iarr(p, "pads", pads, 4, 0);
        int groups = p_int(p, "groups", 1), relu = n->fuse_relu6 ? 2 : p_int(p, "relu", 0);
        if (!w) return -1;
            T* bp = b;
            T* add = NULL;
            const GraphNodeFusion* conv_add_fusion = graph_opt_node_fusion(idx, GRAPH_FUSION_CONV_ADD);
            if (conv_add_fusion) {
                add = t_find(conv_add_fusion->aux_tensor);
                if (!add || add->dtype != T_F32 || add->numel != o->numel || add->data == o->data) return -1;
                materialize_tensor_f32(add);
            }
            T wtmp, btmp;
            T* wp = prepare_conv_weight_image(idx, w, p, in->shape[3], o->shape[3], groups, &wtmp);
            if (!wp) return -1;
            if (b && b->dtype == T_F16) {
                float* bf = f16_tensor_f32_cache(g_f16bcache, idx, b);
                if (!bf) return -1;
                btmp = *b; btmp.data = bf; btmp.dtype = T_F32; btmp.elem_size = sizeof(float);
                bp = &btmp;
            }
            const GraphNodeFusion* dw_pw_fusion = graph_opt_node_fusion(idx, GRAPH_FUSION_DEPTHWISE_POINTWISE);
            int pw_idx = dw_pw_fusion ? dw_pw_fusion->peer_idx : -1;
            if (pw_idx >= 0) {
                Node* pw = &g_n[pw_idx];
                T* po = t_find(pw->out);
                T* pw_w = nin(pw, "weight");
                T* pw_b = nin(pw, "bias");
                if (!po || !pw_w) return -1;
                T pw_wtmp, pw_btmp;
                T* pw_wp = prepare_conv_weight_image(pw_idx, pw_w, pw->params, o->shape[3], po->shape[3], 1, &pw_wtmp);
                if (!pw_wp) return -1;
                T* pw_bp = pw_b;
                if (pw_b && pw_b->dtype == T_F16) {
                    float* pbf = f16_tensor_f32_cache(g_f16bcache, pw_idx, pw_b);
                    if (!pbf) return -1;
                    pw_btmp = *pw_b; pw_btmp.data = pbf; pw_btmp.dtype = T_F32; pw_btmp.elem_size = sizeof(float);
                    pw_bp = &pw_btmp;
                }
                T* pw_add = NULL;
                const GraphNodeFusion* pw_add_fusion = graph_opt_node_fusion(pw_idx, GRAPH_FUSION_CONV_ADD);
                if (pw_add_fusion) {
                    pw_add = t_find(pw_add_fusion->aux_tensor);
                    if (!pw_add || pw_add->dtype != T_F32 || pw_add->numel != po->numel || pw_add->data == po->data) return -1;
                    materialize_tensor_f32(pw_add);
                }
                int pw_relu = pw->fuse_relu6 ? 2 : p_int(pw->params, "relu", 0);
                if (!conv2d_depthwise_pointwise_f32_image(idx, pw_idx, in, po, wp, bp, pw_wp, pw_bp, pw_add,
                                                          sy, sx, pads, relu, pw_relu)) return -1;
                o = po;
                node_backend = pw_add ? "cpu-conv+dw-pw+add" : "cpu-conv+dw-pw";
                goto done;
            }
            if (wp != w) {
                node_backend = add ? (w->dtype == T_F16 ? "cpu-f16w-pack+add" : "cpu-pack+add")
                                   : (w->dtype == T_F16 ? "cpu-f16w-pack" : "cpu-pack");
            } else {
                node_backend = add ? "cpu-conv+add" : "cpu-conv";
            }
            conv2d_f32_image(idx, in, o, wp, bp, add, sy, sx, pads, groups, dy, dx, relu);
    } else if (!strcmp(op, "QBatchMatMul")) {
        if (qbatch_matmul_physical_forward(n, o) != 0) return -1;
        node_backend = "cpu-qbatch-matmul-w8a8";
    } else if (!strcmp(op, "BatchMatMul")) {
        T* a = nin(n, "a");
        T* b = nin(n, "b");
        materialize_tensor_f32(a);
        materialize_tensor_f32(b);
        if (batch_matmul_f32_forward(n) != 0) return -1;
    } else if (!strcmp(op, "MatMul") || !strcmp(op, "Gemm") || !strcmp(op, "Linear")) {
        /* Generic Linear/MatMul/Gemm keeps F32 activations. I8/U8 weights are
         * the current W8A32 form and obtain scale/zero-point values only from
         * named tensor operands. Fully quantized execution uses QLinear. */
        materialize_tensor_f32(in);
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        if (!in || !w || !o || in->dtype != T_F32 || o->dtype != T_F32) return -1;
        int d_in = in->shape[in->ndim - 1], d_out = o->shape[o->ndim - 1];
        int out_in = 0;
        if (linear_node_weight_layout(n, in, o, w, &out_in) != 0) return -1;
        T wtmp, btmp;
        T* wp = w;
        T* bp = b;
        if (w->dtype == T_F16) {
            float* values = f16_tensor_f32_cache(g_f16wcache, idx, w);
            if (!values) return -1;
            wtmp = *w; wtmp.data = values; wtmp.dtype = T_F32; wtmp.elem_size = sizeof(float);
            wp = &wtmp;
        }
        if (b && b->dtype == T_F16) {
            float* values = f16_tensor_f32_cache(g_f16bcache, idx, b);
            if (!values) return -1;
            btmp = *b; btmp.data = values; btmp.dtype = T_F32; btmp.elem_size = sizeof(float);
            bp = &btmp;
        }
        if (bp && (bp->dtype != T_F32 || bp->numel != d_out)) return -1;
        long total = in->numel / d_in;
        int r0, rc; seq_range(total, &r0, &rc);
        // A selected graph-output row lets the final projection compute only
        // that row.
        if (is_last && g_active_row < 0 && g_execution_row >= 0 && total > 1) { r0 = g_execution_row; rc = 1; }
        const float* in_p = in->data + (long)r0 * d_in;
        float* out_p = o->data + (long)r0 * d_out;
        const float* base_bias = adapter_linear ? NULL : (bp ? bp->data : NULL);
        if (w->dtype == T_I8 || w->dtype == T_U8) {
            if (quantized_linear_f32(idx, n, in, w, bp, o, r0, rc, d_in, d_out, out_in, !adapter_linear) != 0) return -1;
            node_backend = adapter_linear ? "cpu-qlinear+adapter" : "cpu-qlinear";
        } else if (wp->dtype != T_F32) {
            return -1;
        }
        else if (wp->dtype == T_F32) {
            int packed_ran = 0;
            /* Only immutable model weights enter the persistent node cache.
             * Dynamic graph values retain the scalar reference path. */
            if (volvoxai_engine_tensor_is_model_weight_locked(w->name)) {
                const float* packed_weight = vx_gemm_f32_pack_cache(
                    &vx_engine_state_current()->gemm_f32_cache,
                    idx, wp->data, (uint32_t)d_in, (uint32_t)d_out, out_in);
                if (packed_weight) {
                    packed_ran = vx_gemm_f32_run_packed(
                        in_p, packed_weight, base_bias, out_p,
                        (uint32_t)rc, (uint32_t)d_in, (uint32_t)d_out);
                }
            }
            if (!packed_ran) {
                if (out_in) matmul_f32_out_in(in_p, wp->data, base_bias, out_p, rc, d_in, d_out);
                else matmul_f32(in_p, wp->data, base_bias, out_p, rc, d_in, d_out);
            }
            if (adapter_linear) node_backend = "cpu-linear+adapter";
        }
        if (adapter_linear && vx_adapter_apply_linear(w->name, in_p, wp->data, wp->dtype,
                                                       bp ? bp->data : NULL, out_p, rc, d_in, d_out,
                                                       out_in, in->ndim >= 2 ? in->shape[0] : 1,
                                                       r0, total) < 0) {
            return -1;
        }
    } else if (!strcmp(op, "ReduceSum") || !strcmp(op, "ReduceMean")) {
        T* source = NULL;
        long rows = 0;
        int width = 0;
        float scale = 1.0f;
        if (last_axis_reduction_dimensions(n, o, &source, &rows, &width, &scale) != 0)
            return -1;
        materialize_tensor_f32(source);
        for (long row = 0; row < rows; row++) {
            double sum = 0.0;
            for (int column = 0; column < width; column++)
                sum += source->data[row * width + column];
            o->data[row] = (float)sum * scale;
        }
    } else if (!strcmp(op, "ArgMax")) {
        materialize_tensor_f32(in);
        if (argmax_forward_f32(n, o) != 0) return -1;
    } else if (!strcmp(op, "BatchNorm2D")) {
        materialize_tensor_f32(in);
        T* w = nin(n, "weight"); T* b = nin(n, "bias"); T* rm = nin(n, "running_mean"); T* rv = nin(n, "running_var");
        batch_norm2d_f32(in->data, w->data, b->data, rm->data, rv->data, o->data,
                         in->shape[0], in->shape[1], in->shape[2], in->shape[3], p_flt(p, "eps", 1e-5f));
    } else if (!strcmp(op, "GroupNorm")) {
        materialize_tensor_f32(in);
        T* weight = nin(n, "weight");
        if (!weight) weight = nin(n, "scale");
        T* bias = nin(n, "bias");
        materialize_tensor_f32(weight);
        materialize_tensor_f32(bias);
        int groups = p_int(p, "num_groups", p_int(p, "groups", 1));
        if (groupnorm_forward_f32(in, weight, bias, o, groups,
                                  p_flt(p, "eps", 1.0e-5f)) != 0) return -1;
    } else if (!strcmp(op, "MaxPool2D")) {
        int ky, kx, sy, sx, py, px; p_pair(p, "kernel", 1, &ky, &kx); p_pair(p, "stride", 1, &sy, &sx); p_pair(p, "padding", 0, &py, &px);
        materialize_tensor_f32(in);
        maxpool2d_f32_image(in, o, ky, kx, sy, sx, py, px);
        node_backend = "cpu-maxpool";
    } else if (!strcmp(op, "GlobalAveragePool")) {
        materialize_tensor_f32(in);
        global_average_pool_f32(in->data, o->data, in->shape[0], in->shape[1], in->shape[2], in->shape[3]);
    } else if (!strcmp(op, "Add"))  {
        materialize_node_inputs(n);
        if (binop(n, VX_PORTABLE_BINARY_ADD) != 0) return -1;
    } else if (!strcmp(op, "Mul")) {
        materialize_node_inputs(n);
        if (binop(n, VX_PORTABLE_BINARY_MUL) != 0) return -1;
    } else if (!strcmp(op, "Sub")) {
        materialize_node_inputs(n);
        if (binop(n, VX_PORTABLE_BINARY_SUB) != 0) return -1;
    } else if (!strcmp(op, "Div")) {
        materialize_node_inputs(n);
        if (binop(n, VX_PORTABLE_BINARY_DIV) != 0) return -1;
    } else if (!strcmp(op, "Equal")) {
        if (compare_i32_forward(n, 0) != 0) return -1;
    } else if (!strcmp(op, "GreaterOrEqual")) {
        if (compare_i32_forward(n, 1) != 0) return -1;
    } else if (!strcmp(op, "Not")) {
        if (not_i32_forward(n) != 0) return -1;
    } else if (!strcmp(op, "Where")) {
        materialize_node_inputs(n);
        if (where_plain_32_forward(n) != 0) return -1;
    } else if (!strcmp(op, "Gather")) {
        materialize_tensor_f32(node_input_default(n));
        if (gather_f32_i32_forward(n) != 0) return -1;
    } else if (!strcmp(op, "Expand")) {
        materialize_tensor_f32(node_input_default(n));
        if (expand_plain_32_forward(n) != 0) return -1;
    } else if (!strcmp(op, "Slice")) {
        materialize_tensor_f32(node_input_default(n));
        if (slice_plain_32_forward(n) != 0) return -1;
    } else if (!strcmp(op, "Cast")) {
        materialize_tensor_f32(node_input_default(n));
        if (cast_forward(n) != 0) return -1;
    } else if (!strcmp(op, "ReLU")) { materialize_tensor_f32(in); relu_f32(in->data + e_off, o->data + e_off, e_cnt);
    } else if (!strcmp(op, "Sigmoid"))     { materialize_tensor_f32(in); sigmoid_f32(in->data + e_off, o->data + e_off, e_cnt);
    } else if (!strcmp(op, "GELU")) {
        int approximate_tanh = 0;
        materialize_tensor_f32(in);
        if (gelu_approximate_tanh(p, &approximate_tanh) != 0) return -1;
        if (approximate_tanh)
            gelu_tanh_f32(in->data + e_off, o->data + e_off, e_cnt);
        else
            gelu_f32(in->data + e_off, o->data + e_off, e_cnt);
    } else if (!strcmp(op, "SiLU")) { materialize_tensor_f32(in); silu_f32(in->data + e_off, o->data + e_off, e_cnt);
    } else if (!strcmp(op, "Tanh"))        { materialize_tensor_f32(in); tanh_f32(in->data + e_off, o->data + e_off, e_cnt);
    } else if (!strcmp(op, "LeakyReLU"))   { materialize_tensor_f32(in); leakyrelu_f32(in->data + e_off, o->data + e_off, e_cnt, p_flt(p, "alpha", 0.01f));
    } else if (!strcmp(op, "PReLU")) {
        T* weight = nin(n, "weight");
        if (!weight) weight = nin(n, "slope");
        materialize_tensor_f32(in); materialize_tensor_f32(weight);
        int channels = in && in->ndim > 0 ? in->shape[in->ndim - 1] : 0;
        if (!in || !weight || !o || in->dtype != T_F32 || weight->dtype != T_F32 ||
            o->dtype != T_F32 || in->numel != o->numel || channels <= 0 ||
            e_cnt % channels || (weight->numel != 1 && weight->numel != channels)) return -1;
        if (weight->numel == channels) {
            prelu_f32(in->data + e_off, weight->data, o->data + e_off,
                      1, 1, e_cnt / channels, channels);
        } else {
            float alpha = weight->data[0];
            for (int element = 0; element < e_cnt; element++) {
                float value = in->data[e_off + element];
                o->data[e_off + element] = value > 0.0f ? value : value * alpha;
            }
        }
    } else if (!strcmp(op, "HardSwish"))   { materialize_tensor_f32(in); hardswish_f32(in->data + e_off, o->data + e_off, e_cnt);
    } else if (!strcmp(op, "HardSigmoid")) { materialize_tensor_f32(in); hardsigmoid_f32(in->data + e_off, o->data + e_off, e_cnt);
    } else if (!strcmp(op, "Clip")) {
        materialize_tensor_f32(in);
        if (clip_plain_forward(n) != 0) return -1;
    } else if (!strcmp(op, "Transpose")) {
        materialize_tensor_f32(in);
        int perm[8];
        for (int i = 0; i < in->ndim; i++) perm[i] = in->ndim - 1 - i;
        if (p && cJSON_GetObjectItem(p, "perm"))
            p_iarr(p, "perm", perm, in->ndim, 0);
        if (transpose_tensor(in, o, perm) != 0) return -1;
    } else if (!strcmp(op, "Concat")) {
        materialize_node_inputs(n);
        int fused_sigmoid = idx >= 0 && idx < MAXN ? g_concat_sigmoid_fuse[idx] : -1;
        if (fused_sigmoid >= 0) {
            T* so = t_find(g_n[fused_sigmoid].out);
            if (!so) return -1;
            concat_tensor_sigmoid(n, so, p_int(p, "axis", 1));
            o = so;
            node_backend = "cpu+sigmoid";
        } else if (p_int(p, "sigmoid", 0)) {
            concat_tensor_sigmoid(n, o, p_int(p, "axis", 1));
            node_backend = "cpu+sigmoid";
        } else {
            if (concat_tensor(n, o, p_int(p, "axis", 1)) != 0) return -1;
        }
    } else if (!strcmp(op, "Split")) {
        if (cpu_split(n, in) != 0) return -1;
    } else if (!strcmp(op, "ResizeNearest2D") || !strcmp(op, "UpsampleNearest2D")) {
        materialize_tensor_f32(in);
        if (resize_nearest2d_image(in, o) != 0) return -1;
        node_backend = "cpu-resize";
    } else if (!strcmp(op, "Dropout")) {
        materialize_tensor_f32(in);
        if (!in || !o || in->dtype != T_F32 || o->dtype != T_F32 ||
            in->numel != o->numel) return -1;
#if VOLVOXAI_ENABLE_TRAINING
        if (g_native_training_mode) {
            if (dropout_forward_f32(n, idx, in, o, e_off, e_cnt) != 0) return -1;
        } else
#endif
        if (e_off == 0 && e_cnt == o->numel) {
            memcpy(o->data, in->data, (size_t)o->numel * sizeof(float));
        } else {
            copy_f32(in->data + e_off, o->data + e_off, e_cnt);
        }
    } else if (!strcmp(op, "Reshape") || !strcmp(op, "Flatten") || !strcmp(op, "Squeeze")
            || !strcmp(op, "Unsqueeze") || !strcmp(op, "Identity")) {
        materialize_tensor_f32(in);
        if (tensor_copy_plain_32(in, o) != 0) return -1;
    } else if (!strcmp(op, "LayerNorm")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        if (!in || !o || !w || in->ndim <= 0) return -1;
        int d_model = p_int(p, "d_model", in->shape[in->ndim - 1]);
        float epsilon = p_flt(p, "eps", 1e-6f);
        long total;
        if (normalization_dimensions(in, o, w, b, d_model, &total) != 0 ||
            !(epsilon > 0.0f) || !isfinite(epsilon)) return -1;
        materialize_tensor_f32(in);
        materialize_tensor_f32(w);
        if (b) materialize_tensor_f32(b);
        int r0, rc; seq_range(total, &r0, &rc);
        layernorm_f32(in->data + (long)r0 * d_model, w->data,
                      b ? b->data : NULL,
                      o->data + (long)r0 * d_model, rc, d_model, epsilon);
    } else if (!strcmp(op, "RMSNorm")) {
        T* w = nin(n, "weight");
        if (!in || !o || !w || in->ndim <= 0) return -1;
        int d_model = p_int(p, "d_model", in->shape[in->ndim - 1]);
        float epsilon = p_flt(p, "eps", 1e-6f);
        long total;
        if (normalization_dimensions(in, o, w, NULL, d_model, &total) != 0 ||
            !(epsilon > 0.0f) || !isfinite(epsilon)) return -1;
        materialize_tensor_f32(in);
        materialize_tensor_f32(w);
        int r0, rc; seq_range(total, &r0, &rc);
        rmsnorm_f32(in->data + (long)r0 * d_model, w->data,
                    o->data + (long)r0 * d_model, rc, d_model, epsilon);
    } else if (!strcmp(op, "Softmax") || !strcmp(op, "LogSoftmax")) {
        materialize_tensor_f32(in);
        if (!in || !o || in->dtype != T_F32 || o->dtype != T_F32 ||
            in->ndim <= 0 || in->numel != o->numel) return -1;
        int axis = p_int(p, "axis", in->ndim - 1);
        if (axis < 0) axis += in->ndim;
        int d = in->shape[in->ndim - 1];
        if (axis != in->ndim - 1 || d <= 0 || in->numel % d) return -1;
        int b = (int)(in->numel / d);
        if (!strcmp(op, "Softmax")) softmax_f32(in->data, o->data, b, d);
        else logsoftmax_f32(in->data, o->data, b, d);
    } else if (!strcmp(op, "Embedding")) {
        T* w = nin(n, "weight");
        if (!in || !w || !o || in->dtype != T_I32 || w->dtype != T_F32 ||
            o->dtype != T_F32 || w->ndim <= 0) return -1;
        materialize_tensor_f32(w);
        int d_model = w->shape[w->ndim - 1];
        int vocab = d_model > 0 && w->numel % d_model == 0 ? (int)(w->numel / d_model) : 0;
        if (d_model <= 0 || vocab <= 0 || o->numel != in->numel * (long)d_model) return -1;
        const int32_t* ids = (const int32_t*)in->data;
        for (long token = 0; token < in->numel; token++)
            if (ids[token] < 0 || ids[token] >= vocab) return -1;
        long total = in->numel; int r0, rc; seq_range(total, &r0, &rc);
        embedding_f32((uintptr_t)(ids + r0), (uintptr_t)w->data,
                      (uintptr_t)(o->data + (long)r0 * d_model), rc, d_model);
    } else if (!strcmp(op, "SDPA")) {
        materialize_tensor_f32(in);
        T* qkv = nin(n, "qkv");
        if (!qkv || (qkv->ndim != 2 && qkv->ndim != 3)) return -1;
        int num_heads = p_int(p, "heads", 8);
        int d_model = qkv->shape[qkv->ndim - 1] / 3;
        int batch = qkv->ndim == 3 ? qkv->shape[0] : 1;
        int max_seq = qkv->shape[qkv->ndim - 2];
        if (num_heads <= 0 || d_model <= 0 || d_model % num_heads) return -1;
        int head_dim = d_model / num_heads;
        float scale = p_flt(p, "scale", 1.0f / sqrtf((float)head_dim));
        int causal = p_int(p, "causal", 1) ? 1 : 0;
        T* mask = nin(n, "mask");
        int mask_mode = attention_mask_mode(mask, batch, max_seq, max_seq);
        if (mask_mode < 0) return -1;
#if VOLVOXAI_ENABLE_TRAINING
        NativeAttentionDropoutParams dropout;
        if (native_attention_dropout_params(n, idx, &dropout) != 0) return -1;
#endif
        if (batch <= 0 || max_seq <= 0 || qkv->numel != (long)batch * max_seq * 3 * d_model ||
            !o || o->ndim != qkv->ndim || (o->ndim == 3 && o->shape[0] != batch) ||
            o->shape[o->ndim - 2] != max_seq || o->shape[o->ndim - 1] != d_model ||
            o->numel != (long)batch * max_seq * d_model) return -1;
        size_t cache_elements = (size_t)batch;
        if ((size_t)max_seq > SIZE_MAX / cache_elements) return -1;
        cache_elements *= (size_t)max_seq;
        if ((size_t)d_model > SIZE_MAX / cache_elements) return -1;
        cache_elements *= (size_t)d_model;
        if (cache_elements > SIZE_MAX / sizeof(float)) return -1;
        float* key_cache = (float*)realloc(g_kcache[idx], cache_elements * sizeof(float));
        if (!key_cache) return -1;
        g_kcache[idx] = key_cache;
        float* value_cache = (float*)realloc(g_vcache[idx], cache_elements * sizeof(float));
        if (!value_cache) return -1;
        g_vcache[idx] = value_cache;
        if (g_active_row < 0) {
#if VOLVOXAI_ENABLE_TRAINING
            int training_dropout = g_native_training_mode && dropout.enabled;
            if (training_dropout && sdpa_attention_dropout_forward(
                    qkv, mask, o, batch, max_seq, d_model, num_heads, head_dim,
                    scale, causal, mask_mode, &dropout) != 0) return -1;
#endif
            for (int batch_index = 0; batch_index < batch; batch_index++) {
                const float* batch_qkv = qkv->data + (long)batch_index * max_seq * 3 * d_model;
                float* batch_out = o->data + (long)batch_index * max_seq * d_model;
#if VOLVOXAI_ENABLE_TRAINING
                if (!training_dropout) {
#endif
                    int local_mask_mode = 0;
                    const int32_t* batch_mask = attention_mask_batch_view(
                        mask, mask_mode, batch_index, max_seq, max_seq, &local_mask_mode);
                    sdpa_f32(batch_qkv, batch_out, max_seq, d_model, num_heads, head_dim,
                             scale, batch_mask, local_mask_mode, causal);
#if VOLVOXAI_ENABLE_TRAINING
                }
#endif
                for (int pos = 0; pos < max_seq; pos++) {
                    const float* src = batch_qkv + (long)pos * 3 * d_model;
                    long cache_offset = ((long)batch_index * max_seq + pos) * d_model;
                    memcpy(g_kcache[idx] + cache_offset, src + d_model,
                           (size_t)d_model * sizeof(float));
                    memcpy(g_vcache[idx] + cache_offset, src + 2 * d_model,
                           (size_t)d_model * sizeof(float));
                }
            }
        } else {
#if VOLVOXAI_ENABLE_TRAINING
            if (g_native_training_mode && dropout.enabled) return -1;
#endif
            if (!causal || mask) return -1;
            int pos = g_active_row;
            if (pos < 0 || pos >= max_seq) return -1;
            for (int batch_index = 0; batch_index < batch; batch_index++) {
                long qkv_offset = ((long)batch_index * max_seq + pos) * 3 * d_model;
                long cache_offset = ((long)batch_index * max_seq + pos) * d_model;
                long batch_cache_offset = (long)batch_index * max_seq * d_model;
                const float* src = qkv->data + qkv_offset;
                memcpy(g_kcache[idx] + cache_offset, src + d_model,
                       (size_t)d_model * sizeof(float));
                memcpy(g_vcache[idx] + cache_offset, src + 2 * d_model,
                       (size_t)d_model * sizeof(float));
                if (sdpa_cached_row(src, g_kcache[idx] + batch_cache_offset,
                                g_vcache[idx] + batch_cache_offset, pos + 1,
                                o->data + cache_offset, d_model, num_heads, head_dim,
                                scale) != 0) return -1;
            }
        }
    } else if (!strcmp(op, "CrossSDPA")) {
        materialize_tensor_f32(in);
        T* q = nin(n, "q"); T* k = nin(n, "k"); T* v = nin(n, "v");
        if (!q || !k || !v ||
            (q->ndim != 2 && q->ndim != 3) || k->ndim != q->ndim || v->ndim != q->ndim) return -1;
        int num_heads = p_int(p, "heads", 8);
        int d_model = q->shape[q->ndim - 1];
        int batch = q->ndim == 3 ? q->shape[0] : 1;
        int seq_q = q->shape[q->ndim - 2];
        int seq_kv = k->shape[k->ndim - 2];
        if (num_heads <= 0 || d_model <= 0 || d_model % num_heads) return -1;
        int head_dim = d_model / num_heads;
        float scale = p_flt(p, "scale", 1.0f / sqrtf((float)head_dim));
        int causal = p_int(p, "causal", 0) ? 1 : 0;
        T* mask = nin(n, "mask");
        int mask_mode = attention_mask_mode(mask, batch, seq_q, seq_kv);
        if (mask_mode < 0) return -1;
#if VOLVOXAI_ENABLE_TRAINING
        NativeAttentionDropoutParams dropout;
        if (native_attention_dropout_params(n, idx, &dropout) != 0) return -1;
#endif
        if (batch <= 0 || (q->ndim == 3 && (k->shape[0] != batch || v->shape[0] != batch)) ||
            k->shape[k->ndim - 1] != d_model || v->shape[v->ndim - 1] != d_model ||
            v->shape[v->ndim - 2] != seq_kv || !o || o->ndim != q->ndim ||
            (o->ndim == 3 && o->shape[0] != batch) || o->shape[o->ndim - 2] != seq_q ||
            o->shape[o->ndim - 1] != d_model || o->numel != (long)batch * seq_q * d_model) return -1;
#if VOLVOXAI_ENABLE_TRAINING
        if (g_native_training_mode && dropout.enabled) {
            if (cross_sdpa_attention_dropout_forward(q, k, v, mask, o, batch,
                    seq_q, seq_kv, d_model, num_heads, head_dim, scale, causal,
                    mask_mode, &dropout) != 0) return -1;
        } else
#endif
        {
            int row_ran = 0;
            if (g_active_row >= 0 && q->ndim == 3 && batch == 1) {
                int row = g_active_row;
                if (row >= seq_q) return -1;
                /* A key-only mask remains valid when the query tensor is
                 * sliced to one row. Query-dependent layouts keep the full
                 * path below so their original query indexing is preserved. */
                int row_mask_safe = mask_mode == ATTN_MASK_NONE ||
                    mask_mode == ATTN_MASK_KEY || mask_mode == ATTN_MASK_BATCH_KEY;
                int causal_self = causal && seq_q == seq_kv;
                if (row_mask_safe && (!causal || causal_self)) {
                    int local_mask_mode = 0;
                    const int32_t* batch_mask = attention_mask_batch_view(
                        mask, mask_mode, 0, seq_q, seq_kv, &local_mask_mode);
                    int row_seq_kv = causal_self ? row + 1 : seq_kv;
                    long q_offset = (long)row * d_model;
                    /* The causal row sees only its K/V prefix, so the sliced
                     * one-query kernel itself must be noncausal (query 0
                     * would otherwise be limited to key 0). */
                    cross_sdpa_f32(q->data + q_offset, k->data, v->data,
                                   o->data + q_offset, 1, row_seq_kv, d_model,
                                   num_heads, head_dim, scale, batch_mask,
                                   local_mask_mode, 0);
                    row_ran = 1;
                    node_backend = "cpu-cross-sdpa-row";
                }
            }
            if (!row_ran) {
                for (int batch_index = 0; batch_index < batch; batch_index++) {
                    long q_offset = (long)batch_index * seq_q * d_model;
                    long kv_offset = (long)batch_index * seq_kv * d_model;
                    int local_mask_mode = 0;
                    const int32_t* batch_mask = attention_mask_batch_view(
                        mask, mask_mode, batch_index, seq_q, seq_kv, &local_mask_mode);
                    cross_sdpa_f32(q->data + q_offset, k->data + kv_offset,
                                   v->data + kv_offset, o->data + q_offset,
                                   seq_q, seq_kv, d_model, num_heads, head_dim,
                                   scale, batch_mask, local_mask_mode, causal);
                }
            }
        }
    } else {
        fprintf(stderr, "[engine] unsupported op '%s' — node skipped\n", op);
        return -1;
    }
done:
    if (idx >= 0 && idx < MAXN)
        vx_engine_state_current()->runtime_route_backend[idx] = node_backend;
    /* The hybrid handoff is a one-way ownership transition. Its device slots
     * were cleared before the first CPU row and are discarded before GPU
     * execution can resume, so per-node host-dirty bookkeeping is redundant. */
    if (!ran_gpu_graph && !vx_incremental_hybrid_row_active_locked())
        vk_mark_node_outputs_host(n, o);
    if (log_node) {
        double el = volvoxai_engine_now_ms() - node_t0;
        fprintf(stderr, "[debug] node=%03d op=%s backend=%s out=%s numel=%ld %.3f ms\n",
                idx, op, node_backend, n->out, o ? o->numel : 0L, el);
        prof_add(op, el);
    }
    return 0;
}
