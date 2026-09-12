#ifndef VOLVOX_ENGINE_PRIVATE_INCLUDE
#error "graph_opt_fusion.inc is a private include from engine_runtime_model.inc; do not compile directly."
#endif

// ---- Graph Optimization: operator fusion -----------------------------------
// Pattern categories mirror docs/operator_fusion_patterns.md. The optimizer is intentionally
// table-driven so CPU/GPU backends can add specialized lowerings without touching build_graph().
typedef enum {
    GRAPH_OPT_CAT_CONV = 0,
    GRAPH_OPT_CAT_MATMUL,
    GRAPH_OPT_CAT_ELEMENTWISE,
    GRAPH_OPT_CAT_NORMALIZATION,
    GRAPH_OPT_CAT_MISC,
    GRAPH_OPT_CAT_INTERNAL
} GraphOptCategory;

typedef enum {
    GRAPH_OPT_BACKEND_CPU_SCALAR = 1u << 0,
    GRAPH_OPT_BACKEND_CPU_AVX2   = 1u << 1,
    GRAPH_OPT_BACKEND_CPU_NEON   = 1u << 2,
    GRAPH_OPT_BACKEND_WGSL       = 1u << 3,
    GRAPH_OPT_BACKEND_VULKAN     = 1u << 4,
    GRAPH_OPT_BACKEND_OPENGL     = 1u << 5,
    GRAPH_OPT_BACKEND_METAL      = 1u << 6,
    GRAPH_OPT_BACKEND_CUDA       = 1u << 7
} GraphOptBackend;

typedef enum {
    GRAPH_OPT_STAT_RELU6 = 0,
    GRAPH_OPT_STAT_ALIAS,
    GRAPH_OPT_STAT_CONCAT_SIGMOID,
    GRAPH_OPT_STAT_CONV_ADD,
    GRAPH_OPT_STAT_DEPTHWISE_POINTWISE,
    GRAPH_OPT_STAT_CHAINED_ADD
} GraphOptStatSlot;

typedef int (*GraphOptPassFn)(void);

typedef struct {
    GraphFusionId id;
    GraphOptCategory category;
    unsigned backend_mask;
    const char* name;
    const char* enable_env;   // if set, pass is opt-in
    GraphOptStatSlot stat;
    GraphOptPassFn run;
} GraphOptPass;

#define GRAPH_OPT_CPU_ALL (GRAPH_OPT_BACKEND_CPU_SCALAR | GRAPH_OPT_BACKEND_CPU_AVX2 | GRAPH_OPT_BACKEND_CPU_NEON)
#define GRAPH_OPT_GPU_ALL (GRAPH_OPT_BACKEND_WGSL | GRAPH_OPT_BACKEND_VULKAN | GRAPH_OPT_BACKEND_OPENGL | GRAPH_OPT_BACKEND_METAL | GRAPH_OPT_BACKEND_CUDA)

static int env_truthy(const char* name) {
    const char* v = vx_engine_env(name);
    return v && v[0] && strcmp(v, "0");
}

static unsigned graph_opt_current_backend(void) {
    if (g_use_vulkan) return GRAPH_OPT_BACKEND_VULKAN;
    if (g_use_opengl) return GRAPH_OPT_BACKEND_OPENGL;
    if (g_use_metal) return GRAPH_OPT_BACKEND_METAL;
#if VOLVOXAI_ENABLE_CUDA
    if (g_use_cuda) return GRAPH_OPT_BACKEND_CUDA;
#endif
#if defined(__AVX2__)
    return GRAPH_OPT_BACKEND_CPU_AVX2;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    return GRAPH_OPT_BACKEND_CPU_NEON;
#else
    return GRAPH_OPT_BACKEND_CPU_SCALAR;
#endif
}

static int graph_opt_pass_enabled(const GraphOptPass* pass) {
    unsigned backend = graph_opt_current_backend();
    if (pass->backend_mask && !(pass->backend_mask & backend)) return 0;
    if (pass->enable_env && !env_truthy(pass->enable_env)) return 0;
    return 1;
}

static void graph_opt_add_stat(GraphOptStats* st, GraphOptStatSlot slot, int value) {
    switch (slot) {
        case GRAPH_OPT_STAT_RELU6: st->relu6 += value; break;
        case GRAPH_OPT_STAT_ALIAS: st->alias += value; break;
        case GRAPH_OPT_STAT_CONCAT_SIGMOID: st->concat_sigmoid += value; break;
        case GRAPH_OPT_STAT_CONV_ADD: st->conv_add += value; break;
        case GRAPH_OPT_STAT_DEPTHWISE_POINTWISE: st->depthwise_pointwise += value; break;
        case GRAPH_OPT_STAT_CHAINED_ADD: st->chained_add += value; break;
    }
}

static void graph_opt_reset_state(void) {
    memset(&g_graph_opt_stats, 0, sizeof(g_graph_opt_stats));
    for (size_t i = 0; i < g_node_capacity; i++) {
        g_node_fusion[i].id = 0;
        g_node_fusion[i].backend_mask = 0;
        g_node_fusion[i].peer_idx = -1;
        g_node_fusion[i].source_tensor[0] = 0;
        g_node_fusion[i].aux_tensor[0] = 0;
    }
    for (size_t i = 0; i < g_node_capacity; i++)
        g_concat_sigmoid_fuse[i] = -1;
}

static const GraphNodeFusion* graph_opt_node_fusion(int node_idx, GraphFusionId id) {
    if (node_idx < 0 || (size_t)node_idx >= g_node_capacity) return NULL;
    const GraphNodeFusion* f = &g_node_fusion[node_idx];
    if (f->id != id) return NULL;
    unsigned backend = graph_opt_current_backend();
    if (f->backend_mask && !(f->backend_mask & backend)) return NULL;
    return f;
}

static int graph_opt_declared_output(const char* name) {
    cJSON* outputs;
    if (!name || !name[0] || !g_graph_root) return 0;
    outputs = cJSON_GetObjectItem(g_graph_root, "outputs");
    if (!outputs || (!cJSON_IsArray(outputs) && !cJSON_IsObject(outputs)))
        return 0;
    for (cJSON* output = outputs->child; output; output = output->next) {
        if (cJSON_IsString(output) && output->valuestring &&
            !strcmp(output->valuestring, name)) return 1;
    }
    return 0;
}

static int fuse_relu6_pass(void) {
    int applied = 0;
    for (int i = 0; i < g_nn; i++) {
        Node* clip = &g_n[i];
        if (clip->operator_kind != VX_OP_CLIP) continue;
        if (vx_node_param_f32(clip, VX_NODE_PARAM_MIN, -1e30f) != 0.0f ||
            vx_node_param_f32(clip, VX_NODE_PARAM_MAX, 1e30f) != 6.0f)
            continue;
        const char* tin = node_input_name(clip);
        if (!tin || tensor_use_count(tin) != 1) continue;
        Node* prod = NULL;
        for (int j = 0; j < g_nn; j++) if (!g_n[j].skip && !strcmp(g_n[j].out, tin)) { prod = &g_n[j]; break; }
        if (!prod) continue;
        if ((prod->operator_kind != VX_OP_Q_CONV_2D) && (prod->operator_kind != VX_OP_CONV_2D)) continue;
        prod->fuse_relu6 = 1;
        strncpy(prod->out, clip->out, 127); prod->out[127] = 0;
        clip->skip = 1;
        applied++;
    }
    return applied;
}

static int fuse_concat_sigmoid_pass(void) {
    int applied = 0;
    for (int i = 0; i < g_nn; i++) {
        Node* sig = &g_n[i];
        if (sig->skip || (sig->operator_kind != VX_OP_SIGMOID)) continue;
        const char* tin = node_input_name(sig);
        if (!tin || tensor_use_count(tin) != 1) continue;
        int prod_idx = -1;
        for (int j = 0; j < g_nn; j++) {
            if (!g_n[j].skip && !strcmp(g_n[j].out, tin)) { prod_idx = j; break; }
        }
        if (prod_idx < 0 || (g_n[prod_idx].operator_kind != VX_OP_CONCAT)) continue;
        T* a = t_find(g_n[prod_idx].out);
        T* b = t_find(sig->out);
        if (!a || !b || a->dtype != T_F32 || b->dtype != T_F32 || a->numel != b->numel) continue;
        g_concat_sigmoid_fuse[prod_idx] = i;
        sig->skip = 1;
        applied++;
    }
    return applied;
}

static int fuse_conv_add_pass(void) {
    if (g_use_vulkan || g_use_opengl) return 0;
    int applied = 0;
    for (int i = 0; i < g_nn; i++) {
        Node* add = &g_n[i];
        if (add->skip || (add->operator_kind != VX_OP_ADD)) continue;
        if (g_node_fusion[i].id || add->nin != 2) continue;
        if (vx_node_param_i32(add, VX_NODE_PARAM_RELU, 0)) continue;
        T* add_out = t_find(add->out);
        if (!add_out || add_out->dtype != T_F32) continue;

        for (int ai = 0; ai < 2; ai++) {
            const char* conv_out_name = add->ins[ai].name;
            int conv_idx = producer_index_for(conv_out_name);
            if (conv_idx < 0 || conv_idx >= i) continue;
            Node* conv = &g_n[conv_idx];
            if (conv->skip || g_node_fusion[conv_idx].id ||
                (conv->operator_kind != VX_OP_CONV_2D)) continue;
            if (tensor_use_count(conv->out) != 1) continue;
            if (graph_opt_declared_output(conv->out)) continue;
            int conv_occurrences = 0;
            for (int input_index = 0; input_index < 2; input_index++)
                if (!strcmp(add->ins[input_index].name, conv->out))
                    conv_occurrences++;
            if (conv_occurrences != 1) continue;

            T* conv_out = t_find(conv->out);
            T* in = nin(conv, VX_PORT_INPUT);
            T* wt = nin(conv, VX_PORT_WEIGHT);
            if (!conv_out || !in || !wt || conv_out->dtype != T_F32 ||
                in->dtype != T_F32 || (wt->dtype != T_F32 && wt->dtype != T_F16) ||
                in->ndim != 4 || conv_out->ndim != 4 || wt->ndim != 4 ||
                !tensor_same_shape(conv_out, add_out)) continue;

            if (conv->parsed_params.parsed != 1) continue;
            int sy = conv->parsed_params.stride[0];
            int sx = conv->parsed_params.stride[1];
            int dy = conv->parsed_params.dilation[0];
            int dx = conv->parsed_params.dilation[1];
            int pads[4] = {
                conv->parsed_params.pads[0], conv->parsed_params.pads[1],
                conv->parsed_params.pads[2], conv->parsed_params.pads[3],
            };
            int groups = conv->parsed_params.groups;
            if (groups != 1 || sy != 1 || sx != 1 || dy != 1 || dx != 1 ||
                pads[0] != 0 || pads[1] != 0 || pads[2] != 0 || pads[3] != 0) continue;

            int layout = conv->parsed_params.weight_layout;
            int kh = 0, kw = 0;
            if (layout == VX_CONV_WEIGHT_LAYOUT_OHWI) {
                kh = wt->shape[1]; kw = wt->shape[2];
            } else if (layout == VX_CONV_WEIGHT_LAYOUT_HWIO) {
                kh = wt->shape[0]; kw = wt->shape[1];
            } else {
                continue;
            }
            if (kh != 1 || kw != 1) continue;

            int ri = 1 - ai;
            const char* res_name = add->ins[ri].name;
            T* res = t_find(res_name);
            if (!res_name || !res) continue;
            if (!strcmp(res_name, conv->out) || res->dtype != T_F32 ||
                !tensor_same_shape(res, add_out)) continue;
            int res_prod = producer_index_any(res_name);
            if (res_prod >= 0 && res_prod > conv_idx) continue;
            if (!strcmp(res_name, add->out) || !strcmp(conv->out, add->out)) continue;

            g_node_fusion[conv_idx].id = GRAPH_FUSION_CONV_ADD;
            g_node_fusion[conv_idx].backend_mask = graph_opt_current_backend();
            g_node_fusion[conv_idx].peer_idx = i;
            strncpy(g_node_fusion[conv_idx].source_tensor, conv->out, 127);
            g_node_fusion[conv_idx].source_tensor[127] = 0;
            strncpy(g_node_fusion[conv_idx].aux_tensor, res_name, 127);
            g_node_fusion[conv_idx].aux_tensor[127] = 0;
            strncpy(conv->out, add->out, 127);
            conv->out[127] = 0;
            add->skip = 1;
            applied++;
            break;
        }
    }
    return applied;
}

static int fuse_depthwise_pointwise_pass(void) {
    if (g_use_vulkan || g_use_opengl) return 0;
    int applied = 0;
    for (int pw_idx = 0; pw_idx < g_nn; pw_idx++) {
        Node* pw = &g_n[pw_idx];
        if (pw->skip || (pw->operator_kind != VX_OP_CONV_2D)) continue;
        const char* pw_in_name = node_input_name(pw);
        int dw_idx = producer_index_for(pw_in_name);
        if (dw_idx < 0 || dw_idx >= pw_idx) continue;
        Node* dw = &g_n[dw_idx];
        if (dw->skip || (dw->operator_kind != VX_OP_CONV_2D)) continue;
        if (tensor_use_count(dw->out) != 1) continue;

        T* in = nin(dw, VX_PORT_INPUT);
        T* dw_out = t_find(dw->out);
        T* pw_out = t_find(pw->out);
        T* dw_w = nin(dw, VX_PORT_WEIGHT);
        T* pw_w = nin(pw, VX_PORT_WEIGHT);
        if (!in || !dw_out || !pw_out || !dw_w || !pw_w ||
            in->dtype != T_F32 || dw_out->dtype != T_F32 || pw_out->dtype != T_F32 ||
            (dw_w->dtype != T_F32 && dw_w->dtype != T_F16) ||
            (pw_w->dtype != T_F32 && pw_w->dtype != T_F16) ||
            in->ndim != 4 || dw_out->ndim != 4 || pw_out->ndim != 4 ||
            dw_w->ndim != 4 || pw_w->ndim != 4 ||
            in->shape[0] != pw_out->shape[0] || dw_out->shape[0] != pw_out->shape[0] ||
            dw_out->shape[1] != pw_out->shape[1] || dw_out->shape[2] != pw_out->shape[2]) continue;

        if (dw->parsed_params.parsed != 1 ||
            pw->parsed_params.parsed != 1) continue;
        int dw_sy = dw->parsed_params.stride[0];
        int dw_sx = dw->parsed_params.stride[1];
        int dw_dy = dw->parsed_params.dilation[0];
        int dw_dx = dw->parsed_params.dilation[1];
        int dw_pads[4] = {
            dw->parsed_params.pads[0], dw->parsed_params.pads[1],
            dw->parsed_params.pads[2], dw->parsed_params.pads[3],
        };
        int dw_groups = dw->parsed_params.groups;
        if (dw_groups != in->shape[3] || dw_out->shape[3] != in->shape[3] ||
            dw_dy != 1 || dw_dx != 1) continue;

        int dw_layout = dw->parsed_params.weight_layout;
        int dw_kh = 0, dw_kw = 0, dw_c = 0, dw_mult = 1;
        if (dw_layout == VX_CONV_WEIGHT_LAYOUT_1HWO ||
            dw_layout == VX_CONV_WEIGHT_LAYOUT_1HWM) {
            if (dw_w->shape[0] != 1) continue;
            dw_kh = dw_w->shape[1]; dw_kw = dw_w->shape[2]; dw_c = dw_w->shape[3]; dw_mult = 1;
        } else if (dw_layout == VX_CONV_WEIGHT_LAYOUT_HWCM ||
                   dw_layout == VX_CONV_WEIGHT_LAYOUT_HWIO) {
            dw_kh = dw_w->shape[0]; dw_kw = dw_w->shape[1]; dw_c = dw_w->shape[2]; dw_mult = dw_w->shape[3];
        } else {
            continue;
        }
        if (dw_c != in->shape[3] || dw_mult != 1 || dw_kh <= 0 || dw_kw <= 0) continue;
        if (dw_c < 96) continue;
        if (dw_kh == 3 && dw_kw == 3 && dw_sy == 1 && dw_sx == 1) continue;

        int pw_sy = pw->parsed_params.stride[0];
        int pw_sx = pw->parsed_params.stride[1];
        int pw_dy = pw->parsed_params.dilation[0];
        int pw_dx = pw->parsed_params.dilation[1];
        int pw_pads[4] = {
            pw->parsed_params.pads[0], pw->parsed_params.pads[1],
            pw->parsed_params.pads[2], pw->parsed_params.pads[3],
        };
        int pw_groups = pw->parsed_params.groups;
        if (pw_groups != 1 || pw_sy != 1 || pw_sx != 1 || pw_dy != 1 || pw_dx != 1 ||
            pw_pads[0] != 0 || pw_pads[1] != 0 || pw_pads[2] != 0 || pw_pads[3] != 0) continue;

        int pw_layout = pw->parsed_params.weight_layout;
        int pw_kh = 0, pw_kw = 0, pw_ic = 0;
        if (pw_layout == VX_CONV_WEIGHT_LAYOUT_OHWI) {
            pw_kh = pw_w->shape[1]; pw_kw = pw_w->shape[2]; pw_ic = pw_w->shape[3];
        } else if (pw_layout == VX_CONV_WEIGHT_LAYOUT_HWIO) {
            pw_kh = pw_w->shape[0]; pw_kw = pw_w->shape[1]; pw_ic = pw_w->shape[2];
        } else {
            continue;
        }
        if (pw_kh != 1 || pw_kw != 1 || pw_ic != dw_out->shape[3]) continue;

        const GraphNodeFusion* pw_add_fusion = graph_opt_node_fusion(pw_idx, GRAPH_FUSION_CONV_ADD);
        if (pw_add_fusion) {
            T* res = t_find(pw_add_fusion->aux_tensor);
            int res_prod = producer_index_any(pw_add_fusion->aux_tensor);
            if (!res || res->dtype != T_F32 || !tensor_same_shape(res, pw_out) ||
                (res_prod >= 0 && res_prod > dw_idx)) continue;
        }

        g_node_fusion[dw_idx].id = GRAPH_FUSION_DEPTHWISE_POINTWISE;
        g_node_fusion[dw_idx].backend_mask = GRAPH_OPT_CPU_ALL;
        g_node_fusion[dw_idx].peer_idx = pw_idx;
        pw->skip = 1;
        applied++;
    }
    return applied;
}

static int fuse_chained_add_pass(void) {
    int applied = 0;
    for (int i = 0; i < g_nn; i++) {
        Node* add0 = &g_n[i];
        if (add0->skip || (add0->operator_kind != VX_OP_ADD)) continue;
        if (g_node_fusion[i].id || add0->nin != 2) continue;
        if (vx_node_param_i32(add0, VX_NODE_PARAM_RELU, 0)) continue;
        T* add0_out = t_find(add0->out);
        if (!add0_out || add0_out->dtype != T_F32) continue;
        if (tensor_use_count(add0->out) != 1) continue;
        if (graph_opt_declared_output(add0->out)) continue;
        T* runtime_a = nin(add0, VX_PORT_A);
        if (!runtime_a) runtime_a = nin(add0, VX_PORT_INPUT);
        T* runtime_b = nin(add0, VX_PORT_B);
        if (!runtime_a || !runtime_b) continue;

        int add1_idx = -1;
        for (int j = i + 1; j < g_nn; j++) {
            if (g_n[j].skip) continue;
            for (int k = 0; k < g_n[j].nin; k++) {
                if (!strcmp(g_n[j].ins[k].name, add0->out)) {
                    add1_idx = j;
                    break;
                }
            }
            if (add1_idx >= 0) break;
        }
        if (add1_idx < 0) continue;
        Node* add1 = &g_n[add1_idx];
        if ((add1->operator_kind != VX_OP_ADD) || add1->nin != 2 ||
            g_node_fusion[add1_idx].id) continue;
        int peer_relu = vx_node_param_i32(
            add1, VX_NODE_PARAM_RELU, 0);
        if (peer_relu < 0 || peer_relu > 2) continue;
        T* add1_out = t_find(add1->out);
        if (!add1_out || add1_out->dtype != T_F32 || !tensor_same_shape(add0_out, add1_out)) continue;

        const char* third_name = NULL;
        int prior_occurrences = 0;
        for (int k = 0; k < 2; k++) {
            if (!strcmp(add1->ins[k].name, add0->out)) prior_occurrences++;
            else third_name = add1->ins[k].name;
        }
        if (prior_occurrences != 1 || !third_name) continue;
        T* third = t_find(third_name);
        if (!third || third->dtype != T_F32 || !tensor_same_shape(third, add1_out)) continue;
        int third_prod = producer_index_any(third_name);
        if (third_prod >= i) continue;

        if (runtime_a->dtype != T_F32 || runtime_b->dtype != T_F32 ||
            !tensor_same_shape(runtime_a, add1_out) ||
            !tensor_same_shape(runtime_b, add1_out)) continue;

        g_node_fusion[i].id = GRAPH_FUSION_CHAINED_ELEMENTWISE;
        g_node_fusion[i].backend_mask = graph_opt_current_backend();
        g_node_fusion[i].peer_idx = add1_idx;
        strncpy(g_node_fusion[i].source_tensor, add0->out, 127);
        g_node_fusion[i].source_tensor[127] = 0;
        strncpy(g_node_fusion[i].aux_tensor, third_name, 127);
        g_node_fusion[i].aux_tensor[127] = 0;
        strncpy(add0->out, add1->out, 127);
        add0->out[127] = 0;
        add1->skip = 1;
        applied++;
    }
    return applied;
}

/* A physical byte edge carries its scale/zero-point as part of the tensor
 * meaning.  Unlike an F32 view, it cannot be aliased across a descriptor
 * change: doing so would skip the required RequantizeLinear boundary and make
 * later consumers reinterpret the same bytes in a different real domain. */
static int alias_quantization_compatible(const T* in, const T* out) {
    int in_byte = in && (in->dtype == T_I8 || in->dtype == T_U8);
    int out_byte = out && (out->dtype == T_I8 || out->dtype == T_U8);
    if (!in_byte && !out_byte) return 1;
    return in_byte && out_byte && in->quantization.valid && out->quantization.valid &&
           in->quantization.scale == out->quantization.scale &&
           in->quantization.zero_point == out->quantization.zero_point;
}

static int alias_passthrough_pass(void) {
    /* CUDA consumes the resulting host pointer through its normal slot lookup;
     * this pass does not create or share backend-owned device allocations. */
    if (g_use_vulkan || g_use_opengl) return 0;
    int applied = 0;
    for (int i = 0; i < g_nn; i++) {
        Node* n = &g_n[i];
        int passthrough = (n->operator_kind == VX_OP_RESHAPE) || (n->operator_kind == VX_OP_FLATTEN) || (n->operator_kind == VX_OP_SQUEEZE) ||
                          (n->operator_kind == VX_OP_UNSQUEEZE) || (n->operator_kind == VX_OP_DROPOUT) || (n->operator_kind == VX_OP_IDENTITY);
        int dequant_after_concat = 0;
        if (!passthrough && (n->operator_kind == VX_OP_DEQUANTIZE_LINEAR)) {
            int prod_idx = producer_index_for(node_input_name(n));
            dequant_after_concat = prod_idx >= 0 && (g_n[prod_idx].operator_kind == VX_OP_CONCAT);
        }
        if (!passthrough && !dequant_after_concat) {
            continue;
        }
        T* in = nin(n, VX_PORT_INPUT);
        T* out = t_find(n->out);
        if (!in || !out || in->dtype != out->dtype || in->numel != out->numel ||
            !alias_quantization_compatible(in, out)) continue;
        if (out->owns && out->data) free(out->data);
        out->data = in->data;
        out->owns = 0;
        n->skip = 1;
        applied++;
    }
    return applied;
}

static const GraphOptPass g_operator_fusion_passes[] = {
    {
        GRAPH_FUSION_ADD_CLAMP,
        GRAPH_OPT_CAT_ELEMENTWISE,
        GRAPH_OPT_CPU_ALL | GRAPH_OPT_GPU_ALL,
        "clip_relu6_into_conv",
        NULL,
        GRAPH_OPT_STAT_RELU6,
        fuse_relu6_pass
    },
    {
        GRAPH_FUSION_ALIAS_PASSTHROUGH_INTERNAL,
        GRAPH_OPT_CAT_INTERNAL,
        GRAPH_OPT_CPU_ALL | GRAPH_OPT_BACKEND_CUDA,
        "alias_passthrough",
        NULL,
        GRAPH_OPT_STAT_ALIAS,
        alias_passthrough_pass
    },
    {
        GRAPH_FUSION_CONCAT_SIGMOID_INTERNAL,
        GRAPH_OPT_CAT_ELEMENTWISE,
        GRAPH_OPT_CPU_ALL,
        "concat_sigmoid",
        NULL,
        GRAPH_OPT_STAT_CONCAT_SIGMOID,
        fuse_concat_sigmoid_pass
    },
    {
        GRAPH_FUSION_CHAINED_ELEMENTWISE,
        GRAPH_OPT_CAT_ELEMENTWISE,
        GRAPH_OPT_BACKEND_OPENGL,
        "chained_add",
        "VOLVOX_ENABLE_CHAINED_ADD_FUSE",
        GRAPH_OPT_STAT_CHAINED_ADD,
        fuse_chained_add_pass
    },
    {
        GRAPH_FUSION_CHAINED_ELEMENTWISE,
        GRAPH_OPT_CAT_ELEMENTWISE,
        GRAPH_OPT_BACKEND_CUDA,
        "chained_add_cuda",
        NULL,
        GRAPH_OPT_STAT_CHAINED_ADD,
        fuse_chained_add_pass
    },
    {
        GRAPH_FUSION_CONV_ADD,
        GRAPH_OPT_CAT_CONV,
        GRAPH_OPT_BACKEND_CPU_AVX2 | GRAPH_OPT_BACKEND_CUDA,
        "conv_add",
        NULL,
        GRAPH_OPT_STAT_CONV_ADD,
        fuse_conv_add_pass
    },
    {
        GRAPH_FUSION_DEPTHWISE_POINTWISE,
        GRAPH_OPT_CAT_CONV,
        GRAPH_OPT_CPU_ALL,
        "depthwise_pointwise",
        "VOLVOX_ENABLE_DW_PW_FUSE",
        GRAPH_OPT_STAT_DEPTHWISE_POINTWISE,
        fuse_depthwise_pointwise_pass
    },
};

static GraphOptStats graph_optimize_operator_fusion(void) {
    GraphOptStats st;
    memset(&st, 0, sizeof(st));
    if (env_truthy("VOLVOX_DISABLE_OPERATOR_FUSION")) {
        g_graph_opt_stats = st;
        return st;
    }

    for (size_t i = 0; i < sizeof(g_operator_fusion_passes) / sizeof(g_operator_fusion_passes[0]); i++) {
        const GraphOptPass* pass = &g_operator_fusion_passes[i];
        if (!graph_opt_pass_enabled(pass)) continue;
        graph_opt_add_stat(&st, pass->stat, pass->run());
    }
    for (int i = 0; i < g_nn; i++) st.skipped += g_n[i].skip;
    g_graph_opt_stats = st;
    return st;
}
