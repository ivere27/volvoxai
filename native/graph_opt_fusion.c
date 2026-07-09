#ifndef VOLVOX_ENGINE_PRIVATE_INCLUDE
#error "graph_opt_fusion.c is a private include from engine.c; do not compile directly."
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
    GRAPH_OPT_BACKEND_METAL      = 1u << 6
} GraphOptBackend;

typedef enum {
    GRAPH_FUSION_CONV_BIAS = 1,
    GRAPH_FUSION_CONV_BIAS_ACT,
    GRAPH_FUSION_CONV_BIAS_HARDSWISH,
    GRAPH_FUSION_CONV_ADD,
    GRAPH_FUSION_CONV_ADD_ACT,
    GRAPH_FUSION_DEPTHWISE_ACT,
    GRAPH_FUSION_DEPTHWISE_POINTWISE,
    GRAPH_FUSION_DECONV_ACT,
    GRAPH_FUSION_CONV1D_ACT,
    GRAPH_FUSION_CONV3D_ACT,
    GRAPH_FUSION_MATMUL_BIAS,
    GRAPH_FUSION_MATMUL_BIAS_ACT,
    GRAPH_FUSION_MATMUL_ADD,
    GRAPH_FUSION_MATMUL_SOFTMAX,
    GRAPH_FUSION_ATTENTION_SCALE_MASK_SOFTMAX,
    GRAPH_FUSION_SWIGLU,
    GRAPH_FUSION_ADD_ACT,
    GRAPH_FUSION_MUL_ADD,
    GRAPH_FUSION_MUL_SIGMOID,
    GRAPH_FUSION_CHAINED_ELEMENTWISE,
    GRAPH_FUSION_ADD_CLAMP,
    GRAPH_FUSION_SOFTMAX_DECOMPOSED,
    GRAPH_FUSION_LAYERNORM_MATMUL,
    GRAPH_FUSION_RMSNORM_MUL,
    GRAPH_FUSION_INSTANCENORM_ACT,
    GRAPH_FUSION_GLOBAL_AVGPOOL_FLATTEN,
    GRAPH_FUSION_MAXPOOL_ACT,
    GRAPH_FUSION_ARGMAX_GATHER,
    GRAPH_FUSION_SPLIT_MATMUL,
    GRAPH_FUSION_QDQ_REQUANT,
    GRAPH_FUSION_CONCAT_SIGMOID_INTERNAL,
    GRAPH_FUSION_ALIAS_PASSTHROUGH_INTERNAL
} GraphFusionId;

typedef enum {
    GRAPH_OPT_STAT_RELU6 = 0,
    GRAPH_OPT_STAT_ALIAS,
    GRAPH_OPT_STAT_CONCAT_SIGMOID,
    GRAPH_OPT_STAT_CONV_ADD,
    GRAPH_OPT_STAT_DEPTHWISE_POINTWISE,
    GRAPH_OPT_STAT_CHAINED_ADD
} GraphOptStatSlot;

typedef struct {
    int relu6;
    int alias;
    int concat_sigmoid;
    int conv_add;
    int depthwise_pointwise;
    int chained_add;
    int skipped;
} GraphOptStats;

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

typedef struct {
    GraphFusionId id;
    unsigned backend_mask;
    int peer_idx;
    char aux_tensor[128];
} GraphNodeFusion;

#define GRAPH_OPT_CPU_ALL (GRAPH_OPT_BACKEND_CPU_SCALAR | GRAPH_OPT_BACKEND_CPU_AVX2 | GRAPH_OPT_BACKEND_CPU_NEON)
#define GRAPH_OPT_GPU_ALL (GRAPH_OPT_BACKEND_WGSL | GRAPH_OPT_BACKEND_VULKAN | GRAPH_OPT_BACKEND_OPENGL | GRAPH_OPT_BACKEND_METAL)

static GraphOptStats g_graph_opt_stats;
static GraphNodeFusion g_node_fusion[MAXN];

static int env_truthy(const char* name) {
    const char* v = getenv(name);
    return v && v[0] && strcmp(v, "0");
}

static unsigned graph_opt_current_backend(void) {
    if (g_use_vulkan) return GRAPH_OPT_BACKEND_VULKAN;
    if (g_use_opengl) return GRAPH_OPT_BACKEND_OPENGL;
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
    for (int i = 0; i < MAXN; i++) {
        g_node_fusion[i].id = 0;
        g_node_fusion[i].backend_mask = 0;
        g_node_fusion[i].peer_idx = -1;
        g_node_fusion[i].aux_tensor[0] = 0;
    }
    for (int i = 0; i < MAXN; i++) g_concat_sigmoid_fuse[i] = -1;
}

static const GraphNodeFusion* graph_opt_node_fusion(int node_idx, GraphFusionId id) {
    if (node_idx < 0 || node_idx >= MAXN) return NULL;
    const GraphNodeFusion* f = &g_node_fusion[node_idx];
    if (f->id != id) return NULL;
    unsigned backend = graph_opt_current_backend();
    if (f->backend_mask && !(f->backend_mask & backend)) return NULL;
    return f;
}

static int fuse_relu6_pass(void) {
    int applied = 0;
    for (int i = 0; i < g_nn; i++) {
        Node* clip = &g_n[i];
        if (strcmp(clip->op, "Clip")) continue;
        if (p_flt(clip->params, "min", -1e30f) != 0.0f || p_flt(clip->params, "max", 1e30f) != 6.0f) continue;
        const char* tin = node_input_name(clip);
        if (!tin || tensor_use_count(tin) != 1) continue;
        Node* prod = NULL;
        for (int j = 0; j < g_nn; j++) if (!g_n[j].skip && !strcmp(g_n[j].out, tin)) { prod = &g_n[j]; break; }
        if (!prod) continue;
        if (strcmp(prod->op, "QConv2D") && strcmp(prod->op, "Conv2D")) continue;
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
        if (sig->skip || strcmp(sig->op, "Sigmoid")) continue;
        const char* tin = node_input_name(sig);
        if (!tin || tensor_use_count(tin) != 1) continue;
        int prod_idx = -1;
        for (int j = 0; j < g_nn; j++) {
            if (!g_n[j].skip && !strcmp(g_n[j].out, tin)) { prod_idx = j; break; }
        }
        if (prod_idx < 0 || strcmp(g_n[prod_idx].op, "Concat")) continue;
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
        if (add->skip || strcmp(add->op, "Add")) continue;
        if (p_int(add->params, "relu", 0)) continue;
        T* add_out = t_find(add->out);
        if (!add_out || add_out->dtype != T_F32 || add->nin < 2) continue;

        for (int ai = 0; ai < add->nin; ai++) {
            const char* conv_out_name = add->ins[ai].name;
            int conv_idx = producer_index_for(conv_out_name);
            if (conv_idx < 0) continue;
            Node* conv = &g_n[conv_idx];
            if (conv->skip || strcmp(conv->op, "Conv2D")) continue;
            if (tensor_use_count(conv->out) != 1) continue;

            T* conv_out = t_find(conv->out);
            T* in = nin(conv, "input");
            T* wt = nin(conv, "weight");
            if (!conv_out || !in || !wt || conv_out->dtype != T_F32 ||
                in->dtype != T_F32 || (wt->dtype != T_F32 && wt->dtype != T_F16) ||
                in->ndim != 4 || conv_out->ndim != 4 || wt->ndim != 4 ||
                !tensor_same_shape(conv_out, add_out)) continue;

            int sy, sx, py, px, dy, dx;
            p_pair(conv->params, "stride", 1, &sy, &sx);
            p_pair(conv->params, "padding", 0, &py, &px);
            p_pair(conv->params, "dilation", 1, &dy, &dx);
            int pads[4] = {py, px, py, px};
            p_iarr(conv->params, "pads", pads, 4, 0);
            int groups = p_int(conv->params, "groups", 1);
            if (groups != 1 || sy != 1 || sx != 1 || dy != 1 || dx != 1 ||
                pads[0] != 0 || pads[1] != 0 || pads[2] != 0 || pads[3] != 0) continue;

            const char* layout = p_str(conv->params, "weight_layout", "OHWI");
            int kh = 0, kw = 0;
            if (layout_is(layout, "OHWI")) {
                kh = wt->shape[1]; kw = wt->shape[2];
            } else if (layout_is(layout, "HWIO")) {
                kh = wt->shape[0]; kw = wt->shape[1];
            } else {
                continue;
            }
            if (kh != 1 || kw != 1) continue;

            const char* res_name = NULL;
            T* res = NULL;
            for (int ri = 0; ri < add->nin; ri++) {
                if (ri == ai) continue;
                T* t = t_find(add->ins[ri].name);
                if (t && t->dtype == T_F32 && tensor_same_shape(t, add_out)) {
                    res_name = add->ins[ri].name;
                    res = t;
                    break;
                }
            }
            if (!res_name || !res) continue;
            int res_prod = producer_index_any(res_name);
            if (res_prod >= 0 && res_prod > conv_idx) continue;
            if (!strcmp(res_name, add->out) || !strcmp(conv->out, add->out)) continue;

            g_node_fusion[conv_idx].id = GRAPH_FUSION_CONV_ADD;
            g_node_fusion[conv_idx].backend_mask = GRAPH_OPT_CPU_ALL;
            g_node_fusion[conv_idx].peer_idx = i;
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
        if (pw->skip || strcmp(pw->op, "Conv2D")) continue;
        const char* pw_in_name = node_input_name(pw);
        int dw_idx = producer_index_for(pw_in_name);
        if (dw_idx < 0 || dw_idx >= pw_idx) continue;
        Node* dw = &g_n[dw_idx];
        if (dw->skip || strcmp(dw->op, "Conv2D")) continue;
        if (tensor_use_count(dw->out) != 1) continue;

        T* in = nin(dw, "input");
        T* dw_out = t_find(dw->out);
        T* pw_out = t_find(pw->out);
        T* dw_w = nin(dw, "weight");
        T* pw_w = nin(pw, "weight");
        if (!in || !dw_out || !pw_out || !dw_w || !pw_w ||
            in->dtype != T_F32 || dw_out->dtype != T_F32 || pw_out->dtype != T_F32 ||
            (dw_w->dtype != T_F32 && dw_w->dtype != T_F16) ||
            (pw_w->dtype != T_F32 && pw_w->dtype != T_F16) ||
            in->ndim != 4 || dw_out->ndim != 4 || pw_out->ndim != 4 ||
            dw_w->ndim != 4 || pw_w->ndim != 4 ||
            in->shape[0] != pw_out->shape[0] || dw_out->shape[0] != pw_out->shape[0] ||
            dw_out->shape[1] != pw_out->shape[1] || dw_out->shape[2] != pw_out->shape[2]) continue;

        int dw_sy, dw_sx, dw_py, dw_px, dw_dy, dw_dx;
        p_pair(dw->params, "stride", 1, &dw_sy, &dw_sx);
        p_pair(dw->params, "padding", 0, &dw_py, &dw_px);
        p_pair(dw->params, "dilation", 1, &dw_dy, &dw_dx);
        int dw_pads[4] = {dw_py, dw_px, dw_py, dw_px};
        p_iarr(dw->params, "pads", dw_pads, 4, 0);
        int dw_groups = p_int(dw->params, "groups", 1);
        if (dw_groups != in->shape[3] || dw_out->shape[3] != in->shape[3] ||
            dw_dy != 1 || dw_dx != 1) continue;

        const char* dw_layout = p_str(dw->params, "weight_layout", "1HWO");
        int dw_kh = 0, dw_kw = 0, dw_c = 0, dw_mult = 1;
        if (layout_is(dw_layout, "1HWO") || layout_is(dw_layout, "1HWM")) {
            if (dw_w->shape[0] != 1) continue;
            dw_kh = dw_w->shape[1]; dw_kw = dw_w->shape[2]; dw_c = dw_w->shape[3]; dw_mult = 1;
        } else if (layout_is(dw_layout, "HWCM")) {
            dw_kh = dw_w->shape[0]; dw_kw = dw_w->shape[1]; dw_c = dw_w->shape[2]; dw_mult = dw_w->shape[3];
        } else {
            continue;
        }
        if (dw_c != in->shape[3] || dw_mult != 1 || dw_kh <= 0 || dw_kw <= 0) continue;
        if (dw_c < 96) continue;
        if (dw_kh == 3 && dw_kw == 3 && dw_sy == 1 && dw_sx == 1) continue;

        int pw_sy, pw_sx, pw_py, pw_px, pw_dy, pw_dx;
        p_pair(pw->params, "stride", 1, &pw_sy, &pw_sx);
        p_pair(pw->params, "padding", 0, &pw_py, &pw_px);
        p_pair(pw->params, "dilation", 1, &pw_dy, &pw_dx);
        int pw_pads[4] = {pw_py, pw_px, pw_py, pw_px};
        p_iarr(pw->params, "pads", pw_pads, 4, 0);
        int pw_groups = p_int(pw->params, "groups", 1);
        if (pw_groups != 1 || pw_sy != 1 || pw_sx != 1 || pw_dy != 1 || pw_dx != 1 ||
            pw_pads[0] != 0 || pw_pads[1] != 0 || pw_pads[2] != 0 || pw_pads[3] != 0) continue;

        const char* pw_layout = p_str(pw->params, "weight_layout", "OHWI");
        int pw_kh = 0, pw_kw = 0, pw_ic = 0;
        if (layout_is(pw_layout, "OHWI")) {
            pw_kh = pw_w->shape[1]; pw_kw = pw_w->shape[2]; pw_ic = pw_w->shape[3];
        } else if (layout_is(pw_layout, "HWIO")) {
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
        if (add0->skip || strcmp(add0->op, "Add")) continue;
        if (p_int(add0->params, "relu", 0)) continue;
        T* add0_out = t_find(add0->out);
        if (!add0_out || add0_out->dtype != T_F32 || add0->nin < 2) continue;
        if (tensor_use_count(add0->out) != 1) continue;

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
        if (strcmp(add1->op, "Add") || add1->nin < 2) continue;
        T* add1_out = t_find(add1->out);
        if (!add1_out || add1_out->dtype != T_F32 || !tensor_same_shape(add0_out, add1_out)) continue;

        const char* third_name = NULL;
        for (int k = 0; k < add1->nin; k++) {
            if (strcmp(add1->ins[k].name, add0->out)) {
                third_name = add1->ins[k].name;
                break;
            }
        }
        T* third = t_find(third_name);
        if (!third || third->dtype != T_F32 || !tensor_same_shape(third, add1_out)) continue;
        int third_prod = producer_index_any(third_name);
        if (third_prod >= i) continue;

        for (int k = 0; k < add0->nin; k++) {
            T* in = t_find(add0->ins[k].name);
            if (!in || in->dtype != T_F32 || !tensor_same_shape(in, add1_out)) {
                third_name = NULL;
                break;
            }
        }
        if (!third_name) continue;

        g_node_fusion[i].id = GRAPH_FUSION_CHAINED_ELEMENTWISE;
        g_node_fusion[i].backend_mask = GRAPH_OPT_BACKEND_OPENGL;
        g_node_fusion[i].peer_idx = add1_idx;
        strncpy(g_node_fusion[i].aux_tensor, third_name, 127);
        g_node_fusion[i].aux_tensor[127] = 0;
        strncpy(add0->out, add1->out, 127);
        add0->out[127] = 0;
        add1->skip = 1;
        applied++;
    }
    return applied;
}

static int alias_passthrough_pass(void) {
    if (g_use_vulkan || g_use_opengl) return 0;
    int applied = 0;
    for (int i = 0; i < g_nn; i++) {
        Node* n = &g_n[i];
        int passthrough = !strcmp(n->op, "Reshape") || !strcmp(n->op, "Flatten") || !strcmp(n->op, "Squeeze") ||
                          !strcmp(n->op, "Unsqueeze") || !strcmp(n->op, "Dropout") || !strcmp(n->op, "Identity");
        int dequant_after_concat = 0;
        if (!passthrough && !strcmp(n->op, "DequantizeLinear")) {
            int prod_idx = producer_index_for(node_input_name(n));
            dequant_after_concat = prod_idx >= 0 && !strcmp(g_n[prod_idx].op, "Concat");
        }
        if (!passthrough && !dequant_after_concat) {
            continue;
        }
        T* in = nin(n, "input");
        T* out = t_find(n->out);
        if (!in || !out || in->dtype != out->dtype || in->numel != out->numel) continue;
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
        GRAPH_OPT_CPU_ALL,
        "alias_passthrough",
        NULL,
        GRAPH_OPT_STAT_ALIAS,
        alias_passthrough_pass
    },
    {
        GRAPH_FUSION_CONCAT_SIGMOID_INTERNAL,
        GRAPH_OPT_CAT_ELEMENTWISE,
        GRAPH_OPT_CPU_ALL | GRAPH_OPT_GPU_ALL,
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
        GRAPH_FUSION_CONV_ADD,
        GRAPH_OPT_CAT_CONV,
        GRAPH_OPT_BACKEND_CPU_AVX2,
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
