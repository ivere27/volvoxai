/* Shared CPU backward loop; the ordinary instantiation contains no tracing. */
#ifdef VX_TRAINING_TRACE_NODES
#define VX_BACKWARD_FAIL() do { training_trace_end(&traced); return -1; } while (0)
#else
#define VX_BACKWARD_FAIL() return -1
#endif
    for (int i = g_nn - 1; i >= 0; i--) {
        Node* n = &g_n[i];
        if (n->disabled || n->skip) continue;
        int is_split = (n->operator_kind == VX_OP_SPLIT);
        T* out = (n->operator_kind == VX_OP_MOE_ROUTER) ? nout(n, VX_PORT_WEIGHTS) : t_find(n->out);
        if (!out && (n->operator_kind == VX_OP_MOE_ROUTER)) out = nout(n, VX_PORT_ROUTE_WEIGHTS);

        /* Every live output gradient must be consumed by this node's backward
           implementation. Split handles every output; most other operators
           only implement their canonical output. MoERouter implements route
           weights but deliberately does not differentiate its discrete route
           indices. Reject such graphs instead of silently dropping a
           secondary-output branch. */
        for (int output_index = 0; output_index < n->nout; output_index++) {
            T* candidate = t_find(n->outs[output_index].name);
            if (!is_split && candidate && candidate != out && grad_existing(grads, candidate)) {
                fprintf(stderr,
                        "[engine] training backward does not support live secondary output '%s' of op '%s'\n",
                        n->outs[output_index].name, vx_operator_kind_name(n->operator_kind));
                return -1;
            }
        }
        if (is_split) {
            int has_live_output = 0;
            for (int output_index = 0; output_index < n->nout; output_index++) {
                T* candidate = t_find(n->outs[output_index].name);
                if (candidate && grad_existing(grads, candidate)) { has_live_output = 1; break; }
            }
#ifdef VX_TRAINING_TRACE_NODES
            TrainingTrace traced = {0};
            if (has_live_output) traced = training_trace_begin(VX_TRACE_PHASE_BACKWARD,
                vx_operator_kind_name(n->operator_kind), i, NULL, "cpu");
#endif
            int split_result = has_live_output ? backward_split(grads, n) : 0;
#ifdef VX_TRAINING_TRACE_NODES
            training_trace_end(&traced);
#endif
            if (split_result != 0) return -1;
            continue;
        }
        float* go = grad_existing(grads, out);
        if (!out || !go) continue;
#ifdef VX_TRAINING_TRACE_NODES
        TrainingTrace traced = training_trace_begin(VX_TRACE_PHASE_BACKWARD,
            vx_operator_kind_name(n->operator_kind), i, NULL, "cpu");
#endif
        VxOperatorKind op = n->operator_kind;
        if ((op == VX_OP_MATMUL) || (op == VX_OP_LINEAR) || (op == VX_OP_GEMM)) {
            if (backward_matmul(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_ADD) {
            T* a = nin(n, VX_PORT_A);
            if (!a) a = node_input_default(n);
            T* b = nin(n, VX_PORT_B);
            if (!a || !b || a->dtype != T_F32 || b->dtype != T_F32) VX_BACKWARD_FAIL();
            const float* add_go = go;
            float* gated = NULL;
            int relu = vx_node_param_i32(n, VX_NODE_PARAM_RELU, 0);
            if (relu) {
                gated = (float*)malloc((size_t)out->numel * sizeof(float));
                if (!gated) VX_BACKWARD_FAIL();
                for (long j = 0; j < out->numel; j++) {
                    gated[j] = ((relu == 1 && out->data[j] <= 0.0f) ||
                                (relu == 2 && (out->data[j] <= 0.0f || out->data[j] >= 6.0f))) ? 0.0f : go[j];
                }
                add_go = gated;
            }
            float* ga = grad_for(grads, a);
            float* gb = grad_for(grads, b);
            if (!ga || !gb) { free(gated); VX_BACKWARD_FAIL(); }
            if (grad_add_broadcast_tensor(ga, a, add_go, out) != 0 ||
                grad_add_broadcast_tensor(gb, b, add_go, out) != 0) {
                free(gated);
                VX_BACKWARD_FAIL();
            }
            free(gated);
        } else if (op == VX_OP_MUL) {
            if (backward_mul(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_EMBEDDING) {
            if (backward_embedding(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_LAYER_NORM) {
            if (backward_layernorm(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_GROUP_NORM) {
            if (backward_groupnorm(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_RMS_NORM) {
            if (backward_rmsnorm(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if ((op == VX_OP_RELU) || (op == VX_OP_GELU) || (op == VX_OP_SILU) ||
                   (op == VX_OP_SIGMOID) || (op == VX_OP_TANH) ||
                   (op == VX_OP_HARD_SWISH) || (op == VX_OP_HARD_SIGMOID) ||
                   (op == VX_OP_LEAKY_RELU) || (op == VX_OP_CLIP)) {
            if (backward_activation(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_PRELU) {
            if (backward_prelu(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if ((op == VX_OP_SOFTMAX) || (op == VX_OP_LOG_SOFTMAX)) {
            if (backward_softmax(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_SDPA) {
            if (backward_sdpa(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_CROSS_SDPA) {
            if (backward_cross_sdpa(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_MOE_LINEAR) {
            if (backward_moe_linear(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_MOE_ROUTER) {
            if (backward_moe_router(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_CONV_2D) {
            if (backward_conv2d(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_BATCH_NORM_2D) {
            if (backward_batchnorm2d(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_MAX_POOL_2D) {
            if (backward_maxpool2d(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_GLOBAL_AVERAGE_POOL) {
            if (backward_global_average_pool(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if ((op == VX_OP_RESIZE) || (op == VX_OP_RESIZE_NEAREST_2D) || (op == VX_OP_UPSAMPLE_NEAREST_2D)) {
            if (backward_resize2d(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_CONCAT) {
            if (backward_concat(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_TRANSPOSE) {
            T* in = node_input_default(n);
            if (transpose_backward(grads, n, in, out, go) != 0) VX_BACKWARD_FAIL();
        } else if (op == VX_OP_DROPOUT) {
            if (backward_dropout(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if ((op == VX_OP_REDUCE_SUM) || (op == VX_OP_REDUCE_MEAN)) {
            if (backward_last_axis_reduction(grads, n, out, go) != 0) VX_BACKWARD_FAIL();
        } else if ((op == VX_OP_IDENTITY) || (op == VX_OP_RESHAPE) || (op == VX_OP_FLATTEN) ||
                   (op == VX_OP_SQUEEZE) || (op == VX_OP_UNSQUEEZE)) {
            T* in = node_input_default(n);
            if (in && in->dtype == T_F32 && in->numel == out->numel) {
                grad_add_full(grad_for(grads, in), go, out->numel, 1.0f);
            } else VX_BACKWARD_FAIL();
        } else if (training_extended_backward(NULL, grads, n, out, go) != 0) {
            fprintf(stderr, "[engine] training backward does not support op '%s'\n", vx_operator_kind_name(op));
            VX_BACKWARD_FAIL();
        }
#ifdef VX_TRAINING_TRACE_NODES
        training_trace_end(&traced);
#endif
    }

#undef VX_BACKWARD_FAIL
