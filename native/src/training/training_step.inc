/*
 * Training backward driver, graph state, accumulation, and public train-step ABI.
 *
 * Fragment of the training runtime translation unit: included only from
 * training_runtime.inc, never compiled or included on its own. Walks the graph
 * to dispatch CPU/GPU backward, owns training-graph preparation/restore and
 * gradient accumulation, and exposes volvoxai_engine_train_step*().
 */

static int backward_supported_graph(float** grads) {
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
            if (has_live_output && backward_split(grads, n) != 0) return -1;
            continue;
        }
        float* go = grad_existing(grads, out);
        if (!out || !go) continue;
        VxOperatorKind op = n->operator_kind;
        if ((op == VX_OP_MATMUL) || (op == VX_OP_LINEAR) || (op == VX_OP_GEMM)) {
            if (backward_matmul(grads, n, out, go) != 0) return -1;
        } else if (op == VX_OP_ADD) {
            T* a = nin(n, VX_PORT_A);
            if (!a) a = node_input_default(n);
            T* b = nin(n, VX_PORT_B);
            if (!a || !b || a->dtype != T_F32 || b->dtype != T_F32) return -1;
            const float* add_go = go;
            float* gated = NULL;
            int relu = vx_node_param_i32(n, VX_NODE_PARAM_RELU, 0);
            if (relu) {
                gated = (float*)malloc((size_t)out->numel * sizeof(float));
                if (!gated) return -1;
                for (long j = 0; j < out->numel; j++) {
                    gated[j] = ((relu == 1 && out->data[j] <= 0.0f) ||
                                (relu == 2 && (out->data[j] <= 0.0f || out->data[j] >= 6.0f))) ? 0.0f : go[j];
                }
                add_go = gated;
            }
            float* ga = grad_for(grads, a);
            float* gb = grad_for(grads, b);
            if (!ga || !gb) { free(gated); return -1; }
            if (grad_add_broadcast_tensor(ga, a, add_go, out) != 0 ||
                grad_add_broadcast_tensor(gb, b, add_go, out) != 0) {
                free(gated);
                return -1;
            }
            free(gated);
        } else if (op == VX_OP_MUL) {
            if (backward_mul(grads, n, out, go) != 0) return -1;
        } else if (op == VX_OP_EMBEDDING) {
            if (backward_embedding(grads, n, out, go) != 0) return -1;
        } else if (op == VX_OP_LAYER_NORM) {
            if (backward_layernorm(grads, n, out, go) != 0) return -1;
        } else if (op == VX_OP_GROUP_NORM) {
            if (backward_groupnorm(grads, n, out, go) != 0) return -1;
        } else if (op == VX_OP_RMS_NORM) {
            if (backward_rmsnorm(grads, n, out, go) != 0) return -1;
        } else if ((op == VX_OP_RELU) || (op == VX_OP_GELU) || (op == VX_OP_SILU) ||
                   (op == VX_OP_SIGMOID) || (op == VX_OP_TANH) ||
                   (op == VX_OP_HARD_SWISH) || (op == VX_OP_HARD_SIGMOID) ||
                   (op == VX_OP_LEAKY_RELU) || (op == VX_OP_CLIP)) {
            if (backward_activation(grads, n, out, go) != 0) return -1;
        } else if (op == VX_OP_PRELU) {
            if (backward_prelu(grads, n, out, go) != 0) return -1;
        } else if ((op == VX_OP_SOFTMAX) || (op == VX_OP_LOG_SOFTMAX)) {
            if (backward_softmax(grads, n, out, go) != 0) return -1;
        } else if (op == VX_OP_SDPA) {
            if (backward_sdpa(grads, n, out, go) != 0) return -1;
        } else if (op == VX_OP_CROSS_SDPA) {
            if (backward_cross_sdpa(grads, n, out, go) != 0) return -1;
        } else if (op == VX_OP_MOE_LINEAR) {
            if (backward_moe_linear(grads, n, out, go) != 0) return -1;
        } else if (op == VX_OP_MOE_ROUTER) {
            if (backward_moe_router(grads, n, out, go) != 0) return -1;
        } else if (op == VX_OP_CONV_2D) {
            if (backward_conv2d(grads, n, out, go) != 0) return -1;
        } else if (op == VX_OP_BATCH_NORM_2D) {
            if (backward_batchnorm2d(grads, n, out, go) != 0) return -1;
        } else if (op == VX_OP_MAX_POOL_2D) {
            if (backward_maxpool2d(grads, n, out, go) != 0) return -1;
        } else if (op == VX_OP_GLOBAL_AVERAGE_POOL) {
            if (backward_global_average_pool(grads, n, out, go) != 0) return -1;
        } else if ((op == VX_OP_RESIZE) || (op == VX_OP_RESIZE_NEAREST_2D) || (op == VX_OP_UPSAMPLE_NEAREST_2D)) {
            if (backward_resize2d(grads, n, out, go) != 0) return -1;
        } else if (op == VX_OP_CONCAT) {
            if (backward_concat(grads, n, out, go) != 0) return -1;
        } else if (op == VX_OP_TRANSPOSE) {
            T* in = node_input_default(n);
            if (transpose_backward(grads, n, in, out, go) != 0) return -1;
        } else if (op == VX_OP_DROPOUT) {
            if (backward_dropout(grads, n, out, go) != 0) return -1;
        } else if ((op == VX_OP_REDUCE_SUM) || (op == VX_OP_REDUCE_MEAN)) {
            if (backward_last_axis_reduction(grads, n, out, go) != 0) return -1;
        } else if ((op == VX_OP_IDENTITY) || (op == VX_OP_RESHAPE) || (op == VX_OP_FLATTEN) ||
                   (op == VX_OP_SQUEEZE) || (op == VX_OP_UNSQUEEZE)) {
            T* in = node_input_default(n);
            if (in && in->dtype == T_F32 && in->numel == out->numel) {
                grad_add_full(grad_for(grads, in), go, out->numel, 1.0f);
            } else return -1;
        } else if (training_extended_backward(NULL, grads, n, out, go) != 0) {
            fprintf(stderr, "[engine] training backward does not support op '%s'\n", vx_operator_kind_name(op));
            return -1;
        }
    }
    return 0;
}

typedef struct {
    char out[128];
    int skip;
    int fuse_relu6;
} TrainingNodeState;

typedef struct {
    int tensor_index;
    float* inference_data;
    int inference_owns;
} TrainingAliasState;

typedef struct {
    TrainingNodeState* nodes;
    TrainingAliasState* aliases;
    int alias_count;
    GraphNodeFusion* fusions;
    int* concat_sigmoid_fuse;
    int node_count;
    int use_vulkan;
    int use_opengl;
    int use_metal;
    int use_cuda;
    NativeGpuTrainingBackend gpu_training_backend;
    int backends_saved;
    int base_route_set;
    int training_context_saved;
    int saved_training_mode;
    uint32_t saved_training_counter;
} TrainingGraphState;

static void training_graph_release_base_route(TrainingGraphState* state) {
    if (state && state->base_route_set) {
        vx_adapter_request_end();
        state->base_route_set = 0;
    }
}

static void training_graph_restore_backends(TrainingGraphState* state) {
    if (!state || !state->backends_saved) return;
    g_use_vulkan = state->use_vulkan;
    g_use_opengl = state->use_opengl;
    g_use_metal = state->use_metal;
#if VOLVOXAI_ENABLE_CUDA
    g_use_cuda = state->use_cuda;
#endif
    state->backends_saved = 0;
}

static void training_graph_restore_aliases(TrainingGraphState* state) {
    if (!state || !state->aliases) return;
    for (int i = state->alias_count - 1; i >= 0; i--) {
        TrainingAliasState* alias = &state->aliases[i];
        if (alias->tensor_index < 0 || alias->tensor_index >= g_nt) continue;
        T* tensor = &g_t[alias->tensor_index];
        if (tensor->owns && tensor->data && tensor->data != alias->inference_data)
            free(tensor->data);
        tensor->data = alias->inference_data;
        tensor->owns = alias->inference_owns;
    }
    state->alias_count = 0;
}

static int training_graph_detach_inference_aliases(TrainingGraphState* state) {
    if (!state || !state->aliases) return -1;
    /* Collect the complete canonical-output alias group first so passthrough
       chains are recognized before any pointer is changed. Arena aliases were
       already expanded by free_node_caches_for_training(), leaving optimizer
       aliases here. */
    for (int i = 0; i < g_nn; i++) {
        Node* node = &g_n[i];
        T* output = t_find(node->out);
        if (!output || !output->data) continue;
        int aliases_input = 0;
        for (int input_index = 0; input_index < node->nin; input_index++) {
            T* input = t_find(node->ins[input_index].name);
            if (input && input != output && input->data == output->data) {
                aliases_input = 1;
                break;
            }
        }
        if (!aliases_input) continue;
        int output_index = (int)(output - g_t);
        int duplicate = 0;
        for (int j = 0; j < state->alias_count; j++) {
            if (state->aliases[j].tensor_index == output_index) { duplicate = 1; break; }
        }
        if (duplicate || state->alias_count >= g_nn) continue;
        TrainingAliasState* alias = &state->aliases[state->alias_count++];
        alias->tensor_index = output_index;
        alias->inference_data = output->data;
        alias->inference_owns = output->owns;
    }
    for (int i = 0; i < state->alias_count; i++) {
        T* output = &g_t[state->aliases[i].tensor_index];
        if (output->numel <= 0 || output->elem_size == 0 ||
            (uint64_t)output->numel > SIZE_MAX / output->elem_size) {
            training_graph_restore_aliases(state);
            return -1;
        }
        float* training_data = (float*)calloc((size_t)output->numel, output->elem_size);
        if (!training_data) {
            training_graph_restore_aliases(state);
            return -1;
        }
        output->data = training_data;
        output->owns = 1;
    }
    return 0;
}

static void training_graph_state_free(TrainingGraphState* state) {
    if (!state) return;
    training_graph_restore_aliases(state);
    free(state->nodes);
    free(state->aliases);
    free(state->fusions);
    free(state->concat_sigmoid_fuse);
    memset(state, 0, sizeof(*state));
}

/*
 * Inference graph optimization reuses activation buffers and may skip fused
 * nodes. Backward needs every forward activation, so a train step temporarily
 * restores distinct buffers and the unfused execution order. Nothing in this
 * path is touched by inference-only calls.
 */
static uint32_t training_step_counter(long step, uint32_t accumulation_steps, int microbatch) {
    uint32_t counter = (uint32_t)((uint64_t)(step - 1) * accumulation_steps +
                                  (uint64_t)microbatch + 1u);
    return counter ^ vx_engine_state_current()->native_training_rng_seed ^
        vx_engine_state_current()->native_training_shape_hash;
}

static int training_graph_prepare(TrainingGraphState* state, uint32_t training_counter) {
    if (!state || g_nn <= 0 || volvoxai_engine_model_route_lease_active()) return -1;
    memset(state, 0, sizeof(*state));
    state->nodes = (TrainingNodeState*)calloc((size_t)g_nn, sizeof(*state->nodes));
    state->aliases = (TrainingAliasState*)calloc((size_t)g_nn, sizeof(*state->aliases));
    state->fusions = (GraphNodeFusion*)malloc(
        (size_t)g_nn * sizeof(g_node_fusion[0]));
    state->concat_sigmoid_fuse = (int*)malloc(
        (size_t)g_nn * sizeof(g_concat_sigmoid_fuse[0]));
    if (!state->nodes || !state->aliases || !state->fusions || !state->concat_sigmoid_fuse) {
        training_graph_state_free(state);
        return -1;
    }
    state->node_count = g_nn;
    for (int i = 0; i < g_nn; i++) {
        strncpy(state->nodes[i].out, g_n[i].out, sizeof(state->nodes[i].out) - 1);
        state->nodes[i].skip = g_n[i].skip;
        state->nodes[i].fuse_relu6 = g_n[i].fuse_relu6;
    }
    memcpy(state->fusions, g_node_fusion,
           (size_t)g_nn * sizeof(g_node_fusion[0]));
    memcpy(state->concat_sigmoid_fuse, g_concat_sigmoid_fuse,
           (size_t)g_nn * sizeof(g_concat_sigmoid_fuse[0]));

    /* Staged adapters live outside graph autograd. Pin an explicit empty route
       so an active global adapter cannot affect this training forward. */
    if (vx_adapter_request_begin("") != 0) {
        training_graph_state_free(state);
        return -1;
    }
    state->base_route_set = 1;

    /* Select the training route before cache teardown. The teardown decision
     * must know whether CUDA weights can remain resident or whether a CPU/
     * other-backend handoff must first materialize them. */
    state->use_vulkan = g_use_vulkan;
    state->use_opengl = g_use_opengl;
    state->use_metal = g_use_metal;
#if VOLVOXAI_ENABLE_CUDA
    state->use_cuda = g_use_cuda;
#endif
    state->backends_saved = 1;
    state->gpu_training_backend = gpu_training_select_backend();
    g_use_vulkan = state->gpu_training_backend == NATIVE_GPU_TRAINING_VULKAN;
    g_use_opengl = state->gpu_training_backend == NATIVE_GPU_TRAINING_OPENGL;
    g_use_metal = state->gpu_training_backend == NATIVE_GPU_TRAINING_METAL;
#if VOLVOXAI_ENABLE_CUDA
    g_use_cuda = state->gpu_training_backend == NATIVE_GPU_TRAINING_CUDA;
#endif

    if (free_node_caches_for_training() != 0) {
        training_graph_restore_backends(state);
        training_graph_release_base_route(state);
        training_graph_state_free(state);
        return -1;
    }
    if (training_graph_detach_inference_aliases(state) != 0) {
        training_graph_restore_backends(state);
        training_graph_release_base_route(state);
        training_graph_state_free(state);
        return -1;
    }

    /* Restore producer output names changed by Conv+Add and chained-Add fusions. */
    for (int producer = 0; producer < g_nn; producer++) {
        const GraphNodeFusion* fusion = &g_node_fusion[producer];
        int peer = fusion->peer_idx;
        if (peer < 0 || peer >= g_nn) continue;
        if (fusion->id == GRAPH_FUSION_CONV_ADD || fusion->id == GRAPH_FUSION_CHAINED_ELEMENTWISE) {
            Node* consumer = &g_n[peer];
            for (int k = 0; k < consumer->nin; k++) {
                const char* candidate = consumer->ins[k].name;
                if (strcmp(candidate, fusion->aux_tensor)) {
                    strncpy(g_n[producer].out, candidate, sizeof(g_n[producer].out) - 1);
                    g_n[producer].out[sizeof(g_n[producer].out) - 1] = 0;
                    break;
                }
            }
        }
    }
    /* ReLU6 fusion runs before the fusions above, so undo it second. */
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
    /* CUDA training consumes canonical graph weights directly and the
     * training-specific teardown above already removed host CPU packs. Do not
     * enter the generic lazy refresh: it performs a full backend reset and
     * would discard persistent CUDA weight slots before every microbatch.
     * Non-CUDA training retains the established host-cache rebuild path. */
    g_weight_caches_dirty =
        state->gpu_training_backend == NATIVE_GPU_TRAINING_CUDA ? 0 : 1;
    state->saved_training_mode = g_native_training_mode;
    state->saved_training_counter = g_native_training_counter;
    state->training_context_saved = 1;
    g_native_training_mode = 1;
    g_native_training_counter = training_counter;
    return 0;
}

static int training_graph_restore(TrainingGraphState* state) {
    int restored_cuda = 0;
    if (!state) return -1;
    if (state->training_context_saved) {
        g_native_training_mode = state->saved_training_mode;
        g_native_training_counter = state->saved_training_counter;
        state->training_context_saved = 0;
    }
    training_graph_release_base_route(state);
#if VOLVOXAI_ENABLE_CUDA
    if (state->backends_saved) restored_cuda = state->use_cuda;
#endif
    training_graph_restore_backends(state);
    if (!state->nodes || state->node_count != g_nn) {
        training_graph_state_free(state);
        return -1;
    }
    int rc = free_node_caches_for_training();
    for (int i = 0; i < g_nn; i++) {
        strncpy(g_n[i].out, state->nodes[i].out, sizeof(g_n[i].out) - 1);
        g_n[i].out[sizeof(g_n[i].out) - 1] = 0;
        g_n[i].skip = state->nodes[i].skip;
        g_n[i].fuse_relu6 = state->nodes[i].fuse_relu6;
    }
    memcpy(g_node_fusion, state->fusions,
           (size_t)g_nn * sizeof(g_node_fusion[0]));
    memcpy(g_concat_sigmoid_fuse, state->concat_sigmoid_fuse,
           (size_t)g_nn * sizeof(g_concat_sigmoid_fuse[0]));
    training_graph_restore_aliases(state);
    if (rc == 0 && build_quant_params_pass() != 0) rc = -1;
    volvoxai_engine_finish_tensor_table_mutation();
    /* A restored CUDA inference route has no use for host Conv/F16 packs.
     * Leaving them absent also avoids rebuilding host caches at a
     * device-resident training boundary. */
    if (rc == 0 && !restored_cuda && prepack_cpu_weights() != 0) rc = -1;
    if (rc == 0) g_weight_caches_dirty = 0;
    training_graph_state_free(state);
    return rc;
}

static char* training_string_copy(const char* value) {
    if (!value) return NULL;
    size_t n = strlen(value) + 1;
    char* copy = (char*)malloc(n);
    if (copy) memcpy(copy, value, n);
    return copy;
}

void volvoxai_engine_reset_training_accumulation(void) {
    TrainingAccumulationState* state = &g_training_accumulation;
#if VOLVOXAI_ENABLE_CUDA
    cuda_training_window_destroy(state->cuda_window);
#endif
    if (state->gradients) {
        for (int i = 0; i < state->trainable_count; i++) free(state->gradients[i]);
    }
    if (state->trainable_names) {
        for (int i = 0; i < state->trainable_count; i++) free(state->trainable_names[i]);
    }
    if (state->losses) {
        for (int i = 0; i < state->loss_count; i++) {
            free(state->losses[i].name);
            free(state->losses[i].logits_name);
        }
    }
    free(state->gradients);
    free(state->trainable_names);
    free(state->trainable_numel);
    free(state->losses);
    memset(state, 0, sizeof(*state));
}

int volvoxai_engine_training_accumulation_pending(void) {
    return g_training_accumulation.active;
}

int volvoxai_engine_reset_gradient_accumulation(void) {
    if (volvoxai_engine_model_route_lease_active()) return -1;
    volvoxai_engine_model_lock();
    volvoxai_engine_reset_training_accumulation();
    volvoxai_engine_model_unlock();
    return 0;
}

static uint64_t training_hash_bytes(uint64_t hash, const void* data, size_t size) {
    const unsigned char* bytes = (const unsigned char*)data;
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t training_topology_hash(void) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = training_hash_bytes(hash, &g_nn, sizeof(g_nn));
    for (int i = 0; i < g_nn; i++) {
        Node* node = &g_n[i];
        hash = training_hash_bytes(hash, &node->operator_kind, sizeof(node->operator_kind));
        hash = training_hash_bytes(hash, node->out, strlen(node->out) + 1);
        hash = training_hash_bytes(hash, &node->nin, sizeof(node->nin));
        hash = training_hash_bytes(hash, &node->nout, sizeof(node->nout));
        hash = training_hash_bytes(hash, &node->disabled, sizeof(node->disabled));
        for (int j = 0; j < node->nin; j++) {
            hash = training_hash_bytes(hash, node->ins[j].key, sizeof(node->ins[j].key));
            hash = training_hash_bytes(hash, node->ins[j].name, sizeof(node->ins[j].name));
        }
        for (int j = 0; j < node->nout; j++) {
            hash = training_hash_bytes(hash, node->outs[j].key, sizeof(node->outs[j].key));
            hash = training_hash_bytes(hash, node->outs[j].name, sizeof(node->outs[j].name));
        }
        char* params = node->params ? cJSON_PrintUnformatted(node->params) : NULL;
        if (params) {
            hash = training_hash_bytes(hash, params, strlen(params) + 1);
            free(params);
        }
    }
    return hash;
}

static int training_accumulation_signature_matches(
        const volvoxai_cross_entropy_loss_t* losses, int loss_count,
        const char* const* trainable_names, int trainable_count,
        int update_mode, float learning_rate, float beta1, float beta2,
        float epsilon, float weight_decay, float max_grad_norm, long step,
        int accumulation_steps, uint64_t topology_hash,
        NativeGpuTrainingBackend backend) {
    TrainingAccumulationState* state = &g_training_accumulation;
    if (!state->active) return 1;
    if (state->backend != backend || state->loss_count != loss_count ||
        state->trainable_count != trainable_count ||
        state->accumulation_steps != accumulation_steps || state->tensor_count != g_nt ||
        state->node_count != g_nn || state->topology_hash != topology_hash ||
        state->update_mode != update_mode || state->learning_rate != learning_rate ||
        state->beta1 != beta1 || state->beta2 != beta2 || state->epsilon != epsilon ||
        state->weight_decay != weight_decay || state->max_grad_norm != max_grad_norm ||
        state->step != step) return 0;
    for (int i = 0; i < trainable_count; i++) {
        T* tensor = t_find(trainable_names[i]);
        if (!tensor || strcmp(state->trainable_names[i], trainable_names[i]) ||
            state->trainable_numel[i] != tensor->numel) return 0;
    }
    for (int i = 0; i < loss_count; i++) {
        if (strcmp(state->losses[i].name, losses[i].name) ||
            strcmp(state->losses[i].logits_name, losses[i].logits_name) ||
            state->losses[i].weight != losses[i].weight ||
            state->losses[i].normalizer != losses[i].normalizer) return 0;
    }
    return 1;
}

static int training_accumulation_begin(
        const volvoxai_cross_entropy_loss_t* losses, int loss_count,
        const char* const* trainable_names, int trainable_count,
        int update_mode, float learning_rate, float beta1, float beta2,
        float epsilon, float weight_decay, float max_grad_norm, long step,
        int accumulation_steps, uint64_t topology_hash,
        NativeGpuTrainingBackend backend) {
    TrainingAccumulationState* state = &g_training_accumulation;
    if (state->active) return 0;
    memset(state, 0, sizeof(*state));
    state->trainable_names = (char**)calloc((size_t)trainable_count, sizeof(char*));
    state->trainable_numel = (long*)calloc((size_t)trainable_count, sizeof(long));
    state->gradients = (float**)calloc((size_t)trainable_count, sizeof(float*));
    state->losses = (TrainingAccumulationLoss*)calloc((size_t)loss_count, sizeof(*state->losses));
    state->trainable_count = trainable_count;
    state->loss_count = loss_count;
    if (!state->trainable_names || !state->trainable_numel || !state->gradients || !state->losses)
        goto fail;
    for (int i = 0; i < trainable_count; i++) {
        T* tensor = t_find(trainable_names[i]);
        if (!tensor || tensor->numel <= 0) goto fail;
        state->trainable_names[i] = training_string_copy(trainable_names[i]);
        state->trainable_numel[i] = tensor->numel;
        if (backend != NATIVE_GPU_TRAINING_CUDA)
            state->gradients[i] = (float*)calloc(
                (size_t)tensor->numel, sizeof(float));
        if (!state->trainable_names[i] ||
            (backend != NATIVE_GPU_TRAINING_CUDA && !state->gradients[i]))
            goto fail;
    }
    for (int i = 0; i < loss_count; i++) {
        state->losses[i].name = training_string_copy(losses[i].name);
        state->losses[i].logits_name = training_string_copy(losses[i].logits_name);
        state->losses[i].weight = losses[i].weight;
        state->losses[i].normalizer = losses[i].normalizer;
        if (!state->losses[i].name || !state->losses[i].logits_name) goto fail;
    }
    state->accumulation_steps = accumulation_steps;
    state->tensor_count = g_nt;
    state->node_count = g_nn;
    state->topology_hash = topology_hash;
    state->update_mode = update_mode;
    state->learning_rate = learning_rate;
    state->beta1 = beta1;
    state->beta2 = beta2;
    state->epsilon = epsilon;
    state->weight_decay = weight_decay;
    state->max_grad_norm = max_grad_norm;
    state->step = step;
    state->backend = backend;
#if VOLVOXAI_ENABLE_CUDA
    if (backend == NATIVE_GPU_TRAINING_CUDA) {
        void** hosts = (void**)calloc((size_t)trainable_count, sizeof(void*));
        size_t* bytes = (size_t*)calloc((size_t)trainable_count,
                                       sizeof(size_t));
        if (!hosts || !bytes) {
            free(hosts); free(bytes);
            goto fail;
        }
        for (int i = 0; i < trainable_count; i++) {
            T* tensor = t_find(trainable_names[i]);
            hosts[i] = tensor ? tensor->data : NULL;
            if (!tensor || (size_t)tensor->numel > SIZE_MAX / sizeof(float)) {
                free(hosts); free(bytes);
                goto fail;
            }
            bytes[i] = (size_t)tensor->numel * sizeof(float);
        }
        state->cuda_window = cuda_training_window_create(
            hosts, bytes, trainable_count);
        free(hosts); free(bytes);
        if (!state->cuda_window) goto fail;
    }
#else
    if (backend == NATIVE_GPU_TRAINING_CUDA) goto fail;
#endif
    state->active = 1;
    return 0;
fail:
    volvoxai_engine_reset_training_accumulation();
    return -1;
}

static void training_accumulation_outputs(float* out_loss,
        volvoxai_cross_entropy_metric_t* out_metrics,
        int* out_accumulated_microbatches, int* out_update_applied, int applied) {
    TrainingAccumulationState* state = &g_training_accumulation;
    if (out_loss) *out_loss = state->loss;
    if (out_metrics) {
        for (int i = 0; i < state->loss_count; i++) out_metrics[i] = state->losses[i].metric;
    }
    if (out_accumulated_microbatches) *out_accumulated_microbatches = state->microbatches;
    if (out_update_applied) *out_update_applied = applied;
}

#include "training_cuda_trainstep.inc"

static int volvoxai_engine_train_step_multi_locked_impl(
        const volvoxai_cross_entropy_loss_t* losses, int loss_count,
        const char* const* trainable_names, int trainable_count,
        int update_mode, float learning_rate, float beta1, float beta2,
        float epsilon, float weight_decay, float max_grad_norm, long step,
        int accumulation_steps, int flush_accumulation, int reset_accumulation,
        float* out_loss, volvoxai_cross_entropy_metric_t* out_metrics,
        int* out_accumulated_microbatches, int* out_update_applied) {
    g_last_training_backend = NATIVE_GPU_TRAINING_NONE;
    if (out_loss) *out_loss = 0.0f;
    if (out_accumulated_microbatches) *out_accumulated_microbatches = 0;
    if (out_update_applied) *out_update_applied = 0;
    if (!g_loaded || !losses || loss_count <= 0 || !trainable_names || trainable_count <= 0 ||
        accumulation_steps <= 0 || (flush_accumulation != 0 && flush_accumulation != 1) ||
        (reset_accumulation != 0 && reset_accumulation != 1)) return -1;
    if ((update_mode != VOLVOXAI_TENSOR_UPDATE_SGD &&
         update_mode != VOLVOXAI_TENSOR_UPDATE_ADAMW) ||
        volvoxai_engine_model_route_lease_active()) return -1;
    if (!isfinite(learning_rate) || learning_rate < 0.0f ||
        !isfinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f ||
        !isfinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        !isfinite(weight_decay) || weight_decay < 0.0f ||
        !isfinite(max_grad_norm) || max_grad_norm < 0.0f || step <= 0) return -1;
    if (reset_accumulation) volvoxai_engine_reset_training_accumulation();

    for (int i = 0; i < loss_count; i++) {
        const volvoxai_cross_entropy_loss_t* loss = &losses[i];
        if (!loss->name || !loss->name[0] || !loss->logits_name || !loss->logits_name[0] ||
            !loss->targets || loss->target_count <= 0 || loss->row_index < -1 ||
            !isfinite(loss->weight) || loss->weight < 0.0f ||
            !isfinite(loss->normalizer) || loss->normalizer < 0.0f ||
            (accumulation_steps > 1 && loss->normalizer <= 0.0f)) return -1;
        for (int j = 0; j < i; j++) if (!strcmp(losses[j].name, loss->name)) return -1;
    }
    for (int i = 0; i < trainable_count; i++) {
        if (!trainable_names[i] || !trainable_names[i][0]) return -1;
        for (int j = 0; j < i; j++) if (!strcmp(trainable_names[i], trainable_names[j])) return -1;
        T* trainable = t_find(trainable_names[i]);
        if (!trainable || trainable->dtype != T_F32 || trainable->numel <= 0 || !trainable->data ||
            !volvoxai_engine_tensor_is_model_weight_locked(trainable->name)) return -1;
    }

    uint64_t topology_hash = training_topology_hash();
    NativeGpuTrainingBackend accumulation_backend =
        g_training_accumulation.active
        ? g_training_accumulation.backend
        : gpu_training_select_backend();
    if (!training_accumulation_signature_matches(losses, loss_count, trainable_names, trainable_count,
            update_mode, learning_rate, beta1, beta2, epsilon, weight_decay, max_grad_norm,
            step, accumulation_steps, topology_hash,
            accumulation_backend)) return -1;

    uint32_t training_counter = training_step_counter(step, (uint32_t)accumulation_steps,
        g_training_accumulation.active ? g_training_accumulation.microbatches : 0);
    TrainingGraphState training_graph;
    if (training_graph_prepare(&training_graph, training_counter) != 0) return -1;
    if (g_required_training_backend != NATIVE_GPU_TRAINING_NONE &&
        training_graph.gpu_training_backend != g_required_training_backend) {
        training_graph_restore(&training_graph);
        return -1;
    }
    NativeGpuTrainingBackend step_backend =
        training_graph.gpu_training_backend;
    int saved_execution_row = g_execution_row;
    g_execution_row = loss_count == 1 && losses[0].target_count == 1 ? losses[0].row_index : -1;
    int forward_rc = volvoxai_engine_forward_locked();
    g_execution_row = saved_execution_row;
    if (forward_rc != 0) {
        training_graph_restore(&training_graph);
        return -1;
    }
#if VOLVOXAI_ENABLE_CUDA
    if (step_backend == NATIVE_GPU_TRAINING_CUDA) {
        /* Generic elementwise graph APIs do not carry tensor-role metadata.
         * Promote every model-backed tensor actually touched by the forward
         * so constants such as residual offsets survive transient tape
         * cleanup alongside Linear/Conv/Norm parameters. The CUDA helper is
         * classification-only and does not allocate unused weights. */
        for (int tensor_index = 0; tensor_index < g_nt; tensor_index++) {
            T* tensor = &g_t[tensor_index];
            if (!volvoxai_engine_tensor_is_model_weight_locked(
                    tensor->name) || !tensor->data || tensor->numel <= 0 ||
                !tensor->elem_size ||
                (size_t)tensor->numel > SIZE_MAX / tensor->elem_size)
                continue;
            cuda_graph_retain_weight(
                tensor->data, (size_t)tensor->numel * tensor->elem_size);
        }
    }
#endif

    float** grads = (float**)calloc((size_t)g_nt, sizeof(float*));
    volvoxai_cross_entropy_metric_t* current_metrics =
        (volvoxai_cross_entropy_metric_t*)calloc((size_t)loss_count, sizeof(*current_metrics));
    if (!grads || !current_metrics) {
        free(grads); free(current_metrics);
        training_graph_restore(&training_graph);
        return -1;
    }
#if VOLVOXAI_ENABLE_CUDA
    if (step_backend == NATIVE_GPU_TRAINING_CUDA &&
        (!g_training_accumulation.active ||
         g_training_accumulation.backend == NATIVE_GPU_TRAINING_CUDA)) {
        int cuda_rc = cuda_training_run_microbatch(
            &training_graph, losses, loss_count, trainable_names,
            trainable_count, update_mode, learning_rate, beta1, beta2,
            epsilon, weight_decay, max_grad_norm, step, accumulation_steps,
            flush_accumulation, topology_hash, grads, current_metrics,
            out_loss, out_metrics, out_accumulated_microbatches,
            out_update_applied);
        if (cuda_rc <= 0) {
            int restore_rc = training_graph_restore(&training_graph);
            for (int tensor_index = 0; tensor_index < g_nt; tensor_index++)
                free(grads[tensor_index]);
            free(grads);
            free(current_metrics);
            return cuda_rc == 0 && restore_rc == 0 ? 0 : -1;
        }
        /* A non-required CUDA plan may still use the established CPU
           fallback below. The helper returns 1 only before opening a CUDA
           scope or mutating the accumulation window. */
    }
#endif
    int rc = 0;
    int current_has_contribution = 0;
    float current_loss = 0.0f;
    for (int i = 0; i < loss_count; i++) {
        T* logits = t_find(losses[i].logits_name);
        if (!logits) { rc = -1; break; }
        /* Loss construction stays on CPU; fully supported GPU graphs only
           read back their declared logits activations here. */
        if (vk_sync_host_tensor(logits) != 0) { rc = -1; break; }
        int seed_rc = seed_cross_entropy_grad(grads, logits, losses[i].targets,
            losses[i].target_count, losses[i].ignore_index, losses[i].row_index,
            losses[i].weight, losses[i].normalizer, &current_metrics[i].loss,
            &current_metrics[i].correct, &current_metrics[i].examples,
            &current_metrics[i].normalizer);
        if (seed_rc < 0) { rc = -1; break; }
        if (current_metrics[i].examples > 0 && losses[i].weight > 0.0f)
            current_has_contribution = 1;
        current_loss += current_metrics[i].loss;
    }

    if (rc == 0 && current_has_contribution &&
        training_graph.gpu_training_backend != NATIVE_GPU_TRAINING_NONE) {
        GpuTrainingPlan plan = {0};
        if (gpu_build_backward_plan(&plan, grads) == 0) {
            int gpu_rc = gpu_execute_backward_plan(training_graph.gpu_training_backend, &plan, grads,
                                                   trainable_names, trainable_count);
            if (gpu_rc > 0) {
                if (g_required_training_backend != NATIVE_GPU_TRAINING_NONE) {
                    rc = -1;
                } else {
                    if (gpu_training_sync_all_f32_tensors() != 0 ||
                        backward_supported_graph(grads) != 0) rc = -1;
                }
            } else if (gpu_rc < 0) rc = -1;
            else g_last_training_backend = training_graph.gpu_training_backend;
        } else {
            if (g_required_training_backend != NATIVE_GPU_TRAINING_NONE) {
                rc = -1;
            } else {
                if (gpu_training_sync_all_f32_tensors() != 0 ||
                    backward_supported_graph(grads) != 0) rc = -1;
            }
        }
        gpu_training_plan_free(&plan);
    } else if (rc == 0 && current_has_contribution && backward_supported_graph(grads) != 0) {
        rc = -1;
    }

    if (rc == 0 && current_has_contribution) {
        for (int i = 0; i < trainable_count; i++) {
            T* tensor = t_find(trainable_names[i]);
            float* gradient = grad_existing(grads, tensor);
            if (!tensor || !gradient || tensor->numel <= 0) { rc = -1; break; }
            for (long j = 0; j < tensor->numel; j++) {
                if (!isfinite(gradient[j])) { rc = -1; break; }
            }
            if (rc != 0) break;
        }
    }
    if (training_graph_restore(&training_graph) != 0) rc = -1;
    if (rc != 0) goto done;

    if (training_accumulation_begin(losses, loss_count, trainable_names, trainable_count,
            update_mode, learning_rate, beta1, beta2, epsilon, weight_decay, max_grad_norm,
            step, accumulation_steps, topology_hash,
            NATIVE_GPU_TRAINING_NONE) != 0) { rc = -1; goto done; }
    TrainingAccumulationState* accumulation = &g_training_accumulation;
    for (int i = 0; i < trainable_count; i++) {
        if (!current_has_contribution) continue;
        T* tensor = t_find(trainable_names[i]);
        float* gradient = grad_existing(grads, tensor);
        for (long j = 0; j < tensor->numel; j++) accumulation->gradients[i][j] += gradient[j];
    }
    for (int i = 0; i < loss_count; i++) {
        accumulation->losses[i].metric.loss += current_metrics[i].loss;
        accumulation->losses[i].metric.correct += current_metrics[i].correct;
        accumulation->losses[i].metric.examples += current_metrics[i].examples;
        accumulation->losses[i].metric.normalizer = losses[i].normalizer > 0.0f
            ? losses[i].normalizer
            : (float)accumulation->losses[i].metric.examples;
    }
    accumulation->loss += current_loss;
    accumulation->has_contribution |= current_has_contribution;
    accumulation->microbatches++;

    int apply = flush_accumulation || accumulation->microbatches >= accumulation_steps;
    if (!apply) {
        training_accumulation_outputs(out_loss, out_metrics, out_accumulated_microbatches,
                                      out_update_applied, 0);
        goto done;
    }
    if (!accumulation->has_contribution) {
        training_accumulation_outputs(out_loss, out_metrics, out_accumulated_microbatches,
                                      out_update_applied, 0);
        volvoxai_engine_reset_training_accumulation();
        goto done;
    }

    /* Allocate/validate every optimizer state and gradient before the first
       parameter mutation. Clipping is one norm across the full accumulated set. */
    for (int i = 0; i < trainable_count; i++) {
        if (volvoxai_engine_prepare_tensor_update_f32(accumulation->trainable_names[i],
                accumulation->trainable_numel[i], update_mode) != 0) { rc = -1; break; }
    }
    if (rc == 0 && max_grad_norm > 0.0f) {
        double sum_squares = 0.0;
        for (int i = 0; i < trainable_count; i++) {
            for (long j = 0; j < accumulation->trainable_numel[i]; j++) {
                double value = accumulation->gradients[i][j];
                sum_squares += value * value;
            }
        }
        double norm = sqrt(sum_squares);
        if (norm > (double)max_grad_norm) {
            float scale = (float)((double)max_grad_norm / (norm + 1.0e-12));
            for (int i = 0; i < trainable_count; i++) {
                for (long j = 0; j < accumulation->trainable_numel[i]; j++)
                    accumulation->gradients[i][j] *= scale;
            }
        }
    }
    if (rc == 0) {
        for (int i = 0; i < trainable_count; i++) {
            if (volvoxai_engine_apply_tensor_update_f32_locked(accumulation->trainable_names[i],
                    accumulation->gradients[i], accumulation->trainable_numel[i], update_mode,
                    learning_rate, beta1, beta2, epsilon, weight_decay, 0.0f, step) != 0) {
                rc = -1;
                break;
            }
        }
    }
    if (rc == 0) {
        training_accumulation_outputs(out_loss, out_metrics, out_accumulated_microbatches,
                                      out_update_applied, 1);
    }
    /* A failed apply may already have changed an earlier tensor, so it cannot
       safely retain/retry the window. */
    volvoxai_engine_reset_training_accumulation();

done:
    for (int i = 0; i < g_nt; i++) free(grads[i]);
    free(grads);
    free(current_metrics);
    return rc;
}

int volvoxai_engine_train_step_multi(
        const volvoxai_cross_entropy_loss_t* losses, int loss_count,
        const char* const* trainable_names, int trainable_count,
        int update_mode, float learning_rate, float beta1, float beta2,
        float epsilon, float weight_decay, float max_grad_norm, long step,
        int accumulation_steps, int flush_accumulation, int reset_accumulation,
        float* out_loss, volvoxai_cross_entropy_metric_t* out_metrics,
        int* out_accumulated_microbatches, int* out_update_applied) {
    if (volvoxai_engine_model_route_lease_active()) return -1;
    volvoxai_engine_model_lock();
    int rc = volvoxai_engine_train_step_multi_locked_impl(losses, loss_count,
        trainable_names, trainable_count, update_mode, learning_rate, beta1, beta2,
        epsilon, weight_decay, max_grad_norm, step, accumulation_steps,
        flush_accumulation, reset_accumulation, out_loss, out_metrics,
        out_accumulated_microbatches, out_update_applied);
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_engine_train_step(const char* logits_name, const int* targets, int target_count,
                              int ignore_index,
                              const char* const* trainable_names, int trainable_count,
                              int update_mode, float learning_rate, float beta1, float beta2,
                              float epsilon, float weight_decay, float max_grad_norm,
                              long step, float* out_loss, int* out_correct, int* out_examples) {
    volvoxai_cross_entropy_loss_t loss = {
        "loss", logits_name, targets, target_count, ignore_index, g_execution_row, 1.0f, 0.0f
    };
    volvoxai_cross_entropy_metric_t metric = {0};
    int microbatches = 0, update_applied = 0;
    int rc = volvoxai_engine_train_step_multi(&loss, 1, trainable_names, trainable_count,
        update_mode, learning_rate, beta1, beta2, epsilon, weight_decay, max_grad_norm,
        step, 1, 0, 0, out_loss, &metric, &microbatches, &update_applied);
    if (out_correct) *out_correct = metric.correct;
    if (out_examples) *out_examples = metric.examples;
    return rc;
}

int volvoxai_engine_lora_train_step(const char* logits_name, const int* targets, int target_count,
                                    int ignore_index,
                                    const char* const* trainable_names, int trainable_count,
                                    int update_mode, float learning_rate, float beta1, float beta2,
                                    float epsilon, float weight_decay, float max_grad_norm,
                                    long step, float* out_loss, int* out_correct, int* out_examples) {
    return volvoxai_engine_train_step(logits_name, targets, target_count, ignore_index,
                                      trainable_names, trainable_count, update_mode,
                                      learning_rate, beta1, beta2, epsilon, weight_decay,
                                      max_grad_norm, step, out_loss, out_correct, out_examples);
}
