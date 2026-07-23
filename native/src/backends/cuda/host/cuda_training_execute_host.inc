int cuda_training_dispatch(const char* shader_name, const char* entry_point,
                           void* const* hosts, const size_t* bytes,
                           const unsigned char* access,
                           const unsigned char* is_weight, int binding_count,
                           uint32_t groups_x, uint32_t groups_y,
                           uint32_t groups_z) {
    CudaTrainingCommand command = cuda_training_command(shader_name,
                                                        entry_point);
    CudaTensorView source;
    CudaTensorView destination;
    uint32_t params[8] = {0};
    uint32_t elements = 0;
    uint32_t threshold;
    uint32_t seed;
    uint32_t counter;
    float scale;
    CUfunction function;
    void* args[7];
    if (!cuda_training_active || !cuda_training_context_ready ||
        cuda_training_failed || !hosts || !access || !is_weight ||
        !cuda_training_supports(shader_name, entry_point, bytes,
                                binding_count, groups_x, groups_y, groups_z)) {
        cuda_training_failed = 1;
        return -1;
    }
    if (command >= CUDA_TRAINING_COMMAND_TRANSPOSE_BACKWARD &&
        command <= CUDA_TRAINING_COMMAND_POOLING_MAX_POOL_BACKWARD) {
        if (cuda_training_dispatch_shape(command, hosts, bytes, access,
                                         is_weight, groups_x, groups_y,
                                         groups_z) != 0) {
            cuda_training_failed = 1;
            return -1;
        }
        return 0;
    }
    if (command >= CUDA_TRAINING_COMMAND_BATCHNORM2D_INPUT_BACKWARD &&
        command <= CUDA_TRAINING_COMMAND_PRELU_WEIGHT_BACKWARD) {
        if (cuda_training_dispatch_norm_extra(command, hosts, bytes, access,
                                              is_weight, groups_x, groups_y,
                                              groups_z) != 0) {
            cuda_training_failed = 1;
            return -1;
        }
        return 0;
    }
    if (command >= CUDA_TRAINING_COMMAND_MOE_LINEAR_INPUT_BACKWARD &&
        command <= CUDA_TRAINING_COMMAND_MOE_ROUTER_BIAS_BACKWARD) {
        if (cuda_training_dispatch_moe(command, hosts, bytes, access,
                                       is_weight, groups_x, groups_y,
                                       groups_z) != 0) {
            cuda_training_failed = 1;
            return -1;
        }
        return 0;
    }
    if (command == CUDA_TRAINING_COMMAND_MATMUL_INPUT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_MATMUL_WEIGHT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_MATMUL_BIAS_BACKWARD) {
        if (cuda_training_dispatch_matmul(command, hosts, bytes, access,
                                          is_weight, groups_x, groups_y,
                                          groups_z) != 0) {
            cuda_training_failed = 1;
            return -1;
        }
        return 0;
    }
    if (command == CUDA_TRAINING_COMMAND_SDPA_BACKWARD) {
        if (cuda_training_dispatch_sdpa(hosts, bytes, access, is_weight,
                                        groups_x, groups_y, groups_z) != 0) {
            cuda_training_failed = 1;
            return -1;
        }
        return 0;
    }
    if (command == CUDA_TRAINING_COMMAND_CROSS_SDPA_Q_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_CROSS_SDPA_K_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_CROSS_SDPA_V_BACKWARD) {
        if (cuda_training_dispatch_cross_sdpa(command, hosts, bytes, access,
                                              is_weight, groups_x, groups_y,
                                              groups_z) != 0) {
            cuda_training_failed = 1;
            return -1;
        }
        return 0;
    }
    if (command == CUDA_TRAINING_COMMAND_BASIC_A_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BASIC_B_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BASIC_A_COOPERATIVE_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BASIC_B_COOPERATIVE_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BASIC_A_SCALAR_STAGED_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BASIC_B_SCALAR_STAGED_BACKWARD) {
        if (cuda_training_dispatch_basic(command, hosts, bytes, access,
                                         is_weight, groups_x, groups_y,
                                         groups_z) != 0) {
            cuda_training_failed = 1;
            return -1;
        }
        return 0;
    }
    if (command == CUDA_TRAINING_COMMAND_ACTIVATION_BACKWARD) {
        if (cuda_training_dispatch_activation(hosts, bytes, access, is_weight,
                                              groups_x, groups_y,
                                              groups_z) != 0) {
            cuda_training_failed = 1;
            return -1;
        }
        return 0;
    }
    if (command == CUDA_TRAINING_COMMAND_REDUCE_BACKWARD) {
        if (cuda_training_dispatch_reduce(hosts, bytes, access, is_weight,
                                          groups_x, groups_y, groups_z) != 0) {
            cuda_training_failed = 1;
            return -1;
        }
        return 0;
    }
    if (command == CUDA_TRAINING_COMMAND_SOFTMAX_BACKWARD) {
        if (cuda_training_dispatch_softmax(hosts, bytes, access, is_weight,
                                           groups_x, groups_y, groups_z) != 0) {
            cuda_training_failed = 1;
            return -1;
        }
        return 0;
    }
    if (command == CUDA_TRAINING_COMMAND_EMBEDDING_BACKWARD) {
        if (cuda_training_dispatch_embedding(hosts, bytes, access, is_weight,
                                             groups_x, groups_y,
                                             groups_z) != 0) {
            cuda_training_failed = 1;
            return -1;
        }
        return 0;
    }
    if (command == CUDA_TRAINING_COMMAND_LAYERNORM_INPUT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_LAYERNORM_PARAM_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_LAYERNORM_PARAM_COOPERATIVE_BACKWARD) {
        if (cuda_training_dispatch_layernorm(command, hosts, bytes, access,
                                             is_weight, groups_x, groups_y,
                                             groups_z) != 0) {
            cuda_training_failed = 1;
            return -1;
        }
        return 0;
    }
    if (command == CUDA_TRAINING_COMMAND_RMSNORM_INPUT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_RMSNORM_WEIGHT_BACKWARD) {
        if (cuda_training_dispatch_rmsnorm(command, hosts, bytes, access,
                                           is_weight, groups_x, groups_y,
                                           groups_z) != 0) {
            cuda_training_failed = 1;
            return -1;
        }
        return 0;
    }
    if (command == CUDA_TRAINING_COMMAND_CONV2D_INPUT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_CONV2D_WEIGHT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_CONV2D_BIAS_BACKWARD) {
        if (cuda_training_dispatch_conv2d(command, hosts, bytes, access,
                                          is_weight, groups_x, groups_y,
                                          groups_z) != 0) {
            cuda_training_failed = 1;
            return -1;
        }
        return 0;
    }
    if (!hosts[0] || !hosts[1] || !hosts[2] ||
        access[0] != CUDA_TRAINING_READ ||
        access[1] != CUDA_TRAINING_READ_WRITE ||
        access[2] != CUDA_TRAINING_READ || is_weight[0] > 1u ||
        is_weight[1] > 1u || is_weight[2] != 0u) {
        cuda_training_failed = 1;
        return -1;
    }
    memcpy(params, hosts[2], bytes[2]);
    elements = params[0];
    if (!elements || (size_t)elements != bytes[0] / sizeof(float) ||
        !graph_ensure_input_view(hosts[0], bytes[0], is_weight[0], &source) ||
        !graph_ensure_input_view(hosts[1], bytes[1], is_weight[1],
                                 &destination)) {
        cuda_training_failed = 1;
        return -1;
    }
    args[0] = &source.device;
    args[1] = &destination.device;
    args[2] = &elements;
    function = CUDA_TRAINING_FUNCTION(k_training_accumulate);
    if (command == CUDA_TRAINING_COMMAND_DROPOUT_BACKWARD) {
        threshold = params[1];
        seed = params[2];
        counter = params[3];
        memcpy(&scale, &params[4], sizeof(scale));
        if (!isfinite(scale) || scale < 1.0f || params[5] || params[6] ||
            params[7]) {
            cuda_training_failed = 1;
            return -1;
        }
        function = CUDA_TRAINING_FUNCTION(k_training_dropout_backward);
        args[3] = &threshold;
        args[4] = &seed;
        args[5] = &counter;
        args[6] = &scale;
    }
    if (!cuda_launch(function, groups_x, 1u, 1u,
                     CUDA_TRAINING_WORKGROUP_SIZE, 1u, 1u, 0u, args)) {
        cuda_training_failed = 1;
        return -1;
    }
    graph_mark_device(destination.slot);
    return 0;
}

int cuda_training_sync(void* host, size_t bytes) {
    if (!cuda_training_active || !cuda_training_context_ready ||
        cuda_training_failed || !host || !bytes) return -1;
    if (!cuda_graph_sync_host(host, bytes, 0)) {
        cuda_training_failed = 1;
        return -1;
    }
    return 0;
}

void cuda_training_mark_failed(void) {
    if (cuda_training_active) cuda_training_failed = 1;
}

void cuda_training_end(void) {
    int synchronized = 0;
    if (!cuda_training_active) return;
    if (cuda_training_context_ready && cuda_stream) {
        synchronized = cuda_ok(p_cuStreamSynchronize(cuda_stream),
                               "cuStreamSynchronize(training)");
        if (!synchronized) cuda_training_failed = 1;
    }
    cuda_profile_report_training_scope(synchronized, !cuda_training_failed);
    if (cuda_training_context_ready &&
        !cuda_context_leave(&cuda_training_context_guard))
        cuda_training_failed = 1;
    memset(&cuda_training_context_guard, 0,
           sizeof(cuda_training_context_guard));
    cuda_training_active = 0;
    cuda_training_context_ready = 0;
    cuda_training_failed = 0;
    /* Host gradient identities are freed after the tape, so discard every
     * transient slot before allocator address reuse. True model weights keep
     * their device allocation until explicit host invalidation or graph/model
     * reset. */
    cuda_graph_release_transients();
}
