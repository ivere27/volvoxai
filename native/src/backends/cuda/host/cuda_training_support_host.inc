static int cuda_training_shape_supports(
        CudaTrainingCommand command, const size_t* bytes, int binding_count,
        uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
static int cuda_training_dispatch_shape(
        CudaTrainingCommand command, void* const* hosts,
        const size_t* bytes, const unsigned char* access,
        const unsigned char* is_weight, uint32_t groups_x,
        uint32_t groups_y, uint32_t groups_z);
static int cuda_training_norm_extra_supports(
        CudaTrainingCommand command, const size_t* bytes, int binding_count,
        uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
static int cuda_training_dispatch_norm_extra(
        CudaTrainingCommand command, void* const* hosts,
        const size_t* bytes, const unsigned char* access,
        const unsigned char* is_weight, uint32_t groups_x,
        uint32_t groups_y, uint32_t groups_z);
static int cuda_training_moe_supports(
        CudaTrainingCommand command, const size_t* bytes, int binding_count,
        uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
static int cuda_training_dispatch_moe(
        CudaTrainingCommand command, void* const* hosts,
        const size_t* bytes, const unsigned char* access,
        const unsigned char* is_weight, uint32_t groups_x,
        uint32_t groups_y, uint32_t groups_z);

int cuda_training_supports(const char* shader_name, const char* entry_point,
                           const size_t* bytes, int binding_count,
                           uint32_t groups_x, uint32_t groups_y,
                           uint32_t groups_z) {
    CudaTrainingCommand command = cuda_training_command(shader_name,
                                                        entry_point);
    size_t elements;
    size_t params_bytes;
    uint64_t expected_groups;
    if (!cuda_training_available() || command == CUDA_TRAINING_COMMAND_NONE ||
        !bytes || !groups_x || !groups_y || !groups_z) return 0;
    if (command >= CUDA_TRAINING_COMMAND_TRANSPOSE_BACKWARD &&
        command <= CUDA_TRAINING_COMMAND_POOLING_MAX_POOL_BACKWARD)
        return cuda_training_shape_supports(command, bytes, binding_count,
                                            groups_x, groups_y, groups_z);
    if (command >= CUDA_TRAINING_COMMAND_BATCHNORM2D_INPUT_BACKWARD &&
        command <= CUDA_TRAINING_COMMAND_PRELU_WEIGHT_BACKWARD)
        return cuda_training_norm_extra_supports(
            command, bytes, binding_count, groups_x, groups_y, groups_z);
    if (command >= CUDA_TRAINING_COMMAND_MOE_LINEAR_INPUT_BACKWARD &&
        command <= CUDA_TRAINING_COMMAND_MOE_ROUTER_BIAS_BACKWARD)
        return cuda_training_moe_supports(command, bytes, binding_count,
                                          groups_x, groups_y, groups_z);
    if (command == CUDA_TRAINING_COMMAND_BASIC_A_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BASIC_B_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BASIC_A_COOPERATIVE_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BASIC_B_COOPERATIVE_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BASIC_A_SCALAR_STAGED_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BASIC_B_SCALAR_STAGED_BACKWARD) {
        size_t destination_elements;
        size_t output_elements;
        int a_command;
        int cooperative;
        int staged;
        if (binding_count != 6 || bytes[5] != 32u * sizeof(uint32_t) ||
            !bytes[0] || !bytes[1] || !bytes[2] || !bytes[3] || !bytes[4] ||
            bytes[0] != bytes[3] || bytes[1] != bytes[4] || groups_y != 1u ||
            groups_z != 1u)
            return 0;
        for (int binding = 0; binding < 5; binding++)
            if (bytes[binding] % sizeof(float) != 0) return 0;
        a_command = command == CUDA_TRAINING_COMMAND_BASIC_A_BACKWARD ||
            command == CUDA_TRAINING_COMMAND_BASIC_A_COOPERATIVE_BACKWARD ||
            command == CUDA_TRAINING_COMMAND_BASIC_A_SCALAR_STAGED_BACKWARD;
        cooperative =
            command == CUDA_TRAINING_COMMAND_BASIC_A_COOPERATIVE_BACKWARD ||
            command == CUDA_TRAINING_COMMAND_BASIC_B_COOPERATIVE_BACKWARD;
        staged =
            command == CUDA_TRAINING_COMMAND_BASIC_A_SCALAR_STAGED_BACKWARD ||
            command == CUDA_TRAINING_COMMAND_BASIC_B_SCALAR_STAGED_BACKWARD;
        destination_elements = bytes[a_command ? 3 : 4] / sizeof(float);
        output_elements = bytes[2] / sizeof(float);
        if (!destination_elements || destination_elements > UINT32_MAX ||
            !output_elements || output_elements > UINT32_MAX) return 0;
        if (staged) {
            expected_groups = (output_elements +
                CUDA_TRAINING_BASIC_SCALAR_ITEMS_PER_PARTIAL - 1u) /
                CUDA_TRAINING_BASIC_SCALAR_ITEMS_PER_PARTIAL;
            return destination_elements == 1u &&
                output_elements >
                    CUDA_TRAINING_BASIC_SCALAR_ITEMS_PER_PARTIAL &&
                expected_groups <= UINT32_MAX &&
                groups_x == (uint32_t)expected_groups;
        }
        if (cooperative)
            return destination_elements > 1u &&
                destination_elements <= 1024u &&
                destination_elements < output_elements &&
                groups_x == (uint32_t)destination_elements;
        expected_groups = (destination_elements +
            CUDA_TRAINING_WORKGROUP_SIZE - 1u) /
            CUDA_TRAINING_WORKGROUP_SIZE;
        return expected_groups <= UINT32_MAX &&
            groups_x == (uint32_t)expected_groups;
    }
    if (command == CUDA_TRAINING_COMMAND_ACTIVATION_BACKWARD) {
        if (binding_count != 5 || bytes[4] != 4u * sizeof(uint32_t) ||
            !bytes[0] || bytes[0] != bytes[1] || bytes[0] != bytes[2] ||
            bytes[0] != bytes[3] || bytes[0] % sizeof(float) != 0 ||
            groups_y != 1u || groups_z != 1u) return 0;
        return 1;
    }
    if (command == CUDA_TRAINING_COMMAND_REDUCE_BACKWARD) {
        return binding_count == 3 && bytes[2] == 4u * sizeof(uint32_t) &&
            bytes[0] && bytes[1] && bytes[0] % sizeof(float) == 0 &&
            bytes[1] % sizeof(float) == 0 && groups_y == 1u && groups_z == 1u;
    }
    if (command == CUDA_TRAINING_COMMAND_SOFTMAX_BACKWARD) {
        return binding_count == 4 && bytes[3] == 4u * sizeof(uint32_t) &&
            bytes[0] && bytes[0] == bytes[1] && bytes[0] == bytes[2] &&
            bytes[0] % sizeof(float) == 0 && groups_y == 1u && groups_z == 1u;
    }
    if (command == CUDA_TRAINING_COMMAND_EMBEDDING_BACKWARD) {
        if (binding_count != 4 || bytes[3] != 4u * sizeof(uint32_t) ||
            !bytes[0] || !bytes[1] || !bytes[2] || groups_y != 1u ||
            groups_z != 1u)
            return 0;
        return bytes[0] % sizeof(int32_t) == 0 &&
            bytes[1] % sizeof(float) == 0 &&
            bytes[2] % sizeof(float) == 0;
    }
    if (command == CUDA_TRAINING_COMMAND_LAYERNORM_INPUT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_LAYERNORM_PARAM_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_LAYERNORM_PARAM_COOPERATIVE_BACKWARD) {
        if (binding_count != 7 || bytes[6] != 4u * sizeof(uint32_t) ||
            !bytes[0] || !bytes[1] || !bytes[2] || !bytes[3] || !bytes[4] ||
            !bytes[5] || bytes[0] != bytes[2] || bytes[0] != bytes[3] ||
            bytes[1] != bytes[4] || groups_y != 1u || groups_z != 1u) return 0;
        for (int binding = 0; binding < 6; binding++)
            if (bytes[binding] % sizeof(float) != 0) return 0;
        if (command ==
                CUDA_TRAINING_COMMAND_LAYERNORM_PARAM_COOPERATIVE_BACKWARD &&
            (groups_x > (uint32_t)INT_MAX ||
             groups_x > bytes[1] / sizeof(float))) return 0;
        return 1;
    }
    if (command == CUDA_TRAINING_COMMAND_RMSNORM_INPUT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_RMSNORM_WEIGHT_BACKWARD) {
        if (binding_count != 6 || bytes[5] != 4u * sizeof(uint32_t) ||
            !bytes[0] || !bytes[1] || !bytes[2] || !bytes[3] || !bytes[4] ||
            bytes[0] != bytes[2] || bytes[0] != bytes[3] ||
            bytes[1] != bytes[4] || groups_y != 1u || groups_z != 1u) return 0;
        for (int binding = 0; binding < 5; binding++)
            if (bytes[binding] % sizeof(float) != 0) return 0;
        return 1;
    }
    if (command == CUDA_TRAINING_COMMAND_SDPA_BACKWARD) {
        if (binding_count != 5 || bytes[4] != 12u * sizeof(uint32_t) ||
            !bytes[0] || !bytes[1] || !bytes[2] || !bytes[3] ||
            bytes[0] != bytes[3]) return 0;
        for (int binding = 0; binding < 4; binding++)
            if (bytes[binding] % sizeof(uint32_t) != 0) return 0;
        return 1;
    }
    if (command == CUDA_TRAINING_COMMAND_CROSS_SDPA_Q_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_CROSS_SDPA_K_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_CROSS_SDPA_V_BACKWARD) {
        if (binding_count != 9 || bytes[8] != 16u * sizeof(uint32_t) ||
            !bytes[0] || !bytes[1] || !bytes[2] || !bytes[3] || !bytes[4] ||
            !bytes[5] || !bytes[6] || !bytes[7] ||
            bytes[0] != bytes[4] || bytes[0] != bytes[5] ||
            bytes[1] != bytes[2] || bytes[1] != bytes[6] ||
            bytes[1] != bytes[7]) return 0;
        for (int binding = 0; binding < 8; binding++)
            if (bytes[binding] % sizeof(uint32_t) != 0) return 0;
        return 1;
    }
    if (command == CUDA_TRAINING_COMMAND_CONV2D_INPUT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_CONV2D_WEIGHT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_CONV2D_BIAS_BACKWARD) {
        if (binding_count != 8 || bytes[7] != 20u * sizeof(uint32_t) ||
            !bytes[0] || !bytes[1] || !bytes[2] || !bytes[3] ||
            !bytes[4] || !bytes[5] || !bytes[6] ||
            bytes[0] != bytes[4] || bytes[1] != bytes[5] ||
            bytes[2] != bytes[3] || groups_y != 1u || groups_z != 1u)
            return 0;
        for (int binding = 0; binding < 7; binding++)
            if (bytes[binding] % sizeof(float) != 0) return 0;
        return 1;
    }
    if (command == CUDA_TRAINING_COMMAND_MATMUL_INPUT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_MATMUL_WEIGHT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_MATMUL_BIAS_BACKWARD) {
        if (binding_count != 7 || bytes[6] != 8u * sizeof(uint32_t) ||
            !bytes[0] || !bytes[1] || !bytes[2] || !bytes[3] || !bytes[4] ||
            !bytes[5] || bytes[0] != bytes[3] || bytes[1] != bytes[4])
            return 0;
        for (int binding = 0; binding < 6; binding++)
            if (bytes[binding] % sizeof(float) != 0) return 0;
        return groups_z == 1u &&
            (command != CUDA_TRAINING_COMMAND_MATMUL_BIAS_BACKWARD ||
             groups_y == 1u);
    }
    if (binding_count != 3 || !bytes[0] || bytes[0] != bytes[1] ||
        bytes[0] % sizeof(float) != 0 || groups_y != 1u || groups_z != 1u)
        return 0;
    params_bytes = command == CUDA_TRAINING_COMMAND_DROPOUT_BACKWARD
        ? 8u * sizeof(uint32_t) : 4u * sizeof(uint32_t);
    if (bytes[2] != params_bytes) return 0;
    elements = bytes[0] / sizeof(float);
    if (!elements || elements > UINT32_MAX) return 0;
    expected_groups = ((uint64_t)elements + CUDA_TRAINING_WORKGROUP_SIZE - 1u) /
        CUDA_TRAINING_WORKGROUP_SIZE;
    return expected_groups <= UINT32_MAX && groups_x == (uint32_t)expected_groups;
}

int cuda_training_preflight(const char* shader_name, const char* entry_point,
                            const size_t* bytes, int binding_count,
                            uint32_t groups_x, uint32_t groups_y,
                            uint32_t groups_z) {
    CudaTrainingCommand command = cuda_training_command(
        shader_name, entry_point);
    if (!cuda_training_supports(shader_name, entry_point, bytes,
                                binding_count, groups_x, groups_y, groups_z))
        return 0;
    if (command == CUDA_TRAINING_COMMAND_BASIC_A_SCALAR_STAGED_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BASIC_B_SCALAR_STAGED_BACKWARD) {
#if SIZE_MAX < UINT32_MAX
        if ((size_t)groups_x > SIZE_MAX / sizeof(float)) return 0;
#endif
        return cuda_training_basic_workspace_reserve(
            (size_t)groups_x * sizeof(float));
    }
    return 1;
}

int cuda_training_begin(void) {
    if (!cuda_training_available() || cuda_training_active ||
        !cuda_context_enter_owned(&cuda_training_context_guard)) return -1;
    cuda_training_active = 1;
    cuda_training_context_ready = 1;
    cuda_training_failed = 0;
    if (cuda_profile.enabled && !cuda_profile.scope_open)
        cuda_profile_begin_scope(0);
    /* A training pass changes the launch topology and may later mutate model
     * weights. Never retain an inference replay plan across this boundary. */
    cuda_replay_clear_plan_current(1);
    return 0;
}
