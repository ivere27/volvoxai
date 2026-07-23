/*
 * Host-side validation and dispatch for the manual BatchNorm2D, GroupNorm,
 * and PReLU backward CUDA kernels. This file is included by cuda_engine.c
 * after cuda_training_product_bytes has been defined.
 */
#ifndef VOLVOXAI_CUDA_TRAINING_NORM_EXTRA_HOST_INC
#define VOLVOXAI_CUDA_TRAINING_NORM_EXTRA_HOST_INC

static int cuda_training_norm_extra_supports(
        CudaTrainingCommand command, const size_t* bytes, int binding_count,
        uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    int is_batchnorm =
        command == CUDA_TRAINING_COMMAND_BATCHNORM2D_INPUT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BATCHNORM2D_PARAM_BACKWARD;
    int is_groupnorm =
        command == CUDA_TRAINING_COMMAND_GROUPNORM_INPUT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_GROUPNORM_PARAM_BACKWARD ||
        command ==
            CUDA_TRAINING_COMMAND_GROUPNORM_PARAM_COOPERATIVE_BACKWARD;
    int is_prelu =
        command == CUDA_TRAINING_COMMAND_PRELU_INPUT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_PRELU_WEIGHT_BACKWARD;

    if (!bytes || !groups_x || groups_x > (uint32_t)INT_MAX ||
        groups_y != 1u || groups_z != 1u) return 0;
    if (is_batchnorm) {
        if (binding_count != 9 || bytes[8] != 4u * sizeof(uint32_t) ||
            bytes[0] != bytes[4] || bytes[0] != bytes[5] ||
            bytes[1] != bytes[6]) return 0;
        for (int binding = 0; binding < 8; ++binding) {
            if (!bytes[binding] || bytes[binding] % sizeof(float) != 0u)
                return 0;
        }
        return 1;
    }
    if (is_groupnorm) {
        size_t channel_elements;
        if (binding_count != 7 || bytes[6] != 8u * sizeof(uint32_t) ||
            bytes[0] != bytes[2] || bytes[0] != bytes[3] ||
            bytes[1] != bytes[4]) return 0;
        for (int binding = 0; binding < 6; ++binding) {
            if (!bytes[binding] || bytes[binding] % sizeof(float) != 0u)
                return 0;
        }
        channel_elements = bytes[1] / sizeof(float);
        if (!channel_elements || channel_elements > UINT32_MAX) return 0;
        if (command ==
                CUDA_TRAINING_COMMAND_GROUPNORM_PARAM_COOPERATIVE_BACKWARD)
            return groups_x == (uint32_t)channel_elements;
        if (command == CUDA_TRAINING_COMMAND_GROUPNORM_PARAM_BACKWARD)
            return groups_x == (uint32_t)(
                (channel_elements + CUDA_TRAINING_WORKGROUP_SIZE - 1u) /
                CUDA_TRAINING_WORKGROUP_SIZE);
        return 1;
    }
    if (is_prelu) {
        if (binding_count != 6 || bytes[5] != 4u * sizeof(uint32_t) ||
            bytes[0] != bytes[2] || bytes[0] != bytes[3] ||
            bytes[1] != bytes[4]) return 0;
        for (int binding = 0; binding < 5; ++binding) {
            if (!bytes[binding] || bytes[binding] % sizeof(float) != 0u)
                return 0;
        }
        return 1;
    }
    return 0;
}

static int cuda_training_dispatch_norm_extra(
        CudaTrainingCommand command, void* const* hosts,
        const size_t* bytes, const unsigned char* access,
        const unsigned char* is_weight, uint32_t groups_x,
        uint32_t groups_y, uint32_t groups_z) {
    if (!hosts || !bytes || !access || !is_weight ||
        groups_x > (uint32_t)INT_MAX) return -1;

    if (command == CUDA_TRAINING_COMMAND_BATCHNORM2D_INPUT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BATCHNORM2D_PARAM_BACKWARD) {
        uint32_t params[4];
        uint32_t factors[1];
        uint32_t tensor_elements;
        uint32_t channel_elements;
        uint32_t expected_groups;
        size_t tensor_bytes;
        size_t minimum_channel_bytes;
        CudaTensorView views[9];
        CUdeviceptr devices[9];
        CUfunction function;
        void* args[9];

        for (int binding = 0; binding < 9; ++binding)
            if (!hosts[binding]) return -1;
        if (access[0] != CUDA_TRAINING_READ ||
            access[1] != CUDA_TRAINING_READ ||
            access[2] != CUDA_TRAINING_READ ||
            access[3] != CUDA_TRAINING_READ ||
            access[4] != CUDA_TRAINING_READ ||
            access[5] != CUDA_TRAINING_READ_WRITE ||
            access[6] != CUDA_TRAINING_READ_WRITE ||
            access[7] != CUDA_TRAINING_READ_WRITE ||
            access[8] != CUDA_TRAINING_READ || is_weight[8] != 0u)
            return -1;
        for (int binding = 0; binding < 8; ++binding) {
            if (is_weight[binding] > 1u || !bytes[binding] ||
                bytes[binding] % sizeof(float) != 0u) return -1;
        }
        if (bytes[8] != sizeof(params)) return -1;

        memcpy(params, hosts[8], sizeof(params));
        if (!params[0] || !params[1] || params[0] % params[1] != 0u ||
            params[3] != 0u) return -1;
        factors[0] = params[0];
        if (!cuda_training_product_bytes(factors, 1, sizeof(float),
                                         &tensor_elements, &tensor_bytes))
            return -1;
        factors[0] = params[1];
        if (!cuda_training_product_bytes(
                factors, 1, sizeof(float), &channel_elements,
                &minimum_channel_bytes)) return -1;
        if (bytes[0] != tensor_bytes || bytes[4] != tensor_bytes ||
            bytes[5] != tensor_bytes || bytes[1] != bytes[6] ||
            bytes[1] < minimum_channel_bytes ||
            bytes[2] < minimum_channel_bytes ||
            bytes[3] < minimum_channel_bytes ||
            bytes[7] < minimum_channel_bytes) return -1;

        if (command == CUDA_TRAINING_COMMAND_BATCHNORM2D_INPUT_BACKWARD) {
            expected_groups = tensor_elements / CUDA_TRAINING_WORKGROUP_SIZE +
                (tensor_elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
            function = CUDA_TRAINING_FUNCTION(k_training_batchnorm2d_backward_input);
        } else {
            expected_groups = channel_elements /
                CUDA_TRAINING_WORKGROUP_SIZE +
                (channel_elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
            function = CUDA_TRAINING_FUNCTION(k_training_batchnorm2d_backward_param);
        }
        if (groups_x != expected_groups || groups_y != 1u || groups_z != 1u)
            return -1;

        memset(views, 0, sizeof(views));
        memset(devices, 0, sizeof(devices));
        if (command == CUDA_TRAINING_COMMAND_BATCHNORM2D_INPUT_BACKWARD) {
            const int required[] = {1, 3, 4, 5, 8};
            for (int index = 0; index < 5; ++index) {
                int binding = required[index];
                if (!graph_ensure_input_view(
                        hosts[binding], bytes[binding],
                        binding == 8 ? 0 : is_weight[binding],
                        &views[binding])) return -1;
                devices[binding] = views[binding].device;
            }
        } else {
            const int required[] = {0, 2, 3, 4, 6, 7, 8};
            for (int index = 0; index < 7; ++index) {
                int binding = required[index];
                if (!graph_ensure_input_view(
                        hosts[binding], bytes[binding],
                        binding == 8 ? 0 : is_weight[binding],
                        &views[binding])) return -1;
                devices[binding] = views[binding].device;
            }
        }
        for (int binding = 0; binding < 9; ++binding)
            args[binding] = &devices[binding];
        if (!cuda_launch(function, groups_x, 1u, 1u,
                         CUDA_TRAINING_WORKGROUP_SIZE, 1u, 1u, 0u, args))
            return -1;
        if (command == CUDA_TRAINING_COMMAND_BATCHNORM2D_INPUT_BACKWARD) {
            graph_mark_device(views[5].slot);
        } else {
            graph_mark_device(views[6].slot);
            graph_mark_device(views[7].slot);
        }
        return 0;
    }

    if (command == CUDA_TRAINING_COMMAND_GROUPNORM_INPUT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_GROUPNORM_PARAM_BACKWARD ||
        command ==
            CUDA_TRAINING_COMMAND_GROUPNORM_PARAM_COOPERATIVE_BACKWARD) {
        uint32_t params[8];
        uint32_t tensor_factors[4];
        uint32_t group_factors[2];
        uint32_t channel_factor[1];
        uint32_t tensor_elements;
        uint32_t total_groups;
        uint32_t channel_elements;
        uint32_t expected_groups;
        float epsilon;
        size_t tensor_bytes;
        size_t group_bytes;
        size_t channel_bytes;
        CudaTensorView views[7];
        CUdeviceptr devices[7];
        CUfunction function;
        void* args[7];

        for (int binding = 0; binding < 7; ++binding)
            if (!hosts[binding]) return -1;
        if (access[0] != CUDA_TRAINING_READ ||
            access[1] != CUDA_TRAINING_READ ||
            access[2] != CUDA_TRAINING_READ ||
            access[3] != CUDA_TRAINING_READ_WRITE ||
            access[4] != CUDA_TRAINING_READ_WRITE ||
            access[5] != CUDA_TRAINING_READ_WRITE ||
            access[6] != CUDA_TRAINING_READ || is_weight[6] != 0u)
            return -1;
        for (int binding = 0; binding < 6; ++binding) {
            if (is_weight[binding] > 1u || !bytes[binding] ||
                bytes[binding] % sizeof(float) != 0u) return -1;
        }
        if (bytes[6] != sizeof(params)) return -1;

        memcpy(params, hosts[6], sizeof(params));
        memcpy(&epsilon, &params[6], sizeof(epsilon));
        if (!params[0] || !params[1] || !params[2] || !params[3] ||
            !params[4] || params[3] % params[4] != 0u ||
            params[5] > 1u || !(epsilon > 0.0f) || !isfinite(epsilon) ||
            params[7] != 0u) return -1;

        tensor_factors[0] = params[0];
        tensor_factors[1] = params[1];
        tensor_factors[2] = params[2];
        tensor_factors[3] = params[3];
        group_factors[0] = params[0];
        group_factors[1] = params[4];
        channel_factor[0] = params[3];
        if (!cuda_training_product_bytes(
                tensor_factors, 4, sizeof(float), &tensor_elements,
                &tensor_bytes) ||
            !cuda_training_product_bytes(
                group_factors, 2, 1u, &total_groups, &group_bytes) ||
            !cuda_training_product_bytes(
                channel_factor, 1, sizeof(float), &channel_elements,
                &channel_bytes)) return -1;
        if (bytes[0] != tensor_bytes || bytes[2] != tensor_bytes ||
            bytes[3] != tensor_bytes || bytes[1] != channel_bytes ||
            bytes[4] != channel_bytes ||
            bytes[5] != (params[5] ? channel_bytes
                                   : 4u * sizeof(uint32_t))) return -1;

        if (command == CUDA_TRAINING_COMMAND_GROUPNORM_INPUT_BACKWARD) {
            /* One CUDA block cooperatively owns each sample/group. */
            expected_groups = total_groups;
            function = CUDA_TRAINING_FUNCTION(k_training_groupnorm_backward_input);
        } else if (command ==
                   CUDA_TRAINING_COMMAND_GROUPNORM_PARAM_COOPERATIVE_BACKWARD) {
            expected_groups = channel_elements;
            function = CUDA_TRAINING_FUNCTION(k_training_groupnorm_backward_param_cooperative);
        } else {
            expected_groups = channel_elements /
                CUDA_TRAINING_WORKGROUP_SIZE +
                (channel_elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
            function = CUDA_TRAINING_FUNCTION(k_training_groupnorm_backward_param);
        }
        if (groups_x != expected_groups || groups_y != 1u || groups_z != 1u)
            return -1;

        memset(views, 0, sizeof(views));
        memset(devices, 0, sizeof(devices));
        if (command == CUDA_TRAINING_COMMAND_GROUPNORM_INPUT_BACKWARD) {
            const int required[] = {0, 1, 2, 3, 6};
            for (int index = 0; index < 5; ++index) {
                int binding = required[index];
                if (!graph_ensure_input_view(
                        hosts[binding], bytes[binding],
                        binding == 6 ? 0 : is_weight[binding],
                        &views[binding])) return -1;
                devices[binding] = views[binding].device;
            }
        } else {
            const int required[] = {0, 2, 4, 6};
            for (int index = 0; index < 4; ++index) {
                int binding = required[index];
                if (!graph_ensure_input_view(
                        hosts[binding], bytes[binding],
                        binding == 6 ? 0 : is_weight[binding],
                        &views[binding])) return -1;
                devices[binding] = views[binding].device;
            }
            if (params[5]) {
                if (!graph_ensure_input_view(
                        hosts[5], bytes[5], is_weight[5], &views[5]))
                    return -1;
                devices[5] = views[5].device;
            }
        }
        for (int binding = 0; binding < 7; ++binding)
            args[binding] = &devices[binding];
        if (!cuda_launch(function, groups_x, 1u, 1u,
                         CUDA_TRAINING_WORKGROUP_SIZE, 1u, 1u, 0u, args))
            return -1;
        if (command == CUDA_TRAINING_COMMAND_GROUPNORM_INPUT_BACKWARD) {
            graph_mark_device(views[3].slot);
        } else {
            graph_mark_device(views[4].slot);
            if (params[5]) graph_mark_device(views[5].slot);
        }
        (void)tensor_elements;
        (void)group_bytes;
        return 0;
    }

    if (command == CUDA_TRAINING_COMMAND_PRELU_INPUT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_PRELU_WEIGHT_BACKWARD) {
        uint32_t params[4];
        uint32_t tensor_factor[1];
        uint32_t weight_factor[1];
        uint32_t tensor_elements;
        uint32_t weight_elements;
        uint32_t expected_groups;
        size_t tensor_bytes;
        size_t weight_bytes;
        CudaTensorView views[6];
        CUdeviceptr devices[6];
        CUfunction function;
        void* args[6];

        for (int binding = 0; binding < 6; ++binding)
            if (!hosts[binding]) return -1;
        if (access[0] != CUDA_TRAINING_READ ||
            access[1] != CUDA_TRAINING_READ ||
            access[2] != CUDA_TRAINING_READ ||
            access[3] != CUDA_TRAINING_READ_WRITE ||
            access[4] != CUDA_TRAINING_READ_WRITE ||
            access[5] != CUDA_TRAINING_READ || is_weight[5] != 0u)
            return -1;
        for (int binding = 0; binding < 5; ++binding) {
            if (is_weight[binding] > 1u || !bytes[binding] ||
                bytes[binding] % sizeof(float) != 0u) return -1;
        }
        if (bytes[5] != sizeof(params)) return -1;

        memcpy(params, hosts[5], sizeof(params));
        if (!params[0] || !params[1] || !params[2] ||
            params[0] % params[1] != 0u ||
            (params[2] != 1u && params[2] != params[1]) ||
            params[3] != 0u) return -1;
        tensor_factor[0] = params[0];
        weight_factor[0] = params[2];
        if (!cuda_training_product_bytes(
                tensor_factor, 1, sizeof(float), &tensor_elements,
                &tensor_bytes) ||
            !cuda_training_product_bytes(
                weight_factor, 1, sizeof(float), &weight_elements,
                &weight_bytes)) return -1;
        if (bytes[0] != tensor_bytes || bytes[2] != tensor_bytes ||
            bytes[3] != tensor_bytes || bytes[1] != weight_bytes ||
            bytes[4] != weight_bytes) return -1;

        if (command == CUDA_TRAINING_COMMAND_PRELU_INPUT_BACKWARD) {
            expected_groups = tensor_elements / CUDA_TRAINING_WORKGROUP_SIZE +
                (tensor_elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
            function = CUDA_TRAINING_FUNCTION(k_training_prelu_backward_input);
        } else {
            expected_groups = weight_elements / CUDA_TRAINING_WORKGROUP_SIZE +
                (weight_elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
            function = CUDA_TRAINING_FUNCTION(k_training_prelu_backward_weight);
        }
        if (groups_x != expected_groups || groups_y != 1u || groups_z != 1u)
            return -1;

        memset(views, 0, sizeof(views));
        memset(devices, 0, sizeof(devices));
        if (command == CUDA_TRAINING_COMMAND_PRELU_INPUT_BACKWARD) {
            const int required[] = {0, 1, 2, 3, 5};
            for (int index = 0; index < 5; ++index) {
                int binding = required[index];
                if (!graph_ensure_input_view(
                        hosts[binding], bytes[binding],
                        binding == 5 ? 0 : is_weight[binding],
                        &views[binding])) return -1;
                devices[binding] = views[binding].device;
            }
        } else {
            const int required[] = {0, 2, 4, 5};
            for (int index = 0; index < 4; ++index) {
                int binding = required[index];
                if (!graph_ensure_input_view(
                        hosts[binding], bytes[binding],
                        binding == 5 ? 0 : is_weight[binding],
                        &views[binding])) return -1;
                devices[binding] = views[binding].device;
            }
        }
        for (int binding = 0; binding < 6; ++binding)
            args[binding] = &devices[binding];
        if (!cuda_launch(function, groups_x, 1u, 1u,
                         CUDA_TRAINING_WORKGROUP_SIZE, 1u, 1u, 0u, args))
            return -1;
        if (command == CUDA_TRAINING_COMMAND_PRELU_INPUT_BACKWARD)
            graph_mark_device(views[3].slot);
        else
            graph_mark_device(views[4].slot);
        return 0;
    }

    return -1;
}

#endif
