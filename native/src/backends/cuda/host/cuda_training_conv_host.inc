static int cuda_training_dispatch_conv2d(
        CudaTrainingCommand command, void* const* hosts,
        const size_t* bytes, const unsigned char* access,
        const unsigned char* is_weight, uint32_t groups_x,
        uint32_t groups_y, uint32_t groups_z) {
    uint32_t params[20];
    uint32_t input_elements;
    uint32_t output_elements;
    uint32_t weight_elements;
    uint32_t input_factors[4];
    uint32_t output_factors[4];
    uint32_t weight_factors[4];
    uint32_t expected_groups;
    uint32_t destination_elements;
    uint32_t launch_groups_x;
    uint32_t launch_groups_y;
    uint32_t launch_block_x;
    uint32_t launch_block_y;
    size_t input_bytes;
    size_t output_bytes;
    size_t weight_bytes;
    size_t bias_bytes;
    uint64_t max_input_y_term;
    uint64_t max_input_x_term;
    uint64_t max_output_y_term;
    uint64_t max_output_x_term;
    CudaTensorView views[8];
    CUdeviceptr devices[8];
    CUfunction function;
    int destination;
    void* args[8];

    if (!hosts[0] || !hosts[1] || !hosts[2] || !hosts[3] || !hosts[4] ||
        !hosts[5] || !hosts[6] || !hosts[7] ||
        access[0] != CUDA_TRAINING_READ ||
        access[1] != CUDA_TRAINING_READ ||
        access[2] != CUDA_TRAINING_READ ||
        access[3] != CUDA_TRAINING_READ ||
        access[4] != CUDA_TRAINING_READ_WRITE ||
        access[5] != CUDA_TRAINING_READ_WRITE ||
        access[6] != CUDA_TRAINING_READ_WRITE ||
        access[7] != CUDA_TRAINING_READ || is_weight[7] != 0u) return -1;
    for (int binding = 0; binding < 7; binding++)
        if (is_weight[binding] > 1u) return -1;

    memcpy(params, hosts[7], sizeof(params));
    if (!params[0] || !params[1] || !params[2] || !params[3] ||
        !params[4] || !params[5] || !params[6] || !params[7] ||
        !params[8] || !params[9] || !params[10] || !params[13] ||
        !params[14] || !params[15] || params[16] > 2u ||
        params[17] > 1u || !params[18] || params[19] > 3u ||
        params[3] % params[13] != 0u || params[6] % params[13] != 0u ||
        ((params[19] == 1u || params[19] == 3u) &&
         (params[13] != params[3] || params[6] % params[3] != 0u)))
        return -1;
    for (int index = 0; index <= 15; index++)
        if (params[index] > (uint32_t)INT_MAX) return -1;

    input_factors[0] = params[0];
    input_factors[1] = params[1];
    input_factors[2] = params[2];
    input_factors[3] = params[3];
    output_factors[0] = params[0];
    output_factors[1] = params[4];
    output_factors[2] = params[5];
    output_factors[3] = params[6];
    weight_factors[0] = params[7];
    weight_factors[1] = params[8];
    if (params[13] == params[3]) {
        weight_factors[2] = params[3];
        weight_factors[3] = params[6] / params[3];
    } else {
        weight_factors[2] = params[3] / params[13];
        weight_factors[3] = params[6];
    }
    if (!cuda_training_product_bytes(input_factors, 4, sizeof(float),
                                     &input_elements, &input_bytes) ||
        !cuda_training_product_bytes(output_factors, 4, sizeof(float),
                                     &output_elements, &output_bytes) ||
        !cuda_training_product_bytes(weight_factors, 4, sizeof(float),
                                     &weight_elements, &weight_bytes) ||
        !size_product_u32(1u, params[6], sizeof(float), &bias_bytes) ||
        params[18] != weight_elements || bytes[0] != input_bytes ||
        bytes[1] != weight_bytes || bytes[2] != output_bytes ||
        bytes[3] != output_bytes || bytes[4] != input_bytes ||
        bytes[5] != weight_bytes ||
        bytes[6] != (params[17] ? bias_bytes : 4u * sizeof(uint32_t)))
        return -1;

    max_input_y_term = (uint64_t)(params[1] - 1u) + params[11];
    max_input_x_term = (uint64_t)(params[2] - 1u) + params[12];
    max_output_y_term = (uint64_t)(params[4] - 1u) * params[9] +
        (uint64_t)(params[7] - 1u) * params[14];
    max_output_x_term = (uint64_t)(params[5] - 1u) * params[10] +
        (uint64_t)(params[8] - 1u) * params[15];
    if (max_input_y_term > INT_MAX || max_input_x_term > INT_MAX ||
        max_output_y_term > INT_MAX || max_output_x_term > INT_MAX)
        return -1;

    if (command == CUDA_TRAINING_COMMAND_CONV2D_INPUT_BACKWARD) {
        destination = 4;
        destination_elements = input_elements;
        function = CUDA_TRAINING_FUNCTION(k_training_conv2d_backward_input);
    } else if (command == CUDA_TRAINING_COMMAND_CONV2D_WEIGHT_BACKWARD) {
        destination = 5;
        destination_elements = weight_elements;
        function = CUDA_TRAINING_FUNCTION(k_training_conv2d_backward_weight);
    } else {
        if (!params[17]) return -1;
        destination = 6;
        destination_elements = params[6];
        function = CUDA_TRAINING_FUNCTION(k_training_conv2d_backward_bias);
    }
    expected_groups = destination_elements / CUDA_TRAINING_WORKGROUP_SIZE +
        (destination_elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
    if (groups_x != expected_groups || groups_y != 1u || groups_z != 1u)
        return -1;

    launch_groups_x = groups_x;
    launch_groups_y = 1u;
    launch_block_x = CUDA_TRAINING_WORKGROUP_SIZE;
    launch_block_y = 1u;
    if (params[19] == 0u && params[13] == 1u &&
        params[7] == 3u && params[8] == 3u) {
        if (command == CUDA_TRAINING_COMMAND_CONV2D_INPUT_BACKWARD) {
            uint32_t input_rows = input_elements / params[3];
            uint32_t row_groups = input_rows / 16u +
                (input_rows % 16u != 0u);
            if (row_groups <= 65535u) {
                if (params[3] >= 32u) {
                    function =
                        CUDA_TRAINING_FUNCTION(k_training_conv2d_backward_input_hwio_3x3_tiled_c32);
                    launch_groups_x = params[3] / 32u +
                        (params[3] % 32u != 0u);
                } else {
                    function =
                        CUDA_TRAINING_FUNCTION(k_training_conv2d_backward_input_hwio_3x3_tiled);
                    launch_groups_x = params[3] / 16u +
                        (params[3] % 16u != 0u);
                }
                launch_groups_y = row_groups;
                launch_block_x = launch_block_y = 16u;
            }
        } else if (command ==
                   CUDA_TRAINING_COMMAND_CONV2D_WEIGHT_BACKWARD) {
            uint32_t patch_rows = weight_elements / params[6];
            uint32_t row_groups = patch_rows / 16u +
                (patch_rows % 16u != 0u);
            if (row_groups <= 65535u) {
                function = CUDA_TRAINING_FUNCTION(k_training_conv2d_backward_weight_hwio_3x3_tiled);
                launch_groups_x = params[6] / 16u +
                    (params[6] % 16u != 0u);
                launch_groups_y = row_groups;
                launch_block_x = launch_block_y = 16u;
            }
        }
    }

    memset(views, 0, sizeof(views));
    memset(devices, 0, sizeof(devices));
    if (command == CUDA_TRAINING_COMMAND_CONV2D_INPUT_BACKWARD) {
        int required[] = {1, 2, 3, 4, 7};
        for (int index = 0; index < 5; index++) {
            int binding = required[index];
            if (!graph_ensure_input_view(hosts[binding], bytes[binding],
                                         binding == 7 ? 0 : is_weight[binding],
                                         &views[binding])) return -1;
            devices[binding] = views[binding].device;
        }
    } else if (command == CUDA_TRAINING_COMMAND_CONV2D_WEIGHT_BACKWARD) {
        int required[] = {0, 2, 3, 5, 7};
        for (int index = 0; index < 5; index++) {
            int binding = required[index];
            if (!graph_ensure_input_view(hosts[binding], bytes[binding],
                                         binding == 7 ? 0 : is_weight[binding],
                                         &views[binding])) return -1;
            devices[binding] = views[binding].device;
        }
    } else {
        int required[] = {2, 3, 6, 7};
        for (int index = 0; index < 4; index++) {
            int binding = required[index];
            if (!graph_ensure_input_view(hosts[binding], bytes[binding],
                                         binding == 7 ? 0 : is_weight[binding],
                                         &views[binding])) return -1;
            devices[binding] = views[binding].device;
        }
    }
    for (int binding = 0; binding < 8; binding++) args[binding] = &devices[binding];
    if (!cuda_launch(function, launch_groups_x, launch_groups_y, 1u,
                     launch_block_x, launch_block_y, 1u, 0u, args))
        return -1;
    graph_mark_device(views[destination].slot);
    return 0;
}
