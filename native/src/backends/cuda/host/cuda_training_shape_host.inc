/* Host validation and Driver-API dispatch for the CUDA shape/pooling
 * backward kernels. Included by cuda_engine.c in the full profile only. */
#ifndef VOLVOXAI_CUDA_TRAINING_SHAPE_HOST_INC
#define VOLVOXAI_CUDA_TRAINING_SHAPE_HOST_INC

static int cuda_training_shape_command(CudaTrainingCommand command) {
    return command == CUDA_TRAINING_COMMAND_TRANSPOSE_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_CONCAT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_SPLIT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_RESIZE_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_POOLING_GLOBAL_AVERAGE_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_POOLING_MAX_POOL_BACKWARD;
}

static int cuda_training_shape_supports(
        CudaTrainingCommand command, const size_t* bytes, int binding_count,
        uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    size_t destination_bytes;
    size_t params_bytes;
    uint64_t elements;
    uint64_t expected_groups;
    if (!cuda_training_shape_command(command) || !bytes ||
        groups_y != 1u || groups_z != 1u) return 0;
    if (command == CUDA_TRAINING_COMMAND_POOLING_GLOBAL_AVERAGE_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_POOLING_MAX_POOL_BACKWARD) {
        if (binding_count != 4 || bytes[3] != 12u * sizeof(uint32_t) ||
            !bytes[0] || !bytes[1] || !bytes[2] || bytes[0] != bytes[2] ||
            bytes[0] % sizeof(float) || bytes[1] % sizeof(float)) return 0;
        destination_bytes = bytes[2];
    } else {
        if (binding_count != 3 || !bytes[0] || !bytes[1] ||
            bytes[0] % sizeof(float) || bytes[1] % sizeof(float)) return 0;
        if (command == CUDA_TRAINING_COMMAND_TRANSPOSE_BACKWARD) {
            params_bytes = 34u * sizeof(uint32_t);
            destination_bytes = bytes[1];
            if (bytes[0] != bytes[1]) return 0;
        } else if (command == CUDA_TRAINING_COMMAND_CONCAT_BACKWARD) {
            params_bytes = 8u * sizeof(uint32_t);
            destination_bytes = bytes[1];
            if (bytes[0] < bytes[1]) return 0;
        } else if (command == CUDA_TRAINING_COMMAND_SPLIT_BACKWARD) {
            params_bytes = 5u * sizeof(uint32_t);
            destination_bytes = bytes[0];
            if (bytes[1] < bytes[0]) return 0;
        } else {
            params_bytes = 8u * sizeof(uint32_t);
            destination_bytes = bytes[1];
        }
        if (bytes[2] != params_bytes) return 0;
    }
    elements = destination_bytes / sizeof(float);
    if (!elements || elements > UINT32_MAX) return 0;
    expected_groups = (elements + CUDA_TRAINING_WORKGROUP_SIZE - 1u) /
        CUDA_TRAINING_WORKGROUP_SIZE;
    return expected_groups <= UINT32_MAX &&
        groups_x == (uint32_t)expected_groups;
}

static int cuda_training_shape_validate_transpose(
        const uint32_t params[34], const size_t* bytes) {
    uint32_t rank = params[0];
    uint32_t total = params[1];
    uint32_t seen = 0u;
    uint64_t input_product = 1u;
    uint64_t output_product = 1u;
    if (!rank || rank > 8u || !total) return 0;
    for (int dimension = (int)rank - 1; dimension >= 0; dimension--) {
        uint32_t shape = params[2u + (uint32_t)dimension];
        if (!shape || input_product > UINT32_MAX / shape ||
            params[18u + (uint32_t)dimension] != input_product) return 0;
        input_product *= shape;
    }
    for (uint32_t dimension = 0; dimension < rank; dimension++) {
        uint32_t source = params[10u + dimension];
        if (source >= rank || (seen & (1u << source))) return 0;
        seen |= 1u << source;
    }
    for (int dimension = (int)rank - 1; dimension >= 0; dimension--) {
        uint32_t source = params[10u + (uint32_t)dimension];
        uint32_t shape = params[2u + source];
        if (output_product > UINT32_MAX / shape ||
            params[26u + (uint32_t)dimension] != output_product) return 0;
        output_product *= shape;
    }
    for (uint32_t dimension = rank; dimension < 8u; dimension++) {
        if (params[2u + dimension] != 1u || params[10u + dimension] ||
            params[18u + dimension] || params[26u + dimension]) return 0;
    }
    return input_product == total && output_product == total &&
        bytes[0] == (size_t)total * sizeof(float) && bytes[1] == bytes[0];
}

static int cuda_training_shape_validate_concat(
        const uint32_t params[8], const size_t* bytes) {
    uint64_t block;
    uint64_t outer;
    uint64_t output_elements;
    if (!params[0] || !params[2] || !params[3] || !params[4] ||
        params[2] > params[3] || params[1] > params[3] - params[2] ||
        params[5] || params[6] || params[7]) return 0;
    block = (uint64_t)params[2] * params[4];
    if (!block || block > UINT32_MAX || params[0] % block) return 0;
    outer = params[0] / block;
    output_elements = outer * params[3] * params[4];
    return output_elements && output_elements <= UINT32_MAX &&
        bytes[0] == (size_t)output_elements * sizeof(float) &&
        bytes[1] == (size_t)params[0] * sizeof(float);
}

static int cuda_training_shape_validate_split(
        const uint32_t params[5], const size_t* bytes) {
    uint64_t block;
    uint64_t outer;
    uint64_t input_elements;
    if (!params[0] || !params[1] || !params[2] || !params[3] ||
        params[2] > params[3] || params[4] > params[3] - params[2]) return 0;
    block = (uint64_t)params[2] * params[1];
    if (!block || block > UINT32_MAX || params[0] % block) return 0;
    outer = params[0] / block;
    input_elements = outer * params[3] * params[1];
    return input_elements && input_elements <= UINT32_MAX &&
        bytes[0] == (size_t)params[0] * sizeof(float) &&
        bytes[1] == (size_t)input_elements * sizeof(float);
}

static int cuda_training_shape_validate_resize(
        const uint32_t params[8], const size_t* bytes) {
    uint32_t input_elements;
    uint32_t output_elements;
    size_t input_bytes;
    size_t output_bytes;
    uint32_t input_factors[4] = {
        params[0], params[1], params[2], params[3]
    };
    uint32_t output_factors[4] = {
        params[0], params[4], params[5], params[3]
    };
    if (params[6] > 1u || params[7] ||
        !cuda_training_product_bytes(input_factors, 4, sizeof(float),
                                     &input_elements, &input_bytes) ||
        !cuda_training_product_bytes(output_factors, 4, sizeof(float),
                                     &output_elements, &output_bytes)) return 0;
    (void)input_elements;
    (void)output_elements;
    return bytes[0] == output_bytes && bytes[1] == input_bytes;
}

static int cuda_training_shape_validate_pooling(
        CudaTrainingCommand command, const uint32_t params[12],
        const size_t* bytes) {
    uint32_t input_elements;
    uint32_t output_elements;
    uint32_t input_factors[4] = {
        params[0], params[1], params[2], params[3]
    };
    uint32_t output_factors[4];
    size_t input_bytes;
    size_t output_bytes;
    uint64_t max_y;
    uint64_t max_x;
    if (!cuda_training_product_bytes(input_factors, 4, sizeof(float),
                                     &input_elements, &input_bytes)) return 0;
    if (command == CUDA_TRAINING_COMMAND_POOLING_GLOBAL_AVERAGE_BACKWARD) {
        for (int index = 4; index < 12; index++)
            if (params[index]) return 0;
        output_factors[0] = params[0];
        output_factors[1] = params[3];
        if (!cuda_training_product_bytes(output_factors, 2, sizeof(float),
                                         &output_elements, &output_bytes))
            return 0;
    } else {
        for (int index = 0; index < 10; index++)
            if (!params[index]) return 0;
        for (int index = 0; index < 12; index++)
            if (params[index] > (uint32_t)INT_MAX) return 0;
        max_y = (uint64_t)(params[4] - 1u) * params[8] + params[6] - 1u;
        max_x = (uint64_t)(params[5] - 1u) * params[9] + params[7] - 1u;
        if (max_y > INT_MAX || max_x > INT_MAX) return 0;
        output_factors[0] = params[0];
        output_factors[1] = params[4];
        output_factors[2] = params[5];
        output_factors[3] = params[3];
        if (!cuda_training_product_bytes(output_factors, 4, sizeof(float),
                                         &output_elements, &output_bytes))
            return 0;
    }
    (void)input_elements;
    (void)output_elements;
    return bytes[0] == input_bytes && bytes[1] == output_bytes &&
        bytes[2] == input_bytes;
}

static int cuda_training_dispatch_shape(
        CudaTrainingCommand command, void* const* hosts,
        const size_t* bytes, const unsigned char* access,
        const unsigned char* is_weight, uint32_t groups_x,
        uint32_t groups_y, uint32_t groups_z) {
    uint32_t params[34];
    CudaTensorView views[4];
    CUdeviceptr devices[4];
    CUfunction function;
    int binding_count;
    int destination;
    int valid;
    void* args[4];
    if (!cuda_training_shape_supports(command, bytes,
                                      command ==
                                          CUDA_TRAINING_COMMAND_POOLING_GLOBAL_AVERAGE_BACKWARD ||
                                          command ==
                                          CUDA_TRAINING_COMMAND_POOLING_MAX_POOL_BACKWARD
                                          ? 4 : 3,
                                      groups_x, groups_y, groups_z)) return -1;
    binding_count = (command ==
                         CUDA_TRAINING_COMMAND_POOLING_GLOBAL_AVERAGE_BACKWARD ||
                     command == CUDA_TRAINING_COMMAND_POOLING_MAX_POOL_BACKWARD)
        ? 4 : 3;
    for (int binding = 0; binding < binding_count; binding++) {
        if (!hosts[binding] || is_weight[binding] > 1u) return -1;
        if (access[binding] !=
            (unsigned char)(binding == binding_count - 2
                ? CUDA_TRAINING_READ_WRITE : CUDA_TRAINING_READ)) return -1;
    }
    if (is_weight[binding_count - 1] != 0u) return -1;
    memset(params, 0, sizeof(params));
    memcpy(params, hosts[binding_count - 1], bytes[binding_count - 1]);
    if (command == CUDA_TRAINING_COMMAND_TRANSPOSE_BACKWARD)
        valid = cuda_training_shape_validate_transpose(params, bytes);
    else if (command == CUDA_TRAINING_COMMAND_CONCAT_BACKWARD)
        valid = cuda_training_shape_validate_concat(params, bytes);
    else if (command == CUDA_TRAINING_COMMAND_SPLIT_BACKWARD)
        valid = cuda_training_shape_validate_split(params, bytes);
    else if (command == CUDA_TRAINING_COMMAND_RESIZE_BACKWARD)
        valid = cuda_training_shape_validate_resize(params, bytes);
    else
        valid = cuda_training_shape_validate_pooling(command, params, bytes);
    if (!valid) return -1;

    memset(views, 0, sizeof(views));
    memset(devices, 0, sizeof(devices));
    for (int binding = 0; binding < binding_count; binding++) {
        if (!graph_ensure_input_view(hosts[binding], bytes[binding],
                                     binding == binding_count - 1
                                         ? 0 : is_weight[binding],
                                     &views[binding])) return -1;
        devices[binding] = views[binding].device;
        args[binding] = &devices[binding];
    }
    destination = binding_count - 2;
    if (command == CUDA_TRAINING_COMMAND_TRANSPOSE_BACKWARD)
        function = CUDA_TRAINING_FUNCTION(k_training_transpose_backward);
    else if (command == CUDA_TRAINING_COMMAND_CONCAT_BACKWARD)
        function = CUDA_TRAINING_FUNCTION(k_training_concat_backward);
    else if (command == CUDA_TRAINING_COMMAND_SPLIT_BACKWARD)
        function = CUDA_TRAINING_FUNCTION(k_training_split_backward);
    else if (command == CUDA_TRAINING_COMMAND_RESIZE_BACKWARD)
        function = CUDA_TRAINING_FUNCTION(k_training_resize_backward);
    else if (command == CUDA_TRAINING_COMMAND_POOLING_GLOBAL_AVERAGE_BACKWARD)
        function = CUDA_TRAINING_FUNCTION(k_training_pooling_backward_global_average);
    else
        function = CUDA_TRAINING_FUNCTION(k_training_pooling_backward_max_pool);
    if (!cuda_launch(function, groups_x, 1u, 1u,
                     CUDA_TRAINING_WORKGROUP_SIZE, 1u, 1u, 0u, args))
        return -1;
    graph_mark_device(views[destination].slot);
    return 0;
}

#endif
