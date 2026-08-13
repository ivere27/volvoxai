/*
 * Host validation and CUDA Driver dispatch for the manual MoE training
 * kernels. cuda_engine.c includes this fragment after
 * cuda_training_product_bytes() has been defined.
 */
#ifndef VOLVOXAI_CUDA_TRAINING_MOE_HOST_INC
#define VOLVOXAI_CUDA_TRAINING_MOE_HOST_INC

static int cuda_training_moe_linear_command(CudaTrainingCommand command) {
    return command == CUDA_TRAINING_COMMAND_MOE_LINEAR_INPUT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_MOE_LINEAR_WEIGHT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_MOE_LINEAR_BIAS_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_MOE_LINEAR_ROUTE_BACKWARD;
}

static int cuda_training_moe_router_command(CudaTrainingCommand command) {
    return command == CUDA_TRAINING_COMMAND_MOE_ROUTER_LOGIT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_MOE_ROUTER_INPUT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_MOE_ROUTER_WEIGHT_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_MOE_ROUTER_BIAS_BACKWARD;
}

/* supports() has descriptors but not the host Params block. Keep this check
 * structural; dispatch performs the exact parameter, byte, and x-grid
 * validation after reading Params. In particular, router bias storage may be
 * larger than num_experts, so its planner grid cannot be inferred from bytes.
 */
static int cuda_training_moe_supports(
        CudaTrainingCommand command, const size_t* bytes, int binding_count,
        uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    if ((!cuda_training_moe_linear_command(command) &&
         !cuda_training_moe_router_command(command)) ||
        !bytes || !groups_x || groups_y != 1u || groups_z != 1u) return 0;

    for (int binding = 0; binding < 10; ++binding) {
        if (!bytes[binding] || bytes[binding] % sizeof(float) != 0u)
            return 0;
    }

    if (cuda_training_moe_linear_command(command)) {
        return binding_count == 13 &&
            bytes[10] != 0u && bytes[10] % sizeof(uint32_t) == 0u &&
            bytes[11] != 0u && bytes[11] % sizeof(uint32_t) == 0u &&
            bytes[12] == 8u * sizeof(uint32_t) &&
            bytes[0] == bytes[6] && bytes[1] == bytes[7] &&
            bytes[2] == bytes[8] && bytes[3] == bytes[4] &&
            bytes[3] == bytes[9];
    }
    return binding_count == 11 && bytes[10] == 8u * sizeof(uint32_t) &&
        bytes[0] == bytes[7] && bytes[1] == bytes[8] &&
        bytes[2] == bytes[9] && bytes[3] == bytes[4] &&
        bytes[3] == bytes[5];
}

static uint32_t cuda_training_moe_read_u32(const void* data, uint32_t index) {
    uint32_t value;
    memcpy(&value, (const unsigned char*)data + (size_t)index * sizeof(value),
           sizeof(value));
    return value;
}

static int cuda_training_moe_linear_mapping_valid(
        void* const* hosts, const size_t* bytes, const uint32_t* params) {
    const uint32_t experts = params[3];
    const uint32_t top_k = params[4];
    const uint32_t domain = params[6];
    if (!domain) {
        return bytes[10] == sizeof(uint32_t) &&
            bytes[11] == sizeof(uint32_t);
    }
    if (domain < experts || top_k > domain ||
        (size_t)domain > SIZE_MAX / sizeof(uint32_t) ||
        bytes[10] != (size_t)domain * sizeof(uint32_t) ||
        bytes[11] != (size_t)experts * sizeof(uint32_t)) return 0;
    for (uint32_t row = 0; row < experts; ++row) {
        const uint32_t slot = cuda_training_moe_read_u32(hosts[11], row);
        if (slot >= domain ||
            cuda_training_moe_read_u32(hosts[10], slot) != row) return 0;
    }
    for (uint32_t slot = 0; slot < domain; ++slot) {
        const uint32_t row = cuda_training_moe_read_u32(hosts[10], slot);
        if (row == UINT32_MAX) continue;
        if (row >= experts ||
            cuda_training_moe_read_u32(hosts[11], row) != slot) return 0;
    }
    return 1;
}

static int cuda_training_moe_stage(
        int binding, void* const* hosts, const size_t* bytes,
        const unsigned char* is_weight, CudaTensorView* views,
        CUdeviceptr* devices) {
    if (!graph_ensure_input_view(hosts[binding], bytes[binding],
                                 is_weight[binding], &views[binding]))
        return 0;
    devices[binding] = views[binding].device;
    return 1;
}

static int cuda_training_dispatch_moe_linear(
        CudaTrainingCommand command, void* const* hosts,
        const size_t* bytes, const unsigned char* is_weight,
        uint32_t groups_x, CudaTensorView* views, CUdeviceptr* devices,
        void** args) {
    uint32_t params[8];
    uint32_t input_elements;
    uint32_t weight_elements;
    uint32_t output_elements;
    uint32_t bias_elements;
    uint32_t route_elements;
    uint32_t input_factors[2];
    uint32_t weight_factors[3];
    uint32_t output_factors[2];
    uint32_t bias_factors[2];
    uint32_t route_factors[2];
    size_t input_bytes;
    size_t weight_bytes;
    size_t output_bytes;
    size_t bias_bytes;
    size_t route_bytes;
    size_t optional_bias_bytes;
    uint32_t expected_groups;
    int destination;
    CUfunction function;

    memcpy(params, hosts[12], sizeof(params));
    if (!params[0] || !params[1] || !params[2] || !params[3] ||
        !params[4] || params[4] > params[3] || params[4] > 8u ||
        params[5] > 1u || params[7] != 0u ||
        !cuda_training_moe_linear_mapping_valid(hosts, bytes, params)) return -1;

    input_factors[0] = params[0];
    input_factors[1] = params[1];
    weight_factors[0] = params[3];
    weight_factors[1] = params[1];
    weight_factors[2] = params[2];
    output_factors[0] = params[0];
    output_factors[1] = params[2];
    bias_factors[0] = params[3];
    bias_factors[1] = params[2];
    route_factors[0] = params[0];
    route_factors[1] = params[4];
    if (!cuda_training_product_bytes(input_factors, 2, sizeof(float),
                                     &input_elements, &input_bytes) ||
        !cuda_training_product_bytes(weight_factors, 3, sizeof(float),
                                     &weight_elements, &weight_bytes) ||
        !cuda_training_product_bytes(output_factors, 2, sizeof(float),
                                     &output_elements, &output_bytes) ||
        !cuda_training_product_bytes(bias_factors, 2, sizeof(float),
                                     &bias_elements, &bias_bytes) ||
        !cuda_training_product_bytes(route_factors, 2, sizeof(float),
                                     &route_elements, &route_bytes)) return -1;

    optional_bias_bytes = params[5] ? bias_bytes : 4u * sizeof(uint32_t);
    if (bytes[0] != input_bytes || bytes[1] != weight_bytes ||
        bytes[2] != optional_bias_bytes || bytes[3] != route_bytes ||
        bytes[4] != route_bytes || bytes[5] != output_bytes ||
        bytes[6] != input_bytes || bytes[7] != weight_bytes ||
        bytes[8] != optional_bias_bytes || bytes[9] != route_bytes)
        return -1;

    if (command == CUDA_TRAINING_COMMAND_MOE_LINEAR_INPUT_BACKWARD) {
        expected_groups = input_elements / CUDA_TRAINING_WORKGROUP_SIZE +
            (input_elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
        destination = 6;
        function = CUDA_TRAINING_FUNCTION(k_training_moe_linear_backward_input);
    } else if (command ==
               CUDA_TRAINING_COMMAND_MOE_LINEAR_WEIGHT_BACKWARD) {
        expected_groups = weight_elements / CUDA_TRAINING_WORKGROUP_SIZE +
            (weight_elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
        destination = 7;
        function = CUDA_TRAINING_FUNCTION(k_training_moe_linear_backward_weight);
    } else if (command == CUDA_TRAINING_COMMAND_MOE_LINEAR_BIAS_BACKWARD) {
        if (!params[5]) return -1;
        expected_groups = bias_elements / CUDA_TRAINING_WORKGROUP_SIZE +
            (bias_elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
        destination = 8;
        function = CUDA_TRAINING_FUNCTION(k_training_moe_linear_backward_bias);
    } else {
        expected_groups = route_elements / CUDA_TRAINING_WORKGROUP_SIZE +
            (route_elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
        destination = 9;
        function = CUDA_TRAINING_FUNCTION(k_training_moe_linear_backward_route);
    }
    if (groups_x != expected_groups) return -1;

    if (!cuda_training_moe_stage(10, hosts, bytes, is_weight, views, devices) ||
        !cuda_training_moe_stage(11, hosts, bytes, is_weight, views, devices) ||
        !cuda_training_moe_stage(12, hosts, bytes, is_weight, views, devices))
        return -1;
    if (command == CUDA_TRAINING_COMMAND_MOE_LINEAR_INPUT_BACKWARD) {
        int required[] = {1, 3, 4, 5, 6};
        for (int index = 0; index < 5; ++index)
            if (!cuda_training_moe_stage(required[index], hosts, bytes,
                                         is_weight, views, devices)) return -1;
    } else if (command ==
               CUDA_TRAINING_COMMAND_MOE_LINEAR_WEIGHT_BACKWARD) {
        int required[] = {0, 3, 4, 5, 7};
        for (int index = 0; index < 5; ++index)
            if (!cuda_training_moe_stage(required[index], hosts, bytes,
                                         is_weight, views, devices)) return -1;
    } else if (command == CUDA_TRAINING_COMMAND_MOE_LINEAR_BIAS_BACKWARD) {
        int required[] = {3, 4, 5, 8};
        for (int index = 0; index < 4; ++index)
            if (!cuda_training_moe_stage(required[index], hosts, bytes,
                                         is_weight, views, devices)) return -1;
    } else {
        int required[] = {0, 1, 3, 5, 9};
        for (int index = 0; index < 5; ++index)
            if (!cuda_training_moe_stage(required[index], hosts, bytes,
                                         is_weight, views, devices)) return -1;
        if (params[5] && !cuda_training_moe_stage(
                2, hosts, bytes, is_weight, views, devices)) return -1;
    }

    for (int binding = 0; binding < 13; ++binding)
        args[binding] = &devices[binding];
    if (!cuda_launch(function, groups_x, 1u, 1u,
                     CUDA_TRAINING_WORKGROUP_SIZE, 1u, 1u, 0u, args))
        return -1;
    graph_mark_device(views[destination].slot);
    return 0;
}

static int cuda_training_dispatch_moe_router(
        CudaTrainingCommand command, void* const* hosts,
        const size_t* bytes, const unsigned char* is_weight,
        uint32_t groups_x, CudaTensorView* views, CUdeviceptr* devices,
        void** args) {
    uint32_t params[8];
    uint32_t input_elements;
    uint32_t weight_elements;
    uint32_t route_elements;
    uint32_t logit_elements;
    uint32_t bias_elements;
    uint32_t input_factors[2];
    uint32_t weight_factors[2];
    uint32_t route_factors[2];
    uint32_t logit_factors[2];
    uint32_t bias_factors[1];
    size_t input_bytes;
    size_t weight_bytes;
    size_t route_bytes;
    size_t logit_bytes;
    size_t bias_bytes;
    float temperature;
    uint32_t expected_groups;
    int destination;
    CUfunction function;

    memcpy(params, hosts[10], sizeof(params));
    memcpy(&temperature, &params[6], sizeof(temperature));
    if (!params[0] || !params[1] || !params[2] || !params[3] ||
        params[3] > params[2] || params[3] > 8u || params[4] > 1u ||
        params[5] > 1u || !isfinite(temperature) || !(temperature > 0.0f) ||
        params[7] != 0u) return -1;

    input_factors[0] = params[0];
    input_factors[1] = params[1];
    weight_factors[0] = params[1];
    weight_factors[1] = params[2];
    route_factors[0] = params[0];
    route_factors[1] = params[3];
    logit_factors[0] = params[0];
    logit_factors[1] = params[2];
    bias_factors[0] = params[2];
    if (!cuda_training_product_bytes(input_factors, 2, sizeof(float),
                                     &input_elements, &input_bytes) ||
        !cuda_training_product_bytes(weight_factors, 2, sizeof(float),
                                     &weight_elements, &weight_bytes) ||
        !cuda_training_product_bytes(route_factors, 2, sizeof(float),
                                     &route_elements, &route_bytes) ||
        !cuda_training_product_bytes(logit_factors, 2, sizeof(float),
                                     &logit_elements, &logit_bytes) ||
        !cuda_training_product_bytes(bias_factors, 1, sizeof(float),
                                     &bias_elements, &bias_bytes)) return -1;

    if (bytes[0] != input_bytes || bytes[1] != weight_bytes ||
        bytes[3] != route_bytes || bytes[4] != route_bytes ||
        bytes[5] != route_bytes || bytes[6] != logit_bytes ||
        bytes[7] != input_bytes || bytes[8] != weight_bytes ||
        bytes[2] != bytes[9]) return -1;
    if (params[5]) {
        if (bytes[2] < bias_bytes) return -1;
    } else if (bytes[2] != 4u * sizeof(uint32_t)) {
        return -1;
    }

    if (command == CUDA_TRAINING_COMMAND_MOE_ROUTER_LOGIT_BACKWARD) {
        expected_groups = logit_elements / CUDA_TRAINING_WORKGROUP_SIZE +
            (logit_elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
        destination = 6;
        function = CUDA_TRAINING_FUNCTION(k_training_moe_router_backward_logit);
    } else if (command ==
               CUDA_TRAINING_COMMAND_MOE_ROUTER_INPUT_BACKWARD) {
        expected_groups = input_elements / CUDA_TRAINING_WORKGROUP_SIZE +
            (input_elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
        destination = 7;
        function = CUDA_TRAINING_FUNCTION(k_training_moe_router_backward_input);
    } else if (command ==
               CUDA_TRAINING_COMMAND_MOE_ROUTER_WEIGHT_BACKWARD) {
        expected_groups = weight_elements / CUDA_TRAINING_WORKGROUP_SIZE +
            (weight_elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
        destination = 8;
        function = CUDA_TRAINING_FUNCTION(k_training_moe_router_backward_weight);
    } else {
        if (!params[5]) return -1;
        expected_groups = bias_elements / CUDA_TRAINING_WORKGROUP_SIZE +
            (bias_elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
        destination = 9;
        function = CUDA_TRAINING_FUNCTION(k_training_moe_router_backward_bias);
    }
    if (groups_x != expected_groups) return -1;

    if (!cuda_training_moe_stage(10, hosts, bytes, is_weight, views,
                                 devices)) return -1;
    if (command == CUDA_TRAINING_COMMAND_MOE_ROUTER_LOGIT_BACKWARD) {
        int required[] = {3, 4, 5, 6};
        for (int index = 0; index < 4; ++index)
            if (!cuda_training_moe_stage(required[index], hosts, bytes,
                                         is_weight, views, devices)) return -1;
        if (params[4] == 0u) {
            if (!cuda_training_moe_stage(0, hosts, bytes, is_weight, views,
                                         devices) ||
                !cuda_training_moe_stage(1, hosts, bytes, is_weight, views,
                                         devices) ||
                (params[5] && !cuda_training_moe_stage(
                    2, hosts, bytes, is_weight, views, devices))) return -1;
        }
    } else if (command ==
               CUDA_TRAINING_COMMAND_MOE_ROUTER_INPUT_BACKWARD) {
        int required[] = {1, 6, 7};
        for (int index = 0; index < 3; ++index)
            if (!cuda_training_moe_stage(required[index], hosts, bytes,
                                         is_weight, views, devices)) return -1;
    } else if (command ==
               CUDA_TRAINING_COMMAND_MOE_ROUTER_WEIGHT_BACKWARD) {
        int required[] = {0, 6, 8};
        for (int index = 0; index < 3; ++index)
            if (!cuda_training_moe_stage(required[index], hosts, bytes,
                                         is_weight, views, devices)) return -1;
    } else {
        int required[] = {6, 9};
        for (int index = 0; index < 2; ++index)
            if (!cuda_training_moe_stage(required[index], hosts, bytes,
                                         is_weight, views, devices)) return -1;
    }

    for (int binding = 0; binding < 11; ++binding)
        args[binding] = &devices[binding];
    if (!cuda_launch(function, groups_x, 1u, 1u,
                     CUDA_TRAINING_WORKGROUP_SIZE, 1u, 1u, 0u, args))
        return -1;
    graph_mark_device(views[destination].slot);
    return 0;
}

static int cuda_training_dispatch_moe(
        CudaTrainingCommand command, void* const* hosts,
        const size_t* bytes, const unsigned char* access,
        const unsigned char* is_weight, uint32_t groups_x,
        uint32_t groups_y, uint32_t groups_z) {
    CudaTensorView views[13];
    CUdeviceptr devices[13];
    void* args[13];
    const int binding_count = cuda_training_moe_linear_command(command) ? 13 : 11;

    if (!hosts || !bytes || !access || !is_weight ||
        !cuda_training_moe_supports(command, bytes, binding_count, groups_x, groups_y,
                                    groups_z)) return -1;
    for (int binding = 0; binding < binding_count; ++binding) {
        unsigned char expected = binding >= 6 && binding <= 9
            ? CUDA_TRAINING_READ_WRITE : CUDA_TRAINING_READ;
        if (!hosts[binding] || access[binding] != expected) return -1;
        if (binding >= 10) {
            if (is_weight[binding] != 0u) return -1;
        } else if (is_weight[binding] > 1u) {
            return -1;
        }
    }

    memset(views, 0, sizeof(views));
    memset(devices, 0, sizeof(devices));
    memset(args, 0, sizeof(args));
    if (cuda_training_moe_linear_command(command)) {
        return cuda_training_dispatch_moe_linear(command, hosts, bytes,
            is_weight, groups_x, views, devices, args);
    }
    if (cuda_training_moe_router_command(command)) {
        return cuda_training_dispatch_moe_router(command, hosts, bytes,
            is_weight, groups_x, views, devices, args);
    }
    return -1;
}

#endif
