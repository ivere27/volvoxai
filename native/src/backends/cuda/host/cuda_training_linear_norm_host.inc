static int cuda_training_dispatch_matmul(
        CudaTrainingCommand command, void* const* hosts,
        const size_t* bytes, const unsigned char* access,
        const unsigned char* is_weight, uint32_t groups_x,
        uint32_t groups_y, uint32_t groups_z) {
    uint32_t params[8];
    uint32_t rows;
    uint32_t d_in;
    uint32_t d_out;
    uint32_t has_bias;
    uint32_t canonical_din_layout;
    uint32_t weight_din_layout;
    uint32_t expected_x;
    uint32_t expected_y;
    size_t input_bytes;
    size_t weight_bytes;
    size_t output_bytes;
    size_t bias_bytes;
    CudaTensorView views[6];
    CUdeviceptr devices[6] = {0};
    CUfunction function = NULL;
    uint32_t block_x = 0;
    uint32_t block_y = 1u;
    void* args[12];

    if (!hosts[0] || !hosts[1] || !hosts[2] || !hosts[3] || !hosts[4] ||
        !hosts[5] || !hosts[6] ||
        access[0] != CUDA_TRAINING_READ ||
        access[1] != CUDA_TRAINING_READ ||
        access[2] != CUDA_TRAINING_READ ||
        access[3] != CUDA_TRAINING_READ_WRITE ||
        access[4] != CUDA_TRAINING_READ_WRITE ||
        access[5] != CUDA_TRAINING_READ_WRITE ||
        access[6] != CUDA_TRAINING_READ || is_weight[6] != 0u)
        return -1;
    for (int binding = 0; binding < 6; binding++)
        if (is_weight[binding] > 1u) return -1;

    memcpy(params, hosts[6], sizeof(params));
    rows = params[0];
    d_in = params[1];
    d_out = params[2];
    has_bias = params[3];
    canonical_din_layout = params[4];
    weight_din_layout = params[5];
    if (!rows || !d_in || !d_out || has_bias > 1u ||
        canonical_din_layout > 1u || weight_din_layout > 1u ||
        params[6] != 0u || params[7] != 0u ||
        !size_product_u32(rows, d_in, sizeof(float), &input_bytes) ||
        !size_product_u32(d_in, d_out, sizeof(float), &weight_bytes) ||
        !size_product_u32(rows, d_out, sizeof(float), &output_bytes) ||
        !size_product_u32(1u, d_out, sizeof(float), &bias_bytes) ||
        bytes[0] != input_bytes || bytes[1] != weight_bytes ||
        bytes[2] != output_bytes || bytes[3] != input_bytes ||
        bytes[4] != weight_bytes ||
        bytes[5] != (has_bias ? bias_bytes : 4u * sizeof(uint32_t)))
        return -1;

    if (command == CUDA_TRAINING_COMMAND_MATMUL_INPUT_BACKWARD) {
        expected_x = d_in / 16u + (d_in % 16u != 0u);
        expected_y = rows / 16u + (rows % 16u != 0u);
        function = CUDA_TRAINING_FUNCTION(k_training_matmul_backward_input);
        block_x = block_y = 16u;
    } else if (command == CUDA_TRAINING_COMMAND_MATMUL_WEIGHT_BACKWARD) {
        expected_x = d_out / 16u + (d_out % 16u != 0u);
        expected_y = d_in / 16u + (d_in % 16u != 0u);
        function = CUDA_TRAINING_FUNCTION(k_training_matmul_backward_weight);
        block_x = block_y = 16u;
    } else {
        expected_x = d_out / CUDA_TRAINING_WORKGROUP_SIZE +
            (d_out % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
        expected_y = 1u;
        function = CUDA_TRAINING_FUNCTION(k_training_matmul_backward_bias);
        block_x = CUDA_TRAINING_WORKGROUP_SIZE;
    }
    if (groups_x != expected_x || groups_y != expected_y || groups_z != 1u ||
        (command == CUDA_TRAINING_COMMAND_MATMUL_BIAS_BACKWARD && !has_bias))
        return -1;

    memset(views, 0, sizeof(views));
    if (command == CUDA_TRAINING_COMMAND_MATMUL_INPUT_BACKWARD) {
        if (!graph_ensure_input_view(hosts[1], bytes[1], is_weight[1],
                                     &views[1]) ||
            !graph_ensure_input_view(hosts[2], bytes[2], is_weight[2],
                                     &views[2]) ||
            !graph_ensure_input_view(hosts[3], bytes[3], is_weight[3],
                                     &views[3])) return -1;
        devices[1] = views[1].device;
        devices[2] = views[2].device;
        devices[3] = views[3].device;
    } else if (command == CUDA_TRAINING_COMMAND_MATMUL_WEIGHT_BACKWARD) {
        if (!graph_ensure_input_view(hosts[0], bytes[0], is_weight[0],
                                     &views[0]) ||
            !graph_ensure_input_view(hosts[2], bytes[2], is_weight[2],
                                     &views[2]) ||
            !graph_ensure_input_view(hosts[4], bytes[4], is_weight[4],
                                     &views[4])) return -1;
        devices[0] = views[0].device;
        devices[2] = views[2].device;
        devices[4] = views[4].device;
    } else {
        if (!graph_ensure_input_view(hosts[2], bytes[2], is_weight[2],
                                     &views[2]) ||
            !graph_ensure_input_view(hosts[5], bytes[5], is_weight[5],
                                     &views[5])) return -1;
        devices[2] = views[2].device;
        devices[5] = views[5].device;
    }

    for (int binding = 0; binding < 6; binding++) args[binding] = &devices[binding];
    args[6] = &rows;
    args[7] = &d_in;
    args[8] = &d_out;
    args[9] = &has_bias;
    args[10] = &canonical_din_layout;
    args[11] = &weight_din_layout;
    if (!cuda_launch(function, groups_x, groups_y, 1u, block_x, block_y,
                     1u, 0u, args)) return -1;
    if (command == CUDA_TRAINING_COMMAND_MATMUL_INPUT_BACKWARD)
        graph_mark_device(views[3].slot);
    else if (command == CUDA_TRAINING_COMMAND_MATMUL_WEIGHT_BACKWARD)
        graph_mark_device(views[4].slot);
    else
        graph_mark_device(views[5].slot);
    return 0;
}

static int cuda_training_basic_operand_strides_valid(
        const uint32_t* params, const uint32_t dimensions[8],
        uint32_t base, uint32_t expected_length) {
    uint64_t contiguous = 1u;
    int rank = (int)params[0];
    for (int dimension = rank - 1; dimension >= 0; dimension--) {
        uint32_t stride = params[base + (uint32_t)dimension];
        if (!stride) continue;
        if (dimensions[dimension] <= 1u || stride != contiguous) return 0;
        contiguous *= dimensions[dimension];
        if (contiguous > UINT32_MAX) return 0;
    }
    return contiguous == expected_length;
}

static int cuda_training_basic_params_valid(
        const uint32_t params[32], const size_t* bytes,
        uint32_t dimensions[8]) {
    uint32_t rank = params[0];
    uint32_t output_length = params[1];
    uint32_t a_length = params[2];
    uint32_t b_length = params[3];
    if (rank > 8u || !output_length || !a_length || !b_length ||
        params[4] > 3u || params[5] > 2u ||
        (params[4] != 0u && params[5] != 0u) || params[6] || params[7] ||
        bytes[0] != (size_t)a_length * sizeof(float) ||
        bytes[1] != (size_t)b_length * sizeof(float) ||
        bytes[2] != (size_t)output_length * sizeof(float) ||
        bytes[3] != bytes[0] || bytes[4] != bytes[1]) return 0;
    memset(dimensions, 0, 8u * sizeof(uint32_t));
    if (!rank) {
        if (output_length != 1u || a_length != 1u || b_length != 1u)
            return 0;
    } else {
        for (uint32_t dimension = 0; dimension < rank; dimension++) {
            uint32_t stride = params[8u + dimension];
            uint32_t numerator = dimension == 0u
                ? output_length : params[7u + dimension];
            if (!stride || numerator % stride != 0u) return 0;
            dimensions[dimension] = numerator / stride;
            if (!dimensions[dimension]) return 0;
        }
        if (params[7u + rank] != 1u ||
            !cuda_training_basic_operand_strides_valid(
                params, dimensions, 16u, a_length) ||
            !cuda_training_basic_operand_strides_valid(
                params, dimensions, 24u, b_length)) return 0;
    }
    for (uint32_t dimension = rank; dimension < 8u; dimension++) {
        if (params[8u + dimension] || params[16u + dimension] ||
            params[24u + dimension]) return 0;
    }
    return 1;
}

static int cuda_training_dispatch_basic(
        CudaTrainingCommand command, void* const* hosts,
        const size_t* bytes, const unsigned char* access,
        const unsigned char* is_weight, uint32_t groups_x,
        uint32_t groups_y, uint32_t groups_z) {
    uint32_t params[32];
    uint32_t dimensions[8];
    uint32_t destination_length;
    uint32_t expected_groups;
    int a_command;
    int cooperative;
    int staged;
    CudaTensorView views[6];
    CUdeviceptr devices[6];
    CUfunction function;
    void* args[6];
    void* partial_args[5];
    void* final_args[3];

    if (!hosts[0] || !hosts[1] || !hosts[2] || !hosts[3] || !hosts[4] ||
        !hosts[5] || access[0] != CUDA_TRAINING_READ ||
        access[1] != CUDA_TRAINING_READ ||
        access[2] != CUDA_TRAINING_READ ||
        access[3] != CUDA_TRAINING_READ_WRITE ||
        access[4] != CUDA_TRAINING_READ_WRITE ||
        access[5] != CUDA_TRAINING_READ || is_weight[5] != 0u) return -1;
    for (int binding = 0; binding < 5; binding++)
        if (is_weight[binding] > 1u) return -1;
    memcpy(params, hosts[5], sizeof(params));
    if (!cuda_training_basic_params_valid(params, bytes, dimensions)) return -1;
    a_command = command == CUDA_TRAINING_COMMAND_BASIC_A_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BASIC_A_COOPERATIVE_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BASIC_A_SCALAR_STAGED_BACKWARD;
    cooperative =
        command == CUDA_TRAINING_COMMAND_BASIC_A_COOPERATIVE_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BASIC_B_COOPERATIVE_BACKWARD;
    staged =
        command == CUDA_TRAINING_COMMAND_BASIC_A_SCALAR_STAGED_BACKWARD ||
        command == CUDA_TRAINING_COMMAND_BASIC_B_SCALAR_STAGED_BACKWARD;
    destination_length = a_command ? params[2] : params[3];
    if (staged) {
        if (destination_length != 1u ||
            params[1] <= CUDA_TRAINING_BASIC_SCALAR_ITEMS_PER_PARTIAL)
            return -1;
        expected_groups = params[1] /
            CUDA_TRAINING_BASIC_SCALAR_ITEMS_PER_PARTIAL +
            (params[1] % CUDA_TRAINING_BASIC_SCALAR_ITEMS_PER_PARTIAL != 0u);
    } else if (cooperative) {
        if (destination_length <= 1u || destination_length > 1024u ||
            destination_length >= params[1]) return -1;
        expected_groups = destination_length;
    } else {
        expected_groups = destination_length / CUDA_TRAINING_WORKGROUP_SIZE +
            (destination_length % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
    }
    if (groups_x != expected_groups || groups_y != 1u || groups_z != 1u)
        return -1;
    if (staged &&
        (!cuda_training_basic_workspace_reserve(
             (size_t)groups_x * sizeof(float)))) return -1;

    memset(views, 0, sizeof(views));
    memset(devices, 0, sizeof(devices));
    for (int binding = 0; binding < 3; binding++) {
        if (!graph_ensure_input_view(hosts[binding], bytes[binding],
                                     is_weight[binding], &views[binding]))
            return -1;
        devices[binding] = views[binding].device;
    }
    int destination = a_command ? 3 : 4;
    if (!graph_ensure_input_view(hosts[destination], bytes[destination],
                                 is_weight[destination],
                                 &views[destination]) ||
        !graph_ensure_input_view(hosts[5], bytes[5], 0, &views[5])) return -1;
    devices[destination] = views[destination].device;
    devices[5] = views[5].device;
    if (staged) {
        function = a_command
            ? CUDA_TRAINING_FUNCTION(k_training_basic_backward_a_scalar_partials)
            : CUDA_TRAINING_FUNCTION(k_training_basic_backward_b_scalar_partials);
        partial_args[0] = &devices[0];
        partial_args[1] = &devices[1];
        partial_args[2] = &devices[2];
        partial_args[3] = &cuda_training_basic_workspace;
        partial_args[4] = &devices[5];
        final_args[0] = &cuda_training_basic_workspace;
        final_args[1] = &devices[destination];
        final_args[2] = &groups_x;
        if (!cuda_launch(function, groups_x, 1u, 1u,
                         CUDA_TRAINING_WORKGROUP_SIZE, 1u, 1u, 0u,
                         partial_args) ||
            !cuda_launch(CUDA_TRAINING_FUNCTION(k_training_basic_backward_scalar_finalize),
                         1u, 1u, 1u, CUDA_TRAINING_WORKGROUP_SIZE,
                         1u, 1u, 0u, final_args)) return -1;
        graph_mark_device(views[destination].slot);
        return 0;
    }
    if (cooperative) {
        function = a_command ? CUDA_TRAINING_FUNCTION(k_training_basic_backward_a_cooperative)
                             : CUDA_TRAINING_FUNCTION(k_training_basic_backward_b_cooperative);
    } else {
        function = a_command ? CUDA_TRAINING_FUNCTION(k_training_basic_backward_a)
                             : CUDA_TRAINING_FUNCTION(k_training_basic_backward_b);
    }
    for (int binding = 0; binding < 6; binding++) args[binding] = &devices[binding];
    if (!cuda_launch(function, groups_x, 1u, 1u,
                     CUDA_TRAINING_WORKGROUP_SIZE, 1u, 1u, 0u, args)) return -1;
    graph_mark_device(views[destination].slot);
    return 0;
}

static int cuda_training_dispatch_activation(
        void* const* hosts, const size_t* bytes,
        const unsigned char* access, const unsigned char* is_weight,
        uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    uint32_t params[4];
    uint32_t length;
    uint32_t expected_groups;
    float alpha;
    float beta;
    CudaTensorView views[5];
    CUdeviceptr devices[5];
    void* args[5];

    if (!hosts[0] || !hosts[1] || !hosts[2] || !hosts[3] || !hosts[4] ||
        access[0] != CUDA_TRAINING_READ ||
        access[1] != CUDA_TRAINING_READ ||
        access[2] != CUDA_TRAINING_READ ||
        access[3] != CUDA_TRAINING_READ_WRITE ||
        access[4] != CUDA_TRAINING_READ || is_weight[4] != 0u) return -1;
    for (int binding = 0; binding < 4; binding++)
        if (is_weight[binding] > 1u) return -1;
    memcpy(params, hosts[4], sizeof(params));
    length = params[0];
    memcpy(&alpha, &params[2], sizeof(alpha));
    memcpy(&beta, &params[3], sizeof(beta));
    expected_groups = length / CUDA_TRAINING_WORKGROUP_SIZE +
        (length % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
    if (!length || params[1] > 9u || !isfinite(alpha) || !isfinite(beta) ||
        bytes[0] != (size_t)length * sizeof(float) ||
        groups_x != expected_groups || groups_y != 1u || groups_z != 1u ||
        (params[1] == 8u && !(alpha < beta))) return -1;

    memset(views, 0, sizeof(views));
    memset(devices, 0, sizeof(devices));
    for (int binding = 0; binding < 5; binding++) {
        if (!graph_ensure_input_view(hosts[binding], bytes[binding],
                                     binding < 4 ? is_weight[binding] : 0,
                                     &views[binding])) return -1;
        devices[binding] = views[binding].device;
    }
    for (int binding = 0; binding < 5; binding++) args[binding] = &devices[binding];
    if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_training_activation_backward), groups_x, 1u, 1u,
                     CUDA_TRAINING_WORKGROUP_SIZE, 1u, 1u, 0u, args)) return -1;
    graph_mark_device(views[3].slot);
    return 0;
}

static int cuda_training_dispatch_reduce(
        void* const* hosts, const size_t* bytes,
        const unsigned char* access, const unsigned char* is_weight,
        uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    uint32_t params[4];
    uint32_t elements;
    uint32_t row_width;
    uint32_t rows;
    uint32_t expected_groups;
    float scale;
    size_t input_bytes;
    size_t destination_bytes;
    CudaTensorView source;
    CudaTensorView destination;
    void* args[5];

    if (!hosts[0] || !hosts[1] || !hosts[2] ||
        access[0] != CUDA_TRAINING_READ ||
        access[1] != CUDA_TRAINING_READ_WRITE ||
        access[2] != CUDA_TRAINING_READ || is_weight[0] > 1u ||
        is_weight[1] > 1u || is_weight[2] != 0u) return -1;
    memcpy(params, hosts[2], sizeof(params));
    elements = params[0];
    row_width = params[1];
    memcpy(&scale, &params[2], sizeof(scale));
    if (!elements || !row_width || elements % row_width != 0u ||
        !isfinite(scale) || params[3] != 0u) return -1;
    rows = elements / row_width;
    if (!size_product_u32(1u, rows, sizeof(float), &input_bytes) ||
        !size_product_u32(1u, elements, sizeof(float), &destination_bytes))
        return -1;
    expected_groups = elements / CUDA_TRAINING_WORKGROUP_SIZE +
        (elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
    if (bytes[0] != input_bytes || bytes[1] != destination_bytes ||
        groups_x != expected_groups || groups_y != 1u || groups_z != 1u)
        return -1;
    if (!graph_ensure_input_view(hosts[0], bytes[0], is_weight[0], &source) ||
        !graph_ensure_input_view(hosts[1], bytes[1], is_weight[1],
                                 &destination)) return -1;
    args[0] = &source.device;
    args[1] = &destination.device;
    args[2] = &elements;
    args[3] = &row_width;
    args[4] = &scale;
    if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_training_reduce_backward), groups_x, 1u, 1u,
                     CUDA_TRAINING_WORKGROUP_SIZE, 1u, 1u, 0u, args)) return -1;
    graph_mark_device(destination.slot);
    return 0;
}

static int cuda_training_dispatch_softmax(
        void* const* hosts, const size_t* bytes,
        const unsigned char* access, const unsigned char* is_weight,
        uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    uint32_t params[4];
    uint32_t elements;
    uint32_t expected_groups;
    size_t tensor_bytes;
    CudaTensorView views[4];
    CUdeviceptr devices[4];
    void* args[4];

    if (!hosts[0] || !hosts[1] || !hosts[2] || !hosts[3] ||
        access[0] != CUDA_TRAINING_READ ||
        access[1] != CUDA_TRAINING_READ ||
        access[2] != CUDA_TRAINING_READ_WRITE ||
        access[3] != CUDA_TRAINING_READ || is_weight[3] != 0u) return -1;
    for (int binding = 0; binding < 3; binding++)
        if (is_weight[binding] > 1u) return -1;
    memcpy(params, hosts[3], sizeof(params));
    if (!params[0] || !params[1] || params[2] > 1u || params[3] != 0u ||
        !size_product_u32(params[0], params[1], sizeof(float),
                          &tensor_bytes) || bytes[0] != tensor_bytes)
        return -1;
    elements = (uint32_t)(tensor_bytes / sizeof(float));
    expected_groups = elements / CUDA_TRAINING_WORKGROUP_SIZE +
        (elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
    if (groups_x != expected_groups || groups_y != 1u || groups_z != 1u)
        return -1;
    memset(views, 0, sizeof(views));
    memset(devices, 0, sizeof(devices));
    for (int binding = 0; binding < 4; binding++) {
        if (!graph_ensure_input_view(hosts[binding], bytes[binding],
                                     binding < 3 ? is_weight[binding] : 0,
                                     &views[binding])) return -1;
        devices[binding] = views[binding].device;
    }
    for (int binding = 0; binding < 4; binding++) args[binding] = &devices[binding];
    if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_training_softmax_backward), groups_x, 1u, 1u,
                     CUDA_TRAINING_WORKGROUP_SIZE, 1u, 1u, 0u, args)) return -1;
    graph_mark_device(views[2].slot);
    return 0;
}

static int cuda_training_dispatch_embedding(
        void* const* hosts, const size_t* bytes,
        const unsigned char* access, const unsigned char* is_weight,
        uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    uint32_t params[4];
    uint32_t output_elements;
    uint32_t weight_elements;
    uint32_t expected_groups;
    size_t token_bytes;
    size_t output_bytes;
    size_t weight_bytes;
    CudaTensorView views[4];
    CUdeviceptr devices[4];
    void* args[4];

    if (!hosts[0] || !hosts[1] || !hosts[2] || !hosts[3] ||
        access[0] != CUDA_TRAINING_READ ||
        access[1] != CUDA_TRAINING_READ ||
        access[2] != CUDA_TRAINING_READ_WRITE ||
        access[3] != CUDA_TRAINING_READ || is_weight[3] != 0u) return -1;
    for (int binding = 0; binding < 3; binding++)
        if (is_weight[binding] > 1u) return -1;
    memcpy(params, hosts[3], sizeof(params));
    if (!params[0] || !params[1] || !params[2] || params[3] != 0u ||
        !size_product_u32(1u, params[0], sizeof(int32_t), &token_bytes) ||
        !size_product_u32(params[0], params[1], sizeof(float),
                          &output_bytes) ||
        !size_product_u32(params[2], params[1], sizeof(float),
                          &weight_bytes) ||
        bytes[0] != token_bytes ||
        bytes[1] != output_bytes || bytes[2] != weight_bytes) return -1;
    const int32_t* tokens = (const int32_t*)hosts[0];
    for (uint32_t row = 0; row < params[0]; row++)
        if (tokens[row] < 0 || (uint32_t)tokens[row] >= params[2]) return -1;
    output_elements = (uint32_t)(output_bytes / sizeof(float));
    weight_elements = (uint32_t)(weight_bytes / sizeof(float));
    (void)output_elements;
    expected_groups = weight_elements / CUDA_TRAINING_WORKGROUP_SIZE +
        (weight_elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
    if (groups_x != expected_groups || groups_y != 1u || groups_z != 1u)
        return -1;
    memset(views, 0, sizeof(views));
    memset(devices, 0, sizeof(devices));
    for (int binding = 0; binding < 4; binding++) {
        if (!graph_ensure_input_view(hosts[binding], bytes[binding],
                                     binding < 3 ? is_weight[binding] : 0,
                                     &views[binding])) return -1;
        devices[binding] = views[binding].device;
    }
    for (int binding = 0; binding < 4; binding++) args[binding] = &devices[binding];
    if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_training_embedding_backward), groups_x, 1u, 1u,
                     CUDA_TRAINING_WORKGROUP_SIZE, 1u, 1u, 0u, args)) return -1;
    graph_mark_device(views[2].slot);
    return 0;
}

static int cuda_training_dispatch_layernorm(
        CudaTrainingCommand command, void* const* hosts,
        const size_t* bytes, const unsigned char* access,
        const unsigned char* is_weight, uint32_t groups_x,
        uint32_t groups_y, uint32_t groups_z) {
    uint32_t params[4];
    uint32_t elements;
    uint32_t expected_groups;
    float epsilon;
    size_t tensor_bytes;
    size_t parameter_bytes;
    CudaTensorView views[7];
    CUdeviceptr devices[7];
    CUfunction function;
    void* args[7];

    if (!hosts[0] || !hosts[1] || !hosts[2] || !hosts[3] || !hosts[4] ||
        !hosts[5] || !hosts[6] ||
        access[0] != CUDA_TRAINING_READ ||
        access[1] != CUDA_TRAINING_READ ||
        access[2] != CUDA_TRAINING_READ ||
        access[3] != CUDA_TRAINING_READ_WRITE ||
        access[4] != CUDA_TRAINING_READ_WRITE ||
        access[5] != CUDA_TRAINING_READ_WRITE ||
        access[6] != CUDA_TRAINING_READ || is_weight[6] != 0u) return -1;
    for (int binding = 0; binding < 6; binding++)
        if (is_weight[binding] > 1u) return -1;
    memcpy(params, hosts[6], sizeof(params));
    memcpy(&epsilon, &params[2], sizeof(epsilon));
    if (!params[0] || !params[1] || !(epsilon > 0.0f) ||
        !isfinite(epsilon) || params[3] > 1u ||
        !size_product_u32(params[0], params[1], sizeof(float),
                          &tensor_bytes) ||
        !size_product_u32(1u, params[1], sizeof(float),
                          &parameter_bytes) ||
        bytes[0] != tensor_bytes || bytes[2] != tensor_bytes ||
        bytes[3] != tensor_bytes || bytes[1] < parameter_bytes ||
        bytes[1] % sizeof(float) != 0u || bytes[4] != bytes[1] ||
        tensor_bytes / sizeof(float) > UINT32_MAX ||
        (params[3]
            ? (bytes[5] < parameter_bytes ||
               bytes[5] % sizeof(float) != 0u)
            : bytes[5] != 4u * sizeof(uint32_t)))
        return -1;
    elements = (uint32_t)(tensor_bytes / sizeof(float));
    if (command == CUDA_TRAINING_COMMAND_LAYERNORM_INPUT_BACKWARD) {
        expected_groups = elements / CUDA_TRAINING_WORKGROUP_SIZE +
            (elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
        function = CUDA_TRAINING_FUNCTION(k_training_layernorm_backward_input);
    } else if (command == CUDA_TRAINING_COMMAND_LAYERNORM_PARAM_BACKWARD) {
        expected_groups = params[1] / CUDA_TRAINING_WORKGROUP_SIZE +
            (params[1] % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
        function = CUDA_TRAINING_FUNCTION(k_training_layernorm_backward_param);
    } else {
        if (params[1] > (uint32_t)INT_MAX) return -1;
        expected_groups = params[1];
        function = CUDA_TRAINING_FUNCTION(k_training_layernorm_backward_param_cooperative);
    }
    if (groups_x != expected_groups || groups_y != 1u || groups_z != 1u)
        return -1;
    memset(views, 0, sizeof(views));
    memset(devices, 0, sizeof(devices));
    int required[] = {0, 2, 6};
    for (int index = 0; index < 3; index++) {
        int binding = required[index];
        if (!graph_ensure_input_view(hosts[binding], bytes[binding],
                                     binding == 6 ? 0 : is_weight[binding],
                                     &views[binding])) return -1;
        devices[binding] = views[binding].device;
    }
    if (command == CUDA_TRAINING_COMMAND_LAYERNORM_INPUT_BACKWARD) {
        int input_required[] = {1, 3};
        for (int index = 0; index < 2; index++) {
            int binding = input_required[index];
            if (!graph_ensure_input_view(hosts[binding], bytes[binding],
                                         is_weight[binding], &views[binding]))
                return -1;
            devices[binding] = views[binding].device;
        }
    } else {
        if (!graph_ensure_input_view(hosts[4], bytes[4], is_weight[4],
                                     &views[4])) return -1;
        devices[4] = views[4].device;
        if (params[3]) {
            if (!graph_ensure_input_view(hosts[5], bytes[5], is_weight[5],
                                         &views[5])) return -1;
            devices[5] = views[5].device;
        }
    }
    for (int binding = 0; binding < 7; binding++) args[binding] = &devices[binding];
    if (!cuda_launch(function, groups_x, 1u, 1u,
                     CUDA_TRAINING_WORKGROUP_SIZE, 1u, 1u, 0u, args)) return -1;
    if (command == CUDA_TRAINING_COMMAND_LAYERNORM_INPUT_BACKWARD)
        graph_mark_device(views[3].slot);
    else {
        graph_mark_device(views[4].slot);
        if (params[3]) graph_mark_device(views[5].slot);
    }
    return 0;
}

static int cuda_training_dispatch_rmsnorm(
        CudaTrainingCommand command, void* const* hosts,
        const size_t* bytes, const unsigned char* access,
        const unsigned char* is_weight, uint32_t groups_x,
        uint32_t groups_y, uint32_t groups_z) {
    uint32_t params[4];
    uint32_t elements;
    uint32_t expected_groups;
    float epsilon;
    size_t tensor_bytes;
    size_t parameter_bytes;
    CudaTensorView views[6];
    CUdeviceptr devices[6];
    CUfunction function;
    void* args[6];

    if (!hosts[0] || !hosts[1] || !hosts[2] || !hosts[3] || !hosts[4] ||
        !hosts[5] || access[0] != CUDA_TRAINING_READ ||
        access[1] != CUDA_TRAINING_READ ||
        access[2] != CUDA_TRAINING_READ ||
        access[3] != CUDA_TRAINING_READ_WRITE ||
        access[4] != CUDA_TRAINING_READ_WRITE ||
        access[5] != CUDA_TRAINING_READ || is_weight[5] != 0u) return -1;
    for (int binding = 0; binding < 5; binding++)
        if (is_weight[binding] > 1u) return -1;
    memcpy(params, hosts[5], sizeof(params));
    memcpy(&epsilon, &params[2], sizeof(epsilon));
    if (!params[0] || !params[1] || !(epsilon > 0.0f) ||
        !isfinite(epsilon) || params[3] != 0u ||
        !size_product_u32(params[0], params[1], sizeof(float),
                          &tensor_bytes) ||
        !size_product_u32(1u, params[1], sizeof(float),
                          &parameter_bytes) ||
        bytes[0] != tensor_bytes || bytes[2] != tensor_bytes ||
        bytes[3] != tensor_bytes || bytes[1] < parameter_bytes ||
        bytes[1] % sizeof(float) != 0u || bytes[4] != bytes[1]) return -1;
    elements = (uint32_t)(tensor_bytes / sizeof(float));
    if (command == CUDA_TRAINING_COMMAND_RMSNORM_INPUT_BACKWARD) {
        expected_groups = elements / CUDA_TRAINING_WORKGROUP_SIZE +
            (elements % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
        function = CUDA_TRAINING_FUNCTION(k_training_rmsnorm_backward_input);
    } else {
        expected_groups = params[1] / CUDA_TRAINING_WORKGROUP_SIZE +
            (params[1] % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
        function = CUDA_TRAINING_FUNCTION(k_training_rmsnorm_backward_weight);
    }
    if (groups_x != expected_groups || groups_y != 1u || groups_z != 1u)
        return -1;
    memset(views, 0, sizeof(views));
    memset(devices, 0, sizeof(devices));
    int required[] = {0, 2, 5};
    for (int index = 0; index < 3; index++) {
        int binding = required[index];
        if (!graph_ensure_input_view(hosts[binding], bytes[binding],
                                     binding == 5 ? 0 : is_weight[binding],
                                     &views[binding])) return -1;
        devices[binding] = views[binding].device;
    }
    int destination = command == CUDA_TRAINING_COMMAND_RMSNORM_INPUT_BACKWARD
        ? 3 : 4;
    if (command == CUDA_TRAINING_COMMAND_RMSNORM_INPUT_BACKWARD) {
        if (!graph_ensure_input_view(hosts[1], bytes[1], is_weight[1],
                                     &views[1])) return -1;
        devices[1] = views[1].device;
    }
    if (!graph_ensure_input_view(hosts[destination], bytes[destination],
                                 is_weight[destination], &views[destination]))
        return -1;
    devices[destination] = views[destination].device;
    for (int binding = 0; binding < 6; binding++) args[binding] = &devices[binding];
    if (!cuda_launch(function, groups_x, 1u, 1u,
                     CUDA_TRAINING_WORKGROUP_SIZE, 1u, 1u, 0u, args)) return -1;
    graph_mark_device(views[destination].slot);
    return 0;
}

static int cuda_training_product_bytes(const uint32_t* factors, int count,
                                       size_t element_size,
                                       uint32_t* elements, size_t* bytes) {
    uint64_t product = 1u;
    if (!factors || count <= 0) return 0;
    for (int index = 0; index < count; index++) {
        if (!factors[index] || product > UINT32_MAX / factors[index]) return 0;
        product *= factors[index];
    }
    return checked_elements_bytes(product, element_size, elements, bytes);
}

#include "cuda_training_shape_host.inc"
#include "cuda_training_norm_extra_host.inc"
#include "cuda_training_moe_host.inc"
