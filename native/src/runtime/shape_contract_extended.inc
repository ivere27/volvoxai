/* Shape contracts for the late specialty operators: Not/Mask, BatchNorm2D,
 * Interpolate1D, vision profiles, Dropout, Requantize, Concat2, MoE router and
 * linear, CrossAttention and Scan. */

static int vx_shape_infer_extended_not(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    if (vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_i32_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank_range(input, 0, 8, "operator input 'input'.shape", error)) {
        return -1;
    }
    return vx_shape_clone_output(candidate, input->shape, input->rank,
                                 VX_DTYPE_I32, NULL, error);
}

static int vx_shape_infer_extended_mask(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* mask =
        vx_shape_input_descriptor(request, "mask");
    const VxShapeTensorDescriptor* left =
        vx_shape_input_descriptor(request, "a");
    const VxShapeTensorDescriptor* right =
        vx_shape_input_descriptor(request, "b");
    if (vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_unquantized(mask, "operator input 'mask'", error) ||
        vx_shape_unquantized(left, "operator input 'a'", error) ||
        vx_shape_unquantized(right, "operator input 'b'", error) ||
        vx_shape_rank_range(left, 0, 8, "operator data inputs", error)) {
        return -1;
    }
    if (mask->dtype != VX_DTYPE_F32 && mask->dtype != VX_DTYPE_I32) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator input 'mask'.dtype",
                             "must be float32 or int32.");
    }
    if (left->dtype != right->dtype) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator data inputs", "must have the same dtype.");
    }
    if (left->dtype != VX_DTYPE_F32 && left->dtype != VX_DTYPE_I32) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator input 'a'.dtype",
                             "must be float32 or int32.");
    }
    if (!vx_shape_same(mask, left) || !vx_shape_same(left, right)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator inputs",
                             "Mask requires three exactly equal shapes; it never broadcasts.");
    }
    return vx_shape_clone_output(candidate, left->shape, left->rank,
                                 left->dtype, NULL, error);
}

static int vx_shape_infer_extended_batch_norm2d(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"eps"};
    static const char* const PARAMETER_NAMES[] = {
        "weight", "bias", "running_mean", "running_var"
    };
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeParam* eps;
    double value;
    size_t index;
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank(input, 0, 4, 1, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 1, error)) return -1;
    for (index = 0; index < 4; ++index) {
        char path[224];
        const VxShapeTensorDescriptor* parameter =
            vx_shape_input_descriptor(request, PARAMETER_NAMES[index]);
        snprintf(path, sizeof(path), "operator input '%s'", PARAMETER_NAMES[index]);
        if (vx_shape_float_tensor(parameter, path, error)) return -1;
        snprintf(path, sizeof(path), "operator input '%s'.shape", PARAMETER_NAMES[index]);
        if (vx_shape_vector(parameter, input->shape[3], path, error)) return -1;
    }
    eps = vx_shape_find_param(request, "eps");
    if (eps) {
        if (vx_shape_number(eps, "operator params.eps", &value, error)) return -1;
        if (!isfinite(value) || value <= 0.0) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.eps",
                                 "must be finite and positive.");
        }
    }
    return vx_shape_clone_output(candidate, input->shape, input->rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_extended_interpolate1d(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"size"};
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    uint64_t size;
    uint64_t output_shape[3];
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank(input, 0, 3, 1, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_positive_integer_param(request, "size", 0, &size, error)) {
        return -1;
    }
    output_shape[0] = input->shape[0];
    output_shape[1] = input->shape[1];
    output_shape[2] = size;
    return vx_shape_clone_output(candidate, output_shape, 3,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_extended_vision_profile(
        VxOperatorKind kind,
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    uint64_t output_shape[3];
    uint64_t doubled = 0;
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank(input, 0, 4, 1, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, NULL, 0, error)) return -1;
    output_shape[0] = input->shape[0];
    if (kind == VX_OP_PROFILE_X || kind == VX_OP_PROFILE_Y) {
        if (vx_shape_checked_multiply_u64(input->shape[3], 2,
                                          "profile channel extent",
                                          &doubled, error)) return -1;
        output_shape[1] = doubled;
        output_shape[2] = kind == VX_OP_PROFILE_X
            ? input->shape[2] : input->shape[1];
    } else {
        output_shape[1] = input->shape[3];
        output_shape[2] = input->shape[2];
    }
    return vx_shape_clone_output(candidate, output_shape, 3,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_extended_dropout(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"ratio", "p", "probability", "seed"};
    static const char* const PROBABILITY_NAMES[] = {"ratio", "p", "probability"};
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeParam* selected = NULL;
    const VxShapeParam* seed;
    size_t index;
    double value = 0.5;
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank_range(input, 0, 8, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 4, error)) return -1;
    for (index = 0; index < 3; ++index) {
        const VxShapeParam* current =
            vx_shape_find_param(request, PROBABILITY_NAMES[index]);
        if (current && selected) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params",
                                 "must specify at most one Dropout probability field.");
        }
        if (current) selected = current;
    }
    if (selected && vx_shape_number(selected, "operator params.ratio", &value,
                                    error)) return -1;
    if (!isfinite(value) || value < 0.0 || value >= 1.0) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.ratio",
                             "must be finite and in [0, 1).");
    }
    seed = vx_shape_find_param(request, "seed");
    if (seed) {
        if (vx_shape_number(seed, "operator params.seed", &value, error)) return -1;
        if (!vx_shape_is_safe_integer(value) || value < 0.0 ||
            value > 4294967295.0) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.seed",
                                 "must be an unsigned 32-bit integer.");
        }
    }
    return vx_shape_clone_output(candidate, input->shape, input->rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_extended_requantize(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* output;
    const VxShapeQuantization* input_quantization;
    const VxShapeQuantization* output_quantization;
    if (vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_per_tensor_byte(input, "operator input 'input'",
                                 &input_quantization, error) ||
        vx_shape_quantized_declared_output(request, input->shape, input->rank,
                                           &output, error) ||
        vx_shape_per_tensor_byte(output, "declared output 'out'",
                                 &output_quantization, error) ||
        vx_shape_positive_f32_ratio(input_quantization->scale, 1.0f,
                                    output_quantization->scale,
                                    "declared output 'out'.quantization.scale",
                                    error)) {
        return -1;
    }
    return vx_shape_clone_output(candidate, input->shape, input->rank,
                                 output->dtype, output_quantization, error);
}

static int vx_shape_infer_extended_concat2(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis", "sigmoid"};
    VxShapeNamedTensor inputs[2];
    VxShapeParam params[1];
    VxConcreteShapeRequest adapted;
    const VxShapeParam* axis = vx_shape_find_param(request, "axis");
    const VxShapeTensorDescriptor* left =
        vx_shape_input_descriptor(request, "a");
    const VxShapeTensorDescriptor* right =
        vx_shape_input_descriptor(request, "b");
    int sigmoid;
    if (vx_shape_validate_params(request, FIELDS, 2, error) ||
        vx_shape_boolean_param(request, "sigmoid", 0, &sigmoid, error)) {
        return -1;
    }
    if (sigmoid &&
        (vx_shape_float_tensor(left, "operator input 'a'", error) ||
         vx_shape_float_tensor(right, "operator input 'b'", error))) return -1;
    memset(&adapted, 0, sizeof(adapted));
    inputs[0].name = "input0";
    inputs[0].descriptor = *left;
    inputs[1].name = "input1";
    inputs[1].descriptor = *right;
    adapted.inputs = inputs;
    adapted.input_count = 2;
    if (axis) {
        params[0] = *axis;
        adapted.params = params;
        adapted.param_count = 1;
    }
    return vx_shape_infer_concat(&adapted, candidate, error);
}

static int vx_shape_infer_extended_moe_router(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {
        "num_experts", "top_k", "temperature", "normalize"
    };
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight =
        vx_shape_input_descriptor(request, "weight");
    const VxShapeNamedTensor* named_bias = vx_shape_find_input(request, "bias");
    const VxShapeTensorDescriptor* bias =
        named_bias ? &named_bias->descriptor : NULL;
    const VxShapeParam* temperature;
    uint64_t experts;
    uint64_t num_experts;
    uint64_t top_k;
    uint64_t route_shape[8];
    double value;
    int normalize;
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
        vx_shape_rank_range(input, 1, 8, "operator input 'input'.shape", error) ||
        vx_shape_rank(weight, 0, 2, 1, "operator input 'weight'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 4, error)) return -1;
    if (weight->shape[0] != input->shape[input->rank - 1]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'weight'.shape[0]",
                             "must equal the input feature extent.");
    }
    experts = weight->shape[1];
    if (vx_shape_positive_integer_param(request, "num_experts", experts,
                                        &num_experts, error) ||
        vx_shape_positive_integer_param(request, "top_k", 2, &top_k, error) ||
        vx_shape_boolean_param(request, "normalize", 1, &normalize, error)) {
        return -1;
    }
    (void)normalize;
    if (num_experts != experts) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator params.num_experts",
                             "must equal the router weight extent.");
    }
    if (top_k > experts) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.top_k",
                             "must not exceed the expert count.");
    }
    temperature = vx_shape_find_param(request, "temperature");
    if (temperature) {
        if (vx_shape_number(temperature, "operator params.temperature", &value,
                            error)) return -1;
        if (!isfinite(value) || value <= 0.0) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.temperature",
                                 "must be finite and positive.");
        }
    }
    if (bias &&
        (vx_shape_float_tensor(bias, "operator input 'bias'", error) ||
         vx_shape_vector(bias, experts, "operator input 'bias'.shape", error))) {
        return -1;
    }
    if (input->rank > 1) {
        memcpy(route_shape, input->shape,
               (input->rank - 1) * sizeof(*route_shape));
    }
    route_shape[input->rank - 1] = top_k;
    if (vx_shape_result_allocate(candidate, 2, error) ||
        vx_shape_clone_output_at(candidate, 0, "indices", route_shape,
                                 input->rank, VX_DTYPE_F32, NULL, error) ||
        vx_shape_clone_output_at(candidate, 1, "weights", route_shape,
                                 input->rank, VX_DTYPE_F32, NULL, error)) {
        vx_shape_contract_result_clear(candidate);
        return -1;
    }
    return 0;
}

static int vx_shape_infer_extended_moe_linear(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight =
        vx_shape_input_descriptor(request, "expert_weight");
    const VxShapeTensorDescriptor* indices =
        vx_shape_input_descriptor(request, "route_indices");
    const VxShapeTensorDescriptor* routes =
        vx_shape_input_descriptor(request, "route_weights");
    const VxShapeNamedTensor* named_bias =
        vx_shape_find_input(request, "expert_bias");
    const VxShapeTensorDescriptor* bias =
        named_bias ? &named_bias->descriptor : NULL;
    uint64_t output_shape[8];
    uint64_t experts;
    uint64_t output_feature;
    size_t axis;
    if (vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_float_tensor(weight, "operator input 'expert_weight'", error) ||
        vx_shape_float_tensor(indices, "operator input 'route_indices'", error) ||
        vx_shape_float_tensor(routes, "operator input 'route_weights'", error) ||
        vx_shape_rank_range(input, 1, 8, "operator input 'input'.shape", error) ||
        vx_shape_rank(weight, 0, 3, 1,
                      "operator input 'expert_weight'.shape", error)) return -1;
    experts = weight->shape[0];
    output_feature = weight->shape[2];
    if (weight->shape[1] != input->shape[input->rank - 1]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'expert_weight'.shape[1]",
                             "must equal the input feature extent.");
    }
    if (!vx_shape_same(indices, routes) || indices->rank != input->rank) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator route inputs",
                             "must share the input token prefix and one common top-k axis.");
    }
    for (axis = 0; axis + 1 < input->rank; ++axis) {
        if (indices->shape[axis] != input->shape[axis]) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator route inputs",
                                 "must share the input token prefix and one common top-k axis.");
        }
    }
    if (indices->shape[indices->rank - 1] > experts) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'route_indices'.shape",
                             "top-k extent must not exceed the expert count.");
    }
    if (bias) {
        if (vx_shape_float_tensor(bias, "operator input 'expert_bias'", error)) {
            return -1;
        }
        if (bias->rank != 2 || bias->shape[0] != experts ||
            bias->shape[1] != output_feature) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator input 'expert_bias'.shape",
                                 "must equal [experts, output_feature].");
        }
    }
    if (input->rank > 1) {
        memcpy(output_shape, input->shape,
               (input->rank - 1) * sizeof(*output_shape));
    }
    output_shape[input->rank - 1] = output_feature;
    return vx_shape_clone_output(candidate, output_shape, input->rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_extended_cross_attention(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"heads"};
    const VxShapeTensorDescriptor* query =
        vx_shape_input_descriptor(request, "q");
    const VxShapeTensorDescriptor* kv =
        vx_shape_input_descriptor(request, "kv");
    const VxShapeTensorDescriptor* weight =
        vx_shape_input_descriptor(request, "weight");
    const VxShapeNamedTensor* named_scale = vx_shape_find_input(request, "scale");
    const VxShapeNamedTensor* named_bias = vx_shape_find_input(request, "bias");
    const VxShapeTensorDescriptor* scale =
        named_scale ? &named_scale->descriptor : NULL;
    const VxShapeTensorDescriptor* bias =
        named_bias ? &named_bias->descriptor : NULL;
    uint64_t feature;
    uint64_t projection = 0;
    uint64_t heads;
    size_t feature_axis;
    if (vx_shape_float_tensor(query, "operator input 'q'", error) ||
        vx_shape_float_tensor(kv, "operator input 'kv'", error) ||
        vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
        vx_shape_rank_range(query, 2, 3, "operator input 'q'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 1, error)) return -1;
    if (kv->rank != query->rank) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator input 'kv'.shape",
                             "must have the same rank as q.");
    }
    feature_axis = query->rank - 1;
    feature = query->shape[feature_axis];
    if (kv->shape[feature_axis] != feature ||
        (query->rank == 3 && kv->shape[0] != query->shape[0])) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'kv'.shape",
                             "must share q batch and feature dimensions.");
    }
    if (vx_shape_checked_multiply_u64(feature, 3,
                                      "CrossAttention projection extent",
                                      &projection, error)) return -1;
    if (weight->rank != 2 || weight->shape[0] != projection ||
        weight->shape[1] != feature) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'weight'.shape",
                             "must equal [3*feature, feature].");
    }
    if ((scale &&
         (vx_shape_float_tensor(scale, "operator input 'scale'", error) ||
          vx_shape_vector(scale, projection, "operator input 'scale'.shape", error))) ||
        (bias &&
         (vx_shape_float_tensor(bias, "operator input 'bias'", error) ||
          vx_shape_vector(bias, projection, "operator input 'bias'.shape", error)))) {
        return -1;
    }
    if (vx_shape_positive_integer_param(request, "heads", 8, &heads, error)) {
        return -1;
    }
    if (feature % heads) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.heads",
                             "must divide the feature extent.");
    }
    return vx_shape_clone_output(candidate, query->shape, query->rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_scan_bc_mode(const VxShapeTensorDescriptor* descriptor,
                                 const VxShapeTensorDescriptor* input,
                                 uint64_t state_width) {
    size_t rank = input->rank;
    uint64_t batch = rank == 3 ? input->shape[0] : 1;
    uint64_t sequence = input->shape[rank - 2];
    if (descriptor->rank == 1 && descriptor->shape[0] == state_width) return 0;
    if (descriptor->rank == 2 && descriptor->shape[0] == sequence &&
        descriptor->shape[1] == state_width) return 1;
    if (rank == 3 && descriptor->rank == 3 &&
        descriptor->shape[0] == batch && descriptor->shape[1] == sequence &&
        descriptor->shape[2] == state_width) return 2;
    return -1;
}

static int vx_shape_infer_extended_scan(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"delta_softplus"};
    static const char* const REQUIRED[] = {"input", "delta", "A", "B", "C"};
    static const char* const OPTIONAL[] = {"D", "z", "initial_state"};
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* delta =
        vx_shape_input_descriptor(request, "delta");
    const VxShapeTensorDescriptor* a = vx_shape_input_descriptor(request, "A");
    const VxShapeTensorDescriptor* b = vx_shape_input_descriptor(request, "B");
    const VxShapeTensorDescriptor* c = vx_shape_input_descriptor(request, "C");
    uint64_t batch;
    uint64_t channels;
    uint64_t state_width;
    uint64_t state_shape[3];
    size_t index;
    int delta_softplus;
    for (index = 0; index < 5; ++index) {
        char path[224];
        const VxShapeTensorDescriptor* descriptor =
            vx_shape_input_descriptor(request, REQUIRED[index]);
        snprintf(path, sizeof(path), "operator input '%s'", REQUIRED[index]);
        if (vx_shape_float_tensor(descriptor, path, error)) return -1;
    }
    for (index = 0; index < 3; ++index) {
        const VxShapeNamedTensor* named =
            vx_shape_find_input(request, OPTIONAL[index]);
        if (named) {
            char path[224];
            snprintf(path, sizeof(path), "operator input '%s'", OPTIONAL[index]);
            if (vx_shape_float_tensor(&named->descriptor, path, error)) return -1;
        }
    }
    if (vx_shape_rank_range(input, 2, 3, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_boolean_param(request, "delta_softplus", 1,
                               &delta_softplus, error)) return -1;
    (void)delta_softplus;
    if (!vx_shape_same(input, delta)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'delta'.shape",
                             "must equal the input shape.");
    }
    batch = input->rank == 3 ? input->shape[0] : 1;
    channels = input->shape[input->rank - 1];
    if (a->rank != 2 || a->shape[0] != channels) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'A'.shape",
                             "must be [channels, state_width].");
    }
    state_width = a->shape[1];
    if (vx_shape_scan_bc_mode(b, input, state_width) < 0 ||
        vx_shape_scan_bc_mode(c, input, state_width) < 0) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator B/C inputs",
                             "must use [N], [S,N], or [B,S,N] selective-scan layout.");
    }
    state_shape[0] = batch;
    state_shape[1] = channels;
    state_shape[2] = state_width;
    {
        const VxShapeNamedTensor* named_d = vx_shape_find_input(request, "D");
        const VxShapeNamedTensor* named_z = vx_shape_find_input(request, "z");
        const VxShapeNamedTensor* named_initial =
            vx_shape_find_input(request, "initial_state");
        VxShapeTensorDescriptor expected;
        if (named_d && vx_shape_vector(&named_d->descriptor, channels,
                                       "operator input 'D'.shape", error)) {
            return -1;
        }
        if (named_z && !vx_shape_same(&named_z->descriptor, input)) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator input 'z'.shape",
                                 "must equal the input shape.");
        }
        memset(&expected, 0, sizeof(expected));
        expected.rank = 3;
        expected.shape = state_shape;
        if (named_initial && !vx_shape_same(&named_initial->descriptor, &expected)) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator input 'initial_state'.shape",
                                 "must equal [batch, channels, state_width].");
        }
    }
    if (vx_shape_result_allocate(candidate, 2, error) ||
        vx_shape_clone_output_at(candidate, 0, "out", input->shape, input->rank,
                                 VX_DTYPE_F32, NULL, error) ||
        vx_shape_clone_output_at(candidate, 1, "state", state_shape, 3,
                                 VX_DTYPE_F32, NULL, error)) {
        vx_shape_contract_result_clear(candidate);
        return -1;
    }
    return 0;
}
