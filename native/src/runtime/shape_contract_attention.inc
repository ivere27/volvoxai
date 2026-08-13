/* Shape contracts for SDPA, CrossSDPA and RoPE.
 * Mirrors tests/operator_shape_contract_attention_vectors.json. */

static int vx_shape_infer_sdpa(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* qkv =
        vx_shape_input_descriptor(request, "qkv");
    const VxShapeTensorDescriptor* mask =
        vx_shape_input_descriptor(request, "mask");
    uint64_t output_shape[3];
    uint64_t packed_feature;
    uint64_t feature;
    uint64_t sequence;
    uint64_t batch;
    uint64_t heads;
    size_t axis;
    if (vx_shape_float_tensor(qkv, "operator input 'qkv'", error) ||
        vx_shape_attention_rank(qkv, "operator input 'qkv'.shape", error)) {
        return -1;
    }
    packed_feature = qkv->shape[qkv->rank - 1];
    if (packed_feature % 3) {
        char path[224];
        snprintf(path, sizeof(path), "operator input 'qkv'.shape[%zu]",
                 qkv->rank - 1);
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             path,
                             "must be exactly 3 times the output feature extent.");
    }
    feature = packed_feature / 3;
    if (vx_shape_attention_parameters(request, &heads, error)) return -1;
    if (feature % heads) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator params.heads",
                             "must divide the output feature extent.");
    }
    sequence = qkv->shape[qkv->rank - 2];
    batch = qkv->rank == 2 ? 1 : qkv->shape[0];
    if (vx_shape_attention_mask(mask, batch, sequence, sequence, error)) return -1;
    for (axis = 0; axis < qkv->rank; ++axis) output_shape[axis] = qkv->shape[axis];
    output_shape[qkv->rank - 1] = feature;
    return vx_shape_clone_output(candidate, output_shape, qkv->rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_cross_sdpa(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* query =
        vx_shape_input_descriptor(request, "q");
    const VxShapeTensorDescriptor* key =
        vx_shape_input_descriptor(request, "k");
    const VxShapeTensorDescriptor* value =
        vx_shape_input_descriptor(request, "v");
    const VxShapeTensorDescriptor* mask =
        vx_shape_input_descriptor(request, "mask");
    size_t sequence_axis;
    size_t feature_axis;
    uint64_t batch;
    uint64_t heads;
    uint64_t feature;
    if (vx_shape_float_tensor(query, "operator input 'q'", error) ||
        vx_shape_float_tensor(key, "operator input 'k'", error) ||
        vx_shape_float_tensor(value, "operator input 'v'", error) ||
        vx_shape_attention_rank(query, "operator input 'q'.shape", error) ||
        vx_shape_attention_rank(key, "operator input 'k'.shape", error) ||
        vx_shape_attention_rank(value, "operator input 'v'.shape", error)) {
        return -1;
    }
    if (key->rank != query->rank || value->rank != query->rank) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator inputs",
                             "q, k, and v must have the same rank.");
    }
    sequence_axis = query->rank - 2;
    feature_axis = query->rank - 1;
    if (query->rank == 3 &&
        (key->shape[0] != query->shape[0] ||
         value->shape[0] != query->shape[0])) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator inputs",
                             "q, k, and v batch extents must match.");
    }
    if (value->shape[sequence_axis] != key->shape[sequence_axis]) {
        char path[224];
        snprintf(path, sizeof(path), "operator input 'v'.shape[%zu]",
                 sequence_axis);
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             path, "must equal the key sequence extent.");
    }
    if (key->shape[feature_axis] != query->shape[feature_axis] ||
        value->shape[feature_axis] != query->shape[feature_axis]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator inputs",
                             "q, k, and v feature extents must match.");
    }
    feature = query->shape[feature_axis];
    if (vx_shape_attention_parameters(request, &heads, error)) return -1;
    if (feature % heads) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator params.heads",
                             "must divide the feature extent.");
    }
    batch = query->rank == 2 ? 1 : query->shape[0];
    if (vx_shape_attention_mask(mask, batch, query->shape[sequence_axis],
                                key->shape[sequence_axis], error)) return -1;
    return vx_shape_clone_output(candidate, query->shape, query->rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_rope_parameters(
        const VxConcreteShapeRequest* request,
        uint64_t width,
        uint64_t sequence,
        uint64_t* rotary_dimension,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {
        "rotary_dim", "theta", "position_offset", "interleaved"
    };
    const VxShapeParam* param;
    uint64_t position_offset = 0;
    double value;
    float canonical;
    if (vx_shape_validate_params(request, FIELDS, 4, error)) return -1;
    *rotary_dimension = width;
    param = vx_shape_find_param(request, "rotary_dim");
    if (param) {
        if (param->kind != VX_SHAPE_PARAM_NUMBER ||
            !vx_shape_is_safe_integer(param->value.number) ||
            param->value.number <= 0.0 ||
            ((uint64_t)param->value.number % 2)) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.rotary_dim",
                                 "must be a positive even safe integer.");
        }
        *rotary_dimension = (uint64_t)param->value.number;
    }
    if (*rotary_dimension % 2) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.rotary_dim",
                             "resolved rotary width must be even.");
    }
    if (*rotary_dimension > width) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator params.rotary_dim",
                             "must not exceed the feature extent.");
    }
    param = vx_shape_find_param(request, "theta");
    if (param && param->kind != VX_SHAPE_PARAM_NUMBER) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.theta",
                             "must be positive and representable as float32.");
    }
    value = param ? param->value.number : 10000.0;
    if (!isfinite(value) || value <= 0.0) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.theta",
                             "must be positive and representable as float32.");
    }
    canonical = (float)value;
    if (!isfinite(canonical) || canonical <= 0.0f) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.theta",
                             "must be positive and representable as float32.");
    }
    param = vx_shape_find_param(request, "position_offset");
    if (param) {
        if (param->kind != VX_SHAPE_PARAM_NUMBER ||
            !vx_shape_is_safe_integer(param->value.number) ||
            param->value.number < 0.0 ||
            param->value.number > (double)INT32_MAX) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.position_offset",
                                 "must be a non-negative int32 integer.");
        }
        position_offset = (uint64_t)param->value.number;
    }
    if (sequence - 1 > (uint64_t)INT32_MAX - position_offset) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.position_offset",
                             "plus the maximum sequence index must fit int32.");
    }
    param = vx_shape_find_param(request, "interleaved");
    if (param && param->kind != VX_SHAPE_PARAM_BOOLEAN) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.interleaved",
                             "must be boolean.");
    }
    return 0;
}

static int vx_shape_position_ids(
        const VxShapeTensorDescriptor* positions,
        size_t rank,
        uint64_t batch,
        uint64_t sequence,
        VxShapeContractError* error) {
    int valid;
    if (!positions) return 0;
    if (positions->dtype != VX_DTYPE_I32) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator input 'position_ids'.dtype",
                             "must be 'int32'.");
    }
    if (vx_shape_unquantized(positions, "operator input 'position_ids'", error)) {
        return -1;
    }
    valid = (positions->rank == 1 && positions->shape[0] == sequence) ||
            (rank == 3 && positions->rank == 2 &&
             positions->shape[0] == batch && positions->shape[1] == sequence);
    if (!valid) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'position_ids'.shape",
                             "must have shape [S], or [B,S] for a rank-3 input.");
    }
    return 0;
}

static int vx_shape_infer_rope(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* positions =
        vx_shape_input_descriptor(request, "position_ids");
    uint64_t sequence;
    uint64_t batch;
    uint64_t rotary_dimension;
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_attention_rank(input, "operator input 'input'.shape", error)) {
        return -1;
    }
    sequence = input->shape[input->rank - 2];
    batch = input->rank == 2 ? 1 : input->shape[0];
    if (vx_shape_rope_parameters(request, input->shape[input->rank - 1],
                                 sequence, &rotary_dimension, error) ||
        vx_shape_position_ids(positions, input->rank, batch, sequence, error)) {
        return -1;
    }
    return vx_shape_clone_output(candidate, input->shape, input->rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_attention_kind(
        VxOperatorKind kind,
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    switch (kind) {
        case VX_OP_SDPA:
            return vx_shape_infer_sdpa(request, candidate, error);
        case VX_OP_CROSS_SDPA:
            return vx_shape_infer_cross_sdpa(request, candidate, error);
        case VX_OP_ROPE:
            return vx_shape_infer_rope(request, candidate, error);
        default:
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_UNKNOWN_OPERATOR,
                                 "operator",
                                 "has no implemented attention-family contract.");
    }
}

static int vx_shape_quantized_kind(VxOperatorKind kind) {
    switch (kind) {
        case VX_OP_Q_LINEAR:
        case VX_OP_Q_MATMUL:
        case VX_OP_Q_GEMM:
        case VX_OP_Q_BATCH_MATMUL:
        case VX_OP_Q_CONV_2D:
        case VX_OP_Q_ADD:
        case VX_OP_Q_EMBEDDING:
        case VX_OP_Q_GELU:
        case VX_OP_Q_SILU:
        case VX_OP_Q_LAYER_NORM:
        case VX_OP_Q_GROUP_NORM:
        case VX_OP_Q_MASKED_MEAN:
        case VX_OP_Q_SDPA:
        case VX_OP_Q_ARG_MAX:
            return 1;
        default:
            return 0;
    }
}

static int vx_shape_per_tensor_byte(
        const VxShapeTensorDescriptor* descriptor,
        const char* path,
        const VxShapeQuantization** quantization,
        VxShapeContractError* error) {
    char field_path[224];
    if (!descriptor) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR,
                             path, "must be an object.");
    }
    if (descriptor->dtype != VX_DTYPE_I8 && descriptor->dtype != VX_DTYPE_U8) {
        vx_shape_field_path(field_path, sizeof(field_path), path, "dtype");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             field_path, "must be 'int8' or 'uint8'.");
    }
    if (descriptor->quantization.scheme != VX_SHAPE_QUANTIZATION_PER_TENSOR) {
        vx_shape_field_path(field_path, sizeof(field_path), path, "quantization");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                             field_path, "must be per_tensor.");
    }
    if (quantization) *quantization = &descriptor->quantization;
    return 0;
}

static int vx_shape_axis_zero_byte_weight(
        const VxShapeTensorDescriptor* descriptor,
        const char* path,
        const VxShapeQuantization** quantization,
        VxShapeContractError* error) {
    char field_path[224];
    if (!descriptor) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR,
                             path, "must be an object.");
    }
    if (descriptor->dtype != VX_DTYPE_I8 && descriptor->dtype != VX_DTYPE_U8) {
        vx_shape_field_path(field_path, sizeof(field_path), path, "dtype");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             field_path, "must be 'int8' or 'uint8'.");
    }
    if (descriptor->quantization.scheme != VX_SHAPE_QUANTIZATION_PER_AXIS ||
        descriptor->quantization.axis != 0) {
        vx_shape_field_path(field_path, sizeof(field_path), path, "quantization");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                             field_path,
                             "must be per_axis along output/row axis 0.");
    }
    if (quantization) *quantization = &descriptor->quantization;
    return 0;
}

static int vx_shape_i32_tensor(const VxShapeTensorDescriptor* descriptor,
                               const char* path,
                               VxShapeContractError* error) {
    char field_path[224];
    if (!descriptor) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR,
                             path, "must be an object.");
    }
    if (descriptor->dtype != VX_DTYPE_I32) {
        vx_shape_field_path(field_path, sizeof(field_path), path, "dtype");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             field_path, "must be 'int32'.");
    }
    return vx_shape_unquantized(descriptor, path, error);
}

static int vx_shape_quantized_declared_output(
        const VxConcreteShapeRequest* request,
        const uint64_t* shape,
        size_t rank,
        const VxShapeTensorDescriptor** output,
        VxShapeContractError* error) {
    VxShapeTensorDescriptor expected;
    if (vx_shape_declared_output(request, output, error) ||
        vx_shape_per_tensor_byte(*output, "declared output 'out'", NULL,
                                 error)) {
        return -1;
    }
    memset(&expected, 0, sizeof(expected));
    expected.rank = rank;
    expected.shape = shape;
    if (!vx_shape_same(&expected, *output)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "declared output 'out'.shape",
                             "must equal the inferred output shape.");
    }
    return 0;
}

static uint64_t vx_shape_centered_magnitude(VxDataType dtype,
                                            int32_t zero_point) {
    int32_t minimum = dtype == VX_DTYPE_I8 ? -128 : 0;
    int32_t maximum = dtype == VX_DTYPE_I8 ? 127 : 255;
    uint64_t below = (uint64_t)((int64_t)zero_point - minimum);
    uint64_t above = (uint64_t)((int64_t)maximum - zero_point);
    return below > above ? below : above;
}

static uint64_t vx_shape_maximum_weight_magnitude(
        VxDataType dtype,
        const VxShapeQuantization* quantization) {
    uint64_t maximum = 0;
    size_t index;
    for (index = 0; index < quantization->count; ++index) {
        uint64_t magnitude = vx_shape_centered_magnitude(
            dtype, quantization->zero_points[index]);
        if (magnitude > maximum) maximum = magnitude;
    }
    return maximum;
}

static int vx_shape_i32_accumulator_bound(uint64_t terms,
                                          uint64_t left_magnitude,
                                          uint64_t right_magnitude,
                                          const char* path,
                                          VxShapeContractError* error) {
    uint64_t maximum = 0;
    if (vx_shape_checked_multiply_u64(terms, left_magnitude, path,
                                      &maximum, error) ||
        vx_shape_checked_multiply_u64(maximum, right_magnitude, path,
                                      &maximum, error)) {
        return -1;
    }
    if (maximum > (uint64_t)INT32_MAX) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DOMAIN,
                             path,
                             "may require an I32 accumulator magnitude of %llu.",
                             (unsigned long long)maximum);
    }
    return 0;
}

static int vx_shape_i32_sum_bound(uint64_t terms,
                                  uint64_t magnitude,
                                  const char* path,
                                  VxShapeContractError* error) {
    uint64_t maximum = 0;
    if (vx_shape_checked_multiply_u64(terms, magnitude, path,
                                      &maximum, error)) return -1;
    if (maximum > (uint64_t)INT32_MAX) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DOMAIN,
                             path,
                             "may require an I32 accumulator magnitude of %llu.",
                             (unsigned long long)maximum);
    }
    return 0;
}

static int vx_shape_positive_f32_ratio(float left,
                                       float right,
                                       float divisor,
                                       const char* path,
                                       VxShapeContractError* error) {
    float product = left * right;
    float multiplier = product / divisor;
    if (!isfinite(multiplier) || multiplier <= 0.0f) {
        return vx_shape_fail(
            error, VX_SHAPE_CONTRACT_ERROR_UNSAFE_QUANTIZATION_TRANSFORM,
            path,
            "requantization multiplier must be positive and representable as float32.");
    }
    return 0;
}

static int vx_shape_validate_q_dense_multipliers(
        const VxShapeQuantization* input,
        const VxShapeQuantization* weight,
        const VxShapeQuantization* output,
        VxShapeContractError* error) {
    size_t index;
    char path[256];
    for (index = 0; index < weight->count; ++index) {
        snprintf(path, sizeof(path),
                 "operator input 'weight'.quantization.scales[%zu]", index);
        if (vx_shape_positive_f32_ratio(input->scale, weight->scales[index],
                                        output->scale, path, error)) return -1;
    }
    return 0;
}

