/* Shape contracts for the W8A8 quantized operators, and the kind dispatcher
 * that routes to them.  Mirrors tests/operator_shape_contract_quantized_vectors.json. */

static int vx_shape_infer_q_dense(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight =
        vx_shape_input_descriptor(request, "weight");
    const VxShapeTensorDescriptor* bias =
        vx_shape_input_descriptor(request, "bias");
    const VxShapeTensorDescriptor* output;
    const VxShapeQuantization* input_quantization;
    const VxShapeQuantization* weight_quantization;
    uint64_t* output_shape;
    uint64_t output_feature;
    uint64_t contracted;
    size_t rank;
    int status;
    if (vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_rank(input, 1, 0, 0,
                      "operator input 'input'.shape", error) ||
        vx_shape_rank(weight, 0, 2, 1,
                      "operator input 'weight'.shape", error) ||
        vx_shape_per_tensor_byte(input, "operator input 'input'",
                                 &input_quantization, error) ||
        vx_shape_axis_zero_byte_weight(weight, "operator input 'weight'",
                                       &weight_quantization, error) ||
        vx_shape_i32_tensor(bias, "operator input 'bias'", error)) {
        return -1;
    }
    output_feature = weight->shape[0];
    contracted = weight->shape[1];
    if (input->shape[input->rank - 1] != contracted) {
        char path[224];
        snprintf(path, sizeof(path), "operator input 'input'.shape[%zu]",
                 input->rank - 1);
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             path,
                             "must equal the fixed contracted weight extent.");
    }
    if (vx_shape_vector(bias, output_feature,
                        "operator input 'bias'.shape", error)) return -1;
    rank = input->rank;
    if (rank > SIZE_MAX / sizeof(*output_shape)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator output 'out'.shape", "rank is too large.");
    }
    output_shape = (uint64_t*)malloc(rank * sizeof(*output_shape));
    if (!output_shape) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator output 'out'.shape", "allocation failed.");
    }
    memcpy(output_shape, input->shape, rank * sizeof(*output_shape));
    output_shape[rank - 1] = output_feature;
    status = vx_shape_quantized_declared_output(request, output_shape, rank,
                                                 &output, error);
    if (!status) {
        status = vx_shape_validate_q_dense_multipliers(
            input_quantization, weight_quantization, &output->quantization,
            error);
    }
    if (!status) {
        char path[224];
        snprintf(path, sizeof(path), "operator input 'input'.shape[%zu]",
                 input->rank - 1);
        status = vx_shape_i32_accumulator_bound(
            contracted,
            vx_shape_centered_magnitude(input->dtype,
                                        input_quantization->zero_point),
            vx_shape_maximum_weight_magnitude(weight->dtype,
                                               weight_quantization),
            path, error);
    }
    if (!status) {
        status = vx_shape_clone_output(candidate, output_shape, rank,
                                       output->dtype, &output->quantization,
                                       error);
    }
    free(output_shape);
    return status;
}

static int vx_shape_infer_q_batch_matmul(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* left =
        vx_shape_input_descriptor(request, "a");
    const VxShapeTensorDescriptor* right =
        vx_shape_input_descriptor(request, "b");
    const VxShapeTensorDescriptor* output;
    const VxShapeQuantization* left_quantization;
    const VxShapeQuantization* right_quantization;
    uint64_t output_shape[8];
    size_t left_batch;
    size_t right_batch;
    size_t batch_rank;
    size_t axis;
    uint64_t contracted;
    char accumulator_path[224];
    if (vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_rank(left, 2, 0, 0, "operator input 'a'.shape", error) ||
        vx_shape_rank(right, 2, 0, 0, "operator input 'b'.shape", error) ||
        vx_shape_per_tensor_byte(left, "operator input 'a'",
                                 &left_quantization, error) ||
        vx_shape_per_tensor_byte(right, "operator input 'b'",
                                 &right_quantization, error)) return -1;
    if (left->rank > 8 || right->rank > 8) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator inputs",
                             "QBatchMatMul operand ranks must not exceed 8.");
    }
    contracted = left->shape[left->rank - 1];
    snprintf(accumulator_path, sizeof(accumulator_path),
             "operator input 'a'.shape[%zu]", left->rank - 1);
    if (contracted != right->shape[right->rank - 2]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'b'.shape",
                             "contracted matrix dimensions must match.");
    }
    left_batch = left->rank - 2;
    right_batch = right->rank - 2;
    batch_rank = left_batch > right_batch ? left_batch : right_batch;
    for (axis = 0; axis < batch_rank; ++axis) {
        uint64_t left_extent = axis < batch_rank - left_batch
            ? 1 : left->shape[axis - (batch_rank - left_batch)];
        uint64_t right_extent = axis < batch_rank - right_batch
            ? 1 : right->shape[axis - (batch_rank - right_batch)];
        if (left_extent != right_extent && left_extent != 1 && right_extent != 1) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator inputs",
                                 "batch dimensions are not broadcast-compatible.");
        }
        output_shape[axis] = left_extent > right_extent
            ? left_extent : right_extent;
    }
    output_shape[batch_rank] = left->shape[left->rank - 2];
    output_shape[batch_rank + 1] = right->shape[right->rank - 1];
    if (vx_shape_quantized_declared_output(request, output_shape,
                                           batch_rank + 2, &output, error) ||
        vx_shape_positive_f32_ratio(
            left_quantization->scale, right_quantization->scale,
            output->quantization.scale,
            "declared output 'out'.quantization.scale", error) ||
        vx_shape_i32_accumulator_bound(
            contracted,
            vx_shape_centered_magnitude(left->dtype,
                                        left_quantization->zero_point),
            vx_shape_centered_magnitude(right->dtype,
                                        right_quantization->zero_point),
            accumulator_path, error)) {
        return -1;
    }
    return vx_shape_clone_output(candidate, output_shape, batch_rank + 2,
                                 output->dtype, &output->quantization, error);
}

static int vx_shape_infer_q_conv2d(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {
        "stride", "padding", "pads", "dilation", "groups", "relu",
        "data_layout", "weight_layout"
    };
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight =
        vx_shape_input_descriptor(request, "weight");
    const VxShapeTensorDescriptor* bias =
        vx_shape_input_descriptor(request, "bias");
    const VxShapeTensorDescriptor* output;
    const VxShapeQuantization* input_quantization;
    const VxShapeQuantization* weight_quantization;
    const char* data_layout;
    const char* weight_layout;
    uint64_t stride[2];
    uint64_t dilation[2];
    uint64_t pads[4];
    uint64_t groups;
    uint64_t activation;
    uint64_t output_shape[4];
    uint64_t terms = 0;
    if (vx_shape_rank(input, 0, 4, 1,
                      "operator input 'input'.shape", error) ||
        vx_shape_rank(weight, 0, 4, 1,
                      "operator input 'weight'.shape", error) ||
        vx_shape_per_tensor_byte(input, "operator input 'input'",
                                 &input_quantization, error) ||
        vx_shape_axis_zero_byte_weight(weight, "operator input 'weight'",
                                       &weight_quantization, error) ||
        vx_shape_validate_params(request, FIELDS,
                                 sizeof(FIELDS) / sizeof(FIELDS[0]), error) ||
        vx_shape_param_string_default(request, "data_layout", "NHWC",
                                      &data_layout, error) ||
        vx_shape_param_string_default(request, "weight_layout", "OHWI",
                                      &weight_layout, error) ||
        vx_shape_param_pair(request, "stride", 1, 1, 0, stride, error) ||
        vx_shape_spatial_pads(request, 0, pads, error) ||
        vx_shape_param_pair(request, "dilation", 1, 1, 0, dilation, error) ||
        vx_shape_param_scalar(request, "groups", 1, 1, 0, &groups, error) ||
        vx_shape_param_activation(request, &activation, error)) return -1;
    (void)activation;
    if (strcmp(data_layout, "NHWC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NHWC'.");
    }
    if (strcmp(weight_layout, "OHWI")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.weight_layout", "must be 'OHWI'.");
    }
    if (input->shape[3] != weight->shape[3] * groups ||
        weight->shape[0] % groups) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'weight'.shape",
                             "OHWI channels must match input channels and groups.");
    }
    if (bias &&
        (vx_shape_i32_tensor(bias, "operator input 'bias'", error) ||
         vx_shape_vector(bias, weight->shape[0],
                         "operator input 'bias'.shape", error))) return -1;
    output_shape[0] = input->shape[0];
    if (vx_shape_window_output(input->shape[1], weight->shape[1], stride[0],
                               pads[0], pads[2], dilation[0],
                               "operator input 'input'.shape[1]",
                               &output_shape[1], error) ||
        vx_shape_window_output(input->shape[2], weight->shape[2], stride[1],
                               pads[1], pads[3], dilation[1],
                               "operator input 'input'.shape[2]",
                               &output_shape[2], error)) return -1;
    output_shape[3] = weight->shape[0];
    if (vx_shape_quantized_declared_output(request, output_shape, 4,
                                           &output, error) ||
        vx_shape_validate_q_dense_multipliers(
            input_quantization, weight_quantization, &output->quantization,
            error) ||
        vx_shape_checked_multiply_u64(weight->shape[1], weight->shape[2],
                                      "QConv2D accumulator terms", &terms,
                                      error) ||
        vx_shape_checked_multiply_u64(terms, weight->shape[3],
                                      "QConv2D accumulator terms", &terms,
                                      error) ||
        vx_shape_i32_accumulator_bound(
            terms,
            vx_shape_centered_magnitude(input->dtype,
                                        input_quantization->zero_point),
            vx_shape_maximum_weight_magnitude(weight->dtype,
                                               weight_quantization),
            "operator input 'weight'.shape", error)) {
        return -1;
    }
    return vx_shape_clone_output(candidate, output_shape, 4, output->dtype,
                                 &output->quantization, error);
}

static int vx_shape_infer_q_add(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"relu"};
    const VxShapeTensorDescriptor* left =
        vx_shape_input_descriptor(request, "a");
    const VxShapeTensorDescriptor* right =
        vx_shape_input_descriptor(request, "b");
    const VxShapeTensorDescriptor* output;
    const VxShapeQuantization* left_quantization;
    const VxShapeQuantization* right_quantization;
    uint64_t activation;
    if (vx_shape_per_tensor_byte(left, "operator input 'a'",
                                 &left_quantization, error) ||
        vx_shape_per_tensor_byte(right, "operator input 'b'",
                                 &right_quantization, error) ||
        vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_param_activation(request, &activation, error)) return -1;
    (void)activation;
    if (!vx_shape_same(left, right)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator inputs",
                             "QAdd requires exactly equal shapes; Expand is explicit.");
    }
    if (vx_shape_quantized_declared_output(request, left->shape, left->rank,
                                           &output, error) ||
        vx_shape_positive_f32_ratio(
            left_quantization->scale, 1.0f, output->quantization.scale,
            "operator input 'a'.quantization.scale", error) ||
        vx_shape_positive_f32_ratio(
            right_quantization->scale, 1.0f, output->quantization.scale,
            "operator input 'b'.quantization.scale", error)) return -1;
    return vx_shape_clone_output(candidate, left->shape, left->rank,
                                 output->dtype, &output->quantization, error);
}

static int vx_shape_infer_q_embedding(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight =
        vx_shape_input_descriptor(request, "weight");
    const VxShapeTensorDescriptor* output;
    const VxShapeQuantization* weight_quantization;
    uint64_t* output_shape;
    size_t output_rank;
    size_t index;
    int status;
    char path[256];
    if (vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_i32_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank(input, 1, 0, 0,
                      "operator input 'input'.shape", error) ||
        vx_shape_rank(weight, 0, 2, 1,
                      "operator input 'weight'.shape", error) ||
        vx_shape_axis_zero_byte_weight(weight, "operator input 'weight'",
                                       &weight_quantization, error)) return -1;
    if (input->rank == SIZE_MAX ||
        input->rank + 1 > SIZE_MAX / sizeof(*output_shape)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator output 'out'.shape", "rank is too large.");
    }
    output_rank = input->rank + 1;
    output_shape = (uint64_t*)malloc(output_rank * sizeof(*output_shape));
    if (!output_shape) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator output 'out'.shape", "allocation failed.");
    }
    memcpy(output_shape, input->shape, input->rank * sizeof(*output_shape));
    output_shape[input->rank] = weight->shape[1];
    status = vx_shape_quantized_declared_output(request, output_shape,
                                                 output_rank, &output, error);
    for (index = 0; !status && index < weight_quantization->count; ++index) {
        snprintf(path, sizeof(path),
                 "operator input 'weight'.quantization.scales[%zu]", index);
        status = vx_shape_positive_f32_ratio(
            weight_quantization->scales[index], 1.0f,
            output->quantization.scale, path, error);
    }
    if (!status) {
        status = vx_shape_clone_output(candidate, output_shape, output_rank,
                                       output->dtype, &output->quantization,
                                       error);
    }
    free(output_shape);
    return status;
}

static int vx_shape_infer_q_activation(
        VxOperatorKind kind,
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const APPROXIMATE[] = {"approximate"};
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* output;
    const VxShapeParam* param;
    if (vx_shape_per_tensor_byte(input, "operator input 'input'", NULL,
                                 error)) return -1;
    if (kind == VX_OP_Q_GELU) {
        if (vx_shape_validate_params(request, APPROXIMATE, 1, error)) return -1;
        param = vx_shape_find_param(request, "approximate");
        if (param && (param->kind != VX_SHAPE_PARAM_STRING ||
                      !param->value.string ||
                      strcmp(param->value.string, "none"))) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.approximate",
                                 "must be 'none'.");
        }
    } else if (vx_shape_validate_params(request, NULL, 0, error)) {
        return -1;
    }
    if (vx_shape_quantized_declared_output(request, input->shape, input->rank,
                                           &output, error)) return -1;
    return vx_shape_clone_output(candidate, input->shape, input->rank,
                                 output->dtype, &output->quantization, error);
}

static int vx_shape_infer_q_layer_norm(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight =
        vx_shape_input_descriptor(request, "weight");
    const VxShapeTensorDescriptor* bias =
        vx_shape_input_descriptor(request, "bias");
    const VxShapeTensorDescriptor* output;
    const VxShapeParam* eps;
    uint64_t feature;
    if (vx_shape_rank(input, 1, 0, 0,
                      "operator input 'input'.shape", error) ||
        vx_shape_per_tensor_byte(input, "operator input 'input'", NULL,
                                 error) ||
        vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
        vx_shape_float_tensor(bias, "operator input 'bias'", error)) return -1;
    feature = input->shape[input->rank - 1];
    eps = vx_shape_find_param(request, "eps");
    if (eps) {
        double value;
        float canonical;
        if (vx_shape_number(eps, "operator params.eps", &value, error)) return -1;
        canonical = (float)value;
        if (!isfinite(value) || value <= 0.0 ||
            !isfinite(canonical) || canonical <= 0.0f) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.eps",
                                 "must be positive and representable as float32.");
        }
    }
    if (vx_shape_vector(weight, feature,
                        "operator input 'weight'.shape", error) ||
        vx_shape_vector(bias, feature,
                        "operator input 'bias'.shape", error) ||
        vx_shape_normalization_params(request, feature, error) ||
        vx_shape_quantized_declared_output(request, input->shape, input->rank,
                                           &output, error)) return -1;
    return vx_shape_clone_output(candidate, input->shape, input->rank,
                                 output->dtype, &output->quantization, error);
}

static int vx_shape_infer_q_group_norm(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"num_groups", "eps", "data_layout"};
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight =
        vx_shape_input_descriptor(request, "weight");
    const VxShapeTensorDescriptor* bias =
        vx_shape_input_descriptor(request, "bias");
    const VxShapeTensorDescriptor* output;
    const VxShapeParam* param;
    const char* data_layout;
    uint64_t channels;
    double value;
    if (vx_shape_rank(input, 0, 4, 1,
                      "operator input 'input'.shape", error) ||
        vx_shape_per_tensor_byte(input, "operator input 'input'", NULL,
                                 error) ||
        vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
        vx_shape_float_tensor(bias, "operator input 'bias'", error)) return -1;
    channels = input->shape[3];
    if (vx_shape_vector(weight, channels,
                        "operator input 'weight'.shape", error) ||
        vx_shape_vector(bias, channels,
                        "operator input 'bias'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 3, error) ||
        vx_shape_param_string_default(request, "data_layout", "NHWC",
                                      &data_layout, error)) return -1;
    if (strcmp(data_layout, "NHWC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NHWC'.");
    }
    param = vx_shape_find_param(request, "eps");
    if (param) {
        float canonical;
        if (vx_shape_number(param, "operator params.eps", &value, error)) return -1;
        canonical = (float)value;
        if (!isfinite(value) || value <= 0.0 ||
            !isfinite(canonical) || canonical <= 0.0f) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.eps",
                                 "must be positive and representable as float32.");
        }
    }
    param = vx_shape_find_param(request, "num_groups");
    if (!param || param->kind != VX_SHAPE_PARAM_NUMBER ||
        !vx_shape_is_safe_integer(param->value.number) ||
        param->value.number <= 0.0) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.num_groups",
                             "must be a positive safe integer.");
    }
    if (channels % (uint64_t)param->value.number) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator params.num_groups",
                             "must divide the channel extent.");
    }
    if (vx_shape_quantized_declared_output(request, input->shape, input->rank,
                                           &output, error)) return -1;
    return vx_shape_clone_output(candidate, input->shape, input->rank,
                                 output->dtype, &output->quantization, error);
}

static int vx_shape_infer_q_masked_mean(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* mask =
        vx_shape_input_descriptor(request, "mask");
    const VxShapeTensorDescriptor* output;
    const VxShapeQuantization* input_quantization;
    uint64_t output_shape[2];
    if (vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_rank(input, 0, 3, 1,
                      "operator input 'input'.shape", error) ||
        vx_shape_rank(mask, 0, 2, 1,
                      "operator input 'mask'.shape", error) ||
        vx_shape_per_tensor_byte(input, "operator input 'input'",
                                 &input_quantization, error) ||
        vx_shape_i32_tensor(mask, "operator input 'mask'", error)) return -1;
    if (mask->shape[0] != input->shape[0] ||
        mask->shape[1] != input->shape[1]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'mask'.shape",
                             "must be [B,S] for input [B,S,D].");
    }
    output_shape[0] = input->shape[0];
    output_shape[1] = input->shape[2];
    if (vx_shape_quantized_declared_output(request, output_shape, 2,
                                           &output, error) ||
        vx_shape_positive_f32_ratio(
            input_quantization->scale, 1.0f, output->quantization.scale,
            "declared output 'out'.quantization.scale", error) ||
        vx_shape_i32_sum_bound(
            input->shape[1],
            vx_shape_centered_magnitude(input->dtype,
                                        input_quantization->zero_point),
            "operator input 'input'.shape[1]", error)) return -1;
    return vx_shape_clone_output(candidate, output_shape, 2, output->dtype,
                                 &output->quantization, error);
}

static int vx_shape_q_attention_parameters(
        const VxConcreteShapeRequest* request,
        uint64_t* heads,
        float* scale,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"heads", "causal", "scale"};
    const VxShapeParam* param;
    double value;
    if (vx_shape_validate_params(request, FIELDS, 3, error)) return -1;
    param = vx_shape_find_param(request, "heads");
    if (!param || param->kind != VX_SHAPE_PARAM_NUMBER ||
        !vx_shape_is_safe_integer(param->value.number) ||
        param->value.number <= 0.0) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.heads",
                             "must be a positive safe integer.");
    }
    *heads = (uint64_t)param->value.number;
    param = vx_shape_find_param(request, "causal");
    if (!param || param->kind != VX_SHAPE_PARAM_BOOLEAN) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.causal", "must be boolean.");
    }
    param = vx_shape_find_param(request, "scale");
    if (!param || param->kind != VX_SHAPE_PARAM_NUMBER) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.scale",
                             "must be positive and representable as float32.");
    }
    value = param->value.number;
    *scale = (float)value;
    if (!isfinite(value) || value <= 0.0 ||
        !isfinite(*scale) || *scale <= 0.0f) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.scale",
                             "must be positive and representable as float32.");
    }
    return 0;
}

static int vx_shape_infer_q_sdpa(
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
    const VxShapeTensorDescriptor* output;
    const VxShapeQuantization* query_quantization;
    const VxShapeQuantization* key_quantization;
    const VxShapeQuantization* value_quantization;
    uint64_t heads = 0;
    uint64_t feature;
    uint64_t head_dimension;
    uint64_t batch;
    uint64_t queries;
    uint64_t keys;
    uint64_t maximum_dot = 0;
    uint64_t left_magnitude;
    uint64_t right_magnitude;
    float scale = 0.0f;
    float score_multiplier;
    float maximum_score;
    size_t sequence_axis;
    size_t feature_axis;
    if (vx_shape_q_attention_parameters(request, &heads, &scale, error) ||
        vx_shape_per_tensor_byte(query, "operator input 'q'",
                                 &query_quantization, error) ||
        vx_shape_per_tensor_byte(key, "operator input 'k'",
                                 &key_quantization, error) ||
        vx_shape_per_tensor_byte(value, "operator input 'v'",
                                 &value_quantization, error) ||
        vx_shape_attention_rank(query, "operator input 'q'.shape", error)) {
        return -1;
    }
    if (key->rank != query->rank || value->rank != query->rank) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator inputs",
                             "q, k, and v must have the same rank.");
    }
    sequence_axis = query->rank - 2;
    feature_axis = query->rank - 1;
    batch = query->rank == 2 ? 1 : query->shape[0];
    queries = query->shape[sequence_axis];
    keys = key->shape[sequence_axis];
    feature = query->shape[feature_axis];
    if ((query->rank == 3 &&
         (key->shape[0] != batch || value->shape[0] != batch)) ||
        key->shape[feature_axis] != feature ||
        value->shape[feature_axis] != feature ||
        value->shape[sequence_axis] != keys) {
        return vx_shape_fail(
            error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
            "operator inputs",
            "QSDPA q/k/v batch, feature, and K/V sequence geometry must match.");
    }
    if (feature % heads || feature % 4) {
        char path[224];
        snprintf(path, sizeof(path), "operator input 'q'.shape[%zu]",
                 feature_axis);
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             path, "D must be divisible by heads and by 4.");
    }
    head_dimension = feature / heads;
    if (head_dimension % 4 || head_dimension > 64) {
        char path[224];
        snprintf(path, sizeof(path), "operator input 'q'.shape[%zu]",
                 feature_axis);
        return vx_shape_fail(
            error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH, path,
            "head_dim must be divisible by 4 and no greater than 64.");
    }
    if (vx_shape_attention_mask(mask, batch, queries, keys, error) ||
        vx_shape_quantized_declared_output(request, query->shape, query->rank,
                                           &output, error)) return -1;
    left_magnitude = vx_shape_centered_magnitude(
        query->dtype, query_quantization->zero_point);
    right_magnitude = vx_shape_centered_magnitude(
        key->dtype, key_quantization->zero_point);
    {
        char path[224];
        snprintf(path, sizeof(path), "operator input 'q'.shape[%zu]",
                 feature_axis);
        if (vx_shape_i32_accumulator_bound(head_dimension, left_magnitude,
                                           right_magnitude, path, error)) {
            return -1;
        }
    }
    if (vx_shape_checked_multiply_u64(head_dimension, left_magnitude,
                                      "operator params.scale", &maximum_dot,
                                      error) ||
        vx_shape_checked_multiply_u64(maximum_dot, right_magnitude,
                                      "operator params.scale", &maximum_dot,
                                      error)) return -1;
    score_multiplier = (query_quantization->scale * key_quantization->scale) *
        scale;
    maximum_score = (float)maximum_dot * score_multiplier;
    if (!isfinite(score_multiplier) || score_multiplier <= 0.0f ||
        !isfinite(maximum_score)) {
        return vx_shape_fail(
            error, VX_SHAPE_CONTRACT_ERROR_UNSAFE_QUANTIZATION_TRANSFORM,
            "operator params.scale",
            "quantized score scale and maximum score must be finite positive float32 values.");
    }
    if (vx_shape_positive_f32_ratio(
            value_quantization->scale, 1.0f, output->quantization.scale,
            "declared output 'out'.quantization.scale", error)) return -1;
    return vx_shape_clone_output(candidate, query->shape, query->rank,
                                 output->dtype, &output->quantization, error);
}

static int vx_shape_infer_q_arg_max(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis"};
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeParam* axis;
    if (vx_shape_per_tensor_byte(input, "operator input 'input'", NULL,
                                 error)) return -1;
    if (input->rank < 2 || input->rank > 8) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator input 'input'.shape",
                             "must have rank 2 through 8.");
    }
    if (vx_shape_validate_params(request, FIELDS, 1, error)) return -1;
    axis = vx_shape_find_param(request, "axis");
    if (request->param_count != 1 || !axis ||
        axis->kind != VX_SHAPE_PARAM_NUMBER || axis->value.number != -1.0) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.axis",
                             "must be exactly -1 for canonical QArgMax.");
    }
    return vx_shape_clone_output(candidate, input->shape, input->rank - 1,
                                 VX_DTYPE_I32, NULL, error);
}

static int vx_shape_infer_quantized_kind(
        VxOperatorKind kind,
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    switch (kind) {
        case VX_OP_Q_LINEAR:
        case VX_OP_Q_MATMUL:
        case VX_OP_Q_GEMM:
            return vx_shape_infer_q_dense(request, candidate, error);
        case VX_OP_Q_BATCH_MATMUL:
            return vx_shape_infer_q_batch_matmul(request, candidate, error);
        case VX_OP_Q_CONV_2D:
            return vx_shape_infer_q_conv2d(request, candidate, error);
        case VX_OP_Q_ADD:
            return vx_shape_infer_q_add(request, candidate, error);
        case VX_OP_Q_EMBEDDING:
            return vx_shape_infer_q_embedding(request, candidate, error);
        case VX_OP_Q_GELU:
        case VX_OP_Q_SILU:
            return vx_shape_infer_q_activation(kind, request, candidate, error);
        case VX_OP_Q_LAYER_NORM:
            return vx_shape_infer_q_layer_norm(request, candidate, error);
        case VX_OP_Q_GROUP_NORM:
            return vx_shape_infer_q_group_norm(request, candidate, error);
        case VX_OP_Q_MASKED_MEAN:
            return vx_shape_infer_q_masked_mean(request, candidate, error);
        case VX_OP_Q_SDPA:
            return vx_shape_infer_q_sdpa(request, candidate, error);
        case VX_OP_Q_ARG_MAX:
            return vx_shape_infer_q_arg_max(request, candidate, error);
        default:
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_UNKNOWN_OPERATOR,
                                 "operator",
                                 "has no implemented quantized contract.");
    }
}

