/* Shape contracts for BatchMatMul and the spatial operators: Conv1D/2D,
 * ConvTranspose2D, pooling, Resize and Upsample.
 * Mirrors tests/operator_shape_contract_spatial_vectors.json. */

static int vx_shape_infer_batch_matmul(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* left = vx_shape_input_descriptor(request, "a");
    const VxShapeTensorDescriptor* right = vx_shape_input_descriptor(request, "b");
    uint64_t output_shape[8];
    size_t left_batch;
    size_t right_batch;
    size_t batch_rank;
    size_t axis;
    if (vx_shape_float_tensor(left, "operator input 'a'", error) ||
        vx_shape_float_tensor(right, "operator input 'b'", error) ||
        vx_shape_rank(left, 2, 0, 0, "operator input 'a'.shape", error) ||
        vx_shape_rank(right, 2, 0, 0, "operator input 'b'.shape", error)) {
        return -1;
    }
    if (left->rank > 8 || right->rank > 8) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator inputs",
                             "BatchMatMul operand ranks must not exceed 8.");
    }
    if (vx_shape_validate_params(request, NULL, 0, error)) return -1;
    if (left->shape[left->rank - 1] != right->shape[right->rank - 2]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'b'.shape",
                             "contracted dimensions must be equal.");
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
    return vx_shape_clone_output(candidate, output_shape, batch_rank + 2,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_conv1d(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {
        "stride", "padding", "groups", "relu", "data_layout", "weight_layout"
    };
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight = vx_shape_input_descriptor(request, "weight");
    const VxShapeTensorDescriptor* bias = vx_shape_input_descriptor(request, "bias");
    const char* data_layout;
    const char* weight_layout;
    uint64_t stride;
    uint64_t padding;
    uint64_t groups;
    uint64_t activation;
    uint64_t output_length;
    uint64_t output_shape[3];
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
        vx_shape_rank(input, 0, 3, 1, "operator input 'input'.shape", error) ||
        vx_shape_rank(weight, 0, 3, 1, "operator input 'weight'.shape", error) ||
        vx_shape_validate_params(request, FIELDS,
                                 sizeof(FIELDS) / sizeof(FIELDS[0]), error) ||
        vx_shape_param_string_default(request, "data_layout", "NLC",
                                      &data_layout, error) ||
        vx_shape_param_string_default(request, "weight_layout", "WIO",
                                      &weight_layout, error) ||
        vx_shape_param_scalar(request, "stride", 1, 1, 0,
                              &stride, error) ||
        vx_shape_param_scalar(request, "padding", 0, 0, 0,
                              &padding, error) ||
        vx_shape_param_scalar(request, "groups", 1, 1, 0,
                              &groups, error) ||
        vx_shape_param_activation(request, &activation, error)) return -1;
    (void)activation;
    if (strcmp(data_layout, "NLC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NLC'.");
    }
    if (strcmp(weight_layout, "WIO")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.weight_layout", "must be 'WIO'.");
    }
    if (input->shape[2] % groups || weight->shape[2] % groups ||
        weight->shape[1] != input->shape[2] / groups) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'weight'.shape[1]",
                             "is incompatible with input channels and groups.");
    }
    if (bias &&
        (vx_shape_float_tensor(bias, "operator input 'bias'", error) ||
         vx_shape_vector(bias, weight->shape[2],
                         "operator input 'bias'.shape", error))) return -1;
    if (vx_shape_window_output(input->shape[1], weight->shape[0], stride,
                               padding, padding, 1,
                               "operator input 'input'.shape[1]",
                               &output_length, error)) return -1;
    output_shape[0] = input->shape[0];
    output_shape[1] = output_length;
    output_shape[2] = weight->shape[2];
    return vx_shape_clone_output(candidate, output_shape, 3,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_conv2d(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {
        "stride", "padding", "pads", "dilation", "groups", "relu",
        "data_layout", "weight_layout"
    };
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight = vx_shape_input_descriptor(request, "weight");
    const VxShapeTensorDescriptor* bias = vx_shape_input_descriptor(request, "bias");
    const char* data_layout;
    const char* weight_layout;
    uint64_t stride[2];
    uint64_t dilation[2];
    uint64_t pads[4];
    uint64_t groups;
    uint64_t activation;
    uint64_t output_channels = 0;
    uint64_t output_shape[4];
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
        vx_shape_rank(input, 0, 4, 1, "operator input 'input'.shape", error) ||
        vx_shape_rank(weight, 0, 4, 1, "operator input 'weight'.shape", error) ||
        vx_shape_validate_params(request, FIELDS,
                                 sizeof(FIELDS) / sizeof(FIELDS[0]), error) ||
        vx_shape_param_string_default(request, "data_layout", "NHWC",
                                      &data_layout, error) ||
        vx_shape_param_string_default(request, "weight_layout", "HWIO",
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
    if (!strcmp(weight_layout, "HWCM")) {
        if (groups != input->shape[3] || weight->shape[2] != input->shape[3]) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator input 'weight'.shape",
                                 "depthwise Conv2D requires HWCM geometry.");
        }
        if (vx_shape_checked_multiply_u64(input->shape[3], weight->shape[3],
                                          "Conv2D output channels",
                                          &output_channels, error)) return -1;
    } else if (!strcmp(weight_layout, "HWIO")) {
        if (input->shape[3] % groups || weight->shape[3] % groups ||
            weight->shape[2] != input->shape[3] / groups) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator input 'weight'.shape",
                                 "grouped Conv2D requires compatible HWIO geometry.");
        }
        output_channels = weight->shape[3];
    } else {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.weight_layout",
                             "must be 'HWIO' or 'HWCM'.");
    }
    if (bias &&
        (vx_shape_float_tensor(bias, "operator input 'bias'", error) ||
         vx_shape_vector(bias, output_channels,
                         "operator input 'bias'.shape", error))) return -1;
    output_shape[0] = input->shape[0];
    if (vx_shape_window_output(input->shape[1], weight->shape[0], stride[0],
                               pads[0], pads[2], dilation[0],
                               "operator input 'input'.shape[1]",
                               &output_shape[1], error) ||
        vx_shape_window_output(input->shape[2], weight->shape[1], stride[1],
                               pads[1], pads[3], dilation[1],
                               "operator input 'input'.shape[2]",
                               &output_shape[2], error)) return -1;
    output_shape[3] = output_channels;
    return vx_shape_clone_output(candidate, output_shape, 4,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_conv_transpose2d(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {
        "kernel", "stride", "padding", "data_layout", "weight_layout"
    };
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight = vx_shape_input_descriptor(request, "weight");
    const VxShapeTensorDescriptor* bias = vx_shape_input_descriptor(request, "bias");
    const char* data_layout;
    const char* weight_layout;
    uint64_t kernel[2];
    uint64_t stride[2];
    uint64_t padding[2];
    uint64_t output_shape[4];
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
        vx_shape_rank(input, 0, 4, 1, "operator input 'input'.shape", error) ||
        vx_shape_rank(weight, 0, 4, 1, "operator input 'weight'.shape", error) ||
        vx_shape_validate_params(request, FIELDS,
                                 sizeof(FIELDS) / sizeof(FIELDS[0]), error) ||
        vx_shape_param_string_default(request, "data_layout", "NHWC",
                                      &data_layout, error) ||
        vx_shape_param_string_default(request, "weight_layout", "HWIO",
                                      &weight_layout, error) ||
        vx_shape_param_pair(request, "kernel", 1, 1, 1, kernel, error) ||
        vx_shape_param_pair(request, "stride", 1, 1, 0, stride, error) ||
        vx_shape_param_pair(request, "padding", 0, 0, 0,
                            padding, error)) return -1;
    if (strcmp(data_layout, "NHWC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NHWC'.");
    }
    if (strcmp(weight_layout, "HWIO")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.weight_layout", "must be 'HWIO'.");
    }
    if (kernel[0] != weight->shape[0] || kernel[1] != weight->shape[1]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator params.kernel",
                             "must equal the HWIO weight kernel extents.");
    }
    if (weight->shape[2] != input->shape[3]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'weight'.shape[2]",
                             "must equal input channels.");
    }
    if (bias &&
        (vx_shape_float_tensor(bias, "operator input 'bias'", error) ||
         vx_shape_vector(bias, weight->shape[3],
                         "operator input 'bias'.shape", error))) return -1;
    output_shape[0] = input->shape[0];
    if (vx_shape_transpose_output(input->shape[1], kernel[0], stride[0],
                                  padding[0],
                                  "operator input 'input'.shape[1]",
                                  &output_shape[1], error) ||
        vx_shape_transpose_output(input->shape[2], kernel[1], stride[1],
                                  padding[1],
                                  "operator input 'input'.shape[2]",
                                  &output_shape[2], error)) return -1;
    output_shape[3] = weight->shape[3];
    return vx_shape_clone_output(candidate, output_shape, 4,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_pool2d(
        VxOperatorKind kind,
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const MAX_FIELDS[] = {
        "kernel", "stride", "padding", "pads", "dilation", "ceil_mode",
        "data_layout"
    };
    static const char* const AVERAGE_FIELDS[] = {
        "kernel", "stride", "padding", "pads", "dilation", "ceil_mode",
        "count_include_pad", "auto_pad", "data_layout"
    };
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const char* data_layout;
    const char* auto_pad;
    uint64_t kernel[2];
    uint64_t stride[2];
    uint64_t dilation[2];
    uint64_t pads[4];
    uint64_t output_shape[4];
    int average = kind == VX_OP_AVERAGE_POOL_2D;
    if (vx_shape_rank(input, 0, 4, 1,
                      "operator input 'input'.shape", error) ||
        vx_shape_validate_params(
            request, average ? AVERAGE_FIELDS : MAX_FIELDS,
            average ? sizeof(AVERAGE_FIELDS) / sizeof(AVERAGE_FIELDS[0])
                    : sizeof(MAX_FIELDS) / sizeof(MAX_FIELDS[0]), error) ||
        vx_shape_param_string_default(request, "data_layout", "NHWC",
                                      &data_layout, error) ||
        vx_shape_param_false_or_absent(request, "ceil_mode", error)) return -1;
    if (strcmp(data_layout, "NHWC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NHWC'.");
    }
    if (average) {
        if (vx_shape_param_false_or_absent(request, "count_include_pad", error) ||
            vx_shape_param_string_default(request, "auto_pad", "NOTSET",
                                          &auto_pad, error)) return -1;
        if (strcmp(auto_pad, "NOTSET") && strcmp(auto_pad, "")) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.auto_pad",
                                 "must be 'NOTSET', empty, or absent.");
        }
    }
    if (vx_shape_param_pair(request, "dilation", 1, 1, 0,
                            dilation, error)) return -1;
    if (dilation[0] != 1 || dilation[1] != 1) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.dilation",
                             "dilated pooling is not supported.");
    }
    if (vx_shape_param_pair(request, "kernel", 1, 1, 1, kernel, error) ||
        vx_shape_param_pair(request, "stride", 1, 1, 0, stride, error) ||
        vx_shape_spatial_pads(request, average, pads, error)) return -1;

    if (input->dtype == VX_DTYPE_F32) {
        if (vx_shape_float_tensor(input, "operator input 'input'", error)) return -1;
    } else if (!average &&
               (input->dtype == VX_DTYPE_I8 || input->dtype == VX_DTYPE_U8)) {
        if (input->quantization.scheme != VX_SHAPE_QUANTIZATION_PER_TENSOR) {
            return vx_shape_fail(error,
                                 VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                                 "operator input 'input'.quantization",
                                 "raw byte MaxPool2D requires per-tensor quantization.");
        }
    } else {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator input 'input'.dtype",
                             average ? "AveragePool2D requires float32."
                                     : "MaxPool2D requires float32 or I8/U8.");
    }
    output_shape[0] = input->shape[0];
    if (vx_shape_window_output(input->shape[1], kernel[0], stride[0],
                               pads[0], pads[2], 1,
                               "operator input 'input'.shape[1]",
                               &output_shape[1], error) ||
        vx_shape_window_output(input->shape[2], kernel[1], stride[1],
                               pads[1], pads[3], 1,
                               "operator input 'input'.shape[2]",
                               &output_shape[2], error)) return -1;
    output_shape[3] = input->shape[3];
    return vx_shape_clone_output(candidate, output_shape, 4, input->dtype,
                                 &input->quantization, error);
}

static int vx_shape_infer_global_average_pool(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"data_layout"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const char* data_layout;
    uint64_t output_shape[4];
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank(input, 0, 4, 1, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_param_string_default(request, "data_layout", "NHWC",
                                      &data_layout, error)) return -1;
    if (strcmp(data_layout, "NHWC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NHWC'.");
    }
    output_shape[0] = input->shape[0];
    output_shape[1] = 1;
    output_shape[2] = 1;
    output_shape[3] = input->shape[3];
    return vx_shape_clone_output(candidate, output_shape, 4,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_resize_dtype(
        const VxShapeTensorDescriptor* input,
        const VxShapeTensorDescriptor* output,
        VxShapeContractError* error) {
    if (input->dtype == VX_DTYPE_F32) {
        if (vx_shape_float_tensor(input, "operator input 'input'", error)) return -1;
        if (output->dtype != VX_DTYPE_F32) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                                 "declared output 'out'.dtype",
                                 "must be 'float32'.");
        }
        return vx_shape_unquantized(output, "declared output 'out'", error);
    }
    if ((input->dtype != VX_DTYPE_I8 && input->dtype != VX_DTYPE_U8) ||
        input->quantization.scheme != VX_SHAPE_QUANTIZATION_PER_TENSOR) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                             "operator input 'input'.quantization",
                             "raw byte resize requires per-tensor I8/U8 quantization.");
    }
    if (output->dtype != input->dtype ||
        !vx_shape_quantization_equal(&input->quantization,
                                     &output->quantization)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                             "declared output 'out'.quantization",
                             "must preserve the exact input byte domain.");
    }
    return 0;
}

static int vx_shape_infer_resize(
        VxOperatorKind kind,
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {
        "mode", "coordinate_transformation_mode", "coordinate_transform_mode",
        "nearest_mode", "align_corners", "antialias", "data_layout"
    };
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* declared = NULL;
    const VxShapeParam* legacy_transform;
    const char* data_layout;
    const char* mode;
    const char* transform;
    const char* nearest_mode;
    int nearest;
    if (vx_shape_rank(input, 0, 4, 1, "operator input 'input'.shape", error) ||
        vx_shape_declared_output(request, &declared, error) ||
        vx_shape_rank(declared, 0, 4, 1,
                      "declared output 'out'.shape", error) ||
        vx_shape_resize_dtype(input, declared, error) ||
        vx_shape_validate_params(request, FIELDS,
                                 sizeof(FIELDS) / sizeof(FIELDS[0]), error) ||
        vx_shape_param_string_default(request, "data_layout", "NHWC",
                                      &data_layout, error)) return -1;
    if (strcmp(data_layout, "NHWC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NHWC'.");
    }
    legacy_transform = vx_shape_find_param(request, "coordinate_transform_mode");
    if (legacy_transform) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.coordinate_transform_mode",
                             "is not defined; use coordinate_transformation_mode.");
    }
    if (vx_shape_param_string_default(
            request, "mode",
            kind == VX_OP_RESIZE_NEAREST_2D ? "nearest" : "linear",
            &mode, error)) return -1;
    if (kind == VX_OP_RESIZE_NEAREST_2D && strcmp(mode, "nearest")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.mode", "must be 'nearest'.");
    }
    if (kind == VX_OP_RESIZE && strcmp(mode, "nearest") &&
        strcmp(mode, "linear")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.mode",
                             "must be 'nearest' or 'linear'.");
    }
    nearest = kind == VX_OP_RESIZE_NEAREST_2D || !strcmp(mode, "nearest");
    if (input->dtype != VX_DTYPE_F32 && !nearest) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.mode",
                             "raw I8/U8 resize requires explicit nearest mode.");
    }
    if (vx_shape_param_string_default(
            request, "coordinate_transformation_mode",
            nearest ? "asymmetric" : "half_pixel", &transform, error)) return -1;
    if ((nearest && strcmp(transform, "asymmetric")) ||
        (!nearest && strcmp(transform, "half_pixel"))) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.coordinate_transformation_mode",
                             nearest ? "must be 'asymmetric' for nearest resize."
                                     : "must be 'half_pixel' for linear resize.");
    }
    if (nearest) {
        if (vx_shape_param_string_default(request, "nearest_mode", "floor",
                                          &nearest_mode, error)) return -1;
        if (strcmp(nearest_mode, "floor")) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.nearest_mode",
                                 "must be 'floor'.");
        }
    } else if (vx_shape_find_param(request, "nearest_mode")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.nearest_mode",
                             "is valid only for nearest resize.");
    }
    if (vx_shape_param_false_or_absent(request, "align_corners", error) ||
        vx_shape_param_false_or_absent(request, "antialias", error)) return -1;
    if (declared->shape[0] != input->shape[0] ||
        declared->shape[3] != input->shape[3]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "declared output 'out'.shape",
                             "must preserve NHWC batch and channel extents.");
    }
    return vx_shape_clone_output(candidate, declared->shape, declared->rank,
                                 declared->dtype, &declared->quantization, error);
}

static int vx_shape_infer_upsample_nearest2d(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"data_layout"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const char* data_layout;
    uint64_t output_shape[4];
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank(input, 0, 4, 1, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_param_string_default(request, "data_layout", "NHWC",
                                      &data_layout, error)) return -1;
    if (strcmp(data_layout, "NHWC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NHWC'.");
    }
    output_shape[0] = input->shape[0];
    if (vx_shape_checked_multiply_u64(input->shape[1], 2,
                                      "operator input 'input'.shape[1]",
                                      &output_shape[1], error) ||
        vx_shape_checked_multiply_u64(input->shape[2], 2,
                                      "operator input 'input'.shape[2]",
                                      &output_shape[2], error)) return -1;
    output_shape[3] = input->shape[3];
    return vx_shape_clone_output(candidate, output_shape, 4,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_spatial_kind(
        VxOperatorKind kind,
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    switch (kind) {
        case VX_OP_BATCH_MATMUL:
            return vx_shape_infer_batch_matmul(request, candidate, error);
        case VX_OP_CONV_1D:
            return vx_shape_infer_conv1d(request, candidate, error);
        case VX_OP_CONV_2D:
            return vx_shape_infer_conv2d(request, candidate, error);
        case VX_OP_CONV_TRANSPOSE_2D:
            return vx_shape_infer_conv_transpose2d(request, candidate, error);
        case VX_OP_MAX_POOL_2D:
        case VX_OP_AVERAGE_POOL_2D:
            return vx_shape_infer_pool2d(kind, request, candidate, error);
        case VX_OP_GLOBAL_AVERAGE_POOL:
            return vx_shape_infer_global_average_pool(request, candidate, error);
        case VX_OP_RESIZE:
        case VX_OP_RESIZE_NEAREST_2D:
            return vx_shape_infer_resize(kind, request, candidate, error);
        case VX_OP_UPSAMPLE_NEAREST_2D:
            return vx_shape_infer_upsample_nearest2d(request, candidate, error);
        default:
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_UNKNOWN_OPERATOR,
                                 "operator",
                                 "has no implemented spatial contract.");
    }
}


static int vx_shape_param_signed_integer_value(double value,
                                               const char* path,
                                               int64_t* output,
                                               VxShapeContractError* error) {
    if (!vx_shape_is_safe_integer(value)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             path, "must be an integer.");
    }
    *output = (int64_t)value;
    return 0;
}

static int vx_shape_normalize_axis(const VxConcreteShapeRequest* request,
                                   const char* name,
                                   size_t rank,
                                   int64_t default_axis,
                                   int allow_boundary,
                                   size_t* output,
                                   VxShapeContractError* error) {
    const VxShapeParam* param = vx_shape_find_param(request, name);
    int64_t axis = default_axis;
    int64_t upper = (int64_t)rank + (allow_boundary ? 1 : 0);
    char path[224];
    snprintf(path, sizeof(path), "operator params.%s", name);
    if ((!allow_boundary && rank == 0) || rank > (size_t)INT64_MAX) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             path, "requires an input rank of at least 1.");
    }
    if (param) {
        if (param->kind != VX_SHAPE_PARAM_NUMBER ||
            vx_shape_param_signed_integer_value(param->value.number, path,
                                                &axis, error)) return -1;
    }
    if (axis < 0) axis += (int64_t)rank;
    if (axis < 0 || axis >= upper) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             path, "resolves outside the supported rank.");
    }
    *output = (size_t)axis;
    return 0;
}

static int vx_shape_boolean_default(const VxConcreteShapeRequest* request,
                                    const char* name,
                                    int default_value,
                                    int* output,
                                    VxShapeContractError* error) {
    const VxShapeParam* param = vx_shape_find_param(request, name);
    char path[224];
    snprintf(path, sizeof(path), "operator params.%s", name);
    if (!param) {
        *output = default_value;
        return 0;
    }
    if (param->kind != VX_SHAPE_PARAM_BOOLEAN) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             path, "must be boolean.");
    }
    *output = !!param->value.boolean;
    return 0;
}

static int vx_shape_param_number_array(
        const VxConcreteShapeRequest* request,
        const char* name,
        int required,
        const double** values,
        size_t* count,
        VxShapeContractError* error) {
    const VxShapeParam* param = vx_shape_find_param(request, name);
    char path[224];
    snprintf(path, sizeof(path), "operator params.%s", name);
    if (!param) {
        if (required) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 path, "is required.");
        }
        *values = NULL;
        *count = 0;
        return 0;
    }
    if (param->kind != VX_SHAPE_PARAM_NUMBER_ARRAY ||
        (param->value.number_array.count &&
         !param->value.number_array.values)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             path, "must be an array.");
    }
    *values = param->value.number_array.values;
    *count = param->value.number_array.count;
    return 0;
}

static int vx_shape_product_range(const uint64_t* shape,
                                  size_t begin,
                                  size_t end,
                                  const char* path,
                                  uint64_t* output,
                                  VxShapeContractError* error) {
    uint64_t product = 1;
    size_t axis;
    for (axis = begin; axis < end; ++axis) {
        if (vx_shape_checked_multiply_u64(product, shape[axis], path,
                                          &product, error)) return -1;
    }
    *output = product;
    return 0;
}

static int vx_shape_broadcast(const VxShapeTensorDescriptor* left,
                              const VxShapeTensorDescriptor* right,
                              uint64_t output[8],
                              size_t* output_rank,
                              VxShapeContractError* error) {
    size_t rank = left->rank > right->rank ? left->rank : right->rank;
    size_t axis;
    if (rank > 8) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator inputs",
                             "broadcast rank must not exceed 8.");
    }
    for (axis = 0; axis < rank; ++axis) {
        size_t left_offset = rank - left->rank;
        size_t right_offset = rank - right->rank;
        uint64_t a = axis < left_offset ? 1 : left->shape[axis - left_offset];
        uint64_t b = axis < right_offset ? 1 : right->shape[axis - right_offset];
        if (a != b && a != 1 && b != 1) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator inputs",
                                 "input shapes are not broadcast-compatible.");
        }
        output[axis] = a > b ? a : b;
    }
    *output_rank = rank;
    return 0;
}

static int vx_shape_optional_declared_output(
        const VxConcreteShapeRequest* request,
        const VxShapeTensorDescriptor** output,
        VxShapeContractError* error) {
    if (!request->declared_output_count) {
        *output = NULL;
        return 0;
    }
    return vx_shape_declared_output(request, output, error);
}

static int vx_shape_reshape_quantization(
        const VxShapeTensorDescriptor* input,
        const uint64_t* output_shape,
        size_t output_rank,
        VxShapeQuantization* output,
        VxShapeContractError* error) {
    const VxShapeQuantization* quantization = &input->quantization;
    size_t candidate = 0;
    size_t candidate_count = 0;
    size_t axis;
    if (quantization->scheme != VX_SHAPE_QUANTIZATION_PER_AXIS) {
        *output = *quantization;
        return 0;
    }
    for (axis = 0; axis < output_rank; ++axis) {
        uint64_t input_prefix;
        uint64_t output_prefix;
        uint64_t input_suffix;
        uint64_t output_suffix;
        if (input->shape[quantization->axis] != output_shape[axis]) continue;
        if (vx_shape_product_range(input->shape, 0, quantization->axis,
                                   "operator input 'input'.quantization input prefix",
                                   &input_prefix, error) ||
            vx_shape_product_range(output_shape, 0, axis,
                                   "operator input 'input'.quantization output prefix",
                                   &output_prefix, error) ||
            vx_shape_product_range(input->shape, quantization->axis + 1,
                                   input->rank,
                                   "operator input 'input'.quantization input suffix",
                                   &input_suffix, error) ||
            vx_shape_product_range(output_shape, axis + 1, output_rank,
                                   "operator input 'input'.quantization output suffix",
                                   &output_suffix, error)) return -1;
        if (input_prefix == output_prefix && input_suffix == output_suffix) {
            candidate = axis;
            candidate_count++;
        }
    }
    if (candidate_count != 1) {
        return vx_shape_fail(
            error, VX_SHAPE_CONTRACT_ERROR_UNSAFE_QUANTIZATION_TRANSFORM,
            "operator input 'input'.quantization",
            "cannot map the per-axis coordinate uniquely through this reshape.");
    }
    *output = *quantization;
    output->axis = candidate;
    return 0;
}

