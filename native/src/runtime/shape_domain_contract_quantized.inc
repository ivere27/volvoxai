/* Symbolic contracts for the canonical W8A8 quantized operators. This
 * fragment is included only by shape_domain_contract.c. Quantized tensor
 * payloads never enter this proof: it consumes only normalized dtype, shape,
 * affine metadata, and shape-relevant parameters. */

static float vx_sd_quantized_f32(uint32_t bits) {
    float value;
    vx_sd_copy(&value, &bits, (uint32_t)sizeof(value));
    return value;
}

static int32_t vx_sd_quantized_per_tensor(
        VxShapeDomainBuild* build,
        const VxShapeDomainTensorDescriptor* descriptor,
        uint32_t input_index,
        const VxShapeDomainQuantization** quantization) {
    if (!descriptor ||
        (descriptor->dtype != VX_DTYPE_I8 &&
         descriptor->dtype != VX_DTYPE_U8))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                          input_index, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (descriptor->quantization.scheme != 1)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION,
                          input_index, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (quantization) *quantization = &descriptor->quantization;
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_quantized_axis_zero_weight(
        VxShapeDomainBuild* build,
        const VxShapeDomainTensorDescriptor* descriptor,
        uint32_t input_index,
        const VxShapeDomainQuantization** quantization) {
    if (!descriptor ||
        (descriptor->dtype != VX_DTYPE_I8 &&
         descriptor->dtype != VX_DTYPE_U8))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                          input_index, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (descriptor->quantization.scheme != 2 ||
        descriptor->quantization.axis != 0u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION,
                          input_index, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (quantization) *quantization = &descriptor->quantization;
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_quantized_i32(
        VxShapeDomainBuild* build,
        const VxShapeDomainTensorDescriptor* descriptor,
        uint32_t input_index) {
    if (!descriptor || descriptor->dtype != VX_DTYPE_I32)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                          input_index, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (!vx_sd_unquantized(descriptor))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION,
                          input_index, VX_SHAPE_DOMAIN_INDEX_NONE);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_quantized_positive_ratio(
        VxShapeDomainBuild* build,
        float left,
        float right,
        float divisor,
        uint32_t index,
        uint32_t subindex) {
    float product = (float)((double)left * (double)right);
    float multiplier = (float)((double)product / (double)divisor);
    if (!isfinite(multiplier) || multiplier <= 0.0f)
        return vx_sd_fail(build,
            VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, index, subindex);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static uint64_t vx_sd_quantized_centered_magnitude(int32_t dtype,
                                                   int32_t zero_point) {
    int64_t minimum = dtype == VX_DTYPE_I8 ? -128 : 0;
    int64_t maximum = dtype == VX_DTYPE_I8 ? 127 : 255;
    uint64_t below = zero_point >= minimum
        ? (uint64_t)((int64_t)zero_point - minimum)
        : (uint64_t)(minimum - (int64_t)zero_point);
    uint64_t above = zero_point >= maximum
        ? (uint64_t)((int64_t)zero_point - maximum)
        : (uint64_t)(maximum - (int64_t)zero_point);
    return below > above ? below : above;
}

static uint64_t vx_sd_quantized_maximum_weight_magnitude(
        int32_t dtype,
        const VxShapeDomainQuantization* quantization) {
    uint64_t maximum = 0u;
    for (uint32_t index = 0u; index < quantization->count; ++index) {
        uint64_t magnitude = vx_sd_quantized_centered_magnitude(
            dtype, quantization->zero_points[index]);
        if (magnitude > maximum) maximum = magnitude;
    }
    return maximum;
}

static int32_t vx_sd_quantized_accumulator_bound(
        VxShapeDomainBuild* build,
        uint64_t terms,
        uint64_t left_magnitude,
        uint64_t right_magnitude,
        uint32_t index,
        uint32_t subindex) {
    uint64_t maximum;
    if (vx_sd_checked_multiply(terms, left_magnitude, &maximum) ||
        vx_sd_checked_multiply(maximum, right_magnitude, &maximum))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                          index, subindex);
    if (maximum > UINT32_C(2147483647))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          index, subindex);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_quantized_sum_bound(
        VxShapeDomainBuild* build,
        uint64_t terms,
        uint64_t magnitude,
        uint32_t index,
        uint32_t subindex) {
    uint64_t maximum;
    if (vx_sd_checked_multiply(terms, magnitude, &maximum))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                          index, subindex);
    if (maximum > UINT32_C(2147483647))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          index, subindex);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_quantized_dimension_maximum(
        VxShapeDomainBuild* build,
        const VxShapeDomainDimension* dimension,
        uint32_t input_index,
        uint32_t axis,
        uint64_t* maximum) {
    const VxShapeDomainConstraint* constraint;
    if (dimension->kind == VX_SHAPE_DOMAIN_DIMENSION_FIXED) {
        *maximum = (uint64_t)dimension->value;
        return VX_SHAPE_DOMAIN_STATUS_OK;
    }
    if (dimension->symbol_index >= build->request->environment_count)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DIMENSION,
                          input_index, axis);
    constraint = &build->request->environment[dimension->symbol_index];
    if (dimension->kind == VX_SHAPE_DOMAIN_DIMENSION_SYMBOL) {
        *maximum = constraint->maximum;
        return VX_SHAPE_DOMAIN_STATUS_OK;
    }
    if (dimension->kind != VX_SHAPE_DOMAIN_DIMENSION_AFFINE ||
        vx_sd_add_signed(constraint->maximum, dimension->value, maximum))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                          input_index, axis);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_quantized_output(
        VxShapeDomainBuild* build,
        const VxShapeDomainDimension* dimensions,
        uint32_t rank,
        const VxShapeDomainNamedTensor** declared_output) {
    const VxShapeDomainNamedTensor* declared = vx_sd_named(
        build->request->declared_outputs,
        build->request->declared_output_count, "out");
    if (!declared)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS,
                          0u, 1u);
    if (vx_sd_quantized_per_tensor(build, &declared->descriptor,
                                  0u, NULL))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (declared->descriptor.rank != rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          declared->descriptor.rank, rank);
    if (build->request->use_declared_outputs) {
        for (uint32_t axis = 0u; axis < rank; ++axis)
            if (!vx_shape_domain_dimensions_equal(
                    &declared->descriptor.dimensions[axis],
                    &dimensions[axis], build->request->environment,
                    build->request->environment_count))
                return vx_sd_fail(build,
                    VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH, 0u, axis);
    }
    if (declared_output) *declared_output = declared;
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_quantized_set_output(
        VxShapeDomainBuild* build,
        const VxShapeDomainDimension* dimensions,
        uint32_t rank,
        const VxShapeDomainNamedTensor* declared) {
    return vx_sd_set_output(build, "out", declared->descriptor.dtype,
                            dimensions, rank,
                            &declared->descriptor.quantization);
}

static int32_t vx_sd_quantized_dense_multipliers(
        VxShapeDomainBuild* build,
        const VxShapeDomainQuantization* input,
        const VxShapeDomainQuantization* weight,
        const VxShapeDomainQuantization* output) {
    float input_scale = vx_sd_quantized_f32(input->scale_bits);
    float output_scale = vx_sd_quantized_f32(output->scale_bits);
    for (uint32_t channel = 0u; channel < weight->count; ++channel) {
        if (vx_sd_quantized_positive_ratio(build, input_scale,
                vx_sd_quantized_f32(weight->scales[channel]), output_scale,
                1u, channel))
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
    }
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_quantized_positive_f32_param(
        VxShapeDomainBuild* build,
        const char* name,
        int required,
        float* output) {
    const VxShapeDomainParam* param = vx_sd_param(build->request, name);
    float canonical;
    if (!param) {
        if (required)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        if (output) *output = 0.0f;
        return VX_SHAPE_DOMAIN_STATUS_OK;
    }
    if (param->kind != VX_SHAPE_DOMAIN_PARAM_NUMBER ||
        !isfinite(param->number) || param->number <= 0.0)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    canonical = (float)param->number;
    if (!isfinite(canonical) || canonical <= 0.0f)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (output) *output = canonical;
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_quantized_dense(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input", "weight", "bias"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* weight;
    const VxShapeDomainNamedTensor* bias;
    const VxShapeDomainNamedTensor* output;
    const VxShapeDomainQuantization* input_quantization;
    const VxShapeDomainQuantization* weight_quantization;
    VxShapeDomainDimension* dimensions;
    uint64_t output_feature;
    uint64_t contracted;
    uint64_t input_contracted;
    if (vx_sd_exact_ports(build, INPUTS, 3u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, NULL, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    weight = vx_sd_input(build->request, "weight");
    bias = vx_sd_input(build->request, "bias");
    if (!input->descriptor.rank || weight->descriptor.rank != 2u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          input->descriptor.rank, weight->descriptor.rank);
    if (vx_sd_quantized_per_tensor(build, &input->descriptor, 0u,
            &input_quantization) ||
        vx_sd_quantized_axis_zero_weight(build, &weight->descriptor, 1u,
            &weight_quantization) ||
        vx_sd_quantized_i32(build, &bias->descriptor, 2u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!vx_sd_direct_constant_shape(&weight->descriptor) ||
        !vx_sd_direct_constant_shape(&bias->descriptor))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          1u, VX_SHAPE_DOMAIN_INDEX_NONE);
    output_feature = (uint64_t)weight->descriptor.dimensions[0].value;
    contracted = (uint64_t)weight->descriptor.dimensions[1].value;
    if (!vx_shape_domain_fixed_value(
            &input->descriptor.dimensions[input->descriptor.rank - 1u],
            build->request->environment,
            build->request->environment_count, &input_contracted))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          0u, input->descriptor.rank - 1u);
    if (input_contracted != contracted)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, input->descriptor.rank - 1u);
    if (vx_sd_direct_vector(build, &bias->descriptor, output_feature, 2u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    dimensions = vx_sd_direct_output_dimensions(
        build, "out", input->descriptor.rank);
    if (!dimensions)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, input->descriptor.rank);
    for (uint32_t axis = 0u; axis < input->descriptor.rank; ++axis)
        dimensions[axis] = input->descriptor.dimensions[axis];
    dimensions[input->descriptor.rank - 1u] = vx_sd_fixed(output_feature);
    if (vx_sd_quantized_output(build, dimensions, input->descriptor.rank,
            &output) ||
        vx_sd_quantized_dense_multipliers(build, input_quantization,
            weight_quantization, &output->descriptor.quantization) ||
        vx_sd_quantized_accumulator_bound(build, contracted,
            vx_sd_quantized_centered_magnitude(
                input->descriptor.dtype, input_quantization->zero_point),
            vx_sd_quantized_maximum_weight_magnitude(
                weight->descriptor.dtype, weight_quantization),
            0u, input->descriptor.rank - 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT;
    return vx_sd_quantized_set_output(
        build, dimensions, input->descriptor.rank, output);
}

static int32_t vx_sd_quantized_batch_matmul(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"a", "b"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* left;
    const VxShapeDomainNamedTensor* right;
    const VxShapeDomainNamedTensor* output;
    const VxShapeDomainQuantization* left_quantization;
    const VxShapeDomainQuantization* right_quantization;
    VxShapeDomainTensorDescriptor left_batch;
    VxShapeDomainTensorDescriptor right_batch;
    VxShapeDomainDimension dimensions[8];
    uint32_t batch_rank;
    uint64_t terms;
    if (vx_sd_exact_ports(build, INPUTS, 2u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, NULL, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    left = vx_sd_input(build->request, "a");
    right = vx_sd_input(build->request, "b");
    if (left->descriptor.rank < 2u || right->descriptor.rank < 2u ||
        left->descriptor.rank > 8u || right->descriptor.rank > 8u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          left->descriptor.rank, right->descriptor.rank);
    if (vx_sd_quantized_per_tensor(build, &left->descriptor, 0u,
            &left_quantization) ||
        vx_sd_quantized_per_tensor(build, &right->descriptor, 1u,
            &right_quantization))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!vx_shape_domain_dimensions_equal(
            &left->descriptor.dimensions[left->descriptor.rank - 1u],
            &right->descriptor.dimensions[right->descriptor.rank - 2u],
            build->request->environment,
            build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          1u, right->descriptor.rank - 2u);
    vx_sd_descriptor_copy(&left_batch, &left->descriptor);
    vx_sd_descriptor_copy(&right_batch, &right->descriptor);
    left_batch.rank -= 2u;
    right_batch.rank -= 2u;
    if (vx_sd_spatial_broadcast_shape(&left_batch, &right_batch,
            build->request->environment,
            build->request->environment_count, dimensions, &batch_rank))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          0u, 1u);
    dimensions[batch_rank] =
        left->descriptor.dimensions[left->descriptor.rank - 2u];
    dimensions[batch_rank + 1u] =
        right->descriptor.dimensions[right->descriptor.rank - 1u];
    if (vx_sd_quantized_output(build, dimensions, batch_rank + 2u,
            &output) ||
        vx_sd_quantized_positive_ratio(build,
            vx_sd_quantized_f32(left_quantization->scale_bits),
            vx_sd_quantized_f32(right_quantization->scale_bits),
            vx_sd_quantized_f32(output->descriptor.quantization.scale_bits),
            0u, VX_SHAPE_DOMAIN_INDEX_NONE) ||
        vx_sd_quantized_dimension_maximum(build,
            &left->descriptor.dimensions[left->descriptor.rank - 1u],
            0u, left->descriptor.rank - 1u, &terms) ||
        vx_sd_quantized_accumulator_bound(build, terms,
            vx_sd_quantized_centered_magnitude(
                left->descriptor.dtype, left_quantization->zero_point),
            vx_sd_quantized_centered_magnitude(
                right->descriptor.dtype, right_quantization->zero_point),
            0u, left->descriptor.rank - 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_BROADCAST;
    return vx_sd_quantized_set_output(
        build, dimensions, batch_rank + 2u, output);
}

static int32_t vx_sd_quantized_conv2d(VxShapeDomainBuild* build) {
    static const char* const REQUIRED[] = {"input", "weight"};
    static const char* const OPTIONAL[] = {"bias"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const ALLOWED[] = {
        "stride", "padding", "pads", "dilation", "groups", "relu",
        "data_layout", "weight_layout"
    };
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* weight;
    const VxShapeDomainNamedTensor* bias;
    const VxShapeDomainNamedTensor* output;
    const VxShapeDomainQuantization* input_quantization;
    const VxShapeDomainQuantization* weight_quantization;
    VxShapeDomainDimension dimensions[4];
    uint64_t stride[2];
    uint64_t pads[4];
    uint64_t dilation[2];
    uint64_t groups;
    uint64_t input_channels;
    uint64_t expected_channels;
    uint64_t output_channels;
    uint64_t terms;
    if (vx_sd_optional_ports(build, REQUIRED, 2u, OPTIONAL, 1u,
            OUTPUTS, 1u) || vx_sd_allowed_params(build, ALLOWED, 8u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    weight = vx_sd_input(build->request, "weight");
    bias = vx_sd_input(build->request, "bias");
    if (input->descriptor.rank != 4u || weight->descriptor.rank != 4u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          input->descriptor.rank, weight->descriptor.rank);
    if (vx_sd_quantized_per_tensor(build, &input->descriptor, 0u,
            &input_quantization) ||
        vx_sd_quantized_axis_zero_weight(build, &weight->descriptor, 1u,
            &weight_quantization))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!vx_sd_spatial_constant_shape(&weight->descriptor))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          1u, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (vx_sd_spatial_layout(build, "data_layout", "NHWC") ||
        vx_sd_spatial_layout(build, "weight_layout", "OHWI") ||
        vx_sd_spatial_pair(build, "stride", 1u, 0, 0, stride) ||
        vx_sd_spatial_pads(build, 0, pads) ||
        vx_sd_spatial_pair(build, "dilation", 1u, 0, 0, dilation) ||
        vx_sd_spatial_positive_integer(build, "groups", 1u, &groups) ||
        vx_sd_spatial_activation(build))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!vx_shape_domain_fixed_value(&input->descriptor.dimensions[3],
            build->request->environment,
            build->request->environment_count, &input_channels))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE, 0u, 3u);
    output_channels = (uint64_t)weight->descriptor.dimensions[0].value;
    if (vx_sd_checked_multiply(
            (uint64_t)weight->descriptor.dimensions[3].value,
            groups, &expected_channels))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                          1u, 3u);
    if (input_channels != expected_channels || output_channels % groups)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          1u, 3u);
    if (bias) {
        if (vx_sd_quantized_i32(build, &bias->descriptor, 2u))
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
        if (!vx_sd_spatial_constant_shape(&bias->descriptor))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                              2u, VX_SHAPE_DOMAIN_INDEX_NONE);
        if (vx_sd_spatial_vector(build, &bias->descriptor,
                                 output_channels, 2u))
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
    }
    dimensions[0] = input->descriptor.dimensions[0];
    if (vx_sd_spatial_window_dimension(build,
            &input->descriptor.dimensions[1],
            (uint64_t)weight->descriptor.dimensions[1].value,
            stride[0], pads[0], pads[2], dilation[0],
            0u, 1u, &dimensions[1]) ||
        vx_sd_spatial_window_dimension(build,
            &input->descriptor.dimensions[2],
            (uint64_t)weight->descriptor.dimensions[2].value,
            stride[1], pads[1], pads[3], dilation[1],
            0u, 2u, &dimensions[2]))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    dimensions[3] = vx_sd_fixed(output_channels);
    if (vx_sd_quantized_output(build, dimensions, 4u, &output) ||
        vx_sd_quantized_dense_multipliers(build, input_quantization,
            weight_quantization, &output->descriptor.quantization))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (vx_sd_checked_multiply(
            (uint64_t)weight->descriptor.dimensions[1].value,
            (uint64_t)weight->descriptor.dimensions[2].value, &terms) ||
        vx_sd_checked_multiply(terms,
            (uint64_t)weight->descriptor.dimensions[3].value, &terms))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                          1u, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (vx_sd_quantized_accumulator_bound(build, terms,
            vx_sd_quantized_centered_magnitude(
                input->descriptor.dtype, input_quantization->zero_point),
            vx_sd_quantized_maximum_weight_magnitude(
                weight->descriptor.dtype, weight_quantization),
            1u, VX_SHAPE_DOMAIN_INDEX_NONE))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_quantized_set_output(build, dimensions, 4u, output);
}

static int32_t vx_sd_quantized_add(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"a", "b"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const ALLOWED[] = {"relu"};
    const VxShapeDomainNamedTensor* left;
    const VxShapeDomainNamedTensor* right;
    const VxShapeDomainNamedTensor* output;
    const VxShapeDomainQuantization* left_quantization;
    const VxShapeDomainQuantization* right_quantization;
    float output_scale;
    if (vx_sd_exact_ports(build, INPUTS, 2u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, ALLOWED, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    left = vx_sd_input(build->request, "a");
    right = vx_sd_input(build->request, "b");
    if (vx_sd_quantized_per_tensor(build, &left->descriptor, 0u,
            &left_quantization) ||
        vx_sd_quantized_per_tensor(build, &right->descriptor, 1u,
            &right_quantization) ||
        vx_sd_spatial_activation(build))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!vx_sd_shapes_equal(&left->descriptor, &right->descriptor,
            build->request->environment,
            build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, 1u);
    if (vx_sd_quantized_output(build, left->descriptor.dimensions,
            left->descriptor.rank, &output))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    output_scale = vx_sd_quantized_f32(
        output->descriptor.quantization.scale_bits);
    if (vx_sd_quantized_positive_ratio(build,
            vx_sd_quantized_f32(left_quantization->scale_bits), 1.0f,
            output_scale, 0u, VX_SHAPE_DOMAIN_INDEX_NONE) ||
        vx_sd_quantized_positive_ratio(build,
            vx_sd_quantized_f32(right_quantization->scale_bits), 1.0f,
            output_scale, 1u, VX_SHAPE_DOMAIN_INDEX_NONE))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_EXACT_BINARY;
    return vx_sd_quantized_set_output(build, left->descriptor.dimensions,
                                      left->descriptor.rank, output);
}

static int32_t vx_sd_quantized_activation(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const APPROXIMATE[] = {"approximate"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* output;
    const VxShapeDomainParam* approximate;
    if (vx_sd_exact_ports(build, INPUTS, 1u, OUTPUTS, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (vx_sd_quantized_per_tensor(build, &input->descriptor, 0u, NULL))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (build->request->operator_kind == VX_OP_Q_GELU) {
        if (vx_sd_allowed_params(build, APPROXIMATE, 1u))
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
        approximate = vx_sd_param(build->request, "approximate");
        if (approximate &&
            (approximate->kind != VX_SHAPE_DOMAIN_PARAM_STRING ||
             !vx_sd_string_equal(&approximate->string, "none")))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
    } else if (vx_sd_allowed_params(build, NULL, 0u)) {
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    }
    if (vx_sd_quantized_output(build, input->descriptor.dimensions,
            input->descriptor.rank, &output))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_quantized_set_output(build, input->descriptor.dimensions,
                                      input->descriptor.rank, output);
}

static int32_t vx_sd_quantized_embedding(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input", "weight"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* weight;
    const VxShapeDomainNamedTensor* output;
    const VxShapeDomainQuantization* weight_quantization;
    VxShapeDomainDimension* dimensions;
    uint32_t output_rank;
    uint64_t width;
    float output_scale;
    if (vx_sd_exact_ports(build, INPUTS, 2u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, NULL, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    weight = vx_sd_input(build->request, "weight");
    if (vx_sd_quantized_i32(build, &input->descriptor, 0u) ||
        vx_sd_quantized_axis_zero_weight(build, &weight->descriptor, 1u,
            &weight_quantization))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!input->descriptor.rank || weight->descriptor.rank != 2u ||
        !vx_sd_direct_constant_shape(&weight->descriptor) ||
        input->descriptor.rank == UINT32_MAX)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          input->descriptor.rank, weight->descriptor.rank);
    output_rank = input->descriptor.rank + 1u;
    dimensions = vx_sd_direct_output_dimensions(build, "out", output_rank);
    if (!dimensions)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, output_rank);
    width = (uint64_t)weight->descriptor.dimensions[1].value;
    for (uint32_t axis = 0u; axis < input->descriptor.rank; ++axis)
        dimensions[axis] = input->descriptor.dimensions[axis];
    dimensions[input->descriptor.rank] = vx_sd_fixed(width);
    if (vx_sd_quantized_output(build, dimensions, output_rank, &output))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    output_scale = vx_sd_quantized_f32(
        output->descriptor.quantization.scale_bits);
    for (uint32_t row = 0u; row < weight_quantization->count; ++row)
        if (vx_sd_quantized_positive_ratio(build,
                vx_sd_quantized_f32(weight_quantization->scales[row]),
                1.0f, output_scale, 1u, row))
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT;
    return vx_sd_quantized_set_output(
        build, dimensions, output_rank, output);
}

static int32_t vx_sd_quantized_layer_norm(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input", "weight", "bias"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const ALLOWED[] = {"eps", "d_model"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* weight;
    const VxShapeDomainNamedTensor* bias;
    const VxShapeDomainNamedTensor* output;
    const VxShapeDomainParam* model;
    uint64_t feature;
    int64_t model_extent = 0;
    if (vx_sd_exact_ports(build, INPUTS, 3u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, ALLOWED, 2u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    weight = vx_sd_input(build->request, "weight");
    bias = vx_sd_input(build->request, "bias");
    if (!input->descriptor.rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, 0u);
    if (vx_sd_quantized_per_tensor(build, &input->descriptor, 0u, NULL) ||
        vx_sd_assert_float(build, &weight->descriptor, 1u) ||
        vx_sd_assert_float(build, &bias->descriptor, 2u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!vx_shape_domain_fixed_value(
            &input->descriptor.dimensions[input->descriptor.rank - 1u],
            build->request->environment,
            build->request->environment_count, &feature) ||
        !vx_sd_direct_constant_shape(&weight->descriptor) ||
        !vx_sd_direct_constant_shape(&bias->descriptor))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          0u, input->descriptor.rank - 1u);
    if (vx_sd_direct_vector(build, &weight->descriptor, feature, 1u) ||
        vx_sd_direct_vector(build, &bias->descriptor, feature, 2u) ||
        vx_sd_quantized_positive_f32_param(build, "eps", 0, NULL))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    model = vx_sd_param(build->request, "d_model");
    if (model) {
        if (vx_sd_number_integer(build, model, 0, &model_extent) ||
            model_extent <= 0)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        if ((uint64_t)model_extent != feature)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
    }
    if (vx_sd_quantized_output(build, input->descriptor.dimensions,
            input->descriptor.rank, &output))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_quantized_set_output(build, input->descriptor.dimensions,
                                      input->descriptor.rank, output);
}

static int32_t vx_sd_quantized_group_norm(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input", "weight", "bias"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const ALLOWED[] = {"num_groups", "eps", "data_layout"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* weight;
    const VxShapeDomainNamedTensor* bias;
    const VxShapeDomainNamedTensor* output;
    const VxShapeDomainParam* groups_param;
    uint64_t channels;
    int64_t groups;
    if (vx_sd_exact_ports(build, INPUTS, 3u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, ALLOWED, 3u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    weight = vx_sd_input(build->request, "weight");
    bias = vx_sd_input(build->request, "bias");
    if (input->descriptor.rank != 4u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, input->descriptor.rank);
    if (vx_sd_quantized_per_tensor(build, &input->descriptor, 0u, NULL) ||
        vx_sd_assert_float(build, &weight->descriptor, 1u) ||
        vx_sd_assert_float(build, &bias->descriptor, 2u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!vx_shape_domain_fixed_value(&input->descriptor.dimensions[3],
            build->request->environment,
            build->request->environment_count, &channels) ||
        !vx_sd_direct_constant_shape(&weight->descriptor) ||
        !vx_sd_direct_constant_shape(&bias->descriptor))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE, 0u, 3u);
    if (vx_sd_direct_vector(build, &weight->descriptor, channels, 1u) ||
        vx_sd_direct_vector(build, &bias->descriptor, channels, 2u) ||
        vx_sd_spatial_layout(build, "data_layout", "NHWC") ||
        vx_sd_quantized_positive_f32_param(build, "eps", 0, NULL))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    groups_param = vx_sd_param(build->request, "num_groups");
    if (!groups_param ||
        vx_sd_number_integer(build, groups_param, 0, &groups) || groups <= 0)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (channels % (uint64_t)groups)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, 3u);
    if (vx_sd_quantized_output(build, input->descriptor.dimensions,
            input->descriptor.rank, &output))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_quantized_set_output(build, input->descriptor.dimensions,
                                      input->descriptor.rank, output);
}

static int32_t vx_sd_quantized_masked_mean(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input", "mask"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* mask;
    const VxShapeDomainNamedTensor* output;
    const VxShapeDomainQuantization* input_quantization;
    VxShapeDomainDimension dimensions[2];
    uint64_t sequence;
    if (vx_sd_exact_ports(build, INPUTS, 2u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, NULL, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    mask = vx_sd_input(build->request, "mask");
    if (input->descriptor.rank != 3u || mask->descriptor.rank != 2u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          input->descriptor.rank, mask->descriptor.rank);
    if (vx_sd_quantized_per_tensor(build, &input->descriptor, 0u,
            &input_quantization) ||
        vx_sd_quantized_i32(build, &mask->descriptor, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!vx_shape_domain_dimensions_equal(&mask->descriptor.dimensions[0],
            &input->descriptor.dimensions[0], build->request->environment,
            build->request->environment_count) ||
        !vx_shape_domain_dimensions_equal(&mask->descriptor.dimensions[1],
            &input->descriptor.dimensions[1], build->request->environment,
            build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          1u, VX_SHAPE_DOMAIN_INDEX_NONE);
    dimensions[0] = input->descriptor.dimensions[0];
    dimensions[1] = input->descriptor.dimensions[2];
    if (vx_sd_quantized_output(build, dimensions, 2u, &output) ||
        vx_sd_quantized_positive_ratio(build,
            vx_sd_quantized_f32(input_quantization->scale_bits), 1.0f,
            vx_sd_quantized_f32(output->descriptor.quantization.scale_bits),
            0u, VX_SHAPE_DOMAIN_INDEX_NONE) ||
        vx_sd_quantized_dimension_maximum(build,
            &input->descriptor.dimensions[1], 0u, 1u, &sequence) ||
        vx_sd_quantized_sum_bound(build, sequence,
            vx_sd_quantized_centered_magnitude(
                input->descriptor.dtype, input_quantization->zero_point),
            0u, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_quantized_set_output(build, dimensions, 2u, output);
}

static int32_t vx_sd_quantized_attention_mask(
        VxShapeDomainBuild* build,
        const VxShapeDomainTensorDescriptor* mask,
        const VxShapeDomainDimension* batch,
        const VxShapeDomainDimension* queries,
        const VxShapeDomainDimension* keys,
        uint32_t input_index) {
    int valid = 0;
    if (!mask) return VX_SHAPE_DOMAIN_STATUS_OK;
    if (vx_sd_quantized_i32(build, mask, input_index))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (mask->rank == 1u) {
        valid = vx_shape_domain_dimensions_equal(&mask->dimensions[0], keys,
            build->request->environment,
            build->request->environment_count);
    } else if (mask->rank == 2u) {
        valid = vx_shape_domain_dimensions_equal(&mask->dimensions[1], keys,
            build->request->environment,
            build->request->environment_count) &&
            (vx_shape_domain_dimensions_equal(&mask->dimensions[0], batch,
                build->request->environment,
                build->request->environment_count) ||
             vx_shape_domain_dimensions_equal(&mask->dimensions[0], queries,
                build->request->environment,
                build->request->environment_count));
    } else if (mask->rank == 3u) {
        valid = vx_shape_domain_dimensions_equal(&mask->dimensions[0], batch,
            build->request->environment,
            build->request->environment_count) &&
            vx_shape_domain_dimensions_equal(&mask->dimensions[1], queries,
                build->request->environment,
                build->request->environment_count) &&
            vx_shape_domain_dimensions_equal(&mask->dimensions[2], keys,
                build->request->environment,
                build->request->environment_count);
    }
    if (!valid)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          input_index, VX_SHAPE_DOMAIN_INDEX_NONE);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_quantized_sdpa(VxShapeDomainBuild* build) {
    static const char* const REQUIRED[] = {"q", "k", "v"};
    static const char* const OPTIONAL[] = {"mask"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const ALLOWED[] = {"heads", "causal", "scale"};
    const VxShapeDomainNamedTensor* query;
    const VxShapeDomainNamedTensor* key;
    const VxShapeDomainNamedTensor* value;
    const VxShapeDomainNamedTensor* mask;
    const VxShapeDomainNamedTensor* output;
    const VxShapeDomainQuantization* query_quantization;
    const VxShapeDomainQuantization* key_quantization;
    const VxShapeDomainQuantization* value_quantization;
    const VxShapeDomainParam* heads_param;
    const VxShapeDomainParam* causal_param;
    VxShapeDomainDimension one = vx_sd_fixed(1u);
    const VxShapeDomainDimension* batch;
    const VxShapeDomainDimension* queries;
    const VxShapeDomainDimension* keys;
    uint32_t rank;
    uint32_t sequence_axis;
    uint32_t feature_axis;
    uint64_t feature;
    uint64_t head_dimension;
    uint64_t maximum_dot;
    uint64_t left_magnitude;
    uint64_t right_magnitude;
    int64_t heads;
    float scale;
    float score_product;
    float score_multiplier;
    float maximum_score;
    if (vx_sd_optional_ports(build, REQUIRED, 3u, OPTIONAL, 1u,
            OUTPUTS, 1u) || vx_sd_allowed_params(build, ALLOWED, 3u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    query = vx_sd_input(build->request, "q");
    key = vx_sd_input(build->request, "k");
    value = vx_sd_input(build->request, "v");
    mask = vx_sd_input(build->request, "mask");
    if (vx_sd_quantized_per_tensor(build, &query->descriptor, 0u,
            &query_quantization) ||
        vx_sd_quantized_per_tensor(build, &key->descriptor, 1u,
            &key_quantization) ||
        vx_sd_quantized_per_tensor(build, &value->descriptor, 2u,
            &value_quantization))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    rank = query->descriptor.rank;
    if ((rank != 2u && rank != 3u) || key->descriptor.rank != rank ||
        value->descriptor.rank != rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          rank, key->descriptor.rank);
    sequence_axis = rank - 2u;
    feature_axis = rank - 1u;
    batch = rank == 2u ? &one : &query->descriptor.dimensions[0];
    queries = &query->descriptor.dimensions[sequence_axis];
    keys = &key->descriptor.dimensions[sequence_axis];
    if ((rank == 3u &&
         (!vx_shape_domain_dimensions_equal(&key->descriptor.dimensions[0],
              batch, build->request->environment,
              build->request->environment_count) ||
          !vx_shape_domain_dimensions_equal(&value->descriptor.dimensions[0],
              batch, build->request->environment,
              build->request->environment_count))) ||
        !vx_shape_domain_dimensions_equal(
            &key->descriptor.dimensions[feature_axis],
            &query->descriptor.dimensions[feature_axis],
            build->request->environment,
            build->request->environment_count) ||
        !vx_shape_domain_dimensions_equal(
            &value->descriptor.dimensions[feature_axis],
            &query->descriptor.dimensions[feature_axis],
            build->request->environment,
            build->request->environment_count) ||
        !vx_shape_domain_dimensions_equal(
            &value->descriptor.dimensions[sequence_axis], keys,
            build->request->environment,
            build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, 2u);
    if (!vx_shape_domain_fixed_value(
            &query->descriptor.dimensions[feature_axis],
            build->request->environment,
            build->request->environment_count, &feature))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          0u, feature_axis);
    heads_param = vx_sd_param(build->request, "heads");
    causal_param = vx_sd_param(build->request, "causal");
    if (!heads_param ||
        vx_sd_number_integer(build, heads_param, 0, &heads) || heads <= 0 ||
        !causal_param ||
        causal_param->kind != VX_SHAPE_DOMAIN_PARAM_BOOLEAN ||
        (causal_param->boolean != 0 && causal_param->boolean != 1))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (feature % (uint64_t)heads || feature % 4u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, feature_axis);
    head_dimension = feature / (uint64_t)heads;
    if (head_dimension % 4u || head_dimension > 64u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, feature_axis);
    if (vx_sd_param(build->request, "scale")) {
        if (vx_sd_quantized_positive_f32_param(
                build, "scale", 1, &scale))
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
    } else {
        scale = (float)(1.0 / sqrt((double)head_dimension));
        if (!isfinite(scale) || scale <= 0.0f)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
    }
    if (vx_sd_quantized_attention_mask(build,
            mask ? &mask->descriptor : NULL, batch, queries, keys, 3u) ||
        vx_sd_quantized_output(build, query->descriptor.dimensions,
            rank, &output))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    left_magnitude = vx_sd_quantized_centered_magnitude(
        query->descriptor.dtype, query_quantization->zero_point);
    right_magnitude = vx_sd_quantized_centered_magnitude(
        key->descriptor.dtype, key_quantization->zero_point);
    if (vx_sd_quantized_accumulator_bound(build, head_dimension,
            left_magnitude, right_magnitude, 0u, feature_axis))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (vx_sd_checked_multiply(head_dimension, left_magnitude,
                               &maximum_dot) ||
        vx_sd_checked_multiply(maximum_dot, right_magnitude,
                               &maximum_dot))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                          0u, feature_axis);
    score_product = (float)(
        (double)vx_sd_quantized_f32(query_quantization->scale_bits) *
        (double)vx_sd_quantized_f32(key_quantization->scale_bits));
    score_multiplier = (float)((double)score_product * (double)scale);
    maximum_score = (float)((double)maximum_dot * (double)score_multiplier);
    if (!isfinite(score_multiplier) || score_multiplier <= 0.0f ||
        !isfinite(maximum_score))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION,
                          0u, feature_axis);
    if (vx_sd_quantized_positive_ratio(build,
            vx_sd_quantized_f32(value_quantization->scale_bits), 1.0f,
            vx_sd_quantized_f32(output->descriptor.quantization.scale_bits),
            2u, VX_SHAPE_DOMAIN_INDEX_NONE))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_quantized_set_output(
        build, query->descriptor.dimensions, rank, output);
}

static int32_t vx_sd_quantized_argmax(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const ALLOWED[] = {"axis"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainParam* axis;
    uint32_t normalized_axis;
    VxShapeDomainDimension dimensions[8];
    if (vx_sd_exact_ports(build, INPUTS, 1u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, ALLOWED, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (vx_sd_quantized_per_tensor(build, &input->descriptor, 0u, NULL))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (input->descriptor.rank < 2u || input->descriptor.rank > 8u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, input->descriptor.rank);
    axis = vx_sd_param(build->request, "axis");
    if (build->request->param_count != 1u || !axis ||
        axis->kind != VX_SHAPE_DOMAIN_PARAM_NUMBER)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (vx_sd_axis(build, "axis", input->descriptor.rank, -1,
                   &normalized_axis)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    for (uint32_t source = 0u, target = 0u;
         source < input->descriptor.rank; source++)
        if (source != normalized_axis)
            dimensions[target++] = input->descriptor.dimensions[source];
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", VX_DTYPE_I32,
                            dimensions,
                            input->descriptor.rank - 1u, NULL);
}

static int32_t vx_sd_prove_quantized(VxShapeDomainBuild* build) {
    switch (build->request->operator_kind) {
        case VX_OP_Q_LINEAR:
        case VX_OP_Q_MATMUL:
        case VX_OP_Q_GEMM:
            return vx_sd_quantized_dense(build);
        case VX_OP_Q_BATCH_MATMUL:
            return vx_sd_quantized_batch_matmul(build);
        case VX_OP_Q_CONV_2D:
            return vx_sd_quantized_conv2d(build);
        case VX_OP_Q_ADD:
            return vx_sd_quantized_add(build);
        case VX_OP_Q_EMBEDDING:
            return vx_sd_quantized_embedding(build);
        case VX_OP_Q_GELU:
        case VX_OP_Q_SILU:
            return vx_sd_quantized_activation(build);
        case VX_OP_Q_LAYER_NORM:
            return vx_sd_quantized_layer_norm(build);
        case VX_OP_Q_GROUP_NORM:
            return vx_sd_quantized_group_norm(build);
        case VX_OP_Q_MASKED_MEAN:
            return vx_sd_quantized_masked_mean(build);
        case VX_OP_Q_SDPA:
            return vx_sd_quantized_sdpa(build);
        case VX_OP_Q_ARG_MAX:
            return vx_sd_quantized_argmax(build);
        default:
            return VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED;
    }
}
