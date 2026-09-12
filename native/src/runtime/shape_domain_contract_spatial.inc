/* Symbolic contracts for BatchMatMul and the canonical spatial operators.
 * This fragment is included only by shape_domain_contract.c. It deliberately
 * accepts only formulas representable by the v1 fixed/symbol domain: a
 * dynamic window must be an algebraic identity, otherwise its source extent
 * must be fixed over the complete declared domain. */

static int32_t vx_sd_spatial_single_ports(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input"};
    static const char* const OUTPUTS[] = {"out"};
    return vx_sd_exact_ports(build, INPUTS, 1u, OUTPUTS, 1u);
}

static int vx_sd_spatial_constant_shape(
        const VxShapeDomainTensorDescriptor* descriptor) {
    if (!descriptor || (descriptor->rank && !descriptor->dimensions)) return 0;
    for (uint32_t axis = 0u; axis < descriptor->rank; ++axis)
        if (descriptor->dimensions[axis].kind !=
            VX_SHAPE_DOMAIN_DIMENSION_FIXED) return 0;
    return 1;
}

static int32_t vx_sd_spatial_layout(VxShapeDomainBuild* build,
                                    const char* name,
                                    const char* expected) {
    const VxShapeDomainParam* param = vx_sd_param(build->request, name);
    if (param && (param->kind != VX_SHAPE_DOMAIN_PARAM_STRING ||
                  !vx_sd_string_equal(&param->string, expected)))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_spatial_pair(VxShapeDomainBuild* build,
                                  const char* name,
                                  uint64_t default_value,
                                  int allow_zero,
                                  int required,
                                  uint64_t output[2]) {
    const VxShapeDomainParam* param = vx_sd_param(build->request, name);
    int64_t values[2] = {0, 0};
    if (!param) {
        if (required)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        output[0] = default_value;
        output[1] = default_value;
        return VX_SHAPE_DOMAIN_STATUS_OK;
    }
    if (param->kind == VX_SHAPE_DOMAIN_PARAM_NUMBER) {
        if (vx_sd_number_integer(build, param, 0, &values[0]))
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
        values[1] = values[0];
    } else if (param->kind == VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY &&
               (param->value_count == 1u || param->value_count == 2u)) {
        if (vx_sd_param_array_integer(param, 0u, &values[0]) ||
            (param->value_count == 2u &&
             vx_sd_param_array_integer(param, 1u, &values[1])))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        if (param->value_count == 1u) values[1] = values[0];
    } else {
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    }
    for (uint32_t axis = 0u; axis < 2u; ++axis) {
        if (values[axis] < (allow_zero ? 0 : 1))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE, axis);
        output[axis] = (uint64_t)values[axis];
    }
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_spatial_scalar(VxShapeDomainBuild* build,
                                    const char* name,
                                    uint64_t default_value,
                                    int allow_zero,
                                    uint64_t* output) {
    const VxShapeDomainParam* param = vx_sd_param(build->request, name);
    int64_t value = 0;
    if (!param) {
        *output = default_value;
        return VX_SHAPE_DOMAIN_STATUS_OK;
    }
    if (param->kind == VX_SHAPE_DOMAIN_PARAM_NUMBER) {
        if (vx_sd_number_integer(build, param, 0, &value))
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
    } else if (param->kind == VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY &&
               param->value_count == 1u &&
               !vx_sd_param_array_integer(param, 0u, &value)) {
        /* One-element arrays are part of the persisted Conv1D contract. */
    } else {
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    }
    if (value < (allow_zero ? 0 : 1))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    *output = (uint64_t)value;
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_spatial_positive_integer(VxShapeDomainBuild* build,
                                              const char* name,
                                              uint64_t default_value,
                                              uint64_t* output) {
    const VxShapeDomainParam* param = vx_sd_param(build->request, name);
    int64_t value = 0;
    if (!param) {
        *output = default_value;
        return VX_SHAPE_DOMAIN_STATUS_OK;
    }
    if (vx_sd_number_integer(build, param, 0, &value) || value <= 0)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    *output = (uint64_t)value;
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_spatial_activation(VxShapeDomainBuild* build) {
    const VxShapeDomainParam* param = vx_sd_param(build->request, "relu");
    int64_t value = 0;
    if (!param) return VX_SHAPE_DOMAIN_STATUS_OK;
    if (vx_sd_number_integer(build, param, 0, &value) ||
        value < 0 || value > 2)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_spatial_false_or_absent(VxShapeDomainBuild* build,
                                             const char* name) {
    const VxShapeDomainParam* param = vx_sd_param(build->request, name);
    int64_t value = 0;
    if (!param) return VX_SHAPE_DOMAIN_STATUS_OK;
    if (param->kind == VX_SHAPE_DOMAIN_PARAM_BOOLEAN && !param->boolean)
        return VX_SHAPE_DOMAIN_STATUS_OK;
    if (param->kind == VX_SHAPE_DOMAIN_PARAM_NUMBER &&
        !vx_sd_number_integer(build, param, 0, &value) && !value)
        return VX_SHAPE_DOMAIN_STATUS_OK;
    return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                      VX_SHAPE_DOMAIN_INDEX_NONE,
                      VX_SHAPE_DOMAIN_INDEX_NONE);
}

static int32_t vx_sd_spatial_pads(VxShapeDomainBuild* build,
                                  int symmetric_only,
                                  uint64_t output[4]) {
    const VxShapeDomainParam* padding_param =
        vx_sd_param(build->request, "padding");
    const VxShapeDomainParam* pads_param =
        vx_sd_param(build->request, "pads");
    uint64_t padding[2] = {0u, 0u};
    if (vx_sd_spatial_pair(build, "padding", 0u, 1, 0, padding))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!pads_param) {
        output[0] = padding[0];
        output[1] = padding[1];
        output[2] = padding[0];
        output[3] = padding[1];
        return VX_SHAPE_DOMAIN_STATUS_OK;
    }
    if (pads_param->kind != VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY ||
        pads_param->value_count != 4u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    for (uint32_t axis = 0u; axis < 4u; ++axis) {
        int64_t value;
        if (vx_sd_param_array_integer(pads_param, axis, &value) || value < 0)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE, axis);
        output[axis] = (uint64_t)value;
    }
    if (padding_param &&
        (output[0] != padding[0] || output[1] != padding[1] ||
         output[2] != padding[0] || output[3] != padding[1]))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (symmetric_only &&
        (output[0] != output[2] || output[1] != output[3]))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_spatial_window_dimension(
        VxShapeDomainBuild* build,
        const VxShapeDomainDimension* input,
        uint64_t kernel,
        uint64_t stride,
        uint64_t pad_before,
        uint64_t pad_after,
        uint64_t dilation,
        uint32_t input_index,
        uint32_t axis,
        VxShapeDomainDimension* output) {
    uint64_t effective;
    uint64_t pad_total;
    uint64_t fixed;
    uint64_t padded;
    int64_t numerator;
    int64_t quotient;
    int64_t extent;
    if (!kernel || !stride || !dilation ||
        vx_sd_checked_multiply(dilation, kernel - 1u, &effective) ||
        vx_sd_checked_add(effective, 1u, &effective) ||
        vx_sd_checked_add(pad_before, pad_after, &pad_total))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                          input_index, axis);
    if (stride == 1u && pad_total == effective - 1u) {
        *output = *input;
        return VX_SHAPE_DOMAIN_STATUS_OK;
    }
    if (!vx_shape_domain_fixed_value(input, build->request->environment,
            build->request->environment_count, &fixed))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          input_index, axis);
    if (vx_sd_checked_add(fixed, pad_before, &padded) ||
        vx_sd_checked_add(padded, pad_after, &padded))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                          input_index, axis);
    numerator = (int64_t)padded - (int64_t)effective;
    quotient = numerator / (int64_t)stride;
    if (numerator < 0 && numerator % (int64_t)stride) quotient--;
    extent = quotient + 1;
    if (extent <= 0)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          input_index, axis);
    *output = vx_sd_fixed((uint64_t)extent);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_spatial_transpose_dimension(
        VxShapeDomainBuild* build,
        const VxShapeDomainDimension* input,
        uint64_t kernel,
        uint64_t stride,
        uint64_t padding,
        uint32_t input_index,
        uint32_t axis,
        VxShapeDomainDimension* output) {
    uint64_t twice_padding;
    uint64_t fixed;
    uint64_t extent;
    if (!kernel || !stride ||
        vx_sd_checked_multiply(2u, padding, &twice_padding))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                          input_index, axis);
    if (stride == 1u && twice_padding < VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER &&
        kernel == twice_padding + 1u) {
        *output = *input;
        return VX_SHAPE_DOMAIN_STATUS_OK;
    }
    if (!vx_shape_domain_fixed_value(input, build->request->environment,
            build->request->environment_count, &fixed))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          input_index, axis);
    if (vx_sd_checked_multiply(fixed - 1u, stride, &extent) ||
        vx_sd_checked_add(extent, kernel, &extent))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                          input_index, axis);
    if (extent <= twice_padding)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          input_index, axis);
    extent -= twice_padding;
    *output = vx_sd_fixed(extent);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_spatial_vector(VxShapeDomainBuild* build,
                                    const VxShapeDomainTensorDescriptor* value,
                                    uint64_t extent,
                                    uint32_t input_index) {
    if (!value || value->rank != 1u ||
        !vx_sd_spatial_constant_shape(value) ||
        (uint64_t)value->dimensions[0].value != extent)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          input_index, 0u);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int vx_sd_spatial_same_dimension_spelling(
        const VxShapeDomainDimension* left,
        const VxShapeDomainDimension* right) {
    if (left->kind == right->kind &&
        left->symbol_index == right->symbol_index &&
        left->value == right->value) return 1;
    return (left->kind == VX_SHAPE_DOMAIN_DIMENSION_SYMBOL &&
            right->kind == VX_SHAPE_DOMAIN_DIMENSION_AFFINE &&
            left->symbol_index == right->symbol_index && !right->value) ||
        (right->kind == VX_SHAPE_DOMAIN_DIMENSION_SYMBOL &&
         left->kind == VX_SHAPE_DOMAIN_DIMENSION_AFFINE &&
         right->symbol_index == left->symbol_index && !left->value);
}

static int vx_sd_spatial_broadcast_dimension(
        const VxShapeDomainDimension* left,
        const VxShapeDomainDimension* right,
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count,
        VxShapeDomainDimension* output) {
    uint64_t left_fixed;
    uint64_t right_fixed;
    int left_is_fixed = vx_shape_domain_fixed_value(
        left, environment, environment_count, &left_fixed);
    int right_is_fixed = vx_shape_domain_fixed_value(
        right, environment, environment_count, &right_fixed);
    /* Preserve the exact ordering of logicalBroadcastDimension. In
     * particular, literal 1 broadcast against a singleton-spelled symbol
     * returns that symbol rather than collapsing it to the number one. */
    if (vx_sd_spatial_same_dimension_spelling(left, right)) {
        *output = *left;
        return 0;
    }
    if (left->kind == VX_SHAPE_DOMAIN_DIMENSION_FIXED && left->value == 1) {
        *output = *right;
        return 0;
    }
    if (right->kind == VX_SHAPE_DOMAIN_DIMENSION_FIXED && right->value == 1) {
        *output = *left;
        return 0;
    }
    if (left_is_fixed && left_fixed == 1u) {
        *output = *right;
        return 0;
    }
    if (right_is_fixed && right_fixed == 1u) {
        *output = *left;
        return 0;
    }
    if (left_is_fixed && right_is_fixed && left_fixed == right_fixed) {
        *output = vx_sd_fixed(left_fixed);
        return 0;
    }
    return -1;
}

static int vx_sd_spatial_broadcast_shape(
        const VxShapeDomainTensorDescriptor* left,
        const VxShapeDomainTensorDescriptor* right,
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count,
        VxShapeDomainDimension output[8],
        uint32_t* output_rank) {
    uint32_t rank = left->rank > right->rank ? left->rank : right->rank;
    if (rank > 8u) return -1;
    for (uint32_t axis = 0u; axis < rank; ++axis) {
        int64_t left_axis = (int64_t)axis - (int64_t)(rank - left->rank);
        int64_t right_axis = (int64_t)axis - (int64_t)(rank - right->rank);
        VxShapeDomainDimension left_dimension = left_axis < 0
            ? vx_sd_fixed(1u) : left->dimensions[(uint32_t)left_axis];
        VxShapeDomainDimension right_dimension = right_axis < 0
            ? vx_sd_fixed(1u) : right->dimensions[(uint32_t)right_axis];
        if (vx_sd_spatial_broadcast_dimension(&left_dimension,
                &right_dimension, environment, environment_count,
                &output[axis])) return -1;
    }
    *output_rank = rank;
    return 0;
}

static int32_t vx_sd_spatial_batch_matmul(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"a", "b"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* left;
    const VxShapeDomainNamedTensor* right;
    VxShapeDomainTensorDescriptor left_batch;
    VxShapeDomainTensorDescriptor right_batch;
    VxShapeDomainDimension dimensions[8];
    uint32_t batch_rank;
    if (vx_sd_exact_ports(build, INPUTS, 2u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, NULL, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    left = vx_sd_input(build->request, "a");
    right = vx_sd_input(build->request, "b");
    if (vx_sd_assert_float(build, &left->descriptor, 0u) ||
        vx_sd_assert_float(build, &right->descriptor, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (left->descriptor.rank < 2u || right->descriptor.rank < 2u ||
        left->descriptor.rank > 8u || right->descriptor.rank > 8u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          left->descriptor.rank, right->descriptor.rank);
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
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_BROADCAST;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32, dimensions,
                            batch_rank + 2u, NULL);
}

static int32_t vx_sd_spatial_conv1d(VxShapeDomainBuild* build) {
    static const char* const REQUIRED[] = {"input", "weight"};
    static const char* const OPTIONAL[] = {"bias"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const ALLOWED[] = {
        "stride", "padding", "groups", "relu", "data_layout",
        "weight_layout"
    };
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* weight;
    const VxShapeDomainNamedTensor* bias;
    VxShapeDomainDimension dimensions[3];
    uint64_t stride;
    uint64_t padding;
    uint64_t groups;
    uint64_t channels;
    uint64_t weight_channels;
    uint64_t output_channels;
    if (vx_sd_optional_ports(build, REQUIRED, 2u, OPTIONAL, 1u,
            OUTPUTS, 1u) || vx_sd_allowed_params(build, ALLOWED, 6u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    weight = vx_sd_input(build->request, "weight");
    bias = vx_sd_input(build->request, "bias");
    if (vx_sd_assert_float(build, &input->descriptor, 0u) ||
        vx_sd_assert_float(build, &weight->descriptor, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (input->descriptor.rank != 3u || weight->descriptor.rank != 3u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          input->descriptor.rank, weight->descriptor.rank);
    if (!vx_sd_spatial_constant_shape(&weight->descriptor))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE, 1u,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (vx_sd_spatial_layout(build, "data_layout", "NLC") ||
        vx_sd_spatial_layout(build, "weight_layout", "WIO") ||
        vx_sd_spatial_scalar(build, "stride", 1u, 0, &stride) ||
        vx_sd_spatial_scalar(build, "padding", 0u, 1, &padding) ||
        vx_sd_spatial_positive_integer(build, "groups", 1u, &groups) ||
        vx_sd_spatial_activation(build))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!vx_shape_domain_fixed_value(&input->descriptor.dimensions[2],
            build->request->environment,
            build->request->environment_count, &channels))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE, 0u, 2u);
    weight_channels = (uint64_t)weight->descriptor.dimensions[1].value;
    output_channels = (uint64_t)weight->descriptor.dimensions[2].value;
    if (channels % groups || output_channels % groups ||
        weight_channels != channels / groups)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          1u, 1u);
    if (bias) {
        if (vx_sd_assert_float(build, &bias->descriptor, 2u))
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
            (uint64_t)weight->descriptor.dimensions[0].value,
            stride, padding, padding, 1u, 0u, 1u, &dimensions[1]))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    dimensions[2] = vx_sd_fixed(output_channels);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            dimensions, 3u, NULL);
}

static int32_t vx_sd_spatial_conv2d(VxShapeDomainBuild* build) {
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
    const VxShapeDomainParam* layout;
    VxShapeDomainDimension dimensions[4];
    uint64_t stride[2];
    uint64_t pads[4];
    uint64_t dilation[2];
    uint64_t groups;
    uint64_t channels;
    uint64_t output_channels;
    int depthwise = 0;
    if (vx_sd_optional_ports(build, REQUIRED, 2u, OPTIONAL, 1u,
            OUTPUTS, 1u) || vx_sd_allowed_params(build, ALLOWED, 8u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    weight = vx_sd_input(build->request, "weight");
    bias = vx_sd_input(build->request, "bias");
    if (vx_sd_assert_float(build, &input->descriptor, 0u) ||
        vx_sd_assert_float(build, &weight->descriptor, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (input->descriptor.rank != 4u || weight->descriptor.rank != 4u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          input->descriptor.rank, weight->descriptor.rank);
    if (!vx_sd_spatial_constant_shape(&weight->descriptor))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE, 1u,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (vx_sd_spatial_layout(build, "data_layout", "NHWC") ||
        vx_sd_spatial_pair(build, "stride", 1u, 0, 0, stride) ||
        vx_sd_spatial_pads(build, 0, pads) ||
        vx_sd_spatial_pair(build, "dilation", 1u, 0, 0, dilation) ||
        vx_sd_spatial_positive_integer(build, "groups", 1u, &groups) ||
        vx_sd_spatial_activation(build))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    layout = vx_sd_param(build->request, "weight_layout");
    if (layout) {
        if (layout->kind != VX_SHAPE_DOMAIN_PARAM_STRING)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        if (vx_sd_string_equal(&layout->string, "HWCM")) depthwise = 1;
        else if (!vx_sd_string_equal(&layout->string, "HWIO"))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
    }
    if (!vx_shape_domain_fixed_value(&input->descriptor.dimensions[3],
            build->request->environment,
            build->request->environment_count, &channels))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE, 0u, 3u);
    if (depthwise) {
        if (groups != channels ||
            (uint64_t)weight->descriptor.dimensions[2].value != channels)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                              1u, 2u);
        if (vx_sd_checked_multiply(channels,
                (uint64_t)weight->descriptor.dimensions[3].value,
                &output_channels))
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW, 1u, 3u);
    } else {
        output_channels =
            (uint64_t)weight->descriptor.dimensions[3].value;
        if (channels % groups || output_channels % groups ||
            (uint64_t)weight->descriptor.dimensions[2].value !=
                channels / groups)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                              1u, 2u);
    }
    if (bias) {
        if (vx_sd_assert_float(build, &bias->descriptor, 2u))
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
            (uint64_t)weight->descriptor.dimensions[0].value,
            stride[0], pads[0], pads[2], dilation[0],
            0u, 1u, &dimensions[1]) ||
        vx_sd_spatial_window_dimension(build,
            &input->descriptor.dimensions[2],
            (uint64_t)weight->descriptor.dimensions[1].value,
            stride[1], pads[1], pads[3], dilation[1],
            0u, 2u, &dimensions[2]))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    dimensions[3] = vx_sd_fixed(output_channels);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            dimensions, 4u, NULL);
}

static int32_t vx_sd_spatial_conv_transpose2d(
        VxShapeDomainBuild* build) {
    static const char* const REQUIRED[] = {"input", "weight"};
    static const char* const OPTIONAL[] = {"bias"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const ALLOWED[] = {
        "kernel", "stride", "padding", "data_layout", "weight_layout"
    };
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* weight;
    const VxShapeDomainNamedTensor* bias;
    VxShapeDomainDimension dimensions[4];
    uint64_t kernel[2];
    uint64_t stride[2];
    uint64_t padding[2];
    uint64_t channels;
    uint64_t output_channels;
    if (vx_sd_optional_ports(build, REQUIRED, 2u, OPTIONAL, 1u,
            OUTPUTS, 1u) || vx_sd_allowed_params(build, ALLOWED, 5u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    weight = vx_sd_input(build->request, "weight");
    bias = vx_sd_input(build->request, "bias");
    if (vx_sd_assert_float(build, &input->descriptor, 0u) ||
        vx_sd_assert_float(build, &weight->descriptor, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (input->descriptor.rank != 4u || weight->descriptor.rank != 4u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          input->descriptor.rank, weight->descriptor.rank);
    if (!vx_sd_spatial_constant_shape(&weight->descriptor))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE, 1u,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (vx_sd_spatial_layout(build, "data_layout", "NHWC") ||
        vx_sd_spatial_layout(build, "weight_layout", "HWIO") ||
        vx_sd_spatial_pair(build, "kernel", 1u, 0, 1, kernel) ||
        vx_sd_spatial_pair(build, "stride", 1u, 0, 0, stride) ||
        vx_sd_spatial_pair(build, "padding", 0u, 1, 0, padding))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (kernel[0] != (uint64_t)weight->descriptor.dimensions[0].value ||
        kernel[1] != (uint64_t)weight->descriptor.dimensions[1].value)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (!vx_shape_domain_fixed_value(&input->descriptor.dimensions[3],
            build->request->environment,
            build->request->environment_count, &channels))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE, 0u, 3u);
    if ((uint64_t)weight->descriptor.dimensions[2].value != channels)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          1u, 2u);
    output_channels = (uint64_t)weight->descriptor.dimensions[3].value;
    if (bias) {
        if (vx_sd_assert_float(build, &bias->descriptor, 2u))
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
        if (!vx_sd_spatial_constant_shape(&bias->descriptor))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                              2u, VX_SHAPE_DOMAIN_INDEX_NONE);
        if (vx_sd_spatial_vector(build, &bias->descriptor,
                                 output_channels, 2u))
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
    }
    dimensions[0] = input->descriptor.dimensions[0];
    if (vx_sd_spatial_transpose_dimension(build,
            &input->descriptor.dimensions[1], kernel[0], stride[0],
            padding[0], 0u, 1u, &dimensions[1]) ||
        vx_sd_spatial_transpose_dimension(build,
            &input->descriptor.dimensions[2], kernel[1], stride[1],
            padding[1], 0u, 2u, &dimensions[2]))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    dimensions[3] = vx_sd_fixed(output_channels);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            dimensions, 4u, NULL);
}

static int32_t vx_sd_spatial_pool2d(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const MAX_ALLOWED[] = {
        "kernel", "stride", "padding", "pads", "dilation", "ceil_mode",
        "data_layout"
    };
    static const char* const AVERAGE_ALLOWED[] = {
        "kernel", "stride", "padding", "pads", "dilation", "ceil_mode",
        "count_include_pad", "auto_pad", "data_layout"
    };
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainParam* auto_pad;
    VxShapeDomainDimension dimensions[4];
    uint64_t kernel[2];
    uint64_t stride[2];
    uint64_t dilation[2];
    uint64_t pads[4];
    int average = build->request->operator_kind == VX_OP_AVERAGE_POOL_2D;
    if (vx_sd_exact_ports(build, INPUTS, 1u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build,
            average ? AVERAGE_ALLOWED : MAX_ALLOWED,
            average ? 9u : 7u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (input->descriptor.rank != 4u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, input->descriptor.rank);
    if (input->descriptor.dtype == VX_DTYPE_F32) {
        if (!vx_sd_unquantized(&input->descriptor))
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u,
                VX_SHAPE_DOMAIN_INDEX_NONE);
    } else if (!average &&
               (input->descriptor.dtype == VX_DTYPE_I8 ||
                input->descriptor.dtype == VX_DTYPE_U8)) {
        if (input->descriptor.quantization.scheme != 1)
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u,
                VX_SHAPE_DOMAIN_INDEX_NONE);
    } else {
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                          0u, VX_SHAPE_DOMAIN_INDEX_NONE);
    }
    if (vx_sd_spatial_layout(build, "data_layout", "NHWC") ||
        vx_sd_spatial_false_or_absent(build, "ceil_mode"))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (average) {
        if (vx_sd_spatial_false_or_absent(build, "count_include_pad"))
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
        auto_pad = vx_sd_param(build->request, "auto_pad");
        if (auto_pad &&
            (auto_pad->kind != VX_SHAPE_DOMAIN_PARAM_STRING ||
             (!vx_sd_string_equal(&auto_pad->string, "") &&
              !vx_sd_string_equal(&auto_pad->string, "NOTSET"))))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
    }
    if (vx_sd_spatial_pair(build, "dilation", 1u, 0, 0, dilation) ||
        dilation[0] != 1u || dilation[1] != 1u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (vx_sd_spatial_pair(build, "kernel", 1u, 0, 1, kernel) ||
        vx_sd_spatial_pair(build, "stride", 1u, 0, 0, stride) ||
        vx_sd_spatial_pads(build, average, pads))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    dimensions[0] = input->descriptor.dimensions[0];
    if (vx_sd_spatial_window_dimension(build,
            &input->descriptor.dimensions[1], kernel[0], stride[0],
            pads[0], pads[2], 1u, 0u, 1u, &dimensions[1]) ||
        vx_sd_spatial_window_dimension(build,
            &input->descriptor.dimensions[2], kernel[1], stride[1],
            pads[1], pads[3], 1u, 0u, 2u, &dimensions[2]))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    dimensions[3] = input->descriptor.dimensions[3];
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", input->descriptor.dtype,
                            dimensions, 4u,
                            &input->descriptor.quantization);
}

static int32_t vx_sd_spatial_global_average_pool(
        VxShapeDomainBuild* build) {
    static const char* const ALLOWED[] = {"data_layout"};
    const VxShapeDomainNamedTensor* input;
    VxShapeDomainDimension dimensions[4];
    if (vx_sd_spatial_single_ports(build) ||
        vx_sd_allowed_params(build, ALLOWED, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (vx_sd_assert_float(build, &input->descriptor, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (input->descriptor.rank != 4u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, input->descriptor.rank);
    if (vx_sd_spatial_layout(build, "data_layout", "NHWC"))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    dimensions[0] = input->descriptor.dimensions[0];
    dimensions[1] = vx_sd_fixed(1u);
    dimensions[2] = vx_sd_fixed(1u);
    dimensions[3] = input->descriptor.dimensions[3];
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            dimensions, 4u, NULL);
}

static int32_t vx_sd_spatial_resize_params(VxShapeDomainBuild* build,
                                           int byte_storage) {
    static const char* const ALLOWED[] = {
        "mode", "coordinate_transformation_mode", "coordinate_transform_mode",
        "nearest_mode", "align_corners", "antialias", "data_layout"
    };
    const VxShapeDomainParam* mode;
    const VxShapeDomainParam* transform;
    const VxShapeDomainParam* nearest_mode;
    int nearest;
    if (vx_sd_allowed_params(build, ALLOWED, 7u) ||
        vx_sd_spatial_layout(build, "data_layout", "NHWC"))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (vx_sd_param(build->request, "coordinate_transform_mode"))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    mode = vx_sd_param(build->request, "mode");
    if (mode && mode->kind != VX_SHAPE_DOMAIN_PARAM_STRING)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (build->request->operator_kind == VX_OP_RESIZE_NEAREST_2D) {
        if (mode && !vx_sd_string_equal(&mode->string, "nearest"))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        nearest = 1;
    } else {
        if (mode && !vx_sd_string_equal(&mode->string, "nearest") &&
            !vx_sd_string_equal(&mode->string, "linear"))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        nearest = mode && vx_sd_string_equal(&mode->string, "nearest");
    }
    if (byte_storage && !nearest)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    transform = vx_sd_param(build->request, "coordinate_transformation_mode");
    if (transform &&
        (transform->kind != VX_SHAPE_DOMAIN_PARAM_STRING ||
         !vx_sd_string_equal(&transform->string,
                             nearest ? "asymmetric" : "half_pixel")))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    nearest_mode = vx_sd_param(build->request, "nearest_mode");
    if (nearest) {
        if (nearest_mode &&
            (nearest_mode->kind != VX_SHAPE_DOMAIN_PARAM_STRING ||
             !vx_sd_string_equal(&nearest_mode->string, "floor")))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
    } else if (nearest_mode) {
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    }
    if (vx_sd_spatial_false_or_absent(build, "align_corners") ||
        vx_sd_spatial_false_or_absent(build, "antialias"))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_spatial_resize(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* declared;
    int byte_storage = 0;
    if (vx_sd_exact_ports(build, INPUTS, 1u, OUTPUTS, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!build->request->use_declared_outputs)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    input = vx_sd_input(build->request, "input");
    declared = vx_sd_named(build->request->declared_outputs,
                           build->request->declared_output_count, "out");
    if (!declared || input->descriptor.rank != 4u ||
        declared->descriptor.rank != 4u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          input->descriptor.rank,
                          declared ? declared->descriptor.rank : 0u);
    if (input->descriptor.dtype == VX_DTYPE_F32) {
        if (!vx_sd_unquantized(&input->descriptor))
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u,
                VX_SHAPE_DOMAIN_INDEX_NONE);
        if (declared->descriptor.dtype != VX_DTYPE_F32)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                              0u, VX_SHAPE_DOMAIN_INDEX_NONE);
        if (!vx_sd_unquantized(&declared->descriptor))
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u,
                VX_SHAPE_DOMAIN_INDEX_NONE);
    } else {
        byte_storage = 1;
        if ((input->descriptor.dtype != VX_DTYPE_I8 &&
             input->descriptor.dtype != VX_DTYPE_U8) ||
            input->descriptor.quantization.scheme != 1)
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u,
                VX_SHAPE_DOMAIN_INDEX_NONE);
        if (declared->descriptor.dtype != input->descriptor.dtype ||
            !vx_sd_quantization_equal(&input->descriptor.quantization,
                                      &declared->descriptor.quantization))
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u,
                VX_SHAPE_DOMAIN_INDEX_NONE);
    }
    if (vx_sd_spatial_resize_params(build, byte_storage))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!vx_shape_domain_dimensions_equal(
            &declared->descriptor.dimensions[0],
            &input->descriptor.dimensions[0], build->request->environment,
            build->request->environment_count) ||
        !vx_shape_domain_dimensions_equal(
            &declared->descriptor.dimensions[3],
            &input->descriptor.dimensions[3], build->request->environment,
            build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, VX_SHAPE_DOMAIN_INDEX_NONE);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", declared->descriptor.dtype,
                            declared->descriptor.dimensions, 4u,
                            &declared->descriptor.quantization);
}

static int32_t vx_sd_spatial_upsample_nearest2d(
        VxShapeDomainBuild* build) {
    static const char* const ALLOWED[] = {"data_layout"};
    const VxShapeDomainNamedTensor* input;
    VxShapeDomainDimension dimensions[4];
    uint64_t fixed;
    uint64_t doubled;
    if (vx_sd_spatial_single_ports(build) ||
        vx_sd_allowed_params(build, ALLOWED, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (vx_sd_assert_float(build, &input->descriptor, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (input->descriptor.rank != 4u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, input->descriptor.rank);
    if (vx_sd_spatial_layout(build, "data_layout", "NHWC"))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    dimensions[0] = input->descriptor.dimensions[0];
    for (uint32_t axis = 1u; axis <= 2u; ++axis) {
        if (!vx_shape_domain_fixed_value(&input->descriptor.dimensions[axis],
                build->request->environment,
                build->request->environment_count, &fixed))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                              0u, axis);
        if (vx_sd_checked_multiply(fixed, 2u, &doubled))
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW, 0u, axis);
        dimensions[axis] = vx_sd_fixed(doubled);
    }
    dimensions[3] = input->descriptor.dimensions[3];
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            dimensions, 4u, NULL);
}

static int32_t vx_sd_prove_spatial(VxShapeDomainBuild* build) {
    switch (build->request->operator_kind) {
        case VX_OP_BATCH_MATMUL:
            return vx_sd_spatial_batch_matmul(build);
        case VX_OP_CONV_1D:
            return vx_sd_spatial_conv1d(build);
        case VX_OP_CONV_2D:
            return vx_sd_spatial_conv2d(build);
        case VX_OP_CONV_TRANSPOSE_2D:
            return vx_sd_spatial_conv_transpose2d(build);
        case VX_OP_MAX_POOL_2D:
        case VX_OP_AVERAGE_POOL_2D:
            return vx_sd_spatial_pool2d(build);
        case VX_OP_GLOBAL_AVERAGE_POOL:
            return vx_sd_spatial_global_average_pool(build);
        case VX_OP_RESIZE:
        case VX_OP_RESIZE_NEAREST_2D:
            return vx_sd_spatial_resize(build);
        case VX_OP_UPSAMPLE_NEAREST_2D:
            return vx_sd_spatial_upsample_nearest2d(build);
        default:
            return VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED;
    }
}
