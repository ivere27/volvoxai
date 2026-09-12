/* Direct logical shape-domain contracts. This fragment is included only by
 * shape_domain_contract.c so the helpers stay private and allocation-free. */

static int vx_sd_number_param(const VxShapeDomainParam* param,
                              double* output) {
    if (!param || !output || param->kind != VX_SHAPE_DOMAIN_PARAM_NUMBER)
        return -1;
    *output = param->number;
    return 0;
}

static int32_t vx_sd_direct_single_ports(VxShapeDomainBuild* build,
                                         const char* input_name) {
    static const char* const OUTPUTS[] = {"out"};
    const char* inputs[1];
    inputs[0] = input_name;
    return vx_sd_exact_ports(build, inputs, 1u, OUTPUTS, 1u);
}

static int32_t vx_sd_direct_identity(VxShapeDomainBuild* build) {
    const VxShapeDomainNamedTensor* input;
    if (vx_sd_direct_single_ports(build, "input") ||
        vx_sd_allowed_params(build, NULL, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_set_output(build, "out", input->descriptor.dtype,
                            input->descriptor.dimensions,
                            input->descriptor.rank,
                            &input->descriptor.quantization);
}

static int32_t vx_sd_direct_activation_params(
        VxShapeDomainBuild* build,
        const VxShapeDomainTensorDescriptor* input) {
    static const char* const ALPHA[] = {"alpha"};
    static const char* const APPROXIMATE[] = {"approximate"};
    static const char* const CLIP_BOUNDS[] = {"min", "max"};
    static const char* const AXIS[] = {"axis"};
    const VxShapeDomainParam* param;
    double minimum;
    double maximum;
    double number;
    int64_t axis = -1;
    switch (build->request->operator_kind) {
        case VX_OP_LEAKY_RELU:
            if (vx_sd_allowed_params(build, ALPHA, 1u))
                return VX_SHAPE_DOMAIN_STATUS_INVALID;
            param = vx_sd_param(build->request, "alpha");
            if (param && (vx_sd_number_param(param, &number) ||
                          !isfinite(number)))
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                                  VX_SHAPE_DOMAIN_INDEX_NONE,
                                  VX_SHAPE_DOMAIN_INDEX_NONE);
            return VX_SHAPE_DOMAIN_STATUS_OK;
        case VX_OP_GELU:
            if (vx_sd_allowed_params(build, APPROXIMATE, 1u))
                return VX_SHAPE_DOMAIN_STATUS_INVALID;
            param = vx_sd_param(build->request, "approximate");
            if (param && (param->kind != VX_SHAPE_DOMAIN_PARAM_STRING ||
                          (!vx_sd_string_equal(&param->string, "none") &&
                           !vx_sd_string_equal(&param->string, "tanh"))))
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                                  VX_SHAPE_DOMAIN_INDEX_NONE,
                                  VX_SHAPE_DOMAIN_INDEX_NONE);
            return VX_SHAPE_DOMAIN_STATUS_OK;
        case VX_OP_CLIP:
            if (vx_sd_allowed_params(build, CLIP_BOUNDS, 2u))
                return VX_SHAPE_DOMAIN_STATUS_INVALID;
            minimum = input->dtype == VX_DTYPE_I32
                ? -2147483648.0 : -INFINITY;
            maximum = input->dtype == VX_DTYPE_I32
                ? 2147483647.0 : INFINITY;
            param = vx_sd_param(build->request, "min");
            if (param && vx_sd_number_param(param, &minimum))
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                                  VX_SHAPE_DOMAIN_INDEX_NONE,
                                  VX_SHAPE_DOMAIN_INDEX_NONE);
            param = vx_sd_param(build->request, "max");
            if (param && vx_sd_number_param(param, &maximum))
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                                  VX_SHAPE_DOMAIN_INDEX_NONE,
                                  VX_SHAPE_DOMAIN_INDEX_NONE);
            if (isnan(minimum) || isnan(maximum) || minimum > maximum ||
                (input->dtype == VX_DTYPE_I32 &&
                 (!isfinite(minimum) || !isfinite(maximum) ||
                  minimum < -2147483648.0 || maximum > 2147483647.0 ||
                  minimum != (double)(int64_t)minimum ||
                  maximum != (double)(int64_t)maximum)))
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                                  VX_SHAPE_DOMAIN_INDEX_NONE,
                                  VX_SHAPE_DOMAIN_INDEX_NONE);
            return VX_SHAPE_DOMAIN_STATUS_OK;
        case VX_OP_SOFTMAX:
        case VX_OP_LOG_SOFTMAX:
            if (vx_sd_allowed_params(build, AXIS, 1u))
                return VX_SHAPE_DOMAIN_STATUS_INVALID;
            if (!input->rank)
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                                  0u, VX_SHAPE_DOMAIN_INDEX_NONE);
            if (vx_sd_number_integer(build,
                    vx_sd_param(build->request, "axis"), -1, &axis))
                return VX_SHAPE_DOMAIN_STATUS_INVALID;
            if (axis < 0) axis += input->rank;
            if (axis != (int64_t)input->rank - 1)
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                                  VX_SHAPE_DOMAIN_INDEX_NONE,
                                  VX_SHAPE_DOMAIN_INDEX_NONE);
            return VX_SHAPE_DOMAIN_STATUS_OK;
        default:
            return vx_sd_allowed_params(build, NULL, 0u);
    }
}

static int vx_sd_direct_activation_kind(VxOperatorKind kind) {
    switch (kind) {
        case VX_OP_RELU:
        case VX_OP_LEAKY_RELU:
        case VX_OP_GELU:
        case VX_OP_SILU:
        case VX_OP_SIGMOID:
        case VX_OP_HARD_SWISH:
        case VX_OP_HARD_SIGMOID:
        case VX_OP_TANH:
        case VX_OP_SIN:
        case VX_OP_COS:
        case VX_OP_CLIP:
        case VX_OP_SOFTMAX:
        case VX_OP_LOG_SOFTMAX:
            return 1;
        default:
            return 0;
    }
}

static int32_t vx_sd_direct_activation(VxShapeDomainBuild* build) {
    const VxShapeDomainNamedTensor* input;
    static const char* const REQUIRED[] = {"input"};
    static const char* const OPTIONAL[] = {"min", "max"};
    static const char* const OUTPUTS[] = {"out"};
    int clip = build->request->operator_kind == VX_OP_CLIP;
    if (clip ? vx_sd_optional_ports(build, REQUIRED, 1u, OPTIONAL, 2u, OUTPUTS, 1u)
             : vx_sd_direct_single_ports(build, "input"))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (clip) for (uint32_t index = 0; index < 2; ++index) {
        const VxShapeDomainNamedTensor* bound = vx_sd_input(build->request, OPTIONAL[index]);
        if (!bound) continue;
        if (bound->descriptor.dtype != input->descriptor.dtype ||
            !vx_sd_unquantized(&bound->descriptor))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE, index + 1u,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        for (uint32_t axis = 0; axis < bound->descriptor.rank; ++axis) {
            uint64_t extent;
            if (!vx_shape_domain_fixed_value(&bound->descriptor.dimensions[axis],
                    build->request->environment, build->request->environment_count, &extent) || extent != 1u)
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH, index + 1u, axis);
        }
    }
    if (build->request->operator_kind == VX_OP_CLIP &&
        input->descriptor.dtype == VX_DTYPE_I32) {
        if (!vx_sd_unquantized(&input->descriptor))
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u,
                VX_SHAPE_DOMAIN_INDEX_NONE);
    } else if (vx_sd_assert_float(build, &input->descriptor, 0u)) {
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    }
    if (vx_sd_direct_activation_params(build, &input->descriptor))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_set_output(build, "out", input->descriptor.dtype,
                            input->descriptor.dimensions,
                            input->descriptor.rank, NULL);
}

static int32_t vx_sd_direct_prelu(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input", "slope"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* slope;
    uint64_t slope_extent;
    uint64_t feature;
    if (vx_sd_exact_ports(build, INPUTS, 2u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, NULL, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    slope = vx_sd_input(build->request, "slope");
    if (vx_sd_assert_float(build, &input->descriptor, 0u) ||
        vx_sd_assert_float(build, &slope->descriptor, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!input->descriptor.rank || slope->descriptor.rank != 1u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          input->descriptor.rank, slope->descriptor.rank);
    /* constantLogicalShape rejects a symbolic spelling even when its declared
     * interval is a singleton. */
    if (slope->descriptor.dimensions[0].kind !=
            VX_SHAPE_DOMAIN_DIMENSION_FIXED)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE, 1u, 0u);
    slope_extent = (uint64_t)slope->descriptor.dimensions[0].value;
    if (slope_extent != 1u &&
        (!vx_shape_domain_fixed_value(
            &input->descriptor.dimensions[input->descriptor.rank - 1u],
            build->request->environment,
            build->request->environment_count, &feature) ||
         feature != slope_extent))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          1u, 0u);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            input->descriptor.dimensions,
                            input->descriptor.rank, NULL);
}

static int32_t vx_sd_direct_cast(VxShapeDomainBuild* build) {
    static const char* const TO[] = {"to"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainParam* param;
    int32_t dtype;
    if (vx_sd_direct_single_ports(build, "input") ||
        vx_sd_allowed_params(build, TO, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (!vx_sd_unquantized(&input->descriptor))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION,
                          0u, VX_SHAPE_DOMAIN_INDEX_NONE);
    param = vx_sd_param(build->request, "to");
    if (!param || param->kind != VX_SHAPE_DOMAIN_PARAM_STRING)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (vx_sd_string_equal(&param->string, "float32")) dtype = VX_DTYPE_F32;
    else if (vx_sd_string_equal(&param->string, "int32")) dtype = VX_DTYPE_I32;
    else if (vx_sd_string_equal(&param->string, "int8")) dtype = VX_DTYPE_I8;
    else if (vx_sd_string_equal(&param->string, "uint8")) dtype = VX_DTYPE_U8;
    else return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                           VX_SHAPE_DOMAIN_INDEX_NONE,
                           VX_SHAPE_DOMAIN_INDEX_NONE);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_set_output(build, "out", dtype,
                            input->descriptor.dimensions,
                            input->descriptor.rank, NULL);
}

static int32_t vx_sd_direct_arithmetic(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"a", "b"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const RELU[] = {"relu"};
    const VxShapeDomainNamedTensor* left;
    const VxShapeDomainNamedTensor* right;
    VxShapeDomainDimension dimensions[8];
    uint32_t rank;
    int64_t relu;
    if (vx_sd_exact_ports(build, INPUTS, 2u, OUTPUTS, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    left = vx_sd_input(build->request, "a");
    right = vx_sd_input(build->request, "b");
    if (vx_sd_assert_float(build, &left->descriptor, 0u) ||
        vx_sd_assert_float(build, &right->descriptor, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (build->request->operator_kind == VX_OP_ADD) {
        if (vx_sd_allowed_params(build, RELU, 1u) ||
            vx_sd_number_integer(build,
                vx_sd_param(build->request, "relu"), 0, &relu))
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
        if (relu < 0 || relu > 2)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
    } else if (vx_sd_allowed_params(build, NULL, 0u)) {
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    }
    if (vx_sd_broadcast_shape(&left->descriptor, &right->descriptor,
            build->request->environment,
            build->request->environment_count, dimensions, &rank))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          0u, 1u);
    build->result->fact_kind = vx_sd_shapes_equal(&left->descriptor, &right->descriptor,
        build->request->environment, build->request->environment_count)
        ? VX_INTERNAL_SHAPE_DOMAIN_FACT_EXACT_BINARY : VX_INTERNAL_SHAPE_DOMAIN_FACT_BROADCAST;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            dimensions, rank, NULL);
}

static int vx_sd_direct_constant_shape(
        const VxShapeDomainTensorDescriptor* descriptor) {
    if (!descriptor || (descriptor->rank && !descriptor->dimensions)) return 0;
    for (uint32_t axis = 0u; axis < descriptor->rank; ++axis)
        if (descriptor->dimensions[axis].kind !=
            VX_SHAPE_DOMAIN_DIMENSION_FIXED) return 0;
    return 1;
}

static int32_t vx_sd_direct_vector(VxShapeDomainBuild* build,
                                   const VxShapeDomainTensorDescriptor* value,
                                   uint64_t extent,
                                   uint32_t input_index) {
    VxShapeDomainDimension expected = vx_sd_fixed(extent);
    if (!value || value->rank != 1u ||
        !vx_shape_domain_dimensions_equal(&value->dimensions[0], &expected,
            build->request->environment,
            build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          input_index, 0u);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static VxShapeDomainDimension* vx_sd_direct_output_dimensions(
        VxShapeDomainBuild* build,
        const char* name,
        uint32_t rank) {
    uint32_t index;
    if (vx_sd_result_output_index(build, name, &index) ||
        build->result->outputs[index].descriptor.rank != rank) return NULL;
    return (VxShapeDomainDimension*)
        build->result->outputs[index].descriptor.dimensions;
}

static int32_t vx_sd_direct_quantization_parameters(
        VxShapeDomainBuild* build,
        const VxShapeDomainTensorDescriptor* input,
        const VxShapeDomainTensorDescriptor* scale,
        const VxShapeDomainTensorDescriptor* zero_point,
        int32_t storage_dtype,
        const VxShapeDomainQuantization* quantization) {
    uint64_t extent = 1u;
    if (!quantization ||
        (quantization->scheme != 1 && quantization->scheme != 2))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION,
                          0u, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (vx_sd_assert_float(build, scale, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!vx_sd_direct_constant_shape(scale))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          1u, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (zero_point) {
        if (zero_point->dtype != storage_dtype)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                              2u, VX_SHAPE_DOMAIN_INDEX_NONE);
        if (!vx_sd_unquantized(zero_point))
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 2u,
                VX_SHAPE_DOMAIN_INDEX_NONE);
        if (!vx_sd_direct_constant_shape(zero_point))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                              2u, VX_SHAPE_DOMAIN_INDEX_NONE);
    }
    if (quantization->scheme == 2) {
        if (!input || quantization->axis >= input->rank ||
            !vx_shape_domain_fixed_value(
                &input->dimensions[quantization->axis],
                build->request->environment,
                build->request->environment_count, &extent))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                              0u, quantization->axis);
    }
    if (vx_sd_direct_vector(build, scale, extent, 1u) ||
        (zero_point &&
         vx_sd_direct_vector(build, zero_point, extent, 2u)))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_direct_quantize(VxShapeDomainBuild* build,
                                     int dequantize) {
    static const char* const REQUIRED[] = {"input", "scale"};
    static const char* const OPTIONAL[] = {"zero_point"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* scale;
    const VxShapeDomainNamedTensor* zero;
    const VxShapeDomainNamedTensor* declared;
    const VxShapeDomainQuantization* quantization;
    int32_t storage_dtype;
    if (vx_sd_optional_ports(build, REQUIRED, 2u, OPTIONAL, 1u,
            OUTPUTS, 1u) || vx_sd_allowed_params(build, NULL, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    scale = vx_sd_input(build->request, "scale");
    zero = vx_sd_input(build->request, "zero_point");
    if (!dequantize) {
        if (vx_sd_assert_float(build, &input->descriptor, 0u) ||
            !build->request->use_declared_outputs)
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
        declared = vx_sd_named(build->request->declared_outputs,
                               build->request->declared_output_count, "out");
        if (!declared)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS,
                              0u, 1u);
        storage_dtype = declared->descriptor.dtype;
        quantization = &declared->descriptor.quantization;
        if ((storage_dtype != VX_DTYPE_I8 && storage_dtype != VX_DTYPE_U8) ||
            quantization->scheme == 0)
            return vx_sd_fail(build,
                storage_dtype == VX_DTYPE_I8 || storage_dtype == VX_DTYPE_U8
                    ? VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION
                    : VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                0u, VX_SHAPE_DOMAIN_INDEX_NONE);
        if (!vx_sd_shapes_equal(&input->descriptor, &declared->descriptor,
                build->request->environment,
                build->request->environment_count))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                              0u, VX_SHAPE_DOMAIN_INDEX_NONE);
    } else {
        storage_dtype = input->descriptor.dtype;
        quantization = &input->descriptor.quantization;
        /* Float and integer affine inputs are also used by differentiable
         * dequantization. Only byte storage carries quantization metadata. */
        if (storage_dtype == VX_DTYPE_F32 || storage_dtype == VX_DTYPE_I32) {
            if (!vx_sd_unquantized(&input->descriptor) ||
                vx_sd_assert_float(build, &scale->descriptor, 1u) ||
                vx_sd_direct_vector(build, &scale->descriptor, 1u, 1u))
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION,
                                  0u, VX_SHAPE_DOMAIN_INDEX_NONE);
            if (zero && ((zero->descriptor.dtype != VX_DTYPE_F32 &&
                          zero->descriptor.dtype != VX_DTYPE_I32 &&
                          zero->descriptor.dtype != VX_DTYPE_I8 &&
                          zero->descriptor.dtype != VX_DTYPE_U8) ||
                         !vx_sd_unquantized(&zero->descriptor) ||
                         vx_sd_direct_vector(build, &zero->descriptor, 1u, 2u)))
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                                  2u, VX_SHAPE_DOMAIN_INDEX_NONE);
            build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT;
            return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                input->descriptor.dimensions, input->descriptor.rank, NULL);
        }
        if (storage_dtype != VX_DTYPE_I8 && storage_dtype != VX_DTYPE_U8)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                              0u, VX_SHAPE_DOMAIN_INDEX_NONE);
        if (quantization->scheme == 0)
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u,
                VX_SHAPE_DOMAIN_INDEX_NONE);
    }
    if (vx_sd_direct_quantization_parameters(build, &input->descriptor,
            &scale->descriptor, zero ? &zero->descriptor : NULL,
            storage_dtype, quantization))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    build->result->fact_kind = dequantize
        ? VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT
        : VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_set_output(build, "out",
        dequantize ? VX_DTYPE_F32 : storage_dtype,
        input->descriptor.dimensions, input->descriptor.rank,
        dequantize ? NULL : quantization);
}

static int32_t vx_sd_direct_normalization_params(
        VxShapeDomainBuild* build,
        uint64_t feature) {
    static const char* const ALLOWED[] = {"eps", "d_model"};
    const VxShapeDomainParam* eps;
    const VxShapeDomainParam* model;
    int64_t extent = 0;
    if (vx_sd_allowed_params(build, ALLOWED, 2u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    eps = vx_sd_param(build->request, "eps");
    if (eps && (eps->kind != VX_SHAPE_DOMAIN_PARAM_NUMBER ||
                !isfinite(eps->number) || eps->number <= 0.0))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    model = vx_sd_param(build->request, "d_model");
    if (model) {
        if (vx_sd_number_integer(build, model, 0, &extent) || extent <= 0 ||
            (uint64_t)extent != feature)
            return vx_sd_fail(build,
                extent > 0 ? VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH
                           : VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                VX_SHAPE_DOMAIN_INDEX_NONE,
                VX_SHAPE_DOMAIN_INDEX_NONE);
    }
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_direct_feature_norm(VxShapeDomainBuild* build) {
    static const char* const REQUIRED[] = {"input", "weight"};
    static const char* const OPTIONAL[] = {"bias"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* weight;
    const VxShapeDomainNamedTensor* bias;
    uint64_t feature;
    int layer = build->request->operator_kind == VX_OP_LAYER_NORM;
    if ((layer
            ? vx_sd_optional_ports(build, REQUIRED, 2u, OPTIONAL, 1u,
                                   OUTPUTS, 1u)
            : vx_sd_exact_ports(build, REQUIRED, 2u, OUTPUTS, 1u)))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    weight = vx_sd_input(build->request, "weight");
    bias = vx_sd_input(build->request, "bias");
    if (vx_sd_assert_float(build, &input->descriptor, 0u) ||
        vx_sd_assert_float(build, &weight->descriptor, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!input->descriptor.rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, 0u);
    if (!vx_sd_direct_constant_shape(&weight->descriptor) ||
        !vx_shape_domain_fixed_value(
            &input->descriptor.dimensions[input->descriptor.rank - 1u],
            build->request->environment,
            build->request->environment_count, &feature))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          0u, input->descriptor.rank);
    if (vx_sd_direct_vector(build, &weight->descriptor, feature, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (bias && (vx_sd_assert_float(build, &bias->descriptor, 2u) ||
                 !vx_sd_direct_constant_shape(&bias->descriptor) ||
                 vx_sd_direct_vector(build, &bias->descriptor, feature, 2u)))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (vx_sd_direct_normalization_params(build, feature))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            input->descriptor.dimensions,
                            input->descriptor.rank, NULL);
}

static int32_t vx_sd_direct_group_norm(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input", "weight", "bias"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const ALLOWED[] = {"num_groups", "eps"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* weight;
    const VxShapeDomainNamedTensor* bias;
    const VxShapeDomainParam* groups_param;
    const VxShapeDomainParam* eps;
    uint64_t channels;
    int64_t groups;
    if (vx_sd_exact_ports(build, INPUTS, 3u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, ALLOWED, 2u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    weight = vx_sd_input(build->request, "weight");
    bias = vx_sd_input(build->request, "bias");
    if (vx_sd_assert_float(build, &input->descriptor, 0u) ||
        vx_sd_assert_float(build, &weight->descriptor, 1u) ||
        vx_sd_assert_float(build, &bias->descriptor, 2u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (input->descriptor.rank != 4u ||
        !vx_shape_domain_fixed_value(&input->descriptor.dimensions[3],
            build->request->environment,
            build->request->environment_count, &channels))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          0u, 3u);
    if (vx_sd_direct_vector(build, &weight->descriptor, channels, 1u) ||
        vx_sd_direct_vector(build, &bias->descriptor, channels, 2u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    eps = vx_sd_param(build->request, "eps");
    if (eps && (eps->kind != VX_SHAPE_DOMAIN_PARAM_NUMBER ||
                !isfinite(eps->number) || eps->number <= 0.0))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    groups_param = vx_sd_param(build->request, "num_groups");
    if (!groups_param || vx_sd_number_integer(build, groups_param, 0, &groups) ||
        groups <= 0)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (channels % (uint64_t)groups)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, 3u);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            input->descriptor.dimensions,
                            input->descriptor.rank, NULL);
}

static int32_t vx_sd_direct_dense(VxShapeDomainBuild* build) {
    static const char* const REQUIRED[] = {"input", "weight"};
    static const char* const OPTIONAL[] = {"bias"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const ALLOWED[] = {"weight_layout", "transB"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* weight;
    const VxShapeDomainNamedTensor* bias;
    const VxShapeDomainParam* layout_param;
    const VxShapeDomainParam* transpose_param;
    VxShapeDomainDimension* dimensions;
    uint64_t contracted;
    uint64_t input_contracted;
    uint64_t output_feature;
    int dout_din;
    if (vx_sd_optional_ports(build, REQUIRED, 2u, OPTIONAL, 1u,
            OUTPUTS, 1u) || vx_sd_allowed_params(build, ALLOWED, 2u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    weight = vx_sd_input(build->request, "weight");
    bias = vx_sd_input(build->request, "bias");
    if (vx_sd_assert_float(build, &input->descriptor, 0u) ||
        vx_sd_assert_float(build, &weight->descriptor, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!input->descriptor.rank || weight->descriptor.rank != 2u ||
        !vx_sd_direct_constant_shape(&weight->descriptor))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          1u, weight->descriptor.rank);
    layout_param = vx_sd_param(build->request, "weight_layout");
    transpose_param = vx_sd_param(build->request, "transB");
    if (layout_param && transpose_param)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    dout_din = build->request->operator_kind == VX_OP_LINEAR;
    if (layout_param) {
        if (layout_param->kind != VX_SHAPE_DOMAIN_PARAM_STRING ||
            (!vx_sd_string_equal(&layout_param->string, "din_dout") &&
             !vx_sd_string_equal(&layout_param->string, "dout_din")))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        dout_din = vx_sd_string_equal(
            &layout_param->string, "dout_din");
    } else if (transpose_param) {
        if (transpose_param->kind != VX_SHAPE_DOMAIN_PARAM_BOOLEAN ||
            (transpose_param->boolean != 0 && transpose_param->boolean != 1))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        dout_din = transpose_param->boolean;
    }
    contracted = (uint64_t)weight->descriptor.dimensions[dout_din ? 1u : 0u].value;
    output_feature =
        (uint64_t)weight->descriptor.dimensions[dout_din ? 0u : 1u].value;
    if (!vx_shape_domain_fixed_value(
            &input->descriptor.dimensions[input->descriptor.rank - 1u],
            build->request->environment,
            build->request->environment_count, &input_contracted))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          0u, input->descriptor.rank - 1u);
    if (input_contracted != contracted)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, input->descriptor.rank - 1u);
    if (bias && (vx_sd_assert_float(build, &bias->descriptor, 2u) ||
                 !vx_sd_direct_constant_shape(&bias->descriptor) ||
                 vx_sd_direct_vector(build, &bias->descriptor,
                                     output_feature, 2u)))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    dimensions = vx_sd_direct_output_dimensions(
        build, "out", input->descriptor.rank);
    if (!dimensions)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, input->descriptor.rank);
    for (uint32_t axis = 0u; axis < input->descriptor.rank; ++axis)
        dimensions[axis] = input->descriptor.dimensions[axis];
    dimensions[input->descriptor.rank - 1u] = vx_sd_fixed(output_feature);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32, dimensions,
                            input->descriptor.rank, NULL);
}

static int32_t vx_sd_direct_embedding(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input", "weight"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* weight;
    VxShapeDomainDimension* dimensions;
    uint32_t output_rank;
    uint64_t width;
    if (vx_sd_exact_ports(build, INPUTS, 2u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, NULL, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    weight = vx_sd_input(build->request, "weight");
    if (input->descriptor.dtype != VX_DTYPE_I32)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                          0u, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (!vx_sd_unquantized(&input->descriptor))
        return vx_sd_fail(build,
            VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u,
            VX_SHAPE_DOMAIN_INDEX_NONE);
    if (vx_sd_assert_float(build, &weight->descriptor, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!input->descriptor.rank || weight->descriptor.rank != 2u ||
        !vx_sd_direct_constant_shape(&weight->descriptor) ||
        input->descriptor.rank == UINT32_MAX)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          1u, weight->descriptor.rank);
    output_rank = input->descriptor.rank + 1u;
    dimensions = vx_sd_direct_output_dimensions(build, "out", output_rank);
    if (!dimensions)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, output_rank);
    width = (uint64_t)weight->descriptor.dimensions[1].value;
    for (uint32_t axis = 0u; axis < input->descriptor.rank; ++axis)
        dimensions[axis] = input->descriptor.dimensions[axis];
    dimensions[input->descriptor.rank] = vx_sd_fixed(width);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32, dimensions,
                            output_rank, NULL);
}

static int32_t vx_sd_prove_direct(VxShapeDomainBuild* build) {
    switch (build->request->operator_kind) {
        case VX_OP_IDENTITY:
            return vx_sd_direct_identity(build);
        case VX_OP_RELU:
        case VX_OP_LEAKY_RELU:
        case VX_OP_GELU:
        case VX_OP_SILU:
        case VX_OP_SIGMOID:
        case VX_OP_HARD_SWISH:
        case VX_OP_HARD_SIGMOID:
        case VX_OP_TANH:
        case VX_OP_SIN:
        case VX_OP_COS:
        case VX_OP_CLIP:
        case VX_OP_SOFTMAX:
        case VX_OP_LOG_SOFTMAX:
            return vx_sd_direct_activation_kind(
                build->request->operator_kind)
                ? vx_sd_direct_activation(build)
                : vx_sd_unsupported(build);
        case VX_OP_PRELU:
            return vx_sd_direct_prelu(build);
        case VX_OP_CAST:
            return vx_sd_direct_cast(build);
        case VX_OP_QUANTIZE_LINEAR:
            return vx_sd_direct_quantize(build, 0);
        case VX_OP_DEQUANTIZE_LINEAR:
            return vx_sd_direct_quantize(build, 1);
        case VX_OP_LAYER_NORM:
        case VX_OP_RMS_NORM:
            return vx_sd_direct_feature_norm(build);
        case VX_OP_GROUP_NORM:
            return vx_sd_direct_group_norm(build);
        case VX_OP_LINEAR:
        case VX_OP_GEMM:
        case VX_OP_MATMUL:
            return vx_sd_direct_dense(build);
        case VX_OP_EMBEDDING:
            return vx_sd_direct_embedding(build);
        case VX_OP_ADD:
        case VX_OP_MUL:
            return vx_sd_direct_arithmetic(build);
        default:
            return VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED;
    }
}
