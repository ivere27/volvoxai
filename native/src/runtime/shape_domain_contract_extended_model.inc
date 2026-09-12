/* Symbolic contracts for the model-oriented extended operators. This
 * fragment is included only by shape_domain_contract.c, after the structural
 * fragment. It owns exactly MoERouter, MoELinear, CrossAttention,
 * BatchNorm2D, Interpolate1D, Concat2, RequantizeLinear, and Dropout. */

static int32_t vx_sd_ext_model_boolean(VxShapeDomainBuild* build,
                                       const char* name,
                                       int default_value,
                                       int* output) {
    const VxShapeDomainParam* param = vx_sd_param(build->request, name);
    if (!output)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INTERNAL,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (!param) {
        *output = default_value;
        return VX_SHAPE_DOMAIN_STATUS_OK;
    }
    if (param->kind != VX_SHAPE_DOMAIN_PARAM_BOOLEAN ||
        (param->boolean != 0 && param->boolean != 1))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    *output = param->boolean;
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_ext_model_positive_integer(
        VxShapeDomainBuild* build,
        const char* name,
        uint64_t default_value,
        int required,
        uint64_t* output) {
    const VxShapeDomainParam* param = vx_sd_param(build->request, name);
    int64_t value;
    if (!output)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INTERNAL,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (!param) {
        if (required)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
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

static int32_t vx_sd_ext_model_optional_positive_number(
        VxShapeDomainBuild* build,
        const char* name) {
    const VxShapeDomainParam* param = vx_sd_param(build->request, name);
    if (!param) return VX_SHAPE_DOMAIN_STATUS_OK;
    if (param->kind != VX_SHAPE_DOMAIN_PARAM_NUMBER ||
        !isfinite(param->number) || param->number <= 0.0)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_ext_model_dimension_maximum(
        VxShapeDomainBuild* build,
        const VxShapeDomainDimension* dimension,
        uint32_t input_index,
        uint32_t axis,
        uint64_t* output) {
    uint64_t fixed;
    const VxShapeDomainConstraint* constraint;
    if (!output || !dimension)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INTERNAL,
                          input_index, axis);
    if (vx_shape_domain_fixed_value(dimension,
            build->request->environment,
            build->request->environment_count, &fixed)) {
        *output = fixed;
        return VX_SHAPE_DOMAIN_STATUS_OK;
    }
    if (dimension->symbol_index >= build->request->environment_count)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DIMENSION,
                          input_index, axis);
    constraint = &build->request->environment[dimension->symbol_index];
    if (dimension->kind == VX_SHAPE_DOMAIN_DIMENSION_SYMBOL) {
        *output = constraint->maximum;
        return VX_SHAPE_DOMAIN_STATUS_OK;
    }
    if (dimension->kind != VX_SHAPE_DOMAIN_DIMENSION_AFFINE ||
        vx_sd_add_signed(constraint->maximum, dimension->value, output))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                          input_index, axis);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_ext_model_per_tensor_byte(
        VxShapeDomainBuild* build,
        const VxShapeDomainTensorDescriptor* descriptor,
        uint32_t index,
        const VxShapeDomainQuantization** quantization) {
    if (!descriptor ||
        (descriptor->dtype != VX_DTYPE_I8 &&
         descriptor->dtype != VX_DTYPE_U8))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                          index, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (descriptor->quantization.scheme != 1)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION,
                          index, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (quantization) *quantization = &descriptor->quantization;
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static float vx_sd_ext_model_scale(uint32_t bits) {
    float value;
    vx_sd_copy(&value, &bits, (uint32_t)sizeof(value));
    return value;
}

static int32_t vx_sd_ext_model_moe_router(VxShapeDomainBuild* build) {
    static const char* const REQUIRED[] = {"input", "weight"};
    static const char* const OPTIONAL[] = {"bias"};
    static const char* const OUTPUTS[] = {"indices", "weights"};
    static const char* const ALLOWED[] = {
        "num_experts", "top_k", "temperature", "normalize"
    };
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* weight;
    const VxShapeDomainNamedTensor* bias;
    VxShapeDomainDimension dimensions[8];
    uint64_t experts;
    uint64_t num_experts;
    uint64_t top_k;
    int normalize;
    uint32_t rank;
    if (vx_sd_optional_ports(build, REQUIRED, 2u, OPTIONAL, 1u,
            OUTPUTS, 2u) || vx_sd_allowed_params(build, ALLOWED, 4u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    weight = vx_sd_input(build->request, "weight");
    bias = vx_sd_input(build->request, "bias");
    if (vx_sd_assert_float(build, &input->descriptor, 0u) ||
        vx_sd_assert_float(build, &weight->descriptor, 1u) ||
        (bias && vx_sd_assert_float(build, &bias->descriptor, 2u)))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    rank = input->descriptor.rank;
    if (!rank || rank > 8u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, rank);
    if (weight->descriptor.rank != 2u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          1u, weight->descriptor.rank);
    if (!vx_shape_domain_dimensions_equal(
            &input->descriptor.dimensions[rank - 1u],
            &weight->descriptor.dimensions[0],
            build->request->environment,
            build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          1u, 0u);
    if (!vx_shape_domain_fixed_value(&weight->descriptor.dimensions[1],
            build->request->environment,
            build->request->environment_count, &experts))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          1u, 1u);
    if (vx_sd_ext_model_positive_integer(build, "num_experts", experts,
            0, &num_experts) ||
        vx_sd_ext_model_positive_integer(build, "top_k", 2u,
            0, &top_k) ||
        vx_sd_ext_model_optional_positive_number(build, "temperature") ||
        vx_sd_ext_model_boolean(build, "normalize", 1, &normalize))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    (void)normalize;
    if (num_experts != experts)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (top_k > experts)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (bias && (bias->descriptor.rank != 1u ||
        !vx_shape_domain_dimensions_equal(&bias->descriptor.dimensions[0],
            &weight->descriptor.dimensions[1],
            build->request->environment,
            build->request->environment_count)))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          2u, 0u);
    for (uint32_t axis = 0u; axis + 1u < rank; ++axis)
        dimensions[axis] = input->descriptor.dimensions[axis];
    dimensions[rank - 1u] = vx_sd_fixed(top_k);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT;
    if (vx_sd_set_output(build, "indices", VX_DTYPE_F32,
            dimensions, rank, NULL))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    return vx_sd_set_output(build, "weights", VX_DTYPE_F32,
                            dimensions, rank, NULL);
}

static int32_t vx_sd_ext_model_moe_linear(VxShapeDomainBuild* build) {
    static const char* const REQUIRED[] = {
        "input", "expert_weight", "route_indices", "route_weights"
    };
    static const char* const OPTIONAL[] = {"expert_bias"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* weight;
    const VxShapeDomainNamedTensor* indices;
    const VxShapeDomainNamedTensor* routes;
    const VxShapeDomainNamedTensor* bias;
    VxShapeDomainDimension dimensions[8];
    uint64_t experts;
    uint64_t output_width;
    uint64_t top_k_maximum;
    uint32_t rank;
    if (vx_sd_optional_ports(build, REQUIRED, 4u, OPTIONAL, 1u,
            OUTPUTS, 1u) || vx_sd_allowed_params(build, NULL, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    weight = vx_sd_input(build->request, "expert_weight");
    indices = vx_sd_input(build->request, "route_indices");
    routes = vx_sd_input(build->request, "route_weights");
    bias = vx_sd_input(build->request, "expert_bias");
    if (vx_sd_assert_float(build, &input->descriptor, 0u) ||
        vx_sd_assert_float(build, &weight->descriptor, 1u) ||
        vx_sd_assert_float(build, &indices->descriptor, 2u) ||
        vx_sd_assert_float(build, &routes->descriptor, 3u) ||
        (bias && vx_sd_assert_float(build, &bias->descriptor, 4u)))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    rank = input->descriptor.rank;
    if (!rank || rank > 8u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, rank);
    if (weight->descriptor.rank != 3u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          1u, weight->descriptor.rank);
    if (!vx_shape_domain_dimensions_equal(
            &weight->descriptor.dimensions[1],
            &input->descriptor.dimensions[rank - 1u],
            build->request->environment,
            build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          1u, 1u);
    if (!vx_sd_shapes_equal(&indices->descriptor, &routes->descriptor,
            build->request->environment,
            build->request->environment_count) ||
        indices->descriptor.rank != rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          2u, 3u);
    for (uint32_t axis = 0u; axis + 1u < rank; ++axis)
        if (!vx_shape_domain_dimensions_equal(
                &indices->descriptor.dimensions[axis],
                &input->descriptor.dimensions[axis],
                build->request->environment,
                build->request->environment_count))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                              2u, axis);
    if (!vx_shape_domain_fixed_value(&weight->descriptor.dimensions[0],
            build->request->environment,
            build->request->environment_count, &experts) ||
        !vx_shape_domain_fixed_value(&weight->descriptor.dimensions[2],
            build->request->environment,
            build->request->environment_count, &output_width))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          1u, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (vx_sd_ext_model_dimension_maximum(build,
            &indices->descriptor.dimensions[rank - 1u], 2u, rank - 1u,
            &top_k_maximum))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (top_k_maximum > experts)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          2u, rank - 1u);
    if (bias && (bias->descriptor.rank != 2u ||
        !vx_shape_domain_dimensions_equal(&bias->descriptor.dimensions[0],
            &weight->descriptor.dimensions[0],
            build->request->environment,
            build->request->environment_count) ||
        !vx_shape_domain_dimensions_equal(&bias->descriptor.dimensions[1],
            &weight->descriptor.dimensions[2],
            build->request->environment,
            build->request->environment_count)))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          4u, VX_SHAPE_DOMAIN_INDEX_NONE);
    for (uint32_t axis = 0u; axis + 1u < rank; ++axis)
        dimensions[axis] = input->descriptor.dimensions[axis];
    dimensions[rank - 1u] = vx_sd_fixed(output_width);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            dimensions, rank, NULL);
}

static int32_t vx_sd_ext_model_cross_attention(VxShapeDomainBuild* build) {
    static const char* const REQUIRED[] = {"q", "kv", "weight"};
    static const char* const OPTIONAL[] = {"scale", "bias"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const ALLOWED[] = {"heads"};
    const VxShapeDomainNamedTensor* query;
    const VxShapeDomainNamedTensor* kv;
    const VxShapeDomainNamedTensor* weight;
    const VxShapeDomainNamedTensor* scale;
    const VxShapeDomainNamedTensor* bias;
    uint64_t feature;
    uint64_t projection;
    uint64_t heads;
    VxShapeDomainDimension projection_dimension;
    uint32_t rank;
    uint32_t feature_axis;
    if (vx_sd_optional_ports(build, REQUIRED, 3u, OPTIONAL, 2u,
            OUTPUTS, 1u) || vx_sd_allowed_params(build, ALLOWED, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    query = vx_sd_input(build->request, "q");
    kv = vx_sd_input(build->request, "kv");
    weight = vx_sd_input(build->request, "weight");
    scale = vx_sd_input(build->request, "scale");
    bias = vx_sd_input(build->request, "bias");
    if (vx_sd_assert_float(build, &query->descriptor, 0u) ||
        vx_sd_assert_float(build, &kv->descriptor, 1u) ||
        vx_sd_assert_float(build, &weight->descriptor, 2u) ||
        (scale && vx_sd_assert_float(build, &scale->descriptor, 3u)) ||
        (bias && vx_sd_assert_float(build, &bias->descriptor, 4u)))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    rank = query->descriptor.rank;
    if (rank < 2u || rank > 3u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, rank);
    if (kv->descriptor.rank != rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          1u, kv->descriptor.rank);
    feature_axis = rank - 1u;
    if (!vx_shape_domain_dimensions_equal(
            &kv->descriptor.dimensions[feature_axis],
            &query->descriptor.dimensions[feature_axis],
            build->request->environment,
            build->request->environment_count) ||
        (rank == 3u && !vx_shape_domain_dimensions_equal(
            &kv->descriptor.dimensions[0], &query->descriptor.dimensions[0],
            build->request->environment,
            build->request->environment_count)))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          1u, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (!vx_shape_domain_fixed_value(
            &query->descriptor.dimensions[feature_axis],
            build->request->environment,
            build->request->environment_count, &feature))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          0u, feature_axis);
    if (vx_sd_checked_multiply(feature, 3u, &projection))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                          2u, 0u);
    projection_dimension = vx_sd_fixed(projection);
    if (weight->descriptor.rank != 2u ||
        !vx_shape_domain_dimensions_equal(&weight->descriptor.dimensions[0],
            &projection_dimension, build->request->environment,
            build->request->environment_count) ||
        !vx_shape_domain_dimensions_equal(&weight->descriptor.dimensions[1],
            &query->descriptor.dimensions[feature_axis],
            build->request->environment,
            build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          2u, VX_SHAPE_DOMAIN_INDEX_NONE);
    {
        const VxShapeDomainNamedTensor* optional_values[2] = {scale, bias};
        for (uint32_t index = 0u; index < 2u; ++index) {
            const VxShapeDomainNamedTensor* optional = optional_values[index];
            if (optional && (optional->descriptor.rank != 1u ||
                !vx_shape_domain_dimensions_equal(
                    &optional->descriptor.dimensions[0],
                    &projection_dimension,
                    build->request->environment,
                    build->request->environment_count)))
                return vx_sd_fail(build,
                    VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH, 3u + index, 0u);
        }
    }
    if (vx_sd_ext_model_positive_integer(build, "heads", 8u,
            0, &heads))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (feature % heads)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            query->descriptor.dimensions, rank, NULL);
}

static int32_t vx_sd_ext_model_batch_norm2d(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {
        "input", "weight", "bias", "running_mean", "running_var"
    };
    static const char* const OUTPUTS[] = {"out"};
    static const char* const ALLOWED[] = {"eps"};
    const VxShapeDomainNamedTensor* input;
    if (vx_sd_exact_ports(build, INPUTS, 5u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, ALLOWED, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (vx_sd_assert_float(build, &input->descriptor, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (input->descriptor.rank != 4u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, input->descriptor.rank);
    for (uint32_t index = 1u; index < 5u; ++index) {
        const VxShapeDomainNamedTensor* parameter =
            vx_sd_input(build->request, INPUTS[index]);
        if (vx_sd_assert_float(build, &parameter->descriptor, index))
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
        if (parameter->descriptor.rank != 1u ||
            !vx_shape_domain_dimensions_equal(
                &parameter->descriptor.dimensions[0],
                &input->descriptor.dimensions[3],
                build->request->environment,
                build->request->environment_count))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                              index, 0u);
    }
    if (vx_sd_ext_model_optional_positive_number(build, "eps"))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            input->descriptor.dimensions, 4u, NULL);
}

static int32_t vx_sd_ext_model_interpolate1d(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const ALLOWED[] = {"size"};
    const VxShapeDomainNamedTensor* input;
    VxShapeDomainDimension dimensions[3];
    uint64_t size;
    if (vx_sd_exact_ports(build, INPUTS, 1u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, ALLOWED, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (vx_sd_assert_float(build, &input->descriptor, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (input->descriptor.rank != 3u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, input->descriptor.rank);
    if (vx_sd_ext_model_positive_integer(build, "size", 0u, 1, &size))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    dimensions[0] = input->descriptor.dimensions[0];
    dimensions[1] = input->descriptor.dimensions[1];
    dimensions[2] = vx_sd_fixed(size);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            dimensions, 3u, NULL);
}

static int32_t vx_sd_ext_model_concat2(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"a", "b"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const ALLOWED[] = {"axis", "sigmoid"};
    static const uint8_t INPUT0_NAME[] = {
        (uint8_t)'i', (uint8_t)'n', (uint8_t)'p', (uint8_t)'u',
        (uint8_t)'t', (uint8_t)'0'
    };
    static const uint8_t INPUT1_NAME[] = {
        (uint8_t)'i', (uint8_t)'n', (uint8_t)'p', (uint8_t)'u',
        (uint8_t)'t', (uint8_t)'1'
    };
    VxShapeDomainNamedTensor inputs[2];
    VxShapeDomainParam params[1];
    VxShapeDomainProofRequest request;
    VxShapeDomainBuild adapted;
    const VxShapeDomainParam* axis;
    const VxShapeDomainNamedTensor* source_a;
    const VxShapeDomainNamedTensor* source_b;
    int sigmoid;
    if (vx_sd_exact_ports(build, INPUTS, 2u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, ALLOWED, 2u) ||
        vx_sd_ext_model_boolean(build, "sigmoid", 0, &sigmoid))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    source_a = vx_sd_input(build->request, "a");
    source_b = vx_sd_input(build->request, "b");
    inputs[0].name.bytes = source_a->name.bytes;
    inputs[0].name.length = source_a->name.length;
    inputs[0].tensor_index = source_a->tensor_index;
    vx_sd_descriptor_copy(&inputs[0].descriptor, &source_a->descriptor);
    inputs[1].name.bytes = source_b->name.bytes;
    inputs[1].name.length = source_b->name.length;
    inputs[1].tensor_index = source_b->tensor_index;
    vx_sd_descriptor_copy(&inputs[1].descriptor, &source_b->descriptor);
    if (sigmoid &&
        (vx_sd_assert_float(build, &inputs[0].descriptor, 0u) ||
         vx_sd_assert_float(build, &inputs[1].descriptor, 1u)))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    inputs[0].name.bytes = INPUT0_NAME;
    inputs[0].name.length = (uint32_t)sizeof(INPUT0_NAME);
    inputs[1].name.bytes = INPUT1_NAME;
    inputs[1].name.length = (uint32_t)sizeof(INPUT1_NAME);
    request.operator_kind = build->request->operator_kind;
    request.operator_name.bytes = build->request->operator_name.bytes;
    request.operator_name.length = build->request->operator_name.length;
    request.environment = build->request->environment;
    request.environment_count = build->request->environment_count;
    request.inputs = build->request->inputs;
    request.input_count = build->request->input_count;
    request.declared_outputs = build->request->declared_outputs;
    request.declared_output_count = build->request->declared_output_count;
    request.use_declared_outputs = build->request->use_declared_outputs;
    request.params = build->request->params;
    request.param_count = build->request->param_count;
    request.inputs = inputs;
    request.input_count = 2u;
    /* proveConcat2 intentionally does not forward declaredOutputs to its
     * internal Concat proof. Preserve that observable v1 restriction: a
     * dynamic Concat2 axis remains unprovable even when the graph declares an
     * affine-looking output symbol. */
    request.use_declared_outputs = 0;
    request.params = NULL;
    request.param_count = 0u;
    axis = vx_sd_param(build->request, "axis");
    if (axis) {
        params[0].name.bytes = axis->name.bytes;
        params[0].name.length = axis->name.length;
        params[0].kind = axis->kind;
        params[0].reserved = axis->reserved;
        params[0].number = axis->number;
        params[0].boolean = axis->boolean;
        params[0].string.bytes = axis->string.bytes;
        params[0].string.length = axis->string.length;
        params[0].values = axis->values;
        params[0].value_count = axis->value_count;
        request.params = params;
        request.param_count = 1u;
    }
    adapted.request = build->request;
    adapted.result = build->result;
    adapted.error = build->error;
    adapted.arena = build->arena;
    adapted.affine_capacity = build->affine_capacity;
    adapted.request = &request;
    return vx_sd_struct_concat(&adapted);
}

static int32_t vx_sd_ext_model_requantize(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* output;
    const VxShapeDomainQuantization* input_quantization;
    const VxShapeDomainQuantization* output_quantization;
    float input_scale;
    float output_scale;
    float ratio;
    if (vx_sd_exact_ports(build, INPUTS, 1u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, NULL, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    output = vx_sd_named(build->request->declared_outputs,
                         build->request->declared_output_count, "out");
    if (vx_sd_ext_model_per_tensor_byte(build, &input->descriptor, 0u,
            &input_quantization) ||
        vx_sd_ext_model_per_tensor_byte(build, &output->descriptor, 0u,
            &output_quantization))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!vx_sd_shapes_equal(&input->descriptor, &output->descriptor,
            build->request->environment,
            build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, VX_SHAPE_DOMAIN_INDEX_NONE);
    input_scale = vx_sd_ext_model_scale(input_quantization->scale_bits);
    output_scale = vx_sd_ext_model_scale(output_quantization->scale_bits);
    ratio = (float)((double)input_scale / (double)output_scale);
    if (!isfinite(ratio) || ratio <= 0.0f)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION,
                          0u, VX_SHAPE_DOMAIN_INDEX_NONE);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_set_output(build, "out", output->descriptor.dtype,
                            input->descriptor.dimensions,
                            input->descriptor.rank, output_quantization);
}

static int32_t vx_sd_ext_model_dropout(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const ALLOWED[] = {
        "ratio", "p", "probability", "seed"
    };
    static const char* const PROBABILITY_NAMES[] = {
        "ratio", "p", "probability"
    };
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainParam* probability = NULL;
    const VxShapeDomainParam* seed;
    int64_t seed_value;
    if (vx_sd_exact_ports(build, INPUTS, 1u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, ALLOWED, 4u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (vx_sd_assert_float(build, &input->descriptor, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (input->descriptor.rank > 8u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, input->descriptor.rank);
    for (uint32_t index = 0u; index < 3u; ++index) {
        const VxShapeDomainParam* current =
            vx_sd_param(build->request, PROBABILITY_NAMES[index]);
        if (current && probability)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        if (current) probability = current;
    }
    if (probability &&
        (probability->kind != VX_SHAPE_DOMAIN_PARAM_NUMBER ||
         !isfinite(probability->number) || probability->number < 0.0 ||
         probability->number >= 1.0))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    seed = vx_sd_param(build->request, "seed");
    if (seed && (vx_sd_number_integer(build, seed, 0, &seed_value) ||
                 seed_value < 0 || (uint64_t)seed_value > UINT32_MAX))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            input->descriptor.dimensions,
                            input->descriptor.rank, NULL);
}

static int32_t vx_sd_prove_extended_model(VxShapeDomainBuild* build) {
    switch (build->request->operator_kind) {
        case VX_OP_MOE_ROUTER:
            return vx_sd_ext_model_moe_router(build);
        case VX_OP_MOE_LINEAR:
            return vx_sd_ext_model_moe_linear(build);
        case VX_OP_CROSS_ATTENTION:
            return vx_sd_ext_model_cross_attention(build);
        case VX_OP_BATCH_NORM_2D:
            return vx_sd_ext_model_batch_norm2d(build);
        case VX_OP_INTERPOLATE_1D:
            return vx_sd_ext_model_interpolate1d(build);
        case VX_OP_CONCAT2:
            return vx_sd_ext_model_concat2(build);
        case VX_OP_REQUANTIZE_LINEAR:
            return vx_sd_ext_model_requantize(build);
        case VX_OP_DROPOUT:
            return vx_sd_ext_model_dropout(build);
        default:
            return VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED;
    }
}
