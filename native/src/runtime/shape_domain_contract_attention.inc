/* Symbolic contracts for SDPA, CrossSDPA, and RoPE. This fragment is
 * included only by shape_domain_contract.c. Attention payloads and position
 * values never enter this bounded-domain proof. */

static int32_t vx_sd_attention_i32(
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

static int32_t vx_sd_attention_positive_f32_param(
        VxShapeDomainBuild* build,
        const VxShapeDomainParam* param) {
    float canonical;
    if (!param || param->kind != VX_SHAPE_DOMAIN_PARAM_NUMBER ||
        !isfinite(param->number) || param->number <= 0.0)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    canonical = (float)param->number;
    if (!isfinite(canonical) || canonical <= 0.0f)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_attention_parameters(VxShapeDomainBuild* build,
                                          uint64_t* heads) {
    static const char* const ALLOWED[] = {
        "heads", "causal", "scale", "dropout", "attention_dropout",
        "dropout_seed", "training_seed", "seed"
    };
    static const char* const PROBABILITIES[] = {
        "dropout", "attention_dropout"
    };
    static const char* const SEEDS[] = {
        "dropout_seed", "training_seed", "seed"
    };
    const VxShapeDomainParam* heads_param;
    const VxShapeDomainParam* causal;
    const VxShapeDomainParam* scale;
    const VxShapeDomainParam* selected = NULL;
    int64_t integer;
    if (!heads || vx_sd_allowed_params(build, ALLOWED, 8u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    heads_param = vx_sd_param(build->request, "heads");
    if (!heads_param ||
        vx_sd_number_integer(build, heads_param, 0, &integer) || integer <= 0)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    *heads = (uint64_t)integer;
    causal = vx_sd_param(build->request, "causal");
    if (!causal || causal->kind != VX_SHAPE_DOMAIN_PARAM_BOOLEAN ||
        (causal->boolean != 0 && causal->boolean != 1))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    scale = vx_sd_param(build->request, "scale");
    if (scale && vx_sd_attention_positive_f32_param(build, scale))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    for (uint32_t index = 0u; index < 2u; ++index) {
        const VxShapeDomainParam* current =
            vx_sd_param(build->request, PROBABILITIES[index]);
        if (current && selected)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        if (current) selected = current;
    }
    if (selected &&
        (selected->kind != VX_SHAPE_DOMAIN_PARAM_NUMBER ||
         !isfinite(selected->number) || selected->number < 0.0 ||
         selected->number >= 1.0))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    selected = NULL;
    for (uint32_t index = 0u; index < 3u; ++index) {
        const VxShapeDomainParam* current =
            vx_sd_param(build->request, SEEDS[index]);
        if (current && selected)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        if (current) selected = current;
    }
    if (selected &&
        (vx_sd_number_integer(build, selected, 0, &integer) || integer < 0 ||
         (uint64_t)integer > UINT32_MAX))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_attention_mask(
        VxShapeDomainBuild* build,
        const VxShapeDomainTensorDescriptor* mask,
        const VxShapeDomainDimension* batch,
        const VxShapeDomainDimension* queries,
        const VxShapeDomainDimension* keys,
        uint32_t input_index) {
    int valid = 0;
    if (!mask) return VX_SHAPE_DOMAIN_STATUS_OK;
    if (vx_sd_attention_i32(build, mask, input_index))
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

static int32_t vx_sd_attention_sdpa(VxShapeDomainBuild* build) {
    static const char* const REQUIRED[] = {"qkv"};
    static const char* const OPTIONAL[] = {"mask"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* qkv;
    const VxShapeDomainNamedTensor* mask;
    VxShapeDomainDimension dimensions[3];
    VxShapeDomainDimension one = vx_sd_fixed(1u);
    const VxShapeDomainDimension* batch;
    const VxShapeDomainDimension* sequence;
    uint32_t rank;
    uint32_t feature_axis;
    uint64_t packed_feature;
    uint64_t feature;
    uint64_t heads;
    if (vx_sd_optional_ports(build, REQUIRED, 1u, OPTIONAL, 1u,
            OUTPUTS, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    qkv = vx_sd_input(build->request, "qkv");
    mask = vx_sd_input(build->request, "mask");
    if (vx_sd_assert_float(build, &qkv->descriptor, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    rank = qkv->descriptor.rank;
    if (rank != 2u && rank != 3u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, rank);
    feature_axis = rank - 1u;
    if (!vx_shape_domain_fixed_value(
            &qkv->descriptor.dimensions[feature_axis],
            build->request->environment,
            build->request->environment_count, &packed_feature))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          0u, feature_axis);
    if (packed_feature % 3u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, feature_axis);
    feature = packed_feature / 3u;
    if (vx_sd_attention_parameters(build, &heads))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (feature % heads)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, feature_axis);
    sequence = &qkv->descriptor.dimensions[rank - 2u];
    batch = rank == 2u ? &one : &qkv->descriptor.dimensions[0];
    if (vx_sd_attention_mask(build,
            mask ? &mask->descriptor : NULL,
            batch, sequence, sequence, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    for (uint32_t axis = 0u; axis < rank; ++axis)
        dimensions[axis] = qkv->descriptor.dimensions[axis];
    dimensions[feature_axis] = vx_sd_fixed(feature);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            dimensions, rank, NULL);
}

static int32_t vx_sd_attention_cross_sdpa(VxShapeDomainBuild* build) {
    static const char* const REQUIRED[] = {"q", "k", "v"};
    static const char* const OPTIONAL[] = {"mask"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* query;
    const VxShapeDomainNamedTensor* key;
    const VxShapeDomainNamedTensor* value;
    const VxShapeDomainNamedTensor* mask;
    VxShapeDomainDimension one = vx_sd_fixed(1u);
    const VxShapeDomainDimension* batch;
    const VxShapeDomainDimension* queries;
    const VxShapeDomainDimension* keys;
    uint32_t rank;
    uint32_t sequence_axis;
    uint32_t feature_axis;
    uint64_t feature;
    uint64_t heads;
    if (vx_sd_optional_ports(build, REQUIRED, 3u, OPTIONAL, 1u,
            OUTPUTS, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    query = vx_sd_input(build->request, "q");
    key = vx_sd_input(build->request, "k");
    value = vx_sd_input(build->request, "v");
    mask = vx_sd_input(build->request, "mask");
    if (vx_sd_assert_float(build, &query->descriptor, 0u) ||
        vx_sd_assert_float(build, &key->descriptor, 1u) ||
        vx_sd_assert_float(build, &value->descriptor, 2u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    rank = query->descriptor.rank;
    if ((rank != 2u && rank != 3u) ||
        (key->descriptor.rank != 2u && key->descriptor.rank != 3u) ||
        (value->descriptor.rank != 2u && value->descriptor.rank != 3u))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, rank);
    if (key->descriptor.rank != rank || value->descriptor.rank != rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          key->descriptor.rank, value->descriptor.rank);
    sequence_axis = rank - 2u;
    feature_axis = rank - 1u;
    batch = rank == 2u ? &one : &query->descriptor.dimensions[0];
    queries = &query->descriptor.dimensions[sequence_axis];
    keys = &key->descriptor.dimensions[sequence_axis];
    if ((rank == 3u &&
         (!vx_shape_domain_dimensions_equal(
              &key->descriptor.dimensions[0], batch,
              build->request->environment,
              build->request->environment_count) ||
          !vx_shape_domain_dimensions_equal(
              &value->descriptor.dimensions[0], batch,
              build->request->environment,
              build->request->environment_count))) ||
        !vx_shape_domain_dimensions_equal(
            &value->descriptor.dimensions[sequence_axis], keys,
            build->request->environment,
            build->request->environment_count) ||
        !vx_shape_domain_dimensions_equal(
            &key->descriptor.dimensions[feature_axis],
            &query->descriptor.dimensions[feature_axis],
            build->request->environment,
            build->request->environment_count) ||
        !vx_shape_domain_dimensions_equal(
            &value->descriptor.dimensions[feature_axis],
            &query->descriptor.dimensions[feature_axis],
            build->request->environment,
            build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, feature_axis);
    if (!vx_shape_domain_fixed_value(
            &query->descriptor.dimensions[feature_axis],
            build->request->environment,
            build->request->environment_count, &feature))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          0u, feature_axis);
    if (vx_sd_attention_parameters(build, &heads))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (feature % heads)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, feature_axis);
    if (vx_sd_attention_mask(build,
            mask ? &mask->descriptor : NULL,
            batch, queries, keys, 3u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            query->descriptor.dimensions, rank, NULL);
}

static int32_t vx_sd_attention_dimension_bound(
        VxShapeDomainBuild* build,
        const VxShapeDomainDimension* dimension,
        int maximum,
        uint64_t* output) {
    const VxShapeDomainConstraint* constraint;
    if (!dimension || !output)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INTERNAL,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (dimension->kind == VX_SHAPE_DOMAIN_DIMENSION_FIXED) {
        *output = (uint64_t)dimension->value;
        return VX_SHAPE_DOMAIN_STATUS_OK;
    }
    if (dimension->kind != VX_SHAPE_DOMAIN_DIMENSION_SYMBOL ||
        dimension->symbol_index >= build->request->environment_count)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    constraint = &build->request->environment[dimension->symbol_index];
    *output = maximum ? constraint->maximum : constraint->minimum;
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_attention_position_ids(
        VxShapeDomainBuild* build,
        const VxShapeDomainTensorDescriptor* positions,
        uint32_t rank,
        const VxShapeDomainDimension* batch,
        const VxShapeDomainDimension* sequence,
        uint32_t input_index) {
    int valid = 0;
    if (!positions) return VX_SHAPE_DOMAIN_STATUS_OK;
    if (vx_sd_attention_i32(build, positions, input_index))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (positions->rank == 1u) {
        valid = vx_shape_domain_dimensions_equal(
            &positions->dimensions[0], sequence,
            build->request->environment,
            build->request->environment_count);
    } else if (rank == 3u && positions->rank == 2u) {
        valid = vx_shape_domain_dimensions_equal(
            &positions->dimensions[0], batch,
            build->request->environment,
            build->request->environment_count) &&
            vx_shape_domain_dimensions_equal(
                &positions->dimensions[1], sequence,
                build->request->environment,
                build->request->environment_count);
    }
    if (!valid)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          input_index, VX_SHAPE_DOMAIN_INDEX_NONE);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_attention_rope(VxShapeDomainBuild* build) {
    static const char* const REQUIRED[] = {"input"};
    static const char* const OPTIONAL[] = {"position_ids"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const ALLOWED[] = {
        "rotary_dim", "theta", "position_offset", "interleaved"
    };
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* positions;
    const VxShapeDomainParam* rotary;
    const VxShapeDomainParam* theta;
    const VxShapeDomainParam* offset;
    const VxShapeDomainParam* interleaved;
    VxShapeDomainDimension one = vx_sd_fixed(1u);
    const VxShapeDomainDimension* batch;
    const VxShapeDomainDimension* sequence;
    const VxShapeDomainDimension* width;
    uint32_t rank;
    uint64_t fixed_width;
    uint64_t minimum_width;
    uint64_t maximum_sequence;
    uint64_t rotary_dimension = 0u;
    uint64_t position_offset = 0u;
    int64_t integer;
    if (vx_sd_optional_ports(build, REQUIRED, 1u, OPTIONAL, 1u,
            OUTPUTS, 1u) || vx_sd_allowed_params(build, ALLOWED, 4u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    positions = vx_sd_input(build->request, "position_ids");
    if (vx_sd_assert_float(build, &input->descriptor, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    rank = input->descriptor.rank;
    if (rank != 2u && rank != 3u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, rank);
    sequence = &input->descriptor.dimensions[rank - 2u];
    width = &input->descriptor.dimensions[rank - 1u];
    batch = rank == 2u ? &one : &input->descriptor.dimensions[0];
    rotary = vx_sd_param(build->request, "rotary_dim");
    if (rotary) {
        if (vx_sd_number_integer(build, rotary, 0, &integer) ||
            integer <= 0 || ((uint64_t)integer % 2u))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        rotary_dimension = (uint64_t)integer;
    }
    theta = vx_sd_param(build->request, "theta");
    if (theta && vx_sd_attention_positive_f32_param(build, theta))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    offset = vx_sd_param(build->request, "position_offset");
    if (offset) {
        if (vx_sd_number_integer(build, offset, 0, &integer) || integer < 0 ||
            (uint64_t)integer > (uint64_t)INT32_MAX)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        position_offset = (uint64_t)integer;
    }
    interleaved = vx_sd_param(build->request, "interleaved");
    if (interleaved &&
        (interleaved->kind != VX_SHAPE_DOMAIN_PARAM_BOOLEAN ||
         (interleaved->boolean != 0 && interleaved->boolean != 1)))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (vx_shape_domain_fixed_value(width,
            build->request->environment,
            build->request->environment_count, &fixed_width)) {
        if (!rotary_dimension) {
            if (fixed_width % 2u)
                return vx_sd_fail(build,
                    VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                    VX_SHAPE_DOMAIN_INDEX_NONE,
                    VX_SHAPE_DOMAIN_INDEX_NONE);
        } else if (rotary_dimension > fixed_width) {
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        }
    } else if (!rotary_dimension) {
        const VxShapeDomainConstraint* constraint;
        if (width->kind != VX_SHAPE_DOMAIN_DIMENSION_SYMBOL ||
            width->symbol_index >= build->request->environment_count)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                              0u, rank - 1u);
        constraint = &build->request->environment[width->symbol_index];
        if (constraint->multiple_of % 2u)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                              0u, rank - 1u);
    } else {
        if (vx_sd_attention_dimension_bound(
                build, width, 0, &minimum_width))
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
        if (rotary_dimension > minimum_width)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
    }
    if (vx_sd_attention_dimension_bound(
            build, sequence, 1, &maximum_sequence))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (maximum_sequence - 1u >
        (uint64_t)INT32_MAX - position_offset)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (vx_sd_attention_position_ids(build,
            positions ? &positions->descriptor : NULL,
            rank, batch, sequence, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            input->descriptor.dimensions, rank, NULL);
}

static int32_t vx_sd_prove_attention(VxShapeDomainBuild* build) {
    switch (build->request->operator_kind) {
        case VX_OP_SDPA:
            return vx_sd_attention_sdpa(build);
        case VX_OP_CROSS_SDPA:
            return vx_sd_attention_cross_sdpa(build);
        case VX_OP_ROPE:
            return vx_sd_attention_rope(build);
        default:
            return VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED;
    }
}
