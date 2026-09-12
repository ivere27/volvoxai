/* Symbolic contracts for logical selection, selective scan, and vision
 * profile operators. This fragment is included only by
 * shape_domain_contract.c. It owns exactly Not, Mask, SSMScan,
 * SelectiveScan, SpatialSoftargmaxY, MeanHeight, ProfileX, and ProfileY. */

static int32_t vx_sd_ext_sequence_i32(
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

static int32_t vx_sd_ext_sequence_boolean(
        VxShapeDomainBuild* build,
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

static void vx_sd_ext_sequence_copy_dimension(
        VxShapeDomainDimension* output,
        const VxShapeDomainDimension* input) {
    output->kind = input->kind;
    output->symbol_index = input->symbol_index;
    output->value = input->value;
}

static void vx_sd_ext_sequence_fixed_dimension(
        VxShapeDomainDimension* output,
        uint64_t value) {
    output->kind = VX_SHAPE_DOMAIN_DIMENSION_FIXED;
    output->symbol_index = VX_SHAPE_DOMAIN_INDEX_NONE;
    output->value = (int64_t)value;
}

static int32_t vx_sd_ext_sequence_not(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* input;
    if (vx_sd_exact_ports(build, INPUTS, 1u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, NULL, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (vx_sd_ext_sequence_i32(build, &input->descriptor, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (input->descriptor.rank > 8u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, input->descriptor.rank);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE;
    return vx_sd_set_output(build, "out", VX_DTYPE_I32,
                            input->descriptor.dimensions,
                            input->descriptor.rank, NULL);
}

static int32_t vx_sd_ext_sequence_mask(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"mask", "a", "b"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* mask;
    const VxShapeDomainNamedTensor* left;
    const VxShapeDomainNamedTensor* right;
    if (vx_sd_exact_ports(build, INPUTS, 3u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, NULL, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    mask = vx_sd_input(build->request, "mask");
    left = vx_sd_input(build->request, "a");
    right = vx_sd_input(build->request, "b");
    if (!vx_sd_unquantized(&mask->descriptor))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION,
                          0u, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (!vx_sd_unquantized(&left->descriptor))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION,
                          1u, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (!vx_sd_unquantized(&right->descriptor))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION,
                          2u, VX_SHAPE_DOMAIN_INDEX_NONE);
    if ((mask->descriptor.dtype != VX_DTYPE_F32 &&
         mask->descriptor.dtype != VX_DTYPE_I32) ||
        left->descriptor.dtype != right->descriptor.dtype ||
        (left->descriptor.dtype != VX_DTYPE_F32 &&
         left->descriptor.dtype != VX_DTYPE_I32))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                          0u, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (left->descriptor.rank > 8u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          1u, left->descriptor.rank);
    if (!vx_sd_shapes_equal(&mask->descriptor, &left->descriptor,
            build->request->environment,
            build->request->environment_count) ||
        !vx_sd_shapes_equal(&left->descriptor, &right->descriptor,
            build->request->environment,
            build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_EXACT_BINARY;
    return vx_sd_set_output(build, "out", left->descriptor.dtype,
                            left->descriptor.dimensions,
                            left->descriptor.rank, NULL);
}

static int vx_sd_ext_sequence_scan_bc_mode(
        const VxShapeDomainTensorDescriptor* descriptor,
        const VxShapeDomainTensorDescriptor* input,
        const VxShapeDomainDimension* state_width,
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count) {
    VxShapeDomainDimension expected[3];
    VxShapeDomainTensorDescriptor candidate;
    uint32_t rank;
    if (!descriptor || !input || !state_width) return -1;
    rank = input->rank;
    candidate.dtype = descriptor->dtype;
    candidate.dimensions = expected;
    candidate.quantization.scheme = 0;
    candidate.quantization.scale_bits = 0u;
    candidate.quantization.zero_point = 0;
    candidate.quantization.axis = 0u;
    candidate.quantization.count = 0u;
    candidate.quantization.scales = NULL;
    candidate.quantization.zero_points = NULL;
    vx_sd_ext_sequence_copy_dimension(&expected[0], state_width);
    candidate.rank = 1u;
    if (vx_sd_shapes_equal(descriptor, &candidate,
            environment, environment_count)) return 0;
    vx_sd_ext_sequence_copy_dimension(
        &expected[0], &input->dimensions[rank - 2u]);
    vx_sd_ext_sequence_copy_dimension(&expected[1], state_width);
    candidate.rank = 2u;
    if (vx_sd_shapes_equal(descriptor, &candidate,
            environment, environment_count)) return 1;
    if (rank != 3u) return -1;
    vx_sd_ext_sequence_copy_dimension(&expected[0], &input->dimensions[0]);
    vx_sd_ext_sequence_copy_dimension(&expected[1], &input->dimensions[1]);
    vx_sd_ext_sequence_copy_dimension(&expected[2], state_width);
    candidate.rank = 3u;
    return vx_sd_shapes_equal(descriptor, &candidate,
            environment, environment_count) ? 2 : -1;
}

static int32_t vx_sd_ext_sequence_scan(VxShapeDomainBuild* build) {
    static const char* const REQUIRED[] = {
        "input", "delta", "A", "B", "C"
    };
    static const char* const OPTIONAL[] = {"D", "z", "initial_state"};
    static const char* const OUTPUTS[] = {"out", "state"};
    static const char* const ALLOWED[] = {"delta_softplus"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* delta;
    const VxShapeDomainNamedTensor* a;
    const VxShapeDomainNamedTensor* b;
    const VxShapeDomainNamedTensor* c;
    const VxShapeDomainNamedTensor* d;
    const VxShapeDomainNamedTensor* z;
    const VxShapeDomainNamedTensor* initial_state;
    VxShapeDomainDimension state_dimensions[3];
    VxShapeDomainTensorDescriptor state_descriptor;
    int delta_softplus;
    int b_mode;
    int c_mode;
    uint32_t rank;
    if (vx_sd_optional_ports(build, REQUIRED, 5u, OPTIONAL, 3u,
            OUTPUTS, 2u) || vx_sd_allowed_params(build, ALLOWED, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    delta = vx_sd_input(build->request, "delta");
    a = vx_sd_input(build->request, "A");
    b = vx_sd_input(build->request, "B");
    c = vx_sd_input(build->request, "C");
    d = vx_sd_input(build->request, "D");
    z = vx_sd_input(build->request, "z");
    initial_state = vx_sd_input(build->request, "initial_state");
    for (uint32_t index = 0u; index < build->request->input_count; ++index)
        if (vx_sd_assert_float(build,
                &build->request->inputs[index].descriptor, index))
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
    rank = input->descriptor.rank;
    if (rank < 2u || rank > 3u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, rank);
    if (!vx_sd_shapes_equal(&delta->descriptor, &input->descriptor,
            build->request->environment,
            build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          1u, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (a->descriptor.rank != 2u ||
        !vx_shape_domain_dimensions_equal(&a->descriptor.dimensions[0],
            &input->descriptor.dimensions[rank - 1u],
            build->request->environment,
            build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          2u, 0u);
    b_mode = vx_sd_ext_sequence_scan_bc_mode(
        &b->descriptor, &input->descriptor, &a->descriptor.dimensions[1],
        build->request->environment, build->request->environment_count);
    c_mode = vx_sd_ext_sequence_scan_bc_mode(
        &c->descriptor, &input->descriptor, &a->descriptor.dimensions[1],
        build->request->environment, build->request->environment_count);
    if (b_mode < 0 || c_mode < 0)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          3u, 4u);
    if (d && (d->descriptor.rank != 1u ||
        !vx_shape_domain_dimensions_equal(&d->descriptor.dimensions[0],
            &input->descriptor.dimensions[rank - 1u],
            build->request->environment,
            build->request->environment_count)))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          5u, 0u);
    if (z && !vx_sd_shapes_equal(&z->descriptor, &input->descriptor,
            build->request->environment,
            build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          6u, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (rank == 3u)
        vx_sd_ext_sequence_copy_dimension(
            &state_dimensions[0], &input->descriptor.dimensions[0]);
    else
        vx_sd_ext_sequence_fixed_dimension(&state_dimensions[0], 1u);
    vx_sd_ext_sequence_copy_dimension(
        &state_dimensions[1], &input->descriptor.dimensions[rank - 1u]);
    vx_sd_ext_sequence_copy_dimension(
        &state_dimensions[2], &a->descriptor.dimensions[1]);
    state_descriptor.dtype = VX_DTYPE_F32;
    state_descriptor.rank = 3u;
    state_descriptor.dimensions = state_dimensions;
    state_descriptor.quantization.scheme = 0;
    state_descriptor.quantization.scale_bits = 0u;
    state_descriptor.quantization.zero_point = 0;
    state_descriptor.quantization.axis = 0u;
    state_descriptor.quantization.count = 0u;
    state_descriptor.quantization.scales = NULL;
    state_descriptor.quantization.zero_points = NULL;
    if (initial_state && !vx_sd_shapes_equal(
            &initial_state->descriptor, &state_descriptor,
            build->request->environment,
            build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          7u, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (vx_sd_ext_sequence_boolean(build, "delta_softplus", 1,
            &delta_softplus))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    (void)delta_softplus;
    (void)b_mode;
    (void)c_mode;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT;
    if (vx_sd_set_output(build, "out", VX_DTYPE_F32,
            input->descriptor.dimensions, rank, NULL))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    return vx_sd_set_output(build, "state", VX_DTYPE_F32,
                            state_dimensions, 3u, NULL);
}

static int32_t vx_sd_ext_sequence_vision_profile(
        VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* input;
    VxShapeDomainDimension dimensions[3];
    uint64_t channels;
    uint64_t doubled;
    if (vx_sd_exact_ports(build, INPUTS, 1u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, NULL, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (vx_sd_assert_float(build, &input->descriptor, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (input->descriptor.rank != 4u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, input->descriptor.rank);
    vx_sd_ext_sequence_copy_dimension(
        &dimensions[0], &input->descriptor.dimensions[0]);
    if (build->request->operator_kind == VX_OP_PROFILE_X ||
        build->request->operator_kind == VX_OP_PROFILE_Y) {
        if (!vx_shape_domain_fixed_value(&input->descriptor.dimensions[3],
                build->request->environment,
                build->request->environment_count, &channels))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                              0u, 3u);
        if (vx_sd_checked_multiply(channels, 2u, &doubled))
            return vx_sd_fail(build,
                              VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                              0u, 3u);
        vx_sd_ext_sequence_fixed_dimension(&dimensions[1], doubled);
        vx_sd_ext_sequence_copy_dimension(
            &dimensions[2],
            build->request->operator_kind == VX_OP_PROFILE_X
                ? &input->descriptor.dimensions[2]
                : &input->descriptor.dimensions[1]);
    } else {
        vx_sd_ext_sequence_copy_dimension(
            &dimensions[1], &input->descriptor.dimensions[3]);
        vx_sd_ext_sequence_copy_dimension(
            &dimensions[2], &input->descriptor.dimensions[2]);
    }
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32,
                            dimensions, 3u, NULL);
}

static int32_t vx_sd_prove_extended_sequence(VxShapeDomainBuild* build) {
    switch (build->request->operator_kind) {
        case VX_OP_NOT:
            return vx_sd_ext_sequence_not(build);
        case VX_OP_MASK:
            return vx_sd_ext_sequence_mask(build);
        case VX_OP_SSM_SCAN:
        case VX_OP_SELECTIVE_SCAN:
            return vx_sd_ext_sequence_scan(build);
        case VX_OP_SPATIAL_SOFTARGMAX_Y:
        case VX_OP_MEAN_HEIGHT:
        case VX_OP_PROFILE_X:
        case VX_OP_PROFILE_Y:
            return vx_sd_ext_sequence_vision_profile(build);
        default:
            return VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED;
    }
}
