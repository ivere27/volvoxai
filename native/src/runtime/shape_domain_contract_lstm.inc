/* LSTM preserves the time and batch domains, with fixed learned projections. */
static int32_t vx_sd_lstm(VxShapeDomainBuild* build) {
    static const char* const REQUIRED[] = {"input", "weight", "recurrent_weight"};
    static const char* const OPTIONAL[] = {"bias", "initial_h", "initial_c"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const PARAMS[] = {"direction"};
    if (vx_sd_optional_ports(build, REQUIRED, 3u, OPTIONAL, 3u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, PARAMS, 1u)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    const VxShapeDomainTensorDescriptor* x = &vx_sd_input(build->request, "input")->descriptor;
    const VxShapeDomainTensorDescriptor* w = &vx_sd_input(build->request, "weight")->descriptor;
    const VxShapeDomainTensorDescriptor* r = &vx_sd_input(build->request, "recurrent_weight")->descriptor;
    int64_t direction;
    uint64_t hidden, width;
    if (vx_sd_number_integer(build, vx_sd_param(build->request, "direction"), 0, &direction) ||
        direction < 0 || direction > 2)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS, 0u, 0u);
    if (vx_sd_assert_float(build, x, 0u) || vx_sd_assert_float(build, w, 1u) ||
        vx_sd_assert_float(build, r, 2u)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (x->rank != 3u || w->rank != 3u || r->rank != 3u ||
        !vx_sd_direct_constant_shape(w) || !vx_sd_direct_constant_shape(r) ||
        !vx_shape_domain_fixed_value(&r->dimensions[2], build->request->environment,
            build->request->environment_count, &hidden) || !hidden || hidden > UINT32_MAX / 8u ||
        !vx_shape_domain_fixed_value(&x->dimensions[2], build->request->environment,
            build->request->environment_count, &width))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE, 0u, 0u);
    uint64_t directions = direction == 2 ? 2u : 1u;
    VxShapeDomainDimension wd[3] = {vx_sd_fixed(directions), vx_sd_fixed(4u * hidden), vx_sd_fixed(width)};
    VxShapeDomainDimension rd[3] = {vx_sd_fixed(directions), vx_sd_fixed(4u * hidden), vx_sd_fixed(hidden)};
    for (uint32_t axis = 0; axis < 3u; ++axis)
        if (!vx_shape_domain_dimensions_equal(&w->dimensions[axis], &wd[axis], build->request->environment, build->request->environment_count) ||
            !vx_shape_domain_dimensions_equal(&r->dimensions[axis], &rd[axis], build->request->environment, build->request->environment_count))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH, 1u, axis);
    for (uint32_t index = 0; index < 3u; ++index) {
        const VxShapeDomainNamedTensor* optional = vx_sd_input(build->request, OPTIONAL[index]);
        if (!optional) continue;
        const VxShapeDomainTensorDescriptor* value = &optional->descriptor;
        uint32_t rank = index ? 3u : 2u;
        VxShapeDomainDimension expected[3] = {vx_sd_fixed(directions),
            index ? x->dimensions[1] : vx_sd_fixed(8u * hidden), vx_sd_fixed(hidden)};
        if (vx_sd_assert_float(build, value, index + 3u)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
        if (value->rank != rank) return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK, index + 3u, 0u);
        for (uint32_t axis = 0; axis < rank; ++axis)
            if (!vx_shape_domain_dimensions_equal(&value->dimensions[axis], &expected[axis], build->request->environment, build->request->environment_count))
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH, index + 3u, axis);
    }
    VxShapeDomainDimension output[4] = {x->dimensions[0], vx_sd_fixed(directions), x->dimensions[1], vx_sd_fixed(hidden)};
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32, output, 4u, NULL);
}

static int32_t vx_sd_packed_lstm(VxShapeDomainBuild* build) {
    static const char* const REQUIRED[] = {
        "input", "weight", "scale", "recurrent_weight", "recurrent_scale"
    };
    static const char* const OPTIONAL[] = {"bias", "initial_h", "initial_c"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const PARAMS[] = {"direction", "group_size", "weight_bits"};
    if (vx_sd_optional_ports(build, REQUIRED, 5u, OPTIONAL, 3u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, PARAMS, 3u)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    const VxShapeDomainTensorDescriptor* x = &vx_sd_input(build->request, "input")->descriptor;
    const VxShapeDomainTensorDescriptor* w = &vx_sd_input(build->request, "weight")->descriptor;
    const VxShapeDomainTensorDescriptor* ws = &vx_sd_input(build->request, "scale")->descriptor;
    const VxShapeDomainTensorDescriptor* r = &vx_sd_input(build->request, "recurrent_weight")->descriptor;
    const VxShapeDomainTensorDescriptor* rs = &vx_sd_input(build->request, "recurrent_scale")->descriptor;
    int64_t direction, group, weight_bits;
    uint64_t gates_hidden, hidden, width;
    if (vx_sd_number_integer(build, vx_sd_param(build->request, "direction"), 0, &direction) ||
        direction < 0 || direction > 2 ||
        vx_sd_number_integer(build, vx_sd_param(build->request, "group_size"), 0, &group) ||
        group < 1 || group > INT32_MAX || !vx_packed_group_size_supported((int)group) ||
        vx_sd_number_integer(build, vx_sd_param(build->request, "weight_bits"), 4, &weight_bits) ||
        (weight_bits != 2 && weight_bits != 3 && weight_bits != 4 && weight_bits != 8))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS, 0u, 0u);
    if (vx_sd_assert_float(build, x, 0u) || vx_sd_assert_float(build, ws, 2u) ||
        vx_sd_assert_float(build, rs, 4u)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (w->dtype != VX_DTYPE_I32 || w->quantization.scheme)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE, 1u, 0u);
    if (r->dtype != VX_DTYPE_I32 || r->quantization.scheme)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE, 3u, 0u);
    if (x->rank != 3u || w->rank != 3u || ws->rank != 3u || r->rank != 3u || rs->rank != 3u ||
        !vx_sd_direct_constant_shape(w) || !vx_sd_direct_constant_shape(ws) ||
        !vx_sd_direct_constant_shape(r) || !vx_sd_direct_constant_shape(rs) ||
        !vx_shape_domain_fixed_value(&r->dimensions[1], build->request->environment,
            build->request->environment_count, &gates_hidden) ||
        !gates_hidden || (gates_hidden % 4u) != 0u || (gates_hidden / 4u) > UINT32_MAX / 8u ||
        !vx_shape_domain_fixed_value(&x->dimensions[2], build->request->environment,
            build->request->environment_count, &width) || !width || width > INT32_MAX)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE, 0u, 0u);
    hidden = gates_hidden / 4u;
    uint64_t directions = direction == 2 ? 2u : 1u;
    uint64_t w_words = (width / 32u) * (uint64_t)weight_bits +
        ((width % 32u) * (uint64_t)weight_bits + 31u) / 32u;
    uint64_t w_groups = width / (uint64_t)group + (width % (uint64_t)group != 0u);
    uint64_t r_words = (hidden / 32u) * (uint64_t)weight_bits +
        ((hidden % 32u) * (uint64_t)weight_bits + 31u) / 32u;
    uint64_t r_groups = hidden / (uint64_t)group + (hidden % (uint64_t)group != 0u);
    VxShapeDomainDimension wd[3] = {vx_sd_fixed(directions), vx_sd_fixed(4u * hidden), vx_sd_fixed(w_words)};
    VxShapeDomainDimension wsd[3] = {vx_sd_fixed(directions), vx_sd_fixed(4u * hidden), vx_sd_fixed(w_groups)};
    VxShapeDomainDimension rd[3] = {vx_sd_fixed(directions), vx_sd_fixed(4u * hidden), vx_sd_fixed(r_words)};
    VxShapeDomainDimension rsd[3] = {vx_sd_fixed(directions), vx_sd_fixed(4u * hidden), vx_sd_fixed(r_groups)};
    for (uint32_t axis = 0; axis < 3u; ++axis) {
        if (!vx_shape_domain_dimensions_equal(&w->dimensions[axis], &wd[axis], build->request->environment, build->request->environment_count) ||
            !vx_shape_domain_dimensions_equal(&ws->dimensions[axis], &wsd[axis], build->request->environment, build->request->environment_count) ||
            !vx_shape_domain_dimensions_equal(&r->dimensions[axis], &rd[axis], build->request->environment, build->request->environment_count) ||
            !vx_shape_domain_dimensions_equal(&rs->dimensions[axis], &rsd[axis], build->request->environment, build->request->environment_count))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH, 1u, axis);
    }
    for (uint32_t index = 0; index < 3u; ++index) {
        const VxShapeDomainNamedTensor* optional = vx_sd_input(build->request, OPTIONAL[index]);
        if (!optional) continue;
        const VxShapeDomainTensorDescriptor* value = &optional->descriptor;
        uint32_t rank = index ? 3u : 2u;
        VxShapeDomainDimension expected[3] = {vx_sd_fixed(directions),
            index ? x->dimensions[1] : vx_sd_fixed(8u * hidden), vx_sd_fixed(hidden)};
        if (vx_sd_assert_float(build, value, index + 5u)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
        if (value->rank != rank) return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK, index + 5u, 0u);
        for (uint32_t axis = 0; axis < rank; ++axis)
            if (!vx_shape_domain_dimensions_equal(&value->dimensions[axis], &expected[axis], build->request->environment, build->request->environment_count))
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH, index + 5u, axis);
    }
    VxShapeDomainDimension output[4] = {x->dimensions[0], vx_sd_fixed(directions), x->dimensions[1], vx_sd_fixed(hidden)};
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32, output, 4u, NULL);
}
