/* Structural logical shape-domain contracts. This fragment is included only
 * by shape_domain_contract.c and uses the caller's per-call scratch arena. */

static int32_t vx_sd_struct_single_ports(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input"};
    static const char* const OUTPUTS[] = {"out"};
    return vx_sd_exact_ports(build, INPUTS, 1u, OUTPUTS, 1u);
}

static VxShapeDomainDimension* vx_sd_output_dimensions(
        VxShapeDomainBuild* build,
        const char* name) {
    uint32_t index;
    if (vx_sd_result_output_index(build, name, &index)) return NULL;
    return (VxShapeDomainDimension*)
        build->result->outputs[index].descriptor.dimensions;
}

static int vx_sd_remap_reshape_quantization(
        const VxShapeDomainTensorDescriptor* input,
        const VxShapeDomainDimension* output_dimensions,
        uint32_t output_rank,
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count,
        VxShapeDomainQuantization* output) {
    const VxShapeDomainQuantization* quantization = &input->quantization;
    uint32_t candidate = VX_SHAPE_DOMAIN_INDEX_NONE;
    uint32_t candidate_count = 0u;
    if (!output) return -1;
    *output = *quantization;
    if (quantization->scheme != 2) return 0;
    if (quantization->axis >= input->rank) return -1;
    for (uint32_t axis = 0u; axis < output_rank; ++axis) {
        if (!vx_shape_domain_dimensions_equal(
                &input->dimensions[quantization->axis],
                &output_dimensions[axis], environment, environment_count))
            continue;
        if (!vx_sd_products_equal(input->dimensions, quantization->axis,
                output_dimensions, axis, environment, environment_count) ||
            !vx_sd_products_equal(
                input->dimensions + quantization->axis + 1u,
                input->rank - quantization->axis - 1u,
                output_dimensions + axis + 1u,
                output_rank - axis - 1u,
                environment, environment_count))
            continue;
        candidate = axis;
        candidate_count++;
    }
    if (candidate_count != 1u) return -1;
    output->axis = candidate;
    return 0;
}

static int vx_sd_normalize_axes(VxShapeDomainBuild* build,
                                const VxShapeDomainParam* param,
                                uint32_t rank,
                                uint32_t* axes,
                                uint32_t* count) {
    if (!param || param->kind != VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY ||
        !axes || !count || (param->value_count && !param->values))
        return -1;
    for (uint32_t index = 0u; index < param->value_count; ++index) {
        int64_t raw;
        if (vx_sd_param_array_integer(param, index, &raw)) return -1;
        if (raw < 0) raw += rank;
        if (raw < 0 || (uint64_t)raw >= rank) return -1;
        for (uint32_t previous = 0u; previous < index; ++previous)
            if (axes[previous] == (uint32_t)raw) return -1;
        axes[index] = (uint32_t)raw;
    }
    *count = param->value_count;
    (void)build;
    return 0;
}

static int32_t vx_sd_struct_broadcast_binary(VxShapeDomainBuild* build,
                                             int comparison) {
    static const char* const INPUTS[] = {"a", "b"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* left;
    const VxShapeDomainNamedTensor* right;
    VxShapeDomainDimension dimensions[8];
    uint32_t rank;
    if (vx_sd_exact_ports(build, INPUTS, 2u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, NULL, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    left = vx_sd_input(build->request, "a");
    right = vx_sd_input(build->request, "b");
    if (comparison) {
        if (left->descriptor.dtype != VX_DTYPE_I32 ||
            right->descriptor.dtype != VX_DTYPE_I32)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                              0u, 1u);
        if (!vx_sd_unquantized(&left->descriptor) ||
            !vx_sd_unquantized(&right->descriptor))
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u, 1u);
    } else if (vx_sd_assert_float(build, &left->descriptor, 0u) ||
               vx_sd_assert_float(build, &right->descriptor, 1u)) {
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    }
    if (vx_sd_broadcast_shape(&left->descriptor, &right->descriptor,
            build->request->environment,
            build->request->environment_count, dimensions, &rank))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          0u, 1u);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_BROADCAST;
    return vx_sd_set_output(build, "out",
                            comparison ? VX_DTYPE_I32 : VX_DTYPE_F32,
                            dimensions, rank, NULL);
}

static int32_t vx_sd_struct_where(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"condition", "a", "b"};
    static const char* const OUTPUTS[] = {"out"};
    const VxShapeDomainNamedTensor* condition;
    const VxShapeDomainNamedTensor* left;
    const VxShapeDomainNamedTensor* right;
    VxShapeDomainDimension data[8];
    VxShapeDomainDimension dimensions[8];
    VxShapeDomainTensorDescriptor intermediate;
    uint32_t data_rank;
    uint32_t rank;
    if (vx_sd_exact_ports(build, INPUTS, 3u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, NULL, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    condition = vx_sd_input(build->request, "condition");
    left = vx_sd_input(build->request, "a");
    right = vx_sd_input(build->request, "b");
    if ((condition->descriptor.dtype != VX_DTYPE_F32 &&
         condition->descriptor.dtype != VX_DTYPE_I32) ||
        left->descriptor.dtype != right->descriptor.dtype ||
        (left->descriptor.dtype != VX_DTYPE_F32 &&
         left->descriptor.dtype != VX_DTYPE_I32))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                          0u, 1u);
    if (!vx_sd_unquantized(&condition->descriptor) ||
        !vx_sd_unquantized(&left->descriptor) ||
        !vx_sd_unquantized(&right->descriptor))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION,
                          0u, 1u);
    if (vx_sd_broadcast_shape(&left->descriptor, &right->descriptor,
            build->request->environment,
            build->request->environment_count, data, &data_rank))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE, 1u, 2u);
    vx_sd_clear(&intermediate, (uint32_t)sizeof(intermediate));
    intermediate.dtype = left->descriptor.dtype;
    intermediate.rank = data_rank;
    intermediate.dimensions = data;
    if (vx_sd_broadcast_shape(&condition->descriptor, &intermediate,
            build->request->environment,
            build->request->environment_count, dimensions, &rank))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE, 0u, 1u);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_BROADCAST;
    return vx_sd_set_output(build, "out", left->descriptor.dtype,
                            dimensions, rank, NULL);
}

static int32_t vx_sd_struct_transpose(VxShapeDomainBuild* build) {
    static const char* const PERM[] = {"perm"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainParam* param;
    VxShapeDomainDimension dimensions[8];
    VxShapeDomainQuantization quantization;
    uint32_t permutation[8];
    uint32_t rank;
    if (vx_sd_struct_single_ports(build) ||
        vx_sd_allowed_params(build, PERM, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    rank = input->descriptor.rank;
    if (!rank || rank > 8u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, rank);
    param = vx_sd_param(build->request, "perm");
    if (!param) {
        for (uint32_t axis = 0u; axis < rank; ++axis)
            permutation[axis] = rank - axis - 1u;
    } else {
        uint32_t count;
        if (vx_sd_normalize_axes(build, param, rank, permutation, &count) ||
            count != rank)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
    }
    for (uint32_t axis = 0u; axis < rank; ++axis)
        dimensions[axis] = input->descriptor.dimensions[permutation[axis]];
    quantization = input->descriptor.quantization;
    if (quantization.scheme == 2) {
        uint32_t output_axis = VX_SHAPE_DOMAIN_INDEX_NONE;
        for (uint32_t axis = 0u; axis < rank; ++axis)
            if (permutation[axis] == quantization.axis) output_axis = axis;
        if (output_axis == VX_SHAPE_DOMAIN_INDEX_NONE)
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u,
                quantization.axis);
        quantization.axis = output_axis;
    }
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", input->descriptor.dtype,
                            dimensions, rank, &quantization);
}

static int32_t vx_sd_struct_flatten(VxShapeDomainBuild* build) {
    static const char* const AXIS[] = {"axis"};
    const VxShapeDomainNamedTensor* input;
    VxShapeDomainDimension dimensions[2];
    VxShapeDomainQuantization quantization;
    int64_t raw;
    uint32_t axis;
    if (vx_sd_struct_single_ports(build) ||
        vx_sd_allowed_params(build, AXIS, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (vx_sd_number_integer(build, vx_sd_param(build->request, "axis"),
            1, &raw)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (raw < 0) raw += input->descriptor.rank;
    if (raw < 0 || (uint64_t)raw > input->descriptor.rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    axis = (uint32_t)raw;
    if (vx_sd_collapse_dimensions(input->descriptor.dimensions, axis,
            build->request->environment,
            build->request->environment_count, &dimensions[0]) ||
        vx_sd_collapse_dimensions(axis < input->descriptor.rank
                ? input->descriptor.dimensions + axis : NULL,
            input->descriptor.rank - axis, build->request->environment,
            build->request->environment_count, &dimensions[1]))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          0u, axis);
    if (vx_sd_remap_reshape_quantization(&input->descriptor, dimensions, 2u,
            build->request->environment,
            build->request->environment_count, &quantization))
        return vx_sd_fail(build,
            VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u,
            input->descriptor.quantization.axis);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", input->descriptor.dtype,
                            dimensions, 2u, &quantization);
}

static int32_t vx_sd_struct_squeeze(VxShapeDomainBuild* build) {
    static const char* const AXES[] = {"axes"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainParam* param;
    VxShapeDomainDimension dimensions[8];
    VxShapeDomainQuantization quantization;
    uint32_t axes[8];
    uint32_t axis_count;
    uint32_t output_rank = 0u;
    if (vx_sd_struct_single_ports(build) ||
        vx_sd_allowed_params(build, AXES, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (input->descriptor.rank > 8u)
        return vx_sd_unsupported(build);
    param = vx_sd_param(build->request, "axes");
    if (vx_sd_normalize_axes(build, param, input->descriptor.rank,
            axes, &axis_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    for (uint32_t index = 0u; index < axis_count; ++index) {
        uint64_t extent;
        if (!vx_shape_domain_fixed_value(
                &input->descriptor.dimensions[axes[index]],
                build->request->environment,
                build->request->environment_count, &extent) || extent != 1u)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                              0u, axes[index]);
    }
    for (uint32_t axis = 0u; axis < input->descriptor.rank; ++axis) {
        int removed = 0;
        for (uint32_t index = 0u; index < axis_count; ++index)
            if (axes[index] == axis) removed = 1;
        if (!removed) dimensions[output_rank++] =
            input->descriptor.dimensions[axis];
    }
    if (vx_sd_remap_reshape_quantization(&input->descriptor, dimensions,
            output_rank, build->request->environment,
            build->request->environment_count, &quantization))
        return vx_sd_fail(build,
            VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u,
            input->descriptor.quantization.axis);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", input->descriptor.dtype,
                            dimensions, output_rank, &quantization);
}

static int32_t vx_sd_struct_unsqueeze(VxShapeDomainBuild* build) {
    static const char* const AXES[] = {"axes"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainParam* param;
    VxShapeDomainDimension dimensions[8];
    VxShapeDomainQuantization quantization;
    uint32_t axes[8];
    uint32_t axis_count;
    uint32_t output_rank;
    uint32_t input_axis = 0u;
    if (vx_sd_struct_single_ports(build) ||
        vx_sd_allowed_params(build, AXES, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    param = vx_sd_param(build->request, "axes");
    if (!param || param->kind != VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY ||
        param->value_count > UINT32_MAX - input->descriptor.rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    output_rank = input->descriptor.rank + param->value_count;
    if (output_rank > 8u) return vx_sd_unsupported(build);
    if (vx_sd_normalize_axes(build, param, output_rank, axes, &axis_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    for (uint32_t axis = 0u; axis < output_rank; ++axis) {
        int inserted = 0;
        for (uint32_t index = 0u; index < axis_count; ++index)
            if (axes[index] == axis) inserted = 1;
        dimensions[axis] = inserted
            ? vx_sd_fixed(1u)
            : input->descriptor.dimensions[input_axis++];
    }
    if (vx_sd_remap_reshape_quantization(&input->descriptor, dimensions,
            output_rank, build->request->environment,
            build->request->environment_count, &quantization))
        return vx_sd_fail(build,
            VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u,
            input->descriptor.quantization.axis);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", input->descriptor.dtype,
                            dimensions, output_rank, &quantization);
}

static int32_t vx_sd_target_shape(VxShapeDomainBuild* build,
                                  int32_t dtype,
                                  VxShapeDomainDimension* dimensions,
                                  uint32_t capacity,
                                  uint32_t* rank) {
    static const char* const SHAPE[] = {"shape"};
    const VxShapeDomainParam* param;
    const VxShapeDomainNamedTensor* declared;
    if (vx_sd_allowed_params(build, SHAPE, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    param = vx_sd_param(build->request, "shape");
    if (!param || param->kind != VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY ||
        (param->value_count && !param->values))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (param->value_count > capacity) return vx_sd_unsupported(build);
    for (uint32_t axis = 0u; axis < param->value_count; ++axis) {
        if (vx_sd_target_dimension(&param->values[axis], &dimensions[axis]) ||
            !vx_sd_dimension_valid(&dimensions[axis],
                build->request->environment,
                build->request->environment_count))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE, axis);
    }
    *rank = param->value_count;
    if (!build->request->use_declared_outputs)
        return VX_SHAPE_DOMAIN_STATUS_OK;
    declared = vx_sd_named(build->request->declared_outputs,
                           build->request->declared_output_count, "out");
    if (!declared)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (declared->descriptor.dtype != dtype)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                          0u, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (declared->descriptor.rank != *rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          declared->descriptor.rank, *rank);
    for (uint32_t axis = 0u; axis < *rank; ++axis)
        if (!vx_shape_domain_dimensions_equal(&dimensions[axis],
                &declared->descriptor.dimensions[axis],
                build->request->environment,
                build->request->environment_count))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                              0u, axis);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_struct_reshape_or_expand(VxShapeDomainBuild* build,
                                             int expand) {
    const VxShapeDomainNamedTensor* input;
    VxShapeDomainDimension dimensions[8];
    VxShapeDomainQuantization quantization;
    uint32_t rank;
    int32_t status;
    if (vx_sd_struct_single_ports(build))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    status = vx_sd_target_shape(build, input->descriptor.dtype, dimensions,
                                8u, &rank);
    if (status != VX_SHAPE_DOMAIN_STATUS_OK) return status;
    if (!expand) {
        if (!vx_sd_products_equal(input->descriptor.dimensions,
                input->descriptor.rank, dimensions, rank,
                build->request->environment,
                build->request->environment_count))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                              0u, VX_SHAPE_DOMAIN_INDEX_NONE);
        if (vx_sd_remap_reshape_quantization(&input->descriptor, dimensions,
                rank, build->request->environment,
                build->request->environment_count, &quantization))
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u,
                input->descriptor.quantization.axis);
    } else {
        uint32_t offset;
        if (rank < input->descriptor.rank || rank > 8u)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                              input->descriptor.rank, rank);
        offset = rank - input->descriptor.rank;
        for (uint32_t axis = 0u; axis < rank; ++axis) {
            VxShapeDomainDimension singleton = vx_sd_fixed(1u);
            const VxShapeDomainDimension* source = axis < offset
                ? &singleton
                : &input->descriptor.dimensions[axis - offset];
            uint64_t fixed;
            VxShapeDomainDimension broadcast;
            if (vx_sd_broadcast_dimension(source, &dimensions[axis],
                    build->request->environment,
                    build->request->environment_count, &broadcast) ||
                ((!vx_shape_domain_fixed_value(source,
                      build->request->environment,
                      build->request->environment_count, &fixed) ||
                  fixed != 1u) &&
                 !vx_shape_domain_dimensions_equal(source, &dimensions[axis],
                     build->request->environment,
                     build->request->environment_count)))
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                                  0u, axis);
        }
        quantization = input->descriptor.quantization;
        if (quantization.scheme == 2) {
            uint32_t output_axis = quantization.axis + offset;
            if (output_axis >= rank ||
                !vx_shape_domain_dimensions_equal(
                    &input->descriptor.dimensions[quantization.axis],
                    &dimensions[output_axis], build->request->environment,
                    build->request->environment_count))
                return vx_sd_fail(build,
                    VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u,
                    quantization.axis);
            quantization.axis = output_axis;
        }
    }
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", input->descriptor.dtype,
                            dimensions, rank, &quantization);
}

static int32_t vx_sd_concat_quantization(VxShapeDomainBuild* build,
                                         uint32_t axis,
                                         VxShapeDomainQuantization* output) {
    const VxShapeDomainNamedTensor* first =
        vx_sd_variadic_input(build->request, "input", 0u);
    uint64_t total = 0u;
    int all_equal = 1;
    *output = first->descriptor.quantization;
    for (uint32_t index = 1u; index < build->request->input_count; ++index) {
        const VxShapeDomainNamedTensor* input =
            vx_sd_variadic_input(build->request, "input", index);
        if (!vx_sd_quantization_equal(&first->descriptor.quantization,
                                      &input->descriptor.quantization))
            all_equal = 0;
    }
    if (output->scheme != 2 || output->axis != axis) {
        if (!all_equal)
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u, axis);
        return VX_SHAPE_DOMAIN_STATUS_OK;
    }
    for (uint32_t index = 0u; index < build->request->input_count; ++index) {
        const VxShapeDomainNamedTensor* input =
            vx_sd_variadic_input(build->request, "input", index);
        const VxShapeDomainQuantization* quantization =
            &input->descriptor.quantization;
        if (quantization->scheme != 2 || quantization->axis != axis ||
            total + quantization->count > UINT32_MAX)
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, index, axis);
        total += quantization->count;
    }
    if (total) {
        uint32_t* scales = (uint32_t*)vx_sd_allocate(build->arena,
            (uint32_t)total, (uint32_t)sizeof(*scales), 4u);
        int32_t* zero_points = (int32_t*)vx_sd_allocate(build->arena,
            (uint32_t)total, (uint32_t)sizeof(*zero_points), 4u);
        uint32_t cursor = 0u;
        if (!scales || !zero_points)
            return VX_SHAPE_DOMAIN_STATUS_SCRATCH_TOO_SMALL;
        for (uint32_t index = 0u;
             index < build->request->input_count; ++index) {
            const VxShapeDomainQuantization* quantization =
                &vx_sd_variadic_input(build->request, "input", index)
                    ->descriptor.quantization;
            for (uint32_t item = 0u; item < quantization->count; ++item) {
                scales[cursor] = quantization->scales[item];
                zero_points[cursor] = quantization->zero_points[item];
                cursor++;
            }
        }
        output->scales = scales;
        output->zero_points = zero_points;
    }
    output->count = (uint32_t)total;
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_struct_concat(VxShapeDomainBuild* build) {
    static const char* const AXIS[] = {"axis", "sigmoid"};
    const VxShapeDomainNamedTensor* first;
    const VxShapeDomainNamedTensor* declared;
    VxShapeDomainDimension dimensions[8];
    VxShapeDomainQuantization quantization;
    VxShapeDomainProgression source_progression;
    VxShapeDomainProgression target_progression;
    uint64_t fixed_sum = 0u;
    uint32_t dynamic_symbol = VX_SHAPE_DOMAIN_INDEX_NONE;
    uint32_t dynamic_input = VX_SHAPE_DOMAIN_INDEX_NONE;
    uint32_t axis;
    uint32_t rank;
    int32_t status;
    if (!vx_sd_variadic_inputs_valid(build->request, "input", 2u) ||
        build->request->declared_output_count != 1u ||
        !vx_sd_named(build->request->declared_outputs, 1u, "out"))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS,
                          build->request->input_count,
                          build->request->declared_output_count);
    if (vx_sd_allowed_params(build, AXIS, 2u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    first = vx_sd_variadic_input(build->request, "input", 0u);
    const VxShapeDomainParam* sigmoid = vx_sd_param(build->request, "sigmoid");
    if (sigmoid) {
        int enabled;
        if (sigmoid->kind == VX_SHAPE_DOMAIN_PARAM_BOOLEAN) enabled = sigmoid->boolean != 0;
        else if (sigmoid->kind == VX_SHAPE_DOMAIN_PARAM_NUMBER &&
                 (sigmoid->number == 0.0 || sigmoid->number == 1.0)) enabled = sigmoid->number != 0.0;
        else return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS, 0u, 0u);
        if (enabled && first->descriptor.dtype != VX_DTYPE_F32)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE, 0u, 0u);
    }
    rank = first->descriptor.rank;
    if (!rank || rank > 8u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, rank);
    if (vx_sd_axis(build, "axis", rank, 0, &axis))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    for (uint32_t dimension = 0u; dimension < rank; ++dimension)
        dimensions[dimension] = first->descriptor.dimensions[dimension];
    for (uint32_t input_index = 0u;
         input_index < build->request->input_count; ++input_index) {
        const VxShapeDomainNamedTensor* input =
            vx_sd_variadic_input(build->request, "input", input_index);
        uint64_t fixed;
        if (input->descriptor.rank != rank)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                              input_index, input->descriptor.rank);
        if (input->descriptor.dtype != first->descriptor.dtype)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                              input_index, VX_SHAPE_DOMAIN_INDEX_NONE);
        for (uint32_t dimension = 0u; dimension < rank; ++dimension)
            if (dimension != axis &&
                !vx_shape_domain_dimensions_equal(
                    &input->descriptor.dimensions[dimension],
                    &first->descriptor.dimensions[dimension],
                    build->request->environment,
                    build->request->environment_count))
                return vx_sd_fail(build,
                    VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                    input_index, dimension);
        if (vx_shape_domain_fixed_value(
                &input->descriptor.dimensions[axis],
                build->request->environment,
                build->request->environment_count, &fixed)) {
            if (vx_sd_checked_add(fixed_sum, fixed, &fixed_sum))
                return vx_sd_fail(build,
                    VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                    input_index, axis);
        } else {
            const VxShapeDomainDimension* dimension =
                &input->descriptor.dimensions[axis];
            if (dynamic_symbol != VX_SHAPE_DOMAIN_INDEX_NONE ||
                dimension->kind != VX_SHAPE_DOMAIN_DIMENSION_SYMBOL)
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                                  input_index, axis);
            dynamic_symbol = dimension->symbol_index;
            dynamic_input = input_index;
        }
    }
    status = vx_sd_concat_quantization(build, axis, &quantization);
    if (status != VX_SHAPE_DOMAIN_STATUS_OK) return status;
    if (dynamic_symbol == VX_SHAPE_DOMAIN_INDEX_NONE) {
        dimensions[axis] = vx_sd_fixed(fixed_sum);
        build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
        return vx_sd_set_output(build, "out", first->descriptor.dtype,
                                dimensions, rank, &quantization);
    }
    if (!build->request->use_declared_outputs)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          dynamic_input, axis);
    declared = vx_sd_named(build->request->declared_outputs, 1u, "out");
    if (declared->descriptor.rank != rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, declared->descriptor.rank);
    if (declared->descriptor.dtype != first->descriptor.dtype)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                          0u, VX_SHAPE_DOMAIN_INDEX_NONE);
    for (uint32_t dimension = 0u; dimension < rank; ++dimension) {
        if (dimension == axis) continue;
        if (!vx_shape_domain_dimensions_equal(
                &declared->descriptor.dimensions[dimension],
                &dimensions[dimension], build->request->environment,
                build->request->environment_count))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                              0u, dimension);
        /* Match the TS observable spelling: non-axis dimensions are replaced
         * by the graph-authored descriptor after equality is proved. */
        dimensions[dimension] = declared->descriptor.dimensions[dimension];
    }
    if (declared->descriptor.dimensions[axis].kind !=
            VX_SHAPE_DOMAIN_DIMENSION_SYMBOL)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE, 0u, axis);
    for (uint32_t input_index = 0u;
         input_index < build->request->input_count; ++input_index) {
        const VxShapeDomainNamedTensor* input =
            vx_sd_variadic_input(build->request, "input", input_index);
        for (uint32_t dimension = 0u;
             dimension < input->descriptor.rank; ++dimension)
            if (input->descriptor.dimensions[dimension].kind !=
                    VX_SHAPE_DOMAIN_DIMENSION_FIXED &&
                input->descriptor.dimensions[dimension].symbol_index ==
                    declared->descriptor.dimensions[axis].symbol_index)
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                                  input_index, dimension);
    }
    if (vx_shape_domain_legal_progression(build->request->environment,
            build->request->environment_count, dynamic_symbol,
            &source_progression) ||
        vx_shape_domain_legal_progression(build->request->environment,
            build->request->environment_count,
            declared->descriptor.dimensions[axis].symbol_index,
            &target_progression))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                          0u, axis);
    {
        uint64_t expected_first;
        uint64_t expected_last;
        if (vx_sd_checked_add(source_progression.first, fixed_sum,
                &expected_first) ||
            vx_sd_checked_add(source_progression.last, fixed_sum,
                &expected_last))
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW, 0u, axis);
        if (target_progression.first != expected_first ||
            target_progression.last != expected_last ||
            target_progression.count != source_progression.count)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                              0u, axis);
    }
    dimensions[axis] = declared->descriptor.dimensions[axis];
    if (fixed_sum > INT64_MAX || vx_sd_add_affine(build,
            dimensions[axis].symbol_index, dynamic_symbol,
            (int64_t)fixed_sum))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INTERNAL,
                          dimensions[axis].symbol_index, dynamic_symbol);
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_AFFINE;
    return vx_sd_set_output(build, "out", first->descriptor.dtype,
                            dimensions, rank, &quantization);
}

static int32_t vx_sd_set_variadic_output(
        VxShapeDomainBuild* build,
        const char* prefix,
        uint32_t suffix,
        int32_t dtype,
        const VxShapeDomainDimension* dimensions,
        uint32_t rank,
        const VxShapeDomainQuantization* quantization) {
    VxShapeDomainNamedTensor* output = NULL;
    uint32_t output_index = VX_SHAPE_DOMAIN_INDEX_NONE;
    for (uint32_t index = 0u; index < build->result->output_count; ++index) {
        uint32_t parsed;
        if (!vx_sd_port_suffix(&build->result->outputs[index].name,
                               prefix, &parsed) && parsed == suffix) {
            output = &build->result->outputs[index];
            output_index = index;
            break;
        }
    }
    if (!output)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS,
                          suffix, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (output->descriptor.rank != rank || (rank && !dimensions))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          output_index, rank);
    for (uint32_t axis = 0u; axis < rank; ++axis) {
        if (!vx_sd_dimension_valid(&dimensions[axis],
                build->request->environment,
                build->request->environment_count))
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_DIMENSION, output_index, axis);
        ((VxShapeDomainDimension*)output->descriptor.dimensions)[axis] =
            dimensions[axis];
    }
    output->descriptor.dtype = dtype;
    if (quantization) output->descriptor.quantization = *quantization;
    else vx_sd_clear(&output->descriptor.quantization,
                     (uint32_t)sizeof(output->descriptor.quantization));
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_struct_split(VxShapeDomainBuild* build) {
    static const char* const ALLOWED[] = {"axis", "split", "num_outputs"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainParam* split;
    const VxShapeDomainParam* count_param;
    uint64_t axis_extent;
    uint64_t sum = 0u;
    uint64_t common_size = 0u;
    uint32_t output_count;
    uint32_t axis;
    if (build->request->input_count != 1u ||
        !vx_sd_input(build->request, "input") ||
        !vx_sd_variadic_outputs_valid(build->request, "out", 1u))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS,
                          build->request->input_count,
                          build->request->declared_output_count);
    if (vx_sd_allowed_params(build, ALLOWED, 3u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (!input->descriptor.rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, 0u);
    if (vx_sd_axis(build, "axis", input->descriptor.rank, 0, &axis))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!vx_shape_domain_fixed_value(&input->descriptor.dimensions[axis],
            build->request->environment,
            build->request->environment_count, &axis_extent))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE, 0u, axis);
    split = vx_sd_param(build->request, "split");
    count_param = vx_sd_param(build->request, "num_outputs");
    if (split && count_param)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (split) {
        if (split->kind != VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY ||
            !split->value_count ||
            split->value_count != build->request->declared_output_count ||
            !split->values)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        output_count = split->value_count;
        for (uint32_t index = 0u; index < output_count; ++index) {
            int64_t size;
            uint64_t next;
            if (vx_sd_param_array_integer(split, index, &size) || size <= 0 ||
                vx_sd_checked_add(sum, (uint64_t)size, &next))
                return vx_sd_fail(build,
                    VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS, index,
                    VX_SHAPE_DOMAIN_INDEX_NONE);
            sum = next;
        }
    } else {
        int64_t raw_count;
        if (vx_sd_number_integer(build, count_param, 0, &raw_count) ||
            raw_count <= 0 || (uint64_t)raw_count > UINT32_MAX)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE,
                              VX_SHAPE_DOMAIN_INDEX_NONE);
        output_count = (uint32_t)raw_count;
        if (output_count != build->request->declared_output_count ||
            axis_extent % output_count)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                              output_count,
                              build->request->declared_output_count);
        common_size = axis_extent / output_count;
        sum = axis_extent;
    }
    if (sum != axis_extent)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          (uint32_t)(sum > UINT32_MAX ? UINT32_MAX : sum),
                          axis);
    {
        uint64_t quantization_offset = 0u;
        for (uint32_t index = 0u; index < output_count; ++index) {
            VxShapeDomainNamedTensor* target = NULL;
            VxShapeDomainDimension* dimensions;
            VxShapeDomainQuantization quantization =
                input->descriptor.quantization;
            uint64_t size = common_size;
            if (split) {
                int64_t parsed;
                if (vx_sd_param_array_integer(split, index, &parsed))
                    return vx_sd_fail(build,
                        VX_SHAPE_DOMAIN_ERROR_INTERNAL, index, axis);
                size = (uint64_t)parsed;
            }
            for (uint32_t candidate = 0u;
                 candidate < build->result->output_count; ++candidate) {
                uint32_t suffix;
                if (!vx_sd_port_suffix(&build->result->outputs[candidate].name,
                                       "out", &suffix) && suffix == index) {
                    target = &build->result->outputs[candidate];
                    break;
                }
            }
            if (!target || target->descriptor.rank != input->descriptor.rank)
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                                  index, input->descriptor.rank);
            dimensions = (VxShapeDomainDimension*)
                target->descriptor.dimensions;
            for (uint32_t dimension = 0u;
                 dimension < input->descriptor.rank; ++dimension)
                dimensions[dimension] = input->descriptor.dimensions[dimension];
            dimensions[axis] = vx_sd_fixed(size);
            if (quantization.scheme == 2 && quantization.axis == axis) {
                if (quantization_offset + size > quantization.count)
                    return vx_sd_fail(build,
                        VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION,
                        index, axis);
                quantization.count = (uint32_t)size;
                quantization.scales += (uint32_t)quantization_offset;
                quantization.zero_points += (uint32_t)quantization_offset;
                quantization_offset += size;
            }
            if (vx_sd_set_variadic_output(build, "out", index,
                    input->descriptor.dtype, dimensions,
                    input->descriptor.rank, &quantization))
                return VX_SHAPE_DOMAIN_STATUS_INVALID;
        }
    }
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_slice_parameters(
        VxShapeDomainBuild* build,
        uint32_t rank,
        uint32_t* count,
        uint32_t* axes,
        int64_t* starts,
        int64_t* ends,
        uint64_t* steps) {
    static const char* const ALLOWED[] = {"starts", "ends", "axes", "steps"};
    const VxShapeDomainParam* start_param;
    const VxShapeDomainParam* end_param;
    const VxShapeDomainParam* axis_param;
    const VxShapeDomainParam* step_param;
    if (vx_sd_allowed_params(build, ALLOWED, 4u)) return -1;
    start_param = vx_sd_param(build->request, "starts");
    end_param = vx_sd_param(build->request, "ends");
    axis_param = vx_sd_param(build->request, "axes");
    step_param = vx_sd_param(build->request, "steps");
    if (!start_param || !end_param ||
        start_param->kind != VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY ||
        end_param->kind != VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY ||
        start_param->value_count != end_param->value_count ||
        start_param->value_count > rank ||
        (start_param->value_count &&
         (!start_param->values || !end_param->values))) return -1;
    *count = start_param->value_count;
    if (axis_param) {
        uint32_t axis_count;
        if (axis_param->value_count != *count ||
            vx_sd_normalize_axes(build, axis_param, rank, axes, &axis_count) ||
            axis_count != *count) return -1;
    } else {
        for (uint32_t index = 0u; index < *count; ++index) axes[index] = index;
    }
    if (step_param &&
        (step_param->kind != VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY ||
         step_param->value_count != *count ||
         (*count && !step_param->values))) return -1;
    for (uint32_t index = 0u; index < *count; ++index) {
        int64_t step = 1;
        if (vx_sd_param_array_integer(start_param, index, &starts[index]) ||
            vx_sd_param_array_integer(end_param, index, &ends[index]) ||
            (step_param &&
             vx_sd_param_array_integer(step_param, index, &step)) ||
            step <= 0) return -1;
        steps[index] = (uint64_t)step;
    }
    return 0;
}

static void vx_sd_slice_coordinate(int64_t raw,
                                   uint64_t extent,
                                   uint64_t* output) {
    int64_t translated = raw;
    if (translated < 0) translated += (int64_t)extent;
    if (translated < 0) *output = 0u;
    else if ((uint64_t)translated > extent) *output = extent;
    else *output = (uint64_t)translated;
}

static int32_t vx_sd_slice_quantization(
        VxShapeDomainBuild* build,
        const VxShapeDomainQuantization* input,
        const uint64_t* starts,
        const uint64_t* ends,
        const uint64_t* steps,
        VxShapeDomainQuantization* output) {
    uint32_t axis;
    uint64_t count;
    *output = *input;
    if (input->scheme != 2) return VX_SHAPE_DOMAIN_STATUS_OK;
    axis = input->axis;
    if (axis >= 8u || ends[axis] > input->count ||
        starts[axis] >= ends[axis] || !steps[axis])
        return vx_sd_fail(build,
            VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u, axis);
    count = (ends[axis] - starts[axis] - 1u) / steps[axis] + 1u;
    if (steps[axis] == 1u) {
        output->count = (uint32_t)count;
        output->scales += (uint32_t)starts[axis];
        output->zero_points += (uint32_t)starts[axis];
        return VX_SHAPE_DOMAIN_STATUS_OK;
    }
    {
        uint32_t* scales = (uint32_t*)vx_sd_allocate(build->arena,
            (uint32_t)count, (uint32_t)sizeof(*scales), 4u);
        int32_t* zero_points = (int32_t*)vx_sd_allocate(build->arena,
            (uint32_t)count, (uint32_t)sizeof(*zero_points), 4u);
        if (!scales || !zero_points)
            return VX_SHAPE_DOMAIN_STATUS_SCRATCH_TOO_SMALL;
        for (uint32_t index = 0u; index < (uint32_t)count; ++index) {
            uint32_t source = (uint32_t)(starts[axis] + index * steps[axis]);
            scales[index] = input->scales[source];
            zero_points[index] = input->zero_points[source];
        }
        output->count = (uint32_t)count;
        output->scales = scales;
        output->zero_points = zero_points;
    }
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_struct_slice(VxShapeDomainBuild* build) {
    const VxShapeDomainNamedTensor* input;
    VxShapeDomainDimension dimensions[8];
    VxShapeDomainQuantization quantization;
    uint32_t axes[8];
    int64_t raw_starts[8];
    int64_t raw_ends[8];
    uint64_t raw_steps[8];
    uint64_t starts[8];
    uint64_t ends[8];
    uint64_t steps[8];
    uint32_t count;
    int32_t status;
    if (vx_sd_struct_single_ports(build))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (!input->descriptor.rank || input->descriptor.rank > 8u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, input->descriptor.rank);
    if (vx_sd_slice_parameters(build, input->descriptor.rank, &count, axes,
            raw_starts, raw_ends, raw_steps))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    for (uint32_t axis = 0u; axis < input->descriptor.rank; ++axis) {
        uint64_t fixed;
        dimensions[axis] = input->descriptor.dimensions[axis];
        starts[axis] = 0u;
        steps[axis] = 1u;
        if (vx_shape_domain_fixed_value(&input->descriptor.dimensions[axis],
                build->request->environment,
                build->request->environment_count, &fixed)) {
            ends[axis] = fixed;
        } else {
            const VxShapeDomainDimension* dimension =
                &input->descriptor.dimensions[axis];
            if (dimension->kind != VX_SHAPE_DOMAIN_DIMENSION_SYMBOL)
                return vx_sd_unsupported(build);
            ends[axis] = build->request->environment[
                dimension->symbol_index].maximum;
        }
    }
    for (uint32_t index = 0u; index < count; ++index) {
        uint32_t axis = axes[index];
        uint64_t extent;
        if (vx_shape_domain_fixed_value(&input->descriptor.dimensions[axis],
                build->request->environment,
                build->request->environment_count, &extent)) {
            uint64_t start;
            uint64_t end;
            uint64_t length;
            vx_sd_slice_coordinate(raw_starts[index], extent, &start);
            vx_sd_slice_coordinate(raw_ends[index], extent, &end);
            if (end <= start)
                return vx_sd_fail(build,
                    VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH, index, axis);
            length = (end - start - 1u) / raw_steps[index] + 1u;
            dimensions[axis] = vx_sd_fixed(length);
            starts[axis] = start;
            ends[axis] = end;
            steps[axis] = raw_steps[index];
        } else {
            const VxShapeDomainDimension* dimension =
                &input->descriptor.dimensions[axis];
            const VxShapeDomainConstraint* constraint =
                &build->request->environment[dimension->symbol_index];
            if (raw_starts[index] != 0 || raw_steps[index] != 1u ||
                raw_ends[index] < 0 ||
                (uint64_t)raw_ends[index] < constraint->maximum)
                return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                                  0u, axis);
            starts[axis] = 0u;
            ends[axis] = constraint->maximum;
            steps[axis] = 1u;
        }
    }
    status = vx_sd_slice_quantization(build, &input->descriptor.quantization,
                                      starts, ends, steps, &quantization);
    if (status != VX_SHAPE_DOMAIN_STATUS_OK) return status;
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", input->descriptor.dtype,
                            dimensions, input->descriptor.rank,
                            &quantization);
}

static int32_t vx_sd_struct_pad(VxShapeDomainBuild* build) {
    static const char* const ALLOWED[] = {"pads", "value"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainParam* pads;
    const VxShapeDomainParam* value;
    VxShapeDomainDimension dimensions[8];
    VxShapeDomainQuantization quantization;
    uint32_t rank;
    if (vx_sd_struct_single_ports(build) ||
        vx_sd_allowed_params(build, ALLOWED, 2u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    rank = input->descriptor.rank;
    if (!rank || rank > 8u)
        return vx_sd_unsupported(build);
    pads = vx_sd_param(build->request, "pads");
    value = vx_sd_param(build->request, "value");
    if (!pads || pads->kind != VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY ||
        pads->value_count != rank * 2u || !pads->values ||
        (value && (value->kind != VX_SHAPE_DOMAIN_PARAM_NUMBER ||
                   !isfinite(value->number))))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    quantization = input->descriptor.quantization;
    for (uint32_t axis = 0u; axis < rank; ++axis) {
        int64_t before;
        int64_t after;
        uint64_t amount;
        uint64_t fixed;
        if (vx_sd_param_array_integer(pads, axis, &before) ||
            vx_sd_param_array_integer(pads, rank + axis, &after) ||
            before < 0 || after < 0 ||
            vx_sd_checked_add((uint64_t)before, (uint64_t)after, &amount))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              VX_SHAPE_DOMAIN_INDEX_NONE, axis);
        if (quantization.scheme == 2 && quantization.axis == axis && amount)
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u, axis);
        if (!amount) dimensions[axis] = input->descriptor.dimensions[axis];
        else if (!vx_shape_domain_fixed_value(
                    &input->descriptor.dimensions[axis],
                    build->request->environment,
                    build->request->environment_count, &fixed) ||
                 vx_sd_checked_add(fixed, amount, &fixed))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                              0u, axis);
        else dimensions[axis] = vx_sd_fixed(fixed);
    }
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", input->descriptor.dtype,
                            dimensions, rank, &quantization);
}

static int32_t vx_sd_assert_indices(VxShapeDomainBuild* build,
                                    const VxShapeDomainNamedTensor* indices,
                                    uint32_t input_index) {
    if (indices->descriptor.dtype != VX_DTYPE_I32)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                          input_index, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (!vx_sd_unquantized(&indices->descriptor))
        return vx_sd_fail(build,
            VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, input_index,
            VX_SHAPE_DOMAIN_INDEX_NONE);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_struct_gather(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input", "indices"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const AXIS[] = {"axis"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* indices;
    VxShapeDomainDimension* dimensions;
    VxShapeDomainQuantization quantization;
    uint32_t output_rank;
    uint32_t axis;
    uint32_t cursor = 0u;
    if (vx_sd_exact_ports(build, INPUTS, 2u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, AXIS, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    indices = vx_sd_input(build->request, "indices");
    if (!input->descriptor.rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, 0u);
    if (vx_sd_assert_indices(build, indices, 1u) ||
        vx_sd_axis(build, "axis", input->descriptor.rank, 0, &axis))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (indices->descriptor.rank > UINT32_MAX - input->descriptor.rank + 1u)
        return vx_sd_unsupported(build);
    output_rank = input->descriptor.rank - 1u + indices->descriptor.rank;
    dimensions = vx_sd_output_dimensions(build, "out");
    if ((!dimensions && output_rank) ||
        build->result->outputs[0].descriptor.rank != output_rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, output_rank);
    for (uint32_t input_axis = 0u; input_axis < axis; ++input_axis)
        dimensions[cursor++] = input->descriptor.dimensions[input_axis];
    for (uint32_t index_axis = 0u;
         index_axis < indices->descriptor.rank; ++index_axis)
        dimensions[cursor++] = indices->descriptor.dimensions[index_axis];
    for (uint32_t input_axis = axis + 1u;
         input_axis < input->descriptor.rank; ++input_axis)
        dimensions[cursor++] = input->descriptor.dimensions[input_axis];
    quantization = input->descriptor.quantization;
    if (quantization.scheme == 2) {
        if (quantization.axis == axis)
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u, axis);
        if (quantization.axis > axis) {
            if (!indices->descriptor.rank) quantization.axis--;
            else quantization.axis += indices->descriptor.rank - 1u;
        }
    }
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", input->descriptor.dtype,
                            dimensions, output_rank, &quantization);
}

static int vx_sd_declared_dimension_bounds(
        const VxShapeDomainDimension* dimension,
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count,
        uint64_t* minimum,
        uint64_t* maximum) {
    if (!vx_sd_dimension_valid(dimension, environment, environment_count))
        return -1;
    if (dimension->kind == VX_SHAPE_DOMAIN_DIMENSION_FIXED) {
        *minimum = (uint64_t)dimension->value;
        *maximum = *minimum;
        return 0;
    }
    *minimum = environment[dimension->symbol_index].minimum;
    *maximum = environment[dimension->symbol_index].maximum;
    if (dimension->kind == VX_SHAPE_DOMAIN_DIMENSION_AFFINE &&
        (vx_sd_add_signed(*minimum, dimension->value, minimum) ||
         vx_sd_add_signed(*maximum, dimension->value, maximum)))
        return -1;
    return 0;
}

static int32_t vx_sd_struct_gather_elements(VxShapeDomainBuild* build) {
    static const char* const INPUTS[] = {"input", "indices"};
    static const char* const OUTPUTS[] = {"out"};
    static const char* const AXIS[] = {"axis"};
    const VxShapeDomainNamedTensor* input;
    const VxShapeDomainNamedTensor* indices;
    VxShapeDomainQuantization quantization;
    uint32_t axis;
    if (vx_sd_exact_ports(build, INPUTS, 2u, OUTPUTS, 1u) ||
        vx_sd_allowed_params(build, AXIS, 1u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    indices = vx_sd_input(build->request, "indices");
    if (!input->descriptor.rank ||
        indices->descriptor.rank != input->descriptor.rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          1u, indices->descriptor.rank);
    if (vx_sd_assert_indices(build, indices, 1u) ||
        vx_sd_axis(build, "axis", input->descriptor.rank, 0, &axis))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    for (uint32_t dimension = 0u;
         dimension < input->descriptor.rank; ++dimension) {
        uint64_t input_minimum;
        uint64_t input_maximum;
        uint64_t indices_minimum;
        uint64_t indices_maximum;
        if (dimension == axis || vx_shape_domain_dimensions_equal(
                &indices->descriptor.dimensions[dimension],
                &input->descriptor.dimensions[dimension],
                build->request->environment,
                build->request->environment_count)) continue;
        if (vx_sd_declared_dimension_bounds(
                &input->descriptor.dimensions[dimension],
                build->request->environment,
                build->request->environment_count,
                &input_minimum, &input_maximum) ||
            vx_sd_declared_dimension_bounds(
                &indices->descriptor.dimensions[dimension],
                build->request->environment,
                build->request->environment_count,
                &indices_minimum, &indices_maximum) ||
            indices_maximum > input_minimum)
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_UNPROVABLE,
                              1u, dimension);
        (void)input_maximum;
        (void)indices_minimum;
    }
    quantization = input->descriptor.quantization;
    if (quantization.scheme == 2) {
        uint64_t extent;
        if (quantization.axis == axis)
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 0u, axis);
        if (!vx_shape_domain_fixed_value(
                &indices->descriptor.dimensions[quantization.axis],
                build->request->environment,
                build->request->environment_count, &extent) ||
            extent > quantization.count)
            return vx_sd_fail(build,
                VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION, 1u,
                quantization.axis);
        quantization.count = (uint32_t)extent;
    }
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", input->descriptor.dtype,
                            indices->descriptor.dimensions,
                            indices->descriptor.rank, &quantization);
}

static int32_t vx_sd_struct_boolean_param(VxShapeDomainBuild* build,
                                          const char* name,
                                          int default_value,
                                          int* output) {
    const VxShapeDomainParam* param = vx_sd_param(build->request, name);
    if (!output) return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INTERNAL,
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

static int32_t vx_sd_struct_reduction(VxShapeDomainBuild* build) {
    static const char* const ALLOWED[] = {"axis", "keepdims"};
    const VxShapeDomainNamedTensor* input;
    VxShapeDomainDimension* dimensions;
    uint32_t axis;
    uint32_t output_rank;
    int keepdims;
    if (vx_sd_struct_single_ports(build) ||
        vx_sd_allowed_params(build, ALLOWED, 2u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (vx_sd_assert_float(build, &input->descriptor, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!input->descriptor.rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, 0u);
    if (vx_sd_axis(build, "axis", input->descriptor.rank, -1, &axis) ||
        vx_sd_struct_boolean_param(build, "keepdims", 1, &keepdims))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (axis != input->descriptor.rank - 1u)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE, axis);
    output_rank = keepdims
        ? input->descriptor.rank : input->descriptor.rank - 1u;
    dimensions = vx_sd_output_dimensions(build, "out");
    if (!dimensions && output_rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, output_rank);
    if (build->result->outputs[0].descriptor.rank != output_rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, output_rank);
    for (uint32_t index = 0u; index < output_rank; ++index)
        dimensions[index] = keepdims && index == output_rank - 1u
            ? vx_sd_fixed(1u) : input->descriptor.dimensions[index];
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", VX_DTYPE_F32, dimensions,
                            output_rank, NULL);
}

static int32_t vx_sd_struct_argmax(VxShapeDomainBuild* build) {
    static const char* const ALLOWED[] = {
        "axis", "keepdims", "select_last_index"
    };
    const VxShapeDomainNamedTensor* input;
    VxShapeDomainDimension* dimensions;
    uint32_t axis;
    uint32_t output_rank;
    uint32_t cursor = 0u;
    int keepdims;
    int64_t select_last = 0;
    if (vx_sd_struct_single_ports(build) ||
        vx_sd_allowed_params(build, ALLOWED, 3u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    input = vx_sd_input(build->request, "input");
    if (vx_sd_assert_float(build, &input->descriptor, 0u))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (!input->descriptor.rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          0u, 0u);
    if (vx_sd_axis(build, "axis", input->descriptor.rank, 0, &axis) ||
        vx_sd_struct_boolean_param(build, "keepdims", 1, &keepdims) ||
        vx_sd_number_integer(build,
            vx_sd_param(build->request, "select_last_index"),
            0, &select_last))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if (select_last != 0)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    output_rank = keepdims
        ? input->descriptor.rank : input->descriptor.rank - 1u;
    dimensions = vx_sd_output_dimensions(build, "out");
    if ((!dimensions && output_rank) ||
        build->result->outputs[0].descriptor.rank != output_rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          0u, output_rank);
    for (uint32_t input_axis = 0u;
         input_axis < input->descriptor.rank; ++input_axis) {
        if (input_axis == axis) {
            if (keepdims) dimensions[cursor++] = vx_sd_fixed(1u);
        } else {
            dimensions[cursor++] = input->descriptor.dimensions[input_axis];
        }
    }
    build->result->fact_kind = VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL;
    return vx_sd_set_output(build, "out", VX_DTYPE_I32, dimensions,
                            output_rank, NULL);
}

static int32_t vx_sd_prove_structural(VxShapeDomainBuild* build) {
    switch (build->request->operator_kind) {
        case VX_OP_REDUCE_SUM:
        case VX_OP_REDUCE_MEAN:
            return vx_sd_struct_reduction(build);
        case VX_OP_ARG_MAX:
            return vx_sd_struct_argmax(build);
        case VX_OP_SUB:
        case VX_OP_DIV:
            return vx_sd_struct_broadcast_binary(build, 0);
        case VX_OP_EQUAL:
        case VX_OP_GREATER_OR_EQUAL:
            return vx_sd_struct_broadcast_binary(build, 1);
        case VX_OP_WHERE:
            return vx_sd_struct_where(build);
        case VX_OP_TRANSPOSE:
            return vx_sd_struct_transpose(build);
        case VX_OP_FLATTEN:
            return vx_sd_struct_flatten(build);
        case VX_OP_SQUEEZE:
            return vx_sd_struct_squeeze(build);
        case VX_OP_UNSQUEEZE:
            return vx_sd_struct_unsqueeze(build);
        case VX_OP_RESHAPE:
            return vx_sd_struct_reshape_or_expand(build, 0);
        case VX_OP_EXPAND:
        case VX_OP_BROADCAST:
            return vx_sd_struct_reshape_or_expand(build, 1);
        case VX_OP_CONCAT:
            return vx_sd_struct_concat(build);
        case VX_OP_SPLIT:
            return vx_sd_struct_split(build);
        case VX_OP_SLICE:
            return vx_sd_struct_slice(build);
        case VX_OP_PAD:
            return vx_sd_struct_pad(build);
        case VX_OP_GATHER:
            return vx_sd_struct_gather(build);
        case VX_OP_GATHER_ELEMENTS:
            return vx_sd_struct_gather_elements(build);
        default:
            return VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED;
    }
}
