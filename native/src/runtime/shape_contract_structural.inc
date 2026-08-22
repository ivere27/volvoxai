/* Shape contracts for the broadcasting and structural operators: elementwise
 * broadcast arithmetic and comparison, Where, reductions, ArgMax, and the
 * shape movers (Transpose, Flatten, Squeeze, Unsqueeze, Reshape, Expand,
 * Concat, Split, Slice, Pad, Gather, GatherElements). */

static int vx_shape_structural_kind(VxOperatorKind kind) {
    switch (kind) {
        case VX_OP_SUB:
        case VX_OP_DIV:
        case VX_OP_EQUAL:
        case VX_OP_GREATER_OR_EQUAL:
        case VX_OP_WHERE:
        case VX_OP_REDUCE_SUM:
        case VX_OP_REDUCE_MEAN:
        case VX_OP_ARG_MAX:
        case VX_OP_TRANSPOSE:
        case VX_OP_FLATTEN:
        case VX_OP_SQUEEZE:
        case VX_OP_UNSQUEEZE:
        case VX_OP_RESHAPE:
        case VX_OP_EXPAND:
        case VX_OP_CONCAT:
        case VX_OP_SPLIT:
        case VX_OP_SLICE:
        case VX_OP_PAD:
        case VX_OP_GATHER:
        case VX_OP_GATHER_ELEMENTS:
            return 1;
        default:
            return 0;
    }
}

static int vx_shape_infer_broadcast_arithmetic(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* left = vx_shape_input_descriptor(request, "a");
    const VxShapeTensorDescriptor* right = vx_shape_input_descriptor(request, "b");
    uint64_t output_shape[8];
    size_t output_rank;
    if (vx_shape_float_tensor(left, "operator input 'a'", error) ||
        vx_shape_float_tensor(right, "operator input 'b'", error) ||
        vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_broadcast(left, right, output_shape, &output_rank, error)) {
        return -1;
    }
    return vx_shape_clone_output(candidate, output_shape, output_rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_broadcast_comparison(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* left = vx_shape_input_descriptor(request, "a");
    const VxShapeTensorDescriptor* right = vx_shape_input_descriptor(request, "b");
    uint64_t output_shape[8];
    size_t output_rank;
    if (left->dtype != VX_DTYPE_I32 || right->dtype != VX_DTYPE_I32) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator inputs",
                             "comparison inputs must both be int32.");
    }
    if (vx_shape_unquantized(left, "operator input 'a'", error) ||
        vx_shape_unquantized(right, "operator input 'b'", error) ||
        vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_broadcast(left, right, output_shape, &output_rank, error)) {
        return -1;
    }
    return vx_shape_clone_output(candidate, output_shape, output_rank,
                                 VX_DTYPE_I32, NULL, error);
}

static int vx_shape_infer_where(const VxConcreteShapeRequest* request,
                                VxConcreteShapeResult* candidate,
                                VxShapeContractError* error) {
    const VxShapeTensorDescriptor* condition =
        vx_shape_input_descriptor(request, "condition");
    const VxShapeTensorDescriptor* left = vx_shape_input_descriptor(request, "a");
    const VxShapeTensorDescriptor* right = vx_shape_input_descriptor(request, "b");
    VxShapeTensorDescriptor data_shape;
    uint64_t data_dimensions[8];
    uint64_t output_shape[8];
    size_t data_rank;
    size_t output_rank;
    if (condition->dtype != VX_DTYPE_F32 && condition->dtype != VX_DTYPE_I32) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator input 'condition'.dtype",
                             "must be float32 or int32.");
    }
    if (vx_shape_unquantized(condition, "operator input 'condition'", error)) return -1;
    if (left->dtype != right->dtype) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator inputs 'a' and 'b'",
                             "must have the same dtype.");
    }
    if (left->dtype != VX_DTYPE_F32 && left->dtype != VX_DTYPE_I32) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator data inputs",
                             "must be float32 or int32.");
    }
    if (vx_shape_unquantized(left, "operator input 'a'", error) ||
        vx_shape_unquantized(right, "operator input 'b'", error) ||
        vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_broadcast(left, right, data_dimensions, &data_rank, error)) {
        return -1;
    }
    memset(&data_shape, 0, sizeof(data_shape));
    data_shape.rank = data_rank;
    data_shape.shape = data_dimensions;
    data_shape.dtype = left->dtype;
    if (vx_shape_broadcast(condition, &data_shape, output_shape,
                           &output_rank, error)) return -1;
    return vx_shape_clone_output(candidate, output_shape, output_rank,
                                 left->dtype, NULL, error);
}

static int vx_shape_infer_reduction(VxOperatorKind kind,
                                    const VxConcreteShapeRequest* request,
                                    VxConcreteShapeResult* candidate,
                                    VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis", "keepdims"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    uint64_t output_shape[8];
    size_t axis;
    size_t output_rank;
    int keepdims;
    (void)kind;
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank(input, 1, 0, 0, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 2, error) ||
        vx_shape_normalize_axis(request, "axis", input->rank, -1, 0,
                                &axis, error) ||
        vx_shape_boolean_default(request, "keepdims", 1, &keepdims, error)) {
        return -1;
    }
    if (axis != input->rank - 1) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.axis",
                             "must resolve to the last axis.");
    }
    output_rank = keepdims ? input->rank : input->rank - 1;
    if (output_rank) memcpy(output_shape, input->shape,
                            output_rank * sizeof(*output_shape));
    if (keepdims) output_shape[output_rank - 1] = 1;
    return vx_shape_clone_output(candidate, output_shape, output_rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_arg_max(const VxConcreteShapeRequest* request,
                                  VxConcreteShapeResult* candidate,
                                  VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis", "keepdims", "select_last_index"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeParam* select_last;
    uint64_t output_shape[8];
    size_t axis;
    size_t input_axis;
    size_t output_rank = 0;
    int keepdims;
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank(input, 1, 0, 0, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 3, error) ||
        vx_shape_normalize_axis(request, "axis", input->rank, 0, 0,
                                &axis, error) ||
        vx_shape_boolean_default(request, "keepdims", 1, &keepdims, error)) {
        return -1;
    }
    select_last = vx_shape_find_param(request, "select_last_index");
    if (select_last &&
        (select_last->kind != VX_SHAPE_PARAM_NUMBER ||
         select_last->value.number != 0.0)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.select_last_index",
                             "must be 0 (first-index ties).");
    }
    for (input_axis = 0; input_axis < input->rank; ++input_axis) {
        if (input_axis == axis) {
            if (keepdims) output_shape[output_rank++] = 1;
        } else {
            output_shape[output_rank++] = input->shape[input_axis];
        }
    }
    return vx_shape_clone_output(candidate, output_shape, output_rank,
                                 VX_DTYPE_I32, NULL, error);
}

static int vx_shape_infer_transpose(const VxConcreteShapeRequest* request,
                                    VxConcreteShapeResult* candidate,
                                    VxShapeContractError* error) {
    static const char* const FIELDS[] = {"perm"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const double* raw = NULL;
    size_t count = 0;
    uint64_t output_shape[8];
    size_t perm[8];
    size_t axis;
    VxShapeQuantization quantization = input->quantization;
    if (vx_shape_rank(input, 1, 0, 0, "operator input 'input'.shape", error) ||
        input->rank > 8 ||
        vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_param_number_array(request, "perm", 0, &raw, &count, error)) {
        if (input->rank > 8 && error && error->code == VX_SHAPE_CONTRACT_ERROR_NONE) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                                 "operator input 'input'.shape",
                                 "must have rank at most 8.");
        }
        return -1;
    }
    if (!raw) count = input->rank;
    if (count != input->rank) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.perm", "must be a permutation.");
    }
    for (axis = 0; axis < input->rank; ++axis) {
        size_t other;
        int64_t value = raw ? (int64_t)raw[axis]
                            : (int64_t)(input->rank - axis - 1);
        if (raw && (!vx_shape_is_safe_integer(raw[axis]) || value < 0 ||
                    (size_t)value >= input->rank)) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.perm", "must be a permutation.");
        }
        perm[axis] = (size_t)value;
        for (other = 0; other < axis; ++other) {
            if (perm[other] == perm[axis]) {
                return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                     "operator params.perm", "must be a permutation.");
            }
        }
        output_shape[axis] = input->shape[perm[axis]];
    }
    if (quantization.scheme == VX_SHAPE_QUANTIZATION_PER_AXIS) {
        for (axis = 0; axis < input->rank; ++axis) {
            if (perm[axis] == quantization.axis) {
                quantization.axis = axis;
                break;
            }
        }
    }
    return vx_shape_clone_output(candidate, output_shape, input->rank,
                                 input->dtype, &quantization, error);
}

static int vx_shape_infer_flatten(const VxConcreteShapeRequest* request,
                                  VxConcreteShapeResult* candidate,
                                  VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    uint64_t output_shape[2];
    size_t axis;
    VxShapeQuantization quantization;
    if (vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_normalize_axis(request, "axis", input->rank, 1, 1,
                                &axis, error) ||
        vx_shape_product_range(input->shape, 0, axis,
                               "Flatten prefix product", &output_shape[0], error) ||
        vx_shape_product_range(input->shape, axis, input->rank,
                               "Flatten suffix product", &output_shape[1], error) ||
        vx_shape_reshape_quantization(input, output_shape, 2,
                                      &quantization, error)) return -1;
    return vx_shape_clone_output(candidate, output_shape, 2, input->dtype,
                                 &quantization, error);
}

static int vx_shape_normalize_axes(const double* raw,
                                   size_t count,
                                   size_t normalization_rank,
                                   const char* path,
                                   size_t* axes,
                                   VxShapeContractError* error) {
    size_t index;
    if (!raw || !count) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             path, "must be a non-empty array.");
    }
    for (index = 0; index < count; ++index) {
        int64_t value = 0;
        size_t other;
        char item_path[224];
        snprintf(item_path, sizeof(item_path), "%s[%zu]", path, index);
        if (vx_shape_param_signed_integer_value(raw[index], item_path,
                                                &value, error)) return -1;
        if (value < 0) value += (int64_t)normalization_rank;
        if (value < 0 || (size_t)value >= normalization_rank) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 item_path, "resolves outside the supported rank.");
        }
        axes[index] = (size_t)value;
        for (other = 0; other < index; ++other) {
            if (axes[other] == axes[index]) {
                return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                     path, "must contain unique axes.");
            }
        }
    }
    return 0;
}

static int vx_shape_infer_squeeze(const VxConcreteShapeRequest* request,
                                  VxConcreteShapeResult* candidate,
                                  VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axes"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const double* raw;
    size_t count;
    size_t* axes = NULL;
    uint64_t* output_shape = NULL;
    size_t output_rank;
    size_t input_axis;
    size_t output_axis = 0;
    VxShapeQuantization quantization;
    int status = -1;
    if (vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_param_number_array(request, "axes", 1, &raw, &count, error)) {
        return -1;
    }
    if (count > input->rank || count > SIZE_MAX / sizeof(*axes)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.axes", "contains too many axes.");
    }
    axes = (size_t*)malloc(count * sizeof(*axes));
    if (!axes) return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                    "operator params.axes", "allocation failed.");
    if (vx_shape_normalize_axes(raw, count, input->rank,
                                "operator params.axes", axes, error)) goto cleanup;
    output_rank = input->rank - count;
    if (output_rank) {
        output_shape = (uint64_t*)malloc(output_rank * sizeof(*output_shape));
        if (!output_shape) {
            vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                          "operator output 'out'.shape", "allocation failed.");
            goto cleanup;
        }
    }
    for (input_axis = 0; input_axis < input->rank; ++input_axis) {
        size_t index;
        int selected = 0;
        for (index = 0; index < count; ++index) selected |= axes[index] == input_axis;
        if (selected) {
            if (input->shape[input_axis] != 1) {
                char path[224];
                snprintf(path, sizeof(path), "operator input 'input'.shape[%zu]",
                         input_axis);
                vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                              path, "must be 1 to squeeze it.");
                goto cleanup;
            }
        } else {
            output_shape[output_axis++] = input->shape[input_axis];
        }
    }
    if (vx_shape_reshape_quantization(input, output_shape, output_rank,
                                      &quantization, error) ||
        vx_shape_clone_output(candidate, output_shape, output_rank,
                              input->dtype, &quantization, error)) goto cleanup;
    status = 0;
cleanup:
    free(output_shape);
    free(axes);
    return status;
}

static int vx_shape_infer_unsqueeze(const VxConcreteShapeRequest* request,
                                    VxConcreteShapeResult* candidate,
                                    VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axes"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const double* raw;
    size_t count;
    size_t output_rank;
    size_t* axes = NULL;
    uint64_t* output_shape = NULL;
    size_t output_axis;
    size_t input_axis = 0;
    VxShapeQuantization quantization;
    int status = -1;
    if (vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_param_number_array(request, "axes", 1, &raw, &count, error) ||
        input->rank > SIZE_MAX - count) return -1;
    output_rank = input->rank + count;
    if (count > SIZE_MAX / sizeof(*axes) ||
        output_rank > SIZE_MAX / sizeof(*output_shape)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator params.axes", "rank is too large.");
    }
    axes = (size_t*)malloc(count * sizeof(*axes));
    output_shape = (uint64_t*)malloc(output_rank * sizeof(*output_shape));
    if (!axes || !output_shape) {
        vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                      "operator output 'out'.shape", "allocation failed.");
        goto cleanup;
    }
    if (vx_shape_normalize_axes(raw, count, output_rank,
                                "operator params.axes", axes, error)) goto cleanup;
    for (output_axis = 0; output_axis < output_rank; ++output_axis) {
        size_t index;
        int inserted = 0;
        for (index = 0; index < count; ++index) inserted |= axes[index] == output_axis;
        output_shape[output_axis] = inserted ? 1 : input->shape[input_axis++];
    }
    if (vx_shape_reshape_quantization(input, output_shape, output_rank,
                                      &quantization, error) ||
        vx_shape_clone_output(candidate, output_shape, output_rank,
                              input->dtype, &quantization, error)) goto cleanup;
    status = 0;
cleanup:
    free(output_shape);
    free(axes);
    return status;
}

static int vx_shape_target(const VxConcreteShapeRequest* request,
                           const VxShapeTensorDescriptor* input,
                           uint64_t** output_shape,
                           size_t* output_rank,
                           VxShapeContractError* error) {
    static const char* const FIELDS[] = {"shape"};
    const double* raw;
    size_t count;
    size_t axis;
    uint64_t* shape = NULL;
    const VxShapeTensorDescriptor* declared = NULL;
    if (vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_param_number_array(request, "shape", 1, &raw, &count, error) ||
        vx_shape_optional_declared_output(request, &declared, error)) return -1;
    if (count > SIZE_MAX / sizeof(*shape)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator params.shape", "rank is too large.");
    }
    if (declared && declared->dtype != input->dtype) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "declared output 'out'.dtype",
                             "must equal the input dtype.");
    }
    if (declared && declared->rank != count) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "declared output 'out'.shape",
                             "has a different rank from params.shape.");
    }
    if (count) {
        shape = (uint64_t*)malloc(count * sizeof(*shape));
        if (!shape) return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                         "operator params.shape", "allocation failed.");
    }
    for (axis = 0; axis < count; ++axis) {
        char path[224];
        snprintf(path, sizeof(path), "operator params.shape[%zu]", axis);
        if (vx_shape_param_integer_value(raw[axis], 1, path,
                                         &shape[axis], error)) {
            free(shape);
            return -1;
        }
        if (declared && declared->shape[axis] != shape[axis]) {
            snprintf(path, sizeof(path), "declared output 'out'.shape[%zu]", axis);
            free(shape);
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 path, "must equal params.shape.");
        }
    }
    *output_shape = shape;
    *output_rank = count;
    return 0;
}

static int vx_shape_infer_reshape(const VxConcreteShapeRequest* request,
                                  VxConcreteShapeResult* candidate,
                                  VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    uint64_t* output_shape = NULL;
    size_t output_rank = 0;
    uint64_t input_elements;
    uint64_t output_elements;
    VxShapeQuantization quantization;
    int status = -1;
    if (vx_shape_target(request, input, &output_shape, &output_rank, error) ||
        vx_shape_product_range(input->shape, 0, input->rank, "Reshape input",
                               &input_elements, error) ||
        vx_shape_product_range(output_shape, 0, output_rank, "Reshape output",
                               &output_elements, error)) goto cleanup;
    if (input_elements != output_elements) {
        vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                      "operator params.shape",
                      "must preserve the exact element count.");
        goto cleanup;
    }
    if (vx_shape_reshape_quantization(input, output_shape, output_rank,
                                      &quantization, error) ||
        vx_shape_clone_output(candidate, output_shape, output_rank,
                              input->dtype, &quantization, error)) goto cleanup;
    status = 0;
cleanup:
    free(output_shape);
    return status;
}

static int vx_shape_infer_expand(const VxConcreteShapeRequest* request,
                                 VxConcreteShapeResult* candidate,
                                 VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    uint64_t* output_shape = NULL;
    size_t output_rank = 0;
    size_t offset;
    size_t axis;
    VxShapeQuantization quantization = input->quantization;
    int status = -1;
    if (vx_shape_target(request, input, &output_shape, &output_rank, error)) return -1;
    if (output_rank < input->rank || output_rank > 8) {
        vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                      "operator params.shape",
                      "must have rank between input rank and 8.");
        goto cleanup;
    }
    offset = output_rank - input->rank;
    for (axis = 0; axis < output_rank; ++axis) {
        uint64_t input_dimension = axis < offset ? 1 : input->shape[axis - offset];
        if (input_dimension != 1 && input_dimension != output_shape[axis]) {
            char path[224];
            snprintf(path, sizeof(path), "operator params.shape[%zu]", axis);
            vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                          path, "is not a valid broadcast target.");
            goto cleanup;
        }
    }
    if (quantization.scheme == VX_SHAPE_QUANTIZATION_PER_AXIS) {
        size_t output_axis = quantization.axis + offset;
        if (input->shape[quantization.axis] != output_shape[output_axis]) {
            char path[224];
            snprintf(path, sizeof(path), "operator params.shape[%zu]", output_axis);
            vx_shape_fail(error,
                          VX_SHAPE_CONTRACT_ERROR_UNSAFE_QUANTIZATION_TRANSFORM,
                          path, "must not expand the per-axis quantization extent.");
            goto cleanup;
        }
        quantization.axis = output_axis;
    }
    if (vx_shape_clone_output(candidate, output_shape, output_rank,
                              input->dtype, &quantization, error)) goto cleanup;
    status = 0;
cleanup:
    free(output_shape);
    return status;
}

static int vx_shape_infer_concat(const VxConcreteShapeRequest* request,
                                 VxConcreteShapeResult* candidate,
                                 VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis"};
    const VxShapeTensorDescriptor* first = vx_shape_input_descriptor(request, "input0");
    uint64_t output_shape[8];
    size_t axis;
    size_t input_index;
    VxShapeQuantization quantization = first->quantization;
    float* scales = NULL;
    int32_t* zero_points = NULL;
    int status = -1;
    if (first->rank < 1 || first->rank > 8) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator inputs",
                             "must all have one equal rank in [1, 8].");
    }
    if (vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_normalize_axis(request, "axis", first->rank, 0, 0,
                                &axis, error)) return -1;
    memcpy(output_shape, first->shape, first->rank * sizeof(*output_shape));
    output_shape[axis] = 0;
    for (input_index = 0; input_index < request->input_count; ++input_index) {
        char name[64];
        const VxShapeTensorDescriptor* input;
        size_t dimension;
        snprintf(name, sizeof(name), "input%zu", input_index);
        input = vx_shape_input_descriptor(request, name);
        if (input->rank != first->rank) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                                 "operator inputs",
                                 "must all have one equal rank in [1, 8].");
        }
        if (input->dtype != first->dtype) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                                 "operator inputs", "must all have the same dtype.");
        }
        for (dimension = 0; dimension < first->rank; ++dimension) {
            if (dimension != axis && input->shape[dimension] != first->shape[dimension]) {
                char path[224];
                snprintf(path, sizeof(path),
                         "operator input 'input%zu'.shape[%zu]",
                         input_index, dimension);
                return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                     path, "must match input0.");
            }
        }
        if (vx_shape_checked_add_u64(output_shape[axis], input->shape[axis],
                                     "Concat axis sum", &output_shape[axis],
                                     error)) return -1;
    }
    if (quantization.scheme == VX_SHAPE_QUANTIZATION_NONE) {
        for (input_index = 1; input_index < request->input_count; ++input_index) {
            char name[64];
            snprintf(name, sizeof(name), "input%zu", input_index);
            if (vx_shape_input_descriptor(request, name)->quantization.scheme !=
                VX_SHAPE_QUANTIZATION_NONE) {
                return vx_shape_fail(error,
                                     VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                                     "operator inputs",
                                     "Concat inputs must have compatible affine metadata.");
            }
        }
    } else if (quantization.scheme == VX_SHAPE_QUANTIZATION_PER_TENSOR ||
               quantization.axis != axis) {
        for (input_index = 1; input_index < request->input_count; ++input_index) {
            char name[64];
            snprintf(name, sizeof(name), "input%zu", input_index);
            if (!vx_shape_quantization_equal(
                    &quantization,
                    &vx_shape_input_descriptor(request, name)->quantization)) {
                return vx_shape_fail(error,
                                     VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                                     "operator inputs",
                                     "Concat inputs must have identical affine metadata unless concatenating their common per-axis dimension.");
            }
        }
    } else {
        size_t count = (size_t)output_shape[axis];
        size_t offset = 0;
        if ((uint64_t)count != output_shape[axis] ||
            count > SIZE_MAX / sizeof(*scales) ||
            count > SIZE_MAX / sizeof(*zero_points)) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                 "operator inputs", "affine metadata is too large.");
        }
        scales = (float*)malloc(count * sizeof(*scales));
        zero_points = (int32_t*)malloc(count * sizeof(*zero_points));
        if (!scales || !zero_points) {
            vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                          "operator inputs", "allocation failed.");
            goto cleanup;
        }
        for (input_index = 0; input_index < request->input_count; ++input_index) {
            char name[64];
            const VxShapeQuantization* current;
            snprintf(name, sizeof(name), "input%zu", input_index);
            current = &vx_shape_input_descriptor(request, name)->quantization;
            if (current->scheme != VX_SHAPE_QUANTIZATION_PER_AXIS ||
                current->axis != axis) {
                vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                              "operator inputs", "have incompatible per-axis metadata.");
                goto cleanup;
            }
            memcpy(scales + offset, current->scales,
                   current->count * sizeof(*scales));
            memcpy(zero_points + offset, current->zero_points,
                   current->count * sizeof(*zero_points));
            offset += current->count;
        }
        quantization.count = count;
        quantization.scales = scales;
        quantization.zero_points = zero_points;
    }
    if (vx_shape_clone_output(candidate, output_shape, first->rank,
                              first->dtype, &quantization, error)) goto cleanup;
    status = 0;
cleanup:
    free(zero_points);
    free(scales);
    return status;
}

static int vx_shape_infer_split(const VxConcreteShapeRequest* request,
                                VxConcreteShapeResult* candidate,
                                VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis", "split", "num_outputs"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeParam* split = vx_shape_find_param(request, "split");
    const VxShapeParam* count_param = vx_shape_find_param(request, "num_outputs");
    const double* raw = NULL;
    size_t count = 0;
    size_t axis;
    uint64_t* sizes = NULL;
    uint64_t sum = 0;
    size_t index;
    size_t offset = 0;
    int status = -1;
    if (vx_shape_rank(input, 1, 0, 0, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 3, error) ||
        vx_shape_normalize_axis(request, "axis", input->rank, 0, 0,
                                &axis, error)) return -1;
    if (split && count_param) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params",
                             "must specify split or num_outputs, not both.");
    }
    if (split) {
        if (vx_shape_param_number_array(request, "split", 1,
                                        &raw, &count, error)) return -1;
        if (!count) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.split",
                                 "must be a non-empty array.");
        }
    } else {
        double raw_count = 0.0;
        if (!count_param ||
            vx_shape_number(count_param, "operator params.num_outputs",
                            &raw_count, error) ||
            !vx_shape_is_safe_integer(raw_count) || raw_count <= 0.0 ||
            raw_count > (double)SIZE_MAX) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.num_outputs",
                                 "must be a positive safe integer.");
        }
        count = (size_t)raw_count;
        if (input->shape[axis] % count) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator params.num_outputs",
                                 "must divide the axis extent.");
        }
    }
    if (count > SIZE_MAX / sizeof(*sizes)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator params.split", "output count is too large.");
    }
    sizes = (uint64_t*)malloc(count * sizeof(*sizes));
    if (!sizes) return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                     "operator params.split", "allocation failed.");
    for (index = 0; index < count; ++index) {
        char path[224];
        if (raw) {
            snprintf(path, sizeof(path), "operator params.split[%zu]", index);
            if (vx_shape_param_integer_value(raw[index], 1, path,
                                             &sizes[index], error)) goto cleanup;
        } else {
            sizes[index] = input->shape[axis] / count;
        }
        if (vx_shape_checked_add_u64(sum, sizes[index], "Split size sum",
                                     &sum, error)) goto cleanup;
    }
    if (sum != input->shape[axis]) {
        vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                      "operator params.split",
                      "must sum to the input axis extent.");
        goto cleanup;
    }
    if (vx_shape_result_allocate(candidate, count, error)) goto cleanup;
    for (index = 0; index < count; ++index) {
        uint64_t output_shape[8];
        VxShapeQuantization quantization = input->quantization;
        char name[64];
        memcpy(output_shape, input->shape, input->rank * sizeof(*output_shape));
        output_shape[axis] = sizes[index];
        if (quantization.scheme == VX_SHAPE_QUANTIZATION_PER_AXIS &&
            quantization.axis == axis) {
            quantization.count = (size_t)sizes[index];
            quantization.scales = input->quantization.scales + offset;
            quantization.zero_points = input->quantization.zero_points + offset;
        }
        snprintf(name, sizeof(name), "out%zu", index);
        if (vx_shape_clone_output_at(candidate, index, name, output_shape,
                                     input->rank, input->dtype,
                                     &quantization, error)) {
            vx_shape_contract_result_clear(candidate);
            goto cleanup;
        }
        offset += (size_t)sizes[index];
    }
    status = 0;
cleanup:
    free(sizes);
    return status;
}

static int vx_shape_infer_slice(const VxConcreteShapeRequest* request,
                                VxConcreteShapeResult* candidate,
                                VxShapeContractError* error) {
    static const char* const FIELDS[] = {"starts", "ends", "axes", "steps"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const double* starts_raw;
    const double* ends_raw;
    const double* axes_raw = NULL;
    const double* steps_raw = NULL;
    size_t starts_count;
    size_t ends_count;
    size_t axes_count;
    size_t steps_count;
    size_t* axes = NULL;
    uint64_t output_shape[8];
    size_t starts[8] = {0};
    size_t ends[8] = {0};
    size_t steps[8] = {0};
    size_t index;
    VxShapeQuantization quantization = input->quantization;
    if (input->rank < 1 || input->rank > 8) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator input 'input'.shape",
                             "must have rank in [1, 8].");
    }
    if (vx_shape_validate_params(request, FIELDS, 4, error) ||
        vx_shape_param_number_array(request, "starts", 1, &starts_raw,
                                    &starts_count, error) ||
        vx_shape_param_number_array(request, "ends", 1, &ends_raw,
                                    &ends_count, error) ||
        vx_shape_param_number_array(request, "axes", 0, &axes_raw,
                                    &axes_count, error) ||
        vx_shape_param_number_array(request, "steps", 0, &steps_raw,
                                    &steps_count, error)) return -1;
    if (!starts_count || starts_count != ends_count ||
        (axes_raw && axes_count != starts_count) ||
        (steps_raw && steps_count != starts_count)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params",
                             "starts, ends, axes, and steps must have equal nonzero lengths.");
    }
    axes = (size_t*)malloc(starts_count * sizeof(*axes));
    if (!axes) return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                    "operator params.axes", "allocation failed.");
    if (axes_raw) {
        if (vx_shape_normalize_axes(axes_raw, axes_count, input->rank,
                                    "operator params.axes", axes, error)) {
            free(axes);
            return -1;
        }
    } else {
        if (starts_count > input->rank) {
            free(axes);
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.axes",
                                 "resolves outside the supported rank.");
        }
        for (index = 0; index < starts_count; ++index) axes[index] = index;
    }
    memcpy(output_shape, input->shape, input->rank * sizeof(*output_shape));
    for (index = 0; index < input->rank; ++index) {
        starts[index] = 0;
        ends[index] = (size_t)input->shape[index];
        steps[index] = 1;
    }
    for (index = 0; index < starts_count; ++index) {
        int64_t raw_start = 0;
        int64_t raw_end = 0;
        uint64_t step = 1;
        uint64_t extent = input->shape[axes[index]];
        int64_t start;
        int64_t end;
        char path[224];
        snprintf(path, sizeof(path), "operator params.starts[%zu]", index);
        if (vx_shape_param_signed_integer_value(starts_raw[index], path,
                                                &raw_start, error)) goto slice_fail;
        snprintf(path, sizeof(path), "operator params.ends[%zu]", index);
        if (vx_shape_param_signed_integer_value(ends_raw[index], path,
                                                &raw_end, error)) goto slice_fail;
        if (steps_raw) {
            snprintf(path, sizeof(path), "operator params.steps[%zu]", index);
            if (vx_shape_param_integer_value(steps_raw[index], 1, path,
                                             &step, error)) goto slice_fail;
        }
        start = raw_start < 0 ? raw_start + (int64_t)extent : raw_start;
        end = raw_end < 0 ? raw_end + (int64_t)extent : raw_end;
        if (start < 0) start = 0;
        if (end < 0) end = 0;
        if ((uint64_t)start > extent) start = (int64_t)extent;
        if ((uint64_t)end > extent) end = (int64_t)extent;
        if (end <= start) {
            snprintf(path, sizeof(path), "operator params.ends[%zu]", index);
            vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                          path, "selects an empty axis, which v1 forbids.");
            goto slice_fail;
        }
        starts[axes[index]] = (size_t)start;
        ends[axes[index]] = (size_t)end;
        steps[axes[index]] = (size_t)step;
        output_shape[axes[index]] =
            ((uint64_t)(end - start) - 1) / step + 1;
    }
    if (quantization.scheme == VX_SHAPE_QUANTIZATION_PER_AXIS) {
        size_t qaxis = quantization.axis;
        size_t count = 0;
        size_t position;
        for (position = starts[qaxis]; position < ends[qaxis];
             position += steps[qaxis]) count++;
        quantization.count = count;
        if (steps[qaxis] == 1) {
            quantization.scales += starts[qaxis];
            quantization.zero_points += starts[qaxis];
        } else {
            float* gathered_scales = (float*)malloc(count * sizeof(float));
            int32_t* gathered_zeros = (int32_t*)malloc(count * sizeof(int32_t));
            size_t out = 0;
            int clone_status;
            if (!gathered_scales || !gathered_zeros) {
                free(gathered_scales);
                free(gathered_zeros);
                vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                              "operator input 'input'.quantization",
                              "allocation failed.");
                goto slice_fail;
            }
            for (position = starts[qaxis]; position < ends[qaxis];
                 position += steps[qaxis]) {
                gathered_scales[out] = input->quantization.scales[position];
                gathered_zeros[out++] = input->quantization.zero_points[position];
            }
            quantization.scales = gathered_scales;
            quantization.zero_points = gathered_zeros;
            clone_status = vx_shape_clone_output(candidate, output_shape, input->rank,
                                                 input->dtype, &quantization, error);
            free(gathered_scales);
            free(gathered_zeros);
            free(axes);
            return clone_status;
        }
    }
    free(axes);
    return vx_shape_clone_output(candidate, output_shape, input->rank,
                                 input->dtype, &quantization, error);
slice_fail:
    free(axes);
    return -1;
}

static int vx_shape_infer_pad(const VxConcreteShapeRequest* request,
                              VxConcreteShapeResult* candidate,
                              VxShapeContractError* error) {
    static const char* const FIELDS[] = {"pads", "value"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeParam* value_param;
    const double* raw;
    size_t count;
    uint64_t* output_shape = NULL;
    size_t axis;
    VxShapeQuantization quantization = input->quantization;
    int status = -1;
    if (vx_shape_rank(input, 1, 0, 0, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 2, error) ||
        vx_shape_param_number_array(request, "pads", 1, &raw, &count, error)) {
        return -1;
    }
    if (count != input->rank * 2) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.pads",
                             "must have exactly twice the input rank entries.");
    }
    value_param = vx_shape_find_param(request, "value");
    if (value_param &&
        (value_param->kind != VX_SHAPE_PARAM_NUMBER ||
         !isfinite(value_param->value.number))) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.value", "must be finite.");
    }
    output_shape = (uint64_t*)malloc(input->rank * sizeof(*output_shape));
    if (!output_shape) return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                            "operator output 'out'.shape",
                                            "allocation failed.");
    for (axis = 0; axis < input->rank; ++axis) {
        uint64_t before = 0;
        uint64_t after = 0;
        char path[224];
        snprintf(path, sizeof(path), "operator params.pads[%zu]", axis);
        if (vx_shape_param_integer_value(raw[axis], 0, path, &before, error)) goto cleanup;
        snprintf(path, sizeof(path), "operator params.pads[%zu]", input->rank + axis);
        if (vx_shape_param_integer_value(raw[input->rank + axis], 0, path,
                                         &after, error) ||
            vx_shape_checked_add_u64(input->shape[axis], before,
                                     "Pad axis", &output_shape[axis], error) ||
            vx_shape_checked_add_u64(output_shape[axis], after,
                                     "Pad axis", &output_shape[axis], error)) goto cleanup;
        if (quantization.scheme == VX_SHAPE_QUANTIZATION_PER_AXIS &&
            quantization.axis == axis && (before || after)) {
            snprintf(path, sizeof(path), "operator params.pads[%zu]", axis);
            vx_shape_fail(error,
                          VX_SHAPE_CONTRACT_ERROR_UNSAFE_QUANTIZATION_TRANSFORM,
                          path, "must not extend a per-axis quantization dimension.");
            goto cleanup;
        }
    }
    if (vx_shape_clone_output(candidate, output_shape, input->rank,
                              input->dtype, &quantization, error)) goto cleanup;
    status = 0;
cleanup:
    free(output_shape);
    return status;
}

static int vx_shape_indices(const VxShapeTensorDescriptor* indices,
                            VxShapeContractError* error) {
    if (indices->dtype != VX_DTYPE_I32) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator input 'indices'.dtype", "must be int32.");
    }
    return vx_shape_unquantized(indices, "operator input 'indices'", error);
}

static int vx_shape_infer_gather(const VxConcreteShapeRequest* request,
                                 VxConcreteShapeResult* candidate,
                                 VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* indices = vx_shape_input_descriptor(request, "indices");
    uint64_t* output_shape;
    size_t output_rank;
    size_t axis;
    size_t output_axis = 0;
    size_t input_axis;
    VxShapeQuantization quantization = input->quantization;
    int status;
    if (vx_shape_rank(input, 1, 0, 0, "operator input 'input'.shape", error) ||
        vx_shape_indices(indices, error) ||
        vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_normalize_axis(request, "axis", input->rank, 0, 0,
                                &axis, error)) return -1;
    if (input->rank - 1 > SIZE_MAX - indices->rank) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator output 'out'.shape", "rank is too large.");
    }
    output_rank = input->rank - 1 + indices->rank;
    output_shape = output_rank
        ? (uint64_t*)malloc(output_rank * sizeof(*output_shape)) : NULL;
    if (output_rank && !output_shape) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator output 'out'.shape", "allocation failed.");
    }
    for (input_axis = 0; input_axis < axis; ++input_axis)
        output_shape[output_axis++] = input->shape[input_axis];
    for (input_axis = 0; input_axis < indices->rank; ++input_axis)
        output_shape[output_axis++] = indices->shape[input_axis];
    for (input_axis = axis + 1; input_axis < input->rank; ++input_axis)
        output_shape[output_axis++] = input->shape[input_axis];
    if (quantization.scheme == VX_SHAPE_QUANTIZATION_PER_AXIS) {
        if (quantization.axis == axis) {
            free(output_shape);
            return vx_shape_fail(
                error, VX_SHAPE_CONTRACT_ERROR_UNSAFE_QUANTIZATION_TRANSFORM,
                "operator input 'input'.quantization.axis",
                "Gather indices would reorder the per-axis affine metadata.");
        }
        if (quantization.axis > axis) {
            if (indices->rank == 0) quantization.axis--;
            else quantization.axis += indices->rank - 1;
        }
    }
    status = vx_shape_clone_output(candidate, output_shape, output_rank,
                                   input->dtype, &quantization, error);
    free(output_shape);
    return status;
}

static int vx_shape_infer_gather_elements(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* indices = vx_shape_input_descriptor(request, "indices");
    size_t axis;
    size_t dimension;
    VxShapeQuantization quantization = input->quantization;
    if (vx_shape_rank(input, 1, 0, 0, "operator input 'input'.shape", error) ||
        vx_shape_indices(indices, error)) return -1;
    if (indices->rank != input->rank) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator input 'indices'.shape",
                             "must have the same rank as input.");
    }
    if (vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_normalize_axis(request, "axis", input->rank, 0, 0,
                                &axis, error)) return -1;
    for (dimension = 0; dimension < input->rank; ++dimension) {
        if (dimension != axis && indices->shape[dimension] > input->shape[dimension]) {
            char path[224];
            snprintf(path, sizeof(path), "operator input 'indices'.shape[%zu]",
                     dimension);
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 path,
                                 "must not exceed the corresponding input dimension.");
        }
    }
    if (quantization.scheme == VX_SHAPE_QUANTIZATION_PER_AXIS) {
        if (quantization.axis == axis) {
            return vx_shape_fail(
                error, VX_SHAPE_CONTRACT_ERROR_UNSAFE_QUANTIZATION_TRANSFORM,
                "operator input 'input'.quantization.axis",
                "GatherElements indices would reorder the per-axis affine metadata.");
        }
        if (indices->shape[quantization.axis] > quantization.count) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator input indices",
                                 "exceeds the per-axis input extent.");
        }
        quantization.count = (size_t)indices->shape[quantization.axis];
    }
    return vx_shape_clone_output(candidate, indices->shape, indices->rank,
                                 input->dtype, &quantization, error);
}

static int vx_shape_infer_structural_kind(
        VxOperatorKind kind,
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    switch (kind) {
        case VX_OP_SUB:
        case VX_OP_DIV:
            return vx_shape_infer_broadcast_arithmetic(request, candidate, error);
        case VX_OP_EQUAL:
        case VX_OP_GREATER_OR_EQUAL:
            return vx_shape_infer_broadcast_comparison(request, candidate, error);
        case VX_OP_WHERE:
            return vx_shape_infer_where(request, candidate, error);
        case VX_OP_REDUCE_SUM:
        case VX_OP_REDUCE_MEAN:
            return vx_shape_infer_reduction(kind, request, candidate, error);
        case VX_OP_ARG_MAX:
            return vx_shape_infer_arg_max(request, candidate, error);
        case VX_OP_TRANSPOSE:
            return vx_shape_infer_transpose(request, candidate, error);
        case VX_OP_FLATTEN:
            return vx_shape_infer_flatten(request, candidate, error);
        case VX_OP_SQUEEZE:
            return vx_shape_infer_squeeze(request, candidate, error);
        case VX_OP_UNSQUEEZE:
            return vx_shape_infer_unsqueeze(request, candidate, error);
        case VX_OP_RESHAPE:
            return vx_shape_infer_reshape(request, candidate, error);
        case VX_OP_EXPAND:
            return vx_shape_infer_expand(request, candidate, error);
        case VX_OP_CONCAT:
            return vx_shape_infer_concat(request, candidate, error);
        case VX_OP_SPLIT:
            return vx_shape_infer_split(request, candidate, error);
        case VX_OP_SLICE:
            return vx_shape_infer_slice(request, candidate, error);
        case VX_OP_PAD:
            return vx_shape_infer_pad(request, candidate, error);
        case VX_OP_GATHER:
            return vx_shape_infer_gather(request, candidate, error);
        case VX_OP_GATHER_ELEMENTS:
            return vx_shape_infer_gather_elements(request, candidate, error);
        default:
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_UNKNOWN_OPERATOR,
                                 "operator",
                                 "has no implemented structural contract.");
    }
}

static int vx_shape_attention_parameters(
        const VxConcreteShapeRequest* request,
        uint64_t* heads,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"heads", "causal", "scale"};
    const VxShapeParam* param;
    double value;
    float canonical;
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
    if (!param) return 0;
    if (param->kind != VX_SHAPE_PARAM_NUMBER) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.scale",
                             "must be positive and representable as float32.");
    }
    value = param->value.number;
    canonical = (float)value;
    if (!isfinite(value) || value <= 0.0 ||
        !isfinite(canonical) || canonical <= 0.0f) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.scale",
                             "must be positive and representable as float32.");
    }
    return 0;
}

static int vx_shape_attention_rank(
        const VxShapeTensorDescriptor* descriptor,
        const char* path,
        VxShapeContractError* error) {
    if (descriptor->rank != 2 && descriptor->rank != 3) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             path, "must have rank 2 or 3.");
    }
    return 0;
}

static int vx_shape_attention_mask(
        const VxShapeTensorDescriptor* mask,
        uint64_t batch,
        uint64_t queries,
        uint64_t keys,
        VxShapeContractError* error) {
    int valid;
    if (!mask) return 0;
    if (mask->dtype != VX_DTYPE_I32) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator input 'mask'.dtype",
                             "must be 'int32'.");
    }
    if (vx_shape_unquantized(mask, "operator input 'mask'", error)) return -1;
    valid = (mask->rank == 1 && mask->shape[0] == keys) ||
            (mask->rank == 2 && mask->shape[1] == keys &&
             (mask->shape[0] == batch || mask->shape[0] == queries)) ||
            (mask->rank == 3 && mask->shape[0] == batch &&
             mask->shape[1] == queries && mask->shape[2] == keys);
    if (!valid) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'mask'.shape",
                             "must have shape [K], [B,K], [Q,K], or [B,Q,K].");
    }
    return 0;
}

