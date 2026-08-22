/* Graph package envelope and schema validation.
 *
 * Fragment of public_api.c, included where these definitions used to sit.
 * Everything here reads untrusted model JSON and decides whether it is a
 * well-formed package at all, before any shape or backend reasoning starts. */

static int vx_graph_outputs_envelope_valid(const cJSON* outputs) {
    int count;
    if (!cJSON_IsArray(outputs) || !outputs->child) return 0;
    count = cJSON_GetArraySize(outputs);
    if (count <= 0 || count > MAXT) return 0;
    for (const cJSON* item = outputs->child; item; item = item->next) {
        if (!cJSON_IsString(item) || !item->valuestring ||
            !item->valuestring[0] || strlen(item->valuestring) >= 128u) return 0;
        for (const cJSON* prior = outputs->child; prior && prior != item;
             prior = prior->next) {
            if (cJSON_IsString(prior) && prior->valuestring &&
                !strcmp(prior->valuestring, item->valuestring)) return 0;
        }
    }
    return 1;
}

static int vx_graph_quantization_envelope_valid(const cJSON* root) {
    const cJSON* quantization;
    const cJSON* format;
    const cJSON* tensors;
    if (cJSON_GetObjectItemCaseSensitive(root, "weights_quantization") ||
        cJSON_GetObjectItemCaseSensitive(
            root, "weights_quantization_storage")) return 0;
    quantization = cJSON_GetObjectItemCaseSensitive(root, "quantization");
    if (!quantization) return 1;
    if (!cJSON_IsObject(quantization) ||
        cJSON_GetArraySize(quantization) != 2) return 0;
    format = cJSON_GetObjectItemCaseSensitive(quantization, "format");
    tensors = cJSON_GetObjectItemCaseSensitive(quantization, "tensors");
    if (!cJSON_IsString(format) || !format->valuestring ||
        strcmp(format->valuestring, "volvox-affine-safetensors/v1") ||
        !cJSON_IsObject(tensors) || !tensors->child) return 0;
    for (const cJSON* item = quantization->child; item; item = item->next) {
        if (!item->string || (strcmp(item->string, "format") &&
                              strcmp(item->string, "tensors"))) return 0;
    }
    for (const cJSON* descriptor = tensors->child; descriptor;
         descriptor = descriptor->next) {
        const cJSON* scheme;
        const cJSON* axis;
        const cJSON* scale;
        const cJSON* zero;
        int per_axis;
        int expected_fields;
        if (!descriptor->string || !descriptor->string[0] ||
            strlen(descriptor->string) >= 128u ||
            !cJSON_IsObject(descriptor)) return 0;
        scheme = cJSON_GetObjectItemCaseSensitive(descriptor, "scheme");
        axis = cJSON_GetObjectItemCaseSensitive(descriptor, "axis");
        scale = cJSON_GetObjectItemCaseSensitive(descriptor, "scale_tensor");
        zero = cJSON_GetObjectItemCaseSensitive(
            descriptor, "zero_point_tensor");
        if (!cJSON_IsString(scheme) || !scheme->valuestring) return 0;
        if (!strcmp(scheme->valuestring, "per_tensor")) per_axis = 0;
        else if (!strcmp(scheme->valuestring, "per_axis")) per_axis = 1;
        else return 0;
        expected_fields = per_axis ? 4 : 3;
        if (cJSON_GetArraySize(descriptor) != expected_fields ||
            (per_axis && (!cJSON_IsNumber(axis) ||
                          axis->valuedouble != (double)axis->valueint)) ||
            (!per_axis && axis) ||
            !cJSON_IsString(scale) || !scale->valuestring ||
            !scale->valuestring[0] || strlen(scale->valuestring) >= 128u ||
            !cJSON_IsString(zero) || !zero->valuestring ||
            !zero->valuestring[0] || strlen(zero->valuestring) >= 128u ||
            !strcmp(scale->valuestring, zero->valuestring)) return 0;
        for (const cJSON* item = descriptor->child; item; item = item->next) {
            if (!item->string ||
                (strcmp(item->string, "scheme") &&
                 strcmp(item->string, "axis") &&
                 strcmp(item->string, "scale_tensor") &&
                 strcmp(item->string, "zero_point_tensor"))) return 0;
        }
    }
    return 1;
}

static int vx_graph_value_contains_affine_alias(const cJSON* value) {
    static const char* const forbidden[] = {
        "quantization",
        "input_scale", "input_zero_point",
        "output_scale", "output_zero_point",
        "weight_scale", "weight_zero_point",
        "zero_point",
        "scales", "zero_points",
        "scale_tensor", "zero_point_tensor",
    };
    for (const cJSON* item = value ? value->child : NULL;
         item; item = item->next) {
        for (size_t index = 0;
             item->string && index < sizeof(forbidden) / sizeof(forbidden[0]);
             index++) {
            if (!strcmp(item->string, forbidden[index])) return 1;
        }
        if (vx_graph_value_contains_affine_alias(item)) return 1;
    }
    return 0;
}

static int vx_graph_params_affine_payload_free(const cJSON* params) {
    if (!params) return 1;
    return cJSON_IsObject(params) &&
           !vx_graph_value_contains_affine_alias(params);
}

static int vx_graph_envelope_schema_valid(const cJSON* root) {
    const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    const cJSON* nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    if ((inputs && !cJSON_IsObject(inputs)) ||
        (nodes && !cJSON_IsArray(nodes)) ||
        !vx_graph_quantization_envelope_valid(root)) return 0;
    for (const cJSON* input = inputs ? inputs->child : NULL;
         input; input = input->next) {
        const cJSON* dtype = cJSON_GetObjectItemCaseSensitive(input, "dtype");
        if (!cJSON_IsObject(input) || !cJSON_IsString(dtype) ||
            !dtype->valuestring ||
            cJSON_GetObjectItemCaseSensitive(input, "quantization")) return 0;
        if (strcmp(dtype->valuestring, "float32") &&
            strcmp(dtype->valuestring, "int32") &&
            strcmp(dtype->valuestring, "int8") &&
            strcmp(dtype->valuestring, "uint8")) return 0;
    }
    for (const cJSON* node = nodes ? nodes->child : NULL;
         node; node = node->next) {
        const cJSON* op = cJSON_GetObjectItemCaseSensitive(node, "opType");
        const cJSON* output_dtypes =
            cJSON_GetObjectItemCaseSensitive(node, "outputs_dtype");
        const cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
        if (!cJSON_IsObject(node) || !cJSON_IsString(op) ||
            !op->valuestring || !op->valuestring[0] ||
            !vx_graph_runtime_operator_name_is_current(op->valuestring) ||
            cJSON_GetObjectItemCaseSensitive(node, "op") ||
            cJSON_GetObjectItemCaseSensitive(node, "outputs_quantization") ||
            !vx_graph_params_affine_payload_free(params) ||
            (output_dtypes && !cJSON_IsObject(output_dtypes))) return 0;
        for (const cJSON* dtype = output_dtypes ? output_dtypes->child : NULL;
             dtype; dtype = dtype->next) {
            if (!cJSON_IsString(dtype) || !dtype->valuestring ||
                (strcmp(dtype->valuestring, "float32") &&
                 strcmp(dtype->valuestring, "int32") &&
                 strcmp(dtype->valuestring, "int8") &&
                 strcmp(dtype->valuestring, "uint8"))) return 0;
        }
    }
    return 1;
}

static char* vx_string_copy(const char* value);
static void vx_model_bank_residency_clear(VxModel* model);

static size_t vx_execution_dtype_size(VxDataType dtype) {
    switch (dtype) {
        case VX_DTYPE_I8:
        case VX_DTYPE_U8: return 1u;
        case VX_DTYPE_F32:
        case VX_DTYPE_I32: return 4u;
        default: return 0u;
    }
}

static void vx_declared_outputs_free(VxDeclaredOutput* outputs, size_t count) {
    if (!outputs) return;
    for (size_t index = 0; index < count; index++) {
        free(outputs[index].name);
        for (uint32_t axis = 0; axis < outputs[index].rank; axis++)
            free(outputs[index].symbols[axis]);
    }
    free(outputs);
}

static int vx_graph_execution_dtype(const cJSON* item, VxDataType* out) {
    if (!cJSON_IsString(item) || !item->valuestring || !out) return -1;
    if (!strcmp(item->valuestring, "float32")) *out = VX_DTYPE_F32;
    else if (!strcmp(item->valuestring, "int32")) *out = VX_DTYPE_I32;
    else if (!strcmp(item->valuestring, "int8")) *out = VX_DTYPE_I8;
    else if (!strcmp(item->valuestring, "uint8")) *out = VX_DTYPE_U8;
    else return -1;
    return 0;
}

static int vx_declared_output_set_json(VxDeclaredOutput* output,
                                       const cJSON* shape,
                                       const cJSON* dtype) {
    size_t elements = 1u;
    size_t element_size;
    int rank;
    if (!output || !cJSON_IsArray(shape) ||
        vx_graph_execution_dtype(dtype, &output->dtype) != 0)
        return -1;
    rank = cJSON_GetArraySize(shape);
    if (rank < 0 || rank > (int)VX_MAX_TENSOR_RANK) return -1;
    memset(output->shape, 0, sizeof(output->shape));
    for (int axis = 0; axis < rank; axis++) {
        const cJSON* dimension = cJSON_GetArrayItem(shape, axis);
        if (!cJSON_IsNumber(dimension) ||
            dimension->valuedouble != (double)dimension->valueint ||
            dimension->valueint <= 0 ||
            (size_t)dimension->valueint > SIZE_MAX / elements)
            return -1;
        output->shape[axis] = dimension->valueint;
        elements *= (size_t)dimension->valueint;
    }
    element_size = vx_execution_dtype_size(output->dtype);
    if (!element_size || elements > SIZE_MAX / element_size) return -1;
    output->rank = (uint32_t)rank;
    output->byte_size = elements * element_size;
    return 0;
}

static int vx_declared_output_set_weight(VxDeclaredOutput* output,
                                         const SafetensorsTensor* tensor) {
    size_t storage_size;
    size_t execution_size;
    size_t elements = 1u;
    VxDataType dtype;
    if (!output || !tensor || tensor->ndim < 0 ||
        tensor->ndim > (int)VX_MAX_TENSOR_RANK)
        return -1;
    switch (tensor->dtype) {
        case SAFETENSORS_DTYPE_F32: dtype = VX_DTYPE_F32; break;
        case SAFETENSORS_DTYPE_F16: dtype = VX_DTYPE_F32; break;
        case SAFETENSORS_DTYPE_I32: dtype = VX_DTYPE_I32; break;
        case SAFETENSORS_DTYPE_I8: dtype = VX_DTYPE_I8; break;
        case SAFETENSORS_DTYPE_U8: dtype = VX_DTYPE_U8; break;
        default: return -1;
    }
    for (int axis = 0; axis < tensor->ndim; axis++) {
        if (tensor->shape[axis] <= 0 ||
            (size_t)tensor->shape[axis] > SIZE_MAX / elements)
            return -1;
        elements *= (size_t)tensor->shape[axis];
    }
    if (safetensors_tensor_nbytes(tensor->dtype, tensor->shape,
                                  tensor->ndim, &storage_size) != 0 ||
        storage_size != tensor->nbytes ||
        elements > SIZE_MAX / vx_execution_dtype_size(dtype))
        return -1;
    execution_size = elements * vx_execution_dtype_size(dtype);
    output->dtype = dtype;
    output->rank = (uint32_t)tensor->ndim;
    memset(output->shape, 0, sizeof(output->shape));
    for (int axis = 0; axis < tensor->ndim; axis++)
        output->shape[axis] = tensor->shape[axis];
    output->byte_size = execution_size;
    return 0;
}

static VxStatus vx_graph_package_validate_envelope(
    const char* path,
    const char* const* weight_paths,
    size_t weight_path_count,
    int require_resolved_outputs,
    VxDeclaredOutput** out_outputs,
    size_t* out_output_count) {
    FILE* file;
    long signed_size;
    size_t size;
    size_t read_size;
    char* bytes;
    cJSON* root;
    cJSON* format;
    cJSON* outputs;
    VxStatus status = VX_STATUS_INVALID_GRAPH;
    VxDeclaredOutput* declared_outputs = NULL;
    unsigned char* graph_resolved = NULL;
    size_t output_count = 0;
    if (!out_outputs || !out_output_count ||
        (weight_path_count && !weight_paths))
        return VX_STATUS_INVALID_ARGUMENT;
    *out_outputs = NULL;
    *out_output_count = 0;
    file = fopen(path, "rb");
    if (!file) return VX_STATUS_IO_ERROR;
    if (fseek(file, 0, SEEK_END) != 0 ||
        (signed_size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return VX_STATUS_IO_ERROR;
    }
    size = (size_t)signed_size;
    if ((long)size != signed_size || size == SIZE_MAX) {
        fclose(file);
        return VX_STATUS_INVALID_GRAPH;
    }
    bytes = (char*)malloc(size + 1u);
    if (!bytes) {
        fclose(file);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    read_size = fread(bytes, 1, size, file);
    if (read_size != size || ferror(file)) {
        fclose(file);
        free(bytes);
        return VX_STATUS_IO_ERROR;
    }
    if (fclose(file) != 0) {
        free(bytes);
        return VX_STATUS_IO_ERROR;
    }
    bytes[size] = '\0';
    root = cJSON_ParseWithLength(bytes, size + 1u);
    free(bytes);
    if (!root) return VX_STATUS_INVALID_GRAPH;
    format = cJSON_GetObjectItemCaseSensitive(root, "format");
    outputs = cJSON_GetObjectItemCaseSensitive(root, "outputs");
    if (cJSON_IsObject(root) && vx_json_object_keys_unique_recursive(root) &&
        cJSON_IsString(format) && format->valuestring &&
        !strcmp(format->valuestring, "volvox-graph/v1") &&
        vx_graph_outputs_envelope_valid(outputs) &&
        vx_graph_envelope_schema_valid(root)) {
        output_count = (size_t)cJSON_GetArraySize(outputs);
        declared_outputs = (VxDeclaredOutput*)calloc(
            output_count, sizeof(*declared_outputs));
        graph_resolved = (unsigned char*)calloc(output_count, 1u);
        if (!declared_outputs || !graph_resolved) {
            status = VX_STATUS_OUT_OF_MEMORY;
        } else {
            status = VX_STATUS_OK;
            for (size_t index = 0; index < output_count; index++) {
                const cJSON* item = cJSON_GetArrayItem(
                    outputs, (int)index);
                const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(
                    root, "inputs");
                const cJSON* input = cJSON_IsObject(inputs)
                    ? cJSON_GetObjectItemCaseSensitive(inputs,
                                                       item->valuestring)
                    : NULL;
                declared_outputs[index].dtype = (VxDataType)-1;
                declared_outputs[index].name = vx_string_copy(
                    item->valuestring);
                if (!declared_outputs[index].name) {
                    status = VX_STATUS_OUT_OF_MEMORY;
                    break;
                }
                if (input) {
                    const cJSON* shape = cJSON_GetObjectItemCaseSensitive(
                        input, "shape");
                    const cJSON* dtype = cJSON_GetObjectItemCaseSensitive(
                        input, "dtype");
                    if (vx_declared_output_set_json(
                            &declared_outputs[index], shape, dtype) != 0) {
                        status = VX_STATUS_INVALID_GRAPH;
                        break;
                    }
                    graph_resolved[index] = 1u;
                    continue;
                }
                const cJSON* nodes = cJSON_GetObjectItemCaseSensitive(
                    root, "nodes");
                for (const cJSON* node = nodes ? nodes->child : NULL;
                     node; node = node->next) {
                    const cJSON* node_outputs =
                        cJSON_GetObjectItemCaseSensitive(node, "outputs");
                    for (const cJSON* node_output = node_outputs
                             ? node_outputs->child : NULL;
                         node_output; node_output = node_output->next) {
                        if (!cJSON_IsString(node_output) ||
                            !node_output->valuestring ||
                            strcmp(node_output->valuestring,
                                   item->valuestring))
                            continue;
                        const cJSON* shapes =
                            cJSON_GetObjectItemCaseSensitive(
                                node, "outputs_shape");
                        const cJSON* dtypes =
                            cJSON_GetObjectItemCaseSensitive(
                                node, "outputs_dtype");
                        const cJSON* shape = cJSON_IsObject(shapes)
                            ? cJSON_GetObjectItemCaseSensitive(
                                  shapes, node_output->string)
                            : NULL;
                        const cJSON* dtype = dtypes
                            ? cJSON_GetObjectItemCaseSensitive(
                                  dtypes, node_output->string)
                            : NULL;
                        cJSON default_dtype = {0};
                        if (graph_resolved[index] || !shape ||
                            (dtypes && !dtype)) {
                            status = VX_STATUS_INVALID_GRAPH;
                            break;
                        }
                        if (!dtype) {
                            default_dtype.type = cJSON_String;
                            default_dtype.valuestring = (char*)"float32";
                            dtype = &default_dtype;
                        }
                        if (vx_declared_output_set_json(
                                &declared_outputs[index], shape, dtype) != 0) {
                            status = VX_STATUS_INVALID_GRAPH;
                            break;
                        }
                        graph_resolved[index] = 1u;
                    }
                    if (status != VX_STATUS_OK) break;
                }
                if (status != VX_STATUS_OK) break;
            }
            for (size_t weight_index = 0;
                 status == VX_STATUS_OK && weight_index < weight_path_count;
                 weight_index++) {
                SafetensorsFile weights = {0};
                int needed = 0;
                for (size_t output_index = 0;
                     output_index < output_count; output_index++)
                    if (!graph_resolved[output_index]) needed = 1;
                if (!needed) break;
                if (safetensors_load(weight_paths[weight_index], &weights) != 0) {
                    status = VX_STATUS_INVALID_GRAPH;
                    break;
                }
                for (size_t output_index = 0;
                     output_index < output_count; output_index++) {
                    const SafetensorsTensor* tensor;
                    if (graph_resolved[output_index]) continue;
                    tensor = safetensors_find_tensor(
                        &weights, declared_outputs[output_index].name);
                    if (tensor && vx_declared_output_set_weight(
                                      &declared_outputs[output_index],
                                      tensor) != 0)
                        declared_outputs[output_index].dtype =
                            (VxDataType)-2;
                }
                safetensors_free(&weights);
            }
            if (require_resolved_outputs) {
                for (size_t index = 0;
                     status == VX_STATUS_OK && index < output_count; index++)
                    if (!vx_execution_dtype_size(declared_outputs[index].dtype))
                        status = VX_STATUS_INVALID_GRAPH;
            }
        }
    }
    cJSON_Delete(root);
    free(graph_resolved);
    if (status != VX_STATUS_OK) {
        vx_declared_outputs_free(declared_outputs, output_count);
        return status;
    }
    *out_outputs = declared_outputs;
    *out_output_count = output_count;
    return status;
}

static int vx_json_object_fields(const cJSON* object,
                                 const char* const* allowed,
                                 size_t allowed_count,
                                 size_t minimum_count,
                                 size_t maximum_count) {
    size_t count = 0;
    if (!cJSON_IsObject(object)) return 0;
    for (const cJSON* item = object->child; item; item = item->next) {
        int known = 0;
        count++;
        if (!item->string) return 0;
        for (size_t index = 0; index < allowed_count; index++)
            if (!strcmp(item->string, allowed[index])) known = 1;
        if (!known) return 0;
    }
    return count >= minimum_count && count <= maximum_count;
}

static int vx_shape_symbol_valid(const char* symbol) {
    size_t length;
    if (!symbol || !symbol[0]) return 0;
    length = strlen(symbol);
    if (length > 64u || !((symbol[0] >= 'A' && symbol[0] <= 'Z') ||
                          (symbol[0] >= 'a' && symbol[0] <= 'z'))) return 0;
    for (size_t index = 1; index < length; index++) {
        unsigned char value = (unsigned char)symbol[index];
        if (!((value >= 'A' && value <= 'Z') ||
              (value >= 'a' && value <= 'z') ||
              (value >= '0' && value <= '9') || value == '_')) return 0;
    }
    return 1;
}

static int vx_json_positive_safe_i64(const cJSON* value, int64_t* out) {
    const double safe_max = 9007199254740991.0;
    int64_t converted;
    if (!cJSON_IsNumber(value) || value->valuedouble <= 0.0 ||
        value->valuedouble > safe_max) return 0;
    converted = (int64_t)value->valuedouble;
    if ((double)converted != value->valuedouble) return 0;
    if (out) *out = converted;
    return 1;
}

static int vx_dimension_json(const cJSON* dimensions,
                             const char* symbol,
                             int64_t* minimum,
                             int64_t* maximum,
                             int64_t* multiple) {
    static const char* const fields[] = {"min", "max", "multiple_of"};
    const cJSON* descriptor;
    const cJSON* min_item;
    const cJSON* max_item;
    const cJSON* multiple_item;
    int64_t min_value;
    int64_t max_value;
    int64_t multiple_value = 1;
    int64_t remainder;
    int64_t adjustment;
    if (!vx_shape_symbol_valid(symbol) || !cJSON_IsObject(dimensions)) return 0;
    descriptor = cJSON_GetObjectItemCaseSensitive(dimensions, symbol);
    if (!descriptor ||
        !vx_json_object_fields(descriptor, fields, 3u, 2u, 3u)) return 0;
    min_item = cJSON_GetObjectItemCaseSensitive(descriptor, "min");
    max_item = cJSON_GetObjectItemCaseSensitive(descriptor, "max");
    multiple_item = cJSON_GetObjectItemCaseSensitive(descriptor, "multiple_of");
    if (!vx_json_positive_safe_i64(min_item, &min_value) ||
        !vx_json_positive_safe_i64(max_item, &max_value) ||
        min_value > max_value ||
        (multiple_item &&
         !vx_json_positive_safe_i64(multiple_item, &multiple_value))) return 0;
    remainder = min_value % multiple_value;
    adjustment = remainder ? multiple_value - remainder : 0;
    if (adjustment > max_value - min_value) return 0;
    *minimum = min_value;
    *maximum = max_value;
    *multiple = multiple_value;
    return 1;
}

static int vx_declared_tensor_set(VxDeclaredTensor* tensor,
                                  const char* name,
                                  const cJSON* shape,
                                  const cJSON* dtype,
                                  const cJSON* dimensions) {
    size_t maximum_elements = 1u;
    size_t element_size;
    int rank;
    int all_fixed = 1;
    if (!tensor || !name || !name[0] || strlen(name) >= 128u ||
        !cJSON_IsArray(shape) ||
        vx_graph_execution_dtype(dtype, &tensor->dtype) != 0) return -1;
    rank = cJSON_GetArraySize(shape);
    if (rank < 0 || rank > (int)VX_MAX_TENSOR_RANK) return -1;
    tensor->name = vx_string_copy(name);
    if (!tensor->name) return -2;
    tensor->rank = (uint32_t)rank;
    for (int axis = 0; axis < rank; axis++) {
        const cJSON* dimension = cJSON_GetArrayItem(shape, axis);
        int64_t minimum;
        int64_t maximum;
        int64_t multiple;
        if (vx_json_positive_safe_i64(dimension, &minimum)) {
            maximum = minimum;
            multiple = 1;
            tensor->kinds[axis] = VX_DIMENSION_FIXED;
            tensor->shape[axis] = minimum;
        } else if (cJSON_IsString(dimension) && dimension->valuestring &&
                   vx_dimension_json(dimensions, dimension->valuestring,
                                     &minimum, &maximum, &multiple)) {
            tensor->symbols[axis] = vx_string_copy(dimension->valuestring);
            if (!tensor->symbols[axis]) return -2;
            tensor->kinds[axis] = VX_DIMENSION_SYMBOLIC;
            all_fixed = 0;
        } else {
            return -1;
        }
        tensor->minimums[axis] = minimum;
        tensor->maximums[axis] = maximum;
        tensor->multiples[axis] = multiple;
        if ((uint64_t)maximum > SIZE_MAX / maximum_elements) return -1;
        maximum_elements *= (size_t)maximum;
    }
    element_size = vx_execution_dtype_size(tensor->dtype);
    if (!element_size || maximum_elements > SIZE_MAX / element_size) return -1;
    tensor->maximum_byte_size = maximum_elements * element_size;
    if (all_fixed) tensor->byte_size = tensor->maximum_byte_size;
    return 0;
}

static int vx_declared_tensor_clone(VxDeclaredTensor* target,
                                    const VxDeclaredTensor* source) {
    *target = *source;
    target->name = vx_string_copy(source->name);
    memset(target->symbols, 0, sizeof(target->symbols));
    if (!target->name) return -1;
    for (uint32_t axis = 0; axis < source->rank; axis++) {
        if (!source->symbols[axis]) continue;
        target->symbols[axis] = vx_string_copy(source->symbols[axis]);
        if (!target->symbols[axis]) return -1;
    }
    return 0;
}

static int vx_declared_tensor_set_weight(VxDeclaredTensor* tensor,
                                         const char* name,
                                         const SafetensorsTensor* source) {
    VxDeclaredOutput concrete = {0};
    if (vx_declared_output_set_weight(&concrete, source) != 0) return -1;
    concrete.name = vx_string_copy(name);
    if (!concrete.name) return -2;
    for (uint32_t axis = 0; axis < concrete.rank; axis++) {
        concrete.kinds[axis] = VX_DIMENSION_FIXED;
        concrete.minimums[axis] = concrete.shape[axis];
        concrete.maximums[axis] = concrete.shape[axis];
        concrete.multiples[axis] = 1;
    }
    concrete.maximum_byte_size = concrete.byte_size;
    *tensor = concrete;
    return 0;
}

typedef struct VxValidatedWeightBank {
    const char* name;
    int64_t minimum;
    int64_t maximum;
    int64_t multiple;
    uint64_t slot_count;
    size_t occurrences;
} VxValidatedWeightBank;

static int vx_graph_schema_valid(const cJSON* root) {
    static const char* const root_fields[] = {
        "format", "dimensions", "inputs", "nodes",
        "outputs", "banks", "quantization"
    };
    static const char* const input_fields[] = {"dtype", "shape"};
    static const char* const node_fields[] = {
        "id", "opType", "inputs", "outputs", "params"
    };
    static const char* const output_fields[] = {"tensor", "dtype", "shape"};
    const cJSON* format = cJSON_GetObjectItemCaseSensitive(root, "format");
    const cJSON* dimensions =
        cJSON_GetObjectItemCaseSensitive(root, "dimensions");
    const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    const cJSON* nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    const cJSON* outputs = cJSON_GetObjectItemCaseSensitive(root, "outputs");
    if (!cJSON_IsObject(root) ||
        !vx_json_object_fields(root, root_fields, 7u, 5u, 7u) ||
        !cJSON_IsString(format) || !format->valuestring ||
        strcmp(format->valuestring, "volvox-graph/v1") ||
        !cJSON_IsObject(dimensions) || !cJSON_IsObject(inputs) ||
        !cJSON_IsArray(nodes) || !vx_graph_outputs_envelope_valid(outputs) ||
        !vx_graph_quantization_envelope_valid(root)) return 0;
    for (const cJSON* dimension = dimensions->child; dimension;
         dimension = dimension->next) {
        int64_t minimum;
        int64_t maximum;
        int64_t multiple;
        if (!dimension->string ||
            !vx_dimension_json(dimensions, dimension->string,
                               &minimum, &maximum, &multiple)) return 0;
    }
    /* Optional weight banks: tensor name -> the bounded dimension governing its
     * slot axis.  Slot residency is a context concern; the document only
     * declares that axis 0 of that weight is runtime-selected. */
    {
        const cJSON* banks = cJSON_GetObjectItemCaseSensitive(root, "banks");
        if (banks) {
            if (!cJSON_IsObject(banks)) return 0;
            for (const cJSON* bank = banks->child; bank; bank = bank->next) {
                int64_t minimum;
                int64_t maximum;
                int64_t multiple;
                if (!bank->string || !cJSON_IsString(bank) || !bank->valuestring ||
                    !vx_dimension_json(dimensions, bank->valuestring,
                                       &minimum, &maximum, &multiple)) return 0;
            }
        }
    }
    for (const cJSON* input = inputs->child; input; input = input->next) {
        VxDeclaredTensor temporary = {0};
        int parsed;
        if (!input->string ||
            !vx_json_object_fields(input, input_fields, 2u, 2u, 2u)) return 0;
        parsed = vx_declared_tensor_set(
            &temporary, input->string,
            cJSON_GetObjectItemCaseSensitive(input, "shape"),
            cJSON_GetObjectItemCaseSensitive(input, "dtype"), dimensions);
        if (parsed != 0) {
            free(temporary.name);
            for (uint32_t axis = 0; axis < temporary.rank; axis++)
                free(temporary.symbols[axis]);
            return 0;
        }
        free(temporary.name);
        for (uint32_t axis = 0; axis < temporary.rank; axis++)
            free(temporary.symbols[axis]);
    }
    for (const cJSON* node = nodes->child; node; node = node->next) {
        const cJSON* id = cJSON_GetObjectItemCaseSensitive(node, "id");
        const cJSON* op = cJSON_GetObjectItemCaseSensitive(node, "opType");
        const cJSON* node_inputs =
            cJSON_GetObjectItemCaseSensitive(node, "inputs");
        const cJSON* node_outputs =
            cJSON_GetObjectItemCaseSensitive(node, "outputs");
        const cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
        if (!vx_json_object_fields(node, node_fields, 5u, 5u, 5u) ||
            !cJSON_IsString(id) || !id->valuestring || !id->valuestring[0] ||
            !cJSON_IsString(op) || !op->valuestring ||
            !vx_graph_runtime_operator_name_is_current(op->valuestring) ||
            !cJSON_IsObject(node_inputs) || !cJSON_IsObject(node_outputs) ||
            !node_outputs->child || !cJSON_IsObject(params) ||
            !vx_graph_params_affine_payload_free(params)) return 0;
        for (const cJSON* output = node_outputs->child; output;
             output = output->next) {
            VxDeclaredTensor temporary = {0};
            const cJSON* tensor;
            int parsed;
            if (!vx_json_object_fields(output, output_fields, 3u, 3u, 3u))
                return 0;
            tensor = cJSON_GetObjectItemCaseSensitive(output, "tensor");
            if (!cJSON_IsString(tensor) || !tensor->valuestring) return 0;
            parsed = vx_declared_tensor_set(
                &temporary, tensor->valuestring,
                cJSON_GetObjectItemCaseSensitive(output, "shape"),
                cJSON_GetObjectItemCaseSensitive(output, "dtype"), dimensions);
            free(temporary.name);
            for (uint32_t axis = 0; axis < temporary.rank; axis++)
                free(temporary.symbols[axis]);
            if (parsed != 0) return 0;
        }
    }
    return 1;
}

static VxStatus vx_graph_package_validate(
    const char* path,
    const char* const* weight_paths,
    size_t weight_path_count,
    int require_resolved_outputs,
    const VxBankResidency* bank_residency,
    size_t bank_residency_count,
    VxDeclaredTensor** out_inputs,
    size_t* out_input_count,
    VxDeclaredTensor** out_outputs,
    size_t* out_output_count) {
    FILE* file = NULL;
    char* bytes = NULL;
    cJSON* root = NULL;
    VxDeclaredTensor* inputs = NULL;
    VxDeclaredTensor* outputs = NULL;
    VxValidatedWeightBank* validated_banks = NULL;
    unsigned char* resolved = NULL;
    VxStatus status = VX_STATUS_INVALID_GRAPH;
    long signed_size;
    size_t size;
    size_t input_count = 0u;
    size_t output_count = 0u;
    if (!path || !out_inputs || !out_input_count || !out_outputs ||
        !out_output_count || (weight_path_count && !weight_paths) ||
        (bank_residency_count && !bank_residency))
        return VX_STATUS_INVALID_ARGUMENT;
    *out_inputs = NULL;
    *out_input_count = 0;
    *out_outputs = NULL;
    *out_output_count = 0;
    file = fopen(path, "rb");
    if (!file) return VX_STATUS_IO_ERROR;
    if (fseek(file, 0, SEEK_END) != 0 ||
        (signed_size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0 ||
        (size_t)signed_size == SIZE_MAX) {
        fclose(file);
        return VX_STATUS_IO_ERROR;
    }
    size = (size_t)signed_size;
    bytes = (char*)malloc(size + 1u);
    if (!bytes) {
        fclose(file);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    {
        size_t bytes_read = fread(bytes, 1, size, file);
        int read_failed = bytes_read != size || ferror(file);
        int close_failed = fclose(file) != 0;
        if (read_failed || close_failed) {
            free(bytes);
            return VX_STATUS_IO_ERROR;
        }
    }
    bytes[size] = '\0';
    root = cJSON_ParseWithLength(bytes, size + 1u);
    free(bytes);
    if (!root || !vx_json_object_keys_unique_recursive(root) ||
        !vx_graph_schema_valid(root)) goto done;
    {
        const cJSON* dimensions =
            cJSON_GetObjectItemCaseSensitive(root, "dimensions");
        const cJSON* input_object =
            cJSON_GetObjectItemCaseSensitive(root, "inputs");
        const cJSON* output_array =
            cJSON_GetObjectItemCaseSensitive(root, "outputs");
        const cJSON* bank_object =
            cJSON_GetObjectItemCaseSensitive(root, "banks");
        size_t bank_count = bank_object
            ? (size_t)cJSON_GetArraySize(bank_object) : 0u;
        input_count = (size_t)cJSON_GetArraySize(input_object);
        output_count = (size_t)cJSON_GetArraySize(output_array);
        inputs = input_count
            ? (VxDeclaredTensor*)calloc(input_count, sizeof(*inputs)) : NULL;
        outputs = (VxDeclaredTensor*)calloc(output_count, sizeof(*outputs));
        resolved = (unsigned char*)calloc(output_count, 1u);
        validated_banks = bank_count
            ? (VxValidatedWeightBank*)calloc(
                  bank_count, sizeof(*validated_banks))
            : NULL;
        if ((input_count && !inputs) || !outputs || !resolved ||
            (bank_count && !validated_banks)) {
            status = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
        if (bank_count) {
            size_t bank_index = 0;
            for (const cJSON* bank = bank_object->child; bank;
                 bank = bank->next, bank_index++) {
                VxValidatedWeightBank* validated =
                    &validated_banks[bank_index];
                validated->name = bank->string;
                if (!vx_dimension_json(
                        dimensions, bank->valuestring,
                        &validated->minimum, &validated->maximum,
                        &validated->multiple)) {
                    status = VX_STATUS_INVALID_GRAPH;
                    goto done;
                }
            }
        }
        size_t index = 0;
        for (const cJSON* item = input_object->child; item;
             item = item->next, index++) {
            int parsed = vx_declared_tensor_set(
                &inputs[index], item->string,
                cJSON_GetObjectItemCaseSensitive(item, "shape"),
                cJSON_GetObjectItemCaseSensitive(item, "dtype"), dimensions);
            if (parsed != 0) {
                status = parsed == -2 ? VX_STATUS_OUT_OF_MEMORY :
                                        VX_STATUS_INVALID_GRAPH;
                goto done;
            }
        }
        status = VX_STATUS_OK;
        for (index = 0; index < output_count; index++) {
            const cJSON* output_name =
                cJSON_GetArrayItem(output_array, (int)index);
            for (size_t input_index = 0; input_index < input_count;
                 input_index++) {
                if (strcmp(inputs[input_index].name,
                           output_name->valuestring)) continue;
                if (vx_declared_tensor_clone(&outputs[index],
                                             &inputs[input_index]) != 0) {
                    status = VX_STATUS_OUT_OF_MEMORY;
                    goto done;
                }
                resolved[index] = 1u;
            }
            const cJSON* nodes =
                cJSON_GetObjectItemCaseSensitive(root, "nodes");
            for (const cJSON* node = nodes->child; node; node = node->next) {
                const cJSON* node_outputs =
                    cJSON_GetObjectItemCaseSensitive(node, "outputs");
                for (const cJSON* descriptor = node_outputs->child; descriptor;
                     descriptor = descriptor->next) {
                    const cJSON* tensor =
                        cJSON_GetObjectItemCaseSensitive(descriptor, "tensor");
                    int parsed;
                    if (strcmp(tensor->valuestring,
                               output_name->valuestring)) continue;
                    if (resolved[index]) {
                        status = VX_STATUS_INVALID_GRAPH;
                        goto done;
                    }
                    parsed = vx_declared_tensor_set(
                        &outputs[index], tensor->valuestring,
                        cJSON_GetObjectItemCaseSensitive(descriptor, "shape"),
                        cJSON_GetObjectItemCaseSensitive(descriptor, "dtype"),
                        dimensions);
                    if (parsed != 0) {
                        status = parsed == -2 ? VX_STATUS_OUT_OF_MEMORY :
                                                VX_STATUS_INVALID_GRAPH;
                        goto done;
                    }
                    resolved[index] = 1u;
                }
            }
            if (!resolved[index]) {
                outputs[index].name = vx_string_copy(output_name->valuestring);
                outputs[index].dtype = (VxDataType)-1;
                if (!outputs[index].name) {
                    status = VX_STATUS_OUT_OF_MEMORY;
                    goto done;
                }
            }
        }
        for (size_t weight_index = 0; weight_index < weight_path_count;
             weight_index++) {
            SafetensorsFile weights = {0};
            if (safetensors_load(weight_paths[weight_index], &weights) != 0) {
                status = VX_STATUS_INVALID_GRAPH;
                goto done;
            }
            for (index = 0; index < output_count; index++) {
                const SafetensorsTensor* tensor;
                if (resolved[index]) continue;
                tensor = safetensors_find_tensor(&weights, outputs[index].name);
                if (!tensor) continue;
                char* name = outputs[index].name;
                outputs[index].name = NULL;
                int parsed = vx_declared_tensor_set_weight(
                    &outputs[index], name, tensor);
                free(name);
                if (parsed != 0) {
                    safetensors_free(&weights);
                    status = parsed == -2 ? VX_STATUS_OUT_OF_MEMORY :
                                            VX_STATUS_INVALID_GRAPH;
                    goto done;
                }
                resolved[index] = 1u;
            }
            for (int tensor_index = 0; tensor_index < weights.tensor_count;
                 tensor_index++) {
                const SafetensorsTensor* tensor =
                    &weights.tensors[tensor_index];
                VxValidatedWeightBank* validated = NULL;
                int shape_positive;
                uint64_t slot_count;
                for (size_t bank_index = 0; bank_index < bank_count;
                     bank_index++)
                    if (!strcmp(validated_banks[bank_index].name,
                                tensor->name)) {
                        validated = &validated_banks[bank_index];
                        break;
                    }
                if (!validated) continue;
                shape_positive = tensor->ndim >= 2;
                for (int axis = 0; shape_positive && axis < tensor->ndim;
                     axis++)
                    shape_positive = tensor->shape[axis] > 0;
                if (validated->occurrences != 0u || !shape_positive) {
                    safetensors_free(&weights);
                    status = VX_STATUS_INVALID_GRAPH;
                    goto done;
                }
                slot_count = (uint64_t)tensor->shape[0];
                if (slot_count < (uint64_t)validated->minimum ||
                    slot_count > (uint64_t)validated->maximum ||
                    slot_count % (uint64_t)validated->multiple != 0u) {
                    safetensors_free(&weights);
                    status = VX_STATUS_INVALID_GRAPH;
                    goto done;
                }
                validated->slot_count = slot_count;
                validated->occurrences = 1u;
            }
            safetensors_free(&weights);
        }
        if (require_resolved_outputs) {
            for (index = 0; index < output_count; index++)
                if (!resolved[index]) status = VX_STATUS_INVALID_GRAPH;
            for (size_t bank_index = 0;
                 status == VX_STATUS_OK && bank_index < bank_count;
                 bank_index++)
                if (validated_banks[bank_index].occurrences != 1u)
                    status = VX_STATUS_INVALID_GRAPH;
            for (size_t residency_index = 0;
                 status == VX_STATUS_OK &&
                     residency_index < bank_residency_count;
                 residency_index++) {
                const VxBankResidency* requested =
                    &bank_residency[residency_index];
                const VxValidatedWeightBank* validated = NULL;
                for (size_t bank_index = 0; bank_index < bank_count;
                     bank_index++)
                    if (!strcmp(validated_banks[bank_index].name,
                                requested->bank)) {
                        validated = &validated_banks[bank_index];
                        break;
                    }
                if (!validated || !requested->slots ||
                    !requested->slot_count ||
                    (uint64_t)requested->slots[requested->slot_count - 1u] >=
                        validated->slot_count)
                    status = VX_STATUS_INVALID_ARGUMENT;
            }
        }
        if (status == VX_STATUS_OK) {
            *out_inputs = inputs;
            *out_input_count = input_count;
            *out_outputs = outputs;
            *out_output_count = output_count;
            inputs = NULL;
            outputs = NULL;
        }
    }
done:
    cJSON_Delete(root);
    free(validated_banks);
    free(resolved);
    vx_declared_outputs_free(inputs, inputs ? input_count : 0u);
    vx_declared_outputs_free(outputs, outputs ? output_count : 0u);
    return status;
}
