/* Typed logical definitions -> the existing C graph authority. No backend or
 * payload is created here; every completed draft takes the ordinary plan path. */
typedef struct VxGraphWriter {
    VxApiScratch* scratch;
    VxStatus status;
} VxGraphWriter;

static const char* vx_graph_string(VxGraphWriter* w,
                                   const SynurangLiteBytes* text,
                                   int allow_empty) {
    const char* value;
    if (!text || (!allow_empty && !text->len) ||
        (text->len && (!text->data || memchr(text->data, 0, text->len)))) {
        w->status = VX_STATUS_INVALID_ARGUMENT;
        return NULL;
    }
    if (!text->len) return "";
    value = vx_api_scratch_cstr(w->scratch, text);
    if (!value) w->status = VX_STATUS_OUT_OF_MEMORY;
    return value;
}

/* Both insertion helpers consume value, including on failure. */
static void vx_graph_member(VxGraphWriter* w, cJSON* object,
                             const char* name, cJSON* value) {
    if (w->status != VX_STATUS_OK) { cJSON_Delete(value); return; }
    if (!object || !name || !value) {
        w->status = VX_STATUS_OUT_OF_MEMORY;
        cJSON_Delete(value);
    } else if (cJSON_GetObjectItemCaseSensitive(object, name)) {
        w->status = VX_STATUS_INVALID_ARGUMENT;
        cJSON_Delete(value);
    } else if (!cJSON_AddItemToObject(object, name, value)) {
        w->status = VX_STATUS_OUT_OF_MEMORY;
        cJSON_Delete(value);
    }
}

static void vx_graph_append(VxGraphWriter* w, cJSON* array, cJSON* value) {
    if (w->status != VX_STATUS_OK) { cJSON_Delete(value); return; }
    if (!array || !value || !cJSON_AddItemToArray(array, value)) {
        w->status = VX_STATUS_OUT_OF_MEMORY;
        cJSON_Delete(value);
    }
}

static cJSON* vx_graph_integer(VxGraphWriter* w, int64_t value) {
    if (value < -INT64_C(9007199254740991) ||
        value > INT64_C(9007199254740991)) {
        w->status = VX_STATUS_INVALID_ARGUMENT;
        return NULL;
    }
    return cJSON_CreateNumber((double)value);
}

static cJSON* vx_graph_dimension(VxGraphWriter* w,
                                  const VolvoxaiV1GraphDimension* source) {
    cJSON* result = cJSON_CreateObject();
    vx_graph_member(w, result, "min", vx_graph_integer(w, source->field_minimum));
    vx_graph_member(w, result, "max", vx_graph_integer(w, source->field_maximum));
    vx_graph_member(w, result, "multiple_of", vx_graph_integer(w,
        source->field_multiple_of ? source->field_multiple_of : 1));
    return result;
}

static cJSON* vx_graph_tensor(VxGraphWriter* w,
                               const VolvoxaiV1GraphTensorDefinition* source,
                               int is_output) {
    cJSON* result = cJSON_CreateObject();
    cJSON* shape = cJSON_CreateArray();
    const char* dtype = NULL;
    if (!source) { w->status = VX_STATUS_INVALID_ARGUMENT; goto done; }
    switch ((VxDataType)source->field_dtype) {
        case VX_DTYPE_F32: dtype = "float32"; break;
        case VX_DTYPE_I32: dtype = "int32"; break;
        case VX_DTYPE_I8: dtype = "int8"; break;
        case VX_DTYPE_U8: dtype = "uint8"; break;
        default: w->status = VX_STATUS_INVALID_ARGUMENT; goto done;
    }
    if (is_output) vx_graph_member(w, result, "tensor", cJSON_CreateString(
        vx_graph_string(w, &source->field_name, 0)));
    vx_graph_member(w, result, "dtype", cJSON_CreateString(dtype));
    for (size_t i = 0; i < source->field_shape.len; ++i) {
        const VolvoxaiV1GraphAxis* axis = &source->field_shape.data[i];
        if (axis->which_extent == 1)
            vx_graph_append(w, shape, vx_graph_integer(w, axis->field_fixed_extent));
        else if (axis->which_extent == 2)
            vx_graph_append(w, shape, cJSON_CreateString(
                vx_graph_string(w, &axis->field_dimension, 0)));
        else w->status = VX_STATUS_INVALID_ARGUMENT;
    }
done:
    vx_graph_member(w, result, "shape", shape);
    return result;
}

static cJSON* vx_graph_parameter(VxGraphWriter* w,
                                  const VolvoxaiV1NodeParameterValue* source) {
    cJSON* result;
    if (!source) { w->status = VX_STATUS_INVALID_ARGUMENT; return NULL; }
    switch (source->which_value) {
        case 1: return cJSON_CreateBool(source->field_bool_value);
        case 2: return vx_graph_integer(w, source->field_integer_value);
        case 3:
            if (!isfinite(source->field_number_value)) break;
            return cJSON_CreateNumber(source->field_number_value);
        case 4: return cJSON_CreateString(vx_graph_string(w, &source->field_string_value, 1));
        case 5:
            if (!source->field_integer_list) break;
            result = cJSON_CreateArray();
            for (size_t i = 0; i < source->field_integer_list->field_values.len; ++i)
                vx_graph_append(w, result, vx_graph_integer(w,
                    source->field_integer_list->field_values.data[i]));
            return result;
        case 6:
            if (!source->field_shape_list) break;
            result = cJSON_CreateArray();
            for (size_t i = 0; i < source->field_shape_list->field_values.len; ++i) {
                const VolvoxaiV1ShapeExpressionAxis* axis =
                    &source->field_shape_list->field_values.data[i];
                if (axis->which_value == 1)
                    vx_graph_append(w, result, vx_graph_integer(w, axis->field_literal));
                else if (axis->which_value == 2)
                    vx_graph_append(w, result, cJSON_CreateString(
                        vx_graph_string(w, &axis->field_dimension, 0)));
                else w->status = VX_STATUS_INVALID_ARGUMENT;
            }
            return result;
    }
    w->status = VX_STATUS_INVALID_ARGUMENT;
    return NULL;
}

static int vx_graph_node_index(const cJSON* nodes, const char* id) {
    int index = 0;
    for (const cJSON* n = nodes ? nodes->child : NULL; n; n = n->next, ++index) {
        const cJSON* value = cJSON_GetObjectItemCaseSensitive(n, "id");
        if (cJSON_IsString(value) && !strcmp(value->valuestring, id)) return index;
    }
    return -1;
}

static const char* vx_graph_next_id(VxGraphWriter* w, const cJSON* nodes,
                                     const VolvoxaiV1GraphDefinition* definition) {
    char* id = vx_api_scratch_alloc(w->scratch, 48);
    if (!id) { w->status = VX_STATUS_OUT_OF_MEMORY; return NULL; }
    for (uint64_t i = 0;; ++i) {
        int occupied;
        snprintf(id, 48, "node_%" PRIu64, i);
        occupied = vx_graph_node_index(nodes, id) >= 0;
        for (size_t n = 0; !occupied && definition && n < definition->field_nodes.len; ++n) {
            const SynurangLiteBytes* explicit_id = &definition->field_nodes.data[n].field_id;
            occupied = explicit_id->len == strlen(id) && explicit_id->data &&
                !memcmp(explicit_id->data, id, explicit_id->len);
        }
        if (!occupied) return id;
    }
}

static cJSON* vx_graph_node(VxGraphWriter* w,
                             const VolvoxaiV1GraphNodeDefinition* source,
                             const char* default_id) {
    cJSON* result = cJSON_CreateObject();
    cJSON* inputs = cJSON_CreateObject();
    cJSON* outputs = cJSON_CreateObject();
    cJSON* params = cJSON_CreateObject();
    if (!source) { w->status = VX_STATUS_INVALID_ARGUMENT; goto done; }
    vx_graph_member(w, result, "id", cJSON_CreateString(source->field_id.len
        ? vx_graph_string(w, &source->field_id, 0) : default_id));
    vx_graph_member(w, result, "opType", cJSON_CreateString(
        vx_graph_string(w, &source->field_operator_name, 0)));
    for (size_t i = 0; i < source->field_inputs.len; ++i) {
        const VolvoxaiV1GraphInputBinding* input = &source->field_inputs.data[i];
        vx_graph_member(w, inputs, vx_graph_string(w, &input->field_port, 0),
            cJSON_CreateString(vx_graph_string(w, &input->field_tensor_name, 0)));
    }
    for (size_t i = 0; i < source->field_outputs.len; ++i) {
        const VolvoxaiV1GraphOutputDefinition* output = &source->field_outputs.data[i];
        vx_graph_member(w, outputs, vx_graph_string(w, &output->field_port, 0),
            vx_graph_tensor(w, output->field_tensor, 1));
    }
    for (size_t i = 0; i < source->field_parameters.len; ++i) {
        const VolvoxaiV1GraphParameter* p = &source->field_parameters.data[i];
        vx_graph_member(w, params, vx_graph_string(w, &p->field_name, 0),
            vx_graph_parameter(w, p->field_value));
    }
done:
    vx_graph_member(w, result, "inputs", inputs);
    vx_graph_member(w, result, "outputs", outputs);
    vx_graph_member(w, result, "params", params);
    return result;
}

static cJSON* vx_graph_outputs(VxGraphWriter* w,
                                const SynurangLiteBytes* names, size_t count) {
    cJSON* result = cJSON_CreateArray();
    for (size_t i = 0; i < count; ++i)
        vx_graph_append(w, result, cJSON_CreateString(vx_graph_string(w, &names[i], 0)));
    return result;
}

static cJSON* vx_graph_quantization(VxGraphWriter* w,
        const VolvoxaiV1GraphQuantizationReference* refs, size_t count) {
    cJSON* result = cJSON_CreateObject();
    cJSON* tensors = cJSON_CreateObject();
    vx_graph_member(w, result, "format", cJSON_CreateString("volvox-affine-safetensors/v1"));
    for (size_t i = 0; i < count; ++i) {
        const VolvoxaiV1GraphQuantizationReference* ref = &refs[i];
        cJSON* item = cJSON_CreateObject();
        vx_graph_member(w, item, "scheme", cJSON_CreateString(ref->has_axis ? "per_axis" : "per_tensor"));
        if (ref->has_axis) vx_graph_member(w, item, "axis", vx_graph_integer(w, ref->field_axis));
        vx_graph_member(w, item, "scale_tensor", cJSON_CreateString(vx_graph_string(w, &ref->field_scale_tensor, 0)));
        vx_graph_member(w, item, "zero_point_tensor", cJSON_CreateString(vx_graph_string(w, &ref->field_zero_point_tensor, 0)));
        vx_graph_member(w, tensors, vx_graph_string(w, &ref->field_tensor_name, 0), item);
    }
    vx_graph_member(w, result, "tensors", tensors);
    return result;
}

static VxStatus vx_graph_definition_json(VxApiScratch* scratch,
        const VolvoxaiV1GraphDefinition* source, const uint8_t** bytes, size_t* size) {
    VxGraphWriter w = {scratch, VX_STATUS_OK};
    cJSON* root = cJSON_CreateObject();
    cJSON* dimensions = cJSON_CreateObject();
    cJSON* inputs = cJSON_CreateObject();
    cJSON* nodes = cJSON_CreateArray();
    char* json = NULL;
    vx_graph_member(&w, root, "format", cJSON_CreateString("volvox-graph/v1"));
    for (size_t i = 0; i < source->field_dimensions.len; ++i) {
        const VolvoxaiV1GraphDimension* d = &source->field_dimensions.data[i];
        vx_graph_member(&w, dimensions, vx_graph_string(&w, &d->field_name, 0), vx_graph_dimension(&w, d));
    }
    for (size_t i = 0; i < source->field_inputs.len; ++i) {
        const VolvoxaiV1GraphTensorDefinition* t = &source->field_inputs.data[i];
        vx_graph_member(&w, inputs, vx_graph_string(&w, &t->field_name, 0), vx_graph_tensor(&w, t, 0));
    }
    for (size_t i = 0; i < source->field_nodes.len; ++i) {
        const VolvoxaiV1GraphNodeDefinition* n = &source->field_nodes.data[i];
        const char* id = n->field_id.len ? NULL : vx_graph_next_id(&w, nodes, source);
        vx_graph_append(&w, nodes, vx_graph_node(&w, n, id));
    }
    vx_graph_member(&w, root, "dimensions", dimensions);
    vx_graph_member(&w, root, "inputs", inputs);
    vx_graph_member(&w, root, "nodes", nodes);
    vx_graph_member(&w, root, "outputs", vx_graph_outputs(&w, source->field_outputs.data, source->field_outputs.len));
    if (source->field_quantization.len) vx_graph_member(&w, root, "quantization",
        vx_graph_quantization(&w, source->field_quantization.data, source->field_quantization.len));
    if (w.status == VX_STATUS_OK) {
        json = cJSON_PrintUnformatted(root);
        if (!json) w.status = VX_STATUS_OUT_OF_MEMORY;
        else {
            *size = strlen(json);
            void* owned = vx_api_scratch_alloc(scratch, *size);
            if (!owned) w.status = VX_STATUS_OUT_OF_MEMORY;
            else { memcpy(owned, json, *size); *bytes = owned; }
        }
    }
    free(json);
    cJSON_Delete(root);
    return w.status;
}
