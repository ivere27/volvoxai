/* Full-only graph composition. C emits ordinary logical operators, so LoRA
 * uses the same planner, backward kernels and optimizer as any other graph. */
#include "vx_api_authoring.h"

static const VolvoxaiV1GraphTensor* vx_lora_tensor(const VolvoxaiV1GraphPlan* plan,
                                                    const char* name) {
    for (size_t i = 0; name && i < plan->field_tensors.len; ++i) {
        const VolvoxaiV1GraphTensor* tensor = &plan->field_tensors.data[i];
        if (tensor->field_name.len == strlen(name) &&
            !memcmp(tensor->field_name.data, name, tensor->field_name.len)) return tensor;
    }
    return NULL;
}

static const char* vx_lora_name(VxGraphWriter* w, const char* prefix, const char* suffix) {
    size_t left = strlen(prefix), right = strlen(suffix);
    if (left > SIZE_MAX - right - 1u) { w->status = VX_STATUS_INVALID_ARGUMENT; return NULL; }
    char* result = vx_api_scratch_alloc(w->scratch, left + right + 1u);
    if (!result) { w->status = VX_STATUS_OUT_OF_MEMORY; return NULL; }
    memcpy(result, prefix, left); memcpy(result + left, suffix, right + 1u);
    return result;
}

static void vx_lora_node(VxGraphWriter* w, cJSON* nodes, const char* id,
        const char* op, const char* const* ports, const char* const* tensors,
        size_t input_count, const char* output_name, const VolvoxaiV1GraphTensor* input,
        int64_t width, const char* layout) {
    cJSON* node = cJSON_CreateObject();
    cJSON* inputs = cJSON_CreateObject();
    cJSON* outputs = cJSON_CreateObject();
    cJSON* params = cJSON_CreateObject();
    vx_graph_member(w, node, "id", cJSON_CreateString(id));
    vx_graph_member(w, node, "opType", cJSON_CreateString(op));
    for (size_t i = 0; i < input_count; ++i)
        vx_graph_member(w, inputs, ports[i], cJSON_CreateString(tensors[i]));
    VolvoxaiV1GraphTensorDefinition output;
    volvoxai_v1_graph_tensor_definition_init(&output);
    output.field_name.data = (uint8_t*)output_name;
    output.field_name.len = output_name ? strlen(output_name) : 0;
    output.field_dtype = VOLVOXAI_V1_DATA_TYPE_F32;
    output.field_shape.len = input->field_shape.len;
    output.field_shape.data = vx_api_scratch_alloc(w->scratch,
        output.field_shape.len * sizeof(*output.field_shape.data));
    if (!output.field_shape.data) w->status = VX_STATUS_OUT_OF_MEMORY;
    else {
        memcpy(output.field_shape.data, input->field_shape.data,
            output.field_shape.len * sizeof(*output.field_shape.data));
        VolvoxaiV1GraphAxis* last = &output.field_shape.data[output.field_shape.len - 1u];
        /* Borrowed symbolic strings in the leading axes remain untouched. */
        volvoxai_v1_graph_axis_init(last);
        last->which_extent = 1; last->field_fixed_extent = width;
        vx_graph_member(w, outputs, "out", vx_graph_tensor(w, &output, 1));
    }
    if (layout) vx_graph_member(w, params, "weight_layout", cJSON_CreateString(layout));
    vx_graph_member(w, node, "inputs", inputs);
    vx_graph_member(w, node, "outputs", outputs);
    vx_graph_member(w, node, "params", params);
    vx_graph_append(w, nodes, node);
}

static VxStatus vx_lora_weight(VolvoxaiV1GraphPlanningSource* source,
        VolvoxaiV1LoraLinearPlan* response, const char* name, const int64_t* shape,
        size_t rank, const VolvoxaiV1TensorInitializer* initializer,
        const float* constant) {
    float* values = NULL;
    size_t count = 1;
    VxStatus status = VX_STATUS_OK;
    if (!name) return VX_STATUS_OUT_OF_MEMORY;
    if (!constant) status = vx_api_initialize_f32(shape, rank, initializer, &values, &count);
    if (status != VX_STATUS_OK) return status;
    const SynurangLiteAllocator* a = response->_allocator;
    VolvoxaiV1Tensor* tensor = volvoxai_v1_lora_linear_plan_add_initialized_weights(response);
    VolvoxaiV1PlanningWeight* descriptor = volvoxai_v1_graph_planning_source_add_weights(source);
    if (!tensor || !descriptor || !vx_plan_text(a, &tensor->field_name, name) ||
        !vx_plan_text(source->_allocator, &descriptor->field_name, name) ||
        !vx_plan_bytes(a, &tensor->field_inline, constant ? constant : values, count * sizeof(float))) {
        status = VX_STATUS_OUT_OF_MEMORY; goto done;
    }
    tensor->field_dtype = descriptor->field_dtype = VOLVOXAI_V1_DATA_TYPE_F32;
    tensor->which_payload = 5;
    for (size_t i = 0; i < rank; ++i) {
        int64_t* axis = volvoxai_v1_tensor_add_shape(tensor);
        int64_t* weight_axis = volvoxai_v1_planning_weight_add_shape(descriptor);
        if (!axis || !weight_axis) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
        *axis = *weight_axis = shape[i];
    }
done:
    free(values);
    return status;
}

int vx_api_build_lora_linear(const VolvoxaiV1BuildLoraLinearRequest* request,
        VolvoxaiV1LoraLinearPlan* response, void* user_data) {
    const SynurangLiteAllocator* a = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_INIT;
    VxGraphWriter w = {&scratch, VX_STATUS_OK};
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxGraphPlanView view;
    VolvoxaiV1GraphPlanningSource source;
    VolvoxaiV1GraphPlan tensors;
    VolvoxaiV1GraphPlanHandle created;
    cJSON* draft = NULL;
    int result = 0;
    volvoxai_v1_graph_planning_source_init(&source);
    volvoxai_v1_graph_plan_init(&tensors);
    volvoxai_v1_graph_plan_handle_init_with_allocator(&created, a);
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_GRAPH_PLAN, request->field_graph_plan_id, &lease)) {
        w.status = VX_STATUS_HANDLE_DISPOSED; goto done;
    }
    w.status = vx_graph_plan_internal_view(lease.pointer, &view);
    if (w.status != VX_STATUS_OK) goto done;
    if (!vx_graph_source_metadata(&view, &source) || !vx_plan_tensors(tensors._allocator, &view, &tensors) ||
        !(draft = cJSON_Duplicate(view.logical_graph_document, 1))) { w.status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    const char* input_name = vx_graph_string(&w, &request->field_input_name, 0);
    const char* weight_name = vx_graph_string(&w, &request->field_base_weight_name, 0);
    const char* bias_name = request->field_bias_name.len ? vx_graph_string(&w, &request->field_bias_name, 0) : NULL;
    const char* op = request->field_base_operator.len ? vx_graph_string(&w, &request->field_base_operator, 0) : "Linear";
    VxOperatorKind operator_kind = vx_operator_kind_from_name(op);
    if (w.status != VX_STATUS_OK) goto done;
    const VolvoxaiV1GraphTensor* input = vx_lora_tensor(&tensors, input_name);
    const VolvoxaiV1GraphTensor* weight = vx_lora_tensor(&tensors, weight_name);
    if (!input || !input->field_shape.len || input->field_dtype != VOLVOXAI_V1_DATA_TYPE_F32 ||
        !weight || weight->field_kind != VOLVOXAI_V1_GRAPH_TENSOR_KIND_WEIGHT ||
        weight->field_dtype != VOLVOXAI_V1_DATA_TYPE_F32 || weight->field_shape.len != 2 ||
        !request->field_rank || request->field_rank > INT_MAX ||
        (operator_kind != VX_OP_LINEAR && operator_kind != VX_OP_MATMUL)) goto invalid;
    int64_t input_width = weight->field_shape.data[request->field_transpose_base_weight ? 1 : 0].field_fixed_extent;
    int64_t output_width = weight->field_shape.data[request->field_transpose_base_weight ? 0 : 1].field_fixed_extent;
    if (input_width <= 0 || output_width <= 0 ||
        (uint64_t)input_width * request->field_rank > UINT32_MAX / sizeof(float) ||
        (uint64_t)output_width * request->field_rank > UINT32_MAX / sizeof(float)) goto invalid;
    if (bias_name) {
        const VolvoxaiV1GraphTensor* bias = vx_lora_tensor(&tensors, bias_name);
        if (!bias || bias->field_dtype != VOLVOXAI_V1_DATA_TYPE_F32 || bias->field_shape.len != 1 ||
            bias->field_shape.data[0].which_extent != 1 || bias->field_shape.data[0].field_fixed_extent != output_width) goto invalid;
    }
    double scaling = request->which_scaling == 6 ? request->field_alpha / request->field_rank :
        request->which_scaling == 7 ? request->field_scale : 1;
    if ((request->which_scaling && request->which_scaling != 6 && request->which_scaling != 7) ||
        !isfinite(scaling) || scaling <= 0 || !isfinite((float)scaling) || (float)scaling <= 0) goto invalid;
    response->field_scale = (float)scaling;
    cJSON* nodes = cJSON_GetObjectItemCaseSensitive(draft, "nodes");
    char generated_name[64];
    snprintf(generated_name, sizeof(generated_name), "lora_linear_%d", cJSON_GetArraySize(nodes));
    const char* name = request->field_name.len ? vx_graph_string(&w, &request->field_name, 0) : generated_name;
    if (!name) goto done;
    const char* a_name = vx_lora_name(&w, name, ".lora_a");
    const char* b_name = vx_lora_name(&w, name, ".lora_b");
    const char* scale_name = vx_lora_name(&w, name, ".lora_scale");
    const char* base = vx_lora_name(&w, name, ".base_out");
    const char* hidden = vx_lora_name(&w, name, ".lora_hidden");
    const char* delta = vx_lora_name(&w, name, ".lora_delta");
    const char* scaled = vx_lora_name(&w, name, ".lora_scaled");
    const char* output = vx_lora_name(&w, name, ".out");
    if (w.status != VX_STATUS_OK) goto done;
    const char* derived[] = {a_name,b_name,scale_name,base,hidden,delta,scaled,output};
    for (size_t i = 0; i < sizeof(derived)/sizeof(derived[0]); ++i)
        if (strlen(derived[i]) >= 128u || vx_lora_tensor(&tensors, derived[i])) goto invalid;
    const char* dense_ports[] = {"input", "weight", "bias"};
    const char* binary_ports[] = {"a", "b"};
    const char* bindings[] = {input_name,weight_name,bias_name};
    vx_lora_node(&w, nodes, vx_lora_name(&w,name,".base"), op, dense_ports, bindings,
        bias_name ? 3u : 2u, base, input, output_width, request->field_transpose_base_weight ? "dout_din" : "din_dout");
    bindings[1] = a_name;
    vx_lora_node(&w, nodes, vx_lora_name(&w,name,".lora_down"), "MatMul", dense_ports, bindings, 2u,
        hidden, input, request->field_rank, "din_dout");
    bindings[0] = hidden; bindings[1] = b_name;
    vx_lora_node(&w, nodes, vx_lora_name(&w,name,".lora_up"), "MatMul", dense_ports, bindings, 2u,
        delta, input, output_width, "din_dout");
    bindings[0] = delta; bindings[1] = scale_name;
    vx_lora_node(&w, nodes, vx_lora_name(&w,name,".lora_scale"), "Mul", binary_ports, bindings, 2u,
        scaled, input, output_width, NULL);
    bindings[0] = base; bindings[1] = scaled;
    vx_lora_node(&w, nodes, name, "Add", binary_ports, bindings, 2u, output, input, output_width, NULL);
    if (request->field_select_output) {
        SynurangLiteBytes selection = {0}; selection.data = (uint8_t*)output; selection.len = strlen(output);
        vx_graph_replace_member(&w, draft, "outputs", vx_graph_outputs(&w, &selection, 1u));
    }
    if (w.status != VX_STATUS_OK) goto done;
    VolvoxaiV1TensorInitializer default_a;
    VolvoxaiV1XavierInitializer xavier;
    volvoxai_v1_tensor_initializer_init(&default_a); volvoxai_v1_xavier_initializer_init(&xavier);
    default_a.which_distribution = 4; default_a.field_xavier_uniform = &xavier;
    int64_t shape_a[] = {input_width, request->field_rank};
    int64_t shape_b[] = {request->field_rank, output_width};
    int64_t scalar[] = {1};
    w.status = vx_lora_weight(&source, response, a_name, shape_a, 2u,
        request->field_a_initializer ? request->field_a_initializer : &default_a, NULL);
    if (w.status != VX_STATUS_OK) goto done;
    w.status = vx_lora_weight(&source, response, b_name, shape_b, 2u, request->field_b_initializer, NULL);
    if (w.status != VX_STATUS_OK) goto done;
    w.status = vx_lora_weight(&source, response, scale_name, scalar, 1u, NULL, &response->field_scale);
    if (w.status != VX_STATUS_OK) goto done;
    for (size_t i = 0; i < 2; ++i) {
        SynurangLiteBytes* trainable = volvoxai_v1_lora_linear_plan_add_trainable_names(response);
        if (!trainable || !vx_plan_text(a, trainable, i ? b_name : a_name)) { w.status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    }
    if (!vx_plan_text(a, &response->field_output_name, output) || !vx_graph_source_document(&source, draft)) {
        w.status = VX_STATUS_OUT_OF_MEMORY; goto done;
    }
    VolvoxaiV1CreateGraphPlanRequest create;
    volvoxai_v1_create_graph_plan_request_init(&create);
    create.which_source = 2; create.field_graph = &source;
    result = vx_api_create_graph_plan(&create, &created, user_data);
    if (!result && created.field_plan && (!created.field_plan->field_shape_domain ||
        created.field_plan->field_shape_domain->which_outcome != 1)) w.status = VX_STATUS_INVALID_GRAPH;
    goto done;
invalid:
    w.status = VX_STATUS_INVALID_ARGUMENT;
done:
    if (w.status != VX_STATUS_OK || !created.field_graph_plan_id || result) {
        if (created.field_graph_plan_id)
            vx_api_handle_remove(user_data, VX_API_HANDLE_GRAPH_PLAN, created.field_graph_plan_id);
        volvoxai_v1_lora_linear_plan_free(response);
        volvoxai_v1_lora_linear_plan_init_with_allocator(response, a);
    } else {
        response->field_graph_plan_id = created.field_graph_plan_id;
        response->field_plan = created.field_plan; created.field_plan = NULL;
    }
    response->field_report = created.field_report; created.field_report = NULL;
    if (w.status != VX_STATUS_OK) result = vx_api_report_fail(a, &response->field_report, w.status,
        VX_STAGE_LORA_AUTHOR, w.status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
        w.status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY : VX_CODE_INVALID_ARGUMENT,
        "LoRA construction failed; source plan is unchanged") ? 0 : -1;
    if (response->field_report) response->field_report->field_stage = VOLVOXAI_V1_OPERATION_STAGE_LORA_AUTHOR;
    volvoxai_v1_graph_plan_handle_free(&created);
    volvoxai_v1_graph_plan_free(&tensors);
    volvoxai_v1_graph_planning_source_free(&source);
    cJSON_Delete(draft);
    vx_api_scratch_release(&scratch);
    vx_api_handle_lease_release(&lease);
    return result;
}

int vx_api_build_routed_adapter(const VolvoxaiV1BuildRoutedAdapterRequest* request,
        VolvoxaiV1RoutedAdapterPlan* response, void* user_data) {
    const SynurangLiteAllocator* a = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_INIT;
    VxGraphWriter w = {&scratch, VX_STATUS_OK};
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxGraphPlanView view;
    VolvoxaiV1GraphPlanningSource source;
    VolvoxaiV1GraphPlan tensors;
    VolvoxaiV1GraphPlanHandle created;
    cJSON* draft = NULL;
    int result = 0;
    volvoxai_v1_graph_planning_source_init(&source);
    volvoxai_v1_graph_plan_init(&tensors);
    volvoxai_v1_graph_plan_handle_init_with_allocator(&created, a);
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_GRAPH_PLAN, request->field_graph_plan_id, &lease)) {
        w.status = VX_STATUS_HANDLE_DISPOSED; goto done;
    }
    w.status = vx_graph_plan_internal_view(lease.pointer, &view);
    if (w.status != VX_STATUS_OK) goto done;
    if (!vx_graph_source_metadata(&view, &source) || !vx_plan_tensors(tensors._allocator, &view, &tensors) ||
        !(draft = cJSON_Duplicate(view.logical_graph_document, 1))) { w.status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    const char* input_name = vx_graph_string(&w, &request->field_input_name, 0);
    const char* down_name = vx_graph_string(&w, &request->field_down_weight_name, 0);
    const char* up_name = vx_graph_string(&w, &request->field_up_weight_name, 0);
    const char* indices = vx_graph_string(&w, &request->field_route_indices_name, 0);
    const char* routes = vx_graph_string(&w, &request->field_route_weights_name, 0);
    const char* down_bias = request->field_down_bias_name.len ? vx_graph_string(&w, &request->field_down_bias_name, 0) : NULL;
    const char* up_bias = request->field_up_bias_name.len ? vx_graph_string(&w, &request->field_up_bias_name, 0) : NULL;
    if (w.status != VX_STATUS_OK) goto done;
    const VolvoxaiV1GraphTensor* input = vx_lora_tensor(&tensors, input_name);
    const VolvoxaiV1GraphTensor* down = vx_lora_tensor(&tensors, down_name);
    const VolvoxaiV1GraphTensor* up = vx_lora_tensor(&tensors, up_name);
    if (!input || !input->field_shape.len || input->field_dtype != VOLVOXAI_V1_DATA_TYPE_F32 ||
        !down || !up || down->field_shape.len != 3u || up->field_shape.len != 3u ||
        down->field_dtype != VOLVOXAI_V1_DATA_TYPE_F32 || up->field_dtype != VOLVOXAI_V1_DATA_TYPE_F32 ||
        down->field_kind != VOLVOXAI_V1_GRAPH_TENSOR_KIND_WEIGHT || up->field_kind != VOLVOXAI_V1_GRAPH_TENSOR_KIND_WEIGHT ||
        !isfinite(request->field_dropout) || request->field_dropout < 0 || request->field_dropout >= 1 ||
        (float)request->field_dropout >= 1.0f) goto invalid;
    for (size_t axis = 0; axis < 3; ++axis)
        if (down->field_shape.data[axis].which_extent != 1 || up->field_shape.data[axis].which_extent != 1) goto invalid;
    int64_t width = down->field_shape.data[1].field_fixed_extent;
    int64_t hidden_width = down->field_shape.data[2].field_fixed_extent;
    if (down->field_shape.data[0].field_fixed_extent != up->field_shape.data[0].field_fixed_extent ||
        hidden_width != up->field_shape.data[1].field_fixed_extent || width != up->field_shape.data[2].field_fixed_extent) goto invalid;
    cJSON* nodes = cJSON_GetObjectItemCaseSensitive(draft, "nodes");
    char generated_name[64];
    snprintf(generated_name, sizeof(generated_name), "routed_adapter_%d", cJSON_GetArraySize(nodes));
    const char* name = request->field_name.len ? vx_graph_string(&w, &request->field_name, 0) : generated_name;
    if (!name) goto done;
    const char* reduced = vx_lora_name(&w, name, ".down.out");
    const char* activated = vx_lora_name(&w, name, ".gelu");
    const char* expanded = vx_lora_name(&w, name, ".up.out");
    const char* dropped = vx_lora_name(&w, name, ".dropout.out");
    const char* output = vx_lora_name(&w, name, ".out");
    if (w.status != VX_STATUS_OK) goto done;
    const char* names[] = {reduced, activated, expanded, dropped, output};
    for (size_t i = 0; i < 5u; ++i) if (strlen(names[i]) >= 128u || vx_lora_tensor(&tensors, names[i])) goto invalid;
    const char* ports[] = {"input", "expert_weight", "route_indices", "route_weights", "expert_bias"};
    const char* bindings[] = {input_name, down_name, indices, routes, down_bias};
    vx_lora_node(&w, nodes, vx_lora_name(&w, name, ".down"), "MoELinear", ports, bindings,
        down_bias ? 5u : 4u, reduced, input, hidden_width, NULL);
    const char* unary[] = {"input"}; const char* bound[] = {reduced};
    vx_lora_node(&w, nodes, activated, "GELU", unary, bound, 1u, activated, input, hidden_width, NULL);
    bindings[0] = activated; bindings[1] = up_name; bindings[4] = up_bias;
    vx_lora_node(&w, nodes, vx_lora_name(&w, name, ".up"), "MoELinear", ports, bindings,
        up_bias ? 5u : 4u, expanded, input, width, NULL);
    if (request->field_dropout > 0) {
        bound[0] = expanded;
        vx_lora_node(&w, nodes, vx_lora_name(&w, name, ".dropout"), "Dropout", unary, bound,
            1u, dropped, input, width, NULL);
        cJSON* params = cJSON_GetObjectItemCaseSensitive(cJSON_GetArrayItem(nodes, cJSON_GetArraySize(nodes) - 1), "params");
        vx_graph_member(&w, params, "ratio", cJSON_CreateNumber(request->field_dropout));
        vx_graph_member(&w, params, "seed", cJSON_CreateNumber(request->field_seed));
    }
    const char* binary[] = {"a", "b"}; const char* residual[] = {input_name, request->field_dropout > 0 ? dropped : expanded};
    vx_lora_node(&w, nodes, name, "Add", binary, residual, 2u, output, input, width, NULL);
    if (request->field_select_output) {
        SynurangLiteBytes selection = {0}; selection.data = (uint8_t*)output; selection.len = strlen(output);
        vx_graph_replace_member(&w, draft, "outputs", vx_graph_outputs(&w, &selection, 1u));
    }
    if (w.status != VX_STATUS_OK) goto done;
    if (!vx_plan_text(a, &response->field_output_name, output) || !vx_graph_source_document(&source, draft)) {
        w.status = VX_STATUS_OUT_OF_MEMORY; goto done;
    }
    VolvoxaiV1CreateGraphPlanRequest create;
    volvoxai_v1_create_graph_plan_request_init(&create);
    create.which_source = 2; create.field_graph = &source;
    result = vx_api_create_graph_plan(&create, &created, user_data);
    if (!result && created.field_plan && (!created.field_plan->field_shape_domain ||
        created.field_plan->field_shape_domain->which_outcome != 1)) w.status = VX_STATUS_INVALID_GRAPH;
    goto done;
invalid:
    w.status = VX_STATUS_INVALID_ARGUMENT;
done:
    if (w.status != VX_STATUS_OK || !created.field_graph_plan_id || result) {
        if (created.field_graph_plan_id) vx_api_handle_remove(user_data, VX_API_HANDLE_GRAPH_PLAN, created.field_graph_plan_id);
        volvoxai_v1_routed_adapter_plan_free(response);
        volvoxai_v1_routed_adapter_plan_init_with_allocator(response, a);
    } else {
        response->field_graph_plan_id = created.field_graph_plan_id;
        response->field_plan = created.field_plan; created.field_plan = NULL;
    }
    response->field_report = created.field_report; created.field_report = NULL;
    if (w.status != VX_STATUS_OK) result = vx_api_report_fail(a, &response->field_report, w.status,
        VX_STAGE_GRAPH_PLAN_EDIT, w.status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
        w.status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY : VX_CODE_INVALID_ARGUMENT,
        "routed adapter construction failed; source plan is unchanged") ? 0 : -1;
    if (response->field_report) response->field_report->field_stage = VOLVOXAI_V1_OPERATION_STAGE_GRAPH_PLAN_EDIT;
    volvoxai_v1_graph_plan_handle_free(&created);
    volvoxai_v1_graph_plan_free(&tensors);
    volvoxai_v1_graph_planning_source_free(&source);
    cJSON_Delete(draft);
    vx_api_scratch_release(&scratch);
    vx_api_handle_lease_release(&lease);
    return result;
}
