/* Retained immutable GraphPlan transactions. Request buffers are never kept. */
static int vx_graph_copy_weight(VolvoxaiV1PlanningWeight* target,
                                 const VolvoxaiV1PlanningWeight* source) {
    target->field_dtype = source->field_dtype;
    if (!vx_plan_bytes(target->_allocator, &target->field_name,
                       source->field_name.data, source->field_name.len)) return 0;
    for (size_t d = 0; d < source->field_shape.len; ++d) {
        int64_t* axis = volvoxai_v1_planning_weight_add_shape(target);
        if (!axis) return 0;
        *axis = source->field_shape.data[d];
    }
    return 1;
}

static int vx_graph_copy_quantization(VolvoxaiV1TensorAffineQuantization* target,
        const VolvoxaiV1TensorAffineQuantization* source) {
    uint8_t* bytes = NULL;
    size_t size = 0;
    SynurangLiteStatus status = volvoxai_v1_tensor_affine_quantization_encode(source, &bytes, &size);
    if (status == SYNURANG_LITE_OK)
        status = volvoxai_v1_tensor_affine_quantization_decode(target, bytes, size);
    if (bytes) source->_allocator->deallocate(source->_allocator->context, bytes);
    return status == SYNURANG_LITE_OK;
}

static int vx_graph_source_metadata(const VxGraphPlanView* view,
                                      VolvoxaiV1GraphPlanningSource* source) {
    VolvoxaiV1GraphPlan tensors;
    int ok = 0;
    volvoxai_v1_graph_plan_init_with_allocator(&tensors, source->_allocator);
    for (size_t i = 0; i < view->stored_weight_count; ++i) {
        const VxPlanningStoredWeight* w = &view->stored_weights[i];
        VolvoxaiV1PlanningWeight* target = volvoxai_v1_graph_planning_source_add_weights(source);
        if (!target || !vx_plan_text(source->_allocator, &target->field_name, w->name)) goto done;
        target->field_dtype = (VolvoxaiV1DataType)w->dtype;
        for (size_t d = 0; d < w->rank; ++d) {
            int64_t* axis = volvoxai_v1_planning_weight_add_shape(target);
            if (!axis) goto done;
            *axis = w->shape[d];
        }
    }
    if (!vx_plan_tensors(source->_allocator, view, &tensors)) goto done;
    for (size_t i = 0; i < tensors.field_tensors.len; ++i) {
        VolvoxaiV1GraphTensor* tensor = &tensors.field_tensors.data[i];
        if (!tensor->field_quantization) continue;
        VolvoxaiV1TensorAffineQuantization* q = volvoxai_v1_graph_planning_source_add_quantization(source);
        if (!q || !vx_plan_bytes(source->_allocator, &q->field_tensor_name,
                                 tensor->field_name.data, tensor->field_name.len)) goto done;
        q->field_parameters = tensor->field_quantization;
        tensor->field_quantization = NULL;
    }
    ok = 1;
done:
    volvoxai_v1_graph_plan_free(&tensors);
    return ok;
}

static int vx_graph_source_document(VolvoxaiV1GraphPlanningSource* source, const cJSON* root) {
    char* json = cJSON_PrintUnformatted(root);
    if (!json) return 0;
    source->which_definition_source = 1;
    int ok = vx_plan_text(source->_allocator, &source->field_graph_document, json);
    free(json);
    return ok;
}

static int vx_api_export_graph_plan(const VolvoxaiV1GraphPlanRef* request,
        VolvoxaiV1ExportedGraphPlan* response, void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* a = response->_allocator;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxGraphPlanView view;
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_GRAPH_PLAN, request->field_graph_plan_id, &lease))
        return vx_api_report_fail(a, &response->field_report, VX_STATUS_HANDLE_DISPOSED,
            VX_STAGE_GRAPH_PLAN_EXPORT, VX_CODE_HANDLE_DISPOSED, "unknown or released graph plan") ? 0 : -1;
    response->field_source = a->allocate(a->context, sizeof(*response->field_source));
    if (!response->field_source) { vx_api_handle_lease_release(&lease); return -1; }
    volvoxai_v1_graph_planning_source_init_with_allocator(response->field_source, a);
    int ok = vx_graph_plan_internal_view(lease.pointer, &view) == VX_STATUS_OK &&
        vx_graph_source_metadata(&view, response->field_source) &&
        vx_graph_source_document(response->field_source, view.logical_graph_document) &&
        vx_api_report_ok(a, &response->field_report, VX_STAGE_GRAPH_PLAN_EXPORT) &&
        vx_plan_apply_lineage(response->field_report, &lease.lineage) &&
        vx_plan_apply_semantic_lineage(response->field_report, &view);
    vx_api_handle_lease_release(&lease);
    return ok ? 0 : -1;
}

static int vx_graph_bytes_equal(const SynurangLiteBytes* a, const SynurangLiteBytes* b) {
    return a->len == b->len && (!a->len || !memcmp(a->data, b->data, a->len));
}

static void vx_graph_remove_weight(VolvoxaiV1GraphPlanningSource* source,
                                    const SynurangLiteBytes* name) {
    for (size_t i = 0; i < source->field_weights.len; ++i) {
        VolvoxaiV1PlanningWeight* weights = source->field_weights.data;
        if (!vx_graph_bytes_equal(&weights[i].field_name, name)) continue;
        volvoxai_v1_planning_weight_free(&weights[i]);
        --source->field_weights.len;
        memmove(&weights[i], &weights[i + 1], (source->field_weights.len - i) * sizeof(*weights));
        memset(&weights[source->field_weights.len], 0, sizeof(*weights));
        return;
    }
}

static void vx_graph_replace_member(VxGraphWriter* w, cJSON* object,
                                      const char* name, cJSON* value) {
    if (object && name) cJSON_DeleteItemFromObjectCaseSensitive(object, name);
    vx_graph_member(w, object, name, value);
}

static VxStatus vx_graph_apply_edits(VxApiScratch* scratch, cJSON* root,
        VolvoxaiV1GraphPlanningSource* source, const VolvoxaiV1EditGraphPlanRequest* request) {
    VxGraphWriter w = {scratch, VX_STATUS_OK};
    cJSON* dimensions = cJSON_GetObjectItemCaseSensitive(root, "dimensions");
    cJSON* inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    cJSON* nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    for (size_t i = 0; i < request->field_edits.len && w.status == VX_STATUS_OK; ++i) {
        const VolvoxaiV1GraphEdit* e = &request->field_edits.data[i];
        const char* name = NULL;
        int index;
        cJSON* node;
        switch (e->which_operation) {
            case 1:
                if (!e->field_set_dimension) goto invalid;
                name = vx_graph_string(&w, &e->field_set_dimension->field_name, 0);
                vx_graph_replace_member(&w, dimensions, name, vx_graph_dimension(&w, e->field_set_dimension));
                break;
            case 2:
                name = vx_graph_string(&w, &e->field_remove_dimension, 0);
                if (name) cJSON_DeleteItemFromObjectCaseSensitive(dimensions, name);
                break;
            case 3:
                if (!e->field_set_input) goto invalid;
                name = vx_graph_string(&w, &e->field_set_input->field_name, 0);
                vx_graph_replace_member(&w, inputs, name, vx_graph_tensor(&w, e->field_set_input, 0));
                break;
            case 4:
                name = vx_graph_string(&w, &e->field_remove_input, 0);
                if (name) cJSON_DeleteItemFromObjectCaseSensitive(inputs, name);
                break;
            case 5: {
                if (!e->field_set_weight) goto invalid;
                if (!vx_graph_string(&w, &e->field_set_weight->field_name, 0)) break;
                vx_graph_remove_weight(source, &e->field_set_weight->field_name);
                VolvoxaiV1PlanningWeight* target = volvoxai_v1_graph_planning_source_add_weights(source);
                if (!target || !vx_graph_copy_weight(target, e->field_set_weight)) w.status = VX_STATUS_OUT_OF_MEMORY;
                break;
            }
            case 6:
                if (vx_graph_string(&w, &e->field_remove_weight, 0)) vx_graph_remove_weight(source, &e->field_remove_weight);
                break;
            case 7:
                if (!e->field_add_node) goto invalid;
                name = e->field_add_node->field_id.len ? NULL : vx_graph_next_id(&w, nodes, NULL);
                vx_graph_append(&w, nodes, vx_graph_node(&w, e->field_add_node, name));
                break;
            case 8:
                if (!e->field_replace_node || !e->field_replace_node->field_node) goto invalid;
                name = vx_graph_string(&w, &e->field_replace_node->field_id, 0);
                if (!name) break;
                index = vx_graph_node_index(nodes, name);
                if (index < 0) goto invalid;
                node = vx_graph_node(&w, e->field_replace_node->field_node, name);
                if (w.status == VX_STATUS_OK && !cJSON_ReplaceItemInArray(nodes, index, node)) w.status = VX_STATUS_OUT_OF_MEMORY;
                if (w.status != VX_STATUS_OK) cJSON_Delete(node);
                break;
            case 9:
                name = vx_graph_string(&w, &e->field_remove_node, 0);
                if (name && (index = vx_graph_node_index(nodes, name)) >= 0) cJSON_DeleteItemFromArray(nodes, index);
                break;
            case 10:
                if (!e->field_select_outputs) goto invalid;
                vx_graph_replace_member(&w, root, "outputs", vx_graph_outputs(&w,
                    e->field_select_outputs->field_names.data, e->field_select_outputs->field_names.len));
                break;
            case 11: {
                const VolvoxaiV1GraphQuantizationReplacement* q = e->field_replace_quantization;
                if (!q) goto invalid;
                cJSON_DeleteItemFromObjectCaseSensitive(root, "quantization");
                if (q->field_references.len) vx_graph_member(&w, root, "quantization",
                    vx_graph_quantization(&w, q->field_references.data, q->field_references.len));
                for (size_t j = 0; j < source->field_quantization.len; ++j)
                    volvoxai_v1_tensor_affine_quantization_free(&source->field_quantization.data[j]);
                source->field_quantization.len = 0;
                for (size_t j = 0; j < q->field_values.len; ++j) {
                    VolvoxaiV1TensorAffineQuantization* target = volvoxai_v1_graph_planning_source_add_quantization(source);
                    if (!target || !vx_graph_copy_quantization(target, &q->field_values.data[j])) {
                        w.status = VX_STATUS_OUT_OF_MEMORY;
                        break;
                    }
                }
                break;
            }
            default: goto invalid;
        }
    }
    return w.status;
invalid:
    return VX_STATUS_INVALID_ARGUMENT;
}

static int vx_api_edit_graph_plan(const VolvoxaiV1EditGraphPlanRequest* request,
        VolvoxaiV1GraphPlanHandle* response, void* user_data) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxApiScratch scratch = VX_API_SCRATCH_INIT;
    VxGraphPlanView view;
    VolvoxaiV1GraphPlanningSource source;
    VolvoxaiV1CreateGraphPlanRequest create;
    VxStatus status = VX_STATUS_OK;
    cJSON* draft = NULL;
    int result = 0;
    volvoxai_v1_graph_planning_source_init(&source);
    volvoxai_v1_create_graph_plan_request_init(&create);
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_GRAPH_PLAN, request->field_graph_plan_id, &lease)) {
        status = VX_STATUS_HANDLE_DISPOSED;
        goto done;
    }
    status = vx_graph_plan_internal_view(lease.pointer, &view);
    if (status != VX_STATUS_OK) goto done;
    draft = cJSON_Duplicate(view.logical_graph_document, 1);
    if (!draft || !vx_graph_source_metadata(&view, &source)) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    status = vx_graph_apply_edits(&scratch, draft, &source, request);
    if (status != VX_STATUS_OK) goto done;
    if (!vx_graph_source_document(&source, draft)) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    create.which_source = 2;
    create.field_graph = &source; /* Borrowed only for the synchronous constructor. */
    result = vx_api_create_graph_plan(&create, response, user_data);
done:
    if (status != VX_STATUS_OK) result = vx_api_report_fail(response->_allocator,
        &response->field_report, status, VX_STAGE_GRAPH_PLAN_EDIT,
        status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
        status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY : VX_CODE_INVALID_ARGUMENT,
        status == VX_STATUS_HANDLE_DISPOSED ? "unknown or released graph plan" :
        "graph edit transaction failed; source plan is unchanged") ? 0 : -1;
    if (response->field_report) response->field_report->field_stage =
        VOLVOXAI_V1_OPERATION_STAGE_GRAPH_PLAN_EDIT;
    cJSON_Delete(draft);
    volvoxai_v1_graph_planning_source_free(&source);
    vx_api_scratch_release(&scratch);
    vx_api_handle_lease_release(&lease);
    return result;
}
