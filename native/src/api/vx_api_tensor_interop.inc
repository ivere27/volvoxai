static int vx_api_get_tensor_interop_info(const VolvoxaiV1ExecutionContextRef* request,
    VolvoxaiV1TensorInteropInfo* response, void* user_data) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxReport report = VX_REPORT_INIT;
    VxNativeBuffer device = {0};
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_CONTEXT, request->field_context_id, &lease))
        return vx_api_report_fail(allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED,
            VX_STAGE_INPUT, VX_CODE_HANDLE_DISPOSED, "execution context is not live") ? 0 : -1;
    VxStatus status = vx_execution_context_tensor_interop(lease.pointer,
        &device, &response->field_consumer_stream, &report);

    int ok = vx_api_report_attach(allocator, &response->field_report, &report);
    if (status == VX_STATUS_OK) ok = ok && vx_api_resource_descriptor(allocator, &response->field_device, &device) && synurang_lite_bytes_assign(allocator,
        &response->field_backend, report.backend, strlen(report.backend)) == SYNURANG_LITE_OK;
    vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    vx_api_handle_lease_release(&lease);
    return ok ? 0 : -1;
}

static int vx_api_execute_tensors(const VolvoxaiV1ExecuteTensorsRequest* request,
    VolvoxaiV1TensorBatch* response, void* user_data) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxTensorBinding* bindings = NULL;
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    size_t failed_index = SIZE_MAX;
    int ok = 1;
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_CONTEXT, request->field_context_id, &lease))
        return vx_api_report_fail(allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED,
            VX_STAGE_EXECUTE, VX_CODE_HANDLE_DISPOSED, "execution context is not live") ? 0 : -1;
    VxStatus status = vx_api_bindings(&scratch, request->field_inputs.data,
        request->field_inputs.len, &bindings, &failed_index);
    if (status != VX_STATUS_OK) {
        ok = vx_api_report_binding_fail(allocator, &response->field_report, status,
            VX_STAGE_INPUT, "input buffer descriptor is invalid",
            failed_index < request->field_inputs.len ? &request->field_inputs.data[failed_index] : NULL,
            failed_index);
        goto done;
    }
    VxTensorRunOptions options = {0};
    if (request->field_reuse_inputs.len > SIZE_MAX - request->field_feedback.len) { ok = 0; goto done; }
    options.reference_count = request->field_reuse_inputs.len + request->field_feedback.len;
    if (options.reference_count > SIZE_MAX / sizeof(VxTensorReference)) { ok = 0; goto done; }
    VxTensorReference* refs = options.reference_count ? vx_api_scratch_alloc(&scratch,
        options.reference_count * sizeof(*refs)) : NULL;
    if (options.reference_count && !refs) { ok = 0; goto done; }
    options.references = refs;
    for (size_t i = 0; i < request->field_reuse_inputs.len; i++) {
        refs[i].input_name = vx_api_scratch_cstr(&scratch, &request->field_reuse_inputs.data[i]);
        refs[i].output_name = NULL;
    }
    for (size_t i = 0; i < request->field_feedback.len; i++) {
        size_t index = request->field_reuse_inputs.len + i;
        refs[index].input_name = vx_api_scratch_cstr(&scratch, &request->field_feedback.data[i].field_input_name);
        refs[index].output_name = vx_api_scratch_cstr(&scratch, &request->field_feedback.data[i].field_output_name);
    }
    if (request->field_outputs) {
        options.select_outputs = 1;
        options.output_count = request->field_outputs->field_names.len;
        options.output_names = vx_api_scratch_cstr_array(&scratch,
            request->field_outputs->field_names.data, sizeof(SynurangLiteBytes), options.output_count);
    }
    if (vx_api_scratch_failed(&scratch)) { ok = 0; goto done; }
    status = vx_execution_context_execute_tensors(lease.pointer, bindings,
        request->field_inputs.len, &options, &result, &report);
    ok = vx_api_report_attach(allocator, &response->field_report, &report);
    if (!ok || status != VX_STATUS_OK) goto done;
    size_t count = vx_result_output_count(result);
    for (size_t i = 0; i < count; i++) {
        VolvoxaiV1Tensor* output = volvoxai_v1_tensor_batch_add_outputs(response);
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        VxNativeBuffer memory;
        if (!output || vx_result_tensor_view(result, i, &info, &memory, &report) != VX_STATUS_OK) {
            ok = 0; goto done;
        }
        VxNativeStorage* storage = NULL;
        VxResult* child = vx_result_detach_tensor(result, i, &storage);
        if (!child) { ok = 0; goto done; }
        VxApiBuffer* buffer = vx_api_buffer_create(&memory, storage, child, vx_api_release_result, 0);
        if (!buffer) { vx_result_release(child); ok = 0; goto done; }
        int64_t id = vx_api_buffer_publish(user_data, buffer);
        if (!id) { vx_api_buffer_release(buffer); ok = 0; goto done; }
        if (!vx_api_buffer_tensor(allocator, output, &info, id)) {
            vx_api_handle_remove(user_data, VX_API_HANDLE_BUFFER, id);
            ok = 0; goto done;
        }
    }
done:
    if (!ok) for (size_t i = 0; i < response->field_outputs.len; i++) {
        VolvoxaiV1BufferView* view = response->field_outputs.data[i].field_buffer;
        if (view) {
            vx_api_handle_remove(user_data, VX_API_HANDLE_BUFFER, view->field_buffer_id);
            view->field_buffer_id = 0;
        }
    }
    if (result) vx_result_release(result);
    vx_api_scratch_release(&scratch);
    vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    vx_api_handle_lease_release(&lease);
    return ok ? 0 : -1;
}
