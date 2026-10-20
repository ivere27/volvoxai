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
