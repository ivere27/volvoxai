/* Independent ordinary-planning reference from frozen source007.
 * It intentionally has no descriptor reuse or temporary proof scope.
 * Compare precommit acceptance and state preservation, not implementation. */
static VxStatus vx_context_preflight_decode_rows_reference(VxExecutionContext* context,
    const int32_t* positions, const VxTensorBinding* inputs, size_t input_count, const T* required_output) {
    if (context->compiled->provider || context->decode_row_mode != VX_DECODE_ROW_REQUIRED ||
        (context->compiled->builtin_backend != VX_PORTABLE_BACKEND_KIND &&
         context->compiled->builtin_backend != VX_BACKEND_KIND_WEBGPU &&
         (!vx_native_gpu_backend(context->compiled->builtin_backend) ||
          !context->engine_state->decode_row_stage_reserved))) return VX_STATUS_OK;
    VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
    int took_model_lock = !volvoxai_engine_model_route_lease_active();
    if (took_model_lock) volvoxai_engine_model_lock();
    unsigned char* dirty = malloc((size_t)g_nt);
    if (!dirty) {
        if (took_model_lock) volvoxai_engine_model_unlock();
        vx_engine_state_scope_leave(scope);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    memcpy(dirty, g_dirty, (size_t)g_nt);
    VxDecodeRowSet saved_rows = g_decode_rows;
    int saved_slots = g_decode_slots, prepared_row = g_hybrid_prepared_row;
    VxStatus status = VX_STATUS_BACKEND_UNSUPPORTED;
    int row = 0;
    memset(&g_decode_rows, 0, sizeof(g_decode_rows));
    if (vx_paged_row_set_init_locked(&g_decode_rows, (int)context->decode_slots, positions) != VX_DECODE_ROW_SET_OK) goto done;
    g_decode_slots = (int)context->decode_slots;
    for (uint32_t slot = 0; slot < context->decode_slots; slot++) if (positions[slot] > row) row = positions[slot];
    memset(g_dirty, 0, (size_t)g_nt);
    if (input_count) {
        for (size_t i = 0; i < input_count; i++) vx_incremental_mark_tensor_locked(t_find(inputs[i].name));
    } else if (context->decode_default_inputs) {
        for (size_t i = 0; i < context->compiled->model->input_count; i++)
            if (context->decode_default_inputs[i]) vx_incremental_mark_tensor_locked(t_find(context->compiled->model->inputs[i].name));
    }
    if (required_output && !vx_incremental_tensor_dirty_before_node_locked(required_output, g_nn)) goto done;
    int accepted = vx_incremental_prepare_hybrid_row_locked(row);
    status = accepted == 1 ? VX_STATUS_OK : accepted < 0 ? VX_STATUS_EXECUTION_FAILED : VX_STATUS_BACKEND_UNSUPPORTED;
done:
    memcpy(g_dirty, dirty, (size_t)g_nt);
    free(dirty);
    vx_decode_row_set_dispose(&g_decode_rows);
    g_decode_rows = saved_rows; g_decode_slots = saved_slots; g_hybrid_prepared_row = prepared_row;
    if (took_model_lock) volvoxai_engine_model_unlock();
    vx_engine_state_scope_leave(scope);
    return status;
}
