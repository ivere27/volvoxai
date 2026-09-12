/* Generated service handlers project context-owned cache state. */
static VxStatus vx_api_write_decode_cache(const VxDecodeCacheView* view, void* user) {
    VolvoxaiV1DecodeCacheState* response = user;
    response->field_cache_generation = view->cache_generation;
    if (!view->cache) return VX_STATUS_OK;
    const VxPagedKVCache* cache = view->cache;
    response->field_configured = 1;
    response->field_policy = (VolvoxaiV1DecodeCachePolicy)view->policy;
    response->field_page_tokens = (uint32_t)vx_paged_kv_page_tokens(cache);
    response->field_lane_token_capacity = (uint32_t)vx_paged_kv_lane_token_capacity(cache);
    response->field_max_pages = (uint32_t)vx_paged_kv_max_pages(cache);
    response->field_pool_bytes = view->pool_bytes;
    response->field_prefix_snapshot_bytes = view->prefix_snapshot_bytes;
    for (int i = 0; i < vx_paged_kv_lanes(cache); i++) {
        VolvoxaiV1DecodeCacheLane* lane = volvoxai_v1_decode_cache_state_add_lanes(response);
        if (!lane) return VX_STATUS_OUT_OF_MEMORY;
        lane->field_active_length = (uint32_t)vx_paged_kv_lengths(cache)[i];
        lane->field_generation = (uint32_t)vx_paged_kv_lane_generations(cache)[i];
        for (int page = 0; page < vx_paged_kv_pages_per_lane(cache); page++) {
            int32_t* value = volvoxai_v1_decode_cache_lane_add_pages(lane);
            if (!value) return VX_STATUS_OUT_OF_MEMORY;
            *value = vx_paged_kv_physical_page(cache, i, page);
        }
    }
    for (int page = 0; page < vx_paged_kv_max_pages(cache); page++) {
        uint32_t* value = volvoxai_v1_decode_cache_state_add_page_generations(response);
        if (!value) return VX_STATUS_OUT_OF_MEMORY;
        *value = (uint32_t)vx_paged_kv_page_generation(cache, page);
    }
    for (size_t i = 0; i < view->tensor_count; i++) {
        SynurangLiteBytes* value = volvoxai_v1_decode_cache_state_add_tensors(response);
        if (!value || synurang_lite_bytes_assign(response->_allocator, value,
                (const uint8_t*)view->tensors[i], strlen(view->tensors[i])) != SYNURANG_LITE_OK)
            return VX_STATUS_OUT_OF_MEMORY;
    }
    VxPagedKVTelemetry telemetry;
    vx_paged_kv_telemetry(cache, &telemetry);
#define VX_CACHE_FIELD(name) response->field_##name = telemetry.name
    VX_CACHE_FIELD(logical_bytes); VX_CACHE_FIELD(resident_bytes); VX_CACHE_FIELD(reserved_bytes);
    VX_CACHE_FIELD(resident_pages); VX_CACHE_FIELD(reserved_pages); VX_CACHE_FIELD(free_pages);
    VX_CACHE_FIELD(shared_pages); VX_CACHE_FIELD(fragmentation_bytes); VX_CACHE_FIELD(high_water_pages);
    VX_CACHE_FIELD(prefix_hits); VX_CACHE_FIELD(prefix_misses); VX_CACHE_FIELD(copy_on_writes);
    VX_CACHE_FIELD(evictions); VX_CACHE_FIELD(allocation_failures);
#undef VX_CACHE_FIELD
    return VX_STATUS_OK;
}

static int vx_api_decode_cache_call(VxApiRegistry* user_data, int64_t context_id, const VxDecodeCacheCommand* command,
    VolvoxaiV1DecodeCacheState* response) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxReport report = VX_REPORT_INIT;
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_CONTEXT, context_id, &lease))
        return vx_api_report_fail(response->_allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED,
            VX_STAGE_DECODE, VX_CODE_HANDLE_DISPOSED, "unknown or released context") ? 0 : -1;
    VxStatus status = vx_execution_context_decode_cache(lease.pointer, command, vx_api_write_decode_cache, response, &report);
    if (status != VX_STATUS_OK) volvoxai_v1_decode_cache_state_free(response);
    int result = vx_api_report_attach(response->_allocator, &response->field_report, &report) ? 0 : -1;
    vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_configure_decode_cache(const VolvoxaiV1ConfigureDecodeCacheRequest* request,
    VolvoxaiV1DecodeCacheState* response, void* user_data) {
    (void)user_data;
    VxApiScratch scratch = VX_API_SCRATCH_INIT;
    VxDecodeCacheCommand command = {.action = VX_DECODE_CACHE_CONFIGURE,
        .tensor_count = request->field_tensors.len, .page_tokens = request->field_page_tokens,
        .max_pages = request->field_max_pages, .has_max_pages = request->has_max_pages,
        .policy = request->field_policy, .clear_on_recycle = request->field_clear_on_recycle};
    command.tensors = vx_api_scratch_cstr_array(&scratch, request->field_tensors.data,
        sizeof(*request->field_tensors.data), request->field_tensors.len);
    int result = vx_api_scratch_failed(&scratch) ? -1 : vx_api_decode_cache_call(user_data, request->field_context_id, &command, response);
    vx_api_scratch_release(&scratch);
    return result;
}

static int vx_api_get_decode_cache(const VolvoxaiV1ExecutionContextRef* request,
    VolvoxaiV1DecodeCacheState* response, void* user_data) {
    (void)user_data;
    VxDecodeCacheCommand command = {.action = VX_DECODE_CACHE_INSPECT};
    return vx_api_decode_cache_call(user_data, request->field_context_id, &command, response);
}

static int vx_api_publish_decode_prefix(const VolvoxaiV1PublishDecodePrefixRequest* request,
    VolvoxaiV1DecodeCacheState* response, void* user_data) {
    (void)user_data;
    VxApiScratch scratch = VX_API_SCRATCH_INIT;
    VxDecodeCacheCommand command = {.action = VX_DECODE_CACHE_PUBLISH,
        .lane = request->field_lane, .tokens = request->field_tokens,
        .key = vx_api_scratch_cstr(&scratch, &request->field_key)};
    int result = vx_api_scratch_failed(&scratch) ? -1 : vx_api_decode_cache_call(user_data, request->field_context_id, &command, response);
    vx_api_scratch_release(&scratch);
    return result;
}

static int vx_api_reuse_decode_prefix(const VolvoxaiV1ReuseDecodePrefixRequest* request,
    VolvoxaiV1DecodeCacheState* response, void* user_data) {
    (void)user_data;
    VxApiScratch scratch = VX_API_SCRATCH_INIT;
    VxDecodeCacheCommand command = {.action = VX_DECODE_CACHE_REUSE, .lane = request->field_lane,
        .key = vx_api_scratch_cstr(&scratch, &request->field_key)};
    int result = vx_api_scratch_failed(&scratch) ? -1 : vx_api_decode_cache_call(user_data, request->field_context_id, &command, response);
    vx_api_scratch_release(&scratch);
    return result;
}

static int vx_api_release_decode_lane(const VolvoxaiV1DecodeLaneRef* request,
    VolvoxaiV1DecodeCacheState* response, void* user_data) {
    (void)user_data;
    VxDecodeCacheCommand command = {.action = VX_DECODE_CACHE_RELEASE_LANE, .lane = request->field_lane};
    return vx_api_decode_cache_call(user_data, request->field_context_id, &command, response);
}

static int vx_api_evict_decode_prefixes(const VolvoxaiV1EvictDecodePrefixesRequest* request,
    VolvoxaiV1DecodeCacheState* response, void* user_data) {
    (void)user_data;
    VxDecodeCacheCommand command = {.action = VX_DECODE_CACHE_EVICT, .free_pages = request->field_free_pages};
    return vx_api_decode_cache_call(user_data, request->field_context_id, &command, response);
}
