/* Immutable full banks stay on the host. Each context owns fixed device planes
 * and replaces complete experts between ordered row launches. No weight tensor
 * descriptor, logical shape or source bytes are changed by residency. */
static uint32_t cuda_weight_storage_bytes(uint32_t dtype) {
    return dtype == VX_DTYPE_F32 ? 4u :
        (dtype == VX_DTYPE_F16 || dtype == VX_DTYPE_BF16 ? 2u : 0u);
}

static int cuda_moe_cache_bounds(uint32_t experts, uint32_t d_in,
        uint32_t hidden, uint32_t d_out, uint32_t maximum_top_k,
        uint32_t slots, const uint32_t dtype[3], int has_bias, uint64_t* host_bytes,
        uint64_t* device_bytes) {
    if (!host_bytes || !device_bytes) return 0;
    *host_bytes = *device_bytes = 0;
    if (!experts || experts > 16777216u || !d_in || !d_out || !slots ||
        slots > experts || !maximum_top_k || maximum_top_k > slots ||
        hidden > 8192u) return 0;
    uint64_t first = (uint64_t)d_in * (hidden ? hidden : d_out);
    uint64_t last = hidden ? (uint64_t)hidden * d_out : 0;
    if (first > UINT32_MAX / slots || last > UINT32_MAX / slots ||
        (has_bias && d_out > SIZE_MAX / sizeof(float) / experts)) return 0;
    uint64_t elements[3] = {first, hidden ? first : 0, last}, device = 0;
    for (unsigned i = 0; i < 3; ++i) if (elements[i]) {
        uint32_t size = cuda_weight_storage_bytes(dtype[i]);
        if (!size || elements[i] > SIZE_MAX / size / experts) return 0;
        device += elements[i] * slots * size;
        device = (device + 3u) & ~UINT64_C(3);
    }
    if (has_bias) device += (uint64_t)d_out * slots * sizeof(float);
    device += (uint64_t)maximum_top_k * sizeof(float);
    if (device > SIZE_MAX) return 0;
    uint64_t host = sizeof(CudaMoeCache) + (uint64_t)slots *
        (sizeof(uint64_t) + sizeof(uint32_t) + sizeof(unsigned char)) +
        (uint64_t)maximum_top_k * sizeof(float);
    if (host > SIZE_MAX) return 0;
    *host_bytes = host;
    *device_bytes = device;
    return 1;
}

int cuda_graph_moe_cache_linear_storage_bounds(uint32_t experts, uint32_t d_in,
        uint32_t d_out, uint32_t maximum_top_k, uint32_t resident_slots,
        uint32_t weight_dtype, int has_bias, uint64_t* host_bytes, uint64_t* device_bytes) {
    const uint32_t dtype[3] = {weight_dtype, VX_DTYPE_F32, VX_DTYPE_F32};
    return cuda_moe_cache_bounds(experts,d_in,0,d_out,maximum_top_k,
        resident_slots,dtype,has_bias != 0,host_bytes,device_bytes);
}

int cuda_graph_moe_cache_gated_storage_bounds(uint32_t experts, uint32_t d_in,
        uint32_t hidden, uint32_t d_out, uint32_t maximum_top_k, uint32_t resident_slots,
        uint32_t gate_dtype, uint32_t up_dtype, uint32_t down_dtype,
        uint64_t* host_bytes, uint64_t* device_bytes) {
    const uint32_t dtype[3] = {gate_dtype,up_dtype,down_dtype};
    return hidden && cuda_moe_cache_bounds(experts,d_in,hidden,d_out,maximum_top_k,
        resident_slots,dtype,0,host_bytes,device_bytes);
}

int cuda_graph_moe_cache_linear_bounds(uint32_t experts, uint32_t d_in,
        uint32_t d_out, uint32_t maximum_top_k, uint32_t resident_slots,
        int has_bias, uint64_t* host_bytes, uint64_t* device_bytes) {
    return cuda_graph_moe_cache_linear_storage_bounds(experts,d_in,d_out,
        maximum_top_k,resident_slots,VX_DTYPE_F32,has_bias,host_bytes,device_bytes);
}

int cuda_graph_moe_cache_gated_bounds(uint32_t experts, uint32_t d_in,
        uint32_t hidden, uint32_t d_out, uint32_t maximum_top_k,
        uint32_t resident_slots, uint64_t* host_bytes, uint64_t* device_bytes) {
    return cuda_graph_moe_cache_gated_storage_bounds(experts,d_in,hidden,d_out,
        maximum_top_k,resident_slots,VX_DTYPE_F32,VX_DTYPE_F32,VX_DTYPE_F32,host_bytes,device_bytes);
}

static CudaMoeCache* cuda_moe_cache_find(const void* primary) {
    CudaContextState* state = cuda_context_state_get();
    for (CudaMoeCache* cache = state ? state->moe_caches : NULL;
            cache; cache = cache->next)
        if (cache->source[0] == primary) return cache;
    return NULL;
}

int cuda_graph_moe_cache_source(const void* base, size_t bytes) {
    if (!base || !bytes) return 0;
    CudaContextState* state = cuda_context_state_get();
    for (const CudaMoeCache* cache = state ? state->moe_caches : NULL;
            cache; cache = cache->next) {
        for (unsigned i = 0; i < 3; ++i)
            if (base == cache->source[i] && bytes ==
                cache->expert_bytes[i] * cache->experts) return 1;
        if (cache->bias && base == cache->bias &&
            bytes == cache->bias_bytes * cache->experts) return 1;
    }
    return 0;
}

void cuda_graph_moe_cache_bytes(uint64_t* host_bytes, uint64_t* device_bytes) {
    if (host_bytes) *host_bytes = 0;
    if (device_bytes) *device_bytes = 0;
    CudaContextState* state = cuda_context_state_get();
    for (const CudaMoeCache* cache = state ? state->moe_caches : NULL;
            cache; cache = cache->next) {
        if (host_bytes) *host_bytes += cache->host_bytes;
        if (device_bytes) *device_bytes += cache->device_bytes;
    }
}

uint64_t cuda_graph_moe_cache_staging_bound(void) {
    /* Slab growth allocates the replacement before freeing the drained old
     * allocation. Charge both even though only one is retained afterwards. */
    return UINT64_C(2) * VX_CUDA_TRANSFER_LIMIT_BYTES;
}

static void cuda_moe_cache_inventory(void) {
    CudaContextState* state = cuda_context_state_get();
    VxEngineState* owner = vx_engine_state_current();
    for (const CudaMoeCache* cache = state ? state->moe_caches : NULL;
            cache; cache = cache->next) {
        vx_memory_record(&owner->memory_observer, VX_MEMORY_HOST_METADATA,
            (uint64_t)(uintptr_t)cache, cache->host_bytes, VX_TRACE_MEMORY_ACTION_EXISTING);
        vx_memory_record(&owner->memory_observer, VX_MEMORY_CUDA_GRAPH,
            (uint64_t)cache->allocation, cache->device_bytes, VX_TRACE_MEMORY_ACTION_EXISTING);
    }
}

/* Keep failed Driver allocations attached to their owner. The ordinary CUDA
 * teardown then retries or quarantines them with its device reference. */
static int cuda_moe_cache_release(void) {
    CudaContextState* state = cuda_context_state_get();
    CudaContextGuard guard;
    if (!state || !state->moe_caches) return 1;
    if (!cuda_context_enter(&guard)) return 0;
    if (!cuda_ok(cuda_trace_wait_stream(state->submission.stream),
                 "cuStreamSynchronize(expert cache)")) {
        (void)cuda_context_leave(&guard);
        return 0;
    }
    while (state->moe_caches) {
        CudaMoeCache* cache = state->moe_caches;
        if (cache->allocation && !cuda_ok(cuda_memory_free(cache->allocation),
                                          "cuMemFree(expert cache)")) {
            (void)cuda_context_leave(&guard);
            return 0;
        }
        state->moe_caches = cache->next;
        vx_memory_record(&vx_engine_state_current()->memory_observer,
            VX_MEMORY_HOST_METADATA, (uint64_t)(uintptr_t)cache, 0,
            VX_TRACE_MEMORY_ACTION_FREE);
        free(cache);
    }
    return cuda_context_leave(&guard);
}

static int cuda_moe_cache_register(const void* first, const void* second,
        const void* third, const float* bias, const uint32_t dtype[3], uint32_t experts,
        uint32_t d_in, uint32_t hidden, uint32_t d_out,
        uint32_t maximum_top_k, uint32_t slots) {
    CudaContextState* state = cuda_context_state_get();
    uint64_t host_bytes, device_bytes, registered_device = 0;
    CudaContextGuard guard;
    if (!state || !state->ready || !first || (hidden && (!second || !third)) ||
        state->domain_enforced || state->invariant_preload_complete ||
        state->forward_scope_active ||
        !cuda_moe_cache_bounds(experts, d_in, hidden, d_out, maximum_top_k,
            slots, dtype, bias != NULL, &host_bytes, &device_bytes)) return 0;
    CudaMoeCache* existing = cuda_moe_cache_find(first);
    if (existing)
        return existing->source[1] == second && existing->source[2] == third &&
            existing->bias == bias && existing->experts == experts &&
            existing->d_in == d_in && existing->hidden == hidden &&
            existing->d_out == d_out && existing->maximum_top_k == maximum_top_k &&
            existing->slots == slots && !memcmp(existing->dtype,dtype,sizeof(existing->dtype));
    cuda_graph_moe_cache_bytes(NULL, &registered_device);
    if (registered_device > cuda_total_memory ||
        device_bytes > cuda_total_memory - registered_device) {
        cuda_graph_allocation_failed = 1;
        return 0;
    }
    CudaMoeCache* cache = calloc(1, (size_t)host_bytes);
    if (!cache) { cuda_graph_allocation_failed = 1; return 0; }
    cache->host_bytes = (size_t)host_bytes;
    cache->device_bytes = (size_t)device_bytes;
    cache->source[0] = first; cache->source[1] = second; cache->source[2] = third;
    memcpy(cache->dtype,dtype,sizeof(cache->dtype));
    cache->bias = bias;
    cache->experts = experts; cache->d_in = d_in; cache->hidden = hidden;
    cache->d_out = d_out; cache->maximum_top_k = maximum_top_k; cache->slots = slots;
    cache->expert_bytes[0] = (size_t)d_in * (hidden ? hidden : d_out) * cuda_weight_storage_bytes(dtype[0]);
    cache->expert_bytes[1] = hidden ? (size_t)d_in * hidden * cuda_weight_storage_bytes(dtype[1]) : 0;
    cache->expert_bytes[2] = hidden ? (size_t)hidden * d_out * cuda_weight_storage_bytes(dtype[2]) : 0;
    cache->bias_bytes = bias ? (size_t)d_out * sizeof(float) : 0;
    cache->ages = (uint64_t*)(cache + 1);
    cache->expert_ids = (uint32_t*)(cache->ages + slots);
    cache->routes_host = (float*)(cache->expert_ids + slots);
    cache->pinned = (unsigned char*)(cache->routes_host + maximum_top_k);
    for (uint32_t i = 0; i < slots; ++i) cache->expert_ids[i] = UINT32_MAX;
    if (!cuda_context_enter(&guard)) { free(cache); return 0; }
    if (!cuda_ok(cuda_memory_allocate(&cache->allocation, cache->device_bytes),
                 "cuMemAlloc(expert cache)")) {
        cuda_graph_allocation_failed = 1;
        (void)cuda_context_leave(&guard);
        free(cache);
        return 0;
    }
    size_t offset = 0;
    for (unsigned i = 0; i < 3; ++i) if (cache->expert_bytes[i]) {
        cache->weight[i] = cache->allocation + offset;
        offset += cache->expert_bytes[i] * slots;
        offset = (offset + 3u) & ~(size_t)3u;
    }
    if (bias) { cache->bias_device = cache->allocation + offset; offset += cache->bias_bytes * slots; }
    cache->routes_device = cache->allocation + offset;
    cache->next = state->moe_caches;
    state->moe_caches = cache;
    vx_memory_record(&vx_engine_state_current()->memory_observer,
        VX_MEMORY_HOST_METADATA, (uint64_t)(uintptr_t)cache, cache->host_bytes,
        VX_TRACE_MEMORY_ACTION_ALLOCATE);
    return cuda_context_leave(&guard);
}

int cuda_graph_moe_cache_register_linear_storage(const void* weight, uint32_t weight_dtype,
        const float* bias,
        uint32_t experts, uint32_t d_in, uint32_t d_out, uint32_t maximum_top_k,
        uint32_t resident_slots) {
    const uint32_t dtype[3] = {weight_dtype,VX_DTYPE_F32,VX_DTYPE_F32};
    return cuda_moe_cache_register(weight, NULL, NULL, bias, dtype, experts, d_in, 0,
        d_out, maximum_top_k, resident_slots);
}

int cuda_graph_moe_cache_register_gated_storage(const void* gate, uint32_t gate_dtype,
        const void* up, uint32_t up_dtype, const void* down, uint32_t down_dtype,
        uint32_t experts, uint32_t d_in, uint32_t hidden,
        uint32_t d_out, uint32_t maximum_top_k, uint32_t resident_slots) {
    const uint32_t dtype[3] = {gate_dtype,up_dtype,down_dtype};
    return hidden && cuda_moe_cache_register(gate, up, down, NULL, dtype, experts,
        d_in, hidden, d_out, maximum_top_k, resident_slots);
}

int cuda_graph_moe_cache_register_linear(const float* weight, const float* bias,
        uint32_t experts, uint32_t d_in, uint32_t d_out, uint32_t maximum_top_k,
        uint32_t resident_slots) {
    return cuda_graph_moe_cache_register_linear_storage(weight,VX_DTYPE_F32,bias,
        experts,d_in,d_out,maximum_top_k,resident_slots);
}

int cuda_graph_moe_cache_register_gated(const float* gate, const float* up,
        const float* down, uint32_t experts, uint32_t d_in, uint32_t hidden,
        uint32_t d_out, uint32_t maximum_top_k, uint32_t resident_slots) {
    return cuda_graph_moe_cache_register_gated_storage(gate,VX_DTYPE_F32,up,VX_DTYPE_F32,
        down,VX_DTYPE_F32,experts,d_in,hidden,d_out,maximum_top_k,resident_slots);
}

static uint32_t cuda_moe_cache_lookup(const CudaMoeCache* cache, uint32_t expert) {
    for (uint32_t i = 0; i < cache->slots; ++i)
        if (cache->expert_ids[i] == expert) return i;
    return UINT32_MAX;
}

/* The preceding synchronous route readback completes all earlier row kernels.
 * Pin every hit before selecting victims, then keep every requested slot pinned
 * until this row has been submitted. Duplicate routes share one cache slot. */
static int cuda_moe_cache_resolve(CudaMoeCache* cache, uint32_t top_k) {
    memset(cache->pinned, 0, cache->slots);
    for (uint32_t i = 0; i < top_k; ++i) {
        float route = cache->routes_host[i];
        if (!isfinite(route) || route < 0 || route >= (float)cache->experts ||
            route != floorf(route)) return 0;
        uint32_t hit = cuda_moe_cache_lookup(cache, (uint32_t)route);
        if (hit != UINT32_MAX) cache->pinned[hit] = 1;
    }
    for (uint32_t i = 0; i < top_k; ++i) {
        uint32_t expert = (uint32_t)cache->routes_host[i];
        uint32_t slot = cuda_moe_cache_lookup(cache, expert);
        if (slot == UINT32_MAX) {
            uint64_t oldest = UINT64_MAX;
            for (uint32_t candidate = 0; candidate < cache->slots; ++candidate) {
                if (cache->pinned[candidate]) continue;
                if (cache->expert_ids[candidate] == UINT32_MAX) { slot = candidate; break; }
                if (slot == UINT32_MAX || cache->ages[candidate] < oldest) {
                    slot = candidate; oldest = cache->ages[candidate];
                }
            }
            if (slot == UINT32_MAX) return 0;
            cache->expert_ids[slot] = UINT32_MAX;
            for (unsigned plane = 0; plane < 3; ++plane) if (cache->source[plane]) {
                size_t bytes = cache->expert_bytes[plane];
                if (!cuda_ok(cuda_copy_htod_current(cache->weight[plane] + (size_t)slot * bytes,
                        (const unsigned char*)cache->source[plane] + (size_t)expert * bytes, bytes),
                        "cuMemcpyHtoD(expert cache)")) return 0;
            }
            if (cache->bias && !cuda_ok(cuda_copy_htod_current(
                    cache->bias_device + (size_t)slot * cache->bias_bytes,
                    (const unsigned char*)cache->bias + (size_t)expert * cache->bias_bytes,
                    cache->bias_bytes), "cuMemcpyHtoD(expert bias cache)")) return 0;
            cache->expert_ids[slot] = expert;
        }
        cache->pinned[slot] = 1;
        if (++cache->clock == 0) {
            /* Wrap changes replacement priority only, never expert identity. */
            memset(cache->ages, 0, (size_t)cache->slots * sizeof(*cache->ages));
            cache->clock = 1;
        }
        cache->ages[slot] = cache->clock;
        cache->routes_host[i] = (float)slot;
    }
    return cuda_ok(cuda_copy_htod_current(cache->routes_device, cache->routes_host,
        (size_t)top_k * sizeof(float)), "cuMemcpyHtoD(expert routes)");
}

static int cuda_moe_cache_execute(CudaMoeCache* cache, const float* input,
        const float* indices, const float* routes, float* output,
        uint32_t rows, uint32_t top_k, uint32_t input_scale, float limit) {
    size_t input_bytes, output_bytes, route_bytes;
    CudaTensorView input_view, indices_view, routes_view, output_view;
    CudaContextGuard guard;
    if (!input || !indices || !routes || !output || !rows || !top_k ||
        top_k > cache->maximum_top_k || input_scale > 1 || !isfinite(limit) || limit < 0 ||
        !size_product_u32(rows, cache->d_in, sizeof(float), &input_bytes) ||
        !size_product_u32(rows, cache->d_out, sizeof(float), &output_bytes) ||
        !size_product_u32(rows, top_k, sizeof(float), &route_bytes) ||
        ranges_overlap(output, output_bytes, input, input_bytes) ||
        ranges_overlap(output, output_bytes, indices, route_bytes) ||
        ranges_overlap(output, output_bytes, routes, route_bytes) ||
        cuda_replay_memory_locked() || !cuda_context_enter(&guard)) return 0;
    int ok = graph_ensure_input_view(input, input_bytes, 0, &input_view) &&
        graph_ensure_input_view(indices, route_bytes, 0, &indices_view) &&
        graph_ensure_input_view(routes, route_bytes, 0, &routes_view) &&
        graph_output_view(output, output_bytes, &output_view);
    for (uint32_t row = 0; ok && row < rows; ++row) {
        size_t route_offset = (size_t)row * top_k * sizeof(float);
        ok = cuda_ok(cuda_copy_dtoh_current(cache->routes_host,
            indices_view.device + route_offset, (size_t)top_k * sizeof(float)),
            "cuMemcpyDtoH(expert routes)") && cuda_moe_cache_resolve(cache, top_k);
        if (!ok) break;
        CUdeviceptr row_input = input_view.device + (size_t)row * cache->d_in * sizeof(float);
        CUdeviceptr row_output = output_view.device + (size_t)row * cache->d_out * sizeof(float);
        CUdeviceptr row_routes = routes_view.device + route_offset;
        uint32_t one = 1;
        if (cache->hidden) {
            void* args[] = {&row_input,&cache->weight[0],&cache->weight[1],&cache->weight[2],
                &cache->routes_device,&row_routes,&row_output,&one,&cache->d_in,&cache->hidden,
                &cache->d_out,&cache->slots,&top_k,&input_scale,&limit,
                &cache->dtype[0],&cache->dtype[1],&cache->dtype[2]};
            int compact = cache->dtype[0] != VX_DTYPE_F32 || cache->dtype[1] != VX_DTYPE_F32 ||
                cache->dtype[2] != VX_DTYPE_F32;
            ok = cuda_launch(compact ? CUDA_FORWARD_FUNCTION(k_moe_gated_mlp_storage) :
                CUDA_FORWARD_FUNCTION(k_moe_gated_mlp),1,1,1,128,1,1,0,args);
        } else {
            uint32_t has_bias = cache->bias != NULL, no_domain = 0;
            CUdeviceptr no_table = 0;
            void* args[] = {&row_input,&cache->weight[0],&cache->bias_device,
                &cache->routes_device,&row_routes,&row_output,&one,&cache->d_in,&cache->d_out,
                &cache->slots,&top_k,&has_bias,&no_table,&no_domain,&cache->dtype[0]};
            ok = cuda_launch_1d(cache->dtype[0] == VX_DTYPE_F32 ? CUDA_FORWARD_FUNCTION(k_moe_linear) :
                CUDA_FORWARD_FUNCTION(k_moe_linear_storage),cache->d_out,args);
        }
    }
    if (ok) graph_mark_device(output_view.slot);
    if (!cuda_context_leave(&guard)) ok = 0;
    return ok;
}
