int vx_trace_memory_enabled(const VxTrace* trace) {
    return trace && trace->memory && atomic_load_explicit(&trace->owner->enabled, memory_order_acquire);
}
/* Included by profiling.c. No process-wide allocator interception or pointers
 * in public records. Existing allocator owners supply their physical keys. */
static const char* const vx_memory_names[VX_MEMORY_ALLOCATOR_COUNT] = {
    "host.arena", "host.tensor", "cuda.graph", "opengl.graph", "vulkan.graph",
    "webgpu.buffer", "host.result", "cuda.result", "opengl.result",
    "vulkan.result", "metal.result"
};
static const char* const vx_memory_backends[VX_MEMORY_ALLOCATOR_COUNT] = {
    "cpu", "cpu", "cuda", "opengl", "vulkan", "webgpu", "cpu", "cuda",
    "opengl", "vulkan", "metal"
};
const char* vx_memory_allocator_name(VxMemoryAllocator allocator) {
    return vx_memory_names[allocator];
}
const char* vx_memory_allocator_backend(VxMemoryAllocator allocator) {
    return vx_memory_backends[allocator];
}
VxMemorySpace vx_memory_allocator_space(VxMemoryAllocator allocator) {
    if (allocator != VX_MEMORY_HOST_ARENA && allocator != VX_MEMORY_HOST_TENSOR &&
        allocator != VX_MEMORY_RESULT_HOST) return VX_MEMORY_SPACE_DEVICE;
#if defined(__wasm__)
    return VX_MEMORY_SPACE_WASM_LINEAR;
#else
    return VX_MEMORY_SPACE_NATIVE_HEAP;
#endif
}
void vx_memory_observer_clear(VxMemoryObserver* observer) {
    if (!observer) return;
    vx_memory_owner_release(observer->owner);
    memset(observer, 0, sizeof(*observer));
}
void vx_memory_observer_copy(VxMemoryObserver* to, const VxMemoryObserver* from) {
    *to = *from;
    if (to->owner) atomic_fetch_add_explicit(&to->owner->references, 1, memory_order_relaxed);
}
int vx_memory_observer_attach(VxMemoryObserver* observer, const VxTraceScope* scope) {
    if (!scope || !scope->trace || !scope->trace->memory) return 0;
    VxTraceMemoryOwner* owner = scope->trace->owner;
    if (observer->owner == owner) return 0;
    vx_memory_observer_clear(observer);
    atomic_fetch_add_explicit(&owner->references, 1, memory_order_relaxed);
    observer->owner = owner;
    pthread_mutex_lock(&owner->mutex);
    observer->source_id = ++owner->next_source;
    pthread_mutex_unlock(&owner->mutex);
    observer->identity = scope->identity;
    observer->identity.execution_id = 0; /* Allocations can outlive one call. */
    return 1;
}
static uint64_t vx_memory_sum(uint64_t left, uint64_t right, int* complete) {
    if (right > UINT64_MAX - left) { *complete = 0; return UINT64_MAX; }
    return left + right;
}
static size_t vx_memory_slot(const VxTrace* trace, uint64_t key,
    uint64_t source, VxMemoryAllocator allocator) {
    uint64_t hash = (key ^ (key >> 33) ^ source ^ ((uint64_t)allocator << 21)) * UINT64_C(0xff51afd7ed558ccd);
    return (size_t)(hash ^ (hash >> 33)) & (trace->allocation_capacity - 1);
}
/* Backshift deletion keeps lookup cost dependent on live allocations, not the
 * number of allocations ever observed. It also works when the table was full. */
static void vx_memory_remove(VxTrace* trace, size_t hole) {
    size_t mask = trace->allocation_capacity - 1;
    for (size_t step = 1, slot = (hole + 1) & mask;
         step < trace->allocation_capacity; step++, slot = (slot + 1) & mask) {
        VxTraceAllocation* next = &trace->allocations[slot];
        if (!next->state) break;
        size_t home = vx_memory_slot(trace, next->key, next->owner, next->allocator);
        if (((hole - home) & mask) < ((slot - home) & mask)) {
            trace->allocations[hole] = *next;
            hole = slot;
        }
    }
    trace->allocations[hole].state = 0;
}
void vx_memory_record(VxMemoryObserver* observer, VxMemoryAllocator allocator,
    uint64_t key, uint64_t bytes, VxTraceMemoryAction action) {
    if (!key || !observer || !observer->owner || allocator >= VX_MEMORY_ALLOCATOR_COUNT) return;
    VxTraceMemoryOwner* owner = observer->owner;
    if (!atomic_load_explicit(&owner->enabled, memory_order_acquire)) return;
    pthread_mutex_lock(&owner->mutex);
    VxTrace* trace = owner->trace;
    if (!trace || trace->stopped) { pthread_mutex_unlock(&owner->mutex); return; }
    VxAllocatorMemory* counter = &trace->allocators[allocator];
    uint64_t source = observer->source_id;
    size_t slot = vx_memory_slot(trace, key, source, allocator);
    VxTraceAllocation* allocation = NULL;
    VxTraceAllocation* vacant = NULL;
    for (size_t i = 0; i < trace->allocation_capacity; i++) {
        VxTraceAllocation* candidate = &trace->allocations[slot];
        if (candidate->state == 1 && candidate->owner == source &&
            candidate->key == key && candidate->allocator == allocator) { allocation = candidate; break; }
        if (candidate->state != 1 && !vacant) vacant = candidate;
        if (!candidate->state) break;
        slot = (slot + 1) & (trace->allocation_capacity - 1);
    }
    if (action == VX_TRACE_MEMORY_ACTION_FREE && !allocation) goto done;
    if (action != VX_TRACE_MEMORY_ACTION_FREE && allocation) goto done;
    uint64_t now = vx_trace_now_ns();
    if (!counter->observed) { counter->observed = 1; counter->start_ns = now - trace->origin_ns; }
    VxTraceAllocation freed;
    if (action != VX_TRACE_MEMORY_ACTION_FREE) {
        if (!vacant) {
            counter->complete = 0; counter->dropped++; trace->dropped++; goto done;
        }
        allocation = vacant;
        *allocation = (VxTraceAllocation){source, key, ++trace->next_allocation_id, bytes, allocator, 1};
        counter->live = vx_memory_sum(counter->live, bytes, &counter->complete);
        uint64_t* total = action == VX_TRACE_MEMORY_ACTION_EXISTING ? &counter->existing : &counter->allocated;
        *total = vx_memory_sum(*total, bytes, &counter->complete);
        if (counter->live > counter->peak) counter->peak = counter->live;
    } else {
        bytes = allocation->bytes;
        counter->live -= bytes;
        counter->freed = vx_memory_sum(counter->freed, bytes, &counter->complete);
        freed = *allocation;
        vx_memory_remove(trace, (size_t)(allocation - trace->allocations));
        allocation = &freed;
    }
    VxTraceScope scope = {.trace = trace, .identity = observer->identity,
        .start_ns = now, .name = vx_memory_allocator_name(allocator),
        .backend = vx_memory_allocator_backend(allocator), .work = {.index = -1}};
    VxTraceRecord* record = vx_trace_record_locked(&scope, now, now, -1,
        VX_TRACE_EVENT_MEMORY, scope.name, "", 0);
    if (record) {
        record->allocation_id = allocation->id; record->allocation_bytes = bytes;
        record->live_bytes = counter->live; record->allocator = allocator;
        record->memory_action = action;
    } else counter->dropped++;
done:
    pthread_mutex_unlock(&owner->mutex);
}

void vx_memory_lost(VxMemoryObserver* observer, VxMemoryAllocator allocator) {
    if (!observer || !observer->owner) return;
    VxTraceMemoryOwner* owner = observer->owner;
    if (!atomic_load_explicit(&owner->enabled, memory_order_acquire)) return;
    pthread_mutex_lock(&owner->mutex);
    VxTrace* trace = owner->trace;
    if (trace && !trace->stopped) {
        VxAllocatorMemory* counter = &trace->allocators[allocator];
        if (!counter->observed) counter->start_ns = vx_trace_now_ns() - trace->origin_ns;
        counter->observed = 1; counter->complete = 0;
        counter->dropped++; trace->dropped++;
    }
    pthread_mutex_unlock(&owner->mutex);
}
void vx_trace_memory_bridge(VxTraceScope* scope,
    uint32_t (*start)(VxTraceScope*, uint32_t), void (*stop)(uint32_t)) {
    if (!scope || !scope->trace || !scope->trace->memory) return;
    VxTrace* trace = scope->trace;
    pthread_mutex_lock(&trace->owner->mutex);
    int begin = !trace->stopped && !trace->memory_bridge_started;
    if (begin) trace->memory_bridge_started = 1;
    pthread_mutex_unlock(&trace->owner->mutex);
    if (!begin) return;
    /* JS can synchronously report its initial inventory back into C. */
    uint32_t ticket = start(scope, (uint32_t)trace->allocation_capacity);
    pthread_mutex_lock(&trace->owner->mutex);
    int closed = trace->stopped;
    if (!closed) { trace->memory_bridge_ticket = ticket; trace->memory_bridge_stop = stop; }
    pthread_mutex_unlock(&trace->owner->mutex);
    if (closed && ticket) stop(ticket);
}
