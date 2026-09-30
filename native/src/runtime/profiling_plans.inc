/* Immutable plan metadata lives inside one preallocated collector region. */
int vx_trace_plans_enabled(const VxTraceScope* scope) {
    return scope && scope->trace && scope->trace->execution_plans;
}
const VxExecutionPlan* vx_trace_plan_at(const VxTraceView* view, size_t offset) {
    if (!view || !view->plan_data || offset >= view->plan_count) return NULL;
    const unsigned char* cursor = view->plan_data;
    while (offset--) cursor += ((const VxExecutionPlan*)cursor)->bytes;
    return (const VxExecutionPlan*)cursor;
}
void vx_trace_capture_plan(VxTraceScope* scope, const char* fingerprint,
    const char* signature, uint64_t storage_generation, uint32_t steps, uint32_t tensors, uint32_t references,
    uint32_t allocations, void (*fill)(VxExecutionPlan*, void*), void* context) {
    if (!vx_trace_plans_enabled(scope)) return;
    VxTrace* trace = scope->trace;
    if (!signature) signature = "";
    if (!fingerprint) fingerprint = "";
    pthread_mutex_lock(&trace->owner->mutex);
    if (trace->abandoned) goto done;
    unsigned char* cursor = trace->plan_data;
    for (size_t i = 0; i < trace->plan_count; i++) {
        VxExecutionPlan* plan = (VxExecutionPlan*)cursor;
        if (plan->identity.context_id == scope->identity.context_id &&
            plan->identity.compiled_model_id == scope->identity.compiled_model_id &&
            plan->identity.graph_id == scope->identity.graph_id &&
            plan->identity.graph_revision == scope->identity.graph_revision &&
            plan->storage_generation == storage_generation &&
            !strcmp(plan->signature, signature)) { scope->plan_id = plan->id; goto done; }
        cursor += plan->bytes;
    }
    size_t bytes = vx_execution_plan_size(steps, tensors, references, allocations, signature);
    if (!bytes || bytes > trace->plan_capacity - trace->plan_bytes) { trace->dropped_plans++; goto done; }
    VxExecutionPlan* plan = (VxExecutionPlan*)(trace->plan_data + trace->plan_bytes);
    vx_execution_plan_init(plan, bytes, steps, tensors, references, allocations, signature);
    plan->id = ++trace->plan_count;
    plan->storage_generation = storage_generation;
    plan->identity = scope->identity; plan->identity.execution_id = 0;
    snprintf(plan->fingerprint, sizeof(plan->fingerprint), "%s", fingerprint);
    snprintf(plan->backend, sizeof(plan->backend), "%s", scope->backend ? scope->backend : "");
    trace->plan_bytes += (size_t)bytes;
    scope->plan_id = plan->id;
    /* The active scope pins the collector and prevents READY. Fill outside
     * this mutex to preserve engine/storage -> collector lock ordering. */
    pthread_mutex_unlock(&trace->owner->mutex);
    fill(plan, context);
    return;
done:
    pthread_mutex_unlock(&trace->owner->mutex);
}

void vx_memory_inventory_add(VxMemoryInventory* inventory, VxMemoryAllocator allocator,
    uint64_t key, uint64_t bytes, VxMemoryResourceRole role, const char* name) {
    if (!inventory || !key || !bytes) return;
    VxMemorySpace space = vx_memory_allocator_space(allocator);
    for (size_t i = 0; i < inventory->count; i++) {
        VxMemoryAllocation* item = &inventory->items[i];
        if (item->key == key && (item->space == space || (item->space != VX_MEMORY_SPACE_DEVICE && space != VX_MEMORY_SPACE_DEVICE)) &&
            (space != VX_MEMORY_SPACE_DEVICE || (item->allocator == allocator &&
                item->owner_id == inventory->owner_id))) return;
    }
    if (inventory->count == inventory->capacity) { inventory->truncated = 1; return; }
    VxMemoryAllocation* item = &inventory->items[inventory->count++];
    *item = (VxMemoryAllocation){.key = key, .owner_id = inventory->owner_id,
        .owner_kind = inventory->owner_kind, .capacity = bytes,
        .role = role, .space = space, .allocator = allocator};
    snprintf(item->name, sizeof(item->name), "%s", name ? name : "");
}
void vx_memory_inventory_inspect(void* opaque, VxMemoryAllocator allocator,
    uint64_t key, uint64_t bytes) {
    vx_memory_inventory_add(opaque, allocator, key, bytes,
        vx_memory_allocator_role(allocator), vx_memory_allocator_name(allocator));
}
