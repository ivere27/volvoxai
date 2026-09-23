typedef struct {
    id<MTLBuffer> buffer;
    id<MTLCommandQueue> queue;
} MetalNativeAllocation;

static int metal_tensor_device(VxNativeBuffer* view) {
    *view = (VxNativeBuffer){.kind = VX_NATIVE_BUFFER_METAL,
        .device_context = (uint64_t)(uintptr_t)(__bridge void*)device};
    return device != nil && view->device_context;
}
static int metal_tensor_validate(const VxNativeBuffer* source) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(void*)(uintptr_t)source->handle;
    return !(source->length & 3u) && buffer && [buffer device] == device && source->offset <= [buffer length] &&
        source->length <= [buffer length] - source->offset;
}
static int metal_tensor_blit(id<MTLCommandQueue> queue, id<MTLBuffer> source,
    size_t offset, id<MTLBuffer> destination, size_t bytes) {
    @autoreleasepool {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        if (!command || !blit) return 0;
        [blit copyFromBuffer:source sourceOffset:offset toBuffer:destination destinationOffset:0 size:bytes];
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        return [command status] == MTLCommandBufferStatusCompleted;
    }
}
static int metal_tensor_copy(const void* host, const VxNativeBuffer* source) {
    if (graph_flush_commands() != 0) return 0;
    MetalTensorSlot* slot = graph_get_slot(host, (size_t)source->length, 0);
    if (!slot) return 0;
    if ((!slot->buffer || slot->is_alias || slot->cap < source->length) &&
        !graph_slot_ensure_capacity(slot, (size_t)source->length, NULL, 0, 0)) return 0;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(void*)(uintptr_t)source->handle;
    if (!metal_tensor_blit(commandQueue, buffer, (size_t)source->offset, slot->buffer, (size_t)source->length)) return 0;
    slot->host_dirty = 0;
    slot->device_dirty = 1;
    return 1;
}
static int metal_tensor_snapshot(const void* host, size_t bytes, VxNativeStorage* storage) {
    /* macOS blits require four-byte spans, including INT8 tensor payloads. */
    if (bytes & 3u) return 0;
    MetalTensorSlot* slot = graph_ensure_device(host, bytes, 0);
    if (!slot || graph_flush_commands() != 0) return 0;
    MetalNativeAllocation* allocation = storage->allocation;
    if (!allocation) {
        allocation = calloc(1, sizeof(*allocation));
        if (!allocation) return 0;
        storage->allocation = allocation;
        allocation->buffer = create_buffer(storage->capacity, NULL);
        allocation->queue = [device newCommandQueue];
        if (!allocation->buffer || !allocation->queue) return 0;
        storage->buffer.handle = (uint64_t)(uintptr_t)(__bridge void*)allocation->buffer;
        metal_device_lock();
        g_metal_device_state.reference_count++;
        metal_device_unlock();
    }
    if (!metal_tensor_blit(allocation->queue, slot->buffer, 0, allocation->buffer, bytes)) return 0;
    return 1;
}
static int metal_tensor_read(const VxNativeStorage* storage, void* host, size_t bytes) {
    MetalNativeAllocation* allocation = storage->allocation;
    if (!allocation || ![allocation->buffer contents]) return 0;
    /* ExecuteTensors and callers complete producer work before host reads. */
    memcpy(host, [allocation->buffer contents], bytes);
    return 1;
}
static void metal_tensor_destroy(VxNativeStorage* storage) {
    MetalNativeAllocation* allocation = storage->allocation;
    if (!allocation) return;
    VX_METAL_RELEASE(allocation->buffer);
    VX_METAL_RELEASE(allocation->queue);
    free(allocation);
    if (storage->buffer.handle) {
        metal_device_lock();
        if (--g_metal_device_state.reference_count == 0) metal_device_shutdown_locked();
        metal_device_unlock();
    }
    storage->buffer.handle = 0;
}
static int metal_tensor_host_view(const VxNativeStorage* storage, VxNativeBuffer* view) {
    MetalNativeAllocation* allocation = storage->allocation;
    if (!allocation || [allocation->buffer storageMode] != MTLStorageModeShared ||
        ![allocation->buffer contents]) return 0;
    *view = (VxNativeBuffer){.kind = VX_NATIVE_BUFFER_HOST,
        .handle = (uint64_t)(uintptr_t)[allocation->buffer contents],
        .length = storage->buffer.length};
    return 1;
}
const VxNativeTensorOps vx_metal_tensor_ops = {
    VX_NATIVE_BUFFER_METAL, metal_tensor_device, metal_tensor_validate,
    metal_tensor_copy, metal_tensor_snapshot, metal_tensor_read, metal_tensor_destroy,
    NULL, NULL, NULL, metal_tensor_host_view
};
