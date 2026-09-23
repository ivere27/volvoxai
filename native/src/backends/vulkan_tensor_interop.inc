static _Thread_local int vk_tensor_batch_active;
static int vk_tensor_batch_begin(int outputs) {
    (void)outputs;
    if (vk_tensor_batch_active || !vk_is_ready()) return 0;
    vk_tensor_batch_active = 1;
    return 1;
}
static int vk_tensor_batch_end(void) {
    if (!vk_tensor_batch_active) return 1;
    vk_tensor_batch_active = 0;
    return vk_graph_flush_wait();
}

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize capacity;
    VkDeviceSize allocation_size;
    VkMemoryPropertyFlags flags;
} VkTensorAllocation;

static int vk_tensor_device(VxNativeBuffer* view) {
    *view = (VxNativeBuffer){.kind = VX_NATIVE_BUFFER_VULKAN,
        .device_context = (uint64_t)(uintptr_t)g_vulkan_device.device_handle};
    return g_vulkan_device.initialized && view->device_context;
}
static void vk_tensor_free(VkTensorAllocation* allocation) {
    if (!allocation) return;
    if (allocation->buffer) vkDestroyBuffer(device, allocation->buffer, NULL);
    if (allocation->memory) vkFreeMemory(device, allocation->memory, NULL);
    free(allocation);
}
static VkTensorAllocation* vk_tensor_allocate(size_t bytes, int host_visible) {
    if (!bytes || bytes > SIZE_MAX - 3u) return NULL;
    VkTensorAllocation* allocation = calloc(1, sizeof(*allocation));
    if (!allocation) return NULL;
    allocation->capacity = (bytes + 3u) & ~(size_t)3u;
    VkBufferCreateInfo buffer_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = allocation->capacity,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(device, &buffer_info, NULL, &allocation->buffer) != VK_SUCCESS) goto fail;
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, allocation->buffer, &requirements);
    uint32_t index = find_preferred_memory_type(requirements.memoryTypeBits,
        host_visible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        host_visible ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : 0);
    if (index == UINT32_MAX) goto fail;
    allocation->flags = mem_props.memoryTypes[index].propertyFlags;
    allocation->allocation_size = requirements.size;
    VkMemoryAllocateInfo memory_info = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size, .memoryTypeIndex = index};
    if (vkAllocateMemory(device, &memory_info, NULL, &allocation->memory) != VK_SUCCESS ||
        vkBindBufferMemory(device, allocation->buffer, allocation->memory, 0) != VK_SUCCESS) goto fail;
    return allocation;
fail:
    vk_tensor_free(allocation);
    return NULL;
}
static void vk_tensor_barrier(VkCommandBuffer command, int before) {
    VkMemoryBarrier barrier = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = before ? VK_ACCESS_MEMORY_WRITE_BIT : VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = before ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_MEMORY_READ_BIT};
    vkCmdPipelineBarrier(command, before ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
        before ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);
}
static int vk_tensor_copy(const void* host, const VxNativeBuffer* source) {
    size_t bytes = (size_t)source->length;
    if ((source->offset & 3u) || bytes > SIZE_MAX - 3u) return 0;
    VkTensorSlot* slot = graph_output_slot(host, bytes);
    size_t span;
    if (!slot || !vk_staging_aligned_copy_span(slot->offset, bytes, &span) ||
        !vk_graph_begin_recording()) return 0;
    if ((VkBuffer)source->handle != io_buffer || source->offset != slot->offset) {
        vk_tensor_barrier(cmd_buf, 1);
        VkBufferCopy copy = {.srcOffset = source->offset, .dstOffset = slot->offset, .size = span};
        vkCmdCopyBuffer(cmd_buf, (VkBuffer)source->handle, io_buffer, 1, &copy);
        vk_tensor_barrier(cmd_buf, 0);
    }
    if (!vk_tensor_batch_active && !vk_graph_flush_wait()) return 0;
    slot->host_dirty = 0;
    slot->device_dirty = 1;
    return 1;
}
static int vk_tensor_snapshot(const void* host, size_t bytes, VxNativeStorage* storage) {
    VkTensorSlot* slot = graph_ensure_device(host, bytes, 0);
    size_t span;
    if (!slot || !vk_staging_aligned_copy_span(slot->offset, bytes, &span)) return 0;
    VkTensorAllocation* allocation = storage->allocation;
    if (!allocation) {
        allocation = vk_tensor_allocate(storage->capacity, 0);
        if (!allocation) return 0;
        storage->allocation = allocation;
        storage->buffer.handle = (uint64_t)allocation->buffer;
        pthread_mutex_lock(&g_vulkan_device.mutex);
        g_vulkan_device.context_count++;
        pthread_mutex_unlock(&g_vulkan_device.mutex);
    }
    int ok = vk_graph_begin_recording();
    if (ok) {
        vk_tensor_barrier(cmd_buf, 1);
        VkBufferCopy copy = {.srcOffset = slot->offset, .size = span};
        vkCmdCopyBuffer(cmd_buf, io_buffer, allocation->buffer, 1, &copy);
        vk_tensor_barrier(cmd_buf, 0);
        if (!vk_tensor_batch_active) ok = vk_graph_flush_wait();
    }
    return ok;
}
/* Independent readback: no execution context, arena, or staging ring survives
 * in a published tensor. The device itself is module-owned. */
static int vk_tensor_read(const VxNativeStorage* storage, void* host, size_t bytes) {
    VkTensorAllocation* allocation = storage->allocation;
    VkTensorAllocation* stage = vk_tensor_allocate(bytes, 1);
    VkCommandPool read_pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    void* mapped = NULL;
    int ok = 0;
    if (!stage) return 0;
    VkCommandPoolCreateInfo pool_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = queue_family_index};
    if (vkCreateCommandPool(device, &pool_info, NULL, &read_pool) != VK_SUCCESS) goto done;
    VkCommandBufferAllocateInfo command_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = read_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    if (vkAllocateCommandBuffers(device, &command_info, &command) != VK_SUCCESS) goto done;
    VkFenceCreateInfo fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(device, &fence_info, NULL, &fence) != VK_SUCCESS) goto done;
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) goto done;
    vk_tensor_barrier(command, 1);
    VkBufferCopy copy = {.size = stage->capacity};
    vkCmdCopyBuffer(command, allocation->buffer, stage->buffer, 1, &copy);
    VkMemoryBarrier barrier = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);
    if (vkEndCommandBuffer(command) != VK_SUCCESS) goto done;
    VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &command};
    pthread_mutex_lock(&g_vulkan_device.mutex);
    VkResult submitted = vkQueueSubmit(compute_queue, 1, &submit, fence);
    pthread_mutex_unlock(&g_vulkan_device.mutex);
    if (submitted != VK_SUCCESS || vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) goto done;
    if (vkMapMemory(device, stage->memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) goto done;
    if (!(stage->flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        VkMappedMemoryRange range = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = stage->memory, .size = VK_WHOLE_SIZE};
        if (vkInvalidateMappedMemoryRanges(device, 1, &range) != VK_SUCCESS) goto done;
    }
    memcpy(host, mapped, bytes);
    ok = 1;
done:
    if (mapped) vkUnmapMemory(device, stage->memory);
    if (fence) vkDestroyFence(device, fence, NULL);
    if (read_pool) vkDestroyCommandPool(device, read_pool, NULL);
    vk_tensor_free(stage);
    return ok;
}
static void vk_tensor_destroy(VxNativeStorage* storage) {
    if (!storage->allocation) return;
    vk_tensor_free(storage->allocation);
    storage->buffer.handle = 0;
    pthread_mutex_lock(&g_vulkan_device.mutex);
    g_vulkan_device.context_count--;
    pthread_mutex_unlock(&g_vulkan_device.mutex);
}
static int vk_tensor_view(const void* host, size_t bytes, VxNativeBuffer* view) {
    VkTensorSlot* slot = graph_ensure_device(host, bytes, 0);
    if (!slot) return 0;
    view->handle = (uint64_t)io_buffer;
    view->offset = slot->offset;
    view->length = bytes;
    return 1;
}
const VxNativeTensorOps vx_vulkan_tensor_ops = {
    VX_NATIVE_BUFFER_VULKAN, vk_tensor_device, NULL, vk_tensor_copy,
    vk_tensor_snapshot, vk_tensor_read, vk_tensor_destroy, vk_tensor_batch_begin, vk_tensor_batch_end,
    vk_tensor_view
};
