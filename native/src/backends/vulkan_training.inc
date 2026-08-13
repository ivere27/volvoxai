/* Vulkan training: touched-tensor tracking, gradient pool and the backward
 * plan admission check.  The whole fragment sits inside
 * #if VOLVOXAI_ENABLE_TRAINING, so it compiles away in inference profiles. */

#if VOLVOXAI_ENABLE_TRAINING
static void vk_training_track_tensor(VkTensorSlot* tensor) {
    if (!tensor) return;
    for (int i = 0; i < training_touched_count; i++) {
        if (training_touched_slots[i] == tensor) return;
    }
    if (training_touched_count < VK_GRAPH_MAX_TENSORS) {
        training_touched_slots[training_touched_count++] = tensor;
    }
}

static const VkTrainingSpec* vk_training_find_spec(const char* shader, const char* entry) {
    if (!shader || !entry) return NULL;
    size_t count = sizeof(training_specs) / sizeof(training_specs[0]);
    for (size_t i = 0; i < count; i++) {
        if (!strcmp(shader, training_specs[i].shader) &&
            !strcmp(entry, training_specs[i].entry)) {
            return &training_specs[i];
        }
    }
    return NULL;
}

static int vk_training_prepare_pool(void) {
    if (training_desc_pool != VK_NULL_HANDLE) return 1;
    VkTrainingKernelSlot* kernels = (VkTrainingKernelSlot*)calloc(
        VK_TRAINING_MAX_KERNELS, sizeof(*kernels));
    VkGraphDispatchSet* sets = (VkGraphDispatchSet*)calloc(
        VK_TRAINING_MAX_DISPATCH_SETS, sizeof(*sets));
    VkTensorSlot** touched = (VkTensorSlot**)calloc(
        VK_GRAPH_MAX_TENSORS, sizeof(*touched));
    if (!kernels || !sets || !touched) {
        free(kernels);
        free(sets);
        free(touched);
        return 0;
    }
    VkDescriptorPoolSize sizes[2];
    memset(sizes, 0, sizeof(sizes));
    sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[0].descriptorCount = VK_TRAINING_MAX_DISPATCH_SETS * 12u +
                               VK_TRAINING_MAX_KERNELS * 12u;
    sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = VK_TRAINING_MAX_DISPATCH_SETS +
                               VK_TRAINING_MAX_KERNELS;
    VkDescriptorPoolCreateInfo info = {0};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.maxSets = VK_TRAINING_MAX_DISPATCH_SETS + VK_TRAINING_MAX_KERNELS;
    info.poolSizeCount = 2;
    info.pPoolSizes = sizes;
    pthread_mutex_lock(&g_vulkan_device.mutex);
    VkResult pool_result = vkCreateDescriptorPool(
        device, &info, NULL, &training_desc_pool);
    pthread_mutex_unlock(&g_vulkan_device.mutex);
    if (pool_result != VK_SUCCESS) {
        free(kernels);
        free(sets);
        free(touched);
        return 0;
    }
    training_kernels = kernels;
    training_dispatch_sets = sets;
    training_touched_slots = touched;
    return 1;
}

static VkKernel* vk_training_kernel(const VkTrainingSpec* spec) {
    if (!spec || training_desc_pool == VK_NULL_HANDLE) return NULL;
    for (int i = 0; i < training_kernel_count; i++) {
        if (training_kernels[i].spec == spec) return &training_kernels[i].kernel;
    }
    if (training_kernel_count >= VK_TRAINING_MAX_KERNELS) return NULL;
    VkTrainingKernelSlot* slot = &training_kernels[training_kernel_count++];
    memset(slot, 0, sizeof(*slot));
    slot->spec = spec;
    int n = snprintf(slot->path, sizeof(slot->path),
                     "spv/%s.spv", spec->shader);
    if (n <= 0 || (size_t)n >= sizeof(slot->path)) {
        training_kernel_count--;
        memset(slot, 0, sizeof(*slot));
        return NULL;
    }
    slot->kernel.name = spec->shader;
    slot->kernel.path = slot->path;
    slot->kernel.binding_count = spec->binding_count;
    slot->kernel.uniform_binding = spec->uniform_binding;
    return &slot->kernel;
}

static VkDescriptorSet vk_training_dispatch_set(
        VkKernel* kernel, const VkPreparedKernel* prepared) {
    if (!kernel || !prepared || prepared->prepared_desc_layout == VK_NULL_HANDLE ||
        training_dispatch_set_cursor >= VK_TRAINING_MAX_DISPATCH_SETS) {
        return VK_NULL_HANDLE;
    }
    int index = training_dispatch_set_cursor++;
    VkGraphDispatchSet* slot = &training_dispatch_sets[index];
    if (slot->descriptor_set != VK_NULL_HANDLE && slot->kernel == kernel) return slot->descriptor_set;

    VkDescriptorSetAllocateInfo info = {0};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.descriptorPool = training_desc_pool;
    info.descriptorSetCount = 1;
    info.pSetLayouts = &prepared->prepared_desc_layout;
    if (vkAllocateDescriptorSets(device, &info, &slot->descriptor_set) != VK_SUCCESS) {
        slot->descriptor_set = VK_NULL_HANDLE;
        slot->kernel = NULL;
        return VK_NULL_HANDLE;
    }
    slot->kernel = kernel;
    if (index >= training_dispatch_set_count) training_dispatch_set_count = index + 1;
    return slot->descriptor_set;
}

static int vk_training_dispatch_kernel(VkKernel* kernel,
                                       const VkGraphBinding* bindings,
                                       uint32_t gx, uint32_t gy, uint32_t gz) {
    VkPreparedKernel* prepared;
    if (!vk_dispatch_dimensions_valid(gx, gy, gz) || !bindings ||
        kernel->binding_count > 16) return 0;
    prepared = vk_prepare_kernel(kernel);
    if (!prepared) return 0;
    VkDescriptorSet set = vk_training_dispatch_set(kernel, prepared);
    if (set == VK_NULL_HANDLE) return 0;

    VkDescriptorBufferInfo infos[16];
    VkWriteDescriptorSet writes[16];
    memset(infos, 0, sizeof(infos));
    memset(writes, 0, sizeof(writes));
    for (int i = 0; i < kernel->binding_count; i++) {
        infos[i].buffer = io_buffer;
        infos[i].offset = bindings[i].offset;
        infos[i].range = bindings[i].bytes;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorType = i == kernel->uniform_binding ?
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device, (uint32_t)kernel->binding_count, writes, 0, NULL);

    if (!vk_graph_begin_recording()) return 0;
    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                      prepared->pipeline);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            prepared->prepared_pipeline_layout, 0, 1, &set, 0, NULL);
    vkCmdDispatch(cmd_buf, gx, gy, gz);
    if (vkCmdPipelineBarrier) {
        VkMemoryBarrier barrier = {0};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             1, &barrier, 0, NULL, 0, NULL);
    }
    return 1;
}

int vk_training_available(void) {
    return vk_is_ready() ? 1 : 0;
}

int vk_training_supports(const char* shader_name, const char* entry_point,
                         const size_t* bytes, int binding_count,
                         uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    const VkTrainingSpec* spec = vk_training_find_spec(shader_name, entry_point);
    if (!vk_training_available() || !spec || !bytes ||
        binding_count != spec->binding_count || binding_count <= 0 ||
        binding_count > 16 || spec->params_binding != binding_count - 1 ||
        !vk_dispatch_dimensions_valid(groups_x, groups_y, groups_z) ||
        training_max_workgroup_size_x < 64 ||
        training_max_workgroup_invocations < 64) return 0;

    uint32_t storage_bindings = spec->uniform_binding >= 0
        ? (uint32_t)(binding_count - 1) : (uint32_t)binding_count;
    uint32_t uniform_bindings = spec->uniform_binding >= 0 ? 1u : 0u;
    if (storage_bindings > training_max_storage_bindings ||
        uniform_bindings > training_max_uniform_bindings) return 0;
    for (int i = 0; i < binding_count; i++) {
        VkDeviceSize limit = i == spec->uniform_binding
            ? training_max_uniform_range : training_max_storage_range;
        if (bytes[i] == 0 || (VkDeviceSize)bytes[i] > limit) return 0;
    }
    return 1;
}

int vk_training_plan_supported(int command_count, const size_t* params_bytes,
                               const VkTrainingTensorRequirement* tensors,
                               int tensor_count) {
    if (!vk_training_available() || command_count <= 0 ||
        command_count > VK_TRAINING_MAX_DISPATCH_SETS || !params_bytes ||
        tensor_count < 0 || (tensor_count > 0 && !tensors) ||
        graph_alignment == 0 || io_size < VK_GRAPH_SCRATCH_BYTES) return 0;

    size_t arena_limit = io_size - VK_GRAPH_SCRATCH_BYTES;
    size_t arena_cursor = graph_bump;
    int additional_slots = 0;
    for (int i = 0; i < tensor_count; i++) {
        if (!tensors[i].host || tensors[i].bytes == 0) return 0;
        int slot = graph_find_slot(tensors[i].host);
        if (slot >= 0 && tensors[i].bytes <= graph_slots[slot].bytes) continue;
        if (slot < 0) additional_slots++;
        if (arena_cursor > SIZE_MAX - (graph_alignment - 1)) return 0;
        arena_cursor = ALIGN_UP(arena_cursor, graph_alignment);
        if (arena_cursor > arena_limit ||
            tensors[i].bytes > arena_limit - arena_cursor) return 0;
        arena_cursor += tensors[i].bytes;
        if (arena_cursor > SIZE_MAX - (graph_alignment - 1)) return 0;
        arena_cursor = ALIGN_UP(arena_cursor, graph_alignment);
    }
    if (additional_slots > VK_GRAPH_MAX_TENSORS - graph_slot_count) return 0;

    size_t cursor = io_size - VK_GRAPH_SCRATCH_BYTES;
    if (cursor > SIZE_MAX - (graph_alignment - 1)) return 0;
    cursor = ALIGN_UP(cursor, graph_alignment);
    for (int i = 0; i < command_count; i++) {
        size_t bytes = params_bytes[i];
        if (bytes == 0 || cursor > io_size || bytes > io_size - cursor) return 0;
        cursor += bytes;
        if (cursor > SIZE_MAX - (graph_alignment - 1)) return 0;
        cursor = ALIGN_UP(cursor, graph_alignment);
    }
    return cursor <= io_size;
}

int vk_training_begin(void) {
    if (!vk_is_ready() || training_active || !vk_graph_flush_wait()) return -1;
    int reuse_pool = training_desc_pool != VK_NULL_HANDLE;
    if (!vk_training_prepare_pool()) return -1;
    if (reuse_pool) {
        if (!vkResetDescriptorPool ||
            vkResetDescriptorPool(device, training_desc_pool, 0) != VK_SUCCESS) return -1;
        memset(training_dispatch_sets, 0,
               VK_TRAINING_MAX_DISPATCH_SETS * sizeof(*training_dispatch_sets));
        training_dispatch_set_count = 0;
    }
    scratch_bump = 0;
    scratch_high_water = 0;
    training_dispatch_set_cursor = 0;
    training_touched_count = 0;
    training_active = 1;
    return 0;
}

int vk_training_dispatch(const char* shader_name, const char* entry_point,
                         void* const* hosts, const size_t* bytes,
                         const unsigned char* access,
                         const unsigned char* is_weight,
                         int binding_count,
                         uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    const VkTrainingSpec* spec = vk_training_find_spec(shader_name, entry_point);
    if (!training_active || !spec || !hosts || !bytes || !access ||
        !vk_training_supports(shader_name, entry_point, bytes, binding_count,
                              groups_x, groups_y, groups_z)) return -1;

    graph_scratch_begin();
    VkGraphBinding bindings[16];
    VkTensorSlot* writable[16];
    memset(bindings, 0, sizeof(bindings));
    memset(writable, 0, sizeof(writable));
    for (int i = 0; i < binding_count; i++) {
        unsigned char mode = access[i];
        if (!hosts[i] || !bytes[i] || (is_weight && is_weight[i] > 1) ||
            mode < VK_TRAINING_READ ||
            mode > (VK_TRAINING_READ | VK_TRAINING_WRITE)) return -1;
        unsigned char expected = spec->read_write_mask & (1u << i) ?
            (VK_TRAINING_READ | VK_TRAINING_WRITE) : VK_TRAINING_READ;
        if (mode != expected) return -1;
        if (i == spec->params_binding) {
            if (mode != VK_TRAINING_READ) return -1;
            size_t offset = graph_scratch_upload(hosts[i], bytes[i]);
            if (offset == SIZE_MAX) return -1;
            bindings[i].offset = offset;
            bindings[i].bytes = bytes[i];
            continue;
        }

        VkTensorSlot* tensor = NULL;
        if (mode & VK_TRAINING_READ) {
            tensor = graph_ensure_device(hosts[i], bytes[i], is_weight ? is_weight[i] != 0 : 0);
        } else {
            tensor = graph_get_slot(hosts[i], bytes[i], is_weight ? is_weight[i] != 0 : 0);
            if (tensor && tensor->host_dirty) {
                if (!vk_staging_zero(tensor->offset, bytes[i])) return -1;
                tensor->host_dirty = 0;
                tensor->device_dirty = 0;
            }
        }
        if (!tensor) return -1;
        vk_training_track_tensor(tensor);
        bindings[i].offset = tensor->offset;
        bindings[i].bytes = bytes[i];
        if (mode & VK_TRAINING_WRITE) writable[i] = tensor;
    }

    VkKernel* kernel = vk_training_kernel(spec);
    if (!kernel || !vk_training_dispatch_kernel(kernel, bindings,
                                                 groups_x, groups_y, groups_z)) return -1;
    for (int i = 0; i < binding_count; i++) {
        if (writable[i]) {
            graph_mark_device(writable[i]);
        }
    }
    return 0;
}

int vk_training_sync(void* host, size_t bytes) {
    if (!training_active || !host || !bytes) return -1;
    return vk_graph_sync_host(host, bytes, 0) ? 0 : -1;
}

void vk_training_end(void) {
    if (!training_active) return;
    (void)vk_graph_flush_wait();
    /* Training arrays are normally released after a step.  Mark every host
       identity touched by training as a new generation so malloc/stack
       address reuse cannot retain old inputs or gradients.  device_dirty is
       intentionally preserved: callers that kept a buffer can still download
       it through vk_graph_sync_host().  A later GPU forward output clears the
       host-dirty bit when it overwrites the same stable model tensor. */
    for (int i = 0; i < training_touched_count; i++) {
        if (training_touched_slots[i]) training_touched_slots[i]->host_dirty = 1;
        training_touched_slots[i] = NULL;
    }
    training_touched_count = 0;
    scratch_bump = 0;
    scratch_high_water = 0;
    training_dispatch_set_cursor = 0;
    training_active = 0;
}
#endif

/* See opengl_graph_alias_f32: device slots are keyed by host pointer and the
 * runtime pools many disjoint-lifetime tensors into one host span, so aliasing
 * across distinct host storage merges two spans on the device instead of
 * extending one tensor's lifetime. Distinct host storage must be a real copy. */
int vk_graph_alias_f32(const float* in, float* out, long n) {
    if (n <= 0 || !in || !out) return 0;
    if (in != out) return vk_graph_copy_f32(in, out, n);
    size_t bytes = (size_t)n * sizeof(float);
    VkTensorSlot* slot = graph_ensure_device(in, bytes, 0);
    if (!slot) return 0;
    graph_mark_device(slot);
    return 1;
}
