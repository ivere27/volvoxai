/* Query pools exist only during collection. Writes share the ordinary command
 * buffer, and reads follow the existing fence; never wait per node. */
struct VkTracePass {
    VxTraceScope* scope;
    VkQueryPool pool;
    size_t capacity, count;
    int failed;
    VxDeviceTraceSpan* spans;
    const char* program;
    VxTraceQueue queue, previous_queue;
    uint64_t host_before, anchor_device, anchor_host, anchor_error;
    int calibrated;
    PFN_vkCreateQueryPool create;
    PFN_vkDestroyQueryPool destroy;
    PFN_vkCmdResetQueryPool reset;
    PFN_vkCmdWriteTimestamp write;
    PFN_vkGetQueryPoolResults read;
};
static VxTraceQueue vk_trace_queue(void) {
    return (VxTraceQueue){vx_trace_object_id(&g_vulkan_device.trace_device_id),
        vx_trace_object_id(&g_vulkan_device.trace_queue_id), 0};
}
static int vk_trace_calibrate(VkTracePass* pass) {
#if defined(VK_EXT_calibrated_timestamps) && !defined(_WIN32)
    if (!g_vulkan_device.calibrated_timestamps) return 0;
    PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT domains =
        (PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT");
    PFN_vkGetCalibratedTimestampsEXT sample =
        (PFN_vkGetCalibratedTimestampsEXT)vkGetInstanceProcAddr(instance, "vkGetCalibratedTimestampsEXT");
    VkTimeDomainEXT available[16]; uint32_t count = 16;
    if (!domains || !sample || domains(physical_device, &count, available) != VK_SUCCESS) return 0;
    int monotonic = 0;
    for (uint32_t i = 0; i < count; i++) monotonic |= available[i] == VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT;
    if (!monotonic) return 0;
    VkCalibratedTimestampInfoEXT info[2] = {
        {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, NULL, VK_TIME_DOMAIN_DEVICE_EXT},
        {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, NULL, VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT}};
    uint64_t values[2], error;
    if (sample(device, 2, info, values, &error) != VK_SUCCESS || !error) return 0;
    pass->anchor_device = values[0]; pass->anchor_host = values[1]; pass->anchor_error = error;
    return 1;
#else
    (void)pass; return 0;
#endif
}
static uint64_t vk_trace_host_start(void) {
    VxTraceScope* scope = vx_engine_state_current()->profiling;
    return scope && vx_trace_nodes(scope) ? vx_trace_now_ns() : 0;
}
static void vk_trace_host_end(uint64_t start, const char* name, VxTraceActivity activity,
    VxMemorySpace source, VxMemorySpace destination, uint64_t bytes) {
    if (!start) return;
    VxTraceScope* scope = vx_engine_state_current()->profiling;
    VxDeviceTraceSpan span = {.name = name, .index = -1, .activity = activity,
        .copy_source = source, .copy_destination = destination, .copy_bytes = bytes,
        .queue = scope->queue.queue_id ? scope->queue : vk_trace_queue()};
    vx_trace_host_activity(scope, start, vx_trace_now_ns(), &span);
}
static void vk_trace_copy(VkCommandBuffer commands, VkBuffer source, VkBuffer destination,
    const VkBufferCopy* copy) {
    uint64_t start = vk_trace_host_start();
    VkTracePass* p = vk_context_current()->trace_pass;
    int slot = -1;
    VxMemorySpace from = source == vk_stage_buffer ? VX_MEMORY_SPACE_HOST_VISIBLE_DEVICE : VX_MEMORY_SPACE_DEVICE;
    VxMemorySpace to = destination == vk_stage_buffer ? VX_MEMORY_SPACE_HOST_VISIBLE_DEVICE : VX_MEMORY_SPACE_DEVICE;
    if (p && p->scope && vx_trace_device_nodes(p->scope)) {
        if (p->count < p->capacity) {
            slot = (int)p->count++;
            p->spans[slot] = (VxDeviceTraceSpan){.host_start_ns = start, .name = "Buffer copy", .index = -1,
                .phase = p->scope->work.phase, .activity = VX_TRACE_ACTIVITY_COPY,
                .copy_source = from, .copy_destination = to, .copy_bytes = copy->size};
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 0, NULL);
            vx_trace_device_status(p->scope, 1, 1, 0, 1);
            p->write(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, p->pool, (uint32_t)slot * 2);
        } else vx_trace_drop(p->scope);
    }
    vkCmdCopyBuffer(commands, source, destination, 1, copy);
    if (slot >= 0) p->write(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, p->pool, (uint32_t)slot * 2 + 1);
    vk_trace_host_end(start, "vkCmdCopyBuffer", VX_TRACE_ACTIVITY_COPY, from, to, copy->size);
}
static int vk_graph_begin_recording(void);
static void vk_trace_discard(VulkanContextState* context) {
    VkTracePass* p = context->trace_pass;
    if (!p) return;
    if (p->scope) p->scope->queue = p->previous_queue;
    context->dispatch = vkCmdDispatch; context->bind_pipeline = vkCmdBindPipeline;
    p->destroy(device, p->pool, NULL); free(p->spans); free(p);
    context->trace_pass = NULL;
}
static void VKAPI_CALL vk_trace_program_bind(VkCommandBuffer commands, VkPipelineBindPoint point, VkPipeline pipeline) {
    VulkanContextState* context = vk_context_current();
    VkTracePass* p = context->trace_pass;
    p->program = NULL;
    for (size_t i = 0; i < context->prepared_kernel_count; i++) {
        const VkPreparedKernel* kernel = &context->prepared_kernels[i];
        if (kernel->pipeline == pipeline) { p->program = kernel->definition->name; break; }
    }
    if (pipeline == matmul_pipeline) p->program = "linearF32";
    if (pipeline == matmul_tiled_pipeline) p->program = "linearF32Tiled";
    vkCmdBindPipeline(commands, point, pipeline);
}
static void VKAPI_CALL vk_trace_program_dispatch(VkCommandBuffer commands, uint32_t x, uint32_t y, uint32_t z) {
    VkTracePass* p = vk_context_current()->trace_pass;
    if (!p->program || p->count == p->capacity || p->failed) {
        vx_trace_drop(p->scope); vkCmdDispatch(commands, x, y, z); return;
    }
    size_t slot = p->count++;
    p->spans[slot] = vx_trace_program_span(p->scope, p->program, "main");
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 0, NULL);
    vx_trace_device_status(p->scope, 1, 1, 0, 1);
    p->write(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, p->pool, (uint32_t)slot * 2);
    vkCmdDispatch(commands, x, y, z);
    p->write(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, p->pool, (uint32_t)slot * 2 + 1);
}
static void vk_trace_begin(const char* name) {
    VxTraceScope* scope = vx_engine_state_current()->profiling;
    VulkanContextState* context = vk_context_current();
    if (!scope || !vx_trace_device_enabled(scope) || context->trace_pass) return;
    int available = g_vulkan_device.timestamp_bits && g_vulkan_device.timestamp_bits <= 64 && g_vulkan_device.timestamp_period > 0;
    VkTracePass functions = {0};
#define VX_QUERY_PROC(field, name) functions.field = (PFN_##name)vkGetInstanceProcAddr(instance, #name)
    VX_QUERY_PROC(create, vkCreateQueryPool);
    VX_QUERY_PROC(destroy, vkDestroyQueryPool);
    VX_QUERY_PROC(reset, vkCmdResetQueryPool);
    VX_QUERY_PROC(write, vkCmdWriteTimestamp);
    VX_QUERY_PROC(read, vkGetQueryPoolResults);
#undef VX_QUERY_PROC
    available = available && functions.create && functions.destroy && functions.reset && functions.write && functions.read;
    vx_trace_device_status(scope, available, available, 0, 0);
    vx_trace_program_status(scope, available);
    if (!available) return;
    size_t capacity = vx_trace_device_capacity(scope,
        vx_trace_device_nodes(scope) ? 1024 : 1);
    if (!capacity) { vx_trace_drop(scope); return; }
    VkTracePass* p = calloc(1, sizeof(*p));
    if (!p) { vx_trace_device_fail(scope); return; }
    *p = functions;
    p->scope = scope; p->capacity = capacity;
    p->queue = vk_trace_queue(); p->queue.submission_id = vx_trace_next_id();
    p->previous_queue = scope->queue;
    p->spans = calloc(capacity, sizeof(*p->spans));
    VkQueryPoolCreateInfo info = {0};
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = VK_QUERY_TYPE_TIMESTAMP; info.queryCount = (uint32_t)(capacity * 2);
    if (!p->spans || p->create(device, &info, NULL, &p->pool) != VK_SUCCESS) {
        vx_trace_device_fail(scope); free(p->spans); free(p); return;
    }
    if (!vk_graph_begin_recording()) {
        vx_trace_device_fail(scope); p->destroy(device, p->pool, NULL); free(p->spans); free(p); return;
    }
    scope->queue = p->queue;
    p->host_before = vx_trace_now_ns();
    p->calibrated = vk_trace_calibrate(p);
    p->reset(cmd_buf, p->pool, 0, info.queryCount);
    p->spans[0] = (VxDeviceTraceSpan){vx_trace_now_ns(), name, "", -1, 0};
    p->count = 1;
    p->write(cmd_buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, p->pool, 0);
    context->trace_pass = p;
    if (vx_trace_device_nodes(scope)) {
        context->dispatch = vk_trace_program_dispatch;
        context->bind_pipeline = vk_trace_program_bind;
    }
}
int vk_trace_node_begin(int index, const char* name, const char* output, int fused) {
    VulkanContextState* context = vk_context_current();
    VkTracePass* p = context ? context->trace_pass : NULL;
    if (!p || !p->scope || p->failed || !vx_trace_device_nodes(p->scope)) return -1;
    if (p->count == p->capacity || !vk_graph_begin_recording()) { vx_trace_drop(p->scope); return -1; }
    int token = (int)p->count++;
    p->spans[token] = (VxDeviceTraceSpan){vx_trace_now_ns(), name, output, index, fused, VX_TRACE_PHASE_FORWARD};
    /* Timestamp writes alone do not order later pipeline stages. This GPU-only
     * dependency prevents a node's interval from including unfinished preceding
     * nodes. It exists only in the instrumented schedule; no host wait is added. */
    vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 0, NULL);
    vx_trace_device_status(p->scope, 1, 1, 0, 1);
    p->write(cmd_buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, p->pool, (uint32_t)token * 2);
    return token;
}
void vk_trace_node_end(int token) {
    if (token < 0) return;
    VkTracePass* p = vk_context_current()->trace_pass;
    if (!vk_graph_begin_recording()) { p->failed = 1; return; }
    p->write(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, p->pool, (uint32_t)token * 2 + 1);
}
static void vk_trace_end_record(void) {
    if (vk_context_current()->trace_pass) vk_trace_node_end(0);
}
static void vk_trace_finish(int complete) {
    VulkanContextState* context = vk_context_current();
    VkTracePass* p = context->trace_pass;
    if (!p || !p->scope) return;
    context->dispatch = vkCmdDispatch; context->bind_pipeline = vkCmdBindPipeline;
    uint32_t bits = g_vulkan_device.timestamp_bits;
    uint64_t host_end = vx_trace_now_ns();
    VkTracePass end_anchor = {0};
    int calibrated_end = complete && vk_trace_calibrate(&end_anchor);
    uint64_t first = 0, total = 0; int pass_valid = 0;
    VxTraceClock pass_clock = {0};
    for (size_t i = 0; i < p->count; i++) {
        uint64_t values[4] = {0};
        int valid = complete && !p->failed && p->read(device, p->pool, (uint32_t)i * 2, 2,
            sizeof(values), values, sizeof(uint64_t) * 2,
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS && values[1] && values[3];
        VxDeviceTraceSpan* s = &p->spans[i];
        uint64_t duration;
        if (!valid || host_end < s->host_start_ns || !vx_trace_timestamp_duration(values[0], values[2], bits,
                g_vulkan_device.timestamp_period, host_end - s->host_start_ns, &duration)) {
            vx_trace_device_fail(p->scope); continue;
        }
        if (i == 0) {
            first = values[0]; total = duration; pass_valid = 1;
            if (p->calibrated && calibrated_end) {
                VxTraceClock a = vx_trace_clock_calibrated(first, p->anchor_device, bits,
                    g_vulkan_device.timestamp_period, p->anchor_host, p->anchor_error,
                    p->host_before, host_end, total);
                VxTraceClock b = vx_trace_clock_calibrated(first, end_anchor.anchor_device, bits,
                    g_vulkan_device.timestamp_period, end_anchor.anchor_host, end_anchor.anchor_error,
                    p->host_before, host_end, total);
                if (a.method && b.method) {
                    if (b.earliest_ns < a.earliest_ns) a.earliest_ns = b.earliest_ns;
                    if (b.latest_ns > a.latest_ns) a.latest_ns = b.latest_ns;
                    pass_clock = a;
                }
            }
        }
        s->queue = p->queue;
        uint64_t offset;
        if (pass_valid && vx_trace_timestamp_duration(first, values[0], bits, g_vulkan_device.timestamp_period,
                host_end - p->host_before, &offset)) {
            s->clock = vx_trace_clock_bounds(p->host_before, host_end, total, offset, duration);
            /* Keep every interval on the pass's one affine clock mapping. */
            if (s->clock.method && pass_clock.method) {
                s->clock = pass_clock;
                s->clock.earliest_ns += offset; s->clock.latest_ns += offset;
            }
        }
        vx_trace_device_span(p->scope, s, duration);
    }
    p->scope->queue = p->previous_queue;
    /* A failed wait may leave commands in flight. Keep the pool until context
     * destruction rather than destroying an object still referenced by them. */
    if (!complete && (context->command_pending || context->command_recording)) {
        p->scope = NULL; p->count = 0; p->failed = 1; return;
    }
    p->destroy(device, p->pool, NULL); free(p->spans); free(p);
    context->trace_pass = NULL;
}
