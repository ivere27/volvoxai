/* Native GPU (Vulkan/OpenGL/Metal) declared-domain proof.
 *
 * Included by public_api.c at the point these definitions used to sit.  This
 * is a fragment of that translation unit, not a separate one: the proof code
 * reads file-static declared-tensor helpers and must stay under the same
 * VOLVOXAI_ENABLE_* composition as its caller.
 *
 * Contents: backend limit queries, resource/allocation bounds, and the
 * per-operator `vx_native_gpu_*_proven` checks reached through
 * vx_native_gpu_node_domain_proof(), plus the vx_public_api_test_* shims that
 * expose those bounds to native/tests. */
typedef struct VxNativeGpuDomainLimits {
    uint64_t maximum_buffer_bytes;
    uint64_t maximum_uniform_bytes;
    uint64_t maximum_total_span_bytes;
    uint64_t fixed_compute_allocation_bytes;
    uint64_t fixed_staging_allocation_bytes;
    uint64_t maximum_scratch_bytes;
    uint64_t current_graph_scratch_bytes;
    uint64_t storage_alignment;
    uint32_t maximum_grid[3];
    uint32_t maximum_block[3];
    uint32_t maximum_threads_per_block;
    uint32_t maximum_tensor_slots;
    uint32_t maximum_storage_bindings;
    uint32_t maximum_uniform_bindings;
    uint32_t replay_plan_capacity;
    uint64_t replay_fixed_host_metadata_bytes;
} VxNativeGpuDomainLimits;

enum {
    VX_NATIVE_GPU_LAYOUT_LIMIT_QUERY_FAILED = -10,
    VX_NATIVE_GPU_LAYOUT_LIMIT_SET_INVALID = -11,
    VX_NATIVE_GPU_LAYOUT_SPAN_INVALID = -12,
    VX_NATIVE_GPU_LAYOUT_ARENA_EXHAUSTED = -13,
    VX_NATIVE_GPU_LAYOUT_QNORM_EXHAUSTED = -14,
    VX_NATIVE_GPU_LAYOUT_SLOT_EXHAUSTED = -15,
    VX_NATIVE_GPU_LAYOUT_SCRATCH_EXHAUSTED = -16,
};

static int vx_native_gpu_query_domain_limits(
        VolvoxAIEngineBackend backend,
        VxNativeGpuDomainLimits* limits) {
    if (!limits) return -1;
    memset(limits, 0, sizeof(*limits));
#if defined(VOLVOXAI_ENABLE_VULKAN) && VOLVOXAI_ENABLE_VULKAN
    if (backend == VOLVOXAI_BACKEND_VULKAN) {
        VulkanDomainLimits physical = {0};
        if (vk_query_domain_limits(&physical) != 0) return -1;
        limits->maximum_buffer_bytes = physical.maximum_storage_buffer_bytes;
        limits->maximum_uniform_bytes = physical.maximum_uniform_buffer_bytes;
        limits->maximum_total_span_bytes = physical.maximum_total_span_bytes;
        limits->fixed_compute_allocation_bytes =
            physical.compute_arena_allocation_bytes;
        limits->fixed_staging_allocation_bytes =
            physical.staging_allocation_bytes;
        limits->maximum_scratch_bytes = physical.maximum_scratch_bytes;
        limits->current_graph_scratch_bytes =
            physical.current_graph_scratch_bytes;
        limits->storage_alignment = physical.storage_alignment;
        memcpy(limits->maximum_grid, physical.maximum_workgroups,
               sizeof(limits->maximum_grid));
        memcpy(limits->maximum_block, physical.maximum_workgroup_size,
               sizeof(limits->maximum_block));
        limits->maximum_threads_per_block =
            physical.maximum_workgroup_invocations;
        limits->maximum_tensor_slots = physical.maximum_tensor_slots;
        limits->maximum_storage_bindings =
            physical.maximum_storage_bindings;
        limits->maximum_uniform_bindings =
            physical.maximum_uniform_bindings;
    return 0;
}
#endif
#if defined(VOLVOXAI_ENABLE_OPENGL) && VOLVOXAI_ENABLE_OPENGL
    if (backend == VOLVOXAI_BACKEND_OPENGL) {
        OpenGLDomainLimits physical = {0};
        if (opengl_query_domain_limits(&physical) != 0) return -1;
        limits->maximum_buffer_bytes =
            physical.maximum_storage_buffer_bytes;
        limits->maximum_uniform_bytes =
            physical.maximum_uniform_buffer_bytes;
        limits->storage_alignment = 1u;
        memcpy(limits->maximum_grid, physical.maximum_workgroups,
               sizeof(limits->maximum_grid));
        memcpy(limits->maximum_block, physical.maximum_workgroup_size,
               sizeof(limits->maximum_block));
        limits->maximum_threads_per_block =
            physical.maximum_workgroup_invocations;
        limits->maximum_tensor_slots = physical.maximum_tensor_slots;
        limits->maximum_storage_bindings =
            physical.maximum_storage_bindings;
        limits->maximum_uniform_bindings =
            physical.maximum_uniform_bindings;
        return 0;
    }
#endif
#if defined(VOLVOXAI_ENABLE_METAL) && VOLVOXAI_ENABLE_METAL
    if (backend == VOLVOXAI_BACKEND_METAL) {
        MetalDomainLimits physical = {0};
        if (metal_query_domain_limits(&physical) != 0) return -1;
        limits->maximum_buffer_bytes = physical.maximum_buffer_bytes;
        limits->maximum_uniform_bytes = physical.maximum_buffer_bytes;
        limits->storage_alignment = 1u;
        memcpy(limits->maximum_grid, physical.maximum_workgroups,
               sizeof(limits->maximum_grid));
        memcpy(limits->maximum_block, physical.maximum_workgroup_size,
               sizeof(limits->maximum_block));
        limits->maximum_threads_per_block =
            physical.maximum_threads_per_workgroup;
        limits->maximum_tensor_slots = physical.maximum_tensor_slots;
        limits->maximum_storage_bindings = physical.maximum_bindings;
        limits->maximum_uniform_bindings = 1u;
        return 0;
    }
#endif
#if defined(VOLVOXAI_ENABLE_CUDA) && VOLVOXAI_ENABLE_CUDA
    if (backend == VOLVOXAI_BACKEND_CUDA) {
        CudaDomainLimits physical = {0};
        if (cuda_query_domain_limits(&physical) != 0) return -1;
        limits->maximum_buffer_bytes = physical.total_memory_bytes;
        limits->maximum_uniform_bytes = physical.total_memory_bytes;
        limits->maximum_total_span_bytes = physical.total_memory_bytes;
        limits->storage_alignment = 1u;
        memcpy(limits->maximum_grid, physical.maximum_grid,
               sizeof(limits->maximum_grid));
        memcpy(limits->maximum_block, physical.maximum_block,
               sizeof(limits->maximum_block));
        limits->maximum_threads_per_block =
            physical.maximum_threads_per_block;
        limits->maximum_tensor_slots = physical.maximum_tensor_slots;
        limits->replay_plan_capacity = physical.replay_plan_capacity;
        limits->replay_fixed_host_metadata_bytes =
            physical.replay_fixed_host_metadata_bytes;
        limits->maximum_storage_bindings = UINT32_MAX;
        limits->maximum_uniform_bindings = UINT32_MAX;
        return 0;
    }
#endif
    return -1;
}

static int vx_native_gpu_slot_capacity_proven(
        uint64_t tensor_count,
        uint64_t node_count,
        uint64_t physical_span_count,
        uint64_t slot_limit,
        uint64_t* required_out) {
    uint64_t required;
    if (!required_out || !slot_limit ||
        node_count > (UINT64_MAX - tensor_count) / 4u ||
        physical_span_count >
            UINT64_MAX - tensor_count - node_count * 4u)
        return 0;
    required = tensor_count + node_count * 4u + physical_span_count;
    *required_out = required;
    return required <= slot_limit;
}

static int vx_native_gpu_accumulate_resource_bytes(
    uint64_t* total, uint64_t bytes, uint64_t copies);
static int vx_native_gpu_align_resource_bytes(
    uint64_t bytes, uint64_t alignment, uint64_t* aligned_out);

/* Vulkan reserves one aligned tail arena for maximum-domain normalization
 * statistics followed by all shape-invariant dispatch data observed during
 * the validation forward.  This mirrors vk_graph_domain_scratch_layout: each
 * statistics range ends at an independently aligned cursor and the measured
 * bootstrap high-water mark is then reserved after both ranges. */
static int vx_native_gpu_vulkan_scratch_capacity_proven(
        uint64_t qgroupnorm_stats_bytes,
        uint64_t qlayernorm_stats_bytes,
        uint64_t validation_graph_scratch_bytes,
        uint64_t storage_alignment,
        uint64_t maximum_scratch_bytes,
        uint64_t* required_out) {
    uint64_t required = 0u;
    uint64_t aligned;
    if (!required_out || !storage_alignment || !maximum_scratch_bytes ||
        validation_graph_scratch_bytes % storage_alignment)
        return 0;
    if (qgroupnorm_stats_bytes) {
        if (!vx_native_gpu_align_resource_bytes(
                qgroupnorm_stats_bytes, storage_alignment, &aligned) ||
            !vx_native_gpu_accumulate_resource_bytes(
                &required, aligned, 1u))
            return 0;
    }
    if (qlayernorm_stats_bytes) {
        if (!vx_native_gpu_align_resource_bytes(
                qlayernorm_stats_bytes, storage_alignment, &aligned) ||
            !vx_native_gpu_accumulate_resource_bytes(
                &required, aligned, 1u))
            return 0;
    }
    if (!vx_native_gpu_accumulate_resource_bytes(
            &required, validation_graph_scratch_bytes, 1u))
        return 0;
    *required_out = required;
    return required <= maximum_scratch_bytes;
}

static int vx_native_gpu_maximum_layout_proof(
        VxEngineState* validation,
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        VolvoxAIEngineBackend backend,
        VxNativeGpuDomainLimits* limits,
        uint64_t* arena_bytes_out,
        uint64_t* device_span_bytes_out,
        uint64_t* span_count_out,
        uint64_t* qgroupnorm_stats_bytes_out,
        uint64_t* qlayernorm_stats_bytes_out,
        uint64_t* scratch_required_bytes_out) {
    VolvoxAIEngineMaximumTensor* maximums = NULL;
    VxEngineStateScope scope;
    const VxDynamicShapeMaximumLayout* layout;
    uint64_t maximum_slots;
    int configured;
    int result = -1;
    if (!validation || !tensors || !tensor_count || !limits ||
        !arena_bytes_out || !device_span_bytes_out || !span_count_out ||
        !qgroupnorm_stats_bytes_out || !qlayernorm_stats_bytes_out ||
        !scratch_required_bytes_out ||
        !vx_native_gpu_backend(backend) || validation->tensor_count < 0 ||
        validation->node_count < 0)
        return -1;
    maximums = (VolvoxAIEngineMaximumTensor*)calloc(
        tensor_count, sizeof(*maximums));
    if (!maximums) return -2;
    for (size_t index = 0; index < tensor_count; index++) {
        maximums[index].name = tensors[index].name;
        maximums[index].dtype = (int)tensors[index].dtype;
        maximums[index].rank = (int)tensors[index].rank;
        for (uint32_t axis = 0; axis < tensors[index].rank; axis++)
            maximums[index].maximum_shape[axis] =
                (int)tensors[index].maximums[axis];
        maximums[index].maximum_byte_size = tensors[index].maximum_byte_size;
    }
    scope = vx_engine_state_scope_enter(validation);
    configured = volvoxai_engine_configure_dynamic_shape_domain(
        maximums, tensor_count);
    if (configured != 0) {
        result = configured;
        goto done;
    }
    if (vx_native_gpu_query_domain_limits(backend, limits) != 0) {
        result = VX_NATIVE_GPU_LAYOUT_LIMIT_QUERY_FAILED;
        goto done;
    }
    layout = &validation->dynamic_shape_maximum_layout;
    if (!layout->configured || layout->tensor_count < 0 ||
        (size_t)layout->tensor_count != tensor_count ||
        layout->physical_span_count <= 0 || !layout->arena_bytes ||
        !layout->physical_capacities ||
        !limits->maximum_buffer_bytes || !limits->maximum_uniform_bytes ||
        !limits->maximum_grid[0] || !limits->maximum_grid[1] ||
        !limits->storage_alignment || !limits->maximum_grid[2] ||
        !limits->maximum_block[0] ||
        !limits->maximum_block[1] || !limits->maximum_block[2] ||
        !limits->maximum_threads_per_block ||
        (backend == VOLVOXAI_BACKEND_VULKAN &&
         (!limits->fixed_compute_allocation_bytes ||
          !limits->fixed_staging_allocation_bytes ||
          !limits->maximum_scratch_bytes ||
          limits->current_graph_scratch_bytes >
              limits->maximum_scratch_bytes ||
          limits->current_graph_scratch_bytes %
              limits->storage_alignment)) ||
        limits->maximum_storage_bindings < 7u ||
        limits->maximum_uniform_bindings < 1u) {
        result = VX_NATIVE_GPU_LAYOUT_LIMIT_SET_INVALID;
        goto done;
    }
    *device_span_bytes_out = 0u;
    for (int index = 0; index < layout->physical_span_count; index++) {
        uint64_t capacity = (uint64_t)layout->physical_capacities[index];
        uint64_t remainder;
        uint64_t aligned;
        if (!capacity || capacity > limits->maximum_buffer_bytes) {
            result = VX_NATIVE_GPU_LAYOUT_SPAN_INVALID;
            goto done;
        }
        remainder = capacity % limits->storage_alignment;
        if (remainder && capacity >
                UINT64_MAX - (limits->storage_alignment - remainder)) {
            result = VX_NATIVE_GPU_LAYOUT_SPAN_INVALID;
            goto done;
        }
        aligned = remainder
            ? capacity + (limits->storage_alignment - remainder)
            : capacity;
        if (aligned > UINT64_MAX - *device_span_bytes_out) {
            result = VX_NATIVE_GPU_LAYOUT_SPAN_INVALID;
            goto done;
        }
        *device_span_bytes_out += aligned;
    }
    if (limits->maximum_total_span_bytes &&
        *device_span_bytes_out > limits->maximum_total_span_bytes) {
        result = VX_NATIVE_GPU_LAYOUT_ARENA_EXHAUSTED;
        goto done;
    }
    if ((layout->qgroupnorm_stats_bytes &&
         layout->qgroupnorm_stats_bytes > limits->maximum_buffer_bytes) ||
        (layout->qlayernorm_stats_bytes &&
         layout->qlayernorm_stats_bytes > limits->maximum_buffer_bytes)) {
        result = VX_NATIVE_GPU_LAYOUT_QNORM_EXHAUSTED;
        goto done;
    }
    *scratch_required_bytes_out = 0u;
    if (backend == VOLVOXAI_BACKEND_VULKAN &&
        !vx_native_gpu_vulkan_scratch_capacity_proven(
            (uint64_t)layout->qgroupnorm_stats_bytes,
            (uint64_t)layout->qlayernorm_stats_bytes,
            limits->current_graph_scratch_bytes,
            limits->storage_alignment,
            limits->maximum_scratch_bytes,
            scratch_required_bytes_out)) {
        result = VX_NATIVE_GPU_LAYOUT_SCRATCH_EXHAUSTED;
        goto done;
    }
    /* Bootstrap graph slots are keyed by host address: every T can own at
     * most one slot, while the qualified quantized routes add at most four
     * persistent metadata keys per node (QConv2D is the largest at three).
     * Domain publication discards logical transient slots, retains the
     * immutable subset, and adds one slot per physical span. Keeping the full
     * T count plus both conservative additions therefore bounds both phases
     * without charging the logical tensor table twice. */
    if (!vx_native_gpu_slot_capacity_proven(
            (uint64_t)validation->tensor_count,
            (uint64_t)validation->node_count,
            (uint64_t)layout->physical_span_count,
            (uint64_t)limits->maximum_tensor_slots,
            &maximum_slots)) {
        result = VX_NATIVE_GPU_LAYOUT_SLOT_EXHAUSTED;
        goto done;
    }
    *arena_bytes_out = (uint64_t)layout->arena_bytes;
    *span_count_out = (uint64_t)layout->physical_span_count;
    *qgroupnorm_stats_bytes_out =
        (uint64_t)layout->qgroupnorm_stats_bytes;
    *qlayernorm_stats_bytes_out =
        (uint64_t)layout->qlayernorm_stats_bytes;
    result = 0;
done:
    vx_engine_state_scope_leave(scope);
    free(maximums);
    return result;
}

static VxStatus vx_native_gpu_domain_reject(const cJSON* node,
                                      const char* reason,
                                      char* evidence,
                                      size_t evidence_capacity) {
    const cJSON* node_id = node
        ? cJSON_GetObjectItemCaseSensitive(node, "id") : NULL;
    const cJSON* op = node
        ? cJSON_GetObjectItemCaseSensitive(node, "opType") : NULL;
    snprintf(evidence, evidence_capacity,
             "unsupported_node=%s;unsupported_op=%s;"
             "required=native-gpu-full-bounded-domain;predicate_reason=%s",
             cJSON_IsString(node_id) && node_id->valuestring
                 ? node_id->valuestring : "<unknown>",
             cJSON_IsString(op) && op->valuestring
                 ? op->valuestring : "<unknown>",
             reason ? reason : "invalid");
    return VX_STATUS_BACKEND_UNSUPPORTED;
}

static const VxDeclaredTensor* vx_native_gpu_node_input_named(
        const cJSON* node,
        const char* role,
        const VxDeclaredTensor* tensors,
        size_t tensor_count) {
    const cJSON* inputs = node
        ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
    const cJSON* reference = cJSON_IsObject(inputs) && role
        ? cJSON_GetObjectItemCaseSensitive(inputs, role) : NULL;
    return cJSON_IsString(reference) && reference->valuestring
        ? vx_declared_tensor_find(tensors, tensor_count, reference->valuestring)
        : NULL;
}

static const VxDeclaredTensor* vx_native_gpu_node_input_domain(
        const cJSON* node,
        const char* role,
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        const VxEngineState* validation,
        VxDeclaredTensor* fixed_storage) {
    const VxDeclaredTensor* logical = vx_native_gpu_node_input_named(
        node, role, tensors, tensor_count);
    const cJSON* inputs;
    const cJSON* reference;
    if (logical) return logical;
    if (!validation || !fixed_storage) return NULL;
    inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
    reference = cJSON_IsObject(inputs)
        ? cJSON_GetObjectItemCaseSensitive(inputs, role) : NULL;
    if (!cJSON_IsString(reference) || !reference->valuestring) return NULL;
    for (int index = 0; index < validation->tensor_count; index++) {
        const T* tensor = &validation->tensors[index];
        if (strcmp(tensor->name, reference->valuestring)) continue;
        if (tensor->ndim <= 0 || tensor->ndim > (int)VX_MAX_TENSOR_RANK ||
            tensor->numel <= 0 || !tensor->elem_size) return NULL;
        memset(fixed_storage, 0, sizeof(*fixed_storage));
        fixed_storage->name = (char*)tensor->name;
        fixed_storage->dtype = tensor->dtype;
        fixed_storage->rank = (uint32_t)tensor->ndim;
        for (uint32_t axis = 0; axis < fixed_storage->rank; axis++) {
            if (tensor->shape[axis] <= 0) return NULL;
            fixed_storage->kinds[axis] = VX_DIMENSION_FIXED;
            fixed_storage->minimums[axis] = tensor->shape[axis];
            fixed_storage->maximums[axis] = tensor->shape[axis];
            fixed_storage->multiples[axis] = 1;
        }
        if ((uint64_t)tensor->numel > SIZE_MAX / tensor->elem_size)
            return NULL;
        fixed_storage->maximum_byte_size =
            (size_t)tensor->numel * tensor->elem_size;
        return fixed_storage;
    }
    return NULL;
}

static const T* vx_native_gpu_validation_tensor_find(
        const VxEngineState* validation,
        const char* name) {
    if (!validation || !name) return NULL;
    for (int index = 0; index < validation->tensor_count; index++)
        if (!strcmp(validation->tensors[index].name, name))
            return &validation->tensors[index];
    return NULL;
}

static const T* vx_native_gpu_validation_node_input(
        const VxEngineState* validation,
        const Node* node,
        const char* role) {
    if (!validation || !node || !role) return NULL;
    for (int index = 0; index < node->nin; index++)
        if (!strcmp(node->ins[index].key, role))
            return vx_native_gpu_validation_tensor_find(
                validation, node->ins[index].name);
    return NULL;
}

static int vx_native_gpu_accumulate_resource_bytes(uint64_t* total,
                                             uint64_t bytes,
                                             uint64_t copies) {
    if (!total || !copies || bytes > (UINT64_MAX - *total) / copies)
        return 0;
    *total += bytes * copies;
    return 1;
}

static int vx_native_gpu_align_resource_bytes(uint64_t bytes,
                                               uint64_t alignment,
                                               uint64_t* aligned_out) {
    uint64_t remainder;
    if (!bytes || !alignment || !aligned_out) return 0;
    remainder = bytes % alignment;
    if (remainder && bytes > UINT64_MAX - (alignment - remainder)) return 0;
    *aligned_out = remainder ? bytes + (alignment - remainder) : bytes;
    return 1;
}

/* Public built-in results are published one output at a time.  For output i,
 * all preceding snapshots, the temporary readback buffer, and the newly
 * appended owned snapshot coexist until the temporary is released.  This is
 * intentionally an ordered peak rather than 2 * sum(outputs): only the
 * current output is duplicated. */
static int vx_native_gpu_result_publication_peak(
        const VxModel* model,
        uint64_t* snapshot_bytes_out,
        uint64_t* publication_peak_out) {
    uint64_t published = 0u;
    uint64_t peak = 0u;
    if (!model || !snapshot_bytes_out || !publication_peak_out)
        return 0;
    for (size_t index = 0; index < model->output_count; index++) {
        uint64_t bytes = (uint64_t)model->outputs[index].maximum_byte_size;
        uint64_t candidate;
        if (bytes > (UINT64_MAX - published) / 2u)
            return 0;
        candidate = published + bytes * 2u;
        if (candidate > peak) peak = candidate;
        if (bytes > UINT64_MAX - published) return 0;
        published += bytes;
    }
    *snapshot_bytes_out = published;
    *publication_peak_out = peak;
    return 1;
}

static int vx_native_gpu_maximum_signature_bytes(
        const VxModel* model, uint64_t* bytes_out) {
    uint64_t bytes = 4u;
    if (!model || !bytes_out) return 0;
    for (size_t index = 0; index < model->input_count; index++) {
        uint64_t name_bytes = (uint64_t)strlen(model->inputs[index].name);
        uint64_t axes_bytes = (uint64_t)model->inputs[index].rank * 24u;
        if (name_bytes > UINT64_MAX - 4u - axes_bytes ||
            bytes > UINT64_MAX - name_bytes - 4u - axes_bytes)
            return 0;
        bytes += name_bytes + 4u + axes_bytes;
    }
    *bytes_out = bytes;
    return 1;
}

/* A full replay-cache miss owns the candidate signature before it can select
 * and destroy the LRU victim. GraphExec count remains at the retained cache
 * capacity, but the host metadata peak must include that fifth transactional
 * signature. */
static int vx_native_gpu_cuda_replay_metadata_peak(
        uint64_t fixed_host_metadata_bytes,
        uint64_t maximum_signature_bytes,
        uint64_t replay_plan_capacity,
        uint64_t* peak_out) {
    uint64_t peak = 0u;
    if (!fixed_host_metadata_bytes || !maximum_signature_bytes ||
        !replay_plan_capacity || replay_plan_capacity == UINT64_MAX ||
        !peak_out ||
        !vx_native_gpu_accumulate_resource_bytes(
            &peak, fixed_host_metadata_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &peak, maximum_signature_bytes, replay_plan_capacity + 1u))
        return 0;
    *peak_out = peak;
    return 1;
}

/* Bound every heap-owned maximum-layout/plan allocation in a dynamic native
 * GPU context.  A cold bind can hold a full four-entry LRU while constructing
 * its fifth candidate, together with the resolved request/signature and the
 * larger of plan-build or bind-publication temporaries.  Maximum-layout build
 * has a separate phase; take the larger phase instead of summing allocations
 * that cannot coexist. */
static int vx_native_gpu_dynamic_metadata_peak(
        const VxModel* model,
        uint64_t logical_tensor_count,
        uint64_t engine_tensor_count,
        uint64_t physical_span_count,
        uint64_t* peak_out) {
    uint64_t signature_bytes;
    uint64_t plan_bytes = 0u;
    uint64_t maximum_layout_bytes = 0u;
    uint64_t layout_build_temporary = 0u;
    uint64_t layout_build_peak;
    uint64_t resolved_request_bytes = 0u;
    uint64_t plan_build_temporary = 0u;
    uint64_t bind_temporary = sizeof(void*);
    uint64_t transaction_temporary;
    uint64_t plan_transaction_peak = 0u;
    if (!model || !logical_tensor_count || !peak_out ||
        !vx_native_gpu_maximum_signature_bytes(model, &signature_bytes))
        return 0;

    if (!vx_native_gpu_accumulate_resource_bytes(
            &plan_bytes, signature_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_bytes, logical_tensor_count, sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_bytes, logical_tensor_count, 2u * sizeof(size_t)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_bytes, logical_tensor_count, 9u * sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &maximum_layout_bytes, logical_tensor_count, sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &maximum_layout_bytes, logical_tensor_count,
            4u * sizeof(size_t)))
        return 0;

    /* The public maximum descriptors remain live while the private maximum
     * layout builder owns starts, ends, aliases, buffer_of, buffer_until and
     * its two size_t arrays. descriptor/producers are indexed by the complete
     * private T table. */
    if (!vx_native_gpu_accumulate_resource_bytes(
            &layout_build_temporary, logical_tensor_count,
            sizeof(VolvoxAIEngineMaximumTensor)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &layout_build_temporary, logical_tensor_count,
            5u * sizeof(int) + 2u * sizeof(size_t)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &layout_build_temporary, engine_tensor_count,
            3u * sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &resolved_request_bytes, logical_tensor_count,
            sizeof(VolvoxAIEngineResolvedTensor)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &resolved_request_bytes, signature_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &resolved_request_bytes, (uint64_t)model->input_count,
            sizeof(VolvoxAIEngineInputBinding)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_build_temporary, engine_tensor_count, sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &bind_temporary, logical_tensor_count,
            sizeof(int) + sizeof(void*)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &bind_temporary, physical_span_count,
            sizeof(VolvoxAIEnginePhysicalSpan)))
        return 0;
    if (maximum_layout_bytes > UINT64_MAX - layout_build_temporary)
        return 0;
    layout_build_peak = maximum_layout_bytes + layout_build_temporary;
    transaction_temporary = plan_build_temporary > bind_temporary
        ? plan_build_temporary : bind_temporary;
    if (!vx_native_gpu_accumulate_resource_bytes(
            &plan_transaction_peak, maximum_layout_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_transaction_peak, plan_bytes,
            VX_DYNAMIC_SHAPE_PLAN_CACHE_CAPACITY + 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_transaction_peak, resolved_request_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_transaction_peak, transaction_temporary, 1u))
        return 0;
    *peak_out = layout_build_peak > plan_transaction_peak
        ? layout_build_peak : plan_transaction_peak;
    return 1;
}

static int vx_native_gpu_resident_capacity_proven(
        uint64_t base_bytes,
        uint64_t result_publication_bytes,
        uint64_t dynamic_metadata_bytes,
        uint64_t hard_limit_bytes,
        uint64_t* required_out) {
    uint64_t required = base_bytes;
    if (!required_out ||
        !vx_native_gpu_accumulate_resource_bytes(
            &required, result_publication_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &required, dynamic_metadata_bytes, 1u))
        return 0;
    *required_out = required;
    return !hard_limit_bytes || required <= hard_limit_bytes;
}

static int vx_native_gpu_fixed_allocation_bound(
        uint64_t compute_allocation_bytes,
        uint64_t staging_allocation_bytes,
        uint64_t* total_out) {
    if (!total_out || !compute_allocation_bytes ||
        !staging_allocation_bytes ||
        compute_allocation_bytes > UINT64_MAX - staging_allocation_bytes)
        return 0;
    *total_out = compute_allocation_bytes + staging_allocation_bytes;
    return 1;
}

static int vx_native_gpu_accumulate_immutable_allocation(
        uint64_t host_bytes,
        uint64_t device_bytes,
        uint64_t device_alignment,
        uint64_t* resident_total,
        uint64_t* device_total) {
    uint64_t aligned_device;
    if (!host_bytes || !device_bytes || !resident_total || !device_total ||
        !vx_native_gpu_align_resource_bytes(
            device_bytes, device_alignment, &aligned_device) ||
        !vx_native_gpu_accumulate_resource_bytes(
            resident_total, host_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            resident_total, aligned_device, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            device_total, aligned_device, 1u))
        return 0;
    return 1;
}

static int vx_native_gpu_maskless_qsdpa_immutable_bound(
        VolvoxAIEngineBackend backend,
        int has_mask,
        uint64_t device_alignment,
        uint64_t* resident_total,
        uint64_t* device_total) {
    /* Vulkan/OpenGL/Metal bind a shared static I32 zero when QSDPA has no
     * graph mask. It is not a logical/validation tensor, but graph_ensure_*()
     * retains it as an immutable slot. CUDA passes a null device pointer and
     * therefore owns no corresponding allocation. */
    if (has_mask || backend == VOLVOXAI_BACKEND_CUDA) return 1;
    return vx_native_gpu_accumulate_immutable_allocation(
        sizeof(int32_t), sizeof(int32_t), device_alignment,
        resident_total, device_total);
}

typedef struct {
    void* values;
    size_t elements;
    void* next;
} VxNativeGpuZeroBiasBackingProof;

static int vx_native_gpu_accumulate_zero_bias_backing(
        uint64_t* resident_total) {
    return vx_native_gpu_accumulate_resource_bytes(
        resident_total, sizeof(VxNativeGpuZeroBiasBackingProof), 1u);
}

/* A native GPU may materialize an immutable input into a node-local representation
 * (for example an F16-to-F32 or convolution-layout cache), and quantized
 * nodes own aligned affine metadata that is not itself represented by T.
 * Charge one host representation and one alignment-rounded device allocation
 * per consumer. This is deliberately conservative when consumers share a
 * cache key, but it remains sound for Vulkan's suballocated hard arena. Byte
 * tensors occupy complete u32 words on Vulkan/OpenGL/Metal, so their device
 * allocation is rounded independently from the exact host representation. */
static int vx_native_gpu_validation_immutable_bytes_bound(
        const VxEngineState* validation,
        const VxDeclaredTensor* logical_tensors,
        size_t logical_tensor_count,
        VolvoxAIEngineBackend backend,
        uint64_t device_alignment,
        uint64_t* bytes_out,
        uint64_t* device_bytes_out) {
    uint64_t total = 0u;
    uint64_t device_total = 0u;
    if (!validation || !logical_tensors || !device_alignment || !bytes_out ||
        !device_bytes_out ||
        validation->node_count < 0 || validation->tensor_count < 0)
        return 0;
    for (int node_index = 0; node_index < validation->node_count;
         node_index++) {
        const Node* node = &validation->nodes[node_index];
        for (int input_index = 0; input_index < node->nin; input_index++) {
            const T* tensor = vx_native_gpu_validation_tensor_find(
                validation, node->ins[input_index].name);
            uint64_t element_size;
            uint64_t bytes;
            uint64_t device_bytes;
            if (!tensor || tensor->numel <= 0 || !tensor->elem_size)
                return 0;
            if (vx_declared_tensor_find(
                    logical_tensors, logical_tensor_count, tensor->name))
                continue;
            element_size = tensor->dtype == T_F16
                ? (uint64_t)sizeof(float) : (uint64_t)tensor->elem_size;
            if ((uint64_t)tensor->numel > UINT64_MAX / element_size)
                return 0;
            bytes = (uint64_t)tensor->numel * element_size;
            device_bytes = bytes;
            if (tensor->dtype == T_I8 || tensor->dtype == T_U8) {
                if (device_bytes > UINT64_MAX - 3u) return 0;
                device_bytes = (device_bytes + 3u) & ~UINT64_C(3);
            }
            if (!vx_native_gpu_accumulate_immutable_allocation(
                    bytes, device_bytes, device_alignment,
                    &total, &device_total))
                return 0;
        }
        if ((!strcmp(node->op, "Linear") || !strcmp(node->op, "MatMul") ||
             !strcmp(node->op, "Gemm")) &&
            backend != VOLVOXAI_BACKEND_CUDA) {
            const T* output = vx_native_gpu_validation_tensor_find(
                validation, node->out);
            const T* bias = vx_native_gpu_validation_node_input(
                validation, node, "bias");
            uint64_t aligned_dummy;
            /* Vulkan/OpenGL retain these graph bindings. Metal creates the
             * equivalent buffers per dispatch. Charging one independently
             * aligned allocation per node is conservative for both lifetime
             * models and covers output-major's otherwise hidden scale slot. */
            if (!output || output->ndim <= 0 ||
                output->shape[output->ndim - 1] <= 0 ||
                (!bias &&
                 !vx_native_gpu_accumulate_immutable_allocation(
                     (uint64_t)output->shape[output->ndim - 1] *
                         sizeof(float),
                     (uint64_t)output->shape[output->ndim - 1] *
                         sizeof(float),
                     device_alignment, &total, &device_total)) ||
                !vx_native_gpu_align_resource_bytes(
                    sizeof(float), device_alignment, &aligned_dummy) ||
                !vx_native_gpu_accumulate_resource_bytes(
                    &total, aligned_dummy, 1u) ||
                !vx_native_gpu_accumulate_resource_bytes(
                    &device_total, aligned_dummy, 1u))
                return 0;
            if (!bias &&
                (backend == VOLVOXAI_BACKEND_VULKAN ||
                 backend == VOLVOXAI_BACKEND_OPENGL) &&
                !vx_native_gpu_accumulate_zero_bias_backing(&total))
                return 0;
        }
        if ((!strcmp(node->op, "QLinear") ||
             !strcmp(node->op, "QMatMul") ||
             !strcmp(node->op, "QGemm")) &&
            validation->qlinear_metadata[node_index].valid) {
            const QLinearMetadata* metadata =
                &validation->qlinear_metadata[node_index];
            uint64_t component_bytes = (uint64_t)metadata->d_out *
                sizeof(uint32_t);
            if (metadata->d_out <= 0 ||
                !vx_native_gpu_accumulate_immutable_allocation(
                    component_bytes, component_bytes, device_alignment,
                    &total, &device_total) ||
                !vx_native_gpu_accumulate_immutable_allocation(
                    component_bytes, component_bytes, device_alignment,
                    &total, &device_total) ||
                !vx_native_gpu_accumulate_immutable_allocation(
                    component_bytes, component_bytes, device_alignment,
                    &total, &device_total) ||
                (metadata->packed_weight &&
                 !vx_native_gpu_accumulate_resource_bytes(
                    &total, metadata->packed_weight_bytes, 1u)))
                return 0;
        } else if (!strcmp(node->op, "QConv2D") &&
                   validation->qconv_metadata[node_index].valid) {
            const QConv2DMetadata* metadata =
                &validation->qconv_metadata[node_index];
            uint64_t component_bytes =
                (uint64_t)metadata->output_channels *
                sizeof(uint32_t);
            uint64_t small_c_bytes = 0u;
            if (metadata->small_c_packed_weight) {
                small_c_bytes = (uint64_t)metadata->kernel_height;
                if (!metadata->kernel_width ||
                    small_c_bytes > UINT64_MAX / metadata->kernel_width)
                    return 0;
                small_c_bytes *= metadata->kernel_width;
                if (!metadata->input_per_group ||
                    small_c_bytes > UINT64_MAX / metadata->input_per_group)
                    return 0;
                small_c_bytes *= metadata->input_per_group;
                if (!metadata->output_channels ||
                    small_c_bytes > UINT64_MAX / metadata->output_channels)
                    return 0;
                small_c_bytes *= metadata->output_channels;
            }
            if (!metadata->output_channels ||
                !vx_native_gpu_accumulate_immutable_allocation(
                    component_bytes, component_bytes, device_alignment,
                    &total, &device_total) ||
                !vx_native_gpu_accumulate_immutable_allocation(
                    component_bytes, component_bytes, device_alignment,
                    &total, &device_total) ||
                !vx_native_gpu_accumulate_immutable_allocation(
                    component_bytes, component_bytes, device_alignment,
                    &total, &device_total) ||
                (metadata->packed_weight &&
                 !vx_native_gpu_accumulate_resource_bytes(
                    &total, metadata->packed_weight_bytes, 1u)) ||
                (small_c_bytes && !vx_native_gpu_accumulate_resource_bytes(
                    &total, small_c_bytes, 1u)))
                return 0;
            if (!metadata->bias && backend != VOLVOXAI_BACKEND_CUDA &&
                !vx_native_gpu_accumulate_zero_bias_backing(&total))
                return 0;
        } else if (!strcmp(node->op, "QEmbedding") &&
                   validation->qembedding_metadata[node_index].valid) {
            const QEmbeddingMetadata* metadata =
                &validation->qembedding_metadata[node_index];
            uint64_t component_bytes = (uint64_t)metadata->vocab *
                sizeof(uint32_t);
            if (!metadata->vocab ||
                !vx_native_gpu_accumulate_immutable_allocation(
                    component_bytes, component_bytes, device_alignment,
                    &total, &device_total) ||
                !vx_native_gpu_accumulate_immutable_allocation(
                    component_bytes, component_bytes, device_alignment,
                    &total, &device_total))
                return 0;
        }
        if (!strcmp(node->op, "QSDPA") &&
            !vx_native_gpu_maskless_qsdpa_immutable_bound(
                backend,
                vx_native_gpu_validation_node_input(
                    validation, node, "mask") != NULL,
                device_alignment, &total, &device_total))
            return 0;
        /* CUDA retains each QSiLU/QGELU descriptor's 256-byte lookup table as
         * an immutable device slot through invariant preload/domain
         * publication. Charging one table per node is conservative when two
         * nodes share the cache key and exact when their descriptors differ. */
        if (backend == VOLVOXAI_BACKEND_CUDA &&
            (!strcmp(node->op, "QSiLU") || !strcmp(node->op, "QGELU")) &&
            !vx_native_gpu_accumulate_immutable_allocation(
                256u, 256u, device_alignment, &total, &device_total))
            return 0;
    }
    *bytes_out = total;
    *device_bytes_out = device_total;
    return 1;
}

static int vx_native_gpu_validation_bootstrap_activation_bound(
        const VxEngineState* validation,
        const VxDeclaredTensor* logical_tensors,
        size_t logical_tensor_count,
        uint64_t device_alignment,
        uint64_t* host_bytes_out,
        uint64_t* device_bytes_out) {
    uint64_t host_total = 0u;
    uint64_t device_total = 0u;
    if (!validation || !logical_tensors || !logical_tensor_count ||
        !device_alignment || !host_bytes_out || !device_bytes_out)
        return 0;
    for (size_t index = 0; index < logical_tensor_count; index++) {
        const T* tensor = vx_native_gpu_validation_tensor_find(
            validation, logical_tensors[index].name);
        uint64_t host_bytes;
        uint64_t device_bytes;
        uint64_t aligned_device;
        if (!tensor || !tensor->data || tensor->numel <= 0 ||
            !tensor->elem_size ||
            (uint64_t)tensor->numel > UINT64_MAX / tensor->elem_size)
            return 0;
        host_bytes = (uint64_t)tensor->numel * tensor->elem_size;
        device_bytes = host_bytes;
        if (tensor->dtype == T_I8 || tensor->dtype == T_U8) {
            if (device_bytes > UINT64_MAX - 3u) return 0;
            device_bytes = (device_bytes + 3u) & ~UINT64_C(3);
        }
        if (!vx_native_gpu_align_resource_bytes(
                device_bytes, device_alignment, &aligned_device) ||
            !vx_native_gpu_accumulate_resource_bytes(
                &host_total, host_bytes, 1u) ||
            !vx_native_gpu_accumulate_resource_bytes(
                &device_total, aligned_device, 1u))
            return 0;
    }
    *host_bytes_out = host_total;
    *device_bytes_out = device_total;
    return 1;
}

static int vx_native_gpu_dispatch_conv2d_bias_bound(
        uint64_t output_channels,
        uint64_t device_alignment,
        uint64_t* total) {
    uint64_t bias_bytes;
    uint64_t aligned_bias_bytes;
    if (!output_channels || !device_alignment || !total ||
        output_channels > UINT64_MAX / sizeof(float))
        return 0;
    bias_bytes = output_channels * sizeof(float);
    return vx_native_gpu_align_resource_bytes(
               bias_bytes, device_alignment, &aligned_bias_bytes) &&
        vx_native_gpu_accumulate_resource_bytes(total, bias_bytes, 1u) &&
        vx_native_gpu_accumulate_resource_bytes(
            total, aligned_bias_bytes, 1u);
}

static int vx_native_gpu_dispatch_qlinear_host_bound(
        uint64_t output_channels, uint64_t* total) {
    if (!output_channels || !total ||
        output_channels > UINT64_MAX / sizeof(float))
        return 0;
    return vx_native_gpu_accumulate_resource_bytes(
        total, output_channels * sizeof(float), 1u);
}

static int vx_native_gpu_dispatch_opengl_packed_clear_bound(
        uint64_t output_bytes, uint64_t* total) {
    uint64_t packed_bytes;
    if (!output_bytes || !total || output_bytes > UINT64_MAX - 3u)
        return 0;
    packed_bytes = (output_bytes + 3u) & ~UINT64_C(3);
    return vx_native_gpu_accumulate_resource_bytes(
        total, packed_bytes, 1u);
}

static int vx_native_gpu_validation_dispatch_temporary_bound(
        const VxEngineState* validation,
        const VxDeclaredTensor* logical_tensors,
        size_t logical_tensor_count,
        VolvoxAIEngineBackend backend,
        uint64_t device_alignment,
        uint64_t* bytes_out) {
    uint64_t total = 0u;
    if (!validation || !logical_tensors || !device_alignment || !bytes_out ||
        validation->node_count < 0)
        return 0;
    if (backend == VOLVOXAI_BACKEND_CUDA) {
        *bytes_out = 0u;
        return 1;
    }
    for (int index = 0; index < validation->node_count; index++) {
        const Node* node = &validation->nodes[index];
        uint64_t dispatch_count = 1u;
        uint64_t aligned_bytes;
        /* Metal retains every bound buffer until its batched graph command
         * completes; OpenGL deletion is likewise deferred while commands
         * reference a buffer.  Qualified parameter blocks are at most 17
         * u32 words. Reserve one independently aligned 256-byte block for
         * each possible per-input/per-output dispatch of a node (Concat and
         * Split are the multi-dispatch cases). Quantize/Dequantize expose
         * their scale and zero-point ports, so max(nin,nout) also covers their
         * three scalar/parameter buffers. */
        if (backend != VOLVOXAI_BACKEND_VULKAN) {
            if (node->nin > 1) dispatch_count = (uint64_t)node->nin;
            if (node->nout > 1 && (uint64_t)node->nout > dispatch_count)
                dispatch_count = (uint64_t)node->nout;
            if (!vx_native_gpu_align_resource_bytes(
                    256u, device_alignment, &aligned_bytes) ||
                !vx_native_gpu_accumulate_resource_bytes(
                    &total, aligned_bytes, dispatch_count))
                return 0;
        }

        /* QLinear/QMatMul/QGemm build d_out F32 multipliers in a host malloc
         * on Vulkan/OpenGL/Metal. Their device copy is already covered by the
         * immutable-component/scratch proof, but the simultaneously live host
         * array is an additional resident transient. */
        if ((!strcmp(node->op, "QLinear") ||
             !strcmp(node->op, "QMatMul") ||
             !strcmp(node->op, "QGemm")) &&
            validation->qlinear_metadata[index].valid &&
            !vx_native_gpu_dispatch_qlinear_host_bound(
                (uint64_t)validation->qlinear_metadata[index].d_out,
                &total))
            return 0;

        /* Of the currently exporter-qualified native-GPU operators, Conv2D
         * has one additional shape-sized dispatch allocation: when bias is
         * absent, OpenGL/Metal create a zero-filled out_c F32 host array and a
         * device buffer before releasing either. Conv1D and MoE have similar
         * private paths but are deliberately outside the qualified set; they
         * must extend this proof before their registry qualification changes. */
        if (backend != VOLVOXAI_BACKEND_VULKAN &&
            !strcmp(node->op, "Conv2D") &&
            !vx_native_gpu_validation_node_input(
                validation, node, "bias")) {
            const char* output_name = node->out[0]
                ? node->out
                : node->nout > 0 ? node->outs[0].name : NULL;
            const VxDeclaredTensor* output = vx_declared_tensor_find(
                logical_tensors, logical_tensor_count, output_name);
            if (!output || !output->rank ||
                output->maximums[output->rank - 1u] <= 0 ||
                !vx_native_gpu_dispatch_conv2d_bias_bound(
                    (uint64_t)output->maximums[output->rank - 1u],
                    device_alignment, &total))
                return 0;
        }

        /* OpenGL initializes a qualified byte Concat output through a host
         * zero array rounded to its packed-u32 allocation before uploading it.
         * Metal clears its shared device buffer in place and Vulkan uses its
         * fixed scratch/device arena, so only OpenGL owns this host transient. */
        if (backend == VOLVOXAI_BACKEND_OPENGL &&
            !strcmp(node->op, "Concat")) {
            const char* output_name = node->out[0]
                ? node->out
                : node->nout > 0 ? node->outs[0].name : NULL;
            const VxDeclaredTensor* output = vx_declared_tensor_find(
                logical_tensors, logical_tensor_count, output_name);
            if (!output ||
                ((output->dtype == VX_DTYPE_I8 ||
                  output->dtype == VX_DTYPE_U8) &&
                 !vx_native_gpu_dispatch_opengl_packed_clear_bound(
                     (uint64_t)output->maximum_byte_size, &total)))
                return 0;
        }
    }
    *bytes_out = total;
    return 1;
}

/* Context creation retains the minimum-shape host tensors until the first
 * exact binding publishes the fixed maximum host arena. OpenGL and Metal also
 * retain every bootstrap device buffer until all candidate domain and stats
 * buffers have been allocated transactionally. Vulkan reuses ranges in one
 * fixed arena, and CUDA explicitly releases bootstrap transients first, so
 * those backends need the larger phase rather than their sum. */
static int vx_native_gpu_activation_resource_peak(
        VolvoxAIEngineBackend backend,
        uint64_t bootstrap_host_bytes,
        uint64_t maximum_host_arena_bytes,
        uint64_t bootstrap_device_bytes,
        uint64_t candidate_device_bytes,
        uint64_t qgroupnorm_stats_bytes,
        uint64_t qlayernorm_stats_bytes,
        uint64_t dispatch_temporary_bytes,
        uint64_t vulkan_reserved_scratch_bytes,
        uint64_t* peak_out) {
    uint64_t peak = 0u;
    uint64_t device_phase;
    uint64_t qnorm_bytes;
    if (!peak_out || !vx_native_gpu_backend(backend) ||
        qgroupnorm_stats_bytes > UINT64_MAX - qlayernorm_stats_bytes)
        return 0;
    qnorm_bytes = qgroupnorm_stats_bytes + qlayernorm_stats_bytes;
    if (!vx_native_gpu_accumulate_resource_bytes(
            &peak, bootstrap_host_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &peak, maximum_host_arena_bytes, 1u))
        return 0;
    if (backend == VOLVOXAI_BACKEND_OPENGL ||
        backend == VOLVOXAI_BACKEND_METAL) {
        if (bootstrap_device_bytes > UINT64_MAX - candidate_device_bytes)
            return 0;
        device_phase = bootstrap_device_bytes + candidate_device_bytes;
        if (!vx_native_gpu_accumulate_resource_bytes(
                &peak, qnorm_bytes, 2u))
            return 0;
    } else {
        device_phase = bootstrap_device_bytes > candidate_device_bytes
            ? bootstrap_device_bytes : candidate_device_bytes;
        if (backend == VOLVOXAI_BACKEND_VULKAN &&
            !vx_native_gpu_accumulate_resource_bytes(
                &peak, vulkan_reserved_scratch_bytes, 1u))
            return 0;
    }
    if (!vx_native_gpu_accumulate_resource_bytes(&peak, device_phase, 1u))
        return 0;
    if (!vx_native_gpu_accumulate_resource_bytes(
            &peak, dispatch_temporary_bytes, 1u))
        return 0;
    *peak_out = peak;
    return 1;
}

static int vx_native_gpu_device_capacity_proven(
        uint64_t immutable_device_bytes,
        uint64_t bootstrap_device_bytes,
        uint64_t candidate_device_bytes,
        uint64_t hard_limit_bytes,
        uint64_t* peak_out) {
    uint64_t transient_phase = bootstrap_device_bytes > candidate_device_bytes
        ? bootstrap_device_bytes : candidate_device_bytes;
    uint64_t peak;
    if (!peak_out || immutable_device_bytes > UINT64_MAX - transient_phase)
        return 0;
    peak = immutable_device_bytes + transient_phase;
    *peak_out = peak;
    return !hard_limit_bytes || peak <= hard_limit_bytes;
}

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
int vx_public_api_test_native_gpu_align_resource_bytes(
        uint64_t bytes, uint64_t alignment, uint64_t* aligned_out) {
    return vx_native_gpu_align_resource_bytes(bytes, alignment, aligned_out);
}

int vx_public_api_test_native_gpu_immutable_components(
        uint64_t component_bytes,
        uint64_t component_count,
        uint64_t device_alignment,
        uint64_t* resident_out,
        uint64_t* device_out) {
    uint64_t resident = 0u;
    uint64_t device = 0u;
    if (!component_count || !resident_out || !device_out) return 0;
    for (uint64_t index = 0; index < component_count; index++)
        if (!vx_native_gpu_accumulate_immutable_allocation(
                component_bytes, component_bytes, device_alignment,
                &resident, &device))
            return 0;
    *resident_out = resident;
    *device_out = device;
    return 1;
}

int vx_public_api_test_native_gpu_maskless_qsdpa_immutable_bound(
        VolvoxAIEngineBackend backend,
        int has_mask,
        uint64_t device_alignment,
        uint64_t* resident_out,
        uint64_t* device_out) {
    uint64_t resident = 0u;
    uint64_t device = 0u;
    if (!resident_out || !device_out ||
        !vx_native_gpu_maskless_qsdpa_immutable_bound(
            backend, has_mask, device_alignment, &resident, &device))
        return 0;
    *resident_out = resident;
    *device_out = device;
    return 1;
}

int vx_public_api_test_native_gpu_zero_bias_backing_bound(
        uint64_t count, uint64_t* bytes_out) {
    uint64_t total = 0u;
    if (!count || !bytes_out ||
        !vx_native_gpu_accumulate_resource_bytes(
            &total, sizeof(VxNativeGpuZeroBiasBackingProof), count))
        return 0;
    *bytes_out = total;
    return 1;
}

int vx_public_api_test_native_gpu_activation_resource_peak(
        VolvoxAIEngineBackend backend,
        uint64_t bootstrap_host_bytes,
        uint64_t maximum_host_arena_bytes,
        uint64_t bootstrap_device_bytes,
        uint64_t candidate_device_bytes,
        uint64_t qgroupnorm_stats_bytes,
        uint64_t qlayernorm_stats_bytes,
        uint64_t dispatch_temporary_bytes,
        uint64_t vulkan_reserved_scratch_bytes,
        uint64_t* peak_out) {
    return vx_native_gpu_activation_resource_peak(
        backend, bootstrap_host_bytes, maximum_host_arena_bytes,
        bootstrap_device_bytes, candidate_device_bytes,
        qgroupnorm_stats_bytes, qlayernorm_stats_bytes,
        dispatch_temporary_bytes, vulkan_reserved_scratch_bytes, peak_out);
}

int vx_public_api_test_native_gpu_conv2d_dispatch_temporary_bound(
        uint64_t output_channels,
        int has_bias,
        uint64_t parameter_blocks,
        uint64_t device_alignment,
        uint64_t* bytes_out) {
    uint64_t total = 0u;
    uint64_t aligned_parameter;
    if (!bytes_out || !parameter_blocks ||
        !vx_native_gpu_align_resource_bytes(
            256u, device_alignment, &aligned_parameter) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &total, aligned_parameter, parameter_blocks) ||
        (!has_bias && !vx_native_gpu_dispatch_conv2d_bias_bound(
            output_channels, device_alignment, &total)))
        return 0;
    *bytes_out = total;
    return 1;
}

int vx_public_api_test_native_gpu_qlinear_dispatch_host_bound(
        uint64_t output_channels, uint64_t* bytes_out) {
    uint64_t total = 0u;
    if (!bytes_out || !vx_native_gpu_dispatch_qlinear_host_bound(
            output_channels, &total))
        return 0;
    *bytes_out = total;
    return 1;
}

int vx_public_api_test_native_gpu_opengl_packed_clear_bound(
        uint64_t output_bytes, uint64_t* bytes_out) {
    uint64_t total = 0u;
    if (!bytes_out ||
        !vx_native_gpu_dispatch_opengl_packed_clear_bound(
            output_bytes, &total))
        return 0;
    *bytes_out = total;
    return 1;
}

int vx_public_api_test_native_gpu_vulkan_scratch_capacity_proven(
        uint64_t qgroupnorm_stats_bytes,
        uint64_t qlayernorm_stats_bytes,
        uint64_t validation_graph_scratch_bytes,
        uint64_t storage_alignment,
        uint64_t maximum_scratch_bytes,
        uint64_t* required_out) {
    return vx_native_gpu_vulkan_scratch_capacity_proven(
        qgroupnorm_stats_bytes, qlayernorm_stats_bytes,
        validation_graph_scratch_bytes, storage_alignment,
        maximum_scratch_bytes, required_out);
}

int vx_public_api_test_native_gpu_device_capacity_proven(
        uint64_t immutable_device_bytes,
        uint64_t bootstrap_device_bytes,
        uint64_t candidate_device_bytes,
        uint64_t hard_limit_bytes,
        uint64_t* peak_out) {
    return vx_native_gpu_device_capacity_proven(
        immutable_device_bytes, bootstrap_device_bytes,
        candidate_device_bytes, hard_limit_bytes, peak_out);
}

int vx_public_api_test_native_gpu_fixed_allocation_bound(
        uint64_t compute_allocation_bytes,
        uint64_t staging_allocation_bytes,
        uint64_t* total_out) {
    return vx_native_gpu_fixed_allocation_bound(
        compute_allocation_bytes, staging_allocation_bytes, total_out);
}

int vx_public_api_test_native_gpu_slot_capacity_proven(
        uint64_t tensor_count,
        uint64_t node_count,
        uint64_t physical_span_count,
        uint64_t slot_limit,
        uint64_t* required_out) {
    return vx_native_gpu_slot_capacity_proven(
        tensor_count, node_count, physical_span_count,
        slot_limit, required_out);
}

int vx_public_api_test_native_gpu_result_publication_peak(
        const uint64_t* output_bytes,
        size_t output_count,
        uint64_t* snapshot_bytes_out,
        uint64_t* publication_peak_out) {
    uint64_t published = 0u;
    uint64_t peak = 0u;
    if ((!output_bytes && output_count) || !snapshot_bytes_out ||
        !publication_peak_out)
        return 0;
    for (size_t index = 0; index < output_count; index++) {
        uint64_t bytes = output_bytes[index];
        uint64_t candidate;
        if (bytes > (UINT64_MAX - published) / 2u) return 0;
        candidate = published + bytes * 2u;
        if (candidate > peak) peak = candidate;
        if (bytes > UINT64_MAX - published) return 0;
        published += bytes;
    }
    *snapshot_bytes_out = published;
    *publication_peak_out = peak;
    return 1;
}

int vx_public_api_test_native_gpu_dynamic_metadata_peak(
        uint64_t signature_bytes,
        uint64_t input_count,
        uint64_t logical_tensor_count,
        uint64_t engine_tensor_count,
        uint64_t physical_span_count,
        uint64_t* peak_out) {
    uint64_t plan_bytes = 0u;
    uint64_t maximum_layout_bytes = 0u;
    uint64_t layout_temporary = 0u;
    uint64_t layout_peak;
    uint64_t request_bytes = 0u;
    uint64_t build_temporary = 0u;
    uint64_t bind_temporary = sizeof(void*);
    uint64_t transaction_temporary;
    uint64_t transaction_peak = 0u;
    if (!signature_bytes || !logical_tensor_count || !peak_out ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_bytes, signature_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_bytes, logical_tensor_count,
            sizeof(int) + 2u * sizeof(size_t) + 9u * sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &maximum_layout_bytes, logical_tensor_count,
            sizeof(int) + 4u * sizeof(size_t)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &layout_temporary, logical_tensor_count,
            sizeof(VolvoxAIEngineMaximumTensor)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &layout_temporary, logical_tensor_count,
            5u * sizeof(int) + 2u * sizeof(size_t)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &layout_temporary, engine_tensor_count, 3u * sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &request_bytes, logical_tensor_count,
            sizeof(VolvoxAIEngineResolvedTensor)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &request_bytes, signature_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &request_bytes, input_count,
            sizeof(VolvoxAIEngineInputBinding)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &build_temporary, engine_tensor_count, sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &bind_temporary, logical_tensor_count,
            sizeof(int) + sizeof(void*)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &bind_temporary, physical_span_count,
            sizeof(VolvoxAIEnginePhysicalSpan)))
        return 0;
    if (maximum_layout_bytes > UINT64_MAX - layout_temporary) return 0;
    layout_peak = maximum_layout_bytes + layout_temporary;
    transaction_temporary = build_temporary > bind_temporary
        ? build_temporary : bind_temporary;
    if (!vx_native_gpu_accumulate_resource_bytes(
            &transaction_peak, maximum_layout_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &transaction_peak, plan_bytes,
            VX_DYNAMIC_SHAPE_PLAN_CACHE_CAPACITY + 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &transaction_peak, request_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &transaction_peak, transaction_temporary, 1u))
        return 0;
    *peak_out = layout_peak > transaction_peak
        ? layout_peak : transaction_peak;
    return 1;
}

int vx_public_api_test_native_gpu_cuda_replay_metadata_peak(
        uint64_t fixed_host_metadata_bytes,
        uint64_t maximum_signature_bytes,
        uint64_t replay_plan_capacity,
        uint64_t* peak_out) {
    return vx_native_gpu_cuda_replay_metadata_peak(
        fixed_host_metadata_bytes, maximum_signature_bytes,
        replay_plan_capacity, peak_out);
}

int vx_public_api_test_native_gpu_resident_capacity_proven(
        uint64_t base_bytes,
        uint64_t result_publication_bytes,
        uint64_t dynamic_metadata_bytes,
        uint64_t hard_limit_bytes,
        uint64_t* required_out) {
    return vx_native_gpu_resident_capacity_proven(
        base_bytes, result_publication_bytes, dynamic_metadata_bytes,
        hard_limit_bytes, required_out);
}
#endif

static const VxDeclaredTensor* vx_native_gpu_node_data_domain(
        const cJSON* node,
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        const VxEngineState* validation,
        VxDeclaredTensor* fixed_storage) {
    static const char* const roles[] = {"input", "data", "x", "a"};
    for (size_t index = 0; index < sizeof(roles) / sizeof(roles[0]); index++) {
        const VxDeclaredTensor* tensor = vx_native_gpu_node_input_domain(
            node, roles[index], tensors, tensor_count, validation,
            fixed_storage);
        if (tensor) return tensor;
    }
    return NULL;
}

static const VxDeclaredTensor* vx_native_gpu_node_output(
        const cJSON* node,
        const VxDeclaredTensor* tensors,
        size_t tensor_count) {
    const cJSON* outputs = node
        ? cJSON_GetObjectItemCaseSensitive(node, "outputs") : NULL;
    const cJSON* port = cJSON_IsObject(outputs)
        ? cJSON_GetObjectItemCaseSensitive(outputs, "out") : NULL;
    const cJSON* reference;
    if (!port && cJSON_IsObject(outputs)) port = outputs->child;
    reference = cJSON_IsObject(port)
        ? cJSON_GetObjectItemCaseSensitive(port, "tensor") : NULL;
    return cJSON_IsString(reference) && reference->valuestring
        ? vx_declared_tensor_find(tensors, tensor_count, reference->valuestring)
        : NULL;
}

static int vx_native_gpu_tensor_maximum_elements(
        const VxDeclaredTensor* tensor,
        uint64_t* elements) {
    uint64_t product = 1u;
    if (!tensor || !elements || !tensor->rank ||
        tensor->rank > VX_MAX_TENSOR_RANK) return 0;
    for (uint32_t axis = 0; axis < tensor->rank; axis++) {
        uint64_t extent;
        if (tensor->maximums[axis] <= 0) return 0;
        extent = (uint64_t)tensor->maximums[axis];
        if (product > UINT64_MAX / extent) return 0;
        product *= extent;
    }
    *elements = product;
    return 1;
}

static int vx_native_gpu_tensor_maximum_product(
        const VxDeclaredTensor* tensor,
        uint32_t begin,
        uint32_t end,
        uint64_t limit,
        uint64_t* product_out) {
    uint64_t product = 1u;
    if (!tensor || begin > end || end > tensor->rank) return 0;
    for (uint32_t axis = begin; axis < end; axis++) {
        uint64_t extent;
        if (tensor->maximums[axis] <= 0) return 0;
        extent = (uint64_t)tensor->maximums[axis];
        if (product > limit / extent) return 0;
        product *= extent;
    }
    if (product_out) *product_out = product;
    return 1;
}

static int vx_native_gpu_integer_parameter(const cJSON* params,
                                     const char* name,
                                     int64_t default_value,
                                     int64_t minimum,
                                     int64_t maximum,
                                     int64_t* value_out) {
    const cJSON* value = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, name) : NULL;
    int64_t converted = default_value;
    if (value) {
        if (cJSON_IsBool(value)) {
            converted = cJSON_IsTrue(value) ? 1 : 0;
        } else {
            if (!cJSON_IsNumber(value) ||
                value->valuedouble < (double)INT_MIN ||
                value->valuedouble > (double)INT_MAX)
                return 0;
            converted = (int64_t)value->valuedouble;
            if ((double)converted != value->valuedouble) return 0;
        }
    }
    if (converted < minimum || converted > maximum) return 0;
    if (value_out) *value_out = converted;
    return 1;
}

static int vx_native_gpu_integer_array(const cJSON* value,
                                 int64_t* values,
                                 size_t capacity,
                                 size_t* count_out) {
    size_t count;
    size_t index = 0;
    if (!cJSON_IsArray(value) || !values || !count_out) return 0;
    count = (size_t)cJSON_GetArraySize(value);
    if (!count || count > capacity) return 0;
    for (const cJSON* item = value->child; item; item = item->next) {
        int64_t converted;
        if (!cJSON_IsNumber(item) ||
            item->valuedouble < (double)INT_MIN ||
            item->valuedouble > (double)INT_MAX)
            return 0;
        converted = (int64_t)item->valuedouble;
        if ((double)converted != item->valuedouble || index >= count) return 0;
        values[index++] = converted;
    }
    if (index != count) return 0;
    *count_out = count;
    return 1;
}

static int vx_native_gpu_pair_parameter(const cJSON* params,
                                        const char* name,
                                        int64_t default_value,
                                        int64_t minimum,
                                        int64_t maximum,
                                        int64_t* first_out,
                                        int64_t* second_out) {
    const cJSON* value = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, name) : NULL;
    int64_t values[2] = {default_value, default_value};
    size_t count = 0u;
    if (!name || !first_out || !second_out || minimum > maximum ||
        default_value < minimum || default_value > maximum)
        return 0;
    if (value) {
        if (cJSON_IsNumber(value)) {
            if (value->valuedouble < (double)minimum ||
                value->valuedouble > (double)maximum)
                return 0;
            values[0] = (int64_t)value->valuedouble;
            if ((double)values[0] != value->valuedouble) return 0;
            values[1] = values[0];
        } else {
            if (!vx_native_gpu_integer_array(
                    value, values, 2u, &count) || count > 2u ||
                values[0] < minimum || values[0] > maximum ||
                (count == 2u &&
                 (values[1] < minimum || values[1] > maximum)))
                return 0;
            values[1] = count == 1u ? values[0] : values[1];
        }
    }
    *first_out = values[0];
    *second_out = values[1];
    return 1;
}

static int vx_native_gpu_tensor_shapes_equal(const VxDeclaredTensor* left,
                                       const VxDeclaredTensor* right) {
    if (!left || !right || left->rank != right->rank) return 0;
    for (uint32_t axis = 0; axis < left->rank; axis++)
        if (!vx_declared_dimensions_provably_equal(
                left, axis, right, axis)) return 0;
    return 1;
}

static int vx_native_gpu_tensor_domains_equal(const VxDeclaredTensor* left,
                                        const VxDeclaredTensor* right) {
    return left && right && left->dtype == right->dtype &&
        vx_native_gpu_tensor_shapes_equal(left, right);
}

static int vx_native_gpu_dimension_domain_contains(
        const VxDeclaredTensor* tensor,
        uint32_t axis,
        int64_t value) {
    VxDimensionProgression progression;
    int64_t fixed;
    if (vx_declared_dimension_fixed_value(tensor, axis, &fixed))
        return fixed == value;
    return vx_declared_dimension_progression(
               tensor, axis, &progression) &&
        value >= progression.first && value <= progression.last &&
        (value - progression.first) % progression.step == 0;
}

static int vx_native_gpu_one_d_launch_proven(
        uint64_t elements,
        VolvoxAIEngineBackend backend,
        const VxNativeGpuDomainLimits* limits);

static int vx_native_gpu_dense_launch_proven(
        uint64_t rows,
        uint64_t input_width,
        uint64_t output_width,
        int quantized,
        VolvoxAIEngineBackend backend,
        const VxNativeGpuDomainLimits* limits) {
    uint64_t output_elements;
    if (!rows || !input_width || !output_width || !limits ||
        rows > UINT32_MAX / output_width)
        return 0;
    output_elements = rows * output_width;
    if (backend == VOLVOXAI_BACKEND_CUDA) {
        if (quantized)
            return vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits);
        /* cuda_matmul_impl() has one fixed 16x16 launch. CUDA additionally
         * rejects grid-y above 65535 even when the physical device reports a
         * wider architectural grid domain. */
        return limits->maximum_block[0] >= 16u &&
            limits->maximum_block[1] >= 16u &&
            limits->maximum_threads_per_block >= 256u &&
            (output_width + 15u) / 16u <= limits->maximum_grid[0] &&
            (rows + 15u) / 16u <= limits->maximum_grid[1] &&
            (rows + 15u) / 16u <= 65535u;
    }
    if (!quantized) {
        uint64_t scalar_grid_x = (output_width + 63u) / 64u;
        /* Every dynamic row domain can select the scalar M=1 route. */
        if (limits->maximum_block[0] < 64u ||
            limits->maximum_threads_per_block < 64u ||
            scalar_grid_x > limits->maximum_grid[0] ||
            rows > limits->maximum_grid[1])
            return 0;
        /* Multi-row matrices use an 8x8 workgroup that emits a 16x16 tile. */
        if (rows > 1u && input_width >= 16u && output_width >= 16u &&
            (limits->maximum_block[0] < 8u ||
             limits->maximum_block[1] < 8u ||
             limits->maximum_threads_per_block < 64u ||
             (output_width + 15u) / 16u > limits->maximum_grid[0] ||
             (rows + 15u) / 16u > limits->maximum_grid[1]))
            return 0;
    } else {
        int tiled = input_width >= 16u && output_width >= 32u &&
            (output_width & 3u) == 0u;
        if (!vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return 0;
        if (tiled &&
            (limits->maximum_block[0] < 8u ||
             limits->maximum_block[1] < 8u ||
             limits->maximum_threads_per_block < 64u ||
             (output_width + 31u) / 32u > limits->maximum_grid[0] ||
             (rows + 7u) / 8u > limits->maximum_grid[1]))
            return 0;
    }
    return 1;
}

static int vx_native_gpu_dense_dynamic_rows_proven(
        const VxDeclaredTensor* input,
        const VxDeclaredTensor* output,
        int quantized,
        VolvoxAIEngineBackend backend,
        const VxNativeGpuDomainLimits* limits) {
    uint64_t rows;
    uint64_t input_width;
    uint64_t output_width;
    if (!input || !output || !limits || !input->rank ||
        input->rank != output->rank ||
        vx_dimension_domain_is_dynamic(input, input->rank - 1u) ||
        vx_dimension_domain_is_dynamic(output, output->rank - 1u))
        return 0;
    for (uint32_t axis = 0; axis + 1u < input->rank; axis++)
        if (!vx_declared_dimensions_provably_equal(
                input, axis, output, axis)) return 0;
    /* seq_range and the ordinary one-shot routes carry rows as signed int. */
    if (!vx_native_gpu_tensor_maximum_product(
            input, 0u, input->rank - 1u, INT_MAX, &rows)) return 0;
    input_width = (uint64_t)input->maximums[input->rank - 1u];
    output_width = (uint64_t)output->maximums[output->rank - 1u];
    return vx_native_gpu_dense_launch_proven(
        rows, input_width, output_width, quantized, backend, limits);
}

static int vx_native_gpu_linear_storage_domain_proven(
        const cJSON* node,
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        const VxEngineState* validation,
        const VxDeclaredTensor* input,
        const VxDeclaredTensor* output) {
    VxDeclaredTensor weight_fixed = {0};
    VxDeclaredTensor bias_fixed = {0};
    const VxDeclaredTensor* weight;
    const VxDeclaredTensor* bias = NULL;
    const cJSON* inputs = node
        ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
    const cJSON* bias_ref = cJSON_IsObject(inputs)
        ? cJSON_GetObjectItemCaseSensitive(inputs, "bias") : NULL;
    const cJSON* params = node
        ? cJSON_GetObjectItemCaseSensitive(node, "params") : NULL;
    const cJSON* layout = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, "weight_layout") : NULL;
    int64_t trans_b = 0;
    int64_t weight_rows;
    int64_t weight_columns;
    int64_t d_in;
    int64_t d_out;
    int output_major = -1;
    uint64_t bias_elements;
    if (!node || !validation || !input || !output || !input->rank ||
        !output->rank ||
        vx_dimension_domain_is_dynamic(input, input->rank - 1u) ||
        vx_dimension_domain_is_dynamic(output, output->rank - 1u))
        return 0;
    d_in = input->maximums[input->rank - 1u];
    d_out = output->maximums[output->rank - 1u];
    weight = vx_native_gpu_node_input_domain(
        node, "weight", tensors, tensor_count, validation, &weight_fixed);
    if (!weight || weight->dtype != VX_DTYPE_F32 || weight->rank != 2u ||
        vx_declared_tensor_find(
            tensors, tensor_count, weight->name) != NULL ||
        !vx_declared_dimension_fixed_value(weight, 0u, &weight_rows) ||
        !vx_declared_dimension_fixed_value(weight, 1u, &weight_columns))
        return 0;
    if (layout) {
        if (!cJSON_IsString(layout) || !layout->valuestring) return 0;
        if (!strcmp(layout->valuestring, "dout_din")) output_major = 1;
        else if (!strcmp(layout->valuestring, "din_dout")) output_major = 0;
        else if (layout->valuestring[0]) return 0;
    }
    if (output_major < 0) {
        if (!vx_native_gpu_integer_parameter(
                params, "transB", 0, 0, 1, &trans_b))
            return 0;
        if (trans_b) output_major = 1;
    }
    if (output_major < 0) {
        int canonical = weight_rows == d_in && weight_columns == d_out;
        int transposed = weight_rows == d_out && weight_columns == d_in;
        output_major = transposed && !canonical ? 1 : canonical ? 0 : -1;
    }
    if (output_major < 0 ||
        (!output_major &&
         (weight_rows != d_in || weight_columns != d_out)) ||
        (output_major &&
         (weight_rows != d_out || weight_columns != d_in)))
        return 0;
    if (bias_ref) {
        if (!cJSON_IsString(bias_ref) || !bias_ref->valuestring) return 0;
        bias = vx_native_gpu_node_input_domain(
            node, "bias", tensors, tensor_count, validation, &bias_fixed);
        if (!bias || bias->dtype != VX_DTYPE_F32 ||
            vx_declared_tensor_find(tensors, tensor_count, bias->name) != NULL ||
            !vx_native_gpu_tensor_maximum_elements(bias, &bias_elements) ||
            bias_elements != (uint64_t)d_out)
            return 0;
        for (uint32_t axis = 0; axis < bias->rank; axis++)
            if (vx_dimension_domain_is_dynamic(bias, axis)) return 0;
    }
    return 1;
}

static int vx_native_gpu_centered_byte_distance(
        VxDataType dtype, int32_t zero_point, uint64_t* distance_out) {
    int64_t zero = zero_point;
    if (!distance_out) return 0;
    if (dtype == VX_DTYPE_I8) {
        if (zero < -128 || zero > 127) return 0;
        *distance_out = (uint64_t)(zero < 0 ? 127 - zero : zero + 128);
        return 1;
    }
    if (dtype == VX_DTYPE_U8) {
        if (zero < 0 || zero > 255) return 0;
        *distance_out = (uint64_t)(zero < 128 ? 255 - zero : zero);
        return 1;
    }
    return 0;
}

static int vx_native_gpu_quantized_accumulator_channel_proven(
        uint64_t terms,
        VxDataType input_dtype,
        int32_t input_zero_point,
        VxDataType weight_dtype,
        int32_t weight_zero_point,
        const int32_t* bias) {
    uint64_t input_distance;
    uint64_t weight_distance;
    uint64_t bias_distance = 0u;
    uint64_t product;
    if (!terms ||
        !vx_native_gpu_centered_byte_distance(
            input_dtype, input_zero_point, &input_distance) ||
        !vx_native_gpu_centered_byte_distance(
            weight_dtype, weight_zero_point, &weight_distance) ||
        terms > UINT64_MAX / input_distance ||
        terms * input_distance > UINT64_MAX / weight_distance)
        return 0;
    if (bias) {
        int64_t value = *bias;
        bias_distance = (uint64_t)(value < 0 ? -value : value);
    }
    product = terms * input_distance * weight_distance;
    return product <= INT32_MAX &&
        bias_distance <= (uint64_t)INT32_MAX - product;
}

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
int vx_public_api_test_native_gpu_quantized_accumulator_channel_proven(
        uint64_t terms,
        VxDataType input_dtype,
        int32_t input_zero_point,
        VxDataType weight_dtype,
        int32_t weight_zero_point,
        const int32_t* bias) {
    return vx_native_gpu_quantized_accumulator_channel_proven(
        terms, input_dtype, input_zero_point, weight_dtype,
        weight_zero_point, bias);
}
#endif

static int vx_native_gpu_qlinear_accumulator_domain_proven(
        const VxEngineState* validation,
        const VxDeclaredTensor* input,
        const VxDeclaredTensor* output) {
    const Node* runtime_node = NULL;
    const T* runtime_input = NULL;
    const QLinearMetadata* metadata = NULL;
    int64_t input_distance;
    if (!validation || !input || !output || !input->rank || !output->rank ||
        !output->name) return 0;
    for (int index = 0; index < validation->node_count; index++) {
        const Node* candidate = &validation->nodes[index];
        if ((!strcmp(candidate->op, "QLinear") ||
             !strcmp(candidate->op, "QMatMul") ||
             !strcmp(candidate->op, "QGemm")) &&
            !strcmp(candidate->out, output->name)) {
            if (runtime_node) return 0;
            runtime_node = candidate;
            metadata = &validation->qlinear_metadata[index];
        }
    }
    if (!runtime_node || !metadata || !metadata->valid ||
        !metadata->weight_zero_points || !metadata->bias ||
        metadata->d_in <= 0 || metadata->d_out <= 0 ||
        input->maximums[input->rank - 1u] != metadata->d_in ||
        output->maximums[output->rank - 1u] != metadata->d_out)
        return 0;
    runtime_input = vx_native_gpu_validation_node_input(
        validation, runtime_node, "input");
    if (!runtime_input) runtime_input = vx_native_gpu_validation_node_input(
        validation, runtime_node, "x");
    if (!runtime_input) runtime_input = vx_native_gpu_validation_node_input(
        validation, runtime_node, "a");
    if (!runtime_input || !runtime_input->quantization.valid ||
        (metadata->input_dtype != VX_DTYPE_I8 &&
         metadata->input_dtype != VX_DTYPE_U8) ||
        (metadata->weight_dtype != VX_DTYPE_I8 &&
         metadata->weight_dtype != VX_DTYPE_U8))
        return 0;
    if ((metadata->input_dtype == VX_DTYPE_I8 &&
         (runtime_input->quantization.zero_point < -128 ||
          runtime_input->quantization.zero_point > 127)) ||
        (metadata->input_dtype == VX_DTYPE_U8 &&
         (runtime_input->quantization.zero_point < 0 ||
          runtime_input->quantization.zero_point > 255)))
        return 0;
    if (metadata->input_dtype == VX_DTYPE_I8) {
        int64_t zero = runtime_input->quantization.zero_point;
        input_distance = zero < 0 ? 127 - zero : zero + 128;
    } else {
        int64_t zero = runtime_input->quantization.zero_point;
        input_distance = zero < 128 ? 255 - zero : zero;
    }
    for (int channel = 0; channel < metadata->d_out; channel++) {
        int64_t zero = metadata->weight_zero_points[channel];
        if ((metadata->weight_dtype == VX_DTYPE_I8 &&
             (zero < -128 || zero > 127)) ||
            (metadata->weight_dtype == VX_DTYPE_U8 &&
             (zero < 0 || zero > 255)))
            return 0;
        int64_t weight_distance = metadata->weight_dtype == VX_DTYPE_I8
            ? (zero < 0 ? 127 - zero : zero + 128)
            : (zero < 128 ? 255 - zero : zero);
        int64_t bias_distance = metadata->bias[channel] < 0
            ? -(int64_t)metadata->bias[channel]
            : (int64_t)metadata->bias[channel];
        uint64_t product = (uint64_t)metadata->d_in *
            (uint64_t)input_distance * (uint64_t)weight_distance;
        if (product > INT32_MAX ||
            (uint64_t)bias_distance > (uint64_t)INT32_MAX - product)
            return 0;
    }
    return 1;
}

static int vx_native_gpu_qconv_accumulator_domain_proven(
        const VxEngineState* validation,
        const VxDeclaredTensor* input,
        const VxDeclaredTensor* output) {
    const Node* runtime_node = NULL;
    const T* runtime_input = NULL;
    const QConv2DMetadata* metadata = NULL;
    uint64_t terms;
    if (!validation || !input || !output || input->rank != 4u ||
        output->rank != 4u || !output->name)
        return 0;
    for (int index = 0; index < validation->node_count; index++) {
        const Node* candidate = &validation->nodes[index];
        if (!strcmp(candidate->op, "QConv2D") &&
            !strcmp(candidate->out, output->name)) {
            if (runtime_node) return 0;
            runtime_node = candidate;
            metadata = &validation->qconv_metadata[index];
        }
    }
    if (!runtime_node || !metadata || !metadata->valid ||
        !metadata->weight_zero_points || !metadata->kernel_height ||
        !metadata->kernel_width || !metadata->input_per_group ||
        !metadata->input_channels || !metadata->output_channels ||
        input->maximums[3] != (int64_t)metadata->input_channels ||
        output->maximums[3] != (int64_t)metadata->output_channels ||
        (metadata->input_dtype != VX_DTYPE_I8 &&
         metadata->input_dtype != VX_DTYPE_U8) ||
        (metadata->weight_dtype != VX_DTYPE_I8 &&
         metadata->weight_dtype != VX_DTYPE_U8))
        return 0;
    runtime_input = vx_native_gpu_validation_node_input(
        validation, runtime_node, "input");
    if (!runtime_input) runtime_input = vx_native_gpu_validation_node_input(
        validation, runtime_node, "x");
    if (!runtime_input || !runtime_input->quantization.valid ||
        runtime_input->dtype != metadata->input_dtype)
        return 0;
    terms = metadata->kernel_height;
    if (terms > UINT64_MAX / metadata->kernel_width) return 0;
    terms *= metadata->kernel_width;
    if (terms > UINT64_MAX / metadata->input_per_group) return 0;
    terms *= metadata->input_per_group;
    for (uint32_t channel = 0; channel < metadata->output_channels;
         channel++) {
        const int32_t* bias = metadata->bias
            ? &metadata->bias[channel] : NULL;
        if (!vx_native_gpu_quantized_accumulator_channel_proven(
                terms, metadata->input_dtype,
                runtime_input->quantization.zero_point,
                metadata->weight_dtype,
                metadata->weight_zero_points[channel], bias))
            return 0;
    }
    return 1;
}

static int vx_native_gpu_expand_domain_proven(const VxDeclaredTensor* input,
                                        const VxDeclaredTensor* output) {
    uint32_t leading;
    if (!input || !output || !input->rank || input->rank > output->rank)
        return 0;
    leading = output->rank - input->rank;
    for (uint32_t axis = 0; axis < input->rank; axis++) {
        int64_t fixed;
        if (vx_declared_dimension_fixed_value(input, axis, &fixed) &&
            fixed == 1) continue;
        if (!vx_declared_dimensions_provably_equal(
                input, axis, output, leading + axis)) return 0;
    }
    return 1;
}

static int vx_native_gpu_one_d_launch_proven(
        uint64_t elements,
        VolvoxAIEngineBackend backend,
        const VxNativeGpuDomainLimits* limits) {
    const uint64_t lanes = backend == VOLVOXAI_BACKEND_CUDA ? 256u : 64u;
    uint64_t grid;
    if (!elements || elements > UINT32_MAX || !limits ||
        limits->maximum_block[0] < lanes ||
        limits->maximum_threads_per_block < lanes)
        return 0;
    /* CUDA's common 1-D launcher uses 256 threads; Vulkan/OpenGL/Metal shader
     * routes use 64 scalar or packed-u32 lanes. Keep the proof identical to
     * the selected backend instead of relying on customary device minima. */
    grid = elements / lanes + (elements % lanes != 0u);
    return grid <= limits->maximum_grid[0];
}

static int vx_native_gpu_cuda_conv2d_tiled_range_proven(
        uint64_t minimum_rows,
        uint64_t maximum_rows,
        uint64_t output_channels,
        uint64_t tile_rows,
        uint64_t tile_columns,
        uint64_t block_x,
        uint64_t block_y,
        const VxNativeGpuDomainLimits* limits) {
    const uint64_t tiled_grid_y_limit = 65535u;
    uint64_t last_tiled_row;
    uint64_t grid_x;
    uint64_t grid_y;
    if (minimum_rows > maximum_rows) return 1;
    if (!minimum_rows || !output_channels || !tile_rows || !tile_columns ||
        !block_x || !block_y || !limits ||
        tile_rows > UINT64_MAX / tiled_grid_y_limit)
        return 0;
    last_tiled_row = tile_rows * tiled_grid_y_limit;
    /* cuda_graph_conv2d_impl() switches to its 256-thread 1-D kernel when a
     * selected tiled tactic would exceed this explicit grid-y boundary. */
    if (minimum_rows > last_tiled_row) return 1;
    if (maximum_rows > last_tiled_row) maximum_rows = last_tiled_row;
    grid_x = output_channels / tile_columns +
        (output_channels % tile_columns != 0u);
    grid_y = maximum_rows / tile_rows +
        (maximum_rows % tile_rows != 0u);
    return block_x <= limits->maximum_block[0] &&
        block_y <= limits->maximum_block[1] &&
        limits->maximum_block[2] >= 1u &&
        block_x <= UINT64_MAX / block_y &&
        block_x * block_y <= limits->maximum_threads_per_block &&
        grid_x <= limits->maximum_grid[0] &&
        grid_y <= limits->maximum_grid[1] &&
        limits->maximum_grid[2] >= 1u;
}

enum {
    VX_NATIVE_GPU_CUDA_CONV_TILED_NEVER = 0,
    VX_NATIVE_GPU_CUDA_CONV_TILED_POSSIBLE = 1,
    VX_NATIVE_GPU_CUDA_CONV_TILED_FIXED = 2,
};

/* CUDA Conv2D has no batch/output-channel grid-z axis. Generic Conv2D and
 * every depthwise tactic use cuda_launch_1d(output elements or fewer). The
 * 1x1 path chooses one of three row/channel tiles and falls back to the same
 * 1-D kernel above its explicit 65535 grid-y boundary. Prove both families so
 * admission follows the launches which can actually occur for the domain. */
static int vx_native_gpu_cuda_conv2d_launch_proven(
        uint64_t minimum_rows,
        uint64_t maximum_rows,
        uint64_t input_channels,
        uint64_t output_channels,
        int tiled_1x1_route,
        const VxNativeGpuDomainLimits* limits) {
    uint64_t output_elements;
    int one_d_required;
    if (!minimum_rows || minimum_rows > maximum_rows || !input_channels ||
        !output_channels || output_channels > UINT32_MAX ||
        tiled_1x1_route < VX_NATIVE_GPU_CUDA_CONV_TILED_NEVER ||
        tiled_1x1_route > VX_NATIVE_GPU_CUDA_CONV_TILED_FIXED || !limits ||
        maximum_rows > UINT32_MAX / output_channels)
        return 0;
    output_elements = maximum_rows * output_channels;
    /* route=0 never tiles, route=1 can bind either family, and route=2 has
     * fixed 1x1 geometry. A fixed tiled route reaches 1-D only after its
     * selected tactic crosses cuda_graph_conv2d_impl()'s grid-y cutoff. */
    one_d_required =
        tiled_1x1_route != VX_NATIVE_GPU_CUDA_CONV_TILED_FIXED;
    if (tiled_1x1_route == VX_NATIVE_GPU_CUDA_CONV_TILED_FIXED) {
        uint64_t tiled_row_limit = input_channels >= 16u &&
            output_channels >= 48u ? 32u * 65535u : 16u * 65535u;
        one_d_required = maximum_rows > tiled_row_limit;
    }
    if (one_d_required &&
        (!vx_native_gpu_one_d_launch_proven(
             output_elements, VOLVOXAI_BACKEND_CUDA, limits) ||
         limits->maximum_grid[1] < 1u || limits->maximum_grid[2] < 1u ||
         limits->maximum_block[1] < 1u || limits->maximum_block[2] < 1u))
        return 0;
    if (!tiled_1x1_route) return 1;

    if (input_channels >= 16u && output_channels >= 48u) {
        uint64_t first;
        uint64_t last;
        /* rows < 8: BM16/BN16/BK1, block 16x16. */
        first = minimum_rows;
        last = maximum_rows < 7u ? maximum_rows : 7u;
        if (!vx_native_gpu_cuda_conv2d_tiled_range_proven(
                first, last, output_channels,
                16u, 16u, 16u, 16u, limits))
            return 0;
        /* 8 <= rows < 17: BM16/BN64/BK16, block 16x8. */
        first = minimum_rows > 8u ? minimum_rows : 8u;
        last = maximum_rows < 16u ? maximum_rows : 16u;
        if (!vx_native_gpu_cuda_conv2d_tiled_range_proven(
                first, last, output_channels,
                16u, 64u, 16u, 8u, limits))
            return 0;
        /* rows >= 17: BM32/BN32/BK16, block 8x16. */
        first = minimum_rows > 17u ? minimum_rows : 17u;
        if (!vx_native_gpu_cuda_conv2d_tiled_range_proven(
                first, maximum_rows, output_channels,
                32u, 32u, 8u, 16u, limits))
            return 0;
        return 1;
    }
    return vx_native_gpu_cuda_conv2d_tiled_range_proven(
        minimum_rows, maximum_rows, output_channels,
        16u, 16u, 16u, 16u, limits);
}

/* Vulkan and OpenGL select an out16-family Conv2D tactic whenever groups=1
 * and the fixed output-channel count is divisible by 16. All such tactics
 * launch one Z workgroup per batch/output-channel block. Metal retains the
 * scalar one-workgroup-per-channel geometry. The public V/O entry points
 * still encode n*out_c in u32 before tactic selection, so prove that ABI
 * bound independently of the smaller selected dispatch. */
static int vx_native_gpu_vom_conv2d_launch_proven(
        uint64_t batch,
        uint64_t output_height,
        uint64_t output_width,
        uint64_t output_channels,
        int groups_one,
        VolvoxAIEngineBackend backend,
        const VxNativeGpuDomainLimits* limits) {
    uint64_t logical_lanes;
    uint64_t dispatch_channels;
    uint64_t grid_x;
    uint64_t grid_y;
    uint64_t grid_z;
    if (!batch || !output_height || !output_width || !output_channels ||
        (groups_one != 0 && groups_one != 1) || !limits ||
        (backend != VOLVOXAI_BACKEND_VULKAN &&
         backend != VOLVOXAI_BACKEND_OPENGL &&
         backend != VOLVOXAI_BACKEND_METAL) ||
        batch > UINT32_MAX / output_channels)
        return 0;
    logical_lanes = batch * output_channels;
    dispatch_channels =
        (backend == VOLVOXAI_BACKEND_VULKAN ||
         backend == VOLVOXAI_BACKEND_OPENGL) &&
        groups_one && (output_channels & 15u) == 0u
        ? output_channels / 16u : output_channels;
    grid_x = output_width / 8u + (output_width % 8u != 0u);
    grid_y = output_height / 8u + (output_height % 8u != 0u);
    grid_z = batch * dispatch_channels;
    return logical_lanes <= UINT32_MAX &&
        limits->maximum_block[0] >= 8u &&
        limits->maximum_block[1] >= 8u &&
        limits->maximum_block[2] >= 1u &&
        limits->maximum_threads_per_block >= 64u &&
        grid_x <= limits->maximum_grid[0] &&
        grid_y <= limits->maximum_grid[1] &&
        grid_z <= limits->maximum_grid[2];
}

/* MaxPool2D and ResizeNearest2D share one native GPU contract: NHWC F32 whose
 * batch and channel extents carry through unchanged while only the spatial
 * extents move. Vulkan and OpenGL tile 8x8 over output width/height with one Z
 * workgroup per batch/channel pair, matching Conv2D's geometry; CUDA launches
 * one flat thread per output value. */
static int vx_native_gpu_nhwc_spatial_domain_proven(
        const VxDeclaredTensor* input,
        const VxDeclaredTensor* output,
        VolvoxAIEngineBackend backend,
        const VxNativeGpuDomainLimits* limits) {
    uint64_t output_elements;
    uint64_t channels;
    uint64_t grid_x;
    uint64_t grid_y;
    if (!input || !output || !limits ||
        input->rank != 4u || output->rank != 4u ||
        input->dtype != VX_DTYPE_F32 || output->dtype != VX_DTYPE_F32 ||
        !vx_declared_dimensions_provably_equal(input, 0u, output, 0u) ||
        !vx_declared_dimensions_provably_equal(input, 3u, output, 3u) ||
        vx_dimension_domain_is_dynamic(output, 3u) ||
        !vx_native_gpu_tensor_maximum_elements(output, &output_elements))
        return 0;
    if (backend == VOLVOXAI_BACKEND_CUDA)
        return vx_native_gpu_one_d_launch_proven(
            output_elements, backend, limits);
    channels = (uint64_t)output->maximums[3];
    if (!channels || (uint64_t)output->maximums[0] > UINT64_MAX / channels)
        return 0;
    grid_x = (uint64_t)output->maximums[2] / 8u +
        ((uint64_t)output->maximums[2] % 8u != 0u);
    grid_y = (uint64_t)output->maximums[1] / 8u +
        ((uint64_t)output->maximums[1] % 8u != 0u);
    return limits->maximum_block[0] >= 8u && limits->maximum_block[1] >= 8u &&
        limits->maximum_threads_per_block >= 64u &&
        grid_x <= limits->maximum_grid[0] &&
        grid_y <= limits->maximum_grid[1] &&
        (uint64_t)output->maximums[0] * channels <= limits->maximum_grid[2];
}

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
int vx_public_api_test_native_gpu_vom_conv2d_launch_proven(
        uint64_t batch,
        uint64_t output_height,
        uint64_t output_width,
        uint64_t output_channels,
        int groups_one,
        VolvoxAIEngineBackend backend,
        uint64_t maximum_grid_x,
        uint64_t maximum_grid_y,
        uint64_t maximum_grid_z,
        uint64_t maximum_block_x,
        uint64_t maximum_block_y,
        uint64_t maximum_block_z,
        uint64_t maximum_threads_per_block) {
    VxNativeGpuDomainLimits limits = {0};
    if (maximum_grid_x > UINT32_MAX || maximum_grid_y > UINT32_MAX ||
        maximum_grid_z > UINT32_MAX || maximum_block_x > UINT32_MAX ||
        maximum_block_y > UINT32_MAX || maximum_block_z > UINT32_MAX ||
        maximum_threads_per_block > UINT32_MAX)
        return 0;
    limits.maximum_grid[0] = (uint32_t)maximum_grid_x;
    limits.maximum_grid[1] = (uint32_t)maximum_grid_y;
    limits.maximum_grid[2] = (uint32_t)maximum_grid_z;
    limits.maximum_block[0] = (uint32_t)maximum_block_x;
    limits.maximum_block[1] = (uint32_t)maximum_block_y;
    limits.maximum_block[2] = (uint32_t)maximum_block_z;
    limits.maximum_threads_per_block =
        (uint32_t)maximum_threads_per_block;
    return vx_native_gpu_vom_conv2d_launch_proven(
        batch, output_height, output_width, output_channels, groups_one,
        backend, &limits);
}

int vx_public_api_test_native_gpu_cuda_conv2d_launch_proven(
        uint64_t minimum_rows,
        uint64_t maximum_rows,
        uint64_t input_channels,
        uint64_t output_channels,
        int tiled_1x1_route,
        uint64_t maximum_grid_x,
        uint64_t maximum_grid_y,
        uint64_t maximum_grid_z,
        uint64_t maximum_block_x,
        uint64_t maximum_block_y,
        uint64_t maximum_block_z,
        uint64_t maximum_threads_per_block) {
    VxNativeGpuDomainLimits limits = {0};
    if (maximum_grid_x > UINT32_MAX || maximum_grid_y > UINT32_MAX ||
        maximum_grid_z > UINT32_MAX || maximum_block_x > UINT32_MAX ||
        maximum_block_y > UINT32_MAX || maximum_block_z > UINT32_MAX ||
        maximum_threads_per_block > UINT32_MAX)
        return 0;
    limits.maximum_grid[0] = (uint32_t)maximum_grid_x;
    limits.maximum_grid[1] = (uint32_t)maximum_grid_y;
    limits.maximum_grid[2] = (uint32_t)maximum_grid_z;
    limits.maximum_block[0] = (uint32_t)maximum_block_x;
    limits.maximum_block[1] = (uint32_t)maximum_block_y;
    limits.maximum_block[2] = (uint32_t)maximum_block_z;
    limits.maximum_threads_per_block =
        (uint32_t)maximum_threads_per_block;
    return vx_native_gpu_cuda_conv2d_launch_proven(
        minimum_rows, maximum_rows, input_channels, output_channels,
        tiled_1x1_route, &limits);
}

int vx_public_api_test_native_gpu_dense_launch_proven(
        uint64_t rows,
        uint64_t input_width,
        uint64_t output_width,
        int quantized,
        VolvoxAIEngineBackend backend,
        uint64_t maximum_grid_x,
        uint64_t maximum_grid_y,
        uint64_t maximum_block_x,
        uint64_t maximum_block_y,
        uint64_t maximum_threads_per_block) {
    VxNativeGpuDomainLimits limits = {0};
    limits.maximum_grid[0] = maximum_grid_x;
    limits.maximum_grid[1] = maximum_grid_y;
    limits.maximum_grid[2] = 1u;
    limits.maximum_block[0] = maximum_block_x;
    limits.maximum_block[1] = maximum_block_y;
    limits.maximum_block[2] = 1u;
    limits.maximum_threads_per_block = maximum_threads_per_block;
    return vx_native_gpu_dense_launch_proven(
        rows, input_width, output_width, quantized, backend, &limits);
}
#endif

static int vx_native_gpu_broadcast_axis_proven(
        const VxDeclaredTensor* operand,
        uint32_t operand_axis,
        const VxDeclaredTensor* output,
        uint32_t output_axis) {
    int64_t fixed;
    return (vx_declared_dimension_fixed_value(
                operand, operand_axis, &fixed) && fixed == 1) ||
        vx_declared_dimensions_provably_equal(
            operand, operand_axis, output, output_axis);
}

static int vx_native_gpu_binary_broadcast_domain_proven(
        const VxDeclaredTensor* a,
        const VxDeclaredTensor* b,
        const VxDeclaredTensor* output,
        int exact) {
    uint32_t a_leading;
    uint32_t b_leading;
    if (!a || !b || !output || !a->rank || !b->rank ||
        a->rank > output->rank || b->rank > output->rank)
        return 0;
    a_leading = output->rank - a->rank;
    b_leading = output->rank - b->rank;
    for (uint32_t axis = 0; axis < output->rank; axis++) {
        int a_present = axis >= a_leading;
        int b_present = axis >= b_leading;
        int a_equal = !a_present || vx_declared_dimensions_provably_equal(
            a, axis - a_leading, output, axis);
        int b_equal = !b_present || vx_declared_dimensions_provably_equal(
            b, axis - b_leading, output, axis);
        if (exact) {
            if (!a_present || !b_present || !a_equal || !b_equal) return 0;
            continue;
        }
        if ((a_present && !vx_native_gpu_broadcast_axis_proven(
                 a, axis - a_leading, output, axis)) ||
            (b_present && !vx_native_gpu_broadcast_axis_proven(
                 b, axis - b_leading, output, axis)) ||
            (!a_equal && !b_equal))
            return 0;
    }
    return 1;
}

static int vx_native_gpu_batch_matmul_domain_proven(
        const VxDeclaredTensor* a,
        const VxDeclaredTensor* b,
        const VxDeclaredTensor* output) {
    uint32_t output_batch_rank;
    uint32_t a_batch_rank;
    uint32_t b_batch_rank;
    uint32_t a_leading;
    uint32_t b_leading;
    if (!a || !b || !output || a->rank < 2u || a->rank > 8u ||
        b->rank < 2u || b->rank > 8u || output->rank < 2u ||
        output->rank > 8u ||
        output->rank != (a->rank > b->rank ? a->rank : b->rank) ||
        !vx_declared_dimensions_provably_equal(
            a, a->rank - 1u, b, b->rank - 2u) ||
        !vx_declared_dimensions_provably_equal(
            a, a->rank - 2u, output, output->rank - 2u) ||
        !vx_declared_dimensions_provably_equal(
            b, b->rank - 1u, output, output->rank - 1u))
        return 0;
    output_batch_rank = output->rank - 2u;
    a_batch_rank = a->rank - 2u;
    b_batch_rank = b->rank - 2u;
    a_leading = output_batch_rank - a_batch_rank;
    b_leading = output_batch_rank - b_batch_rank;
    for (uint32_t axis = 0; axis < output_batch_rank; axis++) {
        int a_present = axis >= a_leading;
        int b_present = axis >= b_leading;
        int a_equal = !a_present || vx_declared_dimensions_provably_equal(
            a, axis - a_leading, output, axis);
        int b_equal = !b_present || vx_declared_dimensions_provably_equal(
            b, axis - b_leading, output, axis);
        if ((a_present && !vx_native_gpu_broadcast_axis_proven(
                 a, axis - a_leading, output, axis)) ||
            (b_present && !vx_native_gpu_broadcast_axis_proven(
                 b, axis - b_leading, output, axis)) ||
            (!a_equal && !b_equal))
            return 0;
    }
    return 1;
}

static size_t vx_native_gpu_dynamic_symbol_count(
        const VxDeclaredTensor* tensor,
        const char* symbol) {
    size_t count = 0u;
    if (!tensor || !symbol) return 0u;
    for (uint32_t axis = 0; axis < tensor->rank; axis++)
        if (vx_dimension_domain_is_dynamic(tensor, axis) &&
            tensor->symbols[axis] &&
            !strcmp(tensor->symbols[axis], symbol))
            count++;
    return count;
}

static int vx_native_gpu_reshape_element_domain_proven(
        const VxDeclaredTensor* input,
        const VxDeclaredTensor* output) {
    uint64_t input_fixed = 1u;
    uint64_t output_fixed = 1u;
    if (!input || !output || !input->rank || !output->rank) return 0;
    for (uint32_t axis = 0; axis < input->rank; axis++) {
        if (vx_dimension_domain_is_dynamic(input, axis)) {
            if (!input->symbols[axis] ||
                vx_native_gpu_dynamic_symbol_count(
                    input, input->symbols[axis]) !=
                vx_native_gpu_dynamic_symbol_count(
                    output, input->symbols[axis]))
                return 0;
        } else {
            uint64_t extent = (uint64_t)input->maximums[axis];
            if (!extent || input_fixed > UINT64_MAX / extent) return 0;
            input_fixed *= extent;
        }
    }
    for (uint32_t axis = 0; axis < output->rank; axis++) {
        if (vx_dimension_domain_is_dynamic(output, axis)) {
            if (!output->symbols[axis] ||
                vx_native_gpu_dynamic_symbol_count(
                    output, output->symbols[axis]) !=
                vx_native_gpu_dynamic_symbol_count(
                    input, output->symbols[axis]))
                return 0;
        } else {
            uint64_t extent = (uint64_t)output->maximums[axis];
            if (!extent || output_fixed > UINT64_MAX / extent) return 0;
            output_fixed *= extent;
        }
    }
    return input_fixed == output_fixed;
}

static int vx_native_gpu_transpose_domain_proven(
        const VxDeclaredTensor* input,
        const VxDeclaredTensor* output,
        const cJSON* params) {
    int64_t permutation[8] = {0};
    const cJSON* perm = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, "perm") : NULL;
    size_t count = 0u;
    unsigned int seen = 0u;
    if (!input || !output || !input->rank || input->rank > 8u ||
        input->rank != output->rank) return 0;
    if (perm) {
        if (!vx_native_gpu_integer_array(perm, permutation, 8u, &count) ||
            count != input->rank) return 0;
    } else {
        count = input->rank;
        for (size_t axis = 0; axis < count; axis++)
            permutation[axis] = (int64_t)count - 1 - (int64_t)axis;
    }
    for (uint32_t output_axis = 0; output_axis < output->rank; output_axis++) {
        int64_t input_axis = permutation[output_axis];
        if (input_axis < 0) input_axis += input->rank;
        if (input_axis < 0 || input_axis >= input->rank ||
            (seen & (1u << (unsigned int)input_axis)) ||
            !vx_declared_dimensions_provably_equal(
                input, (uint32_t)input_axis, output, output_axis))
            return 0;
        seen |= 1u << (unsigned int)input_axis;
    }
    return 1;
}

static int vx_native_gpu_slice_extent_at(int64_t input_extent,
                                   int64_t raw_start,
                                   int64_t raw_end,
                                   int64_t step,
                                   int64_t* output_extent) {
    int64_t start;
    int64_t end;
    if (input_extent <= 0 || raw_start < 0 || step <= 0 ||
        !output_extent) return 0;
    start = raw_start;
    end = raw_end < 0 ? raw_end + input_extent : raw_end;
    if (start > input_extent) start = input_extent;
    if (end < 0) end = 0;
    if (end > input_extent) end = input_extent;
    if (end <= start) return 0;
    *output_extent = (end - start - 1) / step + 1;
    return *output_extent > 0;
}

static int vx_native_gpu_slice_axis_domain_proven(
        const VxDeclaredTensor* input,
        const VxDeclaredTensor* output,
        uint32_t axis,
        int selected,
        int64_t raw_start,
        int64_t raw_end,
        int64_t step) {
    VxDimensionProgression input_domain;
    int64_t output_fixed;
    int64_t first_output;
    int64_t last_output;
    uint64_t last_index;
    if (!input || !output || axis >= input->rank || axis >= output->rank)
        return 0;
    if (!selected)
        return vx_declared_dimensions_provably_equal(
            input, axis, output, axis);
    if (!vx_declared_dimension_progression(
            input, axis, &input_domain)) {
        int64_t fixed;
        if (!vx_declared_dimension_fixed_value(input, axis, &fixed) ||
            fixed <= 0) return 0;
        input_domain.first = fixed;
        input_domain.last = fixed;
        input_domain.step = 1;
        input_domain.count = 1u;
    }

    /* A symbolic output can only repeat the input equality class here. The
     * static Slice contract has no graph-owned affine relation with which to
     * bind a distinct output symbol. The sole varying case representable by
     * that equality class is an unclipped identity slice. */
    if (vx_declared_dimensions_provably_equal(
            input, axis, output, axis) &&
        vx_dimension_domain_is_dynamic(input, axis)) {
        return raw_start == 0 && step == 1 && raw_end >= input_domain.last;
    }

    if (!vx_declared_dimension_fixed_value(output, axis, &output_fixed) ||
        output_fixed <= 0 ||
        !vx_native_gpu_slice_extent_at(
            input_domain.first, raw_start, raw_end, step, &first_output) ||
        !vx_native_gpu_slice_extent_at(
            input_domain.last, raw_start, raw_end, step, &last_output) ||
        first_output != output_fixed || last_output != output_fixed)
        return 0;

    /* The native GPU wrappers address the last selected element directly. Prove
     * their exact runtime guard for the smallest legal input extent; the
     * output is fixed above, so every larger extent is then safe as well. */
    if ((uint64_t)(output_fixed - 1) >
            (UINT64_MAX - (uint64_t)raw_start) / (uint64_t)step)
        return 0;
    last_index = (uint64_t)raw_start +
        (uint64_t)(output_fixed - 1) * (uint64_t)step;
    return last_index < (uint64_t)input_domain.first;
}

static int vx_native_gpu_concat_launch_domain_proven(
        const cJSON* node,
        const VxDeclaredTensor* output) {
    const cJSON* inputs = node
        ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
    const cJSON* params = node
        ? cJSON_GetObjectItemCaseSensitive(node, "params") : NULL;
    int input_count = cJSON_IsObject(inputs)
        ? cJSON_GetArraySize(inputs) : 0;
    size_t axis;
    uint64_t inner;
    if (!output || !output->rank || input_count < 2 ||
        input_count > MAXIN || input_count > 256 ||
        !vx_concat_axis(params, output->rank, &axis) ||
        output->maximums[axis] <= 0 ||
        output->maximums[axis] > INT_MAX ||
        !vx_native_gpu_tensor_maximum_product(
            output, (uint32_t)axis + 1u, output->rank,
            INT_MAX, &inner))
        return 0;
    return inner > 0u;
}

enum VxNativeGpuValueProof {
    VX_NATIVE_GPU_VALUE_REJECTED = 0,
    VX_NATIVE_GPU_VALUE_PUBLIC_PREFLIGHT = 1,
    VX_NATIVE_GPU_VALUE_STATICALLY_PROVEN = 2,
};

static const char* vx_native_gpu_validation_node_input_name(
        const Node* node, const char* role) {
    if (!node || !role) return NULL;
    for (int index = 0; index < node->nin; index++)
        if (!strcmp(node->ins[index].key, role))
            return node->ins[index].name;
    return NULL;
}

static const char* vx_native_gpu_validation_node_data_name(
        const Node* node) {
    static const char* const roles[] = {"input", "data", "x", "a"};
    if (!node) return NULL;
    for (size_t index = 0; index < sizeof(roles) / sizeof(roles[0]); index++) {
        const char* name = vx_native_gpu_validation_node_input_name(
            node, roles[index]);
        if (name) return name;
    }
    return node->nin > 0 ? node->ins[0].name : NULL;
}

static const Node* vx_native_gpu_validation_node_producer(
        const VxEngineState* validation, const char* tensor_name) {
    if (!validation || !tensor_name) return NULL;
    for (int index = 0; index < validation->node_count; index++) {
        const Node* node = &validation->nodes[index];
        if (node->out[0] && !strcmp(node->out, tensor_name)) return node;
        for (int output = 0; output < node->nout; output++)
            if (!strcmp(node->outs[output].name, tensor_name)) return node;
    }
    return NULL;
}

static int vx_native_gpu_validation_node_index(
        const VxEngineState* validation, const Node* node) {
    if (!validation || !node || node < validation->nodes ||
        node >= validation->nodes + validation->node_count)
        return -1;
    return (int)(node - validation->nodes);
}

static int vx_native_gpu_value_preserving_node(const Node* producer) {
    return producer &&
        (!strcmp(producer->op, "Reshape") ||
         !strcmp(producer->op, "Squeeze") ||
         !strcmp(producer->op, "Unsqueeze") ||
         !strcmp(producer->op, "Identity") ||
         !strcmp(producer->op, "Expand"));
}

/* Prove the origin of values that a GPU wrapper otherwise examines through a
 * host pointer. Public values are deferred to the execution-wide preflight;
 * immutable values are checked here while their host storage is authoritative.
 * A deliberately small producer proof prevents device-authored data from
 * being mistaken for current host data. */
static int vx_native_gpu_i32_value_source_proof(
        const VxEngineState* validation,
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        const char* tensor_name,
        int64_t minimum,
        int64_t maximum_exclusive,
        int before_node,
        int depth) {
    const T* tensor;
    const Node* producer;
    if (!validation || !tensor_name || minimum >= maximum_exclusive ||
        before_node < 0 || before_node > validation->node_count ||
        depth < 0 || depth > validation->node_count)
        return VX_NATIVE_GPU_VALUE_REJECTED;
    tensor = vx_native_gpu_validation_tensor_find(validation, tensor_name);
    if (!tensor || tensor->dtype != T_I32 || !tensor->data ||
        tensor->numel <= 0)
        return VX_NATIVE_GPU_VALUE_REJECTED;
    if (tensor->is_graph_input)
        return VX_NATIVE_GPU_VALUE_PUBLIC_PREFLIGHT;
    producer = vx_native_gpu_validation_node_producer(validation, tensor_name);
    if (!producer) {
        for (long index = 0; index < tensor->numel; index++) {
            int32_t value;
            memcpy(&value,
                   (const unsigned char*)tensor->data +
                       (size_t)index * sizeof(value),
                   sizeof(value));
            if ((int64_t)value < minimum ||
                (int64_t)value >= maximum_exclusive)
                return VX_NATIVE_GPU_VALUE_REJECTED;
        }
        return VX_NATIVE_GPU_VALUE_STATICALLY_PROVEN;
    }
    {
        int producer_index = vx_native_gpu_validation_node_index(
            validation, producer);
        if (producer_index < 0 || producer_index >= before_node)
            return VX_NATIVE_GPU_VALUE_REJECTED;
        before_node = producer_index;
    }
    if (!strcmp(producer->op, "Clip")) {
        int64_t clip_min;
        int64_t clip_max;
        if (!vx_native_gpu_integer_parameter(
                producer->params, "min", INT32_MIN,
                INT32_MIN, INT32_MAX, &clip_min) ||
            !vx_native_gpu_integer_parameter(
                producer->params, "max", INT32_MAX,
                INT32_MIN, INT32_MAX, &clip_max) ||
            clip_min > clip_max || clip_min < minimum ||
            clip_max >= maximum_exclusive)
            return VX_NATIVE_GPU_VALUE_REJECTED;
        return VX_NATIVE_GPU_VALUE_STATICALLY_PROVEN;
    }
    if (!strcmp(producer->op, "ArgMax") ||
        !strcmp(producer->op, "QArgMax")) {
        const char* source_name = vx_native_gpu_validation_node_data_name(
            producer);
        const VxDeclaredTensor* source = vx_declared_tensor_find(
            tensors, tensor_count, source_name);
        int64_t axis;
        int qargmax = !strcmp(producer->op, "QArgMax");
        if (!source || !source->rank || minimum > 0 ||
            (qargmax &&
             (!producer->params ||
              cJSON_GetArraySize(producer->params) != 1)) ||
            !vx_native_gpu_integer_parameter(
                producer->params, "axis", qargmax ? -1 : 0,
                qargmax ? -1 : -(int64_t)source->rank,
                qargmax ? -1 : (int64_t)source->rank - 1, &axis))
            return VX_NATIVE_GPU_VALUE_REJECTED;
        if (axis < 0) axis += source->rank;
        return source->maximums[axis] > 0 &&
               source->maximums[axis] <= maximum_exclusive
            ? VX_NATIVE_GPU_VALUE_STATICALLY_PROVEN
            : VX_NATIVE_GPU_VALUE_REJECTED;
    }
    if (vx_native_gpu_value_preserving_node(producer)) {
        const char* source_name = vx_native_gpu_validation_node_data_name(
            producer);
        return vx_native_gpu_i32_value_source_proof(
            validation, tensors, tensor_count, source_name,
            minimum, maximum_exclusive, before_node, depth + 1);
    }
    return VX_NATIVE_GPU_VALUE_REJECTED;
}

static int vx_native_gpu_finite_f32_value_source_proof(
        const VxEngineState* validation,
        const char* tensor_name,
        int before_node,
        int depth) {
    const T* tensor;
    const Node* producer;
    if (!validation || !tensor_name || before_node < 0 ||
        before_node > validation->node_count || depth < 0 ||
        depth > validation->node_count)
        return VX_NATIVE_GPU_VALUE_REJECTED;
    tensor = vx_native_gpu_validation_tensor_find(validation, tensor_name);
    if (!tensor || tensor->dtype != T_F32 || !tensor->data ||
        tensor->numel <= 0)
        return VX_NATIVE_GPU_VALUE_REJECTED;
    if (tensor->is_graph_input)
        return VX_NATIVE_GPU_VALUE_PUBLIC_PREFLIGHT;
    producer = vx_native_gpu_validation_node_producer(validation, tensor_name);
    if (!producer) {
        for (long index = 0; index < tensor->numel; index++) {
            float value;
            memcpy(&value,
                   (const unsigned char*)tensor->data +
                       (size_t)index * sizeof(value),
                   sizeof(value));
            if (!isfinite(value))
                return VX_NATIVE_GPU_VALUE_REJECTED;
        }
        return VX_NATIVE_GPU_VALUE_STATICALLY_PROVEN;
    }
    {
        int producer_index = vx_native_gpu_validation_node_index(
            validation, producer);
        if (producer_index < 0 || producer_index >= before_node)
            return VX_NATIVE_GPU_VALUE_REJECTED;
        before_node = producer_index;
    }
    if (vx_native_gpu_value_preserving_node(producer)) {
        const char* source_name = vx_native_gpu_validation_node_data_name(
            producer);
        return vx_native_gpu_finite_f32_value_source_proof(
            validation, source_name, before_node, depth + 1);
    }
    return VX_NATIVE_GPU_VALUE_REJECTED;
}

static VxStatus vx_native_gpu_node_domain_proof(
        const cJSON* node,
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        const VxEngineState* validation,
        VolvoxAIEngineBackend backend,
        const VxNativeGpuDomainLimits* limits,
        char* evidence,
        size_t evidence_capacity) {
    const cJSON* op_json = node
        ? cJSON_GetObjectItemCaseSensitive(node, "opType") : NULL;
    const cJSON* params = node
        ? cJSON_GetObjectItemCaseSensitive(node, "params") : NULL;
    const char* op = cJSON_IsString(op_json) ? op_json->valuestring : NULL;
    VxDeclaredTensor input_fixed = {0};
    const VxDeclaredTensor* input = vx_native_gpu_node_data_domain(
        node, tensors, tensor_count, validation, &input_fixed);
    const VxDeclaredTensor* output = vx_native_gpu_node_output(
        node, tensors, tensor_count);
    const Node* validation_node = output
        ? vx_native_gpu_validation_node_producer(validation, output->name)
        : NULL;
    const int validation_node_index = vx_native_gpu_validation_node_index(
        validation, validation_node);
    int64_t value;
    if (!op || !output || !limits)
        return vx_native_gpu_domain_reject(
            node, "missing-logical-output", evidence, evidence_capacity);

    if (!strcmp(op, "QConv2D")) {
        const cJSON* weight_only = cJSON_IsObject(params)
            ? cJSON_GetObjectItemCaseSensitive(params, "weight_only")
            : NULL;
        uint64_t output_elements;
        /* Weight-only QConv2D dequantizes into a CPU cache and has no strict
         * native-GPU device route. Reject before validation forward instead
         * of allocating an unbounded CPU-only cache on a doomed route. */
        if (weight_only && (!cJSON_IsBool(weight_only) ||
                            cJSON_IsTrue(weight_only)))
            return vx_native_gpu_domain_reject(
                node, "qconv-weight-only-device-route", evidence,
                evidence_capacity);
        if (!input || input->rank != 4u || output->rank != 4u ||
            !vx_declared_dimensions_provably_equal(input, 0u, output, 0u) ||
            vx_dimension_domain_is_dynamic(input, 3u) ||
            vx_dimension_domain_is_dynamic(output, 3u) ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "qconv-channel-or-batch-domain", evidence,
                evidence_capacity);
        /* Kernel geometry, activation zero point, per-channel weight zero
         * points, and the optional aligned bias copy are invariant across a
         * shape rebind. Prove the exact worst-case I32 sum for every output
         * channel before admitting the device route. */
        if (!vx_native_gpu_qconv_accumulator_domain_proven(
                validation, input, output))
            return vx_native_gpu_domain_reject(
                node, "qconv-accumulator-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "QLinear") || !strcmp(op, "QMatMul") ||
               !strcmp(op, "QGemm")) {
        if (!input ||
            (input->dtype != VX_DTYPE_I8 &&
             input->dtype != VX_DTYPE_U8) ||
            (output->dtype != VX_DTYPE_I8 &&
             output->dtype != VX_DTYPE_U8) ||
            !vx_native_gpu_dense_dynamic_rows_proven(
                input, output, 1, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "qlinear-nonrow-dynamic-axis", evidence,
                evidence_capacity);
        /* K, affine metadata, per-channel zero points, and bias are immutable
         * across a shape rebind. Prove their exact I32 accumulator bound once;
         * only the independent row count may vary at execution. */
        if (!vx_native_gpu_qlinear_accumulator_domain_proven(
                validation, input, output))
            return vx_native_gpu_domain_reject(
                node, "qlinear-accumulator-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Linear")) {
        if (!input || input->dtype != VX_DTYPE_F32 ||
            output->dtype != VX_DTYPE_F32 ||
            !vx_native_gpu_dense_dynamic_rows_proven(
                input, output, 0, backend, limits) ||
            !vx_native_gpu_linear_storage_domain_proven(
                node, tensors, tensor_count, validation, input, output))
            return vx_native_gpu_domain_reject(
                node, "linear-f32-immutable-layout-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "GELU") || !strcmp(op, "SiLU") ||
               !strcmp(op, "Sigmoid") ||
               !strcmp(op, "QGELU") || !strcmp(op, "QSiLU") ||
               !strcmp(op, "Clip") || !strcmp(op, "Not") ||
               !strcmp(op, "Cast") ||
               !strcmp(op, "QuantizeLinear") ||
               !strcmp(op, "DequantizeLinear") ||
               !strcmp(op, "RequantizeLinear")) {
        uint64_t output_elements;
        if (!vx_native_gpu_tensor_shapes_equal(input, output) ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "shape-preserving-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Reshape") || !strcmp(op, "Squeeze") ||
               !strcmp(op, "Unsqueeze")) {
        uint64_t output_elements;
        if (!vx_native_gpu_reshape_element_domain_proven(input, output) ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "reshape-element-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Transpose")) {
        uint64_t output_elements;
        if (!vx_native_gpu_transpose_domain_proven(input, output, params) ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "transpose-permutation-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Concat")) {
        /* The affine shape proof above establishes every input/output shape
         * relation. This native-GPU tail covers the host stack arrays and
         * its signed-int inner/output-axis launch parameters. */
        uint64_t output_elements;
        if (!vx_native_gpu_concat_launch_domain_proven(node, output) ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "concat-count-or-index-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Add") || !strcmp(op, "QAdd") ||
               !strcmp(op, "Mul") ||
               !strcmp(op, "Sub") || !strcmp(op, "Div") ||
               !strcmp(op, "Equal") ||
               !strcmp(op, "GreaterOrEqual")) {
        VxDeclaredTensor a_fixed = {0};
        VxDeclaredTensor b_fixed = {0};
        const VxDeclaredTensor* a = vx_native_gpu_node_input_domain(
            node, "a", tensors, tensor_count, validation, &a_fixed);
        const VxDeclaredTensor* b = vx_native_gpu_node_input_domain(
            node, "b", tensors, tensor_count, validation, &b_fixed);
        int exact = !strcmp(op, "Add") || !strcmp(op, "QAdd");
        if (!a) a = vx_native_gpu_node_input_domain(
            node, "input", tensors, tensor_count, validation, &a_fixed);
        if (!a) a = vx_native_gpu_node_input_domain(
            node, "data", tensors, tensor_count, validation, &a_fixed);
        if (!vx_native_gpu_binary_broadcast_domain_proven(a, b, output, exact))
            return vx_native_gpu_domain_reject(
                node, exact ? "binary-exact-domain" :
                    "binary-broadcast-domain",
                evidence, evidence_capacity);
        {
            uint64_t output_elements;
            if (!vx_native_gpu_tensor_maximum_elements(
                    output, &output_elements) ||
                !vx_native_gpu_one_d_launch_proven(
                    output_elements, backend, limits))
                return vx_native_gpu_domain_reject(
                    node, "binary-launch-domain", evidence,
                    evidence_capacity);
        }
    } else if (!strcmp(op, "BatchMatMul") ||
               !strcmp(op, "QBatchMatMul")) {
        VxDeclaredTensor a_fixed = {0};
        VxDeclaredTensor b_fixed = {0};
        const VxDeclaredTensor* a = vx_native_gpu_node_input_domain(
            node, "a", tensors, tensor_count, validation, &a_fixed);
        const VxDeclaredTensor* b = vx_native_gpu_node_input_domain(
            node, "b", tensors, tensor_count, validation, &b_fixed);
        int quantized = !strcmp(op, "QBatchMatMul");
        if (!a || !b ||
            (!quantized &&
             (a->dtype != VX_DTYPE_F32 || b->dtype != VX_DTYPE_F32 ||
              output->dtype != VX_DTYPE_F32)) ||
            (quantized &&
             ((a->dtype != VX_DTYPE_I8 && a->dtype != VX_DTYPE_U8) ||
              (b->dtype != VX_DTYPE_I8 && b->dtype != VX_DTYPE_U8) ||
              (output->dtype != VX_DTYPE_I8 &&
               output->dtype != VX_DTYPE_U8))) ||
            !vx_native_gpu_batch_matmul_domain_proven(a, b, output))
            return vx_native_gpu_domain_reject(
                node, "batch-matmul-broadcast-or-contract-domain",
                evidence, evidence_capacity);
        if (quantized) {
            uint64_t output_elements;
            if (!vx_native_gpu_tensor_maximum_elements(
                    output, &output_elements) ||
                !vx_native_gpu_one_d_launch_proven(
                    output_elements, backend, limits))
                return vx_native_gpu_domain_reject(
                    node, "batch-matmul-launch-domain", evidence,
                    evidence_capacity);
            uint64_t k = (uint64_t)a->maximums[a->rank - 1u];
            if (k > (uint64_t)INT32_MAX / UINT64_C(65025))
                return vx_native_gpu_domain_reject(
                    node, "qbatch-accumulator-domain", evidence,
                    evidence_capacity);
        } else if (backend == VOLVOXAI_BACKEND_CUDA) {
            uint64_t output_elements;
            /* CUDA's F32 implementation is a flat output-element launch,
             * unlike the 8x8 tiled V/O/M kernels below. Prove that launch
             * against the queried grid as well as the tensor byte domain. */
            if (!vx_native_gpu_tensor_maximum_elements(
                    output, &output_elements) ||
                !vx_native_gpu_one_d_launch_proven(
                    output_elements, backend, limits))
                return vx_native_gpu_domain_reject(
                    node, "batch-matmul-launch-domain", evidence,
                    evidence_capacity);
        } else {
            uint64_t batches;
            uint64_t m = (uint64_t)output->maximums[output->rank - 2u];
            uint64_t n = (uint64_t)output->maximums[output->rank - 1u];
            if (!vx_native_gpu_tensor_maximum_product(
                    output, 0u, output->rank - 2u,
                    UINT32_MAX, &batches) ||
                limits->maximum_block[0] < 8u ||
                limits->maximum_block[1] < 8u ||
                limits->maximum_threads_per_block < 64u ||
                (n + 7u) / 8u > limits->maximum_grid[0] ||
                (m + 7u) / 8u > limits->maximum_grid[1] ||
                batches > limits->maximum_grid[2])
                return vx_native_gpu_domain_reject(
                    node, "batch-matmul-tiled-launch-domain", evidence,
                    evidence_capacity);
        }
    } else if (!strcmp(op, "QSDPA")) {
        VxDeclaredTensor q_fixed = {0};
        VxDeclaredTensor k_fixed = {0};
        VxDeclaredTensor v_fixed = {0};
        VxDeclaredTensor mask_fixed = {0};
        const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(
            node, "inputs");
        const cJSON* mask_reference = cJSON_IsObject(inputs)
            ? cJSON_GetObjectItemCaseSensitive(inputs, "mask") : NULL;
        const VxDeclaredTensor* q = vx_native_gpu_node_input_domain(
            node, "q", tensors, tensor_count, validation, &q_fixed);
        const VxDeclaredTensor* k = vx_native_gpu_node_input_domain(
            node, "k", tensors, tensor_count, validation, &k_fixed);
        const VxDeclaredTensor* v = vx_native_gpu_node_input_domain(
            node, "v", tensors, tensor_count, validation, &v_fixed);
        const VxDeclaredTensor* mask = vx_native_gpu_node_input_domain(
            node, "mask", tensors, tensor_count, validation, &mask_fixed);
        int64_t heads;
        int64_t d_model;
        uint64_t batch;
        uint64_t seq_q;
        uint64_t tasks;
        int mask_proven = !mask_reference;
        if (!q || !k || !v || (q->rank != 2u && q->rank != 3u) ||
            k->rank != q->rank || v->rank != q->rank ||
            output->rank != q->rank ||
            (q->dtype != VX_DTYPE_I8 && q->dtype != VX_DTYPE_U8) ||
            (k->dtype != VX_DTYPE_I8 && k->dtype != VX_DTYPE_U8) ||
            (v->dtype != VX_DTYPE_I8 && v->dtype != VX_DTYPE_U8) ||
            (output->dtype != VX_DTYPE_I8 &&
             output->dtype != VX_DTYPE_U8) ||
            !vx_native_gpu_tensor_shapes_equal(q, output) ||
            !vx_native_gpu_tensor_shapes_equal(k, v) ||
            !vx_declared_dimensions_provably_equal(
                q, q->rank - 1u, k, k->rank - 1u) ||
            !vx_declared_dimension_fixed_value(
                q, q->rank - 1u, &d_model) ||
            d_model <= 0 ||
            !vx_native_gpu_integer_parameter(
                params, "heads", 0, 1, INT_MAX, &heads) ||
            d_model % heads || d_model % 4 ||
            d_model / heads > 64 || (d_model / heads) % 4)
            return vx_native_gpu_domain_reject(
                node, "qsdpa-rank-feature-or-head-domain", evidence,
                evidence_capacity);
        if (q->rank == 3u &&
            (!vx_declared_dimensions_provably_equal(q, 0u, k, 0u) ||
             !vx_declared_dimensions_provably_equal(q, 0u, v, 0u)))
            return vx_native_gpu_domain_reject(
                node, "qsdpa-batch-domain", evidence, evidence_capacity);
        batch = q->rank == 3u ? (uint64_t)q->maximums[0] : 1u;
        seq_q = (uint64_t)q->maximums[q->rank - 2u];
        if (!batch || !seq_q || (uint64_t)heads > UINT32_MAX ||
            batch > UINT32_MAX / seq_q ||
            batch * seq_q > UINT32_MAX / (uint64_t)heads)
            return vx_native_gpu_domain_reject(
                node, "qsdpa-task-domain", evidence, evidence_capacity);
        tasks = batch * seq_q * (uint64_t)heads;
        if (seq_q > limits->maximum_grid[0] ||
            (uint64_t)heads > limits->maximum_grid[1] ||
            batch > limits->maximum_grid[2] ||
            !vx_native_gpu_one_d_launch_proven(tasks, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "qsdpa-launch-domain", evidence, evidence_capacity);
        if (mask_reference && !mask)
            return vx_native_gpu_domain_reject(
                node, "qsdpa-unresolved-mask-domain", evidence,
                evidence_capacity);
        if (mask) {
            int64_t fixed_first = 0;
            if (mask->dtype == VX_DTYPE_I32 && mask->rank == 1u &&
                vx_declared_dimensions_provably_equal(
                    mask, 0u, k, k->rank - 2u)) {
                mask_proven = 1;
            } else if (mask->dtype == VX_DTYPE_I32 && mask->rank == 2u &&
                       vx_declared_dimensions_provably_equal(
                           mask, 1u, k, k->rank - 2u) &&
                       (vx_declared_dimensions_provably_equal(
                            mask, 0u, q, q->rank - 2u) ||
                        (q->rank == 3u &&
                         vx_declared_dimensions_provably_equal(
                             mask, 0u, q, 0u)) ||
                        (q->rank == 2u &&
                         vx_declared_dimension_fixed_value(
                             mask, 0u, &fixed_first) && fixed_first == 1))) {
                mask_proven = 1;
            } else if (mask->dtype == VX_DTYPE_I32 && mask->rank == 3u &&
                       q->rank == 3u &&
                       vx_declared_dimensions_provably_equal(
                           mask, 0u, q, 0u) &&
                       vx_declared_dimensions_provably_equal(
                           mask, 1u, q, 1u) &&
                       vx_declared_dimensions_provably_equal(
                           mask, 2u, k, 1u)) {
                mask_proven = 1;
            }
        }
        if (!mask_proven)
            return vx_native_gpu_domain_reject(
                node, "qsdpa-mask-domain", evidence, evidence_capacity);
    } else if (!strcmp(op, "QMaskedMean")) {
        VxDeclaredTensor mask_fixed = {0};
        const VxDeclaredTensor* mask = vx_native_gpu_node_input_domain(
            node, "mask", tensors, tensor_count, validation, &mask_fixed);
        uint64_t output_elements;
        if (!input || !mask || input->rank != 3u || mask->rank != 2u ||
            output->rank != 2u || mask->dtype != VX_DTYPE_I32 ||
            (input->dtype != VX_DTYPE_I8 &&
             input->dtype != VX_DTYPE_U8) ||
            (output->dtype != VX_DTYPE_I8 &&
             output->dtype != VX_DTYPE_U8) ||
            !vx_declared_dimensions_provably_equal(input, 0u, mask, 0u) ||
            !vx_declared_dimensions_provably_equal(input, 1u, mask, 1u) ||
            !vx_declared_dimensions_provably_equal(input, 0u, output, 0u) ||
            !vx_declared_dimensions_provably_equal(input, 2u, output, 1u) ||
            (uint64_t)input->maximums[1] >
                (uint64_t)INT32_MAX / 255u ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "qmaskedmean-shape-accumulator-or-launch-domain",
                evidence, evidence_capacity);
    } else if (!strcmp(op, "QArgMax")) {
        uint64_t outer;
        uint64_t inner;
        uint64_t output_elements;
        int64_t axis;
        if (!input || input->rank < 2u || input->rank > 8u ||
            output->rank + 1u != input->rank ||
            output->dtype != VX_DTYPE_I32 ||
            !vx_native_gpu_integer_parameter(
                params, "axis", 0, -(int64_t)input->rank,
                (int64_t)input->rank - 1, &axis))
            return vx_native_gpu_domain_reject(
                node, "qargmax-parameters-or-rank", evidence,
                evidence_capacity);
        if (axis < 0) axis += input->rank;
        for (uint32_t input_axis = 0, output_axis = 0;
             input_axis < input->rank; input_axis++) {
            if (input_axis == (uint32_t)axis) continue;
            if (output_axis >= output->rank ||
                !vx_declared_dimensions_provably_equal(
                    input, input_axis, output, output_axis++))
                return vx_native_gpu_domain_reject(
                    node, "qargmax-output-domain", evidence,
                    evidence_capacity);
        }
        if (!vx_native_gpu_tensor_maximum_product(
                input, 0u, (uint32_t)axis, UINT32_MAX, &outer) ||
            !vx_native_gpu_tensor_maximum_product(
                input, (uint32_t)axis + 1u, input->rank,
                UINT32_MAX, &inner) ||
            outer > UINT32_MAX / inner ||
            (uint64_t)input->maximums[axis] >
                UINT32_MAX / (outer * inner))
            return vx_native_gpu_domain_reject(
                node, "qargmax-index-domain", evidence,
                evidence_capacity);
        output_elements = outer * inner;
        if (!vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "qargmax-launch-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "ArgMax")) {
        uint64_t outer;
        uint64_t inner;
        uint64_t output_elements;
        int64_t axis;
        int64_t keepdims;
        int64_t select_last;
        if (!input || input->rank < 1u || input->rank > 8u ||
            !vx_native_gpu_integer_parameter(
                params, "axis", 0, -(int64_t)input->rank,
                (int64_t)input->rank - 1, &axis) ||
            !vx_native_gpu_integer_parameter(
                params, "keepdims", 1, 0, 1, &keepdims) ||
            !vx_native_gpu_integer_parameter(
                params, "select_last_index", 0, 0, 0, &select_last))
            return vx_native_gpu_domain_reject(
                node, "argmax-parameters", evidence, evidence_capacity);
        if (axis < 0) axis += input->rank;
        if (!vx_native_gpu_tensor_maximum_product(
                input, 0u, (uint32_t)axis, UINT32_MAX, &outer) ||
            !vx_native_gpu_tensor_maximum_product(
                input, (uint32_t)axis + 1u, input->rank,
                UINT32_MAX, &inner) ||
            outer > UINT32_MAX / inner)
            return vx_native_gpu_domain_reject(
                node, "argmax-index-domain", evidence, evidence_capacity);
        output_elements = outer * inner;
        if ((uint64_t)input->maximums[axis] >
                UINT32_MAX / output_elements ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "argmax-element-or-launch-domain", evidence,
                evidence_capacity);
        (void)keepdims;
        (void)select_last;
    } else if (!strcmp(op, "Gather")) {
        const VxDeclaredTensor* indices = vx_native_gpu_node_input_named(
            node, "indices", tensors, tensor_count);
        const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(
            node, "inputs");
        const cJSON* indices_reference = cJSON_IsObject(inputs)
            ? cJSON_GetObjectItemCaseSensitive(inputs, "indices") : NULL;
        uint64_t indices_elements;
        uint64_t output_elements;
        if (!input || input->dtype != VX_DTYPE_F32 ||
            !vx_native_gpu_integer_parameter(params, "axis", 0, 0, 0, &value) ||
            !indices || indices->dtype != VX_DTYPE_I32 ||
            !cJSON_IsString(indices_reference) ||
            !indices_reference->valuestring ||
            !validation_node || validation_node->resident_slot_rows ||
            input->minimums[0] <= 0 || input->minimums[0] > INT_MAX ||
            vx_native_gpu_i32_value_source_proof(
                validation, tensors, tensor_count,
                indices_reference->valuestring,
                -input->minimums[0], input->minimums[0],
                validation_node_index, 0) ==
                VX_NATIVE_GPU_VALUE_REJECTED ||
            !vx_native_gpu_tensor_maximum_elements(indices, &indices_elements) ||
            !vx_native_gpu_tensor_maximum_elements(output, &output_elements) ||
            indices_elements > INT_MAX || output_elements > INT_MAX ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, validation_node && validation_node->resident_slot_rows
                    ? "banked-gather-global-slot-domain"
                    : "gather-axis0-count-or-index-origin-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Slice")) {
        const cJSON* starts = cJSON_IsObject(params)
            ? cJSON_GetObjectItemCaseSensitive(params, "starts") : NULL;
        const cJSON* ends = cJSON_IsObject(params)
            ? cJSON_GetObjectItemCaseSensitive(params, "ends") : NULL;
        const cJSON* steps = cJSON_IsObject(params)
            ? cJSON_GetObjectItemCaseSensitive(params, "steps") : NULL;
        const cJSON* axes = cJSON_IsObject(params)
            ? cJSON_GetObjectItemCaseSensitive(params, "axes") : NULL;
        int64_t starts_values[8] = {0};
        int64_t ends_values[8] = {0};
        int64_t steps_values[8] = {0};
        int64_t axes_values[8] = {0};
        size_t start_count = 0;
        size_t end_count = 0;
        size_t step_count = 0;
        size_t axis_count = 0;
        unsigned int seen = 0u;
        int64_t axis_starts[8] = {0};
        int64_t axis_ends[8] = {0};
        int64_t axis_steps[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        if (!input || !input->rank || input->rank > 8u ||
            output->rank != input->rank || output->dtype != input->dtype ||
            (input->dtype != VX_DTYPE_F32 &&
             input->dtype != VX_DTYPE_I32) ||
            !vx_native_gpu_integer_array(
                starts, starts_values, 8u, &start_count) ||
            !vx_native_gpu_integer_array(
                ends, ends_values, 8u, &end_count) ||
            end_count != start_count ||
            (steps && (!vx_native_gpu_integer_array(
                steps, steps_values, 8u, &step_count) ||
                step_count != start_count)) ||
            (axes && (!vx_native_gpu_integer_array(
                axes, axes_values, 8u, &axis_count) ||
                axis_count != start_count)) || start_count > input->rank)
            return vx_native_gpu_domain_reject(
                node, "slice-static-parameters", evidence,
                evidence_capacity);
        for (size_t index = 0; index < start_count; index++) {
            int64_t axis = axes ? axes_values[index] : (int64_t)index;
            int64_t step = steps ? steps_values[index] : 1;
            if (axis < 0) axis += input->rank;
            if (axis < 0 || axis >= input->rank || step <= 0 ||
                starts_values[index] < 0 ||
                (seen & (1u << (unsigned int)axis)))
                return vx_native_gpu_domain_reject(
                    node, "slice-positive-static-domain", evidence,
                    evidence_capacity);
            seen |= 1u << (unsigned int)axis;
            axis_starts[axis] = starts_values[index];
            axis_ends[axis] = ends_values[index];
            axis_steps[axis] = step;
        }
        for (uint32_t axis = 0; axis < input->rank; axis++)
            if (!vx_native_gpu_slice_axis_domain_proven(
                    input, output, axis,
                    (seen & (1u << axis)) != 0u,
                    axis_starts[axis], axis_ends[axis], axis_steps[axis]))
                return vx_native_gpu_domain_reject(
                    node, "slice-output-or-index-domain", evidence,
                    evidence_capacity);
        {
            uint64_t output_elements;
            if (!vx_native_gpu_tensor_maximum_elements(
                    output, &output_elements) ||
                !vx_native_gpu_one_d_launch_proven(
                    output_elements, backend, limits))
                return vx_native_gpu_domain_reject(
                    node, "slice-launch-domain", evidence,
                    evidence_capacity);
        }
    } else if (!strcmp(op, "Softmax")) {
        int64_t axis;
        uint64_t rows;
        if (!input || !input->rank ||
            !vx_native_gpu_integer_parameter(
                params, "axis", (int64_t)input->rank - 1,
                -(int64_t)input->rank,
                (int64_t)input->rank - 1, &axis))
            return vx_native_gpu_domain_reject(
                node, "softmax-parameters", evidence, evidence_capacity);
        if (axis < 0) axis += input->rank;
        if (axis != (int64_t)input->rank - 1 ||
            !vx_native_gpu_tensor_maximum_product(
                input, 0u, input->rank - 1u, INT_MAX, &rows) ||
            !vx_native_gpu_one_d_launch_proven(rows, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "softmax-last-axis-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "ReduceSum")) {
        int64_t axis;
        uint64_t rows;
        if (!input || !input->rank ||
            !vx_native_gpu_integer_parameter(
                params, "axis", (int64_t)input->rank - 1,
                -(int64_t)input->rank,
                (int64_t)input->rank - 1, &axis))
            return vx_native_gpu_domain_reject(
                node, "reduce-parameters", evidence, evidence_capacity);
        if (axis < 0) axis += input->rank;
        if (axis != (int64_t)input->rank - 1 ||
            !vx_native_gpu_tensor_maximum_product(
                input, 0u, input->rank - 1u, INT_MAX, &rows) ||
            !vx_native_gpu_one_d_launch_proven(rows, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "reduce-last-axis-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Where")) {
        const VxDeclaredTensor* condition = vx_native_gpu_node_input_named(
            node, "condition", tensors, tensor_count);
        const VxDeclaredTensor* a = vx_native_gpu_node_input_named(
            node, "x", tensors, tensor_count);
        const VxDeclaredTensor* b = vx_native_gpu_node_input_named(
            node, "y", tensors, tensor_count);
        if (!condition) condition = vx_native_gpu_node_input_named(
            node, "cond", tensors, tensor_count);
        if (!a) a = vx_native_gpu_node_input_named(
            node, "a", tensors, tensor_count);
        if (!b) b = vx_native_gpu_node_input_named(
            node, "b", tensors, tensor_count);
        uint64_t output_elements;
        if (!condition || !a || !b ||
            condition->dtype != VX_DTYPE_I32 ||
            (a->dtype != VX_DTYPE_F32 && a->dtype != VX_DTYPE_I32) ||
            !vx_native_gpu_tensor_shapes_equal(condition, output) ||
            !vx_native_gpu_tensor_domains_equal(a, output) ||
            !vx_native_gpu_tensor_domains_equal(b, output) ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "where-exact-i32-condition-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Expand")) {
        uint64_t output_elements;
        if (!input ||
            (input->dtype != VX_DTYPE_F32 &&
             input->dtype != VX_DTYPE_I32) ||
            input->dtype != output->dtype ||
            !vx_native_gpu_expand_domain_proven(input, output) ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "expand-broadcast-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "LayerNorm") || !strcmp(op, "QLayerNorm")) {
        const char* weight_name = validation_node
            ? vx_native_gpu_validation_node_input_name(
                validation_node, "weight") : NULL;
        const char* bias_name = validation_node
            ? vx_native_gpu_validation_node_input_name(
                validation_node, "bias") : NULL;
        uint64_t rows;
        uint64_t output_elements;
        if (!input || !input->rank ||
            !vx_native_gpu_tensor_shapes_equal(input, output) ||
            vx_dimension_domain_is_dynamic(input, input->rank - 1u) ||
            vx_dimension_domain_is_dynamic(output, output->rank - 1u) ||
            !vx_native_gpu_tensor_maximum_product(
                input, 0u, input->rank - 1u, INT_MAX, &rows) ||
            rows > limits->maximum_grid[0] ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "layernorm-feature-row-or-launch-domain", evidence,
                evidence_capacity);
        if (!strcmp(op, "QLayerNorm") &&
            (!weight_name || !bias_name ||
             vx_native_gpu_finite_f32_value_source_proof(
                 validation, weight_name, validation_node_index, 0) ==
                 VX_NATIVE_GPU_VALUE_REJECTED ||
             vx_native_gpu_finite_f32_value_source_proof(
                 validation, bias_name, validation_node_index, 0) ==
                 VX_NATIVE_GPU_VALUE_REJECTED))
            return vx_native_gpu_domain_reject(
                node, "qlayernorm-affine-value-origin-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "GroupNorm") || !strcmp(op, "QGroupNorm")) {
        const char* weight_name = validation_node
            ? vx_native_gpu_validation_node_input_name(
                validation_node, "weight") : NULL;
        const char* bias_name = validation_node
            ? vx_native_gpu_validation_node_input_name(
                validation_node, "bias") : NULL;
        const cJSON* num_groups = cJSON_IsObject(params)
            ? cJSON_GetObjectItemCaseSensitive(params, "num_groups") : NULL;
        int64_t groups = 1;
        uint64_t tasks;
        uint64_t output_elements;
        if (!input || input->rank != 4u || output->rank != 4u ||
            !vx_native_gpu_tensor_shapes_equal(input, output) ||
            vx_dimension_domain_is_dynamic(input, 3u) ||
            vx_dimension_domain_is_dynamic(output, 3u) ||
            !vx_native_gpu_integer_parameter(
                params, num_groups ? "num_groups" : "groups", 1,
                1, INT_MAX, &groups) ||
            groups > input->maximums[3] ||
            input->maximums[3] % groups ||
            (uint64_t)input->maximums[0] >
                (uint64_t)INT_MAX / (uint64_t)groups)
            return vx_native_gpu_domain_reject(
                node, "groupnorm-channel-or-task-domain", evidence,
                evidence_capacity);
        tasks = (uint64_t)input->maximums[0] * (uint64_t)groups;
        if (limits->maximum_block[0] < 64u ||
            limits->maximum_threads_per_block < 64u ||
            tasks > limits->maximum_grid[0])
            return vx_native_gpu_domain_reject(
                node, "groupnorm-launch-domain", evidence,
                evidence_capacity);
        if (!vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "groupnorm-apply-launch-domain", evidence,
                evidence_capacity);
        if (!strcmp(op, "QGroupNorm") &&
            (!weight_name || !bias_name ||
             vx_native_gpu_finite_f32_value_source_proof(
                 validation, weight_name, validation_node_index, 0) ==
                 VX_NATIVE_GPU_VALUE_REJECTED ||
             vx_native_gpu_finite_f32_value_source_proof(
                 validation, bias_name, validation_node_index, 0) ==
                 VX_NATIVE_GPU_VALUE_REJECTED))
            return vx_native_gpu_domain_reject(
                node, "qgroupnorm-affine-value-origin-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Conv2D")) {
        VxDeclaredTensor weight_fixed = {0};
        const VxDeclaredTensor* logical_weight =
            vx_native_gpu_node_input_named(
                node, "weight", tensors, tensor_count);
        const VxDeclaredTensor* logical_bias =
            vx_native_gpu_node_input_named(
                node, "bias", tensors, tensor_count);
        const VxDeclaredTensor* weight = vx_native_gpu_node_input_domain(
            node, "weight", tensors, tensor_count, validation,
            &weight_fixed);
        const cJSON* layout_item = cJSON_IsObject(params)
            ? cJSON_GetObjectItemCaseSensitive(params, "weight_layout")
            : NULL;
        const char* layout = !layout_item || cJSON_IsNull(layout_item)
            ? "HWIO"
            : cJSON_IsString(layout_item) && layout_item->valuestring
                ? layout_item->valuestring : NULL;
        int direct_f32_layout = layout &&
            (!strcmp(layout, "HWIO") || !strcmp(layout, "HWCM"));
        uint64_t output_channels;
        if (!input || !weight || input->rank != 4u ||
            weight->rank != 4u || output->rank != 4u ||
            !vx_declared_dimensions_provably_equal(input, 0u, output, 0u) ||
            vx_dimension_domain_is_dynamic(input, 3u) ||
            vx_dimension_domain_is_dynamic(output, 3u))
            return vx_native_gpu_domain_reject(
                node, "conv-dynamic-channel", evidence,
                evidence_capacity);
        /* The current Conv2D shape contract is canonical HWIO/HWCM. Reject
         * legacy transformed layouts even for immutable storage rather than
         * compiling a route whose first public execute cannot shape-bind.
         * Within the canonical layouts, immutable F16 may own a widening
         * cache while a logical/public weight must remain direct F32. */
        if (!direct_f32_layout)
            return vx_native_gpu_domain_reject(
                node, "conv-noncanonical-weight-layout-domain",
                evidence, evidence_capacity);
        if ((logical_weight && weight->dtype != VX_DTYPE_F32) ||
            (logical_bias && logical_bias->dtype == VX_DTYPE_F16))
            return vx_native_gpu_domain_reject(
                node, logical_weight
                    ? "conv-mutable-transformed-weight-domain"
                    : "conv-mutable-f16-bias-domain",
                evidence, evidence_capacity);
        output_channels = (uint64_t)output->maximums[3];
        if (backend == VOLVOXAI_BACKEND_CUDA) {
            const cJSON* pads_item = cJSON_IsObject(params)
                ? cJSON_GetObjectItemCaseSensitive(params, "pads") : NULL;
            int64_t stride_y;
            int64_t stride_x;
            int64_t padding_y;
            int64_t padding_x;
            int64_t dilation_y;
            int64_t dilation_x;
            int64_t groups;
            int64_t pads[4];
            size_t pad_count = 0u;
            uint64_t minimum_rows = 1u;
            uint64_t maximum_rows = 1u;
            int64_t fixed_kernel_height;
            int64_t fixed_kernel_width;
            int tiled_1x1_route = VX_NATIVE_GPU_CUDA_CONV_TILED_NEVER;
            if (!vx_native_gpu_pair_parameter(
                    params, "stride", 1, 1, INT_MAX,
                    &stride_y, &stride_x) ||
                !vx_native_gpu_pair_parameter(
                    params, "padding", 0, 0, INT_MAX,
                    &padding_y, &padding_x) ||
                !vx_native_gpu_pair_parameter(
                    params, "dilation", 1, 1, INT_MAX,
                    &dilation_y, &dilation_x) ||
                !vx_native_gpu_integer_parameter(
                    params, "groups", 1, 1, INT_MAX, &groups))
                return vx_native_gpu_domain_reject(
                    node, "conv-cuda-launch-domain", evidence,
                    evidence_capacity);
            pads[0] = padding_y;
            pads[1] = padding_x;
            pads[2] = padding_y;
            pads[3] = padding_x;
            if (pads_item &&
                (!vx_native_gpu_integer_array(
                     pads_item, pads, 4u, &pad_count) ||
                 pad_count != 4u || pads[0] < 0 || pads[1] < 0 ||
                 pads[2] < 0 || pads[3] < 0))
                return vx_native_gpu_domain_reject(
                    node, "conv-cuda-launch-domain", evidence,
                    evidence_capacity);
            for (uint32_t axis = 0u; axis < 3u; axis++) {
                uint64_t minimum_extent;
                uint64_t maximum_extent;
                if (output->minimums[axis] <= 0 ||
                    output->maximums[axis] < output->minimums[axis])
                    return vx_native_gpu_domain_reject(
                        node, "conv-cuda-launch-domain", evidence,
                        evidence_capacity);
                minimum_extent = (uint64_t)output->minimums[axis];
                maximum_extent = (uint64_t)output->maximums[axis];
                if (minimum_rows > UINT64_MAX / minimum_extent ||
                    maximum_rows > UINT64_MAX / maximum_extent)
                    return vx_native_gpu_domain_reject(
                        node, "conv-cuda-launch-domain", evidence,
                        evidence_capacity);
                minimum_rows *= minimum_extent;
                maximum_rows *= maximum_extent;
            }
            if (groups == 1 && stride_y == 1 && stride_x == 1 &&
                dilation_y == 1 && dilation_x == 1 &&
                pads[0] == 0 && pads[1] == 0 &&
                pads[2] == 0 && pads[3] == 0 &&
                vx_native_gpu_dimension_domain_contains(weight, 0u, 1) &&
                vx_native_gpu_dimension_domain_contains(weight, 1u, 1)) {
                int height_fixed_one = vx_declared_dimension_fixed_value(
                    weight, 0u, &fixed_kernel_height) &&
                    fixed_kernel_height == 1;
                int width_fixed_one = vx_declared_dimension_fixed_value(
                    weight, 1u, &fixed_kernel_width) &&
                    fixed_kernel_width == 1;
                int spatial_fixed_equal =
                    vx_declared_dimensions_provably_equal(
                        input, 1u, output, 1u) &&
                    vx_declared_dimensions_provably_equal(
                        input, 2u, output, 2u);
                tiled_1x1_route = height_fixed_one && width_fixed_one &&
                    spatial_fixed_equal
                    ? VX_NATIVE_GPU_CUDA_CONV_TILED_FIXED
                    : VX_NATIVE_GPU_CUDA_CONV_TILED_POSSIBLE;
            }
            if (!vx_native_gpu_cuda_conv2d_launch_proven(
                    minimum_rows, maximum_rows,
                    (uint64_t)input->maximums[3], output_channels,
                    tiled_1x1_route, limits))
                return vx_native_gpu_domain_reject(
                    node, "conv-cuda-launch-domain", evidence,
                    evidence_capacity);
            return VX_STATUS_OK;
        }
        {
            int groups_one = 0;
            if (backend == VOLVOXAI_BACKEND_VULKAN ||
                backend == VOLVOXAI_BACKEND_OPENGL) {
                int64_t groups;
                if (!vx_native_gpu_integer_parameter(
                        params, "groups", 1, 1, INT_MAX, &groups))
                    return vx_native_gpu_domain_reject(
                        node, "conv-grid-domain", evidence,
                        evidence_capacity);
                groups_one = groups == 1;
            }
            if (!vx_native_gpu_vom_conv2d_launch_proven(
                    (uint64_t)output->maximums[0],
                    (uint64_t)output->maximums[1],
                    (uint64_t)output->maximums[2], output_channels,
                    groups_one, backend, limits))
                return vx_native_gpu_domain_reject(
                    node, "conv-grid-domain", evidence,
                    evidence_capacity);
        }
    } else if (!strcmp(op, "MaxPool2D") ||
               !strcmp(op, "ResizeNearest2D")) {
        /* The byte domain reaches its pooling and resize kernels through the
         * separate W8A8 route, whose packed-lane launch this proof does not
         * cover, so it stays refused here rather than qualified on a proof it
         * never received. */
        if (!vx_native_gpu_nhwc_spatial_domain_proven(
                input, output, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "nhwc-spatial-launch-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "QEmbedding")) {
        VxDeclaredTensor weight_fixed = {0};
        const VxDeclaredTensor* weight = vx_native_gpu_node_input_domain(
            node, "weight", tensors, tensor_count, validation,
            &weight_fixed);
        const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(
            node, "inputs");
        const cJSON* ids_reference = cJSON_IsObject(inputs)
            ? cJSON_GetObjectItemCaseSensitive(inputs, "input") : NULL;
        uint64_t token_count;
        uint64_t output_elements;
        if (!input || !input->rank || output->rank != input->rank + 1u ||
            input->dtype != VX_DTYPE_I32 ||
            (output->dtype != VX_DTYPE_I8 &&
             output->dtype != VX_DTYPE_U8) ||
            !weight || weight->rank != 2u ||
            (weight->dtype != VX_DTYPE_I8 &&
             weight->dtype != VX_DTYPE_U8) ||
            !cJSON_IsString(ids_reference) || !ids_reference->valuestring ||
            !validation_node || validation_node->resident_slot_rows ||
            vx_dimension_domain_is_dynamic(weight, 0u) ||
            vx_dimension_domain_is_dynamic(weight, 1u) ||
            weight->minimums[0] <= 0 || weight->minimums[0] > INT_MAX ||
            vx_native_gpu_i32_value_source_proof(
                validation, tensors, tensor_count,
                ids_reference->valuestring, 0, weight->minimums[0],
                validation_node_index, 0) ==
                VX_NATIVE_GPU_VALUE_REJECTED ||
            vx_dimension_domain_is_dynamic(output, output->rank - 1u) ||
            !vx_declared_dimensions_provably_equal(
                weight, 1u, output, output->rank - 1u) ||
            !vx_native_gpu_tensor_maximum_elements(input, &token_count) ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            token_count > UINT32_MAX || output_elements > UINT32_MAX ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, validation_node && validation_node->resident_slot_rows
                    ? "banked-qembedding-global-slot-domain"
                    : "qembedding-feature-or-index-origin-domain", evidence,
                evidence_capacity);
        for (uint32_t axis = 0; axis < input->rank; axis++)
            if (!vx_declared_dimensions_provably_equal(
                    input, axis, output, axis))
                return vx_native_gpu_domain_reject(
                    node, "qembedding-prefix-domain", evidence,
                    evidence_capacity);
    } else if (!strcmp(op, "Embedding")) {
        VxDeclaredTensor weight_fixed = {0};
        const VxDeclaredTensor* weight = vx_native_gpu_node_input_domain(
            node, "weight", tensors, tensor_count, validation,
            &weight_fixed);
        const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(
            node, "inputs");
        const cJSON* ids_reference = cJSON_IsObject(inputs)
            ? cJSON_GetObjectItemCaseSensitive(inputs, "input") : NULL;
        uint64_t token_count;
        uint64_t vocabulary;
        if (!input || !input->rank || output->rank != input->rank + 1u ||
            input->dtype != VX_DTYPE_I32 || output->dtype != VX_DTYPE_F32 ||
            !weight || weight->dtype != VX_DTYPE_F32 || !weight->rank ||
            !cJSON_IsString(ids_reference) || !ids_reference->valuestring ||
            !validation_node || validation_node->resident_slot_rows ||
            vx_dimension_domain_is_dynamic(output, output->rank - 1u) ||
            !vx_declared_dimensions_provably_equal(
                weight, weight->rank - 1u,
                output, output->rank - 1u) ||
            !vx_native_gpu_tensor_maximum_elements(input, &token_count) ||
            token_count > INT_MAX ||
            !vx_native_gpu_tensor_maximum_product(
                weight, 0u, weight->rank - 1u, INT_MAX, &vocabulary) ||
            vocabulary > INT_MAX ||
            vx_native_gpu_i32_value_source_proof(
                validation, tensors, tensor_count,
                ids_reference->valuestring, 0, (int64_t)vocabulary,
                validation_node_index, 0) ==
                VX_NATIVE_GPU_VALUE_REJECTED)
            return vx_native_gpu_domain_reject(
                node, validation_node && validation_node->resident_slot_rows
                    ? "banked-embedding-global-slot-domain"
                    : "embedding-feature-or-index-origin-domain", evidence,
                evidence_capacity);
        if (!vx_native_gpu_one_d_launch_proven(
                (uint64_t)output->maximum_byte_size / sizeof(float),
                backend, limits))
            return vx_native_gpu_domain_reject(
                node, "embedding-launch-domain", evidence,
                evidence_capacity);
        for (uint32_t axis = 0; axis < input->rank; axis++)
            if (!vx_declared_dimensions_provably_equal(
                    input, axis, output, axis))
                return vx_native_gpu_domain_reject(
                    node, "embedding-prefix-domain", evidence,
                    evidence_capacity);
    } else {
        return vx_native_gpu_domain_reject(
            node, "missing-native-gpu-domain-proof", evidence,
            evidence_capacity);
    }
    return VX_STATUS_OK;
}
