/* vx_builtin_bounded_domain_proof(): the whole-graph declared-domain proof.
 *
 * Fragment of public_api.c, included where this definition used to sit.  This
 * is the compile-time gate that must prove every legal binding of the declared
 * domain before a model is accepted; it drives the per-node checks in
 * public_api_native_gpu_domain.inc for GPU backends. */

static int vx_native_gpu_tensor_size_domain_proven(
        VxBackendKind backend,
        VxDataType dtype,
        uint32_t rank,
        uint64_t maximum_elements,
        uint64_t maximum_byte_size) {
    if (!vx_native_gpu_backend(backend) || rank > 8u ||
        maximum_elements > UINT32_MAX ||
        maximum_byte_size > (uint64_t)SIZE_MAX ||
        (backend != VX_BACKEND_KIND_CUDA &&
         maximum_byte_size > UINT32_MAX) ||
        (dtype == VX_DTYPE_F16 &&
         maximum_elements > UINT32_MAX / sizeof(float)))
        return 0;
    return 1;
}


#include "public_api_webgpu_domain.inc"

static VxStatus vx_builtin_bounded_domain_proof(
        const VxModel* model,
        const VxCompiledWeightStore* weight_store,
        const VxCompiledCpuWeightStore* cpu_weights,
        VxBackendKind backend,
        VxEngineState* validation,
        VxBuiltinResourceDomain* resources,
        char* evidence,
        size_t evidence_capacity) {
    VxDeclaredTensor* tensors = NULL;
    size_t tensor_count = 0;
    const cJSON* root = NULL;
    const cJSON* nodes;
    const char* registry_backend;
    uint64_t digest = UINT64_C(1469598103934665603);
    /* The capsule and graph metadata transaction peak are resident
     * independently of tensor arenas, mapped weights, and its opaque pool.
     * Reserve publishes only after allocating its candidate, so this peak is
     * the exact high-water of old + candidate across weight shards and graph. */
    uint64_t maximum_resident = (uint64_t)sizeof(VxEngineState);
    /* The compiled store already validated every snapshot through the
     * safetensors 64-bit size path.  Reopening the paths here with ftell()
     * would truncate files above 2 GiB on 64-bit Windows and could observe a
     * different file than the immutable mapping used for execution. */
    uint64_t maximum_weight_bytes;
    uint64_t native_gpu_immutable_bytes = 0;
    uint64_t native_gpu_device_immutable_bytes = 0;
    uint64_t native_gpu_bootstrap_host_bytes = 0;
    uint64_t native_gpu_bootstrap_device_bytes = 0;
    uint64_t native_gpu_arena_bytes = 0;
    uint64_t native_gpu_device_span_bytes = 0;
    uint64_t native_gpu_dispatch_temporary_bytes = 0;
    uint64_t native_gpu_activation_peak_bytes = 0;
    uint64_t native_gpu_device_peak_bytes = 0;
    uint64_t native_gpu_fixed_allocation_bytes = 0;
    uint64_t native_gpu_span_count = 0;
    uint64_t native_gpu_qgroupnorm_stats_bytes = 0;
    uint64_t native_gpu_qlayernorm_stats_bytes = 0;
    uint64_t native_gpu_scratch_required_bytes = 0;
    uint64_t native_gpu_result_snapshot_bytes = 0;
    uint64_t native_gpu_result_publication_peak_bytes = 0;
    uint64_t native_gpu_dynamic_metadata_peak_bytes = 0;
    uint64_t native_gpu_backend_metadata_bytes = 0;
    uint64_t cuda_replay_signature_bytes = 0;
    char cuda_graph_resource_evidence[32] = {0};
    uint64_t maximum_launch_elements = 0;
    size_t maximum_typed_scratch = 0u;
    size_t kernel_pool_heap_bytes = 0u;
    size_t kernel_worker_count = 0u;
    size_t kernel_worker_stack_reservation_bytes = 0u;
    size_t operator_count = 0;
    size_t canonical_count = 0;
    size_t affine_relation_count = 0;
    VxCallSequenceResourceBounds call_sequence_resources = {0};
    int native_gpu_validation_bounds_ready = 0;
    int dynamic = 0;
    VxStatus status;
    VxNativeGpuDomainLimits native_gpu_limits = {0};
    if (!model || !weight_store || !validation || !resources || !evidence ||
        !evidence_capacity)
        return VX_STATUS_INVALID_ARGUMENT;
    maximum_weight_bytes = weight_store->raw_bytes;
    if (cpu_weights) {
        if (cpu_weights->materialized_bytes >
                UINT64_MAX - maximum_weight_bytes)
            return VX_STATUS_OUT_OF_MEMORY;
        maximum_weight_bytes += cpu_weights->materialized_bytes;
    }
    memset(resources, 0, sizeof(*resources));
    evidence[0] = '\0';
    uint64_t graph_metadata_bytes, graph_metadata_peak;
    if (vx_engine_state_graph_metadata_owned_bytes(validation,
            &graph_metadata_bytes, &graph_metadata_peak) ||
        graph_metadata_peak > UINT64_MAX - maximum_resident) {
        snprintf(evidence, evidence_capacity,
                 "required=graph-metadata-resource-bound");
        return VX_STATUS_BACKEND_UNSUPPORTED;
    }
    maximum_resident += graph_metadata_peak;
    digest = vx_revision_update(digest, &graph_metadata_peak, sizeof(graph_metadata_peak));
    /* Each context owns its capsule. The Runtime CPU domain owns one opaque
     * pool and maximum lazy worker population shared by its contexts; this
     * per-compiled envelope conservatively charges that domain once. The pool
     * reports the exact pthread stack/guard attributes used at creation. */
    if (vx_kernel_thread_pool_resource_bound(
            validation->kernel_thread_pool, &kernel_pool_heap_bytes,
            &kernel_worker_count,
            &kernel_worker_stack_reservation_bytes) != 0 ||
        kernel_pool_heap_bytes > SIZE_MAX -
            kernel_worker_stack_reservation_bytes ||
        (uint64_t)(kernel_pool_heap_bytes +
            kernel_worker_stack_reservation_bytes) >
                UINT64_MAX - maximum_resident) {
        snprintf(evidence, evidence_capacity,
                 "required=kernel-thread-pool-resource-bound");
        return VX_STATUS_BACKEND_UNSUPPORTED;
    }
    maximum_resident += (uint64_t)kernel_pool_heap_bytes +
        (uint64_t)kernel_worker_stack_reservation_bytes;
    digest = vx_revision_update(
        digest, &kernel_pool_heap_bytes, sizeof(kernel_pool_heap_bytes));
    digest = vx_revision_update(
        digest, &kernel_worker_count, sizeof(kernel_worker_count));
    digest = vx_revision_update(
        digest, &kernel_worker_stack_reservation_bytes,
        sizeof(kernel_worker_stack_reservation_bytes));
    registry_backend = backend == VX_PORTABLE_BACKEND_KIND ? "native-cpu" :
        backend == VX_BACKEND_KIND_VULKAN ? "vulkan" :
        backend == VX_BACKEND_KIND_OPENGL ? "opengl" :
        backend == VX_BACKEND_KIND_METAL ? "metal" :
        backend == VX_BACKEND_KIND_CUDA ? "cuda" :
        backend == VX_BACKEND_KIND_WEBGPU ? "webgpu" : NULL;
    if (!registry_backend) return VX_STATUS_BACKEND_UNSUPPORTED;
    status = vx_logical_tensors_load(model, &tensors, &tensor_count);
    if (status != VX_STATUS_OK) return status;
    for (size_t index = 0; index < tensor_count; index++)
        for (uint32_t axis = 0; axis < tensors[index].rank; axis++)
            dynamic |= vx_dimension_domain_is_dynamic(
                &tensors[index], axis);
    for (size_t index = 0; index < tensor_count; index++) {
        const VxDeclaredTensor* tensor = &tensors[index];
        uint64_t resident_charge = (uint64_t)tensor->maximum_byte_size;
        uint64_t maximum_elements = 0u;
        int tensor_dynamic = 0;
        if (dynamic && vx_native_gpu_backend(backend) &&
            tensor->dtype == VX_DTYPE_F16) {
            /* The native graph ABI has no F16 activation slots. Some legacy
             * copy-shaped routes widen their source to F32, so reserving the
             * declared two-byte domain would under-size the rebound output.
             * Immutable F16 model weights remain supported: they are outside
             * this closed logical activation table and preload as F32. */
            snprintf(evidence, evidence_capacity,
                     "unsupported_tensor=%s;"
                     "required=native-gpu-full-bounded-domain;"
                     "predicate_reason="
                     "native-gpu-f32-or-quantized-logical-domain",
                     tensor->name ? tensor->name : "<unknown>");
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        if (vx_native_gpu_backend(backend) &&
            (!vx_native_gpu_tensor_maximum_elements(tensor, &maximum_elements) ||
             !vx_native_gpu_tensor_size_domain_proven(
                 backend, tensor->dtype, tensor->rank, maximum_elements,
                 (uint64_t)tensor->maximum_byte_size))) {
            snprintf(evidence, evidence_capacity,
                     "unsupported_tensor=%s;"
                     "required=native-gpu-u32-tensor-domain",
                     tensor->name ? tensor->name : "<unknown>");
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        if (vx_native_gpu_backend(backend) &&
            tensor->dtype == VX_DTYPE_F16)
            resident_charge = maximum_elements * sizeof(float);
        if (maximum_elements > maximum_launch_elements)
            maximum_launch_elements = maximum_elements;
        for (uint32_t axis = 0; axis < tensor->rank; axis++)
            tensor_dynamic |= vx_dimension_domain_is_dynamic(tensor, axis);
        /* Native-GPU dynamic activation ownership is charged from the exact
         * maximum layout and the backend-specific bootstrap/candidate phases
         * after that layout has been built. */
        if (!dynamic || !vx_native_gpu_backend(backend)) {
            if (tensor_dynamic) {
                uint64_t copies = vx_native_gpu_backend(backend) &&
                    tensor->dtype == VX_DTYPE_F16 ? 4u : 3u;
                /* Widening an owned F16 bootstrap tensor keeps its retired
                 * F16 storage until teardown, in addition to the F32
                 * bootstrap, maximum host arena, and device span. Four
                 * F32-width copies cover that exceptional lifetime. */
                if (resident_charge > UINT64_MAX / copies) {
                    status = VX_STATUS_BACKEND_UNSUPPORTED;
                    goto done;
                }
                resident_charge *= copies;
            }
        } else {
            resident_charge = 0u;
        }
        if (resident_charge > UINT64_MAX - maximum_resident) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        maximum_resident += resident_charge;
        digest = vx_revision_update(digest, tensor->name,
                                    strlen(tensor->name) + 1u);
        digest = vx_revision_update(digest, &tensor->dtype,
                                    sizeof(tensor->dtype));
        digest = vx_revision_update(digest, &tensor->rank,
                                    sizeof(tensor->rank));
        for (uint32_t axis = 0; axis < tensor->rank; axis++) {
            if (tensor->maximums[axis] > INT_MAX) {
                status = VX_STATUS_BACKEND_UNSUPPORTED;
                goto done;
            }
            digest = vx_revision_update(digest, &tensor->kinds[axis],
                                        sizeof(tensor->kinds[axis]));
            digest = vx_revision_update(digest, &tensor->minimums[axis],
                                        sizeof(tensor->minimums[axis]));
            digest = vx_revision_update(digest, &tensor->maximums[axis],
                                        sizeof(tensor->maximums[axis]));
            digest = vx_revision_update(digest, &tensor->multiples[axis],
                                        sizeof(tensor->multiples[axis]));
            if (tensor->symbols[axis])
                digest = vx_revision_update(
                    digest, tensor->symbols[axis],
                    strlen(tensor->symbols[axis]) + 1u);
        }
    }
    if (vx_native_gpu_backend(backend)) {
        if (!vx_native_gpu_validation_immutable_bytes_bound(
                validation, tensors, tensor_count,
                backend,
                1u, &native_gpu_immutable_bytes,
                &native_gpu_device_immutable_bytes)) {
            snprintf(evidence, evidence_capacity,
                     "required=native-gpu-immutable-byte-bound");
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        /* The mapped host revision remains alive while the native GPU owns
         * converted, packed, and aligned immutable copies. They are simultaneous
         * resources, so max(file, device) would understate the context. */
        if (native_gpu_immutable_bytes >
            UINT64_MAX - maximum_weight_bytes) {
            snprintf(evidence, evidence_capacity,
                     "required=native-gpu-immutable-byte-sum");
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        maximum_weight_bytes += native_gpu_immutable_bytes;
        native_gpu_validation_bounds_ready = 1;
    }
    if (maximum_weight_bytes > UINT64_MAX - maximum_resident) {
        snprintf(evidence, evidence_capacity,
                 "required=resident-byte-sum");
        status = VX_STATUS_BACKEND_UNSUPPORTED;
        goto done;
    }
    maximum_resident += maximum_weight_bytes;
    if (dynamic && backend != VX_PORTABLE_BACKEND_KIND &&
        backend != VX_BACKEND_KIND_WEBGPU &&
        !vx_native_gpu_backend(backend)) {
        snprintf(evidence, evidence_capacity,
                 "required=dynamic-domain-on-non-gpu-backend");
        status = VX_STATUS_BACKEND_UNSUPPORTED;
        goto done;
    }
    /* Value-origin and exact launch predicates are safety contracts for every
     * native-GPU graph, not only graphs whose logical extents vary.  Dynamic
     * graphs query these limits as part of maximum-layout construction below;
     * fixed graphs still need the same device limits before per-node proof. */
    if (!dynamic && vx_native_gpu_backend(backend)) {
        int limits_query = -1;
        if (validation && native_gpu_validation_bounds_ready) {
            /* The device limits live on the engine state, so the query only
             * sees an initialized device while that state is current. The
             * dynamic path enters this scope before its own query; the static
             * path used to call straight through and always read an
             * uninitialized device, which refused every fully static graph on
             * every native GPU backend. */
            VxEngineStateScope limits_scope =
                vx_engine_state_scope_enter(validation);
            limits_query = vx_native_gpu_query_domain_limits(
                backend, &native_gpu_limits);
            vx_engine_state_scope_leave(limits_scope);
        }
        if (limits_query != 0) {
            snprintf(evidence, evidence_capacity,
                     "required=native-gpu-static-domain-limits;"
                     "predicate_reason=validation=%d,bounds_ready=%d,limits_query=%d",
                     validation ? 1 : 0, native_gpu_validation_bounds_ready,
                     limits_query);
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        digest = vx_revision_update(
            digest, &native_gpu_limits, sizeof(native_gpu_limits));
    }
    if (dynamic && vx_native_gpu_backend(backend)) {
        int proof_status;
        uint64_t aligned_immutable_bytes = 0u;
        uint64_t aligned_device_immutable_bytes = 0u;
        uint64_t immutable_alignment_delta = 0u;
        if (!validation || !native_gpu_validation_bounds_ready) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        proof_status = vx_native_gpu_maximum_layout_proof(
            validation, tensors, tensor_count, backend, &native_gpu_limits,
            &native_gpu_arena_bytes, &native_gpu_device_span_bytes,
            &native_gpu_span_count,
            &native_gpu_qgroupnorm_stats_bytes,
            &native_gpu_qlayernorm_stats_bytes,
            &native_gpu_scratch_required_bytes);
        if (proof_status != 0) {
            snprintf(
                evidence, evidence_capacity,
                "required=native-gpu-device-domain-capacity;"
                "backend=%s;layout_status=%d;arena_bytes=%" PRIu64
                ";immutable_bytes=%" PRIu64,
                registry_backend, proof_status,
                native_gpu_arena_bytes, native_gpu_immutable_bytes);
            status = proof_status == -2
                ? VX_STATUS_OUT_OF_MEMORY : VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        if (!vx_native_gpu_validation_immutable_bytes_bound(
                validation, tensors, tensor_count,
                backend,
                native_gpu_limits.storage_alignment,
                &aligned_immutable_bytes,
                &aligned_device_immutable_bytes) ||
            aligned_immutable_bytes < native_gpu_immutable_bytes ||
            aligned_device_immutable_bytes <
                native_gpu_device_immutable_bytes) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        immutable_alignment_delta =
            aligned_immutable_bytes - native_gpu_immutable_bytes;
        if (immutable_alignment_delta > UINT64_MAX - maximum_weight_bytes ||
            immutable_alignment_delta > UINT64_MAX - maximum_resident) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        maximum_weight_bytes += immutable_alignment_delta;
        maximum_resident += immutable_alignment_delta;
        native_gpu_immutable_bytes = aligned_immutable_bytes;
        native_gpu_device_immutable_bytes =
            aligned_device_immutable_bytes;
        if (!vx_native_gpu_validation_bootstrap_activation_bound(
                validation, tensors, tensor_count,
                native_gpu_limits.storage_alignment,
                &native_gpu_bootstrap_host_bytes,
                &native_gpu_bootstrap_device_bytes) ||
            !vx_native_gpu_validation_dispatch_temporary_bound(
                validation, tensors, tensor_count, backend,
                native_gpu_limits.storage_alignment,
                &native_gpu_dispatch_temporary_bytes) ||
            !vx_native_gpu_activation_resource_peak(
                backend, native_gpu_bootstrap_host_bytes,
                native_gpu_arena_bytes,
                native_gpu_bootstrap_device_bytes,
                native_gpu_device_span_bytes,
                native_gpu_qgroupnorm_stats_bytes,
                native_gpu_qlayernorm_stats_bytes,
                native_gpu_dispatch_temporary_bytes,
                native_gpu_scratch_required_bytes,
                &native_gpu_activation_peak_bytes) ||
            native_gpu_activation_peak_bytes >
                UINT64_MAX - maximum_resident) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        /* maximum_total_span_bytes covers the graph arena shared by logical
         * activations and resident weights. Each Vulkan suballocation is
         * rounded to the queried device alignment before this comparison.
         * CUDA frees bootstrap transients before reservation; Vulkan returns
         * their ranges to the same arena, so the larger phase is sufficient.
         * Quantized-norm statistics live in separately proved scratch/buffers
         * and are not part of that address range. */
        if (!vx_native_gpu_device_capacity_proven(
                native_gpu_device_immutable_bytes,
                native_gpu_bootstrap_device_bytes,
                native_gpu_device_span_bytes,
                native_gpu_limits.maximum_total_span_bytes,
                &native_gpu_device_peak_bytes)) {
            snprintf(
                evidence, evidence_capacity,
                "required=native-gpu-device-domain-capacity;"
                "backend=%s;device_bytes=%" PRIu64
                ";device_limit=%" PRIu64,
                registry_backend,
                native_gpu_device_peak_bytes,
                native_gpu_limits.maximum_total_span_bytes);
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        digest = vx_revision_update(
            digest, &native_gpu_limits, sizeof(native_gpu_limits));
        digest = vx_revision_update(
            digest, &native_gpu_arena_bytes, sizeof(native_gpu_arena_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_device_span_bytes,
            sizeof(native_gpu_device_span_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_bootstrap_host_bytes,
            sizeof(native_gpu_bootstrap_host_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_bootstrap_device_bytes,
            sizeof(native_gpu_bootstrap_device_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_dispatch_temporary_bytes,
            sizeof(native_gpu_dispatch_temporary_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_span_count, sizeof(native_gpu_span_count));
        digest = vx_revision_update(
            digest, &native_gpu_qgroupnorm_stats_bytes,
            sizeof(native_gpu_qgroupnorm_stats_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_qlayernorm_stats_bytes,
            sizeof(native_gpu_qlayernorm_stats_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_scratch_required_bytes,
            sizeof(native_gpu_scratch_required_bytes));
        maximum_resident += native_gpu_activation_peak_bytes;
    }
    if (backend == VX_BACKEND_KIND_VULKAN) {
        if (!vx_native_gpu_fixed_allocation_bound(
                native_gpu_limits.fixed_compute_allocation_bytes,
                native_gpu_limits.fixed_staging_allocation_bytes,
                &native_gpu_fixed_allocation_bytes) ||
            native_gpu_fixed_allocation_bytes >
                UINT64_MAX - maximum_resident) {
            snprintf(
                evidence, evidence_capacity,
                "required=native-gpu-fixed-allocation-capacity;"
                "backend=%s;compute_bytes=%" PRIu64
                ";staging_bytes=%" PRIu64,
                registry_backend,
                native_gpu_limits.fixed_compute_allocation_bytes,
                native_gpu_limits.fixed_staging_allocation_bytes);
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        maximum_resident += native_gpu_fixed_allocation_bytes;
        digest = vx_revision_update(
            digest, &native_gpu_fixed_allocation_bytes,
            sizeof(native_gpu_fixed_allocation_bytes));
    }
    if (vx_native_gpu_backend(backend)) {
        uint64_t proved_resident = 0u;
        if (!vx_native_gpu_result_publication_peak(
                model, &native_gpu_result_snapshot_bytes,
                &native_gpu_result_publication_peak_bytes) ||
            (dynamic && !vx_native_gpu_dynamic_metadata_peak(
                model, (uint64_t)tensor_count,
                validation ? (uint64_t)validation->tensor_count : 0u,
                native_gpu_span_count,
                &native_gpu_dynamic_metadata_peak_bytes))) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        if (dynamic && backend == VX_BACKEND_KIND_CUDA) {
            if (!native_gpu_limits.replay_plan_capacity ||
                !native_gpu_limits.replay_fixed_host_metadata_bytes ||
                !vx_native_gpu_maximum_signature_bytes(
                    model, &cuda_replay_signature_bytes) ||
                !vx_native_gpu_cuda_replay_metadata_peak(
                    native_gpu_limits.replay_fixed_host_metadata_bytes,
                    cuda_replay_signature_bytes,
                    native_gpu_limits.replay_plan_capacity,
                    &native_gpu_backend_metadata_bytes) ||
                !vx_native_gpu_accumulate_resource_bytes(
                    &native_gpu_dynamic_metadata_peak_bytes,
                    native_gpu_backend_metadata_bytes, 1u)) {
                status = VX_STATUS_BACKEND_UNSUPPORTED;
                goto done;
            }
            {
                int length = snprintf(
                    cuda_graph_resource_evidence,
                    sizeof(cuda_graph_resource_evidence),
                    ";cuda_graph_cache=%" PRIu32,
                    native_gpu_limits.replay_plan_capacity);
                if (length < 0 ||
                    (size_t)length >= sizeof(cuda_graph_resource_evidence)) {
                    status = VX_STATUS_BACKEND_UNSUPPORTED;
                    goto done;
                }
            }
        }
        if (
            !vx_native_gpu_resident_capacity_proven(
                maximum_resident,
                native_gpu_result_publication_peak_bytes,
                native_gpu_dynamic_metadata_peak_bytes,
                (uint64_t)SIZE_MAX, &proved_resident)) {
            snprintf(
                evidence, evidence_capacity,
                "required=native-gpu-resident-capacity;"
                "backend=%s;resident_base=%" PRIu64
                ";result_peak=%" PRIu64 ";plan_metadata_peak=%" PRIu64
                ";resident_limit=%zu",
                registry_backend,
                maximum_resident,
                native_gpu_result_publication_peak_bytes,
                native_gpu_dynamic_metadata_peak_bytes, SIZE_MAX);
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        maximum_resident = proved_resident;
        digest = vx_revision_update(
            digest, &native_gpu_result_snapshot_bytes,
            sizeof(native_gpu_result_snapshot_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_result_publication_peak_bytes,
            sizeof(native_gpu_result_publication_peak_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_dynamic_metadata_peak_bytes,
            sizeof(native_gpu_dynamic_metadata_peak_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_backend_metadata_bytes,
            sizeof(native_gpu_backend_metadata_bytes));
    }
    if (dynamic && backend == VX_BACKEND_KIND_CUDA) {
        uint64_t one_d_grid;
        one_d_grid = maximum_launch_elements / 256u +
            (maximum_launch_elements % 256u != 0u);
        if (native_gpu_limits.maximum_block[0] < 256u ||
            native_gpu_limits.maximum_threads_per_block < 256u ||
            one_d_grid > native_gpu_limits.maximum_grid[0]) {
            snprintf(
                evidence, evidence_capacity,
                "required=cuda-device-domain-capacity;"
                "launch_elements=%" PRIu64 ";grid_x=%" PRIu32,
                maximum_launch_elements,
                native_gpu_limits.maximum_grid[0]);
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
    }
    root = model->logical_graph;
    if (!root) {
        status = VX_STATUS_IO_ERROR;
        goto done;
    }
    nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    if (model->graph_domain_terminal_kind !=
            VX_MODEL_GRAPH_DOMAIN_TERMINAL_ACCEPTED ||
        model->graph_domain_status_v1 != VX_GRAPH_DOMAIN_STATUS_OK ||
        !model->graph_domain_v1 ||
        model->graph_domain_v1_bytes < sizeof(VxGraphDomainResponseV1)) {
        status = VX_STATUS_INTERNAL;
        goto done;
    }
    affine_relation_count = (size_t)vx_graph_plan_load_u32(
        model->graph_domain_v1 + 96u);
    for (const cJSON* node = cJSON_IsArray(nodes) ? nodes->child : NULL;
         node; node = node->next) {
        const cJSON* op = cJSON_GetObjectItemCaseSensitive(node, "opType");
        const cJSON* node_id = cJSON_GetObjectItemCaseSensitive(node, "id");
        const VxGeneratedShapeContractRoute* shape_route;
        const VxKernelRegistration* backend_route;
        if (!cJSON_IsString(op) || !op->valuestring ||
            !(shape_route = vx_kernel_shape_contract_find(vx_operator_kind_from_name(op->valuestring))) ||
            !(backend_route = vx_kernel_registry_find(
                  backend == VX_PORTABLE_BACKEND_KIND ? VX_BACKEND_KIND_NATIVE_CPU : backend,
                  shape_route->operator_kind)) ||
            !shape_route->shape_function_id ||
            !shape_route->shape_function_id[0] ||
            !backend_route->route || !backend_route->route[0] ||
            !backend_route->predicate_id || !backend_route->predicate_id[0] ||
            backend_route->operator_kind != shape_route->operator_kind) {
            snprintf(evidence, evidence_capacity,
                     "unsupported_node=%s;unsupported_op=%s;"
                     "required=registered-canonical-backend-route",
                     cJSON_IsString(node_id) ? node_id->valuestring : "<unknown>",
                     cJSON_IsString(op) ? op->valuestring : "<unknown>");
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        if (dynamic && !backend_route->dynamic) {
            snprintf(
                evidence, evidence_capacity,
                "unsupported_node=%s;unsupported_op=%s;"
                "required=dynamic-native-route;reexport=required",
                cJSON_IsString(node_id) && node_id->valuestring
                    ? node_id->valuestring : "<unknown>",
                op->valuestring);
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        /* Exporter qualification describes an exporter target's tested
         * conversion subset. Runtime admission uses the canonical contract
         * and this backend's complete physical proof, independently. */
        if (vx_native_gpu_backend(backend)) {
            status = vx_native_gpu_node_domain_proof(
                node, tensors, tensor_count, validation, backend,
                &native_gpu_limits,
                evidence, evidence_capacity);
            if (status != VX_STATUS_OK) goto done;
        }
#if VOLVOXAI_ENABLE_WEBGPU
        if (backend == VX_BACKEND_KIND_WEBGPU) {
            if (!vx_webgpu_node_domain_proven(node, tensors, tensor_count, validation)) {
                snprintf(evidence, evidence_capacity,
                         "unsupported_op=%s;required=webgpu-shape-value-and-physical-domain",
                         op->valuestring);
                status = VX_STATUS_BACKEND_UNSUPPORTED;
                goto done;
            }
        }
#endif
        digest = vx_revision_update(digest, shape_route->shape_function_id,
                                    strlen(shape_route->shape_function_id) + 1u);
        digest = vx_revision_update(digest, &shape_route->classification,
                                    sizeof(shape_route->classification));
        digest = vx_revision_update(digest, backend_route->route,
                                    strlen(backend_route->route) + 1u);
        digest = vx_revision_update(digest, backend_route->predicate_id,
                                    strlen(backend_route->predicate_id) + 1u);
        if (backend == VX_PORTABLE_BACKEND_KIND &&
            shape_route->operator_kind == VX_OP_Q_BATCH_MATMUL) {
            const size_t node_workspace =
                vx_qbatch_cpu_workspace_domain_bound(
                    node, tensors, tensor_count);
            if (node_workspace > maximum_typed_scratch)
                maximum_typed_scratch = node_workspace;
            digest = vx_revision_update(
                digest, &node_workspace, sizeof(node_workspace));
        }
        canonical_count++;
        operator_count++;
    }
    if (backend == VX_PORTABLE_BACKEND_KIND) {
        static const char workspace_contract[] =
            "native.cpu.typed-workspace.qbatch-packed/v2";
        if ((uint64_t)maximum_typed_scratch >
            UINT64_MAX - maximum_resident) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        maximum_resident += (uint64_t)maximum_typed_scratch;
        digest = vx_revision_update(
            digest, workspace_contract, sizeof(workspace_contract));
        digest = vx_revision_update(
            digest, &maximum_typed_scratch, sizeof(maximum_typed_scratch));
    }
    if (backend == VX_PORTABLE_BACKEND_KIND) {
        static const char call_sequence_resource_contract[] =
            "native.cpu.call-sequence-resources/v1";
        const uint64_t resident_without_call_sequence = maximum_resident;
        uint64_t forward_resident;
        uint64_t sequence_resident = 0u;
        if (!vx_call_sequence_resource_bounds_from_graph_plan(
                model->graph_plan_request_v1,
                model->graph_plan_request_v1_bytes,
                model->graph_plan_v1,
                model->graph_plan_v1_bytes,
                model->input_count,
                &call_sequence_resources)) {
            snprintf(evidence, evidence_capacity,
                     "required=call-sequence-graph-plan-scalar-evidence");
            status = VX_STATUS_INTERNAL;
            goto done;
        }
        /* Forward-only is an unconditional CPU context requirement. Decode
         * has an independent admission flag below: an oversized maximum
         * changed-input transaction must not invalidate ordinary forward. */
        if (!call_sequence_resources.forward_supported ||
            resident_without_call_sequence > (uint64_t)SIZE_MAX ||
            call_sequence_resources.forward.concurrent_bytes >
                UINT64_MAX - resident_without_call_sequence ||
            call_sequence_resources.forward.concurrent_bytes >
                (uint64_t)SIZE_MAX - resident_without_call_sequence) {
            snprintf(
                evidence, evidence_capacity,
                "required=call-sequence-forward-resource-bound;"
                "csf_qps=%" PRIu64 "/%" PRIu64 "/%" PRIu64
                ";csf_rtc=%" PRIu64 "/%" PRIu64 "/%" PRIu64,
                call_sequence_resources.forward.request_capacity_bytes,
                call_sequence_resources.forward.response_capacity_bytes,
                call_sequence_resources.forward.scratch_capacity_bytes,
                call_sequence_resources.forward.retained_bytes,
                call_sequence_resources.forward.transaction_bytes,
                call_sequence_resources.forward.concurrent_bytes);
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        forward_resident = resident_without_call_sequence +
            call_sequence_resources.forward.concurrent_bytes;
        if (call_sequence_resources.sequence_supported) {
            if (call_sequence_resources.sequence.concurrent_bytes >
                    UINT64_MAX - resident_without_call_sequence ||
                call_sequence_resources.sequence.concurrent_bytes >
                    (uint64_t)SIZE_MAX - resident_without_call_sequence) {
                call_sequence_resources.sequence_supported = 0u;
            } else {
                sequence_resident = resident_without_call_sequence +
                    call_sequence_resources.sequence.concurrent_bytes;
            }
        }
        maximum_resident = call_sequence_resources.sequence_supported &&
                sequence_resident > forward_resident
            ? sequence_resident : forward_resident;
        digest = vx_revision_update(
            digest, call_sequence_resource_contract,
            sizeof(call_sequence_resource_contract));
        digest = vx_revision_update(
            digest, &call_sequence_resources.forward,
            sizeof(call_sequence_resources.forward));
        digest = vx_revision_update(
            digest, &call_sequence_resources.sequence,
            sizeof(call_sequence_resources.sequence));
        digest = vx_revision_update(
            digest, &call_sequence_resources.forward_supported,
            sizeof(call_sequence_resources.forward_supported));
        digest = vx_revision_update(
            digest, &call_sequence_resources.sequence_supported,
            sizeof(call_sequence_resources.sequence_supported));
        digest = vx_revision_update(
            digest, &maximum_resident, sizeof(maximum_resident));
    }
    if (vx_native_gpu_backend(backend)) {
        /* The public route-evidence ABI is fixed at 512 bytes. Backend and
         * capacity fields are more actionable here than repeating the equal
         * canonical/native-route counts already implied by domain_route=all.
         * Keep every attested field in one bounded record; ordering the
         * dispatch, result, and plan peaks before the digest prevents silent
         * truncation when the built-in route prefix is prepended. */
        snprintf(evidence, evidence_capacity,
                 "shape_proof=%s;domain_route=all;dynamic=%d;"
                 "native_gpu_backend=%s%s"
                ";native_gpu_fixed_alloc=%" PRIu64 "/%" PRIu64
                ";native_gpu_domain_spans=%" PRIu64
                ";native_gpu_domain_bytes=%" PRIu64
                ";native_gpu_storage_alignment=%" PRIu64
                ";native_gpu_fixed_scratch=%" PRIu64 "/%" PRIu64
                "/%" PRIu64
                 ";native_gpu_result_peak=%" PRIu64
                 ";native_gpu_plan_metadata_peak=%" PRIu64
                 ";max_resident=%" PRIu64
                 ";domain_digest=%016" PRIx64,
                model->shape_domain_proof_identity, dynamic,
                registry_backend,
                cuda_graph_resource_evidence,
                native_gpu_limits.fixed_compute_allocation_bytes,
                native_gpu_limits.fixed_staging_allocation_bytes,
                native_gpu_span_count,
                native_gpu_arena_bytes,
                native_gpu_limits.storage_alignment,
                native_gpu_scratch_required_bytes,
                native_gpu_limits.current_graph_scratch_bytes,
                native_gpu_limits.maximum_scratch_bytes,
                native_gpu_result_publication_peak_bytes,
                native_gpu_dynamic_metadata_peak_bytes,
                maximum_resident,
                digest);
    } else {
        snprintf(evidence, evidence_capacity,
                 "csf_qps=%" PRIu64 "/%" PRIu64 "/%" PRIu64
                 ";csf_rtc=%" PRIu64 "/%" PRIu64 "/%" PRIu64
                 ";css_qps=%" PRIu64 "/%" PRIu64 "/%" PRIu64
                 ";css_rtc=%" PRIu64 "/%" PRIu64 "/%" PRIu64
                 ";css_ok=%" PRIu32
                 ";max_resident=%" PRIu64
                 ";domain_digest=%016" PRIx64 ";"
                 "shape_proof=%s;domain_route=all;dynamic=%d;"
                 "operators=%zu;canonical=%zu;native_routes=%zu;"
                 "affine_relations=%zu;"
                 "max_tensor=%" PRIu64 ";max_weights=%" PRIu64
                 ";thread_pool=%zu/%zu/%zu;max_typed_scratch=%zu",
                 call_sequence_resources.forward.request_capacity_bytes,
                 call_sequence_resources.forward.response_capacity_bytes,
                 call_sequence_resources.forward.scratch_capacity_bytes,
                 call_sequence_resources.forward.retained_bytes,
                 call_sequence_resources.forward.transaction_bytes,
                 call_sequence_resources.forward.concurrent_bytes,
                 call_sequence_resources.sequence.request_capacity_bytes,
                 call_sequence_resources.sequence.response_capacity_bytes,
                 call_sequence_resources.sequence.scratch_capacity_bytes,
                 call_sequence_resources.sequence.retained_bytes,
                 call_sequence_resources.sequence.transaction_bytes,
                 call_sequence_resources.sequence.concurrent_bytes,
                 call_sequence_resources.sequence_supported,
                 maximum_resident, digest,
                 model->shape_domain_proof_identity, dynamic,
                 operator_count, canonical_count, operator_count,
                 affine_relation_count, model->maximum_tensor_bytes,
                 maximum_weight_bytes, kernel_pool_heap_bytes,
                 kernel_worker_count,
                 kernel_worker_stack_reservation_bytes,
                 maximum_typed_scratch);
    }
    resources->maximum_typed_scratch_bytes = maximum_typed_scratch;
    resources->maximum_resident_bytes = maximum_resident;
    resources->call_sequence_resources = call_sequence_resources;
    status = VX_STATUS_OK;
done:
    vx_declared_outputs_free(tensors, tensor_count);
    return status;
}

typedef struct VxShapeParamStorage {
    VxShapeParam* values;
    double** arrays;
    size_t count;
} VxShapeParamStorage;
