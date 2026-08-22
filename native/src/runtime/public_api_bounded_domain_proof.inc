/* vx_builtin_bounded_domain_proof(): the whole-graph declared-domain proof.
 *
 * Fragment of public_api.c, included where this definition used to sit.  This
 * is the compile-time gate that must prove every legal binding of the declared
 * domain before a model is accepted; it drives the per-node checks in
 * public_api_native_gpu_domain.inc for GPU backends. */

static int vx_native_gpu_tensor_size_domain_proven(
        VolvoxAIEngineBackend backend,
        VxDataType dtype,
        uint32_t rank,
        uint64_t maximum_elements,
        uint64_t maximum_byte_size) {
    if (!vx_native_gpu_backend(backend) || rank > 8u ||
        maximum_elements > UINT32_MAX ||
        maximum_byte_size > (uint64_t)SIZE_MAX ||
        (backend != VOLVOXAI_BACKEND_CUDA &&
         maximum_byte_size > UINT32_MAX) ||
        (dtype == VX_DTYPE_F16 &&
         maximum_elements > UINT32_MAX / sizeof(float)))
        return 0;
    return 1;
}

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
int vx_public_api_test_native_gpu_tensor_size_domain_proven(
        VolvoxAIEngineBackend backend,
        VxDataType dtype,
        uint32_t rank,
        uint64_t maximum_elements,
        uint64_t maximum_byte_size) {
    return vx_native_gpu_tensor_size_domain_proven(
        backend, dtype, rank, maximum_elements, maximum_byte_size);
}
#endif

static VxStatus vx_builtin_bounded_domain_proof(
        const VxModel* model,
        const VxWeightRevisionRecord* weights,
        VolvoxAIEngineBackend backend,
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
    uint64_t maximum_resident = 0;
    uint64_t maximum_weight_bytes = 0;
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
    size_t operator_count = 0;
    size_t canonical_count = 0;
    size_t affine_relation_count = 0;
    int native_gpu_validation_bounds_ready = 0;
    int dynamic = 0;
    VxStatus status;
    VxNativeGpuDomainLimits native_gpu_limits = {0};
    if (!model || !resources || !evidence || !evidence_capacity)
        return VX_STATUS_INVALID_ARGUMENT;
    memset(resources, 0, sizeof(*resources));
    evidence[0] = '\0';
    registry_backend = backend == VOLVOXAI_BACKEND_CPU ? "native-cpu" :
        backend == VOLVOXAI_BACKEND_VULKAN ? "vulkan" :
        backend == VOLVOXAI_BACKEND_OPENGL ? "opengl" :
        backend == VOLVOXAI_BACKEND_METAL ? "metal" :
        backend == VOLVOXAI_BACKEND_CUDA ? "cuda" : NULL;
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
    for (size_t index = 0; weights && index < weights->path_count; index++) {
        FILE* file = fopen(weights->paths[index], "rb");
        long size;
        if (!file) {
            status = VX_STATUS_IO_ERROR;
            goto done;
        }
        if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0 ||
            (uint64_t)size > UINT64_MAX - maximum_weight_bytes) {
            (void)fclose(file);
            status = VX_STATUS_IO_ERROR;
            goto done;
        }
        if (fclose(file) != 0) {
            status = VX_STATUS_IO_ERROR;
            goto done;
        }
        maximum_weight_bytes += (uint64_t)size;
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
    if (dynamic && backend != VOLVOXAI_BACKEND_CPU &&
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
    if (backend == VOLVOXAI_BACKEND_VULKAN) {
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
        if (dynamic && backend == VOLVOXAI_BACKEND_CUDA) {
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
    if (dynamic && backend == VOLVOXAI_BACKEND_CUDA) {
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
    status = vx_graph_affine_concat_domain_proof(
        model, tensors, tensor_count, &digest, &affine_relation_count,
        evidence, evidence_capacity);
    if (status != VX_STATUS_OK) goto done;
    for (const cJSON* node = cJSON_IsArray(nodes) ? nodes->child : NULL;
         node; node = node->next) {
        const cJSON* op = cJSON_GetObjectItemCaseSensitive(node, "opType");
        const cJSON* node_id = cJSON_GetObjectItemCaseSensitive(node, "id");
        const VxGeneratedShapeContractRoute* shape_route;
        const VxKernelRegistration* backend_route;
        const char* implemented_shape_function;
        if (!cJSON_IsString(op) || !op->valuestring ||
            !(shape_route = vx_kernel_shape_contract_find(op->valuestring)) ||
            !(backend_route = vx_kernel_registry_find(
                  registry_backend, op->valuestring)) ||
            !shape_route->shape_function_id ||
            !shape_route->shape_function_id[0] ||
            !backend_route->route || !backend_route->route[0] ||
            !backend_route->predicate_id || !backend_route->predicate_id[0] ||
            backend_route->operator_kind != shape_route->operator_kind) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        implemented_shape_function =
            vx_shape_contract_function_id(op->valuestring);
        if (shape_route->classification !=
                VX_SHAPE_CONTRACT_CLASSIFICATION_CANONICAL ||
            !implemented_shape_function ||
            strcmp(implemented_shape_function,
                   shape_route->shape_function_id) ||
            !backend_route->exporter_qualified) {
            snprintf(
                evidence, evidence_capacity,
                "unsupported_node=%s;unsupported_op=%s;"
                "required=canonical-native-shape-contract;reexport=required",
                cJSON_IsString(node_id) && node_id->valuestring
                    ? node_id->valuestring : "<unknown>",
                op->valuestring);
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
        if (vx_native_gpu_backend(backend)) {
            status = vx_native_gpu_node_domain_proof(
                node, tensors, tensor_count, validation, backend,
                &native_gpu_limits,
                evidence, evidence_capacity);
            if (status != VX_STATUS_OK) goto done;
        }
        digest = vx_revision_update(digest, shape_route->shape_function_id,
                                    strlen(shape_route->shape_function_id) + 1u);
        digest = vx_revision_update(digest, &shape_route->classification,
                                    sizeof(shape_route->classification));
        digest = vx_revision_update(digest, backend_route->route,
                                    strlen(backend_route->route) + 1u);
        digest = vx_revision_update(digest, backend_route->predicate_id,
                                    strlen(backend_route->predicate_id) + 1u);
        if (backend == VOLVOXAI_BACKEND_CPU &&
            !strcmp(op->valuestring, "QBatchMatMul")) {
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
    if (backend == VOLVOXAI_BACKEND_CPU) {
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
                 "shape_proof=%s;domain_route=all;dynamic=%d;"
                 "operators=%zu;canonical=%zu;native_routes=%zu;"
                 "affine_relations=%zu;"
                 "max_tensor=%" PRIu64 ";max_weights=%" PRIu64
                 ";max_typed_scratch=%zu;max_resident=%" PRIu64
                 ";domain_digest=%016" PRIx64,
                 model->shape_domain_proof_identity, dynamic,
                 operator_count, canonical_count, operator_count,
                 affine_relation_count, model->maximum_tensor_bytes,
                 maximum_weight_bytes, maximum_typed_scratch,
                 maximum_resident, digest);
    }
    resources->maximum_typed_scratch_bytes = maximum_typed_scratch;
    resources->maximum_resident_bytes = maximum_resident;
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
