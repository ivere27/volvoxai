/* Execution-context creation, input binding and pre-dispatch proofs.
 *
 * Fragment of public_api.c, included where these definitions used to sit.
 * Covers context open/close, adapter and native-GPU dynamic reservation,
 * declared-input specs, staged binding batches, and the native-GPU preflight
 * value-source checks that must pass before any dispatch. */

static VxStatus vx_context_operation_begin(VxExecutionContext* context,
                                           VxContextOperation* operation,
                                           VxStage stage,
                                           VxReport* report) {
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!context || !operation)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, stage, NULL,
                       VX_CODE_INVALID_CONTEXT, "execution context is NULL");
    memset(operation, 0, sizeof(*operation));
    pthread_mutex_lock(&context->queue_mutex);
    if (context->closing || context->closed) {
        VxStatus closed_status;
        pthread_mutex_unlock(&context->queue_mutex);
        closed_status = vx_fail(report, VX_STATUS_HANDLE_DISPOSED, stage,
                                context->compiled->backend,
                                VX_CODE_HANDLE_DISPOSED,
                                "execution context is closing or closed");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        return closed_status;
    }
#if defined(VOLVOXAI_NO_THREADS)
    if (context->next_ticket != context->serving_ticket) {
        pthread_mutex_unlock(&context->queue_mutex);
        return vx_fail(report, VX_STATUS_BUSY, stage,
                       context->compiled->backend,
                       VX_CODE_BUSY, "execution context re-entrant invocation");
    }
#endif
    operation->ticket = context->next_ticket++;
    operation->accepted = 1;
#if !defined(VOLVOXAI_NO_THREADS)
    while (operation->ticket != context->serving_ticket)
        pthread_cond_wait(&context->queue_condition, &context->queue_mutex);
#endif
    pthread_mutex_unlock(&context->queue_mutex);
    return VX_STATUS_OK;
}

static void vx_context_operation_end(VxExecutionContext* context,
                                     VxContextOperation* operation) {
    if (!context || !operation || !operation->accepted) return;
    pthread_mutex_lock(&context->queue_mutex);
    context->serving_ticket++;
    pthread_cond_broadcast(&context->queue_condition);
    pthread_mutex_unlock(&context->queue_mutex);
    operation->accepted = 0;
}

static size_t vx_model_input_index(const VxModel* model, const char* name);

static int vx_context_options_valid(const VxContextOptions* options) {
    return options && options->struct_size == sizeof(*options) &&
        (options->decode_row_mode == VX_DECODE_ROW_DISABLED ||
         options->decode_row_mode == VX_DECODE_ROW_AUTO ||
         options->decode_row_mode == VX_DECODE_ROW_REQUIRED) &&
        (options->require_incremental == 0 ||
         options->require_incremental == 1) &&
        options->decode_lanes >= 1 &&
        options->decode_lanes <= INT32_MAX &&
        (!options->decode_input_count || (options->decode_inputs && options->decode_row_mode != VX_DECODE_ROW_DISABLED)) &&
        (options->decode_lanes == 1 || options->decode_row_mode != VX_DECODE_ROW_DISABLED) &&
        (options->decode_row_mode != VX_DECODE_ROW_DISABLED ||
         !options->require_incremental);
}

static const char* vx_decode_mode_name(VolvoxAIDecodeMode mode) {
    switch (mode) {
        case VOLVOXAI_DECODE_MODE_ORDINARY_FORWARD: return "ordinary";
        case VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY: return "dependency";
        case VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW: return "row";
        default: return "none";
    }
}

static void vx_report_write_decode_state(VxReport* report,
                                         VxExecutionContext* context,
                                         int prefilled,
                                         const char* mode,
                                         const char* last) {
    if (context->decode_has_successful_step && context->decode_lanes == 1) {
        snprintf(report->decode_state, sizeof(report->decode_state),
                 "enabled=1;prefilled=%d;mode=%s;last=%s;position=%" PRId32,
                 prefilled, mode, last, context->decode_last_position);
    } else {
        snprintf(report->decode_state, sizeof(report->decode_state),
                 "enabled=1;prefilled=%d;mode=%s;last=%s",
                 prefilled, mode, last);
    }
}

static void vx_report_set_builtin_decode_state(VxReport* report,
                                               VxExecutionContext* context) {
    VolvoxAIDecodeMode mode;
    VolvoxAIDecodeMode last;
    int prefilled;
    if (!report || report->struct_size != sizeof(*report) || !context ||
        !context->decode_session) return;
    mode = volvoxai_engine_decode_session_mode(context->decode_session);
    last = volvoxai_engine_decode_session_last_execution_mode(
        context->decode_session);
    prefilled = volvoxai_engine_decode_session_prefilled(context->decode_session);
    vx_report_write_decode_state(report, context, prefilled,
                                 vx_decode_mode_name(mode),
                                 vx_decode_mode_name(last));
}

static void vx_report_set_context_decode_state(VxReport* report,
                                               VxExecutionContext* context) {
    if (!report || report->struct_size != sizeof(*report) || !context ||
        context->decode_row_mode == VX_DECODE_ROW_DISABLED) return;
    if (context->compiled->provider) {
        const char* last = context->decode_last_operation == 1 ? "prefill" :
                           context->decode_last_operation == 2 ? "step" : "none";
        vx_report_write_decode_state(report, context,
                                     context->decode_prefilled ? 1 : 0,
                                     "provider", last);
    } else {
        VxEngineStateScope scope =
            vx_engine_state_scope_enter(context->engine_state);
        vx_report_set_builtin_decode_state(report, context);
        vx_engine_state_scope_leave(scope);
    }
}

static int vx_adapter_is_base(const VxAdapterRevisionRecord* adapter) {
    return adapter && adapter->name && !strcmp(adapter->name, "__base__");
}

static VxStatus vx_context_prepare_adapter(
    VxExecutionContext* context,
    const VxAdapterRevisionRecord* adapter,
    char* prepared_route,
    size_t prepared_route_capacity,
    VxReport* report) {
    if (!context || !adapter || !prepared_route || !prepared_route_capacity)
        return VX_STATUS_INVALID_ARGUMENT;
    prepared_route[0] = '\0';
    if (context->compiled->provider) {
        if (vx_adapter_is_base(adapter) &&
            (!context->adapter || vx_adapter_is_base(context->adapter)))
            return VX_STATUS_OK;
        if (!context->compiled->provider->context_select_adapter)
            return vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED,
                           VX_STAGE_ADAPTER, context->compiled->backend,
                           VX_CODE_ADAPTER_UNSUPPORTED,
                           "backend does not implement adapter selection");
        VxStatus status = context->compiled->provider->context_select_adapter(
            context->provider_context, adapter->identity, adapter->revision,
            adapter->package_path, adapter->version_name, report);
        vx_report_normalize_provider_output(report);
        return status;
    }
    if (!vx_adapter_is_base(adapter) && adapter->package_path) {
        VxEngineStateScope scope =
            vx_engine_state_scope_enter(context->engine_state);
        int written = snprintf(prepared_route,
                               prepared_route_capacity,
                               "vx-%016" PRIx64 "-%016" PRIx64,
                               adapter->identity, adapter->revision);
        int loaded = written > 0 &&
            (size_t)written < prepared_route_capacity &&
            volvoxai_engine_adapter_load(
                adapter->package_path, prepared_route) == 0;
        if (!loaded && prepared_route[0]) {
            loaded = volvoxai_engine_adapter_route_begin(
                prepared_route) == 0;
            if (loaded) volvoxai_engine_adapter_route_end();
        }
        vx_engine_state_scope_leave(scope);
        if (!loaded) {
            return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                           VX_STAGE_ADAPTER, context->compiled->backend,
                           VX_CODE_ADAPTER_PACKAGE_REJECTED,
                           "adapter package is incompatible with the model");
        }
    }
    return VX_STATUS_OK;
}

static int vx_context_has_dynamic_logical_domain(
        const VxExecutionContext* context) {
    if (!context) return 0;
    for (size_t index = 0; index < context->logical_tensor_count; index++) {
        const VxDeclaredTensor* tensor = &context->logical_tensors[index];
        for (uint32_t axis = 0; axis < tensor->rank; axis++)
            if (vx_dimension_domain_is_dynamic(tensor, axis)) return 1;
    }
    return 0;
}

static VxStatus vx_context_prepare_native_gpu_dynamic_reservation(
        VxExecutionContext* context) {
    VolvoxAIEngineMaximumTensor* maximums;
    VxEngineStateScope scope;
    int configured;
    int prewarmed;
    int logical_preload_demoted;
    int weights_retained;
    int preload_complete = 0;
    int reserved;
    int allocation_failed = 0;
    size_t arena_bytes;
    if (!context || !context->engine_state ||
        !vx_native_gpu_backend(context->compiled->builtin_backend) ||
        !context->logical_tensor_count)
        return VX_STATUS_INVALID_ARGUMENT;
    maximums = (VolvoxAIEngineMaximumTensor*)calloc(
        context->logical_tensor_count, sizeof(*maximums));
    if (!maximums) return VX_STATUS_OUT_OF_MEMORY;
    for (size_t index = 0; index < context->logical_tensor_count; index++) {
        maximums[index].name = context->logical_tensors[index].name;
        maximums[index].dtype = (int)context->logical_tensors[index].dtype;
        maximums[index].rank = (int)context->logical_tensors[index].rank;
        for (uint32_t axis = 0;
             axis < context->logical_tensors[index].rank; axis++)
            maximums[index].maximum_shape[axis] = (int)
                context->logical_tensors[index].maximums[axis];
        maximums[index].maximum_byte_size =
            context->logical_tensors[index].maximum_byte_size;
    }
    scope = vx_engine_state_scope_enter(context->engine_state);
    configured = volvoxai_engine_configure_dynamic_shape_domain(
        maximums, context->logical_tensor_count);
    /* The bootstrap projection is one legal minimum binding used only to
     * materialize every invariant weight/metadata/LUT route. The pointwise
     * maximum layout above, never this execution sample, proves the domain. */
#if defined(VOLVOXAI_ENABLE_CUDA) && VOLVOXAI_ENABLE_CUDA
    if (configured == 0 &&
        context->compiled->builtin_backend == VX_BACKEND_KIND_CUDA)
        cuda_graph_clear_allocation_failure();
#endif
    prewarmed = configured == 0 ? volvoxai_engine_forward() : -1;
    logical_preload_demoted = prewarmed == 0
        ? volvoxai_engine_demote_preloaded_logical_tensors_locked() : -1;
    weights_retained = logical_preload_demoted == 0
        ? volvoxai_engine_retain_preloaded_model_weights_locked() : -1;
#if defined(VOLVOXAI_ENABLE_CUDA) && VOLVOXAI_ENABLE_CUDA
    if (weights_retained == 0 &&
        context->compiled->builtin_backend == VX_BACKEND_KIND_CUDA)
        preload_complete = cuda_graph_complete_invariant_preload();
#endif
    reserved = preload_complete == 0
        ? volvoxai_engine_reserve_dynamic_shape_domain() : -1;
#if defined(VOLVOXAI_ENABLE_CUDA) && VOLVOXAI_ENABLE_CUDA
    if (context->compiled->builtin_backend == VX_BACKEND_KIND_CUDA)
        allocation_failed = cuda_graph_last_allocation_failed();
#endif
    arena_bytes = context->engine_state->dynamic_arena_capacity_bytes;
    vx_engine_state_scope_leave(scope);
    free(maximums);
    if (configured == -2 || reserved == -2 ||
        (prewarmed != 0 && allocation_failed))
        return VX_STATUS_OUT_OF_MEMORY;
    if (configured != 0 || prewarmed != 0 ||
        logical_preload_demoted != 0 || weights_retained != 0 ||
        preload_complete != 0 || reserved != 0)
        return VX_STATUS_BACKEND_UNSUPPORTED;
    if ((uint64_t)arena_bytes > UINT64_MAX - context->allocated_bytes)
        return VX_STATUS_OUT_OF_MEMORY;
    context->allocated_bytes += (uint64_t)arena_bytes;
    return VX_STATUS_OK;
}

VxStatus vx_compiled_model_create_context(
    VxCompiledModel* compiled,
    const VxContextOptions* options,
    VxExecutionContext** out_context,
    VxReport* report) {
    VxContextOptions resolved = VX_CONTEXT_OPTIONS_INIT;
    VxExecutionContext* context;
    VxStatus status;
    char prepared_route[96];
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!compiled || !out_context ||
        (options && !vx_context_options_valid(options)))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_CONTEXT_CREATE, compiled ? compiled->backend : NULL,
                       VX_CODE_INVALID_CONTEXT_OPTIONS, "context options are invalid");
    *out_context = NULL;
    if (options) resolved = *options;
    for (size_t i = 0; i < resolved.decode_input_count; i++) {
        if (vx_model_input_index(compiled->model, resolved.decode_inputs[i]) == SIZE_MAX)
            return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_CONTEXT_CREATE, compiled->backend,
                VX_CODE_INVALID_DECODE_INPUT, "decode input must name a declared model input");
        for (size_t j = 0; j < i; j++) if (!strcmp(resolved.decode_inputs[i], resolved.decode_inputs[j]))
            return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_CONTEXT_CREATE, compiled->backend,
                VX_CODE_DUPLICATE_DECODE_INPUT, "decode input names must be unique");
    }
    size_t lane_bytes = sizeof(uint32_t) + sizeof(int32_t) + sizeof(uint8_t);
    if ((size_t)resolved.decode_lanes > (SIZE_MAX - sizeof(*context)) / lane_bytes)
        return vx_fail(report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_CONTEXT_CREATE, compiled->backend,
            VX_CODE_OUT_OF_MEMORY, "decode lane storage is too large");
    size_t context_bytes = sizeof(*context) + (size_t)resolved.decode_lanes * lane_bytes;
    context = (VxExecutionContext*)calloc(1, context_bytes);
    if (!context)
        return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                       VX_STAGE_CONTEXT_CREATE, compiled->backend,
                       VX_CODE_OUT_OF_MEMORY, "context allocation failed");
    atomic_init(&context->references, 1);
    context->compiled = compiled;
    context->compiled_reference_owned = 1;
    context->adapter = compiled->adapter;
    vx_adapter_revision_retain(context->adapter);
    context->identity = vx_next_object_identity(compiled->model->runtime);
    context->allocated_bytes = context_bytes;
    context->decode_active_lengths = (uint32_t*)(context + 1);
    context->decode_positions = (int32_t*)(context->decode_active_lengths + resolved.decode_lanes);
    context->decode_parked = (uint8_t*)(context->decode_positions + resolved.decode_lanes);
    context->decode_row_mode = resolved.decode_row_mode;
    context->require_incremental = resolved.require_incremental;
    context->decode_lanes = resolved.decode_lanes;
    vx_compiled_model_retain(compiled);
    if (pthread_mutex_init(&context->queue_mutex, NULL) != 0) {
        vx_adapter_revision_release(context->adapter);
        vx_compiled_model_release(compiled);
        free(context);
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_CONTEXT_CREATE,
                       compiled->backend, VX_CODE_QUEUE_INIT_FAILED,
                       "context queue initialization failed");
    }
    if (pthread_cond_init(&context->queue_condition, NULL) != 0) {
        pthread_mutex_destroy(&context->queue_mutex);
        vx_adapter_revision_release(context->adapter);
        vx_compiled_model_release(compiled);
        free(context);
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_CONTEXT_CREATE,
                       compiled->backend, VX_CODE_QUEUE_INIT_FAILED,
                       "context queue initialization failed");
    }
    if (!compiled->provider &&
        compiled->builtin_backend == VX_PORTABLE_BACKEND_KIND &&
        resolved.decode_row_mode != VX_DECODE_ROW_DISABLED &&
        !compiled->call_sequence_resources.sequence_supported) {
        status = vx_fail(
            report, VX_STATUS_BACKEND_UNSUPPORTED,
            VX_STAGE_CONTEXT_CREATE, compiled->backend,
            VX_CODE_DECODE_RESOURCE_DOMAIN_UNSUPPORTED,
            "requested decode mode exceeds the compiled portable call-sequence resource domain");
        pthread_cond_destroy(&context->queue_condition);
        pthread_mutex_destroy(&context->queue_mutex);
        vx_adapter_revision_release(context->adapter);
        vx_compiled_model_release(compiled);
        free(context);
        return status;
    }
    if (compiled->provider) {
        status = compiled->provider->context_create(compiled->provider_compiled,
                                                     &resolved,
                                                     &context->provider_context,
                                                     report);
        vx_report_normalize_provider_output(report);
        if (status != VX_STATUS_OK || !context->provider_context) {
            if (context->provider_context)
                compiled->provider->context_destroy(context->provider_context);
            pthread_cond_destroy(&context->queue_condition);
            pthread_mutex_destroy(&context->queue_mutex);
            vx_adapter_revision_release(context->adapter);
            vx_compiled_model_release(compiled);
            free(context);
            return vx_fail(report,
                           status == VX_STATUS_OK ? VX_STATUS_BACKEND_UNAVAILABLE : status,
                           VX_STAGE_CONTEXT_CREATE, compiled->backend,
                           VX_CODE_BACKEND_CONTEXT_CREATE_FAILED,
                           "backend context instance creation failed");
        }
    } else {
        context->engine_state = (VxEngineState*)calloc(1, sizeof(*context->engine_state));
        if (!context->engine_state ||
            vx_runtime_engine_state_init(compiled->model->runtime,
                                         context->engine_state) != 0) {
            free(context->engine_state);
            pthread_cond_destroy(&context->queue_condition);
            pthread_mutex_destroy(&context->queue_mutex);
            vx_adapter_revision_release(context->adapter);
            vx_compiled_model_release(compiled);
            free(context);
            return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                           VX_STAGE_CONTEXT_CREATE, compiled->backend,
                           VX_CODE_STATE_ALLOCATION_FAILED, "context state allocation failed");
        }
        context->engine_state_initialized = 1;
        if (!compiled->resources ||
            !compiled->resources->weight_store) {
            status = vx_fail(report, VX_STATUS_INTERNAL,
                             VX_STAGE_CONTEXT_CREATE, compiled->backend,
                             VX_CODE_WEIGHT_STORE_MISSING,
                             "compiled built-in weight store is unavailable");
            vx_private_engine_unload(context);
            pthread_cond_destroy(&context->queue_condition);
            pthread_mutex_destroy(&context->queue_mutex);
            vx_adapter_revision_release(context->adapter);
            vx_compiled_model_release(compiled);
            free(context);
            return status;
        }
        context->resource_lease = compiled->resources;
        vx_compiled_resource_owner_retain(context->resource_lease);
        context->engine_state->compiled_cpu_weights =
            context->resource_lease->cpu_weights;
        context->allocated_bytes += sizeof(*context->engine_state);
        context->dynamic_shape_stats =
            (VolvoxAIEngineDynamicShapeStats)
                VOLVOXAI_ENGINE_DYNAMIC_SHAPE_STATS_INIT;
        status = vx_logical_tensors_load(
            compiled->model, &context->logical_tensors,
            &context->logical_tensor_count);
        if (status == VX_STATUS_OK)
            status = vx_logical_tensor_name_index_build(context);
        if (status != VX_STATUS_OK) {
            vx_private_engine_unload(context);
            pthread_cond_destroy(&context->queue_condition);
            pthread_mutex_destroy(&context->queue_mutex);
            vx_adapter_revision_release(context->adapter);
            vx_compiled_model_release(compiled);
            free(context);
            return vx_fail(report, status, VX_STAGE_CONTEXT_CREATE,
                           compiled->backend,
                           status == VX_STATUS_OUT_OF_MEMORY
                               ? VX_CODE_OUT_OF_MEMORY : VX_CODE_LOGICAL_GRAPH_INVALID,
                           status == VX_STATUS_OUT_OF_MEMORY
                               ? "logical tensor table allocation failed"
                               : "closed v1 logical tensor table is invalid");
        }
        /* shutdown owns cleanup for both successful and partial loads */
        context->engine_loaded = 1;
        status = vx_private_engine_load(context->engine_state, compiled->model,
                                        compiled->weights,
                                        context->resource_lease->weight_store,
                                        &compiled->model->runtime->options,
                                        compiled->builtin_backend,
                                        compiled->backend,
                                        report, VX_STAGE_CONTEXT_CREATE);
        if (status != VX_STATUS_OK) {
            vx_private_engine_unload(context);
            pthread_cond_destroy(&context->queue_condition);
            pthread_mutex_destroy(&context->queue_mutex);
            vx_adapter_revision_release(context->adapter);
            vx_compiled_model_release(compiled);
            free(context);
            return status;
        }
        if (compiled->builtin_backend == VX_PORTABLE_BACKEND_KIND) {
            VxEngineStateScope scope =
                vx_engine_state_scope_enter(context->engine_state);
            int activation_status =
                volvoxai_engine_configure_portable_cpu_activation_plan(
                    compiled->graph_plan_request_v1,
                    compiled->graph_plan_request_v1_bytes,
                    compiled->graph_plan_v1,
                    compiled->graph_plan_v1_bytes);
            vx_engine_state_scope_leave(scope);
            if (activation_status != 0) {
                VxStatus failure = vx_fail(
                    report,
                    activation_status == -2 ? VX_STATUS_OUT_OF_MEMORY :
                                               VX_STATUS_INVALID_GRAPH,
                    VX_STAGE_CONTEXT_CREATE, compiled->backend,
                    activation_status == -2 ? VX_CODE_OUT_OF_MEMORY :
                                               VX_CODE_ACTIVATION_PLAN_INVALID,
                    activation_status == -2
                        ? "portable CPU activation-plan setup allocation failed"
                        : "portable CPU activation-plan topology is unavailable");
                vx_private_engine_unload(context);
                pthread_cond_destroy(&context->queue_condition);
                pthread_mutex_destroy(&context->queue_mutex);
                vx_adapter_revision_release(context->adapter);
                vx_compiled_model_release(compiled);
                free(context);
                return failure;
            }
        }
        uint64_t graph_metadata_bytes, graph_metadata_peak;
        if (vx_engine_state_graph_metadata_owned_bytes(context->engine_state,
                &graph_metadata_bytes, &graph_metadata_peak) ||
            graph_metadata_bytes > UINT64_MAX - context->allocated_bytes) {
            vx_private_engine_unload(context);
            pthread_cond_destroy(&context->queue_condition);
            pthread_mutex_destroy(&context->queue_mutex);
            vx_adapter_revision_release(context->adapter);
            vx_compiled_model_release(compiled);
            free(context);
            return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                           VX_STAGE_CONTEXT_CREATE, compiled->backend,
                           VX_CODE_STATE_ALLOCATION_OVERFLOW,
                           "context graph metadata allocation overflowed accounting");
        }
        context->allocated_bytes += graph_metadata_bytes;
        context->engine_state->bounded_gpu_value_domain_proven =
            vx_builtin_gpu_backend(compiled->builtin_backend);
        if (vx_native_gpu_backend(compiled->builtin_backend) &&
            vx_context_has_dynamic_logical_domain(context)) {
            if (compiled->builtin_backend == VX_BACKEND_KIND_CUDA &&
                (!vx_adapter_is_base(context->adapter) ||
                 resolved.decode_row_mode != VX_DECODE_ROW_DISABLED)) {
                vx_private_engine_unload(context);
                pthread_cond_destroy(&context->queue_condition);
                pthread_mutex_destroy(&context->queue_mutex);
                vx_adapter_revision_release(context->adapter);
                vx_compiled_model_release(compiled);
                free(context);
                return vx_fail(
                    report, VX_STATUS_BACKEND_UNSUPPORTED,
                    VX_STAGE_CONTEXT_CREATE, compiled->backend,
                    !vx_adapter_is_base(compiled->adapter)
                        ? VX_CODE_DYNAMIC_CUDA_ADAPTER_UNSUPPORTED
                        : VX_CODE_DYNAMIC_CUDA_DECODE_UNSUPPORTED,
                    !vx_adapter_is_base(compiled->adapter)
                        ? "dynamic CUDA contexts require the base adapter until adapter residency is proved"
                        : "dynamic CUDA decode requires a separately proved persistent device layout");
            }
            status = vx_context_prepare_native_gpu_dynamic_reservation(context);
            if (status != VX_STATUS_OK) {
                vx_private_engine_unload(context);
                pthread_cond_destroy(&context->queue_condition);
                pthread_mutex_destroy(&context->queue_mutex);
                vx_adapter_revision_release(context->adapter);
                vx_compiled_model_release(compiled);
                free(context);
                return vx_fail(
                    report, status, VX_STAGE_CONTEXT_CREATE,
                    compiled->backend,
                    status == VX_STATUS_OUT_OF_MEMORY
                        ? VX_CODE_DYNAMIC_DOMAIN_OUT_OF_MEMORY
                        : VX_CODE_DYNAMIC_DOMAIN_RESERVATION_FAILED,
                    status == VX_STATUS_OUT_OF_MEMORY
                        ? "fixed native GPU dynamic-domain reservation exhausted host or device memory"
                        : "native GPU could not preload invariants and reserve the complete fixed dynamic domain");
            }
        }
        if (compiled->builtin_backend == VX_PORTABLE_BACKEND_KIND) {
            const size_t workspace_bound =
                compiled->maximum_cpu_typed_workspace_bytes;
            VxEngineStateScope scope =
                vx_engine_state_scope_enter(context->engine_state);
            const int workspace_status =
                volvoxai_engine_configure_cpu_typed_workspace(
                    workspace_bound);
            vx_engine_state_scope_leave(scope);
            if (workspace_status != 0 || (uint64_t)workspace_bound >
                    UINT64_MAX - context->allocated_bytes) {
                VxStatus failure = vx_fail(
                    report, VX_STATUS_OUT_OF_MEMORY,
                    VX_STAGE_CONTEXT_CREATE, compiled->backend,
                    VX_CODE_WORKSPACE_ALLOCATION_FAILED,
                    "proved context CPU workspace allocation failed");
                vx_private_engine_unload(context);
                pthread_cond_destroy(&context->queue_condition);
                pthread_mutex_destroy(&context->queue_mutex);
                vx_adapter_revision_release(context->adapter);
                vx_compiled_model_release(compiled);
                free(context);
                return failure;
            }
            context->allocated_bytes += (uint64_t)workspace_bound;
        }
        if (resolved.decode_row_mode != VX_DECODE_ROW_DISABLED) {
            VolvoxAIDecodeSessionOptions decode_options =
                VOLVOXAI_DECODE_SESSION_OPTIONS_INIT;
            VxEngineStateScope scope =
                vx_engine_state_scope_enter(context->engine_state);
            decode_options.row_mode = resolved.decode_row_mode ==
                VX_DECODE_ROW_REQUIRED ? VOLVOXAI_DECODE_ROW_REQUIRED :
                                         VOLVOXAI_DECODE_ROW_AUTO;
            decode_options.require_incremental = resolved.require_incremental;
            decode_options.lanes = resolved.decode_lanes;
            context->decode_session = volvoxai_engine_decode_session_create(
                &decode_options);
            vx_engine_state_scope_leave(scope);
            if (!context->decode_session) {
                vx_private_engine_unload(context);
                pthread_cond_destroy(&context->queue_condition);
                pthread_mutex_destroy(&context->queue_mutex);
                vx_adapter_revision_release(context->adapter);
                vx_compiled_model_release(compiled);
                free(context);
                return vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED,
                               VX_STAGE_CONTEXT_CREATE, compiled->backend,
                               VX_CODE_DECODE_MODE_UNSUPPORTED,
                               "requested decode mode is unavailable");
            }
        }
    }
    if (resolved.decode_row_mode != VX_DECODE_ROW_DISABLED &&
        compiled->model->input_count) {
        size_t shape_bytes = compiled->model->input_count * VX_MAX_TENSOR_RANK * sizeof(int64_t);
        context->decode_input_shapes = calloc(1, shape_bytes + compiled->model->input_count);
        if (!context->decode_input_shapes) {
            if (compiled->provider) {
                if (compiled->provider->context_close)
                    (void)compiled->provider->context_close(
                        context->provider_context, NULL);
                compiled->provider->context_destroy(context->provider_context);
            } else {
                vx_private_engine_unload(context);
            }
            pthread_cond_destroy(&context->queue_condition);
            pthread_mutex_destroy(&context->queue_mutex);
            vx_adapter_revision_release(context->adapter);
            vx_compiled_model_release(compiled);
            free(context);
            return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                           VX_STAGE_CONTEXT_CREATE, compiled->backend,
                           VX_CODE_OUT_OF_MEMORY,
                           "decode shape signature allocation failed");
        }
        context->decode_default_inputs = (uint8_t*)context->decode_input_shapes + shape_bytes;
        if (!resolved.decode_input_count) memset(context->decode_default_inputs, 1, compiled->model->input_count);
        else for (size_t i = 0; i < resolved.decode_input_count; i++)
            context->decode_default_inputs[vx_model_input_index(compiled->model, resolved.decode_inputs[i])] = 1;
        context->allocated_bytes += shape_bytes + compiled->model->input_count;
    }
    if (!compiled->provider &&
        compiled->builtin_backend == VX_PORTABLE_BACKEND_KIND) {
        const VxCallSequencePolicyKindV1 policy_kind =
            resolved.decode_row_mode == VX_DECODE_ROW_DISABLED
                ? VX_CALL_SEQUENCE_POLICY_FORWARD_ONLY
                : VX_CALL_SEQUENCE_POLICY_ALL_CROSS_CALL_LIVE;
        status = vx_context_call_sequence_prepare(context, policy_kind);
        if (status != VX_STATUS_OK) {
            vx_private_engine_unload(context);
            pthread_cond_destroy(&context->queue_condition);
            pthread_mutex_destroy(&context->queue_mutex);
            free(context->decode_input_shapes);
            vx_adapter_revision_release(context->adapter);
            vx_compiled_model_release(compiled);
            free(context);
            return vx_fail(
                report, status, VX_STAGE_CONTEXT_CREATE, compiled->backend,
                status == VX_STATUS_OUT_OF_MEMORY
                    ? VX_CODE_OUT_OF_MEMORY : VX_CODE_CALL_SEQUENCE_POLICY_INVALID,
                status == VX_STATUS_OUT_OF_MEMORY
                    ? "portable call-sequence evidence allocation failed"
                    : "portable call-sequence policy provenance validation failed");
        }
    }
    status = vx_context_prepare_adapter(context, context->adapter,
                                        prepared_route,
                                        sizeof(prepared_route), report);
    if (status != VX_STATUS_OK) {
        if (compiled->provider) {
            if (compiled->provider->context_close)
                (void)compiled->provider->context_close(
                    context->provider_context, NULL);
            compiled->provider->context_destroy(context->provider_context);
        } else {
            vx_private_engine_unload(context);
        }
        pthread_cond_destroy(&context->queue_condition);
        pthread_mutex_destroy(&context->queue_mutex);
        free(context->decode_input_shapes);
        vx_context_call_sequence_clear(&context->call_sequence_evidence);
        vx_adapter_revision_release(context->adapter);
        vx_compiled_model_release(compiled);
        free(context);
        return status;
    }
    snprintf(context->adapter_route_version,
             sizeof(context->adapter_route_version), "%s", prepared_route);
    *out_context = context;
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_CONTEXT_CREATE,
                    compiled->backend,
                    compiled->provider ? NULL :
                        vx_builtin_device_identity(compiled->backend),
                    VX_CODE_NONE,
                    "execution context created", 0);
    vx_report_set_lineage(report, NULL, NULL, compiled, context);
    vx_report_set_plan_evidence(report, compiled);
    if (context->decode_row_mode != VX_DECODE_ROW_DISABLED) {
        if (!compiled->provider) {
            VxEngineStateScope scope =
                vx_engine_state_scope_enter(context->engine_state);
            vx_report_set_builtin_decode_state(report, context);
            vx_engine_state_scope_leave(scope);
        } else {
            snprintf(report->decode_state, sizeof(report->decode_state), "%s",
                     "enabled=1;prefilled=0;mode=provider;last=none");
        }
    }
    return VX_STATUS_OK;
}

void vx_execution_context_retain(VxExecutionContext* context) {
    if (context)
        atomic_fetch_add_explicit(&context->references, 1, memory_order_relaxed);
}

VxStatus vx_execution_context_close(VxExecutionContext* context,
                                    VxReport* report) {
    uint64_t ticket;
    VxStatus status = VX_STATUS_OK;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!context)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_CLOSE,
                       NULL, VX_CODE_INVALID_CONTEXT, "execution context is NULL");
    pthread_mutex_lock(&context->queue_mutex);
    if (context->closed) {
        pthread_mutex_unlock(&context->queue_mutex);
        vx_report_write(report, VX_STATUS_OK, VX_STAGE_CLOSE,
                        context->compiled->backend, NULL, VX_CODE_NONE,
                        "execution context already closed", 0);
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        return VX_STATUS_OK;
    }
    if (context->closing) {
#if !defined(VOLVOXAI_NO_THREADS)
        while (!context->closed)
            pthread_cond_wait(&context->queue_condition, &context->queue_mutex);
#endif
        if (!context->closed) {
            pthread_mutex_unlock(&context->queue_mutex);
            return vx_fail(report, VX_STATUS_BUSY, VX_STAGE_CLOSE,
                           context->compiled->backend,
                           VX_CODE_BUSY, "execution context closing concurrently");
        }
        pthread_mutex_unlock(&context->queue_mutex);
        vx_report_write(report, VX_STATUS_OK, VX_STAGE_CLOSE,
                        context->compiled->backend, NULL, VX_CODE_NONE,
                        "execution context closed", 0);
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        return VX_STATUS_OK;
    }
#if defined(VOLVOXAI_NO_THREADS)
    if (context->next_ticket != context->serving_ticket) {
        pthread_mutex_unlock(&context->queue_mutex);
        return vx_fail(report, VX_STATUS_BUSY, VX_STAGE_CLOSE,
                       context->compiled->backend,
                       VX_CODE_BUSY, "execution context re-entrant invocation");
    }
#endif
    context->closing = 1;
    ticket = context->next_ticket++;
#if !defined(VOLVOXAI_NO_THREADS)
    while (ticket != context->serving_ticket)
        pthread_cond_wait(&context->queue_condition, &context->queue_mutex);
#endif
    pthread_mutex_unlock(&context->queue_mutex);

    if (context->compiled->provider) {
        if (context->compiled->provider->context_close)
            status = context->compiled->provider->context_close(
                context->provider_context, report);
        vx_report_normalize_provider_output(report);
    } else {
        vx_private_engine_unload(context);
    }

    pthread_mutex_lock(&context->queue_mutex);
    context->closed = 1;
    context->serving_ticket++;
    pthread_cond_broadcast(&context->queue_condition);
    pthread_mutex_unlock(&context->queue_mutex);
    if (status == VX_STATUS_OK)
        vx_report_write(report, VX_STATUS_OK, VX_STAGE_CLOSE,
                        context->compiled->backend, NULL, VX_CODE_NONE,
                        "execution context closed", 0);
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    vx_report_set_plan_evidence(report, context->compiled);
    return status;
}

void vx_execution_context_release(VxExecutionContext* context) {
    if (!context || atomic_fetch_sub_explicit(&context->references, 1,
                                              memory_order_acq_rel) != 1) return;
    (void)vx_execution_context_close(context, NULL);
    if (context->compiled->provider && context->provider_context)
        context->compiled->provider->context_destroy(context->provider_context);
    free(context->decode_input_shapes);
    vx_context_call_sequence_clear(&context->call_sequence_evidence);
    pthread_cond_destroy(&context->queue_condition);
    pthread_mutex_destroy(&context->queue_mutex);
    vx_adapter_revision_release(context->adapter);
    if (context->compiled_reference_owned)
        vx_compiled_model_release(context->compiled);
    free(context);
}

size_t vx_execution_context_input_count(VxExecutionContext* context) {
    VxContextOperation operation;
    size_t count = 0;
    if (vx_context_operation_begin(context, &operation, VX_STAGE_INPUT, NULL) !=
        VX_STATUS_OK) return 0;
    count = context->compiled->model->input_count;
    vx_context_operation_end(context, &operation);
    return count;
}

VxStatus vx_execution_context_input_spec(VxExecutionContext* context,
                                         size_t index,
                                         VxTensorSpec* spec,
                                         VxReport* report) {
    VxContextOperation operation;
    VxStatus status;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!spec || spec->struct_size != sizeof(*spec))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_INPUT,
                       context ? context->compiled->backend : NULL,
                       VX_CODE_INVALID_TENSOR_SPEC, "tensor spec buffer is invalid");
    status = vx_context_operation_begin(context, &operation, VX_STAGE_INPUT, report);
    if (status != VX_STATUS_OK) return status;
    if (index >= context->compiled->model->input_count) {
        status = vx_fail(report, VX_STATUS_NOT_FOUND, VX_STAGE_INPUT,
                         context->compiled->backend, VX_CODE_INPUT_NOT_FOUND,
                         "input index is out of range");
    } else {
        vx_declared_tensor_spec(&context->compiled->model->inputs[index], spec);
        status = VX_STATUS_OK;
        vx_report_write(report, status, VX_STAGE_INPUT,
                        context->compiled->backend,
                        context->compiled->provider ? NULL :
                            vx_builtin_device_identity(context->compiled->backend),
                        VX_CODE_NONE, "logical input spec returned", 0);
    }
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    vx_report_set_plan_evidence(report, context->compiled);
    vx_context_operation_end(context, &operation);
    return status;
}

VxStatus vx_execution_context_input_affine_quantization(
    VxExecutionContext* context,
    const char* name,
    VxAffineQuantization* quantization,
    VxReport* report) {
    VxContextOperation operation;
    VxStatus status;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!name || !name[0] || !quantization ||
        quantization->struct_size != sizeof(*quantization))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_INPUT,
                       context ? context->compiled->backend : NULL,
                       VX_CODE_INVALID_AFFINE_QUANTIZATION_REQUEST,
                       "input affine quantization request is invalid");
    quantization->defined = 0;
    quantization->scale = 0.0f;
    quantization->zero_point = 0;
    status = vx_context_operation_begin(context, &operation, VX_STAGE_INPUT, report);
    if (status != VX_STATUS_OK) return status;
    if (context->compiled->provider) {
        status = vx_fail(
            report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_INPUT,
            context->compiled->backend, VX_CODE_INPUT_QUANTIZATION_INTROSPECTION_UNSUPPORTED,
            "provider does not expose input affine quantization introspection");
    } else {
        VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
        float scale = 0.0f;
        int zero_point = 0;
        int result = volvoxai_engine_input_affine_quantization(
            name, &scale, &zero_point);
        vx_engine_state_scope_leave(scope);
        if (result < 0) {
            status = vx_fail(report, VX_STATUS_NOT_FOUND, VX_STAGE_INPUT,
                             context->compiled->backend, VX_CODE_INPUT_NOT_FOUND,
                             "input name was not found");
        } else {
            quantization->defined = result;
            quantization->scale = result ? scale : 0.0f;
            quantization->zero_point = result ? zero_point : 0;
            status = VX_STATUS_OK;
            vx_report_write(report, status, VX_STAGE_INPUT,
                            context->compiled->backend,
                            vx_builtin_device_identity(context->compiled->backend),
                            VX_CODE_NONE, "input affine quantization returned", 0);
        }
    }
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    vx_report_set_plan_evidence(report, context->compiled);
    vx_context_operation_end(context, &operation);
    return status;
}

typedef struct VxStagedBindingBatch {
    VxTensorBinding* bindings;
    size_t count;
    int64_t* resolved_input_shapes;
    VxInputValidationEvidence issue;
} VxStagedBindingBatch;

typedef enum VxContextRunKind {
    VX_CONTEXT_RUN_FORWARD = 0,
    VX_CONTEXT_RUN_DECODE_PREFILL = 1,
    VX_CONTEXT_RUN_DECODE_STEP = 2,
    VX_CONTEXT_RUN_PREFIX = 3
} VxContextRunKind;

static void vx_staged_binding_batch_clear(VxStagedBindingBatch* batch) {
    if (!batch) return;
    free(batch->bindings);
    free(batch->resolved_input_shapes);
    batch->bindings = NULL;
    batch->resolved_input_shapes = NULL;
    batch->count = 0;
    /* Keep failure evidence until the caller has populated its report. */
}

static size_t vx_model_input_index(const VxModel* model, const char* name) {
    if (!model || !name) return SIZE_MAX;
    for (size_t index = 0; index < model->input_count; index++)
        if (!strcmp(model->inputs[index].name, name)) return index;
    return SIZE_MAX;
}

static int64_t vx_binding_symbol_extent(const VxModel* model,
                                         const int64_t* resolved_shapes,
                                         size_t input_index, uint32_t axis) {
    const char* symbol = model->inputs[input_index].symbols[axis];
    if (!symbol) return 0;
    for (size_t prior = 0; prior < model->input_count; prior++) {
        const VxDeclaredTensor* spec = &model->inputs[prior];
        for (uint32_t dim = 0; dim < spec->rank; dim++) {
            int64_t extent = resolved_shapes[prior * VX_MAX_TENSOR_RANK + dim];
            if (extent && spec->symbols[dim] && !strcmp(spec->symbols[dim], symbol))
                return extent;
        }
    }
    return 0;
}

static void vx_input_evidence_text(char* target, size_t capacity,
                                    const char* source, int* truncated) {
    size_t length = strlen(source);
    if (length >= capacity) {
        *truncated = 1;
        length = capacity - 1u;
        /* Do not turn a valid UTF-8 name into an invalid protobuf string by
         * cutting through a multibyte code point. */
        while (length && ((unsigned char)source[length] & 0xc0u) == 0x80u)
            length--;
    }
    memcpy(target, source, length);
    target[length] = '\0';
}

static void vx_binding_issue(VxInputValidationEvidence* issue,
                              VxInputValidationCode code, size_t index,
                              const VxTensorBinding* binding,
                              const VxDeclaredTensor* spec) {
    const char* name = binding && binding->name ? binding->name : spec ? spec->name : "";
    memset(issue, 0, sizeof(*issue));
    issue->code = code;
    issue->has_input_index = index != SIZE_MAX;
    issue->input_index = index == SIZE_MAX ? 0 : index;
    vx_input_evidence_text(issue->name, sizeof(issue->name), name, &issue->names_truncated);
    if (binding) {
        issue->has_actual = 1;
        issue->actual_dtype = binding->dtype;
        issue->actual_location = binding->location;
        issue->actual_rank = binding->rank;
        for (uint32_t axis = 0; axis < binding->rank && axis < VX_MAX_TENSOR_RANK; axis++)
            issue->actual_shape[axis] = binding->shape[axis];
        issue->actual_bytes = binding->byte_size;
    }
    if (spec) {
        issue->has_expected = 1;
        issue->expected_dtype = spec->dtype;
        issue->expected_rank = spec->rank;
        for (uint32_t axis = 0; axis < spec->rank; axis++) {
            const char* symbol = spec->symbols[axis] ? spec->symbols[axis] : "";
            issue->kinds[axis] = spec->kinds[axis];
            issue->minimums[axis] = spec->minimums[axis];
            issue->maximums[axis] = spec->maximums[axis];
            issue->multiples[axis] = spec->multiples[axis];
            vx_input_evidence_text(issue->symbols[axis], sizeof(issue->symbols[axis]),
                                   symbol, &issue->names_truncated);
        }
    }
}

static VxStatus vx_stage_binding_batch(
    VxExecutionContext* context,
    const VxTensorBinding* inputs,
    size_t input_count,
    int require_complete,
    int decode_step,
    VxStagedBindingBatch* staged) {
    const VxModel* model = context->compiled->model;
    unsigned char* seen = NULL;
    size_t* indices = NULL;
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    memset(staged, 0, sizeof(*staged));
    if ((input_count && !inputs) || (decode_step && !context->decode_prefilled))
        return status;
    seen = model->input_count
        ? (unsigned char*)calloc(model->input_count, 1u) : NULL;
    indices = input_count ? (size_t*)calloc(input_count, sizeof(*indices)) : NULL;
    staged->resolved_input_shapes = model->input_count
        ? (int64_t*)calloc(model->input_count * VX_MAX_TENSOR_RANK,
                           sizeof(*staged->resolved_input_shapes)) : NULL;
    if ((model->input_count && (!seen || !staged->resolved_input_shapes)) ||
        (input_count && !indices)) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    if (decode_step && model->input_count)
        memcpy(staged->resolved_input_shapes, context->decode_input_shapes,
               model->input_count * VX_MAX_TENSOR_RANK * sizeof(int64_t));
    for (size_t index = 0; index < input_count; index++) {
        const VxTensorBinding* binding = &inputs[index];
        const VxDeclaredTensor* spec;
        size_t model_index;
        size_t elements = 1u;
        size_t element_size;
        size_t expected_bytes;
        VxInputValidationCode failure = VX_INPUT_UNSPECIFIED;
        uint32_t failed_axis = 0;
        int has_axis = 0;
        int64_t required_extent = 0;
        spec = NULL;
        expected_bytes = 0;
        if (binding->struct_size != sizeof(*binding)) goto done;
        if (!binding->name || !binding->name[0]) {
            failure = VX_INPUT_INVALID_NAME;
            goto invalid_tensor;
        }
        model_index = vx_model_input_index(model, binding->name);
        if (model_index == SIZE_MAX) {
            failure = VX_INPUT_UNKNOWN_INPUT;
            goto invalid_tensor;
        }
        spec = &model->inputs[model_index];
        if (seen[model_index]) {
            failure = VX_INPUT_DUPLICATE_INPUT;
            goto invalid_tensor;
        }
        seen[model_index] = 1u;
        indices[index] = model_index;
        if (binding->location != VX_MEMORY_HOST) {
            failure = VX_INPUT_LOCATION_UNSUPPORTED;
            goto invalid_tensor;
        }
        if (binding->dtype != spec->dtype) {
            failure = VX_INPUT_DTYPE_MISMATCH;
            goto invalid_tensor;
        }
        if (binding->rank != spec->rank || binding->rank > VX_MAX_TENSOR_RANK) {
            failure = VX_INPUT_RANK_MISMATCH;
            goto invalid_tensor;
        }
        for (uint32_t axis = 0; axis < binding->rank; axis++) {
            int64_t extent = binding->shape[axis];
            int64_t symbol_extent = vx_binding_symbol_extent(
                model, staged->resolved_input_shapes, model_index, axis);
            has_axis = 1;
            failed_axis = axis;
            if (extent <= 0 || extent < spec->minimums[axis] ||
                extent > spec->maximums[axis] || spec->multiples[axis] <= 0 ||
                extent % spec->multiples[axis] != 0 ||
                (spec->kinds[axis] == VX_DIMENSION_FIXED &&
                 (spec->symbols[axis] || extent != spec->minimums[axis])) ||
                (spec->kinds[axis] == VX_DIMENSION_SYMBOLIC &&
                 (!spec->symbols[axis] || !spec->symbols[axis][0]))) {
                failure = VX_INPUT_DIMENSION_OUT_OF_RANGE;
                goto invalid_tensor;
            }
            if (decode_step && extent != context->decode_input_shapes[
                    model_index * VX_MAX_TENSOR_RANK + axis]) {
                failure = VX_INPUT_DECODE_SHAPE_MISMATCH;
                required_extent = context->decode_input_shapes[
                    model_index * VX_MAX_TENSOR_RANK + axis];
                goto invalid_tensor;
            }
            if (symbol_extent && symbol_extent != extent) {
                failure = VX_INPUT_SYMBOL_MISMATCH;
                required_extent = symbol_extent;
                goto invalid_tensor;
            }
            if ((uint64_t)extent > SIZE_MAX / elements) {
                failure = VX_INPUT_SIZE_OVERFLOW;
                goto invalid_tensor;
            }
            elements *= (size_t)extent;
            staged->resolved_input_shapes[
                model_index * VX_MAX_TENSOR_RANK + axis] = extent;
        }
        has_axis = 0;
        element_size = vx_execution_dtype_size(binding->dtype);
        if (!element_size || elements > SIZE_MAX / element_size) {
            failure = VX_INPUT_SIZE_OVERFLOW;
            goto invalid_tensor;
        }
        expected_bytes = elements * element_size;
        if (binding->byte_size != expected_bytes) {
            failure = VX_INPUT_BYTE_SIZE_MISMATCH;
            goto invalid_tensor;
        }
        if (!binding->data) {
            failure = VX_INPUT_PAYLOAD_REQUIRED;
            goto invalid_tensor;
        }
        continue;
invalid_tensor:
        vx_binding_issue(&staged->issue, failure, index, binding, spec);
        staged->issue.has_axis = has_axis;
        staged->issue.axis = failed_axis;
        staged->issue.has_expected_bytes = expected_bytes != 0;
        staged->issue.expected_bytes = expected_bytes;
        staged->issue.has_required_extent = required_extent != 0;
        staged->issue.required_extent = required_extent;
        goto done;
    }
    if (require_complete)
        for (size_t index = 0; index < model->input_count; index++)
            if (!seen[index]) {
                vx_binding_issue(&staged->issue, VX_INPUT_MISSING_INPUT,
                                 SIZE_MAX, NULL, &model->inputs[index]);
                goto done;
            }
    staged->bindings = input_count
        ? (VxTensorBinding*)calloc(input_count, sizeof(*staged->bindings)) : NULL;
    if (input_count && !staged->bindings) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    staged->count = input_count;
    for (size_t index = 0; index < input_count; index++) {
        const VxTensorBinding* source = &inputs[index];
        VxTensorBinding* target = &staged->bindings[index];
        *target = *source;
        target->struct_size = sizeof(*target);
        target->name = model->inputs[indices[index]].name;
        target->location = VX_MEMORY_HOST;
        /* The synchronous execution API borrows host input storage until it
         * returns. Keep normalization copy-free; the selected backend copies
         * or uploads only after the complete graph plan commits. */
        target->data = source->data;
    }
    status = VX_STATUS_OK;
done:
    free(indices);
    free(seen);
    if (status != VX_STATUS_OK) vx_staged_binding_batch_clear(staged);
    return status;
}

static const VxTensorBinding* vx_staged_binding_find(
        const VxStagedBindingBatch* staged, const char* name) {
    if (!staged || !name) return NULL;
    for (size_t index = 0; index < staged->count; index++)
        if (!strcmp(staged->bindings[index].name, name))
            return &staged->bindings[index];
    return NULL;
}

/* Return 1 with a public graph-input name, 2 for an immutable/statically
 * ranged source, and 0 for an unexpected device-produced origin. The compile
 * proof admits exactly this same producer subset. */
static int vx_native_gpu_public_i32_source(
        const VxEngineState* state,
        const char* tensor_name,
        const char** public_name,
        int before_node,
        int depth) {
    const T* tensor;
    const Node* producer;
    if (!state || !tensor_name || !public_name || before_node < 0 ||
        before_node > state->node_count || depth < 0 ||
        depth > state->node_count)
        return 0;
    tensor = vx_native_gpu_validation_tensor_find(state, tensor_name);
    if (!tensor || tensor->dtype != T_I32) return 0;
    if (tensor->is_graph_input) {
        *public_name = tensor->name;
        return 1;
    }
    producer = vx_native_gpu_validation_node_producer(state, tensor_name);
    if (!producer) return 2;
    {
        int producer_index = vx_native_gpu_validation_node_index(
            state, producer);
        if (producer_index < 0 || producer_index >= before_node) return 0;
        before_node = producer_index;
    }
    if ((producer->operator_kind == VX_OP_CLIP) ||
        (producer->operator_kind == VX_OP_ARG_MAX) ||
        (producer->operator_kind == VX_OP_Q_ARG_MAX) ||
        (producer->operator_kind == VX_OP_EQUAL) ||
        (producer->operator_kind == VX_OP_GREATER_OR_EQUAL) ||
        (producer->operator_kind == VX_OP_NOT))
        return 2;
    if (producer->operator_kind == VX_OP_CAST) {
        const char* source_name = vx_native_gpu_validation_node_data_name(producer);
        const T* source = vx_native_gpu_validation_tensor_find(state, source_name);
        if (!source) return 0;
        if (source->dtype == T_U8 || source->dtype == T_I8) return 2;
        return vx_native_gpu_public_i32_source(state, source_name,
            public_name, before_node, depth + 1);
    }
    if (vx_native_gpu_value_preserving_node(producer))
        return vx_native_gpu_public_i32_source(
            state, vx_native_gpu_validation_node_data_name(producer),
            public_name, before_node, depth + 1);
    return 0;
}

static int vx_native_gpu_public_f32_source(
        const VxEngineState* state,
        const char* tensor_name,
        const char** public_name,
        int before_node,
        int depth) {
    const T* tensor;
    const Node* producer;
    if (!state || !tensor_name || !public_name || before_node < 0 ||
        before_node > state->node_count || depth < 0 ||
        depth > state->node_count)
        return 0;
    tensor = vx_native_gpu_validation_tensor_find(state, tensor_name);
    if (!tensor || tensor->dtype != T_F32) return 0;
    if (tensor->is_graph_input) {
        *public_name = tensor->name;
        return 1;
    }
    producer = vx_native_gpu_validation_node_producer(state, tensor_name);
    if (!producer) return 2;
    {
        int producer_index = vx_native_gpu_validation_node_index(
            state, producer);
        if (producer_index < 0 || producer_index >= before_node) return 0;
        before_node = producer_index;
    }
    if (producer->operator_kind == VX_OP_MOE_ROUTER) return 2;
    if (vx_native_gpu_value_preserving_node(producer))
        return vx_native_gpu_public_f32_source(
            state, vx_native_gpu_validation_node_data_name(producer),
            public_name, before_node, depth + 1);
    return 0;
}

static const VolvoxAIEngineResolvedTensor* vx_resolved_tensor_find(
        const VolvoxAIEngineResolvedTensor* resolved,
        size_t resolved_count,
        const char* name) {
    if (!resolved || !name) return NULL;
    for (size_t index = 0; index < resolved_count; index++)
        if (!strcmp(resolved[index].name, name)) return &resolved[index];
    return NULL;
}

static int vx_native_gpu_preflight_i32_binding(
        const VxStagedBindingBatch* staged,
        const char* public_name,
        int64_t minimum,
        int64_t maximum_exclusive,
        const Node* bank_node) {
    const VxTensorBinding* binding = vx_staged_binding_find(
        staged, public_name);
    if (!binding) return 1; /* An unchanged decode-step input was preflighted. */
    if (binding->dtype != VX_DTYPE_I32 || !binding->data ||
        binding->byte_size % sizeof(int32_t) ||
        minimum >= maximum_exclusive)
        return 0;
    size_t count = binding->byte_size / sizeof(int32_t);
    for (size_t index = 0; index < count; index++) {
        int32_t value;
        memcpy(&value,
               (const unsigned char*)binding->data +
                   index * sizeof(value),
               sizeof(value));
        if ((int64_t)value < minimum ||
            (int64_t)value >= maximum_exclusive)
            return 0;
        if (bank_node && bank_node->resident_slot_rows) {
            int64_t slot = value < 0 ? (int64_t)value + bank_node->resident_slot_domain : value;
            if (slot < 0 || (uint64_t)slot >= bank_node->resident_slot_domain ||
                bank_node->resident_slot_rows[slot] == UINT32_MAX) return 0;
        }
    }
    return 1;
}

static int vx_native_gpu_preflight_finite_binding(
        const VxStagedBindingBatch* staged,
        const char* public_name) {
    const VxTensorBinding* binding = vx_staged_binding_find(
        staged, public_name);
    if (!binding) return 1;
    if (binding->dtype != VX_DTYPE_F32 || !binding->data ||
        binding->byte_size % sizeof(float))
        return 0;
    size_t count = binding->byte_size / sizeof(float);
    for (size_t index = 0; index < count; index++) {
        float value;
        memcpy(&value,
               (const unsigned char*)binding->data +
                   index * sizeof(value),
               sizeof(value));
        if (!isfinite(value)) return 0;
    }
    return 1;
}

#if VOLVOXAI_ENABLE_WEBGPU
static int vx_webgpu_preflight_route_binding(const VxStagedBindingBatch* staged,
    const char* name, uint32_t experts, const Node* consumer) {
    const VxTensorBinding* binding = vx_staged_binding_find(staged, name);
    if (!binding) return 1;
    if (binding->dtype != VX_DTYPE_F32 || !binding->data || binding->byte_size % 4u) return 0;
    for (size_t i = 0; i < binding->byte_size / 4u; i++) {
        float value; memcpy(&value, (const unsigned char*)binding->data + i * 4u, 4u);
        if (!isfinite(value)) return 0;
        if (experts && (value < 0 || (double)value >= experts || value != (float)(uint32_t)value ||
            (consumer->resident_slot_rows && consumer->resident_slot_rows[(uint32_t)value] == UINT32_MAX))) return 0;
    }
    return 1;
}
#endif

/* Value predicates are graph-wide and run after canonical shape resolution
 * but before volvoxai_engine_commit_dynamic_shape copies an input, publishes a
 * plan, uploads a byte, or allows the first GPU dispatch. */
static VxStatus vx_native_gpu_preflight_public_values(
        const VxExecutionContext* context,
        const VxStagedBindingBatch* staged,
        const VolvoxAIEngineResolvedTensor* resolved,
        size_t resolved_count) {
    const cJSON* nodes;
    const VxEngineState* state;
    if (!context || !staged ||
        !vx_builtin_gpu_backend(context->compiled->builtin_backend))
        return VX_STATUS_OK;
    state = context->engine_state;
    nodes = cJSON_GetObjectItemCaseSensitive(
        context->compiled->model->logical_graph, "nodes");
    if (!state || !cJSON_IsArray(nodes)) return VX_STATUS_INVALID_GRAPH;
    for (const cJSON* node = nodes->child; node; node = node->next) {
        const cJSON* op_item = cJSON_GetObjectItemCaseSensitive(
            node, "opType");
        const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(
            node, "inputs");
        VxOperatorKind op = vx_operator_kind_from_name(cJSON_GetStringValue(op_item));
        const VxDeclaredTensor* logical_output = vx_native_gpu_node_output(
            node, context->logical_tensors, context->logical_tensor_count);
        const Node* validation_node = logical_output
            ? vx_native_gpu_validation_node_producer(
                  state, logical_output->name)
            : NULL;
        const int validation_node_index =
            vx_native_gpu_validation_node_index(state, validation_node);
        if (!op || !cJSON_IsObject(inputs) || validation_node_index < 0)
            return VX_STATUS_INVALID_GRAPH;
        if ((op == VX_OP_GATHER) || (op == VX_OP_GATHER_ELEMENTS)) {
            const cJSON* data_ref = cJSON_GetObjectItemCaseSensitive(
                inputs, "input");
            const cJSON* indices_ref = cJSON_GetObjectItemCaseSensitive(
                inputs, "indices");
            const char* public_name = NULL;
            int source_kind;
            int64_t axis_size = 0;
            if (!cJSON_IsString(data_ref) || !data_ref->valuestring ||
                !cJSON_IsString(indices_ref) || !indices_ref->valuestring)
                return VX_STATUS_INVALID_GRAPH;
            source_kind = vx_native_gpu_public_i32_source(
                state, indices_ref->valuestring, &public_name,
                validation_node_index, 0);
            if (!source_kind) return VX_STATUS_INVALID_GRAPH;
            if (source_kind != 1) continue;
            {
                const VolvoxAIEngineResolvedTensor* data =
                    vx_resolved_tensor_find(
                        resolved, resolved_count, data_ref->valuestring);
                const T* fixed = vx_native_gpu_validation_tensor_find(
                    state, data_ref->valuestring);
                int rank = data ? (int)data->rank : fixed ? fixed->ndim : 0;
                int64_t axis;
                if (!rank || !vx_native_gpu_integer_parameter(validation_node->params,
                        "axis", 0, -(int64_t)rank, (int64_t)rank - 1, &axis))
                    return VX_STATUS_INVALID_GRAPH;
                if (axis < 0) axis += rank;
                axis_size = data ? data->shape[axis] : fixed->shape[axis];
                if (validation_node->resident_slot_domain)
                    axis_size = validation_node->resident_slot_domain;
            }
            if (axis_size <= 0 ||
                !vx_native_gpu_preflight_i32_binding(
                    staged, public_name, -axis_size, axis_size, validation_node))
                return VX_STATUS_INVALID_ARGUMENT;
        } else if ((op == VX_OP_EMBEDDING) ||
                   (op == VX_OP_Q_EMBEDDING)) {
            const cJSON* ids_ref = cJSON_GetObjectItemCaseSensitive(
                inputs, "input");
            const cJSON* weight_ref = cJSON_GetObjectItemCaseSensitive(
                inputs, "weight");
            const T* weight;
            const char* public_name = NULL;
            int source_kind;
            if (!cJSON_IsString(ids_ref) || !ids_ref->valuestring ||
                !cJSON_IsString(weight_ref) || !weight_ref->valuestring)
                return VX_STATUS_INVALID_GRAPH;
            source_kind = vx_native_gpu_public_i32_source(
                state, ids_ref->valuestring, &public_name,
                validation_node_index, 0);
            if (!source_kind) return VX_STATUS_INVALID_GRAPH;
            if (source_kind != 1) continue;
            weight = vx_native_gpu_validation_tensor_find(
                state, weight_ref->valuestring);
            if (!weight || weight->ndim <= 0 || weight->shape[0] <= 0 ||
                !vx_native_gpu_preflight_i32_binding(
                    staged, public_name, 0, weight->shape[0], NULL))
                return VX_STATUS_INVALID_ARGUMENT;
        } else if ((op == VX_OP_Q_GROUP_NORM) ||
                   (op == VX_OP_Q_LAYER_NORM)) {
            static const char* const roles[] = {"weight", "bias"};
            for (size_t role = 0; role < 2u; role++) {
                const cJSON* reference = cJSON_GetObjectItemCaseSensitive(
                    inputs, roles[role]);
                if (!reference && role == 0)
                    reference = cJSON_GetObjectItemCaseSensitive(inputs, "scale");
                const char* public_name = NULL;
                int source_kind;
                if (!cJSON_IsString(reference) || !reference->valuestring)
                    return VX_STATUS_INVALID_GRAPH;
                source_kind = vx_native_gpu_public_f32_source(
                    state, reference->valuestring, &public_name,
                    validation_node_index, 0);
                if (!source_kind) return VX_STATUS_INVALID_GRAPH;
                if (source_kind == 1 &&
                    !vx_native_gpu_preflight_finite_binding(
                        staged, public_name))
                    return VX_STATUS_INVALID_ARGUMENT;
            }
#if VOLVOXAI_ENABLE_WEBGPU
        } else if ((op == VX_OP_MOE_LINEAR) &&
                   context->compiled->builtin_backend == VX_BACKEND_KIND_WEBGPU) {
            const T* weight = vx_native_gpu_validation_tensor_find(state,
                vx_native_gpu_validation_node_input_name(validation_node, VX_PORT_EXPERT_WEIGHT));
            if (!weight || weight->ndim != 3) return VX_STATUS_INVALID_GRAPH;
            uint32_t experts = validation_node->resident_slot_domain
                ? validation_node->resident_slot_domain : (uint32_t)weight->shape[0];
            for (int role = 0; role < 2; role++) {
                const char* name = vx_native_gpu_validation_node_input_name(validation_node,
                    role ? VX_PORT_ROUTE_WEIGHTS : VX_PORT_ROUTE_INDICES);
                const char* public_name = NULL;
                int kind = vx_native_gpu_public_f32_source(state, name, &public_name, validation_node_index, 0);
                if (!kind) return VX_STATUS_INVALID_GRAPH;
                if (kind == 1 && !vx_webgpu_preflight_route_binding(staged,
                        public_name, role ? 0 : experts, validation_node)) return VX_STATUS_INVALID_ARGUMENT;
            }
#endif
        }
    }
    return VX_STATUS_OK;
}

static VxStatus vx_commit_builtin_bindings(
    VxExecutionContext* context,
    const VxStagedBindingBatch* staged,
    const int64_t* resolved_input_shapes,
    VxContextRunKind kind,
    const char* required_shape_signature) {
    VolvoxAIEngineInputBinding* engine_inputs = NULL;
    VolvoxAIEngineResolvedTensor* resolved = NULL;
    VolvoxAIEngineDynamicShapeStats stats =
        VOLVOXAI_ENGINE_DYNAMIC_SHAPE_STATS_INIT;
    char* signature = NULL;
    size_t resolved_count = 0;
    double started_ms = vx_report_now_ms();
    VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
    VxStatus failure_status = VX_STATUS_OK;
    int result = 0;
    volvoxai_engine_tensor_name_index_invalidate();
    volvoxai_engine_tensor_name_index_rebuild();
    if (staged->count)
        engine_inputs = (VolvoxAIEngineInputBinding*)calloc(
            staged->count, sizeof(*engine_inputs));
    if (staged->count && !engine_inputs) {
        failure_status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    for (size_t index = 0; index < staged->count; index++) {
        engine_inputs[index].name = staged->bindings[index].name;
        engine_inputs[index].dtype = (int)staged->bindings[index].dtype;
        engine_inputs[index].data = staged->bindings[index].data;
        engine_inputs[index].byte_size = staged->bindings[index].byte_size;
    }
    if (!vx_builtin_dynamic_shape_backend(context->compiled->backend)) {
        failure_status = vx_native_gpu_preflight_public_values(
            context, staged, NULL, 0u);
        if (failure_status != VX_STATUS_OK) goto done;
        result = volvoxai_engine_commit_input_bindings(
            engine_inputs, staged->count);
        goto done;
    }
    if (kind == VX_CONTEXT_RUN_DECODE_STEP) {
        failure_status = vx_native_gpu_preflight_public_values(
            context, staged, NULL, 0u);
        if (failure_status != VX_STATUS_OK) goto done;
        result = context->committed_shape_signature
            ? volvoxai_engine_commit_input_bindings(
                  engine_inputs, staged->count)
            : -1;
        if (result == 0) stats = context->dynamic_shape_stats;
        stats.plan_cache_hit = 1;
        goto done;
    }
    if (!context->logical_tensor_count) {
        result = volvoxai_engine_commit_input_bindings(
            engine_inputs, staged->count);
        if (result == 0) stats.plan_cache_hit = 1;
        goto done;
    }
    {
        VxStatus plan_status = vx_resolved_plan_create(
            context, resolved_input_shapes, &resolved,
            &resolved_count, &signature);
        if (plan_status != VX_STATUS_OK) {
            if (context->engine_state->debug)
                vx_engine_log(
                        "[debug] dynamic shape resolution failed status=%d tensors=%zu\n",
                        (int)plan_status, context->logical_tensor_count);
            failure_status = plan_status;
            goto done;
        }
    }
    if (required_shape_signature &&
        strcmp(required_shape_signature, signature)) {
        result = -4;
        goto done;
    }
    failure_status = vx_native_gpu_preflight_public_values(
        context, staged, resolved, resolved_count);
    if (failure_status != VX_STATUS_OK) goto done;
    result = volvoxai_engine_commit_dynamic_shape(
        signature, resolved, resolved_count, engine_inputs, staged->count,
        &stats);
    if (result == 0) {
        free(context->committed_shape_signature);
        context->committed_shape_signature = signature;
        signature = NULL;
        context->dynamic_shape_stats = stats;
    }
done:
    if (result == 0)
        context->dynamic_shape_bind_time_ms =
            vx_report_now_ms() - started_ms;
    vx_engine_state_scope_leave(scope);
    free(engine_inputs);
    free(resolved);
    free(signature);
    return failure_status != VX_STATUS_OK ? failure_status :
        result == 0 ? VX_STATUS_OK :
        result == -2 ? VX_STATUS_OUT_OF_MEMORY :
        result == -3 ? VX_STATUS_BACKEND_UNSUPPORTED :
        result == -4 ? VX_STATUS_INVALID_ARGUMENT :
                       VX_STATUS_EXECUTION_FAILED;
}

#if VOLVOXAI_ENABLE_TRAINING
VxStatus vx_model_internal_bind_authoring_inputs(
        VxModel* model,
        VxEngineState* state,
        VxBackendKind backend,
        const VxTensorBinding* inputs,
        size_t input_count,
        const char* required_shape_signature,
        char** out_shape_signature,
        VolvoxAIEngineDynamicShapeStats* out_stats,
        VxReport* report) {
    VxCompiledModel compiled;
    VxExecutionContext context;
    VxStagedBindingBatch staged = {0};
    VxStatus status;
    const char* backend_name = vx_builtin_backend_name(backend);
    if (!model || !state || !backend_name || !out_shape_signature) {
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_TRAINER_INPUT, backend_name,
                       VX_CODE_INVALID_INPUT_BATCH,
                       "authoring input binding arguments are invalid");
    }
    *out_shape_signature = NULL;
    memset(&compiled, 0, sizeof(compiled));
    memset(&context, 0, sizeof(context));
    compiled.model = model;
    compiled.builtin_backend = backend;
    snprintf(compiled.backend, sizeof(compiled.backend), "%s", backend_name);
    context.compiled = &compiled;
    context.engine_state = state;
    status = vx_logical_tensors_load(
        model, &context.logical_tensors, &context.logical_tensor_count);
    if (status == VX_STATUS_OK)
        status = vx_logical_tensor_name_index_build(&context);
    if (status != VX_STATUS_OK) goto done;
    status = vx_stage_binding_batch(
        &context, inputs, input_count, 1, 0, &staged);
    if (status != VX_STATUS_OK) goto done;
    status = vx_commit_builtin_bindings(
        &context, &staged, staged.resolved_input_shapes,
        VX_CONTEXT_RUN_FORWARD, required_shape_signature);
    if (status == VX_STATUS_OK) {
        if (!context.committed_shape_signature) {
            status = VX_STATUS_INTERNAL;
            goto done;
        }
        *out_shape_signature = context.committed_shape_signature;
        if (out_stats) *out_stats = context.dynamic_shape_stats;
        context.committed_shape_signature = NULL;
    }
done:
    vx_staged_binding_batch_clear(&staged);
    vx_declared_outputs_free(context.logical_tensors,
                             context.logical_tensor_count);
    free(context.logical_tensor_name_slots);
    free(context.committed_shape_signature);
    if (status != VX_STATUS_OK) {
        if (state->activation_budget_exceeded)
            return vx_fail(report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_TRAINER_INPUT, backend_name,
                VX_CODE_ACTIVATION_CAPACITY_EXCEEDED, "input binding exceeds the configured activation capacity limit");
        vx_report_write(
            report, status, VX_STAGE_TRAINER_INPUT, backend_name,
            vx_builtin_device_identity(backend_name),
            status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY :
            status == VX_STATUS_BACKEND_UNSUPPORTED
                ? VX_CODE_BOUNDED_DOMAIN_UNSUPPORTED :
            status == VX_STATUS_INVALID_GRAPH ? VX_CODE_INVALID_GRAPH :
            status == VX_STATUS_INVALID_ARGUMENT && required_shape_signature
                ? VX_CODE_ACTIVATION_SIGNATURE_MISMATCH :
                VX_CODE_INVALID_INPUT_BATCH,
            status == VX_STATUS_INVALID_ARGUMENT && required_shape_signature
                ? "input shape changed inside an accumulation window"
                : status == VX_STATUS_INVALID_GRAPH
                    ? "resolved shape plan violates the canonical graph shape contract"
                : "input batch violates the complete logical shape contract",
            0);
    }
    if (report && status != VX_STATUS_OK) report->input_issue = staged.issue;
    return status;
}
#endif

static void vx_owned_output_clear(VxOwnedOutput* output) {
    if (!output) return;
    if (output->device_snapshot.ticket)
        output->device_snapshot.release(output->device_snapshot.ticket);
    free(output->name);
    free(output->data);
    memset(output, 0, sizeof(*output));
}

static void vx_result_destroy(VxResult* result) {
    vx_result_release(result->completion_parent);
    vx_result_budget_release(&result->result_budget);
    for (size_t index = 0; index < result->output_count; index++)
        vx_owned_output_clear(&result->outputs[index]);
    free(result->outputs);
    free(result->input_shapes);
    free(result);
}

static const VxDeclaredOutput* vx_model_output_descriptor(
    const VxModel* model,
    const char* name) {
    if (!model || !name) return NULL;
    for (size_t index = 0; index < model->output_count; index++)
        if (!strcmp(model->outputs[index].name, name))
            return &model->outputs[index];
    return NULL;
}

static int vx_result_symbol_extent(const VxResult* result,
                                   const char* symbol,
                                   int64_t* out_extent) {
    const VxModel* model = result->publishing_model;
    int found = 0;
    int64_t extent = 0;
    if (!model || !symbol || !result->input_shapes) return 0;
    for (size_t input = 0; input < model->input_count; input++) {
        for (uint32_t axis = 0; axis < model->inputs[input].rank; axis++) {
            int64_t candidate;
            if (!model->inputs[input].symbols[axis] ||
                strcmp(model->inputs[input].symbols[axis], symbol)) continue;
            candidate = result->input_shapes[
                input * VX_MAX_TENSOR_RANK + axis];
            if (!candidate || (found && candidate != extent)) return 0;
            extent = candidate;
            found = 1;
        }
    }
    if (found && out_extent) *out_extent = extent;
    return found;
}

static VxStatus vx_result_append(VxResult* result,
                                 const char* name,
                                 VxDataType dtype,
                                 const int64_t* shape,
                                 uint32_t rank,
                                 const void* data,
                                 size_t byte_size) {
    VxOwnedOutput* output;
    const VxDeclaredOutput* descriptor;
    size_t elements = 1u;
    size_t element_size;
    size_t expected_bytes;
    if (!result || !name || !name[0] ||
        rank > VX_MAX_TENSOR_RANK ||
        (rank && !shape))
        return VX_STATUS_INVALID_ARGUMENT;
    descriptor = vx_model_output_descriptor(result->publishing_model, name);
    if (!descriptor || !vx_execution_dtype_size(dtype) ||
        dtype != descriptor->dtype || rank != descriptor->rank)
        return VX_STATUS_INVALID_ARGUMENT;
    for (uint32_t axis = 0; axis < rank; axis++) {
        int64_t extent = shape[axis];
        int64_t resolved_symbol = 0;
        if (extent <= 0 || extent < descriptor->minimums[axis] ||
            extent > descriptor->maximums[axis] ||
            extent % descriptor->multiples[axis] != 0 ||
            (descriptor->kinds[axis] == VX_DIMENSION_FIXED &&
             (descriptor->symbols[axis] ||
              extent != descriptor->minimums[axis])) ||
            (descriptor->kinds[axis] == VX_DIMENSION_SYMBOLIC &&
             (!descriptor->symbols[axis] ||
              (vx_result_symbol_extent(result, descriptor->symbols[axis],
                                       &resolved_symbol) &&
               extent != resolved_symbol))) ||
            (uint64_t)extent > SIZE_MAX / elements)
            return VX_STATUS_INVALID_ARGUMENT;
        elements *= (size_t)extent;
    }
    element_size = vx_execution_dtype_size(dtype);
    if (!element_size || elements > SIZE_MAX / element_size)
        return VX_STATUS_INVALID_ARGUMENT;
    expected_bytes = elements * element_size;
    if (byte_size != expected_bytes) return VX_STATUS_INVALID_ARGUMENT;
    for (size_t index = 0; index < result->output_count; index++)
        if (!strcmp(result->outputs[index].name, name))
            return VX_STATUS_INVALID_ARGUMENT;
    if (result->output_count == result->output_capacity) {
        size_t next_capacity = result->output_capacity
            ? result->output_capacity * 2u : 4u;
        VxOwnedOutput* next = (VxOwnedOutput*)realloc(
            result->outputs, next_capacity * sizeof(*next));
        if (!next) return VX_STATUS_OUT_OF_MEMORY;
        memset(next + result->output_capacity, 0,
               (next_capacity - result->output_capacity) * sizeof(*next));
        result->outputs = next;
        result->output_capacity = next_capacity;
    }
    output = &result->outputs[result->output_count];
    output->name = vx_string_copy(name);
    if (byte_size) output->data = (unsigned char*)malloc(byte_size);
    if (!output->name || (byte_size && !output->data)) {
        vx_owned_output_clear(output);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    output->dtype = dtype;
    output->rank = rank;
    for (uint32_t axis = 0; axis < rank; axis++) output->shape[axis] = shape[axis];
    output->byte_size = byte_size;
    if (byte_size && data) memcpy(output->data, data, byte_size);
    if (UINT64_MAX - result->snapshot_bytes < byte_size) {
        vx_owned_output_clear(output);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    result->snapshot_bytes += byte_size;
    result->output_count++;
    return VX_STATUS_OK;
}

static VxStatus vx_provider_output_write(void* user_data,
                                         const char* name,
                                         VxDataType dtype,
                                         const int64_t* shape,
                                         uint32_t rank,
                                         const void* data,
                                         size_t byte_size) {
    VxResult* result = (VxResult*)user_data;
    VxStatus status;
    if (!result || (!data && byte_size)) return VX_STATUS_INVALID_ARGUMENT;
    if (result->output_sink_status != VX_STATUS_OK)
        return result->output_sink_status;
    status = vx_result_append(result, name, dtype, shape, rank, data,
                              byte_size);
    if (status != VX_STATUS_OK) result->output_sink_status = status;
    return status;
}

/* Allocate output storage and capture device copies before leaving Execute.
 * Readback tickets never reference mutable context memory after submission. */
static VxStatus vx_snapshot_cpu_outputs(VxExecutionContext* context,
                                        VxResult* result) {
    int count = volvoxai_engine_graph_output_count();
    if (count < 0) return VX_STATUS_INTERNAL;
    for (int index = 0; index < count; index++) {
        const char* name = volvoxai_engine_graph_output_name(index);
        long numel;
        int shape[8] = {0}, rank, dtype;
        size_t element_size;
        int64_t shape64[VX_MAX_TENSOR_RANK] = {0};
        if (!name || volvoxai_engine_tensor_info_ex(name, &numel, shape, &rank,
                &dtype, &element_size) != 0 || rank < 0 ||
            rank > (int)VX_MAX_TENSOR_RANK || numel < 0 || !element_size)
            return VX_STATUS_INTERNAL;
        VxDataType output_dtype = dtype == T_F16 ? VX_DTYPE_F32 : (VxDataType)dtype;
        if (dtype == T_F16) element_size = sizeof(float);
        if ((size_t)numel > SIZE_MAX / element_size) return VX_STATUS_OUT_OF_MEMORY;
        size_t bytes = (size_t)numel * element_size;
        for (int axis = 0; axis < rank; axis++) shape64[axis] = shape[axis];
        VxStatus status = vx_result_append(result, name, output_dtype, shape64,
                                           (uint32_t)rank, NULL, bytes);
        if (status != VX_STATUS_OK) return status;
        VxOwnedOutput* output = &result->outputs[result->output_count - 1u];
        int captured = bytes ? volvoxai_engine_snapshot_tensor(
            name, &output->device_snapshot) : 0;
        if (captured < 0) return VX_STATUS_EXECUTION_FAILED;
        if (captured > 0) {
            result->state = VX_RESULT_STATE_PENDING;
        } else if (bytes) {
            int copied = dtype == T_F16
                ? volvoxai_engine_copy_tensor_f32(name, (float*)output->data, numel)
                : volvoxai_engine_copy_tensor_raw(name, output->data, bytes);
            if (copied != 0) return VX_STATUS_EXECUTION_FAILED;
        }
    }
    (void)context;
    return VX_STATUS_OK;
}
