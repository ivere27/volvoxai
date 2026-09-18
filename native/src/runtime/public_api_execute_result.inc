/* Execution, decode stepping, adapter rebinding and result readback.
 *
 * Fragment of public_api.c, included where these definitions used to sit.
 * vx_execution_context_run() is the single funnel every public execute and
 * decode entry point reaches; the result accessors below it read the snapshot
 * that funnel published. */

#include "public_api_decode_feedback.inc"

static void vx_context_clear_decode_tracking(VxExecutionContext* context) {
    if (!context) return;
    vx_context_decode_cache_clear(context);
    context->decode_prefilled = 0;
    context->decode_position_known = 0;
    context->decode_has_successful_step = 0;
    context->decode_last_operation = 0;
    memset(context->decode_active_lengths, 0, context->decode_lanes * sizeof(*context->decode_active_lengths));
    memset(context->decode_parked, 0, context->decode_lanes * sizeof(*context->decode_parked));
    context->decode_cache_generation++;
    if (context->decode_input_shapes && context->compiled &&
        context->compiled->model) {
        memset(context->decode_input_shapes, 0,
               context->compiled->model->input_count *
                   VX_MAX_TENSOR_RANK * sizeof(int64_t));
    }
}

/* Prove a required row transaction before input bytes commit. The temporary
 * dirty roots describe the requested edit, while all geometry and weights
 * remain the immutable prefill binding. Device planning performs no commands. */
static VxStatus vx_context_preflight_decode_rows(VxExecutionContext* context,
    const int32_t* positions, const VxTensorBinding* inputs, size_t input_count, const T* required_output) {
    if (context->compiled->provider || context->decode_row_mode != VX_DECODE_ROW_REQUIRED ||
        (context->compiled->builtin_backend != VX_PORTABLE_BACKEND_KIND &&
         context->compiled->builtin_backend != VX_BACKEND_KIND_WEBGPU)) return VX_STATUS_OK;
    VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
    int took_model_lock = !volvoxai_engine_model_route_lease_active();
    if (took_model_lock) volvoxai_engine_model_lock();
    unsigned char* dirty = malloc((size_t)g_nt);
    if (!dirty) {
        if (took_model_lock) volvoxai_engine_model_unlock();
        vx_engine_state_scope_leave(scope);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    memcpy(dirty, g_dirty, (size_t)g_nt);
    VxDecodeRowSet saved_rows = g_decode_rows;
    int saved_lanes = g_decode_lanes, prepared_row = g_hybrid_prepared_row;
    VxStatus status = VX_STATUS_BACKEND_UNSUPPORTED;
    int row = 0;
    memset(&g_decode_rows, 0, sizeof(g_decode_rows));
    if (vx_paged_row_set_init_locked(&g_decode_rows, (int)context->decode_lanes, positions) != VX_DECODE_ROW_SET_OK) goto done;
    g_decode_lanes = (int)context->decode_lanes;
    for (uint32_t lane = 0; lane < context->decode_lanes; lane++) if (positions[lane] > row) row = positions[lane];
    memset(g_dirty, 0, (size_t)g_nt);
    if (input_count) {
        for (size_t i = 0; i < input_count; i++) vx_incremental_mark_tensor_locked(t_find(inputs[i].name));
    } else if (context->decode_default_inputs) {
        for (size_t i = 0; i < context->compiled->model->input_count; i++)
            if (context->decode_default_inputs[i]) vx_incremental_mark_tensor_locked(t_find(context->compiled->model->inputs[i].name));
    }
    if (required_output && !vx_incremental_tensor_dirty_before_node_locked(required_output, g_nn)) goto done;
    int accepted = vx_incremental_prepare_hybrid_row_locked(row);
    status = accepted == 1 ? VX_STATUS_OK : accepted < 0 ? VX_STATUS_EXECUTION_FAILED : VX_STATUS_BACKEND_UNSUPPORTED;
done:
    memcpy(g_dirty, dirty, (size_t)g_nt);
    free(dirty);
    vx_decode_row_set_dispose(&g_decode_rows);
    g_decode_rows = saved_rows; g_decode_lanes = saved_lanes; g_hybrid_prepared_row = prepared_row;
    if (took_model_lock) volvoxai_engine_model_unlock();
    vx_engine_state_scope_leave(scope);
    return status;
}

#include "public_api_decode_cache.inc"

static VxStatus vx_execution_context_run_lanes_with_budget(
                                         VxExecutionContext* context,
                                         VxContextRunKind kind,
                                         int32_t position,
                                         const int32_t* lane_positions,
                                         size_t lane_count,
                                         const VxDecodeFeedbackOptions* feedback_options,
                                         const VxTensorRunOptions* tensor_options,
                                         int dependency_step,
                                         const VxTensorBinding* inputs,
                                         size_t input_count,
                                         VxResultBudgetTicket* reserved_budget,
                                         VxResult** out_result,
                                         VxReport* report) {
    VxContextOperation operation;
    VxStagedBindingBatch staged = {0};
    VxContextCallSequenceEvidence fixed_selection = {0};
    VxResult* result;
    VxStatus status;
    VxResultBudgetTicket result_budget = {0};
    uint64_t execution_id;
    VxReport execution_report = VX_REPORT_INIT;
    double execution_started_ms;
    int attestation_failure = 0;
    int fallback_forbidden_failure = 0;
    int portable_fixed_step = 0;
    int portable_selection_installed = 0;
    int builtin_binding_committed = 0;
    int32_t* resolved_positions;
    VxDecodeFeedback feedback = {0};
    VxCacheTransaction cache_transaction = {0};
    const int decode_operation = kind == VX_CONTEXT_RUN_DECODE_PREFILL ||
        kind == VX_CONTEXT_RUN_DECODE_STEP;
    VxStage stage = decode_operation ? VX_STAGE_DECODE : VX_STAGE_EXECUTE;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!out_result)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, stage,
                       context ? context->compiled->backend : NULL,
                       VX_CODE_INVALID_RESULT_POINTER, "result output pointer is NULL");
    *out_result = NULL;
    status = vx_context_operation_begin(context, &operation, stage, report);
    if (status != VX_STATUS_OK) return status;
    if (kind == VX_CONTEXT_RUN_TENSORS &&
        (context->compiled->provider ||
         !vx_native_tensor_ops(context->compiled->builtin_backend) ||
         (context->compiled->builtin_backend != VX_PORTABLE_BACKEND_KIND &&
          context->compiled->operator_fallback != VX_OPERATOR_FALLBACK_FORBID))) {
        status = vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, stage,
            context->compiled->backend, VX_CODE_BACKEND_UNSUPPORTED,
            "retained tensors require a native buffer backend with GPU operator fallback forbidden");
        vx_context_operation_end(context, &operation);
        return status;
    }
    resolved_positions = context->decode_positions;
    if (dependency_step && (!context->decode_prefilled || context->decode_lanes != 1 ||
        context->decode_cache || context->decode_row_mode != VX_DECODE_ROW_AUTO)) {
        status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, stage, context->compiled->backend,
            VX_CODE_INVALID_DECODE_DEPENDENCY, "dependency updates require a prefilled single-lane AUTO context without paged KV");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_context_decode_state(report, context);
        vx_context_operation_end(context, &operation);
        return status;
    }
    if (decode_operation && !dependency_step && context->decode_row_mode != VX_DECODE_ROW_DISABLED) {
        int valid = context->decode_lanes >= 1 && context->decode_lanes <= INT32_MAX;
        int advancing = 0;
        if ((lane_count && (!lane_positions || lane_count != context->decode_lanes)) ||
            (!lane_count && context->decode_lanes > 1 && position >= 0) ||
            (context->compiled->provider && context->decode_lanes > 1)) valid = 0;
        int32_t representative = 0;
        for (uint32_t lane = 0; valid && lane < context->decode_lanes; lane++) {
            int32_t requested = lane_count ? lane_positions[lane] : position;
            if (kind == VX_CONTEXT_RUN_DECODE_PREFILL) {
                if (!lane_count && position == -1) requested = 0;
                if (requested < 0 || requested == INT32_MAX) valid = 0;
            } else if (!lane_count && position == -1) {
                requested = (int32_t)context->decode_active_lengths[lane];
                if (requested < 0 || (!context->decode_cache && requested == 0) || requested == INT32_MAX) valid = 0;
                advancing++;
            } else if (requested == -2) {
                if (!context->decode_active_lengths[lane]) valid = 0;
                requested = (int32_t)context->decode_active_lengths[lane] - 1;
            } else if (requested != -1) {
                if (requested < 0 || (!context->decode_cache && requested == 0) ||
                    (uint32_t)requested != context->decode_active_lengths[lane] ||
                    requested == INT32_MAX) valid = 0;
                advancing++;
            }
            resolved_positions[lane] = requested;
            if (requested > representative) representative = requested;
        }
        if (kind == VX_CONTEXT_RUN_DECODE_STEP && (!advancing || !context->decode_prefilled)) valid = 0;
        if (!valid) {
            status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, stage, context->compiled->backend,
                VX_CODE_INVALID_DECODE_POSITION, "decode cursor must match each lane's next position; idle/parked lanes do not advance");
            vx_report_set_lineage(report, NULL, NULL, NULL, context);
            vx_report_set_context_decode_state(report, context);
            vx_context_operation_end(context, &operation);
            return status;
        }
        position = representative;
    }
    portable_fixed_step = !context->compiled->provider &&
        context->compiled->builtin_backend == VX_PORTABLE_BACKEND_KIND &&
        kind == VX_CONTEXT_RUN_DECODE_STEP;
    if (decode_operation &&
        context->decode_row_mode == VX_DECODE_ROW_DISABLED) {
        status = vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, stage,
                         context->compiled->backend,
                         VX_CODE_DECODE_NOT_ENABLED,
                         "execution context was not created for decode");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        vx_context_operation_end(context, &operation);
        return status;
    }
    if (kind == VX_CONTEXT_RUN_DECODE_STEP && position < -1) {
        status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, stage,
                         context->compiled->backend,
                         VX_CODE_INVALID_DECODE_POSITION,
                         "decode position must be -1 or non-negative");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        if (decode_operation)
            vx_report_set_context_decode_state(report, context);
        vx_context_operation_end(context, &operation);
        return status;
    }
    if (kind == VX_CONTEXT_RUN_DECODE_PREFILL && position < 0) {
        status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, stage,
                         context->compiled->backend,
                         VX_CODE_INVALID_DECODE_POSITION,
                         "decode prefill position must be non-negative");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        vx_report_set_context_decode_state(report, context);
        vx_context_operation_end(context, &operation);
        return status;
    }
    if (portable_fixed_step &&
        context->decode_row_mode == VX_DECODE_ROW_REQUIRED && position < 0) {
        status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, stage,
                         context->compiled->backend,
                         VX_CODE_INVALID_DECODE_POSITION,
                         "required row decode needs a non-negative position");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        vx_report_set_context_decode_state(report, context);
        vx_context_operation_end(context, &operation);
        return status;
    }
    if (kind == VX_CONTEXT_RUN_PREFIX && position <= 0) {
        status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, stage,
                         context->compiled->backend,
                         VX_CODE_INVALID_PREFIX_ROW_COUNT,
                         "prefix row count must be positive");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        vx_context_operation_end(context, &operation);
        return status;
    }
    if (feedback_options) {
        status = vx_context_feedback_describe(context, feedback_options, position, &feedback);
        if (status != VX_STATUS_OK) {
            vx_fail(report, status, stage, context->compiled->backend, VX_CODE_INVALID_DECODE_FEEDBACK,
                "generation requires a current single-lane row cache, matching embedding/argmax vocabulary and an in-bounds token range");
            vx_report_set_lineage(report, NULL, NULL, NULL, context);
            vx_report_set_context_decode_state(report, context);
            vx_context_operation_end(context, &operation);
            return status;
        }
    }
    status = kind == VX_CONTEXT_RUN_TENSORS
        ? vx_stage_tensor_batch(context, inputs, input_count, tensor_options, &staged)
        : vx_stage_binding_batch(
        context, inputs, input_count,
        kind != VX_CONTEXT_RUN_DECODE_STEP,
        kind == VX_CONTEXT_RUN_DECODE_STEP,
        kind == VX_CONTEXT_RUN_TENSORS, &staged);
    if (status != VX_STATUS_OK) {
        status = vx_fail(report, status, stage, context->compiled->backend,
                         status == VX_STATUS_OUT_OF_MEMORY ?
                             VX_CODE_OUT_OF_MEMORY : VX_CODE_INVALID_INPUT_BATCH,
                         status == VX_STATUS_OUT_OF_MEMORY ?
                             "input batch staging allocation failed" :
                             "input batch violates the complete logical shape contract");
        if (report) report->input_issue = staged.issue;
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        if (decode_operation)
            vx_report_set_context_decode_state(report, context);
        vx_context_operation_end(context, &operation);
        return status;
    }
    if (kind == VX_CONTEXT_RUN_DECODE_PREFILL && context->decode_default_inputs) {
        int valid = 1;
        for (size_t i = 0; valid && i < context->compiled->model->input_count; i++) {
            if (!context->decode_default_inputs[i]) continue;
            const VxDeclaredTensor* tensor = &context->compiled->model->inputs[i];
            uint32_t axis = context->decode_lanes > 1 ? 1u : 0u;
            if (context->decode_lanes == 1)
                while (axis < tensor->rank && tensor->maximums[axis] == 1) axis++;
            if (axis >= tensor->rank) continue;
            int64_t capacity = staged.resolved_input_shapes[i * VX_MAX_TENSOR_RANK + axis];
            if (context->decode_lanes > 1 && staged.resolved_input_shapes[i * VX_MAX_TENSOR_RANK] != context->decode_lanes) valid = 0;
            for (uint32_t lane = 0; lane < context->decode_lanes; lane++)
                if (resolved_positions[lane] >= capacity) valid = 0;
        }
        if (!valid) {
            vx_staged_binding_batch_clear(&staged);
            status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, stage, context->compiled->backend,
                VX_CODE_INVALID_DECODE_POSITION, "prompt position or lane count exceeds the supplied decode input geometry");
            vx_report_set_lineage(report, NULL, NULL, NULL, context);
            vx_report_set_context_decode_state(report, context);
            vx_context_operation_end(context, &operation);
            return status;
        }
    }
    if (portable_fixed_step) {
        if (!vx_context_call_sequence_engine_prefilled(context)) {
            status = vx_fail(
                report, VX_STATUS_EXECUTION_FAILED, stage,
                context->compiled->backend,
                VX_CODE_DECODE_PREFILL_REQUIRED,
                "decode cache is not prefilled for a fixed dependency step");
            vx_report_set_lineage(report, NULL, NULL, NULL, context);
            vx_report_set_plan_evidence(report, context->compiled);
            vx_report_set_context_decode_state(report, context);
            vx_staged_binding_batch_clear(&staged);
            vx_context_operation_end(context, &operation);
            return status;
        }
        status = vx_context_call_sequence_prepare_fixed(
            context, feedback_options ? feedback.roots : staged.bindings,
            feedback_options ? 2 : staged.count, !dependency_step, &fixed_selection);
        if (status != VX_STATUS_OK) {
            status = vx_fail(
                report, status, stage, context->compiled->backend,
                status == VX_STATUS_OUT_OF_MEMORY
                    ? VX_CODE_OUT_OF_MEMORY : VX_CODE_CALL_SEQUENCE_POLICY_INVALID,
                status == VX_STATUS_OUT_OF_MEMORY
                    ? "fixed call-sequence evidence allocation failed"
                    : "fixed call-sequence compile/replay validation failed");
            vx_report_set_lineage(report, NULL, NULL, NULL, context);
            vx_report_set_plan_evidence(report, context->compiled);
            vx_report_set_context_decode_state(report, context);
            vx_staged_binding_batch_clear(&staged);
            vx_context_operation_end(context, &operation);
            return status;
        }
        status = vx_context_call_sequence_install_fixed(
            context, &fixed_selection);
        vx_context_call_sequence_clear(&fixed_selection);
        if (status != VX_STATUS_OK) {
            status = vx_fail(
                report, status, stage, context->compiled->backend,
                VX_CODE_CALL_SEQUENCE_SELECTION_INVALID,
                "fixed call-sequence selection cannot bind to the current CPU cache");
            vx_report_set_lineage(report, NULL, NULL, NULL, context);
            vx_report_set_plan_evidence(report, context->compiled);
            vx_report_set_context_decode_state(report, context);
            vx_staged_binding_batch_clear(&staged);
            vx_context_operation_end(context, &operation);
            return status;
        }
        portable_selection_installed = 1;
    }
    if (kind == VX_CONTEXT_RUN_DECODE_STEP && !dependency_step) {
        status = vx_cache_transaction_reserve(context, resolved_positions,
            feedback_options ? feedback.end : 0, &cache_transaction);
        if (status == VX_STATUS_OK) status = vx_context_preflight_decode_rows(context, resolved_positions,
            feedback_options ? feedback.roots : staged.bindings, feedback_options ? 2 : staged.count,
            feedback_options ? feedback.output : NULL);
        if (status == VX_STATUS_OK && feedback_options && feedback.end - 1 != position) {
            resolved_positions[0] = feedback.end - 1;
            status = vx_context_preflight_decode_rows(context, resolved_positions, feedback.roots, 2, feedback.output);
            resolved_positions[0] = position;
        }
        if (status != VX_STATUS_OK) {
            vx_cache_transaction_finish(&cache_transaction, 0);
            if (portable_selection_installed) vx_context_call_sequence_uninstall_fixed(context);
            vx_context_call_sequence_clear(&fixed_selection);
            vx_staged_binding_batch_clear(&staged);
            vx_fail(report, status, stage, context->compiled->backend,
                VX_CODE_DECODE_ROW_DOMAIN_UNSUPPORTED, "decode cursor or changed input closure is outside the retained row domain");
            vx_report_set_lineage(report, NULL, NULL, NULL, context);
            vx_report_set_context_decode_state(report, context);
            vx_context_operation_end(context, &operation);
            return status;
        }
    }
    if (reserved_budget && reserved_budget->reserved)
        vx_result_budget_move(&result_budget, reserved_budget);
    else
        status = vx_result_budget_reserve_selected(context->compiled,
                                          &result_budget, tensor_options);
    if (status != VX_STATUS_OK) {
        vx_cache_transaction_finish(&cache_transaction, 0);
        if (portable_selection_installed) {
            vx_context_call_sequence_uninstall_fixed(context);
            portable_selection_installed = 0;
        }
        vx_context_call_sequence_clear(&fixed_selection);
        vx_staged_binding_batch_clear(&staged);
        vx_context_operation_end(context, &operation);
        status = vx_fail(
            report, status, stage, context->compiled->backend,
            status == VX_STATUS_OVERLOADED ? VX_CODE_OVERLOADED :
                                             VX_CODE_RESULT_BUDGET_INVALID,
            status == VX_STATUS_OVERLOADED
                ? "Runtime unconsumed-result slot/byte budget is full"
                : "compiled maximum output bytes cannot be reserved");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        return status;
    }
    execution_started_ms = vx_report_now_ms();
    result = (VxResult*)calloc(1, sizeof(*result));
    if (!result) {
        vx_cache_transaction_finish(&cache_transaction, 0);
        if (portable_selection_installed) {
            vx_context_call_sequence_uninstall_fixed(context);
            portable_selection_installed = 0;
        }
        vx_context_call_sequence_clear(&fixed_selection);
        vx_result_budget_release(&result_budget);
        vx_staged_binding_batch_clear(&staged);
        vx_context_operation_end(context, &operation);
        vx_report_write(&execution_report, VX_STATUS_OUT_OF_MEMORY,
                        stage, context->compiled->backend,
                        context->compiled->provider ? NULL :
                            vx_builtin_device_identity(context->compiled->backend),
                        VX_CODE_OUT_OF_MEMORY, "result allocation failed", 0);
        vx_report_set_lineage(&execution_report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(&execution_report, context->compiled);
        execution_report.execution_time_ms =
            vx_report_now_ms() - execution_started_ms;
        vx_report_publish(report, &execution_report);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    atomic_init(&result->references, 1);
    result->state = VX_RESULT_STATE_READY;
    if (kind == VX_CONTEXT_RUN_TENSORS) {
        result->native_tensors = 1;
    }
    vx_result_budget_move(&result->result_budget, &result_budget);
    result->publishing_model = context->compiled->model;
    result->input_shapes = staged.resolved_input_shapes;
    staged.resolved_input_shapes = NULL;
    execution_id = vx_next_execution_identity(
        context->compiled->model->runtime);
    result->execution_id = execution_id;
    execution_report = context->compiled->report;
    execution_report.struct_size = sizeof(execution_report);
    execution_report.execution_id = execution_id;
    execution_report.stage = stage;
    execution_report.compile_time_ms = context->compiled->report.compile_time_ms;
    execution_report.execution_time_ms = 0.0;
    execution_report.offending_node[0] = '\0';
    if (context->compiled->provider) {
        VxBackendOutputSink sink = {
            .struct_size = sizeof(sink),
            .user_data = result,
            .write = vx_provider_output_write,
        };
        if (kind == VX_CONTEXT_RUN_FORWARD) {
            status = context->compiled->provider->context_execute(
                context->provider_context, staged.bindings, staged.count,
                &sink, &execution_report);
        } else if (kind == VX_CONTEXT_RUN_DECODE_PREFILL &&
                   context->compiled->provider->context_decode_prefill) {
            status = context->compiled->provider->context_decode_prefill(
                context->provider_context, position,
                staged.bindings, staged.count,
                &sink, &execution_report);
        } else if (kind == VX_CONTEXT_RUN_DECODE_STEP &&
                   context->compiled->provider->context_decode_step) {
            status = context->compiled->provider->context_decode_step(
                context->provider_context, position,
                staged.bindings, staged.count, &sink, &execution_report);
        } else {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
        }
        vx_report_normalize_provider_output(&execution_report);
        if (result->output_sink_status != VX_STATUS_OK)
            status = result->output_sink_status == VX_STATUS_OUT_OF_MEMORY
                ? VX_STATUS_OUT_OF_MEMORY : VX_STATUS_EXECUTION_FAILED;
        if (status == VX_STATUS_OK) {
            attestation_failure = !execution_report.route_attested ||
                                  !execution_report.route_evidence[0];
            fallback_forbidden_failure =
                context->compiled->operator_fallback ==
                    VX_OPERATOR_FALLBACK_FORBID &&
                execution_report.operator_fallback_used;
            if (fallback_forbidden_failure)
                status = VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN;
            else if (attestation_failure)
                status = VX_STATUS_EXECUTION_FAILED;
        }
    } else {
        VxEngineStateScope scope;
        VxStatus binding_status = VX_STATUS_OK;
        if (kind != VX_CONTEXT_RUN_DECODE_STEP && context->decode_cache)
            vx_context_clear_decode_tracking(context);
        if (binding_status == VX_STATUS_OK) {
            binding_status = vx_commit_builtin_bindings(
                context, &staged, result->input_shapes, kind, NULL, 0);
            builtin_binding_committed = binding_status == VX_STATUS_OK;
        }
        int forward_status = binding_status == VX_STATUS_OK ? 0 : -1;
        int adapter_route_active = 0;
        scope = vx_engine_state_scope_enter(context->engine_state);
        if (forward_status == 0 && vx_cache_transaction_prepare(&cache_transaction))
            forward_status = -1;
        if (forward_status == 0) {
            volvoxai_engine_tensor_name_index_invalidate();
            volvoxai_engine_tensor_name_index_rebuild();
            if (kind == VX_CONTEXT_RUN_DECODE_STEP && !dependency_step && !feedback_options && !input_count && context->decode_default_inputs) {
                for (size_t i = 0; i < context->compiled->model->input_count; i++)
                    if (context->decode_default_inputs[i])
                        vx_incremental_mark_tensor_locked(t_find(context->compiled->model->inputs[i].name));
            }
            if (context->adapter_route_version[0]) {
                adapter_route_active = volvoxai_engine_adapter_route_begin(
                    context->adapter_route_version) == 0;
                if (!adapter_route_active) forward_status = -1;
            }
        }
        if (forward_status == 0 && kind == VX_CONTEXT_RUN_DECODE_PREFILL)
            forward_status = volvoxai_engine_decode_session_prefill(
                context->decode_session);
        else if (forward_status == 0 && feedback_options) {
            forward_status = vx_context_feedback_run(context, &feedback, resolved_positions);
            position = resolved_positions[0];
        } else if (forward_status == 0 && kind == VX_CONTEXT_RUN_DECODE_STEP)
            forward_status = dependency_step
                ? volvoxai_engine_decode_session_step(context->decode_session, -1)
                : volvoxai_engine_decode_session_step_rows(
                context->decode_session, resolved_positions, (int)context->decode_lanes);
        else if (forward_status == 0 && kind == VX_CONTEXT_RUN_PREFIX)
            forward_status = volvoxai_engine_forward_prefix(position);
        else if (forward_status == 0)
            forward_status = volvoxai_engine_forward();
        status = forward_status == 0 ? vx_snapshot_cpu_outputs(context, result, tensor_options) :
                 binding_status != VX_STATUS_OK ? binding_status :
                                                   VX_STATUS_EXECUTION_FAILED;
        if (adapter_route_active) volvoxai_engine_adapter_route_end();
        if (portable_selection_installed) {
            volvoxai_engine_model_lock();
            vx_incremental_selection_clear_locked();
            volvoxai_engine_model_unlock();
            portable_selection_installed = 0;
        }
        if (context->decode_session)
            context->decode_prefilled =
                volvoxai_engine_decode_session_prefilled(
                    context->decode_session);
        if (decode_operation)
            vx_report_set_builtin_decode_state(&execution_report, context);
        vx_engine_state_scope_leave(scope);
        vx_report_set_builtin_route(&execution_report, context->engine_state,
                                    context->compiled->backend);
        vx_report_append_dynamic_shape(&execution_report, context);
        if (status == VX_STATUS_OK &&
            !vx_report_prepend_backend_execution_evidence(
                &execution_report, context)) {
            attestation_failure = 1;
            status = VX_STATUS_EXECUTION_FAILED;
        }
        if (status == VX_STATUS_OK) {
            attestation_failure = !execution_report.route_attested ||
                                  !execution_report.route_evidence[0];
            fallback_forbidden_failure =
                context->compiled->operator_fallback ==
                    VX_OPERATOR_FALLBACK_FORBID &&
                execution_report.operator_fallback_used;
            if (fallback_forbidden_failure)
                status = VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN;
            else if (attestation_failure)
                status = VX_STATUS_EXECUTION_FAILED;
        }
        if (status != VX_STATUS_OK)
            vx_report_set_builtin_failure_node(&execution_report,
                                               context->engine_state);
    }
    vx_context_call_sequence_clear(&fixed_selection);
    if (status == VX_STATUS_OK &&
        result->output_count != (tensor_options && tensor_options->select_outputs
            ? tensor_options->output_count : context->compiled->model->output_count))
        status = VX_STATUS_EXECUTION_FAILED;
    if (status == VX_STATUS_OK)
        status = vx_result_budget_shrink(&result->result_budget,
                                         result->snapshot_bytes);
    vx_cache_transaction_finish(&cache_transaction, status == VX_STATUS_OK);
    /* This check deliberately follows every status-producing boundary,
     * including route attestation, output-count verification, and result
     * budget shrink. Once input bytes commit, no rejected call may leave a
     * decode prefill that combines those roots with older cached outputs. */
    if (status != VX_STATUS_OK && builtin_binding_committed &&
        context->decode_session) {
        VxEngineStateScope reset_scope =
            vx_engine_state_scope_enter(context->engine_state);
        (void)volvoxai_engine_decode_session_reset(context->decode_session);
        context->decode_prefilled =
            volvoxai_engine_decode_session_prefilled(
                context->decode_session);
        vx_context_clear_decode_tracking(context);
        if (decode_operation)
            vx_report_set_builtin_decode_state(&execution_report, context);
        vx_engine_state_scope_leave(reset_scope);
    }
    if (status != VX_STATUS_OK) {
        execution_report.status = status;
        execution_report.stage = stage;
        execution_report.execution_id = execution_id;
        snprintf(execution_report.backend, sizeof(execution_report.backend), "%s",
                 context->compiled->backend);
        if (!context->compiled->provider)
            snprintf(execution_report.device, sizeof(execution_report.device),
                     "%s", vx_builtin_device_identity(
                         context->compiled->backend));
        if (status == VX_STATUS_OUT_OF_MEMORY) {
            execution_report.code = VX_CODE_OUT_OF_MEMORY;
            snprintf(execution_report.message,
                     sizeof(execution_report.message), "%s",
                     "resolved shape plan or arena growth allocation failed");
        } else if (status == VX_STATUS_INVALID_ARGUMENT) {
            execution_report.code = VX_CODE_INVALID_INPUT_VALUES;
            snprintf(execution_report.message,
                     sizeof(execution_report.message), "%s",
                     "input values violate a graph-wide native GPU preflight predicate");
        } else if (status == VX_STATUS_INVALID_GRAPH) {
            execution_report.code = VX_CODE_INVALID_GRAPH;
            snprintf(execution_report.message,
                     sizeof(execution_report.message), "%s",
                     "resolved shape plan violates the canonical graph shape contract");
        } else if (status == VX_STATUS_BACKEND_UNSUPPORTED) {
            execution_report.code = VX_CODE_BACKEND_UNSUPPORTED;
            snprintf(execution_report.message,
                     sizeof(execution_report.message), "%s",
                     "backend does not implement the requested execution operation");
        } else if (status == VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN) {
            execution_report.code = VX_CODE_OPERATOR_FALLBACK_FORBIDDEN;
            snprintf(execution_report.message,
                     sizeof(execution_report.message), "%s",
                     "execution used an operator fallback forbidden by the compiled policy");
        } else {
            execution_report.code = VX_CODE_EXECUTION_FAILED;
            snprintf(execution_report.message,
                     sizeof(execution_report.message), "%s",
                     attestation_failure
                         ? "execution route did not provide its compiled attestation"
                         : "execution or output snapshot failed");
        }
        if (decode_operation &&
            context->compiled->provider && !execution_report.decode_state[0])
            snprintf(execution_report.decode_state,
                     sizeof(execution_report.decode_state), "%s",
                     "enabled=1;prefilled=unknown;mode=provider;last=failed");
        vx_report_set_lineage(&execution_report, NULL, NULL, NULL, context);
        execution_report.result_bytes = result->snapshot_bytes;
        execution_report.execution_time_ms =
            vx_report_now_ms() - execution_started_ms;
        vx_report_publish(report, &execution_report);
        vx_staged_binding_batch_clear(&staged);
        vx_result_destroy(result);
        vx_context_operation_end(context, &operation);
        return status;
    }
    if (kind == VX_CONTEXT_RUN_DECODE_PREFILL) {
        if (context->compiled->model->input_count)
            memcpy(context->decode_input_shapes, result->input_shapes,
                   context->compiled->model->input_count *
                       VX_MAX_TENSOR_RANK * sizeof(int64_t));
        context->decode_prefilled = 1;
        context->decode_position_known = 1;
        context->decode_has_successful_step = 0;
        context->decode_last_operation = 1;
        context->decode_last_position = position;
        context->decode_cache_generation++;
        for (uint32_t lane = 0; lane < context->decode_lanes; lane++) {
            context->decode_active_lengths[lane] = (uint32_t)resolved_positions[lane] + 1u;
            context->decode_parked[lane] = 0;
        }
    } else if (dependency_step) {
        context->decode_position_known = 0;
        context->decode_has_successful_step = 0;
        context->decode_last_operation = 2;
    } else if (kind == VX_CONTEXT_RUN_DECODE_STEP) {
        for (uint32_t lane = 0; lane < context->decode_lanes; lane++) {
            context->decode_parked[lane] = resolved_positions[lane] == -1;
            if (resolved_positions[lane] >= 0)
                context->decode_active_lengths[lane] = (uint32_t)resolved_positions[lane] + 1u;
        }
        if (position >= 0) {
            context->decode_last_position = position;
            context->decode_position_known = 1;
        } else if (context->decode_position_known &&
                   context->decode_last_position < INT32_MAX) {
            context->decode_last_position++;
        } else {
            context->decode_position_known = 0;
        }
        context->decode_has_successful_step = context->decode_position_known;
        context->decode_last_operation = 2;
    }
    *out_result = result;
    context->native_execution_valid = kind == VX_CONTEXT_RUN_TENSORS;
    execution_report.status = VX_STATUS_OK;
    execution_report.stage = stage;
    execution_report.execution_id = execution_id;
    snprintf(execution_report.backend, sizeof(execution_report.backend), "%s",
             context->compiled->backend);
    if (!context->compiled->provider)
        snprintf(execution_report.device, sizeof(execution_report.device), "%s",
                 vx_builtin_device_identity(context->compiled->backend));
    execution_report.code = VX_CODE_NONE;
    snprintf(execution_report.message, sizeof(execution_report.message), "%s",
             !decode_operation ?
                 "execution completed with owned output snapshots" :
                 "decode operation completed with owned output snapshots");
    if (decode_operation)
        vx_report_set_context_decode_state(&execution_report, context);
    execution_report.offending_node[0] = '\0';
    vx_report_set_lineage(&execution_report, NULL, NULL, NULL, context);
    execution_report.result_bytes = result->snapshot_bytes;
    vx_report_add_result_allocation(&execution_report, result);
    execution_report.execution_time_ms = vx_report_now_ms() - execution_started_ms;
    vx_result_capture_evidence(result, &execution_report);
    vx_report_publish(report, &execution_report);
    vx_staged_binding_batch_clear(&staged);
    vx_context_operation_end(context, &operation);
    return VX_STATUS_OK;
}

static VxStatus vx_execution_context_run_with_budget(VxExecutionContext* context,
    VxContextRunKind kind, int32_t position, const VxTensorBinding* inputs,
    size_t input_count, VxResultBudgetTicket* budget, VxResult** result, VxReport* report) {
    return vx_execution_context_run_lanes_with_budget(context, kind, position,
        NULL, 0, NULL, NULL, kind == VX_CONTEXT_RUN_DECODE_STEP && position == -1,
        inputs, input_count, budget, result, report);
}

VxStatus vx_execution_context_decode_lanes(VxExecutionContext* context,
    int prefill, int32_t position, const int32_t* positions, size_t lane_count,
    const VxTensorBinding* inputs, size_t input_count, VxResult** result, VxReport* report) {
    return vx_execution_context_run_lanes_with_budget(context,
        prefill ? VX_CONTEXT_RUN_DECODE_PREFILL : VX_CONTEXT_RUN_DECODE_STEP,
        position, positions, lane_count, NULL, NULL, 0, inputs, input_count, NULL, result, report);
}

VxStatus vx_execution_context_decode_generate(VxExecutionContext* context,
    const VxDecodeFeedbackOptions* options, VxResult** result, VxReport* report) {
    if (!options) return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_DECODE, NULL,
        VX_CODE_INVALID_DECODE_FEEDBACK, "generation options are required");
    return vx_execution_context_run_lanes_with_budget(context, VX_CONTEXT_RUN_DECODE_STEP,
        -1, NULL, 0, options, NULL, 0, NULL, 0, NULL, result, report);
}

VxStatus vx_execution_context_inspect_decode(VxExecutionContext* context,
    VxDecodeStateWriter write, void* user, VxReport* report) {
    VxContextOperation operation;
    VxStatus status = vx_context_operation_begin(context, &operation, VX_STAGE_DECODE, report);
    if (status != VX_STATUS_OK) return status;
    VxDecodeStateView state = {context->decode_lanes, context->decode_prefilled,
        VX_DECODE_MODE_DISABLED, context->decode_active_lengths, context->decode_parked,
        context->decode_cache_generation};
    if (context->decode_row_mode != VX_DECODE_ROW_DISABLED) {
        if (context->compiled->provider) state.mode = VX_DECODE_MODE_PROVIDER;
        else {
            VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
            state.mode = volvoxai_engine_decode_session_mode(context->decode_session) == VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW
                ? VX_DECODE_MODE_ROW : VX_DECODE_MODE_DEPENDENCY;
            vx_engine_state_scope_leave(scope);
        }
    }
    status = write ? write(&state, user) : VX_STATUS_INVALID_ARGUMENT;
    if (status == VX_STATUS_OK) vx_report_write(report, status, VX_STAGE_DECODE,
        context->compiled->backend, NULL, VX_CODE_NONE, "decode state inspected", 0);
    else vx_fail(report, status, VX_STAGE_DECODE, context->compiled->backend,
        VX_CODE_DECODE_STATE_WRITE_FAILED, "decode state could not be returned");
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    vx_report_set_context_decode_state(report, context);
    vx_context_operation_end(context, &operation);
    return status;
}

VxStatus vx_execution_context_execute(VxExecutionContext* context,
                                      const VxTensorBinding* inputs,
                                      size_t input_count,
                                      VxResult** out_result,
                                      VxReport* report) {
    return vx_execution_context_run_with_budget(
        context, VX_CONTEXT_RUN_FORWARD, -1, inputs, input_count, NULL,
        out_result, report);
}

VxStatus vx_execution_context_execute_tensors(VxExecutionContext* context,
    const VxTensorBinding* inputs, size_t input_count, const VxTensorRunOptions* options,
    VxResult** result, VxReport* report) {
    return vx_execution_context_run_lanes_with_budget(context, VX_CONTEXT_RUN_TENSORS,
        -1, NULL, 0, NULL, options, 0, inputs, input_count, NULL, result, report);
}

VxStatus vx_execution_context_tensor_interop(VxExecutionContext* context,
    VxNativeBuffer* device, int64_t* stream, VxReport* report) {
    VxContextOperation operation;
    VxStatus status = vx_context_operation_begin(context, &operation, VX_STAGE_INPUT, report);
    if (status != VX_STATUS_OK) return status;
    memset(device, 0, sizeof(*device));
    *stream = 0;
    if (context->compiled->provider ||
        (context->compiled->builtin_backend != VX_PORTABLE_BACKEND_KIND &&
         context->compiled->operator_fallback != VX_OPERATOR_FALLBACK_FORBID)) {
        status = VX_STATUS_BACKEND_UNSUPPORTED;
    } else {
        status = vx_native_tensor_device(context->compiled->builtin_backend, device)
            ? VX_STATUS_OK : VX_STATUS_BACKEND_UNSUPPORTED;
        if (status == VX_STATUS_OK && device->kind == VX_NATIVE_BUFFER_CUDA) *stream = 1;
    }
    vx_report_write(report, status, VX_STAGE_INPUT, context->compiled->backend, NULL,
        status == VX_STATUS_OK ? VX_CODE_NONE : VX_CODE_BACKEND_UNSUPPORTED,
        status == VX_STATUS_OK ? "native dense tensor interoperation available" :
            "tensor interoperation requires a supported native backend", 0);
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    vx_context_operation_end(context, &operation);
    return status;
}

VxResult* vx_result_detach_tensor(VxResult* result, size_t index, VxNativeStorage** storage) {
    if (!result || !result->native_tensors || index >= result->output_count ||
        !result->outputs[index].native_storage) return NULL;
    VxResult* child = calloc(1, sizeof(*child));
    if (!child) return NULL;
    child->outputs = calloc(1, sizeof(*child->outputs));
    if (!child->outputs) { free(child); return NULL; }
    atomic_init(&child->references, 1);
    child->native_tensors = 1;
    child->state = result->state;
    child->completion_status = result->completion_status;
    child->evidence = result->evidence;
    child->output_count = child->output_capacity = 1;
    child->outputs[0] = result->outputs[index];
    child->snapshot_bytes = child->outputs[0].native_storage->buffer.length;
    if (result->result_budget.reserved) {
        child->result_budget = result->result_budget;
        child->result_budget.bytes = child->snapshot_bytes;
        child->result_budget.slot_reserved = 0;
        result->result_budget.bytes -= child->snapshot_bytes;
        vx_runtime_retain(child->result_budget.runtime);
    }
    memset(&result->outputs[index], 0, sizeof(result->outputs[index]));
    child->completion_parent = result;
    vx_result_retain(result);
    *storage = child->outputs[0].native_storage;
    return child;
}

VxStatus vx_result_tensor_view(VxResult* result, size_t index, VxTensorInfo* info,
    VxNativeBuffer* buffer, VxReport* report) {
    if (!result || !result->native_tensors || !buffer)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_READBACK,
            NULL, VX_CODE_INVALID_ARGUMENT, "result does not own retained tensor storage");
    VxStatus status = vx_result_output_info(result, index, info, report);
    if (status == VX_STATUS_OK) *buffer = result->outputs[index].native_storage->buffer;
    return status;
}

VxStatus vx_execution_context_execute_prefix(VxExecutionContext* context,
                                             int32_t row_count,
                                             const VxTensorBinding* inputs,
                                             size_t input_count,
                                             VxResult** out_result,
                                             VxReport* report) {
    return vx_execution_context_run_with_budget(
        context, VX_CONTEXT_RUN_PREFIX, row_count, inputs, input_count, NULL,
        out_result, report);
}

VxStatus vx_execution_context_decode_prefill(VxExecutionContext* context,
                                          int32_t position,
                                          const VxTensorBinding* inputs,
                                          size_t input_count,
                                          VxResult** out_result,
                                          VxReport* report) {
    return vx_execution_context_run_with_budget(
        context, VX_CONTEXT_RUN_DECODE_PREFILL, position, inputs, input_count, NULL,
        out_result, report);
}

VxStatus vx_execution_context_decode_step(VxExecutionContext* context,
                                          int32_t position,
                                          const VxTensorBinding* inputs,
                                          size_t input_count,
                                          VxResult** out_result,
                                          VxReport* report) {
    return vx_execution_context_run_with_budget(
        context, VX_CONTEXT_RUN_DECODE_STEP, position, inputs, input_count,
        NULL, out_result, report);
}

VxStatus vx_execution_context_decode_reset(VxExecutionContext* context,
                                           VxReport* report) {
    VxContextOperation operation;
    VxStatus status;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    status = vx_context_operation_begin(context, &operation, VX_STAGE_DECODE,
                                        report);
    if (status != VX_STATUS_OK) return status;
    if (context->decode_row_mode == VX_DECODE_ROW_DISABLED) {
        status = vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED,
                         VX_STAGE_DECODE, context->compiled->backend,
                         VX_CODE_DECODE_NOT_ENABLED,
                         "execution context was not created for decode");
    } else if (context->compiled->provider) {
        if (context->compiled->provider->context_decode_reset)
            status = context->compiled->provider->context_decode_reset(
                context->provider_context, report);
        else
            status = vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED,
                             VX_STAGE_DECODE, context->compiled->backend,
                             VX_CODE_BACKEND_UNSUPPORTED,
                             "backend does not implement decode reset");
        vx_report_normalize_provider_output(report);
        if (status == VX_STATUS_OK) {
            vx_report_write(report, status, VX_STAGE_DECODE,
                            context->compiled->backend, NULL, VX_CODE_NONE,
                            "decode state reset", 0);
            if (report && report->struct_size == sizeof(*report))
                snprintf(report->decode_state, sizeof(report->decode_state),
                         "%s",
                         "enabled=1;prefilled=0;mode=provider;last=none");
        }
    } else {
        VxEngineStateScope scope =
            vx_engine_state_scope_enter(context->engine_state);
        status = volvoxai_engine_decode_session_reset(context->decode_session) == 0
            ? VX_STATUS_OK : VX_STATUS_EXECUTION_FAILED;
        if (status == VX_STATUS_OK)
            vx_report_write(report, status, VX_STAGE_DECODE,
                            context->compiled->backend,
                            vx_builtin_device_identity(context->compiled->backend),
                            VX_CODE_NONE, "decode state reset", 0);
        else
            vx_fail(report, status, VX_STAGE_DECODE,
                    context->compiled->backend,
                    VX_CODE_EXECUTION_FAILED, "decode reset failed");
        vx_report_set_builtin_decode_state(report, context);
        vx_engine_state_scope_leave(scope);
    }
    if (status == VX_STATUS_OK)
        vx_context_clear_decode_tracking(context);
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    vx_report_set_plan_evidence(report, context->compiled);
    if (status == VX_STATUS_OK &&
        report && report->struct_size == sizeof(*report)) {
        if (context->compiled->provider)
            snprintf(report->decode_state, sizeof(report->decode_state), "%s",
                     "enabled=1;prefilled=0;mode=provider;last=none");
        else {
            VxEngineStateScope scope =
                vx_engine_state_scope_enter(context->engine_state);
            vx_report_set_builtin_decode_state(report, context);
            vx_engine_state_scope_leave(scope);
        }
    }
    vx_context_operation_end(context, &operation);
    return status;
}

static VxAdapterRevisionRecord* vx_model_find_adapter_revision(
    VxModel* model,
    uint64_t identity,
    uint64_t revision,
    int current) {
    VxAdapterRevisionRecord* selected = NULL;
    pthread_mutex_lock(&model->revision_mutex);
    if (current) {
        selected = atomic_load_explicit(&model->current_adapter,
                                        memory_order_acquire);
    } else {
        for (VxAdapterRevisionRecord* item = model->adapters;
             item; item = item->next) {
            if ((!identity && !revision && vx_adapter_is_base(item)) ||
                (item->identity == identity && item->revision == revision)) {
                selected = item;
                break;
            }
        }
    }
    vx_adapter_revision_retain(selected);
    pthread_mutex_unlock(&model->revision_mutex);
    return selected;
}

static VxStatus vx_context_select_adapter_record(
    VxExecutionContext* context,
    VxAdapterRevisionRecord* selected,
    VxReport* report) {
    VxStatus status;
    char prepared_route[96];
    if (!context->compiled->provider &&
        context->compiled->builtin_backend == VX_BACKEND_KIND_CUDA &&
        vx_context_has_dynamic_logical_domain(context) &&
        !vx_adapter_is_base(selected)) {
        VxStatus failure;
        vx_adapter_revision_release(selected);
        failure = vx_fail(
            report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_ADAPTER,
            context->compiled->backend,
            VX_CODE_DYNAMIC_CUDA_ADAPTER_UNSUPPORTED,
            "dynamic CUDA adapter selection requires a separately proved atomic residency reservation");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        return failure;
    }
    if (selected == context->adapter) {
        vx_adapter_revision_release(selected);
        vx_report_write(report, VX_STATUS_OK, VX_STAGE_ADAPTER,
                        context->compiled->backend,
                        context->compiled->provider ? NULL :
                            vx_builtin_device_identity(context->compiled->backend),
                        VX_CODE_NONE,
                        "adapter revision already selected", 0);
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        return VX_STATUS_OK;
    }
    status = vx_context_prepare_adapter(context, selected, prepared_route,
                                        sizeof(prepared_route), report);
    if (status != VX_STATUS_OK) {
        vx_adapter_revision_release(selected);
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        return status;
    }
    if (!context->compiled->provider && context->decode_session) {
        VxEngineStateScope scope =
            vx_engine_state_scope_enter(context->engine_state);
        int reset = volvoxai_engine_decode_session_reset(
            context->decode_session);
        vx_engine_state_scope_leave(scope);
        if (reset != 0) {
            VxStatus failure;
            vx_adapter_revision_release(selected);
            failure = vx_fail(
                report, VX_STATUS_EXECUTION_FAILED, VX_STAGE_ADAPTER,
                context->compiled->backend,
                VX_CODE_DECODE_RESET_FAILED,
                "decode state could not be reset before adapter selection");
            vx_report_set_lineage(report, NULL, NULL, NULL, context);
            vx_report_set_plan_evidence(report, context->compiled);
            return failure;
        }
    }
    if (context->decode_row_mode != VX_DECODE_ROW_DISABLED)
        vx_context_clear_decode_tracking(context);
    {
        VxAdapterRevisionRecord* previous = context->adapter;
        context->adapter = selected;
        snprintf(context->adapter_route_version,
                 sizeof(context->adapter_route_version), "%s",
                 prepared_route);
        vx_adapter_revision_release(previous);
    }
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_ADAPTER,
                    context->compiled->backend,
                    context->compiled->provider ? NULL :
                        vx_builtin_device_identity(context->compiled->backend),
                    VX_CODE_NONE,
                    "adapter revision selected", 0);
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    vx_report_set_plan_evidence(report, context->compiled);
    return status;
}

VxStatus vx_execution_context_select_adapter(
    VxExecutionContext* context,
    const VxAdapterRevision* revision,
    VxReport* report) {
    VxContextOperation operation;
    VxAdapterRevisionRecord* selected;
    VxStatus status;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (revision && (revision->struct_size != sizeof(*revision) ||
                     !revision->adapter_id || !revision->adapter_revision))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_ADAPTER,
                       context ? context->compiled->backend : NULL,
                       VX_CODE_INVALID_ADAPTER_REVISION,
                       "adapter revision key is invalid");
    status = vx_context_operation_begin(context, &operation,
                                        VX_STAGE_ADAPTER, report);
    if (status != VX_STATUS_OK) return status;
    selected = vx_model_find_adapter_revision(
        context->compiled->model,
        revision ? revision->adapter_id : 0,
        revision ? revision->adapter_revision : 0,
        0);
    if (!selected) {
        status = vx_fail(report, VX_STATUS_NOT_FOUND, VX_STAGE_ADAPTER,
                         context->compiled->backend,
                         revision ? VX_CODE_ADAPTER_REVISION_NOT_FOUND :
                                    VX_CODE_BASE_ADAPTER_MISSING,
                         revision
                             ? "adapter revision is not published by this model"
                             : "model has no immutable base adapter");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
    } else {
        status = vx_context_select_adapter_record(context, selected, report);
    }
    vx_context_operation_end(context, &operation);
    return status;
}

VxStatus vx_execution_context_rebind_adapter(
    VxExecutionContext* context,
    VxReport* report) {
    VxContextOperation operation;
    VxAdapterRevisionRecord* selected;
    VxStatus status;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    status = vx_context_operation_begin(context, &operation,
                                        VX_STAGE_ADAPTER, report);
    if (status != VX_STATUS_OK) return status;
    selected = vx_model_find_adapter_revision(context->compiled->model,
                                              0, 0, 1);
    if (!selected) {
        status = vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_ADAPTER,
                         context->compiled->backend,
                         VX_CODE_ADAPTER_REVISION_MISSING,
                         "model has no published adapter revision");
    } else {
        status = vx_context_select_adapter_record(context, selected, report);
    }
    vx_context_operation_end(context, &operation);
    return status;
}

void vx_result_retain(VxResult* result) {
    if (result) atomic_fetch_add_explicit(&result->references, 1, memory_order_relaxed);
}

void vx_result_release(VxResult* result) {
    if (!result || atomic_fetch_sub_explicit(&result->references, 1,
                                             memory_order_acq_rel) != 1) return;
    vx_result_destroy(result);
}

uint64_t vx_result_execution_id(const VxResult* result) {
    return result ? result->execution_id : 0;
}

VxResultState vx_result_state(const VxResult* result) {
    if (!result) return VX_RESULT_STATE_UNSPECIFIED;
    return result->state;
}

/* Only a threadless WASM device produces PENDING results. Native snapshots
 * are immutable on publication and this path never mutates them. */
VxStatus vx_result_poll(VxResult* result, VxReport* report) {
    if (!result || !vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (result->state == VX_RESULT_STATE_PENDING && result->completion_parent) {
        VxResult* parent = result->completion_parent;
        VxReport parent_report = VX_REPORT_INIT;
        VxStatus status = vx_result_poll(parent, &parent_report);
        if (parent->state == VX_RESULT_STATE_PENDING) {
            *report = parent_report;
            report->execution_id = result->execution_id;
            vx_report_set_result_evidence(report, result);
            return status;
        }
        if (status == VX_STATUS_OK) {
            /* Shapes and slice sizes were checked before publishing any lane.
             * No allocation or mutable context is needed to retire a batch. */
            for (size_t index = 0; index < result->output_count; index++) {
                const VxOwnedOutput* source = &parent->outputs[index];
                VxOwnedOutput* destination = &result->outputs[index];
                size_t inner_bytes = vx_execution_dtype_size(source->dtype);
                for (uint32_t axis = result->completion_batch_axis + 1u;
                     axis < source->rank; axis++)
                    inner_bytes *= (size_t)source->shape[axis];
                size_t outer = destination->byte_size / inner_bytes;
                for (size_t row = 0; row < outer; row++)
                    memcpy(destination->data + row * inner_bytes,
                           source->data + (row * result->completion_lane_count +
                                          result->completion_lane) * inner_bytes,
                           inner_bytes);
            }
            result->state = VX_RESULT_STATE_READY;
        } else {
            result->state = VX_RESULT_STATE_FAILED;
            result->completion_status = status;
        }
        result->completion_parent = NULL;
        vx_result_release(parent);
    }
    if (result->state == VX_RESULT_STATE_PENDING) {
        int pending = 0, failed = 0;
        for (size_t index = 0; index < result->output_count; index++) {
            VxOwnedOutput* output = &result->outputs[index];
            VxDeviceSnapshot* snapshot = &output->device_snapshot;
            if (!snapshot->ticket) continue;
            int status = snapshot->poll(snapshot->ticket, output->data, snapshot->byte_size);
            if (status > 0) { pending = 1; continue; }
            if (status < 0) {
                failed = 1;
                if (status == VX_STATUS_DEVICE_LOST) result->completion_status = VX_STATUS_DEVICE_LOST;
                else if (!result->completion_status) result->completion_status = VX_STATUS_EXECUTION_FAILED;
            }
            else if (snapshot->convert) snapshot->convert(output->data, snapshot->byte_size);
            snapshot->release(snapshot->ticket);
            snapshot->ticket = 0;
        }
        if (failed) {
            result->state = VX_RESULT_STATE_FAILED;
            for (size_t index = 0; index < result->output_count; index++) {
                VxDeviceSnapshot* snapshot = &result->outputs[index].device_snapshot;
                if (snapshot->ticket) snapshot->release(snapshot->ticket);
                snapshot->ticket = 0;
            }
        } else if (!pending) result->state = VX_RESULT_STATE_READY;
    }
    VxStatus status = result->state == VX_RESULT_STATE_FAILED
        ? result->completion_status : VX_STATUS_OK;
    vx_report_write(report, status, VX_STAGE_READBACK, result->evidence.backend,
        result->evidence.device, status == VX_STATUS_OK ? VX_CODE_NONE :
        status == VX_STATUS_DEVICE_LOST ? VX_CODE_DEVICE_LOST : VX_CODE_DEVICE_EXECUTION_FAILED,
        result->state == VX_RESULT_STATE_PENDING ? "output snapshot pending" :
        status == VX_STATUS_OK ? "output snapshot ready" : "device execution or readback failed",
        result->execution_id);
    vx_report_set_result_evidence(report, result);
    return status;
}

static VxStatus vx_result_require_ready(const VxResult* result, VxReport* report) {
    VxStatus status = vx_result_poll((VxResult*)result, report);
    if (status != VX_STATUS_OK) return status;
    if (vx_result_state(result) == VX_RESULT_STATE_PENDING) {
        status = vx_fail(report, VX_STATUS_BUSY, VX_STAGE_READBACK,
            result->evidence.backend, VX_CODE_RESULT_PENDING, "output snapshot is not ready");
        vx_report_set_result_evidence(report, result);
    }
    return status;
}

size_t vx_result_output_count(const VxResult* result) {
    return vx_result_state(result) == VX_RESULT_STATE_READY ? result->output_count : 0;
}

static const char* vx_result_backend(const VxResult* result) {
    return result && result->evidence.struct_size == sizeof(result->evidence)
        ? result->evidence.backend : NULL;
}

VxStatus vx_result_output_info(const VxResult* result,
                               size_t index,
                               VxTensorInfo* info,
                               VxReport* report) {
    const VxOwnedOutput* output;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!result || !info || info->struct_size != sizeof(*info))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_READBACK,
                       NULL, VX_CODE_INVALID_OUTPUT_INFO, "output descriptor request is invalid");
    VxStatus readiness = vx_result_require_ready(result, report);
    if (readiness != VX_STATUS_OK) return readiness;
    if (index >= result->output_count) {
        VxStatus status = vx_fail(report, VX_STATUS_NOT_FOUND, VX_STAGE_READBACK,
                                  vx_result_backend(result),
                                  VX_CODE_OUTPUT_NOT_FOUND,
                                  "output index is out of range");
        vx_report_set_result_evidence(report, result);
        return status;
    }
    output = &result->outputs[index];
    info->name = output->name;
    info->dtype = output->dtype;
    info->rank = output->rank;
    memcpy(info->shape, output->shape, sizeof(info->shape));
    info->byte_size = output->byte_size;
    info->location = output->native_storage && output->native_storage->buffer.kind != VX_NATIVE_BUFFER_HOST
        ? VX_MEMORY_DEVICE : VX_MEMORY_HOST;
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_READBACK,
                    vx_result_backend(result), "host", VX_CODE_NONE,
                    "output descriptor returned", result->execution_id);
    vx_report_set_result_evidence(report, result);
    return VX_STATUS_OK;
}

VxStatus vx_result_read(const VxResult* result,
                        const char* name,
                        void* destination,
                        size_t capacity,
                        size_t* required,
                        VxReport* report) {
    const VxOwnedOutput* output = NULL;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!result || !name || !name[0] || (!destination && capacity))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_READBACK,
                       NULL, VX_CODE_INVALID_READ, "output read request is invalid");
    VxStatus readiness = vx_result_require_ready(result, report);
    if (readiness != VX_STATUS_OK) return readiness;
    for (size_t index = 0; index < result->output_count; index++) {
        if (!strcmp(result->outputs[index].name, name)) {
            output = &result->outputs[index];
            break;
        }
    }
    if (!output) {
        VxStatus status = vx_fail(report, VX_STATUS_NOT_FOUND, VX_STAGE_READBACK,
                                  vx_result_backend(result),
                                  VX_CODE_OUTPUT_NOT_FOUND,
                                  "named output does not exist");
        vx_report_set_result_evidence(report, result);
        return status;
    }
    if (required) *required = output->byte_size;
    if (!destination && capacity == 0) {
        vx_report_write(report, VX_STATUS_OK, VX_STAGE_READBACK,
                        vx_result_backend(result), "host", VX_CODE_NONE,
                        "output byte count returned", result->execution_id);
        vx_report_set_result_evidence(report, result);
        return VX_STATUS_OK;
    }
    if (capacity < output->byte_size) {
        VxStatus status = vx_fail(report, VX_STATUS_BUFFER_TOO_SMALL,
                                  VX_STAGE_READBACK,
                                  vx_result_backend(result),
                                  VX_CODE_BUFFER_TOO_SMALL,
                                  "destination cannot hold the output snapshot");
        vx_report_set_result_evidence(report, result);
        return status;
    }
    if (output->native_storage) {
        if (!vx_native_storage_read(output->native_storage, destination, output->byte_size))
            return vx_fail(report, VX_STATUS_EXECUTION_FAILED, VX_STAGE_READBACK,
                vx_result_backend(result), VX_CODE_EXECUTION_FAILED, "native tensor readback failed");
    } else if (output->byte_size) memcpy(destination, output->data, output->byte_size);
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_READBACK,
                    vx_result_backend(result), "host", VX_CODE_NONE,
                    "output snapshot copied", result->execution_id);
    vx_report_set_result_evidence(report, result);
    return VX_STATUS_OK;
