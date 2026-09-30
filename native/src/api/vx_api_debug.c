/* Generated unary dispatch, explicit execution commands, immutable reads. */
#include "vx_api.h"
#include "vx_api_convert.h"
#include "debugging.h"
#include "vx_training_lifecycle.h"
#include "vx_api_authoring.h"
#include "vx_api_buffer.h"
#include "generated/api_limits.h"
#include "volvoxai_ffi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VX_DEBUG_NEW(parent, field, codec) do { \
    (parent)->field_##field = (parent)->_allocator->allocate((parent)->_allocator->context, sizeof(*(parent)->field_##field)); \
    if (!(parent)->field_##field) return -1; \
    volvoxai_v1_##codec##_init_with_allocator((parent)->field_##field, (parent)->_allocator); \
} while (0)

static int vx_api_debug_error(const SynurangLiteAllocator* allocator,
    VolvoxaiV1OperationReport** report, VxStatus status, const char* message) {
    return vx_api_report_fail(allocator, report, status, VX_STAGE_EXECUTE,
        status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
        status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY : VX_CODE_INVALID_ARGUMENT, message) ? 0 : -1;
}
static void vx_api_debug_retain(void* p) { vx_debug_retain(p); }
static void vx_api_debug_release(void* p) { vx_debug_release(p); }

static int vx_api_debug_info(const VxDebugView* view, void* opaque) {
    VolvoxaiV1DebugSessionInfo* out = opaque;
    out->field_state = view->state; out->field_stop_reason = view->stop_reason;
    out->field_revision = view->revision;
    out->field_next_step = view->next_step; out->field_step_count = view->plan->step_count;
    VX_DEBUG_NEW(out, events, record_count);
    out->field_events->field_count = view->event_count; out->field_events->field_dropped = view->dropped_events;
    VX_DEBUG_NEW(out, snapshots, record_count);
    out->field_snapshots->field_count = view->snapshot_count; out->field_snapshots->field_dropped = view->dropped_snapshots;
    out->field_retained_bytes = view->retained_bytes; out->field_peak_bytes = view->peak_bytes;
    out->field_max_bytes = view->max_bytes;
    out->field_capture_complete = view->capture_complete;
    out->field_target = view->target;
    out->field_result_id = view->result_id;
    out->field_modified = view->modified != 0;
    if (view->train_completed && view->train_owner) {
        const VxTrainStepResult* train = vx_trainer_debug_result(view->train_owner);
        out->field_train_step_result = out->_allocator->allocate(out->_allocator->context,
            sizeof(*out->field_train_step_result));
        if (!out->field_train_step_result) return -1;
        volvoxai_v1_train_step_result_init_with_allocator(out->field_train_step_result, out->_allocator);
        VxReport report = VX_REPORT_INIT;
        report.stage = VX_STAGE_TRAINER_STEP;
        if (vx_api_training_result(out->field_train_step_result, train, &report)) return -1;
    }
    VX_DEBUG_NEW(out, capabilities, debug_capabilities);
    out->field_capabilities->field_step = 1;
    out->field_capabilities->field_tensor_capture = VOLVOXAI_V1_OBSERVATION_STATUS_AVAILABLE;
    out->field_capabilities->field_device_synchronization = view->device_synchronization;
    out->field_capabilities->field_preserves_node_boundaries = view->preserves_node_boundaries;
    /* The RPC succeeded; a failing model is session state, reported separately. */
    VxReport report = view->report;
    if (view->state == VX_DEBUG_STATE_FAILED) {
        if (!vx_api_report_attach(out->_allocator, &out->field_failure, &view->report)) return -1;
        report.status = VX_STATUS_OK; report.code = VX_CODE_NONE;
        snprintf(report.message, sizeof(report.message), "%s", "debug session failed; see failure");
    }
    if (!vx_api_report_attach(out->_allocator, &out->field_report, &report)) return -1;
    const VxExecutionIdentity* identity = &view->plan->identity;
    vx_api_report_set_public_lineage(out->field_report, identity->runtime_id,
        identity->model_id, identity->compiled_model_id, identity->context_id);
    if (out->field_failure) vx_api_report_set_public_lineage(out->field_failure, identity->runtime_id,
        identity->model_id, identity->compiled_model_id, identity->context_id);
    return 0;
}

/* A completed decode target's result becomes an ordinary result handle once. */
static void vx_api_debug_publish(VxApiRegistry* registry, VxDebugSession* session,
    const VxApiHandleLineage* lineage) {
    VxResult* result = vx_debug_take_result(session);
    if (result) vx_debug_set_result_id(session, vx_api_publish_result_handle(registry, result, lineage));
}

/* Non-empty, NUL-free strings copied into call scratch. NULL on failure. */
typedef struct { SynurangLiteBytes* data; size_t len; } VxApiDebugStringList;
static const char* const* vx_api_debug_strings(VxApiScratch* scratch, const void* list) {
    const VxApiDebugStringList* in = list;
    const char** names = vx_api_scratch_alloc(scratch, in->len * sizeof(*names));
    if (!names) return NULL;
    for (size_t i = 0; i < in->len; i++) {
        names[i] = vx_api_scratch_cstr(scratch, &in->data[i]);
        if (!names[i] || !names[i][0] || strlen(names[i]) != in->data[i].len) return NULL;
    }
    return names;
}
static int vx_api_create_debug_session(const VolvoxaiV1CreateDebugSessionRequest* request,
    VolvoxaiV1DebugSessionInfo* response, void* registry) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(registry);
    VxApiDecodeCursor cursor = {0};
    const VolvoxaiV1Tensor* tensors = NULL;
    size_t count = 0;
    int result = 0;
    VxDebugSession* session = NULL;
    VxReport report = VX_REPORT_INIT;
    VxDebugOptions options = {.inputs = 1, .outputs = 1, .constants = 1, .gradients = 1, .max_bytes = VX_API_DEBUG_LIMITS_MAX_BYTES_DEFAULT,
        .max_events = VX_API_DEBUG_LIMITS_MAX_EVENTS_DEFAULT, .max_snapshots = VX_API_DEBUG_LIMITS_MAX_SNAPSHOTS_DEFAULT};
    const VolvoxaiV1DebugCapture* capture = request->field_capture;
    const VolvoxaiV1DebugLimits* limits = request->field_limits;
    if (request->which_target == 1 && request->field_forward) {
        if (!vx_api_handle_acquire(registry, VX_API_HANDLE_COMPILED_MODEL, request->field_forward->field_compiled_model_id, &lease))
            return vx_api_debug_error(response->_allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released compiled model");
        tensors = request->field_forward->field_inputs.data;
        count = request->field_forward->field_inputs.len;
    } else if ((request->which_target == 2 && request->field_decode_prefill) ||
               (request->which_target == 3 && request->field_decode_step)) {
        int prefill = request->which_target == 2;
        int64_t context_id = prefill ? request->field_decode_prefill->field_context_id : request->field_decode_step->field_context_id;
        int parsed = prefill ? vx_api_decode_prefill_cursor(request->field_decode_prefill, &cursor)
                             : vx_api_decode_step_cursor(request->field_decode_step, &cursor);
        if (parsed == -2) return -1;
        if (parsed) return vx_api_report_fail(response->_allocator, &response->field_report, VX_STATUS_INVALID_ARGUMENT,
            VX_STAGE_DECODE, VX_CODE_INVALID_DECODE_POSITION, "decode cursor must contain a valid scalar or one action per slot") ? 0 : -1;
        if (!vx_api_handle_acquire(registry, VX_API_HANDLE_CONTEXT, context_id, &lease)) {
            free(cursor.owned);
            return vx_api_debug_error(response->_allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released context");
        }
        tensors = prefill ? request->field_decode_prefill->field_inputs.data : request->field_decode_step->field_inputs.data;
        count = prefill ? request->field_decode_prefill->field_inputs.len : request->field_decode_step->field_inputs.len;
    } else if (request->which_target == 6 && request->field_train_step) {
        if (!vx_api_handle_acquire(registry, VX_API_HANDLE_TRAINER, request->field_train_step->field_trainer_id, &lease))
            return vx_api_debug_error(response->_allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released trainer");
    } else {
        return vx_api_debug_error(response->_allocator, &response->field_report, VX_STATUS_INVALID_ARGUMENT,
            "a forward, decode_prefill, decode_step or train_step target is required");
    }
    if (limits) {
        if (limits->has_max_bytes) options.max_bytes = limits->field_max_bytes;
        if (limits->has_max_events) options.max_events = limits->field_max_events;
        if (limits->has_max_snapshots) options.max_snapshots = limits->field_max_snapshots;
    }
    if (options.max_bytes < VX_API_DEBUG_LIMITS_MAX_BYTES_MINIMUM || options.max_bytes > VX_API_DEBUG_LIMITS_MAX_BYTES_MAXIMUM ||
        options.max_events < VX_API_DEBUG_LIMITS_MAX_EVENTS_MINIMUM || options.max_events > VX_API_DEBUG_LIMITS_MAX_EVENTS_MAXIMUM ||
        options.max_snapshots < VX_API_DEBUG_LIMITS_MAX_SNAPSHOTS_MINIMUM || options.max_snapshots > VX_API_DEBUG_LIMITS_MAX_SNAPSHOTS_MAXIMUM) {
        result = vx_api_debug_error(response->_allocator, &response->field_report, VX_STATUS_INVALID_ARGUMENT, "debug limits are out of range");
        goto done;
    }
    if (capture) {
        options.node_count = capture->field_node_ids.len;
        if (options.node_count && !(options.node_ids = vx_api_debug_strings(&scratch, &capture->field_node_ids))) {
            result = vx_api_debug_error(response->_allocator, &response->field_report, VX_STATUS_INVALID_ARGUMENT, "invalid debug node ID");
            goto done;
        }
        options.inputs = capture->has_inputs ? capture->field_inputs : 1;
        options.outputs = capture->has_outputs ? capture->field_outputs : 1;
        options.values = capture->field_values;
        options.constants = capture->has_constants ? capture->field_constants : 1;
        options.gradients = capture->has_gradients ? capture->field_gradients : 1;
        options.tensor_count = capture->field_tensor_names.len;
        if (options.tensor_count && !(options.tensor_names = vx_api_debug_strings(&scratch, &capture->field_tensor_names))) {
            result = vx_api_debug_error(response->_allocator, &response->field_report, VX_STATUS_INVALID_ARGUMENT, "invalid debug tensor name");
            goto done;
        }
    }
    VxTensorBinding* bindings = count ? vx_api_scratch_alloc(&scratch, count * sizeof(*bindings)) : NULL;
    if (count && !bindings) { result = -1; goto done; }
    for (size_t i = 0; i < count; i++) {
        VxStatus status = vx_api_binding_from_tensor(&scratch, &bindings[i], &tensors[i]);
        if (status != VX_STATUS_OK) {
            result = vx_api_report_binding_fail(response->_allocator, &response->field_report, status, VX_STAGE_EXECUTE,
                "invalid debug input tensor", &tensors[i], i) ? 0 : -1;
            goto done;
        }
    }
    VxStatus status;
    if (request->which_target == 1) {
        status = vx_debug_create(lease.pointer, bindings, count, &options, &session, &report);
    } else if (request->which_target == 6) {
        VxTrainStepOptions train = VX_TRAIN_STEP_OPTIONS_INIT;
        int converted = vx_api_train_step_options(&scratch, request->field_train_step, &train,
            response->_allocator, &response->field_report);
        if (converted) { result = converted > 0 ? 0 : -1; goto done; }
        status = vx_trainer_debug_session(lease.pointer, &train, &options, &session, &report);
    } else {
        VxDebugDecodeTarget target = {.prefill = request->which_target == 2,
            .dependency_update = cursor.dependency_update, .position = cursor.position,
            .slot_positions = cursor.positions, .slot_count = cursor.count};
        status = vx_debug_create_decode(lease.pointer, &target, bindings, count, &options, &session, &report);
    }
    if (status != VX_STATUS_OK) {
        result = vx_api_report_attach(response->_allocator, &response->field_report, &report) ? 0 : -1;
        goto done;
    }
    response->field_debug_session_id = vx_api_handle_insert_with_lineage(registry, VX_API_HANDLE_DEBUG_SESSION,
        session, vx_api_debug_retain, vx_api_debug_release, &lease.lineage);
    if (!response->field_debug_session_id) { vx_debug_detach(session); vx_debug_release(session); result = -1; goto done; }
    vx_api_debug_publish(registry, session, &lease.lineage);
    result = vx_debug_inspect(session, vx_api_debug_info, response);
    if (result) {
        vx_debug_detach(session);
        vx_api_handle_remove(registry, VX_API_HANDLE_DEBUG_SESSION, response->field_debug_session_id);
    }
done:
    free(cursor.owned);
    vx_api_scratch_release(&scratch);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_get_debug_session(const VolvoxaiV1DebugSessionRef* request,
    VolvoxaiV1DebugSessionInfo* response, void* registry) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_DEBUG_SESSION, request->field_debug_session_id, &lease))
        return vx_api_debug_error(response->_allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released debug session");
    response->field_debug_session_id = request->field_debug_session_id;
    vx_api_debug_publish(registry, lease.pointer, &lease.lineage);
    int result = vx_debug_inspect(lease.pointer, vx_api_debug_info, response);
    vx_api_handle_lease_release(&lease);
    return result;
}
static int vx_api_cancel_debug_session(const VolvoxaiV1DebugSessionRef* request,
    VolvoxaiV1DebugSessionInfo* response, void* registry) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_DEBUG_SESSION, request->field_debug_session_id, &lease))
        return vx_api_debug_error(response->_allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released debug session");
    vx_debug_cancel(lease.pointer);
    response->field_debug_session_id = request->field_debug_session_id;
    int result = vx_debug_inspect(lease.pointer, vx_api_debug_info, response);
    vx_api_handle_lease_release(&lease);
    return result;
}
static int vx_api_set_debug_tensor(const VolvoxaiV1SetDebugTensorRequest* request,
    VolvoxaiV1DebugSessionInfo* response, void* registry) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (!request->has_expected_revision || !request->field_value)
        return vx_api_debug_error(allocator, &response->field_report, VX_STATUS_INVALID_ARGUMENT,
            "expected_revision and value are required");
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_DEBUG_SESSION, request->field_debug_session_id, &lease))
        return vx_api_debug_error(allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released debug session");
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(registry);
    VxTensorBinding binding = VX_TENSOR_BINDING_INIT;
    VxReport report = VX_REPORT_INIT;
    int result;
    VxStatus status = vx_api_binding_from_tensor(&scratch, &binding, request->field_value);
    if (status == VX_STATUS_OK && binding.location != VX_MEMORY_HOST) status = VX_STATUS_INVALID_ARGUMENT;
    if (status != VX_STATUS_OK)
        result = vx_api_debug_error(allocator, &response->field_report, status, "value must be a host tensor");
    else if (vx_debug_set_tensor(lease.pointer, request->field_expected_revision, request->field_tensor_id,
            &binding, &report) != VX_STATUS_OK)
        result = vx_api_report_attach(allocator, &response->field_report, &report) ? 0 : -1;
    else {
        response->field_debug_session_id = request->field_debug_session_id;
        result = vx_debug_inspect(lease.pointer, vx_api_debug_info, response);
    }
    vx_api_scratch_release(&scratch);
    vx_api_handle_lease_release(&lease);
    return result;
}
static int vx_api_release_debug_session(const VolvoxaiV1DebugSessionRef* request,
    VolvoxaiV1OperationReport* response, void* registry) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (vx_api_handle_acquire(registry, VX_API_HANDLE_DEBUG_SESSION, request->field_debug_session_id, &lease)) {
        vx_debug_detach(lease.pointer);
        vx_api_handle_remove(registry, VX_API_HANDLE_DEBUG_SESSION, request->field_debug_session_id);
        vx_api_handle_lease_release(&lease);
    }
    response->field_stage = VX_STAGE_CLOSE;
    return 0;
}

static void vx_api_debug_statistics(VolvoxaiV1DebugTensorStatistics* stats, const VxDebugStatistics* source) {
    stats->field_finite_count = source->finite;
    stats->field_nan_count = source->nan;
    stats->field_positive_infinity_count = source->positive_infinity;
    stats->field_negative_infinity_count = source->negative_infinity;
    if (source->finite) {
        stats->has_minimum = stats->has_maximum = stats->has_mean = stats->has_variance = 1;
        stats->field_minimum = source->minimum; stats->field_maximum = source->maximum;
        stats->field_mean = source->mean; stats->field_variance = source->variance;
    }
}
static int vx_api_debug_quantization(VolvoxaiV1AffineQuantizationParameters* params, const VxDebugQuantization* quant) {
    if (quant->per_axis) {
        params->which_parameters = 2;
        VX_DEBUG_NEW(params, per_axis, per_axis_affine_quantization);
        params->field_per_axis->field_axis = quant->axis;
        for (size_t i = 0; i < quant->count; i++) {
            float* scale = volvoxai_v1_per_axis_affine_quantization_add_scales(params->field_per_axis);
            int32_t* zero = volvoxai_v1_per_axis_affine_quantization_add_zero_points(params->field_per_axis);
            if (!scale || !zero) return -1;
            *scale = quant->scales[i]; *zero = quant->zero_points[i];
        }
    } else {
        params->which_parameters = 1;
        VX_DEBUG_NEW(params, per_tensor, per_tensor_affine_quantization);
        params->field_per_tensor->field_scale = quant->scales[0];
        params->field_per_tensor->field_zero_point = quant->zero_points[0];
    }
    return 0;
}

static int vx_api_debug_snapshot(VolvoxaiV1DebugTensorSnapshot* out, const VxDebugSnapshot* snapshot,
    const VxDebugQuantization* quant) {
    out->field_snapshot_id = snapshot->id;
    out->field_tensor_id = snapshot->tensor_id; out->field_status = snapshot->status;
    out->field_dtype = snapshot->dtype; out->field_logical_bytes = snapshot->bytes;
    for (uint32_t i = 0; i < snapshot->rank; i++) {
        int64_t* dim = volvoxai_v1_debug_tensor_snapshot_add_shape(out);
        if (!dim) return -1;
        *dim = snapshot->shape[i];
    }
    if (snapshot->status == VX_DEBUG_TENSOR_STATUS_AVAILABLE || snapshot->status == VX_DEBUG_TENSOR_STATUS_STATISTICS_ONLY) {
        VX_DEBUG_NEW(out, statistics, debug_tensor_statistics);
        vx_api_debug_statistics(out->field_statistics, &snapshot->statistics);
    }
    if (snapshot->has_real_statistics) {
        VX_DEBUG_NEW(out, real_statistics, debug_tensor_statistics);
        vx_api_debug_statistics(out->field_real_statistics, &snapshot->real_statistics);
    }
    if (quant->count) {
        VX_DEBUG_NEW(out, quantization, affine_quantization_parameters);
        if (vx_api_debug_quantization(out->field_quantization, quant)) return -1;
    }
    return 0;
}
static int vx_api_debug_plan(const VxDebugView* view, void* opaque) {
    VolvoxaiV1GetDebugPlanResponse* out = opaque;
    VX_DEBUG_NEW(out, plan, execution_plan);
    if (!vx_api_execution_plan(out->field_plan, view->plan)) return -1;
    return vx_api_report_ok(out->_allocator, &out->field_report, VX_STAGE_EXECUTE) ? 0 : -1;
}
static int vx_api_get_debug_plan(const VolvoxaiV1DebugSessionRef* request,
    VolvoxaiV1GetDebugPlanResponse* response, void* registry) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_DEBUG_SESSION, request->field_debug_session_id, &lease))
        return vx_api_debug_error(response->_allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released debug session");
    int result = vx_debug_inspect(lease.pointer, vx_api_debug_plan, response);
    vx_api_handle_lease_release(&lease);
    return result;
}
typedef struct { const VolvoxaiV1ListDebugEventsRequest* request; VolvoxaiV1ListDebugEventsResponse* response; } VxApiDebugEvents;
static int vx_api_debug_events(const VxDebugView* view, void* opaque) {
    VxApiDebugEvents* page = opaque;
    const VolvoxaiV1ListDebugEventsRequest* in = page->request;
    VolvoxaiV1ListDebugEventsResponse* out = page->response;
    uint64_t limit = in->has_page_size ? in->field_page_size : VX_API_LIST_DEBUG_EVENTS_REQUEST_PAGE_SIZE_DEFAULT;
    uint64_t total = view->event_count, offset;
    if (limit < VX_API_LIST_DEBUG_EVENTS_REQUEST_PAGE_SIZE_MINIMUM || limit > VX_API_LIST_DEBUG_EVENTS_REQUEST_PAGE_SIZE_MAXIMUM)
        return vx_api_debug_error(out->_allocator, &out->field_report, VX_STATUS_INVALID_ARGUMENT, "page_size is out of range");
    if (!vx_api_page_token_offset(&in->field_page_token, total, &offset))
        return vx_api_debug_error(out->_allocator, &out->field_report, VX_STATUS_INVALID_ARGUMENT, "invalid page_token");
    uint64_t count = total - offset < limit ? total - offset : limit;
    for (uint64_t i = offset; i < offset + count; i++) {
        const VxDebugEvent* source = &view->events[i];
        VolvoxaiV1DebugEvent* event = volvoxai_v1_list_debug_events_response_add_events(out);
        if (!event) return -1;
        event->field_event_id = source->id; event->field_step = source->step; event->field_point = source->point;
        for (size_t j = 0; j < source->snapshot_count; j++) {
            const VxDebugSnapshot* snapshot = &view->snapshots[source->first_snapshot + j];
            VolvoxaiV1DebugTensorSnapshot* tensor = volvoxai_v1_debug_event_add_snapshots(event);
            if (!tensor || vx_api_debug_snapshot(tensor, snapshot, &view->quantization[snapshot->tensor_id])) return -1;
        }
    }
    out->field_total_size = total;
    if (!vx_api_next_page_token(out->_allocator, &out->field_next_page_token, offset + count, total)) return -1;
    return vx_api_report_ok(out->_allocator, &out->field_report, VX_STAGE_EXECUTE) ? 0 : -1;
}
static int vx_api_list_debug_events(const VolvoxaiV1ListDebugEventsRequest* request,
    VolvoxaiV1ListDebugEventsResponse* response, void* registry) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_DEBUG_SESSION, request->field_debug_session_id, &lease))
        return vx_api_debug_error(response->_allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released debug session");
    VxApiDebugEvents page = {request, response};
    int result = vx_debug_inspect(lease.pointer, vx_api_debug_events, &page);
    vx_api_handle_lease_release(&lease);
    return result;
}
typedef struct { const VolvoxaiV1ReadDebugTensorRequest* request; VolvoxaiV1ReadDebugTensorResponse* response; } VxApiDebugChunk;
static int vx_api_debug_chunk(const VxDebugView* view, void* opaque) {
    VxApiDebugChunk* chunk = opaque;
    const VolvoxaiV1ReadDebugTensorRequest* in = chunk->request;
    VolvoxaiV1ReadDebugTensorResponse* out = chunk->response;
    uint64_t limit = in->has_read_limit ? in->field_read_limit : VX_API_READ_DEBUG_TENSOR_REQUEST_READ_LIMIT_DEFAULT;
    if (limit < VX_API_READ_DEBUG_TENSOR_REQUEST_READ_LIMIT_MINIMUM || limit > VX_API_READ_DEBUG_TENSOR_REQUEST_READ_LIMIT_MAXIMUM)
        return vx_api_debug_error(out->_allocator, &out->field_report, VX_STATUS_INVALID_ARGUMENT, "read_limit is out of range");
    if (!in->field_snapshot_id || in->field_snapshot_id > view->snapshot_count)
        return vx_api_report_fail(out->_allocator, &out->field_report, VX_STATUS_NOT_FOUND, VX_STAGE_EXECUTE,
            VX_CODE_INVALID_ARGUMENT, "unknown snapshot_id") ? 0 : -1;
    const VxDebugSnapshot* snapshot = &view->snapshots[in->field_snapshot_id - 1];
    out->field_status = snapshot->status;
    if (snapshot->status != VX_DEBUG_TENSOR_STATUS_AVAILABLE)
        return vx_api_report_ok(out->_allocator, &out->field_report, VX_STAGE_EXECUTE) ? 0 : -1;
    out->field_size_bytes = snapshot->bytes;
    if (in->field_read_offset > snapshot->bytes)
        return vx_api_debug_error(out->_allocator, &out->field_report, VX_STATUS_INVALID_ARGUMENT, "read_offset is beyond the snapshot");
    size_t count = snapshot->bytes - (size_t)in->field_read_offset;
    if (count > limit) count = (size_t)limit;
    if (count && synurang_lite_bytes_assign(out->_allocator, &out->field_data,
        snapshot->data + in->field_read_offset, count) != SYNURANG_LITE_OK) return -1;
    return vx_api_report_ok(out->_allocator, &out->field_report, VX_STAGE_EXECUTE) ? 0 : -1;
}
static int vx_api_read_debug_tensor(const VolvoxaiV1ReadDebugTensorRequest* request,
    VolvoxaiV1ReadDebugTensorResponse* response, void* registry) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_DEBUG_SESSION, request->field_debug_session_id, &lease))
        return vx_api_debug_error(response->_allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released debug session");
    VxApiDebugChunk chunk = {request, response};
    int result = vx_debug_inspect(lease.pointer, vx_api_debug_chunk, &chunk);
    vx_api_handle_lease_release(&lease);
    return result;
}

/* A snapshot copied into an ordinary read-only host buffer joins the buffer
 * service (CopyTensors, DLPack) and outlives the session. */
typedef struct {
    const VolvoxaiV1ExportDebugTensorRequest* request;
    VolvoxaiV1TensorBatch* response;
    VxApiRegistry* registry;
} VxApiDebugExport;
static int vx_api_debug_export(const VxDebugView* view, void* opaque) {
    VxApiDebugExport* export = opaque;
    VolvoxaiV1TensorBatch* out = export->response;
    uint64_t id = export->request->field_snapshot_id;
    if (!id || id > view->snapshot_count)
        return vx_api_report_fail(out->_allocator, &out->field_report, VX_STATUS_NOT_FOUND, VX_STAGE_EXECUTE,
            VX_CODE_INVALID_ARGUMENT, "unknown snapshot_id") ? 0 : -1;
    const VxDebugSnapshot* snapshot = &view->snapshots[id - 1];
    if (snapshot->status != VX_DEBUG_TENSOR_STATUS_AVAILABLE || !snapshot->bytes)
        return vx_api_debug_error(out->_allocator, &out->field_report, VX_STATUS_INVALID_ARGUMENT,
            "only AVAILABLE snapshots with bytes can be exported");
    void* data = malloc(snapshot->bytes);
    if (!data) return vx_api_debug_error(out->_allocator, &out->field_report, VX_STATUS_OUT_OF_MEMORY, "snapshot copy failed");
    memcpy(data, snapshot->data, snapshot->bytes);
    VxNativeBuffer memory = {.kind = VX_NATIVE_BUFFER_HOST, .handle = (uint64_t)(uintptr_t)data, .length = snapshot->bytes};
    VxApiBuffer* buffer = vx_api_buffer_create(&memory, NULL, data, free, 0);
    if (!buffer) { free(data); return -1; }
    int64_t handle = vx_api_buffer_publish(export->registry, buffer);
    if (!handle) { vx_api_buffer_release(buffer); return -1; }
    VxTensorInfo info = {.struct_size = sizeof(info), .name = view->plan->tensors[snapshot->tensor_id].name,
        .dtype = snapshot->dtype, .rank = snapshot->rank, .byte_size = snapshot->bytes};
    for (uint32_t i = 0; i < snapshot->rank && i < VX_MAX_TENSOR_RANK; i++) info.shape[i] = snapshot->shape[i];
    VolvoxaiV1Tensor* tensor = volvoxai_v1_tensor_batch_add_outputs(out);
    if (!tensor || !vx_api_buffer_tensor(out->_allocator, tensor, &info, handle)) {
        vx_api_handle_remove(export->registry, VX_API_HANDLE_BUFFER, handle);
        return -1;
    }
    return vx_api_report_ok(out->_allocator, &out->field_report, VX_STAGE_EXECUTE) ? 0 : -1;
}
static int vx_api_export_debug_tensor(const VolvoxaiV1ExportDebugTensorRequest* request,
    VolvoxaiV1TensorBatch* response, void* registry) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_DEBUG_SESSION, request->field_debug_session_id, &lease))
        return vx_api_debug_error(response->_allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released debug session");
    VxApiDebugExport export = {request, response, registry};
    int result = vx_debug_inspect(lease.pointer, vx_api_debug_export, &export);
    vx_api_handle_lease_release(&lease);
    return result;
}

/* Step and Continue are asynchronous unary commands. One module poll executes
 * at most one node, yielding to cancellation and WebGPU completion between
 * nodes. Get/Read never call progress. Transport cancellation also cancels this
 * command's execution; a disconnected debugger cannot leave an orphan runner. */
typedef struct {
    VxApiPending pending;
    VxApiRegistry* registry;
    SynurangStream* stream;
    VxApiHandleLease lease;
    int64_t id;
    int single_step;
} VxApiDebugCommand;
static SynurangStatus vx_api_debug_respond(VxApiDebugCommand* command, SynurangStream* stream,
    VolvoxaiV1DebugSessionInfo* response) {
    return command->single_step ? vx_debug_step_debug_session_respond(stream, response) :
        vx_debug_continue_debug_session_respond(stream, response);
}
static void vx_api_debug_detach(VxApiDebugCommand* command) {
    if (!command || !command->stream) return;
    vx_api_pending_remove(&command->pending);
    vx_api_handle_lease_release(&command->lease);
    SynurangStream* stream = command->stream; command->stream = NULL;
    synurang_stream_release(stream);
}
static void vx_api_debug_poll(VxApiPending* pending) {
    VxApiDebugCommand* command = (VxApiDebugCommand*)pending;
    int result = vx_debug_progress(command->lease.pointer);
    if (result >= 0) { if (result) vx_api_pending_notify(pending); return; }
    VolvoxaiV1DebugSessionInfo response;
    volvoxai_v1_debug_session_info_init(&response);
    response.field_debug_session_id = command->id;
    vx_api_debug_publish(command->registry, command->lease.pointer, &command->lease.lineage);
    result = vx_debug_inspect(command->lease.pointer, vx_api_debug_info, &response);
    SynurangStatus status = result ? SYNURANG_INTERNAL : vx_api_debug_respond(command, command->stream, &response);
    volvoxai_v1_debug_session_info_free(&response);
    if (status == SYNURANG_OK) (void)synurang_stream_finish(command->stream);
    else (void)synurang_stream_fail_error(command->stream, status, 13, "debug response construction failed");
    vx_api_debug_detach(command);
}
static void* vx_api_debug_open(SynurangStream* stream, const SynurangCallOptions* options, void* registry) {
    (void)options;
    VxApiDebugCommand* command = calloc(1, sizeof(*command));
    if (!command) { (void)synurang_stream_fail_error(stream, SYNURANG_OUT_OF_MEMORY, 8, "debug command allocation failed"); return NULL; }
    command->registry = registry; command->pending.poll = vx_api_debug_poll;
    return command;
}
static void vx_api_debug_command(SynurangStream* stream, VxApiDebugCommand* command,
    int64_t id, int has_revision, uint64_t revision, int single_step, const void* break_nodes, int break_on_nonfinite) {
    if (!command) return;
    command->id = id; command->single_step = single_step;
    VolvoxaiV1DebugSessionInfo response;
    volvoxai_v1_debug_session_info_init(&response);
    response.field_debug_session_id = id;
    VxReport report = VX_REPORT_INIT;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(command->registry);
    const VxApiDebugStringList* breaks = break_nodes;
    const char* const* names = NULL;
    int result = 0;
    if (!vx_api_handle_acquire(command->registry, VX_API_HANDLE_DEBUG_SESSION, id, &command->lease))
        result = vx_api_debug_error(response._allocator, &response.field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released debug session");
    else if (!has_revision)
        result = vx_api_debug_error(response._allocator, &response.field_report, VX_STATUS_INVALID_ARGUMENT, "expected_revision is required");
    else if (breaks && breaks->len && !(names = vx_api_debug_strings(&scratch, breaks)))
        result = vx_api_debug_error(response._allocator, &response.field_report, VX_STATUS_INVALID_ARGUMENT, "invalid breakpoint node ID");
    else if (vx_debug_start(command->lease.pointer, revision, single_step, names, breaks ? breaks->len : 0,
        break_on_nonfinite, &report) != VX_STATUS_OK)
        result = vx_api_report_attach(response._allocator, &response.field_report, &report) ? 0 : -1;
    else {
        command->stream = synurang_stream_retain(stream);
        vx_api_pending_add(command->registry, &command->pending);
        volvoxai_v1_debug_session_info_free(&response);
        vx_api_scratch_release(&scratch);
        return;
    }
    vx_api_scratch_release(&scratch);
    SynurangStatus status = result ? SYNURANG_INTERNAL : vx_api_debug_respond(command, stream, &response);
    volvoxai_v1_debug_session_info_free(&response);
    vx_api_handle_lease_release(&command->lease);
    if (status == SYNURANG_OK) (void)synurang_stream_finish(stream);
    else (void)synurang_stream_fail_error(stream, status, 13, "debug response construction failed");
}
static void vx_api_step_debug_session_call(SynurangStream* stream, const VolvoxaiV1StepDebugSessionRequest* in, void* data) {
    vx_api_debug_command(stream, data, in->field_debug_session_id, in->has_expected_revision, in->field_expected_revision, 1, NULL, 0);
}
static void vx_api_continue_debug_session_call(SynurangStream* stream, const VolvoxaiV1ContinueDebugSessionRequest* in, void* data) {
    vx_api_debug_command(stream, data, in->field_debug_session_id, in->has_expected_revision, in->field_expected_revision, 0,
        &in->field_break_before_nodes, in->field_break_on_nonfinite);
}
static void vx_api_debug_cancel_call(SynurangStream* stream, void* data) {
    (void)stream;
    VxApiDebugCommand* command = data;
    if (command && command->lease.pointer) vx_debug_cancel(command->lease.pointer);
    vx_api_debug_detach(command);
}
static void vx_api_debug_destroy_call(void* data) { vx_api_debug_detach(data); free(data); }

static int vx_api_debug_decode_state(const VxDebugView* view, void* opaque) {
    VolvoxaiV1DebugDecodeState* out = opaque;
    if (view->target != VX_DEBUG_TARGET_DECODE_PREFILL && view->target != VX_DEBUG_TARGET_DECODE_STEP)
        return vx_api_debug_error(out->_allocator, &out->field_report, VX_STATUS_INVALID_ARGUMENT,
            "decode state exists only for decode targets");
    out->field_target = view->target;
    for (uint32_t i = 0; i < view->slot_count; i++) {
        const VxDebugSlot* source = &view->slots[i];
        VolvoxaiV1DebugDecodeSlot* slot = volvoxai_v1_debug_decode_state_add_slots(out);
        if (!slot) return -1;
        slot->field_slot = source->slot;
        slot->field_length_before = source->length_before; slot->field_length_after = source->length_after;
        slot->field_write_begin = source->write_begin; slot->field_write_end = source->write_end;
        slot->field_empty = source->empty;
    }
    for (size_t i = 0; i < view->cache_count; i++) {
        const VxDebugKVCache* source = &view->caches[i];
        VolvoxaiV1DebugKVCache* cache = volvoxai_v1_debug_decode_state_add_caches(out);
        if (!cache) return -1;
        cache->field_cache_id = source->id; cache->field_role = source->role;
        cache->field_tensor_id = source->tensor_id; cache->field_attention_step = source->attention_step;
        cache->field_dtype = source->dtype; cache->field_token_capacity = source->token_capacity;
        cache->field_page_tokens = source->page_tokens;
        for (uint32_t d = 0; d < source->token_rank; d++) {
            int64_t* dim = volvoxai_v1_debug_kv_cache_add_token_shape(cache);
            if (!dim) return -1;
            *dim = source->token_shape[d];
        }
        if (source->writer_step >= 0) {
            cache->has_writer_step = 1;
            cache->field_writer_step = (uint32_t)source->writer_step;
            cache->field_written = view->executed && view->executed[source->writer_step];
        }
    }
    return vx_api_report_ok(out->_allocator, &out->field_report, VX_STAGE_DECODE) ? 0 : -1;
}
static int vx_api_get_debug_decode_state(const VolvoxaiV1DebugSessionRef* request,
    VolvoxaiV1DebugDecodeState* response, void* registry) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_DEBUG_SESSION, request->field_debug_session_id, &lease))
        return vx_api_debug_error(response->_allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released debug session");
    int result = vx_debug_inspect(lease.pointer, vx_api_debug_decode_state, response);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_debug_kv_rows(const VxDebugKVRows* rows, void* opaque) {
    VolvoxaiV1ReadDebugKVCacheResponse* out = opaque;
    out->field_dtype = rows->dtype;
    out->field_token_count = rows->token_count;
    for (uint32_t d = 0; d < rows->rank; d++) {
        int64_t* dim = volvoxai_v1_read_debug_kv_cache_response_add_shape(out);
        if (!dim) return -1;
        *dim = rows->shape[d];
    }
    if (rows->bytes && synurang_lite_bytes_assign(out->_allocator, &out->field_data, rows->data, rows->bytes) != SYNURANG_LITE_OK)
        return -1;
    if (rows->has_statistics) {
        VX_DEBUG_NEW(out, statistics, debug_tensor_statistics);
        vx_api_debug_statistics(out->field_statistics, &rows->statistics);
    }
    if (rows->quantization) {
        VX_DEBUG_NEW(out, quantization, affine_quantization_parameters);
        if (vx_api_debug_quantization(out->field_quantization, rows->quantization)) return -1;
    }
    return 0;
}

/* KV reads are asynchronous commands: a device-resident cache is copied back
 * before the reply, and the session stays readable meanwhile. */
typedef struct {
    VxApiPending pending;
    VxApiRegistry* registry;
    SynurangStream* stream;
    VxApiHandleLease lease;
} VxApiDebugRead;
static void vx_api_debug_read_detach(VxApiDebugRead* command) {
    if (!command || !command->stream) return;
    vx_api_pending_remove(&command->pending);
    vx_api_handle_lease_release(&command->lease);
    SynurangStream* stream = command->stream; command->stream = NULL;
    synurang_stream_release(stream);
}
static void vx_api_debug_read_reply(VxApiDebugRead* command, SynurangStream* stream) {
    VolvoxaiV1ReadDebugKVCacheResponse response;
    volvoxai_v1_read_debug_kv_cache_response_init(&response);
    VxReport report = VX_REPORT_INIT;
    int result = 0;
    VxStatus status = vx_debug_kv_read_finish(command->lease.pointer, vx_api_debug_kv_rows, &response, &report);
    if (status == VX_STATUS_INTERNAL) result = -1;
    else if (status != VX_STATUS_OK) {
        volvoxai_v1_read_debug_kv_cache_response_free(&response);
        volvoxai_v1_read_debug_kv_cache_response_init(&response);
        result = vx_api_report_attach(response._allocator, &response.field_report, &report) ? 0 : -1;
    } else result = vx_api_report_ok(response._allocator, &response.field_report, VX_STAGE_DECODE) ? 0 : -1;
    SynurangStatus sent = result ? SYNURANG_INTERNAL : vx_debug_read_debug_k_v_cache_respond(stream, &response);
    volvoxai_v1_read_debug_kv_cache_response_free(&response);
    if (sent == SYNURANG_OK) (void)synurang_stream_finish(stream);
    else (void)synurang_stream_fail_error(stream, sent, 13, "debug response construction failed");
}
static void vx_api_debug_read_poll(VxApiPending* pending) {
    VxApiDebugRead* command = (VxApiDebugRead*)pending;
    if (vx_debug_kv_read_poll(command->lease.pointer)) return;
    vx_api_debug_read_reply(command, command->stream);
    vx_api_debug_read_detach(command);
}
static void* vx_api_debug_read_open(SynurangStream* stream, const SynurangCallOptions* options, void* registry) {
    (void)options;
    VxApiDebugRead* command = calloc(1, sizeof(*command));
    if (!command) { (void)synurang_stream_fail_error(stream, SYNURANG_OUT_OF_MEMORY, 8, "debug command allocation failed"); return NULL; }
    command->registry = registry; command->pending.poll = vx_api_debug_read_poll;
    return command;
}
static void vx_api_read_debug_kv_cache_call(SynurangStream* stream, const VolvoxaiV1ReadDebugKVCacheRequest* in, void* data) {
    VxApiDebugRead* command = data;
    if (!command) return;
    VolvoxaiV1ReadDebugKVCacheResponse response;
    volvoxai_v1_read_debug_kv_cache_response_init(&response);
    VxReport report = VX_REPORT_INIT;
    int result = 0;
    uint32_t limit = in->has_token_limit ? in->field_token_limit : VX_API_READ_DEBUG_K_V_CACHE_REQUEST_TOKEN_LIMIT_DEFAULT;
    if (!vx_api_handle_acquire(command->registry, VX_API_HANDLE_DEBUG_SESSION, in->field_debug_session_id, &command->lease))
        result = vx_api_debug_error(response._allocator, &response.field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released debug session");
    else if (limit < VX_API_READ_DEBUG_K_V_CACHE_REQUEST_TOKEN_LIMIT_MINIMUM || limit > VX_API_READ_DEBUG_K_V_CACHE_REQUEST_TOKEN_LIMIT_MAXIMUM)
        result = vx_api_debug_error(response._allocator, &response.field_report, VX_STATUS_INVALID_ARGUMENT, "token_limit is out of range");
    else {
        VxDebugKVReadRequest request = {in->field_cache_id, in->field_slot, in->field_token_offset, limit};
        if (vx_debug_kv_read_begin(command->lease.pointer, &request, &report) != VX_STATUS_OK)
            result = vx_api_report_attach(response._allocator, &response.field_report, &report) ? 0 : -1;
        else if (vx_debug_kv_read_poll(command->lease.pointer)) {
            command->stream = synurang_stream_retain(stream);
            vx_api_pending_add(command->registry, &command->pending);
            volvoxai_v1_read_debug_kv_cache_response_free(&response);
            return;
        } else {
            volvoxai_v1_read_debug_kv_cache_response_free(&response);
            vx_api_debug_read_reply(command, stream);
            vx_api_handle_lease_release(&command->lease);
            return;
        }
    }
    SynurangStatus status = result ? SYNURANG_INTERNAL : vx_debug_read_debug_k_v_cache_respond(stream, &response);
    volvoxai_v1_read_debug_kv_cache_response_free(&response);
    vx_api_handle_lease_release(&command->lease);
    if (status == SYNURANG_OK) (void)synurang_stream_finish(stream);
    else (void)synurang_stream_fail_error(stream, status, 13, "debug response construction failed");
}
static void vx_api_debug_read_cancel_call(SynurangStream* stream, void* data) {
    (void)stream;
    VxApiDebugRead* command = data;
    if (command && command->stream && command->lease.pointer) vx_debug_kv_read_cancel(command->lease.pointer);
    vx_api_debug_read_detach(command);
}
static void vx_api_debug_read_destroy_call(void* data) {
    VxApiDebugRead* command = data;
    if (command && command->stream && command->lease.pointer) vx_debug_kv_read_cancel(command->lease.pointer);
    vx_api_debug_read_detach(command);
    free(command);
}

VX_API_UNARY(vx_api_create_debug_session, VolvoxaiV1CreateDebugSessionRequest, VolvoxaiV1DebugSessionInfo,
    volvoxai_v1_debug_session_info, vx_debug_create_debug_session_respond)
VX_API_UNARY(vx_api_get_debug_session, VolvoxaiV1DebugSessionRef, VolvoxaiV1DebugSessionInfo,
    volvoxai_v1_debug_session_info, vx_debug_get_debug_session_respond)
VX_API_UNARY(vx_api_cancel_debug_session, VolvoxaiV1DebugSessionRef, VolvoxaiV1DebugSessionInfo,
    volvoxai_v1_debug_session_info, vx_debug_cancel_debug_session_respond)
VX_API_UNARY(vx_api_release_debug_session, VolvoxaiV1DebugSessionRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_debug_release_debug_session_respond)
VX_API_UNARY(vx_api_get_debug_plan, VolvoxaiV1DebugSessionRef, VolvoxaiV1GetDebugPlanResponse,
    volvoxai_v1_get_debug_plan_response, vx_debug_get_debug_plan_respond)
VX_API_UNARY(vx_api_list_debug_events, VolvoxaiV1ListDebugEventsRequest, VolvoxaiV1ListDebugEventsResponse,
    volvoxai_v1_list_debug_events_response, vx_debug_list_debug_events_respond)
VX_API_UNARY(vx_api_read_debug_tensor, VolvoxaiV1ReadDebugTensorRequest, VolvoxaiV1ReadDebugTensorResponse,
    volvoxai_v1_read_debug_tensor_response, vx_debug_read_debug_tensor_respond)
VX_API_UNARY(vx_api_export_debug_tensor, VolvoxaiV1ExportDebugTensorRequest, VolvoxaiV1TensorBatch,
    volvoxai_v1_tensor_batch, vx_debug_export_debug_tensor_respond)
VX_API_UNARY(vx_api_set_debug_tensor, VolvoxaiV1SetDebugTensorRequest, VolvoxaiV1DebugSessionInfo,
    volvoxai_v1_debug_session_info, vx_debug_set_debug_tensor_respond)
VX_API_UNARY(vx_api_get_debug_decode_state, VolvoxaiV1DebugSessionRef, VolvoxaiV1DebugDecodeState,
    volvoxai_v1_debug_decode_state, vx_debug_get_debug_decode_state_respond)
int vx_api_install_debug_handlers(SynurangInstance* instance, VxApiRegistry* registry) {
    VxDebugServiceHandlers handlers;
    memset(&handlers, 0, sizeof(handlers));
    handlers.create_debug_session.message = vx_api_create_debug_session_call;
    handlers.get_debug_session.message = vx_api_get_debug_session_call;
    handlers.cancel_debug_session.message = vx_api_cancel_debug_session_call;
    handlers.release_debug_session.message = vx_api_release_debug_session_call;
    handlers.get_debug_plan.message = vx_api_get_debug_plan_call;
    handlers.list_debug_events.message = vx_api_list_debug_events_call;
    handlers.read_debug_tensor.message = vx_api_read_debug_tensor_call;
    handlers.export_debug_tensor.message = vx_api_export_debug_tensor_call;
    handlers.get_debug_decode_state.message = vx_api_get_debug_decode_state_call;
    handlers.set_debug_tensor.message = vx_api_set_debug_tensor_call;
    handlers.read_debug_k_v_cache.open = vx_api_debug_read_open;
    handlers.read_debug_k_v_cache.message = vx_api_read_debug_kv_cache_call;
    handlers.read_debug_k_v_cache.cancel = vx_api_debug_read_cancel_call;
    handlers.read_debug_k_v_cache.destroy = vx_api_debug_read_destroy_call;
    handlers.step_debug_session.open = vx_api_debug_open;
    handlers.step_debug_session.message = vx_api_step_debug_session_call;
    handlers.step_debug_session.cancel = vx_api_debug_cancel_call;
    handlers.step_debug_session.destroy = vx_api_debug_destroy_call;
    handlers.continue_debug_session.open = vx_api_debug_open;
    handlers.continue_debug_session.message = vx_api_continue_debug_session_call;
    handlers.continue_debug_session.cancel = vx_api_debug_cancel_call;
    handlers.continue_debug_session.destroy = vx_api_debug_destroy_call;
    return vx_debug_register(instance, &handlers, registry);
}
#undef VX_DEBUG_NEW
