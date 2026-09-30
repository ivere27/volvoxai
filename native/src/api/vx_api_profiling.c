/* Generated profiling service projection. Engine collection owns no transport. */
#include "vx_api.h"
#include "vx_api_convert.h"
#include "volvoxai_ffi.h"
#include "profiling.h"
#include "generated/api_limits.h"
#include "vx_thread.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    atomic_uint references;
    pthread_mutex_t mutex;
    VxRuntime* runtime;
    VxTrace* trace;
} VxApiTrace;
static void vx_api_trace_retain_trace(void* pointer) {
    VxApiTrace* trace = pointer;
    atomic_fetch_add_explicit(&trace->references, 1, memory_order_relaxed);
}
static void vx_api_trace_stop_trace(VxApiTrace* trace, int retiring) {
    pthread_mutex_lock(&trace->mutex);
    VxRuntime* runtime = trace->runtime;
    if (runtime) {
        vx_runtime_trace_stop(runtime, trace->trace);
        VxTraceView view; vx_trace_view(trace->trace, &view);
        if (retiring || view.state == VX_TRACE_STATE_READY) trace->runtime = NULL;
        else runtime = NULL; /* Keep the drain guard reachable until final release. */
    }
    pthread_mutex_unlock(&trace->mutex);
    if (runtime) vx_runtime_release(runtime);
}
static void vx_api_trace_view(VxApiTrace* owned, VxTraceView* view) {
    vx_trace_view(owned->trace, view);
    if (view->state == VX_TRACE_STATE_READY) vx_api_trace_stop_trace(owned, 0);
}
static void vx_api_trace_release_trace(void* pointer) {
    VxApiTrace* trace = pointer;
    if (atomic_fetch_sub_explicit(&trace->references, 1, memory_order_acq_rel) != 1) return;
    vx_trace_abandon(trace->trace);
    vx_api_trace_stop_trace(trace, 1);
    vx_trace_release(trace->trace);
    pthread_mutex_destroy(&trace->mutex);
    free(trace);
}
static int vx_api_trace_report_status(const SynurangLiteAllocator* allocator,
    VolvoxaiV1OperationReport** report, VxStatus status, const char* message) {
    VxOperationCode code = status == VX_STATUS_OK ? VX_CODE_NONE :
        status == VX_STATUS_BUSY ? VX_CODE_BUSY :
        status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
        status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY : VX_CODE_INVALID_ARGUMENT;
    return vx_api_report_fail(allocator, report, status, VX_STAGE_NONE, code, message) ? 0 : -1;
}
static int vx_api_trace_text_field(const SynurangLiteAllocator* allocator, SynurangLiteBytes* field, const char* text) {
    return synurang_lite_bytes_assign(allocator, field, text, strlen(text)) == SYNURANG_LITE_OK;
}
#define VX_TRACE_NEW(parent, field, codec) do { \
    (parent)->field_##field = (parent)->_allocator->allocate((parent)->_allocator->context, sizeof(*(parent)->field_##field)); \
    if (!(parent)->field_##field) return -1; \
    volvoxai_v1_##codec##_init_with_allocator((parent)->field_##field, (parent)->_allocator); \
} while (0)
static int vx_api_trace_info(VolvoxaiV1TraceInfo* response, int64_t id, VxApiTrace* owned, VxTraceView* view) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    vx_api_trace_view(owned, view);
    response->field_trace_id = id;
    response->field_state = view->state;
    VX_TRACE_NEW(response, options, trace_options);
    VolvoxaiV1TraceOptions* options = response->field_options;
    options->field_detail = view->detail;
    options->field_device_timing = view->device_timing;
    options->field_memory = view->memory;
    options->field_utilization = view->resources;
    options->field_execution_plans = view->execution_plans;
    options->has_capacity_bytes = 1; options->field_capacity_bytes = view->capacity_bytes;
    options->has_sample_interval_ns = 1; options->field_sample_interval_ns = view->sample_interval_ns;
    if (view->external_annotations || view->external_capture) {
        VX_TRACE_NEW(options, external, trace_external_options);
        options->field_external->field_annotations = view->external_annotations;
        options->field_external->field_capture = view->external_capture;
    }
    for (int i = 0; i < VX_TRACE_EXTERNAL_COUNT; i++) {
        if (view->external[i].status == VX_OBSERVATION_UNSPECIFIED) continue;
        VolvoxaiV1TraceExternalCoverage* out = volvoxai_v1_trace_info_add_external(response);
        if (!out || !vx_api_trace_text_field(allocator, &out->field_mechanism, vx_trace_external_mechanism_name(i)) ||
            !vx_api_trace_text_field(allocator, &out->field_backend, vx_trace_external_mechanism_backend(i))) return -1;
        out->field_status = view->external[i].status;
        out->field_ranges = view->external[i].ranges;
    }
    response->field_capture_origin_ns = view->origin_ns;
    VX_TRACE_NEW(response, events, record_count);
    response->field_events->field_count = view->count;
    response->field_events->field_dropped = view->dropped;
    VX_TRACE_NEW(response, resource_snapshots, record_count);
    response->field_resource_snapshots->field_count = view->resource_count;
    response->field_resource_snapshots->field_dropped = view->dropped_resource_samples;
    VX_TRACE_NEW(response, plans, record_count);
    response->field_plans->field_count = view->plan_count;
    response->field_plans->field_dropped = view->dropped_plans;
    response->field_active_operations = view->active - view->pending;
    response->field_pending_device_intervals = view->pending;
    response->has_host_clock_resolution_ns = view->clock_resolution_ns != 0;
    response->field_host_clock_resolution_ns = view->clock_resolution_ns;
    response->field_collector_bytes = view->collector_bytes;
    for (size_t i = 0; i < view->device_count; i++) {
        VolvoxaiV1TraceDeviceCoverage* out = volvoxai_v1_trace_info_add_devices(response);
        const VxTraceDeviceCoverage* in = &view->devices[i];
        if (!out || !vx_api_trace_text_field(allocator, &out->field_backend, in->backend)) return -1;
        out->field_support = in->support;
        out->field_node_timing_available = in->node_timing_available;
        out->field_program_timing_available = in->program_timing_available;
        out->field_splits_passes = in->splits_passes; out->field_adds_barriers = in->adds_barriers;
        VX_TRACE_NEW(out, device_intervals, trace_device_interval_counts);
        VolvoxaiV1TraceDeviceIntervalCounts* device = out->field_device_intervals;
        device->field_passes = in->pass_intervals; device->field_nodes = in->node_intervals;
        device->field_programs = in->program_intervals; device->field_copies = in->copy_intervals;
        device->field_calibrated = in->calibrated_intervals; device->field_bounded = in->bounded_intervals;
        device->field_failed = in->failed_intervals; device->field_unsupported_passes = in->unsupported_passes;
        VX_TRACE_NEW(out, host_activities, trace_host_activity_counts);
        VolvoxaiV1TraceHostActivityCounts* host = out->field_host_activities;
        host->field_copies = in->host_copy_calls; host->field_submits = in->host_submit_calls;
        host->field_synchronizations = in->host_synchronize_calls; host->field_completions = in->host_completions;
    }
    for (int i = 0; i < VX_MEMORY_ALLOCATOR_COUNT; i++) {
        const VxAllocatorMemory* in = &view->allocators[i];
        if (!in->observed) continue;
        VolvoxaiV1TraceAllocatorMemory* out = volvoxai_v1_trace_info_add_allocators(response);
        if (!out || !vx_api_trace_text_field(allocator, &out->field_allocator, vx_memory_allocator_name(i)) ||
            !vx_api_trace_text_field(allocator, &out->field_backend, vx_memory_allocator_backend(i))) return -1;
        out->field_space = vx_memory_allocator_space(i);
        out->field_inventory = VOLVOXAI_V1_MEMORY_INVENTORY_KIND_PARTIAL;
        out->field_scope = i == VX_MEMORY_WEBGPU_BUFFER
            ? VOLVOXAI_V1_MEMORY_OWNER_KIND_BACKEND_SHARED
            : VOLVOXAI_V1_MEMORY_OWNER_KIND_RUNTIME;
        out->field_observation_start_ns = in->start_ns;
        out->field_existing_bytes = in->existing;
        out->field_live_bytes = in->live;
        out->field_peak_bytes = in->peak;
        out->field_allocated_bytes = in->allocated;
        out->field_freed_bytes = in->freed;
        out->field_dropped_events = in->dropped;
        out->field_accounting_complete = in->complete;
    }
    return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_OK, "trace");
}
static int vx_api_start_trace(const VolvoxaiV1StartTraceRequest* request,
    VolvoxaiV1TraceInfo* response, void* registry) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VolvoxaiV1TraceOptions defaults;
    volvoxai_v1_trace_options_init(&defaults);
    const VolvoxaiV1TraceOptions* options = request->field_options ? request->field_options : &defaults;
    uint64_t capacity = options->has_capacity_bytes ? options->field_capacity_bytes :
        VX_API_TRACE_OPTIONS_CAPACITY_BYTES_DEFAULT;
    uint64_t interval = options->has_sample_interval_ns ? options->field_sample_interval_ns :
        VX_API_TRACE_OPTIONS_SAMPLE_INTERVAL_NS_DEFAULT;
    if (interval < VX_API_TRACE_OPTIONS_SAMPLE_INTERVAL_NS_MINIMUM || interval > VX_API_TRACE_OPTIONS_SAMPLE_INTERVAL_NS_MAXIMUM)
        return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_INVALID_ARGUMENT, "sample_interval_ns is out of range");
    if (capacity < VX_API_TRACE_OPTIONS_CAPACITY_BYTES_MINIMUM || capacity > VX_API_TRACE_OPTIONS_CAPACITY_BYTES_MAXIMUM ||
        (options->field_detail != VOLVOXAI_V1_TRACE_DETAIL_BASIC &&
         options->field_detail != VOLVOXAI_V1_TRACE_DETAIL_NODES))
        return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_INVALID_ARGUMENT, "capacity_bytes or detail is out of range");
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_RUNTIME, request->field_runtime_id, &lease))
        return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released runtime");
    VxApiTrace* owned = calloc(1, sizeof(*owned));
    if (!owned) { vx_api_handle_lease_release(&lease); return -1; }
    atomic_init(&owned->references, 1);
    if (pthread_mutex_init(&owned->mutex, NULL) != 0) {
        free(owned); vx_api_handle_lease_release(&lease); return -1;
    }
    owned->trace = vx_trace_create((size_t)capacity, options->field_detail, options->field_device_timing, options->field_memory,
        options->field_utilization, interval, options->field_execution_plans);
    if (!owned->trace) {
        vx_api_trace_release_trace(owned); vx_api_handle_lease_release(&lease);
        return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_OUT_OF_MEMORY, "trace allocation failed");
    }
    if (options->field_external)
        vx_trace_set_external(owned->trace, options->field_external->field_annotations,
                              options->field_external->field_capture);
    VxStatus status = vx_runtime_trace_start(lease.pointer, owned->trace);
    if (status != VX_STATUS_OK) {
        vx_api_trace_release_trace(owned); vx_api_handle_lease_release(&lease);
        return vx_api_trace_report_status(allocator, &response->field_report, status, "runtime already collecting/draining or closed");
    }
    owned->runtime = lease.pointer;
    vx_runtime_retain(owned->runtime);
    int64_t id = vx_api_handle_insert_with_lineage(registry, VX_API_HANDLE_TRACE,
        owned, vx_api_trace_retain_trace, vx_api_trace_release_trace, &lease.lineage);
    vx_api_handle_lease_release(&lease);
    if (!id) { vx_api_trace_release_trace(owned); return -1; }
    VxTraceView view;
    int result = vx_api_trace_info(response, id, owned, &view);
    if (result) vx_api_handle_remove(registry, VX_API_HANDLE_TRACE, id);
    return result;
}
static int vx_api_get_trace(const VolvoxaiV1TraceRef* request,
    VolvoxaiV1TraceInfo* response, void* registry) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_TRACE, request->field_trace_id, &lease))
        return vx_api_trace_report_status(response->_allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released trace");
    VxTraceView view;
    int result = vx_api_trace_info(response, request->field_trace_id, lease.pointer, &view);
    vx_api_handle_lease_release(&lease);
    return result;
}
static int vx_api_trace_event(VolvoxaiV1TraceEvent* output, const VxTraceRecord* input) {
    const SynurangLiteAllocator* allocator = output->_allocator;
    output->field_sequence = input->sequence;
    output->has_plan_id = input->plan_id != 0;
    output->field_plan_id = input->plan_id;
    if (input->kind == VX_TRACE_EVENT_MEMORY) {
        output->which_observation = 10;
        output->field_memory = allocator->allocate(allocator->context, sizeof(*output->field_memory));
        if (!output->field_memory) return 0;
        volvoxai_v1_trace_memory_event_init_with_allocator(output->field_memory, allocator);
        VolvoxaiV1TraceMemoryEvent* memory = output->field_memory;
        memory->field_timestamp_ns = input->start_ns;
        memory->field_allocation_id = input->allocation_id;
        memory->field_bytes = input->allocation_bytes;
        memory->field_live_bytes = input->live_bytes;
        memory->field_action = input->memory_action;
        memory->field_role = vx_memory_allocator_role(input->allocator);
        memory->field_space = vx_memory_allocator_space(input->allocator);
        if (!vx_api_trace_text_field(allocator, &memory->field_allocator,
            vx_memory_allocator_name(input->allocator))) return 0;
    } else if (input->kind == VX_TRACE_EVENT_DEVICE) {
        output->which_observation = 9;
        output->field_device = allocator->allocate(allocator->context, sizeof(*output->field_device));
        if (!output->field_device) return 0;
        volvoxai_v1_trace_device_interval_init_with_allocator(output->field_device, allocator);
        output->field_device->field_host_observed_ns = input->start_ns;
        output->field_device->field_elapsed_ns = input->duration_ns;
        if (input->clock.method) {
            VolvoxaiV1TraceClockCorrelation* clock = allocator->allocate(allocator->context, sizeof(*clock));
            if (!clock) return 0;
            volvoxai_v1_trace_clock_correlation_init_with_allocator(clock, allocator);
            output->field_device->field_correlation = clock;
            clock->field_method = input->clock.method;
            clock->field_earliest_start_ns = input->clock.earliest_ns;
            clock->field_latest_start_ns = input->clock.latest_ns;
        }
    } else {
        output->which_observation = 8;
        output->field_host = allocator->allocate(allocator->context, sizeof(*output->field_host));
        if (!output->field_host) return 0;
        volvoxai_v1_trace_host_span_init_with_allocator(output->field_host, allocator);
        output->field_host->field_start_ns = input->start_ns;
        output->field_host->field_duration_ns = input->duration_ns;
    }
    output->field_track_id = input->track_id;
    output->field_activity = input->activity;
    if (input->queue.queue_id) {
        VolvoxaiV1TraceQueue* queue = allocator->allocate(allocator->context, sizeof(*queue));
        if (!queue) return 0;
        volvoxai_v1_trace_queue_init_with_allocator(queue, allocator);
        output->field_queue = queue;
        queue->field_device_id = input->queue.device_id;
        queue->field_queue_id = input->queue.queue_id;
        queue->has_submission_id = input->queue.submission_id != 0;
        queue->field_submission_id = input->queue.submission_id;
    }
    if (input->activity == VX_TRACE_ACTIVITY_COPY) {
        VolvoxaiV1TraceCopy* copy = allocator->allocate(allocator->context, sizeof(*copy));
        if (!copy) return 0;
        volvoxai_v1_trace_copy_init_with_allocator(copy, allocator);
        output->field_copy = copy;
        copy->field_source = input->copy_source;
        copy->field_destination = input->copy_destination;
        copy->field_bytes = input->copy_bytes;
    }
    if (input->schedule_index >= 0) {
        output->field_node = allocator->allocate(allocator->context, sizeof(*output->field_node));
        if (!output->field_node) return 0;
        volvoxai_v1_trace_node_init_with_allocator(output->field_node, allocator);
        output->field_node->field_schedule_index = (uint32_t)input->schedule_index;
        output->field_node->field_fused = input->fused;
        if (!vx_api_trace_text_field(allocator, &output->field_node->field_output_name, input->output)) return 0;
    }
    output->field_phase = input->phase;
    if (input->program[0]) {
        output->field_program = allocator->allocate(allocator->context, sizeof(*output->field_program));
        if (!output->field_program) return 0;
        volvoxai_v1_trace_program_init_with_allocator(output->field_program, allocator);
        if (!vx_api_trace_text_field(allocator, &output->field_program->field_name, input->program) ||
            !vx_api_trace_text_field(allocator, &output->field_program->field_entry_point, input->entry)) return 0;
    }
    if (!vx_api_trace_text_field(allocator, &output->field_tensor_name, input->tensor)) return 0;
    output->field_metadata_truncated = input->truncated;
    if (!vx_api_trace_text_field(allocator, &output->field_name, input->name) ||
        !vx_api_trace_text_field(allocator, &output->field_backend, input->backend)) return 0;
    output->field_lineage = allocator->allocate(allocator->context, sizeof(*output->field_lineage));
    if (!output->field_lineage) return 0;
    volvoxai_v1_lineage_init_with_allocator(output->field_lineage, allocator);
    VolvoxaiV1Lineage* lineage = output->field_lineage;
    lineage->field_runtime_id = input->identity.runtime_id;
    lineage->field_model_id = input->identity.model_id;
    lineage->field_compiled_model_id = input->identity.compiled_model_id;
    lineage->field_context_id = input->identity.context_id;
    lineage->field_execution_id = input->identity.execution_id;
    lineage->field_graph_id = input->identity.graph_id;
    lineage->field_graph_revision = input->identity.graph_revision;
    return 1;
}
/* Leases a READY trace. Returns 0 with *view filled; otherwise the report
 * already names the refusal and the caller returns its result. */
static int vx_api_ready_trace(void* registry, int64_t id, VxApiHandleLease* lease, VxTraceView* view,
    const SynurangLiteAllocator* allocator, VolvoxaiV1OperationReport** report, int* result) {
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_TRACE, id, lease)) {
        *result = vx_api_trace_report_status(allocator, report, VX_STATUS_HANDLE_DISPOSED, "unknown or released trace");
        return 0;
    }
    vx_api_trace_view(lease->pointer, view);
    if (view->state == VX_TRACE_STATE_READY) return 1;
    *result = vx_api_trace_report_status(allocator, report, VX_STATUS_BUSY, "StopTrace has not completed; the trace is not READY");
    vx_api_handle_lease_release(lease);
    return 0;
}
static int vx_api_trace_page(const SynurangLiteAllocator* allocator, VolvoxaiV1OperationReport** report,
    int has_size, uint32_t size, uint64_t maximum, uint64_t fallback, const SynurangLiteBytes* token,
    uint64_t total, uint64_t* offset, uint64_t* count, int* result) {
    uint64_t limit = has_size ? size : fallback;
    if (!limit || limit > maximum) {
        *result = vx_api_trace_report_status(allocator, report, VX_STATUS_INVALID_ARGUMENT, "page_size is out of range");
        return 0;
    }
    if (!vx_api_page_token_offset(token, total, offset)) {
        *result = vx_api_trace_report_status(allocator, report, VX_STATUS_INVALID_ARGUMENT, "invalid page_token");
        return 0;
    }
    *count = total - *offset < limit ? total - *offset : limit;
    return 1;
}
static int vx_api_list_trace_events(const VolvoxaiV1ListTraceEventsRequest* request,
    VolvoxaiV1ListTraceEventsResponse* response, void* registry) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxTraceView view;
    int result = 0;
    uint64_t offset, count;
    if (!vx_api_ready_trace(registry, request->field_trace_id, &lease, &view, allocator, &response->field_report, &result))
        return result;
    if (vx_api_trace_page(allocator, &response->field_report, request->has_page_size, request->field_page_size,
            VX_API_LIST_TRACE_EVENTS_REQUEST_PAGE_SIZE_MAXIMUM, VX_API_LIST_TRACE_EVENTS_REQUEST_PAGE_SIZE_DEFAULT,
            &request->field_page_token, view.count, &offset, &count, &result)) {
        response->field_total_size = view.count;
        for (uint64_t i = 0; i < count && !result; i++) {
            VolvoxaiV1TraceEvent* event = volvoxai_v1_list_trace_events_response_add_events(response);
            if (!event || !vx_api_trace_event(event, &view.records[offset + i])) result = -1;
        }
        if (!result && !vx_api_next_page_token(allocator, &response->field_next_page_token, offset + count, view.count)) result = -1;
        if (!result) result = vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_OK, "trace events");
    }
    vx_api_handle_lease_release(&lease);
    return result;
}
static int vx_api_trace_summary_row(VolvoxaiV1TraceSummaryRow* out, const VxTraceSummaryRow* in) {
    const SynurangLiteAllocator* allocator = out->_allocator;
    VX_TRACE_NEW(out, lineage, lineage);
    out->field_lineage->field_model_id = in->identity.model_id;
    out->field_lineage->field_compiled_model_id = in->identity.compiled_model_id;
    out->field_lineage->field_context_id = in->identity.context_id;
    out->has_plan_id = in->plan_id != 0;
    out->field_plan_id = in->plan_id;
    if (!vx_api_trace_text_field(allocator, &out->field_backend, in->backend) ||
        !vx_api_trace_text_field(allocator, &out->field_name, in->name) ||
        !vx_api_trace_text_field(allocator, &out->field_output_name, in->output)) return -1;
    out->field_phase = in->phase;
    out->field_activity = in->activity;
    out->field_domain = in->device ? VOLVOXAI_V1_TRACE_TIME_DOMAIN_DEVICE : VOLVOXAI_V1_TRACE_TIME_DOMAIN_HOST;
    out->has_schedule_index = in->schedule_index >= 0;
    out->field_schedule_index = in->schedule_index >= 0 ? (uint32_t)in->schedule_index : 0;
    out->field_fused = in->fused != 0;
    if (in->step) {
        const char* sources[2] = {in->step->source_node_id, in->step->fused_source_node_id};
        for (int i = 0; i < 2; i++) {
            if (!sources[i][0]) continue;
            SynurangLiteBytes* source = volvoxai_v1_trace_summary_row_add_source_node_ids(out);
            if (!source || !vx_api_trace_text_field(allocator, source, sources[i])) return -1;
        }
        if (!vx_api_step_cost(allocator, &out->field_cost_per_call, in->step)) return -1;
    }
    out->field_count = in->count; out->field_total_ns = in->total_ns;
    out->field_min_ns = in->min_ns; out->field_median_ns = in->median_ns;
    out->field_p95_ns = in->p95_ns; out->field_max_ns = in->max_ns;
    out->field_share = in->share;
    out->has_achieved_flops_per_second = in->has_rates && in->flops_per_second > 0.0;
    out->field_achieved_flops_per_second = in->flops_per_second;
    out->has_achieved_bytes_per_second = in->has_rates;
    out->field_achieved_bytes_per_second = in->bytes_per_second;
    out->field_copy_bytes = in->copy_bytes;
    return 0;
}
static int vx_api_get_trace_summary(const VolvoxaiV1GetTraceSummaryRequest* request,
    VolvoxaiV1TraceSummary* response, void* registry) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxTraceView view;
    int result = 0;
    uint64_t max_rows = request->has_max_rows ? request->field_max_rows :
        VX_API_GET_TRACE_SUMMARY_REQUEST_MAX_ROWS_DEFAULT;
    if (request->field_group_by < VOLVOXAI_V1_TRACE_SUMMARY_GROUPING_NODE ||
        request->field_group_by > VOLVOXAI_V1_TRACE_SUMMARY_GROUPING_ACTIVITY ||
        max_rows < VX_API_GET_TRACE_SUMMARY_REQUEST_MAX_ROWS_MINIMUM ||
        max_rows > VX_API_GET_TRACE_SUMMARY_REQUEST_MAX_ROWS_MAXIMUM)
        return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_INVALID_ARGUMENT,
            "group_by or max_rows is out of range");
    if (!vx_api_ready_trace(registry, request->field_trace_id, &lease, &view, allocator, &response->field_report, &result))
        return result;
    VxTraceSummary summary;
    if (vx_trace_summarize(&view, (VxTraceSummaryGrouping)request->field_group_by, (size_t)max_rows, &summary) != 0) {
        vx_api_handle_lease_release(&lease);
        return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_OUT_OF_MEMORY,
            "trace summary allocation failed");
    }
    response->field_group_by = request->field_group_by;
    response->field_truncated = summary.truncated != 0;
    response->field_excluded_events = summary.excluded;
    response->field_events = allocator->allocate(allocator->context, sizeof(*response->field_events));
    if (!response->field_events) result = -1;
    else {
        volvoxai_v1_record_count_init_with_allocator(response->field_events, allocator);
        response->field_events->field_count = view.count;
        response->field_events->field_dropped = view.dropped;
    }
    for (size_t i = 0; i < summary.count && !result; i++) {
        VolvoxaiV1TraceSummaryRow* row = volvoxai_v1_trace_summary_add_rows(response);
        if (!row || vx_api_trace_summary_row(row, &summary.rows[i])) result = -1;
    }
    vx_trace_summary_free(&summary);
    if (!result) result = vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_OK, "trace summary");
    vx_api_handle_lease_release(&lease);
    return result;
}
static int vx_api_list_trace_resource_snapshots(const VolvoxaiV1ListTraceResourceSnapshotsRequest* request,
    VolvoxaiV1ListTraceResourceSnapshotsResponse* response, void* registry) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxTraceView view;
    int result = 0;
    uint64_t offset, count;
    if (!vx_api_ready_trace(registry, request->field_trace_id, &lease, &view, allocator, &response->field_report, &result))
        return result;
    if (vx_api_trace_page(allocator, &response->field_report, request->has_page_size, request->field_page_size,
            VX_API_LIST_TRACE_RESOURCE_SNAPSHOTS_REQUEST_PAGE_SIZE_MAXIMUM,
            VX_API_LIST_TRACE_RESOURCE_SNAPSHOTS_REQUEST_PAGE_SIZE_DEFAULT,
            &request->field_page_token, view.resource_count, &offset, &count, &result)) {
        response->field_total_size = view.resource_count;
        for (uint64_t i = 0; i < count && !result; i++) {
            VolvoxaiV1ResourceSnapshot* sample = volvoxai_v1_list_trace_resource_snapshots_response_add_resource_snapshots(response);
            if (!sample || !vx_api_resource_snapshot(sample, &view.resource_samples[offset + i])) result = -1;
        }
        if (!result && !vx_api_next_page_token(allocator, &response->field_next_page_token, offset + count, view.resource_count)) result = -1;
        if (!result) result = vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_OK, "trace resource snapshots");
    }
    vx_api_handle_lease_release(&lease);
    return result;
}
static int vx_api_get_trace_plan(const VolvoxaiV1GetTracePlanRequest* request,
    VolvoxaiV1GetTracePlanResponse* response, void* registry) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxTraceView view;
    int result = 0;
    if (!vx_api_ready_trace(registry, request->field_trace_id, &lease, &view, allocator, &response->field_report, &result))
        return result;
    if (!request->field_plan_id || request->field_plan_id > view.plan_count)
        result = vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_NOT_FOUND, "unknown plan_id");
    else {
        response->field_plan = allocator->allocate(allocator->context, sizeof(*response->field_plan));
        if (!response->field_plan) result = -1;
        else {
            volvoxai_v1_execution_plan_init_with_allocator(response->field_plan, allocator);
            result = vx_api_execution_plan(response->field_plan, vx_trace_plan_at(&view, (size_t)request->field_plan_id - 1)) ?
                vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_OK, "trace plan") : -1;
        }
    }
    vx_api_handle_lease_release(&lease);
    return result;
}
static int vx_api_export_chrome_trace(const VolvoxaiV1ExportChromeTraceRequest* request,
    VolvoxaiV1ExportChromeTraceResponse* response, void* registry) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxTraceView view;
    int result = 0;
    uint64_t offset, count;
    if (!vx_api_ready_trace(registry, request->field_trace_id, &lease, &view, allocator, &response->field_report, &result))
        return result;
    if (vx_api_trace_page(allocator, &response->field_report, request->has_page_size, request->field_page_size,
            VX_API_EXPORT_CHROME_TRACE_REQUEST_PAGE_SIZE_MAXIMUM, VX_API_EXPORT_CHROME_TRACE_REQUEST_PAGE_SIZE_DEFAULT,
            &request->field_page_token, view.count, &offset, &count, &result)) {
        char* json = NULL;
        size_t size = 0;
        uint64_t limit = request->has_page_size ? request->field_page_size : VX_API_EXPORT_CHROME_TRACE_REQUEST_PAGE_SIZE_DEFAULT;
        int exported = vx_trace_export(&view, (size_t)offset, (size_t)limit, &json, &size);
        if (exported < 1) result = vx_api_trace_report_status(allocator, &response->field_report,
            VX_STATUS_OUT_OF_MEMORY, "Chrome trace serialization failed");
        else if (synurang_lite_bytes_assign(allocator, &response->field_data, json, size) != SYNURANG_LITE_OK ||
            !vx_api_next_page_token(allocator, &response->field_next_page_token, offset + count, view.count)) result = -1;
        else result = vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_OK, "Chrome trace page");
        free(json);
    }
    vx_api_handle_lease_release(&lease);
    return result;
}
static int vx_api_annotate_trace(const VolvoxaiV1AnnotateTraceRequest* request,
    VolvoxaiV1OperationReport* response, void* registry) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxStatus status = VX_STATUS_HANDLE_DISPOSED;
    const char* message = "unknown or released trace";
    char name[96];
    if (!request->field_name.len || request->field_name.len >= sizeof(name) ||
        memchr(request->field_name.data, 0, request->field_name.len)) {
        status = VX_STATUS_INVALID_ARGUMENT; message = "name must be 1-95 bytes without NUL";
    } else if (request->field_end_ns < request->field_start_ns) {
        status = VX_STATUS_INVALID_ARGUMENT; message = "end_ns precedes start_ns";
    } else if (vx_api_handle_acquire(registry, VX_API_HANDLE_TRACE, request->field_trace_id, &lease)) {
        memcpy(name, request->field_name.data, request->field_name.len); name[request->field_name.len] = 0;
        status = vx_trace_annotate(((VxApiTrace*)lease.pointer)->trace, name,
            request->field_start_ns, request->field_end_ns);
        message = status == VX_STATUS_OK ? "annotation recorded" : "the trace is not COLLECTING";
        vx_api_handle_lease_release(&lease);
    }
    VxReport report = VX_REPORT_INIT;
    report.status = status;
    report.code = status == VX_STATUS_OK ? VX_CODE_NONE :
        status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED : VX_CODE_INVALID_ARGUMENT;
    snprintf(report.message, sizeof(report.message), "%s", message);
    return vx_api_report_from_native(response, &report) ? 0 : -1;
}
static int vx_api_release_trace(const VolvoxaiV1TraceRef* request,
    VolvoxaiV1OperationReport* response, void* registry) {
    /* Stop immediately even if a concurrent read retains an operation lease. */
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (vx_api_handle_acquire(registry, VX_API_HANDLE_TRACE, request->field_trace_id, &lease)) {
        /* Retire device tickets now: a waiting StopTrace holds a lease, but no
         * caller can read this trace after release, so it must not keep a GPU
         * query alive. The waiter then observes READY and replies. */
        vx_trace_abandon(((VxApiTrace*)lease.pointer)->trace);
        vx_api_trace_stop_trace(lease.pointer, 0);
        vx_api_handle_lease_release(&lease);
    }
    vx_api_handle_remove(registry, VX_API_HANDLE_TRACE, request->field_trace_id);
    VxReport report = VX_REPORT_INIT;
    return vx_api_report_from_native(response, &report) ? 0 : -1;
}
static int vx_api_get_resource_snapshot(const VolvoxaiV1GetResourceSnapshotRequest* request,
    VolvoxaiV1ResourceSnapshotResponse* response, void* registry) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    uint64_t capacity = request->has_max_allocations ? request->field_max_allocations :
        VX_API_GET_RESOURCE_SNAPSHOT_REQUEST_MAX_ALLOCATIONS_DEFAULT;
    if (capacity < VX_API_GET_RESOURCE_SNAPSHOT_REQUEST_MAX_ALLOCATIONS_MINIMUM ||
        capacity > VX_API_GET_RESOURCE_SNAPSHOT_REQUEST_MAX_ALLOCATIONS_MAXIMUM)
        return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_INVALID_ARGUMENT, "max_allocations is out of range");
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    int kind = VX_MEMORY_OWNER_MODULE;
    uint64_t id = 0;
    if (request->which_scope == 2 || request->which_scope == 3) {
        int context = request->which_scope == 3;
        id = context ? request->field_context_id : request->field_runtime_id;
        kind = context ? VX_MEMORY_OWNER_EXECUTION_CONTEXT : VX_MEMORY_OWNER_RUNTIME;
        if (!vx_api_handle_acquire(registry, context ? VX_API_HANDLE_CONTEXT : VX_API_HANDLE_RUNTIME, (int64_t)id, &lease))
            return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released resource scope");
    } else if (request->which_scope != 1)
        return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_INVALID_ARGUMENT, "select exactly one scope: module, runtime_id or context_id");
    /* Sample before allocating inspection buffers, which are not engine load. */
    VxResourceSample sample;
    vx_resource_sample(&sample, 1, request->field_include_device);
    VxMemoryInventory inventory = {.capacity = capacity};
    inventory.items = calloc(capacity, sizeof(*inventory.items));
    VxMemoryView view = {0};
    VxStatus status = inventory.items ? VX_STATUS_OK : VX_STATUS_OUT_OF_MEMORY;
    int result = -1;
    if (status == VX_STATUS_OK && request->which_scope == 3)
        status = vx_context_memory_inventory(lease.pointer, &inventory, &view);
    else if (status == VX_STATUS_OK) {
        if (request->which_scope == 2) status = vx_runtime_memory_view(lease.pointer, &view);
        /* Leases prevent concurrent release; never hold the registry lock
         * while entering an engine context. The inventory remains partial. */
        const size_t max_owners = 256;
        VxApiHandleLease* owners = calloc(max_owners, sizeof(*owners));
        if (!owners) status = VX_STATUS_OUT_OF_MEMORY;
        for (int owner_kind = VX_API_HANDLE_MODEL; owners && owner_kind <= VX_API_HANDLE_CONTEXT; owner_kind++) {
            size_t count = vx_api_handle_snapshot(registry, owner_kind, id, owners, max_owners, &inventory.truncated);
            for (size_t i = 0; i < count; i++) {
                if (owner_kind == VX_API_HANDLE_MODEL) vx_model_memory_inventory(owners[i].pointer, &inventory);
                else if (owner_kind == VX_API_HANDLE_COMPILED_MODEL) vx_compiled_memory_inventory(owners[i].pointer, &inventory);
                else {
                    VxMemoryView context_view;
                    if (vx_context_memory_inventory(owners[i].pointer, &inventory, &context_view) != VX_STATUS_OK)
                        inventory.truncated = 1;
                }
                vx_api_handle_lease_release(&owners[i]);
            }
        }
        free(owners);
    }
    sample.end_ns = vx_trace_now_ns();
    if (status == VX_STATUS_OK) {
        response->field_snapshot = allocator->allocate(allocator->context, sizeof(*response->field_snapshot));
        if (response->field_snapshot) {
            volvoxai_v1_resource_snapshot_init_with_allocator(response->field_snapshot, allocator);
            if (vx_api_resource_snapshot(response->field_snapshot, &sample) &&
                vx_api_memory_inventory(response->field_snapshot, kind, id, &inventory,
                    request->which_scope == 1 ? NULL : &view, request->which_scope == 2))
                result = vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_OK, "resource snapshot; partial allocation inventory");
        }
    } else result = vx_api_trace_report_status(allocator, &response->field_report, status, "resource scope unavailable");
    free(inventory.items);
    vx_api_handle_lease_release(&lease);
    return result;
}

/* StopTrace replies when the trace is READY. Host scopes that end after stop
 * wake the waiter; pending device timestamps resolve only when polled, so the
 * waiter re-polls while they remain. Transport cancellation stops waiting. */
typedef struct {
    VxApiPending pending;
    VxApiRegistry* registry;
    SynurangStream* stream;
    VxApiHandleLease lease;
    int64_t id;
} VxApiStopTrace;
static void vx_api_stop_trace_detach(VxApiStopTrace* command) {
    if (!command || !command->stream) return;
    vx_trace_watch_drain(((VxApiTrace*)command->lease.pointer)->trace, NULL, &command->pending);
    vx_api_pending_notify(&command->pending); /* Let another waiter re-arm the watch. */
    vx_api_pending_remove(&command->pending);
    vx_api_handle_lease_release(&command->lease);
    SynurangStream* stream = command->stream; command->stream = NULL;
    synurang_stream_release(stream);
}
static int vx_api_stop_trace_reply(VxApiStopTrace* command, SynurangStream* stream) {
    VolvoxaiV1TraceInfo response;
    volvoxai_v1_trace_info_init(&response);
    VxTraceView view;
    int result = vx_api_trace_info(&response, command->id, command->lease.pointer, &view);
    SynurangStatus status = result ? SYNURANG_INTERNAL : vx_profiling_stop_trace_respond(stream, &response);
    volvoxai_v1_trace_info_free(&response);
    if (status == SYNURANG_OK) (void)synurang_stream_finish(stream);
    else (void)synurang_stream_fail_error(stream, status, 13, "trace response construction failed");
    return 0;
}
static int vx_api_stop_trace_ready(VxApiStopTrace* command) {
    VxApiTrace* owned = command->lease.pointer;
    vx_trace_watch_drain(owned->trace, vx_api_pending_notify, &command->pending);
    VxTraceView view;
    vx_api_trace_view(owned, &view);
    if (view.state == VX_TRACE_STATE_READY) return 1;
    if (view.pending) vx_api_pending_notify(&command->pending);
    return 0;
}
static void vx_api_stop_trace_poll(VxApiPending* pending) {
    VxApiStopTrace* command = (VxApiStopTrace*)pending;
    if (!vx_api_stop_trace_ready(command)) return;
    vx_api_stop_trace_reply(command, command->stream);
    vx_api_stop_trace_detach(command);
}
static void* vx_api_stop_trace_open(SynurangStream* stream, const SynurangCallOptions* options, void* registry) {
    (void)options;
    VxApiStopTrace* command = calloc(1, sizeof(*command));
    if (!command) { (void)synurang_stream_fail_error(stream, SYNURANG_OUT_OF_MEMORY, 8, "trace command allocation failed"); return NULL; }
    command->registry = registry; command->pending.poll = vx_api_stop_trace_poll;
    return command;
}
static void vx_api_stop_trace_call(SynurangStream* stream, const VolvoxaiV1TraceRef* request, void* data) {
    VxApiStopTrace* command = data;
    if (!command) return;
    command->id = request->field_trace_id;
    if (!vx_api_handle_acquire(command->registry, VX_API_HANDLE_TRACE, command->id, &command->lease)) {
        VolvoxaiV1TraceInfo response;
        volvoxai_v1_trace_info_init(&response);
        int result = vx_api_trace_report_status(response._allocator, &response.field_report,
            VX_STATUS_HANDLE_DISPOSED, "unknown or released trace");
        SynurangStatus status = result ? SYNURANG_INTERNAL : vx_profiling_stop_trace_respond(stream, &response);
        volvoxai_v1_trace_info_free(&response);
        if (status == SYNURANG_OK) (void)synurang_stream_finish(stream);
        else (void)synurang_stream_fail_error(stream, status, 13, "trace response construction failed");
        return;
    }
    vx_api_trace_stop_trace(command->lease.pointer, 0);
    if (vx_api_stop_trace_ready(command)) {
        vx_trace_watch_drain(((VxApiTrace*)command->lease.pointer)->trace, NULL, &command->pending);
        vx_api_stop_trace_reply(command, stream);
        vx_api_handle_lease_release(&command->lease);
        return;
    }
    command->stream = synurang_stream_retain(stream);
    vx_api_pending_add(command->registry, &command->pending);
}
static void vx_api_stop_trace_cancel(SynurangStream* stream, void* data) {
    (void)stream;
    vx_api_stop_trace_detach(data);
}
static void vx_api_stop_trace_destroy(void* data) { vx_api_stop_trace_detach(data); free(data); }

VX_API_UNARY(vx_api_start_trace, VolvoxaiV1StartTraceRequest, VolvoxaiV1TraceInfo,
    volvoxai_v1_trace_info, vx_profiling_start_trace_respond)
VX_API_UNARY(vx_api_get_trace, VolvoxaiV1TraceRef, VolvoxaiV1TraceInfo,
    volvoxai_v1_trace_info, vx_profiling_get_trace_respond)
VX_API_UNARY(vx_api_list_trace_events, VolvoxaiV1ListTraceEventsRequest, VolvoxaiV1ListTraceEventsResponse,
    volvoxai_v1_list_trace_events_response, vx_profiling_list_trace_events_respond)
VX_API_UNARY(vx_api_list_trace_resource_snapshots, VolvoxaiV1ListTraceResourceSnapshotsRequest,
    VolvoxaiV1ListTraceResourceSnapshotsResponse, volvoxai_v1_list_trace_resource_snapshots_response,
    vx_profiling_list_trace_resource_snapshots_respond)
VX_API_UNARY(vx_api_get_trace_plan, VolvoxaiV1GetTracePlanRequest, VolvoxaiV1GetTracePlanResponse,
    volvoxai_v1_get_trace_plan_response, vx_profiling_get_trace_plan_respond)
VX_API_UNARY(vx_api_export_chrome_trace, VolvoxaiV1ExportChromeTraceRequest, VolvoxaiV1ExportChromeTraceResponse,
    volvoxai_v1_export_chrome_trace_response, vx_profiling_export_chrome_trace_respond)
VX_API_UNARY(vx_api_annotate_trace, VolvoxaiV1AnnotateTraceRequest, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_profiling_annotate_trace_respond)
VX_API_UNARY(vx_api_get_trace_summary, VolvoxaiV1GetTraceSummaryRequest, VolvoxaiV1TraceSummary,
    volvoxai_v1_trace_summary, vx_profiling_get_trace_summary_respond)
VX_API_UNARY(vx_api_release_trace, VolvoxaiV1TraceRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_profiling_release_trace_respond)
VX_API_UNARY(vx_api_get_resource_snapshot, VolvoxaiV1GetResourceSnapshotRequest, VolvoxaiV1ResourceSnapshotResponse,
    volvoxai_v1_resource_snapshot_response, vx_profiling_get_resource_snapshot_respond)
int vx_api_install_profiling_handlers(SynurangInstance* instance, VxApiRegistry* registry) {
    VxProfilingServiceHandlers handlers;
    memset(&handlers, 0, sizeof(handlers));
    handlers.start_trace.message = vx_api_start_trace_call;
    handlers.stop_trace.open = vx_api_stop_trace_open;
    handlers.stop_trace.message = vx_api_stop_trace_call;
    handlers.stop_trace.cancel = vx_api_stop_trace_cancel;
    handlers.stop_trace.destroy = vx_api_stop_trace_destroy;
    handlers.get_trace.message = vx_api_get_trace_call;
    handlers.list_trace_events.message = vx_api_list_trace_events_call;
    handlers.list_trace_resource_snapshots.message = vx_api_list_trace_resource_snapshots_call;
    handlers.get_trace_plan.message = vx_api_get_trace_plan_call;
    handlers.export_chrome_trace.message = vx_api_export_chrome_trace_call;
    handlers.annotate_trace.message = vx_api_annotate_trace_call;
    handlers.get_trace_summary.message = vx_api_get_trace_summary_call;
    handlers.release_trace.message = vx_api_release_trace_call;
    handlers.get_resource_snapshot.message = vx_api_get_resource_snapshot_call;
    return vx_profiling_register(instance, &handlers, registry);
}
#undef VX_TRACE_NEW
