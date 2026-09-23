/* Generated profiling service projection. Engine collection owns no transport. */
#include "vx_api.h"
#include "vx_api_convert.h"
#include "volvoxai_ffi.h"
#include "profiling.h"
#include "vx_thread.h"
#include <stdatomic.h>
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
static int vx_api_trace_info(VolvoxaiV1TraceInfo* response, int64_t id, VxApiTrace* owned, VxTraceView* view) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    vx_api_trace_view(owned, view);
    response->field_trace_id = id;
    response->field_state = view->state;
    response->field_capacity_bytes = view->capacity_bytes;
    response->field_event_count = view->count;
    response->field_dropped_events = view->dropped;
    response->field_active_operations = view->active - view->pending;
    response->field_pending_device_intervals = view->pending;
    response->field_host_clock_resolution_ns = view->clock_resolution_ns;
    response->field_detail = view->detail;
    response->field_device_timing = view->device_timing;
    response->field_memory = view->memory;
    if (view->device_count) {
        response->field_devices.data = allocator->allocate(allocator->context,
            sizeof(*response->field_devices.data) * view->device_count);
        if (!response->field_devices.data) return -1;
        response->field_devices.len = response->field_devices.cap = view->device_count;
        for (size_t i = 0; i < view->device_count; i++)
            volvoxai_v1_trace_device_coverage_init_with_allocator(&response->field_devices.data[i], allocator);
        for (size_t i = 0; i < view->device_count; i++) {
            VolvoxaiV1TraceDeviceCoverage* out = &response->field_devices.data[i];
            const VxTraceDeviceCoverage* in = &view->devices[i];
            if (!vx_api_trace_text_field(allocator, &out->field_backend, in->backend)) return -1;
            out->field_support = in->support; out->field_node_timing_available = in->node_timing_available;
            out->field_pass_intervals = in->pass_intervals; out->field_node_intervals = in->node_intervals;
            out->field_failed_intervals = in->failed_intervals; out->field_unavailable_passes = in->unavailable_passes;
            out->field_program_timing_available = in->program_timing_available;
            out->field_program_intervals = in->program_intervals;
            out->field_calibrated_intervals = in->calibrated_intervals;
            out->field_bounded_intervals = in->bounded_intervals;
            out->field_copy_intervals = in->copy_intervals;
            out->field_host_copy_calls = in->host_copy_calls;
            out->field_host_wait_calls = in->host_wait_calls;
            out->field_host_submit_calls = in->host_submit_calls;
            out->field_host_awaits = in->host_awaits;
            out->field_splits_passes = in->splits_passes; out->field_adds_barriers = in->adds_barriers;
        }
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
    uint64_t capacity = request->has_capacity_bytes ? request->field_capacity_bytes : 4u * 1024u * 1024u;
    if (capacity < 4096 || capacity > 64u * 1024u * 1024u ||
        (request->field_detail != VOLVOXAI_V1_TRACE_DETAIL_BASIC &&
         request->field_detail != VOLVOXAI_V1_TRACE_DETAIL_NODES))
        return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_INVALID_ARGUMENT, "invalid trace capacity or detail");
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_RUNTIME, request->field_runtime_id, &lease))
        return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released runtime");
    VxApiTrace* owned = calloc(1, sizeof(*owned));
    if (!owned) { vx_api_handle_lease_release(&lease); return -1; }
    atomic_init(&owned->references, 1);
    if (pthread_mutex_init(&owned->mutex, NULL) != 0) {
        free(owned); vx_api_handle_lease_release(&lease); return -1;
    }
    owned->trace = vx_trace_create((size_t)capacity, request->field_detail, request->field_device_timing, request->field_memory);
    if (!owned->trace) {
        vx_api_trace_release_trace(owned); vx_api_handle_lease_release(&lease);
        return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_OUT_OF_MEMORY, "trace allocation failed");
    }
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
static int vx_api_stop_trace(const VolvoxaiV1TraceRef* request,
    VolvoxaiV1TraceInfo* response, void* registry) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_TRACE, request->field_trace_id, &lease))
        return vx_api_trace_report_status(response->_allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released trace");
    VxApiTrace* owned = lease.pointer;
    vx_api_trace_stop_trace(owned, 0);
    VxTraceView view;
    int result = vx_api_trace_info(response, request->field_trace_id, owned, &view);
    vx_api_handle_lease_release(&lease);
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
static int vx_api_read_trace(const VolvoxaiV1ReadTraceRequest* request,
    VolvoxaiV1TracePage* response, void* registry) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_TRACE, request->field_trace_id, &lease))
        return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released trace");
    VxApiTrace* owned = lease.pointer;
    VxTraceView view;
    vx_api_trace_view(owned, &view);
    int result = vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_OK, "trace page");
    if (result) goto done;
    uint32_t limit = request->has_limit ? request->field_limit : 256;
    if (view.state != VX_TRACE_STATE_READY) {
        result = vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_BUSY, "stop and drain trace before reading"); goto done;
    }
    if (!limit || limit > 4096 || request->field_offset > view.count) {
        result = vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_INVALID_ARGUMENT, "invalid trace offset or limit"); goto done;
    }
    size_t count = (size_t)(view.count - request->field_offset);
    if (count > limit) count = limit;
    if (count) {
        response->field_events.data = allocator->allocate(allocator->context, sizeof(*response->field_events.data) * count);
        if (!response->field_events.data) { result = -1; goto done; }
        response->field_events.len = response->field_events.cap = count;
        for (size_t i = 0; i < count; i++)
            volvoxai_v1_trace_event_init_with_allocator(&response->field_events.data[i], allocator);
        for (size_t i = 0; i < count; i++)
            if (!vx_api_trace_event(&response->field_events.data[i], &view.records[request->field_offset + i])) { result = -1; goto done; }
    }
    response->field_next_offset = request->field_offset + count;
    response->field_eof = response->field_next_offset == view.count;
    if (view.memory && request->field_offset == 0) {
        response->field_process_memory.data = allocator->allocate(allocator->context, 2 * sizeof(*response->field_process_memory.data));
        if (!response->field_process_memory.data) { result = -1; goto done; }
        response->field_process_memory.len = response->field_process_memory.cap = 2;
        for (size_t i = 0; i < 2; i++)
            volvoxai_v1_memory_snapshot_init_with_allocator(&response->field_process_memory.data[i], allocator);
        for (size_t i = 0; i < 2; i++) {
            VolvoxaiV1MemorySnapshot* snapshot = &response->field_process_memory.data[i];
            snapshot->field_sequence = i + 1;
            if (!vx_api_memory_snapshot(snapshot, VOLVOXAI_V1_MEMORY_OWNER_KIND_PROCESS, 0,
                &view.samples[i], view.memory_start_ns[i], view.memory_end_ns[i])) { result = -1; goto done; }
        }
    }
done:
    vx_api_handle_lease_release(&lease);
    return result;
}
static int vx_api_export_chrome_trace(const VolvoxaiV1ExportChromeTraceRequest* request,
    VolvoxaiV1TraceChunk* response, void* registry) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    uint32_t limit = request->has_limit ? request->field_limit : 128;
    if (!limit || limit > 1024)
        return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_INVALID_ARGUMENT, "invalid chunk limit");
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_TRACE, request->field_trace_id, &lease))
        return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released trace");
    VxApiTrace* owned = lease.pointer;
    char* json = NULL;
    size_t size = 0;
    VxTraceView view; vx_api_trace_view(owned, &view);
    int exported = view.state != VX_TRACE_STATE_READY ? 0 :
        request->field_offset > view.count ? -2 :
        vx_trace_export(&view, (size_t)request->field_offset, limit, &json, &size);
    int result;
    if (exported == -2) result = vx_api_trace_report_status(allocator, &response->field_report,
        VX_STATUS_INVALID_ARGUMENT, "invalid event offset");
    else if (exported < 1) result = vx_api_trace_report_status(allocator, &response->field_report,
        exported < 0 ? VX_STATUS_OUT_OF_MEMORY : VX_STATUS_BUSY, "trace export not ready");
    else {
        size_t count = view.count - (size_t)request->field_offset;
        if (count > limit) count = limit;
        result = synurang_lite_bytes_assign(allocator, &response->field_data,
            json, size) == SYNURANG_LITE_OK ? 0 : -1;
        response->field_next_offset = request->field_offset + count;
        response->field_eof = response->field_next_offset == view.count;
        if (!result) result = vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_OK, "trace exported");
    }
    free(json);
    vx_api_handle_lease_release(&lease);
    return result;
}
static int vx_api_release_trace(const VolvoxaiV1TraceRef* request,
    VolvoxaiV1OperationReport* response, void* registry) {
    /* Stop immediately even if a concurrent read retains an operation lease. */
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (vx_api_handle_acquire(registry, VX_API_HANDLE_TRACE, request->field_trace_id, &lease)) {
        vx_api_trace_stop_trace(lease.pointer, 0);
        vx_api_handle_lease_release(&lease);
    }
    vx_api_handle_remove(registry, VX_API_HANDLE_TRACE, request->field_trace_id);
    VxReport report = VX_REPORT_INIT;
    return vx_api_report_from_native(response, &report) ? 0 : -1;
}
static int vx_api_get_memory_snapshot(const VolvoxaiV1GetMemorySnapshotRequest* request,
    VolvoxaiV1MemorySnapshotResponse* response, void* registry) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    int kind = VOLVOXAI_V1_MEMORY_OWNER_KIND_PROCESS;
    uint64_t id = 0;
    if (request->which_scope == 2 || request->which_scope == 3) {
        int context = request->which_scope == 3;
        id = context ? request->field_context_id : request->field_runtime_id;
        kind = context ? VOLVOXAI_V1_MEMORY_OWNER_KIND_EXECUTION_CONTEXT : VOLVOXAI_V1_MEMORY_OWNER_KIND_RUNTIME;
        if (!vx_api_handle_acquire(registry, context ? VX_API_HANDLE_CONTEXT : VX_API_HANDLE_RUNTIME, (int64_t)id, &lease))
            return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED, "unknown or released memory scope");
    } else if (request->which_scope != 1 || !request->field_process)
        return vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_INVALID_ARGUMENT, "select a memory scope");
    uint64_t start = vx_trace_now_ns();
    VxMemoryView view = {0};
    VxStatus observed = VX_STATUS_OK;
    if (request->which_scope == 2) observed = vx_runtime_memory_view(lease.pointer, &view);
    if (request->which_scope == 3) observed = vx_context_memory_view(lease.pointer, &view);
    if (observed != VX_STATUS_OK) {
        vx_api_handle_lease_release(&lease);
        return vx_api_trace_report_status(allocator, &response->field_report, observed, "memory scope unavailable");
    }
    VxProcessMemorySampleV1 sample = VX_PROCESS_MEMORY_SAMPLE_V1_INIT;
    int process = request->which_scope == 1 || request->field_include_process;
    if (process) (void)vx_process_memory_sample_v1(&sample);
    response->field_snapshot = allocator->allocate(allocator->context, sizeof(*response->field_snapshot));
    int result = -1;
    if (response->field_snapshot) {
        volvoxai_v1_memory_snapshot_init_with_allocator(response->field_snapshot, allocator);
        int ok = vx_api_memory_snapshot(response->field_snapshot, kind, id, process ? &sample : NULL, start, vx_trace_now_ns());
        if (ok && request->which_scope == 2)
            ok = vx_api_memory_counter(response->field_snapshot, "unconsumed_result_budget", view.active_output_bytes, VOLVOXAI_V1_MEMORY_METRIC_RESERVED);
        if (ok && request->which_scope == 3 && view.has_arena)
            ok = vx_api_memory_counter(response->field_snapshot, "host_arena", view.arena_capacity_bytes, VOLVOXAI_V1_MEMORY_METRIC_CAPACITY);
        if (ok && request->which_scope == 3)
            ok = vx_api_memory_counter(response->field_snapshot, "retained_result_capacity", view.result_capacity_bytes, VOLVOXAI_V1_MEMORY_METRIC_CAPACITY) &&
                vx_api_memory_counter(response->field_snapshot, "idle_result_capacity", view.idle_result_bytes, VOLVOXAI_V1_MEMORY_METRIC_CAPACITY);
        if (ok) result = vx_api_trace_report_status(allocator, &response->field_report, VX_STATUS_OK, "memory snapshot; partial inventory");
    }
    vx_api_handle_lease_release(&lease);
    return result;
}

VX_API_UNARY(vx_api_start_trace, VolvoxaiV1StartTraceRequest, VolvoxaiV1TraceInfo,
    volvoxai_v1_trace_info, vx_profiling_start_trace_respond)
VX_API_UNARY(vx_api_stop_trace, VolvoxaiV1TraceRef, VolvoxaiV1TraceInfo,
    volvoxai_v1_trace_info, vx_profiling_stop_trace_respond)
VX_API_UNARY(vx_api_get_trace, VolvoxaiV1TraceRef, VolvoxaiV1TraceInfo,
    volvoxai_v1_trace_info, vx_profiling_get_trace_respond)
VX_API_UNARY(vx_api_read_trace, VolvoxaiV1ReadTraceRequest, VolvoxaiV1TracePage,
    volvoxai_v1_trace_page, vx_profiling_read_trace_respond)
VX_API_UNARY(vx_api_export_chrome_trace, VolvoxaiV1ExportChromeTraceRequest, VolvoxaiV1TraceChunk,
    volvoxai_v1_trace_chunk, vx_profiling_export_chrome_trace_respond)
VX_API_UNARY(vx_api_release_trace, VolvoxaiV1TraceRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_profiling_release_trace_respond)
VX_API_UNARY(vx_api_get_memory_snapshot, VolvoxaiV1GetMemorySnapshotRequest, VolvoxaiV1MemorySnapshotResponse,
    volvoxai_v1_memory_snapshot_response, vx_profiling_get_memory_snapshot_respond)
int vx_api_install_profiling_handlers(SynurangInstance* instance, VxApiRegistry* registry) {
    VxProfilingServiceHandlers handlers;
    memset(&handlers, 0, sizeof(handlers));
    handlers.start_trace.message = vx_api_start_trace_call;
    handlers.stop_trace.message = vx_api_stop_trace_call;
    handlers.get_trace.message = vx_api_get_trace_call;
    handlers.read_trace.message = vx_api_read_trace_call;
    handlers.export_chrome_trace.message = vx_api_export_chrome_trace_call;
    handlers.release_trace.message = vx_api_release_trace_call;
    handlers.get_memory_snapshot.message = vx_api_get_memory_snapshot_call;
    return vx_profiling_register(instance, &handlers, registry);
}
