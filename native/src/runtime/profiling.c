#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "profiling.h"
#include "vx_thread.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <inttypes.h>
#include <time.h>
#include <math.h>

/* A weak owner outlives trace buffers when an observed allocation is retained.
 * All collector access uses this mutex, including allocation callbacks. */
struct VxTraceMemoryOwner {
    atomic_uint references;
    atomic_int enabled;
    pthread_mutex_t mutex;
    VxTrace* trace;
    uint64_t next_source;
};
typedef struct {
    uint64_t owner;
    uint64_t key, id, bytes;
    VxMemoryAllocator allocator;
    unsigned char state; /* empty or occupied; deletion repairs the probe chain */
} VxTraceAllocation;

struct VxTrace {
    atomic_uint references;
    VxTraceMemoryOwner* owner;
    VxTraceDetail detail;
    int device_timing;
    int stopped, memory, sampled_stop, finalized, abandoned;
    size_t device_count;
    VxTraceDeviceCoverage devices[VX_TRACE_MAX_BACKENDS];
    uint64_t origin_ns, active, dropped, clock_resolution_ns;
    size_t capacity, count, capacity_bytes, pending, pending_device, first_pending;
    VxTraceRecord* records;
    uint64_t memory_start_ns[2], memory_end_ns[2];
    VxProcessMemorySampleV1 samples[2];
    VxAllocatorMemory allocators[VX_MEMORY_ALLOCATOR_COUNT];
    VxTraceAllocation* allocations;
    size_t allocation_capacity;
    uint64_t next_allocation_id;
    int memory_bridge_started;
    uint32_t memory_bridge_ticket;
    void (*memory_bridge_stop)(uint32_t);
};
/* Called only under the collector mutex. Backend names are fixed engine labels. */
static VxTraceDeviceCoverage* vx_trace_coverage(VxTrace* trace, const char* backend) {
    if (!backend) backend = "";
    for (size_t i = 0; i < trace->device_count; i++)
        if (!strcmp(trace->devices[i].backend, backend)) return &trace->devices[i];
    if (trace->device_count == VX_TRACE_MAX_BACKENDS) return NULL;
    VxTraceDeviceCoverage* coverage = &trace->devices[trace->device_count++];
    snprintf(coverage->backend, sizeof(coverage->backend), "%s", backend);
    return coverage;
}
static void vx_trace_device_count(VxTrace* trace, const VxTraceRecord* e, int success) {
    VxTraceDeviceCoverage* coverage = vx_trace_coverage(trace, e->backend);
    if (!coverage) return;
    if (!success) coverage->failed_intervals++;
    else if (e->activity == VX_TRACE_ACTIVITY_COPY) coverage->copy_intervals++;
    else if (e->program[0]) coverage->program_intervals++;
    else if (e->schedule_index >= 0) coverage->node_intervals++;
    else coverage->pass_intervals++;
    if (success && e->clock.method == VX_TRACE_CLOCK_METHOD_CALIBRATED) coverage->calibrated_intervals++;
    if (success && e->clock.method == VX_TRACE_CLOCK_METHOD_BOUNDED) coverage->bounded_intervals++;
}
static atomic_uint_fast64_t vx_trace_next_track = 1;
static atomic_uint_fast64_t vx_trace_next_identity = 1;
static _Thread_local uint64_t vx_trace_thread_track;

uint64_t vx_trace_next_id(void) {
    return atomic_fetch_add_explicit(&vx_trace_next_identity, 1, memory_order_relaxed);
}
uint64_t vx_trace_object_id(atomic_uint_fast64_t* identity) {
    uint64_t value = atomic_load_explicit(identity, memory_order_relaxed);
    if (!value) {
        uint64_t candidate = vx_trace_next_id();
        if (atomic_compare_exchange_strong_explicit(identity, &value, candidate,
                memory_order_relaxed, memory_order_relaxed)) value = candidate;
    }
    return value;
}
VxTraceClock vx_trace_clock_bounds(uint64_t before, uint64_t after,
    uint64_t total, uint64_t offset, uint64_t elapsed) {
    VxTraceClock none = {0};
    if (after < before || total > after - before || offset > total ||
        elapsed > total - offset) return none;
    return (VxTraceClock){before + offset, after - (total - offset), VX_TRACE_CLOCK_METHOD_BOUNDED};
}
VxTraceClock vx_trace_clock_calibrated(uint64_t start, uint64_t anchor,
    unsigned bits, double period, uint64_t host, uint64_t uncertainty,
    uint64_t before, uint64_t after, uint64_t elapsed) {
    VxTraceClock none = {0};
    if (!bits || bits > 64 || !isfinite(period) || period <= 0 || after < before ||
        elapsed > after - before || after - before > UINT64_C(1000000000)) return none;
    uint64_t mask = bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
    uint64_t half = UINT64_C(1) << (bits - 1);
    if ((long double)(after - before) >= (long double)half * period) return none;
    uint64_t delta = (start - anchor) & mask;
    long double ticks = delta < half ? (long double)delta : -(long double)((anchor - start) & mask);
    long double center = (long double)host + ticks * period;
    /* Host samples are quantized to microseconds. Device ticks and integer
     * conversion also contribute uncertainty; never imply nanosecond accuracy. */
    long double error = (long double)uncertainty + 1000.0L + period;
    long double low = center - error, high = center + error;
    if (high < before || low > after - elapsed || low < 0 || high >= (long double)UINT64_MAX) return none;
    if (low < before) low = before;
    if (high > after - elapsed) high = after - elapsed;
    uint64_t upper = (uint64_t)high;
    if ((long double)upper < high) upper++;
    return (VxTraceClock){(uint64_t)low, upper, VX_TRACE_CLOCK_METHOD_CALIBRATED};
}

/* Same monotonic clock as scheduler deadlines, at its actual microsecond
 * resolution. No wall-clock conversion or epoch subtraction on hot paths. */
uint64_t vx_trace_now_ns(void) { return vx_runtime_monotonic_time_micros() * UINT64_C(1000); }

static uint64_t vx_trace_clock_resolution(void) {
#if defined(__wasm__)
    /* Browser precision can be reduced dynamically by the host. */
    return 0;
#else
    struct timespec resolution;
    if (clock_getres(CLOCK_MONOTONIC, &resolution) != 0 || resolution.tv_sec < 0 ||
        (uint64_t)resolution.tv_sec > UINT64_MAX / UINT64_C(1000000000)) return 0;
    uint64_t ns = (uint64_t)resolution.tv_sec * UINT64_C(1000000000) + (uint64_t)resolution.tv_nsec;
    return ns < 1000 ? 1000 : ns;
#endif
}

static void vx_trace_sample_memory(VxTrace* trace, int index) {
    trace->samples[index] = (VxProcessMemorySampleV1)VX_PROCESS_MEMORY_SAMPLE_V1_INIT;
    trace->memory_start_ns[index] = vx_trace_now_ns();
    (void)vx_process_memory_sample_v1(&trace->samples[index]);
    trace->memory_end_ns[index] = vx_trace_now_ns();
}
VxTrace* vx_trace_create(size_t capacity, VxTraceDetail detail, int device_timing, int memory) {
    if (capacity < 4096 || capacity > 64u * 1024u * 1024u ||
        (detail != VX_TRACE_DETAIL_BASIC && detail != VX_TRACE_DETAIL_NODES)) return NULL;
    VxTrace* trace = calloc(1, sizeof(*trace));
    if (!trace) return NULL;
    if (capacity <= sizeof(*trace) + sizeof(VxTraceMemoryOwner) + sizeof(VxTraceRecord)) {
        free(trace); return NULL;
    }
    size_t available = capacity - sizeof(*trace) - sizeof(VxTraceMemoryOwner);
    if (memory) {
        size_t limit = available / 4 / sizeof(VxTraceAllocation);
        trace->allocation_capacity = 1;
        while (trace->allocation_capacity <= limit / 2) trace->allocation_capacity *= 2;
        trace->allocations = calloc(trace->allocation_capacity, sizeof(*trace->allocations));
        available -= trace->allocation_capacity * sizeof(*trace->allocations);
    }
    trace->capacity = available / sizeof(VxTraceRecord);
    trace->records = calloc(trace->capacity, sizeof(*trace->records));
    trace->owner = calloc(1, sizeof(*trace->owner));
    if (!trace->records || !trace->owner || (memory && !trace->allocations) ||
        pthread_mutex_init(&trace->owner->mutex, NULL) != 0) {
        free(trace->owner); free(trace->allocations); free(trace->records); free(trace); return NULL;
    }
    atomic_init(&trace->owner->references, 1);
    atomic_init(&trace->owner->enabled, !!memory);
    trace->owner->trace = trace;
    for (int i = 0; i < VX_MEMORY_ALLOCATOR_COUNT; i++) trace->allocators[i].complete = 1;
    atomic_init(&trace->references, 1);
    trace->capacity_bytes = capacity;
    trace->detail = detail; trace->device_timing = !!device_timing;
    trace->memory = memory;
    trace->origin_ns = vx_trace_now_ns();
    trace->clock_resolution_ns = vx_trace_clock_resolution();
    if (memory) vx_trace_sample_memory(trace, 0);
    return trace;
}
void vx_trace_retain(VxTrace* trace) {
    if (trace) atomic_fetch_add_explicit(&trace->references, 1, memory_order_relaxed);
}
static void vx_memory_owner_release(VxTraceMemoryOwner* owner) {
    if (owner && atomic_fetch_sub_explicit(&owner->references, 1, memory_order_acq_rel) == 1) {
        pthread_mutex_destroy(&owner->mutex); free(owner);
    }
}
void vx_trace_release(VxTrace* trace) {
    if (!trace || atomic_fetch_sub_explicit(&trace->references, 1, memory_order_acq_rel) != 1) return;
    pthread_mutex_lock(&trace->owner->mutex);
    atomic_store_explicit(&trace->owner->enabled, 0, memory_order_release);
    if (trace->memory_bridge_ticket) {
        trace->memory_bridge_stop(trace->memory_bridge_ticket);
        trace->memory_bridge_ticket = 0;
    }
    for (size_t i = 0; i < trace->count; i++)
        if (trace->records[i].poll) trace->records[i].release(trace->records[i].ticket);
    trace->owner->trace = NULL;
    pthread_mutex_unlock(&trace->owner->mutex);
    vx_memory_owner_release(trace->owner);
    free(trace->allocations); free(trace->records); free(trace);
}
/* Called with collector mutex held; acquires no engine/device lock. */
static void vx_trace_finish_memory(VxTrace* trace) {
    if (trace->stopped && !trace->active && trace->memory && !trace->sampled_stop) {
        vx_trace_sample_memory(trace, 1);
        trace->sampled_stop = 1;
    }
}
void vx_trace_stop(VxTrace* trace) {
    if (!trace) return;
    pthread_mutex_lock(&trace->owner->mutex);
    atomic_store_explicit(&trace->owner->enabled, 0, memory_order_release);
    if (trace->memory_bridge_ticket) {
        trace->memory_bridge_stop(trace->memory_bridge_ticket);
        trace->memory_bridge_ticket = 0;
    }
    trace->stopped = 1;
    vx_trace_finish_memory(trace);
    pthread_mutex_unlock(&trace->owner->mutex);
}
/* Public release no longer needs device observations. Retire their tickets
 * immediately; already accepted host scopes retain the collector until exit. */
void vx_trace_abandon(VxTrace* trace) {
    if (!trace) return;
    pthread_mutex_lock(&trace->owner->mutex);
    atomic_store_explicit(&trace->owner->enabled, 0, memory_order_release);
    if (trace->memory_bridge_ticket) {
        trace->memory_bridge_stop(trace->memory_bridge_ticket);
        trace->memory_bridge_ticket = 0;
    }
    trace->stopped = 1; trace->abandoned = 1;
    for (size_t i = 0; trace->pending && i < trace->count; i++) {
        VxTraceRecord* e = &trace->records[i];
        if (!e->poll) continue;
        e->release(e->ticket);
        e->poll = NULL; e->release = NULL; e->ticket = 0; e->invalid = 1;
        trace->pending--; trace->active--; trace->dropped++;
        if (e->kind == VX_TRACE_EVENT_DEVICE) trace->pending_device--;
    }
    vx_trace_finish_memory(trace);
    pthread_mutex_unlock(&trace->owner->mutex);
}
/* Only asynchronous device records have callbacks. Completed data is compacted
 * before READY and is immutable afterwards. Pending slots count against the
 * same fixed record capacity, so an unpolled application cannot grow storage. */
static void vx_trace_clock_relative(VxTrace* trace, VxTraceClock* clock) {
    if (!clock->method) return;
    if (clock->latest_ns < clock->earliest_ns || clock->latest_ns < trace->origin_ns) {
        *clock = (VxTraceClock){0}; return;
    }
    clock->earliest_ns = clock->earliest_ns > trace->origin_ns ? clock->earliest_ns - trace->origin_ns : 0;
    clock->latest_ns -= trace->origin_ns;
}
static void vx_trace_poll_locked(VxTrace* trace) {
    size_t next_pending = trace->count;
    if (trace->pending) for (size_t i = trace->first_pending; i < trace->count; i++) {
        VxTraceRecord* e = &trace->records[i];
        if (!e->poll) continue;
        VxDeviceTraceResult result = {0};
        int status = e->poll(e->ticket, &result);
        if (status == 1) {
            if (next_pending == trace->count) next_pending = i;
            continue;
        }
        e->release(e->ticket);
        e->poll = NULL; e->release = NULL; e->ticket = 0;
        trace->pending--; trace->active--;
        if (e->kind == VX_TRACE_EVENT_DEVICE) trace->pending_device--;
        if (status == 0) {
            e->duration_ns = result.elapsed_ns;
            e->clock = result.clock;
            vx_trace_clock_relative(trace, &e->clock);
            if (e->kind == VX_TRACE_EVENT_DEVICE) vx_trace_device_count(trace, e, 1);
            else {
                e->start_ns = result.host_start_ns >= trace->origin_ns ? result.host_start_ns - trace->origin_ns : 0;
                VxTraceDeviceCoverage* c = vx_trace_coverage(trace, e->backend);
                if (c) c->host_awaits++;
            }
        } else {
            e->invalid = 1; trace->dropped++;
            if (e->kind == VX_TRACE_EVENT_DEVICE) vx_trace_device_count(trace, e, 0);
        }
    }
    trace->first_pending = next_pending;
    if (trace->stopped && !trace->active && !trace->finalized) {
        trace->finalized = 1;
        size_t count = 0;
        for (size_t i = 0; i < trace->count; i++) {
            if (trace->records[i].invalid) continue;
            trace->records[count] = trace->records[i];
            trace->records[count].sequence = count + 1;
            count++;
        }
        trace->count = count;
        vx_trace_finish_memory(trace);
    }
}
void vx_trace_view(VxTrace* trace, VxTraceView* view) {
    memset(view, 0, sizeof(*view));
    pthread_mutex_lock(&trace->owner->mutex);
    vx_trace_poll_locked(trace);
    view->state = !trace->stopped ? VX_TRACE_STATE_COLLECTING :
        trace->active ? VX_TRACE_STATE_DRAINING : VX_TRACE_STATE_READY;
    view->detail = trace->detail; view->memory = trace->memory;
    view->device_timing = trace->device_timing;
    view->pending = trace->pending_device; view->device_count = trace->device_count;
    memcpy(view->devices, trace->devices, sizeof(view->devices));
    view->clock_resolution_ns = trace->clock_resolution_ns;
    view->capacity_bytes = trace->capacity_bytes;
    view->count = trace->count; view->dropped = trace->dropped; view->active = trace->active;
    if (view->state == VX_TRACE_STATE_READY) view->records = trace->records;
    memcpy(view->allocators, trace->allocators, sizeof(view->allocators));
    memcpy(view->samples, trace->samples, sizeof(view->samples));
    memcpy(view->memory_start_ns, trace->memory_start_ns, sizeof(view->memory_start_ns));
    memcpy(view->memory_end_ns, trace->memory_end_ns, sizeof(view->memory_end_ns));
    for (int i = 0; i < 2; i++) {
        view->memory_start_ns[i] = view->memory_start_ns[i] >= trace->origin_ns ? view->memory_start_ns[i] - trace->origin_ns : 0;
        view->memory_end_ns[i] = view->memory_end_ns[i] >= trace->origin_ns ? view->memory_end_ns[i] - trace->origin_ns : 0;
    }
    pthread_mutex_unlock(&trace->owner->mutex);
}
int vx_trace_scope_begin(VxTrace* trace, VxTraceScope* scope,
    const VxTraceIdentity* identity, const char* name, const char* backend) {
    pthread_mutex_lock(&trace->owner->mutex);
    if (trace->stopped) { pthread_mutex_unlock(&trace->owner->mutex); return 0; }
    /* Retire completed observations at the next captured operation without
     * making applications poll during collection. Polling never waits. */
    if (trace->pending) vx_trace_poll_locked(trace);
    trace->active++;
    vx_trace_retain(trace);
    pthread_mutex_unlock(&trace->owner->mutex);
    if (!vx_trace_thread_track) vx_trace_thread_track = atomic_fetch_add_explicit(&vx_trace_next_track, 1, memory_order_relaxed);
    *scope = (VxTraceScope){trace, *identity, vx_trace_now_ns(), vx_trace_thread_track, name, backend, {.index = -1}};
    return 1;
}
static void vx_trace_copy_text(char* to, size_t capacity, const char* from, int* truncated) {
    if (!from) from = "";
    size_t n = strlen(from);
    if (n >= capacity) {
        n = capacity - 1;
        /* Do not split a UTF-8 codepoint. */
        while (n && (((unsigned char)from[n] & 0xc0u) == 0x80u)) n--;
        *truncated = 1;
    }
    memcpy(to, from, n); to[n] = 0;
}
static VxTraceRecord* vx_trace_record_locked(VxTraceScope* scope, uint64_t start, uint64_t end, int index, VxTraceEventKind kind,
    const char* name, const char* output, int fused) {
    VxTrace* trace = scope->trace;
    VxTraceRecord* event = NULL;
    if (trace->abandoned) return NULL;
    if (trace->count == trace->capacity) trace->dropped++;
    else {
        event = &trace->records[trace->count++];
        event->sequence = trace->count;
        event->start_ns = start >= trace->origin_ns ? start - trace->origin_ns : 0;
        event->duration_ns = end >= start ? end - start : 0;
        event->track_id = scope->track_id; event->identity = scope->identity;
        event->kind = kind; event->schedule_index = index;
        event->queue = scope->queue;
        event->fused = fused;
        vx_trace_copy_text(event->name, sizeof(event->name), name, &event->truncated);
        vx_trace_copy_text(event->backend, sizeof(event->backend), scope->backend, &event->truncated);
        vx_trace_copy_text(event->output, sizeof(event->output), output, &event->truncated);
    }
    return event;
}
static void vx_trace_record(VxTraceScope* scope, uint64_t start, uint64_t end, int index,
    VxTraceEventKind kind, const char* name, const char* output, int fused) {
    pthread_mutex_lock(&scope->trace->owner->mutex);
    VxTraceRecord* e = vx_trace_record_locked(scope, start, end, index, kind, name, output, fused);
    if (e && kind == VX_TRACE_EVENT_DEVICE) vx_trace_device_count(scope->trace, e, 1);
    pthread_mutex_unlock(&scope->trace->owner->mutex);
}
void vx_trace_scope_end(VxTraceScope* scope) {
    if (!scope->trace) return;
    VxTrace* trace = scope->trace;
    vx_trace_record(scope, scope->start_ns, vx_trace_now_ns(), -1, VX_TRACE_EVENT_HOST_OPERATION, scope->name, "", 0);
    pthread_mutex_lock(&trace->owner->mutex);
    trace->active--;
    vx_trace_finish_memory(trace);
    pthread_mutex_unlock(&trace->owner->mutex);
    scope->trace = NULL;
    vx_trace_release(trace);
}
int vx_trace_device_enabled(const VxTraceScope* scope) {
    return scope && scope->trace && scope->trace->device_timing;
}
int vx_trace_device_nodes(const VxTraceScope* scope) {
    return vx_trace_device_enabled(scope) && scope->trace->detail == VX_TRACE_DETAIL_NODES;
}
int vx_trace_nodes(const VxTraceScope* scope) {
    return scope && scope->trace && scope->trace->detail == VX_TRACE_DETAIL_NODES;
}
void vx_trace_device_status(VxTraceScope* scope, int available, int nodes, int splits, int barriers) {
    if (!vx_trace_device_enabled(scope)) return;
    VxTrace* trace = scope->trace;
    pthread_mutex_lock(&trace->owner->mutex);
    VxTraceDeviceCoverage* coverage = vx_trace_coverage(trace, scope->backend);
    if (coverage) {
        VxTraceSupport support = available ? VX_TRACE_SUPPORT_AVAILABLE : VX_TRACE_SUPPORT_UNAVAILABLE;
        coverage->support = coverage->support == VX_TRACE_SUPPORT_UNOBSERVED ? support :
            coverage->support == support ? support : VX_TRACE_SUPPORT_MIXED;
        coverage->node_timing_available |= available && nodes;
        if (!available) coverage->unavailable_passes++;
        coverage->splits_passes |= splits; coverage->adds_barriers |= barriers;
    }
    pthread_mutex_unlock(&trace->owner->mutex);
}
void vx_trace_program_status(VxTraceScope* scope, int available) {
    if (!vx_trace_device_enabled(scope) || !available) return;
    pthread_mutex_lock(&scope->trace->owner->mutex);
    VxTraceDeviceCoverage* coverage = vx_trace_coverage(scope->trace, scope->backend);
    if (coverage) coverage->program_timing_available = 1;
    pthread_mutex_unlock(&scope->trace->owner->mutex);
}
void vx_trace_device_fail(VxTraceScope* scope) {
    if (!scope || !scope->trace) return;
    pthread_mutex_lock(&scope->trace->owner->mutex);
    scope->trace->dropped++;
    VxTraceDeviceCoverage* coverage = vx_trace_coverage(scope->trace, scope->backend);
    if (coverage) coverage->failed_intervals++;
    pthread_mutex_unlock(&scope->trace->owner->mutex);
}
static void vx_trace_span_metadata(VxTraceRecord* event, const VxDeviceTraceSpan* span) {
    if (!event) return;
    event->phase = span->phase;
    if (span->queue.queue_id) event->queue = span->queue;
    event->activity = span->activity;
    event->copy_source = span->copy_source;
    event->copy_destination = span->copy_destination;
    event->copy_bytes = span->copy_bytes;
    event->clock = span->clock;
    vx_trace_copy_text(event->program, sizeof(event->program), span->program, &event->truncated);
    vx_trace_copy_text(event->entry, sizeof(event->entry), span->entry, &event->truncated);
    vx_trace_copy_text(event->tensor, sizeof(event->tensor), span->tensor, &event->truncated);
}
void vx_trace_host_activity(VxTraceScope* scope, uint64_t start, uint64_t end,
    const VxDeviceTraceSpan* span) {
    if (!vx_trace_nodes(scope)) return;
    pthread_mutex_lock(&scope->trace->owner->mutex);
    VxTraceRecord* e = vx_trace_record_locked(scope, start, end, span->index,
        VX_TRACE_EVENT_HOST_WORK, span->name, span->output, span->fused);
    vx_trace_span_metadata(e, span);
    if (e) {
        VxTraceDeviceCoverage* c = vx_trace_coverage(scope->trace, scope->backend);
        if (c) {
            if (span->activity == VX_TRACE_ACTIVITY_COPY) c->host_copy_calls++;
            if (span->activity == VX_TRACE_ACTIVITY_WAIT) c->host_wait_calls++;
            if (span->activity == VX_TRACE_ACTIVITY_SUBMIT) c->host_submit_calls++;
            if (span->activity == VX_TRACE_ACTIVITY_AWAIT) c->host_awaits++;
        }
    }
    pthread_mutex_unlock(&scope->trace->owner->mutex);
}
VxDeviceTraceSpan vx_trace_program_span(VxTraceScope* scope, const char* program, const char* entry) {
    const VxTraceWork* work = &scope->work;
    return (VxDeviceTraceSpan){vx_trace_now_ns(), program, work->output,
        work->index, work->fused, work->phase, program, entry, work->tensor};
}
void vx_trace_host_work(VxTraceScope* scope, uint64_t start,
    const VxTraceWork* work, const char* backend) {
    if (!vx_trace_nodes(scope)) return;
    VxTraceScope local = *scope;
    if (backend) local.backend = backend;
    pthread_mutex_lock(&scope->trace->owner->mutex);
    VxTraceRecord* event = vx_trace_record_locked(&local, start, vx_trace_now_ns(), work->index,
        work->index >= 0 ? VX_TRACE_EVENT_HOST_NODE : VX_TRACE_EVENT_HOST_WORK,
        work->name, work->output, work->fused);
    if (event) {
        event->phase = work->phase;
        vx_trace_copy_text(event->tensor, sizeof(event->tensor), work->tensor, &event->truncated);
    }
    pthread_mutex_unlock(&scope->trace->owner->mutex);
}
void vx_trace_node(VxTraceScope* scope, uint64_t start, int index,
    const char* op, const char* output, int fused) {
    VxTraceWork work = {VX_TRACE_PHASE_FORWARD, index, fused, op, output, NULL};
    vx_trace_host_work(scope, start, &work, NULL);
}
void vx_trace_host_program(VxTraceScope* scope, uint64_t start,
    const char* program, const char* entry) {
    VxDeviceTraceSpan span = vx_trace_program_span(scope, program, entry);
    pthread_mutex_lock(&scope->trace->owner->mutex);
    VxTraceRecord* event = vx_trace_record_locked(scope, start, vx_trace_now_ns(),
        span.index, VX_TRACE_EVENT_HOST_WORK, program, span.output, span.fused);
    vx_trace_span_metadata(event, &span);
    pthread_mutex_unlock(&scope->trace->owner->mutex);
}
void vx_trace_device_span(VxTraceScope* scope, const VxDeviceTraceSpan* span, uint64_t duration) {
    if (!scope || !scope->trace || duration > UINT64_MAX - span->host_start_ns) return;
    pthread_mutex_lock(&scope->trace->owner->mutex);
    VxTraceRecord* e = vx_trace_record_locked(scope, span->host_start_ns, span->host_start_ns + duration,
        span->index, VX_TRACE_EVENT_DEVICE, span->name, span->output, span->fused);
    vx_trace_span_metadata(e, span);
    if (e) { vx_trace_clock_relative(scope->trace, &e->clock); vx_trace_device_count(scope->trace, e, 1); }
    pthread_mutex_unlock(&scope->trace->owner->mutex);
}
void vx_trace_defer_device_span(VxTraceScope* scope, const VxDeviceTraceSpan* span,
    uint32_t ticket, int (*poll)(uint32_t, VxDeviceTraceResult*), void (*release)(uint32_t)) {
    if (!scope || !scope->trace) { release(ticket); return; }
    VxTrace* trace = scope->trace;
    pthread_mutex_lock(&trace->owner->mutex);
    VxTraceRecord* e = vx_trace_record_locked(scope, span->host_start_ns, span->host_start_ns,
        span->index, VX_TRACE_EVENT_DEVICE, span->name, span->output, span->fused);
    vx_trace_span_metadata(e, span);
    if (e) {
        e->ticket = ticket; e->poll = poll; e->release = release;
        if (!trace->pending) trace->first_pending = trace->count - 1;
        trace->active++; trace->pending++; trace->pending_device++;
    }
    pthread_mutex_unlock(&trace->owner->mutex);
    if (!e) release(ticket);
}
void vx_trace_defer_host_activity(VxTraceScope* scope, const VxDeviceTraceSpan* span,
    uint32_t ticket, int (*poll)(uint32_t, VxDeviceTraceResult*), void (*release)(uint32_t)) {
    if (!vx_trace_nodes(scope)) { release(ticket); return; }
    VxTrace* trace = scope->trace;
    pthread_mutex_lock(&trace->owner->mutex);
    VxTraceRecord* e = vx_trace_record_locked(scope, span->host_start_ns, span->host_start_ns,
        span->index, VX_TRACE_EVENT_HOST_WORK, span->name, span->output, span->fused);
    vx_trace_span_metadata(e, span);
    if (e) {
        e->ticket = ticket; e->poll = poll; e->release = release;
        if (!trace->pending) trace->first_pending = trace->count - 1;
        trace->active++; trace->pending++;
    }
    pthread_mutex_unlock(&trace->owner->mutex);
    if (!e) release(ticket);
}

void vx_trace_drop(VxTraceScope* scope) {
    if (!scope || !scope->trace) return;
    pthread_mutex_lock(&scope->trace->owner->mutex);
    scope->trace->dropped++;
    pthread_mutex_unlock(&scope->trace->owner->mutex);
}

void vx_trace_device(VxTraceScope* scope, uint64_t host_start, uint64_t duration, const char* name) {
    if (!scope || !scope->trace || duration > UINT64_MAX - host_start) return;
    vx_trace_record(scope, host_start, host_start + duration, -1, VX_TRACE_EVENT_DEVICE, name, "", 0);
}

void vx_trace_device_node(VxTraceScope* scope, uint64_t host_start, uint64_t duration,
    int index, const char* name, const char* output, int fused) {
    VxDeviceTraceSpan span = {host_start, name, output, index, fused,
        index >= 0 ? VX_TRACE_PHASE_FORWARD : VX_TRACE_PHASE_UNSPECIFIED};
    vx_trace_device_span(scope, &span, duration);
}
int vx_trace_timestamp_duration(uint64_t start, uint64_t end, unsigned bits,
    double period, uint64_t host_elapsed, uint64_t* duration) {
    if (!bits || bits > 64 || !isfinite(period) || period <= 0) return 0;
    uint64_t mask = bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
    /* A short counter can wrap more than once. The host envelope must rule
     * that out before modular subtraction is a meaningful elapsed time. */
    if (bits < 64 && (double)host_elapsed >= (double)(UINT64_C(1) << bits) * period) return 0;
    double ns = (double)((end - start) & mask) * period;
    if (!isfinite(ns) || ns >= (double)UINT64_MAX) return 0;
    *duration = (uint64_t)ns;
    return 1;
}
size_t vx_trace_device_capacity(VxTraceScope* scope, size_t requested) {
    if (!scope || !scope->trace) return 0;
    VxTrace* trace = scope->trace;
    pthread_mutex_lock(&trace->owner->mutex);
    size_t remaining = trace->capacity - trace->count;
    /* One native pass never needs an unbounded array of driver objects. */
    if (requested > 1024) requested = 1024;
    if (requested > remaining) requested = remaining;
    if (trace->abandoned) requested = 0;
    pthread_mutex_unlock(&trace->owner->mutex);
    return requested;
}
void vx_trace_defer_device_node(VxTraceScope* scope, uint64_t host_start,
    int index, const char* name, const char* output, int fused,
    uint32_t ticket, int (*poll)(uint32_t, VxDeviceTraceResult*), void (*release)(uint32_t)) {
    VxDeviceTraceSpan span = {host_start, name, output, index, fused,
        index >= 0 ? VX_TRACE_PHASE_FORWARD : VX_TRACE_PHASE_UNSPECIFIED};
    vx_trace_defer_device_span(scope, &span, ticket, poll, release);
}
void vx_trace_defer_device(VxTraceScope* scope, uint64_t host_start, const char* name,
    uint32_t ticket, int (*poll)(uint32_t, VxDeviceTraceResult*), void (*release)(uint32_t)) {
    vx_trace_defer_device_node(scope, host_start, -1, name, "", 0, ticket, poll, release);
}

#include "profiling_memory.inc"

/* JSON serialization runs only after collection drains, one bounded page at
 * a time. Each response owns its bytes and leaves the trace immutable. */
typedef struct { char* data; size_t size, capacity; int failed; } vx_trace_Json;
static int vx_trace_reserve(vx_trace_Json* out, size_t length) {
    if (out->failed) return 0;
    if (length > SIZE_MAX - out->size - 1) { out->failed = 1; return 0; }
    size_t needed = out->size + length + 1;
    if (needed > out->capacity) {
        size_t next = out->capacity <= SIZE_MAX / 2 ? out->capacity * 2 : needed;
        if (next < needed) next = needed;
        char* data = realloc(out->data, next);
        if (!data) { out->failed = 1; return 0; }
        out->data = data; out->capacity = next;
    }
    return 1;
}
static void vx_trace_bytes(vx_trace_Json* out, const char* data, size_t length) {
    if (!vx_trace_reserve(out, length)) return;
    memcpy(out->data + out->size, data, length);
    out->size += length;
    out->data[out->size] = 0;
}
static void vx_trace_append(vx_trace_Json* out, const char* format, ...) {
    if (out->failed) return;
    va_list args, copy;
    va_start(args, format); va_copy(copy, args);
    int length = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if (length < 0) out->failed = 1;
    else if (vx_trace_reserve(out, (size_t)length)) {
        vsnprintf(out->data + out->size, out->capacity - out->size, format, args);
        out->size += (size_t)length;
    }
    va_end(args);
}
static void vx_trace_quoted(vx_trace_Json* out, const char* text) {
    vx_trace_bytes(out, "\"", 1);
    const unsigned char* p = (const unsigned char*)text;
    while (*p) {
        const unsigned char* start = p;
        while (*p >= 32 && *p != '"' && *p != '\\') p++;
        vx_trace_bytes(out, (const char*)start, (size_t)(p - start));
        if (!*p) break;
        if (*p == '"' || *p == '\\') vx_trace_append(out, "\\%c", *p);
        else vx_trace_append(out, "\\u%04x", *p);
        p++;
    }
    vx_trace_bytes(out, "\"", 1);
}
static void vx_trace_time_us(vx_trace_Json* out, uint64_t ns) {
    vx_trace_append(out, "%" PRIu64 ".%03u", ns / 1000, (unsigned)(ns % 1000));
}
static int vx_trace_compare_track(const void* a, const void* b) {
    uint64_t left = *(const uint64_t*)a, right = *(const uint64_t*)b;
    return (left > right) - (left < right);
}
static int vx_trace_compare_queue(const void* a, const void* b) {
    const VxTraceRecord* left = *(const VxTraceRecord* const*)a;
    const VxTraceRecord* right = *(const VxTraceRecord* const*)b;
    return (left->queue.queue_id > right->queue.queue_id) - (left->queue.queue_id < right->queue.queue_id);
}
static const char* vx_trace_support_label(VxTraceSupport support) {
    switch (support) {
        case VX_TRACE_SUPPORT_AVAILABLE: return "available";
        case VX_TRACE_SUPPORT_UNAVAILABLE: return "unavailable";
        case VX_TRACE_SUPPORT_MIXED: return "mixed";
        default: return "unobserved";
    }
}
static const char* vx_trace_phase_label(VxTracePhase phase) {
    switch (phase) {
        case VX_TRACE_PHASE_FORWARD: return "forward";
        case VX_TRACE_PHASE_LOSS: return "loss";
        case VX_TRACE_PHASE_BACKWARD: return "backward";
        case VX_TRACE_PHASE_GRADIENT: return "gradient";
        case VX_TRACE_PHASE_OPTIMIZER: return "optimizer";
        default: return "unspecified";
    }
}
static const char* vx_trace_activity_label(VxTraceActivity activity) {
    switch (activity) {
        case VX_TRACE_ACTIVITY_COPY: return "copy";
        case VX_TRACE_ACTIVITY_SUBMIT: return "submit";
        case VX_TRACE_ACTIVITY_WAIT: return "wait";
        case VX_TRACE_ACTIVITY_AWAIT: return "await";
        default: return "work";
    }
}
/* READY views borrow immutable records under their caller's trace lease.
 * No full-trace cache or scan: scratch storage and serialization are page-local. */
int vx_trace_export(const VxTraceView* trace, size_t offset, size_t limit,
    char** json, size_t* size) {
    *json = NULL; *size = 0;
    if (trace->state != VX_TRACE_STATE_READY) return 0;
    if (offset > trace->count || !limit || limit > 1024) return -2;
    if (offset && offset == trace->count) return 1;
    size_t count = trace->count - offset;
    if (count > limit) count = limit;
    size_t end = offset + count;
    vx_trace_Json out = {0};
    if (!offset) {
        vx_trace_append(&out, "{\"displayTimeUnit\":\"ns\",\"otherData\":{\"format\":\"volvoxai-trace/v6\",\"hostClockResolutionNs\":%" PRIu64 ",\"droppedEvents\":\"%" PRIu64 "\",\"detail\":\"%s\",\"deviceTiming\":%s,\"memory\":%s,\"devices\":[",
            trace->clock_resolution_ns, trace->dropped, trace->detail == VX_TRACE_DETAIL_NODES ? "nodes" : "basic", trace->device_timing ? "true" : "false", trace->memory ? "true" : "false");
        for (size_t i = 0; i < trace->device_count; i++) {
            const VxTraceDeviceCoverage* c = &trace->devices[i];
            vx_trace_append(&out, "%s{\"backend\":", i ? "," : ""); vx_trace_quoted(&out, c->backend);
            vx_trace_append(&out, ",\"support\":\"%s\",\"nodeTimingAvailable\":%s,\"passIntervals\":\"%" PRIu64 "\",\"nodeIntervals\":\"%" PRIu64 "\",\"failedIntervals\":\"%" PRIu64 "\",\"unavailablePasses\":\"%" PRIu64 "\",\"splitsPasses\":%s,\"addsBarriers\":%s",
                vx_trace_support_label(c->support), c->node_timing_available ? "true" : "false", c->pass_intervals, c->node_intervals, c->failed_intervals, c->unavailable_passes,
                c->splits_passes ? "true" : "false", c->adds_barriers ? "true" : "false");
            vx_trace_append(&out, ",\"programTimingAvailable\":%s,\"programIntervals\":\"%" PRIu64 "\"",
                c->program_timing_available ? "true" : "false", c->program_intervals);
            vx_trace_append(&out, ",\"calibratedIntervals\":\"%" PRIu64 "\",\"boundedIntervals\":\"%" PRIu64 "\",\"copyIntervals\":\"%" PRIu64 "\",\"hostCopyCalls\":\"%" PRIu64 "\",\"hostWaitCalls\":\"%" PRIu64 "\",\"hostSubmitCalls\":\"%" PRIu64 "\",\"hostAwaits\":\"%" PRIu64 "\"}",
                c->calibrated_intervals, c->bounded_intervals, c->copy_intervals,
                c->host_copy_calls, c->host_wait_calls, c->host_submit_calls, c->host_awaits);
        }
        vx_trace_append(&out, "],\"allocators\":[");
        int memory_written = 0;
        for (int i = 0; i < VX_MEMORY_ALLOCATOR_COUNT; i++) {
            const VxAllocatorMemory* m = &trace->allocators[i];
            if (!m->observed) continue;
            vx_trace_append(&out, "%s{\"allocator\":", memory_written++ ? "," : "");
            vx_trace_quoted(&out, vx_memory_allocator_name(i));
            vx_trace_append(&out, ",\"scope\":\"%s\"",
                i == VX_MEMORY_WEBGPU_BUFFER ? "backend_shared" : "runtime");
            vx_trace_append(&out, ",\"inventory\":\"partial\",\"valueRelation\":\"requested\",\"observationStartNs\":\"%" PRIu64 "\",\"existingBytes\":\"%" PRIu64 "\",\"liveBytes\":\"%" PRIu64 "\",\"peakBytes\":\"%" PRIu64 "\",\"allocatedBytes\":\"%" PRIu64 "\",\"freedBytes\":\"%" PRIu64 "\",\"droppedEvents\":\"%" PRIu64 "\",\"accountingComplete\":%s}",
                m->start_ns, m->existing, m->live, m->peak, m->allocated, m->freed,
                m->dropped, m->complete ? "true" : "false");
        }
        vx_trace_append(&out, "]},\"traceEvents\":[");
        vx_trace_append(&out, "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"args\":{\"name\":\"VolvoxAI\"}}");
        vx_trace_append(&out, ",{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":2,\"args\":{\"name\":\"GPU queues\"}}");
        vx_trace_append(&out, ",{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":3,\"args\":{\"name\":\"Asynchronous completion\"}}");
    }
    uint64_t* tracks = count ? malloc(count * sizeof(*tracks)) : NULL;
    if (count && !tracks) out.failed = 1;
    if (tracks) {
        for (size_t i = 0; i < count; i++) tracks[i] = trace->records[offset + i].track_id;
        qsort(tracks, count, sizeof(*tracks), vx_trace_compare_track);
        for (size_t i = 0; i < count; i++) {
            if (i && tracks[i] == tracks[i - 1]) continue;
            vx_trace_append(&out, ",{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":%" PRIu64 ",\"args\":{\"name\":\"Host thread %" PRIu64 "\"}}", tracks[i], tracks[i]);
        }
        free(tracks);
    }
    const VxTraceRecord** queues = count ? malloc(count * sizeof(*queues)) : NULL;
    if (count && !queues) out.failed = 1;
    if (queues) {
        size_t count = 0;
        for (size_t i = offset; i < end; i++)
            if (trace->records[i].kind == VX_TRACE_EVENT_DEVICE && trace->records[i].queue.queue_id)
                queues[count++] = &trace->records[i];
        qsort(queues, count, sizeof(*queues), vx_trace_compare_queue);
        for (size_t i = 0; i < count; i++) {
            const VxTraceRecord* e = queues[i];
            if (i && e->queue.queue_id == queues[i - 1]->queue.queue_id) continue;
            vx_trace_append(&out, ",{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":2,\"tid\":%" PRIu64 ",\"args\":{\"name\":", e->queue.queue_id);
            char label[128]; snprintf(label, sizeof(label), "%s device %" PRIu64 " queue %" PRIu64, e->backend, e->queue.device_id, e->queue.queue_id);
            vx_trace_quoted(&out, label); vx_trace_append(&out, "}}");
        }
        free(queues);
    }
    for (size_t i = offset; i < end; i++) {
        const VxTraceRecord* e = &trace->records[i];
        int device = e->kind == VX_TRACE_EVENT_DEVICE;
        int aligned = device && e->clock.method == VX_TRACE_CLOCK_METHOD_CALIBRATED;
        int instant = (device && !aligned) || e->kind == VX_TRACE_EVENT_MEMORY;
        int asynchronous = e->activity == VX_TRACE_ACTIVITY_AWAIT;
        uint64_t timestamp = aligned ? e->clock.earliest_ns + (e->clock.latest_ns - e->clock.earliest_ns) / 2 : e->start_ns;
        uint64_t track = device && e->queue.queue_id ? e->queue.queue_id : e->track_id;
        vx_trace_append(&out, ",");
        vx_trace_append(&out, "{\"name\":"); vx_trace_quoted(&out, e->name);
        vx_trace_append(&out, ",\"cat\":\"%s\",\"ph\":\"%s\",\"pid\":%d,\"tid\":%" PRIu64 ",\"ts\":",
            e->kind == VX_TRACE_EVENT_MEMORY ? "memory.allocation" :
            e->activity == VX_TRACE_ACTIVITY_COPY ? (device ? "device.copy" : "host.copy") :
            e->activity == VX_TRACE_ACTIVITY_WAIT ? "host.wait" :
            e->activity == VX_TRACE_ACTIVITY_AWAIT ? "host.await" :
            e->activity == VX_TRACE_ACTIVITY_SUBMIT ? "host.submit" :
            e->kind == VX_TRACE_EVENT_DEVICE ? (e->program[0] ? "device.program" : "device.interval") :
            e->kind == VX_TRACE_EVENT_HOST_WORK ? (e->program[0] ? "host.program" : "host.work") :
                e->kind == VX_TRACE_EVENT_HOST_NODE ? "host.node" : "host.operation",
            instant ? "i" : asynchronous ? "b" : "X", device ? 2 : asynchronous ? 3 : 1, track);
        vx_trace_time_us(&out, timestamp);
        if (instant) vx_trace_append(&out, ",\"s\":\"t\"");
        else if (asynchronous) vx_trace_append(&out, ",\"id\":\"%" PRIu64 "\"", e->sequence);
        else { vx_trace_append(&out, ",\"dur\":"); vx_trace_time_us(&out, e->duration_ns); }
        vx_trace_append(&out, ",\"args\":{\"runtimeId\":\"%" PRIu64 "\",\"modelId\":\"%" PRIu64 "\",\"compiledModelId\":\"%" PRIu64 "\",\"contextId\":\"%" PRIu64 "\",\"executionId\":\"%" PRIu64 "\",\"graphId\":\"%" PRIu64 "\",\"graphRevision\":\"%" PRIu64 "\",\"backend\":", e->identity.runtime_id, e->identity.model_id, e->identity.compiled_model_id, e->identity.context_id, e->identity.execution_id, e->identity.graph_id, e->identity.graph_revision);
        vx_trace_quoted(&out, e->backend);
        vx_trace_append(&out, ",\"phase\":\"%s\"", vx_trace_phase_label(e->phase));
        vx_trace_append(&out, ",\"activity\":\"%s\"", vx_trace_activity_label(e->activity));
        if (e->queue.queue_id) vx_trace_append(&out,
            ",\"deviceId\":\"%" PRIu64 "\",\"queueId\":\"%" PRIu64 "\",\"submissionId\":\"%" PRIu64 "\"",
            e->queue.device_id, e->queue.queue_id, e->queue.submission_id);
        if (e->activity == VX_TRACE_ACTIVITY_COPY) vx_trace_append(&out,
            ",\"copySource\":%d,\"copyDestination\":%d,\"copyBytes\":\"%" PRIu64 "\"",
            e->copy_source, e->copy_destination, e->copy_bytes);
        if (e->program[0]) {
            vx_trace_append(&out, ",\"program\":"); vx_trace_quoted(&out, e->program);
            vx_trace_append(&out, ",\"entryPoint\":"); vx_trace_quoted(&out, e->entry);
        }
        if (e->tensor[0]) {
            vx_trace_append(&out, ",\"tensorName\":"); vx_trace_quoted(&out, e->tensor);
        }
        if (e->kind == VX_TRACE_EVENT_DEVICE) {
            vx_trace_append(&out, ",\"deviceDurationNs\":\"%" PRIu64 "\"", e->duration_ns);
            vx_trace_append(&out, ",\"hostObservedNs\":\"%" PRIu64 "\",\"clockAligned\":%s", e->start_ns, aligned ? "true" : "false");
            if (e->clock.method) vx_trace_append(&out,
                ",\"clockMethod\":\"%s\",\"earliestStartNs\":\"%" PRIu64 "\",\"latestStartNs\":\"%" PRIu64 "\"",
                aligned ? "calibrated" : "bounded", e->clock.earliest_ns, e->clock.latest_ns);
        }
        if (e->kind == VX_TRACE_EVENT_MEMORY) {
            const char* action = e->memory_action == VX_TRACE_MEMORY_ACTION_EXISTING ? "existing" :
                e->memory_action == VX_TRACE_MEMORY_ACTION_ALLOCATE ? "allocate" : "free";
            vx_trace_append(&out, ",\"allocationId\":\"%" PRIu64 "\",\"action\":\"%s\",\"bytes\":\"%" PRIu64 "\",\"liveBytes\":\"%" PRIu64 "\",\"valueRelation\":\"requested\"",
                e->allocation_id, action, e->allocation_bytes, e->live_bytes);
        }
        if (e->schedule_index >= 0) {
            vx_trace_append(&out, ",\"scheduleIndex\":%d,\"output\":", e->schedule_index); vx_trace_quoted(&out, e->output);
            vx_trace_append(&out, ",\"fused\":%s", e->fused ? "true" : "false");
        }
        vx_trace_append(&out, ",\"metadataTruncated\":%s}}", e->truncated ? "true" : "false");
        /* Completion waits may overlap without nesting. Async pairs keep
         * them independent instead of inventing occupied CPU threads. */
        if (asynchronous) {
            vx_trace_append(&out, ",{\"name\":"); vx_trace_quoted(&out, e->name);
            vx_trace_append(&out, ",\"cat\":\"host.await\",\"ph\":\"e\",\"pid\":3,\"tid\":%" PRIu64 ",\"id\":\"%" PRIu64 "\",\"ts\":", track, e->sequence);
            vx_trace_time_us(&out, e->start_ns + e->duration_ns);
            vx_trace_append(&out, "}");
        }
        if (e->kind == VX_TRACE_EVENT_MEMORY) {
            vx_trace_append(&out, ",{\"name\":"); vx_trace_quoted(&out, e->name);
            vx_trace_append(&out, ",\"cat\":\"memory.capacity\",\"ph\":\"C\",\"pid\":1,\"tid\":0,\"ts\":");
            vx_trace_time_us(&out, e->start_ns);
            vx_trace_append(&out, ",\"args\":{\"requestedBytes\":%" PRIu64 "}}", e->live_bytes);
        }
    }
    if (end == trace->count) {
        if (trace->memory) for (int i = 0; i < 2; i++) {
            const VxProcessMemorySampleV1* sample = &trace->samples[i];
            if (!(sample->available_mask & VX_PROCESS_MEMORY_AVAILABLE_CURRENT_RSS)) continue;
            vx_trace_append(&out, ",");
            vx_trace_append(&out, "{\"name\":\"Process RSS\",\"cat\":\"memory\",\"ph\":\"C\",\"pid\":1,\"tid\":0,\"ts\":");
            vx_trace_time_us(&out, trace->memory_end_ns[i]);
            vx_trace_append(&out, ",\"args\":{\"bytes\":%" PRIu64 "}}", sample->rss_bytes);
        }
        vx_trace_append(&out, "]}");
    }
    if (out.failed) { free(out.data); return -1; }
    *json = out.data; *size = out.size;
    return 1;
}
