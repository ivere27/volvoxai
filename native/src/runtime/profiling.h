/* Private, bounded observations. No collector is consulted by numerical kernels. */
#ifndef VOLVOXAI_PROFILING_H
#define VOLVOXAI_PROFILING_H
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
#include "vx_lifecycle.h"
#include "resource_sampling.h"
#include "execution_plan.h"

typedef struct VxTrace VxTrace;
/* Collector-only tag. Public events use an explicit timing oneof. */
typedef enum { VX_TRACE_EVENT_HOST_OPERATION, VX_TRACE_EVENT_HOST_NODE,
    VX_TRACE_EVENT_DEVICE, VX_TRACE_EVENT_MEMORY, VX_TRACE_EVENT_HOST_WORK } VxTraceEventKind;
#define VX_TRACE_MAX_BACKENDS 8
typedef struct {
    char backend[32];
    VxTraceTimingSupport support;
    int node_timing_available;
    uint64_t pass_intervals, node_intervals, failed_intervals, unsupported_passes;
    int splits_passes, adds_barriers;
    int program_timing_available;
    uint64_t program_intervals;
    uint64_t calibrated_intervals, bounded_intervals, copy_intervals;
    uint64_t host_copy_calls, host_synchronize_calls, host_submit_calls, host_completions;
} VxTraceDeviceCoverage;
typedef struct { uint64_t device_id, queue_id, submission_id; } VxTraceQueue;
/* Internal absolute host times; records convert them to capture-relative ns. */
typedef struct {
    uint64_t earliest_ns, latest_ns;
    VxTraceClockMethod method;
} VxTraceClock;
typedef struct {
    uint64_t elapsed_ns;
    VxTraceClock clock;
    uint64_t host_start_ns; /* Only used for a deferred host completion span. */
} VxDeviceTraceResult;
/* Borrowed only while the owning execution is active. The collector copies it. */
typedef struct {
    VxTracePhase phase;
    int index, fused;
    const char *name, *output, *tensor;
} VxTraceWork;
/* Borrowed only for a synchronous native pass. Records copy these strings. */
typedef struct {
    uint64_t host_start_ns;
    const char *name, *output;
    int index, fused;
    VxTracePhase phase;
    const char *program, *entry, *tensor;
    VxTraceQueue queue;
    VxTraceActivity activity;
    VxMemorySpace copy_source, copy_destination;
    uint64_t copy_bytes;
    VxTraceClock clock;
} VxDeviceTraceSpan;
typedef struct VxTraceMemoryOwner VxTraceMemoryOwner;
typedef void (*VxMemoryInspect)(void*, VxMemoryAllocator, uint64_t, uint64_t);
typedef struct {
    VxTraceMemoryOwner* owner;
    uint64_t source_id;
    VxExecutionIdentity identity;
    VxMemoryInspect inspect;
    void* inspection;
} VxMemoryObserver;
typedef struct {
    uint64_t start_ns, existing, live, peak, allocated, freed, dropped;
    int observed, complete;
} VxAllocatorMemory;
typedef struct {
    uint64_t sequence, start_ns, duration_ns, track_id, plan_id;
    VxExecutionIdentity identity;
    VxTraceEventKind kind;
    int schedule_index, fused, truncated;
    /* Interned in the trace's fixed text pool: never NULL, stable until release. */
    const char *name, *backend, *output;
    VxTracePhase phase;
    const char *program, *entry, *tensor;
    VxTraceQueue queue;
    VxTraceActivity activity;
    VxMemorySpace copy_source, copy_destination;
    uint64_t copy_bytes;
    VxTraceClock clock;
    /* Private async device ticket; never exposed in a completed trace. */
    uint32_t ticket;
    int invalid;
    uint64_t allocation_id, allocation_bytes, live_bytes;
    VxMemoryAllocator allocator;
    VxTraceMemoryAction memory_action;
    int (*poll)(uint32_t ticket, VxDeviceTraceResult* result);
    void (*release)(uint32_t ticket);
} VxTraceRecord;
typedef struct VxTraceScope {
    VxTrace* trace;
    VxExecutionIdentity identity;
    uint64_t start_ns, track_id;
    const char* name;
    const char* backend;
    VxTraceWork work;
    VxTraceQueue queue;
    uint64_t plan_id;
    uint64_t external_range; /* NVTX range of this operation, when annotated. */
    /* Capture once after execution preparation has finalized tensor storage. */
    void (*prepare_plan)(void*, struct VxTraceScope*);
    void* plan_context;
} VxTraceScope;
static inline void vx_trace_prepare_plan(VxTraceScope* scope) {
    if (scope && scope->prepare_plan) {
        void (*prepare)(void*, VxTraceScope*) = scope->prepare_plan;
        scope->prepare_plan = NULL;
        prepare(scope->plan_context, scope);
    }
}
/* Vendor-tool mechanisms reported in TraceInfo.external. */
typedef enum {
    VX_TRACE_EXTERNAL_NVTX, VX_TRACE_EXTERNAL_VK_DEBUG_UTILS, VX_TRACE_EXTERNAL_KHR_DEBUG,
    VX_TRACE_EXTERNAL_RENDERDOC, VX_TRACE_EXTERNAL_WEBGPU_DEBUG_GROUP, VX_TRACE_EXTERNAL_COUNT
} VxTraceExternalMechanism;
typedef struct {
    VxObservationStatus status; /* UNSPECIFIED: not attempted */
    uint64_t ranges;
} VxTraceExternalCoverage;
const char* vx_trace_external_mechanism_name(VxTraceExternalMechanism mechanism);
const char* vx_trace_external_mechanism_backend(VxTraceExternalMechanism mechanism);
typedef struct {
    VxTraceState state;
    VxTraceDetail detail;
    int device_timing;
    int memory, resources, execution_plans;
    uint64_t resource_count, dropped_resource_samples, collector_bytes, sample_interval_ns;
    const VxResourceSample* resource_samples;
    uint64_t plan_count, dropped_plans;
    const unsigned char* plan_data;
    uint64_t capacity_bytes, count, dropped, active, pending;
    size_t device_count;
    VxTraceDeviceCoverage devices[VX_TRACE_MAX_BACKENDS];
    uint64_t clock_resolution_ns;
    const VxTraceRecord* records; /* Only readable in TRACE_STATE_READY. */
    uint64_t memory_start_ns[2], memory_end_ns[2];
    VxProcessMemorySampleV1 samples[2];
    VxAllocatorMemory allocators[VX_MEMORY_ALLOCATOR_COUNT];
    int external_annotations, external_capture;
    VxTraceExternalCoverage external[VX_TRACE_EXTERNAL_COUNT];
    uint64_t origin_ns; /* Capture-relative zero on the host monotonic clock. */
} VxTraceView;

/* Observers retain only a small weak owner, never completed trace storage.
 * Owners serialize their own allocation mutations; the collector serializes
 * cross-owner accounting. Returns true only when a new inventory is needed. */
int vx_memory_observer_attach(VxMemoryObserver* observer, const VxTraceScope* scope);
void vx_memory_observer_clear(VxMemoryObserver* observer);
void vx_memory_observer_copy(VxMemoryObserver* to, const VxMemoryObserver* from);
void vx_memory_record(VxMemoryObserver* observer, VxMemoryAllocator allocator,
    uint64_t key, uint64_t bytes, VxTraceMemoryAction action);
const char* vx_memory_allocator_name(VxMemoryAllocator allocator);
const char* vx_memory_allocator_backend(VxMemoryAllocator allocator);
VxMemorySpace vx_memory_allocator_space(VxMemoryAllocator allocator);
uint64_t vx_memory_allocation_id(VxMemoryObserver*, VxMemoryAllocator, uint64_t key);
void vx_memory_lost(VxMemoryObserver* observer, VxMemoryAllocator allocator);
void vx_trace_memory_bridge(VxTraceScope* scope,
    uint32_t (*start)(VxTraceScope*, uint32_t), void (*stop)(uint32_t));

uint64_t vx_trace_now_ns(void);
/* IDs are allocated only on active trace routes, never native pointer values. */
uint64_t vx_trace_next_id(void);
uint64_t vx_trace_object_id(atomic_uint_fast64_t* identity);
VxTraceClock vx_trace_clock_bounds(uint64_t host_before, uint64_t host_after,
    uint64_t device_elapsed, uint64_t offset, uint64_t interval_elapsed);
VxTraceClock vx_trace_clock_calibrated(uint64_t device_start, uint64_t device_anchor,
    unsigned bits, double period_ns, uint64_t host_anchor, uint64_t uncertainty,
    uint64_t host_before, uint64_t host_after, uint64_t elapsed);
void vx_trace_host_activity(VxTraceScope* scope, uint64_t start, uint64_t end,
    const VxDeviceTraceSpan* metadata);
VxTrace* vx_trace_create(size_t capacity, VxTraceDetail detail, int device_timing,
    int memory, int resources, uint64_t resource_interval_ns, int execution_plans);
/* TraceOptions.external. Set before the trace is attached to a runtime. */
void vx_trace_set_external(VxTrace* trace, int annotations, int capture);
/* Annotations run only on instrumented routes of an annotating trace. */
int vx_trace_external(const VxTraceScope* scope);
/* A host range for one node or training phase (NVTX push/pop). */
void vx_trace_external_push(VxTraceScope* scope, VxTracePhase phase, int index,
    const char* name, const char* output);
void vx_trace_external_pop(VxTraceScope* scope);
/* Records what a backend mechanism did for this trace's coverage. */
void vx_trace_external_note(VxTraceScope* scope, VxTraceExternalMechanism mechanism,
    VxObservationStatus status, uint64_t ranges);
/* Formats "<index> <name> -> <output>" for device labels. */
void vx_trace_external_label(char* label, size_t capacity, VxTracePhase phase, int index,
    const char* name, const char* output);
void vx_trace_retain(VxTrace* trace);
void vx_trace_release(VxTrace* trace);
void vx_trace_stop(VxTrace* trace);
/* Registers a callback run when a stopped trace's host operations finish.
 * A null notify clears the registration only if it still names context. */
void vx_trace_watch_drain(VxTrace* trace, void (*notify)(void*), void* context);
/* Application range in platform monotonic nanoseconds; COLLECTING traces only. */
VxStatus vx_trace_annotate(VxTrace* trace, const char* name, uint64_t start_ns, uint64_t end_ns);
void vx_trace_abandon(VxTrace* trace);
void vx_trace_view(VxTrace* trace, VxTraceView* view);
int vx_trace_scope_begin(VxTrace* trace, VxTraceScope* scope,
    const VxExecutionIdentity* identity, const char* name, const char* backend);
void vx_trace_scope_end(VxTraceScope* scope);
int vx_trace_memory_enabled(const VxTrace* trace);
int vx_trace_nodes(const VxTraceScope* scope);
int vx_trace_device_enabled(const VxTraceScope* scope);
int vx_trace_device_nodes(const VxTraceScope* scope);
void vx_trace_device_status(VxTraceScope* scope, int available, int nodes, int splits, int barriers);
void vx_trace_program_status(VxTraceScope* scope, int available);
void vx_trace_device_fail(VxTraceScope* scope);
void vx_trace_drop(VxTraceScope* scope);
void vx_trace_device(VxTraceScope* scope, uint64_t host_start, uint64_t duration, const char* name);
void vx_trace_device_node(VxTraceScope* scope, uint64_t host_start, uint64_t duration,
    int index, const char* name, const char* output, int fused);
size_t vx_trace_device_capacity(VxTraceScope* scope, size_t requested);
int vx_trace_timestamp_duration(uint64_t start, uint64_t end, unsigned valid_bits,
    double period_ns, uint64_t host_elapsed_ns, uint64_t* duration_ns);
/* Poll returns 1 while pending, 0 on success, -1 on failure. Takes ownership
 * of the ticket even when the bounded collector cannot accept it. Callbacks
 * run synchronously on the WASM owner, never from a device promise. */
void vx_trace_defer_device(VxTraceScope* scope, uint64_t host_start, const char* name,
    uint32_t ticket, int (*poll)(uint32_t, VxDeviceTraceResult*), void (*release)(uint32_t));
void vx_trace_defer_device_node(VxTraceScope* scope, uint64_t host_start,
    int index, const char* name, const char* output, int fused,
    uint32_t ticket, int (*poll)(uint32_t, VxDeviceTraceResult*), void (*release)(uint32_t));
void vx_trace_node(VxTraceScope* scope, uint64_t start, int index,
    const char* op, const char* output, int fused);
void vx_trace_host_work(VxTraceScope* scope, uint64_t start,
    const VxTraceWork* work, const char* backend);
void vx_trace_host_program(VxTraceScope* scope, uint64_t start,
    const char* program, const char* entry);
VxDeviceTraceSpan vx_trace_program_span(VxTraceScope* scope,
    const char* program, const char* entry);
void vx_trace_device_span(VxTraceScope* scope, const VxDeviceTraceSpan* span, uint64_t duration);
void vx_trace_defer_device_span(VxTraceScope* scope, const VxDeviceTraceSpan* span,
    uint32_t ticket, int (*poll)(uint32_t, VxDeviceTraceResult*), void (*release)(uint32_t));
void vx_trace_defer_host_activity(VxTraceScope* scope, const VxDeviceTraceSpan* span,
    uint32_t ticket, int (*poll)(uint32_t, VxDeviceTraceResult*), void (*release)(uint32_t));
/* GetTraceSummary over a READY view. Strings borrow the trace's interned text
 * and plans; the caller frees `rows` with vx_trace_summary_free. */
typedef struct {
    VxExecutionIdentity identity; /* model, compiled model and context */
    uint64_t plan_id;
    const char *backend, *name, *output;
    VxTracePhase phase;
    VxTraceActivity activity;
    int device, schedule_index, fused;
    uint64_t count, total_ns, min_ns, median_ns, p95_ns, max_ns, copy_bytes;
    double share;
    const VxExecutionPlanStep* step; /* node rows with a captured plan */
    int has_rates;
    double flops_per_second, bytes_per_second;
} VxTraceSummaryRow;
typedef struct {
    VxTraceSummaryRow* rows;
    size_t count;
    int truncated;
    uint64_t excluded;
} VxTraceSummary;
int vx_trace_summarize(const VxTraceView* view, VxTraceSummaryGrouping grouping,
    size_t max_rows, VxTraceSummary* summary);
void vx_trace_summary_free(VxTraceSummary* summary);
/* Export one immutable event page; the caller frees the returned JSON bytes. */
int vx_trace_export(const VxTraceView* view, size_t offset, size_t limit,
    char** json, size_t* size);

/* Copied metadata. A plan occupies one bounded contiguous collector block. */
int vx_trace_plans_enabled(const VxTraceScope* scope);
void vx_trace_capture_plan(VxTraceScope* scope, const char* fingerprint,
    const char* signature, uint64_t storage_generation, uint32_t steps, uint32_t tensors, uint32_t references,
    uint32_t allocations, void (*fill)(VxExecutionPlan*, void*), void* context);
const VxExecutionPlan* vx_trace_plan_at(const VxTraceView* view, size_t offset);
VxMemoryResourceRole vx_memory_allocator_role(VxMemoryAllocator allocator);

typedef struct {
    uint64_t active_output_bytes;
    uint64_t result_capacity_bytes, idle_result_bytes;
} VxMemoryView;

typedef struct VxMemoryInventory {
    VxMemoryAllocation* items;
    size_t count, capacity;
    int truncated;
    VxMemoryOwnerKind owner_kind;
    uint64_t owner_id;
} VxMemoryInventory;
void vx_memory_inventory_add(VxMemoryInventory*, VxMemoryAllocator,
    uint64_t key, uint64_t bytes, VxMemoryResourceRole, const char* name);
void vx_memory_inventory_inspect(void*, VxMemoryAllocator, uint64_t, uint64_t);
void vx_model_memory_inventory(VxModel*, VxMemoryInventory*);
void vx_compiled_memory_inventory(VxCompiledModel*, VxMemoryInventory*);
VxStatus vx_context_memory_inventory(VxExecutionContext*, VxMemoryInventory*, VxMemoryView*);

/* Runtime attachment uses the existing runtime lock; the inactive execution
 * boundary only performs one atomic load. These are not public APIs. */
void vx_model_profile_begin(VxModel* model, VxTraceScope* scope,
    const char* name, const char* backend);
void vx_model_profile_end(VxModel* model, VxTraceScope* scope);
void vx_runtime_profile_end(VxRuntime* runtime, VxTraceScope* scope);
VxStatus vx_runtime_trace_start(VxRuntime* runtime, VxTrace* trace);
void vx_runtime_trace_stop(VxRuntime* runtime, VxTrace* trace);
void vx_profiling_set_public_id(void* object, int kind, uint64_t id);

VxStatus vx_runtime_memory_view(VxRuntime* runtime, VxMemoryView* view);
#endif
