/*
 * WebGPU as a VxBackend over the host device bridge.
 *
 * Every other GPU backend calls its API directly because that API is
 * synchronous C. WebGPU is not reachable from C at all, so this backend keeps
 * the same VxBackend shape and forwards the device half to the host through
 * gpu_bridge.h. Planning stays here; the host holds the GPUDevice.
 *
 * The bookkeeping below exists because the bridge names buffers by host
 * pointer -- the identity the runtime already uses for device slots -- while
 * the host needs to be told when a pointer starts and stops naming device
 * storage. Nothing else is remembered on this side.
 */

#include "backend_config.h"

#if VOLVOXAI_ENABLE_WEBGPU

#include "gpu_bridge.h"
#include "shader_catalog_profile.h"
#include "backend.h"
#include "webgpu_domain.h"
#include "engine_internal.h"
#include "profiling.h"
#include "incremental_runtime.h"
#include "../runtime/attention_mask.h"
#include <math.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* Generation-tagged observer IDs cross the bridge, never WASM pointers. JS
 * reports successful GPUBuffer creation/destruction on the serialized owner. */
#define VX_WEBGPU_MEMORY_OBSERVERS 64
static struct { uint32_t id; VxMemoryObserver observer; } vx_webgpu_memory_observers[VX_WEBGPU_MEMORY_OBSERVERS];
static uint32_t vx_webgpu_next_memory_id;
void vx_wasm_gpu_memory_event(uint32_t id, uint32_t action, uint32_t resource,
    uint32_t bytes_low, uint32_t bytes_high) {
    for (size_t i = 0; i < VX_WEBGPU_MEMORY_OBSERVERS; i++) {
        if (!id || vx_webgpu_memory_observers[i].id != id) continue;
        VxMemoryObserver* observer = &vx_webgpu_memory_observers[i].observer;
        if (action > 2) vx_memory_lost(observer, VX_MEMORY_WEBGPU_BUFFER);
        else vx_memory_record(observer, VX_MEMORY_WEBGPU_BUFFER, resource,
            (uint64_t)bytes_low | ((uint64_t)bytes_high << 32), (VxTraceMemoryAction)action);
        return;
    }
}
static void vx_webgpu_memory_stop(uint32_t id) {
    vx_gpu_memory_stop(id);
    for (size_t i = 0; i < VX_WEBGPU_MEMORY_OBSERVERS; i++) if (vx_webgpu_memory_observers[i].id == id) {
        vx_webgpu_memory_observers[i].id = 0;
        vx_memory_observer_clear(&vx_webgpu_memory_observers[i].observer);
        return;
    }
}
static uint32_t vx_webgpu_memory_start(VxTraceScope* scope, uint32_t capacity) {
    for (size_t i = 0; i < VX_WEBGPU_MEMORY_OBSERVERS; i++) {
        if (vx_webgpu_memory_observers[i].id || vx_webgpu_next_memory_id == UINT32_MAX) continue;
        uint32_t id = ++vx_webgpu_next_memory_id;
        vx_memory_observer_attach(&vx_webgpu_memory_observers[i].observer, scope);
        /* GPU buffers are shared by this bridge, rather than owned by the
         * first context which requests its inventory. */
        vx_webgpu_memory_observers[i].observer.identity = (VxTraceIdentity){.runtime_id = scope->identity.runtime_id};
        vx_webgpu_memory_observers[i].id = id;
        if (vx_gpu_memory_start(id, capacity)) return id;
        vx_memory_lost(&vx_webgpu_memory_observers[i].observer, VX_MEMORY_WEBGPU_BUFFER);
        vx_webgpu_memory_stop(id);
        return 0;
    }
    VxMemoryObserver observer = {0};
    vx_memory_observer_attach(&observer, scope);
    vx_memory_lost(&observer, VX_MEMORY_WEBGPU_BUFFER);
    vx_memory_observer_clear(&observer);
    return 0;
}
void vx_webgpu_memory_begin(VxTraceScope* scope) {
    vx_trace_memory_bridge(scope, vx_webgpu_memory_start, vx_webgpu_memory_stop);
}

/*
 * One resident span. A graph reuses activation storage, so several logical
 * tensors can name the same pointer; the table is keyed by pointer for that
 * reason and holds no per-tensor state.
 */
typedef struct {
    uint32_t host_ptr;
    uint32_t bytes;
    int host_dirty;      /* Device wrote it; a host read must sync first. */
} VxWebGpuSpan;

/* Stable blocks grow with admitted graph metadata and row staging. Existing
 * host addresses remain valid while encoded commands reference them. */
typedef struct VxWebGpuArenaBlock {
    struct VxWebGpuArenaBlock* next;
    size_t capacity;
    _Alignas(16) unsigned char data[];
} VxWebGpuArenaBlock;

typedef struct {
    unsigned references;
    VxGpuLimits limits;
    int device_resident;
    VxWebGpuArenaBlock* arena;
    size_t arena_used;
    VxWebGpuSpan* spans;
    size_t span_capacity;
    size_t span_count;
    int pass_open;
    int encode_failed;
    /* Variant the host built for the most recent node, so a report can name
     * the shader that actually ran rather than the one that was preferred. */
    int last_variant;
    int (*encode)(uint32_t);
} VxWebGpuState;

#define g_webgpu (*(VxWebGpuState*)vx_engine_state_current()->webgpu_state)
#define g_webgpu_arena (g_webgpu.arena)
#define g_webgpu_arena_used (g_webgpu.arena_used)
#define validating (vx_engine_state_current()->webgpu_validation)

static VxWebGpuSpan* vx_webgpu_span(uint32_t host_ptr) {
    size_t index;
    if (!host_ptr) return NULL;
    for (index = 0; index < g_webgpu.span_count; index++)
        if (g_webgpu.spans[index].host_ptr == host_ptr) return &g_webgpu.spans[index];
    for (index = 0; index < g_webgpu.span_count; index++) {
        VxWebGpuSpan* span = &g_webgpu.spans[index];
        if (host_ptr >= span->host_ptr && host_ptr < span->host_ptr + span->bytes)
            return span;
    }
    return NULL;
}

/*
 * Metadata the engine materializes for a shader to read.
 *
 * A strided shader takes its extents as a storage buffer rather than a
 * uniform, so the engine has to put those words somewhere the host can name.
 * Each node needs its own region: `writeBuffer` copies at the call, but the
 * dispatches all run at submit, so one buffer reused across nodes would have
 * every node reading the last node's strides.
 *
 * The cursor rewinds at each forward; stable blocks retain their high-water
 * capacity. Device limits and allocation failures bound growth.
 */
/* Four header words and three stride vectors of rank <= 8. */
#define VX_WEBGPU_SCRATCH_WORDS 28u

/* Each allocation has a stable address, including when a later row or graph
 * needs another block. Reallocating a single byte arena would invalidate the
 * addresses already retained in GPU bindings. */
static void* vx_webgpu_arena(size_t bytes) {
    if (!bytes || bytes > UINT32_MAX - 15u || bytes > g_webgpu.limits.maxBufferSize) return NULL;
    size_t offset = g_webgpu_arena_used, base = 0;
    VxWebGpuArenaBlock** link = &g_webgpu.arena;
    while (*link) {
        VxWebGpuArenaBlock* block = *link;
        if (offset < block->capacity) {
            size_t aligned = (offset + 15u) & ~(size_t)15u;
            if (aligned <= block->capacity && bytes <= block->capacity - aligned) {
                g_webgpu_arena_used = base + aligned + bytes;
                return block->data + aligned;
            }
            offset = block->capacity;
        }
        offset -= block->capacity;
        base += block->capacity;
        link = &block->next;
    }
    size_t capacity = 64u * 1024u;
    while (capacity < bytes && capacity <= SIZE_MAX / 2u) capacity *= 2u;
    if (capacity < bytes || capacity > SIZE_MAX - sizeof(VxWebGpuArenaBlock) || base > SIZE_MAX - capacity) return NULL;
    VxWebGpuArenaBlock* block = malloc(sizeof(*block) + capacity);
    if (!block) return NULL;
    block->next = NULL; block->capacity = capacity; *link = block;
    g_webgpu_arena_used = base + bytes;
    return block->data;
}

static uint32_t* vx_webgpu_scratch(void) {
    return (uint32_t*)vx_webgpu_arena(
        VX_WEBGPU_SCRATCH_WORDS * sizeof(uint32_t));
}

/*
 * Strides that read `input` while walking `output`'s coordinates.
 *
 * The same rule `gpu_broadcast_strides` applies for the native backends: the
 * operand shape is right-aligned against the output, and an axis of extent one
 * gets a stride of zero so every output coordinate reads the same element.
 */
static int vx_webgpu_broadcast_strides(const T* input, const T* output,
                                       uint32_t* strides) {
    long contiguous[8];
    long stride = 1;
    int offset, dimension;
    if (!input || !output || output->ndim < 0 || output->ndim > 8 ||
        input->ndim < 0 || input->ndim > output->ndim) return 0;
    for (dimension = input->ndim - 1; dimension >= 0; dimension--) {
        if (input->shape[dimension] <= 0 || stride > UINT32_MAX) return 0;
        contiguous[dimension] = stride;
        stride *= input->shape[dimension];
    }
    offset = output->ndim - input->ndim;
    for (dimension = 0; dimension < output->ndim; dimension++) {
        int source = dimension - offset;
        int input_dim = source < 0 ? 1 : input->shape[source];
        int output_dim = output->shape[dimension];
        if (output_dim <= 0 || (input_dim != 1 && input_dim != output_dim)) return 0;
        strides[dimension] = (source >= 0 && input_dim != 1)
            ? (uint32_t)contiguous[source] : 0u;
    }
    return 1;
}

/* Contiguous strides of a shape, in the same order the shaders walk. */
static int vx_webgpu_contiguous_strides(const T* tensor, uint32_t* strides) {
    long stride = 1;
    int dimension;
    if (!tensor || tensor->ndim < 0 || tensor->ndim > 8) return 0;
    for (dimension = tensor->ndim - 1; dimension >= 0; dimension--) {
        if (tensor->shape[dimension] <= 0 || stride > UINT32_MAX) return 0;
        strides[dimension] = (uint32_t)stride;
        stride *= tensor->shape[dimension];
    }
    return 1;
}

static int vx_webgpu_init(void* user_data) {
    VxEngineState* state = vx_engine_state_current();
    (void)user_data;
    /* Registry replacement initializes its successor before detaching the old
     * registry. Keep one context state alive across that overlap. */
    if (!state->webgpu_state) {
        if (!vx_gpu_available((uint32_t)(uintptr_t)VX_GPU_BRIDGE_ABI_HASH,
                (uint32_t)(uintptr_t)VX_SHADER_CATALOG_HASH))
            return VX_BACKEND_INIT_UNAVAILABLE;
        VxWebGpuState* created = calloc(1, sizeof(VxWebGpuState));
        if (!created) return VX_BACKEND_INIT_ERROR;
        if (!vx_gpu_limits((uint32_t)(uintptr_t)&created->limits, sizeof(created->limits))) {
            free(created);
            return VX_BACKEND_INIT_UNAVAILABLE;
        }
        created->encode = vx_gpu_encode;
        state->webgpu_state = created;
    }
    g_webgpu.references++;
    return VX_BACKEND_INIT_READY;
}

static void vx_webgpu_teardown(void* user_data) {
    VxEngineState* state = vx_engine_state_current();
    (void)user_data;
    if (!state->webgpu_state || --g_webgpu.references) return;
    if (g_webgpu.device_resident)
        for (size_t index = 0; index < g_webgpu.span_count; index++)
            vx_gpu_release(g_webgpu.spans[index].host_ptr);
    while (g_webgpu.arena) {
        VxWebGpuArenaBlock* next = g_webgpu.arena->next;
        free(g_webgpu.arena);
        g_webgpu.arena = next;
    }
    free(g_webgpu.spans);
    free(state->webgpu_state);
    state->webgpu_state = NULL;
}

/*
 * Residency.
 *
 * The runtime never calls VxBackend's alloc/upload hooks -- no backend fills
 * them, and Vulkan does its own residency from inside run(). This does the
 * same: the engine owns the bytes in linear memory, and a node's operands are
 * made to agree with the device just before they are encoded.
 */
static int vx_webgpu_resident(const void* host, size_t bytes, int is_weight) {
    VxWebGpuSpan* span;
    if (!host || !bytes || bytes > UINT32_MAX - 3u ||
        ((bytes + 3u) & ~(size_t)3u) > g_webgpu.limits.maxBufferSize) return 0;
    span = vx_webgpu_span((uint32_t)(uintptr_t)host);
    if (!span || span->host_ptr != (uint32_t)(uintptr_t)host) {
        if (g_webgpu.span_count == g_webgpu.span_capacity) {
            size_t capacity = g_webgpu.span_capacity ? g_webgpu.span_capacity * 2u : 64u;
            if (capacity <= g_webgpu.span_count || capacity > SIZE_MAX / sizeof(VxWebGpuSpan)) return 0;
            VxWebGpuSpan* grown = realloc(g_webgpu.spans, capacity * sizeof(*grown));
            if (!grown) return 0;
            g_webgpu.spans = grown; g_webgpu.span_capacity = capacity;
        }
        span = &g_webgpu.spans[g_webgpu.span_count];
    }
    if (!validating && !vx_gpu_ensure((uint32_t)(uintptr_t)host, (uint32_t)bytes, is_weight)) return 0;
    if (!validating) g_webgpu.device_resident = 1;
    if (span != &g_webgpu.spans[g_webgpu.span_count]) {
        span->bytes = (uint32_t)bytes;
        return 1;
    }
    g_webgpu.span_count++;
    span->host_ptr = (uint32_t)(uintptr_t)host;
    span->bytes = (uint32_t)bytes;
    span->host_dirty = 0;
    return 1;
}

static void vx_webgpu_reset(void* user_data) {
    (void)user_data;
    g_webgpu.pass_open = 0;
    g_webgpu.encode_failed = 0;
    g_webgpu.last_variant = -1;
}

static atomic_uint_fast64_t vx_webgpu_trace_device_id, vx_webgpu_trace_queue_id;
static VxTraceQueue vx_webgpu_trace_queue(void) {
    return (VxTraceQueue){vx_trace_object_id(&vx_webgpu_trace_device_id),
        vx_trace_object_id(&vx_webgpu_trace_queue_id), 0};
}
static int vx_webgpu_await_poll(uint32_t ticket, VxDeviceTraceResult* result) {
    int status = vx_gpu_await_read(ticket, (uint32_t)(uintptr_t)result);
    return status == VX_GPU_PENDING ? 1 : status == VX_GPU_OK ? 0 : -1;
}
/* Called only synchronously from an active GPU import. Device promises retain
 * bounded bridge tickets, never a C scope, tensor or linear-memory pointer. */
void vx_wasm_gpu_activity(uint32_t activity, uint32_t source, uint32_t destination,
    uint32_t bytes, uint32_t ticket, double start_us, double end_us) {
    VxTraceScope* scope = vx_engine_state_current()->profiling;
    if (!vx_trace_nodes(scope) || !isfinite(start_us) || start_us < 0 ||
        start_us >= (double)UINT64_MAX / 1000.0) {
        if (ticket) vx_gpu_await_release(ticket);
        return;
    }
    VxDeviceTraceSpan span = {.host_start_ns = (uint64_t)start_us * 1000,
        .name = activity == VX_TRACE_ACTIVITY_COPY ? "WebGPU buffer copy" :
            activity == VX_TRACE_ACTIVITY_SUBMIT ? "GPUQueue.submit" : "WebGPU asynchronous completion",
        .index = -1, .phase = scope->work.phase, .activity = (VxTraceActivity)activity,
        .copy_source = (VxMemorySpace)source, .copy_destination = (VxMemorySpace)destination,
        .copy_bytes = bytes, .queue = scope->queue.queue_id ? scope->queue : vx_webgpu_trace_queue()};
    if (activity == VX_TRACE_ACTIVITY_AWAIT) {
        if (ticket) vx_trace_defer_host_activity(scope, &span, ticket, vx_webgpu_await_poll, vx_gpu_await_release);
        else vx_trace_drop(scope);
    }
    else if (isfinite(end_us) && end_us >= start_us && end_us < (double)UINT64_MAX / 1000.0)
        vx_trace_host_activity(scope, span.host_start_ns, (uint64_t)end_us * 1000, &span);
}
_Static_assert(offsetof(VxDeviceTraceResult, clock) == 8 &&
    offsetof(VxDeviceTraceResult, host_start_ns) == 32 && sizeof(VxDeviceTraceResult) == 40,
    "private WebGPU timing result layout");
static int vx_webgpu_trace_poll(uint32_t ticket, VxDeviceTraceResult* duration) {
    int status = vx_gpu_trace_read(ticket, (uint32_t)(uintptr_t)duration);
    return status == VX_GPU_PENDING ? 1 : status == VX_GPU_OK ? 0 : -1;
}
/* Selected once per pass. Ordinary encoding does not consult the collector. */
static int vx_webgpu_encode_profiled(uint32_t dispatch) {
    VxTraceScope* scope = vx_engine_state_current()->profiling;
    uint64_t start = vx_trace_now_ns();
    int ticket = vx_gpu_trace_program_begin();
    int variant = vx_gpu_encode(dispatch);
    vx_gpu_trace_program_end();
    if (ticket > 0 && variant >= 0) {
        const VxGpuDispatch* header = (const VxGpuDispatch*)(uintptr_t)dispatch;
        const VxGpuVariant* candidates = (const VxGpuVariant*)(header + 1);
        uint32_t shader = (uint32_t)variant < header->variant_count
            ? candidates[variant].shader_id : VX_SHADER_COUNT;
        if (shader < VX_SHADER_COUNT) {
            VxDeviceTraceSpan span = vx_trace_program_span(scope,
                vx_shader_programs[shader].name, vx_shader_programs[shader].entry);
            span.host_start_ns = start;
            vx_trace_defer_device_span(scope, &span, (uint32_t)ticket,
                vx_webgpu_trace_poll, vx_gpu_trace_release);
        } else { vx_gpu_trace_release((uint32_t)ticket); vx_trace_device_fail(scope); }
    } else if (ticket > 0) {
        vx_gpu_trace_release((uint32_t)ticket); vx_trace_device_fail(scope);
    } else if (ticket < 0) vx_trace_drop(scope);
    return variant;
}
void vx_webgpu_begin_pass(int allow_nodes) {
    VxTraceScope* scope = vx_engine_state_current()->profiling;
    g_webgpu.encode = vx_gpu_encode;
    if (!scope) { vx_gpu_begin(); return; }
    if (scope && (vx_trace_nodes(scope) || vx_trace_device_enabled(scope))) {
        scope->queue = vx_webgpu_trace_queue();
        scope->queue.submission_id = vx_trace_next_id();
    }
    if (!vx_trace_device_enabled(scope)) {
        if (vx_trace_nodes(scope)) vx_gpu_begin_activity(); else vx_gpu_begin();
        return;
    }
    (void)allow_nodes; /* Both forward and training now carry program detail. */
    int nodes = vx_trace_device_nodes(scope);
    size_t capacity = vx_trace_device_capacity(scope, nodes ? 1024 : 1);
    if (!capacity) { vx_trace_drop(scope); vx_gpu_begin(); return; }
    uint64_t start = vx_trace_now_ns();
    int ticket = vx_gpu_begin_trace((uint32_t)capacity, nodes);
    vx_trace_device_status(scope, ticket != 0, 1, nodes && ticket > 0, 0);
    vx_trace_program_status(scope, ticket != 0);
    if (nodes && ticket > 0) g_webgpu.encode = vx_webgpu_encode_profiled;
    if (ticket > 0) {
        if (nodes) vx_gpu_trace_release((uint32_t)ticket); /* Batch admission lease. */
        else vx_trace_defer_device(scope, start, "WebGPU compute pass",
            (uint32_t)ticket, vx_webgpu_trace_poll, vx_gpu_trace_release);
    } else if (ticket < 0) vx_trace_device_fail(scope);
}
int vx_webgpu_end_pass(void) {
    int result = vx_gpu_end();
    VxTraceScope* scope = vx_engine_state_current()->profiling;
    if (scope) scope->queue.submission_id = 0;
    return result;
}
void vx_webgpu_trace_node_begin(int index, const char* name, const char* output, int fused) {
    VxTraceScope* scope = vx_engine_state_current()->profiling;
    if (!vx_trace_device_nodes(scope)) return;
    uint64_t start = vx_trace_now_ns();
    int ticket = vx_gpu_trace_node_begin();
    if (ticket > 0) vx_trace_defer_device_node(scope, start, index, name, output, fused,
        (uint32_t)ticket, vx_webgpu_trace_poll, vx_gpu_trace_release);
    else if (ticket < 0) vx_trace_drop(scope);
}
void vx_webgpu_trace_node_end(void) {
    if (vx_trace_device_nodes(vx_engine_state_current()->profiling)) vx_gpu_trace_node_end();
}

static void vx_webgpu_begin_forward(void* user_data) {
    (void)user_data;
    if (!g_use_webgpu || g_webgpu.pass_open) return;
    /* Everything handed out in the previous forward has been submitted and
     * read, so the arena starts again here rather than growing. */
    g_webgpu_arena_used = 0;
    if (!validating) vx_webgpu_begin_pass(1);
    g_webgpu.pass_open = 1;
    g_webgpu.encode_failed = 0;
}

static int vx_webgpu_end_forward(void* user_data) {
    int status;
    (void)user_data;
    if (!g_webgpu.pass_open) return 0;
    status = validating ? VX_GPU_OK : vx_webgpu_end_pass();
    g_webgpu.pass_open = 0;
    /* Submission is synchronous. A node that failed to encode cannot be
     * rescued by a successful submit, so report the earlier failure. */
    if (g_webgpu.encode_failed) return -1;
    return status == VX_GPU_OK ? 0 : -1;
}

static void vx_webgpu_mark_host(void* user_data, const void* host,
                                size_t bytes, int is_weight) {
    (void)user_data; (void)bytes; (void)is_weight;
    VxWebGpuSpan* span = vx_webgpu_span((uint32_t)(uintptr_t)host);
    if (span) {
        span->host_dirty = 0;
        vx_gpu_invalidate(span->host_ptr);
    }
}

/* The synchronous graph loop cannot suspend for a GPU -> CPU edge. Whole
 * graph admission prevents such edges; a violation fails before host math. */
static int vx_webgpu_sync_host(void* user_data, const void* host,
                               size_t bytes, int is_weight) {
    (void)user_data; (void)bytes; (void)is_weight;
    VxWebGpuSpan* span = vx_webgpu_span((uint32_t)(uintptr_t)host);
    return span && span->host_dirty ? VX_BACKEND_SYNC_FAILED : VX_BACKEND_SYNC_OK;
}

static int vx_webgpu_readback_poll(uint32_t ticket, void* data, size_t bytes) {
    int status = vx_gpu_readback(ticket, (uint32_t)(uintptr_t)data, (uint32_t)bytes);
    return status == VX_GPU_DEVICE_LOST ? VX_STATUS_DEVICE_LOST :
        status == VX_GPU_PENDING ? 1 : status == VX_GPU_OK ? 0 : -1;
}

int vx_webgpu_snapshot(const void* host, size_t bytes, VxDeviceSnapshot* snapshot) {
    if (!vx_engine_state_current()->webgpu_state) return 0;
    VxWebGpuSpan* span = vx_webgpu_span((uint32_t)(uintptr_t)host);
    if (!span || !span->host_dirty) return 0;
    size_t offset = (uint32_t)(uintptr_t)host - span->host_ptr;
    if (offset > span->bytes || bytes > span->bytes - offset) return -1;
    VxTraceScope* scope = vx_engine_state_current()->profiling;
    int ticket = vx_gpu_snapshot(span->host_ptr, (uint32_t)offset, (uint32_t)bytes, scope && vx_trace_nodes(scope));
    if (ticket <= 0) return -1;
    snapshot->ticket = (uint32_t)ticket;
    snapshot->poll = vx_webgpu_readback_poll;
    snapshot->release = vx_gpu_readback_release;
    return 1;
}

/*
 * Planning.
 *
 * One node becomes a shader choice, a binding list and a workgroup count. The
 * catalogue that `shader_id` indexes is generated, so adding an operator adds
 * a row there rather than a function here.
 *
 * Plan refusal rejects the complete WebGPU graph during compilation.
 */

#ifndef VX_WEBGPU_MAX_BINDINGS
#define VX_WEBGPU_MAX_BINDINGS 16u
#endif
#ifndef VX_WEBGPU_MAX_VARIANTS
#define VX_WEBGPU_MAX_VARIANTS 2u
#endif
#ifndef VX_WEBGPU_MAX_PARAM_BYTES
#define VX_WEBGPU_MAX_PARAM_BYTES 256u
#endif

/*
 * The wire the host reads: header, then variants, bindings and uniform bytes
 * in that order, packed to the counts the header declares.
 *
 * Packed, not a struct of maximum-sized arrays: the host finds each section by
 * adding the previous section's count, which is the only layout gpu_bridge.h
 * describes. A fixed-size struct would put the bindings where the host expects
 * the unused variants, and every field after the first would be read from the
 * wrong offset. The buffer is sized for the worst case; the header says how
 * much of it is meant.
 */
typedef struct {
    unsigned char bytes[sizeof(VxGpuDispatch) +
                        VX_WEBGPU_MAX_VARIANTS * sizeof(VxGpuVariant) +
                        VX_WEBGPU_MAX_BINDINGS * sizeof(VxGpuBinding) +
                        VX_WEBGPU_MAX_PARAM_BYTES];
} VxWebGpuDispatchBuffer;

/* Append one section to the wire, returning the next write position. */
static unsigned char* vx_webgpu_pack(unsigned char* cursor,
                                     const void* source, size_t bytes) {
    if (bytes) memcpy(cursor, source, bytes);
    return cursor + bytes;
}

/* One node's plan, before it is laid out for the host. */
typedef struct {
    uint32_t variant_count;
    VxGpuVariant variants[VX_WEBGPU_MAX_VARIANTS];
    uint32_t binding_count;
    VxGpuBinding bindings[VX_WEBGPU_MAX_BINDINGS];
    uint32_t params_bytes;
    uint32_t params_slot;
    unsigned char params[VX_WEBGPU_MAX_PARAM_BYTES];
} VxWebGpuNodePlan;

/* Variable-arity copies need one dispatch per port. Row execution adds at
 * most a gather per input, a mask, two numerical passes and one scatter. */
static uint32_t vx_webgpu_plan_capacity(const Node* node) {
    if (!node || node->nin < 0 || node->nout < 0) return 0;
    uint64_t count = (uint64_t)node->nin + (uint64_t)node->nout + 4u;
    if (count > UINT32_MAX || count > SIZE_MAX / sizeof(VxWebGpuNodePlan)) return 0;
    return (uint32_t)count;
}


/*
 * Typed node parameters.
 *
 * `engine_runtime_dispatch.inc` reads them the same way, through the same
 * public accessor, but its helpers are file-static. Repeating the two-line
 * shape here keeps this backend independent of that translation unit's
 * internals; the accessor they share is what keeps the readings identical.
 */
static int vx_webgpu_param_pair(const Node* node, VxNodeParamKey key,
                                VxNodeParamArraySlot slot, int fallback,
                                int* y, int* x) {
    int count = 0;
    const int32_t* values = vx_node_param_array(node, key, slot, &count);
    if (!values) { *y = fallback; *x = fallback; return 0; }
    if (count != 2) return -1;
    *y = values[0];
    *x = values[1];
    return 0;
}

static int vx_webgpu_pad_origin(const Node* node, int* y, int* x) {
    const VxCachedNodeParam* pads_param;
    const int32_t* pads;
    int count = 0;
    if (vx_webgpu_param_pair(node, VX_NODE_PARAM_PADDING,
                             VX_NODE_ARRAY_PADDING, 0, y, x) != 0) return -1;
    pads_param = vx_node_param(node, VX_NODE_PARAM_PADS);
    if (!pads_param || pads_param->kind == VX_NODE_PARAM_ABSENT ||
        pads_param->kind == VX_NODE_PARAM_NULL) return 0;
    pads = vx_node_param_array(node, VX_NODE_PARAM_PADS,
                               VX_NODE_ARRAY_PADS, &count);
    if (!pads || count != 4) return -1;
    *y = pads[0];
    *x = pads[1];
    return 0;
}

/*
 * Add one binding, making its span resident first.
 *
 * Binding and residency are the same moment: a slot the shader reads is a span
 * the device must already agree with. Splitting them would let a plan name a
 * buffer the host never uploaded.
 */
static int vx_webgpu_bind(VxWebGpuNodePlan* plan, const void* host, size_t bytes) {
    VxGpuBinding* binding;
    if (!host || !bytes || bytes > UINT32_MAX) return 0;
    if (plan->binding_count >= VX_WEBGPU_MAX_BINDINGS) return 0;
    if (!vx_webgpu_resident(host, bytes, 0)) return 0;
    binding = &plan->bindings[plan->binding_count];
    /* The slot this occupies and whether the dispatch writes it are the
     * shader's own declarations; `vx_webgpu_seal` fills them in from the
     * generated interface once the plan names its shader. */
    binding->slot = 0;
    binding->host_ptr = (uint32_t)(uintptr_t)host;
    binding->offset = 0;
    binding->bytes = (uint32_t)bytes;
    binding->writes = 0;
    plan->binding_count++;
    return 1;
}

/*
 * Finish a plan against what its shader declares.
 *
 * Every planner names a shader, binds its operands in order and supplies its
 * uniform words. Which slot each operand occupies, which of them the dispatch
 * writes, and how many uniform bytes the shader reads are not restated by the
 * planner -- they are read here from the generated interface. A planner that
 * bound the wrong number of operands, or built the wrong number of uniform
 * words, is refused rather than dispatched.
 */

static int vx_webgpu_seal(VxWebGpuNodePlan* plan) {
    const VxShaderInterface* declared;
    const VxGpuLimits* limits = &g_webgpu.limits;
    uint32_t index;
    if (!plan->variant_count || plan->variant_count > VX_WEBGPU_MAX_VARIANTS) return 0;
    for (index = 0; index < plan->variant_count; index++) {
        uint32_t id = plan->variants[index].shader_id;
        if (id >= VX_SHADER_COUNT) return 0;
    }
    if (plan->variants[0].shader_id >= VX_SHADER_COUNT) return 0;
    declared = &vx_shader_interface[plan->variants[0].shader_id];
    /* Alternative shaders share one binding list, so they must agree about
     * what that list means. */
    for (index = 1; index < plan->variant_count; index++) {
        const VxShaderInterface* other;
        if (plan->variants[index].shader_id >= VX_SHADER_COUNT) return 0;
        other = &vx_shader_interface[plan->variants[index].shader_id];
        if (other->storage_count != declared->storage_count ||
            other->writes_mask != declared->writes_mask ||
            other->params_slot != declared->params_slot ||
            other->params_bytes != declared->params_bytes ||
            memcmp(other->slots, declared->slots, sizeof(other->slots)) != 0) return 0;
    }
    if (declared->storage_count > VX_SHADER_MAX_PLANNED_BINDINGS) return 0;
    if (plan->binding_count != declared->storage_count) return 0;
    if (plan->params_bytes != declared->params_bytes) return 0;
    if (!limits->maxBindGroups ||
        plan->binding_count > limits->maxStorageBuffersPerShaderStage ||
        (plan->params_bytes && (!limits->maxUniformBuffersPerShaderStage ||
         declared->params_slot >= limits->maxBindingsPerBindGroup ||
         ((plan->params_bytes + 15u) & ~15u) > limits->maxUniformBufferBindingSize ||
         ((plan->params_bytes + 15u) & ~15u) > limits->maxBufferSize)))
        return 0;
    for (index = 0; index < plan->binding_count; index++) {
        if (declared->slots[index] >= limits->maxBindingsPerBindGroup ||
            !limits->minStorageBufferOffsetAlignment ||
            plan->bindings[index].offset % limits->minStorageBufferOffsetAlignment ||
            plan->bindings[index].bytes > UINT32_MAX - 3u ||
            ((plan->bindings[index].bytes + 3u) & ~3u) > limits->maxStorageBufferBindingSize)
            return 0;
        plan->bindings[index].slot = declared->slots[index];
        plan->bindings[index].writes = (declared->writes_mask >> index) & 1u;
    }
    plan->params_slot = declared->params_slot;
    uint32_t admitted = 0;
    for (index = 0; index < plan->variant_count; index++) {
        const VxGpuVariant* variant = &plan->variants[index];
        const VxShaderInterface* shader = &vx_shader_interface[variant->shader_id];
        if (shader->workgroup[0] > limits->maxComputeWorkgroupSizeX ||
            shader->workgroup[1] > limits->maxComputeWorkgroupSizeY ||
            shader->workgroup[2] > limits->maxComputeWorkgroupSizeZ ||
            (uint64_t)shader->workgroup[0] * shader->workgroup[1] * shader->workgroup[2] >
                limits->maxComputeInvocationsPerWorkgroup ||
            shader->workgroup_bytes > limits->maxComputeWorkgroupStorageSize)
            continue;
        if (variant->workgroup[0] > limits->maxComputeWorkgroupsPerDimension ||
            variant->workgroup[1] > limits->maxComputeWorkgroupsPerDimension ||
            variant->workgroup[2] > limits->maxComputeWorkgroupsPerDimension)
            continue;
        plan->variants[admitted++] = *variant;
    }
    plan->variant_count = admitted;
    return admitted != 0u;
}

static int vx_webgpu_params(VxWebGpuNodePlan* plan,
                            const uint32_t* values, size_t count) {
    if (count * sizeof(*values) > VX_WEBGPU_MAX_PARAM_BYTES) return 0;
    memcpy(plan->params, values, count * sizeof(*values));
    plan->params_bytes = (uint32_t)(count * sizeof(*values));
    return 1;
}

#if VOLVOXAI_ENABLE_TRAINING
void vx_webgpu_training_clear(void) {
    if (!vx_engine_state_current()->webgpu_state) return;
    while (g_webgpu.span_count)
        vx_gpu_release(g_webgpu.spans[--g_webgpu.span_count].host_ptr);
    g_webgpu.pass_open = 0;
}

int vx_webgpu_training_dispatch(const char* shader, const char* entry,
    void* const* hosts, const size_t* bytes, int count,
    const uint32_t groups[3], int validate) {
    uint32_t id = VX_SHADER_COUNT;
    if (!vx_engine_state_current()->webgpu_state || !shader || !entry ||
        !hosts || !bytes || !groups || count <= 0) return -1;
    for (size_t i = 0; i < sizeof(vx_training_shader_entries) / sizeof(vx_training_shader_entries[0]); i++) {
        if (!strcmp(shader, vx_training_shader_entries[i].name) &&
            !strcmp(entry, vx_training_shader_entries[i].entry)) {
            id = vx_training_shader_entries[i].id;
            break;
        }
    }
    if (id == VX_SHADER_COUNT) return -1;
    const VxShaderInterface* layout = &vx_shader_interface[id];
    /* Commands retain the shared module's numbered slots; a particular entry
       only consumes the subset declared by its generated interface. */
    VxWebGpuNodePlan plan = {0};
    plan.variant_count = 1;
    plan.variants[0].shader_id = id;
    memcpy(plan.variants[0].workgroup, groups, sizeof(plan.variants[0].workgroup));
    int previous = validating, accepted = 0;
    validating = validate;
    for (uint32_t i = 0; i < layout->storage_count; i++) {
        uint32_t slot = layout->slots[i];
        if (slot >= (uint32_t)count || !vx_webgpu_bind(&plan, hosts[slot], bytes[slot])) goto done;
    }
    if (layout->params_bytes) {
        if (layout->params_slot >= (uint32_t)count ||
            bytes[layout->params_slot] < layout->params_bytes ||
            !vx_webgpu_params(&plan, hosts[layout->params_slot], layout->params_bytes / 4u)) goto done;
    }
    if (!vx_webgpu_seal(&plan)) goto done;
    /* Writable aliases are forbidden by WebGPU, even in different slots. */
    for (uint32_t i = 0; i < plan.binding_count; i++)
        for (uint32_t j = 0; j < i; j++)
            if (plan.bindings[i].host_ptr == plan.bindings[j].host_ptr &&
                (plan.bindings[i].writes || plan.bindings[j].writes)) goto done;
    if (!validate) {
        VxWebGpuDispatchBuffer wire;
        VxGpuDispatch header = {1u, plan.binding_count, plan.params_bytes, 0u, plan.params_slot};
        unsigned char* cursor = vx_webgpu_pack(wire.bytes, &header, sizeof(header));
        cursor = vx_webgpu_pack(cursor, plan.variants, sizeof(VxGpuVariant));
        cursor = vx_webgpu_pack(cursor, plan.bindings, plan.binding_count * sizeof(VxGpuBinding));
        vx_webgpu_pack(cursor, plan.params, plan.params_bytes);
        if (g_webgpu.encode((uint32_t)(uintptr_t)wire.bytes) < 0) goto done;
        for (uint32_t i = 0; i < plan.binding_count; i++) {
            VxWebGpuSpan* span = vx_webgpu_span(plan.bindings[i].host_ptr);
            if (span && plan.bindings[i].writes) span->host_dirty = 1;
        }
    }
    accepted = 1;
done:
    validating = previous;
    return accepted ? 0 : -1;
}
#endif

/*
 * Dispatch grids are bounded by the exact device used for compilation.
 */
#define VX_WEBGPU_WORKGROUP_LIMIT (g_webgpu.limits.maxComputeWorkgroupsPerDimension)

/*
 * Spread `elements` over a 2D grid the way the uniform-elementwise shaders
 * index themselves: idx = gid.x + gid.y * (grid.x * divisor).
 *
 * The arithmetic mirrors `dispatchLinear2D` in WebGPUPhysicalDomain.ts, which
 * is the rule the shaders were written against.
 */
static void vx_webgpu_linear_grid(uint32_t elements, uint32_t divisor,
                                  uint32_t* workgroup) {
    uint32_t groups = divisor ? (elements + divisor - 1u) / divisor : 0u;
    uint32_t stride_limit = divisor ? UINT32_MAX / divisor : 1u;
    uint32_t x = groups < VX_WEBGPU_WORKGROUP_LIMIT
        ? (groups < stride_limit ? groups : stride_limit)
        : (VX_WEBGPU_WORKGROUP_LIMIT < stride_limit
               ? VX_WEBGPU_WORKGROUP_LIMIT : stride_limit);
    if (x == 0u) x = 1u;
    workgroup[0] = x;
    workgroup[1] = (groups + x - 1u) / x;
    workgroup[2] = 1u;
}

/*
 * A one-dimensional grid, for shaders that index with gid.x alone.
 *
 * Refused rather than truncated when it exceeds the device limit: a grid
 * that does not reach every element would leave the tail of the output
 * unwritten, which is a wrong answer rather than a failure.
 */
static int vx_webgpu_linear_1d(uint32_t elements, uint32_t divisor,
                               uint32_t* workgroup) {
    uint32_t groups = divisor ? (elements + divisor - 1u) / divisor : 0u;
    if (groups == 0u || groups > VX_WEBGPU_WORKGROUP_LIMIT) return 0;
    workgroup[0] = groups;
    workgroup[1] = 1u;
    workgroup[2] = 1u;
    return 1;
}

/* Resolve one of a node's inputs by port name, the way the engine's own GPU
 * selector does. `run` is handed only the primary input, and a binary operator
 * needs both. */
static T* vx_webgpu_port(const Node* node, VxPortKind key) {
    int index;
    if (!node) return NULL;
    for (index = 0; index < node->nin; index++)
        if (node->ins[index].port == key) return t_find(node->ins[index].name);
    return NULL;
}

/* Element count and byte span, refused when either leaves u32 descriptor
 * arithmetic -- the same bound the TypeScript domain proof enforces. */
static int vx_webgpu_extent(const T* tensor, uint32_t* elements, size_t* bytes) {
    if (!tensor || tensor->numel <= 0 || tensor->elem_size == 0) return 0;
    if ((uint64_t)tensor->numel > (uint64_t)UINT32_MAX / tensor->elem_size) return 0;
    *elements = (uint32_t)tensor->numel;
    *bytes = (size_t)tensor->numel * tensor->elem_size;
    return 1;
}

static int vx_webgpu_f32_extent(const T* tensor, uint32_t* elements,
                                size_t* bytes) {
    return tensor && tensor->dtype == T_F32 &&
        vx_webgpu_extent(tensor, elements, bytes);
}

/* A float parameter as the bits a uniform carries. */
static uint32_t vx_webgpu_f32_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}
/*
 * MaxPool2D and AveragePool2D.
 *
 * Not in the table: the twelve uniform words come from paired kernel, stride
 * and padding parameters, and reading a pair is a decision about what a node
 * declared rather than a value a tensor already knows.
 */
static int vx_webgpu_plan_pool2d(const Node* node, T* input, T* output,
                                 VxWebGpuNodePlan* plan, uint32_t shader_id) {
    int ky, kx, sy, sx, py, px;
    uint32_t params[16] = {0};
    uint32_t elements, out_elements;
    size_t input_bytes, output_bytes;
    int n, h, w, c, out_h, out_w;
    if (!input || !output || input->ndim != 4 || output->ndim != 4) return 0;
    int packed = input->dtype == T_I8 || input->dtype == T_U8;
    if (input->dtype != output->dtype || (!packed && input->dtype != T_F32)) return 0;
    if (packed && (shader_id != VX_SHADER_INFERENCE_MAX_POOL2_D ||
        !input->quantization.valid || !output->quantization.valid ||
        input->quantization.scale != output->quantization.scale ||
        input->quantization.zero_point != output->quantization.zero_point)) return 0;
    if (!vx_webgpu_extent(input, &elements, &input_bytes) ||
        !vx_webgpu_extent(output, &out_elements, &output_bytes)) return 0;
    if (vx_webgpu_param_pair(node, VX_NODE_PARAM_KERNEL, VX_NODE_ARRAY_KERNEL,
                             1, &ky, &kx) != 0 ||
        vx_webgpu_param_pair(node, VX_NODE_PARAM_STRIDE, VX_NODE_ARRAY_STRIDE,
                             1, &sy, &sx) != 0 ||
        vx_webgpu_pad_origin(node, &py, &px) != 0) return 0;
    if (ky <= 0 || kx <= 0 || sy <= 0 || sx <= 0) return 0;

    n = input->shape[0]; h = input->shape[1]; w = input->shape[2]; c = input->shape[3];
    out_h = output->shape[1]; out_w = output->shape[2];
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || out_h <= 0 || out_w <= 0) return 0;
    if (output->shape[0] != n || output->shape[3] != c) return 0;

    params[0] = (uint32_t)n;     params[1] = (uint32_t)h;
    params[2] = (uint32_t)w;     params[3] = (uint32_t)c;
    params[4] = (uint32_t)out_h; params[5] = (uint32_t)out_w;
    params[6] = (uint32_t)ky;    params[7] = (uint32_t)kx;
    params[8] = (uint32_t)sy;    params[9] = (uint32_t)sx;
    params[10] = (uint32_t)py;   params[11] = (uint32_t)px;

    params[12] = (uint32_t)input->dtype;
    if (!vx_webgpu_bind(plan, input->data, input_bytes) ||
        !vx_webgpu_bind(plan, output->data, output_bytes) ||
        !vx_webgpu_params(plan, params, packed ? 16u : 12u)) return 0;
    if (packed) {
        plan->variants[0].shader_id = VX_SHADER_INFERENCE_MAX_POOL2_D_TYPED;
        plan->variant_count = 1u;
        return vx_webgpu_linear_1d(out_elements, 256u, plan->variants[0].workgroup);
    }

    plan->variants[0].shader_id = shader_id;
    plan->variants[0].workgroup[0] = ((uint32_t)out_w + 7u) / 8u;
    plan->variants[0].workgroup[1] = ((uint32_t)out_h + 7u) / 8u;
    plan->variants[0].workgroup[2] = (uint32_t)n * (uint32_t)c;
    /* A grid that cannot reach every plane would leave part of the output
     * unwritten, which is a wrong answer rather than a failure. */
    if (plan->variants[0].workgroup[0] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[1] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[2] > VX_WEBGPU_WORKGROUP_LIMIT) return 0;
    plan->variant_count = 1u;
    return 1;
}

static int vx_webgpu_plan_clip(const Node* node, T* input, T* output,
                               VxWebGpuNodePlan* plan) {
    uint32_t size, out_size, params[12];
    size_t in_bytes, out_bytes;
    float lo, hi;
    T* lower = vx_webgpu_port(node, VX_PORT_MIN);
    T* upper = vx_webgpu_port(node, VX_PORT_MAX);
    if (!input || !output || input->dtype != output->dtype) return 0;
    if (input->dtype != T_F32 && input->dtype != T_I32) return 0;
    if (!vx_webgpu_extent(input, &size, &in_bytes) ||
        !vx_webgpu_extent(output, &out_size, &out_bytes) ||
        size != out_size) return 0;
    /* Like the former WebGPU compiler, tensor bounds are immutable scalar
     * weights. The CPU route can also read runtime/computed scalar bounds. */
    T* bounds[2] = {lower, upper};
    for (size_t i = 0; i < 2; ++i) if (bounds[i] &&
        (bounds[i]->dtype != input->dtype || bounds[i]->numel != 1 || !bounds[i]->data ||
         !volvoxai_engine_tensor_is_model_weight_locked(bounds[i]->name))) return 0;
    lo = lower ? lower->data[0] : node->parsed_params.min_val;
    hi = upper ? upper->data[0] : node->parsed_params.max_val;
    memset(params, 0, sizeof(params));
    params[0] = size;
    params[1] = (uint32_t)input->dtype;
    if (input->dtype == T_F32) {
        if (isnan(lo) || isnan(hi) || lo > hi) return 0;
        params[4] = vx_webgpu_f32_bits(lo);
        params[5] = vx_webgpu_f32_bits(hi);
    } else {
        int32_t low = lower ? ((int32_t*)lower->data)[0] : node->parsed_params.min_val_i32;
        int32_t high = upper ? ((int32_t*)upper->data)[0] : node->parsed_params.max_val_i32;
        if (low > high) return 0;
        params[8] = (uint32_t)low;
        params[9] = (uint32_t)high;
    }
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 12u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_CLIP_TYPED;
    if (!vx_webgpu_linear_1d(size, 64u, plan->variants[0].workgroup)) return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * Row-wise operators: LayerNorm and RMSNorm.
 *
 * Both read one row of the last axis per invocation, so they share the same
 * two numbers -- how many rows, how wide. Native computes them the same way,
 * including the axis and width defaults, which is what keeps the two backends
 * normalising over the same elements. The table reaches this helper too.
 */
static int vx_webgpu_rows(const Node* node, const T* tensor,
                          uint32_t* rows, uint32_t* width) {
    int declared;
    long count;
    if (!tensor || tensor->ndim <= 0) return 0;
    declared = vx_node_param_i32(node, VX_NODE_PARAM_D_MODEL,
                                 tensor->shape[tensor->ndim - 1]);
    if (declared <= 0 || tensor->numel % declared != 0) return 0;
    count = tensor->numel / declared;
    if (count <= 0 || count > INT32_MAX) return 0;
    *rows = (uint32_t)count;
    *width = (uint32_t)declared;
    return 1;
}

static int vx_webgpu_plan_feature_norm(const Node* node, T* input, T* output,
                                       VxWebGpuNodePlan* plan, int layer_norm) {
    T* weight = vx_webgpu_port(node, VX_PORT_WEIGHT);
    T* bias = layer_norm ? vx_webgpu_port(node, VX_PORT_BIAS) : NULL;
    uint32_t size, out_size, rows, width, weight_count, bias_count, params[4];
    size_t in_bytes, out_bytes, weight_bytes, bias_bytes = 0;
    float epsilon;
    if (!vx_webgpu_f32_extent(input, &size, &in_bytes) ||
        !vx_webgpu_f32_extent(output, &out_size, &out_bytes) ||
        !vx_webgpu_f32_extent(weight, &weight_count, &weight_bytes) ||
        size != out_size) return 0;
    if (layer_norm &&
        !vx_webgpu_f32_extent(bias, &bias_count, &bias_bytes)) return 0;
    if (!vx_webgpu_rows(node, input, &rows, &width) ||
        width != (uint32_t)input->shape[input->ndim - 1]) return 0;
    if (layer_norm ? (weight_count != width || bias_count != width)
                   : weight_count < width) return 0;
    epsilon = vx_node_param_f32(node, VX_NODE_PARAM_EPS,
                                layer_norm ? 1.0e-6f : 1.0e-6f);
    if (!(epsilon > 0.0f) || epsilon > 1.0e30f) return 0;
    params[0] = rows;
    params[1] = width;
    params[2] = vx_webgpu_f32_bits(epsilon);
    params[3] = 0u;
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, weight->data, weight_bytes)) return 0;
    if (layer_norm && !vx_webgpu_bind(plan, bias->data, bias_bytes)) return 0;
    if (!vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, layer_norm ? 4u : 3u)) return 0;
    plan->variants[0].shader_id = layer_norm ? VX_SHADER_INFERENCE_LAYER_NORM
                                             : VX_SHADER_INFERENCE_R_MS_NORM;
    if (!vx_webgpu_linear_1d(rows, 64u, plan->variants[0].workgroup)) return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * Planning from the table.
 *
 * Every operator whose descriptor is a mapping rather than a decision lives in
 * `webgpu_dispatch.inc`: which shader, which operands in which order, which
 * uniform words, which grid. This walks one row of that table. What it does
 * not decide is admission -- whether the shader is correct for these shapes --
 * because that is a claim, not a transcription, and each claim is a named
 * predicate below.
 */
enum {
    VX_WEBGPU_VALUE_CONST,
    VX_WEBGPU_VALUE_CONST_F32,
    VX_WEBGPU_VALUE_ELEMENTS,
    VX_WEBGPU_VALUE_STORAGE_WORDS,
    VX_WEBGPU_VALUE_LASTDIM,
    VX_WEBGPU_VALUE_ROWS,
    VX_WEBGPU_VALUE_AXIS,
    VX_WEBGPU_VALUE_DTYPE,
    VX_WEBGPU_VALUE_PARAM_I32,
    VX_WEBGPU_VALUE_PARAM_F32,
    VX_WEBGPU_VALUE_INV_LASTDIM,
    VX_WEBGPU_VALUE_GRID_STRIDE,
    VX_WEBGPU_VALUE_APPROX_TANH,
};

enum { VX_WEBGPU_GRID_LINEAR_1D, VX_WEBGPU_GRID_LINEAR_2D };

enum {
    VX_WEBGPU_OPERAND_OUTPUT,
    VX_WEBGPU_OPERAND_INPUT,
    VX_WEBGPU_OPERAND_PORT,
};

enum {
    VX_WEBGPU_ADMIT_EQUAL_ELEMENTS,
    VX_WEBGPU_ADMIT_EQUAL_ELEMENTS_LAST_AXIS,
    VX_WEBGPU_ADMIT_EQUAL_ELEMENTS_DISTINCT,
    VX_WEBGPU_ADMIT_ROW_REDUCTION,
    VX_WEBGPU_ADMIT_SAME_DTYPE_ELEMENTS_DISTINCT,
    VX_WEBGPU_ADMIT_PORTABLE_CAST,
};

typedef struct {
    uint8_t kind;
    uint8_t operand;
    int32_t a;
    uint32_t word;
} VxWebGpuValue;

enum {
    VX_WEBGPU_SIZE_ANY,
    VX_WEBGPU_SIZE_FULL,   /* one element per output element */
    VX_WEBGPU_SIZE_LANE,   /* at least one per output channel */
};

typedef struct {
    uint8_t kind;
    VxPortKind port;
    VxPortKind alias;
    int dtype;          /* zero admits any element type */
    uint8_t sizing;
} VxWebGpuOperand;

typedef struct {
    VxOperatorKind operator_kind;
    uint32_t shader;
    const VxWebGpuOperand* operands;
    uint8_t operand_count;
    const VxWebGpuValue* params;
    uint8_t param_count;
    uint8_t grid;
    VxWebGpuValue grid_value;
    uint16_t divisor;
    uint8_t admit;
} VxWebGpuOperatorPlan;

#include "webgpu_dispatch.inc"

/* One operand, resolved and measured. */
typedef struct {
    T* tensor;
    uint32_t elements;
    size_t bytes;
} VxWebGpuBound;

static uint32_t vx_webgpu_value(const VxWebGpuValue* value,
                                const VxWebGpuBound* bound,
                                const Node* node,
                                uint32_t grid_stride) {
    const T* tensor = bound[value->operand].tensor;
    uint32_t rows, width;
    switch (value->kind) {
        case VX_WEBGPU_VALUE_CONST:
        case VX_WEBGPU_VALUE_CONST_F32:
            return value->word;
        case VX_WEBGPU_VALUE_STORAGE_WORDS:
            return (uint32_t)(bound[value->operand].bytes / 4u + (bound[value->operand].bytes % 4u != 0u));
        case VX_WEBGPU_VALUE_ELEMENTS:
            return bound[value->operand].elements;
        case VX_WEBGPU_VALUE_LASTDIM:
            return (uint32_t)tensor->shape[tensor->ndim - 1];
        case VX_WEBGPU_VALUE_ROWS:
            return vx_webgpu_rows(node, tensor, &rows, &width) ? rows : 0u;
        case VX_WEBGPU_VALUE_AXIS:
            return (uint32_t)tensor->shape[value->a];
        case VX_WEBGPU_VALUE_DTYPE:
            return (uint32_t)tensor->dtype;
        case VX_WEBGPU_VALUE_PARAM_I32:
            return (uint32_t)vx_node_param_i32(node, (VxNodeParamKey)value->a,
                                               (int)value->word);
        case VX_WEBGPU_VALUE_PARAM_F32: {
            float fallback;
            memcpy(&fallback, &value->word, sizeof(fallback));
            return vx_webgpu_f32_bits(
                vx_node_param_f32(node, (VxNodeParamKey)value->a, fallback));
        }
        case VX_WEBGPU_VALUE_INV_LASTDIM:
            return vx_webgpu_f32_bits(
                1.0f / (float)tensor->shape[tensor->ndim - 1]);
        case VX_WEBGPU_VALUE_GRID_STRIDE:
            return grid_stride;
        case VX_WEBGPU_VALUE_APPROX_TANH: {
            const VxCachedNodeParam* cached =
                vx_node_param(node, VX_NODE_PARAM_APPROXIMATE);
            VxNodeParamSymbol symbol;
            if (cached && cached->kind != VX_NODE_PARAM_ABSENT &&
                cached->kind != VX_NODE_PARAM_SYMBOL) return UINT32_MAX;
            symbol = vx_node_param_symbol(node, VX_NODE_PARAM_APPROXIMATE,
                                          VX_NODE_SYMBOL_NONE);
            if (symbol == VX_NODE_SYMBOL_TANH) return 1u;
            return symbol == VX_NODE_SYMBOL_NONE ? 0u : UINT32_MAX;
        }
        default:
            return UINT32_MAX;
    }
}

/*
 * Whether this shader is correct for these shapes.
 *
 * Named rather than spelled out in the table on purpose: a wrong admission
 * does not refuse a node, it runs a shader on a shape it cannot handle and
 * returns numbers. Each of these is small enough to read in one sitting.
 */
static int vx_webgpu_admits(uint8_t admit, const Node* node,
                            const VxWebGpuBound* bound, uint8_t count) {
    const T* first = bound[0].tensor;
    const T* last = bound[count - 1].tensor;
    uint32_t rows, width;
    int axis;
    switch (admit) {
        case VX_WEBGPU_ADMIT_EQUAL_ELEMENTS:
            return bound[0].elements == bound[count - 1].elements;
        case VX_WEBGPU_ADMIT_EQUAL_ELEMENTS_LAST_AXIS:
            if (bound[0].elements != bound[count - 1].elements) return 0;
            axis = vx_node_param_i32(node, VX_NODE_PARAM_AXIS, first->ndim - 1);
            if (axis < 0) axis += first->ndim;
            return axis == first->ndim - 1 &&
                vx_webgpu_rows(node, first, &rows, &width);
        case VX_WEBGPU_ADMIT_EQUAL_ELEMENTS_DISTINCT:
            return bound[0].elements == bound[count - 1].elements &&
                first->data != last->data;
        case VX_WEBGPU_ADMIT_ROW_REDUCTION:
            axis = vx_node_param_i32(node, VX_NODE_PARAM_AXIS, first->ndim - 1);
            if (axis < 0) axis += first->ndim;
            if (axis != first->ndim - 1) return 0;
            width = (uint32_t)first->shape[first->ndim - 1];
            return width > 0u && bound[0].elements ==
                bound[count - 1].elements * width;
        case VX_WEBGPU_ADMIT_SAME_DTYPE_ELEMENTS_DISTINCT:
            return first->dtype == last->dtype && (first->elem_size == 4u || first->elem_size == 1u) &&
                bound[0].elements == bound[count - 1].elements &&
                first->data != last->data;
        case VX_WEBGPU_ADMIT_PORTABLE_CAST:
            return (first->dtype == T_F32 || first->dtype == T_I32 || first->dtype == T_I8 || first->dtype == T_U8) &&
                (last->dtype == T_F32 || last->dtype == T_I32 || last->dtype == T_I8 || last->dtype == T_U8) &&
                bound[0].elements == bound[count - 1].elements;
        default:
            return 0;
    }
}

static const VxWebGpuOperatorPlan* vx_webgpu_operator(VxOperatorKind op) {
    size_t index;
    for (index = 0; index < VX_WEBGPU_OPERATOR_COUNT; index++)
        if (vx_webgpu_operator_table[index].operator_kind == op)
            return &vx_webgpu_operator_table[index];
    return NULL;
}

static int vx_webgpu_plan_tabled(const VxWebGpuOperatorPlan* entry,
                                 const Node* node, T* input, T* output,
                                 VxWebGpuNodePlan* plan) {
    VxWebGpuBound bound[VX_WEBGPU_MAX_BINDINGS];
    uint32_t words[VX_WEBGPU_MAX_PARAM_BYTES / sizeof(uint32_t)];
    uint32_t grid_stride, extent;
    uint8_t index;
    if (entry->operand_count > VX_WEBGPU_MAX_BINDINGS) return 0;
    for (index = 0; index < entry->operand_count; index++) {
        const VxWebGpuOperand* declared = &entry->operands[index];
        T* tensor = declared->kind == VX_WEBGPU_OPERAND_OUTPUT ? output
                  : declared->kind == VX_WEBGPU_OPERAND_INPUT ? input
                  : vx_webgpu_port(node, declared->port);
        if (!tensor && declared->kind == VX_WEBGPU_OPERAND_PORT && declared->alias)
            tensor = vx_webgpu_port(node, declared->alias);
        if (!tensor) return 0;
        if (declared->dtype && tensor->dtype != declared->dtype) return 0;
        if (!vx_webgpu_extent(tensor, &bound[index].elements, &bound[index].bytes))
            return 0;
        bound[index].tensor = tensor;
    }
    /* Sizing is checked once the output is known, because both relations are
     * stated against it. A shader indexes what it is given without bounds. */
    for (index = 0; index < entry->operand_count; index++) {
        const T* out = bound[entry->operand_count - 1].tensor;
        uint32_t lane = out->ndim > 0 ? (uint32_t)out->shape[out->ndim - 1] : 1u;
        if (entry->operands[index].sizing == VX_WEBGPU_SIZE_FULL &&
            bound[index].elements != bound[entry->operand_count - 1].elements)
            return 0;
        if (entry->operands[index].sizing == VX_WEBGPU_SIZE_LANE &&
            bound[index].elements < lane) return 0;
    }
    if (!vx_webgpu_admits(entry->admit, node, bound, entry->operand_count))
        return 0;

    /* The grid first: a uniform word may be the stride the grid implies. */
    extent = vx_webgpu_value(&entry->grid_value, bound, node, 0u);
    if (extent == UINT32_MAX) return 0;
    int packed_copy = entry->shader == VX_SHADER_INFERENCE_COPY32 && input && input->elem_size == 1u;
    if (packed_copy) extent = extent / 4u + (extent % 4u != 0u);
    if (entry->grid == VX_WEBGPU_GRID_LINEAR_1D) {
        if (!vx_webgpu_linear_1d(extent, entry->divisor,
                                 plan->variants[0].workgroup)) return 0;
    } else {
        vx_webgpu_linear_grid(extent, entry->divisor, plan->variants[0].workgroup);
    }
    grid_stride = plan->variants[0].workgroup[0] * entry->divisor;

    if ((size_t)entry->param_count * sizeof(uint32_t) > sizeof(words)) return 0;
    for (index = 0; index < entry->param_count; index++) {
        words[index] = vx_webgpu_value(&entry->params[index], bound, node,
                                       grid_stride);
        if (words[index] == UINT32_MAX) return 0;
    }
    for (index = 0; index < entry->operand_count; index++)
        if (!vx_webgpu_bind(plan, bound[index].tensor->data, bound[index].bytes))
            return 0;
    if (!vx_webgpu_params(plan, words, entry->param_count)) return 0;
    plan->variants[0].shader_id = packed_copy ? VX_SHADER_INFERENCE_COPY_TYPED : entry->shader;
    plan->variant_count = 1u;
    return 1;
}

static int vx_webgpu_plan_global_average_pool(T* input, T* output,
                                              VxWebGpuNodePlan* plan) {
    uint32_t size, out_size, params[4];
    size_t in_bytes, out_bytes;
    if (!vx_webgpu_f32_extent(input, &size, &in_bytes) ||
        !vx_webgpu_f32_extent(output, &out_size, &out_bytes)) return 0;
    if (input->ndim != 4) return 0;
    params[0] = (uint32_t)input->shape[0];
    params[1] = (uint32_t)input->shape[1];
    params[2] = (uint32_t)input->shape[2];
    params[3] = (uint32_t)input->shape[3];
    if (out_size != params[0] * params[3]) return 0;
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 4u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_GLOBAL_AVERAGE_POOL;
    plan->variants[0].workgroup[0] = (params[3] + 63u) / 64u;
    plan->variants[0].workgroup[1] = params[0];
    plan->variants[0].workgroup[2] = 1u;
    if (plan->variants[0].workgroup[0] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[1] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[1] == 0u) return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * Where and Mask.
 *
 * The shader reads the condition as raw words and is told which dtype they
 * are, because the two spell "nonzero" differently: negative float zero is
 * false, while the same bit pattern as an integer is true.
 */
static int vx_webgpu_plan_where(const Node* node, T* output,
                                VxWebGpuNodePlan* plan) {
    T* condition = vx_webgpu_port(node, VX_PORT_CONDITION);
    T* a = vx_webgpu_port(node, VX_PORT_X);
    T* b = vx_webgpu_port(node, VX_PORT_Y);
    uint32_t size, count, params[4];
    size_t out_bytes, condition_bytes, a_bytes, b_bytes;
    if (!condition) condition = vx_webgpu_port(node, VX_PORT_COND);
    if (!condition) condition = vx_webgpu_port(node, VX_PORT_MASK);
    if (!a) a = vx_webgpu_port(node, VX_PORT_A);
    if (!b) b = vx_webgpu_port(node, VX_PORT_B);
    if (!condition || !a || !b || !output) return 0;
    if (condition->dtype != T_F32 && condition->dtype != T_I32) return 0;
    if (a->dtype != b->dtype || a->dtype != output->dtype) return 0;
    if (a->elem_size != 4u || condition->elem_size != 4u) return 0;
    if (!vx_webgpu_extent(output, &size, &out_bytes) ||
        !vx_webgpu_extent(condition, &count, &condition_bytes) || count != size ||
        !vx_webgpu_extent(a, &count, &a_bytes) || count != size ||
        !vx_webgpu_extent(b, &count, &b_bytes) || count != size) return 0;
    params[0] = size;
    params[1] = (uint32_t)condition->dtype;
    params[2] = 0u;
    params[3] = 0u;
    if (!vx_webgpu_bind(plan, condition->data, condition_bytes) ||
        !vx_webgpu_bind(plan, a->data, a_bytes) ||
        !vx_webgpu_bind(plan, b->data, b_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 4u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_WHERE_TYPED;
    if (!vx_webgpu_linear_1d(size, 64u, plan->variants[0].workgroup)) return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * Expand and Broadcast.
 *
 * Both shapes travel in the uniform as two vec4 pairs, so rank is capped at
 * eight and nothing needs the scratch pool. The shader derives its own strides
 * from the shapes it is given.
 */
static int vx_webgpu_plan_expand(T* input, T* output, VxWebGpuNodePlan* plan) {
    uint32_t size, in_size, params[20];
    size_t in_bytes, out_bytes;
    int axis;
    if (!input || !output || input->dtype != output->dtype) return 0;
    int packed = input->dtype == T_I8 || input->dtype == T_U8;
    if (!packed && input->dtype != T_F32 && input->dtype != T_I32) return 0;
    if (packed && (!input->quantization.valid || !output->quantization.valid ||
        input->quantization.scale != output->quantization.scale ||
        input->quantization.zero_point != output->quantization.zero_point)) return 0;
    if (input->ndim <= 0 || input->ndim > 8 ||
        output->ndim <= 0 || output->ndim > 8 ||
        input->ndim > output->ndim) return 0;
    if (!vx_webgpu_extent(input, &in_size, &in_bytes) ||
        !vx_webgpu_extent(output, &size, &out_bytes)) return 0;
    memset(params, 0, sizeof(params));
    params[0] = (uint32_t)input->ndim;
    params[1] = (uint32_t)output->ndim;
    params[3] = size;
    for (axis = 0; axis < input->ndim; axis++) {
        if (input->shape[axis] <= 0) return 0;
        params[4 + axis] = (uint32_t)input->shape[axis];
    }
    for (axis = 0; axis < output->ndim; axis++) {
        if (output->shape[axis] <= 0) return 0;
        params[12 + axis] = (uint32_t)output->shape[axis];
    }
    plan->variants[0].shader_id = packed ? VX_SHADER_INFERENCE_EXPAND_TYPED : VX_SHADER_INFERENCE_EXPAND;
    vx_webgpu_linear_grid(size, packed ? 256u : 64u, plan->variants[0].workgroup);
    params[2] = plan->variants[0].workgroup[0] * 64u;
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 20u)) return 0;
    plan->variant_count = 1u;
    return 1;
}

/* BatchNorm2D over NHWC: five reads, one write, statistics per channel. */
static int vx_webgpu_plan_batch_norm2d(const Node* node, T* input, T* output,
                                       VxWebGpuNodePlan* plan) {
    T* weight = vx_webgpu_port(node, VX_PORT_WEIGHT);
    T* bias = vx_webgpu_port(node, VX_PORT_BIAS);
    T* mean = vx_webgpu_port(node, VX_PORT_RUNNING_MEAN);
    T* variance = vx_webgpu_port(node, VX_PORT_RUNNING_VAR);
    uint32_t size, out_size, count, params[5], channels;
    size_t in_bytes, out_bytes, weight_bytes, bias_bytes, mean_bytes, var_bytes;
    float epsilon;
    if (!input || !output || input->ndim != 4 || output->ndim != 4) return 0;
    if (!vx_webgpu_f32_extent(input, &size, &in_bytes) ||
        !vx_webgpu_f32_extent(output, &out_size, &out_bytes) ||
        size != out_size) return 0;
    channels = (uint32_t)input->shape[3];
    if (!vx_webgpu_f32_extent(weight, &count, &weight_bytes) || count < channels ||
        !vx_webgpu_f32_extent(bias, &count, &bias_bytes) || count < channels ||
        !vx_webgpu_f32_extent(mean, &count, &mean_bytes) || count < channels ||
        !vx_webgpu_f32_extent(variance, &count, &var_bytes) || count < channels)
        return 0;
    epsilon = vx_node_param_f32(node, VX_NODE_PARAM_EPS, 1.0e-5f);
    if (!(epsilon > 0.0f) || epsilon > 1.0e30f) return 0;
    params[0] = (uint32_t)input->shape[0];
    params[1] = channels;
    params[2] = (uint32_t)input->shape[1];
    params[3] = (uint32_t)input->shape[2];
    params[4] = vx_webgpu_f32_bits(epsilon);
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, weight->data, weight_bytes) ||
        !vx_webgpu_bind(plan, bias->data, bias_bytes) ||
        !vx_webgpu_bind(plan, mean->data, mean_bytes) ||
        !vx_webgpu_bind(plan, variance->data, var_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 5u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_BATCH_NORM2_D;
    if (!vx_webgpu_linear_1d(size, 64u, plan->variants[0].workgroup)) return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * GroupNorm over NHWC.
 *
 * The shader binds a bias whether or not one exists, so a graph without one
 * would need a buffer of zeros this backend does not own; native asks for the
 * same thing, and both decline instead of inventing it.
 */
static int vx_webgpu_plan_group_norm(const Node* node, T* input, T* output,
                                     VxWebGpuNodePlan* plan) {
    T* weight = vx_webgpu_port(node, VX_PORT_WEIGHT);
    T* bias = vx_webgpu_port(node, VX_PORT_BIAS);
    uint32_t size, out_size, count, params[8], channels;
    size_t in_bytes, out_bytes, weight_bytes, bias_bytes;
    int groups = node->parsed_params.num_groups;
    float epsilon = node->parsed_params.eps;
    if (!weight) weight = vx_webgpu_port(node, VX_PORT_SCALE);
    if (!input || !output || input->ndim != 4) return 0;
    if (!vx_webgpu_f32_extent(input, &size, &in_bytes) ||
        !vx_webgpu_f32_extent(output, &out_size, &out_bytes) ||
        size != out_size) return 0;
    channels = (uint32_t)input->shape[3];
    if (groups <= 0 || channels == 0u || channels % (uint32_t)groups != 0) return 0;
    if (!vx_webgpu_f32_extent(weight, &count, &weight_bytes) || count < channels ||
        !vx_webgpu_f32_extent(bias, &count, &bias_bytes) || count < channels)
        return 0;
    if (!(epsilon > 0.0f) || epsilon > 1.0e30f) return 0;
    params[0] = (uint32_t)input->shape[0];
    params[1] = (uint32_t)input->shape[1];
    params[2] = (uint32_t)input->shape[2];
    params[3] = channels;
    params[4] = (uint32_t)groups;
    params[5] = 1u;
    params[6] = vx_webgpu_f32_bits(epsilon);
    params[7] = 0u;
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, weight->data, weight_bytes) ||
        !vx_webgpu_bind(plan, bias->data, bias_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 8u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_GROUP_NORM;
    plan->variants[0].workgroup[0] = params[0] * (uint32_t)groups;
    plan->variants[0].workgroup[1] = 1u;
    plan->variants[0].workgroup[2] = 1u;
    if (plan->variants[0].workgroup[0] == 0u ||
        plan->variants[0].workgroup[0] > VX_WEBGPU_WORKGROUP_LIMIT) return 0;
    plan->variant_count = 1u;
    return 1;
}

/* UpsampleNearest2D, which the shader fixes at exactly two. */
static int vx_webgpu_plan_upsample2x(T* input, T* output,
                                     VxWebGpuNodePlan* plan) {
    uint32_t size, out_size, params[4];
    size_t in_bytes, out_bytes;
    if (!input || !output || input->ndim != 4 || output->ndim != 4) return 0;
    if (!vx_webgpu_f32_extent(input, &size, &in_bytes) ||
        !vx_webgpu_f32_extent(output, &out_size, &out_bytes)) return 0;
    if (output->shape[1] != input->shape[1] * 2 ||
        output->shape[2] != input->shape[2] * 2 ||
        output->shape[0] != input->shape[0] ||
        output->shape[3] != input->shape[3]) return 0;
    params[0] = (uint32_t)input->shape[0];
    params[1] = (uint32_t)input->shape[1];
    params[2] = (uint32_t)input->shape[2];
    params[3] = (uint32_t)input->shape[3];
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 4u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_UPSAMPLE2X;
    plan->variants[0].workgroup[0] = ((uint32_t)output->shape[2] + 7u) / 8u;
    plan->variants[0].workgroup[1] = ((uint32_t)output->shape[1] + 7u) / 8u;
    plan->variants[0].workgroup[2] = params[0] * params[3];
    if (plan->variants[0].workgroup[0] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[1] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[2] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[2] == 0u) return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * ArgMax over any one axis.
 *
 * The axis splits the tensor into an outer count, the axis itself and an inner
 * count, which is exactly what the shader walks. `select_last_index` changes
 * which of two equal maxima wins, and the shader implements only the first.
 */
static int vx_webgpu_plan_argmax(const Node* node, T* input, T* output,
                                 VxWebGpuNodePlan* plan) {
    uint32_t out_size, params[4];
    size_t in_bytes, out_bytes, unused_bytes;
    uint32_t size;
    uint64_t outer = 1u, inner = 1u;
    int axis, dimension;
    if (!input || !output || input->dtype != T_F32 || output->dtype != T_I32)
        return 0;
    if (input->ndim <= 0 || input->ndim > 8 || input->data == output->data) return 0;
    if (!vx_webgpu_extent(input, &size, &in_bytes) ||
        !vx_webgpu_extent(output, &out_size, &out_bytes)) return 0;
    (void)unused_bytes;
    axis = node->parsed_params.axis;
    if (axis < 0) axis += input->ndim;
    if (axis < 0 || axis >= input->ndim || input->shape[axis] <= 0) return 0;
    if (node->parsed_params.select_last_index != 0) return 0;
    if (node->parsed_params.keepdims != 0 && node->parsed_params.keepdims != 1)
        return 0;
    for (dimension = 0; dimension < axis; dimension++) {
        if (input->shape[dimension] <= 0) return 0;
        outer *= (uint64_t)input->shape[dimension];
    }
    for (dimension = axis + 1; dimension < input->ndim; dimension++) {
        if (input->shape[dimension] <= 0) return 0;
        inner *= (uint64_t)input->shape[dimension];
    }
    if (outer > UINT32_MAX || inner > UINT32_MAX) return 0;
    if ((uint64_t)out_size != outer * inner) return 0;
    params[0] = (uint32_t)outer;
    params[1] = (uint32_t)input->shape[axis];
    params[2] = (uint32_t)inner;
    params[3] = out_size;
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 4u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_ARG_MAX_F32_I32_NATIVE;
    if (!vx_webgpu_linear_1d(out_size, 64u, plan->variants[0].workgroup)) return 0;
    plan->variant_count = 1u;
    return 1;
}

/* A node parameter that must be absent or false, the way native reads it. */
static int vx_webgpu_absent_or_false(const Node* node, VxNodeParamKey key) {
    const VxCachedNodeParam* cached = vx_node_param(node, key);
    if (!cached || cached->kind == VX_NODE_PARAM_ABSENT) return 1;
    return vx_node_param_i32(node, key, 0) == 0;
}

/*
 * Resize and ResizeNearest2D over NHWC.
 *
 * The shader implements one interpretation of each mode -- nearest with an
 * asymmetric transform and a floor, linear with half-pixel centres -- so a
 * node that asks for any other spelling is declined rather than approximated.
 * Native applies the same test before it dispatches.
 */
static int vx_webgpu_plan_resize(const Node* node, T* input, T* output,
                                 VxWebGpuNodePlan* plan) {
    uint32_t size, out_size, params[8] = {0};
    size_t in_bytes, out_bytes;
    int nearest = 1;
    if (!input || !output || input->ndim != 4 || output->ndim != 4) return 0;
    int packed = input->dtype == T_I8 || input->dtype == T_U8;
    if (input->dtype != output->dtype || (!packed && input->dtype != T_F32)) return 0;
    if (packed && (!input->quantization.valid || !output->quantization.valid ||
        input->quantization.scale != output->quantization.scale ||
        input->quantization.zero_point != output->quantization.zero_point)) return 0;
    if (!vx_webgpu_extent(input, &size, &in_bytes) ||
        !vx_webgpu_extent(output, &out_size, &out_bytes)) return 0;
    if (output->shape[0] != input->shape[0] ||
        output->shape[3] != input->shape[3]) return 0;
    if (node->operator_kind == VX_OP_RESIZE) {
        VxNodeParamSymbol mode = vx_node_param_symbol(node, VX_NODE_PARAM_MODE,
                                                      VX_NODE_SYMBOL_LINEAR);
        VxNodeParamSymbol transform, nearest_mode, layout;
        const VxCachedNodeParam* legacy;
        if (mode == VX_NODE_SYMBOL_LINEAR) nearest = 0;
        else if (mode == VX_NODE_SYMBOL_NEAREST) nearest = 1;
        else return 0;
        legacy = vx_node_param(node, VX_NODE_PARAM_COORDINATE_TRANSFORM_MODE);
        if (legacy && legacy->kind != VX_NODE_PARAM_ABSENT) return 0;
        transform = vx_node_param_symbol(
            node, VX_NODE_PARAM_COORDINATE_TRANSFORMATION_MODE,
            nearest ? VX_NODE_SYMBOL_ASYMMETRIC : VX_NODE_SYMBOL_HALF_PIXEL);
        if (transform != (nearest ? VX_NODE_SYMBOL_ASYMMETRIC
                                  : VX_NODE_SYMBOL_HALF_PIXEL)) return 0;
        nearest_mode = vx_node_param_symbol(node, VX_NODE_PARAM_NEAREST_MODE,
                                            VX_NODE_SYMBOL_FLOOR);
        if (nearest_mode != VX_NODE_SYMBOL_FLOOR ||
            (!nearest && vx_node_param_has(node, VX_NODE_PARAM_NEAREST_MODE)))
            return 0;
        layout = vx_node_param_symbol(node, VX_NODE_PARAM_DATA_LAYOUT,
                                      VX_NODE_SYMBOL_NHWC);
        if (layout != VX_NODE_SYMBOL_NHWC) return 0;
        if (!vx_webgpu_absent_or_false(node, VX_NODE_PARAM_ALIGN_CORNERS) ||
            !vx_webgpu_absent_or_false(node, VX_NODE_PARAM_ANTIALIAS)) return 0;
    }
    if (packed && !nearest) return 0;
    params[0] = (uint32_t)input->shape[0];
    params[1] = (uint32_t)input->shape[1];
    params[2] = (uint32_t)input->shape[2];
    params[3] = (uint32_t)input->shape[3];
    params[4] = (uint32_t)output->shape[1];
    params[5] = (uint32_t)output->shape[2];
    params[6] = nearest ? 0u : 1u;
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, packed ? 8u : 7u)) return 0;
    if (packed) {
        plan->variants[0].shader_id = VX_SHADER_INFERENCE_RESIZE_NEAREST_TYPED;
        plan->variant_count = 1u;
        return vx_webgpu_linear_1d(out_size, 256u, plan->variants[0].workgroup);
    }
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_RESIZE;
    plan->variants[0].workgroup[0] = (params[5] + 7u) / 8u;
    plan->variants[0].workgroup[1] = (params[4] + 7u) / 8u;
    plan->variants[0].workgroup[2] = params[0] * params[3];
    if (plan->variants[0].workgroup[0] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[1] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[2] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[2] == 0u) return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * Scaled dot-product attention over a packed QKV activation.
 *
 * The mask's layout is not this backend's to decide: `attention_mask_mode`
 * is the engine's own rule for which of the five spellings a mask tensor is,
 * and reading it here rather than restating it keeps one answer. A node with
 * no mask still binds a buffer, because the shader declares one -- and reads
 * it only when the mode says to.
 */
static int vx_webgpu_plan_sdpa(const Node* node, T* input, T* output,
                               VxWebGpuNodePlan* plan) {
    T* qkv = vx_webgpu_port(node, VX_PORT_QKV);
    T* mask = vx_webgpu_port(node, VX_PORT_MASK);
    uint32_t qkv_elements, out_elements, mask_elements, params[12] = {0};
    size_t param_count = 8u;
    uint32_t shader = VX_SHADER_INFERENCE_S_DPA;
    size_t qkv_bytes, out_bytes, mask_bytes;
    int d_model, batch, seq_len, heads, head_dim, mask_mode;
    float scale;
    if (!qkv) qkv = input;
    if (!qkv || !output) return 0;
    if ((qkv->ndim != 2 && qkv->ndim != 3) || output->ndim != qkv->ndim) return 0;
    if (!vx_webgpu_f32_extent(qkv, &qkv_elements, &qkv_bytes) ||
        !vx_webgpu_f32_extent(output, &out_elements, &out_bytes)) return 0;
    d_model = output->shape[output->ndim - 1];
    if (d_model <= 0 || qkv->shape[qkv->ndim - 1] != d_model * 3) return 0;
    batch = qkv->ndim == 3 ? qkv->shape[0] : 1;
    seq_len = qkv->shape[qkv->ndim - 2];
    heads = node->parsed_params.heads;
    if (heads <= 0 || d_model % heads) return 0;
    head_dim = d_model / heads;
    if (batch <= 0 || seq_len <= 0 || head_dim > 64) return 0;
    if ((output->ndim == 3 ? output->shape[0] : 1) != batch) return 0;
    if (output->shape[output->ndim - 2] != seq_len) return 0;
    if ((uint64_t)qkv_elements != (uint64_t)batch * seq_len * 3 * d_model) return 0;
    if ((uint64_t)out_elements != (uint64_t)batch * seq_len * d_model) return 0;
    mask_mode = attention_mask_mode(mask, batch, seq_len, seq_len);
    if (mask_mode < 0) return 0;
    scale = node->parsed_params.has_scale
        ? node->parsed_params.scale : 1.0f / sqrtf((float)head_dim);
    params[0] = (uint32_t)seq_len;
    params[1] = (uint32_t)d_model;
    params[2] = (uint32_t)heads;
    params[3] = (uint32_t)head_dim;
    params[4] = (uint32_t)batch;
    params[5] = vx_webgpu_f32_bits(scale);
    params[6] = node->parsed_params.causal ? 1u : 0u;
    params[7] = (uint32_t)mask_mode;
    if (!vx_webgpu_bind(plan, qkv->data, qkv_bytes)) return 0;
    if (mask_mode != 0 && mask) {
        if (!vx_webgpu_extent(mask, &mask_elements, &mask_bytes) ||
            mask->dtype != T_I32 ||
            !vx_webgpu_bind(plan, mask->data, mask_bytes)) return 0;
    } else {
        /* No mask to read, but the shader declares the binding. */
        uint32_t* slot = vx_webgpu_scratch();
        if (!slot) return 0;
        memset(slot, 0, VX_WEBGPU_SCRATCH_WORDS * sizeof(*slot));
        if (!vx_webgpu_bind(plan, slot,
                            VX_WEBGPU_SCRATCH_WORDS * sizeof(*slot))) return 0;
    }
#if VOLVOXAI_ENABLE_TRAINING
    if (g_native_training_mode) {
        if (vx_webgpu_training_rng((int)(node - g_n), 1, params + 8)) return 0;
        param_count = 12u;
        shader = VX_SHADER_TRAINING_SDPA_TRAINING;
    }
#endif
    if (!vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, param_count)) return 0;
    plan->variants[0].shader_id = shader;
    /* One invocation per (query, head, batch): the shader reads all three. */
    plan->variants[0].workgroup[0] = ((uint32_t)seq_len + 63u) / 64u;
    plan->variants[0].workgroup[1] = (uint32_t)heads;
    plan->variants[0].workgroup[2] = (uint32_t)batch;
    if (plan->variants[0].workgroup[0] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[1] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[2] > VX_WEBGPU_WORKGROUP_LIMIT) return 0;
    plan->variant_count = 1u;
    return 1;
}

/* QArgMax: the float ArgMax's axis decomposition over byte storage. */
static int vx_webgpu_plan_qargmax(const Node* node, T* input, T* output,
                                  VxWebGpuNodePlan* plan) {
    uint32_t elements, out_elements, params[4];
    size_t in_bytes, out_bytes;
    uint64_t outer = 1u, inner = 1u;
    int axis, dimension;
    if (!input || !output || output->dtype != T_I32) return 0;
    if (input->dtype != T_I8 && input->dtype != T_U8) return 0;
    if (input->ndim <= 0 || input->ndim > 8) return 0;
    if (!vx_webgpu_extent(input, &elements, &in_bytes) ||
        !vx_webgpu_extent(output, &out_elements, &out_bytes)) return 0;
    axis = node->parsed_params.axis;
    if (axis < 0) axis += input->ndim;
    if (axis < 0 || axis >= input->ndim || input->shape[axis] <= 0) return 0;
    if (node->parsed_params.select_last_index != 0) return 0;
    if (node->parsed_params.keepdims != 0 && node->parsed_params.keepdims != 1)
        return 0;
    for (dimension = 0; dimension < axis; dimension++) {
        if (input->shape[dimension] <= 0) return 0;
        outer *= (uint64_t)input->shape[dimension];
    }
    for (dimension = axis + 1; dimension < input->ndim; dimension++) {
        if (input->shape[dimension] <= 0) return 0;
        inner *= (uint64_t)input->shape[dimension];
    }
    if (outer > UINT32_MAX || inner > UINT32_MAX) return 0;
    if ((uint64_t)out_elements != outer * inner) return 0;
    params[0] = (uint32_t)outer;
    params[1] = (uint32_t)input->shape[axis];
    params[2] = (uint32_t)inner;
    params[3] = (uint32_t)input->dtype;
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 4u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_Q_ARG_MAX_INT8;
    if (!vx_webgpu_linear_1d(out_elements, 64u, plan->variants[0].workgroup))
        return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * QConv2D over NHWC with an OHWI weight.
 *
 * Every extent, padding and stride is already in the node's parsed metadata,
 * which is what the engine validated the graph against, so this reads them
 * rather than deriving them again. The per-channel scales, zero points and
 * integer bias are arrays the engine holds in linear memory and are bound
 * where they lie; the shader forms its own requantization multiplier from
 * them, so nothing has to be precomputed here.
 */
static int vx_webgpu_plan_qconv2d(const Node* node, T* output,
                                  VxWebGpuNodePlan* plan) {
    const QConv2DMetadata* meta;
    T* input = vx_webgpu_port(node, VX_PORT_INPUT);
    T* weight = vx_webgpu_port(node, VX_PORT_WEIGHT);
    uint32_t in_elements, weight_elements, out_elements, params[28];
    size_t in_bytes, weight_bytes, out_bytes;
    const int32_t* bias;
    int index = g_vx_runtime_node.idx;
    if (index < 0 || (size_t)index >= g_node_capacity || !g_qconv_meta) return 0;
    meta = &g_qconv_meta[index];
    if (!meta->valid || !meta->weight_scales || !meta->weight_zero_points ||
        meta->output_channels == 0u) return 0;
    if (!input) input = vx_webgpu_port(node, VX_PORT_X);
    if (!input || !weight || !output) return 0;
    if (input->dtype != meta->input_dtype || weight->dtype != meta->weight_dtype ||
        output->dtype != meta->output_dtype) return 0;
    if (!input->quantization.valid || !output->quantization.valid) return 0;
    if (input->ndim != 4 || weight->ndim != 4 || output->ndim != 4) return 0;
    if (!vx_webgpu_extent(input, &in_elements, &in_bytes) ||
        !vx_webgpu_extent(weight, &weight_elements, &weight_bytes) ||
        !vx_webgpu_extent(output, &out_elements, &out_bytes)) return 0;
    if ((uint32_t)input->shape[3] != meta->input_channels ||
        (uint32_t)output->shape[3] != meta->output_channels) return 0;
    if (input->shape[0] != output->shape[0]) return 0;
    /*
     * The weight is OHWI with its last extent per group, and the shader
     * derives that extent by dividing rather than reading it. Refuse unless
     * the division agrees with the metadata, because a disagreement is not a
     * wrong answer -- it is the shader indexing a different weight.
     */
    if (meta->groups == 0u || meta->input_per_group == 0u ||
        meta->input_per_group * meta->groups != meta->input_channels ||
        meta->output_channels % meta->groups != 0u) return 0;
    if ((uint32_t)weight->shape[0] != meta->output_channels ||
        (uint32_t)weight->shape[1] != meta->kernel_height ||
        (uint32_t)weight->shape[2] != meta->kernel_width ||
        (uint32_t)weight->shape[3] != meta->input_per_group) return 0;
    if (!meta->stride_y || !meta->stride_x ||
        !meta->dilation_y || !meta->dilation_x ||
        !meta->kernel_height || !meta->kernel_width) return 0;
    /* Activation geometry follows the current binding, including the maximum
     * descriptor used by domain proof. Pack/affine metadata stays invariant. */
    uint64_t effective_height = (uint64_t)(meta->kernel_height - 1u) * meta->dilation_y + 1u;
    uint64_t effective_width = (uint64_t)(meta->kernel_width - 1u) * meta->dilation_x + 1u;
    uint64_t padded_height = (uint64_t)(uint32_t)input->shape[1] + meta->padding_top + meta->padding_bottom;
    uint64_t padded_width = (uint64_t)(uint32_t)input->shape[2] + meta->padding_left + meta->padding_right;
    if (padded_height < effective_height || padded_width < effective_width ||
        (padded_height - effective_height) / meta->stride_y + 1u != (uint32_t)output->shape[1] ||
        (padded_width - effective_width) / meta->stride_x + 1u != (uint32_t)output->shape[2]) return 0;
    bias = meta->bias;
    if (!bias) {
        if ((size_t)meta->output_channels > SIZE_MAX / sizeof(int32_t)) return 0;
        size_t bytes = (size_t)meta->output_channels * sizeof(int32_t);
        int32_t* zeros = vx_webgpu_arena(bytes);
        if (!zeros) return 0;
        memset(zeros, 0, bytes);
        bias = zeros;
    }

    memset(params, 0, sizeof(params));
    params[0] = (uint32_t)input->shape[0];
    params[1] = (uint32_t)input->shape[1];
    params[2] = (uint32_t)input->shape[2];
    params[3] = meta->input_channels;
    params[4] = (uint32_t)output->shape[1];
    params[5] = (uint32_t)output->shape[2];
    params[6] = meta->output_channels;
    params[7] = meta->kernel_height;
    params[8] = meta->kernel_width;
    params[9] = meta->stride_y;
    params[10] = meta->stride_x;
    params[11] = meta->dilation_y;
    params[12] = meta->dilation_x;
    params[13] = meta->padding_top;
    params[14] = meta->padding_left;
    params[15] = meta->groups;
    params[16] = (uint32_t)input->dtype;
    params[17] = (uint32_t)weight->dtype;
    params[18] = (uint32_t)output->dtype;
    params[19] = meta->relu;
    params[20] = (uint32_t)input->quantization.zero_point;
    params[21] = (uint32_t)output->quantization.zero_point;
    params[24] = vx_webgpu_f32_bits(input->quantization.scale);
    params[25] = vx_webgpu_f32_bits(output->quantization.scale);

    plan->variants[0].shader_id = VX_SHADER_INFERENCE_Q_CONV2_D_INT8;
    /* One invocation per packed output word, spread over a two-axis grid. */
    vx_webgpu_linear_grid(out_elements, 256u, plan->variants[0].workgroup);
    params[22] = plan->variants[0].workgroup[0] * 64u;

    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, weight->data, weight_bytes) ||
        !vx_webgpu_bind(plan, meta->weight_scales,
                        (size_t)meta->output_channels * sizeof(float)) ||
        !vx_webgpu_bind(plan, meta->weight_zero_points,
                        (size_t)meta->output_channels * sizeof(int32_t)) ||
        !vx_webgpu_bind(plan, bias,
                        (size_t)meta->output_channels * sizeof(int32_t)) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 28u)) return 0;
    plan->variant_count = 1u;
    return 1;
}

/* Matrix batch axes broadcast after right-aligning operand ranks. */
static int vx_webgpu_batch_dimension(const T* tensor, int output_rank, int axis) {
    int source_axis = axis - (output_rank - tensor->ndim);
    return source_axis < 0 ? 1 : tensor->shape[source_axis];
}

/*
 * QBatchMatMul: the float batch matrix multiply over byte operands.
 *
 * The shape rules are the float route's -- last two axes are the matrix, the
 * rest a broadcast batch grid -- and the only additions are the three affine
 * descriptors. The strides travel in the arena as they do there, except that
 * this shader keeps its counts in the uniform, so the buffer is strides alone.
 */
static int vx_webgpu_plan_qbatch_matmul(const Node* node, T* output,
                                        VxWebGpuNodePlan* plan) {
    T* a = vx_webgpu_port(node, VX_PORT_A);
    T* b = vx_webgpu_port(node, VX_PORT_B);
    uint32_t words[VX_WEBGPU_SCRATCH_WORDS];
    uint32_t a_elements, b_elements, out_elements, params[16];
    size_t a_bytes, b_bytes, out_bytes, stride_bytes;
    uint32_t m, k, n, batches = 1u, stride = 1u;
    uint32_t* strides;
    int rank, batch_rank, axis;
    if (!a || !b || !output) return 0;
    if ((a->dtype != T_I8 && a->dtype != T_U8) ||
        (b->dtype != T_I8 && b->dtype != T_U8) ||
        (output->dtype != T_I8 && output->dtype != T_U8)) return 0;
    if (!a->quantization.valid || !b->quantization.valid ||
        !output->quantization.valid) return 0;
    if (!vx_webgpu_extent(a, &a_elements, &a_bytes) ||
        !vx_webgpu_extent(b, &b_elements, &b_bytes) ||
        !vx_webgpu_extent(output, &out_elements, &out_bytes)) return 0;
    rank = output->ndim;
    if (rank < 2 || rank > 8 || a->ndim < 2 || b->ndim < 2 ||
        rank != (a->ndim > b->ndim ? a->ndim : b->ndim)) return 0;
    batch_rank = rank - 2;
    m = (uint32_t)output->shape[rank - 2];
    n = (uint32_t)output->shape[rank - 1];
    k = (uint32_t)a->shape[a->ndim - 1];
    if (m == 0u || n == 0u || k == 0u) return 0;
    if (a->shape[a->ndim - 2] != (int)m || b->shape[b->ndim - 2] != (int)k ||
        b->shape[b->ndim - 1] != (int)n) return 0;
    for (axis = 0; axis < batch_rank; axis++) {
        int extent = output->shape[axis];
        if (extent <= 0) return 0;
        if ((vx_webgpu_batch_dimension(a, rank, axis) != extent && vx_webgpu_batch_dimension(a, rank, axis) != 1) ||
            (vx_webgpu_batch_dimension(b, rank, axis) != extent && vx_webgpu_batch_dimension(b, rank, axis) != 1)) return 0;
        batches *= (uint32_t)extent;
    }
    if ((uint64_t)out_elements != (uint64_t)batches * m * n) return 0;

    memset(words, 0, sizeof(words));
    for (axis = batch_rank - 1; axis >= 0; axis--) {
        words[axis] = stride;
        stride *= (uint32_t)output->shape[axis];
    }
    {
        uint32_t a_stride = 1u, b_stride = 1u;
        for (axis = batch_rank - 1; axis >= 0; axis--) {
            words[batch_rank + axis] = vx_webgpu_batch_dimension(a, rank, axis) == 1 ? 0u : a_stride * m * k;
            words[2 * batch_rank + axis] =
                vx_webgpu_batch_dimension(b, rank, axis) == 1 ? 0u : b_stride * k * n;
            a_stride *= (uint32_t)vx_webgpu_batch_dimension(a, rank, axis);
            b_stride *= (uint32_t)vx_webgpu_batch_dimension(b, rank, axis);
        }
    }
    stride_bytes = (size_t)(batch_rank ? 3 * batch_rank : 1) * sizeof(uint32_t);
    strides = (uint32_t*)vx_webgpu_arena(stride_bytes);
    if (!strides) return 0;
    memcpy(strides, words, stride_bytes);

    memset(params, 0, sizeof(params));
    params[0] = (uint32_t)batch_rank;
    params[1] = m;
    params[2] = k;
    params[3] = n;
    params[4] = out_elements;
    params[5] = (uint32_t)a->dtype;
    params[6] = (uint32_t)b->dtype;
    params[7] = (uint32_t)output->dtype;
    params[8] = (uint32_t)a->quantization.zero_point;
    params[9] = (uint32_t)b->quantization.zero_point;
    params[10] = (uint32_t)output->quantization.zero_point;
    params[12] = vx_webgpu_f32_bits(a->quantization.scale);
    params[13] = vx_webgpu_f32_bits(b->quantization.scale);
    params[14] = vx_webgpu_f32_bits(output->quantization.scale);
    if (!vx_webgpu_bind(plan, a->data, a_bytes) ||
        !vx_webgpu_bind(plan, b->data, b_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_bind(plan, strides, stride_bytes) ||
        !vx_webgpu_params(plan, params, 16u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_Q_BATCH_MAT_MUL;
    if (!vx_webgpu_linear_1d(out_elements, 256u, plan->variants[0].workgroup))
        return 0;
    plan->variant_count = 1u;
    (void)node;
    return 1;
}

/*
 * QMaskedMean: average the sequence positions a mask keeps.
 *
 * The mask is ordinary int32 and carries no affine descriptor of its own --
 * it selects rather than measures, which is why it is the one operand here
 * without a scale.
 */
static int vx_webgpu_plan_qmaskedmean(const Node* node, T* input, T* output,
                                      VxWebGpuNodePlan* plan) {
    T* mask = vx_webgpu_port(node, VX_PORT_MASK);
    uint32_t elements, mask_elements, out_elements, params[12];
    size_t in_bytes, mask_bytes, out_bytes;
    uint32_t batch, sequence, width;
    if (!input || !mask || !output) return 0;
    if ((input->dtype != T_I8 && input->dtype != T_U8) ||
        (output->dtype != T_I8 && output->dtype != T_U8)) return 0;
    if (mask->dtype != T_I32 || mask->quantization.valid) return 0;
    if (!input->quantization.valid || !output->quantization.valid) return 0;
    if (input->ndim != 3 || mask->ndim != 2 || output->ndim != 2) return 0;
    batch = (uint32_t)input->shape[0];
    sequence = (uint32_t)input->shape[1];
    width = (uint32_t)input->shape[2];
    if (batch == 0u || sequence == 0u || width == 0u) return 0;
    if (mask->shape[0] != (int)batch || mask->shape[1] != (int)sequence) return 0;
    if (output->shape[0] != (int)batch || output->shape[1] != (int)width) return 0;
    if (!vx_webgpu_extent(input, &elements, &in_bytes) ||
        !vx_webgpu_extent(mask, &mask_elements, &mask_bytes) ||
        !vx_webgpu_extent(output, &out_elements, &out_bytes)) return 0;
    if (elements != batch * sequence * width ||
        mask_elements != batch * sequence ||
        out_elements != batch * width) return 0;
    memset(params, 0, sizeof(params));
    params[0] = batch;
    params[1] = sequence;
    params[2] = width;
    params[3] = (uint32_t)input->dtype;
    params[4] = (uint32_t)output->dtype;
    params[8] = (uint32_t)input->quantization.zero_point;
    params[9] = (uint32_t)output->quantization.zero_point;
    params[10] = vx_webgpu_f32_bits(input->quantization.scale);
    params[11] = vx_webgpu_f32_bits(output->quantization.scale);
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, mask->data, mask_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 12u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_Q_MASKED_MEAN_INT8;
    if (!vx_webgpu_linear_1d(out_elements, 256u, plan->variants[0].workgroup))
        return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * QLinear, QGemm and QMatMul: a byte dense layer.
 *
 * Three of the shader's six operands are per-output-channel, and the engine
 * already holds two of them -- the weight zero points and the integer bias --
 * as arrays in linear memory, so those are bound where they lie. The third is
 * the requantization multiplier, which is the product of the input and weight
 * scales over the output scale. That array does not exist until it is asked
 * for, so it is built into the forward's arena.
 *
 * Everything else is read from the node's parsed metadata rather than
 * recomputed, because that metadata is what the engine validated the graph
 * against.
 */
static int vx_webgpu_plan_qdense(const Node* node, T* output,
                                 VxWebGpuNodePlan* plan) {
    const QLinearMetadata* meta;
    T* input = vx_webgpu_port(node, VX_PORT_INPUT);
    T* weight = vx_webgpu_port(node, VX_PORT_WEIGHT);
    T* bias = vx_webgpu_port(node, VX_PORT_BIAS);
    uint32_t in_elements, weight_elements, out_elements, bias_elements;
    size_t in_bytes, weight_bytes, out_bytes, bias_bytes;
    uint32_t params[16];
    float* multipliers;
    long rows;
    int index = g_vx_runtime_node.idx;
    int channel;
    if (index < 0 || (size_t)index >= g_node_capacity || !g_qlinear_meta) return 0;
    meta = &g_qlinear_meta[index];
    if (!meta->valid || !meta->weight_scales || !meta->weight_zero_points ||
        !meta->bias || meta->d_in <= 0 || meta->d_out <= 0) return 0;
    if (!input) input = vx_webgpu_port(node, VX_PORT_A);
    if (!input || !weight || !bias || !output) return 0;
    if (input->dtype != meta->input_dtype || weight->dtype != meta->weight_dtype ||
        output->dtype != meta->output_dtype) return 0;
    if (!input->quantization.valid || !output->quantization.valid) return 0;
    if (!(output->quantization.scale > 0.0f)) return 0;
    if (input->ndim <= 0 || output->ndim != input->ndim || weight->ndim != 2) return 0;
    if (bias->dtype != T_I32 || bias->numel != meta->d_out) return 0;
    if (input->shape[input->ndim - 1] != meta->d_in ||
        output->shape[output->ndim - 1] != meta->d_out ||
        weight->shape[0] != meta->d_out || weight->shape[1] != meta->d_in) return 0;
    for (channel = 0; channel < input->ndim - 1; channel++)
        if (input->shape[channel] != output->shape[channel]) return 0;
    if (!vx_webgpu_extent(input, &in_elements, &in_bytes) ||
        !vx_webgpu_extent(weight, &weight_elements, &weight_bytes) ||
        !vx_webgpu_extent(output, &out_elements, &out_bytes) ||
        !vx_webgpu_extent(bias, &bias_elements, &bias_bytes)) return 0;
    if (weight_elements != (uint32_t)meta->d_out * (uint32_t)meta->d_in) return 0;
    rows = input->numel / meta->d_in;
    if (rows <= 0 || input->numel != rows * meta->d_in ||
        output->numel != rows * meta->d_out || rows > INT32_MAX) return 0;

    multipliers = (float*)vx_webgpu_arena(
        (size_t)meta->d_out * sizeof(*multipliers));
    if (!multipliers) return 0;
    for (channel = 0; channel < meta->d_out; channel++)
        multipliers[channel] = input->quantization.scale *
            meta->weight_scales[channel] / output->quantization.scale;

    memset(params, 0, sizeof(params));
    params[0] = (uint32_t)rows;
    params[1] = (uint32_t)meta->d_in;
    params[2] = (uint32_t)meta->d_out;
    params[3] = (uint32_t)input->dtype;
    params[4] = (uint32_t)weight->dtype;
    params[5] = (uint32_t)output->dtype;
    params[8] = (uint32_t)input->quantization.zero_point;
    params[9] = (uint32_t)output->quantization.zero_point;
    params[12] = vx_webgpu_f32_bits(input->quantization.scale);
    params[13] = vx_webgpu_f32_bits(output->quantization.scale);
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, weight->data, weight_bytes) ||
        !vx_webgpu_bind(plan, multipliers,
                        (size_t)meta->d_out * sizeof(*multipliers)) ||
        !vx_webgpu_bind(plan, meta->weight_zero_points,
                        (size_t)meta->d_out * sizeof(int32_t)) ||
        !vx_webgpu_bind(plan, meta->bias,
                        (size_t)meta->d_out * sizeof(int32_t)) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 16u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_Q_LINEAR_INT8;
    if (!vx_webgpu_linear_1d(out_elements, 256u, plan->variants[0].workgroup))
        return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * QuantizeLinear and DequantizeLinear.
 *
 * Unlike the other quantized operators these take their scale and zero point
 * as tensors rather than uniform words, because a graph states them as ports.
 * The zero point is optional and the shader says so through `has_zero_point`;
 * the binding is filled with a scratch slot either way, because a declared
 * binding must have a buffer even when nothing reads it.
 */
static int vx_webgpu_bind_optional_zero(const Node* node,
                                        VxWebGpuNodePlan* plan,
                                        T** zero_out) {
    T* zero = vx_webgpu_port(node, VX_PORT_ZERO_POINT);
    uint32_t elements;
    size_t bytes;
    *zero_out = NULL;
    if (zero) {
        if (zero->dtype != T_I8 && zero->dtype != T_U8 &&
            zero->dtype != T_I32 && zero->dtype != T_F32) return 0;
        if (!vx_webgpu_extent(zero, &elements, &bytes) || elements != 1u) return 0;
        *zero_out = zero;
        return vx_webgpu_bind(plan, zero->data, bytes);
    }
    {
        uint32_t* slot = vx_webgpu_scratch();
        if (!slot) return 0;
        memset(slot, 0, VX_WEBGPU_SCRATCH_WORDS * sizeof(*slot));
        return vx_webgpu_bind(plan, slot,
                              VX_WEBGPU_SCRATCH_WORDS * sizeof(*slot));
    }
}

static int vx_webgpu_plan_quantize(const Node* node, T* input, T* output,
                                   VxWebGpuNodePlan* plan) {
    T* scale = vx_webgpu_port(node, VX_PORT_SCALE);
    T* zero = NULL;
    uint32_t elements, out_elements, scale_elements, params[4];
    size_t in_bytes, out_bytes, scale_bytes;
    if (!input || !output || input->dtype != T_F32) return 0;
    if (output->dtype != T_I8 && output->dtype != T_U8) return 0;
    if (!vx_webgpu_f32_extent(scale, &scale_elements, &scale_bytes) ||
        scale_elements == 0u) return 0;
    if (!vx_webgpu_extent(input, &elements, &in_bytes) ||
        !vx_webgpu_extent(output, &out_elements, &out_bytes) ||
        elements != out_elements) return 0;
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, scale->data, scale_bytes) ||
        !vx_webgpu_bind_optional_zero(node, plan, &zero) ||
        !vx_webgpu_bind(plan, output->data, out_bytes)) return 0;
    params[0] = elements;
    params[1] = (uint32_t)output->dtype;
    params[2] = zero ? (uint32_t)zero->dtype : (uint32_t)output->dtype;
    params[3] = zero ? 1u : 0u;
    if (!vx_webgpu_params(plan, params, 4u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_QUANTIZE_LINEAR_TYPED;
    /* Four elements to a word, and the grid indexes words. */
    vx_webgpu_linear_grid(elements, 256u, plan->variants[0].workgroup);
    plan->variant_count = 1u;
    return 1;
}

static int vx_webgpu_plan_dequantize(const Node* node, T* input, T* output,
                                     VxWebGpuNodePlan* plan) {
    T* scale = vx_webgpu_port(node, VX_PORT_SCALE);
    T* zero = NULL;
    uint32_t elements, out_elements, scale_elements, params[8];
    size_t in_bytes, out_bytes, scale_bytes;
    if (!input || !output || output->dtype != T_F32) return 0;
    if (input->dtype != T_I8 && input->dtype != T_U8 && input->dtype != T_F32 && input->dtype != T_I32) return 0;
    if (!vx_webgpu_f32_extent(scale, &scale_elements, &scale_bytes) ||
        scale_elements != 1u) return 0;
    if (!vx_webgpu_extent(input, &elements, &in_bytes) ||
        !vx_webgpu_extent(output, &out_elements, &out_bytes) ||
        elements != out_elements) return 0;
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, scale->data, scale_bytes) ||
        !vx_webgpu_bind_optional_zero(node, plan, &zero) ||
        !vx_webgpu_bind(plan, output->data, out_bytes)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_DEQUANTIZE_LINEAR_TYPED;
    /* One element per invocation: the result is f32. */
    vx_webgpu_linear_grid(elements, 64u, plan->variants[0].workgroup);
    memset(params, 0, sizeof(params));
    params[0] = elements;
    params[1] = (uint32_t)input->dtype;
    params[2] = (uint32_t)T_F32;
    params[3] = zero ? (uint32_t)zero->dtype : (uint32_t)input->dtype;
    params[4] = (uint32_t)T_F32;
    params[5] = zero ? 1u : 0u;
    params[6] = plan->variants[0].workgroup[0] * 64u;
    if (!vx_webgpu_params(plan, params, 8u)) return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * RequantizeLinear: one affine descriptor to another, both byte.
 *
 * The shader takes a single multiplier rather than the two scales, which is
 * the ratio native divides by. Precomputing it is one rounding earlier than
 * the host does; that is inside the one-step bound quantized comparison
 * already allows and is why the bound exists.
 */
static int vx_webgpu_plan_requantize(T* input, T* output,
                                     VxWebGpuNodePlan* plan) {
    uint32_t elements, out_elements, params[12];
    size_t in_bytes, out_bytes;
    if (!input || !output) return 0;
    if ((input->dtype != T_I8 && input->dtype != T_U8) ||
        (output->dtype != T_I8 && output->dtype != T_U8)) return 0;
    if (!input->quantization.valid || !output->quantization.valid) return 0;
    if (!(output->quantization.scale > 0.0f)) return 0;
    if (!vx_webgpu_extent(input, &elements, &in_bytes) ||
        !vx_webgpu_extent(output, &out_elements, &out_bytes) ||
        elements != out_elements) return 0;
    memset(params, 0, sizeof(params));
    params[0] = elements;
    params[1] = (uint32_t)input->dtype;
    params[2] = (uint32_t)output->dtype;
    params[4] = (uint32_t)input->quantization.zero_point;
    params[5] = (uint32_t)output->quantization.zero_point;
    params[8] = vx_webgpu_f32_bits(
        input->quantization.scale / output->quantization.scale);
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 12u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_REQUANTIZE_LINEAR_TYPED;
    if (!vx_webgpu_linear_1d(elements, 256u, plan->variants[0].workgroup))
        return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * QAdd: two byte operands and one byte result, three affine descriptors.
 *
 * The fused activation is the same node parameter the float Add reads, and the
 * shader takes it as the twelfth word.
 */
static int vx_webgpu_plan_qadd(const Node* node, T* input, T* output,
                               VxWebGpuNodePlan* plan) {
    T* a = vx_webgpu_port(node, VX_PORT_A);
    T* b = vx_webgpu_port(node, VX_PORT_B);
    uint32_t elements, other, params[12];
    size_t a_bytes, b_bytes, out_bytes;
    int relu;
    if (!a) a = input;
    if (!a || !b || !output) return 0;
    if ((a->dtype != T_I8 && a->dtype != T_U8) ||
        (b->dtype != T_I8 && b->dtype != T_U8) ||
        (output->dtype != T_I8 && output->dtype != T_U8)) return 0;
    if (!a->quantization.valid || !b->quantization.valid ||
        !output->quantization.valid) return 0;
    if (!vx_webgpu_extent(output, &elements, &out_bytes) ||
        !vx_webgpu_extent(a, &other, &a_bytes) || other != elements ||
        !vx_webgpu_extent(b, &other, &b_bytes) || other != elements) return 0;
    relu = vx_node_param_i32(node, VX_NODE_PARAM_RELU, 0);
    if (relu < 0 || relu > 2) return 0;
    memset(params, 0, sizeof(params));
    params[0] = elements;
    params[1] = (uint32_t)a->dtype;
    params[2] = (uint32_t)b->dtype;
    params[3] = (uint32_t)output->dtype;
    params[4] = (uint32_t)a->quantization.zero_point;
    params[5] = (uint32_t)b->quantization.zero_point;
    params[6] = (uint32_t)output->quantization.zero_point;
    params[8] = vx_webgpu_f32_bits(a->quantization.scale);
    params[9] = vx_webgpu_f32_bits(b->quantization.scale);
    params[10] = vx_webgpu_f32_bits(output->quantization.scale);
    params[11] = (uint32_t)relu;
    if (!vx_webgpu_bind(plan, a->data, a_bytes) ||
        !vx_webgpu_bind(plan, b->data, b_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 12u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_Q_ADD;
    if (!vx_webgpu_linear_1d(elements, 256u, plan->variants[0].workgroup))
        return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * QSiLU and QGELU: a byte activation with an affine mapping on each side.
 *
 * Both scalar transforms take the same twelve words -- how many elements,
 * which byte types, and the two affine descriptors -- because what the shader
 * does between them is the only difference. The descriptors come from the
 * tensors themselves, which is where the engine already keeps them.
 */
static int vx_webgpu_plan_q_activation(T* input, T* output,
                                       VxWebGpuNodePlan* plan,
                                       uint32_t shader_id) {
    uint32_t elements, out_elements, params[12];
    size_t in_bytes, out_bytes;
    if (!input || !output) return 0;
    if ((input->dtype != T_I8 && input->dtype != T_U8) ||
        (output->dtype != T_I8 && output->dtype != T_U8)) return 0;
    if (!input->quantization.valid || !output->quantization.valid) return 0;
    if (input->data == output->data) return 0;
    if (!vx_webgpu_extent(input, &elements, &in_bytes) ||
        !vx_webgpu_extent(output, &out_elements, &out_bytes) ||
        elements != out_elements) return 0;
    memset(params, 0, sizeof(params));
    params[0] = elements;
    params[1] = (uint32_t)input->dtype;
    params[2] = (uint32_t)output->dtype;
    params[4] = (uint32_t)input->quantization.zero_point;
    params[5] = (uint32_t)output->quantization.zero_point;
    params[8] = vx_webgpu_f32_bits(input->quantization.scale);
    params[9] = vx_webgpu_f32_bits(output->quantization.scale);
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 12u)) return 0;
    plan->variants[0].shader_id = shader_id;
    /* Four byte elements to a word, and the shader steps one word per lane. */
    if (!vx_webgpu_linear_1d(elements, 256u, plan->variants[0].workgroup))
        return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * Cross attention: queries from one sequence, keys and values from another.
 *
 * The same mask rule as SDPA, asked with two sequence lengths, and the same
 * arrangement for a node without one.
 */
static int vx_webgpu_plan_cross_sdpa_masked(const Node* node, T* output,
                                     VxWebGpuNodePlan* plan, T* mask, int row) {
    T* q = vx_webgpu_port(node, VX_PORT_Q);
    T* k = vx_webgpu_port(node, VX_PORT_K);
    T* v = vx_webgpu_port(node, VX_PORT_V);
    uint32_t q_elements, k_elements, v_elements, out_elements, mask_elements;
    size_t q_bytes, k_bytes, v_bytes, out_bytes, mask_bytes;
    uint32_t params[16] = {0};
    size_t param_count = 12u;
    uint32_t shader = VX_SHADER_INFERENCE_CROSS_SDPA;
    int d_model, batch, seq_q, seq_kv, heads, head_dim, mask_mode;
    float scale;
    if (!q || !k || !v || !output) return 0;
    if ((q->ndim != 2 && q->ndim != 3) || k->ndim != q->ndim ||
        v->ndim != q->ndim || output->ndim != q->ndim) return 0;
    if (!vx_webgpu_f32_extent(q, &q_elements, &q_bytes) ||
        !vx_webgpu_f32_extent(k, &k_elements, &k_bytes) ||
        !vx_webgpu_f32_extent(v, &v_elements, &v_bytes) ||
        !vx_webgpu_f32_extent(output, &out_elements, &out_bytes)) return 0;
    d_model = output->shape[output->ndim - 1];
    if (d_model <= 0 || q->shape[q->ndim - 1] != d_model ||
        k->shape[k->ndim - 1] != d_model || v->shape[v->ndim - 1] != d_model)
        return 0;
    batch = q->ndim == 3 ? q->shape[0] : 1;
    if ((k->ndim == 3 ? k->shape[0] : 1) != batch ||
        (v->ndim == 3 ? v->shape[0] : 1) != batch ||
        (output->ndim == 3 ? output->shape[0] : 1) != batch) return 0;
    seq_q = q->shape[q->ndim - 2];
    seq_kv = k->shape[k->ndim - 2];
    if (batch <= 0 || seq_q <= 0 || seq_kv <= 0) return 0;
    if (v->shape[v->ndim - 2] != seq_kv ||
        output->shape[output->ndim - 2] != seq_q) return 0;
    if ((uint64_t)v_elements != (uint64_t)batch * seq_kv * d_model ||
        k_elements != v_elements ||
        (uint64_t)q_elements != (uint64_t)batch * seq_q * d_model ||
        out_elements != q_elements) return 0;
    heads = node->parsed_params.heads;
    if (heads <= 0 || d_model % heads) return 0;
    head_dim = d_model / heads;
    if (head_dim > 64) return 0;
    mask_mode = attention_mask_mode(mask, batch, seq_q, seq_kv);
    if (mask_mode < 0) return 0;
    scale = node->parsed_params.has_scale
        ? node->parsed_params.scale : 1.0f / sqrtf((float)head_dim);
    memset(params, 0, sizeof(params));
    params[0] = (uint32_t)seq_q;
    params[1] = (uint32_t)seq_kv;
    params[2] = (uint32_t)d_model;
    params[3] = (uint32_t)heads;
    params[4] = (uint32_t)head_dim;
    params[5] = (uint32_t)batch;
    params[6] = vx_webgpu_f32_bits(scale);
    params[7] = !row && node->parsed_params.causal ? 1u : 0u;
    params[8] = (uint32_t)mask_mode;
    if (!vx_webgpu_bind(plan, q->data, q_bytes) ||
        !vx_webgpu_bind(plan, k->data, k_bytes) ||
        !vx_webgpu_bind(plan, v->data, v_bytes)) return 0;
    if (mask_mode != 0 && mask) {
        if (!vx_webgpu_extent(mask, &mask_elements, &mask_bytes) ||
            mask->dtype != T_I32 ||
            !vx_webgpu_bind(plan, mask->data, mask_bytes)) return 0;
    } else {
        uint32_t* slot = vx_webgpu_scratch();
        if (!slot) return 0;
        memset(slot, 0, VX_WEBGPU_SCRATCH_WORDS * sizeof(*slot));
        if (!vx_webgpu_bind(plan, slot,
                            VX_WEBGPU_SCRATCH_WORDS * sizeof(*slot))) return 0;
    }
#if VOLVOXAI_ENABLE_TRAINING
    if (g_native_training_mode) {
        if (vx_webgpu_training_rng((int)(node - g_n), 1, params + 9)) return 0;
        param_count = 16u;
        shader = VX_SHADER_TRAINING_CROSS_SDPA_TRAINING;
    }
#endif
    if (!vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, param_count)) return 0;
    plan->variants[0].shader_id = shader;
    plan->variants[0].workgroup[0] = ((uint32_t)seq_q + 63u) / 64u;
    plan->variants[0].workgroup[1] = (uint32_t)heads;
    plan->variants[0].workgroup[2] = (uint32_t)batch;
    if (plan->variants[0].workgroup[0] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[1] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[2] > VX_WEBGPU_WORKGROUP_LIMIT) return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * Conv2D over NHWC.
 *
 * Claimed only for the two weight layouts the shader indexes directly --
 * HWIO for a grouped or plain convolution, HWCM for a depthwise one -- and
 * only for f32 weights. Every other layout is a repacking native does into a
 * per-node cache this backend does not own, so those nodes stay where they are.
 *
 * A missing bias binds a C-owned zero vector for the same shader.
 */
static int vx_webgpu_plan_conv2d(const Node* node, T* input, T* output,
                                 VxWebGpuNodePlan* plan) {
    T* weight = vx_webgpu_port(node, VX_PORT_WEIGHT);
    T* bias = vx_webgpu_port(node, VX_PORT_BIAS);
    uint32_t in_elements, out_elements, weight_elements, bias_elements;
    size_t in_bytes, out_bytes, weight_bytes, bias_bytes = 0;
    const void* bias_data = NULL;
    uint32_t params[17];
    int layout, groups, relu, sy, sx, dy, dx;
    if (!input || !output || input->ndim != 4 || output->ndim != 4) return 0;
    if (!vx_webgpu_f32_extent(input, &in_elements, &in_bytes) ||
        !vx_webgpu_f32_extent(output, &out_elements, &out_bytes) ||
        !vx_webgpu_f32_extent(weight, &weight_elements, &weight_bytes) ||
        (bias && !vx_webgpu_f32_extent(bias, &bias_elements, &bias_bytes))) return 0;
    if (weight->ndim != 4) return 0;
    if (output->shape[0] != input->shape[0]) return 0;
    layout = node->parsed_params.weight_layout;
    if (layout != VX_CONV_WEIGHT_LAYOUT_HWIO &&
        layout != VX_CONV_WEIGHT_LAYOUT_HWCM) return 0;
    groups = node->parsed_params.groups;
    if (groups <= 0) groups = 1;
    relu = node->fuse_relu6 ? 2 : node->parsed_params.relu;
    if (relu < 0 || relu > 2) return 0;
    sy = node->parsed_params.stride[0];
    sx = node->parsed_params.stride[1];
    dy = node->parsed_params.dilation[0];
    dx = node->parsed_params.dilation[1];
    if (sy <= 0 || sx <= 0 || dy <= 0 || dx <= 0) return 0;
    if (node->parsed_params.pads[0] < 0 || node->parsed_params.pads[1] < 0) return 0;
    if (bias && bias_elements != (uint32_t)output->shape[3]) return 0;
    if (bias) bias_data = bias->data;
    else {
        bias_bytes = (size_t)output->shape[3] * sizeof(float);
        void* zeros = vx_webgpu_arena(bias_bytes);
        if (!zeros) return 0;
        memset(zeros, 0, bias_bytes);
        bias_data = zeros;
    }

    params[0] = (uint32_t)input->shape[0];
    params[1] = (uint32_t)input->shape[1];
    params[2] = (uint32_t)input->shape[2];
    params[3] = (uint32_t)input->shape[3];
    params[4] = (uint32_t)output->shape[3];
    params[5] = (uint32_t)output->shape[1];
    params[6] = (uint32_t)output->shape[2];
    params[7] = (uint32_t)weight->shape[0];
    params[8] = (uint32_t)weight->shape[1];
    params[9] = (uint32_t)sy;
    params[10] = (uint32_t)sx;
    params[11] = (uint32_t)node->parsed_params.pads[0];
    params[12] = (uint32_t)node->parsed_params.pads[1];
    params[13] = (uint32_t)groups;
    params[14] = (uint32_t)relu;
    params[15] = (uint32_t)dy;
    params[16] = (uint32_t)dx;
    if (params[3] == 0u || params[4] == 0u || params[7] == 0u || params[8] == 0u)
        return 0;
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, weight->data, weight_bytes) ||
        !vx_webgpu_bind(plan, bias_data, bias_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 17u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_CONV2_D;
    plan->variants[0].workgroup[0] = (params[6] + 7u) / 8u;
    plan->variants[0].workgroup[1] = (params[5] + 7u) / 8u;
    plan->variants[0].workgroup[2] = params[0] * params[4];
    if (plan->variants[0].workgroup[0] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[1] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[2] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[2] == 0u) return 0;
    plan->variant_count = 1u;
    return 1;
}

/* Defined with the strided routes below; BatchMatMul reaches it first. */
static int vx_webgpu_bind_scratch(VxWebGpuNodePlan* plan,
                                  const uint32_t* words, size_t count);

/*
 * GatherElements.
 *
 * One index per output element, read along one axis. Both shapes travel in
 * the uniform, so nothing needs the scratch pool, and the indices are proved
 * in range before the dispatch for the same reason Gather proves them.
 */
static int vx_webgpu_plan_gather_elements(const Node* node, T* input, T* output,
                                          VxWebGpuNodePlan* plan) {
    T* indices = vx_webgpu_port(node, VX_PORT_INDICES);
    uint32_t data_elements, index_count, out_elements, params[20];
    size_t in_bytes, index_bytes, out_bytes;
    int axis, dimension;
    if (!input) input = vx_webgpu_port(node, VX_PORT_DATA);
    if (!input || !indices || !output) return 0;
    if (input->dtype != T_F32 || output->dtype != T_F32 ||
        indices->dtype != T_I32) return 0;
    if (input->ndim <= 0 || input->ndim > 8 ||
        indices->ndim != input->ndim || output->ndim != input->ndim) return 0;
    if (!vx_webgpu_extent(input, &data_elements, &in_bytes) ||
        !vx_webgpu_extent(indices, &index_count, &index_bytes) ||
        !vx_webgpu_extent(output, &out_elements, &out_bytes)) return 0;
    if (index_count != out_elements) return 0;
    for (dimension = 0; dimension < input->ndim; dimension++)
        if (output->shape[dimension] != indices->shape[dimension]) return 0;
    axis = node->parsed_params.axis;
    if (axis < 0) axis += input->ndim;
    if (axis < 0 || axis >= input->ndim || input->shape[axis] <= 0) return 0;
    if (!g_bounded_gpu_value_domain_proven) {
        uint32_t position;
        for (position = 0; position < index_count; position++) {
            int32_t id;
            memcpy(&id, (const unsigned char*)indices->data +
                       (size_t)position * sizeof(id), sizeof(id));
            if (id < -input->shape[axis] || id >= input->shape[axis]) return 0;
        }
    }
    memset(params, 0, sizeof(params));
    params[0] = (uint32_t)input->ndim;
    params[1] = (uint32_t)axis;
    params[2] = out_elements;
    for (dimension = 0; dimension < input->ndim; dimension++) {
        params[4 + dimension] = (uint32_t)input->shape[dimension];
        params[12 + dimension] = (uint32_t)indices->shape[dimension];
    }
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, indices->data, index_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 20u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_GATHER_ELEMENTS;
    if (!vx_webgpu_linear_1d(out_elements, 64u, plan->variants[0].workgroup))
        return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * BatchMatMul.
 *
 * The last two axes are the matrix; everything before them is a batch grid the
 * operands broadcast over. The shader decomposes a flat batch index through
 * strides the engine supplies, so those travel in the scratch pool -- one slot
 * per node, as every strided route does.
 *
 * The operand strides are in elements rather than batches, because that is
 * what the shader adds to its row and column offsets.
 */
static int vx_webgpu_plan_batch_matmul(const Node* node, T* output,
                                       VxWebGpuNodePlan* plan) {
    T* a = vx_webgpu_port(node, VX_PORT_A);
    T* b = vx_webgpu_port(node, VX_PORT_B);
    uint32_t words[VX_WEBGPU_SCRATCH_WORDS];
    uint32_t a_elements, b_elements, out_elements;
    size_t a_bytes, b_bytes, out_bytes;
    uint32_t m, k, n, batches = 1u, stride = 1u;
    int rank, batch_rank, axis;
    if (!a || !b || !output) return 0;
    if (!vx_webgpu_f32_extent(a, &a_elements, &a_bytes) ||
        !vx_webgpu_f32_extent(b, &b_elements, &b_bytes) ||
        !vx_webgpu_f32_extent(output, &out_elements, &out_bytes)) return 0;
    rank = output->ndim;
    if (rank < 2 || rank > 8 || a->ndim < 2 || b->ndim < 2 ||
        rank != (a->ndim > b->ndim ? a->ndim : b->ndim)) return 0;
    batch_rank = rank - 2;
    m = (uint32_t)output->shape[rank - 2];
    n = (uint32_t)output->shape[rank - 1];
    k = (uint32_t)a->shape[a->ndim - 1];
    if (m == 0u || n == 0u || k == 0u) return 0;
    if (a->shape[a->ndim - 2] != (int)m || b->shape[b->ndim - 2] != (int)k ||
        b->shape[b->ndim - 1] != (int)n) return 0;
    for (axis = 0; axis < batch_rank; axis++) {
        int extent = output->shape[axis];
        if (extent <= 0) return 0;
        if ((vx_webgpu_batch_dimension(a, rank, axis) != extent && vx_webgpu_batch_dimension(a, rank, axis) != 1) ||
            (vx_webgpu_batch_dimension(b, rank, axis) != extent && vx_webgpu_batch_dimension(b, rank, axis) != 1)) return 0;
        batches *= (uint32_t)extent;
    }
    if ((uint64_t)out_elements != (uint64_t)batches * m * n) return 0;

    memset(words, 0, sizeof(words));
    words[0] = (uint32_t)batch_rank;
    words[1] = m;
    words[2] = k;
    words[3] = n;
    words[4] = batches;
    /* Output batch strides count batches; operand strides count elements. */
    for (axis = batch_rank - 1; axis >= 0; axis--) {
        words[5 + axis] = stride;
        stride *= (uint32_t)output->shape[axis];
    }
    {
        uint32_t a_stride = 1u, b_stride = 1u;
        for (axis = batch_rank - 1; axis >= 0; axis--) {
            words[5 + batch_rank + axis] =
                vx_webgpu_batch_dimension(a, rank, axis) == 1 ? 0u : a_stride * m * k;
            words[5 + 2 * batch_rank + axis] =
                vx_webgpu_batch_dimension(b, rank, axis) == 1 ? 0u : b_stride * k * n;
            a_stride *= (uint32_t)vx_webgpu_batch_dimension(a, rank, axis);
            b_stride *= (uint32_t)vx_webgpu_batch_dimension(b, rank, axis);
        }
    }
    if (!vx_webgpu_bind(plan, a->data, a_bytes) ||
        !vx_webgpu_bind(plan, b->data, b_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_bind_scratch(plan, words, (size_t)(5 + 3 * batch_rank)))
        return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_BATCH_MAT_MUL;
    plan->variants[0].workgroup[0] = (n + 7u) / 8u;
    plan->variants[0].workgroup[1] = (m + 7u) / 8u;
    plan->variants[0].workgroup[2] = batches;
    if (plan->variants[0].workgroup[0] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[1] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[2] > VX_WEBGPU_WORKGROUP_LIMIT) return 0;
    plan->variant_count = 1u;
    (void)node;
    return 1;
}

/* Elements after `axis`, which is the stride one step along it costs. */
static uint32_t vx_webgpu_inner_after(const T* tensor, int axis) {
    uint32_t inner = 1u;
    int dimension;
    for (dimension = axis + 1; dimension < tensor->ndim; dimension++) {
        if (tensor->shape[dimension] <= 0) return 0u;
        inner *= (uint32_t)tensor->shape[dimension];
    }
    return inner;
}

/*
 * QLayerNorm and QGroupNorm, in two dispatches.
 *
 * Neither normalisation fits one pass: the first reduces each row (or each
 * batch-group) to a mean and an inverse standard deviation, and the second
 * applies them. The statistics live in the forward's arena between the two,
 * which is exactly what an arena reset per forward is for.
 *
 * Both take the same twelve uniform words in both dispatches, so the pair is
 * built once and used twice.
 */
static int vx_webgpu_plan_qnorm(const Node* node, T* input, T* output,
                                VxWebGpuNodePlan* plans, uint32_t* count,
                                int group_norm) {
    T* gamma = vx_webgpu_port(node, VX_PORT_WEIGHT);
    T* beta = vx_webgpu_port(node, VX_PORT_BIAS);
    uint32_t elements, out_elements, gamma_count, beta_count;
    size_t in_bytes, out_bytes, gamma_bytes, beta_bytes, stats_bytes;
    uint32_t params[12], reductions, width;
    float* stats;
    float epsilon;
    int dimension;
    if (!gamma) gamma = vx_webgpu_port(node, VX_PORT_SCALE);
    if (!input || !output || !gamma || !beta) return 0;
    if ((input->dtype != T_I8 && input->dtype != T_U8) ||
        (output->dtype != T_I8 && output->dtype != T_U8)) return 0;
    if (!input->quantization.valid || !output->quantization.valid) return 0;
    if (input->data == output->data) return 0;
    if (gamma->dtype != T_F32 || beta->dtype != T_F32) return 0;
    if (!vx_webgpu_extent(input, &elements, &in_bytes) ||
        !vx_webgpu_extent(output, &out_elements, &out_bytes) ||
        !vx_webgpu_f32_extent(gamma, &gamma_count, &gamma_bytes) ||
        !vx_webgpu_f32_extent(beta, &beta_count, &beta_bytes) ||
        elements != out_elements) return 0;
    if (input->ndim <= 0 || output->ndim != input->ndim) return 0;
    for (dimension = 0; dimension < input->ndim; dimension++)
        if (input->shape[dimension] != output->shape[dimension] ||
            input->shape[dimension] <= 0) return 0;
    epsilon = vx_node_param_f32(node, VX_NODE_PARAM_EPS, 1.0e-5f);
    if (!(epsilon > 0.0f) || epsilon > 1.0e30f) return 0;

    memset(params, 0, sizeof(params));
    if (group_norm) {
        int groups = node->parsed_params.num_groups;
        if (input->ndim != 4 || groups <= 0) return 0;
        width = (uint32_t)input->shape[3];
        if (width == 0u || width % (uint32_t)groups) return 0;
        if (gamma_count != width || beta_count != width) return 0;
        params[0] = (uint32_t)input->shape[0];
        params[1] = (uint32_t)input->shape[1];
        params[2] = (uint32_t)input->shape[2];
        params[3] = width;
        params[4] = (uint32_t)groups;
        params[5] = (uint32_t)input->dtype;
        params[6] = (uint32_t)output->dtype;
        params[8] = (uint32_t)input->quantization.zero_point;
        params[9] = (uint32_t)output->quantization.zero_point;
        reductions = params[0] * (uint32_t)groups;
    } else {
        width = (uint32_t)input->shape[input->ndim - 1];
        if (width == 0u || elements % width) return 0;
        if (gamma_count != width || beta_count != width) return 0;
        reductions = elements / width;
        params[0] = reductions;
        params[1] = width;
        params[2] = (uint32_t)input->dtype;
        params[3] = (uint32_t)output->dtype;
        params[4] = (uint32_t)input->quantization.zero_point;
        params[5] = (uint32_t)output->quantization.zero_point;
    }
    if (reductions == 0u || reductions > VX_WEBGPU_WORKGROUP_LIMIT) return 0;
    /* The two scale words and epsilon sit at different offsets in the two
     * uniforms, because one carries a spatial shape and the other a width. */
    if (group_norm) {
        params[10] = vx_webgpu_f32_bits(input->quantization.scale);
        params[11] = vx_webgpu_f32_bits(output->quantization.scale);
    } else {
        params[8] = vx_webgpu_f32_bits(input->quantization.scale);
        params[9] = vx_webgpu_f32_bits(output->quantization.scale);
        params[10] = vx_webgpu_f32_bits(epsilon);
    }

    stats_bytes = (size_t)reductions * 2u * sizeof(float);
    stats = (float*)vx_webgpu_arena(stats_bytes);
    if (!stats) return 0;

    /* First: reduce. One workgroup per row or per batch-group. */
    memset(&plans[0], 0, sizeof(plans[0]));
    if (!vx_webgpu_bind(&plans[0], input->data, in_bytes) ||
        !vx_webgpu_bind(&plans[0], stats, stats_bytes) ||
        !vx_webgpu_params(&plans[0], params, 12u)) return 0;
    plans[0].variants[0].shader_id = group_norm
        ? VX_SHADER_INFERENCE_Q_GROUP_NORM_STATS
        : VX_SHADER_INFERENCE_Q_LAYER_NORM_STATS;
    plans[0].variants[0].workgroup[0] = reductions;
    plans[0].variants[0].workgroup[1] = 1u;
    plans[0].variants[0].workgroup[2] = 1u;
    plans[0].variant_count = 1u;

    /* Second: apply, one invocation per packed output word. */
    memset(&plans[1], 0, sizeof(plans[1]));
    if (!vx_webgpu_bind(&plans[1], input->data, in_bytes) ||
        !vx_webgpu_bind(&plans[1], gamma->data, gamma_bytes) ||
        !vx_webgpu_bind(&plans[1], beta->data, beta_bytes) ||
        !vx_webgpu_bind(&plans[1], stats, stats_bytes) ||
        !vx_webgpu_bind(&plans[1], output->data, out_bytes) ||
        !vx_webgpu_params(&plans[1], params, 12u)) return 0;
    plans[1].variants[0].shader_id = group_norm
        ? VX_SHADER_INFERENCE_Q_GROUP_NORM_APPLY
        : VX_SHADER_INFERENCE_Q_LAYER_NORM_APPLY;
    if (!vx_webgpu_linear_1d(elements, 256u, plans[1].variants[0].workgroup))
        return 0;
    plans[1].variant_count = 1u;
    *count = 2u;
    return 1;
}

/*
 * Concat and Split, as a sequence of copies.
 *
 * Neither is one dispatch: Concat copies each input into its slice of the
 * output, Split copies each slice of the input into its own output. The engine
 * hands `run` one output tensor, so the rest are resolved from the node.
 *
 * Every copy is its own descriptor, and they are encoded in order into the
 * open pass -- which is what makes a node with several dispatches ordinary
 * rather than a special case in the wire.
 */
static int vx_webgpu_plan_concat(const Node* node, T* output,
                                 VxWebGpuNodePlan* plans, uint32_t* count) {
    uint32_t offset = 0u, inner, out_axis, total = 0u;
    int axis, index, dimension;
    if (!output || output->ndim <= 0 || output->ndim > 8) return 0;
    int packed = output->dtype == T_I8 || output->dtype == T_U8;
    int sigmoid = vx_node_param_i32(node, VX_NODE_PARAM_SIGMOID, 0);
    if ((!packed && output->dtype != T_F32 && output->dtype != T_I32) ||
        (sigmoid && output->dtype != T_F32) ||
        (packed && (!output->quantization.valid || !isfinite(output->quantization.scale) ||
                    output->quantization.scale <= 0.0f))) return 0;
    if (node->nin <= 0) return 0;
    axis = vx_node_param_i32(node, VX_NODE_PARAM_AXIS, 0);
    if (axis < 0) axis += output->ndim;
    if (axis < 0 || axis >= output->ndim || output->shape[axis] <= 0) return 0;
    inner = vx_webgpu_inner_after(output, axis);
    out_axis = (uint32_t)output->shape[axis];
    if (inner == 0u) return 0;
    for (index = 0; index < node->nin; index++) {
        T* source = t_find(node->ins[index].name);
        VxWebGpuNodePlan* plan = &plans[index];
        uint32_t elements, out_elements, params[8];
        size_t in_bytes, out_bytes;
        if (!source || source->dtype != output->dtype ||
            source->ndim != output->ndim || source->shape[axis] <= 0) return 0;
        if (packed && (!source->quantization.valid ||
            source->quantization.scale != output->quantization.scale ||
            source->quantization.zero_point != output->quantization.zero_point)) return 0;
        for (dimension = 0; dimension < output->ndim; dimension++)
            if (dimension != axis &&
                source->shape[dimension] != output->shape[dimension]) return 0;
        if (!vx_webgpu_extent(source, &elements, &in_bytes) ||
            !vx_webgpu_extent(output, &out_elements, &out_bytes)) return 0;
        memset(plan, 0, sizeof(*plan));
        params[0] = elements;
        params[1] = offset;
        params[2] = (uint32_t)source->shape[axis];
        params[3] = out_axis;
        params[4] = inner;
        params[5] = sigmoid ? 1u : 0u; params[6] = 0u; params[7] = 0u;
        if (!vx_webgpu_bind(plan, source->data, in_bytes) ||
            !vx_webgpu_bind(plan, output->data, out_bytes) ||
            !vx_webgpu_params(plan, params, 8u)) return 0;
        plan->variants[0].shader_id = packed ? VX_SHADER_INFERENCE_CONCAT_COPY_TYPED
            : sigmoid ? VX_SHADER_INFERENCE_CONCAT_COPY : VX_SHADER_INFERENCE_CONCAT_COPY32;
        if (!vx_webgpu_linear_1d(elements, 64u, plan->variants[0].workgroup))
            return 0;
        plan->variant_count = 1u;
        if ((uint32_t)source->shape[axis] > out_axis - offset || elements > UINT32_MAX - total) return 0;
        offset += (uint32_t)source->shape[axis];
        total += elements;
    }
    if (offset != out_axis || total != (uint32_t)output->numel) return 0;
    *count = (uint32_t)node->nin;
    return 1;
}

static int vx_webgpu_plan_split(const Node* node, T* input,
                                VxWebGpuNodePlan* plans, uint32_t* count) {
    uint32_t offset = 0u, inner, axis_in;
    int axis, index, dimension;
    if (!input || input->ndim <= 0 || input->ndim > 8) return 0;
    if (input->elem_size != 4u) return 0;
    if (node->nout <= 0) return 0;
    axis = vx_node_param_i32(node, VX_NODE_PARAM_AXIS, 0);
    if (axis < 0) axis += input->ndim;
    if (axis < 0 || axis >= input->ndim || input->shape[axis] <= 0) return 0;
    inner = vx_webgpu_inner_after(input, axis);
    axis_in = (uint32_t)input->shape[axis];
    if (inner == 0u) return 0;
    for (index = 0; index < node->nout; index++) {
        T* piece = t_find(node->outs[index].name);
        VxWebGpuNodePlan* plan = &plans[index];
        uint32_t elements, in_elements, params[5];
        size_t in_bytes, out_bytes;
        if (!piece || piece->dtype != input->dtype ||
            piece->ndim != input->ndim || piece->shape[axis] <= 0) return 0;
        for (dimension = 0; dimension < input->ndim; dimension++)
            if (dimension != axis &&
                piece->shape[dimension] != input->shape[dimension]) return 0;
        if (!vx_webgpu_extent(input, &in_elements, &in_bytes) ||
            !vx_webgpu_extent(piece, &elements, &out_bytes)) return 0;
        memset(plan, 0, sizeof(*plan));
        params[0] = elements;
        params[1] = inner;
        params[2] = (uint32_t)piece->shape[axis];
        params[3] = axis_in;
        params[4] = offset;
        if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
            !vx_webgpu_bind(plan, piece->data, out_bytes) ||
            !vx_webgpu_params(plan, params, 5u)) return 0;
        plan->variants[0].shader_id = VX_SHADER_INFERENCE_SPLIT;
        if (!vx_webgpu_linear_1d(elements, 64u, plan->variants[0].workgroup))
            return 0;
        plan->variant_count = 1u;
        offset += (uint32_t)piece->shape[axis];
    }
    if (offset != axis_in) return 0;
    *count = (uint32_t)node->nout;
    return 1;
}

/*
 * Gather along one axis.
 *
 * Indices are read on the device without a bounds check, so their range is
 * proved before the dispatch is encoded -- by the graph's value-domain proof
 * where one exists, and otherwise index by index.
 *
 * The shader also declares a partial-residency table it only reads when the
 * bank domain is non-zero. WebGPU still requires every declared binding to
 * have a buffer, so a scratch slot stands in and the domain says to ignore it.
 */
static int vx_webgpu_plan_gather(const Node* node, T* input, T* output,
                                 VxWebGpuNodePlan* plan) {
    T* indices = vx_webgpu_port(node, VX_PORT_INDICES);
    uint32_t data_elements, index_count, out_elements, params[24];
    size_t in_bytes, index_bytes, out_bytes;
    uint32_t bank[4] = {0u, 0u, 0u, 0u};
    int axis, dimension, output_rank;
    if (!input) input = vx_webgpu_port(node, VX_PORT_DATA);
    if (!input || !indices || !output) return 0;
    if (input->dtype != T_F32 || output->dtype != T_F32 ||
        indices->dtype != T_I32) return 0;
    if (input->ndim <= 0 || input->ndim > 8 || indices->ndim < 0 ||
        indices->ndim > 8) return 0;
    output_rank = input->ndim + indices->ndim - 1;
    if (output_rank < 0 || output_rank > 8 || output->ndim != output_rank)
        return 0;
    if (!vx_webgpu_extent(input, &data_elements, &in_bytes) ||
        !vx_webgpu_extent(indices, &index_count, &index_bytes) ||
        !vx_webgpu_extent(output, &out_elements, &out_bytes)) return 0;
    axis = node->parsed_params.axis;
    if (axis < 0) axis += input->ndim;
    if (axis < 0 || axis >= input->ndim || input->shape[axis] <= 0) return 0;
    if (node->resident_slot_domain) {
        if (axis != 0 || !node->resident_slot_rows) return 0;
        bank[0] = node->resident_slot_domain;
    }
    if (!g_bounded_gpu_value_domain_proven) {
        uint32_t position;
        for (position = 0; position < index_count; position++) {
            int32_t id;
            memcpy(&id, (const unsigned char*)indices->data +
                       (size_t)position * sizeof(id), sizeof(id));
            int64_t domain = bank[0] ? bank[0] : (uint32_t)input->shape[axis];
            int64_t normalized = id < 0 ? (int64_t)id + domain : id;
            if (normalized < 0 || normalized >= domain || (bank[0] &&
                node->resident_slot_rows[normalized] >= (uint32_t)input->shape[0])) return 0;
        }
    }
    memset(params, 0, sizeof(params));
    params[0] = (uint32_t)input->ndim;
    params[1] = (uint32_t)indices->ndim;
    params[2] = (uint32_t)axis;
    params[3] = out_elements;
    for (dimension = 0; dimension < input->ndim; dimension++)
        params[4 + dimension] = (uint32_t)input->shape[dimension];
    for (dimension = 0; dimension < output_rank; dimension++)
        params[12 + dimension] = (uint32_t)output->shape[dimension];
    memcpy(&params[20], bank, sizeof(bank));
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, indices->data, index_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes)) return 0;
    if (bank[0]) {
        if (!vx_webgpu_bind(plan, node->resident_slot_rows,
                (size_t)bank[0] * sizeof(uint32_t))) return 0;
    } else {
        uint32_t* slot = vx_webgpu_scratch();
        if (!slot) return 0;
        memset(slot, 0, VX_WEBGPU_SCRATCH_WORDS * sizeof(*slot));
        if (!vx_webgpu_bind(plan, slot,
                            VX_WEBGPU_SCRATCH_WORDS * sizeof(*slot))) return 0;
    }
    if (!vx_webgpu_params(plan, params, 24u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_GATHER_INT32;
    if (!vx_webgpu_linear_1d(out_elements, 64u, plan->variants[0].workgroup))
        return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * MatMul, Gemm and Linear, over a row-major weight.
 *
 * The shader reads `bias[col]` unconditionally and indexes the weight as
 * `[d_in, d_out]`, so this claims a node only when a bias exists and the
 * layout resolves that way. A transposed weight needs the other dense shader,
 * which also binds a scale buffer the engine does not own; those nodes stay on
 * the path that already runs them.
 *
 * The layout rule is native's: an explicit `weight_layout` or `transB` wins,
 * and otherwise the operator decides -- Linear is dout_din, MatMul and Gemm
 * are din_dout. Guessing from a square weight's shape would be ambiguous.
 */
static int vx_webgpu_plan_dense(const Node* node, T* input, T* output,
                                VxWebGpuNodePlan* plan) {
    T* weight = vx_webgpu_port(node, VX_PORT_WEIGHT);
    T* bias = vx_webgpu_port(node, VX_PORT_BIAS);
    uint32_t in_elements, out_elements, weight_elements, bias_elements;
    size_t in_bytes, out_bytes, weight_bytes, bias_bytes;
    uint32_t d_in, d_out, rows, params[8] = {0};
    VxNodeParamSymbol layout;
    int transposed = -1;
    if (!input || !output || input->ndim <= 0 || output->ndim <= 0) return 0;
    if (!vx_webgpu_f32_extent(input, &in_elements, &in_bytes) ||
        !vx_webgpu_f32_extent(output, &out_elements, &out_bytes) ||
        !vx_webgpu_f32_extent(weight, &weight_elements, &weight_bytes) ||
        (bias && !vx_webgpu_f32_extent(bias, &bias_elements, &bias_bytes))) return 0;
    if (weight->ndim != 2) return 0;
    d_in = (uint32_t)input->shape[input->ndim - 1];
    d_out = (uint32_t)output->shape[output->ndim - 1];
    if (d_in == 0u || d_out == 0u) return 0;
    if (in_elements % d_in || out_elements % d_out) return 0;
    rows = in_elements / d_in;
    if (rows == 0u || out_elements / d_out != rows) return 0;
    if ((uint64_t)weight_elements != (uint64_t)d_in * d_out) return 0;
    if (bias && bias_elements != d_out) return 0;
    if (vx_node_param_f32(node, VX_NODE_PARAM_ALPHA, 1.0f) != 1.0f ||
        vx_node_param_f32(node, VX_NODE_PARAM_BETA, 1.0f) != 1.0f) return 0;

    layout = vx_node_param_symbol(node, VX_NODE_PARAM_WEIGHT_LAYOUT,
                                  VX_NODE_SYMBOL_EMPTY);
    if (layout != VX_NODE_SYMBOL_EMPTY) {
        if (layout == VX_NODE_SYMBOL_DOUT_DIN) transposed = 1;
        else if (layout == VX_NODE_SYMBOL_DIN_DOUT) transposed = 0;
        else return 0;
    }
    if (transposed < 0 && vx_node_param_has(node, VX_NODE_PARAM_TRANS_B))
        transposed = vx_node_param_i32(node, VX_NODE_PARAM_TRANS_B, 0) ? 1 : 0;
    if (transposed < 0) transposed = (node->operator_kind == VX_OP_LINEAR) ? 1 : 0;
    if (weight->shape[0] != (int)(transposed ? d_out : d_in) ||
        weight->shape[1] != (int)(transposed ? d_in : d_out)) return 0;

    params[0] = rows;
    params[1] = d_in;
    params[2] = d_out;
    params[3] = (uint32_t)transposed;
    params[4] = bias != NULL;
    uint32_t zero = 0;
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, weight->data, weight_bytes) ||
        !(bias ? vx_webgpu_bind(plan, bias->data, bias_bytes) : vx_webgpu_bind_scratch(plan, &zero, 1u)) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 8u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_DENSE_F32;
    plan->variants[0].workgroup[0] = (d_out + 63u) / 64u;
    plan->variants[0].workgroup[1] = rows;
    plan->variants[0].workgroup[2] = 1u;
    if (plan->variants[0].workgroup[0] > VX_WEBGPU_WORKGROUP_LIMIT ||
        plan->variants[0].workgroup[1] > VX_WEBGPU_WORKGROUP_LIMIT) return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * Embedding.
 *
 * The identifiers are read on the device without a bounds check, so the range
 * has to be proved before the dispatch is encoded -- once by the graph's own
 * value-domain proof, and otherwise here, token by token. Native asks the same
 * question in the same order.
 */
static int vx_webgpu_plan_embedding(const Node* node, T* input, T* output,
                                    VxWebGpuNodePlan* plan) {
    T* weight = vx_webgpu_port(node, VX_PORT_WEIGHT);
    uint32_t tokens, out_elements, weight_elements, params[4];
    size_t token_bytes, weight_bytes, out_bytes;
    uint32_t width, vocab;
    if (!input || !output || input->dtype != T_I32) return 0;
    if (!vx_webgpu_f32_extent(weight, &weight_elements, &weight_bytes) ||
        !vx_webgpu_f32_extent(output, &out_elements, &out_bytes) ||
        !vx_webgpu_extent(input, &tokens, &token_bytes)) return 0;
    if (weight->ndim <= 0) return 0;
    width = (uint32_t)weight->shape[weight->ndim - 1];
    if (width == 0u || weight_elements % width != 0u) return 0;
    vocab = weight_elements / width;
    if (vocab == 0u || out_elements != tokens * width) return 0;
    if (!g_bounded_gpu_value_domain_proven) {
        uint32_t index;
        for (index = 0; index < tokens; index++) {
            int32_t id;
            memcpy(&id, (const unsigned char*)input->data + (size_t)index * sizeof(id),
                   sizeof(id));
            if (id < 0 || (uint32_t)id >= vocab) return 0;
        }
    }
    params[0] = tokens;
    params[1] = width;
    params[2] = vocab;
    params[3] = 0u;
    if (!vx_webgpu_bind(plan, input->data, token_bytes) ||
        !vx_webgpu_bind(plan, weight->data, weight_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 4u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_EMBEDDING;
    if (!vx_webgpu_linear_1d(tokens, 64u, plan->variants[0].workgroup)) return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * Slice.
 *
 * The whole descriptor is a uniform: rank, the output extents, and per axis a
 * start, a step and the input stride to walk. Starts are normalised the way
 * native normalises them, and an axis whose last selected element falls
 * outside the input is refused rather than clamped -- clamping would return a
 * plausible tensor of the wrong elements.
 */
static int vx_webgpu_plan_slice(const Node* node, T* input, T* output,
                                VxWebGpuNodePlan* plan) {
    uint32_t strides[8], params[36];
    uint32_t size, in_size;
    size_t in_bytes, out_bytes;
    int starts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int steps[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    unsigned selected = 0u;
    int rank, axis, index;
    if (!input || !output || input->dtype != output->dtype) return 0;
    if (input->elem_size != 4u) return 0;
    if (input->ndim <= 0 || input->ndim > 8 || output->ndim != input->ndim) return 0;
    if (!node->parsed_params.has_slice) return 0;
    if (!vx_webgpu_extent(input, &in_size, &in_bytes) ||
        !vx_webgpu_extent(output, &size, &out_bytes)) return 0;
    rank = input->ndim;
    if (node->parsed_params.slice_rank <= 0 ||
        node->parsed_params.slice_rank > rank) return 0;
    for (index = 0; index < node->parsed_params.slice_rank; index++) {
        long long start, last;
        axis = node->parsed_params.axes[index];
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank || (selected & (1u << (unsigned)axis))) return 0;
        selected |= 1u << (unsigned)axis;
        steps[axis] = node->parsed_params.steps[index];
        if (steps[axis] <= 0) return 0;
        start = node->parsed_params.starts[index];
        if (start < 0) start += input->shape[axis];
        if (start < 0) start = 0;
        if (start > input->shape[axis]) start = input->shape[axis];
        last = start + (long long)(output->shape[axis] - 1) * steps[axis];
        if (last < 0 || last >= input->shape[axis]) return 0;
        starts[axis] = (int)start;
    }
    for (axis = 0; axis < rank; axis++)
        if (!(selected & (1u << (unsigned)axis)) &&
            output->shape[axis] != input->shape[axis]) return 0;
    if (!vx_webgpu_contiguous_strides(input, strides)) return 0;
    memset(params, 0, sizeof(params));
    params[0] = (uint32_t)rank;
    params[1] = size;
    for (axis = 0; axis < rank; axis++) {
        params[4 + axis] = (uint32_t)output->shape[axis];
        params[12 + axis] = (uint32_t)starts[axis];
        params[20 + axis] = (uint32_t)steps[axis];
        params[28 + axis] = strides[axis];
    }
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_params(plan, params, 36u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_SLICE_ND;
    if (!vx_webgpu_linear_1d(size, 64u, plan->variants[0].workgroup)) return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * The strided routes: general broadcast, integer comparison, transpose.
 *
 * All three walk the output's coordinates and read their operands through
 * strides the engine computes. The strides travel as a storage buffer from the
 * scratch pool rather than a uniform, because they are sized by rank.
 */
static int vx_webgpu_bind_scratch(VxWebGpuNodePlan* plan,
                                  const uint32_t* words, size_t count) {
    uint32_t* slot = vx_webgpu_scratch();
    if (!slot || count > VX_WEBGPU_SCRATCH_WORDS) return 0;
    memcpy(slot, words, count * sizeof(*words));
    if (!validating) vx_webgpu_mark_host(NULL, slot, sizeof(*slot) * count, 0);
    /* Always the whole slot: a stable span keeps the host's device buffer for
     * this address the same size from one forward to the next. */
    return vx_webgpu_bind(plan, slot,
                          VX_WEBGPU_SCRATCH_WORDS * sizeof(*slot));
}

/*
 * Metadata for a shader that walks the output and reads two strided operands:
 *
 *     [total, rank, output strides, a strides, b strides, trailing]
 *
 * The comparison shaders read the trailing word as which comparison to make.
 */
static int vx_webgpu_strided_binary_metadata(T* a, T* b, T* output,
                                             uint32_t* words, size_t* count,
                                             uint32_t trailing) {
    uint32_t out_strides[8], a_strides[8], b_strides[8];
    int rank = output->ndim, axis;
    if (rank < 0 || rank > 8) return 0;
    if (!vx_webgpu_contiguous_strides(output, out_strides) ||
        !vx_webgpu_broadcast_strides(a, output, a_strides) ||
        !vx_webgpu_broadcast_strides(b, output, b_strides)) return 0;
    words[0] = (uint32_t)output->numel;
    words[1] = (uint32_t)rank;
    for (axis = 0; axis < rank; axis++) {
        words[2 + axis] = out_strides[axis];
        words[2 + rank + axis] = a_strides[axis];
        words[2 + 2 * rank + axis] = b_strides[axis];
    }
    words[2 + 3 * rank] = trailing;
    *count = (size_t)(3 + 3 * rank);
    return 1;
}

/* Equal and GreaterOrEqual, over int32. */
static int vx_webgpu_plan_compare(const Node* node, T* input, T* output,
                                  VxWebGpuNodePlan* plan) {
    T* a = vx_webgpu_port(node, VX_PORT_A);
    T* b = vx_webgpu_port(node, VX_PORT_B);
    uint32_t words[VX_WEBGPU_SCRATCH_WORDS];
    size_t count, a_bytes, b_bytes, out_bytes;
    uint32_t size, unused;
    if (!a) a = input;
    if (!a || !b || a->dtype != T_I32 || b->dtype != T_I32 ||
        output->dtype != T_I32) return 0;
    if (!vx_webgpu_extent(a, &unused, &a_bytes) ||
        !vx_webgpu_extent(b, &unused, &b_bytes) ||
        !vx_webgpu_extent(output, &size, &out_bytes)) return 0;
    if (!vx_webgpu_strided_binary_metadata(
            a, b, output, words, &count,
            (node->operator_kind == VX_OP_EQUAL) ? 0u : 1u)) return 0;
    if (!vx_webgpu_bind(plan, a->data, a_bytes) ||
        !vx_webgpu_bind(plan, b->data, b_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_bind_scratch(plan, words, count)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_COMPARE_I32;
    if (!vx_webgpu_linear_1d(size, 64u, plan->variants[0].workgroup)) return 0;
    plan->variant_count = 1u;
    return 1;
}

/*
 * Transpose.
 *
 * The permutation says which input axis each output axis came from, so the
 * stride the shader needs for output axis d is the input's contiguous stride
 * at perm[d]. The default is the full reversal, which is what native uses when
 * a node declares no permutation.
 */
static int vx_webgpu_plan_transpose(const Node* node, T* input, T* output,
                                    VxWebGpuNodePlan* plan) {
    uint32_t in_strides[8], out_strides[8], words[VX_WEBGPU_SCRATCH_WORDS];
    int perm[8], rank, axis;
    uint32_t size, in_size;
    size_t in_bytes, out_bytes;
    if (!input || !output || input->dtype != output->dtype) return 0;
    if ((input->elem_size != 4u &&
         !(input->elem_size == 1u &&
           (input->dtype == T_I8 || input->dtype == T_U8))) ||
        input->ndim <= 0 || input->ndim > 8) return 0;
    if (!vx_webgpu_extent(input, &in_size, &in_bytes) ||
        !vx_webgpu_extent(output, &size, &out_bytes) || size != in_size) return 0;
    rank = input->ndim;
    if (output->ndim != rank) return 0;
    for (axis = 0; axis < rank; axis++) perm[axis] = rank - 1 - axis;
    if (node->parsed_params.has_perm) {
        if (node->parsed_params.perm_rank != rank) return 0;
        memcpy(perm, node->parsed_params.perm, (size_t)rank * sizeof(perm[0]));
    }
    if (!vx_webgpu_contiguous_strides(input, in_strides) ||
        !vx_webgpu_contiguous_strides(output, out_strides)) return 0;
    words[0] = size;
    words[1] = (uint32_t)rank;
    for (axis = 0; axis < rank; axis++) {
        if (perm[axis] < 0 || perm[axis] >= rank) return 0;
        words[2 + axis] = out_strides[axis];
        words[2 + rank + axis] = in_strides[perm[axis]];
    }
    if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_bind_scratch(plan, words, (size_t)(2 + 2 * rank))) return 0;
    plan->variants[0].shader_id = input->elem_size == 1u ?
        VX_SHADER_INFERENCE_TRANSPOSE_TYPED : VX_SHADER_INFERENCE_GENERAL_TRANSPOSE;
    vx_webgpu_linear_grid(input->elem_size == 1u ?
        size / 4u + (size % 4u != 0u) : size,
        64u, plan->variants[0].workgroup);
    plan->variant_count = 1u;
    return 1;
}

/* C owns broadcast strides for every arithmetic shape, including fused Add/ReLU. */
static int vx_webgpu_plan_binary(const Node* node, T* input, T* output,
                                 VxWebGpuNodePlan* plan) {
    T* a = vx_webgpu_port(node, VX_PORT_A);
    T* b = vx_webgpu_port(node, VX_PORT_B);
    uint32_t size, a_count, b_count, words[28] = {0};
    size_t out_bytes, a_bytes, b_bytes;
    if (!a) a = input;
    if (!vx_webgpu_f32_extent(a, &a_count, &a_bytes) ||
        !vx_webgpu_f32_extent(b, &b_count, &b_bytes) ||
        !vx_webgpu_f32_extent(output, &size, &out_bytes) ||
        !vx_webgpu_contiguous_strides(output, words + 4) ||
        !vx_webgpu_broadcast_strides(a, output, words + 12) ||
        !vx_webgpu_broadcast_strides(b, output, words + 20)) return 0;
    words[0] = size;
    words[1] = (uint32_t)output->ndim;
    words[2] = (node->operator_kind == VX_OP_SUB) ? 1u :
               (node->operator_kind == VX_OP_DIV) ? 2u :
               (node->operator_kind == VX_OP_ADD) ? 3u : 0u;
    words[3] = words[2] == 3u ? (uint32_t)vx_node_param_i32(node, VX_NODE_PARAM_RELU, 0) : 0u;
    /* Direct indexing avoids stride division when both operands are full
     * tensors or scalars. General broadcasting uses the same validated shapes. */
    int direct = words[2] == 3u ? (a_count == size && b_count == size) :
        ((a_count == size || a_count == 1u) && (b_count == size || b_count == 1u));
    if (direct && vx_webgpu_linear_1d(size, 64u, plan->variants[0].workgroup)) {
        uint32_t params[5] = {size, b_count == 1u, b_count, a_count, a_count == 1u};
        uint32_t shader = words[2] == 1u ? VX_SHADER_INFERENCE_SUB :
                          words[2] == 2u ? VX_SHADER_INFERENCE_DIV :
                          words[2] == 3u ? VX_SHADER_INFERENCE_ADD_RELU : VX_SHADER_INFERENCE_MUL;
        if (words[2] == 3u) params[1] = words[3];
        if (!vx_webgpu_bind(plan, a->data, a_bytes) ||
            !vx_webgpu_bind(plan, b->data, b_bytes) ||
            !vx_webgpu_bind(plan, output->data, out_bytes) ||
            !vx_webgpu_params(plan, params, words[2] == 3u ? 2u : 5u)) return 0;
        plan->variants[0].shader_id = shader;
        plan->variant_count = 1u;
        return 1;
    }
    if (!vx_webgpu_bind(plan, a->data, a_bytes) ||
        !vx_webgpu_bind(plan, b->data, b_bytes) ||
        !vx_webgpu_bind(plan, output->data, out_bytes) ||
        !vx_webgpu_bind_scratch(plan, words, 28u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_BROADCAST_BINARY_NATIVE;
    vx_webgpu_linear_grid(size, 64u, plan->variants[0].workgroup);
    plan->variant_count = 1u;
    return 1;
}

/*
 * Two questions, asked separately.
 *
 * `supports` is asked without tensors and can only answer "is this an operator
 * this backend ever plans". `run` has the operands and answers "can this node
 * be planned". Folding them into one function meant the tensorless answer
 * could reach run() -- a claim with nothing planned, which run() then had to
 * treat as a malformed plan rather than a decline. A binary node has no
 * primary input at all, so that was not hypothetical.
 *
 * The table answers first, and everything after it is an operator whose
 * descriptor is a decision rather than a mapping.
 */
#include "webgpu_extended_plans.inc"

static int vx_webgpu_plan_node(const Node* node, T* input, T* output,
                               VxWebGpuNodePlan* plan) {
    const VxWebGpuOperatorPlan* tabled;
    if (!node || !output) return 0;
    if (node->operator_kind == VX_OP_MOE_ROUTER) return vx_webgpu_plan_moe_router(node, input, plan);
    if (node->operator_kind == VX_OP_CONV_1D) return vx_webgpu_plan_conv1d(node,input,output,plan);
    if (node->operator_kind == VX_OP_CONV_TRANSPOSE_2D) return vx_webgpu_plan_conv_transpose(node,input,output,plan);
    if (node->operator_kind == VX_OP_PAD) return vx_webgpu_plan_pad(node,input,output,plan);
    if (node->operator_kind == VX_OP_INTERPOLATE_1D) return vx_webgpu_plan_interp(input,output,plan);
    if ((node->operator_kind == VX_OP_SPATIAL_SOFTARGMAX_Y) || (node->operator_kind == VX_OP_PROFILE_X) ||
        (node->operator_kind == VX_OP_PROFILE_Y) || (node->operator_kind == VX_OP_MEAN_HEIGHT))
        return vx_webgpu_plan_vision(node,input,output,plan);
    if (node->operator_kind == VX_OP_CROSS_ATTENTION) return vx_webgpu_plan_cross_attention(node,output,plan);
    if (node->operator_kind == VX_OP_Q_EMBEDDING) return vx_webgpu_plan_qembedding(node,input,output,plan);
    if (node->operator_kind == VX_OP_Q_SDPA) return vx_webgpu_plan_qsdpa_masked(node,output,plan,vx_webgpu_port(node,VX_PORT_MASK),0);
    if (node->operator_kind == VX_OP_MOE_LINEAR) return vx_webgpu_plan_moe_linear(node, input, output, plan);
#if VOLVOXAI_ENABLE_TRAINING
    if (g_native_training_mode && (node->operator_kind == VX_OP_DROPOUT)) {
        uint32_t size, out_size, params[8] = {0};
        size_t in_bytes, out_bytes;
        if (!vx_webgpu_f32_extent(input, &size, &in_bytes) ||
            !vx_webgpu_f32_extent(output, &out_size, &out_bytes) || size != out_size ||
            vx_webgpu_training_rng((int)(node - g_n), 0, params + 1)) return 0;
        params[0] = size;
        if (!vx_webgpu_bind(plan, input->data, in_bytes) ||
            !vx_webgpu_bind(plan, output->data, out_bytes) ||
            !vx_webgpu_params(plan, params, 8u) ||
            !vx_webgpu_linear_1d(size, 64u, plan->variants[0].workgroup)) return 0;
        plan->variants[0].shader_id = VX_SHADER_TRAINING_DROPOUT;
        plan->variant_count = 1u;
        return 1;
    }
#endif
    /* Operators whose descriptor is a mapping are declared, not written. */
    tabled = vx_webgpu_operator(node->operator_kind);
    if (tabled) return vx_webgpu_plan_tabled(tabled, node, input, output, plan);
    if (node->operator_kind == VX_OP_MAX_POOL_2D)
        return input && vx_webgpu_plan_pool2d(node, input, output, plan,
                                              VX_SHADER_INFERENCE_MAX_POOL2_D);
    if (node->operator_kind == VX_OP_AVERAGE_POOL_2D)
        return input && vx_webgpu_plan_pool2d(node, input, output, plan,
                                              VX_SHADER_INFERENCE_AVERAGE_POOL2_D);
    /* A binary operator resolves its own operands by port, so a missing
     * primary input is ordinary rather than a reason to decline. */
    if ((node->operator_kind == VX_OP_ADD) || (node->operator_kind == VX_OP_SUB) ||
        (node->operator_kind == VX_OP_MUL) || (node->operator_kind == VX_OP_DIV))
        return vx_webgpu_plan_binary(node, input, output, plan);
    if (node->operator_kind == VX_OP_CLIP)
        return vx_webgpu_plan_clip(node, input, output, plan);
    if (node->operator_kind == VX_OP_LAYER_NORM)
        return input && vx_webgpu_plan_feature_norm(node, input, output, plan, 1);
    if (node->operator_kind == VX_OP_RMS_NORM)
        return input && vx_webgpu_plan_feature_norm(node, input, output, plan, 0);
    if ((node->operator_kind == VX_OP_EQUAL) || (node->operator_kind == VX_OP_GREATER_OR_EQUAL))
        return vx_webgpu_plan_compare(node, input, output, plan);
    if (node->operator_kind == VX_OP_TRANSPOSE)
        return input && vx_webgpu_plan_transpose(node, input, output, plan);
    if (node->operator_kind == VX_OP_GLOBAL_AVERAGE_POOL)
        return input && vx_webgpu_plan_global_average_pool(input, output, plan);
    if ((node->operator_kind == VX_OP_WHERE) || (node->operator_kind == VX_OP_MASK))
        return vx_webgpu_plan_where(node, output, plan);
    if ((node->operator_kind == VX_OP_EXPAND) || (node->operator_kind == VX_OP_BROADCAST))
        return input && vx_webgpu_plan_expand(input, output, plan);
    if (node->operator_kind == VX_OP_BATCH_NORM_2D)
        return input && vx_webgpu_plan_batch_norm2d(node, input, output, plan);
    if (node->operator_kind == VX_OP_GROUP_NORM)
        return input && vx_webgpu_plan_group_norm(node, input, output, plan);
    if (node->operator_kind == VX_OP_UPSAMPLE_NEAREST_2D)
        return input && vx_webgpu_plan_upsample2x(input, output, plan);
    if (node->operator_kind == VX_OP_ARG_MAX)
        return input && vx_webgpu_plan_argmax(node, input, output, plan);
    if (node->operator_kind == VX_OP_EMBEDDING)
        return input && vx_webgpu_plan_embedding(node, input, output, plan);
    if (node->operator_kind == VX_OP_SLICE)
        return input && vx_webgpu_plan_slice(node, input, output, plan);
    if ((node->operator_kind == VX_OP_MATMUL) || (node->operator_kind == VX_OP_GEMM) ||
        (node->operator_kind == VX_OP_LINEAR))
        return input && vx_webgpu_plan_dense(node, input, output, plan);
    if ((node->operator_kind == VX_OP_RESIZE) || (node->operator_kind == VX_OP_RESIZE_NEAREST_2D))
        return input && vx_webgpu_plan_resize(node, input, output, plan);
    if (node->operator_kind == VX_OP_GATHER)
        return vx_webgpu_plan_gather(node, input, output, plan);
    if (node->operator_kind == VX_OP_GATHER_ELEMENTS)
        return vx_webgpu_plan_gather_elements(node, input, output, plan);
    if (node->operator_kind == VX_OP_BATCH_MATMUL)
        return vx_webgpu_plan_batch_matmul(node, output, plan);
    if (node->operator_kind == VX_OP_CONV_2D)
        return input && vx_webgpu_plan_conv2d(node, input, output, plan);
    if (node->operator_kind == VX_OP_SDPA)
        return vx_webgpu_plan_sdpa(node, input, output, plan);
    if (node->operator_kind == VX_OP_CROSS_SDPA)
        return vx_webgpu_plan_cross_sdpa_masked(node, output, plan, vx_webgpu_port(node, VX_PORT_MASK), 0);
    if (node->operator_kind == VX_OP_Q_SILU)
        return input && vx_webgpu_plan_q_activation(
            input, output, plan, VX_SHADER_INFERENCE_Q_SI_LU_INT8);
    if (node->operator_kind == VX_OP_Q_GELU)
        return input && vx_webgpu_plan_q_activation(
            input, output, plan, VX_SHADER_INFERENCE_Q_GELU_INT8);
    if (node->operator_kind == VX_OP_Q_ADD)
        return vx_webgpu_plan_qadd(node, input, output, plan);
    if (node->operator_kind == VX_OP_REQUANTIZE_LINEAR)
        return input && vx_webgpu_plan_requantize(input, output, plan);
    if (node->operator_kind == VX_OP_QUANTIZE_LINEAR)
        return input && vx_webgpu_plan_quantize(node, input, output, plan);
    if (node->operator_kind == VX_OP_DEQUANTIZE_LINEAR)
        return input && vx_webgpu_plan_dequantize(node, input, output, plan);
    if ((node->operator_kind == VX_OP_Q_LINEAR) || (node->operator_kind == VX_OP_Q_GEMM) ||
        (node->operator_kind == VX_OP_Q_MATMUL))
        return vx_webgpu_plan_qdense(node, output, plan);
    if (node->operator_kind == VX_OP_Q_ARG_MAX)
        return input && vx_webgpu_plan_qargmax(node, input, output, plan);
    if (node->operator_kind == VX_OP_Q_MASKED_MEAN)
        return input && vx_webgpu_plan_qmaskedmean(node, input, output, plan);
    if (node->operator_kind == VX_OP_Q_BATCH_MATMUL)
        return vx_webgpu_plan_qbatch_matmul(node, output, plan);
    if (node->operator_kind == VX_OP_Q_CONV_2D)
        return vx_webgpu_plan_qconv2d(node, output, plan);
    return 0;
}

static int vx_webgpu_pointwise(const Node* node) {
    if (!node || node->nin != 1 || node->nout != 1) return 0;
    VxOperatorKind op = node->operator_kind;
    return (op == VX_OP_RELU) || (op == VX_OP_LEAKY_RELU) || (op == VX_OP_GELU) ||
        (op == VX_OP_SILU) || (op == VX_OP_SIGMOID) || (op == VX_OP_TANH) ||
        (op == VX_OP_SIN) || (op == VX_OP_COS) || (op == VX_OP_HARD_SIGMOID) ||
        (op == VX_OP_HARD_SWISH) || (op == VX_OP_CLIP);
}


#include "webgpu_row_plans.inc"

static int vx_webgpu_supports(void* user_data, const Node* node) {
    (void)user_data; (void)node;
    // Once selected, WebGPU owns the whole numerical graph. A missing plan
    // is a compile refusal, never an implicit synchronous CPU continuation.
    return g_use_webgpu ? VX_BACKEND_HANDLED : VX_BACKEND_DECLINED;
}

/*
 * A node's dispatches, in the order the pass must record them.
 *
 * Most operators are one. Concat writes each input into its slice of the
 * output and Split reads each slice into its own output, so those are one per
 * operand -- and the engine hands `run` only the primary output, which is why
 * they resolve the rest from the node themselves.
 */
static int vx_webgpu_plan_sequence(const Node* node, T* input, T* output,
                                   VxWebGpuNodePlan* plans, uint32_t* count) {
    if (g_active_row >= 0)
        return vx_webgpu_plan_row(node, input, output, plans, count);
    if ((node->operator_kind == VX_OP_CONCAT) || (node->operator_kind == VX_OP_CONCAT2))
        return vx_webgpu_plan_concat(node, output, plans, count);
    if (node->operator_kind == VX_OP_SPLIT)
        return input && vx_webgpu_plan_split(node, input, plans, count);
    if (node->operator_kind == VX_OP_Q_LAYER_NORM)
        return input && vx_webgpu_plan_qnorm(node, input, output, plans, count, 0);
    if (node->operator_kind == VX_OP_Q_GROUP_NORM)
        return input && vx_webgpu_plan_qnorm(node, input, output, plans, count, 1);
    memset(&plans[0], 0, sizeof(plans[0]));
    if (!vx_webgpu_plan_node(node, input, output, &plans[0])) return 0;
    *count = 1u;
    return 1;
}

/* Canonical shape and value proofs precede this monotonic resource check.
 * Use a private metadata arena: maximum descriptors must neither enlarge live
 * spans nor leave bootstrap scratch pointers naming fictitious host storage. */
int vx_webgpu_prove_node_domain(VxEngineState* state, const char* output_name,
    const VxWebGpuTensorBound* bounds, size_t count) {
    if (!state || !state->webgpu_state || !output_name || !bounds ||
        count > SIZE_MAX / sizeof(T) || count > SIZE_MAX / sizeof(T*)) return 0;
    T* saved = calloc(count, sizeof(*saved));
    T** changed = calloc(count, sizeof(*changed));
    if (!saved || !changed) { free(saved); free(changed); return 0; }
    VxEngineStateScope scope = vx_engine_state_scope_enter(state);
    VxWebGpuState* previous_state = state->webgpu_state;
    VxWebGpuState proof = {.limits = previous_state->limits};
    state->webgpu_state = &proof;
    size_t changed_count = 0;
    int accepted = 0, previous = validating;
    int previous_values = g_bounded_gpu_value_domain_proven;
    int previous_node = g_vx_runtime_node.idx;
    validating = 1;
    g_bounded_gpu_value_domain_proven = 1;
    for (size_t i = 0; i < count; i++) {
        T* tensor = t_find(bounds[i].name);
        if (!tensor || bounds[i].rank != (uint32_t)tensor->ndim || bounds[i].rank > 8u) goto done;
        size_t prior;
        for (prior = 0; prior < changed_count; prior++) if (changed[prior] == tensor) break;
        if (prior != changed_count) continue;
        saved[changed_count] = *tensor;
        changed[changed_count++] = tensor;
        uint64_t elements = 1;
        for (uint32_t axis = 0; axis < bounds[i].rank; axis++) {
            int64_t extent = bounds[i].maximum[axis];
            if (extent <= 0 || extent > INT32_MAX || elements > INT32_MAX / (uint64_t)extent) goto done;
            tensor->shape[axis] = (int)extent;
            elements *= (uint64_t)extent;
        }
        tensor->numel = (long)elements;
    }
    for (int i = 0; i < state->node_count; i++) {
        Node* node = &state->nodes[i];
        int matches = !strcmp(node->out, output_name);
        for (int port = 0; !matches && port < node->nout; port++)
            matches = !strcmp(node->outs[port].name, output_name);
        if (!matches) continue;
        /* Quantized planners index immutable metadata by the current node. */
        g_vx_runtime_node.idx = i;
        uint32_t capacity = vx_webgpu_plan_capacity(node), plan_count = 0;
        VxWebGpuNodePlan* plans = capacity ? calloc(capacity, sizeof(*plans)) : NULL;
        if (!plans) goto done;
        T* input = vx_webgpu_port(node, VX_PORT_INPUT);
        T* output = t_find(node->out);
        T copy_output;
        float copy_identity[4] = {0};
        if (input && output && input->data == output->data &&
            ((node->operator_kind == VX_OP_IDENTITY) || (node->operator_kind == VX_OP_RESHAPE) ||
             (node->operator_kind == VX_OP_FLATTEN) || (node->operator_kind == VX_OP_SQUEEZE) ||
             (node->operator_kind == VX_OP_UNSQUEEZE) || (node->operator_kind == VX_OP_DROPOUT))) {
            /* Bootstrap aliases may become separate storage after a rebind.
             * Prove the copy route with a distinct metadata-only identity;
             * validation never reads or uploads this projected output. */
            copy_output = *output;
            copy_output.data = copy_identity;
            output = &copy_output;
        }
        accepted = vx_webgpu_plan_sequence(node, input, output, plans, &plan_count) &&
            plan_count && plan_count <= capacity;
        for (uint32_t plan = 0; accepted && plan < plan_count; plan++)
            accepted = vx_webgpu_seal(&plans[plan]);
        free(plans);
        break;
    }
done:
    while (changed_count) { changed_count--; *changed[changed_count] = saved[changed_count]; }
    while (proof.arena) {
        VxWebGpuArenaBlock* next = proof.arena->next;
        free(proof.arena); proof.arena = next;
    }
    free(proof.spans);
    state->webgpu_state = previous_state;
    g_vx_runtime_node.idx = previous_node;
    g_bounded_gpu_value_domain_proven = previous_values;
    validating = previous;
    vx_engine_state_scope_leave(scope);
    free(saved); free(changed);
    return accepted;
}

static int vx_webgpu_encode_plans(VxWebGpuNodePlan* plans, uint32_t count, uint32_t node_index) {
    VxWebGpuDispatchBuffer wire;
    VxGpuDispatch header;
    unsigned char* cursor;
    uint32_t index;
    int variant = 0;
    if (!g_webgpu.pass_open) return VX_BACKEND_ERROR;
    for (index = 0; index < count; index++) {
        VxWebGpuNodePlan* plan = &plans[index];
        if (plan->binding_count > VX_WEBGPU_MAX_BINDINGS ||
            plan->params_bytes > VX_WEBGPU_MAX_PARAM_BYTES)
            return VX_BACKEND_ERROR;
        /* A plan that does not match what its shader declares is a planning
         * bug, not a shape this backend declines: refusing it here keeps a
         * mismatched dispatch from reaching a device that would run it. */
        if (!vx_webgpu_seal(plan)) return VX_BACKEND_ERROR;

        header.variant_count = plan->variant_count;
        header.binding_count = plan->binding_count;
        header.params_bytes = plan->params_bytes;
        header.node_index = node_index;
        header.params_slot = plan->params_slot;
        memset(&wire, 0, sizeof(wire));
        cursor = vx_webgpu_pack(wire.bytes, &header, sizeof(header));
        cursor = vx_webgpu_pack(cursor, plan->variants,
                                plan->variant_count * sizeof(*plan->variants));
        cursor = vx_webgpu_pack(cursor, plan->bindings,
                                plan->binding_count * sizeof(*plan->bindings));
        (void)vx_webgpu_pack(cursor, plan->params, plan->params_bytes);

        variant = validating ? 0 : g_webgpu.encode((uint32_t)(uintptr_t)wire.bytes);
        if (variant < 0) {
            /* Remember the failure for end_forward. A submitted pass cannot
             * undo a node the host could not record. */
            g_webgpu.encode_failed = 1;
            return VX_BACKEND_ERROR;
        }
        if (!validating) {
            for (uint32_t binding = 0; binding < plan->binding_count; binding++) {
                if (!plan->bindings[binding].writes) continue;
                VxWebGpuSpan* span = vx_webgpu_span(plan->bindings[binding].host_ptr);
                if (span) span->host_dirty = 1;
            }
        }
    }
    return variant;
}

void vx_webgpu_release_storage(void* host) {
    if (!host || !vx_engine_state_current()->webgpu_state) return;
    uint32_t pointer = (uint32_t)(uintptr_t)host;
    for (size_t i = 0; i < g_webgpu.span_count; i++) {
        if (g_webgpu.spans[i].host_ptr != pointer) continue;
        vx_gpu_release(pointer);
        g_webgpu.spans[i] = g_webgpu.spans[--g_webgpu.span_count];
        return;
    }
}

/* Raw row copies are planned in C, including packed byte tails. Prefix
 * ownership uses persistent host identities; no device data returns to JS. */
int vx_webgpu_transfer_rows(void* full, size_t full_bytes, size_t row_bytes,
    const int* indices, int count, void* packed, int scatter, int upload_packed) {
    if (!g_use_webgpu || !full || !packed || !indices || !row_bytes || count <= 0 ||
        full_bytes % row_bytes || full_bytes > UINT32_MAX ||
        row_bytes > UINT32_MAX / (uint32_t)count) return -1;
    for (int i = 0; i < count; i++)
        if (indices[i] < 0 || (size_t)indices[i] >= full_bytes / row_bytes) return -1;
    vx_webgpu_begin_forward(NULL);
    VxWebGpuNodePlan plan = {0};
    size_t bytes = (size_t)count * row_bytes;
    uint32_t params[4] = {(uint32_t)count, (uint32_t)row_bytes,
        (uint32_t)(full_bytes / row_bytes), (uint32_t)scatter};
    if (upload_packed) vx_webgpu_mark_host(NULL, packed, bytes, 0);
    int ok = vx_webgpu_bind(&plan, scatter ? packed : full, scatter ? bytes : full_bytes) &&
        vx_webgpu_row_metadata(&plan, indices, (size_t)count) &&
        vx_webgpu_bind(&plan, scatter ? full : packed, scatter ? full_bytes : bytes) &&
        vx_webgpu_params(&plan, params, 4) &&
        vx_webgpu_grid(&plan, VX_SHADER_INFERENCE_ROW_INDEX_TRANSFER,
            (uint32_t)(bytes / 256u + (bytes % 256u != 0u)), 1, 1);
    if (ok) ok = vx_webgpu_encode_plans(&plan, 1, UINT32_MAX) >= 0;
    if (!ok) g_webgpu.encode_failed = 1;
    return vx_webgpu_end_forward(NULL) || !ok ? -1 : 0;
}

int vx_webgpu_decode_feedback(const char* token_name, const char* keep_name, const char* output_name, int position) {
    T* tokens = t_find(token_name), *keep = t_find(keep_name), *output = t_find(output_name);
    if (!g_use_webgpu || !vx_engine_state_current()->webgpu_state || position < 1 || !tokens || !keep || !output) return -1;
    vx_webgpu_begin_forward(NULL);
    VxWebGpuNodePlan plans[3] = {0};
    int previous = position - 1, parked = 0, length = position + 1;
    VxDecodeLanePages pages = {0};
    VxDecodeRowSet rows = {.lanes = 1, .live = 1, .positions = &previous, .parked = &parked,
        .kv_lengths = &length, .pages = &pages, .key_capacity = length, .unpaged = 1};
    VxDecodeRowOperand source = {output, 1, output->shape[1], 1, 0};
    VxDecodeRowOperand target = {tokens, 1, tokens->shape[1], 1, 0};
    VxDecodeRowOperand mask = {keep, 1, keep->shape[1], 1, 0};
    T token = *tokens, one = *keep;
    token.shape[1] = one.shape[1] = 1; token.numel = one.numel = 1;
    token.data = vx_webgpu_arena(4); one.data = vx_webgpu_arena(4);
    int result = -1;
    if (!token.data || !one.data) goto done;
    *(int32_t*)one.data = 1;
    vx_webgpu_mark_host(NULL, one.data, 4, 0);
    if (!vx_webgpu_row_transfer(&plans[0], &source, &token, &rows, 0)) goto done;
    rows.positions = &position;
    if (!vx_webgpu_row_transfer(&plans[1], &target, &token, &rows, 1) ||
        !vx_webgpu_row_transfer(&plans[2], &mask, &one, &rows, 1)) goto done;
    result = vx_webgpu_encode_plans(plans, 3, UINT32_MAX) < 0 ? -1 : 0;
done:
    if (result) (void)vx_webgpu_end_forward(NULL);
    return result;
}

static int vx_webgpu_run(void* user_data, const Node* node,
                         T* input, T* output) {
    VxWebGpuNodePlan* plans;
    uint32_t count = 0u;
    int variant = 0;
    (void)user_data;
    if (!node || !output) return VX_BACKEND_ERROR;
    /* A node can only be recorded into an open pass. The engine reaches run()
     * during attestation as well as execution, and encoding there would submit
     * work nobody asked for. */
    if (!g_webgpu.pass_open) return VX_BACKEND_ERROR;
    uint32_t capacity = vx_webgpu_plan_capacity(node);
    if (!capacity || !(plans = calloc(capacity, sizeof(*plans)))) return VX_BACKEND_ERROR;
    if (!vx_webgpu_plan_sequence(node, input, output, plans, &count) || !count || count > capacity) {
        free(plans);
        return VX_BACKEND_ERROR;
    }
    variant = vx_webgpu_encode_plans(plans, count, (uint32_t)g_vx_runtime_node.idx);
    free(plans);
    if (variant < 0) return VX_BACKEND_ERROR;
    g_webgpu.last_variant = variant;
    /* Name the route from inside run(), the way every graph backend does. The
     * node loop's default is "cpu", and route evidence is what a caller reads
     * to know a device ran the node -- leaving it would report a device run as
     * a host one. */
    if (g_vx_runtime_node.node_backend)
        *g_vx_runtime_node.node_backend = "webgpu";
    return VX_BACKEND_HANDLED;
}

const VxBackend g_vx_webgpu_backend = {
    .kind = VX_BACKEND_KIND_WEBGPU,
    .name = "webgpu",
    .init = vx_webgpu_init,
    .supports = vx_webgpu_supports,
    .run = vx_webgpu_run,
    .outputs_device_resident = 1,
    .reset = vx_webgpu_reset,
    .begin_forward = vx_webgpu_begin_forward,
    .end_forward = vx_webgpu_end_forward,
    .mark_host = vx_webgpu_mark_host,
    .sync_host = vx_webgpu_sync_host,
    .teardown = vx_webgpu_teardown,
};

#endif /* VOLVOXAI_ENABLE_WEBGPU */
