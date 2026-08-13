#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "volvoxai.h"
#include "volvoxai_backend.h"

#include "backend_sdk.h"
#include "engine_internal.h"
#include "engine_core.h"
#include "generated/kernel_registry.h"
#include "inference_kernels.h"
#include "json_validation.h"
#include "public_api_internal.h"
#include "runtime_state.h"
#include "safetensors.h"
#include "shape_contract.h"
#include "thread_pool.h"

#if defined(VOLVOXAI_ENABLE_VULKAN) && VOLVOXAI_ENABLE_VULKAN
#include "vulkan_engine.h"
#endif
#if defined(VOLVOXAI_ENABLE_OPENGL) && VOLVOXAI_ENABLE_OPENGL
#include "opengl_engine.h"
#endif
#if defined(VOLVOXAI_ENABLE_METAL) && VOLVOXAI_ENABLE_METAL
#include "metal_engine.h"
#endif
#if defined(VOLVOXAI_ENABLE_CUDA) && VOLVOXAI_ENABLE_CUDA
#include "cuda_engine.h"
#endif

#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#if !defined(_WIN32) && \
    (defined(__linux__) || defined(__ANDROID__) || defined(__APPLE__) || \
     defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
     defined(__DragonFly__))
#include <sys/resource.h>
#define VX_PROCESS_MEMORY_HAS_GETRUSAGE 1
#else
#define VX_PROCESS_MEMORY_HAS_GETRUSAGE 0
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct VxOwnedOutput {
    char* name;
    VxDataType dtype;
    uint32_t rank;
    int64_t shape[VX_MAX_TENSOR_RANK];
    size_t byte_size;
    unsigned char* data;
} VxOwnedOutput;

typedef struct VxDeclaredTensor {
    char* name;
    char* symbols[VX_MAX_TENSOR_RANK];
    VxDataType dtype;
    uint32_t rank;
    VxDimensionKind kinds[VX_MAX_TENSOR_RANK];
    int64_t minimums[VX_MAX_TENSOR_RANK];
    int64_t maximums[VX_MAX_TENSOR_RANK];
    int64_t multiples[VX_MAX_TENSOR_RANK];
    size_t maximum_byte_size;
    /* Concrete mirrors are populated only for singleton/static descriptors;
     * they keep private-engine result validation isolated from dynamic arena
     * work deferred to the next tranche. */
    int64_t shape[VX_MAX_TENSOR_RANK];
    size_t byte_size;
} VxDeclaredTensor;

typedef VxDeclaredTensor VxDeclaredOutput;

struct VxWeightRevisionRecord {
    atomic_uint references;
    uint64_t identity;
    uint64_t revision;
    uint64_t content_hash;
    char** paths;
    size_t path_count;
    uint64_t allocated_bytes;
};

typedef struct VxAdapterRevisionRecord {
    atomic_uint references;
    uint64_t identity;
    uint64_t revision;
    uint64_t content_hash;
    char* name;
    char* package_path;
    char* version_name;
    uint64_t allocated_bytes;
    struct VxAdapterRevisionRecord* next;
} VxAdapterRevisionRecord;

typedef struct VxRuntimeCoordinator VxRuntimeCoordinator;

typedef struct VxResultBudgetTicket {
    VxRuntime* runtime;
    size_t bytes;
    int reserved;
} VxResultBudgetTicket;

struct VxRequest {
    atomic_uint references;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int condition_monotonic;
    uint64_t identity;
    VxRuntimeRequestState state;
    VxStatus terminal_status;
    VxCompiledModel* compiled;
    VxTensorBinding* inputs;
    size_t input_count;
    size_t owned_input_bytes;
    int64_t* resolved_input_shapes;
    int batch_eligible;
    VxResult* result;
    VxReport report;
    int cancel_requested;
    int superseded_requested;
    int32_t priority;
    uint64_t deadline_monotonic_micros;
    VxRuntimeFreshness freshness;
    uint64_t stream_key;
    uint64_t submitted_monotonic_micros;
    int deadline_missed;
    int budget_reserved;
    int request_count_reserved;
    int route_claim_reserved;
    size_t reserved_input_bytes;
    VxResultBudgetTicket result_budget;
    VxRuntimeCoordinator* coordinator;
    struct VxRequest* next;
    struct VxRequest* replacement_next;
    struct VxRequest* inflight_next;
};

struct VxRuntimeCoordinator {
    VxRuntime* runtime;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int condition_monotonic;
    pthread_t worker;
    int worker_started;
    int stopping;
    int join_started;
    int joined;
    int destroy_runtime_on_worker_exit;
    VxRequest* head;
    VxRequest* tail;
    /* LATEST payload snapshots copy outside the global queue mutex. One
     * transaction per compiled+stream key is allowed; unrelated streams can
     * still admit while close waits for every transaction to settle. */
    VxRequest* replacements;
    /* Selected requests stay visible here until their single physical callback
     * is atomically settled. Newer LATEST admissions can therefore suppress a
     * submitted result without pretending that its resources were reclaimed. */
    VxRequest* inflight;
    size_t active_requests;
    size_t active_input_bytes;
    size_t active_batch_bytes;
    uint64_t steps;
    uint64_t dispatches;
};

static VxStatus vx_runtime_coordinator_close(VxRuntimeCoordinator* coordinator);
static void vx_runtime_coordinator_destroy(VxRuntimeCoordinator* coordinator);

struct VxRuntime {
    atomic_uint references;
    atomic_uint_fast64_t next_object_identity;
    atomic_uint_fast64_t next_execution_identity;
    pthread_mutex_t mutex;
    pthread_cond_t direct_condition;
    size_t active_direct_calls;
    pthread_mutex_t result_budget_mutex;
    size_t active_unconsumed_results;
    size_t active_unconsumed_result_bytes;
    int closed;
    uint64_t identity;
    VxRuntimeOptions options;
    VxProviderRegistry* providers;
    /* The sole scheduled-execution authority. NULL until the first scheduled
     * submission. Direct execution never touches this field. */
    VxRuntimeCoordinator* coordinator;
};

typedef struct VxRuntimeDirectScope {
    VxRuntime* runtime;
    struct VxRuntimeDirectScope* previous;
} VxRuntimeDirectScope;

static _Thread_local VxRuntimeDirectScope* g_runtime_direct_scope;

static int vx_runtime_direct_thread_active(const VxRuntime* runtime) {
    for (VxRuntimeDirectScope* scope = g_runtime_direct_scope;
         scope; scope = scope->previous)
        if (scope->runtime == runtime) return 1;
    return 0;
}

static VxStatus vx_runtime_direct_begin(VxRuntime* runtime,
                                        VxRuntimeDirectScope* scope) {
    if (!runtime || !scope) return VX_STATUS_INVALID_ARGUMENT;
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->closed) {
        pthread_mutex_unlock(&runtime->mutex);
        return VX_STATUS_HANDLE_DISPOSED;
    }
    runtime->active_direct_calls++;
    pthread_mutex_unlock(&runtime->mutex);
    scope->runtime = runtime;
    scope->previous = g_runtime_direct_scope;
    g_runtime_direct_scope = scope;
    return VX_STATUS_OK;
}

static void vx_runtime_direct_end(VxRuntimeDirectScope* scope) {
    VxRuntime* runtime;
    if (!scope || !scope->runtime) return;
    runtime = scope->runtime;
    if (g_runtime_direct_scope == scope)
        g_runtime_direct_scope = scope->previous;
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->active_direct_calls) runtime->active_direct_calls--;
    if (!runtime->active_direct_calls)
        pthread_cond_broadcast(&runtime->direct_condition);
    pthread_mutex_unlock(&runtime->mutex);
    scope->runtime = NULL;
    scope->previous = NULL;
}

struct VxModel {
    atomic_uint references;
    VxRuntime* runtime;
    char* graph_path;
    cJSON* logical_graph;
    VxDeclaredTensor* inputs;
    size_t input_count;
    VxDeclaredTensor* outputs;
    size_t output_count;
    char graph_fingerprint[64];
    char shape_domain_proof_identity[96];
    uint64_t maximum_tensor_bytes;
    pthread_mutex_t revision_mutex;
    _Atomic(VxWeightRevisionRecord*) current_weights;
    _Atomic(VxAdapterRevisionRecord*) current_adapter;
    VxAdapterRevisionRecord* adapters;
    uint64_t identity;
    uint64_t graph_identity;
    uint64_t graph_revision;
    uint64_t weight_identity;
    uint64_t allocated_bytes;
    /* Load-time weight-bank residency, replayed into validation and execution
     * states so each borrowed descriptor table gets its own selected-row COW. */
    VxBankResidency* bank_residency;
    uint32_t** bank_residency_slots;
    size_t bank_residency_count;
};

typedef struct VxCompiledWeightStore {
    atomic_uint references;
    SafetensorsFile files[MAX_WEIGHT_FILES];
    size_t file_count;
    uint64_t raw_bytes;
    uint64_t allocated_bytes;
} VxCompiledWeightStore;

struct VxCompiledModel {
    atomic_uint references;
    VxModel* model;
    VxWeightRevisionRecord* weights;
    /* One immutable parse/read of the accepted weight revision. The compiled
     * owner keeps it alive even while no execution contexts exist. */
    VxCompiledWeightStore* weight_store;
    VxAdapterRevisionRecord* adapter;
    uint64_t identity;
    uint64_t allocated_bytes;
    VxBackendPolicyMode mode;
    VxOperatorFallback operator_fallback;
    size_t maximum_cpu_typed_workspace_bytes;
    uint64_t maximum_resident_bytes;
    char backend[VX_BACKEND_NAME_CAPACITY];
    VolvoxAIEngineBackend builtin_backend;
    const VxBackendProvider* provider;
    void* provider_runtime;
    void* provider_compiled;
    VxBackendBatchContract batch_contract;
    int independent_batch_supported;
    char independent_batch_proof_identity[128];
    char independent_batch_rejection[192];
    char independent_batch_failed_node[VX_REPORT_NODE_CAPACITY];
    pthread_mutex_t route_mutex;
    int route_mutex_initialized;
    /* Scheduled claims are counted from queue publication through terminal
     * settlement so repeated direct try-locks cannot barge between requests. */
    atomic_uint scheduled_route_waiters;
    VxExecutionContext* route_context;
    VxReport report;
};

struct VxExecutionContext {
    atomic_uint references;
    VxCompiledModel* compiled;
    int compiled_reference_owned;
    uint64_t identity;
    uint64_t allocated_bytes;
    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_condition;
    uint64_t next_ticket;
    uint64_t serving_ticket;
    int closing;
    int closed;
    VxEngineState* engine_state;
    int engine_state_initialized;
    int engine_loaded;
    /* Built-in contexts retain the compiled store while their borrowed file
     * views are live. Closing the context drops this reference. */
    VxCompiledWeightStore* weight_store;
    VolvoxAIDecodeSession* decode_session;
    VxDecodeRowMode decode_row_mode;
    int require_incremental;
    VxAdapterRevisionRecord* adapter;
    char adapter_route_version[96];
    void* provider_context;
    int decode_seeded;
    int64_t* decode_input_shapes;
    VxDeclaredTensor* logical_tensors;
    size_t logical_tensor_count;
    size_t* logical_tensor_name_slots;
    size_t logical_tensor_name_capacity;
    char* committed_shape_signature;
    VolvoxAIEngineDynamicShapeStats dynamic_shape_stats;
    double dynamic_shape_bind_time_ms;
};

struct VxResult {
    atomic_uint references;
    VxExecutionContext* context;
    VxCompiledModel* compiled_lease;
    uint64_t execution_id;
    VxOwnedOutput* outputs;
    size_t output_count;
    size_t output_capacity;
    VxStatus output_sink_status;
    uint64_t snapshot_bytes;
    uint64_t evidence_allocated_bytes;
    uint64_t adapter_id;
    uint64_t adapter_revision;
    int32_t operator_fallback_used;
    int32_t route_attested;
    char route_evidence[VX_REPORT_ROUTE_CAPACITY];
    char fallback_evidence[VX_REPORT_FALLBACK_CAPACITY];
    char offending_node[VX_REPORT_NODE_CAPACITY];
    char decode_state[VX_REPORT_DECODE_CAPACITY];
    int64_t* input_shapes;
    size_t input_shape_count;
    VxResultBudgetTicket result_budget;
};

static int vx_compiled_maximum_result_bytes(const VxCompiledModel* compiled,
                                            size_t* out_bytes) {
    size_t bytes = 0;
    if (!compiled || !compiled->model || !out_bytes) return 0;
    for (size_t index = 0; index < compiled->model->output_count; index++) {
        size_t output_bytes =
            compiled->model->outputs[index].maximum_byte_size;
        if (output_bytes > SIZE_MAX - bytes) return 0;
        bytes += output_bytes;
    }
    *out_bytes = bytes;
    return 1;
}

static VxStatus vx_result_budget_reserve(VxCompiledModel* compiled,
                                         VxResultBudgetTicket* ticket) {
    VxRuntime* runtime;
    size_t bytes;
    if (!compiled || !ticket || ticket->reserved ||
        !vx_compiled_maximum_result_bytes(compiled, &bytes))
        return VX_STATUS_INVALID_ARGUMENT;
    runtime = compiled->model->runtime;
    pthread_mutex_lock(&runtime->result_budget_mutex);
    if (runtime->active_unconsumed_results >=
            runtime->options.max_unconsumed_results ||
        runtime->active_unconsumed_result_bytes >
            runtime->options.max_unconsumed_result_bytes ||
        bytes > runtime->options.max_unconsumed_result_bytes -
                    runtime->active_unconsumed_result_bytes) {
        pthread_mutex_unlock(&runtime->result_budget_mutex);
        return VX_STATUS_OVERLOADED;
    }
    runtime->active_unconsumed_results++;
    runtime->active_unconsumed_result_bytes += bytes;
    vx_runtime_retain(runtime);
    pthread_mutex_unlock(&runtime->result_budget_mutex);
    ticket->runtime = runtime;
    ticket->bytes = bytes;
    ticket->reserved = 1;
    return VX_STATUS_OK;
}

static void vx_result_budget_release(VxResultBudgetTicket* ticket) {
    VxRuntime* runtime;
    if (!ticket || !ticket->reserved || !ticket->runtime) return;
    runtime = ticket->runtime;
    pthread_mutex_lock(&runtime->result_budget_mutex);
    if (runtime->active_unconsumed_results)
        runtime->active_unconsumed_results--;
    if (ticket->bytes <= runtime->active_unconsumed_result_bytes)
        runtime->active_unconsumed_result_bytes -= ticket->bytes;
    else
        runtime->active_unconsumed_result_bytes = 0;
    pthread_mutex_unlock(&runtime->result_budget_mutex);
    memset(ticket, 0, sizeof(*ticket));
    vx_runtime_release(runtime);
}

static void vx_result_budget_move(VxResultBudgetTicket* destination,
                                  VxResultBudgetTicket* source) {
    if (!destination || !source || destination->reserved ||
        !source->reserved) return;
    *destination = *source;
    memset(source, 0, sizeof(*source));
}

static VxStatus vx_result_budget_shrink(VxResultBudgetTicket* ticket,
                                       uint64_t actual_bytes) {
    VxRuntime* runtime;
    size_t actual;
    size_t released;
    if (!ticket || !ticket->reserved || !ticket->runtime ||
        actual_bytes > SIZE_MAX)
        return VX_STATUS_INTERNAL;
    actual = (size_t)actual_bytes;
    if (actual > ticket->bytes) return VX_STATUS_INTERNAL;
    released = ticket->bytes - actual;
    runtime = ticket->runtime;
    pthread_mutex_lock(&runtime->result_budget_mutex);
    if (released > runtime->active_unconsumed_result_bytes) {
        pthread_mutex_unlock(&runtime->result_budget_mutex);
        return VX_STATUS_INTERNAL;
    }
    runtime->active_unconsumed_result_bytes -= released;
    ticket->bytes = actual;
    pthread_mutex_unlock(&runtime->result_budget_mutex);
    return VX_STATUS_OK;
}

typedef struct VxContextOperation {
    uint64_t ticket;
    int accepted;
} VxContextOperation;

static int vx_builtin_dynamic_shape_backend(const char* backend) {
    return backend && (!strcmp(backend, "cpu") ||
        !strcmp(backend, "vulkan") || !strcmp(backend, "opengl") ||
        !strcmp(backend, "metal") || !strcmp(backend, "cuda"));
}

static int vx_native_gpu_backend(VolvoxAIEngineBackend backend) {
    return backend == VOLVOXAI_BACKEND_VULKAN ||
        backend == VOLVOXAI_BACKEND_OPENGL ||
        backend == VOLVOXAI_BACKEND_METAL ||
        backend == VOLVOXAI_BACKEND_CUDA;
}

/* Built-in device integrations use one process-wide default physical device
 * per backend today.  These addresses are structural resource-domain tokens,
 * not caller strings; a future multi-device selector replaces them with the
 * selected device owner while preserving the same contract. */
static const void* vx_builtin_gpu_resource_domain(
        VolvoxAIEngineBackend backend) {
    static const unsigned char vulkan_domain = 1;
    static const unsigned char opengl_domain = 2;
    static const unsigned char metal_domain = 3;
    static const unsigned char cuda_domain = 4;
    switch (backend) {
        case VOLVOXAI_BACKEND_VULKAN: return &vulkan_domain;
        case VOLVOXAI_BACKEND_OPENGL: return &opengl_domain;
        case VOLVOXAI_BACKEND_METAL: return &metal_domain;
        case VOLVOXAI_BACKEND_CUDA: return &cuda_domain;
        default: return NULL;
    }
}

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
typedef void (*VxPublicApiCpuExecuteHook)(void* user_data);
static VxPublicApiCpuExecuteHook g_cpu_execute_hook;
static void* g_cpu_execute_hook_user_data;

void vx_public_api_test_set_cpu_execute_hook(VxPublicApiCpuExecuteHook hook,
                                              void* user_data) {
    g_cpu_execute_hook = hook;
    g_cpu_execute_hook_user_data = user_data;
}

int vx_public_api_test_dynamic_shape_state(
        const VxExecutionContext* context,
        uint64_t* resource_generation,
        size_t* arena_capacity,
        size_t* arena_high_water,
        uint64_t* arena_grow_count) {
    if (!context || !context->engine_state ||
        !vx_builtin_dynamic_shape_backend(context->compiled->backend))
        return -1;
    if (resource_generation)
        *resource_generation =
            context->engine_state->dynamic_resource_generation;
    if (arena_capacity)
        *arena_capacity =
            context->engine_state->dynamic_arena_capacity_bytes;
    if (arena_high_water)
        *arena_high_water =
            context->engine_state->dynamic_arena_high_water_bytes;
    if (arena_grow_count)
        *arena_grow_count =
            context->engine_state->dynamic_arena_grow_count;
    return 0;
}

int vx_public_api_test_fill_engine_i32_tensor(
        VxExecutionContext* context, const char* name, int32_t value) {
    T* tensor = NULL;
    if (!context || !context->engine_state || !name) return -1;
    for (int index = 0; index < context->engine_state->tensor_count; index++)
        if (!strcmp(context->engine_state->tensors[index].name, name)) {
            tensor = &context->engine_state->tensors[index];
            break;
        }
    if (!tensor || tensor->dtype != T_I32 || !tensor->data ||
        tensor->numel <= 0) return -1;
    for (long index = 0; index < tensor->numel; index++)
        memcpy((unsigned char*)tensor->data +
                   (size_t)index * sizeof(value),
               &value, sizeof(value));
    return 0;
}

int vx_public_api_test_conv_cache_state(
        const VxExecutionContext* context,
        int node_index,
        int* transformed_weight,
        int* cpu_pack,
        int* cpu_indirection) {
    const VxEngineState* state;
    if (!context || !(state = context->engine_state) ||
        node_index < 0 || node_index >= state->node_count)
        return -1;
    if (transformed_weight)
        *transformed_weight = state->conv_wcache[node_index] != NULL;
    if (cpu_pack)
        *cpu_pack = state->conv_pwf32_pack[node_index] != NULL ||
            state->conv_pwf32_pack_plain[node_index] != NULL ||
            state->conv_f32_igemm_pack[node_index] != NULL;
    if (cpu_indirection)
        *cpu_indirection =
            state->conv_f32_igemm_indir[node_index] != NULL ||
            state->conv_f32_igemm_zero[node_index] != NULL;
    return 0;
}

#if defined(VOLVOXAI_ENABLE_CUDA) && VOLVOXAI_ENABLE_CUDA && \
    defined(VOLVOXAI_CUDA_TESTING)
int vx_public_api_test_cuda_dynamic_reservation(
        const VxExecutionContext* context,
        uint64_t* allocation_count,
        size_t* span_count,
        int* preload_complete,
        int* enforced,
        int* replay_plan) {
    VxEngineStateScope scope;
    int result;
    if (!context || !context->engine_state ||
        context->compiled->builtin_backend != VOLVOXAI_BACKEND_CUDA)
        return -1;
    scope = vx_engine_state_scope_enter(context->engine_state);
    if (allocation_count)
        *allocation_count = cuda_test_graph_allocation_count();
    result = cuda_test_graph_domain_reservation(
        span_count, preload_complete, enforced, replay_plan);
    vx_engine_state_scope_leave(scope);
    return result;
}

int vx_public_api_test_cuda_dynamic_replay_state(
        const VxExecutionContext* context,
        int* graph_api_available,
        int* active_plan,
        size_t* plan_count,
        size_t* ready_plan_count,
        size_t* destroy_pending_count,
        uint64_t* capture_count,
        uint64_t* replay_count,
        uint64_t* invalidation_count,
        uint64_t* graph_exec_destroy_count,
        uint64_t* slot_epoch,
        uint64_t* capacity_generation,
        int* last_forward_replayed) {
    VxEngineStateScope scope;
    CudaGraphDynamicStateProbe probe = {0};
    int result;
    if (!context || !context->engine_state ||
        context->compiled->builtin_backend != VOLVOXAI_BACKEND_CUDA)
        return -1;
    scope = vx_engine_state_scope_enter(context->engine_state);
    result = cuda_test_graph_dynamic_state(NULL, &probe);
    if (result == 0) {
        if (graph_api_available)
            *graph_api_available = cuda_test_graph_api_available();
        if (active_plan) *active_plan = probe.replay_plan;
        if (plan_count) *plan_count = probe.replay_plan_count;
        if (ready_plan_count)
            *ready_plan_count = probe.replay_ready_plan_count;
        if (destroy_pending_count)
            *destroy_pending_count = probe.replay_destroy_pending_count;
        if (capture_count) *capture_count = cuda_test_graph_capture_count();
        if (replay_count) *replay_count = cuda_test_graph_replay_count();
        if (invalidation_count)
            *invalidation_count = cuda_test_graph_invalidation_count();
        if (graph_exec_destroy_count)
            *graph_exec_destroy_count =
                cuda_test_graph_exec_destroy_count();
        if (slot_epoch) *slot_epoch = probe.slot_epoch;
        if (capacity_generation)
            *capacity_generation = probe.capacity_generation;
        if (last_forward_replayed)
            *last_forward_replayed = cuda_graph_last_forward_replayed();
    }
    vx_engine_state_scope_leave(scope);
    return result;
}

int vx_public_api_test_cuda_invalidate_replay_key(
        const VxExecutionContext* context, int key) {
    VxEngineStateScope scope;
    if (!context || !context->engine_state || key < 1 || key > 3 ||
        context->compiled->builtin_backend != VOLVOXAI_BACKEND_CUDA)
        return -1;
    scope = vx_engine_state_scope_enter(context->engine_state);
    if (key == 1)
        cuda_test_mismatch_graph_replay_model_generation();
    else if (key == 2)
        cuda_test_advance_graph_slot_epoch();
    else
        cuda_test_advance_graph_capacity_generation();
    vx_engine_state_scope_leave(scope);
    return 0;
}

int vx_public_api_test_cuda_fail_next_graph_exec_destroy(
        const VxExecutionContext* context) {
    VxEngineStateScope scope;
    if (!context || !context->engine_state ||
        context->compiled->builtin_backend != VOLVOXAI_BACKEND_CUDA)
        return -1;
    scope = vx_engine_state_scope_enter(context->engine_state);
    cuda_test_fail_next_graph_exec_destroy();
    vx_engine_state_scope_leave(scope);
    return 0;
}

int vx_public_api_test_cuda_cleanup(
        const VxExecutionContext* context) {
    VxEngineStateScope scope;
    if (!context || !context->engine_state ||
        context->compiled->builtin_backend != VOLVOXAI_BACKEND_CUDA)
        return -1;
    scope = vx_engine_state_scope_enter(context->engine_state);
    cuda_cleanup();
    vx_engine_state_scope_leave(scope);
    return 0;
}

int vx_public_api_test_cuda_replay_resource_limits(
        const VxExecutionContext* context,
        uint32_t* plan_capacity,
        uint64_t* fixed_host_metadata_bytes) {
    VxEngineStateScope scope;
    CudaDomainLimits limits = {0};
    int result;
    if (!context || !context->engine_state ||
        context->compiled->builtin_backend != VOLVOXAI_BACKEND_CUDA)
        return -1;
    scope = vx_engine_state_scope_enter(context->engine_state);
    result = cuda_query_domain_limits(&limits);
    if (result == 0) {
        if (plan_capacity)
            *plan_capacity = limits.replay_plan_capacity;
        if (fixed_host_metadata_bytes)
            *fixed_host_metadata_bytes =
                limits.replay_fixed_host_metadata_bytes;
    }
    vx_engine_state_scope_leave(scope);
    return result;
}
#endif

int vx_public_api_test_cpu_typed_workspace_state(
        const VxExecutionContext* context,
        uintptr_t* address,
        size_t* bound_bytes,
        size_t* capacity_bytes) {
    if (!context || !context->engine_state || !context->compiled ||
        context->compiled->provider ||
        context->compiled->builtin_backend != VOLVOXAI_BACKEND_CPU)
        return -1;
    if (address)
        *address = (uintptr_t)context->engine_state->cpu_typed_workspace;
    if (bound_bytes)
        *bound_bytes =
            context->engine_state->cpu_typed_workspace_bound_bytes;
    if (capacity_bytes)
        *capacity_bytes =
            context->engine_state->cpu_typed_workspace_capacity_bytes;
    return 0;
}

int vx_public_api_test_cpu_typed_workspace_reconfigure(
        VxExecutionContext* context,
        size_t bounded_bytes) {
    VxEngineStateScope scope;
    int result;
    if (!context || !context->engine_state || !context->compiled ||
        context->compiled->provider ||
        context->compiled->builtin_backend != VOLVOXAI_BACKEND_CPU)
        return -1;
    scope = vx_engine_state_scope_enter(context->engine_state);
    result = volvoxai_engine_configure_cpu_typed_workspace(bounded_bytes);
    vx_engine_state_scope_leave(scope);
    return result;
}

int vx_public_api_test_compiled_resource_bounds(
        const VxCompiledModel* compiled,
        size_t* maximum_typed_scratch_bytes,
        uint64_t* maximum_resident_bytes) {
    if (!compiled || compiled->provider ||
        compiled->builtin_backend != VOLVOXAI_BACKEND_CPU)
        return -1;
    if (maximum_typed_scratch_bytes)
        *maximum_typed_scratch_bytes =
            compiled->maximum_cpu_typed_workspace_bytes;
    if (maximum_resident_bytes)
        *maximum_resident_bytes = compiled->maximum_resident_bytes;
    return 0;
}

int vx_public_api_test_copy_tensor(const VxExecutionContext* context,
                                   const char* name,
                                   void* bytes,
                                   size_t byte_size) {
    VxEngineStateScope scope;
    int result;
    if (!context || !context->engine_state || !name || !bytes ||
        strcmp(context->compiled->backend, "cpu")) return -1;
    scope = vx_engine_state_scope_enter(context->engine_state);
    result = volvoxai_engine_copy_tensor_raw(name, bytes, byte_size);
    vx_engine_state_scope_leave(scope);
    return result;
}

int vx_public_api_test_compiled_weight_store_state(
        const VxCompiledModel* compiled,
        unsigned* references,
        size_t* file_count,
        uint64_t* raw_bytes,
        uint64_t* store_allocated_bytes,
        uint64_t* compiled_allocated_bytes) {
    const VxCompiledWeightStore* store;
    if (!compiled || compiled->provider || !(store = compiled->weight_store))
        return -1;
    if (references)
        *references = atomic_load_explicit(&store->references,
                                           memory_order_acquire);
    if (file_count) *file_count = store->file_count;
    if (raw_bytes) *raw_bytes = store->raw_bytes;
    if (store_allocated_bytes)
        *store_allocated_bytes = store->allocated_bytes;
    if (compiled_allocated_bytes)
        *compiled_allocated_bytes = compiled->allocated_bytes;
    return 0;
}

int vx_public_api_test_compiled_weight_tensor_state(
        const VxCompiledModel* compiled,
        const char* name,
        uintptr_t* blob_address,
        uintptr_t* descriptor_table_address,
        uintptr_t* data_address,
        VxDataType* dtype,
        int* first_dimension,
        size_t* byte_size) {
    const VxCompiledWeightStore* store;
    if (!compiled || !name || compiled->provider ||
        !(store = compiled->weight_store)) return -1;
    for (size_t file_index = store->file_count; file_index > 0;
         file_index--) {
        const SafetensorsFile* file = &store->files[file_index - 1u];
        const SafetensorsTensor* tensor =
            safetensors_find_tensor(file, name);
        if (!tensor) continue;
        if (blob_address) *blob_address = (uintptr_t)file->blob;
        if (descriptor_table_address)
            *descriptor_table_address = (uintptr_t)file->tensors;
        if (data_address) *data_address = (uintptr_t)tensor->data;
        if (dtype) *dtype = tensor->dtype;
        if (first_dimension)
            *first_dimension = tensor->ndim > 0 ? tensor->shape[0] : 0;
        if (byte_size) *byte_size = tensor->nbytes;
        return 0;
    }
    return -1;
}

int vx_public_api_test_context_weight_tensor_state(
        const VxExecutionContext* context,
        const char* name,
        uintptr_t* blob_address,
        uintptr_t* descriptor_table_address,
        uintptr_t* stored_data_address,
        uintptr_t* execution_data_address,
        VxDataType* stored_dtype,
        int* execution_dtype,
        int* execution_owns,
        int* first_dimension,
        int* borrowed) {
    const VxEngineState* state;
    const SafetensorsTensor* stored = NULL;
    const T* execution = NULL;
    int found_file = -1;
    if (!context || !name || !(state = context->engine_state) ||
        !context->weight_store) return -1;
    for (int file_index = state->weight_file_count - 1;
         file_index >= 0; file_index--) {
        stored = safetensors_find_tensor(&state->weight_files[file_index],
                                         name);
        if (stored) {
            found_file = file_index;
            break;
        }
    }
    for (int tensor_index = 0; tensor_index < state->tensor_count;
         tensor_index++) {
        if (!strcmp(state->tensors[tensor_index].name, name)) {
            execution = &state->tensors[tensor_index];
            break;
        }
    }
    if (!stored || !execution || found_file < 0) return -1;
    if (blob_address)
        *blob_address = (uintptr_t)state->weight_files[found_file].blob;
    if (descriptor_table_address)
        *descriptor_table_address =
            (uintptr_t)state->weight_files[found_file].tensors;
    if (stored_data_address)
        *stored_data_address = (uintptr_t)stored->data;
    if (execution_data_address)
        *execution_data_address = (uintptr_t)execution->data;
    if (stored_dtype) *stored_dtype = stored->dtype;
    if (execution_dtype) *execution_dtype = execution->dtype;
    if (execution_owns) *execution_owns = execution->owns;
    if (first_dimension)
        *first_dimension = execution->ndim > 0 ? execution->shape[0] : 0;
    if (borrowed) *borrowed = state->weight_files_borrowed;
    return 0;
}
#endif

static uint64_t vx_runtime_identity(const VxRuntime* runtime) {
    struct timespec now = {0};
    uint64_t identity = (uint64_t)(uintptr_t)runtime;
    if (timespec_get(&now, TIME_UTC) == TIME_UTC) {
        identity ^= (uint64_t)now.tv_sec;
        identity ^= (uint64_t)now.tv_nsec << 32u;
    }
    identity ^= identity >> 30u;
    identity *= UINT64_C(0xbf58476d1ce4e5b9);
    identity ^= identity >> 27u;
    identity *= UINT64_C(0x94d049bb133111eb);
    identity ^= identity >> 31u;
    return identity ? identity : UINT64_C(1);
}

static uint64_t vx_next_object_identity(VxRuntime* runtime) {
    uint64_t identity = atomic_fetch_add_explicit(&runtime->next_object_identity, 1,
                                                   memory_order_relaxed);
    if (!identity)
        identity = atomic_fetch_add_explicit(&runtime->next_object_identity, 1,
                                              memory_order_relaxed);
    return identity;
}

static uint64_t vx_next_execution_identity(VxRuntime* runtime) {
    uint64_t identity = atomic_fetch_add_explicit(
        &runtime->next_execution_identity, 1, memory_order_relaxed);
    if (!identity)
        identity = atomic_fetch_add_explicit(
            &runtime->next_execution_identity, 1, memory_order_relaxed);
    return identity;
}

static double vx_report_now_ms(void) {
    struct timespec now;
    if (timespec_get(&now, TIME_UTC) != TIME_UTC) return 0.0;
    return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1000000.0;
}

static uint64_t vx_revision_update(uint64_t revision,
                                   const void* bytes,
                                   size_t byte_count) {
    const unsigned char* cursor = (const unsigned char*)bytes;
    for (size_t index = 0; index < byte_count; index++) {
        revision ^= cursor[index];
        revision *= UINT64_C(1099511628211);
    }
    return revision;
}

static void vx_snapshot_path_release(char* path) {
    if (!path) return;
    (void)remove(path);
    free(path);
}

static int vx_is_canonical_graph_path(const char* path) {
    const char* basename;
    const char* slash;
    const char* backslash;
    size_t length;
    static const char named_suffix[] = ".graph.json";
    if (!path || !path[0]) return 0;
    slash = strrchr(path, '/');
    backslash = strrchr(path, '\\');
    basename = slash && (!backslash || slash > backslash) ? slash + 1
        : backslash ? backslash + 1 : path;
    if (strcmp(basename, "graph.json") == 0) return 1;
    length = strlen(basename);
    return length > sizeof(named_suffix) - 1u &&
        strcmp(basename + length - (sizeof(named_suffix) - 1u), named_suffix) == 0;
}

/* The private engine still consumes filesystem paths. Copy each accepted
 * source into a process-owned, mode-0600 snapshot so later compilation and
 * context creation never reread caller-owned mutable files. */
static VxStatus vx_file_snapshot(const char* source_path,
                                 char** out_snapshot_path,
                                 uint64_t* out_revision) {
    unsigned char bytes[16384];
    uint64_t revision = UINT64_C(1469598103934665603);
    FILE* source = NULL;
    FILE* snapshot = NULL;
    char path[PATH_MAX];
    char* owned_path = NULL;
    size_t count;
    int failed = 0;
    if (!source_path || !source_path[0] || !out_snapshot_path)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_snapshot_path = NULL;
    source = fopen(source_path, "rb");
    if (!source) return VX_STATUS_IO_ERROR;
#ifdef _WIN32
    {
        char directory[MAX_PATH];
        DWORD length = GetTempPathA((DWORD)sizeof(directory), directory);
        if (!length || length >= sizeof(directory) ||
            !GetTempFileNameA(directory, "vxr", 0, path)) {
            fclose(source);
            return VX_STATUS_IO_ERROR;
        }
        snapshot = fopen(path, "wb");
        if (!snapshot) {
            (void)remove(path);
            fclose(source);
            return VX_STATUS_IO_ERROR;
        }
    }
#else
    {
        const char* directory = getenv("TMPDIR");
        int descriptor;
        int written;
        if (!directory || !directory[0]) directory = "/tmp";
        written = snprintf(path, sizeof(path), "%s%svolvoxai-snapshot-XXXXXX",
                           directory,
                           directory[strlen(directory) - 1u] == '/' ? "" : "/");
        if (written < 0 || (size_t)written >= sizeof(path)) {
            fclose(source);
            return VX_STATUS_IO_ERROR;
        }
        descriptor = mkstemp(path);
        if (descriptor < 0) {
            fclose(source);
            return VX_STATUS_IO_ERROR;
        }
        snapshot = fdopen(descriptor, "wb");
        if (!snapshot) {
            close(descriptor);
            (void)remove(path);
            fclose(source);
            return VX_STATUS_IO_ERROR;
        }
    }
#endif
    owned_path = (char*)malloc(strlen(path) + 1u);
    if (!owned_path) {
        fclose(snapshot);
        (void)remove(path);
        fclose(source);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    memcpy(owned_path, path, strlen(path) + 1u);
    while ((count = fread(bytes, 1, sizeof(bytes), source)) != 0) {
        revision = vx_revision_update(revision, bytes, count);
        if (fwrite(bytes, 1, count, snapshot) != count) {
            failed = 1;
            break;
        }
    }
    if (ferror(source)) failed = 1;
    if (fflush(snapshot) != 0) failed = 1;
    if (fclose(source) != 0) failed = 1;
    if (fclose(snapshot) != 0) failed = 1;
    if (failed) {
        vx_snapshot_path_release(owned_path);
        return VX_STATUS_IO_ERROR;
    }
    if (out_revision)
        *out_revision = revision ? revision : UINT64_C(1);
    *out_snapshot_path = owned_path;
    return VX_STATUS_OK;
}

#include "public_api_graph_package.inc"

static cJSON* vx_json_file_load(const char* path) {
    FILE* file = NULL;
    char* bytes = NULL;
    cJSON* root = NULL;
    long signed_size;
    size_t size;
    if (!path || !(file = fopen(path, "rb"))) return NULL;
    if (fseek(file, 0, SEEK_END) != 0 ||
        (signed_size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0 ||
        (size_t)signed_size == SIZE_MAX) goto done;
    size = (size_t)signed_size;
    bytes = (char*)malloc(size + 1u);
    if (!bytes) goto done;
    if (fread(bytes, 1, size, file) != size || ferror(file)) goto done;
    bytes[size] = '\0';
    root = cJSON_ParseWithLength(bytes, size + 1u);
done:
    free(bytes);
    fclose(file);
    return root;
}

static VxStatus vx_text_snapshot(const char* text, char** out_path) {
    FILE* snapshot = NULL;
    char path[PATH_MAX];
    char* owned = NULL;
    size_t length;
    if (!text || !out_path) return VX_STATUS_INVALID_ARGUMENT;
    *out_path = NULL;
    length = strlen(text);
#ifdef _WIN32
    {
        char directory[MAX_PATH];
        DWORD directory_length = GetTempPathA(
            (DWORD)sizeof(directory), directory);
        if (!directory_length || directory_length >= sizeof(directory) ||
            !GetTempFileNameA(directory, "vxl", 0, path))
            return VX_STATUS_IO_ERROR;
        snapshot = fopen(path, "wb");
        if (!snapshot) {
            (void)remove(path);
            return VX_STATUS_IO_ERROR;
        }
    }
#else
    {
        const char* directory = getenv("TMPDIR");
        int descriptor;
        int written;
        if (!directory || !directory[0]) directory = "/tmp";
        written = snprintf(path, sizeof(path),
                           "%s%svolvoxai-lowered-XXXXXX", directory,
                           directory[strlen(directory) - 1u] == '/' ? "" : "/");
        if (written < 0 || (size_t)written >= sizeof(path))
            return VX_STATUS_IO_ERROR;
        descriptor = mkstemp(path);
        if (descriptor < 0) return VX_STATUS_IO_ERROR;
        snapshot = fdopen(descriptor, "wb");
        if (!snapshot) {
            close(descriptor);
            (void)remove(path);
            return VX_STATUS_IO_ERROR;
        }
    }
#endif
    owned = vx_string_copy(path);
    if (!owned || (length && fwrite(text, 1, length, snapshot) != length) ||
        fflush(snapshot) != 0 || fclose(snapshot) != 0) {
        if (owned) vx_snapshot_path_release(owned);
        else {
            fclose(snapshot);
            (void)remove(path);
        }
        return owned ? VX_STATUS_IO_ERROR : VX_STATUS_OUT_OF_MEMORY;
    }
    *out_path = owned;
    return VX_STATUS_OK;
}

static cJSON* vx_shape_minimum_projection(const cJSON* shape,
                                          const cJSON* dimensions) {
    cJSON* projected;
    int rank;
    if (!cJSON_IsArray(shape)) return NULL;
    rank = cJSON_GetArraySize(shape);
    if (rank < 0 || rank > (int)VX_MAX_TENSOR_RANK) return NULL;
    projected = cJSON_CreateArray();
    if (!projected) return NULL;
    for (int axis = 0; axis < rank; axis++) {
        const cJSON* item = cJSON_GetArrayItem(shape, axis);
        int64_t extent;
        if (!vx_json_positive_safe_i64(item, &extent)) {
            int64_t minimum;
            int64_t maximum;
            int64_t multiple;
            int64_t remainder;
            int64_t adjustment;
            if (!cJSON_IsString(item) || !item->valuestring ||
                !vx_dimension_json(dimensions, item->valuestring,
                                   &minimum, &maximum, &multiple)) {
                cJSON_Delete(projected);
                return NULL;
            }
            remainder = minimum % multiple;
            adjustment = remainder ? multiple - remainder : 0;
            if (adjustment > maximum - minimum) {
                cJSON_Delete(projected);
                return NULL;
            }
            extent = minimum + adjustment;
        }
        /* The private physical tensor table intentionally remains int-sized.
         * Providers may support wider logical dimensions, but the built-in
         * native route must reject such a domain at context creation. */
        if (extent <= 0 || extent > INT_MAX ||
            !cJSON_AddItemToArray(projected,
                                  cJSON_CreateNumber((double)extent))) {
            cJSON_Delete(projected);
            return NULL;
        }
    }
    return projected;
}

/* Lower the closed logical v1 envelope into the private engine's concrete
 * bootstrap projection. This file is process-owned and immediately removed
 * after build_graph() has parsed it; it is not a compatibility input path. */
static VxStatus vx_graph_lower_private_bootstrap(const char* graph_path,
                                                 char** out_path) {
    cJSON* root = NULL;
    cJSON* dimensions;
    cJSON* inputs;
    cJSON* nodes;
    char* encoded = NULL;
    VxStatus status = VX_STATUS_INVALID_GRAPH;
    if (!graph_path || !out_path) return VX_STATUS_INVALID_ARGUMENT;
    *out_path = NULL;
    root = vx_json_file_load(graph_path);
    if (!root || !vx_json_object_keys_unique_recursive(root) ||
        !vx_graph_schema_valid(root)) goto done;
    dimensions = cJSON_GetObjectItemCaseSensitive(root, "dimensions");
    inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    for (cJSON* input = inputs->child; input; input = input->next) {
        cJSON* projected = vx_shape_minimum_projection(
            cJSON_GetObjectItemCaseSensitive(input, "shape"), dimensions);
        if (!projected || !cJSON_ReplaceItemInObjectCaseSensitive(
                              input, "shape", projected)) {
            cJSON_Delete(projected);
            goto done;
        }
    }
    for (cJSON* node = nodes->child; node; node = node->next) {
        cJSON* logical_outputs =
            cJSON_GetObjectItemCaseSensitive(node, "outputs");
        cJSON* physical_outputs = cJSON_CreateObject();
        cJSON* output_shapes = cJSON_CreateObject();
        cJSON* output_dtypes = cJSON_CreateObject();
        if (!physical_outputs || !output_shapes || !output_dtypes) {
            cJSON_Delete(physical_outputs);
            cJSON_Delete(output_shapes);
            cJSON_Delete(output_dtypes);
            status = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
        for (cJSON* output = logical_outputs->child; output;
             output = output->next) {
            cJSON* tensor = cJSON_GetObjectItemCaseSensitive(output, "tensor");
            cJSON* dtype = cJSON_GetObjectItemCaseSensitive(output, "dtype");
            cJSON* projected = vx_shape_minimum_projection(
                cJSON_GetObjectItemCaseSensitive(output, "shape"), dimensions);
            if (!output->string || !cJSON_IsString(tensor) ||
                !tensor->valuestring || !cJSON_IsString(dtype) ||
                !dtype->valuestring || !projected) {
                cJSON_Delete(projected);
                cJSON_Delete(physical_outputs);
                cJSON_Delete(output_shapes);
                cJSON_Delete(output_dtypes);
                goto done;
            }
            if (!cJSON_AddStringToObject(physical_outputs, output->string,
                                         tensor->valuestring) ||
                !cJSON_AddItemToObject(output_shapes, output->string,
                                       projected)) {
                cJSON_Delete(projected);
                cJSON_Delete(physical_outputs);
                cJSON_Delete(output_shapes);
                cJSON_Delete(output_dtypes);
                goto done;
            }
            projected = NULL; /* output_shapes owns it now. */
            if (!cJSON_AddStringToObject(output_dtypes, output->string,
                                         dtype->valuestring)) {
                cJSON_Delete(physical_outputs);
                cJSON_Delete(output_shapes);
                cJSON_Delete(output_dtypes);
                goto done;
            }
        }
        if (!cJSON_ReplaceItemInObjectCaseSensitive(node, "outputs",
                                                     physical_outputs)) {
            cJSON_Delete(physical_outputs);
            cJSON_Delete(output_shapes);
            cJSON_Delete(output_dtypes);
            goto done;
        }
        cJSON_AddItemToObject(node, "outputs_shape", output_shapes);
        cJSON_AddItemToObject(node, "outputs_dtype", output_dtypes);
    }
    cJSON_DeleteItemFromObjectCaseSensitive(root, "dimensions");
    /* "banks" is kept: vx_runtime_load_model checks residency requests against
     * it, and the engine needs it to know which weights are sliceable. */
    encoded = cJSON_PrintUnformatted(root);
    if (!encoded) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    status = vx_text_snapshot(encoded, out_path);
done:
    free(encoded);
    cJSON_Delete(root);
    return status;
}

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
/* Tests that exercise full-profile engine internals still enter through the
 * canonical bounded graph contract.  Keep the concrete bootstrap format
 * private by exposing only this test-build bridge, never a legacy graph input
 * path or a production symbol. */
int vx_public_api_test_private_engine_init(const char* graph_path,
                                           const char* weights_path) {
    char* lowered_graph_path = NULL;
    VxStatus status = vx_graph_lower_private_bootstrap(
        graph_path, &lowered_graph_path);
    int result;
    if (status != VX_STATUS_OK) return -1;
    result = volvoxai_engine_init(lowered_graph_path, weights_path);
    vx_snapshot_path_release(lowered_graph_path);
    return result;
}
#endif

static VxStatus vx_logical_tensors_load(const VxModel* model,
                                        VxDeclaredTensor** out_tensors,
                                        size_t* out_count) {
    const cJSON* root = NULL;
    const cJSON* dimensions;
    const cJSON* inputs;
    const cJSON* nodes;
    VxDeclaredTensor* tensors = NULL;
    size_t count = 0;
    size_t cursor = 0;
    VxStatus status = VX_STATUS_INVALID_GRAPH;
    if (!model || !out_tensors || !out_count) return VX_STATUS_INVALID_ARGUMENT;
    *out_tensors = NULL;
    *out_count = 0;
    root = model->logical_graph;
    if (!root || !vx_json_object_keys_unique_recursive(root) ||
        !vx_graph_schema_valid(root)) goto done;
    dimensions = cJSON_GetObjectItemCaseSensitive(root, "dimensions");
    inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    count = (size_t)cJSON_GetArraySize(inputs);
    for (const cJSON* node = nodes->child; node; node = node->next) {
        const cJSON* outputs =
            cJSON_GetObjectItemCaseSensitive(node, "outputs");
        size_t output_count = (size_t)cJSON_GetArraySize(outputs);
        if (output_count > MAXT - count) goto done;
        count += output_count;
    }
    if (count > MAXT) goto done;
    tensors = (VxDeclaredTensor*)calloc(count, sizeof(*tensors));
    if (!tensors) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    for (const cJSON* input = inputs->child; input;
         input = input->next, cursor++) {
        int parsed = vx_declared_tensor_set(
            &tensors[cursor], input->string,
            cJSON_GetObjectItemCaseSensitive(input, "shape"),
            cJSON_GetObjectItemCaseSensitive(input, "dtype"), dimensions);
        if (parsed != 0) {
            status = parsed == -2 ? VX_STATUS_OUT_OF_MEMORY :
                                    VX_STATUS_INVALID_GRAPH;
            goto done;
        }
    }
    for (const cJSON* node = nodes->child; node; node = node->next) {
        const cJSON* outputs =
            cJSON_GetObjectItemCaseSensitive(node, "outputs");
        for (const cJSON* output = outputs->child; output;
             output = output->next, cursor++) {
            const cJSON* tensor =
                cJSON_GetObjectItemCaseSensitive(output, "tensor");
            int parsed = vx_declared_tensor_set(
                &tensors[cursor], tensor->valuestring,
                cJSON_GetObjectItemCaseSensitive(output, "shape"),
                cJSON_GetObjectItemCaseSensitive(output, "dtype"), dimensions);
            if (parsed != 0) {
                status = parsed == -2 ? VX_STATUS_OUT_OF_MEMORY :
                                        VX_STATUS_INVALID_GRAPH;
                goto done;
            }
            for (size_t prior = 0; prior < cursor; prior++)
                if (!strcmp(tensors[prior].name, tensors[cursor].name))
                    goto done;
        }
    }
    /* Output-only symbols are legal. They remain unbound until the first
     * canonical node inference that produces them, then participate in the
     * same graph-wide equality class as public-input symbols. */
    *out_tensors = tensors;
    *out_count = count;
    tensors = NULL;
    status = VX_STATUS_OK;
done:
    vx_declared_outputs_free(tensors, tensors ? count : 0u);
    return status;
}

#include "public_api_symbolic_dimensions.inc"

#include "public_api_native_gpu_domain.inc"

typedef struct VxBuiltinResourceDomain {
    size_t maximum_typed_scratch_bytes;
    uint64_t maximum_resident_bytes;
} VxBuiltinResourceDomain;

/* Compilation proves the declared domain from immutable logical metadata and
 * the generated native descriptor-predicate registry. The private minimum
 * projection remains a bootstrap/prepack validation only; it is not used as
 * evidence that other legal shapes are routable. */
#include "public_api_bounded_domain_proof.inc"
#include "public_api_independent_batch.inc"
#include "public_api_resolved_shape_plan.inc"
static char* vx_string_copy(const char* value) {
    size_t length;
    char* copy;
    if (!value) return NULL;
    length = strlen(value);
    copy = (char*)malloc(length + 1u);
    if (copy) memcpy(copy, value, length + 1u);
    return copy;
}

static void vx_model_bank_residency_clear(VxModel* model) {
    if (!model) return;
    if (model->bank_residency_slots) {
        for (size_t index = 0; index < model->bank_residency_count; index++)
            free(model->bank_residency_slots[index]);
    }
    free(model->bank_residency_slots);
    free(model->bank_residency);
    model->bank_residency_slots = NULL;
    model->bank_residency = NULL;
    model->bank_residency_count = 0;
}

static void vx_weight_revision_retain(VxWeightRevisionRecord* revision) {
    if (revision)
        atomic_fetch_add_explicit(&revision->references, 1,
                                  memory_order_relaxed);
}

static void vx_weight_revision_release(VxWeightRevisionRecord* revision) {
    if (!revision || atomic_fetch_sub_explicit(&revision->references, 1,
                                               memory_order_acq_rel) != 1)
        return;
    for (size_t index = 0; index < revision->path_count; index++)
        vx_snapshot_path_release(revision->paths[index]);
    free(revision->paths);
    free(revision);
}

static void vx_compiled_weight_store_retain(VxCompiledWeightStore* store) {
    if (store)
        atomic_fetch_add_explicit(&store->references, 1,
                                  memory_order_relaxed);
}

static void vx_compiled_weight_store_release(VxCompiledWeightStore* store) {
    if (!store || atomic_fetch_sub_explicit(&store->references, 1,
                                            memory_order_acq_rel) != 1)
        return;
    for (size_t index = 0; index < store->file_count; index++)
        safetensors_free(&store->files[index]);
    free(store);
}

static VxStatus vx_compiled_weight_store_create(
        const VxWeightRevisionRecord* revision,
        VxCompiledWeightStore** out_store) {
    VxCompiledWeightStore* store;
    if (!revision || !out_store || revision->path_count > MAX_WEIGHT_FILES)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_store = NULL;
    store = (VxCompiledWeightStore*)calloc(1, sizeof(*store));
    if (!store) return VX_STATUS_OUT_OF_MEMORY;
    atomic_init(&store->references, 1);
    store->allocated_bytes = sizeof(*store);
    for (size_t index = 0; index < revision->path_count; index++) {
        SafetensorsFile* file = &store->files[index];
        uint64_t bytes;
        if (safetensors_load(revision->paths[index], file) != 0) {
            vx_compiled_weight_store_release(store);
            return VX_STATUS_INVALID_GRAPH;
        }
        store->file_count = index + 1u;
        if (file->size < 0) {
            vx_compiled_weight_store_release(store);
            return VX_STATUS_INVALID_GRAPH;
        }
        bytes = (uint64_t)file->size + 1u;
        if ((uint64_t)file->tensor_count >
                UINT64_MAX / sizeof(*file->tensors) ||
            bytes > UINT64_MAX -
                (uint64_t)file->tensor_count * sizeof(*file->tensors)) {
            vx_compiled_weight_store_release(store);
            return VX_STATUS_OUT_OF_MEMORY;
        }
        bytes += (uint64_t)file->tensor_count * sizeof(*file->tensors);
        if (file->metadata_json) {
            size_t metadata_bytes = strlen(file->metadata_json) + 1u;
            if ((uint64_t)metadata_bytes > UINT64_MAX - bytes) {
                vx_compiled_weight_store_release(store);
                return VX_STATUS_OUT_OF_MEMORY;
            }
            bytes += (uint64_t)metadata_bytes;
        }
        if ((uint64_t)file->size > UINT64_MAX - store->raw_bytes ||
            bytes > UINT64_MAX - store->allocated_bytes) {
            vx_compiled_weight_store_release(store);
            return VX_STATUS_OUT_OF_MEMORY;
        }
        store->raw_bytes += (uint64_t)file->size;
        store->allocated_bytes += bytes;
    }
    *out_store = store;
    return VX_STATUS_OK;
}

static void vx_adapter_revision_retain(VxAdapterRevisionRecord* revision) {
    if (revision)
        atomic_fetch_add_explicit(&revision->references, 1,
                                  memory_order_relaxed);
}

static void vx_adapter_revision_release(VxAdapterRevisionRecord* revision) {
    if (!revision || atomic_fetch_sub_explicit(&revision->references, 1,
                                               memory_order_acq_rel) != 1)
        return;
    free(revision->name);
    vx_snapshot_path_release(revision->package_path);
    free(revision->version_name);
    free(revision);
}

static VxWeightRevisionRecord* vx_weight_revision_create(
    const char* const* paths,
    size_t path_count,
    uint64_t identity,
    uint64_t revision,
    VxStatus* out_status) {
    VxWeightRevisionRecord* record;
    uint64_t content_hash = UINT64_C(1469598103934665603);
    if (out_status) *out_status = VX_STATUS_INVALID_ARGUMENT;
    if (path_count > (size_t)MAX_WEIGHT_FILES || (path_count && !paths))
        return NULL;
    record = (VxWeightRevisionRecord*)calloc(1, sizeof(*record));
    if (!record) {
        if (out_status) *out_status = VX_STATUS_OUT_OF_MEMORY;
        return NULL;
    }
    atomic_init(&record->references, 1);
    record->identity = identity;
    record->revision = revision;
    content_hash = vx_revision_update(content_hash, &path_count,
                                      sizeof(path_count));
    if (path_count)
        record->paths = (char**)calloc(path_count, sizeof(*record->paths));
    if (path_count && !record->paths) goto oom;
    for (size_t index = 0; index < path_count; index++) {
        uint64_t file_hash;
        VxStatus snapshot_status;
        if (!paths[index] || !paths[index][0]) {
            if (out_status) *out_status = VX_STATUS_INVALID_ARGUMENT;
            vx_weight_revision_release(record);
            return NULL;
        }
        snapshot_status = vx_file_snapshot(paths[index], &record->paths[index],
                                           &file_hash);
        if (snapshot_status != VX_STATUS_OK) {
            if (out_status) *out_status = snapshot_status;
            vx_weight_revision_release(record);
            return NULL;
        }
        record->path_count = index + 1u;
        content_hash = vx_revision_update(content_hash, &index, sizeof(index));
        content_hash = vx_revision_update(content_hash, &file_hash,
                                          sizeof(file_hash));
    }
    record->path_count = path_count;
    record->content_hash = content_hash ? content_hash : UINT64_C(1);
    record->allocated_bytes = sizeof(*record) +
        path_count * sizeof(*record->paths);
    for (size_t index = 0; index < path_count; index++)
        record->allocated_bytes += strlen(record->paths[index]) + 1u;
    if (out_status) *out_status = VX_STATUS_OK;
    return record;
oom:
    if (out_status) *out_status = VX_STATUS_OUT_OF_MEMORY;
    vx_weight_revision_release(record);
    return NULL;
}

static VxAdapterRevisionRecord* vx_adapter_revision_create(
    const char* name,
    const char* package_path,
    const char* version_name,
    uint64_t identity,
    uint64_t revision,
    VxStatus* out_status) {
    VxAdapterRevisionRecord* record;
    VxStatus snapshot_status = VX_STATUS_OK;
    uint64_t content_hash = UINT64_C(1);
    if (out_status) *out_status = VX_STATUS_OUT_OF_MEMORY;
    if (!name || !name[0]) return NULL;
    record = (VxAdapterRevisionRecord*)calloc(1, sizeof(*record));
    if (!record) return NULL;
    atomic_init(&record->references, 1);
    record->identity = identity;
    record->revision = revision;
    record->name = vx_string_copy(name);
    if (package_path)
        snapshot_status = vx_file_snapshot(package_path, &record->package_path,
                                           &content_hash);
    record->content_hash = content_hash;
    record->version_name = version_name ? vx_string_copy(version_name) : NULL;
    if (snapshot_status != VX_STATUS_OK || !record->name ||
        (package_path && !record->package_path) ||
        (version_name && !record->version_name)) {
        if (out_status) *out_status = snapshot_status != VX_STATUS_OK
            ? snapshot_status : VX_STATUS_OUT_OF_MEMORY;
        vx_adapter_revision_release(record);
        return NULL;
    }
    record->allocated_bytes = sizeof(*record) + strlen(record->name) + 1u;
    if (record->package_path)
        record->allocated_bytes += strlen(record->package_path) + 1u;
    if (record->version_name)
        record->allocated_bytes += strlen(record->version_name) + 1u;
    if (out_status) *out_status = VX_STATUS_OK;
    return record;
}

static void vx_report_write(VxReport* report,
                            VxStatus status,
                            VxStage stage,
                            const char* backend,
                            const char* device,
                            const char* reason,
                            const char* message,
                            uint64_t execution_id) {
    size_t struct_size;
    if (!report || report->struct_size != sizeof(*report)) return;
    struct_size = report->struct_size;
    memset(report, 0, sizeof(*report));
    report->struct_size = struct_size;
    report->status = status;
    report->stage = stage;
    report->execution_id = execution_id;
    snprintf(report->backend, sizeof(report->backend), "%s", backend ? backend : "");
    snprintf(report->device, sizeof(report->device), "%s", device ? device : "");
    snprintf(report->reason, sizeof(report->reason), "%s", reason ? reason : "");
    snprintf(report->message, sizeof(report->message), "%s", message ? message : "");
}

#include "public_api_report_diagnostics.inc"

static void vx_model_source_view(const VxModel* model,
                                 const VxWeightRevisionRecord* weights,
                                 VxModelSource* source) {
    *source = (VxModelSource)VX_MODEL_SOURCE_INIT;
    source->struct_size = sizeof(*source);
    source->graph_path = model->graph_path;
    source->weight_paths = weights
        ? (const char* const*)weights->paths : NULL;
    source->weight_path_count = weights ? weights->path_count : 0;
    source->bank_residency = model->bank_residency;
    source->bank_residency_count = model->bank_residency_count;
}

static void vx_declared_tensor_spec(const VxDeclaredTensor* declared,
                                    VxTensorSpec* spec) {
    memset(spec, 0, sizeof(*spec));
    spec->struct_size = sizeof(*spec);
    spec->name = declared->name;
    spec->dtype = declared->dtype;
    spec->rank = declared->rank;
    spec->location = VX_MEMORY_HOST;
    for (uint32_t axis = 0; axis < declared->rank; axis++) {
        VxDimensionConstraint* dimension = &spec->dimensions[axis];
        dimension->struct_size = sizeof(*dimension);
        dimension->kind = declared->kinds[axis];
        dimension->symbol = declared->symbols[axis];
        dimension->min = declared->minimums[axis];
        dimension->max = declared->maximums[axis];
        dimension->multiple_of = declared->multiples[axis];
    }
}

static VxTensorSpec* vx_declared_tensor_specs(
    const VxDeclaredTensor* declared,
    size_t count) {
    VxTensorSpec* specs = count
        ? (VxTensorSpec*)calloc(count, sizeof(*specs)) : NULL;
    if (count && !specs) return NULL;
    for (size_t index = 0; index < count; index++)
        vx_declared_tensor_spec(&declared[index], &specs[index]);
    return specs;
}

static int vx_provider_attestation_valid(
    const VxModel* model,
    const VxBackendShapeDomainAttestation* attestation) {
    if (!model || !attestation ||
        attestation->struct_size != sizeof(*attestation) ||
        !attestation->graph_fingerprint ||
        !attestation->shape_domain_proof_identity ||
        strcmp(attestation->graph_fingerprint, model->graph_fingerprint) ||
        strcmp(attestation->shape_domain_proof_identity,
               model->shape_domain_proof_identity) ||
        attestation->maximum_tensor_bytes != model->maximum_tensor_bytes ||
        attestation->maximum_tensor_bytes >
            attestation->maximum_resident_bytes ||
        (attestation->has_resource_limit != 0 &&
         attestation->has_resource_limit != 1) ||
        (attestation->has_resource_limit &&
         attestation->maximum_resident_bytes >
             attestation->resource_limit_bytes)) return 0;
    return 1;
}

static int vx_policy_valid(const VxBackendPolicy* policy) {
    if (!policy || policy->struct_size != sizeof(*policy) ||
        (policy->mode != VX_BACKEND_PREFER &&
         policy->mode != VX_BACKEND_REQUIRE) ||
        (policy->operator_fallback != VX_OPERATOR_FALLBACK_ALLOW &&
         policy->operator_fallback != VX_OPERATOR_FALLBACK_FORBID) ||
        policy->backend_count > VX_MAX_BACKEND_CANDIDATES ||
        (policy->backend_count && !policy->backends) ||
        (policy->mode == VX_BACKEND_REQUIRE && policy->backend_count != 1u) ||
        (policy->mode == VX_BACKEND_PREFER && policy->backends == NULL &&
         policy->backend_count != 0u) ||
        (policy->mode == VX_BACKEND_PREFER && policy->backends != NULL &&
         policy->backend_count == 0u)) return 0;
    for (size_t index = 0; index < policy->backend_count; index++) {
        const char* name = policy->backends[index];
        size_t length;
        if (!name || !name[0]) return 0;
        length = strlen(name);
        if (length >= VX_BACKEND_NAME_CAPACITY || name[0] < 'a' || name[0] > 'z')
            return 0;
        for (size_t offset = 1; offset < length; offset++) {
            unsigned char c = (unsigned char)name[offset];
            if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                  c == '-')) return 0;
        }
        for (size_t previous = 0; previous < index; previous++)
            if (!strcmp(policy->backends[previous], name)) return 0;
    }
    return 1;
}

static int vx_builtin_backend_by_name(const char* name,
                                      VolvoxAIEngineBackend* backend) {
    if (!name || !backend) return 0;
    if (!strcmp(name, "cpu")) *backend = VOLVOXAI_BACKEND_CPU;
    else if (!strcmp(name, "vulkan")) *backend = VOLVOXAI_BACKEND_VULKAN;
    else if (!strcmp(name, "opengl")) *backend = VOLVOXAI_BACKEND_OPENGL;
    else if (!strcmp(name, "metal")) *backend = VOLVOXAI_BACKEND_METAL;
    else if (!strcmp(name, "cuda")) *backend = VOLVOXAI_BACKEND_CUDA;
    else return 0;
    return 1;
}

#if VOLVOXAI_ENABLE_TRAINING
static const char* vx_builtin_backend_name(VolvoxAIEngineBackend backend) {
    switch (backend) {
        case VOLVOXAI_BACKEND_CPU: return "cpu";
        case VOLVOXAI_BACKEND_VULKAN: return "vulkan";
        case VOLVOXAI_BACKEND_OPENGL: return "opengl";
        case VOLVOXAI_BACKEND_METAL: return "metal";
        case VOLVOXAI_BACKEND_CUDA: return "cuda";
        default: return NULL;
    }
}
#endif

static const char* vx_builtin_device_identity(const char* backend) {
    if (backend && !strcmp(backend, "cpu")) return "host";
    if (backend && !strcmp(backend, "vulkan")) return "builtin:vulkan";
    if (backend && !strcmp(backend, "opengl")) return "builtin:opengl";
    if (backend && !strcmp(backend, "metal")) return "builtin:metal";
    if (backend && !strcmp(backend, "cuda")) return "builtin:cuda";
    return "builtin:unknown";
}

static VxStatus vx_private_engine_load(VxEngineState* state,
                                       const VxModel* model,
                                       const VxWeightRevisionRecord* weights,
                                       const VxCompiledWeightStore* weight_store,
                                       const VxRuntimeOptions* options,
                                       VolvoxAIEngineBackend backend,
                                       const char* backend_name,
                                       VxReport* report,
                                       VxStage stage) {
    VolvoxAIEngineOptions core_options = {
        .backend = backend,
        .debug = options->debug ? 1 : 0,
        .cpu_threads = options->cpu_threads,
    };
    VxEngineStateScope scope;
    VolvoxAIEngineOptions selected_options = {0};
    char* lowered_graph_path = NULL;
    VxStatus lower_status;
    int configured = 0;
    int status;
    lower_status = vx_graph_lower_private_bootstrap(
        model->graph_path, &lowered_graph_path);
    if (lower_status != VX_STATUS_OK) {
        vx_report_write(report, lower_status, stage, backend_name, NULL,
                        lower_status == VX_STATUS_OUT_OF_MEMORY
                            ? "OUT_OF_MEMORY" : "GRAPH_LOWERING_FAILED",
                        lower_status == VX_STATUS_OUT_OF_MEMORY
                            ? "private graph lowering allocation failed"
                            : "closed v1 graph could not be lowered for the built-in engine",
                        0);
        return lower_status;
    }
    scope = vx_engine_state_scope_enter(state);
    volvoxai_engine_tensor_name_index_invalidate();
    status = volvoxai_engine_configure(&core_options);
    if (status == 0 &&
        volvoxai_engine_get_options(&selected_options) == 0 &&
        selected_options.backend == backend)
        configured = 1;
    else
        status = -1;
    if (status == 0) {
        /* Register residency before init: the engine slices the banks between
         * loading the weight files and building the graph. */
        for (size_t index = 0; status == 0 && index < model->bank_residency_count;
             index++) {
            const VxBankResidency* residency = &model->bank_residency[index];
            status = volvoxai_engine_add_bank_residency(
                residency->bank, residency->slots, residency->slot_count);
        }
    }
    if (status == 0) {
        if (weight_store) {
            if (!weights || weight_store->file_count != weights->path_count)
                status = -1;
            else
                status = volvoxai_engine_init_with_borrowed_weight_files(
                    lowered_graph_path, weight_store->files,
                    (const char* const*)weights->paths,
                    (int)weights->path_count);
        } else {
            status = volvoxai_engine_init_with_weight_files(
                lowered_graph_path,
                weights ? (const char* const*)weights->paths : NULL,
                weights ? (int)weights->path_count : 0);
        }
    }
    vx_engine_state_scope_leave(scope);
    vx_snapshot_path_release(lowered_graph_path);
    if (status != 0) {
        VxStatus failure = configured
            ? VX_STATUS_INVALID_GRAPH : VX_STATUS_BACKEND_UNAVAILABLE;
        vx_report_write(report, failure, stage, backend_name, NULL,
                        failure == VX_STATUS_BACKEND_UNAVAILABLE
                            ? "BACKEND_UNAVAILABLE" : "GRAPH_INITIALIZATION_FAILED",
                        failure == VX_STATUS_BACKEND_UNAVAILABLE
                            ? "requested built-in backend is unavailable"
                            : "native graph validation or initialization failed",
                        0);
        return failure;
    }
    vx_report_write(report, VX_STATUS_OK, stage, backend_name,
                    vx_builtin_device_identity(backend_name), "OK",
                    "native built-in graph initialized", 0);
    return VX_STATUS_OK;
}

static void vx_private_engine_unload(VxExecutionContext* context) {
    VxEngineStateScope scope;
    if (!context) return;
    vx_declared_outputs_free(context->logical_tensors,
                             context->logical_tensor_count);
    context->logical_tensors = NULL;
    context->logical_tensor_count = 0;
    free(context->logical_tensor_name_slots);
    context->logical_tensor_name_slots = NULL;
    context->logical_tensor_name_capacity = 0;
    free(context->committed_shape_signature);
    context->committed_shape_signature = NULL;
    if (context->engine_state_initialized) {
        if (context->engine_loaded) {
            scope = vx_engine_state_scope_enter(context->engine_state);
            if (context->decode_session) {
                volvoxai_engine_decode_session_destroy(context->decode_session);
                context->decode_session = NULL;
            }
            volvoxai_engine_tensor_name_index_invalidate();
            volvoxai_engine_tensor_name_index_rebuild();
            volvoxai_engine_shutdown();
            vx_engine_state_scope_leave(scope);
            context->engine_loaded = 0;
        }
        vx_engine_state_deinit(context->engine_state);
        context->engine_state_initialized = 0;
        free(context->engine_state);
        context->engine_state = NULL;
    }
    vx_compiled_weight_store_release(context->weight_store);
    context->weight_store = NULL;
}

#if VOLVOXAI_ENABLE_TRAINING
static VxStatus vx_private_weight_revision_validate(
    const VxModel* model,
    const VxWeightRevisionRecord* weights,
    VxReport* report) {
    VxEngineState* validation =
        (VxEngineState*)calloc(1, sizeof(*validation));
    VxStatus status;
    if (!validation || vx_engine_state_init(validation) != 0) {
        free(validation);
        return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                       VX_STAGE_MODEL_LOAD, "cpu", "OUT_OF_MEMORY",
                       "weight validation state allocation failed");
    }
    status = vx_private_engine_load(validation, model, weights, NULL,
                                    &model->runtime->options,
                                    VOLVOXAI_BACKEND_CPU, "cpu", report,
                                    VX_STAGE_MODEL_LOAD);
    {
        VxExecutionContext temporary = {0};
        temporary.engine_state = validation;
        temporary.engine_state_initialized = 1;
        /* Shutdown is the rollback path for both complete and partial loads. */
        temporary.engine_loaded = 1;
        vx_private_engine_unload(&temporary);
    }
    return status;
}
#endif

VxStatus vx_runtime_create(const VxRuntimeOptions* options,
                           VxRuntime** out_runtime,
                           VxReport* report) {
    VxRuntimeOptions resolved = VX_RUNTIME_OPTIONS_INIT;
    VxRuntime* runtime;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!out_runtime || (options && options->struct_size != sizeof(*options)) ||
        (options && (options->cpu_threads < 0 ||
                     (options->execution_mode != VX_EXECUTION_MODE_DIRECT &&
                      options->execution_mode != VX_EXECUTION_MODE_SCHEDULED) ||
                     !options->max_scheduled_requests ||
                     !options->max_scheduled_input_bytes ||
                     !options->max_unconsumed_results ||
                     !options->max_unconsumed_result_bytes)))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_RUNTIME_CREATE, NULL, "INVALID_OPTIONS",
                       "runtime options are invalid");
    *out_runtime = NULL;
    if (options) resolved = *options;
    runtime = (VxRuntime*)calloc(1, sizeof(*runtime));
    if (!runtime)
        return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                       VX_STAGE_RUNTIME_CREATE, NULL, "OUT_OF_MEMORY",
                       "runtime allocation failed");
    atomic_init(&runtime->references, 1);
    atomic_init(&runtime->next_object_identity, 1);
    atomic_init(&runtime->next_execution_identity, 1);
    if (pthread_mutex_init(&runtime->mutex, NULL) != 0) {
        free(runtime);
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_RUNTIME_CREATE,
                       NULL, "MUTEX_INIT_FAILED", "runtime mutex initialization failed");
    }
    if (pthread_mutex_init(&runtime->result_budget_mutex, NULL) != 0) {
        pthread_mutex_destroy(&runtime->mutex);
        free(runtime);
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_RUNTIME_CREATE,
                       NULL, "MUTEX_INIT_FAILED",
                       "runtime result-budget mutex initialization failed");
    }
    if (pthread_cond_init(&runtime->direct_condition, NULL) != 0) {
        pthread_mutex_destroy(&runtime->result_budget_mutex);
        pthread_mutex_destroy(&runtime->mutex);
        free(runtime);
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_RUNTIME_CREATE,
                       NULL, "CONDITION_INIT_FAILED",
                       "runtime direct condition initialization failed");
    }
    runtime->providers = vx_provider_registry_create();
    if (!runtime->providers) {
        pthread_cond_destroy(&runtime->direct_condition);
        pthread_mutex_destroy(&runtime->result_budget_mutex);
        pthread_mutex_destroy(&runtime->mutex);
        free(runtime);
        return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                       VX_STAGE_RUNTIME_CREATE, NULL, "OUT_OF_MEMORY",
                       "provider registry allocation failed");
    }
    runtime->options = resolved;
    runtime->identity = vx_runtime_identity(runtime);
    *out_runtime = runtime;
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_RUNTIME_CREATE, NULL, NULL,
                    "OK", "runtime created", 0);
    vx_report_set_lineage(report, runtime, NULL, NULL, NULL);
    return VX_STATUS_OK;
}

void vx_runtime_retain(VxRuntime* runtime) {
    if (runtime) atomic_fetch_add_explicit(&runtime->references, 1, memory_order_relaxed);
}

void vx_runtime_release(VxRuntime* runtime) {
    if (!runtime || atomic_fetch_sub_explicit(&runtime->references, 1,
                                              memory_order_acq_rel) != 1) return;
    if (runtime->coordinator && runtime->coordinator->worker_started &&
        pthread_equal(pthread_self(), runtime->coordinator->worker)) {
        VxRuntimeCoordinator* coordinator = runtime->coordinator;
        pthread_mutex_lock(&coordinator->mutex);
        coordinator->stopping = 1;
        coordinator->destroy_runtime_on_worker_exit = 1;
        pthread_cond_broadcast(&coordinator->condition);
        pthread_mutex_unlock(&coordinator->mutex);
        return;
    }
    pthread_mutex_lock(&runtime->mutex);
    runtime->closed = 1;
    while (runtime->active_direct_calls)
        pthread_cond_wait(&runtime->direct_condition, &runtime->mutex);
    pthread_mutex_unlock(&runtime->mutex);
    vx_runtime_coordinator_destroy(runtime->coordinator);
    vx_provider_registry_destroy(runtime->providers);
    pthread_cond_destroy(&runtime->direct_condition);
    pthread_mutex_destroy(&runtime->result_budget_mutex);
    pthread_mutex_destroy(&runtime->mutex);
    free(runtime);
}

VxStatus vx_runtime_register_provider(VxRuntime* runtime,
                                      const VxBackendProvider* provider,
                                      VxReport* report) {
    VxStatus status;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!runtime)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_RUNTIME_CREATE, NULL, "INVALID_RUNTIME",
                       "runtime is NULL");
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->closed) {
        pthread_mutex_unlock(&runtime->mutex);
        status = vx_fail(report, VX_STATUS_HANDLE_DISPOSED,
                         VX_STAGE_RUNTIME_CREATE, NULL, "HANDLE_DISPOSED",
                         "runtime is closed");
        vx_report_set_lineage(report, runtime, NULL, NULL, NULL);
        return status;
    }
    status = vx_provider_registry_register(runtime->providers,
                                           &runtime->options,
                                           provider, report);
    pthread_mutex_unlock(&runtime->mutex);
    vx_report_set_lineage(report, runtime, NULL, NULL, NULL);
    return status;
}

VxStatus vx_runtime_close(VxRuntime* runtime, VxReport* report) {
    VxRuntimeCoordinator* coordinator;
    VxStatus status;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!runtime)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_CLOSE,
                       NULL, "INVALID_RUNTIME", "runtime is NULL");
    if (vx_runtime_direct_thread_active(runtime))
        return vx_fail(report, VX_STATUS_BUSY, VX_STAGE_CLOSE, NULL, "BUSY",
                       "runtime close cannot wait from an active direct call");
    pthread_mutex_lock(&runtime->mutex);
    runtime->closed = 1;
    while (runtime->active_direct_calls)
        pthread_cond_wait(&runtime->direct_condition, &runtime->mutex);
    coordinator = runtime->coordinator;
    pthread_mutex_unlock(&runtime->mutex);
    status = vx_runtime_coordinator_close(coordinator);
    if (status != VX_STATUS_OK) {
        status = vx_fail(report, status, VX_STAGE_CLOSE, NULL,
                         status == VX_STATUS_BUSY ? "BUSY" : "INTERNAL",
                         status == VX_STATUS_BUSY
                             ? "runtime close cannot wait from its coordinator worker"
                             : "runtime coordinator shutdown failed");
        vx_report_set_lineage(report, runtime, NULL, NULL, NULL);
        return status;
    }
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_CLOSE, NULL, NULL, "OK",
                    "runtime closed", 0);
    vx_report_set_lineage(report, runtime, NULL, NULL, NULL);
    return VX_STATUS_OK;
}

VxStatus vx_runtime_load_model(VxRuntime* runtime,
                               const VxModelSource* source,
                               VxModel** out_model,
                               VxReport* report) {
    VxModel* model;
    VxWeightRevisionRecord* weights = NULL;
    VxAdapterRevisionRecord* base_adapter = NULL;
    VxStatus envelope_status;
    VxStatus weight_status;
    VxStatus adapter_status;
    VxStatus graph_snapshot_status;
    uint64_t graph_revision;
    char* graph_snapshot = NULL;
    VxDeclaredTensor* declared_inputs = NULL;
    size_t input_count = 0;
    VxDeclaredOutput* declared_outputs = NULL;
    size_t output_count = 0;
    VxDeclaredTensor* domain_tensors = NULL;
    size_t domain_tensor_count = 0;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!runtime || !source || source->struct_size != sizeof(*source) ||
        !source->graph_path || !source->graph_path[0] || !out_model ||
        source->weight_path_count > (size_t)MAX_WEIGHT_FILES ||
        (source->weight_path_count && !source->weight_paths) ||
        (source->bank_residency_count && !source->bank_residency) ||
        source->bank_residency_count >
            SIZE_MAX / sizeof(*source->bank_residency) ||
        source->bank_residency_count >
            SIZE_MAX / sizeof(uint32_t*))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_MODEL_LOAD,
                       NULL, "INVALID_MODEL_SOURCE", "model source is invalid");
    for (size_t index = 0; index < source->bank_residency_count; index++) {
        const VxBankResidency* residency = &source->bank_residency[index];
        if (residency->struct_size != sizeof(*residency) || !residency->bank ||
            !residency->bank[0] || !residency->slots || !residency->slot_count ||
            residency->slot_count > SIZE_MAX / sizeof(*residency->slots))
            return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_MODEL_LOAD,
                           NULL, "INVALID_BANK_RESIDENCY",
                           "bank residency entry is invalid");
        for (size_t prior = 0; prior < index; prior++)
            if (!strcmp(source->bank_residency[prior].bank, residency->bank))
                return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                               VX_STAGE_MODEL_LOAD, NULL,
                               "INVALID_BANK_RESIDENCY",
                               "bank residency entries must name unique banks");
        for (size_t slot = 1; slot < residency->slot_count; slot++)
            if (residency->slots[slot] <= residency->slots[slot - 1])
                return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                               VX_STAGE_MODEL_LOAD, NULL,
                               "INVALID_BANK_RESIDENCY",
                               "bank residency slots must be strictly ascending");
    }

    if (!vx_is_canonical_graph_path(source->graph_path))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_MODEL_LOAD,
                       NULL, "INVALID_GRAPH_PATH",
                       "graph_path must name graph.json or a named *.graph.json document");
    *out_model = NULL;
    pthread_mutex_lock(&runtime->mutex);
    int runtime_closed = runtime->closed;
    pthread_mutex_unlock(&runtime->mutex);
    if (runtime_closed)
        return vx_fail(report, VX_STATUS_HANDLE_DISPOSED, VX_STAGE_MODEL_LOAD,
                       NULL, "HANDLE_DISPOSED", "runtime is closed");
    graph_snapshot_status = vx_file_snapshot(
        source->graph_path, &graph_snapshot, &graph_revision);
    if (graph_snapshot_status != VX_STATUS_OK)
        return vx_fail(
            report, graph_snapshot_status, VX_STAGE_MODEL_LOAD, NULL,
            graph_snapshot_status == VX_STATUS_OUT_OF_MEMORY
                ? "OUT_OF_MEMORY" : "GRAPH_NOT_READABLE",
            graph_snapshot_status == VX_STATUS_OUT_OF_MEMORY
                ? "graph snapshot allocation failed"
                : "graph package file could not be snapshotted");
    envelope_status = vx_graph_package_validate(
        graph_snapshot, NULL, 0, 0, NULL, 0,
        &declared_inputs, &input_count,
        &declared_outputs, &output_count);
    if (envelope_status != VX_STATUS_OK) {
        vx_snapshot_path_release(graph_snapshot);
        if (envelope_status == VX_STATUS_OUT_OF_MEMORY)
            return vx_fail(report, envelope_status, VX_STAGE_MODEL_LOAD, NULL,
                           "OUT_OF_MEMORY", "graph package validation allocation failed");
        if (envelope_status == VX_STATUS_IO_ERROR)
            return vx_fail(report, envelope_status, VX_STAGE_MODEL_LOAD, NULL,
                           "GRAPH_NOT_READABLE", "graph package file could not be read");
        return vx_fail(report, VX_STATUS_INVALID_GRAPH, VX_STAGE_MODEL_LOAD, NULL,
                       "INVALID_GRAPH_CONTRACT",
                       "graph.json must use the volvox-graph/v1 contract");
    }
    vx_declared_outputs_free(declared_inputs, input_count);
    declared_inputs = NULL;
    input_count = 0;
    vx_declared_outputs_free(declared_outputs, output_count);
    declared_outputs = NULL;
    output_count = 0;
    model = (VxModel*)calloc(1, sizeof(*model));
    if (!model) {
        vx_snapshot_path_release(graph_snapshot);
        return vx_fail(report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_MODEL_LOAD,
                       NULL, "OUT_OF_MEMORY", "model allocation failed");
    }
    atomic_init(&model->references, 1);
    if (pthread_mutex_init(&model->revision_mutex, NULL) != 0) {
        vx_snapshot_path_release(graph_snapshot);
        free(model);
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_MODEL_LOAD, NULL,
                       "MUTEX_INIT_FAILED",
                       "model revision mutex initialization failed");
    }
    model->graph_path = graph_snapshot;
    graph_snapshot = NULL;
    model->logical_graph = vx_json_file_load(model->graph_path);
    if (!model->logical_graph) goto oom;
    if (source->bank_residency_count) {
        const cJSON* declared =
            cJSON_GetObjectItemCaseSensitive(model->logical_graph, "banks");
        const cJSON* dimensions =
            cJSON_GetObjectItemCaseSensitive(model->logical_graph,
                                             "dimensions");
        model->bank_residency = (VxBankResidency*)calloc(
            source->bank_residency_count, sizeof(*model->bank_residency));
        model->bank_residency_slots = (uint32_t**)calloc(
            source->bank_residency_count, sizeof(*model->bank_residency_slots));
        if (!model->bank_residency || !model->bank_residency_slots) goto oom;
        for (size_t index = 0; index < source->bank_residency_count; index++) {
            const VxBankResidency* requested = &source->bank_residency[index];
            const cJSON* declared_bank = declared
                ? cJSON_GetObjectItemCaseSensitive(declared, requested->bank)
                : NULL;
            int64_t bank_minimum;
            int64_t bank_maximum;
            int64_t bank_multiple;
            uint32_t* slots;
            if (!declared_bank || !cJSON_IsString(declared_bank) ||
                !declared_bank->valuestring ||
                !vx_dimension_json(dimensions, declared_bank->valuestring,
                                   &bank_minimum, &bank_maximum,
                                   &bank_multiple) ||
                (uint64_t)requested->slots[requested->slot_count - 1u] >=
                    (uint64_t)bank_maximum)
                goto invalid_bank_residency;
            slots = (uint32_t*)malloc(requested->slot_count * sizeof(uint32_t));
            if (!slots) goto oom;
            memcpy(slots, requested->slots,
                   requested->slot_count * sizeof(uint32_t));
            model->bank_residency_slots[index] = slots;
            model->bank_residency[index] = *requested;
            model->bank_residency[index].struct_size =
                sizeof(model->bank_residency[index]);
            /* The parsed graph owns this canonical key for the whole model
             * lifetime; never retain the caller's transient bank string. */
            model->bank_residency[index].bank = declared_bank->string;
            model->bank_residency[index].slots = slots;
            model->bank_residency_count = index + 1;
        }
    }
    model->runtime = runtime;
    model->identity = vx_next_object_identity(runtime);
    model->graph_identity = vx_next_object_identity(runtime);
    model->graph_revision = graph_revision;
    model->weight_identity = vx_next_object_identity(runtime);
    weights = vx_weight_revision_create(source->weight_paths,
                                        source->weight_path_count,
                                        model->weight_identity, 1,
                                        &weight_status);
    if (!weights) {
        if (weight_status == VX_STATUS_IO_ERROR) goto weights_unreadable;
        if (weight_status == VX_STATUS_INVALID_ARGUMENT)
            goto weights_invalid;
        goto oom;
    }
    envelope_status = vx_graph_package_validate(
        model->graph_path, (const char* const*)weights->paths,
        weights->path_count, 1,
        model->bank_residency, model->bank_residency_count,
        &declared_inputs, &input_count,
        &declared_outputs, &output_count);
    if (envelope_status != VX_STATUS_OK) {
        vx_weight_revision_release(weights);
        weights = NULL;
        vx_declared_outputs_free(declared_inputs, input_count);
        vx_declared_outputs_free(declared_outputs, output_count);
        vx_model_bank_residency_clear(model);
        cJSON_Delete(model->logical_graph);
        vx_snapshot_path_release(model->graph_path);
        pthread_mutex_destroy(&model->revision_mutex);
        free(model);
        if (envelope_status == VX_STATUS_OUT_OF_MEMORY)
            return vx_fail(report, envelope_status, VX_STAGE_MODEL_LOAD, NULL,
                           "OUT_OF_MEMORY",
                           "output descriptor allocation failed");
        if (envelope_status == VX_STATUS_IO_ERROR)
            return vx_fail(report, envelope_status, VX_STAGE_MODEL_LOAD, NULL,
                           "GRAPH_NOT_READABLE",
                           "graph package snapshot could not be read");
        if (envelope_status == VX_STATUS_INVALID_ARGUMENT)
            return vx_fail(report, envelope_status, VX_STAGE_MODEL_LOAD, NULL,
                           "INVALID_BANK_RESIDENCY",
                           "bank residency exceeds the supplied bank extent");
        return vx_fail(report, VX_STATUS_INVALID_GRAPH, VX_STAGE_MODEL_LOAD,
                       NULL, "INVALID_GRAPH_CONTRACT",
                       "every graph output must have one canonical execution descriptor");
    }
    model->inputs = declared_inputs;
    model->input_count = input_count;
    declared_inputs = NULL;
    input_count = 0;
    model->outputs = declared_outputs;
    model->output_count = output_count;
    declared_outputs = NULL;
    output_count = 0;
    envelope_status = vx_logical_tensors_load(
        model, &domain_tensors, &domain_tensor_count);
    if (envelope_status != VX_STATUS_OK) {
        if (envelope_status == VX_STATUS_OUT_OF_MEMORY) goto oom;
        vx_weight_revision_release(weights);
        vx_declared_outputs_free(model->inputs, model->input_count);
        vx_declared_outputs_free(model->outputs, model->output_count);
        vx_model_bank_residency_clear(model);
        cJSON_Delete(model->logical_graph);
        vx_snapshot_path_release(model->graph_path);
        pthread_mutex_destroy(&model->revision_mutex);
        free(model);
        return vx_fail(report, VX_STATUS_INVALID_GRAPH,
                       VX_STAGE_MODEL_LOAD, NULL,
                       "INVALID_LOGICAL_DOMAIN",
                       "logical activation descriptors do not form one closed bounded domain");
    }
    for (size_t index = 0; index < domain_tensor_count; index++)
        if (domain_tensors[index].maximum_byte_size >
            model->maximum_tensor_bytes)
            model->maximum_tensor_bytes =
                (uint64_t)domain_tensors[index].maximum_byte_size;
    vx_declared_outputs_free(domain_tensors, domain_tensor_count);
    domain_tensors = NULL;
    domain_tensor_count = 0;
    base_adapter = vx_adapter_revision_create(
        "__base__", NULL, NULL, vx_next_object_identity(runtime), 1,
        &adapter_status);
    if (!base_adapter) goto oom;
    model->adapters = base_adapter;
    atomic_init(&model->current_weights, weights);
    atomic_init(&model->current_adapter, base_adapter);
    model->allocated_bytes = sizeof(*model) + strlen(model->graph_path) + 1u;
    snprintf(model->graph_fingerprint, sizeof(model->graph_fingerprint),
             "fnv1a64:%016" PRIx64, model->graph_revision);
    snprintf(model->shape_domain_proof_identity,
             sizeof(model->shape_domain_proof_identity),
             "%s:%016" PRIx64,
             VX_BACKEND_SHAPE_PROOF_PROTOCOL, model->graph_revision);
    model->allocated_bytes += model->input_count * sizeof(*model->inputs);
    for (size_t index = 0; index < model->input_count; index++) {
        model->allocated_bytes += strlen(model->inputs[index].name) + 1u;
        if (model->inputs[index].maximum_byte_size > model->maximum_tensor_bytes)
            model->maximum_tensor_bytes =
                (uint64_t)model->inputs[index].maximum_byte_size;
        for (uint32_t axis = 0; axis < model->inputs[index].rank; axis++)
            if (model->inputs[index].symbols[axis])
                model->allocated_bytes +=
                    strlen(model->inputs[index].symbols[axis]) + 1u;
    }
    model->allocated_bytes += model->output_count * sizeof(*model->outputs);
    for (size_t index = 0; index < model->output_count; index++) {
        model->allocated_bytes += strlen(model->outputs[index].name) + 1u;
        if (model->outputs[index].maximum_byte_size > model->maximum_tensor_bytes)
            model->maximum_tensor_bytes =
                (uint64_t)model->outputs[index].maximum_byte_size;
        for (uint32_t axis = 0; axis < model->outputs[index].rank; axis++)
            if (model->outputs[index].symbols[axis])
                model->allocated_bytes +=
                    strlen(model->outputs[index].symbols[axis]) + 1u;
    }
    if (model->bank_residency_count) {
        if (!vx_native_gpu_accumulate_resource_bytes(
                &model->allocated_bytes,
                sizeof(*model->bank_residency),
                (uint64_t)model->bank_residency_count) ||
            !vx_native_gpu_accumulate_resource_bytes(
                &model->allocated_bytes,
                sizeof(*model->bank_residency_slots),
                (uint64_t)model->bank_residency_count)) goto oom;
        for (size_t index = 0; index < model->bank_residency_count; index++)
            if (!vx_native_gpu_accumulate_resource_bytes(
                    &model->allocated_bytes,
                    sizeof(*model->bank_residency[index].slots),
                    (uint64_t)model->bank_residency[index].slot_count))
                goto oom;
    }
    vx_runtime_retain(runtime);
    *out_model = model;
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_MODEL_LOAD, NULL, NULL,
                    "OK", "model source retained", 0);
    vx_report_set_lineage(report, NULL, model, NULL, NULL);
    return VX_STATUS_OK;
oom:
    vx_declared_outputs_free(domain_tensors, domain_tensor_count);
    vx_weight_revision_release(weights);
    vx_adapter_revision_release(base_adapter);
    vx_declared_outputs_free(declared_inputs, input_count);
    vx_declared_outputs_free(declared_outputs, output_count);
    vx_declared_outputs_free(model->inputs, model->input_count);
    vx_declared_outputs_free(model->outputs, model->output_count);
    vx_model_bank_residency_clear(model);
    cJSON_Delete(model->logical_graph);
    vx_snapshot_path_release(model->graph_path);
    pthread_mutex_destroy(&model->revision_mutex);
    free(model);
    return vx_fail(report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_MODEL_LOAD, NULL,
                   "OUT_OF_MEMORY", "model source copy failed");
weights_unreadable:
    vx_declared_outputs_free(declared_inputs, input_count);
    vx_declared_outputs_free(declared_outputs, output_count);
    vx_declared_outputs_free(model->inputs, model->input_count);
    vx_declared_outputs_free(model->outputs, model->output_count);
    vx_model_bank_residency_clear(model);
    cJSON_Delete(model->logical_graph);
    vx_snapshot_path_release(model->graph_path);
    pthread_mutex_destroy(&model->revision_mutex);
    free(model);
    return vx_fail(report, VX_STATUS_IO_ERROR, VX_STAGE_MODEL_LOAD, NULL,
                   "WEIGHTS_NOT_READABLE",
                   "a weight revision file could not be read");
weights_invalid:
    vx_declared_outputs_free(declared_inputs, input_count);
    vx_declared_outputs_free(declared_outputs, output_count);
    vx_declared_outputs_free(model->inputs, model->input_count);
    vx_declared_outputs_free(model->outputs, model->output_count);
    vx_model_bank_residency_clear(model);
    cJSON_Delete(model->logical_graph);
    vx_snapshot_path_release(model->graph_path);
    pthread_mutex_destroy(&model->revision_mutex);
    free(model);
    return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_MODEL_LOAD,
                   NULL, "INVALID_MODEL_SOURCE",
                   "a weight source path is empty");
invalid_bank_residency:
    vx_declared_outputs_free(declared_inputs, input_count);
    vx_declared_outputs_free(declared_outputs, output_count);
    vx_model_bank_residency_clear(model);
    cJSON_Delete(model->logical_graph);
    vx_snapshot_path_release(model->graph_path);
    pthread_mutex_destroy(&model->revision_mutex);
    free(model);
    return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_MODEL_LOAD,
                   NULL, "INVALID_BANK_RESIDENCY",
                   "bank residency does not match the graph's declared bank domain");
}

#include "public_api_model_lifecycle.inc"

#include "public_api_execution_context.inc"

#include "public_api_execute_result.inc"
}

#include "public_api_runtime_requests.inc"
