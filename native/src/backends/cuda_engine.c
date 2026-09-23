#include "cuda_engine.h"
#include "qlinear_multiplier.h"
#include "runtime_state.h"
#include "profiling.h"

#include "embedded_cuda_ptx.h"
#if VOLVOXAI_ENABLE_TRAINING
#include "embedded_cuda_training_ptx.h"
#endif

#include <errno.h>
#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

/* Narrow declarations for the stable CUDA Driver API.  Keeping these private
 * avoids a CUDA SDK header or link dependency in the native runtime. */
typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef struct CUgraph_st* CUgraph;
typedef struct CUgraphExec_st* CUgraphExec;
typedef struct CUevent_st* CUevent;
typedef uintptr_t CUdeviceptr;

enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_NOT_READY = 600,
    CU_STREAM_NON_BLOCKING = 1,
    CU_EVENT_DISABLE_TIMING = 2,
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
};

#ifdef _WIN32
#define CUDAAPI __stdcall
#else
#define CUDAAPI
#endif

typedef CUresult (CUDAAPI *PFN_cuInit)(unsigned int);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetCount)(int*);
typedef CUresult (CUDAAPI *PFN_cuDeviceGet)(CUdevice*, int);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetName)(char*, int, CUdevice);
typedef CUresult (CUDAAPI *PFN_cuDeviceComputeCapability)(int*, int*, CUdevice);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetAttribute)(int*, int, CUdevice);
typedef CUresult (CUDAAPI *PFN_cuDeviceTotalMem)(size_t*, CUdevice);
typedef CUresult (CUDAAPI *PFN_cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
typedef CUresult (CUDAAPI *PFN_cuDevicePrimaryCtxRelease)(CUdevice);
typedef CUresult (CUDAAPI *PFN_cuCtxGetCurrent)(CUcontext*);
typedef CUresult (CUDAAPI *PFN_cuCtxSetCurrent)(CUcontext);
typedef CUresult (CUDAAPI *PFN_cuCtxSynchronize)(void);
typedef CUresult (CUDAAPI *PFN_cuStreamCreate)(CUstream*, unsigned int);
typedef CUresult (CUDAAPI *PFN_cuStreamDestroy)(CUstream);
typedef CUresult (CUDAAPI *PFN_cuStreamSynchronize)(CUstream);
typedef CUresult (CUDAAPI *PFN_cuStreamWaitEvent)(CUstream, CUevent, unsigned int);
typedef CUresult (CUDAAPI *PFN_cuStreamGetCtx)(CUstream, CUcontext*);
typedef CUresult (CUDAAPI *PFN_cuStreamIsCapturing)(CUstream, int*);
typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture)(CUstream, int);
typedef CUresult (CUDAAPI *PFN_cuStreamEndCapture)(CUstream, CUgraph*);
typedef CUresult (CUDAAPI *PFN_cuGraphInstantiateWithFlags)(CUgraphExec*,
                                                            CUgraph,
                                                            unsigned long long);
typedef CUresult (CUDAAPI *PFN_cuGraphLaunch)(CUgraphExec, CUstream);
typedef CUresult (CUDAAPI *PFN_cuGraphDestroy)(CUgraph);
typedef CUresult (CUDAAPI *PFN_cuGraphExecDestroy)(CUgraphExec);
typedef CUresult (CUDAAPI *PFN_cuModuleLoadDataEx)(CUmodule*, const void*,
                                                   unsigned int, void*, void**);
typedef CUresult (CUDAAPI *PFN_cuModuleUnload)(CUmodule);
typedef CUresult (CUDAAPI *PFN_cuModuleGetFunction)(CUfunction*, CUmodule,
                                                    const char*);
typedef CUresult (CUDAAPI *PFN_cuMemAlloc)(CUdeviceptr*, size_t);
typedef CUresult (CUDAAPI *PFN_cuMemFree)(CUdeviceptr);
typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream);
typedef CUresult (CUDAAPI *PFN_cuMemAllocHost)(void**, size_t);
typedef CUresult (CUDAAPI *PFN_cuMemFreeHost)(void*);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
typedef CUresult (CUDAAPI *PFN_cuPointerGetAttribute)(void*, int, CUdeviceptr);
typedef CUresult (CUDAAPI *PFN_cuMemGetAddressRange)(CUdeviceptr*, size_t*, CUdeviceptr);
typedef CUresult (CUDAAPI *PFN_cuLaunchKernel)(CUfunction,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, CUstream, void**, void**);
typedef CUresult (CUDAAPI *PFN_cuGetErrorName)(CUresult, const char**);
typedef CUresult (CUDAAPI *PFN_cuEventCreate)(CUevent*, unsigned int);
typedef CUresult (CUDAAPI *PFN_cuEventRecord)(CUevent, CUstream);
typedef CUresult (CUDAAPI *PFN_cuEventDestroy)(CUevent);
typedef CUresult (CUDAAPI *PFN_cuEventQuery)(CUevent);
typedef CUresult (CUDAAPI *PFN_cuEventSynchronize)(CUevent);
typedef CUresult (CUDAAPI *PFN_cuEventElapsedTime)(float*, CUevent, CUevent);

enum {
#define VOLVOXAI_CUDA_FORWARD_FUNCTION(requirement, handle, symbol) \
    CUDA_FORWARD_REGISTRY_HANDLE_##handle,
#include "cuda/host/cuda_forward_function_registry_host.inc"
#undef VOLVOXAI_CUDA_FORWARD_FUNCTION
    CUDA_FORWARD_FUNCTION_COUNT
};
_Static_assert(CUDA_FORWARD_FUNCTION_COUNT == 91,
               "CUDA forward function registry coverage");

#if VOLVOXAI_ENABLE_TRAINING
enum {
#define VOLVOXAI_CUDA_TRAINING_FUNCTION( \
        requirement, handle, symbol, operation) \
    CUDA_TRAINING_REGISTRY_HANDLE_##handle,
#include "cuda/host/cuda_training_function_registry_host.inc"
#undef VOLVOXAI_CUDA_TRAINING_FUNCTION
    CUDA_TRAINING_FUNCTION_COUNT
};
_Static_assert(CUDA_TRAINING_FUNCTION_COUNT == 79,
               "CUDA training function registry coverage");

#endif

/* Each serialized engine owns a submission queue and pinned transfer slab.
 * Operations on retained buffers outside an engine use a separate, serialized
 * interop queue. Neither queue holds the device lifetime lock while waiting. */
typedef struct {
    CUstream stream;
    CUevent input_handoff;
    void* transfer_host;
    size_t transfer_capacity;
    CUresult transfer_error;
    PFN_cuLaunchKernel launch;
    atomic_uint_fast64_t trace_queue_id;
} CudaSubmissionState;

/* Shared immutable Driver/module state. The mutex protects only lifetime and
 * attachment, never a forward pass, training step, or GPU completion wait. */
typedef struct {
    pthread_mutex_t mutex;
    unsigned int reference_count;
    atomic_uint_fast64_t trace_device_id;
#ifdef _WIN32
    HMODULE library;
#else
    void* library;
#endif
    PFN_cuInit cuInit;
    PFN_cuDeviceGetCount cuDeviceGetCount;
    PFN_cuDeviceGet cuDeviceGet;
    PFN_cuDeviceGetName cuDeviceGetName;
    PFN_cuDeviceComputeCapability cuDeviceComputeCapability;
    PFN_cuDeviceGetAttribute cuDeviceGetAttribute;
    PFN_cuDeviceTotalMem cuDeviceTotalMem;
    PFN_cuDevicePrimaryCtxRetain cuDevicePrimaryCtxRetain;
    PFN_cuDevicePrimaryCtxRelease cuDevicePrimaryCtxRelease;
    PFN_cuCtxGetCurrent cuCtxGetCurrent;
    PFN_cuCtxSetCurrent cuCtxSetCurrent;
    PFN_cuCtxSynchronize cuCtxSynchronize;
    PFN_cuStreamCreate cuStreamCreate;
    PFN_cuStreamDestroy cuStreamDestroy;
    PFN_cuStreamSynchronize cuStreamSynchronize;
    PFN_cuStreamWaitEvent cuStreamWaitEvent;
    PFN_cuStreamGetCtx cuStreamGetCtx;
    PFN_cuStreamIsCapturing cuStreamIsCapturing;
    PFN_cuStreamBeginCapture cuStreamBeginCapture;
    PFN_cuStreamEndCapture cuStreamEndCapture;
    PFN_cuGraphInstantiateWithFlags cuGraphInstantiateWithFlags;
    PFN_cuGraphLaunch cuGraphLaunch;
    PFN_cuGraphDestroy cuGraphDestroy;
    PFN_cuGraphExecDestroy cuGraphExecDestroy;
    PFN_cuModuleLoadDataEx cuModuleLoadDataEx;
    PFN_cuModuleUnload cuModuleUnload;
    PFN_cuModuleGetFunction cuModuleGetFunction;
    PFN_cuMemAlloc cuMemAlloc;
    PFN_cuMemFree cuMemFree;
    PFN_cuMemcpyHtoD cuMemcpyHtoD;
    PFN_cuMemcpyDtoH cuMemcpyDtoH;
    PFN_cuMemcpyHtoDAsync cuMemcpyHtoDAsync;
    PFN_cuMemcpyDtoHAsync cuMemcpyDtoHAsync;
    PFN_cuMemAllocHost cuMemAllocHost;
    PFN_cuMemFreeHost cuMemFreeHost;
    PFN_cuMemcpyDtoD cuMemcpyDtoD;
    PFN_cuMemcpyDtoDAsync cuMemcpyDtoDAsync;
    PFN_cuPointerGetAttribute cuPointerGetAttribute;
    PFN_cuMemGetAddressRange cuMemGetAddressRange;
    PFN_cuLaunchKernel cuLaunchKernel;
    PFN_cuGetErrorName cuGetErrorName;
    PFN_cuEventCreate cuEventCreate;
    PFN_cuEventRecord cuEventRecord;
    PFN_cuEventDestroy cuEventDestroy;
    PFN_cuEventQuery cuEventQuery;
    PFN_cuEventSynchronize cuEventSynchronize;
    CUdevice device;
    CUcontext context;
    CUmodule module;
#if VOLVOXAI_ENABLE_TRAINING
    CUmodule training_module;
#endif
    CudaSubmissionState interop;
    int selected_device_index;
    int primary_retained;
    int ready;
    atomic_int error_logged;
    int graph_api_supported;
    unsigned int max_grid[3];
    unsigned int max_block[3];
    unsigned int max_threads_per_block;
    unsigned int max_shared_memory_per_block;
    size_t total_memory;
    CUfunction forward_functions[CUDA_FORWARD_FUNCTION_COUNT];
#if VOLVOXAI_ENABLE_TRAINING
    CUfunction training_functions[CUDA_TRAINING_FUNCTION_COUNT];
#endif
} CudaDeviceState;

static CudaDeviceState cuda_device_state = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
};

#define cuda_library (cuda_device_state.library)
#define p_cuInit (cuda_device_state.cuInit)
#define p_cuDeviceGetCount (cuda_device_state.cuDeviceGetCount)
#define p_cuDeviceGet (cuda_device_state.cuDeviceGet)
#define p_cuDeviceGetName (cuda_device_state.cuDeviceGetName)
#define p_cuDeviceComputeCapability (cuda_device_state.cuDeviceComputeCapability)
#define p_cuDeviceGetAttribute (cuda_device_state.cuDeviceGetAttribute)
#define p_cuDeviceTotalMem (cuda_device_state.cuDeviceTotalMem)
#define p_cuDevicePrimaryCtxRetain (cuda_device_state.cuDevicePrimaryCtxRetain)
#define p_cuDevicePrimaryCtxRelease (cuda_device_state.cuDevicePrimaryCtxRelease)
#define p_cuCtxGetCurrent (cuda_device_state.cuCtxGetCurrent)
#define p_cuCtxSetCurrent (cuda_device_state.cuCtxSetCurrent)
#define p_cuCtxSynchronize (cuda_device_state.cuCtxSynchronize)
#define p_cuStreamCreate (cuda_device_state.cuStreamCreate)
#define p_cuStreamDestroy (cuda_device_state.cuStreamDestroy)
#define p_cuStreamSynchronize (cuda_device_state.cuStreamSynchronize)
#define p_cuStreamWaitEvent (cuda_device_state.cuStreamWaitEvent)
#define p_cuStreamGetCtx (cuda_device_state.cuStreamGetCtx)
#define p_cuStreamIsCapturing (cuda_device_state.cuStreamIsCapturing)
#define p_cuStreamBeginCapture (cuda_device_state.cuStreamBeginCapture)
#define p_cuStreamEndCapture (cuda_device_state.cuStreamEndCapture)
#define p_cuGraphInstantiateWithFlags (cuda_device_state.cuGraphInstantiateWithFlags)
#define p_cuGraphLaunch (cuda_device_state.cuGraphLaunch)
#define p_cuGraphDestroy (cuda_device_state.cuGraphDestroy)
#define p_cuGraphExecDestroy (cuda_device_state.cuGraphExecDestroy)
#define p_cuModuleLoadDataEx (cuda_device_state.cuModuleLoadDataEx)
#define p_cuModuleUnload (cuda_device_state.cuModuleUnload)
#define p_cuModuleGetFunction (cuda_device_state.cuModuleGetFunction)
#define p_cuMemAlloc (cuda_device_state.cuMemAlloc)
#define p_cuMemFree (cuda_device_state.cuMemFree)
#define p_cuMemcpyHtoD (cuda_device_state.cuMemcpyHtoD)
#define p_cuMemcpyDtoH (cuda_device_state.cuMemcpyDtoH)
#define p_cuMemcpyHtoDAsync (cuda_device_state.cuMemcpyHtoDAsync)
#define p_cuMemcpyDtoHAsync (cuda_device_state.cuMemcpyDtoHAsync)
#define p_cuMemAllocHost (cuda_device_state.cuMemAllocHost)
#define p_cuMemFreeHost (cuda_device_state.cuMemFreeHost)
#define p_cuLaunchKernel (cuda_device_state.cuLaunchKernel)
#define p_cuGetErrorName (cuda_device_state.cuGetErrorName)
#define p_cuEventCreate (cuda_device_state.cuEventCreate)
#define p_cuEventRecord (cuda_device_state.cuEventRecord)
#define p_cuEventDestroy (cuda_device_state.cuEventDestroy)
#define p_cuEventQuery (cuda_device_state.cuEventQuery)
#define p_cuEventSynchronize (cuda_device_state.cuEventSynchronize)
#define cuda_device (cuda_device_state.device)
#define cuda_context (cuda_device_state.context)
#define cuda_module (cuda_device_state.module)
#if VOLVOXAI_ENABLE_TRAINING
#define cuda_training_module (cuda_device_state.training_module)
#endif
#define cuda_stream (cuda_submission_current()->stream)
#define cuda_primary_retained (cuda_device_state.primary_retained)
#define cuda_error_logged (cuda_device_state.error_logged)
#define cuda_max_grid (cuda_device_state.max_grid)
#define cuda_max_block (cuda_device_state.max_block)
#define cuda_max_threads_per_block (cuda_device_state.max_threads_per_block)
#define cuda_max_shared_memory_per_block \
    (cuda_device_state.max_shared_memory_per_block)
#define cuda_total_memory (cuda_device_state.total_memory)

#define CUDA_FORWARD_FUNCTION(handle) \
    (cuda_device_state.forward_functions[CUDA_FORWARD_REGISTRY_HANDLE_##handle])
#if VOLVOXAI_ENABLE_TRAINING
#define CUDA_TRAINING_FUNCTION(handle) \
    (cuda_device_state.training_functions[CUDA_TRAINING_REGISTRY_HANDLE_##handle])
#endif

#define CUDA_GRAPH_MAX_TENSORS 8192
#define CUDA_GRAPH_SLOT_HASH_CAPACITY (CUDA_GRAPH_MAX_TENSORS * 2)
#define CUDA_GRAPH_MAX_LAUNCHES 16384
#define CUDA_GRAPH_REPLAY_PLAN_CAPACITY 4
_Static_assert((CUDA_GRAPH_SLOT_HASH_CAPACITY &
                (CUDA_GRAPH_SLOT_HASH_CAPACITY - 1)) == 0,
               "CUDA tensor-slot hash capacity must be a power of two");
_Static_assert(CUDA_GRAPH_MAX_TENSORS < UINT16_MAX,
               "CUDA tensor-slot indices must fit in the hash entry");
typedef struct {
    const void* host;
    size_t bytes;
    size_t capacity;
    CUdeviceptr device;
    uint64_t shape_generation;
    uint64_t capacity_generation;
    int host_dirty;
    int device_dirty;
    int is_weight;
    int pool_eligible;
    int domain_span;
} CudaTensorSlot;

typedef struct {
    CudaTensorSlot* slot;
    CUdeviceptr device;
} CudaTensorView;

typedef struct {
    CUfunction function;
    unsigned int grid_x;
    unsigned int grid_y;
    unsigned int grid_z;
    unsigned int block_x;
    unsigned int block_y;
    unsigned int block_z;
    unsigned int shared_bytes;
} CudaLaunchSignature;

typedef struct {
    CUevent start, end;
    VxDeviceTraceSpan span;
    int ended;
} CudaTraceNode;
typedef struct {
    size_t capacity, count;
    CudaTraceNode* entries;
} CudaTraceNodes;
typedef struct {
    CUevent start, end;
    PFN_cuEventElapsedTime elapsed_time;
    uint64_t host_start_ns;
    uint64_t host_end_ns, elapsed_ns;
    VxTraceQueue queue, previous_queue;
    const char* name;
    VxTraceScope* scope;
    int end_recorded;
    CudaTraceNodes* nodes;
    CudaTraceNodes* programs;
    int owns_programs;
    PFN_cuLaunchKernel launch;
    CUresult (CUDAAPI *record_external)(CUevent, CUstream, unsigned int);
} CudaTracePass;

enum {
    CUDA_REPLAY_PLAN_EMPTY = 0,
    CUDA_REPLAY_PLAN_OBSERVED = 1,
    CUDA_REPLAY_PLAN_READY = 2,
};

enum {
    CUDA_REPLAY_PASS_INELIGIBLE = 0,
    CUDA_REPLAY_PASS_OBSERVE = 1,
    CUDA_REPLAY_PASS_CAPTURE = 2,
    CUDA_REPLAY_PASS_VALIDATE = 3,
};

typedef struct {
    int plan;
    int pass;
    int prepared;
    int capture_active;
    int failed;
    int unsafe;
    int prestaging;
    int destroy_pending;
    int invalidation_counted;
    uint64_t generation;
    uint64_t slot_epoch;
    uint64_t shape_generation;
    uint64_t capacity_generation;
    int domain_enforced;
    char* shape_signature;
    uint64_t last_used;
    uint32_t launch_count;
    uint32_t launch_index;
    CUgraph captured_graph;
    CUgraphExec executable;
    unsigned char initial_reads[CUDA_GRAPH_MAX_TENSORS];
    unsigned char pass_writes[CUDA_GRAPH_MAX_TENSORS];
    CudaLaunchSignature launches[CUDA_GRAPH_MAX_LAUNCHES];
} CudaReplayState;

typedef struct {
    CudaReplayState plan;
    CudaTraceNodes* timing;
    CudaTraceNodes* programs;
} CudaTraceReplay;

enum {
    CUDA_QACT_LUT_VALUES = 256,
    CUDA_QACT_LUT_SETS = 8,
    CUDA_QACT_LUT_WAYS = 4,
};

typedef struct {
    uint32_t input_scale_bits;
    uint32_t output_scale_bits;
    int32_t input_zero_point;
    int32_t output_zero_point;
    uint8_t input_dtype;
    uint8_t output_dtype;
    uint8_t operation;
    uint8_t valid;
    uint8_t values[CUDA_QACT_LUT_VALUES];
} CudaQactLutEntry;

typedef struct {
    CudaQactLutEntry entries[CUDA_QACT_LUT_SETS][CUDA_QACT_LUT_WAYS];
    uint8_t next_victim[CUDA_QACT_LUT_SETS];
} CudaQactLutCache;

typedef struct {
    CUcontext previous;
    int active;
    int borrowed;
    int owns_interop_lock;
    int owns_device_reference;
} CudaContextGuard;

struct CudaTrainingOptimizerMirror;

/* One capsule is attached lazily to each VxEngineState. It is the named owner
 * of CUDA graph residency, replay observations, request workspaces, profile
 * evidence, and every full-profile optimizer/training field. */
typedef struct {
    CudaSubmissionState submission;
    int device_attached;
    int ready;
    int graph_api_available;
    CudaTensorSlot graph_slots[CUDA_GRAPH_MAX_TENSORS];
    int graph_slot_count;
    /* Zero is empty; populated entries hold graph_slots index plus one. */
    uint16_t graph_slot_hash[CUDA_GRAPH_SLOT_HASH_CAPACITY];
    int graph_slot_hash_valid;
    uint64_t graph_slot_epoch;
    char* shape_signature;
    uint64_t shape_generation;
    uint64_t capacity_generation;
    size_t domain_span_count;
    int invariant_preload_complete;
    int domain_enforced;
    int graph_allocation_failed;
    CudaReplayState replay;
    CudaReplayState* replay_plans[CUDA_GRAPH_REPLAY_PLAN_CAPACITY];
    size_t replay_plan_count;
    CudaReplayState* replay_active;
    uint64_t replay_clock;
    uint64_t replay_capture_total;
    uint64_t replay_launch_total;
    uint64_t replay_hit_total;
    uint64_t replay_invalidation_total;
    int replay_last_forward_hit;
    CudaQactLutCache qact_lut_cache;
    CudaContextGuard forward_context_guard;
    int forward_scope_active;
    int forward_context_ready;
    int forward_context_failed;
    uint64_t qbatch_matmul_dp4a_group_count;
    uint64_t qbatch_matmul_scalar_tail_count;
    CUdeviceptr lora_workspace;
    size_t lora_workspace_bytes;
#if VOLVOXAI_ENABLE_TRAINING
    CudaContextGuard training_context_guard;
    int training_active;
    int training_context_ready;
    int training_failed;
    CUdeviceptr training_attention_workspace;
    size_t training_attention_workspace_bytes;
    CUdeviceptr training_basic_workspace;
    size_t training_basic_workspace_bytes;
    struct CudaTrainingOptimizerMirror* training_optimizer_mirrors;
#endif
    CudaTracePass trace_pass;
    CudaTraceReplay* traced_replay;
} CudaContextState;

static void cuda_context_state_destroy(void* opaque_state);

static CudaContextState* cuda_context_state_get(void) {
    VxEngineState* engine_state = vx_engine_state_current();
    return engine_state
        ? (CudaContextState*)engine_state->cuda_context_state : NULL;
}

static CudaContextState* cuda_context_state_require(void) {
    VxEngineState* engine_state = vx_engine_state_current();
    CudaContextState* state;
    if (!engine_state) return NULL;
    state = (CudaContextState*)engine_state->cuda_context_state;
    if (state) return state;
    state = (CudaContextState*)calloc(1, sizeof(*state));
    if (!state) return NULL;
    state->graph_slot_hash_valid = 1;
    state->graph_slot_epoch = 1;
    state->shape_generation = 1;
    state->capacity_generation = 1;
    state->replay_plans[0] = &state->replay;
    state->replay_plan_count = 1u;
    state->replay_active = &state->replay;
    engine_state->cuda_context_state = state;
    engine_state->cuda_context_state_destroy = cuda_context_state_destroy;
    return state;
}

static pthread_mutex_t cuda_interop_mutex = PTHREAD_MUTEX_INITIALIZER;
/* Completion creation is brief; waits lock only the affected allocation. */
static pthread_mutex_t cuda_completion_mutex = PTHREAD_MUTEX_INITIALIZER;
static CudaSubmissionState* cuda_submission_current(void) {
    CudaContextState* state = cuda_context_state_get();
    return state && state->device_attached ? &state->submission : &cuda_device_state.interop;
}
static void cuda_device_teardown_locked(void);
static void cuda_buffer_release_external(void);

static inline CudaContextState* cuda_context_state_current(void) {
    CudaContextState* state = cuda_context_state_get();
    assert(state);
    return state;
}

#define CUDA_CONTEXT_FIELD(field) (cuda_context_state_current()->field)
#define cuda_ready CUDA_CONTEXT_FIELD(ready)
#define cuda_graph_api_available CUDA_CONTEXT_FIELD(graph_api_available)
#define graph_slots CUDA_CONTEXT_FIELD(graph_slots)
#define graph_slot_count CUDA_CONTEXT_FIELD(graph_slot_count)
#define graph_slot_hash CUDA_CONTEXT_FIELD(graph_slot_hash)
#define graph_slot_hash_valid CUDA_CONTEXT_FIELD(graph_slot_hash_valid)
#define graph_slot_epoch CUDA_CONTEXT_FIELD(graph_slot_epoch)
#define cuda_shape_signature CUDA_CONTEXT_FIELD(shape_signature)
#define cuda_shape_generation CUDA_CONTEXT_FIELD(shape_generation)
#define cuda_capacity_generation CUDA_CONTEXT_FIELD(capacity_generation)
#define cuda_domain_span_count CUDA_CONTEXT_FIELD(domain_span_count)
#define cuda_invariant_preload_complete \
    CUDA_CONTEXT_FIELD(invariant_preload_complete)
#define cuda_domain_enforced CUDA_CONTEXT_FIELD(domain_enforced)
#define cuda_graph_allocation_failed \
    CUDA_CONTEXT_FIELD(graph_allocation_failed)
static inline CudaReplayState* cuda_replay_state_current(void) {
    CudaContextState* state = cuda_context_state_current();
    return state->replay_active ? state->replay_active : &state->replay;
}
#define cuda_replay (*cuda_replay_state_current())
#define cuda_replay_plans CUDA_CONTEXT_FIELD(replay_plans)
#define cuda_replay_plan_count CUDA_CONTEXT_FIELD(replay_plan_count)
#define cuda_replay_active CUDA_CONTEXT_FIELD(replay_active)
#define cuda_replay_clock CUDA_CONTEXT_FIELD(replay_clock)
#define cuda_replay_capture_total CUDA_CONTEXT_FIELD(replay_capture_total)
#define cuda_replay_launch_total CUDA_CONTEXT_FIELD(replay_launch_total)
#define cuda_replay_hit_total CUDA_CONTEXT_FIELD(replay_hit_total)
#define cuda_replay_invalidation_total \
    CUDA_CONTEXT_FIELD(replay_invalidation_total)
#define cuda_replay_last_forward_hit \
    CUDA_CONTEXT_FIELD(replay_last_forward_hit)
#define qact_lut_cache CUDA_CONTEXT_FIELD(qact_lut_cache)
#define cuda_forward_context_guard CUDA_CONTEXT_FIELD(forward_context_guard)
#define cuda_forward_scope_active CUDA_CONTEXT_FIELD(forward_scope_active)
#define cuda_forward_context_ready CUDA_CONTEXT_FIELD(forward_context_ready)
#define cuda_forward_context_failed CUDA_CONTEXT_FIELD(forward_context_failed)
#define cuda_qbatch_matmul_dp4a_group_count \
    CUDA_CONTEXT_FIELD(qbatch_matmul_dp4a_group_count)
#define cuda_qbatch_matmul_scalar_tail_count \
    CUDA_CONTEXT_FIELD(qbatch_matmul_scalar_tail_count)
#define cuda_lora_workspace CUDA_CONTEXT_FIELD(lora_workspace)
#define cuda_lora_workspace_bytes CUDA_CONTEXT_FIELD(lora_workspace_bytes)
#if VOLVOXAI_ENABLE_TRAINING
#define cuda_training_context_guard CUDA_CONTEXT_FIELD(training_context_guard)
#define cuda_training_active CUDA_CONTEXT_FIELD(training_active)
#define cuda_training_context_ready CUDA_CONTEXT_FIELD(training_context_ready)
#define cuda_training_failed CUDA_CONTEXT_FIELD(training_failed)
#define cuda_training_attention_workspace \
    CUDA_CONTEXT_FIELD(training_attention_workspace)
#define cuda_training_attention_workspace_bytes \
    CUDA_CONTEXT_FIELD(training_attention_workspace_bytes)
#define cuda_training_basic_workspace CUDA_CONTEXT_FIELD(training_basic_workspace)
#define cuda_training_basic_workspace_bytes \
    CUDA_CONTEXT_FIELD(training_basic_workspace_bytes)
#define cuda_training_optimizer_mirrors \
    CUDA_CONTEXT_FIELD(training_optimizer_mirrors)
#endif

static void* cuda_symbol(const char* name);
#include "cuda/host/cuda_trace_host.inc"
#include "cuda/host/cuda_trace_io_host.inc"

typedef struct {
    uint32_t output_strides[8];
    uint32_t a_strides[8];
    uint32_t b_strides[8];
    uint32_t rank;
} VxCudaBroadcastParams;

typedef struct {
    uint32_t output_strides[8];
    uint32_t input_strides[8];
    uint32_t rank;
} VxCudaTransposeParams;

typedef struct {
    uint32_t output_batch_strides[8];
    uint32_t a_batch_strides[8];
    uint32_t b_batch_strides[8];
    uint32_t batch_rank;
    uint32_t m;
    uint32_t k;
    uint32_t n;
} VxCudaBatchMatMulParams;

typedef struct {
    uint32_t output_strides[8];
    uint32_t input_strides[8];
    uint32_t starts[8];
    uint32_t steps[8];
    uint32_t rank;
} VxCudaSliceParams;

_Static_assert(sizeof(VxCudaBroadcastParams) == 100,
               "CUDA broadcast parameter ABI");
_Static_assert(sizeof(VxCudaTransposeParams) == 68,
               "CUDA transpose parameter ABI");
_Static_assert(sizeof(VxCudaBatchMatMulParams) == 112,
               "CUDA batch-matmul parameter ABI");
_Static_assert(sizeof(VxCudaSliceParams) == 132,
               "CUDA slice parameter ABI");

#include "cuda/host/cuda_driver_host.inc"
#include "cuda/host/cuda_transfer_host.inc"

#include "cuda/host/cuda_graph_memory_host.inc"

#if VOLVOXAI_ENABLE_TRAINING
#include "cuda/host/cuda_training_step_host.inc"
#include "cuda/host/cuda_quantize_w8_host.inc"
#endif

#include "cuda/host/cuda_graph_quantized_common_host.inc"

#include "cuda/host/cuda_backend_lifecycle_host.inc"
#include "cuda/host/cuda_tensor_interop_host.inc"

#include "cuda/host/cuda_training_dispatch_host.inc"

#include "cuda/host/cuda_graph_lifecycle_host.inc"
#include "cuda/host/cuda_graph_f32_basic_host.inc"
#include "cuda/host/cuda_graph_shape_pool_host.inc"
#include "cuda/host/cuda_graph_typed_graph_host.inc"
#include "cuda/host/cuda_graph_conv_attention_host.inc"
#include "cuda/host/cuda_graph_dense_moe_host.inc"
#include "cuda/host/cuda_graph_quantized_host.inc"
