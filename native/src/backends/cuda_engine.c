#include "cuda_engine.h"
#include "qlinear_multiplier.h"
#include "runtime_state.h"

#include "embedded_cuda_ptx.h"
#if VOLVOXAI_ENABLE_TRAINING
#include "embedded_cuda_training_ptx.h"
#endif

#include <errno.h>
#include <assert.h>
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
#if VOLVOXAI_ENABLE_TRAINING
typedef struct CUevent_st* CUevent;
#endif
typedef uintptr_t CUdeviceptr;

enum {
    CUDA_SUCCESS = 0,
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
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
typedef CUresult (CUDAAPI *PFN_cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
typedef CUresult (CUDAAPI *PFN_cuDevicePrimaryCtxRelease)(CUdevice);
typedef CUresult (CUDAAPI *PFN_cuCtxGetCurrent)(CUcontext*);
typedef CUresult (CUDAAPI *PFN_cuCtxSetCurrent)(CUcontext);
typedef CUresult (CUDAAPI *PFN_cuCtxSynchronize)(void);
typedef CUresult (CUDAAPI *PFN_cuStreamCreate)(CUstream*, unsigned int);
typedef CUresult (CUDAAPI *PFN_cuStreamDestroy)(CUstream);
typedef CUresult (CUDAAPI *PFN_cuStreamSynchronize)(CUstream);
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
typedef CUresult (CUDAAPI *PFN_cuLaunchKernel)(CUfunction,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, CUstream, void**, void**);
typedef CUresult (CUDAAPI *PFN_cuGetErrorName)(CUresult, const char**);
#if VOLVOXAI_ENABLE_TRAINING
typedef CUresult (CUDAAPI *PFN_cuEventCreate)(CUevent*, unsigned int);
typedef CUresult (CUDAAPI *PFN_cuEventRecord)(CUevent, CUstream);
typedef CUresult (CUDAAPI *PFN_cuEventElapsedTime)(float*, CUevent, CUevent);
typedef CUresult (CUDAAPI *PFN_cuEventDestroy)(CUevent);
#endif

enum {
#define VOLVOXAI_CUDA_FORWARD_FUNCTION(requirement, handle, symbol) \
    CUDA_FORWARD_REGISTRY_HANDLE_##handle,
#include "cuda/host/cuda_forward_function_registry_host.inc"
#undef VOLVOXAI_CUDA_FORWARD_FUNCTION
    CUDA_FORWARD_FUNCTION_COUNT
};
_Static_assert(CUDA_FORWARD_FUNCTION_COUNT == 88,
               "CUDA forward function registry coverage");

#if VOLVOXAI_ENABLE_TRAINING
enum {
#define VOLVOXAI_CUDA_TRAINING_FUNCTION( \
        requirement, profile_timing, handle, symbol, operation) \
    CUDA_TRAINING_REGISTRY_HANDLE_##handle,
#include "cuda/host/cuda_training_function_registry_host.inc"
#undef VOLVOXAI_CUDA_TRAINING_FUNCTION
    CUDA_TRAINING_FUNCTION_COUNT
};
_Static_assert(CUDA_TRAINING_FUNCTION_COUNT == 79,
               "CUDA training function registry coverage");

#endif

/* The one intentional process-wide CUDA object owns only the physical Driver
 * API connection and immutable module/function cache. Its mutex serializes
 * initialization, teardown, primary-context switching, and work submitted to
 * the shared stream. Graph/request/training state lives in CudaContextState. */
typedef struct {
    pthread_mutex_t mutex;
    unsigned int reference_count;
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
    PFN_cuDevicePrimaryCtxRetain cuDevicePrimaryCtxRetain;
    PFN_cuDevicePrimaryCtxRelease cuDevicePrimaryCtxRelease;
    PFN_cuCtxGetCurrent cuCtxGetCurrent;
    PFN_cuCtxSetCurrent cuCtxSetCurrent;
    PFN_cuCtxSynchronize cuCtxSynchronize;
    PFN_cuStreamCreate cuStreamCreate;
    PFN_cuStreamDestroy cuStreamDestroy;
    PFN_cuStreamSynchronize cuStreamSynchronize;
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
    PFN_cuLaunchKernel cuLaunchKernel;
    PFN_cuGetErrorName cuGetErrorName;
#if VOLVOXAI_ENABLE_TRAINING
    PFN_cuEventCreate cuEventCreate;
    PFN_cuEventRecord cuEventRecord;
    PFN_cuEventElapsedTime cuEventElapsedTime;
    PFN_cuEventDestroy cuEventDestroy;
#endif
    CUdevice device;
    CUcontext context;
    CUmodule module;
#if VOLVOXAI_ENABLE_TRAINING
    CUmodule training_module;
#endif
    CUstream stream;
    int selected_device_index;
    int primary_retained;
    int ready;
    int error_logged;
    int graph_api_supported;
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
#define p_cuDevicePrimaryCtxRetain (cuda_device_state.cuDevicePrimaryCtxRetain)
#define p_cuDevicePrimaryCtxRelease (cuda_device_state.cuDevicePrimaryCtxRelease)
#define p_cuCtxGetCurrent (cuda_device_state.cuCtxGetCurrent)
#define p_cuCtxSetCurrent (cuda_device_state.cuCtxSetCurrent)
#define p_cuCtxSynchronize (cuda_device_state.cuCtxSynchronize)
#define p_cuStreamCreate (cuda_device_state.cuStreamCreate)
#define p_cuStreamDestroy (cuda_device_state.cuStreamDestroy)
#define p_cuStreamSynchronize (cuda_device_state.cuStreamSynchronize)
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
#define p_cuLaunchKernel (cuda_device_state.cuLaunchKernel)
#define p_cuGetErrorName (cuda_device_state.cuGetErrorName)
#if VOLVOXAI_ENABLE_TRAINING
#define p_cuEventCreate (cuda_device_state.cuEventCreate)
#define p_cuEventRecord (cuda_device_state.cuEventRecord)
#define p_cuEventElapsedTime (cuda_device_state.cuEventElapsedTime)
#define p_cuEventDestroy (cuda_device_state.cuEventDestroy)
#endif
#define cuda_device (cuda_device_state.device)
#define cuda_context (cuda_device_state.context)
#define cuda_module (cuda_device_state.module)
#if VOLVOXAI_ENABLE_TRAINING
#define cuda_training_module (cuda_device_state.training_module)
#endif
#define cuda_stream (cuda_device_state.stream)
#define cuda_primary_retained (cuda_device_state.primary_retained)
#define cuda_error_logged (cuda_device_state.error_logged)

#define CUDA_FORWARD_FUNCTION(handle) \
    (cuda_device_state.forward_functions[CUDA_FORWARD_REGISTRY_HANDLE_##handle])
#if VOLVOXAI_ENABLE_TRAINING
#define CUDA_TRAINING_FUNCTION(handle) \
    (cuda_device_state.training_functions[CUDA_TRAINING_REGISTRY_HANDLE_##handle])
#endif

#define CUDA_GRAPH_MAX_TENSORS 8192
#define CUDA_GRAPH_SLOT_HASH_CAPACITY (CUDA_GRAPH_MAX_TENSORS * 2)
#define CUDA_GRAPH_MAX_LAUNCHES 16384
_Static_assert((CUDA_GRAPH_SLOT_HASH_CAPACITY &
                (CUDA_GRAPH_SLOT_HASH_CAPACITY - 1)) == 0,
               "CUDA tensor-slot hash capacity must be a power of two");
_Static_assert(CUDA_GRAPH_MAX_TENSORS < UINT16_MAX,
               "CUDA tensor-slot indices must fit in the hash entry");
typedef struct {
    const void* host;
    size_t bytes;
    CUdeviceptr device;
    int host_dirty;
    int device_dirty;
    int is_weight;
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

#if VOLVOXAI_ENABLE_TRAINING
enum { CUDA_PROFILE_MAX_FUNCTIONS = 192 };
enum { CUDA_PROFILE_PATH_CAPACITY = 4096 };

typedef struct {
    CUfunction function;
    const char* name;
} CudaProfileFunction;

typedef struct {
    CUevent start;
    CUevent end;
    CudaLaunchSignature signature;
    const char* name;
} CudaProfileRecord;

typedef struct {
    CUevent start;
    CUevent end;
    CudaLaunchSignature signature;
    const char* name;
    int active;
} CudaProfilePendingLaunch;

typedef struct {
    int enabled;
    int api_available;
    int incomplete;
    int scope_open;
    int loaded_model_trainstep;
    int non_cuda_route;
    int header_written;
    uint64_t scope;
    FILE* output;
    int owns_output;
    CudaProfileFunction functions[CUDA_PROFILE_MAX_FUNCTIONS];
    size_t function_count;
    CudaProfileRecord* records;
    size_t record_count;
    size_t record_capacity;
} CudaProfileState;

typedef struct {
    const char* name;
    CudaLaunchSignature signature;
    uint64_t count;
    double total_ms;
    float max_ms;
} CudaProfileAggregate;

typedef struct CudaProfilePathHistory {
    struct CudaProfilePathHistory* next;
    char path[];
} CudaProfilePathHistory;
#endif

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
    uint64_t generation;
    uint64_t slot_epoch;
    uint32_t launch_count;
    uint32_t launch_index;
    CUgraph captured_graph;
    CUgraphExec executable;
    unsigned char initial_reads[CUDA_GRAPH_MAX_TENSORS];
    unsigned char pass_writes[CUDA_GRAPH_MAX_TENSORS];
    CudaLaunchSignature launches[CUDA_GRAPH_MAX_LAUNCHES];
} CudaReplayState;

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
    int owns_device_lock;
} CudaContextGuard;

struct CudaTrainingOptimizerMirror;

/* One capsule is attached lazily to each VxEngineState. It is the named owner
 * of CUDA graph residency, replay observations, request workspaces, profile
 * evidence, test counters, and every full-profile optimizer/training field. */
typedef struct {
    int device_attached;
    int ready;
    int unavailable_failure;
    int graph_api_available;
    CudaTensorSlot graph_slots[CUDA_GRAPH_MAX_TENSORS];
    int graph_slot_count;
    /* Zero is empty; populated entries hold graph_slots index plus one. */
    uint16_t graph_slot_hash[CUDA_GRAPH_SLOT_HASH_CAPACITY];
    int graph_slot_hash_valid;
    uint64_t graph_slot_epoch;
    CudaReplayState replay;
    CudaQactLutCache qact_lut_cache;
    CudaContextGuard forward_context_guard;
    int forward_scope_active;
    int forward_context_ready;
    int forward_context_failed;
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
    uint64_t training_step_stream_sync_count;
    CudaProfileState profile;
    CudaProfilePathHistory* profile_path_history;
    uint64_t profile_scope_serial;
#endif
#if defined(VOLVOXAI_CUDA_TESTING)
    uint64_t ownership_probe;
    uint64_t launch_count;
    uint64_t host_to_device_count;
    uint64_t device_to_host_count;
    uint64_t host_to_device_bytes;
    uint64_t device_to_host_bytes;
    uint64_t weight_host_to_device_count;
    uint64_t weight_host_to_device_bytes;
    uint64_t weight_promotion_count;
    uint64_t weight_promotion_bytes;
    uint64_t conv2d_1x1_tiled_launch_count;
    uint64_t conv2d_1x1_bm32_bn32_bk16_launch_count;
    uint64_t conv2d_1x1_bm16_bn64_bk16_launch_count;
    uint64_t depthwise_conv2d_3x3_c1_launch_count;
    uint64_t depthwise_conv2d_3x3_c4_launch_count;
    uint64_t depthwise_conv2d_5x5_c1_launch_count;
    uint64_t depthwise_conv2d_5x5_c4_launch_count;
    uint64_t training_conv2d_input_hwio_3x3_tiled_launch_count;
    uint64_t training_conv2d_input_hwio_3x3_tiled_c32_launch_count;
    uint64_t training_conv2d_weight_hwio_3x3_tiled_launch_count;
    uint64_t add3_relu_launch_count;
    uint64_t conv2d_add_launch_count;
    uint64_t conv2d_1x1_tiled_add_launch_count;
    uint64_t conv2d_1x1_bm32_bn32_bk16_add_launch_count;
    uint64_t conv2d_1x1_bm16_bn64_bk16_add_launch_count;
    uint64_t context_get_current_count;
    uint64_t context_set_current_count;
    uint64_t graph_capture_count;
    uint64_t graph_launch_count;
    uint64_t graph_replay_count;
    uint64_t graph_invalidation_count;
    uint64_t slot_exact_lookup_count;
    uint64_t slot_hash_probe_count;
    uint64_t slot_containing_scan_count;
    int transient_release_failure;
#endif
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
    state->unavailable_failure = 1;
    state->graph_slot_hash_valid = 1;
    state->graph_slot_epoch = 1;
    engine_state->cuda_context_state = state;
    engine_state->cuda_context_state_destroy = cuda_context_state_destroy;
    return state;
}

static inline CudaContextState* cuda_context_state_current(void) {
    CudaContextState* state = cuda_context_state_get();
    assert(state);
    return state;
}

#define CUDA_CONTEXT_FIELD(field) (cuda_context_state_current()->field)
#define cuda_ready CUDA_CONTEXT_FIELD(ready)
#define cuda_unavailable_failure CUDA_CONTEXT_FIELD(unavailable_failure)
#define cuda_graph_api_available CUDA_CONTEXT_FIELD(graph_api_available)
#define graph_slots CUDA_CONTEXT_FIELD(graph_slots)
#define graph_slot_count CUDA_CONTEXT_FIELD(graph_slot_count)
#define graph_slot_hash CUDA_CONTEXT_FIELD(graph_slot_hash)
#define graph_slot_hash_valid CUDA_CONTEXT_FIELD(graph_slot_hash_valid)
#define graph_slot_epoch CUDA_CONTEXT_FIELD(graph_slot_epoch)
#define cuda_replay CUDA_CONTEXT_FIELD(replay)
#define qact_lut_cache CUDA_CONTEXT_FIELD(qact_lut_cache)
#define cuda_forward_context_guard CUDA_CONTEXT_FIELD(forward_context_guard)
#define cuda_forward_scope_active CUDA_CONTEXT_FIELD(forward_scope_active)
#define cuda_forward_context_ready CUDA_CONTEXT_FIELD(forward_context_ready)
#define cuda_forward_context_failed CUDA_CONTEXT_FIELD(forward_context_failed)
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
#define cuda_training_step_stream_sync_count \
    CUDA_CONTEXT_FIELD(training_step_stream_sync_count)
#define cuda_profile CUDA_CONTEXT_FIELD(profile)
#define cuda_profile_path_history CUDA_CONTEXT_FIELD(profile_path_history)
#define cuda_profile_scope_serial CUDA_CONTEXT_FIELD(profile_scope_serial)
#endif
#if defined(VOLVOXAI_CUDA_TESTING)
#define cuda_launch_count CUDA_CONTEXT_FIELD(launch_count)
#define cuda_host_to_device_count CUDA_CONTEXT_FIELD(host_to_device_count)
#define cuda_device_to_host_count CUDA_CONTEXT_FIELD(device_to_host_count)
#define cuda_host_to_device_bytes CUDA_CONTEXT_FIELD(host_to_device_bytes)
#define cuda_device_to_host_bytes CUDA_CONTEXT_FIELD(device_to_host_bytes)
#define cuda_weight_host_to_device_count \
    CUDA_CONTEXT_FIELD(weight_host_to_device_count)
#define cuda_weight_host_to_device_bytes \
    CUDA_CONTEXT_FIELD(weight_host_to_device_bytes)
#define cuda_weight_promotion_count CUDA_CONTEXT_FIELD(weight_promotion_count)
#define cuda_weight_promotion_bytes CUDA_CONTEXT_FIELD(weight_promotion_bytes)
#define cuda_conv2d_1x1_tiled_launch_count \
    CUDA_CONTEXT_FIELD(conv2d_1x1_tiled_launch_count)
#define cuda_conv2d_1x1_bm32_bn32_bk16_launch_count \
    CUDA_CONTEXT_FIELD(conv2d_1x1_bm32_bn32_bk16_launch_count)
#define cuda_conv2d_1x1_bm16_bn64_bk16_launch_count \
    CUDA_CONTEXT_FIELD(conv2d_1x1_bm16_bn64_bk16_launch_count)
#define cuda_depthwise_conv2d_3x3_c1_launch_count \
    CUDA_CONTEXT_FIELD(depthwise_conv2d_3x3_c1_launch_count)
#define cuda_depthwise_conv2d_3x3_c4_launch_count \
    CUDA_CONTEXT_FIELD(depthwise_conv2d_3x3_c4_launch_count)
#define cuda_depthwise_conv2d_5x5_c1_launch_count \
    CUDA_CONTEXT_FIELD(depthwise_conv2d_5x5_c1_launch_count)
#define cuda_depthwise_conv2d_5x5_c4_launch_count \
    CUDA_CONTEXT_FIELD(depthwise_conv2d_5x5_c4_launch_count)
#define cuda_training_conv2d_input_hwio_3x3_tiled_launch_count \
    CUDA_CONTEXT_FIELD(training_conv2d_input_hwio_3x3_tiled_launch_count)
#define cuda_training_conv2d_input_hwio_3x3_tiled_c32_launch_count \
    CUDA_CONTEXT_FIELD(training_conv2d_input_hwio_3x3_tiled_c32_launch_count)
#define cuda_training_conv2d_weight_hwio_3x3_tiled_launch_count \
    CUDA_CONTEXT_FIELD(training_conv2d_weight_hwio_3x3_tiled_launch_count)
#define cuda_add3_relu_launch_count CUDA_CONTEXT_FIELD(add3_relu_launch_count)
#define cuda_conv2d_add_launch_count CUDA_CONTEXT_FIELD(conv2d_add_launch_count)
#define cuda_conv2d_1x1_tiled_add_launch_count \
    CUDA_CONTEXT_FIELD(conv2d_1x1_tiled_add_launch_count)
#define cuda_conv2d_1x1_bm32_bn32_bk16_add_launch_count \
    CUDA_CONTEXT_FIELD(conv2d_1x1_bm32_bn32_bk16_add_launch_count)
#define cuda_conv2d_1x1_bm16_bn64_bk16_add_launch_count \
    CUDA_CONTEXT_FIELD(conv2d_1x1_bm16_bn64_bk16_add_launch_count)
#define cuda_context_get_current_count \
    CUDA_CONTEXT_FIELD(context_get_current_count)
#define cuda_context_set_current_count \
    CUDA_CONTEXT_FIELD(context_set_current_count)
#define cuda_graph_capture_count CUDA_CONTEXT_FIELD(graph_capture_count)
#define cuda_graph_launch_count CUDA_CONTEXT_FIELD(graph_launch_count)
#define cuda_graph_replay_count CUDA_CONTEXT_FIELD(graph_replay_count)
#define cuda_graph_invalidation_count CUDA_CONTEXT_FIELD(graph_invalidation_count)
#define cuda_slot_exact_lookup_count CUDA_CONTEXT_FIELD(slot_exact_lookup_count)
#define cuda_slot_hash_probe_count CUDA_CONTEXT_FIELD(slot_hash_probe_count)
#define cuda_slot_containing_scan_count \
    CUDA_CONTEXT_FIELD(slot_containing_scan_count)
#define cuda_test_transient_release_failure \
    CUDA_CONTEXT_FIELD(transient_release_failure)
#endif

#include "cuda/host/cuda_profile_host.inc"

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

#include "cuda/host/cuda_graph_memory_host.inc"

#if VOLVOXAI_ENABLE_TRAINING
#include "cuda/host/cuda_training_step_host.inc"
#include "cuda/host/cuda_quantize_w8_host.inc"
#endif

#include "cuda/host/cuda_graph_quantized_common_host.inc"

#include "cuda/host/cuda_backend_lifecycle_host.inc"

#include "cuda/host/cuda_training_dispatch_host.inc"

#include "cuda/host/cuda_graph_lifecycle_host.inc"
#include "cuda/host/cuda_graph_f32_basic_host.inc"
#include "cuda/host/cuda_graph_shape_pool_host.inc"
#include "cuda/host/cuda_graph_typed_graph_host.inc"
#include "cuda/host/cuda_graph_conv_attention_host.inc"
#include "cuda/host/cuda_graph_dense_moe_host.inc"
#include "cuda/host/cuda_graph_quantized_host.inc"
