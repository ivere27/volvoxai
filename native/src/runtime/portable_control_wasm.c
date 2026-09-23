#include "generated/proto_methods.h"
#if !defined(__wasm__)
#error "portable_control_wasm.c is a browser wasm32 translation unit."
#endif

#ifndef VOLVOXAI_WASM_FREESTANDING_EXTENDED_LIBC
#define VOLVOXAI_WASM_FREESTANDING_EXTENDED_LIBC 1
#endif
#ifndef SYNURANG_RUNTIME_NO_THREADS
#define SYNURANG_RUNTIME_NO_THREADS 1
#endif
#ifndef VOLVOXAI_NO_THREADS
#define VOLVOXAI_NO_THREADS 1
#endif
#ifndef VOLVOXAI_VERSION
#define VOLVOXAI_VERSION "0.6.0"
#endif

#include "wasm_freestanding/include/stdio.h"
#include "wasm_freestanding/include/stdlib.h"
#include "wasm_freestanding/include/string.h"
#include "wasm_freestanding/include/math.h"
#include "wasm_freestanding/include/errno.h"
#include "wasm_freestanding/include/time.h"

/* Freestanding C runtime & VFS */
#include "wasm_libc.c"

/* Third-party libraries */
#include "../../third_party/cJSON.c"
#include "../../third_party/synurang/src/c_runtime.c"

/* Generated registry tables, protobuf codecs and dispatch */
#include "../generated/operator_vocabulary.c"
#include "../generated/kernel_registry.c"
#if defined(VOLVOXAI_ENABLE_TRAINING) && VOLVOXAI_ENABLE_TRAINING
#include "../generated/kernel_registry_full.c"
#include "../../../runtime/generated/c/volvoxai_lite.c"
#include "../../../runtime/generated/c/volvoxai_ffi.c"
#else
#include "../../../runtime/generated/c/inference/volvoxai_lite.c"
#include "../../../runtime/generated/c/inference/volvoxai_ffi.c"
#endif

/* Keep internal planning, contract, and parsing APIs private under --export-all */
#define VX_GRAPH_DOMAIN_API static __attribute__((unused))
#define VX_SHAPE_DOMAIN_CONTRACT_API static __attribute__((unused))
#define VX_INDEPENDENT_BATCH_PROOF_API static __attribute__((unused))
#define VX_SAFETENSORS_HEADER_API static __attribute__((unused))
#define VX_ACTIVATION_PLAN_API static __attribute__((unused))
#define VX_GRAPH_BIND_API static __attribute__((unused))
#define VX_GRAPH_PLAN_API static __attribute__((unused))
#define VX_CALL_SEQUENCE_POLICY_API static __attribute__((unused))

/* Safetensors loading & parsing */
#include "safetensors_header.c"
#include "safetensors_bytes.c"
#include "safetensors.c"

/* Runtime contracts, planning, and binding */
#include "shape_contract.c"
#include "shape_domain_contract.c"
#include "graph_plan.c"
#include "activation_plan.c"
#include "graph_bind.c"
#include "graph_bind_definition.c"
#include "graph_domain.c"
#include "independent_batch_proof.c"
#include "call_sequence_policy.c"
#include "backend_sdk.c"
#include "adapter_runtime.c"
#include "paged_kv.c"
#include "paged_binding.c"
#undef _POSIX_C_SOURCE
#include "batch_scheduler.c"
#include "continuous_batch_scheduler.c"
#include "decode_row_set.c"
#include "batch_decode.c"
#include "incremental_runtime.c"
#include "decode_session.c"
#include "../backends/w8a8_device_ops.c"
#include "native_tensor.c"
#include "profiling.c"
#include "engine_state.c"
#include "runtime_state.c"
#include "sequence_runtime.c"
#include "backend.c"
#include "arena.c"
#include "compiled_weight_resources.c"
#include "attention_mask.c"

/* Public API implementation & Engine Runtime */
#include "../api/vx_api_handles.c"
#include "../api/vx_api_convert.c"
#include "../api/vx_api_buffer.c"
#include "../api/vx_api_memory.c"
#include "../api/vx_api_profiling.c"
#include "../api/vx_api_inference.c"
#include "../api/vx_api_platform.c"
#include "../api/vx_api_scheduler.c"
#include "tokenizer.c"
#include "../api/vx_api_text.c"
/* Full-profile API surfaces. Authoring builds, edits and serializes graphs;
   inference lowers them inside LoadModel/CompileModel. Mirrors FULL_PROFILE_SRCS
   in native/CMakeLists.txt and the registration guard in vx_api.c. */
#if defined(VOLVOXAI_ENABLE_TRAINING) && VOLVOXAI_ENABLE_TRAINING
#include "../api/vx_api_planning.c"
#include "../api/vx_api_training.c"
#include "../api/vx_api_quantization.c"
#endif
#include "../api/vx_api.c"
#include "public_api.c"

/* Native kernel fallback implementations under wasm */
#include "../kernels/cpu_features.c"
#include "../kernels/quant_cpu_isa.c"
#include "../kernels/conv_f32_isa.c"
#include "../backends/backend_manager.c"
#if VOLVOXAI_ENABLE_WEBGPU
#include "../backends/webgpu_backend.c"
#endif
#include "../kernels/paged_attention.c"
#include "../kernels/transpose_w8a8_native.c"
#include "../kernels/qgroupnorm_w8a8_native.c"
#include "../kernels/qnorm_activation_w8a8_native.c"
#include "../kernels/qsdpa_w8a8_native.c"
#include "../kernels/qbatch_matmul_w8a8_native.c"
#include "../kernels/qlinear_w8a8_x86.c"
#include "../kernels/qconv_w8a8_x86.c"
#include "../kernels/tensor_f32_isa.c"

#include "engine_runtime.c"

/* Unused native-only shader stubs */
void volvoxai_shader_store_shutdown(void) {}

/* Private, read-only transport preparation. The module pointer selects the
 * same registry as generated call dispatch; no ambient native owner exists. */
#define VX_CONTROL_WASM_EXPORT(name) \
    __attribute__((export_name(name), visibility("default"), used))

VX_CONTROL_WASM_EXPORT("vx_wasm_inference_path_preflight_v1")
char* vx_wasm_inference_path_preflight_v1(
    SynurangInstance* instance,
    const char* method, const char* data, int data_len, int* response_len) {
    VxApiRegistry* registry = vx_api_module_registry(instance);
    uint8_t* encoded = NULL;
    size_t size = 0;
    if (!response_len) return NULL;
    /* Negative lengths report private preflight failures without allocating a
     * response: -3 malformed protobuf, -13 internal preparation failure. */
    *response_len = -13;
    if (!registry || !method || data_len < 0 || (!data && data_len) ||
        !vx_api_wasm_path_preflight_begin()) return NULL;
#define VX_PREFLIGHT(Method, Request, Response, request_codec, response_codec, handler) \
    if (strcmp(method, Method) == 0) { \
        Request request; Response response; \
        request_codec##_init(&request); response_codec##_init(&response); \
        if (request_codec##_decode(&request, (const uint8_t*)data, (size_t)data_len) != SYNURANG_LITE_OK) \
            *response_len = -3; \
        else if (handler(&request, &response, registry) == 0) \
            (void)response_codec##_encode(&response, &encoded, &size); \
        request_codec##_free(&request); response_codec##_free(&response); \
    }
    VX_PREFLIGHT(VX_RPC_VX_INFERENCE_SERVICE_LOAD_MODEL,
        VolvoxaiV1LoadModelRequest, VolvoxaiV1ModelHandle,
        volvoxai_v1_load_model_request, volvoxai_v1_model_handle, vx_api_load_model)
    VX_PREFLIGHT(VX_RPC_VX_INFERENCE_SERVICE_PUBLISH_ADAPTER,
        VolvoxaiV1PublishAdapterRequest, VolvoxaiV1AdapterRevision,
        volvoxai_v1_publish_adapter_request, volvoxai_v1_adapter_revision, vx_api_publish_adapter)
#undef VX_PREFLIGHT
    vx_api_wasm_path_preflight_end();
    if (!encoded || size > INT32_MAX) { free(encoded); return NULL; }
    *response_len = (int)size;
    return (char*)encoded;
}

#if VOLVOXAI_ENABLE_WEBGPU
static char vx_wasm_model_gpu_preparation(VxModel* model) {
    VxRuntime* runtime = model->runtime;
    pthread_mutex_lock(&runtime->mutex);
    VxTraceScope scope = {.trace = runtime->trace, .work = {.index = -1}};
    int timing = atomic_load_explicit(&runtime->profiling_enabled, memory_order_acquire) &&
        vx_trace_device_enabled(&scope);
    pthread_mutex_unlock(&runtime->mutex);
    return 1 | (timing ? 2 : 0);
}
/* Read-only device preparation for the asynchronous browser transport. The
 * generated proto decoder and the ordinary C policy validator decide whether
 * this request needs a physical GPU. Invalid requests reach normal dispatch
 * unchanged, without allocating a device. No engine operation is executed or
 * replayed here. Response bits: 1=prepare GPU, 2=prepare optional timestamps.
 * Timestamp preparation alone must not acquire a device for a CPU runtime. */
VX_CONTROL_WASM_EXPORT("vx_wasm_prepare_gpu_v1")
char* vx_wasm_prepare_gpu_v1(
    SynurangInstance* instance,
    const char* method,
    const char* data,
    int data_len,
    int* response_len) {
    char* response;
    if (!response_len) return NULL;
    *response_len = 0;
    response = malloc(1);
    if (!response) return NULL;
    *response = 0;
    if (method && data_len >= 0 && (data || !data_len) &&
        strcmp(method, VX_RPC_VX_PROFILING_SERVICE_START_TRACE) == 0) {
        VolvoxaiV1StartTraceRequest request;
        VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
        volvoxai_v1_start_trace_request_init(&request);
        if (volvoxai_v1_start_trace_request_decode(&request, (const uint8_t*)data,
                (size_t)data_len) == SYNURANG_LITE_OK && request.field_device_timing &&
            (!request.has_capacity_bytes || (request.field_capacity_bytes >= 4096 &&
                request.field_capacity_bytes <= 64u * 1024u * 1024u)) &&
            (request.field_detail == VOLVOXAI_V1_TRACE_DETAIL_BASIC ||
                request.field_detail == VOLVOXAI_V1_TRACE_DETAIL_NODES) &&
            vx_api_handle_acquire(vx_api_module_registry(instance), VX_API_HANDLE_RUNTIME,
                request.field_runtime_id, &lease)) *response = 2;
        vx_api_handle_lease_release(&lease);
        volvoxai_v1_start_trace_request_free(&request);
    }
    if (method && data_len >= 0 && (data || !data_len) &&
        strcmp(method, VX_RPC_VX_INFERENCE_SERVICE_COMPILE_MODEL) == 0) {
        VolvoxaiV1CompileModelRequest request;
        VxApiScratch scratch = VX_API_SCRATCH_INIT;
        VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
        volvoxai_v1_compile_model_request_init(&request);
        if (volvoxai_v1_compile_model_request_decode(
                &request, (const uint8_t*)data, (size_t)data_len) == SYNURANG_LITE_OK &&
            vx_api_handle_acquire(vx_api_module_registry(instance), VX_API_HANDLE_MODEL, request.field_model_id, &lease)) {
            VxBackendPolicy policy = vx_api_compile_policy(&request, &scratch);
            if (vx_policy_valid(&policy))
                for (size_t index = 0; index < policy.backend_count; index++)
                    if (vx_backend_kind_from_name(policy.backends[index]) == VX_BACKEND_KIND_WEBGPU) {
                        *response = vx_wasm_model_gpu_preparation(lease.pointer);
                        break;
                    }
        }
        vx_api_handle_lease_release(&lease);
        vx_api_scratch_release(&scratch);
        volvoxai_v1_compile_model_request_free(&request);
    }
#if VOLVOXAI_ENABLE_TRAINING
    if (method && data_len >= 0 && (data || !data_len) &&
        strcmp(method, VX_RPC_VX_TRAINING_SERVICE_CREATE_TRAINER) == 0) {
        VolvoxaiV1CreateTrainerRequest request;
        VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
        volvoxai_v1_create_trainer_request_init(&request);
        if (volvoxai_v1_create_trainer_request_decode(&request, (const uint8_t*)data,
                (size_t)data_len) == SYNURANG_LITE_OK &&
            vx_backend_kind_from_bytes(request.field_backend.data,
                request.field_backend.len) == VX_BACKEND_KIND_WEBGPU &&
            vx_api_handle_acquire(vx_api_module_registry(instance), VX_API_HANDLE_MODEL, request.field_model_id, &lease))
            *response = vx_wasm_model_gpu_preparation(lease.pointer);
        vx_api_handle_lease_release(&lease);
        volvoxai_v1_create_trainer_request_free(&request);
    }
#endif
    *response_len = 1;
    return response;
}
#endif

#undef VX_CONTROL_WASM_EXPORT
