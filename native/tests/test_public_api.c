#include "volvoxai.h"
#include "volvoxai_backend.h"
#include "engine_core.h"
#include "safetensors.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

_Static_assert(VX_STATUS_OK == 0, "protobuf status contract");
_Static_assert(VX_STATUS_INVALID_ARGUMENT == -1, "protobuf status contract");
_Static_assert(VX_STATUS_OUT_OF_MEMORY == -2, "protobuf status contract");
_Static_assert(VX_STATUS_HANDLE_DISPOSED == -3, "protobuf status contract");
_Static_assert(VX_STATUS_IO_ERROR == -4, "protobuf status contract");
_Static_assert(VX_STATUS_INVALID_GRAPH == -5, "protobuf status contract");
_Static_assert(VX_STATUS_BACKEND_UNAVAILABLE == -6, "protobuf status contract");
_Static_assert(VX_STATUS_BACKEND_UNSUPPORTED == -7, "protobuf status contract");
_Static_assert(VX_STATUS_BACKEND_REQUIRED == -8, "protobuf status contract");
_Static_assert(VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN == -9,
               "protobuf status contract");
_Static_assert(VX_STATUS_EXECUTION_FAILED == -10, "protobuf status contract");
_Static_assert(VX_STATUS_RESULT_DISPOSED == -11, "protobuf status contract");
_Static_assert(VX_STATUS_DEVICE_LOST == -12, "protobuf status contract");
_Static_assert(VX_STATUS_ABI_UNSUPPORTED == -13, "protobuf status contract");
_Static_assert(VX_STATUS_NOT_FOUND == -14, "protobuf status contract");
_Static_assert(VX_STATUS_BUFFER_TOO_SMALL == -15, "protobuf status contract");
_Static_assert(VX_STATUS_INTERNAL == -16, "protobuf status contract");
_Static_assert(VX_STAGE_NONE == 0 && VX_STAGE_RUNTIME_CREATE == 1 &&
               VX_STAGE_MODEL_LOAD == 2 && VX_STAGE_COMPILE == 3 &&
               VX_STAGE_CONTEXT_CREATE == 4 && VX_STAGE_INPUT == 5 &&
               VX_STAGE_EXECUTE == 6 && VX_STAGE_READBACK == 7 &&
               VX_STAGE_CLOSE == 8 && VX_STAGE_DECODE == 9 &&
               VX_STAGE_ADAPTER == 10, "protobuf stage contract");
_Static_assert(VX_DTYPE_UNSPECIFIED == 0 && VX_DTYPE_BOOL == 1 &&
               VX_DTYPE_F4 == 2 && VX_DTYPE_F6_E2M3 == 3 &&
               VX_DTYPE_F6_E3M2 == 4 && VX_DTYPE_U8 == 5 &&
               VX_DTYPE_I8 == 6 && VX_DTYPE_F8_E5M2 == 7 &&
               VX_DTYPE_F8_E4M3 == 8 && VX_DTYPE_F8_E8M0 == 9 &&
               VX_DTYPE_F8_E4M3FNUZ == 10 &&
               VX_DTYPE_F8_E5M2FNUZ == 11 && VX_DTYPE_I16 == 12 &&
               VX_DTYPE_U16 == 13 && VX_DTYPE_F16 == 14 &&
               VX_DTYPE_BF16 == 15 && VX_DTYPE_I32 == 16 &&
               VX_DTYPE_U32 == 17 && VX_DTYPE_F32 == 18 &&
               VX_DTYPE_C64 == 19 && VX_DTYPE_F64 == 20 &&
               VX_DTYPE_I64 == 21 && VX_DTYPE_U64 == 22,
               "protobuf dtype contract");
_Static_assert(SAFETENSORS_DTYPE_UNKNOWN == VX_DTYPE_UNSPECIFIED &&
               SAFETENSORS_DTYPE_U8 == VX_DTYPE_U8 &&
               SAFETENSORS_DTYPE_I8 == VX_DTYPE_I8 &&
               SAFETENSORS_DTYPE_F16 == VX_DTYPE_F16 &&
               SAFETENSORS_DTYPE_I32 == VX_DTYPE_I32 &&
               SAFETENSORS_DTYPE_F32 == VX_DTYPE_F32 &&
               SAFETENSORS_DTYPE_U64 == VX_DTYPE_U64,
               "safetensors uses protobuf dtype contract");
_Static_assert(VX_BACKEND_PREFER == 0 && VX_BACKEND_REQUIRE == 1,
               "protobuf policy contract");
_Static_assert(VX_OPERATOR_FALLBACK_ALLOW == 0 &&
               VX_OPERATOR_FALLBACK_FORBID == 1,
               "protobuf fallback contract");
_Static_assert(VX_MEMORY_HOST == 0 && VX_MEMORY_DEVICE == 1,
               "protobuf memory contract");
_Static_assert(VX_DECODE_ROW_DISABLED == 0 && VX_DECODE_ROW_AUTO == 1 &&
               VX_DECODE_ROW_REQUIRED == 2, "protobuf decode contract");

typedef void (*VxPublicApiCpuExecuteHook)(void* user_data);
extern void vx_public_api_test_set_cpu_execute_hook(
    VxPublicApiCpuExecuteHook hook, void* user_data);
extern VxStatus vx_public_api_test_evaluate_builtin_route(
    const char* selected_backend,
    const char* actual_route,
    VxOperatorFallback operator_fallback,
    VxReport* report);
extern int vx_public_api_test_cpu_typed_workspace_state(
    const VxExecutionContext* context, uintptr_t* address,
    size_t* bound_bytes, size_t* capacity_bytes);
extern int vx_public_api_test_cpu_typed_workspace_reconfigure(
    VxExecutionContext* context, size_t bounded_bytes);
extern int vx_public_api_test_compiled_resource_bounds(
    const VxCompiledModel* compiled, size_t* maximum_typed_scratch_bytes,
    uint64_t* maximum_resident_bytes);
extern int vx_public_api_test_native_gpu_align_resource_bytes(
    uint64_t bytes, uint64_t alignment, uint64_t* aligned_out);
extern int vx_public_api_test_native_gpu_immutable_components(
    uint64_t component_bytes, uint64_t component_count,
    uint64_t device_alignment, uint64_t* resident_out,
    uint64_t* device_out);
extern int vx_public_api_test_native_gpu_maskless_qsdpa_immutable_bound(
    int backend, int has_mask, uint64_t device_alignment,
    uint64_t* resident_out, uint64_t* device_out);
extern int vx_public_api_test_native_gpu_zero_bias_backing_bound(
    uint64_t count, uint64_t* bytes_out);
extern int vx_public_api_test_native_gpu_activation_resource_peak(
    int backend, uint64_t bootstrap_host_bytes,
    uint64_t maximum_host_arena_bytes, uint64_t bootstrap_device_bytes,
    uint64_t candidate_device_bytes, uint64_t qgroupnorm_stats_bytes,
    uint64_t qlayernorm_stats_bytes, uint64_t dispatch_temporary_bytes,
    uint64_t vulkan_reserved_scratch_bytes,
    uint64_t* peak_out);
extern int vx_public_api_test_native_gpu_conv2d_dispatch_temporary_bound(
    uint64_t output_channels, int has_bias, uint64_t parameter_blocks,
    uint64_t device_alignment, uint64_t* bytes_out);
extern int vx_public_api_test_native_gpu_qlinear_dispatch_host_bound(
    uint64_t output_channels, uint64_t* bytes_out);
extern int vx_public_api_test_native_gpu_opengl_packed_clear_bound(
    uint64_t output_bytes, uint64_t* bytes_out);
extern int vx_public_api_test_native_gpu_vulkan_scratch_capacity_proven(
    uint64_t qgroupnorm_stats_bytes, uint64_t qlayernorm_stats_bytes,
    uint64_t validation_graph_scratch_bytes, uint64_t storage_alignment,
    uint64_t maximum_scratch_bytes, uint64_t* required_out);
extern int vx_public_api_test_native_gpu_device_capacity_proven(
    uint64_t immutable_device_bytes, uint64_t bootstrap_device_bytes,
    uint64_t candidate_device_bytes, uint64_t hard_limit_bytes,
    uint64_t* peak_out);
extern int vx_public_api_test_native_gpu_slot_capacity_proven(
    uint64_t tensor_count, uint64_t node_count,
    uint64_t physical_span_count, uint64_t slot_limit,
    uint64_t* required_out);
extern int vx_public_api_test_native_gpu_result_publication_peak(
    const uint64_t* output_bytes, size_t output_count,
    uint64_t* snapshot_bytes_out, uint64_t* publication_peak_out);
extern int vx_public_api_test_native_gpu_dynamic_metadata_peak(
    uint64_t signature_bytes, uint64_t input_count,
    uint64_t logical_tensor_count, uint64_t engine_tensor_count,
    uint64_t physical_span_count, uint64_t* peak_out);
extern int vx_public_api_test_native_gpu_resident_capacity_proven(
    uint64_t base_bytes, uint64_t result_publication_bytes,
    uint64_t dynamic_metadata_bytes, uint64_t hard_limit_bytes,
    uint64_t* required_out);
extern int vx_public_api_test_native_gpu_dense_launch_proven(
    uint64_t rows, uint64_t input_width, uint64_t output_width,
    int quantized, int backend, uint64_t maximum_grid_x,
    uint64_t maximum_grid_y, uint64_t maximum_block_x,
    uint64_t maximum_block_y, uint64_t maximum_threads_per_block);
extern int vx_public_api_test_native_gpu_quantized_accumulator_channel_proven(
    uint64_t terms, VxDataType input_dtype, int32_t input_zero_point,
    VxDataType weight_dtype, int32_t weight_zero_point,
    const int32_t* bias);

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

static int test_native_gpu_resource_peak_projection(void) {
    enum {
        TEST_BACKEND_VULKAN = 1,
        TEST_BACKEND_OPENGL = 2,
        TEST_BACKEND_METAL = 3,
        TEST_BACKEND_CUDA = 6,
    };
    uint64_t value = 0u;
    uint64_t device_value = 0u;
    uint64_t snapshot_value = 0u;
    CHECK(vx_public_api_test_native_gpu_align_resource_bytes(
              5u, 64u, &value) == 1 && value == 64u);
    CHECK(vx_public_api_test_native_gpu_align_resource_bytes(
              UINT64_MAX, 64u, &value) == 0);
    CHECK(vx_public_api_test_native_gpu_immutable_components(
              4u, 3u, 256u, &value, &device_value) == 1 &&
          value == 780u && device_value == 768u);
    CHECK(vx_public_api_test_native_gpu_maskless_qsdpa_immutable_bound(
              TEST_BACKEND_VULKAN, 0, 256u, &value, &device_value) == 1 &&
          value == 260u && device_value == 256u);
    CHECK(vx_public_api_test_native_gpu_maskless_qsdpa_immutable_bound(
              TEST_BACKEND_OPENGL, 1, 256u, &value, &device_value) == 1 &&
          value == 0u && device_value == 0u);
    CHECK(vx_public_api_test_native_gpu_maskless_qsdpa_immutable_bound(
              TEST_BACKEND_CUDA, 0, 256u, &value, &device_value) == 1 &&
          value == 0u && device_value == 0u);
    CHECK(vx_public_api_test_native_gpu_maskless_qsdpa_immutable_bound(
              TEST_BACKEND_METAL, 0, 0u, &value, &device_value) == 0);
    CHECK(vx_public_api_test_native_gpu_zero_bias_backing_bound(
              3u, &value) == 1 &&
          value == 3u * (sizeof(void*) * 2u + sizeof(size_t)));
    CHECK(vx_public_api_test_native_gpu_zero_bias_backing_bound(
              UINT64_MAX, &value) == 0);

    /* The 68-byte host arena represents two odd-byte physical spans with a
     * 64-byte host offset between them. Vulkan/CUDA reuse or release the
     * bootstrap device phase; Vulkan still owns the supplied host dispatch
     * transient. OpenGL/Metal publish only after old and new allocations
     * (including both qnorm stats buffers) coexist. */
    CHECK(vx_public_api_test_native_gpu_activation_resource_peak(
              TEST_BACKEND_VULKAN, 3u, 68u, 64u, 128u,
              8u, 16u, 64u, 88u, &value) == 1 && value == 351u);
    CHECK(vx_public_api_test_native_gpu_activation_resource_peak(
              TEST_BACKEND_OPENGL, 3u, 68u, 64u, 128u,
              8u, 16u, 64u, 0u, &value) == 1 && value == 375u);
    CHECK(vx_public_api_test_native_gpu_activation_resource_peak(
              TEST_BACKEND_METAL, 3u, 68u, 64u, 128u,
              8u, 16u, 64u, 0u, &value) == 1 && value == 375u);
    CHECK(vx_public_api_test_native_gpu_activation_resource_peak(
              TEST_BACKEND_CUDA, 3u, 68u, 64u, 128u,
              8u, 16u, 0u, 0u, &value) == 1 && value == 199u);

    /* One 256-byte parameter buffer plus a missing-bias host/device pair.
     * With 1025 output channels the old fixed 256-byte estimate was far too
     * small: 4100 host bytes and a 4352-byte aligned device allocation coexist. */
    CHECK(vx_public_api_test_native_gpu_conv2d_dispatch_temporary_bound(
              1025u, 0, 1u, 256u, &value) == 1 && value == 8708u);
    CHECK(vx_public_api_test_native_gpu_conv2d_dispatch_temporary_bound(
              1025u, 1, 1u, 256u, &value) == 1 && value == 256u);
    CHECK(vx_public_api_test_native_gpu_conv2d_dispatch_temporary_bound(
              UINT64_MAX, 0, 1u, 256u, &value) == 0);
    CHECK(vx_public_api_test_native_gpu_qlinear_dispatch_host_bound(
              1025u, &value) == 1 && value == 4100u);
    CHECK(vx_public_api_test_native_gpu_qlinear_dispatch_host_bound(
              UINT64_MAX, &value) == 0);
    CHECK(vx_public_api_test_native_gpu_opengl_packed_clear_bound(
              1025u, &value) == 1 && value == 1028u);
    CHECK(vx_public_api_test_native_gpu_opengl_packed_clear_bound(
              UINT64_MAX, &value) == 0);

    CHECK(vx_public_api_test_native_gpu_vulkan_scratch_capacity_proven(
              1u, 257u, 256u, 256u, 1024u, &value) == 1 &&
          value == 1024u);
    CHECK(vx_public_api_test_native_gpu_vulkan_scratch_capacity_proven(
              1u, 257u, 256u, 256u, 1023u, &value) == 0 &&
          value == 1024u);
    CHECK(vx_public_api_test_native_gpu_vulkan_scratch_capacity_proven(
              UINT64_MAX, 1u, 0u, 256u, UINT64_MAX, &value) == 0);

    CHECK(vx_public_api_test_native_gpu_device_capacity_proven(
              256u, 64u, 512u, 768u, &value) == 1 && value == 768u);
    CHECK(vx_public_api_test_native_gpu_device_capacity_proven(
              256u, 64u, 512u, 767u, &value) == 0 && value == 768u);
    CHECK(vx_public_api_test_native_gpu_device_capacity_proven(
              UINT64_MAX, 1u, 1u, 0u, &value) == 0);
    /* Bootstrap T slots and published physical spans are different phases;
     * charging the complete tensor table twice can spuriously reject a large
     * graph even though its true conservative slot bound fits. */
    CHECK(vx_public_api_test_native_gpu_slot_capacity_proven(
              894u, 532u, 64u, 4096u, &value) == 1 && value == 3086u);
    CHECK(vx_public_api_test_native_gpu_slot_capacity_proven(
              1000u, 800u, 100u, 4096u, &value) == 0 && value == 4300u);
    CHECK(vx_public_api_test_native_gpu_slot_capacity_proven(
              UINT64_MAX, 1u, 1u, UINT64_MAX, &value) == 0);

    {
        const uint64_t outputs[] = {10u, 100u, 20u};
        CHECK(vx_public_api_test_native_gpu_result_publication_peak(
                  outputs, 3u, &snapshot_value, &value) == 1 &&
              snapshot_value == 130u && value == 210u);
        CHECK(vx_public_api_test_native_gpu_result_publication_peak(
                  (const uint64_t[]){UINT64_MAX}, 1u,
                  &snapshot_value, &value) == 0);
    }

    {
        const uint64_t signature_bytes = 17u;
        const uint64_t input_count = 2u;
        const uint64_t logical_count = 3u;
        const uint64_t engine_count = 5u;
        const uint64_t span_count = 2u;
        const uint64_t plan_bytes = signature_bytes + logical_count *
            (10u * sizeof(int) + 2u * sizeof(size_t));
        const uint64_t maximum_layout_bytes = logical_count *
            (sizeof(int) + 4u * sizeof(size_t));
        const uint64_t layout_temporary = logical_count *
                (sizeof(VolvoxAIEngineMaximumTensor) +
                 5u * sizeof(int) + 2u * sizeof(size_t)) +
            engine_count * 3u * sizeof(int);
        const uint64_t layout_peak =
            maximum_layout_bytes + layout_temporary;
        const uint64_t request_bytes = logical_count *
                sizeof(VolvoxAIEngineResolvedTensor) +
            signature_bytes + input_count *
                sizeof(VolvoxAIEngineInputBinding);
        const uint64_t plan_build_temporary = engine_count * sizeof(int);
        const uint64_t bind_temporary = sizeof(void*) + logical_count *
                (sizeof(int) + sizeof(void*)) + span_count *
                sizeof(VolvoxAIEnginePhysicalSpan);
        const uint64_t transaction_temporary = plan_build_temporary >
                bind_temporary ? plan_build_temporary : bind_temporary;
        const uint64_t transaction_peak = maximum_layout_bytes +
            5u * plan_bytes + request_bytes + transaction_temporary;
        const uint64_t expected = layout_peak > transaction_peak
            ? layout_peak : transaction_peak;
        CHECK(vx_public_api_test_native_gpu_dynamic_metadata_peak(
                  signature_bytes, input_count, logical_count,
                  engine_count, span_count, &value) == 1 &&
              value == expected);
        CHECK(vx_public_api_test_native_gpu_dynamic_metadata_peak(
                  1u, 1u, UINT64_MAX, 1u, 1u, &value) == 0);
    }

    CHECK(vx_public_api_test_native_gpu_resident_capacity_proven(
              100u, 50u, 25u, 175u, &value) == 1 && value == 175u);
    CHECK(vx_public_api_test_native_gpu_resident_capacity_proven(
              100u, 50u, 25u, 174u, &value) == 0 && value == 175u);
    CHECK(vx_public_api_test_native_gpu_resident_capacity_proven(
              UINT64_MAX, 1u, 0u, 0u, &value) == 0);

    /* CUDA F32 dense is exactly 16x16 and independently caps grid-y at
     * 65535. CUDA physical QLinear uses the common 256-thread 1-D launcher;
     * V/O/M retain their 64-lane scalar/tiled contracts. */
    CHECK(vx_public_api_test_native_gpu_dense_launch_proven(
              65535u * 16u, 32u, 32u, 0, TEST_BACKEND_CUDA,
              UINT32_MAX, UINT32_MAX, 1024u, 1024u, 1024u) == 1);
    CHECK(vx_public_api_test_native_gpu_dense_launch_proven(
              65535u * 16u + 1u, 32u, 32u, 0, TEST_BACKEND_CUDA,
              UINT32_MAX, UINT32_MAX, 1024u, 1024u, 1024u) == 0);
    CHECK(vx_public_api_test_native_gpu_dense_launch_proven(
              16u, 32u, 32u, 0, TEST_BACKEND_CUDA,
              UINT32_MAX, UINT32_MAX, 16u, 15u, 256u) == 0);
    CHECK(vx_public_api_test_native_gpu_dense_launch_proven(
              1024u, 32u, 32u, 1, TEST_BACKEND_CUDA,
              128u, 1u, 256u, 1u, 256u) == 1);
    CHECK(vx_public_api_test_native_gpu_dense_launch_proven(
              1024u, 32u, 32u, 1, TEST_BACKEND_CUDA,
              128u, 1u, 128u, 1u, 256u) == 0);
    CHECK(vx_public_api_test_native_gpu_dense_launch_proven(
              32u, 32u, 32u, 0, TEST_BACKEND_VULKAN,
              128u, 128u, 64u, 64u, 64u) == 1);
    {
        const int32_t fitting_bias = 16383;
        const int32_t overflowing_bias = 16384;
        CHECK(vx_public_api_test_native_gpu_quantized_accumulator_channel_proven(
                  131071u, VX_DTYPE_I8, 0, VX_DTYPE_I8, 0,
                  NULL) == 1);
        CHECK(vx_public_api_test_native_gpu_quantized_accumulator_channel_proven(
                  131071u, VX_DTYPE_I8, 0, VX_DTYPE_I8, 0,
                  &fitting_bias) == 1);
        CHECK(vx_public_api_test_native_gpu_quantized_accumulator_channel_proven(
                  131071u, VX_DTYPE_I8, 0, VX_DTYPE_I8, 0,
                  &overflowing_bias) == 0);
    }
    return 0;
}

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    size_t length = strlen(text);
    int ok = fwrite(text, 1, length, file) == length && fclose(file) == 0;
    return ok ? 0 : -1;
}

static int write_typed_weight(const char* path,
                              VxDataType dtype,
                              const void* value,
                              size_t byte_size) {
    const int shape[1] = {1};
    SafetensorsFile file;
    int status;
    if (safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0)
        return -1;
    if (safetensors_add_tensor(&file, "w", dtype, shape, 1, value,
                               byte_size) != 0) {
        safetensors_free(&file);
        return -1;
    }
    status = safetensors_save(path, &file);
    safetensors_free(&file);
    return status;
}

static int write_weight(const char* path, float value) {
    return write_typed_weight(path, SAFETENSORS_DTYPE_F32, &value,
                              sizeof(value));
}

static int write_bank_weights(const char* path,
                              const char* name,
                              int slot_count,
                              int rank) {
    int shape[2] = {slot_count, 1};
    float values[16] = {0};
    SafetensorsFile file;
    int status;
    if (!path || !name || slot_count <= 0 || slot_count > 16 ||
        (rank != 1 && rank != 2)) return -1;
    for (int index = 0; index < slot_count; index++)
        values[index] = (float)(index + 1);
    if (safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0)
        return -1;
    if (safetensors_add_tensor(&file, name, SAFETENSORS_DTYPE_F32,
                               shape, rank, values,
                               (size_t)slot_count * sizeof(values[0])) != 0) {
        safetensors_free(&file);
        return -1;
    }
    status = safetensors_save(path, &file);
    safetensors_free(&file);
    return status;
}

static int write_provider_bank_weights(const char* path) {
    return write_bank_weights(path, "experts", 4, 2);
}

static int write_zero_payload_bank_weights(const char* path,
                                           const char* name) {
    const int shape[2] = {4, 0};
    SafetensorsFile file;
    int status;
    if (!path || !name) return -1;
    if (safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0)
        return -1;
    if (safetensors_add_tensor(&file, name, SAFETENSORS_DTYPE_F32,
                               shape, 2, NULL, 0u) != 0) {
        safetensors_free(&file);
        return -1;
    }
    status = safetensors_save(path, &file);
    safetensors_free(&file);
    return status;
}

static int write_many_bank_package(const char* graph_path,
                                   const char* weights_path,
                                   size_t bank_count) {
    const int shape[2] = {1, 1};
    SafetensorsFile weights;
    FILE* graph = NULL;
    int weights_initialized = 0;
    int status = -1;
    if (!graph_path || !weights_path || !bank_count) return -1;
    graph = fopen(graph_path, "wb");
    if (!graph) return -1;
    if (fputs("{\"format\":\"volvox-graph/v1\","
              "\"dimensions\":{\"F\":{\"min\":1,\"max\":1,"
              "\"multiple_of\":1}},"
              "\"inputs\":{\"x\":{\"shape\":[1],"
              "\"dtype\":\"float32\"}},"
              "\"nodes\":[{\"id\":\"identity\","
              "\"opType\":\"Identity\","
              "\"inputs\":{\"input\":\"x\"},"
              "\"outputs\":{\"out\":{\"tensor\":\"y\","
              "\"dtype\":\"float32\",\"shape\":[1]}},"
              "\"params\":{}}],\"outputs\":[\"y\"],\"banks\":{",
              graph) == EOF)
        goto done;
    for (size_t index = 0; index < bank_count; index++) {
        if (fprintf(graph, "%s\"bank_%zu\":\"F\"",
                    index ? "," : "", index) < 0)
            goto done;
    }
    if (fputs("}}", graph) == EOF) goto done;
    if (fclose(graph) != 0) {
        graph = NULL;
        goto done;
    }
    graph = NULL;
    if (safetensors_init_empty(&weights, SAFETENSORS_OPEN_READ_WRITE) != 0)
        goto done;
    weights_initialized = 1;
    for (size_t index = 0; index < bank_count; index++) {
        char name[32];
        float value = (float)(index + 1u);
        int length = snprintf(name, sizeof(name), "bank_%zu", index);
        if (length < 0 || (size_t)length >= sizeof(name) ||
            safetensors_add_tensor(&weights, name, SAFETENSORS_DTYPE_F32,
                                   shape, 2, &value, sizeof(value)) != 0)
            goto done;
    }
    if (safetensors_save(weights_path, &weights) != 0) goto done;
    status = 0;
done:
    if (graph && fclose(graph) != 0) status = -1;
    if (weights_initialized) safetensors_free(&weights);
    return status;
}

static int write_qbatch_workspace_weights(const char* path) {
    const int shape[1] = {1};
    const float scale = 0.125f;
    const uint8_t a_zero = 128u;
    const int8_t signed_zero = 0;
    SafetensorsFile file;
    int status;
    if (safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0)
        return -1;
#define ADD_QBATCH_PARAMETER(name, dtype, value) \
    if (safetensors_add_tensor(&file, (name), (dtype), shape, 1, \
                               &(value), sizeof(value)) != 0) { \
        safetensors_free(&file); \
        return -1; \
    }
    ADD_QBATCH_PARAMETER("a.scale", SAFETENSORS_DTYPE_F32, scale);
    ADD_QBATCH_PARAMETER("a.zero", SAFETENSORS_DTYPE_U8, a_zero);
    ADD_QBATCH_PARAMETER("b.scale", SAFETENSORS_DTYPE_F32, scale);
    ADD_QBATCH_PARAMETER("b.zero", SAFETENSORS_DTYPE_I8, signed_zero);
    ADD_QBATCH_PARAMETER("y.scale", SAFETENSORS_DTYPE_F32, scale);
    ADD_QBATCH_PARAMETER("y.zero", SAFETENSORS_DTYPE_I8, signed_zero);
#undef ADD_QBATCH_PARAMETER
    status = safetensors_save(path, &file);
    safetensors_free(&file);
    return status;
}

static int read_text(const char* path, char* output, size_t capacity) {
    FILE* file;
    size_t count;
    int failed;
    if (!path || !output || capacity < 2u) return -1;
    file = fopen(path, "rb");
    if (!file) return -1;
    count = fread(output, 1, capacity - 1u, file);
    failed = ferror(file) || !feof(file);
    if (fclose(file) != 0) failed = 1;
    if (failed) return -1;
    output[count] = '\0';
    return 0;
}

static int closef(float a, float b) { return fabsf(a - b) < 1e-6f; }

static int create_cpu_context(VxRuntime* runtime,
                              const char* graph_path,
                              VxModel** out_model,
                              VxCompiledModel** out_compiled,
                              VxExecutionContext** out_context) {
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    source.graph_path = graph_path;
    if (vx_runtime_load_model(runtime, &source, out_model, &report) != VX_STATUS_OK)
        return -1;
    if (vx_model_compile(*out_model, &policy, out_compiled, &report) != VX_STATUS_OK)
        return -1;
    if (strcmp(report.backend, "cpu") != 0 || !report.runtime_id ||
        !report.model_id || !report.compiled_model_id || !report.graph_id ||
        !report.graph_revision || !report.weight_id || !report.weight_revision ||
        !report.adapter_id || !report.adapter_revision ||
        report.candidate_count != 1 || !report.route_attested ||
        report.operator_fallback_used || !report.route_evidence[0] ||
        report.compile_time_ms < 0.0) return -1;
    return vx_compiled_model_create_context(*out_compiled, &context_options,
                                             out_context, &report) == VX_STATUS_OK
        ? 0 : -1;
}

typedef struct ThreadCase {
    VxExecutionContext* context;
    float base;
    int ok;
} ThreadCase;

typedef struct CpuOverlapProbe {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int entered;
    int timed_out;
} CpuOverlapProbe;

typedef struct CpuInputCommitProbe {
    float* values;
    size_t count;
    float replacement;
    int called;
} CpuInputCommitProbe;

static void cpu_input_commit_hook(void* opaque) {
    CpuInputCommitProbe* probe = (CpuInputCommitProbe*)opaque;
    if (!probe || !probe->values) return;
    for (size_t index = 0; index < probe->count; index++)
        probe->values[index] = probe->replacement;
    probe->called++;
}

static void cpu_overlap_hook(void* opaque) {
    CpuOverlapProbe* probe = (CpuOverlapProbe*)opaque;
    struct timespec deadline;
    int wait_status = 0;
    pthread_mutex_lock(&probe->mutex);
    if (probe->entered < 2) probe->entered++;
    pthread_cond_broadcast(&probe->condition);
    timespec_get(&deadline, TIME_UTC);
    deadline.tv_sec += 3;
    while (probe->entered < 2 && !probe->timed_out && wait_status != ETIMEDOUT)
        wait_status = pthread_cond_timedwait(&probe->condition, &probe->mutex,
                                             &deadline);
    if (probe->entered < 2) {
        probe->timed_out = 1;
        pthread_cond_broadcast(&probe->condition);
    }
    pthread_mutex_unlock(&probe->mutex);
}

static void* run_thread_case(void* opaque) {
    ThreadCase* test = (ThreadCase*)opaque;
    test->ok = 1;
    for (int iteration = 0; iteration < 20; iteration++) {
        float a[2] = {test->base + iteration, test->base + iteration + 1.0f};
        float b[2] = {2.0f, -3.0f};
        float output[2] = {0};
        VxResult* result = NULL;
        const VxTensorBinding inputs[2] = {
            {sizeof(VxTensorBinding), "a", VX_DTYPE_F32, 1u, {2},
             a, sizeof(a), VX_MEMORY_HOST},
            {sizeof(VxTensorBinding), "b", VX_DTYPE_F32, 1u, {2},
             b, sizeof(b), VX_MEMORY_HOST},
        };
        if (vx_execution_context_execute(
                test->context, inputs, 2u, &result, NULL) != VX_STATUS_OK ||
            vx_result_read(result, "final", output, sizeof(output), NULL, NULL) !=
                VX_STATUS_OK ||
            !closef(output[0], a[0] + 2.0f * b[0]) ||
            !closef(output[1], a[1] + 2.0f * b[1])) {
            test->ok = 0;
            vx_result_release(result);
            return NULL;
        }
        vx_result_release(result);
    }
    return NULL;
}

typedef struct MockRuntime {
    int marker;
    char provider_name[VX_BACKEND_NAME_CAPACITY];
} MockRuntime;
typedef struct MockCompiled { int marker; } MockCompiled;
typedef struct MockContext { float input; } MockContext;

static int mock_runtime_destroyed;
static int mock_compiled_destroyed;
static int mock_context_destroyed;
static int mock_context_closed;
static int mock_adapter_selected;
static uint64_t mock_adapter_id;
static uint64_t mock_adapter_revision;
static int mock_adapter_package_was_null;
static char mock_adapter_version[96];
static char mock_adapter_package[96];
static const VxBankResidency* mock_caller_bank_residency;
static const char* mock_caller_bank_name;
static const uint32_t* mock_caller_bank_slots;
static int mock_bank_source_seen;

typedef enum MockOutputViolation {
    MOCK_OUTPUT_VALID = 0,
    MOCK_OUTPUT_MISSING,
    MOCK_OUTPUT_WRONG_NAME,
    MOCK_OUTPUT_WRONG_DTYPE,
    MOCK_OUTPUT_WRONG_RANK,
    MOCK_OUTPUT_WRONG_SHAPE,
    MOCK_OUTPUT_WRONG_BYTE_SIZE,
    MOCK_OUTPUT_DUPLICATE,
    MOCK_OUTPUT_F16,
    MOCK_OUTPUT_UNATTESTED_ROUTE
} MockOutputViolation;

static MockOutputViolation mock_output_violation;

static VxStatus mock_runtime_create(void* user_data,
                                    const VxRuntimeOptions* options,
                                    void** out,
                                    VxReport* report) {
    (void)options;
    (void)report;
    MockRuntime* runtime = (MockRuntime*)calloc(1, sizeof(*runtime));
    if (!runtime) return VX_STATUS_OUT_OF_MEMORY;
    runtime->marker = 11;
    snprintf(runtime->provider_name, sizeof(runtime->provider_name), "%s",
             user_data ? (const char*)user_data : "");
    *out = runtime;
    return VX_STATUS_OK;
}

static void mock_runtime_destroy(void* instance) {
    mock_runtime_destroyed++;
    free(instance);
}

static int mock_tensor_spec_layout_valid(const VxTensorSpec* spec) {
    if (!spec || spec->struct_size != sizeof(*spec) ||
        spec->rank > VX_MAX_TENSOR_RANK) return 0;
    for (uint32_t axis = 0; axis < spec->rank; axis++)
        if (spec->dimensions[axis].struct_size !=
            sizeof(spec->dimensions[axis])) return 0;
    return 1;
}

static int mock_compile_input_layout_valid(
    const VxBackendCompileInput* input) {
    if (!input || input->struct_size != sizeof(*input) || !input->source ||
        input->source->struct_size != sizeof(*input->source) ||
        (input->input_count && !input->inputs) ||
        (input->output_count && !input->outputs)) return 0;
    for (size_t index = 0; index < input->input_count; index++)
        if (!mock_tensor_spec_layout_valid(&input->inputs[index])) return 0;
    for (size_t index = 0; index < input->output_count; index++)
        if (!mock_tensor_spec_layout_valid(&input->outputs[index])) return 0;
    return 1;
}

static VxStatus mock_compile(void* runtime_instance,
                             const VxBackendCompileInput* input,
                             const VxBackendPolicy* policy,
                             void** out,
                             VxBackendShapeDomainAttestation* attestation,
                             VxReport* report) {
    MockRuntime* runtime = (MockRuntime*)runtime_instance;
    int selected_present = 0;
    if (!runtime || !mock_compile_input_layout_valid(input) ||
        !input->source->graph_path ||
        !input->graph_fingerprint || !input->shape_domain_proof_identity ||
        !policy || !policy->backends || !policy->backend_count ||
        !attestation || attestation->struct_size != sizeof(*attestation))
        return VX_STATUS_INVALID_ARGUMENT;
    for (size_t index = 0; index < policy->backend_count; index++)
        if (!strcmp(policy->backends[index], runtime->provider_name))
            selected_present = 1;
    if (!selected_present) return VX_STATUS_INVALID_ARGUMENT;
    if (!strcmp(runtime->provider_name, "test-provider")) {
        const VxModelSource* source = input->source;
        const VxBankResidency* residency = source->bank_residency;
        if (source->struct_size != sizeof(*source) ||
            source->weight_path_count != 1u || !source->weight_paths ||
            source->bank_residency_count != 1u || !residency ||
            source->bank_residency == mock_caller_bank_residency ||
            residency->struct_size != sizeof(*residency) ||
            !residency->bank || strcmp(residency->bank, "experts") ||
            residency->bank == mock_caller_bank_name ||
            residency->slot_count != 2u || !residency->slots ||
            residency->slots == mock_caller_bank_slots ||
            residency->slots[0] != 1u || residency->slots[1] != 3u)
            return VX_STATUS_INVALID_ARGUMENT;
        mock_bank_source_seen++;
    }
    MockCompiled* compiled = (MockCompiled*)calloc(1, sizeof(*compiled));
    if (!compiled) return VX_STATUS_OUT_OF_MEMORY;
    compiled->marker = 22;
    *out = compiled;
    attestation->graph_fingerprint = input->graph_fingerprint;
    attestation->shape_domain_proof_identity =
        input->shape_domain_proof_identity;
    attestation->maximum_tensor_bytes = sizeof(float);
    attestation->maximum_resident_bytes = 2u * sizeof(float);
    attestation->resource_limit_bytes = 1024u;
    attestation->has_resource_limit = 1;
    if (!strcmp(runtime->provider_name, "test-provider")) {
        report->route_attested = 1;
        report->operator_fallback_used = 0;
        snprintf(report->device, sizeof(report->device), "%s", "mock-device-0");
        snprintf(report->route_evidence, sizeof(report->route_evidence), "%s",
                 "provider=test-provider;nodes=2;route=test-provider-all");
        snprintf(report->fallback_evidence, sizeof(report->fallback_evidence),
                 "%s", "operator=none");
    }
    return VX_STATUS_OK;
}

static void mock_compiled_destroy(void* instance) {
    mock_compiled_destroyed++;
    free(instance);
}

static VxStatus mock_context_create(void* compiled_instance,
                                    const VxContextOptions* options,
                                    void** out,
                                    VxReport* report) {
    (void)options;
    (void)report;
    if (!compiled_instance) return VX_STATUS_INVALID_ARGUMENT;
    MockContext* context = (MockContext*)calloc(1, sizeof(*context));
    if (!context) return VX_STATUS_OUT_OF_MEMORY;
    *out = context;
    return VX_STATUS_OK;
}

static VxStatus mock_context_execute(void* instance,
                                     const VxTensorBinding* inputs,
                                     size_t input_count,
                                     const VxBackendOutputSink* sink,
                                     VxReport* report) {
    if (!instance || !inputs || input_count != 1u ||
        inputs[0].struct_size != sizeof(inputs[0]) || !inputs[0].name ||
        strcmp(inputs[0].name, "value") || inputs[0].dtype != VX_DTYPE_F32 ||
        inputs[0].rank != 1u || inputs[0].shape[0] != 1 ||
        inputs[0].location != VX_MEMORY_HOST || !inputs[0].data ||
        inputs[0].byte_size != sizeof(float) ||
        !sink || sink->struct_size != sizeof(*sink) || !sink->write)
        return VX_STATUS_INVALID_ARGUMENT;
    memcpy(&((MockContext*)instance)->input, inputs[0].data, sizeof(float));
    float output[2] = {((MockContext*)instance)->input * 2.0f, -1.0f};
    int64_t shape[1] = {1};
    VxStatus status;
    switch (mock_output_violation) {
        case MOCK_OUTPUT_MISSING:
            return VX_STATUS_OK;
        case MOCK_OUTPUT_WRONG_NAME:
            return sink->write(sink->user_data, "other", VX_DTYPE_F32,
                               shape, 1, output, sizeof(output[0]));
        case MOCK_OUTPUT_WRONG_DTYPE:
            return sink->write(sink->user_data, "mock-out", VX_DTYPE_I32,
                               shape, 1, output, sizeof(output[0]));
        case MOCK_OUTPUT_WRONG_RANK:
            return sink->write(sink->user_data, "mock-out", VX_DTYPE_F32,
                               NULL, 0, output, sizeof(output[0]));
        case MOCK_OUTPUT_WRONG_SHAPE:
            shape[0] = 2;
            return sink->write(sink->user_data, "mock-out", VX_DTYPE_F32,
                               shape, 1, output, sizeof(output));
        case MOCK_OUTPUT_WRONG_BYTE_SIZE:
            return sink->write(sink->user_data, "mock-out", VX_DTYPE_F32,
                               shape, 1, output, sizeof(output));
        case MOCK_OUTPUT_DUPLICATE:
            status = sink->write(sink->user_data, "mock-out", VX_DTYPE_F32,
                                 shape, 1, output, sizeof(output[0]));
            return status == VX_STATUS_OK
                ? sink->write(sink->user_data, "mock-out", VX_DTYPE_F32,
                              shape, 1, output, sizeof(output[0]))
                : status;
        case MOCK_OUTPUT_F16:
            return sink->write(sink->user_data, "mock-out", (VxDataType)4,
                               shape, 1, output, 2u);
        case MOCK_OUTPUT_UNATTESTED_ROUTE:
            report->route_attested = 0;
            report->route_evidence[0] = '\0';
            return sink->write(sink->user_data, "mock-out", VX_DTYPE_F32,
                               shape, 1, output, sizeof(output[0]));
        case MOCK_OUTPUT_VALID:
        default:
            return sink->write(sink->user_data, "mock-out", VX_DTYPE_F32,
                               shape, 1, output, sizeof(output[0]));
    }
}

static VxStatus mock_layout_sink_write(void* user_data,
                                       const char* name,
                                       VxDataType dtype,
                                       const int64_t* shape,
                                       uint32_t rank,
                                       const void* data,
                                       size_t byte_size) {
    (void)user_data;
    (void)name;
    (void)dtype;
    (void)shape;
    (void)rank;
    (void)data;
    (void)byte_size;
    return VX_STATUS_OK;
}

static int test_mock_provider_oversized_descriptors(void) {
    const char* backend_names[1] = {"layout-provider"};
    float value = 1.0f;
    MockRuntime runtime = {11, "layout-provider"};
    MockContext context = {0};
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxTensorSpec input_spec = VX_TENSOR_SPEC_INIT;
    VxTensorSpec output_spec = VX_TENSOR_SPEC_INIT;
    VxBackendCompileInput input = VX_BACKEND_COMPILE_INPUT_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxBackendShapeDomainAttestation attestation =
        VX_BACKEND_SHAPE_DOMAIN_ATTESTATION_INIT;
    VxReport report = VX_REPORT_INIT;
    VxTensorBinding binding = {
        sizeof(VxTensorBinding), "value", VX_DTYPE_F32, 1u, {1},
        &value, sizeof(value), VX_MEMORY_HOST,
    };
    VxBackendOutputSink sink = {
        sizeof(VxBackendOutputSink), NULL, mock_layout_sink_write,
    };
    void* compiled = NULL;

    source.graph_path = "unused";
    input_spec.name = "value";
    input_spec.dtype = VX_DTYPE_F32;
    input_spec.rank = 1u;
    input_spec.location = VX_MEMORY_HOST;
    input_spec.dimensions[0] =
        (VxDimensionConstraint)VX_DIMENSION_CONSTRAINT_INIT;
    output_spec = input_spec;
    output_spec.name = "mock-out";
    input.source = &source;
    input.graph_fingerprint = "graph";
    input.shape_domain_proof_identity = "proof";
    input.inputs = &input_spec;
    input.input_count = 1u;
    input.outputs = &output_spec;
    input.output_count = 1u;
    policy.backends = backend_names;
    policy.backend_count = 1u;

    input.struct_size++;
    CHECK(mock_compile(&runtime, &input, &policy, &compiled, &attestation,
                       &report) == VX_STATUS_INVALID_ARGUMENT);
    input.struct_size = sizeof(input);
    attestation.struct_size++;
    CHECK(mock_compile(&runtime, &input, &policy, &compiled, &attestation,
                       &report) == VX_STATUS_INVALID_ARGUMENT);
    attestation.struct_size = sizeof(attestation);
    source.struct_size++;
    CHECK(mock_compile(&runtime, &input, &policy, &compiled, &attestation,
                       &report) == VX_STATUS_INVALID_ARGUMENT);
    source.struct_size = sizeof(source);
    input_spec.struct_size++;
    CHECK(mock_compile(&runtime, &input, &policy, &compiled, &attestation,
                       &report) == VX_STATUS_INVALID_ARGUMENT);
    input_spec.struct_size = sizeof(input_spec);
    input_spec.dimensions[0].struct_size++;
    CHECK(mock_compile(&runtime, &input, &policy, &compiled, &attestation,
                       &report) == VX_STATUS_INVALID_ARGUMENT);
    input_spec.dimensions[0].struct_size =
        sizeof(input_spec.dimensions[0]);
    sink.struct_size++;
    CHECK(mock_context_execute(&context, &binding, 1u, &sink, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    sink.struct_size = sizeof(sink);
    binding.struct_size++;
    CHECK(mock_context_execute(&context, &binding, 1u, &sink, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    return 0;
}

static VxStatus mock_context_select_adapter(void* instance,
                                            uint64_t adapter_id,
                                            uint64_t adapter_revision,
                                            const char* package_path,
                                            const char* version_name,
                                            VxReport* report) {
    (void)report;
    if (!instance || !adapter_id || !adapter_revision)
        return VX_STATUS_INVALID_ARGUMENT;
    mock_adapter_selected++;
    mock_adapter_id = adapter_id;
    mock_adapter_revision = adapter_revision;
    mock_adapter_package_was_null = package_path == NULL;
    mock_adapter_package[0] = '\0';
    if (package_path && read_text(package_path, mock_adapter_package,
                                  sizeof(mock_adapter_package)) != 0)
        return VX_STATUS_IO_ERROR;
    snprintf(mock_adapter_version, sizeof(mock_adapter_version), "%s",
             version_name ? version_name : "");
    return VX_STATUS_OK;
}

static VxStatus mock_context_close(void* instance, VxReport* report) {
    (void)report;
    if (!instance) return VX_STATUS_INVALID_ARGUMENT;
    mock_context_closed++;
    return VX_STATUS_OK;
}

static void mock_context_destroy(void* instance) {
    mock_context_destroyed++;
    free(instance);
}

static int expect_bank_package_rejected(VxRuntime* runtime,
                                        const char* graph_path,
                                        const char* const* weight_paths,
                                        size_t weight_path_count) {
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    source.graph_path = graph_path;
    source.weight_paths = weight_paths;
    source.weight_path_count = weight_path_count;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) ==
          VX_STATUS_INVALID_GRAPH);
    CHECK(model == NULL && !strcmp(report.reason, "INVALID_GRAPH_CONTRACT"));
    return 0;
}

static int test_graph_bank_package_validation(const char* graph_path) {
    const char* valid_path =
        "/tmp/volvox-public-api-bank-valid.safetensors";
    const char* missing_path =
        "/tmp/volvox-public-api-bank-missing.safetensors";
    const char* rank_one_path =
        "/tmp/volvox-public-api-bank-rank-one.safetensors";
    const char* zero_payload_path =
        "/tmp/volvox-public-api-bank-zero-payload.safetensors";
    const char* below_minimum_path =
        "/tmp/volvox-public-api-bank-below-minimum.safetensors";
    const char* wrong_multiple_path =
        "/tmp/volvox-public-api-bank-wrong-multiple.safetensors";
    const char* above_maximum_path =
        "/tmp/volvox-public-api-bank-above-maximum.safetensors";
    const char* duplicate_path =
        "/tmp/volvox-public-api-bank-duplicate.safetensors";
    const char* valid_paths[1] = {valid_path};
    const char* missing_paths[1] = {missing_path};
    const char* rank_one_paths[1] = {rank_one_path};
    const char* zero_payload_paths[1] = {zero_payload_path};
    const char* below_minimum_paths[1] = {below_minimum_path};
    const char* wrong_multiple_paths[1] = {wrong_multiple_path};
    const char* above_maximum_paths[1] = {above_maximum_path};
    const char* duplicate_paths[2] = {valid_path, duplicate_path};
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;

    CHECK(write_bank_weights(valid_path, "experts", 8, 2) == 0);
    CHECK(write_bank_weights(missing_path, "other", 4, 2) == 0);
    CHECK(write_bank_weights(rank_one_path, "experts", 4, 1) == 0);
    CHECK(write_zero_payload_bank_weights(zero_payload_path, "experts") == 0);
    CHECK(write_bank_weights(below_minimum_path, "experts", 3, 2) == 0);
    CHECK(write_bank_weights(wrong_multiple_path, "experts", 6, 2) == 0);
    CHECK(write_bank_weights(above_maximum_path, "experts", 12, 2) == 0);
    CHECK(write_bank_weights(duplicate_path, "experts", 4, 2) == 0);
    CHECK(vx_runtime_create(&runtime_options, &runtime, &report) == VX_STATUS_OK);

    source.graph_path = graph_path;
    source.weight_paths = valid_paths;
    source.weight_path_count = 1u;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) ==
          VX_STATUS_OK);
    vx_model_release(model);
    model = NULL;

    CHECK(expect_bank_package_rejected(
              runtime, graph_path, missing_paths, 1u) == 0);
    CHECK(expect_bank_package_rejected(
              runtime, graph_path, rank_one_paths, 1u) == 0);
    CHECK(expect_bank_package_rejected(
              runtime, graph_path, zero_payload_paths, 1u) == 0);
    CHECK(expect_bank_package_rejected(
              runtime, graph_path, below_minimum_paths, 1u) == 0);
    CHECK(expect_bank_package_rejected(
              runtime, graph_path, wrong_multiple_paths, 1u) == 0);
    CHECK(expect_bank_package_rejected(
              runtime, graph_path, above_maximum_paths, 1u) == 0);
    CHECK(expect_bank_package_rejected(
              runtime, graph_path, duplicate_paths, 2u) == 0);

    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    vx_runtime_release(runtime);
    CHECK(remove(valid_path) == 0);
    CHECK(remove(missing_path) == 0);
    CHECK(remove(rank_one_path) == 0);
    CHECK(remove(zero_payload_path) == 0);
    CHECK(remove(below_minimum_path) == 0);
    CHECK(remove(wrong_multiple_path) == 0);
    CHECK(remove(above_maximum_path) == 0);
    CHECK(remove(duplicate_path) == 0);
    return 0;
}

static int test_many_bank_residencies(void) {
    enum { BANK_COUNT = 17 };
    const char* graph_path =
        "/tmp/volvox-public-api-many-banks.graph.json";
    const char* weights_path =
        "/tmp/volvox-public-api-many-banks.safetensors";
    const char* weight_paths[1] = {weights_path};
    char bank_names[BANK_COUNT][32];
    uint32_t selected_slots[BANK_COUNT] = {0};
    VxBankResidency residencies[BANK_COUNT];
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;

    CHECK(write_many_bank_package(graph_path, weights_path, BANK_COUNT) == 0);
    for (size_t index = 0; index < BANK_COUNT; index++) {
        int length = snprintf(bank_names[index], sizeof(bank_names[index]),
                              "bank_%zu", index);
        CHECK(length > 0 && (size_t)length < sizeof(bank_names[index]));
        residencies[index] = (VxBankResidency)VX_BANK_RESIDENCY_INIT;
        residencies[index].bank = bank_names[index];
        residencies[index].slots = &selected_slots[index];
        residencies[index].slot_count = 1u;
    }
    source.graph_path = graph_path;
    source.weight_paths = weight_paths;
    source.weight_path_count = 1u;
    source.bank_residency = residencies;
    source.bank_residency_count = BANK_COUNT;
    runtime_options.cpu_threads = 1;
    CHECK(vx_runtime_create(&runtime_options, &runtime, &report) == VX_STATUS_OK);
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK);
    CHECK(vx_model_compile(model, &policy, &compiled, &report) == VX_STATUS_OK);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    vx_runtime_release(runtime);
    CHECK(remove(graph_path) == 0);
    CHECK(remove(weights_path) == 0);
    return 0;
}

static int test_provider(const char* readable_graph,
                         const char* readable_weights) {
    CHECK(VX_NATIVE_API_VERSION == 1u);
    CHECK(VX_BACKEND_ABI_VERSION == 1u);
    CHECK(test_mock_provider_oversized_descriptors() == 0);
    VxBackendProvider provider = {
        .struct_size = sizeof(VxBackendProvider),
        .abi_version = VX_BACKEND_ABI_VERSION,
        .name = "test-provider",
        .user_data = (void*)"test-provider",
        .shape_domain = {
            sizeof(VxBackendShapeDomainCapability),
            VX_BACKEND_SHAPE_PROOF_PROTOCOL,
            VX_BACKEND_RESOURCE_PROTOCOL,
            VX_BACKEND_SHAPE_DOMAIN_FULL,
        },
        .runtime_create = mock_runtime_create,
        .runtime_destroy = mock_runtime_destroy,
        .compile = mock_compile,
        .compiled_destroy = mock_compiled_destroy,
        .context_create = mock_context_create,
        .context_execute = mock_context_execute,
        .context_select_adapter = mock_context_select_adapter,
        .context_close = mock_context_close,
        .context_destroy = mock_context_destroy,
    };
    VxBackendProvider invalid = provider;
    VxBackendProvider unattested = provider;
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxResult* result = NULL;
    VxAdapterSource adapter_source = VX_ADAPTER_SOURCE_INIT;
    VxAdapterRevision first_adapter_revision = VX_ADAPTER_REVISION_INIT;
    VxAdapterRevision second_adapter_revision = VX_ADAPTER_REVISION_INIT;
    float input = 7.5f;
    float output = 0.0f;
    VxTensorBinding input_binding = {
        sizeof(VxTensorBinding), "value", VX_DTYPE_F32, 1u, {1},
        &input, sizeof(input), VX_MEMORY_HOST,
    };
    const char* required_backend[1];
    const char* weight_paths[1] = {readable_weights};
    char bank_name[16] = "experts";
    uint32_t bank_slots[2] = {1u, 3u};
    VxBankResidency bank_residency = {
        sizeof(VxBankResidency), bank_name, bank_slots, 2u,
    };
    const char* adapter_path = "/tmp/volvox-public-api-provider-adapter.bin";

    {
        VxRuntimeOptions invalid_options = runtime_options;
        VxReport invalid_report = VX_REPORT_INIT;
        VxRuntime* rejected = NULL;
        invalid_options.struct_size++;
        CHECK(vx_runtime_create(&invalid_options, &rejected, &report) ==
              VX_STATUS_INVALID_ARGUMENT);
        CHECK(rejected == NULL);
        invalid_report.struct_size++;
        CHECK(vx_runtime_create(&runtime_options, &rejected,
                                &invalid_report) ==
              VX_STATUS_INVALID_ARGUMENT);
        CHECK(rejected == NULL);
    }
    CHECK(vx_runtime_create(&runtime_options, &runtime, &report) == VX_STATUS_OK);
    invalid.struct_size--;
    CHECK(vx_runtime_register_provider(runtime, &invalid, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    invalid = provider;
    invalid.struct_size++;
    CHECK(vx_runtime_register_provider(runtime, &invalid, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    invalid = provider;
    invalid.shape_domain.struct_size++;
    CHECK(vx_runtime_register_provider(runtime, &invalid, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    invalid = provider;
    invalid.abi_version = UINT32_MAX;
    CHECK(vx_runtime_register_provider(runtime, &invalid, &report) ==
          VX_STATUS_ABI_UNSUPPORTED);
    CHECK(!strcmp(report.reason, "ABI_UNSUPPORTED"));
    invalid = provider;
    invalid.name = "Invalid_Name";
    CHECK(vx_runtime_register_provider(runtime, &invalid, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    invalid = provider;
    invalid.name = "vulkan";
    CHECK(vx_runtime_register_provider(runtime, &invalid, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(vx_runtime_register_provider(runtime, &provider, &report) == VX_STATUS_OK);
    CHECK(vx_runtime_register_provider(runtime, &provider, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    unattested.name = "unattested-provider";
    unattested.user_data = (void*)"unattested-provider";
    CHECK(vx_runtime_register_provider(runtime, &unattested, &report) ==
          VX_STATUS_OK);
    source.graph_path = readable_graph;
    source.weight_paths = weight_paths;
    source.weight_path_count = 1u;
    source.bank_residency = &bank_residency;
    source.bank_residency_count = 1u;
    {
        VxModelSource invalid_source = source;
        VxModel* rejected = NULL;
        invalid_source.bank_residency_count =
            SIZE_MAX / sizeof(VxBankResidency) + 1u;
        CHECK(vx_runtime_load_model(runtime, &invalid_source, &rejected,
                                    &report) == VX_STATUS_INVALID_ARGUMENT);
        CHECK(rejected == NULL && !strcmp(report.reason,
                                          "INVALID_MODEL_SOURCE"));
    }
    {
        uint32_t one_slot = 0u;
        VxBankResidency invalid_residency = bank_residency;
        VxModelSource invalid_source = source;
        VxModel* rejected = NULL;
        invalid_residency.slots = &one_slot;
        invalid_residency.slot_count = SIZE_MAX / sizeof(uint32_t) + 1u;
        invalid_source.bank_residency = &invalid_residency;
        CHECK(vx_runtime_load_model(runtime, &invalid_source, &rejected,
                                    &report) == VX_STATUS_INVALID_ARGUMENT);
        CHECK(rejected == NULL && !strcmp(report.reason,
                                          "INVALID_BANK_RESIDENCY"));
    }
    {
        VxBankResidency invalid_residency = bank_residency;
        VxModelSource invalid_source = source;
        VxModel* rejected = NULL;
        invalid_residency.bank = "undeclared";
        invalid_source.bank_residency = &invalid_residency;
        CHECK(vx_runtime_load_model(runtime, &invalid_source, &rejected,
                                    &report) == VX_STATUS_INVALID_ARGUMENT);
        CHECK(rejected == NULL && !strcmp(report.reason,
                                          "INVALID_BANK_RESIDENCY"));
    }
    {
        uint32_t out_of_domain_slots[2] = {1u, 4u};
        VxBankResidency invalid_residency = bank_residency;
        VxModelSource invalid_source = source;
        VxModel* rejected = NULL;
        invalid_residency.slots = out_of_domain_slots;
        invalid_source.bank_residency = &invalid_residency;
        CHECK(vx_runtime_load_model(runtime, &invalid_source, &rejected,
                                    &report) == VX_STATUS_INVALID_ARGUMENT);
        CHECK(rejected == NULL && !strcmp(report.reason,
                                          "INVALID_BANK_RESIDENCY"));
    }
    mock_caller_bank_residency = &bank_residency;
    mock_caller_bank_name = bank_name;
    mock_caller_bank_slots = bank_slots;
    mock_bank_source_seen = 0;
    source.struct_size++;
    {
        VxModel* rejected = NULL;
        CHECK(vx_runtime_load_model(runtime, &source, &rejected, &report) ==
              VX_STATUS_INVALID_ARGUMENT);
        CHECK(rejected == NULL && !strcmp(report.reason,
                                          "INVALID_MODEL_SOURCE"));
    }
    source.struct_size = sizeof(source);
    bank_residency.struct_size++;
    {
        VxModel* rejected = NULL;
        CHECK(vx_runtime_load_model(runtime, &source, &rejected, &report) ==
              VX_STATUS_INVALID_ARGUMENT);
        CHECK(rejected == NULL && !strcmp(report.reason,
                                          "INVALID_BANK_RESIDENCY"));
    }
    bank_residency.struct_size = sizeof(bank_residency);
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK);
    snprintf(bank_name, sizeof(bank_name), "%s", "poison");
    bank_slots[0] = 0u;
    bank_slots[1] = 2u;
    bank_residency.bank = NULL;
    bank_residency.slots = NULL;
    bank_residency.slot_count = 0u;
    source.bank_residency = NULL;
    source.bank_residency_count = 0u;
    policy.mode = VX_BACKEND_REQUIRE;
    policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    required_backend[0] = "unattested-provider";
    policy.backends = required_backend;
    policy.backend_count = 1;
    CHECK(vx_model_compile(model, &policy, &compiled, &report) ==
          VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN);
    CHECK(compiled == NULL &&
          !strcmp(report.reason, "OPERATOR_FALLBACK_FORBIDDEN"));
    policy.operator_fallback = VX_OPERATOR_FALLBACK_ALLOW;
    required_backend[0] = "test-provider";
    policy.struct_size++;
    CHECK(vx_model_compile(model, &policy, &compiled, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(compiled == NULL);
    policy.struct_size = sizeof(policy);
    CHECK(vx_model_compile(model, &policy, &compiled, &report) == VX_STATUS_OK);
    CHECK(mock_bank_source_seen == 1);
    CHECK(!strcmp(report.backend, "test-provider"));
    CHECK(!strcmp(report.device, "mock-device-0") && report.route_attested &&
          !report.operator_fallback_used && report.route_evidence[0] &&
          report.operator_fallback == VX_OPERATOR_FALLBACK_ALLOW);
    context_options.struct_size++;
    CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                            &context, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(context == NULL);
    context_options.struct_size = sizeof(context_options);
    CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                            &context, &report) == VX_STATUS_OK);
    {
        VxAffineQuantization quantization = VX_AFFINE_QUANTIZATION_INIT;
        VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
        spec.struct_size++;
        CHECK(vx_execution_context_input_spec(context, 0u, &spec, &report) ==
              VX_STATUS_INVALID_ARGUMENT);
        quantization.struct_size++;
        CHECK(vx_execution_context_input_affine_quantization(
                  context, "value", &quantization, &report) ==
              VX_STATUS_INVALID_ARGUMENT);
        quantization.struct_size = sizeof(quantization);
        quantization.defined = 1;
        quantization.scale = 3.0f;
        quantization.zero_point = 17;
        CHECK(vx_execution_context_input_affine_quantization(
                  context, "value", &quantization, &report) ==
              VX_STATUS_BACKEND_UNSUPPORTED);
        CHECK(!quantization.defined && quantization.scale == 0.0f &&
              quantization.zero_point == 0 &&
              !strcmp(report.reason,
                      "INPUT_QUANTIZATION_INTROSPECTION_UNSUPPORTED"));
    }
    CHECK(write_text(adapter_path, "adapter-one") == 0);
    adapter_source.adapter_name = "provider-route";
    adapter_source.version_name = "adapter-one";
    adapter_source.package_path = adapter_path;
    CHECK(vx_model_publish_adapter(model, &adapter_source,
                                   &first_adapter_revision,
                                   &report) == VX_STATUS_OK);
    CHECK(write_text(adapter_path, "changed-one") == 0);
    CHECK(remove(adapter_path) == 0);
    CHECK(vx_execution_context_select_adapter(context, &first_adapter_revision,
                                               &report) == VX_STATUS_OK);
    CHECK(mock_adapter_selected == 1 &&
          mock_adapter_id == first_adapter_revision.adapter_id &&
          mock_adapter_revision == first_adapter_revision.adapter_revision &&
          !mock_adapter_package_was_null &&
          !strcmp(mock_adapter_package, "adapter-one") &&
          !strcmp(mock_adapter_version, "adapter-one"));
    CHECK(write_text(adapter_path, "adapter-two") == 0);
    adapter_source.version_name = "adapter-two";
    CHECK(vx_model_publish_adapter(model, &adapter_source,
                                   &second_adapter_revision,
                                   &report) == VX_STATUS_OK);
    CHECK(second_adapter_revision.adapter_id == first_adapter_revision.adapter_id &&
          second_adapter_revision.adapter_revision ==
              first_adapter_revision.adapter_revision + 1u);
    CHECK(write_text(adapter_path, "changed-two") == 0);
    CHECK(remove(adapter_path) == 0);
    CHECK(vx_execution_context_rebind_adapter(context, &report) == VX_STATUS_OK);
    CHECK(mock_adapter_selected == 2 &&
          !strcmp(mock_adapter_package, "adapter-two") &&
          !strcmp(mock_adapter_version, "adapter-two"));
    CHECK(vx_execution_context_select_adapter(context, &first_adapter_revision,
                                               &report) == VX_STATUS_OK);
    CHECK(mock_adapter_selected == 3 &&
          !strcmp(mock_adapter_package, "adapter-one") &&
          !strcmp(mock_adapter_version, "adapter-one"));
    {
        VxTensorBinding invalid_binding = input_binding;
        invalid_binding.struct_size++;
        CHECK(vx_execution_context_execute(
                  context, &invalid_binding, 1u, &result, &report) ==
              VX_STATUS_INVALID_ARGUMENT);
        CHECK(result == NULL);
        invalid_binding = input_binding;
        invalid_binding.dtype = (VxDataType)4;
        invalid_binding.byte_size = 2u;
        CHECK(vx_execution_context_execute(
                  context, &invalid_binding, 1u, &result, &report) ==
              VX_STATUS_INVALID_ARGUMENT);
        CHECK(result == NULL);
    }
    CHECK(vx_execution_context_execute(
              context, &input_binding, 1u, &result, &report) == VX_STATUS_OK);
    CHECK(report.context_id && report.execution_id && report.route_attested &&
          !report.operator_fallback_used && report.execution_time_ms >= 0.0);
    CHECK(vx_result_output_count(result) == 1);
    {
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        info.struct_size++;
        CHECK(vx_result_output_info(result, 0u, &info, &report) ==
              VX_STATUS_INVALID_ARGUMENT);
    }
    {
        VxResult* unsupported = NULL;
        CHECK(vx_execution_context_execute_prefix(
                  context, 1, &input_binding, 1u,
                  &unsupported, &report) ==
              VX_STATUS_BACKEND_UNSUPPORTED);
        CHECK(unsupported == NULL &&
              !strcmp(report.reason, "BACKEND_UNSUPPORTED"));
    }
    {
        const MockOutputViolation violations[] = {
            MOCK_OUTPUT_MISSING,
            MOCK_OUTPUT_WRONG_NAME,
            MOCK_OUTPUT_WRONG_DTYPE,
            MOCK_OUTPUT_WRONG_RANK,
            MOCK_OUTPUT_WRONG_SHAPE,
            MOCK_OUTPUT_WRONG_BYTE_SIZE,
            MOCK_OUTPUT_DUPLICATE,
            MOCK_OUTPUT_F16,
            MOCK_OUTPUT_UNATTESTED_ROUTE,
        };
        for (size_t index = 0;
             index < sizeof(violations) / sizeof(violations[0]); index++) {
            VxResult* rejected = NULL;
            mock_output_violation = violations[index];
            CHECK(vx_execution_context_execute(
                      context, &input_binding, 1u, &rejected, &report) ==
                  VX_STATUS_EXECUTION_FAILED);
            CHECK(rejected == NULL && report.status == VX_STATUS_EXECUTION_FAILED &&
                  !strcmp(report.reason, "EXECUTION_FAILED"));
        }
        mock_output_violation = MOCK_OUTPUT_VALID;
        {
            VxResult* recovered = NULL;
            CHECK(vx_execution_context_execute(
                      context, &input_binding, 1u, &recovered, &report) ==
                  VX_STATUS_OK);
            vx_result_release(recovered);
        }
    }
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    CHECK(vx_runtime_register_provider(runtime, &provider, &report) ==
          VX_STATUS_HANDLE_DISPOSED);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    vx_runtime_release(runtime);
    CHECK(vx_result_read(result, "mock-out", &output, sizeof(output), NULL, &report) ==
          VX_STATUS_OK);
    CHECK(closef(output, 15.0f));
    vx_result_release(result);
    CHECK(mock_context_closed == 1);
    CHECK(mock_context_destroyed == 1);
    CHECK(mock_compiled_destroyed == 2);
    CHECK(mock_runtime_destroyed == 2);
    mock_caller_bank_residency = NULL;
    mock_caller_bank_name = NULL;
    mock_caller_bank_slots = NULL;
    remove(adapter_path);
    return 0;
}

static int test_model_source_snapshots(void) {
    const char* graph_path = "/tmp/volvox-public-api-snapshot.graph.json";
    const char* weight_path =
        "/tmp/volvox-public-api-snapshot-weights.safetensors";
    const char* original_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},"
        "\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"add\",\"opType\":\"Add\","
        "\"inputs\":{\"a\":\"x\",\"b\":\"w\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[1]}},\"params\":{}}],"
        "\"outputs\":[\"y\"]}";
    const char* replacement_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},"
        "\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"identity\",\"opType\":\"Identity\","
        "\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"different\","
        "\"dtype\":\"float32\",\"shape\":[1]}},\"params\":{}}],"
        "\"outputs\":[\"different\"]}";
    const char* weights[] = {weight_path};
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxResult* result = NULL;
    float input = 3.0f;
    float output = 0.0f;
    const VxTensorBinding binding = {
        sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 1u, {1},
        &input, sizeof(input), VX_MEMORY_HOST,
    };

    CHECK(write_text(graph_path, original_graph) == 0);
    CHECK(write_weight(weight_path, 2.0f) == 0);
    source.graph_path = graph_path;
    source.weight_paths = weights;
    source.weight_path_count = 1;
    CHECK(vx_runtime_create(&runtime_options, &runtime, &report) == VX_STATUS_OK);
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK);

    /* Compilation must consume the accepted graph and weight bytes, not these
     * caller-owned paths after load_model returns. */
    CHECK(write_text(graph_path, replacement_graph) == 0);
    CHECK(write_weight(weight_path, 99.0f) == 0);
    CHECK(remove(graph_path) == 0);
    CHECK(remove(weight_path) == 0);
    CHECK(vx_model_compile(model, &policy, &compiled, &report) == VX_STATUS_OK);
    CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                            &context, &report) == VX_STATUS_OK);
    CHECK(vx_execution_context_execute(
              context, &binding, 1u, &result, &report) == VX_STATUS_OK);
    CHECK(vx_result_read(result, "y", &output, sizeof(output), NULL, &report) ==
          VX_STATUS_OK);
    CHECK(closef(output, 5.0f));

    vx_result_release(result);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    vx_runtime_release(runtime);
    remove(graph_path);
    remove(weight_path);
    return 0;
}

static int test_declared_output_descriptors(void) {
    const char* direct_graph_path =
        "/tmp/volvox-public-api-direct-output.graph.json";
    const char* weight_graph_path =
        "/tmp/volvox-public-api-weight-output.graph.json";
    const char* f16_graph_path =
        "/tmp/volvox-public-api-f16-storage.graph.json";
    const char* weight_path =
        "/tmp/volvox-public-api-output-weight.safetensors";
    const char* f16_weight_path =
        "/tmp/volvox-public-api-f16-weight.safetensors";
    const char* direct_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},"
        "\"inputs\":{\"direct\":{\"shape\":[2],\"dtype\":\"int32\"}},"
        "\"nodes\":[],\"outputs\":[\"direct\"]}";
    const char* weight_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},"
        "\"inputs\":{},"
        "\"nodes\":[],\"outputs\":[\"w\"]}";
    const char* f16_compute_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},"
        "\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"add\",\"opType\":\"Add\","
        "\"inputs\":{\"a\":\"x\",\"b\":\"w\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[1]}},\"params\":{}}],"
        "\"outputs\":[\"y\"]}";
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    uint16_t f16_two = UINT16_C(0x4000);
    float weight_value = 6.25f;

    CHECK(write_text(direct_graph_path, direct_graph) == 0);
    CHECK(write_text(weight_graph_path, weight_graph) == 0);
    CHECK(write_text(f16_graph_path, f16_compute_graph) == 0);
    CHECK(write_weight(weight_path, weight_value) == 0);
    CHECK(write_typed_weight(f16_weight_path, SAFETENSORS_DTYPE_F16,
                             &f16_two, sizeof(f16_two)) == 0);
    CHECK(vx_runtime_create(&runtime_options, &runtime, &report) == VX_STATUS_OK);

    {
        VxModelSource source = VX_MODEL_SOURCE_INIT;
        VxModel* model = NULL;
        VxCompiledModel* compiled = NULL;
        VxExecutionContext* context = NULL;
        VxResult* result = NULL;
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        const int32_t input[2] = {17, -9};
        const VxTensorBinding binding = {
            sizeof(VxTensorBinding), "direct", VX_DTYPE_I32, 1u, {2},
            input, sizeof(input), VX_MEMORY_HOST,
        };
        int32_t output[2] = {0};
        source.graph_path = direct_graph_path;
        CHECK(vx_runtime_load_model(runtime, &source, &model, &report) ==
              VX_STATUS_OK);
        CHECK(vx_model_compile(model, &policy, &compiled, &report) ==
              VX_STATUS_OK);
        CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                                &context, &report) ==
              VX_STATUS_OK);
        CHECK(vx_execution_context_execute(
                  context, &binding, 1u, &result, &report) ==
              VX_STATUS_OK);
        CHECK(vx_result_output_info(result, 0, &info, &report) == VX_STATUS_OK);
        CHECK(!strcmp(info.name, "direct") && info.dtype == VX_DTYPE_I32 &&
              info.rank == 1 && info.shape[0] == 2 &&
              info.byte_size == sizeof(input));
        CHECK(vx_result_read(result, "direct", output, sizeof(output), NULL,
                             &report) == VX_STATUS_OK);
        CHECK(!memcmp(input, output, sizeof(input)));
        vx_result_release(result);
        CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
        vx_execution_context_release(context);
        vx_compiled_model_release(compiled);
        vx_model_release(model);
    }

    {
        const char* weights[] = {weight_path};
        VxModelSource source = VX_MODEL_SOURCE_INIT;
        VxModel* model = NULL;
        VxCompiledModel* compiled = NULL;
        VxExecutionContext* context = NULL;
        VxResult* result = NULL;
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        float output = 0.0f;
        int32_t replacement = 123;
        source.graph_path = weight_graph_path;
        source.weight_paths = weights;
        source.weight_path_count = 1;
        CHECK(vx_runtime_load_model(runtime, &source, &model, &report) ==
              VX_STATUS_OK);
        CHECK(write_typed_weight(weight_path, SAFETENSORS_DTYPE_I32,
                                 &replacement, sizeof(replacement)) == 0);
        CHECK(remove(weight_path) == 0);
        CHECK(vx_model_compile(model, &policy, &compiled, &report) ==
              VX_STATUS_OK);
        CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                                &context, &report) ==
              VX_STATUS_OK);
        CHECK(vx_execution_context_execute(context, NULL, 0u,
                                            &result, &report) ==
              VX_STATUS_OK);
        CHECK(vx_result_output_info(result, 0, &info, &report) == VX_STATUS_OK);
        CHECK(!strcmp(info.name, "w") && info.dtype == VX_DTYPE_F32 &&
              info.rank == 1 && info.shape[0] == 1 &&
              info.byte_size == sizeof(output));
        CHECK(vx_result_read(result, "w", &output, sizeof(output), NULL,
                             &report) == VX_STATUS_OK);
        CHECK(closef(output, weight_value));
        vx_result_release(result);
        CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
        vx_execution_context_release(context);
        vx_compiled_model_release(compiled);
        vx_model_release(model);
    }

    {
        const char* weights[] = {f16_weight_path};
        VxModelSource source = VX_MODEL_SOURCE_INIT;
        VxModel* model = NULL;
        VxCompiledModel* compiled = NULL;
        VxExecutionContext* context = NULL;
        VxResult* result = NULL;
        float input = 3.0f;
        float output = 0.0f;
        const VxTensorBinding binding = {
            sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 1u, {1},
            &input, sizeof(input), VX_MEMORY_HOST,
        };
        source.graph_path = f16_graph_path;
        source.weight_paths = weights;
        source.weight_path_count = 1;
        CHECK(vx_runtime_load_model(runtime, &source, &model, &report) ==
              VX_STATUS_OK);
        CHECK(vx_model_compile(model, &policy, &compiled, &report) ==
              VX_STATUS_OK);
        CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                                &context, &report) ==
              VX_STATUS_OK);
        CHECK(vx_execution_context_execute(
                  context, &binding, 1u, &result, &report) ==
              VX_STATUS_OK);
        CHECK(vx_result_read(result, "y", &output, sizeof(output), NULL,
                             &report) == VX_STATUS_OK);
        CHECK(closef(output, 5.0f));
        vx_result_release(result);
        CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
        vx_execution_context_release(context);
        vx_compiled_model_release(compiled);
        vx_model_release(model);

        source.graph_path = weight_graph_path;
        model = NULL;
        compiled = NULL;
        context = NULL;
        result = NULL;
        output = 0.0f;
        CHECK(vx_runtime_load_model(runtime, &source, &model, &report) ==
              VX_STATUS_OK);
        CHECK(vx_model_compile(model, &policy, &compiled, &report) ==
              VX_STATUS_OK);
        CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                                &context, &report) ==
              VX_STATUS_OK);
        CHECK(vx_execution_context_execute(context, NULL, 0u,
                                            &result, &report) ==
              VX_STATUS_OK);
        {
            VxTensorInfo info = VX_TENSOR_INFO_INIT;
            CHECK(vx_result_output_info(result, 0, &info, &report) ==
                  VX_STATUS_OK);
            CHECK(info.dtype == VX_DTYPE_F32 && info.rank == 1 &&
                  info.shape[0] == 1 && info.byte_size == sizeof(float));
        }
        CHECK(vx_result_read(result, "w", &output, sizeof(output), NULL,
                             &report) == VX_STATUS_OK);
        CHECK(closef(output, 2.0f));
        vx_result_release(result);
        CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
        vx_execution_context_release(context);
        vx_compiled_model_release(compiled);
        vx_model_release(model);
    }

    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    vx_runtime_release(runtime);
    remove(direct_graph_path);
    remove(weight_graph_path);
    remove(f16_graph_path);
    remove(weight_path);
    remove(f16_weight_path);
    return 0;
}

static int compile_affine_concat_case(
        VxRuntime* runtime,
        const char* graph_path,
        const char* dimensions,
        const char* inputs,
        const char* nodes,
        const char* outputs,
        VxStatus expected_status,
        const char* expected_relation_reason) {
    char graph[16384];
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    int length = snprintf(
        graph, sizeof(graph),
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":%s,\"inputs\":%s,\"nodes\":%s,"
        "\"outputs\":%s}",
        dimensions, inputs, nodes, outputs);
    CHECK(length > 0 && (size_t)length < sizeof(graph));
    CHECK(write_text(graph_path, graph) == 0);
    source.graph_path = graph_path;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) ==
          VX_STATUS_OK);
    CHECK(model != NULL);
    CHECK(vx_model_compile(model, &policy, &compiled, &report) ==
          expected_status);
    if (expected_status == VX_STATUS_OK) {
        CHECK(compiled != NULL && report.status == VX_STATUS_OK);
        CHECK(report.route_attested &&
              strstr(report.route_evidence, "affine_relations=1"));
    } else {
        CHECK(compiled == NULL &&
              report.status == VX_STATUS_BACKEND_UNSUPPORTED &&
              !strcmp(report.reason, "BOUNDED_DOMAIN_UNSUPPORTED") &&
              strstr(report.route_evidence,
                     "required=canonical-affine-concat-domain") &&
              expected_relation_reason &&
              strstr(report.route_evidence, expected_relation_reason));
    }
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int execute_repeated_affine_concat_case(VxRuntime* runtime,
                                               const char* graph_path) {
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxResult* first = NULL;
    VxResult* second = NULL;
    VxTensorInfo info = VX_TENSOR_INFO_INIT;
    float image[210];
    int32_t image_mask[210];
    const float question_one[1] = {1000.0f};
    const int32_t question_mask_one[1] = {1000};
    const float question_three[3] = {1000.0f, 1001.0f, 1002.0f};
    const int32_t question_mask_three[3] = {1000, 1001, 1002};
    float memory[213] = {0};
    int32_t mask[213] = {0};
    VxTensorBinding first_inputs[4] = {
        {sizeof(VxTensorBinding), "image", VX_DTYPE_F32, 3u, {1, 210, 1},
         image, sizeof(image), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "question", VX_DTYPE_F32, 3u, {1, 1, 1},
         question_one, sizeof(question_one), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "image_mask", VX_DTYPE_I32, 2u, {1, 210},
         image_mask, sizeof(image_mask), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "question_mask", VX_DTYPE_I32, 2u, {1, 1},
         question_mask_one, sizeof(question_mask_one), VX_MEMORY_HOST},
    };
    VxTensorBinding second_inputs[4] = {
        {sizeof(VxTensorBinding), "image", VX_DTYPE_F32, 3u, {1, 210, 1},
         image, sizeof(image), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "question", VX_DTYPE_F32, 3u, {1, 3, 1},
         question_three, sizeof(question_three), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "image_mask", VX_DTYPE_I32, 2u, {1, 210},
         image_mask, sizeof(image_mask), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "question_mask", VX_DTYPE_I32, 2u, {1, 3},
         question_mask_three, sizeof(question_mask_three), VX_MEMORY_HOST},
    };
    for (size_t index = 0; index < 210u; index++) {
        image[index] = (float)index + 0.25f;
        image_mask[index] = (int32_t)index + 7;
    }
    source.graph_path = graph_path;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) ==
          VX_STATUS_OK);
    CHECK(vx_model_compile(model, &policy, &compiled, &report) ==
          VX_STATUS_OK);
    CHECK(vx_compiled_model_create_context(
              compiled, &options, &context, &report) == VX_STATUS_OK);
    CHECK(vx_execution_context_execute(
              context, first_inputs, 4u, &first, &report) == VX_STATUS_OK);
    CHECK(vx_result_output_count(first) == 2u);
    CHECK(vx_result_output_info(first, 0u, &info, &report) == VX_STATUS_OK);
    CHECK(!strcmp(info.name, "memory") && info.dtype == VX_DTYPE_F32 &&
          info.rank == 3u && info.shape[0] == 1 && info.shape[1] == 211 &&
          info.shape[2] == 1 && info.byte_size == 211u * sizeof(float));
    info = (VxTensorInfo)VX_TENSOR_INFO_INIT;
    CHECK(vx_result_output_info(first, 1u, &info, &report) == VX_STATUS_OK);
    CHECK(!strcmp(info.name, "mask") && info.dtype == VX_DTYPE_I32 &&
          info.rank == 2u && info.shape[0] == 1 && info.shape[1] == 211 &&
          info.byte_size == 211u * sizeof(int32_t));
    CHECK(vx_result_read(first, "memory", memory, sizeof(memory), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(!memcmp(memory, image, sizeof(image)) &&
          memory[210] == question_one[0]);
    CHECK(vx_result_read(first, "mask", mask, sizeof(mask), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(!memcmp(mask, image_mask, sizeof(image_mask)) &&
          mask[210] == question_mask_one[0]);

    memset(memory, 0, sizeof(memory));
    memset(mask, 0, sizeof(mask));
    CHECK(vx_execution_context_execute(
              context, second_inputs, 4u, &second, &report) == VX_STATUS_OK);
    CHECK(vx_result_output_count(second) == 2u);
    info = (VxTensorInfo)VX_TENSOR_INFO_INIT;
    CHECK(vx_result_output_info(second, 0u, &info, &report) == VX_STATUS_OK);
    CHECK(!strcmp(info.name, "memory") && info.dtype == VX_DTYPE_F32 &&
          info.rank == 3u && info.shape[0] == 1 && info.shape[1] == 213 &&
          info.shape[2] == 1 && info.byte_size == 213u * sizeof(float));
    info = (VxTensorInfo)VX_TENSOR_INFO_INIT;
    CHECK(vx_result_output_info(second, 1u, &info, &report) == VX_STATUS_OK);
    CHECK(!strcmp(info.name, "mask") && info.dtype == VX_DTYPE_I32 &&
          info.rank == 2u && info.shape[0] == 1 && info.shape[1] == 213 &&
          info.byte_size == 213u * sizeof(int32_t));
    CHECK(vx_result_read(second, "memory", memory, sizeof(memory), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(!memcmp(memory, image, sizeof(image)) &&
          !memcmp(memory + 210u, question_three, sizeof(question_three)));
    CHECK(vx_result_read(second, "mask", mask, sizeof(mask), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(!memcmp(mask, image_mask, sizeof(image_mask)) &&
          !memcmp(mask + 210u, question_mask_three,
                  sizeof(question_mask_three)));
    /* Rebinding the context must not mutate the first immutable result. */
    info = (VxTensorInfo)VX_TENSOR_INFO_INIT;
    CHECK(vx_result_output_info(first, 0u, &info, &report) == VX_STATUS_OK);
    CHECK(info.shape[1] == 211 && info.byte_size == 211u * sizeof(float));

    vx_result_release(second);
    vx_result_release(first);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int test_affine_concat_domain_proof(void) {
    static const char* const graph_path =
        "/tmp/volvox-public-api-affine-concat.graph.json";
    static const char* const repeated_dimensions =
        "{\"B\":{\"min\":1,\"max\":4},"
        "\"Q\":{\"min\":1,\"max\":192},"
        "\"M\":{\"min\":211,\"max\":402}}";
    static const char* const repeated_inputs =
        "{\"image\":{\"shape\":[\"B\",210,1],\"dtype\":\"float32\"},"
        "\"question\":{\"shape\":[\"B\",\"Q\",1],"
        "\"dtype\":\"float32\"},"
        "\"image_mask\":{\"shape\":[\"B\",210],"
        "\"dtype\":\"int32\"},"
        "\"question_mask\":{\"shape\":[\"B\",\"Q\"],"
        "\"dtype\":\"int32\"}}";
    static const char* const repeated_inputs_with_public_m =
        "{\"image\":{\"shape\":[\"B\",210,1],\"dtype\":\"float32\"},"
        "\"question\":{\"shape\":[\"B\",\"Q\",1],"
        "\"dtype\":\"float32\"},"
        "\"image_mask\":{\"shape\":[\"B\",210],"
        "\"dtype\":\"int32\"},"
        "\"question_mask\":{\"shape\":[\"B\",\"Q\"],"
        "\"dtype\":\"int32\"},"
        "\"declared_m\":{\"shape\":[\"M\"],\"dtype\":\"int32\"}}";
    static const char* const repeated_nodes =
        "[{\"id\":\"memory\",\"opType\":\"Concat\","
        "\"inputs\":{\"input0\":\"image\",\"input1\":\"question\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"memory\","
        "\"dtype\":\"float32\",\"shape\":[\"B\",\"M\",1]}},"
        "\"params\":{\"axis\":1}},"
        "{\"id\":\"mask\",\"opType\":\"Concat\","
        "\"inputs\":{\"input0\":\"image_mask\","
        "\"input1\":\"question_mask\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"mask\","
        "\"dtype\":\"int32\",\"shape\":[\"B\",\"M\"]}},"
        "\"params\":{\"axis\":1}}]";
    static const char* const prior_dimensions =
        "{\"B\":{\"min\":1,\"max\":4},"
        "\"Q\":{\"min\":1,\"max\":192},"
        "\"S\":{\"min\":211,\"max\":402},"
        "\"M\":{\"min\":211,\"max\":402}}";
    static const char* const prior_inputs =
        "{\"seed\":{\"shape\":[\"B\",\"S\"],\"dtype\":\"float32\"},"
        "\"fixed\":{\"shape\":[\"B\",210],\"dtype\":\"float32\"},"
        "\"question\":{\"shape\":[\"B\",\"Q\"],"
        "\"dtype\":\"float32\"}}";
    static const char* const prior_nodes =
        "[{\"id\":\"define_m\",\"opType\":\"Identity\","
        "\"inputs\":{\"input\":\"seed\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"defined_m\","
        "\"dtype\":\"float32\",\"shape\":[\"B\",\"M\"]}},"
        "\"params\":{}},"
        "{\"id\":\"joined\",\"opType\":\"Concat\","
        "\"inputs\":{\"input0\":\"fixed\",\"input1\":\"question\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"joined\","
        "\"dtype\":\"float32\",\"shape\":[\"B\",\"M\"]}},"
        "\"params\":{\"axis\":1}}]";
    static const char* const conflict_dimensions =
        "{\"B\":{\"min\":1,\"max\":4},"
        "\"Q\":{\"min\":1,\"max\":192},"
        "\"R\":{\"min\":2,\"max\":193},"
        "\"M\":{\"min\":211,\"max\":402}}";
    static const char* const conflict_inputs =
        "{\"fixed_q\":{\"shape\":[\"B\",210],\"dtype\":\"float32\"},"
        "\"question\":{\"shape\":[\"B\",\"Q\"],"
        "\"dtype\":\"float32\"},"
        "\"fixed_r\":{\"shape\":[\"B\",209],\"dtype\":\"float32\"},"
        "\"response\":{\"shape\":[\"B\",\"R\"],"
        "\"dtype\":\"float32\"}}";
    static const char* const conflict_nodes =
        "[{\"id\":\"from_q\",\"opType\":\"Concat\","
        "\"inputs\":{\"input0\":\"fixed_q\",\"input1\":\"question\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"from_q\","
        "\"dtype\":\"float32\",\"shape\":[\"B\",\"M\"]}},"
        "\"params\":{\"axis\":1}},"
        "{\"id\":\"from_r\",\"opType\":\"Concat\","
        "\"inputs\":{\"input0\":\"fixed_r\",\"input1\":\"response\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"from_r\","
        "\"dtype\":\"float32\",\"shape\":[\"B\",\"M\"]}},"
        "\"params\":{\"axis\":1}}]";
    static const char* const two_dynamic_dimensions =
        "{\"B\":{\"min\":1,\"max\":4},"
        "\"Q\":{\"min\":1,\"max\":192},"
        "\"R\":{\"min\":1,\"max\":192},"
        "\"N\":{\"min\":2,\"max\":384}}";
    static const char* const two_dynamic_inputs =
        "{\"question\":{\"shape\":[\"B\",\"Q\"],"
        "\"dtype\":\"float32\"},"
        "\"response\":{\"shape\":[\"B\",\"R\"],"
        "\"dtype\":\"float32\"}}";
    static const char* const two_dynamic_nodes =
        "[{\"id\":\"two_dynamic\",\"opType\":\"Concat\","
        "\"inputs\":{\"input0\":\"question\",\"input1\":\"response\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"joined\","
        "\"dtype\":\"float32\",\"shape\":[\"B\",\"N\"]}},"
        "\"params\":{\"axis\":1}}]";
    static const char* const one_dynamic_inputs =
        "{\"fixed\":{\"shape\":[\"B\",210],\"dtype\":\"float32\"},"
        "\"question\":{\"shape\":[\"B\",\"Q\"],"
        "\"dtype\":\"float32\"}}";
    static const char* const one_dynamic_nodes =
        "[{\"id\":\"malformed\",\"opType\":\"Concat\","
        "\"inputs\":{\"input0\":\"fixed\",\"input1\":\"question\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"joined\","
        "\"dtype\":\"float32\",\"shape\":[\"B\",\"M\"]}},"
        "\"params\":{\"axis\":1}}]";
    VxRuntimeOptions options = VX_RUNTIME_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;

    CHECK(vx_runtime_create(&options, &runtime, &report) == VX_STATUS_OK);
    CHECK(compile_affine_concat_case(
              runtime, graph_path, repeated_dimensions, repeated_inputs,
              repeated_nodes, "[\"memory\",\"mask\"]", VX_STATUS_OK,
              NULL) == 0);
    CHECK(execute_repeated_affine_concat_case(runtime, graph_path) == 0);
    CHECK(compile_affine_concat_case(
              runtime, graph_path,
              "{\"B\":{\"min\":1,\"max\":4},"
              "\"Q\":{\"min\":8,\"max\":192,\"multiple_of\":8},"
              "\"M\":{\"min\":216,\"max\":400,\"multiple_of\":8}}",
              "{\"fixed\":{\"shape\":[\"B\",208],"
              "\"dtype\":\"float32\"},"
              "\"question\":{\"shape\":[\"B\",\"Q\"],"
              "\"dtype\":\"float32\"}}",
              one_dynamic_nodes, "[\"joined\"]", VX_STATUS_OK, NULL) == 0);
    CHECK(compile_affine_concat_case(
              runtime, graph_path, repeated_dimensions,
              repeated_inputs_with_public_m, repeated_nodes,
              "[\"memory\",\"mask\"]", VX_STATUS_BACKEND_UNSUPPORTED,
              "relation_reason=target-predefined") == 0);
    CHECK(compile_affine_concat_case(
              runtime, graph_path, prior_dimensions, prior_inputs, prior_nodes,
              "[\"joined\"]", VX_STATUS_BACKEND_UNSUPPORTED,
              "relation_reason=target-predefined") == 0);
    CHECK(compile_affine_concat_case(
              runtime, graph_path, conflict_dimensions, conflict_inputs,
              conflict_nodes, "[\"from_q\",\"from_r\"]",
              VX_STATUS_BACKEND_UNSUPPORTED,
              "relation_reason=affine-relation-conflict") == 0);
    CHECK(compile_affine_concat_case(
              runtime, graph_path, two_dynamic_dimensions, two_dynamic_inputs,
              two_dynamic_nodes, "[\"joined\"]",
              VX_STATUS_BACKEND_UNSUPPORTED,
              "relation_reason=two-dynamic-terms") == 0);
    CHECK(compile_affine_concat_case(
              runtime, graph_path,
              "{\"B\":{\"min\":1,\"max\":4},"
              "\"Q\":{\"min\":1,\"max\":192}}",
              one_dynamic_inputs,
              "[{\"id\":\"input_bound_target\","
              "\"opType\":\"Concat\","
              "\"inputs\":{\"input0\":\"fixed\","
              "\"input1\":\"question\"},"
              "\"outputs\":{\"out\":{\"tensor\":\"joined\","
              "\"dtype\":\"float32\",\"shape\":[\"B\",\"Q\"]}},"
              "\"params\":{\"axis\":1}}]",
              "[\"joined\"]", VX_STATUS_BACKEND_UNSUPPORTED,
              "relation_reason=target-not-output-only") == 0);
    CHECK(compile_affine_concat_case(
              runtime, graph_path,
              "{\"B\":{\"min\":1,\"max\":4},"
              "\"Q\":{\"min\":1,\"max\":192},"
              "\"M\":{\"min\":212,\"max\":402}}",
              one_dynamic_inputs, one_dynamic_nodes, "[\"joined\"]",
              VX_STATUS_BACKEND_UNSUPPORTED,
              "relation_reason=affine-progression-mismatch") == 0);
    CHECK(compile_affine_concat_case(
              runtime, graph_path,
              "{\"B\":{\"min\":1,\"max\":4},"
              "\"Q\":{\"min\":1,\"max\":192},"
              "\"M\":{\"min\":211,\"max\":401}}",
              one_dynamic_inputs, one_dynamic_nodes, "[\"joined\"]",
              VX_STATUS_BACKEND_UNSUPPORTED,
              "relation_reason=affine-progression-mismatch") == 0);
    CHECK(compile_affine_concat_case(
              runtime, graph_path,
              "{\"B\":{\"min\":1,\"max\":4},"
              "\"Q\":{\"min\":1,\"max\":192},"
              "\"M\":{\"min\":211,\"max\":402,\"multiple_of\":2}}",
              one_dynamic_inputs, one_dynamic_nodes, "[\"joined\"]",
              VX_STATUS_BACKEND_UNSUPPORTED,
              "relation_reason=affine-progression-mismatch") == 0);
    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    vx_runtime_release(runtime);
    CHECK(remove(graph_path) == 0);
    return 0;
}

static int test_symbolic_target_shape_params(void) {
    const char* valid_path =
        "/tmp/volvox-public-api-symbolic-target.graph.json";
    const char* invalid_target_path =
        "/tmp/volvox-public-api-invalid-symbolic-target.graph.json";
    const char* invalid_array_path =
        "/tmp/volvox-public-api-invalid-symbolic-array.graph.json";
    const char* valid_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"B\":{\"min\":1,\"max\":2},"
        "\"Q\":{\"min\":1,\"max\":3}},"
        "\"inputs\":{\"x\":{\"shape\":[\"B\",\"Q\",1],"
        "\"dtype\":\"float32\"}},\"nodes\":["
        "{\"id\":\"expand\",\"opType\":\"Expand\","
        "\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"expanded\",\"dtype\":\"float32\","
        "\"shape\":[\"B\",\"Q\",2]}},"
        "\"params\":{\"shape\":[\"B\",\"Q\",2]}},"
        "{\"id\":\"reshape\",\"opType\":\"Reshape\","
        "\"inputs\":{\"input\":\"expanded\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"y\",\"dtype\":\"float32\","
        "\"shape\":[\"B\",2,\"Q\"]}},"
        "\"params\":{\"shape\":[\"B\",2,\"Q\"]}}],"
        "\"outputs\":[\"y\"]}";
    const char* invalid_target_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"B\":{\"min\":1,\"max\":2},"
        "\"Q\":{\"min\":1,\"max\":2}},"
        "\"inputs\":{\"x\":{\"shape\":[\"B\",\"Q\",1],"
        "\"dtype\":\"float32\"}},\"nodes\":["
        "{\"id\":\"bad_target\",\"opType\":\"Expand\","
        "\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"y\",\"dtype\":\"float32\","
        "\"shape\":[\"B\",\"Q\",2]}},"
        "\"params\":{\"shape\":[\"B\",\"B\",2]}}],"
        "\"outputs\":[\"y\"]}";
    const char* invalid_array_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"B\":{\"min\":1,\"max\":2}},"
        "\"inputs\":{\"x\":{\"shape\":[\"B\",2],"
        "\"dtype\":\"float32\"}},\"nodes\":["
        "{\"id\":\"bad_perm\",\"opType\":\"Transpose\","
        "\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"y\",\"dtype\":\"float32\","
        "\"shape\":[2,\"B\"]}},"
        "\"params\":{\"perm\":[\"B\",0]}}],"
        "\"outputs\":[\"y\"]}";
    VxRuntimeOptions options = VX_RUNTIME_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxResult* result = NULL;
    const float first_input[3] = {1.0f, 2.0f, 3.0f};
    const float second_input[2] = {4.0f, 5.0f};
    const float first_expected[6] = {1.0f, 1.0f, 2.0f,
                                     2.0f, 3.0f, 3.0f};
    const float second_expected[4] = {4.0f, 4.0f, 5.0f, 5.0f};
    float output[6] = {0};
    VxTensorBinding first_binding = {
        sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 3u, {1, 3, 1},
        first_input, sizeof(first_input), VX_MEMORY_HOST,
    };
    VxTensorBinding second_binding = {
        sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 3u, {2, 1, 1},
        second_input, sizeof(second_input), VX_MEMORY_HOST,
    };

    CHECK(write_text(valid_path, valid_graph) == 0);
    CHECK(write_text(invalid_target_path, invalid_target_graph) == 0);
    CHECK(write_text(invalid_array_path, invalid_array_graph) == 0);
    CHECK(vx_runtime_create(&options, &runtime, &report) == VX_STATUS_OK);
    CHECK(create_cpu_context(runtime, valid_path, &model, &compiled,
                             &context) == 0);
    CHECK(vx_execution_context_execute(
              context, &first_binding, 1u, &result, &report) == VX_STATUS_OK);
    {
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        CHECK(vx_result_output_info(result, 0u, &info, &report) == VX_STATUS_OK);
        CHECK(!strcmp(info.name, "y") && info.rank == 3u &&
              info.shape[0] == 1 && info.shape[1] == 2 &&
              info.shape[2] == 3 && info.byte_size == sizeof(first_expected));
    }
    CHECK(vx_result_read(result, "y", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(!memcmp(output, first_expected, sizeof(first_expected)));
    vx_result_release(result);
    result = NULL;
    memset(output, 0, sizeof(output));
    CHECK(vx_execution_context_execute(
              context, &second_binding, 1u, &result, &report) == VX_STATUS_OK);
    {
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        CHECK(vx_result_output_info(result, 0u, &info, &report) == VX_STATUS_OK);
        CHECK(info.rank == 3u && info.shape[0] == 2 &&
              info.shape[1] == 2 && info.shape[2] == 1 &&
              info.byte_size == sizeof(second_expected));
    }
    CHECK(vx_result_read(result, "y", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(!memcmp(output, second_expected, sizeof(second_expected)));
    vx_result_release(result);
    result = NULL;
    CHECK(vx_execution_context_execute(
              context, &first_binding, 1u, &result, &report) == VX_STATUS_OK);
    CHECK(strstr(report.route_evidence, "shape_plan=hit") != NULL);
    memset(output, 0, sizeof(output));
    CHECK(vx_result_read(result, "y", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(!memcmp(output, first_expected, sizeof(first_expected)));
    vx_result_release(result);
    result = NULL;
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);

    context = NULL;
    compiled = NULL;
    model = NULL;
    CHECK(create_cpu_context(runtime, invalid_target_path, &model, &compiled,
                             &context) == 0);
    first_binding.rank = 3u;
    first_binding.shape[0] = 1;
    first_binding.shape[1] = 2;
    first_binding.shape[2] = 1;
    first_binding.byte_size = 2u * sizeof(float);
    CHECK(vx_execution_context_execute(
              context, &first_binding, 1u, &result, &report) ==
          VX_STATUS_INVALID_GRAPH);
    CHECK(result == NULL && !strcmp(report.reason, "INVALID_GRAPH"));
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);

    context = NULL;
    compiled = NULL;
    model = NULL;
    CHECK(create_cpu_context(runtime, invalid_array_path, &model, &compiled,
                             &context) == 0);
    first_binding.rank = 2u;
    first_binding.shape[1] = 2;
    first_binding.byte_size = 2u * sizeof(float);
    CHECK(vx_execution_context_execute(
              context, &first_binding, 1u, &result, &report) ==
          VX_STATUS_INVALID_GRAPH);
    CHECK(result == NULL && !strcmp(report.reason, "INVALID_GRAPH"));
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    vx_runtime_release(runtime);
    CHECK(remove(valid_path) == 0);
    CHECK(remove(invalid_target_path) == 0);
    CHECK(remove(invalid_array_path) == 0);
    return 0;
}

static int test_qbatch_context_workspace_resource_proof(void) {
    const char* graph_path =
        "/tmp/volvox-public-api-qbatch-workspace.graph.json";
    const char* weights_path =
        "/tmp/volvox-public-api-qbatch-workspace.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{"
        "\"M\":{\"min\":2,\"max\":6},"
        "\"K\":{\"min\":4,\"max\":10,\"multiple_of\":2},"
        "\"N\":{\"min\":16,\"max\":18,\"multiple_of\":2}},"
        "\"inputs\":{"
        "\"a\":{\"shape\":[\"M\",\"K\"],\"dtype\":\"uint8\"},"
        "\"b\":{\"shape\":[\"K\",\"N\"],\"dtype\":\"int8\"}},"
        "\"nodes\":[{\"id\":\"qmm\",\"opType\":\"QBatchMatMul\","
        "\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"int8\",\"shape\":[\"M\",\"N\"]}},"
        "\"params\":{}}],\"outputs\":[\"y\"],"
        "\"quantization\":{"
        "\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{"
        "\"a\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"a.scale\",\"zero_point_tensor\":\"a.zero\"},"
        "\"b\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"b.scale\",\"zero_point_tensor\":\"b.zero\"},"
        "\"y\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"y.scale\",\"zero_point_tensor\":\"y.zero\"}}}}";
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* first = NULL;
    VxExecutionContext* second = NULL;
    VxResult* result = NULL;
    const char* weights[] = {weights_path};
    uint8_t a[60];
    int8_t b[180] = {0};
    int8_t first_output[32];
    int8_t maximum_output[108];
    VxTensorBinding bindings[2] = {
        {sizeof(VxTensorBinding), "a", VX_DTYPE_U8, 2u, {2, 4},
         a, 8u, VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "b", VX_DTYPE_I8, 2u, {4, 16},
         b, 64u, VX_MEMORY_HOST},
    };
    size_t maximum_scratch = 0u;
    uint64_t maximum_resident = 0u;
    uint64_t expected_resident = 0u;
    uintptr_t first_address = 0u;
    uintptr_t second_address = 0u;
    uintptr_t first_original_address = 0u;
    size_t first_bound = 0u;
    size_t first_capacity = 0u;
    size_t second_bound = 0u;
    size_t second_capacity = 0u;
    char scratch_evidence[64];
    char resident_evidence[64];
    FILE* weight_file;
    long weight_file_bytes = -1;
#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
    const size_t expected_scratch = 760u;
#else
    const size_t expected_scratch = 0u;
#endif

    memset(a, 128, sizeof(a));
    CHECK(write_text(graph_path, graph) == 0);
    CHECK(write_qbatch_workspace_weights(weights_path) == 0);
    weight_file = fopen(weights_path, "rb");
    CHECK(weight_file != NULL && fseek(weight_file, 0, SEEK_END) == 0 &&
          (weight_file_bytes = ftell(weight_file)) >= 0 &&
          fclose(weight_file) == 0);
    /* Every logical tensor is dynamic, so the existing resident-domain
     * evidence charges three times its maximum bytes (3 * (60 + 180 + 108)),
     * the raw weight file, and the context-owned QBatch workspace. */
    expected_resident = (uint64_t)weight_file_bytes + UINT64_C(1044) +
        (uint64_t)expected_scratch;
    CHECK(vx_runtime_create(
              &runtime_options, &runtime, &report) == VX_STATUS_OK);
    source.graph_path = graph_path;
    source.weight_paths = weights;
    source.weight_path_count = 1u;
    CHECK(vx_runtime_load_model(
              runtime, &source, &model, &report) == VX_STATUS_OK);
    CHECK(vx_model_compile(
              model, &policy, &compiled, &report) == VX_STATUS_OK);
    CHECK(vx_public_api_test_compiled_resource_bounds(
              compiled, &maximum_scratch, &maximum_resident) == 0);
    CHECK(maximum_scratch == expected_scratch &&
          maximum_resident == expected_resident);
    CHECK(snprintf(scratch_evidence, sizeof(scratch_evidence),
                   "max_typed_scratch=%zu", expected_scratch) > 0);
    CHECK(strstr(report.route_evidence, scratch_evidence) != NULL);
    CHECK(snprintf(resident_evidence, sizeof(resident_evidence),
                   "max_resident=%" PRIu64, expected_resident) > 0);
    CHECK(strstr(report.route_evidence, resident_evidence) != NULL);

    CHECK(vx_compiled_model_create_context(
              compiled, &context_options, &first, &report) == VX_STATUS_OK);
    CHECK(vx_compiled_model_create_context(
              compiled, &context_options, &second, &report) == VX_STATUS_OK);
    CHECK(vx_public_api_test_cpu_typed_workspace_state(
              first, &first_address, &first_bound, &first_capacity) == 0);
    CHECK(vx_public_api_test_cpu_typed_workspace_state(
              second, &second_address, &second_bound, &second_capacity) == 0);
    first_original_address = first_address;
    CHECK(first_bound == expected_scratch &&
          first_capacity == expected_scratch &&
          second_bound == expected_scratch &&
          second_capacity == expected_scratch);
    if (expected_scratch) {
        CHECK(first_address != 0u && second_address != 0u &&
              first_address != second_address);
    } else {
        CHECK(first_address == 0u && second_address == 0u);
    }
    CHECK(vx_public_api_test_cpu_typed_workspace_reconfigure(
              first, expected_scratch + 4u) == -1);
    CHECK(vx_public_api_test_cpu_typed_workspace_state(
              first, &first_address, &first_bound, &first_capacity) == 0);
    CHECK(first_address == first_original_address &&
          first_bound == expected_scratch &&
          first_capacity == expected_scratch);

    CHECK(vx_execution_context_execute(
              first, bindings, 2u, &result, &report) == VX_STATUS_OK);
    memset(first_output, 1, sizeof(first_output));
    CHECK(vx_result_read(result, "y", first_output, sizeof(first_output), NULL,
                         &report) == VX_STATUS_OK);
    for (size_t index = 0; index < sizeof(first_output); index++)
        CHECK(first_output[index] == 0);
    vx_result_release(result);
    result = NULL;
    CHECK(vx_execution_context_close(first, &report) == VX_STATUS_OK);
    vx_execution_context_release(first);
    first = NULL;

    /* Destroying one context cannot release or replace another context's
     * workspace. Exercise the declared maximum so ASAN also checks the exact
     * end of the proved 760-byte packed layout. */
    bindings[0].shape[0] = 6;
    bindings[0].shape[1] = 10;
    bindings[0].byte_size = sizeof(a);
    bindings[1].shape[0] = 10;
    bindings[1].shape[1] = 18;
    bindings[1].byte_size = sizeof(b);
    CHECK(vx_execution_context_execute(
              second, bindings, 2u, &result, &report) == VX_STATUS_OK);
    memset(maximum_output, 1, sizeof(maximum_output));
    CHECK(vx_result_read(result, "y", maximum_output,
                         sizeof(maximum_output), NULL,
                         &report) == VX_STATUS_OK);
    for (size_t index = 0; index < sizeof(maximum_output); index++)
        CHECK(maximum_output[index] == 0);
    vx_result_release(result);
    result = NULL;
    CHECK(vx_public_api_test_cpu_typed_workspace_state(
              second, &first_address, &first_bound, &first_capacity) == 0);
    CHECK(first_address == second_address &&
          first_bound == second_bound && first_capacity == second_capacity);

    CHECK(vx_execution_context_close(second, &report) == VX_STATUS_OK);
    vx_execution_context_release(second);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    vx_runtime_release(runtime);
    CHECK(remove(graph_path) == 0);
    CHECK(remove(weights_path) == 0);
    return 0;
}

int main(int argc, char** argv) {
    const char* graph_a = "/tmp/volvox-public-api-a.graph.json";
    const char* graph_b = "/tmp/volvox-public-api-b.graph.json";
    const char* provider_graph_path =
        "/tmp/volvox-public-api-provider.graph.json";
    const char* provider_weights_path =
        "/tmp/volvox-public-api-provider.safetensors";
    const char* invalid_graph_path = "/tmp/volvox-public-api-invalid.graph.json";
    const char* add_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},"
        "\"inputs\":{\"a\":{\"shape\":[2],\"dtype\":\"float32\"},"
        "\"b\":{\"shape\":[2],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"id\":\"sum\",\"opType\":\"Add\","
        "\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"sum\","
        "\"dtype\":\"float32\",\"shape\":[2]}},\"params\":{}},"
        "{\"id\":\"final\",\"opType\":\"Add\","
        "\"inputs\":{\"a\":\"sum\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"final\","
        "\"dtype\":\"float32\",\"shape\":[2]}},\"params\":{}}],"
        "\"outputs\":[\"sum\",\"final\"]}";
    const char* identity_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},"
        "\"inputs\":{\"x\":{\"shape\":[3],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"identity\",\"opType\":\"Identity\","
        "\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"y\",\"dtype\":\"float32\",\"shape\":[3]}},"
        "\"params\":{}}],\"outputs\":[\"y\"]}";
    const char* provider_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"F\":{\"min\":4,\"max\":8,"
        "\"multiple_of\":4}},"
        "\"inputs\":{\"value\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"identity\",\"opType\":\"Identity\","
        "\"inputs\":{\"input\":\"value\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"mock-out\","
        "\"dtype\":\"float32\",\"shape\":[1]}},\"params\":{}}],"
        "\"outputs\":[\"mock-out\"],"
        "\"banks\":{\"experts\":\"F\"}}";
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model_a = NULL;
    VxCompiledModel* compiled_a = NULL;
    VxExecutionContext* context_a = NULL;
    VxExecutionContext* context_b = NULL;
    VxModel* model_b = NULL;
    VxCompiledModel* compiled_b = NULL;
    VxExecutionContext* context_c = NULL;
    VxExecutionContext* decode_a = NULL;
    VxExecutionContext* decode_b = NULL;
    VxResult* retained = NULL;
    VxResult* stable_first = NULL;
    pthread_t first_thread;
    pthread_t second_thread;
    ThreadCase first = {0};
    ThreadCase second = {0};
    CpuOverlapProbe overlap = {0};

    if (argc == 2 && !strcmp(argv[1], "--affine-concat-domain")) {
        CHECK(test_affine_concat_domain_proof() == 0);
        puts("native public affine Concat domain tests passed");
        return 0;
    }
    CHECK(argc == 1);
    CHECK(write_text(graph_a, add_graph) == 0);
    CHECK(write_text(graph_b, identity_graph) == 0);
    CHECK(write_text(provider_graph_path, provider_graph) == 0);
    CHECK(write_provider_bank_weights(provider_weights_path) == 0);
    CHECK(write_text(invalid_graph_path, "{\"inputs\":{},\"nodes\":[],\"outputs\":[]}") == 0);
    runtime_options.cpu_threads = 2;
    CHECK(vx_runtime_create(&runtime_options, &runtime, &report) == VX_STATUS_OK);
    {
        VxModelSource noncanonical = VX_MODEL_SOURCE_INIT;
        VxModel* rejected = NULL;
        noncanonical.graph_path = "/tmp/config.json";
        CHECK(vx_runtime_load_model(runtime, &noncanonical, &rejected, &report) ==
              VX_STATUS_INVALID_ARGUMENT);
        CHECK(rejected == NULL && !strcmp(report.reason, "INVALID_GRAPH_PATH"));
    }
    {
        VxModelSource missing = VX_MODEL_SOURCE_INIT;
        VxModel* rejected = NULL;
        missing.graph_path = "/tmp/volvox-public-api-does-not-exist.graph.json";
        CHECK(vx_runtime_load_model(runtime, &missing, &rejected, &report) ==
              VX_STATUS_IO_ERROR);
        CHECK(report.status == VX_STATUS_IO_ERROR && report.stage == VX_STAGE_MODEL_LOAD);
    }
    {
        VxModelSource invalid_source = VX_MODEL_SOURCE_INIT;
        VxModel* invalid_model = NULL;
        const char* unreadable_weights[] = {
            "/tmp/volvox-public-api-does-not-exist.safetensors"
        };
        invalid_source.graph_path = invalid_graph_path;
        invalid_source.weight_paths = unreadable_weights;
        invalid_source.weight_path_count = 1;
        CHECK(vx_runtime_load_model(runtime, &invalid_source, &invalid_model, &report) ==
              VX_STATUS_INVALID_GRAPH);
        CHECK(invalid_model == NULL && report.stage == VX_STAGE_MODEL_LOAD &&
              !strcmp(report.reason, "INVALID_GRAPH_CONTRACT"));
    }
    {
        const char* invalid_contracts[] = {
            "{\"format\":\"volvox-graph/v1\","
                "\"shape_system\":\"volvox-bounded-shape/v1\","
                "\"dimensions\":{},\"inputs\":{"
                "\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
                "\"nodes\":[],\"outputs\":[\"x\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{},\"nodes\":[],"
                "\"outputs\":{\"out\":\"y\"}}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{},\"nodes\":[]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{},\"nodes\":[],"
                "\"outputs\":[]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{},\"nodes\":[],"
                "\"outputs\":[\"y\",\"y\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{},\"nodes\":[],"
                "\"outputs\":[\"\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{},\"nodes\":["
                "{\"op\":\"Identity\"}],\"outputs\":[\"y\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
                "\"x\":{\"shape\":[1],\"dtype\":\"F32\"}},"
                "\"nodes\":[],\"outputs\":[\"x\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
                "\"x\":{\"shape\":[1]}},\"nodes\":[],"
                "\"outputs\":[\"x\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
                "\"x\":{\"shape\":[1],\"dtype\":\"float16\"}},"
                "\"nodes\":[],\"outputs\":[\"x\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
                "\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
                "\"nodes\":[{\"opType\":\"Identity\","
                "\"inputs\":{\"input\":\"x\"},"
                "\"outputs\":{\"out\":\"y\"},"
                "\"outputs_shape\":{\"out\":[1]},"
                "\"outputs_dtype\":{\"out\":\"float16\"}}],"
                "\"outputs\":[\"y\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
                "\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
                "\"nodes\":[{\"opType\":\"Identity\","
                "\"inputs\":{\"input\":\"x\"},"
                "\"outputs\":{\"out\":\"y\"},"
                "\"outputs_shape\":{\"out\":[1]},"
                "\"params\":{\"backend\":{\"options\":["
                "{\"zero_point\":0}]}}}],\"outputs\":[\"y\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
                "\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
                "\"nodes\":[{\"opType\":\"RotaryEmbedding\","
                "\"inputs\":{\"input\":\"x\"},"
                "\"outputs\":{\"out\":\"y\"},"
                "\"outputs_shape\":{\"out\":[1]}}],"
                "\"outputs\":[\"y\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
                "\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
                "\"nodes\":[{\"opType\":\"DefinitelyUnknown\","
                "\"inputs\":{\"input\":\"x\"},"
                "\"outputs\":{\"out\":\"y\"},"
                "\"outputs_shape\":{\"out\":[1]}}],"
                "\"outputs\":[\"y\"]}",
        };
        const char* unreadable_weights[] = {
            "/tmp/volvox-public-api-does-not-exist.safetensors"
        };
        for (size_t index = 0;
             index < sizeof(invalid_contracts) / sizeof(invalid_contracts[0]);
             index++) {
            VxModelSource invalid_source = VX_MODEL_SOURCE_INIT;
            VxModel* invalid_model = NULL;
            CHECK(write_text(invalid_graph_path, invalid_contracts[index]) == 0);
            invalid_source.graph_path = invalid_graph_path;
            invalid_source.weight_paths = unreadable_weights;
            invalid_source.weight_path_count = 1;
            CHECK(vx_runtime_load_model(runtime, &invalid_source, &invalid_model,
                                        &report) == VX_STATUS_INVALID_GRAPH);
            CHECK(invalid_model == NULL && report.stage == VX_STAGE_MODEL_LOAD);
        }
    }
    {
        const char* raw_byte_graph =
            "{\"format\":\"volvox-graph/v1\","
            "\"dimensions\":{},\"inputs\":{"
            "\"x\":{\"shape\":[1],\"dtype\":\"int8\"}},"
            "\"nodes\":[],\"outputs\":[\"x\"]}";
        VxModelSource source = VX_MODEL_SOURCE_INIT;
        VxModel* raw_model = NULL;
        CHECK(write_text(invalid_graph_path, raw_byte_graph) == 0);
        source.graph_path = invalid_graph_path;
        CHECK(vx_runtime_load_model(runtime, &source, &raw_model, &report) ==
              VX_STATUS_OK);
        CHECK(raw_model != NULL);
        vx_model_release(raw_model);
    }
    CHECK(create_cpu_context(runtime, graph_a, &model_a, &compiled_a, &context_a) == 0);
    {
        VxAdapterSource adapter_source = VX_ADAPTER_SOURCE_INIT;
        VxAdapterRevision first_revision = VX_ADAPTER_REVISION_INIT;
        VxAdapterRevision second_revision = VX_ADAPTER_REVISION_INIT;
        VxAdapterRevision missing_revision = VX_ADAPTER_REVISION_INIT;
        VxRevisionInfo model_revision = VX_REVISION_INFO_INIT;
        VxReport pinned = VX_REPORT_INIT;
        VxTensorSpec input_spec = VX_TENSOR_SPEC_INIT;
        CHECK(vx_compiled_model_report(compiled_a, &pinned) == VX_STATUS_OK);
        uint64_t base_adapter_id = pinned.adapter_id;
        uint64_t base_adapter_revision = pinned.adapter_revision;
        adapter_source.adapter_name = "request-lora";
        CHECK(vx_model_publish_adapter(model_a, &adapter_source,
                                       &first_revision, &report) == VX_STATUS_OK);
        CHECK(first_revision.adapter_id && first_revision.adapter_revision == 1);
        CHECK(vx_model_revision_info(model_a, &model_revision, &report) ==
              VX_STATUS_OK);
        CHECK(model_revision.adapter_id == first_revision.adapter_id &&
              model_revision.adapter_revision == first_revision.adapter_revision);
        CHECK(vx_compiled_model_report(compiled_a, &pinned) == VX_STATUS_OK);
        CHECK(pinned.adapter_id == base_adapter_id &&
              pinned.adapter_revision == base_adapter_revision);
        CHECK(vx_execution_context_rebind_adapter(context_a, &report) ==
              VX_STATUS_OK);
        CHECK(report.adapter_id == first_revision.adapter_id &&
              report.adapter_revision == first_revision.adapter_revision);

        CHECK(vx_model_publish_adapter(model_a, &adapter_source,
                                       &second_revision, &report) == VX_STATUS_OK);
        CHECK(second_revision.adapter_id == first_revision.adapter_id &&
              second_revision.adapter_revision == first_revision.adapter_revision + 1u);
        CHECK(vx_execution_context_input_spec(
                  context_a, 0u, &input_spec, &report) == VX_STATUS_OK);
        CHECK(report.adapter_revision == first_revision.adapter_revision);
        CHECK(vx_execution_context_select_adapter(context_a, &first_revision,
                                                   &report) == VX_STATUS_OK);
        CHECK(report.adapter_revision == first_revision.adapter_revision);
        CHECK(vx_execution_context_rebind_adapter(context_a, &report) ==
              VX_STATUS_OK);
        CHECK(report.adapter_revision == second_revision.adapter_revision);
        missing_revision.adapter_id = second_revision.adapter_id;
        missing_revision.adapter_revision = second_revision.adapter_revision + 100u;
        CHECK(vx_execution_context_select_adapter(context_a, &missing_revision,
                                                   &report) == VX_STATUS_NOT_FOUND);
        CHECK(report.adapter_revision == second_revision.adapter_revision);
    }
    {
        VxBackendPolicy preferred = VX_BACKEND_POLICY_INIT;
        VxCompiledModel* fallback = NULL;
        const char* ordered[] = { "missing-provider", "cpu" };
        preferred.backends = ordered;
        preferred.backend_count = 2;
        CHECK(vx_model_compile(model_a, &preferred, &fallback, &report) == VX_STATUS_OK);
        CHECK(fallback && !strcmp(report.backend, "cpu") &&
              report.tier_fallback_used && report.candidate_count == 2 &&
              strstr(report.candidate_outcomes, "missing-provider:unavailable") &&
              strstr(report.fallback_evidence, "->cpu"));
        vx_compiled_model_release(fallback);
    }
    {
        VxBackendPolicy unavailable = VX_BACKEND_POLICY_INIT;
        VxCompiledModel* rejected = NULL;
        VxReport compiled_report = VX_REPORT_INIT;
        const char* required[] = { "missing-provider" };
        CHECK(vx_compiled_model_report(compiled_a, &compiled_report) == VX_STATUS_OK);
        unavailable.mode = VX_BACKEND_REQUIRE;
        unavailable.backends = required;
        unavailable.backend_count = 1;
        CHECK(vx_model_compile(model_a, &unavailable, &rejected, &report) ==
              VX_STATUS_BACKEND_REQUIRED);
        CHECK(rejected == NULL && report.status == VX_STATUS_BACKEND_REQUIRED &&
              !strcmp(report.reason, "BACKEND_REQUIRED") &&
              report.candidate_count == 1 && report.model_id ==
              compiled_report.model_id);
    }
    {
        VxReport route = VX_REPORT_INIT;
        CHECK(vx_public_api_test_evaluate_builtin_route(
                  "cuda", "cpu", VX_OPERATOR_FALLBACK_ALLOW, &route) ==
              VX_STATUS_OK);
        CHECK(route.route_attested && route.operator_fallback_used &&
              strstr(route.route_evidence, "selected=0;fallback=1") &&
              strstr(route.fallback_evidence, "operator=used"));
        route = (VxReport)VX_REPORT_INIT;
        CHECK(vx_public_api_test_evaluate_builtin_route(
                  "cuda", "cpu", VX_OPERATOR_FALLBACK_FORBID, &route) ==
              VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN);
        CHECK(route.operator_fallback_used &&
              !strcmp(route.reason, "OPERATOR_FALLBACK_FORBIDDEN") &&
              strstr(route.offending_node, "op=Identity"));
        route = (VxReport)VX_REPORT_INIT;
        CHECK(vx_public_api_test_evaluate_builtin_route(
                  "cuda", "cuda-linear", VX_OPERATOR_FALLBACK_FORBID,
                  &route) == VX_STATUS_OK);
        CHECK(route.route_attested && !route.operator_fallback_used);
    }
    {
        VxBackendPolicy builtin = VX_BACKEND_POLICY_INIT;
        VxCompiledModel* selected = NULL;
        const char* required[] = { "vulkan" };
        builtin.mode = VX_BACKEND_REQUIRE;
        builtin.backends = required;
        builtin.backend_count = 1;
        CHECK(vx_model_compile(model_a, &builtin, &selected, &report) ==
              VX_STATUS_BACKEND_UNAVAILABLE);
        CHECK(selected == NULL && !strcmp(report.backend, "vulkan") &&
              !strcmp(report.reason, "BACKEND_UNAVAILABLE"));
        {
            const char* preferred[] = { "vulkan", "cpu" };
            builtin.mode = VX_BACKEND_PREFER;
            builtin.backends = preferred;
            builtin.backend_count = 2;
            CHECK(vx_model_compile(model_a, &builtin, &selected, &report) ==
                  VX_STATUS_OK);
            CHECK(selected && !strcmp(report.backend, "cpu") &&
                  report.tier_fallback_used &&
                  strstr(report.candidate_outcomes, "vulkan:unavailable") &&
                  strstr(report.candidate_outcomes, "cpu:selected"));
            vx_compiled_model_release(selected);
        }
    }
    {
        VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
        CHECK(vx_compiled_model_create_context(compiled_a, &options,
                                                &context_b, &report) == VX_STATUS_OK);
    }
    CHECK(create_cpu_context(runtime, graph_b, &model_b, &compiled_b, &context_c) == 0);
    {
        VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
        options.decode_row_mode = VX_DECODE_ROW_AUTO;
        options.require_incremental = 1;
        CHECK(vx_compiled_model_create_context(compiled_b, &options,
                                                &decode_a, &report) == VX_STATUS_OK);
        CHECK(strstr(report.decode_state, "enabled=1") &&
              strstr(report.decode_state, "seeded=0"));
        CHECK(vx_compiled_model_create_context(compiled_b, &options,
                                                &decode_b, &report) == VX_STATUS_OK);
    }
    CHECK(vx_execution_context_input_count(context_a) == 2);
    CHECK(vx_execution_context_input_count(context_c) == 1);
    {
        VxReport report_a = VX_REPORT_INIT;
        VxReport report_b = VX_REPORT_INIT;
        VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
        uint64_t context_a_id;
        CHECK(vx_compiled_model_report(compiled_a, &report_a) == VX_STATUS_OK);
        CHECK(vx_compiled_model_report(compiled_b, &report_b) == VX_STATUS_OK);
        CHECK(report_a.runtime_id == report_b.runtime_id &&
              report_a.model_id != report_b.model_id &&
              report_a.compiled_model_id != report_b.compiled_model_id &&
              report_a.graph_id != report_b.graph_id &&
              report_a.graph_revision != report_b.graph_revision &&
              report_a.weight_id != report_b.weight_id &&
              report_a.weight_revision == report_b.weight_revision);
        CHECK(vx_execution_context_input_spec(context_a, 0, &spec, &report) ==
              VX_STATUS_OK);
        {
            VxAffineQuantization quantization =
                VX_AFFINE_QUANTIZATION_INIT;
            CHECK(vx_execution_context_input_affine_quantization(
                      context_a, "a", &quantization, &report) == VX_STATUS_OK);
            CHECK(!quantization.defined && quantization.scale == 0.0f &&
                  quantization.zero_point == 0);
        }
        context_a_id = report.context_id;
        spec = (VxTensorSpec)VX_TENSOR_SPEC_INIT;
        CHECK(vx_execution_context_input_spec(context_b, 0, &spec, &report) ==
              VX_STATUS_OK);
        CHECK(context_a_id && report.context_id &&
              context_a_id != report.context_id);
    }

    {
        float a[2] = {1.0f, 2.0f};
        const float b[2] = {3.0f, 4.0f};
        const VxTensorBinding inputs[2] = {
            {sizeof(VxTensorBinding), "a", VX_DTYPE_F32, 1u, {2},
             a, sizeof(a), VX_MEMORY_HOST},
            {sizeof(VxTensorBinding), "b", VX_DTYPE_F32, 1u, {2},
             b, sizeof(b), VX_MEMORY_HOST},
        };
        float sum[2] = {0};
        float too_small = 0.0f;
        size_t required = 0;
        uint64_t execution_adapter_revision;
        VxTensorInfo first_output = VX_TENSOR_INFO_INIT;
        VxTensorInfo second_output = VX_TENSOR_INFO_INIT;
        VxResult* later = NULL;
        CpuInputCommitProbe commit_probe = {a, 2u, 1000.0f, 0};
        CHECK(vx_execution_context_execute(
                  context_a, inputs, 2u, &stable_first, &report) ==
              VX_STATUS_OK);
        CHECK(report.context_id && report.execution_id && report.result_bytes ==
              2u * sizeof(sum) && report.route_attested &&
              !report.operator_fallback_used && report.route_evidence[0] &&
              report.execution_time_ms >= 0.0);
        execution_adapter_revision = report.adapter_revision;
        CHECK(vx_result_output_count(stable_first) == 2);
        CHECK(vx_result_output_info(stable_first, 0, &first_output, &report) == VX_STATUS_OK);
        CHECK(vx_result_output_info(stable_first, 1, &second_output, &report) == VX_STATUS_OK);
        CHECK(!strcmp(first_output.name, "sum") && !strcmp(second_output.name, "final"));
        CHECK(vx_result_read(stable_first, "sum", &too_small, sizeof(too_small),
                             &required, &report) == VX_STATUS_BUFFER_TOO_SMALL);
        CHECK(required == sizeof(sum));
        vx_public_api_test_set_cpu_execute_hook(cpu_input_commit_hook,
                                                &commit_probe);
        CHECK(vx_execution_context_execute(
                  context_a, inputs, 2u, &later, &report) == VX_STATUS_OK);
        vx_public_api_test_set_cpu_execute_hook(NULL, NULL);
        CHECK(commit_probe.called == 1 && a[0] == 1000.0f && a[1] == 1000.0f);
        CHECK(vx_result_read(later, "sum", sum, sizeof(sum), NULL, &report) ==
              VX_STATUS_OK);
        CHECK(closef(sum[0], 4.0f) && closef(sum[1], 6.0f));
        vx_result_release(later);
        {
            VxAdapterSource next_adapter = VX_ADAPTER_SOURCE_INIT;
            VxAdapterRevision next_revision = VX_ADAPTER_REVISION_INIT;
            next_adapter.adapter_name = "request-lora";
            CHECK(vx_model_publish_adapter(model_a, &next_adapter,
                                           &next_revision, &report) ==
                  VX_STATUS_OK);
            CHECK(vx_execution_context_rebind_adapter(context_a, &report) ==
                  VX_STATUS_OK);
            CHECK(next_revision.adapter_revision != execution_adapter_revision);
        }
        CHECK(vx_result_read(stable_first, "sum", sum, sizeof(sum), NULL, &report) ==
              VX_STATUS_OK);
        CHECK(closef(sum[0], 4.0f) && closef(sum[1], 6.0f));
        CHECK(report.adapter_revision == execution_adapter_revision);
    }

    first.context = context_a;
    first.base = 10.0f;
    second.context = context_b;
    second.base = 1000.0f;
    CHECK(pthread_mutex_init(&overlap.mutex, NULL) == 0);
    CHECK(pthread_cond_init(&overlap.condition, NULL) == 0);
    vx_public_api_test_set_cpu_execute_hook(cpu_overlap_hook, &overlap);
    CHECK(pthread_create(&first_thread, NULL, run_thread_case, &first) == 0);
    CHECK(pthread_create(&second_thread, NULL, run_thread_case, &second) == 0);
    CHECK(pthread_join(first_thread, NULL) == 0);
    CHECK(pthread_join(second_thread, NULL) == 0);
    vx_public_api_test_set_cpu_execute_hook(NULL, NULL);
    CHECK(first.ok && second.ok);
    CHECK(overlap.entered == 2 && !overlap.timed_out);
    pthread_cond_destroy(&overlap.condition);
    pthread_mutex_destroy(&overlap.mutex);

    {
        float input[3] = {3.0f, 4.0f, 5.0f};
        float output[3] = {0};
        const VxTensorBinding binding = {
            sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 1u, {3},
            input, sizeof(input), VX_MEMORY_HOST,
        };
        VxResult* prefix = NULL;
        VxResult* rejected = NULL;
        CHECK(vx_execution_context_execute_prefix(context_c, 0,
                                                  &binding, 1u, &rejected,
                                                  &report) ==
              VX_STATUS_INVALID_ARGUMENT);
        CHECK(rejected == NULL &&
              !strcmp(report.reason, "INVALID_PREFIX_ROW_COUNT"));
        CHECK(vx_execution_context_execute_prefix(context_c, 1,
                                                  &binding, 1u, &prefix,
                                                  &report) == VX_STATUS_OK);
        CHECK(report.stage == VX_STAGE_EXECUTE &&
              vx_result_read(prefix, "y", output, sizeof(output), NULL,
                             &report) == VX_STATUS_OK);
        CHECK(!memcmp(input, output, sizeof(input)));
        vx_result_release(prefix);
        memset(output, 0, sizeof(output));
        CHECK(vx_execution_context_execute(
                  context_c, &binding, 1u, &retained, &report) == VX_STATUS_OK);
        CHECK(vx_result_output_count(retained) == 1);
        CHECK(vx_result_read(retained, "y", output, sizeof(output), NULL, &report) ==
              VX_STATUS_OK);
        CHECK(!memcmp(input, output, sizeof(input)));
    }

    {
        float input_a[3] = {10.0f, 11.0f, 12.0f};
        float input_b[3] = {20.0f, 21.0f, 22.0f};
        float next_a[3] = {13.0f, 14.0f, 15.0f};
        float next_b[3] = {23.0f, 24.0f, 25.0f};
        const VxTensorBinding seed_binding_a = {
            sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 1u, {3},
            input_a, sizeof(input_a), VX_MEMORY_HOST,
        };
        const VxTensorBinding seed_binding_b = {
            sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 1u, {3},
            input_b, sizeof(input_b), VX_MEMORY_HOST,
        };
        const VxTensorBinding step_binding_a = {
            sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 1u, {3},
            next_a, sizeof(next_a), VX_MEMORY_HOST,
        };
        const VxTensorBinding step_binding_b = {
            sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 1u, {3},
            next_b, sizeof(next_b), VX_MEMORY_HOST,
        };
        float output[3] = {0};
        VxResult* seed_a = NULL;
        VxResult* seed_b = NULL;
        VxResult* step_a = NULL;
        VxResult* step_b = NULL;
        CHECK(vx_execution_context_decode_seed(
                  decode_a, &seed_binding_a, 1u, &seed_a, &report) ==
              VX_STATUS_OK);
        CHECK(report.stage == VX_STAGE_DECODE &&
              strstr(report.decode_state, "seeded=1") &&
              strstr(report.decode_state, "last=dependency"));
        CHECK(vx_execution_context_decode_seed(
                  decode_b, &seed_binding_b, 1u, &seed_b, &report) ==
              VX_STATUS_OK);
        CHECK(vx_result_read(seed_a, "y", output, sizeof(output), NULL,
                             &report) == VX_STATUS_OK);
        CHECK(!memcmp(output, input_a, sizeof(output)));
        {
            const char* invalid_adapter_path =
                "/tmp/volvox-public-api-invalid-adapter.bin";
            VxAdapterSource invalid_adapter = VX_ADAPTER_SOURCE_INIT;
            VxAdapterRevision invalid_revision = VX_ADAPTER_REVISION_INIT;
            CHECK(write_text(invalid_adapter_path, "not-an-adapter") == 0);
            invalid_adapter.adapter_name = "invalid-decode-adapter";
            invalid_adapter.package_path = invalid_adapter_path;
            CHECK(vx_model_publish_adapter(model_b, &invalid_adapter,
                                           &invalid_revision, &report) ==
                  VX_STATUS_OK);
            CHECK(remove(invalid_adapter_path) == 0);
            CHECK(vx_execution_context_select_adapter(
                      decode_b, &invalid_revision, &report) ==
                  VX_STATUS_INVALID_ARGUMENT);
            /* A rejected adapter package must leave the already seeded decode
             * stream usable with its previously pinned revision. */
            CHECK(vx_execution_context_decode_step(
                      decode_b, -1, &step_binding_b, 1u,
                      &step_b, &report) == VX_STATUS_OK);
            CHECK(vx_result_read(step_b, "y", output, sizeof(output), NULL,
                                 &report) == VX_STATUS_OK);
            CHECK(!memcmp(output, next_b, sizeof(output)));
        }
        CHECK(vx_execution_context_decode_step(
                  decode_a, -1, &step_binding_a, 1u,
                  &step_a, &report) == VX_STATUS_OK);
        CHECK(vx_result_read(step_a, "y", output, sizeof(output), NULL,
                             &report) == VX_STATUS_OK);
        CHECK(!memcmp(output, next_a, sizeof(output)));
        CHECK(vx_execution_context_decode_reset(decode_a, &report) ==
              VX_STATUS_OK);
        {
            VxResult* rejected = NULL;
            CHECK(vx_execution_context_decode_step(
                      decode_a, -1, &step_binding_a, 1u,
                      &rejected, &report) == VX_STATUS_INVALID_ARGUMENT);
            CHECK(rejected == NULL && strstr(report.decode_state, "seeded=0"));
        }
        vx_result_release(step_b);
        vx_result_release(step_a);
        vx_result_release(seed_b);
        vx_result_release(seed_a);
    }

    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    {
        VxCompiledModel* rejected_compile = NULL;
        VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
        VxModelSource source = VX_MODEL_SOURCE_INIT;
        VxModel* rejected_model = NULL;
        float output[3] = {0};
        float final_input[3] = {23.0f, 24.0f, 25.0f};
        const VxTensorBinding final_binding = {
            sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 1u, {3},
            final_input, sizeof(final_input), VX_MEMORY_HOST,
        };
        VxResult* final_step = NULL;
        source.graph_path = graph_b;
        CHECK(vx_model_compile(model_b, &policy, &rejected_compile, &report) ==
              VX_STATUS_HANDLE_DISPOSED);
        CHECK(rejected_compile == NULL);
        CHECK(vx_runtime_load_model(runtime, &source, &rejected_model, &report) ==
              VX_STATUS_HANDLE_DISPOSED);
        CHECK(rejected_model == NULL);
        CHECK(vx_execution_context_decode_step(
                  decode_b, -1, &final_binding, 1u,
                  &final_step, &report) == VX_STATUS_OK);
        CHECK(vx_result_read(final_step, "y", output, sizeof(output), NULL,
                             &report) == VX_STATUS_OK);
        CHECK(closef(output[0], 23.0f));
        vx_result_release(final_step);
    }
    vx_execution_context_release(decode_b);
    vx_execution_context_release(decode_a);
    CHECK(vx_execution_context_close(context_c, &report) == VX_STATUS_OK);
    vx_execution_context_release(context_c);
    vx_compiled_model_release(compiled_b);
    vx_model_release(model_b);
    vx_execution_context_release(context_b);
    vx_execution_context_release(context_a);
    vx_compiled_model_release(compiled_a);
    vx_model_release(model_a);
    vx_runtime_release(runtime);
    {
        float sum[2] = {0};
        CHECK(vx_result_read(stable_first, "sum", sum, sizeof(sum), NULL, &report) ==
              VX_STATUS_OK);
        CHECK(closef(sum[0], 4.0f) && closef(sum[1], 6.0f));
    }
    vx_result_release(stable_first);
    {
        float first_read[3] = {0};
        float second_read[3] = {0};
        size_t required = 0;
        CHECK(vx_result_read(retained, "y", NULL, 0, &required, &report) == VX_STATUS_OK);
        CHECK(required == sizeof(first_read));
        CHECK(vx_result_read(retained, "y", first_read, sizeof(first_read), NULL, &report) ==
              VX_STATUS_OK);
        first_read[0] = -999.0f;
        CHECK(vx_result_read(retained, "y", second_read, sizeof(second_read), NULL, &report) ==
              VX_STATUS_OK);
        CHECK(closef(second_read[0], 3.0f));
    }
    vx_result_release(retained);

    CHECK(test_model_source_snapshots() == 0);
    CHECK(test_declared_output_descriptors() == 0);
    CHECK(test_symbolic_target_shape_params() == 0);
    CHECK(test_qbatch_context_workspace_resource_proof() == 0);
    CHECK(test_native_gpu_resource_peak_projection() == 0);
    CHECK(test_graph_bank_package_validation(provider_graph_path) == 0);
    CHECK(test_many_bank_residencies() == 0);
    CHECK(test_provider(provider_graph_path, provider_weights_path) == 0);
    vx_runtime_release(NULL);
    vx_model_release(NULL);
    vx_compiled_model_release(NULL);
    vx_execution_context_release(NULL);
    vx_result_release(NULL);
    remove(graph_a);
    remove(graph_b);
    remove(provider_graph_path);
    remove(provider_weights_path);
    remove(invalid_graph_path);
    puts("native public handle and provider API tests passed");
    return 0;
}
