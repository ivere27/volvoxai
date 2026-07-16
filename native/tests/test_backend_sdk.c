#include "safetensors.h"
#include "volvoxai.h"
#include "volvoxai_backend.h"
#include "android_nnapi_backend.h"
#include "host_backend.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

/* The CPU-only test profile intentionally omits the device shader store. */
void volvoxai_shader_store_shutdown(void) {}

typedef struct {
    int init_count;
    int supports_count;
    int run_count;
    int reset_count;
    int begin_count;
    int end_count;
    int teardown_count;
    int callback_error;
} MockState;

typedef struct {
    MockState calls;
    void* pending_host;
    unsigned char pending[sizeof(float) * 2];
    size_t pending_bytes;
    int fail_sync;
} DeviceState;

typedef struct {
    MockState calls;
    void* first_output;
    void* second_output;
    unsigned marked_outputs;
} MultiOutputState;

static int closef(float left, float right) {
    return fabsf(left - right) <= 1.0e-6f;
}

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    size_t length = strlen(text);
    int ok = fwrite(text, 1, length, file) == length && fclose(file) == 0;
    return ok ? 0 : -1;
}

static int write_weight(const char* path) {
    static const int shape[2] = {2, 2};
    static const int scale_shape[1] = {2};
    static const int conv_shape[4] = {1, 1, 1, 1};
    static const int8_t weight[4] = {1, 2, 3, 4};
    static const float companion_scales[2] = {0.125f, 0.75f};
    static const float linear_weight[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    static const float conv_weight[1] = {1.0f};
    SafetensorsFile file;
    if (safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0) return -1;
    if (safetensors_add_tensor(&file, "vendor_weight", SAFETENSORS_DTYPE_I8,
                               shape, 2, weight, sizeof(weight)) != 0) {
        safetensors_free(&file);
        return -1;
    }
    if (safetensors_add_tensor(&file, "vendor_weight_scale",
                               SAFETENSORS_DTYPE_F32, scale_shape, 1,
                               companion_scales, sizeof(companion_scales)) != 0) {
        safetensors_free(&file);
        return -1;
    }
    if (safetensors_add_tensor(&file, "linear_weight", SAFETENSORS_DTYPE_F32,
                               shape, 2, linear_weight, sizeof(linear_weight)) != 0) {
        safetensors_free(&file);
        return -1;
    }
    if (safetensors_add_tensor(&file, "conv_weight", SAFETENSORS_DTYPE_F32,
                               conv_shape, 4, conv_weight, sizeof(conv_weight)) != 0) {
        safetensors_free(&file);
        return -1;
    }
    if (safetensors_set_metadata_json(
            &file,
            "{\"weights_quantization_storage\":"
            "\"volvoxai-f32-companion-scales-v1\","
            "\"weights_quantization\":"
            "\"{\\\"vendor_weight\\\":{\\\"axis\\\":0,"
            "\\\"scheme\\\":\\\"per_axis\\\"}}\"}") != 0) {
        safetensors_free(&file);
        return -1;
    }
    int result = safetensors_save(path, &file);
    safetensors_free(&file);
    return result;
}

static int host_init(void* user_data) {
    MockState* state = (MockState*)user_data;
    state->init_count++;
    return VX_INIT_READY;
}

static int host_supports(void* user_data, const VxNode* node) {
    MockState* state = (MockState*)user_data;
    state->supports_count++;
    return vx_node_op(node) && !strcmp(vx_node_op(node), "VendorScale")
        ? VX_HANDLED : VX_DECLINED;
}

static int host_run(void* user_data, const VxNode* node) {
    MockState* state = (MockState*)user_data;
    const VxTensor* input = vx_node_input_by_key(node, "input");
    const VxTensor* weight = vx_node_input_by_key(node, "weight");
    VxTensor* output = vx_node_output_by_key(node, "out");
    int32_t axis = -1;
    int32_t channels = 0;
    int32_t zero_point = -1;
    int32_t axes[3] = {0};
    float coefficients[2] = {0};
    float scale = 0.0f;
    const char* mode = vx_node_attr_string(node, "mode", "");
    const int binary_scales = !strcmp(mode, "binary");
    state->run_count++;
    if (!input || !weight || !output || vx_node_input_count(node) != 2 ||
        vx_node_output_count(node) != 1 || strcmp(vx_node_input_key(node, 0), "input") ||
        strcmp(vx_node_input_key(node, 1), "weight") ||
        strcmp(vx_node_output_key(node, 0), "out") || vx_node_input(node, 0) != input ||
        vx_node_output(node, 0) != output || vx_tensor_dtype(input) != VX_DTYPE_F32 ||
        vx_tensor_dtype(weight) != VX_DTYPE_I8 || vx_tensor_ndim(weight) != 2 ||
        !vx_tensor_shape(weight) || vx_tensor_shape(weight)[0] != 2 ||
        vx_tensor_shape(weight)[1] != 2 || vx_tensor_numel(weight) != 4 ||
        vx_tensor_element_size(weight) != 1 || vx_tensor_is_graph_input(weight) ||
        !vx_tensor_is_weight(weight) || vx_tensor_quantization_kind(weight) != VX_QUANT_PER_AXIS ||
        !vx_tensor_quant_axis(weight, &axis, &channels) || axis != 0 || channels != 2 ||
        !vx_tensor_quant_axis_value(weight, 0, &scale, &zero_point) ||
        !closef(scale, binary_scales ? 0.125f : 0.5f) || zero_point != 0 ||
        !vx_tensor_quant_axis_value(weight, 1, &scale, &zero_point) ||
        !closef(scale, binary_scales ? 0.75f : 0.25f) ||
        zero_point != (binary_scales ? 0 : 1) ||
        (!binary_scales && strcmp(mode, "host")) ||
        vx_node_attr_int(node, "version", -1) != 3 ||
        vx_node_attr_int(node, "fraction", 9) != 9 ||
        !closef(vx_node_attr_float(node, "gain", -1.0f), 2.0f) ||
        !vx_node_attr_bool(node, "enabled", 0) ||
        vx_node_attr_ints(node, "axes", axes, 3) != 2 || axes[0] != 0 || axes[1] != 1 ||
        vx_node_attr_ints(node, "axes", NULL, 0) != 2 ||
        vx_node_attr_ints(node, "bad_axes", axes, 3) != -1 ||
        vx_node_attr_ints(node, "long_axes", axes, 2) != -1 ||
        vx_node_attr_floats(node, "coefficients", coefficients, 2) != 2 ||
        vx_node_attr_floats(node, "bad_coefficients", coefficients, 2) != -1 ||
        !closef(coefficients[0], 0.25f) || !closef(coefficients[1], 0.75f) ||
        vx_tensor_sync_host(input) != 0 || !vx_tensor_cdata(input) ||
        !vx_tensor_data(output)) {
        state->callback_error = 1;
        return VX_ERROR;
    }
    const float* source = (const float*)vx_tensor_cdata(input);
    float* destination = (float*)vx_tensor_data(output);
    for (int64_t index = 0; index < vx_tensor_numel(input); index++)
        destination[index] = source[index] * 2.0f;
    return VX_HANDLED;
}

static void host_reset(void* user_data) { ((MockState*)user_data)->reset_count++; }
static void host_begin(void* user_data) { ((MockState*)user_data)->begin_count++; }
static int host_end(void* user_data) { ((MockState*)user_data)->end_count++; return 0; }
static void host_teardown(void* user_data) { ((MockState*)user_data)->teardown_count++; }

static int unavailable_init(void* user_data) {
    ((MockState*)user_data)->init_count++;
    return VX_INIT_UNAVAILABLE;
}

static int never_supports(void* user_data, const VxNode* node) {
    (void)user_data;
    (void)node;
    return VX_DECLINED;
}

static int never_run(void* user_data, const VxNode* node) {
    (void)user_data;
    (void)node;
    return VX_DECLINED;
}

static int fused_producer_probe_supports(void* user_data, const VxNode* node) {
    MockState* state = (MockState*)user_data;
    const char* op = vx_node_op(node);
    if (op && (!strcmp(op, "Conv2D") || !strcmp(op, "Concat"))) {
        state->supports_count++;
        return VX_ERROR; /* A transformed producer must never reach V1. */
    }
    return VX_DECLINED;
}

static void unavailable_teardown(void* user_data) {
    ((MockState*)user_data)->teardown_count++;
}

static int nnapi_probe_supports(void* user_data, const VxNode* node) {
    MockState* state = (MockState*)user_data;
    state->supports_count++;
    return volvoxai_example_nnapi_add_supports(node);
}

static int nnapi_probe_run(void* user_data, const VxNode* node) {
    MockState* state = (MockState*)user_data;
    const VxTensor* left = vx_node_input_by_key(node, "a");
    const VxTensor* right = vx_node_input_by_key(node, "b");
    VxTensor* output = vx_node_output_by_key(node, "out");
    if (volvoxai_example_nnapi_add_supports(node) != VX_HANDLED ||
        !left || !right || !output || vx_tensor_sync_host(left) != 0 ||
        vx_tensor_sync_host(right) != 0 || !vx_tensor_cdata(left) ||
        !vx_tensor_cdata(right) || !vx_tensor_data(output)) {
        state->callback_error = 1;
        return VX_ERROR;
    }
    const float* left_data = (const float*)vx_tensor_cdata(left);
    const float* right_data = (const float*)vx_tensor_cdata(right);
    float* output_data = (float*)vx_tensor_data(output);
    for (int64_t index = 0; index < vx_tensor_numel(output); index++)
        output_data[index] = left_data[index] + right_data[index];
    state->run_count++;
    return VX_HANDLED;
}

static int device_run(void* user_data, const VxNode* node) {
    DeviceState* state = (DeviceState*)user_data;
    const VxTensor* input = vx_node_input_by_key(node, "input");
    VxTensor* output = vx_node_output_by_key(node, "out");
    if (!input || !output || vx_tensor_numel(input) != 2 ||
        vx_tensor_numel(output) != 2 || !vx_tensor_cdata(input) ||
        !vx_tensor_data(output)) return VX_ERROR;
    const float* source = (const float*)vx_tensor_cdata(input);
    if (vx_tensor_dtype(output) == VX_DTYPE_F32) {
        float values[2] = {source[0] * 2.0f, source[1] * 2.0f};
        memcpy(state->pending, values, sizeof(values));
        state->pending_bytes = sizeof(values);
    } else if (vx_tensor_dtype(output) == VX_DTYPE_F16 &&
               closef(source[0], 1.0f) && closef(source[1], 2.0f)) {
        /* Exact IEEE-754 binary16 encodings for 2.0 and 4.0. */
        const uint16_t values[2] = {UINT16_C(0x4000), UINT16_C(0x4400)};
        memcpy(state->pending, values, sizeof(values));
        state->pending_bytes = sizeof(values);
    } else {
        return VX_ERROR;
    }
    state->pending_host = vx_tensor_data(output);
    state->calls.run_count++;
    return VX_HANDLED;
}

static void device_mark(void* user_data, const void* host, size_t bytes, int is_weight) {
    DeviceState* state = (DeviceState*)user_data;
    (void)bytes;
    (void)is_weight;
    if (host == state->pending_host) state->pending_host = NULL;
}

static int device_sync(void* user_data, void* host, size_t bytes, int is_weight) {
    DeviceState* state = (DeviceState*)user_data;
    (void)is_weight;
    if (host == state->pending_host) {
        if (state->fail_sync) return -1;
        if (bytes != state->pending_bytes) return -1;
        memcpy(host, state->pending, state->pending_bytes);
    }
    return 0; /* Public SDK convention: zero is success. */
}

static int multi_output_supports(void* user_data, const VxNode* node) {
    MultiOutputState* state = (MultiOutputState*)user_data;
    if (vx_node_op(node) && !strcmp(vx_node_op(node), "Split")) {
        state->calls.supports_count++;
        return VX_HANDLED;
    }
    return VX_DECLINED;
}

static int multi_output_run(void* user_data, const VxNode* node) {
    MultiOutputState* state = (MultiOutputState*)user_data;
    const VxTensor* input = vx_node_input_by_key(node, "input");
    VxTensor* first = vx_node_output(node, 0);
    VxTensor* second = vx_node_output(node, 1);
    if (!input || !first || !second || vx_tensor_dtype(input) != VX_DTYPE_F32 ||
        vx_tensor_dtype(first) != VX_DTYPE_F32 || vx_tensor_dtype(second) != VX_DTYPE_F32 ||
        vx_tensor_numel(input) != 4 || vx_tensor_numel(first) != 2 ||
        vx_tensor_numel(second) != 2 || vx_tensor_sync_host(input) != 0 ||
        !vx_tensor_cdata(input) || !vx_tensor_data(first) || !vx_tensor_data(second))
        return VX_ERROR;
    const float* source = (const float*)vx_tensor_cdata(input);
    memcpy(vx_tensor_data(first), source, 2 * sizeof(float));
    memcpy(vx_tensor_data(second), source + 2, 2 * sizeof(float));
    state->first_output = vx_tensor_data(first);
    state->second_output = vx_tensor_data(second);
    state->marked_outputs = 0;
    state->calls.run_count++;
    return VX_HANDLED;
}

static void multi_output_mark(void* user_data, const void* host,
                              size_t bytes, int is_weight) {
    MultiOutputState* state = (MultiOutputState*)user_data;
    (void)bytes;
    (void)is_weight;
    if (host == state->first_output) state->marked_outputs |= 1u;
    if (host == state->second_output) state->marked_outputs |= 2u;
}

static int multi_output_sync(void* user_data, void* host,
                             size_t bytes, int is_weight) {
    (void)user_data;
    (void)host;
    (void)bytes;
    (void)is_weight;
    return 0;
}

int main(void) {
    static const char* const add_config =
        "{\"inputs\":{\"a\":{\"shape\":[2]},\"b\":{\"shape\":[2]}},\"nodes\":[{"
        "\"op\":\"Add\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"sum\"},\"outputs_shape\":{\"out\":[2]}}],"
        "\"outputs\":[\"sum\"]}";
    static const char* const broadcast_add_config =
        "{\"inputs\":{\"a\":{\"shape\":[2]},\"b\":{\"shape\":[1]}},\"nodes\":[{"
        "\"op\":\"Add\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"sum\"},\"outputs_shape\":{\"out\":[2]}}],"
        "\"outputs\":[\"sum\"]}";
    static const char* const fused_add_config =
        "{\"inputs\":{\"a\":{\"shape\":[2]},\"b\":{\"shape\":[2]}},\"nodes\":[{"
        "\"op\":\"Add\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"sum\"},\"outputs_shape\":{\"out\":[2]},"
        "\"params\":{\"relu\":true}}],\"outputs\":[\"sum\"]}";
    static const char* const vendor_config =
        "{\"inputs\":{\"x\":{\"shape\":[2]},\"b\":{\"shape\":[2]}},"
        "\"weights_quantization\":{\"vendor_weight\":{\"scheme\":\"per_axis\","
        "\"axis\":0,\"scales\":[0.5,0.25],\"zero_points\":[0,1]}},\"nodes\":[{"
        "\"op\":\"VendorScale\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"vendor_weight\"},\"outputs\":{\"out\":\"scaled\"},"
        "\"outputs_shape\":{\"out\":[2]},\"params\":{\"mode\":\"host\","
        "\"version\":3,\"fraction\":1.5,\"gain\":2.0,\"enabled\":true,\"axes\":[0,1],"
        "\"coefficients\":[0.25,0.75],\"bad_axes\":[0,\"bad\",1],"
        "\"long_axes\":[0,1,2],\"bad_coefficients\":[0.25,\"bad\"]}},{\"op\":\"Add\","
        "\"inputs\":{\"a\":\"scaled\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"result\"},\"outputs_shape\":{\"out\":[2]}}],"
        "\"outputs\":[\"result\"]}";
    static const char* const vendor_binary_config =
        "{\"inputs\":{\"x\":{\"shape\":[2]}},"
        "\"weights_quantization_storage\":{\"format\":"
        "\"volvoxai-f32-companion-scales-v1\"},"
        "\"nodes\":[{"
        "\"op\":\"VendorScale\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"vendor_weight\"},\"outputs\":{\"out\":\"result\"},"
        "\"outputs_shape\":{\"out\":[2]},\"params\":{\"mode\":\"binary\","
        "\"version\":3,\"fraction\":1.5,\"gain\":2.0,\"enabled\":true,"
        "\"axes\":[0,1],\"coefficients\":[0.25,0.75],"
        "\"bad_axes\":[0,\"bad\",1],\"long_axes\":[0,1,2],"
        "\"bad_coefficients\":[0.25,\"bad\"]}}],\"outputs\":[\"result\"]}";
    static const char* const vendor_f16_config =
        "{\"inputs\":{\"x\":{\"shape\":[2]}},\"nodes\":[{"
        "\"op\":\"VendorScale\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"scaled_half\"},\"outputs_shape\":{\"out\":[2]},"
        "\"outputs_dtype\":{\"out\":\"float16\"},\"params\":{}},{"
        "\"op\":\"Linear\",\"inputs\":{\"input\":\"scaled_half\","
        "\"weight\":\"linear_weight\"},"
        "\"outputs\":{\"out\":\"result\"},\"outputs_shape\":{\"out\":[2]},"
        "\"outputs_dtype\":{\"out\":\"float32\"},"
        "\"params\":{\"weight_layout\":\"OUT_IN\"}}],\"outputs\":[\"result\"]}";
    static const char* const fused_relu6_config =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,1]}},\"nodes\":[{"
        "\"op\":\"Conv2D\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"conv_weight\"},\"outputs\":{\"out\":\"conv\"},"
        "\"outputs_shape\":{\"out\":[1,1,1,1]},"
        "\"params\":{\"weight_layout\":\"OHWI\",\"stride\":[1,1],"
        "\"padding\":[0,0],\"dilation\":[1,1],\"groups\":1}},{"
        "\"op\":\"Clip\",\"inputs\":{\"input\":\"conv\"},"
        "\"outputs\":{\"out\":\"result\"},\"outputs_shape\":{\"out\":[1,1,1,1]},"
        "\"params\":{\"min\":0,\"max\":6}}],\"outputs\":[\"result\"]}";
    static const char* const fused_concat_config =
        "{\"inputs\":{\"a\":{\"shape\":[1,1]},\"b\":{\"shape\":[1,1]}},"
        "\"nodes\":[{\"op\":\"Concat\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"joined\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"params\":{\"axis\":1}},{\"op\":\"Sigmoid\","
        "\"inputs\":{\"input\":\"joined\"},\"outputs\":{\"out\":\"result\"},"
        "\"outputs_shape\":{\"out\":[1,2]},\"params\":{}}],"
        "\"outputs\":[\"result\"]}";
    static const char* const vendor_only_config =
        "{\"inputs\":{\"x\":{\"shape\":[2]}},\"nodes\":[{"
        "\"op\":\"VendorScale\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"result\"},\"outputs_shape\":{\"out\":[2]},"
        "\"params\":{}}],\"outputs\":[\"result\"]}";
    static const char* const multi_output_config =
        "{\"inputs\":{\"x\":{\"shape\":[4]}},\"nodes\":[{"
        "\"op\":\"Split\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"first\":\"left\",\"second\":\"right\"},"
        "\"outputs_shape\":{\"first\":[2],\"second\":[2]},"
        "\"params\":{\"axis\":0}}],\"outputs\":[\"left\",\"right\"]}";
    const char* add_path = "/tmp/volvox-backend-sdk-add.json";
    const char* broadcast_add_path = "/tmp/volvox-backend-sdk-add-broadcast.json";
    const char* fused_add_path = "/tmp/volvox-backend-sdk-add-fused.json";
    const char* vendor_path = "/tmp/volvox-backend-sdk-vendor.json";
    const char* vendor_binary_path = "/tmp/volvox-backend-sdk-vendor-binary.json";
    const char* vendor_f16_path = "/tmp/volvox-backend-sdk-vendor-f16.json";
    const char* fused_relu6_path = "/tmp/volvox-backend-sdk-fused-relu6.json";
    const char* fused_concat_path = "/tmp/volvox-backend-sdk-fused-concat.json";
    const char* vendor_only_path = "/tmp/volvox-backend-sdk-vendor-only.json";
    const char* multi_output_path = "/tmp/volvox-backend-sdk-multi-output.json";
    const char* weight_path = "/tmp/volvox-backend-sdk.safetensors";
    const float a[2] = {1.0f, 2.0f};
    const float b[2] = {3.0f, 4.0f};
    float result[2] = {0};
    MockState host_state = {0};
    MockState unavailable_state = {0};
    MockState nnapi_probe_state = {0};
    MockState fused_probe_state = {0};
    DeviceState device_state = {0};
    MultiOutputState multi_output_state = {0};
    VxBackendV1 host = {
        .struct_size = sizeof(VxBackendV1),
        .abi_version = VX_BACKEND_ABI_V1,
        .name = "example-npu",
        .user_data = &host_state,
        .init = host_init,
        .supports = host_supports,
        .run = host_run,
        .teardown = host_teardown,
        .reset = host_reset,
        .begin_forward = host_begin,
        .end_forward = host_end,
    };
    VxBackendV1 unavailable = {
        .struct_size = sizeof(VxBackendV1),
        .abi_version = VX_BACKEND_ABI_V1,
        .name = "unavailable-npu",
        .user_data = &unavailable_state,
        .init = unavailable_init,
        .supports = never_supports,
        .run = never_run,
        .teardown = unavailable_teardown,
    };
    VxBackendV1 device = {
        .struct_size = sizeof(VxBackendV1),
        .abi_version = VX_BACKEND_ABI_V1,
        .name = "device-npu",
        .user_data = &device_state,
        .flags = VX_BACKEND_FLAG_DEVICE_RESIDENT_OUTPUTS,
        .init = host_init,
        .supports = host_supports,
        .run = device_run,
        .teardown = host_teardown,
        .begin_forward = host_begin,
        .end_forward = host_end,
        .mark_host = device_mark,
        .sync_host = device_sync,
    };
    VxBackendV1 nnapi_probe = {
        .struct_size = sizeof(VxBackendV1),
        .abi_version = VX_BACKEND_ABI_V1,
        .name = "nnapi-add-probe",
        .user_data = &nnapi_probe_state,
        .init = host_init,
        .supports = nnapi_probe_supports,
        .run = nnapi_probe_run,
        .teardown = host_teardown,
    };
    VxBackendV1 fused_probe = {
        .struct_size = sizeof(VxBackendV1),
        .abi_version = VX_BACKEND_ABI_V1,
        .name = "fused-producer-probe",
        .user_data = &fused_probe_state,
        .init = host_init,
        .supports = fused_producer_probe_supports,
        .run = never_run,
        .teardown = host_teardown,
    };
    VxBackendV1 multi_output = {
        .struct_size = sizeof(VxBackendV1),
        .abi_version = VX_BACKEND_ABI_V1,
        .name = "multi-output-host",
        .user_data = &multi_output_state,
        .init = host_init,
        .supports = multi_output_supports,
        .run = multi_output_run,
        .teardown = host_teardown,
        .mark_host = multi_output_mark,
        .sync_host = multi_output_sync,
    };
    VxBackendV1 invalid = host;

    CHECK(write_text(add_path, add_config) == 0);
    CHECK(write_text(broadcast_add_path, broadcast_add_config) == 0);
    CHECK(write_text(fused_add_path, fused_add_config) == 0);
    CHECK(write_text(vendor_path, vendor_config) == 0);
    CHECK(write_text(vendor_binary_path, vendor_binary_config) == 0);
    CHECK(write_text(vendor_f16_path, vendor_f16_config) == 0);
    CHECK(write_text(fused_relu6_path, fused_relu6_config) == 0);
    CHECK(write_text(fused_concat_path, fused_concat_config) == 0);
    CHECK(write_text(vendor_only_path, vendor_only_config) == 0);
    CHECK(write_text(multi_output_path, multi_output_config) == 0);
    CHECK(write_weight(weight_path) == 0);
    invalid.struct_size = (uint32_t)(sizeof(VxBackendV1) - 1u);
    CHECK(volvoxai_register_backend(&invalid) != 0);
    invalid = host;
    invalid.name = "nnapi"; /* Legacy built-in names cannot be shadowed. */
    CHECK(volvoxai_register_backend(&invalid) != 0);
    CHECK(volvoxai_register_backend(&host) == 0);
    CHECK(volvoxai_register_backend(&host) != 0);
    CHECK(volvoxai_register_backend(&unavailable) == 0);
    CHECK(volvoxai_register_backend(&device) == 0);
    CHECK(volvoxai_example_nnapi_backend_register() == 0);
    CHECK(volvoxai_register_backend(&nnapi_probe) == 0);
    CHECK(volvoxai_register_backend(&fused_probe) == 0);
    CHECK(volvoxai_register_backend(&multi_output) == 0);
    CHECK(volvoxai_example_host_backend_register() == 0);

    /* The public-header-only Android example registers on every platform but
     * is explicitly unavailable on a host build.  Failed selection preserves
     * the legacy CPU policy. */
    CHECK(volvoxai_engine_configure_backend("android-nnapi-add") != 0);
    CHECK(!strcmp(volvoxai_engine_backend_name(), "CPU"));

    CHECK(volvoxai_engine_configure_backend("example-host") == 0);
    CHECK(volvoxai_engine_init(add_path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw("a", VOLVOXAI_DTYPE_F32, a, sizeof(a)) == 0);
    CHECK(volvoxai_engine_set_input_raw("b", VOLVOXAI_DTYPE_F32, b, sizeof(b)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_example_host_backend_handled_nodes() == 1);
    CHECK(volvoxai_engine_copy_tensor_f32("sum", result, 2) == 0);
    CHECK(closef(result[0], 4.0f) && closef(result[1], 6.0f));
    volvoxai_engine_shutdown();

    /* Operator fusion owns redirected outputs and private epilogues that are
     * outside ABI v1.  Default ReLU6 and Concat+Sigmoid producers therefore
     * bypass public callbacks and retain their CPU-correct graph outputs. */
    CHECK(volvoxai_engine_configure_backend("fused-producer-probe") == 0);
    CHECK(volvoxai_engine_init(fused_relu6_path, weight_path) == 0);
    {
        const float input = 10.0f;
        float output = 0.0f;
        CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                            &input, sizeof(input)) == 0);
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(volvoxai_engine_copy_tensor_f32("result", &output, 1) == 0);
        CHECK(closef(output, 6.0f));
    }
    CHECK(fused_probe_state.supports_count == 0);
    volvoxai_engine_shutdown();

    CHECK(volvoxai_engine_configure_backend("fused-producer-probe") == 0);
    CHECK(volvoxai_engine_init(fused_concat_path, NULL) == 0);
    {
        const float zero = 0.0f;
        CHECK(volvoxai_engine_set_input_raw("a", VOLVOXAI_DTYPE_F32,
                                            &zero, sizeof(zero)) == 0);
        CHECK(volvoxai_engine_set_input_raw("b", VOLVOXAI_DTYPE_F32,
                                            &zero, sizeof(zero)) == 0);
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(volvoxai_engine_copy_tensor_f32("result", result, 2) == 0);
        CHECK(fabsf(result[0] - 0.5f) < 0.02f &&
              fabsf(result[1] - 0.5f) < 0.02f);
    }
    CHECK(fused_probe_state.supports_count == 0);
    volvoxai_engine_shutdown();

    /* Registration is inert: an explicit CPU policy never invokes the SDK. */
    CHECK(volvoxai_engine_configure_backend("cpu") == 0);
    CHECK(volvoxai_engine_init(add_path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw("a", VOLVOXAI_DTYPE_F32, a, sizeof(a)) == 0);
    CHECK(volvoxai_engine_set_input_raw("b", VOLVOXAI_DTYPE_F32, b, sizeof(b)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("sum", result, 2) == 0);
    CHECK(closef(result[0], 4.0f) && closef(result[1], 6.0f));
    CHECK(host_state.init_count == 0 && host_state.supports_count == 0 && host_state.run_count == 0);
    volvoxai_engine_shutdown();

    /* Explicit unavailability is a configuration error and keeps CPU active. */
    CHECK(volvoxai_engine_configure_backend("unavailable-npu") != 0);
    CHECK(unavailable_state.init_count == 1 && unavailable_state.teardown_count == 1);
    CHECK(!strcmp(volvoxai_engine_backend_name(), "CPU"));

    CHECK(volvoxai_engine_configure_backend("example-npu") == 0);
    CHECK(host_state.init_count == 1);
    CHECK(!strcmp(volvoxai_engine_backend_name(), "example-npu"));
    CHECK(volvoxai_engine_configure_backend("example-npu") == 0);
    CHECK(host_state.init_count == 1 && host_state.teardown_count == 0);
    CHECK(volvoxai_engine_configure_backend("unavailable-npu") != 0);
    CHECK(unavailable_state.init_count == 2 && unavailable_state.teardown_count == 2);
    CHECK(host_state.init_count == 1 && host_state.teardown_count == 0);
    CHECK(!strcmp(volvoxai_engine_backend_name(), "example-npu"));
    VolvoxAIEngineOptions options = {0};
    CHECK(volvoxai_engine_get_options(&options) == 0 &&
          options.backend == VOLVOXAI_BACKEND_CUSTOM);
    CHECK(volvoxai_engine_init(vendor_path, weight_path) == 0);
    CHECK(host_state.reset_count == 1);
    CHECK(volvoxai_engine_configure_backend("cpu") != 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, a, sizeof(a)) == 0);
    CHECK(volvoxai_engine_set_input_raw("b", VOLVOXAI_DTYPE_F32, b, sizeof(b)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(!host_state.callback_error && host_state.run_count == 1);
    CHECK(host_state.supports_count == 2); /* VendorScale handled, Add declined. */
    CHECK(host_state.begin_count == 1 && host_state.end_count == 1);
    CHECK(volvoxai_engine_copy_tensor_f32("result", result, 2) == 0);
    CHECK(closef(result[0], 5.0f) && closef(result[1], 8.0f));
    volvoxai_engine_shutdown();

    /* A device readback failure is never treated as stale-but-successful host
     * data.  It fails both direct copies and CPU fallback, while end_forward
     * still balances every begun lifecycle before the next attempt. */
    CHECK(volvoxai_engine_configure_backend("device-npu") == 0);
    CHECK(volvoxai_engine_init(vendor_only_path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, a, sizeof(a)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    device_state.fail_sync = 1;
    CHECK(volvoxai_engine_copy_tensor_f32("result", result, 2) != 0);
    device_state.fail_sync = 0;
    CHECK(volvoxai_engine_copy_tensor_f32("result", result, 2) == 0);
    CHECK(closef(result[0], 2.0f) && closef(result[1], 4.0f));
    volvoxai_engine_shutdown();

    CHECK(volvoxai_engine_configure_backend("device-npu") == 0);
    CHECK(volvoxai_engine_init(vendor_path, weight_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, a, sizeof(a)) == 0);
    CHECK(volvoxai_engine_set_input_raw("b", VOLVOXAI_DTYPE_F32, b, sizeof(b)) == 0);
    {
        int begins = device_state.calls.begin_count;
        int ends = device_state.calls.end_count;
        device_state.fail_sync = 1;
        CHECK(volvoxai_engine_forward() != 0);
        CHECK(device_state.calls.begin_count == begins + 1 &&
              device_state.calls.end_count == ends + 1);
        device_state.fail_sync = 0;
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(device_state.calls.begin_count == begins + 2 &&
              device_state.calls.end_count == ends + 2);
    }
    CHECK(volvoxai_engine_copy_tensor_f32("result", result, 2) == 0);
    CHECK(closef(result[0], 5.0f) && closef(result[1], 8.0f));
    volvoxai_engine_shutdown();

    /* Host SDK callbacks may author every declared output.  The runtime must
     * mark secondary outputs host-current too, not only Node.out. */
    CHECK(volvoxai_engine_configure_backend("multi-output-host") == 0);
    CHECK(volvoxai_engine_init(multi_output_path, NULL) == 0);
    {
        const float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float left[2] = {0};
        float right[2] = {0};
        CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                            input, sizeof(input)) == 0);
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(multi_output_state.calls.run_count == 1 &&
              multi_output_state.marked_outputs == 3u);
        CHECK(volvoxai_engine_copy_tensor_f32("left", left, 2) == 0);
        CHECK(volvoxai_engine_copy_tensor_f32("right", right, 2) == 0);
        CHECK(closef(left[0], 1.0f) && closef(left[1], 2.0f) &&
              closef(right[0], 3.0f) && closef(right[1], 4.0f));
    }
    volvoxai_engine_shutdown();

    /* Public device storage covers every SDK dtype.  In particular, a raw
     * F16 output must synchronize before CPU Linear widens it to F32. */
    CHECK(volvoxai_engine_configure_backend("device-npu") == 0);
    CHECK(volvoxai_engine_init(vendor_f16_path, weight_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, a, sizeof(a)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(device_state.calls.run_count == 4);
    void* retained_half_key = device_state.pending_host;
    CHECK(volvoxai_engine_copy_tensor_f32("result", result, 2) == 0);
    CHECK(closef(result[0], 2.0f) && closef(result[1], 4.0f));
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(device_state.calls.run_count == 5 &&
          device_state.pending_host != retained_half_key);
    CHECK(volvoxai_engine_copy_tensor_f32("result", result, 2) == 0);
    CHECK(closef(result[0], 2.0f) && closef(result[1], 4.0f));
    volvoxai_engine_shutdown();
    CHECK(host_state.teardown_count == 1);
    CHECK(!strcmp(volvoxai_engine_backend_name(), "CPU"));

    /* Device output remains off-host until the following CPU fallback asks the
     * SDK sync hook to materialize it.  This also fixes the public zero-success
     * convention against the private registry's truthy-success convention. */
    CHECK(volvoxai_engine_configure_backend("device-npu") == 0);
    CHECK(volvoxai_engine_init(vendor_path, weight_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, a, sizeof(a)) == 0);
    CHECK(volvoxai_engine_set_input_raw("b", VOLVOXAI_DTYPE_F32, b, sizeof(b)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(device_state.calls.run_count == 6);
    CHECK(volvoxai_engine_copy_tensor_f32("result", result, 2) == 0);
    CHECK(closef(result[0], 5.0f) && closef(result[1], 8.0f));
    volvoxai_engine_shutdown();

    /* Exercise the Android example's public eligibility contract with a host
     * implementation: exact F32 Add is handled, while broadcasting and fused
     * activation are conservatively declined to CPU. */
    CHECK(volvoxai_engine_configure_backend("nnapi-add-probe") == 0);
    CHECK(volvoxai_engine_init(add_path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw("a", VOLVOXAI_DTYPE_F32, a, sizeof(a)) == 0);
    CHECK(volvoxai_engine_set_input_raw("b", VOLVOXAI_DTYPE_F32, b, sizeof(b)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(!nnapi_probe_state.callback_error && nnapi_probe_state.supports_count == 1 &&
          nnapi_probe_state.run_count == 1);
    CHECK(volvoxai_engine_copy_tensor_f32("sum", result, 2) == 0);
    CHECK(closef(result[0], 4.0f) && closef(result[1], 6.0f));
    CHECK(volvoxai_engine_set_execution_row(0) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(nnapi_probe_state.supports_count == 1 && nnapi_probe_state.run_count == 1);
    CHECK(volvoxai_engine_copy_tensor_f32("sum", result, 2) == 0);
    CHECK(closef(result[0], 4.0f) && closef(result[1], 6.0f));
    CHECK(volvoxai_engine_set_execution_row(-1) == 0);
    volvoxai_engine_shutdown();
    CHECK(nnapi_probe_state.teardown_count == 1);

    const float scalar = 10.0f;
    CHECK(volvoxai_engine_configure_backend("nnapi-add-probe") == 0);
    CHECK(volvoxai_engine_init(broadcast_add_path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw("a", VOLVOXAI_DTYPE_F32, a, sizeof(a)) == 0);
    CHECK(volvoxai_engine_set_input_raw("b", VOLVOXAI_DTYPE_F32,
                                        &scalar, sizeof(scalar)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(nnapi_probe_state.supports_count == 2 && nnapi_probe_state.run_count == 1);
    CHECK(volvoxai_engine_copy_tensor_f32("sum", result, 2) == 0);
    CHECK(closef(result[0], 11.0f) && closef(result[1], 12.0f));
    volvoxai_engine_shutdown();
    CHECK(nnapi_probe_state.teardown_count == 2);

    CHECK(volvoxai_engine_configure_backend("nnapi-add-probe") == 0);
    CHECK(volvoxai_engine_init(fused_add_path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw("a", VOLVOXAI_DTYPE_F32, a, sizeof(a)) == 0);
    CHECK(volvoxai_engine_set_input_raw("b", VOLVOXAI_DTYPE_F32, b, sizeof(b)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(nnapi_probe_state.supports_count == 3 && nnapi_probe_state.run_count == 1);
    CHECK(volvoxai_engine_copy_tensor_f32("sum", result, 2) == 0);
    CHECK(closef(result[0], 4.0f) && closef(result[1], 6.0f));
    volvoxai_engine_shutdown();
    CHECK(nnapi_probe_state.teardown_count == 3);

    /* Public backends receive the same exact binary companion scales as the
     * built-in native kernels, with symmetric zero points synthesized as 0. */
    CHECK(volvoxai_engine_configure_backend("example-npu") == 0);
    CHECK(volvoxai_engine_init(vendor_binary_path, weight_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, a, sizeof(a)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(!host_state.callback_error);
    CHECK(volvoxai_engine_copy_tensor_f32("result", result, 2) == 0);
    CHECK(closef(result[0], 2.0f) && closef(result[1], 4.0f));
    volvoxai_engine_shutdown();

    remove(add_path);
    remove(broadcast_add_path);
    remove(fused_add_path);
    remove(vendor_path);
    remove(vendor_binary_path);
    remove(vendor_f16_path);
    remove(fused_relu6_path);
    remove(fused_concat_path);
    remove(vendor_only_path);
    remove(multi_output_path);
    remove(weight_path);
    puts("public backend SDK tests passed");
    return 0;
}
