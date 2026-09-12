/* Receipt digit reader: native application code on VolvoxAI's generated C API.
 *
 * Preprocessing, manifest policy, and slot decoding live here. Runtime,
 * compilation, shape proof, and kernel dispatch stay behind generated handles.
 */

#include "receipt_digit_reader.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../../native/cli/call_client.h"
#include "../../../native/src/generated/proto_methods.h"
#include "volvoxai_lite.h"

#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static void set_error(char* error, size_t error_size, const char* message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
}

/* ITU-R BT.601 luma, matching Pillow's `convert("L")` rounding. */
static unsigned char luma_u8(unsigned int red, unsigned int green, unsigned int blue) {
    return (unsigned char)((red * 299u + green * 587u + blue * 114u + 500u) / 1000u);
}

int receipt_digit_preprocess(const unsigned char* pixels, int source_width,
                             int source_height, int channels, float* destination,
                             int target_width, int target_height) {
    unsigned char* plane = NULL;
    float scale_x;
    float scale_y;
    int row;
    if (!pixels || !destination || source_width <= 0 || source_height <= 0 ||
        target_width <= 0 || target_height <= 0 ||
        (channels != 1 && channels != 3 && channels != 4)) {
        return -1;
    }
    plane = (unsigned char*)malloc((size_t)source_width * (size_t)source_height);
    if (!plane) return -1;
    for (row = 0; row < source_width * source_height; row++) {
        const unsigned char* pixel = pixels + (size_t)row * (size_t)channels;
        plane[row] = channels == 1 ? pixel[0] : luma_u8(pixel[0], pixel[1], pixel[2]);
    }

    /* Half-pixel centers, matching Pillow. The corner convention instead
     * shifts the sampled grid by up to half a pixel, which visibly moves thin
     * digit strokes and costs accuracy on the narrow slots. */
    scale_x = (float)source_width / (float)target_width;
    scale_y = (float)source_height / (float)target_height;
    for (row = 0; row < target_height; row++) {
        float source_y = ((float)row + 0.5f) * scale_y - 0.5f;
        int top_index = (int)floorf(source_y);
        float weight_y = source_y - (float)top_index;
        int top = top_index < 0 ? 0 : (top_index >= source_height ? source_height - 1 : top_index);
        int bottom_index = top_index + 1;
        int bottom = bottom_index < 0 ? 0
                   : (bottom_index >= source_height ? source_height - 1 : bottom_index);
        int column;
        for (column = 0; column < target_width; column++) {
            float source_x = ((float)column + 0.5f) * scale_x - 0.5f;
            int left_index = (int)floorf(source_x);
            float weight_x = source_x - (float)left_index;
            int left = left_index < 0 ? 0
                     : (left_index >= source_width ? source_width - 1 : left_index);
            int right_index = left_index + 1;
            int right = right_index < 0 ? 0
                      : (right_index >= source_width ? source_width - 1 : right_index);
            float top_left = (float)plane[(size_t)top * (size_t)source_width + (size_t)left];
            float top_right = (float)plane[(size_t)top * (size_t)source_width + (size_t)right];
            float bottom_left = (float)plane[(size_t)bottom * (size_t)source_width + (size_t)left];
            float bottom_right = (float)plane[(size_t)bottom * (size_t)source_width + (size_t)right];
            float upper = top_left + (top_right - top_left) * weight_x;
            float lower = bottom_left + (bottom_right - bottom_left) * weight_x;
            float value = upper + (lower - upper) * weight_y;
            destination[(size_t)row * (size_t)target_width + (size_t)column] =
                (value / 255.0f - 0.5f) / 0.5f;
        }
    }
    free(plane);
    return 0;
}

int receipt_digit_decode(const float* logits, const ReceiptDigitLayout* layout,
                         ReceiptDigitRecord* record) {
    int slot;
    int phone_length = 0;
    int street_length = 0;
    int phone_done = 0;
    int street_done = 0;
    if (!logits || !layout || !record) return -1;
    if (layout->slots <= 0 || layout->slots > RECEIPT_DIGIT_MAX_SLOTS ||
        layout->num_classes <= 1 || layout->phone_slots < 0 ||
        layout->phone_slots > layout->slots ||
        layout->blank_class < 0 || layout->blank_class >= layout->num_classes) {
        return -1;
    }
    memset(record, 0, sizeof(*record));
    for (slot = 0; slot < layout->slots; slot++) {
        const float* row = logits + (size_t)slot * (size_t)layout->num_classes;
        int best = 0;
        int candidate;
        for (candidate = 1; candidate < layout->num_classes; candidate++) {
            if (row[candidate] > row[best]) best = candidate;
        }
        /* Stop at the first blank rather than dropping it: the slots are
         * left-aligned, so a blank means the number ended there and skipping
         * it would splice unrelated digits together. */
        if (slot < layout->phone_slots) {
            if (best == layout->blank_class) { phone_done = 1; continue; }
            if (!phone_done) record->phone[phone_length++] = (char)('0' + best);
        } else {
            if (best == layout->blank_class) { street_done = 1; continue; }
            if (!street_done) record->street[street_length++] = (char)('0' + best);
        }
    }
    record->phone[phone_length] = '\0';
    record->street[street_length] = '\0';
    return 0;
}

int receipt_digit_load_image(const char* path, float* destination, int target_width,
                             int target_height, char* error, size_t error_size) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* pixels;
    int status;
    if (!path || !destination) {
        set_error(error, error_size, "image path and destination are required");
        return -1;
    }
    pixels = stbi_load(path, &width, &height, &channels, 0);
    if (!pixels) {
        set_error(error, error_size, stbi_failure_reason());
        return -1;
    }
    status = receipt_digit_preprocess(pixels, width, height, channels, destination,
                                      target_width, target_height);
    stbi_image_free(pixels);
    if (status != 0) set_error(error, error_size, "unsupported image geometry or channel count");
    return status;
}

/* Minimal scalar lookups into the flat manifest. The manifest is written by
 * this example's importer, so a full JSON parser would be scope the example
 * does not need; anything unexpected fails closed rather than defaulting. */
static int manifest_int(const char* text, const char* key, int* out) {
    const char* cursor = strstr(text, key);
    if (!cursor) return -1;
    cursor = strchr(cursor + strlen(key), ':');
    if (!cursor) return -1;
    *out = (int)strtol(cursor + 1, NULL, 10);
    return 0;
}

static int manifest_string(const char* text, const char* key, char* out, size_t out_size) {
    const char* cursor = strstr(text, key);
    const char* start;
    const char* end;
    size_t length;
    if (!cursor) return -1;
    cursor = strchr(cursor + strlen(key), ':');
    if (!cursor) return -1;
    start = strchr(cursor, '"');
    if (!start) return -1;
    end = strchr(start + 1, '"');
    if (!end) return -1;
    length = (size_t)(end - start - 1);
    if (length + 1 > out_size) return -1;
    memcpy(out, start + 1, length);
    out[length] = '\0';
    return 0;
}

int receipt_digit_read_manifest(const char* package_directory, ReceiptDigitLayout* layout,
                                int* width, int* height, char* input_name,
                                size_t input_name_size, char* error, size_t error_size) {
    char path[1024];
    char* text = NULL;
    long size;
    FILE* handle;
    int ok = -1;
    if (!package_directory || !layout || !width || !height) {
        set_error(error, error_size, "manifest read requires a package and outputs");
        return -1;
    }
    snprintf(path, sizeof(path), "%s/manifest.json", package_directory);
    handle = fopen(path, "rb");
    if (!handle) {
        set_error(error, error_size, "package manifest.json could not be opened");
        return -1;
    }
    if (fseek(handle, 0, SEEK_END) != 0 || (size = ftell(handle)) <= 0 ||
        fseek(handle, 0, SEEK_SET) != 0) {
        fclose(handle);
        set_error(error, error_size, "package manifest.json could not be sized");
        return -1;
    }
    text = (char*)malloc((size_t)size + 1);
    if (!text || fread(text, 1, (size_t)size, handle) != (size_t)size) {
        free(text);
        fclose(handle);
        set_error(error, error_size, "package manifest.json could not be read");
        return -1;
    }
    text[size] = '\0';
    fclose(handle);

    if (!strstr(text, "volvoxai-receipt-digit-reader-onnx-package-v1")) {
        set_error(error, error_size,
                  "package is not volvoxai-receipt-digit-reader-onnx-package-v1");
        goto done;
    }
    if (manifest_int(text, "\"slots\"", &layout->slots) != 0 ||
        manifest_int(text, "\"phone_slots\"", &layout->phone_slots) != 0 ||
        manifest_int(text, "\"num_classes\"", &layout->num_classes) != 0 ||
        manifest_int(text, "\"blank_class\"", &layout->blank_class) != 0 ||
        manifest_int(text, "\"width\"", width) != 0 ||
        manifest_int(text, "\"height\"", height) != 0) {
        set_error(error, error_size, "package manifest is missing a decode or preprocess field");
        goto done;
    }
    if (input_name && input_name_size &&
        manifest_string(text, "\"name\"", input_name, input_name_size) != 0) {
        set_error(error, error_size, "package manifest is missing its input name");
        goto done;
    }
    if (layout->slots <= 0 || layout->slots > RECEIPT_DIGIT_MAX_SLOTS ||
        *width <= 0 || *height <= 0) {
        set_error(error, error_size, "package manifest declares an unusable geometry");
        goto done;
    }
    ok = 0;
done:
    free(text);
    return ok;
}

enum { RECEIPT_DIGIT_TENSOR_INLINE_PAYLOAD = 5 };

static void report_failure(char* error, size_t error_size, const char* action,
                           const VolvoxaiV1OperationReport* report) {
    if (!error || !error_size) return;
    snprintf(error, error_size, "%s failed%s%.*s", action,
             report && report->field_message.len ? ": " : "",
             report && report->field_message.len ? (int)report->field_message.len : 0,
             report && report->field_message.len
                 ? (const char*)report->field_message.data : "");
}

static void dispatch_failure(VxCallClient* client, char* error, size_t error_size, const char* action) {
    int32_t length = 0;
    const uint8_t* message = vx_call_error(client, &length);
    if (!error || !error_size) return;
    snprintf(error, error_size, "%s failed%s%.*s", action,
             message && length > 0 ? ": " : "", message && length > 0 ? length : 0,
             message && length > 0 ? (const char*)message : "");
}

static int report_ok(char* error, size_t error_size, const char* action,
                     const VolvoxaiV1OperationReport* report) {
    if (report && report->field_status == VOLVOXAI_V1_NATIVE_STATUS_OK) return 1;
    report_failure(error, error_size, action, report);
    return 0;
}

static int assign_text(const SynurangLiteAllocator* allocator,
                       SynurangLiteBytes* destination, const char* value) {
    return synurang_lite_bytes_assign(allocator, destination, value,
                                      value ? strlen(value) : 0u) == SYNURANG_LITE_OK
               ? 0
               : -1;
}

static void free_encoded(uint8_t* bytes) {
    if (bytes) synurang_lite_default_allocator()->deallocate(
        synurang_lite_default_allocator()->context, bytes);
}

static void release_result_id(VxCallClient* client, int64_t id) {
    VolvoxaiV1ResultRef request;
    uint8_t* response;
    int32_t response_len;
    if (id <= 0) return;
    volvoxai_v1_result_ref_init(&request);
    request.field_result_id = id;
    VX_CALL_MESSAGE(client, VX_RPC_VX_INFERENCE_SERVICE_RELEASE_RESULT,
                    volvoxai_v1_result_ref, &request, response, response_len);
    volvoxai_v1_result_ref_free(&request);
    vx_call_free(client, response);
}

struct ReceiptDigitSession {
    VxCallClient client;
    int64_t runtime_id;
    int64_t model_id;
    int64_t compiled_model_id;
    int64_t context_id;
    ReceiptDigitLayout layout;
    char input_name[128];
    float* logits;
    size_t logits_count;
    int width;
    int height;
};

ReceiptDigitSession* receipt_digit_session_open(const char* package_directory,
                                               const char* backend, char* error,
                                               size_t error_size) {
    ReceiptDigitSession* session;
    VolvoxaiV1CreateRuntimeRequest create_runtime;
    VolvoxaiV1RuntimeHandle runtime;
    VolvoxaiV1LoadModelRequest load_model;
    VolvoxaiV1ModelHandle model;
    VolvoxaiV1BackendPolicy policy;
    VolvoxaiV1CompileModelRequest compile_request;
    VolvoxaiV1CompiledModelHandle compiled;
    VolvoxaiV1ExecutionContextHandle context;
    SynurangLiteBytes* backend_entry;
    SynurangLiteBytes* weight_path;
    uint8_t* encoded = NULL;
    uint8_t* response = NULL;
    size_t encoded_len = 0;
    int32_t response_len = 0;
    char graph_path[1024];
    char weights_path[1024];

    if (!package_directory || !backend) {
        set_error(error, error_size, "package and backend are required");
        return NULL;
    }
    if (error && error_size) error[0] = '\0';
    session = (ReceiptDigitSession*)calloc(1, sizeof(*session));
    if (!session) {
        set_error(error, error_size, "session could not be allocated");
        return NULL;
    }
    if (!vx_call_client_open(&session->client)) {
        set_error(error, error_size, "module instance could not be created");
        free(session);
        return NULL;
    }
    if (receipt_digit_read_manifest(package_directory, &session->layout, &session->width,
                                    &session->height, session->input_name,
                                    sizeof(session->input_name), error, error_size) != 0) {
        goto fail;
    }
    session->logits_count = (size_t)session->layout.slots * (size_t)session->layout.num_classes;
    session->logits = (float*)malloc(session->logits_count * sizeof(float));
    if (!session->logits) {
        set_error(error, error_size, "host output tensor could not be allocated");
        goto fail;
    }

    snprintf(graph_path, sizeof(graph_path), "%s/graph.json", package_directory);
    snprintf(weights_path, sizeof(weights_path), "%s/model.safetensors", package_directory);
    volvoxai_v1_create_runtime_request_init(&create_runtime);
    create_runtime.has_execution_mode = 1;
    create_runtime.field_execution_mode = VOLVOXAI_V1_EXECUTION_MODE_DIRECT;
    if (volvoxai_v1_create_runtime_request_encode(&create_runtime, &encoded, &encoded_len) !=
        SYNURANG_LITE_OK) {
        volvoxai_v1_create_runtime_request_free(&create_runtime);
        set_error(error, error_size, "runtime request could not be encoded");
        goto fail;
    }
    volvoxai_v1_create_runtime_request_free(&create_runtime);
    response = vx_call_bytes(&session->client, VX_RPC_VX_INFERENCE_SERVICE_CREATE_RUNTIME, encoded, (int32_t)encoded_len, &response_len);
    free_encoded(encoded);
    encoded = NULL;
    if (!response) {
        dispatch_failure(&session->client, error, error_size, "runtime creation");
        goto fail;
    }
    volvoxai_v1_runtime_handle_init(&runtime);
    if (volvoxai_v1_runtime_handle_decode(&runtime, response, (size_t)response_len) !=
            SYNURANG_LITE_OK ||
        !report_ok(error, error_size, "runtime creation", runtime.field_report)) {
        volvoxai_v1_runtime_handle_free(&runtime);
        vx_call_free(&session->client, response);
        response = NULL;
        goto fail;
    }
    session->runtime_id = runtime.field_runtime_id;
    volvoxai_v1_runtime_handle_free(&runtime);
    vx_call_free(&session->client, response);
    response = NULL;

    volvoxai_v1_load_model_request_init(&load_model);
    load_model.field_runtime_id = session->runtime_id;
    weight_path = volvoxai_v1_load_model_request_add_weight_paths(&load_model);
    if (assign_text(load_model._allocator, &load_model.field_graph_path, graph_path) != 0 ||
        !weight_path || assign_text(load_model._allocator, weight_path, weights_path) != 0 ||
        volvoxai_v1_load_model_request_encode(&load_model, &encoded, &encoded_len) !=
            SYNURANG_LITE_OK) {
        volvoxai_v1_load_model_request_free(&load_model);
        set_error(error, error_size, "model request could not be encoded");
        goto fail;
    }
    volvoxai_v1_load_model_request_free(&load_model);
    response = vx_call_bytes(&session->client, VX_RPC_VX_INFERENCE_SERVICE_LOAD_MODEL, encoded, (int32_t)encoded_len, &response_len);
    free_encoded(encoded);
    encoded = NULL;
    if (!response) {
        dispatch_failure(&session->client, error, error_size, "model load");
        goto fail;
    }
    volvoxai_v1_model_handle_init(&model);
    if (volvoxai_v1_model_handle_decode(&model, response, (size_t)response_len) !=
            SYNURANG_LITE_OK ||
        !report_ok(error, error_size, "model load", model.field_report)) {
        volvoxai_v1_model_handle_free(&model);
        vx_call_free(&session->client, response);
        response = NULL;
        goto fail;
    }
    session->model_id = model.field_model_id;
    volvoxai_v1_model_handle_free(&model);
    vx_call_free(&session->client, response);
    response = NULL;

    /* A reader deployed on a chosen device should fail loudly rather than land
     * silently on a slower provider, and operator fallback would defeat the
     * point of the byte-domain variants. */
    volvoxai_v1_compile_model_request_init(&compile_request);
    volvoxai_v1_backend_policy_init(&policy);
    compile_request.field_model_id = session->model_id;
    compile_request.field_policy = &policy;
    policy.field_mode = VOLVOXAI_V1_BACKEND_POLICY_MODE_REQUIRE;
    policy.field_operator_fallback = VOLVOXAI_V1_OPERATOR_FALLBACK_FORBID;
    backend_entry = volvoxai_v1_backend_policy_add_backends(&policy);
    if (!backend_entry || assign_text(policy._allocator, backend_entry, backend) != 0 ||
        volvoxai_v1_compile_model_request_encode(&compile_request, &encoded, &encoded_len) !=
            SYNURANG_LITE_OK) {
        compile_request.field_policy = NULL;
        volvoxai_v1_compile_model_request_free(&compile_request);
        volvoxai_v1_backend_policy_free(&policy);
        set_error(error, error_size, "compile request could not be encoded");
        goto fail;
    }
    compile_request.field_policy = NULL;
    volvoxai_v1_compile_model_request_free(&compile_request);
    volvoxai_v1_backend_policy_free(&policy);
    response = vx_call_bytes(&session->client, VX_RPC_VX_INFERENCE_SERVICE_COMPILE_MODEL,
                             encoded, encoded_len, &response_len);
    free_encoded(encoded);
    encoded = NULL;
    if (!response) {
        dispatch_failure(&session->client, error, error_size, "model compile");
        goto fail;
    }
    volvoxai_v1_compiled_model_handle_init(&compiled);
    if (volvoxai_v1_compiled_model_handle_decode(&compiled, response, (size_t)response_len) !=
            SYNURANG_LITE_OK ||
        !report_ok(error, error_size, "model compile", compiled.field_report)) {
        volvoxai_v1_compiled_model_handle_free(&compiled);
        vx_call_free(&session->client, response);
        response = NULL;
        goto fail;
    }
    session->compiled_model_id = compiled.field_compiled_model_id;
    volvoxai_v1_compiled_model_handle_free(&compiled);
    vx_call_free(&session->client, response);
    response = NULL;

    {
        VolvoxaiV1CreateExecutionContextRequest request;
        volvoxai_v1_create_execution_context_request_init(&request);
        request.field_compiled_model_id = session->compiled_model_id;
        if (volvoxai_v1_create_execution_context_request_encode(
                &request, &encoded, &encoded_len) != SYNURANG_LITE_OK) {
            volvoxai_v1_create_execution_context_request_free(&request);
            set_error(error, error_size, "Cannot encode context creation request.");
            goto fail;
        }
        volvoxai_v1_create_execution_context_request_free(&request);
        response = vx_call_bytes(&session->client, VX_RPC_VX_INFERENCE_SERVICE_CREATE_EXECUTION_CONTEXT,
            encoded, (int32_t)encoded_len, &response_len);
        free_encoded(encoded);
        encoded = NULL;
    }
    if (!response) {
        dispatch_failure(&session->client, error, error_size, "context creation");
        goto fail;
    }
    volvoxai_v1_execution_context_handle_init(&context);
    if (volvoxai_v1_execution_context_handle_decode(&context, response, (size_t)response_len) !=
            SYNURANG_LITE_OK ||
        !report_ok(error, error_size, "context creation", context.field_report)) {
        volvoxai_v1_execution_context_handle_free(&context);
        vx_call_free(&session->client, response);
        response = NULL;
        goto fail;
    }
    session->context_id = context.field_context_id;
    volvoxai_v1_execution_context_handle_free(&context);
    vx_call_free(&session->client, response);
    return session;

fail:
    free_encoded(encoded);
    vx_call_free(&session->client, response);
    receipt_digit_session_close(session);
    return NULL;
}

void receipt_digit_session_geometry(const ReceiptDigitSession* session, int* width,
                                    int* height, ReceiptDigitLayout* layout) {
    if (!session) return;
    if (width) *width = session->width;
    if (height) *height = session->height;
    if (layout) *layout = session->layout;
}

static double monotonic_ms(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1.0e6;
}

int receipt_digit_session_execute(ReceiptDigitSession* session, const float* plane,
                                  ReceiptDigitRecord* record, double* elapsed_ms,
                                  char* error, size_t error_size) {
    VolvoxaiV1ExecuteRequest execute;
    VolvoxaiV1Tensor* input;
    VolvoxaiV1ExecutionResultHandle execution;
    VolvoxaiV1ReadOutputResponse output;
    uint8_t* encoded = NULL;
    uint8_t* response = NULL;
    size_t encoded_len = 0;
    int32_t response_len = 0;
    int64_t result_id = 0;
    double started;
    int status_code = -1;

    if (!session || session->context_id <= 0 || !plane || !record) {
        set_error(error, error_size, "session, plane, and record are required");
        return -1;
    }
    if (error && error_size) error[0] = '\0';
    volvoxai_v1_execute_request_init(&execute);
    execute.field_context_id = session->context_id;
    input = volvoxai_v1_execute_request_add_inputs(&execute);
    if (!input ||
        assign_text(execute._allocator, &input->field_name,
                    session->input_name[0] ? session->input_name : "input0") != 0 ||
        !volvoxai_v1_tensor_add_shape(input) || !volvoxai_v1_tensor_add_shape(input) ||
        !volvoxai_v1_tensor_add_shape(input) || !volvoxai_v1_tensor_add_shape(input)) {
        volvoxai_v1_execute_request_free(&execute);
        set_error(error, error_size, "execution request could not be allocated");
        return -1;
    }
    input->field_shape.data[0] = 1;
    input->field_shape.data[1] = 1;
    input->field_shape.data[2] = session->height;
    input->field_shape.data[3] = session->width;
    input->field_dtype = VOLVOXAI_V1_DATA_TYPE_F32;
    input->field_location = VOLVOXAI_V1_MEMORY_LOCATION_HOST;
    input->which_payload = RECEIPT_DIGIT_TENSOR_INLINE_PAYLOAD;
    if (synurang_lite_bytes_assign(execute._allocator, &input->field_inline, plane,
                                   (size_t)session->width * (size_t)session->height *
                                       sizeof(*plane)) != SYNURANG_LITE_OK ||
        volvoxai_v1_execute_request_encode(&execute, &encoded, &encoded_len) !=
            SYNURANG_LITE_OK) {
        volvoxai_v1_execute_request_free(&execute);
        free_encoded(encoded);
        set_error(error, error_size, "execution request could not be encoded");
        return -1;
    }
    volvoxai_v1_execute_request_free(&execute);

    started = monotonic_ms();
    response = vx_call_bytes(&session->client, VX_RPC_VX_INFERENCE_SERVICE_EXECUTE, encoded, (int32_t)encoded_len, &response_len);
    free_encoded(encoded);
    encoded = NULL;
    if (!response) {
        dispatch_failure(&session->client, error, error_size, "execution");
        goto done;
    }
    volvoxai_v1_execution_result_handle_init(&execution);
    if (volvoxai_v1_execution_result_handle_decode(&execution, response,
                                                   (size_t)response_len) != SYNURANG_LITE_OK ||
        !report_ok(error, error_size, "execution", execution.field_report)) {
        volvoxai_v1_execution_result_handle_free(&execution);
        goto done;
    }
    result_id = execution.field_result_id;
    volvoxai_v1_execution_result_handle_free(&execution);
    vx_call_free(&session->client, response);
    {
        VolvoxaiV1ReadOutputRequest request;
        volvoxai_v1_read_output_request_init(&request);
        request.field_result_id = result_id;
        if (assign_text(request._allocator, &request.field_name, "slot_logits") != 0) {
            volvoxai_v1_read_output_request_free(&request);
            response = NULL;
            goto done;
        }
        VX_CALL_MESSAGE(&session->client, VX_RPC_VX_INFERENCE_SERVICE_READ_OUTPUT,
                        volvoxai_v1_read_output_request, &request, response, response_len);
        volvoxai_v1_read_output_request_free(&request);
    }
    if (!response) {
        dispatch_failure(&session->client, error, error_size, "result read");
        goto done;
    }
    volvoxai_v1_read_output_response_init(&output);
    if (volvoxai_v1_read_output_response_decode(&output, response,
                                                (size_t)response_len) != SYNURANG_LITE_OK ||
        !report_ok(error, error_size, "result read", output.field_report) ||
        !output.field_tensor ||
        output.field_tensor->which_payload != RECEIPT_DIGIT_TENSOR_INLINE_PAYLOAD ||
        output.field_tensor->field_inline.len != session->logits_count * sizeof(*session->logits)) {
        volvoxai_v1_read_output_response_free(&output);
        if (!error || !error_size || !error[0]) {
            set_error(error, error_size, "result output is not slot logits");
        }
        goto done;
    }
    memcpy(session->logits, output.field_tensor->field_inline.data,
           session->logits_count * sizeof(*session->logits));
    volvoxai_v1_read_output_response_free(&output);
    /* Stop the clock after the owned output snapshot and before decoding: the
     * decode is host string work that belongs to the application, not to the
     * inference latency being compared. */
    if (elapsed_ms) *elapsed_ms = monotonic_ms() - started;
    if (receipt_digit_decode(session->logits, &session->layout, record) != 0) {
        set_error(error, error_size, "slot decode rejected the package layout");
        goto done;
    }
    status_code = 0;

done:
    vx_call_free(&session->client, response);
    release_result_id(&session->client, result_id);
    return status_code;
}

void receipt_digit_session_close(ReceiptDigitSession* session) {
    if (!session) return;
    vx_call_client_close(&session->client);
    free(session->logits);
    free(session);
}

int receipt_digit_read_receipt(const char* package_directory, const char* image_path,
                               const char* backend, ReceiptDigitRecord* record,
                               char* error, size_t error_size) {
    ReceiptDigitSession* session;
    float* image = NULL;
    int width = 0;
    int height = 0;
    int status_code = -1;

    if (!image_path || !record) {
        set_error(error, error_size, "image and record are required");
        return -1;
    }
    session = receipt_digit_session_open(package_directory, backend, error, error_size);
    if (!session) return -1;
    receipt_digit_session_geometry(session, &width, &height, NULL);
    image = (float*)malloc((size_t)width * (size_t)height * sizeof(float));
    if (!image) {
        set_error(error, error_size, "host input tensor could not be allocated");
        goto done;
    }
    if (receipt_digit_load_image(image_path, image, width, height, error, error_size) != 0) {
        goto done;
    }
    status_code = receipt_digit_session_execute(session, image, record, NULL,
                                                error, error_size);
done:
    free(image);
    receipt_digit_session_close(session);
    return status_code;
}
