/* Receipt digit reader: native application code on VolvoxAI's C inference ABI.
 *
 * Preprocessing, manifest policy, and slot decoding live here. Runtime,
 * compilation, shape proof, and kernel dispatch stay behind the opaque
 * VxRuntime/VxModel/VxCompiledModel handles.
 */

#include "receipt_digit_reader.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "volvoxai.h"

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

static void report_failure(char* error, size_t error_size, const char* action,
                           VxStatus status, const VxReport* report) {
    if (!error || !error_size) return;
    /* Carry the offending node and route evidence through. A bare
     * "backend cannot attest the shape domain" says a backend was refused but
     * not which node refused it, which is the only part that is actionable. */
    snprintf(error, error_size, "%s failed: %s%s%s%s%s%s%s", action,
             vx_status_string(status),
             report && report->message[0] ? ": " : "",
             report && report->message[0] ? report->message : "",
             report && report->offending_node[0] ? " [node " : "",
             report && report->offending_node[0] ? report->offending_node : "",
             report && report->offending_node[0] ? "]" : "",
             report && report->route_evidence[0] ? report->route_evidence : "");
}

struct ReceiptDigitSession {
    VxRuntime* runtime;
    VxModel* model;
    VxCompiledModel* compiled;
    VxExecutionContext* context;
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
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxReport report = VX_REPORT_INIT;
    const char* backends[1];
    const char* weight_paths[1];
    char graph_path[1024];
    char weights_path[1024];
    VxStatus status;

    if (!package_directory || !backend) {
        set_error(error, error_size, "package and backend are required");
        return NULL;
    }
    session = (ReceiptDigitSession*)calloc(1, sizeof(*session));
    if (!session) {
        set_error(error, error_size, "session could not be allocated");
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
    status = vx_runtime_create(&runtime_options, &session->runtime, &report);
    if (status != VX_STATUS_OK) {
        report_failure(error, error_size, "runtime creation", status, &report);
        goto fail;
    }
    weight_paths[0] = weights_path;
    source.graph_path = graph_path;
    source.weight_paths = weight_paths;
    source.weight_path_count = 1;
    report = (VxReport)VX_REPORT_INIT;
    status = vx_runtime_load_model(session->runtime, &source, &session->model, &report);
    if (status != VX_STATUS_OK) {
        report_failure(error, error_size, "model load", status, &report);
        goto fail;
    }
    /* A reader deployed on a chosen device should fail loudly rather than land
     * silently on a slower provider, and operator fallback would defeat the
     * point of the byte-domain variants. */
    backends[0] = backend;
    policy.mode = VX_BACKEND_REQUIRE;
    policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    policy.backends = backends;
    policy.backend_count = 1;
    report = (VxReport)VX_REPORT_INIT;
    status = vx_model_compile(session->model, &policy, &session->compiled, &report);
    if (status != VX_STATUS_OK) {
        report_failure(error, error_size, "model compile", status, &report);
        goto fail;
    }
    report = (VxReport)VX_REPORT_INIT;
    status = vx_compiled_model_create_context(session->compiled, NULL, &session->context,
                                              &report);
    if (status != VX_STATUS_OK) {
        report_failure(error, error_size, "context creation", status, &report);
        goto fail;
    }
    return session;

fail:
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
    VxTensorBinding binding = VX_TENSOR_BINDING_INIT;
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    size_t required = 0;
    double started;
    VxStatus status;
    int status_code = -1;

    if (!session || !session->context || !plane || !record) {
        set_error(error, error_size, "session, plane, and record are required");
        return -1;
    }
    binding.name = session->input_name[0] ? session->input_name : "input0";
    binding.dtype = VX_DTYPE_F32;
    binding.rank = 4;
    binding.shape[0] = 1;
    binding.shape[1] = 1;
    binding.shape[2] = session->height;
    binding.shape[3] = session->width;
    binding.data = plane;
    binding.byte_size = (size_t)session->width * (size_t)session->height * sizeof(float);

    started = monotonic_ms();
    status = vx_execution_context_execute(session->context, &binding, 1, &result, &report);
    if (status != VX_STATUS_OK) {
        report_failure(error, error_size, "execution", status, &report);
        goto done;
    }
    report = (VxReport)VX_REPORT_INIT;
    status = vx_result_read(result, "slot_logits", session->logits,
                            session->logits_count * sizeof(float), &required, &report);
    if (status != VX_STATUS_OK) {
        report_failure(error, error_size, "result read", status, &report);
        goto done;
    }
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
    if (result) vx_result_release(result);
    return status_code;
}

void receipt_digit_session_close(ReceiptDigitSession* session) {
    if (!session) return;
    if (session->context) {
        VxReport report = VX_REPORT_INIT;
        vx_execution_context_close(session->context, &report);
        vx_execution_context_release(session->context);
    }
    if (session->compiled) vx_compiled_model_release(session->compiled);
    if (session->model) vx_model_release(session->model);
    if (session->runtime) {
        VxReport report = VX_REPORT_INIT;
        vx_runtime_close(session->runtime, &report);
        vx_runtime_release(session->runtime);
    }
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
