/* PTQ input/output selection through the generated public dispatch.
 *
 * An empty template_graph_path asks for PtqTemplateInfo.template_graph bytes.
 * A nonempty path writes a file and leaves those response bytes empty. Input
 * paths and input bytes support either output form independently, so pin both
 * mixed modes end to end. */
#include "volvoxai_ffi.h"
#include "../cli/call_client.h"
#include "../src/generated/proto_methods.h"
#include <assert.h>
#include <stdlib.h>

#include "volvoxai_lite.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static VxCallClient client;
static inline const uint8_t* call_error(int32_t* size) { return vx_call_error(&client, size); }
static void close_client(void) { vx_call_client_close(&client); }

static int failures;

#define CHECK(condition, message)                                              \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "FAIL: %s\n", (message));                        \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static void print_last_error(void) {
    int32_t length = 0;
    const uint8_t* message = call_error(&length);
    if (message && length > 0) {
        fprintf(stderr, "  dispatch error: %.*s\n", (int)length,
                (const char*)message);
    }
}

static int invoke_author(
        const VolvoxaiV1AuthorPtqTemplateRequest* request,
        VolvoxaiV1PtqTemplateInfo* response) {
    const SynurangLiteAllocator* allocator = request->_allocator;
    uint8_t* encoded = NULL;
    size_t encoded_length = 0u;
    uint8_t* payload;
    int32_t payload_length = 0;

    if (volvoxai_v1_author_ptq_template_request_encode(
            request, &encoded, &encoded_length) != SYNURANG_LITE_OK) {
        fprintf(stderr, "FAIL: AuthorPtqTemplateRequest encodes\n");
        failures++;
        return 0;
    }
    payload = vx_call_bytes(&client, VX_RPC_VX_QUANTIZATION_SERVICE_AUTHOR_PTQ_TEMPLATE,
        encoded, (int32_t)encoded_length, &payload_length);
    allocator->deallocate(allocator->context, encoded);
    if (!payload || payload_length <= 0) {
        fprintf(stderr, "FAIL: AuthorPtqTemplate returns a response payload\n");
        failures++;
        if (payload) vx_call_free(&client, payload);
        print_last_error();
        return 0;
    }

    volvoxai_v1_ptq_template_info_init(response);
    if (volvoxai_v1_ptq_template_info_decode(
            response, payload, (size_t)payload_length) != SYNURANG_LITE_OK) {
        fprintf(stderr, "FAIL: PtqTemplateInfo decodes\n");
        failures++;
        volvoxai_v1_ptq_template_info_free(response);
        vx_call_free(&client, payload);
        return 0;
    }
    vx_call_free(&client, payload);
    return 1;
}

static void check_ok_report(const VolvoxaiV1PtqTemplateInfo* response) {
    CHECK(response->field_report != NULL, "authoring report is present");
    if (response->field_report) {
        CHECK(response->field_report->field_status ==
                  VOLVOXAI_V1_NATIVE_STATUS_OK,
              "authoring reports OK");
    }
}

static void check_invalid_argument_report(
        const VolvoxaiV1PtqTemplateInfo* response,
        const char* message) {
    CHECK(response->field_report != NULL, "rejected authoring has a report");
    if (response->field_report) {
        CHECK(response->field_report->field_status ==
                  VOLVOXAI_V1_NATIVE_STATUS_INVALID_ARGUMENT,
              message);
    }
}

static int assign_path_with_nul(const SynurangLiteAllocator* allocator,
                                SynurangLiteBytes* field,
                                const char* prefix) {
    static const char suffix[] = "\0ignored-suffix";
    size_t prefix_length = strlen(prefix);
    size_t length = prefix_length + sizeof(suffix) - 1u;
    char* bytes = (char*)malloc(length);
    int assigned;
    if (!bytes) return 0;
    memcpy(bytes, prefix, prefix_length);
    memcpy(bytes + prefix_length, suffix, sizeof(suffix) - 1u);
    assigned = synurang_lite_bytes_assign(allocator, field, bytes, length) ==
               SYNURANG_LITE_OK;
    free(bytes);
    return assigned;
}

int main(int argc, char** argv) {
    if (!vx_call_client_open(&client)) return 1;
    atexit(close_client);
    static const char graph[] =
        "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},"
        "\"inputs\":{\"input\":{\"shape\":[1,4],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"gelu\",\"opType\":\"GELU\","
        "\"inputs\":{\"input\":\"input\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"output\","
        "\"shape\":[1,4],\"dtype\":\"float32\"}},\"params\":{}}],"
        "\"outputs\":[\"output\"]}";
    const char* graph_path = argc > 1 ? argv[1] : "ptq_api_source.graph.json";
    const char* output_path =
        argc > 2 ? argv[2] : "ptq_api_authored.graph.json";
    VolvoxaiV1AuthorPtqTemplateRequest request;
    VolvoxaiV1PtqTemplateInfo response;
    FILE* handle = fopen(graph_path, "wb");

    CHECK(handle != NULL, "source graph file opens");
    if (!handle) return 1;
    CHECK(fwrite(graph, 1u, sizeof(graph) - 1u, handle) == sizeof(graph) - 1u,
          "source graph file is written");
    CHECK(fclose(handle) == 0, "source graph file closes");
    if (failures) {
        remove(graph_path);
        return 1;
    }

    volvoxai_v1_author_ptq_template_request_init(&request);
    CHECK(synurang_lite_bytes_assign(
              request._allocator, &request.field_source_graph_path,
              graph_path, strlen(graph_path)) == SYNURANG_LITE_OK,
          "source_graph_path is assigned");
    /* template_graph_path intentionally remains empty. */
    if (invoke_author(&request, &response)) {
        char* authored_text;
        check_ok_report(&response);
        CHECK(response.field_quantized_nodes == 1u,
              "the path-backed graph is authored");
        CHECK(response.field_template_graph.len > 0u,
              "empty output path returns template_graph bytes");

        authored_text = (char*)malloc(response.field_template_graph.len + 1u);
        CHECK(authored_text != NULL, "template inspection buffer allocates");
        if (authored_text) {
            memcpy(authored_text, response.field_template_graph.data,
                   response.field_template_graph.len);
            authored_text[response.field_template_graph.len] = '\0';
            CHECK(strstr(authored_text, "QGELU") != NULL,
                  "returned template contains the authored QGELU node");
            free(authored_text);
        }
        volvoxai_v1_ptq_template_info_free(&response);
    }
    volvoxai_v1_author_ptq_template_request_free(&request);

    /* The inverse mixed mode: byte input plus a native output path writes the
     * file and leaves template_graph empty. */
    (void)remove(output_path);
    volvoxai_v1_author_ptq_template_request_init(&request);
    CHECK(synurang_lite_bytes_assign(
              request._allocator, &request.field_source_graph,
              graph, sizeof(graph) - 1u) == SYNURANG_LITE_OK,
          "source_graph bytes are assigned");
    CHECK(synurang_lite_bytes_assign(
              request._allocator, &request.field_template_graph_path,
              output_path, strlen(output_path)) == SYNURANG_LITE_OK,
          "template_graph_path is assigned");
    if (invoke_author(&request, &response)) {
        char authored_file[4096];
        size_t authored_length;
        check_ok_report(&response);
        CHECK(response.field_quantized_nodes == 1u,
              "the byte-backed graph is authored");
        CHECK(response.field_template_graph.len == 0u,
              "nonempty output path leaves template_graph empty");
        volvoxai_v1_ptq_template_info_free(&response);

        handle = fopen(output_path, "rb");
        CHECK(handle != NULL, "authored template file opens");
        if (handle) {
            authored_length = fread(authored_file, 1u,
                                    sizeof(authored_file) - 1u, handle);
            authored_file[authored_length] = '\0';
            CHECK(!ferror(handle), "authored template file reads");
            CHECK(fclose(handle) == 0, "authored template file closes");
            CHECK(strstr(authored_file, "QGELU") != NULL,
                  "output file contains the authored QGELU node");
        }
    }
    volvoxai_v1_author_ptq_template_request_free(&request);

    /* Path fields are protobuf bytes until this handler converts them to C
     * strings. Embedded NUL must be rejected, never treated as a terminator
     * that silently discards an attacker-controlled suffix. */
    (void)remove(output_path);
    volvoxai_v1_author_ptq_template_request_init(&request);
    CHECK(synurang_lite_bytes_assign(
              request._allocator, &request.field_source_graph,
              graph, sizeof(graph) - 1u) == SYNURANG_LITE_OK,
          "source_graph bytes are assigned for NUL output-path test");
    CHECK(assign_path_with_nul(request._allocator,
                               &request.field_template_graph_path,
                               output_path),
          "embedded-NUL template_graph_path is assigned");
    if (invoke_author(&request, &response)) {
        FILE* unexpected;
        check_invalid_argument_report(
            &response, "embedded-NUL template path is INVALID_ARGUMENT");
        volvoxai_v1_ptq_template_info_free(&response);
        unexpected = fopen(output_path, "rb");
        CHECK(unexpected == NULL,
              "embedded-NUL template path does not create its prefix");
        if (unexpected) fclose(unexpected);
    }
    volvoxai_v1_author_ptq_template_request_free(&request);

    volvoxai_v1_author_ptq_template_request_init(&request);
    CHECK(assign_path_with_nul(request._allocator,
                               &request.field_source_graph_path,
                               graph_path),
          "embedded-NUL source_graph_path is assigned");
    if (invoke_author(&request, &response)) {
        check_invalid_argument_report(
            &response, "embedded-NUL source path is INVALID_ARGUMENT");
        volvoxai_v1_ptq_template_info_free(&response);
    }
    volvoxai_v1_author_ptq_template_request_free(&request);

    volvoxai_v1_author_ptq_template_request_init(&request);
    CHECK(synurang_lite_bytes_assign(
              request._allocator, &request.field_source_graph_path,
              graph_path, strlen(graph_path)) == SYNURANG_LITE_OK,
          "source_graph_path is assigned for NUL weight-path test");
    request.field_weight_paths.data = (SynurangLiteBytes*)
        request._allocator->allocate(request._allocator->context,
                                     sizeof(SynurangLiteBytes));
    CHECK(request.field_weight_paths.data != NULL,
          "weight_paths entry allocates");
    if (request.field_weight_paths.data) {
        memset(request.field_weight_paths.data, 0, sizeof(SynurangLiteBytes));
        request.field_weight_paths.len = 1u;
        request.field_weight_paths.cap = 1u;
        CHECK(assign_path_with_nul(request._allocator,
                                   &request.field_weight_paths.data[0],
                                   graph_path),
              "embedded-NUL weight path is assigned");
    }
    if (request.field_weight_paths.data && invoke_author(&request, &response)) {
        check_invalid_argument_report(
            &response, "embedded-NUL weight path is INVALID_ARGUMENT");
        volvoxai_v1_ptq_template_info_free(&response);
    }
    volvoxai_v1_author_ptq_template_request_free(&request);

    CHECK(remove(graph_path) == 0, "source graph fixture is removed");
    (void)remove(output_path);
    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("PTQ input and output forms are independent\n");
    return 0;
}
