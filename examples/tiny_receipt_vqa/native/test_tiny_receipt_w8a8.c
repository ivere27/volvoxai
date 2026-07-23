#include "tiny_receipt_w8a8.h"

#include "tiny_receipt_image.h"
#include "safetensors.h"
#include "volvoxai.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return -1; \
    } \
} while (0)

#define ROUTER_WEIGHT_FILES_JSON \
    "\"weight_files\":[\"router.safetensors\"],"
#define FAMILY_WEIGHT_FILES_JSON \
    "\"weight_files\":[\"family.safetensors\"],"

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(text, 1, strlen(text), file) == strlen(text));
    CHECK(fclose(file) == 0);
    return 0;
}

static int write_bytes(const char* path, const unsigned char* bytes, size_t count) {
    FILE* file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(bytes, 1, count, file) == count);
    CHECK(fclose(file) == 0);
    return 0;
}

static char* remove_all_copy(const char* source, const char* needle) {
    const char* cursor = source;
    const char* found;
    size_t source_length;
    size_t needle_length;
    size_t occurrences = 0;
    char* output;
    char* destination;
    if (!source || !needle || !(needle_length = strlen(needle))) return NULL;
    source_length = strlen(source);
    while ((found = strstr(cursor, needle)) != NULL) {
        occurrences++;
        cursor = found + needle_length;
    }
    output = (char*)malloc(source_length - occurrences * needle_length + 1);
    if (!output) return NULL;
    cursor = source;
    destination = output;
    while ((found = strstr(cursor, needle)) != NULL) {
        size_t prefix = (size_t)(found - cursor);
        memcpy(destination, cursor, prefix);
        destination += prefix;
        cursor = found + needle_length;
    }
    strcpy(destination, cursor);
    return output;
}

static int output_has_line(const char* output, const char* expected) {
    const char* cursor = output;
    const size_t expected_length = strlen(expected);
    while (cursor && *cursor) {
        const char* newline = strchr(cursor, '\n');
        size_t length = newline ? (size_t)(newline - cursor) : strlen(cursor);
        if (length == expected_length &&
            memcmp(cursor, expected, length) == 0) return 1;
        cursor = newline ? newline + 1 : NULL;
    }
    return 0;
}

static int run_and_capture(int argc, char** argv, const char* expected_answer,
                           char* captured_output, size_t captured_capacity) {
    FILE* capture = NULL;
    int saved_stdout = -1;
    int redirected = 0;
    int status = -1;
    long output_size;
    char output[4096];

    if (fflush(stdout) != 0 || !(capture = tmpfile()) ||
        (saved_stdout = dup(STDOUT_FILENO)) < 0 ||
        dup2(fileno(capture), STDOUT_FILENO) < 0) goto done;
    redirected = 1;
    if (tiny_receipt_w8a8_run(argc, argv) != 0 || fflush(stdout) != 0 ||
        dup2(saved_stdout, STDOUT_FILENO) < 0) goto done;
    redirected = 0;
    if (fseek(capture, 0, SEEK_END) != 0 ||
        (output_size = ftell(capture)) < 0 ||
        (size_t)output_size >= sizeof(output) ||
        fseek(capture, 0, SEEK_SET) != 0 ||
        fread(output, 1, (size_t)output_size, capture) != (size_t)output_size)
        goto done;
    output[output_size] = 0;
    if (!output_has_line(output, expected_answer)) {
        fprintf(stderr, "FAIL: expected answer line '%s', captured:\n%s",
                expected_answer, output);
        goto done;
    }
    if (captured_output) {
        if ((size_t)output_size >= captured_capacity) goto done;
        memcpy(captured_output, output, (size_t)output_size + 1);
    }
    status = 0;

done:
    if (redirected) {
        (void)fflush(stdout);
        if (saved_stdout >= 0) (void)dup2(saved_stdout, STDOUT_FILENO);
    }
    if (saved_stdout >= 0) close(saved_stdout);
    if (capture) fclose(capture);
    return status;
}

static int write_test_weights(const char* path, int include_router,
                              int include_family, int filler_count) {
    const int router_shape[2] = {1, 8};
    const int family_table_shape[2] = {5, 8};
    const int family_table_parameter_shape[1] = {5};
    const int8_t router_logits[8] = {8, 0, 0, 0, 0, 0, 0, 0};
    const int8_t family_table[40] = {
        0, 0, 0, 8, 0, 0, 0, 0,  /* PAD -> UNK when its keep bit is enabled */
        0, 0, 0, 0, 8, 0, 0, 0,  /* BOS -> A */
        0, 0, 8, 0, 0, 0, 0, 0,
        0, 0, 0, 8, 0, 0, 0, 0,
        0, 0, 8, 0, 0, 0, 0, 0,  /* A -> EOS */
    };
    const float family_table_scales[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const int8_t family_table_zero_points[5] = {0, 0, 0, 0, 0};
    const int parameter_shape[1] = {1};
    const float unit_scale[1] = {1.0f};
    const int8_t zero_point[1] = {0};
    SafetensorsFile file;
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    if (include_router) {
        CHECK(safetensors_add_tensor(&file, "router_logits", SAFETENSORS_DTYPE_I8,
                                     router_shape, 2, router_logits, sizeof(router_logits)) == 0);
        CHECK(safetensors_add_tensor(&file, "router_scale", SAFETENSORS_DTYPE_F32,
                                     parameter_shape, 1, unit_scale, sizeof(unit_scale)) == 0);
        CHECK(safetensors_add_tensor(&file, "router_zero_point", SAFETENSORS_DTYPE_I8,
                                     parameter_shape, 1, zero_point, sizeof(zero_point)) == 0);
    }
    if (include_family) {
        CHECK(safetensors_add_tensor(&file, "family_table", SAFETENSORS_DTYPE_I8,
                                     family_table_shape, 2, family_table,
                                     sizeof(family_table)) == 0);
        CHECK(safetensors_add_tensor(&file, "family_table_scale", SAFETENSORS_DTYPE_F32,
                                     family_table_parameter_shape, 1,
                                     family_table_scales,
                                     sizeof(family_table_scales)) == 0);
        CHECK(safetensors_add_tensor(&file, "family_table_zero_point", SAFETENSORS_DTYPE_I8,
                                     family_table_parameter_shape, 1,
                                     family_table_zero_points,
                                     sizeof(family_table_zero_points)) == 0);
        CHECK(safetensors_add_tensor(&file, "family_scale", SAFETENSORS_DTYPE_F32,
                                     parameter_shape, 1, unit_scale, sizeof(unit_scale)) == 0);
        CHECK(safetensors_add_tensor(&file, "family_zero_point", SAFETENSORS_DTYPE_I8,
                                     parameter_shape, 1, zero_point, sizeof(zero_point)) == 0);
    }
    for (int index = 0; index < filler_count; index++) {
        const int one[1] = {1};
        const int8_t zero = 0;
        char name[32];
        snprintf(name, sizeof(name), "unused_%04d", index);
        CHECK(safetensors_add_tensor(&file, name, SAFETENSORS_DTYPE_I8, one, 1, &zero, sizeof(zero)) == 0);
    }
    CHECK(safetensors_save(path, &file) == 0);
    safetensors_free(&file);
    return 0;
}

static int test_grayscale_first_bilinear_preprocessing(void) {
    /* PIL's source contract is RGB -> 8-bit L -> BILINEAR.  Red and blue
     * become L values 76 and 29, whose one-pixel bilinear result is 53.
     * Resizing RGB first would instead retain a fractional luminance. */
    const unsigned char rgb[] = {255, 0, 0, 0, 0, 255};
    float output[1] = {0.0f};
    const float expected = ((float)53 / 255.0f) * 2.0f - 1.0f;
    CHECK(tiny_receipt_rgb_to_grayscale_bilinear(rgb, 2, 1, output, 1, 1) == 0);
    CHECK(fabsf(output[0] - expected) < 1.0e-6f);
    return 0;
}

static int test_typed_tinyreceipt_session(void) {
    const char* dir = "/tmp/volvox-tinyreceipt-w8a8-example";
    const char* manifest_path = "/tmp/volvox-tinyreceipt-w8a8-example/package_manifest.json";
    const char* weights_path = "/tmp/volvox-tinyreceipt-w8a8-example/model.safetensors";
    const char* router_weights_path = "/tmp/volvox-tinyreceipt-w8a8-example/router.safetensors";
    const char* family_weights_path = "/tmp/volvox-tinyreceipt-w8a8-example/family.safetensors";
    const char* router_path = "/tmp/volvox-tinyreceipt-w8a8-example/router.graph.json";
    const char* family_path = "/tmp/volvox-tinyreceipt-w8a8-example/family.graph.json";
    const char* vocab_path = "/tmp/volvox-tinyreceipt-w8a8-example/vocab.json";
    const char* image_path = "/tmp/volvox-tinyreceipt-w8a8-example/image.png";
    const char* router_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"q_ids\":{\"shape\":[1,4],\"dtype\":\"int32\"},"
        "\"router_keep\":{\"shape\":[1,4],\"dtype\":\"int32\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{"
        "\"router_logits\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"router_scale\","
        "\"zero_point_tensor\":\"router_zero_point\"}}},"
        "\"nodes\":[{\"opType\":\"QArgMax\",\"inputs\":{\"input\":\"router_logits\"},"
        "\"outputs\":{\"out\":\"router_family\"},\"outputs_shape\":{\"out\":[1]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},\"params\":{\"axis\":-1}}],"
        "\"outputs\":[\"router_family\"]}";
    const char* family_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"image\":{\"shape\":[1,1,1,1],\"dtype\":\"float32\"},"
        "\"q_ids\":{\"shape\":[1,4],\"dtype\":\"int32\"},"
        "\"router_keep\":{\"shape\":[1,4],\"dtype\":\"int32\"},"
        "\"memory_keep\":{\"shape\":[1,5],\"dtype\":\"int32\"},"
        "\"y_ids\":{\"shape\":[1,2],\"dtype\":\"int32\"},"
        "\"y_keep\":{\"shape\":[1,2],\"dtype\":\"int32\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{"
        "\"family_table\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"family_table_scale\","
        "\"zero_point_tensor\":\"family_table_zero_point\"},"
        "\"family_embed\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"family_scale\",\"zero_point_tensor\":\"family_zero_point\"},"
        "\"family_attention\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"family_scale\",\"zero_point_tensor\":\"family_zero_point\"}}},"
        "\"nodes\":[{\"opType\":\"QEmbedding\","
        "\"inputs\":{\"input\":\"y_ids\",\"weight\":\"family_table\"},"
        "\"outputs\":{\"out\":\"family_embed\"},\"outputs_shape\":{\"out\":[1,2,8]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}},"
        "{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"family_embed\","
        "\"k\":\"family_embed\",\"v\":\"family_embed\",\"mask\":\"y_keep\"},"
        "\"outputs\":{\"out\":\"family_attention\"},\"outputs_shape\":{\"out\":[1,2,8]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"heads\":1,\"causal\":true,\"scale\":1.0}},"
        "{\"opType\":\"QArgMax\",\"inputs\":{\"input\":\"family_attention\"},"
        "\"outputs\":{\"out\":\"token_ids\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},\"params\":{\"axis\":-1}}],"
        "\"outputs\":[\"token_ids\"]}";
    const char* vocab = "{\"itos\":[\"<pad>\",\"<bos>\",\"<eos>\",\"<unk>\",\"A\"]}\n";
    const char* manifest =
        "{\"format\":\"volvoxai-tiny-receipt-vqa-w8a8-materialized-package-v1\","
        "\"weights\":{\"file\":\"model.safetensors\"},"
        "\"router\":{\"graph\":\"router.graph.json\"," ROUTER_WEIGHT_FILES_JSON
        "\"output_name\":\"router_family\","
        "\"inputs\":{\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\"}},"
        "\"explicit_families\":{"
        "\"phone\":{\"graph\":\"family.graph.json\"," FAMILY_WEIGHT_FILES_JSON "\"interface\":{\"inputs\":{\"image\":\"image\",\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\",\"memory_keep\":\"memory_keep\",\"y_ids\":\"y_ids\",\"y_keep\":\"y_keep\"},\"output_name\":\"token_ids\"}},"
        "\"address\":{\"graph\":\"family.graph.json\"," FAMILY_WEIGHT_FILES_JSON "\"interface\":{\"inputs\":{\"image\":\"image\",\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\",\"memory_keep\":\"memory_keep\",\"y_ids\":\"y_ids\",\"y_keep\":\"y_keep\"},\"output_name\":\"token_ids\"}},"
        "\"store\":{\"graph\":\"family.graph.json\"," FAMILY_WEIGHT_FILES_JSON "\"interface\":{\"inputs\":{\"image\":\"image\",\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\",\"memory_keep\":\"memory_keep\",\"y_ids\":\"y_ids\",\"y_keep\":\"y_keep\"},\"output_name\":\"token_ids\"}},"
        "\"item_row\":{\"graph\":\"family.graph.json\"," FAMILY_WEIGHT_FILES_JSON "\"interface\":{\"inputs\":{\"image\":\"image\",\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\",\"memory_keep\":\"memory_keep\",\"y_ids\":\"y_ids\",\"y_keep\":\"y_keep\"},\"output_name\":\"token_ids\"}},"
        "\"item_math\":{\"graph\":\"family.graph.json\"," FAMILY_WEIGHT_FILES_JSON "\"interface\":{\"inputs\":{\"image\":\"image\",\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\",\"memory_keep\":\"memory_keep\",\"y_ids\":\"y_ids\",\"y_keep\":\"y_keep\"},\"output_name\":\"token_ids\"}},"
        "\"item_lookup\":{\"graph\":\"family.graph.json\"," FAMILY_WEIGHT_FILES_JSON "\"interface\":{\"inputs\":{\"image\":\"image\",\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\",\"memory_keep\":\"memory_keep\",\"y_ids\":\"y_ids\",\"y_keep\":\"y_keep\"},\"output_name\":\"token_ids\"}},"
        "\"math\":{\"graph\":\"family.graph.json\"," FAMILY_WEIGHT_FILES_JSON "\"interface\":{\"inputs\":{\"image\":\"image\",\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\",\"memory_keep\":\"memory_keep\",\"y_ids\":\"y_ids\",\"y_keep\":\"y_keep\"},\"output_name\":\"token_ids\"}},"
        "\"other\":{\"graph\":\"family.graph.json\"," FAMILY_WEIGHT_FILES_JSON "\"interface\":{\"inputs\":{\"image\":\"image\",\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\",\"memory_keep\":\"memory_keep\",\"y_ids\":\"y_ids\",\"y_keep\":\"y_keep\"},\"output_name\":\"token_ids\"}}},"
        "\"vocab\":{\"file\":\"vocab.json\",\"token_ids\":{\"pad\":0,\"bos\":1,\"eos\":2,\"unk\":3}},"
        "\"preprocessing\":{\"color_space\":\"grayscale\",\"resize\":{\"width\":1,\"height\":1,\"resample\":\"bilinear\"},\"normalization\":\"minus-one-one\"}}";
    static const unsigned char png[] = {
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
        0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00,
        0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0xf8, 0xcf, 0xc0, 0xf0,
        0x1f, 0x00, 0x05, 0x00, 0x01, 0xff, 0x89, 0x99, 0x3d, 0x1d, 0x00, 0x00,
        0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
    };
    char* ordinary_argv[] = {
        "tiny_receipt_w8a8", (char*)dir, "--image", (char*)image_path,
        "--prompt", "  A\\t", "--max-new", "2", NULL,
    };
    char* incremental_argv[] = {
        "tiny_receipt_w8a8", (char*)dir, "--image", (char*)image_path,
        "--prompt", "  A\\t", "--max-new", "2", "--incremental", NULL,
    };
    char ordinary_output[4096] = {0};
    char incremental_output[4096] = {0};
    char* without_router_scope = NULL;
    char* legacy_manifest = NULL;

    (void)mkdir(dir, 0700);
    CHECK(write_text(router_path, router_graph) == 0);
    CHECK(write_text(family_path, family_graph) == 0);
    CHECK(write_text(vocab_path, vocab) == 0);
    CHECK(write_bytes(image_path, png, sizeof(png)) == 0);

    /* A v1 package without graph-scoped fields still falls back to the
     * original top-level weights.file contract. */
    CHECK(write_test_weights(weights_path, 1, 1, 0) == 0);
    without_router_scope = remove_all_copy(
        manifest, "\"weight_files\":[\"router.safetensors\"],");
    CHECK(without_router_scope != NULL);
    legacy_manifest = remove_all_copy(
        without_router_scope, "\"weight_files\":[\"family.safetensors\"],");
    CHECK(legacy_manifest != NULL);
    CHECK(write_text(manifest_path, legacy_manifest) == 0);
    CHECK(run_and_capture(8, ordinary_argv, "A", ordinary_output,
                          sizeof(ordinary_output)) == 0);
    free(without_router_scope);
    free(legacy_manifest);
    without_router_scope = NULL;
    legacy_manifest = NULL;

    /* The global compatibility file now exceeds native MAXT=2048.  Successful
     * inference proves that the runner loads only each graph's scoped file. */
    CHECK(write_test_weights(weights_path, 1, 1, 2050) == 0);
    CHECK(write_test_weights(router_weights_path, 1, 0, 0) == 0);
    CHECK(write_test_weights(family_weights_path, 0, 1, 0) == 0);
    CHECK(write_text(manifest_path, manifest) == 0);
    /* The public context decode lifecycle must accept the same package and
     * preserve its two-token greedy output through reset, seed, and row 1. */
    CHECK(run_and_capture(9, incremental_argv, "A", incremental_output,
                          sizeof(incremental_output)) == 0);
    CHECK(strcmp(ordinary_output, incremental_output) == 0);

    free(without_router_scope);
    free(legacy_manifest);
    remove(image_path);
    remove(manifest_path);
    remove(vocab_path);
    remove(family_path);
    remove(router_path);
    remove(family_weights_path);
    remove(router_weights_path);
    remove(weights_path);
    rmdir(dir);
    return 0;
}

int main(void) {
    CHECK(test_grayscale_first_bilinear_preprocessing() == 0);
    CHECK(test_typed_tinyreceipt_session() == 0);
    puts("TinyReceipt W8A8 native example tests passed");
    return 0;
}
