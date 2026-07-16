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

static int write_test_weights(const char* path) {
    const int router_shape[2] = {1, 8};
    const int family_shape[3] = {1, 2, 5};
    const int8_t router_logits[8] = {8, 0, 0, 0, 0, 0, 0, 0};
    const int8_t family_logits[10] = {
        -3, -3, -3, -3, 8,  /* first generated token is "A" */
        -3, -3, 8, -3, -3,  /* then EOS */
    };
    SafetensorsFile file;
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "router_logits", SAFETENSORS_DTYPE_I8,
                                 router_shape, 2, router_logits, sizeof(router_logits)) == 0);
    CHECK(safetensors_add_tensor(&file, "family_logits", SAFETENSORS_DTYPE_I8,
                                 family_shape, 3, family_logits, sizeof(family_logits)) == 0);
    /* The real shared TinyReceipt package has 1320 persisted tensors.  Keep
     * this boundary here so the native metadata ceiling cannot silently drop
     * back below the one-file package contract. */
    for (int index = 0; index < 1318; index++) {
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
    const char* router_path = "/tmp/volvox-tinyreceipt-w8a8-example/router.config.json";
    const char* family_path = "/tmp/volvox-tinyreceipt-w8a8-example/family.config.json";
    const char* vocab_path = "/tmp/volvox-tinyreceipt-w8a8-example/vocab.json";
    const char* image_path = "/tmp/volvox-tinyreceipt-w8a8-example/image.png";
    const char* router_config =
        "{\"inputs\":{"
        "\"q_ids\":{\"shape\":[1,4],\"dtype\":\"int32\"},"
        "\"router_keep\":{\"shape\":[1,4],\"dtype\":\"int32\"}},"
        "\"weights_quantization\":{\"router_logits\":{\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":0}},"
        "\"nodes\":[{\"opType\":\"QArgMax\",\"inputs\":{\"input\":\"router_logits\"},"
        "\"outputs\":{\"out\":\"router_family\"},\"outputs_shape\":{\"out\":[1]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},\"params\":{\"axis\":-1}}]}";
    const char* family_config =
        "{\"inputs\":{"
        "\"image\":{\"shape\":[1,1,1,1],\"dtype\":\"float32\"},"
        "\"q_ids\":{\"shape\":[1,4],\"dtype\":\"int32\"},"
        "\"router_keep\":{\"shape\":[1,4],\"dtype\":\"int32\"},"
        "\"memory_keep\":{\"shape\":[1,5],\"dtype\":\"int32\"},"
        "\"y_ids\":{\"shape\":[1,2],\"dtype\":\"int32\"},"
        "\"y_keep\":{\"shape\":[1,2],\"dtype\":\"int32\"}},"
        "\"weights_quantization\":{\"family_logits\":{\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":0}},"
        "\"nodes\":[{\"opType\":\"Identity\",\"inputs\":{\"input\":\"family_logits\"},"
        "\"outputs\":{\"out\":\"family_logits_activation\"},\"outputs_shape\":{\"out\":[1,2,5]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":0}},\"params\":{}},"
        "{\"opType\":\"QArgMax\",\"inputs\":{\"input\":\"family_logits_activation\"},"
        "\"outputs\":{\"out\":\"token_ids\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},\"params\":{\"axis\":-1}}]}";
    const char* vocab = "{\"itos\":[\"<pad>\",\"<bos>\",\"<eos>\",\"<unk>\",\"A\"]}\n";
    const char* manifest =
        "{\"format\":\"volvoxai-tiny-receipt-vqa-w8a8-materialized-package-v1\","
        "\"weights\":{\"file\":\"model.safetensors\"},"
        "\"router\":{\"config\":\"router.config.json\",\"output_name\":\"router_family\","
        "\"inputs\":{\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\"}},"
        "\"explicit_families\":{"
        "\"phone\":{\"config\":\"family.config.json\",\"interface\":{\"inputs\":{\"image\":\"image\",\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\",\"memory_keep\":\"memory_keep\",\"y_ids\":\"y_ids\",\"y_keep\":\"y_keep\"},\"output_name\":\"token_ids\"}},"
        "\"address\":{\"config\":\"family.config.json\",\"interface\":{\"inputs\":{\"image\":\"image\",\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\",\"memory_keep\":\"memory_keep\",\"y_ids\":\"y_ids\",\"y_keep\":\"y_keep\"},\"output_name\":\"token_ids\"}},"
        "\"store\":{\"config\":\"family.config.json\",\"interface\":{\"inputs\":{\"image\":\"image\",\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\",\"memory_keep\":\"memory_keep\",\"y_ids\":\"y_ids\",\"y_keep\":\"y_keep\"},\"output_name\":\"token_ids\"}},"
        "\"item_row\":{\"config\":\"family.config.json\",\"interface\":{\"inputs\":{\"image\":\"image\",\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\",\"memory_keep\":\"memory_keep\",\"y_ids\":\"y_ids\",\"y_keep\":\"y_keep\"},\"output_name\":\"token_ids\"}},"
        "\"item_math\":{\"config\":\"family.config.json\",\"interface\":{\"inputs\":{\"image\":\"image\",\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\",\"memory_keep\":\"memory_keep\",\"y_ids\":\"y_ids\",\"y_keep\":\"y_keep\"},\"output_name\":\"token_ids\"}},"
        "\"item_lookup\":{\"config\":\"family.config.json\",\"interface\":{\"inputs\":{\"image\":\"image\",\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\",\"memory_keep\":\"memory_keep\",\"y_ids\":\"y_ids\",\"y_keep\":\"y_keep\"},\"output_name\":\"token_ids\"}},"
        "\"math\":{\"config\":\"family.config.json\",\"interface\":{\"inputs\":{\"image\":\"image\",\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\",\"memory_keep\":\"memory_keep\",\"y_ids\":\"y_ids\",\"y_keep\":\"y_keep\"},\"output_name\":\"token_ids\"}},"
        "\"other\":{\"config\":\"family.config.json\",\"interface\":{\"inputs\":{\"image\":\"image\",\"q_ids\":\"q_ids\",\"router_keep\":\"router_keep\",\"memory_keep\":\"memory_keep\",\"y_ids\":\"y_ids\",\"y_keep\":\"y_keep\"},\"output_name\":\"token_ids\"}}},"
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
    char* argv[] = {
        "tiny_receipt_w8a8", (char*)dir, "--image", (char*)image_path,
        "--prompt", "  A\\t", "--max-new", "2", "--incremental", NULL,
    };

    (void)mkdir(dir, 0700);
    CHECK(write_test_weights(weights_path) == 0);
    CHECK(write_text(router_path, router_config) == 0);
    CHECK(write_text(family_path, family_config) == 0);
    CHECK(write_text(vocab_path, vocab) == 0);
    CHECK(write_text(manifest_path, manifest) == 0);
    CHECK(write_bytes(image_path, png, sizeof(png)) == 0);
    CHECK(tiny_receipt_w8a8_run(9, argv) == 0);

    remove(image_path);
    remove(manifest_path);
    remove(vocab_path);
    remove(family_path);
    remove(router_path);
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
