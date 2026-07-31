#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif

#include "tiny_receipt_split_w8a8.h"

#include "safetensors.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return -1; \
    } \
} while (0)

int tiny_receipt_split_w8a8_sha256_file(const char* path, char digest_hex[65]);

typedef struct {
    char root[PATH_MAX];
    char manifest[PATH_MAX];
    char config[PATH_MAX];
    char vocab[PATH_MAX];
    char image[PATH_MAX];
    char encoder_dir[PATH_MAX];
    char encoder_graph[PATH_MAX];
    char encoder_weights[PATH_MAX];
    char encoder_report[PATH_MAX];
    char decoder_dir[PATH_MAX];
    char decoder_graph[PATH_MAX];
    char decoder_weights[PATH_MAX];
    char decoder_report[PATH_MAX];
} SplitFixture;

typedef enum {
    BPE_FIXTURE_VALID = 0,
    BPE_FIXTURE_ATOMIC_SWAPPED = 1,
    BPE_FIXTURE_ATOMIC_DUPLICATE = 2,
    BPE_FIXTURE_ATOMIC_SHORT = 3,
    BPE_FIXTURE_UNUSED_DUPLICATE = 4,
} BpeFixtureVariant;

static const char* bpe_fixture_hash(BpeFixtureVariant variant) {
    static const char* const hashes[] = {
        "612e8425883fd7e3f0292912ab39bd44c72a1da9e399ee455639e6e612acb939",
        "093ab381b98940c1317de73284599aeb022e11f6554f9d1db84d06c1a6c1a270",
        "1b916bde545747dd6100051cc8badd8048c240802ecbed59058b8ab69c62b848",
        "f2cf1d9c73ae4a73285f8d684a0cd3f3ea1fef1c572ae11a2d2fcec948be0f1b",
        "9ca706f797118cc68a9f65f665d6f4794c1260626569eb9ef40ef1025ef03e09",
    };
    return variant >= BPE_FIXTURE_VALID && variant <= BPE_FIXTURE_UNUSED_DUPLICATE
        ? hashes[(int)variant] : NULL;
}

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

static long file_size(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 && st.st_size > 0 ? (long)st.st_size : -1;
}

static int replace_once_in_file(const char* path, const char* needle,
                                const char* replacement) {
    FILE* file = NULL;
    char* original = NULL;
    char* updated = NULL;
    char* match;
    long length;
    size_t original_size;
    size_t needle_size;
    size_t replacement_size;
    size_t prefix_size;
    size_t updated_size;
    int status = -1;

    if (!path || !needle || !needle[0] || !replacement ||
        !(file = fopen(path, "rb")) || fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0)
        goto done;
    original_size = (size_t)length;
    if (!(original = (char*)malloc(original_size + 1)) ||
        fread(original, 1, original_size, file) != original_size) goto done;
    if (fclose(file) != 0) {
        file = NULL;
        goto done;
    }
    file = NULL;
    original[original_size] = 0;
    needle_size = strlen(needle);
    replacement_size = strlen(replacement);
    match = strstr(original, needle);
    if (!match || strstr(match + needle_size, needle)) goto done;
    if (replacement_size > needle_size &&
        original_size > SIZE_MAX - (replacement_size - needle_size)) goto done;
    updated_size = original_size - needle_size + replacement_size;
    if (!(updated = (char*)malloc(updated_size))) goto done;
    prefix_size = (size_t)(match - original);
    memcpy(updated, original, prefix_size);
    memcpy(updated + prefix_size, replacement, replacement_size);
    memcpy(updated + prefix_size + replacement_size, match + needle_size,
           original_size - prefix_size - needle_size);
    status = write_bytes(path, (const unsigned char*)updated, updated_size);

done:
    if (file) fclose(file);
    free(updated);
    free(original);
    return status;
}

static int output_has_line(const char* output, const char* expected) {
    const char* cursor = output;
    const size_t expected_length = strlen(expected);
    while (cursor && *cursor) {
        const char* newline = strchr(cursor, '\n');
        size_t length = newline ? (size_t)(newline - cursor) : strlen(cursor);
        if (length == expected_length && memcmp(cursor, expected, length) == 0) return 1;
        cursor = newline ? newline + 1 : NULL;
    }
    return 0;
}

static int run_and_capture(int argc, char** argv, int expect_success,
                           const char* expected_answer) {
    FILE* capture = NULL;
    int saved_stdout = -1;
    int redirected = 0;
    int run_status;
    int status = -1;
    long output_size;
    char output[4096];

    if (fflush(stdout) != 0 || !(capture = tmpfile()) ||
        (saved_stdout = dup(STDOUT_FILENO)) < 0 ||
        dup2(fileno(capture), STDOUT_FILENO) < 0) goto done;
    redirected = 1;
    run_status = tiny_receipt_split_w8a8_run(argc, argv);
    if (fflush(stdout) != 0 || dup2(saved_stdout, STDOUT_FILENO) < 0) goto done;
    redirected = 0;
    if ((run_status == 0) != expect_success ||
        fseek(capture, 0, SEEK_END) != 0 ||
        (output_size = ftell(capture)) < 0 ||
        (size_t)output_size >= sizeof(output) ||
        fseek(capture, 0, SEEK_SET) != 0 ||
        fread(output, 1, (size_t)output_size, capture) != (size_t)output_size)
        goto done;
    output[output_size] = 0;
    if (expected_answer && !output_has_line(output, expected_answer)) {
        fprintf(stderr, "FAIL: expected visible answer line '%s', captured:\n%s",
                expected_answer, output);
        goto done;
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

static int test_sha256_known_vector(void) {
    static const char expected[] =
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    char path[] = "/tmp/volvox-tinyreceipt-sha256-XXXXXX";
    char actual[65];
    int descriptor = mkstemp(path);
    int status = -1;
    if (descriptor < 0 || write(descriptor, "abc", 3) != 3) goto done;
    if (close(descriptor) != 0) {
        descriptor = -1;
        goto done;
    }
    descriptor = -1;
    if (tiny_receipt_split_w8a8_sha256_file(path, actual) != 0 ||
        strcmp(actual, expected) != 0) goto done;
    status = 0;

done:
    if (descriptor >= 0) close(descriptor);
    unlink(path);
    return status;
}

static int path_join(char* output, size_t capacity, const char* root, const char* leaf) {
    int count = snprintf(output, capacity, "%s/%s", root, leaf);
    return count > 0 && (size_t)count < capacity ? 0 : -1;
}

static int fixture_paths(SplitFixture* fixture, const char* root) {
    CHECK(fixture != NULL && root != NULL);
    memset(fixture, 0, sizeof(*fixture));
    CHECK(strlen(root) < sizeof(fixture->root));
    memcpy(fixture->root, root, strlen(root) + 1);
    CHECK(path_join(fixture->manifest, sizeof(fixture->manifest), root,
                    "package_manifest.json") == 0);
    CHECK(path_join(fixture->config, sizeof(fixture->config), root, "config.json") == 0);
    CHECK(path_join(fixture->vocab, sizeof(fixture->vocab), root, "vocab.json") == 0);
    CHECK(path_join(fixture->image, sizeof(fixture->image), root, "image.png") == 0);
    CHECK(path_join(fixture->encoder_dir, sizeof(fixture->encoder_dir), root, "encoder") == 0);
    CHECK(path_join(fixture->encoder_graph, sizeof(fixture->encoder_graph),
                    fixture->encoder_dir, "graph.json") == 0);
    CHECK(path_join(fixture->encoder_weights, sizeof(fixture->encoder_weights),
                    fixture->encoder_dir, "model.safetensors") == 0);
    CHECK(path_join(fixture->encoder_report, sizeof(fixture->encoder_report),
                    fixture->encoder_dir, "export_report.json") == 0);
    CHECK(path_join(fixture->decoder_dir, sizeof(fixture->decoder_dir), root, "decoder") == 0);
    CHECK(path_join(fixture->decoder_graph, sizeof(fixture->decoder_graph),
                    fixture->decoder_dir, "graph.json") == 0);
    CHECK(path_join(fixture->decoder_weights, sizeof(fixture->decoder_weights),
                    fixture->decoder_dir, "model.safetensors") == 0);
    CHECK(path_join(fixture->decoder_report, sizeof(fixture->decoder_report),
                    fixture->decoder_dir, "export_report.json") == 0);
    return 0;
}

static void cleanup_fixture(const SplitFixture* fixture) {
    if (!fixture || !fixture->root[0]) return;
    remove(fixture->manifest);
    remove(fixture->image);
    remove(fixture->vocab);
    remove(fixture->config);
    remove(fixture->decoder_report);
    remove(fixture->decoder_graph);
    remove(fixture->decoder_weights);
    remove(fixture->encoder_report);
    remove(fixture->encoder_graph);
    remove(fixture->encoder_weights);
    rmdir(fixture->decoder_dir);
    rmdir(fixture->encoder_dir);
    rmdir(fixture->root);
}

static int write_vocab(const char* path, int duplicate) {
    FILE* file = fopen(path, "wb");
    static const char* const special[] = {
        "<pad>", "<bos>", "<eos>", "<unk>", "A",
    };
    if (!file) return -1;
    if (fputs("{\"itos\":[", file) == EOF) goto fail;
    for (int index = 0; index < 760; index++) {
        if (index > 0 && fputc(',', file) == EOF) goto fail;
        if (index < 5) {
            if (fprintf(file, "\"%s\"", special[index]) < 0) goto fail;
        } else if (fprintf(file, "\"fixture-token-%03d\"",
                           duplicate && index == 6 ? 5 : index) < 0) {
            goto fail;
        }
    }
    if (fputs("]}\n", file) == EOF || fclose(file) != 0) return -1;
    return 0;
fail:
    fclose(file);
    return -1;
}

static int write_bpe_vocab(const char* path, BpeFixtureVariant variant) {
    static const char* const atomic[] = {
        "<field>", "</field>", "<value>", "</value>",
        "<op>", "</op>", "<answer>", "</answer>",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    };
    static const char* const extras[] = {
        "p", "h", "ph", "o", "pho", "n", "phon", "e", "phone", " ",
    };
    const char* hash = bpe_fixture_hash(variant);
    const int atomic_count = variant == BPE_FIXTURE_ATOMIC_SHORT ? 17 : 18;
    FILE* file = fopen(path, "wb");
    if (!file || !hash) {
        if (file) fclose(file);
        return -1;
    }
    if (fputs("{\"type\":\"byte_fallback_bpe\",\"version\":1,"
              "\"vocab_size\":1536,\"itos\":[", file) == EOF) goto fail;
    for (int index = 0; index < 1536; index++) {
        if (index && fputc(',', file) == EOF) goto fail;
        if (index < 4) {
            static const char* const special[] = {
                "<pad>", "<bos>", "<eos>", "<unk>",
            };
            if (fprintf(file, "\"%s\"", special[index]) < 0) goto fail;
        } else if (index < 22) {
            if (fprintf(file, "\"%s\"", atomic[index - 4]) < 0) goto fail;
        } else if (index < 278) {
            if (fprintf(file, "\"<0x%02X>\"", index - 22) < 0) goto fail;
        } else if (index < 288) {
            if (fprintf(file, "\"%s\"", extras[index - 278]) < 0) goto fail;
        } else if (fprintf(file, "\"<unused_%04d>\"", index - 288) < 0) {
            goto fail;
        }
    }
    if (fputs("],\"merges\":[[\"p\",\"h\"],[\"ph\",\"o\"],"
              "[\"pho\",\"n\"],[\"phon\",\"e\"]],"
              "\"normalization\":\"NFC\",\"atomic_tokens\":[", file) == EOF)
        goto fail;
    for (int index = 0; index < atomic_count; index++) {
        int source_index = index;
        if (variant == BPE_FIXTURE_ATOMIC_SWAPPED && index < 2)
            source_index = 1 - index;
        if (variant == BPE_FIXTURE_ATOMIC_DUPLICATE && index == 1)
            source_index = 0;
        if ((index && fputc(',', file) == EOF) ||
            fprintf(file, "\"%s\"", atomic[source_index]) < 0) goto fail;
    }
    if (fputs("],\"byte_tokens\":[", file) == EOF) goto fail;
    for (int byte = 0; byte < 256; byte++) {
        if ((byte && fputc(',', file) == EOF) ||
            fprintf(file, "\"<0x%02X>\"", byte) < 0) goto fail;
    }
    if (fputs("],\"unused_tokens\":[", file) == EOF) goto fail;
    for (int index = 0; index < 1248; index++) {
        int source_index = variant == BPE_FIXTURE_UNUSED_DUPLICATE && index == 1
            ? 0 : index;
        if ((index && fputc(',', file) == EOF) ||
            fprintf(file, "\"<unused_%04d>\"", source_index) < 0) goto fail;
    }
    if (fprintf(file,
                "],\"special_tokens\":{\"pad\":\"<pad>\","
                "\"bos\":\"<bos>\",\"eos\":\"<eos>\","
                "\"unk\":\"<unk>\"},\"tokenizer_hash\":\"%s\"}\n",
                hash) < 0 || fclose(file) != 0) return -1;
    return 0;

fail:
    fclose(file);
    return -1;
}

static int write_encoder_weights(const char* path) {
    const int memory_shape[3] = {1, 402, 320};
    const int mask_shape[2] = {1, 402};
    const int router_shape[2] = {1, 8};
    const int family_shape[1] = {1};
    const float router_logits[8] = {8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int32_t selected_family[1] = {0};
    float* memory = (float*)calloc(402u * 320u, sizeof(*memory));
    int32_t* mask = (int32_t*)calloc(402u, sizeof(*mask));
    SafetensorsFile file;
    int status = -1;

    if (!memory || !mask) goto done;
    if (safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0) goto done;
    if (safetensors_add_tensor(&file, "memory_value", SAFETENSORS_DTYPE_F32,
                               memory_shape, 3, memory,
                               402u * 320u * sizeof(*memory)) != 0 ||
        safetensors_add_tensor(&file, "mask_value", SAFETENSORS_DTYPE_I32,
                               mask_shape, 2, mask, 402u * sizeof(*mask)) != 0 ||
        safetensors_add_tensor(&file, "router_value", SAFETENSORS_DTYPE_F32,
                               router_shape, 2, router_logits,
                               sizeof(router_logits)) != 0 ||
        safetensors_add_tensor(&file, "family_value", SAFETENSORS_DTYPE_I32,
                               family_shape, 1, selected_family,
                               sizeof(selected_family)) != 0 ||
        safetensors_save(path, &file) != 0) {
        safetensors_free(&file);
        goto done;
    }
    safetensors_free(&file);
    status = 0;
done:
    free(mask);
    free(memory);
    return status;
}

static int write_decoder_weights(const char* path, int runtime_family,
                                 int vocab_count) {
    const int table_shape[2] = {vocab_count, vocab_count};
    const int table_parameter_shape[1] = {vocab_count};
    const int scalar_shape[1] = {1};
    const float unit_scale[1] = {1.0f};
    const int8_t unit_zero_point[1] = {0};
    const int32_t address_family[1] = {1};
    int32_t address_token_ids[192];
    size_t table_elements;
    int8_t* table = NULL;
    float* table_scales = NULL;
    int8_t* table_zero_points = NULL;
    SafetensorsFile file;
    int status = -1;

    if (vocab_count <= 4 || (size_t)vocab_count > SIZE_MAX / (size_t)vocab_count)
        return -1;
    table_elements = (size_t)vocab_count * (size_t)vocab_count;
    table = (int8_t*)calloc(table_elements, sizeof(*table));
    table_scales = (float*)malloc((size_t)vocab_count * sizeof(*table_scales));
    table_zero_points = (int8_t*)calloc((size_t)vocab_count,
                                        sizeof(*table_zero_points));
    if (!table || !table_scales || !table_zero_points) goto done;
    for (int token = 0; token < vocab_count; token++) {
        table[(size_t)token * (size_t)vocab_count + 2u] = 8;
        table_scales[token] = 1.0f;
    }
    table[(size_t)vocab_count + 2u] = 0;
    table[(size_t)vocab_count + 4u] = 8; /* BOS -> token ID 4 */
    for (int index = 0; index < 192; index++) address_token_ids[index] = 2;
    address_token_ids[0] = 5;
    if (safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0) goto done;
    if (safetensors_add_tensor(&file, "next_token_table", SAFETENSORS_DTYPE_I8,
                               table_shape, 2, table, table_elements) != 0 ||
        safetensors_add_tensor(&file, "table.scale", SAFETENSORS_DTYPE_F32,
                               table_parameter_shape, 1, table_scales,
                               (size_t)vocab_count * sizeof(*table_scales)) != 0 ||
        safetensors_add_tensor(&file, "table.zero_point", SAFETENSORS_DTYPE_I8,
                               table_parameter_shape, 1, table_zero_points,
                               (size_t)vocab_count * sizeof(*table_zero_points)) != 0 ||
        safetensors_add_tensor(&file, "unit.scale", SAFETENSORS_DTYPE_F32,
                               scalar_shape, 1, unit_scale, sizeof(unit_scale)) != 0 ||
        safetensors_add_tensor(&file, "unit.zero_point", SAFETENSORS_DTYPE_I8,
                               scalar_shape, 1, unit_zero_point,
                               sizeof(unit_zero_point)) != 0 ||
        (runtime_family &&
         (safetensors_add_tensor(&file, "address_family", SAFETENSORS_DTYPE_I32,
                                 scalar_shape, 1, address_family,
                                 sizeof(address_family)) != 0 ||
          safetensors_add_tensor(&file, "address_token_ids", SAFETENSORS_DTYPE_I32,
                                 (const int[]){1, 192}, 2, address_token_ids,
                                 sizeof(address_token_ids)) != 0)) ||
        safetensors_save(path, &file) != 0) {
        safetensors_free(&file);
        goto done;
    }
    safetensors_free(&file);
    status = 0;
done:
    free(table_zero_points);
    free(table_scales);
    free(table);
    return status;
}

static int write_manifest(const SplitFixture* fixture, const char* format,
                          int runtime_family, int direct_logits, int bpe) {
    enum {
        ASSET_CONFIG,
        ASSET_VOCAB,
        ASSET_ENCODER_GRAPH,
        ASSET_ENCODER_WEIGHTS,
        ASSET_ENCODER_REPORT,
        ASSET_DECODER_GRAPH,
        ASSET_DECODER_WEIGHTS,
        ASSET_DECODER_REPORT,
        ASSET_COUNT,
    };
    char manifest[16384];
    char digests[ASSET_COUNT][65];
    const char* const paths[ASSET_COUNT] = {
        fixture->config,
        fixture->vocab,
        fixture->encoder_graph,
        fixture->encoder_weights,
        fixture->encoder_report,
        fixture->decoder_graph,
        fixture->decoder_weights,
        fixture->decoder_report,
    };
    const long vocab_bytes = file_size(fixture->vocab);
    const long config_bytes = file_size(fixture->config);
    const long encoder_graph_bytes = file_size(fixture->encoder_graph);
    const long encoder_weights_bytes = file_size(fixture->encoder_weights);
    const long encoder_report_bytes = file_size(fixture->encoder_report);
    const long decoder_graph_bytes = file_size(fixture->decoder_graph);
    const long decoder_weights_bytes = file_size(fixture->decoder_weights);
    const long decoder_report_bytes = file_size(fixture->decoder_report);
    const char* routing = runtime_family ?
        "\"routing\":{\"mode\":\"runtime\",\"family_inputs\":{"
        "\"encoder\":\"enc_family_phys\",\"decoder\":\"dec_family_phys\"}}," :
        "\"routing\":{\"mode\":\"specialized\",\"family_id\":0},";
    const char* generation = direct_logits ?
        "\"generation\":{\"strategy\":\"greedy-autoregressive\","
        "\"decoder_input_length\":192,\"maximum_new_tokens\":191,"
        "\"bos_token_id\":1,\"eos_token_id\":2,\"pad_token_id\":0,"
        "\"logits_row\":\"prefix_length_minus_one\","
        "\"tie_policy\":\"first-index\"}," :
        "\"generation\":{\"strategy\":\"greedy-autoregressive\","
        "\"decoder_input_length\":192,\"maximum_new_tokens\":191,"
        "\"bos_token_id\":1,\"eos_token_id\":2,\"pad_token_id\":0,"
        "\"decoder_output\":\"token_ids\","
        "\"token_ids_row\":\"prefix_length_minus_one\","
        "\"tie_policy\":\"first-index\"},";
    const char* tokenizer = bpe ?
        "\"tokenizer\":{\"type\":\"byte_fallback_bpe\",\"version\":1,"
        "\"vocab_size\":1536,\"normalization\":\"NFC\","
        "\"tokenizer_hash\":"
        "\"612e8425883fd7e3f0292912ab39bd44c72a1da9e399ee455639e6e612acb939\","
        "\"itos_key\":\"itos\",\"merges_key\":\"merges\","
        "\"token_ids\":{\"pad\":0,\"bos\":1,\"eos\":2,\"unk\":3}}," :
        "\"tokenizer\":{\"type\":\"char-vocab\",\"version\":1,"
        "\"itos_key\":\"itos\","
        "\"token_ids\":{\"pad\":0,\"bos\":1,\"eos\":2,\"unk\":3}},";
    int count;

    CHECK(vocab_bytes > 0 && config_bytes > 0 && encoder_graph_bytes > 0 &&
          encoder_weights_bytes > 0 && encoder_report_bytes > 0 &&
          decoder_graph_bytes > 0 && decoder_weights_bytes > 0 &&
          decoder_report_bytes > 0);
    for (int index = 0; index < ASSET_COUNT; index++) {
        CHECK(tiny_receipt_split_w8a8_sha256_file(paths[index], digests[index]) == 0);
    }
    count = snprintf(
        manifest, sizeof(manifest),
        "{\"format\":\"%s\","
        "\"assets\":{"
        "\"config\":{\"path\":\"config.json\",\"bytes\":%ld,\"sha256\":\"%s\"},"
        "\"vocab\":{\"path\":\"vocab.json\",\"bytes\":%ld,\"sha256\":\"%s\"}},"
        "%s"
        "\"preprocessing\":{\"layout\":\"NCHW\",\"shape\":[1,1,320,672],"
        "\"color\":\"grayscale\",\"resize\":{\"width\":672,\"height\":320,"
        "\"method\":\"bilinear\"},\"normalization\":\"(x / 255 - 0.5) / 0.5\"},"
        "\"families\":{\"auto_id\":-1,\"ordered_names\":[\"phone\",\"address\","
        "\"store\",\"item_row\",\"item_math\",\"item_lookup\",\"math\",\"other\"],"
        "\"name_to_id\":{\"phone\":0,\"address\":1,\"store\":2,\"item_row\":3,"
        "\"item_math\":4,\"item_lookup\":5,\"math\":6,\"other\":7}},"
        "%s%s"
        "\"graphs\":{"
        "\"encoder\":{"
        "\"graph\":{\"path\":\"encoder/graph.json\",\"bytes\":%ld,\"sha256\":\"%s\"},"
        "\"weights\":{\"path\":\"encoder/model.safetensors\",\"bytes\":%ld,"
        "\"sha256\":\"%s\"},"
        "\"export_report\":{\"path\":\"encoder/export_report.json\",\"bytes\":%ld,"
        "\"sha256\":\"%s\"},"
        "\"inputs\":{\"image\":\"enc_pixels_phys\","
        "\"question_ids\":\"enc_question_phys\"%s},"
        "\"outputs\":{\"memory\":\"enc_memory_phys\","
        "\"memory_padding_mask\":\"enc_mask_phys\","
        "\"router_logits\":\"enc_router_phys\","
        "\"selected_family_ids\":\"enc_selected_phys\"}},"
        "\"decoder\":{"
        "\"graph\":{\"path\":\"decoder/graph.json\",\"bytes\":%ld,\"sha256\":\"%s\"},"
        "\"weights\":{\"path\":\"decoder/model.safetensors\",\"bytes\":%ld,"
        "\"sha256\":\"%s\"},"
        "\"export_report\":{\"path\":\"decoder/export_report.json\",\"bytes\":%ld,"
        "\"sha256\":\"%s\"},"
        "\"inputs\":{\"decoder_input_ids\":\"dec_ids_phys\","
        "\"memory\":\"dec_memory_phys\","
        "\"memory_padding_mask\":\"dec_mask_phys\"%s%s},"
        "\"outputs\":{%s}}},"
        "\"mask_semantics\":{\"memory_padding_mask\":\"nonzero_means_blocked\"}}\n",
        format, config_bytes, digests[ASSET_CONFIG],
        vocab_bytes, digests[ASSET_VOCAB],
        tokenizer, routing, generation,
        encoder_graph_bytes, digests[ASSET_ENCODER_GRAPH],
        encoder_weights_bytes, digests[ASSET_ENCODER_WEIGHTS],
        encoder_report_bytes, digests[ASSET_ENCODER_REPORT],
        runtime_family ? ",\"family_ids\":\"enc_family_phys\"" : "",
        decoder_graph_bytes, digests[ASSET_DECODER_GRAPH],
        decoder_weights_bytes, digests[ASSET_DECODER_WEIGHTS],
        decoder_report_bytes, digests[ASSET_DECODER_REPORT],
        runtime_family ? ",\"family_ids\":\"dec_family_phys\"" : "",
        ",\"v4_keep\":\"dec_keep_phys\"",
        direct_logits ? "\"logits\":\"dec_logits_phys\"" :
                        "\"token_ids\":\"dec_tokens_phys\"");
    CHECK(count > 0 && (size_t)count < sizeof(manifest));
    return write_text(fixture->manifest, manifest);
}

static int create_fixture(SplitFixture* fixture, int runtime_family,
                          int direct_logits) {
    static const char specialized_encoder_graph[] =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"enc_pixels_phys\":{\"shape\":[1,1,320,672],\"dtype\":\"float32\"},"
        "\"enc_question_phys\":{\"shape\":[1,192],\"dtype\":\"int32\"}},"
        "\"nodes\":["
        "{\"opType\":\"Identity\",\"inputs\":{\"input\":\"memory_value\"},"
        "\"outputs\":{\"out\":\"enc_memory_phys\"},"
        "\"outputs_shape\":{\"out\":[1,402,320]},"
        "\"outputs_dtype\":{\"out\":\"float32\"},\"params\":{}},"
        "{\"opType\":\"Identity\",\"inputs\":{\"input\":\"mask_value\"},"
        "\"outputs\":{\"out\":\"enc_mask_phys\"},"
        "\"outputs_shape\":{\"out\":[1,402]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},\"params\":{}},"
        "{\"opType\":\"Identity\",\"inputs\":{\"input\":\"router_value\"},"
        "\"outputs\":{\"out\":\"enc_router_phys\"},"
        "\"outputs_shape\":{\"out\":[1,8]},"
        "\"outputs_dtype\":{\"out\":\"float32\"},\"params\":{}},"
        "{\"opType\":\"Identity\",\"inputs\":{\"input\":\"family_value\"},"
        "\"outputs\":{\"out\":\"enc_selected_phys\"},"
        "\"outputs_shape\":{\"out\":[1]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},\"params\":{}}],"
        "\"outputs\":[\"enc_memory_phys\",\"enc_mask_phys\",\"enc_router_phys\","
        "\"enc_selected_phys\"]}";
    static const char runtime_encoder_graph[] =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"enc_pixels_phys\":{\"shape\":[1,1,320,672],\"dtype\":\"float32\"},"
        "\"enc_question_phys\":{\"shape\":[1,192],\"dtype\":\"int32\"},"
        "\"enc_family_phys\":{\"shape\":[1],\"dtype\":\"int32\"}},"
        "\"nodes\":["
        "{\"opType\":\"Identity\",\"inputs\":{\"input\":\"memory_value\"},"
        "\"outputs\":{\"out\":\"enc_memory_phys\"},"
        "\"outputs_shape\":{\"out\":[1,402,320]},"
        "\"outputs_dtype\":{\"out\":\"float32\"},\"params\":{}},"
        "{\"opType\":\"Identity\",\"inputs\":{\"input\":\"mask_value\"},"
        "\"outputs\":{\"out\":\"enc_mask_phys\"},"
        "\"outputs_shape\":{\"out\":[1,402]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},\"params\":{}},"
        "{\"opType\":\"Identity\",\"inputs\":{\"input\":\"router_value\"},"
        "\"outputs\":{\"out\":\"enc_router_phys\"},"
        "\"outputs_shape\":{\"out\":[1,8]},"
        "\"outputs_dtype\":{\"out\":\"float32\"},\"params\":{}},"
        "{\"opType\":\"Clip\",\"inputs\":{\"input\":\"enc_family_phys\"},"
        "\"outputs\":{\"out\":\"enc_selected_phys\"},"
        "\"outputs_shape\":{\"out\":[1]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"min\":0,\"max\":7}}],"
        "\"outputs\":[\"enc_memory_phys\",\"enc_mask_phys\",\"enc_router_phys\","
        "\"enc_selected_phys\"]}";
    static const char specialized_decoder_graph[] =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"dec_ids_phys\":{\"shape\":[1,192],\"dtype\":\"int32\"},"
        "\"dec_memory_phys\":{\"shape\":[1,402,320],\"dtype\":\"float32\"},"
        "\"dec_mask_phys\":{\"shape\":[1,402],\"dtype\":\"int32\"},"
        "\"dec_keep_phys\":{\"shape\":[1,192],\"dtype\":\"int32\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"next_token_table\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"table.scale\",\"zero_point_tensor\":\"table.zero_point\"},"
        "\"next_token_logits\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\",\"zero_point_tensor\":\"unit.zero_point\"}}},"
        "\"nodes\":["
        "{\"opType\":\"QEmbedding\","
        "\"inputs\":{\"input\":\"dec_ids_phys\",\"weight\":\"next_token_table\"},"
        "\"outputs\":{\"out\":\"next_token_logits\"},"
        "\"outputs_shape\":{\"out\":[1,192,760]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}},"
        "{\"opType\":\"QArgMax\",\"inputs\":{\"input\":\"next_token_logits\"},"
        "\"outputs\":{\"out\":\"dec_tokens_phys\"},"
        "\"outputs_shape\":{\"out\":[1,192]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},\"params\":{\"axis\":-1}}],"
        "\"outputs\":[\"dec_tokens_phys\"]}";
    static const char runtime_decoder_graph[] =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"dec_ids_phys\":{\"shape\":[1,192],\"dtype\":\"int32\"},"
        "\"dec_memory_phys\":{\"shape\":[1,402,320],\"dtype\":\"float32\"},"
        "\"dec_mask_phys\":{\"shape\":[1,402],\"dtype\":\"int32\"},"
        "\"dec_family_phys\":{\"shape\":[1],\"dtype\":\"int32\"},"
        "\"dec_keep_phys\":{\"shape\":[1,192],\"dtype\":\"int32\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"next_token_table\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"table.scale\",\"zero_point_tensor\":\"table.zero_point\"},"
        "\"next_token_logits\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\",\"zero_point_tensor\":\"unit.zero_point\"}}},"
        "\"nodes\":["
        "{\"opType\":\"QEmbedding\","
        "\"inputs\":{\"input\":\"dec_ids_phys\",\"weight\":\"next_token_table\"},"
        "\"outputs\":{\"out\":\"next_token_logits\"},"
        "\"outputs_shape\":{\"out\":[1,192,760]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}},"
        "{\"opType\":\"QArgMax\",\"inputs\":{\"input\":\"next_token_logits\"},"
        "\"outputs\":{\"out\":\"base_token_ids\"},"
        "\"outputs_shape\":{\"out\":[1,192]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},\"params\":{\"axis\":-1}},"
        "{\"opType\":\"Equal\","
        "\"inputs\":{\"a\":\"dec_family_phys\",\"b\":\"address_family\"},"
        "\"outputs\":{\"out\":\"address_selected\"},"
        "\"outputs_shape\":{\"out\":[1]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},\"params\":{}},"
        "{\"opType\":\"Expand\","
        "\"inputs\":{\"input\":\"address_selected\"},"
        "\"outputs\":{\"out\":\"address_selected_expanded\"},"
        "\"outputs_shape\":{\"out\":[1,192]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},\"params\":{}},"
        "{\"opType\":\"Where\","
        "\"inputs\":{\"condition\":\"address_selected_expanded\","
        "\"x\":\"address_token_ids\",\"y\":\"base_token_ids\"},"
        "\"outputs\":{\"out\":\"dec_tokens_phys\"},"
        "\"outputs_shape\":{\"out\":[1,192]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},\"params\":{}}],"
        "\"outputs\":[\"dec_tokens_phys\"]}";
    static const char direct_logits_decoder_graph[] =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"dec_ids_phys\":{\"shape\":[1,192],\"dtype\":\"int32\"},"
        "\"dec_memory_phys\":{\"shape\":[1,402,320],\"dtype\":\"float32\"},"
        "\"dec_mask_phys\":{\"shape\":[1,402],\"dtype\":\"int32\"},"
        "\"dec_family_phys\":{\"shape\":[1],\"dtype\":\"int32\"},"
        "\"dec_keep_phys\":{\"shape\":[1,192],\"dtype\":\"int32\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"next_token_table\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"table.scale\","
        "\"zero_point_tensor\":\"table.zero_point\"},"
        "\"next_token_logits\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"}}},"
        "\"nodes\":["
        "{\"opType\":\"QEmbedding\","
        "\"inputs\":{\"input\":\"dec_ids_phys\",\"weight\":\"next_token_table\"},"
        "\"outputs\":{\"out\":\"next_token_logits\"},"
        "\"outputs_shape\":{\"out\":[1,192,760]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}},"
        "{\"opType\":\"DequantizeLinear\","
        "\"inputs\":{\"input\":\"next_token_logits\","
        "\"scale\":\"unit.scale\",\"zero_point\":\"unit.zero_point\"},"
        "\"outputs\":{\"out\":\"dec_logits_phys\"},"
        "\"outputs_shape\":{\"out\":[1,192,760]},"
        "\"outputs_dtype\":{\"out\":\"float32\"},\"params\":{}}],"
        "\"outputs\":[\"dec_logits_phys\"]}";
    static const unsigned char png[] = {
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
        0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00,
        0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0xf8, 0xcf, 0xc0, 0xf0,
        0x1f, 0x00, 0x05, 0x00, 0x01, 0xff, 0x89, 0x99, 0x3d, 0x1d, 0x00, 0x00,
        0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
    };

    CHECK(mkdir(fixture->encoder_dir, 0700) == 0);
    CHECK(mkdir(fixture->decoder_dir, 0700) == 0);
    CHECK(write_text(fixture->encoder_graph,
                     runtime_family ? runtime_encoder_graph :
                                      specialized_encoder_graph) == 0);
    CHECK(write_encoder_weights(fixture->encoder_weights) == 0);
    CHECK(write_text(fixture->encoder_report, "{}\n") == 0);
    CHECK(write_text(fixture->decoder_graph,
                     direct_logits ? direct_logits_decoder_graph :
                     (runtime_family ? runtime_decoder_graph :
                                       specialized_decoder_graph)) == 0);
    CHECK(write_decoder_weights(fixture->decoder_weights, runtime_family, 760) == 0);
    CHECK(write_text(fixture->decoder_report, "{}\n") == 0);
    CHECK(write_text(fixture->config, "{}\n") == 0);
    CHECK(write_vocab(fixture->vocab, 0) == 0);
    CHECK(write_bytes(fixture->image, png, sizeof(png)) == 0);
    CHECK(write_manifest(fixture,
        "volvoxai-tiny-receipt-vqa-split-onnx-package-v1", runtime_family,
        direct_logits, 0) == 0);
    return 0;
}

static int test_split_session(void) {
    char template_path[] = "/tmp/volvox-tinyreceipt-split-w8a8-XXXXXX";
    char* root = mkdtemp(template_path);
    SplitFixture fixture;
    int status = -1;
    char* default_argv[12];
    char* incremental_argv[13];
    char* required_row_argv[13];
    char* incremental_required_row_argv[14];
    char* no_kv_argv[13];
    char* ordinary_argv[13];
    char* ordinary_incremental_argv[14];
    char* ordinary_required_row_argv[14];
    char* no_kv_incremental_argv[14];
    char* no_kv_ordinary_argv[14];
    char* no_kv_required_row_argv[14];
    int32_t tokenizer_ids[4];
    size_t tokenizer_count = 0;
    char tokenizer_text[16];

    memset(&fixture, 0, sizeof(fixture));
    if (!root || fixture_paths(&fixture, root) != 0 ||
        create_fixture(&fixture, 0, 0) != 0)
        goto done;
    if (tiny_receipt_split_w8a8_tokenizer_roundtrip(
            fixture.root, "  A\t", tokenizer_ids,
            sizeof(tokenizer_ids) / sizeof(tokenizer_ids[0]),
            &tokenizer_count, tokenizer_text, sizeof(tokenizer_text)) != 0 ||
        tokenizer_count != 2 || tokenizer_ids[0] != 4 ||
        tokenizer_ids[1] != 2 || strcmp(tokenizer_text, "A") != 0)
        goto done;

    default_argv[0] = "tiny_receipt_split_w8a8";
    default_argv[1] = fixture.root;
    default_argv[2] = "--image";
    default_argv[3] = fixture.image;
    default_argv[4] = "--prompt";
    default_argv[5] = "A";
    default_argv[6] = "--max-new";
    default_argv[7] = "2";
    default_argv[8] = "--family";
    default_argv[9] = "phone";
    default_argv[10] = "--cpu";
    default_argv[11] = NULL;
    if (run_and_capture(11, default_argv, 1, "A") != 0) goto done;

    memcpy(incremental_argv, default_argv, 11 * sizeof(*incremental_argv));
    incremental_argv[11] = "--incremental";
    incremental_argv[12] = NULL;
    if (run_and_capture(12, incremental_argv, 1, "A") != 0) goto done;

    memcpy(required_row_argv, default_argv, 11 * sizeof(*required_row_argv));
    required_row_argv[11] = "--require-row";
    required_row_argv[12] = NULL;
    if (run_and_capture(12, required_row_argv, 1, "A") != 0) goto done;

    memcpy(incremental_required_row_argv, default_argv,
           11 * sizeof(*incremental_required_row_argv));
    incremental_required_row_argv[11] = "--incremental";
    incremental_required_row_argv[12] = "--require-row";
    incremental_required_row_argv[13] = NULL;
    if (run_and_capture(13, incremental_required_row_argv, 1, "A") != 0) goto done;

    memcpy(no_kv_argv, default_argv, 11 * sizeof(*no_kv_argv));
    no_kv_argv[11] = "--no-kv";
    no_kv_argv[12] = NULL;
    if (run_and_capture(12, no_kv_argv, 1, "A") != 0) goto done;

    memcpy(ordinary_argv, default_argv, 11 * sizeof(*ordinary_argv));
    ordinary_argv[11] = "--ordinary";
    ordinary_argv[12] = NULL;
    if (run_and_capture(12, ordinary_argv, 1, "A") != 0) goto done;

    memcpy(ordinary_incremental_argv, default_argv,
           11 * sizeof(*ordinary_incremental_argv));
    ordinary_incremental_argv[11] = "--ordinary";
    ordinary_incremental_argv[12] = "--incremental";
    ordinary_incremental_argv[13] = NULL;
    if (tiny_receipt_split_w8a8_run(13, ordinary_incremental_argv) != 2) goto done;

    memcpy(ordinary_required_row_argv, default_argv,
           11 * sizeof(*ordinary_required_row_argv));
    ordinary_required_row_argv[11] = "--ordinary";
    ordinary_required_row_argv[12] = "--require-row";
    ordinary_required_row_argv[13] = NULL;
    if (tiny_receipt_split_w8a8_run(13, ordinary_required_row_argv) != 2) goto done;

    memcpy(no_kv_incremental_argv, default_argv,
           11 * sizeof(*no_kv_incremental_argv));
    no_kv_incremental_argv[11] = "--no-kv";
    no_kv_incremental_argv[12] = "--incremental";
    no_kv_incremental_argv[13] = NULL;
    if (tiny_receipt_split_w8a8_run(13, no_kv_incremental_argv) != 2) goto done;

    memcpy(no_kv_ordinary_argv, default_argv,
           11 * sizeof(*no_kv_ordinary_argv));
    no_kv_ordinary_argv[11] = "--no-kv";
    no_kv_ordinary_argv[12] = "--ordinary";
    no_kv_ordinary_argv[13] = NULL;
    if (tiny_receipt_split_w8a8_run(13, no_kv_ordinary_argv) != 2) goto done;

    memcpy(no_kv_required_row_argv, default_argv,
           11 * sizeof(*no_kv_required_row_argv));
    no_kv_required_row_argv[11] = "--no-kv";
    no_kv_required_row_argv[12] = "--require-row";
    no_kv_required_row_argv[13] = NULL;
    if (tiny_receipt_split_w8a8_run(13, no_kv_required_row_argv) != 2) goto done;

    {
        const long original_size = file_size(fixture.config);
        if (write_text(fixture.config, "[]\n") != 0 ||
            file_size(fixture.config) != original_size ||
            run_and_capture(11, default_argv, 0, NULL) != 0 ||
            write_text(fixture.config, "{}\n") != 0) goto done;
    }

    if (replace_once_in_file(
            fixture.manifest,
            "\"tie_policy\":\"first-index\"",
            "\"tie_policy\":\"first-index\",\"tie_policy\":\"last-index\"") != 0 ||
        run_and_capture(11, default_argv, 0, NULL) != 0 ||
        write_manifest(&fixture,
            "volvoxai-tiny-receipt-vqa-split-onnx-package-v1", 0, 0, 0) != 0)
        goto done;

    if (write_vocab(fixture.vocab, 1) != 0 ||
        write_manifest(&fixture,
            "volvoxai-tiny-receipt-vqa-split-onnx-package-v1", 0, 0, 0) != 0 ||
        run_and_capture(11, default_argv, 0, NULL) != 0 ||
        write_vocab(fixture.vocab, 0) != 0 ||
        write_manifest(&fixture,
            "volvoxai-tiny-receipt-vqa-split-onnx-package-v1", 0, 0, 0) != 0)
        goto done;

    if (write_manifest(&fixture, "volvoxai-tiny-receipt-vqa-split-onnx-package-v0",
                       0, 0, 0) != 0)
        goto done;
    if (run_and_capture(11, default_argv, 0, NULL) != 0) goto done;

    status = 0;
done:
    cleanup_fixture(&fixture);
    return status;
}

static int test_runtime_family_session(void) {
    char template_path[] = "/tmp/volvox-tinyreceipt-split-runtime-family-XXXXXX";
    char* root = mkdtemp(template_path);
    SplitFixture fixture;
    int status = -1;
    char* auto_argv[10];
    char* address_argv[12];
    char* address_required_row_argv[13];

    memset(&fixture, 0, sizeof(fixture));
    if (!root || fixture_paths(&fixture, root) != 0 ||
        create_fixture(&fixture, 1, 0) != 0)
        goto done;

    auto_argv[0] = "tiny_receipt_split_w8a8";
    auto_argv[1] = fixture.root;
    auto_argv[2] = "--image";
    auto_argv[3] = fixture.image;
    auto_argv[4] = "--prompt";
    auto_argv[5] = "A";
    auto_argv[6] = "--max-new";
    auto_argv[7] = "1";
    auto_argv[8] = "--cpu";
    auto_argv[9] = NULL;
    if (run_and_capture(9, auto_argv, 1, "A") != 0) goto done;

    memcpy(address_argv, auto_argv, 8 * sizeof(*address_argv));
    address_argv[8] = "--family";
    address_argv[9] = "address";
    address_argv[10] = "--cpu";
    address_argv[11] = NULL;
    if (run_and_capture(11, address_argv, 1, "fixture-token-005") != 0) goto done;

    memcpy(address_required_row_argv, address_argv,
           11 * sizeof(*address_required_row_argv));
    address_required_row_argv[11] = "--require-row";
    address_required_row_argv[12] = NULL;
    if (run_and_capture(12, address_required_row_argv, 1,
                        "fixture-token-005") != 0) goto done;

    if (replace_once_in_file(fixture.manifest,
                             "\"family_ids\":\"dec_family_phys\",", "") != 0 ||
        run_and_capture(11, address_argv, 0, NULL) != 0 ||
        write_manifest(&fixture,
            "volvoxai-tiny-receipt-vqa-split-onnx-package-v1", 1, 0, 0) != 0)
        goto done;

    status = 0;
done:
    cleanup_fixture(&fixture);
    return status;
}

static int test_direct_logits_session(void) {
    char template_path[] = "/tmp/volvox-tinyreceipt-split-direct-logits-XXXXXX";
    char* root = mkdtemp(template_path);
    SplitFixture fixture;
    int status = -1;
    char* argv[15];

    memset(&fixture, 0, sizeof(fixture));
    if (!root || fixture_paths(&fixture, root) != 0 ||
        create_fixture(&fixture, 1, 1) != 0)
        goto done;

    argv[0] = "tiny_receipt_split_w8a8";
    argv[1] = fixture.root;
    argv[2] = "--image";
    argv[3] = fixture.image;
    argv[4] = "--prompt";
    argv[5] = "A";
    argv[6] = "--max-new";
    argv[7] = "2";
    argv[8] = "--family";
    argv[9] = "phone";
    argv[10] = "--cpu";
    argv[11] = "--incremental";
    argv[12] = "--require-row";
    argv[13] = "--timing";
    argv[14] = NULL;
    if (run_and_capture(14, argv, 1, "A") != 0) goto done;

    if (replace_once_in_file(
            fixture.manifest,
            "\"outputs\":{\"logits\":\"dec_logits_phys\"}",
            "\"outputs\":{\"logits\":\"dec_logits_phys\","
            "\"token_ids\":\"dec_tokens_phys\"}") != 0 ||
        run_and_capture(14, argv, 0, NULL) != 0 ||
        write_manifest(&fixture,
            "volvoxai-tiny-receipt-vqa-split-onnx-package-v1", 1, 1, 0) != 0)
        goto done;

    if (replace_once_in_file(
            fixture.manifest,
            "\"decoder\":\"dec_family_phys\"",
            "\"decoder\":\"wrong_family_input\"") != 0 ||
        run_and_capture(14, argv, 0, NULL) != 0)
        goto done;

    status = 0;
done:
    cleanup_fixture(&fixture);
    return status;
}

static int test_bpe1536_tokenizer_session(void) {
    static const int32_t expected_question[] = {
        286, 287, 19, 217, 191, 248, 174, 153, 2,
    };
    static const int32_t expected_atomic[] = {10, 16, 14, 11, 2};
    static const char original_hash[] =
        "612e8425883fd7e3f0292912ab39bd44c72a1da9e399ee455639e6e612acb939";
    static const char corrupt_hash[] =
        "712e8425883fd7e3f0292912ab39bd44c72a1da9e399ee455639e6e612acb939";
    char template_path[] = "/tmp/volvox-tinyreceipt-split-bpe1536-XXXXXX";
    char* root = mkdtemp(template_path);
    SplitFixture fixture;
    int32_t ids[32];
    size_t ids_count = 0;
    char decoded[128];
    char* argv[14];
    int status = -1;

    memset(&fixture, 0, sizeof(fixture));
    if (!root || fixture_paths(&fixture, root) != 0 ||
        create_fixture(&fixture, 0, 0) != 0 ||
        write_bpe_vocab(fixture.vocab, BPE_FIXTURE_VALID) != 0 ||
        write_manifest(&fixture,
            "volvoxai-tiny-receipt-vqa-split-onnx-package-v1", 0, 0, 1) != 0)
        goto done;

    if (tiny_receipt_split_w8a8_tokenizer_roundtrip(
            fixture.root, "  phone\t7\xc3\xa9\xe2\x98\x83\n", ids,
            sizeof(ids) / sizeof(ids[0]), &ids_count, decoded,
            sizeof(decoded)) != 0 ||
        ids_count != sizeof(expected_question) / sizeof(expected_question[0]) ||
        memcmp(ids, expected_question, sizeof(expected_question)) != 0 ||
        strcmp(decoded, "phone 7\xc3\xa9\xe2\x98\x83") != 0)
        goto done;

    if (tiny_receipt_split_w8a8_tokenizer_roundtrip(
            fixture.root, "<answer>42</answer>", ids,
            sizeof(ids) / sizeof(ids[0]), &ids_count, decoded,
            sizeof(decoded)) != 0 ||
        ids_count != sizeof(expected_atomic) / sizeof(expected_atomic[0]) ||
        memcmp(ids, expected_atomic, sizeof(expected_atomic)) != 0 ||
        strcmp(decoded, "<answer>42</answer>") != 0)
        goto done;

    if (tiny_receipt_split_w8a8_tokenizer_roundtrip(
            fixture.root, "\xc3\xa9", ids, 2, &ids_count, decoded,
            sizeof(decoded)) != 0 || ids_count != 2 || ids[0] != 217 ||
        ids[1] != 2 || strcmp(decoded, "\xef\xbf\xbd") != 0)
        goto done;
    if (tiny_receipt_split_w8a8_tokenizer_roundtrip(
            fixture.root, "\xc3", ids, sizeof(ids) / sizeof(ids[0]),
            &ids_count, decoded, sizeof(decoded)) == 0)
        goto done;

    argv[0] = "tiny_receipt_split_w8a8";
    argv[1] = fixture.root;
    argv[2] = "--image";
    argv[3] = fixture.image;
    argv[4] = "--prompt";
    argv[5] = "phone";
    argv[6] = "--max-new";
    argv[7] = "2";
    argv[8] = "--family";
    argv[9] = "phone";
    argv[10] = "--cpu";
    argv[11] = "--threads";
    argv[12] = "1";
    argv[13] = NULL;
    if (run_and_capture(13, argv, 1, "<field>") != 0) goto done;
    argv[12] = "0";
    if (tiny_receipt_split_w8a8_run(13, argv) != 2) goto done;
    argv[12] = "1";

    if (replace_once_in_file(fixture.vocab, original_hash, corrupt_hash) != 0 ||
        write_manifest(&fixture,
            "volvoxai-tiny-receipt-vqa-split-onnx-package-v1", 0, 0, 1) != 0 ||
        tiny_receipt_split_w8a8_tokenizer_roundtrip(
            fixture.root, "phone", ids, sizeof(ids) / sizeof(ids[0]),
            &ids_count, decoded, sizeof(decoded)) == 0 ||
        write_bpe_vocab(fixture.vocab, BPE_FIXTURE_VALID) != 0 ||
        write_manifest(&fixture,
            "volvoxai-tiny-receipt-vqa-split-onnx-package-v1", 0, 0, 1) != 0)
        goto done;

    {
        static const BpeFixtureVariant tampered[] = {
            BPE_FIXTURE_ATOMIC_SWAPPED,
            BPE_FIXTURE_ATOMIC_DUPLICATE,
            BPE_FIXTURE_ATOMIC_SHORT,
            BPE_FIXTURE_UNUSED_DUPLICATE,
        };
        for (size_t index = 0; index < sizeof(tampered) / sizeof(tampered[0]);
             index++) {
            const char* tampered_hash = bpe_fixture_hash(tampered[index]);
            if (!tampered_hash ||
                write_bpe_vocab(fixture.vocab, tampered[index]) != 0 ||
                write_manifest(&fixture,
                    "volvoxai-tiny-receipt-vqa-split-onnx-package-v1",
                    0, 0, 1) != 0 ||
                replace_once_in_file(fixture.manifest, original_hash,
                                     tampered_hash) != 0 ||
                tiny_receipt_split_w8a8_tokenizer_roundtrip(
                    fixture.root, "phone", ids,
                    sizeof(ids) / sizeof(ids[0]), &ids_count, decoded,
                    sizeof(decoded)) == 0)
                goto done;
        }
    }

    if (write_bpe_vocab(fixture.vocab, BPE_FIXTURE_VALID) != 0 ||
        write_manifest(&fixture,
            "volvoxai-tiny-receipt-vqa-split-onnx-package-v1", 0, 0, 1) != 0)
        goto done;

    if (replace_once_in_file(fixture.manifest,
                             "\"merges_key\":\"merges\",", "") != 0 ||
        tiny_receipt_split_w8a8_tokenizer_roundtrip(
            fixture.root, "phone", ids, sizeof(ids) / sizeof(ids[0]),
            &ids_count, decoded, sizeof(decoded)) == 0)
        goto done;

    status = 0;
done:
    cleanup_fixture(&fixture);
    return status;
}

static int test_bpe1536_direct_logits_session(void) {
    static const char int8_shape_760[] =
        "\"outputs_shape\":{\"out\":[1,192,760]},"
        "\"outputs_dtype\":{\"out\":\"int8\"}";
    static const char int8_shape_1536[] =
        "\"outputs_shape\":{\"out\":[1,192,1536]},"
        "\"outputs_dtype\":{\"out\":\"int8\"}";
    static const char f32_shape_760[] =
        "\"outputs_shape\":{\"out\":[1,192,760]},"
        "\"outputs_dtype\":{\"out\":\"float32\"}";
    static const char f32_shape_1536[] =
        "\"outputs_shape\":{\"out\":[1,192,1536]},"
        "\"outputs_dtype\":{\"out\":\"float32\"}";
    char template_path[] = "/tmp/volvox-tinyreceipt-split-bpe1536-logits-XXXXXX";
    char* root = mkdtemp(template_path);
    SplitFixture fixture;
    char* argv[15];
    int status = -1;

    memset(&fixture, 0, sizeof(fixture));
    if (!root || fixture_paths(&fixture, root) != 0 ||
        create_fixture(&fixture, 1, 1) != 0 ||
        replace_once_in_file(fixture.decoder_graph, int8_shape_760,
                             int8_shape_1536) != 0 ||
        replace_once_in_file(fixture.decoder_graph, f32_shape_760,
                             f32_shape_1536) != 0 ||
        write_decoder_weights(fixture.decoder_weights, 1, 1536) != 0 ||
        write_bpe_vocab(fixture.vocab, BPE_FIXTURE_VALID) != 0 ||
        write_manifest(&fixture,
            "volvoxai-tiny-receipt-vqa-split-onnx-package-v1", 1, 1, 1) != 0)
        goto done;

    argv[0] = "tiny_receipt_split_w8a8";
    argv[1] = fixture.root;
    argv[2] = "--image";
    argv[3] = fixture.image;
    argv[4] = "--prompt";
    argv[5] = "phone";
    argv[6] = "--max-new";
    argv[7] = "2";
    argv[8] = "--family";
    argv[9] = "phone";
    argv[10] = "--cpu";
    argv[11] = "--threads";
    argv[12] = "1";
    argv[13] = "--ordinary";
    argv[14] = NULL;
    if (run_and_capture(14, argv, 1, "<field>") != 0) goto done;

    status = 0;
done:
    cleanup_fixture(&fixture);
    return status;
}

int main(void) {
    CHECK(test_sha256_known_vector() == 0);
    CHECK(test_split_session() == 0);
    CHECK(test_runtime_family_session() == 0);
    CHECK(test_direct_logits_session() == 0);
    CHECK(test_bpe1536_tokenizer_session() == 0);
    CHECK(test_bpe1536_direct_logits_session() == 0);
    puts("TinyReceipt split W8A8 native example tests passed");
    return 0;
}
