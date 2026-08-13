#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif

#include "tiny_receipt_split_w8a8.h"
#include "tiny_receipt_image.h"

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

static int test_image_normalization_f32_contract(void) {
    static const unsigned char rgb[] = {
        64, 64, 64, 128, 128, 128, 191, 191, 191,
    };
    static const uint32_t expected[] = {
        0xbefefefeu, 0x3b808100u, 0x3efeff00u,
    };
    float output[3];
    CHECK(tiny_receipt_rgb_to_grayscale_bilinear(rgb, 3, 1, output, 3, 1) == 0);
    for (size_t index = 0; index < 3; index++) {
        uint32_t bits;
        memcpy(&bits, &output[index], sizeof(bits));
        CHECK(bits == expected[index]);
    }
    return 0;
}

static VxReport strict_report_fixture(const char* backend) {
    VxReport report = VX_REPORT_INIT;
    report.policy_mode = VX_BACKEND_REQUIRE;
    report.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    report.route_attested = 1;
    snprintf(report.backend, sizeof(report.backend), "%s", backend);
    snprintf(report.route_evidence, sizeof(report.route_evidence), "%s",
             "provider=builtin:cpu;nodes=1;selected=1;fallback=0;missing=0");
    snprintf(report.fallback_evidence, sizeof(report.fallback_evidence), "%s",
             "tier=none;operator=none");
    return report;
}

static int test_strict_backend_report_contract(void) {
    VxReport report = strict_report_fixture("cpu");
    CHECK(tiny_receipt_split_w8a8_report_proves_strict_backend(
        &report, "cpu") == 1);
    CHECK(tiny_receipt_split_w8a8_report_proves_strict_backend(
        &report, "cuda") == 0);
    report.operator_fallback = VX_OPERATOR_FALLBACK_ALLOW;
    CHECK(tiny_receipt_split_w8a8_report_proves_strict_backend(
        &report, "cpu") == 0);
    report = strict_report_fixture("cpu");
    report.policy_mode = VX_BACKEND_PREFER;
    CHECK(tiny_receipt_split_w8a8_report_proves_strict_backend(
        &report, "cpu") == 0);
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

static size_t output_line_count(const char* output, const char* expected,
                                int exact) {
    const char* cursor = output;
    const size_t expected_length = strlen(expected);
    size_t count = 0;
    while (cursor && *cursor) {
        const char* newline = strchr(cursor, '\n');
        size_t length = newline ? (size_t)(newline - cursor) : strlen(cursor);
        if (length >= expected_length &&
            memcmp(cursor, expected, expected_length) == 0 &&
            (!exact || length == expected_length))
            count++;
        cursor = newline ? newline + 1 : NULL;
    }
    return count;
}

static int read_capture(FILE* capture, char** output) {
    long output_size;
    char* value = NULL;
    if (!capture || !output || fseek(capture, 0, SEEK_END) != 0 ||
        (output_size = ftell(capture)) < 0 ||
        fseek(capture, 0, SEEK_SET) != 0 ||
        !(value = (char*)malloc((size_t)output_size + 1)) ||
        fread(value, 1, (size_t)output_size, capture) != (size_t)output_size) {
        free(value);
        return -1;
    }
    value[output_size] = 0;
    *output = value;
    return 0;
}

static int run_and_capture_warmup(int argc, char** argv, int warmup_count,
                                  const char* expected_answer) {
    FILE* stdout_capture = NULL;
    FILE* stderr_capture = NULL;
    int saved_stdout = -1;
    int saved_stderr = -1;
    int stdout_redirected = 0;
    int stderr_redirected = 0;
    int run_status;
    int status = -1;
    char* output = NULL;
    char* debug = NULL;
    char count_evidence[96];

    if (warmup_count < 1 || !expected_answer ||
        fflush(stdout) != 0 || fflush(stderr) != 0 ||
        !(stdout_capture = tmpfile()) || !(stderr_capture = tmpfile()) ||
        (saved_stdout = dup(STDOUT_FILENO)) < 0 ||
        (saved_stderr = dup(STDERR_FILENO)) < 0 ||
        dup2(fileno(stdout_capture), STDOUT_FILENO) < 0)
        goto done;
    stdout_redirected = 1;
    if (dup2(fileno(stderr_capture), STDERR_FILENO) < 0) goto done;
    stderr_redirected = 1;
    run_status = tiny_receipt_split_w8a8_run(argc, argv);
    if (fflush(stdout) != 0 || fflush(stderr) != 0 ||
        dup2(saved_stdout, STDOUT_FILENO) < 0 ||
        dup2(saved_stderr, STDERR_FILENO) < 0)
        goto done;
    stdout_redirected = 0;
    stderr_redirected = 0;
    if (run_status != 0 || read_capture(stdout_capture, &output) != 0 ||
        read_capture(stderr_capture, &debug) != 0)
        goto done;
    snprintf(count_evidence, sizeof(count_evidence),
             "count=%d warmup_timed=0 measured_runs=1", warmup_count);
    if (output_line_count(output, expected_answer, 1) != 1 ||
        output_line_count(output, "WARMUP_RESULT ", 0) != 1 ||
        !strstr(output, count_evidence) ||
        !strstr(output,
                "same_runtime=1 same_encoder_context=1 same_decoder_context=1 ") ||
        !strstr(output,
                "strict_no_fallback=1 token_parity=1 cache_parity=1 cache_reset=1 ") ||
        !strstr(output,
                "family_id=0 tokens=2 token_digest=9a76ad00c5544905 ") ||
        !strstr(output,
                "token_ids=4,2 seed_P=1 seed_R=2 last_P=2 last_R=3 cache_preserved=1") ||
        output_line_count(debug, "[debug] tinyreceipt split ABI=", 0) != 1 ||
        output_line_count(debug,
                          "[debug] tinyreceipt split input_f32_sha256=", 0) != 1 ||
        output_line_count(debug,
                          "[debug] tinyreceipt split question_token_ids=", 0) != 1 ||
        output_line_count(debug, "[debug] tinyreceipt split router=", 0) != 1 ||
        output_line_count(debug,
                          "[debug] tinyreceipt split encoder shape ", 0) != 1 ||
        output_line_count(debug,
                          "[debug] tinyreceipt explicit-kv family=phone step=", 0) != 2 ||
        output_line_count(debug,
                          "[debug] tinyreceipt split decoder shape ", 0) != 2 ||
        output_line_count(debug,
                          "[debug] tinyreceipt explicit-kv family=phone tokens=", 0) != 1 ||
        output_line_count(debug,
                          "[debug] tinyreceipt split emitted_token_ids=", 0) != 1 ||
        output_line_count(debug,
                          "[debug] tinyreceipt split shape mode=", 0) != 1) {
        fprintf(stderr,
                "FAIL: warmup output is not uniquely parseable\nstdout:\n%s\nstderr:\n%s",
                output, debug);
        goto done;
    }
    status = 0;

done:
    if (stdout_redirected) {
        (void)fflush(stdout);
        if (saved_stdout >= 0) (void)dup2(saved_stdout, STDOUT_FILENO);
    }
    if (stderr_redirected) {
        (void)fflush(stderr);
        if (saved_stderr >= 0) (void)dup2(saved_stderr, STDERR_FILENO);
    }
    if (saved_stdout >= 0) close(saved_stdout);
    if (saved_stderr >= 0) close(saved_stderr);
    if (stdout_capture) fclose(stdout_capture);
    if (stderr_capture) fclose(stderr_capture);
    free(output);
    free(debug);
    return status;
}

static int run_and_capture(int argc, char** argv, int expected_status,
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
    if (run_status != expected_status ||
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
    const int memory_shape[3] = {1, 210, 320};
    const int mask_shape[2] = {1, 210};
    const int embedding_shape[2] = {1536, 320};
    const int router_shape[2] = {1, 8};
    const int family_shape[1] = {1};
    const int cross_shape[4] = {1, 8, 402, 40};
    const float router_logits[8] = {8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int32_t selected_family[1] = {0};
    float* memory = (float*)calloc(210u * 320u, sizeof(*memory));
    int32_t* mask = (int32_t*)calloc(210u, sizeof(*mask));
    float* embedding = (float*)calloc(1536u * 320u, sizeof(*embedding));
    float* cross = (float*)calloc(8u * 402u * 40u, sizeof(*cross));
    SafetensorsFile file;
    int status = -1;

    if (!memory || !mask || !embedding || !cross) goto done;
    for (size_t index = 0; index < 8u * 402u * 40u; index++)
        cross[index] = 0.25f;
    if (safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0) goto done;
    if (safetensors_add_tensor(&file, "memory_value", SAFETENSORS_DTYPE_F32,
                               memory_shape, 3, memory,
                               210u * 320u * sizeof(*memory)) != 0 ||
        safetensors_add_tensor(&file, "mask_value", SAFETENSORS_DTYPE_I32,
                               mask_shape, 2, mask, 210u * sizeof(*mask)) != 0 ||
        safetensors_add_tensor(&file, "question_embedding",
                               SAFETENSORS_DTYPE_F32,
                               embedding_shape, 2, embedding,
                               1536u * 320u * sizeof(*embedding)) != 0 ||
        safetensors_add_tensor(&file, "router_value", SAFETENSORS_DTYPE_F32,
                               router_shape, 2, router_logits,
                               sizeof(router_logits)) != 0 ||
        safetensors_add_tensor(&file, "family_value", SAFETENSORS_DTYPE_I32,
                               family_shape, 1, selected_family,
                               sizeof(selected_family)) != 0 ||
        safetensors_add_tensor(&file, "cross_value", SAFETENSORS_DTYPE_F32,
                               cross_shape, 4, cross,
                               8u * 402u * 40u * sizeof(*cross)) != 0 ||
        safetensors_save(path, &file) != 0) {
        safetensors_free(&file);
        goto done;
    }
    safetensors_free(&file);
    status = 0;
done:
    free(cross);
    free(embedding);
    free(mask);
    free(memory);
    return status;
}

static int write_decoder_weights(const char* path, int vocab_count,
                                 int bos_next_token) {
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

    if (vocab_count <= 4 || bos_next_token < 0 || bos_next_token >= vocab_count ||
        (size_t)vocab_count > SIZE_MAX / (size_t)vocab_count)
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
    table[(size_t)vocab_count + (size_t)bos_next_token] = 8;
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
        safetensors_add_tensor(&file, "address_family", SAFETENSORS_DTYPE_I32,
                               scalar_shape, 1, address_family,
                               sizeof(address_family)) != 0 ||
        safetensors_add_tensor(&file, "address_token_ids", SAFETENSORS_DTYPE_I32,
                               (const int[]){1, 192}, 2, address_token_ids,
                               sizeof(address_token_ids)) != 0 ||
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

static int write_kv_encoder_graph(const char* path) {
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    if (fputs(
            "{\"format\":\"volvox-graph/v1\","
            "\"dimensions\":{\"B\":{\"min\":1,\"max\":1},"
            "\"Q\":{\"min\":1,\"max\":192},"
            "\"M\":{\"min\":211,\"max\":402}},\"inputs\":{"
            "\"enc_pixels_phys\":{\"shape\":[1,1,320,672],\"dtype\":\"float32\"},"
            "\"enc_question_phys\":{\"shape\":[\"B\",\"Q\"],\"dtype\":\"int32\"},"
            "\"enc_family_phys\":{\"shape\":[\"B\"],\"dtype\":\"int32\"},"
            "\"enc_positions_phys\":{\"shape\":[\"B\",\"Q\"],\"dtype\":\"int32\"}},"
            "\"nodes\":["
            "{\"id\":\"image_rows\",\"opType\":\"Reshape\","
            "\"inputs\":{\"input\":\"enc_pixels_phys\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"image_rows\","
            "\"shape\":[\"B\",672,320],\"dtype\":\"float32\"}},"
            "\"params\":{\"shape\":[1,672,320]}},"
            "{\"id\":\"image_memory\",\"opType\":\"Slice\","
            "\"inputs\":{\"input\":\"image_rows\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"image_memory\","
            "\"shape\":[\"B\",210,320],\"dtype\":\"float32\"}},"
            "\"params\":{\"starts\":[0],\"ends\":[210],\"axes\":[1],\"steps\":[1]}},"
            "{\"id\":\"image_flat\",\"opType\":\"Reshape\","
            "\"inputs\":{\"input\":\"enc_pixels_phys\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"image_flat\","
            "\"shape\":[\"B\",215040],\"dtype\":\"float32\"}},"
            "\"params\":{\"shape\":[1,215040]}},"
            "{\"id\":\"image_mask_values\",\"opType\":\"Slice\","
            "\"inputs\":{\"input\":\"image_flat\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"image_mask_values\","
            "\"shape\":[\"B\",210],\"dtype\":\"float32\"}},"
            "\"params\":{\"starts\":[0],\"ends\":[210],\"axes\":[1],\"steps\":[1]}},"
            "{\"id\":\"image_mask_i32\",\"opType\":\"Cast\","
            "\"inputs\":{\"input\":\"image_mask_values\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"image_mask_i32\","
            "\"shape\":[\"B\",210],\"dtype\":\"int32\"}},"
            "\"params\":{\"to\":\"int32\"}},"
            "{\"id\":\"image_mask\",\"opType\":\"Equal\","
            "\"inputs\":{\"a\":\"image_mask_i32\",\"b\":\"image_mask_i32\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"image_mask\","
            "\"shape\":[\"B\",210],\"dtype\":\"int32\"}},\"params\":{}},"
            "{\"id\":\"question_embedding\",\"opType\":\"Embedding\","
            "\"inputs\":{\"input\":\"enc_question_phys\","
            "\"weight\":\"question_embedding\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"question_memory\","
            "\"shape\":[\"B\",\"Q\",320],\"dtype\":\"float32\"}},\"params\":{}},"
            "{\"id\":\"encoder_memory\",\"opType\":\"Concat\","
            "\"inputs\":{\"input0\":\"image_memory\",\"input1\":\"question_memory\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"enc_memory_phys\","
            "\"shape\":[\"B\",\"M\",320],\"dtype\":\"float32\"}},"
            "\"params\":{\"axis\":1}},"
            "{\"id\":\"question_mask\",\"opType\":\"Equal\","
            "\"inputs\":{\"a\":\"enc_question_phys\",\"b\":\"enc_question_phys\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"question_mask\","
            "\"shape\":[\"B\",\"Q\"],\"dtype\":\"int32\"}},\"params\":{}},"
            "{\"id\":\"encoder_mask\",\"opType\":\"Concat\","
            "\"inputs\":{\"input0\":\"image_mask\",\"input1\":\"question_mask\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"enc_mask_phys\","
            "\"shape\":[\"B\",\"M\"],\"dtype\":\"int32\"}},"
            "\"params\":{\"axis\":1}},"
            "{\"id\":\"encoder_router\",\"opType\":\"Identity\","
            "\"inputs\":{\"input\":\"router_value\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"enc_router_phys\","
            "\"shape\":[1,8],\"dtype\":\"float32\"}},\"params\":{}},"
            "{\"id\":\"encoder_selected\",\"opType\":\"Clip\","
            "\"inputs\":{\"input\":\"enc_family_phys\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"enc_selected_phys\","
            "\"shape\":[1],\"dtype\":\"int32\"}},"
            "\"params\":{\"min\":0,\"max\":7}},"
            "{\"id\":\"encoder_cross_head\",\"opType\":\"Slice\","
            "\"inputs\":{\"input\":\"enc_memory_phys\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"cross_bmh\","
            "\"shape\":[\"B\",\"M\",40],\"dtype\":\"float32\"}},"
            "\"params\":{\"starts\":[0],\"ends\":[40],\"axes\":[2],\"steps\":[1]}},"
            "{\"id\":\"encoder_cross_unsqueeze\",\"opType\":\"Unsqueeze\","
            "\"inputs\":{\"input\":\"cross_bmh\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"cross_b1mh\","
            "\"shape\":[\"B\",1,\"M\",40],\"dtype\":\"float32\"}},"
            "\"params\":{\"axes\":[1]}},"
            "{\"id\":\"encoder_cross_heads\",\"opType\":\"Concat\","
            "\"inputs\":{\"input0\":\"cross_b1mh\",\"input1\":\"cross_b1mh\","
            "\"input2\":\"cross_b1mh\",\"input3\":\"cross_b1mh\","
            "\"input4\":\"cross_b1mh\",\"input5\":\"cross_b1mh\","
            "\"input6\":\"cross_b1mh\",\"input7\":\"cross_b1mh\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"cross_value_dynamic\","
            "\"shape\":[\"B\",8,\"M\",40],\"dtype\":\"float32\"}},"
            "\"params\":{\"axis\":1}}",
            file) == EOF)
        goto fail;
    for (int index = 0; index < 8; index++) {
        const char* kind = index % 2 ? "v" : "k";
        const int layer = index / 2;
        if (fprintf(
                file,
                ",{\"id\":\"encoder_cross_%s_%d\",\"opType\":\"Identity\","
                "\"inputs\":{\"input\":\"cross_value_dynamic\"},"
                "\"outputs\":{\"out\":{\"tensor\":\"enc_cross_%s_%d_phys\","
                "\"shape\":[\"B\",8,\"M\",40],\"dtype\":\"float32\"}},"
                "\"params\":{}}",
                kind, layer, kind, layer) < 0)
            goto fail;
    }
    if (fputs(
            "],\"outputs\":[\"enc_memory_phys\",\"enc_mask_phys\","
            "\"enc_router_phys\",\"enc_selected_phys\"",
            file) == EOF)
        goto fail;
    for (int index = 0; index < 8; index++) {
        if (fprintf(file, ",\"enc_cross_%c_%d_phys\"",
                    index % 2 ? 'v' : 'k', index / 2) < 0)
            goto fail;
    }
    if (fputs("]}\n", file) == EOF || fclose(file) != 0) return -1;
    return 0;

fail:
    fclose(file);
    return -1;
}

static int write_kv_decoder_graph(const char* path) {
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    if (fputs(
            "{\"format\":\"volvox-graph/v1\","
            "\"dimensions\":{\"B\":{\"min\":1,\"max\":1},"
            "\"M\":{\"min\":211,\"max\":402},"
            "\"P\":{\"min\":1,\"max\":191},"
            "\"R\":{\"min\":2,\"max\":192}},\"inputs\":{"
            "\"dec_ids_phys\":{\"shape\":[\"B\",1],\"dtype\":\"int32\"},"
            "\"dec_position_phys\":{\"shape\":[\"B\"],\"dtype\":\"int32\"},"
            "\"dec_family_phys\":{\"shape\":[\"B\"],\"dtype\":\"int32\"},"
            "\"dec_memory_mask_phys\":{\"shape\":[\"B\",\"M\"],\"dtype\":\"int32\"},"
            "\"dec_past_mask_phys\":{\"shape\":[\"B\",\"P\"],\"dtype\":\"int32\"}",
            file) == EOF)
        goto fail;
    for (int index = 0; index < 8; index++) {
        if (fprintf(file,
                    ",\"dec_cross_%c_%d_phys\":{\"shape\":[\"B\",8,\"M\",40],"
                    "\"dtype\":\"float32\"}",
                    index % 2 ? 'v' : 'k', index / 2) < 0)
            goto fail;
    }
    for (int index = 0; index < 8; index++) {
        if (fprintf(file,
                    ",\"dec_past_%c_%d_phys\":{\"shape\":[\"B\",8,\"P\",40],"
                    "\"dtype\":\"float32\"}",
                    index % 2 ? 'v' : 'k', index / 2) < 0)
            goto fail;
    }
    if (fputs(
            "},\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
            "\"tensors\":{\"next_token_table\":{\"scheme\":\"per_axis\","
            "\"axis\":0,\"scale_tensor\":\"table.scale\","
            "\"zero_point_tensor\":\"table.zero_point\"},"
            "\"next_token_logits\":{\"scheme\":\"per_tensor\","
            "\"scale_tensor\":\"unit.scale\","
            "\"zero_point_tensor\":\"unit.zero_point\"}}},"
            "\"nodes\":["
            "{\"id\":\"decoder_embedding\",\"opType\":\"QEmbedding\","
            "\"inputs\":{\"input\":\"dec_ids_phys\","
            "\"weight\":\"next_token_table\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"next_token_logits\","
            "\"shape\":[\"B\",1,1536],\"dtype\":\"int8\"}},\"params\":{}},"
            "{\"id\":\"decoder_dequantize\",\"opType\":\"DequantizeLinear\","
            "\"inputs\":{\"input\":\"next_token_logits\","
            "\"scale\":\"unit.scale\",\"zero_point\":\"unit.zero_point\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"dec_logits_phys\","
            "\"shape\":[\"B\",1,1536],\"dtype\":\"float32\"}},\"params\":{}},"
            "{\"id\":\"current_mask_one\",\"opType\":\"Equal\","
            "\"inputs\":{\"a\":\"dec_ids_phys\",\"b\":\"dec_ids_phys\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"current_mask_one\","
            "\"shape\":[\"B\",1],\"dtype\":\"int32\"}},\"params\":{}},"
            "{\"id\":\"current_mask_zero\",\"opType\":\"Not\","
            "\"inputs\":{\"input\":\"current_mask_one\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"current_mask_zero\","
            "\"shape\":[\"B\",1],\"dtype\":\"int32\"}},\"params\":{}},"
            "{\"id\":\"current_mask\",\"opType\":\"Equal\","
            "\"inputs\":{\"a\":\"dec_ids_phys\",\"b\":\"current_mask_zero\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"current_mask\","
            "\"shape\":[\"B\",1],\"dtype\":\"int32\"}},\"params\":{}},"
            "{\"id\":\"current_cache\",\"opType\":\"Slice\","
            "\"inputs\":{\"input\":\"dec_cross_k_0_phys\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"current_cache\","
            "\"shape\":[\"B\",8,1,40],\"dtype\":\"float32\"}},"
            "\"params\":{\"starts\":[0],\"ends\":[1],\"axes\":[2],\"steps\":[1]}},"
            "{\"id\":\"present_mask\",\"opType\":\"Concat\","
            "\"inputs\":{\"input0\":\"dec_past_mask_phys\","
            "\"input1\":\"current_mask\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"dec_present_mask_phys\","
            "\"shape\":[\"B\",\"R\"],\"dtype\":\"int32\"}},"
            "\"params\":{\"axis\":1}}",
            file) == EOF)
        goto fail;
    for (int index = 0; index < 8; index++) {
        const char kind = index % 2 ? 'v' : 'k';
        const int layer = index / 2;
        if (fprintf(
                file,
                ",{\"id\":\"present_%c_%d\",\"opType\":\"Concat\","
                "\"inputs\":{\"input0\":\"dec_past_%c_%d_phys\","
                "\"input1\":\"current_cache\"},"
                "\"outputs\":{\"out\":{\"tensor\":\"dec_present_%c_%d_phys\","
                "\"shape\":[\"B\",8,\"R\",40],\"dtype\":\"float32\"}},"
                "\"params\":{\"axis\":2}}",
                kind, layer, kind, layer, kind, layer) < 0)
            goto fail;
    }
    if (fputs(
            "],\"outputs\":[\"dec_logits_phys\",\"dec_present_mask_phys\"",
            file) == EOF)
        goto fail;
    for (int index = 0; index < 8; index++) {
        if (fprintf(file, ",\"dec_present_%c_%d_phys\"",
                    index % 2 ? 'v' : 'k', index / 2) < 0)
            goto fail;
    }
    if (fputs("]}\n", file) == EOF || fclose(file) != 0) return -1;
    return 0;

fail:
    fclose(file);
    return -1;
}

static int write_kv_manifest(const SplitFixture* fixture) {
    enum {
        KV_ASSET_CONFIG,
        KV_ASSET_VOCAB,
        KV_ASSET_ENCODER_GRAPH,
        KV_ASSET_ENCODER_WEIGHTS,
        KV_ASSET_ENCODER_REPORT,
        KV_ASSET_DECODER_GRAPH,
        KV_ASSET_DECODER_WEIGHTS,
        KV_ASSET_DECODER_REPORT,
        KV_ASSET_COUNT,
    };
    const char* const paths[KV_ASSET_COUNT] = {
        fixture->config, fixture->vocab, fixture->encoder_graph,
        fixture->encoder_weights, fixture->encoder_report,
        fixture->decoder_graph, fixture->decoder_weights,
        fixture->decoder_report,
    };
    long bytes[KV_ASSET_COUNT];
    char digests[KV_ASSET_COUNT][65];
    FILE* file = NULL;
    for (int index = 0; index < KV_ASSET_COUNT; index++) {
        bytes[index] = file_size(paths[index]);
        CHECK(bytes[index] > 0);
        CHECK(tiny_receipt_split_w8a8_sha256_file(paths[index], digests[index]) == 0);
    }
    file = fopen(fixture->manifest, "wb");
    if (!file) return -1;
    if (fprintf(
            file,
            "{\"format\":\"volvoxai-tiny-receipt-vqa-split-kv-onnx-package-v2\","
            "\"assets\":{"
            "\"config\":{\"path\":\"config.json\",\"bytes\":%ld,\"sha256\":\"%s\"},"
            "\"vocab\":{\"path\":\"vocab.json\",\"bytes\":%ld,\"sha256\":\"%s\"}},"
            "\"tokenizer\":{\"type\":\"byte_fallback_bpe\",\"version\":1,"
            "\"vocab_size\":1536,\"normalization\":\"NFC\","
            "\"tokenizer_hash\":"
            "\"612e8425883fd7e3f0292912ab39bd44c72a1da9e399ee455639e6e612acb939\","
            "\"itos_key\":\"itos\",\"merges_key\":\"merges\","
            "\"token_ids\":{\"pad\":0,\"bos\":1,\"eos\":2,\"unk\":3}},"
            "\"preprocessing\":{\"layout\":\"NCHW\",\"shape\":[1,1,320,672],"
            "\"color\":\"grayscale\",\"resize\":{\"width\":672,\"height\":320,"
            "\"method\":\"bilinear\"},\"normalization\":\"(x / 255 - 0.5) / 0.5\"},"
            "\"families\":{\"auto_id\":-1,\"ordered_names\":[\"phone\",\"address\","
            "\"store\",\"item_row\",\"item_math\",\"item_lookup\",\"math\",\"other\"],"
            "\"name_to_id\":{\"phone\":0,\"address\":1,\"store\":2,\"item_row\":3,"
            "\"item_math\":4,\"item_lookup\":5,\"math\":6,\"other\":7}},"
            "\"routing\":{\"mode\":\"runtime\",\"family_inputs\":{"
            "\"encoder\":\"enc_family_phys\",\"decoder\":\"dec_family_phys\"}},"
            "\"generation\":{\"strategy\":\"greedy-autoregressive-explicit-kv\","
            "\"maximum_target_length\":192,\"maximum_new_tokens\":191,"
            "\"bos_token_id\":1,\"eos_token_id\":2,\"pad_token_id\":0,"
            "\"logits_row\":\"current_token\",\"tie_policy\":\"first-index\"},"
            "\"shape_contract\":{"
            "\"graph_shape_mode\":\"bounded-explicit-kv-v2\","
            "\"dimensions\":{\"B\":{\"min\":1,\"max\":1},"
            "\"Q\":{\"min\":1,\"max\":192},"
            "\"M\":{\"min\":211,\"max\":402},"
            "\"P\":{\"min\":1,\"max\":191},"
            "\"R\":{\"min\":2,\"max\":192}},"
            "\"fixed_geometry\":{\"image\":[1,1,320,672],\"image_tokens\":210,"
            "\"feature_width\":320,\"attention_heads\":8,"
            "\"attention_head_width\":40,\"decoder_layers\":4,"
            "\"adapter_families\":8},"
            "\"relations\":{"
            "\"encoder_memory\":{\"operator\":\"Concat\",\"axis\":1,"
            "\"fixed_image_tokens\":210,\"dynamic_question_dimension\":\"Q\","
            "\"derived_memory_dimension\":\"M\"},"
            "\"present_cache\":{\"operator\":\"Concat\",\"axis\":2,"
            "\"past_dimension\":\"P\",\"fixed_current_tokens\":1,"
            "\"derived_present_dimension\":\"R\"}},"
            "\"semantic_inputs\":{\"question_position_ids\":{"
            "\"shape\":[\"B\",\"Q\"],\"values\":\"zero_based_contiguous\"}}},"
            "\"cache_contract\":{\"format\":\"masked-zero-sentinel-v1\","
            "\"layers\":4,\"heads\":8,\"head_width\":40,"
            "\"past_dimension\":\"P\",\"present_dimension\":\"R\","
            "\"initial_past_length\":1,\"sentinel_mask_value\":1,"
            "\"cache_dtype\":\"float32\"},"
            "\"graphs\":{\"encoder\":{"
            "\"graph\":{\"path\":\"encoder/graph.json\",\"bytes\":%ld,\"sha256\":\"%s\"},"
            "\"weights\":{\"path\":\"encoder/model.safetensors\",\"bytes\":%ld,"
            "\"sha256\":\"%s\"},"
            "\"export_report\":{\"path\":\"encoder/export_report.json\",\"bytes\":%ld,"
            "\"sha256\":\"%s\"},"
            "\"inputs\":{\"image\":\"enc_pixels_phys\","
            "\"question_ids\":\"enc_question_phys\","
            "\"family_ids\":\"enc_family_phys\","
            "\"question_position_ids\":\"enc_positions_phys\"},"
            "\"outputs\":{\"memory\":\"enc_memory_phys\","
            "\"memory_padding_mask\":\"enc_mask_phys\","
            "\"router_logits\":\"enc_router_phys\","
            "\"selected_family_ids\":\"enc_selected_phys\"",
            bytes[KV_ASSET_CONFIG], digests[KV_ASSET_CONFIG],
            bytes[KV_ASSET_VOCAB], digests[KV_ASSET_VOCAB],
            bytes[KV_ASSET_ENCODER_GRAPH], digests[KV_ASSET_ENCODER_GRAPH],
            bytes[KV_ASSET_ENCODER_WEIGHTS], digests[KV_ASSET_ENCODER_WEIGHTS],
            bytes[KV_ASSET_ENCODER_REPORT], digests[KV_ASSET_ENCODER_REPORT]) < 0)
        goto fail;
    for (int index = 0; index < 8; index++) {
        if (fprintf(file, ",\"cross_%c_%d\":\"enc_cross_%c_%d_phys\"",
                    index % 2 ? 'v' : 'k', index / 2,
                    index % 2 ? 'v' : 'k', index / 2) < 0)
            goto fail;
    }
    if (fprintf(
            file,
            "}},\"decoder\":{"
            "\"graph\":{\"path\":\"decoder/graph.json\",\"bytes\":%ld,\"sha256\":\"%s\"},"
            "\"weights\":{\"path\":\"decoder/model.safetensors\",\"bytes\":%ld,"
            "\"sha256\":\"%s\"},"
            "\"export_report\":{\"path\":\"decoder/export_report.json\",\"bytes\":%ld,"
            "\"sha256\":\"%s\"},"
            "\"inputs\":{\"decoder_input_ids\":\"dec_ids_phys\","
            "\"position_ids\":\"dec_position_phys\","
            "\"family_ids\":\"dec_family_phys\","
            "\"memory_padding_mask\":\"dec_memory_mask_phys\","
            "\"past_padding_mask\":\"dec_past_mask_phys\"",
            bytes[KV_ASSET_DECODER_GRAPH], digests[KV_ASSET_DECODER_GRAPH],
            bytes[KV_ASSET_DECODER_WEIGHTS], digests[KV_ASSET_DECODER_WEIGHTS],
            bytes[KV_ASSET_DECODER_REPORT], digests[KV_ASSET_DECODER_REPORT]) < 0)
        goto fail;
    for (int index = 0; index < 8; index++) {
        if (fprintf(file, ",\"cross_%c_%d\":\"dec_cross_%c_%d_phys\"",
                    index % 2 ? 'v' : 'k', index / 2,
                    index % 2 ? 'v' : 'k', index / 2) < 0)
            goto fail;
    }
    for (int index = 0; index < 8; index++) {
        if (fprintf(file, ",\"past_%c_%d\":\"dec_past_%c_%d_phys\"",
                    index % 2 ? 'v' : 'k', index / 2,
                    index % 2 ? 'v' : 'k', index / 2) < 0)
            goto fail;
    }
    if (fputs(
            "},\"outputs\":{\"logits\":\"dec_logits_phys\","
            "\"present_padding_mask\":\"dec_present_mask_phys\"",
            file) == EOF)
        goto fail;
    for (int index = 0; index < 8; index++) {
        if (fprintf(file, ",\"present_%c_%d\":\"dec_present_%c_%d_phys\"",
                    index % 2 ? 'v' : 'k', index / 2,
                    index % 2 ? 'v' : 'k', index / 2) < 0)
            goto fail;
    }
    if (fputs(
            "}}},\"mask_semantics\":{"
            "\"memory_padding_mask\":\"nonzero_means_blocked\","
            "\"past_padding_mask\":\"nonzero_means_blocked\"}}\n",
            file) == EOF || fclose(file) != 0)
        return -1;
    return 0;

fail:
    fclose(file);
    return -1;
}

static int create_kv_fixture(SplitFixture* fixture, int bos_next_token) {
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
    CHECK(write_kv_encoder_graph(fixture->encoder_graph) == 0);
    CHECK(write_encoder_weights(fixture->encoder_weights) == 0);
    CHECK(write_text(fixture->encoder_report, "{}\n") == 0);
    CHECK(write_kv_decoder_graph(fixture->decoder_graph) == 0);
    CHECK(write_decoder_weights(fixture->decoder_weights, 1536,
                                bos_next_token) == 0);
    CHECK(write_text(fixture->decoder_report, "{}\n") == 0);
    CHECK(write_text(fixture->config, "{}\n") == 0);
    CHECK(write_bpe_vocab(fixture->vocab, BPE_FIXTURE_VALID) == 0);
    CHECK(write_bytes(fixture->image, png, sizeof(png)) == 0);
    CHECK(write_kv_manifest(fixture) == 0);
    return 0;
}

static int test_explicit_kv_two_step_session(void) {
    char template_path[] = "/tmp/volvox-tinyreceipt-explicit-kv-XXXXXX";
    char* root = mkdtemp(template_path);
    SplitFixture fixture;
    char prompt[192];
    const char* prompts[1];
    const int32_t maximum_new_tokens[1] = {2};
    TinyReceiptSplitShapeEvidence evidence[1];
    char* argv[17];
    int status = -1;
    memset(&fixture, 0, sizeof(fixture));
    memset(prompt, 'A', sizeof(prompt) - 1);
    prompt[sizeof(prompt) - 1] = 0;
    prompts[0] = prompt;
    if (!root || fixture_paths(&fixture, root) != 0 ||
        create_kv_fixture(&fixture, 4) != 0 ||
        tiny_receipt_split_w8a8_profile_sequence(
            fixture.root, fixture.image, prompts, maximum_new_tokens, 1,
            evidence) != 0)
        goto done;
    CHECK(evidence[0].question_length == 192);
    CHECK(evidence[0].memory_length == 402);
    CHECK(evidence[0].decoder_seed_past_length == 1);
    CHECK(evidence[0].decoder_seed_present_length == 2);
    CHECK(evidence[0].decoder_step_past_length == 2);
    CHECK(evidence[0].decoder_step_present_length == 3);
    CHECK(evidence[0].explicit_kv_sentinel_preserved == 1);
    CHECK(evidence[0].generated_tokens == 2);
    CHECK(evidence[0].decoder_seed_result_bytes ==
          1536u * sizeof(float) + 2u * sizeof(int32_t) +
              8u * 8u * 2u * 40u * sizeof(float));
    CHECK(evidence[0].decoder_step_result_bytes ==
          1536u * sizeof(float) + 3u * sizeof(int32_t) +
              8u * 8u * 3u * 40u * sizeof(float));

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
    argv[11] = "--shape-mode";
    argv[12] = "maximum-padded";
    argv[13] = NULL;
    if (run_and_capture(13, argv, 0, "<field>") != 0) goto done;
    argv[13] = "--warmup";
    argv[14] = "2";
    argv[15] = "--timing";
    argv[16] = NULL;
    if (run_and_capture_warmup(16, argv, 2, "<field>") != 0) goto done;
    argv[13] = NULL;
    if (replace_once_in_file(fixture.manifest,
                             "\"sentinel_mask_value\":1",
                             "\"sentinel_mask_value\":0") != 0 ||
        run_and_capture(13, argv, 1, NULL) != 0)
        goto done;
    status = 0;

done:
    cleanup_fixture(&fixture);
    return status;
}

static int test_ordinary_cli_defaults_to_strict_cpu(void) {
    char template_path[] = "/tmp/volvox-tinyreceipt-default-cpu-XXXXXX";
    char* root = mkdtemp(template_path);
    SplitFixture fixture;
    char* argv[11];
    int status = -1;

    memset(&fixture, 0, sizeof(fixture));
    if (!root || fixture_paths(&fixture, root) != 0 ||
        create_kv_fixture(&fixture, 4) != 0)
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
    argv[10] = NULL;
    if (run_and_capture(10, argv, 0, "<field>") != 0) goto done;
    status = 0;

done:
    cleanup_fixture(&fixture);
    return status;
}

static int test_explicit_kv_pad_mask_two_step(void) {
    char template_path[] = "/tmp/volvox-tinyreceipt-explicit-kv-pad-XXXXXX";
    char* root = mkdtemp(template_path);
    SplitFixture fixture;
    char prompt[192];
    const char* prompts[1];
    const int32_t maximum_new_tokens[1] = {2};
    TinyReceiptSplitShapeEvidence evidence[1];
    int status = -1;

    memset(&fixture, 0, sizeof(fixture));
    memset(prompt, 'A', sizeof(prompt) - 1);
    prompt[sizeof(prompt) - 1] = 0;
    prompts[0] = prompt;
    if (!root || fixture_paths(&fixture, root) != 0 ||
        create_kv_fixture(&fixture, 0) != 0 ||
        tiny_receipt_split_w8a8_profile_sequence(
            fixture.root, fixture.image, prompts, maximum_new_tokens, 1,
            evidence) != 0)
        goto done;
    CHECK(evidence[0].decoder_seed_past_length == 1);
    CHECK(evidence[0].decoder_seed_present_length == 2);
    CHECK(evidence[0].decoder_step_past_length == 2);
    CHECK(evidence[0].decoder_step_present_length == 3);
    CHECK(evidence[0].explicit_kv_sentinel_preserved == 1);
    CHECK(evidence[0].generated_tokens == 2);
    status = 0;

done:
    cleanup_fixture(&fixture);
    return status;
}

static int test_explicit_kv_context_reuse(void) {
    char template_path[] = "/tmp/volvox-tinyreceipt-explicit-kv-reuse-XXXXXX";
    char* root = mkdtemp(template_path);
    SplitFixture fixture;
    char prompt_a[192];
    char prompt_b[192];
    char prompt_c[192];
    const char* prompts[] = {prompt_a, prompt_b, prompt_c};
    const int32_t maximum_new_tokens[] = {2, 2, 2};
    TinyReceiptSplitShapeEvidence evidence[3];
    int status = -1;

    memset(&fixture, 0, sizeof(fixture));
    memset(prompt_a, 'A', sizeof(prompt_a) - 1);
    memset(prompt_b, 'B', sizeof(prompt_b) - 1);
    memset(prompt_c, 'C', sizeof(prompt_c) - 1);
    prompt_a[sizeof(prompt_a) - 1] = 0;
    prompt_b[sizeof(prompt_b) - 1] = 0;
    prompt_c[sizeof(prompt_c) - 1] = 0;
    if (!root || fixture_paths(&fixture, root) != 0 ||
        create_kv_fixture(&fixture, 4) != 0 ||
        tiny_receipt_split_w8a8_profile_sequence(
            fixture.root, fixture.image, prompts, maximum_new_tokens, 3,
            evidence) != 0)
        goto done;
    for (size_t index = 0; index < 3; index++) {
        CHECK(evidence[index].question_length > 0);
        CHECK(evidence[index].memory_length ==
              evidence[index].question_length + 210);
        CHECK(evidence[index].decoder_seed_past_length == 1);
        CHECK(evidence[index].decoder_seed_present_length == 2);
        CHECK(evidence[index].decoder_step_past_length == 2);
        CHECK(evidence[index].decoder_step_present_length == 3);
        CHECK(evidence[index].explicit_kv_sentinel_preserved == 1);
        CHECK(evidence[index].generated_tokens == 2);
    }
    status = 0;

done:
    cleanup_fixture(&fixture);
    return status;
}

static int test_explicit_kv_dynamic_qualification(void) {
    char template_path[] = "/tmp/volvox-tinyreceipt-explicit-kv-dynamic-XXXXXX";
    char* root = mkdtemp(template_path);
    SplitFixture fixture;
    const char* prompts[] = {"A", "phone number last one", "phone number last one", "A"};
    const int32_t maximum_new_tokens[] = {2, 2, 2, 2};
    const int32_t shape_modes[] = {0, 0, 1, 0};
    TinyReceiptSplitShapeEvidence evidence[4];
    char* argv[9];
    int status = -1;

    memset(&fixture, 0, sizeof(fixture));
    if (!root || fixture_paths(&fixture, root) != 0 ||
        create_kv_fixture(&fixture, 4) != 0 ||
        tiny_receipt_split_w8a8_qualify_sequence(
            fixture.root, fixture.image, prompts, maximum_new_tokens,
            shape_modes, 4, "cpu", evidence) != 0)
        goto done;
    CHECK(evidence[0].shape_mode == 0);
    CHECK(evidence[1].shape_mode == 0);
    CHECK(evidence[2].shape_mode == 1);
    CHECK(evidence[3].shape_mode == 0);
    CHECK(evidence[0].question_length < evidence[1].question_length);
    CHECK(evidence[1].question_length == evidence[2].question_length);
    CHECK(evidence[1].memory_length == evidence[1].question_length + 210);
    CHECK(evidence[2].memory_length == 402);
    CHECK(evidence[3].question_length == evidence[0].question_length);
    CHECK(evidence[3].memory_length == evidence[0].memory_length);
    CHECK(evidence[0].token_digest == evidence[3].token_digest);
    CHECK(evidence[1].token_digest == evidence[2].token_digest);
    for (size_t index = 0; index < 4; index++) {
        CHECK(evidence[index].selected_family_id == 0);
        CHECK(evidence[index].generated_tokens == 2);
        CHECK(evidence[index].decoder_seed_past_length == 1);
        CHECK(evidence[index].decoder_seed_present_length == 2);
        CHECK(evidence[index].decoder_step_past_length == 2);
        CHECK(evidence[index].decoder_step_present_length == 3);
        CHECK(evidence[index].explicit_kv_sentinel_preserved == 1);
    }

    argv[0] = "tiny_receipt_split_w8a8";
    argv[1] = fixture.root;
    argv[2] = "--image";
    argv[3] = fixture.image;
    argv[4] = "--qualify-dynamic";
    argv[5] = NULL;
    if (run_and_capture(5, argv, 2, NULL) != 0) goto done;
    argv[5] = "--cpu";
    argv[6] = NULL;
    if (run_and_capture(
            6, argv, 0,
            "DYNAMIC_REBIND_RESULT status=pass backend=cpu timed=0 "
            "same_runtime=1 same_encoder_context=1 same_decoder_context=1 "
            "strict_no_fallback=1 cpu_threads=1") != 0)
        goto done;
    argv[6] = "--threads";
    argv[7] = "3";
    argv[8] = NULL;
    if (run_and_capture(
            8, argv, 0,
            "DYNAMIC_REBIND_RESULT status=pass backend=cpu timed=0 "
            "same_runtime=1 same_encoder_context=1 same_decoder_context=1 "
            "strict_no_fallback=1 cpu_threads=3") != 0)
        goto done;
    status = 0;

done:
    cleanup_fixture(&fixture);
    return status;
}

static int test_bpe1536_tokenizer_v1(void) {
    static const int32_t expected_question[] = {
        286, 287, 19, 217, 191, 248, 174, 153, 2,
    };
    static const int32_t expected_atomic[] = {10, 16, 14, 11, 2};
    static const char original_hash[] =
        "612e8425883fd7e3f0292912ab39bd44c72a1da9e399ee455639e6e612acb939";
    static const char corrupt_hash[] =
        "712e8425883fd7e3f0292912ab39bd44c72a1da9e399ee455639e6e612acb939";
    char template_path[] = "/tmp/volvox-tinyreceipt-explicit-kv-bpe-XXXXXX";
    char* root = mkdtemp(template_path);
    SplitFixture fixture;
    int32_t ids[32];
    size_t ids_count = 0;
    char decoded[128];
    int status = -1;

    memset(&fixture, 0, sizeof(fixture));
    if (!root || fixture_paths(&fixture, root) != 0 ||
        create_kv_fixture(&fixture, 4) != 0)
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

    if (replace_once_in_file(fixture.vocab, original_hash, corrupt_hash) != 0 ||
        write_kv_manifest(&fixture) != 0 ||
        tiny_receipt_split_w8a8_tokenizer_roundtrip(
            fixture.root, "phone", ids, sizeof(ids) / sizeof(ids[0]),
            &ids_count, decoded, sizeof(decoded)) == 0)
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
                write_kv_manifest(&fixture) != 0 ||
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
        write_kv_manifest(&fixture) != 0 ||
        replace_once_in_file(fixture.manifest,
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

static int test_explicit_kv_manifest_contract(void) {
    char template_path[] = "/tmp/volvox-tinyreceipt-explicit-kv-manifest-XXXXXX";
    char* root = mkdtemp(template_path);
    SplitFixture fixture;
    char prompt[192];
    char* argv[10];
    int status = -1;

    memset(&fixture, 0, sizeof(fixture));
    memset(prompt, 'A', sizeof(prompt) - 1);
    prompt[sizeof(prompt) - 1] = 0;
    if (!root || fixture_paths(&fixture, root) != 0 ||
        create_kv_fixture(&fixture, 4) != 0)
        goto done;
    argv[0] = "tiny_receipt_split_w8a8";
    argv[1] = fixture.root;
    argv[2] = "--image";
    argv[3] = fixture.image;
    argv[4] = "--prompt";
    argv[5] = prompt;
    argv[6] = "--max-new";
    argv[7] = "2";
    argv[8] = "--cpu";
    argv[9] = NULL;

    if (write_text(fixture.config, "[]\n") != 0 ||
        run_and_capture(9, argv, 1, NULL) != 0 ||
        write_text(fixture.config, "{}\n") != 0 ||
        write_kv_manifest(&fixture) != 0)
        goto done;
    if (replace_once_in_file(
            fixture.manifest,
            "\"shape_contract\":{\"graph_shape_mode\"",
            "\"shape_contract\":{\"shape_system\":"
            "\"volvox-bounded-shape/v1\",\"graph_shape_mode\"") != 0 ||
        run_and_capture(9, argv, 1, NULL) != 0 ||
        write_kv_manifest(&fixture) != 0)
        goto done;
    if (replace_once_in_file(
            fixture.manifest,
            "\"tie_policy\":\"first-index\"",
            "\"tie_policy\":\"first-index\",\"tie_policy\":\"last-index\"") != 0 ||
        run_and_capture(9, argv, 1, NULL) != 0 ||
        write_kv_manifest(&fixture) != 0)
        goto done;
    if (replace_once_in_file(
            fixture.manifest,
            "\"past_v_3\":\"dec_past_v_3_phys\"",
            "\"past_v_3\":\"dec_past_k_3_phys\"") != 0 ||
        run_and_capture(9, argv, 1, NULL) != 0 ||
        write_kv_manifest(&fixture) != 0)
        goto done;
    if (replace_once_in_file(
            fixture.manifest,
            "\"routing\":{\"mode\":\"runtime\"",
            "\"routing\":{\"mode\":\"specialized\"") != 0 ||
        run_and_capture(9, argv, 1, NULL) != 0)
        goto done;
    status = 0;

done:
    cleanup_fixture(&fixture);
    return status;
}

static int test_wrong_package_format_rejected(void) {
    char template_path[] = "/tmp/volvox-tinyreceipt-explicit-kv-only-XXXXXX";
    char* root = mkdtemp(template_path);
    SplitFixture fixture;
    char* argv[7];
    int status = -1;

    memset(&fixture, 0, sizeof(fixture));
    if (!root || fixture_paths(&fixture, root) != 0 ||
        create_kv_fixture(&fixture, 4) != 0)
        goto done;
    argv[0] = "tiny_receipt_split_w8a8";
    argv[1] = fixture.root;
    argv[2] = "--image";
    argv[3] = fixture.image;
    argv[4] = "--prompt";
    argv[5] = "A";
    argv[6] = NULL;
    if (replace_once_in_file(
            fixture.manifest,
            "volvoxai-tiny-receipt-vqa-split-kv-onnx-package-v2",
            "unsupported-package-format") != 0 ||
        run_and_capture(6, argv, 1, NULL) != 0)
        goto done;
    status = 0;

done:
    cleanup_fixture(&fixture);
    return status;
}

int main(void) {
    CHECK(test_sha256_known_vector() == 0);
    CHECK(test_image_normalization_f32_contract() == 0);
    CHECK(test_strict_backend_report_contract() == 0);
    CHECK(test_explicit_kv_two_step_session() == 0);
    CHECK(test_ordinary_cli_defaults_to_strict_cpu() == 0);
    CHECK(test_explicit_kv_pad_mask_two_step() == 0);
    CHECK(test_explicit_kv_context_reuse() == 0);
    CHECK(test_explicit_kv_dynamic_qualification() == 0);
    CHECK(test_bpe1536_tokenizer_v1() == 0);
    CHECK(test_explicit_kv_manifest_contract() == 0);
    CHECK(test_wrong_package_format_rejected() == 0);
    puts("TinyReceipt split W8A8 native example tests passed");
    return 0;
}
