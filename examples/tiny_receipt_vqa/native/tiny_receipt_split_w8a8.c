#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif

#include "tiny_receipt_split_w8a8.h"

#include "cJSON.h"
#include "tiny_receipt_image.h"
#include "volvoxai.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define SPLIT_PACKAGE_FORMAT "volvoxai-tiny-receipt-vqa-split-kv-onnx-package-v2"
#define SPLIT_FAMILY_COUNT 8
#define SPLIT_BPE_VOCAB_COUNT 1536
#define SPLIT_IMAGE_HEIGHT 320
#define SPLIT_IMAGE_WIDTH 672
#define SPLIT_IMAGE_TOKENS 210
#define SPLIT_QUESTION_LENGTH 192
#define SPLIT_MEMORY_LENGTH 402
#define SPLIT_MODEL_WIDTH 320
#define SPLIT_DECODER_LENGTH 192
#define SPLIT_MAX_NEW_TOKENS 191
#define SPLIT_MAX_WARMUP_RUNS 20
#define SPLIT_MAX_NAME 128
#define SPLIT_MAX_JSON_BYTES (16u * 1024u * 1024u)
#define SPLIT_CACHE_LAYERS 4
#define SPLIT_CACHE_HEADS 8
#define SPLIT_CACHE_HEAD_WIDTH 40
#define SPLIT_CACHE_TENSORS (SPLIT_CACHE_LAYERS * 2)
#define SPLIT_MAX_BINDINGS 24
#define SPLIT_MAX_GRAPH_OUTPUTS 16

static const char* const k_split_family_names[SPLIT_FAMILY_COUNT] = {
    "phone", "address", "store", "item_row", "item_math", "item_lookup", "math", "other",
};

static const char* const k_split_cross_semantics[SPLIT_CACHE_TENSORS] = {
    "cross_k_0", "cross_v_0", "cross_k_1", "cross_v_1",
    "cross_k_2", "cross_v_2", "cross_k_3", "cross_v_3",
};

static const char* const k_split_past_semantics[SPLIT_CACHE_TENSORS] = {
    "past_k_0", "past_v_0", "past_k_1", "past_v_1",
    "past_k_2", "past_v_2", "past_k_3", "past_v_3",
};

static const char* const k_split_present_semantics[SPLIT_CACHE_TENSORS] = {
    "present_k_0", "present_v_0", "present_k_1", "present_v_1",
    "present_k_2", "present_v_2", "present_k_3", "present_v_3",
};

static const char* const k_split_bpe_atomic_tokens[] = {
    "<field>", "</field>", "<value>", "</value>",
    "<op>", "</op>", "<answer>", "</answer>",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
};

typedef enum {
    SPLIT_TOKENIZER_INVALID = 0,
    SPLIT_TOKENIZER_BYTE_FALLBACK_BPE = 1,
} SplitTokenizerKind;

typedef struct {
    int left;
    int right;
    int output;
} SplitBpeMerge;

typedef struct {
    SplitTokenizerKind kind;
    char** items;
    int count;
    int pad;
    int bos;
    int eos;
    int unk;
    unsigned char* is_atomic;
    unsigned char* is_byte;
    unsigned char* is_unused;
    unsigned char* byte_value;
    int byte_ids[256];
    int* atomic_ids;
    int atomic_count;
    SplitBpeMerge* merges;
    int merge_count;
} SplitVocab;

typedef struct {
    char graph[PATH_MAX];
    char weights[PATH_MAX];
    char export_report[PATH_MAX];
    char image[SPLIT_MAX_NAME];
    char question_ids[SPLIT_MAX_NAME];
    char question_position_ids[SPLIT_MAX_NAME];
    char family_ids[SPLIT_MAX_NAME];
    char memory[SPLIT_MAX_NAME];
    char memory_padding_mask[SPLIT_MAX_NAME];
    char router_logits[SPLIT_MAX_NAME];
    char selected_family_ids[SPLIT_MAX_NAME];
    char cross_kv[SPLIT_CACHE_TENSORS][SPLIT_MAX_NAME];
} SplitEncoderDefinition;

typedef struct {
    char graph[PATH_MAX];
    char weights[PATH_MAX];
    char export_report[PATH_MAX];
    char decoder_input_ids[SPLIT_MAX_NAME];
    char memory_padding_mask[SPLIT_MAX_NAME];
    char family_ids[SPLIT_MAX_NAME];
    char logits[SPLIT_MAX_NAME];
    char position_ids[SPLIT_MAX_NAME];
    char past_padding_mask[SPLIT_MAX_NAME];
    char cross_kv[SPLIT_CACHE_TENSORS][SPLIT_MAX_NAME];
    char past_kv[SPLIT_CACHE_TENSORS][SPLIT_MAX_NAME];
    char present_padding_mask[SPLIT_MAX_NAME];
    char present_kv[SPLIT_CACHE_TENSORS][SPLIT_MAX_NAME];
} SplitDecoderDefinition;

typedef struct {
    char package_dir[PATH_MAX];
    char config_path[PATH_MAX];
    char vocab_path[PATH_MAX];
    SplitEncoderDefinition encoder;
    SplitDecoderDefinition decoder;
    SplitTokenizerKind tokenizer_kind;
    int tokenizer_vocab_count;
    char tokenizer_hash[65];
} SplitPackage;

typedef enum {
    SPLIT_SHAPE_MODE_ACTIVE = 0,
    SPLIT_SHAPE_MODE_MAXIMUM_PADDED = 1,
} SplitShapeMode;

typedef struct {
    const char* package_arg;
    const char* image_path;
    const char* prompt;
    const char* family;
    int family_id;
    int max_new;
    int warmup;
    SplitShapeMode shape_mode;
    int timing;
    int qualify_dynamic;
    VxRuntimeOptions runtime_options;
    VxBackendPolicy backend_policy;
    const char* backend_candidates[1];
} SplitCommand;

typedef struct {
    VxModel* model;
    VxCompiledModel* compiled;
    VxExecutionContext* context;
    VxBackendPolicyMode expected_policy_mode;
    VxOperatorFallback expected_operator_fallback;
    char expected_backend[VX_REPORT_BACKEND_CAPACITY];
} SplitGraph;

typedef struct {
    VxTensorBinding values[SPLIT_MAX_BINDINGS];
    size_t count;
} SplitBindingBatch;

typedef struct {
    float* memory;
    int32_t* memory_padding_mask;
    float* cross_kv[SPLIT_CACHE_TENSORS];
    float router_logits[SPLIT_FAMILY_COUNT];
    int selected_family_id;
    int question_length;
    int memory_length;
    double execution_ms;
} SplitEncoderOutput;

typedef struct {
    uint32_t state[8];
    uint64_t total_bytes;
    unsigned char block[64];
    size_t block_length;
} SplitSha256;

static uint32_t split_sha256_rotate_right(uint32_t value, unsigned int count) {
    return (value >> count) | (value << (32u - count));
}

static void split_sha256_compress(SplitSha256* sha, const unsigned char block[64]) {
    static const uint32_t constants[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    uint32_t words[64];
    uint32_t a = sha->state[0];
    uint32_t b = sha->state[1];
    uint32_t c = sha->state[2];
    uint32_t d = sha->state[3];
    uint32_t e = sha->state[4];
    uint32_t f = sha->state[5];
    uint32_t g = sha->state[6];
    uint32_t h = sha->state[7];

    for (size_t index = 0; index < 16; index++) {
        const size_t offset = index * 4;
        words[index] = ((uint32_t)block[offset] << 24) |
                       ((uint32_t)block[offset + 1] << 16) |
                       ((uint32_t)block[offset + 2] << 8) |
                       (uint32_t)block[offset + 3];
    }
    for (size_t index = 16; index < 64; index++) {
        uint32_t first = split_sha256_rotate_right(words[index - 15], 7) ^
                         split_sha256_rotate_right(words[index - 15], 18) ^
                         (words[index - 15] >> 3);
        uint32_t second = split_sha256_rotate_right(words[index - 2], 17) ^
                          split_sha256_rotate_right(words[index - 2], 19) ^
                          (words[index - 2] >> 10);
        words[index] = words[index - 16] + first + words[index - 7] + second;
    }
    for (size_t index = 0; index < 64; index++) {
        uint32_t choose = (e & f) ^ ((~e) & g);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t upper_e = split_sha256_rotate_right(e, 6) ^
                           split_sha256_rotate_right(e, 11) ^
                           split_sha256_rotate_right(e, 25);
        uint32_t upper_a = split_sha256_rotate_right(a, 2) ^
                           split_sha256_rotate_right(a, 13) ^
                           split_sha256_rotate_right(a, 22);
        uint32_t first = h + upper_e + choose + constants[index] + words[index];
        uint32_t second = upper_a + majority;
        h = g;
        g = f;
        f = e;
        e = d + first;
        d = c;
        c = b;
        b = a;
        a = first + second;
    }
    sha->state[0] += a;
    sha->state[1] += b;
    sha->state[2] += c;
    sha->state[3] += d;
    sha->state[4] += e;
    sha->state[5] += f;
    sha->state[6] += g;
    sha->state[7] += h;
}

static void split_sha256_init(SplitSha256* sha) {
    static const uint32_t initial_state[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    memcpy(sha->state, initial_state, sizeof(initial_state));
    sha->total_bytes = 0;
    sha->block_length = 0;
}

static void split_sha256_update(SplitSha256* sha, const unsigned char* bytes,
                                size_t count) {
    sha->total_bytes += (uint64_t)count;
    while (count > 0) {
        size_t space = sizeof(sha->block) - sha->block_length;
        size_t copied = count < space ? count : space;
        memcpy(sha->block + sha->block_length, bytes, copied);
        sha->block_length += copied;
        bytes += copied;
        count -= copied;
        if (sha->block_length == sizeof(sha->block)) {
            split_sha256_compress(sha, sha->block);
            sha->block_length = 0;
        }
    }
}

static void split_sha256_final(SplitSha256* sha, unsigned char digest[32]) {
    const uint64_t bit_count = sha->total_bytes * 8u;
    sha->block[sha->block_length++] = 0x80u;
    if (sha->block_length > 56) {
        memset(sha->block + sha->block_length, 0,
               sizeof(sha->block) - sha->block_length);
        split_sha256_compress(sha, sha->block);
        sha->block_length = 0;
    }
    memset(sha->block + sha->block_length, 0, 56 - sha->block_length);
    for (size_t index = 0; index < 8; index++) {
        sha->block[63 - index] = (unsigned char)(bit_count >> (index * 8));
    }
    split_sha256_compress(sha, sha->block);
    for (size_t index = 0; index < 8; index++) {
        digest[index * 4] = (unsigned char)(sha->state[index] >> 24);
        digest[index * 4 + 1] = (unsigned char)(sha->state[index] >> 16);
        digest[index * 4 + 2] = (unsigned char)(sha->state[index] >> 8);
        digest[index * 4 + 3] = (unsigned char)sha->state[index];
    }
}

int tiny_receipt_split_w8a8_sha256_file(const char* path, char digest_hex[65]) {
    static const char hex[] = "0123456789abcdef";
    unsigned char buffer[8192];
    unsigned char digest[32];
    SplitSha256 sha;
    FILE* file;
    if (!path || !digest_hex) return -1;
    file = fopen(path, "rb");
    if (!file) return -1;
    split_sha256_init(&sha);
    for (;;) {
        size_t count = fread(buffer, 1, sizeof(buffer), file);
        if (count > 0) split_sha256_update(&sha, buffer, count);
        if (count < sizeof(buffer)) {
            if (ferror(file)) {
                fclose(file);
                return -1;
            }
            break;
        }
    }
    if (fclose(file) != 0) return -1;
    split_sha256_final(&sha, digest);
    for (size_t index = 0; index < sizeof(digest); index++) {
        digest_hex[index * 2] = hex[digest[index] >> 4];
        digest_hex[index * 2 + 1] = hex[digest[index] & 0x0fu];
    }
    digest_hex[64] = 0;
    return 0;
}

static int split_sha256_f32_le(const float* values, size_t count,
                               char digest_hex[65]) {
    static const char hex[] = "0123456789abcdef";
    unsigned char digest[32];
    SplitSha256 sha;
    if (!values || !digest_hex) return -1;
    split_sha256_init(&sha);
    for (size_t index = 0; index < count; index++) {
        unsigned char bytes[4];
        uint32_t bits;
        memcpy(&bits, &values[index], sizeof(bits));
        bytes[0] = (unsigned char)bits;
        bytes[1] = (unsigned char)(bits >> 8);
        bytes[2] = (unsigned char)(bits >> 16);
        bytes[3] = (unsigned char)(bits >> 24);
        split_sha256_update(&sha, bytes, sizeof(bytes));
    }
    split_sha256_final(&sha, digest);
    for (size_t index = 0; index < sizeof(digest); index++) {
        digest_hex[index * 2] = hex[digest[index] >> 4];
        digest_hex[index * 2 + 1] = hex[digest[index] & 0x0fu];
    }
    digest_hex[64] = 0;
    return 0;
}

static double split_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void split_help(const char* argv0) {
    printf("Usage: %s <package-dir|package_manifest.json> --image <png|jpg> --prompt <text> [options]\n\n",
           argv0);
    printf("Run a qualified TinyReceiptVQA split encoder/decoder package.\n");
    printf("Requires the explicit KV-cache v2 package ABI.\n");
    printf("F32 logits use host first-index argmax; cache tensors advance explicitly.\n\n");
    printf("Options:\n");
    printf("  --image <file>               Receipt PNG/JPEG (required).\n");
    printf("  --prompt <text>              Valid UTF-8 question already normalized to NFC;\n");
    printf("                               encoded by the packaged tokenizer (required).\n");
    printf("  --family <name|auto>         Request a runtime family or let the router select\n");
    printf("                               (default auto).\n");
    printf("  --max-new <0..191>           Maximum generated tokens (default 191).\n");
    printf("  --warmup <0..20>             Untimed same-context full requests (default 0).\n");
    printf("  --shape-mode <mode>          Bind active Q/M or maximum-padded encoder extents\n");
    printf("                               (decoder advances P/R one token per call; default active).\n");
    printf("  --timing                     Emit application timings without per-node tracing.\n");
    printf("  --qualify-dynamic            Untimed short/grow/maximum-padded/shrink qualification.\n");
    printf("  --threads <n>                Set the positive CPU worker count.\n");
    printf("  --cpu | --vulkan | --opengl | --metal | --cuda\n");
    printf("  --debug\n");
}

static int split_is_dir(const char* path) {
    struct stat st;
    return path && path[0] && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int split_copy_string(char* out, size_t out_size, const char* value,
                             const char* label) {
    size_t length;
    if (!out || out_size == 0 || !value || !value[0] ||
        (length = strlen(value)) >= out_size) {
        fprintf(stderr, "[tinyreceipt] invalid or overlong %s\n", label);
        return -1;
    }
    memcpy(out, value, length + 1);
    return 0;
}

static int split_read_text(const char* path, char** out) {
    FILE* file = NULL;
    long size = 0;
    char* text = NULL;
    if (!path || !out) return -1;
    *out = NULL;
    file = fopen(path, "rb");
    if (!file) return -1;
    if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0 || (unsigned long)size > SPLIT_MAX_JSON_BYTES) {
        fclose(file);
        return -1;
    }
    text = (char*)malloc((size_t)size + 1);
    if (!text) {
        fclose(file);
        return -1;
    }
    if (size > 0 && fread(text, 1, (size_t)size, file) != (size_t)size) {
        free(text);
        fclose(file);
        return -1;
    }
    text[size] = 0;
    fclose(file);
    *out = text;
    return 0;
}

static int split_path_is_within(const char* root, const char* path) {
    size_t root_length;
    if (!root || !path) return 0;
    root_length = strlen(root);
    return strncmp(root, path, root_length) == 0 &&
           (path[root_length] == 0 || path[root_length] == '/');
}

static int split_asset_path_valid(const char* path) {
    const unsigned char* cursor = (const unsigned char*)path;
    size_t component_length = 0;
    if (!cursor || !cursor[0] || cursor[0] == '/' || cursor[0] == '\\') return 0;
    for (;; cursor++) {
        unsigned char ch = *cursor;
        if (ch == '/' || ch == 0) {
            const unsigned char* start = cursor - component_length;
            if (component_length == 0 ||
                (component_length == 1 && start[0] == '.') ||
                (component_length == 2 && start[0] == '.' && start[1] == '.')) return 0;
            component_length = 0;
            if (ch == 0) return 1;
        } else {
            if (!((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
                  (ch >= '0' && ch <= '9') || ch == '.' || ch == '_' ||
                  ch == '-')) return 0;
            component_length++;
        }
    }
}

static int split_resolve_package_file(const char* package_dir,
                                      const char* relative, uint64_t expected_size,
                                      char* out, size_t out_size,
                                      const char* label) {
    char candidate[PATH_MAX];
    char root_real[PATH_MAX];
    char file_real[PATH_MAX];
    struct stat st;
    if (!package_dir || !split_asset_path_valid(relative)) {
        fprintf(stderr, "[tinyreceipt] %s must be a bounded package-relative asset path\n", label);
        return -1;
    }
    if (!realpath(package_dir, root_real) ||
        snprintf(candidate, sizeof(candidate), "%s/%s", root_real, relative) >=
            (int)sizeof(candidate) ||
        !realpath(candidate, file_real) ||
        !split_path_is_within(root_real, file_real) || stat(file_real, &st) != 0 ||
        !S_ISREG(st.st_mode) || st.st_size < 0 ||
        (uint64_t)st.st_size != expected_size) {
        fprintf(stderr, "[tinyreceipt] cannot resolve bounded %s or its byte size differs: %s\n",
                label, relative ? relative : "<null>");
        return -1;
    }
    return split_copy_string(out, out_size, file_real, label);
}

static const cJSON* split_object(const cJSON* object, const char* key,
                                 const char* label) {
    const cJSON* value = object ?
        cJSON_GetObjectItemCaseSensitive((cJSON*)object, key) : NULL;
    if (!cJSON_IsObject(value)) {
        fprintf(stderr, "[tinyreceipt] manifest requires object %s\n", label);
        return NULL;
    }
    return value;
}

static const char* split_string(const cJSON* object, const char* key,
                                const char* label) {
    const cJSON* value = object ?
        cJSON_GetObjectItemCaseSensitive((cJSON*)object, key) : NULL;
    if (!cJSON_IsString(value) || !value->valuestring || !value->valuestring[0]) {
        fprintf(stderr, "[tinyreceipt] manifest requires string %s\n", label);
        return NULL;
    }
    return value->valuestring;
}

static int split_exact_keys(const cJSON* object, const char* const* keys,
                            size_t key_count, const char* label) {
    const cJSON* child;
    size_t count = 0;
    if (!cJSON_IsObject(object)) return -1;
    cJSON_ArrayForEach(child, object) count++;
    if (count != key_count) {
        fprintf(stderr, "[tinyreceipt] %s must contain exactly the qualified contract keys\n", label);
        return -1;
    }
    for (size_t index = 0; index < key_count; index++) {
        if (!cJSON_GetObjectItemCaseSensitive((cJSON*)object, keys[index])) {
            fprintf(stderr, "[tinyreceipt] %s is missing %s\n", label, keys[index]);
            return -1;
        }
    }
    return 0;
}

static int split_json_keys_unique(const cJSON* value) {
    const cJSON* child;
    const cJSON* other;
    if (!value) return 1;
    if (cJSON_IsObject(value)) {
        cJSON_ArrayForEach(child, value) {
            if (!child->string) return 0;
            for (other = child->next; other; other = other->next) {
                if (!other->string || strcmp(child->string, other->string) == 0)
                    return 0;
            }
        }
    }
    if (cJSON_IsObject(value) || cJSON_IsArray(value)) {
        cJSON_ArrayForEach(child, value) {
            if (!split_json_keys_unique(child)) return 0;
        }
    }
    return 1;
}

static int split_integer(const cJSON* object, const char* key, const char* label,
                         int minimum, int maximum, int* out) {
    const cJSON* value = object ?
        cJSON_GetObjectItemCaseSensitive((cJSON*)object, key) : NULL;
    double number;
    int converted;
    if (!cJSON_IsNumber(value) || (number = value->valuedouble) < (double)minimum ||
        number > (double)maximum) {
        fprintf(stderr, "[tinyreceipt] manifest requires integer %s in range\n", label);
        return -1;
    }
    converted = (int)number;
    if (number != (double)converted) {
        fprintf(stderr, "[tinyreceipt] manifest requires integer %s\n", label);
        return -1;
    }
    *out = converted;
    return 0;
}

static int split_positive_u64(const cJSON* object, const char* key,
                              const char* label, uint64_t* out) {
    const cJSON* value = object ?
        cJSON_GetObjectItemCaseSensitive((cJSON*)object, key) : NULL;
    double number;
    uint64_t converted;
    if (!cJSON_IsNumber(value) || (number = value->valuedouble) < 1.0 ||
        number > 9007199254740991.0) {
        fprintf(stderr, "[tinyreceipt] manifest requires positive integer %s\n", label);
        return -1;
    }
    converted = (uint64_t)number;
    if (number != (double)converted) {
        fprintf(stderr, "[tinyreceipt] manifest requires integer %s\n", label);
        return -1;
    }
    *out = converted;
    return 0;
}

static int split_sha256_string_valid(const char* digest) {
    if (!digest || strlen(digest) != 64) return 0;
    for (size_t index = 0; index < 64; index++) {
        if (!((digest[index] >= '0' && digest[index] <= '9') ||
              (digest[index] >= 'a' && digest[index] <= 'f'))) return 0;
    }
    return 1;
}

static int split_parse_asset(const cJSON* value, const char* package_dir,
                             char* out, size_t out_size, const char* label) {
    static const char* const keys[] = {"path", "bytes", "sha256"};
    const char* path;
    const char* sha256;
    char actual_sha256[65];
    uint64_t bytes;
    if (!cJSON_IsObject(value) || split_exact_keys(value, keys, 3, label) != 0) {
        fprintf(stderr, "[tinyreceipt] manifest requires asset record %s\n", label);
        return -1;
    }
    path = split_string(value, "path", label);
    sha256 = split_string(value, "sha256", label);
    if (!path || !sha256 || !split_sha256_string_valid(sha256) ||
        split_positive_u64(value, "bytes", label, &bytes) != 0) {
        fprintf(stderr, "[tinyreceipt] invalid asset metadata %s\n", label);
        return -1;
    }
    if (split_resolve_package_file(package_dir, path, bytes, out, out_size, label) != 0)
        return -1;
    if (tiny_receipt_split_w8a8_sha256_file(out, actual_sha256) != 0) {
        fprintf(stderr, "[tinyreceipt] cannot hash package asset %s: %s\n", label, path);
        return -1;
    }
    if (memcmp(actual_sha256, sha256, 64) != 0) {
        fprintf(stderr, "[tinyreceipt] SHA-256 differs for package asset %s: %s\n",
                label, path);
        return -1;
    }
    return 0;
}

static int split_shape_equals(const cJSON* value, const int* shape, int rank) {
    if (!cJSON_IsArray(value) || cJSON_GetArraySize(value) != rank) return 0;
    for (int axis = 0; axis < rank; axis++) {
        const cJSON* item = cJSON_GetArrayItem((cJSON*)value, axis);
        if (!cJSON_IsNumber(item) || item->valuedouble != (double)shape[axis]) return 0;
    }
    return 1;
}

static int split_symbolic_shape_equals(const cJSON* value,
                                       const char* const* shape, int rank) {
    if (!cJSON_IsArray(value) || cJSON_GetArraySize(value) != rank) return 0;
    for (int axis = 0; axis < rank; axis++) {
        const cJSON* item = cJSON_GetArrayItem((cJSON*)value, axis);
        if (!cJSON_IsString(item) || !item->valuestring ||
            strcmp(item->valuestring, shape[axis]) != 0) return 0;
    }
    return 1;
}

static int split_exact_dimension(const cJSON* dimensions, const char* name,
                                 int minimum, int maximum) {
    static const char* const keys[] = {"min", "max"};
    const cJSON* dimension = dimensions ?
        cJSON_GetObjectItemCaseSensitive((cJSON*)dimensions, name) : NULL;
    int value;
    return !cJSON_IsObject(dimension) ||
        split_exact_keys(dimension, keys, 2, name) != 0 ||
        split_integer(dimension, "min", name, minimum, minimum, &value) != 0 ||
        split_integer(dimension, "max", name, maximum, maximum, &value) != 0;
}

static int split_copy_mapping(const cJSON* mapping, const char* semantic,
                              char* out, size_t out_size, const char* label) {
    const char* value = split_string(mapping, semantic, label);
    return value ? split_copy_string(out, out_size, value, label) : -1;
}

static int split_mappings_unique(const char* const* values, size_t count,
                                 const char* label) {
    for (size_t left = 0; left < count; left++) {
        if (!values[left] || !values[left][0]) return -1;
        for (size_t right = left + 1; right < count; right++) {
            if (!strcmp(values[left], values[right])) {
                fprintf(stderr, "[tinyreceipt] %s semantic tensors may not alias\n",
                        label);
                return -1;
            }
        }
    }
    return 0;
}

static int split_parse_encoder_v1(const cJSON* graph, const char* package_dir,
                                  SplitEncoderDefinition* encoder) {
    static const char* const input_keys[] = {
        "image", "question_ids", "family_ids", "question_position_ids",
    };
    static const char* const output_keys[] = {
        "memory", "memory_padding_mask", "router_logits", "selected_family_ids",
        "cross_k_0", "cross_v_0", "cross_k_1", "cross_v_1",
        "cross_k_2", "cross_v_2", "cross_k_3", "cross_v_3",
    };
    const cJSON* inputs;
    const cJSON* outputs;
    const char* input_values[4];
    const char* output_values[4 + SPLIT_CACHE_TENSORS];
    char label[128];
    if (!graph || !encoder) return -1;
    inputs = split_object(graph, "inputs", "graphs.encoder.inputs");
    outputs = split_object(graph, "outputs", "graphs.encoder.outputs");
    if (!inputs || !outputs ||
        split_exact_keys(inputs, input_keys, 4, "graphs.encoder.inputs") != 0 ||
        split_exact_keys(outputs, output_keys, 4 + SPLIT_CACHE_TENSORS,
                         "graphs.encoder.outputs") != 0 ||
        split_parse_asset(cJSON_GetObjectItemCaseSensitive((cJSON*)graph, "graph"),
                          package_dir, encoder->graph, sizeof(encoder->graph),
                          "graphs.encoder.graph") != 0 ||
        split_parse_asset(cJSON_GetObjectItemCaseSensitive((cJSON*)graph, "weights"),
                          package_dir, encoder->weights, sizeof(encoder->weights),
                          "graphs.encoder.weights") != 0 ||
        split_parse_asset(cJSON_GetObjectItemCaseSensitive((cJSON*)graph,
                                                           "export_report"),
                          package_dir, encoder->export_report,
                          sizeof(encoder->export_report),
                          "graphs.encoder.export_report") != 0 ||
        split_copy_mapping(inputs, "image", encoder->image, sizeof(encoder->image),
                           "graphs.encoder.inputs.image") != 0 ||
        split_copy_mapping(inputs, "question_ids", encoder->question_ids,
                           sizeof(encoder->question_ids),
                           "graphs.encoder.inputs.question_ids") != 0 ||
        split_copy_mapping(inputs, "family_ids", encoder->family_ids,
                           sizeof(encoder->family_ids),
                           "graphs.encoder.inputs.family_ids") != 0 ||
        split_copy_mapping(inputs, "question_position_ids",
                           encoder->question_position_ids,
                           sizeof(encoder->question_position_ids),
                           "graphs.encoder.inputs.question_position_ids") != 0 ||
        split_copy_mapping(outputs, "memory", encoder->memory,
                           sizeof(encoder->memory),
                           "graphs.encoder.outputs.memory") != 0 ||
        split_copy_mapping(outputs, "memory_padding_mask",
                           encoder->memory_padding_mask,
                           sizeof(encoder->memory_padding_mask),
                           "graphs.encoder.outputs.memory_padding_mask") != 0 ||
        split_copy_mapping(outputs, "router_logits", encoder->router_logits,
                           sizeof(encoder->router_logits),
                           "graphs.encoder.outputs.router_logits") != 0 ||
        split_copy_mapping(outputs, "selected_family_ids",
                           encoder->selected_family_ids,
                           sizeof(encoder->selected_family_ids),
                           "graphs.encoder.outputs.selected_family_ids") != 0)
        return -1;
    for (int index = 0; index < SPLIT_CACHE_TENSORS; index++) {
        if (snprintf(label, sizeof(label), "graphs.encoder.outputs.%s",
                     k_split_cross_semantics[index]) >= (int)sizeof(label) ||
            split_copy_mapping(outputs, k_split_cross_semantics[index],
                               encoder->cross_kv[index],
                               sizeof(encoder->cross_kv[index]), label) != 0)
            return -1;
    }
    input_values[0] = encoder->image;
    input_values[1] = encoder->question_ids;
    input_values[2] = encoder->family_ids;
    input_values[3] = encoder->question_position_ids;
    output_values[0] = encoder->memory;
    output_values[1] = encoder->memory_padding_mask;
    output_values[2] = encoder->router_logits;
    output_values[3] = encoder->selected_family_ids;
    for (int index = 0; index < SPLIT_CACHE_TENSORS; index++)
        output_values[4 + index] = encoder->cross_kv[index];
    if (split_mappings_unique(input_values, 4, "encoder input") != 0 ||
        split_mappings_unique(output_values, 4 + SPLIT_CACHE_TENSORS,
                              "encoder output") != 0)
        return -1;
    return 0;
}

static int split_parse_decoder_v1(const cJSON* graph, const char* package_dir,
                                  SplitDecoderDefinition* decoder) {
    static const char* const input_keys[] = {
        "decoder_input_ids", "position_ids", "family_ids",
        "memory_padding_mask", "past_padding_mask",
        "cross_k_0", "cross_v_0", "cross_k_1", "cross_v_1",
        "cross_k_2", "cross_v_2", "cross_k_3", "cross_v_3",
        "past_k_0", "past_v_0", "past_k_1", "past_v_1",
        "past_k_2", "past_v_2", "past_k_3", "past_v_3",
    };
    static const char* const output_keys[] = {
        "logits", "present_padding_mask",
        "present_k_0", "present_v_0", "present_k_1", "present_v_1",
        "present_k_2", "present_v_2", "present_k_3", "present_v_3",
    };
    const cJSON* inputs;
    const cJSON* outputs;
    const char* input_values[5 + SPLIT_CACHE_TENSORS * 2];
    const char* output_values[2 + SPLIT_CACHE_TENSORS];
    char label[128];
    if (!graph || !decoder) return -1;
    inputs = split_object(graph, "inputs", "graphs.decoder.inputs");
    outputs = split_object(graph, "outputs", "graphs.decoder.outputs");
    if (!inputs || !outputs ||
        split_exact_keys(inputs, input_keys, 5 + SPLIT_CACHE_TENSORS * 2,
                         "graphs.decoder.inputs") != 0 ||
        split_exact_keys(outputs, output_keys, 2 + SPLIT_CACHE_TENSORS,
                         "graphs.decoder.outputs") != 0 ||
        split_parse_asset(cJSON_GetObjectItemCaseSensitive((cJSON*)graph, "graph"),
                          package_dir, decoder->graph, sizeof(decoder->graph),
                          "graphs.decoder.graph") != 0 ||
        split_parse_asset(cJSON_GetObjectItemCaseSensitive((cJSON*)graph, "weights"),
                          package_dir, decoder->weights, sizeof(decoder->weights),
                          "graphs.decoder.weights") != 0 ||
        split_parse_asset(cJSON_GetObjectItemCaseSensitive((cJSON*)graph,
                                                           "export_report"),
                          package_dir, decoder->export_report,
                          sizeof(decoder->export_report),
                          "graphs.decoder.export_report") != 0 ||
        split_copy_mapping(inputs, "decoder_input_ids",
                           decoder->decoder_input_ids,
                           sizeof(decoder->decoder_input_ids),
                           "graphs.decoder.inputs.decoder_input_ids") != 0 ||
        split_copy_mapping(inputs, "position_ids", decoder->position_ids,
                           sizeof(decoder->position_ids),
                           "graphs.decoder.inputs.position_ids") != 0 ||
        split_copy_mapping(inputs, "family_ids", decoder->family_ids,
                           sizeof(decoder->family_ids),
                           "graphs.decoder.inputs.family_ids") != 0 ||
        split_copy_mapping(inputs, "memory_padding_mask",
                           decoder->memory_padding_mask,
                           sizeof(decoder->memory_padding_mask),
                           "graphs.decoder.inputs.memory_padding_mask") != 0 ||
        split_copy_mapping(inputs, "past_padding_mask",
                           decoder->past_padding_mask,
                           sizeof(decoder->past_padding_mask),
                           "graphs.decoder.inputs.past_padding_mask") != 0 ||
        split_copy_mapping(outputs, "logits", decoder->logits,
                           sizeof(decoder->logits),
                           "graphs.decoder.outputs.logits") != 0 ||
        split_copy_mapping(outputs, "present_padding_mask",
                           decoder->present_padding_mask,
                           sizeof(decoder->present_padding_mask),
                           "graphs.decoder.outputs.present_padding_mask") != 0)
        return -1;
    for (int index = 0; index < SPLIT_CACHE_TENSORS; index++) {
        if (snprintf(label, sizeof(label), "graphs.decoder.inputs.%s",
                     k_split_cross_semantics[index]) >= (int)sizeof(label) ||
            split_copy_mapping(inputs, k_split_cross_semantics[index],
                               decoder->cross_kv[index],
                               sizeof(decoder->cross_kv[index]), label) != 0 ||
            snprintf(label, sizeof(label), "graphs.decoder.inputs.%s",
                     k_split_past_semantics[index]) >= (int)sizeof(label) ||
            split_copy_mapping(inputs, k_split_past_semantics[index],
                               decoder->past_kv[index],
                               sizeof(decoder->past_kv[index]), label) != 0 ||
            snprintf(label, sizeof(label), "graphs.decoder.outputs.%s",
                     k_split_present_semantics[index]) >= (int)sizeof(label) ||
            split_copy_mapping(outputs, k_split_present_semantics[index],
                               decoder->present_kv[index],
                               sizeof(decoder->present_kv[index]), label) != 0)
            return -1;
    }
    input_values[0] = decoder->decoder_input_ids;
    input_values[1] = decoder->position_ids;
    input_values[2] = decoder->family_ids;
    input_values[3] = decoder->memory_padding_mask;
    input_values[4] = decoder->past_padding_mask;
    for (int index = 0; index < SPLIT_CACHE_TENSORS; index++) {
        input_values[5 + index] = decoder->cross_kv[index];
        input_values[5 + SPLIT_CACHE_TENSORS + index] = decoder->past_kv[index];
    }
    output_values[0] = decoder->logits;
    output_values[1] = decoder->present_padding_mask;
    for (int index = 0; index < SPLIT_CACHE_TENSORS; index++)
        output_values[2 + index] = decoder->present_kv[index];
    if (split_mappings_unique(input_values, 5 + SPLIT_CACHE_TENSORS * 2,
                              "decoder input") != 0 ||
        split_mappings_unique(output_values, 2 + SPLIT_CACHE_TENSORS,
                              "decoder output") != 0)
        return -1;
    return 0;
}

static int split_parse_routing(const cJSON* root, SplitPackage* package) {
    static const char* const routing_keys[] = {"mode", "family_inputs"};
    static const char* const family_input_keys[] = {"encoder", "decoder"};
    const cJSON* routing;
    const cJSON* family_inputs;
    const char* mode;
    const char* encoder_input;
    const char* decoder_input;
    if (!root || !package) return -1;
    routing = split_object(root, "routing", "routing");
    mode = routing ? split_string(routing, "mode", "routing.mode") : NULL;
    if (!routing || !mode) return -1;
    family_inputs = split_object(routing, "family_inputs",
                                 "routing.family_inputs");
    encoder_input = family_inputs ? split_string(
        family_inputs, "encoder", "routing.family_inputs.encoder") : NULL;
    decoder_input = family_inputs ? split_string(
        family_inputs, "decoder", "routing.family_inputs.decoder") : NULL;
    if (strcmp(mode, "runtime") != 0 ||
        split_exact_keys(routing, routing_keys, 2, "routing") != 0 ||
        !family_inputs ||
        split_exact_keys(family_inputs, family_input_keys, 2,
                         "routing.family_inputs") != 0 ||
        !encoder_input || !decoder_input ||
        strcmp(encoder_input, package->encoder.family_ids) != 0 ||
        strcmp(decoder_input, package->decoder.family_ids) != 0) {
        fprintf(stderr,
                "[tinyreceipt] explicit-KV v2 routing must exactly name both graph family_ids inputs\n");
        return -1;
    }
    return 0;
}

static int split_validate_tokenizer(const cJSON* root, SplitPackage* package) {
    static const char* const bpe_keys[] = {
        "type", "version", "vocab_size", "normalization", "tokenizer_hash",
        "itos_key", "merges_key", "token_ids",
    };
    static const char* const token_id_keys[] = {"pad", "bos", "eos", "unk"};
    const cJSON* tokenizer = split_object(root, "tokenizer", "tokenizer");
    const cJSON* token_ids;
    const char* type;
    const char* itos_key;
    const char* merges_key;
    const char* normalization;
    const char* tokenizer_hash;
    int version;
    int vocab_size;
    int pad;
    int bos;
    int eos;
    int unk;
    if (!tokenizer || !package) return -1;
    type = split_string(tokenizer, "type", "tokenizer.type");
    token_ids = split_object(tokenizer, "token_ids", "tokenizer.token_ids");
    itos_key = split_string(tokenizer, "itos_key", "tokenizer.itos_key");
    merges_key = split_string(tokenizer, "merges_key",
                              "tokenizer.merges_key");
    normalization = split_string(tokenizer, "normalization",
                                 "tokenizer.normalization");
    tokenizer_hash = split_string(tokenizer, "tokenizer_hash",
                                  "tokenizer.tokenizer_hash");
    if (!type || strcmp(type, "byte_fallback_bpe") != 0 ||
        split_exact_keys(tokenizer, bpe_keys, 8, "tokenizer") != 0 ||
        !token_ids || !itos_key || strcmp(itos_key, "itos") != 0 ||
        !merges_key || strcmp(merges_key, "merges") != 0 ||
        split_exact_keys(token_ids, token_id_keys, 4,
                         "tokenizer.token_ids") != 0 ||
        !normalization || strcmp(normalization, "NFC") != 0 ||
        !tokenizer_hash || !split_sha256_string_valid(tokenizer_hash) ||
        split_integer(tokenizer, "version", "tokenizer.version", 1, 1,
                      &version) != 0 ||
        split_integer(tokenizer, "vocab_size", "tokenizer.vocab_size",
                      SPLIT_BPE_VOCAB_COUNT, SPLIT_BPE_VOCAB_COUNT,
                      &vocab_size) != 0 ||
        split_integer(token_ids, "pad", "tokenizer.token_ids.pad", 0, 0,
                      &pad) != 0 ||
        split_integer(token_ids, "bos", "tokenizer.token_ids.bos", 1, 1,
                      &bos) != 0 ||
        split_integer(token_ids, "eos", "tokenizer.token_ids.eos", 2, 2,
                      &eos) != 0 ||
        split_integer(token_ids, "unk", "tokenizer.token_ids.unk", 3, 3,
                      &unk) != 0 ||
        split_copy_string(package->tokenizer_hash,
                          sizeof(package->tokenizer_hash), tokenizer_hash,
                          "tokenizer hash") != 0) {
        fprintf(stderr,
                "[tinyreceipt] tokenizer must be the qualified byte_fallback_bpe v1 1536-token NFC contract\n");
        return -1;
    }
    package->tokenizer_kind = SPLIT_TOKENIZER_BYTE_FALLBACK_BPE;
    package->tokenizer_vocab_count = vocab_size;
    return 0;
}

static int split_validate_preprocessing(const cJSON* root) {
    const cJSON* preprocessing = split_object(root, "preprocessing", "preprocessing");
    const cJSON* resize;
    const cJSON* shape;
    const char* layout;
    const char* color;
    const char* method;
    const char* normalization;
    const int expected[] = {1, 1, SPLIT_IMAGE_HEIGHT, SPLIT_IMAGE_WIDTH};
    int value;
    if (!preprocessing) return -1;
    layout = split_string(preprocessing, "layout", "preprocessing.layout");
    color = split_string(preprocessing, "color", "preprocessing.color");
    resize = split_object(preprocessing, "resize", "preprocessing.resize");
    method = resize ? split_string(resize, "method", "preprocessing.resize.method") : NULL;
    normalization = split_string(preprocessing, "normalization",
                                 "preprocessing.normalization");
    shape = cJSON_GetObjectItemCaseSensitive((cJSON*)preprocessing, "shape");
    if (!layout || !color || !resize || !method || !normalization ||
        strcmp(layout, "NCHW") != 0 || strcmp(color, "grayscale") != 0 ||
        strcmp(method, "bilinear") != 0 ||
        strcmp(normalization, "(x / 255 - 0.5) / 0.5") != 0 ||
        !split_shape_equals(shape, expected, 4) ||
        split_integer(resize, "width", "preprocessing.resize.width",
                      SPLIT_IMAGE_WIDTH, SPLIT_IMAGE_WIDTH, &value) != 0 ||
        split_integer(resize, "height", "preprocessing.resize.height",
                      SPLIT_IMAGE_HEIGHT, SPLIT_IMAGE_HEIGHT, &value) != 0) {
        fprintf(stderr, "[tinyreceipt] preprocessing must be grayscale F32 NCHW [1,1,320,672], bilinear, minus-one-one\n");
        return -1;
    }
    return 0;
}

static int split_validate_families(const cJSON* root) {
    const cJSON* families = split_object(root, "families", "families");
    const cJSON* ordered;
    const cJSON* mapping;
    int auto_id;
    if (!families) return -1;
    ordered = cJSON_GetObjectItemCaseSensitive((cJSON*)families, "ordered_names");
    mapping = split_object(families, "name_to_id", "families.name_to_id");
    if (!cJSON_IsArray(ordered) || cJSON_GetArraySize(ordered) != SPLIT_FAMILY_COUNT ||
        !mapping || split_integer(families, "auto_id", "families.auto_id", -1, -1,
                                  &auto_id) != 0) {
        fprintf(stderr, "[tinyreceipt] invalid canonical family declaration\n");
        return -1;
    }
    if (split_exact_keys(mapping, k_split_family_names, SPLIT_FAMILY_COUNT,
                         "families.name_to_id") != 0) return -1;
    for (int index = 0; index < SPLIT_FAMILY_COUNT; index++) {
        const cJSON* item = cJSON_GetArrayItem((cJSON*)ordered, index);
        int id;
        if (!cJSON_IsString(item) || !item->valuestring ||
            strcmp(item->valuestring, k_split_family_names[index]) != 0 ||
            split_integer(mapping, k_split_family_names[index],
                          "families.name_to_id.<name>", index, index, &id) != 0) {
            fprintf(stderr, "[tinyreceipt] families must preserve the canonical eight-family order\n");
            return -1;
        }
    }
    return 0;
}

static int split_validate_generation_v1(const cJSON* root) {
    static const char* const keys[] = {
        "strategy", "maximum_target_length", "maximum_new_tokens",
        "bos_token_id", "eos_token_id", "pad_token_id", "logits_row",
        "tie_policy",
    };
    const cJSON* generation = split_object(root, "generation", "generation");
    const char* strategy;
    const char* logits_row;
    const char* tie_policy;
    int value;
    if (!generation || split_exact_keys(generation, keys, 8, "generation") != 0)
        goto invalid;
    strategy = split_string(generation, "strategy", "generation.strategy");
    logits_row = split_string(generation, "logits_row", "generation.logits_row");
    tie_policy = split_string(generation, "tie_policy", "generation.tie_policy");
    if (!strategy || strcmp(strategy, "greedy-autoregressive-explicit-kv") != 0 ||
        !logits_row || strcmp(logits_row, "current_token") != 0 ||
        !tie_policy || strcmp(tie_policy, "first-index") != 0 ||
        split_integer(generation, "maximum_target_length",
                      "generation.maximum_target_length", SPLIT_DECODER_LENGTH,
                      SPLIT_DECODER_LENGTH, &value) != 0 ||
        split_integer(generation, "maximum_new_tokens",
                      "generation.maximum_new_tokens", SPLIT_MAX_NEW_TOKENS,
                      SPLIT_MAX_NEW_TOKENS, &value) != 0 ||
        split_integer(generation, "bos_token_id", "generation.bos_token_id",
                      1, 1, &value) != 0 ||
        split_integer(generation, "eos_token_id", "generation.eos_token_id",
                      2, 2, &value) != 0 ||
        split_integer(generation, "pad_token_id", "generation.pad_token_id",
                      0, 0, &value) != 0)
        goto invalid;
    return 0;

invalid:
    fprintf(stderr,
            "[tinyreceipt] generation must match the explicit KV-cache v2 contract\n");
    return -1;
}

static int split_validate_mask_semantics_v1(const cJSON* root) {
    static const char* const keys[] = {
        "memory_padding_mask", "past_padding_mask",
    };
    const cJSON* masks = split_object(root, "mask_semantics", "mask_semantics");
    const char* memory = masks ? split_string(
        masks, "memory_padding_mask", "mask_semantics.memory_padding_mask") : NULL;
    const char* past = masks ? split_string(
        masks, "past_padding_mask", "mask_semantics.past_padding_mask") : NULL;
    if (!masks || split_exact_keys(masks, keys, 2, "mask_semantics") != 0 ||
        !memory || strcmp(memory, "nonzero_means_blocked") != 0 ||
        !past || strcmp(past, "nonzero_means_blocked") != 0) {
        fprintf(stderr,
                "[tinyreceipt] explicit KV masks must declare nonzero as blocked\n");
        return -1;
    }
    return 0;
}

static int split_validate_cache_contract_v1(const cJSON* root) {
    static const char* const keys[] = {
        "format", "layers", "heads", "head_width", "past_dimension",
        "present_dimension", "initial_past_length", "sentinel_mask_value",
        "cache_dtype",
    };
    const cJSON* cache = split_object(root, "cache_contract", "cache_contract");
    const char* format;
    const char* past_dimension;
    const char* present_dimension;
    const char* cache_dtype;
    int value;
    if (!cache || split_exact_keys(cache, keys, 9, "cache_contract") != 0)
        goto invalid;
    format = split_string(cache, "format", "cache_contract.format");
    past_dimension = split_string(
        cache, "past_dimension", "cache_contract.past_dimension");
    present_dimension = split_string(
        cache, "present_dimension", "cache_contract.present_dimension");
    cache_dtype = split_string(cache, "cache_dtype", "cache_contract.cache_dtype");
    if (!format || strcmp(format, "masked-zero-sentinel-v1") != 0 ||
        !past_dimension || strcmp(past_dimension, "P") != 0 ||
        !present_dimension || strcmp(present_dimension, "R") != 0 ||
        !cache_dtype || strcmp(cache_dtype, "float32") != 0 ||
        split_integer(cache, "layers", "cache_contract.layers",
                      SPLIT_CACHE_LAYERS, SPLIT_CACHE_LAYERS, &value) != 0 ||
        split_integer(cache, "heads", "cache_contract.heads",
                      SPLIT_CACHE_HEADS, SPLIT_CACHE_HEADS, &value) != 0 ||
        split_integer(cache, "head_width", "cache_contract.head_width",
                      SPLIT_CACHE_HEAD_WIDTH, SPLIT_CACHE_HEAD_WIDTH, &value) != 0 ||
        split_integer(cache, "initial_past_length",
                      "cache_contract.initial_past_length", 1, 1, &value) != 0 ||
        split_integer(cache, "sentinel_mask_value",
                      "cache_contract.sentinel_mask_value", 1, 1, &value) != 0)
        goto invalid;
    return 0;

invalid:
    fprintf(stderr,
            "[tinyreceipt] cache_contract must be the P=1 masked zero-sentinel ABI\n");
    return -1;
}

static int split_validate_shape_contract_v1(const cJSON* root) {
    static const char* const contract_keys[] = {
        "graph_shape_mode", "dimensions", "fixed_geometry", "relations",
        "semantic_inputs",
    };
    static const char* const dimension_keys[] = {"B", "Q", "M", "P", "R"};
    static const char* const geometry_keys[] = {
        "image", "image_tokens", "feature_width", "attention_heads",
        "attention_head_width", "decoder_layers", "adapter_families",
    };
    static const char* const relation_keys[] = {
        "encoder_memory", "present_cache",
    };
    static const char* const encoder_relation_keys[] = {
        "operator", "axis", "fixed_image_tokens", "dynamic_question_dimension",
        "derived_memory_dimension",
    };
    static const char* const present_relation_keys[] = {
        "operator", "axis", "past_dimension", "fixed_current_tokens",
        "derived_present_dimension",
    };
    static const char* const semantic_keys[] = {"question_position_ids"};
    static const char* const position_keys[] = {"shape", "values"};
    static const char* const bq[] = {"B", "Q"};
    const int image_shape[] = {1, 1, SPLIT_IMAGE_HEIGHT, SPLIT_IMAGE_WIDTH};
    const cJSON* contract = split_object(root, "shape_contract", "shape_contract");
    const cJSON* dimensions;
    const cJSON* geometry;
    const cJSON* relations;
    const cJSON* encoder_relation;
    const cJSON* present_relation;
    const cJSON* semantics;
    const cJSON* question_positions;
    const char* graph_mode;
    const char* value;
    int integer;
    if (!contract || split_exact_keys(contract, contract_keys, 5,
                                      "shape_contract") != 0)
        goto invalid;
    graph_mode = split_string(
        contract, "graph_shape_mode", "shape_contract.graph_shape_mode");
    dimensions = split_object(contract, "dimensions", "shape_contract.dimensions");
    geometry = split_object(contract, "fixed_geometry",
                            "shape_contract.fixed_geometry");
    relations = split_object(contract, "relations", "shape_contract.relations");
    semantics = split_object(contract, "semantic_inputs",
                             "shape_contract.semantic_inputs");
    if (!graph_mode || strcmp(graph_mode, "bounded-explicit-kv-v2") != 0 ||
        !dimensions || split_exact_keys(dimensions, dimension_keys, 5,
                                        "shape_contract.dimensions") != 0 ||
        split_exact_dimension(dimensions, "B", 1, 1) ||
        split_exact_dimension(dimensions, "Q", 1, SPLIT_QUESTION_LENGTH) ||
        split_exact_dimension(dimensions, "M", SPLIT_IMAGE_TOKENS + 1,
                              SPLIT_MEMORY_LENGTH) ||
        split_exact_dimension(dimensions, "P", 1, SPLIT_MAX_NEW_TOKENS) ||
        split_exact_dimension(dimensions, "R", 2, SPLIT_DECODER_LENGTH) ||
        !geometry || split_exact_keys(geometry, geometry_keys, 7,
                                      "shape_contract.fixed_geometry") != 0 ||
        !split_shape_equals(cJSON_GetObjectItemCaseSensitive(
                                (cJSON*)geometry, "image"),
                            image_shape, 4) ||
        split_integer(geometry, "image_tokens", "fixed_geometry.image_tokens",
                      SPLIT_IMAGE_TOKENS, SPLIT_IMAGE_TOKENS, &integer) != 0 ||
        split_integer(geometry, "feature_width", "fixed_geometry.feature_width",
                      SPLIT_MODEL_WIDTH, SPLIT_MODEL_WIDTH, &integer) != 0 ||
        split_integer(geometry, "attention_heads", "fixed_geometry.attention_heads",
                      SPLIT_CACHE_HEADS, SPLIT_CACHE_HEADS, &integer) != 0 ||
        split_integer(geometry, "attention_head_width",
                      "fixed_geometry.attention_head_width",
                      SPLIT_CACHE_HEAD_WIDTH, SPLIT_CACHE_HEAD_WIDTH, &integer) != 0 ||
        split_integer(geometry, "decoder_layers", "fixed_geometry.decoder_layers",
                      SPLIT_CACHE_LAYERS, SPLIT_CACHE_LAYERS, &integer) != 0 ||
        split_integer(geometry, "adapter_families",
                      "fixed_geometry.adapter_families", SPLIT_FAMILY_COUNT,
                      SPLIT_FAMILY_COUNT, &integer) != 0 ||
        !relations || split_exact_keys(relations, relation_keys, 2,
                                       "shape_contract.relations") != 0 ||
        !semantics || split_exact_keys(semantics, semantic_keys, 1,
                                       "shape_contract.semantic_inputs") != 0)
        goto invalid;
    encoder_relation = split_object(
        relations, "encoder_memory", "shape_contract.relations.encoder_memory");
    present_relation = split_object(
        relations, "present_cache", "shape_contract.relations.present_cache");
    if (!encoder_relation || split_exact_keys(
            encoder_relation, encoder_relation_keys, 5,
            "shape_contract.relations.encoder_memory") != 0 ||
        !(value = split_string(encoder_relation, "operator",
                               "encoder_memory.operator")) ||
        strcmp(value, "Concat") != 0 ||
        split_integer(encoder_relation, "axis", "encoder_memory.axis", 1, 1,
                      &integer) != 0 ||
        split_integer(encoder_relation, "fixed_image_tokens",
                      "encoder_memory.fixed_image_tokens", SPLIT_IMAGE_TOKENS,
                      SPLIT_IMAGE_TOKENS, &integer) != 0 ||
        !(value = split_string(encoder_relation, "dynamic_question_dimension",
                               "encoder_memory.dynamic_question_dimension")) ||
        strcmp(value, "Q") != 0 ||
        !(value = split_string(encoder_relation, "derived_memory_dimension",
                               "encoder_memory.derived_memory_dimension")) ||
        strcmp(value, "M") != 0 ||
        !present_relation || split_exact_keys(
            present_relation, present_relation_keys, 5,
            "shape_contract.relations.present_cache") != 0 ||
        !(value = split_string(present_relation, "operator",
                               "present_cache.operator")) ||
        strcmp(value, "Concat") != 0 ||
        split_integer(present_relation, "axis", "present_cache.axis", 2, 2,
                      &integer) != 0 ||
        !(value = split_string(present_relation, "past_dimension",
                               "present_cache.past_dimension")) ||
        strcmp(value, "P") != 0 ||
        split_integer(present_relation, "fixed_current_tokens",
                      "present_cache.fixed_current_tokens", 1, 1, &integer) != 0 ||
        !(value = split_string(present_relation, "derived_present_dimension",
                               "present_cache.derived_present_dimension")) ||
        strcmp(value, "R") != 0)
        goto invalid;
    question_positions = split_object(
        semantics, "question_position_ids",
        "shape_contract.semantic_inputs.question_position_ids");
    value = question_positions ? split_string(
        question_positions, "values", "question_position_ids.values") : NULL;
    if (!question_positions || split_exact_keys(
            question_positions, position_keys, 2,
            "shape_contract.semantic_inputs.question_position_ids") != 0 ||
        !value || strcmp(value, "zero_based_contiguous") != 0 ||
        !split_symbolic_shape_equals(cJSON_GetObjectItemCaseSensitive(
            (cJSON*)question_positions, "shape"), bq, 2))
        goto invalid;
    return 0;

invalid:
    fprintf(stderr,
            "[tinyreceipt] shape_contract must be the bounded explicit-KV v2 ABI\n");
    return -1;
}

static int split_load_package(const char* package_arg, SplitPackage* package) {
    char manifest_path[PATH_MAX];
    char manifest_real[PATH_MAX];
    char* text = NULL;
    cJSON* root = NULL;
    const cJSON* assets;
    const cJSON* graphs;
    const char* format;
    int rc = -1;
    if (!package_arg || !package) return -1;
    memset(package, 0, sizeof(*package));
    if (split_is_dir(package_arg)) {
        if (snprintf(manifest_path, sizeof(manifest_path), "%s/package_manifest.json",
                     package_arg) >= (int)sizeof(manifest_path)) {
            fprintf(stderr, "[tinyreceipt] package path is too long\n");
            return -1;
        }
    } else if (split_copy_string(manifest_path, sizeof(manifest_path), package_arg,
                                 "package manifest path") != 0) {
        return -1;
    }
    if (!realpath(manifest_path, manifest_real)) {
        fprintf(stderr, "[tinyreceipt] cannot open package manifest: %s\n", manifest_path);
        return -1;
    }
    {
        char* slash = strrchr(manifest_real, '/');
        if (!slash || slash == manifest_real) {
            fprintf(stderr, "[tinyreceipt] package manifest must have a parent directory\n");
            return -1;
        }
        *slash = 0;
        if (split_copy_string(package->package_dir, sizeof(package->package_dir),
                              manifest_real, "package directory") != 0) return -1;
        *slash = '/';
    }
    if (split_read_text(manifest_real, &text) != 0 ||
        !(root = cJSON_ParseWithOpts(text, NULL, 1))) {
        fprintf(stderr, "[tinyreceipt] cannot parse package manifest: %s\n", manifest_path);
        goto cleanup;
    }
    if (!split_json_keys_unique(root)) {
        fprintf(stderr, "[tinyreceipt] package manifest contains duplicate object keys\n");
        goto cleanup;
    }
    format = split_string(root, "format", "format");
    if (!format || strcmp(format, SPLIT_PACKAGE_FORMAT) != 0) {
        fprintf(stderr,
                "[tinyreceipt] unsupported package format; expected explicit-KV v2\n");
        goto cleanup;
    }
    assets = split_object(root, "assets", "assets");
    graphs = split_object(root, "graphs", "graphs");
    if (!assets || !graphs ||
        split_parse_asset(cJSON_GetObjectItemCaseSensitive((cJSON*)assets, "config"),
                          package->package_dir, package->config_path,
                          sizeof(package->config_path), "assets.config") != 0 ||
        split_parse_asset(cJSON_GetObjectItemCaseSensitive((cJSON*)assets, "vocab"),
                          package->package_dir, package->vocab_path,
                          sizeof(package->vocab_path), "assets.vocab") != 0 ||
        split_validate_tokenizer(root, package) != 0 ||
        split_validate_preprocessing(root) != 0 ||
        split_validate_families(root) != 0)
        goto cleanup;
    if (split_validate_shape_contract_v1(root) != 0 ||
        split_validate_mask_semantics_v1(root) != 0 ||
        split_validate_cache_contract_v1(root) != 0 ||
        split_parse_encoder_v1(
            split_object(graphs, "encoder", "graphs.encoder"),
            package->package_dir, &package->encoder) != 0 ||
        split_parse_decoder_v1(
            split_object(graphs, "decoder", "graphs.decoder"),
            package->package_dir, &package->decoder) != 0 ||
        split_parse_routing(root, package) != 0 ||
        split_validate_generation_v1(root) != 0) {
        goto cleanup;
    }
    rc = 0;

cleanup:
    cJSON_Delete(root);
    free(text);
    return rc;
}

static void split_vocab_free(SplitVocab* vocab) {
    if (!vocab) return;
    for (int index = 0; index < vocab->count; index++) free(vocab->items[index]);
    free(vocab->items);
    free(vocab->is_atomic);
    free(vocab->is_byte);
    free(vocab->is_unused);
    free(vocab->byte_value);
    free(vocab->atomic_ids);
    free(vocab->merges);
    memset(vocab, 0, sizeof(*vocab));
}

static int split_vocab_find(const SplitVocab* vocab, const char* text,
                            size_t length) {
    if (!vocab || !text) return -1;
    for (int index = 0; index < vocab->count; index++) {
        const char* item = vocab->items[index];
        if (item && strlen(item) == length && memcmp(item, text, length) == 0) return index;
    }
    return -1;
}

static int split_utf8_decode(const unsigned char* text, size_t remaining,
                             uint32_t* codepoint, size_t* length) {
    unsigned char first;
    uint32_t value;
    size_t count;
    if (!text || !remaining || !codepoint || !length) return -1;
    first = text[0];
    if (first < 0x80) {
        *codepoint = first;
        *length = 1;
        return 0;
    }
    if (first >= 0xc2 && first <= 0xdf) {
        count = 2;
        value = (uint32_t)(first & 0x1fu);
    } else if (first >= 0xe0 && first <= 0xef) {
        count = 3;
        value = (uint32_t)(first & 0x0fu);
    } else if (first >= 0xf0 && first <= 0xf4) {
        count = 4;
        value = (uint32_t)(first & 0x07u);
    } else {
        return -1;
    }
    if (remaining < count) return -1;
    if ((text[1] & 0xc0u) != 0x80u ||
        (first == 0xe0 && text[1] < 0xa0) ||
        (first == 0xed && text[1] > 0x9f) ||
        (first == 0xf0 && text[1] < 0x90) ||
        (first == 0xf4 && text[1] > 0x8f)) return -1;
    for (size_t index = 1; index < count; index++) {
        if ((text[index] & 0xc0u) != 0x80u) return -1;
        value = (value << 6) | (uint32_t)(text[index] & 0x3fu);
    }
    *codepoint = value;
    *length = count;
    return 0;
}

static int split_utf8_string_valid(const char* text) {
    const unsigned char* bytes = (const unsigned char*)text;
    size_t remaining = text ? strlen(text) : 0;
    while (remaining > 0) {
        uint32_t codepoint;
        size_t length;
        if (split_utf8_decode(bytes, remaining, &codepoint, &length) != 0) return 0;
        (void)codepoint;
        bytes += length;
        remaining -= length;
    }
    return text != NULL;
}

static int split_bpe_space_codepoint(uint32_t codepoint) {
    if (codepoint >= 0x2000 && codepoint <= 0x200a) return 1;
    switch (codepoint) {
        case 0x0009:
        case 0x000a:
        case 0x000b:
        case 0x000c:
        case 0x000d:
        case 0x0020:
        case 0x0085:
        case 0x00a0:
        case 0x1680:
        case 0x2028:
        case 0x2029:
        case 0x202f:
        case 0x205f:
        case 0x3000:
            return 1;
        default:
            return 0;
    }
}

static void split_sha256_text(SplitSha256* sha, const char* text) {
    split_sha256_update(sha, (const unsigned char*)text, strlen(text));
}

static int split_sha256_json_string(SplitSha256* sha, const char* value) {
    static const char hex[] = "0123456789abcdef";
    const unsigned char* bytes = (const unsigned char*)value;
    size_t length;
    if (!sha || !value || !split_utf8_string_valid(value)) return -1;
    length = strlen(value);
    split_sha256_text(sha, "\"");
    for (size_t index = 0; index < length; index++) {
        unsigned char byte = bytes[index];
        const char* escaped = NULL;
        char control[7];
        switch (byte) {
            case '\b': escaped = "\\b"; break;
            case '\t': escaped = "\\t"; break;
            case '\n': escaped = "\\n"; break;
            case '\f': escaped = "\\f"; break;
            case '\r': escaped = "\\r"; break;
            case '"': escaped = "\\\""; break;
            case '\\': escaped = "\\\\"; break;
            default: break;
        }
        if (escaped) {
            split_sha256_text(sha, escaped);
        } else if (byte < 0x20) {
            control[0] = '\\';
            control[1] = 'u';
            control[2] = '0';
            control[3] = '0';
            control[4] = hex[byte >> 4];
            control[5] = hex[byte & 0x0fu];
            control[6] = 0;
            split_sha256_text(sha, control);
        } else {
            split_sha256_update(sha, &byte, 1);
        }
    }
    split_sha256_text(sha, "\"");
    return 0;
}

static int split_sha256_string_array(SplitSha256* sha, const cJSON* array) {
    int count;
    if (!sha || !cJSON_IsArray(array)) return -1;
    count = cJSON_GetArraySize((cJSON*)array);
    split_sha256_text(sha, "[");
    for (int index = 0; index < count; index++) {
        const cJSON* item = cJSON_GetArrayItem((cJSON*)array, index);
        if (!cJSON_IsString(item) || !item->valuestring) return -1;
        if (index) split_sha256_text(sha, ",");
        if (split_sha256_json_string(sha, item->valuestring) != 0) return -1;
    }
    split_sha256_text(sha, "]");
    return 0;
}

static int split_sha256_merges(SplitSha256* sha, const cJSON* merges) {
    int count;
    if (!sha || !cJSON_IsArray(merges)) return -1;
    count = cJSON_GetArraySize((cJSON*)merges);
    split_sha256_text(sha, "[");
    for (int index = 0; index < count; index++) {
        const cJSON* pair = cJSON_GetArrayItem((cJSON*)merges, index);
        const cJSON* left;
        const cJSON* right;
        if (!cJSON_IsArray(pair) || cJSON_GetArraySize((cJSON*)pair) != 2 ||
            !(left = cJSON_GetArrayItem((cJSON*)pair, 0)) ||
            !(right = cJSON_GetArrayItem((cJSON*)pair, 1)) ||
            !cJSON_IsString(left) || !left->valuestring ||
            !cJSON_IsString(right) || !right->valuestring) return -1;
        if (index) split_sha256_text(sha, ",");
        split_sha256_text(sha, "[");
        if (split_sha256_json_string(sha, left->valuestring) != 0) return -1;
        split_sha256_text(sha, ",");
        if (split_sha256_json_string(sha, right->valuestring) != 0) return -1;
        split_sha256_text(sha, "]");
    }
    split_sha256_text(sha, "]");
    return 0;
}

static int split_bpe_fingerprint(const cJSON* root, char digest_hex[65]) {
    static const char hex[] = "0123456789abcdef";
    static const char* const special_names[] = {"bos", "eos", "pad", "unk"};
    const cJSON* atomic = cJSON_GetObjectItemCaseSensitive((cJSON*)root,
                                                          "atomic_tokens");
    const cJSON* bytes = cJSON_GetObjectItemCaseSensitive((cJSON*)root,
                                                         "byte_tokens");
    const cJSON* itos = cJSON_GetObjectItemCaseSensitive((cJSON*)root, "itos");
    const cJSON* merges = cJSON_GetObjectItemCaseSensitive((cJSON*)root, "merges");
    const cJSON* unused = cJSON_GetObjectItemCaseSensitive((cJSON*)root,
                                                          "unused_tokens");
    const cJSON* special = cJSON_GetObjectItemCaseSensitive((cJSON*)root,
                                                           "special_tokens");
    const char* normalization = split_string(root, "normalization",
                                             "vocab.normalization");
    const char* type = split_string(root, "type", "vocab.type");
    unsigned char digest[32];
    SplitSha256 sha;
    int version;
    int vocab_size;
    if (!root || !digest_hex || !normalization || !type || !cJSON_IsObject(special) ||
        split_integer(root, "version", "vocab.version", 1, 1, &version) != 0 ||
        split_integer(root, "vocab_size", "vocab.vocab_size", 1, INT_MAX,
                      &vocab_size) != 0) return -1;
    split_sha256_init(&sha);
    split_sha256_text(&sha, "{\"atomic_tokens\":");
    if (split_sha256_string_array(&sha, atomic) != 0) return -1;
    split_sha256_text(&sha, ",\"byte_tokens\":");
    if (split_sha256_string_array(&sha, bytes) != 0) return -1;
    split_sha256_text(&sha, ",\"itos\":");
    if (split_sha256_string_array(&sha, itos) != 0) return -1;
    split_sha256_text(&sha, ",\"merges\":");
    if (split_sha256_merges(&sha, merges) != 0) return -1;
    split_sha256_text(&sha, ",\"normalization\":");
    if (split_sha256_json_string(&sha, normalization) != 0) return -1;
    split_sha256_text(&sha, ",\"special_tokens\":{");
    for (int index = 0; index < 4; index++) {
        const char* value = split_string(special, special_names[index],
                                         "vocab.special_tokens");
        if (!value) return -1;
        if (index) split_sha256_text(&sha, ",");
        if (split_sha256_json_string(&sha, special_names[index]) != 0) return -1;
        split_sha256_text(&sha, ":");
        if (split_sha256_json_string(&sha, value) != 0) return -1;
    }
    split_sha256_text(&sha, "},\"type\":");
    if (split_sha256_json_string(&sha, type) != 0) return -1;
    split_sha256_text(&sha, ",\"unused_tokens\":");
    if (split_sha256_string_array(&sha, unused) != 0) return -1;
    {
        char numbers[96];
        int count = snprintf(numbers, sizeof(numbers),
                             ",\"version\":%d,\"vocab_size\":%d}",
                             version, vocab_size);
        if (count <= 0 || (size_t)count >= sizeof(numbers)) return -1;
        split_sha256_text(&sha, numbers);
    }
    split_sha256_final(&sha, digest);
    for (size_t index = 0; index < sizeof(digest); index++) {
        digest_hex[index * 2] = hex[digest[index] >> 4];
        digest_hex[index * 2 + 1] = hex[digest[index] & 0x0fu];
    }
    digest_hex[64] = 0;
    return 0;
}

static int split_vocab_copy_items(SplitVocab* vocab, const cJSON* itos,
                                  int expected_count, const char* label,
                                  int require_nonempty) {
    if (!vocab || !cJSON_IsArray(itos) ||
        (vocab->count = cJSON_GetArraySize((cJSON*)itos)) != expected_count ||
        !(vocab->items = (char**)calloc((size_t)vocab->count,
                                       sizeof(*vocab->items)))) {
        fprintf(stderr, "[tinyreceipt] %s must contain exactly %d entries\n",
                label, expected_count);
        return -1;
    }
    for (int index = 0; index < vocab->count; index++) {
        const cJSON* item = cJSON_GetArrayItem((cJSON*)itos, index);
        size_t length;
        if (!cJSON_IsString(item) || !item->valuestring) {
            fprintf(stderr, "[tinyreceipt] invalid %s entry %d\n", label, index);
            return -1;
        }
        length = strlen(item->valuestring);
        if ((require_nonempty && length == 0) || length > 4096 ||
            !split_utf8_string_valid(item->valuestring) ||
            !(vocab->items[index] = (char*)malloc(length + 1))) {
            fprintf(stderr, "[tinyreceipt] invalid %s entry %d\n", label, index);
            return -1;
        }
        memcpy(vocab->items[index], item->valuestring, length + 1);
        for (int previous = 0; previous < index; previous++) {
            if (strcmp(vocab->items[previous], vocab->items[index]) == 0) {
                fprintf(stderr,
                        "[tinyreceipt] duplicate %s entries at IDs %d and %d\n",
                        label, previous, index);
                return -1;
            }
        }
    }
    vocab->pad = split_vocab_find(vocab, "<pad>", 5);
    vocab->bos = split_vocab_find(vocab, "<bos>", 5);
    vocab->eos = split_vocab_find(vocab, "<eos>", 5);
    vocab->unk = split_vocab_find(vocab, "<unk>", 5);
    if (vocab->pad != 0 || vocab->bos != 1 || vocab->eos != 2 || vocab->unk != 3) {
        fprintf(stderr, "[tinyreceipt] vocab special tokens must occupy IDs 0,1,2,3\n");
        return -1;
    }
    return 0;
}

static int split_bpe_validate_token_array(const SplitVocab* vocab,
                                          const cJSON* array,
                                          unsigned char* flags,
                                          int* ids, const char* label) {
    int count;
    if (!vocab || !cJSON_IsArray(array) || !flags) return -1;
    count = cJSON_GetArraySize((cJSON*)array);
    for (int index = 0; index < count; index++) {
        const cJSON* item = cJSON_GetArrayItem((cJSON*)array, index);
        int id;
        if (!cJSON_IsString(item) || !item->valuestring || !item->valuestring[0] ||
            (id = split_vocab_find(vocab, item->valuestring,
                                   strlen(item->valuestring))) < 0) {
            fprintf(stderr, "[tinyreceipt] %s contains a token absent from itos\n",
                    label);
            return -1;
        }
        if (flags[id]) {
            fprintf(stderr, "[tinyreceipt] %s must not contain duplicates\n",
                    label);
            return -1;
        }
        flags[id] = 1;
        if (ids) ids[index] = id;
    }
    return 0;
}

static int split_bpe_merged_text_valid(const char* text) {
    const unsigned char* cursor = (const unsigned char*)text;
    size_t remaining = strlen(text);
    while (remaining > 0) {
        uint32_t codepoint;
        size_t length;
        if (split_utf8_decode(cursor, remaining, &codepoint, &length) != 0 ||
            (codepoint >= '0' && codepoint <= '9') ||
            split_bpe_space_codepoint(codepoint)) return 0;
        cursor += length;
        remaining -= length;
    }
    return 1;
}

static int split_load_bpe_metadata(const cJSON* root, const SplitPackage* package,
                                   SplitVocab* vocab) {
    static const char* const root_keys[] = {
        "type", "version", "vocab_size", "itos", "merges", "normalization",
        "atomic_tokens", "byte_tokens", "unused_tokens", "special_tokens",
        "tokenizer_hash",
    };
    static const char* const special_keys[] = {"pad", "bos", "eos", "unk"};
    static const char* const special_values[] = {
        "<pad>", "<bos>", "<eos>", "<unk>",
    };
    const cJSON* itos = cJSON_GetObjectItemCaseSensitive((cJSON*)root, "itos");
    const cJSON* merges = cJSON_GetObjectItemCaseSensitive((cJSON*)root, "merges");
    const cJSON* atomic = cJSON_GetObjectItemCaseSensitive((cJSON*)root,
                                                          "atomic_tokens");
    const cJSON* bytes = cJSON_GetObjectItemCaseSensitive((cJSON*)root,
                                                         "byte_tokens");
    const cJSON* unused = cJSON_GetObjectItemCaseSensitive((cJSON*)root,
                                                          "unused_tokens");
    const cJSON* special = cJSON_GetObjectItemCaseSensitive((cJSON*)root,
                                                           "special_tokens");
    const char* type = split_string(root, "type", "vocab.type");
    const char* normalization = split_string(root, "normalization",
                                             "vocab.normalization");
    const char* tokenizer_hash = split_string(root, "tokenizer_hash",
                                              "vocab.tokenizer_hash");
    char actual_hash[65];
    int version;
    int vocab_size;
    if (split_exact_keys(root, root_keys, 11, "BPE vocab.json") != 0 ||
        !type || strcmp(type, "byte_fallback_bpe") != 0 ||
        !normalization || strcmp(normalization, "NFC") != 0 ||
        !tokenizer_hash || !split_sha256_string_valid(tokenizer_hash) ||
        strcmp(tokenizer_hash, package->tokenizer_hash) != 0 ||
        split_integer(root, "version", "vocab.version", 1, 1, &version) != 0 ||
        split_integer(root, "vocab_size", "vocab.vocab_size",
                      package->tokenizer_vocab_count,
                      package->tokenizer_vocab_count, &vocab_size) != 0 ||
        split_bpe_fingerprint(root, actual_hash) != 0 ||
        strcmp(actual_hash, tokenizer_hash) != 0 ||
        !cJSON_IsArray(merges) || !cJSON_IsArray(atomic) ||
        !cJSON_IsArray(bytes) || cJSON_GetArraySize((cJSON*)bytes) != 256 ||
        !cJSON_IsArray(unused) || !cJSON_IsObject(special) ||
        split_exact_keys(special, special_keys, 4, "vocab.special_tokens") != 0) {
        fprintf(stderr,
                "[tinyreceipt] invalid byte_fallback_bpe v1 vocabulary contract\n");
        return -1;
    }
    if (cJSON_GetArraySize((cJSON*)atomic) !=
        (int)(sizeof(k_split_bpe_atomic_tokens) /
              sizeof(k_split_bpe_atomic_tokens[0]))) {
        fprintf(stderr,
                "[tinyreceipt] atomic_tokens must contain the exact 18-token OCR contract\n");
        return -1;
    }
    for (int index = 0;
         index < (int)(sizeof(k_split_bpe_atomic_tokens) /
                       sizeof(k_split_bpe_atomic_tokens[0]));
         index++) {
        const cJSON* item = cJSON_GetArrayItem((cJSON*)atomic, index);
        if (!cJSON_IsString(item) || !item->valuestring ||
            strcmp(item->valuestring, k_split_bpe_atomic_tokens[index]) != 0) {
            fprintf(stderr,
                    "[tinyreceipt] atomic_tokens must preserve structural tokens followed by digits 0-9\n");
            return -1;
        }
    }
    for (int index = 0; index < 4; index++) {
        const char* value = split_string(special, special_keys[index],
                                         "vocab.special_tokens");
        if (!value || strcmp(value, special_values[index]) != 0) {
            fprintf(stderr, "[tinyreceipt] invalid BPE special token declaration\n");
            return -1;
        }
    }
    vocab->kind = SPLIT_TOKENIZER_BYTE_FALLBACK_BPE;
    if (split_vocab_copy_items(vocab, itos, vocab_size, "BPE itos", 1) != 0)
        return -1;
    vocab->is_atomic = (unsigned char*)calloc((size_t)vocab->count, 1);
    vocab->is_byte = (unsigned char*)calloc((size_t)vocab->count, 1);
    vocab->is_unused = (unsigned char*)calloc((size_t)vocab->count, 1);
    vocab->byte_value = (unsigned char*)calloc((size_t)vocab->count, 1);
    vocab->atomic_count = cJSON_GetArraySize((cJSON*)atomic);
    vocab->atomic_ids = (int*)calloc((size_t)(vocab->atomic_count > 0 ?
                                              vocab->atomic_count : 1),
                                    sizeof(*vocab->atomic_ids));
    if (!vocab->is_atomic || !vocab->is_byte || !vocab->is_unused ||
        !vocab->byte_value || !vocab->atomic_ids ||
        split_bpe_validate_token_array(vocab, atomic, vocab->is_atomic,
                                       vocab->atomic_ids,
                                       "atomic_tokens") != 0 ||
        split_bpe_validate_token_array(vocab, unused, vocab->is_unused, NULL,
                                       "unused_tokens") != 0) return -1;
    for (int byte = 0; byte < 256; byte++) {
        const cJSON* item = cJSON_GetArrayItem((cJSON*)bytes, byte);
        char expected[7];
        int id;
        snprintf(expected, sizeof(expected), "<0x%02X>", byte);
        if (!cJSON_IsString(item) || !item->valuestring ||
            strcmp(item->valuestring, expected) != 0 ||
            (id = split_vocab_find(vocab, expected, 6)) < 0) {
            fprintf(stderr,
                    "[tinyreceipt] byte_tokens must be exactly <0x00> through <0xFF>\n");
            return -1;
        }
        vocab->is_byte[id] = 1;
        vocab->byte_value[id] = (unsigned char)byte;
        vocab->byte_ids[byte] = id;
    }
    vocab->merge_count = cJSON_GetArraySize((cJSON*)merges);
    vocab->merges = (SplitBpeMerge*)calloc(
        (size_t)(vocab->merge_count > 0 ? vocab->merge_count : 1),
        sizeof(*vocab->merges));
    if (!vocab->merges) return -1;
    for (int index = 0; index < vocab->merge_count; index++) {
        const cJSON* pair = cJSON_GetArrayItem((cJSON*)merges, index);
        const cJSON* left_item;
        const cJSON* right_item;
        const char* left;
        const char* right;
        char* merged;
        size_t left_length;
        size_t right_length;
        int left_id;
        int right_id;
        int output_id;
        if (!cJSON_IsArray(pair) || cJSON_GetArraySize((cJSON*)pair) != 2 ||
            !(left_item = cJSON_GetArrayItem((cJSON*)pair, 0)) ||
            !(right_item = cJSON_GetArrayItem((cJSON*)pair, 1)) ||
            !cJSON_IsString(left_item) || !(left = left_item->valuestring) ||
            !cJSON_IsString(right_item) || !(right = right_item->valuestring) ||
            (left_id = split_vocab_find(vocab, left, strlen(left))) < 0 ||
            (right_id = split_vocab_find(vocab, right, strlen(right))) < 0 ||
            vocab->is_byte[left_id] || vocab->is_byte[right_id] ||
            vocab->is_atomic[left_id] || vocab->is_atomic[right_id]) {
            fprintf(stderr, "[tinyreceipt] invalid BPE merge pair %d\n", index);
            return -1;
        }
        for (int previous = 0; previous < index; previous++) {
            if (vocab->merges[previous].left == left_id &&
                vocab->merges[previous].right == right_id) {
                fprintf(stderr, "[tinyreceipt] duplicate BPE merge pair\n");
                return -1;
            }
        }
        left_length = strlen(left);
        right_length = strlen(right);
        if (left_length > SIZE_MAX - right_length - 1 ||
            !(merged = (char*)malloc(left_length + right_length + 1))) return -1;
        memcpy(merged, left, left_length);
        memcpy(merged + left_length, right, right_length + 1);
        output_id = split_vocab_find(vocab, merged, left_length + right_length);
        if (output_id < 0 || !split_bpe_merged_text_valid(merged)) {
            fprintf(stderr, "[tinyreceipt] invalid BPE merge output %s\n", merged);
            free(merged);
            return -1;
        }
        free(merged);
        vocab->merges[index].left = left_id;
        vocab->merges[index].right = right_id;
        vocab->merges[index].output = output_id;
    }
    return 0;
}

static int split_load_vocab(const SplitPackage* package, SplitVocab* vocab) {
    char* text = NULL;
    cJSON* root = NULL;
    int rc = -1;
    if (!package || !vocab) return -1;
    memset(vocab, 0, sizeof(*vocab));
    vocab->pad = vocab->bos = vocab->eos = vocab->unk = -1;
    for (int byte = 0; byte < 256; byte++) vocab->byte_ids[byte] = -1;
    if (split_read_text(package->vocab_path, &text) != 0 ||
        !(root = cJSON_ParseWithOpts(text, NULL, 1))) {
        fprintf(stderr, "[tinyreceipt] cannot parse tokenizer JSON: %s\n",
                package->vocab_path);
        goto cleanup;
    }
    if (!split_json_keys_unique(root)) {
        fprintf(stderr, "[tinyreceipt] tokenizer JSON contains duplicate object keys\n");
        goto cleanup;
    }
    if (package->tokenizer_kind != SPLIT_TOKENIZER_BYTE_FALLBACK_BPE) {
        fprintf(stderr, "[tinyreceipt] package has no canonical BPE tokenizer contract\n");
        goto cleanup;
    }
    if (split_load_bpe_metadata(root, package, vocab) != 0) goto cleanup;
    rc = 0;

cleanup:
    cJSON_Delete(root);
    free(text);
    if (rc != 0) split_vocab_free(vocab);
    return rc;
}

static char* split_clean_bpe_text(const char* input) {
    const unsigned char* source = (const unsigned char*)(input ? input : "");
    size_t input_length = strlen((const char*)source);
    char* output = (char*)malloc(input_length + 1);
    size_t in = 0;
    size_t out = 0;
    int pending_space = 0;
    if (!output) return NULL;
    while (in < input_length) {
        uint32_t codepoint;
        size_t length;
        if (split_utf8_decode(source + in, input_length - in, &codepoint,
                              &length) != 0) {
            fprintf(stderr, "[tinyreceipt] BPE prompt must be valid UTF-8\n");
            free(output);
            return NULL;
        }
        if (split_bpe_space_codepoint(codepoint) ||
            (codepoint >= 0x001c && codepoint <= 0x001f)) {
            pending_space = out > 0;
            in += length;
            continue;
        }
        if (pending_space) output[out++] = ' ';
        pending_space = 0;
        memcpy(output + out, source + in, length);
        out += length;
        in += length;
    }
    output[out] = 0;
    return output;
}

static int split_bpe_atomic_match(const SplitVocab* vocab, const char* text,
                                  size_t remaining, size_t* matched_length) {
    int selected = -1;
    size_t selected_length = 0;
    if (!vocab || !text || !matched_length) return -1;
    for (int index = 0; index < vocab->atomic_count; index++) {
        int id = vocab->atomic_ids[index];
        size_t length = strlen(vocab->items[id]);
        uint32_t first_codepoint = 0;
        size_t first_length;
        int multiple_characters = length > 0 &&
            split_utf8_decode((const unsigned char*)vocab->items[id], length,
                              &first_codepoint, &first_length) == 0 &&
            first_length < length;
        (void)first_codepoint;
        if (multiple_characters && length <= remaining &&
            length > selected_length &&
            memcmp(text, vocab->items[id], length) == 0) {
            selected = id;
            selected_length = length;
        }
    }
    *matched_length = selected_length;
    return selected;
}

static int split_bpe_encode_span(const SplitVocab* vocab, const char* text,
                                 size_t length, int32_t* output,
                                 size_t output_capacity, size_t* output_count) {
    int* tokens = NULL;
    size_t token_count = 0;
    size_t cursor = 0;
    int rc = -1;
    if (!vocab || (!text && length) || !output || !output_count ||
        *output_count > output_capacity) return -1;
    tokens = (int*)malloc((length ? length : 1) * sizeof(*tokens));
    if (!tokens) return -1;
    while (cursor < length) {
        uint32_t codepoint;
        size_t character_length;
        int id;
        if (split_utf8_decode((const unsigned char*)text + cursor,
                              length - cursor, &codepoint,
                              &character_length) != 0) goto cleanup;
        (void)codepoint;
        id = split_vocab_find(vocab, text + cursor, character_length);
        if (id >= 0 && !vocab->is_byte[id] && !vocab->is_unused[id]) {
            tokens[token_count++] = id;
        } else {
            for (size_t byte = 0; byte < character_length; byte++) {
                int byte_id = vocab->byte_ids[
                    ((const unsigned char*)text)[cursor + byte]];
                if (byte_id < 0) goto cleanup;
                tokens[token_count++] = byte_id;
            }
        }
        cursor += character_length;
    }
    while (token_count > 1) {
        int best_rank = vocab->merge_count;
        for (size_t index = 0; index + 1 < token_count; index++) {
            for (int rank = 0; rank < best_rank; rank++) {
                if (vocab->merges[rank].left == tokens[index] &&
                    vocab->merges[rank].right == tokens[index + 1]) {
                    best_rank = rank;
                    break;
                }
            }
        }
        if (best_rank == vocab->merge_count) break;
        {
            size_t read = 0;
            size_t write = 0;
            while (read < token_count) {
                if (read + 1 < token_count &&
                    tokens[read] == vocab->merges[best_rank].left &&
                    tokens[read + 1] == vocab->merges[best_rank].right) {
                    tokens[write++] = vocab->merges[best_rank].output;
                    read += 2;
                } else {
                    tokens[write++] = tokens[read++];
                }
            }
            token_count = write;
        }
    }
    if (token_count > output_capacity - *output_count) goto cleanup;
    for (size_t index = 0; index < token_count; index++)
        output[(*output_count)++] = tokens[index];
    rc = 0;

cleanup:
    free(tokens);
    return rc;
}

static int split_encode_bpe_question(const SplitVocab* vocab, const char* prompt,
                                     int32_t* ids, int capacity) {
    char* clean = NULL;
    int32_t* encoded = NULL;
    size_t clean_length;
    size_t encoded_count = 0;
    size_t cursor = 0;
    size_t span_start = 0;
    int result = -1;
    if (!vocab || !ids || capacity <= 0) return -1;
    clean = split_clean_bpe_text(prompt);
    if (!clean) return -1;
    clean_length = strlen(clean);
    if (clean_length > (SIZE_MAX / sizeof(*encoded)) - 1 ||
        !(encoded = (int32_t*)malloc((clean_length + 1) * sizeof(*encoded))))
        goto cleanup;
    while (cursor < clean_length) {
        uint32_t codepoint;
        size_t character_length;
        size_t tag_length = 0;
        int tag_id;
        int is_digit;
        int is_space;
        if (split_utf8_decode((const unsigned char*)clean + cursor,
                              clean_length - cursor, &codepoint,
                              &character_length) != 0) goto cleanup;
        tag_id = split_bpe_atomic_match(vocab, clean + cursor,
                                        clean_length - cursor, &tag_length);
        is_digit = codepoint >= '0' && codepoint <= '9';
        is_space = split_bpe_space_codepoint(codepoint);
        if (tag_id < 0 && !is_digit && !is_space) {
            cursor += character_length;
            continue;
        }
        if (span_start < cursor &&
            split_bpe_encode_span(vocab, clean + span_start,
                                  cursor - span_start, encoded,
                                  clean_length + 1, &encoded_count) != 0)
            goto cleanup;
        if (tag_id >= 0) {
            encoded[encoded_count++] = tag_id;
            cursor += tag_length;
        } else if (is_digit) {
            int digit_id = split_vocab_find(vocab, clean + cursor,
                                            character_length);
            if (digit_id < 0) goto cleanup;
            encoded[encoded_count++] = digit_id;
            cursor += character_length;
        } else {
            int id = split_vocab_find(vocab, clean + cursor, character_length);
            if (id >= 0 && !vocab->is_byte[id]) {
                encoded[encoded_count++] = id;
            } else {
                for (size_t byte = 0; byte < character_length; byte++) {
                    int byte_id = vocab->byte_ids[
                        ((const unsigned char*)clean)[cursor + byte]];
                    if (byte_id < 0) goto cleanup;
                    encoded[encoded_count++] = byte_id;
                }
            }
            cursor += character_length;
        }
        span_start = cursor;
    }
    if (span_start < clean_length &&
        split_bpe_encode_span(vocab, clean + span_start,
                              clean_length - span_start, encoded,
                              clean_length + 1, &encoded_count) != 0)
        goto cleanup;
    encoded[encoded_count++] = vocab->eos;
    if (encoded_count > (size_t)capacity) encoded_count = (size_t)capacity;
    memcpy(ids, encoded, encoded_count * sizeof(*ids));
    if (ids[encoded_count - 1] != vocab->eos) ids[encoded_count - 1] = vocab->eos;
    result = (int)encoded_count;

cleanup:
    free(encoded);
    free(clean);
    return result;
}

static int split_encode_question(const SplitVocab* vocab, const char* prompt,
                                 int32_t* ids, int capacity) {
    if (!vocab || vocab->kind != SPLIT_TOKENIZER_BYTE_FALLBACK_BPE) return -1;
    return split_encode_bpe_question(vocab, prompt, ids, capacity);
}

static size_t split_utf8_replacement_advance(const unsigned char* bytes,
                                             size_t remaining) {
    unsigned char first;
    size_t expected;
    if (!bytes || !remaining) return 0;
    first = bytes[0];
    if (first >= 0xc2 && first <= 0xdf) expected = 2;
    else if (first >= 0xe0 && first <= 0xef) expected = 3;
    else if (first >= 0xf0 && first <= 0xf4) expected = 4;
    else return 1;
    if (remaining > 1 &&
        ((bytes[1] & 0xc0u) != 0x80u ||
         (first == 0xe0 && bytes[1] < 0xa0) ||
         (first == 0xed && bytes[1] > 0x9f) ||
         (first == 0xf0 && bytes[1] < 0x90) ||
         (first == 0xf4 && bytes[1] > 0x8f))) return 1;
    for (size_t index = 2; index < expected && index < remaining; index++) {
        if ((bytes[index] & 0xc0u) != 0x80u) return index;
    }
    return remaining < expected ? remaining : 1;
}

static int split_vocab_decode_alloc(const SplitVocab* vocab, const int32_t* ids,
                                    size_t count, char** decoded,
                                    size_t* decoded_length) {
    static const unsigned char replacement[] = {0xef, 0xbf, 0xbd};
    unsigned char* raw = NULL;
    char* output = NULL;
    size_t raw_capacity = 0;
    size_t raw_length = 0;
    size_t cursor = 0;
    size_t out = 0;
    int rc = -1;
    if (!vocab || (!ids && count) || !decoded || !decoded_length) return -1;
    *decoded = NULL;
    *decoded_length = 0;
    for (size_t index = 0; index < count; index++) {
        int id = ids[index];
        size_t added;
        if (id == vocab->eos) break;
        if (id == vocab->pad || id == vocab->bos || id < 0 || id >= vocab->count ||
            (vocab->is_unused && vocab->is_unused[id])) continue;
        added = vocab->is_byte && vocab->is_byte[id] ? 1 : strlen(vocab->items[id]);
        if (raw_capacity > SIZE_MAX - added) return -1;
        raw_capacity += added;
    }
    raw = (unsigned char*)malloc(raw_capacity ? raw_capacity : 1);
    if (!raw) return -1;
    for (size_t index = 0; index < count; index++) {
        int id = ids[index];
        if (id == vocab->eos) break;
        if (id == vocab->pad || id == vocab->bos || id < 0 || id >= vocab->count ||
            (vocab->is_unused && vocab->is_unused[id])) continue;
        if (vocab->is_byte && vocab->is_byte[id]) {
            raw[raw_length++] = vocab->byte_value[id];
        } else {
            size_t length = strlen(vocab->items[id]);
            memcpy(raw + raw_length, vocab->items[id], length);
            raw_length += length;
        }
    }
    if (raw_length > (SIZE_MAX - 1) / 3 ||
        !(output = (char*)malloc(raw_length * 3 + 1))) goto cleanup;
    while (cursor < raw_length) {
        uint32_t codepoint;
        size_t length;
        if (split_utf8_decode(raw + cursor, raw_length - cursor,
                              &codepoint, &length) == 0) {
            (void)codepoint;
            memcpy(output + out, raw + cursor, length);
            out += length;
            cursor += length;
        } else {
            size_t advance = split_utf8_replacement_advance(
                raw + cursor, raw_length - cursor);
            memcpy(output + out, replacement, sizeof(replacement));
            out += sizeof(replacement);
            cursor += advance ? advance : 1;
        }
    }
    output[out] = 0;
    *decoded = output;
    *decoded_length = out;
    output = NULL;
    rc = 0;

cleanup:
    free(output);
    free(raw);
    return rc;
}

int tiny_receipt_split_w8a8_tokenizer_roundtrip(
        const char* package_path, const char* prompt, int32_t* ids,
        size_t ids_capacity, size_t* ids_count, char* decoded,
        size_t decoded_capacity) {
    SplitPackage package;
    SplitVocab vocab;
    char* result = NULL;
    size_t result_length = 0;
    int count;
    int rc = -1;
    if (!package_path || !prompt || !ids || !ids_count || !decoded ||
        !ids_capacity || ids_capacity > INT_MAX || !decoded_capacity) return -1;
    memset(&package, 0, sizeof(package));
    memset(&vocab, 0, sizeof(vocab));
    *ids_count = 0;
    decoded[0] = 0;
    if (split_load_package(package_path, &package) != 0 ||
        split_load_vocab(&package, &vocab) != 0 ||
        (count = split_encode_question(&vocab, prompt, ids,
                                       (int)ids_capacity)) < 0 ||
        split_vocab_decode_alloc(&vocab, ids, (size_t)count, &result,
                                 &result_length) != 0 ||
        result_length >= decoded_capacity) goto cleanup;
    memcpy(decoded, result, result_length + 1);
    *ids_count = (size_t)count;
    rc = 0;

cleanup:
    free(result);
    split_vocab_free(&vocab);
    return rc;
}

static int split_family_id(const char* value) {
    if (!value || !value[0] || strcmp(value, "auto") == 0) return -1;
    for (int index = 0; index < SPLIT_FAMILY_COUNT; index++) {
        if (strcmp(value, k_split_family_names[index]) == 0) return index;
    }
    return -2;
}

static void split_report_failure(const char* action, VxStatus status,
                                 const VxReport* report) {
    fprintf(stderr, "[tinyreceipt] %s failed: %s", action, vx_status_string(status));
    if (report && report->reason[0]) fprintf(stderr, " (%s)", report->reason);
    if (report && report->message[0]) fprintf(stderr, ": %s", report->message);
    if (report && report->route_evidence[0])
        fprintf(stderr, " {%s}", report->route_evidence);
    if (report && report->offending_node[0])
        fprintf(stderr, " <node=%s>", report->offending_node);
    if (report && report->decode_state[0]) fprintf(stderr, " [%s]", report->decode_state);
    fputc('\n', stderr);
}

static int split_report_has_no_operator_fallback(const char* evidence) {
    static const char expected[] = "operator=none";
    const char* cursor = evidence;
    if (!cursor) return 0;
    while (*cursor) {
        const char* end = strchr(cursor, ';');
        size_t length = end ? (size_t)(end - cursor) : strlen(cursor);
        if (length == sizeof(expected) - 1 &&
            memcmp(cursor, expected, sizeof(expected) - 1) == 0)
            return 1;
        if (!end) break;
        cursor = end + 1;
    }
    return 0;
}

int tiny_receipt_split_w8a8_report_proves_strict_backend(
        const VxReport* report, const char* expected_backend) {
    return report && expected_backend && expected_backend[0] &&
        report->policy_mode == VX_BACKEND_REQUIRE &&
        report->operator_fallback == VX_OPERATOR_FALLBACK_FORBID &&
        report->backend[0] && strcmp(report->backend, expected_backend) == 0 &&
        report->route_attested && report->route_evidence[0] &&
        report->operator_fallback_used == 0 &&
        split_report_has_no_operator_fallback(report->fallback_evidence);
}

static int split_report_proves_strict_backend(const SplitGraph* graph,
                                              const VxReport* report,
                                              const char* label) {
    if (graph && graph->expected_policy_mode == VX_BACKEND_REQUIRE &&
        graph->expected_operator_fallback == VX_OPERATOR_FALLBACK_FORBID &&
        tiny_receipt_split_w8a8_report_proves_strict_backend(
            report, graph->expected_backend))
        return 0;
    fprintf(stderr,
            "[tinyreceipt] %s did not attest the requested strict provider with fallback 0",
            label ? label : "execution");
    if (graph && graph->expected_backend[0])
        fprintf(stderr, " expected=%s", graph->expected_backend);
    if (report) {
        fprintf(stderr, " actual=%s policy=%d operator_fallback=%d",
                report->backend[0] ? report->backend : "unknown",
                (int)report->policy_mode, (int)report->operator_fallback);
        if (report->route_evidence[0])
            fprintf(stderr, " {%s}", report->route_evidence);
    }
    fputc('\n', stderr);
    return -1;
}

static int split_select_backend(SplitCommand* command, const char* backend) {
    VxBackendPolicy* policy = &command->backend_policy;
    if (policy->backend_count != 0 && strcmp(policy->backends[0], backend) != 0) {
        fprintf(stderr, "[tinyreceipt] pass at most one explicit backend flag\n");
        return -1;
    }
    policy->mode = VX_BACKEND_REQUIRE;
    policy->operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    command->backend_candidates[0] = backend;
    policy->backends = command->backend_candidates;
    policy->backend_count = 1;
    return 1;
}

static int split_parse_engine_flag(const char* arg, SplitCommand* command) {
    if (!strcmp(arg, "--cpu")) return split_select_backend(command, "cpu");
    if (!strcmp(arg, "--vulkan")) return split_select_backend(command, "vulkan");
    if (!strcmp(arg, "--opengl")) return split_select_backend(command, "opengl");
    if (!strcmp(arg, "--metal")) return split_select_backend(command, "metal");
    if (!strcmp(arg, "--cuda")) return split_select_backend(command, "cuda");
    if (!strcmp(arg, "--debug")) {
        command->runtime_options.debug = 1;
        return 1;
    }
    return 0;
}

static int split_parse_nonnegative(const char* value, int maximum, int* out) {
    char* end = NULL;
    long parsed;
    if (!value || !value[0]) return -1;
    errno = 0;
    parsed = strtol(value, &end, 10);
    if (errno != 0 || !end || *end != 0 || parsed < 0 || parsed > maximum) return -1;
    *out = (int)parsed;
    return 0;
}

static void split_graph_close(SplitGraph* graph) {
    if (!graph) return;
    if (graph->context) (void)vx_execution_context_close(graph->context, NULL);
    vx_execution_context_release(graph->context);
    vx_compiled_model_release(graph->compiled);
    vx_model_release(graph->model);
    memset(graph, 0, sizeof(*graph));
}

static int split_graph_open(VxRuntime* runtime, const char* graph_path,
                            const char* weights_path,
                            const VxBackendPolicy* policy,
                            const VxContextOptions* context_options,
                            const char* label, int debug, SplitGraph* graph) {
    const char* weight_paths[1] = {weights_path};
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxReport report = VX_REPORT_INIT;
    VxStatus status;
    char action[64];
    double compile_ms = 0.0;
    memset(graph, 0, sizeof(*graph));
    if (!policy || policy->mode != VX_BACKEND_REQUIRE ||
        policy->operator_fallback != VX_OPERATOR_FALLBACK_FORBID ||
        policy->backend_count != 1 || !policy->backends ||
        !policy->backends[0] || !policy->backends[0][0] ||
        strlen(policy->backends[0]) >= sizeof(graph->expected_backend)) {
        fprintf(stderr,
                "[tinyreceipt] %s requires one exact backend with operator fallback forbidden\n",
                label ? label : "graph");
        goto fail;
    }
    graph->expected_policy_mode = policy->mode;
    graph->expected_operator_fallback = policy->operator_fallback;
    snprintf(graph->expected_backend, sizeof(graph->expected_backend), "%s",
             policy->backends[0]);
    source.graph_path = graph_path;
    source.weight_paths = weight_paths;
    source.weight_path_count = 1;
    status = vx_runtime_load_model(runtime, &source, &graph->model, &report);
    if (status != VX_STATUS_OK) {
        snprintf(action, sizeof(action), "%s model load", label);
        split_report_failure(action, status, &report);
        goto fail;
    }
    report = (VxReport)VX_REPORT_INIT;
    status = vx_model_compile(graph->model, policy, &graph->compiled, &report);
    if (status != VX_STATUS_OK) {
        snprintf(action, sizeof(action), "%s model compile", label);
        split_report_failure(action, status, &report);
        goto fail;
    }
    snprintf(action, sizeof(action), "%s model compile", label);
    if (split_report_proves_strict_backend(graph, &report, action) != 0)
        goto fail;
    compile_ms = report.compile_time_ms;
    report = (VxReport)VX_REPORT_INIT;
    status = vx_compiled_model_create_context(graph->compiled, context_options,
                                              &graph->context, &report);
    if (status != VX_STATUS_OK) {
        snprintf(action, sizeof(action), "%s context creation", label);
        split_report_failure(action, status, &report);
        goto fail;
    }
    snprintf(action, sizeof(action), "%s context creation", label);
    if (split_report_proves_strict_backend(graph, &report, action) != 0)
        goto fail;
    if (debug) {
        fprintf(stderr, "[debug] tinyreceipt split %s compile=%.3f ms backend=%s%s%s\n",
                label, compile_ms, report.backend[0] ? report.backend : "unknown",
                report.decode_state[0] ? " decode=" : "",
                report.decode_state[0] ? report.decode_state : "");
    }
    return 0;

fail:
    split_graph_close(graph);
    return -1;
}

static int split_find_input(VxExecutionContext* context, const char* name,
                            VxTensorSpec* found) {
    size_t count = vx_execution_context_input_count(context);
    for (size_t index = 0; index < count; index++) {
        VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
        if (vx_execution_context_input_spec(context, index, &spec, NULL) ==
                VX_STATUS_OK &&
            spec.name && strcmp(spec.name, name) == 0) {
            *found = spec;
            return 0;
        }
    }
    return -1;
}

static int split_exact_inputs(VxExecutionContext* context,
                              const char* const* names, size_t expected,
                              const char* label) {
    int seen[SPLIT_MAX_BINDINGS] = {0};
    size_t count = vx_execution_context_input_count(context);
    if (expected > SPLIT_MAX_BINDINGS || count != expected) {
        fprintf(stderr, "[tinyreceipt] %s graph must expose exactly %zu inputs\n",
                label, expected);
        return -1;
    }
    for (size_t index = 0; index < count; index++) {
        VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
        size_t match;
        if (vx_execution_context_input_spec(context, index, &spec, NULL) !=
                VX_STATUS_OK || !spec.name) return -1;
        for (match = 0; match < expected; match++) {
            if (!strcmp(spec.name, names[match])) break;
        }
        if (match == expected || seen[match]) {
            fprintf(stderr, "[tinyreceipt] %s runtime inputs differ from manifest mapping\n",
                    label);
            return -1;
        }
        seen[match] = 1;
    }
    return 0;
}

static int split_input_spec(VxExecutionContext* context, const char* name,
                            VxDataType dtype, const int64_t* shape, uint32_t rank,
                            size_t byte_size, const char* label) {
    VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
    VxAffineQuantization affine = VX_AFFINE_QUANTIZATION_INIT;
    VxStatus affine_status;
    size_t expected_bytes = dtype == VX_DTYPE_F32 || dtype == VX_DTYPE_I32
        ? 4u : dtype == VX_DTYPE_I8 || dtype == VX_DTYPE_U8 ? 1u : 0u;
    if (split_find_input(context, name, &spec) != 0 || spec.dtype != dtype ||
        spec.rank != rank || !expected_bytes) goto invalid;
    for (uint32_t axis = 0; axis < rank; axis++) {
        const VxDimensionConstraint* dimension = &spec.dimensions[axis];
        if (shape[axis] <= 0 || shape[axis] < dimension->min ||
            shape[axis] > dimension->max ||
            shape[axis] % dimension->multiple_of != 0 ||
            (dimension->kind == VX_DIMENSION_FIXED &&
             shape[axis] != dimension->min) ||
            (uint64_t)shape[axis] > SIZE_MAX / expected_bytes) goto invalid;
        expected_bytes *= (size_t)shape[axis];
    }
    if (expected_bytes != byte_size) goto invalid;
    affine_status = vx_execution_context_input_affine_quantization(
        context, name, &affine, NULL);
    /* Built-in contexts expose affine metadata, so enforce the package's
     * unquantized public ABI there. External providers may intentionally omit
     * this optional introspection; dtype/shape/size remain authoritative. */
    if ((affine_status == VX_STATUS_OK && affine.defined) ||
        (affine_status != VX_STATUS_OK &&
         affine_status != VX_STATUS_BACKEND_UNSUPPORTED)) goto invalid;
    return 0;

invalid:
    fprintf(stderr, "[tinyreceipt] %s does not match the qualified unquantized ABI\n",
            label);
    return -1;
}

static int split_validate_encoder_inputs(VxExecutionContext* context,
                                         const SplitEncoderDefinition* encoder,
                                         int question_length) {
    const char* names[] = {
        encoder->image, encoder->question_ids,
        encoder->question_position_ids, encoder->family_ids,
    };
    const int64_t image_shape[] = {1, 1, SPLIT_IMAGE_HEIGHT, SPLIT_IMAGE_WIDTH};
    const int64_t question_shape[] = {1, question_length};
    const int64_t family_shape[] = {1};
    return question_length < 1 || question_length > SPLIT_QUESTION_LENGTH ||
        split_exact_inputs(context, names, 4, "encoder") ||
        split_input_spec(context, encoder->image, VX_DTYPE_F32, image_shape, 4,
                         (size_t)SPLIT_IMAGE_HEIGHT * SPLIT_IMAGE_WIDTH * sizeof(float),
                         "encoder image") ||
        split_input_spec(context, encoder->question_ids, VX_DTYPE_I32,
                         question_shape, 2,
                         (size_t)question_length * sizeof(int32_t),
                         "encoder question_ids") ||
        split_input_spec(context, encoder->question_position_ids, VX_DTYPE_I32,
                         question_shape, 2,
                         (size_t)question_length * sizeof(int32_t),
                         "encoder question_position_ids") ||
        split_input_spec(context, encoder->family_ids, VX_DTYPE_I32,
                         family_shape, 1, sizeof(int32_t),
                         "encoder family_ids");
}

static int split_validate_decoder_inputs_v1(
        VxExecutionContext* context, const SplitDecoderDefinition* decoder,
        int memory_length, int past_length) {
    const char* names[5 + SPLIT_CACHE_TENSORS * 2];
    const int64_t token_shape[] = {1, 1};
    const int64_t scalar_shape[] = {1};
    const int64_t memory_mask_shape[] = {1, memory_length};
    const int64_t past_mask_shape[] = {1, past_length};
    const int64_t cross_shape[] = {
        1, SPLIT_CACHE_HEADS, memory_length, SPLIT_CACHE_HEAD_WIDTH,
    };
    const int64_t past_shape[] = {
        1, SPLIT_CACHE_HEADS, past_length, SPLIT_CACHE_HEAD_WIDTH,
    };
    const size_t cross_bytes = (size_t)SPLIT_CACHE_HEADS *
        (size_t)memory_length * SPLIT_CACHE_HEAD_WIDTH * sizeof(float);
    const size_t past_bytes = (size_t)SPLIT_CACHE_HEADS *
        (size_t)past_length * SPLIT_CACHE_HEAD_WIDTH * sizeof(float);
    size_t input_count = 0;
    names[input_count++] = decoder->decoder_input_ids;
    names[input_count++] = decoder->position_ids;
    names[input_count++] = decoder->family_ids;
    names[input_count++] = decoder->memory_padding_mask;
    names[input_count++] = decoder->past_padding_mask;
    for (int index = 0; index < SPLIT_CACHE_TENSORS; index++)
        names[input_count++] = decoder->cross_kv[index];
    for (int index = 0; index < SPLIT_CACHE_TENSORS; index++)
        names[input_count++] = decoder->past_kv[index];
    if (memory_length < SPLIT_IMAGE_TOKENS + 1 ||
        memory_length > SPLIT_MEMORY_LENGTH || past_length < 1 ||
        past_length > SPLIT_MAX_NEW_TOKENS ||
        split_exact_inputs(context, names, input_count, "decoder") != 0 ||
        split_input_spec(context, decoder->decoder_input_ids, VX_DTYPE_I32,
                         token_shape, 2, sizeof(int32_t),
                         "decoder decoder_input_ids") != 0 ||
        split_input_spec(context, decoder->position_ids, VX_DTYPE_I32,
                         scalar_shape, 1, sizeof(int32_t),
                         "decoder position_ids") != 0 ||
        split_input_spec(context, decoder->family_ids, VX_DTYPE_I32,
                         scalar_shape, 1, sizeof(int32_t),
                         "decoder family_ids") != 0 ||
        split_input_spec(context, decoder->memory_padding_mask, VX_DTYPE_I32,
                         memory_mask_shape, 2,
                         (size_t)memory_length * sizeof(int32_t),
                         "decoder memory_padding_mask") != 0 ||
        split_input_spec(context, decoder->past_padding_mask, VX_DTYPE_I32,
                         past_mask_shape, 2,
                         (size_t)past_length * sizeof(int32_t),
                         "decoder past_padding_mask") != 0)
        return -1;
    for (int index = 0; index < SPLIT_CACHE_TENSORS; index++) {
        if (split_input_spec(context, decoder->cross_kv[index], VX_DTYPE_F32,
                             cross_shape, 4, cross_bytes,
                             k_split_cross_semantics[index]) != 0 ||
            split_input_spec(context, decoder->past_kv[index], VX_DTYPE_F32,
                             past_shape, 4, past_bytes,
                             k_split_past_semantics[index]) != 0)
            return -1;
    }
    return 0;
}

static int split_add_input(VxExecutionContext* context,
                           SplitBindingBatch* batch, const char* name,
                           VxDataType dtype, const int64_t* shape,
                           uint32_t rank, const void* data,
                           size_t byte_size) {
    VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
    VxTensorBinding* binding;
    if (!batch || batch->count >= sizeof(batch->values) / sizeof(batch->values[0]) ||
        !data || split_find_input(context, name, &spec) != 0 ||
        split_input_spec(context, name, dtype, shape, rank, byte_size,
                         name) != 0) return -1;
    for (size_t index = 0; index < batch->count; index++)
        if (!strcmp(batch->values[index].name, spec.name)) return -1;
    binding = &batch->values[batch->count++];
    *binding = (VxTensorBinding)VX_TENSOR_BINDING_INIT;
    binding->name = spec.name;
    binding->dtype = dtype;
    binding->rank = rank;
    memcpy(binding->shape, shape, rank * sizeof(*shape));
    binding->data = data;
    binding->byte_size = byte_size;
    binding->location = VX_MEMORY_HOST;
    return 0;
}

static int split_find_output(const VxResult* result, const char* name,
                             VxTensorInfo* found) {
    size_t count = vx_result_output_count(result);
    for (size_t index = 0; index < count; index++) {
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        if (vx_result_output_info(result, index, &info, NULL) == VX_STATUS_OK &&
            info.name && strcmp(info.name, name) == 0) {
            *found = info;
            return 0;
        }
    }
    return -1;
}

static int split_exact_outputs(const VxResult* result, const char* const* names,
                               size_t expected, const char* label) {
    int seen[SPLIT_MAX_GRAPH_OUTPUTS] = {0};
    size_t count = vx_result_output_count(result);
    if (expected > SPLIT_MAX_GRAPH_OUTPUTS || count != expected) goto invalid;
    for (size_t index = 0; index < count; index++) {
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        size_t match;
        if (vx_result_output_info(result, index, &info, NULL) != VX_STATUS_OK ||
            !info.name) goto invalid;
        for (match = 0; match < expected; match++) {
            if (!strcmp(info.name, names[match])) break;
        }
        if (match == expected || seen[match]) goto invalid;
        seen[match] = 1;
    }
    return 0;

invalid:
    fprintf(stderr, "[tinyreceipt] %s result outputs differ from the qualified manifest\n",
            label);
    return -1;
}

static int split_read_output(const VxResult* result, const char* name,
                             VxDataType dtype, const int64_t* shape, uint32_t rank,
                             void* destination, size_t byte_size,
                             const char* label) {
    VxTensorInfo info = VX_TENSOR_INFO_INIT;
    VxReport report = VX_REPORT_INIT;
    size_t required = 0;
    VxStatus status;
    if (split_find_output(result, name, &info) != 0 || info.dtype != dtype ||
        info.rank != rank || info.byte_size != byte_size) goto invalid;
    for (uint32_t axis = 0; axis < rank; axis++) {
        if (info.shape[axis] != shape[axis]) goto invalid;
    }
    status = vx_result_read(result, name, destination, byte_size, &required, &report);
    if (status != VX_STATUS_OK || required != byte_size) {
        if (status != VX_STATUS_OK) split_report_failure("result read", status, &report);
        goto invalid;
    }
    return 0;

invalid:
    fprintf(stderr, "[tinyreceipt] %s does not match the qualified output ABI\n", label);
    return -1;
}

static void split_encoder_output_release(SplitEncoderOutput* output) {
    if (!output) return;
    free(output->memory);
    free(output->memory_padding_mask);
    for (int index = 0; index < SPLIT_CACHE_TENSORS; index++)
        free(output->cross_kv[index]);
    memset(output, 0, sizeof(*output));
}

static int split_run_encoder(SplitGraph* graph,
                             const SplitEncoderDefinition* definition,
                             const SplitVocab* vocab, const char* image_path,
                             const char* prompt, int requested_family,
                             SplitShapeMode shape_mode, int debug,
                             SplitEncoderOutput* output,
                             TinyReceiptSplitShapeEvidence* evidence) {
    const char* output_names[4 + SPLIT_CACHE_TENSORS];
    int64_t memory_shape[] = {1, 0, SPLIT_MODEL_WIDTH};
    int64_t mask_shape[] = {1, 0};
    int64_t cross_shape[] = {
        1, SPLIT_CACHE_HEADS, 0, SPLIT_CACHE_HEAD_WIDTH,
    };
    const int64_t logits_shape[] = {1, SPLIT_FAMILY_COUNT};
    const int64_t selected_shape[] = {1};
    const int image_shape[] = {1, SPLIT_IMAGE_HEIGHT, SPLIT_IMAGE_WIDTH, 1};
    float* image = NULL;
    int32_t question_ids[SPLIT_QUESTION_LENGTH];
    int32_t question_position_ids[SPLIT_QUESTION_LENGTH];
    VxResult* result = NULL;
    VxReport report = VX_REPORT_INIT;
    SplitBindingBatch bindings = {0};
    VxStatus status;
    char image_error[256] = {0};
    double started;
    int question_count;
    int bound_question_length;
    int32_t family_id = requested_family;
    int32_t selected = -1;
    int router_family = 0;
    int rc = -1;
    int memory_length;
    if (!graph || !definition || !vocab || !output) return -1;
    memset(output, 0, sizeof(*output));
    output_names[0] = definition->memory;
    output_names[1] = definition->memory_padding_mask;
    output_names[2] = definition->router_logits;
    output_names[3] = definition->selected_family_ids;
    for (int index = 0; index < SPLIT_CACHE_TENSORS; index++)
        output_names[4 + index] = definition->cross_kv[index];
    image = (float*)malloc((size_t)SPLIT_IMAGE_HEIGHT * SPLIT_IMAGE_WIDTH * sizeof(float));
    output->memory = (float*)malloc(
        (size_t)SPLIT_MEMORY_LENGTH * SPLIT_MODEL_WIDTH * sizeof(float));
    output->memory_padding_mask = (int32_t*)malloc(
        (size_t)SPLIT_MEMORY_LENGTH * sizeof(int32_t));
    if (!image || !output->memory || !output->memory_padding_mask) goto cleanup;
    {
        const size_t cross_elements = (size_t)SPLIT_CACHE_HEADS *
            SPLIT_MEMORY_LENGTH * SPLIT_CACHE_HEAD_WIDTH;
        for (int tensor = 0; tensor < SPLIT_CACHE_TENSORS; tensor++) {
            output->cross_kv[tensor] = (float*)malloc(
                cross_elements * sizeof(*output->cross_kv[tensor]));
            if (!output->cross_kv[tensor]) goto cleanup;
            for (size_t index = 0; index < cross_elements; index++)
                output->cross_kv[tensor][index] = NAN;
        }
    }
    for (size_t index = 0;
         index < (size_t)SPLIT_MEMORY_LENGTH * SPLIT_MODEL_WIDTH; index++)
        output->memory[index] = NAN;
    for (int index = 0; index < SPLIT_MEMORY_LENGTH; index++)
        output->memory_padding_mask[index] = INT32_MIN;
    if (tiny_receipt_load_image_to_tensor(image_path, image, image_shape, 4,
                                          image_error, sizeof(image_error)) != 0) {
        fprintf(stderr, "[tinyreceipt] image load failed: %s\n",
                image_error[0] ? image_error : image_path);
        goto cleanup;
    }
    if (debug) {
        char input_digest[65];
        if (split_sha256_f32_le(
                image,
                (size_t)SPLIT_IMAGE_HEIGHT * (size_t)SPLIT_IMAGE_WIDTH,
                input_digest) != 0)
            goto cleanup;
        fprintf(stderr,
                "[debug] tinyreceipt split input_f32_sha256=%s\n",
                input_digest);
    }
    for (int index = 0; index < SPLIT_QUESTION_LENGTH; index++)
        question_ids[index] = vocab->pad;
    question_count = split_encode_question(vocab, prompt, question_ids,
                                           SPLIT_QUESTION_LENGTH);
    if (question_count < 1 || question_count > SPLIT_QUESTION_LENGTH) goto cleanup;
    bound_question_length = shape_mode == SPLIT_SHAPE_MODE_MAXIMUM_PADDED ?
        SPLIT_QUESTION_LENGTH : question_count;
    memory_length = bound_question_length + SPLIT_IMAGE_TOKENS;
    memory_shape[1] = memory_length;
    mask_shape[1] = memory_length;
    cross_shape[2] = memory_length;
    for (int index = 0; index < bound_question_length; index++)
        question_position_ids[index] = index;
    if (split_validate_encoder_inputs(
            graph->context, definition, bound_question_length) != 0) goto cleanup;
    if (debug) {
        fprintf(stderr, "[debug] tinyreceipt split question_tokens=%d\n",
                question_count);
        fputs("[debug] tinyreceipt split question_token_ids=", stderr);
        for (int index = 0; index < question_count; index++)
            fprintf(stderr, "%s%d", index == 0 ? "" : ",",
                    question_ids[index]);
        fputc('\n', stderr);
    }
    if (split_add_input(
            graph->context, &bindings, definition->image, VX_DTYPE_F32,
            (const int64_t[]){1, 1, SPLIT_IMAGE_HEIGHT, SPLIT_IMAGE_WIDTH},
            4u, image,
            (size_t)SPLIT_IMAGE_HEIGHT * SPLIT_IMAGE_WIDTH * sizeof(float)) != 0 ||
        split_add_input(
            graph->context, &bindings, definition->question_ids,
            VX_DTYPE_I32,
            (const int64_t[]){1, bound_question_length}, 2u,
            question_ids,
            (size_t)bound_question_length * sizeof(*question_ids)) != 0 ||
        split_add_input(
            graph->context, &bindings, definition->question_position_ids,
            VX_DTYPE_I32,
            (const int64_t[]){1, bound_question_length}, 2u,
            question_position_ids,
            (size_t)bound_question_length * sizeof(*question_position_ids)) != 0 ||
        split_add_input(graph->context, &bindings, definition->family_ids,
                        VX_DTYPE_I32, (const int64_t[]){1}, 1u,
                        &family_id, sizeof(family_id)) != 0) goto cleanup;
    started = split_now_ms();
    status = vx_execution_context_execute(
        graph->context, bindings.values, bindings.count, &result, &report);
    output->execution_ms = split_now_ms() - started;
    if (status != VX_STATUS_OK) {
        split_report_failure("encoder execution", status, &report);
        goto cleanup;
    }
    if (split_report_proves_strict_backend(
            graph, &report, "encoder execution") != 0)
        goto cleanup;
    if (split_exact_outputs(result, output_names, 4 + SPLIT_CACHE_TENSORS,
                            "encoder") != 0 ||
        split_read_output(result, definition->memory, VX_DTYPE_F32, memory_shape, 3,
                          output->memory,
                          (size_t)memory_length * SPLIT_MODEL_WIDTH * sizeof(float),
                          "encoder memory") != 0 ||
        split_read_output(result, definition->memory_padding_mask, VX_DTYPE_I32,
                          mask_shape, 2, output->memory_padding_mask,
                          (size_t)memory_length * sizeof(int32_t),
                          "encoder memory_padding_mask") != 0 ||
        split_read_output(result, definition->router_logits, VX_DTYPE_F32,
                          logits_shape, 2, output->router_logits,
                          sizeof(output->router_logits), "encoder router_logits") != 0 ||
        split_read_output(result, definition->selected_family_ids, VX_DTYPE_I32,
                          selected_shape, 1, &selected, sizeof(selected),
                          "encoder selected_family_ids") != 0) goto cleanup;
    {
        const size_t active_elements = (size_t)SPLIT_CACHE_HEADS *
            (size_t)memory_length * SPLIT_CACHE_HEAD_WIDTH;
        for (int tensor = 0; tensor < SPLIT_CACHE_TENSORS; tensor++) {
            if (split_read_output(
                    result, definition->cross_kv[tensor], VX_DTYPE_F32,
                    cross_shape, 4, output->cross_kv[tensor],
                    active_elements * sizeof(*output->cross_kv[tensor]),
                    k_split_cross_semantics[tensor]) != 0)
                goto cleanup;
            for (size_t index = 0; index < active_elements; index++) {
                if (!isfinite(output->cross_kv[tensor][index])) {
                    fprintf(stderr,
                            "[tinyreceipt] encoder %s contains a non-finite value\n",
                            k_split_cross_semantics[tensor]);
                    goto cleanup;
                }
            }
        }
    }
    for (int index = 0; index < memory_length; index++) {
        if (output->memory_padding_mask[index] != 0 &&
            output->memory_padding_mask[index] != 1) {
            fprintf(stderr, "[tinyreceipt] encoder memory_padding_mask contains a value other than 0 or 1\n");
            goto cleanup;
        }
    }
    for (size_t index = 0;
         index < (size_t)memory_length * SPLIT_MODEL_WIDTH; index++) {
        if (!isfinite(output->memory[index])) {
            fprintf(stderr,
                    "[tinyreceipt] encoder memory contains a non-finite value\n");
            goto cleanup;
        }
    }
    for (size_t index = (size_t)memory_length * SPLIT_MODEL_WIDTH;
         index < (size_t)SPLIT_MEMORY_LENGTH * SPLIT_MODEL_WIDTH; index++) {
        if (!isnan(output->memory[index])) {
            fprintf(stderr, "[tinyreceipt] encoder result read overwrote inactive memory tail\n");
            goto cleanup;
        }
    }
    for (int index = memory_length; index < SPLIT_MEMORY_LENGTH; index++) {
        if (output->memory_padding_mask[index] != INT32_MIN) {
            fprintf(stderr, "[tinyreceipt] encoder result read overwrote inactive mask tail\n");
            goto cleanup;
        }
    }
    for (int index = 0; index < SPLIT_FAMILY_COUNT; index++) {
        if (!isfinite(output->router_logits[index])) {
            fprintf(stderr, "[tinyreceipt] encoder router_logits contains a non-finite value\n");
            goto cleanup;
        }
        if (index > 0 && output->router_logits[index] >
                             output->router_logits[router_family]) {
            router_family = index;
        }
    }
    if (selected < 0 || selected >= SPLIT_FAMILY_COUNT) {
        fprintf(stderr, "[tinyreceipt] encoder selected an invalid family ID %d\n", selected);
        goto cleanup;
    }
    if (requested_family >= 0 && selected != requested_family) {
        fprintf(stderr,
                "[tinyreceipt] encoder selected %s, but --family requested %s\n",
                k_split_family_names[selected], k_split_family_names[requested_family]);
        goto cleanup;
    }
    output->selected_family_id = selected;
    output->question_length = question_count;
    output->memory_length = memory_length;
    if (evidence) {
        evidence->shape_mode = (int32_t)shape_mode;
        evidence->question_length = question_count;
        evidence->memory_length = memory_length;
        evidence->selected_family_id = selected;
        evidence->encoder_result_bytes = report.result_bytes;
        evidence->encoder_ms = report.execution_time_ms;
        snprintf(evidence->encoder_route, sizeof(evidence->encoder_route), "%s",
                 report.route_evidence);
    }
    if (debug) {
        fprintf(stderr, "[debug] tinyreceipt split router=%s selected=%s%s encoder=%.3f ms\n",
                k_split_family_names[router_family], k_split_family_names[selected],
                requested_family >= 0 ? " (requested)" : "",
                output->execution_ms);
        fprintf(stderr, "[debug] tinyreceipt split encoder shape %s\n",
                report.route_evidence);
    }
    rc = 0;

cleanup:
    vx_result_release(result);
    free(image);
    if (rc != 0) {
        split_encoder_output_release(output);
    }
    return rc;
}

static int split_argmax_logits_row(const float* logits, int row, int vocab_count,
                                   int32_t* token_id) {
    size_t offset;
    int best = 0;
    if (!logits || !token_id || row < 0 || row >= SPLIT_DECODER_LENGTH ||
        vocab_count <= 0) return -1;
    offset = (size_t)row * (size_t)vocab_count;
    for (int token = 0; token < vocab_count; token++) {
        float value = logits[offset + (size_t)token];
        if (!isfinite(value)) {
            fprintf(stderr,
                    "[tinyreceipt] decoder logits row contains a non-finite value\n");
            return -1;
        }
        if (token > 0 && value > logits[offset + (size_t)best]) best = token;
    }
    *token_id = best;
    return 0;
}

static int split_run_decoder(SplitGraph* graph,
                             const SplitDecoderDefinition* definition,
                             const SplitVocab* vocab,
                             const SplitEncoderOutput* encoded, int max_new,
                             int debug, int emit_answer,
                             TinyReceiptSplitShapeEvidence* evidence) {
    const char* output_names[2 + SPLIT_CACHE_TENSORS];
    const size_t maximum_cache_elements = (size_t)SPLIT_CACHE_HEADS *
        SPLIT_DECODER_LENGTH * SPLIT_CACHE_HEAD_WIDTH;
    float* past_kv[SPLIT_CACHE_TENSORS] = {0};
    float* present_kv[SPLIT_CACHE_TENSORS] = {0};
    int32_t* past_padding_mask = NULL;
    int32_t* present_padding_mask = NULL;
    float* logits = NULL;
    int32_t generated_ids[SPLIT_MAX_NEW_TOKENS];
    int32_t emitted_ids[SPLIT_MAX_NEW_TOKENS];
    int32_t current_token;
    int32_t family_id;
    int past_length = 1;
    int generated = 0;
    int emitted = 0;
    uint64_t token_digest = UINT64_C(1469598103934665603);
    char* decoded = NULL;
    size_t decoded_length = 0;
    VxResult* result = NULL;
    VxReport report = VX_REPORT_INIT;
    double generation_started = split_now_ms();
    int rc = -1;
    if (!graph || !definition || !vocab || !encoded || max_new < 0 ||
        max_new > SPLIT_MAX_NEW_TOKENS)
        return -1;
    output_names[0] = definition->logits;
    output_names[1] = definition->present_padding_mask;
    for (int index = 0; index < SPLIT_CACHE_TENSORS; index++)
        output_names[2 + index] = definition->present_kv[index];
    logits = (float*)malloc((size_t)vocab->count * sizeof(*logits));
    past_padding_mask = (int32_t*)calloc(
        SPLIT_DECODER_LENGTH, sizeof(*past_padding_mask));
    present_padding_mask = (int32_t*)malloc(
        SPLIT_DECODER_LENGTH * sizeof(*present_padding_mask));
    if (!logits || !past_padding_mask || !present_padding_mask) goto cleanup;
    past_padding_mask[0] = 1;
    for (int tensor = 0; tensor < SPLIT_CACHE_TENSORS; tensor++) {
        past_kv[tensor] = (float*)calloc(
            maximum_cache_elements, sizeof(*past_kv[tensor]));
        present_kv[tensor] = (float*)malloc(
            maximum_cache_elements * sizeof(*present_kv[tensor]));
        if (!past_kv[tensor] || !present_kv[tensor]) goto cleanup;
    }
    current_token = vocab->bos;
    family_id = encoded->selected_family_id;
    if (evidence) evidence->target_length = max_new + 1;
    generation_started = split_now_ms();
    for (int step = 0; step < max_new; step++) {
        SplitBindingBatch bindings = {0};
        const int present_length = past_length + 1;
        const int32_t position_id = past_length - 1;
        const int64_t token_shape[] = {1, 1};
        const int64_t scalar_shape[] = {1};
        const int64_t memory_mask_shape[] = {1, encoded->memory_length};
        const int64_t past_mask_shape[] = {1, past_length};
        const int64_t cross_shape[] = {
            1, SPLIT_CACHE_HEADS, encoded->memory_length, SPLIT_CACHE_HEAD_WIDTH,
        };
        const int64_t past_shape[] = {
            1, SPLIT_CACHE_HEADS, past_length, SPLIT_CACHE_HEAD_WIDTH,
        };
        const int64_t logits_shape[] = {1, 1, vocab->count};
        const int64_t present_mask_shape[] = {1, present_length};
        const int64_t present_shape[] = {
            1, SPLIT_CACHE_HEADS, present_length, SPLIT_CACHE_HEAD_WIDTH,
        };
        const size_t cross_bytes = (size_t)SPLIT_CACHE_HEADS *
            (size_t)encoded->memory_length * SPLIT_CACHE_HEAD_WIDTH * sizeof(float);
        const size_t past_bytes = (size_t)SPLIT_CACHE_HEADS *
            (size_t)past_length * SPLIT_CACHE_HEAD_WIDTH * sizeof(float);
        const size_t present_bytes = (size_t)SPLIT_CACHE_HEADS *
            (size_t)present_length * SPLIT_CACHE_HEAD_WIDTH * sizeof(float);
        VxStatus status;
        int32_t next;
        double step_started;
        double step_ms;
        if (past_padding_mask[0] != 1 ||
            split_validate_decoder_inputs_v1(
                graph->context, definition, encoded->memory_length,
                past_length) != 0)
            goto cleanup;
        for (int tensor = 0; tensor < SPLIT_CACHE_TENSORS; tensor++) {
            for (int head = 0; head < SPLIT_CACHE_HEADS; head++) {
                const size_t sentinel = (size_t)head * (size_t)past_length *
                    SPLIT_CACHE_HEAD_WIDTH;
                for (int width = 0; width < SPLIT_CACHE_HEAD_WIDTH; width++) {
                    if (past_kv[tensor][sentinel + (size_t)width] != 0.0f) {
                        fprintf(stderr,
                                "[tinyreceipt] explicit KV sentinel changed before step %d\n",
                                step);
                        goto cleanup;
                    }
                }
            }
        }
        if (split_add_input(graph->context, &bindings,
                            definition->decoder_input_ids, VX_DTYPE_I32,
                            token_shape, 2, &current_token,
                            sizeof(current_token)) != 0 ||
            split_add_input(graph->context, &bindings, definition->position_ids,
                            VX_DTYPE_I32, scalar_shape, 1, &position_id,
                            sizeof(position_id)) != 0 ||
            split_add_input(graph->context, &bindings, definition->family_ids,
                            VX_DTYPE_I32, scalar_shape, 1, &family_id,
                            sizeof(family_id)) != 0 ||
            split_add_input(graph->context, &bindings,
                            definition->memory_padding_mask, VX_DTYPE_I32,
                            memory_mask_shape, 2, encoded->memory_padding_mask,
                            (size_t)encoded->memory_length * sizeof(int32_t)) != 0 ||
            split_add_input(graph->context, &bindings,
                            definition->past_padding_mask, VX_DTYPE_I32,
                            past_mask_shape, 2, past_padding_mask,
                            (size_t)past_length * sizeof(int32_t)) != 0)
            goto cleanup;
        for (int tensor = 0; tensor < SPLIT_CACHE_TENSORS; tensor++) {
            if (split_add_input(graph->context, &bindings,
                                definition->cross_kv[tensor], VX_DTYPE_F32,
                                cross_shape, 4, encoded->cross_kv[tensor],
                                cross_bytes) != 0 ||
                split_add_input(graph->context, &bindings,
                                definition->past_kv[tensor], VX_DTYPE_F32,
                                past_shape, 4, past_kv[tensor], past_bytes) != 0)
                goto cleanup;
        }
        step_started = split_now_ms();
        report = (VxReport)VX_REPORT_INIT;
        status = vx_execution_context_execute(
            graph->context, bindings.values, bindings.count, &result, &report);
        step_ms = split_now_ms() - step_started;
        if (status != VX_STATUS_OK) {
            split_report_failure("explicit KV decoder execution", status, &report);
            goto cleanup;
        }
        if (split_report_proves_strict_backend(
                graph, &report, "decoder execution") != 0)
            goto cleanup;
        if (split_exact_outputs(result, output_names,
                                2 + SPLIT_CACHE_TENSORS, "decoder") != 0 ||
            split_read_output(result, definition->logits, VX_DTYPE_F32,
                              logits_shape, 3, logits,
                              (size_t)vocab->count * sizeof(*logits),
                              "decoder logits") != 0 ||
            split_read_output(result, definition->present_padding_mask,
                              VX_DTYPE_I32, present_mask_shape, 2,
                              present_padding_mask,
                              (size_t)present_length * sizeof(int32_t),
                              "decoder present_padding_mask") != 0 ||
            split_argmax_logits_row(logits, 0, vocab->count, &next) != 0)
            goto cleanup;
        for (int index = 0; index < present_length; index++) {
            if ((present_padding_mask[index] != 0 &&
                 present_padding_mask[index] != 1) ||
                (index < past_length &&
                 present_padding_mask[index] != past_padding_mask[index])) {
                fprintf(stderr,
                        "[tinyreceipt] explicit KV present mask did not preserve its past prefix\n");
                goto cleanup;
            }
        }
        if (present_padding_mask[0] != 1 ||
            present_padding_mask[past_length] !=
                (current_token == vocab->pad ? 1 : 0)) {
            fprintf(stderr,
                    "[tinyreceipt] explicit KV sentinel/current mask lifecycle is invalid\n");
            goto cleanup;
        }
        for (int tensor = 0; tensor < SPLIT_CACHE_TENSORS; tensor++) {
            if (split_read_output(result, definition->present_kv[tensor],
                                  VX_DTYPE_F32, present_shape, 4,
                                  present_kv[tensor], present_bytes,
                                  k_split_present_semantics[tensor]) != 0)
                goto cleanup;
            for (int head = 0; head < SPLIT_CACHE_HEADS; head++) {
                const size_t past_offset = (size_t)head * (size_t)past_length *
                    SPLIT_CACHE_HEAD_WIDTH;
                const size_t present_offset = (size_t)head *
                    (size_t)present_length * SPLIT_CACHE_HEAD_WIDTH;
                const size_t prefix_bytes = (size_t)past_length *
                    SPLIT_CACHE_HEAD_WIDTH * sizeof(float);
                if (memcmp(present_kv[tensor] + present_offset,
                           past_kv[tensor] + past_offset,
                           prefix_bytes) != 0) {
                    fprintf(stderr,
                            "[tinyreceipt] explicit KV %s did not preserve its past prefix\n",
                            k_split_present_semantics[tensor]);
                    goto cleanup;
                }
                for (int width = 0; width < SPLIT_CACHE_HEAD_WIDTH; width++) {
                    const float appended = present_kv[tensor][
                        present_offset + (size_t)past_length *
                            SPLIT_CACHE_HEAD_WIDTH + (size_t)width];
                    if (!isfinite(appended) ||
                        present_kv[tensor][present_offset + (size_t)width] != 0.0f) {
                        fprintf(stderr,
                                "[tinyreceipt] explicit KV %s has an invalid sentinel or append\n",
                                k_split_present_semantics[tensor]);
                        goto cleanup;
                    }
                }
            }
        }
        token_digest ^= (uint32_t)next;
        token_digest *= UINT64_C(1099511628211);
        emitted_ids[emitted++] = next;
        if (evidence) {
            if (step == 0) {
                evidence->decoder_seed_result_bytes = report.result_bytes;
                evidence->decoder_seed_ms = report.execution_time_ms;
                evidence->decoder_seed_past_length = past_length;
                evidence->decoder_seed_present_length = present_length;
                snprintf(evidence->decoder_seed_route,
                         sizeof(evidence->decoder_seed_route), "%s",
                         report.route_evidence);
            } else {
                evidence->decoder_step_result_bytes = report.result_bytes;
                evidence->decoder_warm_ms = report.execution_time_ms;
                evidence->decoder_step_past_length = past_length;
                evidence->decoder_step_present_length = present_length;
                snprintf(evidence->decoder_step_route,
                         sizeof(evidence->decoder_step_route), "%s",
                         report.route_evidence);
            }
            evidence->explicit_kv_sentinel_preserved = 1;
        }
        if (debug) {
            fprintf(stderr,
                    "[debug] tinyreceipt explicit-kv family=%s step=%d P=%d R=%d token=%d time=%.3f ms\n",
                    k_split_family_names[encoded->selected_family_id], step,
                    past_length, present_length, next, step_ms);
            fprintf(stderr, "[debug] tinyreceipt split decoder shape %s\n",
                    report.route_evidence);
        }
        vx_result_release(result);
        result = NULL;
        for (int tensor = 0; tensor < SPLIT_CACHE_TENSORS; tensor++) {
            float* swap = past_kv[tensor];
            past_kv[tensor] = present_kv[tensor];
            present_kv[tensor] = swap;
        }
        {
            int32_t* swap = past_padding_mask;
            past_padding_mask = present_padding_mask;
            present_padding_mask = swap;
        }
        past_length = present_length;
        if (next < 0 || next >= vocab->count) {
            fprintf(stderr,
                    "[tinyreceipt] decoder emitted out-of-vocabulary token %d\n",
                    next);
            goto cleanup;
        }
        if (next == vocab->eos) break;
        generated_ids[generated++] = next;
        current_token = next;
    }
    if (split_vocab_decode_alloc(vocab, generated_ids, (size_t)generated,
                                 &decoded, &decoded_length) != 0 ||
        (emit_answer && decoded_length > 0 &&
         fwrite(decoded, 1, decoded_length, stdout) != decoded_length))
        goto cleanup;
    if (emit_answer) {
        fflush(stdout);
        fputc('\n', stdout);
    }
    if (debug) {
        const double total_ms = split_now_ms() - generation_started;
        fprintf(stderr,
                "[debug] tinyreceipt explicit-kv family=%s tokens=%d total=%.3f ms\n",
                k_split_family_names[encoded->selected_family_id], emitted,
                total_ms);
        fprintf(stderr, "[debug] tinyreceipt split emitted_token_ids=");
        if (emitted == 0) {
            fputs("none", stderr);
        } else {
            for (int index = 0; index < emitted; index++)
                fprintf(stderr, "%s%d", index == 0 ? "" : ",",
                        emitted_ids[index]);
        }
        fputc('\n', stderr);
    }
    if (evidence) {
        evidence->generated_tokens = emitted;
        memcpy(evidence->emitted_token_ids, emitted_ids,
               (size_t)emitted * sizeof(*emitted_ids));
        evidence->token_digest = token_digest;
    }
    rc = 0;

cleanup:
    vx_result_release(result);
    free(decoded);
    free(logits);
    free(past_padding_mask);
    free(present_padding_mask);
    for (int tensor = 0; tensor < SPLIT_CACHE_TENSORS; tensor++) {
        free(past_kv[tensor]);
        free(present_kv[tensor]);
    }
    return rc;
}
static int split_qualify_sequence(
    const char* package_path, const char* image_path,
    const char* const* prompts, const int32_t* maximum_new_tokens,
    const int32_t* shape_modes, size_t request_count, const char* backend,
    int cpu_threads, int debug, TinyReceiptSplitShapeEvidence* evidence) {
    const char* backends[1];
    SplitPackage package;
    SplitVocab vocab;
    SplitGraph encoder_graph = {0};
    SplitGraph decoder_graph = {0};
    SplitEncoderOutput encoded = {0};
    VxRuntime* runtime = NULL;
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions encoder_options = VX_CONTEXT_OPTIONS_INIT;
    VxContextOptions decoder_options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    int rc = -1;
    if (!package_path || !image_path || !prompts || !maximum_new_tokens ||
        !shape_modes || !backend || !backend[0] || cpu_threads <= 0 ||
        !evidence || request_count == 0)
        return -1;
    memset(&package, 0, sizeof(package));
    memset(&vocab, 0, sizeof(vocab));
    memset(evidence, 0, request_count * sizeof(*evidence));
    runtime_options.cpu_threads = cpu_threads;
    policy.mode = VX_BACKEND_REQUIRE;
    policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    backends[0] = backend;
    policy.backends = backends;
    policy.backend_count = 1;
    if (vx_runtime_create(&runtime_options, &runtime, &report) != VX_STATUS_OK ||
        split_load_package(package_path, &package) != 0 ||
        split_load_vocab(&package, &vocab) != 0)
        goto cleanup_sequence;
    if (split_graph_open(runtime, package.encoder.graph, package.encoder.weights,
                         &policy, &encoder_options, "encoder", debug,
                         &encoder_graph) != 0 ||
        split_graph_open(runtime, package.decoder.graph, package.decoder.weights,
                         &policy, &decoder_options, "decoder", debug,
                         &decoder_graph) != 0) goto cleanup_sequence;
    for (size_t index = 0; index < request_count; index++) {
        if (!prompts[index] || maximum_new_tokens[index] < 2 ||
            maximum_new_tokens[index] > SPLIT_MAX_NEW_TOKENS ||
            (shape_modes[index] != SPLIT_SHAPE_MODE_ACTIVE &&
             shape_modes[index] != SPLIT_SHAPE_MODE_MAXIMUM_PADDED) ||
            split_run_encoder(
                &encoder_graph, &package.encoder, &vocab, image_path,
                prompts[index], 0, (SplitShapeMode)shape_modes[index], debug,
                &encoded, &evidence[index]) != 0 ||
            split_run_decoder(&decoder_graph, &package.decoder, &vocab,
                              &encoded, maximum_new_tokens[index], debug, 1,
                              &evidence[index]) != 0)
            goto cleanup_sequence;
        split_encoder_output_release(&encoded);
    }
    rc = 0;

cleanup_sequence:
    split_encoder_output_release(&encoded);
    split_graph_close(&decoder_graph);
    split_graph_close(&encoder_graph);
    split_vocab_free(&vocab);
    if (runtime) (void)vx_runtime_close(runtime, NULL);
    vx_runtime_release(runtime);
    return rc;
}

int tiny_receipt_split_w8a8_qualify_sequence(
    const char* package_path, const char* image_path,
    const char* const* prompts, const int32_t* maximum_new_tokens,
    const int32_t* shape_modes, size_t request_count, const char* backend,
    TinyReceiptSplitShapeEvidence* evidence) {
    return split_qualify_sequence(
        package_path, image_path, prompts, maximum_new_tokens, shape_modes,
        request_count, backend, 1, 0, evidence);
}

int tiny_receipt_split_w8a8_profile_sequence(
    const char* package_path, const char* image_path,
    const char* const* prompts, const int32_t* maximum_new_tokens,
    size_t request_count, TinyReceiptSplitShapeEvidence* evidence) {
    int32_t* shape_modes;
    int rc;
    if (!request_count || request_count > SIZE_MAX / sizeof(*shape_modes) ||
        !(shape_modes = (int32_t*)calloc(request_count, sizeof(*shape_modes))))
        return -1;
    rc = tiny_receipt_split_w8a8_qualify_sequence(
        package_path, image_path, prompts, maximum_new_tokens, shape_modes,
        request_count, "cpu", evidence);
    free(shape_modes);
    return rc;
}

static int split_run_dynamic_qualification(const SplitCommand* command) {
    static const char* const prompts[] = {
        "x", "phone number last one", "phone number last one", "x",
    };
    static const int32_t maximum_new_tokens[] = {4, 4, 4, 4};
    static const int32_t shape_modes[] = {
        SPLIT_SHAPE_MODE_ACTIVE, SPLIT_SHAPE_MODE_ACTIVE,
        SPLIT_SHAPE_MODE_MAXIMUM_PADDED, SPLIT_SHAPE_MODE_ACTIVE,
    };
    TinyReceiptSplitShapeEvidence evidence[4];
    const char* backend;
    if (!command || command->backend_policy.backend_count != 1)
        return -1;
    backend = command->backend_policy.backends[0];
    if (split_qualify_sequence(
            command->package_arg, command->image_path, prompts,
            maximum_new_tokens, shape_modes, 4, backend,
            command->runtime_options.cpu_threads > 0 ?
                command->runtime_options.cpu_threads : 1,
            1, evidence) != 0)
        return -1;
    if (evidence[0].question_length >= evidence[1].question_length ||
        evidence[1].question_length != evidence[2].question_length ||
        evidence[1].memory_length == SPLIT_MEMORY_LENGTH ||
        evidence[2].memory_length != SPLIT_MEMORY_LENGTH ||
        evidence[3].question_length != evidence[0].question_length ||
        evidence[3].memory_length != evidence[0].memory_length ||
        evidence[3].selected_family_id != evidence[0].selected_family_id ||
        evidence[3].token_digest != evidence[0].token_digest ||
        evidence[2].selected_family_id != evidence[1].selected_family_id ||
        evidence[2].token_digest != evidence[1].token_digest) {
        fprintf(stderr, "[tinyreceipt] dynamic qualification parity failed\n");
        return -1;
    }
    for (size_t index = 0; index < 4; index++) {
        const TinyReceiptSplitShapeEvidence* item = &evidence[index];
        const int bound_q = item->memory_length - SPLIT_IMAGE_TOKENS;
        if (item->generated_tokens < 2 ||
            item->decoder_seed_past_length != 1 ||
            item->decoder_seed_present_length != 2 ||
            item->decoder_step_past_length < 2 ||
            item->decoder_step_present_length != item->decoder_step_past_length + 1 ||
            !item->explicit_kv_sentinel_preserved) {
            fprintf(stderr, "[tinyreceipt] dynamic qualification cache lifecycle failed\n");
            return -1;
        }
        printf("DYNAMIC_REBIND_RUN index=%zu mode=%s logical_Q=%d bound_Q=%d "
               "logical_M=%d bound_M=%d family_id=%d tokens=%d token_digest=%016llx "
               "token_ids=",
               index,
               item->shape_mode == SPLIT_SHAPE_MODE_MAXIMUM_PADDED ?
                   "maximum-padded" : "active",
               item->question_length, bound_q,
               item->question_length + SPLIT_IMAGE_TOKENS, item->memory_length,
               item->selected_family_id, item->generated_tokens,
               (unsigned long long)item->token_digest);
        for (int token = 0; token < item->generated_tokens; token++)
            printf("%s%d", token == 0 ? "" : ",",
                   item->emitted_token_ids[token]);
        printf(" seed_P=%d seed_R=%d step_P=%d step_R=%d cache_preserved=1\n",
               item->decoder_seed_past_length, item->decoder_seed_present_length,
               item->decoder_step_past_length, item->decoder_step_present_length);
    }
    printf("DYNAMIC_REBIND_RESULT status=pass backend=%s timed=0 same_runtime=1 "
           "same_encoder_context=1 same_decoder_context=1 strict_no_fallback=1 "
           "cpu_threads=%d\n",
           backend,
           command->runtime_options.cpu_threads > 0 ?
               command->runtime_options.cpu_threads : 1);
    return 0;
}

static int split_request_evidence_is_valid(
        const TinyReceiptSplitShapeEvidence* evidence, int max_new) {
    if (!evidence || max_new < 0 ||
        evidence->target_length != max_new + 1 ||
        evidence->question_length < 1 ||
        (evidence->shape_mode == SPLIT_SHAPE_MODE_ACTIVE &&
         evidence->memory_length !=
             evidence->question_length + SPLIT_IMAGE_TOKENS) ||
        (evidence->shape_mode == SPLIT_SHAPE_MODE_MAXIMUM_PADDED &&
         evidence->memory_length != SPLIT_MEMORY_LENGTH) ||
        (evidence->shape_mode != SPLIT_SHAPE_MODE_ACTIVE &&
         evidence->shape_mode != SPLIT_SHAPE_MODE_MAXIMUM_PADDED) ||
        evidence->selected_family_id < 0 ||
        evidence->selected_family_id >= SPLIT_FAMILY_COUNT ||
        evidence->generated_tokens < 0 ||
        evidence->generated_tokens > max_new)
        return 0;
    if (evidence->generated_tokens == 0) {
        return evidence->decoder_seed_past_length == 0 &&
            evidence->decoder_seed_present_length == 0 &&
            evidence->decoder_step_past_length == 0 &&
            evidence->decoder_step_present_length == 0 &&
            evidence->explicit_kv_sentinel_preserved == 0;
    }
    if (evidence->decoder_seed_past_length != 1 ||
        evidence->decoder_seed_present_length != 2 ||
        !evidence->explicit_kv_sentinel_preserved)
        return 0;
    if (evidence->generated_tokens == 1) {
        return evidence->decoder_step_past_length == 0 &&
            evidence->decoder_step_present_length == 0;
    }
    return evidence->decoder_step_past_length == evidence->generated_tokens &&
        evidence->decoder_step_present_length == evidence->generated_tokens + 1;
}

static int split_request_evidence_has_parity(
        const TinyReceiptSplitShapeEvidence* expected,
        const TinyReceiptSplitShapeEvidence* actual) {
    if (!expected || !actual ||
        expected->shape_mode != actual->shape_mode ||
        expected->question_length != actual->question_length ||
        expected->target_length != actual->target_length ||
        expected->memory_length != actual->memory_length ||
        expected->selected_family_id != actual->selected_family_id ||
        expected->generated_tokens != actual->generated_tokens ||
        expected->token_digest != actual->token_digest ||
        expected->decoder_seed_past_length !=
            actual->decoder_seed_past_length ||
        expected->decoder_seed_present_length !=
            actual->decoder_seed_present_length ||
        expected->decoder_step_past_length !=
            actual->decoder_step_past_length ||
        expected->decoder_step_present_length !=
            actual->decoder_step_present_length ||
        expected->explicit_kv_sentinel_preserved !=
            actual->explicit_kv_sentinel_preserved)
        return 0;
    return memcmp(expected->emitted_token_ids, actual->emitted_token_ids,
                  (size_t)expected->generated_tokens *
                      sizeof(*expected->emitted_token_ids)) == 0;
}

static void split_emit_warmup_result(
        int count, const TinyReceiptSplitShapeEvidence* evidence) {
    int last_past = 0;
    int last_present = 0;
    if (evidence->generated_tokens == 1) {
        last_past = evidence->decoder_seed_past_length;
        last_present = evidence->decoder_seed_present_length;
    } else if (evidence->generated_tokens > 1) {
        last_past = evidence->decoder_step_past_length;
        last_present = evidence->decoder_step_present_length;
    }
    printf("WARMUP_RESULT status=pass count=%d warmup_timed=0 measured_runs=1 "
           "same_runtime=1 same_encoder_context=1 same_decoder_context=1 "
           "strict_no_fallback=1 token_parity=1 cache_parity=1 cache_reset=1 "
           "family_id=%d tokens=%d token_digest=%016llx token_ids=",
           count, evidence->selected_family_id, evidence->generated_tokens,
           (unsigned long long)evidence->token_digest);
    if (evidence->generated_tokens == 0) {
        fputs("none", stdout);
    } else {
        for (int index = 0; index < evidence->generated_tokens; index++)
            printf("%s%d", index == 0 ? "" : ",",
                   evidence->emitted_token_ids[index]);
    }
    printf(" seed_P=%d seed_R=%d last_P=%d last_R=%d cache_preserved=%d\n",
           evidence->decoder_seed_past_length,
           evidence->decoder_seed_present_length, last_past, last_present,
           evidence->explicit_kv_sentinel_preserved);
}

int tiny_receipt_split_w8a8_run(int argc, char** argv) {
    SplitCommand command;
    SplitPackage package;
    SplitVocab vocab;
    SplitGraph encoder_graph = {0};
    SplitGraph decoder_graph = {0};
    SplitEncoderOutput encoded = {0};
    VxRuntime* runtime = NULL;
    VxContextOptions encoder_options = VX_CONTEXT_OPTIONS_INIT;
    VxContextOptions decoder_options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    TinyReceiptSplitShapeEvidence warmup_reference = {0};
    TinyReceiptSplitShapeEvidence request_evidence = {0};
    int emit_timing = 0;
    int rc = 1;
    memset(&command, 0, sizeof(command));
    memset(&package, 0, sizeof(package));
    memset(&vocab, 0, sizeof(vocab));
    command.family = "auto";
    command.family_id = -1;
    command.max_new = SPLIT_MAX_NEW_TOKENS;
    command.runtime_options = (VxRuntimeOptions)VX_RUNTIME_OPTIONS_INIT;
    command.backend_policy = (VxBackendPolicy)VX_BACKEND_POLICY_INIT;
    if (argc < 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
        split_help(argv[0]);
        return argc < 2 ? 1 : 0;
    }
    command.package_arg = argv[1];
    for (int index = 2; index < argc; index++) {
        const char* arg = argv[index];
        int engine_flag = split_parse_engine_flag(arg, &command);
        if (engine_flag < 0) return 2;
        if (engine_flag > 0) continue;
        if (!strcmp(arg, "--image") && index + 1 < argc) {
            command.image_path = argv[++index];
        } else if (!strcmp(arg, "--prompt") && index + 1 < argc) {
            command.prompt = argv[++index];
        } else if (!strcmp(arg, "--family") && index + 1 < argc) {
            command.family = argv[++index];
        } else if (!strcmp(arg, "--max-new") && index + 1 < argc) {
            if (split_parse_nonnegative(argv[++index], SPLIT_MAX_NEW_TOKENS,
                                        &command.max_new) != 0) {
                fprintf(stderr, "[tinyreceipt] --max-new must be an integer from 0 through 191\n");
                return 2;
            }
        } else if (!strcmp(arg, "--shape-mode") && index + 1 < argc) {
            const char* mode = argv[++index];
            if (!strcmp(mode, "active")) {
                command.shape_mode = SPLIT_SHAPE_MODE_ACTIVE;
            } else if (!strcmp(mode, "maximum-padded")) {
                command.shape_mode = SPLIT_SHAPE_MODE_MAXIMUM_PADDED;
            } else {
                fprintf(stderr,
                        "[tinyreceipt] --shape-mode must be active or maximum-padded\n");
                return 2;
            }
        } else if (!strcmp(arg, "--timing")) {
            command.timing = 1;
        } else if (!strcmp(arg, "--warmup")) {
            if (index + 1 >= argc ||
                split_parse_nonnegative(argv[++index], SPLIT_MAX_WARMUP_RUNS,
                                        &command.warmup) != 0) {
                fprintf(stderr,
                        "[tinyreceipt] --warmup must be an integer from 0 through 20\n");
                return 2;
            }
        } else if (!strcmp(arg, "--qualify-dynamic")) {
            command.qualify_dynamic = 1;
        } else if (!strcmp(arg, "--threads")) {
            int threads;
            if (index + 1 >= argc ||
                split_parse_nonnegative(argv[++index], INT_MAX, &threads) != 0 ||
                threads == 0) {
                fprintf(stderr,
                        "[tinyreceipt] --threads must be a positive 32-bit integer\n");
                return 2;
            }
            command.runtime_options.cpu_threads = threads;
        } else {
            fprintf(stderr, "Unknown tinyreceipt split option: %s\n", arg);
            split_help(argv[0]);
            return 2;
        }
    }
    if (!command.image_path || (!command.qualify_dynamic && !command.prompt)) {
        fprintf(stderr,
                "[tinyreceipt] --image and either --prompt or --qualify-dynamic are required\n");
        split_help(argv[0]);
        return 2;
    }
    if (command.qualify_dynamic) {
        if (command.backend_policy.backend_count != 1) {
            fprintf(stderr,
                    "[tinyreceipt] --qualify-dynamic requires one explicit backend\n");
            return 2;
        }
        return split_run_dynamic_qualification(&command) == 0 ? 0 : 1;
    }
    if (command.backend_policy.backend_count == 0 &&
        split_select_backend(&command, "cpu") < 0)
        return 2;
    command.family_id = split_family_id(command.family);
    if (command.family_id == -2) {
        fprintf(stderr, "[tinyreceipt] unknown family: %s\n", command.family);
        return 2;
    }
    emit_timing = command.timing || command.runtime_options.debug;
    printf("VolvoxAI Native Runtime\n");
    {
        VxStatus status = vx_runtime_create(&command.runtime_options, &runtime, &report);
        if (status != VX_STATUS_OK) {
            split_report_failure("runtime creation", status, &report);
            goto cleanup;
        }
    }
    printf("Backend policy: %s\n", command.backend_policy.backend_count ?
           command.backend_policy.backends[0] : "cpu");
    if (split_load_package(command.package_arg, &package) != 0) goto cleanup;
    if (emit_timing) {
        fprintf(stderr,
                "[debug] tinyreceipt split ABI=%s routing=%s decoder_output=%s shape_mode=%s argmax=%s\n",
                "explicit-kv-v2",
                "runtime",
                "f32_logits",
                command.shape_mode == SPLIT_SHAPE_MODE_MAXIMUM_PADDED ?
                    "maximum-padded" : "active",
                "host-first-index");
    }
    if (split_load_vocab(&package, &vocab) != 0 ||
        split_graph_open(runtime, package.encoder.graph, package.encoder.weights,
                         &command.backend_policy, &encoder_options, "encoder",
                         emit_timing, &encoder_graph) != 0 ||
        split_graph_open(runtime, package.decoder.graph, package.decoder.weights,
                         &command.backend_policy, &decoder_options, "decoder",
                         emit_timing, &decoder_graph) != 0)
        goto cleanup;
    for (int index = 0; index < command.warmup; index++) {
        memset(&request_evidence, 0, sizeof(request_evidence));
        if (split_run_encoder(
                &encoder_graph, &package.encoder, &vocab,
                command.image_path, command.prompt, command.family_id,
                command.shape_mode, 0, &encoded, &request_evidence) != 0 ||
            split_run_decoder(
                &decoder_graph, &package.decoder, &vocab, &encoded,
                command.max_new, 0, 0, &request_evidence) != 0 ||
            !split_request_evidence_is_valid(
                &request_evidence, command.max_new) ||
            (index > 0 && !split_request_evidence_has_parity(
                &warmup_reference, &request_evidence))) {
            fprintf(stderr,
                    "[tinyreceipt] same-context warmup correctness/parity failed\n");
            goto cleanup;
        }
        if (index == 0) warmup_reference = request_evidence;
        split_encoder_output_release(&encoded);
    }
    memset(&request_evidence, 0, sizeof(request_evidence));
    if (
        split_run_encoder(&encoder_graph, &package.encoder, &vocab,
                          command.image_path, command.prompt, command.family_id,
                          command.shape_mode, emit_timing, &encoded,
                          command.warmup > 0 ? &request_evidence : NULL) != 0 ||
        split_run_decoder(&decoder_graph, &package.decoder, &vocab, &encoded,
                          command.max_new, emit_timing, 1,
                          command.warmup > 0 ? &request_evidence : NULL) != 0)
        goto cleanup;
    if (command.warmup > 0 &&
        (!split_request_evidence_is_valid(&request_evidence, command.max_new) ||
         !split_request_evidence_has_parity(
             &warmup_reference, &request_evidence))) {
        fprintf(stderr,
                "[tinyreceipt] measured request differs from same-context warmup\n");
        goto cleanup;
    }
    if (emit_timing) {
        const int bound_question_length =
            encoded.memory_length - SPLIT_IMAGE_TOKENS;
        const int logical_target_length = command.max_new + 1;
        fprintf(stderr,
                "[debug] tinyreceipt split shape mode=%s logical_Q=%d bound_Q=%d logical_M=%d bound_M=%d seed_P=1 maximum_R=%d\n",
                command.shape_mode == SPLIT_SHAPE_MODE_MAXIMUM_PADDED ?
                    "maximum-padded" : "active",
                encoded.question_length, bound_question_length,
                encoded.question_length + SPLIT_IMAGE_TOKENS,
                encoded.memory_length, logical_target_length);
    }
    if (command.warmup > 0)
        split_emit_warmup_result(command.warmup, &request_evidence);
    rc = 0;

cleanup:
    split_encoder_output_release(&encoded);
    split_graph_close(&decoder_graph);
    split_graph_close(&encoder_graph);
    split_vocab_free(&vocab);
    if (runtime) (void)vx_runtime_close(runtime, NULL);
    vx_runtime_release(runtime);
    return rc;
}
