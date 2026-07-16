#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif

#include "tiny_receipt_w8a8.h"

#include "cJSON.h"
#include "tiny_receipt_image.h"
#include "volvoxai.h"

#include <ctype.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define TINY_RECEIPT_PACKAGE_FORMAT "volvoxai-tiny-receipt-vqa-w8a8-materialized-package-v1"
#define TINY_RECEIPT_FAMILY_COUNT 8
#define TINY_RECEIPT_MAX_TENSORS 8

static const char* const k_family_names[TINY_RECEIPT_FAMILY_COUNT] = {
    "phone", "address", "store", "item_row", "item_math", "item_lookup", "math", "other",
};

typedef struct {
    char** items;
    int count;
    int pad;
    int bos;
    int eos;
    int unk;
} TinyReceiptVocab;

typedef struct {
    char config[PATH_MAX];
    char output_name[128];
    char q_ids[128];
    char router_keep[128];
} TinyReceiptRouter;

typedef struct {
    char config[PATH_MAX];
    char output_name[128];
    char image[128];
    char q_ids[128];
    char router_keep[128];
    char memory_keep[128];
    char y_ids[128];
    char y_keep[128];
} TinyReceiptFamily;

typedef struct {
    char package_dir[PATH_MAX];
    char weights[PATH_MAX];
    char vocab_path[PATH_MAX];
    int resize_width;
    int resize_height;
    int token_pad;
    int token_bos;
    int token_eos;
    int token_unk;
    TinyReceiptRouter router;
    TinyReceiptFamily families[TINY_RECEIPT_FAMILY_COUNT];
} TinyReceiptPackage;

typedef struct {
    const char* package_arg;
    const char* image_path;
    const char* prompt;
    const char* family;
    int max_new;
    int incremental;
    VolvoxAIEngineOptions engine;
} TinyReceiptCommand;

static double tiny_receipt_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void tiny_receipt_help(const char* argv0) {
    printf("Usage: %s <package-dir|package_manifest.json> --image <png|jpg> --prompt <text> [options]\n\n", argv0);
    printf("Run a materialized TinyReceiptVQA W8A8 package with its typed I32/QArgMax ABI.\n");
    printf("The command always runs the hard router first, then an explicit family graph.\n\n");
    printf("Options:\n");
    printf("  --image <file>               Receipt PNG/JPEG (required).\n");
    printf("  --prompt <text>              Question encoded by the packaged CharVocab (required).\n");
    printf("  --family <name|auto>         Explicit family override; auto uses the router (default auto).\n");
    printf("  --max-new <n>                Maximum generated character tokens (default 192).\n");
    printf("  --incremental                Cache input-independent branches; CPU also reuses decoder rows/KV.\n");
    printf("  --vulkan | --opengl | --metal | --nnapi\n");
    printf("  --debug\n");
    printf("\nOrdinary full-graph forward remains the default; --incremental enables backward-compatible cached decoding.\n");
}

static int tiny_receipt_is_dir(const char* path) {
    struct stat st;
    return path && path[0] && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int tiny_receipt_copy_string(char* out, size_t out_size, const char* value, const char* label) {
    if (!out || out_size == 0 || !value || !value[0] || strlen(value) >= out_size) {
        fprintf(stderr, "[tinyreceipt] invalid %s\n", label);
        return -1;
    }
    memcpy(out, value, strlen(value) + 1);
    return 0;
}

static int tiny_receipt_read_text(const char* path, char** out) {
    FILE* file = NULL;
    long size = 0;
    char* text = NULL;
    if (!path || !out) return -1;
    *out = NULL;
    file = fopen(path, "rb");
    if (!file) return -1;
    if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0 || (unsigned long)size > SIZE_MAX - 1) {
        fclose(file);
        return -1;
    }
    text = (char*)malloc((size_t)size + 1);
    if (!text) { fclose(file); return -1; }
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

static int tiny_receipt_path_is_within(const char* root, const char* path) {
    size_t root_len;
    if (!root || !path) return 0;
    root_len = strlen(root);
    return strncmp(root, path, root_len) == 0 &&
           (path[root_len] == 0 || path[root_len] == '/');
}

/* Package manifests are data, not an instruction to read arbitrary files.
 * Resolve each referenced asset through realpath and require it to remain in
 * the package directory. */
static int tiny_receipt_resolve_package_file(const char* package_dir, const char* relative,
                                             char* out, size_t out_size, const char* label) {
    char candidate[PATH_MAX];
    char root_real[PATH_MAX];
    char file_real[PATH_MAX];
    if (!package_dir || !relative || !relative[0] || relative[0] == '/' ||
        strstr(relative, "..") != NULL) {
        fprintf(stderr, "[tinyreceipt] %s must be a bounded package-relative path\n", label);
        return -1;
    }
    if (!realpath(package_dir, root_real) ||
        snprintf(candidate, sizeof(candidate), "%s/%s", root_real, relative) >= (int)sizeof(candidate) ||
        !realpath(candidate, file_real) || !tiny_receipt_path_is_within(root_real, file_real)) {
        fprintf(stderr, "[tinyreceipt] cannot resolve bounded %s: %s\n", label, relative);
        return -1;
    }
    return tiny_receipt_copy_string(out, out_size, file_real, label);
}

static const cJSON* tiny_receipt_object(const cJSON* object, const char* key, const char* label) {
    const cJSON* value = object ? cJSON_GetObjectItemCaseSensitive((cJSON*)object, key) : NULL;
    if (!cJSON_IsObject(value)) {
        fprintf(stderr, "[tinyreceipt] manifest requires object %s\n", label);
        return NULL;
    }
    return value;
}

static const char* tiny_receipt_string(const cJSON* object, const char* key, const char* label) {
    const cJSON* value = object ? cJSON_GetObjectItemCaseSensitive((cJSON*)object, key) : NULL;
    if (!cJSON_IsString(value) || !value->valuestring || !value->valuestring[0]) {
        fprintf(stderr, "[tinyreceipt] manifest requires string %s\n", label);
        return NULL;
    }
    return value->valuestring;
}

/* The materializer records an input's dtype/shape inline under its canonical
 * input name.  Accept a string alias too, but never infer a name from an
 * arbitrary field: the object key remains the ABI name unless it explicitly
 * provides a non-empty `name`. */
static const char* tiny_receipt_input_name(const cJSON* inputs, const char* key, const char* label) {
    const cJSON* value = inputs ? cJSON_GetObjectItemCaseSensitive((cJSON*)inputs, key) : NULL;
    if (cJSON_IsString(value) && value->valuestring && value->valuestring[0]) return value->valuestring;
    if (cJSON_IsObject(value)) {
        const cJSON* name = cJSON_GetObjectItemCaseSensitive((cJSON*)value, "name");
        if (cJSON_IsString(name) && name->valuestring && name->valuestring[0]) return name->valuestring;
        return key;
    }
    fprintf(stderr, "[tinyreceipt] manifest requires input declaration %s\n", label);
    return NULL;
}

static int tiny_receipt_number(const cJSON* object, const char* key, const char* label, int* out) {
    const cJSON* value = object ? cJSON_GetObjectItemCaseSensitive((cJSON*)object, key) : NULL;
    if (!cJSON_IsNumber(value) || value->valuedouble < 0 || value->valuedouble > INT_MAX ||
        value->valuedouble != (double)value->valueint) {
        fprintf(stderr, "[tinyreceipt] manifest requires non-negative integer %s\n", label);
        return -1;
    }
    *out = value->valueint;
    return 0;
}

static int tiny_receipt_parse_input_names(const cJSON* interface, TinyReceiptFamily* family) {
    const cJSON* inputs = tiny_receipt_object(interface, "inputs", "family.interface.inputs");
    const char* output = tiny_receipt_string(interface, "output_name", "family.interface.output_name");
    const char* image;
    const char* q_ids;
    const char* router_keep;
    const char* memory_keep;
    const char* y_ids;
    const char* y_keep;
    if (!inputs || !output) return -1;
    image = tiny_receipt_input_name(inputs, "image", "family.interface.inputs.image");
    q_ids = tiny_receipt_input_name(inputs, "q_ids", "family.interface.inputs.q_ids");
    router_keep = tiny_receipt_input_name(inputs, "router_keep", "family.interface.inputs.router_keep");
    memory_keep = tiny_receipt_input_name(inputs, "memory_keep", "family.interface.inputs.memory_keep");
    y_ids = tiny_receipt_input_name(inputs, "y_ids", "family.interface.inputs.y_ids");
    y_keep = tiny_receipt_input_name(inputs, "y_keep", "family.interface.inputs.y_keep");
    if (!image || !q_ids || !router_keep || !memory_keep || !y_ids || !y_keep) return -1;
    return tiny_receipt_copy_string(family->output_name, sizeof(family->output_name), output, "family output name") ||
           tiny_receipt_copy_string(family->image, sizeof(family->image), image, "family image input") ||
           tiny_receipt_copy_string(family->q_ids, sizeof(family->q_ids), q_ids, "family q_ids input") ||
           tiny_receipt_copy_string(family->router_keep, sizeof(family->router_keep), router_keep, "family router_keep input") ||
           tiny_receipt_copy_string(family->memory_keep, sizeof(family->memory_keep), memory_keep, "family memory_keep input") ||
           tiny_receipt_copy_string(family->y_ids, sizeof(family->y_ids), y_ids, "family y_ids input") ||
           tiny_receipt_copy_string(family->y_keep, sizeof(family->y_keep), y_keep, "family y_keep input");
}

static int tiny_receipt_load_package(const char* package_arg, TinyReceiptPackage* package) {
    char manifest_path[PATH_MAX];
    char manifest_real[PATH_MAX];
    char* text = NULL;
    cJSON* root = NULL;
    const cJSON* weights;
    const cJSON* router;
    const cJSON* router_inputs;
    const cJSON* families;
    const cJSON* vocab;
    const cJSON* token_ids;
    const cJSON* preprocess;
    const cJSON* resize;
    const char* format;
    const char* weights_file;
    const char* router_config;
    const char* router_output;
    const char* q_ids;
    const char* router_keep;
    const char* vocab_file;
    const char* color_space;
    const char* resample;
    const char* normalization;
    int rc = -1;

    if (!package_arg || !package) return -1;
    memset(package, 0, sizeof(*package));
    if (tiny_receipt_is_dir(package_arg)) {
        if (snprintf(manifest_path, sizeof(manifest_path), "%s/package_manifest.json", package_arg) >=
            (int)sizeof(manifest_path)) {
            fprintf(stderr, "[tinyreceipt] package path is too long\n");
            return -1;
        }
    } else if (tiny_receipt_copy_string(manifest_path, sizeof(manifest_path), package_arg, "package manifest path") != 0) {
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
        if (tiny_receipt_copy_string(package->package_dir, sizeof(package->package_dir), manifest_real,
                                     "package directory") != 0) return -1;
        *slash = '/';
    }
    if (tiny_receipt_read_text(manifest_real, &text) != 0 || !(root = cJSON_Parse(text))) {
        fprintf(stderr, "[tinyreceipt] cannot parse package manifest: %s\n", manifest_path);
        goto cleanup;
    }
    format = tiny_receipt_string(root, "format", "format");
    if (!format || strcmp(format, TINY_RECEIPT_PACKAGE_FORMAT) != 0) {
        fprintf(stderr, "[tinyreceipt] unsupported package format (not a materialized TinyReceipt W8A8 package)\n");
        goto cleanup;
    }
    weights = tiny_receipt_object(root, "weights", "weights");
    router = tiny_receipt_object(root, "router", "router");
    families = tiny_receipt_object(root, "explicit_families", "explicit_families");
    vocab = tiny_receipt_object(root, "vocab", "vocab");
    preprocess = tiny_receipt_object(root, "preprocessing", "preprocessing");
    if (!weights || !router || !families || !vocab || !preprocess) goto cleanup;
    weights_file = tiny_receipt_string(weights, "file", "weights.file");
    router_config = tiny_receipt_string(router, "config", "router.config");
    router_output = tiny_receipt_string(router, "output_name", "router.output_name");
    router_inputs = tiny_receipt_object(router, "inputs", "router.inputs");
    vocab_file = tiny_receipt_string(vocab, "file", "vocab.file");
    token_ids = tiny_receipt_object(vocab, "token_ids", "vocab.token_ids");
    color_space = tiny_receipt_string(preprocess, "color_space", "preprocessing.color_space");
    resize = tiny_receipt_object(preprocess, "resize", "preprocessing.resize");
    normalization = tiny_receipt_string(preprocess, "normalization", "preprocessing.normalization");
    if (!weights_file || !router_config || !router_output || !router_inputs || !vocab_file || !token_ids || !color_space ||
        !resize || !normalization) goto cleanup;
    q_ids = tiny_receipt_input_name(router_inputs, "q_ids", "router.inputs.q_ids");
    router_keep = tiny_receipt_input_name(router_inputs, "router_keep", "router.inputs.router_keep");
    resample = tiny_receipt_string(resize, "resample", "preprocessing.resize.resample");
    if (!q_ids || !router_keep || !resample ||
        tiny_receipt_number(resize, "width", "preprocessing.resize.width", &package->resize_width) != 0 ||
        tiny_receipt_number(resize, "height", "preprocessing.resize.height", &package->resize_height) != 0 ||
        tiny_receipt_number(token_ids, "pad", "vocab.token_ids.pad", &package->token_pad) != 0 ||
        tiny_receipt_number(token_ids, "bos", "vocab.token_ids.bos", &package->token_bos) != 0 ||
        tiny_receipt_number(token_ids, "eos", "vocab.token_ids.eos", &package->token_eos) != 0 ||
        tiny_receipt_number(token_ids, "unk", "vocab.token_ids.unk", &package->token_unk) != 0) goto cleanup;
    if (strcmp(color_space, "grayscale") != 0 || strcmp(resample, "bilinear") != 0 ||
        (strcmp(normalization, "minus-one-one") != 0 && strcmp(normalization, "minus_one_one") != 0)) {
        fprintf(stderr, "[tinyreceipt] package preprocessing must be grayscale + bilinear + minus-one-one\n");
        goto cleanup;
    }
    if (package->resize_width <= 0 || package->resize_height <= 0 ||
        tiny_receipt_resolve_package_file(package->package_dir, weights_file, package->weights,
                                          sizeof(package->weights), "weights.file") != 0 ||
        tiny_receipt_resolve_package_file(package->package_dir, router_config, package->router.config,
                                          sizeof(package->router.config), "router.config") != 0 ||
        tiny_receipt_resolve_package_file(package->package_dir, vocab_file, package->vocab_path,
                                          sizeof(package->vocab_path), "vocab.file") != 0 ||
        tiny_receipt_copy_string(package->router.output_name, sizeof(package->router.output_name), router_output,
                                 "router output name") != 0 ||
        tiny_receipt_copy_string(package->router.q_ids, sizeof(package->router.q_ids), q_ids, "router q_ids input") != 0 ||
        tiny_receipt_copy_string(package->router.router_keep, sizeof(package->router.router_keep), router_keep,
                                 "router router_keep input") != 0) goto cleanup;

    for (int index = 0; index < TINY_RECEIPT_FAMILY_COUNT; index++) {
        const char* family_name = k_family_names[index];
        const cJSON* family = cJSON_GetObjectItemCaseSensitive((cJSON*)families, family_name);
        const cJSON* interface;
        const char* config;
        if (!cJSON_IsObject(family)) {
            fprintf(stderr, "[tinyreceipt] manifest requires explicit family %s\n", family_name);
            goto cleanup;
        }
        config = tiny_receipt_string(family, "config", "explicit_families.<name>.config");
        interface = tiny_receipt_object(family, "interface", "explicit_families.<name>.interface");
        if (!config || !interface ||
            tiny_receipt_resolve_package_file(package->package_dir, config, package->families[index].config,
                                              sizeof(package->families[index].config), "family config") != 0 ||
            tiny_receipt_parse_input_names(interface, &package->families[index]) != 0) goto cleanup;
    }
    rc = 0;

cleanup:
    cJSON_Delete(root);
    free(text);
    return rc;
}

static void tiny_receipt_vocab_free(TinyReceiptVocab* vocab) {
    if (!vocab) return;
    for (int index = 0; index < vocab->count; index++) free(vocab->items[index]);
    free(vocab->items);
    memset(vocab, 0, sizeof(*vocab));
}

static int tiny_receipt_vocab_find(const TinyReceiptVocab* vocab, const char* text, size_t length) {
    if (!vocab || !text) return -1;
    for (int index = 0; index < vocab->count; index++) {
        const char* item = vocab->items[index];
        if (item && strlen(item) == length && memcmp(item, text, length) == 0) return index;
    }
    return -1;
}

static int tiny_receipt_load_vocab(const TinyReceiptPackage* package, TinyReceiptVocab* vocab) {
    char* text = NULL;
    cJSON* root = NULL;
    cJSON* itos = NULL;
    int rc = -1;
    if (!package || !vocab) return -1;
    memset(vocab, 0, sizeof(*vocab));
    vocab->pad = vocab->bos = vocab->eos = vocab->unk = -1;
    if (tiny_receipt_read_text(package->vocab_path, &text) != 0 || !(root = cJSON_Parse(text)) ||
        !(itos = cJSON_GetObjectItemCaseSensitive(root, "itos")) || !cJSON_IsArray(itos)) {
        fprintf(stderr, "[tinyreceipt] cannot parse CharVocab JSON: %s\n", package->vocab_path);
        goto cleanup;
    }
    vocab->count = cJSON_GetArraySize(itos);
    if (vocab->count <= 0 || vocab->count > 65536 ||
        !(vocab->items = (char**)calloc((size_t)vocab->count, sizeof(*vocab->items)))) {
        fprintf(stderr, "[tinyreceipt] invalid CharVocab length\n");
        goto cleanup;
    }
    for (int index = 0; index < vocab->count; index++) {
        cJSON* item = cJSON_GetArrayItem(itos, index);
        size_t length;
        if (!cJSON_IsString(item) || !item->valuestring || (length = strlen(item->valuestring)) > 4096 ||
            !(vocab->items[index] = (char*)malloc(length + 1))) {
            fprintf(stderr, "[tinyreceipt] invalid CharVocab entry %d\n", index);
            goto cleanup;
        }
        memcpy(vocab->items[index], item->valuestring, length + 1);
    }
    vocab->pad = tiny_receipt_vocab_find(vocab, "<pad>", 5);
    vocab->bos = tiny_receipt_vocab_find(vocab, "<bos>", 5);
    vocab->eos = tiny_receipt_vocab_find(vocab, "<eos>", 5);
    vocab->unk = tiny_receipt_vocab_find(vocab, "<unk>", 5);
    if (vocab->pad < 0 || vocab->bos < 0 || vocab->eos < 0 || vocab->unk < 0) {
        fprintf(stderr, "[tinyreceipt] CharVocab must contain <pad>, <bos>, <eos>, and <unk>\n");
        goto cleanup;
    }
    if (vocab->pad != package->token_pad || vocab->bos != package->token_bos ||
        vocab->eos != package->token_eos || vocab->unk != package->token_unk) {
        fprintf(stderr, "[tinyreceipt] package token IDs do not match its CharVocab\n");
        goto cleanup;
    }
    rc = 0;

cleanup:
    cJSON_Delete(root);
    free(text);
    if (rc != 0) tiny_receipt_vocab_free(vocab);
    return rc;
}

static int tiny_receipt_utf8_len(const unsigned char* text, size_t remaining) {
    unsigned char first;
    if (!text || remaining == 0) return 0;
    first = text[0];
    if (first < 0x80) return 1;
    if (first >= 0xc2 && first <= 0xdf && remaining >= 2 && (text[1] & 0xc0) == 0x80) return 2;
    if (first >= 0xe0 && first <= 0xef && remaining >= 3 && (text[1] & 0xc0) == 0x80 &&
        (text[2] & 0xc0) == 0x80) return 3;
    if (first >= 0xf0 && first <= 0xf4 && remaining >= 4 && (text[1] & 0xc0) == 0x80 &&
        (text[2] & 0xc0) == 0x80 && (text[3] & 0xc0) == 0x80) return 4;
    return 1;
}

/* Python's clean_text() uses re.sub(r"\\s+", " ", ...) then strip().  This
 * covers the Unicode whitespace sequences users are likely to pass in UTF-8,
 * while preserving every non-whitespace code point byte-for-byte for the
 * CharVocab lookup below. */
static int tiny_receipt_is_unicode_space(const unsigned char* text, int length) {
    if (length == 1) return isspace(text[0]) != 0;
    if (length == 2) return text[0] == 0xc2 && text[1] == 0xa0;
    if (length != 3) return 0;
    if (text[0] == 0xe1 && text[1] == 0x9a && text[2] == 0x80) return 1; /* U+1680 */
    if (text[0] == 0xe2 && text[1] == 0x80 && text[2] >= 0x80 && text[2] <= 0x8a) return 1;
    if (text[0] == 0xe2 && text[1] == 0x80 && (text[2] == 0xa8 || text[2] == 0xa9 || text[2] == 0xaf)) return 1;
    if (text[0] == 0xe2 && text[1] == 0x81 && text[2] == 0x9f) return 1;
    return text[0] == 0xe3 && text[1] == 0x80 && text[2] == 0x80; /* U+3000 */
}

static char* tiny_receipt_clean_text(const char* input) {
    const unsigned char* src = (const unsigned char*)(input ? input : "");
    size_t input_length = strlen((const char*)src);
    char* output = (char*)malloc(input_length + 1);
    size_t in = 0;
    size_t out = 0;
    int pending_space = 0;
    if (!output) return NULL;
    while (in < input_length) {
        int length = tiny_receipt_utf8_len(src + in, input_length - in);
        if (tiny_receipt_is_unicode_space(src + in, length)) {
            pending_space = out > 0;
            in += (size_t)length;
            continue;
        }
        if (pending_space) output[out++] = ' ';
        pending_space = 0;
        memcpy(output + out, src + in, (size_t)length);
        out += (size_t)length;
        in += (size_t)length;
    }
    output[out] = 0;
    return output;
}

/* Exact CharVocab policy: clean question text, encode one Unicode code point
 * at a time, append EOS, then truncate and force EOS at the final available
 * position. */
static int tiny_receipt_encode_question(const TinyReceiptVocab* vocab, const char* prompt,
                                        int32_t* ids, int capacity) {
    char* clean;
    size_t length;
    size_t offset = 0;
    int count = 0;
    if (!vocab || !ids || capacity <= 0) return -1;
    clean = tiny_receipt_clean_text(prompt);
    if (!clean) return -1;
    length = strlen(clean);
    while (offset < length && count < capacity) {
        int bytes = tiny_receipt_utf8_len((const unsigned char*)clean + offset, length - offset);
        int id = tiny_receipt_vocab_find(vocab, clean + offset, (size_t)bytes);
        ids[count++] = id >= 0 ? id : vocab->unk;
        offset += (size_t)bytes;
    }
    if (count < capacity) ids[count++] = vocab->eos;
    else ids[capacity - 1] = vocab->eos;
    free(clean);
    return count;
}

static int tiny_receipt_find_family(const char* value) {
    if (!value || !value[0] || strcmp(value, "auto") == 0) return -1;
    for (int index = 0; index < TINY_RECEIPT_FAMILY_COUNT; index++) {
        if (strcmp(value, k_family_names[index]) == 0) return index;
    }
    return -2;
}

static void tiny_receipt_engine_options_init(VolvoxAIEngineOptions* options) {
    memset(options, 0, sizeof(*options));
    options->backend = VOLVOXAI_BACKEND_CPU;
}

static const char* tiny_receipt_requested_backend_name(VolvoxAIEngineBackend backend) {
    switch (backend) {
        case VOLVOXAI_BACKEND_VULKAN: return "Vulkan";
        case VOLVOXAI_BACKEND_OPENGL: return "OpenGL";
        case VOLVOXAI_BACKEND_METAL: return "Metal";
        case VOLVOXAI_BACKEND_NNAPI: return "NNAPI";
        case VOLVOXAI_BACKEND_CPU:
        default: return "CPU";
    }
}

static int tiny_receipt_select_backend(VolvoxAIEngineOptions* options,
                                       VolvoxAIEngineBackend backend) {
    if (options->backend != VOLVOXAI_BACKEND_CPU && options->backend != backend) {
        fprintf(stderr, "[tinyreceipt] pass at most one accelerator backend flag\n");
        return -1;
    }
    options->backend = backend;
    return 1;
}

static int tiny_receipt_parse_engine_flag(const char* arg,
                                          VolvoxAIEngineOptions* options) {
    if (!strcmp(arg, "--vulkan")) {
        return tiny_receipt_select_backend(options, VOLVOXAI_BACKEND_VULKAN);
    }
    if (!strcmp(arg, "--opengl")) {
        return tiny_receipt_select_backend(options, VOLVOXAI_BACKEND_OPENGL);
    }
    if (!strcmp(arg, "--metal")) {
        return tiny_receipt_select_backend(options, VOLVOXAI_BACKEND_METAL);
    }
    if (!strcmp(arg, "--nnapi")) {
        return tiny_receipt_select_backend(options, VOLVOXAI_BACKEND_NNAPI);
    }
    if (!strcmp(arg, "--debug")) {
        options->debug = 1;
        return 1;
    }
    return 0;
}

static int tiny_receipt_configure_engine(const VolvoxAIEngineOptions* options) {
    printf("VolvoxAI Native Engine\n");
    if (volvoxai_engine_configure(options) != 0) {
        fprintf(stderr, "[tinyreceipt] cannot configure requested backend: %s\n",
                tiny_receipt_requested_backend_name(options->backend));
        return -1;
    }
    printf("Backend: %s\n", volvoxai_engine_backend_name());
    return 0;
}

static int tiny_receipt_info_i32_2d(const char* name, int* sequence) {
    long numel = 0;
    int shape[TINY_RECEIPT_MAX_TENSORS] = {0};
    int ndim = 0;
    int dtype = -1;
    size_t element_size = 0;
    if (volvoxai_engine_tensor_info_ex(name, &numel, shape, &ndim, &dtype, &element_size) != 0 ||
        !volvoxai_engine_is_graph_input(name) || dtype != VOLVOXAI_DTYPE_I32 || element_size != sizeof(int32_t) ||
        ndim != 2 || shape[0] != 1 || shape[1] <= 0 || numel != shape[1]) {
        fprintf(stderr, "[tinyreceipt] required graph input %s must be I32 [1,S]\n", name);
        return -1;
    }
    *sequence = shape[1];
    return 0;
}

static int tiny_receipt_info_image(const char* name, int expected_width, int expected_height, long* numel,
                                   int* shape_out) {
    int shape[TINY_RECEIPT_MAX_TENSORS] = {0};
    int ndim = 0;
    int dtype = -1;
    size_t element_size = 0;
    if (volvoxai_engine_tensor_info_ex(name, numel, shape, &ndim, &dtype, &element_size) != 0 ||
        !volvoxai_engine_is_graph_input(name) || dtype != VOLVOXAI_DTYPE_F32 || element_size != sizeof(float) ||
        ndim != 4 || shape[0] != 1 || shape[1] != expected_height || shape[2] != expected_width || shape[3] != 1 ||
        *numel != (long)shape[1] * shape[2]) {
        fprintf(stderr, "[tinyreceipt] required image input %s must be F32 [1,%d,%d,1]\n",
                name, expected_height, expected_width);
        return -1;
    }
    memcpy(shape_out, shape, sizeof(shape));
    return 0;
}

static int tiny_receipt_set_i32(const char* name, const int32_t* data, int count) {
    if (count < 0 || (size_t)count > SIZE_MAX / sizeof(*data) ||
        volvoxai_engine_set_input_raw(name, VOLVOXAI_DTYPE_I32, data, (size_t)count * sizeof(*data)) != 0) {
        fprintf(stderr, "[tinyreceipt] could not set typed I32 input %s\n", name);
        return -1;
    }
    return 0;
}

static int tiny_receipt_run_router(const TinyReceiptPackage* package, const TinyReceiptVocab* vocab,
                                   const char* prompt, int32_t** ids_out, int32_t** keep_out, int* q_length,
                                   int* router_family, int debug) {
    int query_length = 0;
    int keep_length = 0;
    int32_t* ids = NULL;
    int32_t* keep = NULL;
    int32_t route = -1;
    long output_numel = 0;
    int output_shape[TINY_RECEIPT_MAX_TENSORS] = {0};
    int output_ndim = 0;
    int output_dtype = -1;
    size_t output_element_size = 0;
    if (volvoxai_engine_init(package->router.config, package->weights) != 0 ||
        tiny_receipt_info_i32_2d(package->router.q_ids, &query_length) != 0 ||
        tiny_receipt_info_i32_2d(package->router.router_keep, &keep_length) != 0 || query_length != keep_length) {
        fprintf(stderr, "[tinyreceipt] router graph does not expose its declared typed ABI\n");
        goto fail;
    }
    if (!(ids = (int32_t*)malloc((size_t)query_length * sizeof(*ids))) ||
        !(keep = (int32_t*)calloc((size_t)query_length, sizeof(*keep)))) goto fail;
    for (int index = 0; index < query_length; index++) ids[index] = vocab->pad;
    {
        int count = tiny_receipt_encode_question(vocab, prompt, ids, query_length);
        if (count < 0) goto fail;
        for (int index = 0; index < query_length; index++) keep[index] = ids[index] != vocab->pad;
        if (debug) fprintf(stderr, "[debug] tinyreceipt question_tokens=%d\n", count);
    }
    if (tiny_receipt_set_i32(package->router.q_ids, ids, query_length) != 0 ||
        tiny_receipt_set_i32(package->router.router_keep, keep, query_length) != 0 ||
        volvoxai_engine_forward() != 0 ||
        volvoxai_engine_tensor_info_ex(package->router.output_name, &output_numel, output_shape, &output_ndim,
                                       &output_dtype, &output_element_size) != 0 ||
        output_dtype != VOLVOXAI_DTYPE_I32 || output_element_size != sizeof(route) || output_numel != 1 ||
        volvoxai_engine_copy_tensor_raw(package->router.output_name, &route, sizeof(route)) != 0 ||
        route < 0 || route >= TINY_RECEIPT_FAMILY_COUNT) {
        fprintf(stderr, "[tinyreceipt] router output %s must be one valid I32 family ID\n", package->router.output_name);
        goto fail;
    }
    *ids_out = ids;
    *keep_out = keep;
    *q_length = query_length;
    *router_family = route;
    return 0;

fail:
    free(ids);
    free(keep);
    return -1;
}

static int tiny_receipt_run_family(const TinyReceiptPackage* package, const TinyReceiptVocab* vocab,
                                   const TinyReceiptFamily* family, int family_id, const char* image_path,
                                   int max_new, int incremental, const int32_t* q_ids,
                                   const int32_t* router_keep, int q_length, int debug) {
    int full_q_length = 0;
    int full_router_keep_length = 0;
    int memory_length = 0;
    int decoder_length = 0;
    int decoder_keep_length = 0;
    int image_shape[TINY_RECEIPT_MAX_TENSORS] = {0};
    long image_numel = 0;
    float* image = NULL;
    int32_t* memory_keep = NULL;
    int32_t* y_ids = NULL;
    int32_t* y_keep = NULL;
    int32_t* token_ids = NULL;
    char image_error[256] = {0};
    long output_numel = 0;
    int output_shape[TINY_RECEIPT_MAX_TENSORS] = {0};
    int output_ndim = 0;
    int output_dtype = -1;
    size_t output_element_size = 0;
    int generated = 0;
    int row_decode = 0;
    int steady_steps = 0;
    VolvoxAIDecodeSession* decode_session = NULL;
    double start_ms;
    double seed_ms = 0.0;
    double steady_ms = 0.0;
    int rc = -1;

    if (volvoxai_engine_init(family->config, package->weights) != 0 ||
        tiny_receipt_info_i32_2d(family->q_ids, &full_q_length) != 0 ||
        tiny_receipt_info_i32_2d(family->router_keep, &full_router_keep_length) != 0 ||
        tiny_receipt_info_i32_2d(family->memory_keep, &memory_length) != 0 ||
        tiny_receipt_info_i32_2d(family->y_ids, &decoder_length) != 0 ||
        tiny_receipt_info_i32_2d(family->y_keep, &decoder_keep_length) != 0 ||
        tiny_receipt_info_image(family->image, package->resize_width, package->resize_height, &image_numel, image_shape) != 0 ||
        full_q_length != q_length || full_router_keep_length != q_length || decoder_length != decoder_keep_length ||
        memory_length < q_length) {
        fprintf(stderr, "[tinyreceipt] family graph does not match the router or typed sequence ABI\n");
        goto cleanup;
    }
    if (!(image = (float*)malloc((size_t)image_numel * sizeof(*image))) ||
        !(memory_keep = (int32_t*)malloc((size_t)memory_length * sizeof(*memory_keep))) ||
        !(y_ids = (int32_t*)malloc((size_t)decoder_length * sizeof(*y_ids))) ||
        !(y_keep = (int32_t*)calloc((size_t)decoder_length, sizeof(*y_keep)))) goto cleanup;
    if (tiny_receipt_load_image_to_tensor(image_path, image, image_shape, 4,
                                          image_error, sizeof(image_error)) != 0) {
        fprintf(stderr, "[tinyreceipt] image load failed: %s\n", image_error[0] ? image_error : image_path);
        goto cleanup;
    }
    for (int index = 0; index < memory_length - q_length; index++) memory_keep[index] = 1;
    memcpy(memory_keep + (memory_length - q_length), router_keep, (size_t)q_length * sizeof(*memory_keep));
    for (int index = 0; index < decoder_length; index++) y_ids[index] = vocab->pad;
    y_ids[0] = vocab->bos;
    y_keep[0] = 1;

    if (volvoxai_engine_set_input_raw(family->image, VOLVOXAI_DTYPE_F32, image,
                                      (size_t)image_numel * sizeof(*image)) != 0 ||
        tiny_receipt_set_i32(family->q_ids, q_ids, q_length) != 0 ||
        tiny_receipt_set_i32(family->router_keep, router_keep, q_length) != 0 ||
        tiny_receipt_set_i32(family->memory_keep, memory_keep, memory_length) != 0) goto cleanup;
    if (volvoxai_engine_tensor_info_ex(family->output_name, &output_numel, output_shape, &output_ndim,
                                       &output_dtype, &output_element_size) != 0 ||
        output_dtype != VOLVOXAI_DTYPE_I32 || output_element_size != sizeof(*token_ids) || output_ndim != 2 ||
        output_shape[0] != 1 || output_shape[1] != decoder_length || output_numel != decoder_length ||
        !(token_ids = (int32_t*)malloc((size_t)decoder_length * sizeof(*token_ids)))) {
        fprintf(stderr, "[tinyreceipt] family output %s must be I32 [1,max_out_len] QArgMax token IDs\n",
                family->output_name);
        goto cleanup;
    }
    if (max_new > decoder_length) max_new = decoder_length;
    if (incremental) {
        VolvoxAIDecodeSessionOptions decode_options = VOLVOXAI_DECODE_SESSION_OPTIONS_INIT;
        decode_session = volvoxai_engine_decode_session_create(&decode_options);
        if (!decode_session) {
            fprintf(stderr, "[tinyreceipt] could not create the incremental decode session\n");
            goto cleanup;
        }
        row_decode = volvoxai_engine_decode_session_mode(decode_session) ==
            VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW;
    }
    start_ms = tiny_receipt_now_ms();
    for (int step = 0; step < max_new; step++) {
        int32_t next;
        double step_start_ms = debug ? tiny_receipt_now_ms() : 0.0;
        if (tiny_receipt_set_i32(family->y_ids, y_ids, decoder_length) != 0 ||
            tiny_receipt_set_i32(family->y_keep, y_keep, decoder_length) != 0 ||
            (incremental
                ? (step == 0
                    ? volvoxai_engine_decode_session_seed(decode_session)
                    : volvoxai_engine_decode_session_step(decode_session, step))
                : volvoxai_engine_forward()) != 0 ||
            volvoxai_engine_copy_tensor_raw(family->output_name, token_ids,
                                            (size_t)decoder_length * sizeof(*token_ids)) != 0) {
            fprintf(stderr, "[tinyreceipt] typed family forward failed at token %d\n", step);
            goto cleanup;
        }
        if (debug) {
            double step_ms = tiny_receipt_now_ms() - step_start_ms;
            if (step == 0) seed_ms = step_ms;
            else {
                steady_ms += step_ms;
                steady_steps++;
            }
        }
        next = token_ids[step]; /* QArgMax has already reduced logits; never re-argmax F32. */
        if (debug) fprintf(stderr, "[debug] tinyreceipt family=%s step=%d token=%d\n",
                           k_family_names[family_id], step, next);
        if (next < 0 || next >= vocab->count) {
            fprintf(stderr, "[tinyreceipt] QArgMax emitted out-of-vocabulary token %d\n", next);
            goto cleanup;
        }
        if (next == vocab->eos) break;
        if (next != vocab->pad && next != vocab->bos) fputs(vocab->items[next], stdout);
        fflush(stdout);
        generated++;
        if (step + 1 < decoder_length) {
            y_ids[step + 1] = next;
            y_keep[step + 1] = 1;
        }
    }
    printf("\n");
    if (debug) {
        double elapsed = tiny_receipt_now_ms() - start_ms;
        double tokens_per_second = generated > 0 && elapsed > 0.0 ? (double)generated * 1000.0 / elapsed : 0.0;
        fprintf(stderr, "[debug] tinyreceipt family=%s tokens=%d total=%.3f ms tok/s=%.2f (%s)\n",
                k_family_names[family_id], generated, elapsed, tokens_per_second,
                incremental
                    ? (row_decode ? "incremental dependency + CPU row KV cache"
                                  : "incremental dependency cache")
                    : "ordinary forward only");
        if (incremental) {
            double steady_mean_ms = steady_steps > 0
                ? steady_ms / (double)steady_steps : 0.0;
            double steady_tokens_per_second = steady_ms > 0.0
                ? (double)steady_steps * 1000.0 / steady_ms : 0.0;
            fprintf(stderr,
                    "[debug] tinyreceipt timing seed=%.3f ms steady_steps=%d "
                    "steady_mean=%.3f ms steady_tok/s=%.2f\n",
                    seed_ms, steady_steps, steady_mean_ms,
                    steady_tokens_per_second);
        }
    }
    rc = 0;

cleanup:
    volvoxai_engine_decode_session_destroy(decode_session);
    free(image);
    free(memory_keep);
    free(y_ids);
    free(y_keep);
    free(token_ids);
    return rc;
}

int tiny_receipt_w8a8_run(int argc, char** argv) {
    TinyReceiptCommand command;
    TinyReceiptPackage package;
    TinyReceiptVocab vocab;
    int32_t* q_ids = NULL;
    int32_t* router_keep = NULL;
    int q_length = 0;
    int predicted_family = -1;
    int selected_family = -1;
    int rc = 1;

    memset(&command, 0, sizeof(command));
    memset(&vocab, 0, sizeof(vocab));
    command.family = "auto";
    command.max_new = 192;
    tiny_receipt_engine_options_init(&command.engine);
    if (argc < 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
        tiny_receipt_help(argv[0]);
        return argc < 2 ? 1 : 0;
    }
    command.package_arg = argv[1];
    for (int index = 2; index < argc; index++) {
        const char* arg = argv[index];
        int engine_flag = tiny_receipt_parse_engine_flag(arg, &command.engine);
        if (engine_flag < 0) return 2;
        if (engine_flag > 0) continue;
        if (!strcmp(arg, "--image") && index + 1 < argc) command.image_path = argv[++index];
        else if (!strcmp(arg, "--prompt") && index + 1 < argc) command.prompt = argv[++index];
        else if (!strcmp(arg, "--family") && index + 1 < argc) command.family = argv[++index];
        else if (!strcmp(arg, "--max-new") && index + 1 < argc) command.max_new = atoi(argv[++index]);
        else if (!strcmp(arg, "--incremental")) command.incremental = 1;
        else { fprintf(stderr, "Unknown tinyreceipt option: %s\n", arg); tiny_receipt_help(argv[0]); return 2; }
    }
    if (!command.image_path || !command.prompt || command.max_new < 1) {
        fprintf(stderr, "[tinyreceipt] --image, --prompt, and positive --max-new are required\n");
        tiny_receipt_help(argv[0]);
        return 2;
    }
    if ((selected_family = tiny_receipt_find_family(command.family)) == -2) {
        fprintf(stderr, "[tinyreceipt] unknown family: %s\n", command.family);
        return 2;
    }
    if (tiny_receipt_configure_engine(&command.engine) != 0) goto cleanup;
    if (tiny_receipt_load_package(command.package_arg, &package) != 0 ||
        tiny_receipt_load_vocab(&package, &vocab) != 0) goto cleanup;
    if (tiny_receipt_run_router(&package, &vocab, command.prompt, &q_ids, &router_keep, &q_length,
                                &predicted_family, command.engine.debug) != 0) goto cleanup;
    if (selected_family < 0) selected_family = predicted_family;
    if (command.engine.debug) {
        fprintf(stderr, "[debug] tinyreceipt router=%s selected=%s%s\n",
                k_family_names[predicted_family], k_family_names[selected_family],
                command.family && strcmp(command.family, "auto") != 0
                    ? " (manual override)" : "");
    }
    rc = tiny_receipt_run_family(&package, &vocab, &package.families[selected_family], selected_family,
                                 command.image_path, command.max_new, command.incremental,
                                 q_ids, router_keep, q_length, command.engine.debug) == 0 ? 0 : 1;

cleanup:
    free(q_ids);
    free(router_keep);
    tiny_receipt_vocab_free(&vocab);
    volvoxai_engine_shutdown();
    return rc;
}
