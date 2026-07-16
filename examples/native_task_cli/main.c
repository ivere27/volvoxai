#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#ifndef VOLVOXAI_ENABLE_TRAINING
#define VOLVOXAI_ENABLE_TRAINING 0
#endif

#include "volvoxai.h"
#if VOLVOXAI_ENABLE_TRAINING
#include "volvoxai_training.h"
#endif
#include "cJSON.h"
#include "image_io.h"
#include "volvoxai_tokenizer.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#ifndef VOLVOXAI_VERSION
#define VOLVOXAI_VERSION "unknown"
#endif
#ifndef VOLVOXAI_GIT_COMMIT
#define VOLVOXAI_GIT_COMMIT "unknown"
#endif
#ifndef VOLVOXAI_BUILD_DATE
#define VOLVOXAI_BUILD_DATE "unknown"
#endif

int g_warmup_runs = 0;   // --warmup_runs: discarded forwards before timing
int g_num_runs = 1;      // --num_runs: timed forwards (reports first/avg/min/max)

#define MAX_BINDINGS 32

typedef struct {
    char name[128];
    char path[PATH_MAX];
} Binding;

typedef struct {
    char config[PATH_MAX];
    char weights[PATH_MAX];
    char vocab[PATH_MAX];
    char merges[PATH_MAX];
} ModelPaths;

typedef struct {
    const char* weights;
    VolvoxAIEngineOptions engine;
    Binding inputs[MAX_BINDINGS];
    Binding images[MAX_BINDINGS];
    Binding outputs[MAX_BINDINGS];
    int n_inputs;
    int n_images;
    int n_outputs;
    int image_norm;
    int image_norm_explicit;
    int execution_row;
} GraphOptions;

typedef struct {
    char** items;
    int count;
} LabelList;

static void print_release_info(void);

static double now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#endif
}

static int file_exists(const char* path) {
    struct stat st;
    return path && path[0] && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int is_dir(const char* path) {
    struct stat st;
    return path && path[0] && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static void join_path(char* out, size_t out_size, const char* a, const char* b) {
    size_t n = strlen(a);
    snprintf(out, out_size, "%s%s%s", a, (n && a[n - 1] == '/') ? "" : "/", b);
}

static int resolve_model_paths(const char* model, const char* weights, const char* vocab,
                               const char* merges, ModelPaths* out) {
    memset(out, 0, sizeof(*out));
    if (is_dir(model)) {
        join_path(out->config, sizeof(out->config), model, "config.json");
        join_path(out->weights, sizeof(out->weights), model, "model.safetensors");
        join_path(out->vocab, sizeof(out->vocab), model, "vocab.bin");
        join_path(out->merges, sizeof(out->merges), model, "merges.txt");
        if (!file_exists(out->weights)) out->weights[0] = '\0';
    } else {
        snprintf(out->config, sizeof(out->config), "%s", model);
    }
    if (weights) snprintf(out->weights, sizeof(out->weights), "%s", weights);
    if (vocab) snprintf(out->vocab, sizeof(out->vocab), "%s", vocab);
    if (merges) snprintf(out->merges, sizeof(out->merges), "%s", merges);
    if (!file_exists(out->config)) {
        fprintf(stderr, "Missing config.json: %s\n", out->config);
        return -1;
    }
    return 0;
}

static const char* resolve_labels_path(const char* model, const char* labels,
                                       char* discovered, size_t discovered_size) {
    if (labels) return labels;
    if (!is_dir(model)) return NULL;
    join_path(discovered, discovered_size, model, "labels.txt");
    return file_exists(discovered) ? discovered : NULL;
}

static int parse_binding(const char* arg, Binding* b, int require_name) {
    const char* eq = strchr(arg, '=');
    memset(b, 0, sizeof(*b));
    if (eq) {
        size_t n = (size_t)(eq - arg);
        if (n == 0 || n >= sizeof(b->name)) return -1;
        memcpy(b->name, arg, n);
        b->name[n] = 0;
        snprintf(b->path, sizeof(b->path), "%s", eq + 1);
        return b->path[0] ? 0 : -1;
    }
    if (require_name) return -1;
    snprintf(b->path, sizeof(b->path), "%s", arg);
    return b->path[0] ? 0 : -1;
}

static int has_suffix(const char* path, const char* suffix) {
    size_t lp = strlen(path), ls = strlen(suffix);
    if (lp < ls) return 0;
    return strcmp(path + lp - ls, suffix) == 0;
}

static const char* dtype_name(int dtype) {
    switch (dtype) {
        case VOLVOXAI_DTYPE_F32: return "F32";
        case VOLVOXAI_DTYPE_I8: return "I8";
        case VOLVOXAI_DTYPE_U8: return "U8";
        case VOLVOXAI_DTYPE_I32: return "I32";
        case VOLVOXAI_DTYPE_F16: return "F16";
        default: return "unknown";
    }
}

static const char* dtype_file_suffix(int dtype) {
    switch (dtype) {
        case VOLVOXAI_DTYPE_F32: return ".f32";
        case VOLVOXAI_DTYPE_I8: return ".i8";
        case VOLVOXAI_DTYPE_U8: return ".u8";
        case VOLVOXAI_DTYPE_I32: return ".i32";
        case VOLVOXAI_DTYPE_F16: return ".f16";
        default: return NULL;
    }
}

static char* read_file_bytes(const char* path, long* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0 || fseek(f, 0, SEEK_SET) != 0 ||
        (unsigned long)sz > SIZE_MAX - 1) {
        fclose(f);
        return NULL;
    }
    char* buf = (char*)malloc(sz > 0 ? (size_t)sz + 1 : 1);
    if (!buf) { fclose(f); return NULL; }
    if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[sz] = 0;
    fclose(f);
    if (out_size) *out_size = sz;
    return buf;
}

static int load_tensor_binding(const Binding* b) {
    long numel = 0;
    int shape[8] = {0};
    int ndim = 0;
    int dtype = -1;
    size_t element_size = 0;
    if (volvoxai_engine_tensor_info_ex(b->name, &numel, shape, &ndim, &dtype,
            &element_size) != 0 || volvoxai_engine_is_graph_input(b->name) != 1 ||
        numel < 0 || element_size == 0 || (size_t)numel > SIZE_MAX / element_size) {
        fprintf(stderr, "Unknown input tensor: %s\n", b->name);
        return -1;
    }

    const char* suffix = dtype_file_suffix(dtype);
    if (!suffix || !has_suffix(b->path, suffix)) {
        fprintf(stderr, "Input %s has dtype %s and requires a %s file: %s\n",
                b->name, dtype_name(dtype), suffix ? suffix : "supported raw", b->path);
        return -1;
    }

    long sz = 0;
    char* data = read_file_bytes(b->path, &sz);
    if (!data) {
        fprintf(stderr, "Cannot read input file: %s\n", b->path);
        return -1;
    }

    size_t expected = (size_t)numel * element_size;
    if (sz < 0 || (size_t)sz != expected) {
        fprintf(stderr, "Input %s expects %zu raw bytes, got %ld from %s\n",
                b->name, expected, sz, b->path);
        free(data);
        return -1;
    }
    int rc = volvoxai_engine_set_input_raw(b->name, dtype, data, expected);
    free(data);
    if (rc != 0) {
        fprintf(stderr, "Cannot set input tensor: %s\n", b->name);
        return -1;
    }
    return 0;
}

static int load_image_binding(const Binding* b, int normalize) {
    long numel = 0;
    int shape[8] = {0};
    int ndim = 0;
    int dtype = -1;
    size_t element_size = 0;
    if (volvoxai_engine_tensor_info_ex(b->name, &numel, shape, &ndim, &dtype,
            &element_size) != 0 || volvoxai_engine_is_graph_input(b->name) != 1 ||
        numel <= 0) {
        fprintf(stderr, "Unknown image input tensor: %s\n", b->name);
        return -1;
    }

    if (!((dtype == VOLVOXAI_DTYPE_F32 && element_size == sizeof(float)) ||
          ((dtype == VOLVOXAI_DTYPE_U8 || dtype == VOLVOXAI_DTYPE_I8) &&
           element_size == 1))) {
        fprintf(stderr, "Image input %s has unsupported dtype %s; expected F32, U8, or I8.\n",
                b->name, dtype_name(dtype));
        return -1;
    }
    if ((size_t)numel > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "Image input tensor is too large: %s\n", b->name);
        return -1;
    }

    float* decoded = (float*)malloc((size_t)numel * sizeof(float));
    if (!decoded) {
        fprintf(stderr, "Cannot allocate image input tensor: %s\n", b->name);
        return -1;
    }
    char err[256] = {0};
    if (volvox_load_image_to_tensor(b->path, decoded, shape, ndim,
            normalize, err, sizeof(err)) != 0) {
        fprintf(stderr, "Image load failed for %s: %s\n", b->path,
                err[0] ? err : "unknown error");
        free(decoded);
        return -1;
    }

    int rc = -1;
    if (normalize == VOLVOX_IMAGE_RAW_255 &&
        (dtype == VOLVOXAI_DTYPE_U8 || dtype == VOLVOXAI_DTYPE_I8)) {
        void* raw = malloc((size_t)numel);
        if (!raw) {
            fprintf(stderr, "Cannot allocate image input tensor: %s\n", b->name);
            free(decoded);
            return -1;
        }
        for (long i = 0; i < numel; i++) {
            int pixel = (int)lroundf(decoded[i]);
            if (pixel < 0) pixel = 0;
            if (pixel > 255) pixel = 255;
            if (dtype == VOLVOXAI_DTYPE_U8) {
                ((uint8_t*)raw)[i] = (uint8_t)pixel;
            } else {
                ((int8_t*)raw)[i] = (int8_t)(pixel - 128);
            }
        }
        rc = volvoxai_engine_set_input_raw(b->name, dtype, raw, (size_t)numel);
        free(raw);
    } else {
        rc = volvoxai_engine_set_input_f32(b->name, decoded, numel);
    }
    free(decoded);
    if (rc != 0) {
        if (dtype != VOLVOXAI_DTYPE_F32 && normalize != VOLVOX_IMAGE_RAW_255) {
            fprintf(stderr,
                    "Cannot set image input tensor %s; normalized I8/U8 inputs require "
                    "valid per-tensor quantization metadata.\n",
                    b->name);
        } else {
            fprintf(stderr, "Cannot set image input tensor: %s\n", b->name);
        }
        return -1;
    }
    return 0;
}

static int write_output_binding(const Binding* b) {
    const void* src = NULL;
    void* owned = NULL;
    size_t nbytes = 0;
    int shape[8] = {0};
    int ndim = 0;
    int dtype = -1;
    size_t element_size = 0;
    long numel = 0;
    const char* name = b->name[0] ? b->name : volvoxai_engine_graph_output_name(0);
    if (!name || !name[0] ||
        volvoxai_engine_tensor_info_ex(name, &numel, shape, &ndim, &dtype,
            &element_size) != 0 || numel <= 0 || element_size == 0 ||
        (size_t)numel > SIZE_MAX / element_size) {
        fprintf(stderr, "Unknown output tensor: %s\n", b->name[0] ? b->name : "<primary>");
        return -1;
    }

    const char* suffix = dtype_file_suffix(dtype);
    if (!suffix || !has_suffix(b->path, suffix)) {
        fprintf(stderr, "Output %s has dtype %s and requires a %s file: %s\n",
                name, dtype_name(dtype), suffix ? suffix : "supported raw", b->path);
        return -1;
    }

    int row = volvoxai_engine_execution_row();
    if (row >= 0 && ndim >= 2) {
        if (dtype != VOLVOXAI_DTYPE_F32 || element_size != sizeof(float)) {
            fprintf(stderr, "--last-token output currently requires an F32 tensor: %s\n", name);
            return -1;
        }
        int count = 0;
        src = volvoxai_engine_tensor_row_f32(name, row, &count);
        if (!src || count <= 0 || (size_t)count > SIZE_MAX / sizeof(float)) {
            fprintf(stderr, "No output row available for %s\n", name);
            return -1;
        }
        nbytes = (size_t)count * sizeof(float);
    } else {
        nbytes = (size_t)numel * element_size;
        owned = malloc(nbytes);
        if (!owned || volvoxai_engine_copy_tensor_raw(name, owned, nbytes) != 0) {
            free(owned);
            fprintf(stderr, "Cannot snapshot output tensor: %s\n", name);
            return -1;
        }
        src = owned;
    }
    FILE* f = fopen(b->path, "wb");
    if (!f) {
        fprintf(stderr, "Cannot open output file: %s\n", b->path);
        free(owned);
        return -1;
    }
    int write_failed = fwrite(src, 1, nbytes, f) != nbytes;
    int close_failed = fclose(f) != 0;
    free(owned);
    if (write_failed || close_failed) {
        fprintf(stderr, "Cannot write output file: %s\n", b->path);
        return -1;
    }
    return 0;
}

static void print_root_help(const char* argv0) {
    print_release_info();
    printf("Usage: %s <command> [args]\n\n", argv0);
    printf("Commands:\n");
    printf("  run <model-dir|config.json>       Run one generic tensor graph forward pass.\n");
    printf("  generate <model-dir|config.json>  Generate text with a causal LM package.\n");
    printf("  classify <model-dir|config.json>  Run classification and print top-k scores.\n");
    printf("  detect <model-dir|config.json>    Run detection and print/write raw tensors.\n");
    printf("  ctc <model-dir|config.json>       Run CTC logits and greedy-decode text.\n");
    printf("  seq2seq <model-dir|config.json>   Run an encoder-decoder text loop.\n");
    printf("  chat <model-dir|config.json>      User-facing alias for multimodal seq2seq.\n");
#if VOLVOXAI_ENABLE_TRAINING
    printf("  train <model-dir|config.json>     Train selected model weights with cross-entropy.\n");
#endif
    printf("  version                           Print release info.\n\n");
    printf("Common backend flags: --vulkan --opengl --metal --nnapi --debug\n");
    printf("Run '%s <command> --help' for command-specific options.\n", argv0);
}

static void print_run_help(const char* argv0) {
    printf("Usage: %s run <model-dir|config.json> [options]\n\n", argv0);
    printf("Options:\n");
    printf("  --weights <file>              Override model.safetensors path.\n");
    printf("  --input <name=file>           Load exact typed raw data (.f32/.f16/.i32/.i8/.u8).\n");
    printf("  --image <name=file>           Decode PNG/JPEG into a [1,H,W,C] or [H,W,C] input.\n");
    printf("  --image-normalize <mode>      zero-one | minus-one-one | raw-255; required if package omits it.\n");
    printf("  --output <name=file|file>     Write exact typed raw data; suffix must match dtype.\n");
    printf("  --last-token <index>          Write one logits row from sequence outputs.\n");
    printf("  --vulkan | --opengl | --metal | --nnapi\n");
    printf("  --debug\n");
}

#if VOLVOXAI_ENABLE_TRAINING
static void print_train_help(const char* argv0) {
    printf("Usage: %s train <model-dir|config.json> [options]\n\n", argv0);
    printf("Required:\n");
    printf("  --targets <file.i32>          Raw int32 cross-entropy targets.\n");
    printf("  --logits <tensor>             Logits tensor used by cross-entropy.\n");
    printf("  --trainable <tensor>          F32 model weight to update. Repeatable.\n\n");
    printf("Model inputs:\n");
    printf("  --weights <file>              Override model.safetensors path.\n");
    printf("  --input <name=file>           Load exact typed raw data (.f32/.f16/.i32/.i8/.u8).\n");
    printf("  --image <name=file>           Decode PNG/JPEG into an image input.\n");
    printf("  --image-normalize <mode>      zero-one | minus-one-one | raw-255; required if package omits it.\n");
    printf("  --last-token <index>          Train one sequence position for one target.\n\n");
    printf("Optimizer:\n");
    printf("  --steps <n>                   Number of Adam/AdamW steps (default 1).\n");
    printf("  --learning-rate <value>       Learning rate (default 0.001).\n");
    printf("  --beta1 <value>               Adam beta1 (default 0.9).\n");
    printf("  --beta2 <value>               Adam beta2 (default 0.999).\n");
    printf("  --epsilon <value>             Adam epsilon (default 1e-8).\n");
    printf("  --weight-decay <value>        0 selects Adam; positive selects AdamW.\n");
    printf("  --max-grad-norm <value>       Global gradient clipping; 0 disables it.\n");
    printf("  --ignore-index <id>           Target value ignored by the loss (default -100).\n");
    printf("  --input-optimizer <file>      Resume Adam/AdamW state and its training step.\n\n");
    printf("Outputs:\n");
    printf("  --output-weights <file>       Save final model weights.\n");
    printf("  --output-optimizer <file>     Save final Adam/AdamW state.\n");
    printf("  --output <name=file|file>     Write exact typed raw data; suffix must match dtype.\n");
    printf("  --vulkan | --opengl | --metal | --nnapi\n");
    printf("  --debug\n");
}
#endif

static void print_generate_help(const char* argv0) {
    printf("Usage: %s generate <model-dir|config.json> --prompt <text> [options]\n\n", argv0);
    printf("Options:\n");
    printf("  --weights <file>              Override model.safetensors path.\n");
    printf("  --vocab <file>                Native vocab.bin path (defaults to model dir).\n");
    printf("  --merges <file>               Optional merges.txt path (defaults to model dir).\n");
    printf("  --prompt <text>               Text prompt.\n");
    printf("  --max-new <n>                 New tokens to generate (default 50).\n");
    printf("  --pad-token <id>              Token used to pad unused context slots (default 50256).\n");
    printf("  --eos-token <id>              Stop after this token id (default disabled).\n");
    printf("  --vulkan | --opengl | --metal | --nnapi\n");
    printf("  --debug\n");
}

static void print_classify_help(const char* argv0) {
    printf("Usage: %s classify <model-dir|config.json> [options]\n\n", argv0);
    printf("Options:\n");
    printf("  --weights <file>              Override model.safetensors path.\n");
    printf("  --input <name=file>           Load exact typed raw data (.f32/.f16/.i32/.i8/.u8).\n");
    printf("  --image <name=file>           Decode PNG/JPEG into a [1,H,W,C] or [H,W,C] input.\n");
    printf("  --image-normalize <mode>      zero-one | minus-one-one | raw-255; required if package omits it.\n");
    printf("  --logits <tensor>             Tensor to rank (defaults to final output).\n");
    printf("  --labels <file>               Optional one-label-per-line class names.\n");
    printf("  --top-k <n>                   Number of classes to print (default 5).\n");
    printf("  --output <name=file|file>     Write exact typed raw data; suffix must match dtype.\n");
    printf("  --vulkan | --opengl | --metal | --nnapi\n");
    printf("  --debug\n");
}

static void print_detect_help(const char* argv0) {
    printf("Usage: %s detect <model-dir|config.json> [options]\n\n", argv0);
    printf("Options:\n");
    printf("  --weights <file>              Override model.safetensors path.\n");
    printf("  --input <name=file>           Load exact typed raw data (.f32/.f16/.i32/.i8/.u8).\n");
    printf("  --image <name=file>           Decode PNG/JPEG into a [1,H,W,C] or [H,W,C] input.\n");
    printf("  --image-normalize <mode>      zero-one | minus-one-one | raw-255; required if package omits it.\n");
    printf("  --boxes <tensor>              Box tensor, usually [...,4] (default boxes).\n");
    printf("  --scores <tensor>             Score tensor, [N] or [N,C] (default scores).\n");
    printf("  --classes <tensor>            Optional class-id tensor [N].\n");
    printf("  --labels <file>               Class names (defaults to <model-dir>/labels.txt).\n");
    printf("  --max-det <n>                 Number of detections to print (default 20).\n");
    printf("  --output <name=file|file>     Write exact typed raw data; suffix must match dtype.\n");
    printf("  --num_threads <n>             CPU worker threads (default: auto).\n");
    printf("  --warmup_runs <n>             Discarded forwards before timing (default 0).\n");
    printf("  --num_runs <n>                Timed forwards; reports first/avg/min/max (default 1).\n");
    printf("  --vulkan | --opengl | --metal | --nnapi\n");
    printf("  --debug\n");
}

static void print_ctc_help(const char* argv0) {
    printf("Usage: %s ctc <model-dir|config.json> [options]\n\n", argv0);
    printf("Options:\n");
    printf("  --weights <file>              Override model.safetensors path.\n");
    printf("  --input <name=file>           Load exact typed raw data (.f32/.f16/.i32/.i8/.u8).\n");
    printf("  --image <name=file>           Decode PNG/JPEG into a [1,H,W,C] or [H,W,C] input.\n");
    printf("  --image-normalize <mode>      zero-one | minus-one-one | raw-255; required if package omits it.\n");
    printf("  --logits <tensor>             CTC logits tensor, shape [...,T,C].\n");
    printf("  --labels <file>               Optional one-token-per-line labels.\n");
    printf("  --blank <id>                  Blank token id (default 0).\n");
    printf("  --output <name=file|file>     Write exact typed raw data; suffix must match dtype.\n");
    printf("  --vulkan | --opengl | --metal | --nnapi\n");
    printf("  --debug\n");
}

static void print_seq2seq_help(const char* argv0, int is_chat) {
    printf("Usage: %s %s <model-dir|config.json> [options]\n\n", argv0, is_chat ? "chat" : "seq2seq");
    printf("Options:\n");
    printf("  --weights <file>              Override model.safetensors path.\n");
    printf("  --input <name=file>           Load exact typed raw data (.f32/.f16/.i32/.i8/.u8).\n");
    printf("  --image <name=file>           Decode PNG/JPEG into a [1,H,W,C] or [H,W,C] input.\n");
    printf("  --image-normalize <mode>      zero-one | minus-one-one | raw-255; required if package omits it.\n");
    printf("  --prompt <text>               Text prompt to tokenize into the prompt input.\n");
    printf("  --text-file <file>            Append a UTF-8 text file to the prompt.\n");
    printf("  --vocab <file>                Native vocab.bin path (defaults to model dir).\n");
    printf("  --merges <file>               Optional merges.txt path (defaults to model dir).\n");
    printf("  --prompt-input <name>         Prompt token input name (default prompt_tokens/q_tokens/tokens).\n");
    printf("  --decoder-input <name>        Decoder token input name (default decoder_tokens/y_tokens/tokens).\n");
    printf("  --logits <tensor>             Logits tensor name (defaults to final output).\n");
    printf("  --max-new <n>                 New tokens to generate (default 192).\n");
    printf("  --bos-token <id>              Decoder BOS token id (default 1).\n");
    printf("  --eos-token <id>              Stop after this token id (default 2).\n");
    printf("  --pad-token <id>              Token used to pad unused slots (default 0).\n");
    printf("  --output <name=file|file>     Write exact typed raw data; suffix must match dtype.\n");
    printf("  --vulkan | --opengl | --metal | --nnapi\n");
    printf("  --debug\n");
}

static void engine_options_init(VolvoxAIEngineOptions* options) {
    memset(options, 0, sizeof(*options));
    options->backend = VOLVOXAI_BACKEND_CPU;
}

static const char* requested_backend_name(VolvoxAIEngineBackend backend) {
    switch (backend) {
        case VOLVOXAI_BACKEND_VULKAN: return "Vulkan";
        case VOLVOXAI_BACKEND_OPENGL: return "OpenGL";
        case VOLVOXAI_BACKEND_METAL: return "Metal";
        case VOLVOXAI_BACKEND_NNAPI: return "NNAPI";
        default: return "CPU";
    }
}

static int select_backend(VolvoxAIEngineOptions* options,
                          VolvoxAIEngineBackend backend) {
    if (options->backend != VOLVOXAI_BACKEND_CPU && options->backend != backend) {
        fprintf(stderr, "Pass at most one accelerator backend flag.\n");
        return -1;
    }
    options->backend = backend;
    return 1;
}

static int parse_backend_flag(const char* arg, VolvoxAIEngineOptions* options) {
    if (!strcmp(arg, "--vulkan")) return select_backend(options, VOLVOXAI_BACKEND_VULKAN);
    if (!strcmp(arg, "--nnapi")) return select_backend(options, VOLVOXAI_BACKEND_NNAPI);
    if (!strcmp(arg, "--opengl")) return select_backend(options, VOLVOXAI_BACKEND_OPENGL);
    if (!strcmp(arg, "--metal")) return select_backend(options, VOLVOXAI_BACKEND_METAL);
    if (!strcmp(arg, "--debug")) { options->debug = 1; return 1; }
    return 0;
}

static int configure_engine(const VolvoxAIEngineOptions* options) {
    printf("VolvoxAI Native Engine\n");
    if (volvoxai_engine_configure(options) != 0) {
        fprintf(stderr, "Cannot configure requested backend: %s\n",
                requested_backend_name(options->backend));
        return -1;
    }
    printf("Backend: %s\n", volvoxai_engine_backend_name());
    return 0;
}

static int image_normalize_mode(const char* value) {
    if (!strcmp(value, "zero-one")) return VOLVOX_IMAGE_ZERO_ONE;
    if (!strcmp(value, "minus-one-one")) return VOLVOX_IMAGE_MINUS_ONE_ONE;
    if (!strcmp(value, "raw-255")) return VOLVOX_IMAGE_RAW_255;
    return -1;
}

static int resolve_package_image_normalization(const char* config_path,
                                               const char* input_name,
                                               int* out) {
    long size = 0;
    char* bytes = read_file_bytes(config_path, &size);
    if (!bytes) {
        fprintf(stderr, "Cannot read config.json image metadata: %s\n", config_path);
        return -1;
    }
    cJSON* root = cJSON_Parse(bytes);
    free(bytes);
    if (!root) {
        fprintf(stderr, "Cannot parse config.json image metadata: %s\n", config_path);
        return -1;
    }

    cJSON* inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    cJSON* input = cJSON_IsObject(inputs)
        ? cJSON_GetObjectItemCaseSensitive(inputs, input_name) : NULL;
    cJSON* normalization = cJSON_IsObject(input)
        ? cJSON_GetObjectItemCaseSensitive(input, "image_normalization") : NULL;
    if (!normalization) {
        fprintf(stderr,
                "Input %s does not declare image_normalization; add package metadata "
                "or pass --image-normalize explicitly.\n",
                input_name);
        cJSON_Delete(root);
        return -1;
    }
    if (!cJSON_IsString(normalization) || !normalization->valuestring) {
        fprintf(stderr,
                "Input %s has invalid image_normalization metadata; expected "
                "zero-one, minus-one-one, or raw-255.\n",
                input_name);
        cJSON_Delete(root);
        return -1;
    }
    int mode = image_normalize_mode(normalization->valuestring);
    if (mode < 0) {
        fprintf(stderr,
                "Input %s has unsupported image_normalization '%s'; expected "
                "zero-one, minus-one-one, or raw-255.\n",
                input_name, normalization->valuestring);
        cJSON_Delete(root);
        return -1;
    }
    *out = mode;
    cJSON_Delete(root);
    return 0;
}

static const char* cpu_arch_name(void) {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__arm__) || defined(_M_ARM)
    return "arm";
#elif defined(__riscv)
    return "riscv";
#else
    return "unknown";
#endif
}

static const char* cpu_feature_string(void) {
#if defined(__AVX2__) && defined(__FMA__)
    return "avx2,fma";
#elif defined(__AVX2__)
    return "avx2";
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    return "neon";
#else
    return "scalar";
#endif
}

static void print_release_info(void) {
    printf("VolvoxAI Native Engine %s (commit %s, built %s)\n",
           VOLVOXAI_VERSION, VOLVOXAI_GIT_COMMIT, VOLVOXAI_BUILD_DATE);
}

static void debug_fprint_shape(FILE* f, const int* shape, int ndim) {
    fprintf(f, "[");
    for (int i = 0; i < ndim; i++) fprintf(f, "%s%d", i ? "," : "", shape[i]);
    fprintf(f, "]");
}

static void debug_print_runtime(const VolvoxAIEngineOptions* options) {
    if (options->cpu_threads > 0) {
        fprintf(stderr, "[debug] cpu_arch=%s cpu_features=%s threads=%d backend=%s\n",
                cpu_arch_name(), cpu_feature_string(), options->cpu_threads,
                volvoxai_engine_backend_name());
    } else {
        fprintf(stderr, "[debug] cpu_arch=%s cpu_features=%s threads=auto backend=%s\n",
                cpu_arch_name(), cpu_feature_string(), volvoxai_engine_backend_name());
    }
}

static void debug_print_model_summary(const char* config_path) {
    long sz = 0;
    char* cfg = read_file_bytes(config_path, &sz);
    if (!cfg) return;
    cJSON* root = cJSON_Parse(cfg);
    free(cfg);
    if (!root) return;

    cJSON* inputs = cJSON_GetObjectItem(root, "inputs");
    if (inputs) {
        for (cJSON* it = inputs->child; it; it = it->next) {
            cJSON* shape = cJSON_GetObjectItem(it, "shape");
            cJSON* dtype = cJSON_GetObjectItem(it, "dtype");
            int sh[8] = {0};
            int ndim = shape ? cJSON_GetArraySize(shape) : 0;
            if (ndim > 8) ndim = 8;
            for (int i = 0; i < ndim; i++) sh[i] = cJSON_GetArrayItem(shape, i)->valueint;
            fprintf(stderr, "[debug] model_input name=%s shape=", it->string ? it->string : "<unnamed>");
            debug_fprint_shape(stderr, sh, ndim);
            fprintf(stderr, " dtype=%s\n", cJSON_IsString(dtype) ? dtype->valuestring : "float32");
        }
    }

    cJSON* outputs = cJSON_GetObjectItem(root, "outputs");
    if (outputs) {
        for (cJSON* it = outputs->child; it; it = it->next) {
            const char* tensor_name = cJSON_IsString(it) ? it->valuestring : NULL;
            long n = 0;
            int sh[8] = {0};
            int ndim = 0;
            fprintf(stderr, "[debug] model_output");
            if (it->string) fprintf(stderr, " source=%s", it->string);
            if (tensor_name) fprintf(stderr, " name=%s", tensor_name);
            if (tensor_name && volvoxai_engine_tensor_info(tensor_name, &n, sh, &ndim) == 0) {
                fprintf(stderr, " shape=");
                debug_fprint_shape(stderr, sh, ndim);
                fprintf(stderr, " numel=%ld", n);
            }
            fprintf(stderr, "\n");
        }
    }
    cJSON_Delete(root);
}

static void graph_options_init(GraphOptions* opt) {
    memset(opt, 0, sizeof(*opt));
    engine_options_init(&opt->engine);
    opt->image_norm = -1;
    opt->execution_row = -1;
}

static int parse_graph_option(int argc, char** argv, int* i, GraphOptions* opt) {
    int backend_flag = parse_backend_flag(argv[*i], &opt->engine);
    if (backend_flag != 0) return backend_flag;
    if (!strcmp(argv[*i], "--weights") && *i + 1 < argc) {
        opt->weights = argv[++(*i)];
        return 1;
    }
    if (!strcmp(argv[*i], "--input") && *i + 1 < argc) {
        if (opt->n_inputs >= MAX_BINDINGS || parse_binding(argv[++(*i)], &opt->inputs[opt->n_inputs++], 1) != 0) {
            fprintf(stderr, "--input expects name=file\n");
            return -1;
        }
        return 1;
    }
    if (!strcmp(argv[*i], "--image") && *i + 1 < argc) {
        if (opt->n_images >= MAX_BINDINGS || parse_binding(argv[++(*i)], &opt->images[opt->n_images++], 1) != 0) {
            fprintf(stderr, "--image expects name=file\n");
            return -1;
        }
        return 1;
    }
    if (!strcmp(argv[*i], "--image-normalize") && *i + 1 < argc) {
        opt->image_norm = image_normalize_mode(argv[++(*i)]);
        if (opt->image_norm < 0) {
            fprintf(stderr, "Invalid --image-normalize value.\n");
            return -1;
        }
        opt->image_norm_explicit = 1;
        return 1;
    }
    if (!strcmp(argv[*i], "--output") && *i + 1 < argc) {
        if (opt->n_outputs >= MAX_BINDINGS || parse_binding(argv[++(*i)], &opt->outputs[opt->n_outputs++], 0) != 0) {
            fprintf(stderr, "--output expects name=file or file\n");
            return -1;
        }
        return 1;
    }
    if (!strcmp(argv[*i], "--last-token") && *i + 1 < argc) {
        opt->execution_row = atoi(argv[++(*i)]);
        return 1;
    }
    return 0;
}

static int init_graph_with_options(const char* model, const GraphOptions* opt) {
    ModelPaths paths;
    if (resolve_model_paths(model, opt->weights, NULL, NULL, &paths) != 0) return -1;

    if (configure_engine(&opt->engine) != 0) return -1;
    printf("Loading graph: %s\n", paths.config);
    double t0 = now_ms();
    if (volvoxai_engine_init(paths.config, paths.weights[0] ? paths.weights : NULL) != 0) {
        fprintf(stderr, "Native init failed.\n");
        volvoxai_engine_shutdown();
        return -1;
    }
    if (volvoxai_engine_set_execution_row(opt->execution_row) != 0) {
        fprintf(stderr, "Invalid --last-token row: %d\n", opt->execution_row);
        volvoxai_engine_shutdown();
        return -1;
    }
    if (opt->engine.debug) {
        fprintf(stderr, "[debug] volvoxai_engine_init %.3f ms\n", now_ms() - t0);
        debug_print_runtime(&opt->engine);
        debug_print_model_summary(paths.config);
    }

    for (int i = 0; i < opt->n_inputs; i++) {
        if (load_tensor_binding(&opt->inputs[i]) != 0) {
            volvoxai_engine_shutdown();
            return -1;
        }
    }
    for (int i = 0; i < opt->n_images; i++) {
        int normalization = opt->image_norm;
        if (!opt->image_norm_explicit &&
            resolve_package_image_normalization(paths.config, opt->images[i].name,
                                                &normalization) != 0) {
            volvoxai_engine_shutdown();
            return -1;
        }
        if (load_image_binding(&opt->images[i], normalization) != 0) {
            volvoxai_engine_shutdown();
            return -1;
        }
    }
    return 0;
}

static int run_forward_once(const char* label, int debug) {
    int warm = g_warmup_runs < 0 ? 0 : g_warmup_runs;
    int runs = g_num_runs < 1 ? 1 : g_num_runs;
    for (int r = 0; r < warm; r++) {
        if (volvoxai_engine_forward() != 0) { fprintf(stderr, "Native inference failed.\n"); return -1; }
    }
    double first = 0, sum = 0, best = 1e30, worst = 0;
    for (int r = 0; r < runs; r++) {
        double t0 = now_ms();
        if (volvoxai_engine_forward() != 0) { fprintf(stderr, "Native inference failed.\n"); return -1; }
        double dt = now_ms() - t0;
        if (r == 0) first = dt;
        sum += dt;
        if (dt < best) best = dt;
        if (dt > worst) worst = dt;
    }
    if (warm > 0 || runs > 1) {
        // TFLite-style profile line (printed regardless of --debug, like benchmark_model)
        fprintf(stderr, "[bench] %s: first=%.3f avg=%.3f min=%.3f max=%.3f ms  (warmup=%d, runs=%d)\n",
                label, first, sum / runs, best, worst, warm, runs);
    } else if (debug) {
        fprintf(stderr, "[debug] %s forward %.3f ms\n", label, best);
    }
    return 0;
}

static const float* read_output_tensor(const char* name, long* count, int* shape, int* ndim) {
    static float* snapshot;
    static size_t snapshot_capacity;
    const float* src = NULL;
    long n = 0;
    int local_shape[8] = {0};
    int local_ndim = 0;

    const char* resolved_name = name && name[0]
        ? name : volvoxai_engine_graph_output_name(0);
    if (!resolved_name || !resolved_name[0] ||
        volvoxai_engine_tensor_info(resolved_name, &n, local_shape, &local_ndim) != 0 ||
        n <= 0 || (size_t)n > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "Unknown output tensor: %s\n",
                name && name[0] ? name : "<primary>");
        return NULL;
    }

    int row = volvoxai_engine_execution_row();
    if (row >= 0 && local_ndim >= 2) {
        int row_count = 0;
        src = volvoxai_engine_tensor_row_f32(resolved_name, row, &row_count);
        if (!src || row_count <= 0) {
            fprintf(stderr, "Cannot read row %d from output tensor: %s\n",
                    row, resolved_name);
            return NULL;
        }
        n = row_count;
        local_shape[0] = row_count;
        local_ndim = 1;
    } else {
        if (snapshot_capacity < (size_t)n) {
            float* next = (float*)realloc(snapshot, (size_t)n * sizeof(float));
            if (!next) return NULL;
            snapshot = next;
            snapshot_capacity = (size_t)n;
        }
        if (volvoxai_engine_copy_tensor_f32(resolved_name, snapshot, n) != 0) return NULL;
        src = snapshot;
    }
    if (!src || n <= 0) return NULL;
    if (count) *count = n;
    if (shape) for (int i = 0; i < local_ndim; i++) shape[i] = local_shape[i];
    if (ndim) *ndim = local_ndim;
    return src;
}

static float* copy_output_tensor(const char* name, long* count, int* shape, int* ndim) {
    const float* source = read_output_tensor(name, count, shape, ndim);
    if (!source || !count || *count <= 0 || (size_t)*count > SIZE_MAX / sizeof(float)) return NULL;
    float* copy = (float*)malloc((size_t)*count * sizeof(float));
    if (!copy) {
        fprintf(stderr, "Cannot allocate output snapshot: %s\n",
                name && name[0] ? name : "<primary>");
        return NULL;
    }
    memcpy(copy, source, (size_t)*count * sizeof(float));
    return copy;
}

static char* dup_slice(const char* s, int n) {
    char* out = (char*)malloc((size_t)n + 1);
    if (!out) return NULL;
    if (n > 0) memcpy(out, s, (size_t)n);
    out[n] = 0;
    return out;
}

static int load_labels(const char* path, LabelList* labels) {
    memset(labels, 0, sizeof(*labels));
    if (!path) return 0;
    long sz = 0;
    char* data = read_file_bytes(path, &sz);
    if (!data) {
        fprintf(stderr, "Cannot read labels file: %s\n", path);
        return -1;
    }
    int cap = 16;
    labels->items = (char**)calloc((size_t)cap, sizeof(char*));
    if (!labels->items) { free(data); return -1; }
    long start = 0;
    for (long i = 0; i <= sz; i++) {
        if (i == sz || data[i] == '\n') {
            if (i == sz && start == sz) break;
            int len = (int)(i - start);
            if (len > 0 && data[start + len - 1] == '\r') len--;
            if (labels->count >= cap) {
                cap *= 2;
                char** next = (char**)realloc(labels->items, (size_t)cap * sizeof(char*));
                if (!next) { free(data); return -1; }
                labels->items = next;
            }
            labels->items[labels->count++] = dup_slice(data + start, len);
            start = i + 1;
        }
    }
    free(data);
    return 0;
}

static void free_labels(LabelList* labels) {
    for (int i = 0; i < labels->count; i++) free(labels->items[i]);
    free(labels->items);
    labels->items = NULL;
    labels->count = 0;
}

static const char* label_at(const LabelList* labels, int idx) {
    if (!labels || idx < 0 || idx >= labels->count || !labels->items[idx]) return "";
    return labels->items[idx];
}

static int argmax_f32(const float* values, int count, float* out_value) {
    int best = 0;
    float best_v = values[0];
    for (int i = 1; i < count; i++) {
        if (values[i] > best_v) { best_v = values[i]; best = i; }
    }
    if (out_value) *out_value = best_v;
    return best;
}

static void print_topk(const float* values, int count, int top_k, const LabelList* labels) {
    if (top_k < 1) top_k = 1;
    if (top_k > count) top_k = count;
    if (top_k > 64) top_k = 64;
    int selected[64];
    for (int r = 0; r < top_k; r++) selected[r] = -1;
    for (int r = 0; r < top_k; r++) {
        int best = -1;
        float best_v = -1e30f;
        for (int i = 0; i < count; i++) {
            int used = 0;
            for (int j = 0; j < r; j++) if (selected[j] == i) { used = 1; break; }
            if (!used && values[i] > best_v) { best_v = values[i]; best = i; }
        }
        selected[r] = best;
        const char* label = label_at(labels, best);
        if (label[0]) printf("%d\t%d\t%.6g\t%s\n", r + 1, best, best_v, label);
        else printf("%d\t%d\t%.6g\n", r + 1, best, best_v);
    }
}

static void print_shape(const int* shape, int ndim) {
    printf("[");
    for (int i = 0; i < ndim; i++) printf("%s%d", i ? "," : "", shape[i]);
    printf("]");
}

static void print_tensor_summary(const char* name, const float* data, long count,
                                 const int* shape, int ndim, int max_values) {
    printf("%s shape=", name && name[0] ? name : "output");
    print_shape(shape, ndim);
    printf(" values=");
    long n = count < max_values ? count : max_values;
    for (long i = 0; i < n; i++) printf("%s%.6g", i ? "," : "", data[i]);
    if (count > n) printf(",...");
    printf("\n");
}

static const char* find_graph_input(const char* explicit_name, const char** defaults,
                                    int n_defaults) {
    if (explicit_name && explicit_name[0]) {
        return volvoxai_engine_is_graph_input(explicit_name) == 1 ? explicit_name : NULL;
    }
    for (int i = 0; i < n_defaults; i++) {
        if (volvoxai_engine_is_graph_input(defaults[i]) == 1) return defaults[i];
    }
    return NULL;
}

static int token_input_info(const char* name, long* numel, int* dtype) {
    int shape[8] = {0};
    int ndim = 0;
    size_t element_size = 0;
    if (!name || volvoxai_engine_tensor_info_ex(name, numel, shape, &ndim, dtype,
            &element_size) != 0 || volvoxai_engine_is_graph_input(name) != 1 ||
        *numel <= 0 || (size_t)*numel > SIZE_MAX / sizeof(int32_t)) {
        return -1;
    }
    if ((*dtype == VOLVOXAI_DTYPE_I32 && element_size == sizeof(int32_t)) ||
        (*dtype == VOLVOXAI_DTYPE_F32 && element_size == sizeof(float))) {
        return 0;
    }
    fprintf(stderr, "Token input %s must use I32 or F32 storage, got %s.\n",
            name, dtype_name(*dtype));
    return -1;
}

static int set_token_input(const char* name, int dtype, const int32_t* values, long numel) {
    if (!name || !values || numel <= 0 || (size_t)numel > SIZE_MAX / sizeof(int32_t)) {
        return -1;
    }
    if (dtype == VOLVOXAI_DTYPE_I32) {
        return volvoxai_engine_set_input_raw(
            name, dtype, values, (size_t)numel * sizeof(int32_t));
    }
    if (dtype == VOLVOXAI_DTYPE_F32) {
        float* converted = (float*)malloc((size_t)numel * sizeof(float));
        if (!converted) return -1;
        for (long i = 0; i < numel; i++) converted[i] = (float)values[i];
        int rc = volvoxai_engine_set_input_raw(
            name, dtype, converted, (size_t)numel * sizeof(float));
        free(converted);
        return rc;
    }
    return -1;
}

#if VOLVOXAI_ENABLE_TRAINING
typedef struct {
    GraphOptions graph;
    const char* targets_path;
    const char* logits_name;
    const char* trainable_names[MAX_BINDINGS];
    int trainable_count;
    long steps;
    int ignore_index;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    float max_grad_norm;
    const char* input_optimizer_path;
    const char* output_weights_path;
    const char* output_optimizer_path;
} TrainOptions;

static int parse_train_long(const char* text, long* out) {
    if (!text || !text[0] || !out) return -1;
    errno = 0;
    char* end = NULL;
    long value = strtol(text, &end, 10);
    if (errno == ERANGE || !end || *end != '\0') return -1;
    *out = value;
    return 0;
}

static int parse_train_float(const char* text, float* out) {
    if (!text || !text[0] || !out) return -1;
    errno = 0;
    char* end = NULL;
    float value = strtof(text, &end);
    if (errno == ERANGE || !end || *end != '\0' || !isfinite(value)) return -1;
    *out = value;
    return 0;
}

static int trainable_name_is_duplicate(const TrainOptions* opt, const char* name) {
    for (int i = 0; i < opt->trainable_count; i++) {
        if (!strcmp(opt->trainable_names[i], name)) return 1;
    }
    return 0;
}

static const char* training_backend_name(int backend) {
    switch (backend) {
        case 1: return "Vulkan";
        case 2: return "OpenGL";
        case 3: return "Metal";
        default: return "CPU";
    }
}

static int validate_training_tensors(const TrainOptions* opt, const int32_t* targets,
                                     int target_count) {
    long logits_numel = 0;
    int logits_shape[8] = {0};
    int logits_ndim = 0;
    int logits_dtype = -1;
    size_t logits_element_size = 0;
    if (volvoxai_engine_tensor_info_ex(opt->logits_name, &logits_numel, logits_shape,
            &logits_ndim, &logits_dtype, &logits_element_size) != 0 ||
        logits_dtype != 0 || logits_element_size != sizeof(float) || logits_numel <= 0 ||
        logits_ndim <= 0 || logits_ndim > 8) {
        fprintf(stderr, "--logits must name a nonempty F32 tensor: %s\n", opt->logits_name);
        return -1;
    }

    int classes = logits_shape[logits_ndim - 1];
    if (classes <= 0 || logits_numel % classes != 0 ||
        logits_numel / classes > INT_MAX) {
        fprintf(stderr, "Logits tensor has an invalid cross-entropy shape: %s\n",
                opt->logits_name);
        return -1;
    }
    int rows = (int)(logits_numel / classes);
    if (target_count <= 0 || target_count > rows) {
        fprintf(stderr, "Target count %d exceeds logits row count %d.\n", target_count, rows);
        return -1;
    }
    for (int i = 0; i < target_count; i++) {
        if (targets[i] != opt->ignore_index && (targets[i] < 0 || targets[i] >= classes)) {
            fprintf(stderr, "Target %d at index %d is outside [0, %d).\n",
                    targets[i], i, classes);
            return -1;
        }
    }

    int execution_row = volvoxai_engine_execution_row();
    if (execution_row >= 0) {
        int batch = logits_ndim >= 3 ? logits_shape[0] : 1;
        int per_batch = batch > 1 && target_count == batch && rows % batch == 0;
        int limit = target_count == 1 ? rows : (per_batch ? rows / batch : 0);
        if (limit <= 0) {
            fprintf(stderr,
                    "--last-token requires one target, or one target per logits batch.\n");
            return -1;
        }
        if (execution_row >= limit) {
            fprintf(stderr, "--last-token %d is outside [0, %d).\n",
                    execution_row, limit);
            return -1;
        }
    }

    for (int i = 0; i < opt->trainable_count; i++) {
        const char* name = opt->trainable_names[i];
        long numel = 0;
        int shape[8] = {0};
        int ndim = 0;
        int dtype = -1;
        size_t element_size = 0;
        if (volvoxai_engine_tensor_info_ex(name, &numel, shape, &ndim, &dtype,
                &element_size) != 0 || dtype != 0 || element_size != sizeof(float) ||
            numel <= 0 || volvoxai_engine_is_model_weight(name) != 1) {
            fprintf(stderr, "--trainable must name a nonempty F32 model weight: %s\n", name);
            return -1;
        }
    }
    return 0;
}

static int command_train(int argc, char** argv) {
    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_train_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }

    const char* model = argv[2];
    TrainOptions opt;
    memset(&opt, 0, sizeof(opt));
    graph_options_init(&opt.graph);
    opt.steps = 1;
    opt.ignore_index = -100;
    opt.learning_rate = 1.0e-3f;
    opt.beta1 = 0.9f;
    opt.beta2 = 0.999f;
    opt.epsilon = 1.0e-8f;
    for (int i = 3; i < argc; i++) {
        const char* arg = argv[i];
        if (!strcmp(arg, "--targets")) {
            if (i + 1 >= argc || opt.targets_path) {
                fprintf(stderr, "--targets requires exactly one file.\n");
                return 2;
            }
            opt.targets_path = argv[++i];
        } else if (!strcmp(arg, "--logits")) {
            if (i + 1 >= argc || opt.logits_name) {
                fprintf(stderr, "--logits requires exactly one tensor name.\n");
                return 2;
            }
            opt.logits_name = argv[++i];
        } else if (!strcmp(arg, "--trainable")) {
            if (i + 1 >= argc || opt.trainable_count >= MAX_BINDINGS) {
                fprintf(stderr, "--trainable requires a tensor name (maximum %d).\n",
                        MAX_BINDINGS);
                return 2;
            }
            const char* name = argv[++i];
            if (!name[0] || trainable_name_is_duplicate(&opt, name)) {
                fprintf(stderr, "Duplicate or empty --trainable tensor: %s\n", name);
                return 2;
            }
            opt.trainable_names[opt.trainable_count++] = name;
        } else if (!strcmp(arg, "--steps")) {
            if (i + 1 >= argc || parse_train_long(argv[++i], &opt.steps) != 0 ||
                opt.steps <= 0) {
                fprintf(stderr, "--steps must be a positive integer.\n");
                return 2;
            }
        } else if (!strcmp(arg, "--learning-rate")) {
            if (i + 1 >= argc || parse_train_float(argv[++i], &opt.learning_rate) != 0) {
                fprintf(stderr, "--learning-rate must be finite.\n");
                return 2;
            }
        } else if (!strcmp(arg, "--beta1")) {
            if (i + 1 >= argc || parse_train_float(argv[++i], &opt.beta1) != 0) {
                fprintf(stderr, "--beta1 must be finite.\n");
                return 2;
            }
        } else if (!strcmp(arg, "--beta2")) {
            if (i + 1 >= argc || parse_train_float(argv[++i], &opt.beta2) != 0) {
                fprintf(stderr, "--beta2 must be finite.\n");
                return 2;
            }
        } else if (!strcmp(arg, "--epsilon")) {
            if (i + 1 >= argc || parse_train_float(argv[++i], &opt.epsilon) != 0) {
                fprintf(stderr, "--epsilon must be finite.\n");
                return 2;
            }
        } else if (!strcmp(arg, "--weight-decay")) {
            if (i + 1 >= argc || parse_train_float(argv[++i], &opt.weight_decay) != 0) {
                fprintf(stderr, "--weight-decay must be finite.\n");
                return 2;
            }
        } else if (!strcmp(arg, "--max-grad-norm")) {
            if (i + 1 >= argc || parse_train_float(argv[++i], &opt.max_grad_norm) != 0) {
                fprintf(stderr, "--max-grad-norm must be finite.\n");
                return 2;
            }
        } else if (!strcmp(arg, "--ignore-index")) {
            long value = 0;
            if (i + 1 >= argc || parse_train_long(argv[++i], &value) != 0 ||
                value < INT_MIN || value > INT_MAX) {
                fprintf(stderr, "--ignore-index must be a 32-bit integer.\n");
                return 2;
            }
            opt.ignore_index = (int)value;
        } else if (!strcmp(arg, "--last-token")) {
            long value = 0;
            if (i + 1 >= argc || parse_train_long(argv[++i], &value) != 0 ||
                value < -1 || value > INT_MAX) {
                fprintf(stderr, "--last-token must be -1 or a non-negative 32-bit integer.\n");
                return 2;
            }
            opt.graph.execution_row = (int)value;
        } else if (!strcmp(arg, "--output-weights")) {
            if (i + 1 >= argc || opt.output_weights_path) {
                fprintf(stderr, "--output-weights requires exactly one file.\n");
                return 2;
            }
            opt.output_weights_path = argv[++i];
        } else if (!strcmp(arg, "--input-optimizer")) {
            if (i + 1 >= argc || opt.input_optimizer_path) {
                fprintf(stderr, "--input-optimizer requires exactly one file.\n");
                return 2;
            }
            opt.input_optimizer_path = argv[++i];
        } else if (!strcmp(arg, "--output-optimizer")) {
            if (i + 1 >= argc || opt.output_optimizer_path) {
                fprintf(stderr, "--output-optimizer requires exactly one file.\n");
                return 2;
            }
            opt.output_optimizer_path = argv[++i];
        } else {
            int parsed = parse_graph_option(argc, argv, &i, &opt.graph);
            if (parsed < 0) return 2;
            if (!parsed) {
                fprintf(stderr, "Unknown train option: %s\n", argv[i]);
                return 2;
            }
        }
    }

    if (!opt.targets_path || !opt.targets_path[0] || !opt.logits_name ||
        !opt.logits_name[0] || opt.trainable_count <= 0) {
        fprintf(stderr,
                "train requires --targets, --logits, and at least one --trainable.\n");
        return 2;
    }
    if ((opt.input_optimizer_path && !opt.input_optimizer_path[0]) ||
        (opt.output_weights_path && !opt.output_weights_path[0]) ||
        (opt.output_optimizer_path && !opt.output_optimizer_path[0])) {
        fprintf(stderr, "Optimizer and weight paths must not be empty.\n");
        return 2;
    }
    if (opt.learning_rate < 0.0f || opt.beta1 < 0.0f || opt.beta1 >= 1.0f ||
        opt.beta2 < 0.0f || opt.beta2 >= 1.0f || opt.epsilon <= 0.0f ||
        opt.weight_decay < 0.0f || opt.max_grad_norm < 0.0f) {
        fprintf(stderr,
                "Invalid optimizer values: lr/decay/clip must be non-negative, "
                "betas in [0,1), and epsilon positive.\n");
        return 2;
    }
    if (opt.output_weights_path && opt.output_optimizer_path &&
        !strcmp(opt.output_weights_path, opt.output_optimizer_path)) {
        fprintf(stderr, "Weight and optimizer outputs must use different files.\n");
        return 2;
    }
    if (opt.output_weights_path && opt.input_optimizer_path &&
        !strcmp(opt.output_weights_path, opt.input_optimizer_path)) {
        fprintf(stderr, "Weight output must not overwrite the input optimizer file.\n");
        return 2;
    }

    long target_bytes = 0;
    char* target_storage = read_file_bytes(opt.targets_path, &target_bytes);
    if (!target_storage || target_bytes <= 0 ||
        target_bytes % (long)sizeof(int32_t) != 0 ||
        target_bytes / (long)sizeof(int32_t) > INT_MAX) {
        fprintf(stderr, "--targets must be a nonempty raw int32 file: %s\n",
                opt.targets_path);
        free(target_storage);
        return 1;
    }
    int target_count = (int)(target_bytes / (long)sizeof(int32_t));
    const int32_t* targets = (const int32_t*)target_storage;

    int result = 1;
    if (init_graph_with_options(model, &opt.graph) != 0) goto done;
    if (validate_training_tensors(&opt, targets, target_count) != 0) goto done;

    long loaded_step = 0;
    if (opt.input_optimizer_path &&
        (volvoxai_engine_load_optimizer_state(opt.input_optimizer_path, &loaded_step) != 0 ||
         loaded_step < 0)) {
        fprintf(stderr, "Failed to load optimizer state: %s\n", opt.input_optimizer_path);
        goto done;
    }
    if (loaded_step > LONG_MAX - opt.steps) {
        fprintf(stderr, "Loaded optimizer step plus --steps exceeds LONG_MAX.\n");
        goto done;
    }
    long completed_step = loaded_step + opt.steps;

    printf("Training optimizer=%s steps=%ld start_step=%ld trainables=%d targets=%d\n",
           opt.weight_decay > 0.0f ? "AdamW" : "Adam", opt.steps,
           loaded_step + 1, opt.trainable_count, target_count);
    for (long index = 0; index < opt.steps; index++) {
        long step = loaded_step + index + 1;
        float loss = 0.0f;
        int correct = 0;
        int examples = 0;
        double started = now_ms();
        int update_mode = opt.weight_decay > 0.0f ? 3 : 2;
        if (volvoxai_engine_train_step(opt.logits_name, targets, target_count,
                opt.ignore_index, opt.trainable_names, opt.trainable_count, update_mode,
                opt.learning_rate, opt.beta1, opt.beta2, opt.epsilon,
                opt.weight_decay, opt.max_grad_norm, step, &loss, &correct,
                &examples) != 0 || !isfinite(loss) || correct < 0 ||
            examples < 0 || correct > examples) {
            fprintf(stderr, "Training step %ld failed.\n", step);
            goto done;
        }
        int backend = volvoxai_engine_last_training_backend();
        if (examples > 0) {
            printf("step=%ld loss=%.8g accuracy=%.2f%% (%d/%d) backend=%s time=%.3fms\n",
                   step, loss, 100.0 * (double)correct / examples, correct, examples,
                   training_backend_name(backend), now_ms() - started);
        } else {
            printf("step=%ld loss=%.8g accuracy=n/a (0/0) backend=%s time=%.3fms\n",
                   step, loss, training_backend_name(backend), now_ms() - started);
        }
    }

    for (int i = 0; i < opt.graph.n_outputs; i++) {
        if (write_output_binding(&opt.graph.outputs[i]) != 0) goto done;
    }
    if (opt.output_weights_path &&
        volvoxai_engine_save_weight_file(0, opt.output_weights_path) != 0) {
        fprintf(stderr, "Failed to save trained weights: %s\n", opt.output_weights_path);
        goto done;
    }
    if (opt.output_optimizer_path &&
        volvoxai_engine_save_optimizer_state(opt.output_optimizer_path, completed_step) != 0) {
        fprintf(stderr, "Failed to save optimizer state: %s\n",
                opt.output_optimizer_path);
        goto done;
    }
    result = 0;

done:
    volvoxai_engine_shutdown();
    free(target_storage);
    return result;
}
#endif

static int command_run(int argc, char** argv) {
    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_run_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }

    const char* model = argv[2];
    GraphOptions opt;
    graph_options_init(&opt);

    for (int i = 3; i < argc; i++) {
        int parsed = parse_graph_option(argc, argv, &i, &opt);
        if (parsed < 0) return 2;
        if (!parsed) {
            fprintf(stderr, "Unknown run option: %s\n", argv[i]);
            return 2;
        }
    }

    if (init_graph_with_options(model, &opt) != 0) return 1;
    if (run_forward_once("run", opt.engine.debug) != 0) {
        volvoxai_engine_shutdown();
        return 1;
    }

    const char* vd = getenv("VDUMP");
    if (vd) fprintf(stderr, "[debug] VDUMP is handled by volvoxai_engine_run only; use --output for subcommand run.\n");

    for (int i = 0; i < opt.n_outputs; i++) {
        if (write_output_binding(&opt.outputs[i]) != 0) { volvoxai_engine_shutdown(); return 1; }
    }
    volvoxai_engine_shutdown();
    return 0;
}

static int command_classify(int argc, char** argv) {
    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_classify_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }

    const char* model = argv[2];
    const char* logits_name = NULL;
    const char* labels_path = NULL;
    int top_k = 5;
    GraphOptions opt;
    LabelList labels;
    graph_options_init(&opt);
    memset(&labels, 0, sizeof(labels));

    for (int i = 3; i < argc; i++) {
        int parsed = parse_graph_option(argc, argv, &i, &opt);
        if (parsed < 0) return 2;
        if (parsed) continue;
        if (!strcmp(argv[i], "--logits") && i + 1 < argc) logits_name = argv[++i];
        else if (!strcmp(argv[i], "--labels") && i + 1 < argc) labels_path = argv[++i];
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) top_k = atoi(argv[++i]);
        else { fprintf(stderr, "Unknown classify option: %s\n", argv[i]); return 2; }
    }

    if (load_labels(labels_path, &labels) != 0) return 1;
    if (init_graph_with_options(model, &opt) != 0) { free_labels(&labels); return 1; }
    if (run_forward_once("classify", opt.engine.debug) != 0) {
        volvoxai_engine_shutdown();
        free_labels(&labels);
        return 1;
    }

    long count = 0;
    int shape[8] = {0};
    int ndim = 0;
    const float* logits = read_output_tensor(logits_name, &count, shape, &ndim);
    if (!logits || count <= 0) {
        fprintf(stderr, "No classification output available.\n");
        volvoxai_engine_shutdown();
        free_labels(&labels);
        return 1;
    }
    print_topk(logits, (int)count, top_k, &labels);

    for (int i = 0; i < opt.n_outputs; i++) {
        if (write_output_binding(&opt.outputs[i]) != 0) { volvoxai_engine_shutdown(); free_labels(&labels); return 1; }
    }
    volvoxai_engine_shutdown();
    free_labels(&labels);
    return 0;
}

static int detection_score_at(const float* scores, const int* score_shape, int score_ndim,
                              long score_count, int det, int* cls, float* score) {
    if (!scores || score_count <= 0) return -1;
    if (score_ndim >= 2) {
        int c = score_shape[score_ndim - 1];
        long base = (long)det * c;
        if (base + c > score_count) return -1;
        float best_v = 0.0f;
        int best = argmax_f32(scores + base, c, &best_v);
        if (cls) *cls = best;
        if (score) *score = best_v;
        return 0;
    }
    if (det >= score_count) return -1;
    if (cls) *cls = -1;
    if (score) *score = scores[det];
    return 0;
}

static void print_detections(const float* boxes, long box_count, const int* box_shape, int box_ndim,
                             const float* scores, long score_count, const int* score_shape, int score_ndim,
                             const float* classes, long class_count, const LabelList* labels, int max_det) {
    if (!boxes || box_count < 4 || box_ndim < 1 || box_shape[box_ndim - 1] != 4 || !scores) return;
    int dets = (int)(box_count / 4);
    if (score_ndim >= 2) {
        int c = score_shape[score_ndim - 1];
        int score_dets = (int)(score_count / c);
        if (score_dets < dets) dets = score_dets;
    } else if (score_count < dets) {
        dets = (int)score_count;
    }
    if (max_det < 1) max_det = 1;
    if (max_det > dets) max_det = dets;
    if (max_det > 256) max_det = 256;

    int selected[256];
    for (int i = 0; i < max_det; i++) selected[i] = -1;
    int print_labels = labels && labels->count > 0;
    if (print_labels) {
        printf("rank\tindex\tscore\tscore_pct\tclass\tlabel\tx0\ty0\tx1\ty1\n");
    } else {
        printf("rank\tindex\tscore\tscore_pct\tclass\tx0\ty0\tx1\ty1\n");
    }
    for (int r = 0; r < max_det; r++) {
        int best_det = -1;
        int best_cls = -1;
        float best_score = -1e30f;
        for (int d = 0; d < dets; d++) {
            int used = 0;
            for (int j = 0; j < r; j++) if (selected[j] == d) { used = 1; break; }
            if (used) continue;
            int cls = -1;
            float score = 0.0f;
            if (detection_score_at(scores, score_shape, score_ndim, score_count, d, &cls, &score) != 0) continue;
            if (score > best_score) { best_score = score; best_det = d; best_cls = cls; }
        }
        if (best_det < 0) break;
        selected[r] = best_det;
        if (classes && best_det < class_count) best_cls = (int)classes[best_det];
        const float* b = boxes + (long)best_det * 4;
        if (print_labels) {
            const char* label = label_at(labels, best_cls);
            printf("%d\t%d\t%.6g\t%.2f%%\t%d\t%s\t%.6g\t%.6g\t%.6g\t%.6g\n",
                   r + 1, best_det, best_score, best_score * 100.0f,
                   best_cls, label[0] ? label : "-",
                   b[0], b[1], b[2], b[3]);
        } else {
            printf("%d\t%d\t%.6g\t%.2f%%\t%d\t%.6g\t%.6g\t%.6g\t%.6g\n",
                   r + 1, best_det, best_score, best_score * 100.0f,
                   best_cls, b[0], b[1], b[2], b[3]);
        }
    }
}

static int command_detect(int argc, char** argv) {
    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_detect_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }

    const char* model = argv[2];
    const char* boxes_name = "boxes";
    const char* scores_name = "scores";
    const char* classes_name = NULL;
    const char* labels_path = NULL;
    char discovered_labels[PATH_MAX] = {0};
    int max_det = 20;
    GraphOptions opt;
    LabelList labels;
    graph_options_init(&opt);
    memset(&labels, 0, sizeof(labels));

    for (int i = 3; i < argc; i++) {
        int parsed = parse_graph_option(argc, argv, &i, &opt);
        if (parsed < 0) return 2;
        if (parsed) continue;
        if (!strcmp(argv[i], "--boxes") && i + 1 < argc) boxes_name = argv[++i];
        else if (!strcmp(argv[i], "--scores") && i + 1 < argc) scores_name = argv[++i];
        else if (!strcmp(argv[i], "--classes") && i + 1 < argc) classes_name = argv[++i];
        else if (!strcmp(argv[i], "--labels") && i + 1 < argc) labels_path = argv[++i];
        else if (!strcmp(argv[i], "--max-det") && i + 1 < argc) max_det = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "--num_threads") || !strcmp(argv[i], "--num-threads")) && i + 1 < argc) {
            opt.engine.cpu_threads = atoi(argv[++i]);
            if (opt.engine.cpu_threads <= 0) {
                fprintf(stderr, "--num_threads must be a positive integer.\n");
                return 2;
            }
        }
        else if ((!strcmp(argv[i], "--warmup_runs") || !strcmp(argv[i], "--warmup-runs")) && i + 1 < argc)
            g_warmup_runs = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "--num_runs") || !strcmp(argv[i], "--num-runs")) && i + 1 < argc)
            g_num_runs = atoi(argv[++i]);
        else { fprintf(stderr, "Unknown detect option: %s\n", argv[i]); return 2; }
    }

    labels_path = resolve_labels_path(model, labels_path, discovered_labels, sizeof(discovered_labels));
    if (load_labels(labels_path, &labels) != 0) return 1;
    if (init_graph_with_options(model, &opt) != 0) { free_labels(&labels); return 1; }
    if (run_forward_once("detect", opt.engine.debug) != 0) {
        volvoxai_engine_shutdown();
        free_labels(&labels);
        return 1;
    }

    long box_count = 0, score_count = 0, class_count = 0, fallback_count = 0;
    int box_shape[8] = {0}, score_shape[8] = {0}, class_shape[8] = {0}, fallback_shape[8] = {0};
    int box_ndim = 0, score_ndim = 0, class_ndim = 0, fallback_ndim = 0;
    float* boxes = boxes_name ? copy_output_tensor(boxes_name, &box_count, box_shape, &box_ndim) : NULL;
    float* scores = scores_name ? copy_output_tensor(scores_name, &score_count, score_shape, &score_ndim) : NULL;
    float* classes = classes_name ? copy_output_tensor(classes_name, &class_count, class_shape, &class_ndim) : NULL;
    if ((boxes_name && !boxes) || (scores_name && !scores) || (classes_name && !classes)) {
        free(boxes);
        free(scores);
        free(classes);
        volvoxai_engine_shutdown();
        free_labels(&labels);
        return 1;
    }

    if (boxes && scores) {
        print_detections(boxes, box_count, box_shape, box_ndim, scores, score_count, score_shape, score_ndim,
                         classes, class_count, &labels, max_det);
    } else {
        if (boxes) print_tensor_summary(boxes_name, boxes, box_count, box_shape, box_ndim, 12);
        if (scores) print_tensor_summary(scores_name, scores, score_count, score_shape, score_ndim, 12);
        if (classes) print_tensor_summary(classes_name, classes, class_count, class_shape, class_ndim, 12);
        if (!boxes && !scores && !classes) {
            const float* out = read_output_tensor(NULL, &fallback_count, fallback_shape, &fallback_ndim);
            if (out) print_tensor_summary("output", out, fallback_count, fallback_shape, fallback_ndim, 12);
        }
    }

    for (int i = 0; i < opt.n_outputs; i++) {
        if (write_output_binding(&opt.outputs[i]) != 0) {
            free(boxes);
            free(scores);
            free(classes);
            volvoxai_engine_shutdown();
            free_labels(&labels);
            return 1;
        }
    }
    free(boxes);
    free(scores);
    free(classes);
    volvoxai_engine_shutdown();
    free_labels(&labels);
    return 0;
}

static int command_ctc(int argc, char** argv) {
    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_ctc_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }

    const char* model = argv[2];
    const char* logits_name = NULL;
    const char* labels_path = NULL;
    int blank = 0;
    GraphOptions opt;
    LabelList labels;
    graph_options_init(&opt);
    memset(&labels, 0, sizeof(labels));

    for (int i = 3; i < argc; i++) {
        int parsed = parse_graph_option(argc, argv, &i, &opt);
        if (parsed < 0) return 2;
        if (parsed) continue;
        if (!strcmp(argv[i], "--logits") && i + 1 < argc) logits_name = argv[++i];
        else if (!strcmp(argv[i], "--labels") && i + 1 < argc) labels_path = argv[++i];
        else if (!strcmp(argv[i], "--blank") && i + 1 < argc) blank = atoi(argv[++i]);
        else { fprintf(stderr, "Unknown ctc option: %s\n", argv[i]); return 2; }
    }

    if (load_labels(labels_path, &labels) != 0) return 1;
    if (init_graph_with_options(model, &opt) != 0) { free_labels(&labels); return 1; }
    if (run_forward_once("ctc", opt.engine.debug) != 0) {
        volvoxai_engine_shutdown();
        free_labels(&labels);
        return 1;
    }

    long count = 0;
    int shape[8] = {0};
    int ndim = 0;
    const float* logits = read_output_tensor(logits_name, &count, shape, &ndim);
    if (!logits || ndim < 2) {
        fprintf(stderr, "CTC requires a logits tensor with shape [...,T,C].\n");
        volvoxai_engine_shutdown();
        free_labels(&labels);
        return 1;
    }
    int t_steps = shape[ndim - 2];
    int classes = shape[ndim - 1];
    int prev = -1;
    int printed = 0;
    for (int t = 0; t < t_steps; t++) {
        float best_v = 0.0f;
        int id = argmax_f32(logits + (long)t * classes, classes, &best_v);
        if (id != blank && id != prev) {
            const char* label = label_at(&labels, id);
            if (label[0]) printf("%s", label);
            else printf("%s%d", printed ? " " : "", id);
            printed = 1;
        }
        prev = id;
    }
    printf("\n");

    for (int i = 0; i < opt.n_outputs; i++) {
        if (write_output_binding(&opt.outputs[i]) != 0) { volvoxai_engine_shutdown(); free_labels(&labels); return 1; }
    }
    volvoxai_engine_shutdown();
    free_labels(&labels);
    return 0;
}

static char* build_prompt_text(const char* prompt, const char* text_file) {
    if (!text_file) return prompt ? dup_slice(prompt, (int)strlen(prompt)) : NULL;
    long sz = 0;
    char* text = read_file_bytes(text_file, &sz);
    if (!text) {
        fprintf(stderr, "Cannot read text file: %s\n", text_file);
        return NULL;
    }
    if (!prompt || !prompt[0]) return text;
    size_t lp = strlen(prompt);
    size_t lt = strlen(text);
    char* out = (char*)malloc(lp + lt + 3);
    if (!out) { free(text); return NULL; }
    memcpy(out, prompt, lp);
    out[lp] = '\n';
    out[lp + 1] = '\n';
    memcpy(out + lp + 2, text, lt + 1);
    free(text);
    return out;
}

static int fill_prompt_input(VolvoxAITokenizer* tok, const char* prompt, const char* explicit_name,
                             int pad_token, int eos_token, int debug) {
    if (!prompt || !prompt[0]) return 0;
    if (!tok) {
        fprintf(stderr, "--prompt/--text-file requires --vocab or a model-dir vocab.bin for now.\n");
        return -1;
    }
    const char* defaults[] = {"prompt_tokens", "q_tokens", "question_tokens", "tokens"};
    const char* found = find_graph_input(explicit_name, defaults, 4);
    long n = 0;
    int dtype = -1;
    if (!found || token_input_info(found, &n, &dtype) != 0) {
        fprintf(stderr, "No prompt input found. Use --prompt-input <name> or --input <name=file>.\n");
        return -1;
    }
    int32_t* dst = (int32_t*)malloc((size_t)n * sizeof(int32_t));
    if (!dst) return -1;
    for (long i = 0; i < n; i++) dst[i] = pad_token;
    int cap = n < 4096 ? (int)n : 4096;
    int tokens[4096];
    double t0 = now_ms();
    int num = volvoxai_tokenizer_encode(tok, prompt, tokens, cap);
    if (num < 0) num = 0;
    for (int i = 0; i < num && i < cap; i++) dst[i] = tokens[i];
    if (eos_token >= 0 && num < cap) dst[num++] = eos_token;
    if (set_token_input(found, dtype, dst, n) != 0) {
        fprintf(stderr, "Cannot set prompt input: %s\n", found);
        free(dst);
        return -1;
    }
    free(dst);
    if (debug) {
        fprintf(stderr, "[debug] prompt_input=%s tokens=%d %.3f ms\n", found, num, now_ms() - t0);
    }
    return 0;
}

static int command_seq2seq_like(int argc, char** argv, int is_chat) {
    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_seq2seq_help(argv[0], is_chat);
        return argc < 3 ? 1 : 0;
    }

    const char* model = argv[2];
    const char* vocab = NULL;
    const char* merges = NULL;
    const char* prompt = NULL;
    const char* text_file = NULL;
    const char* prompt_input = NULL;
    const char* decoder_input = NULL;
    const char* logits_name = NULL;
    int max_new = 192;
    int bos_token = 1;
    int eos_token = 2;
    int pad_token = 0;
    GraphOptions opt;
    graph_options_init(&opt);

    for (int i = 3; i < argc; i++) {
        int parsed = parse_graph_option(argc, argv, &i, &opt);
        if (parsed < 0) return 2;
        if (parsed) continue;
        if (!strcmp(argv[i], "--vocab") && i + 1 < argc) vocab = argv[++i];
        else if (!strcmp(argv[i], "--merges") && i + 1 < argc) merges = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--text-file") && i + 1 < argc) text_file = argv[++i];
        else if (!strcmp(argv[i], "--prompt-input") && i + 1 < argc) prompt_input = argv[++i];
        else if (!strcmp(argv[i], "--decoder-input") && i + 1 < argc) decoder_input = argv[++i];
        else if (!strcmp(argv[i], "--logits") && i + 1 < argc) logits_name = argv[++i];
        else if (!strcmp(argv[i], "--max-new") && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bos-token") && i + 1 < argc) bos_token = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--eos-token") && i + 1 < argc) eos_token = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pad-token") && i + 1 < argc) pad_token = atoi(argv[++i]);
        else {
            fprintf(stderr, "Unknown %s option: %s\n", is_chat ? "chat" : "seq2seq", argv[i]);
            return 2;
        }
    }
    if (max_new < 1) max_new = 1;

    ModelPaths token_paths;
    if (resolve_model_paths(model, opt.weights, vocab, merges, &token_paths) != 0) return 1;
    const char* vocab_path = file_exists(token_paths.vocab) ? token_paths.vocab : NULL;
    const char* merges_path = file_exists(token_paths.merges) ? token_paths.merges : NULL;
    char* prompt_text = build_prompt_text(prompt, text_file);
    if (text_file && !prompt_text) return 1;

    VolvoxAITokenizer* tok = NULL;
    if (vocab_path) {
        double t0 = now_ms();
        tok = volvoxai_tokenizer_init(vocab_path, merges_path);
        if (!tok) { free(prompt_text); fprintf(stderr, "Tokenizer init failed.\n"); return 1; }
        if (opt.engine.debug) {
            fprintf(stderr, "[debug] tokenizer_init %.3f ms\n", now_ms() - t0);
        }
    }

    if (init_graph_with_options(model, &opt) != 0) {
        volvoxai_tokenizer_free(tok);
        free(prompt_text);
        return 1;
    }
    if (fill_prompt_input(tok, prompt_text, prompt_input, pad_token, eos_token,
                          opt.engine.debug) != 0) {
        volvoxai_engine_shutdown();
        volvoxai_tokenizer_free(tok);
        free(prompt_text);
        return 1;
    }

    const char* decoder_defaults[] = {"decoder_tokens", "y_tokens", "output_tokens", "tokens"};
    const char* decoder_found = find_graph_input(decoder_input, decoder_defaults, 4);
    long dec_n = 0;
    int dec_dtype = -1;
    if (!decoder_found || token_input_info(decoder_found, &dec_n, &dec_dtype) != 0) {
        fprintf(stderr, "No decoder input found. Use --decoder-input <name> or bind it with --input.\n");
        volvoxai_engine_shutdown();
        volvoxai_tokenizer_free(tok);
        free(prompt_text);
        return 1;
    }
    int32_t* dec = (int32_t*)malloc((size_t)dec_n * sizeof(int32_t));
    if (!dec) {
        volvoxai_engine_shutdown();
        volvoxai_tokenizer_free(tok);
        free(prompt_text);
        return 1;
    }
    for (long i = 0; i < dec_n; i++) dec[i] = pad_token;
    dec[0] = bos_token;
    if (set_token_input(decoder_found, dec_dtype, dec, dec_n) != 0) {
        fprintf(stderr, "Cannot set decoder input: %s\n", decoder_found);
        free(dec);
        volvoxai_engine_shutdown();
        volvoxai_tokenizer_free(tok);
        free(prompt_text);
        return 1;
    }
    int dec_cap = dec_n < 4096 ? (int)dec_n : 4096;
    if (max_new > dec_cap - 1) max_new = dec_cap - 1;

    if (opt.engine.debug) {
        fprintf(stderr, "[debug] decoder_input=%s max_new=%d bos=%d eos=%d pad=%d\n",
                decoder_found, max_new, bos_token, eos_token, pad_token);
    }

    int generated = 0;
    double gen_t0 = now_ms();
    for (int step = 0; step < max_new; step++) {
        if (volvoxai_engine_set_execution_row(step) != 0) {
            fprintf(stderr, "Cannot select decoder row %d.\n", step);
            free(dec);
            volvoxai_engine_shutdown();
            volvoxai_tokenizer_free(tok);
            free(prompt_text);
            return 1;
        }
        if (run_forward_once(is_chat ? "chat" : "seq2seq", opt.engine.debug) != 0) {
            free(dec);
            volvoxai_engine_shutdown();
            volvoxai_tokenizer_free(tok);
            free(prompt_text);
            return 1;
        }
        long count = 0;
        int shape[8] = {0};
        int ndim = 0;
        const float* logits = read_output_tensor(logits_name, &count, shape, &ndim);
        if (!logits || count <= 0) {
            fprintf(stderr, "No logits output available.\n");
            free(dec);
            volvoxai_engine_shutdown();
            volvoxai_tokenizer_free(tok);
            free(prompt_text);
            return 1;
        }
        float best_v = 0.0f;
        int best_id = argmax_f32(logits, (int)count, &best_v);
        if (opt.engine.debug) {
            fprintf(stderr, "[debug] decode step=%d token=%d logit=%.5f\n", step, best_id, best_v);
        }
        if (best_id == eos_token) break;
        if (tok) printf("%s", volvoxai_tokenizer_decode(tok, best_id));
        else printf("%s%d", generated ? " " : "", best_id);
        fflush(stdout);
        generated++;
        if (step + 1 < dec_cap) {
            dec[step + 1] = best_id;
            if (set_token_input(decoder_found, dec_dtype, dec, dec_n) != 0) {
                fprintf(stderr, "Cannot update decoder input: %s\n", decoder_found);
                free(dec);
                volvoxai_engine_shutdown();
                volvoxai_tokenizer_free(tok);
                free(prompt_text);
                return 1;
            }
        }
    }
    if (opt.engine.debug) {
        double gen_ms = now_ms() - gen_t0;
        double tps = generated > 0 && gen_ms > 0.0 ? (double)generated * 1000.0 / gen_ms : 0.0;
        fprintf(stderr, "[debug] %s tokens=%d total=%.3f ms tok/s=%.2f\n",
                is_chat ? "chat" : "seq2seq", generated, gen_ms, tps);
    }
    printf("\n");

    for (int i = 0; i < opt.n_outputs; i++) {
        if (write_output_binding(&opt.outputs[i]) != 0) {
            free(dec);
            volvoxai_engine_shutdown();
            volvoxai_tokenizer_free(tok);
            free(prompt_text);
            return 1;
        }
    }
    free(dec);
    volvoxai_engine_shutdown();
    volvoxai_tokenizer_free(tok);
    free(prompt_text);
    return 0;
}

static int command_generate(int argc, char** argv) {
    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_generate_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }

    const char* model = argv[2];
    const char* weights = NULL;
    const char* vocab = NULL;
    const char* merges = NULL;
    const char* prompt = NULL;
    int max_new = 50;
    int pad_token = 50256;
    int eos_token = -1;
    VolvoxAIEngineOptions engine;
    engine_options_init(&engine);

    for (int i = 3; i < argc; i++) {
        int backend_flag = parse_backend_flag(argv[i], &engine);
        if (backend_flag < 0) return 2;
        if (backend_flag > 0) continue;
        if (!strcmp(argv[i], "--weights") && i + 1 < argc) weights = argv[++i];
        else if (!strcmp(argv[i], "--vocab") && i + 1 < argc) vocab = argv[++i];
        else if (!strcmp(argv[i], "--merges") && i + 1 < argc) merges = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--max-new") && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pad-token") && i + 1 < argc) pad_token = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--eos-token") && i + 1 < argc) eos_token = atoi(argv[++i]);
        else { fprintf(stderr, "Unknown generate option: %s\n", argv[i]); return 2; }
    }
    if (!prompt) { fprintf(stderr, "generate requires --prompt.\n"); return 2; }
    if (max_new < 1) max_new = 1;

    ModelPaths paths;
    if (resolve_model_paths(model, weights, vocab, merges, &paths) != 0) return 1;
    if (!file_exists(paths.weights)) { fprintf(stderr, "Missing weights: %s\n", paths.weights); return 1; }
    if (!file_exists(paths.vocab)) { fprintf(stderr, "Missing vocab: %s\n", paths.vocab); return 1; }
    const char* merges_path = file_exists(paths.merges) ? paths.merges : NULL;

    if (configure_engine(&engine) != 0) return 1;
    printf("Loading graph: %s\n", paths.config);

    double t0 = now_ms();
    VolvoxAITokenizer* tok = volvoxai_tokenizer_init(paths.vocab, merges_path);
    if (!tok) {
        volvoxai_engine_shutdown();
        fprintf(stderr, "Tokenizer init failed.\n");
        return 1;
    }
    if (engine.debug) fprintf(stderr, "[debug] tokenizer_init %.3f ms\n", now_ms() - t0);

    t0 = now_ms();
    if (volvoxai_engine_init(paths.config, paths.weights) != 0) {
        volvoxai_engine_shutdown();
        volvoxai_tokenizer_free(tok);
        fprintf(stderr, "Native init failed.\n");
        return 1;
    }
    if (engine.debug) {
        fprintf(stderr, "[debug] volvoxai_engine_init %.3f ms\n", now_ms() - t0);
        debug_print_runtime(&engine);
    }

    const char* output_name = volvoxai_engine_graph_output_name(0);
    if (!output_name || !output_name[0]) {
        volvoxai_engine_shutdown();
        volvoxai_tokenizer_free(tok);
        fprintf(stderr, "Model has no primary graph output.\n");
        return 1;
    }

    const char* token_defaults[] = {"tokens", "input_ids"};
    const char* token_name = find_graph_input(NULL, token_defaults, 2);
    long n = 0;
    int token_dtype = -1;
    if (!token_name || token_input_info(token_name, &n, &token_dtype) != 0) {
        volvoxai_engine_shutdown();
        volvoxai_tokenizer_free(tok);
        fprintf(stderr, "Model has no I32/F32 token input named 'tokens' or 'input_ids'.\n");
        return 1;
    }
    int32_t* tok_in = (int32_t*)malloc((size_t)n * sizeof(int32_t));
    if (!tok_in) {
        volvoxai_engine_shutdown();
        volvoxai_tokenizer_free(tok);
        return 1;
    }
    int cap = n < 4096 ? (int)n : 4096;
    int tokens[4096];
    t0 = now_ms();
    int num_tokens = volvoxai_tokenizer_encode(tok, prompt, tokens, cap);
    if (num_tokens <= 0) {
        tokens[0] = pad_token;
        num_tokens = 1;
    }
    if (num_tokens > cap) num_tokens = cap;
    if (engine.debug) {
        fprintf(stderr, "[debug] tokenizer_encode tokens=%d %.3f ms\n",
                num_tokens, now_ms() - t0);
    }
    for (long i = 0; i < n; i++) tok_in[i] = pad_token;
    for (int i = 0; i < num_tokens; i++) tok_in[i] = tokens[i];
    if (set_token_input(token_name, token_dtype, tok_in, n) != 0) {
        free(tok_in);
        volvoxai_engine_shutdown();
        volvoxai_tokenizer_free(tok);
        fprintf(stderr, "Cannot set token input: %s\n", token_name);
        return 1;
    }

    const char* position_defaults[] = {"positions", "position_ids"};
    const char* position_name = find_graph_input(NULL, position_defaults, 2);
    if (position_name) {
        long position_n = 0;
        int position_dtype = -1;
        if (token_input_info(position_name, &position_n, &position_dtype) != 0) {
            free(tok_in);
            volvoxai_engine_shutdown();
            volvoxai_tokenizer_free(tok);
            return 1;
        }
        int32_t* positions = (int32_t*)malloc((size_t)position_n * sizeof(int32_t));
        if (!positions) {
            free(tok_in);
            volvoxai_engine_shutdown();
            volvoxai_tokenizer_free(tok);
            return 1;
        }
        for (long i = 0; i < position_n; i++) positions[i] = (int32_t)i;
        int position_rc = set_token_input(position_name, position_dtype, positions, position_n);
        free(positions);
        if (position_rc != 0) {
            free(tok_in);
            volvoxai_engine_shutdown();
            volvoxai_tokenizer_free(tok);
            fprintf(stderr, "Cannot set position input: %s\n", position_name);
            return 1;
        }
    }

    printf("Encoded prompt into %d tokens.\n\n%s", num_tokens, prompt);
    fflush(stdout);

    if (engine.debug) volvoxai_engine_profile_reset();
    t0 = now_ms();
    if (volvoxai_engine_forward_prefix(num_tokens) != 0) {
        free(tok_in);
        volvoxai_engine_shutdown();
        volvoxai_tokenizer_free(tok);
        fprintf(stderr, "\nNative prefill failed.\n");
        return 1;
    }
    if (engine.debug) {
        fprintf(stderr, "[debug] prefill tokens=%d %.3f ms\n",
                num_tokens, now_ms() - t0);
    }

    int pos = num_tokens - 1;
    int generated = 0;
    int result = 0;
    double gen_t0 = now_ms();
    for (int step = 0; step < max_new && pos < cap - 1; step++) {
        int vocab_count = 0;
        const float* logits = volvoxai_engine_tensor_row_f32(
            output_name, pos, &vocab_count);
        if (!logits || vocab_count <= 0) {
            fprintf(stderr, "\nCannot read logits row %d from %s.\n", pos, output_name);
            result = 1;
            break;
        }
        int best_id = 0;
        float best_val = -1e30f;
        for (int i = 0; i < vocab_count; i++) {
            if (logits[i] > best_val) { best_val = logits[i]; best_id = i; }
        }
        printf("%s", volvoxai_tokenizer_decode(tok, best_id));
        fflush(stdout);
        generated++;
        if (best_id == eos_token) break;
        pos++;
        tok_in[pos] = best_id;
        if (set_token_input(token_name, token_dtype, tok_in, n) != 0) {
            fprintf(stderr, "\nCannot update token input: %s\n", token_name);
            result = 1;
            break;
        }
        double dec_t0 = now_ms();
        if (volvoxai_engine_forward_row(pos) != 0) {
            fprintf(stderr, "\nNative decode failed.\n");
            result = 1;
            break;
        }
        if (engine.debug) {
            fprintf(stderr, "[debug] decode step=%d pos=%d token=%d logit=%.5f %.3f ms\n",
                    step, pos, best_id, best_val, now_ms() - dec_t0);
        }
    }
    if (engine.debug) {
        double gen_ms = now_ms() - gen_t0;
        double tps = generated > 0 && gen_ms > 0.0 ? (double)generated * 1000.0 / gen_ms : 0.0;
        fprintf(stderr, "[debug] generate tokens=%d total=%.3f ms tok/s=%.2f\n", generated, gen_ms, tps);
        volvoxai_engine_profile_report();
    }
    printf("\n");

    free(tok_in);
    volvoxai_engine_shutdown();
    volvoxai_tokenizer_free(tok);
    return result;
}

int main(int argc, char** argv) {
    if (argc < 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
        print_root_help(argv[0]);
        return argc < 2 ? 1 : 0;
    }
    if (!strcmp(argv[1], "--version") || !strcmp(argv[1], "-V") ||
        !strcmp(argv[1], "version")) {
        print_release_info();
        return 0;
    }

    int rc = 0;
    if (!strcmp(argv[1], "run")) rc = command_run(argc, argv);
    else if (!strcmp(argv[1], "generate")) rc = command_generate(argc, argv);
    else if (!strcmp(argv[1], "classify")) rc = command_classify(argc, argv);
    else if (!strcmp(argv[1], "detect")) rc = command_detect(argc, argv);
    else if (!strcmp(argv[1], "ctc")) rc = command_ctc(argc, argv);
    else if (!strcmp(argv[1], "seq2seq")) rc = command_seq2seq_like(argc, argv, 0);
    else if (!strcmp(argv[1], "chat")) rc = command_seq2seq_like(argc, argv, 1);
#if VOLVOXAI_ENABLE_TRAINING
    else if (!strcmp(argv[1], "train")) rc = command_train(argc, argv);
#endif
    else {
        fprintf(stderr, "Unknown command: %s\n\n", argv[1]);
        print_root_help(argv[0]);
        rc = 2;
    }

    return rc;
}
