#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "engine.h"
#include "cJSON.h"
#include "image_io.h"
#include "kie_runtime.h"
#include "nnapi_engine.h"
#include "opengl_engine.h"
#include "safetensors.h"
#include "tensor.h"
#include "tokenizer.h"
#include "vulkan_engine.h"
#ifdef __APPLE__
#include "metal_engine.h"
#endif

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

// Provide the heap base for the WASM kernel's bump allocator when compiled natively.
unsigned char __heap_base[1024 * 1024 * 64]; // 64 MB heap

int g_use_vulkan = 0;
int g_use_nnapi = 0;
int g_use_opengl = 0;
int g_use_metal = 0;
int g_last_token = -1;
int g_debug = 0;
int g_warmup_runs = 0;   // --warmup_runs: discarded forwards before timing
int g_num_runs = 1;      // --num_runs: timed forwards (reports first/avg/min/max)

extern int vx_kernels_thread_count(void);
extern void vx_set_num_threads(int n);

#define MAX_BINDINGS 32

typedef struct {
    char name[128];
    char path[PATH_MAX];
} Binding;

typedef struct {
    int use_opengl;
    int use_metal;
} BackendOptions;

typedef struct {
    char config[PATH_MAX];
    char weights[PATH_MAX];
    char vocab[PATH_MAX];
    char merges[PATH_MAX];
} ModelPaths;

typedef struct {
    const char* weights;
    BackendOptions backend;
    Binding inputs[MAX_BINDINGS];
    Binding images[MAX_BINDINGS];
    Binding outputs[MAX_BINDINGS];
    int n_inputs;
    int n_images;
    int n_outputs;
    int image_norm;
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

static char* read_file_bytes(const char* path, long* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
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
    long n = 0;
    float* dst = engine_input_ptr(b->name, &n);
    if (!dst) {
        fprintf(stderr, "Unknown input tensor: %s\n", b->name);
        return -1;
    }
    long sz = 0;
    char* data = read_file_bytes(b->path, &sz);
    if (!data) {
        fprintf(stderr, "Cannot read input file: %s\n", b->path);
        return -1;
    }
    long count = sz / 4;
    if (count > n) count = n;
    if (has_suffix(b->path, ".i32")) {
        const int32_t* src = (const int32_t*)data;
        for (long i = 0; i < count; i++) dst[i] = (float)src[i];
    } else {
        memcpy(dst, data, (size_t)count * sizeof(float));
    }
    free(data);
    return 0;
}

static int load_image_binding(const Binding* b, int normalize) {
    long n = 0;
    int shape[8] = {0};
    int ndim = 0;
    float* dst = engine_input_ptr(b->name, &n);
    if (!dst || engine_tensor_info(b->name, &n, shape, &ndim) != 0) {
        fprintf(stderr, "Unknown image input tensor: %s\n", b->name);
        return -1;
    }
    char err[256] = {0};
    if (volvox_load_image_to_tensor(b->path, dst, shape, ndim, normalize, err, sizeof(err)) != 0) {
        fprintf(stderr, "Image load failed for %s: %s\n", b->path, err[0] ? err : "unknown error");
        return -1;
    }
    return 0;
}

static int write_output_binding(const Binding* b) {
    const float* src = NULL;
    int count = 0;
    int shape[8] = {0};
    int ndim = 0;
    long n = 0;

    if (b->name[0]) {
        src = engine_tensor_ptr(b->name, &n);
        if (!src || engine_tensor_info(b->name, &n, shape, &ndim) != 0) {
            fprintf(stderr, "Unknown output tensor: %s\n", b->name);
            return -1;
        }
        if (g_last_token >= 0 && ndim >= 2) {
            int d = shape[ndim - 1];
            src += (long)g_last_token * d;
            count = d;
        } else {
            count = (int)n;
        }
    } else {
        src = engine_last_logits(&count);
    }
    if (!src || count <= 0) {
        fprintf(stderr, "No output data available for %s\n", b->name[0] ? b->name : "<last>");
        return -1;
    }
    FILE* f = fopen(b->path, "wb");
    if (!f) {
        fprintf(stderr, "Cannot open output file: %s\n", b->path);
        return -1;
    }
    fwrite(src, sizeof(float), (size_t)count, f);
    fclose(f);
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
    printf("  version                           Print release info.\n\n");
    printf("Common backend flags: --vulkan --opengl --metal --nnapi --debug\n");
    printf("Run '%s <command> --help' for command-specific options.\n", argv0);
}

static void print_run_help(const char* argv0) {
    printf("Usage: %s run <model-dir|config.json> [options]\n\n", argv0);
    printf("Options:\n");
    printf("  --weights <file>              Override model.safetensors path.\n");
    printf("  --input <name=file>           Load raw tensor input (.f32 or .i32). Repeatable.\n");
    printf("  --image <name=file>           Decode PNG/JPEG into a [1,H,W,C] or [H,W,C] input.\n");
    printf("  --image-normalize <mode>      zero-one (default), minus-one-one, or raw-255.\n");
    printf("  --output <name=file|file>     Write an output tensor. Repeatable.\n");
    printf("  --last-token <index>          Write one logits row from sequence outputs.\n");
    printf("  --vulkan | --opengl | --metal | --nnapi\n");
    printf("  --debug\n");
}

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
    printf("  --input <name=file>           Load raw tensor input (.f32 or .i32). Repeatable.\n");
    printf("  --image <name=file>           Decode PNG/JPEG into a [1,H,W,C] or [H,W,C] input.\n");
    printf("  --image-normalize <mode>      zero-one (default), minus-one-one, or raw-255.\n");
    printf("  --logits <tensor>             Tensor to rank (defaults to final output).\n");
    printf("  --labels <file>               Optional one-label-per-line class names.\n");
    printf("  --top-k <n>                   Number of classes to print (default 5).\n");
    printf("  --output <name=file|file>     Also write raw output tensor. Repeatable.\n");
    printf("  --vulkan | --opengl | --metal | --nnapi\n");
    printf("  --debug\n");
}

static void print_detect_help(const char* argv0) {
    printf("Usage: %s detect <model-dir|config.json> [options]\n\n", argv0);
    printf("Options:\n");
    printf("  --weights <file>              Override model.safetensors path.\n");
    printf("  --input <name=file>           Load raw tensor input (.f32 or .i32). Repeatable.\n");
    printf("  --image <name=file>           Decode PNG/JPEG into a [1,H,W,C] or [H,W,C] input.\n");
    printf("  --image-normalize <mode>      zero-one (default), minus-one-one, or raw-255.\n");
    printf("  --boxes <tensor>              Box tensor to summarize, usually [...,4].\n");
    printf("  --scores <tensor>             Score tensor to rank, either [N] or [N,C].\n");
    printf("  --classes <tensor>            Optional class-id tensor [N].\n");
    printf("  --max-det <n>                 Number of detections to print (default 20).\n");
    printf("  --output <name=file|file>     Write raw tensor output. Repeatable.\n");
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
    printf("  --input <name=file>           Load raw tensor input (.f32 or .i32). Repeatable.\n");
    printf("  --image <name=file>           Decode PNG/JPEG into a [1,H,W,C] or [H,W,C] input.\n");
    printf("  --image-normalize <mode>      zero-one (default), minus-one-one, or raw-255.\n");
    printf("  --logits <tensor>             CTC logits tensor, shape [...,T,C].\n");
    printf("  --labels <file>               Optional one-token-per-line labels.\n");
    printf("  --blank <id>                  Blank token id (default 0).\n");
    printf("  --output <name=file|file>     Also write raw output tensor. Repeatable.\n");
    printf("  --vulkan | --opengl | --metal | --nnapi\n");
    printf("  --debug\n");
}

static void print_seq2seq_help(const char* argv0, int is_chat) {
    printf("Usage: %s %s <model-dir|config.json> [options]\n\n", argv0, is_chat ? "chat" : "seq2seq");
    printf("Options:\n");
    printf("  --weights <file>              Override model.safetensors path.\n");
    printf("  --input <name=file>           Load raw tensor input (.f32 or .i32). Repeatable.\n");
    printf("  --image <name=file>           Decode PNG/JPEG into a [1,H,W,C] or [H,W,C] input.\n");
    printf("  --image-normalize <mode>      zero-one (default), minus-one-one, or raw-255.\n");
    printf("  --prompt <text>               Text prompt to tokenize into the prompt input.\n");
    printf("  --text-file <file>            Append a UTF-8 text file to the prompt.\n");
    printf("  --family <name|auto>          Adapter family override; auto uses the model router.\n");
    printf("  --vocab <file>                Native vocab.bin path (defaults to model dir).\n");
    printf("  --merges <file>               Optional merges.txt path (defaults to model dir).\n");
    printf("  --prompt-input <name>         Prompt token input name (default prompt_tokens/q_tokens/tokens).\n");
    printf("  --decoder-input <name>        Decoder token input name (default decoder_tokens/y_tokens/tokens).\n");
    printf("  --logits <tensor>             Logits tensor name (defaults to final output).\n");
    printf("  --max-new <n>                 New tokens to generate (default 192).\n");
    printf("  --bos-token <id>              Decoder BOS token id (default 1).\n");
    printf("  --eos-token <id>              Stop after this token id (default 2).\n");
    printf("  --pad-token <id>              Token used to pad unused slots (default 0).\n");
    printf("  --output <name=file|file>     Also write raw output tensor. Repeatable.\n");
    printf("  --vulkan | --opengl | --metal | --nnapi\n");
    printf("  --debug\n");
}

static int parse_backend_flag(const char* arg, BackendOptions* backend) {
    if (!strcmp(arg, "--vulkan")) { g_use_vulkan = 1; return 1; }
    if (!strcmp(arg, "--nnapi")) { g_use_nnapi = 1; return 1; }
    if (!strcmp(arg, "--opengl")) { backend->use_opengl = 1; g_use_opengl = 1; return 1; }
    if (!strcmp(arg, "--metal")) { backend->use_metal = 1; g_use_metal = 1; return 1; }
    if (!strcmp(arg, "--debug")) { g_debug = 1; return 1; }
    return 0;
}

static void init_backend(const BackendOptions* backend) {
    printf("VolvoxAI Native Engine\n");
    if (g_use_nnapi) {
        if (nnapi_init() == 0) printf("Backend: NNAPI\n");
        else { printf("Backend: CPU (NNAPI unavailable)\n"); g_use_nnapi = 0; }
    } else if (g_use_vulkan) {
        if (vk_init() == 0) printf("Backend: Vulkan\n");
        else { printf("Backend: CPU (Vulkan unavailable)\n"); g_use_vulkan = 0; }
    } else if (backend->use_metal) {
#ifdef __APPLE__
        if (metal_init() == 0) printf("Backend: Metal\n");
        else { printf("Backend: CPU (Metal unavailable)\n"); g_use_metal = 0; }
#else
        printf("Backend: CPU (Metal is macOS-only)\n");
        g_use_metal = 0;
#endif
    } else if (backend->use_opengl) {
        if (opengl_init() == 0) printf("Backend: OpenGL\n");
        else { printf("Backend: CPU (OpenGL unavailable)\n"); g_use_opengl = 0; }
    } else {
        printf("Backend: CPU\n");
    }
}

static int image_normalize_mode(const char* value) {
    if (!strcmp(value, "zero-one")) return VOLVOX_IMAGE_ZERO_ONE;
    if (!strcmp(value, "minus-one-one")) return VOLVOX_IMAGE_MINUS_ONE_ONE;
    if (!strcmp(value, "raw-255")) return VOLVOX_IMAGE_RAW_255;
    return -1;
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

static void debug_print_runtime(void) {
    fprintf(stderr, "[debug] cpu_arch=%s cpu_features=%s threads=%d\n",
            cpu_arch_name(), cpu_feature_string(), vx_kernels_thread_count());
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
            if (tensor_name && engine_tensor_info(tensor_name, &n, sh, &ndim) == 0) {
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
    opt->image_norm = VOLVOX_IMAGE_ZERO_ONE;
}

static int parse_graph_option(int argc, char** argv, int* i, GraphOptions* opt) {
    if (parse_backend_flag(argv[*i], &opt->backend)) return 1;
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
        g_last_token = atoi(argv[++(*i)]);
        return 1;
    }
    return 0;
}

static int init_graph_with_options(const char* model, const GraphOptions* opt) {
    ModelPaths paths;
    if (resolve_model_paths(model, opt->weights, NULL, NULL, &paths) != 0) return -1;
    if (!file_exists(paths.weights)) {
        fprintf(stderr, "Missing weights: %s\n", paths.weights);
        return -1;
    }

    init_backend(&opt->backend);
    printf("Loading graph: %s\n", paths.config);
    double t0 = now_ms();
    if (engine_init(paths.config, paths.weights) != 0) {
        fprintf(stderr, "Native init failed.\n");
        return -1;
    }
    if (g_debug) fprintf(stderr, "[debug] engine_init %.3f ms\n", now_ms() - t0);
    if (g_debug) {
        debug_print_runtime();
        debug_print_model_summary(paths.config);
    }

    for (int i = 0; i < opt->n_inputs; i++) {
        if (load_tensor_binding(&opt->inputs[i]) != 0) {
            engine_free_ctx();
            return -1;
        }
    }
    for (int i = 0; i < opt->n_images; i++) {
        if (load_image_binding(&opt->images[i], opt->image_norm) != 0) {
            engine_free_ctx();
            return -1;
        }
    }
    return 0;
}

static int run_forward_once(const char* label) {
    int warm = g_warmup_runs < 0 ? 0 : g_warmup_runs;
    int runs = g_num_runs < 1 ? 1 : g_num_runs;
    for (int r = 0; r < warm; r++) {
        if (engine_forward() != 0) { fprintf(stderr, "Native inference failed.\n"); return -1; }
    }
    double first = 0, sum = 0, best = 1e30, worst = 0;
    for (int r = 0; r < runs; r++) {
        double t0 = now_ms();
        if (engine_forward() != 0) { fprintf(stderr, "Native inference failed.\n"); return -1; }
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
    } else if (g_debug) {
        fprintf(stderr, "[debug] %s forward %.3f ms\n", label, best);
    }
    return 0;
}

static const float* read_output_tensor(const char* name, long* count, int* shape, int* ndim) {
    const float* src = NULL;
    long n = 0;
    int local_shape[8] = {0};
    int local_ndim = 0;

    if (name && name[0]) {
        src = engine_tensor_ptr(name, &n);
        if (!src || engine_tensor_info(name, &n, local_shape, &local_ndim) != 0) {
            fprintf(stderr, "Unknown output tensor: %s\n", name);
            return NULL;
        }
        if (g_last_token >= 0 && local_ndim >= 2) {
            int d = local_shape[local_ndim - 1];
            src += (long)g_last_token * d;
            n = d;
            local_shape[0] = d;
            local_ndim = 1;
        }
    } else {
        int c = 0;
        src = engine_last_logits(&c);
        n = c;
        local_shape[0] = c;
        local_ndim = 1;
    }
    if (!src || n <= 0) return NULL;
    if (count) *count = n;
    if (shape) for (int i = 0; i < local_ndim; i++) shape[i] = local_shape[i];
    if (ndim) *ndim = local_ndim;
    return src;
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

static float* find_first_tensor(const char** names, int n_names, const char** found, long* numel) {
    for (int i = 0; i < n_names; i++) {
        float* p = engine_input_ptr(names[i], numel);
        if (p) {
            if (found) *found = names[i];
            return p;
        }
    }
    return NULL;
}

static int command_run(int argc, char** argv) {
    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_run_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }

    const char* model = argv[2];
    GraphOptions opt;
    graph_options_init(&opt);
    g_last_token = -1;

    for (int i = 3; i < argc; i++) {
        int parsed = parse_graph_option(argc, argv, &i, &opt);
        if (parsed < 0) return 2;
        if (!parsed) {
            fprintf(stderr, "Unknown run option: %s\n", argv[i]);
            return 2;
        }
    }

    if (init_graph_with_options(model, &opt) != 0) return 1;
    if (run_forward_once("run") != 0) { engine_free_ctx(); return 1; }

    const char* vd = getenv("VDUMP");
    if (vd) fprintf(stderr, "[debug] VDUMP is handled by engine_run only; use --output for subcommand run.\n");

    for (int i = 0; i < opt.n_outputs; i++) {
        if (write_output_binding(&opt.outputs[i]) != 0) { engine_free_ctx(); return 1; }
    }
    engine_free_ctx();
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
    g_last_token = -1;

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
    if (run_forward_once("classify") != 0) { engine_free_ctx(); free_labels(&labels); return 1; }

    long count = 0;
    int shape[8] = {0};
    int ndim = 0;
    const float* logits = read_output_tensor(logits_name, &count, shape, &ndim);
    if (!logits || count <= 0) {
        fprintf(stderr, "No classification output available.\n");
        engine_free_ctx();
        free_labels(&labels);
        return 1;
    }
    print_topk(logits, (int)count, top_k, &labels);

    for (int i = 0; i < opt.n_outputs; i++) {
        if (write_output_binding(&opt.outputs[i]) != 0) { engine_free_ctx(); free_labels(&labels); return 1; }
    }
    engine_free_ctx();
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
                             const float* classes, long class_count, int max_det) {
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
    printf("rank\tindex\tscore\tclass\tx0\ty0\tx1\ty1\n");
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
        printf("%d\t%d\t%.6g\t%d\t%.6g\t%.6g\t%.6g\t%.6g\n",
               r + 1, best_det, best_score, best_cls, b[0], b[1], b[2], b[3]);
    }
}

static int command_detect(int argc, char** argv) {
    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_detect_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }

    const char* model = argv[2];
    const char* boxes_name = NULL;
    const char* scores_name = NULL;
    const char* classes_name = NULL;
    int max_det = 20;
    GraphOptions opt;
    graph_options_init(&opt);
    g_last_token = -1;

    for (int i = 3; i < argc; i++) {
        int parsed = parse_graph_option(argc, argv, &i, &opt);
        if (parsed < 0) return 2;
        if (parsed) continue;
        if (!strcmp(argv[i], "--boxes") && i + 1 < argc) boxes_name = argv[++i];
        else if (!strcmp(argv[i], "--scores") && i + 1 < argc) scores_name = argv[++i];
        else if (!strcmp(argv[i], "--classes") && i + 1 < argc) classes_name = argv[++i];
        else if (!strcmp(argv[i], "--max-det") && i + 1 < argc) max_det = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "--num_threads") || !strcmp(argv[i], "--num-threads")) && i + 1 < argc)
            vx_set_num_threads(atoi(argv[++i]));
        else if ((!strcmp(argv[i], "--warmup_runs") || !strcmp(argv[i], "--warmup-runs")) && i + 1 < argc)
            g_warmup_runs = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "--num_runs") || !strcmp(argv[i], "--num-runs")) && i + 1 < argc)
            g_num_runs = atoi(argv[++i]);
        else { fprintf(stderr, "Unknown detect option: %s\n", argv[i]); return 2; }
    }

    if (init_graph_with_options(model, &opt) != 0) return 1;
    if (run_forward_once("detect") != 0) { engine_free_ctx(); return 1; }

    long box_count = 0, score_count = 0, class_count = 0, fallback_count = 0;
    int box_shape[8] = {0}, score_shape[8] = {0}, class_shape[8] = {0}, fallback_shape[8] = {0};
    int box_ndim = 0, score_ndim = 0, class_ndim = 0, fallback_ndim = 0;
    const float* boxes = boxes_name ? read_output_tensor(boxes_name, &box_count, box_shape, &box_ndim) : NULL;
    const float* scores = scores_name ? read_output_tensor(scores_name, &score_count, score_shape, &score_ndim) : NULL;
    const float* classes = classes_name ? read_output_tensor(classes_name, &class_count, class_shape, &class_ndim) : NULL;
    if ((boxes_name && !boxes) || (scores_name && !scores) || (classes_name && !classes)) {
        engine_free_ctx();
        return 1;
    }

    if (boxes && scores) {
        print_detections(boxes, box_count, box_shape, box_ndim, scores, score_count, score_shape, score_ndim,
                         classes, class_count, max_det);
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
        if (write_output_binding(&opt.outputs[i]) != 0) { engine_free_ctx(); return 1; }
    }
    engine_free_ctx();
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
    g_last_token = -1;

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
    if (run_forward_once("ctc") != 0) { engine_free_ctx(); free_labels(&labels); return 1; }

    long count = 0;
    int shape[8] = {0};
    int ndim = 0;
    const float* logits = read_output_tensor(logits_name, &count, shape, &ndim);
    if (!logits || ndim < 2) {
        fprintf(stderr, "CTC requires a logits tensor with shape [...,T,C].\n");
        engine_free_ctx();
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
        if (write_output_binding(&opt.outputs[i]) != 0) { engine_free_ctx(); free_labels(&labels); return 1; }
    }
    engine_free_ctx();
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

static float* find_named_or_default_input(const char* explicit_name, const char** defaults,
                                          int n_defaults, const char** found, long* numel) {
    if (explicit_name && explicit_name[0]) {
        float* p = engine_input_ptr(explicit_name, numel);
        if (p && found) *found = explicit_name;
        return p;
    }
    return find_first_tensor(defaults, n_defaults, found, numel);
}

static int fill_prompt_input(Tokenizer* tok, const char* prompt, const char* explicit_name,
                             int pad_token, int eos_token) {
    if (!prompt || !prompt[0]) return 0;
    if (!tok) {
        fprintf(stderr, "--prompt/--text-file requires --vocab or a model-dir vocab.bin for now.\n");
        return -1;
    }
    const char* defaults[] = {"prompt_tokens", "q_tokens", "question_tokens", "tokens"};
    const char* found = NULL;
    long n = 0;
    float* dst = find_named_or_default_input(explicit_name, defaults, 4, &found, &n);
    if (!dst || n <= 0) {
        fprintf(stderr, "No prompt input found. Use --prompt-input <name> or --input <name=file>.\n");
        return -1;
    }
    for (long i = 0; i < n; i++) dst[i] = (float)pad_token;
    int cap = n < 4096 ? (int)n : 4096;
    int tokens[4096];
    double t0 = now_ms();
    int num = tokenizer_encode(tok, prompt, tokens, cap);
    if (num < 0) num = 0;
    for (int i = 0; i < num && i < cap; i++) dst[i] = (float)tokens[i];
    if (eos_token >= 0 && num < cap) dst[num++] = (float)eos_token;
    if (g_debug) {
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
    const char* family = NULL;
    const char* prompt_input = NULL;
    const char* decoder_input = NULL;
    const char* logits_name = NULL;
    int max_new = 192;
    int bos_token = 1;
    int eos_token = 2;
    int pad_token = 0;
    GraphOptions opt;
    graph_options_init(&opt);
    g_last_token = -1;

    for (int i = 3; i < argc; i++) {
        int parsed = parse_graph_option(argc, argv, &i, &opt);
        if (parsed < 0) return 2;
        if (parsed) continue;
        if (!strcmp(argv[i], "--vocab") && i + 1 < argc) vocab = argv[++i];
        else if (!strcmp(argv[i], "--merges") && i + 1 < argc) merges = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--text-file") && i + 1 < argc) text_file = argv[++i];
        else if (!strcmp(argv[i], "--family") && i + 1 < argc) family = argv[++i];
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

    if (is_chat && kie_config_is_tiny_receipt(token_paths.config)) {
        if (!file_exists(token_paths.weights)) {
            fprintf(stderr, "Missing weights: %s\n", token_paths.weights);
            free(prompt_text);
            return 1;
        }
        if (opt.n_images < 1) {
            fprintf(stderr, "KIE chat requires --image image=<png|jpg>.\n");
            free(prompt_text);
            return 2;
        }
        if (!prompt_text) prompt_text = dup_slice("", 0);
        int rc = kie_chat(token_paths.config, token_paths.weights, opt.images[0].path,
                          prompt_text, family, max_new, g_debug);
        free(prompt_text);
        return rc;
    }

    Tokenizer* tok = NULL;
    if (vocab_path) {
        double t0 = now_ms();
        tok = tokenizer_init(vocab_path, merges_path);
        if (!tok) { free(prompt_text); fprintf(stderr, "Tokenizer init failed.\n"); return 1; }
        if (g_debug) fprintf(stderr, "[debug] tokenizer_init %.3f ms\n", now_ms() - t0);
    }

    if (init_graph_with_options(model, &opt) != 0) {
        tokenizer_free(tok);
        free(prompt_text);
        return 1;
    }
    if (fill_prompt_input(tok, prompt_text, prompt_input, pad_token, eos_token) != 0) {
        engine_free_ctx();
        tokenizer_free(tok);
        free(prompt_text);
        return 1;
    }

    const char* decoder_defaults[] = {"decoder_tokens", "y_tokens", "output_tokens", "tokens"};
    const char* decoder_found = NULL;
    long dec_n = 0;
    float* dec = find_named_or_default_input(decoder_input, decoder_defaults, 4, &decoder_found, &dec_n);
    if (!dec || dec_n <= 0) {
        fprintf(stderr, "No decoder input found. Use --decoder-input <name> or bind it with --input.\n");
        engine_free_ctx();
        tokenizer_free(tok);
        free(prompt_text);
        return 1;
    }
    for (long i = 0; i < dec_n; i++) dec[i] = (float)pad_token;
    dec[0] = (float)bos_token;
    int dec_cap = dec_n < 4096 ? (int)dec_n : 4096;
    if (max_new > dec_cap - 1) max_new = dec_cap - 1;

    if (g_debug) {
        fprintf(stderr, "[debug] decoder_input=%s max_new=%d bos=%d eos=%d pad=%d\n",
                decoder_found, max_new, bos_token, eos_token, pad_token);
    }

    int generated = 0;
    double gen_t0 = now_ms();
    for (int step = 0; step < max_new; step++) {
        g_last_token = step;
        if (run_forward_once(is_chat ? "chat" : "seq2seq") != 0) {
            engine_free_ctx();
            tokenizer_free(tok);
            free(prompt_text);
            return 1;
        }
        long count = 0;
        int shape[8] = {0};
        int ndim = 0;
        const float* logits = read_output_tensor(logits_name, &count, shape, &ndim);
        if (!logits || count <= 0) {
            fprintf(stderr, "No logits output available.\n");
            engine_free_ctx();
            tokenizer_free(tok);
            free(prompt_text);
            return 1;
        }
        float best_v = 0.0f;
        int best_id = argmax_f32(logits, (int)count, &best_v);
        if (g_debug) {
            fprintf(stderr, "[debug] decode step=%d token=%d logit=%.5f\n", step, best_id, best_v);
        }
        if (best_id == eos_token) break;
        if (tok) printf("%s", tokenizer_decode(tok, best_id));
        else printf("%s%d", generated ? " " : "", best_id);
        fflush(stdout);
        generated++;
        if (step + 1 < dec_cap) dec[step + 1] = (float)best_id;
    }
    if (g_debug) {
        double gen_ms = now_ms() - gen_t0;
        double tps = generated > 0 && gen_ms > 0.0 ? (double)generated * 1000.0 / gen_ms : 0.0;
        fprintf(stderr, "[debug] %s tokens=%d total=%.3f ms tok/s=%.2f\n",
                is_chat ? "chat" : "seq2seq", generated, gen_ms, tps);
    }
    printf("\n");

    for (int i = 0; i < opt.n_outputs; i++) {
        if (write_output_binding(&opt.outputs[i]) != 0) {
            engine_free_ctx();
            tokenizer_free(tok);
            free(prompt_text);
            return 1;
        }
    }
    engine_free_ctx();
    tokenizer_free(tok);
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
    BackendOptions backend = {0};

    for (int i = 3; i < argc; i++) {
        if (parse_backend_flag(argv[i], &backend)) continue;
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

    init_backend(&backend);
    printf("Loading graph: %s\n", paths.config);

    double t0 = now_ms();
    Tokenizer* tok = tokenizer_init(paths.vocab, merges_path);
    if (!tok) { fprintf(stderr, "Tokenizer init failed.\n"); return 1; }
    if (g_debug) fprintf(stderr, "[debug] tokenizer_init %.3f ms\n", now_ms() - t0);

    t0 = now_ms();
    if (engine_init(paths.config, paths.weights) != 0) {
        tokenizer_free(tok);
        fprintf(stderr, "Native init failed.\n");
        return 1;
    }
    if (g_debug) fprintf(stderr, "[debug] engine_init %.3f ms\n", now_ms() - t0);

    long n = 0;
    float* tok_in = engine_input_ptr("tokens", &n);
    if (!tok_in || n <= 0) {
        engine_free_ctx();
        tokenizer_free(tok);
        fprintf(stderr, "Model has no 'tokens' input.\n");
        return 1;
    }
    int cap = n < 4096 ? (int)n : 4096;
    int tokens[4096];
    t0 = now_ms();
    int num_tokens = tokenizer_encode(tok, prompt, tokens, cap);
    if (num_tokens <= 0) num_tokens = 1;
    if (g_debug) fprintf(stderr, "[debug] tokenizer_encode tokens=%d %.3f ms\n", num_tokens, now_ms() - t0);
    for (int i = 0; i < cap; i++) tok_in[i] = (i < num_tokens) ? (float)tokens[i] : (float)pad_token;

    printf("Encoded prompt into %d tokens.\n\n%s", num_tokens, prompt);
    fflush(stdout);

    if (g_debug) engine_profile_reset();
    t0 = now_ms();
    if (engine_prefill(num_tokens) != 0) {
        engine_free_ctx();
        tokenizer_free(tok);
        fprintf(stderr, "\nNative prefill failed.\n");
        return 1;
    }
    if (g_debug) fprintf(stderr, "[debug] prefill tokens=%d %.3f ms\n", num_tokens, now_ms() - t0);

    int pos = num_tokens - 1;
    int generated = 0;
    double gen_t0 = now_ms();
    for (int step = 0; step < max_new && pos < cap - 1; step++) {
        int vocab_count = 0;
        const float* logits = engine_last_logits(&vocab_count);
        int best_id = 0;
        float best_val = -1e30f;
        for (int i = 0; i < vocab_count; i++) {
            if (logits[i] > best_val) { best_val = logits[i]; best_id = i; }
        }
        printf("%s", tokenizer_decode(tok, best_id));
        fflush(stdout);
        generated++;
        if (best_id == eos_token) break;
        pos++;
        tok_in[pos] = (float)best_id;
        double dec_t0 = now_ms();
        if (engine_decode(pos) != 0) { fprintf(stderr, "\nNative decode failed.\n"); break; }
        if (g_debug) {
            fprintf(stderr, "[debug] decode step=%d pos=%d token=%d logit=%.5f %.3f ms\n",
                    step, pos, best_id, best_val, now_ms() - dec_t0);
        }
    }
    if (g_debug) {
        double gen_ms = now_ms() - gen_t0;
        double tps = generated > 0 && gen_ms > 0.0 ? (double)generated * 1000.0 / gen_ms : 0.0;
        fprintf(stderr, "[debug] generate tokens=%d total=%.3f ms tok/s=%.2f\n", generated, gen_ms, tps);
        engine_profile_report();
    }
    printf("\n");

    engine_free_ctx();
    tokenizer_free(tok);
    return 0;
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
    else {
        fprintf(stderr, "Unknown command: %s\n\n", argv[1]);
        print_root_help(argv[0]);
        rc = 2;
    }

    vk_cleanup();
    nnapi_cleanup();
    opengl_cleanup();
#ifdef __APPLE__
    metal_cleanup();
#endif
    return rc;
}
