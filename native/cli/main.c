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

static int g_warmup_runs = 0;
static int g_num_runs = 1;

#define MAX_BINDINGS 32

typedef struct {
    char name[128];
    char path[PATH_MAX];
} Binding;

typedef struct {
    char config[PATH_MAX];
    char weights[PATH_MAX];
} ModelPaths;

typedef struct {
    const char* weights;
    VolvoxAIEngineOptions engine;
    Binding inputs[MAX_BINDINGS];
    Binding outputs[MAX_BINDINGS];
    int n_inputs;
    int n_outputs;
    int execution_row;
} GraphOptions;

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

static int resolve_model_paths(const char* model, const char* weights,
                               ModelPaths* out) {
    memset(out, 0, sizeof(*out));
    if (is_dir(model)) {
        join_path(out->config, sizeof(out->config), model, "config.json");
        join_path(out->weights, sizeof(out->weights), model, "model.safetensors");
        if (!file_exists(out->weights)) out->weights[0] = '\0';
    } else {
        snprintf(out->config, sizeof(out->config), "%s", model);
    }
    if (weights) snprintf(out->weights, sizeof(out->weights), "%s", weights);
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

static int parse_row_index(const char* text, int* out) {
    if (!text || !text[0] || !out) return -1;
    errno = 0;
    char* end = NULL;
    long value = strtol(text, &end, 10);
    if (errno == ERANGE || !end || *end != '\0' || value < -1 || value > INT_MAX) {
        return -1;
    }
    *out = (int)value;
    return 0;
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
            fprintf(stderr, "--row output currently requires an F32 tensor: %s\n", name);
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
#if VOLVOXAI_ENABLE_TRAINING
    printf("  train <model-dir|config.json>     Train selected weights with cross-entropy.\n");
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
    printf("  --output <name=file|file>     Write exact typed raw data; suffix must match dtype.\n");
    printf("  --row <index>                 Write one F32 row from rank-2-or-higher outputs.\n");
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
    printf("  --row <index>                 Train one selected logits row.\n\n");
    printf("Optimizer:\n");
    printf("  --steps <n>                   Number of SGD/AdamW steps (default 1).\n");
    printf("  --learning-rate <value>       Learning rate (default 0.001).\n");
    printf("  --beta1 <value>               Adam beta1 (default 0.9).\n");
    printf("  --beta2 <value>               Adam beta2 (default 0.999).\n");
    printf("  --epsilon <value>             Adam epsilon (default 1e-8).\n");
    printf("  --weight-decay <value>        0 selects SGD; positive selects AdamW.\n");
    printf("  --max-grad-norm <value>       Global gradient clipping; 0 disables it.\n");
    printf("  --ignore-index <id>           Target value ignored by the loss (default -100).\n");
    printf("  --input-optimizer <file>      Resume optimizer state and its training step.\n\n");
    printf("Outputs:\n");
    printf("  --output-weights <file>       Save final model weights.\n");
    printf("  --output-optimizer <file>     Save final optimizer state.\n");
    printf("  --output <name=file|file>     Write final exact typed raw tensor data.\n");
    printf("  --vulkan | --opengl | --metal | --nnapi\n");
    printf("  --debug\n");
}
#endif

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

static void debug_print_tensor(const char* kind, const char* name) {
    long numel = 0;
    int shape[8] = {0};
    int ndim = 0;
    int dtype = -1;
    size_t element_size = 0;
    fprintf(stderr, "[debug] model_%s name=%s", kind, name ? name : "<unnamed>");
    if (name && volvoxai_engine_tensor_info_ex(name, &numel, shape, &ndim, &dtype,
            &element_size) == 0) {
        fprintf(stderr, " shape=");
        debug_fprint_shape(stderr, shape, ndim);
        fprintf(stderr, " dtype=%s element_size=%zu numel=%ld",
                dtype_name(dtype), element_size, numel);
    }
    fprintf(stderr, "\n");
}

static void debug_print_model_summary(void) {
    int input_count = volvoxai_engine_graph_input_count();
    for (int i = 0; i < input_count; i++) {
        debug_print_tensor("input", volvoxai_engine_graph_input_name(i));
    }
    int output_count = volvoxai_engine_graph_output_count();
    for (int i = 0; i < output_count; i++) {
        debug_print_tensor("output", volvoxai_engine_graph_output_name(i));
    }
}

static void graph_options_init(GraphOptions* opt) {
    memset(opt, 0, sizeof(*opt));
    engine_options_init(&opt->engine);
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
    if (!strcmp(argv[*i], "--output") && *i + 1 < argc) {
        if (opt->n_outputs >= MAX_BINDINGS || parse_binding(argv[++(*i)], &opt->outputs[opt->n_outputs++], 0) != 0) {
            fprintf(stderr, "--output expects name=file or file\n");
            return -1;
        }
        return 1;
    }
    if (!strcmp(argv[*i], "--row")) {
        if (*i + 1 >= argc || parse_row_index(argv[++(*i)], &opt->execution_row) != 0) {
            fprintf(stderr, "--row must be -1 or a non-negative 32-bit integer.\n");
            return -1;
        }
        return 1;
    }
    return 0;
}

static int init_graph_with_options(const char* model, const GraphOptions* opt) {
    ModelPaths paths;
    if (resolve_model_paths(model, opt->weights, &paths) != 0) return -1;

    if (configure_engine(&opt->engine) != 0) return -1;
    printf("Loading graph: %s\n", paths.config);
    double t0 = now_ms();
    if (volvoxai_engine_init(paths.config, paths.weights[0] ? paths.weights : NULL) != 0) {
        fprintf(stderr, "Native init failed.\n");
        volvoxai_engine_shutdown();
        return -1;
    }
    if (volvoxai_engine_set_execution_row(opt->execution_row) != 0) {
        fprintf(stderr, "Invalid --row index: %d\n", opt->execution_row);
        volvoxai_engine_shutdown();
        return -1;
    }
    if (opt->engine.debug) {
        fprintf(stderr, "[debug] volvoxai_engine_init %.3f ms\n", now_ms() - t0);
        debug_print_runtime(&opt->engine);
        debug_print_model_summary();
    }

    for (int i = 0; i < opt->n_inputs; i++) {
        if (load_tensor_binding(&opt->inputs[i]) != 0) {
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
        case VOLVOXAI_TRAINING_BACKEND_VULKAN: return "Vulkan";
        case VOLVOXAI_TRAINING_BACKEND_OPENGL: return "OpenGL";
        case VOLVOXAI_TRAINING_BACKEND_METAL: return "Metal";
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
        logits_dtype != VOLVOXAI_DTYPE_F32 || logits_element_size != sizeof(float) ||
        logits_numel <= 0 ||
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
                    "--row requires one target, or one target per logits batch.\n");
            return -1;
        }
        if (execution_row >= limit) {
            fprintf(stderr, "--row %d is outside [0, %d).\n",
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
                &element_size) != 0 || dtype != VOLVOXAI_DTYPE_F32 ||
            element_size != sizeof(float) ||
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
        } else if (!strcmp(arg, "--row")) {
            if (i + 1 >= argc || parse_row_index(argv[++i],
                    &opt.graph.execution_row) != 0) {
                fprintf(stderr, "--row must be -1 or a non-negative 32-bit integer.\n");
                return 2;
            }
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
           opt.weight_decay > 0.0f ? "AdamW" : "SGD", opt.steps,
           loaded_step + 1, opt.trainable_count, target_count);
    for (long index = 0; index < opt.steps; index++) {
        long step = loaded_step + index + 1;
        float loss = 0.0f;
        int correct = 0;
        int examples = 0;
        double started = now_ms();
        int update_mode = opt.weight_decay > 0.0f
            ? VOLVOXAI_TENSOR_UPDATE_ADAMW : VOLVOXAI_TENSOR_UPDATE_SGD;
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
