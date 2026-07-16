#include "cJSON.h"
#include "adapter_runtime_internal.h"
#include "volvoxai.h"
#include "volvoxai_backend.h"
#include "volvoxai_training.h"
#include "engine_internal.h"
#include "safetensors.h"
#if VOLVOXAI_ENABLE_VULKAN
#include "vulkan_engine.h"
#else
/* Keep the common native test binary linkable when Vulkan is compiled out.
 * Vulkan-specific cases already treat initialization failure as a skip. */
static int vk_init(void) { return -1; }
static int vk_training_available(void) { return 0; }
static void vk_cleanup(void) {}
#endif

#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return -1; } } while (0)

static int g_training_sdk_supports;
static int g_training_sdk_teardowns;

static int training_sdk_init(void* user_data) {
    (void)user_data;
    return VX_INIT_READY;
}

static int training_sdk_supports(void* user_data, const VxNode* node) {
    (void)user_data;
    (void)node;
    g_training_sdk_supports++;
    return VX_DECLINED;
}

static int training_sdk_run(void* user_data, const VxNode* node) {
    (void)user_data;
    (void)node;
    return VX_ERROR;
}

static void training_sdk_teardown(void* user_data) {
    (void)user_data;
    g_training_sdk_teardowns++;
}

static T* add_tensor(const char* name, const int* shape, int ndim, const float* values) {
    if (g_nt >= MAXT) return NULL;
    T* t = &g_t[g_nt++];
    memset(t, 0, sizeof(*t));
    strncpy(t->name, name, sizeof(t->name) - 1);
    t->ndim = ndim;
    t->numel = 1;
    for (int i = 0; i < ndim; i++) { t->shape[i] = shape[i]; t->numel *= shape[i]; }
    t->dtype = T_F32;
    t->elem_size = sizeof(float);
    if (values) {
        if (g_weight_file_count != 1 ||
            safetensors_add_tensor(&g_weight_files[0], name, SAFETENSORS_DTYPE_F32,
                                   shape, ndim, values,
                                   (size_t)t->numel * sizeof(float)) != 0) {
            memset(t, 0, sizeof(*t));
            g_nt--;
            return NULL;
        }
        SafetensorsTensor* stored = safetensors_find_tensor_mutable(&g_weight_files[0], name);
        t->owns = 0;
        t->data = stored ? (float*)stored->data : NULL;
    } else {
        t->owns = 1;
        t->data = (float*)calloc(t->numel > 0 ? (size_t)t->numel : 1u, sizeof(float));
    }
    if (!t->data) return NULL;
    return t;
}

static T* add_i32_tensor(const char* name, const int* shape, int ndim, const int32_t* values) {
    if (g_nt >= MAXT) return NULL;
    T* t = &g_t[g_nt++];
    memset(t, 0, sizeof(*t));
    strncpy(t->name, name, sizeof(t->name) - 1);
    t->ndim = ndim;
    t->numel = 1;
    for (int i = 0; i < ndim; i++) { t->shape[i] = shape[i]; t->numel *= shape[i]; }
    t->dtype = T_I32;
    t->elem_size = sizeof(int32_t);
    if (values) {
        if (g_weight_file_count != 1 ||
            safetensors_add_tensor(&g_weight_files[0], name, SAFETENSORS_DTYPE_I32,
                                   shape, ndim, values,
                                   (size_t)t->numel * sizeof(int32_t)) != 0) {
            memset(t, 0, sizeof(*t));
            g_nt--;
            return NULL;
        }
        SafetensorsTensor* stored = safetensors_find_tensor_mutable(&g_weight_files[0], name);
        t->owns = 0;
        t->data = stored ? (float*)stored->data : NULL;
    } else {
        t->owns = 1;
        t->data = (float*)calloc(t->numel > 0 ? (size_t)t->numel : 1u, sizeof(int32_t));
    }
    if (!t->data) return NULL;
    return t;
}

static Node* add_node(const char* op, const char* out, const char* params_json) {
    if (g_nn >= MAXN) return NULL;
    Node* n = &g_n[g_nn++];
    memset(n, 0, sizeof(*n));
    strncpy(n->op, op, sizeof(n->op) - 1);
    strncpy(n->out, out, sizeof(n->out) - 1);
    strncpy(n->outs[0].key, "output", sizeof(n->outs[0].key) - 1);
    strncpy(n->outs[0].name, out, sizeof(n->outs[0].name) - 1);
    n->nout = 1;
    n->params = cJSON_Parse(params_json ? params_json : "{}");
    n->owns_params = 1;
    return n->params ? n : NULL;
}

static int node_input(Node* n, const char* key, const char* name) {
    if (!n || n->nin >= MAXIN) return -1;
    strncpy(n->ins[n->nin].key, key, sizeof(n->ins[n->nin].key) - 1);
    strncpy(n->ins[n->nin].name, name, sizeof(n->ins[n->nin].name) - 1);
    n->nin++;
    return 0;
}

static void begin_graph(void) {
    memset(g_t, 0, sizeof(g_t));
    memset(g_n, 0, sizeof(g_n));
    for (int index = 0; index < MAXN; index++) g_concat_sigmoid_fuse[index] = -1;
    g_nt = 0;
    volvoxai_engine_tensor_name_index_invalidate();
    g_nn = 0;
    memset(g_weight_files, 0, sizeof(g_weight_files));
    if (safetensors_init_empty(&g_weight_files[0], SAFETENSORS_OPEN_READ_WRITE) != 0) abort();
    g_weight_file_count = 1;
    g_loaded = 1;
    g_weight_caches_dirty = 0;
    g_active_row = -1;
    g_prefix_rows = 0;
    g_execution_row = -1;
}

static int finish_graph(void) {
    volvoxai_engine_shutdown();
    return 0;
}

static float cross_entropy_loss(const T* logits, const int* targets, int count) {
    if (!logits || !targets || logits->ndim <= 0 || count <= 0) return NAN;
    int classes = logits->shape[logits->ndim - 1];
    if (classes <= 0 || logits->numel != (long)count * classes) return NAN;
    double loss = 0.0;
    for (int row = 0; row < count; row++) {
        const float* values = logits->data + (long)row * classes;
        float max_value = values[0];
        for (int c = 1; c < classes; c++) if (values[c] > max_value) max_value = values[c];
        double sum = 0.0;
        for (int c = 0; c < classes; c++) sum += exp((double)values[c] - max_value);
        loss += (double)max_value + log(sum) - values[targets[row]];
    }
    return (float)(loss / count);
}

static float test_fast_tanh(float x) {
    union { float f; uint32_t i; } value;
    value.i = (uint32_t)(12102203.0f * (2.0f * x) + 1064866805.0f);
    return (value.f - 1.0f) / (value.f + 1.0f);
}

static float reference_gelu(float x, int approximate_tanh) {
    if (approximate_tanh) {
        return 0.5f * x * (1.0f + test_fast_tanh(
            0.79788456f * (x + 0.044715f * x * x * x)));
    }
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

static float reference_gelu_derivative(float x, int approximate_tanh) {
    if (approximate_tanh) {
        float x2 = x * x;
        float u = 0.7978845608028654f * (x + 0.044715f * x * x2);
        float t = tanhf(u);
        return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) *
            0.7978845608028654f * (1.0f + 0.134145f * x2);
    }
    return 0.5f * (1.0f + erff(x * 0.7071067811865475f)) +
        x * 0.3989422804014327f * expf(-0.5f * x * x);
}

static int test_public_backend_bypasses_training_tape(void) {
    const VxBackendV1 backend = {
        .struct_size = sizeof(VxBackendV1),
        .abi_version = VX_BACKEND_ABI_V1,
        .name = "training-guard-probe",
        .init = training_sdk_init,
        .supports = training_sdk_supports,
        .run = training_sdk_run,
        .teardown = training_sdk_teardown,
    };
    CHECK(volvoxai_register_backend(&backend) == 0);
    CHECK(volvoxai_engine_configure_backend("training-guard-probe") == 0);
    begin_graph();
    const int shape[2] = {1, 2};
    const float input[2] = {-0.5f, 1.0f};
    CHECK(add_tensor("probe.input", shape, 2, input));
    CHECK(add_tensor("logits", shape, 2, NULL));
    Node* node = add_node("GELU", "logits", NULL);
    CHECK(node && node_input(node, "input", "probe.input") == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(g_training_sdk_supports == 1);
    const int target[1] = {1};
    const char* trainable[1] = {"probe.input"};
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, trainable, 1,
                                     VOLVOXAI_TENSOR_UPDATE_SGD, 0.01f,
                                     0.9f, 0.999f, 1e-8f, 0.0f, 0.0f,
                                     1, NULL, NULL, NULL) == 0);
    CHECK(g_training_sdk_supports == 1);
    CHECK(finish_graph() == 0);
    CHECK(g_training_sdk_teardowns == 1);
    return 0;
}

static int test_gelu_mode_case(int approximate_tanh) {
    begin_graph();
    g_use_vulkan = g_use_nnapi = g_use_opengl = g_use_metal = 0;
    const int shape[2] = {1, 5};
    const float input[5] = {-2.0f, -0.75f, 0.0f, 0.5f, 1.5f};
    CHECK(add_tensor("gelu.input", shape, 2, input));
    CHECK(add_tensor("logits", shape, 2, NULL));
    Node* node = add_node("GELU", "logits",
        approximate_tanh ? "{\"approximate\":\"tanh\"}" : NULL);
    CHECK(node && node_input(node, "input", "gelu.input") == 0);
    CHECK(volvoxai_engine_forward() == 0);
    float expected_output[5];
    for (int i = 0; i < 5; i++) {
        expected_output[i] = reference_gelu(input[i], approximate_tanh);
        CHECK(fabsf(t_find("logits")->data[i] - expected_output[i]) < 2.0e-6f);
    }
    if (!approximate_tanh) {
        cJSON_AddStringToObject(node->params, "approximate", "none");
        CHECK(volvoxai_engine_forward() == 0);
        for (int i = 0; i < 5; i++)
            CHECK(fabsf(t_find("logits")->data[i] - expected_output[i]) < 2.0e-6f);
    }
    float max_value = expected_output[0];
    for (int i = 1; i < 5; i++) if (expected_output[i] > max_value) max_value = expected_output[i];
    float probabilities[5];
    double denominator = 0.0;
    for (int i = 0; i < 5; i++) {
        probabilities[i] = expf(expected_output[i] - max_value);
        denominator += probabilities[i];
    }
    for (int i = 0; i < 5; i++) probabilities[i] /= (float)denominator;
    const int target[1] = {3};
    const char* trainable[1] = {"gelu.input"};
    const float learning_rate = 0.1f;
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, trainable, 1,
                                     2, learning_rate, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 1, NULL, NULL, NULL) == 0);
    for (int i = 0; i < 5; i++) {
        float upstream = probabilities[i] - (i == target[0] ? 1.0f : 0.0f);
        float expected_gradient = upstream * reference_gelu_derivative(input[i], approximate_tanh);
        float actual_gradient = (input[i] - t_find("gelu.input")->data[i]) / learning_rate;
        CHECK(fabsf(actual_gradient - expected_gradient) < 8.0e-5f);
    }
    return finish_graph();
}

static int test_gelu_exact_and_tanh(void) {
    CHECK(test_gelu_mode_case(0) == 0);
    CHECK(test_gelu_mode_case(1) == 0);
    return 0;
}

static int test_transformer_backward(void) {
    begin_graph();
    const int token_shape[1] = {2};
    const int hidden_shape[2] = {2, 2};
    const int qkv_shape[2] = {2, 6};
    const int emb_shape[2] = {3, 2};
    const int vec2_shape[1] = {2};
    const int qkv_w_shape[2] = {2, 6};
    const int logits_w_shape[2] = {2, 2};
    const int32_t tokens[2] = {0, 1};
    const float embedding[6] = {0.8f, -0.3f, -0.2f, 1.1f, 0.5f, 0.4f};
    const float norm_weight[2] = {1.1f, 0.9f};
    const float norm_bias[2] = {0.1f, -0.05f};
    const float qkv_weight[12] = {
        0.7f, -0.2f, 0.4f, 0.1f, 0.5f, -0.3f,
       -0.1f,  0.8f, 0.2f, 0.6f, 0.3f,  0.9f
    };
    const float logits_weight[4] = {0.6f, -0.4f, -0.2f, 0.7f};
    CHECK(add_i32_tensor("tokens", token_shape, 1, tokens));
    CHECK(add_tensor("embedding.weight", emb_shape, 2, embedding));
    CHECK(add_tensor("emb", hidden_shape, 2, NULL));
    CHECK(add_tensor("ln.weight", vec2_shape, 1, norm_weight));
    CHECK(add_tensor("ln.bias", vec2_shape, 1, norm_bias));
    CHECK(add_tensor("ln", hidden_shape, 2, NULL));
    CHECK(add_tensor("residual", hidden_shape, 2, NULL));
    CHECK(add_tensor("rms.weight", vec2_shape, 1, norm_weight));
    CHECK(add_tensor("rms", hidden_shape, 2, NULL));
    CHECK(add_tensor("qkv.weight", qkv_w_shape, 2, qkv_weight));
    CHECK(add_tensor("qkv", qkv_shape, 2, NULL));
    CHECK(add_tensor("attn", hidden_shape, 2, NULL));
    CHECK(add_tensor("gelu", hidden_shape, 2, NULL));
    CHECK(add_tensor("silu", hidden_shape, 2, NULL));
    CHECK(add_tensor("sigmoid", hidden_shape, 2, NULL));
    CHECK(add_tensor("tanh", hidden_shape, 2, NULL));
    CHECK(add_tensor("relu", hidden_shape, 2, NULL));
    CHECK(add_tensor("logits.weight", logits_w_shape, 2, logits_weight));
    CHECK(add_tensor("logits", hidden_shape, 2, NULL));

    Node* n = add_node("Embedding", "emb", NULL);
    CHECK(n && node_input(n, "input", "tokens") == 0 && node_input(n, "weight", "embedding.weight") == 0);
    n = add_node("LayerNorm", "ln", "{\"d_model\":2,\"eps\":1e-5}");
    CHECK(n && node_input(n, "input", "emb") == 0 && node_input(n, "weight", "ln.weight") == 0 && node_input(n, "bias", "ln.bias") == 0);
    n = add_node("Add", "residual", NULL);
    CHECK(n && node_input(n, "a", "ln") == 0 && node_input(n, "b", "emb") == 0);
    n = add_node("RMSNorm", "rms", "{\"d_model\":2,\"eps\":1e-5}");
    CHECK(n && node_input(n, "input", "residual") == 0 && node_input(n, "weight", "rms.weight") == 0);
    n = add_node("MatMul", "qkv", NULL);
    CHECK(n && node_input(n, "input", "rms") == 0 && node_input(n, "weight", "qkv.weight") == 0);
    n = add_node("SDPA", "attn", "{\"heads\":1,\"scale\":0.70710678}");
    CHECK(n && node_input(n, "qkv", "qkv") == 0);
    n = add_node("GELU", "gelu", NULL); CHECK(n && node_input(n, "input", "attn") == 0);
    n = add_node("SiLU", "silu", NULL); CHECK(n && node_input(n, "input", "gelu") == 0);
    n = add_node("Sigmoid", "sigmoid", NULL); CHECK(n && node_input(n, "input", "silu") == 0);
    n = add_node("Tanh", "tanh", NULL); CHECK(n && node_input(n, "input", "sigmoid") == 0);
    n = add_node("ReLU", "relu", NULL); CHECK(n && node_input(n, "input", "tanh") == 0);
    n = add_node("MatMul", "logits", NULL);
    CHECK(n && node_input(n, "input", "relu") == 0 && node_input(n, "weight", "logits.weight") == 0);

    float before[6]; memcpy(before, t_find("embedding.weight")->data, sizeof(before));
    const int targets[2] = {1, 0};
    const char* trainable[1] = {"embedding.weight"};
    float loss = 0.0f;
    int correct = 0, examples = 0;
    /* Exercise backend-policy restoration without fabricating initialized GPU
       devices. The dedicated optional Vulkan case below covers a live device. */
    g_use_vulkan = 0;
    g_use_nnapi = 1;
    g_use_opengl = 0;
    g_use_metal = 0;
    CHECK(volvoxai_engine_train_step("logits", targets, 2, INT_MIN, trainable, 1,
                                          2, 0.05f, 0.9f, 0.999f, 1e-8f,
                                          0.0f, 0.0f, 1, &loss, &correct, &examples) == 0);
    CHECK(g_use_vulkan == 0 && g_use_nnapi == 1 && g_use_opengl == 0 && g_use_metal == 0);
    g_use_vulkan = g_use_nnapi = g_use_opengl = g_use_metal = 0;
    CHECK(isfinite(loss) && loss > 0.0f && examples == 2 && correct >= 0 && correct <= 2);
    T* trained = t_find("embedding.weight");
    int changed = 0;
    for (int i = 0; i < 6; i++) if (fabsf(trained->data[i] - before[i]) > 1e-7f) changed = 1;
    CHECK(changed);
    /* The first step restores the inference arena; a second step verifies that
       training de-pools it again before consuming saved activations. */
    CHECK(volvoxai_engine_train_step("logits", targets, 2, INT_MIN, trainable, 1,
                                          2, 0.01f, 0.9f, 0.999f, 1e-8f,
                                          0.0f, 0.0f, 2, &loss, &correct, &examples) == 0);
    CHECK(isfinite(loss) && examples == 2);
    return finish_graph();
}

static int test_cnn_backward(void) {
    begin_graph();
    const int input_shape[4] = {1, 3, 3, 1};
    const int conv_w_shape[4] = {2, 2, 1, 2};
    const int conv_shape[4] = {1, 2, 2, 2};
    const int pooled_shape[4] = {1, 1, 1, 2};
    const int vec2_shape[1] = {2};
    const int linear_shape[2] = {2, 2};
    const int logits_shape[2] = {1, 2};
    const float input[9] = {0.2f, -0.1f, 0.4f, 0.7f, 0.3f, -0.2f, 0.5f, 0.9f, 0.1f};
    const float conv_w[8] = {0.3f, -0.2f, 0.4f, 0.1f, -0.1f, 0.5f, 0.2f, 0.6f};
    const float zeros[2] = {0, 0};
    const float ones[2] = {1, 1};
    const float linear[4] = {0.6f, -0.4f, 0.3f, 0.8f};
    CHECK(add_tensor("image", input_shape, 4, input));
    CHECK(add_tensor("conv.weight", conv_w_shape, 4, conv_w));
    CHECK(add_tensor("conv.bias", vec2_shape, 1, zeros));
    CHECK(add_tensor("conv", conv_shape, 4, NULL));
    CHECK(add_tensor("bn.weight", vec2_shape, 1, ones));
    CHECK(add_tensor("bn.bias", vec2_shape, 1, zeros));
    CHECK(add_tensor("bn.mean", vec2_shape, 1, zeros));
    CHECK(add_tensor("bn.var", vec2_shape, 1, ones));
    CHECK(add_tensor("bn", conv_shape, 4, NULL));
    CHECK(add_tensor("relu", conv_shape, 4, NULL));
    CHECK(add_tensor("pool", pooled_shape, 4, NULL));
    CHECK(add_tensor("resize", conv_shape, 4, NULL));
    CHECK(add_tensor("gap", logits_shape, 2, NULL));
    CHECK(add_tensor("head.weight", linear_shape, 2, linear));
    CHECK(add_tensor("logits", logits_shape, 2, NULL));

    Node* n = add_node("Conv2D", "conv", "{\"weight_layout\":\"HWIO\",\"stride\":[1,1],\"padding\":[0,0]}");
    CHECK(n && node_input(n, "input", "image") == 0 && node_input(n, "weight", "conv.weight") == 0 && node_input(n, "bias", "conv.bias") == 0);
    n = add_node("BatchNorm2D", "bn", "{\"eps\":1e-5}");
    CHECK(n && node_input(n, "input", "conv") == 0 && node_input(n, "weight", "bn.weight") == 0 &&
          node_input(n, "bias", "bn.bias") == 0 && node_input(n, "running_mean", "bn.mean") == 0 &&
          node_input(n, "running_var", "bn.var") == 0);
    n = add_node("ReLU", "relu", NULL); CHECK(n && node_input(n, "input", "bn") == 0);
    n = add_node("MaxPool2D", "pool", "{\"kernel\":[2,2],\"stride\":[2,2]}");
    CHECK(n && node_input(n, "input", "relu") == 0);
    n = add_node("ResizeNearest2D", "resize", NULL); CHECK(n && node_input(n, "input", "pool") == 0);
    n = add_node("GlobalAveragePool", "gap", NULL); CHECK(n && node_input(n, "input", "resize") == 0);
    n = add_node("MatMul", "logits", NULL);
    CHECK(n && node_input(n, "input", "gap") == 0 && node_input(n, "weight", "head.weight") == 0);

    float before[8]; memcpy(before, t_find("conv.weight")->data, sizeof(before));
    const int target[1] = {1};
    const char* trainable[1] = {"conv.weight"};
    float loss = 0.0f;
    int examples = 0;
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, trainable, 1,
                                          2, 0.05f, 0.9f, 0.999f, 1e-8f,
                                          0.0f, 0.0f, 1, &loss, NULL, &examples) == 0);
    CHECK(isfinite(loss) && loss > 0.0f && examples == 1);
    T* trained = t_find("conv.weight");
    int changed = 0;
    for (int i = 0; i < 8; i++) if (fabsf(trained->data[i] - before[i]) > 1e-7f) changed = 1;
    CHECK(changed);
    return finish_graph();
}

static float numerical_attention_gradient(const char* tensor_name, long index,
                                          const char* logits_name,
                                          const int* targets, int target_count) {
    T* tensor = t_find(tensor_name);
    if (!tensor || tensor->dtype != T_F32 || index < 0 || index >= tensor->numel) return NAN;
    const float epsilon = 1.0e-2f;
    float original = tensor->data[index];
    tensor->data[index] = original + epsilon;
    if (volvoxai_engine_forward() != 0) {
        tensor->data[index] = original;
        return NAN;
    }
    float positive = cross_entropy_loss(t_find(logits_name), targets, target_count);
    tensor->data[index] = original - epsilon;
    if (volvoxai_engine_forward() != 0) {
        tensor->data[index] = original;
        return NAN;
    }
    float negative = cross_entropy_loss(t_find(logits_name), targets, target_count);
    tensor->data[index] = original;
    return (positive - negative) / (2.0f * epsilon);
}

static int gradient_close(float actual, float expected) {
    float tolerance = 4.0e-4f + 4.0e-3f * fmaxf(fabsf(actual), fabsf(expected));
    return isfinite(actual) && isfinite(expected) && fabsf(actual - expected) <= tolerance;
}

static int approximate_attention_gradient_close(float actual, float expected) {
    /* The native forward kernel uses fast_expf while backward uses expf. Its
       finite-difference derivative is therefore approximate, especially for
       the query/key paths, but still catches batch-offset and routing errors. */
    float tolerance = 6.0e-3f + 0.1f * fmaxf(fabsf(actual), fabsf(expected));
    return isfinite(actual) && isfinite(expected) && fabsf(actual - expected) <= tolerance;
}

static int test_attention_batches(void) {
    begin_graph();
    const int qkv_shape[3] = {2, 2, 6};
    const int out_shape[3] = {2, 2, 2};
    const float qkv_values[24] = {
         0.20f, -0.10f,  0.30f,  0.40f,  0.50f, -0.20f,
        -0.30f,  0.60f,  0.10f, -0.50f,  0.70f,  0.20f,
        -0.40f,  0.70f,  0.20f, -0.10f,  0.30f,  0.60f,
         0.80f, -0.20f, -0.50f,  0.40f, -0.60f,  0.10f
    };
    CHECK(add_tensor("qkv", qkv_shape, 3, qkv_values));
    CHECK(add_tensor("logits", out_shape, 3, NULL));
    Node* n = add_node("SDPA", "logits", "{\"heads\":1,\"scale\":0.70710678}");
    CHECK(n && node_input(n, "qkv", "qkv") == 0);
    CHECK(volvoxai_engine_forward() == 0);

    /* Changing batch zero must not affect batch one. */
    float batch_one[4];
    memcpy(batch_one, t_find("logits")->data + 4, sizeof(batch_one));
    t_find("qkv")->data[0] += 0.35f;
    CHECK(volvoxai_engine_forward() == 0);
    for (int i = 0; i < 4; i++) CHECK(fabsf(t_find("logits")->data[4 + i] - batch_one[i]) < 1.0e-7f);
    t_find("qkv")->data[0] = qkv_values[0];

    /* Decode uses one independent K/V cache per batch row. The position-one
       decode result must match the corresponding full causal forward row. */
    float full_decode_rows[4] = {
        t_find("logits")->data[2], t_find("logits")->data[3],
        t_find("logits")->data[6], t_find("logits")->data[7],
    };
    memset(t_find("logits")->data, 0, (size_t)t_find("logits")->numel * sizeof(float));
    g_active_row = 1;
    CHECK(run_node(n, 0, 1) == 0);
    CHECK(fabsf(t_find("logits")->data[2] - full_decode_rows[0]) < 2.0e-3f);
    CHECK(fabsf(t_find("logits")->data[3] - full_decode_rows[1]) < 2.0e-3f);
    CHECK(fabsf(t_find("logits")->data[6] - full_decode_rows[2]) < 2.0e-3f);
    CHECK(fabsf(t_find("logits")->data[7] - full_decode_rows[3]) < 2.0e-3f);
    g_active_row = -1;

    /* One target per batch row selects the requested/last sequence position
       in each row, rather than a contiguous suffix from only the final row. */
    const int per_batch_targets[2] = {1, 0};
    const char* per_batch_trainable[1] = {"qkv"};
    int per_batch_examples = 0;
    CHECK(volvoxai_engine_train_step("logits", per_batch_targets, 2, INT_MIN, per_batch_trainable, 1,
                                     2, 0.01f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 1, NULL, NULL, &per_batch_examples) == 0);
    CHECK(per_batch_examples == 2);
    int changed_batch_zero = 0, changed_batch_one = 0;
    for (int i = 0; i < 12; i++) if (fabsf(t_find("qkv")->data[i] - qkv_values[i]) > 1e-8f) changed_batch_zero = 1;
    for (int i = 12; i < 24; i++) if (fabsf(t_find("qkv")->data[i] - qkv_values[i]) > 1e-8f) changed_batch_one = 1;
    CHECK(changed_batch_zero && changed_batch_one);
    memcpy(t_find("qkv")->data, qkv_values, sizeof(qkv_values));

    const int targets[4] = {0, 1, 1, 0};
    const long probes[2] = {4, 16};
    float numeric[2];
    for (int i = 0; i < 2; i++) {
        numeric[i] = numerical_attention_gradient("qkv", probes[i], "logits", targets, 4);
        CHECK(isfinite(numeric[i]));
    }
    float before[24]; memcpy(before, t_find("qkv")->data, sizeof(before));
    const char* trainable[1] = {"qkv"};
    const float learning_rate = 0.05f;
    float loss = 0.0f;
    int examples = 0;
    CHECK(volvoxai_engine_train_step("logits", targets, 4, INT_MIN, trainable, 1,
                                     2, learning_rate, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 1, &loss, NULL, &examples) == 0);
    CHECK(isfinite(loss) && examples == 4);
    for (int i = 0; i < 2; i++) {
        float analytic = (before[probes[i]] - t_find("qkv")->data[probes[i]]) / learning_rate;
        CHECK(gradient_close(analytic, numeric[i]));
    }
    volvoxai_engine_shutdown();

    begin_graph();
    const int q_shape[3] = {2, 2, 2};
    const int kv_shape[3] = {2, 3, 2};
    const int cross_out_shape[3] = {2, 2, 2};
    const float q_values[8] = {0.2f, -0.1f, -0.3f, 0.6f, -0.4f, 0.7f, 0.8f, -0.2f};
    const float k_values[12] = {
        0.3f, 0.4f, 0.1f, -0.5f, -0.2f, 0.8f,
        0.6f, -0.1f, -0.4f, 0.2f, 0.7f, 0.3f
    };
    const float v_values[12] = {
         0.5f, -0.2f, 0.7f, 0.2f, -0.6f, 0.1f,
        -0.3f,  0.9f, 0.4f, 0.6f,  0.2f, -0.7f
    };
    CHECK(add_tensor("q", q_shape, 3, q_values));
    CHECK(add_tensor("k", kv_shape, 3, k_values));
    CHECK(add_tensor("v", kv_shape, 3, v_values));
    CHECK(add_tensor("cross", cross_out_shape, 3, NULL));
    n = add_node("CrossSDPA", "cross", "{\"heads\":1,\"scale\":0.37}");
    CHECK(n && node_input(n, "q", "q") == 0 && node_input(n, "k", "k") == 0 &&
          node_input(n, "v", "v") == 0);
    CHECK(volvoxai_engine_forward() == 0);

    memcpy(batch_one, t_find("cross")->data + 4, sizeof(batch_one));
    t_find("q")->data[0] += 0.35f;
    CHECK(volvoxai_engine_forward() == 0);
    for (int i = 0; i < 4; i++) CHECK(fabsf(t_find("cross")->data[4 + i] - batch_one[i]) < 1.0e-7f);
    t_find("q")->data[0] = q_values[0];

    const char* tensor_names[3] = {"q", "k", "v"};
    const long cross_probes[3] = {1, 10, 5};
    float cross_numeric[3];
    float cross_before[3];
    for (int i = 0; i < 3; i++) {
        cross_numeric[i] = numerical_attention_gradient(tensor_names[i], cross_probes[i], "cross", targets, 4);
        CHECK(isfinite(cross_numeric[i]));
        cross_before[i] = t_find(tensor_names[i])->data[cross_probes[i]];
    }
    CHECK(volvoxai_engine_train_step("cross", targets, 4, INT_MIN, tensor_names, 3,
                                     2, learning_rate, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 1, &loss, NULL, &examples) == 0);
    CHECK(isfinite(loss) && examples == 4);
    for (int i = 0; i < 3; i++) {
        float analytic = (cross_before[i] - t_find(tensor_names[i])->data[cross_probes[i]]) / learning_rate;
        CHECK(approximate_attention_gradient_close(analytic, cross_numeric[i]));
    }
    return finish_graph();
}

static int test_vulkan_engine_train_step_optional(void) {
    if (vk_init() != 0) {
        puts("native Vulkan train_step integration skipped: no Vulkan compute device");
        return 0;
    }
    if (!vk_training_available()) {
        vk_cleanup();
        puts("native Vulkan train_step integration skipped: training backend unavailable");
        return 0;
    }

    begin_graph();
    const int matrix_shape[2] = {2, 2};
    const float input[4] = {0.8f, -0.3f, -0.2f, 1.1f};
    const float weight[4] = {0.6f, -0.4f, -0.2f, 0.7f};
    CHECK(add_tensor("input", matrix_shape, 2, input));
    CHECK(add_tensor("weight", matrix_shape, 2, weight));
    CHECK(add_tensor("logits", matrix_shape, 2, NULL));
    Node* n = add_node("MatMul", "logits", NULL);
    CHECK(n && node_input(n, "input", "input") == 0 && node_input(n, "weight", "weight") == 0);

    const int targets[2] = {1, 0};
    const char* trainable[1] = {"weight"};
    float before[4];
    memcpy(before, t_find("weight")->data, sizeof(before));
    float loss = 0.0f;
    int examples = 0;
    g_use_vulkan = 1;
    g_use_nnapi = 1;
    g_use_opengl = 0;
    g_use_metal = 0;
    CHECK(volvoxai_engine_train_step("logits", targets, 2, INT_MIN, trainable, 1,
                                     2, 0.05f, 0.9f, 0.999f, 1.0e-8f,
                                     0.0f, 0.0f, 1, &loss, NULL, &examples) == 0);
    CHECK(g_use_vulkan == 1 && g_use_nnapi == 1 && g_use_opengl == 0 && g_use_metal == 0);
    CHECK(isfinite(loss) && examples == 2);
    int changed = 0;
    for (int i = 0; i < 4; i++) {
        if (fabsf(t_find("weight")->data[i] - before[i]) > 1.0e-7f) changed = 1;
    }
    CHECK(changed);

    g_use_vulkan = g_use_nnapi = g_use_opengl = g_use_metal = 0;
    CHECK(finish_graph() == 0);
    vk_cleanup();
    return 0;
}

static int test_append_first_node(void) {
    begin_graph();
    const int shape[2] = {1, 2};
    const float input[2] = {1, 2};
    CHECK(add_tensor("x", shape, 2, input));
    const char* patch =
        "{\"opType\":\"Identity\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"output\":\"y\"},\"output_shapes\":{\"output\":[1,2]}}";
    CHECK(volvoxai_engine_patch_node_json(-1, patch, VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_AFTER, 0) == 0);
    CHECK(g_nn == 1 && !strcmp(g_n[0].op, "Identity") && t_find("y") != NULL);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(t_find("y")->data[0] == 1.0f && t_find("y")->data[1] == 2.0f);
    return finish_graph();
}

static int test_patch_graph_transactions(void) {
    begin_graph();
    const char* saved_path = "/tmp/volvox-patch-graph.json";
    const int shape[2] = {1, 2};
    const float input[2] = {1.0f, -2.0f};
    CHECK(add_tensor("x", shape, 2, input));
    g_t[0].is_graph_input = 1;
    g_cfg_root = cJSON_Parse("{\"inputs\":{\"x\":{\"shape\":[1,2]}},\"nodes\":[]}");
    CHECK(g_cfg_root != NULL);
    const char* add_y =
        "{\"opType\":\"Identity\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"output\":\"y\"},\"output_shapes\":{\"output\":[1,2]}}";
    const char* add_z =
        "{\"opType\":\"ReLU\",\"inputs\":{\"input\":\"y\"},"
        "\"outputs\":{\"output\":\"z\"},\"output_shapes\":{\"output\":[1,2]}}";
    CHECK(volvoxai_engine_patch_node_json(-1, add_y, VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_AFTER, 0) == 0);
    CHECK(volvoxai_engine_patch_node_json(-1, add_z, VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_AFTER, 0) == 0);
    CHECK(g_nn == 2 && g_nt == 3 && volvoxai_engine_forward() == 0);
    CHECK(t_find("z")->data[0] == 1.0f && t_find("z")->data[1] == 0.0f);

    char* inspection = volvoxai_engine_inspect_model_json(1, 1, 0, 1);
    CHECK(inspection != NULL);
    cJSON* inspected = cJSON_Parse(inspection);
    free(inspection);
    CHECK(inspected && cJSON_GetObjectItem(inspected, "num_ops")->valueint == 2 &&
          cJSON_GetArraySize(cJSON_GetObjectItem(inspected, "nodes")) == 2);
    cJSON_Delete(inspected);
    CHECK(volvoxai_engine_save_config(saved_path) == 0);
    long saved_size = 0;
    char* saved_text = read_file(saved_path, &saved_size);
    cJSON* saved = saved_text ? cJSON_Parse(saved_text) : NULL;
    free(saved_text);
    CHECK(saved && cJSON_GetArraySize(cJSON_GetObjectItem(saved, "nodes")) == 2);
    cJSON_Delete(saved);

    const char* invalid_topology =
        "[{\"node_index\":0,\"mode\":3,\"patch\":{\"opType\":\"Identity\","
        "\"inputs\":{\"input\":\"z\"},\"outputs\":{\"output\":\"bad\"},"
        "\"output_shapes\":{\"output\":[1,2]}}}]";
    CHECK(volvoxai_engine_patch_graph_json(invalid_topology, 0, 1) == -1);
    CHECK(g_nn == 2 && g_nt == 3 && t_find("bad") == NULL && volvoxai_engine_forward() == 0);

    const char* invalid_batch =
        "[{\"node_index\":-1,\"mode\":4,\"patch\":{\"opType\":\"Identity\","
        "\"inputs\":{\"input\":\"z\"},\"outputs\":{\"output\":\"u\"},"
        "\"output_shapes\":{\"output\":[1,2]}}},"
        "{\"node_index\":-1,\"mode\":4,\"patch\":{\"opType\":\"Identity\","
        "\"inputs\":{\"input\":\"x\"},\"outputs\":{\"output\":\"y\"},"
        "\"output_shapes\":{\"output\":[1,2]}}}]";
    CHECK(volvoxai_engine_patch_graph_json(invalid_batch, 0, 1) == -1);
    CHECK(g_nn == 2 && g_nt == 3 && t_find("u") == NULL && volvoxai_engine_forward() == 0);
    const char* failed_persist =
        "{\"patches\":[{\"node_index\":-1,\"mode\":4,\"patch\":{"
        "\"opType\":\"Identity\",\"inputs\":{\"input\":\"z\"},"
        "\"outputs\":{\"output\":\"persist_fail\"},"
        "\"output_shapes\":{\"output\":[1,2]}}}],"
        "\"persist_config_path\":\"/no/such/volvox/directory/config.json\"}";
    CHECK(volvoxai_engine_patch_graph_json(failed_persist, 0, 1) == -1);
    CHECK(g_nn == 2 && g_nt == 3 && t_find("persist_fail") == NULL && volvoxai_engine_forward() == 0);

    CHECK(volvoxai_engine_patch_node_json(0, "{}", VOLVOXAI_ENGINE_NODE_PATCH_MODE_DELETE, 0) == -1);
    CHECK(g_nn == 2 && volvoxai_engine_forward() == 0);
    const char* delete_both =
        "[{\"node_index\":1,\"mode\":5,\"patch\":{}},"
        "{\"node_index\":0,\"mode\":5,\"patch\":{}}]";
    CHECK(volvoxai_engine_patch_graph_json(delete_both, 0, 0) == 0);
    CHECK(g_nn == 0);
    inspection = volvoxai_engine_inspect_model_json(1, 1, 0, 0);
    CHECK(inspection != NULL);
    inspected = cJSON_Parse(inspection);
    free(inspection);
    CHECK(inspected && cJSON_GetObjectItem(inspected, "num_ops")->valueint == 0 &&
          cJSON_GetObjectItem(inspected, "num_tensors")->valueint == 1 &&
          cJSON_GetArraySize(cJSON_GetObjectItem(inspected, "nodes")) == 0);
    cJSON_Delete(inspected);
    float removed_values[2];
    CHECK(volvoxai_engine_copy_tensor_f32("y", removed_values, 2) == -1);
    CHECK(volvoxai_engine_tensor_info("z", NULL, NULL, NULL) == -1);
    const char* consume_removed =
        "{\"opType\":\"Identity\",\"inputs\":{\"input\":\"y\"},"
        "\"outputs\":{\"output\":\"q\"},\"output_shapes\":{\"output\":[1,2]}}";
    CHECK(volvoxai_engine_patch_node_json(-1, consume_removed,
                                           VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_AFTER, 0) == -1);
    CHECK(g_nn == 0 && t_find("q") == NULL);
    CHECK(volvoxai_engine_save_config(saved_path) == 0);
    saved_text = read_file(saved_path, &saved_size);
    saved = saved_text ? cJSON_Parse(saved_text) : NULL;
    free(saved_text);
    CHECK(saved && cJSON_GetArraySize(cJSON_GetObjectItem(saved, "nodes")) == 0);
    cJSON_Delete(saved);

    CHECK(volvoxai_engine_patch_node_json(-1, add_y, VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_AFTER, 0) == 0);
    CHECK(volvoxai_engine_patch_node_json(-1, add_z, VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_AFTER, 0) == 0);
    CHECK(g_nn == 2 && g_nt == 3 && volvoxai_engine_forward() == 0);
    for (int i = 0; i < 32; i++) {
        CHECK(volvoxai_engine_patch_node_json(1, "{}", VOLVOXAI_ENGINE_NODE_PATCH_MODE_DELETE, 0) == 0);
        CHECK(g_nn == 1);
        char unique_patch[512];
        snprintf(unique_patch, sizeof(unique_patch),
                 "{\"opType\":\"ReLU\",\"inputs\":{\"input\":\"y\"},"
                 "\"outputs\":{\"output\":\"z_%d\"},"
                 "\"output_shapes\":{\"output\":[1,2]}}", i);
        CHECK(volvoxai_engine_patch_node_json(-1, unique_patch,
                                               VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_AFTER, 0) == 0);
        CHECK(g_nn == 2 && g_nt == 3);
    }
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_patch_graph_json(
              "{\"patches\":[],\"declared_outputs\":[\"z_31\"],"
              "\"persist_config_path\":\"/tmp/volvox-patch-graph.json\"}", 0, 0) == 0);
    saved_text = read_file(saved_path, &saved_size);
    saved = saved_text ? cJSON_Parse(saved_text) : NULL;
    free(saved_text);
    cJSON* declared = saved ? cJSON_GetObjectItem(saved, "outputs") : NULL;
    cJSON* declared_name = declared ? cJSON_GetObjectItem(declared, "z_31") : NULL;
    CHECK(saved && cJSON_GetArraySize(cJSON_GetObjectItem(saved, "nodes")) == 2 &&
          cJSON_IsString(declared_name) && !strcmp(declared_name->valuestring, "z_31"));
    cJSON_Delete(saved);

    const char* empty_weights_path = "/tmp/volvox-patch-empty.safetensors";
    SafetensorsFile empty_weights;
    CHECK(safetensors_init_empty(&empty_weights, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_save(empty_weights_path, &empty_weights) == 0);
    safetensors_free(&empty_weights);
    volvoxai_engine_shutdown();
    CHECK(volvoxai_engine_init(saved_path, empty_weights_path) == 0);
    declared = cJSON_GetObjectItem(g_cfg_root, "outputs");
    declared_name = declared ? cJSON_GetObjectItem(declared, "z_31") : NULL;
    CHECK(cJSON_IsString(declared_name) && !strcmp(declared_name->valuestring, "z_31"));
    long input_count = 0;
    float* reloaded_input = volvoxai_engine_input_ptr("x", &input_count);
    CHECK(reloaded_input && input_count == 2);
    reloaded_input[0] = 3.0f; reloaded_input[1] = -4.0f;
    CHECK(volvoxai_engine_forward() == 0 && t_find("z_31")->data[0] == 3.0f &&
          t_find("z_31")->data[1] == 0.0f);
    volvoxai_engine_shutdown();
    remove(empty_weights_path);
    remove(saved_path);
    return 0;
}

static int test_moe_routing_backward(void) {
    begin_graph();
    const int input_shape[2] = {2, 2};
    const int router_w_shape[2] = {2, 3};
    const int router_b_shape[1] = {3};
    const int route_shape[2] = {2, 2};
    const int expert_w_shape[3] = {3, 2, 2};
    const int expert_b_shape[2] = {3, 2};
    const float input[4] = {1.0f, 0.2f, -0.3f, 0.8f};
    const float router_w[6] = {0.8f, -0.2f, 0.3f, 0.1f, 0.7f, -0.4f};
    const float router_b[3] = {0.05f, -0.1f, 0.2f};
    const float expert_w[12] = {
        0.7f, -0.2f, 0.1f, 0.5f,
       -0.4f,  0.6f, 0.8f, 0.2f,
        0.3f,  0.9f, -0.5f, 0.4f
    };
    const float expert_b[6] = {0.1f, 0.0f, -0.1f, 0.2f, 0.05f, -0.05f};
    CHECK(add_tensor("x", input_shape, 2, input));
    CHECK(add_tensor("router.weight", router_w_shape, 2, router_w));
    CHECK(add_tensor("router.bias", router_b_shape, 1, router_b));
    CHECK(add_tensor("route.indices", route_shape, 2, NULL));
    CHECK(add_tensor("route.weights", route_shape, 2, NULL));
    CHECK(add_tensor("expert.weight", expert_w_shape, 3, expert_w));
    CHECK(add_tensor("expert.bias", expert_b_shape, 2, expert_b));
    CHECK(add_tensor("logits", input_shape, 2, NULL));

    Node* n = add_node("MoERouter", "route.indices",
                       "{\"num_experts\":3,\"top_k\":2,\"normalize\":1,\"temperature\":0.5}");
    CHECK(n && node_input(n, "input", "x") == 0 && node_input(n, "weight", "router.weight") == 0 &&
          node_input(n, "bias", "router.bias") == 0);
    Node* router_node = n;
    strncpy(n->outs[0].key, "indices", sizeof(n->outs[0].key) - 1);
    strncpy(n->outs[1].key, "weights", sizeof(n->outs[1].key) - 1);
    strncpy(n->outs[1].name, "route.weights", sizeof(n->outs[1].name) - 1);
    n->nout = 2;
    n = add_node("MoELinear", "logits", NULL);
    CHECK(n && node_input(n, "input", "x") == 0 && node_input(n, "expert_weight", "expert.weight") == 0 &&
          node_input(n, "expert_bias", "expert.bias") == 0 && node_input(n, "route_indices", "route.indices") == 0 &&
          node_input(n, "route_weights", "route.weights") == 0);

    CHECK(volvoxai_engine_forward() == 0);
    T* indices = t_find("route.indices");
    T* weights = t_find("route.weights");
    for (int token = 0; token < 2; token++) {
        CHECK(indices->data[token * 2] >= 0.0f && indices->data[token * 2] < 3.0f);
        CHECK(indices->data[token * 2 + 1] >= 0.0f && indices->data[token * 2 + 1] < 3.0f);
        CHECK(indices->data[token * 2] != indices->data[token * 2 + 1]);
        CHECK(fabsf(weights->data[token * 2] + weights->data[token * 2 + 1] - 1.0f) < 1e-5f);
    }
    CHECK(indices->data[0] == 0.0f && indices->data[1] == 2.0f);
    CHECK(indices->data[2] == 1.0f && indices->data[3] == 0.0f);
    CHECK(fabsf(weights->data[0] - 1.0f / (1.0f + expf(-0.9f))) < 1e-5f);
    CHECK(fabsf(weights->data[2] - 1.0f / (1.0f + expf(-1.26f))) < 1e-5f);
    float router_before[6], expert_before[12];
    memcpy(router_before, t_find("router.weight")->data, sizeof(router_before));
    memcpy(expert_before, t_find("expert.weight")->data, sizeof(expert_before));
    const int targets[2] = {0, 1};
    float* router_values = t_find("router.weight")->data;
    const float finite_difference_eps = 1.0e-3f;
    router_values[0] = router_before[0] + finite_difference_eps;
    CHECK(volvoxai_engine_forward() == 0);
    float loss_plus = cross_entropy_loss(t_find("logits"), targets, 2);
    router_values[0] = router_before[0] - finite_difference_eps;
    CHECK(volvoxai_engine_forward() == 0);
    float loss_minus = cross_entropy_loss(t_find("logits"), targets, 2);
    router_values[0] = router_before[0];
    float numeric_gradient = (loss_plus - loss_minus) / (2.0f * finite_difference_eps);
    CHECK(isfinite(numeric_gradient));
    const char* trainable[2] = {"router.weight", "expert.weight"};
    float loss = 0.0f;
    CHECK(volvoxai_engine_train_step("logits", targets, 2, INT_MIN, trainable, 2,
                                          2, 0.01f, 0.9f, 0.999f, 1e-8f,
                                          0.0f, 0.0f, 1, &loss, NULL, NULL) == 0);
    CHECK(isfinite(loss) && loss > 0.0f);
    float observed_gradient = (router_before[0] - t_find("router.weight")->data[0]) / 0.01f;
    CHECK(fabsf(observed_gradient - numeric_gradient) < 5.0e-3f * (1.0f + fabsf(numeric_gradient)));
    int router_changed = 0, expert_changed = 0;
    for (int i = 0; i < 6; i++) if (fabsf(t_find("router.weight")->data[i] - router_before[i]) > 1e-7f) router_changed = 1;
    for (int i = 0; i < 12; i++) if (fabsf(t_find("expert.weight")->data[i] - expert_before[i]) > 1e-7f) expert_changed = 1;
    CHECK(router_changed && expert_changed);

    memcpy(t_find("router.weight")->data, router_before, sizeof(router_before));
    cJSON_ReplaceItemInObject(router_node->params, "normalize", cJSON_CreateNumber(0.0));
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(weights->data[0] + weights->data[1] < 1.0f);
    router_values = t_find("router.weight")->data;
    router_values[0] = router_before[0] + finite_difference_eps;
    CHECK(volvoxai_engine_forward() == 0);
    loss_plus = cross_entropy_loss(t_find("logits"), targets, 2);
    router_values[0] = router_before[0] - finite_difference_eps;
    CHECK(volvoxai_engine_forward() == 0);
    loss_minus = cross_entropy_loss(t_find("logits"), targets, 2);
    router_values[0] = router_before[0];
    numeric_gradient = (loss_plus - loss_minus) / (2.0f * finite_difference_eps);
    const char* router_trainable[1] = {"router.weight"};
    CHECK(volvoxai_engine_train_step("logits", targets, 2, INT_MIN, router_trainable, 1,
                                     2, 0.01f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 2, &loss, NULL, NULL) == 0);
    observed_gradient = (router_before[0] - t_find("router.weight")->data[0]) / 0.01f;
    CHECK(fabsf(observed_gradient - numeric_gradient) < 5.0e-3f * (1.0f + fabsf(numeric_gradient)));

    cJSON_ReplaceItemInObject(router_node->params, "temperature", cJSON_CreateNumber(0.0));
    CHECK(volvoxai_engine_forward() == -1);
    cJSON* invalid_temperature = cJSON_GetObjectItem(router_node->params, "temperature");
    CHECK(invalid_temperature != NULL);
    invalid_temperature->valuedouble = NAN;
    CHECK(volvoxai_engine_forward() == -1);
    memcpy(t_find("router.weight")->data, router_before, sizeof(router_before));
    cJSON_ReplaceItemInObject(router_node->params, "normalize", cJSON_CreateNumber(1.0));
    cJSON_DeleteItemFromObject(router_node->params, "temperature");
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(indices->data[0] == 0.0f && indices->data[1] == 2.0f);
    CHECK(fabsf(weights->data[0] - 1.0f / (1.0f + expf(-0.45f))) < 1e-5f);
    return finish_graph();
}

static int test_prelu_logsoftmax_backward(void) {
    begin_graph();
    const int matrix_shape[2] = {2, 2};
    const int channel_shape[1] = {2};
    const float input[4] = {-1.0f, 2.0f, -0.5f, 1.5f};
    const float slope_values[2] = {0.2f, 0.4f};
    CHECK(add_tensor("x", matrix_shape, 2, input));
    CHECK(add_tensor("slope", channel_shape, 1, slope_values));
    CHECK(add_tensor("prelu", matrix_shape, 2, NULL));
    CHECK(add_tensor("logits", matrix_shape, 2, NULL));

    Node* n = add_node("PReLU", "prelu", NULL);
    CHECK(n && node_input(n, "input", "x") == 0 && node_input(n, "weight", "slope") == 0);
    n = add_node("LogSoftmax", "logits", "{\"axis\":-1}");
    CHECK(n && node_input(n, "input", "prelu") == 0);

    CHECK(volvoxai_engine_forward() == 0);
    T* logits = t_find("logits");
    for (int row = 0; row < 2; row++) {
        float probability_sum = expf(logits->data[row * 2]) + expf(logits->data[row * 2 + 1]);
        CHECK(fabsf(probability_sum - 1.0f) < 1.0e-6f);
    }

    const int targets[2] = {0, 1};
    const float epsilon = 1.0e-3f;
    float before[2];
    memcpy(before, t_find("slope")->data, sizeof(before));
    t_find("slope")->data[0] = before[0] + epsilon;
    CHECK(volvoxai_engine_forward() == 0);
    float loss_plus = cross_entropy_loss(t_find("logits"), targets, 2);
    t_find("slope")->data[0] = before[0] - epsilon;
    CHECK(volvoxai_engine_forward() == 0);
    float loss_minus = cross_entropy_loss(t_find("logits"), targets, 2);
    t_find("slope")->data[0] = before[0];
    float numeric_gradient = (loss_plus - loss_minus) / (2.0f * epsilon);

    const char* trainable[1] = {"slope"};
    const float learning_rate = 0.01f;
    CHECK(volvoxai_engine_train_step("logits", targets, 2, INT_MIN, trainable, 1,
                                     2, learning_rate, 0.9f, 0.999f, 1.0e-8f,
                                     0.0f, 0.0f, 1, NULL, NULL, NULL) == 0);
    float observed_gradient = (before[0] - t_find("slope")->data[0]) / learning_rate;
    CHECK(fabsf(observed_gradient - numeric_gradient) <
          5.0e-3f * (1.0f + fabsf(numeric_gradient)));
    CHECK(t_find("slope")->data[1] == before[1]);
    return finish_graph();
}

static int test_split_multi_output_backward(void) {
    begin_graph();
    const int input_shape[2] = {2, 4};
    const int output_shape[2] = {2, 2};
    const float input_values[8] = {
        0.2f, -0.4f, 0.7f, -0.1f,
        -0.3f, 0.8f, 0.1f, -0.6f
    };
    CHECK(add_tensor("parameter", input_shape, 2, input_values));
    CHECK(add_tensor("left", output_shape, 2, NULL));
    CHECK(add_tensor("right", output_shape, 2, NULL));
    CHECK(add_tensor("logits", output_shape, 2, NULL));

    Node* n = add_node("Split", "left", "{\"axis\":1}");
    CHECK(n && node_input(n, "input", "parameter") == 0);
    strncpy(n->outs[0].key, "left", sizeof(n->outs[0].key) - 1);
    strncpy(n->outs[1].key, "right", sizeof(n->outs[1].key) - 1);
    strncpy(n->outs[1].name, "right", sizeof(n->outs[1].name) - 1);
    n->nout = 2;
    n = add_node("Add", "logits", NULL);
    CHECK(n && node_input(n, "a", "left") == 0 && node_input(n, "b", "right") == 0);

    CHECK(volvoxai_engine_forward() == 0);
    CHECK(t_find("left")->data[0] == input_values[0] && t_find("left")->data[1] == input_values[1]);
    CHECK(t_find("right")->data[0] == input_values[2] && t_find("right")->data[1] == input_values[3]);
    CHECK(t_find("left")->data[2] == input_values[4] && t_find("left")->data[3] == input_values[5]);
    CHECK(t_find("right")->data[2] == input_values[6] && t_find("right")->data[3] == input_values[7]);

    const int target[2] = {1, 0};
    const float epsilon = 1.0e-3f;
    float before[8], numeric[8];
    memcpy(before, t_find("parameter")->data, sizeof(before));
    for (int index = 0; index < 8; index++) {
        t_find("parameter")->data[index] = before[index] + epsilon;
        CHECK(volvoxai_engine_forward() == 0);
        float loss_plus = cross_entropy_loss(t_find("logits"), target, 2);
        t_find("parameter")->data[index] = before[index] - epsilon;
        CHECK(volvoxai_engine_forward() == 0);
        float loss_minus = cross_entropy_loss(t_find("logits"), target, 2);
        t_find("parameter")->data[index] = before[index];
        numeric[index] = (loss_plus - loss_minus) / (2.0f * epsilon);
    }

    const char* trainable[1] = {"parameter"};
    const float learning_rate = 0.01f;
    CHECK(volvoxai_engine_train_step("logits", target, 2, INT_MIN, trainable, 1,
                                     2, learning_rate, 0.9f, 0.999f, 1.0e-8f,
                                     0.0f, 0.0f, 1, NULL, NULL, NULL) == 0);
    for (int index = 0; index < 8; index++) {
        float observed = (before[index] - t_find("parameter")->data[index]) / learning_rate;
        CHECK(fabsf(observed - numeric[index]) < 5.0e-3f * (1.0f + fabsf(numeric[index])));
    }
    return finish_graph();
}

static int test_secondary_output_gradient_rejected(void) {
    begin_graph();
    const int hidden_shape[2] = {1, 2};
    const int router_weight_shape[2] = {2, 2};
    const int route_shape[2] = {1, 1};
    const int route_projection_shape[2] = {1, 2};
    const float input[2] = {0.8f, -0.3f};
    const float router_weight[4] = {0.7f, -0.4f, -0.2f, 0.9f};
    const float route_projection[2] = {0.25f, -0.5f};
    const float direct_weight[4] = {0.6f, -0.1f, 0.2f, 0.7f};
    CHECK(add_tensor("x", hidden_shape, 2, input));
    CHECK(add_tensor("router.weight", router_weight_shape, 2, router_weight));
    CHECK(add_tensor("route.indices", route_shape, 2, NULL));
    CHECK(add_tensor("route.weights", route_shape, 2, NULL));
    CHECK(add_tensor("route_projection.weight", route_projection_shape, 2, route_projection));
    CHECK(add_tensor("routed_logits", hidden_shape, 2, NULL));
    CHECK(add_tensor("direct.weight", router_weight_shape, 2, direct_weight));
    CHECK(add_tensor("direct_logits", hidden_shape, 2, NULL));
    CHECK(add_tensor("logits", hidden_shape, 2, NULL));

    Node* n = add_node("MoERouter", "route.indices",
                       "{\"num_experts\":2,\"top_k\":1,\"normalize\":1}");
    CHECK(n && node_input(n, "input", "x") == 0 &&
          node_input(n, "weight", "router.weight") == 0);
    strncpy(n->outs[0].key, "indices", sizeof(n->outs[0].key) - 1);
    strncpy(n->outs[1].key, "weights", sizeof(n->outs[1].key) - 1);
    strncpy(n->outs[1].name, "route.weights", sizeof(n->outs[1].name) - 1);
    n->nout = 2;
    n = add_node("MatMul", "routed_logits", NULL);
    CHECK(n && node_input(n, "input", "route.indices") == 0 &&
          node_input(n, "weight", "route_projection.weight") == 0);
    n = add_node("MatMul", "direct_logits", NULL);
    CHECK(n && node_input(n, "input", "x") == 0 && node_input(n, "weight", "direct.weight") == 0);
    n = add_node("Add", "logits", NULL);
    CHECK(n && node_input(n, "a", "routed_logits") == 0 && node_input(n, "b", "direct_logits") == 0);

    CHECK(volvoxai_engine_forward() == 0);
    float before[4];
    memcpy(before, t_find("direct.weight")->data, sizeof(before));
    const int targets[1] = {1};
    const char* trainable[1] = {"direct.weight"};
    float loss = 0.0f;
    CHECK(volvoxai_engine_train_step("logits", targets, 1, INT_MIN, trainable, 1,
                                    2, 0.05f, 0.9f, 0.999f, 1e-8f,
                                    0.0f, 0.0f, 1, &loss, NULL, NULL) != 0);
    CHECK(memcmp(before, t_find("direct.weight")->data, sizeof(before)) == 0);
    return finish_graph();
}

static int test_train_base_route_and_prevalidation(void) {
    begin_graph();
    const int matrix_shape[2] = {2, 2};
    const int input_shape[2] = {1, 2};
    const int scalar_shape[1] = {1};
    const float x_values[2] = {1.0f, -0.5f};
    const float w_values[4] = {0.2f, -0.1f, 0.4f, 0.3f};
    const float unused_value[1] = {0.0f};
    CHECK(add_tensor("x", input_shape, 2, x_values));
    CHECK(add_tensor("w", matrix_shape, 2, w_values));
    CHECK(add_tensor("unused", scalar_shape, 1, unused_value));
    CHECK(add_tensor("logits", input_shape, 2, NULL));
    Node* n = add_node("MatMul", "logits", NULL);
    CHECK(n && node_input(n, "input", "x") == 0 && node_input(n, "weight", "w") == 0);

    const float adapter_a[2] = {1.0f, 1.0f};
    const float adapter_b[2] = {4.0f, -4.0f};
    VxAdapterTargetSpec adapter_target;
    memset(&adapter_target, 0, sizeof(adapter_target));
    adapter_target.weight_name = "w";
    adapter_target.kind = VX_ADAPTER_LORA;
    adapter_target.d_in = 2;
    adapter_target.d_out = 2;
    adapter_target.rank = 1;
    adapter_target.alpha = 1.0f;
    adapter_target.scale = 1.0f;
    adapter_target.a = (VxAdapterTensorSpec){adapter_a, sizeof(adapter_a), VX_ADAPTER_DTYPE_F32, 2, 1};
    adapter_target.b = (VxAdapterTensorSpec){adapter_b, sizeof(adapter_b), VX_ADAPTER_DTYPE_F32, 1, 2};
    VxAdapterVersionSpec adapter = {"train-adapter", "train-adapter", &adapter_target, 1, NULL};
    CHECK(vx_adapter_stage(&adapter) == 0 && vx_adapter_activate("train-adapter") == 0);

    const int target[1] = {0};
    const char* trainable[1] = {"w"};
    float before[4]; memcpy(before, t_find("w")->data, sizeof(before));
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, trainable, 1,
                                     2, 0.1f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 1, NULL, NULL, NULL) == 0);
    float p1 = expf(-0.25f) / (1.0f + expf(-0.25f));
    const float expected_gradient[4] = {-p1, p1, 0.5f * p1, -0.5f * p1};
    for (int i = 0; i < 4; i++) {
        CHECK(fabsf(t_find("w")->data[i] - (before[i] - 0.1f * expected_gradient[i])) < 1e-5f);
    }
    char active[128];
    CHECK(vx_adapter_get_active(active, sizeof(active)) == 0 && !strcmp(active, "train-adapter"));

    memcpy(t_find("w")->data, before, sizeof(before));
    const char* duplicate[2] = {"w", "w"};
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, duplicate, 2,
                                     2, 0.1f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 1, NULL, NULL, NULL) == -1);
    CHECK(memcmp(t_find("w")->data, before, sizeof(before)) == 0);
    const char* missing[2] = {"w", "missing"};
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, missing, 2,
                                     2, 0.1f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 1, NULL, NULL, NULL) == -1);
    CHECK(memcmp(t_find("w")->data, before, sizeof(before)) == 0);
    const char* missing_gradient[2] = {"w", "unused"};
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, missing_gradient, 2,
                                     2, 0.1f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 1, NULL, NULL, NULL) == -1);
    CHECK(memcmp(t_find("w")->data, before, sizeof(before)) == 0);

    CHECK(volvoxai_engine_adapter_route_begin("train-adapter") == 0);
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, trainable, 1,
                                     2, 0.1f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 1, NULL, NULL, NULL) == -1);
    CHECK(memcmp(t_find("w")->data, before, sizeof(before)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(t_find("logits")->data[0] > 1.9f && t_find("logits")->data[1] < -2.0f);
    volvoxai_engine_adapter_route_end();
    CHECK(vx_adapter_get_active(active, sizeof(active)) == 0 && !strcmp(active, "train-adapter"));
    return finish_graph();
}

static int test_explicit_graph_lora_training(void) {
    begin_graph();
    const int input_shape[2] = {1, 2};
    const int base_shape[2] = {2, 2};
    const int a_shape[2] = {2, 1};
    const int b_shape[2] = {1, 2};
    const int hidden_shape[2] = {1, 1};
    const float x[2] = {0.7f, -0.2f};
    const float base_w[4] = {0.2f, -0.1f, 0.3f, 0.4f};
    const float a[2] = {0.5f, -0.3f};
    const float b[2] = {0.2f, -0.4f};
    CHECK(add_tensor("x", input_shape, 2, x));
    CHECK(add_tensor("base.weight", base_shape, 2, base_w));
    CHECK(add_tensor("lora.a", a_shape, 2, a));
    CHECK(add_tensor("lora.b", b_shape, 2, b));
    CHECK(add_tensor("base", input_shape, 2, NULL));
    CHECK(add_tensor("lora.hidden", hidden_shape, 2, NULL));
    CHECK(add_tensor("lora.out", input_shape, 2, NULL));
    CHECK(add_tensor("logits", input_shape, 2, NULL));
    Node* n = add_node("MatMul", "base", NULL);
    CHECK(n && node_input(n, "input", "x") == 0 && node_input(n, "weight", "base.weight") == 0);
    n = add_node("MatMul", "lora.hidden", NULL);
    CHECK(n && node_input(n, "input", "x") == 0 && node_input(n, "weight", "lora.a") == 0);
    n = add_node("MatMul", "lora.out", NULL);
    CHECK(n && node_input(n, "input", "lora.hidden") == 0 && node_input(n, "weight", "lora.b") == 0);
    n = add_node("Add", "logits", NULL);
    CHECK(n && node_input(n, "a", "base") == 0 && node_input(n, "b", "lora.out") == 0);
    float a_before[2], b_before[2];
    memcpy(a_before, t_find("lora.a")->data, sizeof(a_before));
    memcpy(b_before, t_find("lora.b")->data, sizeof(b_before));
    const int target[1] = {1};
    const char* trainable[2] = {"lora.a", "lora.b"};
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, trainable, 2,
                                     2, 0.1f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 1, NULL, NULL, NULL) == 0);
    CHECK(memcmp(t_find("lora.a")->data, a_before, sizeof(a_before)) != 0);
    CHECK(memcmp(t_find("lora.b")->data, b_before, sizeof(b_before)) != 0);
    return finish_graph();
}

static int test_attention_masks_and_ignored_targets(void) {
    begin_graph();
    g_use_vulkan = g_use_nnapi = g_use_opengl = g_use_metal = 0;
    const int qkv_shape[3] = {1, 2, 3};
    const int out_shape[3] = {1, 2, 1};
    const int mask_shape[3] = {1, 2, 2};
    const float qkv[6] = {
        1.0f, 0.0f, 1.0f,
        1.0f, 10.0f, 9.0f,
    };
    const int32_t mask[4] = {1, 0, 0, 1};
    CHECK(add_tensor("qkv", qkv_shape, 3, qkv));
    CHECK(add_i32_tensor("mask", mask_shape, 3, mask));
    CHECK(add_tensor("full", out_shape, 3, NULL));
    CHECK(add_tensor("masked", out_shape, 3, NULL));
    Node* full = add_node("SDPA", "full", "{\"heads\":1,\"scale\":1,\"causal\":false}");
    CHECK(full && node_input(full, "qkv", "qkv") == 0);
    Node* masked = add_node("SDPA", "masked", "{\"heads\":1,\"scale\":1,\"causal\":false}");
    CHECK(masked && node_input(masked, "qkv", "qkv") == 0 && node_input(masked, "mask", "mask") == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(t_find("full")->data[0] > 8.5f); /* Encoder query zero sees the future key. */
    CHECK(fabsf(t_find("masked")->data[0] - 1.0f) < 1e-6f);
    CHECK(fabsf(t_find("masked")->data[1] - 9.0f) < 1e-6f);
    volvoxai_engine_shutdown();

    begin_graph();
    const int matrix_shape[2] = {2, 2};
    const float x[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float w[4] = {0.2f, -0.1f, 0.3f, 0.4f};
    CHECK(add_tensor("x", matrix_shape, 2, x));
    CHECK(add_tensor("w", matrix_shape, 2, w));
    CHECK(add_tensor("logits", matrix_shape, 2, NULL));
    Node* projection = add_node("MatMul", "logits", NULL);
    CHECK(projection && node_input(projection, "input", "x") == 0 &&
          node_input(projection, "weight", "w") == 0);
    const int targets[2] = {1, -100};
    const char* trainable[1] = {"w"};
    float before[4]; memcpy(before, w, sizeof(before));
    int examples = 0;
    CHECK(volvoxai_engine_train_step("logits", targets, 2, -100, trainable, 1,
                                     2, 0.1f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 1, NULL, NULL, &examples) == 0);
    CHECK(examples == 1);
    CHECK(fabsf(t_find("w")->data[0] - before[0]) > 1e-7f ||
          fabsf(t_find("w")->data[1] - before[1]) > 1e-7f);
    CHECK(fabsf(t_find("w")->data[2] - before[2]) < 1e-7f &&
          fabsf(t_find("w")->data[3] - before[3]) < 1e-7f);
    float after[4]; memcpy(after, t_find("w")->data, sizeof(after));
    const int ignored[2] = {-100, -100};
    examples = -1;
    CHECK(volvoxai_engine_train_step("logits", ignored, 2, -100, trainable, 1,
                                     2, 0.1f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 2, NULL, NULL, &examples) == 0);
    CHECK(examples == 0 && memcmp(after, t_find("w")->data, sizeof(after)) == 0);
    return finish_graph();
}

static int test_native_seq2seq_training(void) {
    begin_graph();
    g_use_vulkan = g_use_nnapi = g_use_opengl = g_use_metal = 0;
    const int ids_shape[2] = {1, 2};
    const int hidden_shape[3] = {1, 2, 2};
    const int qkv_shape[3] = {1, 2, 6};
    const int table_shape[2] = {4, 2};
    const int qkv_weight_shape[2] = {2, 6};
    const int logits_shape[3] = {1, 2, 4};
    const int lm_shape[2] = {2, 4};
    const int32_t source_ids[2] = {2, 0};
    const int32_t decoder_ids[2] = {1, 3};
    /* Keep two encoder keys live so CrossSDPA remains query-dependent and the
       decoder embedding branch receives a non-zero training gradient. */
    const int32_t source_mask[2] = {1, 1};
    const int32_t decoder_mask[2] = {1, 1};
    const float source_table[8] = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.7f, 0.2f, 0.6f};
    const float target_table[8] = {-0.2f, 0.3f, 0.8f, -0.4f, 0.1f, 0.5f, -0.6f, 0.2f};
    const float encoder_qkv_weight[12] = {
        0.2f, -0.1f, 0.3f, 0.5f, -0.4f, 0.7f,
       -0.3f,  0.6f, 0.4f, 0.1f,  0.8f, -0.2f,
    };
    const float decoder_qkv_weight[12] = {
       -0.1f, 0.4f, 0.5f, -0.2f, 0.6f, 0.3f,
        0.7f, 0.2f, -0.4f, 0.8f, 0.1f, -0.5f,
    };
    const float lm_weight[8] = {0.4f, -0.2f, 0.1f, 0.3f, -0.5f, 0.7f, 0.6f, -0.1f};
    CHECK(add_i32_tensor("source.ids", ids_shape, 2, source_ids));
    CHECK(add_i32_tensor("decoder.ids", ids_shape, 2, decoder_ids));
    CHECK(add_i32_tensor("source.mask", ids_shape, 2, source_mask));
    CHECK(add_i32_tensor("decoder.mask", ids_shape, 2, decoder_mask));
    CHECK(add_tensor("source.table", table_shape, 2, source_table));
    CHECK(add_tensor("target.table", table_shape, 2, target_table));
    CHECK(add_tensor("source.emb", hidden_shape, 3, NULL));
    CHECK(add_tensor("target.emb", hidden_shape, 3, NULL));
    CHECK(add_tensor("encoder.qkv.weight", qkv_weight_shape, 2, encoder_qkv_weight));
    CHECK(add_tensor("decoder.qkv.weight", qkv_weight_shape, 2, decoder_qkv_weight));
    CHECK(add_tensor("encoder.qkv", qkv_shape, 3, NULL));
    CHECK(add_tensor("decoder.qkv", qkv_shape, 3, NULL));
    CHECK(add_tensor("memory", hidden_shape, 3, NULL));
    CHECK(add_tensor("decoder.self", hidden_shape, 3, NULL));
    CHECK(add_tensor("cross", hidden_shape, 3, NULL));
    CHECK(add_tensor("lm.weight", lm_shape, 2, lm_weight));
    CHECK(add_tensor("logits", logits_shape, 3, NULL));

    Node* node = add_node("Embedding", "source.emb", NULL);
    CHECK(node && node_input(node, "input", "source.ids") == 0 && node_input(node, "weight", "source.table") == 0);
    node = add_node("MatMul", "encoder.qkv", NULL);
    CHECK(node && node_input(node, "input", "source.emb") == 0 && node_input(node, "weight", "encoder.qkv.weight") == 0);
    node = add_node("SDPA", "memory", "{\"heads\":1,\"causal\":false}");
    CHECK(node && node_input(node, "qkv", "encoder.qkv") == 0 && node_input(node, "mask", "source.mask") == 0);
    node = add_node("Embedding", "target.emb", NULL);
    CHECK(node && node_input(node, "input", "decoder.ids") == 0 && node_input(node, "weight", "target.table") == 0);
    node = add_node("MatMul", "decoder.qkv", NULL);
    CHECK(node && node_input(node, "input", "target.emb") == 0 && node_input(node, "weight", "decoder.qkv.weight") == 0);
    node = add_node("SDPA", "decoder.self", "{\"heads\":1,\"causal\":true}");
    CHECK(node && node_input(node, "qkv", "decoder.qkv") == 0 && node_input(node, "mask", "decoder.mask") == 0);
    node = add_node("CrossSDPA", "cross", "{\"heads\":1,\"causal\":false}");
    CHECK(node && node_input(node, "q", "decoder.self") == 0 && node_input(node, "k", "memory") == 0 &&
          node_input(node, "v", "memory") == 0 && node_input(node, "mask", "source.mask") == 0);
    node = add_node("MatMul", "logits", NULL);
    CHECK(node && node_input(node, "input", "cross") == 0 && node_input(node, "weight", "lm.weight") == 0);

    float source_before[8], target_before[8];
    memcpy(source_before, source_table, sizeof(source_before));
    memcpy(target_before, target_table, sizeof(target_before));
    const int labels[2] = {3, -100};
    const char* trainable[5] = {
        "source.table", "target.table", "encoder.qkv.weight", "decoder.qkv.weight", "lm.weight",
    };
    float loss = 0.0f;
    int examples = 0;
    CHECK(volvoxai_engine_train_step("logits", labels, 2, -100, trainable, 5,
                                     3, 0.005f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 1.0f, 1, &loss, NULL, &examples) == 0);
    CHECK(isfinite(loss) && examples == 1);
    CHECK(memcmp(source_before, t_find("source.table")->data, sizeof(source_before)) != 0);
    CHECK(memcmp(target_before, t_find("target.table")->data, sizeof(target_before)) != 0);
    return finish_graph();
}

static int test_fused_graph_training(void) {
    const char* config_path = "/tmp/volvox-training-fused.json";
    const char* weights_path = "/tmp/volvox-training-fused.safetensors";
    const char* config =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,1]}},\"nodes\":["
        "{\"opType\":\"Conv2D\",\"inputs\":{\"input\":\"x\",\"weight\":\"conv.weight\"},"
        "\"outputs\":{\"output\":\"conv\"},\"outputs_shape\":{\"output\":[1,1,1,2]},"
        "\"params\":{\"weight_layout\":\"HWIO\"}},"
        "{\"opType\":\"Clip\",\"inputs\":{\"input\":\"conv\"},"
        "\"outputs\":{\"output\":\"clipped\"},\"outputs_shape\":{\"output\":[1,1,1,2]},"
        "\"params\":{\"min\":0,\"max\":6}},"
        "{\"opType\":\"GlobalAveragePool\",\"inputs\":{\"input\":\"clipped\"},"
        "\"outputs\":{\"output\":\"gap\"},\"outputs_shape\":{\"output\":[1,2]}},"
        "{\"opType\":\"MatMul\",\"inputs\":{\"input\":\"gap\",\"weight\":\"head.weight\"},"
        "\"outputs\":{\"output\":\"logits\"},\"outputs_shape\":{\"output\":[1,2]}}]}";
    FILE* f = fopen(config_path, "wb");
    CHECK(f && fwrite(config, 1, strlen(config), f) == strlen(config) && fclose(f) == 0);
    const int conv_shape[4] = {1, 1, 1, 2};
    const int head_shape[2] = {2, 2};
    const float conv_weight[2] = {0.5f, 1.0f};
    const float head_weight[4] = {0.7f, -0.2f, 0.1f, 0.8f};
    SafetensorsFile weights;
    CHECK(safetensors_init_empty(&weights, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&weights, "conv.weight", SAFETENSORS_DTYPE_F32,
                                 conv_shape, 4, conv_weight, sizeof(conv_weight)) == 0);
    CHECK(safetensors_add_tensor(&weights, "head.weight", SAFETENSORS_DTYPE_F32,
                                 head_shape, 2, head_weight, sizeof(head_weight)) == 0);
    CHECK(safetensors_save(weights_path, &weights) == 0);
    safetensors_free(&weights);
    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    CHECK(g_nn == 4 && g_n[0].fuse_relu6 == 1 && g_n[1].skip == 1 && !strcmp(g_n[0].out, "clipped"));
    long count = 0;
    float* x = volvoxai_engine_input_ptr("x", &count);
    CHECK(x && count == 1);
    x[0] = 1.5f;
    float before[2]; memcpy(before, t_find("conv.weight")->data, sizeof(before));
    const int target[1] = {1};
    const char* trainable[1] = {"conv.weight"};
    const char* input_trainable[1] = {"x"};
    float loss = 0.0f;
    CHECK(volvoxai_engine_is_model_weight("conv.weight") == 1);
    CHECK(volvoxai_engine_is_model_weight("x") == 0);
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, input_trainable, 1,
                                     3, 0.05f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 1, &loss, NULL, NULL) == -1);
    const float input_update[1] = {0.25f};
    CHECK(volvoxai_engine_apply_tensor_update_f32("x", input_update, 1,
                                                  3, 0.05f, 0.9f, 0.999f, 1e-8f,
                                                  0.0f, 0.0f, 1) == -1);
    CHECK(volvoxai_engine_apply_tensor_update_f32("x", input_update, 1,
                                                  1, 0.05f, 0.9f, 0.999f, 1e-8f,
                                                  0.0f, 0.0f, 1) == -1);
    CHECK(volvoxai_engine_set_tensor_f32("x", input_update, 1) == -1);
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, trainable, 1,
                                          2, 0.05f, 0.9f, 0.999f, 1e-8f,
                                          0.0f, 0.0f, 1, &loss, NULL, NULL) == 0);
    CHECK(isfinite(loss) && (fabsf(t_find("conv.weight")->data[0] - before[0]) > 1e-7f ||
                             fabsf(t_find("conv.weight")->data[1] - before[1]) > 1e-7f));
    CHECK(g_n[0].fuse_relu6 == 1 && g_n[1].skip == 1 && !strcmp(g_n[0].out, "clipped"));
    CHECK(volvoxai_engine_forward() == 0);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int test_optimized_dropout_alias_training(void) {
    const char* config_path = "/tmp/volvox-training-dropout-alias.json";
    const char* weights_path = "/tmp/volvox-training-dropout-alias.safetensors";
    const char* config =
        "{\"inputs\":{},\"nodes\":["
        "{\"opType\":\"Dropout\",\"inputs\":{\"input\":\"features\"},"
        "\"outputs\":{\"output\":\"dropped\"},\"outputs_shape\":{\"output\":[1,64]},"
        "\"params\":{\"p\":0.5,\"seed\":12345}},"
        "{\"opType\":\"Reshape\",\"inputs\":{\"input\":\"dropped\"},"
        "\"outputs\":{\"output\":\"reshaped\"},\"outputs_shape\":{\"output\":[1,64]}},"
        "{\"opType\":\"MatMul\",\"inputs\":{\"input\":\"reshaped\","
        "\"weight\":\"head.weight\"},\"outputs\":{\"output\":\"logits\"},"
        "\"outputs_shape\":{\"output\":[1,2]}}]}";
    FILE* file = fopen(config_path, "wb");
    CHECK(file && fwrite(config, 1, strlen(config), file) == strlen(config) && fclose(file) == 0);
    const int feature_shape[2] = {1, 64};
    const int head_shape[2] = {64, 2};
    float features[64], head[128];
    for (int i = 0; i < 64; i++) {
        features[i] = 1.0f;
        head[i * 2] = (i & 1) ? 0.01f : -0.01f;
        head[i * 2 + 1] = -head[i * 2];
    }
    SafetensorsFile weights;
    CHECK(safetensors_init_empty(&weights, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&weights, "features", SAFETENSORS_DTYPE_F32,
                                 feature_shape, 2, features, sizeof(features)) == 0);
    CHECK(safetensors_add_tensor(&weights, "head.weight", SAFETENSORS_DTYPE_F32,
                                 head_shape, 2, head, sizeof(head)) == 0);
    CHECK(safetensors_save(weights_path, &weights) == 0);
    safetensors_free(&weights);

    g_use_vulkan = g_use_nnapi = g_use_opengl = g_use_metal = 0;
    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* feature_tensor = t_find("features");
    T* dropped_tensor = t_find("dropped");
    T* reshaped_tensor = t_find("reshaped");
    CHECK(g_nn == 3 && g_n[0].skip == 1 && g_n[1].skip == 1 &&
          feature_tensor && dropped_tensor && reshaped_tensor);
    CHECK(dropped_tensor->data == feature_tensor->data && dropped_tensor->owns == 0);
    CHECK(reshaped_tensor->data == feature_tensor->data && reshaped_tensor->owns == 0);
    CHECK(volvoxai_engine_forward() == 0);
    float logits_before[2];
    memcpy(logits_before, t_find("logits")->data, sizeof(logits_before));
    float features_before[64];
    memcpy(features_before, feature_tensor->data, sizeof(features_before));

    const int target[1] = {1};
    const char* trainable[1] = {"features"};
    float loss = 0.0f;
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, trainable, 1,
                                     2, 0.0f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 7, &loss, NULL, NULL) == 0);
    CHECK(isfinite(loss));
    CHECK(memcmp(features_before, feature_tensor->data, sizeof(features_before)) == 0);
    CHECK(g_n[0].skip == 1 && g_n[1].skip == 1 &&
          dropped_tensor->data == feature_tensor->data && dropped_tensor->owns == 0 &&
          reshaped_tensor->data == feature_tensor->data && reshaped_tensor->owns == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(memcmp(logits_before, t_find("logits")->data, sizeof(logits_before)) == 0);

    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int test_groupnorm_training(void) {
    begin_graph();
    g_use_vulkan = g_use_nnapi = g_use_opengl = g_use_metal = 0;
    const int image_shape[4] = {1, 2, 2, 4};
    const int channel_shape[1] = {4};
    const int pooled_shape[2] = {1, 4};
    const int head_shape[2] = {4, 2};
    const int logits_shape[2] = {1, 2};
    const float image[16] = {
        -1.2f, 0.4f, 1.1f, -0.7f, 0.3f, 1.4f, -0.2f, 0.8f,
         1.7f, -0.5f, 0.6f, 1.9f, -0.9f, 0.2f, -1.5f, 0.1f,
    };
    const float weight[4] = {1.0f, 0.8f, 1.2f, 0.6f};
    const float bias[4] = {0.1f, -0.2f, 0.3f, 0.05f};
    const float head[8] = {0.5f, -0.3f, -0.2f, 0.7f, 0.9f, -0.4f, -0.6f, 0.8f};
    CHECK(add_tensor("image", image_shape, 4, image));
    CHECK(add_tensor("group.weight", channel_shape, 1, weight));
    CHECK(add_tensor("group.bias", channel_shape, 1, bias));
    CHECK(add_tensor("normalized", image_shape, 4, NULL));
    CHECK(add_tensor("pooled", pooled_shape, 2, NULL));
    CHECK(add_tensor("head.weight", head_shape, 2, head));
    CHECK(add_tensor("logits", logits_shape, 2, NULL));
    Node* node = add_node("GroupNorm", "normalized", "{\"num_groups\":2,\"eps\":0.0001}");
    CHECK(node && node_input(node, "input", "image") == 0 &&
          node_input(node, "weight", "group.weight") == 0 &&
          node_input(node, "bias", "group.bias") == 0);
    node = add_node("GlobalAveragePool", "pooled", NULL);
    CHECK(node && node_input(node, "input", "normalized") == 0);
    node = add_node("MatMul", "logits", NULL);
    CHECK(node && node_input(node, "input", "pooled") == 0 &&
          node_input(node, "weight", "head.weight") == 0);

    CHECK(volvoxai_engine_forward() == 0);
    float expected[16];
    for (int group = 0; group < 2; group++) {
        double sum = 0.0, square_sum = 0.0;
        for (int spatial = 0; spatial < 4; spatial++) {
            for (int local = 0; local < 2; local++) {
                float value = image[spatial * 4 + group * 2 + local];
                sum += value;
                square_sum += (double)value * value;
            }
        }
        float mean = (float)(sum / 8.0);
        float inverse_stddev = 1.0f / sqrtf((float)(square_sum / 8.0 - mean * mean) + 0.0001f);
        for (int spatial = 0; spatial < 4; spatial++) {
            for (int local = 0; local < 2; local++) {
                int channel = group * 2 + local;
                int index = spatial * 4 + channel;
                expected[index] = (image[index] - mean) * inverse_stddev * weight[channel] + bias[channel];
            }
        }
    }
    for (int i = 0; i < 16; i++)
        CHECK(fabsf(t_find("normalized")->data[i] - expected[i]) < 2.0e-5f);

    float image_before[16], weight_before[4], bias_before[4];
    memcpy(image_before, t_find("image")->data, sizeof(image_before));
    memcpy(weight_before, t_find("group.weight")->data, sizeof(weight_before));
    memcpy(bias_before, t_find("group.bias")->data, sizeof(bias_before));
    const char* trainables[3] = {"image", "group.weight", "group.bias"};
    const int target[1] = {1};
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, trainables, 3,
                                     2, 0.01f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 1, NULL, NULL, NULL) == 0);
    CHECK(memcmp(image_before, t_find("image")->data, sizeof(image_before)) != 0);
    CHECK(memcmp(weight_before, t_find("group.weight")->data, sizeof(weight_before)) != 0);
    CHECK(memcmp(bias_before, t_find("group.bias")->data, sizeof(bias_before)) != 0);
    return finish_graph();
}

static int test_dropout_training_and_inference(void) {
    begin_graph();
    g_use_vulkan = g_use_nnapi = g_use_opengl = g_use_metal = 0;
    const int feature_shape[2] = {1, 64};
    const int head_shape[2] = {64, 2};
    const int logits_shape[2] = {1, 2};
    float features[64], head[128];
    for (int i = 0; i < 64; i++) {
        features[i] = 1.0f;
        head[i * 2] = (i & 1) ? 0.01f : -0.01f;
        head[i * 2 + 1] = -head[i * 2];
    }
    CHECK(add_tensor("features", feature_shape, 2, features));
    CHECK(add_tensor("dropped", feature_shape, 2, NULL));
    CHECK(add_tensor("head.weight", head_shape, 2, head));
    CHECK(add_tensor("logits", logits_shape, 2, NULL));
    Node* node = add_node("Dropout", "dropped", "{\"p\":0.5,\"seed\":12345}");
    CHECK(node && node_input(node, "input", "features") == 0);
    node = add_node("MatMul", "logits", NULL);
    CHECK(node && node_input(node, "input", "dropped") == 0 &&
          node_input(node, "weight", "head.weight") == 0);

    CHECK(volvoxai_engine_forward() == 0);
    for (int i = 0; i < 64; i++) CHECK(t_find("dropped")->data[i] == features[i]);

    const int target[1] = {1};
    const char* trainable[1] = {"features"};
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, trainable, 1,
                                     2, 0.001f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 7, NULL, NULL, NULL) == 0);
    float first_update[64];
    memcpy(first_update, t_find("features")->data, sizeof(first_update));
    int kept = 0, dropped = 0;
    for (int i = 0; i < 64; i++) {
        if (fabsf(first_update[i] - 1.0f) > 1.0e-8f) {
            kept++;
        } else dropped++;
    }
    CHECK(kept > 0 && dropped > 0);

    memcpy(t_find("features")->data, features, sizeof(features));
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, trainable, 1,
                                     2, 0.001f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 7, NULL, NULL, NULL) == 0);
    CHECK(memcmp(first_update, t_find("features")->data, sizeof(first_update)) == 0);

    memcpy(t_find("features")->data, features, sizeof(features));
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, trainable, 1,
                                     2, 0.001f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 8, NULL, NULL, NULL) == 0);
    CHECK(memcmp(first_update, t_find("features")->data, sizeof(first_update)) != 0);

    memcpy(t_find("features")->data, features, sizeof(features));
    CHECK(run_node(&g_n[0], 0, 0) == 0);
    CHECK(memcmp(t_find("features")->data, t_find("dropped")->data, sizeof(features)) == 0);
    return finish_graph();
}

static int test_attention_dropout_training_and_inference(void) {
    const int qkv_shape[2] = {2, 6};
    const int attention_shape[2] = {2, 2};
    const float qkv_values[12] = {
        0.4f, -0.2f, 0.1f, 0.5f, 0.7f, -0.3f,
        -0.1f, 0.6f, 0.3f, -0.4f, 0.2f, 0.8f
    };
    const float expected_inference[4] = {
        0.4270835221f, 0.3004162014f, 0.4988606870f, 0.1425064802f
    };
    const float expected_qkv_gradient[12] = {
         0.0100736711f, -0.0453315191f,
        -0.0240402836f,  0.0334313251f,
        -0.1305007488f,  0.1305007488f,
        -0.0077858847f,  0.0350364782f,
         0.0240402836f, -0.0334313251f,
         0.1535121948f, -0.1535121948f
    };
    const int targets[2] = {0, 1};
    const char* qkv_trainable[1] = {"qkv"};
    const float learning_rate = 0.1f;

    begin_graph();
    g_use_vulkan = g_use_nnapi = g_use_opengl = g_use_metal = 0;
    CHECK(add_tensor("qkv", qkv_shape, 2, qkv_values));
    CHECK(add_tensor("logits", attention_shape, 2, NULL));
    Node* node = add_node("SDPA", "logits",
        "{\"heads\":1,\"scale\":0.7071067811865475,\"causal\":false,"
        "\"dropout\":0.5,\"dropout_seed\":18}");
    CHECK(node && node_input(node, "qkv", "qkv") == 0);
    CHECK(volvoxai_engine_forward() == 0);
    for (int i = 0; i < 4; i++)
        CHECK(fabsf(t_find("logits")->data[i] - expected_inference[i]) < 2.0e-5f);

    CHECK(volvoxai_engine_train_step("logits", targets, 2, INT_MIN, qkv_trainable, 1,
                                     2, learning_rate, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 7, NULL, NULL, NULL) == 0);
    float first_update[12];
    memcpy(first_update, t_find("qkv")->data, sizeof(first_update));
    for (int i = 0; i < 12; i++) {
        float gradient = (qkv_values[i] - first_update[i]) / learning_rate;
        CHECK(fabsf(gradient - expected_qkv_gradient[i]) < 3.0e-4f);
    }

    memcpy(t_find("qkv")->data, qkv_values, sizeof(qkv_values));
    CHECK(volvoxai_engine_train_step("logits", targets, 2, INT_MIN, qkv_trainable, 1,
                                     2, learning_rate, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 7, NULL, NULL, NULL) == 0);
    CHECK(memcmp(first_update, t_find("qkv")->data, sizeof(first_update)) == 0);
    memcpy(t_find("qkv")->data, qkv_values, sizeof(qkv_values));
    CHECK(volvoxai_engine_train_step("logits", targets, 2, INT_MIN, qkv_trainable, 1,
                                     2, learning_rate, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 8, NULL, NULL, NULL) == 0);
    CHECK(memcmp(first_update, t_find("qkv")->data, sizeof(first_update)) != 0);
    CHECK(finish_graph() == 0);

    const int q_shape[2] = {2, 2};
    const float q_values[4] = {0.4f, -0.2f, -0.1f, 0.6f};
    const float k_values[4] = {0.1f, 0.5f, 0.3f, -0.4f};
    const float v_values[4] = {0.7f, -0.3f, 0.2f, 0.8f};
    const float expected_q_gradient[4] = {
        0.0100736711f, -0.0453315191f, -0.0077858847f, 0.0350364782f
    };
    const float expected_k_gradient[4] = {
        -0.0240402836f, 0.0334313251f, 0.0240402836f, -0.0334313251f
    };
    const float expected_v_gradient[4] = {
        -0.1305007488f, 0.1305007488f, 0.1535121948f, -0.1535121948f
    };
    const char* cross_trainables[3] = {"q", "k", "v"};
    begin_graph();
    g_use_vulkan = g_use_nnapi = g_use_opengl = g_use_metal = 0;
    CHECK(add_tensor("q", q_shape, 2, q_values));
    CHECK(add_tensor("k", q_shape, 2, k_values));
    CHECK(add_tensor("v", q_shape, 2, v_values));
    CHECK(add_tensor("cross", attention_shape, 2, NULL));
    node = add_node("CrossSDPA", "cross",
        "{\"heads\":1,\"scale\":0.7071067811865475,\"causal\":false,"
        "\"attention_dropout\":0.5,\"training_seed\":18}");
    CHECK(node && node_input(node, "q", "q") == 0 && node_input(node, "k", "k") == 0 &&
          node_input(node, "v", "v") == 0);
    CHECK(volvoxai_engine_forward() == 0);
    for (int i = 0; i < 4; i++)
        CHECK(fabsf(t_find("cross")->data[i] - expected_inference[i]) < 2.0e-2f);
    CHECK(volvoxai_engine_train_step("cross", targets, 2, INT_MIN, cross_trainables, 3,
                                     2, learning_rate, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 7, NULL, NULL, NULL) == 0);
    for (int i = 0; i < 4; i++) {
        float q_gradient = (q_values[i] - t_find("q")->data[i]) / learning_rate;
        float k_gradient = (k_values[i] - t_find("k")->data[i]) / learning_rate;
        float v_gradient = (v_values[i] - t_find("v")->data[i]) / learning_rate;
        CHECK(fabsf(q_gradient - expected_q_gradient[i]) < 3.0e-4f);
        CHECK(fabsf(k_gradient - expected_k_gradient[i]) < 3.0e-4f);
        CHECK(fabsf(v_gradient - expected_v_gradient[i]) < 3.0e-4f);
    }
    return finish_graph();
}

static int test_train_step_global_gradient_clipping(void) {
    begin_graph();
    g_use_vulkan = g_use_nnapi = g_use_opengl = g_use_metal = 0;
    const int input_shape[2] = {1, 1};
    const int weight_shape[2] = {1, 2};
    const int logits_shape[2] = {1, 2};
    const float x1[1] = {1.0f};
    const float x2[1] = {2.0f};
    const float zeros[2] = {0.0f, 0.0f};
    CHECK(add_tensor("x1", input_shape, 2, x1));
    CHECK(add_tensor("x2", input_shape, 2, x2));
    CHECK(add_tensor("w1", weight_shape, 2, zeros));
    CHECK(add_tensor("w2", weight_shape, 2, zeros));
    CHECK(add_tensor("branch1", logits_shape, 2, NULL));
    CHECK(add_tensor("branch2", logits_shape, 2, NULL));
    CHECK(add_tensor("logits", logits_shape, 2, NULL));
    Node* node = add_node("MatMul", "branch1", NULL);
    CHECK(node && node_input(node, "input", "x1") == 0 && node_input(node, "weight", "w1") == 0);
    node = add_node("MatMul", "branch2", NULL);
    CHECK(node && node_input(node, "input", "x2") == 0 && node_input(node, "weight", "w2") == 0);
    node = add_node("Add", "logits", NULL);
    CHECK(node && node_input(node, "a", "branch1") == 0 && node_input(node, "b", "branch2") == 0);
    const int target[1] = {0};
    const char* trainables[2] = {"w1", "w2"};
    CHECK(volvoxai_engine_train_step("logits", target, 1, INT_MIN, trainables, 2,
                                     2, 1.0f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 1.0f, 1, NULL, NULL, NULL) == 0);
    float scale = 1.0f / sqrtf(2.5f);
    CHECK(fabsf(t_find("w1")->data[0] - 0.5f * scale) < 1.0e-6f);
    CHECK(fabsf(t_find("w1")->data[1] + 0.5f * scale) < 1.0e-6f);
    CHECK(fabsf(t_find("w2")->data[0] - scale) < 1.0e-6f);
    CHECK(fabsf(t_find("w2")->data[1] + scale) < 1.0e-6f);
    return finish_graph();
}

static int test_shape_aware_broadcast_and_reduction_backward(void) {
    begin_graph();
    g_use_vulkan = g_use_nnapi = g_use_opengl = g_use_metal = 0;
    const int output_shape[3] = {2, 3, 2};
    const int batch_mask_shape[3] = {2, 1, 2};
    const int position_shape[3] = {1, 3, 2};
    const int logits_shape[2] = {2, 3};
    const float input[12] = {
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
    };
    const float batch_mask[4] = {2, 3, 5, 7};
    const float position[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    CHECK(add_tensor("input", output_shape, 3, input));
    CHECK(add_tensor("batch.mask", batch_mask_shape, 3, batch_mask));
    CHECK(add_tensor("position", position_shape, 3, position));
    CHECK(add_tensor("multiplied", output_shape, 3, NULL));
    CHECK(add_tensor("added", output_shape, 3, NULL));
    CHECK(add_tensor("logits", logits_shape, 2, NULL));
    Node* node = add_node("Mul", "multiplied", NULL);
    CHECK(node && node_input(node, "a", "input") == 0 &&
          node_input(node, "b", "batch.mask") == 0);
    node = add_node("Add", "added", NULL);
    CHECK(node && node_input(node, "a", "multiplied") == 0 &&
          node_input(node, "b", "position") == 0);
    node = add_node("ReduceMean", "logits", "{\"axis\":-1}");
    CHECK(node && node_input(node, "input", "added") == 0);
    CHECK(volvoxai_engine_forward() == 0);
    for (int batch = 0; batch < 2; batch++) {
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 2; column++) {
                int index = (batch * 3 + row) * 2 + column;
                float expected = input[index] * batch_mask[batch * 2 + column] +
                                 position[row * 2 + column];
                CHECK(fabsf(t_find("added")->data[index] - expected) < 1.0e-6f);
            }
            int base = (batch * 3 + row) * 2;
            float expected_mean = (t_find("added")->data[base] +
                                   t_find("added")->data[base + 1]) * 0.5f;
            CHECK(fabsf(t_find("logits")->data[batch * 3 + row] - expected_mean) < 1.0e-6f);
        }
    }
    float mask_before[4], position_before[6];
    memcpy(mask_before, t_find("batch.mask")->data, sizeof(mask_before));
    memcpy(position_before, t_find("position")->data, sizeof(position_before));
    const int targets[2] = {0, 2};
    const char* trainables[2] = {"batch.mask", "position"};
    CHECK(volvoxai_engine_train_step("logits", targets, 2, INT_MIN, trainables, 2,
                                     2, 0.001f, 0.9f, 0.999f, 1e-8f,
                                     0.0f, 0.0f, 1, NULL, NULL, NULL) == 0);
    CHECK(memcmp(mask_before, t_find("batch.mask")->data, sizeof(mask_before)) != 0);
    CHECK(memcmp(position_before, t_find("position")->data, sizeof(position_before)) != 0);
    return finish_graph();
}

static int test_weighted_multi_loss_accumulation(void) {
    begin_graph();
    g_use_vulkan = g_use_nnapi = g_use_opengl = g_use_metal = 0;
    const int input_shape[2] = {1, 2};
    const int hidden_shape[2] = {1, 2};
    const int shared_shape[2] = {2, 2};
    const int lm_weight_shape[2] = {2, 3};
    const int lm_shape[2] = {1, 3};
    const int router_weight_shape[2] = {2, 2};
    const int router_shape[2] = {1, 2};
    const float input[2] = {1.0f, -0.5f};
    const float shared[4] = {0.2f, -0.3f, 0.4f, 0.1f};
    const float lm_weight[6] = {0.1f, 0.2f, -0.2f, 0.4f, -0.1f, 0.3f};
    const float router_weight[4] = {-0.2f, 0.3f, 0.5f, -0.4f};
    CHECK(add_tensor("x", input_shape, 2, input));
    CHECK(add_tensor("shared", shared_shape, 2, shared));
    CHECK(add_tensor("hidden", hidden_shape, 2, NULL));
    CHECK(add_tensor("lm.weight", lm_weight_shape, 2, lm_weight));
    CHECK(add_tensor("lm.logits", lm_shape, 2, NULL));
    CHECK(add_tensor("router.weight", router_weight_shape, 2, router_weight));
    CHECK(add_tensor("router.logits", router_shape, 2, NULL));
    Node* node = add_node("MatMul", "hidden", NULL);
    CHECK(node && node_input(node, "input", "x") == 0 &&
          node_input(node, "weight", "shared") == 0);
    node = add_node("MatMul", "lm.logits", NULL);
    CHECK(node && node_input(node, "input", "hidden") == 0 &&
          node_input(node, "weight", "lm.weight") == 0);
    node = add_node("MatMul", "router.logits", NULL);
    CHECK(node && node_input(node, "input", "hidden") == 0 &&
          node_input(node, "weight", "router.weight") == 0);

    const char* trainables[3] = {"shared", "lm.weight", "router.weight"};
    const int lm_targets[2] = {2, 0};
    const int router_targets[2] = {1, 0};
    volvoxai_cross_entropy_loss_t losses[2] = {
        {"language", "lm.logits", &lm_targets[0], 1, INT_MIN, -1, 1.0f, 2.0f},
        {"router", "router.logits", &router_targets[0], 1, INT_MIN, -1, 0.1f, 2.0f},
    };
    float shared_before[4], lm_before[6], router_before[4];
    memcpy(shared_before, t_find("shared")->data, sizeof(shared_before));
    memcpy(lm_before, t_find("lm.weight")->data, sizeof(lm_before));
    memcpy(router_before, t_find("router.weight")->data, sizeof(router_before));
    float loss = 0.0f;
    volvoxai_cross_entropy_metric_t metrics[2] = {{0}};
    int microbatches = 0, applied = 0;
    CHECK(volvoxai_engine_train_step_multi(losses, 2, trainables, 3,
          2, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f, 0.0f, 1,
          2, 0, 0, &loss, metrics, &microbatches, &applied) == 0);
    CHECK(!applied && microbatches == 1 && metrics[0].examples == 1 &&
          metrics[1].examples == 1 && fabsf(loss - metrics[0].loss - metrics[1].loss) < 1e-6f);
    CHECK(volvoxai_engine_set_tensor_f32("shared", shared_before, 4) != 0);
    CHECK(memcmp(shared_before, t_find("shared")->data, sizeof(shared_before)) == 0);
    CHECK(memcmp(lm_before, t_find("lm.weight")->data, sizeof(lm_before)) == 0);
    CHECK(memcmp(router_before, t_find("router.weight")->data, sizeof(router_before)) == 0);

    losses[0].targets = &lm_targets[1];
    losses[1].targets = &router_targets[1];
    CHECK(volvoxai_engine_train_step_multi(losses, 2, trainables, 3,
          2, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f, 0.0f, 1,
          2, 0, 0, &loss, metrics, &microbatches, &applied) == 0);
    CHECK(applied && microbatches == 2 && metrics[0].examples == 2 &&
          metrics[1].examples == 2 && metrics[0].normalizer == 2.0f &&
          metrics[1].normalizer == 2.0f);
    CHECK(memcmp(shared_before, t_find("shared")->data, sizeof(shared_before)) != 0);
    CHECK(memcmp(lm_before, t_find("lm.weight")->data, sizeof(lm_before)) != 0);
    CHECK(memcmp(router_before, t_find("router.weight")->data, sizeof(router_before)) != 0);

    /* Accumulation requires a full-window denominator, and an all-ignore
       window closes without applying weight decay or advancing the optimizer. */
    losses[0].normalizer = 0.0f;
    CHECK(volvoxai_engine_train_step_multi(losses, 2, trainables, 3,
          2, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f, 0.0f, 2,
          2, 0, 0, &loss, metrics, &microbatches, &applied) != 0);
    losses[0].normalizer = 2.0f;
    const int ignored = -100;
    losses[0].targets = &ignored;
    losses[0].ignore_index = ignored;
    losses[1].targets = &ignored;
    losses[1].ignore_index = ignored;
    memcpy(shared_before, t_find("shared")->data, sizeof(shared_before));
    CHECK(volvoxai_engine_train_step_multi(losses, 2, trainables, 3,
          3, 0.05f, 0.9f, 0.999f, 1e-8f, 0.2f, 0.0f, 2,
          2, 0, 0, &loss, metrics, &microbatches, &applied) == 0);
    CHECK(!applied && microbatches == 1);
    CHECK(volvoxai_engine_reset_gradient_accumulation() == 0);
    CHECK(volvoxai_engine_set_tensor_f32("shared", shared_before, 4) == 0);
    CHECK(volvoxai_engine_train_step_multi(losses, 2, trainables, 3,
          3, 0.05f, 0.9f, 0.999f, 1e-8f, 0.2f, 0.0f, 2,
          2, 0, 0, &loss, metrics, &microbatches, &applied) == 0);
    CHECK(!applied && microbatches == 1);
    CHECK(volvoxai_engine_train_step_multi(losses, 2, trainables, 3,
          3, 0.05f, 0.9f, 0.999f, 1e-8f, 0.2f, 0.0f, 2,
          2, 0, 0, &loss, metrics, &microbatches, &applied) == 0);
    CHECK(!applied && microbatches == 2 && metrics[0].examples == 0 &&
          metrics[1].examples == 0 && loss == 0.0f);
    CHECK(memcmp(shared_before, t_find("shared")->data, sizeof(shared_before)) == 0);
    return finish_graph();
}

int main(void) {
    CHECK(test_public_backend_bypasses_training_tape() == 0);
    CHECK(test_gelu_exact_and_tanh() == 0);
    CHECK(test_transformer_backward() == 0);
    CHECK(test_cnn_backward() == 0);
    CHECK(test_attention_batches() == 0);
    CHECK(test_moe_routing_backward() == 0);
    CHECK(test_prelu_logsoftmax_backward() == 0);
    CHECK(test_split_multi_output_backward() == 0);
    CHECK(test_secondary_output_gradient_rejected() == 0);
    CHECK(test_train_base_route_and_prevalidation() == 0);
    CHECK(test_explicit_graph_lora_training() == 0);
    CHECK(test_attention_masks_and_ignored_targets() == 0);
    CHECK(test_native_seq2seq_training() == 0);
    CHECK(test_groupnorm_training() == 0);
    CHECK(test_dropout_training_and_inference() == 0);
    CHECK(test_attention_dropout_training_and_inference() == 0);
    CHECK(test_train_step_global_gradient_clipping() == 0);
    CHECK(test_shape_aware_broadcast_and_reduction_backward() == 0);
    CHECK(test_weighted_multi_loss_accumulation() == 0);
    CHECK(test_fused_graph_training() == 0);
    CHECK(test_optimized_dropout_alias_training() == 0);
    CHECK(test_append_first_node() == 0);
    CHECK(test_patch_graph_transactions() == 0);
    CHECK(test_vulkan_engine_train_step_optional() == 0);
    puts("native training backward tests passed");
    return 0;
}
