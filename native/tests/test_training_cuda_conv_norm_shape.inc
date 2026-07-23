#ifndef VOLVOXAI_TEST_TRAINING_CUDA_CONV_NORM_SHAPE_INC
#define VOLVOXAI_TEST_TRAINING_CUDA_CONV_NORM_SHAPE_INC

/* This file is textually included by test_training_backward.c after its graph
 * construction helpers and inside VOLVOXAI_ENABLE_CUDA.  These are deliberately
 * TrainStep tests rather than direct-kernel tests: the CPU and required-CUDA
 * runs both execute the graph forward, build the backward plan, and apply SGD.
 */

typedef struct {
    float conv_weight[8];
    float conv_bias[2];
    float batchnorm_weight[2];
    float batchnorm_bias[2];
} CudaCnnTrainingSnapshot;

typedef struct {
    float image[18];
    float conv_weight[8];
    float conv_bias[2];
    float groupnorm_weight[2];
    float groupnorm_bias[2];
    float prelu_weight[2];
} CudaDepthwiseNormTrainingSnapshot;

typedef struct {
    float input[6];
    float weight[1];
} CudaScalarPreluTrainingSnapshot;

typedef struct {
    float first[2];
    float second[2];
} CudaShapeTrainingSnapshot;

typedef struct {
    float a[6];
    float b[6];
} CudaFusedAddTrainingSnapshot;

typedef struct {
    float input[8];
    float group_weight[4];
    float layer_weight[6];
    float rms_weight[5];
    float slope[4];
} CudaOptionalContractTrainingSnapshot;

typedef struct {
    float input[18];
    float weight[16];
} CudaConvLayoutTrainingSnapshot;

static int cuda_conv_norm_shape_arrays_close(const float* cpu,
                                              const float* cuda,
                                              int count,
                                              float tolerance) {
    if (!cpu || !cuda || count <= 0 || !(tolerance >= 0.0f)) return 0;
    for (int index = 0; index < count; index++) {
        float scale = 1.0f + fmaxf(fabsf(cpu[index]), fabsf(cuda[index]));
        if (!isfinite(cpu[index]) || !isfinite(cuda[index]) ||
            fabsf(cpu[index] - cuda[index]) > tolerance * scale) return 0;
    }
    return 1;
}

static int cuda_conv_norm_shape_any_changed(const float* before,
                                             const float* after,
                                             int count) {
    if (!before || !after || count <= 0) return 0;
    for (int index = 0; index < count; index++)
        if (fabsf(before[index] - after[index]) > 1.0e-7f) return 1;
    return 0;
}

#define CUDA_COPY_TRAINING_SNAPSHOT(tensor_name, member)                     \
    CHECK(volvoxai_engine_copy_tensor_f32(                                  \
        (tensor_name), snapshot->member,                                    \
        (long)(sizeof(snapshot->member) / sizeof(snapshot->member[0]))) == 0)

static int cuda_conv_norm_shape_begin(int use_cuda) {
    begin_graph();
    g_use_vulkan = g_use_nnapi = g_use_opengl = g_use_metal = 0;
    g_use_cuda = use_cuda;
    if (use_cuda && cuda_init() != 0) {
        int unavailable = cuda_init_failure_is_unavailable();
        if (finish_graph() != 0) return -1;
        return unavailable ? 1 : -1;
    }
    if (volvoxai_engine_require_training_backend(
            use_cuda ? VOLVOXAI_BACKEND_CUDA : VOLVOXAI_BACKEND_CPU) != 0) {
        finish_graph();
        return -1;
    }
    return 0;
}

static int cuda_conv_norm_shape_finish(int use_cuda) {
    /* The public accessor reads the planner's g_last_training_backend while
     * keeping that translation-unit-local implementation detail private. */
    CHECK(volvoxai_engine_last_training_backend() ==
          (use_cuda ? VOLVOXAI_BACKEND_CUDA : VOLVOXAI_BACKEND_CPU));
    CHECK(volvoxai_engine_require_training_backend(VOLVOXAI_BACKEND_CPU) == 0);
    return finish_graph();
}

static int run_cuda_cnn_training_parity_case(
        int use_cuda, CudaCnnTrainingSnapshot* snapshot) {
    const int input_shape[4] = {1, 3, 3, 1};
    const int conv_weight_shape[4] = {2, 2, 1, 2};
    const int conv_shape[4] = {1, 2, 2, 2};
    const int pooled_shape[4] = {1, 1, 1, 2};
    const int channel_shape[1] = {2};
    const int head_shape[2] = {2, 2};
    const int logits_shape[2] = {1, 2};
    const float input[9] = {
        0.2f, -0.1f, 0.4f,
        0.7f, 0.3f, -0.2f,
        0.5f, 0.9f, 0.1f,
    };
    const float conv_weight[8] = {
        0.3f, -0.2f, 0.4f, 0.1f,
        -0.1f, 0.5f, 0.2f, 0.6f,
    };
    const float zeros[2] = {0.0f, 0.0f};
    const float ones[2] = {1.0f, 1.0f};
    const float head_weight[4] = {0.6f, -0.4f, 0.3f, 0.8f};
    const char* trainables[4] = {
        "cnn.conv.weight", "cnn.conv.bias",
        "cnn.bn.weight", "cnn.bn.bias",
    };
    const int target[1] = {1};
    float loss = 0.0f;
    int examples = 0;

    if (!snapshot) return -1;
    int begin_result = cuda_conv_norm_shape_begin(use_cuda);
    if (begin_result != 0) return begin_result;

    CHECK(add_tensor("cnn.image", input_shape, 4, input));
    CHECK(add_tensor("cnn.conv.weight", conv_weight_shape, 4, conv_weight));
    CHECK(add_tensor("cnn.conv.bias", channel_shape, 1, zeros));
    CHECK(add_tensor("cnn.conv", conv_shape, 4, NULL));
    CHECK(add_tensor("cnn.bn.weight", channel_shape, 1, ones));
    CHECK(add_tensor("cnn.bn.bias", channel_shape, 1, zeros));
    CHECK(add_tensor("cnn.bn.mean", channel_shape, 1, zeros));
    CHECK(add_tensor("cnn.bn.var", channel_shape, 1, ones));
    CHECK(add_tensor("cnn.bn", conv_shape, 4, NULL));
    CHECK(add_tensor("cnn.pool", pooled_shape, 4, NULL));
    CHECK(add_tensor("cnn.resize", conv_shape, 4, NULL));
    CHECK(add_tensor("cnn.gap", logits_shape, 2, NULL));
    CHECK(add_tensor("cnn.head.weight", head_shape, 2, head_weight));
    CHECK(add_tensor("cnn.logits", logits_shape, 2, NULL));

    Node* node = add_node(
        "Conv2D", "cnn.conv",
        "{\"weight_layout\":\"HWIO\",\"stride\":[1,1],\"padding\":[0,0]}");
    CHECK(node && node_input(node, "input", "cnn.image") == 0 &&
          node_input(node, "weight", "cnn.conv.weight") == 0 &&
          node_input(node, "bias", "cnn.conv.bias") == 0);
    node = add_node("BatchNorm2D", "cnn.bn", "{\"eps\":0.00001}");
    CHECK(node && node_input(node, "input", "cnn.conv") == 0 &&
          node_input(node, "weight", "cnn.bn.weight") == 0 &&
          node_input(node, "bias", "cnn.bn.bias") == 0 &&
          node_input(node, "running_mean", "cnn.bn.mean") == 0 &&
          node_input(node, "running_var", "cnn.bn.var") == 0);
    node = add_node("MaxPool2D", "cnn.pool",
                    "{\"kernel\":[2,2],\"stride\":[2,2]}");
    CHECK(node && node_input(node, "input", "cnn.bn") == 0);
    node = add_node("ResizeNearest2D", "cnn.resize", NULL);
    CHECK(node && node_input(node, "input", "cnn.pool") == 0);
    node = add_node("GlobalAveragePool", "cnn.gap", NULL);
    CHECK(node && node_input(node, "input", "cnn.resize") == 0);
    node = add_node("MatMul", "cnn.logits", NULL);
    CHECK(node && node_input(node, "input", "cnn.gap") == 0 &&
          node_input(node, "weight", "cnn.head.weight") == 0);

    CHECK(volvoxai_engine_train_step(
        "cnn.logits", target, 1, INT_MIN, trainables, 4,
        VOLVOXAI_TENSOR_UPDATE_SGD, 0.025f, 0.9f, 0.999f, 1.0e-8f,
        0.0f, 0.0f, 1, &loss, NULL, &examples) == 0);
    CHECK(isfinite(loss) && loss > 0.0f && examples == 1);
    CUDA_COPY_TRAINING_SNAPSHOT("cnn.conv.weight", conv_weight);
    CUDA_COPY_TRAINING_SNAPSHOT("cnn.conv.bias", conv_bias);
    CUDA_COPY_TRAINING_SNAPSHOT("cnn.bn.weight", batchnorm_weight);
    CUDA_COPY_TRAINING_SNAPSHOT("cnn.bn.bias", batchnorm_bias);
    CHECK(cuda_conv_norm_shape_any_changed(
        conv_weight, snapshot->conv_weight, 8));
    CHECK(cuda_conv_norm_shape_any_changed(
        ones, snapshot->batchnorm_weight, 2) ||
          cuda_conv_norm_shape_any_changed(
        zeros, snapshot->batchnorm_bias, 2));
    return cuda_conv_norm_shape_finish(use_cuda);
}

static int run_cuda_depthwise_norm_training_parity_case(
        int use_cuda, CudaDepthwiseNormTrainingSnapshot* snapshot) {
    const int image_shape[4] = {1, 3, 3, 2};
    const int weight_shape[4] = {2, 2, 2, 1};
    const int feature_shape[4] = {1, 2, 2, 2};
    const int channel_shape[1] = {2};
    const int logits_shape[2] = {1, 2};
    const float image[18] = {
        -1.0f, 0.5f, 0.25f, -0.75f, 1.25f, 0.2f,
        0.6f, -1.1f, -0.4f, 0.9f, 1.4f, -0.3f,
        -0.8f, 0.7f, 0.1f, -1.3f, 0.95f, 0.45f,
    };
    const float conv_weight[8] = {
        0.5f, -0.25f, -0.3f, 0.4f,
        0.2f, 0.6f, -0.5f, 0.1f,
    };
    const float conv_bias[2] = {0.1f, -0.2f};
    const float group_weight[2] = {1.1f, 0.8f};
    const float group_bias[2] = {0.05f, -0.15f};
    const float prelu_weight[2] = {0.2f, 0.35f};
    const char* trainables[6] = {
        "depth.image", "depth.conv.weight", "depth.conv.bias",
        "depth.group.weight", "depth.group.bias", "depth.prelu.weight",
    };
    const int target[1] = {1};
    float loss = 0.0f;

    if (!snapshot) return -1;
    int begin_result = cuda_conv_norm_shape_begin(use_cuda);
    if (begin_result != 0) return begin_result;

    CHECK(add_tensor("depth.image", image_shape, 4, image));
    CHECK(add_tensor("depth.conv.weight", weight_shape, 4, conv_weight));
    CHECK(add_tensor("depth.conv.bias", channel_shape, 1, conv_bias));
    CHECK(add_tensor("depth.conv", feature_shape, 4, NULL));
    CHECK(add_tensor("depth.group.weight", channel_shape, 1, group_weight));
    CHECK(add_tensor("depth.group.bias", channel_shape, 1, group_bias));
    CHECK(add_tensor("depth.group", feature_shape, 4, NULL));
    CHECK(add_tensor("depth.prelu.weight", channel_shape, 1, prelu_weight));
    CHECK(add_tensor("depth.prelu", feature_shape, 4, NULL));
    CHECK(add_tensor("depth.logits", logits_shape, 2, NULL));

    Node* node = add_node(
        "Conv2D", "depth.conv",
        "{\"weight_layout\":\"HWCM\",\"groups\":2,"
        "\"stride\":[1,1],\"padding\":[0,0]}");
    CHECK(node && node_input(node, "input", "depth.image") == 0 &&
          node_input(node, "weight", "depth.conv.weight") == 0 &&
          node_input(node, "bias", "depth.conv.bias") == 0);
    node = add_node("GroupNorm", "depth.group",
                    "{\"num_groups\":1,\"eps\":0.0001}");
    CHECK(node && node_input(node, "input", "depth.conv") == 0 &&
          node_input(node, "weight", "depth.group.weight") == 0 &&
          node_input(node, "bias", "depth.group.bias") == 0);
    node = add_node("PReLU", "depth.prelu", NULL);
    CHECK(node && node_input(node, "input", "depth.group") == 0 &&
          node_input(node, "weight", "depth.prelu.weight") == 0);
    node = add_node("GlobalAveragePool", "depth.logits", NULL);
    CHECK(node && node_input(node, "input", "depth.prelu") == 0);

    CHECK(volvoxai_engine_train_step(
        "depth.logits", target, 1, INT_MIN, trainables, 6,
        VOLVOXAI_TENSOR_UPDATE_SGD, 0.015f, 0.9f, 0.999f, 1.0e-8f,
        0.0f, 0.0f, 2, &loss, NULL, NULL) == 0);
    CHECK(isfinite(loss) && loss > 0.0f);
    CUDA_COPY_TRAINING_SNAPSHOT("depth.image", image);
    CUDA_COPY_TRAINING_SNAPSHOT("depth.conv.weight", conv_weight);
    CUDA_COPY_TRAINING_SNAPSHOT("depth.conv.bias", conv_bias);
    CUDA_COPY_TRAINING_SNAPSHOT("depth.group.weight", groupnorm_weight);
    CUDA_COPY_TRAINING_SNAPSHOT("depth.group.bias", groupnorm_bias);
    CUDA_COPY_TRAINING_SNAPSHOT("depth.prelu.weight", prelu_weight);
    CHECK(cuda_conv_norm_shape_any_changed(image, snapshot->image, 18));
    CHECK(cuda_conv_norm_shape_any_changed(
        conv_weight, snapshot->conv_weight, 8));
    CHECK(cuda_conv_norm_shape_any_changed(
        group_weight, snapshot->groupnorm_weight, 2) ||
          cuda_conv_norm_shape_any_changed(
        group_bias, snapshot->groupnorm_bias, 2));
    CHECK(cuda_conv_norm_shape_any_changed(
        prelu_weight, snapshot->prelu_weight, 2));
    return cuda_conv_norm_shape_finish(use_cuda);
}

static int run_cuda_scalar_prelu_training_parity_case(
        int use_cuda, CudaScalarPreluTrainingSnapshot* snapshot) {
    const int input_shape[2] = {2, 3};
    const int weight_shape[1] = {1};
    const float input[6] = {-1.0f, 0.5f, 2.0f, 1.5f, -0.75f, 0.25f};
    const float weight[1] = {0.3f};
    const int targets[2] = {0, 2};
    const char* trainables[2] = {
        "scalar_prelu.input", "scalar_prelu.weight",
    };

    if (!snapshot) return -1;
    int begin_result = cuda_conv_norm_shape_begin(use_cuda);
    if (begin_result != 0) return begin_result;

    CHECK(add_tensor("scalar_prelu.input", input_shape, 2, input));
    CHECK(add_tensor("scalar_prelu.weight", weight_shape, 1, weight));
    CHECK(add_tensor("scalar_prelu.logits", input_shape, 2, NULL));
    Node* node = add_node("PReLU", "scalar_prelu.logits", NULL);
    CHECK(node && node_input(node, "input", "scalar_prelu.input") == 0 &&
          node_input(node, "weight", "scalar_prelu.weight") == 0);

    CHECK(volvoxai_engine_train_step(
        "scalar_prelu.logits", targets, 2, INT_MIN, trainables, 2,
        VOLVOXAI_TENSOR_UPDATE_SGD, 0.02f, 0.9f, 0.999f, 1.0e-8f,
        0.0f, 0.0f, 3, NULL, NULL, NULL) == 0);
    CUDA_COPY_TRAINING_SNAPSHOT("scalar_prelu.input", input);
    CUDA_COPY_TRAINING_SNAPSHOT("scalar_prelu.weight", weight);
    CHECK(cuda_conv_norm_shape_any_changed(input, snapshot->input, 6));
    CHECK(cuda_conv_norm_shape_any_changed(weight, snapshot->weight, 1));
    return cuda_conv_norm_shape_finish(use_cuda);
}

static int run_cuda_shape_training_parity_case(
        int use_cuda, CudaShapeTrainingSnapshot* snapshot) {
    const int input_shape[2] = {2, 1};
    const int matrix_shape[2] = {2, 2};
    const int split_shape[2] = {1, 2};
    const float first[2] = {0.2f, -0.4f};
    const float second[2] = {0.7f, 0.1f};
    const int target[1] = {1};
    const char* trainables[2] = {"shape.first", "shape.second"};

    if (!snapshot) return -1;
    int begin_result = cuda_conv_norm_shape_begin(use_cuda);
    if (begin_result != 0) return begin_result;

    CHECK(add_tensor("shape.first", input_shape, 2, first));
    CHECK(add_tensor("shape.second", input_shape, 2, second));
    CHECK(add_tensor("shape.concat", matrix_shape, 2, NULL));
    CHECK(add_tensor("shape.transpose", matrix_shape, 2, NULL));
    CHECK(add_tensor("shape.left", split_shape, 2, NULL));
    CHECK(add_tensor("shape.right", split_shape, 2, NULL));
    CHECK(add_tensor("shape.logits", split_shape, 2, NULL));

    Node* node = add_node("Concat", "shape.concat", "{\"axis\":1}");
    CHECK(node && node_input(node, "first", "shape.first") == 0 &&
          node_input(node, "second", "shape.second") == 0);
    node = add_node("Transpose", "shape.transpose", "{\"perm\":[1,0]}");
    CHECK(node && node_input(node, "input", "shape.concat") == 0);
    node = add_node("Split", "shape.left", "{\"axis\":0}");
    CHECK(node && node_input(node, "input", "shape.transpose") == 0);
    strncpy(node->outs[0].key, "left", sizeof(node->outs[0].key) - 1);
    strncpy(node->outs[1].key, "right", sizeof(node->outs[1].key) - 1);
    strncpy(node->outs[1].name, "shape.right",
            sizeof(node->outs[1].name) - 1);
    node->nout = 2;
    node = add_node("Add", "shape.logits", NULL);
    CHECK(node && node_input(node, "a", "shape.left") == 0 &&
          node_input(node, "b", "shape.right") == 0);

    CHECK(volvoxai_engine_train_step(
        "shape.logits", target, 1, INT_MIN, trainables, 2,
        VOLVOXAI_TENSOR_UPDATE_SGD, 0.02f, 0.9f, 0.999f, 1.0e-8f,
        0.0f, 0.0f, 4, NULL, NULL, NULL) == 0);
    CUDA_COPY_TRAINING_SNAPSHOT("shape.first", first);
    CUDA_COPY_TRAINING_SNAPSHOT("shape.second", second);
    CHECK(cuda_conv_norm_shape_any_changed(first, snapshot->first, 2));
    CHECK(cuda_conv_norm_shape_any_changed(second, snapshot->second, 2));
    return cuda_conv_norm_shape_finish(use_cuda);
}

static int run_cuda_fused_add_training_parity_case(
        int use_cuda, int relu, CudaFusedAddTrainingSnapshot* snapshot) {
    const int shape[2] = {2, 3};
    const float a[6] = {-2.0f, 0.5f, 4.0f, -1.0f, 2.0f, 8.0f};
    const float b[6] = {1.0f, 0.5f, 2.0f, 1.0f, 1.0f, -1.0f};
    const int targets[2] = {1, 1};
    const char* trainables[2] = {"fused_add.a", "fused_add.b"};
    const char* params = relu == 1 ? "{\"relu\":1}" : "{\"relu\":2}";

    if (!snapshot || (relu != 1 && relu != 2)) return -1;
    int begin_result = cuda_conv_norm_shape_begin(use_cuda);
    if (begin_result != 0) return begin_result;
    CHECK(add_tensor("fused_add.a", shape, 2, a));
    CHECK(add_tensor("fused_add.b", shape, 2, b));
    CHECK(add_tensor("fused_add.logits", shape, 2, NULL));
    Node* node = add_node("Add", "fused_add.logits", params);
    CHECK(node && node_input(node, "a", "fused_add.a") == 0 &&
          node_input(node, "b", "fused_add.b") == 0);
    CHECK(volvoxai_engine_train_step(
        "fused_add.logits", targets, 2, INT_MIN, trainables, 2,
        VOLVOXAI_TENSOR_UPDATE_SGD, 0.03f, 0.9f, 0.999f, 1.0e-8f,
        0.0f, 0.0f, 41 + relu, NULL, NULL, NULL) == 0);
    CUDA_COPY_TRAINING_SNAPSHOT("fused_add.a", a);
    CUDA_COPY_TRAINING_SNAPSHOT("fused_add.b", b);
    CHECK(cuda_conv_norm_shape_any_changed(a, snapshot->a, 6));
    CHECK(cuda_conv_norm_shape_any_changed(b, snapshot->b, 6));
    return cuda_conv_norm_shape_finish(use_cuda);
}

static int run_cuda_optional_contract_training_parity_case(
        int use_cuda, CudaOptionalContractTrainingSnapshot* snapshot) {
    const int feature_shape[4] = {1, 2, 1, 4};
    const int channel_shape[1] = {4};
    const int layer_shape[1] = {6};
    const int rms_shape[1] = {5};
    const int logits_shape[2] = {1, 4};
    const float input[8] = {
        -1.0f, 0.5f, 2.0f, -0.25f,
        0.75f, -1.5f, 0.25f, 1.25f,
    };
    const float group_weight[4] = {0.8f, 1.1f, 0.9f, 1.2f};
    const float layer_weight[6] = {1.0f, 0.7f, 1.3f, 0.6f, 9.0f, -7.0f};
    const float rms_weight[5] = {0.9f, 1.2f, 0.8f, 1.1f, 5.0f};
    const float slope[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    const int target[1] = {2};
    const char* trainables[5] = {
        "optional.input", "optional.group_weight", "optional.layer_weight",
        "optional.rms_weight", "optional.slope",
    };

    if (!snapshot) return -1;
    int begin_result = cuda_conv_norm_shape_begin(use_cuda);
    if (begin_result != 0) return begin_result;
    CHECK(add_tensor("optional.input", feature_shape, 4, input));
    CHECK(add_tensor("optional.group_weight", channel_shape, 1,
                     group_weight));
    CHECK(add_tensor("optional.group", feature_shape, 4, NULL));
    CHECK(add_tensor("optional.layer_weight", layer_shape, 1,
                     layer_weight));
    CHECK(add_tensor("optional.layer", feature_shape, 4, NULL));
    CHECK(add_tensor("optional.rms_weight", rms_shape, 1, rms_weight));
    CHECK(add_tensor("optional.rms", feature_shape, 4, NULL));
    CHECK(add_tensor("optional.slope", channel_shape, 1, slope));
    CHECK(add_tensor("optional.prelu", feature_shape, 4, NULL));
    CHECK(add_tensor("optional.logits", logits_shape, 2, NULL));

    Node* node = add_node("GroupNorm", "optional.group",
                          "{\"num_groups\":2,\"eps\":0.0001}");
    CHECK(node && node_input(node, "input", "optional.input") == 0 &&
          node_input(node, "scale", "optional.group_weight") == 0);
    node = add_node("LayerNorm", "optional.layer", "{\"eps\":0.00001}");
    CHECK(node && node_input(node, "input", "optional.group") == 0 &&
          node_input(node, "weight", "optional.layer_weight") == 0);
    node = add_node("RMSNorm", "optional.rms", "{\"eps\":0.00001}");
    CHECK(node && node_input(node, "input", "optional.layer") == 0 &&
          node_input(node, "weight", "optional.rms_weight") == 0);
    node = add_node("PReLU", "optional.prelu", NULL);
    CHECK(node && node_input(node, "input", "optional.rms") == 0 &&
          node_input(node, "slope", "optional.slope") == 0);
    node = add_node("GlobalAveragePool", "optional.logits", NULL);
    CHECK(node && node_input(node, "input", "optional.prelu") == 0);

    CHECK(volvoxai_engine_train_step(
        "optional.logits", target, 1, INT_MIN, trainables, 5,
        VOLVOXAI_TENSOR_UPDATE_SGD, 0.01f, 0.9f, 0.999f, 1.0e-8f,
        0.0f, 0.0f, 53, NULL, NULL, NULL) == 0);
    CUDA_COPY_TRAINING_SNAPSHOT("optional.input", input);
    CUDA_COPY_TRAINING_SNAPSHOT("optional.group_weight", group_weight);
    CUDA_COPY_TRAINING_SNAPSHOT("optional.layer_weight", layer_weight);
    CUDA_COPY_TRAINING_SNAPSHOT("optional.rms_weight", rms_weight);
    CUDA_COPY_TRAINING_SNAPSHOT("optional.slope", slope);
    CHECK(snapshot->layer_weight[4] == layer_weight[4] &&
          snapshot->layer_weight[5] == layer_weight[5] &&
          snapshot->rms_weight[4] == rms_weight[4]);
    return cuda_conv_norm_shape_finish(use_cuda);
}

static int run_cuda_conv_layout_training_parity_case(
        int use_cuda, const char* layout, int depthwise,
        CudaConvLayoutTrainingSnapshot* snapshot) {
    const int input_shape[4] = {1, 3, 3, 2};
    const int regular_weight_shape[4] = {2, 2, 2, 2};
    const int depthwise_weight_shape[4] = {1, 2, 2, 4};
    const int regular_feature_shape[4] = {1, 2, 2, 2};
    const int depthwise_feature_shape[4] = {1, 2, 2, 4};
    const int regular_logits_shape[2] = {1, 2};
    const int depthwise_logits_shape[2] = {1, 4};
    const float input[18] = {
        -1.0f, 0.5f, 0.25f, -0.75f, 1.25f, 0.2f,
        0.6f, -1.1f, -0.4f, 0.9f, 1.4f, -0.3f,
        -0.8f, 0.7f, 0.1f, -1.3f, 0.95f, 0.45f,
    };
    const float weight[16] = {
        0.5f, -0.25f, -0.3f, 0.4f,
        0.2f, 0.6f, -0.5f, 0.1f,
        -0.15f, 0.35f, 0.45f, -0.2f,
        0.55f, -0.4f, 0.25f, 0.7f,
    };
    const char* trainables[2] = {"layout.input", "layout.weight"};
    const int regular_target[1] = {1};
    const int depthwise_target[1] = {2};
    char params[128];

    if (!layout || !snapshot) return -1;
    int begin_result = cuda_conv_norm_shape_begin(use_cuda);
    if (begin_result != 0) return begin_result;
    CHECK(snprintf(params, sizeof(params),
                   "{\"weight_layout\":\"%s\",\"groups\":%d,"
                   "\"stride\":[1,1],\"padding\":[0,0]}",
                   layout, depthwise ? 2 : 1) > 0);
    CHECK(add_tensor("layout.input", input_shape, 4, input));
    CHECK(add_tensor("layout.weight",
                     depthwise ? depthwise_weight_shape : regular_weight_shape,
                     4, weight));
    CHECK(add_tensor("layout.feature",
                     depthwise ? depthwise_feature_shape : regular_feature_shape,
                     4, NULL));
    CHECK(add_tensor("layout.logits",
                     depthwise ? depthwise_logits_shape : regular_logits_shape,
                     2, NULL));
    Node* node = add_node("Conv2D", "layout.feature", params);
    CHECK(node && node_input(node, "input", "layout.input") == 0 &&
          node_input(node, "weight", "layout.weight") == 0);
    node = add_node("GlobalAveragePool", "layout.logits", NULL);
    CHECK(node && node_input(node, "input", "layout.feature") == 0);
    CHECK(volvoxai_engine_train_step(
        "layout.logits", depthwise ? depthwise_target : regular_target, 1,
        INT_MIN, trainables, 2, VOLVOXAI_TENSOR_UPDATE_SGD, 0.01f,
        0.9f, 0.999f, 1.0e-8f, 0.0f, 0.0f, 61, NULL, NULL, NULL) == 0);
    CUDA_COPY_TRAINING_SNAPSHOT("layout.input", input);
    CUDA_COPY_TRAINING_SNAPSHOT("layout.weight", weight);
    CHECK(cuda_conv_norm_shape_any_changed(input, snapshot->input, 18));
    CHECK(cuda_conv_norm_shape_any_changed(weight, snapshot->weight, 16));
    return cuda_conv_norm_shape_finish(use_cuda);
}

static int test_cuda_fused_add_malformed_contract(void) {
    const int shape[2] = {2, 3};
    const float a[6] = {-2.0f, 0.5f, 4.0f, -1.0f, 2.0f, 8.0f};
    const float b[6] = {1.0f, 0.5f, 2.0f, 1.0f, 1.0f, -1.0f};
    const int targets[2] = {1, 1};
    const char* trainables[2] = {"bad_add.a", "bad_add.b"};
    int begin_result = cuda_conv_norm_shape_begin(1);
    if (begin_result != 0) return begin_result;
    CHECK(add_tensor("bad_add.a", shape, 2, a));
    CHECK(add_tensor("bad_add.b", shape, 2, b));
    CHECK(add_tensor("bad_add.logits", shape, 2, NULL));
    Node* node = add_node("Add", "bad_add.logits", "{\"relu\":3}");
    CHECK(node && node_input(node, "a", "bad_add.a") == 0 &&
          node_input(node, "b", "bad_add.b") == 0);
    CHECK(volvoxai_engine_train_step(
        "bad_add.logits", targets, 2, INT_MIN, trainables, 2,
        VOLVOXAI_TENSOR_UPDATE_SGD, 0.03f, 0.9f, 0.999f, 1.0e-8f,
        0.0f, 0.0f, 71, NULL, NULL, NULL) != 0);
    CHECK(memcmp(t_find("bad_add.a")->data, a, sizeof(a)) == 0);
    CHECK(memcmp(t_find("bad_add.b")->data, b, sizeof(b)) == 0);
    CHECK(volvoxai_engine_require_training_backend(VOLVOXAI_BACKEND_CPU) == 0);
    return finish_graph();
}

static int test_cuda_conv_norm_shape_runtime_optional(void) {
    CudaCnnTrainingSnapshot cpu_cnn, cuda_cnn;
    CudaDepthwiseNormTrainingSnapshot cpu_depthwise, cuda_depthwise;
    CudaScalarPreluTrainingSnapshot cpu_scalar, cuda_scalar;
    CudaShapeTrainingSnapshot cpu_shape, cuda_shape;
    CudaFusedAddTrainingSnapshot cpu_add, cuda_add;
    CudaOptionalContractTrainingSnapshot cpu_optional, cuda_optional;
    CudaConvLayoutTrainingSnapshot cpu_layout, cuda_layout;

    CHECK(run_cuda_cnn_training_parity_case(0, &cpu_cnn) == 0);
    int cuda_result = run_cuda_cnn_training_parity_case(1, &cuda_cnn);
    if (cuda_result == 1) {
        puts("native CUDA conv/norm/shape TrainStep integration skipped: no CUDA device");
        return 0;
    }
    CHECK(cuda_result == 0);
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_cnn.conv_weight, cuda_cnn.conv_weight, 8, 2.0e-4f));
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_cnn.conv_bias, cuda_cnn.conv_bias, 2, 2.0e-4f));
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_cnn.batchnorm_weight, cuda_cnn.batchnorm_weight, 2, 2.0e-4f));
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_cnn.batchnorm_bias, cuda_cnn.batchnorm_bias, 2, 2.0e-4f));

    CHECK(run_cuda_depthwise_norm_training_parity_case(
        0, &cpu_depthwise) == 0);
    cuda_result = run_cuda_depthwise_norm_training_parity_case(
        1, &cuda_depthwise);
    CHECK(cuda_result == 0);
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_depthwise.image, cuda_depthwise.image, 18, 5.0e-4f));
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_depthwise.conv_weight, cuda_depthwise.conv_weight, 8, 5.0e-4f));
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_depthwise.conv_bias, cuda_depthwise.conv_bias, 2, 5.0e-4f));
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_depthwise.groupnorm_weight,
        cuda_depthwise.groupnorm_weight, 2, 5.0e-4f));
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_depthwise.groupnorm_bias,
        cuda_depthwise.groupnorm_bias, 2, 5.0e-4f));
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_depthwise.prelu_weight,
        cuda_depthwise.prelu_weight, 2, 5.0e-4f));

    CHECK(run_cuda_scalar_prelu_training_parity_case(0, &cpu_scalar) == 0);
    cuda_result = run_cuda_scalar_prelu_training_parity_case(1, &cuda_scalar);
    CHECK(cuda_result == 0);
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_scalar.input, cuda_scalar.input, 6, 2.0e-6f));
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_scalar.weight, cuda_scalar.weight, 1, 2.0e-6f));

    CHECK(run_cuda_shape_training_parity_case(0, &cpu_shape) == 0);
    cuda_result = run_cuda_shape_training_parity_case(1, &cuda_shape);
    CHECK(cuda_result == 0);
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_shape.first, cuda_shape.first, 2, 2.0e-6f));
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_shape.second, cuda_shape.second, 2, 2.0e-6f));

    for (int relu = 1; relu <= 2; relu++) {
        CHECK(run_cuda_fused_add_training_parity_case(
            0, relu, &cpu_add) == 0);
        CHECK(run_cuda_fused_add_training_parity_case(
            1, relu, &cuda_add) == 0);
        CHECK(cuda_conv_norm_shape_arrays_close(
            cpu_add.a, cuda_add.a, 6, 2.0e-6f));
        CHECK(cuda_conv_norm_shape_arrays_close(
            cpu_add.b, cuda_add.b, 6, 2.0e-6f));
    }
    CHECK(test_cuda_fused_add_malformed_contract() == 0);

    CHECK(run_cuda_optional_contract_training_parity_case(
        0, &cpu_optional) == 0);
    CHECK(run_cuda_optional_contract_training_parity_case(
        1, &cuda_optional) == 0);
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_optional.input, cuda_optional.input, 8, 8.0e-4f));
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_optional.group_weight, cuda_optional.group_weight, 4, 8.0e-4f));
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_optional.layer_weight, cuda_optional.layer_weight, 6, 8.0e-4f));
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_optional.rms_weight, cuda_optional.rms_weight, 5, 8.0e-4f));
    CHECK(cuda_conv_norm_shape_arrays_close(
        cpu_optional.slope, cuda_optional.slope, 4, 8.0e-4f));

    const char* layouts[3] = {"OHWI", "1HWO", "1HWM"};
    for (int layout_index = 0; layout_index < 3; layout_index++) {
        int depthwise = layout_index != 0;
        CHECK(run_cuda_conv_layout_training_parity_case(
            0, layouts[layout_index], depthwise, &cpu_layout) == 0);
        CHECK(run_cuda_conv_layout_training_parity_case(
            1, layouts[layout_index], depthwise, &cuda_layout) == 0);
        CHECK(cuda_conv_norm_shape_arrays_close(
            cpu_layout.input, cuda_layout.input, 18, 7.0e-4f));
        CHECK(cuda_conv_norm_shape_arrays_close(
            cpu_layout.weight, cuda_layout.weight, 16, 7.0e-4f));
    }
    return 0;
}

#undef CUDA_COPY_TRAINING_SNAPSHOT

#endif
