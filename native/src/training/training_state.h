#ifndef VOLVOXAI_TRAINING_STATE_H
#define VOLVOXAI_TRAINING_STATE_H

#include "training_core.h"

#include <stdint.h>

typedef enum NativeGpuTrainingBackend {
    NATIVE_GPU_TRAINING_NONE = 0,
    NATIVE_GPU_TRAINING_VULKAN = 1,
    NATIVE_GPU_TRAINING_OPENGL = 2,
    NATIVE_GPU_TRAINING_METAL = 3,
    NATIVE_GPU_TRAINING_CUDA = VX_BACKEND_KIND_CUDA
} NativeGpuTrainingBackend;

typedef struct EngineOptimizerState {
    char name[128];
    long numel;
    float* m;
    float* v;
} EngineOptimizerState;

typedef struct TrainingAccumulationLoss {
    char* name;
    char* logits_name;
    float weight;
    float normalizer;
    volvoxai_cross_entropy_metric_t metric;
} TrainingAccumulationLoss;

#if VOLVOXAI_ENABLE_CUDA
typedef struct CudaTrainingWindow CudaTrainingWindow;
#endif

typedef struct TrainingAccumulationState {
    int active;
    NativeGpuTrainingBackend backend;
    int microbatches;
    int accumulation_steps;
    int has_contribution;
    int trainable_count;
    int loss_count;
    int tensor_count;
    int node_count;
    uint64_t topology_hash;
    int update_mode;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    float max_grad_norm;
    long step;
    float loss;
    char** trainable_names;
    long* trainable_numel;
    float** gradients;
#if VOLVOXAI_ENABLE_CUDA
    CudaTrainingWindow* cuda_window;
#endif
    TrainingAccumulationLoss* losses;
} TrainingAccumulationState;

#endif /* VOLVOXAI_TRAINING_STATE_H */
