#ifndef VOLVOXAI_API_AUTHORING_H
#define VOLVOXAI_API_AUTHORING_H

/* Private full-profile helpers shared by generated-dispatch adapters. */
#include "vx_api_convert.h"

VxStatus vx_api_initialize_f32(const int64_t* shape, size_t rank,
    const VolvoxaiV1TensorInitializer* spec, float** output, size_t* count);
int vx_api_build_lora_linear(const VolvoxaiV1BuildLoraLinearRequest* request,
    VolvoxaiV1LoraLinearPlan* response, void* user_data);
int vx_api_build_routed_adapter(const VolvoxaiV1BuildRoutedAdapterRequest* request,
    VolvoxaiV1RoutedAdapterPlan* response, void* user_data);
int vx_api_export_quantized_trainer_weights(const VolvoxaiV1ExportQuantizedTrainerWeightsRequest* request,
    VolvoxaiV1QuantizedTrainerWeights* response, void* user_data);

/* TrainStep request/result projections, shared with the train-step debug
 * target. Types come from vx_training_lifecycle.h. */
struct VxTrainStepOptions;
struct VxTrainStepResult;
int vx_api_train_step_options(VxApiScratch* scratch, const VolvoxaiV1TrainStepRequest* request,
    struct VxTrainStepOptions* options, const SynurangLiteAllocator* allocator,
    VolvoxaiV1OperationReport** report);
int vx_api_training_result(VolvoxaiV1TrainStepResult* response,
    const struct VxTrainStepResult* result, const VxReport* report);

#endif
