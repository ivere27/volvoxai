#ifndef VOLVOXAI_PUBLIC_API_INTERNAL_H
#define VOLVOXAI_PUBLIC_API_INTERNAL_H

#include "volvoxai.h"
#include "engine_core.h"

typedef struct VxWeightRevisionRecord VxWeightRevisionRecord;
typedef struct VxEngineState VxEngineState;

#if defined(__GNUC__) && !defined(_WIN32)
#define VX_PUBLIC_INTERNAL __attribute__((visibility("hidden")))
#else
#define VX_PUBLIC_INTERNAL
#endif

#if VOLVOXAI_ENABLE_TRAINING
VX_PUBLIC_INTERNAL VxStatus vx_model_internal_accept_full_owner(
    VxModel* model,
    VxWeightRevisionRecord** out_revision,
    VxReport* report);
VX_PUBLIC_INTERNAL void vx_model_internal_release_weight_revision(
    VxWeightRevisionRecord* revision);
VX_PUBLIC_INTERNAL VxStatus vx_model_internal_create_authoring_engine(
    VxModel* model,
    VxWeightRevisionRecord* exact_revision,
    VolvoxAIEngineBackend backend,
    VxEngineState** out_state,
    VxReport* report);
VX_PUBLIC_INTERNAL void vx_model_internal_destroy_authoring_engine(
    VxEngineState* state);
VX_PUBLIC_INTERNAL size_t vx_model_internal_input_count(const VxModel* model);
VX_PUBLIC_INTERNAL VxStatus vx_model_internal_input_spec(
    const VxModel* model,
    size_t index,
    VxTensorSpec* spec);
VX_PUBLIC_INTERNAL const char* vx_model_internal_logical_fingerprint(
    const VxModel* model);
/* Validate one complete shaped batch, resolve the logical graph, optionally
 * require an existing accumulation signature, and atomically bind the private
 * authoring engine. `out_shape_signature` is heap-owned by the caller. */
VX_PUBLIC_INTERNAL VxStatus vx_model_internal_bind_authoring_inputs(
    VxModel* model,
    VxEngineState* state,
    VolvoxAIEngineBackend backend,
    const VxTensorBinding* inputs,
    size_t input_count,
    const char* required_shape_signature,
    char** out_shape_signature,
    VxReport* report);
VX_PUBLIC_INTERNAL VxStatus vx_model_internal_validate_authoring_revision(
    VxModel* model,
    VxWeightRevisionRecord* exact_revision,
    uint64_t adapter_id,
    uint64_t adapter_revision,
    VxReport* report);
VX_PUBLIC_INTERNAL VxStatus vx_model_internal_prepare_weight_revision(
    VxModel* model,
    const char* const* weight_paths,
    size_t weight_path_count,
    VxWeightRevisionRecord** out_revision,
    VxReport* report);
VX_PUBLIC_INTERNAL VxStatus vx_model_internal_publish_weight_revision(
    VxModel* model,
    VxWeightRevisionRecord* expected,
    VxWeightRevisionRecord* successor,
    VxReport* report);
#endif

#undef VX_PUBLIC_INTERNAL

#endif /* VOLVOXAI_PUBLIC_API_INTERNAL_H */
