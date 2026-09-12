#include "training_control.h"

#include <limits.h>
#include <stddef.h>

static void vx_training_control_zero(void* destination, size_t bytes) {
    volatile unsigned char* output = (volatile unsigned char*)destination;
    while (bytes--) *output++ = 0u;
}

static void vx_training_control_copy(void* destination,
                                     const void* source,
                                     size_t bytes) {
    volatile unsigned char* output = (volatile unsigned char*)destination;
    const volatile unsigned char* input =
        (const volatile unsigned char*)source;
    while (bytes--) *output++ = *input++;
}

VX_TRAINING_CONTROL_CORE_API uint32_t
vx_training_control_core_abi_version(void) {
    return VX_TRAINING_CONTROL_ABI_VERSION;
}

VX_TRAINING_CONTROL_CORE_API VxTrainingControlStatusV1
vx_training_control_state_validate_v1(
        const VxTrainingControlStateV1* state) {
    if (!state) return VX_TRAINING_CONTROL_INVALID_ARGUMENT;
    if (state->magic != VX_TRAINING_CONTROL_STATE_MAGIC ||
        state->abi_version != VX_TRAINING_CONTROL_ABI_VERSION ||
        state->reserved != 0u || state->base_revision == 0u ||
        state->working_dirty > 1u ||
        state->training_step < state->baseline_training_step ||
        state->microbatch_id < state->baseline_microbatch_id ||
        (!state->working_dirty &&
         state->training_step != state->baseline_training_step) ||
        (state->working_dirty &&
         state->training_step == state->baseline_training_step))
        return VX_TRAINING_CONTROL_INVALID_STATE;
    if (state->accumulated_microbatches == 0u) {
        if (state->accumulation_steps != 0u)
            return VX_TRAINING_CONTROL_INVALID_STATE;
    } else if (state->accumulation_steps <= 1u ||
               state->accumulated_microbatches >= state->accumulation_steps) {
        return VX_TRAINING_CONTROL_INVALID_STATE;
    }
    return VX_TRAINING_CONTROL_OK;
}

VX_TRAINING_CONTROL_CORE_API VxTrainingControlStatusV1
vx_training_control_state_init_v1(
        VxTrainingControlStateV1* state,
        uint64_t base_revision,
        uint64_t baseline_training_step) {
    if (!state || base_revision == 0u)
        return VX_TRAINING_CONTROL_INVALID_ARGUMENT;
    vx_training_control_zero(state, sizeof(*state));
    state->magic = VX_TRAINING_CONTROL_STATE_MAGIC;
    state->abi_version = VX_TRAINING_CONTROL_ABI_VERSION;
    state->base_revision = base_revision;
    state->baseline_training_step = baseline_training_step;
    state->training_step = baseline_training_step;
    return VX_TRAINING_CONTROL_OK;
}

static VxTrainingControlStatusV1 vx_training_control_next_transition(
        VxTrainingControlStateV1* state) {
    if (state->transition_id == UINT64_MAX)
        return VX_TRAINING_CONTROL_OVERFLOW;
    state->transition_id++;
    return VX_TRAINING_CONTROL_OK;
}

VX_TRAINING_CONTROL_CORE_API VxTrainingControlStatusV1
vx_training_control_step_begin_v1(
        const VxTrainingControlStateV1* source,
        uint64_t accumulation_steps,
        uint32_t flush_accumulation,
        uint32_t reset_accumulation,
        VxTrainingControlStepV1* step) {
    VxTrainingControlStatusV1 status =
        vx_training_control_state_validate_v1(source);
    uint64_t starting_microbatches;
    if (status != VX_TRAINING_CONTROL_OK) return status;
    if (!step || accumulation_steps == 0u || flush_accumulation > 1u ||
        reset_accumulation > 1u)
        return VX_TRAINING_CONTROL_INVALID_ARGUMENT;
    starting_microbatches = reset_accumulation
        ? 0u : source->accumulated_microbatches;
    if (starting_microbatches != 0u &&
        source->accumulation_steps != accumulation_steps)
        return VX_TRAINING_CONTROL_TRANSITION_MISMATCH;
    if (starting_microbatches == UINT64_MAX)
        return VX_TRAINING_CONTROL_OVERFLOW;
    vx_training_control_zero(step, sizeof(*step));
    step->magic = VX_TRAINING_CONTROL_STEP_MAGIC;
    step->abi_version = VX_TRAINING_CONTROL_ABI_VERSION;
    step->source_transition_id = source->transition_id;
    step->accumulation_steps = accumulation_steps;
    step->starting_microbatches = starting_microbatches;
    step->flush_accumulation = flush_accumulation;
    step->reset_accumulation = reset_accumulation;
    return VX_TRAINING_CONTROL_OK;
}

VX_TRAINING_CONTROL_CORE_API VxTrainingControlStatusV1
vx_training_control_step_finish_v1(
        const VxTrainingControlStateV1* source,
        const VxTrainingControlStepV1* step,
        uint64_t observed_training_step,
        uint64_t observed_accumulation_step,
        uint32_t observed_accumulating,
        uint32_t observed_update_applied,
        VxTrainingControlStateV1* candidate) {
    VxTrainingControlStatusV1 status =
        vx_training_control_state_validate_v1(source);
    uint64_t expected_microbatches;
    uint32_t expected_accumulating;
    if (status != VX_TRAINING_CONTROL_OK) return status;
    if (!step || !candidate ||
        step->magic != VX_TRAINING_CONTROL_STEP_MAGIC ||
        step->abi_version != VX_TRAINING_CONTROL_ABI_VERSION ||
        step->source_transition_id != source->transition_id ||
        step->accumulation_steps == 0u ||
        step->flush_accumulation > 1u || step->reset_accumulation > 1u ||
        observed_accumulating > 1u || observed_update_applied > 1u)
        return VX_TRAINING_CONTROL_INVALID_ARGUMENT;
    if ((!step->reset_accumulation &&
         step->starting_microbatches != source->accumulated_microbatches) ||
        (step->reset_accumulation && step->starting_microbatches != 0u) ||
        (step->starting_microbatches != 0u &&
         source->accumulation_steps != step->accumulation_steps))
        return VX_TRAINING_CONTROL_TRANSITION_MISMATCH;
    if (step->starting_microbatches == UINT64_MAX)
        return VX_TRAINING_CONTROL_OVERFLOW;
    expected_microbatches = step->starting_microbatches + 1u;
    expected_accumulating = !step->flush_accumulation &&
        expected_microbatches < step->accumulation_steps;
    if (observed_accumulation_step != expected_microbatches ||
        observed_accumulating != expected_accumulating ||
        (observed_accumulating && observed_update_applied) ||
        (observed_update_applied
             ? observed_training_step <= source->training_step
             : observed_training_step != source->training_step))
        return VX_TRAINING_CONTROL_TRANSITION_MISMATCH;
    vx_training_control_copy(candidate, source, sizeof(*candidate));
    if (candidate->microbatch_id == UINT64_MAX)
        return VX_TRAINING_CONTROL_OVERFLOW;
    candidate->microbatch_id++;
    if (observed_accumulating) {
        candidate->accumulated_microbatches = expected_microbatches;
        candidate->accumulation_steps = step->accumulation_steps;
    } else {
        candidate->accumulated_microbatches = 0u;
        candidate->accumulation_steps = 0u;
    }
    if (observed_update_applied) {
        candidate->training_step = observed_training_step;
        candidate->working_dirty = 1u;
    }
    return vx_training_control_next_transition(candidate);
}

VX_TRAINING_CONTROL_CORE_API VxTrainingControlStatusV1
vx_training_control_prepare_commit_v1(
        const VxTrainingControlStateV1* source,
        VxTrainingControlStateV1* candidate) {
    VxTrainingControlStatusV1 status =
        vx_training_control_state_validate_v1(source);
    if (status != VX_TRAINING_CONTROL_OK) return status;
    if (!candidate) return VX_TRAINING_CONTROL_INVALID_ARGUMENT;
    if (source->accumulated_microbatches != 0u || !source->working_dirty)
        return VX_TRAINING_CONTROL_TRANSITION_MISMATCH;
    if (source->base_revision == UINT64_MAX)
        return VX_TRAINING_CONTROL_OVERFLOW;
    vx_training_control_copy(candidate, source, sizeof(*candidate));
    candidate->base_revision++;
    candidate->baseline_training_step = candidate->training_step;
    candidate->baseline_microbatch_id = candidate->microbatch_id;
    candidate->working_dirty = 0u;
    return vx_training_control_next_transition(candidate);
}

VX_TRAINING_CONTROL_CORE_API VxTrainingControlStatusV1
vx_training_control_prepare_rollback_v1(
        const VxTrainingControlStateV1* source,
        VxTrainingControlStateV1* candidate) {
    VxTrainingControlStatusV1 status =
        vx_training_control_state_validate_v1(source);
    if (status != VX_TRAINING_CONTROL_OK) return status;
    if (!candidate) return VX_TRAINING_CONTROL_INVALID_ARGUMENT;
    vx_training_control_copy(candidate, source, sizeof(*candidate));
    candidate->training_step = candidate->baseline_training_step;
    candidate->microbatch_id = candidate->baseline_microbatch_id;
    candidate->accumulated_microbatches = 0u;
    candidate->accumulation_steps = 0u;
    candidate->working_dirty = 0u;
    return vx_training_control_next_transition(candidate);
}

VX_TRAINING_CONTROL_CORE_API VxTrainingControlStatusV1
vx_training_control_prepare_reset_accumulation_v1(
        const VxTrainingControlStateV1* source,
        VxTrainingControlStateV1* candidate) {
    VxTrainingControlStatusV1 status =
        vx_training_control_state_validate_v1(source);
    if (status != VX_TRAINING_CONTROL_OK) return status;
    if (!candidate) return VX_TRAINING_CONTROL_INVALID_ARGUMENT;
    vx_training_control_copy(candidate, source, sizeof(*candidate));
    candidate->accumulated_microbatches = 0u;
    candidate->accumulation_steps = 0u;
    return vx_training_control_next_transition(candidate);
}
