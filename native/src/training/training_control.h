#ifndef VOLVOXAI_TRAINING_CONTROL_H
#define VOLVOXAI_TRAINING_CONTROL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef VX_TRAINING_CONTROL_CORE_API
#define VX_TRAINING_CONTROL_CORE_API
#endif

/*
 * Portable full-profile Trainer control state.
 *
 * These records contain only fixed-width integers. They deliberately do not
 * contain C pointers, size_t, FILE, pthread, device objects, or asynchronous
 * host state. Numerical weights, activations, gradients, and optimizer slots
 * remain owned by the selected backend. The browser adapter stores these
 * records behind generation-tagged integer handles; native callers may embed
 * the state directly inside their already-synchronized Trainer owner.
 */
#define VX_TRAINING_CONTROL_ABI_VERSION UINT32_C(1)
#define VX_TRAINING_CONTROL_STATE_MAGIC UINT32_C(0x31544356) /* "VCT1" */
#define VX_TRAINING_CONTROL_STEP_MAGIC UINT32_C(0x31534356)  /* "VCS1" */

typedef int32_t VxTrainingControlStatusV1;
enum {
    VX_TRAINING_CONTROL_OK = 0,
    VX_TRAINING_CONTROL_INVALID_ARGUMENT = -1,
    VX_TRAINING_CONTROL_INVALID_STATE = -2,
    VX_TRAINING_CONTROL_BUSY = -3,
    VX_TRAINING_CONTROL_CAPACITY_EXHAUSTED = -4,
    VX_TRAINING_CONTROL_TRANSITION_MISMATCH = -5,
    VX_TRAINING_CONTROL_OVERFLOW = -6
};

typedef uint32_t VxTrainingControlOperationV1;
enum {
    VX_TRAINING_CONTROL_OPERATION_NONE = 0u,
    VX_TRAINING_CONTROL_OPERATION_STEP = 1u,
    VX_TRAINING_CONTROL_OPERATION_COMMIT = 2u,
    VX_TRAINING_CONTROL_OPERATION_ROLLBACK = 3u,
    VX_TRAINING_CONTROL_OPERATION_RESET_ACCUMULATION = 4u
};

typedef struct VxTrainingControlStateV1 {
    uint32_t magic;
    uint32_t abi_version;
    uint64_t base_revision;
    uint64_t baseline_training_step;
    uint64_t training_step;
    uint64_t microbatch_id;
    uint64_t baseline_microbatch_id;
    uint64_t transition_id;
    uint64_t accumulated_microbatches;
    uint64_t accumulation_steps;
    uint32_t working_dirty;
    uint32_t reserved;
} VxTrainingControlStateV1;

/* A begin-step token is an immutable proposal, not published Trainer state. */
typedef struct VxTrainingControlStepV1 {
    uint32_t magic;
    uint32_t abi_version;
    uint64_t source_transition_id;
    uint64_t accumulation_steps;
    uint64_t starting_microbatches;
    uint32_t flush_accumulation;
    uint32_t reset_accumulation;
} VxTrainingControlStepV1;

#if defined(__cplusplus)
static_assert(sizeof(VxTrainingControlStateV1) == 80,
              "portable training-control state ABI drift");
static_assert(sizeof(VxTrainingControlStepV1) == 40,
              "portable training-control step ABI drift");
#else
_Static_assert(sizeof(VxTrainingControlStateV1) == 80,
               "portable training-control state ABI drift");
_Static_assert(sizeof(VxTrainingControlStepV1) == 40,
               "portable training-control step ABI drift");
#endif

VX_TRAINING_CONTROL_CORE_API uint32_t
vx_training_control_core_abi_version(void);

VX_TRAINING_CONTROL_CORE_API VxTrainingControlStatusV1
vx_training_control_state_init_v1(
    VxTrainingControlStateV1* state,
    uint64_t base_revision,
    uint64_t baseline_training_step);

VX_TRAINING_CONTROL_CORE_API VxTrainingControlStatusV1
vx_training_control_state_validate_v1(
    const VxTrainingControlStateV1* state);

/*
 * Step is intentionally split in two. begin_step validates the accumulation
 * window before numerical work. finish_step accepts only the one outcome that
 * follows from that window and produces a candidate state without changing
 * the source. The host publishes the candidate only after its private backend
 * state has been captured successfully.
 */
VX_TRAINING_CONTROL_CORE_API VxTrainingControlStatusV1
vx_training_control_step_begin_v1(
    const VxTrainingControlStateV1* source,
    uint64_t accumulation_steps,
    uint32_t flush_accumulation,
    uint32_t reset_accumulation,
    VxTrainingControlStepV1* step);

VX_TRAINING_CONTROL_CORE_API VxTrainingControlStatusV1
vx_training_control_step_finish_v1(
    const VxTrainingControlStateV1* source,
    const VxTrainingControlStepV1* step,
    uint64_t observed_training_step,
    uint64_t observed_accumulation_step,
    uint32_t observed_accumulating,
    uint32_t observed_update_applied,
    VxTrainingControlStateV1* candidate);

VX_TRAINING_CONTROL_CORE_API VxTrainingControlStatusV1
vx_training_control_prepare_commit_v1(
    const VxTrainingControlStateV1* source,
    VxTrainingControlStateV1* candidate);

VX_TRAINING_CONTROL_CORE_API VxTrainingControlStatusV1
vx_training_control_prepare_rollback_v1(
    const VxTrainingControlStateV1* source,
    VxTrainingControlStateV1* candidate);

VX_TRAINING_CONTROL_CORE_API VxTrainingControlStatusV1
vx_training_control_prepare_reset_accumulation_v1(
    const VxTrainingControlStateV1* source,
    VxTrainingControlStateV1* candidate);

#ifdef __cplusplus
}
#endif

#endif
