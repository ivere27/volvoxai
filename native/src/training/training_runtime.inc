/*
 * Native training implementation boundary.
 *
 * This file is intentionally included by engine_runtime.c instead of compiled
 * as a separate translation unit. The backward planner operates on the
 * runtime's private static graph helpers and state; keeping one guarded include
 * preserves that encapsulation while making training ownership and profile
 * exclusion explicit. Do not include this file anywhere else.
 *
 * The implementation is split into three cohesive fragments, composed here in
 * dependency order into the single engine translation unit:
 *
 *   training_backward_cpu.inc  CPU reference backward kernels (grad_*, backward_*)
 *   training_gpu_plan.inc      GPU backward plan builder and dispatch/execute
 *   training_step.inc          backward driver, graph state, accumulation, ABI
 */

#include "../runtime/attention_mask.h"
#include "../kernels/training_kernels.h"

static int resize_f32_parameters(const Node* node, int* nearest_out);

#include "training_backward_cpu.inc"
#include "training_gpu_plan.inc"
#include "training_step.inc"
#if VOLVOXAI_ENABLE_WEBGPU
#include "training_webgpu.inc"
#endif
#include "dynamic_autograd.inc"
