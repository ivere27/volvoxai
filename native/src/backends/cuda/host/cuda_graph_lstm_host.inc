#include "../../lstm_plan.h"
#include "../lstm_launch_plan.h"

/* A tactic miss is allowed only before its first launch. The stable context
 * scratch is shared with earlier/later nodes on the same stream and is part
 * of the ordinary replay plan, never a cache of this request's activations. */
static int cuda_graph_lstm_prepared(const VxLstmPlan* source, CudaTensorView* views) {
    VxCudaLstmPreparedPlan plan;
    uint32_t time = source->params[0], batch = source->params[1];
    uint32_t width = source->params[2], hidden = source->params[3];
    uint32_t direction = source->params[4], flags = source->params[5];
    CudaContextState* state = cuda_context_state_get();
    if (!state || !state->matmul_workspace ||
        state->matmul_workspace > UINT64_MAX - VX_CUDA_CONTEXT_WORKSPACE_BYTES ||
        !vx_cuda_lstm_prepared_plan(time, batch, width, hidden, direction,
            VX_CUDA_CONTEXT_WORKSPACE_BYTES, &plan) ||
        !CUDA_FORWARD_FUNCTION(k_lstm_transpose) || !CUDA_FORWARD_FUNCTION(k_lstm_project) ||
        !CUDA_FORWARD_FUNCTION(k_lstm_recurrent) || !CUDA_FORWARD_FUNCTION(k_lstm_update)) return 0;
    for (unsigned i = 0; i < 7u; ++i)
        if (ranges_overlap((const void*)(uintptr_t)state->matmul_workspace,
                plan.bytes, (const void*)(uintptr_t)views[i].device, source->bytes[i])) return 0;
    uint32_t rows = plan.gate_width;
    if (!cuda_launch_dimensions_valid((width + 31u) / 32u, (rows + 31u) / 32u,
            plan.directions, 32u, 8u, 1u, 32u * 33u * sizeof(float)) ||
        !cuda_launch_dimensions_valid((hidden + 31u) / 32u, (rows + 31u) / 32u,
            plan.directions, 32u, 8u, 1u, 32u * 33u * sizeof(float)) ||
        !cuda_launch_dimensions_valid(plan.projection_groups, 1u, 1u, 128u, 1u, 1u, 0u) ||
        !cuda_launch_dimensions_valid(plan.gate_groups, 1u, 1u, 128u, 1u, 1u, 0u) ||
        !cuda_launch_dimensions_valid(plan.state_groups, 1u, 1u, 128u, 1u, 1u, 0u)) return 0;
    CUdeviceptr weight = state->matmul_workspace;
    CUdeviceptr recurrent = weight + plan.recurrent_offset;
    CUdeviceptr projection = weight + plan.projection_offset;
    CUdeviceptr gates = weight + plan.gates_offset;
    CUdeviceptr cells = weight + plan.cells_offset;
    void* transpose_w[] = {&views[1].device, &weight, &rows, &width};
    void* transpose_r[] = {&views[2].device, &recurrent, &rows, &hidden};
    if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_lstm_transpose),
            (width + 31u) / 32u, (rows + 31u) / 32u, plan.directions,
            32u, 8u, 1u, 0u, transpose_w) ||
        !cuda_launch(CUDA_FORWARD_FUNCTION(k_lstm_transpose),
            (hidden + 31u) / 32u, (rows + 31u) / 32u, plan.directions,
            32u, 8u, 1u, 0u, transpose_r)) return -1;
    void* project_args[] = {&views[0].device, &weight, &projection,
        &time, &batch, &width, &hidden, &plan.directions};
    if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_lstm_project), plan.projection_groups,
            1u, 1u, 128u, 1u, 1u, 0u, project_args)) return -1;
    for (uint32_t step = 0; step < time; ++step) {
        void* recurrent_args[] = {&projection, &recurrent, &views[3].device,
            &views[4].device, &views[6].device, &gates, &time, &batch,
            &hidden, &direction, &flags, &step};
        void* update_args[] = {&gates, &views[5].device, &cells, &views[6].device,
            &time, &batch, &hidden, &direction, &flags, &step};
        if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_lstm_recurrent), plan.gate_groups,
                1u, 1u, 128u, 1u, 1u, 0u, recurrent_args) ||
            !cuda_launch(CUDA_FORWARD_FUNCTION(k_lstm_update), plan.state_groups,
                1u, 1u, 128u, 1u, 1u, 0u, update_args)) return -1;
    }
    return 1;
}

int cuda_graph_lstm_f32(const float* input, const float* weight, const float* recurrent,
        const float* bias, const float* initial_h, const float* initial_c,
        float* output, uint32_t time, uint32_t batch, uint32_t width,
        uint32_t hidden, uint32_t direction) {
    VxLstmPlan plan;
    if (!vx_lstm_plan(input, weight, recurrent, bias, initial_h, initial_c,
            output, time, batch, width, hidden, direction, &plan)) return 0;
    CudaTensorView views[7];
    for (unsigned i = 0; i < 6; ++i)
        if (!graph_ensure_input_view(plan.inputs[i], plan.bytes[i],
                i >= 1u && i <= 3u, &views[i])) return 0;
    if (!graph_output_view(output, plan.bytes[6], &views[6])) return 0;
    int prepared = cuda_graph_lstm_prepared(&plan, views);
    if (prepared < 0) return 0;
    if (prepared) {
        graph_mark_device(views[6].slot);
        return 1;
    }
    void* args[13];
    for (unsigned i = 0; i < 7; ++i) args[i] = &views[i].device;
    for (unsigned i = 0; i < 6; ++i) args[i + 7u] = &plan.params[i];
    if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_lstm), plan.groups, 1u, 1u,
                     256u, 1u, 1u, 0u, args)) return 0;
    graph_mark_device(views[6].slot);
    return 1;
}
