#include "../runtime/decode_projection.h"

static int vx_webgpu_row_metadata(VxWebGpuNodePlan* plan, const int* values, size_t count) {
    if (!count || count > SIZE_MAX / sizeof(int)) return 0;
    size_t bytes = count * sizeof(int);
    int* slot = vx_webgpu_arena(bytes);
    if (!slot) return 0;
    memcpy(slot, values, bytes);
    if (!validating) vx_webgpu_mark_host(NULL, slot, bytes, 0);
    return vx_webgpu_bind(plan, slot, bytes);
}

static int vx_webgpu_row_transfer(VxWebGpuNodePlan* plan, const VxDecodeRowOperand* operand,
                                   const T* narrow, const VxDecodeRowSet* rows, int scatter) {
    size_t row_bytes = operand->width * operand->tensor->elem_size;
    int count = rows->lanes * (operand->prefix ? rows->key_capacity : 1);
    if (count <= 0 || row_bytes > UINT32_MAX / (uint32_t)count) return 0;
    int* indices = vx_webgpu_arena((size_t)count * sizeof(int));
    if (!indices) return 0;
    int paged = vx_paged_tensor_locked(operand->tensor);
    int status = operand->prefix
        ? vx_decode_row_set_prefix_rows(rows, operand->sequence, paged, indices)
        : vx_decode_row_set_write_rows(rows, operand->sequence, paged, indices);
    if (status != VX_DECODE_ROW_SET_OK) return 0;
    // The common CPU scatter receives parked flags separately. The device
    // transfer represents the same skipped lane with a negative row index.
    if (!operand->prefix)
        for (int lane = 0; lane < rows->lanes; lane++)
            if (rows->parked[lane]) indices[lane] = -1;
    size_t full_bytes = (size_t)operand->tensor->numel * operand->tensor->elem_size;
    size_t narrow_bytes = (size_t)count * row_bytes;
    for (int i = 0; i < count; i++) if (indices[i] >= 0 && (size_t)indices[i] >= full_bytes / row_bytes) return 0;
    if (!validating) vx_webgpu_mark_host(NULL, indices, (size_t)count * sizeof(int), 0);
    uint32_t params[4] = {(uint32_t)count, (uint32_t)row_bytes, (uint32_t)(full_bytes / row_bytes), (uint32_t)scatter};
    return vx_webgpu_bind(plan, scatter ? narrow->data : operand->tensor->data, scatter ? narrow_bytes : full_bytes) &&
        vx_webgpu_bind(plan, indices, (size_t)count * sizeof(int)) &&
        vx_webgpu_bind(plan, scatter ? operand->tensor->data : narrow->data, scatter ? full_bytes : narrow_bytes) &&
        vx_webgpu_params(plan, params, 4) &&
        vx_webgpu_grid(plan, VX_SHADER_INFERENCE_ROW_INDEX_TRANSFER, ((uint32_t)narrow_bytes + 255u) / 256u, 1, 1);
}

static int vx_webgpu_row_storage(const VxDecodeRowOperand* operand, const VxDecodeRowSet* rows, T* narrow) {
    *narrow = *operand->tensor;
    narrow->shape[operand->axis] = operand->prefix ? rows->key_capacity : 1;
    uint64_t elements = (uint64_t)rows->lanes * narrow->shape[operand->axis] * operand->width;
    if (!elements || elements > INT32_MAX || elements > UINT32_MAX / narrow->elem_size) return 0;
    narrow->numel = (long)elements;
    narrow->data = vx_webgpu_arena((size_t)elements * narrow->elem_size);
    return narrow->data != NULL;
}

static int vx_webgpu_plan_row(const Node* node, T* input, T* output,
                              VxWebGpuNodePlan* plans, uint32_t* count) {
    (void)input;
    VxDecodeRowProjection p;
    T saved[3], narrow[3], narrow_output, row_mask = {0};
    uint32_t used = 0;
    int accepted = 0;
    if (!vx_decode_projection_describe(node, g_active_row, &p) ||
        !vx_webgpu_row_storage(&p.output, &p.rows, &narrow_output)) return 0;
    for (int i = 0; i < p.input_count; i++) {
        saved[i] = *p.inputs[i].tensor;
        if (!vx_webgpu_row_storage(&p.inputs[i], &p.rows, &narrow[i]) ||
            !vx_webgpu_row_transfer(&plans[used++], &p.inputs[i], &narrow[i], &p.rows, 0)) return 0;
    }
    if (p.attention) {
        row_mask.dtype = T_I32; row_mask.elem_size = 4; row_mask.ndim = 2;
        row_mask.shape[0] = p.rows.lanes; row_mask.shape[1] = p.keys;
        row_mask.numel = (long)p.rows.lanes * p.keys;
        if (row_mask.numel <= 0 || (uint64_t)row_mask.numel > UINT32_MAX / 4u) return 0;
        row_mask.data = vx_webgpu_arena((size_t)row_mask.numel * 4u);
        VxWebGpuNodePlan* mask_plan = &plans[used++];
        uint32_t params[6] = {(uint32_t)p.rows.lanes, (uint32_t)p.keys,
            (uint32_t)vx_webgpu_port(node, VX_PORT_K)->shape[p.output.tensor->ndim - 2],
            (uint32_t)p.output.sequence, (uint32_t)p.mask_layout, (uint32_t)p.causal};
        int* positions = vx_webgpu_arena((size_t)p.rows.lanes * sizeof(int));
        int zero = 0;
        if (!positions) return 0;
        for (int lane = 0; lane < p.rows.lanes; lane++) positions[lane] = p.rows.parked[lane] ? -1 : p.rows.positions[lane];
        if (!row_mask.data ||
            !(p.mask ? vx_webgpu_bind(mask_plan, p.mask->data, (size_t)p.mask->numel * 4u) : vx_webgpu_row_metadata(mask_plan, &zero, 1)) ||
            !vx_webgpu_row_metadata(mask_plan, positions, (size_t)p.rows.lanes) ||
            !vx_webgpu_bind(mask_plan, row_mask.data, (size_t)row_mask.numel * 4u) ||
            !vx_webgpu_params(mask_plan, params, 6) ||
            !vx_webgpu_grid(mask_plan, VX_SHADER_INFERENCE_DECODE_KEEP_MASK, ((uint32_t)row_mask.numel + 63u) / 64u, 1, 1)) return 0;
    }
    for (int i = 0; i < p.input_count; i++) *p.inputs[i].tensor = narrow[i];
    T* primary = vx_webgpu_port(node, VX_PORT_INPUT);
    if (!primary) primary = vx_webgpu_port(node, VX_PORT_X);
    if (!primary) primary = vx_webgpu_port(node, VX_PORT_DATA);
    if (!primary) primary = vx_webgpu_port(node, VX_PORT_A);
    if (p.view) {
        // A proved row view has identical bytes. Its original shape operands
        // describe the whole tensor and are not evaluated a second time.
        narrow_output.data = primary->data;
        accepted = 1;
    } else if (p.attention) {
        accepted = (node->operator_kind == VX_OP_Q_SDPA)
            ? vx_webgpu_plan_qsdpa_masked(node, &narrow_output, &plans[used++], &row_mask, 1)
            : vx_webgpu_plan_cross_sdpa_masked(node, &narrow_output, &plans[used++], &row_mask, 1);
    } else if (node->operator_kind == VX_OP_Q_LAYER_NORM) {
        uint32_t numerical = 0;
        accepted = vx_webgpu_plan_qnorm(node, primary, &narrow_output, &plans[used], &numerical, 0);
        used += numerical;
    } else accepted = vx_webgpu_plan_node(node, primary, &narrow_output, &plans[used++]);
    for (int i = 0; i < p.input_count; i++) *p.inputs[i].tensor = saved[i];
    if (!accepted ||
        !vx_webgpu_row_transfer(&plans[used++], &p.output, &narrow_output, &p.rows, 1)) return 0;
    *count = used;
    (void)output;
    return 1;
}

int vx_webgpu_node_row_compatible(const char* output_name, int row) {
    VxEngineState* state = vx_engine_state_current();
    if (!state->webgpu_state || row < 0) return 0;
    const Node* node = NULL;
    for (int i = 0; i < state->node_count; i++)
        if (!strcmp(state->nodes[i].out, output_name)) { node = &state->nodes[i]; break; }
    if (!node) return 0;
    VxWebGpuNodePlan plans[7] = {0};
    VxWebGpuSpan* spans = g_webgpu.spans;
    size_t span_capacity = g_webgpu.span_capacity;
    size_t span_count = g_webgpu.span_count, arena_used = g_webgpu_arena_used;
    int previous = validating, active_row = g_active_row, value_domain = g_bounded_gpu_value_domain_proven;
    int node_index = g_vx_runtime_node.idx;
    g_webgpu.spans = NULL; g_webgpu.span_capacity = 0;
    g_webgpu.span_count = 0; g_webgpu_arena_used = 0;
    validating = 1; g_active_row = row;
    g_vx_runtime_node.idx = (int)(node - g_n);
    // Input value domains were proved against full tensors before staging.
    // A dry-run row buffer carries geometry and is never a host value source.
    g_bounded_gpu_value_domain_proven = 1;
    uint32_t count = 0;
    int accepted = vx_webgpu_plan_row(node, NULL, t_find(output_name), plans, &count);
    for (uint32_t i = 0; accepted && i < count; i++) accepted = vx_webgpu_seal(&plans[i]);
    g_bounded_gpu_value_domain_proven = value_domain;
    g_active_row = active_row; validating = previous;
    g_vx_runtime_node.idx = node_index;
    g_webgpu_arena_used = arena_used; g_webgpu.span_count = span_count;
    free(g_webgpu.spans);
    g_webgpu.spans = spans; g_webgpu.span_capacity = span_capacity;
    return accepted;
}
