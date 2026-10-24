/* Resident CUDA projection of the common decode row set. The five staging
 * spans are reserved with the context maximum physical domain. Host accesses
 * below are exclusively shapes, row indices, keep masks and position IDs;
 * activation values are gathered, computed and scattered on the device. */
#if VOLVOXAI_ENABLE_CUDA
/* The owning metadata span is shared by masks, positions and scratch indices.
 * Validation may write indices without any device command. Forget cached
 * controls if that scratch range overlaps them, even if no upload occurred. */
static void vx_decode_cuda_metadata_write(int32_t* pointer, size_t words) {
    VxEngineState* state = vx_engine_state_current();
    uintptr_t base = (uintptr_t)state->decode_metadata_stage;
    uintptr_t address = (uintptr_t)pointer;
    if (address < base || (address - base) % sizeof(int32_t) ||
        address - base > state->decode_metadata_stage_bytes ||
        words > (state->decode_metadata_stage_bytes - (address - base)) / sizeof(int32_t)) {
        state->decode_cuda_mask_valid = state->decode_cuda_positions_valid = 0;
        return;
    }
    size_t begin = (address - base) / sizeof(int32_t), end = begin + words;
    if (begin < state->decode_cuda_mask_count && end > 0)
        state->decode_cuda_mask_valid = 0;
    if (begin < state->decode_cuda_positions_offset + state->decode_cuda_positions_count &&
        end > state->decode_cuda_positions_offset) state->decode_cuda_positions_valid = 0;
}

/* Add3 is the one CUDA fusion with an extra activation port in this row
 * domain. Preserve the ordinary fused execution, projecting its private third
 * operand just like its two declared inputs. Other private fusions refuse
 * during no-command admission rather than silently reading an unprojected row. */
static int vx_decode_cuda_projection_describe_uncached(Node* node, int idx, int row,
    VxDecodeRowProjection* projection) {
    if (!node || idx < 0 || idx >= g_nn || node->fuse_relu6 ||
        !vx_decode_projection_describe(node, row, projection)) return 0;
    const GraphNodeFusion* fusion = &g_node_fusion[idx];
    if (!fusion->id) return 1;
    if (fusion->id != GRAPH_FUSION_CHAINED_ELEMENTWISE || node->operator_kind != VX_OP_ADD ||
        node->nin != 2 || fusion->peer_idx <= idx || fusion->peer_idx >= g_nn ||
        !fusion->aux_tensor[0] || !fusion->source_tensor[0] ||
        vx_node_param_i32(node, VX_NODE_PARAM_RELU, 0) != 0) return 0;
    Node* peer = &g_n[fusion->peer_idx];
    T* auxiliary = NULL;
    for (int input = 0; input < peer->nin; input++)
        if (!strcmp(peer->ins[input].name, fusion->aux_tensor))
            auxiliary = vx_incremental_node_input_locked(peer, input);
    const VxIncrementalTensorBinding* prepared =
        vx_incremental_tensor_binding_locked(auxiliary, vx_engine_state_current());
    int producer = prepared ? prepared->producer : producer_index_any(fusion->aux_tensor);
    int auxiliary_occurrences = 0, source_occurrences = 0;
    if (!peer->skip || peer->operator_kind != VX_OP_ADD || peer->nin != 2 ||
        strcmp(peer->out, node->out) || !auxiliary || auxiliary->dtype != T_F32 ||
        !tensor_same_shape(auxiliary, projection->output.tensor) ||
        auxiliary->numel != projection->output.tensor->numel ||
        producer >= idx ||
        vx_node_param_i32(peer, VX_NODE_PARAM_RELU, 0) < 0 ||
        vx_node_param_i32(peer, VX_NODE_PARAM_RELU, 0) > 2) return 0;
    for (int input = 0; input < peer->nin; input++) {
        auxiliary_occurrences += !strcmp(peer->ins[input].name, fusion->aux_tensor);
        source_occurrences += !strcmp(peer->ins[input].name, fusion->source_tensor);
    }
    if (auxiliary_occurrences != 1 || source_occurrences != 1 ||
        !vx_decode_projection_add_input(projection, auxiliary, projection->output.sequence, 0)) return 0;
    for (int input = 0; input < projection->input_count; input++)
        if (projection->inputs[input].width != projection->output.width ||
            projection->inputs[input].tensor->dtype != T_F32 ||
            !tensor_same_shape(projection->inputs[input].tensor, projection->output.tensor)) return 0;
    return 1;
}

static VxIncrementalCudaOperand vx_decode_cuda_command_operand(
        const VxDecodeRowOperand* operand) {
    return (VxIncrementalCudaOperand){(int)(operand->tensor - g_t),
        operand->axis, operand->sequence, operand->prefix, operand->width};
}

static VxDecodeRowOperand vx_decode_cuda_command_operand_bind(
        const VxIncrementalCudaOperand* operand) {
    return (VxDecodeRowOperand){&g_t[operand->tensor_index], operand->axis,
        operand->sequence, operand->width, operand->prefix};
}

/* Only immutable non-varying operands can be compiled. Mutable conditioning
 * outside the projected operands keeps the ordinary dependency proof. Mask
 * and position values are explicit controls and are revalidated/rebound below. */
static int vx_decode_cuda_command_compile(Node* node, int idx,
        const VxDecodeRowProjection* p, VxIncrementalCudaCommand* command) {
    if (g_node_fusion[idx].id || node->fuse_relu6 || node->skip || node->disabled ||
        node->parsed_params.parsed != 1 || p->input_count < 1 || p->input_count > 3 ||
        (node->operator_kind == VX_OP_ARG_MAX && node->parsed_params.select_last_index != 0))
        return 0;
    for (int input = 0; input < node->nin; input++) {
        T* tensor = vx_incremental_node_input_locked(node, input);
        int varying = 0;
        for (int i = 0; i < p->input_count; i++) varying |= tensor == p->inputs[i].tensor;
        if (!varying && tensor != p->mask && node->ins[input].port != VX_PORT_POSITION_IDS &&
            !vx_incremental_tensor_is_model_weight_locked(tensor)) return 0;
    }
    memset(command, 0, sizeof(*command));
    command->operator_kind = node->operator_kind;
    command->output = vx_decode_cuda_command_operand(&p->output);
    for (int i = 0; i < p->input_count; i++)
        command->inputs[i] = vx_decode_cuda_command_operand(&p->inputs[i]);
    command->input_count = p->input_count;
    command->view = p->view; command->rope = p->rope;
    command->attention = p->attention; command->causal = p->causal;
    command->mask_index = p->mask ? (int)(p->mask - g_t) : -1;
    command->mask_layout = p->mask_layout;
    command->fixed_keys = p->keys;
    command->weight_index = command->bias_index = command->scale_index = -1;
    T* input = p->inputs[0].tensor;
    T* output = p->output.tensor;
    T* weight = nin(node, VX_PORT_WEIGHT);
    T* bias = nin(node, VX_PORT_BIAS);
    VxOperatorKind op = node->operator_kind;
    if (input->dtype == T_F32 && output->dtype == T_F32) {
        if ((op == VX_OP_LINEAR || op == VX_OP_MATMUL || op == VX_OP_GEMM) &&
            weight && weight->dtype == T_F32 && weight->elem_size == sizeof(float) &&
            (!bias || (bias->dtype == T_F32 && bias->elem_size == sizeof(float))) &&
            input->ndim > 0 && output->ndim > 0) {
            command->width = input->shape[input->ndim - 1];
            command->output_width = output->shape[output->ndim - 1];
            if (command->width > 0 && command->output_width > 0 &&
                (!bias || bias->numel == command->output_width) &&
                linear_node_weight_layout(node, input, output, weight, &command->layout) == 0 &&
                p->input_count == 1 && p->inputs[0].width == (size_t)command->width &&
                p->output.width == (size_t)command->output_width)
                command->numerical = 1;
        } else if (op == VX_OP_PACKED_LINEAR && p->rows.slots == 1 &&
            p->input_count == 1) {
            /* Reuse the authoritative packed shape, dtype, group and alias
             * checks. Every non-varying port was proved immutable above. */
            VxLLMOpArgs packed;
            T* scale = nin(node, VX_PORT_SCALE);
            if (weight && scale && input->elem_size == sizeof(float) &&
                output->elem_size == sizeof(float) && weight->elem_size == sizeof(int32_t) &&
                scale->elem_size == sizeof(float) &&
                (!bias || bias->elem_size == sizeof(float)) &&
                vx_engine_llm_node_args(node, &packed) && packed.rate == 128 &&
                p->inputs[0].width == (size_t)packed.key_dim &&
                p->output.width == (size_t)packed.value_dim) {
                command->width = packed.key_dim;
                command->output_width = packed.value_dim;
                command->weight_bits = packed.groups;
                command->activation_bits = packed.normalize;
                command->scale_index = (int)(scale - g_t);
                command->numerical = 1;
            }
        } else if (op == VX_OP_RMS_NORM && weight && weight->dtype == T_F32 &&
            weight->elem_size == sizeof(float)) {
            command->width = vx_node_param_i32(node, VX_NODE_PARAM_D_MODEL,
                input->shape[input->ndim - 1]);
            command->epsilon = vx_node_param_f32(node, VX_NODE_PARAM_EPS, 1.0e-6f);
            long rows = 0;
            if (normalization_dimensions(input, output, weight, NULL, command->width, &rows) == 0 &&
                command->epsilon > 0.f && isfinite(command->epsilon) && p->input_count == 1 &&
                p->inputs[0].width == p->output.width &&
                p->inputs[0].width % (size_t)command->width == 0)
                command->numerical = 1;
        } else if (op == VX_OP_SILU && p->input_count == 1 &&
            tensor_same_shape(input, output)) command->numerical = 1;
        else if ((op == VX_OP_ADD || op == VX_OP_MUL) && p->input_count == 2 &&
            tensor_same_shape(input, output) &&
            tensor_same_shape(p->inputs[1].tensor, output) &&
            p->inputs[1].tensor->dtype == T_F32) {
            command->relu = vx_node_param_i32(node, VX_NODE_PARAM_RELU, 0);
            if (op == VX_OP_MUL || (command->relu >= 0 && command->relu <= 2))
                command->numerical = 1;
        }
    } else if (op == VX_OP_EMBEDDING && input->dtype == T_I32 &&
        output->dtype == T_F32 && weight && weight->dtype == T_F32 &&
        weight->elem_size == sizeof(float) && weight->ndim == 2 &&
        weight->shape[0] > 0 && weight->shape[1] > 0 &&
        weight->numel == (long)weight->shape[0] * weight->shape[1] &&
        p->input_count == 1 && p->inputs[0].width == 1 &&
        p->output.width == (size_t)weight->shape[1]) {
        command->width = weight->shape[1]; command->output_width = weight->shape[0];
        command->numerical = 1;
    } else if (op == VX_OP_ARG_MAX && input->dtype == T_F32 && output->dtype == T_I32 &&
        input->ndim > 0 && p->input_count == 1 && p->output.width == 1 &&
        p->inputs[0].width == (size_t)input->shape[input->ndim - 1]) {
        int axis = vx_node_param_i32(node, VX_NODE_PARAM_AXIS, -1);
        if (axis < 0) axis += input->ndim;
        if (axis == input->ndim - 1) {
            command->width = input->shape[axis]; command->numerical = 1;
        }
    }
    if (weight) command->weight_index = (int)(weight - g_t);
    if (bias) command->bias_index = (int)(bias - g_t);
    command->valid = command->numerical || command->view || command->rope || command->attention;
    return command->valid;
}

/* Current output geometry shared by ordinary binding and replay admission.
 * Callers first validate the row-set pointers/slot count; no value or owner
 * proof is cached. */
static int vx_decode_cuda_row_geometry_current(const T* output, int sequence,
        size_t width, const VxDecodeRowSet* rows) {
    if ((uint64_t)rows->slots * sequence * width != (uint64_t)output->numel) return 0;
    for (int slot = 0; slot < rows->slots; slot++)
        if (!rows->empty[slot] && (rows->positions[slot] < 0 ||
            rows->positions[slot] >= sequence)) return 0;
    return 1;
}

static int vx_decode_cuda_command_bind(const VxIncrementalCudaCommand* command,
        int row, VxDecodeRowProjection* p) {
    if (row < 0 || row == INT32_MAX) return 0;
    memset(p, 0, sizeof(*p));
    if (g_decode_slots > 0) p->rows = g_decode_rows;
    else {
        p->scalar_position = row; p->scalar_length = row + 1;
        p->rows.slots = p->rows.live = 1; p->rows.key_capacity = row + 1;
        p->rows.positions = &p->scalar_position; p->rows.empty = &p->scalar_empty;
        p->rows.kv_lengths = &p->scalar_length; p->rows.pages = &p->scalar_pages;
        p->rows.unpaged = !vx_paged_bound_locked();
        if (!p->rows.unpaged && vx_paged_slot_pages_locked(0, &p->scalar_pages) != 1) return 0;
    }
    if (p->rows.slots <= 0 || !p->rows.positions || !p->rows.empty || !p->rows.kv_lengths)
        return 0;
    p->output = vx_decode_cuda_command_operand_bind(&command->output);
    /* Layout geometry includes the batch declaration. Changing it retains the
     * ordinary projection instead of silently reusing a different token axis. */
    if (!vx_decode_cuda_row_geometry_current(p->output.tensor,
            p->output.sequence, p->output.width, &p->rows)) return 0;
    p->input_count = command->input_count;
    for (int i = 0; i < p->input_count; i++)
        p->inputs[i] = vx_decode_cuda_command_operand_bind(&command->inputs[i]);
    p->view = command->view; p->rope = command->rope;
    p->attention = command->attention; p->causal = command->causal;
    p->mask = command->mask_index >= 0 ? &g_t[command->mask_index] : NULL;
    p->mask_layout = command->mask_layout;
    p->keys = p->causal ? p->rows.key_capacity : command->fixed_keys;
    return 1;
}

static int vx_decode_cuda_projection_describe(Node* node, int idx, int row,
        VxDecodeRowProjection* p) {
    VxIncrementalCudaCommand* command = NULL;
    if (g_incremental_plan.cuda_commands_active && idx >= 0 && idx < g_nn &&
        vx_incremental_node_bindings_current(node, vx_engine_state_current()) &&
        g_incremental_scratch.cuda_commands)
        command = &g_incremental_scratch.cuda_commands[idx];
    if (command && command->valid && command->operator_kind == node->operator_kind &&
        !node->skip && !node->disabled && !node->fuse_relu6 && !g_node_fusion[idx].id &&
        vx_decode_cuda_command_bind(command, row, p) &&
        /* A noncausal mask is immutable conditioning for this dirty closure;
         * its unchanged address cannot stand in for a current dependency proof. */
        (!p->attention || p->causal || !p->mask ||
         vx_decode_projection_invariant(node, p->mask))) return 1;
    if (!vx_decode_cuda_projection_describe_uncached(node, idx, row, p)) return 0;
    if (command) (void)vx_decode_cuda_command_compile(node, idx, p, command);
    return 1;
}

static int vx_decode_cuda_command_run(const VxIncrementalCudaCommand* command,
        const T* narrow, T* output) {
    const T* weight = command->weight_index >= 0 ? &g_t[command->weight_index] : NULL;
    const T* bias = command->bias_index >= 0 ? &g_t[command->bias_index] : NULL;
    switch (command->operator_kind) {
        case VX_OP_PACKED_LINEAR:
            if (!weight || command->scale_index < 0 ||
                narrow[0].numel != command->width || output->numel != command->output_width)
                return 0;
            return cuda_graph_packed_linear_decode_f32(narrow[0].data,
                (const int32_t*)weight->data, g_t[command->scale_index].data,
                bias ? bias->data : NULL, output->data, command->width,
                command->output_width, command->weight_bits, command->activation_bits) > 0;
        case VX_OP_LINEAR: case VX_OP_MATMUL: case VX_OP_GEMM:
            return cuda_graph_matmul_layout_f32(narrow[0].data, weight->data,
                bias ? bias->data : NULL, output->data, (int)(narrow[0].numel / command->width),
                command->width, command->output_width, command->layout);
        case VX_OP_RMS_NORM:
            return cuda_graph_rmsnorm_f32(narrow[0].data, weight->data, output->data,
                (int)(narrow[0].numel / command->width), command->width, command->epsilon);
        case VX_OP_SILU:
            return cuda_graph_silu_f32(narrow[0].data, output->data, output->numel);
        case VX_OP_ADD:
            return cuda_graph_add_relu_f32(narrow[0].data, narrow[1].data,
                output->data, output->numel, command->relu);
        case VX_OP_MUL: {
            const uint32_t stride = 1;
            return cuda_graph_binary_f32(narrow[0].data, narrow[0].numel,
                narrow[1].data, narrow[1].numel, output->data, output->numel,
                &stride, &stride, &stride, 1, 0);
        }
        case VX_OP_EMBEDDING:
            return cuda_graph_embedding_f32((const int32_t*)narrow[0].data,
                weight->data, output->data, (int)narrow[0].numel,
                command->width, command->output_width);
        case VX_OP_ARG_MAX:
            return cuda_graph_argmax_f32(narrow[0].data, (int32_t*)output->data,
                (uint32_t)output->numel, (uint32_t)command->width, 1u);
        default: return 0;
    }
}

static int vx_decode_cuda_host_control(const T* tensor) {
    return tensor && (tensor->is_graph_input ||
        vx_incremental_tensor_is_model_weight_locked(tensor));
}

static int vx_decode_cuda_storage(const VxDecodeRowOperand* operand,
    const VxDecodeRowSet* rows, int stage, T* narrow) {
    VxEngineState* state = vx_engine_state_current();
    if (!state->decode_row_stage_reserved || stage < 0 || stage >= 4 ||
        !operand->tensor || !operand->tensor->data ||
        (operand->tensor->dtype != T_F32 && operand->tensor->dtype != T_I32) ||
        operand->tensor->elem_size != sizeof(uint32_t) ||
        operand->axis < 0 || operand->axis >= operand->tensor->ndim ||
        rows->slots <= 0 || rows->key_capacity <= 0 || !operand->width) return 0;
    int sequence = operand->prefix ? rows->key_capacity : 1;
    uint64_t elements = (uint64_t)rows->slots;
    if ((uint64_t)sequence > INT32_MAX / elements) return 0;
    elements *= (uint64_t)sequence;
    if (operand->width > INT32_MAX / elements) return 0;
    elements *= operand->width;
    if (!elements || elements > SIZE_MAX / sizeof(uint32_t) ||
        !state->decode_numeric_stage[stage] ||
        elements * sizeof(uint32_t) > state->decode_numeric_stage_bytes[stage]) return 0;
    /* Admission needs the same storage proof without a projected descriptor. */
    if (narrow) {
        *narrow = *operand->tensor;
        narrow->shape[operand->axis] = sequence;
        narrow->numel = (long)elements;
        narrow->data = (float*)state->decode_numeric_stage[stage];
        narrow->owns = 0;
    }
    return 1;
}

static int vx_decode_cuda_selection_geometry(const VxDecodeRowOperand* operand,
    const VxDecodeRowSet* rows, uint32_t* count, uint32_t* indexed_rows,
    size_t* storage_bytes) {
    if (!operand || !operand->tensor || !rows || rows->slots <= 0 ||
        rows->key_capacity <= 0 || !rows->positions || !rows->empty ||
        !rows->kv_lengths || !rows->pages || operand->sequence < 0) return 0;
    uint64_t selected = (uint64_t)rows->slots * (operand->prefix ? rows->key_capacity : 1);
    if (!selected || selected > INT32_MAX || !operand->width ||
        operand->width > UINT32_MAX || operand->width > SIZE_MAX / sizeof(uint32_t) ||
        operand->tensor->numel <= 0 ||
        (uint64_t)operand->tensor->numel > SIZE_MAX / sizeof(uint32_t)) return 0;
    *storage_bytes = (size_t)operand->tensor->numel * sizeof(uint32_t);
    size_t row_bytes = operand->width * sizeof(uint32_t);
    if (*storage_bytes % row_bytes || !(*storage_bytes / row_bytes) ||
        *storage_bytes / row_bytes > INT32_MAX || selected > UINT32_MAX / operand->width) return 0;
    *count = (uint32_t)selected;
    *indexed_rows = (uint32_t)(*storage_bytes / row_bytes);
    return 1;
}

/* A live single slot has no ragged padding. Prove its physical interval
 * directly; contiguous page tables require one check per active page rather
 * than one index per token. Every operand still proves its own storage bound.
 * Page-table contents remain fixed while an admitted pass owns the model lock;
 * run_begin/invalidate clear these two bounded proofs before any later step. */
static int vx_decode_cuda_span(const VxDecodeRowOperand* operand,
    const VxDecodeRowSet* rows, uint32_t* first, uint32_t* count,
    uint32_t* indexed_rows, size_t* storage_bytes) {
    if (!vx_decode_cuda_selection_geometry(operand, rows, count, indexed_rows,
            storage_bytes) || rows->slots != 1 || rows->empty[0] ||
        rows->positions[0] < 0 || rows->kv_lengths[0] <= 0 ||
        (operand->prefix && rows->kv_lengths[0] != rows->key_capacity)) return 0;
    VxEngineState* state = vx_engine_state_current();
    int paged = vx_paged_tensor_locked(operand->tensor);
    const VxDecodeSlotPages* pages = &rows->pages[0];
    VxDecodeCudaSpanCache* cached = &state->decode_cuda_spans[operand->prefix != 0];
    int cacheable = state->incremental_plan.cuda_commands_active;
    if (cacheable && cached->valid && cached->paged == paged &&
        cached->sequence == operand->sequence &&
        cached->position == rows->positions[0] && cached->length == rows->kv_lengths[0] &&
        cached->key_capacity == rows->key_capacity && cached->page_table == pages->page_table &&
        cached->page_tokens == pages->page_tokens && cached->pages_per_slot == pages->pages_per_slot &&
        cached->count == *count && cached->first < *indexed_rows &&
        *count <= *indexed_rows - cached->first) {
        *first = cached->first;
        return 1;
    }
    uint64_t start = operand->prefix ? 0 : (uint32_t)rows->positions[0];
    if (paged && pages->page_table) {
        if (pages->page_tokens <= 0 || pages->pages_per_slot <= 0) return 0;
        uint64_t begin_page = start / (uint32_t)pages->page_tokens;
        uint64_t end_page = (start + *count - 1) / (uint32_t)pages->page_tokens;
        if (end_page >= (uint32_t)pages->pages_per_slot) return 0;
        int page = pages->page_table[begin_page];
        if (page < 0) return 0;
        for (uint64_t logical = begin_page + 1; logical <= end_page; logical++)
            if ((uint64_t)page + logical - begin_page > INT32_MAX ||
                pages->page_table[logical] != (int)((uint64_t)page + logical - begin_page)) return 0;
        start = (uint64_t)page * (uint32_t)pages->page_tokens +
            start % (uint32_t)pages->page_tokens;
    }
    if (start >= *indexed_rows || *count > *indexed_rows - start) return 0;
    *first = (uint32_t)start;
    if (cacheable) *cached = (VxDecodeCudaSpanCache){
        .page_table = pages->page_table, .valid = 1, .paged = paged,
        .sequence = operand->sequence, .position = rows->positions[0],
        .length = rows->kv_lengths[0], .key_capacity = rows->key_capacity,
        .page_tokens = pages->page_tokens, .pages_per_slot = pages->pages_per_slot,
        .first = *first, .count = *count};
    return 1;
}

/* Also run for every operand before the first command: invalid row/page
 * metadata must never partially update a retained numerical cache. */
static int vx_decode_cuda_indices(const VxDecodeRowOperand* operand,
    const VxDecodeRowSet* rows, int32_t* indices, uint32_t* count,
    uint32_t* indexed_rows, size_t* storage_bytes) {
    if (!indices || !vx_decode_cuda_selection_geometry(operand, rows, count,
            indexed_rows, storage_bytes)) return 0;
    VxEngineState* state = vx_engine_state_current();
    uintptr_t begin = (uintptr_t)indices, owner = (uintptr_t)state->decode_metadata_stage;
    if (!owner || begin < owner || begin - owner > state->decode_metadata_stage_bytes ||
        (size_t)*count > (state->decode_metadata_stage_bytes - (begin - owner)) / sizeof(int32_t)) return 0;
    int paged = vx_paged_tensor_locked(operand->tensor);
    if (paged)
        for (int slot = 0; slot < rows->slots; slot++)
            if (rows->pages[slot].page_table &&
                (rows->pages[slot].page_tokens <= 0 || rows->pages[slot].pages_per_slot <= 0)) return 0;
    vx_decode_cuda_metadata_write(indices, *count);
    int status = operand->prefix
        ? vx_decode_row_set_prefix_rows(rows, operand->sequence, paged, indices)
        : vx_decode_row_set_write_rows(rows, operand->sequence, paged, indices);
    if (status != VX_DECODE_ROW_SET_OK) return 0;
    if (!operand->prefix)
        for (int slot = 0; slot < rows->slots; slot++)
            if (rows->empty[slot]) indices[slot] = -1;
    for (uint32_t index = 0; index < *count; index++)
        if (indices[index] < -1 || (indices[index] >= 0 &&
            (uint32_t)indices[index] >= *indexed_rows)) return 0;
    return 1;
}

/* These numerical routes resolve interior pointers through CudaTensorView.
 * Slot-only routes must retain their independently reserved staging spans.
 * Add's broadcast route is slot-only, so admit only its equal-sized route. */
static int vx_decode_cuda_view_safe(const Node* node, const VxDecodeRowProjection* p) {
    if (p->view || p->rope || p->attention) return 1;
    VxOperatorKind op = node->operator_kind;
    if (op == VX_OP_ARG_MAX && p->input_count == 1) {
        const T* input = p->inputs[0].tensor;
        int axis = vx_node_param_i32(node, VX_NODE_PARAM_AXIS, -1);
        if (axis < 0) axis += input->ndim;
        return input->dtype == T_F32 && p->output.tensor->dtype == T_I32 &&
            input->ndim > 0 && axis == input->ndim - 1 &&
            !node->parsed_params.select_last_index && p->output.width == 1 &&
            p->inputs[0].width == (size_t)input->shape[axis];
    }
    if (op == VX_OP_ADD) {
        if (p->input_count < 2) return 0;
        for (int i = 0; i < p->input_count; i++)
            if (p->inputs[i].tensor->numel != p->output.tensor->numel ||
                p->inputs[i].width != p->output.width) return 0;
        return 1;
    }
    return op == VX_OP_LINEAR || op == VX_OP_MATMUL || op == VX_OP_GEMM || op == VX_OP_PACKED_LINEAR ||
        op == VX_OP_LAYER_NORM || op == VX_OP_RMS_NORM || op == VX_OP_EMBEDDING ||
        op == VX_OP_RELU || op == VX_OP_LEAKY_RELU || op == VX_OP_GELU ||
        op == VX_OP_SILU || op == VX_OP_SIGMOID || op == VX_OP_TANH ||
        op == VX_OP_HARD_SIGMOID || op == VX_OP_HARD_SWISH || op == VX_OP_CLIP;
}

static int vx_decode_cuda_contiguous_view(const VxDecodeRowOperand* operand,
    const VxDecodeRowSet* rows, int32_t* indices, T* narrow) {
    uint32_t count, indexed_rows, first;
    size_t storage_bytes;
    if (!vx_decode_cuda_span(operand, rows, &first, &count, &indexed_rows, &storage_bytes)) {
        if (!vx_decode_cuda_indices(operand, rows, indices, &count, &indexed_rows,
            &storage_bytes) || indices[0] < 0 ||
            count > indexed_rows - (uint32_t)indices[0]) return 0;
        first = (uint32_t)indices[0];
        for (uint32_t row = 1; row < count; row++)
            if (indices[row] < 0 || (uint32_t)indices[row] != first + row) return 0;
    }
    if ((uint64_t)narrow->numel != (uint64_t)count * operand->width) return 0;
    size_t offset = (size_t)first * operand->width * sizeof(uint32_t);
    size_t bytes = (size_t)narrow->numel * sizeof(uint32_t);
    if (offset > storage_bytes || bytes > storage_bytes - offset) return 0;
    narrow->data = (float*)((unsigned char*)operand->tensor->data + offset);
    return 1;
}

static int vx_decode_cuda_transfer(const VxDecodeRowOperand* operand,
    const VxDecodeRowSet* rows, const T* narrow, int32_t* indices, int scatter) {
    VxEngineState* state = vx_engine_state_current();
    uint32_t count, indexed_rows, first;
    size_t storage_bytes;
    size_t narrow_bytes = (size_t)narrow->numel * sizeof(uint32_t);
    if (vx_decode_cuda_span(operand, rows, &first, &count, &indexed_rows, &storage_bytes)) {
        int copied = scatter
            ? cuda_graph_row_span_transfer(narrow->data, narrow_bytes, first, count,
                operand->tensor->data, storage_bytes, (uint32_t)operand->width, indexed_rows, 1u)
            : cuda_graph_row_span_transfer(operand->tensor->data, storage_bytes, first, count,
                narrow->data, narrow_bytes, (uint32_t)operand->width, indexed_rows, 0u);
        if (copied >= 0) return copied;
    }
    if (!vx_decode_cuda_indices(operand, rows, indices, &count, &indexed_rows,
                               &storage_bytes)) return 0;
    /* Consecutive rows use a captured host offset, not device indices. Keep
     * mask/position uploads clean in that case. An indexed or overlapping
     * route still registers and dirties the complete owner before an interior
     * index view is used; the backend uploads the current indices itself. */
    int consecutive = indices[0] >= 0 && count <= indexed_rows - (uint32_t)indices[0];
    for (uint32_t row = 1; row < count && consecutive; row++)
        consecutive = indices[row] >= 0 && (uint32_t)indices[row] == (uint32_t)indices[0] + row;
    uintptr_t selected_address = (uintptr_t)operand->tensor->data;
    if (consecutive) selected_address += (size_t)indices[0] * operand->width * sizeof(uint32_t);
    uintptr_t packed_address = (uintptr_t)narrow->data;
    int overlap = selected_address != packed_address &&
        (selected_address <= packed_address
            ? narrow_bytes > packed_address - selected_address
            : narrow_bytes > selected_address - packed_address);
    if (!consecutive || overlap)
        cuda_graph_mark_host(state->decode_metadata_stage,
                             state->decode_metadata_stage_bytes, 0);
    return scatter
        ? cuda_graph_row_index_transfer(narrow->data, narrow_bytes, indices,
            count, operand->tensor->data, storage_bytes, (uint32_t)operand->width,
            indexed_rows, 1u)
        : cuda_graph_row_index_transfer(operand->tensor->data, storage_bytes,
            indices, count, narrow->data, narrow_bytes, (uint32_t)operand->width,
            indexed_rows, 0u);
}

static int vx_decode_cuda_attention(Node* node, const VxDecodeRowProjection* p,
    T* output, const int32_t* mask) {
    T* q = nin(node, VX_PORT_Q), *k = nin(node, VX_PORT_K), *v = nin(node, VX_PORT_V);
    if (node->operator_kind != VX_OP_CROSS_SDPA || !q || !k || !v ||
        q->dtype != T_F32 || k->dtype != T_F32 || v->dtype != T_F32 || output->dtype != T_F32)
        return -1;
    int width = q->shape[q->ndim - 1], kv_width = k->shape[k->ndim - 1];
    int heads = node->parsed_params.heads, kv_heads = node->parsed_params.kv_heads;
    if (heads <= 0 || kv_heads <= 0 || heads % kv_heads || width <= 0 ||
        width % heads || kv_width <= 0 || kv_width % kv_heads ||
        width / heads != kv_width / kv_heads || width / heads > 128 ||
        v->shape[v->ndim - 1] != kv_width || output->shape[output->ndim - 1] != width) return -1;
    float scale = node->parsed_params.has_scale ? node->parsed_params.scale : 1.f / sqrtf((float)(width / heads));
    if (!isfinite(scale)) return -1;
    /* Absolute query position and per-slot length were projected into mask.
     * A local query row numbered zero must not reapply causal clipping. */
    /* The projected mask is current host control, with one row per slot.
     * Preserve the masked backend's span and alias admission before eliding it. */
    if (!q->data || !k->data || !v->data || !output->data || !mask ||
        p->rows.slots <= 0 || p->keys <= 0) return -1;
    uint64_t mask_count = (uint64_t)(uint32_t)p->rows.slots * (uint32_t)p->keys;
    uint64_t output_count = (uint64_t)(uint32_t)p->rows.slots * (uint32_t)width;
    if (mask_count > UINT32_MAX || mask_count > LONG_MAX ||
        mask_count > SIZE_MAX / sizeof(int32_t) || output_count > UINT32_MAX ||
        output_count > SIZE_MAX / sizeof(float)) return -1;
    uintptr_t output_address = (uintptr_t)output->data, mask_address = (uintptr_t)mask;
    if (output_address > UINTPTR_MAX - output_count * sizeof(float) ||
        mask_address > UINTPTR_MAX - mask_count * sizeof(int32_t) ||
        (output_address >= mask_address &&
         output_address - mask_address < mask_count * sizeof(int32_t)) ||
        (mask_address >= output_address &&
         mask_address - output_address < output_count * sizeof(float))) return -1;
    long mask_elements = (long)mask_count;
    int mask_mode = ATTN_MASK_BATCH_KEY;
    int all_allowed = 1;
    for (long index = 0; index < mask_elements; index++)
        if (!mask[index]) { all_allowed = 0; break; }
    if (all_allowed) {
        mask = NULL;
        mask_elements = 0;
        mask_mode = ATTN_MASK_NONE;
    }
    return cuda_graph_cross_sdpa_gqa_f32(q->data, k->data, v->data, mask,
        mask_elements, output->data, 1, p->keys, width,
        kv_width, heads, kv_heads, width / heads, p->rows.slots, scale,
        0, mask_mode) == 1 ? 0 : -1;
}

/* The row planner calls the same no-command checks before input commit.
 * Unsupported controls, storage or private fusions cannot be discovered only
 * after preceding nodes have written their retained output rows. */
static int vx_decode_cuda_project_compatible(Node* node, int idx, VxDecodeRowProjection* p) {
    VxEngineState* state = vx_engine_state_current();
    VxSequenceRoPE rope = {0};
    int slots = p->rows.slots;
    int keys = p->attention ? p->keys : p->rows.key_capacity;
    if (!state->decode_row_stage_reserved || !state->decode_metadata_stage ||
        slots <= 0 || keys <= 0 || p->input_count < 1 || p->input_count > 3 ||
        (uint64_t)slots * keys > INT32_MAX || idx < 0 || idx >= g_nn ||
        (g_node_fusion[idx].id && g_node_fusion[idx].id != GRAPH_FUSION_CHAINED_ELEMENTWISE) ||
        node->fuse_relu6 ||
        (node->operator_kind == VX_OP_ARG_MAX && node->parsed_params.select_last_index != 0)) return 0;
    size_t count = (size_t)slots * keys;
    if (count > (SIZE_MAX / sizeof(int32_t) - (size_t)slots) / 2 ||
        (count * 2 + (size_t)slots) * sizeof(int32_t) > state->decode_metadata_stage_bytes) return 0;
    int32_t* indices = (int32_t*)state->decode_metadata_stage + count + slots;
    if (!vx_decode_cuda_storage(&p->output, &p->rows, 3, NULL)) return 0;
    for (int i = 0; i < p->input_count; i++) {
        if (!vx_decode_cuda_storage(&p->inputs[i], &p->rows, i, NULL) ||
            p->inputs[i].tensor == p->output.tensor) return 0;
    }
    /* Device-produced controls cannot be read through their stale host
     * placeholders. Public bindings and immutable weights are authoritative. */
    if (p->attention && (node->operator_kind != VX_OP_CROSS_SDPA ||
        (p->mask && (!vx_decode_cuda_host_control(p->mask) || p->mask->dtype != T_I32 ||
                    p->mask->elem_size != sizeof(int32_t) || !p->mask->data)))) return 0;
    if (p->rope) {
        T* original_positions = nin(node, VX_PORT_POSITION_IDS);
        if ((original_positions && !vx_decode_cuda_host_control(original_positions)) ||
            !vx_sequence_rope_parameters(node, p->inputs[0].tensor, p->output.tensor, &rope) ||
            rope.batch != slots || rope.theta < 1.f || rope.width / rope.heads > 128) return 0;
    }
    uint32_t selected, indexed_rows, first;
    size_t storage_bytes;
    for (int i = 0; i < p->input_count; i++)
        if (!vx_decode_cuda_span(&p->inputs[i], &p->rows, &first, &selected, &indexed_rows, &storage_bytes) &&
            !vx_decode_cuda_indices(&p->inputs[i], &p->rows, indices,
            &selected, &indexed_rows, &storage_bytes)) return 0;
    if (!vx_decode_cuda_span(&p->output, &p->rows, &first, &selected, &indexed_rows, &storage_bytes) &&
        !vx_decode_cuda_indices(&p->output, &p->rows, indices,
        &selected, &indexed_rows, &storage_bytes)) return 0;
    return 1;
}

static VxDecodeCudaMaskCache vx_decode_cuda_mask_key(const VxDecodeRowProjection* p) {
    VxDecodeCudaMaskCache key = {.valid = 1, .source = p->mask,
        .layout = p->mask_layout, .keys = p->keys,
        .source_keys = p->mask ? p->mask->shape[p->mask->ndim - 1] : p->keys,
        .sequence = p->output.sequence, .causal = p->causal,
        .position = p->rows.positions[0], .length = p->rows.kv_lengths[0]};
    if (p->mask) {
        key.binding.data = p->mask->data;
        key.binding.numel = p->mask->numel;
        key.binding.elem_size = p->mask->elem_size;
        key.binding.ndim = p->mask->ndim;
        key.binding.dtype = p->mask->dtype;
        key.binding.is_graph_input = p->mask->is_graph_input;
        memcpy(key.binding.shape, p->mask->shape, sizeof(key.binding.shape));
    }
    return key;
}

static int vx_decode_cuda_mask_key_equal(const VxDecodeCudaMaskCache* a,
    const VxDecodeCudaMaskCache* b) {
    return a->valid && a->source == b->source && a->layout == b->layout &&
        a->keys == b->keys && a->source_keys == b->source_keys &&
        a->sequence == b->sequence && a->causal == b->causal &&
        a->position == b->position && a->length == b->length &&
        a->binding.data == b->binding.data && a->binding.numel == b->binding.numel &&
        a->binding.elem_size == b->binding.elem_size && a->binding.ndim == b->binding.ndim &&
        a->binding.dtype == b->binding.dtype && a->binding.is_graph_input == b->binding.is_graph_input &&
        !memcmp(a->binding.shape, b->binding.shape, sizeof(a->binding.shape));
}

static int vx_decode_cuda_project(Node* node, int idx, int is_last, VxDecodeRowProjection* p) {
    if (!vx_decode_cuda_project_compatible(node, idx, p)) return -1;
    VxEngineState* state = vx_engine_state_current();
    T saved[3], narrow[3] = {0}, output = {0};
    VxSequenceRoPE rope = {0};
    int result = -1, slots = p->rows.slots;
    int keys = p->attention ? p->keys : p->rows.key_capacity;
    size_t count = (size_t)slots * keys;
    int32_t* mask = (int32_t*)state->decode_metadata_stage;
    int32_t* positions = mask + count;
    int32_t* indices = positions + slots;
    if (!vx_decode_cuda_storage(&p->output, &p->rows, 3, &output)) return -1;
    int view_safe = vx_decode_cuda_view_safe(node, p), direct_output = 0;
    for (int i = 0; i < p->input_count; i++) {
        saved[i] = *p->inputs[i].tensor;
        if (!vx_decode_cuda_storage(&p->inputs[i], &p->rows, i, &narrow[i])) return -1;
        if (view_safe)
            (void)vx_decode_cuda_contiguous_view(&p->inputs[i], &p->rows, indices, &narrow[i]);
    }
    if (view_safe) {
        T direct = output;
        if (vx_decode_cuda_contiguous_view(&p->output, &p->rows, indices, &direct)) {
            /* The staging path separates numerical inputs from outputs even
             * when the graph arena aliases distinct tensors. Preserve that
             * protection instead of exposing an in-place kernel accidentally. */
            size_t bytes = (size_t)direct.numel * sizeof(uint32_t);
            int overlap = 0;
            for (int i = 0; i < p->input_count; i++) {
                uintptr_t a = (uintptr_t)direct.data, b = (uintptr_t)narrow[i].data;
                size_t input_bytes = (size_t)narrow[i].numel * sizeof(uint32_t);
                overlap |= a <= b ? bytes > b - a : input_bytes > a - b;
            }
            if (!overlap) { output = direct; direct_output = 1; }
        }
    }
    if (p->rope && !vx_sequence_rope_parameters(node, p->inputs[0].tensor, p->output.tensor, &rope)) return -1;
    if (p->attention) {
        /* Public controls/immutable weights are authoritative and fixed while
         * this non-stepwise pass holds the model lock. Stepwise and ordinary
         * row calls still inspect their current values on every invocation. */
        int cacheable = state->incremental_plan.cuda_commands_active && slots == 1 &&
            !p->rows.empty[0];
        VxDecodeCudaMaskCache key = {0};
        if (cacheable) key = vx_decode_cuda_mask_key(p);
        int reused = cacheable && state->decode_cuda_mask_valid &&
            state->decode_cuda_mask_count == count &&
            vx_decode_cuda_mask_key_equal(&state->decode_cuda_mask_cache, &key);
        if (!reused) {
            int changed = !state->decode_cuda_mask_valid || state->decode_cuda_mask_count != count;
            /* Writing this mask can cover a previous node's positions when its
             * key count differs. This invalidates only the overlapping cache. */
            vx_decode_cuda_metadata_write(mask, count);
            int source_keys = p->mask ? p->mask->shape[p->mask->ndim - 1] : p->keys;
            for (int slot = 0; slot < slots; slot++)
                for (int key = 0; key < keys; key++) {
                    int32_t value = !p->rows.empty[slot] &&
                        (!p->causal || key < p->rows.kv_lengths[slot]) &&
                        attention_mask_allowed(p->mask, p->mask_layout, slot,
                            p->rows.positions[slot], key, p->output.sequence, source_keys, 0);
                    changed |= mask[(size_t)slot * keys + key] != value;
                    mask[(size_t)slot * keys + key] = value;
                }
            if (changed) cuda_graph_mark_host(state->decode_metadata_stage, state->decode_metadata_stage_bytes, 0);
            state->decode_cuda_mask_count = count;
            state->decode_cuda_mask_valid = 1;
            state->decode_cuda_mask_cache = key;
        }
    }
    if (p->rope) {
        int changed = !state->decode_cuda_positions_valid ||
            state->decode_cuda_positions_offset != count ||
            state->decode_cuda_positions_count != (size_t)slots;
        int wrote = 0;
        for (int slot = 0; slot < slots; slot++) {
            int32_t value = 0;
            int row = p->rows.positions[slot];
            if (!p->rows.empty[slot] && rope.positions) {
                size_t index = rope.position_mode == 1 ? (size_t)row : (size_t)slot * rope.sequence + row;
                memcpy(&value, (const unsigned char*)rope.positions->data + index * sizeof(int32_t), sizeof(int32_t));
            } else if (!p->rows.empty[slot]) value = rope.position_offset + row;
            if (positions[slot] != value) {
                if (!wrote) vx_decode_cuda_metadata_write(positions, (size_t)slots);
                wrote = changed = 1;
                positions[slot] = value;
            }
        }
        if (changed) cuda_graph_mark_host(state->decode_metadata_stage, state->decode_metadata_stage_bytes, 0);
        state->decode_cuda_positions_offset = count;
        state->decode_cuda_positions_count = (size_t)slots;
        state->decode_cuda_positions_valid = 1;
    }
    /* Prepare the entire owning output before the first partial write. A
     * same-address gather performs coherence without a copy or index upload;
     * the later same-address scatter likewise needs no device command. */
    if (direct_output && !vx_decode_cuda_transfer(&p->output, &p->rows,
            &output, indices, 0)) return -1;
    for (int i = 0; i < p->input_count; i++)
        if (!vx_decode_cuda_transfer(&p->inputs[i], &p->rows, &narrow[i], indices, 0)) return -1;
    if (p->rope) {
        result = cuda_graph_rope_heads_f32(narrow[0].data, positions, slots,
            output.data, slots, 1, rope.width, rope.heads, rope.rotary_width,
            rope.theta, 0, rope.interleaved, 2) ? 0 : -1;
    } else {
        const VxIncrementalCudaCommand* command =
            g_incremental_plan.cuda_commands_active && g_incremental_scratch.cuda_commands &&
            idx >= 0 && idx < g_nn ? &g_incremental_scratch.cuda_commands[idx] : NULL;
        if (command && command->valid && command->numerical &&
            command->operator_kind == node->operator_kind) {
            result = vx_decode_cuda_command_run(command, narrow, &output) ? 0 : -1;
            if (result == 0 && g_vx_cuda_route_tracking) cuda_graph_note_node_route(1);
        } else {
            for (int i = 0; i < p->input_count; i++) *p->inputs[i].tensor = narrow[i];
            T saved_output = *p->output.tensor;
            *p->output.tensor = output;
            int active = g_active_row, decode_slots = g_decode_slots, prefix = g_prefix_rows;
            int execution = g_execution_row, paged = state->paged.bound;
            g_active_row = -1; g_decode_slots = 0; g_prefix_rows = 0; g_execution_row = -1;
            state->paged.bound = 0;
            if (p->view) result = cuda_graph_copy_32(narrow[0].data, output.data, output.numel) ? 0 : -1;
            else if (p->attention) result = vx_decode_cuda_attention(node, p, &output, mask);
            else {
                result = run_node(node, idx, is_last);
                const char* route = idx >= 0 && (size_t)idx < g_node_capacity ? state->runtime_route_backend[idx] : NULL;
                if (result == 0 && (!route || strncmp(route, "cuda", 4))) result = -1;
            }
            g_active_row = active; g_decode_slots = decode_slots; g_prefix_rows = prefix; g_execution_row = execution;
            state->paged.bound = paged;
            *p->output.tensor = saved_output;
            for (int i = 0; i < p->input_count; i++) *p->inputs[i].tensor = saved[i];
        }
    }
    if (result == 0 && !vx_decode_cuda_transfer(&p->output, &p->rows, &output, indices, 1)) result = -1;
    return result;
}

/* A committed single-row head chain preserves all three logical owners. It
 * borrows current typed projections, never rewrites topology or skip flags.
 * Refusals precede the backend's metadata/coherence boundary. */
int vx_runtime_cuda_head_rms_rope_triple_locked(int idx, int row) {
    VxEngineState* state = vx_engine_state_current();
    if (!state || !g_use_cuda || state->debug || state->preserve_node_boundaries ||
        !state->shape_policy.retain_activations || vx_trace_nodes(state->profiling) ||
        row < 0 || row == INT32_MAX || idx < 0 || idx >= g_nn - 2 ||
        g_decode_slots != 1 || !g_incremental_plan.cuda_commands_active ||
        !g_incremental_scratch.cuda_commands ||
        g_incremental_scratch.node_capacity < (size_t)g_nn) return 0;
#if VOLVOXAI_ENABLE_TRAINING
    if (g_native_training_mode) return 0;
#endif
    const VxDecodeRowSet* rows = &g_decode_rows;
    if (rows->slots != 1 || rows->live != 1 || !rows->positions || !rows->empty ||
        !rows->kv_lengths || !rows->pages || rows->empty[0] || rows->positions[0] != row ||
        rows->kv_lengths[0] != row + 1 || rows->key_capacity != row + 1) return 0;
    Node* nodes[3] = {&g_n[idx], &g_n[idx + 1], &g_n[idx + 2]};
    const VxOperatorKind kinds[3] = {VX_OP_RMS_NORM, VX_OP_RESHAPE, VX_OP_ROPE};
    VxDecodeRowProjection p[3];
    const int reshape_alias = nodes[1]->skip;
    for (int i = 0; i < 3; i++) {
        Node* node = nodes[i];
        const VxIncrementalCudaCommand* command = &g_incremental_scratch.cuda_commands[idx + i];
        if (node->operator_kind != kinds[i] || (node->skip && i != 1) ||
            node->disabled || node->fuse_relu6 ||
            g_node_fusion[idx + i].id || node->parsed_params.parsed != 1 ||
            !vx_incremental_node_bindings_current(node, state) || !node->ins || !node->outs ||
            node->nin < 1 || node->nin > 2 || (i == 0 && node->nin != 2) ||
            (i == 1 && node->nin != 1) ||
            node->nout != 1 || node->outs[0].port != VX_PORT_OUT ||
            strcmp(node->out, node->outs[0].name)) return 0;
        /* The optimizer can elide this contiguous Reshape while retaining its
         * logical descriptor. Re-prove that view; a skipped node has no valid
         * numerical command to borrow. All other nodes keep their typed proof. */
        if (i == 1 && reshape_alias) {
            if (!vx_decode_cuda_projection_describe_uncached(node, idx + i, row, &p[i]) ||
                !p[i].view || p[i].input_count != 1 || p[i].attention || p[i].rope ||
                p[i].inputs[0].prefix || p[i].output.prefix ||
                p[i].inputs[0].axis != 1 || p[i].output.axis != 1) return 0;
        } else if (!command->valid ||
            command->operator_kind != kinds[i] || command->input_count != 1 ||
            command->inputs[0].tensor_index < 0 || command->inputs[0].tensor_index >= g_nt ||
            command->output.tensor_index < 0 || command->output.tensor_index >= g_nt ||
            command->inputs[0].prefix || command->output.prefix ||
            command->inputs[0].axis != 1 || command->output.axis != 1 ||
            command->mask_index != -1 || command->attention ||
            !vx_decode_cuda_command_bind(command, row, &p[i])) return 0;
        int input_count = 0, weight_count = 0, position_count = 0;
        for (int input = 0; input < node->nin; input++) {
            VxPortKind port = node->ins[input].port;
            if (port == VX_PORT_INPUT) input_count++;
            else if (i == 0 && port == VX_PORT_WEIGHT) weight_count++;
            else if (i == 2 && port == VX_PORT_POSITION_IDS) position_count++;
            else return 0;
        }
        if (input_count != 1 || weight_count != (i == 0) || position_count > 1 ||
            node->nin != 1 + weight_count + position_count ||
            nin(node, VX_PORT_INPUT) != p[i].inputs[0].tensor ||
            vx_incremental_node_output_locked(node, -1) != p[i].output.tensor ||
            strcmp(p[i].output.tensor->name, node->out)) return 0;
    }
    if (!g_incremental_scratch.cuda_commands[idx].numerical ||
        g_incremental_scratch.cuda_commands[idx].width != 128 ||
        (!reshape_alias && !g_incremental_scratch.cuda_commands[idx + 1].view) ||
        !g_incremental_scratch.cuda_commands[idx + 2].rope ||
        p[1].inputs[0].tensor != p[0].output.tensor ||
        p[2].inputs[0].tensor != p[1].output.tensor) return 0;
    T* weight = nin(nodes[0], VX_PORT_WEIGHT);
    T* tensors[5] = {p[0].inputs[0].tensor, weight, p[0].output.tensor,
        p[1].output.tensor, p[2].output.tensor};
    T* input = tensors[0];
    if (!input || input->ndim != 4 || input->shape[0] != 1 ||
        input->shape[1] <= row || input->shape[2] <= 0 || input->shape[2] > INT32_MAX / 128 ||
        input->shape[3] != 128 || !weight || weight->ndim != 1 || weight->shape[0] != 128 ||
        weight->numel != 128 || weight->is_graph_input ||
        !vx_incremental_tensor_is_model_weight_locked(weight) ||
        vx_adapter_run_has_target(weight->name)) return 0;
    /* Only the genuinely elided view may share the normalization span.
     * Both logical tensors still pass the current shape/binding checks below. */
    if (reshape_alias && tensors[2]->data != tensors[3]->data) return 0;
    int sequence = input->shape[1], heads = input->shape[2], width = heads * 128;
    uint64_t elements = (uint64_t)(uint32_t)sequence * (uint32_t)width;
    if (elements > UINT32_MAX || elements > SIZE_MAX / sizeof(float) ||
        vx_node_param_i32(nodes[0], VX_NODE_PARAM_D_MODEL, 128) != 128) return 0;
    for (int i = 0; i < 5; i++) {
        T* tensor = tensors[i];
        const VxIncrementalTensorBinding* binding = vx_incremental_tensor_binding_locked(tensor, state);
        if (!tensor || !tensor->data || tensor->dtype != T_F32 || tensor->elem_size != sizeof(float) ||
            (uint64_t)tensor->numel != (i == 1 ? 128u : elements) ||
            !binding || (i >= 2 && binding->producer != idx + i - 2) ||
            (i == 0 && binding->producer >= idx) || (i == 1 && binding->producer >= 0) || vx_paged_tensor_locked(tensor)) return 0;
        if (i == 0 || i == 2) {
            if (!tensor_same_shape(input, tensor)) return 0;
        } else if (i > 2 && (tensor->ndim != 3 || tensor->shape[0] != 1 ||
            tensor->shape[1] != sequence || tensor->shape[2] != width)) return 0;
    }
    T* positions = nin(nodes[2], VX_PORT_POSITION_IDS);
    if (positions && (!vx_decode_cuda_host_control(positions) || positions->elem_size != sizeof(int32_t))) return 0;
    VxSequenceRoPE rope;
    if (!vx_sequence_rope_parameters(nodes[2], tensors[3], tensors[4], &rope) ||
        rope.batch != 1 || rope.sequence != sequence || rope.width != width ||
        rope.heads != heads || rope.theta < 1.f) return 0;
    float epsilon = vx_node_param_f32(nodes[0], VX_NODE_PARAM_EPS, 1.e-6f);
    if (!isfinite(epsilon) || epsilon <= 0.f) return 0;
    const void* hosts[6] = {tensors[0]->data, weight->data, tensors[2]->data,
        tensors[3]->data, tensors[4]->data, positions ? positions->data : NULL};
    size_t bytes[6] = {(size_t)elements * 4u, 128u * 4u, (size_t)elements * 4u,
        (size_t)elements * 4u, (size_t)elements * 4u,
        positions ? (size_t)positions->numel * sizeof(int32_t) : 0};
    for (int i = 0; i < 6; i++)
        if (hosts[i] && (uintptr_t)hosts[i] > UINTPTR_MAX - bytes[i]) return 0;
    for (int output = 2; output < 5; output++)
        for (int other = 0; other < 6; other++) {
            if (output == other || !hosts[other]) continue;
            if (reshape_alias && ((output == 2 && other == 3) ||
                (output == 3 && other == 2)) && hosts[2] == hosts[3] &&
                bytes[2] == bytes[3]) continue;
            uintptr_t a = (uintptr_t)hosts[output], b = (uintptr_t)hosts[other];
            if (a <= b ? bytes[output] > b - a : bytes[other] > a - b) return 0;
        }
    /* Require contiguous spans before admission, so its index fallback cannot
     * write metadata while a later pure alias/geometry refusal remains. */
    for (int i = 0; i < 3; i++) {
        if (p[i].output.sequence != sequence || p[i].output.width != (size_t)width ||
            p[i].inputs[0].sequence != sequence || p[i].inputs[0].width != (size_t)width) return 0;
        uint32_t first, count, indexed; size_t storage_bytes;
        if (!vx_decode_cuda_span(&p[i].inputs[0], rows, &first, &count, &indexed, &storage_bytes) ||
            first != (uint32_t)row || count != 1 ||
            !vx_decode_cuda_span(&p[i].output, rows, &first, &count, &indexed, &storage_bytes) ||
            first != (uint32_t)row || count != 1 ||
            !vx_decode_cuda_project_compatible(nodes[i], idx + i, &p[i])) return 0;
    }
    int32_t position = rope.position_offset + row;
    if (positions) memcpy(&position, (const unsigned char*)positions->data + (size_t)row * sizeof(position), sizeof(position));
    int result = cuda_graph_head_rms_rope_row_f32(input->data, weight->data,
        tensors[2]->data, tensors[3]->data, tensors[4]->data, sequence, row,
        heads, rope.rotary_width, epsilon, rope.theta, position, rope.interleaved);
    if (result > 0 && state->runtime_route_backend && (size_t)(idx + 2) < g_node_capacity)
        for (int i = 0; i < 3; i++) state->runtime_route_backend[idx + i] = "cuda-graph-row+head-rms-rope";
    return result;
}

/* A compiled row command skips interpretation and registry selection. A miss
 * is side-effect free; a submitted command failure is terminal. Debug/node
 * trace passes never activate this private command domain. */
int vx_runtime_cuda_prepared_row_node_locked(int idx, int row, int is_last) {
    VxEngineState* state = vx_engine_state_current();
    if (!state->incremental_plan.cuda_commands_active || !state->incremental_scratch.cuda_commands ||
        !g_use_cuda || idx < 0 || idx >= g_nn || row < 0) return 0;
    const VxIncrementalCudaCommand* command = &state->incremental_scratch.cuda_commands[idx];
    Node* node = &g_n[idx];
    if (!command->valid || command->operator_kind != node->operator_kind ||
        node->skip || node->disabled || node->fuse_relu6 || g_node_fusion[idx].id) return 0;
    VxDecodeRowProjection projection;
    if (!vx_decode_cuda_command_bind(command, row, &projection)) return 0;
    int result = vx_decode_cuda_project(node, idx, is_last, &projection);
    if (result != 0) return -1;
    if (state->runtime_route_backend && (size_t)idx < g_node_capacity)
        state->runtime_route_backend[idx] = "cuda-graph-row+prepared";
    return 1;
}

/* A replay recipe has one logical cursor and one RoPE control value. Prove
 * those dependencies from the admitted row, rather than guessing which
 * argument bytes happened to change in two previous executions. More general
 * masks, fragmented pages and slot batches retain ordinary lowering. */
int vx_runtime_cuda_prepared_row_replay_locked(int row, int seal) {
    VxEngineState* state = vx_engine_state_current();
    const VxDecodeRowSet* rows = &g_decode_rows;
    if (!g_incremental_plan.cuda_commands_active || !g_incremental_scratch.cuda_commands ||
        !g_incremental_scratch.hybrid_nodes || rows->slots != 1 || rows->live != 1 ||
        !rows->positions || !rows->empty || !rows->kv_lengths || !rows->pages ||
        rows->empty[0] || rows->positions[0] != row || row < 0 || row == INT32_MAX ||
        rows->kv_lengths[0] != row + 1 || rows->key_capacity != row + 1 ||
        !state->decode_metadata_stage ||
        (size_t)row + 2u > state->decode_metadata_stage_bytes / sizeof(int32_t)) return 0;
    const VxDecodeSlotPages* pages = &rows->pages[0];
    if (pages->page_table) {
        if (pages->page_tokens <= 0 || pages->pages_per_slot <= 0 ||
            row / pages->page_tokens >= pages->pages_per_slot) return 0;
        for (int page = 0; page <= row / pages->page_tokens; page++)
            if (pages->page_table[page] != page) return 0;
    }
    int32_t position = 0;
    int have_position = 0, have_mask = 0;
    const T* mask = NULL;
    int mask_layout = 0;
    for (int idx = 0; idx < g_nn; idx++) {
        if (!g_incremental_scratch.hybrid_nodes[idx] || g_n[idx].skip || g_n[idx].disabled) continue;
        const VxIncrementalCudaCommand* command = &g_incremental_scratch.cuda_commands[idx];
        if (!command->valid || command->operator_kind != g_n[idx].operator_kind ||
            g_node_fusion[idx].id || g_n[idx].fuse_relu6) return 0;
        /* Numerical/view replay consumes only the live geometry verdict.
         * Attention, RoPE and scalar paging retain the complete binding. */
        if (g_decode_slots > 0 && !command->attention && !command->rope) {
            if (!vx_decode_cuda_row_geometry_current(&g_t[command->output.tensor_index],
                    command->output.sequence, command->output.width, rows)) return 0;
            continue;
        }
        VxDecodeRowProjection p;
        if (!vx_decode_cuda_command_bind(command, row, &p)) return 0;
        if (p.attention) {
            if (!p.causal || p.keys != row + 1) return 0;
            if (have_mask && (mask != p.mask || mask_layout != p.mask_layout)) return 0;
            if (!have_mask) {
                int source_keys = p.mask ? p.mask->shape[p.mask->ndim - 1] : p.keys;
                for (int key = 0; key < p.keys; key++)
                    if (!attention_mask_allowed(p.mask, p.mask_layout, 0, row, key,
                            p.output.sequence, source_keys, 0)) return 0;
                mask = p.mask; mask_layout = p.mask_layout; have_mask = 1;
            }
        }
        if (p.rope) {
            VxSequenceRoPE rope;
            if (!vx_sequence_rope_parameters(&g_n[idx], p.inputs[0].tensor, p.output.tensor, &rope)) return 0;
            int32_t value;
            if (rope.positions)
                memcpy(&value, (const unsigned char*)rope.positions->data + (size_t)row * sizeof(value), sizeof(value));
            else {
                if (rope.position_offset > INT32_MAX - row) return 0;
                value = rope.position_offset + row;
            }
            if (value < 0 || (have_position && value != position)) return 0;
            position = value; have_position = 1;
        }
    }
    VxCudaReplayRow recipe = {row, row + 1, position,
        g_incremental_scratch.hybrid_nodes, g_nn};
    if (seal) { cuda_graph_replay_row_seal(&recipe); return 0; }
    int result = cuda_graph_replay_row(&recipe);
    if (result > 0) {
        state->decode_cuda_mask_valid = state->decode_cuda_positions_valid = 0;
    }
    return result;
}
#endif
