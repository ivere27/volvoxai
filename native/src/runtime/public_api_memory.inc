/* On-demand inventory uses existing owner locks. It never retains tensor data. */
#include "quant_cpu_isa.h"

static void vx_inventory_host(VxMemoryInventory* inventory, const void* pointer,
    uint64_t bytes, VxMemoryAllocator allocator, const char* name) {
    vx_memory_inventory_add(inventory, allocator, (uint64_t)(uintptr_t)pointer,
        bytes, vx_memory_allocator_role(allocator), name);
}
void vx_model_memory_inventory(VxModel* model, VxMemoryInventory* inventory) {
    inventory->owner_kind = VX_MEMORY_OWNER_MODEL; inventory->owner_id = model->public_id;
    vx_inventory_host(inventory, model, sizeof(*model), VX_MEMORY_HOST_METADATA, "model");
    vx_inventory_host(inventory, model->graph_plan_request_v1, model->graph_plan_request_v1_bytes, VX_MEMORY_HOST_METADATA, "graph definition");
    vx_inventory_host(inventory, model->graph_plan_v1, model->graph_plan_v1_bytes, VX_MEMORY_HOST_METADATA, "graph plan");
    vx_inventory_host(inventory, model->graph_bind_definition_v1, model->graph_bind_definition_v1_bytes, VX_MEMORY_HOST_METADATA, "shape definition");
    vx_inventory_host(inventory, model->graph_domain_v1, model->graph_domain_v1_bytes, VX_MEMORY_HOST_METADATA, "shape domain");
}
static void vx_compiled_memory_resources(VxCompiledModel* compiled, VxMemoryInventory* inventory) {
    vx_model_memory_inventory(compiled->model, inventory);
    inventory->owner_kind = VX_MEMORY_OWNER_COMPILED_MODEL; inventory->owner_id = compiled->public_id;
    vx_inventory_host(inventory, compiled, sizeof(*compiled), VX_MEMORY_HOST_METADATA, "compiled model");
    const VxCompiledResourceOwner* resources = compiled->resources;
    if (!resources) return;
    const VxCompiledWeightStore* weights = resources->weight_store;
    if (weights) for (size_t i = 0; i < weights->file_count; i++) {
        const SafetensorsFile* file = &weights->files[i];
        size_t old_count = inventory->count;
        vx_inventory_host(inventory, file->blob, file->size > 0 ? (uint64_t)file->size : 0,
            VX_MEMORY_HOST_WEIGHTS, "safetensors");
#if !defined(__wasm__)
        if (file->is_mmap && inventory->count > old_count)
            inventory->items[old_count].space = VX_MEMORY_SPACE_MAPPED_FILE;
#else
        /* The VFS mmap adapter borrows malloc storage in linear memory. */
        (void)old_count;
#endif
    }
    if (resources->cpu_weights) for (size_t i = 0; i < resources->cpu_weights->count; i++) {
        const VxCompiledCpuWeight* weight = &resources->cpu_weights->weights[i];
        vx_inventory_host(inventory, weight->data, weight->byte_size, VX_MEMORY_HOST_PACKED_WEIGHTS, weight->name);
    }
}
static void vx_context_memory_inventory_locked(VxExecutionContext* context, VxMemoryInventory* inventory) {
    vx_compiled_memory_resources(context->compiled, inventory);
    /* Run/scheduler contexts belong to the compiled route and have no public
     * context handle. Attribute their storage to that actual lifetime owner. */
    inventory->owner_kind = context->public_id ? VX_MEMORY_OWNER_EXECUTION_CONTEXT : VX_MEMORY_OWNER_COMPILED_MODEL;
    inventory->owner_id = context->public_id ? context->public_id : context->compiled->public_id;
    vx_inventory_host(inventory, context, sizeof(*context), VX_MEMORY_HOST_METADATA, "execution context");
    VxEngineState* state = context->engine_state;
    if (state) {
        vx_inventory_host(inventory, state, sizeof(*state), VX_MEMORY_HOST_METADATA, "engine state");
        vx_inventory_host(inventory, state->graph_metadata_storage, state->graph_metadata_bytes, VX_MEMORY_HOST_METADATA, "graph metadata");
        vx_engine_memory_inspect(state, vx_memory_inventory_inspect, inventory);
#if VOLVOXAI_ENABLE_WEBGPU
        if (state->webgpu_state) vx_webgpu_memory_inventory(inventory);
#endif
        for (int i = 0; i < state->tensor_count; i++) {
            const T* tensor = &state->tensors[i];
            if (tensor->owns && tensor->numel > 0) {
                size_t before = inventory->count;
                vx_inventory_host(inventory, tensor->data, (uint64_t)tensor->numel * tensor->elem_size,
                    VX_MEMORY_HOST_TENSOR, tensor->name);
                if (inventory->count > before && tensor->is_graph_input)
                    inventory->items[before].role = VX_MEMORY_RESOURCE_ROLE_INPUT;
            }
        }
        vx_inventory_host(inventory, state->cpu_typed_workspace, state->cpu_typed_workspace_capacity_bytes, VX_MEMORY_HOST_SCRATCH, "typed workspace");
        vx_inventory_host(inventory, state->decode_stage, state->decode_stage_bytes, VX_MEMORY_HOST_STAGING, "decode staging");
        for (int i = 0; i < state->node_count; i++) {
            if (state->q8wcache && state->q8wcache_bytes)
                vx_inventory_host(inventory, state->q8wcache[i], state->q8wcache_bytes[i], VX_MEMORY_HOST_PACKED_WEIGHTS, state->nodes[i].out);
            if (state->qlinear_metadata) {
                const QLinearMetadata* meta = &state->qlinear_metadata[i];
                vx_inventory_host(inventory, meta->packed_weight, meta->packed_weight_bytes, VX_MEMORY_HOST_PACKED_WEIGHTS, state->nodes[i].out);
            }
            if (state->qconv_metadata) {
                const QConv2DMetadata* meta = &state->qconv_metadata[i];
                vx_inventory_host(inventory, meta->packed_weight, meta->packed_weight_bytes, VX_MEMORY_HOST_PACKED_WEIGHTS, state->nodes[i].out);
                if (meta->small_c_packed_weight)
                    vx_inventory_host(inventory, meta->small_c_packed_weight,
                        vx_w8a8_qconv_small_c_pack_size(meta->kernel_height, meta->kernel_width,
                            meta->input_per_group, meta->output_channels, meta->groups),
                        VX_MEMORY_HOST_PACKED_WEIGHTS, state->nodes[i].out);
            }
        }
    }
    vx_native_pool_inspect(context->native_tensor_pool, inventory);
}
void vx_compiled_memory_inventory(VxCompiledModel* compiled, VxMemoryInventory* inventory) {
    vx_compiled_memory_resources(compiled, inventory);
    /* The route owns an unregistered context. Its mutex serializes every Run
     * and scheduled use. Do not block a snapshot on a busy inference route. */
    if (pthread_mutex_trylock(&compiled->route_mutex) != 0) {
        inventory->truncated = 1;
        return;
    }
    if (compiled->route_context)
        vx_context_memory_inventory_locked(compiled->route_context, inventory);
    pthread_mutex_unlock(&compiled->route_mutex);
}
VxStatus vx_context_memory_inventory(VxExecutionContext* context,
    VxMemoryInventory* inventory, VxMemoryView* view) {
    VxContextOperation operation;
    VxReport report = VX_REPORT_INIT;
    VxStatus status = vx_context_operation_begin(context, &operation, VX_STAGE_NONE, &report);
    if (status != VX_STATUS_OK) return status;
    vx_context_memory_inventory_locked(context, inventory);
    memset(view, 0, sizeof(*view));
    vx_native_pool_memory(context->native_tensor_pool, &view->result_capacity_bytes, &view->idle_result_bytes);
    vx_context_operation_end(context, &operation);
    return VX_STATUS_OK;
}

/* Canonical Node.outs is deliberately retained for shape resolution. Private
 * fusion can redirect Node.out and consume a peer's external operands. The
 * observation plan must describe what dispatch reads/writes, not stale
 * logical intermediate buffers. A depthwise/pointwise pair may also carry
 * one fused residual input. Each original edge is included at most twice. */
static uint64_t vx_context_plan_reference_capacity(const VxEngineState* state) {
    uint64_t count = (uint64_t)state->node_count * 2;
    for (int i = 0; i < state->node_count; i++)
        count += 2 * ((uint64_t)state->nodes[i].nin + (uint64_t)state->nodes[i].nout);
    return count;
}
static const char* vx_plan_effective_output(const VxEngineState* state, uint32_t index) {
    const Node* node = &state->nodes[index];
    if (node->skip || node->disabled) return node->out;
    int peer = state->concat_sigmoid_fuse ? state->concat_sigmoid_fuse[index] : -1;
    if (state->node_fusion && state->node_fusion[index].id == GRAPH_FUSION_DEPTHWISE_POINTWISE)
        peer = state->node_fusion[index].peer_idx;
    return peer >= 0 && peer < state->node_count ? state->nodes[peer].out : node->out;
}
static int vx_plan_reference(VxExecutionPlan* plan, VxEngineState* state,
    const char* name, uint32_t step, int output, uint32_t* cursor) {
    const T* resolved = t_find(name);
    if (!resolved || *cursor >= plan->reference_count) { plan->metadata_truncated = 1; return 0; }
    uint32_t index = (uint32_t)(resolved - state->tensors);
    plan->references[(*cursor)++] = index;
    VxExecutionPlanTensor* tensor = &plan->tensors[index];
    if (tensor->first < 0) tensor->first = output ? (int)step : 0;
    if (tensor->last < (int)step) tensor->last = (int)step;
    return 1;
}
#include "execution_cost.inc"

/* PTQ packages record which float tensor each renamed activation represents
 * in the safetensors metadata ("activation_sources", format volvox.ptq.v1).
 * Provenance only: quantization authority stays in the graph's descriptors. */
static void vx_plan_source_names(VxExecutionPlan* plan, const VxEngineState* state) {
    for (int f = 0; f < state->weight_file_count; f++) {
        const char* metadata_json = state->weight_files[f].metadata_json;
        if (!metadata_json || !strstr(metadata_json, "activation_sources")) continue;
        cJSON* metadata = cJSON_Parse(metadata_json);
        cJSON* format = cJSON_GetObjectItemCaseSensitive(metadata, "format");
        cJSON* encoded = cJSON_GetObjectItemCaseSensitive(metadata, "activation_sources");
        cJSON* sources = cJSON_IsString(format) && !strcmp(format->valuestring, "volvox.ptq.v1") &&
            cJSON_IsString(encoded) ? cJSON_Parse(encoded->valuestring) : NULL;
        for (uint32_t i = 0; cJSON_IsObject(sources) && i < plan->tensor_count; i++) {
            cJSON* source = cJSON_GetObjectItemCaseSensitive(sources, plan->tensors[i].name);
            if (cJSON_IsString(source) && strlen(source->valuestring) < sizeof(plan->tensors[i].source_name))
                strcpy(plan->tensors[i].source_name, source->valuestring);
        }
        cJSON_Delete(sources);
        cJSON_Delete(metadata);
    }
}

static void vx_context_fill_execution_plan(VxExecutionPlan* plan, void* opaque) {
    VxExecutionContext* context = opaque;
    VxEngineState* state = context->engine_state;
    VxMemoryInventory inventory = {.items = plan->allocations, .capacity = plan->allocation_count};
    vx_context_memory_inventory_locked(context, &inventory);
    plan->allocation_count = (uint32_t)inventory.count;
    plan->metadata_truncated = inventory.truncated;
    for (size_t a = 0; a < inventory.count; a++) {
        VxMemoryAllocation* allocation = &inventory.items[a];
        allocation->trace_id = vx_memory_allocation_id(&state->memory_observer, allocation->allocator, allocation->key);
    }
    plan->source_mapping_complete = 1;
    plan->placement_complete = !inventory.truncated && (context->compiled->builtin_backend == VX_PORTABLE_BACKEND_KIND);
    for (uint32_t i = 0; i < plan->tensor_count; i++) {
        const T* input = &state->tensors[i];
        VxExecutionPlanTensor* tensor = &plan->tensors[i];
        snprintf(tensor->name, sizeof(tensor->name), "%s", input->name);
        tensor->dtype = input->dtype; tensor->rank = input->ndim;
        tensor->bytes = input->numel > 0 ? (uint64_t)input->numel * input->elem_size : 0;
        tensor->first = tensor->last = -1;
        tensor->retained = input->is_graph_input;
        for (int d = 0; d < input->ndim && d < 8; d++) tensor->shape[d] = input->shape[d];
        for (size_t o = 0; o < context->compiled->model->output_count; o++)
            if (!strcmp(input->name, context->compiled->model->outputs[o].name)) tensor->retained = 1;
        if ((context->compiled->builtin_backend == VX_PORTABLE_BACKEND_KIND) && input->data) {
            uint64_t pointer = (uint64_t)(uintptr_t)input->data;
            for (size_t a = 0; a < inventory.count; a++) {
                const VxMemoryAllocation* allocation = &inventory.items[a];
                if (allocation->space == VX_MEMORY_SPACE_DEVICE || pointer < allocation->key) continue;
                uint64_t offset = pointer - allocation->key;
                if (offset <= allocation->capacity && tensor->bytes <= allocation->capacity - offset) {
                    tensor->allocation_id = a + 1; tensor->offset = offset;
                    if (allocation->role == VX_MEMORY_RESOURCE_ROLE_WEIGHTS || allocation->role == VX_MEMORY_RESOURCE_ROLE_PACKED_WEIGHTS)
                        tensor->retained = 1;
                    break;
                }
            }
        }
        if (tensor->bytes && !tensor->allocation_id) plan->placement_complete = 0;
    }
    uint32_t ref = 0;
    for (uint32_t i = 0; i < plan->step_count; i++) {
        const Node* node = &state->nodes[i];
        VxExecutionPlanStep* step = &plan->steps[i];
        snprintf(step->operator_name, sizeof(step->operator_name), "%s", vx_operator_kind_name(node->operator_kind));
        /* Logical graph node order is retained by the runtime optimizer; skip
         * and fusion flags describe the schedule's treatment of those nodes. */
        cJSON* source = cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(context->compiled->model->logical_graph, "nodes"), (int)i);
        cJSON* source_id = cJSON_GetObjectItemCaseSensitive(source, "id");
        if (cJSON_IsString(source_id) && source_id->valuestring) {
            if (strlen(source_id->valuestring) >= sizeof(step->source_node_id)) {
                plan->metadata_truncated = 1; plan->source_mapping_complete = 0;
            } else strcpy(step->source_node_id, source_id->valuestring);
        } else plan->source_mapping_complete = 0;
        step->skipped = node->skip || node->disabled; step->fused = node->fuse_relu6;
        /* Name the one source node fused into this step. A fused step whose
         * peer cannot be identified leaves the mapping incomplete. */
        int peer = -1;
        if (state->node_fusion && state->node_fusion[i].id) { step->fused = 1; peer = state->node_fusion[i].peer_idx; }
        else if (state->concat_sigmoid_fuse && state->concat_sigmoid_fuse[i] >= 0) {
            step->fused = 1; peer = state->concat_sigmoid_fuse[i];
        } else if (node->fuse_relu6) {
            for (uint32_t j = 0; j < plan->step_count && peer < 0; j++)
                if (j != i && state->nodes[j].skip && state->nodes[j].operator_kind == VX_OP_CLIP &&
                    !strcmp(state->nodes[j].out, node->out)) peer = (int)j;
        }
        if (step->fused) {
            cJSON* fused = peer >= 0 ? cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(context->compiled->model->logical_graph, "nodes"), peer) : NULL;
            cJSON* fused_id = cJSON_GetObjectItemCaseSensitive(fused, "id");
            if (!cJSON_IsString(fused_id) || !fused_id->valuestring) plan->source_mapping_complete = 0;
            else if (strlen(fused_id->valuestring) >= sizeof(step->fused_source_node_id)) {
                plan->metadata_truncated = 1; plan->source_mapping_complete = 0;
            } else strcpy(step->fused_source_node_id, fused_id->valuestring);
        }
        step->first_input = ref;
        for (int r = 0; r < node->nin; r++) {
            step->input_count += vx_plan_reference(plan, state, node->ins[r].name, i, 0, &ref);
        }
        const GraphNodeFusion* fusion = state->node_fusion ? &state->node_fusion[i] : NULL;
        if (!step->skipped && fusion) {
            if (fusion->id == GRAPH_FUSION_CONV_ADD || fusion->id == GRAPH_FUSION_CHAINED_ELEMENTWISE)
                step->input_count += vx_plan_reference(plan, state, fusion->aux_tensor, i, 0, &ref);
            if (fusion->id == GRAPH_FUSION_DEPTHWISE_POINTWISE && fusion->peer_idx >= 0 && fusion->peer_idx < state->node_count) {
                const Node* peer = &state->nodes[fusion->peer_idx];
                for (int r = 0; r < peer->nin; r++)
                    if (strcmp(peer->ins[r].name, node->out))
                        step->input_count += vx_plan_reference(plan, state, peer->ins[r].name, i, 0, &ref);
                const GraphNodeFusion* residual = &state->node_fusion[fusion->peer_idx];
                if (residual->id == GRAPH_FUSION_CONV_ADD)
                    step->input_count += vx_plan_reference(plan, state, residual->aux_tensor, i, 0, &ref);
            }
        }
        step->first_output = ref;
        for (int r = 0; r < node->nout; r++) {
            const char* name = r == 0 && !step->skipped ? vx_plan_effective_output(state, i) : node->outs[r].name;
            step->output_count += vx_plan_reference(plan, state, name, i, 1, &ref);
        }
        vx_step_cost(plan, state, i, step->fused ? peer : -1, step);
    }
    for (uint32_t i = 0; i < plan->tensor_count; i++)
        if (plan->tensors[i].retained && plan->step_count && plan->tensors[i].first >= 0)
            plan->tensors[i].last = (int)plan->step_count - 1;
    vx_plan_source_names(plan, state);
    plan->reference_count = ref;
}
static void vx_context_capture_trace_plan(void* opaque, VxTraceScope* scope) {
    VxExecutionContext* context = opaque;
    if (!vx_trace_plans_enabled(scope) || !context->engine_state) return;
    VxEngineState* state = context->engine_state;
    uint64_t references = vx_context_plan_reference_capacity(state);
    uint64_t allocations = (uint64_t)state->tensor_count + (uint64_t)state->node_count * 4 + 128;
    if (references > UINT32_MAX || allocations > UINT32_MAX) return;
    vx_trace_capture_plan(scope, context->compiled->model->graph_fingerprint,
        context->committed_shape_signature, state->activation_storage_generation, (uint32_t)state->node_count,
        (uint32_t)state->tensor_count, (uint32_t)references, (uint32_t)allocations,
        vx_context_fill_execution_plan, context);
}
