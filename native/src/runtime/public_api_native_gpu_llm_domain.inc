/* Native GPU physical proof for the canonical whole-prefix LLM contracts. Shape
 * equalities and fixed feature dimensions are owned by shape_domain_contract;
 * this layer proves every buffer and the actual 64-thread launch envelope. */
static int vx_gpu_ngram_metadata_proven(const VxEngineState* validation, const Node* runtime_node) {
    const VxPortKind roles[] = {VX_PORT_MULTIPLIERS, VX_PORT_VOCAB_SIZES, VX_PORT_OFFSETS};
    const T* bank[3];
    for (int i = 0; i < 3; ++i) {
        const char* name = vx_native_gpu_validation_node_input_name(runtime_node, roles[i]);
        bank[i] = vx_native_gpu_validation_tensor_find(validation, name);
        if (!bank[i] || !bank[i]->data || bank[i]->dtype != T_I32 || bank[i]->is_graph_input ||
            vx_native_gpu_validation_node_producer(validation, name)) return 0;
    }
    if (bank[0]->ndim != 2 || bank[0]->shape[1] != 2 || bank[1]->numel != bank[2]->numel) return 0;
    for (long i = 0; i < bank[0]->numel; i += 2) {
        uint32_t words[2];
        memcpy(words, (const unsigned char*)bank[0]->data + (size_t)i * 4u, sizeof(words));
        if ((words[0] == 0 && words[1] == 0) || words[1] > INT32_MAX) return 0;
    }
    for (long i = 0; i < bank[1]->numel; ++i) {
        int32_t size, offset;
        memcpy(&size, (const unsigned char*)bank[1]->data + (size_t)i * 4u, 4u);
        memcpy(&offset, (const unsigned char*)bank[2]->data + (size_t)i * 4u, 4u);
        if (size <= 0 || offset < 0 || (int64_t)size + offset - 1 > INT32_MAX) return 0;
    }
    return 1;
}

static int vx_native_gpu_llm_proven(const cJSON* node,
        const VxDeclaredTensor* tensors, size_t tensor_count,
        const VxEngineState* validation, const VxNativeGpuDomainLimits* limits,
        const Node* runtime_node, int node_index, VxOperatorKind op, VxBackendKind backend) {
    if (op == VX_OP_PACKED_LINEAR && runtime_node &&
        vx_node_param_i32(runtime_node, VX_NODE_PARAM_WEIGHT_BITS, 4) < 4) return 0;
    const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
    const cJSON* outputs = cJSON_GetObjectItemCaseSensitive(node, "outputs");
    const cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
    if (!runtime_node || node_index < 0 || !cJSON_IsObject(inputs) ||
        !cJSON_IsObject(outputs) || !limits || limits->maximum_threads_per_block < 64 ||
        limits->maximum_block[0] < 64 || !limits->maximum_grid[0] ||
        (backend != VX_BACKEND_KIND_CUDA &&
         (limits->maximum_storage_bindings < 8u || limits->maximum_uniform_bindings < 1u ||
          limits->maximum_uniform_bytes < 128u))) return 0;
    for (const cJSON* port = inputs->child; port; port = port->next) {
        VxDeclaredTensor fixed = {0};
        const VxDeclaredTensor* tensor = vx_native_gpu_node_input_domain(
            node, port->string, tensors, tensor_count, validation, &fixed);
        VxDataType dtype = op == VX_OP_N_GRAM_IDS ||
            (op == VX_OP_PACKED_LINEAR && !strcmp(port->string, "weight")) ||
            (op == VX_OP_EMBEDDING_SHARD && !strcmp(port->string, "input")) ||
            !strcmp(port->string, "mask") || !strcmp(port->string, "position_ids") || !strcmp(port->string,"candidates") || !strcmp(port->string,"compressed_mask")
            ? VX_DTYPE_I32 : VX_DTYPE_F32;
        if (!vx_native_gpu_cuda_typed_span_proven(tensor, dtype, limits) ||
            (backend != VX_BACKEND_KIND_CUDA && tensor->maximum_byte_size > UINT32_MAX)) return 0;
    }
    for (const cJSON* port = outputs->child; port; port = port->next) {
        const char* name = cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(port, "tensor"));
        const VxDeclaredTensor* tensor = vx_declared_tensor_find(tensors, tensor_count, name);
        if (!vx_native_gpu_cuda_typed_span_proven(tensor,
                op == VX_OP_N_GRAM_IDS || op == VX_OP_SPARSE_INDEXER || op == VX_OP_QSA_INDEXER ? VX_DTYPE_I32 : VX_DTYPE_F32, limits) ||
            (backend != VX_BACKEND_KIND_CUDA && tensor->maximum_byte_size > UINT32_MAX)) return 0;
    }
    const VxDeclaredTensor* output = vx_native_gpu_node_output(node, tensors, tensor_count);
    VxDeclaredTensor fixed = {0};
    const char* role = op == VX_OP_N_GRAM_IDS ? "token_ids" :
        op == VX_OP_GATED_DELTA_NET || op == VX_OP_SPARSE_ATTENTION || op == VX_OP_SPARSE_INDEXER || op == VX_OP_QSA_INDEXER ? "q" : "input";
    const VxDeclaredTensor* input = vx_native_gpu_node_input_domain(
        node, role, tensors, tensor_count, validation, &fixed);
    uint64_t work;
    if (!input || !vx_native_gpu_tensor_maximum_elements(output, &work)) return 0;
    if (op == VX_OP_GATED_DELTA_NET || op == VX_OP_SPARSE_ATTENTION || op == VX_OP_ROTARY_EMBEDDING || op == VX_OP_SPARSE_INDEXER || op == VX_OP_QSA_INDEXER) {
        int64_t width;
        if (input->rank != 4 || !vx_declared_dimension_fixed_value(input, 3, &width) || width > 512)
            return 0;
    }
    if (op == VX_OP_GATED_DELTA_NET) {
        if (output->rank != 4 || output->maximums[1] <= 0) return 0;
        work /= (uint64_t)output->maximums[1]; /* B * value heads * value width. */
        if (backend != VX_BACKEND_KIND_CUDA) {
            uint64_t gates;
            if (output->maximums[3] <= 0 || !vx_native_gpu_tensor_maximum_elements(output, &gates)) return 0;
            gates /= (uint64_t)output->maximums[3];
            /* Gates and final-state ping-pong are simultaneous buffers.
             * The output span proof alone does not bound the four gates. */
            if (gates > UINT32_MAX / 16u || gates > limits->maximum_buffer_bytes / 16u) return 0;
            const cJSON* state_port = cJSON_GetObjectItemCaseSensitive(outputs, "state");
            if (state_port) {
                const char* state_name = cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(state_port, "tensor"));
                const VxDeclaredTensor* state = vx_declared_tensor_find(tensors, tensor_count, state_name);
                if (!state || !state->maximum_byte_size || state->maximum_byte_size > UINT32_MAX ||
                    state->maximum_byte_size > limits->maximum_buffer_bytes) return 0;
            }
            if (gates > work) work = gates;
        }
    } else if (op == VX_OP_KV_COMPRESS || op == VX_OP_HYPER_CONNECTION || op == VX_OP_HYPER_HEAD || op == VX_OP_SPARSE_INDEXER || op == VX_OP_QSA_INDEXER) {
        if (output->rank != 3 || output->maximums[2] <= 0) return 0;
        work /= (uint64_t)output->maximums[2]; /* One independent token per thread. */
    }
    if(op==VX_OP_SPARSE_INDEXER && backend!=VX_BACKEND_KIND_CUDA) {
        if(!vx_native_gpu_tensor_maximum_elements(output,&work))return 0;
    }
    if (op == VX_OP_QSA_INDEXER && backend != VX_BACKEND_KIND_CUDA) {
        VxDeclaredTensor kf = {0};
        const VxDeclaredTensor* keys = vx_native_gpu_node_input_domain(
            node, "k", tensors, tensor_count, validation, &kf);
        int rate = vx_node_param_i32(runtime_node, VX_NODE_PARAM_COMPRESS_RATE, 0);
        if (!keys || keys->rank != 3 || rate <= 0) return 0;
        uint64_t blocks = (uint64_t)keys->maximums[1] / (uint64_t)rate +
            ((uint64_t)keys->maximums[1] % (uint64_t)rate != 0);
        if (!blocks || work > UINT32_MAX / blocks) return 0;
        work *= blocks;
    }
    if (op != VX_OP_N_GRAM_IDS &&
        (!vx_native_gpu_positive_f32_parameter(params, "eps", 1e-6f) ||
         !vx_native_gpu_positive_f32_parameter(params, "normalization_epsilon", 1e-6f))) return 0;
    if ((op == VX_OP_GATED_DELTA_NET || op == VX_OP_SPARSE_ATTENTION) &&
        !vx_native_gpu_positive_f32_parameter(params, "scale", 1.0f)) return 0;
    if (op == VX_OP_SPARSE_ATTENTION) {
        if (backend != VX_BACKEND_KIND_CUDA) {
            VxDeclaredTensor vf = {0}, cf = {0}, kf = {0};
            const VxDeclaredTensor* value = vx_native_gpu_node_input_domain(
                node, "v", tensors, tensor_count, validation, &vf);
            const VxDeclaredTensor* compressed = vx_native_gpu_node_input_domain(
                node, "compressed_kv", tensors, tensor_count, validation, &cf);
            const VxDeclaredTensor* keys = vx_native_gpu_node_input_domain(
                node, "k", tensors, tensor_count, validation, &kf);
            int rate = vx_node_param_i32(runtime_node, VX_NODE_PARAM_COMPRESS_RATE, 0);
            int indexed = vx_native_gpu_validation_node_input_name(runtime_node, VX_PORT_INDEX_Q) &&
                vx_node_param_i32(runtime_node, VX_NODE_PARAM_TOP_K, 0) > 0;
            if (!value || value->rank != 4 || value->maximums[3] > 512 ||
                (compressed && (!keys || rate <= 0 || (indexed && keys->maximums[1] / rate > 512) ||
                    limits->maximum_shared_memory_bytes < 4096u ||
                    (uint64_t)output->maximums[0] * (uint64_t)output->maximums[1] > limits->maximum_grid[0]))) return 0;
            /* The attention shader has one invocation per query head. */
            work /= (uint64_t)output->maximums[3];
        }
        const char* name = vx_native_gpu_validation_node_input_name(runtime_node, VX_PORT_POSITION_IDS);
        if (name && !vx_native_gpu_i32_value_source_proof(validation, tensors, tensor_count,
                name, 0, INT32_MAX, node_index, 0)) return 0;
    }
    if (backend != VX_BACKEND_KIND_CUDA &&
        ((op == VX_OP_KV_COMPRESS && output->maximums[2] > 512) ||
         ((op == VX_OP_HYPER_CONNECTION || op == VX_OP_HYPER_HEAD || op == VX_OP_HYPER_RESIDUAL) &&
          input->maximums[2] > 16))) return 0;
    if (op == VX_OP_EMBEDDING_SHARD) {
        const char* weight_name = vx_native_gpu_validation_node_input_name(runtime_node, VX_PORT_WEIGHT);
        const T* weight = vx_native_gpu_validation_tensor_find(validation, weight_name);
        int vocabulary = vx_node_param_i32(runtime_node, VX_NODE_PARAM_VOCAB_SIZE, 0);
        if (!weight || weight->dtype != T_F32 || weight->ndim != 2 || !weight->data || weight->is_graph_input ||
            vx_native_gpu_validation_node_producer(validation, weight_name) || vocabulary <= 0 ||
            !vx_native_gpu_i32_value_source_proof(validation, tensors, tensor_count,
                vx_native_gpu_validation_node_input_name(runtime_node, VX_PORT_INPUT),
                0, vocabulary, node_index, 0)) return 0;
    }
    if (op == VX_OP_N_GRAM_IDS) {
        if (!vx_gpu_ngram_metadata_proven(validation, runtime_node)) return 0;
        int vocabulary = vx_node_param_i32(runtime_node, VX_NODE_PARAM_VOCAB_SIZE, 0);
        if (vocabulary <= 0 || !vx_native_gpu_i32_value_source_proof(validation, tensors, tensor_count,
                vx_native_gpu_validation_node_input_name(runtime_node, VX_PORT_TOKEN_IDS),
                0, vocabulary, node_index, 0)) return 0;
    }
    /* Packed CUDA assigns one warp to an output; its 64-thread CTA computes
     * two outputs. Portable shaders keep one output per invocation. */
    uint64_t outputs_per_group = op == VX_OP_PACKED_LINEAR && backend == VX_BACKEND_KIND_CUDA ? 2u : 64u;
    return work > 0 && (work + outputs_per_group - 1u) / outputs_per_group <= limits->maximum_grid[0];
}
