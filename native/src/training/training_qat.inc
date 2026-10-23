/* Full-only weight QAT. The optimizer always owns FP32 masters. Temporary
 * reconstructed values are visible only during forward and its VJP. */
typedef struct {
    T* tensor;
    uint32_t rows, columns;
    int transpose, convolution;
    const VxTrainingQatNode* policy;
} TrainingQatResolved;
typedef struct {
    int tensor_index;
    size_t count;
    float* master;
    float* fake;
    unsigned char* pass;
#if VOLVOXAI_ENABLE_CUDA
    CudaTrainingQatWeight* cuda;
#endif
} TrainingQatWeight;
typedef struct {
    TrainingQatWeight* weights;
    size_t count;
    size_t live_bytes;
    int device;
} TrainingQatStep;

static void* training_qat_allocate(TrainingQatStep* step, size_t bytes) {
    if (!bytes || bytes > SIZE_MAX - step->live_bytes) return NULL;
    void* memory = malloc(bytes);
    if (memory) {
        step->live_bytes += bytes;
        VxEngineState* engine = vx_engine_state_current();
        if (step->live_bytes > engine->training_qat_workspace_high_water)
            engine->training_qat_workspace_high_water = step->live_bytes;
    }
    return memory;
}
static void training_qat_free_buffer(TrainingQatStep* step, void* memory, size_t bytes) {
    if (memory) { free(memory); step->live_bytes -= bytes; }
}

static int training_qat_node_index(const char* id) {
    cJSON* nodes = cJSON_GetObjectItemCaseSensitive(vx_engine_state_current()->graph_root, "nodes");
    int found = -1;
    for (int i = 0; i < g_nn; i++) {
        cJSON* name = cJSON_GetObjectItemCaseSensitive(cJSON_GetArrayItem(nodes, i), "id");
        if (cJSON_IsString(name) && !strcmp(name->valuestring, id)) {
            if (found >= 0) return -1;
            found = i;
        }
    }
    return found;
}

static int training_qat_resolve(const VxTrainingQatNode* policy, TrainingQatResolved* out) {
    if (!policy || !policy->node_id || !policy->node_id[0] ||
        (policy->bits != 2 && policy->bits != 3 && policy->bits != 4) ||
        (policy->algorithm != VX_WEIGHT_ALGORITHM_RTN &&
         policy->algorithm != VX_WEIGHT_ALGORITHM_MSE_CLIP)) return -1;
    int index = training_qat_node_index(policy->node_id);
    if (index < 0) return -1;
    Node* node = &g_n[index];
    T* weight = nin(node, VX_PORT_WEIGHT);
    T* input = node_input_default(node);
    T* output = t_find(node->out);
    if (!weight || !input || !output || weight->dtype != T_F32 || !weight->data ||
        weight->numel <= 0 || (uint64_t)weight->numel > SIZE_MAX / sizeof(float) ||
        !volvoxai_engine_tensor_is_model_weight_locked(weight->name)) return -1;
    memset(out, 0, sizeof(*out));
    out->tensor = weight; out->policy = policy;
    if (node->operator_kind == VX_OP_LINEAR || node->operator_kind == VX_OP_MATMUL) {
        int out_in;
        if (!vx_packed_group_size_supported((int)policy->group_size) ||
            linear_node_weight_layout(node, input, output, weight, &out_in)) return -1;
        out->transpose = !out_in;
        out->rows = (uint32_t)weight->shape[out_in ? 0 : 1];
    } else if (node->operator_kind == VX_OP_CONV_2D) {
        if (policy->group_size || weight->ndim != 4 || input->ndim != 4 || output->ndim != 4)
            return -1;
        if (vx_node_param_symbol(node, VX_NODE_PARAM_DATA_LAYOUT, VX_NODE_SYMBOL_NHWC) != VX_NODE_SYMBOL_NHWC)
            return -1;
        VxNodeParamSymbol layout = vx_node_param_symbol(node, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_HWIO);
        if (layout != VX_NODE_SYMBOL_HWIO && layout != VX_NODE_SYMBOL_OHWI) return -1;
        out->convolution = 1;
        out->transpose = layout == VX_NODE_SYMBOL_HWIO;
        out->rows = (uint32_t)weight->shape[out->transpose ? 3 : 0];
        if (out->rows != (uint32_t)output->shape[3]) return -1;
    } else return -1;
    if (!out->rows || out->rows > INT_MAX ||
        (uint64_t)weight->numel % out->rows ||
        (uint64_t)weight->numel / out->rows > UINT32_MAX) return -1;
    out->columns = (uint32_t)((uint64_t)weight->numel / out->rows);
    return out->columns ? 0 : -1;
}

int volvoxai_engine_validate_qat(const VxTrainingQatPolicy* policy) {
    if (!policy) return 0;
    if (!policy->count || !policy->nodes || policy->count > (size_t)g_nn) return -1;
    for (size_t i = 0; i < policy->count; i++) {
        TrainingQatResolved selected;
        if (training_qat_resolve(&policy->nodes[i], &selected)) return -1;
        for (size_t j = 0; j < i; j++)
            if (!strcmp(policy->nodes[i].node_id, policy->nodes[j].node_id)) return -1;
        /* Every consumer of a shared master must explicitly opt into the same
         * layout and policy. Applying one global tensor swap otherwise changes
         * an unselected node, or gives its VJP the wrong interpretation. */
        for (int n = 0; n < g_nn; n++) for (int port = 0; port < g_n[n].nin; port++) {
            if (strcmp(g_n[n].ins[port].name, selected.tensor->name)) continue;
            if (g_n[n].ins[port].port != VX_PORT_WEIGHT ||
                nin(&g_n[n], VX_PORT_WEIGHT) != selected.tensor) return -1;
            int matched = 0;
            for (size_t j = 0; j < policy->count; j++) {
                if (training_qat_node_index(policy->nodes[j].node_id) != n) continue;
                TrainingQatResolved other;
                if (training_qat_resolve(&policy->nodes[j], &other) ||
                    other.tensor != selected.tensor || other.transpose != selected.transpose ||
                    other.convolution != selected.convolution || other.rows != selected.rows ||
                    other.policy->bits != selected.policy->bits ||
                    other.policy->group_size != selected.policy->group_size ||
                    other.policy->algorithm != selected.policy->algorithm) return -1;
                matched = 1;
            }
            if (!matched) return -1;
        }
    }
    return 0;
}

static void training_qat_restore_masters(TrainingQatStep* step) {
    for (size_t i = 0; i < step->count; i++) {
        TrainingQatWeight* weight = &step->weights[i];
        if (weight->master && weight->tensor_index >= 0 && weight->tensor_index < g_nt)
            g_t[weight->tensor_index].data = weight->master;
    }
}
static void training_qat_retire_cuda(TrainingQatStep* step) {
#if VOLVOXAI_ENABLE_CUDA
    if (!step->device) return;
    training_qat_restore_masters(step);
    for (size_t i = 0; i < step->count; i++)
        if (step->weights[i].fake)
            cuda_graph_demote_weight(step->weights[i].fake,step->weights[i].count*sizeof(float));
    /* Retire tape identities before their host shadows are freed. Existing
     * CUDA master slots remain resident, including device-authoritative updates. */
    cuda_graph_release_transients();
    for (size_t i = 0; i < step->count; i++) {
        cuda_training_qat_weight_destroy(step->weights[i].cuda);
        step->weights[i].cuda=NULL;
    }
    step->device=0;
#else
    (void)step;
#endif
}
static void training_qat_restore(TrainingQatStep* step) {
    training_qat_retire_cuda(step);
    training_qat_restore_masters(step);
    for (size_t i = 0; i < step->count; i++) {
        TrainingQatWeight* weight = &step->weights[i];
        free(weight->fake); free(weight->pass);
    }
    free(step->weights);
    memset(step, 0, sizeof(*step));
}

static int training_qat_code(float ratio, int low, int high) {
    if (ratio <= (float)low) return low;
    if (ratio >= (float)high) return high;
    int whole = (int)ratio;
    float remainder = ratio - (float)whole;
    if (remainder > .5f || (remainder == .5f && (whole & 1))) whole++;
    if (remainder < -.5f || (remainder == -.5f && (whole & 1))) whole--;
    return whole;
}

static int training_qat_prepare(TrainingQatStep* step, int device) {
    const VxTrainingQatPolicy* policy = vx_engine_state_current()->training_qat;
    memset(step, 0, sizeof(*step));
    if (!policy) return 0;
#if !VOLVOXAI_ENABLE_CUDA
    if (device) return -1;
#endif
    step->device=device;
    if (volvoxai_engine_validate_qat(policy) || policy->count > SIZE_MAX / sizeof(*step->weights)) return -1;
    step->weights = training_qat_allocate(step, policy->count * sizeof(*step->weights));
    if (!step->weights) return -1;
    memset(step->weights, 0, policy->count * sizeof(*step->weights));
    for (size_t i = 0; i < policy->count; i++) {
        TrainingQatResolved value;
        if (training_qat_resolve(&policy->nodes[i], &value)) goto fail;
        int tensor_index = t_index(value.tensor);
        if (tensor_index < 0) goto fail;
        int duplicate = 0;
        for (size_t j = 0; j < step->count; j++)
            if (step->weights[j].tensor_index == tensor_index) duplicate = 1;
        if (duplicate) continue;
        const VxTrainingQatNode* config = value.policy;
        size_t count = (size_t)value.tensor->numel;
        uint32_t groups = value.convolution ? 1u :
            value.columns / config->group_size + (value.columns % config->group_size != 0);
        uint64_t scale_count = (uint64_t)value.rows * groups;
        uint64_t row_units = ((uint64_t)value.columns * config->bits +
            (value.convolution ? 7u : 31u)) / (value.convolution ? 8u : 32u);
        uint64_t packed_bytes = row_units * value.rows * (value.convolution ? 1u : 4u);
        if (scale_count > SIZE_MAX / sizeof(float) || packed_bytes > SIZE_MAX) goto fail;
        TrainingQatWeight* item = &step->weights[step->count++];
        item->tensor_index = tensor_index;
        item->count = count;
        item->fake = training_qat_allocate(step, count * sizeof(float));
#if VOLVOXAI_ENABLE_CUDA
        if (device) {
            if (!item->fake || count>UINT32_MAX ||
                count>(SIZE_MAX-sizeof(uint32_t))/(sizeof(float)+1u))goto fail;
            item->cuda=cuda_training_qat_weight_prepare(value.tensor->data,item->fake,
                value.rows,value.columns,config->group_size,config->bits,
                (uint32_t)value.transpose,(uint32_t)value.convolution,(uint32_t)config->algorithm);
            if (!item->cuda)goto fail;
            /* Fake device values plus one byte STE mask per master element;
             * this is additional to the already-charged host fake shadow. */
            size_t device_bytes=count*sizeof(float)+count+sizeof(uint32_t);
            if (device_bytes>SIZE_MAX-step->live_bytes)goto fail;
            step->live_bytes+=device_bytes;
            if(step->live_bytes>vx_engine_state_current()->training_qat_workspace_high_water)
                vx_engine_state_current()->training_qat_workspace_high_water=step->live_bytes;
            continue;
        }
#endif
        item->pass = training_qat_allocate(step, count);
        float* canonical = training_qat_allocate(step, count * sizeof(float));
        float* scales = training_qat_allocate(step, (size_t)scale_count * sizeof(float));
        void* packed = training_qat_allocate(step, (size_t)packed_bytes);
        if (!item->fake || !item->pass || !canonical || !scales || !packed) {
            training_qat_free_buffer(step, canonical, count * sizeof(float));
            training_qat_free_buffer(step, scales, (size_t)scale_count * sizeof(float));
            training_qat_free_buffer(step, packed, (size_t)packed_bytes);
            goto fail;
        }
        for (uint32_t row = 0; row < value.rows; row++) for (uint32_t k = 0; k < value.columns; k++)
            canonical[(size_t)row * value.columns + k] = value.tensor->data[
                value.transpose ? (size_t)k * value.rows + row : (size_t)row * value.columns + k];
        int ok = value.convolution ? volvoxai_ptq_pack_conv_weight(canonical,
            (int32_t)value.rows, value.columns, config->bits, config->algorithm,
            NULL, 0u, packed, scales) == 0 :
            volvoxai_training_quantize_grouped_weight(canonical, value.rows, value.columns,
                0, config->bits, config->group_size, config->algorithm, NULL, NULL,
                packed, scales, NULL);
        int low = -(1 << (config->bits - 1u)), high = -low - 1;
        for (uint32_t row = 0; ok && row < value.rows; row++) for (uint32_t k = 0; k < value.columns; k++) {
            float scale = scales[(size_t)row * groups + (value.convolution ? 0u : k / config->group_size)];
            float ratio = canonical[(size_t)row * value.columns + k] / scale;
            size_t index = value.transpose ? (size_t)k * value.rows + row : (size_t)row * value.columns + k;
            item->fake[index] = (float)training_qat_code(ratio, low, high) * scale;
            item->pass[index] = ratio >= (float)low && ratio <= (float)high;
            if (!isfinite(item->fake[index])) ok = 0;
        }
        training_qat_free_buffer(step, canonical, count * sizeof(float));
        training_qat_free_buffer(step, scales, (size_t)scale_count * sizeof(float));
        training_qat_free_buffer(step, packed, (size_t)packed_bytes);
        if (!ok) goto fail;
    }
    /* Installation is atomic after all allocations, fitting and checks. */
    for (size_t i = 0; i < step->count; i++) {
        step->weights[i].master = g_t[step->weights[i].tensor_index].data;
        g_t[step->weights[i].tensor_index].data = step->weights[i].fake;
    }
    return 0;
fail:
    training_qat_restore(step);
    return -1;
}

static int training_qat_backward(TrainingQatStep* step, float** grads) {
    for (size_t i = 0; i < step->count; i++) {
        TrainingQatWeight* weight = &step->weights[i];
        if (weight->tensor_index < 0 || weight->tensor_index >= g_nt ||
            (uint64_t)g_t[weight->tensor_index].numel != weight->count) return -1;
        float* gradient = grad_existing(grads, &g_t[weight->tensor_index]);
#if VOLVOXAI_ENABLE_CUDA
        if(step->device) {
            if(gradient && cuda_training_qat_weight_backward(weight->cuda,gradient))return -1;
            continue;
        }
#endif
        if (gradient) for (size_t k = 0; k < weight->count; k++)
            if (!weight->pass[k] && isfinite(gradient[k])) gradient[k] = 0.0f;
    }
    return 0;
}

static const float* training_qat_master(const TrainingQatStep* step, const T* tensor) {
    int index = t_index((T*)tensor);
    for (size_t i = 0; i < step->count; i++)
        if (step->weights[i].tensor_index == index) return step->weights[i].master;
    return tensor ? tensor->data : NULL;
}
