/* Included by public_api_memory.inc. The work one call of a plan step implies
 * at its concrete shapes, for achieved-throughput summaries. These are counts,
 * not measurements. Multiply-accumulates cover matrix products, convolutions
 * and attention (float or integer). Other arithmetic counts one operation per
 * output element, and exp/tanh/erf-class functions are counted separately, as
 * in XLA's cost analysis. Data movement has no arithmetic. */

typedef struct {
    int status;
    uint64_t macs, elementwise, transcendental;
} VxNodeCost;

static const T* vx_cost_port(const Node* node, VxPortKind port) {
    for (int r = 0; r < node->nin; r++)
        if (node->ins[r].port == port) return t_find(node->ins[r].name);
    return NULL;
}
static uint64_t vx_cost_numel(const T* tensor) {
    return tensor && tensor->numel > 0 ? (uint64_t)tensor->numel : 0;
}
static int vx_cost_multiply(uint64_t a, uint64_t b, uint64_t* out) {
    if (a && b > UINT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}
/* Matrix products: the weight's elements per output column give K whatever
 * the weight layout. Packed or two-activation forms use the input's last axis. */
static int vx_cost_matmul(const Node* node, const T* out, VxNodeCost* cost, int packed) {
    const T* weight = vx_cost_port(node, VX_PORT_WEIGHT);
    uint64_t n = out->ndim > 0 ? (uint64_t)out->shape[out->ndim - 1] : 0, k = 0;
    if (weight && n && vx_cost_numel(weight) % n == 0) k = vx_cost_numel(weight) / n;
    if (!k || packed) {
        const T* input = vx_cost_port(node, VX_PORT_INPUT);
        if (!input) input = vx_cost_port(node, VX_PORT_A);
        if (!input || input->ndim < 1) return VX_COST_STATUS_UNKNOWN;
        k = (uint64_t)input->shape[input->ndim - 1];
        cost->status = VX_COST_STATUS_ESTIMATED;
    }
    return vx_cost_multiply(vx_cost_numel(out), k, &cost->macs) ? cost->status : VX_COST_STATUS_UNKNOWN;
}
/* Convolutions: weight elements per output channel are kh*kw*cin/groups for
 * HWIO, HWCM (depthwise) and OHWI alike; images are channels-last. */
static int vx_cost_convolution(const Node* node, const T* out, VxNodeCost* cost, int transposed) {
    const T* weight = vx_cost_port(node, VX_PORT_WEIGHT);
    const T* reference = transposed ? vx_cost_port(node, VX_PORT_INPUT) : out;
    if (!weight || !reference || reference->ndim < 1) return VX_COST_STATUS_UNKNOWN;
    uint64_t channels = (uint64_t)reference->shape[reference->ndim - 1];
    if (!channels || vx_cost_numel(weight) % channels) return VX_COST_STATUS_UNKNOWN;
    return vx_cost_multiply(vx_cost_numel(reference), vx_cost_numel(weight) / channels, &cost->macs)
        ? cost->status : VX_COST_STATUS_UNKNOWN;
}
/* Attention: Q·K and P·V each take one multiply-accumulate per output element
 * and key, assuming equal key and value widths. Causal masking and skipped
 * keys can make actual work smaller, so the count is an upper estimate. */
static int vx_cost_attention(const Node* node, const T* out, VxNodeCost* cost) {
    const T* keys = vx_cost_port(node, VX_PORT_K);
    const T* sequence = keys ? keys : out;
    if (sequence->ndim < 2) return VX_COST_STATUS_UNKNOWN;
    uint64_t length = (uint64_t)sequence->shape[sequence->ndim - 2], per_key;
    cost->status = VX_COST_STATUS_ESTIMATED;
    return vx_cost_multiply(vx_cost_numel(out), length, &per_key) &&
           vx_cost_multiply(per_key, 2, &cost->macs) ? cost->status : VX_COST_STATUS_UNKNOWN;
}

static int vx_node_cost(const Node* node, VxNodeCost* cost) {
    memset(cost, 0, sizeof(*cost));
    cost->status = VX_COST_STATUS_EXACT;
    if (node->skip || node->disabled) return cost->status;
    const T* out = t_find(node->out);
    if (!out) return cost->status = VX_COST_STATUS_UNKNOWN;
    uint64_t elements = vx_cost_numel(out);
    const T* input = vx_cost_port(node, VX_PORT_INPUT);
    switch (node->operator_kind) {
    case VX_OP_MATMUL: case VX_OP_LINEAR: case VX_OP_GEMM:
        return cost->status = vx_cost_matmul(node, out, cost, 0);
    case VX_OP_Q_LINEAR: case VX_OP_Q_GEMM: case VX_OP_Q_MATMUL:
    case VX_OP_BATCH_MATMUL: case VX_OP_Q_BATCH_MATMUL:
        return cost->status = vx_cost_matmul(node, out, cost,
            node->operator_kind == VX_OP_BATCH_MATMUL || node->operator_kind == VX_OP_Q_BATCH_MATMUL);
    case VX_OP_CONV_2D: case VX_OP_Q_CONV_2D: case VX_OP_CONV_1D:
        return cost->status = vx_cost_convolution(node, out, cost, 0);
    case VX_OP_CONV_TRANSPOSE_2D:
        return cost->status = vx_cost_convolution(node, out, cost, 1);
    case VX_OP_SDPA: case VX_OP_Q_SDPA: case VX_OP_CROSS_SDPA:
        return cost->status = vx_cost_attention(node, out, cost);
    case VX_OP_RESHAPE: case VX_OP_FLATTEN: case VX_OP_SQUEEZE: case VX_OP_UNSQUEEZE:
    case VX_OP_TRANSPOSE: case VX_OP_CONCAT: case VX_OP_CONCAT2: case VX_OP_SPLIT:
    case VX_OP_SLICE: case VX_OP_PAD: case VX_OP_EXPAND: case VX_OP_IDENTITY:
    case VX_OP_BROADCAST: case VX_OP_CAST: case VX_OP_DROPOUT: case VX_OP_GATHER:
    case VX_OP_GATHER_ELEMENTS: case VX_OP_EMBEDDING: case VX_OP_Q_EMBEDDING:
        return cost->status;
    case VX_OP_RELU: case VX_OP_LEAKY_RELU: case VX_OP_PRELU: case VX_OP_CLIP:
    case VX_OP_ADD: case VX_OP_Q_ADD: case VX_OP_SUB: case VX_OP_MUL: case VX_OP_DIV:
    case VX_OP_HARD_SWISH: case VX_OP_HARD_SIGMOID: case VX_OP_EQUAL:
    case VX_OP_GREATER_OR_EQUAL: case VX_OP_NOT: case VX_OP_WHERE: case VX_OP_MASK:
    case VX_OP_QUANTIZE_LINEAR: case VX_OP_DEQUANTIZE_LINEAR: case VX_OP_REQUANTIZE_LINEAR:
        cost->elementwise = elements;
        return cost->status;
    case VX_OP_GELU: case VX_OP_Q_GELU: case VX_OP_SILU: case VX_OP_Q_SILU:
    case VX_OP_SIGMOID: case VX_OP_TANH: case VX_OP_SIN: case VX_OP_COS:
        cost->elementwise = cost->transcendental = elements;
        return cost->status;
    case VX_OP_SOFTMAX: case VX_OP_LOG_SOFTMAX:
        cost->elementwise = cost->transcendental = elements;
        return cost->status = VX_COST_STATUS_ESTIMATED;
    case VX_OP_LAYER_NORM: case VX_OP_Q_LAYER_NORM: case VX_OP_GROUP_NORM:
    case VX_OP_Q_GROUP_NORM: case VX_OP_RMS_NORM: case VX_OP_BATCH_NORM_2D: case VX_OP_ROPE:
        cost->elementwise = elements;
        return cost->status = VX_COST_STATUS_ESTIMATED;
    /* Windowed or reducing operators: one operation per element read. */
    case VX_OP_MAX_POOL_2D: case VX_OP_AVERAGE_POOL_2D: case VX_OP_GLOBAL_AVERAGE_POOL:
    case VX_OP_REDUCE_SUM: case VX_OP_REDUCE_MEAN: case VX_OP_Q_MASKED_MEAN:
    case VX_OP_ARG_MAX: case VX_OP_Q_ARG_MAX: case VX_OP_RESIZE: case VX_OP_INTERPOLATE_1D:
    case VX_OP_RESIZE_NEAREST_2D: case VX_OP_UPSAMPLE_NEAREST_2D:
        cost->elementwise = input && vx_cost_numel(input) > elements ? vx_cost_numel(input) : elements;
        return cost->status = VX_COST_STATUS_ESTIMATED;
    default:
        return cost->status = VX_COST_STATUS_UNKNOWN;
    }
}

/* A fused step executes its fusion peer too; the worst status wins. */
static void vx_step_cost(const VxExecutionPlan* plan, const VxEngineState* state,
    uint32_t index, int peer, VxExecutionPlanStep* step) {
    VxNodeCost own, fused;
    step->cost_status = vx_node_cost(&state->nodes[index], &own);
    step->cost_macs = own.macs;
    step->cost_elementwise = own.elementwise;
    step->cost_transcendental = own.transcendental;
    if (!step->skipped && peer >= 0 && peer < state->node_count && (uint32_t)peer != index) {
        Node shadow = state->nodes[peer];
        shadow.skip = shadow.disabled = 0; /* The peer's work runs inside this step. */
        int status = vx_node_cost(&shadow, &fused);
        if (status > step->cost_status) step->cost_status = status;
        step->cost_macs += fused.macs;
        step->cost_elementwise += fused.elementwise;
        step->cost_transcendental += fused.transcendental;
    }
    step->cost_input_bytes = step->cost_output_bytes = 0;
    for (uint32_t r = 0; r < step->input_count; r++)
        step->cost_input_bytes += plan->tensors[plan->references[step->first_input + r]].bytes;
    for (uint32_t r = 0; r < step->output_count; r++)
        step->cost_output_bytes += plan->tensors[plan->references[step->first_output + r]].bytes;
}
