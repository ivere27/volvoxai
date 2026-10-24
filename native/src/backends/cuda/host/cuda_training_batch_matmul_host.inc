/* Private BatchMatMul backward contract. Six bindings: A, B, dY, dA, dB,
 * Params[32]. The two destination entries launch 16x16 output tiles; every
 * gradient element has one writer, including singleton-batch reductions. */
typedef struct {
    uint32_t values[32];
} VxCudaTrainingBatchMatMulParams;
_Static_assert(sizeof(VxCudaTrainingBatchMatMulParams) == 128u,
               "CUDA BatchMatMul backward parameter ABI");

static int cuda_training_batch_matmul_supports(
        CudaTrainingCommand command, const size_t* bytes, int binding_count,
        uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    if ((command != CUDA_TRAINING_COMMAND_BATCH_MATMUL_A_BACKWARD &&
         command != CUDA_TRAINING_COMMAND_BATCH_MATMUL_B_BACKWARD) ||
        !bytes || binding_count != 6 || bytes[5] != 128u ||
        !groups_x || groups_y != 1u || groups_z != 1u ||
        bytes[0] != bytes[3] || bytes[1] != bytes[4]) return 0;
    for (int i = 0; i < 5; i++)
        if (!bytes[i] || bytes[i] % sizeof(float) || bytes[i] / sizeof(float) > UINT32_MAX)
            return 0;
    const size_t destination = bytes[command == CUDA_TRAINING_COMMAND_BATCH_MATMUL_A_BACKWARD ? 3 : 4];
    return groups_x <= destination / sizeof(float);
}

static int cuda_training_batch_matmul_params_valid(
        const uint32_t p[32], const size_t bytes[6], int a_command,
        uint32_t groups_x) {
    if (!p || !bytes || p[0] > 6u || !p[1] || !p[2] || !p[3] ||
        !p[4] || !p[5] || !p[6] || !p[7]) return 0;
    uint64_t a_extent = (uint64_t)p[1] * p[2];
    uint64_t b_extent = (uint64_t)p[2] * p[3];
    uint64_t out_matrix = (uint64_t)p[1] * p[3];
    uint64_t batches = 1u;
    if (a_extent > UINT32_MAX || b_extent > UINT32_MAX || out_matrix > UINT32_MAX)
        return 0;
    for (int dim = (int)p[0] - 1; dim >= 0; dim--) {
        uint32_t size = p[8 + dim], as = p[14 + dim], bs = p[20 + dim];
        if (!size || p[26 + dim] != batches ||
            (size == 1u && (as || bs)) || (size > 1u && !as && !bs)) return 0;
        if (as) {
            if (as != a_extent || a_extent > UINT32_MAX / size) return 0;
            a_extent *= size;
        }
        if (bs) {
            if (bs != b_extent || b_extent > UINT32_MAX / size) return 0;
            b_extent *= size;
        }
        if (batches > UINT32_MAX / size) return 0;
        batches *= size;
    }
    for (uint32_t dim = p[0]; dim < 6u; dim++)
        if (p[8 + dim] || p[14 + dim] || p[20 + dim] || p[26 + dim]) return 0;
    if (a_extent != p[4] || b_extent != p[5] || batches != p[7] ||
        batches > UINT32_MAX / out_matrix || batches * out_matrix != p[6]) return 0;
    const uint32_t lengths[5] = {p[4], p[5], p[6], p[4], p[5]};
    for (int i = 0; i < 5; i++)
        if (bytes[i] % sizeof(float) || bytes[i] / sizeof(float) != lengths[i]) return 0;
    const uint64_t rows = a_command ? p[1] : p[2];
    const uint64_t cols = a_command ? p[2] : p[3];
    const uint64_t count = (a_command ? a_extent : b_extent) / (rows * cols);
    const uint64_t tiles = count * ((rows - 1u) / 16u + 1u) * ((cols - 1u) / 16u + 1u);
    return tiles && tiles <= UINT32_MAX && groups_x == tiles;
}

static int cuda_training_dispatch_batch_matmul(
        CudaTrainingCommand command, void* const* hosts,
        const size_t* bytes, const unsigned char* access,
        const unsigned char* is_weight, uint32_t groups_x,
        uint32_t groups_y, uint32_t groups_z) {
    if (!hosts || !bytes || !access || !is_weight ||
        !cuda_training_batch_matmul_supports(command, bytes, 6, groups_x, groups_y, groups_z)) return -1;
    for (int i = 0; i < 6; i++) {
        if (!hosts[i] || access[i] != ((i == 3 || i == 4) ? CUDA_TRAINING_READ_WRITE : CUDA_TRAINING_READ) ||
            is_weight[i] > 1u) return -1;
    }
    if (is_weight[5]) return -1;
    VxCudaTrainingBatchMatMulParams params;
    memcpy(&params, hosts[5], sizeof(params));
    const int a_command = command == CUDA_TRAINING_COMMAND_BATCH_MATMUL_A_BACKWARD;
    if (!cuda_training_batch_matmul_params_valid(params.values, bytes, a_command, groups_x)) return -1;
    CudaTensorView views[5] = {{0}};
    CUdeviceptr devices[5] = {0};
    const int operand = a_command ? 1 : 0;
    const int destination = a_command ? 3 : 4;
    if (!graph_ensure_input_view(hosts[operand], bytes[operand], is_weight[operand], &views[operand]) ||
        !graph_ensure_input_view(hosts[2], bytes[2], is_weight[2], &views[2]) ||
        !graph_ensure_input_view(hosts[destination], bytes[destination], is_weight[destination], &views[destination])) return -1;
    devices[operand] = views[operand].device;
    devices[2] = views[2].device;
    devices[destination] = views[destination].device;
    void* args[6] = {&devices[0], &devices[1], &devices[2], &devices[3], &devices[4], &params};
    CUfunction function = a_command ? CUDA_TRAINING_FUNCTION(k_training_batch_matmul_backward_a)
                                   : CUDA_TRAINING_FUNCTION(k_training_batch_matmul_backward_b);
    if (!cuda_launch(function, groups_x, 1u, 1u, 16u, 16u, 1u, 0u, args)) return -1;
    graph_mark_device(views[destination].slot);
    return 0;
}
