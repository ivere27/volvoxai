/* PTQ's CUDA executor only needs the prefix containing every selected
 * observation. In particular an already-packed low-bit head may follow all
 * Conv observers without requiring unsupported low-bit CUDA inference.
 * Scheduling flags are private to this authoring engine and always restored. */
static int ptq_cuda_observation_prefix(const VolvoxAIPTQPlan* plan, int** saved_skip) {
    *saved_skip = NULL;
#if VOLVOXAI_ENABLE_CUDA
    if (g_use_cuda) {
        int last = -1;
        for (int32_t i = 0; i < plan->layer_count; i++) {
            if (plan->layers[i].weight_algorithm != VX_WEIGHT_ALGORITHM_IMPORTANCE) continue;
            int index = plan->layers[i].node_index;
            if (index < 0 || index >= g_nn) return -1;
            /* Executing the selected layer also ensures its input's initial
             * graph upload has happened before a device-only moment read. */
            if (index > last) last = index;
        }
        for (int32_t i = 0; i < plan->tensor_count; i++) {
            const char* name = plan->tensors[i].name;
            T* tensor = t_find(name);
            int found = tensor && (tensor->is_graph_input ||
                ptq_tensor_is_source_weight_locked(name));
            for (int node = 0; node < g_nn; node++) {
                for (int out = 0; out < g_n[node].nout; out++) {
                    if (!strcmp(g_n[node].outs[out].name, name)) {
                        found = 1;
                        if (node > last) last = node;
                    }
                }
            }
            if (!found) return -1;
        }
        if (last < 0 || last >= g_nn || (size_t)g_nn > SIZE_MAX / sizeof(int)) return -1;
        if (last + 1 == g_nn) return 0;
        int* saved = malloc((size_t)g_nn * sizeof(*saved));
        if (!saved) return -1;
        for (int i = 0; i < g_nn; i++) {
            saved[i] = g_n[i].skip;
            if (i > last) g_n[i].skip = 1;
        }
        *saved_skip = saved;
    }
#else
    (void)plan;
#endif
    return 0;
}

static void ptq_cuda_restore_prefix(int* saved_skip) {
    if (!saved_skip) return;
    for (int i = 0; i < g_nn; i++) g_n[i].skip = saved_skip[i];
    free(saved_skip);
}
