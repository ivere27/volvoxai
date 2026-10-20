/* Preserved source005 production name-based domain predicate. */
int reference_paged_tensor_locked(const T* tensor) {
    const VxPagedBindingState* binding = &vx_engine_state_current()->paged;
    if (!binding->bound || !tensor || !tensor->name[0]) return 0;
    for (int index = 0; index < binding->name_count; index++) {
        if (!strcmp(binding->names[index], tensor->name)) return 1;
    }
    return 0;
}

static int reference_paged_read_port_allowed(const Node* node, VxPortKind key) {
    if ((node->operator_kind != VX_OP_CROSS_SDPA) && (node->operator_kind != VX_OP_Q_SDPA)) return 0;
    return !(key != VX_PORT_K) || !(key != VX_PORT_V);
}

int reference_paged_domain_supported_locked(const unsigned char* selected_nodes) {
    const VxPagedBindingState* binding = &vx_engine_state_current()->paged;
    if (!binding->bound) return 1;

    for (int node_index = 0; node_index < g_nn; node_index++) {
        const Node* node = &g_n[node_index];
        if (selected_nodes && !selected_nodes[node_index]) continue;
        for (int input = 0; input < node->nin; input++) {
            const T* tensor = t_find(node->ins[input].name);
            if (!reference_paged_tensor_locked(tensor)) continue;
            if (!reference_paged_read_port_allowed(node, node->ins[input].port)) {
                if (vx_engine_env("VOLVOXAI_ROW_DEBUG")) {
                    vx_engine_log(
                            "[row] node %d op=%s reads paged tensor %s on port %s\n",
                            node_index, vx_operator_kind_name(node->operator_kind), tensor->name, node->ins[input].key);
                }
                return 0;
            }
        }
        for (int output = 0; output < node->nout; output++) {
            const T* tensor = t_find(node->outs[output].name);
            if (!reference_paged_tensor_locked(tensor)) continue;
            for (int index = 0; index < binding->name_count; index++) {
                if (strcmp(binding->names[index], tensor->name)) continue;
                for (int prior = 0; prior < node_index; prior++) {
                    if (selected_nodes && !selected_nodes[prior]) continue;
                    for (int port = 0; port < g_n[prior].nout; port++)
                        if (!strcmp(g_n[prior].outs[port].name, tensor->name)) return 0;
                }
            }
        }
    }
    return 1;
}
