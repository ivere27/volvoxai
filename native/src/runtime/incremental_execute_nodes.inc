/* Instantiated below the selection boundary for ordinary and traced execution. */
    for (int node = 0; node < g_nn; node++) {
        int should_run = force_full ||
            (g_portable_selection_active
                ? g_portable_selected_nodes[node] != 0
                : node_has_dirty_input(node));
        if (!should_run) {
            skipped++;
            continue;
        }
        /*
         * The declared-batch admission, applied to the nodes that actually run.
         *
         * The row planner asks the same question, but only for a device
         * closure: a CPU row step has no backend graph, so it never builds a
         * plan and reaches this loop with nothing having checked. An
         * unconverted operator here would not fail — it would narrow to one
         * contiguous span from `seq_range`, which for `[lanes,S,W]` is row
         * `position` of lane zero, and every lane would read and write another
         * request's bytes while looking entirely plausible.
         *
         * A fused or disabled node is exempt for the same reason it is exempt
         * from `vx_runtime_node_incremental_row_compatible`: `graph_opt_fusion`
         * points its output at the producer's buffer and clears the node, so
         * `run_node` returns before executing anything. It addresses no rows,
         * and reading "does nothing" as "cannot do a batch" would let one
         * elided Reshape refuse an entire decoder.
         */
        if (g_decode_lanes > 1 && !g_n[node].skip && !g_n[node].disabled &&
#if VOLVOXAI_ENABLE_WEBGPU
            !g_use_webgpu &&
#endif
            !vx_runtime_node_incremental_row_compatible(&g_n[node], node, g_active_row)) goto done;
        /* Invalidate an old sidecar before the operator has a chance to
         * publish a replacement, then propagate dirtiness topologically. */
        prepare_dirty_outputs(node);
        if (VX_DISPATCH_NODE(node) != 0) goto done;
        executed++;
    }
