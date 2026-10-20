/* Frozen source010 decode negotiation, before removing the duplicate
 * required-device closure probe. It remains the acceptance/failure oracle. */
static int decode_run_begin_reference_locked(VxDecodeRun* run, VolvoxAIDecodeSession* session,
                                   int prefill, const int* positions, int slots, int stepwise) {
    int position = -1;
    int row_ready = 1;
    memset(run, 0, sizeof(*run));
    run->session = session;
    run->prefill = prefill;
    if (!decode_session_attached(session)) return -1;
    if (prefill) {
        vx_incremental_prepare_ordinary_locked();
        g_execution_row = -1;
        session->prefilled = 0;
        session->last_execution_mode = VOLVOXAI_DECODE_MODE_NONE;
        if (vx_incremental_run_begin_locked(&run->pass, -1, stepwise) == 0) return 0;
        decode_run_finish_locked(run, -1);
        return -1;
    }
    if (positions) {
        if (slots != session->slots || slots < 1) return -1;
        run->rows = 1;
        if (vx_paged_row_set_init_locked(&g_decode_rows, slots, positions) != VX_DECODE_ROW_SET_OK) {
            vx_decode_row_set_dispose(&g_decode_rows);
            run->rows = 0;
            return -1;
        }
        // The first slot can be empty or recomputing at zero. Admission uses a live,
        // advancing row; operators use the complete slot set for addressing.
        position = 0;
        for (int slot = 0; slot < slots; slot++)
            if (!g_decode_rows.empty[slot] && positions[slot] > position) position = positions[slot];
        g_decode_slots = slots;
    }
    if (!session->prefilled ||
        (session->row_mode == VOLVOXAI_DECODE_ROW_REQUIRED && position < 0)) {
        if (run->rows) {
            g_decode_slots = 0;
            vx_decode_row_set_dispose(&g_decode_rows);
            run->rows = 0;
        }
        return -1;
    }
    run->use_row = session->mode == VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW && position >= 0;
    if (run->use_row) {
        row_ready = vx_incremental_prepare_hybrid_row_locked(position);
        if (row_ready == 0 && session->row_mode == VOLVOXAI_DECODE_ROW_AUTO) {
            /* Compatibility preflight has no side effects. Keep the valid GPU
             * prefill and permanently negotiate this session down to ordinary
             * dependency execution for the caller's actual changed closure. */
            session->mode = VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY;
            run->use_row = 0;
            row_ready = 1;
        }
    }
    g_execution_row = -1;
    if (row_ready == 1 &&
        (!run->use_row || volvoxai_engine_execution_row_valid_locked(position)) &&
        vx_incremental_run_begin_locked(&run->pass, run->use_row ? position : -1, stepwise) == 0)
        return 0;
    decode_run_finish_locked(run, -1);
    return -1;
}
