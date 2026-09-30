#include "engine_core.h"
#include "engine_internal.h"
#include "incremental_runtime.h"
#include "paged_binding.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define VOLVOXAI_DECODE_SESSION_MAGIC UINT32_C(0x56584453)

struct VolvoxAIDecodeSession {
    uint32_t magic;
    VolvoxAIDecodeRowMode row_mode;
    VolvoxAIDecodeMode mode;
    VolvoxAIDecodeMode last_execution_mode;
    int previous_execution_row;
    int prefilled;
    int attached;
    int slots;
};

int vx_decode_session_active_locked(void) {
    return g_active_decode_session != NULL;
}

void vx_decode_session_invalidate_cache_locked(void) {
    if (!g_active_decode_session) return;
    g_active_decode_session->prefilled = 0;
    g_active_decode_session->last_execution_mode = VOLVOXAI_DECODE_MODE_NONE;
}

static int decode_session_model_lock(void) {
    int took_model_lock = !volvoxai_engine_model_route_lease_active();
    if (took_model_lock) volvoxai_engine_model_lock();
    return took_model_lock;
}

static void decode_session_model_unlock(int took_model_lock) {
    if (took_model_lock) volvoxai_engine_model_unlock();
}

static int decode_session_valid(const VolvoxAIDecodeSession* session) {
    return session && session->magic == VOLVOXAI_DECODE_SESSION_MAGIC;
}

static int decode_session_attached(const VolvoxAIDecodeSession* session) {
    return decode_session_valid(session) && session->attached &&
        g_active_decode_session == session;
}

void vx_decode_session_invalidate_model_locked(void) {
    if (!g_active_decode_session) return;
    g_active_decode_session->attached = 0;
    vx_decode_session_invalidate_cache_locked();
    g_active_decode_session->mode = VOLVOXAI_DECODE_MODE_NONE;
    g_active_decode_session = NULL;
}

VolvoxAIDecodeSession* volvoxai_engine_decode_session_create(
    const VolvoxAIDecodeSessionOptions* options) {
    VolvoxAIDecodeSessionOptions resolved = VOLVOXAI_DECODE_SESSION_OPTIONS_INIT;
    VolvoxAIDecodeSession* session;
    int row_supported;
    int took_model_lock;

    if (options) {
        if (options->struct_size != sizeof(*options) ||
            options->row_mode < VOLVOXAI_DECODE_ROW_AUTO ||
            options->row_mode > VOLVOXAI_DECODE_ROW_DISABLED ||
            options->slots < 1 || options->slots > INT32_MAX) return NULL;
        resolved = *options;
    }
    took_model_lock = decode_session_model_lock();
    if (!g_loaded || g_active_decode_session) {
        decode_session_model_unlock(took_model_lock);
        return NULL;
    }

    /* Dependency-aware execution is part of every native runtime profile.
     * Row execution is negotiated after model initialization because device
     * graph backends retain whole tensors rather than individual rows. */
    {
        if ((size_t)resolved.slots > SIZE_MAX / sizeof(int)) {
            decode_session_model_unlock(took_model_lock);
            return NULL;
        }
        int* probe = malloc((size_t)resolved.slots * sizeof(int));
        if (!probe) { decode_session_model_unlock(took_model_lock); return NULL; }
        for (uint32_t slot = 0; slot < resolved.slots; slot++) probe[slot] = 1;
        g_decode_slots = (int)resolved.slots;
        int initialized = vx_decode_row_set_init(&g_decode_rows, g_decode_slots, probe, NULL);
        free(probe);
        if (initialized != VX_DECODE_ROW_SET_OK) {
            g_decode_slots = 0;
            decode_session_model_unlock(took_model_lock);
            return NULL;
        }
        row_supported = vx_incremental_row_supported_locked();
        g_decode_slots = 0;
        vx_decode_row_set_dispose(&g_decode_rows);
    }
    if ((resolved.row_mode == VOLVOXAI_DECODE_ROW_REQUIRED || resolved.slots > 1) && !row_supported) {
        decode_session_model_unlock(took_model_lock);
        return NULL;
    }

    session = (VolvoxAIDecodeSession*)calloc(1, sizeof(*session));
    if (!session) {
        decode_session_model_unlock(took_model_lock);
        return NULL;
    }
    session->magic = VOLVOXAI_DECODE_SESSION_MAGIC;
    session->row_mode = resolved.row_mode;
    session->mode = row_supported && resolved.row_mode != VOLVOXAI_DECODE_ROW_DISABLED
        ? VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW
        : VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY;
    session->last_execution_mode = VOLVOXAI_DECODE_MODE_NONE;
    session->previous_execution_row = g_execution_row;
    session->attached = 1;
    session->slots = (int)resolved.slots;
    (void)resolved.require_incremental;
    g_active_decode_session = session;
    g_execution_row = -1;
    decode_session_model_unlock(took_model_lock);
    return session;
}

VolvoxAIDecodeMode volvoxai_engine_decode_session_mode(
    const VolvoxAIDecodeSession* session) {
    VolvoxAIDecodeMode mode;
    int took_model_lock = decode_session_model_lock();
    mode = decode_session_attached(session) ? session->mode : VOLVOXAI_DECODE_MODE_NONE;
    decode_session_model_unlock(took_model_lock);
    return mode;
}

VolvoxAIDecodeMode volvoxai_engine_decode_session_last_execution_mode(
    const VolvoxAIDecodeSession* session) {
    VolvoxAIDecodeMode mode;
    int took_model_lock = decode_session_model_lock();
    mode = decode_session_attached(session)
        ? session->last_execution_mode : VOLVOXAI_DECODE_MODE_NONE;
    decode_session_model_unlock(took_model_lock);
    return mode;
}

int volvoxai_engine_decode_session_prefilled(const VolvoxAIDecodeSession* session) {
    int prefilled;
    int took_model_lock = decode_session_model_lock();
    prefilled = decode_session_attached(session) ? session->prefilled : 0;
    decode_session_model_unlock(took_model_lock);
    return prefilled;
}

/*
 * A prefill or step as begin, nodes and end. The ordinary entry points run the
 * nodes back to back under one model lock; a debugger runs them one at a time
 * through the stepwise form below. Admission, row negotiation and the failure
 * policy are therefore the same code for both.
 */
typedef struct {
    VolvoxAIDecodeSession* session;
    int prefill, use_row, rows;
    VxIncrementalRun pass;
} VxDecodeRun;

static void decode_run_finish_locked(VxDecodeRun* run, int result) {
    VolvoxAIDecodeSession* session = run->session;
    if (run->prefill) {
        if (result != 0) vx_incremental_prepare_ordinary_locked();
        else {
            session->prefilled = 1;
            session->last_execution_mode = VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY;
        }
    } else {
        if (result != 0) {
            session->prefilled = 0;
            session->last_execution_mode = VOLVOXAI_DECODE_MODE_NONE;
            vx_incremental_prepare_ordinary_locked();
        } else {
            session->last_execution_mode = run->use_row
                ? VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW
                : VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY;
        }
        g_execution_row = -1;
    }
    if (run->rows) {
        g_decode_slots = 0;
        vx_decode_row_set_dispose(&g_decode_rows);
        run->rows = 0;
    }
}

/* positions == NULL with slots == 0 is a dependency step. */
static int decode_run_begin_locked(VxDecodeRun* run, VolvoxAIDecodeSession* session,
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

static int decode_run_end_locked(VxDecodeRun* run, int ok) {
    int result = vx_incremental_run_end_locked(&run->pass, ok);
    decode_run_finish_locked(run, result);
    return result;
}

static int decode_run_all_locked(VolvoxAIDecodeSession* session, int prefill,
                                 const int* positions, int slots) {
    VxDecodeRun run;
    int node, ok = 1;
    if (decode_run_begin_locked(&run, session, prefill, positions, slots, 0) != 0) return -1;
    while ((node = vx_incremental_run_next_locked(&run.pass)) >= 0 && node < g_nn)
        if (vx_incremental_run_node_locked(&run.pass, node) != 0) { ok = 0; break; }
    if (node < 0) ok = 0;
    return decode_run_end_locked(&run, ok);
}

int volvoxai_engine_decode_session_prefill(VolvoxAIDecodeSession* session) {
    int took_model_lock = decode_session_model_lock();
    int result = decode_run_all_locked(session, 1, NULL, 0);
    decode_session_model_unlock(took_model_lock);
    return result;
}

int volvoxai_engine_decode_session_step_rows(VolvoxAIDecodeSession* session,
                                             const int* positions, int slots) {
    int took_model_lock = decode_session_model_lock();
    int result = positions ? decode_run_all_locked(session, 0, positions, slots) : -1;
    decode_session_model_unlock(took_model_lock);
    return result;
}

int volvoxai_engine_decode_session_step(VolvoxAIDecodeSession* session, int position) {
    if (position >= 0) return volvoxai_engine_decode_session_step_rows(session, &position, 1);
    int took_model_lock = decode_session_model_lock();
    int result = position == -1 ? decode_run_all_locked(session, 0, NULL, 0) : -1;
    decode_session_model_unlock(took_model_lock);
    return result;
}

#if defined(VOLVOXAI_ENABLE_TRAINING) && VOLVOXAI_ENABLE_TRAINING
/* The debugger's stepwise run. Every call takes and drops the model lock; the
 * caller's context stays reserved to the run until end. */
struct VolvoxAIDecodeRun { VxDecodeRun run; };

VolvoxAIDecodeRun* volvoxai_engine_decode_run_begin(VolvoxAIDecodeSession* session,
    int prefill, const int* positions, int slots) {
    VolvoxAIDecodeRun* run = calloc(1, sizeof(*run));
    if (!run) return NULL;
    int took_model_lock = decode_session_model_lock();
    int result = decode_run_begin_locked(&run->run, session, prefill, positions, slots, 1);
    decode_session_model_unlock(took_model_lock);
    if (result == 0) return run;
    free(run);
    return NULL;
}

int volvoxai_engine_decode_run_next(VolvoxAIDecodeRun* run) {
    int took_model_lock = decode_session_model_lock();
    int node = vx_incremental_run_next_locked(&run->run.pass);
    decode_session_model_unlock(took_model_lock);
    return node;
}

int volvoxai_engine_decode_run_node(VolvoxAIDecodeRun* run, int node) {
    int took_model_lock = decode_session_model_lock();
    VxEngineState* state = vx_engine_state_current();
    state->last_failure_node_index = -1;
    int result = vx_incremental_run_node_locked(&run->run.pass, node);
    if (result != 0) state->last_failure_node_index = node;
    decode_session_model_unlock(took_model_lock);
    return result;
}

int volvoxai_engine_decode_run_end(VolvoxAIDecodeRun* run, int ok) {
    if (!run) return -1;
    int took_model_lock = decode_session_model_lock();
    int result = decode_run_end_locked(&run->run, ok);
    decode_session_model_unlock(took_model_lock);
    free(run);
    return result;
}
#endif

int volvoxai_engine_decode_session_reset(VolvoxAIDecodeSession* session) {
    int took_model_lock = decode_session_model_lock();
    if (!decode_session_attached(session)) {
        decode_session_model_unlock(took_model_lock);
        return -1;
    }
    vx_incremental_prepare_ordinary_locked();
    g_execution_row = -1;
    session->prefilled = 0;
    session->last_execution_mode = VOLVOXAI_DECODE_MODE_NONE;
    decode_session_model_unlock(took_model_lock);
    return 0;
}

void volvoxai_engine_decode_session_destroy(VolvoxAIDecodeSession* session) {
    int took_model_lock;
    if (!session) return;
    took_model_lock = decode_session_model_lock();
    if (!decode_session_valid(session)) {
        decode_session_model_unlock(took_model_lock);
        return;
    }
    if (decode_session_attached(session)) {
        vx_incremental_prepare_ordinary_locked();
        g_execution_row = session->previous_execution_row;
        g_active_decode_session = NULL;
    }
    session->attached = 0;
    session->magic = 0;
    free(session);
    decode_session_model_unlock(took_model_lock);
}
