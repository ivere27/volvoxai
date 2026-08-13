#include "engine_core.h"
#include "engine_internal.h"
#include "incremental_runtime.h"

#include <stdint.h>
#include <stdlib.h>

#define VOLVOXAI_DECODE_SESSION_MAGIC UINT32_C(0x56584453)

struct VolvoxAIDecodeSession {
    uint32_t magic;
    VolvoxAIDecodeRowMode row_mode;
    VolvoxAIDecodeMode mode;
    VolvoxAIDecodeMode last_execution_mode;
    int previous_execution_row;
    int seeded;
    int attached;
};

int vx_decode_session_active_locked(void) {
    return g_active_decode_session != NULL;
}

void vx_decode_session_invalidate_cache_locked(void) {
    if (!g_active_decode_session) return;
    g_active_decode_session->seeded = 0;
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
            options->row_mode > VOLVOXAI_DECODE_ROW_DISABLED) return NULL;
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
    row_supported = vx_incremental_row_supported_locked();
    if (resolved.row_mode == VOLVOXAI_DECODE_ROW_REQUIRED && !row_supported) {
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

int volvoxai_engine_decode_session_seeded(const VolvoxAIDecodeSession* session) {
    int seeded;
    int took_model_lock = decode_session_model_lock();
    seeded = decode_session_attached(session) ? session->seeded : 0;
    decode_session_model_unlock(took_model_lock);
    return seeded;
}

int volvoxai_engine_decode_session_seed(VolvoxAIDecodeSession* session) {
    int result;
    int took_model_lock = decode_session_model_lock();
    if (!decode_session_attached(session)) {
        decode_session_model_unlock(took_model_lock);
        return -1;
    }
    vx_incremental_prepare_ordinary_locked();
    g_execution_row = -1;
    session->seeded = 0;
    session->last_execution_mode = VOLVOXAI_DECODE_MODE_NONE;
    result = volvoxai_engine_forward_incremental_locked();
    if (result != 0) {
        vx_incremental_prepare_ordinary_locked();
        decode_session_model_unlock(took_model_lock);
        return -1;
    }
    session->seeded = 1;
    session->last_execution_mode = VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY;
    decode_session_model_unlock(took_model_lock);
    return 0;
}

int volvoxai_engine_decode_session_step(VolvoxAIDecodeSession* session, int position) {
    int use_row;
    int row_ready = 1;
    int result;
    int took_model_lock = decode_session_model_lock();
    if (!decode_session_attached(session) || !session->seeded || position < -1) {
        decode_session_model_unlock(took_model_lock);
        return -1;
    }
    if (session->row_mode == VOLVOXAI_DECODE_ROW_REQUIRED && position < 1) {
        decode_session_model_unlock(took_model_lock);
        return -1;
    }

    use_row = session->mode == VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW && position >= 1;
    if (use_row) {
        row_ready = vx_incremental_prepare_hybrid_row_locked(position);
        if (row_ready == 0 && session->row_mode == VOLVOXAI_DECODE_ROW_AUTO) {
            /* Compatibility preflight has no side effects. Keep the valid GPU
             * seed and permanently negotiate this session down to ordinary
             * dependency execution for the caller's actual changed closure. */
            session->mode = VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY;
            use_row = 0;
            row_ready = 1;
        }
    }
    if (row_ready != 1) {
        session->seeded = 0;
        session->last_execution_mode = VOLVOXAI_DECODE_MODE_NONE;
        vx_incremental_prepare_ordinary_locked();
        g_execution_row = -1;
        decode_session_model_unlock(took_model_lock);
        return -1;
    }
    g_execution_row = -1;
    result = use_row
        ? volvoxai_engine_forward_incremental_row_locked(position)
        : volvoxai_engine_forward_incremental_locked();
    if (result != 0) {
        session->seeded = 0;
        session->last_execution_mode = VOLVOXAI_DECODE_MODE_NONE;
        vx_incremental_prepare_ordinary_locked();
        g_execution_row = -1;
        decode_session_model_unlock(took_model_lock);
        return -1;
    }
    session->last_execution_mode = use_row
        ? VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW
        : VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY;
    g_execution_row = -1;
    decode_session_model_unlock(took_model_lock);
    return 0;
}

int volvoxai_engine_decode_session_reset(VolvoxAIDecodeSession* session) {
    int took_model_lock = decode_session_model_lock();
    if (!decode_session_attached(session)) {
        decode_session_model_unlock(took_model_lock);
        return -1;
    }
    vx_incremental_prepare_ordinary_locked();
    g_execution_row = -1;
    session->seeded = 0;
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
