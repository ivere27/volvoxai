#ifndef VOLVOXAI_INCREMENTAL_RUNTIME_H
#define VOLVOXAI_INCREMENTAL_RUNTIME_H

#include "engine_internal.h"

/* Private inference scheduler. Callers hold the engine model lock. */
void vx_incremental_invalidate_locked(void);
void vx_incremental_prepare_ordinary_locked(void);
void vx_incremental_mark_tensor_locked(T* tensor);
int vx_incremental_forward_locked(int row);
int vx_incremental_row_supported_locked(void);
/*
 * The token extent and the trailing width of a sequence tensor, after skipping
 * leading unit axes, so [S,D], [1,S,D] and [1,1,S,D] all answer the same.
 *
 * It lives in the header because both the dispatcher and the row planner ask
 * it, and because restating "the token axis is shape[1]" per caller is what
 * made row execution silently unreachable for the sequence-major spelling in
 * three separate places.  static inline rather than a linked symbol: not every
 * target that compiles the row planner also links the dispatcher.
 */
static inline int vx_incremental_row_extent(const T* tensor, long* width) {
    long trailing = 1;
    int axis;
    if (!tensor || tensor->ndim <= 0) return 0;
    for (axis = 0; axis < tensor->ndim && tensor->shape[axis] == 1; axis++) {}
    if (axis >= tensor->ndim) return 0;
    for (int rest = axis + 1; rest < tensor->ndim; rest++)
        trailing *= tensor->shape[rest];
    if (tensor->shape[axis] <= 0 || trailing <= 0) return 0;
    if (width) *width = trailing;
    return (int)tensor->shape[axis];
}

static inline int vx_incremental_row_tensor_3d(const T* tensor, int row) {
    long width = 0;
    return vx_incremental_row_extent(tensor, &width) > row && width > 0;
}

/* Returns 1 when a row can execute now, 0 for a compatibility fallback, and
 * -1 when synchronization failed and the retained cache is no longer safe. */
int vx_incremental_prepare_hybrid_row_locked(int row);
int vx_incremental_hybrid_row_active_locked(void);
void vx_incremental_shutdown_locked(void);

#endif
