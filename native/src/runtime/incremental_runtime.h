#ifndef VOLVOXAI_INCREMENTAL_RUNTIME_H
#define VOLVOXAI_INCREMENTAL_RUNTIME_H

#include "engine_internal.h"

/* Private inference scheduler. Callers hold the engine model lock. */
void vx_incremental_invalidate_locked(void);
void vx_incremental_prepare_ordinary_locked(void);
void vx_incremental_mark_tensor_locked(T* tensor);
int vx_incremental_forward_locked(int row);
int vx_incremental_row_supported_locked(void);
/* Returns 1 when a row can execute now, 0 for a compatibility fallback, and
 * -1 when synchronization failed and the retained cache is no longer safe. */
int vx_incremental_prepare_hybrid_row_locked(int row);
int vx_incremental_hybrid_row_active_locked(void);
void vx_incremental_shutdown_locked(void);

#endif
