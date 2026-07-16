#ifndef VOLVOX_RUNTIME_BACKEND_SDK_H
#define VOLVOX_RUNTIME_BACKEND_SDK_H

#include "backend.h"

/* Internal selection bridge.  Registration is public, but registered entries
 * remain dormant until engine.c selects exactly one by name.  Callers of the
 * selection helpers hold the engine model lock. */
int vx_sdk_select_backend(const char* name);
void vx_sdk_clear_backend(void);
const char* vx_sdk_selected_name(void);
const VxBackend* vx_sdk_selected_backend(void);
const VxBackend* vx_sdk_find_backend(const char* name);

#endif /* VOLVOX_RUNTIME_BACKEND_SDK_H */
