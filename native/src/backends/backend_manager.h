#ifndef VOLVOXAI_BACKEND_MANAGER_H
#define VOLVOXAI_BACKEND_MANAGER_H

#include "engine_core.h"

int vx_backend_manager_activate(VolvoxAIEngineBackend backend);
void vx_backend_manager_deactivate(void);
VolvoxAIEngineBackend vx_backend_manager_current(void);
const char* vx_backend_manager_name(void);

#endif
