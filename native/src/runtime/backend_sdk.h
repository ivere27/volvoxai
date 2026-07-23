#ifndef VOLVOX_RUNTIME_BACKEND_SDK_H
#define VOLVOX_RUNTIME_BACKEND_SDK_H

#include "volvoxai_backend.h"

typedef struct VxProviderRegistry VxProviderRegistry;

typedef struct VxProviderBinding {
    const VxBackendProvider* provider;
    void* runtime_instance;
} VxProviderBinding;

VxProviderRegistry* vx_provider_registry_create(void);
void vx_provider_registry_destroy(VxProviderRegistry* registry);
VxStatus vx_provider_registry_register(VxProviderRegistry* registry,
                                       const VxRuntimeOptions* options,
                                       const VxBackendProvider* provider,
                                       VxReport* report);
int vx_provider_registry_find(VxProviderRegistry* registry,
                              const char* name,
                              VxProviderBinding* binding);

#endif /* VOLVOX_RUNTIME_BACKEND_SDK_H */
