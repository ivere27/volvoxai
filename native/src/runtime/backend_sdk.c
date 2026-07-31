#include "volvoxai_backend.h"

#include "backend_sdk.h"

#include <ctype.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VX_PROVIDER_CAPACITY = 16 };

typedef struct VxProviderSlot {
    VxBackendProvider provider;
    char name[VX_BACKEND_NAME_CAPACITY];
    void* runtime_instance;
} VxProviderSlot;

struct VxProviderRegistry {
    pthread_mutex_t mutex;
    VxProviderSlot slots[VX_PROVIDER_CAPACITY];
    size_t count;
};

static void provider_report(VxReport* report,
                            VxStatus status,
                            const char* reason,
                            const char* message) {
    size_t struct_size;
    if (!report || report->struct_size < sizeof(*report)) return;
    struct_size = report->struct_size;
    memset(report, 0, sizeof(*report));
    report->struct_size = struct_size;
    report->status = status;
    report->stage = VX_STAGE_RUNTIME_CREATE;
    snprintf(report->reason, sizeof(report->reason), "%s", reason ? reason : "");
    snprintf(report->message, sizeof(report->message), "%s", message ? message : "");
}

static int provider_name_valid(const char* name) {
    size_t length;
    if (!name || !name[0] || !strcmp(name, "cpu") ||
        !strcmp(name, "vulkan") || !strcmp(name, "opengl") ||
        !strcmp(name, "metal") || !strcmp(name, "nnapi") ||
        !strcmp(name, "cuda")) return 0;
    length = strlen(name);
    if (length >= VX_BACKEND_NAME_CAPACITY || name[0] < 'a' || name[0] > 'z')
        return 0;
    for (size_t index = 1; index < length; index++) {
        unsigned char c = (unsigned char)name[index];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-'))
            return 0;
    }
    return 1;
}

static int provider_descriptor_valid(const VxBackendProvider* provider) {
    return provider && provider->struct_size >= sizeof(*provider) &&
        provider->abi_version == VX_BACKEND_ABI_VERSION &&
        provider_name_valid(provider->name) && provider->flags == 0 &&
        provider->runtime_create && provider->runtime_destroy &&
        provider->compile && provider->compiled_destroy &&
        provider->context_create && provider->context_set_input &&
        provider->context_execute && provider->context_destroy;
}

VxProviderRegistry* vx_provider_registry_create(void) {
    VxProviderRegistry* registry =
        (VxProviderRegistry*)calloc(1, sizeof(*registry));
    if (!registry) return NULL;
    if (pthread_mutex_init(&registry->mutex, NULL) != 0) {
        free(registry);
        return NULL;
    }
    return registry;
}

void vx_provider_registry_destroy(VxProviderRegistry* registry) {
    if (!registry) return;
    for (size_t index = registry->count; index > 0; index--) {
        VxProviderSlot* slot = &registry->slots[index - 1u];
        if (slot->runtime_instance)
            slot->provider.runtime_destroy(slot->runtime_instance);
    }
    pthread_mutex_destroy(&registry->mutex);
    free(registry);
}

VxStatus vx_provider_registry_register(VxProviderRegistry* registry,
                                       const VxRuntimeOptions* options,
                                       const VxBackendProvider* provider,
                                       VxReport* report) {
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    void* runtime_instance = NULL;
    VxReport callback_report = VX_REPORT_INIT;
    if (!registry || !options) {
        provider_report(report, status, "INVALID_PROVIDER",
                        "provider registry or runtime options are invalid");
        return status;
    }
    if (provider && provider->struct_size >=
            offsetof(VxBackendProvider, abi_version) + sizeof(provider->abi_version) &&
        provider->abi_version != VX_BACKEND_ABI_VERSION) {
        provider_report(report, VX_STATUS_ABI_UNSUPPORTED, "ABI_UNSUPPORTED",
                        "backend provider ABI version is unsupported");
        return VX_STATUS_ABI_UNSUPPORTED;
    }
    if (!provider_descriptor_valid(provider)) {
        provider_report(report, status, "INVALID_PROVIDER",
                        "backend provider descriptor is invalid");
        return status;
    }
    pthread_mutex_lock(&registry->mutex);
    for (size_t index = 0; index < registry->count; index++) {
        if (!strcmp(registry->slots[index].name, provider->name)) goto done;
    }
    if (registry->count >= VX_PROVIDER_CAPACITY) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    status = provider->runtime_create(provider->user_data, options,
                                      &runtime_instance, &callback_report);
    if (status != VX_STATUS_OK || !runtime_instance) {
        if (status == VX_STATUS_OK) status = VX_STATUS_BACKEND_UNAVAILABLE;
        goto done;
    }
    VxProviderSlot* slot = &registry->slots[registry->count++];
    memset(slot, 0, sizeof(*slot));
    slot->provider = *provider;
    memcpy(slot->name, provider->name, strlen(provider->name) + 1u);
    slot->provider.name = slot->name;
    slot->runtime_instance = runtime_instance;
    runtime_instance = NULL;
    status = VX_STATUS_OK;
done:
    pthread_mutex_unlock(&registry->mutex);
    if (runtime_instance) provider->runtime_destroy(runtime_instance);
    provider_report(report, status,
                    status == VX_STATUS_OK ? "OK" :
                    status == VX_STATUS_BACKEND_UNAVAILABLE ?
                        "BACKEND_UNAVAILABLE" : "PROVIDER_ALREADY_REGISTERED",
                    status == VX_STATUS_OK ? "backend provider registered" :
                    status == VX_STATUS_BACKEND_UNAVAILABLE ?
                        "backend provider runtime creation failed" :
                        "backend provider name is unavailable");
    return status;
}

int vx_provider_registry_find(VxProviderRegistry* registry,
                              const char* name,
                              VxProviderBinding* binding) {
    int found = 0;
    if (!registry || !name || !binding) return 0;
    memset(binding, 0, sizeof(*binding));
    pthread_mutex_lock(&registry->mutex);
    for (size_t index = 0; index < registry->count; index++) {
        if (!strcmp(registry->slots[index].name, name)) {
            binding->provider = &registry->slots[index].provider;
            binding->runtime_instance = registry->slots[index].runtime_instance;
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(&registry->mutex);
    return found;
}
