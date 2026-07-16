#include "backend.h"

#include <string.h>

static int backend_has_required_callbacks(const VxBackend* backend) {
    if (!backend || !backend->name || !backend->name[0] || !backend->init ||
        !backend->supports || !backend->run || !backend->teardown) return 0;
    if ((backend->upload == NULL) != (backend->download == NULL)) return 0;
    if ((backend->begin_forward == NULL) != (backend->end_forward == NULL)) return 0;
    if ((backend->mark_host == NULL) != (backend->sync_host == NULL)) return 0;
    if (backend->outputs_device_resident && !backend->sync_host) return 0;
    return 1;
}

int vx_backend_registry_validate(const VxBackend* const* backends,
                                 size_t count,
                                 const VxBackend* cpu_backend) {
    if (!backends || !cpu_backend || count == 0 || count > VX_BACKEND_MAX ||
        backends[count - 1] != cpu_backend) return -1;
    for (size_t index = 0; index < count; index++) {
        const VxBackend* backend = backends[index];
        if (!backend_has_required_callbacks(backend)) return -1;
        for (size_t previous = 0; previous < index; previous++) {
            if (backends[previous] == backend ||
                !strcmp(backends[previous]->name, backend->name)) return -1;
        }
    }
    return 0;
}

void vx_backend_registry_teardown(VxBackendRegistry* registry) {
    if (!registry) return;
    for (size_t index = registry->count; index > 0; index--) {
        size_t current = index - 1;
        if (registry->initialized[current] && registry->backends[current]->teardown)
            registry->backends[current]->teardown(registry->backends[current]->user_data);
    }
    memset(registry, 0, sizeof(*registry));
}

int vx_backend_registry_init(VxBackendRegistry* registry,
                             const VxBackend* const* backends,
                             size_t count,
                             const VxBackend* cpu_backend) {
    if (!registry || registry->ready ||
        vx_backend_registry_validate(backends, count, cpu_backend) != 0) return -1;
    memset(registry, 0, sizeof(*registry));
    for (size_t index = 0; index < count; index++) registry->backends[index] = backends[index];
    registry->count = count;
    registry->cpu_index = count - 1;
    for (size_t index = 0; index < count; index++) {
        int status = registry->backends[index]->init(registry->backends[index]->user_data);
        if (status == VX_BACKEND_INIT_READY) {
            registry->initialized[index] = 1;
            continue;
        }
        registry->backends[index]->teardown(registry->backends[index]->user_data);
        if (status == VX_BACKEND_INIT_UNAVAILABLE && index != registry->cpu_index &&
            !registry->backends[index]->required) continue;
        vx_backend_registry_teardown(registry);
        return -1;
    }
    if (!registry->initialized[registry->cpu_index]) {
        vx_backend_registry_teardown(registry);
        return -1;
    }
    registry->ready = 1;
    return 0;
}

int vx_backend_registry_dispatch(const VxBackendRegistry* registry,
                                 const Node* node,
                                 T* input,
                                 T* output,
                                 const VxBackend** selected) {
    if (selected) *selected = NULL;
    if (!registry || !registry->ready || !node) return VX_BACKEND_ERROR;
    for (size_t index = 0; index < registry->count; index++) {
        const VxBackend* backend;
        int status;
        if (!registry->initialized[index]) continue;
        backend = registry->backends[index];
        status = backend->supports(backend->user_data, node);
        if (status == VX_BACKEND_DECLINED) {
            if (index == registry->cpu_index) return VX_BACKEND_ERROR;
            continue;
        }
        if (status != VX_BACKEND_HANDLED) return VX_BACKEND_ERROR;
        status = backend->run(backend->user_data, node, input, output);
        if (status == VX_BACKEND_HANDLED) {
            if (selected) *selected = backend;
            return VX_BACKEND_HANDLED;
        }
        if (status == VX_BACKEND_DECLINED) {
            if (index == registry->cpu_index) return VX_BACKEND_ERROR;
            continue;
        }
        return VX_BACKEND_ERROR;
    }
    return VX_BACKEND_ERROR;
}

void vx_backend_registry_reset(VxBackendRegistry* registry) {
    if (!registry || !registry->ready) return;
    for (size_t index = 0; index < registry->count; index++) {
        const VxBackend* backend = registry->backends[index];
        if (registry->initialized[index] && backend->reset) backend->reset(backend->user_data);
    }
}

void vx_backend_registry_begin_forward(VxBackendRegistry* registry) {
    if (!registry || !registry->ready) return;
    for (size_t index = 0; index < registry->count; index++) {
        const VxBackend* backend = registry->backends[index];
        if (registry->initialized[index] && backend->begin_forward)
            backend->begin_forward(backend->user_data);
    }
}

int vx_backend_registry_end_forward(VxBackendRegistry* registry) {
    int result = 0;
    if (!registry || !registry->ready) return -1;
    for (size_t index = 0; index < registry->count; index++) {
        const VxBackend* backend = registry->backends[index];
        if (registry->initialized[index] && backend->end_forward &&
            backend->end_forward(backend->user_data) != 0) result = -1;
    }
    return result;
}

void vx_backend_registry_mark_host(VxBackendRegistry* registry,
                                   const void* host,
                                   size_t bytes,
                                   int is_weight) {
    if (!registry || !registry->ready) return;
    for (size_t index = 0; index < registry->count; index++) {
        const VxBackend* backend = registry->backends[index];
        if (registry->initialized[index] && backend->mark_host)
            backend->mark_host(backend->user_data, host, bytes, is_weight);
    }
}

int vx_backend_registry_sync_host(VxBackendRegistry* registry,
                                  const void* host,
                                  size_t bytes,
                                  int is_weight) {
    int result = 0;
    if (!registry || !registry->ready) return -1;
    for (size_t index = 0; index < registry->count; index++) {
        const VxBackend* backend = registry->backends[index];
        if (registry->initialized[index] && backend->sync_host &&
            !backend->sync_host(backend->user_data, host, bytes, is_weight)) result = -1;
    }
    return result;
}
