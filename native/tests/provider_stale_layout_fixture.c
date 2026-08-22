#include "volvoxai.h"

#include <stddef.h>
#include <stdint.h>

/* Deliberately models the provider descriptor layout immediately before the
 * current exact-contract tail was added. This translation unit must not
 * include volvoxai_backend.h: it represents an already-built stale provider,
 * including the coincidentally identical numeric ABI discriminator. */
typedef struct StaleShapeDomainCapability {
    size_t struct_size;
    const char* proof_protocol;
    const char* resource_protocol;
    int support;
} StaleShapeDomainCapability;

typedef struct StaleBackendProvider {
    size_t struct_size;
    uint32_t abi_version;
    const char* name;
    void* user_data;
    uint32_t flags;
    StaleShapeDomainCapability shape_domain;
    VxStatus (*runtime_create)(void* user_data,
                               const VxRuntimeOptions* options,
                               void** out_runtime_instance,
                               VxReport* report);
    /* The remaining stale callbacks are layout-only. Function pointers have
     * the same representation regardless of their signatures on supported
     * native ABIs; none may be read because extent rejection comes first. */
    void (*remaining_callbacks[13])(void);
} StaleBackendProvider;

static int stale_runtime_create_calls;

static VxStatus stale_runtime_create(void* user_data,
                                     const VxRuntimeOptions* options,
                                     void** out_runtime_instance,
                                     VxReport* report) {
    (void)user_data;
    (void)options;
    (void)out_runtime_instance;
    (void)report;
    stale_runtime_create_calls++;
    return VX_STATUS_INTERNAL;
}

static const StaleBackendProvider stale_provider = {
    .struct_size = sizeof(StaleBackendProvider),
    .abi_version = 1u,
    .name = "stale-layout",
    .shape_domain = {
        .struct_size = sizeof(StaleShapeDomainCapability),
        .proof_protocol = "canonical-symbolic-domain-proof/v1",
        .resource_protocol = "bounded-resource-maxima/v1",
        .support = 1,
    },
    .runtime_create = stale_runtime_create,
};

const void* vx_stale_backend_provider_fixture(void) {
    return &stale_provider;
}

size_t vx_stale_backend_provider_fixture_size(void) {
    return sizeof(stale_provider);
}

int vx_stale_backend_provider_runtime_create_calls(void) {
    return stale_runtime_create_calls;
}
