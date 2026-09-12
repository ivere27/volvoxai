#include "volvoxai_backend.h"

#include "backend_sdk.h"
#include "generated/backend_vocabulary.h"

#include <ctype.h>
#include "vx_thread.h"
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
    VxMutex mutex;
    VxProviderSlot slots[VX_PROVIDER_CAPACITY];
    size_t count;
};

/* Deployment composition is process-scoped and immutable once registered.
 * Each Runtime snapshots this descriptor set and owns distinct provider
 * instances, so application handles never need access to an engine pointer. */
static VxMutex vx_composed_provider_mutex = VX_MUTEX_INITIALIZER;
static VxBackendProvider vx_composed_providers[VX_PROVIDER_CAPACITY];
static char vx_composed_provider_names[VX_PROVIDER_CAPACITY]
                                      [VX_BACKEND_NAME_CAPACITY];
static size_t vx_composed_provider_count;

typedef enum ProviderRegistrationOutcome {
    PROVIDER_REGISTRATION_DUPLICATE = 0,
    PROVIDER_REGISTRATION_CAPACITY,
    PROVIDER_REGISTRATION_RUNTIME_FAILURE,
    PROVIDER_REGISTRATION_MISSING_RUNTIME,
    PROVIDER_REGISTRATION_SUCCESS
} ProviderRegistrationOutcome;

static int provider_report_writable(const VxReport* report) {
    return report && report->struct_size == sizeof(*report);
}

static void provider_report_terminate_strings(VxReport* report) {
    report->backend[sizeof(report->backend) - 1u] = '\0';
    report->device[sizeof(report->device) - 1u] = '\0';

    report->message[sizeof(report->message) - 1u] = '\0';
    report->candidate_outcomes[sizeof(report->candidate_outcomes) - 1u] = '\0';
    report->route_evidence[sizeof(report->route_evidence) - 1u] = '\0';
    report->fallback_evidence[sizeof(report->fallback_evidence) - 1u] = '\0';
    report->offending_node[sizeof(report->offending_node) - 1u] = '\0';
    report->decode_state[sizeof(report->decode_state) - 1u] = '\0';
}

static void provider_report(VxReport* report,
                            VxStatus status,
                            VxOperationCode reason,
                            const char* message) {
    if (!provider_report_writable(report)) return;
    memset(report, 0, sizeof(*report));
    report->struct_size = sizeof(*report);
    report->status = status;
    report->stage = VX_STAGE_RUNTIME_CREATE;
    report->code = reason;
    snprintf(report->message, sizeof(report->message), "%s", message ? message : "");
}

static void provider_runtime_failure_report(
    VxReport* report,
    VxStatus status,
    const VxReport* callback_report) {
    VxOperationCode reason = status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY :
        status == VX_STATUS_BACKEND_UNAVAILABLE ? VX_CODE_BACKEND_UNAVAILABLE :
        VX_CODE_PROVIDER_RUNTIME_CREATE_FAILED;
    const char* message = status == VX_STATUS_OUT_OF_MEMORY ?
        "backend provider runtime allocation failed" :
        "backend provider runtime creation failed";
    if (!provider_report_writable(report)) return;
    if (callback_report && callback_report->struct_size == sizeof(*callback_report))
        *report = *callback_report;
    else
        memset(report, 0, sizeof(*report));
    report->struct_size = sizeof(*report);
    report->status = status;
    report->stage = VX_STAGE_RUNTIME_CREATE;
    provider_report_terminate_strings(report);
    if (report->code == VX_CODE_NONE)
        report->code = reason;
    if (!report->message[0])
        snprintf(report->message, sizeof(report->message), "%s", message);
}

static int provider_name_valid(const char* name) {
    size_t length;
    if (!name || !name[0] ||
        vx_backend_kind_from_name(name) != VX_BACKEND_KIND_UNSPECIFIED) return 0;
    length = strlen(name);
    if (length >= VX_BACKEND_NAME_CAPACITY || name[0] < 'a' || name[0] > 'z')
        return 0;
    for (size_t index = 1; index < length; index++) {
        unsigned char c = (unsigned char)name[index];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
              c == '.' || c == '_' || c == '-'))
            return 0;
    }
    return 1;
}

static int provider_shape_domain_valid(
    const VxBackendShapeDomainCapability* capability) {
    return capability && capability->struct_size == sizeof(*capability) &&
        capability->proof_protocol && capability->resource_protocol &&
        !strcmp(capability->proof_protocol,
                VX_BACKEND_SHAPE_PROOF_PROTOCOL) &&
        !strcmp(capability->resource_protocol,
                VX_BACKEND_RESOURCE_PROTOCOL) &&
        (capability->support == VX_BACKEND_SHAPE_DOMAIN_FULL ||
         capability->support == VX_BACKEND_SHAPE_DOMAIN_UNSUPPORTED);
}

static int provider_descriptor_valid(const VxBackendProvider* provider) {
    return provider && provider->struct_size == sizeof(*provider) &&
        provider->abi_version == VX_BACKEND_ABI_VERSION &&
        provider->exact_contract_marker ==
            VX_BACKEND_PROVIDER_EXACT_CONTRACT_MARKER &&
        provider->exact_contract_extent == sizeof(*provider) &&
        provider_name_valid(provider->name) && provider->flags == 0 &&
        provider_shape_domain_valid(&provider->shape_domain) &&
        provider->runtime_create && provider->runtime_destroy &&
        provider->compile && provider->compiled_destroy &&
        (!!provider->compiled_batch_contract ==
         !!provider->context_execute_batch) &&
        provider->context_create && provider->context_execute &&
        provider->context_destroy;
}

VxStatus vx_backend_register_provider(const VxBackendProvider* provider,
                                      VxReport* report) {
    size_t index;
    VxBackendProvider* stored;
    if (!provider_report_writable(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (provider && provider->struct_size == sizeof(*provider) &&
        provider->abi_version != VX_BACKEND_ABI_VERSION) {
        provider_report(report, VX_STATUS_ABI_UNSUPPORTED, VX_CODE_ABI_UNSUPPORTED,
                        "backend provider does not match the current exact ABI");
        return VX_STATUS_ABI_UNSUPPORTED;
    }
    if (!provider_descriptor_valid(provider)) {
        provider_report(report, VX_STATUS_INVALID_ARGUMENT, VX_CODE_INVALID_PROVIDER,
                        "backend provider descriptor does not match the current exact contract");
        return VX_STATUS_INVALID_ARGUMENT;
    }

    vx_mutex_lock(&vx_composed_provider_mutex);
    for (index = 0; index < vx_composed_provider_count; index++) {
        if (!strcmp(vx_composed_provider_names[index], provider->name)) {
            vx_mutex_unlock(&vx_composed_provider_mutex);
            provider_report(report, VX_STATUS_INVALID_ARGUMENT,
                            VX_CODE_DUPLICATE_PROVIDER,
                            "backend provider name is already registered");
            return VX_STATUS_INVALID_ARGUMENT;
        }
    }
    if (vx_composed_provider_count >= VX_PROVIDER_CAPACITY) {
        vx_mutex_unlock(&vx_composed_provider_mutex);
        provider_report(report, VX_STATUS_OUT_OF_MEMORY, VX_CODE_OUT_OF_MEMORY,
                        "backend provider composition capacity is full");
        return VX_STATUS_OUT_OF_MEMORY;
    }

    index = vx_composed_provider_count++;
    memcpy(vx_composed_provider_names[index], provider->name,
           strlen(provider->name) + 1u);
    stored = &vx_composed_providers[index];
    *stored = *provider;
    stored->name = vx_composed_provider_names[index];
    stored->shape_domain.proof_protocol = VX_BACKEND_SHAPE_PROOF_PROTOCOL;
    stored->shape_domain.resource_protocol = VX_BACKEND_RESOURCE_PROTOCOL;
    vx_mutex_unlock(&vx_composed_provider_mutex);

    provider_report(report, VX_STATUS_OK, VX_CODE_NONE,
                    "backend provider registered for future runtimes");
    return VX_STATUS_OK;
}

VxStatus vx_provider_registry_attach_composed(
    VxProviderRegistry* registry,
    const VxRuntimeOptions* options,
    VxReport* report) {
    VxBackendProvider providers[VX_PROVIDER_CAPACITY];
    size_t count;
    size_t index;
    if (!registry || !options || !provider_report_writable(report))
        return VX_STATUS_INVALID_ARGUMENT;

    vx_mutex_lock(&vx_composed_provider_mutex);
    count = vx_composed_provider_count;
    if (count) memcpy(providers, vx_composed_providers,
                      count * sizeof(*providers));
    vx_mutex_unlock(&vx_composed_provider_mutex);

    for (index = 0; index < count; index++) {
        VxReport candidate = VX_REPORT_INIT;
        VxStatus status = vx_provider_registry_register(
            registry, options, &providers[index], &candidate);
        /* Optional hardware may be absent on this host. It is intentionally
         * omitted from ListBackends and an exact CompileModel requirement will
         * report the ordinary unavailable status later. */
        if (status == VX_STATUS_BACKEND_UNAVAILABLE) continue;
        if (status != VX_STATUS_OK) {
            *report = candidate;
            provider_report_terminate_strings(report);
            return status;
        }
    }
    return VX_STATUS_OK;
}

VxProviderRegistry* vx_provider_registry_create(void) {
    VxProviderRegistry* registry =
        (VxProviderRegistry*)calloc(1, sizeof(*registry));
    if (!registry) return NULL;
    if (vx_mutex_init(&registry->mutex) != 0) {
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
    vx_mutex_destroy(&registry->mutex);
    free(registry);
}

VxStatus vx_provider_registry_register(VxProviderRegistry* registry,
                                       const VxRuntimeOptions* options,
                                       const VxBackendProvider* provider,
                                       VxReport* report) {
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    ProviderRegistrationOutcome outcome = PROVIDER_REGISTRATION_DUPLICATE;
    void* runtime_instance = NULL;
    VxReport callback_report = VX_REPORT_INIT;
    if (!registry || !options) {
        provider_report(report, status, VX_CODE_INVALID_PROVIDER,
                        "provider registry or runtime options are invalid");
        return status;
    }
    if (provider && provider->struct_size == sizeof(*provider) &&
        provider->abi_version != VX_BACKEND_ABI_VERSION) {
        provider_report(report, VX_STATUS_ABI_UNSUPPORTED, VX_CODE_ABI_UNSUPPORTED,
                        "backend provider does not match the current exact ABI");
        return VX_STATUS_ABI_UNSUPPORTED;
    }
    if (!provider_descriptor_valid(provider)) {
        provider_report(report, status, VX_CODE_INVALID_PROVIDER,
                        "backend provider descriptor does not match the current exact contract");
        return status;
    }
    vx_mutex_lock(&registry->mutex);
    for (size_t index = 0; index < registry->count; index++) {
        if (!strcmp(registry->slots[index].name, provider->name)) goto done;
    }
    if (registry->count >= VX_PROVIDER_CAPACITY) {
        status = VX_STATUS_OUT_OF_MEMORY;
        outcome = PROVIDER_REGISTRATION_CAPACITY;
        goto done;
    }
    status = provider->runtime_create(provider->user_data, options,
                                      &runtime_instance, &callback_report);
    if (status != VX_STATUS_OK) {
        runtime_instance = NULL;
        outcome = PROVIDER_REGISTRATION_RUNTIME_FAILURE;
        goto done;
    }
    if (!runtime_instance) {
        status = VX_STATUS_BACKEND_UNAVAILABLE;
        outcome = PROVIDER_REGISTRATION_MISSING_RUNTIME;
        goto done;
    }
    VxProviderSlot* slot = &registry->slots[registry->count++];
    memset(slot, 0, sizeof(*slot));
    slot->provider = *provider;
    memcpy(slot->name, provider->name, strlen(provider->name) + 1u);
    slot->provider.name = slot->name;
    slot->provider.shape_domain.proof_protocol =
        VX_BACKEND_SHAPE_PROOF_PROTOCOL;
    slot->provider.shape_domain.resource_protocol =
        VX_BACKEND_RESOURCE_PROTOCOL;
    slot->runtime_instance = runtime_instance;
    runtime_instance = NULL;
    status = VX_STATUS_OK;
    outcome = PROVIDER_REGISTRATION_SUCCESS;
done:
    vx_mutex_unlock(&registry->mutex);
    if (outcome == PROVIDER_REGISTRATION_RUNTIME_FAILURE) {
        provider_runtime_failure_report(report, status, &callback_report);
    } else if (outcome == PROVIDER_REGISTRATION_CAPACITY) {
        provider_report(report, status, VX_CODE_OUT_OF_MEMORY,
                        "backend provider registry is full");
    } else if (outcome == PROVIDER_REGISTRATION_MISSING_RUNTIME) {
        provider_report(report, status, VX_CODE_BACKEND_UNAVAILABLE,
                        "backend provider runtime creation returned no runtime");
    } else if (outcome == PROVIDER_REGISTRATION_SUCCESS) {
        provider_report(report, status, VX_CODE_NONE, "backend provider registered");
    } else {
        provider_report(report, status, VX_CODE_PROVIDER_ALREADY_REGISTERED,
                        "backend provider name is already registered");
    }
    return status;
}

int vx_provider_registry_find(VxProviderRegistry* registry,
                              const char* name,
                              VxProviderBinding* binding) {
    int found = 0;
    if (!registry || !name || !binding) return 0;
    memset(binding, 0, sizeof(*binding));
    vx_mutex_lock(&registry->mutex);
    for (size_t index = 0; index < registry->count; index++) {
        if (!strcmp(registry->slots[index].name, name)) {
            binding->provider = &registry->slots[index].provider;
            binding->runtime_instance = registry->slots[index].runtime_instance;
            found = 1;
            break;
        }
    }
    vx_mutex_unlock(&registry->mutex);
    return found;
}

VxStatus vx_provider_registry_snapshot_names(
    VxProviderRegistry* registry,
    char (**out_names)[VX_BACKEND_NAME_CAPACITY],
    size_t* out_count) {
    char (*names)[VX_BACKEND_NAME_CAPACITY] = NULL;
    if (!registry || !out_names || !out_count) return VX_STATUS_INVALID_ARGUMENT;
    *out_names = NULL;
    *out_count = 0u;
    vx_mutex_lock(&registry->mutex);
    if (registry->count) {
        names = (char (*)[VX_BACKEND_NAME_CAPACITY])calloc(
            registry->count, sizeof(*names));
        if (!names) {
            vx_mutex_unlock(&registry->mutex);
            return VX_STATUS_OUT_OF_MEMORY;
        }
        for (size_t index = 0u; index < registry->count; index++)
            memcpy(names[index], registry->slots[index].name,
                   sizeof(names[index]));
    }
    *out_names = names;
    *out_count = registry->count;
    vx_mutex_unlock(&registry->mutex);
    return VX_STATUS_OK;
}
