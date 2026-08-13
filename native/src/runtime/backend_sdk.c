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
    report->reason[sizeof(report->reason) - 1u] = '\0';
    report->message[sizeof(report->message) - 1u] = '\0';
    report->candidate_outcomes[sizeof(report->candidate_outcomes) - 1u] = '\0';
    report->route_evidence[sizeof(report->route_evidence) - 1u] = '\0';
    report->fallback_evidence[sizeof(report->fallback_evidence) - 1u] = '\0';
    report->offending_node[sizeof(report->offending_node) - 1u] = '\0';
    report->decode_state[sizeof(report->decode_state) - 1u] = '\0';
}

static void provider_report(VxReport* report,
                            VxStatus status,
                            const char* reason,
                            const char* message) {
    if (!provider_report_writable(report)) return;
    memset(report, 0, sizeof(*report));
    report->struct_size = sizeof(*report);
    report->status = status;
    report->stage = VX_STAGE_RUNTIME_CREATE;
    snprintf(report->reason, sizeof(report->reason), "%s", reason ? reason : "");
    snprintf(report->message, sizeof(report->message), "%s", message ? message : "");
}

static void provider_runtime_failure_report(
    VxReport* report,
    VxStatus status,
    const VxReport* callback_report) {
    const char* reason = status == VX_STATUS_OUT_OF_MEMORY ? "OUT_OF_MEMORY" :
        status == VX_STATUS_BACKEND_UNAVAILABLE ? "BACKEND_UNAVAILABLE" :
        "PROVIDER_RUNTIME_CREATE_FAILED";
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
    if (!report->reason[0])
        snprintf(report->reason, sizeof(report->reason), "%s", reason);
    if (!report->message[0])
        snprintf(report->message, sizeof(report->message), "%s", message);
}

static int provider_name_valid(const char* name) {
    size_t length;
    if (!name || !name[0] || !strcmp(name, "cpu") ||
        !strcmp(name, "vulkan") || !strcmp(name, "opengl") ||
        !strcmp(name, "metal") || !strcmp(name, "cuda")) return 0;
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
    ProviderRegistrationOutcome outcome = PROVIDER_REGISTRATION_DUPLICATE;
    void* runtime_instance = NULL;
    VxReport callback_report = VX_REPORT_INIT;
    if (!registry || !options) {
        provider_report(report, status, "INVALID_PROVIDER",
                        "provider registry or runtime options are invalid");
        return status;
    }
    if (provider && provider->struct_size == sizeof(*provider) &&
        provider->abi_version != VX_BACKEND_ABI_VERSION) {
        provider_report(report, VX_STATUS_ABI_UNSUPPORTED, "ABI_UNSUPPORTED",
                        "backend provider does not match the current exact ABI");
        return VX_STATUS_ABI_UNSUPPORTED;
    }
    if (!provider_descriptor_valid(provider)) {
        provider_report(report, status, "INVALID_PROVIDER",
                        "backend provider descriptor does not match the current exact contract");
        return status;
    }
    pthread_mutex_lock(&registry->mutex);
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
    pthread_mutex_unlock(&registry->mutex);
    if (outcome == PROVIDER_REGISTRATION_RUNTIME_FAILURE) {
        provider_runtime_failure_report(report, status, &callback_report);
    } else if (outcome == PROVIDER_REGISTRATION_CAPACITY) {
        provider_report(report, status, "OUT_OF_MEMORY",
                        "backend provider registry is full");
    } else if (outcome == PROVIDER_REGISTRATION_MISSING_RUNTIME) {
        provider_report(report, status, "BACKEND_UNAVAILABLE",
                        "backend provider runtime creation returned no runtime");
    } else if (outcome == PROVIDER_REGISTRATION_SUCCESS) {
        provider_report(report, status, "OK", "backend provider registered");
    } else {
        provider_report(report, status, "PROVIDER_ALREADY_REGISTERED",
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
