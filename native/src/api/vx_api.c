#include "vx_api.h"
#include "vx_thread.h"

#include <stdlib.h>

int vx_api_install_buffer_handlers(SynurangInstance*, VxApiRegistry*);
int vx_api_install_platform_handlers(SynurangInstance*, VxApiRegistry*);
int vx_api_install_inference_handlers(SynurangInstance*, VxApiRegistry*);
int vx_api_install_scheduler_handlers(SynurangInstance*, VxApiRegistry*);
int vx_api_install_text_handlers(SynurangInstance*, VxApiRegistry*);

/* One build macro selects the profile. Every target that links
 * FULL_PROFILE_SRCS sets VOLVOXAI_ENABLE_TRAINING=1 and only those targets do,
 * so it is the exact condition for registering full-profile handlers. The local
 * name says what is being tested here: training kernels are a different axis
 * than the API services a profile publishes. */
#if defined(VOLVOXAI_ENABLE_TRAINING) && VOLVOXAI_ENABLE_TRAINING
#define VX_API_FULL_PROFILE 1
#else
#define VX_API_FULL_PROFILE 0
#endif

#if VX_API_FULL_PROFILE
int vx_api_install_planning_handlers(SynurangInstance*, VxApiRegistry*);
int vx_api_install_training_handlers(SynurangInstance*, VxApiRegistry*);
int vx_api_install_quantization_handlers(SynurangInstance*, VxApiRegistry*);
#endif

/* The module ABI treats this pointer as opaque. Keeping the generated call
 * instance and engine registry together gives native and WASM identical
 * ownership: no constructor, global handler table, or cross-owner handles. */
typedef struct VxApiModule {
    SynurangInstance* calls;
    VxApiRegistry* registry;
    SynurangWakeupFn wakeup;
    void* wakeup_context;
    int poll_pending_next;
    int closing;
#if !defined(VOLVOXAI_NO_THREADS)
    pthread_t cleanup_thread;
    atomic_int cleanup_done;
    int cleanup_started;
#endif
} VxApiModule;

VxApiRegistry* vx_api_module_registry(SynurangInstance* instance) {
    VxApiModule* module = (VxApiModule*)instance;
    return module && module->calls && !module->closing ? module->registry : NULL;
}

static SynurangInstance* vx_api_module_create(const SynurangRuntimeOptions* options) {
    VxApiModule* module = calloc(1, sizeof(*module));
    if (!module) return NULL;
    module->registry = vx_api_registry_create();
    module->calls = synurang_instance_create(options);
    if (!module->registry || !module->calls) goto failed;
    if (options) vx_api_registry_set_wakeup(module->registry,
                                           options->wakeup, options->wakeup_user_data);
    if (options) {
        module->wakeup = options->wakeup;
        module->wakeup_context = options->wakeup_user_data;
    }
#if !defined(VOLVOXAI_NO_THREADS)
    atomic_init(&module->cleanup_done, 0);
#endif
#define VX_REGISTER(service) \
    if (vx_api_install_##service##_handlers(module->calls, module->registry) \
        != SYNURANG_OK) goto failed
    VX_REGISTER(buffer);
    VX_REGISTER(platform);
    VX_REGISTER(inference);
    VX_REGISTER(scheduler);
    VX_REGISTER(text);
#if VX_API_FULL_PROFILE
    /* Authoring builds, edits and serializes graphs and reads/writes
       safetensors. Inference lowers graphs inside LoadModel/CompileModel. */
    VX_REGISTER(planning);
    VX_REGISTER(training);
    VX_REGISTER(quantization);
#endif
#undef VX_REGISTER
    return (SynurangInstance*)module;
failed:
    /* No calls can exist before create returns, so shutdown is immediate. */
    if (module->calls) (void)synurang_instance_destroy(module->calls);
    vx_api_registry_destroy(module->registry);
    free(module);
    return NULL;
}

#if !defined(VOLVOXAI_NO_THREADS)
static void* vx_api_module_cleanup(void* context) {
    VxApiModule* module = context;
    vx_api_registry_destroy(module->registry);
    atomic_store_explicit(&module->cleanup_done, 1, memory_order_release);
    if (module->wakeup) module->wakeup(module->wakeup_context);
    return NULL;
}
#endif

static int vx_api_module_destroy(SynurangInstance* instance) {
    if (!instance) return SYNURANG_INVALID_ARGUMENT;
    VxApiModule* module = (VxApiModule*)instance;
    module->closing = 1;
    if (module->calls) {
        int status = synurang_instance_destroy(module->calls);
        if (status != SYNURANG_OK) return status;
        module->calls = NULL;
    }
    /* A retained producer can still reference engine objects until the call
     * runtime reports OK. Retire every public handle only after that point. */
#if !defined(VOLVOXAI_NO_THREADS)
    if (!module->cleanup_started) {
        if (pthread_create(&module->cleanup_thread, NULL, vx_api_module_cleanup, module) == 0) {
            module->cleanup_started = 1;
            return SYNURANG_PENDING;
        }
        /* Resource exhaustion cannot leave an uncloseable owner. */
        vx_api_registry_destroy(module->registry);
    } else {
        if (!atomic_load_explicit(&module->cleanup_done, memory_order_acquire))
            return SYNURANG_PENDING;
        (void)pthread_join(module->cleanup_thread, NULL);
    }
#else
    vx_api_registry_destroy(module->registry);
#endif
    free(module);
    return SYNURANG_OK;
}

static uint64_t vx_api_module_open(SynurangInstance* instance, const char* method,
                                  const SynurangCallOptions* options) {
    return instance ? synurang_call_open(((VxApiModule*)instance)->calls, method, options) : 0;
}
static int vx_api_module_send(SynurangInstance* instance, uint64_t call,
                              const uint8_t* data, uint32_t size) {
    if (!instance) return SYNURANG_INVALID_ARGUMENT;
    SynurangInstance* calls = ((VxApiModule*)instance)->calls;
    return calls ? synurang_call_send(calls, call, data, size) : SYNURANG_CLOSED;
}
static int vx_api_module_half_close(SynurangInstance* instance, uint64_t call) {
    if (!instance) return SYNURANG_INVALID_ARGUMENT;
    SynurangInstance* calls = ((VxApiModule*)instance)->calls;
    return calls ? synurang_call_half_close(calls, call) : SYNURANG_CLOSED;
}
static int vx_api_module_receive(SynurangInstance* instance, uint64_t call,
                                 SynurangReadResult* result) {
    if (!instance) return SYNURANG_INVALID_ARGUMENT;
    SynurangInstance* calls = ((VxApiModule*)instance)->calls;
    return calls ? synurang_call_receive(calls, call, result) : SYNURANG_CLOSED;
}
static int vx_api_module_cancel(SynurangInstance* instance, uint64_t call, int32_t code) {
    if (!instance) return SYNURANG_INVALID_ARGUMENT;
    SynurangInstance* calls = ((VxApiModule*)instance)->calls;
    return calls ? synurang_call_cancel(calls, call, code) : SYNURANG_CLOSED;
}
static void vx_api_module_release(SynurangInstance* instance, uint64_t call) {
    if (instance) synurang_call_release(((VxApiModule*)instance)->calls, call);
}
static uint32_t vx_api_module_poll(SynurangInstance* instance, uint32_t budget) {
    if (!instance) return 0;
    VxApiModule* module = (VxApiModule*)instance;
    if (!module->calls) return 0;
    if (!budget) budget = 64;
    /* Shutdown drains retained producer cancellation first. A wait whose
     * cancel callback is still queued must not dispatch new engine work. */
    if (module->closing) return synurang_instance_poll(module->calls, budget);
    uint32_t count;
    /* Generated callbacks and retained waits both make bounded progress.
     * Metadata traffic cannot consume every turn ahead of a ready wait. */
    if (budget == 1) {
        if (module->poll_pending_next) {
            count = vx_api_registry_poll(module->registry, 1);
            module->poll_pending_next = 0;
            if (!count) count = synurang_instance_poll(module->calls, 1);
        } else {
            count = synurang_instance_poll(module->calls, 1);
            module->poll_pending_next = 1;
            if (!count) {
                count = vx_api_registry_poll(module->registry, 1);
                module->poll_pending_next = 0;
            }
        }
    } else {
        count = synurang_instance_poll(module->calls, budget / 2 + budget % 2);
        count += vx_api_registry_poll(module->registry, budget - count);
        if (count < budget) count += synurang_instance_poll(module->calls, budget - count);
    }
    vx_api_registry_reap(module->registry, budget);
    return count;
}
static int vx_api_module_has_work(SynurangInstance* instance) {
    if (!instance) return 0;
    VxApiModule* module = (VxApiModule*)instance;
    if (!module->calls) {
#if !defined(VOLVOXAI_NO_THREADS)
        return atomic_load_explicit(&module->cleanup_done, memory_order_acquire);
#else
        return 0;
#endif
    }
    return synurang_instance_has_work(module->calls) || vx_api_registry_has_work(module->registry);
}

SYNURANG_C_RUNTIME_API const SynurangApi* Synurang_GetApi(void) {
    static const SynurangApi api = {
        SYNURANG_CALL_ABI_VERSION, sizeof(SynurangApi),
        vx_api_module_create, vx_api_module_destroy, vx_api_module_open,
        vx_api_module_send, vx_api_module_half_close, vx_api_module_receive,
        vx_api_module_cancel, vx_api_module_release, vx_api_module_poll,
        vx_api_module_has_work, synurang_response_free
    };
    return &api;
}
