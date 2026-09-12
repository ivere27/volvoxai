/* Compile the production implementation in a host-side threadless profile so
 * this regression can drive its private direct-call scope without adding a
 * test seam or symbol to a release artifact. */
#include "../src/runtime/public_api.c"

#include <stdio.h>

typedef struct DestroyState {
    unsigned calls;
    uint64_t identity;
} DestroyState;

static void record_runtime_destroy(uint64_t identity, void* context) {
    DestroyState* state = (DestroyState*)context;
    state->calls++;
    state->identity = identity;
}

static void record_request_change(void* context) {
    ++*(unsigned*)context;
}

static int latest_replacement_notifies_every_observer(void) {
    VxCompiledModel compiled = {0};
    VxRequest queued = {0};
    VxRequest incoming = {0};
    VxRuntimeCoordinator coordinator = {0};
    unsigned first = 0, second = 0, removed = 0;
    VxRequestWatch first_watch = { record_request_change, &first, NULL };
    VxRequestWatch second_watch = { record_request_change, &second, NULL };
    VxRequestWatch removed_watch = { record_request_change, &removed, NULL };
    queued.compiled = incoming.compiled = &compiled;
    queued.freshness = incoming.freshness = VX_REQUEST_FRESHNESS_LATEST;
    queued.stream_key = incoming.stream_key = 7;
    queued.state = VX_REQUEST_STATE_QUEUED;
    coordinator.head = coordinator.tail = &queued;

    vx_request_watch(&queued, &first_watch);
    vx_request_watch(&queued, &second_watch);
    vx_request_watch(&queued, &removed_watch);
    vx_request_unwatch(&queued, &removed_watch);
    if (first != 1 || second != 1 || removed != 1) return 1;

    /* Trigger the actual queued-LATEST transition. Do not poll the request:
     * its terminal transition must wake every retained WaitRequest observer. */
    VxRequest* replaced = vx_coordinator_supersede_latest_stream_locked(
        &coordinator, &incoming);
    int failed = replaced != &queued || coordinator.head || coordinator.tail ||
        queued.state != VX_REQUEST_STATE_SUPERSEDED ||
        first != 2 || second != 2 || removed != 1;
    vx_request_unwatch(&queued, &first_watch);
    vx_request_unwatch(&queued, &second_watch);
    if (queued.watches) failed = 1;
    vx_request_watch(&queued, &first_watch);
    if (first != 3) failed = 1; /* An already-terminal request also wakes. */
    vx_request_unwatch(&queued, &first_watch);
    return failed;
}

int main(void) {
    if (latest_replacement_notifies_every_observer()) {
        fputs("LATEST replacement did not notify retained request observers\n", stderr);
        return 1;
    }
    const uint64_t identity = UINT64_C(0x435333);
    DestroyState destroyed = {0};
    VxRuntimeDirectScope scope = {0};
    VxRuntime* runtime = (VxRuntime*)calloc(1, sizeof(*runtime));

    if (!runtime) {
        fputs("threadless runtime lifetime fixture allocation failed\n", stderr);
        return 1;
    }
    atomic_init(&runtime->references, 1);
    if (pthread_mutex_init(&runtime->mutex, NULL) != 0 ||
        pthread_mutex_init(&runtime->result_budget_mutex, NULL) != 0 ||
        pthread_cond_init(&runtime->direct_condition, NULL) != 0) {
        fputs("threadless runtime lifetime fixture initialization failed\n", stderr);
        free(runtime);
        return 1;
    }
    runtime->identity = identity;
    vx_runtime_set_destroy_callback(runtime, record_runtime_destroy, &destroyed);

    if (vx_runtime_direct_begin(runtime, &scope) != VX_STATUS_OK ||
        runtime->active_direct_calls != 1u ||
        atomic_load_explicit(&runtime->references, memory_order_acquire) != 2u ||
        g_runtime_direct_scope != &scope) {
        fputs("direct begin did not acquire one Runtime scope reference\n", stderr);
        vx_runtime_direct_end(&scope);
        vx_runtime_release(runtime);
        return 1;
    }

    /* Consume the caller's only reference while the direct scope is active.
     * The scope reference must defer final destruction until direct_end. */
    vx_runtime_release(runtime);
    if (destroyed.calls != 0u || runtime->active_direct_calls != 1u ||
        atomic_load_explicit(&runtime->references, memory_order_acquire) != 1u) {
        fputs("active direct scope did not defer final Runtime release\n", stderr);
        if (!destroyed.calls) vx_runtime_direct_end(&scope);
        return 1;
    }

    vx_runtime_direct_end(&scope);
    if (destroyed.calls != 1u || destroyed.identity != identity ||
        g_runtime_direct_scope != NULL) {
        fputs("direct end did not perform exactly one final Runtime release\n", stderr);
        return 1;
    }
    return 0;
}
