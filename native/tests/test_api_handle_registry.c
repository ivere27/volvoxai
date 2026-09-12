#include "vx_api_handles.h"

#include <pthread.h>
#include <stdio.h>

typedef struct TestObject {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int references;
    int acquired;
    int proceed;
    int destroyed;
    int sentinel;
    int64_t id;
    VxApiRegistry* registry;
} TestObject;

static void test_retain(void* pointer) {
    TestObject* object = (TestObject*)pointer;
    pthread_mutex_lock(&object->mutex);
    object->references++;
    pthread_mutex_unlock(&object->mutex);
}

static void test_release(void* pointer) {
    TestObject* object = (TestObject*)pointer;
    pthread_mutex_lock(&object->mutex);
    object->references--;
    if (object->references == 0) object->destroyed = 1;
    pthread_cond_broadcast(&object->condition);
    pthread_mutex_unlock(&object->mutex);
}

static void* operation_thread(void* pointer) {
    TestObject* object = (TestObject*)pointer;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;

    if (!vx_api_handle_acquire(object->registry, VX_API_HANDLE_RUNTIME, object->id, &lease)) {
        return (void*)1;
    }
    pthread_mutex_lock(&object->mutex);
    object->acquired = 1;
    pthread_cond_broadcast(&object->condition);
    while (!object->proceed) {
        pthread_cond_wait(&object->condition, &object->mutex);
    }
    pthread_mutex_unlock(&object->mutex);

    if (lease.pointer != object || object->sentinel != 0x51a5e) {
        vx_api_handle_lease_release(&lease);
        return (void*)1;
    }
    vx_api_handle_lease_release(&lease);
    return NULL;
}

static int fail(const char* message) {
    fprintf(stderr, "test_api_handle_registry: %s\n", message);
    return 1;
}

int main(void) {
    TestObject object = {0};
    VxApiHandleLease rejected = VX_API_HANDLE_LEASE_INIT;
    pthread_t worker;
    void* worker_result = NULL;
    int references;
    int destroyed;

    if (pthread_mutex_init(&object.mutex, NULL) != 0 ||
        pthread_cond_init(&object.condition, NULL) != 0) {
        return fail("synchronization initialization failed");
    }
    object.registry = vx_api_registry_create();
    if (!object.registry) return fail("registry allocation failed");
    VxApiRegistry* other = vx_api_registry_create();
    if (!other) return fail("second registry allocation failed");
    object.references = 1; /* the reference transferred by insert */
    object.acquired = 0;
    object.proceed = 0;
    object.destroyed = 0;
    object.sentinel = 0x51a5e;
    object.id = vx_api_handle_insert(object.registry, VX_API_HANDLE_RUNTIME, &object,
                                     test_retain, test_release);
    if (object.id <= 0) return fail("insert failed");
    if (vx_api_handle_acquire(other, VX_API_HANDLE_RUNTIME, object.id, &rejected))
        return fail("another registry accepted an owner-scoped id");
    if (vx_api_handle_remove(other, VX_API_HANDLE_RUNTIME, object.id))
        return fail("another registry retired an owner-scoped id");
    vx_api_registry_destroy(other);
    if (pthread_create(&worker, NULL, operation_thread, &object) != 0) {
        return fail("pthread_create failed");
    }

    pthread_mutex_lock(&object.mutex);
    while (!object.acquired) {
        pthread_cond_wait(&object.condition, &object.mutex);
    }
    pthread_mutex_unlock(&object.mutex);

    /* The worker's acquire retained under the registry mutex. Removing the
     * public id now drops only the registry reference, so the object must stay
     * alive until the accepted operation releases its lease. */
    if (!vx_api_handle_remove(object.registry, VX_API_HANDLE_RUNTIME, object.id)) {
        return fail("remove did not retire the live id");
    }
    if (vx_api_handle_acquire(object.registry, VX_API_HANDLE_RUNTIME, object.id, &rejected)) {
        vx_api_handle_lease_release(&rejected);
        return fail("retired id was acquired");
    }

    pthread_mutex_lock(&object.mutex);
    references = object.references;
    destroyed = object.destroyed;
    object.proceed = 1;
    pthread_cond_broadcast(&object.condition);
    pthread_mutex_unlock(&object.mutex);
    if (references != 1 || destroyed) {
        return fail("remove destroyed an object with an active operation lease");
    }

    if (pthread_join(worker, &worker_result) != 0 || worker_result != NULL) {
        return fail("operation thread failed");
    }
    pthread_mutex_lock(&object.mutex);
    references = object.references;
    destroyed = object.destroyed;
    pthread_mutex_unlock(&object.mutex);
    if (references != 0 || !destroyed) {
        return fail("final operation lease did not destroy the retired object");
    }
    if (vx_api_handle_remove(object.registry, VX_API_HANDLE_RUNTIME, object.id)) {
        return fail("second remove was not idempotent");
    }

    vx_api_registry_destroy(object.registry);
    pthread_cond_destroy(&object.condition);
    pthread_mutex_destroy(&object.mutex);
    puts("test_api_handle_registry: ok");
    return 0;
}
