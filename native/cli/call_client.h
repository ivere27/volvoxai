/* Blocking native consumer for command-line tools and integration fixtures.
 * Product embeddings may drive the same SynurangApi from their event loop.
 * This helper knows no VolvoxAI operation, message, or engine lifecycle. */
#ifndef VOLVOXAI_CLI_CALL_CLIENT_H
#define VOLVOXAI_CLI_CALL_CLIENT_H

#include <synurang/call.h>
#include "volvoxai_lite.h"
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>

extern const SynurangApi* Synurang_GetApi(void);

typedef struct VxCallClient {
    const SynurangApi* api;
    SynurangInstance* instance;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int notified;
    char last_error[512];
} VxCallClient;

static inline void vx_call_notify(void* context) {
    VxCallClient* client = context;
    pthread_mutex_lock(&client->mutex);
    client->notified = 1;
    pthread_cond_signal(&client->condition);
    pthread_mutex_unlock(&client->mutex);
}

static inline int vx_call_client_open_api(VxCallClient* client, const SynurangApi* api) {
    memset(client, 0, sizeof(*client));
    if (!api || api->abi_version != SYNURANG_CALL_ABI_VERSION ||
        api->struct_size != sizeof(*api)) return 0;
    if (pthread_mutex_init(&client->mutex, NULL) != 0) return 0;
    if (pthread_cond_init(&client->condition, NULL) != 0) {
        pthread_mutex_destroy(&client->mutex);
        return 0;
    }
    client->api = api;
    SynurangRuntimeOptions options = SYNURANG_RUNTIME_OPTIONS_INIT;
    options.execution_mode = SYNURANG_EXECUTION_MANUAL;
    options.wakeup = vx_call_notify;
    options.wakeup_user_data = client;
    client->instance = api->create(&options);
    if (!client->instance) {
        pthread_cond_destroy(&client->condition);
        pthread_mutex_destroy(&client->mutex);
        return 0;
    }
    return 1;
}

static inline int vx_call_client_open(VxCallClient* client) {
    return vx_call_client_open_api(client, Synurang_GetApi());
}

static inline void vx_call_turn(VxCallClient* client) {
    client->api->poll(client->instance, 64);
    if (client->api->has_work(client->instance)) return;
    pthread_mutex_lock(&client->mutex);
    while (!client->notified) pthread_cond_wait(&client->condition, &client->mutex);
    client->notified = 0;
    pthread_mutex_unlock(&client->mutex);
}

static inline void vx_call_client_close(VxCallClient* client) {
    if (!client->instance) return;
    while (client->api->destroy(client->instance) == SYNURANG_PENDING)
        vx_call_turn(client);
    client->instance = NULL;
    pthread_cond_destroy(&client->condition);
    pthread_mutex_destroy(&client->mutex);
}

static inline const uint8_t* vx_call_error(VxCallClient* client, int32_t* size) {
    if (size) *size = (int32_t)strlen(client->last_error);
    return (const uint8_t*)client->last_error;
}

static inline void vx_call_free(VxCallClient* client, void* data) {
    if (data) client->api->free_buffer(data);
}

/* Copying protobuf input into the module is synchronous; the returned response
 * belongs to the caller. Success requires one message followed by OK terminal
 * status. A response followed by an error never escapes as successful output. */
static inline uint8_t* vx_call_bytes(VxCallClient* client, const char* method,
                                    const uint8_t* data, size_t size, int32_t* out_size) {
    SynurangCallOptions options = { sizeof(options), 0, 0, 0, UINT64_MAX };
    uint8_t* response = NULL;
    int32_t response_size = 0;
    int status = SYNURANG_INVALID_ARGUMENT;
    if (out_size) *out_size = 0;
    client->last_error[0] = 0;
    if (!client->instance || size > UINT32_MAX) goto failed;
    uint64_t call = client->api->open(client->instance, method, &options);
    if (!call) goto failed;
    status = client->api->send(client->instance, call, data, (uint32_t)size);
    if (status != SYNURANG_OK) goto release;
    status = client->api->half_close(client->instance, call);
    if (status != SYNURANG_OK) goto release;
    for (;;) {
        SynurangReadResult read = {0};
        status = client->api->receive(client->instance, call, &read);
        if (status != SYNURANG_OK) { vx_call_free(client, read.data); break; }
        if (read.kind == SYNURANG_READ_MESSAGE) {
            if (response || read.size > INT32_MAX) {
                vx_call_free(client, read.data);
                status = SYNURANG_ERROR;
                break;
            }
            response = read.data;
            response_size = (int32_t)read.size;
        } else if (read.kind == SYNURANG_READ_FINISHED) {
            status = read.code == 0 && response ? SYNURANG_OK : SYNURANG_ERROR;
            vx_call_free(client, read.data);
            break;
        } else {
            vx_call_turn(client);
        }
    }
release:
    client->api->release(client->instance, call);
    if (status == SYNURANG_OK) {
        if (out_size) *out_size = response_size;
        return response;
    }
failed:
    vx_call_free(client, response);
    snprintf(client->last_error, sizeof(client->last_error), "%s: call failed (%d)", method, status);
    return NULL;
}

/* Construct requests with generated message types at each call site. This
 * macro only pairs their codec allocation with the byte transport. */
#define VX_CALL_MESSAGE(client, path, codec, request, output, length) do { \
    VxCallClient* vx_call_client_ = (client); \
    const SynurangLiteAllocator* vx_call_allocator_ = (request)->_allocator; \
    uint8_t* vx_call_encoded_ = NULL; \
    size_t vx_call_encoded_size_ = 0; \
    (output) = NULL; \
    (length) = 0; \
    if (codec##_encode((request), &vx_call_encoded_, &vx_call_encoded_size_) == \
        SYNURANG_LITE_OK) { \
        (output) = vx_call_bytes(vx_call_client_, (path), vx_call_encoded_, \
                                 vx_call_encoded_size_, &(length)); \
    } else { \
        snprintf(vx_call_client_->last_error, sizeof(vx_call_client_->last_error), \
                 "%s: request encoding failed", (path)); \
    } \
    if (vx_call_encoded_) vx_call_allocator_->deallocate( \
        vx_call_allocator_->context, vx_call_encoded_); \
} while (0)

#endif
