/* clock_gettime and CLOCK_MONOTONIC are POSIX, not ISO C.  The tests build
 * with -std=c11, which defines __STRICT_ANSI__ and hides them; the declaration
 * must be asked for explicitly rather than arriving by accident through
 * whatever -pthread happens to imply on the build host. */
#define _POSIX_C_SOURCE 199309L

/* Engine-independent batch policy core and native behavioural twin of the
 * TypeScript contribution scheduler. The Runtime-owned production coordinator
 * is a separate execution path. */
#include "batch_scheduler.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VX_BATCH_DEFAULT_QUEUE_DEPTH_WINDOW 1024

typedef struct {
    int id;
    VxBatchRequestKind kind;
    char* model_id;
    char* adapter_revision;
    char* shape_signature_minus_batch;
    int rows_per_lane;
    int prompt_tokens;
    int max_tokens;
    void* payload;
    char* prompt_key;
    int has_prompt_key;
    int shared_tokens;
    int prefilled_tokens;
    int submitted_round;
    int64_t submitted_time_micros;
    VxBatchRequestState state;
    int slot;
    int slot_generation;
    int generated;
    int cancel_requested;
    void* value;
    /* Prefill owns its KV transition from admission through callback
     * settlement.  The published lane length remains unchanged meanwhile. */
    VxPagedKVReservation pending_reservation;
} VxBatchRequest;

/* One round's view of a group key's ready contributions. */
typedef struct {
    VxGroupKey group_key;
    VxBatchRequestKind kind;
    int oldest_id;
    VxBatchRequest** members;
    int count;
    int capacity;
} VxBatchGroup;

typedef struct {
    VxGroupKey key;
    VxBatchRequestKind kind;
} VxBatchGroupIdentity;

typedef struct VxBatchPendingDispatch {
    uint64_t id;
    VxBatchStepWork* works;
    VxBatchStepOutcome* outcomes;
    VxPagedKVReservation* reservations;
    int* member_ids;
    int reservation_count;
    VxBatchDispatchMetadata metadata;
    int64_t started_micros;
    struct VxBatchPendingDispatch* next;
} VxBatchPendingDispatch;

struct VxBatchScheduler {
    VxPagedKVCache* cache;
    int slots;
    int max_queue_depth;
    int token_budget_per_dispatch;
    int max_lanes;
    VxBatchDispatchPolicy policy;
    int multiple_of;

    int next_request_id;
    int round;
    int closed;
    int draining;
    int in_round;
    int deferred;
    uint64_t next_dispatch_id;
    VxBatchPendingDispatch* pending;
    VxBatchPendingDispatch* pending_tail;

    /* Individually allocated active records live in a fixed-capacity pointer
     * pool. A submission never relocates a record held by an in-flight round;
     * terminal records are compacted at safe round boundaries. Lookup is a
     * scan over this bounded pool, never over lifetime request history. */
    VxBatchRequest** requests;
    int request_count;
    int request_capacity;
    int max_request_records_seen;

    int* queue;
    int queue_count;

    int* occupants;

    VxBatchResult* results;
    int result_count;
    int result_capacity;
    int result_start;

    /* Fixed-capacity ring: percentiles over a recent window, not over a list
     * that grows once per round for the life of the process. */
    int* queue_depth_samples;
    int queue_depth_window;
    int queue_depth_sample_count;
    int queue_depth_cursor;

    /* Telemetry counters */
    int dispatches;
    int rows_dispatched;
    int rows_useful;
    int64_t device_busy_micros;
    /* Wall is the span the scheduler was actually driven over: from the start
     * of the first step to the end of the last. Every dispatch happens inside
     * it, so device_busy <= wall holds by construction rather than by clamp. */
    int64_t session_start_micros;
    int64_t last_activity_micros;
    int steps;
    int admitted;
    int completed;
    int cancelled;
    int failed;
    int admission_stalls;
    int max_queue_depth_seen;
    int queue_delay_rounds;
    int prefix_reuses;
    int prefix_publications;
    int prefill_tokens;
    int shared_prompt_tokens;
};

/* Scheduling and telemetry clock: real monotonic microseconds. */
static int64_t monotonic_micros(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000LL + (int64_t)ts.tv_nsec / 1000LL;
}

static int rollback_pending_reservation(VxBatchScheduler* scheduler,
                                        VxBatchRequest* request);

static VxBatchRequest* request_find(const VxBatchScheduler* scheduler, int id) {
    if (!scheduler || id <= 0) return NULL;
    for (int index = 0; index < scheduler->request_count; index++)
        if (scheduler->requests[index]->id == id)
            return scheduler->requests[index];
    return NULL;
}

static VxBatchRequest* request_append(VxBatchScheduler* scheduler) {
    VxBatchRequest* request;
    if (scheduler->request_count >= scheduler->request_capacity) return NULL;
    request = (VxBatchRequest*)calloc(1, sizeof(*request));
    if (!request) return NULL;
    scheduler->requests[scheduler->request_count++] = request;
    if (scheduler->request_count > scheduler->max_request_records_seen)
        scheduler->max_request_records_seen = scheduler->request_count;
    return request;
}

static void result_append(VxBatchScheduler* scheduler,
                          const VxBatchResult* result) {
    int index;
    if (scheduler->result_count < scheduler->result_capacity) {
        index = (scheduler->result_start + scheduler->result_count) %
            scheduler->result_capacity;
        scheduler->result_count++;
    } else {
        index = scheduler->result_start;
        scheduler->result_start = (scheduler->result_start + 1) %
            scheduler->result_capacity;
    }
    scheduler->results[index] = *result;
}

static void batch_request_free(VxBatchRequest* request) {
    if (!request) return;
    free(request->model_id); free(request->adapter_revision);
    free(request->shape_signature_minus_batch); free(request->prompt_key);
    free(request);
}

static void retire_terminal_requests(VxBatchScheduler* scheduler) {
    int write = 0;
    for (int read = 0; read < scheduler->request_count; read++) {
        VxBatchRequest* request = scheduler->requests[read];
        if (request->state == VX_BATCH_REQUEST_COMPLETED ||
            request->state == VX_BATCH_REQUEST_CANCELLED ||
            request->state == VX_BATCH_REQUEST_FAILED) {
            (void)rollback_pending_reservation(scheduler, request);
            batch_request_free(request);
            continue;
        }
        scheduler->requests[write++] = request;
    }
    scheduler->request_count = write;
}

static void queue_remove_at(VxBatchScheduler* scheduler, int index) {
    for (int scan = index; scan + 1 < scheduler->queue_count; scan++) {
        scheduler->queue[scan] = scheduler->queue[scan + 1];
    }
    scheduler->queue_count--;
}

static void queue_remove_id(VxBatchScheduler* scheduler, int request_id) {
    for (int index = 0; index < scheduler->queue_count; index++) {
        if (scheduler->queue[index] == request_id) {
            queue_remove_at(scheduler, index);
            return;
        }
    }
}

static int rollback_pending_reservation(VxBatchScheduler* scheduler,
                                        VxBatchRequest* request) {
    VxPagedKVStatus status;
    if (!scheduler || !request || !request->pending_reservation.open)
        return 0;
    status = vx_paged_kv_rollback(scheduler->cache,
                                  &request->pending_reservation);
    vx_paged_kv_reservation_dispose(&request->pending_reservation);
    return status == VX_PAGED_KV_OK ? 0 : -1;
}

static void release_slot(VxBatchScheduler* scheduler,
                         VxBatchRequest* request) {
    if (request->slot == VX_BATCH_NO_SLOT || !scheduler->cache) return;
    if (rollback_pending_reservation(scheduler, request) != 0) return;
    if (vx_paged_kv_release_lane(scheduler->cache, request->slot) !=
        VX_PAGED_KV_OK) return;
    scheduler->occupants[request->slot] = -1;
    request->slot = VX_BATCH_NO_SLOT;
}

static void publish(VxBatchScheduler* scheduler, VxBatchRequest* request,
                    VxBatchRequestState state) {
    VxBatchResult result;
    if (request->state == VX_BATCH_REQUEST_COMPLETED ||
        request->state == VX_BATCH_REQUEST_CANCELLED ||
        request->state == VX_BATCH_REQUEST_FAILED) return;
    request->state = state;
    if (state == VX_BATCH_REQUEST_COMPLETED) scheduler->completed++;
    else if (state == VX_BATCH_REQUEST_CANCELLED) scheduler->cancelled++;
    else scheduler->failed++;
    result.request_id = request->id;
    result.kind = request->kind;
    result.state = state;
    result.generated = request->generated;
    result.status = (state == VX_BATCH_REQUEST_FAILED) ? VX_BATCH_STEP_FAILED : VX_BATCH_OK;
    result.value = state == VX_BATCH_REQUEST_COMPLETED ? request->value : NULL;
    result_append(scheduler, &result);
}

static char* batch_copy_string(const char* value) {
    if (!value) return NULL;
    size_t bytes = strlen(value) + 1;
    char* copy = malloc(bytes);
    if (copy) memcpy(copy, value, bytes);
    return copy;
}

static void group_key_of(VxGroupKey* out, const VxBatchRequest* request, int rows_per_lane) {
    *out = (VxGroupKey){request->model_id, request->adapter_revision,
        request->shape_signature_minus_batch, rows_per_lane};
}

static int batch_string_equal(const char* a, const char* b) {
    return a && b ? strcmp(a, b) == 0 : a == b;
}

static int group_key_equal(const VxGroupKey* left, const VxGroupKey* right) {
    return left->rows_per_lane == right->rows_per_lane &&
        batch_string_equal(left->model_id, right->model_id) &&
        batch_string_equal(left->adapter_revision, right->adapter_revision) &&
        batch_string_equal(left->shape_signature_minus_batch, right->shape_signature_minus_batch);
}

static int batch_request_identity(VxBatchRequest* request, const char* model,
        const char* adapter, const char* signature, const char* prompt) {
    request->model_id = batch_copy_string(model);
    request->adapter_revision = batch_copy_string(adapter);
    request->shape_signature_minus_batch = batch_copy_string(signature);
    request->prompt_key = batch_copy_string(prompt);
    request->has_prompt_key = prompt && *prompt;
    return request->model_id && request->shape_signature_minus_batch &&
        (!adapter || request->adapter_revision) && (!prompt || request->prompt_key);
}

static int padded_dispatch_count(const VxBatchScheduler* scheduler, int count,
                                 VxBatchRequestKind kind, int* padded_out) {
    int padded = count;
    if (!scheduler || !padded_out || count <= 0) return -1;
    if (kind == VX_BATCH_KIND_STATELESS && scheduler->multiple_of > 1) {
        int remainder = count % scheduler->multiple_of;
        if (remainder != 0) {
            int extra = scheduler->multiple_of - remainder;
            if (count > INT_MAX - extra) return -1;
            padded = count + extra;
        }
    }
    if (padded > scheduler->max_lanes) return -1;
    *padded_out = padded;
    return 0;
}

VxBatchScheduler* vx_batch_scheduler_create(const VxBatchSchedulerOptions* options) {
    VxBatchScheduler* scheduler;
    int max_retained_results;
    scheduler = (VxBatchScheduler*)calloc(1, sizeof(*scheduler));
    if (!scheduler) return NULL;

    if (options && options->cache) {
        scheduler->cache = options->cache;
        scheduler->slots = vx_paged_kv_lanes(options->cache);
    } else {
        scheduler->cache = NULL;
        scheduler->slots = (options && options->max_lanes > 0) ? options->max_lanes : 8;
    }

    if (options && options->max_queue_depth > 0) {
        scheduler->max_queue_depth = options->max_queue_depth;
    } else {
        if (scheduler->slots > INT_MAX / 4) {
            free(scheduler);
            return NULL;
        }
        scheduler->max_queue_depth = scheduler->slots * 4;
    }
    scheduler->token_budget_per_dispatch = (options && options->token_budget_per_dispatch > 0)
        ? options->token_budget_per_dispatch : 2048;
    scheduler->max_lanes = (options && options->max_lanes > 0)
        ? options->max_lanes : scheduler->slots;
    scheduler->multiple_of = (options && options->multiple_of > 0)
        ? options->multiple_of : 1;
    scheduler->policy = options ? options->policy : (VxBatchDispatchPolicy){ VX_DISPATCH_WORK_CONSERVING, 0 };
    scheduler->queue_depth_window = (options && options->queue_depth_window > 0)
        ? options->queue_depth_window : VX_BATCH_DEFAULT_QUEUE_DEPTH_WINDOW;
    max_retained_results = (options && options->max_retained_results > 0)
        ? options->max_retained_results : 0;
    scheduler->session_start_micros = -1;
    scheduler->last_activity_micros = -1;

    if (scheduler->max_lanes <= 0 ||
        scheduler->multiple_of > scheduler->max_lanes ||
        (options && options->max_retained_results < 0) ||
        scheduler->max_queue_depth > (INT_MAX - scheduler->slots) / 2) {
        free(scheduler);
        return NULL;
    }
    if (!max_retained_results)
        max_retained_results = scheduler->max_queue_depth * 2 + scheduler->slots;

    scheduler->next_request_id = 1;
    scheduler->next_dispatch_id = 1;
    scheduler->request_capacity =
        scheduler->max_queue_depth * 2 + scheduler->slots;
    scheduler->result_capacity = max_retained_results;
    scheduler->requests = (VxBatchRequest**)calloc(
        (size_t)scheduler->request_capacity, sizeof(*scheduler->requests));
    scheduler->queue = (int*)malloc((size_t)scheduler->max_queue_depth * sizeof(int));
    scheduler->occupants = (int*)malloc((size_t)scheduler->slots * sizeof(int));
    scheduler->results = (VxBatchResult*)calloc(
        (size_t)scheduler->result_capacity, sizeof(*scheduler->results));
    scheduler->queue_depth_samples =
        (int*)malloc((size_t)scheduler->queue_depth_window * sizeof(int));
    if (scheduler->occupants)
        for (int slot = 0; slot < scheduler->slots; ++slot)
            scheduler->occupants[slot] = -1;
    if (!scheduler->requests || !scheduler->queue || !scheduler->occupants ||
        !scheduler->results || !scheduler->queue_depth_samples) {
        vx_batch_scheduler_destroy(scheduler);
        return NULL;
    }
    for (int slot = 0; slot < scheduler->slots; slot++) {
        scheduler->occupants[slot] = -1;
    }
    return scheduler;
}

void vx_batch_scheduler_destroy(VxBatchScheduler* scheduler) {
    if (!scheduler) return;
    (void)vx_batch_scheduler_close_async(scheduler, 0);
    if (scheduler->cache) {
        for (int index = 0; index < scheduler->request_count; index++)
            (void)rollback_pending_reservation(
                scheduler, scheduler->requests[index]);
    }
    if (scheduler->cache && scheduler->occupants) {
        for (int slot = 0; slot < scheduler->slots; slot++) {
            if (scheduler->occupants[slot] != -1) {
                vx_paged_kv_release_lane(scheduler->cache, slot);
            }
        }
    }
    for (int index = 0; index < scheduler->request_count; index++) {
        batch_request_free(scheduler->requests[index]);
    }
    free(scheduler->requests);
    free(scheduler->queue);
    free(scheduler->occupants);
    free(scheduler->results);
    free(scheduler->queue_depth_samples);
    free(scheduler);
}

VxBatchStatus vx_batch_scheduler_submit_stateless(
    VxBatchScheduler* scheduler,
    const char* model_id,
    const char* adapter_revision,
    const char* shape_signature_minus_batch,
    int rows,
    void* payload,
    int* id_out) {
    VxBatchRequest* request;
    int minimum_count;
    const char* resolved_model = model_id ? model_id : VX_BATCH_DEFAULT_MODEL;
    const char* resolved_signature = shape_signature_minus_batch
        ? shape_signature_minus_batch : "stateless";
    if (!scheduler || rows <= 0 ||
        padded_dispatch_count(scheduler, 1, VX_BATCH_KIND_STATELESS,
                              &minimum_count) != 0 ||
        rows > scheduler->token_budget_per_dispatch / minimum_count) {
        return VX_BATCH_INVALID_ARGUMENT;
    }
    if (scheduler->closed) return VX_BATCH_SCHEDULER_CLOSED;
    if (scheduler->queue_count >= scheduler->max_queue_depth) {
        return VX_BATCH_QUEUE_FULL;
    }
    if (scheduler->request_count >= scheduler->request_capacity)
        return VX_BATCH_QUEUE_FULL;
    if (scheduler->next_request_id <= 0) return VX_BATCH_INVALID_ARGUMENT;
    request = request_append(scheduler);
    if (!request) return VX_BATCH_INVALID_ARGUMENT;

    request->id = scheduler->next_request_id;
    scheduler->next_request_id = request->id == INT_MAX
        ? 0 : request->id + 1;
    request->kind = VX_BATCH_KIND_STATELESS;
    request->rows_per_lane = rows;
    request->payload = payload;
    request->submitted_round = scheduler->round;
    request->submitted_time_micros = monotonic_micros();
    request->state = VX_BATCH_REQUEST_QUEUED;
    request->slot = VX_BATCH_NO_SLOT;
    request->slot_generation = 0;
    request->generated = 0;

    if (!batch_request_identity(request, resolved_model, adapter_revision,
                                resolved_signature, NULL)) {
        scheduler->request_count--;
        batch_request_free(request);
        return VX_BATCH_INVALID_ARGUMENT;
    }

    scheduler->queue[scheduler->queue_count++] = request->id;
    if (scheduler->queue_count > scheduler->max_queue_depth_seen) {
        scheduler->max_queue_depth_seen = scheduler->queue_count;
    }
    if (id_out) *id_out = request->id;
    return VX_BATCH_OK;
}

VxBatchStatus vx_batch_scheduler_submit_llm(
    VxBatchScheduler* scheduler,
    const char* model_id,
    const char* adapter_revision,
    const char* shape_signature_minus_batch,
    int prompt_tokens,
    int max_tokens,
    const char* prompt_key,
    void* payload,
    int* id_out) {
    VxBatchRequest* request;
    int lane_capacity;
    const char* resolved_model = model_id ? model_id : VX_BATCH_DEFAULT_MODEL;
    const char* resolved_signature = shape_signature_minus_batch
        ? shape_signature_minus_batch : "llm";
    if (!scheduler || !scheduler->cache || prompt_tokens <= 0 || max_tokens <= 0) {
        return VX_BATCH_INVALID_ARGUMENT;
    }
    if (scheduler->closed) return VX_BATCH_SCHEDULER_CLOSED;
    lane_capacity = vx_paged_kv_lane_token_capacity(scheduler->cache);
    if (prompt_tokens > lane_capacity || max_tokens > lane_capacity - prompt_tokens) {
        return VX_BATCH_INVALID_ARGUMENT;
    }
    if (scheduler->queue_count >= scheduler->max_queue_depth) {
        return VX_BATCH_QUEUE_FULL;
    }
    if (scheduler->request_count >= scheduler->request_capacity)
        return VX_BATCH_QUEUE_FULL;
    if (scheduler->next_request_id <= 0) return VX_BATCH_INVALID_ARGUMENT;

    request = request_append(scheduler);
    if (!request) return VX_BATCH_INVALID_ARGUMENT;

    request->id = scheduler->next_request_id;
    scheduler->next_request_id = request->id == INT_MAX
        ? 0 : request->id + 1;
    request->kind = VX_BATCH_KIND_DECODE;
    request->rows_per_lane = 1;
    request->prompt_tokens = prompt_tokens;
    request->max_tokens = max_tokens;
    request->payload = payload;
    request->submitted_round = scheduler->round;
    request->submitted_time_micros = monotonic_micros();
    request->state = VX_BATCH_REQUEST_QUEUED;
    request->slot = VX_BATCH_NO_SLOT;
    request->slot_generation = 0;
    request->generated = 0;

    if (!batch_request_identity(request, resolved_model, adapter_revision,
                                resolved_signature, prompt_key)) {
        scheduler->request_count--;
        batch_request_free(request);
        return VX_BATCH_INVALID_ARGUMENT;
    }

    scheduler->queue[scheduler->queue_count++] = request->id;
    if (scheduler->queue_count > scheduler->max_queue_depth_seen) {
        scheduler->max_queue_depth_seen = scheduler->queue_count;
    }
    if (id_out) *id_out = request->id;
    return VX_BATCH_OK;
}

int vx_batch_scheduler_cancel(VxBatchScheduler* scheduler, int request_id) {
    VxBatchRequest* request;
    if (!scheduler) return 0;
    request = request_find(scheduler, request_id);
    if (!request) return 0;
    if (request->state == VX_BATCH_REQUEST_COMPLETED ||
        request->state == VX_BATCH_REQUEST_CANCELLED ||
        request->state == VX_BATCH_REQUEST_FAILED) return 0;

    if (request->state == VX_BATCH_REQUEST_QUEUED) {
        queue_remove_id(scheduler, request_id);
        publish(scheduler, request, VX_BATCH_REQUEST_CANCELLED);
        if (!scheduler->in_round) retire_terminal_requests(scheduler);
        return 1;
    }
    /* Between rounds the slot holds no submitted work, so retire it now.
     * Inside a round the step may already be in flight; deferring to the end
     * of the round is what keeps the lane from being torn down underneath it. */
    if (!scheduler->in_round) {
        release_slot(scheduler, request);
        publish(scheduler, request, VX_BATCH_REQUEST_CANCELLED);
        retire_terminal_requests(scheduler);
        return 1;
    }
    request->cancel_requested = 1;
    return 1;
}

/*
 * Bind a resident prefix for this request's prompt, returning its length.
 *
 * Zero when the request declared no identity, when nothing is resident under
 * it, or when the prefix is longer than this prompt -- a prefix that runs past
 * the prompt is a different prompt, whatever its key says.
 */
static int admit_shared_prefix(VxBatchScheduler* scheduler, int slot,
                               VxBatchRequest* request) {
    int tokens = 0;
    if (!request->has_prompt_key ||
        !vx_paged_kv_has_prefix(scheduler->cache, request->prompt_key)) {
        return 0;
    }
    if (vx_paged_kv_acquire_prefix(scheduler->cache, request->prompt_key, slot, &tokens) != VX_PAGED_KV_OK) {
        return 0;
    }
    if (tokens <= request->prompt_tokens) return tokens;
    vx_paged_kv_release_lane(scheduler->cache, slot);
    return 0;
}

/*
 * Publish this request's prompt so later requests can bind it.
 *
 * Only the whole pages of the prompt: a prefix must end on a page boundary,
 * because a partial tail page is still being appended to.
 */
static void publish_prompt_prefix(VxBatchScheduler* scheduler, VxBatchRequest* request) {
    int page_tokens, publishable;
    if (!scheduler->cache || !request->has_prompt_key ||
        vx_paged_kv_has_prefix(scheduler->cache, request->prompt_key)) return;
    page_tokens = vx_paged_kv_page_tokens(scheduler->cache);
    publishable = (request->prompt_tokens / page_tokens) * page_tokens;
    if (publishable <= 0) return;
    if (vx_paged_kv_publish_prefix(scheduler->cache, request->prompt_key,
                                    request->slot, publishable) == VX_PAGED_KV_OK) {
        scheduler->prefix_publications++;
    }
}

/*
 * Fill free slots from the head of the queue.
 *
 * Strictly head-first, so a request cannot be passed over by a later one that
 * happens to fit -- bounded starvation is a contract, not an accident.  When
 * the head cannot reserve its prompt, eviction is tried once and then
 * admission stops for the round.
 */
static int admit_stateful(VxBatchScheduler* scheduler, VxBatchRequest** admitted_out) {
    int admitted_count = 0;
    if (!scheduler->cache) return 0;

    for (int slot = 0; slot < scheduler->slots; slot++) {
        int queue_idx = -1;
        VxBatchRequest* request = NULL;
        int shared, remaining, pages_needed;
        VxPagedKVReservation reservation = {0};
        VxPagedKVStatus status;

        if (scheduler->occupants[slot] != -1) continue;
        for (int q = 0; q < scheduler->queue_count; q++) {
            VxBatchRequest* cand = request_find(scheduler, scheduler->queue[q]);
            if (cand && cand->kind == VX_BATCH_KIND_DECODE && cand->state == VX_BATCH_REQUEST_QUEUED) {
                queue_idx = q;
                request = cand;
                break;
            }
        }
        if (!request) break;

        shared = admit_shared_prefix(scheduler, slot, request);
        remaining = request->prompt_tokens - shared;
        if (remaining > scheduler->token_budget_per_dispatch)
            remaining = scheduler->token_budget_per_dispatch;
        int page_tokens = vx_paged_kv_page_tokens(scheduler->cache);
        pages_needed = remaining / page_tokens + (remaining % page_tokens != 0);

        status = remaining > 0
            ? vx_paged_kv_reserve(scheduler->cache, slot, remaining,
                                  &reservation)
            : VX_PAGED_KV_OK;
        if (status != VX_PAGED_KV_OK) {
            vx_paged_kv_evict(scheduler->cache, pages_needed);
            status = vx_paged_kv_reserve(scheduler->cache, slot, remaining,
                                         &reservation);
            if (status != VX_PAGED_KV_OK) {
                if (shared > 0) vx_paged_kv_release_lane(scheduler->cache, slot);
                scheduler->admission_stalls++;
                break;
            }
        }
        request->pending_reservation = reservation;
        memset(&reservation, 0, sizeof(reservation));
        request->shared_tokens = shared;
        request->prefilled_tokens = shared;
        scheduler->shared_prompt_tokens += shared;
        if (shared > 0) scheduler->prefix_reuses++;

        queue_remove_at(scheduler, queue_idx);
        request->state = VX_BATCH_REQUEST_PREFILL;
        request->slot = slot;
        request->slot_generation = vx_paged_kv_lane_generations(scheduler->cache)[slot];
        scheduler->occupants[slot] = request->id;
        scheduler->admitted++;
        scheduler->queue_delay_rounds += scheduler->round - request->submitted_round;

        admitted_out[admitted_count++] = request;
    }
    return admitted_count;
}

static void fail_selected_batch(VxBatchScheduler* scheduler,
                                VxBatchRequest** requests,
                                int count,
                                VxBatchRequestKind kind) {
    for (int index = 0; index < count; index++) {
        VxBatchRequest* request = requests[index];
        if (!request) continue;
        if (kind == VX_BATCH_KIND_STATELESS)
            queue_remove_id(scheduler, request->id);
        else
            release_slot(scheduler, request);
        publish(scheduler, request, VX_BATCH_REQUEST_FAILED);
    }
}

static void rollback_local_reservations(VxBatchScheduler* scheduler,
                                        VxPagedKVReservation* reservations,
                                        int count) {
    if (!scheduler->cache || !reservations || count <= 0) return;
    (void)vx_paged_kv_rollback_batch(scheduler->cache, reservations,
                                     (size_t)count);
    for (int index = 0; index < count; index++)
        vx_paged_kv_reservation_dispose(&reservations[index]);
}

static VxBatchStatus batch_settle_dispatch(VxBatchScheduler* scheduler,
        VxBatchPendingDispatch* pending, VxBatchStatus status) {
    VxBatchStepWork* works = pending->works;
    VxBatchStepOutcome* outcomes = pending->outcomes;
    VxPagedKVReservation* reservations = pending->reservations;
    int* member_ids = pending->member_ids;
    int reservation_count = pending->reservation_count;
    int useful_count = pending->metadata.useful_count;
    int useful_rows = pending->metadata.useful_rows;
    VxBatchRequestKind kind = pending->metadata.kind;
    int batch_failed = 0;
    if (pending->started_micros >= 0) {
        scheduler->last_activity_micros = monotonic_micros();
        int64_t elapsed = scheduler->last_activity_micros - pending->started_micros;
        scheduler->device_busy_micros += elapsed > 0 ? elapsed : 0;
        scheduler->dispatches++;
        scheduler->rows_dispatched += pending->metadata.total_rows;
    }
    if (status != VX_BATCH_OK) {
        batch_failed = 1;
    } else {
        for (int index = 0; index < useful_count; index++) {
            VxBatchRequest* request =
                request_find(scheduler, member_ids[index]);
            if (!request || outcomes[index].status != VX_BATCH_OK) {
                batch_failed = 1;
                break;
            }
            if (kind != VX_BATCH_KIND_STATELESS &&
                (request->slot == VX_BATCH_NO_SLOT ||
                 vx_paged_kv_lane_generations(scheduler->cache)[
                     request->slot] != request->slot_generation ||
                 reservations[index].lane != request->slot ||
                 reservations[index].lane_generation !=
                     request->slot_generation)) {
                batch_failed = 1;
                break;
            }
        }
    }

    if (!batch_failed && kind != VX_BATCH_KIND_STATELESS &&
        vx_paged_kv_commit_batch(scheduler->cache, reservations,
                                 (size_t)reservation_count) !=
            VX_PAGED_KV_OK) {
        batch_failed = 1;
    }

    if (batch_failed) {
        rollback_local_reservations(scheduler, reservations,
                                    reservation_count);
        for (int i = 0; i < useful_count; ++i) {
            VxBatchRequest* request = request_find(scheduler, member_ids[i]);
            if (!request) continue;
            release_slot(scheduler, request);
            publish(scheduler, request, request->cancel_requested ?
                VX_BATCH_REQUEST_CANCELLED : VX_BATCH_REQUEST_FAILED);
        }
        free(works);
        free(outcomes);
        free(member_ids);
        free(reservations);
        return status == VX_BATCH_OK ? VX_BATCH_OK : status;
    }

    scheduler->steps += useful_count;
    scheduler->rows_useful += useful_rows;
    for (int index = 0; index < useful_count; index++) {
        VxBatchRequest* request =
            request_find(scheduler, member_ids[index]);
        VxBatchStepOutcome* outcome = &outcomes[index];
        if (request->cancel_requested || request->state == VX_BATCH_REQUEST_CANCELLED) {
            release_slot(scheduler, request);
            publish(scheduler, request, VX_BATCH_REQUEST_CANCELLED);
            continue;
        }
        request->value = outcome->value;
        if (kind == VX_BATCH_KIND_STATELESS) {
            publish(scheduler, request, VX_BATCH_REQUEST_COMPLETED);
        } else if (kind == VX_BATCH_KIND_PREFILL) {
            scheduler->prefill_tokens += works[index].tokens;
            request->prefilled_tokens += works[index].tokens;
            if (request->prefilled_tokens == request->prompt_tokens) {
                publish_prompt_prefix(scheduler, request);
                request->state = VX_BATCH_REQUEST_DECODING;
            } else request->state = VX_BATCH_REQUEST_PREFILL;
        } else {
            request->generated++;
            request->state = VX_BATCH_REQUEST_DECODING;
            if (outcome->finished || request->generated >= request->max_tokens) {
                release_slot(scheduler, request);
                publish(scheduler, request, VX_BATCH_REQUEST_COMPLETED);
            }
        }
    }

    free(works);
    free(outcomes);
    free(member_ids);
    free(reservations);
    return VX_BATCH_OK;
}

static VxBatchStatus dispatch_batch(
    VxBatchScheduler* scheduler,
    VxBatchRequest** requests,
    int count,
    VxBatchRequestKind kind,
    const VxGroupKey* group_key,
    VxBatchRunStep run_step,
    void* user) {
    VxBatchStepWork* works = NULL;
    VxBatchStepOutcome* outcomes = NULL;
    VxPagedKVReservation* reservations = NULL;
    VxBatchDispatchMetadata metadata;
    int* member_ids = NULL;
    int useful_count = 0;
    int reservation_count = 0;
    int padded_count, row_count;
    int total_rows, useful_rows;
    int batch_failed = 0;
    VxBatchStatus status = VX_BATCH_OK;

    if (count <= 0) return VX_BATCH_OK;

    /* Only the bulk [B,...] path can carry a duplicated padding row. */
    if (!group_key || group_key->rows_per_lane <= 0 ||
        padded_dispatch_count(scheduler, count, kind, &padded_count) != 0 ||
        padded_count > scheduler->token_budget_per_dispatch /
                           group_key->rows_per_lane) {
        fail_selected_batch(scheduler, requests, count, kind);
        return VX_BATCH_INVALID_ARGUMENT;
    }

    works = (VxBatchStepWork*)calloc(
        (size_t)padded_count, sizeof(*works));
    outcomes = (VxBatchStepOutcome*)calloc(
        (size_t)padded_count, sizeof(*outcomes));
    member_ids = (int*)calloc((size_t)count, sizeof(*member_ids));
    if (kind != VX_BATCH_KIND_STATELESS)
        reservations = (VxPagedKVReservation*)calloc(
            (size_t)count, sizeof(*reservations));
    if (!works || !outcomes || !member_ids ||
        (kind != VX_BATCH_KIND_STATELESS && !reservations)) {
        fail_selected_batch(scheduler, requests, count, kind);
        free(works);
        free(outcomes);
        free(member_ids);
        free(reservations);
        return VX_BATCH_INVALID_ARGUMENT;
    }

    for (int index = 0; index < count; index++) {
        VxBatchRequest* request = requests[index];
        VxBatchStepWork* work = &works[useful_count];
        int tokens = kind == VX_BATCH_KIND_DECODE ? 1 : group_key->rows_per_lane;
        if (kind == VX_BATCH_KIND_STATELESS)
            tokens = request->rows_per_lane;

        if (kind == VX_BATCH_KIND_DECODE) {
            if (vx_paged_kv_reserve(scheduler->cache, request->slot, tokens,
                                    &reservations[reservation_count]) !=
                VX_PAGED_KV_OK) {
                batch_failed = 1;
                break;
            }
            reservation_count++;
        } else if (kind == VX_BATCH_KIND_PREFILL) {
            if (request->pending_reservation.open) {
                reservations[reservation_count++] = request->pending_reservation;
                memset(&request->pending_reservation, 0, sizeof(request->pending_reservation));
            } else if (vx_paged_kv_reserve(scheduler->cache, request->slot, tokens,
                       &reservations[reservation_count]) == VX_PAGED_KV_OK) reservation_count++;
            else { batch_failed = 1; break; }
        }

        work->request_id = request->id;
        work->kind = kind;
        work->rows = group_key->rows_per_lane;
        work->slot = request->slot;
        work->slot_generation = request->slot_generation;
        work->phase = kind == VX_BATCH_KIND_PREFILL ? 0 : 1;
        work->tokens = tokens;
        work->generated = request->generated;
        work->payload = request->payload;
        work->padding = 0;

        if (kind != VX_BATCH_KIND_STATELESS) {
            VxPagedKVReservation* reservation =
                &reservations[reservation_count - 1];
            work->position = reservation->prior_length;
            work->kv_length = reservation->prior_length + tokens;
            work->page_table = vx_paged_kv_page_table(scheduler->cache) +
                (request->slot *
                 vx_paged_kv_pages_per_lane(scheduler->cache));
            work->page_tokens = vx_paged_kv_page_tokens(scheduler->cache);
        } else {
            work->kv_length = 0;
            work->position = 0;
            work->page_table = NULL;
            work->page_tokens = 0;
        }
        member_ids[useful_count++] = request->id;
    }

    if (batch_failed || useful_count != count) {
        rollback_local_reservations(scheduler, reservations,
                                    reservation_count);
        fail_selected_batch(scheduler, requests, count, kind);
        free(works);
        free(outcomes);
        free(member_ids);
        free(reservations);
        return VX_BATCH_STEP_FAILED;
    }

    if (kind == VX_BATCH_KIND_STATELESS)
        for (int index = 0; index < useful_count; index++)
            queue_remove_id(scheduler, member_ids[index]);

    if (padded_dispatch_count(scheduler, useful_count, kind, &row_count) != 0 ||
        row_count > scheduler->token_budget_per_dispatch /
                        group_key->rows_per_lane) {
        rollback_local_reservations(scheduler, reservations,
                                    reservation_count);
        fail_selected_batch(scheduler, requests, count, kind);
        free(works);
        free(outcomes);
        free(member_ids);
        free(reservations);
        return VX_BATCH_INVALID_ARGUMENT;
    }
    for (int index = useful_count; index < row_count; index++) {
        works[index] = works[0];
        works[index].request_id = -1;
        works[index].padding = 1;
    }

    useful_rows = useful_count * group_key->rows_per_lane;
    total_rows = row_count * group_key->rows_per_lane;
    metadata.kind = kind;
    metadata.group_key = *group_key;
    metadata.useful_count = useful_count;
    metadata.padded_count = row_count;
    metadata.useful_rows = useful_rows;
    metadata.total_rows = total_rows;

    VxBatchPendingDispatch local = {0};
    local.works = works; local.outcomes = outcomes;
    local.reservations = reservations; local.member_ids = member_ids;
    local.reservation_count = reservation_count; local.metadata = metadata;
    local.started_micros = -1;
    if (scheduler->deferred) {
        VxBatchPendingDispatch* pending = malloc(sizeof(*pending));
        if (!pending || scheduler->next_dispatch_id == 0) {
            free(pending);
            return batch_settle_dispatch(scheduler, &local, VX_BATCH_STEP_FAILED);
        }
        *pending = local;
        pending->id = scheduler->next_dispatch_id++;
        if (scheduler->pending_tail) scheduler->pending_tail->next = pending;
        else scheduler->pending = pending;
        scheduler->pending_tail = pending;
        return VX_BATCH_OK;
    }
    local.started_micros = monotonic_micros();
    status = run_step(works, row_count, &metadata, outcomes, user);
    return batch_settle_dispatch(scheduler, &local, status);
}

static void groups_free(VxBatchGroup* groups, int count) {
    for (int index = 0; index < count; index++) free(groups[index].members);
    free(groups);
}

static int group_push(VxBatchGroup* group, VxBatchRequest* request) {
    if (group->count == group->capacity) {
        int capacity = group->capacity ? group->capacity * 2 : 8;
        VxBatchRequest** grown = (VxBatchRequest**)realloc(
            group->members, (size_t)capacity * sizeof(*grown));
        if (!grown) return -1;
        group->members = grown;
        group->capacity = capacity;
    }
    group->members[group->count++] = request;
    if (request->id < group->oldest_id) group->oldest_id = request->id;
    return 0;
}

static VxBatchGroup* groups_find_or_add(VxBatchGroup** groups, int* count, int* capacity,
                                          const VxGroupKey* key, VxBatchRequestKind kind) {
    VxBatchGroup* group;
    for (int index = 0; index < *count; index++) {
        if ((*groups)[index].kind == kind &&
            group_key_equal(&(*groups)[index].group_key, key)) {
            return &(*groups)[index];
        }
    }
    if (*count == *capacity) {
        int grown_capacity = *capacity ? *capacity * 2 : 8;
        VxBatchGroup* grown = (VxBatchGroup*)realloc(
            *groups, (size_t)grown_capacity * sizeof(*grown));
        if (!grown) return NULL;
        *groups = grown;
        *capacity = grown_capacity;
    }
    group = &(*groups)[(*count)++];
    memset(group, 0, sizeof(*group));
    group->group_key = *key;
    group->kind = kind;
    group->oldest_id = INT32_MAX;
    return group;
}

/*
 * Collect this round's ready contributions into groups.
 *
 * The group key is the whole point of the design: contributions that cannot
 * stack into one `[B,...]` must not land in one dispatch.  Merging them and
 * labelling the batch with the first member's key would hand the callback a
 * shape it did not agree to and make `rows_useful` a fiction.
 */
static int collect_ready_groups(VxBatchScheduler* scheduler,
                                VxBatchRequest** just_admitted, int just_admitted_count,
                                VxBatchGroup** groups_out) {
    VxBatchGroup* groups = NULL;
    int count = 0, capacity = 0;

    for (int slot = 0; slot < scheduler->slots; slot++) {
        int id = scheduler->occupants[slot];
        VxBatchRequest* request;
        VxGroupKey key;
        VxBatchGroup* group;
        int skip = 0;
        if (id == -1) continue;
        request = request_find(scheduler, id);
        if (!request || request->state != VX_BATCH_REQUEST_DECODING) continue;
        for (int a = 0; a < just_admitted_count; a++) {
            if (just_admitted[a] == request) { skip = 1; break; }
        }
        if (skip) continue;
        group_key_of(&key, request, 1);
        group = groups_find_or_add(&groups, &count, &capacity, &key, VX_BATCH_KIND_DECODE);
        if (!group || group_push(group, request) != 0) {
            groups_free(groups, count);
            *groups_out = NULL;
            return -1;
        }
    }

    for (int q = 0; q < scheduler->queue_count; q++) {
        VxBatchRequest* request = request_find(scheduler, scheduler->queue[q]);
        VxGroupKey key;
        VxBatchGroup* group;
        if (!request || request->kind != VX_BATCH_KIND_STATELESS ||
            request->state != VX_BATCH_REQUEST_QUEUED) continue;
        group_key_of(&key, request, request->rows_per_lane);
        group = groups_find_or_add(&groups, &count, &capacity, &key, VX_BATCH_KIND_STATELESS);
        if (!group || group_push(group, request) != 0) {
            groups_free(groups, count);
            *groups_out = NULL;
            return -1;
        }
    }

    /*
     * Oldest arrival first. Request IDs are unique, so groups never tie.
     *
     * Every ready group is dispatched in the round, so this decides sequence
     * rather than who gets served -- which is why it needs no cross-round
     * state.  Insertion sort: the group count is bounded by the ready
     * contribution count, which is bounded by slots plus queue depth.
     */
    for (int i = 1; i < count; i++) {
        VxBatchGroup pivot = groups[i];
        int j = i - 1;
        while (j >= 0 && groups[j].oldest_id > pivot.oldest_id) {
            groups[j + 1] = groups[j];
            j--;
        }
        groups[j + 1] = pivot;
    }

    *groups_out = groups;
    return count;
}

/*
 * `fill_first`: hold this group back until it fills or its oldest member ages
 * out.
 *
 * The decision is per group.  Holding every group because the first one
 * examined was short would let a full batch wait on an unrelated partial one.
 * Decode is never held at all: those lanes are already admitted and holding
 * resident KV, so waiting buys no batching and costs a token of latency per
 * round.
 */
static int hold_for_fill(const VxBatchScheduler* scheduler, const VxBatchGroup* group,
                         int64_t policy_now_micros) {
    int64_t oldest = INT64_MAX;
    if (scheduler->policy.mode != VX_DISPATCH_FILL_FIRST) return 0;
    if (group->kind == VX_BATCH_KIND_DECODE) return 0;
    if (group->count >= scheduler->max_lanes) return 0;
    for (int index = 0; index < group->count; index++) {
        if (group->members[index]->submitted_time_micros < oldest) {
            oldest = group->members[index]->submitted_time_micros;
        }
    }
    return (policy_now_micros - oldest) < scheduler->policy.max_wait_micros;
}

static void sample_queue_depth(VxBatchScheduler* scheduler) {
    if (scheduler->queue_depth_sample_count < scheduler->queue_depth_window) {
        scheduler->queue_depth_samples[scheduler->queue_depth_sample_count++] =
            scheduler->queue_count;
        return;
    }
    scheduler->queue_depth_samples[scheduler->queue_depth_cursor] = scheduler->queue_count;
    scheduler->queue_depth_cursor =
        (scheduler->queue_depth_cursor + 1) % scheduler->queue_depth_window;
}

static void batch_finish_round(VxBatchScheduler* scheduler) {
    /* Retire cancelled requests */
    for (int slot = 0; slot < scheduler->slots; slot++) {
        int id = scheduler->occupants[slot];
        VxBatchRequest* req;
        if (id == -1) continue;
        req = request_find(scheduler, id);
        if (req && req->cancel_requested) {
            release_slot(scheduler, req);
            publish(scheduler, req, VX_BATCH_REQUEST_CANCELLED);
        }
    }

    scheduler->in_round = 0;
    retire_terminal_requests(scheduler);
    scheduler->last_activity_micros = monotonic_micros();
    if (scheduler->draining) {
        int active = 0;
        for (int i = 0; i < scheduler->slots; ++i)
            if (scheduler->occupants[i] != -1) active++;
        if (!active) scheduler->draining = 0;
    }
}

VxBatchStatus vx_batch_scheduler_step(
    VxBatchScheduler* scheduler,
    VxBatchRunStep run_step,
    void* user,
    int* worked_out) {
    int worked = 0;
    int64_t telemetry_now;
    int64_t now_policy;
    VxBatchRequest** admitted_list;
    VxBatchGroup* groups = NULL;
    int admitted_count = 0;
    int group_count;

    if (!scheduler || (!run_step && !scheduler->deferred) || scheduler->in_round)
        return VX_BATCH_INVALID_ARGUMENT;
    if (scheduler->closed && !scheduler->draining) return VX_BATCH_SCHEDULER_CLOSED;

    telemetry_now = monotonic_micros();
    if (scheduler->session_start_micros < 0) scheduler->session_start_micros = telemetry_now;

    now_policy = monotonic_micros();

    scheduler->round++;
    scheduler->in_round = 1;
    sample_queue_depth(scheduler);

    /* Sized by the slots that could be filled this round, never by a constant:
     * a scheduler with more lanes than that constant used to admit requests it
     * then never prefilled, wedging them in PREFILL for good. */
    admitted_list = (VxBatchRequest**)calloc((size_t)scheduler->slots + 1,
                                               sizeof(*admitted_list));
    if (!admitted_list) {
        scheduler->in_round = 0;
        return VX_BATCH_INVALID_ARGUMENT;
    }

    /* 1. Admission for stateful requests */
    if (!scheduler->draining && scheduler->cache) {
        admitted_count = admit_stateful(scheduler, admitted_list);
        if (admitted_count > 0) worked = 1;
    }

    /* One bounded prompt chunk per occupied prefill lane in this round.
     * The same list excludes those lanes from decode until the next round. */
    admitted_count = 0;
    for (int slot = 0; slot < scheduler->slots; ++slot) {
        VxBatchRequest* req = request_find(scheduler, scheduler->occupants[slot]);
        if (!req || req->state != VX_BATCH_REQUEST_PREFILL) continue;
        admitted_list[admitted_count++] = req;
        int rows = req->prompt_tokens - req->prefilled_tokens;
        if (rows <= 0) { req->state = VX_BATCH_REQUEST_DECODING; continue; }
        if (rows > scheduler->token_budget_per_dispatch) rows = scheduler->token_budget_per_dispatch;
        worked = 1;
        VxGroupKey key;
        VxBatchRequest* batch_req[1] = {req};
        group_key_of(&key, req, rows);
        dispatch_batch(scheduler, batch_req, 1, VX_BATCH_KIND_PREFILL, &key, run_step, user);
    }

    /* 3. Dispatch every ready group whose policy lets it go this round.
     *
     * Work-conserving means the device does not idle while a ready group
     * waits, so a round dispatches every ready group rather than one and then
     * returning.  Each group takes at most one dispatch per round, which
     * bounds the round: a group held back by the token budget drains over
     * successive rounds instead of spinning inside one. */
    group_count = collect_ready_groups(scheduler, admitted_list, admitted_count, &groups);
    if (group_count < 0) {
        free(admitted_list);
        if (!scheduler->pending) batch_finish_round(scheduler);
        return VX_BATCH_INVALID_ARGUMENT;
    }

    for (int index = 0; index < group_count; index++) {
        VxBatchGroup* group = &groups[index];
        int rows_per_lane = group->group_key.rows_per_lane;
        int max_by_budget, limit, take_count;

        if (hold_for_fill(scheduler, group, now_policy)) continue;

        max_by_budget = scheduler->token_budget_per_dispatch / (rows_per_lane > 0 ? rows_per_lane : 1);
        if (max_by_budget < 1) max_by_budget = 1;
        limit = scheduler->max_lanes < max_by_budget ? scheduler->max_lanes : max_by_budget;
        take_count = group->count < limit ? group->count : limit;

        /* With n >= k, send floor(n/k)*k and keep the remainder queued. The
         * n < k case is padded inside dispatch_batch, where a valid duplicated
         * padding row can actually be built. */
        if (scheduler->multiple_of > 1 && take_count >= scheduler->multiple_of) {
            take_count = (take_count / scheduler->multiple_of) * scheduler->multiple_of;
        }
        if (take_count <= 0) continue;

        worked = 1;
        dispatch_batch(scheduler, group->members, take_count, group->kind,
                       &group->group_key, run_step, user);
    }

    groups_free(groups, group_count);
    free(admitted_list);

    if (!scheduler->pending) batch_finish_round(scheduler);
    if (worked_out) *worked_out = worked;
    return VX_BATCH_OK;
}

VxBatchStatus vx_batch_scheduler_next(VxBatchScheduler* scheduler,
        VxBatchDispatchView* view) {
    if (!scheduler || !view) return VX_BATCH_INVALID_ARGUMENT;
    memset(view, 0, sizeof(*view));
    if (!scheduler->pending) {
        if (scheduler->closed && !scheduler->draining) return VX_BATCH_OK;
        scheduler->deferred = 1;
        VxBatchStatus status = vx_batch_scheduler_step(scheduler, NULL, NULL, NULL);
        scheduler->deferred = 0;
        if (status != VX_BATCH_OK && !scheduler->pending) return status;
    }
    VxBatchPendingDispatch* pending = scheduler->pending;
    if (pending) {
        if (pending->started_micros < 0) pending->started_micros = monotonic_micros();
        *view = (VxBatchDispatchView){pending->id, pending->works,
            &pending->metadata, pending->metadata.padded_count};
    }
    return VX_BATCH_OK;
}

int vx_batch_scheduler_peek(const VxBatchScheduler* scheduler, VxBatchDispatchView* view) {
    if (!scheduler || !view || !scheduler->pending || scheduler->pending->started_micros < 0) return 0;
    const VxBatchPendingDispatch* pending = scheduler->pending;
    *view = (VxBatchDispatchView){pending->id, pending->works, &pending->metadata, pending->metadata.padded_count};
    return 1;
}

VxBatchStatus vx_batch_scheduler_complete(VxBatchScheduler* scheduler, uint64_t id,
        const VxBatchStepOutcome* outcomes, int count, VxBatchStatus worker_status) {
    if (!scheduler || !scheduler->pending || !id || scheduler->pending->id != id ||
        scheduler->pending->started_micros < 0 ||
        (worker_status == VX_BATCH_OK && (!outcomes || count != scheduler->pending->metadata.padded_count)))
        return VX_BATCH_INVALID_ARGUMENT;
    VxBatchPendingDispatch* pending = scheduler->pending;
    if (worker_status == VX_BATCH_OK)
        memcpy(pending->outcomes, outcomes, (size_t)count * sizeof(*outcomes));
    (void)batch_settle_dispatch(scheduler, pending, worker_status);
    scheduler->pending = pending->next;
    if (!scheduler->pending) scheduler->pending_tail = NULL;
    free(pending);
    if (!scheduler->pending) batch_finish_round(scheduler);
    return VX_BATCH_OK;
}

VxBatchStatus vx_batch_scheduler_close_async(VxBatchScheduler* scheduler, int drain) {
    if (!scheduler) return VX_BATCH_INVALID_ARGUMENT;
    scheduler->closed = 1;
    scheduler->draining = drain != 0;
    while (scheduler->queue_count > 0) {
        VxBatchRequest* request = request_find(scheduler, scheduler->queue[0]);
        queue_remove_at(scheduler, 0);
        if (request) publish(scheduler, request, VX_BATCH_REQUEST_CANCELLED);
    }
    if (drain) {
        if (!scheduler->pending && scheduler->occupants) batch_finish_round(scheduler);
        return VX_BATCH_OK;
    }
    for (int i = 0; i < scheduler->request_count; ++i)
        scheduler->requests[i]->cancel_requested = 1;
    while (scheduler->pending) {
        VxBatchPendingDispatch* pending = scheduler->pending;
        scheduler->pending = pending->next;
        (void)batch_settle_dispatch(scheduler, pending, VX_BATCH_STEP_FAILED);
        free(pending);
    }
    scheduler->pending_tail = NULL;
    if (scheduler->occupants) batch_finish_round(scheduler);
    return VX_BATCH_OK;
}

int vx_batch_scheduler_closed(const VxBatchScheduler* scheduler) {
    return scheduler ? scheduler->closed : 1;
}
int vx_batch_scheduler_draining(const VxBatchScheduler* scheduler) {
    return scheduler ? scheduler->draining : 0;
}
uint64_t vx_batch_scheduler_pending_id(const VxBatchScheduler* scheduler) {
    return scheduler && scheduler->pending ? scheduler->pending->id : 0;
}
int vx_batch_scheduler_generated(const VxBatchScheduler* scheduler, int request_id) {
    VxBatchRequest* request = request_find(scheduler, request_id);
    if (request) return request->generated;
    const VxBatchResult* result = vx_batch_scheduler_result_for_request(scheduler, request_id);
    return result ? result->generated : 0;
}

VxBatchStatus vx_batch_scheduler_run_until_idle(
    VxBatchScheduler* scheduler,
    VxBatchRunStep run_step,
    void* user,
    int max_rounds) {
    if (!scheduler) return VX_BATCH_INVALID_ARGUMENT;
    if (max_rounds <= 0) max_rounds = 100000;

    for (int r = 0; r < max_rounds; r++) {
        int worked = 0;
        int active = 0;
        for (int slot = 0; slot < scheduler->slots; slot++) {
            if (scheduler->occupants[slot] != -1) active++;
        }
        if (scheduler->queue_count == 0 && active == 0) return VX_BATCH_OK;
        if (vx_batch_scheduler_step(scheduler, run_step, user, &worked) != VX_BATCH_OK) {
            return VX_BATCH_STEP_FAILED;
        }
        if (!worked) {
            /* Nothing moved and nothing can: a policy holding every ready
             * group would otherwise spin here to max_rounds. */
            int still_active = 0;
            for (int slot = 0; slot < scheduler->slots; slot++) {
                if (scheduler->occupants[slot] != -1) still_active++;
            }
            if (scheduler->queue_count == 0 && still_active == 0) return VX_BATCH_OK;
        }
    }
    return VX_BATCH_STEP_FAILED;
}

VxBatchStatus vx_batch_scheduler_close(
    VxBatchScheduler* scheduler,
    VxBatchRunStep run_step,
    void* user,
    int drain) {
    if (!scheduler) return VX_BATCH_INVALID_ARGUMENT;
    if (scheduler->closed) return VX_BATCH_OK;
    scheduler->closed = 1;

    /* Queued work is cancelled either way: with `drain` the *admitted* work
     * finishes, and a request that never reached a slot has nothing to drain.
     * Publishing it is what keeps close from abandoning accepted work
     * silently. */
    while (scheduler->queue_count > 0) {
        VxBatchRequest* req = request_find(scheduler, scheduler->queue[0]);
        queue_remove_at(scheduler, 0);
        if (req) publish(scheduler, req, VX_BATCH_REQUEST_CANCELLED);
    }
    if (!scheduler->in_round) retire_terminal_requests(scheduler);

    if (drain && run_step) {
        scheduler->draining = 1;
        while (1) {
            int active = 0;
            int worked = 0;
            for (int slot = 0; slot < scheduler->slots; slot++) {
                if (scheduler->occupants[slot] != -1) active++;
            }
            if (active == 0) break;
            if (vx_batch_scheduler_step(scheduler, run_step, user, &worked) != VX_BATCH_OK ||
                !worked) {
                break;
            }
        }
        scheduler->draining = 0;
    }

    for (int slot = 0; slot < scheduler->slots; slot++) {
        int id = scheduler->occupants[slot];
        VxBatchRequest* req;
        if (id == -1) continue;
        req = request_find(scheduler, id);
        if (req) {
            release_slot(scheduler, req);
            publish(scheduler, req, VX_BATCH_REQUEST_CANCELLED);
        }
    }
    if (!scheduler->in_round) retire_terminal_requests(scheduler);
    return VX_BATCH_OK;
}

VxBatchRequestState vx_batch_scheduler_state(const VxBatchScheduler* scheduler, int request_id) {
    VxBatchRequest* req = request_find(scheduler, request_id);
    const VxBatchResult* result;
    if (req) return req->state;
    result = vx_batch_scheduler_result_for_request(scheduler, request_id);
    return result ? result->state : VX_BATCH_REQUEST_UNKNOWN;
}

int vx_batch_scheduler_slot_of(const VxBatchScheduler* scheduler, int request_id) {
    VxBatchRequest* req = request_find(scheduler, request_id);
    return req ? req->slot : VX_BATCH_NO_SLOT;
}

int vx_batch_scheduler_queue_depth(const VxBatchScheduler* scheduler) {
    return scheduler ? scheduler->queue_count : 0;
}

int vx_batch_scheduler_round(const VxBatchScheduler* scheduler) {
    return scheduler ? scheduler->round : 0;
}

int vx_batch_scheduler_result_count(const VxBatchScheduler* scheduler) {
    return scheduler ? scheduler->result_count : 0;
}

const VxBatchResult* vx_batch_scheduler_result(const VxBatchScheduler* scheduler, int index) {
    int physical;
    if (!scheduler || index < 0 || index >= scheduler->result_count) return NULL;
    physical = (scheduler->result_start + index) % scheduler->result_capacity;
    return &scheduler->results[physical];
}

const VxBatchResult* vx_batch_scheduler_result_for_request(
    const VxBatchScheduler* scheduler, int request_id) {
    if (!scheduler || request_id <= 0) return NULL;
    for (int index = scheduler->result_count - 1; index >= 0; index--) {
        const VxBatchResult* result =
            vx_batch_scheduler_result(scheduler, index);
        if (result && result->request_id == request_id) return result;
    }
    return NULL;
}

static int compare_int(const void* left, const void* right) {
    int a = *(const int*)left;
    int b = *(const int*)right;
    return (a > b) - (a < b);
}

int vx_batch_scheduler_forget_result(VxBatchScheduler* scheduler, int request_id) {
    if (!scheduler || request_id <= 0) return 0;
    for (int index = 0; index < scheduler->result_count; ++index) {
        int physical = (scheduler->result_start + index) % scheduler->result_capacity;
        if (scheduler->results[physical].request_id != request_id) continue;
        for (int next = index + 1; next < scheduler->result_count; ++next)
            scheduler->results[(scheduler->result_start + next - 1) % scheduler->result_capacity] =
                scheduler->results[(scheduler->result_start + next) % scheduler->result_capacity];
        scheduler->result_count--;
        return 1;
    }
    return 0;
}

static int queue_depth_percentile(const VxBatchScheduler* scheduler, double fraction) {
    int* sorted;
    int index, value;
    if (scheduler->queue_depth_sample_count <= 0) return scheduler->queue_count;
    sorted = (int*)malloc((size_t)scheduler->queue_depth_sample_count * sizeof(int));
    if (!sorted) return scheduler->queue_count;
    memcpy(sorted, scheduler->queue_depth_samples,
           (size_t)scheduler->queue_depth_sample_count * sizeof(int));
    qsort(sorted, (size_t)scheduler->queue_depth_sample_count, sizeof(int), compare_int);
    index = (int)((double)scheduler->queue_depth_sample_count * fraction);
    if (index >= scheduler->queue_depth_sample_count) {
        index = scheduler->queue_depth_sample_count - 1;
    }
    value = sorted[index];
    free(sorted);
    return value;
}

/* Distinct group keys with work ready right now. */
static int active_group_count(const VxBatchScheduler* scheduler) {
    VxBatchGroupIdentity* groups;
    int capacity;
    int count = 0;

    if (scheduler->queue_count > INT_MAX - scheduler->slots) return 0;
    capacity = scheduler->slots + scheduler->queue_count;
    if (capacity <= 0) return 0;
    groups = malloc((size_t)capacity * sizeof(*groups));
    if (!groups) return 0;

    for (int q = 0; q < scheduler->queue_count; q++) {
        VxBatchRequest* request = request_find(scheduler, scheduler->queue[q]);
        VxGroupKey key;
        int seen = 0;
        if (!request) continue;
        group_key_of(&key, request, request->rows_per_lane);
        for (int i = 0; i < count; i++) {
            if (groups[i].kind == request->kind &&
                group_key_equal(&groups[i].key, &key)) {
                seen = 1;
                break;
            }
        }
        if (!seen && count < capacity) {
            groups[count].key = key;
            groups[count].kind = request->kind;
            count++;
        }
    }
    for (int slot = 0; slot < scheduler->slots; slot++) {
        int id = scheduler->occupants[slot];
        VxBatchRequest* request;
        VxGroupKey key;
        int seen = 0;
        if (id == -1) continue;
        request = request_find(scheduler, id);
        if (!request || request->state != VX_BATCH_REQUEST_DECODING) continue;
        group_key_of(&key, request, 1);
        for (int i = 0; i < count; i++) {
            if (groups[i].kind == VX_BATCH_KIND_DECODE &&
                group_key_equal(&groups[i].key, &key)) {
                seen = 1;
                break;
            }
        }
        if (!seen && count < capacity) {
            groups[count].key = key;
            groups[count].kind = VX_BATCH_KIND_DECODE;
            count++;
        }
    }

    free(groups);
    return count;
}

void vx_batch_scheduler_telemetry(const VxBatchScheduler* scheduler, VxBatchTelemetry* out) {
    int active = 0;
    if (!scheduler || !out) return;
    memset(out, 0, sizeof(*out));

    out->dispatches = scheduler->dispatches;
    out->rows_dispatched = scheduler->rows_dispatched;
    out->rows_useful = scheduler->rows_useful;
    out->device_busy_micros = scheduler->device_busy_micros;
    out->wall_micros = (scheduler->session_start_micros < 0)
        ? 0 : (scheduler->last_activity_micros - scheduler->session_start_micros);
    if (out->wall_micros < 0) out->wall_micros = 0;
    out->utilization = (out->wall_micros > 0)
        ? ((double)scheduler->device_busy_micros / (double)out->wall_micros) : 0.0;
    out->padding_waste = (scheduler->rows_dispatched > 0)
        ? (1.0 - ((double)scheduler->rows_useful / (double)scheduler->rows_dispatched)) : 0.0;

    out->queue_depth = scheduler->queue_count;
    out->max_queue_depth_seen = scheduler->max_queue_depth_seen;
    out->queue_depth_p50 = queue_depth_percentile(scheduler, 0.5);
    out->queue_depth_p99 = queue_depth_percentile(scheduler, 0.99);
    out->rounds = scheduler->round;
    out->steps = scheduler->steps;
    out->admitted = scheduler->admitted;
    out->completed = scheduler->completed;
    out->cancelled = scheduler->cancelled;
    out->failed = scheduler->failed;
    out->admission_stalls = scheduler->admission_stalls;
    out->queue_delay_rounds = scheduler->queue_delay_rounds;
    out->prefix_reuses = scheduler->prefix_reuses;
    out->prefix_publications = scheduler->prefix_publications;
    out->prefill_tokens = scheduler->prefill_tokens;
    out->shared_prompt_tokens = scheduler->shared_prompt_tokens;

    for (int slot = 0; slot < scheduler->slots; slot++) {
        if (scheduler->occupants[slot] != -1) active++;
    }
    out->active_slots = active;
    out->free_slots = scheduler->slots - active;
    out->group_count = active_group_count(scheduler);
    out->active_request_records = scheduler->request_count;
    out->max_request_records_seen = scheduler->max_request_records_seen;
    out->retained_results = scheduler->result_count;
    out->max_retained_results = scheduler->result_capacity;
}
