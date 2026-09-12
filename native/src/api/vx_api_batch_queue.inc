/* Owned wire data and generated dispatch for the engine-independent C policy.
 * Numerical execution belongs to the worker; this owner never holds a device. */
#include "batch_scheduler.h"
#include "safetensors.h"
#include "vx_thread.h"
#include <limits.h>
#include <stdatomic.h>
#include <stdlib.h>

typedef struct VxBatchOwnedWork {
    int id;
    int taken;
    VolvoxaiV1SubmitBatchWorkRequest submission;
    VolvoxaiV1BatchWorkInfo info;
    struct VxBatchOwnedWork* next;
} VxBatchOwnedWork;

typedef struct {
    atomic_uint references;
    VxMutex mutex;
    VxBatchScheduler* scheduler;
    VxPagedKVCache* cache;
    VxBatchOwnedWork* records;
} VxApiBatchQueue;

#define VX_BATCH_WIRE_COPY(type, name) \
static int vx_batch_copy_##name(type* target, const type* source) { \
    uint8_t* bytes = NULL; size_t size = 0; \
    SynurangLiteStatus status = volvoxai_v1_##name##_encode(source, &bytes, &size); \
    if (status == SYNURANG_LITE_OK) status = volvoxai_v1_##name##_decode(target, bytes, size); \
    synurang_lite_release(source->_allocator, bytes); \
    return status == SYNURANG_LITE_OK; \
}
VX_BATCH_WIRE_COPY(VolvoxaiV1SubmitBatchWorkRequest, submit_batch_work_request)
VX_BATCH_WIRE_COPY(VolvoxaiV1BatchWorkOutcome, batch_work_outcome)
VX_BATCH_WIRE_COPY(VolvoxaiV1BatchWorkInfo, batch_work_info)
VX_BATCH_WIRE_COPY(VolvoxaiV1Tensor, tensor)
#undef VX_BATCH_WIRE_COPY

static void vx_batch_owned_work_free(VxBatchOwnedWork* work) {
    volvoxai_v1_submit_batch_work_request_free(&work->submission);
    volvoxai_v1_batch_work_info_free(&work->info);
    free(work);
}
static void vx_batch_queue_retain(void* pointer) {
    atomic_fetch_add_explicit(&((VxApiBatchQueue*)pointer)->references, 1, memory_order_relaxed);
}
static void vx_batch_queue_release(void* pointer) {
    VxApiBatchQueue* queue = pointer;
    if (atomic_fetch_sub_explicit(&queue->references, 1, memory_order_acq_rel) != 1) return;
    vx_batch_scheduler_destroy(queue->scheduler);
    vx_paged_kv_destroy(queue->cache);
    while (queue->records) {
        VxBatchOwnedWork* work = queue->records; queue->records = work->next;
        vx_batch_owned_work_free(work);
    }
    vx_mutex_destroy(&queue->mutex);
    free(queue);
}
static void vx_batch_queue_sweep(VxApiBatchQueue* queue) {
    VxBatchOwnedWork** cursor = &queue->records;
    while (*cursor) {
        VxBatchOwnedWork* work = *cursor;
        if (vx_batch_scheduler_state(queue->scheduler, work->id) == VX_BATCH_REQUEST_UNKNOWN) {
            *cursor = work->next; vx_batch_owned_work_free(work);
        } else cursor = &work->next;
    }
}
static VxBatchOwnedWork* vx_batch_queue_find(VxApiBatchQueue* queue, uint32_t id) {
    if (!id || id > INT_MAX) return NULL;
    for (VxBatchOwnedWork* work = queue->records; work; work = work->next)
        if (work->id == (int)id && !work->taken) return work;
    return NULL;
}
static VxApiBatchQueue* vx_batch_queue_lock(VxApiRegistry* user_data, int64_t id, VxApiHandleLease* lease) {
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_BATCH_QUEUE, id, lease)) return NULL;
    VxApiBatchQueue* queue = lease->pointer;
    vx_mutex_lock(&queue->mutex);
    return queue;
}
static void vx_batch_queue_unlock(VxApiBatchQueue* queue, VxApiHandleLease* lease) {
    if (queue) { vx_batch_queue_sweep(queue); vx_mutex_unlock(&queue->mutex); }
    vx_api_handle_lease_release(lease);
}
static VxStatus vx_batch_status(VxBatchStatus status) {
    switch (status) {
        case VX_BATCH_OK: return VX_STATUS_OK;
        case VX_BATCH_QUEUE_FULL: return VX_STATUS_OVERLOADED;
        case VX_BATCH_SCHEDULER_CLOSED: return VX_STATUS_HANDLE_DISPOSED;
        case VX_BATCH_STEP_FAILED: return VX_STATUS_EXECUTION_FAILED;
        default: return VX_STATUS_INVALID_ARGUMENT;
    }
}
static int vx_batch_api_report(VolvoxaiV1OperationReport* response, VxStatus status) {
    VxReport report = VX_REPORT_INIT;
    report.status = status; report.stage = VX_STAGE_ADMISSION;
    VxOperationCode code = status == VX_STATUS_OK ? VX_CODE_NONE : status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
        status == VX_STATUS_BUSY ? VX_CODE_BUSY :
        status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY : status == VX_STATUS_OVERLOADED ? VX_CODE_QUEUE_FULL :
        status == VX_STATUS_NOT_FOUND ? VX_CODE_NOT_FOUND : VX_CODE_INVALID_ARGUMENT;
    report.code = code;
    return vx_api_report_from_native(response, &report) ? 0 : -1;
}
static int vx_batch_api_attach(const SynurangLiteAllocator* allocator,
        VolvoxaiV1OperationReport** report, VxStatus status) {
    if (!*report) {
        *report = allocator->allocate(allocator->context, sizeof(**report));
        if (!*report) return -1;
        volvoxai_v1_operation_report_init_with_allocator(*report, allocator);
    }
    return vx_batch_api_report(*report, status);
}
static int vx_batch_text_valid(const SynurangLiteBytes* bytes) {
    return !bytes->len || (bytes->data && !memchr(bytes->data, 0, bytes->len));
}
static int vx_batch_text_assign(const SynurangLiteAllocator* allocator,
        SynurangLiteBytes* bytes, const char* text) {
    return synurang_lite_bytes_assign(allocator, bytes, text, text ? strlen(text) : 0) == SYNURANG_LITE_OK;
}

/* Normalize every accepted host payload to owned inline bytes, including native
 * BufferView input. A scheduler must not retain caller pointers between RPCs. */
static VxStatus vx_batch_tensor_own(VolvoxaiV1Tensor* tensor) {
    if (!tensor->field_name.len || !vx_batch_text_valid(&tensor->field_name) ||
        tensor->field_location != VOLVOXAI_V1_MEMORY_LOCATION_HOST) return VX_STATUS_INVALID_ARGUMENT;
    size_t bits = safetensors_dtype_bit_width((VxDataType)tensor->field_dtype), elements = 1;
    if (!bits) return VX_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < tensor->field_shape.len; ++i) {
        int64_t dim = tensor->field_shape.data[i];
        if (dim < 0 || (uint64_t)dim > SIZE_MAX || (dim && elements > SIZE_MAX / (size_t)dim))
            return VX_STATUS_INVALID_ARGUMENT;
        elements *= (size_t)dim;
    }
    if (elements / 8 > SIZE_MAX / bits) return VX_STATUS_INVALID_ARGUMENT;
    size_t bytes = (elements / 8) * bits, tail = ((elements % 8) * bits + 7) / 8;
    if (bytes > SIZE_MAX - tail) return VX_STATUS_INVALID_ARGUMENT;
    bytes += tail;
    if (tensor->which_payload == 5)
        return tensor->field_inline.len == bytes ? VX_STATUS_OK : VX_STATUS_INVALID_ARGUMENT;
    if (tensor->which_payload != 6) return VX_STATUS_INVALID_ARGUMENT;
    void* data = NULL; size_t size = 0;
    VxStatus status = vx_api_buffer_view_resolve(tensor->field_view, &data, &size);
    if (status != VX_STATUS_OK) return status;
    if (size != bytes) return VX_STATUS_INVALID_ARGUMENT;
    if (synurang_lite_bytes_assign(tensor->_allocator, &tensor->field_inline, data, size) != SYNURANG_LITE_OK)
        return VX_STATUS_OUT_OF_MEMORY;
    volvoxai_v1_buffer_view_free(tensor->field_view);
    synurang_lite_release(tensor->_allocator, tensor->field_view);
    tensor->field_view = NULL; tensor->which_payload = 5;
    return VX_STATUS_OK;
}
static VxStatus vx_batch_tensors_own(VolvoxaiV1Tensor* tensors, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        VxStatus status = vx_batch_tensor_own(&tensors[i]);
        if (status != VX_STATUS_OK) return status;
        for (size_t j = 0; j < i; ++j)
            if (tensors[i].field_name.len == tensors[j].field_name.len &&
                !memcmp(tensors[i].field_name.data, tensors[j].field_name.data, tensors[i].field_name.len))
                return VX_STATUS_INVALID_ARGUMENT;
    }
    return VX_STATUS_OK;
}

static int vx_api_create_batch_queue(const VolvoxaiV1CreateBatchQueueRequest* request,
        VolvoxaiV1BatchQueueHandle* response, void* user_data) {
    (void)user_data;
    VxStatus status = VX_STATUS_OK;
    VxBatchSchedulerOptions options = {0}; VxPagedKVOptions pages = {0};
#define VX_BATCH_POSITIVE(wire, core) \
    if (request->has_##wire) { \
        if (!request->field_##wire || request->field_##wire > INT_MAX) status = VX_STATUS_INVALID_ARGUMENT; \
        else options.core = (int)request->field_##wire; \
    }
    VX_BATCH_POSITIVE(max_queue_depth, max_queue_depth)
    VX_BATCH_POSITIVE(token_budget, token_budget_per_dispatch)
    VX_BATCH_POSITIVE(max_lanes, max_lanes)
    VX_BATCH_POSITIVE(multiple_of, multiple_of)
    VX_BATCH_POSITIVE(queue_depth_window, queue_depth_window)
    VX_BATCH_POSITIVE(retained_results, max_retained_results)
#undef VX_BATCH_POSITIVE
    if ((request->field_policy != 0 && request->field_policy != 1) || request->field_max_wait_micros > INT64_MAX)
        status = VX_STATUS_INVALID_ARGUMENT;
    options.policy = (VxBatchDispatchPolicy){(VxDispatchPolicyMode)request->field_policy,
        (int64_t)request->field_max_wait_micros};
    if (request->field_cache) {
        const VolvoxaiV1BatchCacheOptions* cache = request->field_cache;
        if (!cache->field_lanes || cache->field_lanes > INT_MAX || !cache->field_page_tokens ||
            cache->field_page_tokens > INT_MAX || !cache->field_lane_token_capacity || cache->field_lane_token_capacity > INT_MAX ||
            (cache->has_max_pages && (!cache->field_max_pages || cache->field_max_pages > INT_MAX)) ||
            (cache->field_policy != 0 && cache->field_policy != 1)) status = VX_STATUS_INVALID_ARGUMENT;
        pages.lanes = (int)cache->field_lanes; pages.page_tokens = (int)cache->field_page_tokens;
        pages.lane_token_capacity = (int)cache->field_lane_token_capacity;
        pages.max_pages = cache->has_max_pages ? (int)cache->field_max_pages : 0;
        pages.policy = (VxPagedKVPolicy)cache->field_policy;
    }
    int lanes = options.max_lanes ? options.max_lanes : pages.lanes ? pages.lanes : 8;
    int slots = pages.lanes ? pages.lanes : lanes;
    if ((options.multiple_of && options.multiple_of > lanes) ||
        (!options.max_queue_depth && slots > INT_MAX / 4) ||
        (options.max_queue_depth && options.max_queue_depth > (INT_MAX - slots) / 2)) status = VX_STATUS_INVALID_ARGUMENT;
    if (status != VX_STATUS_OK) return vx_batch_api_attach(response->_allocator, &response->field_report, status);
    VxApiBatchQueue* queue = calloc(1, sizeof(*queue));
    if (!queue) return vx_batch_api_attach(response->_allocator, &response->field_report, VX_STATUS_OUT_OF_MEMORY);
    atomic_init(&queue->references, 1);
    if (vx_mutex_init(&queue->mutex)) { free(queue); return -1; }
    if (request->field_cache) {
        queue->cache = vx_paged_kv_create(&pages);
        if (!queue->cache) status = VX_STATUS_INVALID_ARGUMENT;
    }
    options.cache = queue->cache;
    if (status == VX_STATUS_OK && !(queue->scheduler = vx_batch_scheduler_create(&options))) status = VX_STATUS_OUT_OF_MEMORY;
    int result = vx_batch_api_attach(response->_allocator, &response->field_report, status);
    if (status == VX_STATUS_OK && !result) {
        response->field_queue_id = vx_api_handle_insert(user_data, VX_API_HANDLE_BATCH_QUEUE, queue,
            vx_batch_queue_retain, vx_batch_queue_release);
        if (response->field_queue_id) return 0;
        result = vx_batch_api_attach(response->_allocator, &response->field_report, VX_STATUS_OUT_OF_MEMORY);
    }
    vx_batch_queue_release(queue);
    return result;
}

static int vx_api_submit_batch_work(const VolvoxaiV1SubmitBatchWorkRequest* request,
        VolvoxaiV1BatchWorkHandle* response, void* user_data) {
    (void)user_data;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxApiBatchQueue* queue = vx_batch_queue_lock(user_data, request->field_queue_id, &lease);
    VxStatus status = queue ? VX_STATUS_OK : VX_STATUS_HANDLE_DISPOSED;
    VxBatchOwnedWork* work = NULL; VxApiScratch scratch = VX_API_SCRATCH_INIT;
    const char *model = NULL, *adapter = NULL, *signature = NULL, *prefix = NULL;
    if (queue && request->field_group) {
        const VolvoxaiV1BatchGroup* group = request->field_group;
        if (!vx_batch_text_valid(&group->field_model) || !vx_batch_text_valid(&group->field_adapter_revision) ||
            !vx_batch_text_valid(&group->field_shape_signature)) status = VX_STATUS_INVALID_ARGUMENT;
        model = vx_api_scratch_cstr(&scratch, &group->field_model);
        signature = vx_api_scratch_cstr(&scratch, &group->field_shape_signature);
        if (group->has_adapter_revision) {
            adapter = vx_api_scratch_cstr(&scratch, &group->field_adapter_revision);
            if (!adapter) adapter = "";
        }
    }
    if (queue && request->which_work == 5 && request->field_decode) {
        if (!vx_batch_text_valid(&request->field_decode->field_prompt_key)) status = VX_STATUS_INVALID_ARGUMENT;
        prefix = vx_api_scratch_cstr(&scratch, &request->field_decode->field_prompt_key);
    }
    if (scratch.failed) status = VX_STATUS_OUT_OF_MEMORY;
    if (status == VX_STATUS_OK) {
        work = calloc(1, sizeof(*work));
        if (!work) status = VX_STATUS_OUT_OF_MEMORY;
        else {
            volvoxai_v1_submit_batch_work_request_init(&work->submission);
            volvoxai_v1_batch_work_info_init(&work->info);
            if (!vx_batch_copy_submit_batch_work_request(&work->submission, request)) status = VX_STATUS_OUT_OF_MEMORY;
        }
    }
    if (status == VX_STATUS_OK) {
        if (request->which_work == 4 && request->field_stateless) {
            if (!request->field_stateless->field_rows || request->field_stateless->field_rows > INT_MAX)
                status = VX_STATUS_INVALID_ARGUMENT;
            else status = vx_batch_tensors_own(work->submission.field_stateless->field_inputs.data,
                work->submission.field_stateless->field_inputs.len);
        } else if (request->which_work == 5 && request->field_decode) {
            if (!request->field_decode->field_prompt_tokens || request->field_decode->field_prompt_tokens > INT_MAX ||
                !request->field_decode->field_max_tokens || request->field_decode->field_max_tokens > INT_MAX)
                status = VX_STATUS_INVALID_ARGUMENT;
        } else status = VX_STATUS_INVALID_ARGUMENT;
    }
    int result = vx_batch_api_attach(response->_allocator, &response->field_report, status);
    if (status == VX_STATUS_OK && !result) {
        if (request->which_work == 4)
            status = vx_batch_status(vx_batch_scheduler_submit_stateless(queue->scheduler, model, adapter, signature,
                (int)request->field_stateless->field_rows, work, &work->id));
        else status = vx_batch_status(vx_batch_scheduler_submit_llm(queue->scheduler, model, adapter, signature,
            (int)request->field_decode->field_prompt_tokens, (int)request->field_decode->field_max_tokens, prefix, work, &work->id));
        if (status == VX_STATUS_OK) {
            response->field_work_id = (uint32_t)work->id; response->field_queue_id = request->field_queue_id;
            work->next = queue->records; queue->records = work; work = NULL;
        } else result = vx_batch_api_attach(response->_allocator, &response->field_report, status);
    }
    if (work) vx_batch_owned_work_free(work);
    vx_api_scratch_release(&scratch);
    vx_batch_queue_unlock(queue, &lease);
    return result;
}

static int vx_api_next_batch_dispatch(const VolvoxaiV1BatchQueueRef* request,
        VolvoxaiV1BatchDispatch* response, void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* a = response->_allocator;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxApiBatchQueue* queue = vx_batch_queue_lock(user_data, request->field_queue_id, &lease);
    VxBatchDispatchView view = {0};
    VxStatus status = queue ? vx_batch_status(vx_batch_scheduler_next(queue->scheduler, &view)) : VX_STATUS_HANDLE_DISPOSED;
    response->field_queue_id = request->field_queue_id;
    if (queue) response->field_round = (uint64_t)vx_batch_scheduler_round(queue->scheduler);
    if (status == VX_STATUS_OK && view.id) {
        response->field_dispatch_id = view.id; response->field_kind = (int)view.metadata->kind;
        response->field_rows_per_item = (uint32_t)view.metadata->group_key.rows_per_lane;
        response->field_useful_items = (uint32_t)view.metadata->useful_count;
        response->field_useful_rows = (uint32_t)view.metadata->useful_rows;
        response->field_total_rows = (uint32_t)view.metadata->total_rows;
        response->field_group = a->allocate(a->context, sizeof(*response->field_group));
        if (!response->field_group) status = VX_STATUS_OUT_OF_MEMORY;
        else {
            volvoxai_v1_batch_group_init_with_allocator(response->field_group, a);
            const VxGroupKey* key = &view.metadata->group_key;
            response->field_group->has_adapter_revision = key->adapter_revision != NULL;
            if (!vx_batch_text_assign(a, &response->field_group->field_model, key->model_id) ||
                !vx_batch_text_assign(a, &response->field_group->field_adapter_revision, key->adapter_revision) ||
                !vx_batch_text_assign(a, &response->field_group->field_shape_signature, key->shape_signature_minus_batch))
                status = VX_STATUS_OUT_OF_MEMORY;
        }
        for (int i = 0; i < view.count && status == VX_STATUS_OK; ++i) {
            const VxBatchStepWork* source = &view.works[i];
            const VxBatchOwnedWork* owned = source->payload;
            VolvoxaiV1BatchDispatchWork* out = volvoxai_v1_batch_dispatch_add_work(response);
            if (!out) { status = VX_STATUS_OUT_OF_MEMORY; break; }
            out->field_work_id = source->padding ? 0 : (uint32_t)source->request_id;
            out->field_rows = (uint32_t)source->rows; out->field_lane = source->slot;
            out->field_lane_generation = (uint32_t)source->slot_generation;
            out->field_position = (uint32_t)source->position; out->field_tokens = (uint32_t)source->tokens;
            out->field_kv_length = (uint32_t)source->kv_length; out->field_page_tokens = (uint32_t)source->page_tokens;
            out->field_generated = (uint32_t)source->generated; out->field_padding = source->padding != 0;
            if (synurang_lite_bytes_assign(a, &out->field_payload, owned->submission.field_payload.data,
                owned->submission.field_payload.len) != SYNURANG_LITE_OK) status = VX_STATUS_OUT_OF_MEMORY;
            if (source->page_table) for (int p = 0; p < vx_paged_kv_pages_per_lane(queue->cache); ++p) {
                int32_t* page = volvoxai_v1_batch_dispatch_work_add_pages(out);
                if (!page) { status = VX_STATUS_OUT_OF_MEMORY; break; }
                *page = source->page_table[p];
            }
            const VolvoxaiV1StatelessBatchWork* bulk = owned->submission.field_stateless;
            if (owned->submission.which_work == 4 && bulk)
                for (size_t t = 0; t < bulk->field_inputs.len; ++t) {
                    VolvoxaiV1Tensor* tensor = volvoxai_v1_batch_dispatch_work_add_inputs(out);
                    if (!tensor || !vx_batch_copy_tensor(tensor, &bulk->field_inputs.data[t])) { status = VX_STATUS_OUT_OF_MEMORY; break; }
                }
        }
    }
    vx_batch_queue_unlock(queue, &lease);
    return vx_batch_api_attach(a, &response->field_report, status);
}

typedef struct {
    VxBatchOwnedWork* work;
    VolvoxaiV1BatchWorkOutcome outcome;
} VxBatchStagedOutcome;

static int vx_batch_reserve_value(VxBatchOwnedWork* work) {
    size_t count = work->info.field_values.len;
    if (count < work->info.field_values.cap) return 1;
    if (count >= SIZE_MAX / sizeof(VolvoxaiV1BatchValue)) return 0;
    size_t capacity = count ? count + 1 : 1;
    if (capacity <= SIZE_MAX / 2 / sizeof(VolvoxaiV1BatchValue)) capacity *= 2;
    VolvoxaiV1BatchValue* values = realloc(work->info.field_values.data, capacity * sizeof(*values));
    if (!values) return 0;
    work->info.field_values.data = values; work->info.field_values.cap = capacity;
    return 1;
}
static void vx_batch_accept_outcome(VxBatchOwnedWork* work, VolvoxaiV1BatchWorkOutcome* outcome) {
    if (outcome->has_value) {
        VolvoxaiV1BatchValue* value = &work->info.field_values.data[work->info.field_values.len++];
        volvoxai_v1_batch_value_init(value);
        value->field_data = outcome->field_value;
        outcome->field_value = (SynurangLiteBytes){0}; outcome->has_value = 0;
    }
    if (outcome->field_outputs.len) {
        for (size_t i = 0; i < work->info.field_outputs.len; ++i)
            volvoxai_v1_tensor_free(&work->info.field_outputs.data[i]);
        free(work->info.field_outputs.data);
        work->info.field_outputs.data = outcome->field_outputs.data;
        work->info.field_outputs.len = outcome->field_outputs.len;
        work->info.field_outputs.cap = outcome->field_outputs.cap;
        outcome->field_outputs.data = NULL;
        outcome->field_outputs.len = outcome->field_outputs.cap = 0;
    }
}

static int vx_api_complete_batch_dispatch(const VolvoxaiV1CompleteBatchDispatchRequest* request,
        VolvoxaiV1OperationReport* response, void* user_data) {
    (void)user_data;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxApiBatchQueue* queue = vx_batch_queue_lock(user_data, request->field_queue_id, &lease);
    VxStatus status = queue ? VX_STATUS_OK : VX_STATUS_HANDLE_DISPOSED;
    VxBatchDispatchView view = {0}; VxBatchStagedOutcome* staged = NULL;
    VxBatchStepOutcome* outcomes = NULL;
    SynurangLiteBytes error = request->field_error;
    int useful_count = 0;
    if (queue && (!vx_batch_scheduler_peek(queue->scheduler, &view) || view.id != request->field_dispatch_id ||
        (!error.len && request->field_outcomes.len != (size_t)view.count))) status = VX_STATUS_INVALID_ARGUMENT;
    if (status == VX_STATUS_OK) {
        useful_count = view.metadata->useful_count;
        if (!error.len) for (int i = 0; i < useful_count; ++i)
            if (request->field_outcomes.data[i].field_error.len) { error = request->field_outcomes.data[i].field_error; break; }
        staged = calloc((size_t)useful_count, sizeof(*staged));
        outcomes = calloc((size_t)view.count, sizeof(*outcomes));
        if (!staged || !outcomes) status = VX_STATUS_OUT_OF_MEMORY;
    }
    if (staged) for (int i = 0; i < useful_count; ++i)
        volvoxai_v1_batch_work_outcome_init(&staged[i].outcome);
    for (int i = 0; status == VX_STATUS_OK && i < useful_count; ++i) {
        staged[i].work = (VxBatchOwnedWork*)view.works[i].payload;
        if (error.len) {
            if (synurang_lite_bytes_assign(staged[i].outcome._allocator, &staged[i].outcome.field_error,
                error.data, error.len) != SYNURANG_LITE_OK) status = VX_STATUS_OUT_OF_MEMORY;
        } else {
            if (!vx_batch_copy_batch_work_outcome(&staged[i].outcome, &request->field_outcomes.data[i]))
                status = VX_STATUS_OUT_OF_MEMORY;
            if (status == VX_STATUS_OK) status = vx_batch_tensors_own(staged[i].outcome.field_outputs.data,
                staged[i].outcome.field_outputs.len);
            if (status == VX_STATUS_OK && staged[i].outcome.has_value && !vx_batch_reserve_value(staged[i].work))
                status = VX_STATUS_OUT_OF_MEMORY;
            outcomes[i].finished = staged[i].outcome.field_finished != 0;
        }
    }
    /* Response allocations and all owned values are ready before settlement. */
    int result = vx_batch_api_report(response, status);
    if (status == VX_STATUS_OK && !result) {
        status = vx_batch_status(vx_batch_scheduler_complete(queue->scheduler, view.id, outcomes, view.count,
            error.len ? VX_BATCH_STEP_FAILED : VX_BATCH_OK));
        if (status == VX_STATUS_OK) for (int i = 0; i < useful_count; ++i) {
            VxBatchOwnedWork* work = staged[i].work;
            VxBatchRequestState state = vx_batch_scheduler_state(queue->scheduler, work->id);
            if (error.len && state == VX_BATCH_REQUEST_FAILED) {
                synurang_lite_bytes_clear(work->info._allocator, &work->info.field_error);
                work->info.field_error = staged[i].outcome.field_error;
                staged[i].outcome.field_error = (SynurangLiteBytes){0};
            } else if (!error.len && !work->taken && state != VX_BATCH_REQUEST_UNKNOWN &&
                       state != VX_BATCH_REQUEST_CANCELLED && state != VX_BATCH_REQUEST_FAILED)
                vx_batch_accept_outcome(work, &staged[i].outcome);
        }
        else result = vx_batch_api_report(response, status);
    }
    if (staged) {
        for (int i = 0; i < useful_count; ++i) volvoxai_v1_batch_work_outcome_free(&staged[i].outcome);
        free(staged);
    }
    free(outcomes);
    vx_batch_queue_unlock(queue, &lease);
    return result;
}

static int vx_api_get_batch_work(const VolvoxaiV1BatchWorkRef* request,
        VolvoxaiV1BatchWorkInfo* response, void* user_data) {
    (void)user_data;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxApiBatchQueue* queue = vx_batch_queue_lock(user_data, request->field_queue_id, &lease);
    VxBatchOwnedWork* work = queue ? vx_batch_queue_find(queue, request->field_work_id) : NULL;
    VxStatus status = queue ? work ? VX_STATUS_OK : VX_STATUS_NOT_FOUND : VX_STATUS_HANDLE_DISPOSED;
    if (work) {
        work->info.field_state = (int)vx_batch_scheduler_state(queue->scheduler, work->id) + 1;
        work->info.field_generated = (uint32_t)vx_batch_scheduler_generated(queue->scheduler, work->id);
        work->info.field_lane = vx_batch_scheduler_slot_of(queue->scheduler, work->id);
        if (!vx_batch_copy_batch_work_info(response, &work->info)) status = VX_STATUS_OUT_OF_MEMORY;
    }
    response->field_queue_id = request->field_queue_id; response->field_work_id = request->field_work_id;
    vx_batch_queue_unlock(queue, &lease);
    return vx_batch_api_attach(response->_allocator, &response->field_report, status);
}

static int vx_api_take_batch_work(const VolvoxaiV1BatchWorkRef* request,
        VolvoxaiV1BatchWorkInfo* response, void* user_data) {
    (void)user_data;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxApiBatchQueue* queue = vx_batch_queue_lock(user_data, request->field_queue_id, &lease);
    VxBatchOwnedWork* work = queue ? vx_batch_queue_find(queue, request->field_work_id) : NULL;
    VxStatus status = queue ? work ? VX_STATUS_OK : VX_STATUS_NOT_FOUND : VX_STATUS_HANDLE_DISPOSED;
    if (work) {
        VxBatchRequestState state = vx_batch_scheduler_state(queue->scheduler, work->id);
        if (state < VX_BATCH_REQUEST_COMPLETED) status = VX_STATUS_BUSY;
        else {
            work->info.field_state = (int)state + 1;
            work->info.field_generated = (uint32_t)vx_batch_scheduler_generated(queue->scheduler, work->id);
            work->info.field_lane = vx_batch_scheduler_slot_of(queue->scheduler, work->id);
            if (!vx_batch_copy_batch_work_info(response, &work->info)) status = VX_STATUS_OUT_OF_MEMORY;
        }
    }
    response->field_queue_id = request->field_queue_id; response->field_work_id = request->field_work_id;
    int result = vx_batch_api_attach(response->_allocator, &response->field_report, status);
    if (status == VX_STATUS_OK && !result) {
        (void)vx_batch_scheduler_forget_result(queue->scheduler, work->id);
        work->taken = 1;
    }
    vx_batch_queue_unlock(queue, &lease);
    return result;
}

static int vx_api_get_batch_queue(const VolvoxaiV1BatchQueueRef* request,
        VolvoxaiV1BatchQueueInfo* response, void* user_data) {
    (void)user_data;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxApiBatchQueue* queue = vx_batch_queue_lock(user_data, request->field_queue_id, &lease);
    response->field_queue_id = request->field_queue_id;
    if (queue) {
        VxBatchTelemetry info; vx_batch_scheduler_telemetry(queue->scheduler, &info);
        response->field_closed = vx_batch_scheduler_closed(queue->scheduler);
        response->field_draining = vx_batch_scheduler_draining(queue->scheduler);
        response->field_pending_dispatch_id = vx_batch_scheduler_pending_id(queue->scheduler);
#define VX_BATCH_METRIC(wire, core) response->field_##wire = info.core;
        VX_BATCH_METRIC(rounds, rounds)
        VX_BATCH_METRIC(dispatches, dispatches)
        VX_BATCH_METRIC(rows_dispatched, rows_dispatched)
        VX_BATCH_METRIC(rows_useful, rows_useful)
        VX_BATCH_METRIC(worker_busy_micros, device_busy_micros)
        VX_BATCH_METRIC(wall_micros, wall_micros)
        VX_BATCH_METRIC(worker_utilization, utilization)
        VX_BATCH_METRIC(padding_waste, padding_waste)
        VX_BATCH_METRIC(queue_depth, queue_depth)
        VX_BATCH_METRIC(max_queue_depth_seen, max_queue_depth_seen)
        VX_BATCH_METRIC(queue_depth_p50, queue_depth_p50)
        VX_BATCH_METRIC(queue_depth_p99, queue_depth_p99)
        VX_BATCH_METRIC(group_count, group_count)
        VX_BATCH_METRIC(active_lanes, active_slots)
        VX_BATCH_METRIC(free_lanes, free_slots)
        VX_BATCH_METRIC(admitted, admitted)
        VX_BATCH_METRIC(completed, completed)
        VX_BATCH_METRIC(cancelled, cancelled)
        VX_BATCH_METRIC(failed, failed)
        VX_BATCH_METRIC(steps, steps)
        VX_BATCH_METRIC(admission_stalls, admission_stalls)
        VX_BATCH_METRIC(queue_delay_rounds, queue_delay_rounds)
        VX_BATCH_METRIC(prefix_reuses, prefix_reuses)
        VX_BATCH_METRIC(prefix_publications, prefix_publications)
        VX_BATCH_METRIC(prefill_tokens, prefill_tokens)
        VX_BATCH_METRIC(shared_prompt_tokens, shared_prompt_tokens)
        VX_BATCH_METRIC(active_records, active_request_records)
        VX_BATCH_METRIC(retained_results, retained_results)
        VX_BATCH_METRIC(max_retained_results, max_retained_results)
#undef VX_BATCH_METRIC
        if (queue->cache) {
            VxPagedKVTelemetry cache; vx_paged_kv_telemetry(queue->cache, &cache);
            response->field_resident_pages = (uint32_t)cache.resident_pages;
            response->field_reserved_pages = (uint32_t)cache.reserved_pages;
            response->field_free_pages = (uint32_t)cache.free_pages;
            response->field_shared_pages = (uint32_t)cache.shared_pages;
            response->field_evictions = (uint64_t)cache.evictions;
        }
    }
    vx_batch_queue_unlock(queue, &lease);
    return vx_batch_api_attach(response->_allocator, &response->field_report,
        queue ? VX_STATUS_OK : VX_STATUS_HANDLE_DISPOSED);
}

static int vx_api_cancel_batch_work(const VolvoxaiV1BatchWorkRef* request,
        VolvoxaiV1OperationReport* response, void* user_data) {
    (void)user_data;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxApiBatchQueue* queue = vx_batch_queue_lock(user_data, request->field_queue_id, &lease);
    VxStatus status = queue ? vx_batch_queue_find(queue, request->field_work_id) ? VX_STATUS_OK : VX_STATUS_NOT_FOUND : VX_STATUS_HANDLE_DISPOSED;
    int result = vx_batch_api_report(response, status);
    if (status == VX_STATUS_OK && !result) (void)vx_batch_scheduler_cancel(queue->scheduler, (int)request->field_work_id);
    vx_batch_queue_unlock(queue, &lease);
    return result;
}
static int vx_api_close_batch_queue(const VolvoxaiV1CloseBatchQueueRequest* request,
        VolvoxaiV1OperationReport* response, void* user_data) {
    (void)user_data;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxApiBatchQueue* queue = vx_batch_queue_lock(user_data, request->field_queue_id, &lease);
    int result = vx_batch_api_report(response, queue ? VX_STATUS_OK : VX_STATUS_HANDLE_DISPOSED);
    if (queue && !result) (void)vx_batch_scheduler_close_async(queue->scheduler, request->field_drain);
    vx_batch_queue_unlock(queue, &lease);
    return result;
}
static int vx_api_release_batch_queue(const VolvoxaiV1BatchQueueRef* request,
        VolvoxaiV1OperationReport* response, void* user_data) {
    (void)user_data;
    int result = vx_batch_api_report(response, VX_STATUS_OK);
    if (result) return result;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxApiBatchQueue* queue = vx_batch_queue_lock(user_data, request->field_queue_id, &lease);
    if (queue) (void)vx_batch_scheduler_close_async(queue->scheduler, 0);
    (void)vx_api_handle_remove(user_data, VX_API_HANDLE_BATCH_QUEUE, request->field_queue_id);
    vx_batch_queue_unlock(queue, &lease);
    return 0;
}
