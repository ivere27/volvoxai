void cuda_graph_reset(void) {
    /* The backend registry exists in CUDA-enabled binaries even when the
     * selected route is CPU.  A fresh CPU engine has no CUDA state and
     * therefore cannot own graph slots that need releasing. */
    if (!cuda_context_state_get()) return;
    (void)graph_release_slots();
    free(cuda_shape_signature);
    cuda_shape_signature = NULL;
    cuda_domain_span_count = 0;
    cuda_invariant_preload_complete = 0;
    cuda_domain_enforced = 0;
    cuda_graph_allocation_failed = 0;
    cuda_qbatch_matmul_dp4a_group_count = 0u;
    cuda_qbatch_matmul_scalar_tail_count = 0u;
    cuda_shape_generation = cuda_generation_next(cuda_shape_generation);
    cuda_capacity_generation =
        cuda_generation_next(cuda_capacity_generation);
}

/* A single optional instrumented plan preserves all four ordinary cache slots.
 * Its event objects are owned by the plan and are destroyed after its graphs. */
static CudaReplayState* cuda_trace_select_plan(const char* shape, uint64_t generation) {
    CudaContextState* context = cuda_context_state_get();
    CudaTracePass* pass = &context->trace_pass;
    if (!pass->scope || !pass->record_external || !vx_trace_device_nodes(pass->scope)) return NULL;
    size_t available = vx_trace_device_capacity(pass->scope, 1023);
    size_t capacity = (size_t)vx_engine_state_current()->node_count;
    if (capacity > available / 2) capacity = available / 2;
    size_t program_capacity = available - capacity;
    if (!capacity) { vx_trace_drop(pass->scope); return NULL; }
    CudaTraceReplay* cache = context->traced_replay;
    if (!cache) cache = context->traced_replay = calloc(1, sizeof(*cache));
    if (!cache) { vx_trace_device_fail(pass->scope); return NULL; }
    CudaReplayState* plan = &cache->plan;
    if (!cuda_replay_plan_exact_key_matches(plan, shape, generation) ||
        !cache->timing || cache->timing->capacity != capacity ||
        !cache->programs || cache->programs->capacity != program_capacity) {
        if (!cuda_replay_clear_plan(plan, 0)) { vx_trace_device_fail(pass->scope); return NULL; }
        size_t length = strlen(shape) + 1;
        plan->shape_signature = malloc(length);
        if (plan->shape_signature) memcpy(plan->shape_signature, shape, length);
        cache->timing = cuda_trace_nodes_create(capacity);
        cache->programs = cuda_trace_nodes_create(program_capacity);
        if (!plan->shape_signature || !cache->timing || !cache->programs) {
            cuda_replay_clear_plan(plan, 0); vx_trace_device_fail(pass->scope); return NULL;
        }
        plan->generation = generation; plan->slot_epoch = graph_slot_epoch;
        plan->shape_generation = cuda_shape_generation;
        plan->capacity_generation = cuda_capacity_generation;
        plan->domain_enforced = cuda_domain_enforced;
    }
    cache->timing->count = 0;
    pass->nodes = cache->timing;
    if (pass->owns_programs) cuda_trace_nodes_destroy(pass->programs);
    pass->programs = cache->programs; pass->owns_programs = 0;
    if (plan->plan == CUDA_REPLAY_PLAN_READY && plan->executable) {
        /* Captured launches borrow origin strings from the immutable inference
         * engine. Model generation and graph-storage changes invalidate this
         * plan before reuse. The collector copies metadata before retirement;
         * only the host admission marker changes for this submission. */
        for (size_t i = 0; i < cache->programs->count; i++)
            cache->programs->entries[i].span.host_start_ns = pass->host_start_ns;
    } else cache->programs->count = 0;
    cuda_replay_active = plan;
    return plan;
}

void cuda_graph_begin_runtime_forward(int replay_eligible,
                                      uint64_t model_generation) {
    CudaReplayState* selected = NULL;
    const char* current_shape_signature =
        cuda_shape_signature ? cuda_shape_signature : "";
    if (cuda_forward_scope_active) {
        cuda_forward_context_failed = 1;
        return;
    }
    cuda_replay_last_forward_hit = 0;
    memset(&cuda_forward_context_guard, 0,
           sizeof(cuda_forward_context_guard));
    cuda_forward_scope_active = 1;
    cuda_forward_context_failed = 0;
    cuda_forward_context_ready = cuda_ready &&
        cuda_context_enter_owned(&cuda_forward_context_guard);
    if (!cuda_forward_context_ready) {
        cuda_forward_context_failed = 1;
        return;
    }
    /* Select cached events before allocating transient program storage. */
    cuda_trace_begin("CUDA forward", 1);
    if (!replay_eligible || !cuda_graph_api_available || !model_generation) {
        cuda_replay_clear_all_plans_current(1);
        cuda_replay.pass = CUDA_REPLAY_PASS_INELIGIBLE;
        if (cuda_context_state_get()->trace_pass.scope)
            cuda_trace_programs_begin(&cuda_context_state_get()->trace_pass);
        return;
    }
    if (cuda_context_state_get()->trace_pass.scope)
        selected = cuda_trace_select_plan(current_shape_signature, model_generation);
    if (!selected) selected = cuda_replay_select_plan_current(
        current_shape_signature, model_generation);
    if (cuda_context_state_get()->trace_pass.scope)
        cuda_trace_programs_begin(&cuda_context_state_get()->trace_pass);
    if (!selected) {
        cuda_replay.pass = CUDA_REPLAY_PASS_INELIGIBLE;
        return;
    }
    cuda_replay.pass = CUDA_REPLAY_PASS_INELIGIBLE;
    cuda_replay.prepared = 0;
    cuda_replay.capture_active = 0;
    cuda_replay.failed = 0;
    cuda_replay.unsafe = 0;
    cuda_replay.prestaging = 0;
    cuda_replay.launch_index = 0;
    memset(cuda_replay.pass_writes, 0, sizeof(cuda_replay.pass_writes));
    if (cuda_replay.plan == CUDA_REPLAY_PLAN_READY &&
        cuda_replay.executable) {
        cuda_replay.pass = CUDA_REPLAY_PASS_VALIDATE;
        return;
    }
    if (cuda_replay.plan == CUDA_REPLAY_PLAN_OBSERVED) {
        cuda_replay.pass = CUDA_REPLAY_PASS_CAPTURE;
        return;
    }
    cuda_replay.pass = CUDA_REPLAY_PASS_OBSERVE;
}

int cuda_graph_prepare_runtime_forward(void) {
    CUresult result;
    if (!cuda_forward_scope_active || !cuda_forward_context_ready ||
        cuda_forward_context_failed) return -1;
    if (cuda_replay.pass == CUDA_REPLAY_PASS_CAPTURE ||
        cuda_replay.pass == CUDA_REPLAY_PASS_VALIDATE) {
        const char* current_shape_signature =
            cuda_shape_signature ? cuda_shape_signature : "";
        if (!cuda_replay_plan_exact_key_matches(
                &cuda_replay, current_shape_signature,
                cuda_replay.generation)) {
            cuda_replay_fail();
            return -1;
        }
        cuda_replay.prestaging = 1;
        for (int index = 0; index < CUDA_GRAPH_MAX_TENSORS; index++) {
            CudaTensorSlot* slot;
            if (!cuda_replay.initial_reads[index]) continue;
            if (index >= graph_slot_count) {
                cuda_replay_fail();
                break;
            }
            slot = &graph_slots[index];
            if (!slot->host || !slot->device || !slot->bytes ||
                (slot->host_dirty && !graph_upload_slot(slot))) {
                cuda_replay_fail();
                break;
            }
        }
        cuda_replay.prestaging = 0;
        if (cuda_replay.failed) return -1;
    }
    if (cuda_replay.pass == CUDA_REPLAY_PASS_CAPTURE) {
        result = p_cuStreamBeginCapture(
            cuda_stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
        if (result != CUDA_SUCCESS) {
            /* Graph replay is an optimization. A driver that exports the API
             * but cannot capture this stream falls back before any kernel was
             * withheld, leaving this forward fully executable. */
            cuda_replay_clear_all_plans_current(1);
            cuda_graph_api_available = 0;
            cuda_replay.pass = CUDA_REPLAY_PASS_INELIGIBLE;
            cuda_replay.failed = 0;
            cuda_forward_context_failed = 0;
            cuda_replay.prepared = 1;
            return 0;
        }
        cuda_replay.capture_active = 1;
        cuda_replay_capture_total++;
    }
    cuda_replay.prepared = 1;
    return 0;
}

void cuda_graph_note_node_route(int built_in_cuda) {

    if (!cuda_forward_scope_active || built_in_cuda ||
        cuda_replay.pass == CUDA_REPLAY_PASS_INELIGIBLE) return;
    if (cuda_replay.pass == CUDA_REPLAY_PASS_OBSERVE)
        cuda_replay.unsafe = 1;
    else
        cuda_replay_fail();
}

static int cuda_graph_synchronize_current(void) {
    cuda_trace_end_record();
    return cuda_stream
        ? cuda_ok(cuda_trace_wait_stream(cuda_stream), "cuStreamSynchronize")
        : cuda_ok(cuda_trace_wait_context(), "cuCtxSynchronize");
}

int cuda_graph_end_runtime_forward(int forward_ok) {
    int failed;
    int restored = 0;
    int synchronized = 0;
    if (!cuda_forward_scope_active) return -1;
    failed = cuda_forward_context_failed || cuda_replay.failed || !forward_ok;
    if (cuda_forward_context_ready) {
        if (cuda_replay.pass == CUDA_REPLAY_PASS_CAPTURE &&
            cuda_replay.capture_active) {
            CUgraph captured = NULL;
            int ended = cuda_ok(p_cuStreamEndCapture(cuda_stream, &captured),
                                "cuStreamEndCapture");
            cuda_replay.capture_active = 0;
            cuda_replay.captured_graph = captured;
            if (!ended || failed || cuda_replay.unsafe ||
                cuda_replay.launch_index != cuda_replay.launch_count ||
                !captured) {
                failed = 1;
                cuda_replay_clear_plan_current(1);
            } else {
                int instantiated = cuda_ok(p_cuGraphInstantiateWithFlags(
                        &cuda_replay.executable, captured, 0),
                    "cuGraphInstantiateWithFlags");
                int graph_destroyed = p_cuGraphDestroy &&
                    cuda_ok(p_cuGraphDestroy(captured), "cuGraphDestroy");
                if (graph_destroyed)
                    cuda_replay.captured_graph = NULL;
                if (!instantiated || !graph_destroyed ||
                    !cuda_replay.executable ||
                    !cuda_ok(cuda_trace_graph_launch(cuda_replay.executable,
                                            cuda_stream),
                             "cuGraphLaunch")) {
                    failed = 1;
                    cuda_replay_clear_plan_current(1);
                } else {
                    cuda_replay_launch_total++;
                    synchronized = cuda_graph_synchronize_current();
                    if (synchronized) {
                        cuda_replay.plan = CUDA_REPLAY_PLAN_READY;
                        cuda_replay.slot_epoch = graph_slot_epoch;
                        cuda_replay.capacity_generation =
                            cuda_capacity_generation;
                    } else {
                        failed = 1;
                        cuda_replay_clear_plan_current(1);
                    }
                }
            }
        } else if (cuda_replay.pass == CUDA_REPLAY_PASS_VALIDATE) {
            if (failed || cuda_replay.unsafe || !cuda_replay.executable ||
                cuda_replay.launch_index != cuda_replay.launch_count) {
                failed = 1;
                cuda_replay_clear_plan_current(1);
            } else if (!cuda_ok(cuda_trace_graph_launch(cuda_replay.executable,
                                                cuda_stream),
                                "cuGraphLaunch")) {
                failed = 1;
                cuda_replay_clear_plan_current(1);
            } else {
                cuda_replay_launch_total++;
                synchronized = cuda_graph_synchronize_current();
                if (!synchronized) {
                    failed = 1;
                    cuda_replay_clear_plan_current(1);
                } else {
                    cuda_replay_hit_total++;
                    cuda_replay_last_forward_hit = 1;
                }
            }
        } else {
            synchronized = cuda_graph_synchronize_current();
            if (cuda_replay.pass == CUDA_REPLAY_PASS_OBSERVE) {
                if (!failed && synchronized && !cuda_replay.unsafe &&
                    cuda_replay.launch_count > 0) {
                    cuda_replay.plan = CUDA_REPLAY_PLAN_OBSERVED;
                    cuda_replay.slot_epoch = graph_slot_epoch;
                    cuda_replay.capacity_generation =
                        cuda_capacity_generation;
                } else {
                    cuda_replay_clear_plan_current(1);
                }
            }
        }
        if (!synchronized && cuda_replay.pass != CUDA_REPLAY_PASS_CAPTURE &&
            cuda_replay.pass != CUDA_REPLAY_PASS_VALIDATE)
            failed = 1;
        cuda_trace_finish(synchronized && !failed);
        restored = cuda_context_leave(&cuda_forward_context_guard);
    }
    memset(&cuda_forward_context_guard, 0,
           sizeof(cuda_forward_context_guard));
    cuda_forward_scope_active = 0;
    cuda_forward_context_ready = 0;
    cuda_forward_context_failed = 0;
    cuda_replay.pass = CUDA_REPLAY_PASS_INELIGIBLE;
    cuda_replay.prepared = 0;
    cuda_replay.capture_active = 0;
    cuda_replay.failed = 0;
    cuda_replay.unsafe = 0;
    cuda_replay.prestaging = 0;
    return !failed && synchronized && restored ? 0 : -1;
}

void cuda_graph_mark_host(const void* host, size_t bytes, int is_weight) {
    CudaTensorView view;
    if (!graph_get_view(host, bytes, is_weight, &view)) return;
    view.slot->host_dirty = 1;
    view.slot->device_dirty = 0;
}

void cuda_graph_retain_weight(const void* host, size_t bytes) {
    int index;
    uintptr_t offset;
    CudaTensorSlot* slot;
    if (!host || !bytes) return;
    index = graph_find_containing_slot(host, bytes);
    if (index < 0) return;
    slot = &graph_slots[index];
    offset = (uintptr_t)host - (uintptr_t)slot->host;
    if (offset <= slot->bytes && bytes <= slot->bytes - offset &&
        !slot->is_weight) {
        slot->is_weight = 1;
    }
}

void cuda_graph_demote_weight(const void* host, size_t bytes) {
    int index;
    uintptr_t offset;
    CudaTensorSlot* slot;
    if (!host || !bytes) return;
    index = graph_find_containing_slot(host, bytes);
    if (index < 0) return;
    slot = &graph_slots[index];
    offset = (uintptr_t)host - (uintptr_t)slot->host;
    if (slot->domain_span || !slot->bytes || offset > slot->bytes ||
        bytes > slot->bytes - offset)
        return;
    slot->is_weight = 0;
    slot->shape_generation = cuda_shape_generation;
}

int cuda_graph_sync_host(const void* host, size_t bytes, int is_weight) {
    CudaContextState* state = cuda_context_state_get();
    int index;
    int copied;
    int full_slot;
    uintptr_t offset;
    CudaTensorSlot* slot;
    CudaContextGuard guard;
    (void)is_weight;
    if (!host || !bytes) return 0;
    /* CUDA-enabled binaries also initialize the private backend registry on
     * CPU-only machines. With no CUDA context there cannot be a live device
     * slot, so the host mirror is already authoritative. */
    if (!state || !state->ready) return 1;
    index = graph_find_containing_slot(host, bytes);
    if (index < 0) return 1;
    slot = &graph_slots[index];
    if (!slot->device) return 0;
    if (!slot->device_dirty) return 1;
    offset = (uintptr_t)host - (uintptr_t)slot->host;
    if (offset > slot->bytes || bytes > slot->bytes - offset) return 0;
    full_slot = offset == 0 && bytes == slot->bytes;
    if (cuda_forward_scope_active) {
        if (cuda_replay.pass == CUDA_REPLAY_PASS_OBSERVE)
            cuda_replay.unsafe = 1;
        else if (cuda_replay.pass == CUDA_REPLAY_PASS_CAPTURE ||
                 cuda_replay.pass == CUDA_REPLAY_PASS_VALIDATE) {
            cuda_replay_fail();
            return 0;
        }
    }
    if (!cuda_context_enter(&guard)) return 0;
    if (cuda_stream)
        copied = cuda_ok(cuda_trace_wait_stream(cuda_stream),
                         "cuStreamSynchronize");
    else
        copied = 1;
    if (copied)
        copied = cuda_ok(cuda_copy_dtoh_current((void*)host,
                                       slot->device + (CUdeviceptr)offset,
                                       bytes),
                         "cuMemcpyDtoH");
    if (!cuda_context_leave(&guard) || !copied) return 0;
    /* A row readback makes only that host range current. Keep the allocation
     * device-authoritative until a complete synchronization occurs. */
    if (full_slot) slot->device_dirty = 0;
    slot->host_dirty = 0;
    return 1;
}
