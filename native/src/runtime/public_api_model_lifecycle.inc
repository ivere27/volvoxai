/* Model and compiled-model lifecycle.
 *
 * Fragment of public_api.c, included where these definitions used to sit.
 * Covers reference counting, adapter publication, weight-revision staging,
 * the authoring-engine internals, and vx_model_compile() with its candidate
 * route search.  Kept in the caller's translation unit because it reaches
 * file-static model state and the vx_fail/vx_report_* diagnostics helpers. */

void vx_model_retain(VxModel* model) {
    if (model) atomic_fetch_add_explicit(&model->references, 1, memory_order_relaxed);
}

void vx_model_release(VxModel* model) {
    VxAdapterRevisionRecord* adapter;
    if (!model || atomic_fetch_sub_explicit(&model->references, 1,
                                            memory_order_acq_rel) != 1) return;
    vx_weight_revision_release(atomic_load_explicit(
        &model->current_weights, memory_order_relaxed));
    adapter = model->adapters;
    while (adapter) {
        VxAdapterRevisionRecord* next = adapter->next;
        vx_adapter_revision_release(adapter);
        adapter = next;
    }
    vx_declared_outputs_free(model->inputs, model->input_count);
    vx_declared_outputs_free(model->outputs, model->output_count);
    vx_model_bank_residency_clear(model);
    cJSON_Delete(model->logical_graph);
    vx_snapshot_path_release(model->graph_path);
    pthread_mutex_destroy(&model->revision_mutex);
    vx_runtime_release(model->runtime);
    free(model);
}

VxStatus vx_model_revision_info(VxModel* model,
                                VxRevisionInfo* info,
                                VxReport* report) {
    VxWeightRevisionRecord* weights;
    VxAdapterRevisionRecord* adapter;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!model || !info || info->struct_size != sizeof(*info))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_MODEL_LOAD, NULL, "INVALID_REVISION_INFO",
                       "model or revision info buffer is invalid");
    pthread_mutex_lock(&model->revision_mutex);
    weights = atomic_load_explicit(&model->current_weights,
                                   memory_order_acquire);
    adapter = atomic_load_explicit(&model->current_adapter,
                                   memory_order_acquire);
    info->graph_id = model->graph_identity;
    info->graph_revision = model->graph_revision;
    info->weight_id = weights ? weights->identity : 0;
    info->weight_revision = weights ? weights->revision : 0;
    info->adapter_id = adapter ? adapter->identity : 0;
    info->adapter_revision = adapter ? adapter->revision : 0;
    pthread_mutex_unlock(&model->revision_mutex);
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_MODEL_LOAD, NULL, NULL,
                    "OK", "published model revision returned", 0);
    vx_report_set_lineage(report, NULL, model, NULL, NULL);
    return VX_STATUS_OK;
}

VxStatus vx_model_publish_adapter(VxModel* model,
                                  const VxAdapterSource* source,
                                  VxAdapterRevision* published,
                                  VxReport* report) {
    VxAdapterRevisionRecord* previous = NULL;
    VxAdapterRevisionRecord* record;
    VxStatus record_status;
    uint64_t identity;
    uint64_t revision;
    int runtime_closed;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!model || !source || source->struct_size != sizeof(*source) ||
        !source->adapter_name || !source->adapter_name[0] ||
        strlen(source->adapter_name) >= 128u ||
        (source->package_path && !source->package_path[0]) ||
        (source->version_name &&
         (!source->version_name[0] || strlen(source->version_name) >= 96u)) ||
        (published && published->struct_size != sizeof(*published)))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_ADAPTER,
                       NULL, "INVALID_ADAPTER_SOURCE",
                       "adapter source or revision output is invalid");
    pthread_mutex_lock(&model->runtime->mutex);
    runtime_closed = model->runtime->closed;
    pthread_mutex_unlock(&model->runtime->mutex);
    if (runtime_closed)
        return vx_fail(report, VX_STATUS_HANDLE_DISPOSED, VX_STAGE_ADAPTER,
                       NULL, "HANDLE_DISPOSED", "runtime is closed");
    pthread_mutex_lock(&model->revision_mutex);
    for (VxAdapterRevisionRecord* item = model->adapters;
         item; item = item->next) {
        if (!strcmp(item->name, source->adapter_name) &&
            (!previous || item->revision > previous->revision))
            previous = item;
    }
    identity = previous ? previous->identity :
        vx_next_object_identity(model->runtime);
    revision = previous ? previous->revision + 1u : 1u;
    if (!revision) {
        pthread_mutex_unlock(&model->revision_mutex);
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_ADAPTER, NULL,
                       "REVISION_OVERFLOW", "adapter revision overflowed");
    }
    record = vx_adapter_revision_create(source->adapter_name,
                                        source->package_path,
                                        source->version_name,
                                        identity, revision, &record_status);
    if (!record) {
        pthread_mutex_unlock(&model->revision_mutex);
        return vx_fail(
            report, record_status, VX_STAGE_ADAPTER, NULL,
            record_status == VX_STATUS_IO_ERROR ? "ADAPTER_NOT_READABLE" :
                                                  "OUT_OF_MEMORY",
            record_status == VX_STATUS_IO_ERROR
                ? "adapter package file could not be snapshotted"
                : "adapter revision allocation failed");
    }
    record->next = model->adapters;
    model->adapters = record;
    atomic_store_explicit(&model->current_adapter, record,
                          memory_order_release);
    pthread_mutex_unlock(&model->revision_mutex);
    if (published) {
        published->adapter_id = identity;
        published->adapter_revision = revision;
    }
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_ADAPTER, NULL, NULL,
                    "OK", "adapter revision published atomically", 0);
    vx_report_set_lineage(report, NULL, model, NULL, NULL);
    return VX_STATUS_OK;
}

#if VOLVOXAI_ENABLE_TRAINING
VxStatus vx_model_internal_accept_full_owner(
    VxModel* model,
    VxWeightRevisionRecord** out_revision,
    VxReport* report) {
    VxWeightRevisionRecord* revision;
    if (!model || !out_revision)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_MODEL_LOAD, NULL, "INVALID_TRAINER",
                       "trainer acceptance arguments are invalid");
    *out_revision = NULL;
    /* Runtime close and Trainer acceptance linearize on this mutex. A Trainer
     * that retains its exact base revision before close remains a live child;
     * a later creation attempt is rejected. */
    pthread_mutex_lock(&model->runtime->mutex);
    if (model->runtime->closed) {
        pthread_mutex_unlock(&model->runtime->mutex);
        return vx_fail(report, VX_STATUS_HANDLE_DISPOSED,
                       VX_STAGE_MODEL_LOAD, NULL, "HANDLE_DISPOSED",
                       "runtime is closed");
    }
    pthread_mutex_lock(&model->revision_mutex);
    revision = atomic_load_explicit(&model->current_weights,
                                    memory_order_acquire);
    vx_weight_revision_retain(revision);
    pthread_mutex_unlock(&model->revision_mutex);
    pthread_mutex_unlock(&model->runtime->mutex);
    if (!revision)
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_MODEL_LOAD, NULL,
                       "REVISION_MISSING",
                       "model has no published weight revision");
    *out_revision = revision;
    return VX_STATUS_OK;
}

VxStatus vx_model_internal_validate_authoring_revision(
    VxModel* model,
    VxWeightRevisionRecord* exact_revision,
    uint64_t adapter_id,
    uint64_t adapter_revision,
    VxReport* report) {
    VxWeightRevisionRecord* current;
    VxAdapterRevisionRecord* adapter;
    VxStatus status = VX_STATUS_OK;
    if (!model || !exact_revision ||
        exact_revision->identity != model->weight_identity)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_MODEL_LOAD, NULL,
                       "INVALID_AUTHORING_REVISION",
                       "authoring revision arguments are invalid");
    pthread_mutex_lock(&model->runtime->mutex);
    if (model->runtime->closed) {
        status = VX_STATUS_HANDLE_DISPOSED;
    } else {
        pthread_mutex_lock(&model->revision_mutex);
        current = atomic_load_explicit(&model->current_weights,
                                       memory_order_acquire);
        adapter = atomic_load_explicit(&model->current_adapter,
                                       memory_order_acquire);
        /* Authoring engines currently materialize the exact base-weight
         * revision only.  They do not materialize an adapter package, so a
         * plan may be created only from the no-adapter lineage and becomes
         * stale as soon as any adapter is published. */
        if (current != exact_revision ||
            (adapter && (!adapter->name || strcmp(adapter->name, "__base__"))) ||
            adapter_id != 0u || adapter_revision != 0u)
            status = VX_STATUS_REVISION_CONFLICT;
        pthread_mutex_unlock(&model->revision_mutex);
    }
    pthread_mutex_unlock(&model->runtime->mutex);
    if (status == VX_STATUS_OK) {
        vx_report_write(report, status, VX_STAGE_MODEL_LOAD, NULL, NULL,
                        "OK", "authoring revision is current", 0);
        vx_report_set_lineage(report, NULL, model, NULL, NULL);
        return status;
    }
    return vx_fail(
        report, status, VX_STAGE_MODEL_LOAD, NULL,
        status == VX_STATUS_HANDLE_DISPOSED ? "HANDLE_DISPOSED" :
                                              "REVISION_CONFLICT",
        status == VX_STATUS_HANDLE_DISPOSED
            ? "runtime closed after authoring owner creation"
            : "model weights or adapter changed after authoring owner creation");
}

void vx_model_internal_release_weight_revision(
    VxWeightRevisionRecord* revision) {
    vx_weight_revision_release(revision);
}

void vx_model_internal_destroy_authoring_engine(VxEngineState* state) {
    VxEngineStateScope scope;
    if (!state) return;
    scope = vx_engine_state_scope_enter(state);
    volvoxai_engine_tensor_name_index_invalidate();
    volvoxai_engine_tensor_name_index_rebuild();
    /* Shutdown is also the rollback path for a partially loaded graph. */
    volvoxai_engine_shutdown();
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
}

size_t vx_model_internal_input_count(const VxModel* model) {
    return model ? model->input_count : 0u;
}

VxStatus vx_model_internal_input_spec(const VxModel* model,
                                      size_t index,
                                      VxTensorSpec* spec) {
    if (!model || !spec || spec->struct_size != sizeof(*spec))
        return VX_STATUS_INVALID_ARGUMENT;
    if (index >= model->input_count) return VX_STATUS_NOT_FOUND;
    vx_declared_tensor_spec(&model->inputs[index], spec);
    return VX_STATUS_OK;
}

const char* vx_model_internal_logical_fingerprint(const VxModel* model) {
    return model ? model->graph_fingerprint : NULL;
}

VxStatus vx_model_internal_create_authoring_engine(
    VxModel* model,
    VxWeightRevisionRecord* exact_revision,
    VolvoxAIEngineBackend backend,
    VxEngineState** out_state,
    VxReport* report) {
    const char* backend_name = vx_builtin_backend_name(backend);
    VxEngineState* state;
    VxStatus status;
    if (!model || !exact_revision || !out_state || !backend_name ||
        exact_revision->identity != model->weight_identity)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_MODEL_LOAD, backend_name,
                       "INVALID_TRAINING_ENGINE",
                       "training engine arguments are invalid");
    *out_state = NULL;
    /* Only accepted Trainers call this helper. Their retained Model/Runtime
     * ownership survives logical Runtime close, including rollback engine
     * recreation. Public Trainer creation performs the close check. */
    state = (VxEngineState*)calloc(1, sizeof(*state));
    if (!state || vx_engine_state_init(state) != 0) {
        free(state);
        return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                       VX_STAGE_MODEL_LOAD, backend_name, "OUT_OF_MEMORY",
                       "training engine state allocation failed");
    }
    status = vx_private_engine_load(state, model, exact_revision, NULL,
                                    &model->runtime->options, backend,
                                    backend_name, report,
                                    VX_STAGE_MODEL_LOAD);
    if (status != VX_STATUS_OK) {
        vx_model_internal_destroy_authoring_engine(state);
        return status;
    }
    *out_state = state;
    vx_report_set_lineage(report, NULL, model, NULL, NULL);
    return VX_STATUS_OK;
}

VxStatus vx_model_internal_prepare_weight_revision(
    VxModel* model,
    const char* const* weight_paths,
    size_t weight_path_count,
    VxWeightRevisionRecord** out_revision,
    VxReport* report) {
    VxStatus status;
    VxWeightRevisionRecord* successor;
    if (!model || !out_revision ||
        weight_path_count > (size_t)MAX_WEIGHT_FILES ||
        (weight_path_count && !weight_paths))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_MODEL_LOAD, NULL, "INVALID_WEIGHT_REVISION",
                       "successor weight source is invalid");
    *out_revision = NULL;
    /* This hidden path is used by an already-accepted Trainer. The Trainer
     * retains Model and Runtime, so logical Runtime close must not invalidate
     * its private work or atomic publication. New Trainer creation remains
     * rejected by vx_model_internal_create_authoring_engine(). */
    successor = vx_weight_revision_create(weight_paths, weight_path_count,
                                          model->weight_identity, 0, &status);
    if (!successor)
        return vx_fail(report, status, VX_STAGE_MODEL_LOAD, NULL,
                       status == VX_STATUS_IO_ERROR ?
                           "WEIGHTS_NOT_READABLE" : "OUT_OF_MEMORY",
                       status == VX_STATUS_IO_ERROR ?
                           "successor weight file could not be read" :
                           "successor weight revision allocation failed");
    status = vx_private_weight_revision_validate(model, successor, report);
    if (status != VX_STATUS_OK) {
        vx_weight_revision_release(successor);
        return status;
    }
    *out_revision = successor;
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_MODEL_LOAD, NULL, NULL,
                    "OK", "private weight successor prepared", 0);
    vx_report_set_lineage(report, NULL, model, NULL, NULL);
    return VX_STATUS_OK;
}

VxStatus vx_model_internal_publish_weight_revision(
    VxModel* model,
    VxWeightRevisionRecord* expected,
    VxWeightRevisionRecord* successor,
    VxReport* report) {
    VxWeightRevisionRecord* current;
    uint64_t next_revision;
    if (!model || !expected || !successor || successor->revision != 0 ||
        successor->identity != model->weight_identity)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_MODEL_LOAD, NULL, "INVALID_WEIGHT_REVISION",
                       "weight publication arguments are invalid");
    pthread_mutex_lock(&model->revision_mutex);
    current = atomic_load_explicit(&model->current_weights,
                                   memory_order_acquire);
    if (current != expected) {
        pthread_mutex_unlock(&model->revision_mutex);
        return vx_fail(report, VX_STATUS_REVISION_CONFLICT,
                       VX_STAGE_MODEL_LOAD, NULL,
                       "REVISION_CONFLICT",
                       "model weight revision changed since trainer creation");
    }
    next_revision = current->revision + 1u;
    if (!next_revision) {
        pthread_mutex_unlock(&model->revision_mutex);
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_MODEL_LOAD, NULL,
                       "REVISION_OVERFLOW", "weight revision overflowed");
    }
    successor->revision = next_revision;
    vx_weight_revision_retain(successor);
    atomic_store_explicit(&model->current_weights, successor,
                          memory_order_release);
    pthread_mutex_unlock(&model->revision_mutex);
    vx_weight_revision_release(current);
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_MODEL_LOAD, NULL, NULL,
                    "OK", "weight successor published atomically", 0);
    vx_report_set_lineage(report, NULL, model, NULL, NULL);
    return VX_STATUS_OK;
}
#endif

static void vx_compiled_destroy(VxCompiledModel* compiled) {
    vx_execution_context_release(compiled->route_context);
    compiled->route_context = NULL;
    if (compiled->provider) {
        if (compiled->provider_compiled)
            compiled->provider->compiled_destroy(compiled->provider_compiled);
    }
    vx_compiled_weight_store_release(compiled->weight_store);
    compiled->weight_store = NULL;
    vx_weight_revision_release(compiled->weights);
    vx_adapter_revision_release(compiled->adapter);
    vx_model_release(compiled->model);
    if (compiled->route_mutex_initialized)
        pthread_mutex_destroy(&compiled->route_mutex);
    free(compiled);
}

static VxStatus vx_model_compile_candidate(VxModel* model,
                                           const VxBackendPolicy* policy,
                                           const char* requested_backend,
                                           VxCompiledModel** out_compiled,
                                           VxReport* report) {
    VxBackendPolicy resolved = *policy;
    const VxBackendProvider* provider = NULL;
    const char* selected_backend;
    VolvoxAIEngineBackend builtin_backend = VOLVOXAI_BACKEND_CPU;
    VxProviderBinding binding = {0};
    VxCompiledModel* compiled;
    VxIndependentBatchProof independent_batch_proof;
    VxStatus status;
    VxReport compile_report = VX_REPORT_INIT;
    double compile_started_ms = vx_report_now_ms();
    int builtin;
    VxStatus missing_status;
    *out_compiled = NULL;
    selected_backend = requested_backend;
    builtin = vx_builtin_backend_by_name(requested_backend, &builtin_backend);
    if (!builtin)
        if (vx_provider_registry_find(model->runtime->providers,
                                      requested_backend, &binding))
            provider = binding.provider;
    if (!builtin && !provider) {
        missing_status = resolved.mode == VX_BACKEND_REQUIRE
            ? VX_STATUS_BACKEND_REQUIRED : VX_STATUS_BACKEND_UNAVAILABLE;
        vx_report_write(&compile_report, missing_status,
                        VX_STAGE_COMPILE, requested_backend, NULL,
                        resolved.mode == VX_BACKEND_REQUIRE
                            ? "BACKEND_REQUIRED" : "BACKEND_UNAVAILABLE",
                        resolved.mode == VX_BACKEND_REQUIRE
                            ? "required backend provider is unavailable"
                            : "preferred backend provider is unavailable",
                        0);
        vx_report_set_lineage(&compile_report, NULL, model, NULL, NULL);
        compile_report.policy_mode = resolved.mode;
        compile_report.operator_fallback = resolved.operator_fallback;
        compile_report.candidate_count = 1;
        snprintf(compile_report.candidate_outcomes,
                 sizeof(compile_report.candidate_outcomes),
                 "%s:unavailable", requested_backend);
        snprintf(compile_report.fallback_evidence,
                 sizeof(compile_report.fallback_evidence), "%s",
                 "tier=forbidden;operator=not-evaluated");
        compile_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
        vx_report_publish(report, &compile_report);
        return missing_status;
    }
    compiled = (VxCompiledModel*)calloc(1, sizeof(*compiled));
    if (!compiled) {
        vx_report_write(&compile_report, VX_STATUS_OUT_OF_MEMORY,
                        VX_STAGE_COMPILE, selected_backend, NULL,
                        "OUT_OF_MEMORY", "compiled model allocation failed", 0);
        vx_report_set_lineage(&compile_report, NULL, model, NULL, NULL);
        compile_report.policy_mode = resolved.mode;
        compile_report.operator_fallback = resolved.operator_fallback;
        compile_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
        vx_report_publish(report, &compile_report);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    if (pthread_mutex_init(&compiled->route_mutex, NULL) != 0) {
        free(compiled);
        vx_report_write(&compile_report, VX_STATUS_INTERNAL,
                        VX_STAGE_COMPILE, selected_backend, NULL,
                        "MUTEX_INIT_FAILED",
                        "compiled route lease initialization failed", 0);
        vx_report_set_lineage(&compile_report, NULL, model, NULL, NULL);
        vx_report_publish(report, &compile_report);
        return VX_STATUS_INTERNAL;
    }
    compiled->route_mutex_initialized = 1;
    atomic_init(&compiled->references, 1);
    atomic_init(&compiled->scheduled_route_waiters, 0);
    compiled->model = model;
    vx_model_retain(model);
    pthread_mutex_lock(&model->revision_mutex);
    compiled->weights = atomic_load_explicit(&model->current_weights,
                                              memory_order_acquire);
    compiled->adapter = atomic_load_explicit(&model->current_adapter,
                                              memory_order_acquire);
    vx_weight_revision_retain(compiled->weights);
    vx_adapter_revision_retain(compiled->adapter);
    pthread_mutex_unlock(&model->revision_mutex);
    compiled->identity = vx_next_object_identity(model->runtime);
    compiled->batch_contract =
        (VxBackendBatchContract)VX_BACKEND_BATCH_CONTRACT_INIT;
    vx_independent_batch_prove(model, &independent_batch_proof);
    compiled->independent_batch_supported =
        independent_batch_proof.supported;
    snprintf(compiled->independent_batch_proof_identity,
             sizeof(compiled->independent_batch_proof_identity), "%s",
             independent_batch_proof.identity);
    snprintf(compiled->independent_batch_rejection,
             sizeof(compiled->independent_batch_rejection), "%s",
             independent_batch_proof.reason);
    snprintf(compiled->independent_batch_failed_node,
             sizeof(compiled->independent_batch_failed_node), "%s",
             independent_batch_proof.failed_node);
    compiled->allocated_bytes = sizeof(*compiled);
    compiled->mode = resolved.mode;
    compiled->operator_fallback = resolved.operator_fallback;
    compiled->builtin_backend = builtin_backend;
    snprintf(compiled->backend, sizeof(compiled->backend), "%s", selected_backend);
    compiled->report = (VxReport)VX_REPORT_INIT;
    if (provider) {
        VxModelSource source;
        VxBackendCompileInput compile_input = VX_BACKEND_COMPILE_INPUT_INIT;
        VxBackendShapeDomainAttestation attestation =
            VX_BACKEND_SHAPE_DOMAIN_ATTESTATION_INIT;
        VxTensorSpec* input_specs = NULL;
        VxTensorSpec* output_specs = NULL;
        VxReport provider_report = VX_REPORT_INIT;
        compiled->provider = provider;
        compiled->provider_runtime = binding.runtime_instance;
        if (provider->shape_domain.support != VX_BACKEND_SHAPE_DOMAIN_FULL) {
            vx_report_write(&compile_report, VX_STATUS_BACKEND_UNSUPPORTED,
                            VX_STAGE_COMPILE, selected_backend, NULL,
                            "BOUNDED_DOMAIN_UNSUPPORTED",
                            "provider does not support the complete bounded shape domain",
                            0);
            vx_report_set_lineage(&compile_report, NULL, model, compiled, NULL);
            compile_report.candidate_count = 1;
            snprintf(compile_report.candidate_outcomes,
                     sizeof(compile_report.candidate_outcomes),
                     "%s:domain-unsupported", selected_backend);
            vx_report_publish(report, &compile_report);
            vx_compiled_destroy(compiled);
            return VX_STATUS_BACKEND_UNSUPPORTED;
        }
        input_specs = vx_declared_tensor_specs(model->inputs,
                                               model->input_count);
        output_specs = vx_declared_tensor_specs(model->outputs,
                                                model->output_count);
        if ((model->input_count && !input_specs) ||
            (model->output_count && !output_specs)) {
            free(input_specs);
            free(output_specs);
            vx_compiled_destroy(compiled);
            return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                           VX_STAGE_COMPILE, selected_backend,
                           "OUT_OF_MEMORY", "compile-spec allocation failed");
        }
        vx_model_source_view(model, compiled->weights, &source);
        compile_input.source = &source;
        compile_input.graph_fingerprint = model->graph_fingerprint;
        compile_input.shape_domain_proof_identity =
            model->shape_domain_proof_identity;
        compile_input.independent_batch_proof_protocol =
            VX_BACKEND_INDEPENDENT_BATCH_PROOF_PROTOCOL;
        compile_input.independent_batch_proof_identity =
            independent_batch_proof.supported
                ? compiled->independent_batch_proof_identity : NULL;
        compile_input.inputs = input_specs;
        compile_input.input_count = model->input_count;
        compile_input.outputs = output_specs;
        compile_input.output_count = model->output_count;
        provider_report = (VxReport)VX_REPORT_INIT;
        status = provider->compile(compiled->provider_runtime, &compile_input,
                                   &resolved, &compiled->provider_compiled,
                                   &attestation, &provider_report);
        vx_report_normalize_provider_output(&provider_report);
        free(input_specs);
        free(output_specs);
        if (status != VX_STATUS_OK || !compiled->provider_compiled) {
            VxStatus failure = status == VX_STATUS_OK
                ? VX_STATUS_BACKEND_UNSUPPORTED : status;
            vx_report_write(&compile_report, failure, VX_STAGE_COMPILE,
                            selected_backend, provider_report.device,
                            "BACKEND_UNSUPPORTED", "backend compilation failed", 0);
            vx_report_set_lineage(&compile_report, NULL, model, compiled, NULL);
            compile_report.policy_mode = resolved.mode;
            compile_report.operator_fallback = resolved.operator_fallback;
            compile_report.candidate_count = 1;
            snprintf(compile_report.candidate_outcomes,
                     sizeof(compile_report.candidate_outcomes),
                     "%s:compile-failed", selected_backend);
            snprintf(compile_report.offending_node,
                     sizeof(compile_report.offending_node), "%s",
                     provider_report.offending_node);
            snprintf(compile_report.route_evidence,
                     sizeof(compile_report.route_evidence), "%s",
                     provider_report.route_evidence);
            compile_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
            vx_report_publish(report, &compile_report);
            vx_compiled_destroy(compiled);
            return failure;
        }
        if (!vx_provider_attestation_valid(model, &attestation)) {
            vx_report_write(&compile_report, VX_STATUS_ABI_UNSUPPORTED,
                            VX_STAGE_COMPILE, selected_backend,
                            provider_report.device,
                            "SHAPE_DOMAIN_ATTESTATION_INVALID",
                            "provider did not attest the exact bounded shape domain and resources",
                            0);
            vx_report_set_lineage(&compile_report, NULL, model, compiled, NULL);
            compile_report.candidate_count = 1;
            snprintf(compile_report.candidate_outcomes,
                     sizeof(compile_report.candidate_outcomes),
                     "%s:domain-unattested", selected_backend);
            vx_report_publish(report, &compile_report);
            vx_compiled_destroy(compiled);
            return VX_STATUS_ABI_UNSUPPORTED;
        }
        if (!provider_report.route_attested || !provider_report.route_evidence[0] ||
            !provider_report.device[0] ||
            (resolved.operator_fallback == VX_OPERATOR_FALLBACK_FORBID &&
             provider_report.operator_fallback_used)) {
            const int strict = resolved.operator_fallback == VX_OPERATOR_FALLBACK_FORBID;
            vx_report_write(&compile_report,
                            strict ? VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN :
                                     VX_STATUS_BACKEND_UNSUPPORTED,
                            VX_STAGE_COMPILE, selected_backend,
                            provider_report.device,
                            strict ? "OPERATOR_FALLBACK_FORBIDDEN" :
                                     "BACKEND_UNSUPPORTED",
                            strict ? "provider did not attest an all-provider route" :
                                     "provider did not report exact route evidence",
                            0);
            vx_report_set_lineage(&compile_report, NULL, model, compiled, NULL);
            compile_report.policy_mode = resolved.mode;
            compile_report.operator_fallback = resolved.operator_fallback;
            compile_report.operator_fallback_used =
                provider_report.operator_fallback_used;
            compile_report.route_attested = provider_report.route_attested;
            compile_report.candidate_count = 1;
            snprintf(compile_report.candidate_outcomes,
                     sizeof(compile_report.candidate_outcomes),
                     "%s:route-unattested", selected_backend);
            snprintf(compile_report.route_evidence,
                     sizeof(compile_report.route_evidence), "%s",
                     provider_report.route_evidence);
            snprintf(compile_report.fallback_evidence,
                     sizeof(compile_report.fallback_evidence), "%s",
                     provider_report.fallback_evidence[0]
                         ? provider_report.fallback_evidence :
                           "operator=unattested");
            snprintf(compile_report.offending_node,
                     sizeof(compile_report.offending_node), "%s",
                     provider_report.offending_node);
            compile_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
            vx_report_publish(report, &compile_report);
            vx_compiled_destroy(compiled);
            return strict ? VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN :
                            VX_STATUS_BACKEND_UNSUPPORTED;
        }
        if (provider->compiled_batch_contract) {
            VxBackendBatchContract contract =
                VX_BACKEND_BATCH_CONTRACT_INIT;
            VxReport contract_report = VX_REPORT_INIT;
            status = provider->compiled_batch_contract(
                compiled->provider_compiled, &contract, &contract_report);
            vx_report_normalize_provider_output(&contract_report);
            if (status != VX_STATUS_OK ||
                contract.struct_size != sizeof(contract) ||
                !contract.protocol ||
                strcmp(contract.protocol, VX_BACKEND_BATCH_PROTOCOL) ||
                !contract.resource_domain || !contract.compatibility_token ||
                !contract.min_batch || contract.max_batch < contract.min_batch ||
                !contract.multiple_of ||
                contract.min_batch % contract.multiple_of ||
                contract.batch_axis < 0 ||
                (contract.device_resident != 0 &&
                 contract.device_resident != 1)) {
                VxStatus failure = status == VX_STATUS_OUT_OF_MEMORY
                    ? VX_STATUS_OUT_OF_MEMORY : VX_STATUS_ABI_UNSUPPORTED;
                vx_report_write(
                    &compile_report, failure, VX_STAGE_COMPILE,
                    selected_backend, contract_report.device,
                    failure == VX_STATUS_OUT_OF_MEMORY
                        ? "OUT_OF_MEMORY" : "BATCH_CONTRACT_INVALID",
                    failure == VX_STATUS_OUT_OF_MEMORY
                        ? "provider batch contract allocation failed"
                        : "provider returned an invalid frozen batch contract",
                    0);
                vx_report_set_lineage(&compile_report, NULL, model, compiled,
                                      NULL);
                vx_report_publish(report, &compile_report);
                vx_compiled_destroy(compiled);
                return failure;
            }
            if (contract.max_batch > 1u &&
                (!independent_batch_proof.supported ||
                 !contract.graph_fingerprint ||
                 strcmp(contract.graph_fingerprint,
                        model->graph_fingerprint) ||
                 !contract.independent_batch_proof_identity ||
                 strcmp(contract.independent_batch_proof_identity,
                        compiled->independent_batch_proof_identity) ||
                 contract.batch_axis !=
                    (int32_t)independent_batch_proof.batch_axis ||
                 contract.min_batch < independent_batch_proof.min_batch ||
                 contract.max_batch > independent_batch_proof.max_batch ||
                 contract.multiple_of %
                    independent_batch_proof.multiple_of)) {
                vx_report_write(
                    &compile_report, VX_STATUS_ABI_UNSUPPORTED,
                    VX_STAGE_COMPILE, selected_backend,
                    contract_report.device,
                    "BATCH_PROOF_MISMATCH",
                    "provider batch axis or range is outside the exact core-authored independent-batch proof",
                    0);
                vx_report_set_lineage(&compile_report, NULL, model, compiled,
                                      NULL);
                vx_report_publish(report, &compile_report);
                vx_compiled_destroy(compiled);
                return VX_STATUS_ABI_UNSUPPORTED;
            }
            for (size_t input_index = 0;
                 input_index < model->input_count; input_index++) {
                const VxDeclaredTensor* input =
                    &model->inputs[input_index];
                uint32_t axis = (uint32_t)contract.batch_axis;
                if (axis >= input->rank ||
                    input->kinds[axis] != VX_DIMENSION_SYMBOLIC ||
                    !input->symbols[axis] || !input->symbols[axis][0] ||
                    input->minimums[axis] > 1 || input->maximums[axis] < 1 ||
                    1 % input->multiples[axis] ||
                    (uint64_t)input->maximums[axis] < contract.max_batch ||
                    contract.min_batch % (uint64_t)input->multiples[axis] ||
                    contract.multiple_of %
                        (uint64_t)input->multiples[axis]) {
                    vx_report_write(&compile_report,
                                    VX_STATUS_ABI_UNSUPPORTED,
                                    VX_STAGE_COMPILE, selected_backend,
                                    contract_report.device,
                                    "BATCH_AXIS_INVALID",
                                    "provider batch range is outside a symbolic logical input axis",
                                    0);
                    vx_report_set_lineage(&compile_report, NULL, model,
                                          compiled, NULL);
                    vx_report_publish(report, &compile_report);
                    vx_compiled_destroy(compiled);
                    return VX_STATUS_ABI_UNSUPPORTED;
                }
            }
            if (model->input_count) {
                const char* batch_symbol =
                    model->inputs[0].symbols[contract.batch_axis];
                for (size_t input_index = 1;
                     input_index < model->input_count; input_index++) {
                    if (strcmp(batch_symbol,
                               model->inputs[input_index].symbols[
                                   contract.batch_axis])) {
                        vx_report_write(&compile_report,
                                        VX_STATUS_ABI_UNSUPPORTED,
                                        VX_STAGE_COMPILE, selected_backend,
                                        contract_report.device,
                                        "BATCH_AXIS_MISMATCH",
                                        "provider inputs do not share one symbolic batch axis",
                                        0);
                        vx_report_set_lineage(&compile_report, NULL, model,
                                              compiled, NULL);
                        vx_report_publish(report, &compile_report);
                        vx_compiled_destroy(compiled);
                        return VX_STATUS_ABI_UNSUPPORTED;
                    }
                }
                for (size_t output_index = 0;
                     output_index < model->output_count; output_index++) {
                    const VxDeclaredTensor* output =
                        &model->outputs[output_index];
                    uint32_t axis = (uint32_t)contract.batch_axis;
                    if (axis >= output->rank ||
                        output->kinds[axis] != VX_DIMENSION_SYMBOLIC ||
                        !output->symbols[axis] ||
                        strcmp(batch_symbol, output->symbols[axis]) ||
                        output->maximums[axis] <
                            (int64_t)contract.max_batch) {
                        vx_report_write(&compile_report,
                                        VX_STATUS_ABI_UNSUPPORTED,
                                        VX_STAGE_COMPILE, selected_backend,
                                        contract_report.device,
                                        "BATCH_OUTPUT_AXIS_INVALID",
                                        "provider outputs do not preserve the symbolic batch axis",
                                        0);
                        vx_report_set_lineage(&compile_report, NULL, model,
                                              compiled, NULL);
                        vx_report_publish(report, &compile_report);
                        vx_compiled_destroy(compiled);
                        return VX_STATUS_ABI_UNSUPPORTED;
                    }
                }
            }
            compiled->batch_contract = contract;
            compiled->batch_contract.protocol = VX_BACKEND_BATCH_PROTOCOL;
        }
        compile_report = provider_report;
        compile_report.struct_size = sizeof(compile_report);
        compile_report.status = VX_STATUS_OK;
        compile_report.stage = VX_STAGE_COMPILE;
        compile_report.execution_id = 0;
        snprintf(compile_report.backend, sizeof(compile_report.backend), "%s",
                 selected_backend);
        snprintf(compile_report.reason, sizeof(compile_report.reason), "%s", "OK");
        if (!compile_report.message[0])
            snprintf(compile_report.message, sizeof(compile_report.message), "%s",
                     "backend compilation completed");
    } else {
        VxEngineState* validation = (VxEngineState*)calloc(1, sizeof(*validation));
        VxBuiltinResourceDomain resource_domain = {0};
        char domain_evidence[VX_REPORT_ROUTE_CAPACITY] = {0};
        status = vx_compiled_weight_store_create(
            compiled->weights, &compiled->weight_store);
        if (status != VX_STATUS_OK || !compiled->weight_store ||
            compiled->weight_store->allocated_bytes >
                UINT64_MAX - compiled->allocated_bytes) {
            if (status == VX_STATUS_OK) status = VX_STATUS_OUT_OF_MEMORY;
            free(validation);
            vx_report_write(
                &compile_report, status, VX_STAGE_COMPILE, selected_backend,
                vx_builtin_device_identity(selected_backend),
                status == VX_STATUS_OUT_OF_MEMORY ? "OUT_OF_MEMORY" :
                                                    "WEIGHT_STORE_INVALID",
                status == VX_STATUS_OUT_OF_MEMORY
                    ? "compiled immutable weight store allocation failed"
                    : "accepted weight revision could not be prepared",
                0);
            vx_report_set_lineage(&compile_report, NULL, model, compiled, NULL);
            compile_report.policy_mode = resolved.mode;
            compile_report.operator_fallback = resolved.operator_fallback;
            compile_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
            vx_report_publish(report, &compile_report);
            vx_compiled_destroy(compiled);
            return status;
        }
        compiled->allocated_bytes += compiled->weight_store->allocated_bytes;
        if (!validation || vx_engine_state_init(validation) != 0) {
            free(validation);
            vx_report_write(&compile_report, VX_STATUS_OUT_OF_MEMORY,
                            VX_STAGE_COMPILE, selected_backend,
                            vx_builtin_device_identity(selected_backend),
                            "OUT_OF_MEMORY", "validation state allocation failed", 0);
            vx_report_set_lineage(&compile_report, NULL, model, compiled, NULL);
            compile_report.policy_mode = resolved.mode;
            compile_report.operator_fallback = resolved.operator_fallback;
            compile_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
            vx_report_publish(report, &compile_report);
            vx_compiled_destroy(compiled);
            return VX_STATUS_OUT_OF_MEMORY;
        }
        status = vx_private_engine_load(validation, model, compiled->weights,
                                        compiled->weight_store,
                                        &model->runtime->options,
                                        compiled->builtin_backend,
                                        selected_backend,
                                        &compile_report, VX_STAGE_COMPILE);
        if (status == VX_STATUS_OK) {
            status = vx_builtin_bounded_domain_proof(
                model, compiled->weights, compiled->builtin_backend,
                validation,
                &resource_domain,
                domain_evidence,
                sizeof(domain_evidence));
            if (status != VX_STATUS_OK) {
                int contract_unsupported =
                    strstr(domain_evidence,
                           "required=canonical-native-shape-contract") != NULL;
                vx_report_write(
                    &compile_report, status, VX_STAGE_COMPILE,
                    selected_backend,
                    vx_builtin_device_identity(selected_backend),
                    status == VX_STATUS_OUT_OF_MEMORY ? "OUT_OF_MEMORY" :
                    contract_unsupported
                        ? "CANONICAL_SHAPE_CONTRACT_UNSUPPORTED"
                        : "BOUNDED_DOMAIN_UNSUPPORTED",
                    status == VX_STATUS_OUT_OF_MEMORY
                        ? "bounded-domain proof allocation failed"
                    : contract_unsupported
                        ? "node lacks a canonical native shape contract; re-export after native contract support is added"
                        : "built-in backend cannot attest the complete declared shape domain",
                    0);
            }
        }
        if (status == VX_STATUS_OK &&
            compiled->builtin_backend != VOLVOXAI_BACKEND_CPU &&
            validation->node_count > 0) {
            validation->bounded_gpu_value_domain_proven =
                vx_native_gpu_backend(compiled->builtin_backend);
            VxEngineStateScope validation_scope =
                vx_engine_state_scope_enter(validation);
            int forward_status = volvoxai_engine_forward();
            vx_engine_state_scope_leave(validation_scope);
            if (forward_status != 0) {
                status = VX_STATUS_BACKEND_UNSUPPORTED;
                vx_report_write(&compile_report, status, VX_STAGE_COMPILE,
                                selected_backend,
                                vx_builtin_device_identity(selected_backend),
                                "BACKEND_UNSUPPORTED",
                                "built-in backend cannot execute the complete graph",
                                0);
            }
        }
        /* Vulkan's fixed tail scratch also contains every shape-invariant
         * dispatch payload from one whole graph forward. The first proof is
         * intentionally only a gate before execution. Re-run it after the
         * validation forward so the queried high-water mark is included in
         * the authoritative resource bound and maximum-domain reservation. */
        if (status == VX_STATUS_OK &&
            compiled->builtin_backend == VOLVOXAI_BACKEND_VULKAN) {
            status = vx_builtin_bounded_domain_proof(
                model, compiled->weights, compiled->builtin_backend,
                validation, &resource_domain, domain_evidence,
                sizeof(domain_evidence));
            if (status != VX_STATUS_OK) {
                int contract_unsupported =
                    strstr(domain_evidence,
                           "required=canonical-native-shape-contract") != NULL;
                vx_report_write(
                    &compile_report, status, VX_STAGE_COMPILE,
                    selected_backend,
                    vx_builtin_device_identity(selected_backend),
                    status == VX_STATUS_OUT_OF_MEMORY ? "OUT_OF_MEMORY" :
                    contract_unsupported
                        ? "CANONICAL_SHAPE_CONTRACT_UNSUPPORTED"
                        : "BOUNDED_DOMAIN_UNSUPPORTED",
                    status == VX_STATUS_OUT_OF_MEMORY
                        ? "bounded-domain proof allocation failed"
                    : contract_unsupported
                        ? "node lacks a canonical native shape contract; re-export after native contract support is added"
                        : "built-in backend cannot attest the complete declared shape domain",
                    0);
            }
        }
        if (status == VX_STATUS_OK) {
            compiled->maximum_cpu_typed_workspace_bytes =
                resource_domain.maximum_typed_scratch_bytes;
            compiled->maximum_resident_bytes =
                resource_domain.maximum_resident_bytes;
            if (independent_batch_proof.supported &&
                (compiled->builtin_backend == VOLVOXAI_BACKEND_VULKAN ||
                 compiled->builtin_backend == VOLVOXAI_BACKEND_OPENGL ||
                 compiled->builtin_backend == VOLVOXAI_BACKEND_CUDA)) {
                compiled->batch_contract.graph_fingerprint =
                    model->graph_fingerprint;
                compiled->batch_contract.independent_batch_proof_identity =
                    compiled->independent_batch_proof_identity;
                compiled->batch_contract.resource_domain =
                    vx_builtin_gpu_resource_domain(
                        compiled->builtin_backend);
                compiled->batch_contract.compatibility_token = compiled;
                compiled->batch_contract.min_batch =
                    independent_batch_proof.min_batch;
                compiled->batch_contract.max_batch =
                    independent_batch_proof.max_batch;
                compiled->batch_contract.multiple_of =
                    independent_batch_proof.multiple_of;
                compiled->batch_contract.batch_axis =
                    (int32_t)independent_batch_proof.batch_axis;
                compiled->batch_contract.device_resident = 1;
                compiled->batch_contract.device_epoch = 1u;
            }
        }
        if (status == VX_STATUS_OK || status == VX_STATUS_BACKEND_UNSUPPORTED)
            vx_report_set_builtin_route(&compile_report, validation,
                                        selected_backend);
        if (domain_evidence[0]) {
            size_t used = strlen(compile_report.route_evidence);
            if (used < sizeof(compile_report.route_evidence))
                snprintf(compile_report.route_evidence + used,
                         sizeof(compile_report.route_evidence) - used,
                         ";%s", domain_evidence);
        }
        if (status == VX_STATUS_OK &&
            compiled->batch_contract.max_batch > 1u &&
            !vx_report_prepend_batch_contract(
                &compile_report, &compiled->batch_contract)) {
            status = VX_STATUS_INTERNAL;
            vx_report_write(
                &compile_report, status, VX_STAGE_COMPILE,
                selected_backend,
                vx_builtin_device_identity(selected_backend),
                "BATCH_EVIDENCE_UNAVAILABLE",
                "exact public batch-contract evidence did not fit", 0);
        }
        if (status == VX_STATUS_OK)
            status = vx_builtin_compile_route_status(
                &compile_report, resolved.operator_fallback);
        if (status != VX_STATUS_OK)
            vx_report_set_builtin_failure_node(&compile_report, validation);
        {
            VxExecutionContext temporary = {0};
            temporary.engine_state = validation;
            temporary.engine_state_initialized = 1;
            /* shutdown is also the rollback path for partial initialization */
            temporary.engine_loaded = 1;
            vx_private_engine_unload(&temporary);
        }
        if (status != VX_STATUS_OK) {
            vx_report_set_lineage(&compile_report, NULL, model, compiled, NULL);
            compile_report.policy_mode = resolved.mode;
            compile_report.operator_fallback = resolved.operator_fallback;
            compile_report.candidate_count = 1u;
            snprintf(compile_report.candidate_outcomes,
                     sizeof(compile_report.candidate_outcomes), "%s:rejected",
                     selected_backend);
            compile_report.tier_fallback_used = 0;
            compile_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
            vx_report_publish(report, &compile_report);
            vx_compiled_destroy(compiled);
            return status;
        }
    }
    vx_report_set_lineage(&compile_report, NULL, model, compiled, NULL);
    compile_report.policy_mode = resolved.mode;
    compile_report.operator_fallback = resolved.operator_fallback;
    compile_report.candidate_count = 1u;
    compile_report.tier_fallback_used = 0;
    snprintf(compile_report.candidate_outcomes,
             sizeof(compile_report.candidate_outcomes),
             "%s:selected", selected_backend);
    if (!compile_report.fallback_evidence[0])
        snprintf(compile_report.fallback_evidence,
                 sizeof(compile_report.fallback_evidence), "%s",
                 "tier=none;operator=none");
    if (!compile_report.decode_state[0])
        snprintf(compile_report.decode_state, sizeof(compile_report.decode_state),
                 "%s", "none");
    compile_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
    compiled->report = compile_report;
    vx_report_publish(report, &compile_report);
    *out_compiled = compiled;
    return VX_STATUS_OK;
}

static const char* vx_candidate_outcome(VxStatus status,
                                        const VxReport* report) {
    if (status == VX_STATUS_OK) return "selected";
    if (status == VX_STATUS_BACKEND_UNAVAILABLE ||
        status == VX_STATUS_BACKEND_REQUIRED) return "unavailable";
    if (status == VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN)
        return "operator-fallback-forbidden";
    if (status == VX_STATUS_INVALID_GRAPH) return "graph-rejected";
    if (status == VX_STATUS_OUT_OF_MEMORY) return "out-of-memory";
    if (report && !strcmp(report->reason, "BACKEND_UNSUPPORTED"))
        return "unsupported";
    return "rejected";
}

static void vx_candidate_outcomes_append(char* outcomes,
                                         size_t capacity,
                                         const char* backend,
                                         const char* outcome) {
    size_t used;
    if (!outcomes || !capacity || !backend || !outcome) return;
    used = strlen(outcomes);
    if (used >= capacity) return;
    snprintf(outcomes + used, capacity - used, "%s%s:%s",
             used ? ";" : "", backend, outcome);
}

VxStatus vx_model_compile(VxModel* model,
                          const VxBackendPolicy* policy,
                          VxCompiledModel** out_compiled,
                          VxReport* report) {
    static const char* const default_backends[] = { "cpu" };
    VxBackendPolicy resolved = VX_BACKEND_POLICY_INIT;
    VxReport last_report = VX_REPORT_INIT;
    char outcomes[VX_REPORT_CANDIDATES_CAPACITY] = {0};
    double compile_started_ms = vx_report_now_ms();
    VxStatus status = VX_STATUS_BACKEND_UNAVAILABLE;
    int runtime_closed;

    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!model || !out_compiled || (policy && !vx_policy_valid(policy)))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_COMPILE,
                       NULL, "INVALID_POLICY", "backend policy is invalid");
    *out_compiled = NULL;
    if (policy) resolved = *policy;
    if (!resolved.backends && resolved.backend_count == 0u) {
        resolved.backends = default_backends;
        resolved.backend_count = 1u;
    }
    if (!vx_policy_valid(&resolved))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_COMPILE,
                       NULL, "INVALID_POLICY", "backend policy is invalid");

    pthread_mutex_lock(&model->runtime->mutex);
    runtime_closed = model->runtime->closed;
    pthread_mutex_unlock(&model->runtime->mutex);
    if (runtime_closed) {
        status = vx_fail(report, VX_STATUS_HANDLE_DISPOSED, VX_STAGE_COMPILE,
                         NULL, "HANDLE_DISPOSED", "runtime is closed");
        vx_report_set_lineage(report, NULL, model, NULL, NULL);
        return status;
    }

    for (size_t index = 0; index < resolved.backend_count; index++) {
        VxCompiledModel* candidate = NULL;
        VxReport candidate_report = VX_REPORT_INIT;
        status = vx_model_compile_candidate(model, &resolved,
                                            resolved.backends[index],
                                            &candidate, &candidate_report);
        vx_candidate_outcomes_append(outcomes, sizeof(outcomes),
                                     resolved.backends[index],
                                     vx_candidate_outcome(status,
                                                          &candidate_report));
        last_report = candidate_report;
        if (status == VX_STATUS_OK) {
            for (size_t skipped = index + 1u;
                 skipped < resolved.backend_count; skipped++)
                vx_candidate_outcomes_append(outcomes, sizeof(outcomes),
                                             resolved.backends[skipped],
                                             "not-tried");
            candidate_report.candidate_count =
                (uint32_t)resolved.backend_count;
            candidate_report.tier_fallback_used = index > 0u;
            snprintf(candidate_report.candidate_outcomes,
                     sizeof(candidate_report.candidate_outcomes), "%s",
                     outcomes);
            if (index > 0u) {
                snprintf(candidate_report.fallback_evidence,
                         sizeof(candidate_report.fallback_evidence),
                         "tier=%s->%s;selected-index=%zu;operator=%s",
                         resolved.backends[0], resolved.backends[index], index,
                         candidate_report.operator_fallback_used ?
                             "used" : "none");
            }
            candidate_report.compile_time_ms =
                vx_report_now_ms() - compile_started_ms;
            candidate->report = candidate_report;
            *out_compiled = candidate;
            vx_report_publish(report, &candidate_report);
            return VX_STATUS_OK;
        }
        if (resolved.mode == VX_BACKEND_REQUIRE ||
            status == VX_STATUS_OUT_OF_MEMORY ||
            status == VX_STATUS_INTERNAL) break;
    }

    last_report.candidate_count = (uint32_t)resolved.backend_count;
    last_report.tier_fallback_used = 0;
    snprintf(last_report.candidate_outcomes,
             sizeof(last_report.candidate_outcomes), "%s", outcomes);
    last_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
    last_report.policy_mode = resolved.mode;
    last_report.operator_fallback = resolved.operator_fallback;
    if (resolved.mode == VX_BACKEND_PREFER &&
        status == VX_STATUS_BACKEND_UNAVAILABLE) {
        snprintf(last_report.reason, sizeof(last_report.reason), "%s",
                 "BACKEND_UNAVAILABLE");
        snprintf(last_report.message, sizeof(last_report.message), "%s",
                 "no preferred backend candidate is available");
    }
    vx_report_set_lineage(&last_report, NULL, model, NULL, NULL);
    vx_report_publish(report, &last_report);
    return status;
}

void vx_compiled_model_retain(VxCompiledModel* compiled) {
    if (compiled)
        atomic_fetch_add_explicit(&compiled->references, 1, memory_order_relaxed);
}

void vx_compiled_model_release(VxCompiledModel* compiled) {
    if (!compiled || atomic_fetch_sub_explicit(&compiled->references, 1,
                                               memory_order_acq_rel) != 1) return;
    vx_compiled_destroy(compiled);
}

VxStatus vx_compiled_model_report(const VxCompiledModel* compiled,
                                  VxReport* report) {
    if (!compiled || !report || report->struct_size != sizeof(*report))
        return VX_STATUS_INVALID_ARGUMENT;
    *report = compiled->report;
    return VX_STATUS_OK;
}
