/*
 * Native optimizer state, tensor updates, and optimizer checkpoint APIs.
 *
 * This fragment is included exactly once by engine.c behind
 * VOLVOXAI_ENABLE_TRAINING. It shares the engine translation unit so training
 * updates can use the private model/adapter locks without exposing them as a
 * broader internal ABI.
 */

static void volvoxai_engine_free_optimizer_states(void) {
    for (size_t i = 0; i < g_tensor_capacity; i++) {
#if VOLVOXAI_ENABLE_CUDA
        if (g_opt_states[i].m && g_opt_states[i].v)
            cuda_training_optimizer_forget(g_opt_states[i].m,
                                           g_opt_states[i].v);
#endif
        free(g_opt_states[i].m);
        free(g_opt_states[i].v);
        memset(&g_opt_states[i], 0, sizeof(g_opt_states[i]));
    }
}

static EngineOptimizerState* optimizer_state_for(const char* name, long numel) {
    EngineOptimizerState* empty = NULL;
    for (size_t i = 0; i < g_tensor_capacity; i++) {
        if (g_opt_states[i].name[0] && strcmp(g_opt_states[i].name, name) == 0) {
            EngineOptimizerState* state = &g_opt_states[i];
            if (state->numel != numel) {
#if VOLVOXAI_ENABLE_CUDA
                if (state->m && state->v)
                    cuda_training_optimizer_forget(state->m, state->v);
#endif
                free(state->m);
                free(state->v);
                state->m = NULL;
                state->v = NULL;
                state->numel = 0;
            }
            if (!state->m) state->m = (float*)calloc((size_t)numel, sizeof(float));
            if (!state->v) state->v = (float*)calloc((size_t)numel, sizeof(float));
            if (!state->m || !state->v) return NULL;
            state->numel = numel;
            return state;
        }
        if (!g_opt_states[i].name[0] && !empty) empty = &g_opt_states[i];
    }
    if (!empty) return NULL;
    strncpy(empty->name, name, sizeof(empty->name) - 1);
    empty->numel = numel;
    empty->m = (float*)calloc((size_t)numel, sizeof(float));
    empty->v = (float*)calloc((size_t)numel, sizeof(float));
    if (!empty->m || !empty->v) {
        free(empty->m);
        free(empty->v);
        memset(empty, 0, sizeof(*empty));
        return NULL;
    }
    return empty;
}

static void optimizer_state_remove(const char* name) {
    if (!name) return;
    for (size_t i = 0; i < g_tensor_capacity; i++) {
        if (g_opt_states[i].name[0] && !strcmp(g_opt_states[i].name, name)) {
#if VOLVOXAI_ENABLE_CUDA
            if (g_opt_states[i].m && g_opt_states[i].v)
                cuda_training_optimizer_forget(g_opt_states[i].m,
                                               g_opt_states[i].v);
#endif
            free(g_opt_states[i].m);
            free(g_opt_states[i].v);
            memset(&g_opt_states[i], 0, sizeof(g_opt_states[i]));
            return;
        }
    }
}

int volvoxai_engine_optimizer_state_buffers_f32(
        const char* name, long numel, float** first_moment,
        float** second_moment) {
    EngineOptimizerState* state;
    if (!name || !name[0] || numel <= 0 || !first_moment || !second_moment)
        return -1;
    state = optimizer_state_for(name, numel);
    if (!state) return -1;
    *first_moment = state->m;
    *second_moment = state->v;
    return 0;
}

static int optimizer_publish_updated_weight(const char* name, T* tensor,
                                             long numel,
                                             int host_authoritative) {
    int file_index;
    if (!name || !tensor || tensor->dtype != T_F32 ||
        tensor->numel != numel || !tensor->data) return -1;
    file_index = volvoxai_engine_tensor_weight_file_index(name);
    if (host_authoritative && file_index >= 0) {
        SafetensorsTensor* stored =
            safetensors_find_tensor_mutable(&g_weight_files[file_index], name);
        if (stored && stored->dtype == SAFETENSORS_DTYPE_F32 &&
            stored->nbytes == (size_t)numel * sizeof(float)) {
            if (safetensors_set_tensor_data(
                    &g_weight_files[file_index], name, tensor->data,
                    stored->nbytes) != 0) return -1;
        }
    }
    if (host_authoritative && vx_runtime_backend_has_graph())
        vx_runtime_backend_mark_host(
            tensor->data, (size_t)numel * sizeof(float), 1);
    vk_mark_owned_tensors_host_dirty();
    g_weight_caches_dirty = 1;
    /* Optimizer updates change values, never topology or tensor descriptors.
     * Invalidate numerical/decode caches, but retain the exact shape layouts
     * already proved for this private engine. Stale layouts stay stale. */
    uint64_t previous_generation = volvoxai_engine_model_generation_locked();
    volvoxai_engine_model_generation_advance_locked();
    VxEngineState* engine = vx_engine_state_current();
    uint64_t generation = volvoxai_engine_model_generation_locked();
    for (uint32_t i = 0; i < engine->dynamic_shape_plan_capacity; i++) {
        VxDynamicShapePlan* plan = &engine->dynamic_shape_plans[i];
        if (plan->signature && plan->model_generation == previous_generation)
            plan->model_generation = generation;
    }
    if (engine->dynamic_shape_maximum_layout.configured &&
        engine->dynamic_shape_maximum_layout.model_generation == previous_generation)
        engine->dynamic_shape_maximum_layout.model_generation = generation;
    return 0;
}

int volvoxai_engine_finalize_host_tensor_update_f32(const char* name, long numel) {
    T* tensor = t_find(name);
    if (g_weight_files_borrowed || !tensor || numel <= 0 ||
        !volvoxai_engine_tensor_is_model_weight_locked(name)) return -1;
    return optimizer_publish_updated_weight(name, tensor, numel, 1);
}

int volvoxai_engine_finalize_device_tensor_update_f32(const char* name,
                                                       long numel) {
    T* tensor;
    if (g_weight_files_borrowed || !name || !name[0] || numel <= 0 ||
        !volvoxai_engine_tensor_is_model_weight_locked(name)) return -1;
    tensor = t_find(name);
    return optimizer_publish_updated_weight(name, tensor, numel, 0);
}

static int volvoxai_engine_apply_tensor_update_f32_impl(
        const char* name, const float* update, long numel, int mode,
        float learning_rate, float beta1, float beta2, float epsilon,
        float weight_decay, float max_grad_norm, long step) {
    if (g_weight_files_borrowed || !name || !update || numel < 0) return -1;
    if (volvoxai_engine_tensor_is_removed_output(name)) return -1;
    if (volvoxai_engine_tensor_is_quantization_parameter_locked(name)) return -1;
    if (volvoxai_engine_merged_contains_tensor(name)) return -1;
    T* tensor = t_find(name);
    if (!tensor || tensor->dtype != T_F32 || tensor->numel != numel) return -1;
    if (tensor->is_graph_input ||
        ((mode == VOLVOXAI_TENSOR_UPDATE_SGD || mode == VOLVOXAI_TENSOR_UPDATE_ADAMW) &&
         !volvoxai_engine_tensor_is_model_weight_locked(name))) {
        return -1;
    }
    if (!isfinite(learning_rate) || learning_rate < 0.0f ||
        !isfinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f ||
        !isfinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        !isfinite(weight_decay) || weight_decay < 0.0f ||
        !isfinite(max_grad_norm) || max_grad_norm < 0.0f || step <= 0) return -1;
    for (long i = 0; i < numel; i++) if (!isfinite(update[i])) return -1;

    if (vk_sync_host_tensor(tensor) != 0) return -1;
    float* weight = tensor->data;
    float scale = 1.0f;
    if (max_grad_norm > 0.0f &&
        (mode == VOLVOXAI_TENSOR_UPDATE_SGD || mode == VOLVOXAI_TENSOR_UPDATE_ADAMW)) {
        double square_sum = 0.0;
        for (long i = 0; i < numel; i++) {
            square_sum += (double)update[i] * (double)update[i];
        }
        float norm = (float)sqrt(square_sum);
        if (norm > max_grad_norm) scale = max_grad_norm / (norm + 1.0e-12f);
    }

    if (mode == VOLVOXAI_TENSOR_UPDATE_ASSIGN) {
        memcpy(weight, update, (size_t)numel * sizeof(float));
    } else if (mode == VOLVOXAI_TENSOR_UPDATE_ADD) {
        for (long i = 0; i < numel; i++) weight[i] += update[i];
    } else if (mode == VOLVOXAI_TENSOR_UPDATE_SGD) {
        for (long i = 0; i < numel; i++) {
            float gradient = update[i] * scale;
            if (weight_decay != 0.0f) gradient += weight_decay * weight[i];
            weight[i] -= learning_rate * gradient;
        }
    } else if (mode == VOLVOXAI_TENSOR_UPDATE_ADAMW) {
        EngineOptimizerState* state = optimizer_state_for(name, numel);
        if (!state) return -1;
#if VOLVOXAI_ENABLE_CUDA
        /* Route selection may have changed since CUDA made these moments
           device-authoritative. The mirror lookup is a safe no-op for a
           CPU-only engine, and a CPU update must materialize and retire any
           mirror before modifying its host storage. */
        if (cuda_training_optimizer_materialize(
                state->m, state->v, (size_t)numel * sizeof(float)) != 0)
            return -1;
        /* The CPU update below makes the host moments authoritative again.
           Drop the old device mirror so a later CUDA step seeds from these
           newly updated values instead of reusing stale device state. */
        cuda_training_optimizer_forget(state->m, state->v);
#endif
        float bias_correction1 = 1.0f - powf(beta1, (float)step);
        float bias_correction2 = 1.0f - powf(beta2, (float)step);
        if (bias_correction1 <= 0.0f) bias_correction1 = 1.0f;
        if (bias_correction2 <= 0.0f) bias_correction2 = 1.0f;
        for (long i = 0; i < numel; i++) {
            float gradient = update[i] * scale;
            state->m[i] = beta1 * state->m[i] + (1.0f - beta1) * gradient;
            state->v[i] = beta2 * state->v[i] +
                (1.0f - beta2) * gradient * gradient;
            float corrected_m = state->m[i] / bias_correction1;
            float corrected_v = state->v[i] / bias_correction2;
            if (weight_decay != 0.0f) {
                weight[i] -= learning_rate * weight_decay * weight[i];
            }
            weight[i] -= learning_rate * corrected_m / (sqrtf(corrected_v) + epsilon);
        }
    } else {
        return -1;
    }

    return optimizer_publish_updated_weight(name, tensor, numel, 1);
}

int volvoxai_engine_apply_tensor_update_f32_locked(
        const char* name, const float* update, long numel, int mode,
        float learning_rate, float beta1, float beta2, float epsilon,
        float weight_decay, float max_grad_norm, long step) {
    return volvoxai_engine_apply_tensor_update_f32_impl(
        name, update, numel, mode, learning_rate, beta1, beta2, epsilon,
        weight_decay, max_grad_norm, step);
}

int volvoxai_engine_prepare_tensor_update_f32(const char* name, long numel, int mode) {
    if (g_weight_files_borrowed || !name || !name[0] || numel <= 0 ||
        (mode != 2 && mode != 3)) return -1;
    if (volvoxai_engine_tensor_is_removed_output(name)) return -1;
    if (volvoxai_engine_tensor_is_quantization_parameter_locked(name)) return -1;
    if (volvoxai_engine_merged_contains_tensor(name)) return -1;
    T* tensor = t_find(name);
    if (!tensor || tensor->dtype != T_F32 || tensor->numel != numel || !tensor->data) {
        return -1;
    }
    if (!volvoxai_engine_tensor_is_model_weight_locked(name)) return -1;
    if (mode == VOLVOXAI_TENSOR_UPDATE_ADAMW && !optimizer_state_for(name, numel)) return -1;
    return 0;
}

int volvoxai_engine_apply_tensor_update_f32(
        const char* name, const float* update, long numel, int mode,
        float learning_rate, float beta1, float beta2, float epsilon,
        float weight_decay, float max_grad_norm, long step) {
    if (g_engine_route_lease) return -1;
    volvoxai_engine_model_lock();
    if (volvoxai_engine_training_accumulation_pending()) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    volvoxai_engine_metadata_lock();
    int result = volvoxai_engine_apply_tensor_update_f32_impl(
        name, update, numel, mode, learning_rate, beta1, beta2, epsilon,
        weight_decay, max_grad_norm, step);
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    return result;
}

static int optimizer_metadata_long(cJSON* metadata, const char* key, long* out) {
    cJSON* item = metadata ? cJSON_GetObjectItem(metadata, key) : NULL;
    if (!cJSON_IsString(item) || !item->valuestring || !out) return -1;
    char* end = NULL;
    long value = strtol(item->valuestring, &end, 10);
    if (!end || *end) return -1;
    *out = value;
    return 0;
}

int volvoxai_engine_save_optimizer_state(const char* path, long training_step) {
    if (g_engine_route_lease || !path || !path[0] || training_step < 0) return -1;
    volvoxai_engine_model_lock();
    int result = -1;
    SafetensorsFile file;
    memset(&file, 0, sizeof(file));
    cJSON* metadata = NULL;
    char* metadata_text = NULL;
    if (volvoxai_engine_training_accumulation_pending() || !g_loaded ||
        safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0) goto done;
    metadata = cJSON_CreateObject();
    if (!metadata) goto done;
    cJSON_AddStringToObject(metadata, "volvox.optimizer_format", "volvox.optimizer.v1");
    char value[64];
    snprintf(value, sizeof(value), "%ld", training_step);
    cJSON_AddStringToObject(metadata, "volvox.training_step", value);

    int state_count = 0;
    for (size_t i = 0; i < g_tensor_capacity; i++) {
        EngineOptimizerState* state = &g_opt_states[i];
        if (!state->name[0] || state->numel <= 0 || !state->m || !state->v ||
            state->numel > INT_MAX) continue;
#if VOLVOXAI_ENABLE_CUDA
        /* A route switch does not make a dirty CUDA mirror host-authoritative.
           No-mirror CPU states take the side-effect-free fast path. */
        if (cuda_training_optimizer_materialize(
                state->m, state->v,
                (size_t)state->numel * sizeof(float)) != 0) goto done;
#endif
        char m_name[64], v_name[64], key[64];
        snprintf(m_name, sizeof(m_name), "state.%d.m", state_count);
        snprintf(v_name, sizeof(v_name), "state.%d.v", state_count);
        int shape[1] = {(int)state->numel};
        size_t nbytes = (size_t)state->numel * sizeof(float);
        if (safetensors_add_tensor(&file, m_name, SAFETENSORS_DTYPE_F32,
                                   shape, 1, state->m, nbytes) != 0 ||
            safetensors_add_tensor(&file, v_name, SAFETENSORS_DTYPE_F32,
                                   shape, 1, state->v, nbytes) != 0) goto done;
        snprintf(key, sizeof(key), "volvox.optimizer_state.%d.name", state_count);
        cJSON_AddStringToObject(metadata, key, state->name);
        state_count++;
    }
    snprintf(value, sizeof(value), "%d", state_count);
    cJSON_AddStringToObject(metadata, "volvox.optimizer_state_count", value);
    metadata_text = cJSON_PrintUnformatted(metadata);
    if (!metadata_text || safetensors_set_metadata_json(&file, metadata_text) != 0) goto done;
    result = safetensors_save(path, &file);
done:
    free(metadata_text);
    cJSON_Delete(metadata);
    safetensors_free(&file);
    volvoxai_engine_model_unlock();
    return result;
}

int volvoxai_engine_load_optimizer_state(const char* path, long* training_step) {
    if (g_engine_route_lease || !path || !path[0]) return -1;
    volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    int result = -1;
    SafetensorsFile file;
    memset(&file, 0, sizeof(file));
    cJSON* metadata = NULL;
    EngineOptimizerState* next = NULL;
    if (volvoxai_engine_training_accumulation_pending() || !g_loaded ||
        safetensors_load(path, &file) != 0 || !file.has_metadata ||
        !file.metadata_json) goto done;
    metadata = cJSON_Parse(file.metadata_json);
    cJSON* format = metadata ? cJSON_GetObjectItem(metadata, "volvox.optimizer_format") : NULL;
    long step = 0, state_count = 0;
    if (!cJSON_IsString(format) || !format->valuestring ||
        strcmp(format->valuestring, "volvox.optimizer.v1") ||
        optimizer_metadata_long(metadata, "volvox.training_step", &step) != 0 || step < 0 ||
        optimizer_metadata_long(metadata, "volvox.optimizer_state_count", &state_count) != 0 ||
        state_count < 0 || (uint64_t)state_count > g_tensor_capacity) goto done;
    if ((uint64_t)state_count * 2u != (uint64_t)file.tensor_count) goto done;
    next = (EngineOptimizerState*)calloc(
        g_tensor_capacity ? g_tensor_capacity : 1u, sizeof(*next));
    if (!next) goto done;
    for (long i = 0; i < state_count; i++) {
        char key[64], m_name[64], v_name[64];
        snprintf(key, sizeof(key), "volvox.optimizer_state.%ld.name", i);
        cJSON* name = cJSON_GetObjectItem(metadata, key);
        if (!cJSON_IsString(name) || !name->valuestring || !name->valuestring[0] ||
            strlen(name->valuestring) >= sizeof(next[i].name)) goto done;
        for (long prior = 0; prior < i; prior++) {
            if (!strcmp(next[prior].name, name->valuestring)) goto done;
        }
        T* target = t_find(name->valuestring);
        snprintf(m_name, sizeof(m_name), "state.%ld.m", i);
        snprintf(v_name, sizeof(v_name), "state.%ld.v", i);
        const SafetensorsTensor* m = safetensors_find_tensor(&file, m_name);
        const SafetensorsTensor* v = safetensors_find_tensor(&file, v_name);
        if (!target || !volvoxai_engine_tensor_is_model_weight_locked(name->valuestring) ||
            target->dtype != T_F32 || target->numel <= 0 ||
            !m || !v || m->dtype != SAFETENSORS_DTYPE_F32 ||
            v->dtype != SAFETENSORS_DTYPE_F32 || m->ndim != 1 || v->ndim != 1 ||
            m->shape[0] != target->numel || v->shape[0] != target->numel ||
            m->nbytes != v->nbytes ||
            m->nbytes != (size_t)target->numel * sizeof(float)) goto done;
        const float* m_values = (const float*)m->data;
        const float* v_values = (const float*)v->data;
        for (long value_index = 0; value_index < target->numel; value_index++) {
            if (!isfinite(m_values[value_index]) || !isfinite(v_values[value_index]) ||
                v_values[value_index] < 0.0f) goto done;
        }
        strncpy(next[i].name, name->valuestring, sizeof(next[i].name) - 1);
        next[i].numel = target->numel;
        next[i].m = (float*)malloc(m->nbytes);
        next[i].v = (float*)malloc(v->nbytes);
        if (!next[i].m || !next[i].v) goto done;
        memcpy(next[i].m, m->data, m->nbytes);
        memcpy(next[i].v, v->data, v->nbytes);
    }
    volvoxai_engine_free_optimizer_states();
    for (long i = 0; i < state_count; i++) {
        g_opt_states[i] = next[i];
        memset(&next[i], 0, sizeof(next[i]));
    }
    if (training_step) *training_step = step;
    result = 0;
done:
    if (next) {
        for (size_t i = 0; i < g_tensor_capacity; i++) {
            free(next[i].m);
            free(next[i].v);
        }
        free(next);
    }
    cJSON_Delete(metadata);
    safetensors_free(&file);
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    return result;
}
