/* OpenGL ES training: availability probe, slot tracking and synchronization.
 * The whole fragment sits inside #if VOLVOXAI_ENABLE_TRAINING. */

#if VOLVOXAI_ENABLE_TRAINING
static const OglTrainingKernel* training_find_kernel(const char* shader_name,
                                                     const char* entry_point) {
    if (!shader_name || !shader_name[0]) return NULL;
    const char* entry = entry_point && entry_point[0] ? entry_point : "main";
    for (size_t i = 0; i < sizeof(training_kernels) / sizeof(training_kernels[0]); i++) {
        if (!strcmp(training_kernels[i].shader_name, shader_name) &&
            !strcmp(training_kernels[i].entry_point, entry)) return &training_kernels[i];
    }
    return NULL;
}

static int training_find_slot(const void* host) {
    if (!host) return -1;
    for (int i = 0; i < training_slot_count; i++) {
        if (training_slots[i].host == host) return i;
    }
    return -1;
}

static OglTensorSlot* training_get_slot(void* host, size_t bytes, int is_weight) {
    int index = training_find_slot(host);
    if (index < 0) {
        if (training_slot_count >= OGL_GRAPH_MAX_TENSORS) return NULL;
        index = training_slot_count++;
        memset(&training_slots[index], 0, sizeof(training_slots[index]));
        training_slots[index].host = host;
        training_slots[index].host_dirty = 1;
    }
    OglTensorSlot* slot = &training_slots[index];
    slot->bytes = bytes;
    if (is_weight) slot->is_weight = 1;
    return slot;
}

static OglTensorSlot* training_prepare_slot(void* host, size_t bytes,
                                            unsigned char binding_access, int is_weight) {
    OglTensorSlot* slot = training_get_slot(host, bytes, is_weight);
    if (!slot) return NULL;
    int needs_read = (binding_access & OPENGL_TRAINING_ACCESS_READ) != 0;
    int graph_index = graph_find_slot(host);
    if (graph_index >= 0) {
        OglTensorSlot* graph_slot = graph_ensure_device(host, bytes, is_weight);
        if (!graph_slot || !graph_slot->buffer) return NULL;
        if (slot->owns_buffer && slot->buffer && slot->buffer != graph_slot->buffer) {
            p_glDeleteBuffers(1, &slot->buffer);
        }
        slot->buffer = graph_slot->buffer;
        slot->cap = graph_slot->cap;
        slot->bytes = bytes;
        slot->owns_buffer = 0;
        slot->host_dirty = graph_slot->host_dirty;
        slot->device_dirty = graph_slot->device_dirty;
        return slot;
    }
    int needs_allocation = !slot->owns_buffer || !slot->buffer || slot->cap < bytes;
    const void* initial = needs_read ? host : NULL;
    if (needs_allocation) {
        if (!slot_ensure_owned_buffer(slot, bytes, initial)) return NULL;
        slot->host_dirty = 0;
        slot->device_dirty = 0;
    } else if (needs_read && !slot->device_dirty && (!slot->is_weight || slot->host_dirty)) {
        p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, slot->buffer);
        p_glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)bytes, host, GL_DYNAMIC_DRAW);
        p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        slot->host_dirty = 0;
    }
    return slot;
}

static void training_reset_buffers(void) {
    if (p_glDeleteBuffers) {
        for (int i = 0; i < training_slot_count; i++) {
            if (training_slots[i].owns_buffer && training_slots[i].buffer) {
                p_glDeleteBuffers(1, &training_slots[i].buffer);
            }
        }
    }
    memset(training_slots, 0, sizeof(training_slots));
    training_slot_count = 0;
}

static int opengl_training_available_locked(void) {
    return ogl_ready() && max_ssbo_bindings >= 2 && max_uniform_bindings >= 1 &&
           max_compute_invocations >= 64;
}

static int opengl_training_supports_locked(
    const char* shader_name, const char* entry_point, const size_t* bytes,
    int binding_count, uint32_t groups_x, uint32_t groups_y,
    uint32_t groups_z) {
    const OglTrainingKernel* kernel = training_find_kernel(shader_name, entry_point);
    if (!opengl_training_available_locked() || !kernel || !bytes ||
        binding_count != kernel->kernel.binding_count ||
        groups_x == 0 || groups_y == 0 || groups_z == 0 ||
        groups_x > (uint32_t)max_compute_groups[0] ||
        groups_y > (uint32_t)max_compute_groups[1] ||
        groups_z > (uint32_t)max_compute_groups[2]) return 0;
    int uniform_blocks = kernel->kernel.uniform_binding >= 0 ? 1 : 0;
    int storage_blocks = binding_count - uniform_blocks;
    if (storage_blocks > max_compute_ssbo_blocks ||
        uniform_blocks > max_compute_uniform_blocks) return 0;
    for (int i = 0; i < kernel->kernel.binding_count; i++) {
        int limit = i == kernel->kernel.uniform_binding ? max_uniform_bindings : max_ssbo_bindings;
        GLint64 block_limit = i == kernel->kernel.uniform_binding
            ? max_uniform_block_size : max_ssbo_block_size;
        if (i >= limit || bytes[i] == 0 ||
            (uint64_t)bytes[i] > (uint64_t)block_limit) return 0;
    }
    return 1;
}

int opengl_training_available(void) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state) return 0;
    if (state->training_is_active || state->forward_active)
        return opengl_training_available_locked();
    opengl_device_lock();
    int available = opengl_training_available_locked();
    opengl_device_unlock();
    return available;
}

int opengl_training_supports(const char* shader_name, const char* entry_point,
                             const size_t* bytes, int binding_count,
                             uint32_t groups_x, uint32_t groups_y,
                             uint32_t groups_z) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state) return 0;
    if (state->training_is_active || state->forward_active) {
        return opengl_training_supports_locked(
            shader_name, entry_point, bytes, binding_count,
            groups_x, groups_y, groups_z);
    }
    opengl_device_lock();
    int supported = opengl_training_supports_locked(
        shader_name, entry_point, bytes, binding_count,
        groups_x, groups_y, groups_z);
    opengl_device_unlock();
    return supported;
}

int opengl_training_begin(void) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state || state->training_is_active) return -1;
    if (state->forward_active) {
        if (!state->implicit_forward) return -1;
        state->forward_active = 0;
        state->implicit_forward = 0;
        state->training_is_active = 1;
        return 0;
    }
    opengl_device_lock();
    if (!opengl_training_available_locked() ||
        opengl_make_current_locked() != 0) {
        opengl_device_unlock();
        return -1;
    }
    /* Programs and tensor buffers remain unallocated until the first dispatch. */
    state->training_is_active = 1;
    return 0;
}

int opengl_training_dispatch(const char* shader_name, const char* entry_point,
                             void* const* hosts, const size_t* bytes,
                             const unsigned char* access,
                             const unsigned char* is_weight, int binding_count,
                             uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state || !state->training_is_active || !hosts || !bytes || !access ||
        !is_weight ||
        binding_count > OGL_TRAINING_MAX_BINDINGS ||
        !opengl_training_supports_locked(shader_name, entry_point, bytes,
                                         binding_count, groups_x, groups_y,
                                         groups_z)) return -1;
    const OglTrainingKernel* selected = training_find_kernel(shader_name, entry_point);
    if (!selected || selected->kernel.binding_count != binding_count) return -1;

    for (int i = 0; i < binding_count; i++) {
        unsigned char expected = (selected->read_write_mask & (1u << i))
                                     ? OPENGL_TRAINING_ACCESS_READ_WRITE
                                     : OPENGL_TRAINING_ACCESS_READ;
        int limit = i == selected->kernel.uniform_binding
                        ? max_uniform_bindings : max_ssbo_bindings;
        if (!hosts[i] || bytes[i] == 0 || bytes[i] > (size_t)INTPTR_MAX ||
            access[i] != expected || is_weight[i] > 1 || i >= limit ||
            (is_weight[i] && (expected != OPENGL_TRAINING_ACCESS_READ ||
                              i == selected->kernel.uniform_binding))) return -1;
    }

    GLuint buffers[OGL_TRAINING_MAX_BINDINGS] = {0};
    OglTensorSlot* slots[OGL_TRAINING_MAX_BINDINGS] = {0};
    GLuint params = 0;
    for (int i = 0; i < binding_count; i++) {
        if (i == selected->kernel.uniform_binding) {
            params = params_buffer(hosts[i], bytes[i]);
            if (!params) return -1;
            buffers[i] = params;
        } else {
            slots[i] = training_prepare_slot(hosts[i], bytes[i], access[i], is_weight[i]);
            if (!slots[i]) {
                if (params) p_glDeleteBuffers(1, &params);
                return -1;
            }
            buffers[i] = slots[i]->buffer;
        }
    }

    int ok = dispatch_kernel(&selected->kernel, buffers, groups_x, groups_y, groups_z);
    if (params) p_glDeleteBuffers(1, &params);
    if (!ok) return -1;
    for (int i = 0; i < binding_count; i++) {
        if (slots[i] && (access[i] & OPENGL_TRAINING_ACCESS_WRITE)) {
            slots[i]->device_dirty = 1;
            slots[i]->host_dirty = 0;
            int graph_index = graph_find_slot(hosts[i]);
            if (graph_index >= 0 && graph_slots[graph_index].buffer == slots[i]->buffer) {
                graph_mark_device(&graph_slots[graph_index]);
            }
        }
    }
    return 0;
}

int opengl_training_sync(void* host, size_t bytes) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state || !state->training_is_active ||
        !opengl_training_available_locked() || !host ||
        bytes == 0) return -1;
    int index = training_find_slot(host);
    if (index < 0) return -1;
    OglTensorSlot* slot = &training_slots[index];
    if (!slot->buffer || bytes > slot->bytes) return -1;
    if (!slot->device_dirty) return 0;
    p_glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    p_glFinish();
    p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, slot->buffer);
    int ok = read_buffer(GL_SHADER_STORAGE_BUFFER, bytes, host);
    p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    if (!ok) return -1;
    slot->device_dirty = 0;
    slot->host_dirty = 0;
    int graph_index = graph_find_slot(host);
    if (graph_index >= 0 && graph_slots[graph_index].buffer == slot->buffer) {
        graph_slots[graph_index].device_dirty = 0;
        graph_slots[graph_index].host_dirty = 0;
    }
    return 0;
}

void opengl_training_end(void) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state || !state->training_is_active) return;
    if (ogl_ready() && p_glFinish) p_glFinish();
    training_reset_buffers();
    state->training_is_active = 0;
    opengl_release_current_locked();
    opengl_device_unlock();
}

#endif

void opengl_graph_reset(void) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state) return;
#if VOLVOXAI_ENABLE_TRAINING
    if (state->training_is_active) opengl_training_end();
#endif
    if (state->forward_active) (void)opengl_graph_end_forward();
    opengl_device_lock();
    int gl_current = state->device_acquired &&
        opengl_make_current_locked() == 0;
    if (gl_current && p_glDeleteBuffers) {
        for (int i = 0; i < state->graph_slots_count; i++) {
            if (state->graph_slot_storage[i].owns_buffer &&
                state->graph_slot_storage[i].buffer) {
                p_glDeleteBuffers(1, &state->graph_slot_storage[i].buffer);
            }
        }
    }
    memset(state->graph_slot_storage, 0, sizeof(state->graph_slot_storage));
    state->graph_slots_count = 0;
    free(state->shape_signature);
    state->shape_signature = NULL;
    state->shape_generation = opengl_generation_next(state->shape_generation);
    state->capacity_generation =
        opengl_generation_next(state->capacity_generation);
    state->domain_span_count = 0;
    state->domain_qgroupnorm_stats_bytes = 0;
    state->domain_qlayernorm_stats_bytes = 0;
    state->domain_enforced = 0;
    if (gl_current) opengl_release_current_locked();
    opengl_device_unlock();
}

int opengl_graph_bind_shape(const char* signature) {
    OpenGLContextState* state = opengl_context_state_get(0);
    char* candidate;
    size_t length;
    int ended_forward = 0;
    if (!state || !state->device_acquired || !signature || !signature[0] ||
        state->domain_enforced)
        return -1;
    if (state->shape_signature && !strcmp(state->shape_signature, signature))
        return 0;
#if VOLVOXAI_ENABLE_TRAINING
    if (state->training_is_active) return -1;
#endif
    length = strlen(signature);
    if (length > 1024u * 1024u) return -1;
    candidate = (char*)malloc(length + 1u);
    if (!candidate) return -1;
    memcpy(candidate, signature, length + 1u);
    if (state->forward_active) {
        if (opengl_graph_end_forward() != 0) {
            free(candidate);
            return -1;
        }
        ended_forward = 1;
    }
    (void)ended_forward;
    opengl_device_lock();
    if (!ogl_ready() || opengl_make_current_locked() != 0) {
        opengl_device_unlock();
        free(candidate);
        return -1;
    }
    if (p_glFinish) p_glFinish();
    uint64_t next_generation =
        opengl_generation_next(state->shape_generation);
    for (int index = 0; index < state->graph_slots_count; index++) {
        OglTensorSlot* slot = &state->graph_slot_storage[index];
        if (slot->is_weight) continue;
        slot->host = NULL;
        slot->bytes = 0;
        slot->shape_generation = next_generation;
        slot->host_dirty = 0;
        slot->device_dirty = 0;
        if (!slot->owns_buffer) {
            slot->buffer = 0;
            slot->cap = 0;
            slot->capacity_generation = 0;
        }
    }
    free(state->shape_signature);
    state->shape_signature = candidate;
    state->shape_generation = next_generation;
    opengl_release_current_locked();
    opengl_device_unlock();
    return 0;
}
