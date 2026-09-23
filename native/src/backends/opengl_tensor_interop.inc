/* The process device owns the EGL context until module shutdown. Retained
 * buffers own no mutable engine state and can be read after session close. */
#define GL_COPY_READ_BUFFER 0x8F36
#define GL_COPY_WRITE_BUFFER 0x8F37
#define GL_ALL_BARRIER_BITS 0xFFFFFFFFu
static int ogl_tensor_device(VxNativeBuffer* view) {
    *view = (VxNativeBuffer){.kind = VX_NATIVE_BUFFER_OPENGL,
        .device_context = (uint64_t)(uintptr_t)egl_context};
    return egl_context != EGL_NO_CONTEXT && compute_capability.compute_supported && view->device_context;
}
static _Thread_local int ogl_tensor_batch_active;
static void ogl_tensor_end_access(void) {
    if (ogl_tensor_batch_active) return;
    OpenGLContextState* state = opengl_context_state_get(0);
    if (state && state->implicit_forward) {
        p_glFinish();
        state->forward_active = state->implicit_forward = 0;
        opengl_release_current_locked();
        opengl_device_unlock();
    }
}
static int ogl_tensor_batch_begin(int outputs) {
    (void)outputs;
    if (ogl_tensor_batch_active || !opengl_graph_access_ensure()) return 0;
    ogl_tensor_batch_active = 1;
    return 1;
}
static int ogl_tensor_batch_end(void) {
    if (!ogl_tensor_batch_active) return 1;
    ogl_tensor_batch_active = 0;
    int ok = p_glGetError() == GL_NO_ERROR;
    ogl_tensor_end_access();
    return ok;
}
static int ogl_tensor_copy(const void* host, const VxNativeBuffer* source) {
    if (!opengl_graph_access_ensure()) return 0;
    OglTensorSlot* slot = graph_get_slot(host, (size_t)source->length, 0);
    int ok = slot && slot_ensure_owned_buffer(slot, (size_t)source->length, NULL);
    if (ok) {
        opengl_clear_errors();
        p_glMemoryBarrier(GL_ALL_BARRIER_BITS);
        p_glBindBuffer(GL_COPY_READ_BUFFER, (GLuint)source->handle);
        p_glBindBuffer(GL_COPY_WRITE_BUFFER, slot->buffer);
        if ((GLuint)source->handle != slot->buffer || source->offset != 0)
            p_glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                (GLintptr)source->offset, 0, (GLsizeiptr)source->length);
        if (!ogl_tensor_batch_active) p_glFinish();
        ok = p_glGetError() == GL_NO_ERROR;
        p_glBindBuffer(GL_COPY_READ_BUFFER, 0);
        p_glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
        if (ok) { slot->host_dirty = 0; slot->device_dirty = 1; }
    }
    ogl_tensor_end_access();
    return ok;
}
static int ogl_tensor_snapshot(const void* host, size_t bytes, VxNativeStorage* storage) {
    if (!opengl_graph_access_ensure()) return 0;
    OglTensorSlot* slot = graph_ensure_device(host, bytes, 0);
    GLuint buffer = (GLuint)storage->buffer.handle;
    if (!buffer && slot) {
        buffer = create_buffer_target(GL_SHADER_STORAGE_BUFFER, storage->capacity, NULL);
        if (buffer) {
            g_opengl_device_state.reference_count++;
            storage->buffer.handle = buffer;
        }
    }
    int ok = slot && buffer != 0;
    if (ok) {
        opengl_clear_errors();
        p_glMemoryBarrier(GL_ALL_BARRIER_BITS);
        p_glBindBuffer(GL_COPY_READ_BUFFER, slot->buffer);
        p_glBindBuffer(GL_COPY_WRITE_BUFFER, buffer);
        p_glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, (GLsizeiptr)bytes);
        if (!ogl_tensor_batch_active) p_glFinish();
        ok = p_glGetError() == GL_NO_ERROR;
        p_glBindBuffer(GL_COPY_READ_BUFFER, 0);
        p_glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }
    ogl_tensor_end_access();
    return ok;
}
static int ogl_tensor_read(const VxNativeStorage* storage, void* host, size_t bytes) {
    opengl_device_lock();
    int ok = egl_context != EGL_NO_CONTEXT && opengl_make_current_locked() == 0;
    if (ok) {
        opengl_clear_errors();
        p_glMemoryBarrier(GL_ALL_BARRIER_BITS);
        p_glFinish();
        p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, (GLuint)storage->buffer.handle);
        ok = read_buffer(GL_SHADER_STORAGE_BUFFER, bytes, host) && p_glGetError() == GL_NO_ERROR;
        p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        opengl_release_current_locked();
    }
    opengl_device_unlock();
    return ok;
}
static void ogl_tensor_destroy(VxNativeStorage* storage) {
    if (!storage->buffer.handle) return;
    opengl_device_lock();
    if (egl_context != EGL_NO_CONTEXT && opengl_make_current_locked() == 0) {
        GLuint buffer = (GLuint)storage->buffer.handle;
        p_glFinish();
        opengl_driver_delete_buffers(1, &buffer);
        storage->buffer.handle = 0;
        opengl_release_current_locked();
    }
    g_opengl_device_state.reference_count--;
    opengl_device_unlock();
}
static int ogl_tensor_view(const void* host, size_t bytes, VxNativeBuffer* view) {
    if (!opengl_graph_access_ensure()) return 0;
    OglTensorSlot* slot = graph_ensure_device(host, bytes, 0);
    if (slot) { view->handle = slot->buffer; view->length = bytes; }
    ogl_tensor_end_access();
    return slot != NULL;
}
const VxNativeTensorOps vx_opengl_tensor_ops = {
    VX_NATIVE_BUFFER_OPENGL, ogl_tensor_device, NULL, ogl_tensor_copy,
    ogl_tensor_snapshot, ogl_tensor_read, ogl_tensor_destroy, ogl_tensor_batch_begin, ogl_tensor_batch_end,
    ogl_tensor_view
};
