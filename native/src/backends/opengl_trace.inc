/* Desktop timer queries are opt-in. No driver lookup, query allocation or
 * clock read happens on the inactive route. Results follow the existing finish. */
struct OglTracePass {
    VxTraceScope* scope;
    size_t capacity, count;
    unsigned bits;
    GLuint* queries;
    VxDeviceTraceSpan* spans;
    void (*gen)(GLsizei, GLuint*);
    void (*destroy)(GLsizei, const GLuint*);
    void (*counter)(GLuint, GLenum);
    void (*get)(GLuint, GLenum, uint64_t*);
    void (*available)(GLuint, GLenum, GLint*);
    void (*dispatch)(GLuint, GLuint, GLuint);
    void (*use_program)(GLuint);
    const OglKernel* program;
    VxTraceQueue queue, previous_queue;
    uint64_t host_before, anchor_device, anchor_host, anchor_error;
};
#define VX_GL_TIMESTAMP 0x8E28
#define VX_GL_QUERY_RESULT 0x8866
#define VX_GL_QUERY_AVAILABLE 0x8867

static VxTraceQueue opengl_trace_queue(void) {
    return (VxTraceQueue){vx_trace_object_id(&g_opengl_device_state.trace_device_id),
        vx_trace_object_id(&g_opengl_device_state.trace_queue_id), 0};
}
static void opengl_trace_calibrate(OglTracePass* p) {
    uint64_t before = vx_trace_now_ns();
    GLint64 timestamp = 0;
    p_glGetInteger64v(VX_GL_TIMESTAMP, &timestamp);
    uint64_t after = vx_trace_now_ns();
    p->anchor_device = (uint64_t)timestamp;
    p->anchor_host = before + (after - before) / 2;
    p->anchor_error = (after - before + 1) / 2;
}
/* The one EGL context and its existing ownership mutex serialize these slots.
 * They are restored before that context is released; no hooks when disabled. */
static struct {
    VxTraceScope* scope;
    void (*finish)(void);
    void (*data)(GLenum, GLsizeiptr, const void*, GLenum);
    void (*upload)(GLenum, GLintptr, GLsizeiptr, const void*);
    void (*download)(GLenum, GLintptr, GLsizeiptr, void*);
    void (*copy)(GLenum, GLenum, GLintptr, GLintptr, GLsizeiptr);
    void (*dispatch)(GLuint, GLuint, GLuint);
} opengl_trace_io;
static void opengl_trace_submit(GLuint x, GLuint y, GLuint z) {
    uint64_t start = vx_trace_now_ns();
    opengl_trace_io.dispatch(x, y, z);
    VxTraceScope* scope = opengl_trace_io.scope;
    VxDeviceTraceSpan span = {.name = "glDispatchCompute", .index = -1,
        .phase = scope->work.phase, .activity = VX_TRACE_ACTIVITY_SUBMIT,
        .queue = scope->queue.queue_id ? scope->queue : opengl_trace_queue()};
    vx_trace_host_activity(scope, start, vx_trace_now_ns(), &span);
}
static int opengl_trace_copy_begin(const char* name, uint64_t bytes, VxMemorySpace source,
    VxMemorySpace destination, VxDeviceTraceSpan* span) {
    VxTraceScope* scope = opengl_trace_io.scope;
    *span = (VxDeviceTraceSpan){.host_start_ns = vx_trace_now_ns(), .name = name, .index = -1,
        .phase = scope->work.phase, .queue = scope->queue.queue_id ? scope->queue : opengl_trace_queue(),
        .activity = VX_TRACE_ACTIVITY_COPY, .copy_source = source, .copy_destination = destination, .copy_bytes = bytes};
    OpenGLContextState* state = opengl_context_state_get(0);
    OglTracePass* p = state ? state->trace_pass : NULL;
    if (!p || !vx_trace_device_nodes(p->scope)) return -1;
    if (p->count == p->capacity) { vx_trace_drop(p->scope); return -1; }
    int slot = (int)p->count++;
    p->spans[slot] = *span;
    p->counter(p->queries[slot * 2], VX_GL_TIMESTAMP);
    return slot;
}
static void opengl_trace_copy_end(int slot, const VxDeviceTraceSpan* span) {
    OpenGLContextState* state = opengl_context_state_get(0);
    OglTracePass* p = state ? state->trace_pass : NULL;
    if (slot >= 0 && p) p->counter(p->queries[slot * 2 + 1], VX_GL_TIMESTAMP);
    vx_trace_host_activity(opengl_trace_io.scope, span->host_start_ns, vx_trace_now_ns(), span);
}
static void opengl_trace_wait(void) {
    uint64_t start = vx_trace_now_ns();
    opengl_trace_io.finish();
    VxTraceScope* scope = opengl_trace_io.scope;
    VxDeviceTraceSpan span = {.name = "glFinish", .index = -1, .activity = VX_TRACE_ACTIVITY_WAIT,
        .queue = scope->queue.queue_id ? scope->queue : opengl_trace_queue()};
    vx_trace_host_activity(scope, start, vx_trace_now_ns(), &span);
}
static void opengl_trace_buffer_data(GLenum target, GLsizeiptr size, const void* data, GLenum usage) {
    if (!data || size <= 0) { opengl_trace_io.data(target, size, data, usage); return; }
    VxDeviceTraceSpan span;
    int slot = opengl_trace_copy_begin("glBufferData", (uint64_t)size, VX_MEMORY_SPACE_HOST, VX_MEMORY_SPACE_DEVICE, &span);
    opengl_trace_io.data(target, size, data, usage);
    opengl_trace_copy_end(slot, &span);
}
static void opengl_trace_upload(GLenum target, GLintptr offset, GLsizeiptr size, const void* data) {
    VxDeviceTraceSpan span;
    int slot = opengl_trace_copy_begin("glBufferSubData", size > 0 ? (uint64_t)size : 0, VX_MEMORY_SPACE_HOST, VX_MEMORY_SPACE_DEVICE, &span);
    opengl_trace_io.upload(target, offset, size, data);
    opengl_trace_copy_end(slot, &span);
}
static void opengl_trace_download(GLenum target, GLintptr offset, GLsizeiptr size, void* data) {
    VxDeviceTraceSpan span;
    int slot = opengl_trace_copy_begin("glGetBufferSubData", size > 0 ? (uint64_t)size : 0, VX_MEMORY_SPACE_DEVICE, VX_MEMORY_SPACE_HOST, &span);
    opengl_trace_io.download(target, offset, size, data);
    opengl_trace_copy_end(slot, &span);
}
static void opengl_trace_copy_buffer(GLenum read, GLenum write, GLintptr from, GLintptr to, GLsizeiptr size) {
    VxDeviceTraceSpan span;
    int slot = opengl_trace_copy_begin("glCopyBufferSubData", size > 0 ? (uint64_t)size : 0, VX_MEMORY_SPACE_DEVICE, VX_MEMORY_SPACE_DEVICE, &span);
    opengl_trace_io.copy(read, write, from, to, size);
    opengl_trace_copy_end(slot, &span);
}
static void opengl_trace_io_begin(void) {
    VxEngineState* engine = vx_engine_state_current();
    if (opengl_trace_io.scope || !engine || !engine->profiling || !vx_trace_nodes(engine->profiling) || !p_glFinish) return;
    opengl_trace_io.scope = engine->profiling;
    opengl_trace_io.finish = p_glFinish; p_glFinish = opengl_trace_wait;
    opengl_trace_io.data = p_glBufferData; p_glBufferData = opengl_trace_buffer_data;
    opengl_trace_io.upload = p_glBufferSubData; p_glBufferSubData = opengl_trace_upload;
    opengl_trace_io.download = p_glGetBufferSubData;
    if (p_glGetBufferSubData) p_glGetBufferSubData = opengl_trace_download;
    opengl_trace_io.copy = p_glCopyBufferSubData; p_glCopyBufferSubData = opengl_trace_copy_buffer;
    opengl_trace_io.dispatch = p_glDispatchCompute; p_glDispatchCompute = opengl_trace_submit;
}
static void opengl_trace_io_end(void) {
    if (!opengl_trace_io.scope) return;
    p_glFinish = opengl_trace_io.finish; p_glBufferData = opengl_trace_io.data;
    p_glBufferSubData = opengl_trace_io.upload; p_glGetBufferSubData = opengl_trace_io.download;
    p_glDispatchCompute = opengl_trace_io.dispatch;
    p_glCopyBufferSubData = opengl_trace_io.copy;
    memset(&opengl_trace_io, 0, sizeof(opengl_trace_io));
}

static void opengl_trace_program_bind(GLuint program) {
    OglTracePass* p = opengl_context_state_get(0)->trace_pass;
    p->program = NULL;
    for (size_t i = 0; i < g_opengl_device_state.program_cache_count; i++) {
        const OglProgramCacheEntry* entry = &g_opengl_device_state.program_cache[i];
        if (entry->program == program) { p->program = entry->descriptor; break; }
    }
    p->use_program(program);
}
static void opengl_trace_program_dispatch(GLuint x, GLuint y, GLuint z) {
    OglTracePass* p = opengl_context_state_get(0)->trace_pass;
    if (!p->program || p->count == p->capacity) {
        vx_trace_drop(p->scope); p->dispatch(x, y, z); return;
    }
    size_t slot = p->count++;
    p->spans[slot] = vx_trace_program_span(p->scope, p->program->name, "main");
    p->counter(p->queries[slot * 2], VX_GL_TIMESTAMP);
    p->dispatch(x, y, z);
    p->counter(p->queries[slot * 2 + 1], VX_GL_TIMESTAMP);
}
static void opengl_trace_begin(OpenGLContextState* state, const char* name) {
    VxTraceScope* scope = vx_engine_state_current()->profiling;
    if (!scope || !vx_trace_device_enabled(scope) || state->trace_pass) return;
#ifdef __ANDROID__
    /* GLES needs EXT_disjoint_timer_query and its disjoint-state contract. */
    (void)name;
    vx_trace_device_status(scope, 0, 0, 0, 0);
    return;
#else
    void (*get_bits)(GLenum, GLenum, GLint*) = load_gl_proc("glGetQueryiv");
    if (!get_bits) { vx_trace_device_status(scope, 0, 0, 0, 0); return; }
    GLint bits = 0;
    get_bits(VX_GL_TIMESTAMP, 0x8864 /* GL_QUERY_COUNTER_BITS */, &bits);
    if (bits <= 0 || bits > 64) { vx_trace_device_status(scope, 0, 0, 0, 0); return; }
    OglTracePass functions = {0};
    functions.gen = load_gl_proc("glGenQueries");
    functions.destroy = load_gl_proc("glDeleteQueries");
    functions.counter = load_gl_proc("glQueryCounter");
    functions.get = load_gl_proc("glGetQueryObjectui64v");
    functions.available = load_gl_proc("glGetQueryObjectiv");
    if (!functions.gen || !functions.destroy || !functions.counter || !functions.get || !functions.available) {
        vx_trace_device_status(scope, 0, 0, 0, 0); return;
    }
    vx_trace_device_status(scope, 1, 1, 0, 0);
    vx_trace_program_status(scope, 1);
    size_t capacity = vx_trace_device_capacity(scope, vx_trace_device_nodes(scope) ? 1024 : 1);
    if (!capacity) { vx_trace_drop(scope); return; }
    OglTracePass* p = calloc(1, sizeof(*p));
    if (!p) { vx_trace_device_fail(scope); return; }
    *p = functions;
    p->scope = scope; p->capacity = capacity; p->bits = (unsigned)bits;
    p->queue = opengl_trace_queue(); p->queue.submission_id = vx_trace_next_id();
    p->previous_queue = scope->queue;
    p->queries = calloc(capacity * 2, sizeof(*p->queries));
    p->spans = calloc(capacity, sizeof(*p->spans));
    if (!p->queries || !p->spans) {
        vx_trace_device_fail(scope); free(p->queries); free(p->spans); free(p); return;
    }
    p->gen((GLsizei)(capacity * 2), p->queries);
    if (!p->queries[capacity * 2 - 1]) {
        vx_trace_device_fail(scope); p->destroy((GLsizei)(capacity * 2), p->queries);
        free(p->queries); free(p->spans); free(p); return;
    }
    state->trace_pass = p;
    scope->queue = p->queue;
    p->host_before = vx_trace_now_ns();
    opengl_trace_calibrate(p);
    p->spans[0] = (VxDeviceTraceSpan){vx_trace_now_ns(), name, "", -1, 0};
    p->count = 1;
    p->counter(p->queries[0], VX_GL_TIMESTAMP);
    if (vx_trace_device_nodes(scope)) {
        p->dispatch = p_glDispatchCompute; p->use_program = p_glUseProgram;
        p_glDispatchCompute = opengl_trace_program_dispatch;
        p_glUseProgram = opengl_trace_program_bind;
    }
#endif
}
int opengl_trace_node_begin(int index, const char* name, const char* output, int fused) {
    OpenGLContextState* state = opengl_context_state_get(0);
    OglTracePass* p = state ? state->trace_pass : NULL;
    if (!p || !vx_trace_device_nodes(p->scope)) return -1;
    if (p->count == p->capacity) { vx_trace_drop(p->scope); return -1; }
    int token = (int)p->count++;
    p->spans[token] = (VxDeviceTraceSpan){vx_trace_now_ns(), name, output, index, fused, VX_TRACE_PHASE_FORWARD};
    p->counter(p->queries[token * 2], VX_GL_TIMESTAMP);
    return token;
}
void opengl_trace_node_end(int token) {
    if (token < 0) return;
    OglTracePass* p = opengl_context_state_get(0)->trace_pass;
    p->counter(p->queries[token * 2 + 1], VX_GL_TIMESTAMP);
}
static void opengl_trace_end_record(OpenGLContextState* state) {
    if (state->trace_pass) opengl_trace_node_end(0);
}
static void opengl_trace_finish(OpenGLContextState* state, int complete) {
    OglTracePass* p = state->trace_pass;
    if (!p) return;
    if (p->dispatch) {
        p_glDispatchCompute = p->dispatch; p_glUseProgram = p->use_program;
    }
    uint64_t host_end = vx_trace_now_ns();
    OglTracePass end_anchor = {0};
    opengl_trace_calibrate(&end_anchor);
    uint64_t first = 0, total = 0; int pass_valid = 0;
    VxTraceClock pass_clock = {0};
    for (size_t i = 0; i < p->count; i++) {
        GLint ready = 0;
        if (complete) p->available(p->queries[i * 2 + 1], VX_GL_QUERY_AVAILABLE, &ready);
        if (!ready) { vx_trace_device_fail(p->scope); continue; }
        uint64_t start = 0, end = 0;
        p->get(p->queries[i * 2], VX_GL_QUERY_RESULT, &start);
        p->get(p->queries[i * 2 + 1], VX_GL_QUERY_RESULT, &end);
        VxDeviceTraceSpan* s = &p->spans[i];
        uint64_t duration;
        if (host_end < s->host_start_ns || !vx_trace_timestamp_duration(start, end, p->bits,
                1.0, host_end - s->host_start_ns, &duration)) { vx_trace_device_fail(p->scope); continue; }
        if (i == 0) {
            first = start; total = duration; pass_valid = 1;
            VxTraceClock a = vx_trace_clock_calibrated(first, p->anchor_device, p->bits, 1.0,
                p->anchor_host, p->anchor_error, p->host_before, host_end, total);
            VxTraceClock b = vx_trace_clock_calibrated(first, end_anchor.anchor_device, p->bits, 1.0,
                end_anchor.anchor_host, end_anchor.anchor_error, p->host_before, host_end, total);
            if (a.method && b.method) {
                if (b.earliest_ns < a.earliest_ns) a.earliest_ns = b.earliest_ns;
                if (b.latest_ns > a.latest_ns) a.latest_ns = b.latest_ns;
                pass_clock = a;
            }
        }
        s->queue = p->queue;
        uint64_t offset;
        if (pass_valid && vx_trace_timestamp_duration(first, start, p->bits, 1.0,
                host_end - p->host_before, &offset)) {
            s->clock = vx_trace_clock_bounds(p->host_before, host_end, total, offset, duration);
            /* One placement window preserves device-measured relative order.
             * Clipping each child separately can make sequential nodes overlap. */
            if (s->clock.method && pass_clock.method) {
                s->clock = pass_clock;
                s->clock.earliest_ns += offset; s->clock.latest_ns += offset;
            }
        }
        vx_trace_device_span(p->scope, s, duration);
    }
    p->scope->queue = p->previous_queue;
    p->destroy((GLsizei)(p->capacity * 2), p->queries);
    free(p->queries); free(p->spans); free(p);
    state->trace_pass = NULL;
}
