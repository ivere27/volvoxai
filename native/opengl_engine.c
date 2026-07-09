#include "opengl_engine.h"
#include "engine_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef unsigned int GLenum;
typedef unsigned int GLuint;
typedef int GLint;
typedef int GLsizei;
typedef char GLchar;
typedef unsigned int GLbitfield;
typedef intptr_t GLsizeiptr;
typedef intptr_t GLintptr;
typedef unsigned char GLubyte;
typedef unsigned char GLboolean;

#define GL_VENDOR 0x1F00
#define GL_RENDERER 0x1F01
#define GL_VERSION 0x1F02
#define GL_COMPUTE_SHADER 0x91B9
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#define GL_UNIFORM_BUFFER 0x8A11
#define GL_DYNAMIC_DRAW 0x88E8
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_INFO_LOG_LENGTH 0x8B84
#define GL_SHADER_STORAGE_BARRIER_BIT 0x2000
#define GL_BUFFER_UPDATE_BARRIER_BIT 0x0200
#define GL_MAP_READ_BIT 0x0001

typedef void* EGLDisplay;
typedef void* EGLContext;
typedef void* EGLSurface;
typedef void* EGLConfig;
typedef int EGLint;
typedef unsigned int EGLBoolean;

#define EGL_FALSE 0
#define EGL_TRUE 1
#define EGL_DEFAULT_DISPLAY ((void*)0)
#define EGL_NO_DISPLAY ((EGLDisplay)0)
#define EGL_NO_CONTEXT ((EGLContext)0)
#define EGL_NO_SURFACE ((EGLSurface)0)
#define EGL_NONE 0x3038
#define EGL_OPENGL_API 0x30A2
#define EGL_OPENGL_ES_API 0x30A0
#define EGL_OPENGL_BIT 0x0008
#define EGL_OPENGL_ES3_BIT 0x00000040
#define EGL_RENDERABLE_TYPE 0x3040
#define EGL_SURFACE_TYPE 0x3033
#define EGL_PBUFFER_BIT 0x0001
#define EGL_RED_SIZE 0x3024
#define EGL_GREEN_SIZE 0x3023
#define EGL_BLUE_SIZE 0x3022
#define EGL_ALPHA_SIZE 0x3021
#define EGL_WIDTH 0x3057
#define EGL_HEIGHT 0x3056
#define EGL_CONTEXT_MAJOR_VERSION 0x3098
#define EGL_CONTEXT_MINOR_VERSION 0x30FB
#define EGL_CONTEXT_CLIENT_VERSION 0x3098
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD

static void* gl_lib = NULL;
static void* egl_lib = NULL;
static EGLDisplay egl_display = EGL_NO_DISPLAY;
static EGLContext egl_context = EGL_NO_CONTEXT;
static EGLSurface egl_surface = EGL_NO_SURFACE;

static EGLDisplay (*p_eglGetDisplay)(void*) = NULL;
static EGLDisplay (*p_eglGetPlatformDisplayEXT)(EGLint, void*, const EGLint*) = NULL;
static EGLBoolean (*p_eglInitialize)(EGLDisplay, EGLint*, EGLint*) = NULL;
static EGLBoolean (*p_eglBindAPI)(EGLint) = NULL;
static EGLBoolean (*p_eglChooseConfig)(EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*) = NULL;
static EGLSurface (*p_eglCreatePbufferSurface)(EGLDisplay, EGLConfig, const EGLint*) = NULL;
static EGLContext (*p_eglCreateContext)(EGLDisplay, EGLConfig, EGLContext, const EGLint*) = NULL;
static EGLBoolean (*p_eglMakeCurrent)(EGLDisplay, EGLSurface, EGLSurface, EGLContext) = NULL;
static EGLBoolean (*p_eglDestroyContext)(EGLDisplay, EGLContext) = NULL;
static EGLBoolean (*p_eglDestroySurface)(EGLDisplay, EGLSurface) = NULL;
static EGLBoolean (*p_eglTerminate)(EGLDisplay) = NULL;
static void* (*p_eglGetProcAddress)(const char*) = NULL;

static const GLubyte* (*p_glGetString)(GLenum) = NULL;
static GLuint (*p_glCreateShader)(GLenum) = NULL;
static void (*p_glShaderSource)(GLuint, GLsizei, const GLchar* const*, const GLint*) = NULL;
static void (*p_glCompileShader)(GLuint) = NULL;
static void (*p_glGetShaderiv)(GLuint, GLenum, GLint*) = NULL;
static void (*p_glGetShaderInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*) = NULL;
static void (*p_glDeleteShader)(GLuint) = NULL;
static GLuint (*p_glCreateProgram)(void) = NULL;
static void (*p_glAttachShader)(GLuint, GLuint) = NULL;
static void (*p_glLinkProgram)(GLuint) = NULL;
static void (*p_glGetProgramiv)(GLuint, GLenum, GLint*) = NULL;
static void (*p_glGetProgramInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*) = NULL;
static void (*p_glDeleteProgram)(GLuint) = NULL;
static void (*p_glUseProgram)(GLuint) = NULL;
static void (*p_glGenBuffers)(GLsizei, GLuint*) = NULL;
static void (*p_glBindBuffer)(GLenum, GLuint) = NULL;
static void (*p_glBufferData)(GLenum, GLsizeiptr, const void*, GLenum) = NULL;
static void (*p_glBindBufferBase)(GLenum, GLuint, GLuint) = NULL;
static void (*p_glDeleteBuffers)(GLsizei, const GLuint*) = NULL;
static void (*p_glDispatchCompute)(GLuint, GLuint, GLuint) = NULL;
static void (*p_glMemoryBarrier)(GLbitfield) = NULL;
static void (*p_glFinish)(void) = NULL;
static void (*p_glGetBufferSubData)(GLenum, GLintptr, GLsizeiptr, void*) = NULL;
static void* (*p_glMapBufferRange)(GLenum, GLintptr, GLsizeiptr, GLbitfield) = NULL;
static GLboolean (*p_glUnmapBuffer)(GLenum) = NULL;

typedef struct {
    const char* name;
    const char* path;
    GLuint program;
    int binding_count;
    int uniform_binding;
    int ready;
    int failed;
} OglKernel;

#ifdef __ANDROID__
#define OGL_SHADER_DIR "native/shaders/gles"
#else
#define OGL_SHADER_DIR "native/shaders/glsl"
#endif

#define OGL_GRAPH_MAX_TENSORS 4096

typedef struct {
    const void* host;
    size_t bytes;
    size_t cap;
    GLuint buffer;
    int host_dirty;
    int device_dirty;
    int is_weight;
    int owns_buffer;
} OglTensorSlot;

static OglTensorSlot graph_slots[OGL_GRAPH_MAX_TENSORS];
static int graph_slot_count = 0;

static OglKernel k_copy = {"copy", OGL_SHADER_DIR "/copy.comp", 0, 3, 2, 0};
static OglKernel k_add = {"addRelu", OGL_SHADER_DIR "/addRelu.comp", 0, 4, 3, 0};
static OglKernel k_add3 = {"add3Relu", OGL_SHADER_DIR "/add3Relu.comp", 0, 5, 4, 0};
static OglKernel k_clip = {"clip", OGL_SHADER_DIR "/clip.comp", 0, 3, 2, 0};
static OglKernel k_sigmoid = {"sigmoid", OGL_SHADER_DIR "/sigmoid.comp", 0, 3, 2, 0};
static OglKernel k_relu = {"reLU", OGL_SHADER_DIR "/reLU.comp", 0, 3, 2, 0};
static OglKernel k_gelu = {"gELU", OGL_SHADER_DIR "/gELU.comp", 0, 3, 2, 0};
static OglKernel k_silu = {"siLU", OGL_SHADER_DIR "/siLU.comp", 0, 3, 2, 0};
static OglKernel k_tanh = {"tanh", OGL_SHADER_DIR "/tanh.comp", 0, 3, 2, 0};
static OglKernel k_hardswish = {"hardSwish", OGL_SHADER_DIR "/hardSwish.comp", 0, 3, 2, 0};
static OglKernel k_hardsigmoid = {"hardSigmoid", OGL_SHADER_DIR "/hardSigmoid.comp", 0, 3, 2, 0};
static OglKernel k_leaky_relu = {"leakyReLU", OGL_SHADER_DIR "/leakyReLU.comp", 0, 3, 2, 0};
static OglKernel k_prelu = {"pReLU", OGL_SHADER_DIR "/pReLU.comp", 0, 4, 3, 0};
static OglKernel k_layernorm = {"layerNorm", OGL_SHADER_DIR "/layerNorm.comp", 0, 5, 4, 0};
static OglKernel k_rmsnorm = {"rMSNorm", OGL_SHADER_DIR "/rMSNorm.comp", 0, 4, 3, 0};
static OglKernel k_softmax = {"softmax", OGL_SHADER_DIR "/softmax.comp", 0, 3, 2, 0};
static OglKernel k_logsoftmax = {"logSoftmax", OGL_SHADER_DIR "/logSoftmax.comp", 0, 3, 2, 0};
static OglKernel k_reduce = {"reduce", OGL_SHADER_DIR "/reduce.comp", 0, 3, 2, 0};
static OglKernel k_globalavg = {"globalAveragePool", OGL_SHADER_DIR "/globalAveragePool.comp", 0, 3, 2, 0};
static OglKernel k_avgpool = {"averagePool2D", OGL_SHADER_DIR "/averagePool2D.comp", 0, 3, 2, 0};
static OglKernel k_batchnorm = {"batchNorm2D", OGL_SHADER_DIR "/batchNorm2D.comp", 0, 7, 6, 0};
static OglKernel k_embedding = {"embedding", OGL_SHADER_DIR "/embedding.comp", 0, 4, 3, 0};
static OglKernel k_transpose = {"generalTranspose", OGL_SHADER_DIR "/generalTranspose.comp", 0, 3, -1, 0};
static OglKernel k_where = {"where", OGL_SHADER_DIR "/where.comp", 0, 5, 4, 0};
static OglKernel k_expand = {"expand", OGL_SHADER_DIR "/expand.comp", 0, 3, 2, 0};
static OglKernel k_pad = {"pad", OGL_SHADER_DIR "/pad.comp", 0, 3, 2, 0};
static OglKernel k_slice = {"slice", OGL_SHADER_DIR "/slice.comp", 0, 3, 2, 0};
static OglKernel k_gather = {"gather", OGL_SHADER_DIR "/gather.comp", 0, 4, 3, 0};
static OglKernel k_convtranspose = {"convTranspose2D", OGL_SHADER_DIR "/convTranspose2D.comp", 0, 5, 4, 0};
static OglKernel k_interp1d = {"interp1D", OGL_SHADER_DIR "/interp1D.comp", 0, 3, 2, 0};
static OglKernel k_mul = {"mul", OGL_SHADER_DIR "/mul.comp", 0, 4, 3, 0};
static OglKernel k_sub = {"sub", OGL_SHADER_DIR "/sub.comp", 0, 4, 3, 0};
static OglKernel k_div = {"div", OGL_SHADER_DIR "/div.comp", 0, 4, 3, 0};
static OglKernel k_split = {"split", OGL_SHADER_DIR "/split.comp", 0, 3, 2, 0};
static OglKernel k_conv1d = {"conv1D", OGL_SHADER_DIR "/conv1D.comp", 0, 5, 4, 0};
static OglKernel k_sdpa = {"sDPA", OGL_SHADER_DIR "/sDPA.comp", 0, 3, 2, 0};
static OglKernel k_cross_sdpa = {"crossSDPA", OGL_SHADER_DIR "/crossSDPA.comp", 0, 5, 4, 0};
static OglKernel k_cross_attention = {"crossAttentionF32", OGL_SHADER_DIR "/crossAttentionF32.comp", 0, 7, 6, 0};
static OglKernel k_quantize = {"quantizeLinear", OGL_SHADER_DIR "/quantizeLinear.comp", 0, 3, 2, 0};
static OglKernel k_dequantize = {"dequantizeLinear", OGL_SHADER_DIR "/dequantizeLinear.comp", 0, 5, 4, 0};
static OglKernel k_spatial_softargmax_y = {"spatialSoftargmaxY", OGL_SHADER_DIR "/spatialSoftargmaxY.comp", 0, 3, 2, 0};
static OglKernel k_profile_x = {"profileX", OGL_SHADER_DIR "/profileX.comp", 0, 3, 2, 0};
static OglKernel k_profile_y = {"profileY", OGL_SHADER_DIR "/profileY.comp", 0, 3, 2, 0};
static OglKernel k_mean_height = {"meanHeight", OGL_SHADER_DIR "/meanHeight.comp", 0, 3, 2, 0};
static OglKernel k_nms = {"nonMaxSuppression", OGL_SHADER_DIR "/nonMaxSuppression.comp", 0, 4, 3, 0};
static OglKernel k_concat = {"concatCopy", OGL_SHADER_DIR "/concatCopy.comp", 0, 3, 2, 0};
static OglKernel k_concat_sigmoid = {"concatSigmoidCopy", OGL_SHADER_DIR "/concatSigmoidCopy.comp", 0, 3, 2, 0};
static OglKernel k_upsample = {"upsample2x", OGL_SHADER_DIR "/upsample2x.comp", 0, 3, 2, 0};
static OglKernel k_resize = {"resize", OGL_SHADER_DIR "/resize.comp", 0, 3, 2, 0};
static OglKernel k_maxpool = {"maxPool2D", OGL_SHADER_DIR "/maxPool2D.comp", 0, 3, 2, 0};
static OglKernel k_conv2d = {"conv2D", OGL_SHADER_DIR "/conv2D.comp", 0, 5, 4, 0};
static OglKernel k_conv2d_c3out16 = {"conv2DRegularC3Out16", OGL_SHADER_DIR "/conv2DRegularC3Out16.comp", 0, 5, 4, 0};
static OglKernel k_conv2d_dw4 = {"conv2DDepthwise4", OGL_SHADER_DIR "/conv2DDepthwise4.comp", 0, 5, 4, 0};
static OglKernel k_conv2d_dw8 = {"conv2DDepthwise8", OGL_SHADER_DIR "/conv2DDepthwise8.comp", 0, 5, 4, 0};
static OglKernel k_conv2d_pw8 = {"conv2DPointwise8", OGL_SHADER_DIR "/conv2DPointwise8.comp", 0, 5, 4, 0};
static OglKernel k_conv2d_pw8v2 = {"conv2DPointwise8Vec2", OGL_SHADER_DIR "/conv2DPointwise8Vec2.comp", 0, 5, 4, 0};
static OglKernel k_conv2d_pw8v4 = {"conv2DPointwise8Vec4", OGL_SHADER_DIR "/conv2DPointwise8Vec4.comp", 0, 5, 4, 0};
static OglKernel k_conv2d_pw16 = {"conv2DPointwise16", OGL_SHADER_DIR "/conv2DPointwise16.comp", 0, 5, 4, 0};
static OglKernel k_conv2d_pw16tile = {"conv2DPointwise16Tile", OGL_SHADER_DIR "/conv2DPointwise16Tile.comp", 0, 5, 4, 0};
static OglKernel k_matmul = {"linearF32RowMajor", OGL_SHADER_DIR "/linearF32RowMajor.comp", 0, 5, 4, 0};

static void* load_symbol(void* lib, const char* name) {
#ifdef _WIN32
    return lib ? (void*)GetProcAddress((HMODULE)lib, name) : NULL;
#else
    return lib ? dlsym(lib, name) : NULL;
#endif
}

static void* load_gl_proc(const char* name) {
    void* p = p_eglGetProcAddress ? p_eglGetProcAddress(name) : NULL;
    if (!p) p = load_symbol(gl_lib, name);
    return p;
}

#define LOAD_EGL(name) do { p_##name = load_symbol(egl_lib, #name); if (!p_##name) return -1; } while (0)
#define LOAD_GL(name) do { p_##name = load_gl_proc(#name); if (!p_##name) return -1; } while (0)

static int load_egl(void) {
#ifdef _WIN32
    egl_lib = LoadLibraryA("libEGL.dll");
    gl_lib = LoadLibraryA("opengl32.dll");
#else
#ifdef __ANDROID__
    egl_lib = dlopen("libEGL.so", RTLD_NOW | RTLD_LOCAL);
    gl_lib = dlopen("libGLESv3.so", RTLD_NOW | RTLD_LOCAL);
    if (!gl_lib) gl_lib = dlopen("libGLESv2.so", RTLD_NOW | RTLD_LOCAL);
#else
    egl_lib = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!egl_lib) egl_lib = dlopen("libEGL.so", RTLD_NOW | RTLD_LOCAL);
    gl_lib = dlopen("libGL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!gl_lib) gl_lib = dlopen("libGL.so", RTLD_NOW | RTLD_LOCAL);
#endif
#endif
    if (!egl_lib || !gl_lib) return -1;
    LOAD_EGL(eglGetDisplay);
    p_eglGetPlatformDisplayEXT = load_symbol(egl_lib, "eglGetPlatformDisplayEXT");
    LOAD_EGL(eglInitialize);
    LOAD_EGL(eglBindAPI);
    LOAD_EGL(eglChooseConfig);
    LOAD_EGL(eglCreatePbufferSurface);
    LOAD_EGL(eglCreateContext);
    LOAD_EGL(eglMakeCurrent);
    LOAD_EGL(eglDestroyContext);
    LOAD_EGL(eglDestroySurface);
    LOAD_EGL(eglTerminate);
    LOAD_EGL(eglGetProcAddress);
    return 0;
}

static int load_gl(void) {
    LOAD_GL(glGetString);
    LOAD_GL(glCreateShader);
    LOAD_GL(glShaderSource);
    LOAD_GL(glCompileShader);
    LOAD_GL(glGetShaderiv);
    LOAD_GL(glGetShaderInfoLog);
    LOAD_GL(glDeleteShader);
    LOAD_GL(glCreateProgram);
    LOAD_GL(glAttachShader);
    LOAD_GL(glLinkProgram);
    LOAD_GL(glGetProgramiv);
    LOAD_GL(glGetProgramInfoLog);
    LOAD_GL(glDeleteProgram);
    LOAD_GL(glUseProgram);
    LOAD_GL(glGenBuffers);
    LOAD_GL(glBindBuffer);
    LOAD_GL(glBufferData);
    LOAD_GL(glBindBufferBase);
    LOAD_GL(glDeleteBuffers);
    LOAD_GL(glDispatchCompute);
    LOAD_GL(glMemoryBarrier);
    LOAD_GL(glFinish);
    p_glGetBufferSubData = load_gl_proc("glGetBufferSubData");
    p_glMapBufferRange = load_gl_proc("glMapBufferRange");
    p_glUnmapBuffer = load_gl_proc("glUnmapBuffer");
    if (!p_glGetBufferSubData && (!p_glMapBufferRange || !p_glUnmapBuffer)) return -1;
    return 0;
}

static int create_context(void) {
    egl_display = EGL_NO_DISPLAY;
#ifndef __ANDROID__
    if (p_eglGetPlatformDisplayEXT) {
        egl_display = p_eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
        EGLint maj = 0, min = 0;
        if (egl_display != EGL_NO_DISPLAY && p_eglInitialize(egl_display, &maj, &min) == EGL_TRUE) goto initialized;
    }
#endif
    egl_display = p_eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (egl_display == EGL_NO_DISPLAY) return -1;
    EGLint major = 0, minor = 0;
    if (p_eglInitialize(egl_display, &major, &minor) != EGL_TRUE) return -1;

initialized:
#ifdef __ANDROID__
    if (p_eglBindAPI(EGL_OPENGL_ES_API) != EGL_TRUE) return -1;
    EGLint cfg_attrs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_NONE
    };
#else
    if (p_eglBindAPI(EGL_OPENGL_API) != EGL_TRUE) return -1;
    EGLint cfg_attrs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_NONE
    };
#endif
    EGLConfig cfg = NULL;
    EGLint ncfg = 0;
    if (p_eglChooseConfig(egl_display, cfg_attrs, &cfg, 1, &ncfg) != EGL_TRUE || ncfg <= 0) return -1;
    EGLint surf_attrs[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
    egl_surface = p_eglCreatePbufferSurface(egl_display, cfg, surf_attrs);
    if (egl_surface == EGL_NO_SURFACE) return -1;
#ifdef __ANDROID__
    EGLint ctx_attrs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
#else
    EGLint ctx_attrs[] = { EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 3, EGL_NONE };
#endif
    egl_context = p_eglCreateContext(egl_display, cfg, EGL_NO_CONTEXT, ctx_attrs);
    if (egl_context == EGL_NO_CONTEXT) return -1;
    if (p_eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context) != EGL_TRUE) return -1;
    return 0;
}

int opengl_init(void) {
    if (egl_context != EGL_NO_CONTEXT) return 0;
    if (load_egl() != 0 || create_context() != 0 || load_gl() != 0) {
        printf("[VolvoxAI GPU] Failed to initialize OpenGL compute backend.\n");
        opengl_cleanup();
        return -1;
    }
    const GLubyte* vendor = p_glGetString(GL_VENDOR);
    const GLubyte* renderer = p_glGetString(GL_RENDERER);
    const GLubyte* version = p_glGetString(GL_VERSION);
    printf("[VolvoxAI GPU] OpenGL Compute initialized: %s / %s / %s\n",
           vendor ? (const char*)vendor : "unknown",
           renderer ? (const char*)renderer : "unknown",
           version ? (const char*)version : "unknown");
    return 0;
}

static int ogl_ready(void) {
    return egl_context != EGL_NO_CONTEXT && p_glDispatchCompute != NULL;
}

static char* read_text_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long len = ftell(f);
    if (len < 0) { fclose(f); return NULL; }
    rewind(f);
    char* buf = (char*)malloc((size_t)len + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[n] = 0;
    return buf;
}

static GLuint compile_kernel(OglKernel* k) {
    if (!ogl_ready() || !k) return 0;
    if (k->ready) return k->program;
    if (k->failed) return 0;
    char* source = read_text_file(k->path);
    if (!source) {
        fprintf(stderr, "[OpenGL] failed to read generated shader %s\n", k->path);
        k->failed = 1;
        return 0;
    }
    GLuint sh = p_glCreateShader(GL_COMPUTE_SHADER);
    const GLchar* src = (const GLchar*)source;
    p_glShaderSource(sh, 1, &src, NULL);
    p_glCompileShader(sh);
    free(source);
    GLint ok = 0;
    p_glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        GLsizei n = 0;
        p_glGetShaderInfoLog(sh, (GLsizei)sizeof(log), &n, log);
        fprintf(stderr, "[OpenGL] shader compile failed (%s): %s\n", k->name, log);
        p_glDeleteShader(sh);
        k->failed = 1;
        return 0;
    }
    GLuint prog = p_glCreateProgram();
    p_glAttachShader(prog, sh);
    p_glLinkProgram(prog);
    p_glDeleteShader(sh);
    p_glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        GLsizei n = 0;
        p_glGetProgramInfoLog(prog, (GLsizei)sizeof(log), &n, log);
        fprintf(stderr, "[OpenGL] program link failed (%s): %s\n", k->name, log);
        p_glDeleteProgram(prog);
        k->failed = 1;
        return 0;
    }
    k->program = prog;
    k->ready = 1;
    return prog;
}

static GLuint create_buffer_target(GLenum target, size_t bytes, const void* data) {
    if (!ogl_ready() || bytes == 0) return 0;
    GLuint b = 0;
    p_glGenBuffers(1, &b);
    if (!b) return 0;
    p_glBindBuffer(target, b);
    p_glBufferData(target, (GLsizeiptr)bytes, data, GL_DYNAMIC_DRAW);
    p_glBindBuffer(target, 0);
    return b;
}

static GLuint create_buffer(size_t bytes, const void* data) {
    return create_buffer_target(GL_SHADER_STORAGE_BUFFER, bytes, data);
}

static int graph_find_slot(const void* host) {
    if (!host) return -1;
    for (int i = 0; i < graph_slot_count; i++) {
        if (graph_slots[i].host == host) return i;
    }
    return -1;
}

static OglTensorSlot* graph_get_slot(const void* host, size_t bytes, int is_weight) {
    if (!ogl_ready() || !host || bytes == 0) return NULL;
    int idx = graph_find_slot(host);
    if (idx < 0) {
        if (graph_slot_count >= OGL_GRAPH_MAX_TENSORS) return NULL;
        idx = graph_slot_count++;
        memset(&graph_slots[idx], 0, sizeof(graph_slots[idx]));
        graph_slots[idx].host = host;
        graph_slots[idx].host_dirty = 1;
        graph_slots[idx].is_weight = is_weight;
    }
    OglTensorSlot* s = &graph_slots[idx];
    s->bytes = bytes;
    if (is_weight) s->is_weight = 1;
    return s;
}

static int slot_ensure_owned_buffer(OglTensorSlot* s, size_t bytes, const void* data) {
    if (!s || bytes == 0) return 0;
    if (!s->owns_buffer || s->buffer == 0 || s->cap < bytes) {
        if (s->owns_buffer && s->buffer) p_glDeleteBuffers(1, &s->buffer);
        s->buffer = create_buffer(bytes, data);
        s->cap = bytes;
        s->owns_buffer = 1;
        return s->buffer != 0;
    }
    if (data) {
        p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, s->buffer);
        p_glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)bytes, data, GL_DYNAMIC_DRAW);
        p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
    return 1;
}

static OglTensorSlot* graph_ensure_device(const void* host, size_t bytes, int is_weight) {
    OglTensorSlot* s = graph_get_slot(host, bytes, is_weight);
    if (!s) return NULL;
    if (s->host_dirty || s->buffer == 0 || s->cap < bytes) {
        if (!slot_ensure_owned_buffer(s, bytes, host)) return NULL;
        s->host_dirty = 0;
        s->device_dirty = 0;
    }
    return s;
}

static OglTensorSlot* graph_output_slot(const void* host, size_t bytes) {
    OglTensorSlot* s = graph_get_slot(host, bytes, 0);
    if (!s) return NULL;
    if (!slot_ensure_owned_buffer(s, bytes, NULL)) return NULL;
    s->host_dirty = 0;
    return s;
}

static void graph_mark_device(OglTensorSlot* s) {
    if (!s) return;
    s->device_dirty = 1;
    s->host_dirty = 0;
}

static int dispatch_kernel(OglKernel* k, GLuint* buffers, uint32_t gx, uint32_t gy, uint32_t gz) {
    GLuint prog = compile_kernel(k);
    if (!prog || !buffers || gx == 0 || gy == 0 || gz == 0) return 0;
    static int profile_sync = -1;
    if (profile_sync < 0) {
        const char* env = getenv("VOLVOX_GL_PROFILE_SYNC");
        profile_sync = env && env[0] && strcmp(env, "0") ? 1 : 0;
    }
    double t0 = profile_sync ? engine_now_ms() : 0.0;
    p_glUseProgram(prog);
    for (int i = 0; i < k->binding_count; i++) {
        if (!buffers[i]) return 0;
        p_glBindBufferBase(i == k->uniform_binding ? GL_UNIFORM_BUFFER : GL_SHADER_STORAGE_BUFFER,
                           (GLuint)i, buffers[i]);
    }
    p_glDispatchCompute((GLuint)gx, (GLuint)gy, (GLuint)gz);
    p_glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    if (profile_sync && p_glFinish) {
        p_glFinish();
        prof_add_entry(k->name, engine_now_ms() - t0);
    }
    return 1;
}

static GLuint params_buffer(const void* data, size_t bytes) {
    return create_buffer_target(GL_UNIFORM_BUFFER, bytes, data);
}

static int read_buffer(GLenum target, size_t bytes, void* out) {
    if (!out || bytes == 0) return 0;
    if (p_glGetBufferSubData) {
        p_glGetBufferSubData(target, 0, (GLsizeiptr)bytes, out);
        return 1;
    }
    if (!p_glMapBufferRange || !p_glUnmapBuffer) return 0;
    void* mapped = p_glMapBufferRange(target, 0, (GLsizeiptr)bytes, GL_MAP_READ_BIT);
    if (!mapped) return 0;
    memcpy(out, mapped, bytes);
    p_glUnmapBuffer(target);
    return 1;
}

void opengl_graph_reset(void) {
    if (p_glDeleteBuffers) {
        for (int i = 0; i < graph_slot_count; i++) {
            if (graph_slots[i].owns_buffer && graph_slots[i].buffer) {
                p_glDeleteBuffers(1, &graph_slots[i].buffer);
            }
        }
    }
    memset(graph_slots, 0, sizeof(graph_slots));
    graph_slot_count = 0;
}

void opengl_graph_begin_forward(void) {
}

int opengl_graph_end_forward(void) {
    if (!ogl_ready()) return 0;
    p_glFinish();
    return 0;
}

void opengl_graph_mark_host(const void* host, size_t bytes, int is_weight) {
    OglTensorSlot* s = graph_get_slot(host, bytes, is_weight);
    if (!s) return;
    s->host_dirty = 1;
    s->device_dirty = 0;
}

int opengl_graph_sync_host(const void* host, size_t bytes, int is_weight) {
    (void)is_weight;
    if (!ogl_ready() || !host || bytes == 0) return 0;
    int idx = graph_find_slot(host);
    if (idx < 0) return 0;
    OglTensorSlot* s = &graph_slots[idx];
    if (!s->device_dirty || !s->buffer || bytes > s->bytes) return 0;
    p_glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    p_glFinish();
    p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, s->buffer);
    int ok = read_buffer(GL_SHADER_STORAGE_BUFFER, bytes, (void*)host);
    p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    if (!ok) return 0;
    s->device_dirty = 0;
    s->host_dirty = 0;
    return 1;
}

int opengl_graph_alias_f32(const float* in, float* out, long n) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_get_slot(out, bytes, 0);
    if (!src || !dst || !src->buffer) return 0;
    if (dst->owns_buffer && dst->buffer && dst->buffer != src->buffer) p_glDeleteBuffers(1, &dst->buffer);
    dst->buffer = src->buffer;
    dst->cap = src->cap;
    dst->bytes = bytes;
    dst->owns_buffer = 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_copy_f32(const float* in, float* out, long n) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[1] = {(uint32_t)n};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_copy, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_add_f32(const float* a, const float* bptr, float* out, long n) {
    return opengl_graph_add_relu_f32(a, bptr, out, n, 0);
}

int opengl_graph_add_relu_f32(const float* a, const float* bptr, float* out, long n, int relu) {
    if (n <= 0 || !a || !bptr || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* sa = graph_ensure_device(a, bytes, 0);
    OglTensorSlot* sb = graph_ensure_device(bptr, bytes, 0);
    OglTensorSlot* so = graph_output_slot(out, bytes);
    if (!sa || !sb || !so) return 0;
    uint32_t params[2] = {(uint32_t)n, (uint32_t)relu};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[4] = {sa->buffer, sb->buffer, so->buffer, pb};
    int ok = dispatch_kernel(&k_add, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(so);
    return 1;
}

int opengl_graph_add3_relu_f32(const float* a, const float* bptr, const float* c, float* out, long n, int relu) {
    if (n <= 0 || !a || !bptr || !c || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* sa = graph_ensure_device(a, bytes, 0);
    OglTensorSlot* sb = graph_ensure_device(bptr, bytes, 0);
    OglTensorSlot* sc = graph_ensure_device(c, bytes, 0);
    OglTensorSlot* so = graph_output_slot(out, bytes);
    if (!sa || !sb || !sc || !so) return 0;
    uint32_t params[2] = {(uint32_t)n, (uint32_t)relu};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[5] = {sa->buffer, sb->buffer, sc->buffer, so->buffer, pb};
    int ok = dispatch_kernel(&k_add3, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(so);
    return 1;
}

int opengl_graph_clip_f32(const float* in, float* out, long n, float min_v, float max_v) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    struct { uint32_t size; float min_v; float max_v; } params = {(uint32_t)n, min_v, max_v};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_clip, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_sigmoid_f32(const float* in, float* out, long n) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[1] = {(uint32_t)n};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_sigmoid, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

static int opengl_graph_unary_size_f32(OglKernel* k, const float* in, float* out, long n) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[1] = {(uint32_t)n};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(k, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_relu_f32(const float* in, float* out, long n) { return opengl_graph_unary_size_f32(&k_relu, in, out, n); }
int opengl_graph_gelu_f32(const float* in, float* out, long n) { return opengl_graph_unary_size_f32(&k_gelu, in, out, n); }
int opengl_graph_silu_f32(const float* in, float* out, long n) { return opengl_graph_unary_size_f32(&k_silu, in, out, n); }
int opengl_graph_tanh_f32(const float* in, float* out, long n) { return opengl_graph_unary_size_f32(&k_tanh, in, out, n); }
int opengl_graph_hardswish_f32(const float* in, float* out, long n) { return opengl_graph_unary_size_f32(&k_hardswish, in, out, n); }
int opengl_graph_hardsigmoid_f32(const float* in, float* out, long n) { return opengl_graph_unary_size_f32(&k_hardsigmoid, in, out, n); }
int opengl_graph_cast_copy_f32(const float* in, float* out, long n) { return opengl_graph_copy_f32(in, out, n); }

int opengl_graph_leaky_relu_f32(const float* in, float* out, long n, float alpha) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    struct { uint32_t size; float alpha; } params = {(uint32_t)n, alpha};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_leaky_relu, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_prelu_f32(const float* in, const float* weight, float* out, long n, int channels) {
    if (n <= 0 || channels <= 0 || !in || !weight || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    size_t wbytes = (size_t)channels * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* w = graph_ensure_device(weight, wbytes, 1);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !w || !dst) return 0;
    uint32_t params[2] = {(uint32_t)n, (uint32_t)channels};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[4] = {src->buffer, w->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_prelu, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_layernorm_f32(const float* in, const float* weight, const float* bias,
                               float* out, int rows, int d_model) {
    if (rows <= 0 || d_model <= 0 || !in || !weight || !bias || !out) return 0;
    size_t bytes = (size_t)rows * d_model * sizeof(float);
    size_t wbytes = (size_t)d_model * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* w = graph_ensure_device(weight, wbytes, 1);
    OglTensorSlot* b = graph_ensure_device(bias, wbytes, 1);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !w || !b || !dst) return 0;
    uint32_t params[2] = {(uint32_t)rows, (uint32_t)d_model};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[5] = {src->buffer, w->buffer, b->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_layernorm, bufs, ((uint32_t)rows + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_rmsnorm_f32(const float* in, const float* weight, float* out,
                             int rows, int d_model, float eps) {
    if (rows <= 0 || d_model <= 0 || !in || !weight || !out) return 0;
    size_t bytes = (size_t)rows * d_model * sizeof(float);
    size_t wbytes = (size_t)d_model * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* w = graph_ensure_device(weight, wbytes, 1);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !w || !dst) return 0;
    struct { uint32_t rows; uint32_t d_model; float eps; } params = {(uint32_t)rows, (uint32_t)d_model, eps};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[4] = {src->buffer, w->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_rmsnorm, bufs, ((uint32_t)rows + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

static int opengl_graph_softmax_like_f32(OglKernel* k, const float* in, float* out, int rows, int d) {
    if (rows <= 0 || d <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)rows * d * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[2] = {(uint32_t)rows, (uint32_t)d};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(k, bufs, ((uint32_t)rows + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_softmax_f32(const float* in, float* out, int rows, int d) {
    return opengl_graph_softmax_like_f32(&k_softmax, in, out, rows, d);
}

int opengl_graph_logsoftmax_f32(const float* in, float* out, int rows, int d) {
    return opengl_graph_softmax_like_f32(&k_logsoftmax, in, out, rows, d);
}

int opengl_graph_reduce_f32(const float* in, float* out, int rows, int d, float inv) {
    if (rows <= 0 || d <= 0 || !in || !out) return 0;
    size_t in_bytes = (size_t)rows * d * sizeof(float);
    size_t out_bytes = (size_t)rows * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    struct { uint32_t rows; uint32_t d; float inv; } params = {(uint32_t)rows, (uint32_t)d, inv};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_reduce, bufs, ((uint32_t)rows + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_global_average_pool_f32(const float* in, float* out, int n, int h, int w, int c) {
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || !in || !out) return 0;
    size_t in_bytes = (size_t)n * h * w * c * sizeof(float);
    size_t out_bytes = (size_t)n * c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_globalavg, bufs, ((uint32_t)c + 63u) / 64u, (uint32_t)n, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_average_pool2d_f32(const float* in, float* out, int n, int h, int w, int c,
                                    int out_h, int out_w, int ky, int kx, int sy, int sx,
                                    int py, int px) {
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || out_h <= 0 || out_w <= 0 ||
        ky <= 0 || kx <= 0 || sy <= 0 || sx <= 0 || !in || !out) return 0;
    size_t in_bytes = (size_t)n * h * w * c * sizeof(float);
    size_t out_bytes = (size_t)n * out_h * out_w * c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[12] = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c,
                           (uint32_t)out_h, (uint32_t)out_w, (uint32_t)ky, (uint32_t)kx,
                           (uint32_t)sy, (uint32_t)sx, (uint32_t)py, (uint32_t)px};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_avgpool, bufs, ((uint32_t)out_w + 7u) / 8u,
                             ((uint32_t)out_h + 7u) / 8u, (uint32_t)(n * c));
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_batchnorm2d_f32(const float* in, const float* weight, const float* bias,
                                 const float* mean, const float* var, float* out,
                                 int n, int h, int w, int c, float eps) {
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || !in || !weight || !bias || !mean || !var || !out) return 0;
    size_t bytes = (size_t)n * h * w * c * sizeof(float);
    size_t cbytes = (size_t)c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* sw = graph_ensure_device(weight, cbytes, 1);
    OglTensorSlot* sb = graph_ensure_device(bias, cbytes, 1);
    OglTensorSlot* sm = graph_ensure_device(mean, cbytes, 1);
    OglTensorSlot* sv = graph_ensure_device(var, cbytes, 1);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !sw || !sb || !sm || !sv || !dst) return 0;
    struct { uint32_t n; uint32_t c; uint32_t h; uint32_t w; float eps; uint32_t pad[3]; } params =
        {(uint32_t)n, (uint32_t)c, (uint32_t)h, (uint32_t)w, eps, {0, 0, 0}};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[7] = {src->buffer, sw->buffer, sb->buffer, sm->buffer, sv->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_batchnorm, bufs, ((uint32_t)(n * h * w * c) + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_embedding_f32(const float* tokens, const float* weight, float* out,
                               int tokens_len, int d_model, int vocab_size) {
    if (tokens_len <= 0 || d_model <= 0 || vocab_size <= 0 || !tokens || !weight || !out) return 0;
    size_t tbytes = (size_t)tokens_len * sizeof(float);
    size_t wbytes = (size_t)vocab_size * d_model * sizeof(float);
    size_t obytes = (size_t)tokens_len * d_model * sizeof(float);
    OglTensorSlot* st = graph_ensure_device(tokens, tbytes, 0);
    OglTensorSlot* sw = graph_ensure_device(weight, wbytes, 1);
    OglTensorSlot* so = graph_output_slot(out, obytes);
    if (!st || !sw || !so) return 0;
    uint32_t params[2] = {(uint32_t)tokens_len, (uint32_t)d_model};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[4] = {st->buffer, sw->buffer, so->buffer, pb};
    int ok = dispatch_kernel(&k_embedding, bufs, ((uint32_t)tokens_len + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(so);
    return 1;
}

int opengl_graph_transpose_f32(const float* in, float* out, const int* in_shape,
                               const int* perm, int rank) {
    if (rank <= 0 || rank > 8 || !in || !out || !in_shape || !perm) return 0;
    uint32_t in_stride[8] = {0}, out_shape[8] = {0}, out_stride[8] = {0};
    long total = 1;
    for (int i = 0; i < rank; i++) {
        if (in_shape[i] <= 0 || perm[i] < 0 || perm[i] >= rank) return 0;
        out_shape[i] = (uint32_t)in_shape[perm[i]];
        total *= out_shape[i];
    }
    in_stride[rank - 1] = 1;
    out_stride[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
        in_stride[i] = in_stride[i + 1] * (uint32_t)in_shape[i + 1];
        out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
    }
    size_t bytes = (size_t)total * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t meta[2 + 16] = {0};
    meta[0] = (uint32_t)total;
    meta[1] = (uint32_t)rank;
    for (int d = 0; d < rank; d++) {
        meta[2 + d] = out_stride[d];
        meta[2 + rank + d] = in_stride[perm[d]];
    }
    size_t mbytes = (size_t)(2 + 2 * rank) * sizeof(uint32_t);
    GLuint mb = create_buffer(mbytes, meta);
    GLuint bufs[3] = {src->buffer, dst->buffer, mb};
    int ok = dispatch_kernel(&k_transpose, bufs, ((uint32_t)total + 63u) / 64u, 1, 1);
    if (mb) p_glDeleteBuffers(1, &mb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_where_f32(const float* cond, const float* a, const float* b, float* out, long n) {
    if (n <= 0 || !cond || !a || !b || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* sc = graph_ensure_device(cond, bytes, 0);
    OglTensorSlot* sa = graph_ensure_device(a, bytes, 0);
    OglTensorSlot* sb = graph_ensure_device(b, bytes, 0);
    OglTensorSlot* so = graph_output_slot(out, bytes);
    if (!sc || !sa || !sb || !so) return 0;
    uint32_t params[1] = {(uint32_t)n};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[5] = {sc->buffer, sa->buffer, sb->buffer, so->buffer, pb};
    int ok = dispatch_kernel(&k_where, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(so);
    return 1;
}

int opengl_graph_upsample2x_f32(const float* in, float* out, int n, int h, int w, int c) {
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0 || !in || !out) return 0;
    size_t in_bytes = (size_t)n * h * w * c * sizeof(float);
    size_t out_bytes = (size_t)n * h * 2 * w * 2 * c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_upsample, bufs, ((uint32_t)(w * 2) + 7u) / 8u,
                             ((uint32_t)(h * 2) + 7u) / 8u, (uint32_t)(n * c));
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_resize_nearest_f32(const float* in, float* out, int n, int h, int w, int c,
                                    int out_h, int out_w) {
    return opengl_graph_resize_f32(in, out, n, h, w, c, out_h, out_w, 0);
}

int opengl_graph_resize_f32(const float* in, float* out, int n, int h, int w, int c,
                            int out_h, int out_w, int mode) {
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0 || out_h <= 0 || out_w <= 0 || !in || !out) return 0;
    size_t in_bytes = (size_t)n * h * w * c * sizeof(float);
    size_t out_bytes = (size_t)n * out_h * out_w * c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[8] = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c,
                          (uint32_t)out_h, (uint32_t)out_w, (uint32_t)mode, 0u};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_resize, bufs, ((uint32_t)out_w + 7u) / 8u,
                             ((uint32_t)out_h + 7u) / 8u, (uint32_t)(n * c));
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

static void pad4_shape(const int* shape, int rank, uint32_t out[4]) {
    for (int i = 0; i < 4; i++) out[i] = 1u;
    if (!shape || rank <= 0) return;
    int base = 4 - rank;
    if (base < 0) base = 0;
    for (int i = 0; i < rank && i < 4; i++) out[base + i] = (uint32_t)shape[i];
}

int opengl_graph_expand_f32(const float* in, float* out, const int* in_shape, int in_rank,
                            const int* out_shape, int out_rank) {
    if (!in || !out || !in_shape || !out_shape || in_rank <= 0 || out_rank <= 0 || in_rank > 4 || out_rank > 4) return 0;
    uint32_t is[4], os[4];
    pad4_shape(in_shape, in_rank, is);
    pad4_shape(out_shape, out_rank, os);
    size_t in_elems = (size_t)is[0] * is[1] * is[2] * is[3];
    size_t out_elems = (size_t)os[0] * os[1] * os[2] * os[3];
    OglTensorSlot* src = graph_ensure_device(in, in_elems * sizeof(float), 0);
    OglTensorSlot* dst = graph_output_slot(out, out_elems * sizeof(float));
    if (!src || !dst) return 0;
    uint32_t params[8] = {is[0], is[1], is[2], is[3], os[0], os[1], os[2], os[3]};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_expand, bufs, ((uint32_t)out_elems + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_gather_axis0_f32(const float* in, const float* indices, float* out,
                                  int row_size, int input_rows, int num_idx) {
    if (row_size <= 0 || input_rows <= 0 || num_idx <= 0 || !in || !indices || !out) return 0;
    long total = (long)row_size * num_idx;
    OglTensorSlot* src = graph_ensure_device(in, (size_t)input_rows * row_size * sizeof(float), 0);
    OglTensorSlot* idx = graph_ensure_device(indices, (size_t)num_idx * sizeof(float), 0);
    OglTensorSlot* dst = graph_output_slot(out, (size_t)total * sizeof(float));
    if (!src || !idx || !dst) return 0;
    uint32_t params[3] = {(uint32_t)row_size, (uint32_t)num_idx, (uint32_t)total};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[4] = {src->buffer, idx->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_gather, bufs, ((uint32_t)total + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_pad4d_f32(const float* in, float* out, const int* in_shape, int in_rank,
                           const int* out_shape, int out_rank, int pad_top, int pad_left, float value) {
    if (!in || !out || !in_shape || !out_shape || in_rank <= 0 || out_rank <= 0 || in_rank > 4 || out_rank > 4) return 0;
    uint32_t is[4], os[4];
    pad4_shape(in_shape, in_rank, is);
    pad4_shape(out_shape, out_rank, os);
    size_t in_elems = (size_t)is[0] * is[1] * is[2] * is[3];
    size_t out_elems = (size_t)os[0] * os[1] * os[2] * os[3];
    OglTensorSlot* src = graph_ensure_device(in, in_elems * sizeof(float), 0);
    OglTensorSlot* dst = graph_output_slot(out, out_elems * sizeof(float));
    if (!src || !dst) return 0;
    struct {
        uint32_t b, in_h, in_w, c, out_h, out_w, pt, pl;
        float val;
        uint32_t pad[3];
    } params = {is[0], is[1], is[2], is[3], os[1], os[2], (uint32_t)pad_top, (uint32_t)pad_left, value, {0, 0, 0}};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_pad, bufs, ((uint32_t)out_elems + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_slice4d_f32(const float* in, float* out, const int* in_shape, int in_rank,
                             const int* out_shape, int out_rank, const int* starts,
                             const int* steps) {
    if (!in || !out || !in_shape || !out_shape || !starts || !steps ||
        in_rank <= 0 || out_rank <= 0 || in_rank > 4 || out_rank > 4) return 0;
    uint32_t is[4], os[4];
    pad4_shape(in_shape, in_rank, is);
    pad4_shape(out_shape, out_rank, os);
    size_t in_elems = (size_t)is[0] * is[1] * is[2] * is[3];
    size_t out_elems = (size_t)os[0] * os[1] * os[2] * os[3];
    OglTensorSlot* src = graph_ensure_device(in, in_elems * sizeof(float), 0);
    OglTensorSlot* dst = graph_output_slot(out, out_elems * sizeof(float));
    if (!src || !dst) return 0;
    uint32_t params[16] = {
        os[0], os[1], os[2], os[3], is[1], is[2], is[3],
        (uint32_t)starts[0], (uint32_t)starts[1], (uint32_t)starts[2], (uint32_t)starts[3],
        (uint32_t)steps[0], (uint32_t)steps[1], (uint32_t)steps[2], (uint32_t)steps[3],
        (uint32_t)out_elems
    };
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_slice, bufs, ((uint32_t)out_elems + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_conv_transpose2d_f32(const float* in, const float* weight, const float* bias,
                                      float* out, int n, int in_h, int in_w, int in_c,
                                      int out_h, int out_w, int out_c, int kh, int kw,
                                      int sh, int sw, int ph, int pw) {
    if (n <= 0 || in_h <= 0 || in_w <= 0 || in_c <= 0 || out_h <= 0 || out_w <= 0 ||
        out_c <= 0 || kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0 || !in || !weight || !out) return 0;
    size_t in_bytes = (size_t)n * in_h * in_w * in_c * sizeof(float);
    size_t wbytes = (size_t)in_c * out_c * kh * kw * sizeof(float);
    size_t bbytes = (size_t)out_c * sizeof(float);
    size_t out_bytes = (size_t)n * out_h * out_w * out_c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* swt = graph_ensure_device(weight, wbytes, 1);
    static const float zero_bias[1] = {0.0f};
    OglTensorSlot* sb = graph_ensure_device(bias ? bias : zero_bias, bias ? bbytes : sizeof(float), 1);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !swt || !sb || !dst) return 0;
    uint32_t params[16] = {(uint32_t)n, (uint32_t)in_h, (uint32_t)in_w, (uint32_t)in_c,
                           (uint32_t)out_h, (uint32_t)out_w, (uint32_t)out_c,
                           (uint32_t)kh, (uint32_t)kw, (uint32_t)sh, (uint32_t)sw,
                           (uint32_t)ph, (uint32_t)pw, bias ? 1u : 0u, 0u, 0u};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[5] = {src->buffer, swt->buffer, sb->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_convtranspose, bufs, ((uint32_t)out_w + 7u) / 8u,
                             ((uint32_t)out_h + 7u) / 8u, (uint32_t)(n * out_c));
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_interp1d_f32(const float* in, float* out, int channels, int in_l, int out_l) {
    if (channels <= 0 || in_l <= 0 || out_l <= 0 || !in || !out) return 0;
    size_t in_bytes = (size_t)channels * in_l * sizeof(float);
    size_t out_bytes = (size_t)channels * out_l * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[3] = {(uint32_t)channels, (uint32_t)in_l, (uint32_t)out_l};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_interp1d, bufs, ((uint32_t)out_l + 63u) / 64u, (uint32_t)channels, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_binary_f32(const float* a, long a_numel, const float* b, long b_numel,
                            float* out, long out_numel, int op) {
    if (!a || !b || !out || a_numel <= 0 || b_numel <= 0 || out_numel <= 0) return 0;
    if (a_numel > out_numel || b_numel > out_numel) return 0;
    OglKernel* kernel = op == 1 ? &k_sub : (op == 2 ? &k_div : &k_mul);
    size_t a_bytes = (size_t)a_numel * sizeof(float);
    size_t b_bytes = (size_t)b_numel * sizeof(float);
    size_t out_bytes = (size_t)out_numel * sizeof(float);
    OglTensorSlot* sa = graph_ensure_device(a, a_bytes, 0);
    OglTensorSlot* sb = graph_ensure_device(b, b_bytes, 0);
    OglTensorSlot* so = graph_output_slot(out, out_bytes);
    if (!sa || !sb || !so) return 0;
    uint32_t params[5] = {(uint32_t)out_numel, b_numel == 1 ? 1u : 0u, (uint32_t)b_numel,
                          (uint32_t)a_numel, a_numel == 1 ? 1u : 0u};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[4] = {sa->buffer, sb->buffer, so->buffer, pb};
    int ok = dispatch_kernel(kernel, bufs, ((uint32_t)out_numel + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(so);
    return 1;
}

int opengl_graph_split_f32(const float* in, float* out, long input_numel, long output_numel,
                           int inner, int split_size, int axis_in, int offset) {
    if (!in || !out || input_numel <= 0 || output_numel <= 0 ||
        inner <= 0 || split_size <= 0 || axis_in <= 0 || offset < 0 || offset + split_size > axis_in) return 0;
    size_t in_bytes = (size_t)input_numel * sizeof(float);
    size_t out_bytes = (size_t)output_numel * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[5] = {(uint32_t)output_numel, (uint32_t)inner, (uint32_t)split_size,
                          (uint32_t)axis_in, (uint32_t)offset};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_split, bufs, ((uint32_t)output_numel + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_conv1d_f32(const float* in, const float* weight, const float* bias, float* out,
                            int in_c, int in_l, int out_c, int out_l, int kernel,
                            int stride, int pad, int relu) {
    if (!in || !weight || !out || in_c <= 0 || in_l <= 0 || out_c <= 0 || out_l <= 0 ||
        kernel <= 0 || stride <= 0 || pad < 0 || relu < 0 || relu > 1) return 0;
    size_t in_bytes = (size_t)in_c * in_l * sizeof(float);
    size_t wbytes = (size_t)out_c * in_c * kernel * sizeof(float);
    size_t bbytes = (size_t)out_c * sizeof(float);
    size_t out_bytes = (size_t)out_c * out_l * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* sw = graph_ensure_device(weight, wbytes, 1);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !sw || !dst) return 0;
    GLuint bias_buf = 0;
    int delete_bias = 0;
    if (bias) {
        OglTensorSlot* sb = graph_ensure_device(bias, bbytes, 1);
        if (!sb) return 0;
        bias_buf = sb->buffer;
    } else {
        float* zeros = (float*)calloc((size_t)out_c, sizeof(float));
        if (!zeros) return 0;
        bias_buf = create_buffer(bbytes, zeros);
        free(zeros);
        delete_bias = 1;
        if (!bias_buf) return 0;
    }
    uint32_t params[8] = {(uint32_t)in_c, (uint32_t)in_l, (uint32_t)out_c, (uint32_t)kernel,
                          (uint32_t)stride, (uint32_t)pad, (uint32_t)relu, 0u};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[5] = {src->buffer, sw->buffer, bias_buf, dst->buffer, pb};
    int ok = dispatch_kernel(&k_conv1d, bufs, ((uint32_t)out_l + 63u) / 64u, (uint32_t)out_c, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (delete_bias && bias_buf) p_glDeleteBuffers(1, &bias_buf);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_sdpa_f32(const float* qkv, float* out, int seq_len, int d_model,
                          int num_heads, int head_dim, float scale) {
    if (!qkv || !out || seq_len <= 0 || d_model <= 0 || num_heads <= 0 || head_dim <= 0 ||
        head_dim > 64 || d_model != num_heads * head_dim) return 0;
    size_t qkv_bytes = (size_t)seq_len * 3u * (size_t)d_model * sizeof(float);
    size_t out_bytes = (size_t)seq_len * (size_t)d_model * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(qkv, qkv_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    struct { uint32_t seq_len, d_model, num_heads, head_dim; float scale; uint32_t pad[3]; } params =
        {(uint32_t)seq_len, (uint32_t)d_model, (uint32_t)num_heads, (uint32_t)head_dim, scale, {0, 0, 0}};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_sdpa, bufs, ((uint32_t)seq_len + 63u) / 64u, (uint32_t)num_heads, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_cross_sdpa_f32(const float* q, const float* k, const float* v, float* out,
                                int seq_q, int seq_kv, int d_model, int num_heads,
                                int head_dim, float scale) {
    if (!q || !k || !v || !out || seq_q <= 0 || seq_kv <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 || head_dim > 64 || d_model != num_heads * head_dim) return 0;
    size_t q_bytes = (size_t)seq_q * (size_t)d_model * sizeof(float);
    size_t kv_bytes = (size_t)seq_kv * (size_t)d_model * sizeof(float);
    size_t out_bytes = q_bytes;
    OglTensorSlot* sq = graph_ensure_device(q, q_bytes, 0);
    OglTensorSlot* sk = graph_ensure_device(k, kv_bytes, 0);
    OglTensorSlot* sv = graph_ensure_device(v, kv_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sq || !sk || !sv || !dst) return 0;
    struct { uint32_t seq_q, seq_kv, d_model, num_heads, head_dim; float scale; uint32_t pad[2]; } params =
        {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model, (uint32_t)num_heads, (uint32_t)head_dim, scale, {0, 0}};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[5] = {sq->buffer, sk->buffer, sv->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_cross_sdpa, bufs, ((uint32_t)seq_q + 63u) / 64u, (uint32_t)num_heads, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_cross_attention_f32(const float* q, const float* kv, const float* weight,
                                     const float* scale, const float* bias, float* out,
                                     int seq_q, int seq_kv, int d_model, int num_heads,
                                     int head_dim, int has_scale, int has_bias) {
    if (!q || !kv || !weight || !out || seq_q <= 0 || seq_kv <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 || head_dim > 64 || d_model != num_heads * head_dim ||
        d_model > 64) return 0;
    size_t q_bytes = (size_t)seq_q * (size_t)d_model * sizeof(float);
    size_t kv_bytes = (size_t)seq_kv * (size_t)d_model * sizeof(float);
    size_t wbytes = (size_t)3 * (size_t)d_model * (size_t)d_model * sizeof(float);
    size_t sb_bytes = (size_t)3 * (size_t)d_model * sizeof(float);
    size_t out_bytes = q_bytes;
    static const float zero[1] = {0.0f};
    OglTensorSlot* sq = graph_ensure_device(q, q_bytes, 0);
    OglTensorSlot* skv = graph_ensure_device(kv, kv_bytes, 0);
    OglTensorSlot* sw = graph_ensure_device(weight, wbytes, 1);
    OglTensorSlot* ss = graph_ensure_device(has_scale && scale ? scale : zero, has_scale && scale ? sb_bytes : sizeof(float), 1);
    OglTensorSlot* sb = graph_ensure_device(has_bias && bias ? bias : zero, has_bias && bias ? sb_bytes : sizeof(float), 1);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sq || !skv || !sw || !ss || !sb || !dst) return 0;
    struct {
        uint32_t seq_q, seq_kv, d_model, num_heads, head_dim;
        float scale_factor;
        uint32_t has_scale, has_bias;
    } params = {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model, (uint32_t)num_heads,
                (uint32_t)head_dim, 1.0f / sqrtf((float)head_dim),
                (uint32_t)(has_scale && scale), (uint32_t)(has_bias && bias)};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[7] = {sq->buffer, skv->buffer, sw->buffer, ss->buffer, sb->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_cross_attention, bufs, ((uint32_t)seq_q + 63u) / 64u, (uint32_t)num_heads, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_dequantize_linear_f32(const float* in, const float* scale, const float* zero_point,
                                       float* out, long n, int has_zero_point) {
    if (!in || !scale || !out || n <= 0) return 0;
    static const float zero[1] = {0.0f};
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* ss = graph_ensure_device(scale, sizeof(float), 1);
    OglTensorSlot* sz = graph_ensure_device(has_zero_point && zero_point ? zero_point : zero, sizeof(float), 1);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !ss || !sz || !dst) return 0;
    uint32_t params[4] = {(uint32_t)n, (uint32_t)(has_zero_point && zero_point), 0u, 0u};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[5] = {src->buffer, ss->buffer, sz->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_dequantize, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_quantize_linear_i8(const float* in, signed char* out, long n,
                                    float input_scale, int input_zp,
                                    float output_scale, int output_zp) {
    if (!in || !out || n <= 0 || output_scale <= 0.0f) return 0;
    size_t in_bytes = (size_t)n * sizeof(float);
    size_t packed_words = ((size_t)n + 3u) / 4u;
    size_t out_bytes = packed_words * sizeof(uint32_t);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    struct {
        uint32_t size;
        int32_t input_zp;
        int32_t output_zp;
        uint32_t has_input_scale;
        float input_scale;
        float output_scale;
        uint32_t pad0;
        uint32_t pad1;
    } params = {
        (uint32_t)n, (int32_t)input_zp, (int32_t)output_zp,
        input_scale > 0.0f ? 1u : 0u, input_scale, output_scale, 0u, 0u
    };
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_quantize, bufs, (uint32_t)((packed_words + 63u) / 64u), 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return opengl_graph_sync_host(out, (size_t)n, 0);
}

static int opengl_graph_profile_common(OglKernel* kernel, const float* in, float* out,
                                       int h, int w, int c, long out_elems,
                                       uint32_t gx, uint32_t gy) {
    if (!kernel || !in || !out || h <= 0 || w <= 0 || c <= 0 || out_elems <= 0) return 0;
    size_t in_bytes = (size_t)h * (size_t)w * (size_t)c * sizeof(float);
    size_t out_bytes = (size_t)out_elems * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)h, (uint32_t)w, (uint32_t)c, 0u};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(kernel, bufs, gx, gy, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_spatial_softargmax_y_f32(const float* in, float* out, int h, int w, int c) {
    return opengl_graph_profile_common(&k_spatial_softargmax_y, in, out, h, w, c,
                                       (long)c * w, ((uint32_t)w + 63u) / 64u, (uint32_t)c);
}

int opengl_graph_profile_x_f32(const float* in, float* out, int h, int w, int c) {
    return opengl_graph_profile_common(&k_profile_x, in, out, h, w, c,
                                       (long)2 * c * w, ((uint32_t)w + 63u) / 64u, (uint32_t)c);
}

int opengl_graph_profile_y_f32(const float* in, float* out, int h, int w, int c) {
    return opengl_graph_profile_common(&k_profile_y, in, out, h, w, c,
                                       (long)2 * c * h, ((uint32_t)h + 63u) / 64u, (uint32_t)c);
}

int opengl_graph_mean_height_f32(const float* in, float* out, int h, int w, int c) {
    return opengl_graph_profile_common(&k_mean_height, in, out, h, w, c,
                                       (long)c * w, ((uint32_t)w + 63u) / 64u, (uint32_t)c);
}

int opengl_graph_nms_f32(const float* boxes, const float* scores, float* out,
                         int batches, int spatial, int classes, int max_output,
                         int output_rows, float iou_threshold, float score_threshold) {
    if (!boxes || !scores || !out || batches <= 0 || spatial <= 0 || classes <= 0 ||
        max_output <= 0 || output_rows <= 0) return 0;
    size_t boxes_bytes = (size_t)batches * (size_t)spatial * 4u * sizeof(float);
    size_t scores_bytes = (size_t)batches * (size_t)classes * (size_t)spatial * sizeof(float);
    size_t out_bytes = (size_t)output_rows * 3u * sizeof(float);
    OglTensorSlot* sb = graph_ensure_device(boxes, boxes_bytes, 0);
    OglTensorSlot* ss = graph_ensure_device(scores, scores_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sb || !ss || !dst) return 0;
    struct {
        uint32_t batches, spatial, classes, max_output, output_rows;
        float iou_threshold, score_threshold;
        uint32_t pad;
    } params = {(uint32_t)batches, (uint32_t)spatial, (uint32_t)classes, (uint32_t)max_output,
                (uint32_t)output_rows, iou_threshold, score_threshold, 0u};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[4] = {sb->buffer, ss->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_nms, bufs, 1, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_concat_flat_f32(const float** inputs, const long* sizes, int count, float* out) {
    if (!inputs || !sizes || !out || count <= 0) return 0;
    long total = 0;
    for (int i = 0; i < count; i++) {
        if (!inputs[i] || sizes[i] <= 0) return 0;
        total += sizes[i];
    }
    size_t out_bytes = (size_t)total * sizeof(float);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!dst) return 0;
    long offset = 0;
    for (int i = 0; i < count; i++) {
        size_t in_bytes = (size_t)sizes[i] * sizeof(float);
        OglTensorSlot* src = graph_ensure_device(inputs[i], in_bytes, 0);
        if (!src) return 0;
        uint32_t params[2] = {(uint32_t)sizes[i], (uint32_t)offset};
        GLuint pb = params_buffer(params, sizeof(params));
        GLuint bufs[3] = {src->buffer, dst->buffer, pb};
        int ok = dispatch_kernel(&k_concat, bufs, ((uint32_t)sizes[i] + 63u) / 64u, 1, 1);
        if (pb) p_glDeleteBuffers(1, &pb);
        if (!ok) return 0;
        offset += sizes[i];
    }
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_concat_sigmoid_flat_f32(const float** inputs, const long* sizes, int count, float* out) {
    if (!inputs || !sizes || !out || count <= 0) return 0;
    long total = 0;
    for (int i = 0; i < count; i++) {
        if (!inputs[i] || sizes[i] <= 0) return 0;
        total += sizes[i];
    }
    size_t out_bytes = (size_t)total * sizeof(float);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!dst) return 0;
    long offset = 0;
    for (int i = 0; i < count; i++) {
        size_t in_bytes = (size_t)sizes[i] * sizeof(float);
        OglTensorSlot* src = graph_ensure_device(inputs[i], in_bytes, 0);
        if (!src) return 0;
        uint32_t params[2] = {(uint32_t)sizes[i], (uint32_t)offset};
        GLuint pb = params_buffer(params, sizeof(params));
        GLuint bufs[3] = {src->buffer, dst->buffer, pb};
        int ok = dispatch_kernel(&k_concat_sigmoid, bufs, ((uint32_t)sizes[i] + 63u) / 64u, 1, 1);
        if (pb) p_glDeleteBuffers(1, &pb);
        if (!ok) return 0;
        offset += sizes[i];
    }
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_maxpool2d_f32(const float* in, float* out, int h, int width, int c,
                               int out_h, int out_w, int ky, int kx, int sy, int sx,
                               int py, int px) {
    if (!in || !out || c <= 0 || h <= 0 || width <= 0 || out_h <= 0 || out_w <= 0 ||
        ky <= 0 || kx <= 0 || sy <= 0 || sx <= 0) return 0;
    size_t in_bytes = (size_t)h * width * c * sizeof(float);
    size_t out_bytes = (size_t)out_h * out_w * c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[11] = {
        (uint32_t)h, (uint32_t)width, (uint32_t)c, (uint32_t)out_h, (uint32_t)out_w,
        (uint32_t)ky, (uint32_t)kx, (uint32_t)sy, (uint32_t)sx, (uint32_t)py, (uint32_t)px
    };
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_maxpool, bufs, ((uint32_t)out_w + 7u) / 8u,
                             ((uint32_t)out_h + 7u) / 8u, (uint32_t)c);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_conv2d_f32(const float* in, float* out, const float* w, const float* bptr,
                                 int n, int h, int width, int c, int out_c,
                                 int kh, int kw, int out_h, int out_w,
                                 int sy, int sx, int pt, int pl,
                                 int groups, int relu, int dy, int dx) {
    if (!in || !out || !w || n <= 0 || h <= 0 || width <= 0 || c <= 0 || out_c <= 0 ||
        kh <= 0 || kw <= 0 || out_h <= 0 || out_w <= 0 || sy <= 0 || sx <= 0 ||
        dy <= 0 || dx <= 0 || groups <= 0 || relu > 2) return 0;
    if (!(groups == 1 || groups == c)) return 0;
    if (groups == c && out_c % c) return 0;
    size_t in_bytes = (size_t)n * h * width * c * sizeof(float);
    size_t w_elems = groups == c ? (size_t)kh * kw * c * (out_c / c) : (size_t)kh * kw * c * out_c;
    size_t w_bytes = w_elems * sizeof(float);
    size_t out_bytes = (size_t)n * out_h * out_w * out_c * sizeof(float);
    size_t bias_bytes = (size_t)out_c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* wt = graph_ensure_device(w, w_bytes, 1);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !wt || !dst) return 0;
    GLuint temp_bias = 0;
    GLuint bias_buffer = 0;
    if (bptr) {
        OglTensorSlot* bs = graph_ensure_device(bptr, bias_bytes, 1);
        if (!bs) return 0;
        bias_buffer = bs->buffer;
    } else {
        float* zeros = (float*)calloc((size_t)out_c, sizeof(float));
        if (!zeros) return 0;
        temp_bias = create_buffer(bias_bytes, zeros);
        free(zeros);
        if (!temp_bias) return 0;
        bias_buffer = temp_bias;
    }
    uint32_t params[17] = {
        (uint32_t)n, (uint32_t)h, (uint32_t)width, (uint32_t)c,
        (uint32_t)out_c, (uint32_t)out_h, (uint32_t)out_w,
        (uint32_t)kh, (uint32_t)kw, (uint32_t)sy, (uint32_t)sx,
        (uint32_t)pt, (uint32_t)pl, (uint32_t)groups, (uint32_t)relu,
        (uint32_t)dy, (uint32_t)dx
    };
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[5] = {src->buffer, wt->buffer, bias_buffer, dst->buffer, pb};
    OglKernel* kernel = &k_conv2d;
    uint32_t gz = (uint32_t)(n * out_c);
    if (groups == 1 && c == 3 && (out_c & 15) == 0) {
        kernel = &k_conv2d_c3out16;
        gz = (uint32_t)(n * (out_c / 16));
    } else if (groups == c && out_c == c && (out_c & 7) == 0) {
        kernel = &k_conv2d_dw8;
        gz = (uint32_t)(n * ((out_c + 7) / 8));
    } else if (groups == 1 && kh == 1 && kw == 1 && sy == 1 && sx == 1 &&
        pt == 0 && pl == 0 && dy == 1 && dx == 1 && out_h == h && out_w == width) {
        if ((out_c & 15) == 0) {
            kernel = &k_conv2d_pw16tile;
            gz = (uint32_t)(n * (out_c / 16));
        } else if ((out_c & 3) == 0) {
            kernel = &k_conv2d_pw8v4;
            gz = (uint32_t)(n * ((out_c + 7) / 8));
        } else if ((out_c & 1) == 0) {
            kernel = &k_conv2d_pw8v2;
            gz = (uint32_t)(n * ((out_c + 7) / 8));
        } else {
            kernel = &k_conv2d_pw8;
            gz = (uint32_t)(n * ((out_c + 7) / 8));
        }
    }
    uint32_t gx = ((uint32_t)out_w + 7u) / 8u;
    uint32_t gy = ((uint32_t)out_h + 7u) / 8u;
    int ok = dispatch_kernel(kernel, bufs, gx, gy, gz);
    if (!ok && kernel != &k_conv2d) {
        if (kernel == &k_conv2d_pw16tile) {
            ok = dispatch_kernel(&k_conv2d_pw16, bufs, gx, gy, (uint32_t)(n * (out_c / 16)));
        }
        if (!ok && kernel == &k_conv2d_c3out16) ok = dispatch_kernel(&k_conv2d, bufs, gx, gy, (uint32_t)(n * out_c));
        if (!ok && (kernel == &k_conv2d_pw8v4 || kernel == &k_conv2d_pw8v2)) {
            ok = dispatch_kernel(&k_conv2d_pw8, bufs, gx, gy, (uint32_t)(n * ((out_c + 7) / 8)));
        }
        if (!ok && kernel == &k_conv2d_pw16) ok = dispatch_kernel(&k_conv2d_pw8, bufs, gx, gy, (uint32_t)(n * ((out_c + 7) / 8)));
        if (!ok) ok = dispatch_kernel(&k_conv2d, bufs, gx, gy, (uint32_t)(n * out_c));
    }
    if (pb) p_glDeleteBuffers(1, &pb);
    if (temp_bias) p_glDeleteBuffers(1, &temp_bias);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

void opengl_matmul(const float* in, const float* w, const float* b, float* out, int seq, int d_in, int d_out) {
    if (!ogl_ready() || !in || !w || !out || seq <= 0 || d_in <= 0 || d_out <= 0) return;
    size_t in_bytes = (size_t)seq * d_in * sizeof(float);
    size_t w_bytes = (size_t)d_in * d_out * sizeof(float);
    size_t b_bytes = (size_t)d_out * sizeof(float);
    size_t out_bytes = (size_t)seq * d_out * sizeof(float);
    float* zeros = NULL;
    if (!b) {
        zeros = (float*)calloc((size_t)d_out, sizeof(float));
        b = zeros;
    }
    GLuint ib = create_buffer(in_bytes, in);
    GLuint wb = create_buffer(w_bytes, w);
    GLuint bb = create_buffer(b_bytes, b);
    GLuint ob = create_buffer(out_bytes, NULL);
    free(zeros);
    uint32_t params[3] = {(uint32_t)seq, (uint32_t)d_in, (uint32_t)d_out};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[5] = {ib, wb, bb, ob, pb};
    if (dispatch_kernel(&k_matmul, bufs, ((uint32_t)d_out + 15u) / 16u, ((uint32_t)seq + 15u) / 16u, 1)) {
        p_glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, ob);
        read_buffer(GL_SHADER_STORAGE_BUFFER, out_bytes, out);
        p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
    GLuint dels[5] = {ib, wb, bb, ob, pb};
    p_glDeleteBuffers(5, dels);
}

void opengl_free_weight_cache(void) {
}

void opengl_cleanup(void) {
    opengl_graph_reset();
    OglKernel* kernels[] = {&k_copy, &k_add, &k_add3, &k_clip, &k_sigmoid, &k_concat, &k_concat_sigmoid,
                            &k_upsample, &k_resize, &k_maxpool, &k_conv2d,
                            &k_conv2d_c3out16, &k_conv2d_dw4,
                            &k_conv2d_dw8, &k_conv2d_pw8, &k_conv2d_pw8v2,
                            &k_conv2d_pw8v4, &k_conv2d_pw16,
                            &k_conv2d_pw16tile, &k_matmul};
    if (p_glDeleteProgram) {
        for (unsigned i = 0; i < sizeof(kernels) / sizeof(kernels[0]); i++) {
            if (kernels[i]->program) p_glDeleteProgram(kernels[i]->program);
            kernels[i]->program = 0;
            kernels[i]->ready = 0;
            kernels[i]->failed = 0;
        }
    }
    if (egl_display != EGL_NO_DISPLAY) {
        if (p_eglMakeCurrent) p_eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (p_eglDestroyContext && egl_context != EGL_NO_CONTEXT) p_eglDestroyContext(egl_display, egl_context);
        if (p_eglDestroySurface && egl_surface != EGL_NO_SURFACE) p_eglDestroySurface(egl_display, egl_surface);
        if (p_eglTerminate) p_eglTerminate(egl_display);
    }
    egl_context = EGL_NO_CONTEXT;
    egl_surface = EGL_NO_SURFACE;
    egl_display = EGL_NO_DISPLAY;
#ifdef _WIN32
    if (gl_lib) FreeLibrary((HMODULE)gl_lib);
    if (egl_lib) FreeLibrary((HMODULE)egl_lib);
#else
    if (gl_lib) dlclose(gl_lib);
    if (egl_lib) dlclose(egl_lib);
#endif
    gl_lib = NULL;
    egl_lib = NULL;
}
