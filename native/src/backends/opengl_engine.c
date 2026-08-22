#include "opengl_engine.h"
#include "engine_internal.h"
#include "shader_store.h"
#include "batch_matmul_f32_plan.h"
#include "expand_f32_plan.h"
#include "qbatch_matmul_plan.h"
#include "qlinear_multiplier.h"
#include "typed_control_plan.h"

#include <stdint.h>
#include <limits.h>
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
typedef int64_t GLint64;

typedef struct OglKernel OglKernel;
typedef struct {
    const OglKernel* descriptor;
    GLuint program;
    int ready;
    int failed;
} OglProgramCacheEntry;

#define OGL_MAX_PROGRAM_CACHE 256
#define OGL_GRAPH_MAX_TENSORS 4096
/* The widest kernel here binds 8. Dispatch refuses anything past this rather
 * than sizing a stack array from a descriptor field. */
#define OGL_MAX_BINDINGS 16

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
#define GL_MAX_UNIFORM_BUFFER_BINDINGS 0x8A2F
#define GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS 0x90DD
#define GL_MAX_SHADER_STORAGE_BLOCK_SIZE 0x90DE
#define GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT 0x90DF
#define GL_MAX_UNIFORM_BLOCK_SIZE 0x8A30
#define GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS 0x90DB
#define GL_MAX_COMPUTE_UNIFORM_BLOCKS 0x91BB
#define GL_MAX_COMPUTE_WORK_GROUP_COUNT 0x91BE
#define GL_MAX_COMPUTE_WORK_GROUP_SIZE 0x91BF
#define GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS 0x90EB
#define GL_NO_ERROR 0

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

/* One EGL context represents one physical OpenGL submission domain. It and
 * its model-independent program cache are process-shared by necessity, but
 * every access is serialized by this named device mutex. Graph residency,
 * scratch, and training state live in OpenGLContextState below. */
typedef struct {
    pthread_mutex_t mutex;
    unsigned reference_count;
    /* The EGL context is shared across engine states and released at process
     * exit, so the teardown hook is registered exactly once. */
    int exit_hook_registered;
    void* gl_lib;
    void* egl_lib;
    EGLDisplay egl_display;
    EGLContext egl_context;
    EGLSurface egl_surface;
    OpenGLComputeCapability compute_capability;
    GLint max_ssbo_bindings;
    GLint max_uniform_bindings;
    GLint max_compute_groups[3];
    GLint max_compute_group_size[3];
    GLint max_compute_invocations;
    GLint64 max_ssbo_block_size;
    GLint64 max_uniform_block_size;
    /* GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT. Unlike Vulkan there is no
     * arena granularity to fold in: every slot owns a buffer that starts at
     * zero, so the driver limit is the whole constraint on a row binding. */
    GLint ssbo_offset_alignment;
    int qconv_tiled_enabled;
    int profile_sync;
#if VOLVOXAI_ENABLE_TRAINING
    GLint max_compute_ssbo_blocks;
    GLint max_compute_uniform_blocks;
#endif
    EGLDisplay (*p_eglGetDisplay)(void*);
    EGLDisplay (*p_eglGetPlatformDisplayEXT)(EGLint, void*, const EGLint*);
    EGLBoolean (*p_eglInitialize)(EGLDisplay, EGLint*, EGLint*);
    EGLBoolean (*p_eglBindAPI)(EGLint);
    EGLBoolean (*p_eglChooseConfig)(EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*);
    EGLSurface (*p_eglCreatePbufferSurface)(EGLDisplay, EGLConfig, const EGLint*);
    EGLContext (*p_eglCreateContext)(EGLDisplay, EGLConfig, EGLContext, const EGLint*);
    EGLBoolean (*p_eglMakeCurrent)(EGLDisplay, EGLSurface, EGLSurface, EGLContext);
    EGLBoolean (*p_eglDestroyContext)(EGLDisplay, EGLContext);
    EGLBoolean (*p_eglDestroySurface)(EGLDisplay, EGLSurface);
    EGLBoolean (*p_eglTerminate)(EGLDisplay);
    void* (*p_eglGetProcAddress)(const char*);
    const GLubyte* (*p_glGetString)(GLenum);
    void (*p_glGetIntegerv)(GLenum, GLint*);
    void (*p_glGetIntegeri_v)(GLenum, GLuint, GLint*);
    void (*p_glGetInteger64v)(GLenum, GLint64*);
    GLenum (*p_glGetError)(void);
    GLuint (*p_glCreateShader)(GLenum);
    void (*p_glShaderSource)(GLuint, GLsizei, const GLchar* const*, const GLint*);
    void (*p_glCompileShader)(GLuint);
    void (*p_glGetShaderiv)(GLuint, GLenum, GLint*);
    void (*p_glGetShaderInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*);
    void (*p_glDeleteShader)(GLuint);
    GLuint (*p_glCreateProgram)(void);
    void (*p_glAttachShader)(GLuint, GLuint);
    void (*p_glLinkProgram)(GLuint);
    void (*p_glGetProgramiv)(GLuint, GLenum, GLint*);
    void (*p_glGetProgramInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*);
    void (*p_glDeleteProgram)(GLuint);
    void (*p_glUseProgram)(GLuint);
    void (*p_glGenBuffers)(GLsizei, GLuint*);
    void (*p_glBindBuffer)(GLenum, GLuint);
    void (*p_glBufferData)(GLenum, GLsizeiptr, const void*, GLenum);
    void (*p_glBufferSubData)(GLenum, GLintptr, GLsizeiptr, const void*);
    void (*p_glBindBufferBase)(GLenum, GLuint, GLuint);
    void (*p_glBindBufferRange)(GLenum, GLuint, GLuint, GLintptr, GLsizeiptr);
    void (*p_glDeleteBuffers)(GLsizei, const GLuint*);
    void (*p_glDispatchCompute)(GLuint, GLuint, GLuint);
    void (*p_glMemoryBarrier)(GLbitfield);
    void (*p_glFinish)(void);
    void (*p_glGetBufferSubData)(GLenum, GLintptr, GLsizeiptr, void*);
    void* (*p_glMapBufferRange)(GLenum, GLintptr, GLsizeiptr, GLbitfield);
    GLboolean (*p_glUnmapBuffer)(GLenum);
    OglProgramCacheEntry program_cache[OGL_MAX_PROGRAM_CACHE];
    size_t program_cache_count;
} OpenGLDeviceState;

static OpenGLDeviceState g_opengl_device_state = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .egl_display = EGL_NO_DISPLAY,
    .egl_context = EGL_NO_CONTEXT,
    .egl_surface = EGL_NO_SURFACE,
    .compute_capability = {OPENGL_COMPUTE_API_NONE, 0, 0, 0},
    .qconv_tiled_enabled = 1,
    .profile_sync = -1,
};

#define OGL_DEVICE_FIELD(name) g_opengl_device_state.name
#define gl_lib OGL_DEVICE_FIELD(gl_lib)
#define egl_lib OGL_DEVICE_FIELD(egl_lib)
#define egl_display OGL_DEVICE_FIELD(egl_display)
#define egl_context OGL_DEVICE_FIELD(egl_context)
#define egl_surface OGL_DEVICE_FIELD(egl_surface)
#define compute_capability OGL_DEVICE_FIELD(compute_capability)
#define max_ssbo_bindings OGL_DEVICE_FIELD(max_ssbo_bindings)
#define max_uniform_bindings OGL_DEVICE_FIELD(max_uniform_bindings)
#define max_compute_groups OGL_DEVICE_FIELD(max_compute_groups)
#define max_compute_group_size OGL_DEVICE_FIELD(max_compute_group_size)
#define max_compute_invocations OGL_DEVICE_FIELD(max_compute_invocations)
#define max_ssbo_block_size OGL_DEVICE_FIELD(max_ssbo_block_size)
#define max_uniform_block_size OGL_DEVICE_FIELD(max_uniform_block_size)
#define graph_alignment OGL_DEVICE_FIELD(ssbo_offset_alignment)
#define qconv_tiled_enabled OGL_DEVICE_FIELD(qconv_tiled_enabled)
#define profile_sync OGL_DEVICE_FIELD(profile_sync)
#if VOLVOXAI_ENABLE_TRAINING
#define max_compute_ssbo_blocks OGL_DEVICE_FIELD(max_compute_ssbo_blocks)
#define max_compute_uniform_blocks OGL_DEVICE_FIELD(max_compute_uniform_blocks)
#endif
#define p_eglGetDisplay OGL_DEVICE_FIELD(p_eglGetDisplay)
#define p_eglGetPlatformDisplayEXT OGL_DEVICE_FIELD(p_eglGetPlatformDisplayEXT)
#define p_eglInitialize OGL_DEVICE_FIELD(p_eglInitialize)
#define p_eglBindAPI OGL_DEVICE_FIELD(p_eglBindAPI)
#define p_eglChooseConfig OGL_DEVICE_FIELD(p_eglChooseConfig)
#define p_eglCreatePbufferSurface OGL_DEVICE_FIELD(p_eglCreatePbufferSurface)
#define p_eglCreateContext OGL_DEVICE_FIELD(p_eglCreateContext)
#define p_eglMakeCurrent OGL_DEVICE_FIELD(p_eglMakeCurrent)
#define p_eglDestroyContext OGL_DEVICE_FIELD(p_eglDestroyContext)
#define p_eglDestroySurface OGL_DEVICE_FIELD(p_eglDestroySurface)
#define p_eglTerminate OGL_DEVICE_FIELD(p_eglTerminate)
#define p_eglGetProcAddress OGL_DEVICE_FIELD(p_eglGetProcAddress)
#define p_glGetString OGL_DEVICE_FIELD(p_glGetString)
#define p_glGetIntegerv OGL_DEVICE_FIELD(p_glGetIntegerv)
#define p_glGetIntegeri_v OGL_DEVICE_FIELD(p_glGetIntegeri_v)
#define p_glGetInteger64v OGL_DEVICE_FIELD(p_glGetInteger64v)
#define p_glGetError OGL_DEVICE_FIELD(p_glGetError)
#define p_glCreateShader OGL_DEVICE_FIELD(p_glCreateShader)
#define p_glShaderSource OGL_DEVICE_FIELD(p_glShaderSource)
#define p_glCompileShader OGL_DEVICE_FIELD(p_glCompileShader)
#define p_glGetShaderiv OGL_DEVICE_FIELD(p_glGetShaderiv)
#define p_glGetShaderInfoLog OGL_DEVICE_FIELD(p_glGetShaderInfoLog)
#define p_glDeleteShader OGL_DEVICE_FIELD(p_glDeleteShader)
#define p_glCreateProgram OGL_DEVICE_FIELD(p_glCreateProgram)
#define p_glAttachShader OGL_DEVICE_FIELD(p_glAttachShader)
#define p_glLinkProgram OGL_DEVICE_FIELD(p_glLinkProgram)
#define p_glGetProgramiv OGL_DEVICE_FIELD(p_glGetProgramiv)
#define p_glGetProgramInfoLog OGL_DEVICE_FIELD(p_glGetProgramInfoLog)
#define p_glDeleteProgram OGL_DEVICE_FIELD(p_glDeleteProgram)
#define p_glUseProgram OGL_DEVICE_FIELD(p_glUseProgram)
#define p_glGenBuffers OGL_DEVICE_FIELD(p_glGenBuffers)
#define p_glBindBuffer OGL_DEVICE_FIELD(p_glBindBuffer)
#define p_glBufferData OGL_DEVICE_FIELD(p_glBufferData)
#define p_glBufferSubData OGL_DEVICE_FIELD(p_glBufferSubData)
#define p_glBindBufferBase OGL_DEVICE_FIELD(p_glBindBufferBase)
#define p_glBindBufferRange OGL_DEVICE_FIELD(p_glBindBufferRange)
#define p_glDeleteBuffers OGL_DEVICE_FIELD(p_glDeleteBuffers)
#define p_glDispatchCompute OGL_DEVICE_FIELD(p_glDispatchCompute)
#define p_glMemoryBarrier OGL_DEVICE_FIELD(p_glMemoryBarrier)
#define p_glFinish OGL_DEVICE_FIELD(p_glFinish)
#define p_glGetBufferSubData OGL_DEVICE_FIELD(p_glGetBufferSubData)
#define p_glMapBufferRange OGL_DEVICE_FIELD(p_glMapBufferRange)
#define p_glUnmapBuffer OGL_DEVICE_FIELD(p_glUnmapBuffer)

struct OglKernel {
    const char* name;
    const char* path;
    int binding_count;
    int uniform_binding;
};

#ifdef __ANDROID__
#define OGL_SHADER_DIR "gles"
#else
#define OGL_SHADER_DIR "glsl"
#endif

void opengl_set_shader_root(const char* root) {
    if (volvoxai_shader_store_set_override_root(root) != VOLVOXAI_SHADER_STORE_OK) {
        fprintf(stderr, "[OpenGL] invalid configured shader root\n");
    }
}

typedef struct {
    const void* host;
    size_t bytes;
    size_t cap;
    size_t domain_capacity;
    GLuint buffer;
    uint64_t shape_generation;
    uint64_t capacity_generation;
    int host_dirty;
    int device_dirty;
    int is_weight;
    int owns_buffer;
    int domain_span;
} OglTensorSlot;

/* NULL canonical QConv2D biases use an output-channel-sized, zero-filled host
 * key. Retaining prior blocks prevents a graph slot from observing a dangling
 * pointer when a later convolution needs a larger channel count. */
typedef struct QConvZeroBiasBacking {
    int32_t* values;
    size_t elements;
    struct QConvZeroBiasBacking* next;
} QConvZeroBiasBacking;

typedef struct {
    OglTensorSlot graph_slot_storage[OGL_GRAPH_MAX_TENSORS];
    int graph_slots_count;
    /* These statistics buffers have no host tensor identity or readback path;
     * they are scratch owned by one execution context. */
    GLuint qgroupnorm_scratch_buffer;
    size_t qgroupnorm_scratch_capacity;
    GLuint qlayernorm_scratch_buffer;
    size_t qlayernorm_scratch_capacity;
    /* One name is sufficient because every upload below replaces its data
     * store. Queued dispatches retain the store that was current when their
     * command was recorded. */
    GLuint dispatch_params_buffer;
    uint64_t dispatch_params_upload_count;
#ifdef VOLVOX_OPENGL_TESTING
    uint64_t test_conv_out16_dispatch_count;
#endif
    QConvZeroBiasBacking* qconv_zero_bias_storage;
    char* shape_signature;
    uint64_t shape_generation;
    uint64_t capacity_generation;
    size_t domain_span_count;
    size_t domain_qgroupnorm_stats_bytes;
    size_t domain_qlayernorm_stats_bytes;
    int domain_enforced;
#ifdef VOLVOX_OPENGL_TESTING
    int test_domain_allocation_failure_after;
#endif
#if VOLVOXAI_ENABLE_TRAINING
    OglTensorSlot training_slot_storage[OGL_GRAPH_MAX_TENSORS];
    int training_slots_count;
    int training_is_active;
#endif
    int forward_active;
    int implicit_forward;
    int transient_active;
    int device_acquired;
} OpenGLContextState;

static OpenGLContextState* opengl_context_state_get(int create);
static void opengl_context_state_destroy(void* opaque_state);
static uint64_t opengl_generation_next(uint64_t generation);

#define graph_slots (opengl_context_state_get(0)->graph_slot_storage)
#define graph_slot_count (opengl_context_state_get(0)->graph_slots_count)
#define qgroupnorm_stats_buffer \
    (opengl_context_state_get(0)->qgroupnorm_scratch_buffer)
#define qgroupnorm_stats_capacity \
    (opengl_context_state_get(0)->qgroupnorm_scratch_capacity)
#define qlayernorm_stats_buffer \
    (opengl_context_state_get(0)->qlayernorm_scratch_buffer)
#define qlayernorm_stats_capacity \
    (opengl_context_state_get(0)->qlayernorm_scratch_capacity)
#define qconv_zero_bias_backings \
    (opengl_context_state_get(0)->qconv_zero_bias_storage)
static const int32_t* qconv_zero_bias_get(uint32_t output_channels);
static void qconv_zero_bias_release_state(OpenGLContextState* state);

/* Binding 3 of qSDPAInt8 is always an I32 keep-mask buffer. */
static const int32_t qsdpa_dummy_mask[1] = {0};
static const float linear_dummy_scale[1] = {1.0f};

static const OglKernel k_copy = {"copy", OGL_SHADER_DIR "/copy.comp", 3, 2};
static const OglKernel k_add = {"addRelu", OGL_SHADER_DIR "/addRelu.comp", 4, 3};
static const OglKernel k_add3 = {"add3Relu", OGL_SHADER_DIR "/add3Relu.comp", 5, 4};
static const OglKernel k_clip = {"clip", OGL_SHADER_DIR "/clip.comp", 3, 2};
static const OglKernel k_sigmoid = {"sigmoid", OGL_SHADER_DIR "/sigmoid.comp", 3, 2};
static const OglKernel k_relu = {"reLU", OGL_SHADER_DIR "/reLU.comp", 3, 2};
static const OglKernel k_gelu = {"gELU", OGL_SHADER_DIR "/gELU.comp", 3, 2};
static const OglKernel k_silu = {"siLU", OGL_SHADER_DIR "/siLU.comp", 3, 2};
static const OglKernel k_tanh = {"tanh", OGL_SHADER_DIR "/tanh.comp", 3, 2};
static const OglKernel k_hardswish = {"hardSwish", OGL_SHADER_DIR "/hardSwish.comp", 3, 2};
static const OglKernel k_hardsigmoid = {"hardSigmoid", OGL_SHADER_DIR "/hardSigmoid.comp", 3, 2};
static const OglKernel k_leaky_relu = {"leakyReLU", OGL_SHADER_DIR "/leakyReLU.comp", 3, 2};
static const OglKernel k_prelu = {"pReLU", OGL_SHADER_DIR "/pReLU.comp", 4, 3};
static const OglKernel k_layernorm = {"layerNorm", OGL_SHADER_DIR "/layerNorm.comp", 5, 4};
static const OglKernel k_rmsnorm = {"rMSNorm", OGL_SHADER_DIR "/rMSNorm.comp", 4, 3};
static const OglKernel k_softmax = {"softmax", OGL_SHADER_DIR "/softmax.comp", 3, 2};
static const OglKernel k_logsoftmax = {"logSoftmax", OGL_SHADER_DIR "/logSoftmax.comp", 3, 2};
static const OglKernel k_reduce = {"reduce", OGL_SHADER_DIR "/reduce.comp", 3, 2};
static const OglKernel k_globalavg = {"globalAveragePool", OGL_SHADER_DIR "/globalAveragePool.comp", 3, 2};
static const OglKernel k_avgpool = {"averagePool2D", OGL_SHADER_DIR "/averagePool2D.comp", 3, 2};
static const OglKernel k_batchnorm = {"batchNorm2D", OGL_SHADER_DIR "/batchNorm2D.comp", 7, 6};
static const OglKernel k_embedding = {"embedding", OGL_SHADER_DIR "/embedding.comp", 4, 3};
/* Bindings 0-5 storage, 6 uniform params, 7 the resident-slot table. */
static const OglKernel k_moe_linear = {"moeLinear", OGL_SHADER_DIR "/moeLinear.comp", 8, 6};
/* Bindings 0-4 storage, 5 uniform params. */
static const OglKernel k_moe_router = {"moeRouter", OGL_SHADER_DIR "/moeRouter.comp", 6, 5};
static const OglKernel k_transpose = {"generalTranspose", OGL_SHADER_DIR "/generalTranspose.comp", 3, -1};
static const OglKernel k_where = {"where", OGL_SHADER_DIR "/where.comp", 5, 4};
static const OglKernel k_typed_control_32 = {"typedControl32Native", OGL_SHADER_DIR "/typedControl32Native.comp", 4, -1};
static const OglKernel k_where_32 = {"where32Native", OGL_SHADER_DIR "/where32Native.comp", 5, 4};
static const OglKernel k_argmax_f32_i32 = {"argMaxF32I32Native", OGL_SHADER_DIR "/argMaxF32I32Native.comp", 3, 2};
static const OglKernel k_concat_32 = {"concatCopy32Native", OGL_SHADER_DIR "/concatCopy32Native.comp", 3, 2};
static const OglKernel k_expand = {"expand", OGL_SHADER_DIR "/expand.comp", 3, 2};
static const OglKernel k_batch_matmul = {"batchMatMul", OGL_SHADER_DIR "/batchMatMul.comp", 4, -1};
static const OglKernel k_pad = {"pad", OGL_SHADER_DIR "/pad.comp", 3, 2};
static const OglKernel k_slice = {"slice", OGL_SHADER_DIR "/slice.comp", 3, 2};
static const OglKernel k_gather = {"gather", OGL_SHADER_DIR "/gather.comp", 4, 3};
static const OglKernel k_convtranspose = {"convTranspose2D", OGL_SHADER_DIR "/convTranspose2D.comp", 5, 4};
static const OglKernel k_interp1d = {"interp1D", OGL_SHADER_DIR "/interp1D.comp", 3, 2};
static const OglKernel k_mul = {"mul", OGL_SHADER_DIR "/mul.comp", 4, 3};
static const OglKernel k_sub = {"sub", OGL_SHADER_DIR "/sub.comp", 4, 3};
static const OglKernel k_div = {"div", OGL_SHADER_DIR "/div.comp", 4, 3};
static const OglKernel k_broadcast_binary = {"broadcastBinaryNative", OGL_SHADER_DIR "/broadcastBinaryNative.comp", 4, -1};
static const OglKernel k_split = {"split", OGL_SHADER_DIR "/split.comp", 3, 2};
static const OglKernel k_conv1d = {"conv1D", OGL_SHADER_DIR "/conv1D.comp", 5, 4};
static const OglKernel k_sdpa = {"sDPA", OGL_SHADER_DIR "/sDPA.comp", 4, 3};
static const OglKernel k_cross_sdpa = {"crossSDPA", OGL_SHADER_DIR "/crossSDPA.comp", 6, 5};
#if VOLVOXAI_ENABLE_TRAINING
static const OglKernel k_sdpa_training = {"sdpaTraining", OGL_SHADER_DIR "/sdpaTraining.comp", 4, 3};
static const OglKernel k_cross_sdpa_training = {"crossSdpaTraining", OGL_SHADER_DIR "/crossSdpaTraining.comp", 6, 5};
#endif
static const OglKernel k_cross_attention = {"crossAttentionF32", OGL_SHADER_DIR "/crossAttentionF32.comp", 7, 6};
static const OglKernel k_quantize = {"quantizeLinear", OGL_SHADER_DIR "/quantizeLinear.comp", 3, 2};
static const OglKernel k_dequantize = {"dequantizeLinear", OGL_SHADER_DIR "/dequantizeLinear.comp", 5, 4};
static const OglKernel k_qlinear_int8 = {"qLinearInt8", OGL_SHADER_DIR "/qLinearInt8.comp", 7, 6};
static const OglKernel k_qlinear_int8_tiled = {"qLinearInt8Tiled", OGL_SHADER_DIR "/qLinearInt8Tiled.comp", 7, 6};
static const OglKernel k_qembedding_int8 = {"qEmbeddingInt8", OGL_SHADER_DIR "/qEmbeddingInt8.comp", 6, 5};
static const OglKernel k_qconv2d_int8 = {"qConv2DInt8", OGL_SHADER_DIR "/qConv2DInt8.comp", 7, 6};
static const OglKernel k_qconv2d_int8_tiled = {"qConv2DInt8Tiled", OGL_SHADER_DIR "/qConv2DInt8Tiled.comp", 7, 6};
static const OglKernel k_quantize_typed_i8u8 = {"quantizeLinearTyped", OGL_SHADER_DIR "/quantizeLinearTyped.comp", 5, 4};
static const OglKernel k_dequantize_typed_i8u8 = {"dequantizeLinearTyped", OGL_SHADER_DIR "/dequantizeLinearTyped.comp", 5, 4};
static const OglKernel k_qadd_i8u8 = {"qAdd", OGL_SHADER_DIR "/qAdd.comp", 4, 3};
static const OglKernel k_qbatch_matmul_i8u8 = {
    "qBatchMatMul", OGL_SHADER_DIR "/qBatchMatMul.comp", 5, 4
};
static const OglKernel k_qsilu_i8u8 = {"qSiLUInt8", OGL_SHADER_DIR "/qSiLUInt8.comp", 3, 2};
static const OglKernel k_qgelu_i8u8 = {"qGELUInt8", OGL_SHADER_DIR "/qGELUInt8.comp", 3, 2};
static const OglKernel k_qgroupnorm_stats = {"qGroupNormStats", OGL_SHADER_DIR "/qGroupNormStats.comp", 3, 2};
static const OglKernel k_qgroupnorm_apply = {"qGroupNormApply", OGL_SHADER_DIR "/qGroupNormApply.comp", 6, 5};
static const OglKernel k_qlayernorm_stats = {"qLayerNormStats", OGL_SHADER_DIR "/qLayerNormStats.comp", 3, 2};
static const OglKernel k_qlayernorm_apply = {"qLayerNormApply", OGL_SHADER_DIR "/qLayerNormApply.comp", 6, 5};
static const OglKernel k_qsdpa_int8 = {"qSDPAInt8", OGL_SHADER_DIR "/qSDPAInt8.comp", 6, 5};
static const OglKernel k_qargmax_int8 = {"qArgMaxInt8", OGL_SHADER_DIR "/qArgMaxInt8.comp", 3, 2};
static const OglKernel k_row_index_transfer = {"rowIndexTransfer", OGL_SHADER_DIR "/rowIndexTransfer.comp", 4, 3};
static const OglKernel k_qmaskedmean_int8 = {"qMaskedMeanInt8", OGL_SHADER_DIR "/qMaskedMeanInt8.comp", 4, 3};
static const OglKernel k_requantize_linear_i8u8 = {"requantizeLinearTyped", OGL_SHADER_DIR "/requantizeLinearTyped.comp", 3, 2};
static const OglKernel k_copy_typed_i8u8 = {"copyTyped", OGL_SHADER_DIR "/copyTyped.comp", 3, 2};
static const OglKernel k_concat_typed_i8u8 = {"concatCopyTyped", OGL_SHADER_DIR "/concatCopyTyped.comp", 3, 2};
static const OglKernel k_maxpool_typed_i8u8 = {"maxPool2DTyped", OGL_SHADER_DIR "/maxPool2DTyped.comp", 3, 2};
static const OglKernel k_resize_nearest_typed_i8u8 = {"resizeNearestTyped", OGL_SHADER_DIR "/resizeNearestTyped.comp", 3, 2};
static const OglKernel k_transpose_typed_i8u8 = {"transposeTyped", OGL_SHADER_DIR "/transposeTyped.comp", 3, -1};
static const OglKernel k_spatial_softargmax_y = {"spatialSoftargmaxY", OGL_SHADER_DIR "/spatialSoftargmaxY.comp", 3, 2};
static const OglKernel k_profile_x = {"profileX", OGL_SHADER_DIR "/profileX.comp", 3, 2};
static const OglKernel k_profile_y = {"profileY", OGL_SHADER_DIR "/profileY.comp", 3, 2};
static const OglKernel k_mean_height = {"meanHeight", OGL_SHADER_DIR "/meanHeight.comp", 3, 2};
static const OglKernel k_nms = {"nonMaxSuppression", OGL_SHADER_DIR "/nonMaxSuppression.comp", 4, 3};
static const OglKernel k_concat = {"concatCopy", OGL_SHADER_DIR "/concatCopy.comp", 3, 2};
static const OglKernel k_concat_sigmoid = {"concatSigmoidCopy", OGL_SHADER_DIR "/concatSigmoidCopy.comp", 3, 2};
static const OglKernel k_upsample = {"upsample2x", OGL_SHADER_DIR "/upsample2x.comp", 3, 2};
static const OglKernel k_resize = {"resize", OGL_SHADER_DIR "/resize.comp", 3, 2};
static const OglKernel k_maxpool = {"maxPool2D", OGL_SHADER_DIR "/maxPool2D.comp", 3, 2};
static const OglKernel k_conv2d = {"conv2D", OGL_SHADER_DIR "/conv2D.comp", 5, 4};
static const OglKernel k_groupnorm = {"groupNorm", OGL_SHADER_DIR "/groupNorm.comp", 5, 4};
#if VOLVOXAI_ENABLE_TRAINING
static const OglKernel k_dropout = {"dropout", OGL_SHADER_DIR "/dropout.comp", 3, 2};
#endif
static const OglKernel k_conv2d_c3out16 = {"conv2DRegularC3Out16", OGL_SHADER_DIR "/conv2DRegularC3Out16.comp", 5, 4};
static const OglKernel k_conv2d_out16 = {"conv2DRegularOut16", OGL_SHADER_DIR "/conv2DRegularOut16.comp", 5, 4};
static const OglKernel k_conv2d_dw4 = {"conv2DDepthwise4", OGL_SHADER_DIR "/conv2DDepthwise4.comp", 5, 4};
static const OglKernel k_conv2d_dw8 = {"conv2DDepthwise8", OGL_SHADER_DIR "/conv2DDepthwise8.comp", 5, 4};
static const OglKernel k_conv2d_pw8 = {"conv2DPointwise8", OGL_SHADER_DIR "/conv2DPointwise8.comp", 5, 4};
static const OglKernel k_conv2d_pw8v2 = {"conv2DPointwise8Vec2", OGL_SHADER_DIR "/conv2DPointwise8Vec2.comp", 5, 4};
static const OglKernel k_conv2d_pw8v4 = {"conv2DPointwise8Vec4", OGL_SHADER_DIR "/conv2DPointwise8Vec4.comp", 5, 4};
static const OglKernel k_conv2d_pw16 = {"conv2DPointwise16", OGL_SHADER_DIR "/conv2DPointwise16.comp", 5, 4};
static const OglKernel k_conv2d_pw16tile = {"conv2DPointwise16Tile", OGL_SHADER_DIR "/conv2DPointwise16Tile.comp", 5, 4};
static const OglKernel k_matmul = {"linearF32RowMajor", OGL_SHADER_DIR "/linearF32RowMajor.comp", 5, 4};
static const OglKernel k_matmul_tiled = {"linearF32RowMajorTiled", OGL_SHADER_DIR "/linearF32RowMajorTiled.comp", 5, 4};
static const OglKernel k_matmul_out_in = {"linearF32", OGL_SHADER_DIR "/linearF32.comp", 6, 5};
static const OglKernel k_matmul_out_in_tiled = {"linearF32Tiled", OGL_SHADER_DIR "/linearF32Tiled.comp", 6, 5};

#if VOLVOXAI_ENABLE_TRAINING
#define OGL_TRAINING_MAX_BINDINGS 16

typedef struct {
    const char* shader_name;
    const char* entry_point;
    uint32_t read_write_mask;
    OglKernel kernel;
} OglTrainingKernel;

#define OGL_TRAINING_KERNEL(shader, entry, file, bindings, uniform, rw_mask) \
    {shader, entry, rw_mask, \
     {shader "/" entry, OGL_SHADER_DIR "/" file ".comp", bindings, uniform}}

static const OglTrainingKernel training_kernels[] = {
    OGL_TRAINING_KERNEL("activationBackward", "main", "activationBackward", 5, 4, 1u << 3),
    OGL_TRAINING_KERNEL("basicBackward", "a_main", "basicBackward_a_main", 6, -1, (1u << 3) | (1u << 4)),
    OGL_TRAINING_KERNEL("basicBackward", "b_main", "basicBackward_b_main", 6, -1, (1u << 3) | (1u << 4)),
    OGL_TRAINING_KERNEL("batchNorm2DBackward", "input_main", "batchNorm2DBackward_input_main", 9, 8, (1u << 5) | (1u << 6) | (1u << 7)),
    OGL_TRAINING_KERNEL("batchNorm2DBackward", "param_main", "batchNorm2DBackward_param_main", 9, 8, (1u << 5) | (1u << 6) | (1u << 7)),
    OGL_TRAINING_KERNEL("groupNormBackward", "input_main", "groupNormBackward_input_main", 7, 6, (1u << 3) | (1u << 4) | (1u << 5)),
    OGL_TRAINING_KERNEL("groupNormBackward", "param_main", "groupNormBackward_param_main", 7, 6, (1u << 3) | (1u << 4) | (1u << 5)),
    OGL_TRAINING_KERNEL("concatBackward", "main", "concatBackward", 3, 2, 1u << 1),
    OGL_TRAINING_KERNEL("conv2DBackward", "input_main", "conv2DBackward_input_main", 8, 7, (1u << 4) | (1u << 5) | (1u << 6)),
    OGL_TRAINING_KERNEL("conv2DBackward", "weight_main", "conv2DBackward_weight_main", 8, 7, (1u << 4) | (1u << 5) | (1u << 6)),
    OGL_TRAINING_KERNEL("conv2DBackward", "bias_main", "conv2DBackward_bias_main", 8, 7, (1u << 4) | (1u << 5) | (1u << 6)),
    OGL_TRAINING_KERNEL("copyBackward", "main", "copyBackward", 3, 2, 1u << 1),
    OGL_TRAINING_KERNEL("dropoutBackward", "main", "dropoutBackward", 3, 2, 1u << 1),
    OGL_TRAINING_KERNEL("reduceBackward", "main", "reduceBackward", 3, 2, 1u << 1),
    OGL_TRAINING_KERNEL("crossSdpaBackward", "q_main", "crossSdpaBackward_q_main", 9, 8, (1u << 5) | (1u << 6) | (1u << 7)),
    OGL_TRAINING_KERNEL("crossSdpaBackward", "k_main", "crossSdpaBackward_k_main", 9, 8, (1u << 5) | (1u << 6) | (1u << 7)),
    OGL_TRAINING_KERNEL("crossSdpaBackward", "v_main", "crossSdpaBackward_v_main", 9, 8, (1u << 5) | (1u << 6) | (1u << 7)),
    OGL_TRAINING_KERNEL("embeddingBackward", "main", "embeddingBackward", 4, 3, 1u << 2),
    OGL_TRAINING_KERNEL("layerNormBackward", "input_main", "layerNormBackward_input_main", 7, 6, (1u << 3) | (1u << 4) | (1u << 5)),
    OGL_TRAINING_KERNEL("layerNormBackward", "param_main", "layerNormBackward_param_main", 7, 6, (1u << 3) | (1u << 4) | (1u << 5)),
    OGL_TRAINING_KERNEL("matMulBackward", "input_main", "matMulBackward_input_main", 7, 6, (1u << 3) | (1u << 4) | (1u << 5)),
    OGL_TRAINING_KERNEL("matMulBackward", "weight_main", "matMulBackward_weight_main", 7, 6, (1u << 3) | (1u << 4) | (1u << 5)),
    OGL_TRAINING_KERNEL("matMulBackward", "bias_main", "matMulBackward_bias_main", 7, 6, (1u << 3) | (1u << 4) | (1u << 5)),
    OGL_TRAINING_KERNEL("moeLinearBackward", "input_main", "moeLinearBackward_input_main", 13, 12, (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9)),
    OGL_TRAINING_KERNEL("moeLinearBackward", "weight_main", "moeLinearBackward_weight_main", 13, 12, (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9)),
    OGL_TRAINING_KERNEL("moeLinearBackward", "bias_main", "moeLinearBackward_bias_main", 13, 12, (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9)),
    OGL_TRAINING_KERNEL("moeLinearBackward", "route_main", "moeLinearBackward_route_main", 13, 12, (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9)),
    OGL_TRAINING_KERNEL("moeRouterBackward", "logit_main", "moeRouterBackward_logit_main", 11, 10, (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9)),
    OGL_TRAINING_KERNEL("moeRouterBackward", "input_main", "moeRouterBackward_input_main", 11, 10, (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9)),
    OGL_TRAINING_KERNEL("moeRouterBackward", "weight_main", "moeRouterBackward_weight_main", 11, 10, (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9)),
    OGL_TRAINING_KERNEL("moeRouterBackward", "bias_main", "moeRouterBackward_bias_main", 11, 10, (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9)),
    OGL_TRAINING_KERNEL("poolingBackward", "global_average_main", "poolingBackward_global_average_main", 4, 3, 1u << 2),
    OGL_TRAINING_KERNEL("poolingBackward", "max_pool_main", "poolingBackward_max_pool_main", 4, 3, 1u << 2),
    OGL_TRAINING_KERNEL("preluBackward", "input_main", "preluBackward_input_main", 6, 5, (1u << 3) | (1u << 4)),
    OGL_TRAINING_KERNEL("preluBackward", "weight_main", "preluBackward_weight_main", 6, 5, (1u << 3) | (1u << 4)),
    OGL_TRAINING_KERNEL("resizeBackward", "main", "resizeBackward", 3, 2, 1u << 1),
    OGL_TRAINING_KERNEL("rmsNormBackward", "input_main", "rmsNormBackward_input_main", 6, 5, (1u << 3) | (1u << 4)),
    OGL_TRAINING_KERNEL("rmsNormBackward", "weight_main", "rmsNormBackward_weight_main", 6, 5, (1u << 3) | (1u << 4)),
    OGL_TRAINING_KERNEL("sdpaBackward", "main", "sdpaBackward", 5, 4, 1u << 3),
    OGL_TRAINING_KERNEL("softmaxBackward", "main", "softmaxBackward", 4, 3, 1u << 2),
    OGL_TRAINING_KERNEL("splitBackward", "main", "splitBackward", 3, 2, 1u << 1),
    OGL_TRAINING_KERNEL("transposeBackward", "main", "transposeBackward", 3, -1, 1u << 1),
    OGL_TRAINING_KERNEL("loraApply", "main", "loraApply", 5, 4, 1u << 3),
};

#define training_slots (opengl_context_state_get(0)->training_slot_storage)
#define training_slot_count (opengl_context_state_get(0)->training_slots_count)
#define training_active (opengl_context_state_get(0)->training_is_active)

#undef OGL_TRAINING_KERNEL
#endif

static OpenGLContextState* opengl_context_state_get(int create) {
    VxEngineState* owner = vx_engine_state_current();
    if (!owner) return NULL;
    OpenGLContextState* state =
        (OpenGLContextState*)owner->opengl_context_state;
    if (!state && create) {
        state = (OpenGLContextState*)calloc(1, sizeof(*state));
        if (!state) return NULL;
        state->shape_generation = 1;
        state->capacity_generation = 1;
#ifdef VOLVOX_OPENGL_TESTING
        state->test_domain_allocation_failure_after = -1;
#endif
        owner->opengl_context_state = state;
        owner->opengl_context_state_destroy = opengl_context_state_destroy;
    }
    return state;
}

static void opengl_device_lock(void) {
    pthread_mutex_lock(&g_opengl_device_state.mutex);
}

static void opengl_device_unlock(void) {
    pthread_mutex_unlock(&g_opengl_device_state.mutex);
}

static int opengl_make_current_locked(void) {
    if (egl_display == EGL_NO_DISPLAY || egl_surface == EGL_NO_SURFACE ||
        egl_context == EGL_NO_CONTEXT || !p_eglMakeCurrent) return -1;
    return p_eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context) ==
        EGL_TRUE ? 0 : -1;
}

static void opengl_release_current_locked(void) {
    if (egl_display != EGL_NO_DISPLAY && p_eglMakeCurrent) {
        (void)p_eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                               EGL_NO_CONTEXT);
    }
}

static void opengl_device_shutdown_locked(void);

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

int opengl_compute_version_supported(OpenGLComputeApi api, int major, int minor) {
    if (major < 0 || minor < 0) return 0;
    if (api == OPENGL_COMPUTE_API_DESKTOP) return major > 4 || (major == 4 && minor >= 3);
    if (api == OPENGL_COMPUTE_API_GLES) return major > 3 || (major == 3 && minor >= 1);
    return 0;
}

int opengl_get_compute_capability(OpenGLComputeCapability* out) {
    if (!out) return -1;
    OpenGLContextState* state = opengl_context_state_get(0);
    int owns_lock = state &&
        (state->forward_active || state->transient_active
#if VOLVOXAI_ENABLE_TRAINING
         || state->training_is_active
#endif
        );
    if (!owns_lock) opengl_device_lock();
    *out = state && state->device_acquired
        ? compute_capability
        : (OpenGLComputeCapability){OPENGL_COMPUTE_API_NONE, 0, 0, 0};
    if (!owns_lock) opengl_device_unlock();
    return 0;
}

int opengl_query_domain_limits(OpenGLDomainLimits* limits) {
    OpenGLContextState* state = opengl_context_state_get(0);
    int owns_lock;
    int ready = 0;
    if (!limits) return -1;
    memset(limits, 0, sizeof(*limits));
    if (!state || !state->device_acquired) return -1;
    owns_lock = state->forward_active || state->transient_active
#if VOLVOXAI_ENABLE_TRAINING
        || state->training_is_active
#endif
        ;
    if (!owns_lock) opengl_device_lock();
    if (compute_capability.compute_supported && max_ssbo_block_size > 0 &&
        max_uniform_block_size > 0) {
        limits->maximum_storage_buffer_bytes =
            (uint64_t)max_ssbo_block_size;
        limits->maximum_uniform_buffer_bytes =
            (uint64_t)max_uniform_block_size;
        for (size_t axis = 0; axis < 3u; axis++) {
            limits->maximum_workgroups[axis] =
                (uint32_t)max_compute_groups[axis];
            limits->maximum_workgroup_size[axis] =
                (uint32_t)max_compute_group_size[axis];
        }
        limits->maximum_workgroup_invocations =
            (uint32_t)max_compute_invocations;
        limits->maximum_storage_bindings = (uint32_t)max_ssbo_bindings;
        limits->maximum_uniform_bindings = (uint32_t)max_uniform_bindings;
        limits->maximum_tensor_slots = OGL_GRAPH_MAX_TENSORS;
        ready = limits->maximum_workgroups[0] > 0u &&
            limits->maximum_workgroups[1] > 0u &&
            limits->maximum_workgroups[2] > 0u &&
            limits->maximum_workgroup_size[0] > 0u &&
            limits->maximum_workgroup_size[1] > 0u &&
            limits->maximum_workgroup_size[2] > 0u &&
            limits->maximum_workgroup_invocations > 0u &&
            limits->maximum_storage_bindings > 0u &&
            limits->maximum_uniform_bindings > 0u;
    }
    if (!owns_lock) opengl_device_unlock();
    return ready ? 0 : -1;
}

static int parse_gl_version(const char* version, OpenGLComputeApi* api, int* major, int* minor) {
    if (!version || !api || !major || !minor) return -1;
    const char* cursor = version;
    if (!strncmp(cursor, "OpenGL ES", 9)) {
        *api = OPENGL_COMPUTE_API_GLES;
        cursor += 9;
    } else {
        *api = OPENGL_COMPUTE_API_DESKTOP;
    }
    while (*cursor && (*cursor < '0' || *cursor > '9')) cursor++;
    if (!*cursor) return -1;
    char* end = NULL;
    long parsed_major = strtol(cursor, &end, 10);
    if (!end || *end != '.') return -1;
    cursor = end + 1;
    long parsed_minor = strtol(cursor, &end, 10);
    if (parsed_major < 0 || parsed_major > 99 || parsed_minor < 0 || parsed_minor > 99) return -1;
    *major = (int)parsed_major;
    *minor = (int)parsed_minor;
    return 0;
}

#define LOAD_EGL(name) do { p_##name = load_symbol(egl_lib, #name); if (!p_##name) return -1; } while (0)
#define LOAD_GL(name) do { p_##name = load_gl_proc(#name); if (!p_##name) return -1; } while (0)

static int load_egl(void) {
#ifdef _WIN32
    if (!egl_lib) egl_lib = LoadLibraryA("libEGL.dll");
    if (!gl_lib) gl_lib = LoadLibraryA("opengl32.dll");
#else
#ifdef __ANDROID__
    if (!egl_lib) egl_lib = dlopen("libEGL.so", RTLD_NOW | RTLD_LOCAL);
    if (!gl_lib) gl_lib = dlopen("libGLESv3.so", RTLD_NOW | RTLD_LOCAL);
    if (!gl_lib) gl_lib = dlopen("libGLESv2.so", RTLD_NOW | RTLD_LOCAL);
#else
    if (!egl_lib) egl_lib = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!egl_lib) egl_lib = dlopen("libEGL.so", RTLD_NOW | RTLD_LOCAL);
    if (!gl_lib) gl_lib = dlopen("libGL.so.1", RTLD_NOW | RTLD_LOCAL);
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
    const char* version = (const char*)p_glGetString(GL_VERSION);
    OpenGLComputeApi api = OPENGL_COMPUTE_API_NONE;
    int major = 0, minor = 0;
    if (parse_gl_version(version, &api, &major, &minor) != 0 ||
        !opengl_compute_version_supported(api, major, minor)) return -1;
#ifdef __ANDROID__
    if (api != OPENGL_COMPUTE_API_GLES) return -1;
#else
    if (api != OPENGL_COMPUTE_API_DESKTOP) return -1;
#endif
    compute_capability.api = api;
    compute_capability.major = major;
    compute_capability.minor = minor;
    compute_capability.compute_supported = 0;
    LOAD_GL(glGetIntegerv);
    LOAD_GL(glGetIntegeri_v);
    LOAD_GL(glGetInteger64v);
    LOAD_GL(glGetError);
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
    LOAD_GL(glBufferSubData);
    LOAD_GL(glBindBufferBase);
    /* Optional on purpose. glBindBufferRange is core wherever compute shaders
     * are, but a driver that does not hand it over should lose device rows and
     * keep every whole-tensor binding, not fail to initialize. */
    p_glBindBufferRange = load_gl_proc("glBindBufferRange");
    LOAD_GL(glDeleteBuffers);
    LOAD_GL(glDispatchCompute);
    LOAD_GL(glMemoryBarrier);
    LOAD_GL(glFinish);
    p_glGetBufferSubData = load_gl_proc("glGetBufferSubData");
    p_glMapBufferRange = load_gl_proc("glMapBufferRange");
    p_glUnmapBuffer = load_gl_proc("glUnmapBuffer");
    if (api == OPENGL_COMPUTE_API_GLES) {
        if (!p_glMapBufferRange || !p_glUnmapBuffer) return -1;
    } else if (!p_glGetBufferSubData && (!p_glMapBufferRange || !p_glUnmapBuffer)) {
        return -1;
    }
    p_glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &max_ssbo_bindings);
    p_glGetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS, &max_uniform_bindings);
    p_glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &max_compute_invocations);
    p_glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &max_ssbo_block_size);
    p_glGetInteger64v(GL_MAX_UNIFORM_BLOCK_SIZE, &max_uniform_block_size);
    /* A driver that reports nothing here leaves the query untouched; treating
     * that as "align to the whole buffer" refuses every window rather than
     * binding one the device may reject. */
    graph_alignment = 0;
    p_glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &graph_alignment);
    if (graph_alignment <= 0) graph_alignment = 0;
#if VOLVOXAI_ENABLE_TRAINING
    p_glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &max_compute_ssbo_blocks);
    p_glGetIntegerv(GL_MAX_COMPUTE_UNIFORM_BLOCKS, &max_compute_uniform_blocks);
#endif
    for (GLuint axis = 0; axis < 3; axis++) {
        p_glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, axis, &max_compute_groups[axis]);
        p_glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, axis, &max_compute_group_size[axis]);
    }
    if (max_ssbo_bindings <= 0 || max_uniform_bindings <= 0 || max_compute_invocations <= 0 ||
#if VOLVOXAI_ENABLE_TRAINING
        max_compute_ssbo_blocks <= 0 || max_compute_uniform_blocks <= 0 ||
#endif
        max_ssbo_block_size <= 0 || max_uniform_block_size <= 0 ||
        max_compute_groups[0] <= 0 ||
        max_compute_groups[1] <= 0 || max_compute_groups[2] <= 0 ||
        max_compute_group_size[0] <= 0 || max_compute_group_size[1] <= 0 ||
        max_compute_group_size[2] <= 0) return -1;
    compute_capability.compute_supported = 1;
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

#ifndef __ANDROID__
initialized:
#endif
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
    EGLint ctx_attrs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3,
        EGL_CONTEXT_MINOR_VERSION, 1,
        EGL_NONE
    };
#else
    EGLint ctx_attrs[] = { EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 3, EGL_NONE };
#endif
    egl_context = p_eglCreateContext(egl_display, cfg, EGL_NO_CONTEXT, ctx_attrs);
    if (egl_context == EGL_NO_CONTEXT) return -1;
    if (p_eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context) != EGL_TRUE) return -1;
    return 0;
}

int opengl_init(void) {
    OpenGLContextState* state = opengl_context_state_get(1);
    if (!state) return -1;
    opengl_device_lock();
    if (state->device_acquired) {
        opengl_device_unlock();
        return 0;
    }
    /* Key on the context itself, not the reference count. The context now
     * outlives its last holder (compile and execute run under different engine
     * states), so a zero count with a live context must reattach rather than
     * build a second one. */
    if (egl_context != EGL_NO_CONTEXT) {
        g_opengl_device_state.reference_count++;
        state->device_acquired = 1;
        opengl_device_unlock();
        return 0;
    }
    if (load_egl() != 0 || create_context() != 0 || load_gl() != 0) {
        printf("[VolvoxAI GPU] Failed to initialize OpenGL compute backend.\n");
        opengl_device_shutdown_locked();
        opengl_device_unlock();
        return -1;
    }
    {
        const char* disable_tiled = getenv("VOLVOX_OPENGL_DISABLE_TILED_QCONV");
        qconv_tiled_enabled = !(disable_tiled && disable_tiled[0] &&
                                strcmp(disable_tiled, "0") != 0);
    }
    const GLubyte* vendor = p_glGetString(GL_VENDOR);
    const GLubyte* renderer = p_glGetString(GL_RENDERER);
    const GLubyte* version = p_glGetString(GL_VERSION);
    if (!g_opengl_device_state.exit_hook_registered) {
        g_opengl_device_state.exit_hook_registered = 1;
        atexit(opengl_device_release);
    }
    printf("[VolvoxAI GPU] OpenGL Compute initialized: %s / %s / %s\n",
           vendor ? (const char*)vendor : "unknown",
           renderer ? (const char*)renderer : "unknown",
           version ? (const char*)version : "unknown");
    opengl_release_current_locked();
    g_opengl_device_state.reference_count = 1;
    state->device_acquired = 1;
    opengl_device_unlock();
    return 0;
}

static int ogl_ready(void) {
    OpenGLContextState* state = opengl_context_state_get(0);
    return state && state->device_acquired &&
           egl_context != EGL_NO_CONTEXT && compute_capability.compute_supported &&
           p_glDispatchCompute != NULL;
}

static int opengl_workgroup_supported(uint32_t x, uint32_t y, uint32_t z) {
    uint64_t invocations = (uint64_t)x * (uint64_t)y * (uint64_t)z;
    return x > 0 && y > 0 && z > 0 &&
        x <= (uint32_t)max_compute_group_size[0] &&
        y <= (uint32_t)max_compute_group_size[1] &&
        z <= (uint32_t)max_compute_group_size[2] &&
        invocations <= (uint64_t)max_compute_invocations;
}

static OglProgramCacheEntry* program_cache_entry(const OglKernel* descriptor,
                                                 int create) {
    if (!descriptor) return NULL;
    for (size_t i = 0; i < g_opengl_device_state.program_cache_count; i++) {
        OglProgramCacheEntry* entry = &g_opengl_device_state.program_cache[i];
        if (entry->descriptor == descriptor) return entry;
    }
    if (!create || g_opengl_device_state.program_cache_count >= OGL_MAX_PROGRAM_CACHE)
        return NULL;
    OglProgramCacheEntry* entry =
        &g_opengl_device_state.program_cache[
            g_opengl_device_state.program_cache_count++];
    memset(entry, 0, sizeof(*entry));
    entry->descriptor = descriptor;
    return entry;
}

static GLuint compile_kernel(const OglKernel* k) {
    if (!ogl_ready() || !k) return 0;
    OglProgramCacheEntry* entry = program_cache_entry(k, 1);
    if (!entry) return 0;
    if (entry->ready) return entry->program;
    if (entry->failed) return 0;
    VolvoxAIShaderView source;
    if (volvoxai_shader_store_get(k->path, &source) !=
            VOLVOXAI_SHADER_STORE_OK ||
        source.size == 0 || source.size > (size_t)INT_MAX) {
        fprintf(stderr, "[OpenGL] failed to read generated shader %s\n", k->path);
        entry->failed = 1;
        return 0;
    }
    GLuint sh = p_glCreateShader(GL_COMPUTE_SHADER);
    const GLchar* src = (const GLchar*)source.data;
    GLint source_length = (GLint)source.size;
    p_glShaderSource(sh, 1, &src, &source_length);
    p_glCompileShader(sh);
    GLint ok = 0;
    p_glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        GLsizei n = 0;
        p_glGetShaderInfoLog(sh, (GLsizei)sizeof(log), &n, log);
        fprintf(stderr, "[OpenGL] shader compile failed (%s): %s\n", k->name, log);
        p_glDeleteShader(sh);
        entry->failed = 1;
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
        entry->failed = 1;
        return 0;
    }
    entry->program = prog;
    entry->ready = 1;
    return prog;
}

static void opengl_clear_errors(void) {
    if (!p_glGetError) return;
    for (int attempt = 0; attempt < 16; attempt++)
        if (p_glGetError() == GL_NO_ERROR) break;
}

static GLuint create_buffer_target(GLenum target, size_t bytes, const void* data) {
    OpenGLContextState* state = opengl_context_state_get(0);
    GLint64 limit = target == GL_UNIFORM_BUFFER
        ? max_uniform_block_size : max_ssbo_block_size;
    if (!ogl_ready() || !state ||
        (!state->forward_active && !state->transient_active
#if VOLVOXAI_ENABLE_TRAINING
         && !state->training_is_active
#endif
        ) || bytes == 0 || bytes > (size_t)INTPTR_MAX || limit <= 0 ||
        (uint64_t)bytes > (uint64_t)limit) return 0;
    GLuint b = 0;
    opengl_clear_errors();
    p_glGenBuffers(1, &b);
    if (!b) return 0;
    p_glBindBuffer(target, b);
    p_glBufferData(target, (GLsizeiptr)bytes, data, GL_DYNAMIC_DRAW);
    GLenum error = p_glGetError();
    p_glBindBuffer(target, 0);
    if (error != GL_NO_ERROR) {
        p_glDeleteBuffers(1, &b);
        return 0;
    }
    return b;
}

static GLuint create_buffer(size_t bytes, const void* data) {
    return create_buffer_target(GL_SHADER_STORAGE_BUFFER, bytes, data);
}

static GLuint qgroupnorm_stats_ensure(size_t bytes) {
    OpenGLContextState* state = opengl_context_state_get(0);
    GLuint next;
    if (!ogl_ready() || !state || bytes == 0 ||
        bytes > (size_t)INTPTR_MAX) return 0;
    if (qgroupnorm_stats_buffer && qgroupnorm_stats_capacity >= bytes)
        return qgroupnorm_stats_buffer;
    if (state->domain_enforced) return 0;
    next = create_buffer(bytes, NULL);
    if (!next) return 0;
    if (qgroupnorm_stats_buffer && p_glDeleteBuffers)
        p_glDeleteBuffers(1, &qgroupnorm_stats_buffer);
    qgroupnorm_stats_buffer = next;
    qgroupnorm_stats_capacity = bytes;
    state->capacity_generation =
        opengl_generation_next(state->capacity_generation);
    return qgroupnorm_stats_buffer;
}

static GLuint qlayernorm_stats_ensure(size_t bytes) {
    OpenGLContextState* state = opengl_context_state_get(0);
    GLuint next;
    if (!ogl_ready() || !state || bytes == 0 ||
        bytes > (size_t)INTPTR_MAX) return 0;
    if (qlayernorm_stats_buffer && qlayernorm_stats_capacity >= bytes)
        return qlayernorm_stats_buffer;
    if (state->domain_enforced) return 0;
    next = create_buffer(bytes, NULL);
    if (!next) return 0;
    if (qlayernorm_stats_buffer && p_glDeleteBuffers)
        p_glDeleteBuffers(1, &qlayernorm_stats_buffer);
    qlayernorm_stats_buffer = next;
    qlayernorm_stats_capacity = bytes;
    state->capacity_generation =
        opengl_generation_next(state->capacity_generation);
    return qlayernorm_stats_buffer;
}

static int graph_find_slot(const void* host) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!host) return -1;
    for (int i = 0; i < graph_slot_count; i++) {
        if (graph_slots[i].host == host &&
            (graph_slots[i].is_weight ||
             (state && graph_slots[i].shape_generation ==
                           state->shape_generation))) return i;
    }
    return -1;
}

static uint64_t opengl_generation_next(uint64_t generation) {
    generation++;
    return generation ? generation : 1u;
}

static int graph_find_reusable_slot(size_t bytes) {
    int best = -1;
    int empty = -1;
    int grow = -1;
    for (int index = 0; index < graph_slot_count; index++) {
        OglTensorSlot* slot = &graph_slots[index];
        if (slot->host || slot->is_weight) continue;
        if (!slot->owns_buffer || !slot->buffer) {
            if (empty < 0) empty = index;
            continue;
        }
        if (slot->cap >= bytes &&
            (best < 0 || slot->cap < graph_slots[best].cap)) best = index;
        else if (grow < 0 || slot->cap > graph_slots[grow].cap) grow = index;
    }
    return best >= 0 ? best : (empty >= 0 ? empty : grow);
}

static int opengl_graph_access_ensure(void) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state || !state->device_acquired) return 0;
    if (state->forward_active || state->transient_active
#if VOLVOXAI_ENABLE_TRAINING
        || state->training_is_active
#endif
    ) return ogl_ready();
    opengl_device_lock();
    if (!ogl_ready() || opengl_make_current_locked() != 0) {
        opengl_device_unlock();
        return 0;
    }
    state->forward_active = 1;
    state->implicit_forward = 1;
    return 1;
}

static OglTensorSlot* graph_get_slot_metadata(const void* host, size_t bytes,
                                               int is_weight) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state || !state->device_acquired || !host || bytes == 0 ||
        bytes > (size_t)INTPTR_MAX || max_ssbo_block_size <= 0 ||
        (uint64_t)bytes > (uint64_t)max_ssbo_block_size) return NULL;
    int idx = graph_find_slot(host);
    if (state->domain_enforced) {
        OglTensorSlot* slot = idx >= 0 ? &graph_slots[idx] : NULL;
        if (!slot || !slot->owns_buffer || !slot->buffer ||
            bytes > slot->cap) return NULL;
        if (slot->domain_span) {
            /* A public graph input may be consumed through an operator's
             * weight/affine binding without becoming an immutable model
             * weight.  Its pre-reserved domain slot remains mutable and is
             * dirtied by the public binding commit before every execution. */
            if (bytes > slot->domain_capacity) return NULL;
            if (bytes > slot->bytes) {
                slot->bytes = bytes;
                slot->host_dirty = 1;
                slot->device_dirty = 0;
            }
            return slot;
        }
        if (!slot->is_weight || !slot->host || !slot->bytes ||
            bytes > slot->bytes) return NULL;
        return slot;
    }
    if (idx < 0) {
        idx = graph_find_reusable_slot(bytes);
        if (idx < 0) {
            if (graph_slot_count >= OGL_GRAPH_MAX_TENSORS) return NULL;
            idx = graph_slot_count++;
            memset(&graph_slots[idx], 0, sizeof(graph_slots[idx]));
        }
        graph_slots[idx].host = host;
        graph_slots[idx].host_dirty = 1;
        graph_slots[idx].is_weight = is_weight;
        graph_slots[idx].shape_generation = is_weight ? 0 :
            state->shape_generation;
    }
    OglTensorSlot* s = &graph_slots[idx];
    /* Grow, never shrink -- see the Vulkan twin. A caller asking for fewer
     * bytes is asking about part of the tensor, not redefining it, and letting
     * it shrink turns the next whole-tensor read into a growth that re-uploads
     * a host mirror over rows the device just wrote. */
    if (bytes > s->bytes) s->bytes = bytes;
    if (is_weight) s->is_weight = 1;
    return s;
}

static OglTensorSlot* graph_get_slot(const void* host, size_t bytes,
                                     int is_weight) {
    if (!opengl_graph_access_ensure()) return NULL;
    return graph_get_slot_metadata(host, bytes, is_weight);
}

static int slot_ensure_owned_buffer(OglTensorSlot* s, size_t bytes, const void* data) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state || !s || bytes == 0 || bytes > (size_t)INTPTR_MAX ||
        max_ssbo_block_size <= 0 ||
        (uint64_t)bytes > (uint64_t)max_ssbo_block_size) return 0;
    if (state->domain_enforced &&
        (!s->owns_buffer || !s->buffer || s->cap < bytes)) return 0;
    if (!s->owns_buffer || s->buffer == 0 || s->cap < bytes) {
        size_t capacity = bytes;
        GLuint candidate;
        if (s->owns_buffer && s->cap && s->cap <= SIZE_MAX / 2u &&
            s->cap * 2u > capacity &&
            (uint64_t)(s->cap * 2u) <= (uint64_t)max_ssbo_block_size)
            capacity = s->cap * 2u;
        candidate = create_buffer(capacity, NULL);
        if (!candidate) return 0;
        if (data) {
            opengl_clear_errors();
            p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, candidate);
            p_glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                              (GLsizeiptr)bytes, data);
            GLenum error = p_glGetError();
            p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            if (error != GL_NO_ERROR) {
                p_glDeleteBuffers(1, &candidate);
                return 0;
            }
        }
        GLuint previous = s->owns_buffer ? s->buffer : 0;
        s->buffer = candidate;
        s->cap = capacity;
        s->owns_buffer = 1;
        state->capacity_generation++;
        if (!state->capacity_generation) state->capacity_generation = 1;
        s->capacity_generation = state->capacity_generation;
        if (previous) p_glDeleteBuffers(1, &previous);
        return 1;
    }
    if (data) {
        opengl_clear_errors();
        p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, s->buffer);
        p_glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                          (GLsizeiptr)bytes, data);
        GLenum error = p_glGetError();
        p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        if (error != GL_NO_ERROR) return 0;
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

/* qLinearInt8 views raw byte tensors as packed u32 words. Allocate a rounded
 * GPU buffer, but copy only the logical host bytes and zero its tail. */
static int graph_packed_bytes(size_t logical_bytes, size_t* storage_bytes) {
    if (!storage_bytes || logical_bytes == 0 || logical_bytes > SIZE_MAX - 3u) return 0;
    *storage_bytes = (logical_bytes + 3u) & ~(size_t)3u;
    return 1;
}

static OglTensorSlot* graph_ensure_packed_bytes(const void* host, size_t logical_bytes,
                                                 int is_weight) {
    static const unsigned char zero_tail[3] = {0u, 0u, 0u};
    size_t storage_bytes;
    if (!host || !graph_packed_bytes(logical_bytes, &storage_bytes) ||
        storage_bytes > (size_t)INTPTR_MAX) return NULL;
    OglTensorSlot* s = graph_get_slot(host, storage_bytes, is_weight);
    if (!s) return NULL;
    if (s->host_dirty || s->buffer == 0 || s->cap < storage_bytes) {
        size_t tail_bytes = storage_bytes - logical_bytes;
        if (!slot_ensure_owned_buffer(s, storage_bytes, NULL)) return NULL;
        opengl_clear_errors();
        p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, s->buffer);
        p_glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                          (GLsizeiptr)logical_bytes, host);
        if (tail_bytes)
            p_glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                              (GLintptr)logical_bytes,
                              (GLsizeiptr)tail_bytes, zero_tail);
        GLenum error = p_glGetError();
        p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        if (error != GL_NO_ERROR) return NULL;
        s->host_dirty = 0;
        s->device_dirty = 0;
    }
    return s;
}

static OglTensorSlot* graph_output_packed_bytes(const void* host, size_t logical_bytes) {
    size_t storage_bytes;
    if (!host || !graph_packed_bytes(logical_bytes, &storage_bytes) ||
        storage_bytes > (size_t)INTPTR_MAX) return NULL;
    OglTensorSlot* s = graph_get_slot(host, storage_bytes, 0);
    if (!s || !slot_ensure_owned_buffer(s, storage_bytes, NULL)) return NULL;
    s->host_dirty = 0;
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

/* A binding into part of a resident tensor -- the OpenGL spelling of Vulkan's
 * VkGraphWindow, and the reason this backend can keep a decode row on the
 * device instead of handing the prefix back to the host.
 *
 * The two backends differ in exactly one place. Vulkan binds every tensor as
 * `(io_buffer, offset, bytes)` into one arena, so a window needed no new
 * machinery at dispatch. Here each slot owns its own buffer and is bound whole
 * with glBindBufferBase, so a window has to name the buffer as well: it is
 * `(buffer, offset, bytes)`, and dispatch_kernel_ranges below is what binds it.
 * `offset` is relative to `buffer`, which is why the alignment check is on
 * `inner` alone where Vulkan's is on `slot->offset + inner`. */
typedef struct {
    OglTensorSlot* slot;
    GLuint buffer;
    size_t offset;
    size_t bytes;
} OglGraphWindow;

/* One SSBO or UBO binding. `bytes == 0` means the whole buffer, which is what
 * every call site that never asks for a window gets. */
typedef struct {
    GLuint buffer;
    size_t offset;
    size_t bytes;
} OglBufferRange;

/*
 * The resident slot whose span contains `[host, host + bytes)`.
 *
 * Exact base first, then the smallest containing span: a tensor and an arena
 * span that both cover the pointer are both legal answers, and the tighter one
 * is the tensor. Returns -1 when nothing contains it, which is the ordinary
 * "this is a new tensor" case and not an error.
 */
static int graph_find_containing_slot(const void* host, size_t bytes, size_t* inner) {
    OpenGLContextState* state = opengl_context_state_get(0);
    uintptr_t start = (uintptr_t)host;
    uintptr_t end;
    int best = -1;
    int exact;
    if (!state || !host || !bytes || start > UINTPTR_MAX - bytes) return -1;
    end = start + bytes;
    exact = graph_find_slot(host);
    if (exact >= 0 && graph_slots[exact].bytes >= bytes) {
        if (inner) *inner = 0;
        return exact;
    }
    for (int index = 0; index < graph_slot_count; index++) {
        OglTensorSlot* slot = &graph_slots[index];
        uintptr_t slot_start = (uintptr_t)slot->host;
        uintptr_t slot_end;
        if (!slot->host || !slot->bytes || !slot->owns_buffer || !slot->buffer ||
            slot_start > UINTPTR_MAX - slot->bytes) continue;
        if (!slot->is_weight && slot->shape_generation != state->shape_generation)
            continue;
        slot_end = slot_start + slot->bytes;
        if (start < slot_start || end > slot_end) continue;
        if (best < 0 || slot->bytes < graph_slots[best].bytes) best = index;
    }
    if (best >= 0 && inner) {
        *inner = (size_t)((uintptr_t)host - (uintptr_t)graph_slots[best].host);
    }
    return best;
}

/*
 * Finish a window against a slot that is already resident.
 *
 * The alignment check is the one real constraint a device row carries. A row
 * offset is `row * width * element size`, so a row is bindable exactly when its
 * *stride* is a multiple of the device's storage alignment. Refusing here names
 * the constraint; the caller falls back to the host row path.
 *
 * An unreported alignment refuses rather than permits, which is where this
 * departs from Vulkan's `graph_alignment && ...`: there a zero means an arena
 * with no granularity to respect, here it means a driver that did not say, and
 * guessing wrong is a GL_INVALID_VALUE at bind time.
 */
static int graph_window_finish(OglTensorSlot* slot, size_t inner, size_t bytes,
                               OglGraphWindow* out) {
    if (!slot || !out || !bytes || !slot->owns_buffer || !slot->buffer ||
        inner > slot->bytes || bytes > slot->bytes - inner ||
        slot->cap < slot->bytes) return 0;
    if (!p_glBindBufferRange || graph_alignment <= 0 ||
        inner % (size_t)graph_alignment) return 0;
    out->slot = slot;
    out->buffer = slot->buffer;
    out->offset = inner;
    out->bytes = bytes;
    return 1;
}

/*
 * Whether `host` names an interior slice, and of which base.
 *
 * Resolution goes through the ordinary `graph_ensure_*` on the *base* pointer
 * rather than around it, so a window inherits every domain, capacity and
 * dirty-state rule a whole tensor obeys. The only thing that differs is where
 * the binding starts.
 */
static int graph_window_base(const void* host, size_t bytes,
                             const void** base, size_t* base_bytes, size_t* inner) {
    int index = graph_find_containing_slot(host, bytes, inner);
    if (index < 0 || *inner == 0) return 0;
    *base = graph_slots[index].host;
    *base_bytes = graph_slots[index].bytes;
    return 1;
}

static int graph_window_device(const void* host, size_t bytes, int is_weight,
                               OglGraphWindow* out) {
    const void* base = host;
    size_t base_bytes = bytes;
    size_t inner = 0;
    OglTensorSlot* slot;
    if (!host || !bytes || !out) return 0;
    if (!graph_window_base(host, bytes, &base, &base_bytes, &inner)) {
        base = host;
        base_bytes = bytes;
        inner = 0;
    }
    slot = graph_ensure_device(base, base_bytes, is_weight);
    if (!slot) return 0;
    if (inner == 0) {
        /* A whole tensor is not a window; bind it the way every other call
         * site does so a zero-offset dispatch keeps using glBindBufferBase. */
        out->slot = slot;
        out->buffer = slot->buffer;
        out->offset = 0;
        out->bytes = 0;
        return out->buffer != 0;
    }
    return graph_window_finish(slot, inner, bytes, out);
}

/* The packed spelling. A window's own bytes stay logical: only the containing
 * slot is rounded up to whole words, and the tail it pads belongs to the
 * tensor rather than to any one row. */
static int graph_window_packed_bytes(const void* host, size_t logical_bytes,
                                     int is_weight, OglGraphWindow* out) {
    const void* base = host;
    size_t base_bytes = logical_bytes;
    size_t inner = 0;
    size_t storage_bytes;
    OglTensorSlot* slot;
    if (!host || !out || !graph_packed_bytes(logical_bytes, &storage_bytes)) return 0;
    if (graph_window_base(host, logical_bytes, &base, &base_bytes, &inner)) {
        slot = graph_ensure_packed_bytes(base, base_bytes, is_weight);
        return graph_window_finish(slot, inner, storage_bytes, out);
    }
    slot = graph_ensure_packed_bytes(host, logical_bytes, is_weight);
    if (!slot || !slot->buffer) return 0;
    out->slot = slot;
    out->buffer = slot->buffer;
    out->offset = 0;
    out->bytes = 0;
    return 1;
}

/*
 * An output window.
 *
 * A row writes part of a tensor, so the rest has to already be on the device --
 * otherwise the next read of an untouched row sees whatever the buffer held.
 * When the containing slot is host-dirty this uploads it whole first and the
 * kernel then overwrites one row, which is the same order the host row path
 * achieves by syncing before it runs.
 */
static int graph_window_output_packed(const void* host, size_t logical_bytes,
                                      OglGraphWindow* out) {
    const void* base = host;
    size_t base_bytes = logical_bytes;
    size_t inner = 0;
    size_t storage_bytes;
    OglTensorSlot* slot;
    if (!host || !out || !graph_packed_bytes(logical_bytes, &storage_bytes)) return 0;
    if (graph_window_base(host, logical_bytes, &base, &base_bytes, &inner)) {
        slot = graph_ensure_packed_bytes(base, base_bytes, 0);
        return graph_window_finish(slot, inner, storage_bytes, out);
    }
    slot = graph_output_packed_bytes(host, logical_bytes);
    if (!slot || !slot->buffer) return 0;
    out->slot = slot;
    out->buffer = slot->buffer;
    out->offset = 0;
    out->bytes = 0;
    return 1;
}

/*
 * Where in the ids tensor a dispatch starts, as a token index.
 *
 * Every other activation binds as a window, because a row's byte offset is
 * `row * stride` and that is a multiple of the storage alignment whenever the
 * stride is. A token row is a single i32, and four bytes divide some drivers'
 * alignment and not others' -- which would make "does QEmbedding run on the
 * device" a property of the adapter rather than of this code.
 *
 * So the ids bind whole on every backend and the row is named as a scalar, out
 * of reach of the alignment rule. The offset is in tokens, not bytes, because
 * that is what the shader indexes with.
 */
static int graph_token_window(const int32_t* tokens, size_t token_bytes,
                              OglTensorSlot** slot, uint32_t* token_offset) {
    const void* base = tokens;
    size_t base_bytes = token_bytes;
    size_t inner = 0;
    int index;
    if (!tokens || !token_bytes || !slot || !token_offset) return 0;
    *token_offset = 0;
    index = graph_find_containing_slot(tokens, token_bytes, &inner);
    if (index >= 0 && inner) {
        /* A misaligned interior pointer is not a token boundary, so it is not
         * a row of this tensor and nothing here can address it. */
        if (inner % sizeof(int32_t)) return 0;
        if (inner / sizeof(int32_t) > UINT32_MAX) return 0;
        base = graph_slots[index].host;
        base_bytes = graph_slots[index].bytes;
        *token_offset = (uint32_t)(inner / sizeof(int32_t));
    }
    *slot = graph_ensure_device(base, base_bytes, 0);
    return *slot != NULL && (*slot)->buffer != 0;
}

static int opengl_dispatch_dimensions_valid(
        uint32_t gx, uint32_t gy, uint32_t gz) {
    return gx > 0u && gy > 0u && gz > 0u &&
        max_compute_groups[0] > 0 && max_compute_groups[1] > 0 &&
        max_compute_groups[2] > 0 &&
        gx <= (uint32_t)max_compute_groups[0] &&
        gy <= (uint32_t)max_compute_groups[1] &&
        gz <= (uint32_t)max_compute_groups[2];
}

static int dispatch_kernel_ranges(const OglKernel* k, const OglBufferRange* binds,
                                  uint32_t gx, uint32_t gy, uint32_t gz) {
    if (!k || !binds ||
        !opengl_dispatch_dimensions_valid(gx, gy, gz) ||
        k->binding_count <= 0)
        return 0;
    for (int i = 0; i < k->binding_count; i++) {
        int limit = i == k->uniform_binding
            ? max_uniform_bindings : max_ssbo_bindings;
        if (!binds[i].buffer || i >= limit) return 0;
        /* A ranged binding needs the entry point and a size the driver will
         * accept; a zero-length range is a GL error, not an empty read. */
        if ((binds[i].offset || binds[i].bytes) &&
            (!p_glBindBufferRange || binds[i].bytes == 0 ||
             binds[i].bytes > (size_t)INTPTR_MAX ||
             binds[i].offset > (size_t)INTPTR_MAX)) return 0;
    }
    GLuint prog = compile_kernel(k);
    if (!prog) return 0;
    if (profile_sync < 0) {
        const char* env = getenv("VOLVOX_GL_PROFILE_SYNC");
        profile_sync = env && env[0] && strcmp(env, "0") ? 1 : 0;
    }
    double t0 = profile_sync ? volvoxai_engine_now_ms() : 0.0;
    p_glUseProgram(prog);
    for (int i = 0; i < k->binding_count; i++) {
        GLenum target = i == k->uniform_binding
            ? GL_UNIFORM_BUFFER : GL_SHADER_STORAGE_BUFFER;
        if (binds[i].offset || binds[i].bytes)
            p_glBindBufferRange(target, (GLuint)i, binds[i].buffer,
                                (GLintptr)binds[i].offset,
                                (GLsizeiptr)binds[i].bytes);
        else
            p_glBindBufferBase(target, (GLuint)i, binds[i].buffer);
    }
    p_glDispatchCompute((GLuint)gx, (GLuint)gy, (GLuint)gz);
    p_glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    if (profile_sync && p_glFinish) {
        p_glFinish();
        prof_add_entry(k->name, volvoxai_engine_now_ms() - t0);
    }
    return 1;
}

/* The whole-buffer spelling, which is what all but the decoder closure use. */
static int dispatch_kernel(const OglKernel* k, GLuint* buffers, uint32_t gx,
                           uint32_t gy, uint32_t gz) {
    OglBufferRange binds[OGL_MAX_BINDINGS];
    if (!k || !buffers || k->binding_count <= 0 ||
        k->binding_count > OGL_MAX_BINDINGS) return 0;
    for (int i = 0; i < k->binding_count; i++) {
        binds[i].buffer = buffers[i];
        binds[i].offset = 0;
        binds[i].bytes = 0;
    }
    return dispatch_kernel_ranges(k, binds, gx, gy, gz);
}

static GLuint params_buffer(const void* data, size_t bytes) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!ogl_ready() || !state || !data ||
        (!state->forward_active && !state->transient_active
#if VOLVOXAI_ENABLE_TRAINING
         && !state->training_is_active
#endif
        ) || bytes == 0 || bytes > (size_t)INTPTR_MAX ||
        max_uniform_block_size <= 0 ||
        (uint64_t)bytes > (uint64_t)max_uniform_block_size)
        return 0;

    opengl_clear_errors();
    if (!state->dispatch_params_buffer) {
        p_glGenBuffers(1, &state->dispatch_params_buffer);
        if (!state->dispatch_params_buffer) return 0;
    }
    p_glBindBuffer(GL_UNIFORM_BUFFER, state->dispatch_params_buffer);
    /* glBufferData replaces the object's data store. This is the specified
     * orphaning path: an earlier queued dispatch keeps its old store, while
     * this upload becomes visible to subsequent dispatches without a finish
     * or a ring-size assumption. */
    p_glBufferData(GL_UNIFORM_BUFFER, (GLsizeiptr)bytes, data, GL_DYNAMIC_DRAW);
    GLenum error = p_glGetError();
    p_glBindBuffer(GL_UNIFORM_BUFFER, 0);
    if (error != GL_NO_ERROR) return 0;
    state->dispatch_params_upload_count++;
    return state->dispatch_params_buffer;
}

/* Parameter-buffer call sites historically deleted their one-shot UBO after
 * dispatch. Keep those call sites source-compatible while retaining the one
 * context-owned name; all unrelated buffers still reach the GL driver. */
static void opengl_driver_delete_buffers(
        GLsizei count, const GLuint* buffers) {
    if (p_glDeleteBuffers) p_glDeleteBuffers(count, buffers);
}
#undef p_glDeleteBuffers
static void opengl_delete_buffers_preserving_params(
        GLsizei count, const GLuint* buffers) {
    if (count <= 0 || !buffers)
        return;
    OpenGLContextState* state = opengl_context_state_get(0);
    GLuint pending[16];
    GLsizei pending_count = 0;
    for (GLsizei index = 0; index < count; index++) {
        GLuint buffer = buffers[index];
        if (!buffer || (state && buffer == state->dispatch_params_buffer))
            continue;
        pending[pending_count++] = buffer;
        if (pending_count == (GLsizei)(sizeof(pending) / sizeof(pending[0]))) {
            opengl_driver_delete_buffers(pending_count, pending);
            pending_count = 0;
        }
    }
    if (pending_count)
        opengl_driver_delete_buffers(pending_count, pending);
}
#define p_glDeleteBuffers opengl_delete_buffers_preserving_params

static int read_buffer(GLenum target, size_t bytes, void* out) {
    if (!out || bytes == 0) return 0;
    /* glGetBufferSubData is desktop GL, not an OpenGL ES core entry point. */
    if (compute_capability.api == OPENGL_COMPUTE_API_DESKTOP && p_glGetBufferSubData) {
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

#include "opengl_training.inc"

static int opengl_graph_domain_spans_match(
        const VolvoxAIEnginePhysicalSpan* spans,
        size_t span_count) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state || !spans || span_count != state->domain_span_count) return 0;
    for (size_t index = 0; index < span_count; index++) {
        int slot_index = graph_find_slot(spans[index].host);
        OglTensorSlot* slot;
        if (slot_index < 0) return 0;
        slot = &graph_slots[slot_index];
        if (!slot->domain_span || slot->is_weight || !slot->owns_buffer ||
            !slot->buffer || slot->domain_capacity !=
                spans[index].capacity_bytes ||
            slot->bytes > slot->domain_capacity)
            return 0;
    }
    return 1;
}

static int opengl_domain_candidate_allocation_allowed(
        OpenGLContextState* state) {
#ifdef VOLVOX_OPENGL_TESTING
    if (state->test_domain_allocation_failure_after == 0) {
        state->test_domain_allocation_failure_after = -1;
        return 0;
    }
    if (state->test_domain_allocation_failure_after > 0)
        state->test_domain_allocation_failure_after--;
#else
    (void)state;
#endif
    return 1;
}

int opengl_graph_bind_shape_domain(
        const char* signature,
        const VolvoxAIEnginePhysicalSpan* spans,
        size_t span_count,
        size_t qgroupnorm_stats_bytes,
        size_t qlayernorm_stats_bytes) {
    OpenGLContextState* state = opengl_context_state_get(0);
    char* candidate_signature = NULL;
    GLuint* candidate_buffers = NULL;
    GLuint candidate_qgroupnorm_stats_buffer = 0;
    GLuint candidate_qlayernorm_stats_buffer = 0;
    size_t signature_length;
    int weight_count = 0;
    int lock_held = 0;
    int current_held = 0;
    int result = -1;
    uint64_t next_shape_generation;
    uint64_t next_capacity_generation;
    if (!state || !state->device_acquired || !signature || !signature[0] ||
        !spans || !span_count || span_count > OGL_GRAPH_MAX_TENSORS ||
        state->forward_active || state->transient_active)
        return -1;
#if VOLVOXAI_ENABLE_TRAINING
    if (state->training_is_active) return -1;
#endif
    signature_length = strlen(signature);
    if (signature_length > 1024u * 1024u) return -1;
    if (qgroupnorm_stats_bytes > SIZE_MAX - qlayernorm_stats_bytes ||
        (qgroupnorm_stats_bytes &&
         (qgroupnorm_stats_bytes > (size_t)INTPTR_MAX ||
          max_ssbo_block_size <= 0 ||
          (uint64_t)qgroupnorm_stats_bytes >
              (uint64_t)max_ssbo_block_size)) ||
        (qlayernorm_stats_bytes &&
         (qlayernorm_stats_bytes > (size_t)INTPTR_MAX ||
          max_ssbo_block_size <= 0 ||
          (uint64_t)qlayernorm_stats_bytes >
              (uint64_t)max_ssbo_block_size)))
        return -1;
    for (size_t index = 0; index < span_count; index++) {
        uintptr_t start = (uintptr_t)spans[index].host;
        size_t capacity = spans[index].capacity_bytes;
        if (!start || !capacity || start > UINTPTR_MAX - capacity ||
            capacity > (size_t)INTPTR_MAX || max_ssbo_block_size <= 0 ||
            (uint64_t)capacity > (uint64_t)max_ssbo_block_size ||
            (index && (uintptr_t)spans[index - 1u].host +
                spans[index - 1u].capacity_bytes > start))
            return -1;
    }
    if (state->domain_enforced) {
        if (!opengl_graph_domain_spans_match(spans, span_count) ||
            qgroupnorm_stats_bytes !=
                state->domain_qgroupnorm_stats_bytes ||
            qlayernorm_stats_bytes !=
                state->domain_qlayernorm_stats_bytes)
            return -1;
        if (state->shape_signature &&
            !strcmp(state->shape_signature, signature)) {
            for (int index = 0; index < state->graph_slots_count; index++) {
                OglTensorSlot* slot = &state->graph_slot_storage[index];
                if (!slot->domain_span) continue;
                slot->bytes = 0;
                slot->host_dirty = 0;
                slot->device_dirty = 0;
            }
            return 0;
        }
        candidate_signature = (char*)malloc(signature_length + 1u);
        if (!candidate_signature) return -2;
        memcpy(candidate_signature, signature, signature_length + 1u);
        opengl_device_lock();
        lock_held = 1;
        if (!ogl_ready() || opengl_make_current_locked() != 0) goto done;
        current_held = 1;
        if (p_glFinish) p_glFinish();
        next_shape_generation =
            opengl_generation_next(state->shape_generation);
        for (int index = 0; index < state->graph_slots_count; index++) {
            OglTensorSlot* slot = &state->graph_slot_storage[index];
            if (!slot->domain_span) continue;
            slot->shape_generation = next_shape_generation;
            slot->bytes = 0;
            slot->host_dirty = 0;
            slot->device_dirty = 0;
        }
        free(state->shape_signature);
        state->shape_signature = candidate_signature;
        candidate_signature = NULL;
        state->shape_generation = next_shape_generation;
        result = 0;
        goto done;
    }

    for (int index = 0; index < state->graph_slots_count; index++) {
        OglTensorSlot* slot = &state->graph_slot_storage[index];
        if (!slot->is_weight) continue;
        uintptr_t weight_start = (uintptr_t)slot->host;
        if (!weight_start || !slot->owns_buffer || !slot->buffer ||
            !slot->bytes || slot->bytes > slot->cap ||
            weight_start > UINTPTR_MAX - slot->bytes)
            return -1;
        weight_count++;
        for (size_t span_index = 0; span_index < span_count; span_index++) {
            uintptr_t span_start = (uintptr_t)spans[span_index].host;
            uintptr_t span_end = span_start + spans[span_index].capacity_bytes;
            uintptr_t weight_end = weight_start + slot->bytes;
            if (span_start < weight_end && weight_start < span_end)
                return -1;
        }
    }
    if (span_count > (size_t)(OGL_GRAPH_MAX_TENSORS - weight_count))
        return -1;
    candidate_signature = (char*)malloc(signature_length + 1u);
    candidate_buffers = (GLuint*)calloc(span_count, sizeof(*candidate_buffers));
    if (!candidate_signature || !candidate_buffers) {
        result = -2;
        goto done;
    }
    memcpy(candidate_signature, signature, signature_length + 1u);
    opengl_device_lock();
    lock_held = 1;
    if (!ogl_ready() || opengl_make_current_locked() != 0) goto done;
    current_held = 1;
    state->transient_active = 1;
    if (p_glFinish) p_glFinish();
    for (size_t index = 0; index < span_count; index++) {
        if (!opengl_domain_candidate_allocation_allowed(state)) {
            result = -2;
            goto done;
        }
        candidate_buffers[index] =
            create_buffer(spans[index].capacity_bytes, NULL);
        if (!candidate_buffers[index]) {
            result = -2;
            goto done;
        }
    }
    if (qgroupnorm_stats_bytes) {
        if (!opengl_domain_candidate_allocation_allowed(state)) {
            result = -2;
            goto done;
        }
        candidate_qgroupnorm_stats_buffer =
            create_buffer(qgroupnorm_stats_bytes, NULL);
        if (!candidate_qgroupnorm_stats_buffer) {
            result = -2;
            goto done;
        }
    }
    if (qlayernorm_stats_bytes) {
        if (!opengl_domain_candidate_allocation_allowed(state)) {
            result = -2;
            goto done;
        }
        candidate_qlayernorm_stats_buffer =
            create_buffer(qlayernorm_stats_bytes, NULL);
        if (!candidate_qlayernorm_stats_buffer) {
            result = -2;
            goto done;
        }
    }

    next_shape_generation =
        opengl_generation_next(state->shape_generation);
    next_capacity_generation =
        opengl_generation_next(state->capacity_generation);
    for (int index = 0; index < state->graph_slots_count; index++) {
        OglTensorSlot* slot = &state->graph_slot_storage[index];
        if (!slot->is_weight && slot->owns_buffer && slot->buffer)
            p_glDeleteBuffers(1, &slot->buffer);
    }
    if (state->qgroupnorm_scratch_buffer)
        p_glDeleteBuffers(1, &state->qgroupnorm_scratch_buffer);
    state->qgroupnorm_scratch_buffer = candidate_qgroupnorm_stats_buffer;
    state->qgroupnorm_scratch_capacity = qgroupnorm_stats_bytes;
    candidate_qgroupnorm_stats_buffer = 0;
    if (state->qlayernorm_scratch_buffer)
        p_glDeleteBuffers(1, &state->qlayernorm_scratch_buffer);
    state->qlayernorm_scratch_buffer = candidate_qlayernorm_stats_buffer;
    state->qlayernorm_scratch_capacity = qlayernorm_stats_bytes;
    candidate_qlayernorm_stats_buffer = 0;
    {
        int kept = 0;
        int old_slot_count = state->graph_slots_count;
        for (int index = 0; index < old_slot_count; index++) {
            if (!state->graph_slot_storage[index].is_weight) continue;
            if (kept != index)
                state->graph_slot_storage[kept] =
                    state->graph_slot_storage[index];
            kept++;
        }
        if (kept < old_slot_count) {
            memset(&state->graph_slot_storage[kept], 0,
                   (size_t)(old_slot_count - kept) *
                       sizeof(state->graph_slot_storage[0]));
        }
        state->graph_slots_count = kept;
    }
    for (size_t index = 0; index < span_count; index++) {
        OglTensorSlot* slot =
            &state->graph_slot_storage[state->graph_slots_count++];
        memset(slot, 0, sizeof(*slot));
        slot->host = spans[index].host;
        slot->cap = spans[index].capacity_bytes;
        slot->domain_capacity = spans[index].capacity_bytes;
        slot->buffer = candidate_buffers[index];
        candidate_buffers[index] = 0;
        slot->shape_generation = next_shape_generation;
        slot->capacity_generation = next_capacity_generation;
        slot->owns_buffer = 1;
        slot->domain_span = 1;
    }
    free(state->shape_signature);
    state->shape_signature = candidate_signature;
    candidate_signature = NULL;
    state->shape_generation = next_shape_generation;
    state->capacity_generation = next_capacity_generation;
    state->domain_span_count = span_count;
    state->domain_qgroupnorm_stats_bytes = qgroupnorm_stats_bytes;
    state->domain_qlayernorm_stats_bytes = qlayernorm_stats_bytes;
    state->domain_enforced = 1;
    result = 0;

done:
    if (candidate_buffers && current_held && p_glDeleteBuffers) {
        for (size_t index = 0; index < span_count; index++) {
            if (candidate_buffers[index])
                p_glDeleteBuffers(1, &candidate_buffers[index]);
        }
    }
    if (current_held && p_glDeleteBuffers &&
        candidate_qgroupnorm_stats_buffer)
        p_glDeleteBuffers(1, &candidate_qgroupnorm_stats_buffer);
    if (current_held && p_glDeleteBuffers &&
        candidate_qlayernorm_stats_buffer)
        p_glDeleteBuffers(1, &candidate_qlayernorm_stats_buffer);
    state->transient_active = 0;
    if (current_held) opengl_release_current_locked();
    if (lock_held) opengl_device_unlock();
    free(candidate_buffers);
    free(candidate_signature);
    return result;
}

#ifdef VOLVOX_OPENGL_TESTING
int opengl_graph_debug_dynamic_state(OpenGLGraphDynamicStateProbe* probe) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state || !probe) return -1;
    memset(probe, 0, sizeof(*probe));
    probe->shape_generation = state->shape_generation;
    probe->capacity_generation = state->capacity_generation;
    probe->domain_span_count = state->domain_span_count;
    probe->domain_scratch_capacity_bytes =
        state->qgroupnorm_scratch_capacity +
        state->qlayernorm_scratch_capacity;
    probe->domain_enforced = state->domain_enforced;
    probe->slot_count = state->graph_slots_count;
    probe->dispatch_params_buffer_count =
        state->dispatch_params_buffer ? 1 : 0;
    probe->dispatch_params_upload_count =
        state->dispatch_params_upload_count;
    probe->conv_out16_dispatch_count =
        state->test_conv_out16_dispatch_count;
    for (int index = 0; index < state->graph_slots_count; index++) {
        OglTensorSlot* slot = &state->graph_slot_storage[index];
        if (!slot->owns_buffer || !slot->buffer) continue;
        if (slot->host) probe->active_capacity_bytes += slot->cap;
        else probe->pooled_capacity_bytes += slot->cap;
    }
    return 0;
}

int opengl_test_fail_domain_allocation_after(size_t successful_allocations) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state || successful_allocations > (size_t)INT_MAX) return -1;
    state->test_domain_allocation_failure_after =
        (int)successful_allocations;
    return 0;
}
#endif

void opengl_graph_begin_forward(void) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state
#if VOLVOXAI_ENABLE_TRAINING
        || state->training_is_active
#endif
    ) return;
    if (state->forward_active) {
        state->implicit_forward = 0;
        return;
    }
    opengl_device_lock();
    if (!ogl_ready() || opengl_make_current_locked() != 0) {
        opengl_device_unlock();
        return;
    }
    state->forward_active = 1;
    state->implicit_forward = 0;
}

int opengl_graph_end_forward(void) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state || !state->forward_active) return 0;
    int ok = ogl_ready();
    if (ok && p_glFinish) p_glFinish();
    state->forward_active = 0;
    state->implicit_forward = 0;
    opengl_release_current_locked();
    opengl_device_unlock();
    if (!ok) return -1;
    return 0;
}

void opengl_graph_mark_host(const void* host, size_t bytes, int is_weight) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state || !state->device_acquired) return;
    OglTensorSlot* s = graph_get_slot_metadata(host, bytes, is_weight);
    if (!s) return;
    s->host_dirty = 1;
    s->device_dirty = 0;
}

int opengl_graph_sync_host(const void* host, size_t bytes, int is_weight) {
    (void)is_weight;
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state || !state->device_acquired || !host || bytes == 0) return 0;
    int owns_lock = !state->forward_active
#if VOLVOXAI_ENABLE_TRAINING
        && !state->training_is_active
#endif
        ;
    int releases_implicit = state->forward_active && state->implicit_forward;
    if (owns_lock) {
        opengl_device_lock();
        if (!ogl_ready() || opengl_make_current_locked() != 0) {
            opengl_device_unlock();
            return 0;
        }
    } else if (!ogl_ready()) {
        return 0;
    }
    int idx = graph_find_slot(host);
    int ok = 0;
    /* An untracked tensor has no device-owned value, so its host storage is
       already current.  Synchronization is intentionally idempotent for the
       CPU-fallback boundary. */
    if (idx < 0) {
        ok = 1;
        goto done;
    }
    OglTensorSlot* s = &graph_slots[idx];
    if (bytes > s->bytes) goto done;
    if (!s->device_dirty) {
        ok = 1;
        goto done;
    }
    if (!s->buffer) goto done;
    p_glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    p_glFinish();
    p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, s->buffer);
    ok = read_buffer(GL_SHADER_STORAGE_BUFFER, bytes, (void*)host);
    p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    if (!ok) goto done;
    s->device_dirty = 0;
    s->host_dirty = 0;
done:
    if (owns_lock) {
        opengl_release_current_locked();
        opengl_device_unlock();
    } else if (releases_implicit) {
        if (p_glFinish) p_glFinish();
        state->forward_active = 0;
        state->implicit_forward = 0;
        opengl_release_current_locked();
        opengl_device_unlock();
    }
    return ok;
}

void opengl_graph_retain_weight(const void* host, size_t bytes) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state || !host || !bytes) return;
    int index = graph_find_slot(host);
    if (index < 0) return;
    OglTensorSlot* slot = &graph_slots[index];
    if (slot->domain_span || !slot->owns_buffer || !slot->buffer ||
        !slot->bytes || bytes > slot->bytes || bytes > slot->cap)
        return;
    slot->is_weight = 1;
    slot->shape_generation = 0;
}

void opengl_graph_demote_weight(const void* host, size_t bytes) {
    OpenGLContextState* state = opengl_context_state_get(0);
    int index;
    OglTensorSlot* slot;
    if (!state || !host || !bytes) return;
    index = graph_find_slot(host);
    if (index < 0) return;
    slot = &graph_slots[index];
    if (slot->domain_span || !slot->bytes || bytes > slot->bytes)
        return;
    slot->is_weight = 0;
    slot->shape_generation = state->shape_generation;
}

/* A storage-view node may share one device buffer only when the runtime has
 * already given both tensors the same host storage. Device slots are keyed by
 * host pointer, and the runtime pools activations so that many tensors with
 * disjoint lifetimes share one host span. Pointing a second span's slot at this
 * span's buffer therefore does not extend one tensor's lifetime -- it merges the
 * two spans on the device for the rest of the graph, so every later tensor
 * planned into either span writes through both. The planner proved those
 * lifetimes against distinct storage, so the result is an unsynchronized
 * read/write overlap: SiLU -> Reshape -> Transpose reads and writes one buffer
 * in a single dispatch and returns a different wrong answer on every run.
 * Distinct host storage must therefore be a real copy. */
int opengl_graph_alias_f32(const float* in, float* out, long n) {
    if (n <= 0 || !in || !out) return 0;
    if (in != out) return opengl_graph_copy_f32(in, out, n);
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* slot = graph_ensure_device(in, bytes, 0);
    if (!slot || !slot->buffer) return 0;
    graph_mark_device(slot);
    return 1;
}
#include "opengl_graph_f32.inc"

#include "opengl_graph_quantized.inc"

int opengl_graph_spatial_softargmax_y_f32(const float* in, float* out, int n, int h, int w, int c) {
    return opengl_graph_profile_common(&k_spatial_softargmax_y, in, out, n, h, w, c,
                                       (long)c * w, ((uint32_t)w + 63u) / 64u, (uint32_t)c);
}

int opengl_graph_profile_x_f32(const float* in, float* out, int n, int h, int w, int c) {
    return opengl_graph_profile_common(&k_profile_x, in, out, n, h, w, c,
                                       (long)2 * c * w, ((uint32_t)w + 63u) / 64u, (uint32_t)c);
}

int opengl_graph_profile_y_f32(const float* in, float* out, int n, int h, int w, int c) {
    return opengl_graph_profile_common(&k_profile_y, in, out, n, h, w, c,
                                       (long)2 * c * h, ((uint32_t)h + 63u) / 64u, (uint32_t)c);
}

int opengl_graph_mean_height_f32(const float* in, float* out, int n, int h, int w, int c) {
    return opengl_graph_profile_common(&k_mean_height, in, out, n, h, w, c,
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

static int opengl_graph_concat_common(const float** inputs, const long* sizes, const int* input_axes,
                                      int count, float* out, int output_axis, int inner, int sigmoid) {
    if (!inputs || !sizes || !out || count <= 0 || output_axis <= 0 || inner <= 0) return 0;
    long total = 0;
    long outer = -1;
    long summed_axis = 0;
    for (int i = 0; i < count; i++) {
        int input_axis = input_axes ? input_axes[i] : (int)sizes[i];
        long axis_block = (long)input_axis * inner;
        if (!inputs[i] || sizes[i] <= 0 || sizes[i] > UINT32_MAX || input_axis <= 0 ||
            axis_block <= 0 || sizes[i] % axis_block) return 0;
        long input_outer = sizes[i] / axis_block;
        if (outer < 0) outer = input_outer;
        else if (outer != input_outer) return 0;
        if (sizes[i] > LONG_MAX - total) return 0;
        total += sizes[i];
        summed_axis += input_axis;
    }
    if (outer <= 0 || summed_axis != output_axis || total != outer * output_axis * (long)inner) return 0;
    size_t out_bytes = (size_t)total * sizeof(float);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!dst) return 0;
    uint32_t axis_offset = 0;
    const OglKernel* kernel = sigmoid ? &k_concat_sigmoid : &k_concat;
    for (int i = 0; i < count; i++) {
        int input_axis = input_axes ? input_axes[i] : (int)sizes[i];
        size_t in_bytes = (size_t)sizes[i] * sizeof(float);
        OglTensorSlot* src = graph_ensure_device(inputs[i], in_bytes, 0);
        if (!src) return 0;
        uint32_t params[8] = {
            (uint32_t)sizes[i], axis_offset, (uint32_t)input_axis,
            (uint32_t)output_axis, (uint32_t)inner, 0u, 0u, 0u
        };
        GLuint pb = params_buffer(params, sizeof(params));
        GLuint bufs[3] = {src->buffer, dst->buffer, pb};
        int ok = dispatch_kernel(kernel, bufs, ((uint32_t)sizes[i] + 63u) / 64u, 1, 1);
        if (pb) p_glDeleteBuffers(1, &pb);
        if (!ok) return 0;
        axis_offset += (uint32_t)input_axis;
    }
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_concat_f32(const float** inputs, const long* sizes, const int* input_axes,
                            int count, float* out, int output_axis, int inner, int sigmoid) {
    if (!input_axes) return 0;
    return opengl_graph_concat_common(inputs, sizes, input_axes, count, out, output_axis, inner, sigmoid);
}

int opengl_graph_concat_32(
        const void* const* inputs, const long* sizes,
        const int* input_axes, int count, void* output,
        int output_axis, int inner) {
    uint32_t output_elements;
    size_t output_bytes;
    if (!vx_typed_control_concat_plan(
            inputs, sizes, input_axes, count, output,
            output_axis, inner, &output_elements, &output_bytes))
        return 0;
    OglTensorSlot* output_slot =
        graph_output_slot(output, output_bytes);
    if (!output_slot) return 0;
    uint32_t axis_offset = 0u;
    for (int index = 0; index < count; index++) {
        uint32_t input_elements = (uint32_t)sizes[index];
        size_t input_bytes =
            (size_t)input_elements * sizeof(uint32_t);
        OglTensorSlot* input_slot =
            graph_ensure_device(inputs[index], input_bytes, 0);
        if (!input_slot) return 0;
        uint32_t params[8] = {
            input_elements, axis_offset,
            (uint32_t)input_axes[index], (uint32_t)output_axis,
            (uint32_t)inner, 0u, 0u, 0u,
        };
        GLuint params_handle = params_buffer(params, sizeof(params));
        if (!params_handle) return 0;
        GLuint buffers[3] = {
            input_slot->buffer, output_slot->buffer, params_handle,
        };
        int ok = dispatch_kernel(
            &k_concat_32, buffers,
            (input_elements + 63u) / 64u, 1u, 1u);
        p_glDeleteBuffers(1, &params_handle);
        if (!ok) return 0;
        axis_offset += (uint32_t)input_axes[index];
    }
    graph_mark_device(output_slot);
    (void)output_elements;
    return 1;
}

int opengl_graph_concat_flat_f32(const float** inputs, const long* sizes, int count, float* out) {
    long total = 0;
    if (!sizes || count <= 0) return 0;
    for (int i = 0; i < count; i++) total += sizes[i];
    if (total <= 0 || total > INT_MAX) return 0;
    return opengl_graph_concat_common(inputs, sizes, NULL, count, out, (int)total, 1, 0);
}

int opengl_graph_concat_sigmoid_flat_f32(const float** inputs, const long* sizes, int count, float* out) {
    long total = 0;
    if (!sizes || count <= 0) return 0;
    for (int i = 0; i < count; i++) total += sizes[i];
    if (total <= 0 || total > INT_MAX) return 0;
    return opengl_graph_concat_common(inputs, sizes, NULL, count, out, (int)total, 1, 1);
}

int opengl_graph_maxpool2d_f32(const float* in, float* out, int n, int h, int width, int c,
                               int out_h, int out_w, int ky, int kx, int sy, int sx,
                               int py, int px) {
    if (!in || !out || n <= 0 || c <= 0 || h <= 0 || width <= 0 || out_h <= 0 || out_w <= 0 ||
        (uint64_t)(uint32_t)n * (uint32_t)c > UINT32_MAX ||
        ky <= 0 || kx <= 0 || sy <= 0 || sx <= 0) return 0;
    size_t in_bytes = (size_t)n * h * width * c * sizeof(float);
    size_t out_bytes = (size_t)n * out_h * out_w * c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[12] = {
        (uint32_t)n, (uint32_t)h, (uint32_t)width, (uint32_t)c, (uint32_t)out_h, (uint32_t)out_w,
        (uint32_t)ky, (uint32_t)kx, (uint32_t)sy, (uint32_t)sx, (uint32_t)py, (uint32_t)px
    };
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_maxpool, bufs, ((uint32_t)out_w + 7u) / 8u,
                             ((uint32_t)out_h + 7u) / 8u, (uint32_t)n * (uint32_t)c);
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
        dy <= 0 || dx <= 0 || groups <= 0 || relu > 2 ||
        (uint64_t)(uint32_t)n * (uint32_t)out_c > UINT32_MAX) return 0;
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
    const OglKernel* kernel = &k_conv2d;
    uint32_t generic_gz = (uint32_t)(n * out_c);
    uint32_t gz = generic_gz;
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
    } else if (groups == 1 && (out_c & 15) == 0) {
        kernel = &k_conv2d_out16;
        gz = (uint32_t)(n * (out_c / 16));
    }
    uint32_t gx = ((uint32_t)out_w + 7u) / 8u;
    uint32_t gy = ((uint32_t)out_h + 7u) / 8u;
    int generic_geometry_valid =
        opengl_dispatch_dimensions_valid(gx, gy, generic_gz);
    int ok = dispatch_kernel(kernel, bufs, gx, gy, gz);
#ifdef VOLVOX_OPENGL_TESTING
    if (ok && kernel == &k_conv2d_out16)
        opengl_context_state_get(0)->test_conv_out16_dispatch_count++;
#endif
    if (!ok && kernel != &k_conv2d) {
        if (kernel == &k_conv2d_pw16tile) {
            ok = dispatch_kernel(&k_conv2d_pw16, bufs, gx, gy, (uint32_t)(n * (out_c / 16)));
        }
        if (!ok && generic_geometry_valid &&
            kernel == &k_conv2d_c3out16)
            ok = dispatch_kernel(&k_conv2d, bufs, gx, gy, generic_gz);
        if (!ok && generic_geometry_valid && kernel == &k_conv2d_out16)
            ok = dispatch_kernel(&k_conv2d, bufs, gx, gy, generic_gz);
        if (!ok && (kernel == &k_conv2d_pw8v4 || kernel == &k_conv2d_pw8v2)) {
            ok = dispatch_kernel(&k_conv2d_pw8, bufs, gx, gy, (uint32_t)(n * ((out_c + 7) / 8)));
        }
        if (!ok && kernel == &k_conv2d_pw16) ok = dispatch_kernel(&k_conv2d_pw8, bufs, gx, gy, (uint32_t)(n * ((out_c + 7) / 8)));
        if (!ok && generic_geometry_valid)
            ok = dispatch_kernel(&k_conv2d, bufs, gx, gy, generic_gz);
    }
    if (pb) p_glDeleteBuffers(1, &pb);
    if (temp_bias) p_glDeleteBuffers(1, &temp_bias);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

static int opengl_checked_float_matrix_bytes(int rows, int columns, size_t* bytes) {
    size_t elements;
    if (!bytes || rows <= 0 || columns <= 0 ||
        (size_t)rows > SIZE_MAX / (size_t)columns) return 0;
    elements = (size_t)rows * (size_t)columns;
    if (elements > SIZE_MAX / sizeof(float)) return 0;
    *bytes = elements * sizeof(float);
    return 1;
}

int opengl_graph_linear_f32(const float* input, const float* weight,
                            const float* bias, float* output, int rows,
                            int d_in, int d_out,
                            int output_major_weight) {
    size_t input_bytes;
    size_t weight_bytes;
    size_t bias_bytes;
    size_t output_bytes;
    const float* bound_bias;
    OglTensorSlot* input_slot;
    OglTensorSlot* weight_slot;
    OglTensorSlot* bias_slot;
    OglTensorSlot* output_slot;
    OglTensorSlot* scale_slot = NULL;
    uint32_t params[3];
    GLuint params_handle;
    uint32_t groups_x;
    uint32_t groups_y;
    int tiled;
    int ok;
    if (!input || !weight || !output || rows <= 0 || d_in <= 0 ||
        d_out <= 0 || (output_major_weight != 0 &&
                       output_major_weight != 1) ||
        !opengl_checked_float_matrix_bytes(rows, d_in, &input_bytes) ||
        !opengl_checked_float_matrix_bytes(
            d_in, d_out, &weight_bytes) ||
        !opengl_checked_float_matrix_bytes(1, d_out, &bias_bytes) ||
        !opengl_checked_float_matrix_bytes(
            rows, d_out, &output_bytes) ||
        qsdpa_ranges_overlap(output, output_bytes, input, input_bytes) ||
        qsdpa_ranges_overlap(output, output_bytes, weight, weight_bytes) ||
        (bias && qsdpa_ranges_overlap(
            output, output_bytes, bias, bias_bytes)))
        return 0;
    bound_bias = bias ? bias :
        (const float*)qconv_zero_bias_get((uint32_t)d_out);
    if (!bound_bias) return 0;

    input_slot = graph_ensure_device(input, input_bytes, 0);
    weight_slot = graph_ensure_device(weight, weight_bytes, 1);
    bias_slot = graph_ensure_device(bound_bias, bias_bytes, 1);
    output_slot = graph_output_slot(output, output_bytes);
    if (!input_slot || !weight_slot || !bias_slot || !output_slot)
        return 0;
    if (output_major_weight) {
        scale_slot = graph_ensure_device(
            linear_dummy_scale, sizeof(linear_dummy_scale), 1);
        if (!scale_slot) return 0;
    }
    params[0] = (uint32_t)rows;
    params[1] = (uint32_t)d_in;
    params[2] = (uint32_t)d_out;
    params_handle = params_buffer(params, sizeof(params));
    if (!params_handle) return 0;

    tiled = rows > 1 && d_in >= 16 && d_out >= 16 &&
        opengl_workgroup_supported(8u, 8u, 1u);
    groups_x = tiled ? ((uint32_t)d_out + 15u) / 16u
                     : ((uint32_t)d_out + 63u) / 64u;
    groups_y = tiled ? ((uint32_t)rows + 15u) / 16u
                     : (uint32_t)rows;
    if (output_major_weight) {
        GLuint buffers[6] = {
            input_slot->buffer, weight_slot->buffer, scale_slot->buffer,
            bias_slot->buffer, output_slot->buffer, params_handle,
        };
        ok = dispatch_kernel(
            tiled ? &k_matmul_out_in_tiled : &k_matmul_out_in,
            buffers, groups_x, groups_y, 1u);
    } else {
        GLuint buffers[5] = {
            input_slot->buffer, weight_slot->buffer, bias_slot->buffer,
            output_slot->buffer, params_handle,
        };
        ok = dispatch_kernel(
            tiled ? &k_matmul_tiled : &k_matmul,
            buffers, groups_x, groups_y, 1u);
    }
    p_glDeleteBuffers(1, &params_handle);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

static int opengl_matmul_locked(const float* in, const float* w, const float* b,
                                float* out, int seq, int d_in, int d_out) {
    if (!ogl_ready() || !in || !w || !out || seq <= 0 || d_in <= 0 || d_out <= 0) return 0;
    int tiled = seq > 1 && d_in >= 16 && d_out >= 16 &&
                opengl_workgroup_supported(8u, 8u, 1u);
    uint32_t local_x = tiled ? 8u : 64u;
    uint32_t local_y = tiled ? 8u : 1u;
    uint32_t groups_x = tiled ? ((uint32_t)d_out + 15u) / 16u
                              : ((uint32_t)d_out + 63u) / 64u;
    uint32_t groups_y = tiled ? ((uint32_t)seq + 15u) / 16u : (uint32_t)seq;
    if (!opengl_workgroup_supported(local_x, local_y, 1u) ||
        groups_x > (uint32_t)max_compute_groups[0] ||
        groups_y > (uint32_t)max_compute_groups[1]) return 0;
    size_t in_bytes, w_bytes, b_bytes, out_bytes;
    if (!opengl_checked_float_matrix_bytes(seq, d_in, &in_bytes) ||
        !opengl_checked_float_matrix_bytes(d_in, d_out, &w_bytes) ||
        !opengl_checked_float_matrix_bytes(1, d_out, &b_bytes) ||
        !opengl_checked_float_matrix_bytes(seq, d_out, &out_bytes)) return 0;
    float* zeros = NULL;
    if (!b) {
        zeros = (float*)calloc((size_t)d_out, sizeof(float));
        if (!zeros) return 0;
        b = zeros;
    }
    GLuint ib = create_buffer(in_bytes, in);
    GLuint wb = create_buffer(w_bytes, w);
    GLuint bb = create_buffer(b_bytes, b);
    GLuint ob = create_buffer(out_bytes, NULL);
    free(zeros);
    if (!ib || !wb || !bb || !ob) {
        GLuint failed[4] = {ib, wb, bb, ob};
        p_glDeleteBuffers(4, failed);
        return 0;
    }
    uint32_t params[3] = {(uint32_t)seq, (uint32_t)d_in, (uint32_t)d_out};
    GLuint pb = params_buffer(params, sizeof(params));
    if (!pb) {
        GLuint failed[4] = {ib, wb, bb, ob};
        p_glDeleteBuffers(4, failed);
        return 0;
    }
    GLuint bufs[5] = {ib, wb, bb, ob, pb};
    const OglKernel* kernel = tiled ? &k_matmul_tiled : &k_matmul;
    int ok = dispatch_kernel(kernel, bufs, groups_x, groups_y, 1);
    if (ok) {
        p_glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, ob);
        ok = read_buffer(GL_SHADER_STORAGE_BUFFER, out_bytes, out);
        p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
    GLuint dels[5] = {ib, wb, bb, ob, pb};
    p_glDeleteBuffers(5, dels);
    return ok;
}

int opengl_matmul(const float* in, const float* w, const float* b, float* out,
                  int seq, int d_in, int d_out) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state || !state->device_acquired) return 0;
    int owns_lock = !state->forward_active
#if VOLVOXAI_ENABLE_TRAINING
        && !state->training_is_active
#endif
        ;
    if (owns_lock) {
        opengl_device_lock();
        if (!ogl_ready() || opengl_make_current_locked() != 0) {
            opengl_device_unlock();
            return 0;
        }
        state->transient_active = 1;
    }
    int ok = opengl_matmul_locked(in, w, b, out, seq, d_in, d_out);
    if (owns_lock) {
        state->transient_active = 0;
        opengl_release_current_locked();
        opengl_device_unlock();
    }
    return ok;
}

void opengl_free_weight_cache(void) {
}

static void opengl_context_resources_release_locked(OpenGLContextState* state,
                                                     int gl_current) {
    if (!state) return;
#if VOLVOXAI_ENABLE_TRAINING
    if (gl_current && p_glDeleteBuffers) {
        for (int i = 0; i < state->training_slots_count; i++) {
            OglTensorSlot* slot = &state->training_slot_storage[i];
            if (slot->owns_buffer && slot->buffer)
                p_glDeleteBuffers(1, &slot->buffer);
        }
    }
    memset(state->training_slot_storage, 0, sizeof(state->training_slot_storage));
    state->training_slots_count = 0;
    state->training_is_active = 0;
#endif
    if (gl_current && p_glDeleteBuffers) {
        for (int i = 0; i < state->graph_slots_count; i++) {
            OglTensorSlot* slot = &state->graph_slot_storage[i];
            if (slot->owns_buffer && slot->buffer)
                p_glDeleteBuffers(1, &slot->buffer);
        }
    }
    memset(state->graph_slot_storage, 0, sizeof(state->graph_slot_storage));
    state->graph_slots_count = 0;
    if (gl_current && state->qgroupnorm_scratch_buffer && p_glDeleteBuffers) {
        p_glDeleteBuffers(1, &state->qgroupnorm_scratch_buffer);
    }
    state->qgroupnorm_scratch_buffer = 0;
    state->qgroupnorm_scratch_capacity = 0;
    if (gl_current && state->qlayernorm_scratch_buffer && p_glDeleteBuffers) {
        p_glDeleteBuffers(1, &state->qlayernorm_scratch_buffer);
    }
    state->qlayernorm_scratch_buffer = 0;
    state->qlayernorm_scratch_capacity = 0;
    if (gl_current && state->dispatch_params_buffer) {
        opengl_driver_delete_buffers(1, &state->dispatch_params_buffer);
    }
    state->dispatch_params_buffer = 0;
    state->dispatch_params_upload_count = 0;
    qconv_zero_bias_release_state(state);
    free(state->shape_signature);
    state->shape_signature = NULL;
    state->domain_span_count = 0;
    state->domain_qgroupnorm_stats_bytes = 0;
    state->domain_qlayernorm_stats_bytes = 0;
    state->domain_enforced = 0;
    state->forward_active = 0;
    state->implicit_forward = 0;
    state->transient_active = 0;
}

static void opengl_device_shutdown_locked(void) {
    int gl_current = egl_context != EGL_NO_CONTEXT &&
        opengl_make_current_locked() == 0;
    for (size_t i = 0; i < g_opengl_device_state.program_cache_count; i++) {
        OglProgramCacheEntry* entry = &g_opengl_device_state.program_cache[i];
        if (gl_current && p_glDeleteProgram && entry->program)
            p_glDeleteProgram(entry->program);
    }
    memset(g_opengl_device_state.program_cache, 0,
           sizeof(g_opengl_device_state.program_cache));
    g_opengl_device_state.program_cache_count = 0;
    if (egl_display != EGL_NO_DISPLAY) {
        if (p_eglMakeCurrent)
            p_eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                             EGL_NO_CONTEXT);
        if (p_eglDestroyContext && egl_context != EGL_NO_CONTEXT)
            p_eglDestroyContext(egl_display, egl_context);
        if (p_eglDestroySurface && egl_surface != EGL_NO_SURFACE)
            p_eglDestroySurface(egl_display, egl_surface);
        if (p_eglTerminate) p_eglTerminate(egl_display);
    }
    egl_context = EGL_NO_CONTEXT;
    egl_surface = EGL_NO_SURFACE;
    egl_display = EGL_NO_DISPLAY;
    /* Keep the loader libraries resident for the process lifetime. Mesa and
       other EGL implementations may register thread-local destructors inside
       these libraries; dlclose here would leave dangling destructor pointers
       when an RPC worker thread exits. Contexts and all GPU resources above
       are still destroyed on every backend cleanup. */
    compute_capability = (OpenGLComputeCapability){OPENGL_COMPUTE_API_NONE, 0, 0, 0};
    max_ssbo_bindings = 0;
    max_uniform_bindings = 0;
    max_compute_groups[0] = max_compute_groups[1] = max_compute_groups[2] = 0;
    max_compute_group_size[0] = max_compute_group_size[1] = max_compute_group_size[2] = 0;
    max_compute_invocations = 0;
    max_ssbo_block_size = 0;
    max_uniform_block_size = 0;
    qconv_tiled_enabled = 1;
    profile_sync = -1;
#if VOLVOXAI_ENABLE_TRAINING
    max_compute_ssbo_blocks = 0;
    max_compute_uniform_blocks = 0;
#endif
    g_opengl_device_state.reference_count = 0;
}

static void opengl_context_state_destroy(void* opaque_state) {
    OpenGLContextState* state = (OpenGLContextState*)opaque_state;
    if (!state) return;
    opengl_device_lock();
    int gl_current = state->device_acquired &&
        egl_context != EGL_NO_CONTEXT && opengl_make_current_locked() == 0;
    opengl_context_resources_release_locked(state, gl_current);
    if (gl_current) opengl_release_current_locked();
    if (state->device_acquired && g_opengl_device_state.reference_count > 0) {
        state->device_acquired = 0;
        g_opengl_device_state.reference_count--;
        /* The EGL context stays even at zero references: compile and execute run
         * under different engine-state scopes, so the count legitimately drops to
         * zero between them and tearing down here rebuilt the whole GL context on
         * every native run. Released explicitly by opengl_device_release(). */
    }
    opengl_device_unlock();
    free(state);
}

/* Drop the shared EGL/GL context once nothing holds it. Separate from
 * opengl_cleanup so the device survives engine-state scope churn. */
void opengl_device_release(void) {
    opengl_device_lock();
    if (g_opengl_device_state.reference_count == 0 && egl_context != EGL_NO_CONTEXT)
        opengl_device_shutdown_locked();
    opengl_device_unlock();
}

void opengl_cleanup(void) {
    VxEngineState* owner = vx_engine_state_current();
    if (!owner || !owner->opengl_context_state) return;
    OpenGLContextState* state =
        (OpenGLContextState*)owner->opengl_context_state;
#if VOLVOXAI_ENABLE_TRAINING
    if (state->training_is_active) opengl_training_end();
#endif
    if (state->forward_active) (void)opengl_graph_end_forward();
    owner->opengl_context_state = NULL;
    owner->opengl_context_state_destroy = NULL;
    opengl_context_state_destroy(state);
}
