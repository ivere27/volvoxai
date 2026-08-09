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
    OGL_TRAINING_KERNEL("moeLinearBackward", "input_main", "moeLinearBackward_input_main", 11, 10, (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9)),
    OGL_TRAINING_KERNEL("moeLinearBackward", "weight_main", "moeLinearBackward_weight_main", 11, 10, (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9)),
    OGL_TRAINING_KERNEL("moeLinearBackward", "bias_main", "moeLinearBackward_bias_main", 11, 10, (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9)),
    OGL_TRAINING_KERNEL("moeLinearBackward", "route_main", "moeLinearBackward_route_main", 11, 10, (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9)),
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
    s->bytes = bytes;
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

static int dispatch_kernel(const OglKernel* k, GLuint* buffers, uint32_t gx,
                           uint32_t gy, uint32_t gz) {
    if (!k || !buffers || gx == 0 || gy == 0 || gz == 0 ||
        gx > (uint32_t)max_compute_groups[0] || gy > (uint32_t)max_compute_groups[1] ||
        gz > (uint32_t)max_compute_groups[2] || k->binding_count <= 0)
        return 0;
    for (int i = 0; i < k->binding_count; i++) {
        int limit = i == k->uniform_binding
            ? max_uniform_bindings : max_ssbo_bindings;
        if (!buffers[i] || i >= limit) return 0;
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
        p_glBindBufferBase(i == k->uniform_binding ? GL_UNIFORM_BUFFER : GL_SHADER_STORAGE_BUFFER,
                           (GLuint)i, buffers[i]);
    }
    p_glDispatchCompute((GLuint)gx, (GLuint)gy, (GLuint)gz);
    p_glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    if (profile_sync && p_glFinish) {
        p_glFinish();
        prof_add_entry(k->name, volvoxai_engine_now_ms() - t0);
    }
    return 1;
}

static GLuint params_buffer(const void* data, size_t bytes) {
    return create_buffer_target(GL_UNIFORM_BUFFER, bytes, data);
}

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

#ifdef VOLVOX_OPENGL_TESTING
void opengl_training_debug_resource_counts(int* programs, int* buffers,
                                           int* inference_buffers) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state) {
        if (programs) *programs = 0;
        if (buffers) *buffers = 0;
        if (inference_buffers) *inference_buffers = 0;
        return;
    }
    int owns_lock = state && !state->training_is_active && !state->forward_active;
    if (owns_lock) opengl_device_lock();
    int program_count = 0;
    for (size_t i = 0; i < sizeof(training_kernels) / sizeof(training_kernels[0]); i++) {
        OglProgramCacheEntry* entry =
            program_cache_entry(&training_kernels[i].kernel, 0);
        if (entry && entry->program) program_count++;
    }
    if (programs) *programs = program_count;
    if (buffers) *buffers = training_slot_count;
    if (inference_buffers) *inference_buffers = graph_slot_count;
    if (owns_lock) opengl_device_unlock();
}

int opengl_training_debug_compile_all(void) {
    OpenGLContextState* state = opengl_context_state_get(0);
    if (!state) return -1;
    int owns_lock = !state->training_is_active && !state->forward_active;
    if (owns_lock) {
        opengl_device_lock();
        if (!opengl_training_available_locked() ||
            opengl_make_current_locked() != 0) {
            opengl_device_unlock();
            return -1;
        }
    } else if (!opengl_training_available_locked()) {
        return -1;
    }
    for (size_t i = 0; i < sizeof(training_kernels) / sizeof(training_kernels[0]); i++) {
        if (!compile_kernel(&training_kernels[i].kernel)) {
            if (owns_lock) {
                opengl_release_current_locked();
                opengl_device_unlock();
            }
            return -1;
        }
    }
    if (owns_lock) {
        opengl_release_current_locked();
        opengl_device_unlock();
    }
    return 0;
}
#endif
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
#ifdef VOLVOX_OPENGL_TESTING
    state->test_domain_allocation_failure_after = -1;
#endif
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

int opengl_graph_alias_f32(const float* in, float* out, long n) {
    if (n <= 0 || !in || !out) return 0;
    OpenGLContextState* state = opengl_context_state_get(0);
    if (state && state->domain_enforced && in != out)
        return opengl_graph_copy_f32(in, out, n);
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_get_slot(out, bytes, 0);
    if (!src || !dst || !src->buffer) return 0;
    if (dst == src) {
        graph_mark_device(dst);
        return 1;
    }
    if (dst->owns_buffer && dst->buffer && dst->buffer != src->buffer) {
        p_glDeleteBuffers(1, &dst->buffer);
        state->capacity_generation =
            opengl_generation_next(state->capacity_generation);
    }
    dst->buffer = src->buffer;
    dst->cap = src->cap;
    dst->capacity_generation = src->capacity_generation;
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

static int opengl_graph_typed_control_32(
        const void* a, size_t a_bytes, const void* b, size_t b_bytes,
        void* output, size_t output_bytes,
        const VxTypedControlMetadata* metadata) {
    if (!metadata) return 0;
    OglTensorSlot* a_slot = graph_ensure_device(a, a_bytes, 0);
    OglTensorSlot* b_slot = graph_ensure_device(b, b_bytes, 0);
    OglTensorSlot* output_slot =
        graph_output_slot(output, output_bytes);
    if (!a_slot || !b_slot || !output_slot) return 0;
    GLuint metadata_buffer =
        create_buffer(sizeof(*metadata), metadata);
    if (!metadata_buffer) return 0;
    GLuint buffers[4] = {
        a_slot->buffer, b_slot->buffer,
        output_slot->buffer, metadata_buffer,
    };
    uint32_t elements = metadata->values[0];
    int ok = dispatch_kernel(
        &k_typed_control_32, buffers,
        (elements + 63u) / 64u, 1u, 1u);
    p_glDeleteBuffers(1, &metadata_buffer);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_compare_i32(
        const int32_t* a, long a_elements,
        const int32_t* b, long b_elements,
        int32_t* output, long output_elements,
        const uint32_t* output_strides,
        const uint32_t* a_strides,
        const uint32_t* b_strides,
        int rank, int operation) {
    VxTypedControlMetadata metadata;
    size_t a_bytes;
    size_t b_bytes;
    size_t output_bytes;
    if (!vx_typed_control_compare_plan(
            a, a_elements, b, b_elements, output, output_elements,
            output_strides, a_strides, b_strides, rank, operation,
            &metadata, &a_bytes, &b_bytes, &output_bytes))
        return 0;
    return opengl_graph_typed_control_32(
        a, a_bytes, b, b_bytes, output, output_bytes, &metadata);
}

static int opengl_graph_unary_i32(
        const int32_t* input, int32_t* output, long elements,
        int operation, int32_t minimum, int32_t maximum) {
    VxTypedControlMetadata metadata;
    size_t bytes;
    if (!vx_typed_control_unary_plan(
            input, output, elements, operation, minimum, maximum,
            &metadata, &bytes))
        return 0;
    return opengl_graph_typed_control_32(
        input, bytes, input, bytes, output, bytes, &metadata);
}

int opengl_graph_not_i32(
        const int32_t* input, int32_t* output, long elements) {
    return opengl_graph_unary_i32(
        input, output, elements, VX_TYPED_CONTROL_NOT_I32, 0, 0);
}

int opengl_graph_clip_i32(
        const int32_t* input, int32_t* output, long elements,
        int32_t minimum, int32_t maximum) {
    return opengl_graph_unary_i32(
        input, output, elements, VX_TYPED_CONTROL_CLIP_I32,
        minimum, maximum);
}

int opengl_graph_copy_32(
        const void* input, void* output, long elements) {
    VxTypedControlMetadata metadata;
    size_t bytes;
    if (!vx_typed_control_copy_plan(
            input, output, elements, &metadata, &bytes))
        return 0;
    return opengl_graph_typed_control_32(
        input, bytes, input, bytes, output, bytes, &metadata);
}

int opengl_graph_cast_typed(
        const void* input, int input_dtype,
        void* output, int output_dtype, long elements) {
    VxTypedControlMetadata metadata;
    size_t bytes;
    if (!vx_typed_control_cast_plan(
            input, input_dtype, output, output_dtype, elements,
            &metadata, &bytes))
        return 0;
    return opengl_graph_typed_control_32(
        input, bytes, input, bytes, output, bytes, &metadata);
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

static int opengl_graph_unary_size_f32(const OglKernel* k, const float* in,
                                       float* out, long n) {
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
int opengl_graph_gelu_f32(const float* in, float* out, long n, int approximate_tanh) {
    if (n <= 0 || (uint64_t)n > UINT32_MAX || !in || !out ||
        (approximate_tanh != 0 && approximate_tanh != 1)) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)n, (uint32_t)approximate_tanh, 0u, 0u};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_gelu, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}
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
    uint32_t params[4] = {(uint32_t)n, (uint32_t)channels, (uint32_t)channels, 0u};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[4] = {src->buffer, w->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_prelu, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_layernorm_f32(const float* in, const float* weight, const float* bias,
                               float* out, int rows, int d_model, float eps) {
    if (rows <= 0 || d_model <= 0 || !in || !weight || !bias || !out ||
        !(eps > 0.0f) || !isfinite(eps)) return 0;
    size_t bytes = (size_t)rows * d_model * sizeof(float);
    size_t wbytes = (size_t)d_model * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* w = graph_ensure_device(weight, wbytes, 1);
    OglTensorSlot* b = graph_ensure_device(bias, wbytes, 1);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !w || !b || !dst) return 0;
    struct {
        uint32_t rows;
        uint32_t d_model;
        float eps;
        uint32_t pad;
    } params = {(uint32_t)rows, (uint32_t)d_model, eps, 0u};
    _Static_assert(sizeof(params) == 16, "LayerNorm uniform ABI");
    GLuint pb = params_buffer(&params, sizeof(params));
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

static int opengl_graph_softmax_like_f32(const OglKernel* k, const float* in,
                                         float* out, int rows, int d) {
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

int opengl_graph_groupnorm_f32(const float* in, const float* weight, const float* bias,
                               float* out, int n, int h, int w, int c, int groups,
                               float eps) {
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || groups <= 0 || c % groups ||
        !in || !weight || !bias || !out || !(eps > 0.0f)) return 0;
    size_t bytes = (size_t)n * h * w * c * sizeof(float);
    size_t cbytes = (size_t)c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* sw = graph_ensure_device(weight, cbytes, 1);
    OglTensorSlot* sb = graph_ensure_device(bias, cbytes, 1);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !sw || !sb || !dst) return 0;
    struct {
        uint32_t n, h, w, c, groups, has_bias;
        float eps;
        uint32_t pad;
    } params = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c,
                (uint32_t)groups, 1u, eps, 0u};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[5] = {src->buffer, sw->buffer, sb->buffer, dst->buffer, pb};
    uint32_t total_groups = (uint32_t)n * (uint32_t)groups;
    int ok = dispatch_kernel(&k_groupnorm, bufs, (total_groups + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

#if VOLVOXAI_ENABLE_TRAINING
int opengl_graph_dropout_f32(const float* in, float* out, long n, uint32_t threshold,
                             uint32_t seed, uint32_t counter, float scale) {
    if (n <= 0 || (uint64_t)n > UINT32_MAX || !in || !out || !(scale >= 1.0f)) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    struct {
        uint32_t length, threshold, seed, counter;
        float scale;
        uint32_t pad[3];
    } params = {(uint32_t)n, threshold, seed, counter, scale, {0u, 0u, 0u}};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_dropout, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}
#endif

/* Routed expert linear. `experts` counts staged rows; route indices stay in
 * global slot space and are mapped through slot_rows when slot_domain != 0. */
int opengl_graph_moe_linear_f32(const float* input, const float* expert_weight,
                                const float* expert_bias, const float* route_indices,
                                const float* route_weights, float* out, int rows,
                                int d_in, int d_out, int experts, int top_k,
                                const uint32_t* slot_rows, uint32_t slot_domain) {
    if (rows <= 0 || d_in <= 0 || d_out <= 0 || experts <= 0 || top_k <= 0 ||
        !input || !expert_weight || !route_indices || !route_weights || !out) return 0;
    if (slot_rows ? (slot_domain < (uint32_t)experts ||
                     (uint32_t)top_k > slot_domain)
                  : top_k > experts) return 0;
    size_t in_bytes = (size_t)rows * (size_t)d_in * sizeof(float);
    size_t w_bytes = (size_t)experts * (size_t)d_in * (size_t)d_out * sizeof(float);
    size_t bias_bytes = (size_t)experts * (size_t)d_out * sizeof(float);
    size_t route_bytes = (size_t)rows * (size_t)top_k * sizeof(float);
    size_t out_bytes = (size_t)rows * (size_t)d_out * sizeof(float);
    uint32_t slot_extent = slot_domain ? slot_domain : 1u;
    OglTensorSlot* si = graph_ensure_device(input, in_bytes, 0);
    OglTensorSlot* sw = graph_ensure_device(expert_weight, w_bytes, 1);
    OglTensorSlot* sri = graph_ensure_device(route_indices, route_bytes, 0);
    OglTensorSlot* srw = graph_ensure_device(route_weights, route_bytes, 0);
    OglTensorSlot* so = graph_output_slot(out, out_bytes);
    OglTensorSlot* sb = expert_bias
        ? graph_ensure_device(expert_bias, bias_bytes, 1) : NULL;
    if (!si || !sw || !sri || !srw || !so || (expert_bias && !sb)) return 0;
    /* The shader always declares a bias and a slot table; absent ones become
     * zero-filled transients so the binding set stays complete. */
    GLuint bias_dummy = 0;
    if (!expert_bias) {
        float* zeros = (float*)calloc((size_t)experts * (size_t)d_out, sizeof(float));
        if (!zeros) return 0;
        bias_dummy = create_buffer_target(GL_SHADER_STORAGE_BUFFER, bias_bytes, zeros);
        free(zeros);
        if (!bias_dummy) return 0;
    }
    uint32_t* table = (uint32_t*)calloc(slot_extent, sizeof(uint32_t));
    if (!table) {
        if (bias_dummy) p_glDeleteBuffers(1, &bias_dummy);
        return 0;
    }
    if (slot_domain) memcpy(table, slot_rows, (size_t)slot_domain * sizeof(uint32_t));
    GLuint slot_buffer = create_buffer_target(
        GL_SHADER_STORAGE_BUFFER, (size_t)slot_extent * sizeof(uint32_t), table);
    free(table);
    if (!slot_buffer) {
        if (bias_dummy) p_glDeleteBuffers(1, &bias_dummy);
        return 0;
    }
    uint32_t params[8] = {
        (uint32_t)rows, (uint32_t)d_in, (uint32_t)d_out, (uint32_t)experts,
        (uint32_t)top_k, expert_bias ? 1u : 0u, slot_domain, 0u,
    };
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[8] = {
        si->buffer, sw->buffer, expert_bias ? sb->buffer : bias_dummy,
        sri->buffer, srw->buffer, so->buffer, pb, slot_buffer,
    };
    int ok = dispatch_kernel(&k_moe_linear, bufs,
                             ((uint32_t)d_out + 63u) / 64u, (uint32_t)rows, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (bias_dummy) p_glDeleteBuffers(1, &bias_dummy);
    p_glDeleteBuffers(1, &slot_buffer);
    if (!ok) return 0;
    graph_mark_device(so);
    return 1;
}

/* Top-k expert routing. The router weight is [d_model, experts], so its expert
 * axis is 1 and it is never a slot-indexed bank. */
int opengl_graph_moe_router_f32(const float* input, const float* weight,
                                const float* bias, float* route_indices,
                                float* route_weights, int rows, int d_model,
                                int experts, int top_k, float temperature,
                                int normalize) {
    if (rows <= 0 || d_model <= 0 || experts <= 0 || top_k <= 0 ||
        top_k > experts || top_k > 8 || !(temperature > 0.0f) ||
        !input || !weight || !route_indices || !route_weights) return 0;
    size_t in_bytes = (size_t)rows * (size_t)d_model * sizeof(float);
    size_t w_bytes = (size_t)d_model * (size_t)experts * sizeof(float);
    size_t bias_bytes = (size_t)experts * sizeof(float);
    size_t route_bytes = (size_t)rows * (size_t)top_k * sizeof(float);
    OglTensorSlot* si = graph_ensure_device(input, in_bytes, 0);
    OglTensorSlot* sw = graph_ensure_device(weight, w_bytes, 1);
    OglTensorSlot* sidx = graph_output_slot(route_indices, route_bytes);
    OglTensorSlot* srw = graph_output_slot(route_weights, route_bytes);
    OglTensorSlot* sb = bias ? graph_ensure_device(bias, bias_bytes, 1) : NULL;
    if (!si || !sw || !sidx || !srw || (bias && !sb)) return 0;
    GLuint bias_dummy = 0;
    if (!bias) {
        float* zeros = (float*)calloc((size_t)experts, sizeof(float));
        if (!zeros) return 0;
        bias_dummy = create_buffer_target(GL_SHADER_STORAGE_BUFFER, bias_bytes, zeros);
        free(zeros);
        if (!bias_dummy) return 0;
    }
    uint32_t params[8];
    params[0] = (uint32_t)rows;
    params[1] = (uint32_t)d_model;
    params[2] = (uint32_t)experts;
    params[3] = (uint32_t)top_k;
    params[4] = normalize ? 1u : 0u;
    params[5] = bias ? 1u : 0u;
    memcpy(&params[6], &temperature, sizeof(float));
    params[7] = 0u;
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[6] = {
        si->buffer, sw->buffer, bias ? sb->buffer : bias_dummy,
        sidx->buffer, srw->buffer, pb,
    };
    int ok = dispatch_kernel(&k_moe_router, bufs, ((uint32_t)rows + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (bias_dummy) p_glDeleteBuffers(1, &bias_dummy);
    if (!ok) return 0;
    graph_mark_device(sidx);
    graph_mark_device(srw);
    return 1;
}

int opengl_graph_embedding_f32(const int32_t* tokens, const float* weight, float* out,
                               int tokens_len, int d_model, int vocab_size) {
    if (tokens_len <= 0 || d_model <= 0 || vocab_size <= 0 || !tokens || !weight || !out) return 0;
    size_t tbytes = (size_t)tokens_len * sizeof(int32_t);
    size_t wbytes = (size_t)vocab_size * d_model * sizeof(float);
    size_t obytes = (size_t)tokens_len * d_model * sizeof(float);
    OglTensorSlot* st = graph_ensure_device(tokens, tbytes, 0);
    OglTensorSlot* sw = graph_ensure_device(weight, wbytes, 1);
    OglTensorSlot* so = graph_output_slot(out, obytes);
    if (!st || !sw || !so) return 0;
    uint32_t params[4] = {(uint32_t)tokens_len, (uint32_t)d_model,
                          (uint32_t)vocab_size, 0u};
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

int opengl_graph_where_32(
        const int32_t* condition, const void* a,
        const void* b, void* output, long elements) {
    uint32_t count;
    size_t bytes;
    if (!vx_typed_control_where_plan(
            condition, a, b, output, elements, &count, &bytes))
        return 0;
    OglTensorSlot* condition_slot =
        graph_ensure_device(condition, bytes, 0);
    OglTensorSlot* a_slot = graph_ensure_device(a, bytes, 0);
    OglTensorSlot* b_slot = graph_ensure_device(b, bytes, 0);
    OglTensorSlot* output_slot =
        graph_output_slot(output, bytes);
    if (!condition_slot || !a_slot || !b_slot || !output_slot) return 0;
    uint32_t params[4] = {count, 0u, 0u, 0u};
    GLuint params_handle = params_buffer(params, sizeof(params));
    if (!params_handle) return 0;
    GLuint buffers[5] = {
        condition_slot->buffer, a_slot->buffer, b_slot->buffer,
        output_slot->buffer, params_handle,
    };
    int ok = dispatch_kernel(
        &k_where_32, buffers, (count + 63u) / 64u, 1u, 1u);
    p_glDeleteBuffers(1, &params_handle);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_argmax_f32(
        const float* input, int32_t* output,
        uint32_t outer, uint32_t axis_size, uint32_t inner) {
    uint32_t input_elements;
    uint32_t output_elements;
    size_t input_bytes;
    size_t output_bytes;
    if (!vx_typed_control_argmax_plan(
            input, output, outer, axis_size, inner,
            &input_elements, &output_elements,
            &input_bytes, &output_bytes))
        return 0;
    OglTensorSlot* input_slot =
        graph_ensure_device(input, input_bytes, 0);
    OglTensorSlot* output_slot =
        graph_output_slot(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    uint32_t params[4] = {outer, axis_size, inner, 0u};
    GLuint params_handle = params_buffer(params, sizeof(params));
    if (!params_handle) return 0;
    GLuint buffers[3] = {
        input_slot->buffer, output_slot->buffer, params_handle,
    };
    int ok = dispatch_kernel(
        &k_argmax_f32_i32, buffers,
        (output_elements + 63u) / 64u, 1u, 1u);
    p_glDeleteBuffers(1, &params_handle);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    (void)input_elements;
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
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0 || out_h <= 0 || out_w <= 0 || !in || !out ||
        (uint64_t)(uint32_t)n * (uint32_t)c > UINT32_MAX) return 0;
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
    VxExpandF32Plan plan;
    if (!vx_expand_f32_plan(
            in, out, in_shape, in_rank, out_shape, out_rank, &plan))
        return 0;
    OglTensorSlot* src = graph_ensure_device(in, plan.input_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, plan.output_bytes);
    if (!src || !dst) return 0;
    GLuint pb = params_buffer(plan.params, sizeof(plan.params));
    if (!pb) return 0;
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(
        &k_expand, bufs, (plan.output_elements + 63u) / 64u, 1u, 1u);
    p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_expand_32(const void* in, void* out, const int* in_shape,
                           int in_rank, const int* out_shape, int out_rank) {
    /* expand.wgsl is word-typed and therefore preserves every I32 bit. */
    return opengl_graph_expand_f32(
        (const float*)in, (float*)out,
        in_shape, in_rank, out_shape, out_rank);
}

int opengl_graph_batch_matmul_f32(
        const float* a, const int* a_shape, int a_rank,
        const float* b, const int* b_shape, int b_rank,
        float* output, const int* output_shape, int output_rank) {
    VxBatchMatMulF32Plan plan;
    size_t metadata_bytes;
    if (!vx_batch_matmul_f32_plan(
            a, a_shape, a_rank, b, b_shape, b_rank,
            output, output_shape, output_rank, &plan))
        return 0;
    metadata_bytes = (size_t)plan.metadata_words * sizeof(uint32_t);
    OglTensorSlot* a_slot = graph_ensure_device(a, plan.a_bytes, 0);
    OglTensorSlot* b_slot = graph_ensure_device(b, plan.b_bytes, 0);
    OglTensorSlot* output_slot =
        graph_output_slot(output, plan.output_bytes);
    GLuint metadata_buffer = create_buffer(metadata_bytes, plan.metadata);
    if (!a_slot || !b_slot || !output_slot || !metadata_buffer) {
        if (metadata_buffer) p_glDeleteBuffers(1, &metadata_buffer);
        return 0;
    }
    GLuint buffers[4] = {
        a_slot->buffer, b_slot->buffer, output_slot->buffer, metadata_buffer,
    };
    int ok = dispatch_kernel(
        &k_batch_matmul, buffers,
        (plan.n + 7u) / 8u, (plan.m + 7u) / 8u,
        plan.output_batches);
    p_glDeleteBuffers(1, &metadata_buffer);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_gather_i32_f32(const float* input, const int32_t* indices,
                                float* output, int outer, int axis_size,
                                int inner, int indices_elements,
                                int output_elements) {
    uint64_t input_count;
    uint64_t expected_output;
    size_t input_bytes;
    size_t index_bytes;
    size_t output_bytes;
    if (!input || !indices || !output || outer <= 0 || axis_size <= 0 ||
        inner <= 0 || indices_elements <= 0 || output_elements <= 0) return 0;
    input_count = (uint64_t)(uint32_t)outer * (uint32_t)axis_size *
        (uint32_t)inner;
    expected_output = (uint64_t)(uint32_t)outer * (uint32_t)indices_elements *
        (uint32_t)inner;
    if (input_count > SIZE_MAX / sizeof(float) ||
        expected_output != (uint32_t)output_elements ||
        expected_output > SIZE_MAX / sizeof(float)) return 0;
    input_bytes = (size_t)input_count * sizeof(float);
    index_bytes = (size_t)(uint32_t)indices_elements * sizeof(int32_t);
    if (index_bytes / sizeof(int32_t) !=
        (size_t)(uint32_t)indices_elements) return 0;
    output_bytes = (size_t)expected_output * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(input, input_bytes, 0);
    OglTensorSlot* idx = graph_ensure_device(indices, index_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(output, output_bytes);
    if (!src || !idx || !dst) return 0;
    uint32_t params[5] = {
        (uint32_t)outer, (uint32_t)axis_size, (uint32_t)inner,
        (uint32_t)indices_elements, (uint32_t)output_elements,
    };
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[4] = {src->buffer, idx->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_gather, bufs,
                             ((uint32_t)output_elements + 63u) / 64u, 1, 1);
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
                            float* out, long out_numel, const uint32_t* output_strides,
                            const uint32_t* a_strides, const uint32_t* b_strides,
                            int rank, int op) {
    if (!a || !b || !out || !output_strides || !a_strides || !b_strides ||
        a_numel <= 0 || b_numel <= 0 || out_numel <= 0 ||
        (uint64_t)out_numel > UINT32_MAX || rank <= 0 || rank > 8 || op < 0 || op > 3)
        return 0;
    size_t a_bytes = (size_t)a_numel * sizeof(float);
    size_t b_bytes = (size_t)b_numel * sizeof(float);
    size_t out_bytes = (size_t)out_numel * sizeof(float);
    OglTensorSlot* sa = graph_ensure_device(a, a_bytes, 0);
    OglTensorSlot* sb = graph_ensure_device(b, b_bytes, 0);
    OglTensorSlot* so = graph_output_slot(out, out_bytes);
    if (!sa || !sb || !so) return 0;
    uint32_t metadata[28] = {(uint32_t)out_numel, (uint32_t)rank, (uint32_t)op, 0u};
    memcpy(metadata + 4, output_strides, 8 * sizeof(uint32_t));
    memcpy(metadata + 12, a_strides, 8 * sizeof(uint32_t));
    memcpy(metadata + 20, b_strides, 8 * sizeof(uint32_t));
    GLuint pb = params_buffer(metadata, sizeof(metadata));
    GLuint bufs[4] = {sa->buffer, sb->buffer, so->buffer, pb};
    int ok = dispatch_kernel(&k_broadcast_binary, bufs, ((uint32_t)out_numel + 63u) / 64u, 1, 1);
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
                            int batch, int in_c, int in_l, int out_c, int out_l, int kernel,
                            int stride, int pad, int relu) {
    if (!in || !weight || !out || batch <= 0 || in_c <= 0 || in_l <= 0 || out_c <= 0 || out_l <= 0 ||
        kernel <= 0 || stride <= 0 || pad < 0 || relu < 0 || relu > 1) return 0;
    int64_t padded_l = (int64_t)in_l + 2LL * pad;
    if (padded_l < kernel || (padded_l - kernel) / stride + 1 != out_l) return 0;
    size_t in_bytes = (size_t)batch * (size_t)in_c * in_l * sizeof(float);
    size_t wbytes = (size_t)out_c * in_c * kernel * sizeof(float);
    size_t bbytes = (size_t)out_c * sizeof(float);
    size_t out_bytes = (size_t)batch * (size_t)out_c * out_l * sizeof(float);
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
    uint32_t params[12] = {(uint32_t)in_c, (uint32_t)in_l, (uint32_t)out_c,
                           (uint32_t)kernel, (uint32_t)stride, (uint32_t)pad,
                           (uint32_t)relu, (uint32_t)batch, 1u, (uint32_t)in_c,
                           (uint32_t)out_l, 0u};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[5] = {src->buffer, sw->buffer, bias_buf, dst->buffer, pb};
    int ok = dispatch_kernel(&k_conv1d, bufs, ((uint32_t)out_l + 63u) / 64u,
                             (uint32_t)out_c, (uint32_t)batch);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (delete_bias && bias_buf) p_glDeleteBuffers(1, &bias_buf);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

static long attention_mask_numel(int mask_mode, int batch, int seq_q, int seq_kv) {
    if (mask_mode == 0) return 0;
    if (mask_mode == 1) return seq_kv;
    if (mask_mode == 2) return (long)batch * seq_kv;
    if (mask_mode == 3) return (long)seq_q * seq_kv;
    if (mask_mode == 4) return (long)batch * seq_q * seq_kv;
    return -1;
}

int opengl_graph_sdpa_f32(const float* qkv, const int32_t* mask, long mask_numel,
                          float* out, int seq_len, int d_model, int num_heads,
                          int head_dim, int batch, float scale, int causal, int mask_mode) {
    if (!ogl_ready() || !qkv || !out || batch <= 0 || seq_len <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 ||
        head_dim > 64 || d_model != num_heads * head_dim) return 0;
    long expected_mask = attention_mask_numel(mask_mode, batch, seq_len, seq_len);
    if (expected_mask < 0 || (mask_mode != 0 && (!mask || mask_numel != expected_mask))) return 0;
    size_t qkv_bytes = (size_t)batch * (size_t)seq_len * 3u * (size_t)d_model * sizeof(float);
    size_t out_bytes = (size_t)batch * (size_t)seq_len * (size_t)d_model * sizeof(float);
    size_t mask_bytes = mask_mode ? (size_t)expected_mask * sizeof(int32_t) : qkv_bytes;
    OglTensorSlot* src = graph_ensure_device(qkv, qkv_bytes, 0);
    OglTensorSlot* sm = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) : src;
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !sm || !dst) return 0;
    struct {
        uint32_t seq_len, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode;
    } params = {(uint32_t)seq_len, (uint32_t)d_model, (uint32_t)num_heads,
                (uint32_t)head_dim, (uint32_t)batch, scale,
                causal ? 1u : 0u, (uint32_t)mask_mode};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[4] = {src->buffer, sm->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_sdpa, bufs, ((uint32_t)seq_len + 63u) / 64u,
                             (uint32_t)num_heads, (uint32_t)batch);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

#if VOLVOXAI_ENABLE_TRAINING
int opengl_graph_sdpa_training_f32(const float* qkv, const int32_t* mask, long mask_numel,
                                   float* out, int seq_len, int d_model, int num_heads,
                                   int head_dim, int batch, float scale, int causal, int mask_mode,
                                   uint32_t threshold, uint32_t seed, uint32_t counter,
                                   float dropout_scale) {
    if (!ogl_ready() || !qkv || !out || batch <= 0 || seq_len <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 || head_dim > 64 || d_model != num_heads * head_dim ||
        !isfinite(dropout_scale) || dropout_scale < 1.0f) return 0;
    long expected_mask = attention_mask_numel(mask_mode, batch, seq_len, seq_len);
    if (expected_mask < 0 || (mask_mode != 0 && (!mask || mask_numel != expected_mask))) return 0;
    size_t qkv_bytes = (size_t)batch * (size_t)seq_len * 3u * (size_t)d_model * sizeof(float);
    size_t out_bytes = (size_t)batch * (size_t)seq_len * (size_t)d_model * sizeof(float);
    size_t mask_bytes = mask_mode ? (size_t)expected_mask * sizeof(int32_t) : qkv_bytes;
    OglTensorSlot* src = graph_ensure_device(qkv, qkv_bytes, 0);
    OglTensorSlot* sm = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) : src;
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !sm || !dst) return 0;
    struct {
        uint32_t seq_len, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, threshold, seed, counter;
        float dropout_scale;
    } params = {(uint32_t)seq_len, (uint32_t)d_model, (uint32_t)num_heads,
                (uint32_t)head_dim, (uint32_t)batch, scale, causal ? 1u : 0u,
                (uint32_t)mask_mode, threshold, seed, counter, dropout_scale};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[4] = {src->buffer, sm->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_sdpa_training, bufs, ((uint32_t)seq_len + 63u) / 64u,
                             (uint32_t)num_heads, (uint32_t)batch);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}
#endif

int opengl_graph_cross_sdpa_f32(const float* q, const float* k, const float* v,
                                const int32_t* mask, long mask_numel, float* out,
                                int seq_q, int seq_kv, int d_model, int num_heads,
                                int head_dim, int batch, float scale, int causal, int mask_mode) {
    if (!ogl_ready() || !q || !k || !v || !out || batch <= 0 || seq_q <= 0 ||
        seq_kv <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 || head_dim > 64 || d_model != num_heads * head_dim) return 0;
    long expected_mask = attention_mask_numel(mask_mode, batch, seq_q, seq_kv);
    if (expected_mask < 0 || (mask_mode != 0 && (!mask || mask_numel != expected_mask))) return 0;
    size_t q_bytes = (size_t)batch * (size_t)seq_q * (size_t)d_model * sizeof(float);
    size_t kv_bytes = (size_t)batch * (size_t)seq_kv * (size_t)d_model * sizeof(float);
    size_t out_bytes = q_bytes;
    size_t mask_bytes = mask_mode ? (size_t)expected_mask * sizeof(int32_t) : q_bytes;
    OglTensorSlot* sq = graph_ensure_device(q, q_bytes, 0);
    OglTensorSlot* sk = graph_ensure_device(k, kv_bytes, 0);
    OglTensorSlot* sv = graph_ensure_device(v, kv_bytes, 0);
    OglTensorSlot* sm = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) : sq;
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sq || !sk || !sv || !sm || !dst) return 0;
    struct {
        uint32_t seq_q, seq_kv, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, pad[3];
    } params = {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model,
                (uint32_t)num_heads, (uint32_t)head_dim, (uint32_t)batch, scale,
                causal ? 1u : 0u, (uint32_t)mask_mode, {0u, 0u, 0u}};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[6] = {sq->buffer, sk->buffer, sv->buffer, sm->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_cross_sdpa, bufs, ((uint32_t)seq_q + 63u) / 64u,
                             (uint32_t)num_heads, (uint32_t)batch);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

#if VOLVOXAI_ENABLE_TRAINING
int opengl_graph_cross_sdpa_training_f32(const float* q, const float* k, const float* v,
                                         const int32_t* mask, long mask_numel, float* out,
                                         int seq_q, int seq_kv, int d_model, int num_heads,
                                         int head_dim, int batch, float scale, int causal, int mask_mode,
                                         uint32_t threshold, uint32_t seed, uint32_t counter,
                                         float dropout_scale) {
    if (!ogl_ready() || !q || !k || !v || !out || batch <= 0 || seq_q <= 0 ||
        seq_kv <= 0 || d_model <= 0 || num_heads <= 0 || head_dim <= 0 ||
        head_dim > 64 || d_model != num_heads * head_dim ||
        !isfinite(dropout_scale) || dropout_scale < 1.0f) return 0;
    long expected_mask = attention_mask_numel(mask_mode, batch, seq_q, seq_kv);
    if (expected_mask < 0 || (mask_mode != 0 && (!mask || mask_numel != expected_mask))) return 0;
    size_t q_bytes = (size_t)batch * (size_t)seq_q * (size_t)d_model * sizeof(float);
    size_t kv_bytes = (size_t)batch * (size_t)seq_kv * (size_t)d_model * sizeof(float);
    size_t out_bytes = q_bytes;
    size_t mask_bytes = mask_mode ? (size_t)expected_mask * sizeof(int32_t) : q_bytes;
    OglTensorSlot* sq = graph_ensure_device(q, q_bytes, 0);
    OglTensorSlot* sk = graph_ensure_device(k, kv_bytes, 0);
    OglTensorSlot* sv = graph_ensure_device(v, kv_bytes, 0);
    OglTensorSlot* sm = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) : sq;
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sq || !sk || !sv || !sm || !dst) return 0;
    struct {
        uint32_t seq_q, seq_kv, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, threshold, seed, counter;
        float dropout_scale;
        uint32_t pad[3];
    } params = {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model,
                (uint32_t)num_heads, (uint32_t)head_dim, (uint32_t)batch, scale,
                causal ? 1u : 0u, (uint32_t)mask_mode, threshold, seed, counter,
                dropout_scale, {0u, 0u, 0u}};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[6] = {sq->buffer, sk->buffer, sv->buffer, sm->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_cross_sdpa_training, bufs, ((uint32_t)seq_q + 63u) / 64u,
                             (uint32_t)num_heads, (uint32_t)batch);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}
#endif

int opengl_graph_cross_attention_f32(const float* q, const float* kv, const float* weight,
                                     const float* scale, const float* bias, float* out,
                                     int seq_q, int seq_kv, int d_model, int num_heads,
                                     int head_dim, int batch, int has_scale, int has_bias) {
    if (!q || !kv || !weight || !out || seq_q <= 0 || seq_kv <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 || batch <= 0 || head_dim > 64 || d_model != num_heads * head_dim ||
        d_model > 64) return 0;
    size_t q_bytes = (size_t)batch * (size_t)seq_q * (size_t)d_model * sizeof(float);
    size_t kv_bytes = (size_t)batch * (size_t)seq_kv * (size_t)d_model * sizeof(float);
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
        uint32_t has_scale, has_bias, batch, pad[3];
    } params = {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model, (uint32_t)num_heads,
                (uint32_t)head_dim, 1.0f / sqrtf((float)head_dim),
                (uint32_t)(has_scale && scale), (uint32_t)(has_bias && bias), (uint32_t)batch, {0u, 0u, 0u}};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[7] = {sq->buffer, skv->buffer, sw->buffer, ss->buffer, sb->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_cross_attention, bufs, ((uint32_t)seq_q + 63u) / 64u,
                             (uint32_t)num_heads, (uint32_t)batch);
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

typedef struct {
    uint32_t rows;
    uint32_t d_in;
    uint32_t d_out;
    uint32_t input_type;
    uint32_t weight_type;
    uint32_t output_type;
    uint32_t pad0;
    uint32_t pad1;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad2;
    int32_t pad3;
    float input_scale;
    float output_scale;
    float pad4;
    float pad5;
} OglQLinearParams;

_Static_assert(sizeof(OglQLinearParams) == 64, "qLinearInt8 uniform ABI");

static int qlinear_dtype_bounds(uint32_t dtype, int32_t zero_point,
                                int64_t* maximum_distance) {
    int32_t minimum;
    int32_t maximum;
    if (dtype == VX_DTYPE_I8) {
        minimum = -128;
        maximum = 127;
    } else if (dtype == VX_DTYPE_U8) {
        minimum = 0;
        maximum = 255;
    } else {
        return 0;
    }
    if (zero_point < minimum || zero_point > maximum) return 0;
    int64_t lower = (int64_t)minimum - zero_point;
    int64_t upper = (int64_t)maximum - zero_point;
    *maximum_distance = llabs(lower) > llabs(upper) ? llabs(lower) : llabs(upper);
    return 1;
}

static int qlinear_gpu_args_valid(const void* input, const void* weight,
                                  const float* weight_scales,
                                  const int32_t* weight_zero_points,
                                  const int32_t* bias, void* output,
                                  uint32_t rows, uint32_t d_in, uint32_t d_out,
                                  float input_scale, int32_t input_zero_point,
                                  float output_scale, int32_t output_zero_point,
                                  uint32_t input_dtype, uint32_t weight_dtype,
                                  uint32_t output_dtype,
                                  size_t* input_bytes, size_t* weight_bytes,
                                  size_t* output_bytes) {
    if (!input || !weight || !weight_scales || !weight_zero_points || !bias || !output ||
        rows == 0 || d_in == 0 || d_out == 0 || !isfinite(input_scale) ||
        !isfinite(output_scale) || input_scale <= 0.0f || output_scale <= 0.0f) return 0;
    int64_t input_distance = 0;
    int64_t output_distance = 0;
    if (!qlinear_dtype_bounds(input_dtype, input_zero_point, &input_distance) ||
        !qlinear_dtype_bounds(output_dtype, output_zero_point, &output_distance)) return 0;
    (void)output_distance;
    if (weight_dtype != VX_DTYPE_I8 && weight_dtype != VX_DTYPE_U8) return 0;
    if (rows > UINT32_MAX / d_in || rows > UINT32_MAX / d_out ||
        d_out > UINT32_MAX / d_in) return 0;
    uint64_t input_elements = (uint64_t)rows * d_in;
    uint64_t weight_elements = (uint64_t)d_out * d_in;
    uint64_t output_elements = (uint64_t)rows * d_out;
    if (input_elements > SIZE_MAX || weight_elements > SIZE_MAX ||
        output_elements > SIZE_MAX ||
        (size_t)d_out > SIZE_MAX / sizeof(*weight_scales) ||
        (size_t)d_out > SIZE_MAX / sizeof(*weight_zero_points) ||
        (size_t)d_out > SIZE_MAX / sizeof(*bias)) return 0;
    for (uint32_t channel = 0; channel < d_out; channel++) {
        int64_t weight_distance = 0;
        if (!isfinite(weight_scales[channel]) || weight_scales[channel] <= 0.0f ||
            !qlinear_dtype_bounds(weight_dtype, weight_zero_points[channel],
                                  &weight_distance)) return 0;
        int64_t bias_distance = bias[channel] < 0 ? -(int64_t)bias[channel] : bias[channel];
        if (bias_distance > INT32_MAX) return 0;
        int64_t term_distance = input_distance * weight_distance;
        if (term_distance > 0 &&
            (uint64_t)d_in > (uint64_t)(INT32_MAX - bias_distance) /
                                  (uint64_t)term_distance) return 0;
    }
    *input_bytes = (size_t)input_elements;
    *weight_bytes = (size_t)weight_elements;
    *output_bytes = (size_t)output_elements;
    return 1;
}

int opengl_graph_qlinear_i8u8(const void* input, const void* weight,
                              const float* weight_scales, const int32_t* weight_zero_points,
                              const int32_t* bias, void* output,
                              uint32_t rows, uint32_t d_in, uint32_t d_out,
                              float input_scale, int32_t input_zero_point,
                              float output_scale, int32_t output_zero_point,
                              uint32_t input_dtype, uint32_t weight_dtype,
                              uint32_t output_dtype) {
    size_t input_bytes, weight_bytes, output_bytes;
    size_t multiplier_bytes;
    float* multipliers;
    if (max_ssbo_bindings < 6 || max_uniform_bindings < 1) return 0;
    if (!qlinear_gpu_args_valid(input, weight, weight_scales, weight_zero_points, bias, output,
                                rows, d_in, d_out, input_scale, input_zero_point,
                                output_scale, output_zero_point, input_dtype, weight_dtype,
                                output_dtype, &input_bytes, &weight_bytes, &output_bytes)) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    multiplier_bytes = (size_t)d_out * sizeof(*multipliers);
    multipliers = (float*)malloc(multiplier_bytes);
    if (!multipliers ||
        !vx_qlinear_build_multipliers(
            input_scale, weight_scales, output_scale, d_out, multipliers)) {
        free(multipliers);
        return 0;
    }
    OglTensorSlot* src = graph_ensure_packed_bytes(input, input_bytes, 0);
    OglTensorSlot* wt = graph_ensure_packed_bytes(weight, weight_bytes, 1);
    OglTensorSlot* zero_points = graph_ensure_device(weight_zero_points,
                                                      (size_t)d_out * sizeof(*weight_zero_points), 1);
    OglTensorSlot* biases = graph_ensure_device(bias, (size_t)d_out * sizeof(*bias), 1);
    OglTensorSlot* dst = graph_output_packed_bytes(output, output_bytes);
    if (!src || !wt || !zero_points || !biases || !dst) {
        free(multipliers);
        return 0;
    }
    GLuint multiplier_buffer = create_buffer(multiplier_bytes, multipliers);
    free(multipliers);
    if (!multiplier_buffer) return 0;
    OglQLinearParams params = {
        rows, d_in, d_out,
        input_dtype,
        weight_dtype,
        output_dtype, 0u, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, 0.0f, 0.0f
    };
    GLuint pb = params_buffer(&params, sizeof(params));
    if (!pb) {
        p_glDeleteBuffers(1, &multiplier_buffer);
        return 0;
    }
    GLuint bufs[7] = {
        src->buffer, wt->buffer, multiplier_buffer, zero_points->buffer,
        biases->buffer, dst->buffer, pb
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    int tiled = rows > 1u && d_in >= 16u && d_out >= 32u && (d_out & 3u) == 0u &&
                opengl_workgroup_supported(8u, 8u, 1u);
    uint32_t groups_x = tiled ? ((d_out / 4u + 7u) / 8u) : ((packed_words + 63u) / 64u);
    uint32_t groups_y = tiled ? ((rows + 7u) / 8u) : 1u;
    const OglKernel* kernel = tiled ? &k_qlinear_int8_tiled : &k_qlinear_int8;
    int ok = dispatch_kernel(kernel, bufs, groups_x, groups_y, 1u);
    p_glDeleteBuffers(1, &pb);
    p_glDeleteBuffers(1, &multiplier_buffer);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

typedef struct {
    uint32_t tokens;
    uint32_t vocab;
    uint32_t hidden;
    uint32_t weight_type;
    uint32_t output_type;
    int32_t output_zero_point;
    float output_scale;
    uint32_t pad0;
} OglQEmbeddingParams;

_Static_assert(sizeof(OglQEmbeddingParams) == 32, "qEmbeddingInt8 uniform ABI");

static int qembedding_gpu_args_valid(const int32_t* tokens, const void* weight,
                                     const float* weight_scales,
                                     const int32_t* weight_zero_points, void* output,
                                     uint32_t token_count, uint32_t vocab, uint32_t hidden,
                                     float output_scale, int32_t output_zero_point,
                                     uint32_t weight_dtype, uint32_t output_dtype,
                                     size_t* token_bytes, size_t* weight_bytes,
                                     size_t* output_bytes) {
    int64_t distance = 0;
    uint64_t weight_elements;
    uint64_t output_elements;
    if (!tokens || !weight || !weight_scales || !weight_zero_points || !output ||
        !token_count || !vocab || !hidden || !isfinite(output_scale) ||
        output_scale <= 0.0f || !qlinear_dtype_bounds(output_dtype, output_zero_point,
                                                        &distance) ||
        (weight_dtype != VX_DTYPE_I8 && weight_dtype != VX_DTYPE_U8) ||
        token_count > UINT32_MAX / hidden || vocab > UINT32_MAX / hidden) return 0;
    weight_elements = (uint64_t)vocab * hidden;
    output_elements = (uint64_t)token_count * hidden;
    if (weight_elements > SIZE_MAX || output_elements > SIZE_MAX ||
        (size_t)token_count > SIZE_MAX / sizeof(*tokens) ||
        (size_t)vocab > SIZE_MAX / sizeof(*weight_scales) ||
        (size_t)vocab > SIZE_MAX / sizeof(*weight_zero_points)) return 0;
    for (uint32_t row = 0; row < vocab; row++) {
        if (!isfinite(weight_scales[row]) || weight_scales[row] <= 0.0f ||
            !qlinear_dtype_bounds(weight_dtype, weight_zero_points[row], &distance)) return 0;
    }
    for (uint32_t token = 0; token < token_count; token++) {
        if (tokens[token] < 0 || (uint32_t)tokens[token] >= vocab) return 0;
    }
    *token_bytes = (size_t)token_count * sizeof(*tokens);
    *weight_bytes = (size_t)weight_elements;
    *output_bytes = (size_t)output_elements;
    return 1;
}

int opengl_graph_qembedding_i8u8(const int32_t* tokens, const void* weight,
                                 const float* weight_scales,
                                 const int32_t* weight_zero_points, void* output,
                                 uint32_t token_count, uint32_t vocab, uint32_t hidden,
                                 float output_scale, int32_t output_zero_point,
                                 uint32_t weight_dtype, uint32_t output_dtype) {
    size_t token_bytes;
    size_t weight_bytes;
    size_t output_bytes;
    size_t packed_output_bytes;
    if (max_ssbo_bindings < 5 || max_uniform_bindings < 1 ||
        !qembedding_gpu_args_valid(tokens, weight, weight_scales, weight_zero_points,
                                   output, token_count, vocab, hidden, output_scale,
                                   output_zero_point, weight_dtype, output_dtype,
                                   &token_bytes, &weight_bytes, &output_bytes) ||
        !graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    OglTensorSlot* ids = graph_ensure_device(tokens, token_bytes, 0);
    OglTensorSlot* table = graph_ensure_packed_bytes(weight, weight_bytes, 1);
    OglTensorSlot* scales = graph_ensure_device(weight_scales,
                                                 (size_t)vocab * sizeof(*weight_scales), 1);
    OglTensorSlot* zero_points = graph_ensure_device(weight_zero_points,
                                                      (size_t)vocab * sizeof(*weight_zero_points), 1);
    OglTensorSlot* dst = graph_output_packed_bytes(output, output_bytes);
    if (!ids || !table || !scales || !zero_points || !dst) return 0;
    OglQEmbeddingParams params = {
        token_count, vocab, hidden,
        weight_dtype,
        output_dtype, output_zero_point,
        output_scale, 0u
    };
    GLuint pb = params_buffer(&params, sizeof(params));
    if (!pb) return 0;
    GLuint bufs[6] = {
        ids->buffer, table->buffer, scales->buffer, zero_points->buffer,
        dst->buffer, pb
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    int ok = dispatch_kernel(&k_qembedding_int8, bufs,
                             (packed_words + 63u) / 64u, 1u, 1u);
    p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

typedef struct {
    uint32_t elements;
    uint32_t a_type;
    uint32_t b_type;
    uint32_t output_type;
    int32_t a_zero_point;
    int32_t b_zero_point;
    int32_t output_zero_point;
    int32_t pad0;
    float a_scale;
    float b_scale;
    float output_scale;
    uint32_t relu;
} OglQAddParams;

_Static_assert(sizeof(OglQAddParams) == 48, "qAdd uniform ABI");

typedef struct {
    uint32_t batch_rank;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t output_elements;
    uint32_t a_type;
    uint32_t b_type;
    uint32_t output_type;
    int32_t a_zero_point;
    int32_t b_zero_point;
    int32_t output_zero_point;
    int32_t pad0;
    float a_scale;
    float b_scale;
    float output_scale;
    float pad1;
} OglQBatchMatMulParams;

_Static_assert(sizeof(OglQBatchMatMulParams) == 64,
               "qBatchMatMul uniform ABI");

typedef struct {
    uint32_t elements;
    uint32_t input_type;
    uint32_t output_type;
    uint32_t pad0;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad1;
    int32_t pad2;
    float input_scale;
    float output_scale;
    float pad3;
    float pad4;
} OglQByteUnaryParams;

_Static_assert(sizeof(OglQByteUnaryParams) == 48,
               "qSiLUInt8/qGELUInt8 uniform ABI");

typedef struct {
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t groups;
    uint32_t input_type;
    uint32_t output_type;
    uint32_t pad0;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad1;
    int32_t pad2;
    float input_scale;
    float output_scale;
    float epsilon;
    float pad3;
} OglQGroupNormParams;

_Static_assert(sizeof(OglQGroupNormParams) == 64,
               "qGroupNormStats/qGroupNormApply uniform ABI");

typedef struct {
    uint32_t rows;
    uint32_t d_model;
    uint32_t input_type;
    uint32_t output_type;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad0;
    int32_t pad1;
    float input_scale;
    float output_scale;
    float epsilon;
    float pad2;
} OglQLayerNormParams;

_Static_assert(sizeof(OglQLayerNormParams) == 48,
               "qLayerNormStats/qLayerNormApply uniform ABI");

typedef struct {
    uint32_t seq_q;
    uint32_t seq_kv;
    uint32_t d_model;
    uint32_t heads;
    uint32_t batch;
    uint32_t mask_mode;
    uint32_t causal;
    uint32_t dtypes;
    int32_t q_zero_point;
    int32_t k_zero_point;
    int32_t v_zero_point;
    int32_t output_zero_point;
    float q_scale;
    float k_scale;
    float v_scale;
    float output_scale;
    float attention_scale;
    float pad0;
    float pad1;
    float pad2;
} OglQSDPAParams;

_Static_assert(sizeof(OglQSDPAParams) == 80, "qSDPAInt8 uniform ABI");

typedef struct {
    uint32_t outer;
    uint32_t axis_size;
    uint32_t inner;
    uint32_t input_dtype;
} OglQArgMaxParams;

_Static_assert(sizeof(OglQArgMaxParams) == 16, "qArgMaxInt8 uniform ABI");

typedef struct {
    uint32_t batch;
    uint32_t sequence;
    uint32_t width;
    uint32_t input_dtype;
    uint32_t output_dtype;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
    int32_t input_zero_point;
    int32_t output_zero_point;
    float input_scale;
    float output_scale;
} OglQMaskedMeanParams;

_Static_assert(sizeof(OglQMaskedMeanParams) == 48,
               "qMaskedMeanInt8 uniform ABI");

typedef struct {
    uint32_t elements;
    uint32_t input_type;
    uint32_t output_type;
    uint32_t pad0;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad1;
    int32_t pad2;
    float multiplier;
    float pad3;
    float pad4;
    float pad5;
} OglRequantizeLinearParams;

_Static_assert(sizeof(OglRequantizeLinearParams) == 48,
               "requantizeLinearTyped uniform ABI");

static int qbyte_dtype_zero_point_valid(uint32_t dtype, int32_t zero_point) {
    if (dtype == VX_DTYPE_I8) return zero_point >= -128 && zero_point <= 127;
    if (dtype == VX_DTYPE_U8) return zero_point >= 0 && zero_point <= 255;
    return 0;
}

static int qadd_gpu_args_valid(const void* a, uint32_t a_elements,
                               const void* b, uint32_t b_elements,
                               void* output, uint32_t output_elements,
                               float a_scale, int32_t a_zero_point,
                               float b_scale, int32_t b_zero_point,
                               float output_scale, int32_t output_zero_point,
                               uint32_t a_dtype, uint32_t b_dtype,
                               uint32_t output_dtype, uint32_t relu,
                               size_t* logical_bytes) {
    if (!a || !b || !output || output == a || output == b ||
        a_elements == 0 || a_elements != b_elements ||
        a_elements != output_elements || relu > 2u ||
        !isfinite(a_scale) || !isfinite(b_scale) || !isfinite(output_scale) ||
        a_scale <= 0.0f || b_scale <= 0.0f || output_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(a_dtype, a_zero_point) ||
        !qbyte_dtype_zero_point_valid(b_dtype, b_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point)) return 0;
    if ((uint64_t)a_elements > SIZE_MAX) return 0;
    *logical_bytes = (size_t)a_elements;
    return 1;
}

static int qbyte_unary_gpu_args_valid(const void* input, void* output, uint32_t elements,
                                      float input_scale, int32_t input_zero_point,
                                      float output_scale, int32_t output_zero_point,
                                      uint32_t input_dtype, uint32_t output_dtype,
                                      size_t* logical_bytes) {
    if (!input || !output || input == output || !elements ||
        !isfinite(input_scale) || !isfinite(output_scale) ||
        input_scale <= 0.0f || output_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        (uint64_t)elements > SIZE_MAX) return 0;
    *logical_bytes = (size_t)elements;
    return 1;
}

static float qnorm_f32_at(const float* values, uint32_t index) {
    float value;
    memcpy(&value, (const unsigned char*)values + (size_t)index * sizeof(value),
           sizeof(value));
    return value;
}

static int qgroupnorm_gpu_args_valid(const void* input, const float* weight,
                                     const float* bias, void* output,
                                     uint32_t batch, uint32_t height,
                                     uint32_t width, uint32_t channels,
                                     uint32_t groups, float input_scale,
                                     int32_t input_zero_point, float output_scale,
                                     int32_t output_zero_point, float epsilon,
                                     uint32_t input_dtype, uint32_t output_dtype,
                                     size_t* logical_bytes, size_t* affine_bytes,
                                     size_t* stats_bytes) {
    uint64_t elements;
    uint64_t group_count;
    if (!input || !weight || !bias || !output || output == input ||
        output == (const void*)weight || output == (const void*)bias ||
        !batch || !height || !width || !channels || !groups || channels % groups ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        batch > UINT32_MAX / groups) return 0;
    elements = batch;
    if (elements > UINT32_MAX / height) return 0;
    elements *= height;
    if (elements > UINT32_MAX / width) return 0;
    elements *= width;
    if (elements > UINT32_MAX / channels) return 0;
    elements *= channels;
    group_count = (uint64_t)batch * groups;
    if (elements == 0 || elements > SIZE_MAX || group_count == 0 ||
        group_count > SIZE_MAX / (2u * sizeof(float)) ||
        (size_t)channels > SIZE_MAX / sizeof(float)) return 0;
    for (uint32_t channel = 0; channel < channels; channel++) {
        if (!isfinite(qnorm_f32_at(weight, channel)) ||
            !isfinite(qnorm_f32_at(bias, channel))) return 0;
    }
    *logical_bytes = (size_t)elements;
    *affine_bytes = (size_t)channels * sizeof(float);
    *stats_bytes = (size_t)group_count * 2u * sizeof(float);
    return 1;
}

static int qlayernorm_gpu_args_valid(const void* input, const float* weight,
                                     const float* bias, void* output,
                                     uint32_t rows, uint32_t d_model,
                                     float input_scale, int32_t input_zero_point,
                                     float output_scale, int32_t output_zero_point,
                                     float epsilon, uint32_t input_dtype,
                                     uint32_t output_dtype, size_t* logical_bytes,
                                     size_t* affine_bytes, size_t* stats_bytes) {
    uint64_t elements;
    size_t row_count;
    if (!input || !weight || !bias || !output || output == input ||
        output == (const void*)weight || output == (const void*)bias ||
        !rows || !d_model || !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        rows > UINT32_MAX / d_model) return 0;
    elements = (uint64_t)rows * d_model;
    row_count = (size_t)rows;
    if (elements == 0 || elements > SIZE_MAX ||
        row_count > SIZE_MAX / (2u * sizeof(float)) ||
        (size_t)d_model > SIZE_MAX / sizeof(float)) return 0;
    for (uint32_t channel = 0; channel < d_model; channel++) {
        if (!isfinite(qnorm_f32_at(weight, channel)) ||
            !isfinite(qnorm_f32_at(bias, channel))) return 0;
    }
    *logical_bytes = (size_t)elements;
    *affine_bytes = (size_t)d_model * sizeof(float);
    *stats_bytes = row_count * 2u * sizeof(float);
    return 1;
}

static int qsdpa_ranges_overlap(const void* left, size_t left_bytes,
                                const void* right, size_t right_bytes) {
    uintptr_t left_start = (uintptr_t)left;
    uintptr_t right_start = (uintptr_t)right;
    if (left_start > UINTPTR_MAX - left_bytes ||
        right_start > UINTPTR_MAX - right_bytes) return 1;
    return left_start < right_start + right_bytes &&
        right_start < left_start + left_bytes;
}

static int qsdpa_centered_magnitude(uint32_t dtype, int32_t zero_point,
                                    uint64_t* magnitude_out) {
    int64_t low;
    int64_t high;
    uint64_t magnitude;
    if (!magnitude_out) return 0;
    if (dtype == VX_DTYPE_I8) {
        low = -128 - (int64_t)zero_point;
        high = 127 - (int64_t)zero_point;
    } else if (dtype == VX_DTYPE_U8) {
        low = -(int64_t)zero_point;
        high = 255 - (int64_t)zero_point;
    } else {
        return 0;
    }
    magnitude = (uint64_t)(low < 0 ? -low : low);
    if ((uint64_t)(high < 0 ? -high : high) > magnitude)
        magnitude = (uint64_t)(high < 0 ? -high : high);
    *magnitude_out = magnitude;
    return 1;
}

static int qsdpa_mask_bytes(uint32_t batch, uint32_t seq_q, uint32_t seq_kv,
                            uint32_t mask_mode, size_t* mask_bytes) {
    uint64_t elements = seq_kv;
    if (!mask_bytes) return 0;
    if (mask_mode == 0u) {
        *mask_bytes = 0;
        return 1;
    }
    if (mask_mode == 2u) {
        if (elements > UINT32_MAX / batch) return 0;
        elements *= batch;
    } else if (mask_mode == 3u) {
        if (elements > UINT32_MAX / seq_q) return 0;
        elements *= seq_q;
    } else if (mask_mode == 4u) {
        if (elements > UINT32_MAX / seq_q) return 0;
        elements *= seq_q;
        if (elements > UINT32_MAX / batch) return 0;
        elements *= batch;
    } else if (mask_mode != 1u) {
        return 0;
    }
    if (elements == 0 || elements > UINT32_MAX ||
        elements > SIZE_MAX / sizeof(int32_t)) return 0;
    *mask_bytes = (size_t)elements * sizeof(int32_t);
    return 1;
}

static int qsdpa_gpu_args_valid(const void* q, const void* k, const void* v,
                                const int32_t* mask, void* output,
                                uint32_t batch, uint32_t seq_q,
                                uint32_t seq_kv, uint32_t d_model,
                                uint32_t heads, float q_scale,
                                int32_t q_zero_point, float k_scale,
                                int32_t k_zero_point, float v_scale,
                                int32_t v_zero_point, float output_scale,
                                int32_t output_zero_point, float attention_scale,
                                uint32_t q_dtype, uint32_t k_dtype,
                                uint32_t v_dtype, uint32_t output_dtype,
                                uint32_t causal, uint32_t mask_mode,
                                size_t* q_bytes, size_t* kv_bytes,
                                size_t* mask_bytes) {
    uint64_t q_elements = batch;
    uint64_t kv_elements = batch;
    uint64_t q_magnitude;
    uint64_t k_magnitude;
    uint64_t maximum_dot;
    uint32_t head_dim;
    float qk_scale;
    float score_scale;
    if (!q || !k || !v || !output || !batch || !seq_q || !seq_kv ||
        !d_model || !heads || causal > 1u || d_model % heads ||
        d_model % 4u || !isfinite(q_scale) || q_scale <= 0.0f ||
        !isfinite(k_scale) || k_scale <= 0.0f || !isfinite(v_scale) ||
        v_scale <= 0.0f || !isfinite(output_scale) || output_scale <= 0.0f ||
        !isfinite(attention_scale) || attention_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(q_dtype, q_zero_point) ||
        !qbyte_dtype_zero_point_valid(k_dtype, k_zero_point) ||
        !qbyte_dtype_zero_point_valid(v_dtype, v_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        !q_bytes || !kv_bytes || !mask_bytes) return 0;
    head_dim = d_model / heads;
    if (!head_dim || head_dim % 4u || head_dim > 64u ||
        q_elements > UINT32_MAX / seq_q) return 0;
    q_elements *= seq_q;
    if (q_elements > UINT32_MAX / d_model) return 0;
    q_elements *= d_model;
    if (kv_elements > UINT32_MAX / seq_kv) return 0;
    kv_elements *= seq_kv;
    if (kv_elements > UINT32_MAX / d_model) return 0;
    kv_elements *= d_model;
    if (q_elements == 0 || kv_elements == 0 || q_elements > SIZE_MAX ||
        kv_elements > SIZE_MAX || !qsdpa_mask_bytes(batch, seq_q, seq_kv,
                                                     mask_mode, mask_bytes) ||
        (mask_mode == 0u ? mask != NULL : mask == NULL)) return 0;
    qk_scale = q_scale * k_scale;
    score_scale = qk_scale * attention_scale;
    if (!isfinite(qk_scale) || qk_scale <= 0.0f ||
        !isfinite(score_scale) || score_scale <= 0.0f ||
        !qsdpa_centered_magnitude(q_dtype, q_zero_point, &q_magnitude) ||
        !qsdpa_centered_magnitude(k_dtype, k_zero_point, &k_magnitude)) return 0;
    maximum_dot = q_magnitude * k_magnitude * head_dim;
    if (!isfinite((float)maximum_dot * score_scale)) return 0;
    *q_bytes = (size_t)q_elements;
    *kv_bytes = (size_t)kv_elements;
    if (qsdpa_ranges_overlap(output, *q_bytes, q, *q_bytes) ||
        qsdpa_ranges_overlap(output, *q_bytes, k, *kv_bytes) ||
        qsdpa_ranges_overlap(output, *q_bytes, v, *kv_bytes) ||
        (*mask_bytes && qsdpa_ranges_overlap(output, *q_bytes, mask,
                                             *mask_bytes))) return 0;
    return 1;
}

static int qargmax_gpu_args_valid(const void* input, int32_t* output,
                                  uint32_t outer, uint32_t axis_size,
                                  uint32_t inner, uint32_t input_dtype,
                                  size_t* input_bytes, size_t* output_bytes,
                                  uint32_t* output_elements_out) {
    uint64_t input_elements = outer;
    uint64_t output_elements = outer;
    if (!input || !output || !outer || !axis_size || !inner ||
        axis_size > (uint32_t)INT32_MAX ||
        (input_dtype != VX_DTYPE_I8 && input_dtype != VX_DTYPE_U8) ||
        !input_bytes || !output_bytes || !output_elements_out ||
        input_elements > UINT32_MAX / axis_size) return 0;
    input_elements *= axis_size;
    if (input_elements > UINT32_MAX / inner ||
        output_elements > UINT32_MAX / inner) return 0;
    input_elements *= inner;
    output_elements *= inner;
    if (!input_elements || !output_elements || input_elements > SIZE_MAX ||
        output_elements > UINT32_MAX ||
        output_elements > SIZE_MAX / sizeof(*output)) return 0;
    *input_bytes = (size_t)input_elements;
    *output_bytes = (size_t)output_elements * sizeof(*output);
    *output_elements_out = (uint32_t)output_elements;
    return !qsdpa_ranges_overlap(output, *output_bytes, input, *input_bytes);
}

static int qmaskedmean_gpu_args_valid(const void* input, const int32_t* mask,
                                      void* output, uint32_t batch,
                                      uint32_t sequence, uint32_t width,
                                      float input_scale, int32_t input_zero_point,
                                      float output_scale, int32_t output_zero_point,
                                      uint32_t input_dtype, uint32_t output_dtype,
                                      size_t* input_bytes, size_t* mask_bytes,
                                      size_t* output_bytes) {
    uint64_t input_elements = batch;
    uint64_t mask_elements = batch;
    uint64_t output_elements = batch;
    int64_t low;
    int64_t high;
    uint64_t magnitude;
    float multiplier;
    if (!input || !mask || !output || !batch || !sequence || !width ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        !input_bytes || !mask_bytes || !output_bytes ||
        input_elements > UINT32_MAX / sequence) return 0;
    input_elements *= sequence;
    if (input_elements > UINT32_MAX / width ||
        mask_elements > UINT32_MAX / sequence ||
        output_elements > UINT32_MAX / width) return 0;
    input_elements *= width;
    mask_elements *= sequence;
    output_elements *= width;
    if (!input_elements || !mask_elements || !output_elements ||
        input_elements > SIZE_MAX || mask_elements > SIZE_MAX / sizeof(*mask) ||
        output_elements > SIZE_MAX) return 0;
    if (input_dtype == VX_DTYPE_I8) {
        low = -128 - (int64_t)input_zero_point;
        high = 127 - (int64_t)input_zero_point;
    } else {
        low = -(int64_t)input_zero_point;
        high = 255 - (int64_t)input_zero_point;
    }
    magnitude = (uint64_t)(low < 0 ? -low : low);
    if ((uint64_t)(high < 0 ? -high : high) > magnitude)
        magnitude = (uint64_t)(high < 0 ? -high : high);
    multiplier = input_scale / output_scale;
    if ((magnitude != 0 && (uint64_t)sequence > (uint64_t)INT32_MAX / magnitude) ||
        !isfinite(multiplier) || multiplier <= 0.0f) return 0;
    *input_bytes = (size_t)input_elements;
    *mask_bytes = (size_t)mask_elements * sizeof(*mask);
    *output_bytes = (size_t)output_elements;
    return !qsdpa_ranges_overlap(output, *output_bytes, input, *input_bytes) &&
        !qsdpa_ranges_overlap(output, *output_bytes, mask, *mask_bytes);
}

static int requantize_gpu_args_valid(const void* input, uint32_t input_elements,
                                     void* output, uint32_t output_elements,
                                     float input_scale, int32_t input_zero_point,
                                     float output_scale, int32_t output_zero_point,
                                     uint32_t input_dtype, uint32_t output_dtype,
                                     float* multiplier, size_t* logical_bytes) {
    if (!input || !output || input == output || input_elements == 0 ||
        input_elements != output_elements || !isfinite(input_scale) ||
        !isfinite(output_scale) || input_scale <= 0.0f || output_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point)) return 0;
    float ratio = input_scale / output_scale;
    if (!isfinite(ratio) || ratio <= 0.0f || (uint64_t)input_elements > SIZE_MAX) return 0;
    *multiplier = ratio;
    *logical_bytes = (size_t)input_elements;
    return 1;
}

int opengl_graph_qbatch_matmul_i8u8(
        const void* a, const int* a_shape, int a_rank,
        float a_scale, int32_t a_zero_point, uint32_t a_dtype,
        const void* b, const int* b_shape, int b_rank,
        float b_scale, int32_t b_zero_point, uint32_t b_dtype,
        void* output, const int* output_shape, int output_rank,
        float output_scale, int32_t output_zero_point,
        uint32_t output_dtype) {
    VxQBatchMatMulDevicePlan plan;
    uint32_t metadata[24] = {0};
    uint32_t metadata_words;
    size_t metadata_bytes;
    size_t a_packed_bytes;
    size_t b_packed_bytes;
    size_t output_packed_bytes;
    if (!ogl_ready() || max_ssbo_bindings < 4 ||
        max_uniform_bindings < 1 ||
        !vx_qbatch_matmul_device_plan(
            a, a_shape, a_rank, a_scale, a_zero_point, a_dtype,
            b, b_shape, b_rank, b_scale, b_zero_point, b_dtype,
            output, output_shape, output_rank, output_scale,
            output_zero_point, output_dtype, &plan) ||
        !graph_packed_bytes(plan.a_bytes, &a_packed_bytes) ||
        !graph_packed_bytes(plan.b_bytes, &b_packed_bytes) ||
        !graph_packed_bytes(plan.output_bytes, &output_packed_bytes))
        return 0;
    OglTensorSlot* a_slot =
        graph_ensure_packed_bytes(a, plan.a_bytes, 0);
    OglTensorSlot* b_slot =
        graph_ensure_packed_bytes(b, plan.b_bytes, 0);
    OglTensorSlot* output_slot =
        graph_output_packed_bytes(output, plan.output_bytes);
    if (!a_slot || !b_slot || !output_slot) return 0;
    memcpy(metadata, plan.output_batch_strides,
           plan.batch_rank * sizeof(uint32_t));
    memcpy(metadata + plan.batch_rank, plan.a_batch_strides,
           plan.batch_rank * sizeof(uint32_t));
    memcpy(metadata + 2u * plan.batch_rank, plan.b_batch_strides,
           plan.batch_rank * sizeof(uint32_t));
    metadata_words = plan.batch_rank ? 3u * plan.batch_rank : 1u;
    metadata_bytes = (size_t)metadata_words * sizeof(uint32_t);
    OglQBatchMatMulParams params = {
        plan.batch_rank, plan.m, plan.k, plan.n,
        plan.output_elements, a_dtype, b_dtype, output_dtype,
        a_zero_point, b_zero_point, output_zero_point, 0,
        a_scale, b_scale, output_scale, 0.0f,
    };
    GLuint metadata_buffer = create_buffer(metadata_bytes, metadata);
    GLuint params_handle = params_buffer(&params, sizeof(params));
    if (!metadata_buffer || !params_handle) {
        if (metadata_buffer) p_glDeleteBuffers(1, &metadata_buffer);
        if (params_handle) p_glDeleteBuffers(1, &params_handle);
        return 0;
    }
    GLuint buffers[5] = {
        a_slot->buffer, b_slot->buffer, output_slot->buffer,
        metadata_buffer, params_handle,
    };
    uint32_t packed_words =
        (uint32_t)(output_packed_bytes / sizeof(uint32_t));
    int ok = dispatch_kernel(
        &k_qbatch_matmul_i8u8, buffers,
        (packed_words + 63u) / 64u, 1u, 1u);
    p_glDeleteBuffers(1, &metadata_buffer);
    p_glDeleteBuffers(1, &params_handle);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_qadd_i8u8(const void* a, uint32_t a_elements,
                           const void* b, uint32_t b_elements,
                           void* output, uint32_t output_elements,
                           float a_scale, int32_t a_zero_point,
                           float b_scale, int32_t b_zero_point,
                           float output_scale, int32_t output_zero_point,
                           uint32_t a_dtype, uint32_t b_dtype,
                           uint32_t output_dtype, uint32_t relu) {
    if (!ogl_ready() || max_ssbo_bindings < 3 || max_uniform_bindings < 1) return 0;
    size_t logical_bytes;
    if (!qadd_gpu_args_valid(a, a_elements, b, b_elements, output, output_elements,
                             a_scale, a_zero_point, b_scale, b_zero_point,
                             output_scale, output_zero_point, a_dtype, b_dtype,
                             output_dtype, relu, &logical_bytes)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    OglTensorSlot* a_slot = graph_ensure_packed_bytes(a, logical_bytes, 0);
    OglTensorSlot* b_slot = graph_ensure_packed_bytes(b, logical_bytes, 0);
    OglTensorSlot* output_slot = graph_output_packed_bytes(output, logical_bytes);
    if (!a_slot || !b_slot || !output_slot) return 0;
    OglQAddParams params = {
        a_elements, a_dtype,
        b_dtype,
        output_dtype,
        a_zero_point, b_zero_point, output_zero_point, 0,
        a_scale, b_scale, output_scale, relu
    };
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[4] = {
        a_slot->buffer, b_slot->buffer, output_slot->buffer, pb
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    int ok = dispatch_kernel(&k_qadd_i8u8, bufs, (packed_words + 63u) / 64u,
                             1u, 1u);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_qsilu_i8u8(const void* input, void* output, uint32_t elements,
                            float input_scale, int32_t input_zero_point,
                            float output_scale, int32_t output_zero_point,
                            uint32_t input_dtype, uint32_t output_dtype) {
    if (!ogl_ready() || max_ssbo_bindings < 2 || max_uniform_bindings < 1) return 0;
    size_t logical_bytes;
    if (!qbyte_unary_gpu_args_valid(input, output, elements, input_scale,
                                    input_zero_point, output_scale, output_zero_point,
                                    input_dtype, output_dtype, &logical_bytes)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    OglTensorSlot* input_slot = graph_ensure_packed_bytes(input, logical_bytes, 0);
    OglTensorSlot* output_slot = graph_output_packed_bytes(output, logical_bytes);
    if (!input_slot || !output_slot) return 0;
    OglQByteUnaryParams params = {
        elements, input_dtype,
        output_dtype, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, 0.0f, 0.0f
    };
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {input_slot->buffer, output_slot->buffer, pb};
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    int ok = dispatch_kernel(&k_qsilu_i8u8, bufs, (packed_words + 63u) / 64u,
                             1u, 1u);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_qgelu_i8u8(const void* input, void* output, uint32_t elements,
                            float input_scale, int32_t input_zero_point,
                            float output_scale, int32_t output_zero_point,
                            uint32_t input_dtype, uint32_t output_dtype) {
    if (!ogl_ready() || max_ssbo_bindings < 2 || max_uniform_bindings < 1) return 0;
    size_t logical_bytes;
    if (!qbyte_unary_gpu_args_valid(input, output, elements, input_scale,
                                    input_zero_point, output_scale, output_zero_point,
                                    input_dtype, output_dtype, &logical_bytes)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    OglTensorSlot* input_slot = graph_ensure_packed_bytes(input, logical_bytes, 0);
    OglTensorSlot* output_slot = graph_output_packed_bytes(output, logical_bytes);
    if (!input_slot || !output_slot) return 0;
    OglQByteUnaryParams params = {
        elements, input_dtype,
        output_dtype, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, 0.0f, 0.0f
    };
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {input_slot->buffer, output_slot->buffer, pb};
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    int ok = dispatch_kernel(&k_qgelu_i8u8, bufs, (packed_words + 63u) / 64u,
                             1u, 1u);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

/* Statistics writes one F32 pair per [batch, group]; apply then owns whole
 * packed output words. The explicit storage barrier in dispatch_kernel makes
 * the first pass visible to the second without exposing an F32 activation. */
int opengl_graph_qgroupnorm_i8u8(const void* input, const float* weight,
                                 const float* bias, void* output, uint32_t batch,
                                 uint32_t height, uint32_t width, uint32_t channels,
                                 uint32_t groups, float input_scale,
                                 int32_t input_zero_point, float output_scale,
                                 int32_t output_zero_point, float epsilon,
                                 uint32_t input_dtype, uint32_t output_dtype) {
    size_t logical_bytes;
    size_t affine_bytes;
    size_t stats_bytes;
    size_t packed_bytes;
    if (!ogl_ready() || max_ssbo_bindings < 5 || max_uniform_bindings < 1 ||
        !qgroupnorm_gpu_args_valid(input, weight, bias, output, batch, height,
                                   width, channels, groups, input_scale,
                                   input_zero_point, output_scale,
                                   output_zero_point, epsilon, input_dtype,
                                   output_dtype, &logical_bytes, &affine_bytes,
                                   &stats_bytes) ||
        stats_bytes > (size_t)INTPTR_MAX ||
        !graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    OglTensorSlot* input_slot = graph_ensure_packed_bytes(input, logical_bytes, 0);
    OglTensorSlot* weight_slot = graph_ensure_device(weight, affine_bytes, 1);
    OglTensorSlot* bias_slot = graph_ensure_device(bias, affine_bytes, 1);
    OglTensorSlot* output_slot = graph_output_packed_bytes(output, logical_bytes);
    if (!input_slot || !weight_slot || !bias_slot || !output_slot) return 0;
    OglQGroupNormParams params = {
        batch, height, width, channels, groups,
        input_dtype,
        output_dtype, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, epsilon, 0.0f
    };
    GLuint stats_buffer = qgroupnorm_stats_ensure(stats_bytes);
    GLuint params_buffer_handle = params_buffer(&params, sizeof(params));
    if (!stats_buffer || !params_buffer_handle) {
        if (params_buffer_handle) p_glDeleteBuffers(1, &params_buffer_handle);
        return 0;
    }
    GLuint stats_binds[3] = {input_slot->buffer, stats_buffer, params_buffer_handle};
    GLuint apply_binds[6] = {
        input_slot->buffer, weight_slot->buffer, bias_slot->buffer, stats_buffer,
        output_slot->buffer, params_buffer_handle
    };
    uint32_t total_groups = batch * groups;
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    uint32_t apply_groups = packed_words / 64u + (packed_words % 64u != 0u);
    int ok = dispatch_kernel(&k_qgroupnorm_stats, stats_binds, total_groups, 1u, 1u) &&
             dispatch_kernel(&k_qgroupnorm_apply, apply_binds, apply_groups, 1u, 1u);
    p_glDeleteBuffers(1, &params_buffer_handle);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

/* Stats and apply are deliberately separate: the first pass owns a final-axis
 * row and the second owns complete packed output words, with the dispatcher
 * storage barrier making F32 row stats visible between them. */
int opengl_graph_qlayernorm_i8u8(const void* input, const float* weight,
                                 const float* bias, void* output, uint32_t rows,
                                 uint32_t d_model, float input_scale,
                                 int32_t input_zero_point, float output_scale,
                                 int32_t output_zero_point, float epsilon,
                                 uint32_t input_dtype, uint32_t output_dtype) {
    size_t logical_bytes;
    size_t affine_bytes;
    size_t stats_bytes;
    size_t packed_bytes;
    if (!ogl_ready() || max_ssbo_bindings < 5 || max_uniform_bindings < 1 ||
        !qlayernorm_gpu_args_valid(input, weight, bias, output, rows, d_model,
                                   input_scale, input_zero_point, output_scale,
                                   output_zero_point, epsilon, input_dtype,
                                   output_dtype, &logical_bytes, &affine_bytes,
                                   &stats_bytes) || stats_bytes > (size_t)INTPTR_MAX ||
        !graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    OglTensorSlot* input_slot = graph_ensure_packed_bytes(input, logical_bytes, 0);
    OglTensorSlot* weight_slot = graph_ensure_device(weight, affine_bytes, 1);
    OglTensorSlot* bias_slot = graph_ensure_device(bias, affine_bytes, 1);
    OglTensorSlot* output_slot = graph_output_packed_bytes(output, logical_bytes);
    if (!input_slot || !weight_slot || !bias_slot || !output_slot) return 0;
    OglQLayerNormParams params = {
        rows, d_model, input_dtype,
        output_dtype,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, epsilon, 0.0f
    };
    GLuint stats_buffer = qlayernorm_stats_ensure(stats_bytes);
    GLuint params_buffer_handle = params_buffer(&params, sizeof(params));
    if (!stats_buffer || !params_buffer_handle) {
        if (params_buffer_handle) p_glDeleteBuffers(1, &params_buffer_handle);
        return 0;
    }
    GLuint stats_binds[3] = {input_slot->buffer, stats_buffer, params_buffer_handle};
    GLuint apply_binds[6] = {
        input_slot->buffer, weight_slot->buffer, bias_slot->buffer, stats_buffer,
        output_slot->buffer, params_buffer_handle
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    uint32_t apply_groups = packed_words / 64u + (packed_words % 64u != 0u);
    int ok = dispatch_kernel(&k_qlayernorm_stats, stats_binds, rows, 1u, 1u) &&
             dispatch_kernel(&k_qlayernorm_apply, apply_binds, apply_groups, 1u, 1u);
    p_glDeleteBuffers(1, &params_buffer_handle);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_qsdpa_i8u8(const void* q, const void* k, const void* v,
                            const int32_t* mask, void* output, uint32_t batch,
                            uint32_t seq_q, uint32_t seq_kv, uint32_t d_model,
                            uint32_t heads, float q_scale, int32_t q_zero_point,
                            float k_scale, int32_t k_zero_point, float v_scale,
                            int32_t v_zero_point, float output_scale,
                            int32_t output_zero_point, float attention_scale,
                            uint32_t q_dtype, uint32_t k_dtype, uint32_t v_dtype,
                            uint32_t output_dtype, uint32_t causal,
                            uint32_t mask_mode) {
    size_t q_bytes;
    size_t kv_bytes;
    size_t mask_bytes;
    size_t packed_output_bytes;
    if (!ogl_ready() || max_ssbo_bindings < 5 || max_uniform_bindings < 1 ||
        !qsdpa_gpu_args_valid(q, k, v, mask, output, batch, seq_q, seq_kv,
                               d_model, heads, q_scale, q_zero_point, k_scale,
                               k_zero_point, v_scale, v_zero_point, output_scale,
                               output_zero_point, attention_scale, q_dtype, k_dtype,
                               v_dtype, output_dtype, causal, mask_mode, &q_bytes,
                               &kv_bytes, &mask_bytes) ||
        !graph_packed_bytes(q_bytes, &packed_output_bytes)) return 0;
    OglTensorSlot* q_slot = graph_ensure_packed_bytes(q, q_bytes, 0);
    OglTensorSlot* k_slot = graph_ensure_packed_bytes(k, kv_bytes, 0);
    OglTensorSlot* v_slot = graph_ensure_packed_bytes(v, kv_bytes, 0);
    OglTensorSlot* mask_slot = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) :
        graph_ensure_device(qsdpa_dummy_mask, sizeof(qsdpa_dummy_mask), 1);
    OglTensorSlot* output_slot = graph_output_packed_bytes(output, q_bytes);
    if (!q_slot || !k_slot || !v_slot || !mask_slot || !output_slot) return 0;
    OglQSDPAParams params = {
        seq_q, seq_kv, d_model, heads,
        batch, mask_mode, causal,
        q_dtype |
            (k_dtype << 8u) |
            (v_dtype << 16u) |
            (output_dtype << 24u),
        q_zero_point, k_zero_point, v_zero_point, output_zero_point,
        q_scale, k_scale, v_scale, output_scale,
        attention_scale, 0.0f, 0.0f, 0.0f
    };
    GLuint params_buffer_handle = params_buffer(&params, sizeof(params));
    if (!params_buffer_handle) return 0;
    GLuint binds[6] = {
        q_slot->buffer, k_slot->buffer, v_slot->buffer, mask_slot->buffer,
        output_slot->buffer, params_buffer_handle
    };
    int ok = dispatch_kernel(&k_qsdpa_int8, binds, seq_q, heads, batch);
    p_glDeleteBuffers(1, &params_buffer_handle);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

/* qArgMaxInt8 owns one output index per [outer,inner] coordinate. Input stays
 * byte-packed, output is conventional I32 storage, and the 16-byte uniform
 * exactly mirrors the WebGPU ABI. */
int opengl_graph_qargmax_i8u8(const void* input, int32_t* output,
                              uint32_t outer, uint32_t axis_size,
                              uint32_t inner, uint32_t input_dtype) {
    size_t input_bytes;
    size_t output_bytes;
    uint32_t output_elements;
    if (!ogl_ready() || max_ssbo_bindings < 2 || max_uniform_bindings < 1 ||
        !qargmax_gpu_args_valid(input, output, outer, axis_size, inner,
                                input_dtype, &input_bytes, &output_bytes,
                                &output_elements)) return 0;
    OglTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_bytes, 0);
    OglTensorSlot* output_slot = graph_output_slot(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    OglQArgMaxParams params = {
        outer, axis_size, inner, input_dtype
    };
    GLuint params_buffer_handle = params_buffer(&params, sizeof(params));
    if (!params_buffer_handle) return 0;
    GLuint binds[3] = {
        input_slot->buffer, output_slot->buffer, params_buffer_handle
    };
    int ok = dispatch_kernel(&k_qargmax_int8, binds,
                             output_elements / 64u + (output_elements % 64u != 0u),
                             1u, 1u);
    p_glDeleteBuffers(1, &params_buffer_handle);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_qmaskedmean_i8u8(const void* input, const int32_t* mask,
                                  void* output, uint32_t batch,
                                  uint32_t sequence, uint32_t width,
                                  float input_scale, int32_t input_zero_point,
                                  float output_scale, int32_t output_zero_point,
                                  uint32_t input_dtype, uint32_t output_dtype) {
    size_t input_bytes;
    size_t mask_bytes;
    size_t output_bytes;
    size_t packed_output_bytes;
    if (!ogl_ready() || max_ssbo_bindings < 3 || max_uniform_bindings < 1 ||
        !qmaskedmean_gpu_args_valid(input, mask, output, batch, sequence, width,
                                    input_scale, input_zero_point, output_scale,
                                    output_zero_point, input_dtype, output_dtype,
                                    &input_bytes, &mask_bytes, &output_bytes) ||
        !graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    OglTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_bytes, 0);
    OglTensorSlot* mask_slot = graph_ensure_device(mask, mask_bytes, 0);
    OglTensorSlot* output_slot = graph_output_packed_bytes(output, output_bytes);
    if (!input_slot || !mask_slot || !output_slot) return 0;
    OglQMaskedMeanParams params = {
        batch, sequence, width, input_dtype,
        output_dtype, 0u, 0u, 0u,
        input_zero_point, output_zero_point, input_scale, output_scale
    };
    GLuint params_buffer_handle = params_buffer(&params, sizeof(params));
    if (!params_buffer_handle) return 0;
    GLuint binds[4] = {
        input_slot->buffer, mask_slot->buffer, output_slot->buffer,
        params_buffer_handle
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    int ok = dispatch_kernel(&k_qmaskedmean_int8, binds,
                             (packed_words + 63u) / 64u, 1u, 1u);
    p_glDeleteBuffers(1, &params_buffer_handle);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_requantize_linear_i8u8(const void* input, uint32_t input_elements,
                                        void* output, uint32_t output_elements,
                                        float input_scale, int32_t input_zero_point,
                                        float output_scale, int32_t output_zero_point,
                                        uint32_t input_dtype, uint32_t output_dtype) {
    if (!ogl_ready() || max_ssbo_bindings < 2 || max_uniform_bindings < 1) return 0;
    float multiplier;
    size_t logical_bytes;
    if (!requantize_gpu_args_valid(input, input_elements, output, output_elements,
                                   input_scale, input_zero_point,
                                   output_scale, output_zero_point,
                                   input_dtype, output_dtype,
                                   &multiplier, &logical_bytes)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    OglTensorSlot* input_slot = graph_ensure_packed_bytes(input, logical_bytes, 0);
    OglTensorSlot* output_slot = graph_output_packed_bytes(output, logical_bytes);
    if (!input_slot || !output_slot) return 0;
    OglRequantizeLinearParams params = {
        input_elements, input_dtype,
        output_dtype, 0u,
        input_zero_point, output_zero_point, 0, 0,
        multiplier, 0.0f, 0.0f, 0.0f
    };
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {input_slot->buffer, output_slot->buffer, pb};
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    int ok = dispatch_kernel(&k_requantize_linear_i8u8, bufs,
                             (packed_words + 63u) / 64u, 1u, 1u);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

static const int32_t* qconv_zero_bias_get(uint32_t output_channels) {
    if (output_channels == 0 ||
        (size_t)output_channels > SIZE_MAX / sizeof(int32_t)) return NULL;
    for (QConvZeroBiasBacking* block = qconv_zero_bias_backings; block;
         block = block->next) {
        /* Domain-enforced invariant slots retain an exact readable range.
         * Give every distinct channel width a stable host key so a later
         * smaller zero bias cannot narrow an earlier larger binding. */
        if (block->elements == output_channels) return block->values;
    }
    QConvZeroBiasBacking* block = (QConvZeroBiasBacking*)calloc(1, sizeof(*block));
    if (!block) return NULL;
    block->values = (int32_t*)calloc((size_t)output_channels, sizeof(*block->values));
    if (!block->values) {
        free(block);
        return NULL;
    }
    block->elements = output_channels;
    block->next = qconv_zero_bias_backings;
    qconv_zero_bias_backings = block;
    return block->values;
}

static void qconv_zero_bias_release_state(OpenGLContextState* state) {
    if (!state) return;
    while (state->qconv_zero_bias_storage) {
        QConvZeroBiasBacking* block = state->qconv_zero_bias_storage;
        state->qconv_zero_bias_storage = block->next;
        free(block->values);
        free(block);
    }
}

typedef struct {
    uint32_t batch;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t input_channels;
    uint32_t output_height;
    uint32_t output_width;
    uint32_t output_channels;
    uint32_t kernel_height;
    uint32_t kernel_width;
    uint32_t stride_y;
    uint32_t stride_x;
    uint32_t dilation_y;
    uint32_t dilation_x;
    uint32_t pad_top;
    uint32_t pad_left;
    uint32_t groups;
    uint32_t input_type;
    uint32_t weight_type;
    uint32_t output_type;
    uint32_t relu;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad0;
    int32_t pad1;
    float input_scale;
    float output_scale;
    float pad2;
    float pad3;
} OglQConv2DParams;

_Static_assert(sizeof(OglQConv2DParams) == 112, "qConv2DInt8 uniform ABI");

static int qconv_mul_u64(uint64_t left, uint64_t right, uint64_t* out) {
    if (!out || (left != 0 && right > UINT64_MAX / left)) return 0;
    *out = left * right;
    return 1;
}

static int32_t qconv_i32_at(const int32_t* values, uint32_t index) {
    int32_t value;
    memcpy(&value, (const unsigned char*)values + (size_t)index * sizeof(value),
           sizeof(value));
    return value;
}

static float qconv_f32_at(const float* values, uint32_t index) {
    float value;
    memcpy(&value, (const unsigned char*)values + (size_t)index * sizeof(value),
           sizeof(value));
    return value;
}

static int qconv_gpu_args_valid(const void* input, const void* weight,
                                const float* weight_scales,
                                const int32_t* weight_zero_points,
                                const int32_t* bias, void* output,
                                uint32_t batch, uint32_t input_height,
                                uint32_t input_width, uint32_t input_channels,
                                uint32_t output_height, uint32_t output_width,
                                uint32_t output_channels, uint32_t kernel_height,
                                uint32_t kernel_width, uint32_t input_per_group,
                                uint32_t stride_y, uint32_t stride_x,
                                uint32_t dilation_y, uint32_t dilation_x,
                                uint32_t padding_top, uint32_t padding_left,
                                uint32_t padding_bottom, uint32_t padding_right,
                                uint32_t groups, uint32_t relu,
                                float input_scale, int32_t input_zero_point,
                                float output_scale, int32_t output_zero_point,
                                uint32_t input_dtype, uint32_t weight_dtype,
                                uint32_t output_dtype,
                                size_t* input_bytes, size_t* weight_bytes,
                                size_t* output_bytes, size_t* metadata_bytes) {
    uint64_t input_elements = 1, weight_elements = 1, output_elements = 1;
    uint64_t terms = 1, padded_height, padded_width;
    uint64_t effective_height, effective_width;
    uint64_t expected_height, expected_width;
    uint64_t input_magnitude;
    if (!input || !weight || !weight_scales || !weight_zero_points || !output ||
        output == input || output == weight || !batch || !input_height ||
        !input_width || !input_channels || !output_height || !output_width ||
        !output_channels || !kernel_height || !kernel_width || !input_per_group ||
        !stride_y || !stride_x || !dilation_y || !dilation_x || !groups || relu > 2u ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        (weight_dtype != VX_DTYPE_I8 && weight_dtype != VX_DTYPE_U8) ||
        input_channels % groups || output_channels % groups ||
        (uint64_t)input_per_group * groups != input_channels) return 0;
    if (!qconv_mul_u64(input_elements, batch, &input_elements) ||
        !qconv_mul_u64(input_elements, input_height, &input_elements) ||
        !qconv_mul_u64(input_elements, input_width, &input_elements) ||
        !qconv_mul_u64(input_elements, input_channels, &input_elements) ||
        !qconv_mul_u64(weight_elements, output_channels, &weight_elements) ||
        !qconv_mul_u64(weight_elements, kernel_height, &weight_elements) ||
        !qconv_mul_u64(weight_elements, kernel_width, &weight_elements) ||
        !qconv_mul_u64(weight_elements, input_per_group, &weight_elements) ||
        !qconv_mul_u64(output_elements, batch, &output_elements) ||
        !qconv_mul_u64(output_elements, output_height, &output_elements) ||
        !qconv_mul_u64(output_elements, output_width, &output_elements) ||
        !qconv_mul_u64(output_elements, output_channels, &output_elements) ||
        !qconv_mul_u64(terms, kernel_height, &terms) ||
        !qconv_mul_u64(terms, kernel_width, &terms) ||
        !qconv_mul_u64(terms, input_per_group, &terms) ||
        input_elements > UINT32_MAX || weight_elements > UINT32_MAX ||
        output_elements > UINT32_MAX || input_elements > SIZE_MAX ||
        weight_elements > SIZE_MAX || output_elements > SIZE_MAX ||
        (size_t)output_channels > SIZE_MAX / sizeof(float) ||
        (size_t)output_channels > SIZE_MAX / sizeof(int32_t)) return 0;
    if ((uint64_t)(kernel_height - 1u) > (UINT64_MAX - 1u) / dilation_y ||
        (uint64_t)(kernel_width - 1u) > (UINT64_MAX - 1u) / dilation_x) return 0;
    padded_height = (uint64_t)input_height + padding_top + padding_bottom;
    padded_width = (uint64_t)input_width + padding_left + padding_right;
    effective_height = (uint64_t)(kernel_height - 1u) * dilation_y + 1u;
    effective_width = (uint64_t)(kernel_width - 1u) * dilation_x + 1u;
    if (padded_height < effective_height || padded_width < effective_width) return 0;
    expected_height = (padded_height - effective_height) / stride_y + 1u;
    expected_width = (padded_width - effective_width) / stride_x + 1u;
    if (expected_height != output_height || expected_width != output_width) return 0;
    {
        int64_t low = (int64_t)(input_dtype == VX_DTYPE_I8 ? -128 : 0) -
            input_zero_point;
        int64_t high = (int64_t)(input_dtype == VX_DTYPE_I8 ? 127 : 255) -
            input_zero_point;
        uint64_t low_magnitude = (uint64_t)(low < 0 ? -low : low);
        uint64_t high_magnitude = (uint64_t)(high < 0 ? -high : high);
        input_magnitude = low_magnitude > high_magnitude ? low_magnitude : high_magnitude;
    }
    for (uint32_t output_channel = 0; output_channel < output_channels;
         output_channel++) {
        const float weight_scale = qconv_f32_at(weight_scales, output_channel);
        const int32_t weight_zero_point = qconv_i32_at(weight_zero_points, output_channel);
        int64_t low, high;
        uint64_t weight_magnitude, accumulator_bound, bias_magnitude = 0;
        if (!isfinite(weight_scale) || weight_scale <= 0.0f ||
            !qbyte_dtype_zero_point_valid(weight_dtype, weight_zero_point)) return 0;
        low = (int64_t)(weight_dtype == VX_DTYPE_I8 ? -128 : 0) -
            weight_zero_point;
        high = (int64_t)(weight_dtype == VX_DTYPE_I8 ? 127 : 255) -
            weight_zero_point;
        weight_magnitude = (uint64_t)(low < 0 ? -low : low);
        {
            uint64_t high_magnitude = (uint64_t)(high < 0 ? -high : high);
            if (high_magnitude > weight_magnitude) weight_magnitude = high_magnitude;
        }
        if (input_magnitude && weight_magnitude &&
            terms > (uint64_t)INT32_MAX / input_magnitude / weight_magnitude) return 0;
        accumulator_bound = input_magnitude * weight_magnitude * terms;
        if (bias) {
            int64_t bias_value = qconv_i32_at(bias, output_channel);
            bias_magnitude = (uint64_t)(bias_value < 0 ? -bias_value : bias_value);
        }
        if (accumulator_bound > (uint64_t)INT32_MAX ||
            bias_magnitude > (uint64_t)INT32_MAX - accumulator_bound) return 0;
    }
    *input_bytes = (size_t)input_elements;
    *weight_bytes = (size_t)weight_elements;
    *output_bytes = (size_t)output_elements;
    *metadata_bytes = (size_t)output_channels * sizeof(int32_t);
    return 1;
}

int opengl_graph_qconv2d_i8u8(const void* input, const void* weight,
                               const float* weight_scales,
                               const int32_t* weight_zero_points,
                               const int32_t* bias, void* output,
                               uint32_t batch, uint32_t input_height,
                               uint32_t input_width, uint32_t input_channels,
                               uint32_t output_height, uint32_t output_width,
                               uint32_t output_channels, uint32_t kernel_height,
                               uint32_t kernel_width, uint32_t input_per_group,
                               uint32_t stride_y, uint32_t stride_x,
                               uint32_t dilation_y, uint32_t dilation_x,
                               uint32_t padding_top, uint32_t padding_left,
                               uint32_t padding_bottom, uint32_t padding_right,
                               uint32_t groups, uint32_t relu,
                               float input_scale, int32_t input_zero_point,
                               float output_scale, int32_t output_zero_point,
                               uint32_t input_dtype, uint32_t weight_dtype,
                               uint32_t output_dtype) {
    if (!ogl_ready() || max_ssbo_bindings < 6 || max_uniform_bindings < 1) return 0;
    size_t input_bytes, weight_bytes, output_bytes, metadata_bytes;
    if (!qconv_gpu_args_valid(input, weight, weight_scales, weight_zero_points,
                              bias, output, batch, input_height, input_width,
                              input_channels, output_height, output_width,
                              output_channels, kernel_height, kernel_width,
                              input_per_group, stride_y, stride_x, dilation_y,
                              dilation_x, padding_top, padding_left,
                              padding_bottom, padding_right, groups, relu,
                              input_scale, input_zero_point, output_scale,
                              output_zero_point, input_dtype, weight_dtype,
                              output_dtype, &input_bytes, &weight_bytes,
                              &output_bytes, &metadata_bytes)) return 0;
    const int32_t* bound_bias = bias ? bias : qconv_zero_bias_get(output_channels);
    if (!bound_bias) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    OglTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_bytes, 0);
    OglTensorSlot* weight_slot = graph_ensure_packed_bytes(weight, weight_bytes, 1);
    OglTensorSlot* scales_slot = graph_ensure_device(weight_scales,
                                                      (size_t)output_channels * sizeof(float), 1);
    OglTensorSlot* zero_points_slot = graph_ensure_device(weight_zero_points, metadata_bytes, 1);
    OglTensorSlot* bias_slot = graph_ensure_device(bound_bias, metadata_bytes, 1);
    OglTensorSlot* output_slot = graph_output_packed_bytes(output, output_bytes);
    if (!input_slot || !weight_slot || !scales_slot || !zero_points_slot ||
        !bias_slot || !output_slot) return 0;
    OglQConv2DParams params = {
        batch, input_height, input_width, input_channels,
        output_height, output_width, output_channels, kernel_height,
        kernel_width, stride_y, stride_x, dilation_y,
        dilation_x, padding_top, padding_left, groups,
        input_dtype,
        weight_dtype,
        output_dtype, relu,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, 0.0f, 0.0f
    };
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[7] = {
        input_slot->buffer, weight_slot->buffer, scales_slot->buffer,
        zero_points_slot->buffer, bias_slot->buffer, output_slot->buffer, pb
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    const uint64_t reduction_size = (uint64_t)kernel_height * kernel_width * input_per_group;
    int tiled = qconv_tiled_enabled && groups == 1u && output_channels >= 32u &&
                reduction_size >= 16u && (output_channels & 3u) == 0u &&
                opengl_workgroup_supported(8u, 4u, 1u);
    int ok = 0;
    if (tiled) {
        const uint32_t spatial = batch * output_height * output_width;
        const uint32_t words_per_spatial = output_channels / 4u;
        ok = dispatch_kernel(&k_qconv2d_int8_tiled, bufs,
                             (words_per_spatial + 7u) / 8u,
                             (spatial + 3u) / 4u, 1u);
    }
    if (!ok) {
        ok = dispatch_kernel(&k_qconv2d_int8, bufs,
                             (packed_words + 63u) / 64u, 1u, 1u);
    }
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

typedef struct { uint32_t elements, pad0, pad1, pad2; } OglTypedCopyParams;
typedef struct { uint32_t size, axis_offset, input_axis, output_axis, inner, pad0, pad1, pad2; } OglTypedConcatParams;
typedef struct { uint32_t n, h, w, c, out_h, out_w, ky, kx, sy, sx, py, px, dtype, pad0, pad1, pad2; } OglTypedMaxPoolParams;
typedef struct { uint32_t n, h, w, c, out_h, out_w, pad0, pad1; } OglTypedResizeParams;
typedef struct { uint32_t elements, output_type, zero_point_type, has_zero_point; } OglTypedQuantizeParams;
typedef struct { uint32_t size, input_type, scale_type, zero_point_type, output_type, has_zero_point, pad0, pad1; } OglTypedDequantizeParams;

_Static_assert(sizeof(OglTypedCopyParams) == 16, "copyTyped uniform ABI");
_Static_assert(sizeof(OglTypedConcatParams) == 32, "concatCopyTyped uniform ABI");
_Static_assert(sizeof(OglTypedMaxPoolParams) == 64, "maxPool2DTyped uniform ABI");
_Static_assert(sizeof(OglTypedResizeParams) == 32, "resizeNearestTyped uniform ABI");
_Static_assert(sizeof(OglTypedQuantizeParams) == 16, "quantizeLinearTyped uniform ABI");
_Static_assert(sizeof(OglTypedDequantizeParams) == 32, "dequantizeLinearTyped uniform ABI");

static int typed_shape_qdesc_valid(float scale, int32_t zero_point, uint32_t dtype) {
    return isfinite(scale) && scale > 0.0f && qbyte_dtype_zero_point_valid(dtype, zero_point);
}

static int typed_shape_qdesc_same(float input_scale, int32_t input_zero_point,
                                  uint32_t input_dtype, float output_scale,
                                  int32_t output_zero_point, uint32_t output_dtype) {
    return typed_shape_qdesc_valid(input_scale, input_zero_point, input_dtype) &&
        typed_shape_qdesc_valid(output_scale, output_zero_point, output_dtype) &&
        input_dtype == output_dtype && input_zero_point == output_zero_point && input_scale == output_scale;
}

static int typed_shape_nhwc_elements(uint32_t n, uint32_t h, uint32_t w, uint32_t c,
                                     uint32_t* elements) {
    uint64_t value = n;
    if (!n || !h || !w || !c || !elements || !qconv_mul_u64(value, h, &value) ||
        !qconv_mul_u64(value, w, &value) || !qconv_mul_u64(value, c, &value) ||
        value > UINT32_MAX) return 0;
    *elements = (uint32_t)value;
    return 1;
}

static uint32_t typed_shape_groups(uint32_t elements) { return elements / 64u + (elements % 64u != 0u); }
static uint32_t typed_shape_packed_groups(size_t bytes) { return typed_shape_groups((uint32_t)(bytes / sizeof(uint32_t))); }
static uint32_t typed_shape_zero_word(int32_t zero_point, uint32_t dtype) {
    return dtype == VX_DTYPE_I8 ? (uint32_t)(uint8_t)(int8_t)zero_point :
        (uint32_t)(uint8_t)zero_point;
}

static int graph_zero_packed_output(OglTensorSlot* slot, size_t bytes) {
    unsigned char* zeros = (unsigned char*)calloc(bytes, 1);
    if (!slot || !zeros) return 0;
    int ok = slot_ensure_owned_buffer(slot, bytes, zeros);
    free(zeros);
    if (!ok) return 0;
    slot->host_dirty = 0;
    slot->device_dirty = 0;
    return 1;
}

int opengl_graph_quantize_typed_f32_i8u8(const float* input, uint32_t elements,
                                         void* output, float output_scale,
                                         int32_t output_zero_point, uint32_t output_dtype) {
    if (!ogl_ready() || max_ssbo_bindings < 4 || max_uniform_bindings < 1 || !input || !output ||
        input == output || !elements || !typed_shape_qdesc_valid(output_scale, output_zero_point, output_dtype) ||
        (uint64_t)elements > SIZE_MAX / sizeof(float)) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(elements, &packed_output_bytes)) return 0;
    OglTensorSlot* input_slot = graph_ensure_device(input, (size_t)elements * sizeof(float), 0);
    OglTensorSlot* output_slot = graph_output_packed_bytes(output, elements);
    if (!input_slot || !output_slot) return 0;
    uint32_t zero_word = typed_shape_zero_word(output_zero_point, output_dtype);
    OglTypedQuantizeParams params = {
        elements, output_dtype, output_dtype, 1u
    };
    GLuint scale_buffer = create_buffer(sizeof(output_scale), &output_scale);
    GLuint zero_buffer = create_buffer(sizeof(zero_word), &zero_word);
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[5] = {input_slot->buffer, scale_buffer, zero_buffer, output_slot->buffer, pb};
    int ok = dispatch_kernel(&k_quantize_typed_i8u8, bufs, typed_shape_packed_groups(packed_output_bytes), 1u, 1u);
    if (scale_buffer) p_glDeleteBuffers(1, &scale_buffer);
    if (zero_buffer) p_glDeleteBuffers(1, &zero_buffer);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_dequantize_typed_i8u8_f32(const void* input, uint32_t elements,
                                           float input_scale, int32_t input_zero_point,
                                           uint32_t input_dtype, float* output) {
    if (!ogl_ready() || max_ssbo_bindings < 4 || max_uniform_bindings < 1 || !input || !output ||
        input == output || !elements || !typed_shape_qdesc_valid(input_scale, input_zero_point, input_dtype) ||
        (uint64_t)elements > SIZE_MAX / sizeof(float)) return 0;
    size_t packed_input_bytes;
    if (!graph_packed_bytes(elements, &packed_input_bytes)) return 0;
    OglTensorSlot* input_slot = graph_ensure_packed_bytes(input, elements, 0);
    OglTensorSlot* output_slot = graph_output_slot(output, (size_t)elements * sizeof(float));
    if (!input_slot || !output_slot) return 0;
    uint32_t zero_word = typed_shape_zero_word(input_zero_point, input_dtype);
    OglTypedDequantizeParams params = {
        elements, input_dtype, VX_DTYPE_F32, input_dtype,
        VX_DTYPE_F32, 1u, 0u, 0u
    };
    GLuint scale_buffer = create_buffer(sizeof(input_scale), &input_scale);
    GLuint zero_buffer = create_buffer(sizeof(zero_word), &zero_word);
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[5] = {input_slot->buffer, scale_buffer, zero_buffer, output_slot->buffer, pb};
    int ok = dispatch_kernel(&k_dequantize_typed_i8u8, bufs, typed_shape_groups(elements), 1u, 1u);
    if (scale_buffer) p_glDeleteBuffers(1, &scale_buffer);
    if (zero_buffer) p_glDeleteBuffers(1, &zero_buffer);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_copy_i8u8(const void* input, uint32_t input_elements,
                           void* output, uint32_t output_elements,
                           float input_scale, int32_t input_zero_point,
                           float output_scale, int32_t output_zero_point,
                           uint32_t input_dtype, uint32_t output_dtype) {
    if (!ogl_ready() || max_ssbo_bindings < 2 || max_uniform_bindings < 1 || !input || !output ||
        input == output || !input_elements || input_elements != output_elements ||
        !typed_shape_qdesc_same(input_scale, input_zero_point, input_dtype, output_scale, output_zero_point, output_dtype)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(input_elements, &packed_bytes)) return 0;
    OglTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_elements, 0);
    OglTensorSlot* output_slot = graph_output_packed_bytes(output, output_elements);
    if (!input_slot || !output_slot) return 0;
    OglTypedCopyParams params = {input_elements, 0u, 0u, 0u};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {input_slot->buffer, output_slot->buffer, pb};
    int ok = dispatch_kernel(&k_copy_typed_i8u8, bufs, typed_shape_packed_groups(packed_bytes), 1u, 1u);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_transpose_i8u8(
        const void* input, void* output, const uint32_t* input_shape,
        const uint32_t* permutation, uint32_t rank, uint32_t elements,
        float input_scale, int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, uint32_t input_dtype,
        uint32_t output_dtype) {
    uint32_t input_strides[8] = {0};
    uint32_t output_shape[8] = {0};
    uint32_t output_strides[8] = {0};
    uint32_t metadata[18] = {0};
    uint32_t seen = 0;
    uint64_t product = 1;
    uint64_t stride = 1;
    size_t packed_bytes;
    if (!ogl_ready() || max_ssbo_bindings < 3 || !input || !output ||
        !input_shape || !permutation || rank == 0 || rank > 8 ||
        elements == 0 ||
        !typed_shape_qdesc_same(input_scale, input_zero_point, input_dtype,
                                output_scale, output_zero_point, output_dtype) ||
        qsdpa_ranges_overlap(output, elements, input, elements)) return 0;
    for (uint32_t reverse = rank; reverse-- > 0;) {
        if (!input_shape[reverse] || stride > UINT32_MAX) return 0;
        input_strides[reverse] = (uint32_t)stride;
        stride *= input_shape[reverse];
        if (stride > UINT32_MAX) return 0;
    }
    if (stride != elements) return 0;
    for (uint32_t dimension = 0; dimension < rank; dimension++) {
        uint32_t source = permutation[dimension];
        if (source >= rank || (seen & (1u << source))) return 0;
        seen |= 1u << source;
        output_shape[dimension] = input_shape[source];
        if (product > UINT32_MAX / output_shape[dimension]) return 0;
        product *= output_shape[dimension];
    }
    if (product != elements) return 0;
    stride = 1;
    for (uint32_t reverse = rank; reverse-- > 0;) {
        output_strides[reverse] = (uint32_t)stride;
        stride *= output_shape[reverse];
    }
    if (!graph_packed_bytes(elements, &packed_bytes)) return 0;
    OglTensorSlot* input_slot =
        graph_ensure_packed_bytes(input, elements, 0);
    OglTensorSlot* output_slot =
        graph_output_packed_bytes(output, elements);
    if (!input_slot || !output_slot) return 0;
    metadata[0] = elements;
    metadata[1] = rank;
    for (uint32_t dimension = 0; dimension < rank; dimension++) {
        metadata[2 + dimension] = output_strides[dimension];
        metadata[2 + rank + dimension] =
            input_strides[permutation[dimension]];
    }
    size_t metadata_bytes = (size_t)(2 + 2 * rank) * sizeof(uint32_t);
    GLuint metadata_buffer = create_buffer(metadata_bytes, metadata);
    if (!metadata_buffer) return 0;
    GLuint buffers[3] = {
        input_slot->buffer, output_slot->buffer, metadata_buffer,
    };
    int ok = dispatch_kernel(
        &k_transpose_typed_i8u8, buffers,
        typed_shape_packed_groups(packed_bytes), 1u, 1u);
    p_glDeleteBuffers(1, &metadata_buffer);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_concat_i8u8(const void* const* inputs, const uint32_t* input_elements,
                             const uint32_t* input_axes, const float* input_scales,
                             const int32_t* input_zero_points, const uint32_t* input_dtypes,
                             uint32_t input_count, void* output, uint32_t output_elements,
                             uint32_t output_axis, uint32_t inner,
                             float output_scale, int32_t output_zero_point,
                             uint32_t output_dtype) {
    if (!ogl_ready() || max_ssbo_bindings < 2 || max_uniform_bindings < 1 || !inputs || !input_elements ||
        !input_axes || !input_scales || !input_zero_points || !input_dtypes || !output || !input_count ||
        !output_elements || !output_axis || !inner || !typed_shape_qdesc_valid(output_scale, output_zero_point, output_dtype)) return 0;
    uint64_t output_width = (uint64_t)output_axis * inner;
    if (!output_width || output_width > output_elements || output_elements % output_width) return 0;
    uint64_t outer = output_elements / output_width, axis_sum = 0;
    for (uint32_t index = 0; index < input_count; index++) {
        uint64_t expected = 0;
        if (!qconv_mul_u64(outer, input_axes[index], &expected) ||
            !qconv_mul_u64(expected, inner, &expected)) {
            return 0;
        }
        if (!inputs[index] || inputs[index] == output || !input_axes[index] || expected != input_elements[index] ||
            !typed_shape_qdesc_same(input_scales[index], input_zero_points[index], input_dtypes[index], output_scale, output_zero_point, output_dtype) ||
            axis_sum > UINT32_MAX - input_axes[index]) return 0;
        axis_sum += input_axes[index];
    }
    if (axis_sum != output_axis) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_elements, &packed_output_bytes)) return 0;
    OglTensorSlot* output_slot = graph_output_packed_bytes(output, output_elements);
    if (!output_slot || !graph_zero_packed_output(output_slot, packed_output_bytes)) return 0;
    uint32_t axis_offset = 0;
    for (uint32_t index = 0; index < input_count; index++) {
        OglTensorSlot* input_slot = graph_ensure_packed_bytes(inputs[index], input_elements[index], 0);
        if (!input_slot) return 0;
        OglTypedConcatParams params = {input_elements[index], axis_offset, input_axes[index], output_axis, inner, 0u, 0u, 0u};
        GLuint pb = params_buffer(&params, sizeof(params));
        GLuint bufs[3] = {input_slot->buffer, output_slot->buffer, pb};
        int ok = dispatch_kernel(&k_concat_typed_i8u8, bufs, typed_shape_groups(input_elements[index]), 1u, 1u);
        if (pb) p_glDeleteBuffers(1, &pb);
        if (!ok) return 0;
        axis_offset += input_axes[index];
    }
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_maxpool2d_i8u8(const void* input, void* output,
                                uint32_t batch, uint32_t input_height, uint32_t input_width, uint32_t channels,
                                uint32_t output_height, uint32_t output_width, uint32_t kernel_y, uint32_t kernel_x,
                                uint32_t stride_y, uint32_t stride_x, uint32_t padding_top, uint32_t padding_left,
                                uint32_t padding_bottom, uint32_t padding_right, float input_scale,
                                int32_t input_zero_point, float output_scale, int32_t output_zero_point,
                                uint32_t input_dtype, uint32_t output_dtype) {
    uint32_t input_elements, output_elements;
    uint64_t padded_height, padded_width, expected_height, expected_width;
    if (!ogl_ready() || max_ssbo_bindings < 2 || max_uniform_bindings < 1 || !input || !output || input == output ||
        !kernel_y || !kernel_x || !stride_y || !stride_x || !typed_shape_qdesc_same(input_scale, input_zero_point, input_dtype, output_scale, output_zero_point, output_dtype) ||
        !typed_shape_nhwc_elements(batch, input_height, input_width, channels, &input_elements) ||
        !typed_shape_nhwc_elements(batch, output_height, output_width, channels, &output_elements)) return 0;
    padded_height = (uint64_t)input_height + padding_top + padding_bottom;
    padded_width = (uint64_t)input_width + padding_left + padding_right;
    if (padded_height < kernel_y || padded_width < kernel_x) return 0;
    expected_height = (padded_height - kernel_y) / stride_y + 1u;
    expected_width = (padded_width - kernel_x) / stride_x + 1u;
    if (expected_height != output_height || expected_width != output_width) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_elements, &packed_output_bytes)) return 0;
    OglTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_elements, 0);
    OglTensorSlot* output_slot = graph_output_packed_bytes(output, output_elements);
    if (!input_slot || !output_slot) return 0;
    OglTypedMaxPoolParams params = {
        batch, input_height, input_width, channels, output_height, output_width,
        kernel_y, kernel_x, stride_y, stride_x, padding_top, padding_left,
        input_dtype, 0u, 0u, 0u
    };
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {input_slot->buffer, output_slot->buffer, pb};
    int ok = dispatch_kernel(&k_maxpool_typed_i8u8, bufs, typed_shape_packed_groups(packed_output_bytes), 1u, 1u);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_resize_nearest_i8u8(const void* input, void* output,
                                     uint32_t batch, uint32_t input_height, uint32_t input_width, uint32_t channels,
                                     uint32_t output_height, uint32_t output_width, float input_scale,
                                     int32_t input_zero_point, float output_scale, int32_t output_zero_point,
                                     uint32_t input_dtype, uint32_t output_dtype) {
    uint32_t input_elements, output_elements;
    if (!ogl_ready() || max_ssbo_bindings < 2 || max_uniform_bindings < 1 || !input || !output || input == output ||
        !typed_shape_qdesc_same(input_scale, input_zero_point, input_dtype, output_scale, output_zero_point, output_dtype) ||
        !typed_shape_nhwc_elements(batch, input_height, input_width, channels, &input_elements) ||
        !typed_shape_nhwc_elements(batch, output_height, output_width, channels, &output_elements)) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_elements, &packed_output_bytes)) return 0;
    OglTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_elements, 0);
    OglTensorSlot* output_slot = graph_output_packed_bytes(output, output_elements);
    if (!input_slot || !output_slot) return 0;
    OglTypedResizeParams params = {batch, input_height, input_width, channels, output_height, output_width, 0u, 0u};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {input_slot->buffer, output_slot->buffer, pb};
    int ok = dispatch_kernel(&k_resize_nearest_typed_i8u8, bufs, typed_shape_packed_groups(packed_output_bytes), 1u, 1u);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
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

static int opengl_graph_profile_common(const OglKernel* kernel, const float* in,
                                       float* out,
                                       int n, int h, int w, int c, long out_elems_per_batch,
                                       uint32_t gx, uint32_t gy) {
    if (!kernel || !in || !out || n <= 0 || h <= 0 || w <= 0 || c <= 0 || out_elems_per_batch <= 0) return 0;
    size_t in_bytes = (size_t)n * (size_t)h * (size_t)w * (size_t)c * sizeof(float);
    size_t out_bytes = (size_t)n * (size_t)out_elems_per_batch * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)h, (uint32_t)w, (uint32_t)c, (uint32_t)n};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(kernel, bufs, gx, gy, (uint32_t)n);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

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
