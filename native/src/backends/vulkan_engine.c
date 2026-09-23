#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <dlfcn.h>
#include <pthread.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "shader_store.h"
#include "vulkan_engine.h"
#include "runtime_state.h"
#include "profiling.h"

typedef struct VkTracePass VkTracePass;
#include "batch_matmul_f32_plan.h"
#include "expand_f32_plan.h"
#include "qbatch_matmul_plan.h"
#include "qlinear_multiplier.h"
#include "typed_control_plan.h"

void vk_set_shader_root(const char* root) {
    if (volvoxai_shader_store_set_override_root(root) != VOLVOXAI_SHADER_STORE_OK) {
        fprintf(stderr, "[Vulkan] invalid configured shader root\n");
    }
}

#define VK_FUNC(name) PFN_##name p_##name;
typedef struct {
    void* loader;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr;
VK_FUNC(vkCreateInstance)
#if defined(VK_VERSION_1_1)
VK_FUNC(vkEnumerateInstanceVersion)
#endif
VK_FUNC(vkDestroyInstance)
VK_FUNC(vkEnumeratePhysicalDevices)
VK_FUNC(vkGetPhysicalDeviceProperties)
#if defined(VK_VERSION_1_1)
VK_FUNC(vkGetPhysicalDeviceFeatures2)
VK_FUNC(vkGetPhysicalDeviceProperties2)
#endif
VK_FUNC(vkGetPhysicalDeviceQueueFamilyProperties)
VK_FUNC(vkGetPhysicalDeviceMemoryProperties)
VK_FUNC(vkCreateDevice)
VK_FUNC(vkDestroyDevice)
VK_FUNC(vkGetDeviceQueue)
VK_FUNC(vkCreateCommandPool)
VK_FUNC(vkDestroyCommandPool)
VK_FUNC(vkAllocateCommandBuffers)
VK_FUNC(vkCreateBuffer)
VK_FUNC(vkDestroyBuffer)
VK_FUNC(vkGetBufferMemoryRequirements)
VK_FUNC(vkAllocateMemory)
VK_FUNC(vkFreeMemory)
VK_FUNC(vkBindBufferMemory)
VK_FUNC(vkMapMemory)
VK_FUNC(vkUnmapMemory)
VK_FUNC(vkFlushMappedMemoryRanges)
VK_FUNC(vkInvalidateMappedMemoryRanges)
VK_FUNC(vkCreateShaderModule)
VK_FUNC(vkDestroyShaderModule)
VK_FUNC(vkCreateDescriptorSetLayout)
VK_FUNC(vkDestroyDescriptorSetLayout)
VK_FUNC(vkCreatePipelineLayout)
VK_FUNC(vkDestroyPipelineLayout)
VK_FUNC(vkCreatePipelineCache)
VK_FUNC(vkDestroyPipelineCache)
VK_FUNC(vkCreateComputePipelines)
VK_FUNC(vkDestroyPipeline)
VK_FUNC(vkCreateDescriptorPool)
VK_FUNC(vkDestroyDescriptorPool)
#if VOLVOXAI_ENABLE_TRAINING
VK_FUNC(vkResetDescriptorPool)
#endif
VK_FUNC(vkAllocateDescriptorSets)
VK_FUNC(vkUpdateDescriptorSets)
VK_FUNC(vkBeginCommandBuffer)
VK_FUNC(vkCmdBindPipeline)
VK_FUNC(vkCmdBindDescriptorSets)
VK_FUNC(vkCmdPushConstants)
VK_FUNC(vkCmdDispatch)
VK_FUNC(vkCmdCopyBuffer)
VK_FUNC(vkCmdPipelineBarrier)
VK_FUNC(vkEndCommandBuffer)
VK_FUNC(vkQueueSubmit)
VK_FUNC(vkQueueWaitIdle)
VK_FUNC(vkDeviceWaitIdle)
VK_FUNC(vkCreateFence)
VK_FUNC(vkResetFences)
VK_FUNC(vkWaitForFences)
VK_FUNC(vkDestroyFence)
} VulkanApi;
#undef VK_FUNC

#define LOAD_GLOBAL(name) name = (PFN_##name)vkGetInstanceProcAddr(NULL, #name);
#define LOAD_INST(name) name = (PFN_##name)vkGetInstanceProcAddr(instance, #name);

#define ALIGN_UP(x, a) (((x) + ((a) - 1)) & ~((a) - 1))

typedef struct {
    const char* name;
    const char* path;
    int binding_count;
    int uniform_binding;
    VkDescriptorSetLayout desc_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    VkDescriptorSet desc_set;
    int ready;
} VkKernelDefinition;

/* Kernel declarations below are immutable shader/pipeline specifications.
 * Prepared Vulkan handles live in the current engine's Vulkan context. */
#define VkKernel const VkKernelDefinition

static VkKernel k_conv2d    = {"conv2D",      "spv/conv2D.spv",      5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_c3out16 = {"conv2DRegularC3Out16", "spv/conv2DRegularC3Out16.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_out16 = {"conv2DRegularOut16", "spv/conv2DRegularOut16.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_dw4 = {"conv2DDepthwise4", "spv/conv2DDepthwise4.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_dw8 = {"conv2DDepthwise8", "spv/conv2DDepthwise8.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_pw8 = {"conv2DPointwise8", "spv/conv2DPointwise8.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_pw8v2 = {"conv2DPointwise8Vec2", "spv/conv2DPointwise8Vec2.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_pw8v4 = {"conv2DPointwise8Vec4", "spv/conv2DPointwise8Vec4.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_pw16 = {"conv2DPointwise16", "spv/conv2DPointwise16.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_pw16tile = {"conv2DPointwise16Tile", "spv/conv2DPointwise16Tile.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_sigmoid   = {"sigmoid",     "spv/sigmoid.spv",     3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_clip      = {"clip",        "spv/clip.spv",        3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_copy      = {"copy",        "spv/copy.spv",        3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_relu      = {"reLU",        "spv/reLU.spv",        3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_gelu      = {"gELU",        "spv/gELU.spv",        3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_silu      = {"siLU",        "spv/siLU.spv",        3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_tanh      = {"tanh",        "spv/tanh.spv",        3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_hardswish = {"hardSwish",   "spv/hardSwish.spv",   3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_hardsigmoid = {"hardSigmoid", "spv/hardSigmoid.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_leaky_relu = {"leakyReLU",  "spv/leakyReLU.spv",   3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_prelu     = {"pReLU",       "spv/pReLU.spv",       4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_layernorm = {"layerNorm",   "spv/layerNorm.spv",   5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_rmsnorm   = {"rMSNorm",     "spv/rMSNorm.spv",     4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_softmax   = {"softmax",     "spv/softmax.spv",     3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_logsoftmax = {"logSoftmax", "spv/logSoftmax.spv",  3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_reduce    = {"reduce",      "spv/reduce.spv",      3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_globalavg = {"globalAveragePool", "spv/globalAveragePool.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_avgpool   = {"averagePool2D", "spv/averagePool2D.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_batchnorm = {"batchNorm2D", "spv/batchNorm2D.spv", 7, 6, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_groupnorm = {"groupNorm", "spv/groupNorm.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
#if VOLVOXAI_ENABLE_TRAINING
static VkKernel k_dropout = {"dropout", "spv/dropout.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
#endif
static VkKernel k_embedding = {"embedding",   "spv/embedding.spv",   4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
/* Bindings 0-5 storage, 6 uniform params, 7 the resident-slot table. */
static VkKernel k_moe_linear = {"moeLinear", "spv/moeLinear.spv", 8, 6, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
/* Bindings 0-4 storage, 5 uniform params. */
static VkKernel k_moe_router = {"moeRouter", "spv/moeRouter.spv", 6, 5, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_transpose = {"generalTranspose", "spv/generalTranspose.spv", 3, -1, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_where     = {"where",       "spv/where.spv",       5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_typed_control_32 = {"typedControl32Native", "spv/typedControl32Native.spv", 4, -1, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_where_32 = {"where32Native", "spv/where32Native.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_argmax_f32_i32 = {"argMaxF32I32Native", "spv/argMaxF32I32Native.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_concat_32 = {"concatCopy32Native", "spv/concatCopy32Native.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_expand    = {"expand",      "spv/expand.spv",      3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_batch_matmul = {"batchMatMul", "spv/batchMatMul.spv", 4, -1, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_linear_in_out = {"linearF32RowMajor", "spv/linearF32RowMajor.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_linear_in_out_tiled = {"linearF32RowMajorTiled", "spv/linearF32RowMajorTiled.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_linear_out_in = {"linearF32", "spv/linearF32.spv", 6, 5, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_linear_out_in_tiled = {"linearF32Tiled", "spv/linearF32Tiled.spv", 6, 5, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_pad       = {"pad",         "spv/pad.spv",         3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_slice     = {"slice",       "spv/slice.spv",       3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_gather    = {"gather",      "spv/gather.spv",      4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_convtranspose = {"convTranspose2D", "spv/convTranspose2D.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_interp1d  = {"interp1D",    "spv/interp1D.spv",    3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_mul       = {"mul",         "spv/mul.spv",         4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_sub       = {"sub",         "spv/sub.spv",         4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_div       = {"div",         "spv/div.spv",         4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_broadcast_binary = {"broadcastBinaryNative", "spv/broadcastBinaryNative.spv", 4, -1, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_split     = {"split",       "spv/split.spv",       3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv1d    = {"conv1D",      "spv/conv1D.spv",      5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_sdpa      = {"sDPA",        "spv/sDPA.spv",        4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_cross_sdpa = {"crossSDPA",  "spv/crossSDPA.spv",   6, 5, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
#if VOLVOXAI_ENABLE_TRAINING
static VkKernel k_sdpa_training = {"sdpaTraining", "spv/sdpaTraining.spv", 4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_cross_sdpa_training = {"crossSdpaTraining", "spv/crossSdpaTraining.spv", 6, 5, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
#endif
static VkKernel k_cross_attention = {"crossAttentionF32", "spv/crossAttentionF32.spv", 7, 6, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_quantize  = {"quantizeLinear", "spv/quantizeLinear.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_dequantize = {"dequantizeLinear", "spv/dequantizeLinear.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qlinear_int8 = {"qLinearInt8", "spv/qLinearInt8.spv", 7, 6, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qlinear_int8_tiled = {"qLinearInt8Tiled", "spv/qLinearInt8Tiled.spv", 7, 6, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qlinear_int8_dot = {"qLinearInt8Dot", "spv/qLinearInt8Dot.spv", 7, 6, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qlinear_int8_dot_tiled = {"qLinearInt8DotTiled", "spv/qLinearInt8DotTiled.spv", 7, 6, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qembedding_int8 = {"qEmbeddingInt8", "spv/qEmbeddingInt8.spv", 6, 5, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qconv2d_int8 = {"qConv2DInt8", "spv/qConv2DInt8.spv", 7, 6, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qconv2d_int8_tiled = {"qConv2DInt8Tiled", "spv/qConv2DInt8Tiled.spv", 7, 6, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qconv2d_int8_dot_tiled = {"qConv2DInt8DotTiled", "spv/qConv2DInt8DotTiled.spv", 7, 6, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_quantize_typed_i8u8 = {"quantizeLinearTyped", "spv/quantizeLinearTyped.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_dequantize_typed_i8u8 = {"dequantizeLinearTyped", "spv/dequantizeLinearTyped.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qadd_i8u8 = {"qAdd", "spv/qAdd.spv", 4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qbatch_matmul_i8u8 = {"qBatchMatMul", "spv/qBatchMatMul.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qbatch_matmul_i8u8_dot = {"qBatchMatMulDot", "spv/qBatchMatMulDot.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qsilu_i8u8 = {"qSiLUInt8", "spv/qSiLUInt8.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qgelu_i8u8 = {"qGELUInt8", "spv/qGELUInt8.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qgroupnorm_stats = {"qGroupNormStats", "spv/qGroupNormStats.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qgroupnorm_apply = {"qGroupNormApply", "spv/qGroupNormApply.spv", 6, 5, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qlayernorm_stats = {"qLayerNormStats", "spv/qLayerNormStats.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qlayernorm_apply = {"qLayerNormApply", "spv/qLayerNormApply.spv", 6, 5, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qsdpa_int8 = {"qSDPAInt8", "spv/qSDPAInt8.spv", 6, 5, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qargmax_int8 = {"qArgMaxInt8", "spv/qArgMaxInt8.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_row_index_transfer = {"rowIndexTransfer", "spv/rowIndexTransfer.spv", 4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_qmaskedmean_int8 = {"qMaskedMeanInt8", "spv/qMaskedMeanInt8.spv", 4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_requantize_linear_i8u8 = {"requantizeLinearTyped", "spv/requantizeLinearTyped.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_copy_typed_i8u8 = {"copyTyped", "spv/copyTyped.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_concat_typed_i8u8 = {"concatCopyTyped", "spv/concatCopyTyped.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_maxpool_typed_i8u8 = {"maxPool2DTyped", "spv/maxPool2DTyped.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_resize_nearest_typed_i8u8 = {"resizeNearestTyped", "spv/resizeNearestTyped.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_transpose_typed_i8u8 = {"transposeTyped", "spv/transposeTyped.spv", 3, -1, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_spatial_softargmax_y = {"spatialSoftargmaxY", "spv/spatialSoftargmaxY.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_profile_x = {"profileX",    "spv/profileX.spv",    3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_profile_y = {"profileY",    "spv/profileY.spv",    3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_mean_height = {"meanHeight", "spv/meanHeight.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_nms       = {"nonMaxSuppression", "spv/nonMaxSuppression.spv", 4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_add       = {"add",         "spv/add.spv",         4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_add_relu  = {"addRelu",     "spv/addRelu.spv",     4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_upsample  = {"upsample2x",  "spv/upsample2x.spv",  3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_concat    = {"concatCopy",  "spv/concatCopy.spv",  3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_concat_sigmoid = {"concatSigmoidCopy", "spv/concatSigmoidCopy.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_maxpool   = {"maxPool2D",   "spv/maxPool2D.spv",   3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_resize    = {"resize",      "spv/resize.spv",      3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};

#define VK_GRAPH_MAX_TENSORS 8192
#define VK_GRAPH_MAX_FREE_RANGES (VK_GRAPH_MAX_TENSORS + 1)
#define VK_GRAPH_MAX_DISPATCH_SETS 4096
#define VK_GRAPH_SCRATCH_BYTES ((size_t)1024 * 1024)
#define VK_GRAPH_STAGING_BYTES ((size_t)32 * 1024 * 1024)
#define VK_GRAPH_BASE ((size_t)128 * 1024 * 1024)
#define VK_GRAPH_MIB ((size_t)1024 * 1024)
#define VK_GRAPH_ALIGN 256
#define VK_PREPARED_KERNEL_MAX 160
#define WT_CACHE_MAX 512
#define WEIGHTS_LIMIT ((size_t)64 * 1024 * 1024)

static int vk_graph_align_up(size_t value, size_t alignment, size_t* out) {
    size_t remainder;
    if (!out || !alignment) return 0;
    remainder = value % alignment;
    if (remainder && value > SIZE_MAX - (alignment - remainder)) return 0;
    *out = remainder ? value + (alignment - remainder) : value;
    return 1;
}

static int vk_graph_arena_size_valid(size_t bytes, size_t alignment) {
    size_t arena_start;
    size_t scratch_start;
    size_t arena_limit;
    if (!alignment || bytes < VK_GRAPH_SCRATCH_BYTES ||
        !vk_graph_align_up(VK_GRAPH_BASE, alignment, &arena_start) ||
        !vk_graph_align_up(bytes - VK_GRAPH_SCRATCH_BYTES,
                           alignment, &scratch_start) ||
        scratch_start > bytes)
        return 0;
    arena_limit = bytes - VK_GRAPH_SCRATCH_BYTES;
    arena_limit -= arena_limit % alignment;
    return arena_start < arena_limit;
}

static int vk_graph_parse_arena_mebibytes(const char* text,
                                           size_t alignment,
                                           size_t* bytes) {
    const unsigned char* cursor = (const unsigned char*)text;
    const size_t maximum = SIZE_MAX / VK_GRAPH_MIB;
    size_t value = 0u;
    if (!cursor || !cursor[0] || !bytes) return 0;
    for (; *cursor; cursor++) {
        size_t digit;
        if (*cursor < (unsigned char)'0' ||
            *cursor > (unsigned char)'9')
            return 0;
        digit = (size_t)(*cursor - (unsigned char)'0');
        if (value > maximum / 10u ||
            (value == maximum / 10u && digit > maximum % 10u))
            return 0;
        value = value * 10u + digit;
    }
    value *= VK_GRAPH_MIB;
    if (!vk_graph_arena_size_valid(value, alignment)) return 0;
    *bytes = value;
    return 1;
}

typedef struct {
    const void* host;
    size_t bytes;
    size_t capacity;
    size_t domain_capacity;
    size_t offset;
    uint64_t shape_generation;
    uint64_t capacity_generation;
    int host_dirty;
    int device_dirty;
    int is_weight;
    int owns_range;
    int domain_span;
} VkTensorSlot;

typedef struct {
    size_t offset;
    size_t bytes;
} VkGraphFreeRange;

/* A NULL canonical QConv2D bias still needs an output-channel-sized storage
 * binding. Keep every allocation alive until backend cleanup so graph slots
 * never retain a dangling host key while a forward chain is resident. */
typedef struct QConvZeroBiasBacking {
    int32_t* values;
    size_t elements;
    struct QConvZeroBiasBacking* next;
} QConvZeroBiasBacking;

static const int32_t* qconv_zero_bias_get(uint32_t output_channels);

/* qSDPAInt8 always declares mask binding 3.  When no logical mask exists,
 * bind durable conventional I32 storage rather than retyping an activation
 * buffer as a mask descriptor. */
static const int32_t qsdpa_dummy_mask[1] = {0};
static const float linear_dummy_scale[1] = {1.0f};

typedef struct {
    VkKernel* kernel;
    VkDescriptorSet descriptor_set;
} VkGraphDispatchSet;

typedef struct {
    VkKernel* definition;
    VkDescriptorSetLayout prepared_desc_layout;
    VkPipelineLayout prepared_pipeline_layout;
    VkPipeline pipeline;
    int ready;
} VkPreparedKernel;

typedef struct {
    const float* src;
    int d_in;
    int d_out;
    size_t bytes;
    size_t off;
} VkWeightCacheEntry;

#if VOLVOXAI_ENABLE_TRAINING
typedef struct VkTrainingKernelSlot VkTrainingKernelSlot;
#endif

/* Process-shared state is limited to the physical device, queue, immutable
 * capability limits, and synchronized model-independent pipeline caches. */
typedef struct {
    pthread_mutex_t mutex;
    VulkanApi api;
    VkInstance instance_handle;
    VkPhysicalDevice physical_device_handle;
    VkDevice device_handle;
    VkQueue compute_queue_handle;
    uint32_t queue_family_index;
    VkPhysicalDeviceMemoryProperties memory_properties;
    uint32_t max_workgroups[3];
    uint32_t max_workgroup_size[3];
    uint32_t max_workgroup_invocations;
    uint32_t max_storage_bindings;
    uint32_t max_uniform_bindings;
    VkDeviceSize max_storage_range;
    VkDeviceSize max_uniform_range;
    VkDeviceSize non_coherent_atom_size;
    size_t graph_alignment;
    int packed_dot;
    int packed_dot_warned;
#if VOLVOXAI_ENABLE_TRAINING
    uint32_t training_max_storage_bindings;
    uint32_t training_max_uniform_bindings;
    uint32_t training_max_workgroup_size_x;
    uint32_t training_max_workgroup_invocations;
    VkDeviceSize training_max_storage_range;
    VkDeviceSize training_max_uniform_range;
#endif
    VkDescriptorSetLayout matmul_desc_layout;
    VkPipelineLayout matmul_pipeline_layout;
    VkPipelineCache pipeline_cache;
    VkPipeline matmul_pipeline;
    VkPipeline matmul_tiled_pipeline;
    uint64_t prepared_pipeline_creates;
    unsigned context_count;
    int initialized;
    /* The device is shared across engine states and released at process exit,
     * so the teardown hook is registered exactly once. */
    int exit_hook_registered;
    uint32_t timestamp_bits;
    float timestamp_period;
    atomic_uint_fast64_t trace_device_id, trace_queue_id;
    int calibrated_timestamps;
} VulkanDeviceState;

/* Every field below is owned by exactly one VxEngineState. It may be used by
 * only that engine's FIFO execution context at a time. */
typedef struct {
    int device_acquired;
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t arena_size;
    size_t arena_allocation_size;
    uint32_t compute_memory_type_index;
    VkMemoryPropertyFlags compute_memory_flags;
    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
    void* staging_mapped;
    size_t staging_size;
    size_t staging_allocation_size;
    size_t staging_cursor;
    uint32_t staging_memory_type_index;
    VkMemoryPropertyFlags staging_memory_flags;
    uint64_t staging_upload_count;
    uint64_t staging_upload_bytes;
    uint64_t staging_download_count;
    uint64_t staging_download_bytes;
    uint64_t staging_submit_count;
    VkDescriptorPool graph_descriptor_pool;
    VkDescriptorSet matmul_set;
    VkCommandPool pool;
    VkCommandBuffer command;
    VkFence fence;
    VkTensorSlot* tensor_slots;
    int tensor_slot_count;
    VkGraphFreeRange* free_ranges;
    int free_range_count;
    size_t arena_bump;
    size_t scratch_cursor;
    size_t scratch_high_water;
    QConvZeroBiasBacking* zero_bias_backings;
    VkGraphDispatchSet* dispatch_sets;
    int dispatch_set_count;
    int dispatch_set_cursor;
    int command_recording;
    int command_pending;
    int packed_dot_disabled;
    uint64_t qlinear_dot_dispatches;
    uint64_t qlinear_tiled_dispatches;
    uint64_t qlinear_scalar_dispatches;
    uint64_t qbatch_dot_dispatches;
    uint64_t qbatch_scalar_dispatches;
    uint64_t qconv_dot_tiled_dispatches;
    uint64_t qconv_tiled_dispatches;
    uint64_t qconv_scalar_dispatches;
    uint64_t conv_out16_dispatches;
    uint64_t conv_scalar_dispatches;
    uint64_t prepared_pipeline_creates;
    uint64_t prepared_pipeline_creates_at_forward;
    VkPreparedKernel prepared_kernels[VK_PREPARED_KERNEL_MAX];
    int prepared_kernel_count;
    VkWeightCacheEntry weight_cache[WT_CACHE_MAX];
    int weight_cache_count;
    size_t weight_bump;
    char* shape_signature;
    uint64_t shape_generation;
    uint64_t capacity_generation;
    size_t domain_span_count;
    size_t domain_qgroupnorm_stats_bytes;
    size_t domain_qlayernorm_stats_bytes;
    size_t domain_qgroupnorm_stats_offset;
    size_t domain_qlayernorm_stats_offset;
    size_t domain_params_offset;
    int domain_enforced;
#if VOLVOXAI_ENABLE_TRAINING
    VkTrainingKernelSlot* training_kernel_slots;
    int training_kernel_slot_count;
    VkDescriptorPool training_pool;
    VkGraphDispatchSet* training_sets;
    int training_set_count;
    int training_set_cursor;
    VkTensorSlot** training_touched;
    int training_touched_count_value;
    int training_is_active;
#endif
    VkTracePass* trace_pass;
    PFN_vkCmdDispatch dispatch;
    PFN_vkCmdBindPipeline bind_pipeline;
    VxMemoryObserver* memory_observer;
} VulkanContextState;

static VulkanDeviceState g_vulkan_device = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .graph_alignment = VK_GRAPH_ALIGN,
};

static VulkanContextState* vk_context_current(void) {
    VxEngineState* owner = vx_engine_state_current();
    return owner ? (VulkanContextState*)owner->vulkan_context_state : NULL;
}

#define vulkan_lib (g_vulkan_device.api.loader)
#define vkGetInstanceProcAddr (g_vulkan_device.api.get_instance_proc_addr)
#define vkCreateInstance (g_vulkan_device.api.p_vkCreateInstance)
#define vkEnumerateInstanceVersion (g_vulkan_device.api.p_vkEnumerateInstanceVersion)
#define vkDestroyInstance (g_vulkan_device.api.p_vkDestroyInstance)
#define vkEnumeratePhysicalDevices (g_vulkan_device.api.p_vkEnumeratePhysicalDevices)
#define vkGetPhysicalDeviceProperties (g_vulkan_device.api.p_vkGetPhysicalDeviceProperties)
#define vkGetPhysicalDeviceFeatures2 (g_vulkan_device.api.p_vkGetPhysicalDeviceFeatures2)
#define vkGetPhysicalDeviceProperties2 (g_vulkan_device.api.p_vkGetPhysicalDeviceProperties2)
#define vkGetPhysicalDeviceQueueFamilyProperties (g_vulkan_device.api.p_vkGetPhysicalDeviceQueueFamilyProperties)
#define vkGetPhysicalDeviceMemoryProperties (g_vulkan_device.api.p_vkGetPhysicalDeviceMemoryProperties)
#define vkCreateDevice (g_vulkan_device.api.p_vkCreateDevice)
#define vkDestroyDevice (g_vulkan_device.api.p_vkDestroyDevice)
#define vkGetDeviceQueue (g_vulkan_device.api.p_vkGetDeviceQueue)
#define vkCreateCommandPool (g_vulkan_device.api.p_vkCreateCommandPool)
#define vkDestroyCommandPool (g_vulkan_device.api.p_vkDestroyCommandPool)
#define vkAllocateCommandBuffers (g_vulkan_device.api.p_vkAllocateCommandBuffers)
#define vkCreateBuffer (g_vulkan_device.api.p_vkCreateBuffer)
#define vkDestroyBuffer (g_vulkan_device.api.p_vkDestroyBuffer)
#define vkGetBufferMemoryRequirements (g_vulkan_device.api.p_vkGetBufferMemoryRequirements)
#define vkAllocateMemory (g_vulkan_device.api.p_vkAllocateMemory)
#define vkFreeMemory (g_vulkan_device.api.p_vkFreeMemory)
#define vkBindBufferMemory (g_vulkan_device.api.p_vkBindBufferMemory)
#define vkMapMemory (g_vulkan_device.api.p_vkMapMemory)
#define vkUnmapMemory (g_vulkan_device.api.p_vkUnmapMemory)
#define vkFlushMappedMemoryRanges (g_vulkan_device.api.p_vkFlushMappedMemoryRanges)
#define vkInvalidateMappedMemoryRanges (g_vulkan_device.api.p_vkInvalidateMappedMemoryRanges)
#define vkCreateShaderModule (g_vulkan_device.api.p_vkCreateShaderModule)
#define vkDestroyShaderModule (g_vulkan_device.api.p_vkDestroyShaderModule)
#define vkCreateDescriptorSetLayout (g_vulkan_device.api.p_vkCreateDescriptorSetLayout)
#define vkDestroyDescriptorSetLayout (g_vulkan_device.api.p_vkDestroyDescriptorSetLayout)
#define vkCreatePipelineLayout (g_vulkan_device.api.p_vkCreatePipelineLayout)
#define vkDestroyPipelineLayout (g_vulkan_device.api.p_vkDestroyPipelineLayout)
#define vkCreatePipelineCache (g_vulkan_device.api.p_vkCreatePipelineCache)
#define vkDestroyPipelineCache (g_vulkan_device.api.p_vkDestroyPipelineCache)
#define vkCreateComputePipelines (g_vulkan_device.api.p_vkCreateComputePipelines)
#define vkDestroyPipeline (g_vulkan_device.api.p_vkDestroyPipeline)
#define vkCreateDescriptorPool (g_vulkan_device.api.p_vkCreateDescriptorPool)
#define vkDestroyDescriptorPool (g_vulkan_device.api.p_vkDestroyDescriptorPool)
#define vkResetDescriptorPool (g_vulkan_device.api.p_vkResetDescriptorPool)
#define vkAllocateDescriptorSets (g_vulkan_device.api.p_vkAllocateDescriptorSets)
#define vkUpdateDescriptorSets (g_vulkan_device.api.p_vkUpdateDescriptorSets)
#define vkBeginCommandBuffer (g_vulkan_device.api.p_vkBeginCommandBuffer)
#define vkCmdBindPipeline (g_vulkan_device.api.p_vkCmdBindPipeline)
#define vkCmdBindDescriptorSets (g_vulkan_device.api.p_vkCmdBindDescriptorSets)
#define vkCmdPushConstants (g_vulkan_device.api.p_vkCmdPushConstants)
#define vkCmdDispatch (g_vulkan_device.api.p_vkCmdDispatch)
#define vkCmdCopyBuffer (g_vulkan_device.api.p_vkCmdCopyBuffer)
#define vkCmdPipelineBarrier (g_vulkan_device.api.p_vkCmdPipelineBarrier)
#define vkEndCommandBuffer (g_vulkan_device.api.p_vkEndCommandBuffer)
#define vkQueueSubmit (g_vulkan_device.api.p_vkQueueSubmit)
#define vkQueueWaitIdle (g_vulkan_device.api.p_vkQueueWaitIdle)
#define vkDeviceWaitIdle (g_vulkan_device.api.p_vkDeviceWaitIdle)
#define vkCreateFence (g_vulkan_device.api.p_vkCreateFence)
#define vkResetFences (g_vulkan_device.api.p_vkResetFences)
#define vkWaitForFences (g_vulkan_device.api.p_vkWaitForFences)
#define vkDestroyFence (g_vulkan_device.api.p_vkDestroyFence)

#define instance (g_vulkan_device.instance_handle)
#define physical_device (g_vulkan_device.physical_device_handle)
#define device (g_vulkan_device.device_handle)
#define compute_queue (g_vulkan_device.compute_queue_handle)
#define queue_family_index (g_vulkan_device.queue_family_index)
#define mem_props (g_vulkan_device.memory_properties)
#define max_compute_workgroups (g_vulkan_device.max_workgroups)
#define max_storage_buffer_range (g_vulkan_device.max_storage_range)
#define max_uniform_buffer_range (g_vulkan_device.max_uniform_range)
#define non_coherent_atom_size (g_vulkan_device.non_coherent_atom_size)
#define graph_alignment (g_vulkan_device.graph_alignment)
#define vulkan_packed_dot (g_vulkan_device.packed_dot)
#define vulkan_packed_dot_warned (g_vulkan_device.packed_dot_warned)
#define desc_layout (g_vulkan_device.matmul_desc_layout)
#define pipeline_layout (g_vulkan_device.matmul_pipeline_layout)
#define pipeline_cache (g_vulkan_device.pipeline_cache)
#define matmul_pipeline (g_vulkan_device.matmul_pipeline)
#define matmul_tiled_pipeline (g_vulkan_device.matmul_tiled_pipeline)
#if VOLVOXAI_ENABLE_TRAINING
#define training_max_storage_bindings (g_vulkan_device.training_max_storage_bindings)
#define training_max_uniform_bindings (g_vulkan_device.training_max_uniform_bindings)
#define training_max_workgroup_size_x (g_vulkan_device.training_max_workgroup_size_x)
#define training_max_workgroup_invocations (g_vulkan_device.training_max_workgroup_invocations)
#define training_max_storage_range (g_vulkan_device.training_max_storage_range)
#define training_max_uniform_range (g_vulkan_device.training_max_uniform_range)
#endif

#define VX_VK_CONTEXT (*vk_context_current())
#define io_buffer (VX_VK_CONTEXT.buffer)
#define io_memory (VX_VK_CONTEXT.memory)
#define io_size (VX_VK_CONTEXT.arena_size)
#define vk_stage_buffer (VX_VK_CONTEXT.staging_buffer)
#define vk_stage_memory (VX_VK_CONTEXT.staging_memory)
#define vk_stage_mapped (VX_VK_CONTEXT.staging_mapped)
#define vk_stage_size (VX_VK_CONTEXT.staging_size)
#define vk_stage_cursor (VX_VK_CONTEXT.staging_cursor)
#define desc_pool (VX_VK_CONTEXT.graph_descriptor_pool)
#define desc_set (VX_VK_CONTEXT.matmul_set)
#define cmd_pool (VX_VK_CONTEXT.pool)
#define cmd_buf (VX_VK_CONTEXT.command)
#define compute_fence (VX_VK_CONTEXT.fence)
#define graph_slots (VX_VK_CONTEXT.tensor_slots)
#define graph_slot_count (VX_VK_CONTEXT.tensor_slot_count)
#define graph_free_ranges (VX_VK_CONTEXT.free_ranges)
#define graph_free_range_count (VX_VK_CONTEXT.free_range_count)
#define graph_bump (VX_VK_CONTEXT.arena_bump)
#define scratch_bump (VX_VK_CONTEXT.scratch_cursor)
#define scratch_high_water (VX_VK_CONTEXT.scratch_high_water)
#define qconv_zero_bias_backings (VX_VK_CONTEXT.zero_bias_backings)
#define graph_dispatch_sets (VX_VK_CONTEXT.dispatch_sets)
#define graph_dispatch_set_count (VX_VK_CONTEXT.dispatch_set_count)
#define graph_dispatch_set_cursor (VX_VK_CONTEXT.dispatch_set_cursor)
#define graph_cmd_recording (VX_VK_CONTEXT.command_recording)
#define graph_cmd_pending (VX_VK_CONTEXT.command_pending)
#define wt_cache (VX_VK_CONTEXT.weight_cache)
#define wt_cache_n (VX_VK_CONTEXT.weight_cache_count)
#define wt_bump (VX_VK_CONTEXT.weight_bump)
#define graph_shape_signature (VX_VK_CONTEXT.shape_signature)
#define graph_shape_generation (VX_VK_CONTEXT.shape_generation)
#define graph_capacity_generation (VX_VK_CONTEXT.capacity_generation)
#define graph_domain_span_count (VX_VK_CONTEXT.domain_span_count)
#define graph_domain_enforced (VX_VK_CONTEXT.domain_enforced)
#if VOLVOXAI_ENABLE_TRAINING
#define training_kernels (VX_VK_CONTEXT.training_kernel_slots)
#define training_kernel_count (VX_VK_CONTEXT.training_kernel_slot_count)
#define training_desc_pool (VX_VK_CONTEXT.training_pool)
#define training_dispatch_sets (VX_VK_CONTEXT.training_sets)
#define training_dispatch_set_count (VX_VK_CONTEXT.training_set_count)
#define training_dispatch_set_cursor (VX_VK_CONTEXT.training_set_cursor)
#define training_touched_slots (VX_VK_CONTEXT.training_touched)
#define training_touched_count (VX_VK_CONTEXT.training_touched_count_value)
#define training_active (VX_VK_CONTEXT.training_is_active)
#endif

static int vk_graph_flush_wait(void);
static int vk_graph_begin_recording(void);
static int vk_staging_upload(size_t device_offset, const void* source,
                             size_t bytes);
static int vk_staging_zero(size_t device_offset, size_t bytes);
static int vk_staging_download(size_t device_offset, void* destination,
                               size_t bytes);
static int vk_is_ready(void);
void vk_memory_inventory(void) {
    VxEngineState* owner = vx_engine_state_current();
    VulkanContextState* context = vk_context_current();
    if (!owner || !context) return;
    context->memory_observer = &owner->memory_observer;
    vx_memory_record(&owner->memory_observer, VX_MEMORY_VULKAN_GRAPH,
        (uint64_t)context->memory, context->arena_allocation_size, VX_TRACE_MEMORY_ACTION_EXISTING);
    vx_memory_record(&owner->memory_observer, VX_MEMORY_VULKAN_GRAPH,
        (uint64_t)context->staging_memory, context->staging_allocation_size, VX_TRACE_MEMORY_ACTION_EXISTING);
}

static void vk_context_destroy(void* opaque);
static void vk_trace_discard(VulkanContextState* context);
static int vk_graph_allocator_reset(void);
void vk_device_release(void);
static const char* vk_kernel_entry_point(VkKernel* kernel);

static uint32_t find_preferred_memory_type(uint32_t type_filter,
                                           VkMemoryPropertyFlags required,
                                           VkMemoryPropertyFlags preferred) {
    uint32_t fallback = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if (!(type_filter & (1u << i))) continue;
        VkMemoryPropertyFlags flags = mem_props.memoryTypes[i].propertyFlags;
        if ((flags & required) != required) continue;
        if ((flags & preferred) == preferred) return i;
        if (fallback == UINT32_MAX) fallback = i;
    }
    return fallback;
}


static uint32_t* load_spv(const char* path, size_t* size_out) {
    if (!path || !size_out) return NULL;
    *size_out = 0;
    VolvoxAIShaderView view;
    if (volvoxai_shader_store_get(path, &view) !=
            VOLVOXAI_SHADER_STORE_OK ||
        view.size == 0 || (view.size & 3u) != 0) {
        fprintf(stderr, "[Vulkan] failed to load shader %s\n", path);
        return NULL;
    }
    uint32_t* buf = (uint32_t*)malloc(view.size);
    if (!buf) return NULL;
    memcpy(buf, view.data, view.size);
    *size_out = view.size;
    return buf;
}

static int create_matmul_pipeline(const char* path, VkPipeline* output) {
    size_t spv_size = 0;
    uint32_t* spv_code;
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo shader_info = {0};
    VkComputePipelineCreateInfo pipeline_info = {0};
    VkResult result;
    if (!path || !output || pipeline_layout == VK_NULL_HANDLE) return 0;
    spv_code = load_spv(path, &spv_size);
    if (!spv_code) return 0;
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = spv_size;
    shader_info.pCode = spv_code;
    result = vkCreateShaderModule(device, &shader_info, NULL, &shader_module);
    free(spv_code);
    if (result != VK_SUCCESS) return 0;
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shader_module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = pipeline_layout;
    result = vkCreateComputePipelines(device, pipeline_cache, 1,
                                      &pipeline_info, NULL, output);
    if (vkDestroyShaderModule) vkDestroyShaderModule(device, shader_module, NULL);
    return result == VK_SUCCESS;
}

static int vk_device_initialize_locked(void) {
    const char* names[] = { "libvulkan.so.1", "libvulkan.so", "vulkan-1.dll" };
    uint32_t instance_api_version = VK_API_VERSION_1_0;
#if defined(VK_VERSION_1_3) && defined(VK_KHR_shader_integer_dot_product)
    VkPhysicalDeviceShaderIntegerDotProductFeatures dot_features = {0};
#endif
    if (g_vulkan_device.initialized) return 0;
    if (!vulkan_lib) {
        for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
            vulkan_lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
            if (vulkan_lib) break;
        }
    }
    if (!vulkan_lib) return -1;

    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(vulkan_lib, "vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddr) return -1;

    LOAD_GLOBAL(vkCreateInstance)
    if (!vkCreateInstance) return -1;

#if defined(VK_VERSION_1_1)
    vkEnumerateInstanceVersion = (PFN_vkEnumerateInstanceVersion)
        vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceVersion");
    if (vkEnumerateInstanceVersion) {
        uint32_t loader_version = VK_API_VERSION_1_0;
        if (vkEnumerateInstanceVersion(&loader_version) == VK_SUCCESS) {
#if defined(VK_VERSION_1_3)
            instance_api_version = loader_version < VK_API_VERSION_1_3
                ? loader_version : VK_API_VERSION_1_3;
#else
            instance_api_version = loader_version < VK_API_VERSION_1_1
                ? loader_version : VK_API_VERSION_1_1;
#endif
        }
    }
#endif

    VkApplicationInfo appInfo = {0};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = instance_api_version;
    VkInstanceCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&createInfo, NULL, &instance) != VK_SUCCESS) return -1;

    LOAD_INST(vkDestroyInstance)
    LOAD_INST(vkEnumeratePhysicalDevices)
    LOAD_INST(vkGetPhysicalDeviceProperties)
#if defined(VK_VERSION_1_1)
    LOAD_INST(vkGetPhysicalDeviceFeatures2)
    LOAD_INST(vkGetPhysicalDeviceProperties2)
#endif
    LOAD_INST(vkGetPhysicalDeviceQueueFamilyProperties)
    LOAD_INST(vkGetPhysicalDeviceMemoryProperties)
    LOAD_INST(vkCreateDevice)
    LOAD_INST(vkDestroyDevice)
    LOAD_INST(vkGetDeviceQueue)
    LOAD_INST(vkCreateCommandPool)
    LOAD_INST(vkDestroyCommandPool)
    LOAD_INST(vkAllocateCommandBuffers)
    LOAD_INST(vkCreateBuffer)
    LOAD_INST(vkDestroyBuffer)
    LOAD_INST(vkGetBufferMemoryRequirements)
    LOAD_INST(vkAllocateMemory)
    LOAD_INST(vkFreeMemory)
    LOAD_INST(vkBindBufferMemory)
    LOAD_INST(vkMapMemory)
    LOAD_INST(vkUnmapMemory)
    LOAD_INST(vkFlushMappedMemoryRanges)
    LOAD_INST(vkInvalidateMappedMemoryRanges)
    LOAD_INST(vkCreateShaderModule)
    LOAD_INST(vkDestroyShaderModule)
    LOAD_INST(vkCreateDescriptorSetLayout)
    LOAD_INST(vkDestroyDescriptorSetLayout)
    LOAD_INST(vkCreatePipelineLayout)
    LOAD_INST(vkDestroyPipelineLayout)
    LOAD_INST(vkCreatePipelineCache)
    LOAD_INST(vkDestroyPipelineCache)
    LOAD_INST(vkCreateComputePipelines)
    LOAD_INST(vkDestroyPipeline)
    LOAD_INST(vkCreateDescriptorPool)
    LOAD_INST(vkDestroyDescriptorPool)
#if VOLVOXAI_ENABLE_TRAINING
    LOAD_INST(vkResetDescriptorPool)
#endif
    LOAD_INST(vkAllocateDescriptorSets)
    LOAD_INST(vkUpdateDescriptorSets)
    LOAD_INST(vkBeginCommandBuffer)
    LOAD_INST(vkCmdBindPipeline)
    LOAD_INST(vkCmdBindDescriptorSets)
    LOAD_INST(vkCmdPushConstants)
    LOAD_INST(vkCmdDispatch)
    LOAD_INST(vkCmdCopyBuffer)
    LOAD_INST(vkCmdPipelineBarrier)
    LOAD_INST(vkEndCommandBuffer)
    LOAD_INST(vkQueueSubmit)
    LOAD_INST(vkQueueWaitIdle)
    LOAD_INST(vkDeviceWaitIdle)
    LOAD_INST(vkCreateFence)
    LOAD_INST(vkResetFences)
    LOAD_INST(vkWaitForFences)
    LOAD_INST(vkDestroyFence)

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, NULL);
    if (device_count == 0) return -1;
    VkPhysicalDevice* devices = malloc(sizeof(VkPhysicalDevice) * device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices);
    int best_score = -1;
    VkPhysicalDeviceProperties best_props;
    memset(&best_props, 0, sizeof(best_props));
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        int score = 0;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score = 4;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score = 3;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) score = 2;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) score = 1;
        if (score > best_score) {
            best_score = score;
            physical_device = devices[i];
            best_props = props;
        }
    }
    free(devices);

    vulkan_packed_dot = 0;
#if defined(VK_VERSION_1_3) && defined(VK_KHR_shader_integer_dot_product)
    if (instance_api_version >= VK_API_VERSION_1_3 &&
        best_props.apiVersion >= VK_API_VERSION_1_3 &&
        vkGetPhysicalDeviceFeatures2 && vkGetPhysicalDeviceProperties2) {
        VkPhysicalDeviceFeatures2 features = {0};
        VkPhysicalDeviceProperties2 properties = {0};
        VkPhysicalDeviceShaderIntegerDotProductProperties dot_properties = {0};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        dot_features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
        features.pNext = &dot_features;
        vkGetPhysicalDeviceFeatures2(physical_device, &features);
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        dot_properties.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;
        properties.pNext = &dot_properties;
        vkGetPhysicalDeviceProperties2(physical_device, &properties);
        vulkan_packed_dot = dot_features.shaderIntegerDotProduct &&
            dot_properties.integerDotProduct4x8BitPackedSignedAccelerated;
    }
#endif
    {
        const char* disable_dot = getenv("VOLVOX_VULKAN_DISABLE_DOT");
        if (disable_dot && disable_dot[0] && strcmp(disable_dot, "0") != 0) {
            vulkan_packed_dot = 0;
        }
    }

    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
    for (int axis = 0; axis < 3; axis++) {
        max_compute_workgroups[axis] = best_props.limits.maxComputeWorkGroupCount[axis];
        g_vulkan_device.max_workgroup_size[axis] =
            best_props.limits.maxComputeWorkGroupSize[axis];
    }
    g_vulkan_device.max_workgroup_invocations =
        best_props.limits.maxComputeWorkGroupInvocations;
    max_storage_buffer_range = best_props.limits.maxStorageBufferRange;
    max_uniform_buffer_range = best_props.limits.maxUniformBufferRange;
    non_coherent_atom_size = best_props.limits.nonCoherentAtomSize;
    g_vulkan_device.max_storage_bindings =
        best_props.limits.maxPerStageDescriptorStorageBuffers <
                best_props.limits.maxDescriptorSetStorageBuffers
            ? best_props.limits.maxPerStageDescriptorStorageBuffers
            : best_props.limits.maxDescriptorSetStorageBuffers;
    g_vulkan_device.max_uniform_bindings =
        best_props.limits.maxPerStageDescriptorUniformBuffers <
                best_props.limits.maxDescriptorSetUniformBuffers
            ? best_props.limits.maxPerStageDescriptorUniformBuffers
            : best_props.limits.maxDescriptorSetUniformBuffers;
#if VOLVOXAI_ENABLE_TRAINING
    training_max_storage_bindings =
        best_props.limits.maxPerStageDescriptorStorageBuffers <
                best_props.limits.maxDescriptorSetStorageBuffers
            ? best_props.limits.maxPerStageDescriptorStorageBuffers
            : best_props.limits.maxDescriptorSetStorageBuffers;
    training_max_uniform_bindings =
        best_props.limits.maxPerStageDescriptorUniformBuffers <
                best_props.limits.maxDescriptorSetUniformBuffers
            ? best_props.limits.maxPerStageDescriptorUniformBuffers
            : best_props.limits.maxDescriptorSetUniformBuffers;
    training_max_workgroup_size_x = best_props.limits.maxComputeWorkGroupSize[0];
    training_max_workgroup_invocations = best_props.limits.maxComputeWorkGroupInvocations;
    training_max_storage_range = best_props.limits.maxStorageBufferRange;
    training_max_uniform_range = best_props.limits.maxUniformBufferRange;
#endif
    if (best_props.limits.minStorageBufferOffsetAlignment > graph_alignment) {
        graph_alignment = (size_t)best_props.limits.minStorageBufferOffsetAlignment;
    }
    if (best_props.limits.minUniformBufferOffsetAlignment > graph_alignment) {
        graph_alignment = (size_t)best_props.limits.minUniformBufferOffsetAlignment;
    }

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, NULL);
    VkQueueFamilyProperties* queue_props = malloc(sizeof(VkQueueFamilyProperties) * queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_props);
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queue_family_index = i;
            g_vulkan_device.timestamp_bits = queue_props[i].timestampValidBits;
            g_vulkan_device.timestamp_period = best_props.limits.timestampPeriod;
            break;
        }
    }
    free(queue_props);

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {0};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family_index;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_info = {0};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
#if defined(VK_VERSION_1_3) && defined(VK_KHR_shader_integer_dot_product)
    device_info.pNext = vulkan_packed_dot ? &dot_features : NULL;
#endif

#if defined(VK_EXT_calibrated_timestamps)
    /* Extension enablement is a device capability decision. Actual clock
     * sampling happens only inside a requested device capture. */
    const char* clock_extension = VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME;
    PFN_vkEnumerateDeviceExtensionProperties enumerate_extensions =
        (PFN_vkEnumerateDeviceExtensionProperties)vkGetInstanceProcAddr(instance, "vkEnumerateDeviceExtensionProperties");
    uint32_t extension_count = 0;
    if (instance_api_version >= VK_MAKE_VERSION(1, 1, 0) && enumerate_extensions &&
        enumerate_extensions(physical_device, NULL, &extension_count, NULL) == VK_SUCCESS && extension_count < 4096) {
        VkExtensionProperties* extensions = calloc(extension_count, sizeof(*extensions));
        if (extensions && enumerate_extensions(physical_device, NULL, &extension_count, extensions) == VK_SUCCESS)
            for (uint32_t i = 0; i < extension_count; i++)
                if (!strcmp(extensions[i].extensionName, clock_extension)) {
                    device_info.enabledExtensionCount = 1;
                    device_info.ppEnabledExtensionNames = &clock_extension;
                    g_vulkan_device.calibrated_timestamps = 1;
                    break;
                }
        free(extensions);
    }
#endif

    if (vkCreateDevice(physical_device, &device_info, NULL, &device) != VK_SUCCESS) return -1;
    vkGetDeviceQueue(device, queue_family_index, 0, &compute_queue);

    /* Keep the driver's process-local compile cache alive across encoder and
     * decoder contexts. Contexts still own their descriptor sets and prepared
     * pipeline handles; the shared cache only reuses driver compilation data. */
    if (vkCreatePipelineCache) {
        VkPipelineCacheCreateInfo cache_info = {0};
        cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        if (vkCreatePipelineCache(device, &cache_info, NULL,
                                  &pipeline_cache) != VK_SUCCESS) {
            pipeline_cache = VK_NULL_HANDLE;
        }
    }

    /* Model-independent matmul pipelines are immutable after initialization
     * and shared through the synchronized device state. */
    VkDescriptorSetLayoutBinding bindings[6];
    for(int i=0; i<6; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = (i == 5) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = NULL;
    }
    VkDescriptorSetLayoutCreateInfo layout_info = {0};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 6;
    layout_info.pBindings = bindings;
    vkCreateDescriptorSetLayout(device, &layout_info, NULL, &desc_layout);

    VkPipelineLayoutCreateInfo p_layout_info = {0};
    p_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    p_layout_info.setLayoutCount = 1;
    p_layout_info.pSetLayouts = &desc_layout;
    p_layout_info.pushConstantRangeCount = 0;
    p_layout_info.pPushConstantRanges = NULL;
    vkCreatePipelineLayout(device, &p_layout_info, NULL, &pipeline_layout);

    if (!create_matmul_pipeline("spv/linearF32.spv", &matmul_pipeline) ||
        !create_matmul_pipeline("spv/linearF32Tiled.spv", &matmul_tiled_pipeline)) return -1;

    g_vulkan_device.initialized = 1;
    if (!g_vulkan_device.exit_hook_registered) {
        g_vulkan_device.exit_hook_registered = 1;
        atexit(vk_device_release);
    }
    printf("[VolvoxAI GPU] Vulkan Compute initialized successfully! Device: %s; packed INT8 dot: %s\n",
           best_props.deviceName, vulkan_packed_dot ? "enabled" : "unavailable");
    return 0;
}

static void vk_destroy_prepared_kernel_locked(VkPreparedKernel* kernel) {
    if (!kernel || device == VK_NULL_HANDLE) return;
    if (kernel->pipeline != VK_NULL_HANDLE && vkDestroyPipeline)
        vkDestroyPipeline(device, kernel->pipeline, NULL);
    if (kernel->prepared_pipeline_layout != VK_NULL_HANDLE && vkDestroyPipelineLayout)
        vkDestroyPipelineLayout(device, kernel->prepared_pipeline_layout, NULL);
    if (kernel->prepared_desc_layout != VK_NULL_HANDLE && vkDestroyDescriptorSetLayout)
        vkDestroyDescriptorSetLayout(device, kernel->prepared_desc_layout, NULL);
    memset(kernel, 0, sizeof(*kernel));
}

static void vk_device_destroy_locked(void) {
#if defined(RTLD_NOLOAD)
    /* An ICD can load EGL while creating Vulkan pipelines. libEGL installs
     * pthread TLS destructors, but may be unloaded by vkDestroyInstance before
     * the RPC worker exits. Keep an already-loaded EGL loader resident, just
     * as the OpenGL backend does. Device/instance resources are still freed.
     * NOLOAD avoids introducing an EGL dependency for other Vulkan drivers. */
    static void* retained_egl_loader;
    if (!retained_egl_loader) {
        const char* names[] = {"libEGL.so.1", "libEGL.so"};
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
            retained_egl_loader = dlopen(names[i], RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
            if (retained_egl_loader) break;
        }
    }
#endif
    if (device != VK_NULL_HANDLE) {
        if (vkDeviceWaitIdle) (void)vkDeviceWaitIdle(device);
        if (matmul_pipeline != VK_NULL_HANDLE && vkDestroyPipeline)
            vkDestroyPipeline(device, matmul_pipeline, NULL);
        if (matmul_tiled_pipeline != VK_NULL_HANDLE && vkDestroyPipeline)
            vkDestroyPipeline(device, matmul_tiled_pipeline, NULL);
        if (pipeline_cache != VK_NULL_HANDLE && vkDestroyPipelineCache)
            vkDestroyPipelineCache(device, pipeline_cache, NULL);
        if (pipeline_layout != VK_NULL_HANDLE && vkDestroyPipelineLayout)
            vkDestroyPipelineLayout(device, pipeline_layout, NULL);
        if (desc_layout != VK_NULL_HANDLE && vkDestroyDescriptorSetLayout)
            vkDestroyDescriptorSetLayout(device, desc_layout, NULL);
        if (vkDestroyDevice) vkDestroyDevice(device, NULL);
    }
    if (instance != VK_NULL_HANDLE && vkDestroyInstance)
        vkDestroyInstance(instance, NULL);
    instance = VK_NULL_HANDLE;
    physical_device = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
    compute_queue = VK_NULL_HANDLE;
    queue_family_index = 0;
    memset(&mem_props, 0, sizeof(mem_props));
    memset(max_compute_workgroups, 0, sizeof(g_vulkan_device.max_workgroups));
    memset(g_vulkan_device.max_workgroup_size, 0,
           sizeof(g_vulkan_device.max_workgroup_size));
    g_vulkan_device.max_workgroup_invocations = 0;
    g_vulkan_device.max_storage_bindings = 0;
    g_vulkan_device.max_uniform_bindings = 0;
    max_storage_buffer_range = 0;
    max_uniform_buffer_range = 0;
    non_coherent_atom_size = 0;
    graph_alignment = VK_GRAPH_ALIGN;
    vulkan_packed_dot = 0;
    vulkan_packed_dot_warned = 0;
    desc_layout = VK_NULL_HANDLE;
    pipeline_layout = VK_NULL_HANDLE;
    matmul_pipeline = VK_NULL_HANDLE;
    matmul_tiled_pipeline = VK_NULL_HANDLE;
    pipeline_cache = VK_NULL_HANDLE;
    g_vulkan_device.prepared_pipeline_creates = 0u;
#if VOLVOXAI_ENABLE_TRAINING
    training_max_storage_bindings = 0;
    training_max_uniform_bindings = 0;
    training_max_workgroup_size_x = 0;
    training_max_workgroup_invocations = 0;
    training_max_storage_range = 0;
    training_max_uniform_range = 0;
#endif
    g_vulkan_device.initialized = 0;
    g_vulkan_device.calibrated_timestamps = 0;
    atomic_store(&g_vulkan_device.trace_device_id, 0);
    atomic_store(&g_vulkan_device.trace_queue_id, 0);
}

static VulkanContextState* vk_context_allocate(VxEngineState* owner) {
    VulkanContextState* context;
    if (!owner) return NULL;
    if (owner->vulkan_context_state)
        return (VulkanContextState*)owner->vulkan_context_state;
    context = (VulkanContextState*)calloc(1, sizeof(*context));
    if (!context) return NULL;
    context->memory_observer = &owner->memory_observer;
    context->tensor_slots = (VkTensorSlot*)calloc(
        VK_GRAPH_MAX_TENSORS, sizeof(*context->tensor_slots));
    context->free_ranges = (VkGraphFreeRange*)calloc(
        VK_GRAPH_MAX_FREE_RANGES, sizeof(*context->free_ranges));
    context->dispatch_sets = (VkGraphDispatchSet*)calloc(
        VK_GRAPH_MAX_DISPATCH_SETS, sizeof(*context->dispatch_sets));
    if (!context->tensor_slots || !context->free_ranges ||
        !context->dispatch_sets) {
        free(context->dispatch_sets);
        free(context->free_ranges);
        free(context->tensor_slots);
        free(context);
        return NULL;
    }
    context->arena_size = (size_t)512 * 1024 * 1024;
    context->arena_bump = VK_GRAPH_BASE;
    context->shape_generation = 1;
    context->capacity_generation = 1;
    owner->vulkan_context_state = context;
    return context;
}


static int vk_context_create_resources_locked(VulkanContextState* context) {
    context->dispatch = vkCmdDispatch;
    context->bind_pipeline = vkCmdBindPipeline;
    VkBufferCreateInfo buffer_info = {0};
    VkMemoryRequirements memory_requirements;
    VkMemoryRequirements staging_requirements;
    VkMemoryAllocateInfo allocation_info = {0};
    VkDescriptorPoolSize pool_sizes[2] = {0};
    VkDescriptorPoolCreateInfo pool_info = {0};
    VkDescriptorSetAllocateInfo set_info = {0};
    VkCommandPoolCreateInfo command_pool_info = {0};
    VkCommandBufferAllocateInfo command_buffer_info = {0};
    VkFenceCreateInfo fence_info = {0};
    const char* mb_env;
    if (!context || !g_vulkan_device.initialized) return -1;
    if (!vkCmdCopyBuffer || !vkFlushMappedMemoryRanges ||
        !vkInvalidateMappedMemoryRanges || !non_coherent_atom_size) {
        fprintf(stderr,
                "[Vulkan] required staging/copy synchronization functions are unavailable\n");
        return -1;
    }

    mb_env = getenv("VOLVOX_VULKAN_MB");
    if (mb_env) {
        size_t configured_arena_size;
        if (!vk_graph_parse_arena_mebibytes(
                mb_env, graph_alignment, &configured_arena_size)) {
            fprintf(stderr,
                    "[Vulkan] invalid VOLVOX_VULKAN_MB; expected an exact decimal MiB size representable by size_t and large enough for the graph base and scratch regions\n");
            return -1;
        }
        context->arena_size = configured_arena_size;
    }
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = context->arena_size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &buffer_info, NULL, &context->buffer) != VK_SUCCESS) {
        fprintf(stderr,
                "[Vulkan] compute arena buffer creation failed (%zu bytes)\n",
                context->arena_size);
        return -1;
    }
    vkGetBufferMemoryRequirements(device, context->buffer, &memory_requirements);
    if (memory_requirements.size > (VkDeviceSize)SIZE_MAX) {
        fprintf(stderr,
                "[Vulkan] compute arena allocation exceeds host size_t range\n");
        return -1;
    }
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = memory_requirements.size;
    context->compute_memory_type_index = find_preferred_memory_type(
        memory_requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0u);
    if (context->compute_memory_type_index == UINT32_MAX) {
        fprintf(stderr,
                "[Vulkan] no DEVICE_LOCAL memory type can back the compute arena\n");
        return -1;
    }
    allocation_info.memoryTypeIndex = context->compute_memory_type_index;
    if (vkAllocateMemory(device, &allocation_info, NULL,
                         &context->memory) != VK_SUCCESS) {
        fprintf(stderr,
                "[Vulkan] DEVICE_LOCAL compute arena allocation failed (%zu bytes); refusing host-memory fallback\n",
                (size_t)memory_requirements.size);
        return -1;
    }
    if (vx_engine_state_current()->memory_observer.owner)
        vx_memory_record(&vx_engine_state_current()->memory_observer, VX_MEMORY_VULKAN_GRAPH,
            (uint64_t)context->memory, memory_requirements.size, VX_TRACE_MEMORY_ACTION_ALLOCATE);
    context->compute_memory_flags = mem_props.memoryTypes[
        context->compute_memory_type_index].propertyFlags;
    context->arena_allocation_size = (size_t)memory_requirements.size;
    if (vkBindBufferMemory(device, context->buffer,
                           context->memory, 0) != VK_SUCCESS) {
        fprintf(stderr,
                "[Vulkan] DEVICE_LOCAL compute arena bind failed\n");
        return -1;
    }

    context->staging_size = VK_GRAPH_STAGING_BYTES;
    buffer_info.size = context->staging_size;
    buffer_info.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(device, &buffer_info, NULL,
                       &context->staging_buffer) != VK_SUCCESS) {
        fprintf(stderr,
                "[Vulkan] staging buffer creation failed (%zu bytes)\n",
                context->staging_size);
        return -1;
    }
    vkGetBufferMemoryRequirements(
        device, context->staging_buffer, &staging_requirements);
    if (staging_requirements.size > (VkDeviceSize)SIZE_MAX) {
        fprintf(stderr,
                "[Vulkan] staging allocation exceeds host size_t range\n");
        return -1;
    }
    allocation_info.allocationSize = staging_requirements.size;
    context->staging_memory_type_index = find_preferred_memory_type(
        staging_requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (context->staging_memory_type_index == UINT32_MAX) {
        fprintf(stderr,
                "[Vulkan] no HOST_VISIBLE memory type can back the staging buffer\n");
        return -1;
    }
    allocation_info.memoryTypeIndex = context->staging_memory_type_index;
    if (vkAllocateMemory(device, &allocation_info, NULL,
                         &context->staging_memory) != VK_SUCCESS) {
        fprintf(stderr,
                "[Vulkan] HOST_VISIBLE staging allocation failed (%zu bytes)\n",
                (size_t)staging_requirements.size);
        return -1;
    }
    if (vx_engine_state_current()->memory_observer.owner)
        vx_memory_record(&vx_engine_state_current()->memory_observer, VX_MEMORY_VULKAN_GRAPH,
            (uint64_t)context->staging_memory, staging_requirements.size, VX_TRACE_MEMORY_ACTION_ALLOCATE);
    context->staging_memory_flags = mem_props.memoryTypes[
        context->staging_memory_type_index].propertyFlags;
    context->staging_allocation_size = (size_t)staging_requirements.size;
    if (vkBindBufferMemory(device, context->staging_buffer,
                           context->staging_memory, 0) != VK_SUCCESS) {
        fprintf(stderr, "[Vulkan] staging buffer bind failed\n");
        return -1;
    }
    if (vkMapMemory(device, context->staging_memory, 0, VK_WHOLE_SIZE, 0,
                    &context->staging_mapped) != VK_SUCCESS ||
        !context->staging_mapped) {
        fprintf(stderr, "[Vulkan] staging memory map failed\n");
        return -1;
    }

    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[0].descriptorCount = 16384;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[1].descriptorCount = 4096;
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = VK_GRAPH_MAX_DISPATCH_SETS + 64;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = pool_sizes;
    if (vkCreateDescriptorPool(device, &pool_info, NULL,
                               &context->graph_descriptor_pool) != VK_SUCCESS) return -1;
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = context->graph_descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &desc_layout;
    if (vkAllocateDescriptorSets(device, &set_info,
                                 &context->matmul_set) != VK_SUCCESS) return -1;

    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = queue_family_index;
    if (vkCreateCommandPool(device, &command_pool_info, NULL,
                            &context->pool) != VK_SUCCESS) return -1;
    command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_info.commandPool = context->pool;
    command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_info.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &command_buffer_info,
                                 &context->command) != VK_SUCCESS) return -1;
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device, &fence_info, NULL,
                      &context->fence) != VK_SUCCESS) return -1;
    if (!vk_graph_allocator_reset()) return -1;
    return 0;
}

static void vk_context_destroy(void* opaque) {
    VulkanContextState* context = (VulkanContextState*)opaque;
    if (!context) return;
    if (context->device_acquired) {
        pthread_mutex_lock(&g_vulkan_device.mutex);
        if (device != VK_NULL_HANDLE) {
            if (context->command_pending && vkWaitForFences)
                (void)vkWaitForFences(device, 1, &context->fence,
                                      VK_TRUE, UINT64_MAX);
            vk_trace_discard(context);
            for (int i = 0; i < context->prepared_kernel_count; i++)
                vk_destroy_prepared_kernel_locked(&context->prepared_kernels[i]);
#if VOLVOXAI_ENABLE_TRAINING
            if (context->training_pool != VK_NULL_HANDLE &&
                vkDestroyDescriptorPool)
                vkDestroyDescriptorPool(device, context->training_pool, NULL);
#endif
            if (context->graph_descriptor_pool != VK_NULL_HANDLE && vkDestroyDescriptorPool)
                vkDestroyDescriptorPool(device, context->graph_descriptor_pool, NULL);
            if (context->fence != VK_NULL_HANDLE && vkDestroyFence)
                vkDestroyFence(device, context->fence, NULL);
            if (context->pool != VK_NULL_HANDLE && vkDestroyCommandPool)
                vkDestroyCommandPool(device, context->pool, NULL);
            if (context->staging_mapped && vkUnmapMemory)
                vkUnmapMemory(device, context->staging_memory);
            if (context->staging_buffer != VK_NULL_HANDLE && vkDestroyBuffer)
                vkDestroyBuffer(device, context->staging_buffer, NULL);
            if (context->staging_memory != VK_NULL_HANDLE && vkFreeMemory)
                { vkFreeMemory(device, context->staging_memory, NULL);
                  if (context->memory_observer)
                      vx_memory_record(context->memory_observer, VX_MEMORY_VULKAN_GRAPH,
                          (uint64_t)context->staging_memory, 0, VX_TRACE_MEMORY_ACTION_FREE); }
            if (context->buffer != VK_NULL_HANDLE && vkDestroyBuffer)
                vkDestroyBuffer(device, context->buffer, NULL);
            if (context->memory != VK_NULL_HANDLE && vkFreeMemory)
                { vkFreeMemory(device, context->memory, NULL);
                  if (context->memory_observer)
                      vx_memory_record(context->memory_observer, VX_MEMORY_VULKAN_GRAPH,
                          (uint64_t)context->memory, 0, VX_TRACE_MEMORY_ACTION_FREE); }
        }
        context->device_acquired = 0;
        if (g_vulkan_device.context_count > 0)
            g_vulkan_device.context_count--;
        /* The per-context resources above are gone, but the device and instance
         * stay. Compiling a model and creating its execution context run under
         * different engine-state scopes, so the count legitimately returns to
         * zero between them; destroying the device there made every native run
         * build and tear down a full VkDevice twice. vk_device_initialize_locked
         * is idempotent, so the next scope reuses this one. Released explicitly
         * by vk_device_release(). */
        pthread_mutex_unlock(&g_vulkan_device.mutex);
    }
    while (context->zero_bias_backings) {
        QConvZeroBiasBacking* block = context->zero_bias_backings;
        context->zero_bias_backings = block->next;
        free(block->values);
        free(block);
    }
#if VOLVOXAI_ENABLE_TRAINING
    free(context->training_kernel_slots);
    free(context->training_sets);
    free(context->training_touched);
#endif
    free(context->dispatch_sets);
    free(context->shape_signature);
    free(context->free_ranges);
    free(context->tensor_slots);
    free(context);
}

int vk_init(void) {
    VxEngineState* owner = vx_engine_state_current();
    VulkanContextState* context;
    if (!owner) return -1;
    context = vk_context_allocate(owner);
    if (!context) return -1;
    owner->vulkan_context_state_destroy = vk_context_destroy;
    if (context->device_acquired) return vk_is_ready() ? 0 : -1;

    pthread_mutex_lock(&g_vulkan_device.mutex);
    if (vk_device_initialize_locked() != 0) {
        vk_device_destroy_locked();
        pthread_mutex_unlock(&g_vulkan_device.mutex);
        return -1;
    }
    context->device_acquired = 1;
    g_vulkan_device.context_count++;
    if (vk_context_create_resources_locked(context) != 0) {
        pthread_mutex_unlock(&g_vulkan_device.mutex);
        vk_cleanup();
        return -1;
    }
    pthread_mutex_unlock(&g_vulkan_device.mutex);
    return 0;
}

void vk_cleanup(void) {
    VxEngineState* owner = vx_engine_state_current();
    VulkanContextState* context;
    if (!owner || !owner->vulkan_context_state) return;
#if VOLVOXAI_ENABLE_TRAINING
    vk_training_end();
#endif
    context = (VulkanContextState*)owner->vulkan_context_state;
    owner->vulkan_context_state = NULL;
    owner->vulkan_context_state_destroy = NULL;
    vk_context_destroy(context);
}

/* Drop the shared device once nothing holds it. Separate from vk_cleanup so the
 * device survives the engine-state scope churn between compile and execute, and
 * is released only when the backend itself is deactivated. */
void vk_device_release(void) {
    pthread_mutex_lock(&g_vulkan_device.mutex);
    if (g_vulkan_device.context_count == 0 && g_vulkan_device.initialized)
        vk_device_destroy_locked();
    pthread_mutex_unlock(&g_vulkan_device.mutex);
}

static int vk_is_ready(void) {
    VulkanContextState* context = vk_context_current();
    return context && context->device_acquired &&
        device != VK_NULL_HANDLE && context->buffer != VK_NULL_HANDLE &&
        context->memory != VK_NULL_HANDLE &&
        (context->compute_memory_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
        context->staging_buffer != VK_NULL_HANDLE &&
        context->staging_memory != VK_NULL_HANDLE &&
        context->staging_mapped != NULL && context->staging_size > 0u;
}

static int graph_find_slot(const void* host) {
    if (!host) return -1;
    for (int i = 0; i < graph_slot_count; i++) {
        if (graph_slots[i].host == host &&
            (graph_slots[i].is_weight ||
             graph_slots[i].shape_generation == graph_shape_generation))
            return i;
    }
    return -1;
}

static uint64_t vk_graph_generation_next(uint64_t generation) {
    generation++;
    return generation ? generation : 1u;
}

static size_t vk_graph_arena_limit(void) {
    size_t limit = io_size > VK_GRAPH_SCRATCH_BYTES
        ? io_size - VK_GRAPH_SCRATCH_BYTES : 0;
    return graph_alignment ? limit - (limit % graph_alignment) : 0;
}

int vk_query_domain_limits(VulkanDomainLimits* limits) {
    VulkanContextState* context = vk_context_current();
    size_t arena_start = 0;
    size_t arena_limit = 0;
    size_t scratch_start = 0;
    int ready = 0;
    if (!limits) return -1;
    memset(limits, 0, sizeof(*limits));
    if (!context || !context->device_acquired) return -1;
    if (pthread_mutex_lock(&g_vulkan_device.mutex) != 0) return -1;
    if (g_vulkan_device.initialized && vk_is_ready() &&
        vk_graph_align_up(VK_GRAPH_BASE, graph_alignment, &arena_start) &&
        io_size >= VK_GRAPH_SCRATCH_BYTES &&
        vk_graph_align_up(io_size - VK_GRAPH_SCRATCH_BYTES,
                          graph_alignment, &scratch_start) &&
        scratch_start <= io_size &&
        (!scratch_high_water ||
         (scratch_high_water >= scratch_start &&
          scratch_high_water <= io_size))) {
        arena_limit = vk_graph_arena_limit();
        limits->maximum_storage_buffer_bytes =
            (uint64_t)max_storage_buffer_range;
        limits->maximum_uniform_buffer_bytes =
            (uint64_t)max_uniform_buffer_range;
        limits->maximum_total_span_bytes = arena_limit > arena_start
            ? (uint64_t)(arena_limit - arena_start) : 0u;
        limits->compute_arena_allocation_bytes =
            (uint64_t)context->arena_allocation_size;
        limits->staging_allocation_bytes =
            (uint64_t)context->staging_allocation_size;
        limits->maximum_scratch_bytes = (uint64_t)(io_size - scratch_start);
        limits->current_graph_scratch_bytes = scratch_high_water
            ? (uint64_t)(scratch_high_water - scratch_start) : 0u;
        limits->storage_alignment = (uint64_t)graph_alignment;
        for (size_t axis = 0; axis < 3u; axis++) {
            limits->maximum_workgroups[axis] =
                g_vulkan_device.max_workgroups[axis];
            limits->maximum_workgroup_size[axis] =
                g_vulkan_device.max_workgroup_size[axis];
        }
        limits->maximum_workgroup_invocations =
            g_vulkan_device.max_workgroup_invocations;
        limits->maximum_storage_bindings =
            g_vulkan_device.max_storage_bindings;
        limits->maximum_uniform_bindings =
            g_vulkan_device.max_uniform_bindings;
        limits->maximum_tensor_slots = VK_GRAPH_MAX_TENSORS;
        limits->maximum_dispatches = VK_GRAPH_MAX_DISPATCH_SETS;
        ready = limits->maximum_storage_buffer_bytes > 0u &&
            limits->maximum_uniform_buffer_bytes > 0u &&
            limits->maximum_total_span_bytes > 0u &&
            limits->compute_arena_allocation_bytes >= (uint64_t)io_size &&
            limits->staging_allocation_bytes >=
                (uint64_t)context->staging_size &&
            limits->maximum_scratch_bytes > 0u &&
            limits->current_graph_scratch_bytes <=
                limits->maximum_scratch_bytes &&
            limits->storage_alignment > 0u &&
            limits->maximum_workgroups[0] > 0u &&
            limits->maximum_workgroups[1] > 0u &&
            limits->maximum_workgroups[2] > 0u &&
            limits->maximum_workgroup_size[0] > 0u &&
            limits->maximum_workgroup_size[1] > 0u &&
            limits->maximum_workgroup_size[2] > 0u &&
            limits->maximum_workgroup_invocations > 0u &&
            limits->maximum_storage_bindings > 0u &&
            limits->maximum_uniform_bindings > 0u;
    }
    (void)pthread_mutex_unlock(&g_vulkan_device.mutex);
    return ready ? 0 : -1;
}

static int vk_graph_allocator_reset(void) {
    size_t start;
    size_t limit;
    if (!vk_context_current() || !graph_free_ranges ||
        !vk_graph_align_up(VK_GRAPH_BASE, graph_alignment, &start)) return 0;
    limit = vk_graph_arena_limit();
    if (start >= limit) return 0;
    memset(graph_free_ranges, 0,
           VK_GRAPH_MAX_FREE_RANGES * sizeof(*graph_free_ranges));
    graph_free_ranges[0] = (VkGraphFreeRange){start, limit - start};
    graph_free_range_count = 1;
    graph_bump = start;
    return 1;
}

static int vk_graph_range_free(size_t offset, size_t capacity) {
    size_t limit = vk_graph_arena_limit();
    int at = 0;
    if (!capacity || offset < VK_GRAPH_BASE || offset > limit ||
        capacity > limit - offset) return 0;
    while (at < graph_free_range_count &&
           graph_free_ranges[at].offset < offset) at++;
    if (at > 0) {
        VkGraphFreeRange* previous = &graph_free_ranges[at - 1];
        if (previous->offset > offset ||
            previous->bytes > offset - previous->offset) return 0;
        if (previous->offset + previous->bytes == offset) {
            if (capacity > SIZE_MAX - previous->bytes) return 0;
            previous->bytes += capacity;
            if (at < graph_free_range_count &&
                previous->offset + previous->bytes ==
                    graph_free_ranges[at].offset) {
                previous->bytes += graph_free_ranges[at].bytes;
                memmove(&graph_free_ranges[at], &graph_free_ranges[at + 1],
                        (size_t)(graph_free_range_count - at - 1) *
                            sizeof(*graph_free_ranges));
                graph_free_range_count--;
            }
            return 1;
        }
    }
    if (at < graph_free_range_count) {
        VkGraphFreeRange* next = &graph_free_ranges[at];
        if (capacity > next->offset - offset) return 0;
        if (offset + capacity == next->offset) {
            next->offset = offset;
            next->bytes += capacity;
            return 1;
        }
    }
    if (graph_free_range_count >= VK_GRAPH_MAX_FREE_RANGES) return 0;
    memmove(&graph_free_ranges[at + 1], &graph_free_ranges[at],
            (size_t)(graph_free_range_count - at) *
                sizeof(*graph_free_ranges));
    graph_free_ranges[at] = (VkGraphFreeRange){offset, capacity};
    graph_free_range_count++;
    return 1;
}

static int vk_graph_range_alloc(size_t bytes, size_t* offset,
                                size_t* capacity) {
    size_t aligned;
    int best = -1;
    if (!offset || !capacity || !bytes ||
        !vk_graph_align_up(bytes, graph_alignment, &aligned)) return 0;
    for (int index = 0; index < graph_free_range_count; index++) {
        if (graph_free_ranges[index].bytes < aligned) continue;
        if (best < 0 || graph_free_ranges[index].bytes <
                          graph_free_ranges[best].bytes) best = index;
    }
    if (best < 0) return 0;
    *offset = graph_free_ranges[best].offset;
    *capacity = aligned;
    graph_free_ranges[best].offset += aligned;
    graph_free_ranges[best].bytes -= aligned;
    if (!graph_free_ranges[best].bytes) {
        memmove(&graph_free_ranges[best], &graph_free_ranges[best + 1],
                (size_t)(graph_free_range_count - best - 1) *
                    sizeof(*graph_free_ranges));
        graph_free_range_count--;
    }
    if (*offset + aligned > graph_bump) graph_bump = *offset + aligned;
    return 1;
}

static int vk_graph_reclaim_pooled_ranges(int skip_index) {
    int reclaimed = 0;
    for (int index = 0; index < graph_slot_count; index++) {
        VkTensorSlot* slot = &graph_slots[index];
        if (index == skip_index || slot->host || slot->is_weight ||
            !slot->capacity || !slot->owns_range) continue;
        if (!vk_graph_range_free(slot->offset, slot->capacity)) continue;
        slot->offset = 0;
        slot->capacity = 0;
        slot->capacity_generation = 0;
        reclaimed = 1;
    }
    if (reclaimed)
        graph_capacity_generation =
            vk_graph_generation_next(graph_capacity_generation);
    return reclaimed;
}

static int vk_graph_reusable_slot(size_t bytes) {
    int best = -1;
    int empty = -1;
    int grow = -1;
    for (int index = 0; index < graph_slot_count; index++) {
        VkTensorSlot* slot = &graph_slots[index];
        if (slot->host || slot->is_weight ||
            (slot->capacity && !slot->owns_range)) continue;
        if (!slot->capacity) {
            if (empty < 0) empty = index;
            continue;
        }
        if (slot->capacity >= bytes &&
            (best < 0 || slot->capacity < graph_slots[best].capacity))
            best = index;
        else if (slot->capacity < bytes &&
                 (grow < 0 || slot->capacity > graph_slots[grow].capacity))
            grow = index;
    }
    /* Reuse an undersized pooled metadata slot before appending another one.
     * Its range remains live until a replacement range is secured, so growth
     * is still transactional while alternating/increasing shapes keep the
     * slot table bounded by the graph's peak simultaneous tensor count. */
    return best >= 0 ? best : (empty >= 0 ? empty : grow);
}

static VkTensorSlot* graph_get_slot(const void* host, size_t bytes, int is_weight) {
    VkTensorSlot staged;
    size_t new_offset = 0;
    size_t new_capacity = 0;
    size_t requested_capacity = bytes;
    int is_new = 0;
    if (!vk_is_ready() || !host || bytes == 0 ||
        !max_storage_buffer_range ||
        bytes > (size_t)max_storage_buffer_range) return NULL;
    int idx = graph_find_slot(host);
    if (graph_domain_enforced) {
        VkTensorSlot* slot = idx >= 0 ? &graph_slots[idx] : NULL;
        if (!slot || !slot->owns_range || !slot->capacity ||
            bytes > slot->capacity) return NULL;
        if (slot->domain_span) {
            /* `is_weight` describes this use, not the lifetime of a public
             * tensor. Keep a reserved public span mutable even when a kernel
             * binds it as a weight or affine operand. */
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
    if (idx >= 0) {
        VkTensorSlot* s = &graph_slots[idx];
        if (bytes <= s->capacity) {
            /*
             * Grow, never shrink.
             *
             * A slot's byte extent is a property of the tensor for as long as
             * the shape generation lasts; a caller asking for fewer bytes is
             * asking about part of it, not redefining it. Assigning the request
             * made a partial write shrink the tensor, and the next whole-tensor
             * read then looked like a *growth* -- which sets host_dirty and
             * re-uploads the host copy over rows the device had just written.
             *
             * A batched decode row is exactly that shape: one dispatch per
             * contiguous run of lane rows, each naming the tensor base with one
             * run's worth of bytes. The first run shrank the slot, the second
             * found nothing containing it and minted an orphan, and attention
             * then read a tensor re-uploaded from a host mirror that never had
             * the rows. A real shape change retires the slot by generation, so
             * nothing here needs shrinking to express it.
             */
            if (bytes > s->bytes) {
                s->host_dirty = 1;
                s->device_dirty = 0;
                s->bytes = bytes;
            }
            if (is_weight) s->is_weight = 1;
            return s;
        }
    } else {
        idx = vk_graph_reusable_slot(bytes);
        if (idx < 0) {
            if (graph_slot_count >= VK_GRAPH_MAX_TENSORS) return NULL;
            idx = graph_slot_count;
            is_new = 1;
        } else if (graph_slots[idx].capacity >= bytes) {
            VkTensorSlot* reused = &graph_slots[idx];
            reused->host = host;
            reused->bytes = bytes;
            reused->shape_generation = is_weight ? 0 : graph_shape_generation;
            reused->host_dirty = 1;
            reused->device_dirty = 0;
            reused->is_weight = is_weight;
            return reused;
        }
    }

    staged = is_new ? (VkTensorSlot){0} : graph_slots[idx];
    if (staged.capacity && staged.capacity <= SIZE_MAX / 2u &&
        staged.capacity * 2u > requested_capacity)
        requested_capacity = staged.capacity * 2u;
    if (!vk_graph_range_alloc(requested_capacity, &new_offset,
                              &new_capacity) &&
        (requested_capacity == bytes ||
         !vk_graph_range_alloc(bytes, &new_offset, &new_capacity))) {
        (void)vk_graph_reclaim_pooled_ranges(idx);
        if (!vk_graph_range_alloc(bytes, &new_offset, &new_capacity)) {
            printf("[VolvoxAI GPU] Vulkan graph arena exhausted (%zu-byte request / %zu-byte arena). Set VOLVOX_VULKAN_MB higher.\n",
                   bytes, io_size);
            return NULL;
        }
    }
    if (staged.owns_range && staged.capacity && !vk_graph_flush_wait()) {
        (void)vk_graph_range_free(new_offset, new_capacity);
        return NULL;
    }
    graph_capacity_generation =
        vk_graph_generation_next(graph_capacity_generation);
    if (staged.owns_range && staged.capacity &&
        !vk_graph_range_free(staged.offset, staged.capacity)) {
        (void)vk_graph_range_free(new_offset, new_capacity);
        return NULL;
    }
    staged.host = host;
    staged.bytes = bytes;
    staged.capacity = new_capacity;
    staged.offset = new_offset;
    staged.shape_generation = is_weight ? 0 : graph_shape_generation;
    staged.capacity_generation = graph_capacity_generation;
    staged.host_dirty = 1;
    staged.device_dirty = 0;
    staged.is_weight = is_weight || staged.is_weight;
    staged.owns_range = 1;
    graph_slots[idx] = staged;
    if (is_new) graph_slot_count++;
    return &graph_slots[idx];
}

static VkTensorSlot* graph_ensure_device(const void* host, size_t bytes, int is_weight) {
    VkTensorSlot* s = graph_get_slot(host, bytes, is_weight);
    if (!s) return NULL;
    if (s->host_dirty) {
        if (!vk_staging_upload(s->offset, host, bytes)) return NULL;
        s->host_dirty = 0;
        s->device_dirty = 0;
    }
    return s;
}

/* qLinearInt8 addresses byte tensors through array<u32>.  Keep the logical
 * host size distinct from the four-byte GPU backing size so a tail byte is
 * never read past host memory and consecutive W8A8 nodes remain device-local. */
static int graph_packed_bytes(size_t logical_bytes, size_t* storage_bytes) {
    if (!storage_bytes || logical_bytes == 0 || logical_bytes > SIZE_MAX - 3u) return 0;
    *storage_bytes = (logical_bytes + 3u) & ~(size_t)3u;
    return 1;
}

static VkTensorSlot* graph_ensure_packed_bytes(const void* host, size_t logical_bytes,
                                                int is_weight) {
    size_t storage_bytes;
    if (!host || !graph_packed_bytes(logical_bytes, &storage_bytes)) return NULL;
    VkTensorSlot* s = graph_get_slot(host, storage_bytes, is_weight);
    if (!s) return NULL;
    if (s->host_dirty) {
        if (!vk_staging_upload(s->offset, host, logical_bytes)) return NULL;
        s->host_dirty = 0;
        s->device_dirty = 0;
    }
    return s;
}

static VkTensorSlot* graph_output_packed_bytes(const void* host, size_t logical_bytes) {
    size_t storage_bytes;
    if (!host || !graph_packed_bytes(logical_bytes, &storage_bytes)) return NULL;
    return graph_get_slot(host, storage_bytes, 0);
}

static VkTensorSlot* graph_output_slot(const void* host, size_t bytes) {
    return graph_get_slot(host, bytes, 0);
}

/* A binding into part of a resident tensor.
 *
 * Row execution asks for a *slice* of an activation, not for a tensor. The slot
 * map is keyed by complete host-span bases, so an interior pointer finds
 * nothing on an exact lookup, and minting a slot for it would give the same
 * bytes two device allocations that then drift. Resolving it as an offset into
 * the containing slot is what CUDA already does (`graph_get_view`), and it is
 * what stands between this backend and a decode row that stays on the device.
 *
 * `offset` is an `io_buffer` byte offset, which is what every binding here
 * already is -- so a window costs no new machinery at dispatch, only at
 * resolution. */
typedef struct {
    VkTensorSlot* slot;
    size_t offset;
    size_t bytes;
} VkGraphWindow;

/*
 * The resident slot whose span contains `[host, host + bytes)`.
 *
 * Exact base first, then the smallest containing span: a tensor and an arena
 * span that both cover the pointer are both legal answers, and the tighter one
 * is the tensor. Returns -1 when nothing contains it, which is the ordinary
 * "this is a new tensor" case and not an error.
 */
static int graph_find_containing_slot(const void* host, size_t bytes, size_t* inner) {
    uintptr_t start = (uintptr_t)host;
    uintptr_t end;
    int best = -1;
    int exact;
    if (!host || !bytes || start > UINTPTR_MAX - bytes) return -1;
    end = start + bytes;
    exact = graph_find_slot(host);
    if (exact >= 0 && graph_slots[exact].bytes >= bytes) {
        if (inner) *inner = 0;
        return exact;
    }
    for (int index = 0; index < graph_slot_count; index++) {
        VkTensorSlot* slot = &graph_slots[index];
        uintptr_t slot_start = (uintptr_t)slot->host;
        uintptr_t slot_end;
        if (!slot->host || !slot->bytes || !slot->owns_range ||
            slot_start > UINTPTR_MAX - slot->bytes) continue;
        if (!slot->is_weight && slot->shape_generation != graph_shape_generation) continue;
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
 * The alignment check is the one real constraint a device row carries.
 * `VkDescriptorBufferInfo.offset` must be a multiple of the device's storage
 * alignment, and a row offset is `row * width * element size` -- so a row is
 * bindable exactly when its *stride* is a multiple of that alignment. Refusing
 * here names the constraint; the caller falls back to the host row path, which
 * is what happens today for every row.
 */
static int graph_window_finish(VkTensorSlot* slot, size_t inner, size_t bytes,
                               VkGraphWindow* out) {
    if (!slot || !out || !bytes || inner > slot->bytes ||
        bytes > slot->bytes - inner) return 0;
    if (graph_alignment && (slot->offset + inner) % graph_alignment) return 0;
    out->slot = slot;
    out->offset = slot->offset + inner;
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
                               VkGraphWindow* out) {
    const void* base = host;
    size_t base_bytes = bytes;
    size_t inner = 0;
    VkTensorSlot* slot;
    if (!host || !bytes || !out) return 0;
    if (!graph_window_base(host, bytes, &base, &base_bytes, &inner)) {
        base = host;
        base_bytes = bytes;
        inner = 0;
    }
    slot = graph_ensure_device(base, base_bytes, is_weight);
    return graph_window_finish(slot, inner, bytes, out);
}

/* The packed spelling. A window's own bytes stay logical: only the containing
 * slot is rounded up to whole words, and the tail it pads belongs to the
 * tensor rather than to any one row. */
static int graph_window_packed_bytes(const void* host, size_t logical_bytes,
                                     int is_weight, VkGraphWindow* out) {
    const void* base = host;
    size_t base_bytes = logical_bytes;
    size_t inner = 0;
    size_t storage_bytes;
    VkTensorSlot* slot;
    if (!host || !out || !graph_packed_bytes(logical_bytes, &storage_bytes)) return 0;
    if (graph_window_base(host, logical_bytes, &base, &base_bytes, &inner)) {
        slot = graph_ensure_packed_bytes(base, base_bytes, is_weight);
        return graph_window_finish(slot, inner, storage_bytes, out);
    }
    slot = graph_ensure_packed_bytes(host, logical_bytes, is_weight);
    return graph_window_finish(slot, 0, storage_bytes, out);
}

/*
 * An output window.
 *
 * A row writes part of a tensor, so the rest has to already be on the device --
 * otherwise the next read of an untouched row sees whatever the arena held.
 * When the containing slot is host-dirty this uploads it whole first and the
 * kernel then overwrites one row, which is the same order the host row path
 * achieves by syncing before it runs.
 */
static int graph_window_output_packed(const void* host, size_t logical_bytes,
                                      VkGraphWindow* out) {
    const void* base = host;
    size_t base_bytes = logical_bytes;
    size_t inner = 0;
    size_t storage_bytes;
    VkTensorSlot* slot;
    if (!host || !out || !graph_packed_bytes(logical_bytes, &storage_bytes)) return 0;
    if (graph_window_base(host, logical_bytes, &base, &base_bytes, &inner)) {
        slot = graph_ensure_packed_bytes(base, base_bytes, 0);
        return graph_window_finish(slot, inner, storage_bytes, out);
    }
    slot = graph_output_packed_bytes(host, logical_bytes);
    return graph_window_finish(slot, 0, storage_bytes, out);
}

static void graph_mark_device(VkTensorSlot* s) {
    if (!s) return;
    s->device_dirty = 1;
    s->host_dirty = 0;
}

static void graph_scratch_begin(void) {
    if (scratch_bump == 0) {
        VulkanContextState* context = vk_context_current();
        scratch_bump = context && context->domain_enforced
            ? context->domain_params_offset
            : ALIGN_UP(io_size - VK_GRAPH_SCRATCH_BYTES, graph_alignment);
    }
    if (scratch_bump > scratch_high_water)
        scratch_high_water = scratch_bump;
}

static size_t graph_scratch_alloc(size_t bytes) {
    size_t aligned;
    if (!bytes || !vk_graph_align_up(scratch_bump, graph_alignment, &aligned) ||
        aligned > io_size || bytes > io_size - aligned) return SIZE_MAX;
    scratch_bump = aligned;
    size_t off = scratch_bump;
    if (!vk_graph_align_up(scratch_bump + bytes, graph_alignment,
                           &scratch_bump)) return SIZE_MAX;
    if (scratch_bump > scratch_high_water)
        scratch_high_water = scratch_bump;
    return off;
}

static size_t graph_scratch_upload(const void* data, size_t bytes) {
    size_t off = graph_scratch_alloc(bytes);
    if (off == SIZE_MAX) return SIZE_MAX;
    if (!vk_staging_upload(off, data, bytes)) return SIZE_MAX;
    return off;
}

/* Zero-filled scratch for an optional binding the shader always declares. */
static size_t graph_scratch_zero(size_t bytes) {
    size_t off = graph_scratch_alloc(bytes);
    if (off == SIZE_MAX) return SIZE_MAX;
    if (!vk_staging_zero(off, bytes)) return SIZE_MAX;
    return off;
}

void vk_graph_reset(void) {
    if (!vk_context_current()) return;
    vk_graph_flush_wait();
    free(graph_shape_signature);
    graph_shape_signature = NULL;
    graph_shape_generation =
        vk_graph_generation_next(graph_shape_generation);
    graph_capacity_generation =
        vk_graph_generation_next(graph_capacity_generation);
    graph_domain_span_count = 0;
    graph_domain_enforced = 0;
    vk_context_current()->domain_qgroupnorm_stats_bytes = 0;
    vk_context_current()->domain_qlayernorm_stats_bytes = 0;
    vk_context_current()->domain_qgroupnorm_stats_offset = 0;
    vk_context_current()->domain_qlayernorm_stats_offset = 0;
    vk_context_current()->domain_params_offset = 0;
    memset(graph_slots, 0,
           VK_GRAPH_MAX_TENSORS * sizeof(*graph_slots));
    graph_slot_count = 0;
    (void)vk_graph_allocator_reset();
    scratch_bump = 0;
    scratch_high_water = 0;
    graph_dispatch_set_cursor = 0;
}

int vk_graph_bind_shape(const char* signature) {
    VulkanContextState* context = vk_context_current();
    char* candidate;
    size_t length;
    uint64_t next_generation;
    if (!context || !vk_is_ready() || !signature || !signature[0] ||
        graph_domain_enforced) return -1;
    if (graph_shape_signature && !strcmp(graph_shape_signature, signature))
        return 0;
    length = strlen(signature);
    if (length > 1024u * 1024u) return -1;
    candidate = (char*)malloc(length + 1u);
    if (!candidate) return -1;
    memcpy(candidate, signature, length + 1u);
    if (!vk_graph_flush_wait()) {
        free(candidate);
        return -1;
    }
    next_generation = vk_graph_generation_next(graph_shape_generation);
    for (int index = 0; index < graph_slot_count; index++) {
        VkTensorSlot* slot = &graph_slots[index];
        if (slot->is_weight) continue;
        slot->host = NULL;
        slot->bytes = 0;
        slot->shape_generation = next_generation;
        slot->host_dirty = 0;
        slot->device_dirty = 0;
        if (!slot->owns_range) {
            slot->offset = 0;
            slot->capacity = 0;
            slot->capacity_generation = 0;
        }
    }
    free(graph_shape_signature);
    graph_shape_signature = candidate;
    graph_shape_generation = next_generation;
    scratch_bump = 0;
    scratch_high_water = 0;
    graph_dispatch_set_cursor = 0;
    return 0;
}

static int vk_graph_domain_spans_match(
        const VolvoxAIEnginePhysicalSpan* spans,
        size_t span_count) {
    if (!spans || span_count != graph_domain_span_count) return 0;
    for (size_t index = 0; index < span_count; index++) {
        int slot_index = graph_find_slot(spans[index].host);
        VkTensorSlot* slot;
        if (slot_index < 0) return 0;
        slot = &graph_slots[slot_index];
        if (!slot->domain_span || slot->is_weight || !slot->owns_range ||
            !slot->capacity ||
            slot->domain_capacity != spans[index].capacity_bytes ||
            slot->bytes > slot->domain_capacity)
            return 0;
    }
    return 1;
}

static int vk_graph_domain_scratch_layout(
        VulkanContextState* context,
        size_t qgroupnorm_stats_bytes,
        size_t qlayernorm_stats_bytes,
        size_t* qgroupnorm_offset,
        size_t* qlayernorm_offset,
        size_t* params_offset) {
    size_t base;
    size_t cursor;
    size_t bootstrap_bytes = 0;
    if (!context || !qgroupnorm_offset || !qlayernorm_offset ||
        !params_offset || io_size < VK_GRAPH_SCRATCH_BYTES ||
        (qgroupnorm_stats_bytes &&
         qgroupnorm_stats_bytes > (size_t)max_storage_buffer_range) ||
        (qlayernorm_stats_bytes &&
         qlayernorm_stats_bytes > (size_t)max_storage_buffer_range) ||
        !vk_graph_align_up(io_size - VK_GRAPH_SCRATCH_BYTES,
                           graph_alignment, &base) ||
        base > io_size)
        return 0;
    if (scratch_high_water) {
        if (scratch_high_water < base || scratch_high_water > io_size)
            return 0;
        bootstrap_bytes = scratch_high_water - base;
    }
    cursor = base;
    *qgroupnorm_offset = 0;
    *qlayernorm_offset = 0;
    if (qgroupnorm_stats_bytes) {
        *qgroupnorm_offset = cursor;
        if (qgroupnorm_stats_bytes > io_size - cursor ||
            !vk_graph_align_up(cursor + qgroupnorm_stats_bytes,
                               graph_alignment, &cursor) ||
            cursor > io_size)
            return 0;
    }
    if (qlayernorm_stats_bytes) {
        *qlayernorm_offset = cursor;
        if (qlayernorm_stats_bytes > io_size - cursor ||
            !vk_graph_align_up(cursor + qlayernorm_stats_bytes,
                               graph_alignment, &cursor) ||
            cursor > io_size)
            return 0;
    }
    /* Bootstrap usage is a conservative bound for all shape-invariant
     * params and optional-bias scratch. It also contains the minimum-shape
     * norm stats, so adding it here deliberately overcounts rather than
     * permitting a maximum-shape forward to exhaust the fixed 1 MiB tail. */
    if (bootstrap_bytes > io_size - cursor) return 0;
    *params_offset = cursor;
    return 1;
}

int vk_graph_bind_shape_domain(
        const char* signature,
        const VolvoxAIEnginePhysicalSpan* spans,
        size_t span_count,
        size_t qgroupnorm_stats_bytes,
        size_t qlayernorm_stats_bytes) {
    VulkanContextState* context = vk_context_current();
    char* candidate_signature = NULL;
    VkGraphFreeRange* free_snapshot = NULL;
    size_t* candidate_offsets = NULL;
    size_t* candidate_capacities = NULL;
    size_t signature_length;
    size_t old_bump = 0;
    int old_free_count = 0;
    int weight_count = 0;
    int allocator_mutated = 0;
    int result = -1;
    size_t qgroupnorm_stats_offset = 0;
    size_t qlayernorm_stats_offset = 0;
    size_t params_offset = 0;
    uint64_t next_shape_generation;
    uint64_t next_capacity_generation;
    if (!context || !vk_is_ready() || !signature || !signature[0] ||
        !spans || !span_count || span_count > VK_GRAPH_MAX_TENSORS)
        return -1;
#if VOLVOXAI_ENABLE_TRAINING
    if (training_active) return -1;
#endif
    signature_length = strlen(signature);
    if (signature_length > 1024u * 1024u) return -1;
    for (size_t index = 0; index < span_count; index++) {
        uintptr_t start = (uintptr_t)spans[index].host;
        size_t capacity = spans[index].capacity_bytes;
        if (!start || !capacity || start > UINTPTR_MAX - capacity ||
            !max_storage_buffer_range ||
            capacity > (size_t)max_storage_buffer_range ||
            (index && (uintptr_t)spans[index - 1u].host +
                spans[index - 1u].capacity_bytes > start))
            return -1;
    }
    if (graph_domain_enforced) {
        if (!vk_graph_domain_spans_match(spans, span_count) ||
            qgroupnorm_stats_bytes !=
                context->domain_qgroupnorm_stats_bytes ||
            qlayernorm_stats_bytes !=
                context->domain_qlayernorm_stats_bytes)
            return -1;
        if (graph_shape_signature &&
            !strcmp(graph_shape_signature, signature)) {
            for (int index = 0; index < graph_slot_count; index++) {
                VkTensorSlot* slot = &graph_slots[index];
                if (!slot->domain_span) continue;
                slot->bytes = 0;
                slot->host_dirty = 0;
                slot->device_dirty = 0;
            }
            scratch_bump = 0;
            scratch_high_water = 0;
            graph_dispatch_set_cursor = 0;
            return 0;
        }
        candidate_signature = (char*)malloc(signature_length + 1u);
        if (!candidate_signature) return -2;
        memcpy(candidate_signature, signature, signature_length + 1u);
        if (!vk_graph_flush_wait()) goto done;
        next_shape_generation =
            vk_graph_generation_next(graph_shape_generation);
        for (int index = 0; index < graph_slot_count; index++) {
            VkTensorSlot* slot = &graph_slots[index];
            if (!slot->domain_span) continue;
            slot->shape_generation = next_shape_generation;
            slot->bytes = 0;
            slot->host_dirty = 0;
            slot->device_dirty = 0;
        }
        free(graph_shape_signature);
        graph_shape_signature = candidate_signature;
        candidate_signature = NULL;
        graph_shape_generation = next_shape_generation;
        scratch_bump = 0;
        scratch_high_water = 0;
        graph_dispatch_set_cursor = 0;
        result = 0;
        goto done;
    }

    if (!vk_graph_domain_scratch_layout(
            context, qgroupnorm_stats_bytes, qlayernorm_stats_bytes,
            &qgroupnorm_stats_offset, &qlayernorm_stats_offset,
            &params_offset))
        return -1;

    for (int index = 0; index < graph_slot_count; index++) {
        VkTensorSlot* slot = &graph_slots[index];
        if (!slot->is_weight) continue;
        uintptr_t weight_start = (uintptr_t)slot->host;
        if (!weight_start || !slot->owns_range || !slot->capacity ||
            !slot->bytes || slot->bytes > slot->capacity ||
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
    if (span_count > (size_t)(VK_GRAPH_MAX_TENSORS - weight_count))
        return -1;
    candidate_signature = (char*)malloc(signature_length + 1u);
    candidate_offsets = (size_t*)calloc(span_count, sizeof(*candidate_offsets));
    candidate_capacities =
        (size_t*)calloc(span_count, sizeof(*candidate_capacities));
    free_snapshot = (VkGraphFreeRange*)malloc(
        VK_GRAPH_MAX_FREE_RANGES * sizeof(*free_snapshot));
    if (!candidate_signature || !candidate_offsets ||
        !candidate_capacities || !free_snapshot) {
        result = -2;
        goto done;
    }
    memcpy(candidate_signature, signature, signature_length + 1u);
    if (!vk_graph_flush_wait()) goto done;
    memcpy(free_snapshot, graph_free_ranges,
           VK_GRAPH_MAX_FREE_RANGES * sizeof(*free_snapshot));
    old_free_count = graph_free_range_count;
    old_bump = graph_bump;
    allocator_mutated = 1;

    /* The arena allocation itself survives rollback. Returning old transient
     * ranges changes metadata only, so failed candidate placement can restore
     * the exact allocator snapshot and leave all published slots usable. */
    for (int index = 0; index < graph_slot_count; index++) {
        VkTensorSlot* slot = &graph_slots[index];
        if (slot->is_weight || !slot->owns_range || !slot->capacity) continue;
        if (!vk_graph_range_free(slot->offset, slot->capacity)) goto rollback;
    }
    for (size_t index = 0; index < span_count; index++) {
        if (!vk_graph_range_alloc(spans[index].capacity_bytes,
                                  &candidate_offsets[index],
                                  &candidate_capacities[index])) {
            result = -2;
            goto rollback;
        }
    }

    next_shape_generation =
        vk_graph_generation_next(graph_shape_generation);
    next_capacity_generation =
        vk_graph_generation_next(graph_capacity_generation);
    {
        int kept = 0;
        int old_slot_count = graph_slot_count;
        for (int index = 0; index < old_slot_count; index++) {
            if (!graph_slots[index].is_weight) continue;
            if (kept != index) graph_slots[kept] = graph_slots[index];
            kept++;
        }
        if (kept < old_slot_count) {
            memset(&graph_slots[kept], 0,
                   (size_t)(old_slot_count - kept) * sizeof(graph_slots[0]));
        }
        graph_slot_count = kept;
    }
    for (size_t index = 0; index < span_count; index++) {
        VkTensorSlot* slot = &graph_slots[graph_slot_count++];
        memset(slot, 0, sizeof(*slot));
        slot->host = spans[index].host;
        slot->capacity = candidate_capacities[index];
        slot->domain_capacity = spans[index].capacity_bytes;
        slot->offset = candidate_offsets[index];
        slot->shape_generation = next_shape_generation;
        slot->capacity_generation = next_capacity_generation;
        slot->owns_range = 1;
        slot->domain_span = 1;
    }
    free(graph_shape_signature);
    graph_shape_signature = candidate_signature;
    candidate_signature = NULL;
    graph_shape_generation = next_shape_generation;
    graph_capacity_generation = next_capacity_generation;
    graph_domain_span_count = span_count;
    context->domain_qgroupnorm_stats_bytes = qgroupnorm_stats_bytes;
    context->domain_qlayernorm_stats_bytes = qlayernorm_stats_bytes;
    context->domain_qgroupnorm_stats_offset = qgroupnorm_stats_offset;
    context->domain_qlayernorm_stats_offset = qlayernorm_stats_offset;
    context->domain_params_offset = params_offset;
    graph_domain_enforced = 1;
    scratch_bump = 0;
    scratch_high_water = 0;
    graph_dispatch_set_cursor = 0;
    allocator_mutated = 0;
    result = 0;
    goto done;

rollback:
    memcpy(graph_free_ranges, free_snapshot,
           VK_GRAPH_MAX_FREE_RANGES * sizeof(*free_snapshot));
    graph_free_range_count = old_free_count;
    graph_bump = old_bump;
    allocator_mutated = 0;
done:
    if (allocator_mutated) {
        memcpy(graph_free_ranges, free_snapshot,
               VK_GRAPH_MAX_FREE_RANGES * sizeof(*free_snapshot));
        graph_free_range_count = old_free_count;
        graph_bump = old_bump;
    }
    free(free_snapshot);
    free(candidate_capacities);
    free(candidate_offsets);
    free(candidate_signature);
    return result;
}

#include "vulkan_trace.inc"

void vk_graph_begin_forward(void) {
    if (!vk_context_current()) return;
    vk_graph_flush_wait();
    scratch_bump = 0;
    scratch_high_water = 0;
    graph_dispatch_set_cursor = 0;
    VulkanContextState* context = vk_context_current();
    context->qlinear_dot_dispatches = 0u;
    context->qlinear_tiled_dispatches = 0u;
    context->qlinear_scalar_dispatches = 0u;
    context->qbatch_dot_dispatches = 0u;
    context->qbatch_scalar_dispatches = 0u;
    context->qconv_dot_tiled_dispatches = 0u;
    context->qconv_tiled_dispatches = 0u;
    context->qconv_scalar_dispatches = 0u;
    context->conv_out16_dispatches = 0u;
    context->conv_scalar_dispatches = 0u;
    context->staging_upload_count = 0u;
    context->staging_upload_bytes = 0u;
    context->staging_download_count = 0u;
    context->staging_download_bytes = 0u;
    context->staging_submit_count = 0u;
    context->prepared_pipeline_creates_at_forward =
        context->prepared_pipeline_creates;
    if (vx_engine_state_current()->profiling) vk_trace_begin("Vulkan forward");
}

int vk_graph_end_forward(void) {
    VulkanContextState* context = vk_context_current();
    if (!context) return -1;
    if (context->trace_pass) vk_trace_end_record();
    int complete = vk_graph_flush_wait();
    if (context->trace_pass) vk_trace_finish(complete);
    return complete ? 0 : -1;
}

static void vk_graph_append_counter(char* output, size_t output_capacity,
                                    size_t* offset, const char* name,
                                    uint64_t value) {
    int written;
    if (!output || !offset || !name || !value || *offset >= output_capacity)
        return;
    written = snprintf(output + *offset, output_capacity - *offset,
                       ";%s=%" PRIu64, name, value);
    if (written <= 0) return;
    if ((size_t)written >= output_capacity - *offset)
        *offset = output_capacity;
    else
        *offset += (size_t)written;
}

int vk_graph_append_dynamic_telemetry(char* output,
                                      size_t output_capacity) {
    VulkanContextState* context = vk_context_current();
    size_t offset = 0u;
    if (!output || !output_capacity || !context) return -1;
    output[0] = '\0';
    vk_graph_append_counter(output, output_capacity, &offset, "vk_c16",
                            context->conv_out16_dispatches);
    vk_graph_append_counter(output, output_capacity, &offset, "vk_cs",
                            context->conv_scalar_dispatches);
    vk_graph_append_counter(output, output_capacity, &offset, "vk_qld",
                            context->qlinear_dot_dispatches);
    vk_graph_append_counter(output, output_capacity, &offset, "vk_qlt",
                            context->qlinear_tiled_dispatches);
    vk_graph_append_counter(output, output_capacity, &offset, "vk_qls",
                            context->qlinear_scalar_dispatches);
    vk_graph_append_counter(output, output_capacity, &offset, "vk_qbd",
                            context->qbatch_dot_dispatches);
    vk_graph_append_counter(output, output_capacity, &offset, "vk_qbs",
                            context->qbatch_scalar_dispatches);
    vk_graph_append_counter(output, output_capacity, &offset, "vk_qcd",
                            context->qconv_dot_tiled_dispatches);
    vk_graph_append_counter(output, output_capacity, &offset, "vk_qct",
                            context->qconv_tiled_dispatches);
    vk_graph_append_counter(output, output_capacity, &offset, "vk_qcs",
                            context->qconv_scalar_dispatches);
    vk_graph_append_counter(
        output, output_capacity, &offset, "vk_pc",
        context->prepared_pipeline_creates -
            context->prepared_pipeline_creates_at_forward);
    return 0;
}

int vk_graph_execution_evidence(char* output, size_t output_capacity) {
    VulkanContextState* context = vk_context_current();
    int written;
    if (!output || !output_capacity || !context || !vk_is_ready() ||
        !(context->compute_memory_flags &
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
        !(context->staging_memory_flags &
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
        return -1;
    written = snprintf(
        output, output_capacity,
        ";vk_mem=device-local;vk_arena=%zu;vk_stage=%zu;"
        "vk_stage_coherent=%u;vk_up=%" PRIu64 ";vk_down=%" PRIu64,
        context->arena_allocation_size,
        context->staging_allocation_size,
        !!(context->staging_memory_flags &
           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        context->staging_upload_count,
        context->staging_download_count);
    if (written < 0 || (size_t)written >= output_capacity) {
        output[0] = '\0';
        return -1;
    }
    return 0;
}

void vk_graph_mark_host(const void* host, size_t bytes, int is_weight) {
    int index = graph_find_slot(host);
    if (index >= 0 && !graph_slots[index].owns_range) {
        graph_slots[index].offset = 0;
        graph_slots[index].capacity = 0;
        graph_slots[index].capacity_generation = 0;
    }
    VkTensorSlot* s = graph_get_slot(host, bytes, is_weight);
    if (!s) return;
    s->host_dirty = 1;
    s->device_dirty = 0;
}

int vk_graph_sync_host(const void* host, size_t bytes, int is_weight) {
    (void)is_weight;
    if (!vk_is_ready() || !host || bytes == 0) return 0;
    int idx = graph_find_slot(host);
    /* No slot means this backend has never made the tensor device-resident,
       so the host allocation is already authoritative.  Treat synchronization
       as an idempotent ensure-current operation: CPU fallback must not fail
       merely because an immutable weight was never uploaded to Vulkan. */
    if (idx < 0) return 1;
    VkTensorSlot* s = &graph_slots[idx];
    if (bytes > s->bytes) return 0;
    if (s->device_dirty) {
        if (!vk_staging_download(s->offset, (void*)host, bytes)) return 0;
        s->device_dirty = 0;
        s->host_dirty = 0;
    }
    return 1;
}

static VkPreparedKernel* vk_prepare_kernel(VkKernel* k) {
    VulkanContextState* context = vk_context_current();
    VkPreparedKernel prepared = {0};
    if (!vk_is_ready() || !k || !context) return NULL;
    for (int i = 0; i < context->prepared_kernel_count; i++) {
        if (context->prepared_kernels[i].definition == k &&
            context->prepared_kernels[i].ready)
            return &context->prepared_kernels[i];
    }
    if (context->prepared_kernel_count >= VK_PREPARED_KERNEL_MAX) return NULL;
    if (k->binding_count <= 0 || k->binding_count > 16) return 0;

    pthread_mutex_lock(&g_vulkan_device.mutex);
    prepared.definition = k;

    VkDescriptorSetLayoutBinding bindings[16];
    memset(bindings, 0, sizeof(bindings));
    for (int i = 0; i < k->binding_count; i++) {
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = (i == k->uniform_binding) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info = {0};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = (uint32_t)k->binding_count;
    layout_info.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layout_info, NULL,
                                    &prepared.prepared_desc_layout) != VK_SUCCESS) goto fail;

    VkPipelineLayoutCreateInfo pl_info = {0};
    pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_info.setLayoutCount = 1;
    pl_info.pSetLayouts = &prepared.prepared_desc_layout;
    if (vkCreatePipelineLayout(device, &pl_info, NULL,
                               &prepared.prepared_pipeline_layout) != VK_SUCCESS) goto fail;

    size_t spv_size = 0;
    uint32_t* spv = load_spv(k->path, &spv_size);
    if (!spv) goto fail;
    VkShaderModuleCreateInfo shader_info = {0};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = spv_size;
    shader_info.pCode = spv;
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkResult sm_res = vkCreateShaderModule(device, &shader_info, NULL, &shader_module);
    free(spv);
    if (sm_res != VK_SUCCESS) goto fail;

    VkComputePipelineCreateInfo pipe_info = {0};
    pipe_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipe_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipe_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipe_info.stage.module = shader_module;
    pipe_info.stage.pName = vk_kernel_entry_point(k);
    pipe_info.layout = prepared.prepared_pipeline_layout;
    VkResult pipeline_result = vkCreateComputePipelines(
        device, pipeline_cache, 1, &pipe_info, NULL, &prepared.pipeline);
    if (vkDestroyShaderModule) vkDestroyShaderModule(device, shader_module, NULL);
    if (pipeline_result != VK_SUCCESS) goto fail;
    g_vulkan_device.prepared_pipeline_creates++;
    context->prepared_pipeline_creates++;

    prepared.ready = 1;
    context->prepared_kernels[context->prepared_kernel_count] = prepared;
    pthread_mutex_unlock(&g_vulkan_device.mutex);
    return &context->prepared_kernels[context->prepared_kernel_count++];

fail:
    vk_destroy_prepared_kernel_locked(&prepared);
    pthread_mutex_unlock(&g_vulkan_device.mutex);
    return NULL;
}

typedef struct {
    size_t offset;
    size_t bytes;
} VkGraphBinding;

static VkDescriptorSet vk_graph_dispatch_set(VkKernel* k,
                                             const VkPreparedKernel* prepared) {
    if (!k || !prepared || !prepared->prepared_desc_layout) return VK_NULL_HANDLE;
    if (graph_dispatch_set_cursor >= VK_GRAPH_MAX_DISPATCH_SETS) return VK_NULL_HANDLE;
    int idx = graph_dispatch_set_cursor++;
    VkGraphDispatchSet* s = &graph_dispatch_sets[idx];
    if (s->descriptor_set != VK_NULL_HANDLE && s->kernel == k) return s->descriptor_set;

    VkDescriptorSetAllocateInfo set_info = {0};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = desc_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &prepared->prepared_desc_layout;
    if (vkAllocateDescriptorSets(device, &set_info, &s->descriptor_set) != VK_SUCCESS) return VK_NULL_HANDLE;
    s->kernel = k;
    if (idx >= graph_dispatch_set_count) graph_dispatch_set_count = idx + 1;
    return s->descriptor_set;
}

static int vk_graph_begin_recording(void) {
    if (graph_cmd_recording) return 1;
    if (graph_cmd_pending && !vk_graph_flush_wait()) return 0;
    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd_buf, &begin_info) != VK_SUCCESS) return 0;
    graph_cmd_recording = 1;
    return 1;
}

static int vk_graph_submit_recording(void) {
    if (!graph_cmd_recording) return 1;
    if (vkEndCommandBuffer(cmd_buf) != VK_SUCCESS) {
        graph_cmd_recording = 0;
        return 0;
    }
    VkSubmitInfo submit_info = {0};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buf;
    pthread_mutex_lock(&g_vulkan_device.mutex);
    VkResult reset_result = vkResetFences(device, 1, &compute_fence);
    uint64_t trace_start = vk_trace_host_start();
    VkResult submit_result = reset_result == VK_SUCCESS
        ? vkQueueSubmit(compute_queue, 1, &submit_info, compute_fence)
        : reset_result;
    vk_trace_host_end(trace_start, "vkQueueSubmit", VX_TRACE_ACTIVITY_SUBMIT, 0, 0, 0);
    pthread_mutex_unlock(&g_vulkan_device.mutex);
    if (submit_result != VK_SUCCESS) {
        graph_cmd_recording = 0;
        return 0;
    }
    graph_cmd_recording = 0;
    graph_cmd_pending = 1;
    vk_context_current()->staging_submit_count++;
    return 1;
}

static int vk_graph_flush_wait(void) {
    if (!vk_is_ready()) return 1;
    if (graph_cmd_recording && !vk_graph_submit_recording()) return 0;
    if (graph_cmd_pending) {
        uint64_t trace_start = vk_trace_host_start();
        VkResult waited = vkWaitForFences(device, 1, &compute_fence, VK_TRUE, UINT64_MAX);
        vk_trace_host_end(trace_start, "vkWaitForFences", VX_TRACE_ACTIVITY_WAIT, 0, 0, 0);
        if (waited != VK_SUCCESS) return 0;
        graph_cmd_pending = 0;
    }
    /* Every staging range recorded before this point is now retired.  Reuse
     * starts only after the fence, never merely after queue submission. */
    vk_stage_cursor = 0u;
    return 1;
}

static int vk_staging_aligned_copy_span(size_t device_offset, size_t bytes,
                                        size_t* copy_span) {
    if (!copy_span || !bytes || (device_offset & 3u) != 0u ||
        !vk_graph_align_up(bytes, 4u, copy_span) ||
        device_offset > io_size || *copy_span > io_size - device_offset)
        return 0;
    return 1;
}

static int vk_staging_mapped_range(size_t offset, size_t bytes,
                                   VkMappedMemoryRange* range) {
    VulkanContextState* context = vk_context_current();
    size_t atom;
    size_t end;
    size_t aligned_begin;
    size_t aligned_end;
    if (!context || !range || !bytes ||
        non_coherent_atom_size > (VkDeviceSize)SIZE_MAX)
        return 0;
    atom = (size_t)non_coherent_atom_size;
    if (!atom || offset > context->staging_allocation_size ||
        bytes > context->staging_allocation_size - offset)
        return 0;
    end = offset + bytes;
    aligned_begin = offset - offset % atom;
    if (end == context->staging_allocation_size) {
        aligned_end = end;
    } else {
        if (!vk_graph_align_up(end, atom, &aligned_end)) return 0;
        if (aligned_end > context->staging_allocation_size)
            aligned_end = context->staging_allocation_size;
    }
    if (aligned_end <= aligned_begin) return 0;
    memset(range, 0, sizeof(*range));
    range->sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range->memory = vk_stage_memory;
    range->offset = (VkDeviceSize)aligned_begin;
    range->size = (VkDeviceSize)(aligned_end - aligned_begin);
    return 1;
}

static int vk_staging_flush_host_write(size_t offset, size_t bytes) {
    VulkanContextState* context = vk_context_current();
    VkMappedMemoryRange range;
    if (!context ||
        (context->staging_memory_flags &
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        return context != NULL;
    if (!vk_staging_mapped_range(offset, bytes, &range)) return 0;
    return vkFlushMappedMemoryRanges(device, 1u, &range) == VK_SUCCESS;
}

static int vk_staging_invalidate_host_read(size_t offset, size_t bytes) {
    VulkanContextState* context = vk_context_current();
    VkMappedMemoryRange range;
    if (!context ||
        (context->staging_memory_flags &
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        return context != NULL;
    if (!vk_staging_mapped_range(offset, bytes, &range)) return 0;
    return vkInvalidateMappedMemoryRanges(device, 1u, &range) == VK_SUCCESS;
}

static int vk_staging_reserve(size_t bytes, size_t* offset) {
    size_t reservation_alignment;
    size_t reservation_bytes;
    size_t aligned_cursor;
    if (!offset || !bytes || (bytes & 3u) != 0u || bytes > vk_stage_size)
        return 0;
    if (non_coherent_atom_size > (VkDeviceSize)SIZE_MAX) return 0;
    reservation_alignment = (size_t)non_coherent_atom_size;
    if (reservation_alignment < 4u) reservation_alignment = 4u;
    if (!vk_graph_align_up(bytes, reservation_alignment, &reservation_bytes) ||
        reservation_bytes > vk_stage_size ||
        !vk_graph_align_up(vk_stage_cursor, reservation_alignment,
                           &aligned_cursor))
        return 0;
    if (aligned_cursor > vk_stage_size ||
        reservation_bytes > vk_stage_size - aligned_cursor) {
        if (!vk_graph_flush_wait()) return 0;
        aligned_cursor = 0u;
    }
    if (reservation_bytes > vk_stage_size - aligned_cursor) return 0;
    *offset = aligned_cursor;
    vk_stage_cursor = aligned_cursor + reservation_bytes;
    return 1;
}

static void vk_staging_barrier_before_device_write(size_t offset,
                                                    size_t bytes) {
    VkBufferMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask =
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = io_buffer;
    barrier.offset = (VkDeviceSize)offset;
    barrier.size = (VkDeviceSize)bytes;
    vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0u,
                         0u, NULL, 1u, &barrier, 0u, NULL);
}

static void vk_staging_barrier_after_device_write(size_t offset,
                                                   size_t bytes) {
    VkBufferMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = io_buffer;
    barrier.offset = (VkDeviceSize)offset;
    barrier.size = (VkDeviceSize)bytes;
    vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u,
                         0u, NULL, 1u, &barrier, 0u, NULL);
}

static void vk_staging_barrier_before_device_read(size_t offset,
                                                   size_t bytes) {
    VkBufferMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = io_buffer;
    barrier.offset = (VkDeviceSize)offset;
    barrier.size = (VkDeviceSize)bytes;
    vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0u,
                         0u, NULL, 1u, &barrier, 0u, NULL);
}

static int vk_staging_upload(size_t device_offset, const void* source,
                             size_t bytes) {
    VulkanContextState* context = vk_context_current();
    const unsigned char* input = (const unsigned char*)source;
    size_t copy_span;
    size_t consumed = 0u;
    if (!context || !source ||
        !vk_staging_aligned_copy_span(device_offset, bytes, &copy_span))
        return 0;
    while (consumed < bytes) {
        size_t remaining = bytes - consumed;
        size_t logical = remaining > vk_stage_size ? vk_stage_size : remaining;
        size_t copied;
        size_t stage_offset;
        VkBufferCopy copy;
        if (!vk_graph_align_up(logical, 4u, &copied) || copied > vk_stage_size ||
            !vk_graph_begin_recording() ||
            !vk_staging_reserve(copied, &stage_offset) ||
            !vk_graph_begin_recording())
            return 0;
        memcpy((unsigned char*)vk_stage_mapped + stage_offset,
               input + consumed, logical);
        if (copied > logical)
            memset((unsigned char*)vk_stage_mapped + stage_offset + logical,
                   0, copied - logical);
        if (!vk_staging_flush_host_write(stage_offset, copied)) return 0;
        vk_staging_barrier_before_device_write(
            device_offset + consumed, copied);
        copy.srcOffset = (VkDeviceSize)stage_offset;
        copy.dstOffset = (VkDeviceSize)(device_offset + consumed);
        copy.size = (VkDeviceSize)copied;
        vk_trace_copy(cmd_buf, vk_stage_buffer, io_buffer, &copy);
        vk_staging_barrier_after_device_write(
            device_offset + consumed, copied);
        consumed += logical;
    }
    context->staging_upload_count++;
    context->staging_upload_bytes += (uint64_t)bytes;
    (void)copy_span;
    return 1;
}

static int vk_staging_zero(size_t device_offset, size_t bytes) {
    VulkanContextState* context = vk_context_current();
    size_t copy_span;
    size_t consumed = 0u;
    if (!context ||
        !vk_staging_aligned_copy_span(device_offset, bytes, &copy_span))
        return 0;
    while (consumed < bytes) {
        size_t remaining = bytes - consumed;
        size_t logical = remaining > vk_stage_size ? vk_stage_size : remaining;
        size_t copied;
        size_t stage_offset;
        VkBufferCopy copy;
        if (!vk_graph_align_up(logical, 4u, &copied) || copied > vk_stage_size ||
            !vk_graph_begin_recording() ||
            !vk_staging_reserve(copied, &stage_offset) ||
            !vk_graph_begin_recording())
            return 0;
        memset((unsigned char*)vk_stage_mapped + stage_offset, 0, copied);
        if (!vk_staging_flush_host_write(stage_offset, copied)) return 0;
        vk_staging_barrier_before_device_write(
            device_offset + consumed, copied);
        copy.srcOffset = (VkDeviceSize)stage_offset;
        copy.dstOffset = (VkDeviceSize)(device_offset + consumed);
        copy.size = (VkDeviceSize)copied;
        vk_trace_copy(cmd_buf, vk_stage_buffer, io_buffer, &copy);
        vk_staging_barrier_after_device_write(
            device_offset + consumed, copied);
        consumed += logical;
    }
    context->staging_upload_count++;
    context->staging_upload_bytes += (uint64_t)bytes;
    (void)copy_span;
    return 1;
}

static int vk_staging_download(size_t device_offset, void* destination,
                               size_t bytes) {
    VulkanContextState* context = vk_context_current();
    unsigned char* output = (unsigned char*)destination;
    size_t copy_span;
    size_t consumed = 0u;
    if (!context || !destination ||
        !vk_staging_aligned_copy_span(device_offset, bytes, &copy_span))
        return 0;
    while (consumed < bytes) {
        size_t remaining = bytes - consumed;
        size_t logical = remaining > vk_stage_size ? vk_stage_size : remaining;
        size_t copied;
        size_t stage_offset;
        VkBufferCopy copy;
        if (!vk_graph_align_up(logical, 4u, &copied) || copied > vk_stage_size ||
            !vk_graph_begin_recording() ||
            !vk_staging_reserve(copied, &stage_offset) ||
            !vk_graph_begin_recording())
            return 0;
        vk_staging_barrier_before_device_read(
            device_offset + consumed, copied);
        copy.srcOffset = (VkDeviceSize)(device_offset + consumed);
        copy.dstOffset = (VkDeviceSize)stage_offset;
        copy.size = (VkDeviceSize)copied;
        vk_trace_copy(cmd_buf, io_buffer, vk_stage_buffer, &copy);
        if (!vk_graph_flush_wait() ||
            !vk_staging_invalidate_host_read(stage_offset, copied))
            return 0;
        memcpy(output + consumed,
               (unsigned char*)vk_stage_mapped + stage_offset, logical);
        consumed += logical;
    }
    context->staging_download_count++;
    context->staging_download_bytes += (uint64_t)bytes;
    (void)copy_span;
    return 1;
}


static int vk_dispatch_dimensions_valid(uint32_t gx, uint32_t gy, uint32_t gz) {
    return gx > 0 && gy > 0 && gz > 0 &&
        gx <= max_compute_workgroups[0] &&
        gy <= max_compute_workgroups[1] &&
        gz <= max_compute_workgroups[2];
}

static int vk_workgroup_shape_supported(
        uint32_t x, uint32_t y, uint32_t z) {
    uint64_t invocations = (uint64_t)x * (uint64_t)y * (uint64_t)z;
    return x > 0u && y > 0u && z > 0u &&
        x <= g_vulkan_device.max_workgroup_size[0] &&
        y <= g_vulkan_device.max_workgroup_size[1] &&
        z <= g_vulkan_device.max_workgroup_size[2] &&
        invocations <= g_vulkan_device.max_workgroup_invocations;
}

static int vk_dispatch_kernel(VkKernel* k, const VkGraphBinding* binds,
                              uint32_t gx, uint32_t gy, uint32_t gz) {
    VkPreparedKernel* prepared;
    if (!vk_dispatch_dimensions_valid(gx, gy, gz) || !k || !binds ||
        k->binding_count <= 0 || k->binding_count > 16) return 0;
    for (int i = 0; i < k->binding_count; i++) {
        VkDeviceSize range_limit = i == k->uniform_binding
            ? max_uniform_buffer_range : max_storage_buffer_range;
        if (!binds[i].bytes || binds[i].offset > io_size ||
            binds[i].bytes > io_size - binds[i].offset ||
            (VkDeviceSize)binds[i].bytes > range_limit) return 0;
    }
    prepared = vk_prepare_kernel(k);
    if (!prepared) return 0;
    VkDescriptorSet dispatch_set = vk_graph_dispatch_set(k, prepared);
    if (dispatch_set == VK_NULL_HANDLE) return 0;

    VkDescriptorBufferInfo infos[16];
    VkWriteDescriptorSet writes[16];
    memset(infos, 0, sizeof(infos));
    memset(writes, 0, sizeof(writes));
    for (int i = 0; i < k->binding_count; i++) {
        infos[i].buffer = io_buffer;
        infos[i].offset = binds[i].offset;
        infos[i].range = binds[i].bytes;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dispatch_set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorType = (i == k->uniform_binding) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device, (uint32_t)k->binding_count, writes, 0, NULL);

    if (!vk_graph_begin_recording()) return 0;
    vk_context_current()->bind_pipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                      prepared->pipeline);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            prepared->prepared_pipeline_layout, 0, 1,
                            &dispatch_set, 0, NULL);
    vk_context_current()->dispatch(cmd_buf, gx, gy, gz);
    if (vkCmdPipelineBarrier) {
        VkMemoryBarrier barrier = {0};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd_buf,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &barrier, 0, NULL, 0, NULL);
    }
    return 1;
}

void vk_graph_retain_weight(const void* host, size_t bytes) {
    if (!vk_context_current() || !host || !bytes) return;
    int index = graph_find_slot(host);
    if (index < 0) return;
    VkTensorSlot* slot = &graph_slots[index];
    if (slot->domain_span || !slot->owns_range || !slot->capacity ||
        !slot->bytes || bytes > slot->bytes || bytes > slot->capacity)
        return;
    slot->is_weight = 1;
    slot->shape_generation = 0;
}

void vk_graph_demote_weight(const void* host, size_t bytes) {
    int index;
    VkTensorSlot* slot;
    if (!vk_context_current() || !host || !bytes) return;
    index = graph_find_slot(host);
    if (index < 0) return;
    slot = &graph_slots[index];
    if (slot->domain_span || !slot->bytes || bytes > slot->bytes)
        return;
    slot->is_weight = 0;
    slot->shape_generation = graph_shape_generation;
}

#if VOLVOXAI_ENABLE_TRAINING
/* -------------------------------------------------------------------------
 * Lazy training shader dispatcher
 * -------------------------------------------------------------------------
 *
 * Native autograd uses the same WGSL sources as browser WebGPU.  Naga keeps
 * each WGSL compute entry point in the SPIR-V module, so a pipeline is keyed
 * by (shader, entry point).  The table is deliberately closed: accepting an
 * arbitrary path here would make binding layouts uncheckable and turn model
 * data into a file-system path.
 */

typedef struct {
    const char* shader;
    const char* entry;
    int binding_count;
    int params_binding;
    int uniform_binding;
    uint32_t read_write_mask;
} VkTrainingSpec;

#define VK_TRAIN_SPEC(shader, entry, count, params, uniform, rw_mask) \
    {shader, entry, count, params, uniform, rw_mask}

static const VkTrainingSpec training_specs[] = {
    VK_TRAIN_SPEC("activationBackward", "main", 5, 4, 4, 0x008u),
    VK_TRAIN_SPEC("basicBackward", "a_main", 6, 5, -1, 0x018u),
    VK_TRAIN_SPEC("basicBackward", "b_main", 6, 5, -1, 0x018u),
    VK_TRAIN_SPEC("batchNorm2DBackward", "input_main", 9, 8, 8, 0x0e0u),
    VK_TRAIN_SPEC("batchNorm2DBackward", "param_main", 9, 8, 8, 0x0e0u),
    VK_TRAIN_SPEC("groupNormBackward", "input_main", 7, 6, 6, 0x038u),
    VK_TRAIN_SPEC("groupNormBackward", "param_main", 7, 6, 6, 0x038u),
    VK_TRAIN_SPEC("concatBackward", "main", 3, 2, 2, 0x002u),
    VK_TRAIN_SPEC("conv2DBackward", "input_main", 8, 7, 7, 0x070u),
    VK_TRAIN_SPEC("conv2DBackward", "weight_main", 8, 7, 7, 0x070u),
    VK_TRAIN_SPEC("conv2DBackward", "bias_main", 8, 7, 7, 0x070u),
    VK_TRAIN_SPEC("copyBackward", "main", 3, 2, 2, 0x002u),
    VK_TRAIN_SPEC("dropoutBackward", "main", 3, 2, 2, 0x002u),
    VK_TRAIN_SPEC("reduceBackward", "main", 3, 2, 2, 0x002u),
    VK_TRAIN_SPEC("crossSdpaBackward", "q_main", 9, 8, 8, 0x0e0u),
    VK_TRAIN_SPEC("crossSdpaBackward", "k_main", 9, 8, 8, 0x0e0u),
    VK_TRAIN_SPEC("crossSdpaBackward", "v_main", 9, 8, 8, 0x0e0u),
    VK_TRAIN_SPEC("embeddingBackward", "main", 4, 3, 3, 0x004u),
    VK_TRAIN_SPEC("layerNormBackward", "input_main", 7, 6, 6, 0x038u),
    VK_TRAIN_SPEC("layerNormBackward", "param_main", 7, 6, 6, 0x038u),
    VK_TRAIN_SPEC("matMulBackward", "input_main", 7, 6, 6, 0x038u),
    VK_TRAIN_SPEC("matMulBackward", "weight_main", 7, 6, 6, 0x038u),
    VK_TRAIN_SPEC("matMulBackward", "bias_main", 7, 6, 6, 0x038u),
    VK_TRAIN_SPEC("moeLinearBackward", "input_main", 13, 12, 12, 0x3c0u),
    VK_TRAIN_SPEC("moeLinearBackward", "weight_main", 13, 12, 12, 0x3c0u),
    VK_TRAIN_SPEC("moeLinearBackward", "bias_main", 13, 12, 12, 0x3c0u),
    VK_TRAIN_SPEC("moeLinearBackward", "route_main", 13, 12, 12, 0x3c0u),
    VK_TRAIN_SPEC("moeRouterBackward", "logit_main", 11, 10, 10, 0x3c0u),
    VK_TRAIN_SPEC("moeRouterBackward", "input_main", 11, 10, 10, 0x3c0u),
    VK_TRAIN_SPEC("moeRouterBackward", "weight_main", 11, 10, 10, 0x3c0u),
    VK_TRAIN_SPEC("moeRouterBackward", "bias_main", 11, 10, 10, 0x3c0u),
    VK_TRAIN_SPEC("poolingBackward", "global_average_main", 4, 3, 3, 0x004u),
    VK_TRAIN_SPEC("poolingBackward", "max_pool_main", 4, 3, 3, 0x004u),
    VK_TRAIN_SPEC("preluBackward", "input_main", 6, 5, 5, 0x018u),
    VK_TRAIN_SPEC("preluBackward", "weight_main", 6, 5, 5, 0x018u),
    VK_TRAIN_SPEC("resizeBackward", "main", 3, 2, 2, 0x002u),
    VK_TRAIN_SPEC("rmsNormBackward", "input_main", 6, 5, 5, 0x018u),
    VK_TRAIN_SPEC("rmsNormBackward", "weight_main", 6, 5, 5, 0x018u),
    VK_TRAIN_SPEC("sdpaBackward", "main", 5, 4, 4, 0x008u),
    VK_TRAIN_SPEC("softmaxBackward", "main", 4, 3, 3, 0x004u),
    VK_TRAIN_SPEC("splitBackward", "main", 3, 2, 2, 0x002u),
    VK_TRAIN_SPEC("transposeBackward", "main", 3, 2, -1, 0x002u),
};

#undef VK_TRAIN_SPEC

#define VK_TRAINING_MAX_KERNELS 48
#define VK_TRAINING_MAX_DISPATCH_SETS 4096

struct VkTrainingKernelSlot {
    const VkTrainingSpec* spec;
    char path[160];
    VkKernelDefinition kernel;
};
#endif

static const char* vk_kernel_entry_point(VkKernel* kernel) {
#if VOLVOXAI_ENABLE_TRAINING
    for (int i = 0; i < training_kernel_count; i++) {
        if (&training_kernels[i].kernel == kernel && training_kernels[i].spec) {
            return training_kernels[i].spec->entry;
        }
    }
#else
    (void)kernel;
#endif
    return "main";
}

#include "vulkan_training.inc"

#include "vulkan_graph_f32.inc"

#include "vulkan_graph_quantized.inc"

int vk_graph_profile_x_f32(const float* in, float* out, int n, int h, int w, int c) {
    return vk_graph_profile_common(&k_profile_x, in, out, n, h, w, c,
                                   (long)2 * c * w, ((uint32_t)w + 63u) / 64u, (uint32_t)c);
}

int vk_graph_profile_y_f32(const float* in, float* out, int n, int h, int w, int c) {
    return vk_graph_profile_common(&k_profile_y, in, out, n, h, w, c,
                                   (long)2 * c * h, ((uint32_t)h + 63u) / 64u, (uint32_t)c);
}

int vk_graph_mean_height_f32(const float* in, float* out, int n, int h, int w, int c) {
    return vk_graph_profile_common(&k_mean_height, in, out, n, h, w, c,
                                   (long)c * w, ((uint32_t)w + 63u) / 64u, (uint32_t)c);
}

int vk_graph_nms_f32(const float* boxes, const float* scores, float* out,
                     int batches, int spatial, int classes, int max_output,
                     int output_rows, float iou_threshold, float score_threshold) {
    if (!boxes || !scores || !out || batches <= 0 || spatial <= 0 || classes <= 0 ||
        max_output <= 0 || output_rows <= 0) return 0;
    size_t boxes_bytes = (size_t)batches * (size_t)spatial * 4u * sizeof(float);
    size_t scores_bytes = (size_t)batches * (size_t)classes * (size_t)spatial * sizeof(float);
    size_t out_bytes = (size_t)output_rows * 3u * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* sb = graph_ensure_device(boxes, boxes_bytes, 0);
    VkTensorSlot* ss = graph_ensure_device(scores, scores_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sb || !ss || !dst) return 0;
    struct {
        uint32_t batches, spatial, classes, max_output, output_rows;
        float iou_threshold, score_threshold;
        uint32_t pad;
    } params = {(uint32_t)batches, (uint32_t)spatial, (uint32_t)classes, (uint32_t)max_output,
                (uint32_t)output_rows, iou_threshold, score_threshold, 0u};
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {sb->offset, boxes_bytes}, {ss->offset, scores_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_nms, binds, 1, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

static int vk_graph_concat_common(const float** inputs, const long* sizes, const int* input_axes,
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
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!dst) return 0;
    uint32_t axis_offset = 0;
    VkKernel* kernel = sigmoid ? &k_concat_sigmoid : &k_concat;
    for (int i = 0; i < count; i++) {
        int input_axis = input_axes ? input_axes[i] : (int)sizes[i];
        size_t in_bytes = (size_t)sizes[i] * sizeof(float);
        graph_scratch_begin();
        VkTensorSlot* src = graph_ensure_device(inputs[i], in_bytes, 0);
        if (!src) return 0;
        uint32_t params[8] = {
            (uint32_t)sizes[i], axis_offset, (uint32_t)input_axis,
            (uint32_t)output_axis, (uint32_t)inner, 0u, 0u, 0u
        };
        size_t p_off = graph_scratch_upload(params, sizeof(params));
        if (p_off == SIZE_MAX) return 0;
        VkGraphBinding binds[3] = {
            {src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
        };
        if (!vk_dispatch_kernel(kernel, binds, ((uint32_t)sizes[i] + 63u) / 64u, 1, 1)) return 0;
        axis_offset += (uint32_t)input_axis;
    }
    graph_mark_device(dst);
    return 1;
}

int vk_graph_concat_f32(const float** inputs, const long* sizes, const int* input_axes,
                        int count, float* out, int output_axis, int inner, int sigmoid) {
    if (!input_axes) return 0;
    return vk_graph_concat_common(inputs, sizes, input_axes, count, out, output_axis, inner, sigmoid);
}

int vk_graph_concat_32(
        const void* const* inputs, const long* sizes,
        const int* input_axes, int count, void* output,
        int output_axis, int inner) {
    uint32_t output_elements;
    size_t output_bytes;
    if (!vx_typed_control_concat_plan(
            inputs, sizes, input_axes, count, output,
            output_axis, inner, &output_elements, &output_bytes))
        return 0;
    graph_scratch_begin();
    VkTensorSlot* output_slot =
        graph_output_slot(output, output_bytes);
    if (!output_slot) return 0;
    uint32_t axis_offset = 0u;
    for (int index = 0; index < count; index++) {
        uint32_t input_elements = (uint32_t)sizes[index];
        size_t input_bytes =
            (size_t)input_elements * sizeof(uint32_t);
        VkTensorSlot* input_slot =
            graph_ensure_device(inputs[index], input_bytes, 0);
        if (!input_slot) return 0;
        uint32_t params[8] = {
            input_elements, axis_offset,
            (uint32_t)input_axes[index], (uint32_t)output_axis,
            (uint32_t)inner, 0u, 0u, 0u,
        };
        size_t params_offset =
            graph_scratch_upload(params, sizeof(params));
        if (params_offset == SIZE_MAX) return 0;
        VkGraphBinding bindings[3] = {
            {input_slot->offset, input_bytes},
            {output_slot->offset, output_bytes},
            {params_offset, sizeof(params)},
        };
        if (!vk_dispatch_kernel(
                &k_concat_32, bindings,
                (input_elements + 63u) / 64u, 1u, 1u))
            return 0;
        axis_offset += (uint32_t)input_axes[index];
    }
    graph_mark_device(output_slot);
    (void)output_elements;
    return 1;
}

int vk_graph_concat_flat_f32(const float** inputs, const long* sizes, int count, float* out) {
    long total = 0;
    if (!sizes || count <= 0) return 0;
    for (int i = 0; i < count; i++) total += sizes[i];
    if (total <= 0 || total > INT_MAX) return 0;
    return vk_graph_concat_common(inputs, sizes, NULL, count, out, (int)total, 1, 0);
}

int vk_graph_concat_sigmoid_flat_f32(const float** inputs, const long* sizes, int count, float* out) {
    long total = 0;
    if (!sizes || count <= 0) return 0;
    for (int i = 0; i < count; i++) total += sizes[i];
    if (total <= 0 || total > INT_MAX) return 0;
    return vk_graph_concat_common(inputs, sizes, NULL, count, out, (int)total, 1, 1);
}

int vk_graph_maxpool2d_f32(const float* in, float* out, int n, int h, int width, int c,
                           int out_h, int out_w, int ky, int kx, int sy, int sx,
                           int py, int px) {
    if (!in || !out || n <= 0 || c <= 0 || h <= 0 || width <= 0 || out_h <= 0 || out_w <= 0 ||
        (uint64_t)(uint32_t)n * (uint32_t)c > UINT32_MAX ||
        ky <= 0 || kx <= 0 || sy <= 0 || sx <= 0) {
        return 0;
    }
    size_t in_bytes = (size_t)n * h * width * c * sizeof(float);
    size_t out_bytes = (size_t)n * out_h * out_w * c * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[12] = {
        (uint32_t)n, (uint32_t)h, (uint32_t)width, (uint32_t)c,
        (uint32_t)out_h, (uint32_t)out_w,
        (uint32_t)ky, (uint32_t)kx, (uint32_t)sy, (uint32_t)sx,
        (uint32_t)py, (uint32_t)px
    };
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_maxpool, binds, ((uint32_t)out_w + 7u) / 8u,
                            ((uint32_t)out_h + 7u) / 8u, (uint32_t)n * (uint32_t)c)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_conv2d_f32(const float* in, float* out, const float* w, const float* bptr,
                             int n, int h, int width, int c, int out_c,
                             int kh, int kw, int out_h, int out_w,
                             int sy, int sx, int pt, int pl,
                             int groups, int relu, int dy, int dx) {
    if (!in || !out || !w || n <= 0 || h <= 0 || width <= 0 || c <= 0 || out_c <= 0 ||
        kh <= 0 || kw <= 0 || out_h <= 0 || out_w <= 0 || sy <= 0 || sx <= 0 ||
        dy <= 0 || dx <= 0 || groups <= 0 || relu > 2 ||
        (uint64_t)(uint32_t)n * (uint32_t)out_c > UINT32_MAX) {
        return 0;
    }
    if (!(groups == 1 || groups == c)) return 0;
    if (groups == c && out_c % c) return 0;
    size_t in_bytes = (size_t)n * h * width * c * sizeof(float);
    size_t w_elems = groups == c ? (size_t)kh * kw * c * (out_c / c) : (size_t)kh * kw * c * out_c;
    size_t w_bytes = w_elems * sizeof(float);
    size_t out_bytes = (size_t)n * out_h * out_w * out_c * sizeof(float);
    size_t bias_bytes = (size_t)out_c * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* wt = graph_ensure_device(w, w_bytes, 1);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !wt || !dst) return 0;

    size_t bias_off = SIZE_MAX;
    if (bptr) {
        VkTensorSlot* bias = graph_ensure_device(bptr, bias_bytes, 1);
        if (!bias) return 0;
        bias_off = bias->offset;
    } else {
        bias_off = graph_scratch_alloc(bias_bytes);
        if (bias_off == SIZE_MAX) return 0;
        if (!vk_staging_zero(bias_off, bias_bytes)) return 0;
    }

    uint32_t params[17] = {
        (uint32_t)n, (uint32_t)h, (uint32_t)width, (uint32_t)c,
        (uint32_t)out_c, (uint32_t)out_h, (uint32_t)out_w,
        (uint32_t)kh, (uint32_t)kw, (uint32_t)sy, (uint32_t)sx,
        (uint32_t)pt, (uint32_t)pl, (uint32_t)groups, (uint32_t)relu,
        (uint32_t)dy, (uint32_t)dx
    };
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[5] = {
        {src->offset, in_bytes}, {wt->offset, w_bytes}, {bias_off, bias_bytes},
        {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    VkKernel* kernel = &k_conv2d;
    uint32_t generic_gz = (uint32_t)(n * out_c);
    uint32_t gz = generic_gz;
    if (groups == 1 && kh == 1 && kw == 1 && sy == 1 && sx == 1 &&
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
    } else if (groups == 1 && (out_c & 15) == 0 &&
        vk_workgroup_shape_supported(8u, 8u, 1u)) {
        kernel = c == 3 ? &k_conv2d_c3out16 : &k_conv2d_out16;
        gz = (uint32_t)(n * (out_c / 16));
    } else if (groups == c && out_c == c) {
        if ((out_c & 7) == 0) {
            kernel = &k_conv2d_dw8;
            gz = (uint32_t)(n * ((out_c + 7) / 8));
        } else if ((out_c & 3) == 0) {
            kernel = &k_conv2d_dw4;
            gz = (uint32_t)(n * ((out_c + 3) / 4));
        }
    }
    uint32_t gx = ((uint32_t)out_w + 7u) / 8u;
    uint32_t gy = ((uint32_t)out_h + 7u) / 8u;
    int generic_geometry_valid =
        vk_dispatch_dimensions_valid(gx, gy, generic_gz);
    int ok = vk_dispatch_kernel(kernel, binds, gx, gy, gz);
    if (ok && (kernel == &k_conv2d_out16 || kernel == &k_conv2d_c3out16))
        vk_context_current()->conv_out16_dispatches++;
    else if (ok && kernel == &k_conv2d)
        vk_context_current()->conv_scalar_dispatches++;
    if (!ok && kernel != &k_conv2d) {
        if (generic_geometry_valid &&
            (kernel == &k_conv2d_out16 || kernel == &k_conv2d_c3out16))
            ok = vk_dispatch_kernel(
                &k_conv2d, binds, gx, gy, generic_gz);
        if (kernel == &k_conv2d_pw16tile) ok = vk_dispatch_kernel(&k_conv2d_pw16, binds, gx, gy, (uint32_t)(n * (out_c / 16)));
        if (!ok && kernel == &k_conv2d_dw8 && (out_c & 3) == 0) ok = vk_dispatch_kernel(&k_conv2d_dw4, binds, gx, gy, (uint32_t)(n * ((out_c + 3) / 4)));
        if (!ok && (kernel == &k_conv2d_pw8v4 || kernel == &k_conv2d_pw8v2)) {
            ok = vk_dispatch_kernel(&k_conv2d_pw8, binds, gx, gy, (uint32_t)(n * ((out_c + 7) / 8)));
        }
        if (!ok && generic_geometry_valid)
            ok = vk_dispatch_kernel(
                &k_conv2d, binds, gx, gy, generic_gz);
        if (ok && kernel != &k_conv2d)
            vk_context_current()->conv_scalar_dispatches++;
    }
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

static int checked_float_matrix_bytes(int rows, int columns, size_t* bytes) {
    size_t elements;
    if (!bytes || rows <= 0 || columns <= 0 ||
        (size_t)rows > SIZE_MAX / (size_t)columns) return 0;
    elements = (size_t)rows * (size_t)columns;
    if (elements > SIZE_MAX / sizeof(float)) return 0;
    *bytes = elements * sizeof(float);
    return 1;
}

static int checked_add_size(size_t left, size_t right, size_t* output) {
    if (!output || left > SIZE_MAX - right) return 0;
    *output = left + right;
    return 1;
}

static int checked_align_size(size_t value, size_t alignment, size_t* output) {
    size_t added;
    if (!output || alignment == 0 || (alignment & (alignment - 1u)) != 0u ||
        !checked_add_size(value, alignment - 1u, &added)) return 0;
    *output = added & ~(alignment - 1u);
    return 1;
}

int vk_graph_linear_f32(const float* input, const float* weight,
                        const float* bias, float* output, int rows,
                        int d_in, int d_out, int output_major_weight) {
    size_t input_bytes;
    size_t weight_bytes;
    size_t bias_bytes;
    size_t output_bytes;
    const float* bound_bias;
    VkTensorSlot* input_slot;
    VkTensorSlot* weight_slot;
    VkTensorSlot* bias_slot;
    VkTensorSlot* output_slot;
    VkTensorSlot* scale_slot = NULL;
    size_t params_offset;
    uint32_t params[3];
    uint32_t groups_x;
    uint32_t groups_y;
    int tiled;
    int ok;
    if (!input || !weight || !output || rows <= 0 || d_in <= 0 ||
        d_out <= 0 || (output_major_weight != 0 &&
                       output_major_weight != 1) ||
        !checked_float_matrix_bytes(rows, d_in, &input_bytes) ||
        !checked_float_matrix_bytes(d_in, d_out, &weight_bytes) ||
        !checked_float_matrix_bytes(1, d_out, &bias_bytes) ||
        !checked_float_matrix_bytes(rows, d_out, &output_bytes) ||
        qsdpa_ranges_overlap(output, output_bytes, input, input_bytes) ||
        qsdpa_ranges_overlap(output, output_bytes, weight, weight_bytes) ||
        (bias && qsdpa_ranges_overlap(
            output, output_bytes, bias, bias_bytes)))
        return 0;
    bound_bias = bias ? bias :
        (const float*)qconv_zero_bias_get((uint32_t)d_out);
    if (!bound_bias) return 0;

    graph_scratch_begin();
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
    params_offset = graph_scratch_upload(params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;

    tiled = rows > 1 && d_in >= 16 && d_out >= 16;
    groups_x = tiled ? ((uint32_t)d_out + 15u) / 16u
                     : ((uint32_t)d_out + 63u) / 64u;
    groups_y = tiled ? ((uint32_t)rows + 15u) / 16u
                     : (uint32_t)rows;
    if (output_major_weight) {
        VkGraphBinding bindings[6] = {
            {input_slot->offset, input_bytes},
            {weight_slot->offset, weight_bytes},
            {scale_slot->offset, sizeof(linear_dummy_scale)},
            {bias_slot->offset, bias_bytes},
            {output_slot->offset, output_bytes},
            {params_offset, sizeof(params)},
        };
        ok = vk_dispatch_kernel(
            tiled ? &k_linear_out_in_tiled : &k_linear_out_in,
            bindings, groups_x, groups_y, 1u);
    } else {
        VkGraphBinding bindings[5] = {
            {input_slot->offset, input_bytes},
            {weight_slot->offset, weight_bytes},
            {bias_slot->offset, bias_bytes},
            {output_slot->offset, output_bytes},
            {params_offset, sizeof(params)},
        };
        ok = vk_dispatch_kernel(
            tiled ? &k_linear_in_out_tiled : &k_linear_in_out,
            bindings, groups_x, groups_y, 1u);
    }
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

/* Legacy one-shot Linear transposes each constant exported [d_in,d_out]
 * matrix into a resident [d_out,d_in] region once.  The graph route above
 * preserves the declared layout and selects the matching shader instead. */
static int upload_weight(const float* w, int d_in, int d_out, size_t weight_bytes,
                         size_t* output_offset) {
    const size_t transpose_chunk_bytes = (size_t)1024 * 1024;
    float* transpose_chunk = NULL;
    size_t weight_elements;
    size_t off;
    size_t end;
    size_t next;
    if (!w || !output_offset || weight_bytes == 0 || d_in <= 0 || d_out <= 0 ||
        weight_bytes % sizeof(float) != 0u)
        return 0;
    for (int i = 0; i < wt_cache_n; i++) {
        if (wt_cache[i].src == w && wt_cache[i].d_in == d_in &&
            wt_cache[i].d_out == d_out && wt_cache[i].bytes == weight_bytes) {
            *output_offset = wt_cache[i].off;
            return 1;
        }
    }
    if (wt_cache_n >= WT_CACHE_MAX) return 0;
    off = wt_bump;
    /* Validate the complete aligned allocation before staging the transposed
       bytes. The transient input arena starts at WEIGHTS_LIMIT. */
    if (off > WEIGHTS_LIMIT || weight_bytes > WEIGHTS_LIMIT - off ||
        !checked_add_size(off, weight_bytes, &end) ||
        !checked_align_size(end, 65536u, &next) || next > WEIGHTS_LIMIT) return 0;
    weight_elements = weight_bytes / sizeof(float);
    size_t chunk_elements = transpose_chunk_bytes / sizeof(float);
    if (chunk_elements > weight_elements) chunk_elements = weight_elements;
    transpose_chunk = (float*)malloc(chunk_elements * sizeof(float));
    if (!transpose_chunk) return 0;
    for (size_t base = 0u; base < weight_elements; base += chunk_elements) {
        size_t count = weight_elements - base;
        if (count > chunk_elements) count = chunk_elements;
        for (size_t local = 0u; local < count; local++) {
            size_t transposed = base + local;
            size_t output_column = transposed / (size_t)d_in;
            size_t input_column = transposed % (size_t)d_in;
            transpose_chunk[local] =
                w[input_column * (size_t)d_out + output_column];
        }
        if (!vk_staging_upload(
                off + base * sizeof(float), transpose_chunk,
                count * sizeof(float))) {
            free(transpose_chunk);
            return 0;
        }
    }
    free(transpose_chunk);
    wt_bump = next;
    wt_cache[wt_cache_n].src = w;
    wt_cache[wt_cache_n].d_in = d_in;
    wt_cache[wt_cache_n].d_out = d_out;
    wt_cache[wt_cache_n].bytes = weight_bytes;
    wt_cache[wt_cache_n].off = off;
    wt_cache_n++;
    *output_offset = off;
    return 1;
}

// Reset the resident-weight arena (called when the engine reloads, so freed weight
// pointers are never matched against a newly-loaded model that reused the address).
void vk_free_weight_cache(void) {
    if (!vk_context_current()) return;
    wt_cache_n = 0;
    wt_bump = 0;
}

// Vulkan MatMul dispatch. M=1 uses the scalar shader; sufficiently large
// multi-row matrices use the cooperative 16x16 tiled shader.
int vk_matmul(const float* in, const float* w, const float* b, float* out,
              int seq, int d_in, int d_out) {
    size_t in_sz, w_sz, b_sz, out_sz;
    size_t offset_w;
    size_t offset_in = WEIGHTS_LIMIT;
    size_t offset_dummy, offset_b, offset_out, offset_params, end;
    const size_t param_sz = 12; // seq_len, d_in, d_out
    const size_t alignment = 65536u;
    int tiled;
    VkPipeline selected_pipeline;
    uint32_t groups_x;
    uint32_t groups_y;
    if (!vk_context_current() || !in || !w || !out ||
        !vk_is_ready() || io_buffer == VK_NULL_HANDLE ||
        seq <= 0 || d_in <= 0 || d_out <= 0 || !vk_graph_flush_wait() ||
        !checked_float_matrix_bytes(seq, d_in, &in_sz) ||
        !checked_float_matrix_bytes(d_in, d_out, &w_sz) ||
        !checked_float_matrix_bytes(1, d_out, &b_sz) ||
        !checked_float_matrix_bytes(seq, d_out, &out_sz)) return 0;
    tiled = seq > 1 && d_in >= 16 && d_out >= 16;
    selected_pipeline = tiled ? matmul_tiled_pipeline : matmul_pipeline;
    groups_x = tiled ? ((uint32_t)d_out + 15u) / 16u
                     : ((uint32_t)d_out + 63u) / 64u;
    groups_y = tiled ? ((uint32_t)seq + 15u) / 16u : (uint32_t)seq;
    if (selected_pipeline == VK_NULL_HANDLE ||
        !vk_dispatch_dimensions_valid(groups_x, groups_y, 1u) ||
        !upload_weight(w, d_in, d_out, w_sz, &offset_w)) return 0;

    /* Align every transient binding and reject the dispatch before copying if
       the configured IO arena cannot hold it. */
    if (!checked_add_size(offset_in, in_sz, &end) ||
        !checked_align_size(end, alignment, &offset_dummy) ||
        !checked_add_size(offset_dummy, sizeof(float), &end) ||
        !checked_align_size(end, alignment, &offset_b) ||
        !checked_add_size(offset_b, b_sz, &end) ||
        !checked_align_size(end, alignment, &offset_out) ||
        !checked_add_size(offset_out, out_sz, &end) ||
        !checked_align_size(end, alignment, &offset_params) ||
        !checked_add_size(offset_params, param_sz, &end) || end > io_size) return 0;

    // Bias binding always covers d_out columns; zero it when absent.
    if (!vk_staging_upload(offset_in, in, in_sz) ||
        !vk_staging_zero(offset_dummy, sizeof(float)) ||
        !(b ? vk_staging_upload(offset_b, b, b_sz)
            : vk_staging_zero(offset_b, b_sz)))
        return 0;
    uint32_t params[3] = {(uint32_t)seq, (uint32_t)d_in, (uint32_t)d_out};
    if (!vk_staging_upload(offset_params, params, sizeof(params))) return 0;

    VkDescriptorBufferInfo buf_infos[6];
    VkWriteDescriptorSet writes[6];
    buf_infos[0] = (VkDescriptorBufferInfo){io_buffer, offset_in, in_sz};
    buf_infos[1] = (VkDescriptorBufferInfo){io_buffer, offset_w, w_sz};
    buf_infos[2] = (VkDescriptorBufferInfo){io_buffer, offset_dummy, sizeof(float)};
    buf_infos[3] = (VkDescriptorBufferInfo){io_buffer, offset_b, b_sz};
    buf_infos[4] = (VkDescriptorBufferInfo){io_buffer, offset_out, out_sz};
    buf_infos[5] = (VkDescriptorBufferInfo){io_buffer, offset_params, param_sz};
    for (int i = 0; i < 6; i++) {
        writes[i] = (VkWriteDescriptorSet){0};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = desc_set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorType = (i == 5) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                             : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(device, 6, writes, 0, NULL);

    if (!vk_graph_begin_recording()) return 0;
    vk_context_current()->bind_pipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, selected_pipeline);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                            0, 1, &desc_set, 0, NULL);
    vk_context_current()->dispatch(cmd_buf, groups_x, groups_y, 1);
    return vk_staging_download(offset_out, out, out_sz);
}

#include "vulkan_tensor_interop.inc"
