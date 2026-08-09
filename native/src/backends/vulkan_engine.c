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
VK_FUNC(vkCreateShaderModule)
VK_FUNC(vkDestroyShaderModule)
VK_FUNC(vkCreateDescriptorSetLayout)
VK_FUNC(vkDestroyDescriptorSetLayout)
VK_FUNC(vkCreatePipelineLayout)
VK_FUNC(vkDestroyPipelineLayout)
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
#define VK_GRAPH_BASE ((size_t)128 * 1024 * 1024)
#define VK_GRAPH_ALIGN 256
#define VK_PREPARED_KERNEL_MAX 160
#define WT_CACHE_MAX 512
#define WEIGHTS_LIMIT ((size_t)64 * 1024 * 1024)

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
    VkPipeline matmul_pipeline;
    VkPipeline matmul_tiled_pipeline;
    unsigned context_count;
    int initialized;
    /* The device is shared across engine states and released at process exit,
     * so the teardown hook is registered exactly once. */
    int exit_hook_registered;
} VulkanDeviceState;

/* Every field below is owned by exactly one VxEngineState. It may be used by
 * only that engine's FIFO execution context at a time. */
typedef struct {
    int device_acquired;
    VkBuffer buffer;
    VkDeviceMemory memory;
    void* mapped;
    size_t arena_size;
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
#if defined(VOLVOXAI_VULKAN_TESTING)
    int test_domain_allocation_failure_after;
    uint64_t test_conv_pointwise_selections;
#endif
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
#define vkCreateShaderModule (g_vulkan_device.api.p_vkCreateShaderModule)
#define vkDestroyShaderModule (g_vulkan_device.api.p_vkDestroyShaderModule)
#define vkCreateDescriptorSetLayout (g_vulkan_device.api.p_vkCreateDescriptorSetLayout)
#define vkDestroyDescriptorSetLayout (g_vulkan_device.api.p_vkDestroyDescriptorSetLayout)
#define vkCreatePipelineLayout (g_vulkan_device.api.p_vkCreatePipelineLayout)
#define vkDestroyPipelineLayout (g_vulkan_device.api.p_vkDestroyPipelineLayout)
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
#define graph_alignment (g_vulkan_device.graph_alignment)
#define vulkan_packed_dot (g_vulkan_device.packed_dot)
#define vulkan_packed_dot_warned (g_vulkan_device.packed_dot_warned)
#define desc_layout (g_vulkan_device.matmul_desc_layout)
#define pipeline_layout (g_vulkan_device.matmul_pipeline_layout)
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
#define io_mapped (VX_VK_CONTEXT.mapped)
#define io_size (VX_VK_CONTEXT.arena_size)
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
static int vk_is_ready(void);
static void vk_context_destroy(void* opaque);
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
    return fallback != UINT32_MAX ? fallback : 0;
}

// First memory type that has all `required` flags but none of `avoid`. Used to fall
// back from a device-local host-visible type (a small PCIe BAR heap on discrete GPUs)
// to plain host-visible system memory when the BAR is too small for io_size.
static uint32_t find_memory_type_avoiding(uint32_t type_filter,
                                          VkMemoryPropertyFlags required,
                                          VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if (!(type_filter & (1u << i))) continue;
        VkMemoryPropertyFlags flags = mem_props.memoryTypes[i].propertyFlags;
        if ((flags & required) != required) continue;
        if (flags & avoid) continue;
        return i;
    }
    return UINT32_MAX;
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
    result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, output);
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
    LOAD_INST(vkCreateShaderModule)
    LOAD_INST(vkDestroyShaderModule)
    LOAD_INST(vkCreateDescriptorSetLayout)
    LOAD_INST(vkDestroyDescriptorSetLayout)
    LOAD_INST(vkCreatePipelineLayout)
    LOAD_INST(vkDestroyPipelineLayout)
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

    if (vkCreateDevice(physical_device, &device_info, NULL, &device) != VK_SUCCESS) return -1;
    vkGetDeviceQueue(device, queue_family_index, 0, &compute_queue);

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
    if (device != VK_NULL_HANDLE) {
        if (vkDeviceWaitIdle) (void)vkDeviceWaitIdle(device);
        if (matmul_pipeline != VK_NULL_HANDLE && vkDestroyPipeline)
            vkDestroyPipeline(device, matmul_pipeline, NULL);
        if (matmul_tiled_pipeline != VK_NULL_HANDLE && vkDestroyPipeline)
            vkDestroyPipeline(device, matmul_tiled_pipeline, NULL);
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
    graph_alignment = VK_GRAPH_ALIGN;
    vulkan_packed_dot = 0;
    vulkan_packed_dot_warned = 0;
    desc_layout = VK_NULL_HANDLE;
    pipeline_layout = VK_NULL_HANDLE;
    matmul_pipeline = VK_NULL_HANDLE;
    matmul_tiled_pipeline = VK_NULL_HANDLE;
#if VOLVOXAI_ENABLE_TRAINING
    training_max_storage_bindings = 0;
    training_max_uniform_bindings = 0;
    training_max_workgroup_size_x = 0;
    training_max_workgroup_invocations = 0;
    training_max_storage_range = 0;
    training_max_uniform_range = 0;
#endif
    g_vulkan_device.initialized = 0;
}

static VulkanContextState* vk_context_allocate(VxEngineState* owner) {
    VulkanContextState* context;
    if (!owner) return NULL;
    if (owner->vulkan_context_state)
        return (VulkanContextState*)owner->vulkan_context_state;
    context = (VulkanContextState*)calloc(1, sizeof(*context));
    if (!context) return NULL;
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
#if defined(VOLVOXAI_VULKAN_TESTING)
    context->test_domain_allocation_failure_after = -1;
#endif
    owner->vulkan_context_state = context;
    return context;
}

#if defined(VOLVOXAI_VULKAN_TESTING)
int vk_test_context_state_write(const VkContextStateProbe* probe) {
    VxEngineState* owner = vx_engine_state_current();
    VulkanContextState* context;
    if (!owner || !probe || probe->slot_count < 0 ||
        probe->slot_count > VK_GRAPH_MAX_TENSORS ||
        probe->dispatch_cursor < 0 ||
        probe->dispatch_cursor > VK_GRAPH_MAX_DISPATCH_SETS ||
        probe->touched_count < 0 ||
        probe->touched_count > VK_GRAPH_MAX_TENSORS) return -1;
    context = vk_context_allocate(owner);
    if (!context) return -1;
    owner->vulkan_context_state_destroy = vk_context_destroy;
    context->tensor_slot_count = probe->slot_count;
    context->arena_bump = probe->arena_cursor;
    context->dispatch_set_cursor = probe->dispatch_cursor;
#if VOLVOXAI_ENABLE_TRAINING
    context->training_touched_count_value = probe->touched_count;
    context->training_is_active = probe->is_training ? 1 : 0;
#else
    if (probe->touched_count != 0 || probe->is_training) return -1;
#endif
    return 0;
}

int vk_test_context_state_read(VkContextStateProbe* probe) {
    VulkanContextState* context = vk_context_current();
    if (!context || !probe) return -1;
    probe->slot_count = context->tensor_slot_count;
    probe->arena_cursor = context->arena_bump;
    probe->dispatch_cursor = context->dispatch_set_cursor;
#if VOLVOXAI_ENABLE_TRAINING
    probe->touched_count = context->training_touched_count_value;
    probe->is_training = context->training_is_active;
#else
    probe->touched_count = 0;
    probe->is_training = 0;
#endif
    return 0;
}

int vk_test_graph_dynamic_state(VkGraphDynamicStateProbe* probe) {
    VulkanContextState* context = vk_context_current();
    if (!context || !probe) return -1;
    memset(probe, 0, sizeof(*probe));
    probe->shape_generation = context->shape_generation;
    probe->capacity_generation = context->capacity_generation;
    probe->domain_span_count = context->domain_span_count;
    probe->domain_scratch_capacity_bytes =
        context->domain_qgroupnorm_stats_bytes +
        context->domain_qlayernorm_stats_bytes;
    probe->domain_enforced = context->domain_enforced;
    probe->slot_count = context->tensor_slot_count;
    for (int index = 0; index < context->tensor_slot_count; index++) {
        VkTensorSlot* slot = &context->tensor_slots[index];
        if (!slot->owns_range) continue;
        if (slot->host) probe->active_capacity_bytes += slot->capacity;
        else probe->pooled_capacity_bytes += slot->capacity;
    }
    return 0;
}

int vk_test_fail_domain_allocation_after(size_t successful_allocations) {
    VulkanContextState* context = vk_context_current();
    if (!context || successful_allocations > (size_t)INT_MAX) return -1;
    context->test_domain_allocation_failure_after =
        (int)successful_allocations;
    return 0;
}

int vk_test_set_packed_dot_disabled(int disabled) {
    VulkanContextState* context = vk_context_current();
    int previous;
    if (!context || (disabled != 0 && disabled != 1)) return -1;
    previous = context->packed_dot_disabled;
    context->packed_dot_disabled = disabled;
    return previous;
}

void vk_test_qconv_tactic_reset(void) {
    VulkanContextState* context = vk_context_current();
    if (!context) return;
    context->qconv_dot_tiled_dispatches = 0u;
    context->qconv_tiled_dispatches = 0u;
    context->qconv_scalar_dispatches = 0u;
}

int vk_test_qconv_tactic_read(VkQConvTacticProbe* probe) {
    VulkanContextState* context = vk_context_current();
    if (!context || !probe) return -1;
    probe->dot_tiled_dispatches = context->qconv_dot_tiled_dispatches;
    probe->tiled_dispatches = context->qconv_tiled_dispatches;
    probe->scalar_dispatches = context->qconv_scalar_dispatches;
    return 0;
}

void vk_test_conv_tactic_reset(void) {
    VulkanContextState* context = vk_context_current();
    if (!context) return;
    context->test_conv_pointwise_selections = 0u;
    context->conv_out16_dispatches = 0u;
    context->conv_scalar_dispatches = 0u;
}

int vk_test_conv_tactic_read(VkConvTacticProbe* probe) {
    VulkanContextState* context = vk_context_current();
    if (!context || !probe) return -1;
    probe->pointwise_selections = context->test_conv_pointwise_selections;
    probe->out16_dispatches = context->conv_out16_dispatches;
    probe->scalar_dispatches = context->conv_scalar_dispatches;
    return 0;
}
#endif

static int vk_context_create_resources_locked(VulkanContextState* context) {
    VkBufferCreateInfo buffer_info = {0};
    VkMemoryRequirements memory_requirements;
    VkMemoryAllocateInfo allocation_info = {0};
    VkResult allocation_result;
    VkDescriptorPoolSize pool_sizes[2] = {0};
    VkDescriptorPoolCreateInfo pool_info = {0};
    VkDescriptorSetAllocateInfo set_info = {0};
    VkCommandPoolCreateInfo command_pool_info = {0};
    VkCommandBufferAllocateInfo command_buffer_info = {0};
    VkFenceCreateInfo fence_info = {0};
    const VkMemoryPropertyFlags host_requirements =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const char* mb_env;
    uint32_t primary_type;
    if (!context || !g_vulkan_device.initialized) return -1;

    mb_env = getenv("VOLVOX_VULKAN_MB");
    if (mb_env && mb_env[0]) {
        long mb = strtol(mb_env, NULL, 10);
        if (mb >= 192 && mb <= 4096)
            context->arena_size = (size_t)mb * 1024 * 1024;
    }
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = context->arena_size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &buffer_info, NULL, &context->buffer) != VK_SUCCESS)
        return -1;
    vkGetBufferMemoryRequirements(device, context->buffer, &memory_requirements);
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = memory_requirements.size;
    primary_type = find_preferred_memory_type(
        memory_requirements.memoryTypeBits, host_requirements,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    allocation_info.memoryTypeIndex = primary_type;
    allocation_result = vkAllocateMemory(
        device, &allocation_info, NULL, &context->memory);
    if (allocation_result != VK_SUCCESS) {
        uint32_t fallback_type = find_memory_type_avoiding(
            memory_requirements.memoryTypeBits, host_requirements,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (fallback_type != UINT32_MAX && fallback_type != primary_type) {
            allocation_info.memoryTypeIndex = fallback_type;
            allocation_result = vkAllocateMemory(
                device, &allocation_info, NULL, &context->memory);
        }
    }
    if (allocation_result != VK_SUCCESS) return -1;
    if (vkBindBufferMemory(device, context->buffer,
                           context->memory, 0) != VK_SUCCESS) return -1;
    if (vkMapMemory(device, context->memory, 0, context->arena_size, 0,
                    &context->mapped) != VK_SUCCESS || !context->mapped) return -1;

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
            if (context->mapped && vkUnmapMemory)
                vkUnmapMemory(device, context->memory);
            if (context->buffer != VK_NULL_HANDLE && vkDestroyBuffer)
                vkDestroyBuffer(device, context->buffer, NULL);
            if (context->memory != VK_NULL_HANDLE && vkFreeMemory)
                vkFreeMemory(device, context->memory, NULL);
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
        context->mapped != NULL;
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

static int vk_graph_align_up(size_t value, size_t alignment, size_t* out) {
    size_t remainder;
    if (!out || !alignment) return 0;
    remainder = value % alignment;
    if (remainder && value > SIZE_MAX - (alignment - remainder)) return 0;
    *out = remainder ? value + (alignment - remainder) : value;
    return 1;
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
            if (bytes > s->bytes) {
                s->host_dirty = 1;
                s->device_dirty = 0;
            }
            s->bytes = bytes;
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
        memcpy((char*)io_mapped + s->offset, host, bytes);
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
        memcpy((char*)io_mapped + s->offset, host, logical_bytes);
        if (storage_bytes > logical_bytes) {
            memset((char*)io_mapped + s->offset + logical_bytes, 0,
                   storage_bytes - logical_bytes);
        }
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
    memcpy((char*)io_mapped + off, data, bytes);
    return off;
}

/* Zero-filled scratch for an optional binding the shader always declares. */
static size_t graph_scratch_zero(size_t bytes) {
    size_t off = graph_scratch_alloc(bytes);
    if (off == SIZE_MAX) return SIZE_MAX;
    memset((char*)io_mapped + off, 0, bytes);
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
#if defined(VOLVOXAI_VULKAN_TESTING)
    vk_context_current()->test_domain_allocation_failure_after = -1;
#endif
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
#if defined(VOLVOXAI_VULKAN_TESTING)
        if (context->test_domain_allocation_failure_after == 0) {
            context->test_domain_allocation_failure_after = -1;
            result = -2;
            goto rollback;
        }
        if (context->test_domain_allocation_failure_after > 0)
            context->test_domain_allocation_failure_after--;
#endif
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
#if defined(VOLVOXAI_VULKAN_TESTING)
    context->test_conv_pointwise_selections = 0u;
#endif
}

int vk_graph_end_forward(void) {
    if (!vk_context_current()) return -1;
    return vk_graph_flush_wait() ? 0 : -1;
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
    if (!output || !output_capacity || !context) return -1;
    size_t offset = 0u;
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
        if (!vk_graph_flush_wait()) return 0;
        memcpy((void*)host, (char*)io_mapped + s->offset, bytes);
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
        device, VK_NULL_HANDLE, 1, &pipe_info, NULL, &prepared.pipeline);
    if (vkDestroyShaderModule) vkDestroyShaderModule(device, shader_module, NULL);
    if (pipeline_result != VK_SUCCESS) goto fail;

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
    VkResult submit_result = reset_result == VK_SUCCESS
        ? vkQueueSubmit(compute_queue, 1, &submit_info, compute_fence)
        : reset_result;
    pthread_mutex_unlock(&g_vulkan_device.mutex);
    if (submit_result != VK_SUCCESS) {
        graph_cmd_recording = 0;
        return 0;
    }
    graph_cmd_recording = 0;
    graph_cmd_pending = 1;
    return 1;
}

static int vk_graph_flush_wait(void) {
    if (!vk_is_ready()) return 1;
    if (graph_cmd_recording && !vk_graph_submit_recording()) return 0;
    if (graph_cmd_pending) {
        if (vkWaitForFences(device, 1, &compute_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) return 0;
        graph_cmd_pending = 0;
    }
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
    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                      prepared->pipeline);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            prepared->prepared_pipeline_layout, 0, 1,
                            &dispatch_set, 0, NULL);
    vkCmdDispatch(cmd_buf, gx, gy, gz);
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
    VK_TRAIN_SPEC("moeLinearBackward", "input_main", 11, 10, 10, 0x3c0u),
    VK_TRAIN_SPEC("moeLinearBackward", "weight_main", 11, 10, 10, 0x3c0u),
    VK_TRAIN_SPEC("moeLinearBackward", "bias_main", 11, 10, 10, 0x3c0u),
    VK_TRAIN_SPEC("moeLinearBackward", "route_main", 11, 10, 10, 0x3c0u),
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

#if VOLVOXAI_ENABLE_TRAINING
static void vk_training_track_tensor(VkTensorSlot* tensor) {
    if (!tensor) return;
    for (int i = 0; i < training_touched_count; i++) {
        if (training_touched_slots[i] == tensor) return;
    }
    if (training_touched_count < VK_GRAPH_MAX_TENSORS) {
        training_touched_slots[training_touched_count++] = tensor;
    }
}

static const VkTrainingSpec* vk_training_find_spec(const char* shader, const char* entry) {
    if (!shader || !entry) return NULL;
    size_t count = sizeof(training_specs) / sizeof(training_specs[0]);
    for (size_t i = 0; i < count; i++) {
        if (!strcmp(shader, training_specs[i].shader) &&
            !strcmp(entry, training_specs[i].entry)) {
            return &training_specs[i];
        }
    }
    return NULL;
}

static int vk_training_prepare_pool(void) {
    if (training_desc_pool != VK_NULL_HANDLE) return 1;
    VkTrainingKernelSlot* kernels = (VkTrainingKernelSlot*)calloc(
        VK_TRAINING_MAX_KERNELS, sizeof(*kernels));
    VkGraphDispatchSet* sets = (VkGraphDispatchSet*)calloc(
        VK_TRAINING_MAX_DISPATCH_SETS, sizeof(*sets));
    VkTensorSlot** touched = (VkTensorSlot**)calloc(
        VK_GRAPH_MAX_TENSORS, sizeof(*touched));
    if (!kernels || !sets || !touched) {
        free(kernels);
        free(sets);
        free(touched);
        return 0;
    }
    VkDescriptorPoolSize sizes[2];
    memset(sizes, 0, sizeof(sizes));
    sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[0].descriptorCount = VK_TRAINING_MAX_DISPATCH_SETS * 12u +
                               VK_TRAINING_MAX_KERNELS * 12u;
    sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = VK_TRAINING_MAX_DISPATCH_SETS +
                               VK_TRAINING_MAX_KERNELS;
    VkDescriptorPoolCreateInfo info = {0};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.maxSets = VK_TRAINING_MAX_DISPATCH_SETS + VK_TRAINING_MAX_KERNELS;
    info.poolSizeCount = 2;
    info.pPoolSizes = sizes;
    pthread_mutex_lock(&g_vulkan_device.mutex);
    VkResult pool_result = vkCreateDescriptorPool(
        device, &info, NULL, &training_desc_pool);
    pthread_mutex_unlock(&g_vulkan_device.mutex);
    if (pool_result != VK_SUCCESS) {
        free(kernels);
        free(sets);
        free(touched);
        return 0;
    }
    training_kernels = kernels;
    training_dispatch_sets = sets;
    training_touched_slots = touched;
    return 1;
}

static VkKernel* vk_training_kernel(const VkTrainingSpec* spec) {
    if (!spec || training_desc_pool == VK_NULL_HANDLE) return NULL;
    for (int i = 0; i < training_kernel_count; i++) {
        if (training_kernels[i].spec == spec) return &training_kernels[i].kernel;
    }
    if (training_kernel_count >= VK_TRAINING_MAX_KERNELS) return NULL;
    VkTrainingKernelSlot* slot = &training_kernels[training_kernel_count++];
    memset(slot, 0, sizeof(*slot));
    slot->spec = spec;
    int n = snprintf(slot->path, sizeof(slot->path),
                     "spv/%s.spv", spec->shader);
    if (n <= 0 || (size_t)n >= sizeof(slot->path)) {
        training_kernel_count--;
        memset(slot, 0, sizeof(*slot));
        return NULL;
    }
    slot->kernel.name = spec->shader;
    slot->kernel.path = slot->path;
    slot->kernel.binding_count = spec->binding_count;
    slot->kernel.uniform_binding = spec->uniform_binding;
    return &slot->kernel;
}

static VkDescriptorSet vk_training_dispatch_set(
        VkKernel* kernel, const VkPreparedKernel* prepared) {
    if (!kernel || !prepared || prepared->prepared_desc_layout == VK_NULL_HANDLE ||
        training_dispatch_set_cursor >= VK_TRAINING_MAX_DISPATCH_SETS) {
        return VK_NULL_HANDLE;
    }
    int index = training_dispatch_set_cursor++;
    VkGraphDispatchSet* slot = &training_dispatch_sets[index];
    if (slot->descriptor_set != VK_NULL_HANDLE && slot->kernel == kernel) return slot->descriptor_set;

    VkDescriptorSetAllocateInfo info = {0};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.descriptorPool = training_desc_pool;
    info.descriptorSetCount = 1;
    info.pSetLayouts = &prepared->prepared_desc_layout;
    if (vkAllocateDescriptorSets(device, &info, &slot->descriptor_set) != VK_SUCCESS) {
        slot->descriptor_set = VK_NULL_HANDLE;
        slot->kernel = NULL;
        return VK_NULL_HANDLE;
    }
    slot->kernel = kernel;
    if (index >= training_dispatch_set_count) training_dispatch_set_count = index + 1;
    return slot->descriptor_set;
}

static int vk_training_dispatch_kernel(VkKernel* kernel,
                                       const VkGraphBinding* bindings,
                                       uint32_t gx, uint32_t gy, uint32_t gz) {
    VkPreparedKernel* prepared;
    if (!vk_dispatch_dimensions_valid(gx, gy, gz) || !bindings ||
        kernel->binding_count > 16) return 0;
    prepared = vk_prepare_kernel(kernel);
    if (!prepared) return 0;
    VkDescriptorSet set = vk_training_dispatch_set(kernel, prepared);
    if (set == VK_NULL_HANDLE) return 0;

    VkDescriptorBufferInfo infos[16];
    VkWriteDescriptorSet writes[16];
    memset(infos, 0, sizeof(infos));
    memset(writes, 0, sizeof(writes));
    for (int i = 0; i < kernel->binding_count; i++) {
        infos[i].buffer = io_buffer;
        infos[i].offset = bindings[i].offset;
        infos[i].range = bindings[i].bytes;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorType = i == kernel->uniform_binding ?
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device, (uint32_t)kernel->binding_count, writes, 0, NULL);

    if (!vk_graph_begin_recording()) return 0;
    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                      prepared->pipeline);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            prepared->prepared_pipeline_layout, 0, 1, &set, 0, NULL);
    vkCmdDispatch(cmd_buf, gx, gy, gz);
    if (vkCmdPipelineBarrier) {
        VkMemoryBarrier barrier = {0};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             1, &barrier, 0, NULL, 0, NULL);
    }
    return 1;
}

int vk_training_available(void) {
    return vk_is_ready() ? 1 : 0;
}

int vk_training_supports(const char* shader_name, const char* entry_point,
                         const size_t* bytes, int binding_count,
                         uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    const VkTrainingSpec* spec = vk_training_find_spec(shader_name, entry_point);
    if (!vk_training_available() || !spec || !bytes ||
        binding_count != spec->binding_count || binding_count <= 0 ||
        binding_count > 16 || spec->params_binding != binding_count - 1 ||
        !vk_dispatch_dimensions_valid(groups_x, groups_y, groups_z) ||
        training_max_workgroup_size_x < 64 ||
        training_max_workgroup_invocations < 64) return 0;

    uint32_t storage_bindings = spec->uniform_binding >= 0
        ? (uint32_t)(binding_count - 1) : (uint32_t)binding_count;
    uint32_t uniform_bindings = spec->uniform_binding >= 0 ? 1u : 0u;
    if (storage_bindings > training_max_storage_bindings ||
        uniform_bindings > training_max_uniform_bindings) return 0;
    for (int i = 0; i < binding_count; i++) {
        VkDeviceSize limit = i == spec->uniform_binding
            ? training_max_uniform_range : training_max_storage_range;
        if (bytes[i] == 0 || (VkDeviceSize)bytes[i] > limit) return 0;
    }
    return 1;
}

int vk_training_plan_supported(int command_count, const size_t* params_bytes,
                               const VkTrainingTensorRequirement* tensors,
                               int tensor_count) {
    if (!vk_training_available() || command_count <= 0 ||
        command_count > VK_TRAINING_MAX_DISPATCH_SETS || !params_bytes ||
        tensor_count < 0 || (tensor_count > 0 && !tensors) ||
        graph_alignment == 0 || io_size < VK_GRAPH_SCRATCH_BYTES) return 0;

    size_t arena_limit = io_size - VK_GRAPH_SCRATCH_BYTES;
    size_t arena_cursor = graph_bump;
    int additional_slots = 0;
    for (int i = 0; i < tensor_count; i++) {
        if (!tensors[i].host || tensors[i].bytes == 0) return 0;
        int slot = graph_find_slot(tensors[i].host);
        if (slot >= 0 && tensors[i].bytes <= graph_slots[slot].bytes) continue;
        if (slot < 0) additional_slots++;
        if (arena_cursor > SIZE_MAX - (graph_alignment - 1)) return 0;
        arena_cursor = ALIGN_UP(arena_cursor, graph_alignment);
        if (arena_cursor > arena_limit ||
            tensors[i].bytes > arena_limit - arena_cursor) return 0;
        arena_cursor += tensors[i].bytes;
        if (arena_cursor > SIZE_MAX - (graph_alignment - 1)) return 0;
        arena_cursor = ALIGN_UP(arena_cursor, graph_alignment);
    }
    if (additional_slots > VK_GRAPH_MAX_TENSORS - graph_slot_count) return 0;

    size_t cursor = io_size - VK_GRAPH_SCRATCH_BYTES;
    if (cursor > SIZE_MAX - (graph_alignment - 1)) return 0;
    cursor = ALIGN_UP(cursor, graph_alignment);
    for (int i = 0; i < command_count; i++) {
        size_t bytes = params_bytes[i];
        if (bytes == 0 || cursor > io_size || bytes > io_size - cursor) return 0;
        cursor += bytes;
        if (cursor > SIZE_MAX - (graph_alignment - 1)) return 0;
        cursor = ALIGN_UP(cursor, graph_alignment);
    }
    return cursor <= io_size;
}

int vk_training_begin(void) {
    if (!vk_is_ready() || training_active || !vk_graph_flush_wait()) return -1;
    int reuse_pool = training_desc_pool != VK_NULL_HANDLE;
    if (!vk_training_prepare_pool()) return -1;
    if (reuse_pool) {
        if (!vkResetDescriptorPool ||
            vkResetDescriptorPool(device, training_desc_pool, 0) != VK_SUCCESS) return -1;
        memset(training_dispatch_sets, 0,
               VK_TRAINING_MAX_DISPATCH_SETS * sizeof(*training_dispatch_sets));
        training_dispatch_set_count = 0;
    }
    scratch_bump = 0;
    scratch_high_water = 0;
    training_dispatch_set_cursor = 0;
    training_touched_count = 0;
    training_active = 1;
    return 0;
}

int vk_training_dispatch(const char* shader_name, const char* entry_point,
                         void* const* hosts, const size_t* bytes,
                         const unsigned char* access,
                         const unsigned char* is_weight,
                         int binding_count,
                         uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    const VkTrainingSpec* spec = vk_training_find_spec(shader_name, entry_point);
    if (!training_active || !spec || !hosts || !bytes || !access ||
        !vk_training_supports(shader_name, entry_point, bytes, binding_count,
                              groups_x, groups_y, groups_z)) return -1;

    graph_scratch_begin();
    VkGraphBinding bindings[16];
    VkTensorSlot* writable[16];
    memset(bindings, 0, sizeof(bindings));
    memset(writable, 0, sizeof(writable));
    for (int i = 0; i < binding_count; i++) {
        unsigned char mode = access[i];
        if (!hosts[i] || !bytes[i] || (is_weight && is_weight[i] > 1) ||
            mode < VK_TRAINING_READ ||
            mode > (VK_TRAINING_READ | VK_TRAINING_WRITE)) return -1;
        unsigned char expected = spec->read_write_mask & (1u << i) ?
            (VK_TRAINING_READ | VK_TRAINING_WRITE) : VK_TRAINING_READ;
        if (mode != expected) return -1;
        if (i == spec->params_binding) {
            if (mode != VK_TRAINING_READ) return -1;
            size_t offset = graph_scratch_upload(hosts[i], bytes[i]);
            if (offset == SIZE_MAX) return -1;
            bindings[i].offset = offset;
            bindings[i].bytes = bytes[i];
            continue;
        }

        VkTensorSlot* tensor = NULL;
        if (mode & VK_TRAINING_READ) {
            tensor = graph_ensure_device(hosts[i], bytes[i], is_weight ? is_weight[i] != 0 : 0);
        } else {
            tensor = graph_get_slot(hosts[i], bytes[i], is_weight ? is_weight[i] != 0 : 0);
            if (tensor && tensor->host_dirty) {
                memset((char*)io_mapped + tensor->offset, 0, bytes[i]);
                tensor->host_dirty = 0;
                tensor->device_dirty = 0;
            }
        }
        if (!tensor) return -1;
        vk_training_track_tensor(tensor);
        bindings[i].offset = tensor->offset;
        bindings[i].bytes = bytes[i];
        if (mode & VK_TRAINING_WRITE) writable[i] = tensor;
    }

    VkKernel* kernel = vk_training_kernel(spec);
    if (!kernel || !vk_training_dispatch_kernel(kernel, bindings,
                                                 groups_x, groups_y, groups_z)) return -1;
    for (int i = 0; i < binding_count; i++) {
        if (writable[i]) {
            graph_mark_device(writable[i]);
        }
    }
    return 0;
}

int vk_training_sync(void* host, size_t bytes) {
    if (!training_active || !host || !bytes) return -1;
    return vk_graph_sync_host(host, bytes, 0) ? 0 : -1;
}

void vk_training_end(void) {
    if (!training_active) return;
    (void)vk_graph_flush_wait();
    /* Training arrays are normally released after a step.  Mark every host
       identity touched by training as a new generation so malloc/stack
       address reuse cannot retain old inputs or gradients.  device_dirty is
       intentionally preserved: callers that kept a buffer can still download
       it through vk_graph_sync_host().  A later GPU forward output clears the
       host-dirty bit when it overwrites the same stable model tensor. */
    for (int i = 0; i < training_touched_count; i++) {
        if (training_touched_slots[i]) training_touched_slots[i]->host_dirty = 1;
        training_touched_slots[i] = NULL;
    }
    training_touched_count = 0;
    scratch_bump = 0;
    scratch_high_water = 0;
    training_dispatch_set_cursor = 0;
    training_active = 0;
}
#endif

int vk_graph_alias_f32(const float* in, float* out, long n) {
    if (n <= 0 || !in || !out) return 0;
    if (graph_domain_enforced && in != out)
        return vk_graph_copy_f32(in, out, n);
    size_t bytes = (size_t)n * sizeof(float);
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    if (!src) return 0;
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!dst) return 0;
    if (dst == src) {
        graph_mark_device(dst);
        return 1;
    }
    if (dst->owns_range && dst->capacity &&
        !vk_graph_range_free(dst->offset, dst->capacity)) return 0;
    dst->offset = src->offset;
    dst->bytes = bytes;
    dst->capacity = src->capacity;
    dst->capacity_generation = src->capacity_generation;
    dst->owns_range = 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_copy_f32(const float* in, float* out, long n) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[1] = {(uint32_t)n};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding b[3] = {
        {src->offset, bytes}, {dst->offset, bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_copy, b, ((uint32_t)n + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_add_f32(const float* a, const float* bptr, float* out, long n) {
    return vk_graph_add_relu_f32(a, bptr, out, n, 0);
}

int vk_graph_add_relu_f32(const float* a, const float* bptr, float* out, long n, int relu) {
    if (n <= 0 || !a || !bptr || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* sa = graph_ensure_device(a, bytes, 0);
    VkTensorSlot* sb = graph_ensure_device(bptr, bytes, 0);
    VkTensorSlot* so = graph_output_slot(out, bytes);
    if (!sa || !sb || !so) return 0;
    uint32_t params[2] = {(uint32_t)n, (uint32_t)relu};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {sa->offset, bytes}, {sb->offset, bytes}, {so->offset, bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(relu ? &k_add_relu : &k_add, binds, ((uint32_t)n + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(so);
    return 1;
}

int vk_graph_clip_f32(const float* in, float* out, long n, float min_v, float max_v) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    struct { uint32_t size; float min_v; float max_v; } params = {(uint32_t)n, min_v, max_v};
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, bytes}, {dst->offset, bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_clip, binds, ((uint32_t)n + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

static int vk_graph_typed_control_32(
        const void* a, size_t a_bytes, const void* b, size_t b_bytes,
        void* output, size_t output_bytes,
        const VxTypedControlMetadata* metadata) {
    graph_scratch_begin();
    VkTensorSlot* a_slot = graph_ensure_device(a, a_bytes, 0);
    VkTensorSlot* b_slot = graph_ensure_device(b, b_bytes, 0);
    VkTensorSlot* output_slot = graph_output_slot(output, output_bytes);
    if (!a_slot || !b_slot || !output_slot || !metadata) return 0;
    size_t metadata_offset =
        graph_scratch_upload(metadata, sizeof(*metadata));
    if (metadata_offset == SIZE_MAX) return 0;
    VkGraphBinding bindings[4] = {
        {a_slot->offset, a_bytes},
        {b_slot->offset, b_bytes},
        {output_slot->offset, output_bytes},
        {metadata_offset, sizeof(*metadata)},
    };
    uint32_t elements = metadata->values[0];
    if (!vk_dispatch_kernel(
            &k_typed_control_32, bindings,
            (elements + 63u) / 64u, 1u, 1u))
        return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_compare_i32(
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
    return vk_graph_typed_control_32(
        a, a_bytes, b, b_bytes, output, output_bytes, &metadata);
}

static int vk_graph_unary_i32(
        const int32_t* input, int32_t* output, long elements,
        int operation, int32_t minimum, int32_t maximum) {
    VxTypedControlMetadata metadata;
    size_t bytes;
    if (!vx_typed_control_unary_plan(
            input, output, elements, operation, minimum, maximum,
            &metadata, &bytes))
        return 0;
    return vk_graph_typed_control_32(
        input, bytes, input, bytes, output, bytes, &metadata);
}

int vk_graph_not_i32(
        const int32_t* input, int32_t* output, long elements) {
    return vk_graph_unary_i32(
        input, output, elements, VX_TYPED_CONTROL_NOT_I32, 0, 0);
}

int vk_graph_clip_i32(
        const int32_t* input, int32_t* output, long elements,
        int32_t minimum, int32_t maximum) {
    return vk_graph_unary_i32(
        input, output, elements, VX_TYPED_CONTROL_CLIP_I32,
        minimum, maximum);
}

int vk_graph_copy_32(const void* input, void* output, long elements) {
    VxTypedControlMetadata metadata;
    size_t bytes;
    if (!vx_typed_control_copy_plan(
            input, output, elements, &metadata, &bytes))
        return 0;
    return vk_graph_typed_control_32(
        input, bytes, input, bytes, output, bytes, &metadata);
}

int vk_graph_cast_typed(
        const void* input, int input_dtype,
        void* output, int output_dtype, long elements) {
    VxTypedControlMetadata metadata;
    size_t bytes;
    if (!vx_typed_control_cast_plan(
            input, input_dtype, output, output_dtype, elements,
            &metadata, &bytes))
        return 0;
    return vk_graph_typed_control_32(
        input, bytes, input, bytes, output, bytes, &metadata);
}

int vk_graph_sigmoid_f32(const float* in, float* out, long n) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[1] = {(uint32_t)n};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, bytes}, {dst->offset, bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_sigmoid, binds, ((uint32_t)n + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

static int vk_graph_unary_size_f32(VkKernel* k, const float* in, float* out, long n) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[1] = {(uint32_t)n};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, bytes}, {dst->offset, bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(k, binds, ((uint32_t)n + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_relu_f32(const float* in, float* out, long n) { return vk_graph_unary_size_f32(&k_relu, in, out, n); }
int vk_graph_gelu_f32(const float* in, float* out, long n, int approximate_tanh) {
    if (n <= 0 || (uint64_t)n > UINT32_MAX || !in || !out ||
        (approximate_tanh != 0 && approximate_tanh != 1)) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)n, (uint32_t)approximate_tanh, 0u, 0u};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, bytes}, {dst->offset, bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_gelu, binds, ((uint32_t)n + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}
int vk_graph_silu_f32(const float* in, float* out, long n) { return vk_graph_unary_size_f32(&k_silu, in, out, n); }
int vk_graph_tanh_f32(const float* in, float* out, long n) { return vk_graph_unary_size_f32(&k_tanh, in, out, n); }
int vk_graph_hardswish_f32(const float* in, float* out, long n) { return vk_graph_unary_size_f32(&k_hardswish, in, out, n); }
int vk_graph_hardsigmoid_f32(const float* in, float* out, long n) { return vk_graph_unary_size_f32(&k_hardsigmoid, in, out, n); }
int vk_graph_cast_copy_f32(const float* in, float* out, long n) { return vk_graph_copy_f32(in, out, n); }

int vk_graph_leaky_relu_f32(const float* in, float* out, long n, float alpha) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    struct { uint32_t size; float alpha; } params = {(uint32_t)n, alpha};
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, bytes}, {dst->offset, bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_leaky_relu, binds, ((uint32_t)n + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_prelu_f32(const float* in, const float* weight, float* out, long n, int channels) {
    if (n <= 0 || channels <= 0 || !in || !weight || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    size_t wbytes = (size_t)channels * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    VkTensorSlot* w = graph_ensure_device(weight, wbytes, 1);
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !w || !dst) return 0;
    uint32_t params[4] = {(uint32_t)n, (uint32_t)channels, (uint32_t)channels, 0u};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {src->offset, bytes}, {w->offset, wbytes}, {dst->offset, bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_prelu, binds, ((uint32_t)n + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_layernorm_f32(const float* in, const float* weight, const float* bias,
                           float* out, int rows, int d_model, float eps) {
    if (rows <= 0 || d_model <= 0 || !in || !weight || !bias || !out ||
        !(eps > 0.0f) || !isfinite(eps)) return 0;
    size_t bytes = (size_t)rows * d_model * sizeof(float);
    size_t wbytes = (size_t)d_model * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    VkTensorSlot* w = graph_ensure_device(weight, wbytes, 1);
    VkTensorSlot* b = graph_ensure_device(bias, wbytes, 1);
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !w || !b || !dst) return 0;
    struct {
        uint32_t rows;
        uint32_t d_model;
        float eps;
        uint32_t pad;
    } params = {(uint32_t)rows, (uint32_t)d_model, eps, 0u};
    _Static_assert(sizeof(params) == 16, "LayerNorm uniform ABI");
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[5] = {
        {src->offset, bytes}, {w->offset, wbytes}, {b->offset, wbytes},
        {dst->offset, bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_layernorm, binds, ((uint32_t)rows + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_rmsnorm_f32(const float* in, const float* weight, float* out,
                         int rows, int d_model, float eps) {
    if (rows <= 0 || d_model <= 0 || !in || !weight || !out) return 0;
    size_t bytes = (size_t)rows * d_model * sizeof(float);
    size_t wbytes = (size_t)d_model * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    VkTensorSlot* w = graph_ensure_device(weight, wbytes, 1);
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !w || !dst) return 0;
    struct { uint32_t rows; uint32_t d_model; float eps; } params = {(uint32_t)rows, (uint32_t)d_model, eps};
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {src->offset, bytes}, {w->offset, wbytes}, {dst->offset, bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_rmsnorm, binds, ((uint32_t)rows + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

static int vk_graph_softmax_like_f32(VkKernel* k, const float* in, float* out, int rows, int d) {
    if (rows <= 0 || d <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)rows * d * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[2] = {(uint32_t)rows, (uint32_t)d};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, bytes}, {dst->offset, bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(k, binds, ((uint32_t)rows + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_softmax_f32(const float* in, float* out, int rows, int d) {
    return vk_graph_softmax_like_f32(&k_softmax, in, out, rows, d);
}

int vk_graph_logsoftmax_f32(const float* in, float* out, int rows, int d) {
    return vk_graph_softmax_like_f32(&k_logsoftmax, in, out, rows, d);
}

int vk_graph_reduce_f32(const float* in, float* out, int rows, int d, float inv) {
    if (rows <= 0 || d <= 0 || !in || !out) return 0;
    size_t in_bytes = (size_t)rows * d * sizeof(float);
    size_t out_bytes = (size_t)rows * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    struct { uint32_t rows; uint32_t d; float inv; } params = {(uint32_t)rows, (uint32_t)d, inv};
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_reduce, binds, ((uint32_t)rows + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_global_average_pool_f32(const float* in, float* out, int n, int h, int w, int c) {
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || !in || !out) return 0;
    size_t in_bytes = (size_t)n * h * w * c * sizeof(float);
    size_t out_bytes = (size_t)n * c * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_globalavg, binds, ((uint32_t)c + 63u) / 64u, (uint32_t)n, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_average_pool2d_f32(const float* in, float* out, int n, int h, int w, int c,
                                int out_h, int out_w, int ky, int kx, int sy, int sx,
                                int py, int px) {
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || out_h <= 0 || out_w <= 0 ||
        ky <= 0 || kx <= 0 || sy <= 0 || sx <= 0 || !in || !out) return 0;
    size_t in_bytes = (size_t)n * h * w * c * sizeof(float);
    size_t out_bytes = (size_t)n * out_h * out_w * c * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[12] = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c,
                           (uint32_t)out_h, (uint32_t)out_w, (uint32_t)ky, (uint32_t)kx,
                           (uint32_t)sy, (uint32_t)sx, (uint32_t)py, (uint32_t)px};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_avgpool, binds, ((uint32_t)out_w + 7u) / 8u,
                            ((uint32_t)out_h + 7u) / 8u, (uint32_t)(n * c))) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_batchnorm2d_f32(const float* in, const float* weight, const float* bias,
                             const float* mean, const float* var, float* out,
                             int n, int h, int w, int c, float eps) {
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || !in || !weight || !bias || !mean || !var || !out) return 0;
    size_t bytes = (size_t)n * h * w * c * sizeof(float);
    size_t cbytes = (size_t)c * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    VkTensorSlot* sw = graph_ensure_device(weight, cbytes, 1);
    VkTensorSlot* sb = graph_ensure_device(bias, cbytes, 1);
    VkTensorSlot* sm = graph_ensure_device(mean, cbytes, 1);
    VkTensorSlot* sv = graph_ensure_device(var, cbytes, 1);
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !sw || !sb || !sm || !sv || !dst) return 0;
    struct { uint32_t n; uint32_t c; uint32_t h; uint32_t w; float eps; uint32_t pad[3]; } params =
        {(uint32_t)n, (uint32_t)c, (uint32_t)h, (uint32_t)w, eps, {0, 0, 0}};
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[7] = {
        {src->offset, bytes}, {sw->offset, cbytes}, {sb->offset, cbytes},
        {sm->offset, cbytes}, {sv->offset, cbytes}, {dst->offset, bytes},
        {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_batchnorm, binds, ((uint32_t)(n * h * w * c) + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_groupnorm_f32(const float* in, const float* weight, const float* bias,
                           float* out, int n, int h, int w, int c, int groups,
                           float eps) {
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || groups <= 0 || c % groups ||
        !in || !weight || !bias || !out || !(eps > 0.0f)) return 0;
    size_t bytes = (size_t)n * h * w * c * sizeof(float);
    size_t cbytes = (size_t)c * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    VkTensorSlot* sw = graph_ensure_device(weight, cbytes, 1);
    VkTensorSlot* sb = graph_ensure_device(bias, cbytes, 1);
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !sw || !sb || !dst) return 0;
    struct {
        uint32_t n, h, w, c, groups, has_bias;
        float eps;
        uint32_t pad;
    } params = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c,
                (uint32_t)groups, 1u, eps, 0u};
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[5] = {
        {src->offset, bytes}, {sw->offset, cbytes}, {sb->offset, cbytes},
        {dst->offset, bytes}, {p_off, sizeof(params)}
    };
    uint32_t total_groups = (uint32_t)n * (uint32_t)groups;
    if (!vk_dispatch_kernel(&k_groupnorm, binds, (total_groups + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

#if VOLVOXAI_ENABLE_TRAINING
int vk_graph_dropout_f32(const float* in, float* out, long n, uint32_t threshold,
                         uint32_t seed, uint32_t counter, float scale) {
    if (n <= 0 || (uint64_t)n > UINT32_MAX || !in || !out || !(scale >= 1.0f)) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    struct {
        uint32_t length, threshold, seed, counter;
        float scale;
        uint32_t pad[3];
    } params = {(uint32_t)n, threshold, seed, counter, scale, {0u, 0u, 0u}};
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, bytes}, {dst->offset, bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_dropout, binds, ((uint32_t)n + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}
#endif

int vk_graph_embedding_f32(const int32_t* tokens, const float* weight, float* out,
                           int tokens_len, int d_model, int vocab_size) {
    if (tokens_len <= 0 || d_model <= 0 || vocab_size <= 0 || !tokens || !weight || !out) return 0;
    size_t tbytes = (size_t)tokens_len * sizeof(int32_t);
    size_t wbytes = (size_t)vocab_size * d_model * sizeof(float);
    size_t obytes = (size_t)tokens_len * d_model * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* st = graph_ensure_device(tokens, tbytes, 0);
    VkTensorSlot* sw = graph_ensure_device(weight, wbytes, 1);
    VkTensorSlot* so = graph_output_slot(out, obytes);
    if (!st || !sw || !so) return 0;
    uint32_t params[4] = {(uint32_t)tokens_len, (uint32_t)d_model,
                          (uint32_t)vocab_size, 0u};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {st->offset, tbytes}, {sw->offset, wbytes}, {so->offset, obytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_embedding, binds, ((uint32_t)tokens_len + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(so);
    return 1;
}

/* Routed expert linear. `experts` counts the staged rows of a possibly
 * partially resident bank; route indices stay in global slot space and the
 * shader maps them through slot_rows when slot_domain is non-zero. */
int vk_graph_moe_linear_f32(const float* input, const float* expert_weight,
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
    size_t slot_bytes = (size_t)(slot_domain ? slot_domain : 1u) * sizeof(uint32_t);
    graph_scratch_begin();
    VkTensorSlot* si = graph_ensure_device(input, in_bytes, 0);
    VkTensorSlot* sw = graph_ensure_device(expert_weight, w_bytes, 1);
    VkTensorSlot* sri = graph_ensure_device(route_indices, route_bytes, 0);
    VkTensorSlot* srw = graph_ensure_device(route_weights, route_bytes, 0);
    VkTensorSlot* so = graph_output_slot(out, out_bytes);
    if (!si || !sw || !sri || !srw || !so) return 0;
    /* The shader always binds a bias and a slot table; absent ones become
     * scratch zeros so the descriptor set stays complete. */
    size_t bias_off;
    if (expert_bias) {
        VkTensorSlot* sb = graph_ensure_device(expert_bias, bias_bytes, 1);
        if (!sb) return 0;
        bias_off = sb->offset;
    } else {
        bias_off = graph_scratch_zero(bias_bytes);
        if (bias_off == SIZE_MAX) return 0;
    }
    size_t slot_off = slot_domain
        ? graph_scratch_upload(slot_rows, slot_bytes)
        : graph_scratch_zero(slot_bytes);
    if (slot_off == SIZE_MAX) return 0;
    uint32_t params[8] = {
        (uint32_t)rows, (uint32_t)d_in, (uint32_t)d_out, (uint32_t)experts,
        (uint32_t)top_k, expert_bias ? 1u : 0u, slot_domain, 0u,
    };
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[8] = {
        {si->offset, in_bytes}, {sw->offset, w_bytes}, {bias_off, bias_bytes},
        {sri->offset, route_bytes}, {srw->offset, route_bytes},
        {so->offset, out_bytes}, {p_off, sizeof(params)}, {slot_off, slot_bytes},
    };
    if (!vk_dispatch_kernel(&k_moe_linear, binds,
                            ((uint32_t)d_out + 63u) / 64u, (uint32_t)rows, 1)) return 0;
    graph_mark_device(so);
    return 1;
}

/* Top-k expert routing. The router weight is [d_model, experts], so its expert
 * axis is 1 and it is never a slot-indexed bank. */
int vk_graph_moe_router_f32(const float* input, const float* weight,
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
    graph_scratch_begin();
    VkTensorSlot* si = graph_ensure_device(input, in_bytes, 0);
    VkTensorSlot* sw = graph_ensure_device(weight, w_bytes, 1);
    VkTensorSlot* sidx = graph_output_slot(route_indices, route_bytes);
    VkTensorSlot* srw = graph_output_slot(route_weights, route_bytes);
    if (!si || !sw || !sidx || !srw) return 0;
    size_t bias_off;
    if (bias) {
        VkTensorSlot* sb = graph_ensure_device(bias, bias_bytes, 1);
        if (!sb) return 0;
        bias_off = sb->offset;
    } else {
        bias_off = graph_scratch_zero(bias_bytes);
        if (bias_off == SIZE_MAX) return 0;
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
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[6] = {
        {si->offset, in_bytes}, {sw->offset, w_bytes}, {bias_off, bias_bytes},
        {sidx->offset, route_bytes}, {srw->offset, route_bytes},
        {p_off, sizeof(params)},
    };
    if (!vk_dispatch_kernel(&k_moe_router, binds,
                            ((uint32_t)rows + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(sidx);
    graph_mark_device(srw);
    return 1;
}

int vk_graph_transpose_f32(const float* in, float* out, const int* in_shape,
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
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t meta[2 + 16] = {0};
    meta[0] = (uint32_t)total;
    meta[1] = (uint32_t)rank;
    for (int d = 0; d < rank; d++) {
        meta[2 + d] = out_stride[d];
        meta[2 + rank + d] = in_stride[perm[d]];
    }
    size_t mbytes = (size_t)(2 + 2 * rank) * sizeof(uint32_t);
    size_t m_off = graph_scratch_upload(meta, mbytes);
    if (m_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, bytes}, {dst->offset, bytes}, {m_off, mbytes}
    };
    if (!vk_dispatch_kernel(&k_transpose, binds, ((uint32_t)total + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_where_f32(const float* cond, const float* a, const float* b, float* out, long n) {
    if (n <= 0 || !cond || !a || !b || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* sc = graph_ensure_device(cond, bytes, 0);
    VkTensorSlot* sa = graph_ensure_device(a, bytes, 0);
    VkTensorSlot* sb = graph_ensure_device(b, bytes, 0);
    VkTensorSlot* so = graph_output_slot(out, bytes);
    if (!sc || !sa || !sb || !so) return 0;
    uint32_t params[1] = {(uint32_t)n};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[5] = {
        {sc->offset, bytes}, {sa->offset, bytes}, {sb->offset, bytes}, {so->offset, bytes},
        {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_where, binds, ((uint32_t)n + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(so);
    return 1;
}

int vk_graph_where_32(
        const int32_t* condition, const void* a,
        const void* b, void* output, long elements) {
    uint32_t count;
    size_t bytes;
    if (!vx_typed_control_where_plan(
            condition, a, b, output, elements, &count, &bytes))
        return 0;
    graph_scratch_begin();
    VkTensorSlot* condition_slot =
        graph_ensure_device(condition, bytes, 0);
    VkTensorSlot* a_slot = graph_ensure_device(a, bytes, 0);
    VkTensorSlot* b_slot = graph_ensure_device(b, bytes, 0);
    VkTensorSlot* output_slot = graph_output_slot(output, bytes);
    if (!condition_slot || !a_slot || !b_slot || !output_slot) return 0;
    uint32_t params[4] = {count, 0u, 0u, 0u};
    size_t params_offset = graph_scratch_upload(params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding bindings[5] = {
        {condition_slot->offset, bytes},
        {a_slot->offset, bytes},
        {b_slot->offset, bytes},
        {output_slot->offset, bytes},
        {params_offset, sizeof(params)},
    };
    if (!vk_dispatch_kernel(
            &k_where_32, bindings, (count + 63u) / 64u, 1u, 1u))
        return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_argmax_f32(
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
    graph_scratch_begin();
    VkTensorSlot* input_slot =
        graph_ensure_device(input, input_bytes, 0);
    VkTensorSlot* output_slot =
        graph_output_slot(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    uint32_t params[4] = {outer, axis_size, inner, 0u};
    size_t params_offset = graph_scratch_upload(params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding bindings[3] = {
        {input_slot->offset, input_bytes},
        {output_slot->offset, output_bytes},
        {params_offset, sizeof(params)},
    };
    if (!vk_dispatch_kernel(
            &k_argmax_f32_i32, bindings,
            (output_elements + 63u) / 64u, 1u, 1u))
        return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_upsample2x_f32(const float* in, float* out, int n, int h, int w, int c) {
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0 || !in || !out) return 0;
    size_t in_bytes = (size_t)n * h * w * c * sizeof(float);
    size_t out_bytes = (size_t)n * h * 2 * w * 2 * c * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_upsample, binds, ((uint32_t)(w * 2) + 7u) / 8u,
                            ((uint32_t)(h * 2) + 7u) / 8u, (uint32_t)(n * c))) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_resize_nearest_f32(const float* in, float* out, int n, int h, int w, int c,
                                int out_h, int out_w) {
    return vk_graph_resize_f32(in, out, n, h, w, c, out_h, out_w, 0);
}

int vk_graph_resize_f32(const float* in, float* out, int n, int h, int w, int c,
                        int out_h, int out_w, int mode) {
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0 || out_h <= 0 || out_w <= 0 || !in || !out ||
        (uint64_t)(uint32_t)n * (uint32_t)c > UINT32_MAX) return 0;
    size_t in_bytes = (size_t)n * h * w * c * sizeof(float);
    size_t out_bytes = (size_t)n * out_h * out_w * c * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[8] = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c,
                          (uint32_t)out_h, (uint32_t)out_w, (uint32_t)mode, 0u};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_resize, binds, ((uint32_t)out_w + 7u) / 8u,
                            ((uint32_t)out_h + 7u) / 8u, (uint32_t)(n * c))) return 0;
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

int vk_graph_expand_f32(const float* in, float* out, const int* in_shape, int in_rank,
                        const int* out_shape, int out_rank) {
    VxExpandF32Plan plan;
    if (!vx_expand_f32_plan(
            in, out, in_shape, in_rank, out_shape, out_rank, &plan))
        return 0;
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, plan.input_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, plan.output_bytes);
    if (!src || !dst) return 0;
    size_t p_off = graph_scratch_upload(plan.params, sizeof(plan.params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, plan.input_bytes},
        {dst->offset, plan.output_bytes},
        {p_off, sizeof(plan.params)}
    };
    if (!vk_dispatch_kernel(
            &k_expand, binds, (plan.output_elements + 63u) / 64u, 1u, 1u))
        return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_expand_32(const void* in, void* out, const int* in_shape,
                       int in_rank, const int* out_shape, int out_rank) {
    /* expand.wgsl copies u32 words, so the F32 host path is also the exact
     * bit-preserving I32 implementation. */
    return vk_graph_expand_f32(
        (const float*)in, (float*)out,
        in_shape, in_rank, out_shape, out_rank);
}

int vk_graph_batch_matmul_f32(
        const float* a, const int* a_shape, int a_rank,
        const float* b, const int* b_shape, int b_rank,
        float* output, const int* output_shape, int output_rank) {
    VxBatchMatMulF32Plan plan;
    size_t metadata_bytes;
    size_t metadata_offset;
    if (!vx_batch_matmul_f32_plan(
            a, a_shape, a_rank, b, b_shape, b_rank,
            output, output_shape, output_rank, &plan))
        return 0;
    metadata_bytes = (size_t)plan.metadata_words * sizeof(uint32_t);
    graph_scratch_begin();
    VkTensorSlot* a_slot = graph_ensure_device(a, plan.a_bytes, 0);
    VkTensorSlot* b_slot = graph_ensure_device(b, plan.b_bytes, 0);
    VkTensorSlot* output_slot =
        graph_output_slot(output, plan.output_bytes);
    if (!a_slot || !b_slot || !output_slot) return 0;
    metadata_offset = graph_scratch_upload(plan.metadata, metadata_bytes);
    if (metadata_offset == SIZE_MAX) return 0;
    VkGraphBinding bindings[4] = {
        {a_slot->offset, plan.a_bytes},
        {b_slot->offset, plan.b_bytes},
        {output_slot->offset, plan.output_bytes},
        {metadata_offset, metadata_bytes},
    };
    if (!vk_dispatch_kernel(
            &k_batch_matmul, bindings,
            (plan.n + 7u) / 8u, (plan.m + 7u) / 8u,
            plan.output_batches))
        return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_gather_i32_f32(const float* input, const int32_t* indices,
                            float* output, int outer, int axis_size, int inner,
                            int indices_elements, int output_elements) {
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
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(input, input_bytes, 0);
    VkTensorSlot* idx = graph_ensure_device(indices, index_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(output, output_bytes);
    if (!src || !idx || !dst) return 0;
    uint32_t params[5] = {
        (uint32_t)outer, (uint32_t)axis_size, (uint32_t)inner,
        (uint32_t)indices_elements, (uint32_t)output_elements,
    };
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {src->offset, input_bytes}, {idx->offset, index_bytes},
        {dst->offset, output_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_gather, binds,
                            ((uint32_t)output_elements + 63u) / 64u,
                            1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_pad4d_f32(const float* in, float* out, const int* in_shape, int in_rank,
                       const int* out_shape, int out_rank, int pad_top, int pad_left, float value) {
    if (!in || !out || !in_shape || !out_shape || in_rank <= 0 || out_rank <= 0 || in_rank > 4 || out_rank > 4) return 0;
    uint32_t is[4], os[4];
    pad4_shape(in_shape, in_rank, is);
    pad4_shape(out_shape, out_rank, os);
    size_t in_elems = (size_t)is[0] * is[1] * is[2] * is[3];
    size_t out_elems = (size_t)os[0] * os[1] * os[2] * os[3];
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_elems * sizeof(float), 0);
    VkTensorSlot* dst = graph_output_slot(out, out_elems * sizeof(float));
    if (!src || !dst) return 0;
    struct {
        uint32_t b, in_h, in_w, c, out_h, out_w, pt, pl;
        float val;
        uint32_t pad[3];
    } params = {is[0], is[1], is[2], is[3], os[1], os[2], (uint32_t)pad_top, (uint32_t)pad_left, value, {0, 0, 0}};
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, in_elems * sizeof(float)}, {dst->offset, out_elems * sizeof(float)}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_pad, binds, ((uint32_t)out_elems + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_slice4d_f32(const float* in, float* out, const int* in_shape, int in_rank,
                         const int* out_shape, int out_rank, const int* starts,
                         const int* steps) {
    if (!in || !out || !in_shape || !out_shape || !starts || !steps ||
        in_rank <= 0 || out_rank <= 0 || in_rank > 4 || out_rank > 4) return 0;
    uint32_t is[4], os[4];
    pad4_shape(in_shape, in_rank, is);
    pad4_shape(out_shape, out_rank, os);
    size_t in_elems = (size_t)is[0] * is[1] * is[2] * is[3];
    size_t out_elems = (size_t)os[0] * os[1] * os[2] * os[3];
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_elems * sizeof(float), 0);
    VkTensorSlot* dst = graph_output_slot(out, out_elems * sizeof(float));
    if (!src || !dst) return 0;
    uint32_t params[16] = {
        os[0], os[1], os[2], os[3], is[1], is[2], is[3],
        (uint32_t)starts[0], (uint32_t)starts[1], (uint32_t)starts[2], (uint32_t)starts[3],
        (uint32_t)steps[0], (uint32_t)steps[1], (uint32_t)steps[2], (uint32_t)steps[3],
        (uint32_t)out_elems
    };
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, in_elems * sizeof(float)}, {dst->offset, out_elems * sizeof(float)}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_slice, binds, ((uint32_t)out_elems + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_conv_transpose2d_f32(const float* in, const float* weight, const float* bias,
                                  float* out, int n, int in_h, int in_w, int in_c,
                                  int out_h, int out_w, int out_c, int kh, int kw,
                                  int sh, int sw, int ph, int pw) {
    if (n <= 0 || in_h <= 0 || in_w <= 0 || in_c <= 0 || out_h <= 0 || out_w <= 0 ||
        out_c <= 0 || kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0 || !in || !weight || !out) return 0;
    size_t in_bytes = (size_t)n * in_h * in_w * in_c * sizeof(float);
    size_t wbytes = (size_t)in_c * out_c * kh * kw * sizeof(float);
    size_t bbytes = (size_t)out_c * sizeof(float);
    size_t out_bytes = (size_t)n * out_h * out_w * out_c * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* swt = graph_ensure_device(weight, wbytes, 1);
    static const float zero_bias[1] = {0.0f};
    VkTensorSlot* sb = graph_ensure_device(bias ? bias : zero_bias, bias ? bbytes : sizeof(float), 1);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !swt || !sb || !dst) return 0;
    uint32_t params[16] = {(uint32_t)n, (uint32_t)in_h, (uint32_t)in_w, (uint32_t)in_c,
                           (uint32_t)out_h, (uint32_t)out_w, (uint32_t)out_c,
                           (uint32_t)kh, (uint32_t)kw, (uint32_t)sh, (uint32_t)sw,
                           (uint32_t)ph, (uint32_t)pw, bias ? 1u : 0u, 0u, 0u};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[5] = {
        {src->offset, in_bytes}, {swt->offset, wbytes}, {sb->offset, bias ? bbytes : sizeof(float)},
        {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_convtranspose, binds, ((uint32_t)out_w + 7u) / 8u,
                            ((uint32_t)out_h + 7u) / 8u, (uint32_t)(n * out_c))) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_interp1d_f32(const float* in, float* out, int channels, int in_l, int out_l) {
    if (channels <= 0 || in_l <= 0 || out_l <= 0 || !in || !out) return 0;
    size_t in_bytes = (size_t)channels * in_l * sizeof(float);
    size_t out_bytes = (size_t)channels * out_l * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[3] = {(uint32_t)channels, (uint32_t)in_l, (uint32_t)out_l};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_interp1d, binds, ((uint32_t)out_l + 63u) / 64u, (uint32_t)channels, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_binary_f32(const float* a, long a_numel, const float* b, long b_numel,
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
    graph_scratch_begin();
    VkTensorSlot* sa = graph_ensure_device(a, a_bytes, 0);
    VkTensorSlot* sb = graph_ensure_device(b, b_bytes, 0);
    VkTensorSlot* so = graph_output_slot(out, out_bytes);
    if (!sa || !sb || !so) return 0;
    uint32_t metadata[28] = {(uint32_t)out_numel, (uint32_t)rank, (uint32_t)op, 0u};
    memcpy(metadata + 4, output_strides, 8 * sizeof(uint32_t));
    memcpy(metadata + 12, a_strides, 8 * sizeof(uint32_t));
    memcpy(metadata + 20, b_strides, 8 * sizeof(uint32_t));
    size_t p_off = graph_scratch_upload(metadata, sizeof(metadata));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {sa->offset, a_bytes}, {sb->offset, b_bytes}, {so->offset, out_bytes}, {p_off, sizeof(metadata)}
    };
    if (!vk_dispatch_kernel(&k_broadcast_binary, binds, ((uint32_t)out_numel + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(so);
    return 1;
}

int vk_graph_split_f32(const float* in, float* out, long input_numel, long output_numel,
                       int inner, int split_size, int axis_in, int offset) {
    if (!in || !out || input_numel <= 0 || output_numel <= 0 ||
        inner <= 0 || split_size <= 0 || axis_in <= 0 || offset < 0 || offset + split_size > axis_in) return 0;
    size_t in_bytes = (size_t)input_numel * sizeof(float);
    size_t out_bytes = (size_t)output_numel * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[5] = {(uint32_t)output_numel, (uint32_t)inner, (uint32_t)split_size,
                          (uint32_t)axis_in, (uint32_t)offset};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {{src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}};
    if (!vk_dispatch_kernel(&k_split, binds, ((uint32_t)output_numel + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_conv1d_f32(const float* in, const float* weight, const float* bias, float* out,
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
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* sw = graph_ensure_device(weight, wbytes, 1);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !sw || !dst) return 0;
    size_t b_off = SIZE_MAX;
    if (bias) {
        VkTensorSlot* sb = graph_ensure_device(bias, bbytes, 1);
        if (!sb) return 0;
        b_off = sb->offset;
    } else {
        float* zeros = (float*)calloc((size_t)out_c, sizeof(float));
        if (!zeros) return 0;
        b_off = graph_scratch_upload(zeros, bbytes);
        free(zeros);
        if (b_off == SIZE_MAX) return 0;
    }
    uint32_t params[12] = {(uint32_t)in_c, (uint32_t)in_l, (uint32_t)out_c,
                           (uint32_t)kernel, (uint32_t)stride, (uint32_t)pad,
                           (uint32_t)relu, (uint32_t)batch, 1u, (uint32_t)in_c,
                           (uint32_t)out_l, 0u};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[5] = {
        {src->offset, in_bytes}, {sw->offset, wbytes}, {b_off, bbytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_conv1d, binds, ((uint32_t)out_l + 63u) / 64u,
                            (uint32_t)out_c, (uint32_t)batch)) return 0;
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

int vk_graph_sdpa_f32(const float* qkv, const int32_t* mask, long mask_numel,
                      float* out, int seq_len, int d_model, int num_heads,
                      int head_dim, int batch, float scale, int causal, int mask_mode) {
    if (!qkv || !out || seq_len <= 0 || d_model <= 0 || num_heads <= 0 || head_dim <= 0 ||
        batch <= 0 || head_dim > 64 || d_model != num_heads * head_dim) return 0;
    long expected_mask = attention_mask_numel(mask_mode, batch, seq_len, seq_len);
    if (expected_mask < 0 || (mask_mode != 0 && (!mask || mask_numel != expected_mask))) return 0;
    size_t qkv_bytes = (size_t)batch * (size_t)seq_len * 3u * (size_t)d_model * sizeof(float);
    size_t out_bytes = (size_t)batch * (size_t)seq_len * (size_t)d_model * sizeof(float);
    size_t mask_bytes = mask_mode ? (size_t)expected_mask * sizeof(int32_t) : qkv_bytes;
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(qkv, qkv_bytes, 0);
    VkTensorSlot* sm = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) : src;
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !sm || !dst) return 0;
    struct {
        uint32_t seq_len, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode;
    } params = {(uint32_t)seq_len, (uint32_t)d_model, (uint32_t)num_heads,
                (uint32_t)head_dim, (uint32_t)batch, scale,
                causal ? 1u : 0u, (uint32_t)mask_mode};
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {src->offset, qkv_bytes}, {sm->offset, mask_bytes},
        {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_sdpa, binds, ((uint32_t)seq_len + 63u) / 64u,
                            (uint32_t)num_heads, (uint32_t)batch)) return 0;
    graph_mark_device(dst);
    return 1;
}

#if VOLVOXAI_ENABLE_TRAINING
int vk_graph_sdpa_training_f32(const float* qkv, const int32_t* mask, long mask_numel,
                               float* out, int seq_len, int d_model, int num_heads,
                               int head_dim, int batch, float scale, int causal, int mask_mode,
                               uint32_t threshold, uint32_t seed, uint32_t counter,
                               float dropout_scale) {
    if (!qkv || !out || seq_len <= 0 || d_model <= 0 || num_heads <= 0 || head_dim <= 0 ||
        batch <= 0 || head_dim > 64 || d_model != num_heads * head_dim ||
        !isfinite(dropout_scale) || dropout_scale < 1.0f) return 0;
    long expected_mask = attention_mask_numel(mask_mode, batch, seq_len, seq_len);
    if (expected_mask < 0 || (mask_mode != 0 && (!mask || mask_numel != expected_mask))) return 0;
    size_t qkv_bytes = (size_t)batch * (size_t)seq_len * 3u * (size_t)d_model * sizeof(float);
    size_t out_bytes = (size_t)batch * (size_t)seq_len * (size_t)d_model * sizeof(float);
    size_t mask_bytes = mask_mode ? (size_t)expected_mask * sizeof(int32_t) : qkv_bytes;
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(qkv, qkv_bytes, 0);
    VkTensorSlot* sm = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) : src;
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !sm || !dst) return 0;
    struct {
        uint32_t seq_len, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, threshold, seed, counter;
        float dropout_scale;
    } params = {(uint32_t)seq_len, (uint32_t)d_model, (uint32_t)num_heads,
                (uint32_t)head_dim, (uint32_t)batch, scale, causal ? 1u : 0u,
                (uint32_t)mask_mode, threshold, seed, counter, dropout_scale};
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {src->offset, qkv_bytes}, {sm->offset, mask_bytes},
        {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_sdpa_training, binds, ((uint32_t)seq_len + 63u) / 64u,
                            (uint32_t)num_heads, (uint32_t)batch)) return 0;
    graph_mark_device(dst);
    return 1;
}
#endif

int vk_graph_cross_sdpa_f32(const float* q, const float* k, const float* v,
                            const int32_t* mask, long mask_numel, float* out,
                            int seq_q, int seq_kv, int d_model, int num_heads,
                            int head_dim, int batch, float scale, int causal, int mask_mode) {
    if (!q || !k || !v || !out || seq_q <= 0 || seq_kv <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 || batch <= 0 || head_dim > 64 ||
        d_model != num_heads * head_dim) return 0;
    long expected_mask = attention_mask_numel(mask_mode, batch, seq_q, seq_kv);
    if (expected_mask < 0 || (mask_mode != 0 && (!mask || mask_numel != expected_mask))) return 0;
    size_t q_bytes = (size_t)batch * (size_t)seq_q * (size_t)d_model * sizeof(float);
    size_t kv_bytes = (size_t)batch * (size_t)seq_kv * (size_t)d_model * sizeof(float);
    size_t out_bytes = q_bytes;
    size_t mask_bytes = mask_mode ? (size_t)expected_mask * sizeof(int32_t) : q_bytes;
    graph_scratch_begin();
    VkTensorSlot* sq = graph_ensure_device(q, q_bytes, 0);
    VkTensorSlot* sk = graph_ensure_device(k, kv_bytes, 0);
    VkTensorSlot* sv = graph_ensure_device(v, kv_bytes, 0);
    VkTensorSlot* sm = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) : sq;
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sq || !sk || !sv || !sm || !dst) return 0;
    struct {
        uint32_t seq_q, seq_kv, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, pad[3];
    } params = {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model,
                (uint32_t)num_heads, (uint32_t)head_dim, (uint32_t)batch, scale,
                causal ? 1u : 0u, (uint32_t)mask_mode, {0u, 0u, 0u}};
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[6] = {
        {sq->offset, q_bytes}, {sk->offset, kv_bytes}, {sv->offset, kv_bytes},
        {sm->offset, mask_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_cross_sdpa, binds, ((uint32_t)seq_q + 63u) / 64u,
                            (uint32_t)num_heads, (uint32_t)batch)) return 0;
    graph_mark_device(dst);
    return 1;
}

#if VOLVOXAI_ENABLE_TRAINING
int vk_graph_cross_sdpa_training_f32(const float* q, const float* k, const float* v,
                                     const int32_t* mask, long mask_numel, float* out,
                                     int seq_q, int seq_kv, int d_model, int num_heads,
                                     int head_dim, int batch, float scale, int causal, int mask_mode,
                                     uint32_t threshold, uint32_t seed, uint32_t counter,
                                     float dropout_scale) {
    if (!q || !k || !v || !out || seq_q <= 0 || seq_kv <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 || batch <= 0 || head_dim > 64 ||
        d_model != num_heads * head_dim || !isfinite(dropout_scale) || dropout_scale < 1.0f) return 0;
    long expected_mask = attention_mask_numel(mask_mode, batch, seq_q, seq_kv);
    if (expected_mask < 0 || (mask_mode != 0 && (!mask || mask_numel != expected_mask))) return 0;
    size_t q_bytes = (size_t)batch * (size_t)seq_q * (size_t)d_model * sizeof(float);
    size_t kv_bytes = (size_t)batch * (size_t)seq_kv * (size_t)d_model * sizeof(float);
    size_t out_bytes = q_bytes;
    size_t mask_bytes = mask_mode ? (size_t)expected_mask * sizeof(int32_t) : q_bytes;
    graph_scratch_begin();
    VkTensorSlot* sq = graph_ensure_device(q, q_bytes, 0);
    VkTensorSlot* sk = graph_ensure_device(k, kv_bytes, 0);
    VkTensorSlot* sv = graph_ensure_device(v, kv_bytes, 0);
    VkTensorSlot* sm = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) : sq;
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
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
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[6] = {
        {sq->offset, q_bytes}, {sk->offset, kv_bytes}, {sv->offset, kv_bytes},
        {sm->offset, mask_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_cross_sdpa_training, binds, ((uint32_t)seq_q + 63u) / 64u,
                            (uint32_t)num_heads, (uint32_t)batch)) return 0;
    graph_mark_device(dst);
    return 1;
}
#endif

int vk_graph_cross_attention_f32(const float* q, const float* kv, const float* weight,
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
    graph_scratch_begin();
    VkTensorSlot* sq = graph_ensure_device(q, q_bytes, 0);
    VkTensorSlot* skv = graph_ensure_device(kv, kv_bytes, 0);
    VkTensorSlot* sw = graph_ensure_device(weight, wbytes, 1);
    VkTensorSlot* ss = graph_ensure_device(has_scale && scale ? scale : zero, has_scale && scale ? sb_bytes : sizeof(float), 1);
    VkTensorSlot* sb = graph_ensure_device(has_bias && bias ? bias : zero, has_bias && bias ? sb_bytes : sizeof(float), 1);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sq || !skv || !sw || !ss || !sb || !dst) return 0;
    struct {
        uint32_t seq_q, seq_kv, d_model, num_heads, head_dim;
        float scale_factor;
        uint32_t has_scale, has_bias, batch, pad[3];
    } params = {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model, (uint32_t)num_heads,
                (uint32_t)head_dim, 1.0f / sqrtf((float)head_dim),
                (uint32_t)(has_scale && scale), (uint32_t)(has_bias && bias), (uint32_t)batch, {0u, 0u, 0u}};
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[7] = {
        {sq->offset, q_bytes}, {skv->offset, kv_bytes}, {sw->offset, wbytes},
        {ss->offset, ss->bytes}, {sb->offset, sb->bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_cross_attention, binds, ((uint32_t)seq_q + 63u) / 64u,
                            (uint32_t)num_heads, (uint32_t)batch)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_dequantize_linear_f32(const float* in, const float* scale, const float* zero_point,
                                   float* out, long n, int has_zero_point) {
    if (!in || !scale || !out || n <= 0) return 0;
    static const float zero[1] = {0.0f};
    size_t bytes = (size_t)n * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    VkTensorSlot* ss = graph_ensure_device(scale, sizeof(float), 1);
    VkTensorSlot* sz = graph_ensure_device(has_zero_point && zero_point ? zero_point : zero, sizeof(float), 1);
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !ss || !sz || !dst) return 0;
    uint32_t params[4] = {(uint32_t)n, (uint32_t)(has_zero_point && zero_point), 0u, 0u};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[5] = {
        {src->offset, bytes}, {ss->offset, sizeof(float)}, {sz->offset, sizeof(float)}, {dst->offset, bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_dequantize, binds, ((uint32_t)n + 63u) / 64u, 1, 1)) return 0;
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
} VkQLinearParams;

_Static_assert(sizeof(VkQLinearParams) == 64, "qLinearInt8 uniform ABI");

static void vk_disable_packed_dot_after_failure(const char* kernel_name) {
    VulkanContextState* context = vk_context_current();
    if (context) context->packed_dot_disabled = 1;
    pthread_mutex_lock(&g_vulkan_device.mutex);
    if (!vulkan_packed_dot_warned) {
        fprintf(stderr,
                "[Vulkan] packed INT8 dot kernel '%s' unavailable; using portable shader path\n",
                kernel_name ? kernel_name : "unknown");
        vulkan_packed_dot_warned = 1;
    }
    pthread_mutex_unlock(&g_vulkan_device.mutex);
}

static int vk_packed_dot_enabled(void) {
    VulkanContextState* context = vk_context_current();
    return vulkan_packed_dot && context && !context->packed_dot_disabled;
}

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

int vk_graph_qlinear_i8u8(const void* input, const void* weight,
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
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_packed_bytes(input, input_bytes, 0);
    VkTensorSlot* wt = graph_ensure_packed_bytes(weight, weight_bytes, 1);
    VkTensorSlot* zero_points = graph_ensure_device(weight_zero_points,
                                                     (size_t)d_out * sizeof(*weight_zero_points), 1);
    VkTensorSlot* biases = graph_ensure_device(bias, (size_t)d_out * sizeof(*bias), 1);
    VkTensorSlot* dst = graph_output_packed_bytes(output, output_bytes);
    if (!src || !wt || !zero_points || !biases || !dst) {
        free(multipliers);
        return 0;
    }
    size_t multiplier_offset =
        graph_scratch_upload(multipliers, multiplier_bytes);
    free(multipliers);
    if (multiplier_offset == SIZE_MAX) return 0;
    VkQLinearParams params = {
        rows, d_in, d_out,
        input_dtype,
        weight_dtype,
        output_dtype, 0u, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, 0.0f, 0.0f
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[7] = {
        {src->offset, src->bytes}, {wt->offset, wt->bytes},
        {multiplier_offset, multiplier_bytes},
        {zero_points->offset, zero_points->bytes},
        {biases->offset, biases->bytes}, {dst->offset, packed_output_bytes},
        {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    int tiled = rows > 1u && d_in >= 16u && d_out >= 32u && (d_out & 3u) == 0u;
    uint32_t groups_x = tiled ? ((d_out / 4u + 7u) / 8u) : ((packed_words + 63u) / 64u);
    uint32_t groups_y = tiled ? ((rows + 7u) / 8u) : 1u;
    int dispatched = 0;
    if (vk_packed_dot_enabled()) {
        VkKernel* dot_kernel = tiled
            ? &k_qlinear_int8_dot_tiled : &k_qlinear_int8_dot;
        dispatched = vk_dispatch_kernel(dot_kernel, binds,
                                        groups_x, groups_y, 1u);
        if (!dispatched) vk_disable_packed_dot_after_failure(dot_kernel->name);
        else vk_context_current()->qlinear_dot_dispatches++;
    }
    if (!dispatched) {
        VkKernel* kernel = tiled ? &k_qlinear_int8_tiled : &k_qlinear_int8;
        if (!vk_dispatch_kernel(kernel, binds, groups_x, groups_y, 1u)) return 0;
        if (tiled) vk_context_current()->qlinear_tiled_dispatches++;
        else vk_context_current()->qlinear_scalar_dispatches++;
    }
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
} VkQEmbeddingParams;

_Static_assert(sizeof(VkQEmbeddingParams) == 32, "qEmbeddingInt8 uniform ABI");

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

int vk_graph_qembedding_i8u8(const int32_t* tokens, const void* weight,
                             const float* weight_scales,
                             const int32_t* weight_zero_points, void* output,
                             uint32_t token_count, uint32_t vocab, uint32_t hidden,
                             float output_scale, int32_t output_zero_point,
                             uint32_t weight_dtype, uint32_t output_dtype) {
    size_t token_bytes;
    size_t weight_bytes;
    size_t output_bytes;
    size_t packed_output_bytes;
    if (!qembedding_gpu_args_valid(tokens, weight, weight_scales, weight_zero_points,
                                   output, token_count, vocab, hidden, output_scale,
                                   output_zero_point, weight_dtype, output_dtype,
                                   &token_bytes, &weight_bytes, &output_bytes) ||
        !graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* ids = graph_ensure_device(tokens, token_bytes, 0);
    VkTensorSlot* table = graph_ensure_packed_bytes(weight, weight_bytes, 1);
    VkTensorSlot* scales = graph_ensure_device(weight_scales,
                                                (size_t)vocab * sizeof(*weight_scales), 1);
    VkTensorSlot* zero_points = graph_ensure_device(weight_zero_points,
                                                     (size_t)vocab * sizeof(*weight_zero_points), 1);
    VkTensorSlot* dst = graph_output_packed_bytes(output, output_bytes);
    if (!ids || !table || !scales || !zero_points || !dst) return 0;
    VkQEmbeddingParams params = {
        token_count, vocab, hidden,
        weight_dtype,
        output_dtype, output_zero_point,
        output_scale, 0u
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[6] = {
        {ids->offset, ids->bytes}, {table->offset, table->bytes},
        {scales->offset, scales->bytes}, {zero_points->offset, zero_points->bytes},
        {dst->offset, packed_output_bytes}, {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    if (!vk_dispatch_kernel(&k_qembedding_int8, binds,
                            (packed_words + 63u) / 64u, 1u, 1u)) return 0;
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
} VkQAddParams;

_Static_assert(sizeof(VkQAddParams) == 48, "qAdd uniform ABI");

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
} VkQBatchMatMulParams;

_Static_assert(sizeof(VkQBatchMatMulParams) == 64,
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
} VkQByteUnaryParams;

_Static_assert(sizeof(VkQByteUnaryParams) == 48,
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
} VkQGroupNormParams;

_Static_assert(sizeof(VkQGroupNormParams) == 64,
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
} VkQLayerNormParams;

_Static_assert(sizeof(VkQLayerNormParams) == 48,
               "qLayerNormStats/qLayerNormApply uniform ABI");

/* Five tightly packed 16-byte blocks shared verbatim with qSDPAInt8.wgsl.
 * The four canonical protobuf dtype values occupy byte lanes in one u32 so
 * every backend uses the same 80-byte uniform layout. */
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
} VkQSDPAParams;

_Static_assert(sizeof(VkQSDPAParams) == 80, "qSDPAInt8 uniform ABI");

typedef struct {
    uint32_t outer;
    uint32_t axis_size;
    uint32_t inner;
    uint32_t input_dtype;
} VkQArgMaxParams;

_Static_assert(sizeof(VkQArgMaxParams) == 16, "qArgMaxInt8 uniform ABI");

/* Eight u32 fields followed by two I32 zero points and two F32 scales.  This
 * 48-byte block is shared verbatim with qMaskedMeanInt8.wgsl and the browser
 * WebGPU dispatcher. */
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
} VkQMaskedMeanParams;

_Static_assert(sizeof(VkQMaskedMeanParams) == 48,
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
} VkRequantizeLinearParams;

_Static_assert(sizeof(VkRequantizeLinearParams) == 48,
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

/* Validate all descriptor arithmetic before residency/upload.  The finite
 * worst-case score test guarantees qSDPAInt8's online recurrence never sees
 * +/-infinity and therefore cannot evaluate an invalid infinity subtraction. */
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

/* Validate the exact scalar byte-reduction envelope before allocating device
 * residency.  qMaskedMeanInt8 accumulates centered values in I32, so sequence
 * length is bounded by the declared signed/unsigned zero point rather than
 * relying on an implementation-defined overflow. */
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

int vk_graph_qbatch_matmul_i8u8(
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
    if (!vx_qbatch_matmul_device_plan(
            a, a_shape, a_rank, a_scale, a_zero_point, a_dtype,
            b, b_shape, b_rank, b_scale, b_zero_point, b_dtype,
            output, output_shape, output_rank, output_scale,
            output_zero_point, output_dtype, &plan) ||
        !graph_packed_bytes(plan.a_bytes, &a_packed_bytes) ||
        !graph_packed_bytes(plan.b_bytes, &b_packed_bytes) ||
        !graph_packed_bytes(plan.output_bytes, &output_packed_bytes))
        return 0;
    graph_scratch_begin();
    VkTensorSlot* a_slot =
        graph_ensure_packed_bytes(a, plan.a_bytes, 0);
    VkTensorSlot* b_slot =
        graph_ensure_packed_bytes(b, plan.b_bytes, 0);
    VkTensorSlot* output_slot =
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
    VkQBatchMatMulParams params = {
        plan.batch_rank, plan.m, plan.k, plan.n,
        plan.output_elements, a_dtype, b_dtype, output_dtype,
        a_zero_point, b_zero_point, output_zero_point, 0,
        a_scale, b_scale, output_scale, 0.0f,
    };
    size_t metadata_offset =
        graph_scratch_upload(metadata, metadata_bytes);
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (metadata_offset == SIZE_MAX || params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[5] = {
        {a_slot->offset, a_packed_bytes},
        {b_slot->offset, b_packed_bytes},
        {output_slot->offset, output_packed_bytes},
        {metadata_offset, metadata_bytes},
        {params_offset, sizeof(params)},
    };
    uint32_t packed_words =
        (uint32_t)(output_packed_bytes / sizeof(uint32_t));
    int dispatched = 0;
    uint32_t groups = (packed_words + 63u) / 64u;
    if (vk_packed_dot_enabled()) {
        dispatched = vk_dispatch_kernel(
            &k_qbatch_matmul_i8u8_dot, binds, groups, 1u, 1u);
        if (!dispatched) {
            vk_disable_packed_dot_after_failure(
                k_qbatch_matmul_i8u8_dot.name);
        }
        else vk_context_current()->qbatch_dot_dispatches++;
    }
    if (!dispatched && !vk_dispatch_kernel(
            &k_qbatch_matmul_i8u8, binds, groups, 1u, 1u))
        return 0;
    if (!dispatched) vk_context_current()->qbatch_scalar_dispatches++;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_qadd_i8u8(const void* a, uint32_t a_elements,
                       const void* b, uint32_t b_elements,
                       void* output, uint32_t output_elements,
                       float a_scale, int32_t a_zero_point,
                       float b_scale, int32_t b_zero_point,
                       float output_scale, int32_t output_zero_point,
                       uint32_t a_dtype, uint32_t b_dtype,
                       uint32_t output_dtype, uint32_t relu) {
    size_t logical_bytes;
    if (!qadd_gpu_args_valid(a, a_elements, b, b_elements, output, output_elements,
                             a_scale, a_zero_point, b_scale, b_zero_point,
                             output_scale, output_zero_point, a_dtype, b_dtype,
                             output_dtype, relu, &logical_bytes)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* a_slot = graph_ensure_packed_bytes(a, logical_bytes, 0);
    VkTensorSlot* b_slot = graph_ensure_packed_bytes(b, logical_bytes, 0);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, logical_bytes);
    if (!a_slot || !b_slot || !output_slot) return 0;
    VkQAddParams params = {
        a_elements, a_dtype,
        b_dtype,
        output_dtype,
        a_zero_point, b_zero_point, output_zero_point, 0,
        a_scale, b_scale, output_scale, relu
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {a_slot->offset, a_slot->bytes}, {b_slot->offset, b_slot->bytes},
        {output_slot->offset, packed_bytes}, {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    if (!vk_dispatch_kernel(&k_qadd_i8u8, binds, (packed_words + 63u) / 64u,
                            1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_qsilu_i8u8(const void* input, void* output, uint32_t elements,
                        float input_scale, int32_t input_zero_point,
                        float output_scale, int32_t output_zero_point,
                        uint32_t input_dtype, uint32_t output_dtype) {
    size_t logical_bytes;
    if (!qbyte_unary_gpu_args_valid(input, output, elements, input_scale,
                                    input_zero_point, output_scale, output_zero_point,
                                    input_dtype, output_dtype, &logical_bytes)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, logical_bytes, 0);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, logical_bytes);
    if (!input_slot || !output_slot) return 0;
    VkQByteUnaryParams params = {
        elements, input_dtype,
        output_dtype, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, 0.0f, 0.0f
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {input_slot->offset, input_slot->bytes},
        {output_slot->offset, packed_bytes}, {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    if (!vk_dispatch_kernel(&k_qsilu_i8u8, binds, (packed_words + 63u) / 64u,
                            1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_qgelu_i8u8(const void* input, void* output, uint32_t elements,
                        float input_scale, int32_t input_zero_point,
                        float output_scale, int32_t output_zero_point,
                        uint32_t input_dtype, uint32_t output_dtype) {
    size_t logical_bytes;
    if (!qbyte_unary_gpu_args_valid(input, output, elements, input_scale,
                                    input_zero_point, output_scale, output_zero_point,
                                    input_dtype, output_dtype, &logical_bytes)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, logical_bytes, 0);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, logical_bytes);
    if (!input_slot || !output_slot) return 0;
    VkQByteUnaryParams params = {
        elements, input_dtype,
        output_dtype, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, 0.0f, 0.0f
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {input_slot->offset, input_slot->bytes},
        {output_slot->offset, packed_bytes}, {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    if (!vk_dispatch_kernel(&k_qgelu_i8u8, binds, (packed_words + 63u) / 64u,
                            1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

/* The statistics pass owns one [batch, group] workgroup; the apply pass owns
 * complete packed output words.  Keeping them separate prevents byte-lane
 * races whenever a channel-group boundary crosses a u32 word. */
int vk_graph_qgroupnorm_i8u8(const void* input, const float* weight,
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
    if (!qgroupnorm_gpu_args_valid(input, weight, bias, output, batch, height,
                                   width, channels, groups, input_scale,
                                   input_zero_point, output_scale,
                                   output_zero_point, epsilon, input_dtype,
                                   output_dtype, &logical_bytes, &affine_bytes,
                                   &stats_bytes) ||
        !graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, logical_bytes, 0);
    VkTensorSlot* weight_slot = graph_ensure_device(weight, affine_bytes, 1);
    VkTensorSlot* bias_slot = graph_ensure_device(bias, affine_bytes, 1);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, logical_bytes);
    if (!input_slot || !weight_slot || !bias_slot || !output_slot) return 0;
    VulkanContextState* context = vk_context_current();
    size_t stats_offset;
    if (context && context->domain_enforced) {
        if (!context->domain_qgroupnorm_stats_bytes ||
            stats_bytes > context->domain_qgroupnorm_stats_bytes)
            return 0;
        stats_offset = context->domain_qgroupnorm_stats_offset;
    } else {
        stats_offset = graph_scratch_alloc(stats_bytes);
        if (stats_offset == SIZE_MAX) return 0;
    }
    VkQGroupNormParams params = {
        batch, height, width, channels, groups,
        input_dtype,
        output_dtype, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, epsilon, 0.0f
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding stats_binds[3] = {
        {input_slot->offset, input_slot->bytes}, {stats_offset, stats_bytes},
        {params_offset, sizeof(params)}
    };
    VkGraphBinding apply_binds[6] = {
        {input_slot->offset, input_slot->bytes},
        {weight_slot->offset, weight_slot->bytes}, {bias_slot->offset, bias_slot->bytes},
        {stats_offset, stats_bytes}, {output_slot->offset, packed_bytes},
        {params_offset, sizeof(params)}
    };
    uint32_t total_groups = batch * groups;
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    uint32_t apply_groups = packed_words / 64u + (packed_words % 64u != 0u);
    if (!vk_dispatch_kernel(&k_qgroupnorm_stats, stats_binds, total_groups, 1u, 1u) ||
        !vk_dispatch_kernel(&k_qgroupnorm_apply, apply_binds, apply_groups, 1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

/* One 64-lane workgroup produces a [mean_raw, inverse_stddev] pair for each
 * final-axis row; the apply pass then owns whole packed destination words. */
int vk_graph_qlayernorm_i8u8(const void* input, const float* weight,
                             const float* bias, void* output, uint32_t rows,
                             uint32_t d_model, float input_scale,
                             int32_t input_zero_point, float output_scale,
                             int32_t output_zero_point, float epsilon,
                             uint32_t input_dtype, uint32_t output_dtype) {
    size_t logical_bytes;
    size_t affine_bytes;
    size_t stats_bytes;
    size_t packed_bytes;
    if (!qlayernorm_gpu_args_valid(input, weight, bias, output, rows, d_model,
                                   input_scale, input_zero_point, output_scale,
                                   output_zero_point, epsilon, input_dtype,
                                   output_dtype, &logical_bytes, &affine_bytes,
                                   &stats_bytes) ||
        !graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, logical_bytes, 0);
    VkTensorSlot* weight_slot = graph_ensure_device(weight, affine_bytes, 1);
    VkTensorSlot* bias_slot = graph_ensure_device(bias, affine_bytes, 1);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, logical_bytes);
    if (!input_slot || !weight_slot || !bias_slot || !output_slot) return 0;
    VulkanContextState* context = vk_context_current();
    size_t stats_offset;
    if (context && context->domain_enforced) {
        if (!context->domain_qlayernorm_stats_bytes ||
            stats_bytes > context->domain_qlayernorm_stats_bytes)
            return 0;
        stats_offset = context->domain_qlayernorm_stats_offset;
    } else {
        stats_offset = graph_scratch_alloc(stats_bytes);
        if (stats_offset == SIZE_MAX) return 0;
    }
    VkQLayerNormParams params = {
        rows, d_model, input_dtype,
        output_dtype,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, epsilon, 0.0f
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding stats_binds[3] = {
        {input_slot->offset, input_slot->bytes}, {stats_offset, stats_bytes},
        {params_offset, sizeof(params)}
    };
    VkGraphBinding apply_binds[6] = {
        {input_slot->offset, input_slot->bytes},
        {weight_slot->offset, weight_slot->bytes}, {bias_slot->offset, bias_slot->bytes},
        {stats_offset, stats_bytes}, {output_slot->offset, packed_bytes},
        {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    uint32_t apply_groups = packed_words / 64u + (packed_words % 64u != 0u);
    if (!vk_dispatch_kernel(&k_qlayernorm_stats, stats_binds, rows, 1u, 1u) ||
        !vk_dispatch_kernel(&k_qlayernorm_apply, apply_binds, apply_groups, 1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

/* One [query, head, batch] workgroup performs its own online softmax and
 * writes only its head's bytes.  No graph-visible F32 score/value storage is
 * allocated; absent masks bind the dedicated static I32 dummy buffer. */
int vk_graph_qsdpa_i8u8(const void* q, const void* k, const void* v,
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
    if (!qsdpa_gpu_args_valid(q, k, v, mask, output, batch, seq_q, seq_kv,
                               d_model, heads, q_scale, q_zero_point, k_scale,
                               k_zero_point, v_scale, v_zero_point, output_scale,
                               output_zero_point, attention_scale, q_dtype, k_dtype,
                               v_dtype, output_dtype, causal, mask_mode, &q_bytes,
                               &kv_bytes, &mask_bytes) ||
        !graph_packed_bytes(q_bytes, &packed_output_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* q_slot = graph_ensure_packed_bytes(q, q_bytes, 0);
    VkTensorSlot* k_slot = graph_ensure_packed_bytes(k, kv_bytes, 0);
    VkTensorSlot* v_slot = graph_ensure_packed_bytes(v, kv_bytes, 0);
    VkTensorSlot* mask_slot = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) :
        graph_ensure_device(qsdpa_dummy_mask, sizeof(qsdpa_dummy_mask), 1);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, q_bytes);
    if (!q_slot || !k_slot || !v_slot || !mask_slot || !output_slot) return 0;
    VkQSDPAParams params = {
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
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[6] = {
        {q_slot->offset, q_slot->bytes}, {k_slot->offset, k_slot->bytes},
        {v_slot->offset, v_slot->bytes}, {mask_slot->offset, mask_slot->bytes},
        {output_slot->offset, packed_output_bytes}, {params_offset, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_qsdpa_int8, binds, seq_q, heads, batch)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

/* qArgMaxInt8 owns one output index per [outer,inner] coordinate. Input stays
 * byte-packed, output is conventional I32 storage, and the 16-byte uniform
 * exactly mirrors the WebGPU ABI. */
int vk_graph_qargmax_i8u8(const void* input, int32_t* output,
                          uint32_t outer, uint32_t axis_size,
                          uint32_t inner, uint32_t input_dtype) {
    size_t input_bytes;
    size_t output_bytes;
    uint32_t output_elements;
    if (!qargmax_gpu_args_valid(input, output, outer, axis_size, inner,
                                input_dtype, &input_bytes, &output_bytes,
                                &output_elements)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_bytes, 0);
    VkTensorSlot* output_slot = graph_output_slot(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    VkQArgMaxParams params = {
        outer, axis_size, inner, input_dtype
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {input_slot->offset, input_slot->bytes},
        {output_slot->offset, output_slot->bytes},
        {params_offset, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_qargmax_int8, binds,
                            output_elements / 64u + (output_elements % 64u != 0u),
                            1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

/* qMaskedMeanInt8 owns complete packed output words, so byte lanes never
 * race while producing [B,D] output from the [B,S,D] activation and I32
 * [B,S] keep mask.  The reduction's F32 arithmetic is shader-private. */
int vk_graph_qmaskedmean_i8u8(const void* input, const int32_t* mask,
                              void* output, uint32_t batch,
                              uint32_t sequence, uint32_t width,
                              float input_scale, int32_t input_zero_point,
                              float output_scale, int32_t output_zero_point,
                              uint32_t input_dtype, uint32_t output_dtype) {
    size_t input_bytes;
    size_t mask_bytes;
    size_t output_bytes;
    size_t packed_output_bytes;
    if (!qmaskedmean_gpu_args_valid(input, mask, output, batch, sequence, width,
                                    input_scale, input_zero_point, output_scale,
                                    output_zero_point, input_dtype, output_dtype,
                                    &input_bytes, &mask_bytes, &output_bytes) ||
        !graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_bytes, 0);
    VkTensorSlot* mask_slot = graph_ensure_device(mask, mask_bytes, 0);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, output_bytes);
    if (!input_slot || !mask_slot || !output_slot) return 0;
    VkQMaskedMeanParams params = {
        batch, sequence, width, input_dtype,
        output_dtype, 0u, 0u, 0u,
        input_zero_point, output_zero_point, input_scale, output_scale
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {input_slot->offset, input_slot->bytes},
        {mask_slot->offset, mask_slot->bytes},
        {output_slot->offset, packed_output_bytes},
        {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    if (!vk_dispatch_kernel(&k_qmaskedmean_int8, binds,
                            (packed_words + 63u) / 64u, 1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_requantize_linear_i8u8(const void* input, uint32_t input_elements,
                                    void* output, uint32_t output_elements,
                                    float input_scale, int32_t input_zero_point,
                                    float output_scale, int32_t output_zero_point,
                                    uint32_t input_dtype, uint32_t output_dtype) {
    float multiplier;
    size_t logical_bytes;
    if (!requantize_gpu_args_valid(input, input_elements, output, output_elements,
                                   input_scale, input_zero_point,
                                   output_scale, output_zero_point,
                                   input_dtype, output_dtype,
                                   &multiplier, &logical_bytes)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, logical_bytes, 0);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, logical_bytes);
    if (!input_slot || !output_slot) return 0;
    VkRequantizeLinearParams params = {
        input_elements, input_dtype,
        output_dtype, 0u,
        input_zero_point, output_zero_point, 0, 0,
        multiplier, 0.0f, 0.0f, 0.0f
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {input_slot->offset, input_slot->bytes},
        {output_slot->offset, packed_bytes}, {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    if (!vk_dispatch_kernel(&k_requantize_linear_i8u8, binds,
                            (packed_words + 63u) / 64u, 1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
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
} VkQConv2DParams;

_Static_assert(sizeof(VkQConv2DParams) == 112, "qConv2DInt8 uniform ABI");

static const int32_t* qconv_zero_bias_get(uint32_t output_channels) {
    if (output_channels == 0 ||
        (size_t)output_channels > SIZE_MAX / sizeof(int32_t)) return NULL;
    for (QConvZeroBiasBacking* block = qconv_zero_bias_backings; block;
         block = block->next) {
        /* A graph weight slot records the exact readable byte range retained
         * at domain bind. Distinct channel widths therefore need distinct
         * durable keys; reusing a larger block for a later smaller node would
         * shrink that slot and make the earlier node invalid after bind. */
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

int vk_graph_qconv2d_i8u8(const void* input, const void* weight,
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
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_bytes, 0);
    VkTensorSlot* weight_slot = graph_ensure_packed_bytes(weight, weight_bytes, 1);
    VkTensorSlot* scales_slot = graph_ensure_device(weight_scales,
                                                     (size_t)output_channels * sizeof(float), 1);
    VkTensorSlot* zero_points_slot = graph_ensure_device(weight_zero_points, metadata_bytes, 1);
    VkTensorSlot* bias_slot = graph_ensure_device(bound_bias, metadata_bytes, 1);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, output_bytes);
    if (!input_slot || !weight_slot || !scales_slot || !zero_points_slot ||
        !bias_slot || !output_slot) return 0;
    VkQConv2DParams params = {
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
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[7] = {
        {input_slot->offset, input_slot->bytes}, {weight_slot->offset, weight_slot->bytes},
        {scales_slot->offset, scales_slot->bytes}, {zero_points_slot->offset, zero_points_slot->bytes},
        {bias_slot->offset, bias_slot->bytes}, {output_slot->offset, packed_output_bytes},
        {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    const uint64_t reduction_size =
        (uint64_t)kernel_height * kernel_width * input_per_group;
    const int dot_tiled = groups == 1u &&
        (output_channels & 3u) == 0u &&
        vk_workgroup_shape_supported(8u, 4u, 1u);
    const int tiled = groups == 1u && output_channels >= 32u &&
        reduction_size >= 16u && (output_channels & 3u) == 0u &&
        vk_workgroup_shape_supported(8u, 4u, 1u);
    int dispatched = 0;
    if (vk_packed_dot_enabled() && dot_tiled) {
        const uint32_t spatial = batch * output_height * output_width;
        const uint32_t output_words_per_spatial = output_channels / 4u;
        const uint32_t groups_x = (output_words_per_spatial + 7u) / 8u;
        const uint32_t groups_y = (spatial + 3u) / 4u;
        dispatched = vk_dispatch_kernel(&k_qconv2d_int8_dot_tiled, binds,
                                        groups_x, groups_y, 1u);
        if (!dispatched) {
            vk_disable_packed_dot_after_failure(k_qconv2d_int8_dot_tiled.name);
        }
        else {
            vk_context_current()->qconv_dot_tiled_dispatches++;
        }
    }
    if (!dispatched && tiled) {
        const uint32_t spatial = batch * output_height * output_width;
        const uint32_t output_words_per_spatial = output_channels / 4u;
        dispatched = vk_dispatch_kernel(
            &k_qconv2d_int8_tiled, binds,
            (output_words_per_spatial + 7u) / 8u,
            (spatial + 3u) / 4u, 1u);
        if (dispatched)
            vk_context_current()->qconv_tiled_dispatches++;
    }
    if (!dispatched && !vk_dispatch_kernel(&k_qconv2d_int8, binds,
                                            (packed_words + 63u) / 64u,
                                            1u, 1u)) return 0;
    if (!dispatched)
        vk_context_current()->qconv_scalar_dispatches++;
    graph_mark_device(output_slot);
    return 1;
}

typedef struct { uint32_t elements, pad0, pad1, pad2; } VkTypedCopyParams;
typedef struct {
    uint32_t size, axis_offset, input_axis, output_axis;
    uint32_t inner, pad0, pad1, pad2;
} VkTypedConcatParams;
typedef struct {
    uint32_t n, h, w, c, out_h, out_w, ky, kx;
    uint32_t sy, sx, py, px, dtype, pad0, pad1, pad2;
} VkTypedMaxPoolParams;
typedef struct { uint32_t n, h, w, c, out_h, out_w, pad0, pad1; } VkTypedResizeParams;
typedef struct { uint32_t elements, output_type, zero_point_type, has_zero_point; } VkTypedQuantizeParams;
typedef struct {
    uint32_t size, input_type, scale_type, zero_point_type;
    uint32_t output_type, has_zero_point, pad0, pad1;
} VkTypedDequantizeParams;

_Static_assert(sizeof(VkTypedCopyParams) == 16, "copyTyped uniform ABI");
_Static_assert(sizeof(VkTypedConcatParams) == 32, "concatCopyTyped uniform ABI");
_Static_assert(sizeof(VkTypedMaxPoolParams) == 64, "maxPool2DTyped uniform ABI");
_Static_assert(sizeof(VkTypedResizeParams) == 32, "resizeNearestTyped uniform ABI");
_Static_assert(sizeof(VkTypedQuantizeParams) == 16, "quantizeLinearTyped uniform ABI");
_Static_assert(sizeof(VkTypedDequantizeParams) == 32, "dequantizeLinearTyped uniform ABI");

static int typed_shape_qdesc_valid(float scale, int32_t zero_point, uint32_t dtype) {
    return isfinite(scale) && scale > 0.0f &&
        qbyte_dtype_zero_point_valid(dtype, zero_point);
}

static int typed_shape_qdesc_same(float input_scale, int32_t input_zero_point,
                                  uint32_t input_dtype, float output_scale,
                                  int32_t output_zero_point, uint32_t output_dtype) {
    return typed_shape_qdesc_valid(input_scale, input_zero_point, input_dtype) &&
        typed_shape_qdesc_valid(output_scale, output_zero_point, output_dtype) &&
        input_dtype == output_dtype && input_zero_point == output_zero_point &&
        input_scale == output_scale;
}

static int typed_shape_nhwc_elements(uint32_t n, uint32_t h, uint32_t w,
                                     uint32_t c, uint32_t* elements) {
    uint64_t value = n;
    if (!n || !h || !w || !c || !elements ||
        !qconv_mul_u64(value, h, &value) || !qconv_mul_u64(value, w, &value) ||
        !qconv_mul_u64(value, c, &value) || value > UINT32_MAX) return 0;
    *elements = (uint32_t)value;
    return 1;
}

static uint32_t typed_shape_groups(uint32_t elements) {
    return elements / 64u + (elements % 64u != 0u);
}

static uint32_t typed_shape_packed_groups(size_t packed_bytes) {
    uint32_t words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    return typed_shape_groups(words);
}

static uint32_t typed_shape_zero_word(int32_t zero_point, uint32_t dtype) {
    return dtype == VX_DTYPE_I8 ? (uint32_t)(uint8_t)(int8_t)zero_point :
        (uint32_t)(uint8_t)zero_point;
}

int vk_graph_quantize_typed_f32_i8u8(const float* input, uint32_t elements,
                                     void* output, float output_scale,
                                     int32_t output_zero_point, uint32_t output_dtype) {
    if (!input || !output || input == output || elements == 0 ||
        !typed_shape_qdesc_valid(output_scale, output_zero_point, output_dtype) ||
        (uint64_t)elements > SIZE_MAX / sizeof(float)) return 0;
    size_t output_bytes = (size_t)elements;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_device(input, (size_t)elements * sizeof(float), 0);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    uint32_t zero_word = typed_shape_zero_word(output_zero_point, output_dtype);
    VkTypedQuantizeParams params = {
        elements, output_dtype, output_dtype, 1u
    };
    size_t scale_offset = graph_scratch_upload(&output_scale, sizeof(output_scale));
    size_t zero_offset = graph_scratch_upload(&zero_word, sizeof(zero_word));
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (scale_offset == SIZE_MAX || zero_offset == SIZE_MAX || params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[5] = {
        {input_slot->offset, input_slot->bytes}, {scale_offset, sizeof(output_scale)},
        {zero_offset, sizeof(zero_word)}, {output_slot->offset, packed_output_bytes},
        {params_offset, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_quantize_typed_i8u8, binds,
                            typed_shape_packed_groups(packed_output_bytes), 1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_dequantize_typed_i8u8_f32(const void* input, uint32_t elements,
                                       float input_scale, int32_t input_zero_point,
                                       uint32_t input_dtype, float* output) {
    if (!input || !output || input == output || elements == 0 ||
        !typed_shape_qdesc_valid(input_scale, input_zero_point, input_dtype) ||
        (uint64_t)elements > SIZE_MAX / sizeof(float)) return 0;
    size_t input_bytes = (size_t)elements;
    size_t packed_input_bytes;
    if (!graph_packed_bytes(input_bytes, &packed_input_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_bytes, 0);
    VkTensorSlot* output_slot = graph_output_slot(output, (size_t)elements * sizeof(float));
    if (!input_slot || !output_slot) return 0;
    uint32_t zero_word = typed_shape_zero_word(input_zero_point, input_dtype);
    VkTypedDequantizeParams params = {
        elements, input_dtype, VX_DTYPE_F32, input_dtype,
        VX_DTYPE_F32, 1u, 0u, 0u
    };
    size_t scale_offset = graph_scratch_upload(&input_scale, sizeof(input_scale));
    size_t zero_offset = graph_scratch_upload(&zero_word, sizeof(zero_word));
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (scale_offset == SIZE_MAX || zero_offset == SIZE_MAX || params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[5] = {
        {input_slot->offset, packed_input_bytes}, {scale_offset, sizeof(input_scale)},
        {zero_offset, sizeof(zero_word)}, {output_slot->offset, output_slot->bytes},
        {params_offset, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_dequantize_typed_i8u8, binds,
                            typed_shape_groups(elements), 1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_copy_i8u8(const void* input, uint32_t input_elements,
                       void* output, uint32_t output_elements,
                       float input_scale, int32_t input_zero_point,
                       float output_scale, int32_t output_zero_point,
                       uint32_t input_dtype, uint32_t output_dtype) {
    if (!input || !output || input == output || input_elements == 0 ||
        input_elements != output_elements || !typed_shape_qdesc_same(input_scale,
        input_zero_point, input_dtype, output_scale, output_zero_point, output_dtype)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes((size_t)input_elements, &packed_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_elements, 0);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, output_elements);
    if (!input_slot || !output_slot) return 0;
    VkTypedCopyParams params = {input_elements, 0u, 0u, 0u};
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {input_slot->offset, input_slot->bytes}, {output_slot->offset, packed_bytes},
        {params_offset, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_copy_typed_i8u8, binds,
                            typed_shape_packed_groups(packed_bytes), 1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_transpose_i8u8(const void* input, void* output,
                            const uint32_t* input_shape,
                            const uint32_t* permutation, uint32_t rank,
                            uint32_t elements, float input_scale,
                            int32_t input_zero_point, float output_scale,
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
    if (!input || !output || !input_shape || !permutation ||
        rank == 0 || rank > 8 || elements == 0 ||
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
    graph_scratch_begin();
    VkTensorSlot* input_slot =
        graph_ensure_packed_bytes(input, elements, 0);
    VkTensorSlot* output_slot =
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
    size_t metadata_offset =
        graph_scratch_upload(metadata, metadata_bytes);
    if (metadata_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {input_slot->offset, packed_bytes},
        {output_slot->offset, packed_bytes},
        {metadata_offset, metadata_bytes},
    };
    if (!vk_dispatch_kernel(&k_transpose_typed_i8u8, binds,
                            typed_shape_packed_groups(packed_bytes),
                            1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_concat_i8u8(const void* const* inputs, const uint32_t* input_elements,
                         const uint32_t* input_axes, const float* input_scales,
                         const int32_t* input_zero_points, const uint32_t* input_dtypes,
                         uint32_t input_count, void* output, uint32_t output_elements,
                         uint32_t output_axis, uint32_t inner,
                         float output_scale, int32_t output_zero_point,
                         uint32_t output_dtype) {
    if (!inputs || !input_elements || !input_axes || !input_scales ||
        !input_zero_points || !input_dtypes || !output || !input_count ||
        !output_elements || !output_axis || !inner ||
        !typed_shape_qdesc_valid(output_scale, output_zero_point, output_dtype)) return 0;
    uint64_t output_width = (uint64_t)output_axis * inner;
    if (!output_width || output_width > output_elements || output_elements % output_width) return 0;
    uint64_t outer = output_elements / output_width;
    uint64_t axis_sum = 0;
    for (uint32_t index = 0; index < input_count; index++) {
        uint64_t expected = 0;
        if (!qconv_mul_u64(outer, input_axes[index], &expected) ||
            !qconv_mul_u64(expected, inner, &expected)) {
            return 0;
        }
        if (!inputs[index] || inputs[index] == output || !input_axes[index] ||
            expected != input_elements[index] ||
            !typed_shape_qdesc_same(input_scales[index], input_zero_points[index],
                                    input_dtypes[index], output_scale,
                                    output_zero_point, output_dtype) ||
            axis_sum > UINT32_MAX - input_axes[index]) return 0;
        axis_sum += input_axes[index];
    }
    if (axis_sum != output_axis) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_elements, &packed_output_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, output_elements);
    if (!output_slot) return 0;
    memset((char*)io_mapped + output_slot->offset, 0, packed_output_bytes);
    output_slot->host_dirty = 0;
    output_slot->device_dirty = 0;
    uint32_t axis_offset = 0;
    for (uint32_t index = 0; index < input_count; index++) {
        size_t packed_input_bytes;
        if (!graph_packed_bytes(input_elements[index], &packed_input_bytes)) return 0;
        VkTensorSlot* input_slot = graph_ensure_packed_bytes(inputs[index], input_elements[index], 0);
        if (!input_slot) return 0;
        VkTypedConcatParams params = {
            input_elements[index], axis_offset, input_axes[index], output_axis,
            inner, 0u, 0u, 0u
        };
        size_t params_offset = graph_scratch_upload(&params, sizeof(params));
        if (params_offset == SIZE_MAX) return 0;
        VkGraphBinding binds[3] = {
            {input_slot->offset, packed_input_bytes}, {output_slot->offset, packed_output_bytes},
            {params_offset, sizeof(params)}
        };
        if (!vk_dispatch_kernel(&k_concat_typed_i8u8, binds,
                                typed_shape_groups(input_elements[index]), 1u, 1u)) return 0;
        axis_offset += input_axes[index];
    }
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_maxpool2d_i8u8(const void* input, void* output,
                            uint32_t batch, uint32_t input_height,
                            uint32_t input_width, uint32_t channels,
                            uint32_t output_height, uint32_t output_width,
                            uint32_t kernel_y, uint32_t kernel_x,
                            uint32_t stride_y, uint32_t stride_x,
                            uint32_t padding_top, uint32_t padding_left,
                            uint32_t padding_bottom, uint32_t padding_right,
                            float input_scale, int32_t input_zero_point,
                            float output_scale, int32_t output_zero_point,
                            uint32_t input_dtype, uint32_t output_dtype) {
    uint32_t input_elements, output_elements;
    uint64_t padded_height, padded_width, expected_height, expected_width;
    if (!input || !output || input == output || !kernel_y || !kernel_x ||
        !stride_y || !stride_x || !typed_shape_qdesc_same(input_scale,
        input_zero_point, input_dtype, output_scale, output_zero_point, output_dtype) ||
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
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_elements, 0);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, output_elements);
    if (!input_slot || !output_slot) return 0;
    VkTypedMaxPoolParams params = {
        batch, input_height, input_width, channels, output_height, output_width,
        kernel_y, kernel_x, stride_y, stride_x, padding_top, padding_left,
        input_dtype, 0u, 0u, 0u
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {input_slot->offset, input_slot->bytes}, {output_slot->offset, packed_output_bytes},
        {params_offset, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_maxpool_typed_i8u8, binds,
                            typed_shape_packed_groups(packed_output_bytes), 1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_resize_nearest_i8u8(const void* input, void* output,
                                 uint32_t batch, uint32_t input_height,
                                 uint32_t input_width, uint32_t channels,
                                 uint32_t output_height, uint32_t output_width,
                                 float input_scale, int32_t input_zero_point,
                                 float output_scale, int32_t output_zero_point,
                                 uint32_t input_dtype, uint32_t output_dtype) {
    uint32_t input_elements, output_elements;
    if (!input || !output || input == output ||
        !typed_shape_qdesc_same(input_scale, input_zero_point, input_dtype,
                                output_scale, output_zero_point, output_dtype) ||
        !typed_shape_nhwc_elements(batch, input_height, input_width, channels, &input_elements) ||
        !typed_shape_nhwc_elements(batch, output_height, output_width, channels, &output_elements)) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_elements, &packed_output_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_elements, 0);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, output_elements);
    if (!input_slot || !output_slot) return 0;
    VkTypedResizeParams params = {
        batch, input_height, input_width, channels, output_height, output_width, 0u, 0u
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {input_slot->offset, input_slot->bytes}, {output_slot->offset, packed_output_bytes},
        {params_offset, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_resize_nearest_typed_i8u8, binds,
                            typed_shape_packed_groups(packed_output_bytes), 1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_quantize_linear_i8(const float* in, signed char* out, long n,
                                float input_scale, int input_zp,
                                float output_scale, int output_zp) {
    if (!in || !out || n <= 0 || output_scale <= 0.0f) return 0;
    size_t in_bytes = (size_t)n * sizeof(float);
    size_t packed_words = ((size_t)n + 3u) / 4u;
    size_t out_bytes = packed_words * sizeof(uint32_t);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
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
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_quantize, binds, (uint32_t)((packed_words + 63u) / 64u), 1, 1)) return 0;
    graph_mark_device(dst);
    return vk_graph_sync_host(out, (size_t)n, 0);
}

static int vk_graph_profile_common(VkKernel* kernel, const float* in, float* out,
                                   int n, int h, int w, int c, long out_elems_per_batch,
                                   uint32_t gx, uint32_t gy) {
    if (!kernel || !in || !out || n <= 0 || h <= 0 || w <= 0 || c <= 0 || out_elems_per_batch <= 0) return 0;
    size_t in_bytes = (size_t)n * (size_t)h * (size_t)w * (size_t)c * sizeof(float);
    size_t out_bytes = (size_t)n * (size_t)out_elems_per_batch * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)h, (uint32_t)w, (uint32_t)c, (uint32_t)n};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {{src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}};
    if (!vk_dispatch_kernel(kernel, binds, gx, gy, (uint32_t)n)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_spatial_softargmax_y_f32(const float* in, float* out, int n, int h, int w, int c) {
    return vk_graph_profile_common(&k_spatial_softargmax_y, in, out, n, h, w, c,
                                   (long)c * w, ((uint32_t)w + 63u) / 64u, (uint32_t)c);
}

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
        memset((char*)io_mapped + bias_off, 0, bias_bytes);
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
    uint32_t gz = (uint32_t)(n * out_c);
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
#if defined(VOLVOXAI_VULKAN_TESTING)
    if (kernel == &k_conv2d_pw16tile || kernel == &k_conv2d_pw16 ||
        kernel == &k_conv2d_pw8v4 || kernel == &k_conv2d_pw8v2 ||
        kernel == &k_conv2d_pw8) {
        vk_context_current()->test_conv_pointwise_selections++;
    }
#endif
    uint32_t gx = ((uint32_t)out_w + 7u) / 8u;
    uint32_t gy = ((uint32_t)out_h + 7u) / 8u;
    int ok = vk_dispatch_kernel(kernel, binds, gx, gy, gz);
    if (ok && (kernel == &k_conv2d_out16 || kernel == &k_conv2d_c3out16))
        vk_context_current()->conv_out16_dispatches++;
    else if (ok && kernel == &k_conv2d)
        vk_context_current()->conv_scalar_dispatches++;
    if (!ok && kernel != &k_conv2d) {
        if (kernel == &k_conv2d_out16 || kernel == &k_conv2d_c3out16)
            ok = vk_dispatch_kernel(&k_conv2d, binds, gx, gy,
                                    (uint32_t)(n * out_c));
        if (kernel == &k_conv2d_pw16tile) ok = vk_dispatch_kernel(&k_conv2d_pw16, binds, gx, gy, (uint32_t)(n * (out_c / 16)));
        if (!ok && kernel == &k_conv2d_dw8 && (out_c & 3) == 0) ok = vk_dispatch_kernel(&k_conv2d_dw4, binds, gx, gy, (uint32_t)(n * ((out_c + 3) / 4)));
        if (!ok && (kernel == &k_conv2d_pw8v4 || kernel == &k_conv2d_pw8v2)) {
            ok = vk_dispatch_kernel(&k_conv2d_pw8, binds, gx, gy, (uint32_t)(n * ((out_c + 7) / 8)));
        }
        if (!ok) ok = vk_dispatch_kernel(&k_conv2d, binds, gx, gy, (uint32_t)(n * out_c));
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
    size_t off;
    size_t end;
    size_t next;
    if (!w || !output_offset || weight_bytes == 0) return 0;
    for (int i = 0; i < wt_cache_n; i++) {
        if (wt_cache[i].src == w && wt_cache[i].d_in == d_in &&
            wt_cache[i].d_out == d_out && wt_cache[i].bytes == weight_bytes) {
            *output_offset = wt_cache[i].off;
            return 1;
        }
    }
    if (wt_cache_n >= WT_CACHE_MAX) return 0;
    off = wt_bump;
    /* Validate the complete aligned allocation before touching mapped memory.
       The transient input arena starts at WEIGHTS_LIMIT. */
    if (off > WEIGHTS_LIMIT || weight_bytes > WEIGHTS_LIMIT - off ||
        !checked_add_size(off, weight_bytes, &end) ||
        !checked_align_size(end, 65536u, &next) || next > WEIGHTS_LIMIT) return 0;
    float* dst = (float*)((char*)io_mapped + off);
    for (int i = 0; i < d_in; i++)
        for (int j = 0; j < d_out; j++)
            dst[j * d_in + i] = w[i * d_out + j];
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
        !io_mapped || io_buffer == VK_NULL_HANDLE ||
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
    memcpy((char*)io_mapped + offset_in, in, in_sz);
    if (b) memcpy((char*)io_mapped + offset_b, b, b_sz);
    else memset((char*)io_mapped + offset_b, 0, b_sz);
    uint32_t params[3] = {(uint32_t)seq, (uint32_t)d_in, (uint32_t)d_out};
    memcpy((char*)io_mapped + offset_params, params, sizeof(params));

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

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd_buf, &begin_info) != VK_SUCCESS) return 0;
    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, selected_pipeline);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                            0, 1, &desc_set, 0, NULL);
    vkCmdDispatch(cmd_buf, groups_x, groups_y, 1);
    if (vkEndCommandBuffer(cmd_buf) != VK_SUCCESS) return 0;

    VkSubmitInfo submit_info = {0};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buf;
    pthread_mutex_lock(&g_vulkan_device.mutex);
    VkResult reset_result = vkResetFences(device, 1, &compute_fence);
    VkResult submit_result = reset_result == VK_SUCCESS
        ? vkQueueSubmit(compute_queue, 1, &submit_info, compute_fence)
        : reset_result;
    pthread_mutex_unlock(&g_vulkan_device.mutex);
    if (submit_result != VK_SUCCESS ||
        vkWaitForFences(device, 1, &compute_fence,
                        VK_TRUE, UINT64_MAX) != VK_SUCCESS) return 0;
    memcpy(out, (char*)io_mapped + offset_out, out_sz);
    return 1;
}
