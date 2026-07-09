#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <dlfcn.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "vulkan_engine.h"

static void* vulkan_lib = NULL;
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = NULL;

#define VK_FUNC(name) static PFN_##name name = NULL;
VK_FUNC(vkCreateInstance)
VK_FUNC(vkDestroyInstance)
VK_FUNC(vkEnumeratePhysicalDevices)
VK_FUNC(vkGetPhysicalDeviceProperties)
VK_FUNC(vkGetPhysicalDeviceQueueFamilyProperties)
VK_FUNC(vkGetPhysicalDeviceMemoryProperties)
VK_FUNC(vkCreateDevice)
VK_FUNC(vkDestroyDevice)
VK_FUNC(vkGetDeviceQueue)
VK_FUNC(vkCreateCommandPool)
VK_FUNC(vkAllocateCommandBuffers)
VK_FUNC(vkCreateBuffer)
VK_FUNC(vkGetBufferMemoryRequirements)
VK_FUNC(vkAllocateMemory)
VK_FUNC(vkBindBufferMemory)
VK_FUNC(vkMapMemory)
VK_FUNC(vkUnmapMemory)
VK_FUNC(vkCreateShaderModule)
VK_FUNC(vkCreateDescriptorSetLayout)
VK_FUNC(vkCreatePipelineLayout)
VK_FUNC(vkCreateComputePipelines)
VK_FUNC(vkCreateDescriptorPool)
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

#define LOAD_GLOBAL(name) name = (PFN_##name)vkGetInstanceProcAddr(NULL, #name);
#define LOAD_INST(name) name = (PFN_##name)vkGetInstanceProcAddr(instance, #name);
#define LOAD_DEV(name) name = (PFN_##name)vkGetInstanceProcAddr(instance, #name); // using instance level is fine

#define ALIGN_UP(x, a) (((x) + ((a) - 1)) & ~((a) - 1))

static VkInstance instance = VK_NULL_HANDLE;
static VkPhysicalDevice physical_device = VK_NULL_HANDLE;
static VkDevice device = VK_NULL_HANDLE;
static VkQueue compute_queue = VK_NULL_HANDLE;
static uint32_t queue_family_index = 0;
static VkPhysicalDeviceMemoryProperties mem_props;

// Memory mapping
static VkBuffer io_buffer;
static VkDeviceMemory io_memory;
static void* io_mapped = NULL;
static size_t io_size = 1024 * 1024 * 512; // overridable with VOLVOX_VULKAN_MB

static VkDescriptorSetLayout desc_layout;
static VkPipelineLayout pipeline_layout;
static VkPipeline matmul_pipeline;
static VkCommandPool cmd_pool;
static VkCommandBuffer cmd_buf;
static VkFence compute_fence = VK_NULL_HANDLE;
static VkDescriptorPool desc_pool;
static VkDescriptorSet desc_set;

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
} VkKernel;

static VkKernel k_conv2d    = {"conv2D",      "native/shaders/spv/conv2D.spv",      5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_c3out16 = {"conv2DRegularC3Out16", "native/shaders/spv/conv2DRegularC3Out16.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_dw4 = {"conv2DDepthwise4", "native/shaders/spv/conv2DDepthwise4.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_dw8 = {"conv2DDepthwise8", "native/shaders/spv/conv2DDepthwise8.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_pw8 = {"conv2DPointwise8", "native/shaders/spv/conv2DPointwise8.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_pw8v2 = {"conv2DPointwise8Vec2", "native/shaders/spv/conv2DPointwise8Vec2.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_pw8v4 = {"conv2DPointwise8Vec4", "native/shaders/spv/conv2DPointwise8Vec4.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_pw16 = {"conv2DPointwise16", "native/shaders/spv/conv2DPointwise16.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv2d_pw16tile = {"conv2DPointwise16Tile", "native/shaders/spv/conv2DPointwise16Tile.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_sigmoid   = {"sigmoid",     "native/shaders/spv/sigmoid.spv",     3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_clip      = {"clip",        "native/shaders/spv/clip.spv",        3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_copy      = {"copy",        "native/shaders/spv/copy.spv",        3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_relu      = {"reLU",        "native/shaders/spv/reLU.spv",        3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_gelu      = {"gELU",        "native/shaders/spv/gELU.spv",        3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_silu      = {"siLU",        "native/shaders/spv/siLU.spv",        3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_tanh      = {"tanh",        "native/shaders/spv/tanh.spv",        3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_hardswish = {"hardSwish",   "native/shaders/spv/hardSwish.spv",   3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_hardsigmoid = {"hardSigmoid", "native/shaders/spv/hardSigmoid.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_leaky_relu = {"leakyReLU",  "native/shaders/spv/leakyReLU.spv",   3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_prelu     = {"pReLU",       "native/shaders/spv/pReLU.spv",       4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_layernorm = {"layerNorm",   "native/shaders/spv/layerNorm.spv",   5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_rmsnorm   = {"rMSNorm",     "native/shaders/spv/rMSNorm.spv",     4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_softmax   = {"softmax",     "native/shaders/spv/softmax.spv",     3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_logsoftmax = {"logSoftmax", "native/shaders/spv/logSoftmax.spv",  3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_reduce    = {"reduce",      "native/shaders/spv/reduce.spv",      3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_globalavg = {"globalAveragePool", "native/shaders/spv/globalAveragePool.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_avgpool   = {"averagePool2D", "native/shaders/spv/averagePool2D.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_batchnorm = {"batchNorm2D", "native/shaders/spv/batchNorm2D.spv", 7, 6, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_embedding = {"embedding",   "native/shaders/spv/embedding.spv",   4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_transpose = {"generalTranspose", "native/shaders/spv/generalTranspose.spv", 3, -1, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_where     = {"where",       "native/shaders/spv/where.spv",       5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_expand    = {"expand",      "native/shaders/spv/expand.spv",      3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_pad       = {"pad",         "native/shaders/spv/pad.spv",         3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_slice     = {"slice",       "native/shaders/spv/slice.spv",       3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_gather    = {"gather",      "native/shaders/spv/gather.spv",      4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_convtranspose = {"convTranspose2D", "native/shaders/spv/convTranspose2D.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_interp1d  = {"interp1D",    "native/shaders/spv/interp1D.spv",    3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_mul       = {"mul",         "native/shaders/spv/mul.spv",         4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_sub       = {"sub",         "native/shaders/spv/sub.spv",         4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_div       = {"div",         "native/shaders/spv/div.spv",         4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_split     = {"split",       "native/shaders/spv/split.spv",       3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_conv1d    = {"conv1D",      "native/shaders/spv/conv1D.spv",      5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_sdpa      = {"sDPA",        "native/shaders/spv/sDPA.spv",        3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_cross_sdpa = {"crossSDPA",  "native/shaders/spv/crossSDPA.spv",   5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_cross_attention = {"crossAttentionF32", "native/shaders/spv/crossAttentionF32.spv", 7, 6, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_quantize  = {"quantizeLinear", "native/shaders/spv/quantizeLinear.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_dequantize = {"dequantizeLinear", "native/shaders/spv/dequantizeLinear.spv", 5, 4, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_spatial_softargmax_y = {"spatialSoftargmaxY", "native/shaders/spv/spatialSoftargmaxY.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_profile_x = {"profileX",    "native/shaders/spv/profileX.spv",    3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_profile_y = {"profileY",    "native/shaders/spv/profileY.spv",    3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_mean_height = {"meanHeight", "native/shaders/spv/meanHeight.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_nms       = {"nonMaxSuppression", "native/shaders/spv/nonMaxSuppression.spv", 4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_add       = {"add",         "native/shaders/spv/add.spv",         4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_add_relu  = {"addRelu",     "native/shaders/spv/addRelu.spv",     4, 3, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_upsample  = {"upsample2x",  "native/shaders/spv/upsample2x.spv",  3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_concat    = {"concatCopy",  "native/shaders/spv/concatCopy.spv",  3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_concat_sigmoid = {"concatSigmoidCopy", "native/shaders/spv/concatSigmoidCopy.spv", 3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_maxpool   = {"maxPool2D",   "native/shaders/spv/maxPool2D.spv",   3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
static VkKernel k_resize    = {"resize",      "native/shaders/spv/resize.spv",      3, 2, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};

#define VK_GRAPH_MAX_TENSORS 4096
#define VK_GRAPH_MAX_DISPATCH_SETS 4096
#define VK_GRAPH_SCRATCH_BYTES ((size_t)1024 * 1024)
#define VK_GRAPH_BASE ((size_t)128 * 1024 * 1024)
#define VK_GRAPH_ALIGN 256

typedef struct {
    const void* host;
    size_t bytes;
    size_t offset;
    int host_dirty;
    int device_dirty;
    int is_weight;
} VkTensorSlot;

static VkTensorSlot graph_slots[VK_GRAPH_MAX_TENSORS];
static int graph_slot_count = 0;
static size_t graph_bump = VK_GRAPH_BASE;
static size_t scratch_bump = 0;

typedef struct {
    VkKernel* kernel;
    VkDescriptorSet desc_set;
} VkGraphDispatchSet;

static VkGraphDispatchSet graph_dispatch_sets[VK_GRAPH_MAX_DISPATCH_SETS];
static int graph_dispatch_set_count = 0;
static int graph_dispatch_set_cursor = 0;
static int graph_cmd_recording = 0;
static int graph_cmd_pending = 0;

static int vk_graph_flush_wait(void);

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


static uint32_t* load_spv(const char* path, size_t* size_out) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("Failed to open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    *size_out = ftell(f);
    rewind(f);
    uint32_t* buf = malloc(*size_out);
    fread(buf, 1, *size_out, f);
    fclose(f);
    return buf;
}

int vk_init() {
    const char* names[] = { "libvulkan.so.1", "libvulkan.so", "vulkan-1.dll" };
    for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        vulkan_lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (vulkan_lib) break;
    }
    if (!vulkan_lib) return -1;

    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(vulkan_lib, "vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddr) return -1;

    LOAD_GLOBAL(vkCreateInstance)
    if (!vkCreateInstance) return -1;

    VkApplicationInfo appInfo = {0};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&createInfo, NULL, &instance) != VK_SUCCESS) return -1;

    LOAD_INST(vkDestroyInstance)
    LOAD_INST(vkEnumeratePhysicalDevices)
    LOAD_INST(vkGetPhysicalDeviceProperties)
    LOAD_INST(vkGetPhysicalDeviceQueueFamilyProperties)
    LOAD_INST(vkGetPhysicalDeviceMemoryProperties)
    LOAD_INST(vkCreateDevice)
    LOAD_INST(vkDestroyDevice)
    LOAD_INST(vkGetDeviceQueue)
    LOAD_INST(vkCreateCommandPool)
    LOAD_INST(vkAllocateCommandBuffers)
    LOAD_INST(vkCreateBuffer)
    LOAD_INST(vkGetBufferMemoryRequirements)
    LOAD_INST(vkAllocateMemory)
    LOAD_INST(vkBindBufferMemory)
    LOAD_INST(vkMapMemory)
    LOAD_INST(vkUnmapMemory)
    LOAD_INST(vkCreateShaderModule)
    LOAD_INST(vkCreateDescriptorSetLayout)
    LOAD_INST(vkCreatePipelineLayout)
    LOAD_INST(vkCreateComputePipelines)
    LOAD_INST(vkCreateDescriptorPool)
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

    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

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

    if (vkCreateDevice(physical_device, &device_info, NULL, &device) != VK_SUCCESS) return -1;
    vkGetDeviceQueue(device, queue_family_index, 0, &compute_queue);

    const char* mb_env = getenv("VOLVOX_VULKAN_MB");
    if (mb_env && mb_env[0]) {
        long mb = strtol(mb_env, NULL, 10);
        if (mb >= 192 && mb <= 4096) io_size = (size_t)mb * 1024 * 1024;
    }

    // Create one mapped IO buffer. This is intentionally simple; a faster backend should
    // split device-local tensors from a smaller staging buffer and add a liveness planner.
    VkBufferCreateInfo buffer_info = {0};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = io_size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &buffer_info, NULL, &io_buffer);

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device, io_buffer, &mem_reqs);
    VkMemoryAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_preferred_memory_type(
        mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    if (vkAllocateMemory(device, &alloc_info, NULL, &io_memory) != VK_SUCCESS) {
        printf("[VolvoxAI GPU] vkAllocateMemory failed.\n");
        return -1;
    }
    vkBindBufferMemory(device, io_buffer, io_memory, 0);
    if (vkMapMemory(device, io_memory, 0, io_size, 0, &io_mapped) != VK_SUCCESS) {
        printf("[VolvoxAI GPU] vkMapMemory failed.\n");
        return -1;
    }
    if (!io_mapped) {
        printf("[VolvoxAI GPU] io_mapped is NULL after successful vkMapMemory!\n");
        return -1;
    }

    // Setup Pipeline
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

    size_t spv_size = 0;
    uint32_t* spv_code = load_spv("native/shaders/spv/linearF32.spv", &spv_size);
    if (!spv_code) return -1;
    VkShaderModuleCreateInfo shader_info = {0};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = spv_size;
    shader_info.pCode = spv_code;
    VkShaderModule shader_module;
    vkCreateShaderModule(device, &shader_info, NULL, &shader_module);
    free(spv_code);

    VkComputePipelineCreateInfo pipeline_info = {0};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shader_module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = pipeline_layout;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &matmul_pipeline);

    // Descriptor Pool
    VkDescriptorPoolSize pool_sizes[2] = {0};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[0].descriptorCount = 16384;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[1].descriptorCount = 4096;
    VkDescriptorPoolCreateInfo pool_info = {0};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = VK_GRAPH_MAX_DISPATCH_SETS + 64;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = pool_sizes;
    vkCreateDescriptorPool(device, &pool_info, NULL, &desc_pool);

    VkDescriptorSetAllocateInfo alloc_set_info = {0};
    alloc_set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_set_info.descriptorPool = desc_pool;
    alloc_set_info.descriptorSetCount = 1;
    alloc_set_info.pSetLayouts = &desc_layout;
    vkAllocateDescriptorSets(device, &alloc_set_info, &desc_set);

    // Descriptor update deferred to dispatch

    VkCommandPoolCreateInfo cmd_pool_info = {0};
    cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cmd_pool_info.queueFamilyIndex = queue_family_index;
    vkCreateCommandPool(device, &cmd_pool_info, NULL, &cmd_pool);

    VkCommandBufferAllocateInfo cmd_buf_info = {0};
    cmd_buf_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_buf_info.commandPool = cmd_pool;
    cmd_buf_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_buf_info.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &cmd_buf_info, &cmd_buf);

    VkFenceCreateInfo fence_info = {0};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device, &fence_info, NULL, &compute_fence);

    printf("[VolvoxAI GPU] Vulkan Compute initialized successfully! Device: %s\n", best_props.deviceName);
    return 0;
}

void vk_cleanup() {
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device); // if available
    }
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, NULL);
    }
    if (vulkan_lib) dlclose(vulkan_lib);
}

static int vk_is_ready(void) {
    return device != VK_NULL_HANDLE && io_buffer != VK_NULL_HANDLE && io_mapped != NULL;
}

static int graph_find_slot(const void* host) {
    if (!host) return -1;
    for (int i = 0; i < graph_slot_count; i++) {
        if (graph_slots[i].host == host) return i;
    }
    return -1;
}

static VkTensorSlot* graph_get_slot(const void* host, size_t bytes, int is_weight) {
    if (!vk_is_ready() || !host || bytes == 0) return NULL;
    int idx = graph_find_slot(host);
    if (idx >= 0) {
        VkTensorSlot* s = &graph_slots[idx];
        if (bytes <= s->bytes) return s;
        s->bytes = bytes;
        s->host_dirty = 1;
        s->device_dirty = 0;
    } else {
        if (graph_slot_count >= VK_GRAPH_MAX_TENSORS) return NULL;
        idx = graph_slot_count++;
        graph_slots[idx] = (VkTensorSlot){0};
        graph_slots[idx].host = host;
        graph_slots[idx].bytes = bytes;
        graph_slots[idx].host_dirty = 1;
        graph_slots[idx].is_weight = is_weight;
    }

    size_t limit = io_size > VK_GRAPH_SCRATCH_BYTES ? io_size - VK_GRAPH_SCRATCH_BYTES : 0;
    graph_bump = ALIGN_UP(graph_bump, VK_GRAPH_ALIGN);
    if (graph_bump + bytes > limit) {
        printf("[VolvoxAI GPU] Vulkan graph arena exhausted (%zu / %zu bytes). Set VOLVOX_VULKAN_MB higher.\n",
               graph_bump + bytes, io_size);
        return NULL;
    }
    graph_slots[idx].offset = graph_bump;
    graph_bump = ALIGN_UP(graph_bump + bytes, VK_GRAPH_ALIGN);
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
        scratch_bump = ALIGN_UP(io_size - VK_GRAPH_SCRATCH_BYTES, VK_GRAPH_ALIGN);
    }
}

static size_t graph_scratch_alloc(size_t bytes) {
    scratch_bump = ALIGN_UP(scratch_bump, VK_GRAPH_ALIGN);
    if (scratch_bump + bytes > io_size) return SIZE_MAX;
    size_t off = scratch_bump;
    scratch_bump = ALIGN_UP(scratch_bump + bytes, VK_GRAPH_ALIGN);
    return off;
}

static size_t graph_scratch_upload(const void* data, size_t bytes) {
    size_t off = graph_scratch_alloc(bytes);
    if (off == SIZE_MAX) return SIZE_MAX;
    memcpy((char*)io_mapped + off, data, bytes);
    return off;
}

void vk_graph_reset(void) {
    vk_graph_flush_wait();
    graph_slot_count = 0;
    graph_bump = VK_GRAPH_BASE;
    scratch_bump = 0;
    graph_dispatch_set_cursor = 0;
}

void vk_graph_begin_forward(void) {
    vk_graph_flush_wait();
    scratch_bump = 0;
    graph_dispatch_set_cursor = 0;
}

int vk_graph_end_forward(void) {
    return vk_graph_flush_wait() ? 0 : -1;
}

void vk_graph_mark_host(const void* host, size_t bytes, int is_weight) {
    VkTensorSlot* s = graph_get_slot(host, bytes, is_weight);
    if (!s) return;
    s->host_dirty = 1;
    s->device_dirty = 0;
}

int vk_graph_sync_host(const void* host, size_t bytes, int is_weight) {
    (void)is_weight;
    if (!vk_is_ready() || !host || bytes == 0) return 0;
    int idx = graph_find_slot(host);
    if (idx < 0) return 0;
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

static int vk_prepare_kernel(VkKernel* k) {
    if (!vk_is_ready() || !k) return 0;
    if (k->ready) return 1;

    VkDescriptorSetLayoutBinding bindings[8];
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
    if (vkCreateDescriptorSetLayout(device, &layout_info, NULL, &k->desc_layout) != VK_SUCCESS) return 0;

    VkPipelineLayoutCreateInfo pl_info = {0};
    pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_info.setLayoutCount = 1;
    pl_info.pSetLayouts = &k->desc_layout;
    if (vkCreatePipelineLayout(device, &pl_info, NULL, &k->pipeline_layout) != VK_SUCCESS) return 0;

    size_t spv_size = 0;
    uint32_t* spv = load_spv(k->path, &spv_size);
    if (!spv) return 0;
    VkShaderModuleCreateInfo shader_info = {0};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = spv_size;
    shader_info.pCode = spv;
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkResult sm_res = vkCreateShaderModule(device, &shader_info, NULL, &shader_module);
    free(spv);
    if (sm_res != VK_SUCCESS) return 0;

    VkComputePipelineCreateInfo pipe_info = {0};
    pipe_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipe_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipe_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipe_info.stage.module = shader_module;
    pipe_info.stage.pName = "main";
    pipe_info.layout = k->pipeline_layout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipe_info, NULL, &k->pipeline) != VK_SUCCESS) return 0;

    VkDescriptorSetAllocateInfo set_info = {0};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = desc_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &k->desc_layout;
    if (vkAllocateDescriptorSets(device, &set_info, &k->desc_set) != VK_SUCCESS) return 0;

    k->ready = 1;
    return 1;
}

typedef struct {
    size_t offset;
    size_t bytes;
} VkGraphBinding;

static VkDescriptorSet vk_graph_dispatch_set(VkKernel* k) {
    if (!k || !k->desc_layout) return VK_NULL_HANDLE;
    if (graph_dispatch_set_cursor >= VK_GRAPH_MAX_DISPATCH_SETS) return VK_NULL_HANDLE;
    int idx = graph_dispatch_set_cursor++;
    VkGraphDispatchSet* s = &graph_dispatch_sets[idx];
    if (s->desc_set != VK_NULL_HANDLE && s->kernel == k) return s->desc_set;

    VkDescriptorSetAllocateInfo set_info = {0};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = desc_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &k->desc_layout;
    if (vkAllocateDescriptorSets(device, &set_info, &s->desc_set) != VK_SUCCESS) return VK_NULL_HANDLE;
    s->kernel = k;
    if (idx >= graph_dispatch_set_count) graph_dispatch_set_count = idx + 1;
    return s->desc_set;
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
    vkResetFences(device, 1, &compute_fence);
    if (vkQueueSubmit(compute_queue, 1, &submit_info, compute_fence) != VK_SUCCESS) {
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

static int vk_dispatch_kernel(VkKernel* k, const VkGraphBinding* binds,
                              uint32_t gx, uint32_t gy, uint32_t gz) {
    if (!vk_prepare_kernel(k)) return 0;
    if (!binds || gx == 0 || gy == 0 || gz == 0) return 0;
    VkDescriptorSet dispatch_set = vk_graph_dispatch_set(k);
    if (dispatch_set == VK_NULL_HANDLE) return 0;

    VkDescriptorBufferInfo infos[8];
    VkWriteDescriptorSet writes[8];
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
    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline_layout, 0, 1, &dispatch_set, 0, NULL);
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

int vk_graph_alias_f32(const float* in, float* out, long n) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    if (!src) return 0;
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!dst) return 0;
    dst->offset = src->offset;
    dst->bytes = bytes;
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
int vk_graph_gelu_f32(const float* in, float* out, long n) { return vk_graph_unary_size_f32(&k_gelu, in, out, n); }
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
    uint32_t params[2] = {(uint32_t)n, (uint32_t)channels};
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
                           float* out, int rows, int d_model) {
    if (rows <= 0 || d_model <= 0 || !in || !weight || !bias || !out) return 0;
    size_t bytes = (size_t)rows * d_model * sizeof(float);
    size_t wbytes = (size_t)d_model * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, bytes, 0);
    VkTensorSlot* w = graph_ensure_device(weight, wbytes, 1);
    VkTensorSlot* b = graph_ensure_device(bias, wbytes, 1);
    VkTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !w || !b || !dst) return 0;
    uint32_t params[2] = {(uint32_t)rows, (uint32_t)d_model};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
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

int vk_graph_embedding_f32(const float* tokens, const float* weight, float* out,
                           int tokens_len, int d_model, int vocab_size) {
    if (tokens_len <= 0 || d_model <= 0 || vocab_size <= 0 || !tokens || !weight || !out) return 0;
    size_t tbytes = (size_t)tokens_len * sizeof(float);
    size_t wbytes = (size_t)vocab_size * d_model * sizeof(float);
    size_t obytes = (size_t)tokens_len * d_model * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* st = graph_ensure_device(tokens, tbytes, 0);
    VkTensorSlot* sw = graph_ensure_device(weight, wbytes, 1);
    VkTensorSlot* so = graph_output_slot(out, obytes);
    if (!st || !sw || !so) return 0;
    uint32_t params[2] = {(uint32_t)tokens_len, (uint32_t)d_model};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {st->offset, tbytes}, {sw->offset, wbytes}, {so->offset, obytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_embedding, binds, ((uint32_t)tokens_len + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(so);
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
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0 || out_h <= 0 || out_w <= 0 || !in || !out) return 0;
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
    if (!in || !out || !in_shape || !out_shape || in_rank <= 0 || out_rank <= 0 || in_rank > 4 || out_rank > 4) return 0;
    uint32_t is[4], os[4];
    pad4_shape(in_shape, in_rank, is);
    pad4_shape(out_shape, out_rank, os);
    size_t in_elems = (size_t)is[0] * is[1] * is[2] * is[3];
    size_t out_elems = (size_t)os[0] * os[1] * os[2] * os[3];
    size_t in_bytes = in_elems * sizeof(float);
    size_t out_bytes = out_elems * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[8] = {is[0], is[1], is[2], is[3], os[0], os[1], os[2], os[3]};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_expand, binds, ((uint32_t)out_elems + 63u) / 64u, 1, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_gather_axis0_f32(const float* in, const float* indices, float* out,
                              int row_size, int input_rows, int num_idx) {
    if (row_size <= 0 || input_rows <= 0 || num_idx <= 0 || !in || !indices || !out) return 0;
    long total = (long)row_size * num_idx;
    size_t in_bytes = (size_t)input_rows * row_size * sizeof(float);
    size_t idx_bytes = (size_t)num_idx * sizeof(float);
    size_t out_bytes = (size_t)total * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* idx = graph_ensure_device(indices, idx_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !idx || !dst) return 0;
    uint32_t params[3] = {(uint32_t)row_size, (uint32_t)num_idx, (uint32_t)total};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {src->offset, in_bytes}, {idx->offset, idx_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_gather, binds, ((uint32_t)total + 63u) / 64u, 1, 1)) return 0;
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
                        float* out, long out_numel, int op) {
    if (!a || !b || !out || a_numel <= 0 || b_numel <= 0 || out_numel <= 0) return 0;
    if (a_numel > out_numel || b_numel > out_numel) return 0;
    VkKernel* kernel = op == 1 ? &k_sub : (op == 2 ? &k_div : &k_mul);
    size_t a_bytes = (size_t)a_numel * sizeof(float);
    size_t b_bytes = (size_t)b_numel * sizeof(float);
    size_t out_bytes = (size_t)out_numel * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* sa = graph_ensure_device(a, a_bytes, 0);
    VkTensorSlot* sb = graph_ensure_device(b, b_bytes, 0);
    VkTensorSlot* so = graph_output_slot(out, out_bytes);
    if (!sa || !sb || !so) return 0;
    uint32_t params[5] = {(uint32_t)out_numel, b_numel == 1 ? 1u : 0u, (uint32_t)b_numel,
                          (uint32_t)a_numel, a_numel == 1 ? 1u : 0u};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {sa->offset, a_bytes}, {sb->offset, b_bytes}, {so->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(kernel, binds, ((uint32_t)out_numel + 63u) / 64u, 1, 1)) return 0;
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
                        int in_c, int in_l, int out_c, int out_l, int kernel,
                        int stride, int pad, int relu) {
    if (!in || !weight || !out || in_c <= 0 || in_l <= 0 || out_c <= 0 || out_l <= 0 ||
        kernel <= 0 || stride <= 0 || pad < 0 || relu < 0 || relu > 1) return 0;
    size_t in_bytes = (size_t)in_c * in_l * sizeof(float);
    size_t wbytes = (size_t)out_c * in_c * kernel * sizeof(float);
    size_t bbytes = (size_t)out_c * sizeof(float);
    size_t out_bytes = (size_t)out_c * out_l * sizeof(float);
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
    uint32_t params[8] = {(uint32_t)in_c, (uint32_t)in_l, (uint32_t)out_c, (uint32_t)kernel,
                          (uint32_t)stride, (uint32_t)pad, (uint32_t)relu, 0u};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[5] = {
        {src->offset, in_bytes}, {sw->offset, wbytes}, {b_off, bbytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_conv1d, binds, ((uint32_t)out_l + 63u) / 64u, (uint32_t)out_c, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_sdpa_f32(const float* qkv, float* out, int seq_len, int d_model,
                      int num_heads, int head_dim, float scale) {
    if (!qkv || !out || seq_len <= 0 || d_model <= 0 || num_heads <= 0 || head_dim <= 0 ||
        head_dim > 64 || d_model != num_heads * head_dim) return 0;
    size_t qkv_bytes = (size_t)seq_len * 3u * (size_t)d_model * sizeof(float);
    size_t out_bytes = (size_t)seq_len * (size_t)d_model * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(qkv, qkv_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    struct { uint32_t seq_len, d_model, num_heads, head_dim; float scale; uint32_t pad[3]; } params =
        {(uint32_t)seq_len, (uint32_t)d_model, (uint32_t)num_heads, (uint32_t)head_dim, scale, {0, 0, 0}};
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {{src->offset, qkv_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}};
    if (!vk_dispatch_kernel(&k_sdpa, binds, ((uint32_t)seq_len + 63u) / 64u, (uint32_t)num_heads, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_cross_sdpa_f32(const float* q, const float* k, const float* v, float* out,
                            int seq_q, int seq_kv, int d_model, int num_heads,
                            int head_dim, float scale) {
    if (!q || !k || !v || !out || seq_q <= 0 || seq_kv <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 || head_dim > 64 || d_model != num_heads * head_dim) return 0;
    size_t q_bytes = (size_t)seq_q * (size_t)d_model * sizeof(float);
    size_t kv_bytes = (size_t)seq_kv * (size_t)d_model * sizeof(float);
    size_t out_bytes = q_bytes;
    graph_scratch_begin();
    VkTensorSlot* sq = graph_ensure_device(q, q_bytes, 0);
    VkTensorSlot* sk = graph_ensure_device(k, kv_bytes, 0);
    VkTensorSlot* sv = graph_ensure_device(v, kv_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sq || !sk || !sv || !dst) return 0;
    struct { uint32_t seq_q, seq_kv, d_model, num_heads, head_dim; float scale; uint32_t pad[2]; } params =
        {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model, (uint32_t)num_heads, (uint32_t)head_dim, scale, {0, 0}};
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[5] = {
        {sq->offset, q_bytes}, {sk->offset, kv_bytes}, {sv->offset, kv_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_cross_sdpa, binds, ((uint32_t)seq_q + 63u) / 64u, (uint32_t)num_heads, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_cross_attention_f32(const float* q, const float* kv, const float* weight,
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
        uint32_t has_scale, has_bias;
    } params = {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model, (uint32_t)num_heads,
                (uint32_t)head_dim, 1.0f / sqrtf((float)head_dim),
                (uint32_t)(has_scale && scale), (uint32_t)(has_bias && bias)};
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[7] = {
        {sq->offset, q_bytes}, {skv->offset, kv_bytes}, {sw->offset, wbytes},
        {ss->offset, ss->bytes}, {sb->offset, sb->bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_cross_attention, binds, ((uint32_t)seq_q + 63u) / 64u, (uint32_t)num_heads, 1)) return 0;
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
                                   int h, int w, int c, long out_elems,
                                   uint32_t gx, uint32_t gy) {
    if (!kernel || !in || !out || h <= 0 || w <= 0 || c <= 0 || out_elems <= 0) return 0;
    size_t in_bytes = (size_t)h * (size_t)w * (size_t)c * sizeof(float);
    size_t out_bytes = (size_t)out_elems * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)h, (uint32_t)w, (uint32_t)c, 0u};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {{src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}};
    if (!vk_dispatch_kernel(kernel, binds, gx, gy, 1)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_spatial_softargmax_y_f32(const float* in, float* out, int h, int w, int c) {
    return vk_graph_profile_common(&k_spatial_softargmax_y, in, out, h, w, c,
                                   (long)c * w, ((uint32_t)w + 63u) / 64u, (uint32_t)c);
}

int vk_graph_profile_x_f32(const float* in, float* out, int h, int w, int c) {
    return vk_graph_profile_common(&k_profile_x, in, out, h, w, c,
                                   (long)2 * c * w, ((uint32_t)w + 63u) / 64u, (uint32_t)c);
}

int vk_graph_profile_y_f32(const float* in, float* out, int h, int w, int c) {
    return vk_graph_profile_common(&k_profile_y, in, out, h, w, c,
                                   (long)2 * c * h, ((uint32_t)h + 63u) / 64u, (uint32_t)c);
}

int vk_graph_mean_height_f32(const float* in, float* out, int h, int w, int c) {
    return vk_graph_profile_common(&k_mean_height, in, out, h, w, c,
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

int vk_graph_concat_flat_f32(const float** inputs, const long* sizes, int count, float* out) {
    if (!inputs || !sizes || !out || count <= 0) return 0;
    long total = 0;
    for (int i = 0; i < count; i++) {
        if (!inputs[i] || sizes[i] <= 0) return 0;
        total += sizes[i];
    }
    size_t out_bytes = (size_t)total * sizeof(float);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!dst) return 0;
    long offset = 0;
    for (int i = 0; i < count; i++) {
        size_t in_bytes = (size_t)sizes[i] * sizeof(float);
        graph_scratch_begin();
        VkTensorSlot* src = graph_ensure_device(inputs[i], in_bytes, 0);
        if (!src) return 0;
        uint32_t params[2] = {(uint32_t)sizes[i], (uint32_t)offset};
        size_t p_off = graph_scratch_upload(params, sizeof(params));
        if (p_off == SIZE_MAX) return 0;
        VkGraphBinding binds[3] = {
            {src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
        };
        if (!vk_dispatch_kernel(&k_concat, binds, ((uint32_t)sizes[i] + 63u) / 64u, 1, 1)) return 0;
        offset += sizes[i];
    }
    graph_mark_device(dst);
    return 1;
}

int vk_graph_concat_sigmoid_flat_f32(const float** inputs, const long* sizes, int count, float* out) {
    if (!inputs || !sizes || !out || count <= 0) return 0;
    long total = 0;
    for (int i = 0; i < count; i++) {
        if (!inputs[i] || sizes[i] <= 0) return 0;
        total += sizes[i];
    }
    size_t out_bytes = (size_t)total * sizeof(float);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!dst) return 0;
    long offset = 0;
    for (int i = 0; i < count; i++) {
        size_t in_bytes = (size_t)sizes[i] * sizeof(float);
        graph_scratch_begin();
        VkTensorSlot* src = graph_ensure_device(inputs[i], in_bytes, 0);
        if (!src) return 0;
        uint32_t params[2] = {(uint32_t)sizes[i], (uint32_t)offset};
        size_t p_off = graph_scratch_upload(params, sizeof(params));
        if (p_off == SIZE_MAX) return 0;
        VkGraphBinding binds[3] = {
            {src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
        };
        if (!vk_dispatch_kernel(&k_concat_sigmoid, binds, ((uint32_t)sizes[i] + 63u) / 64u, 1, 1)) return 0;
        offset += sizes[i];
    }
    graph_mark_device(dst);
    return 1;
}

int vk_graph_maxpool2d_f32(const float* in, float* out, int h, int width, int c,
                           int out_h, int out_w, int ky, int kx, int sy, int sx,
                           int py, int px) {
    if (!in || !out || c <= 0 || h <= 0 || width <= 0 || out_h <= 0 || out_w <= 0 ||
        ky <= 0 || kx <= 0 || sy <= 0 || sx <= 0) {
        return 0;
    }
    size_t in_bytes = (size_t)h * width * c * sizeof(float);
    size_t out_bytes = (size_t)out_h * out_w * c * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[11] = {
        (uint32_t)h, (uint32_t)width, (uint32_t)c,
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
                            ((uint32_t)out_h + 7u) / 8u, (uint32_t)c)) return 0;
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
        dy <= 0 || dx <= 0 || groups <= 0 || relu > 2) {
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
    if (groups == 1 && c == 3 && (out_c & 15) == 0) {
        kernel = &k_conv2d_c3out16;
        gz = (uint32_t)(n * (out_c / 16));
    } else if (groups == c && out_c == c) {
        if ((out_c & 7) == 0) {
            kernel = &k_conv2d_dw8;
            gz = (uint32_t)(n * ((out_c + 7) / 8));
        } else if ((out_c & 3) == 0) {
            kernel = &k_conv2d_dw4;
            gz = (uint32_t)(n * ((out_c + 3) / 4));
        }
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
    int ok = vk_dispatch_kernel(kernel, binds, gx, gy, gz);
    if (!ok && kernel != &k_conv2d) {
        if (kernel == &k_conv2d_pw16tile) ok = vk_dispatch_kernel(&k_conv2d_pw16, binds, gx, gy, (uint32_t)(n * (out_c / 16)));
        if (!ok && kernel == &k_conv2d_dw8 && (out_c & 3) == 0) ok = vk_dispatch_kernel(&k_conv2d_dw4, binds, gx, gy, (uint32_t)(n * ((out_c + 3) / 4)));
        if (!ok && (kernel == &k_conv2d_pw8v4 || kernel == &k_conv2d_pw8v2)) {
            ok = vk_dispatch_kernel(&k_conv2d_pw8, binds, gx, gy, (uint32_t)(n * ((out_c + 7) / 8)));
        }
        if (!ok) ok = vk_dispatch_kernel(&k_conv2d, binds, gx, gy, (uint32_t)(n * out_c));
    }
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

// Weights are constant across all tokens, so each (transposed) weight is uploaded into
// a RESIDENT region of the GPU buffer exactly once — the 12.8MB lm_head weight is not
// re-copied every token. linearF32.wgsl reads weight[col*d_in + k] (layout [d_out,d_in]);
// the exported weight is [d_in, d_out], so we transpose straight into GPU memory.
#define WT_CACHE_MAX 512
#define WEIGHTS_LIMIT ((size_t)64 * 1024 * 1024)   // weights in [0,64MB); transient above
static struct { const float* src; size_t off; } wt_cache[WT_CACHE_MAX];
static int wt_cache_n = 0;
static size_t wt_bump = 0;

static size_t upload_weight(const float* w, int d_in, int d_out) {
    for (int i = 0; i < wt_cache_n; i++) if (wt_cache[i].src == w) return wt_cache[i].off;
    size_t off = wt_bump;
    float* dst = (float*)((char*)io_mapped + off);
    for (int i = 0; i < d_in; i++)
        for (int j = 0; j < d_out; j++)
            dst[j * d_in + i] = w[i * d_out + j];
    wt_bump = (off + (size_t)d_in * d_out * 4 + 65535) & ~((size_t)65535);
    if (wt_cache_n < WT_CACHE_MAX && wt_bump <= WEIGHTS_LIMIT) { wt_cache[wt_cache_n].src = w; wt_cache[wt_cache_n].off = off; wt_cache_n++; }
    return off;
}

// Reset the resident-weight arena (called when the engine reloads, so freed weight
// pointers are never matched against a newly-loaded model that reused the address).
void vk_free_weight_cache(void) { wt_cache_n = 0; wt_bump = 0; }

// Vulkan MatMul dispatch (reuses the WGSL linearF32 shader compiled to SPIR-V).
void vk_matmul(const float* in, const float* w, const float* b, float* out, int seq, int d_in, int d_out) {
    vk_graph_flush_wait();
    // Align offsets to 65536 bytes to exceed any minStorageBufferOffsetAlignment.
    #define ALIGN_UP(x, a) (((x) + ((a) - 1)) & ~((a) - 1))

    size_t in_sz = (size_t)seq * d_in * 4;
    size_t w_sz = (size_t)d_in * d_out * 4;
    size_t b_sz = (size_t)d_out * 4;   // always reserve a full bias row (zeros if no bias)
    size_t out_sz = (size_t)seq * d_out * 4;
    size_t param_sz = 12; // seq_len, d_in, d_out

    size_t offset_w = upload_weight(w, d_in, d_out);   // resident, uploaded once

    // Transient region above the weights arena, rewritten each call (only small buffers).
    size_t offset_in = WEIGHTS_LIMIT;
    size_t offset_dummy = ALIGN_UP(offset_in + in_sz, 65536);
    size_t offset_b = ALIGN_UP(offset_dummy + 4, 65536); // dummy scale (binding 2)
    size_t offset_out = ALIGN_UP(offset_b + b_sz, 65536);
    size_t offset_params = ALIGN_UP(offset_out + out_sz, 65536);

    memcpy((char*)io_mapped + offset_in, in, in_sz);
    // Bias binding must cover all d_out columns the shader reads; zero it when absent.
    if (b) memcpy((char*)io_mapped + offset_b, b, b_sz);
    else   memset((char*)io_mapped + offset_b, 0, b_sz);

    uint32_t params[3] = {seq, d_in, d_out};
    memcpy((char*)io_mapped + offset_params, params, 12);

    VkDescriptorBufferInfo buf_infos[6];
    VkWriteDescriptorSet writes[6];
    buf_infos[0] = (VkDescriptorBufferInfo){io_buffer, offset_in, in_sz};
    buf_infos[1] = (VkDescriptorBufferInfo){io_buffer, offset_w, w_sz};
    buf_infos[2] = (VkDescriptorBufferInfo){io_buffer, offset_dummy, 4}; // Dummy scale
    buf_infos[3] = (VkDescriptorBufferInfo){io_buffer, offset_b, b_sz};
    buf_infos[4] = (VkDescriptorBufferInfo){io_buffer, offset_out, out_sz};
    buf_infos[5] = (VkDescriptorBufferInfo){io_buffer, offset_params, param_sz};
    
    for(int i=0; i<6; i++) {
        writes[i] = (VkWriteDescriptorSet){0};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = desc_set;
        writes[i].dstBinding = i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorType = (i == 5) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(device, 6, writes, 0, NULL);

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_buf, &begin_info);

    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, matmul_pipeline);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &desc_set, 0, NULL);

    // linearF32.wgsl uses @workgroup_size(64, 1, 1).
    // global_id.x is col (d_out), global_id.y is row (seq)
    // So we dispatch (d_out + 63)/64 on X, seq on Y, 1 on Z
    vkCmdDispatch(cmd_buf, (d_out + 63) / 64, seq, 1);
    vkEndCommandBuffer(cmd_buf);

    VkSubmitInfo submit_info = {0};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buf;
    // Wait on a per-submission fence rather than draining the whole queue.
    vkResetFences(device, 1, &compute_fence);
    vkQueueSubmit(compute_queue, 1, &submit_info, compute_fence);
    vkWaitForFences(device, 1, &compute_fence, VK_TRUE, UINT64_MAX);

    memcpy(out, (char*)io_mapped + offset_out, out_sz);
}
