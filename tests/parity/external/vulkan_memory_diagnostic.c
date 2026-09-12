#define _GNU_SOURCE
#include "vulkan_core.h"
#include <dlfcn.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Test-only shim. Default mode forwards every result unchanged. An explicit
 * diagnostic flag hides one optional extension to isolate wgpu's pre-checks;
 * actual Vulkan allocation results are never changed. */
static pthread_once_t init_once = PTHREAD_ONCE_INIT;
static PFN_vkGetInstanceProcAddr real_gipa;
static PFN_vkGetDeviceProcAddr real_gdpa;
static PFN_vkGetPhysicalDeviceMemoryProperties2 real_memory2;
static PFN_vkGetPhysicalDeviceMemoryProperties2KHR real_memory2_khr;
static PFN_vkCreateDevice real_create_device;
static PFN_vkEnumerateDeviceExtensionProperties real_extensions;

static void init_loader(void) {
    void *loader = dlopen("/lib/x86_64-linux-gnu/libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!loader) { fprintf(stderr, "trace loader: %s\n", dlerror()); abort(); }
    real_gipa = (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
    real_gdpa = (PFN_vkGetDeviceProcAddr)dlsym(loader, "vkGetDeviceProcAddr");
    if (!real_gipa || !real_gdpa) abort();
}

static void log_memory(VkPhysicalDevice physical, VkPhysicalDeviceMemoryProperties2 *properties) {
    const VkPhysicalDeviceMemoryBudgetPropertiesEXT *budget = NULL;
    for (const VkBaseOutStructure *p = properties->pNext; p; p = p->pNext) {
        if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT)
            budget = (const VkPhysicalDeviceMemoryBudgetPropertiesEXT *)p;
    }
    if (!budget) return;
    const VkPhysicalDeviceMemoryProperties *memory = &properties->memoryProperties;
    for (uint32_t i = 0; i < memory->memoryHeapCount; i++) {
        uint32_t flags = 0;
        for (uint32_t j = 0; j < memory->memoryTypeCount; j++)
            if (memory->memoryTypes[j].heapIndex == i)
                flags |= memory->memoryTypes[j].propertyFlags;
        fprintf(stderr, "VKTRACE {\"event\":\"budget\",\"physical\":\"%p\",\"heap\":%u,\"types_flags\":%u,\"size\":%" PRIu64 ",\"budget\":%" PRIu64 ",\"usage\":%" PRIu64 "}\n",
            (void *)physical, i, flags, memory->memoryHeaps[i].size,
            budget->heapBudget[i], budget->heapUsage[i]);
    }
}

static VKAPI_ATTR void VKAPI_CALL trace_memory2(VkPhysicalDevice physical, VkPhysicalDeviceMemoryProperties2 *properties) {
    real_memory2(physical, properties);
    log_memory(physical, properties);
}
static VKAPI_ATTR void VKAPI_CALL trace_memory2_khr(VkPhysicalDevice physical, VkPhysicalDeviceMemoryProperties2 *properties) {
    real_memory2_khr(physical, properties);
    log_memory(physical, properties);
}
static VKAPI_ATTR VkResult VKAPI_CALL trace_create_device(VkPhysicalDevice physical,
    const VkDeviceCreateInfo *info, const VkAllocationCallbacks *callbacks, VkDevice *device) {
    VkResult result = real_create_device(physical, info, callbacks, device);
    fprintf(stderr, "VKTRACE {\"event\":\"create_device\",\"physical\":\"%p\",\"device\":\"%p\",\"result\":%d}\n",
        (void *)physical, result == VK_SUCCESS ? (void *)*device : NULL, result);
    return result;
}
static VKAPI_ATTR VkResult VKAPI_CALL trace_allocate_memory(VkDevice device,
    const VkMemoryAllocateInfo *info, const VkAllocationCallbacks *callbacks, VkDeviceMemory *memory) {
    PFN_vkAllocateMemory fn = (PFN_vkAllocateMemory)real_gdpa(device, "vkAllocateMemory");
    VkResult result = fn(device, info, callbacks, memory);
    fprintf(stderr, "VKTRACE {\"event\":\"allocate_memory\",\"device\":\"%p\",\"size\":%" PRIu64 ",\"memory_type\":%u,\"result\":%d}\n",
        (void *)device, info->allocationSize, info->memoryTypeIndex, result);
    return result;
}
static VKAPI_ATTR VkResult VKAPI_CALL trace_create_buffer(VkDevice device,
    const VkBufferCreateInfo *info, const VkAllocationCallbacks *callbacks, VkBuffer *buffer) {
    PFN_vkCreateBuffer fn = (PFN_vkCreateBuffer)real_gdpa(device, "vkCreateBuffer");
    VkResult result = fn(device, info, callbacks, buffer);
    fprintf(stderr, "VKTRACE {\"event\":\"create_buffer\",\"device\":\"%p\",\"size\":%" PRIu64 ",\"usage\":%u,\"result\":%d}\n",
        (void *)device, info->size, info->usage, result);
    return result;
}
static VKAPI_ATTR void VKAPI_CALL trace_free_memory(VkDevice device, VkDeviceMemory memory,
    const VkAllocationCallbacks *callbacks) {
    PFN_vkFreeMemory fn = (PFN_vkFreeMemory)real_gdpa(device, "vkFreeMemory");
    fn(device, memory, callbacks);
    fprintf(stderr, "VKTRACE {\"event\":\"free_memory\",\"device\":\"%p\"}\n", (void *)device);
}
static VKAPI_ATTR void VKAPI_CALL trace_destroy_device(VkDevice device, const VkAllocationCallbacks *callbacks) {
    PFN_vkDestroyDevice fn = (PFN_vkDestroyDevice)real_gdpa(device, "vkDestroyDevice");
    fn(device, callbacks);
    fprintf(stderr, "VKTRACE {\"event\":\"destroy_device\",\"device\":\"%p\"}\n", (void *)device);
}
static VKAPI_ATTR VkResult VKAPI_CALL trace_extensions(VkPhysicalDevice physical,
    const char *layer, uint32_t *count, VkExtensionProperties *properties) {
    VkResult result = real_extensions(physical, layer, count, properties);
    const char *disabled = getenv("VOLVOXAI_DIAGNOSTIC_DISABLE_BUDGET");
    if (properties && (result == VK_SUCCESS || result == VK_INCOMPLETE)
        && disabled && strcmp(disabled, "1") == 0) {
        for (uint32_t i = 0; i < *count; i++) {
            if (strcmp(properties[i].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) != 0) continue;
            memmove(&properties[i], &properties[i + 1], (*count - i - 1) * sizeof(*properties));
            --*count;
            fprintf(stderr, "VKTRACE {\"event\":\"diagnostic_disable_memory_budget\"}\n");
            break;
        }
    }
    return result;
}
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL trace_gdpa(VkDevice device, const char *name) {
    PFN_vkVoidFunction fn = real_gdpa(device, name);
    if (fn && strcmp(name, "vkAllocateMemory") == 0) return (PFN_vkVoidFunction)trace_allocate_memory;
    if (fn && strcmp(name, "vkCreateBuffer") == 0) return (PFN_vkVoidFunction)trace_create_buffer;
    if (fn && strcmp(name, "vkFreeMemory") == 0) return (PFN_vkVoidFunction)trace_free_memory;
    if (fn && strcmp(name, "vkDestroyDevice") == 0) return (PFN_vkVoidFunction)trace_destroy_device;
    return fn;
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
    pthread_once(&init_once, init_loader);
    PFN_vkVoidFunction fn = real_gipa(instance, name);
    if (!fn) return fn;
    if (strcmp(name, "vkGetInstanceProcAddr") == 0) return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
    if (strcmp(name, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)trace_gdpa;
    if (strcmp(name, "vkGetPhysicalDeviceMemoryProperties2") == 0) {
        real_memory2 = (PFN_vkGetPhysicalDeviceMemoryProperties2)fn;
        return (PFN_vkVoidFunction)trace_memory2;
    }
    if (strcmp(name, "vkGetPhysicalDeviceMemoryProperties2KHR") == 0) {
        real_memory2_khr = (PFN_vkGetPhysicalDeviceMemoryProperties2KHR)fn;
        return (PFN_vkVoidFunction)trace_memory2_khr;
    }
    if (strcmp(name, "vkCreateDevice") == 0) {
        real_create_device = (PFN_vkCreateDevice)fn;
        return (PFN_vkVoidFunction)trace_create_device;
    }
    if (strcmp(name, "vkEnumerateDeviceExtensionProperties") == 0) {
        real_extensions = (PFN_vkEnumerateDeviceExtensionProperties)fn;
        return (PFN_vkVoidFunction)trace_extensions;
    }
    return fn;
}
