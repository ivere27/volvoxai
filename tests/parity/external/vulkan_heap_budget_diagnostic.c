/* Test-only Vulkan probe: model two unreclaimed host-visible device-local
 * pools, query the driver's unmodified budget, then release the first pool.
 * No Deno, wgpu, VolvoxAI, shader or budget override participates here. */
#include "vulkan_core.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        fprintf(stderr, "%s: %d\n", operation, result);
        exit(1);
    }
}

static void snapshot(VkPhysicalDevice physical, const char* phase) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
    };
    VkPhysicalDeviceMemoryProperties2 properties = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
        .pNext = &budget,
    };
    vkGetPhysicalDeviceMemoryProperties2(physical, &properties);
    for (uint32_t i = 0; i < properties.memoryProperties.memoryHeapCount; i++) {
        printf("{\"phase\":\"%s\",\"heap\":%u,\"size\":%" PRIu64
               ",\"budget\":%" PRIu64 ",\"usage\":%" PRIu64
               ",\"would_reject_48_bytes_at_97_percent\":%s}\n",
               phase, i, properties.memoryProperties.memoryHeaps[i].size,
               budget.heapBudget[i], budget.heapUsage[i],
               budget.heapUsage[i] + 48 >= budget.heapBudget[i] / 100 * 97 ? "true" : "false");
    }
}

static VkDeviceMemory allocate(VkDevice device, uint32_t type, VkDeviceSize size) {
    VkMemoryAllocateInfo info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = size,
        .memoryTypeIndex = type,
    };
    VkDeviceMemory memory;
    VkResult result = vkAllocateMemory(device, &info, NULL, &memory);
    printf("{\"event\":\"allocate\",\"memory_type\":%u,\"size\":%" PRIu64 ",\"result\":%d}\n",
           type, size, result);
    check(result, "vkAllocateMemory");
    void* mapped;
    check(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");
    memset(mapped, 0x35, 48);
    vkUnmapMemory(device, memory);
    return memory;
}

int main(void) {
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "heap-budget-diagnostic",
        .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VkInstance instance;
    check(vkCreateInstance(&instance_info, NULL, &instance), "vkCreateInstance");
    uint32_t physical_count = 0;
    check(vkEnumeratePhysicalDevices(instance, &physical_count, NULL), "vkEnumeratePhysicalDevices");
    if (!physical_count) return 1;
    VkPhysicalDevice* physicals = calloc(physical_count, sizeof(*physicals));
    if (!physicals) return 1;
    check(vkEnumeratePhysicalDevices(instance, &physical_count, physicals), "vkEnumeratePhysicalDevices");
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < physical_count; i++) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicals[i], &properties);
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical = physicals[i];
            printf("{\"device\":\"%s\"}\n", properties.deviceName);
            break;
        }
    }
    free(physicals);
    if (!physical) return 1;
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    uint32_t type = UINT32_MAX;
    const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
        if ((memory.memoryTypes[i].propertyFlags & flags) == flags) { type = i; break; }
    }
    if (type == UINT32_MAX) return 1;
    printf("{\"memory_type\":%u,\"heap\":%u,\"flags\":%u}\n",
           type, memory.memoryTypes[type].heapIndex, memory.memoryTypes[type].propertyFlags);
    uint32_t family_count = 0, family = UINT32_MAX;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, NULL);
    VkQueueFamilyProperties* families = calloc(family_count, sizeof(*families));
    if (!families) return 1;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families);
    for (uint32_t i = 0; i < family_count; i++) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = i; break; }
    }
    free(families);
    if (family == UINT32_MAX) return 1;
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority,
    };
    const char* extensions[] = {VK_EXT_MEMORY_BUDGET_EXTENSION_NAME};
    VkDeviceCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue,
        .enabledExtensionCount = 1, .ppEnabledExtensionNames = extensions,
    };
    VkDevice devices[2];
    VkDeviceMemory pools[2];
    snapshot(physical, "baseline");
    for (uint32_t i = 0; i < 2; i++) {
        check(vkCreateDevice(physical, &info, NULL, &devices[i]), "vkCreateDevice");
        pools[i] = allocate(devices[i], type, 64 * 1024 * 1024);
        snapshot(physical, i ? "two_pools" : "one_pool");
    }
    VkDeviceMemory tiny = allocate(devices[1], type, 48);
    snapshot(physical, "tiny_allocation_succeeded");
    vkFreeMemory(devices[1], tiny, NULL);
    vkFreeMemory(devices[0], pools[0], NULL);
    vkDestroyDevice(devices[0], NULL);
    snapshot(physical, "first_pool_released");
    vkFreeMemory(devices[1], pools[1], NULL);
    vkDestroyDevice(devices[1], NULL);
    snapshot(physical, "both_pools_released");
    vkDestroyInstance(instance, NULL);
    return 0;
}
