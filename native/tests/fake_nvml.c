/* Deterministic driver-telemetry fixture; never linked into engine artifacts. */
#include <stdio.h>
#include <stdint.h>
typedef struct { unsigned int gpu, memory; } Utilization;
typedef struct { unsigned long long total, free, used; } Memory;
int nvmlInit_v2(void) { return 0; }
int nvmlShutdown(void) { return 0; }
int nvmlDeviceGetCount_v2(unsigned int* count) { *count = 3; return 0; }
int nvmlDeviceGetHandleByIndex_v2(unsigned int index, void** device) {
    if (index == 2) return 6; /* A device disappeared during enumeration. */
    *device = (void*)(uintptr_t)(index + 1); return 0;
}
int nvmlDeviceGetUUID(void* device, char* text, unsigned int size) {
    snprintf(text, size, "test-gpu-%u", (unsigned)(uintptr_t)device); return 0;
}
int nvmlDeviceGetName(void* device, char* text, unsigned int size) {
    (void)device; snprintf(text, size, "test GPU"); return 0;
}
int nvmlDeviceGetUtilizationRates(void* device, Utilization* value) {
    if ((uintptr_t)device == 2) return 3; /* Unsupported is not zero utilization. */
    value->gpu = 0; value->memory = 25; return 0;
}
int nvmlDeviceGetMemoryInfo(void* device, Memory* value) {
    if ((uintptr_t)device == 2) return 999; /* A failed read is not unsupported. */
    value->total = value->free = 1024; value->used = 0; return 0;
}
