/* Included by profiling.c. Sampling is explicit or occurs at captured operation
 * boundaries; there is no background thread, compute-device acquisition or
 * device synchronization. NVML is optional and reports hardware-wide values. */
#include "resource_sampling.h"
#if !defined(__wasm__) && !defined(_WIN32)
#include <sys/resource.h>
#include <unistd.h>
#endif
#if defined(_WIN32)
#include <windows.h>
#endif
#if defined(__linux__) && !defined(__wasm__)
#include <dlfcn.h>
#endif

static void vx_sample_cpu(VxProcessCpuSample* sample) {
    sample->status = VX_OBSERVATION_UNSUPPORTED;
    sample->source = "unavailable";
#if !defined(__wasm__) && !defined(_WIN32)
    struct rusage usage;
    sample->source = "getrusage(RUSAGE_SELF)";
    sample->status = VX_OBSERVATION_FAILED;
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_utime.tv_sec < 0 ||
        usage.ru_stime.tv_sec < 0 || usage.ru_utime.tv_usec < 0 ||
        usage.ru_stime.tv_usec < 0) return;
    uint64_t seconds = (uint64_t)usage.ru_utime.tv_sec + (uint64_t)usage.ru_stime.tv_sec;
    uint64_t fraction = ((uint64_t)usage.ru_utime.tv_usec + (uint64_t)usage.ru_stime.tv_usec) * 1000u;
    if (seconds > (UINT64_MAX - fraction) / UINT64_C(1000000000)) return;
    sample->process_time_ns = seconds * UINT64_C(1000000000) + fraction;
    long processors = sysconf(_SC_NPROCESSORS_ONLN);
    if (processors > 0 && (uint64_t)processors <= UINT32_MAX)
        sample->online_processors = (uint32_t)processors;
    sample->status = VX_OBSERVATION_AVAILABLE;
#elif defined(_WIN32)
    FILETIME creation, exit_time, kernel, user;
    sample->source = "GetProcessTimes";
    sample->status = VX_OBSERVATION_FAILED;
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit_time, &kernel, &user)) return;
    uint64_t ticks = ((uint64_t)kernel.dwHighDateTime << 32) | kernel.dwLowDateTime;
    uint64_t user_ticks = ((uint64_t)user.dwHighDateTime << 32) | user.dwLowDateTime;
    if (ticks > UINT64_MAX - user_ticks || ticks + user_ticks > UINT64_MAX / 100u) return;
    sample->process_time_ns = (ticks + user_ticks) * 100u;
    SYSTEM_INFO info; GetSystemInfo(&info);
    sample->online_processors = info.dwNumberOfProcessors;
    sample->status = VX_OBSERVATION_AVAILABLE;
#endif
}

#if defined(__linux__) && !defined(__wasm__)
/* Stable NVML C ABI subset; no CUDA SDK or new link-time dependency. */
typedef struct { unsigned int gpu, memory; } VxNvmlUtilization;
typedef struct { unsigned long long total, free, used; } VxNvmlMemory;
typedef struct {
    void* library;
    int ready;
    VxObservationStatus status;
    int (*shutdown)(void);
    int (*count)(unsigned int*);
    int (*device)(unsigned int, void**);
    int (*uuid)(void*, char*, unsigned int);
    int (*name)(void*, char*, unsigned int);
    int (*utilization)(void*, VxNvmlUtilization*);
    int (*memory)(void*, VxNvmlMemory*);
} VxNvml;
static VxNvml vx_nvml;
static pthread_mutex_t vx_nvml_mutex = PTHREAD_MUTEX_INITIALIZER;

static int vx_nvml_initialize(void) {
    if (vx_nvml.ready) return vx_nvml.ready > 0;
    vx_nvml.ready = -1;
    vx_nvml.status = VX_OBSERVATION_UNSUPPORTED;
    void* library = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) return 0;
    int (*initialize)(void) = (int (*)(void))dlsym(library, "nvmlInit_v2");
    vx_nvml.shutdown = (int (*)(void))dlsym(library, "nvmlShutdown");
    vx_nvml.count = (int (*)(unsigned int*))dlsym(library, "nvmlDeviceGetCount_v2");
    vx_nvml.device = (int (*)(unsigned int, void**))dlsym(library, "nvmlDeviceGetHandleByIndex_v2");
    vx_nvml.uuid = (int (*)(void*, char*, unsigned int))dlsym(library, "nvmlDeviceGetUUID");
    vx_nvml.name = (int (*)(void*, char*, unsigned int))dlsym(library, "nvmlDeviceGetName");
    vx_nvml.utilization = (int (*)(void*, VxNvmlUtilization*))dlsym(library, "nvmlDeviceGetUtilizationRates");
    vx_nvml.memory = (int (*)(void*, VxNvmlMemory*))dlsym(library, "nvmlDeviceGetMemoryInfo");
    if (!initialize || !vx_nvml.shutdown || !vx_nvml.count || !vx_nvml.device || !vx_nvml.uuid ||
        !vx_nvml.name || !vx_nvml.utilization || !vx_nvml.memory) {
        dlclose(library);
        return 0;
    }
    int result = initialize();
    if (result != 0) {
        vx_nvml.status = result == 3 ? VX_OBSERVATION_UNSUPPORTED : VX_OBSERVATION_FAILED;
        dlclose(library); return 0;
    }
    vx_nvml.status = VX_OBSERVATION_AVAILABLE;
    /* One module-lifetime read-only sampler. No CUDA context is created. */
    vx_nvml.library = library;
    vx_nvml.ready = 1;
    return 1;
}

__attribute__((destructor)) static void vx_nvml_close(void) {
    if (vx_nvml.ready > 0) vx_nvml.shutdown();
    if (vx_nvml.library) dlclose(vx_nvml.library);
}
static void vx_sample_gpus(VxResourceSample* sample) {
    sample->gpu_source = "nvml";
    sample->gpu_status = VX_OBSERVATION_UNSUPPORTED;
    pthread_mutex_lock(&vx_nvml_mutex);
    if (!vx_nvml_initialize()) { sample->gpu_status = vx_nvml.status; goto done; }
    unsigned int count = 0;
    sample->gpu_status = VX_OBSERVATION_FAILED;
    if (vx_nvml.count(&count) != 0) goto done;
    sample->gpu_status = VX_OBSERVATION_AVAILABLE;
    sample->gpu_truncated = count > VX_RESOURCE_MAX_GPUS;
    if (count > VX_RESOURCE_MAX_GPUS) count = VX_RESOURCE_MAX_GPUS;
    for (unsigned int i = 0; i < count; i++) {
        void* device = NULL;
        if (vx_nvml.device(i, &device) != 0) { sample->gpu_truncated = 1; continue; }
        VxGpuResourceSample* gpu = &sample->gpus[sample->gpu_count++];
        if (vx_nvml.uuid(device, gpu->id, sizeof(gpu->id)) != 0)
            snprintf(gpu->id, sizeof(gpu->id), "nvml-index:%u", i);
        if (vx_nvml.name(device, gpu->name, sizeof(gpu->name)) != 0)
            snprintf(gpu->name, sizeof(gpu->name), "NVIDIA device %u", i);
        gpu->id[sizeof(gpu->id) - 1] = 0;
        gpu->name[sizeof(gpu->name) - 1] = 0;
        VxNvmlUtilization utilization = {0};
        int result = vx_nvml.utilization(device, &utilization);
        gpu->utilization = result == 3 ? VX_OBSERVATION_UNSUPPORTED : VX_OBSERVATION_FAILED;
        if (!result && utilization.gpu <= 100 && utilization.memory <= 100) {
            gpu->utilization = VX_OBSERVATION_AVAILABLE;
            gpu->compute_active = (double)utilization.gpu / 100.0;
            gpu->memory_active = (double)utilization.memory / 100.0;
        }
        VxNvmlMemory memory = {0};
        result = vx_nvml.memory(device, &memory);
        gpu->memory = result == 3 ? VX_OBSERVATION_UNSUPPORTED : VX_OBSERVATION_FAILED;
        if (!result && memory.used <= memory.total) {
            gpu->memory = VX_OBSERVATION_AVAILABLE;
            gpu->capacity_bytes = memory.total;
            gpu->used_bytes = memory.used;
        }
    }
done:
    pthread_mutex_unlock(&vx_nvml_mutex);
}
#else
static void vx_sample_gpus(VxResourceSample* sample) {
    sample->gpu_source = "unavailable";
    sample->gpu_status = VX_OBSERVATION_UNSUPPORTED;
}
#endif

void vx_resource_sample(VxResourceSample* sample, int cpu, int gpu) {
    memset(sample, 0, sizeof(*sample));
    sample->start_ns = vx_trace_now_ns();
    sample->process = (VxProcessMemorySampleV1)VX_PROCESS_MEMORY_SAMPLE_V1_INIT;
    (void)vx_process_memory_sample_v1(&sample->process);
    sample->wasm.status = VX_OBSERVATION_UNSUPPORTED;
#if defined(__wasm__)
    vx_wasm_memory_sample(&sample->wasm);
#endif
    sample->cpu.status = VX_OBSERVATION_NOT_COLLECTED;
    sample->cpu.source = "";
    if (cpu) vx_sample_cpu(&sample->cpu);
    sample->gpu_status = VX_OBSERVATION_NOT_COLLECTED;
    sample->gpu_source = "";
    if (gpu) vx_sample_gpus(sample);
    sample->end_ns = vx_trace_now_ns();
}
