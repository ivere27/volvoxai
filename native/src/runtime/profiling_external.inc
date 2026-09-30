/* Included by profiling.c. Vendor-tool annotations for traced work
 * (TraceOptions.external). Nothing here runs unless a trace requested it:
 * the ordinary execution route never calls these functions.
 *
 * NVTX ranges exist in native builds with a GPU backend. The header-only NVTX
 * library returns after one pointer check when no tool injected itself
 * (NVTX_INJECTION64_PATH). RenderDoc is never loaded by the engine; a capture
 * is requested only from a RenderDoc already present in the process. */
#if !defined(__wasm__) && (VOLVOXAI_ENABLE_CUDA || VOLVOXAI_ENABLE_VULKAN || VOLVOXAI_ENABLE_OPENGL)
#define VX_TRACE_NVTX 1
#include "../../third_party/nvtx3/nvToolsExt.h"
#else
#define VX_TRACE_NVTX 0
#endif
#if !defined(__wasm__) && (defined(__linux__) || defined(_WIN32)) && \
    (VOLVOXAI_ENABLE_VULKAN || VOLVOXAI_ENABLE_OPENGL)
#define VX_TRACE_RENDERDOC 1
#include "../../third_party/renderdoc/renderdoc_app.h"
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#else
#define VX_TRACE_RENDERDOC 0
#endif

static const char* const vx_trace_external_names[VX_TRACE_EXTERNAL_COUNT] = {
    "nvtx", "vk_debug_utils", "khr_debug", "renderdoc", "webgpu_debug_group"};
static const char* const vx_trace_external_backends[VX_TRACE_EXTERNAL_COUNT] = {
    "host", "vulkan", "opengl", "", "webgpu"};
const char* vx_trace_external_mechanism_name(VxTraceExternalMechanism mechanism) {
    return mechanism < VX_TRACE_EXTERNAL_COUNT ? vx_trace_external_names[mechanism] : "";
}
const char* vx_trace_external_mechanism_backend(VxTraceExternalMechanism mechanism) {
    return mechanism < VX_TRACE_EXTERNAL_COUNT ? vx_trace_external_backends[mechanism] : "";
}

int vx_trace_external(const VxTraceScope* scope) {
    return scope && scope->trace && scope->trace->external_annotations;
}

/* The first observation sets a mechanism's status; a failure always wins. */
static void vx_trace_external_record(VxTrace* trace, VxTraceExternalMechanism mechanism,
    VxObservationStatus status, uint64_t ranges) {
    int expected = VX_OBSERVATION_UNSPECIFIED;
    if (status == VX_OBSERVATION_FAILED)
        atomic_store_explicit(&trace->external_status[mechanism], status, memory_order_relaxed);
    else
        atomic_compare_exchange_strong_explicit(&trace->external_status[mechanism], &expected,
            status, memory_order_relaxed, memory_order_relaxed);
    if (ranges) atomic_fetch_add_explicit(&trace->external_ranges[mechanism], ranges, memory_order_relaxed);
}
void vx_trace_external_note(VxTraceScope* scope, VxTraceExternalMechanism mechanism,
    VxObservationStatus status, uint64_t ranges) {
    if (vx_trace_external(scope) && mechanism < VX_TRACE_EXTERNAL_COUNT)
        vx_trace_external_record(scope->trace, mechanism, status, ranges);
}

static const char* vx_trace_external_phase_name(VxTracePhase phase) {
    switch (phase) {
    case VX_TRACE_PHASE_FORWARD: return "forward";
    case VX_TRACE_PHASE_LOSS: return "loss";
    case VX_TRACE_PHASE_BACKWARD: return "backward";
    case VX_TRACE_PHASE_GRADIENT: return "gradient";
    case VX_TRACE_PHASE_OPTIMIZER: return "optimizer";
    default: return "";
    }
}
/* Forward nodes read "12 Linear -> h1", as in the trace's node events.
 * Training work outside the forward pass is prefixed by its phase. */
void vx_trace_external_label(char* label, size_t capacity, VxTracePhase phase, int index,
    const char* name, const char* output) {
    const char* prefix = phase == VX_TRACE_PHASE_FORWARD ? "" : vx_trace_external_phase_name(phase);
    char position[16] = "";
    if (index >= 0) snprintf(position, sizeof(position), "%d ", index);
    snprintf(label, capacity, "%s%s%s%s%s%s", prefix, prefix[0] ? " " : "", position,
        name ? name : "", output && output[0] ? " -> " : "", output ? output : "");
}

#if VX_TRACE_NVTX
static nvtxDomainHandle_t vx_trace_nvtx_domain_handle;
static nvtxStringHandle_t vx_trace_nvtx_trace_string;
static VxOnce vx_trace_nvtx_once = VX_ONCE_INITIALIZER;
static void vx_trace_nvtx_initialize(void) {
    vx_trace_nvtx_domain_handle = nvtxDomainCreateA("VolvoxAI");
    /* Categories are trace phases, so tools can filter backward work. */
    for (VxTracePhase phase = VX_TRACE_PHASE_FORWARD; phase <= VX_TRACE_PHASE_OPTIMIZER; phase++)
        nvtxDomainNameCategoryA(vx_trace_nvtx_domain_handle, (uint32_t)phase, vx_trace_external_phase_name(phase));
    /* A registered name can delimit an Nsight Systems capture range. */
    vx_trace_nvtx_trace_string = nvtxDomainRegisterStringA(vx_trace_nvtx_domain_handle, "VolvoxAI trace");
}
static nvtxDomainHandle_t vx_trace_nvtx_domain(void) {
    vx_once(&vx_trace_nvtx_once, vx_trace_nvtx_initialize);
    return vx_trace_nvtx_domain_handle;
}
static nvtxEventAttributes_t vx_trace_nvtx_attributes(const char* message, VxTracePhase phase,
    int has_payload, int64_t payload) {
    nvtxEventAttributes_t attributes;
    memset(&attributes, 0, sizeof(attributes));
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.category = (uint32_t)phase;
    attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attributes.message.ascii = message;
    if (has_payload) {
        attributes.payloadType = NVTX_PAYLOAD_TYPE_INT64;
        attributes.payload.llValue = payload;
    }
    return attributes;
}
#endif

void vx_trace_external_push(VxTraceScope* scope, VxTracePhase phase, int index,
    const char* name, const char* output) {
    if (!vx_trace_external(scope)) return;
#if VX_TRACE_NVTX
    char label[160];
    vx_trace_external_label(label, sizeof(label), phase, index, name, output);
    nvtxEventAttributes_t attributes = vx_trace_nvtx_attributes(label, phase, index >= 0, index);
    nvtxDomainRangePushEx(vx_trace_nvtx_domain(), &attributes);
    vx_trace_external_record(scope->trace, VX_TRACE_EXTERNAL_NVTX, VX_OBSERVATION_AVAILABLE, 1);
#else
    (void)phase; (void)index; (void)name; (void)output;
    vx_trace_external_record(scope->trace, VX_TRACE_EXTERNAL_NVTX, VX_OBSERVATION_UNSUPPORTED, 0);
#endif
}
void vx_trace_external_pop(VxTraceScope* scope) {
    if (!vx_trace_external(scope)) return;
#if VX_TRACE_NVTX
    nvtxDomainRangePop(vx_trace_nvtx_domain());
#endif
}

/* An operation may finish on another thread, so it uses a start/end range. */
static void vx_trace_external_operation_begin(VxTraceScope* scope) {
    if (!vx_trace_external(scope)) return;
#if VX_TRACE_NVTX
    nvtxEventAttributes_t attributes = vx_trace_nvtx_attributes(scope->name ? scope->name : "",
        VX_TRACE_PHASE_UNSPECIFIED, scope->identity.execution_id != 0, (int64_t)scope->identity.execution_id);
    scope->external_range = nvtxDomainRangeStartEx(vx_trace_nvtx_domain(), &attributes);
    vx_trace_external_record(scope->trace, VX_TRACE_EXTERNAL_NVTX, VX_OBSERVATION_AVAILABLE, 1);
#else
    vx_trace_external_record(scope->trace, VX_TRACE_EXTERNAL_NVTX, VX_OBSERVATION_UNSUPPORTED, 0);
#endif
}
static void vx_trace_external_operation_end(VxTraceScope* scope) {
    if (!vx_trace_external(scope)) return;
#if VX_TRACE_NVTX
    nvtxDomainRangeEnd(vx_trace_nvtx_domain(), scope->external_range);
#endif
}

#if VX_TRACE_RENDERDOC
static RENDERDOC_API_1_1_2* vx_trace_renderdoc(void) {
    pRENDERDOC_GetAPI get = NULL;
#if defined(_WIN32)
    HMODULE module = GetModuleHandleA("renderdoc.dll");
    if (module) get = (pRENDERDOC_GetAPI)(void*)GetProcAddress(module, "RENDERDOC_GetAPI");
#else
    void* module = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
    if (module) {
        get = (pRENDERDOC_GetAPI)dlsym(module, "RENDERDOC_GetAPI");
        dlclose(module); /* RTLD_NOLOAD only added a reference. */
    }
#endif
    RENDERDOC_API_1_1_2* api = NULL;
    if (!get || get(eRENDERDOC_API_Version_1_1_2, (void**)&api) != 1) return NULL;
    return api;
}
#endif

/* Called under the collector mutex at the first annotated operation. A
 * headless engine has no present call, so the capture is explicit. */
static void vx_trace_external_capture_begin(VxTrace* trace) {
    if (!trace->external_capture || trace->external_capture_started) return;
    trace->external_capture_started = 1;
#if VX_TRACE_RENDERDOC
    RENDERDOC_API_1_1_2* api = vx_trace_renderdoc();
    if (api) {
        api->StartFrameCapture(NULL, NULL);
        trace->external_capture_active = 1;
        vx_trace_external_record(trace, VX_TRACE_EXTERNAL_RENDERDOC, VX_OBSERVATION_AVAILABLE, 1);
        return;
    }
#endif
    vx_trace_external_record(trace, VX_TRACE_EXTERNAL_RENDERDOC, VX_OBSERVATION_UNSUPPORTED, 0);
}
static void vx_trace_external_capture_end(VxTrace* trace) {
    if (!trace->external_capture_active) return;
    trace->external_capture_active = 0;
#if VX_TRACE_RENDERDOC
    RENDERDOC_API_1_1_2* api = vx_trace_renderdoc();
    if (!api || api->EndFrameCapture(NULL, NULL) != 1)
        vx_trace_external_record(trace, VX_TRACE_EXTERNAL_RENDERDOC, VX_OBSERVATION_FAILED, 0);
#endif
}

void vx_trace_set_external(VxTrace* trace, int annotations, int capture) {
    if (!trace) return;
    trace->external_annotations = annotations != 0;
    trace->external_capture = capture != 0;
#if VX_TRACE_NVTX
    if (trace->external_annotations) {
        nvtxDomainHandle_t domain = vx_trace_nvtx_domain();
        nvtxEventAttributes_t attributes = vx_trace_nvtx_attributes(NULL, VX_TRACE_PHASE_UNSPECIFIED, 0, 0);
        attributes.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
        attributes.message.registered = vx_trace_nvtx_trace_string;
        trace->external_trace_range = nvtxDomainRangeStartEx(domain, &attributes);
        trace->external_trace_range_open = 1;
    }
#endif
}
/* At StopTrace: close the trace range and any capture it opened. */
static void vx_trace_external_stop(VxTrace* trace) {
    vx_trace_external_capture_end(trace);
#if VX_TRACE_NVTX
    if (trace->external_trace_range_open) {
        nvtxDomainRangeEnd(vx_trace_nvtx_domain(), trace->external_trace_range);
        trace->external_trace_range_open = 0;
    }
#endif
}
static void vx_trace_external_view(const VxTrace* trace, VxTraceView* view) {
    view->external_annotations = trace->external_annotations;
    view->external_capture = trace->external_capture;
    for (int i = 0; i < VX_TRACE_EXTERNAL_COUNT; i++) {
        view->external[i].status = atomic_load_explicit(&trace->external_status[i], memory_order_relaxed);
        view->external[i].ranges = atomic_load_explicit(&trace->external_ranges[i], memory_order_relaxed);
    }
}
