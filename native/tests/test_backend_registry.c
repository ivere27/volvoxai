#include "backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return -1; \
    } \
} while (0)

typedef struct {
    char tag;
    int init_result;
    int supports_result;
    int run_result;
    int end_result;
    int sync_result;
    int write_output;
    float output_value;
} MockBackend;

static MockBackend g_a;
static MockBackend g_b;
static MockBackend g_cpu;
static char g_trace[256];
static size_t g_trace_length;

static void trace(MockBackend* backend, char event) {
    if (g_trace_length + 2u >= sizeof(g_trace)) abort();
    g_trace[g_trace_length++] = backend->tag;
    g_trace[g_trace_length++] = event;
    g_trace[g_trace_length] = 0;
}

static void reset_mock(MockBackend* backend, char tag) {
    memset(backend, 0, sizeof(*backend));
    backend->tag = tag;
    backend->init_result = VX_BACKEND_INIT_READY;
    backend->supports_result = VX_BACKEND_DECLINED;
    backend->run_result = VX_BACKEND_DECLINED;
    backend->sync_result = 1;
}

static void reset_all(void) {
    reset_mock(&g_a, 'A');
    reset_mock(&g_b, 'B');
    reset_mock(&g_cpu, 'C');
    g_trace_length = 0;
    g_trace[0] = 0;
}

static int mock_init(MockBackend* backend) {
    trace(backend, 'i');
    return backend->init_result;
}

static int mock_supports(MockBackend* backend, const Node* node) {
    (void)node;
    trace(backend, 's');
    return backend->supports_result;
}

static int mock_run(MockBackend* backend, const Node* node, T* input, T* output) {
    (void)node;
    (void)input;
    trace(backend, 'r');
    if (backend->write_output && output && output->data) output->data[0] = backend->output_value;
    return backend->run_result;
}

static void* mock_alloc(MockBackend* backend, size_t bytes) {
    trace(backend, 'a');
    return malloc(bytes ? bytes : 1u);
}

static void mock_upload(MockBackend* backend, void* destination, const void* source, size_t bytes) {
    trace(backend, 'u');
    if (bytes) memcpy(destination, source, bytes);
}

static void mock_download(MockBackend* backend, void* destination, const void* source, size_t bytes) {
    trace(backend, 'd');
    if (bytes) memcpy(destination, source, bytes);
}

static void mock_reset(MockBackend* backend) { trace(backend, 'z'); }
static void mock_begin(MockBackend* backend) { trace(backend, 'b'); }

static int mock_end(MockBackend* backend) {
    trace(backend, 'e');
    return backend->end_result;
}

static void mock_mark(MockBackend* backend, const void* host, size_t bytes, int is_weight) {
    (void)host;
    (void)bytes;
    (void)is_weight;
    trace(backend, 'm');
}

static int mock_sync(MockBackend* backend, const void* host, size_t bytes, int is_weight) {
    (void)host;
    (void)bytes;
    (void)is_weight;
    trace(backend, 'y');
    return backend->sync_result;
}

static void mock_teardown(MockBackend* backend) { trace(backend, 't'); }

#define DEFINE_MOCK_CALLBACKS(prefix, state) \
    static int prefix##_init(void* user_data) { (void)user_data; return mock_init(&(state)); } \
    static int prefix##_supports(void* user_data, const Node* node) { \
        (void)user_data; return mock_supports(&(state), node); \
    } \
    static int prefix##_run(void* user_data, const Node* node, T* input, T* output) { \
        (void)user_data; \
        return mock_run(&(state), node, input, output); \
    } \
    static void* prefix##_alloc(void* user_data, size_t bytes) { \
        (void)user_data; return mock_alloc(&(state), bytes); \
    } \
    static void prefix##_upload(void* user_data, void* destination, const void* source, size_t bytes) { \
        (void)user_data; \
        mock_upload(&(state), destination, source, bytes); \
    } \
    static void prefix##_download(void* user_data, void* destination, const void* source, size_t bytes) { \
        (void)user_data; \
        mock_download(&(state), destination, source, bytes); \
    } \
    static void prefix##_reset(void* user_data) { (void)user_data; mock_reset(&(state)); } \
    static void prefix##_begin(void* user_data) { (void)user_data; mock_begin(&(state)); } \
    static int prefix##_end(void* user_data) { (void)user_data; return mock_end(&(state)); } \
    static void prefix##_mark(void* user_data, const void* host, size_t bytes, int is_weight) { \
        (void)user_data; \
        mock_mark(&(state), host, bytes, is_weight); \
    } \
    static int prefix##_sync(void* user_data, const void* host, size_t bytes, int is_weight) { \
        (void)user_data; \
        return mock_sync(&(state), host, bytes, is_weight); \
    } \
    static void prefix##_teardown(void* user_data) { (void)user_data; mock_teardown(&(state)); }

DEFINE_MOCK_CALLBACKS(a, g_a)
DEFINE_MOCK_CALLBACKS(b, g_b)
DEFINE_MOCK_CALLBACKS(cpu, g_cpu)

#define MOCK_BACKEND(label, prefix) { \
    .name = (label), .init = prefix##_init, .supports = prefix##_supports, \
    .run = prefix##_run, .alloc = prefix##_alloc, .upload = prefix##_upload, \
    .download = prefix##_download, .reset = prefix##_reset, \
    .begin_forward = prefix##_begin, .end_forward = prefix##_end, \
    .mark_host = prefix##_mark, .sync_host = prefix##_sync, \
    .teardown = prefix##_teardown, \
}

static const VxBackend k_backend_a = MOCK_BACKEND("a", a);
static const VxBackend k_backend_b = MOCK_BACKEND("b", b);
static const VxBackend k_backend_cpu = MOCK_BACKEND("cpu", cpu);

static void clear_trace(void) {
    g_trace_length = 0;
    g_trace[0] = 0;
}

static int check_trace(const char* expected) {
    if (strcmp(g_trace, expected)) {
        fprintf(stderr, "trace mismatch: expected %s, got %s\n", expected, g_trace);
        return -1;
    }
    return 0;
}

static int initialize_registry(VxBackendRegistry* registry) {
    const VxBackend* const entries[] = { &k_backend_a, &k_backend_b, &k_backend_cpu };
    return vx_backend_registry_init(registry, entries, 3u, &k_backend_cpu);
}

static int test_ordered_selection(void) {
    VxBackendRegistry registry = {0};
    Node node = {0};
    T input = {0};
    T output = {0};
    const VxBackend* selected = NULL;
    float value = -99.0f;
    output.data = &value;
    reset_all();
    g_a.supports_result = VX_BACKEND_HANDLED;
    g_a.run_result = VX_BACKEND_HANDLED;
    g_a.write_output = 1;
    g_a.output_value = 11.0f;
    g_b.supports_result = VX_BACKEND_HANDLED;
    g_b.run_result = VX_BACKEND_HANDLED;
    g_cpu.supports_result = VX_BACKEND_HANDLED;
    g_cpu.run_result = VX_BACKEND_HANDLED;
    CHECK(initialize_registry(&registry) == 0);
    CHECK(check_trace("AiBiCi") == 0);
    clear_trace();
    CHECK(vx_backend_registry_dispatch(&registry, &node, &input, &output, &selected) ==
          VX_BACKEND_HANDLED);
    CHECK(selected == &k_backend_a && value == 11.0f);
    CHECK(check_trace("AsAr") == 0);
    clear_trace();
    vx_backend_registry_teardown(&registry);
    CHECK(check_trace("CtBtAt") == 0);
    return 0;
}

static int test_declines_and_cpu_fallback(void) {
    VxBackendRegistry registry = {0};
    Node node = {0};
    T output = {0};
    const VxBackend* selected = NULL;
    float value = -99.0f;
    output.data = &value;
    reset_all();
    g_a.supports_result = VX_BACKEND_DECLINED;
    g_b.supports_result = VX_BACKEND_HANDLED;
    g_b.run_result = VX_BACKEND_HANDLED;
    g_b.write_output = 1;
    g_b.output_value = 22.0f;
    g_cpu.supports_result = VX_BACKEND_HANDLED;
    g_cpu.run_result = VX_BACKEND_HANDLED;
    g_cpu.write_output = 1;
    g_cpu.output_value = 33.0f;
    CHECK(initialize_registry(&registry) == 0);
    clear_trace();
    CHECK(vx_backend_registry_dispatch(&registry, &node, NULL, &output, &selected) ==
          VX_BACKEND_HANDLED);
    CHECK(selected == &k_backend_b && value == 22.0f);
    CHECK(check_trace("AsBsBr") == 0);

    g_b.supports_result = VX_BACKEND_DECLINED;
    value = -99.0f;
    selected = NULL;
    clear_trace();
    CHECK(vx_backend_registry_dispatch(&registry, &node, NULL, &output, &selected) ==
          VX_BACKEND_HANDLED);
    CHECK(selected == &k_backend_cpu && value == 33.0f);
    CHECK(check_trace("AsBsCsCr") == 0);
    vx_backend_registry_teardown(&registry);
    return 0;
}

static int test_run_decline_and_errors(void) {
    VxBackendRegistry registry = {0};
    Node node = {0};
    T output = {0};
    const VxBackend* selected = NULL;
    float value = -77.0f;
    output.data = &value;

    reset_all();
    g_a.supports_result = VX_BACKEND_HANDLED;
    g_a.run_result = VX_BACKEND_DECLINED;
    g_b.supports_result = VX_BACKEND_HANDLED;
    g_b.run_result = VX_BACKEND_HANDLED;
    g_b.write_output = 1;
    g_b.output_value = 44.0f;
    g_cpu.supports_result = VX_BACKEND_HANDLED;
    g_cpu.run_result = VX_BACKEND_HANDLED;
    CHECK(initialize_registry(&registry) == 0);
    clear_trace();
    CHECK(vx_backend_registry_dispatch(&registry, &node, NULL, &output, &selected) ==
          VX_BACKEND_HANDLED);
    CHECK(selected == &k_backend_b && value == 44.0f);
    CHECK(check_trace("AsArBsBr") == 0);
    vx_backend_registry_teardown(&registry);

    reset_all();
    value = -77.0f;
    output.data = &value;
    g_a.supports_result = VX_BACKEND_ERROR;
    g_b.supports_result = VX_BACKEND_HANDLED;
    g_b.run_result = VX_BACKEND_HANDLED;
    g_cpu.supports_result = VX_BACKEND_HANDLED;
    g_cpu.run_result = VX_BACKEND_HANDLED;
    CHECK(initialize_registry(&registry) == 0);
    clear_trace();
    CHECK(vx_backend_registry_dispatch(&registry, &node, NULL, &output, &selected) ==
          VX_BACKEND_ERROR);
    CHECK(selected == NULL && value == -77.0f);
    CHECK(check_trace("As") == 0);
    vx_backend_registry_teardown(&registry);

    reset_all();
    value = -77.0f;
    output.data = &value;
    g_a.supports_result = VX_BACKEND_HANDLED;
    g_a.run_result = VX_BACKEND_ERROR;
    g_b.supports_result = VX_BACKEND_HANDLED;
    g_b.run_result = VX_BACKEND_HANDLED;
    g_cpu.supports_result = VX_BACKEND_HANDLED;
    g_cpu.run_result = VX_BACKEND_HANDLED;
    CHECK(initialize_registry(&registry) == 0);
    clear_trace();
    CHECK(vx_backend_registry_dispatch(&registry, &node, NULL, &output, &selected) ==
          VX_BACKEND_ERROR);
    CHECK(selected == NULL && value == -77.0f);
    CHECK(check_trace("AsAr") == 0);
    vx_backend_registry_teardown(&registry);

    reset_all();
    value = -77.0f;
    output.data = &value;
    g_a.supports_result = VX_BACKEND_DECLINED;
    g_b.supports_result = VX_BACKEND_DECLINED;
    g_cpu.supports_result = VX_BACKEND_DECLINED;
    CHECK(initialize_registry(&registry) == 0);
    clear_trace();
    CHECK(vx_backend_registry_dispatch(&registry, &node, NULL, &output, &selected) ==
          VX_BACKEND_ERROR);
    CHECK(selected == NULL && value == -77.0f);
    CHECK(check_trace("AsBsCs") == 0);
    vx_backend_registry_teardown(&registry);
    return 0;
}

static int test_validation(void) {
    const VxBackend* const valid[] = { &k_backend_a, &k_backend_b, &k_backend_cpu };
    const VxBackend* const wrong_cpu_order[] = { &k_backend_cpu, &k_backend_a };
    const VxBackend* const duplicate[] = { &k_backend_a, &k_backend_a, &k_backend_cpu };
    VxBackend missing_run = k_backend_a;
    VxBackend unpaired_transfer = k_backend_a;
    VxBackend unpaired_lifecycle = k_backend_a;
    VxBackend unpaired_coherence = k_backend_a;
    VxBackend resident_without_coherence = k_backend_a;
    const VxBackend* invalid_list[3] = { &missing_run, &k_backend_b, &k_backend_cpu };
    CHECK(vx_backend_registry_validate(valid, 3u, &k_backend_cpu) == 0);
    CHECK(vx_backend_registry_validate(wrong_cpu_order, 2u, &k_backend_cpu) != 0);
    CHECK(vx_backend_registry_validate(duplicate, 3u, &k_backend_cpu) != 0);
    missing_run.run = NULL;
    CHECK(vx_backend_registry_validate(invalid_list, 3u, &k_backend_cpu) != 0);
    unpaired_transfer.download = NULL;
    invalid_list[0] = &unpaired_transfer;
    CHECK(vx_backend_registry_validate(invalid_list, 3u, &k_backend_cpu) != 0);
    unpaired_lifecycle.end_forward = NULL;
    invalid_list[0] = &unpaired_lifecycle;
    CHECK(vx_backend_registry_validate(invalid_list, 3u, &k_backend_cpu) != 0);
    unpaired_coherence.sync_host = NULL;
    invalid_list[0] = &unpaired_coherence;
    CHECK(vx_backend_registry_validate(invalid_list, 3u, &k_backend_cpu) != 0);
    resident_without_coherence.mark_host = NULL;
    resident_without_coherence.sync_host = NULL;
    resident_without_coherence.outputs_device_resident = 1;
    invalid_list[0] = &resident_without_coherence;
    CHECK(vx_backend_registry_validate(invalid_list, 3u, &k_backend_cpu) != 0);
    return 0;
}

static int test_lifecycle_fanout(void) {
    VxBackendRegistry registry = {0};
    int source = 17;
    int copied = 0;
    void* device;
    reset_all();
    g_a.init_result = VX_BACKEND_INIT_UNAVAILABLE;
    CHECK(initialize_registry(&registry) == 0);
    CHECK(check_trace("AiAtBiCi") == 0);

    clear_trace();
    vx_backend_registry_begin_forward(&registry);
    CHECK(check_trace("BbCb") == 0);
    clear_trace();
    vx_backend_registry_mark_host(&registry, &source, sizeof(source), 0);
    CHECK(check_trace("BmCm") == 0);
    clear_trace();
    CHECK(vx_backend_registry_sync_host(&registry, &source, sizeof(source), 0) == 0);
    CHECK(check_trace("ByCy") == 0);
    clear_trace();
    CHECK(vx_backend_registry_end_forward(&registry) == 0);
    CHECK(check_trace("BeCe") == 0);
    clear_trace();
    vx_backend_registry_reset(&registry);
    CHECK(check_trace("BzCz") == 0);

    clear_trace();
    device = k_backend_b.alloc(k_backend_b.user_data, sizeof(source));
    CHECK(device != NULL);
    k_backend_b.upload(k_backend_b.user_data, device, &source, sizeof(source));
    k_backend_b.download(k_backend_b.user_data, &copied, device, sizeof(copied));
    free(device);
    CHECK(copied == source);
    CHECK(check_trace("BaBuBd") == 0);

    clear_trace();
    vx_backend_registry_teardown(&registry);
    CHECK(check_trace("CtBt") == 0);

    reset_all();
    g_b.end_result = -1;
    g_b.sync_result = 0;
    CHECK(initialize_registry(&registry) == 0);
    clear_trace();
    CHECK(vx_backend_registry_end_forward(&registry) != 0);
    CHECK(check_trace("AeBeCe") == 0);
    clear_trace();
    CHECK(vx_backend_registry_sync_host(&registry, &source, sizeof(source), 0) != 0);
    CHECK(check_trace("AyByCy") == 0);
    vx_backend_registry_teardown(&registry);
    return 0;
}

static int test_init_failure_tears_down_initialized_backends(void) {
    VxBackendRegistry registry = {0};
    reset_all();
    g_cpu.init_result = VX_BACKEND_INIT_UNAVAILABLE;
    CHECK(initialize_registry(&registry) != 0);
    CHECK(!registry.ready && registry.count == 0u);
    CHECK(check_trace("AiBiCiCtBtAt") == 0);
    return 0;
}

static int test_required_backend_may_not_be_unavailable(void) {
    VxBackendRegistry registry = {0};
    VxBackend required = k_backend_a;
    const VxBackend* entries[2] = { &required, &k_backend_cpu };
    reset_all();
    required.required = 1;
    g_a.init_result = VX_BACKEND_INIT_UNAVAILABLE;
    CHECK(vx_backend_registry_init(&registry, entries, 2u, &k_backend_cpu) != 0);
    CHECK(!registry.ready && registry.count == 0u);
    CHECK(check_trace("AiAt") == 0);
    return 0;
}

int main(void) {
    CHECK(test_ordered_selection() == 0);
    CHECK(test_declines_and_cpu_fallback() == 0);
    CHECK(test_run_decline_and_errors() == 0);
    CHECK(test_validation() == 0);
    CHECK(test_lifecycle_fanout() == 0);
    CHECK(test_init_failure_tears_down_initialized_backends() == 0);
    CHECK(test_required_backend_may_not_be_unavailable() == 0);
    puts("backend registry tests passed");
    return 0;
}
