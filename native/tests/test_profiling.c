#define _POSIX_C_SOURCE 200809L
#include "profiling.h"
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int export_trace(VxTrace* trace, char** json, size_t* size) {
    VxTraceView view; vx_trace_view(trace, &view);
    return vx_trace_export(&view, 0, 1024, json, size);
}

static atomic_uint_fast64_t clock_us = 100;
static unsigned samples;
static pthread_mutex_t gate_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gate_condition = PTHREAD_COND_INITIALIZER;
static unsigned admitted;
static int proceed;
static int device_ready, device_failed;
static unsigned device_released;
static int device_poll(uint32_t ticket, VxDeviceTraceResult* result) {
    assert(ticket == 42);
    if (!device_ready) return 1;
    result->elapsed_ns = 123456;
    return device_failed ? -1 : 0;
}
static void device_release(uint32_t ticket) { assert(ticket == 42); device_released++; }
static int await_ready;
static unsigned await_released;
static uint64_t await_start;
static int await_poll(uint32_t ticket, VxDeviceTraceResult* result) {
    assert(ticket == 43);
    if (!await_ready) return 1;
    result->host_start_ns = await_start; result->elapsed_ns = 7000;
    return 0;
}
static void await_release(uint32_t ticket) { assert(ticket == 43); await_released++; }
static void test_clock_and_activity(void) {
    VxTraceClock c = vx_trace_clock_bounds(1000, 2000, 400, 100, 200);
    assert(c.method == VX_TRACE_CLOCK_METHOD_BOUNDED && c.earliest_ns == 1100 && c.latest_ns == 1700);
    assert(!vx_trace_clock_bounds(2000, 1000, 400, 100, 200).method);
    assert(!vx_trace_clock_bounds(1000, 2000, 1100, 0, 100).method);
    assert(!vx_trace_clock_bounds(1000, 2000, 400, 300, 200).method);
    c = vx_trace_clock_bounds(UINT64_MAX - 1000, UINT64_MAX, 1000, 500, 500);
    assert(c.earliest_ns == UINT64_MAX - 500 && c.latest_ns == c.earliest_ns);
    c = vx_trace_clock_calibrated(1200, 1000, 64, 10, 1500000, 20, 1000000, 2000000, 100);
    assert(c.method == VX_TRACE_CLOCK_METHOD_CALIBRATED && c.earliest_ns == 1500970 && c.latest_ns == 1503030);
    /* A counter wrap is safe only within an unambiguous observation window. */
    c = vx_trace_clock_calibrated(10, 65530, 16, 1, 1005000, 0, 1000000, 1010000, 100);
    assert(c.method == VX_TRACE_CLOCK_METHOD_CALIBRATED && c.earliest_ns == 1004015 && c.latest_ns == 1006017);
    assert(!vx_trace_clock_calibrated(10, 250, 8, 1, 1005000, 0, 1000000, 1010000, 100).method);
    assert(!vx_trace_clock_calibrated(10, 20, 64, 1, 1005000, 0, 0, 2000000000, 100).method);
    assert(!vx_trace_clock_calibrated(10, 20, 64, 0, 1005000, 0, 1000000, 1010000, 100).method);
    assert(!vx_trace_clock_calibrated(10, 20, 64, 1, UINT64_MAX, UINT64_MAX, 1000000, 1010000, 100).method);
    atomic_uint_fast64_t a = 0, b = 0;
    uint64_t first = vx_trace_object_id(&a);
    assert(first && vx_trace_object_id(&a) == first && vx_trace_object_id(&b) != first);

    VxTrace* trace = vx_trace_create(16384, VX_TRACE_DETAIL_NODES, 1, 0);
    VxTraceScope scope = {0}; VxTraceIdentity identity = {.execution_id = 91};
    assert(vx_trace_scope_begin(trace, &scope, &identity, "Execute", "webgpu"));
    uint64_t now = vx_trace_now_ns(); await_start = now;
    VxDeviceTraceSpan span = {.host_start_ns = now, .name = "Buffer copy", .index = -1,
        .queue = {1, 2, 3}, .activity = VX_TRACE_ACTIVITY_COPY,
        .copy_source = VX_MEMORY_SPACE_HOST, .copy_destination = VX_MEMORY_SPACE_DEVICE, .copy_bytes = 128,
        .clock = {now + 100, now + 200, VX_TRACE_CLOCK_METHOD_CALIBRATED}};
    vx_trace_host_activity(&scope, now, now + 900, &span);
    vx_trace_device_span(&scope, &span, 300);
    span.name = "Compute"; span.activity = VX_TRACE_ACTIVITY_WORK;
    span.clock = (VxTraceClock){now, now + 1000, VX_TRACE_CLOCK_METHOD_BOUNDED};
    vx_trace_device_span(&scope, &span, 400);
    span.name = "Completion"; span.activity = VX_TRACE_ACTIVITY_AWAIT; span.clock = (VxTraceClock){0};
    vx_trace_defer_host_activity(&scope, &span, 43, await_poll, await_release);
    vx_trace_scope_end(&scope); vx_trace_stop(trace);
    VxTraceView view; vx_trace_view(trace, &view);
    assert(view.state == VX_TRACE_STATE_DRAINING && view.active == 1 && view.pending == 0);
    await_ready = 1; vx_trace_view(trace, &view);
    assert(view.state == VX_TRACE_STATE_READY && view.count == 5 && await_released == 1);
    assert(view.devices[0].copy_intervals == 1 && view.devices[0].pass_intervals == 1);
    assert(view.devices[0].host_copy_calls == 1 && view.devices[0].host_awaits == 1);
    assert(view.devices[0].calibrated_intervals == 1 && view.devices[0].bounded_intervals == 1);
    assert(view.records[1].copy_bytes == 128 && view.records[1].queue.submission_id == 3);
    assert(view.records[3].duration_ns == 7000 && view.records[3].activity == VX_TRACE_ACTIVITY_AWAIT);
    char* json; size_t length;
    assert(export_trace(trace, &json, &length) == 1);
    assert(strstr(json, "\"cat\":\"device.copy\",\"ph\":\"X\""));
    assert(strstr(json, "\"cat\":\"device.interval\",\"ph\":\"i\""));
    assert(strstr(json, "\"cat\":\"host.await\",\"ph\":\"b\",\"pid\":3"));
    assert(strstr(json, "\"cat\":\"host.await\",\"ph\":\"e\",\"pid\":3"));
    assert(!strstr(json, "\"cat\":\"host.await\",\"ph\":\"X\""));
    assert(strstr(json, "\"clockMethod\":\"calibrated\"") && strstr(json, "\"clockMethod\":\"bounded\""));
    free(json);
    vx_trace_release(trace);
    /* Completed host tickets drain at a subsequent capture boundary, so an
     * application need not poll GetTrace to collect a long training loop. */
    trace = vx_trace_create(65536, VX_TRACE_DETAIL_NODES, 0, 0);
    assert(vx_trace_scope_begin(trace, &scope, &identity, "First", "webgpu"));
    await_ready = 0;
    vx_trace_defer_host_activity(&scope, &span, 43, await_poll, await_release);
    vx_trace_scope_end(&scope);
    unsigned released = await_released;
    await_ready = 1;
    assert(vx_trace_scope_begin(trace, &scope, &identity, "Next", "webgpu"));
    assert(await_released == released + 1);
    vx_trace_scope_end(&scope); vx_trace_stop(trace); vx_trace_view(trace, &view);
    assert(view.state == VX_TRACE_STATE_READY && view.count == 3 && !view.dropped);
    vx_trace_release(trace);
    trace = vx_trace_create(4096, VX_TRACE_DETAIL_NODES, 0, 0);
    assert(vx_trace_scope_begin(trace, &scope, &identity, "Execute", "webgpu"));
    await_ready = 0;
    vx_trace_defer_host_activity(&scope, &span, 43, await_poll, await_release);
    vx_trace_abandon(trace); vx_trace_scope_end(&scope); vx_trace_view(trace, &view);
    assert(view.state == VX_TRACE_STATE_READY && view.pending == 0 && await_released == released + 2 && view.dropped == 1);
    vx_trace_release(trace);
}
static void* record_thread(void* pointer) {
    VxTrace* trace = pointer;
    VxTraceIdentity identity = {.runtime_id = 17};
    VxTraceScope scope = {0};
    assert(vx_trace_scope_begin(trace, &scope, &identity, "Concurrent", "cpu"));
    pthread_mutex_lock(&gate_mutex);
    admitted++;
    pthread_cond_broadcast(&gate_condition);
    while (!proceed) pthread_cond_wait(&gate_condition, &gate_mutex);
    pthread_mutex_unlock(&gate_mutex);
    for (int i = 0; i < 8; i++) vx_trace_node(&scope, vx_trace_now_ns(), i, "Add", "sum", 0);
    vx_trace_scope_end(&scope);
    return NULL;
}
static void* memory_thread(void* pointer) {
    VxTrace* trace = pointer;
    VxTraceScope scope = {0}; VxTraceIdentity identity = {0}; VxMemoryObserver observer = {0};
    assert(vx_trace_scope_begin(trace, &scope, &identity, "Allocation owner", "cpu"));
    assert(vx_memory_observer_attach(&observer, &scope));
    for (int i = 0; i < 256; i++) {
        vx_memory_record(&observer, VX_MEMORY_HOST_ARENA, 1, 16, VX_TRACE_MEMORY_ACTION_ALLOCATE);
        vx_memory_record(&observer, VX_MEMORY_HOST_ARENA, 1, 0, VX_TRACE_MEMORY_ACTION_FREE);
    }
    vx_memory_observer_clear(&observer);
    vx_trace_scope_end(&scope);
    return NULL;
}
uint64_t vx_runtime_monotonic_time_micros(void) {
    return atomic_fetch_add_explicit(&clock_us, 1, memory_order_relaxed);
}
int vx_process_memory_sample_v1(VxProcessMemorySampleV1* sample) {
    samples++;
    sample->available_mask = VX_PROCESS_MEMORY_AVAILABLE_CURRENT_RSS | VX_PROCESS_MEMORY_AVAILABLE_PEAK_RSS;
    sample->rss_bytes = 1024; sample->peak_rss_bytes = 2048;
    return 1;
}
int main(void) {
    test_clock_and_activity();
    /* One detail level controls both observation domains. Device opt-in does
     * not enable host nodes at BASIC, and NODES alone never enables GPU work. */
    for (int detail = VX_TRACE_DETAIL_BASIC; detail <= VX_TRACE_DETAIL_NODES; detail++) {
        for (int device = 0; device <= 1; device++) {
            VxTrace* trace = vx_trace_create(4096, detail, device, 0);
            VxTraceScope scope = {0}; VxTraceIdentity identity = {0};
            assert(vx_trace_scope_begin(trace, &scope, &identity, "Execute", "webgpu"));
            assert(vx_trace_nodes(&scope) == (detail == VX_TRACE_DETAIL_NODES));
            assert(vx_trace_device_enabled(&scope) == device);
            assert(vx_trace_device_nodes(&scope) == (device && detail == VX_TRACE_DETAIL_NODES));
            vx_trace_device_status(&scope, 1, 1, 0, 0);
            vx_trace_node(&scope, vx_trace_now_ns(), 0, "Add", "out", 0);
            vx_trace_scope_end(&scope); vx_trace_stop(trace);
            VxTraceView view; vx_trace_view(trace, &view);
            assert(view.detail == detail && view.device_timing == device && !view.memory);
            assert(view.count == (detail == VX_TRACE_DETAIL_NODES ? 2u : 1u));
            assert(view.device_count == (unsigned)device);
            if (device) {
                assert(view.devices[0].support == VX_TRACE_SUPPORT_AVAILABLE);
                assert(view.devices[0].node_timing_available && !view.devices[0].node_intervals);
            }
            char* json; size_t size;
            assert(export_trace(trace, &json, &size) == 1 && size);
            assert(strstr(json, "\"format\":\"volvoxai-trace/v6\""));
            assert(strstr(json, device ? "\"deviceTiming\":true" : "\"deviceTiming\":false"));
            assert(strstr(json, detail == VX_TRACE_DETAIL_NODES ? "\"detail\":\"nodes\"" : "\"detail\":\"basic\""));
            free(json);
            vx_trace_release(trace);
        }
    }
    assert(samples == 0);
    /* Program origins are copied at admission, and owning a node does not
     * turn a program invocation into a second whole-node measurement. */
    {
        VxTrace* attributed = vx_trace_create(8192, VX_TRACE_DETAIL_NODES, 1, 0);
        VxTraceScope work = {0}; VxTraceIdentity identity = {0};
        assert(vx_trace_scope_begin(attributed, &work, &identity, "TrainStep", "webgpu"));
        char name[] = "matmulBackward", entry[] = "weight_main", output[] = "hidden";
        char tensor[] = "weight";
        work.work = (VxTraceWork){VX_TRACE_PHASE_BACKWARD, 0, 0, "Linear", output, tensor};
        VxDeviceTraceSpan span = vx_trace_program_span(&work, name, entry);
        vx_trace_program_status(&work, 1);
        device_ready = 0;
        vx_trace_defer_device_span(&work, &span, 42, device_poll, device_release);
        vx_trace_host_program(&work, vx_trace_now_ns(), name, entry);
        name[0] = entry[0] = output[0] = tensor[0] = 'X';
        vx_trace_scope_end(&work); vx_trace_stop(attributed);
        VxTraceView observed; vx_trace_view(attributed, &observed);
        assert(observed.state == VX_TRACE_STATE_DRAINING);
        device_ready = 1; vx_trace_view(attributed, &observed);
        assert(observed.count == 3 && observed.devices[0].program_intervals == 1);
        assert(!observed.devices[0].node_intervals && !observed.devices[0].pass_intervals);
        assert(observed.devices[0].program_timing_available);
        for (int i = 0; i < 2; i++) {
            const VxTraceRecord* event = &observed.records[i];
            assert(event->phase == VX_TRACE_PHASE_BACKWARD && event->schedule_index == 0);
            assert(!strcmp(event->program, "matmulBackward") && !strcmp(event->entry, "weight_main"));
            assert(!strcmp(event->output, "hidden") && !strcmp(event->tensor, "weight"));
        }
        char* json; size_t size;
        assert(export_trace(attributed, &json, &size) == 1);
        assert(strstr(json, "\"cat\":\"device.program\"") && strstr(json, "\"cat\":\"host.program\""));
        assert(strstr(json, "\"phase\":\"backward\"") && strstr(json, "\"programIntervals\":\"1\""));
        free(json);
        vx_trace_release(attributed);
        device_released = 0;
    }
    assert(!vx_trace_create(4096, -1, 0, 0));
    assert(!vx_trace_create(4096, 2, 0, 0));
    uint64_t delta = 0;
    assert(vx_trace_timestamp_duration(250, 10, 8, 2, 100, &delta) && delta == 32);
    assert(vx_trace_timestamp_duration(5, 5, 64, 1, 1, &delta) && delta == 0);
    assert(!vx_trace_timestamp_duration(250, 10, 8, 2, 512, &delta));
    assert(!vx_trace_timestamp_duration(0, 1, 0, 1, 1, &delta));
    assert(!vx_trace_timestamp_duration(0, 1, 64, 0, 1, &delta));
    assert(!vx_trace_timestamp_duration(0, UINT64_MAX, 64, 2, 1, &delta));
    /* A stopped asynchronous pass drains independently of its host scope;
     * failures are counted and removed before publishing immutable pages. */
    for (int failure = 0; failure < 2; failure++) {
        VxTrace* async = vx_trace_create(4096, 0, 1, 0);
        VxTraceScope accepted = {0}; VxTraceIdentity id = {.execution_id = 19};
        assert(vx_trace_scope_begin(async, &accepted, &id, "Execute", "webgpu"));
        device_ready = 0; device_failed = failure;
        vx_trace_defer_device(&accepted, vx_trace_now_ns(), "compute pass", 42, device_poll, device_release);
        vx_trace_scope_end(&accepted); vx_trace_stop(async);
        VxTraceView view; vx_trace_view(async, &view);
        assert(view.state == VX_TRACE_STATE_DRAINING && view.active == 1 && !view.records);
        assert(view.pending == 1 && view.device_count == 0);
        device_ready = 1;
        vx_trace_view(async, &view);
        assert(view.state == VX_TRACE_STATE_READY && !view.active);
        assert(view.count == (failure ? 1u : 2u) && view.dropped == (unsigned)failure);
        assert(view.records[0].sequence == 1);
        if (!failure) assert(view.devices[0].pass_intervals == 1 && view.records[0].duration_ns == 123456);
        vx_trace_release(async);
    }
    assert(device_released == 2);
    /* Destruction cancels outstanding tickets without polling or writing a
     * stale engine address; the device bridge owns asynchronous retirement. */
    VxTrace* cancelled = vx_trace_create(4096, 0, 1, 0);
    VxTraceScope pending = {0}; VxTraceIdentity id = {0};
    assert(vx_trace_scope_begin(cancelled, &pending, &id, "Execute", "webgpu"));
    vx_trace_defer_device(&pending, vx_trace_now_ns(), "compute pass", 42, device_poll, device_release);
    vx_trace_abandon(cancelled);
    assert(device_released == 3 && !vx_trace_device_capacity(&pending, 1));
    VxTraceView abandoned; vx_trace_view(cancelled, &abandoned);
    assert(abandoned.state == VX_TRACE_STATE_DRAINING && abandoned.active == 1);
    vx_trace_scope_end(&pending);
    vx_trace_view(cancelled, &abandoned);
    assert(abandoned.state == VX_TRACE_STATE_READY && !abandoned.active);
    vx_trace_release(cancelled);
    /* Support stays independent of collected records and failures. Host-only
     * selection never requests device work; shared detail includes both node kinds. */
    VxTrace* selected = vx_trace_create(131072, VX_TRACE_DETAIL_NODES, 1, 0);
    VxTraceScope selected_scope = {0};
    assert(vx_trace_scope_begin(selected, &selected_scope, &id, "Execute", "webgpu"));
    assert(vx_trace_nodes(&selected_scope) && vx_trace_device_nodes(&selected_scope));
    vx_trace_device_status(&selected_scope, 1, 1, 1, 0);
    VxTraceView coverage; vx_trace_view(selected, &coverage);
    assert(coverage.devices[0].support == VX_TRACE_SUPPORT_AVAILABLE && !coverage.count);
    vx_trace_node(&selected_scope, vx_trace_now_ns(), 0, "Add", "out", 0);
    device_ready = 0; device_failed = 0;
    for (int i = 0; i < 80; i++) vx_trace_defer_device_node(&selected_scope,
        vx_trace_now_ns(), i, "Add", "out", 0, 42, device_poll, device_release);
    vx_trace_view(selected, &coverage);
    assert(coverage.pending == 80 && coverage.count == 81 && !coverage.dropped);
    device_ready = 1;
    vx_trace_device_fail(&selected_scope);
    vx_trace_device_status(&selected_scope, 0, 0, 0, 0);
    vx_trace_scope_end(&selected_scope); vx_trace_stop(selected); vx_trace_view(selected, &coverage);
    assert(coverage.count == 82 && coverage.devices[0].node_intervals == 80);
    assert(coverage.devices[0].support == VX_TRACE_SUPPORT_MIXED);
    assert(coverage.devices[0].node_timing_available);
    assert(coverage.devices[0].unavailable_passes == 1 && coverage.devices[0].failed_intervals == 1);
    vx_trace_release(selected);
    selected = vx_trace_create(4096, VX_TRACE_DETAIL_NODES, 0, 0);
    assert(vx_trace_scope_begin(selected, &selected_scope, &id, "Execute", "webgpu"));
    assert(vx_trace_nodes(&selected_scope) && !vx_trace_device_enabled(&selected_scope));
    vx_trace_device_status(&selected_scope, 1, 1, 1, 0);
    vx_trace_scope_end(&selected_scope); vx_trace_stop(selected); vx_trace_view(selected, &coverage);
    assert(!coverage.device_count && coverage.count == 1);
    vx_trace_release(selected);
    assert(!vx_trace_create(1, 0, 1, 0));
    VxTrace* trace = vx_trace_create(8192, 1, 1, 1);
    assert(trace && samples == 1);
    VxTraceIdentity identity = { .runtime_id = UINT64_MAX, .execution_id = UINT64_MAX - 1 };
    VxTraceScope scope = {0};
    assert(vx_trace_scope_begin(trace, &scope, &identity, "Execute", "cpu"));
    assert(vx_trace_nodes(&scope));
    uint64_t start = vx_trace_now_ns();
    vx_trace_node(&scope, start, 3, "Add", "quote\"line\n한글", 0);
    vx_trace_device(&scope, start, 1250000, "device pass");
    vx_trace_stop(trace);
    VxTraceView view;
    vx_trace_view(trace, &view);
    assert(view.state == 1 && view.active == 1 && !view.records);
    VxTraceScope refused = {0};
    assert(!vx_trace_scope_begin(trace, &refused, &identity, "late", "cpu"));
    char* json = NULL;
    size_t size = 0;
    assert(!export_trace(trace, &json, &size));
    for (unsigned i = 0; i < 100; i++) vx_trace_node(&scope, start, 3, "Add", "output", 0);
    vx_trace_scope_end(&scope);
    vx_trace_view(trace, &view);
    assert(view.state == 2 && !view.active && view.records && view.dropped && samples == 2);
    assert(view.devices[0].pass_intervals == 1);
    assert(view.memory_end_ns[0] < view.records[0].start_ns);
    assert(view.memory_end_ns[1] > view.memory_end_ns[0]);
    assert(export_trace(trace, &json, &size) == 1 && size == strlen(json));
    assert(strstr(json, "18446744073709551615"));
    assert(strstr(json, "quote\\\"line\\u000a한글"));
    assert(strstr(json, "\"deviceDurationNs\":\"1250000\""));
    assert(strstr(json, "\"clockAligned\":false"));
    assert(strstr(json, "\"ph\":\"i\""));
    char* repeat = NULL; size_t again = 0;
    assert(export_trace(trace, &repeat, &again) == 1 && !strcmp(repeat, json) && again == size);
    free(json); free(repeat);
    vx_trace_stop(trace);
    assert(samples == 2);
    vx_trace_release(trace);
    /* A scope, rather than an engine/model, retains a released collector. */
    trace = vx_trace_create(4096, 0, 1, 0);
    assert(vx_trace_scope_begin(trace, &scope, &identity, "Execute", "cpu"));
    vx_trace_stop(trace); vx_trace_release(trace); vx_trace_scope_end(&scope);
    assert(!scope.trace && samples == 2);
    /* Stop closes admission while four host tracks continue. A reader lease
     * outlives public release; events become immutable only after every join. */
    trace = vx_trace_create(65536, 1, 1, 0);
    pthread_t threads[4];
    for (int i = 0; i < 4; i++) assert(!pthread_create(&threads[i], NULL, record_thread, trace));
    pthread_mutex_lock(&gate_mutex);
    while (admitted != 4) pthread_cond_wait(&gate_condition, &gate_mutex);
    pthread_mutex_unlock(&gate_mutex);
    vx_trace_retain(trace);
    vx_trace_stop(trace);
    vx_trace_release(trace);
    vx_trace_view(trace, &view);
    assert(view.state == 1 && view.active == 4);
    pthread_mutex_lock(&gate_mutex);
    proceed = 1;
    pthread_cond_broadcast(&gate_condition);
    pthread_mutex_unlock(&gate_mutex);
    for (int i = 0; i < 4; i++) assert(!pthread_join(threads[i], NULL));
    vx_trace_view(trace, &view);
    assert(view.state == 2 && view.count == 36 && !view.dropped);
    uint64_t tracks[4] = {0};
    int found = 0;
    for (size_t i = 0; i < view.count; i++) {
        const VxTraceRecord* record = &view.records[i];
        assert(record->sequence == i + 1 && record->identity.runtime_id == 17);
        if (record->kind == 0) {
            for (int j = 0; j < found; j++) assert(tracks[j] != record->track_id);
            tracks[found++] = record->track_id;
        }
    }
    assert(found == 4);
    vx_trace_release(trace);
    pthread_cond_destroy(&gate_condition);
    pthread_mutex_destroy(&gate_mutex);
    /* Initial inventory, overlapping growth, address reuse and actual frees.
     * Keeping an observer after release must retain no trace/event storage. */
    trace = vx_trace_create(65536, VX_TRACE_DETAIL_BASIC, 0, 1);
    VxTraceScope memory_scope = {0}; VxTraceIdentity memory_identity = {.context_id = 7};
    VxMemoryObserver observer = {0};
    assert(vx_trace_scope_begin(trace, &memory_scope, &memory_identity, "Execute", "cpu"));
    assert(vx_memory_observer_attach(&observer, &memory_scope));
    assert(!vx_memory_observer_attach(&observer, &memory_scope));
    vx_memory_record(&observer, VX_MEMORY_HOST_ARENA, 10, 100, VX_TRACE_MEMORY_ACTION_EXISTING);
    vx_memory_record(&observer, VX_MEMORY_HOST_ARENA, 10, 100, VX_TRACE_MEMORY_ACTION_EXISTING);
    vx_memory_record(&observer, VX_MEMORY_HOST_ARENA, 20, 200, VX_TRACE_MEMORY_ACTION_ALLOCATE);
    vx_memory_record(&observer, VX_MEMORY_HOST_ARENA, 10, 0, VX_TRACE_MEMORY_ACTION_FREE);
    vx_memory_record(&observer, VX_MEMORY_HOST_ARENA, 10, 50, VX_TRACE_MEMORY_ACTION_ALLOCATE);
    vx_memory_record(&observer, VX_MEMORY_HOST_ARENA, 20, 0, VX_TRACE_MEMORY_ACTION_FREE);
    vx_trace_scope_end(&memory_scope); vx_trace_stop(trace); vx_trace_view(trace, &view);
    VxAllocatorMemory* m = &view.allocators[VX_MEMORY_HOST_ARENA];
    assert(m->existing == 100 && m->live == 50 && m->peak == 300);
    assert(m->allocated == 250 && m->freed == 300 && m->complete && !m->dropped);
    assert(view.count == 6 && view.records[0].allocation_id != view.records[3].allocation_id);
    assert(export_trace(trace, &json, &size) == 1);
    assert(strstr(json, "\"peakBytes\":\"300\""));
    free(json);
    uint64_t stopped_clock = atomic_load(&clock_us);
    vx_memory_record(&observer, VX_MEMORY_HOST_ARENA, 10, 0, VX_TRACE_MEMORY_ACTION_FREE);
    assert(atomic_load(&clock_us) == stopped_clock);
    vx_trace_release(trace);
    vx_memory_record(&observer, VX_MEMORY_HOST_ARENA, 10, 0, VX_TRACE_MEMORY_ACTION_FREE);
    vx_memory_observer_clear(&observer);

    /* Record overflow must preserve accounting; live-identity overflow must
     * explicitly weaken accounting instead of inventing an exact peak. */
    for (int identities = 0; identities < 2; identities++) {
        trace = vx_trace_create(4096, 0, 0, 1);
        memory_scope = (VxTraceScope){0};
        assert(vx_trace_scope_begin(trace, &memory_scope, &memory_identity, "Execute", "cpu"));
        assert(vx_memory_observer_attach(&observer, &memory_scope));
        for (int i = 1; i < 1000; i++) {
            vx_memory_record(&observer, VX_MEMORY_HOST_ARENA, identities ? (uint64_t)i : 1, 10, VX_TRACE_MEMORY_ACTION_ALLOCATE);
            if (!identities) vx_memory_record(&observer, VX_MEMORY_HOST_ARENA, 1, 0, VX_TRACE_MEMORY_ACTION_FREE);
        }
        vx_trace_scope_end(&memory_scope); vx_trace_stop(trace); vx_trace_view(trace, &view);
        m = &view.allocators[VX_MEMORY_HOST_ARENA];
        assert(m->dropped && view.dropped && m->complete == !identities);
        if (!identities) assert(m->live == 0 && m->peak == 10 && m->allocated == 9990);
        vx_trace_release(trace); vx_memory_observer_clear(&observer);
    }
    /* Churn and arbitrary deletion order must preserve every still-live key,
     * including probe chains that wrap around the fixed identity table. */
    trace = vx_trace_create(4096, VX_TRACE_DETAIL_BASIC, 0, 1);
    memory_scope = (VxTraceScope){0};
    assert(vx_trace_scope_begin(trace, &memory_scope, &memory_identity, "Execute", "cpu"));
    assert(vx_memory_observer_attach(&observer, &memory_scope));
    for (uint64_t cycle = 0; cycle < 10000; cycle++) {
        for (uint64_t key = 1; key <= 8; key++)
            vx_memory_record(&observer, VX_MEMORY_HOST_ARENA, cycle * 8 + key, key, VX_TRACE_MEMORY_ACTION_ALLOCATE);
        for (uint64_t key = 2; key <= 8; key += 2)
            vx_memory_record(&observer, VX_MEMORY_HOST_ARENA, cycle * 8 + key, 0, VX_TRACE_MEMORY_ACTION_FREE);
        for (uint64_t key = 1; key <= 8; key += 2)
            vx_memory_record(&observer, VX_MEMORY_HOST_ARENA, cycle * 8 + key, 0, VX_TRACE_MEMORY_ACTION_FREE);
    }
    vx_trace_scope_end(&memory_scope); vx_trace_stop(trace); vx_trace_view(trace, &view);
    m = &view.allocators[VX_MEMORY_HOST_ARENA];
    assert(m->complete && m->live == 0 && m->peak == 36);
    assert(m->allocated == 360000 && m->freed == m->allocated);
    vx_memory_observer_clear(&observer); vx_trace_release(trace);

    /* A one-event export must not scan or allocate for the rest of the trace.
     * Only the requested record is addressable: ASan catches an eager scan. */
    VxTraceRecord single = {.name = "only page", .schedule_index = -1};
    view = (VxTraceView){.state = VX_TRACE_STATE_READY, .count = 1000000, .records = &single};
    assert(vx_trace_export(&view, 0, 1, &json, &size) == 1);
    assert(size < 4096 && strstr(json, "only page") && json[size - 1] != ']');
    free(json);

    /* Equal physical keys in independent owners never alias, even concurrently. */
    trace = vx_trace_create(4u * 1024u * 1024u, VX_TRACE_DETAIL_BASIC, 0, 1);
    for (int i = 0; i < 4; i++) assert(!pthread_create(&threads[i], NULL, memory_thread, trace));
    for (int i = 0; i < 4; i++) assert(!pthread_join(threads[i], NULL));
    vx_trace_stop(trace); vx_trace_view(trace, &view);
    m = &view.allocators[VX_MEMORY_HOST_ARENA];
    assert(m->allocated == 4 * 256 * 16 && m->freed == m->allocated && !m->live);
    assert(m->peak >= 16 && m->peak <= 64 && m->complete && !m->dropped);
    assert(view.count == 4 * (256 * 2 + 1));
    vx_trace_release(trace);
    return 0;
}
