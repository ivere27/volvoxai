/* Execute the production RoPE host adapter with deterministic Driver storage.
 * This checks metadata ownership/ABI and independent rotation values; actual
 * device kernels are qualified separately on CUDA hardware. */
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../src/backends/cuda/host/cuda_rope_cache.h"

typedef uintptr_t CUdeviceptr;
typedef struct { int device_dirty; } CudaTensorSlot;
typedef struct { CudaTensorSlot* slot; CUdeviceptr device; } CudaTensorView;
typedef struct {
    int ready;
    CUdeviceptr rope_powers_device;
    float rope_powers_host[CUDA_ROPE_POWER_CAPACITY];
    CudaRopePowerCache rope_power_cache;
} CudaContextState;
enum { CUDA_REPLAY_PASS_OBSERVE = 1, CUDA_REPLAY_PASS_CAPTURE = 2,
       CUDA_REPLAY_PASS_VALIDATE = 3 };
static CudaContextState* current;
static int cuda_forward_scope_active;
static struct { int pass, unsafe, failed; } cuda_replay;
static unsigned uploads, views, launches;
static int fail_upload;
static CUdeviceptr last_powers;
static CudaTensorSlot input_slot, output_slot;
static CudaContextState* cuda_context_state_get(void) { return current; }
static void cuda_replay_fail(void) { cuda_replay.failed = 1; }
static int cuda_ok(int result, const char* operation) {
    (void)operation; return result == 0;
}
static int cuda_copy_htod_current(CUdeviceptr device, const void* host, size_t bytes) {
    uploads++;
    assert(device >= current->rope_powers_device);
    assert(device - current->rope_powers_device + bytes <= CUDA_ROPE_POWER_STORAGE_BYTES);
    if (fail_upload) return -1;
    memcpy((void*)device, host, bytes); return 0;
}
static int checked_product4_bytes(uint32_t a, uint32_t b, uint32_t c,
        uint32_t d, size_t size, uint32_t* elements, size_t* bytes) {
    uint64_t count = (uint64_t)a * b;
    if (count > UINT32_MAX || c > UINT32_MAX / count) return 0;
    count *= c;
    if (d > UINT32_MAX / count || count * d > SIZE_MAX / size) return 0;
    *elements = (uint32_t)(count * d); *bytes = *elements * size; return 1;
}
static int ranges_overlap(const void* a, size_t an, const void* b, size_t bn) {
    return (uintptr_t)a < (uintptr_t)b + bn && (uintptr_t)b < (uintptr_t)a + an;
}
static int graph_ensure_input_view(const void* host, size_t bytes,
        int weight, CudaTensorView* view) {
    (void)bytes; assert(!weight); views++;
    *view = (CudaTensorView){&input_slot, (CUdeviceptr)host}; return 1;
}
static int graph_output_view(void* host, size_t bytes, CudaTensorView* view) {
    (void)bytes; views++;
    *view = (CudaTensorView){&output_slot, (CUdeviceptr)host}; return 1;
}
static void graph_mark_device(CudaTensorSlot* slot) { slot->device_dirty = 1; }
#define CUDA_FORWARD_FUNCTION(handle) 1
static int cuda_launch_1d(int function, uint32_t elements, void** args) {
    assert(function == 1); launches++;
    const float* input = (const float*)*(CUdeviceptr*)args[0];
    const int32_t* positions = (const int32_t*)*(CUdeviceptr*)args[1];
    float* output = (float*)*(CUdeviceptr*)args[2];
    assert(elements == *(uint32_t*)args[3]);
    uint32_t sequence = *(uint32_t*)args[4], width = *(uint32_t*)args[5];
    uint32_t head_width = width / *(uint32_t*)args[6];
    uint32_t rotary = *(uint32_t*)args[7], offset = *(uint32_t*)args[9];
    uint32_t interleaved = *(uint32_t*)args[10], mode = *(uint32_t*)args[11];
    last_powers = *(CUdeviceptr*)args[8];
    const float* powers = (const float*)last_powers;
    for (uint32_t i = 0; i < elements; i++) {
        uint32_t dimension = i % head_width, row = i / width;
        if (dimension >= rotary) { output[i] = input[i]; continue; }
        uint32_t half = rotary / 2, pair = interleaved ? dimension / 2 : dimension % half;
        uint32_t left = interleaved ? pair * 2 : pair, right = interleaved ? left + 1 : left + half;
        int32_t position = mode == 1 ? positions[row % sequence] :
            mode == 2 ? positions[row] : (int32_t)(offset + row % sequence);
        float angle = (float)position / powers[pair];
        float sine = sinf(angle), cosine = cosf(angle);
        output[i] = dimension == left ? input[i - dimension + left] * cosine -
            input[i - dimension + right] * sine :
            input[i - dimension + left] * sine + input[i - dimension + right] * cosine;
    }
    return 1;
}
#include "../src/backends/cuda/host/cuda_graph_rope_host.inc"

static void init(CudaContextState* state) {
    memset(state, 0, sizeof(*state)); state->ready = 1;
    state->rope_powers_device = (CUdeviceptr)calloc(1, CUDA_ROPE_POWER_STORAGE_BYTES);
    assert(state->rope_powers_device); current = state;
}
static void check_rotation(const float* input, const float* output, int batch,
        int sequence, int width, int heads, int rotary, float theta,
        const int32_t* positions, int offset, int interleaved, int mode) {
    int head_width = width / heads;
    for (int row = 0; row < batch * sequence; row++)
        for (int head = 0; head < heads; head++) {
            const float* x = input + row * width + head * head_width;
            const float* y = output + row * width + head * head_width;
            int position = mode == 1 ? positions[row % sequence] :
                mode == 2 ? positions[row] : offset + row % sequence;
            for (int pair = 0; pair < rotary / 2; pair++) {
                int a = interleaved ? pair * 2 : pair, b = interleaved ? a + 1 : a + rotary / 2;
                double angle = position / pow((double)theta, 2.0 * pair / rotary);
                double c = cos(angle), s = sin(angle);
                assert(fabs(y[a] - (x[a] * c - x[b] * s)) < 2e-5);
                assert(fabs(y[b] - (x[a] * s + x[b] * c)) < 2e-5);
            }
            for (int d = rotary; d < head_width; d++) assert(y[d] == x[d]);
        }
}
static void run(float theta, int rotary, int batch, int sequence, int width,
        int heads, int32_t* positions, int offset, int interleaved, int mode) {
    float input[64], output[64]; assert(batch * sequence * width <= 64);
    for (int i = 0; i < batch * sequence * width; i++) input[i] = (float)(i % 11 - 5) / 8.f;
    long positions_count = mode == 1 ? sequence : mode == 2 ? batch * sequence : 0;
    assert(cuda_graph_rope_heads_f32(input, positions, positions_count, output,
        batch, sequence, width, heads, rotary, theta, offset, interleaved, mode));
    check_rotation(input, output, batch, sequence, width, heads, rotary, theta,
        positions, offset, interleaved, mode);
}

int main(void) {
    _Static_assert(CUDA_ROPE_POWER_STORAGE_BYTES == 2304u, "fixed admission reserve");
    CudaContextState a, b;
    int32_t positions[] = {2, 5, 9, 3, 7, 11};
    init(&a);
    run(10000.f, 4, 2, 3, 8, 1, positions, 0, 0, 2);
    assert(uploads == 1 && a.rope_power_cache.count == 1);
    CUdeviceptr first = last_powers;
    float first_table[CUDA_ROPE_POWER_CAPACITY];
    memcpy(first_table, (void*)first, sizeof(first_table));
    /* Different host pointers, shapes, positions and pairing preserve powers. */
    float rebased_input[8] = {1.f, -.5f, .25f, 2.f, -1.f, .75f, .5f, -2.f};
    float rebased_output[8];
    assert(cuda_graph_rope_heads_f32(rebased_input, NULL, 0, rebased_output,
        1, 1, 8, 1, 4, 10000.f, 3, 0, 0));
    check_rotation(rebased_input, rebased_output, 1, 1, 8, 1, 4, 10000.f,
        NULL, 3, 0, 0);
    assert(uploads == 1 && first == last_powers);
    positions[0] = 13;
    run(10000.f, 4, 1, 3, 16, 2, positions, 0, 1, 1);
    run(10000.f, 4, 1, 1, 8, 1, NULL, 17, 0, 0);
    assert(uploads == 1 && first == last_powers);
    run(10000.f, 8, 1, 2, 8, 1, NULL, 1, 0, 0);
    assert(uploads == 2 && last_powers != first);
    run(nextafterf(10000.f, INFINITY), 4, 1, 1, 8, 1, NULL, 1, 0, 0);
    assert(uploads == 3 && a.rope_power_cache.count == 3);

    cuda_forward_scope_active = 1;
    cuda_replay.pass = CUDA_REPLAY_PASS_CAPTURE;
    run(10000.f, 4, 1, 1, 8, 1, NULL, 1, 0, 0);
    cuda_replay.pass = CUDA_REPLAY_PASS_VALIDATE;
    run(10000.f, 4, 1, 1, 8, 1, NULL, 1, 0, 0);
    cuda_replay.pass = CUDA_REPLAY_PASS_CAPTURE;
    assert(uploads == 3 && !cuda_replay.failed && !cuda_replay.unsafe);
    float input[8] = {0}, output[8] = {0};
    unsigned prior_views = views, prior_launches = launches;
    assert(!cuda_graph_rope_heads_f32(input, NULL, 0, output, 1, 1, 8, 1,
        4, 12345.f, 0, 0, 0));
    assert(cuda_replay.failed && uploads == 3 && views == prior_views && launches == prior_launches);
    cuda_replay.failed = 0; cuda_replay.pass = CUDA_REPLAY_PASS_VALIDATE;
    assert(!cuda_graph_rope_heads_f32(input, NULL, 0, output, 1, 1, 8, 1,
        4, 12345.f, 0, 0, 0));
    assert(cuda_replay.failed && uploads == 3 && a.rope_power_cache.count == 3);
    cuda_forward_scope_active = 0; cuda_replay.failed = 0;

    fail_upload = 1;
    assert(!cuda_graph_rope_heads_f32(input, NULL, 0, output, 1, 1, 8, 1,
        4, 12345.f, 0, 0, 0));
    assert(a.rope_power_cache.count == 3 && views == prior_views && launches == prior_launches);
    fail_upload = 0;
    run(12345.f, 4, 1, 1, 8, 1, NULL, 1, 0, 0);
    assert(a.rope_power_cache.count == 4);
    while (a.rope_power_cache.count < CUDA_ROPE_CACHE_CAPACITY)
        run(20000.f + (float)a.rope_power_cache.count, 4, 1, 1, 8, 1, NULL, 1, 0, 0);
    unsigned filled_uploads = uploads;
    cuda_forward_scope_active = 1; cuda_replay.pass = CUDA_REPLAY_PASS_OBSERVE;
    run(30000.f, 4, 1, 1, 8, 1, NULL, 1, 0, 0);
    assert(cuda_replay.unsafe && a.rope_power_cache.count == CUDA_ROPE_CACHE_CAPACITY);
    run(40000.f, 4, 1, 1, 8, 1, NULL, 1, 0, 0);
    assert(uploads == filled_uploads + 2 && memcmp(first_table, (void*)first, sizeof(first_table)) == 0);
    cuda_replay.unsafe = 0; cuda_replay.pass = CUDA_REPLAY_PASS_CAPTURE;
    run(10000.f, 4, 1, 1, 8, 1, NULL, 1, 0, 0);
    assert(!cuda_replay.unsafe && uploads == filled_uploads + 2);
    assert(!cuda_graph_rope_heads_f32(input, NULL, 0, output, 1, 1, 8, 1,
        4, 40000.f, 0, 0, 0));
    assert(cuda_replay.failed && uploads == filled_uploads + 2);

    /* Invalid positions/alias/shape do not publish or transfer metadata. */
    cuda_forward_scope_active = 0; positions[0] = -1;
    assert(!cuda_graph_rope_heads_f32(input, positions, 1, output, 1, 1, 8, 1,
        4, 50000.f, 0, 0, 1));
    assert(!cuda_graph_rope_heads_f32(input, NULL, 0, input, 1, 1, 8, 1,
        4, 50000.f, 0, 0, 0));
    assert(!cuda_graph_rope_heads_f32(input, NULL, 0, output, 1, 1, 8, 1,
        3, 50000.f, 0, 0, 0));
    assert(uploads == filled_uploads + 2);

    init(&b);
    run(10000.f, 4, 1, 1, 8, 1, NULL, 1, 0, 0);
    assert(b.rope_power_cache.count == 1 && last_powers != first);
    current = &a;
    run(10000.f, 4, 1, 1, 8, 1, NULL, 1, 0, 0);
    assert(last_powers == first && uploads == filled_uploads + 3);
    free((void*)a.rope_powers_device); free((void*)b.rope_powers_device);
    init(&a);
    run(10000.f, 4, 1, 1, 8, 1, NULL, 1, 0, 0);
    assert(a.rope_power_cache.count == 1 && uploads == filled_uploads + 4);
    free((void*)a.rope_powers_device);
    return 0;
}
