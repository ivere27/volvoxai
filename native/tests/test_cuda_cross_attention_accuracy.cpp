/* Run production CUDA block bodies using cooperative host lanes and compare
 * with an independent two-pass F64 softmax. The barrier scheduler checks all
 * active/tail lanes; these shims do not qualify PTX, GPU execution or timing. */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
#include <vector>
#include <array>
#include <cstdlib>
#include <functional>
#include <string>
#include <ucontext.h>

using vx_u32 = std::uint32_t;
using vx_i32 = std::int32_t;
using vx_u64 = std::uint64_t;
struct alignas(16) VxCudaFloat4 { float x, y, z, w; };
#define VX_CUDA_GLOBAL
#define VX_CUDA_DEVICE
#define VX_CUDA_FORCEINLINE inline
#define VX_CUDA_SHARED static
struct HostDimension { vx_u32 x = 0, y = 0, z = 1; };
static HostDimension threadIdx, blockIdx, blockDim{128}, gridDim{7};
static vx_u32 host_lane = 0;
[[maybe_unused]] static vx_u32 vx_global_x() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}
static constexpr vx_u32 lane_count = 128;
static constexpr std::size_t stack_bytes = 32768;
static std::array<ucontext_t, lane_count> lane_contexts;
static ucontext_t scheduler_context;
static std::array<std::array<unsigned char, stack_bytes>, lane_count> lane_stacks;
static std::array<float, lane_count> shuffle_values;
static std::function<void()> host_kernel;
enum class HostWait { Ready, Block, WarpRead, WarpDone, Finished };
static std::array<HostWait, lane_count> lane_wait;
static std::uint64_t block_barriers;
static void host_wait(HostWait kind) {
    lane_wait[host_lane] = kind;
    if (swapcontext(&lane_contexts[host_lane], &scheduler_context)) std::abort();
}
static void vx_sync_threads() { host_wait(HostWait::Block); }
static float vx_cuda_attention_shuffle_down(float value, vx_u32 delta) {
    shuffle_values[host_lane] = value;
    host_wait(HostWait::WarpRead);
    const vx_u32 lane = host_lane & 31u;
    const float result = shuffle_values[lane + delta < 32u
        ? host_lane + delta : host_lane];
    host_wait(HostWait::WarpDone);
    return result;
}
[[maybe_unused]] static float vx_cuda_attention_shuffle_lane(float value, vx_u32 source) {
    shuffle_values[host_lane] = value;
    host_wait(HostWait::WarpRead);
    const float result = shuffle_values[(host_lane & ~31u) + (source & 31u)];
    host_wait(HostWait::WarpDone);
    return result;
}
static void lane_entry() {
    host_kernel();
    lane_wait[host_lane] = HostWait::Finished;
}
static void launch_block(vx_u32 task, const std::function<void()>& kernel) {
    blockIdx.x = task % gridDim.x;
    blockIdx.y = task / gridDim.x;
    host_kernel = kernel;
    lane_wait.fill(HostWait::Ready);
    for (vx_u32 lane = 0; lane < lane_count; ++lane) {
        if (getcontext(&lane_contexts[lane])) std::abort();
        lane_contexts[lane].uc_stack.ss_sp = lane_stacks[lane].data();
        lane_contexts[lane].uc_stack.ss_size = stack_bytes;
        lane_contexts[lane].uc_link = &scheduler_context;
        makecontext(&lane_contexts[lane], lane_entry, 0);
    }
    for (;;) {
        for (host_lane = 0; host_lane < lane_count; ++host_lane) {
            if (lane_wait[host_lane] != HostWait::Ready) continue;
            threadIdx.x = host_lane;
            if (swapcontext(&scheduler_context, &lane_contexts[host_lane])) std::abort();
        }
        bool all_finished = true, all_block = true;
        for (auto kind : lane_wait) {
            all_finished &= kind == HostWait::Finished;
            all_block &= kind == HostWait::Block;
        }
        if (all_finished) break;
        if (all_block) {
            ++block_barriers;
            lane_wait.fill(HostWait::Ready);
            continue;
        }
        bool released = false;
        for (vx_u32 first = 0; first < lane_count; first += 32u)
            for (HostWait kind : {HostWait::WarpRead, HostWait::WarpDone}) {
                bool ready = true;
                for (vx_u32 lane = first; lane < first + 32u; ++lane)
                    ready &= lane_wait[lane] == kind;
                if (ready) {
                    for (vx_u32 lane = first; lane < first + 32u; ++lane)
                        lane_wait[lane] = HostWait::Ready;
                    released = true;
                }
            }
        if (!released) {
            std::fprintf(stderr, "Divergent block/warp rendezvous in block %u\n", task);
            std::abort();
        }
    }
}
static std::uint64_t output_hash(const std::vector<float>& output) {
    std::uint64_t hash = UINT64_C(1469598103934665603);
    for (float value : output) {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        hash ^= bits;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}
static float vx_max(float a, float b) { return std::max(a, b); }
static float vx_min(float a, float b) { return std::min(a, b); }
static float vx_sqrt(float x) { return std::sqrt(x); }
static float vx_exp(float x) {
    if (x < -87.f) return 0.f;
    return std::exp2(vx_min(88.f, x) * 1.4426950408889634f);
}

/* Include the real mask helper as well as the production MHA/GQA bodies.
 * The override permits a preserved old-source failure without editing code. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "../src/backends/cuda/attention_launch_plan.h"
#pragma GCC diagnostic pop
#include "../src/backends/cuda/kernels/cuda_forward_argument_storage.inc"
#include "../src/backends/cuda/kernels/cuda_forward_sdpa_kernels.inc"
#include "test_cuda_argument_record.h"
#ifndef TEST_CUDA_CROSS_ATTENTION_SOURCE
#define TEST_CUDA_CROSS_ATTENTION_SOURCE "../src/backends/cuda/kernels/cuda_forward_cross_attention_kernels.inc"
#endif
#include TEST_CUDA_CROSS_ATTENTION_SOURCE

struct Geometry {
    const char* name;
    vx_u32 batches, queries, keys, heads, kv_heads, width, mask_mode, causal;
    bool all_masked;
    bool cancellation = false;
    bool poison_masked_values = false;
};

static std::uint32_t random_state = 710913u;
static float sample() {
    random_state = random_state * 1664525u + 1013904223u;
    return (static_cast<float>(random_state >> 8) / 16777216.f - .5f) * .3f;
}

static bool reference_allowed(const Geometry& g, const std::vector<vx_i32>& mask,
        vx_u32 batch, vx_u32 row, vx_u32 key) {
    if (g.causal && key > row) return false;
    switch (g.mask_mode) {
        case 0: return true;
        case 1: return mask[key] != 0;
        case 2: return mask[batch * g.keys + key] != 0;
        case 3: return mask[row * g.keys + key] != 0;
        default: return mask[(batch * g.queries + row) * g.keys + key] != 0;
    }
}

static std::vector<double> reference(const Geometry& g,
        const std::vector<float>& q, const std::vector<float>& k,
        const std::vector<float>& v, const std::vector<vx_i32>& mask, float scale) {
    std::vector<double> result(q.size(), 0.), scores(g.keys);
    for (vx_u32 batch = 0; batch < g.batches; ++batch)
        for (vx_u32 row = 0; row < g.queries; ++row)
            for (vx_u32 head = 0; head < g.heads; ++head) {
                const vx_u32 kv_head = head / (g.heads / g.kv_heads);
                const std::size_t qb = ((batch * g.queries + row) * g.heads + head) * g.width;
                double maximum = -std::numeric_limits<double>::infinity();
                for (vx_u32 key = 0; key < g.keys; ++key) {
                    double dot = 0.;
                    const std::size_t kb = ((batch * g.keys + key) * g.kv_heads + kv_head) * g.width;
                    for (vx_u32 channel = 0; channel < g.width; ++channel)
                        dot += static_cast<double>(q[qb + channel]) * k[kb + channel];
                    scores[key] = reference_allowed(g, mask, batch, row, key)
                        ? dot * static_cast<double>(scale) : -std::numeric_limits<double>::infinity();
                    maximum = std::max(maximum, scores[key]);
                }
                if (!std::isfinite(maximum)) continue;
                double denominator = 0.;
                for (double& score : scores) { score = std::exp(score - maximum); denominator += score; }
                for (vx_u32 channel = 0; channel < g.width; ++channel) {
                    double numerator = 0.;
                    for (vx_u32 key = 0; key < g.keys; ++key) {
                        const std::size_t vb = ((batch * g.keys + key) * g.kv_heads + kv_head) * g.width;
                        if (reference_allowed(g, mask, batch, row, key))
                            numerator += scores[key] * static_cast<double>(v[vb + channel]);
                    }
                    result[qb + channel] = numerator / denominator;
                }
            }
    return result;
}

static void reverse_keys(const Geometry& g, std::vector<float>& k,
        std::vector<float>& v, std::vector<vx_i32>& mask) {
    const vx_u32 columns = g.kv_heads * g.width;
    for (vx_u32 batch = 0; batch < g.batches; ++batch)
        for (vx_u32 left = 0, right = g.keys - 1; left < right; ++left, --right)
            for (vx_u32 channel = 0; channel < columns; ++channel) {
                const std::size_t a = (batch * g.keys + left) * columns + channel;
                const std::size_t b = (batch * g.keys + right) * columns + channel;
                std::swap(k[a], k[b]); std::swap(v[a], v[b]);
            }
    for (std::size_t offset = 0; offset < mask.size(); offset += g.keys)
        std::reverse(mask.begin() + offset, mask.begin() + offset + g.keys);
}

static vx_u32 cross_grid_tiles(vx_u32 queries, vx_u32 keys, vx_u32 width) {
    const std::uint32_t maximum_grid[3] = {UINT32_MAX, 65535u, 65535u};
    const std::uint32_t maximum_block[3] = {1024u, 1024u, 64u};
    VxCudaAttentionLaunchPlan plan;
    if (!vx_cuda_cross_attention_launch_plan(1u, queries, keys, width, maximum_grid,
            maximum_block, 1024u, 49152u, &plan)) std::abort();
    return plan.grid[2];
}

static bool run(const Geometry& g) {
    const std::size_t q_count = static_cast<std::size_t>(g.batches) * g.queries * g.heads * g.width;
    const std::size_t kv_count = static_cast<std::size_t>(g.batches) * g.keys * g.kv_heads * g.width;
    const vx_u32 mask_rows = g.mask_mode == 2 ? g.batches : g.mask_mode == 3 ? g.queries :
        g.mask_mode == 4 ? g.batches * g.queries : 1;
    std::vector<float> q(q_count), k(kv_count), v(kv_count);
    std::vector<vx_i32> mask(static_cast<std::size_t>(mask_rows) * g.keys);
    for (float& value : q) value = sample();
    if (g.cancellation)
        for (std::size_t i = 0; i < q_count; ++i)
            q[i] = (i % g.width & 1u ? -1.f : 1.f) * (1.f + q[i]);
    for (vx_u32 batch = 0; batch < g.batches; ++batch)
        for (vx_u32 key = 0; key < g.keys; ++key)
            for (vx_u32 head = 0; head < g.kv_heads; ++head)
                for (vx_u32 channel = 0; channel < g.width; ++channel) {
                    const std::size_t i = ((batch * g.keys + key) * g.kv_heads + head) * g.width + channel;
                    k[i] = (g.cancellation ? 1.f : 0.f) + sample();
                    v[i] = (g.cancellation ? 0.f : 100.f) + .25f * batch + .125f * head + .01337f * (channel % 11)
                        + .001f * (static_cast<int>(key % 7) - 3);
                }
    for (vx_u32 row = 0; row < mask_rows; ++row)
        for (vx_u32 key = 0; key < g.keys; ++key)
            mask[row * g.keys + key] = !g.all_masked && (g.mask_mode <= 1 || row % g.queries != 1) && key % 7 != 0;
    if (g.poison_masked_values) {
        if (g.queries != 1) std::abort();
        for (vx_u32 batch = 0; batch < g.batches; ++batch)
            for (vx_u32 key = 0; key < g.keys; ++key)
                if (!reference_allowed(g, mask, batch, 0, key))
                    for (vx_u32 column = 0; column < g.kv_heads * g.width; ++column)
                        v[(batch * g.keys + key) * g.kv_heads * g.width + column] =
                            std::numeric_limits<float>::quiet_NaN();
    }
    const float scale = g.cancellation ? -.43f : .73f;
    std::vector<double> first_oracle;
    bool pass = true;
    for (int permutation = 0; permutation < 2; ++permutation) {
        auto oracle = reference(g, q, k, v, mask, scale);
        if (!permutation) first_oracle = oracle;
        if (permutation && !g.causal)
            for (std::size_t i = 0; i < oracle.size(); ++i)
                if (std::abs(oracle[i] - first_oracle[i]) > 1e-10) pass = false;
        constexpr std::size_t guard = 8;
        constexpr float sentinel = -98765.5f;
        std::vector<float> output(q_count + 2 * guard, sentinel);
        gridDim.z = cross_grid_tiles(g.queries, g.keys, g.width);
        for (blockIdx.z = 0u; blockIdx.z < gridDim.z; ++blockIdx.z)
        for (vx_u32 task = 0; task < g.batches * g.queries * g.heads + 5; ++task) {
            launch_block(task, [&] {
            if (g.heads == g.kv_heads) {
                auto record=cuda_argument_record(q.data(),k.data(),v.data(),
                    g.mask_mode?mask.data():nullptr,output.data()+guard,g.queries,g.keys,
                    g.heads*g.width,g.heads,g.width,g.batches,scale,g.causal,g.mask_mode);
                if(permutation) vx_cuda_cross_sdpa_f32(nullptr,nullptr,nullptr,nullptr,nullptr,
                    0,0,0,0,0,0,0,0,0,record.bytes.data());
                else vx_cuda_cross_sdpa_f32(q.data(), k.data(), v.data(), g.mask_mode ? mask.data() : nullptr,
                    output.data() + guard, g.queries, g.keys, g.heads * g.width, g.heads, g.width,
                    g.batches, scale, g.causal, g.mask_mode,nullptr);
            } else {
                auto record=cuda_argument_record(q.data(),k.data(),v.data(),
                    g.mask_mode?mask.data():nullptr,output.data()+guard,g.queries,g.keys,
                    g.heads*g.width,g.kv_heads*g.width,g.heads,g.kv_heads,g.width,
                    g.batches,scale,g.causal,g.mask_mode);
                if(permutation) vx_cuda_cross_sdpa_gqa_f32(nullptr,nullptr,nullptr,nullptr,nullptr,
                    0,0,0,0,0,0,0,0,0,0,0,record.bytes.data());
                else vx_cuda_cross_sdpa_gqa_f32(q.data(), k.data(), v.data(), g.mask_mode ? mask.data() : nullptr,
                    output.data() + guard, g.queries, g.keys, g.heads * g.width, g.kv_heads * g.width,
                    g.heads, g.kv_heads, g.width, g.batches, scale, g.causal, g.mask_mode,nullptr);
            }
            });
        }
        double maximum = 0.;
        std::size_t failures = 0;
        for (std::size_t i = 0; i < q_count; ++i) {
            const float actual = output[guard + i];
            const double error = std::abs(static_cast<double>(actual) - oracle[i]);
            maximum = std::max(maximum, error);
            if (!std::isfinite(actual) || error > 2e-6 + 1e-6 * std::abs(oracle[i]) ||
                    (oracle[i] == 0. && actual != 0.f)) {
                if (!failures) std::fprintf(stderr, "%s permutation=%d index=%zu actual=%.9g oracle=%.12g error=%.9g\n",
                    g.name, permutation, i, actual, oracle[i], error);
                ++failures;
            }
        }
        for (std::size_t i = 0; i < guard; ++i)
            if (output[i] != sentinel || output[guard + q_count + i] != sentinel) ++failures;
        std::printf("%s order=%s elements=%zu failures=%zu max_abs=%.12g %s\n", g.name,
            permutation ? "reversed" : "original", q_count, failures, maximum, failures ? "FAIL" : "PASS");
        std::printf("BITS %s order=%d hash=%016llx barriers=%llu\n", g.name, permutation,
            (unsigned long long)output_hash(output), (unsigned long long)block_barriers);
        pass = pass && failures == 0;
        reverse_keys(g, k, v, mask);
    }
    return pass;
}

static bool run_sdpa(const Geometry& g, vx_u32 query_start,
        vx_u32 query_count) {
    const vx_u32 model_width = g.heads * g.width;
    const std::size_t elements = static_cast<std::size_t>(g.batches) *
        g.queries * model_width;
    const vx_u32 mask_rows = g.mask_mode == 2 ? g.batches : g.mask_mode == 3
        ? g.queries : g.mask_mode == 4 ? g.batches * g.queries : 1;
    std::vector<float> q(elements), k(elements), v(elements), qkv(3 * elements);
    std::vector<vx_i32> mask(static_cast<std::size_t>(mask_rows) * g.keys);
    for (std::size_t i = 0; i < elements; ++i) {
        q[i] = sample(); k[i] = sample();
        v[i] = 1.f + sample() + .03125f * (i % 13);
    }
    for (vx_u32 row = 0; row < mask_rows; ++row)
        for (vx_u32 key = 0; key < g.keys; ++key)
            mask[row * g.keys + key] = !g.all_masked && key % 7 != 0 &&
                (g.mask_mode <= 1 || row % g.queries != 1);
    const float scale = -.43f;
    bool pass = true;
    for (int permutation = 0; permutation < 2; ++permutation) {
        auto oracle = reference(g, q, k, v, mask, scale);
        for (vx_u32 row = 0; row < g.batches * g.queries; ++row)
            for (vx_u32 channel = 0; channel < model_width; ++channel) {
                const std::size_t source = static_cast<std::size_t>(row) * model_width + channel;
                const std::size_t destination = static_cast<std::size_t>(row) * 3 * model_width + channel;
                qkv[destination] = q[source];
                qkv[destination + model_width] = k[source];
                qkv[destination + 2 * model_width] = v[source];
            }
        constexpr std::size_t guard = 8;
        constexpr float sentinel = -98765.5f;
        std::vector<float> output(elements + 2 * guard, sentinel);
        gridDim.z = 1u; blockIdx.z = 0u;
        const vx_u32 tasks = g.batches * query_count * g.heads;
        for (vx_u32 task = 0; task < tasks + 5; ++task)
            launch_block(task, [&] {
                vx_cuda_sdpa_f32(qkv.data(), g.mask_mode ? mask.data() : nullptr,
                    output.data() + guard, g.queries, model_width, g.heads,
                    g.width, g.batches, scale, g.causal, g.mask_mode,
                    query_start, query_count);
            });
        std::size_t failures = 0;
        double maximum = 0.;
        for (std::size_t i = 0; i < elements; ++i) {
            const vx_u32 row = (i / model_width) % g.queries;
            const float actual = output[guard + i];
            if (row < query_start || row >= query_start + query_count) {
                if (actual != sentinel) ++failures;
                continue;
            }
            const double error = std::abs(static_cast<double>(actual) - oracle[i]);
            maximum = std::max(maximum, error);
            if (!std::isfinite(actual) || error > 2e-6 + 1e-6 * std::abs(oracle[i]) ||
                    (oracle[i] == 0. && actual != 0.f)) {
                if (!failures) std::fprintf(stderr,
                    "%s order=%d index=%zu actual=%.9g oracle=%.12g error=%.9g\n",
                    g.name, permutation, i, actual, oracle[i], error);
                ++failures;
            }
        }
        for (std::size_t i = 0; i < guard; ++i)
            if (output[i] != sentinel || output[guard + elements + i] != sentinel)
                ++failures;
        std::printf("%s order=%s start=%u count=%u failures=%zu max_abs=%.12g %s\n",
            g.name, permutation ? "reversed" : "original", query_start,
            query_count, failures, maximum, failures ? "FAIL" : "PASS");
        std::printf("BITS %s order=%d hash=%016llx barriers=%llu\n", g.name, permutation,
            (unsigned long long)output_hash(output), (unsigned long long)block_barriers);
        pass = pass && failures == 0;
        reverse_keys(g, k, v, mask);
    }
    return pass;
}

/* A zero denominator forbids Q/K/V loads, including through restored arguments.
 * Null inputs turn an accidental load into a failure rather than a quiet NaN. */
static bool run_masked_null_inputs() {
    bool pass = true;
    for (vx_u32 width : {7u, 31u, 64u, 127u, 128u, 129u}) {
        constexpr vx_u32 keys = 48, heads = 4, kv_heads = 2;
        constexpr std::size_t guard = 8;
        constexpr float sentinel = -98765.5f;
        std::vector<vx_i32> mask(keys, 0);
        for (bool stored : {false, true}) {
            std::vector<float> output(heads * width + 2 * guard, sentinel);
            gridDim.z = cross_grid_tiles(1u, keys, width);
            for (blockIdx.z = 0u; blockIdx.z < gridDim.z; ++blockIdx.z)
            for (vx_u32 task = 0; task < heads + 5; ++task)
                launch_block(task, [&] {
                    auto record = cuda_argument_record((const float*)nullptr, (const float*)nullptr,
                        (const float*)nullptr, mask.data(), output.data() + guard, 1u, keys,
                        heads * width, kv_heads * width, heads, kv_heads, width, 1u, .73f, 0u, 1u);
                    vx_cuda_cross_sdpa_gqa_f32(nullptr, nullptr, nullptr, stored ? nullptr : mask.data(),
                        stored ? nullptr : output.data() + guard, stored ? 0u : 1u, stored ? 0u : keys,
                        stored ? 0u : heads * width, stored ? 0u : kv_heads * width,
                        stored ? 0u : heads, stored ? 0u : kv_heads, stored ? 0u : width,
                        stored ? 0u : 1u, stored ? 0.f : .73f, 0u, stored ? 0u : 1u,
                        stored ? record.bytes.data() : nullptr);
                });
            std::size_t failures = 0;
            for (std::size_t index = 0; index < output.size(); ++index) {
                const float expected = index < guard || index >= guard + heads * width ? sentinel : 0.f;
                if (output[index] != expected || (expected == 0.f && std::signbit(output[index]))) ++failures;
            }
            std::printf("NOLOAD width=%u stored=%d hash=%016llx failures=%zu %s\n", width, stored,
                (unsigned long long)output_hash(output), failures, failures ? "FAIL" : "PASS");
            pass = pass && !failures;
        }
    }
    return pass;
}

int main() {
    const Geometry cases[] = {
        {"Smol-short-row33", 1, 1, 33, 9, 3, 64, 0, 0, false},
        {"Smol-short-row35", 1, 1, 35, 9, 3, 64, 0, 0, false},
        {"Qwen-short-row33", 1, 1, 33, 16, 8, 128, 0, 0, false},
        {"Qwen-short-row35", 1, 1, 35, 16, 8, 128, 0, 0, false},
        {"GQA-short-mask-poison", 2, 1, 34, 4, 2, 128, 2, 0, false, false, true},
        {"GQA-short-all-mask-poison", 2, 1, 48, 4, 2, 64, 2, 0, true, false, true},
        {"MHA-Whisper-long", 2, 7, 1500, 2, 2, 64, 0, 0, false},
        {"MHA-long-tail", 1, 3, 1537, 2, 2, 37, 1, 0, false},
        {"MHA-batch-mask", 2, 3, 513, 2, 2, 40, 2, 0, false},
        {"MHA-query-mask", 1, 7, 1537, 2, 2, 64, 3, 0, false},
        {"GQA-batch-query-mask", 2, 3, 1537, 4, 2, 80, 4, 0, false},
        {"GQA-causal-tail", 2, 7, 513, 4, 2, 37, 4, 1, false},
        {"MHA-all-masked", 2, 3, 1500, 2, 2, 40, 1, 0, true},
        {"MHA-short-47", 2, 3, 47, 2, 2, 37, 4, 0, false},
        {"MHA-boundary-48", 2, 3, 48, 2, 2, 37, 4, 0, false},
        {"MHA-boundary-49", 2, 3, 49, 2, 2, 37, 4, 0, false},
        {"GQA-wide-short", 2, 3, 31, 4, 2, 128, 4, 0, false},
        {"GQA-wide-long", 1, 2, 97, 4, 1, 128, 1, 0, false},
        {"GQA-one-key", 2, 3, 1, 4, 2, 128, 0, 0, false},
        {"GQA-one-query", 2, 1, 49, 4, 1, 128, 2, 0, false},
        {"Smol-short-row34", 1, 1, 34, 9, 3, 64, 0, 0, false},
        {"Qwen-short-row34", 1, 1, 34, 16, 8, 128, 0, 0, false},
        {"GQA-warp-tail31", 2, 3, 33, 4, 2, 31, 4, 0, false},
        {"GQA-warp-boundary32", 2, 3, 33, 4, 2, 32, 4, 0, false},
        {"GQA-warp-tail127", 2, 3, 47, 4, 2, 127, 4, 0, false},
        {"GQA-warp-boundary128", 2, 3, 48, 4, 2, 128, 4, 0, false},
        {"GQA-short-fallback129", 2, 3, 48, 4, 2, 129, 4, 0, false},
        {"GQA-short-all-masked", 2, 1, 34, 4, 2, 128, 2, 0, true},
        {"MHA-short-cancellation", 2, 3, 35, 2, 2, 127, 4, 0, false, true},
        {"GQA-short-cancellation", 2, 3, 48, 4, 2, 128, 4, 0, false, true},
        {"GQA-subgroup-tail7", 2, 3, 35, 4, 2, 7, 4, 0, false},
        {"GQA-subgroup-boundary8", 2, 3, 35, 4, 2, 8, 4, 0, false},
        {"GQA-subgroup-tail9", 2, 3, 35, 4, 2, 9, 4, 0, false},
        {"GQA-subgroup-tail63", 2, 3, 35, 4, 2, 63, 4, 0, false},
        {"GQA-subgroup-boundary64", 2, 3, 35, 4, 2, 64, 4, 0, false},
        {"GQA-subgroup-tail65", 2, 3, 35, 4, 2, 65, 4, 0, false},
        {"GQA-key-group-tail15", 2, 3, 15, 4, 2, 64, 4, 0, false},
        {"GQA-key-group-boundary16", 2, 3, 16, 4, 2, 64, 4, 0, false},
        {"GQA-key-group-tail17", 2, 3, 17, 4, 2, 64, 4, 0, false},
        {"GQA-short64-all-masked", 2, 1, 35, 4, 2, 64, 2, 0, true},
        {"GQA-causal-short-prefill36", 1, 36, 256, 4, 2, 64, 4, 1, false},
    };
    bool pass = true;
    for (const auto& g : cases) pass = run(g) && pass;
    const Geometry self_cases[] = {
        {"SDPA-full-short", 2, 31, 31, 2, 2, 37, 0, 0, false},
        {"SDPA-full-long-mask", 2, 65, 65, 2, 2, 37, 2, 0, false},
        {"SDPA-range-causal", 2, 65, 65, 2, 2, 37, 4, 1, false},
        {"SDPA-query-mask", 1, 49, 49, 2, 2, 37, 3, 0, false},
        {"SDPA-all-masked", 1, 49, 49, 2, 2, 37, 1, 0, true},
        {"SDPA-channel-stride", 2, 3, 3, 2, 2, 257, 4, 0, false},
    };
    for (const auto& g : self_cases) {
        const bool row_range = std::string(g.name) == "SDPA-range-causal";
        pass = run_sdpa(g, row_range ? 49u : 0u,
            row_range ? 2u : g.queries) && pass;
    }
    const Geometry narrow_cases[] = {
        {"GQA-narrow-cancellation", 2, 3, 35, 4, 2, 64, 4, 0, false, true},
        {"GQA-narrow-mask-poison", 2, 1, 34, 4, 2, 64, 2, 0, false, false, true},
    };
    for (const auto& g : narrow_cases) pass = run(g) && pass;
    const Geometry cache_cases[] = {
        {"GQA-cache-boundary2048", 1, 1, 2048, 4, 1, 64, 1, 0, false},
        {"GQA-stream-boundary2049", 1, 1, 2049, 4, 1, 64, 1, 0, false},
        {"GQA-cache-mask-poison", 1, 1, 2048, 4, 1, 64, 2, 0, false, false, true},
        {"GQA-cache-all-mask-poison", 1, 1, 2048, 4, 1, 64, 2, 0, true, false, true},
    };
    for (const auto& g : cache_cases) pass = run(g) && pass;
    const Geometry cache_cancellation_cases[] = {
        {"GQA-cache-cancellation49", 1, 2, 49, 4, 1, 37, 4, 0, false, true},
        {"GQA-cache-cancellation512", 1, 2, 512, 4, 1, 64, 4, 0, false, true},
        {"GQA-cache-cancellation1536", 1, 2, 1536, 4, 1, 64, 4, 0, false, true},
        {"GQA-cache-cancellation2048", 1, 2, 2048, 4, 1, 128, 4, 0, false, true},
        {"GQA-stream-cancellation2049", 1, 2, 2049, 4, 1, 64, 4, 0, false, true},
    };
    for (const auto& g : cache_cancellation_cases) pass = run(g) && pass;
    return run_masked_null_inputs() && pass ? 0 : 1;
}
