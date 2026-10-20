/* Exercise the production CUDA ordering helpers, merge/weight kernels and
 * final CDF on the host. Tile barriers are modeled by executing each disjoint
 * compare stage to completion. This does not qualify GPU barriers or timing. */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using vx_u32 = std::uint32_t;
using vx_i32 = std::int32_t;
using vx_u64 = std::uint64_t;
#define VX_CUDA_GLOBAL
#define VX_CUDA_DEVICE
#define VX_CUDA_SHARED
struct HostDimension { vx_u32 x; };
static HostDimension threadIdx{}, blockIdx{};
static vx_u32 host_index;
static vx_u32 vx_global_x() { return host_index; }
static void vx_sync_threads() {}
#include "../src/backends/cuda/kernels/cuda_forward_generation_kernels.inc"

static double draw(std::uint64_t seed) {
    std::uint64_t value = seed + UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return static_cast<double>(value >> 11) * 0x1p-53;
}

static std::vector<float> fixture(vx_u32 count) {
    std::vector<float> values(count);
    for (vx_u32 i = 0; i < count; ++i)
        values[i] = (static_cast<int>(((i + 1u) * 2654435761u) >> 16) - 32768) / 8192.f;
    if (count > 83) values[17] = values[83] = 4.f;
    return values;
}

static std::vector<vx_i32> ordered(const std::vector<float>& values, vx_u32 keep) {
    std::vector<vx_i32> order(values.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<vx_i32>(i);
    std::stable_sort(order.begin(), order.end(), [&values](vx_i32 a, vx_i32 b) {
        return values[a] > values[b] || (values[a] == values[b] && a < b);
    });
    order.resize(keep);
    return order;
}

static vx_i32 reference(const std::vector<float>& values, double temperature,
        vx_u32 top_k, double top_p, double uniform) {
    for (float value : values) if (std::isnan(value)) return -1;
    const vx_u32 keep = top_k && top_k < values.size() ? top_k : values.size();
    const auto order = ordered(values, keep);
    const double maximum = values[order[0]];
    if (maximum == -std::numeric_limits<double>::infinity()) return -1;
    std::vector<double> weights(keep);
    double total = 0.;
    for (vx_u32 i = 0; i < keep; ++i) {
        weights[i] = maximum == std::numeric_limits<double>::infinity()
            ? (values[order[i]] == maximum ? 1. : 0.)
            : std::exp((static_cast<double>(values[order[i]]) - maximum) / temperature);
        total += weights[i];
    }
    double retained = 0.;
    vx_u32 admitted = 0;
    do { retained += weights[admitted++]; } while (admitted < keep && retained < top_p * total);
    double target = uniform * retained, cumulative = 0.;
    for (vx_u32 i = 0; i < admitted; ++i) {
        cumulative += weights[i];
        if (target < cumulative) return order[i];
    }
    return order[admitted - 1];
}

static bool run(const char* label, const std::vector<float>& values,
        double temperature = 1., vx_u32 top_k = 0, double top_p = 1.,
        const std::vector<double>& draws = {draw(0), draw(7), draw(UINT64_MAX)}) {
    const vx_u32 vocabulary = values.size();
    const vx_u32 keep = top_k && top_k < vocabulary ? top_k : vocabulary;
    constexpr vx_u32 guard = 9;
    const VxCudaGenerationCandidate sentinel{-123456789., -9999};
    std::vector<VxCudaGenerationCandidate> first(vocabulary + 2 * guard, sentinel);
    std::vector<VxCudaGenerationCandidate> second(vocabulary + 2 * guard, sentinel);
    auto* input = first.data() + guard;
    auto* output = second.data() + guard;
    for (vx_u32 base = 0; base < vocabulary; base += 256) {
        VxCudaGenerationCandidate tile[256];
        for (vx_u32 lane = 0; lane < 256; ++lane) {
            tile[lane] = base + lane < vocabulary ? vx_generation_candidate(values[base + lane], base + lane)
                : VxCudaGenerationCandidate{-std::numeric_limits<double>::infinity(), INT32_MAX};
        }
        for (vx_u32 width = 2; width <= 256; width *= 2)
            for (vx_u32 stride = width / 2; stride; stride /= 2)
                for (vx_u32 lane = 0; lane < 256; ++lane)
                    vx_generation_sort_tile_pair(tile, lane, width, stride);
        for (vx_u32 lane = 0; lane < 256 && base + lane < vocabulary && lane < keep; ++lane)
            input[base + lane] = tile[lane];
    }
    for (vx_u32 width = 256; width < vocabulary; width *= 2) {
        for (host_index = 0; host_index < vocabulary + 17; ++host_index)
            vx_cuda_generation_merge_f32(input, output, vocabulary, width, keep);
        std::swap(input, output);
    }
    bool valid = std::none_of(values.begin(), values.end(), [](float x) { return std::isnan(x); });
    bool pass = true;
    if (valid) {
        auto oracle_order = ordered(values, keep);
        for (vx_u32 i = 0; i < keep; ++i)
            if (input[i].id != oracle_order[i] || input[i].score != values[oracle_order[i]]) {
                std::fprintf(stderr, "%s order index=%u actual=%d expected=%d\n", label,
                    i, input[i].id, oracle_order[i]);
                pass = false; break;
            }
    } else if (input[0].id >= 0) pass = false;
    const double maximum = input[0].score;
    for (host_index = 0; host_index < keep + 17; ++host_index)
        vx_cuda_generation_weights_f32(input, output, keep, temperature);
    for (vx_u32 i = 0; i < keep; ++i) {
        const double expected_weight = input[0].id < 0 || maximum == -std::numeric_limits<double>::infinity() ? 0.
            : maximum == std::numeric_limits<double>::infinity() ? (input[i].score == maximum ? 1. : 0.)
            : vx_generation_exp((input[i].score - maximum) / temperature);
        if (std::memcmp(&expected_weight, &output[i].score, sizeof(double)) || output[i].id != input[i].id)
            pass = false;
    }
    for (double uniform : draws) {
        vx_i32 ids[3] = {-555, -555, -555}, selected = -777;
        host_index = 0;
        vx_cuda_generation_sample_f32(nullptr, ids, output, &selected, vocabulary, 1,
            temperature, top_k, top_p, uniform);
        vx_i32 expected = reference(values, temperature, top_k, top_p, uniform);
        if (selected != expected || ids[0] != -555 || ids[2] != -555 ||
                ids[1] != (expected < 0 ? -555 : expected)) {
            std::fprintf(stderr, "%s draw=%.17g selected=%d expected=%d\n", label, uniform, selected, expected);
            pass = false;
        }
        host_index = 1;
        selected = -777;
        vx_cuda_generation_sample_f32(nullptr, ids, output, &selected, vocabulary, 1,
            temperature, top_k, top_p, uniform);
        if (selected != -777) pass = false;
    }
    for (vx_u32 i = 0; i < guard; ++i)
        for (const auto* data : {first.data(), second.data()})
            if (data[i].score != sentinel.score || data[i].id != sentinel.id ||
                    data[guard + vocabulary + i].score != sentinel.score ||
                    data[guard + vocabulary + i].id != sentinel.id) pass = false;
    std::printf("%s vocab=%u top_k=%u draws=%zu %s\n", label, vocabulary, top_k, draws.size(), pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    bool pass = true;
    for (vx_u32 count : {1u, 2u, 7u, 255u, 256u, 257u, 511u, 512u, 513u, 49152u, 151936u, 1048576u}) {
        auto values = fixture(count);
        pass = run("default-stable-order", values) && pass;
        pass = run("filtered-stable-order", values, .7, std::min(count, 40u), .85) && pass;
    }
    pass = run("max-admitted-vocab-topk", fixture(1048576), .7, 40, .85) && pass;
    for (vx_u32 keep : {255u, 256u, 257u})
        pass = run("topk-tile-boundary", fixture(513), .7, keep, .85) && pass;
    pass = run("topk-one-with-ties", std::vector<float>(513, 2.f), 1., 1) && pass;
    std::vector<double> boundaries = {0., std::nextafter(.5, 0.), .5, std::nextafter(.5, 1.), std::nextafter(1., 0.)};
    pass = run("exact-cdf-boundary", {0.f, 0.f}, 1., 0, 1., boundaries) && pass;
    pass = run("exact-top-p-boundary", std::vector<float>(513, 0.f), 1., 0, 256. / 513., boundaries) && pass;
    pass = run("positive-infinity-ties", {INFINITY, -INFINITY, 0.f, INFINITY}, 1., 0, 1., boundaries) && pass;
    pass = run("all-negative-infinity", std::vector<float>(513, -INFINITY)) && pass;
    pass = run("negative-infinity-and-underflow", {0.f, -INFINITY, -1000.f, -745.f, -744.f}) && pass;
    auto invalid = fixture(513); invalid[512] = NAN; invalid[3] = NAN;
    pass = run("multiple-nans-survive-topk", invalid, 1., 1) && pass;
    pass = run("tiny-temperature", {1.f, 1.f, 0.f, -1.f}, std::numeric_limits<double>::min(), 0, 1., boundaries) && pass;
    pass = run("huge-temperature", {1.f, 1.f, 0.f, -1.f}, std::numeric_limits<double>::max(), 0, 1., boundaries) && pass;
    pass = run("signed-zero-stable-ties", {-0.f, 0.f, -0.f, 0.f}, 1., 0, 1., boundaries) && pass;
    return pass ? 0 : 1;
}
