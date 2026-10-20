"""Exercise production CUDA admission predicates with bounded descriptors."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from tools.tests import function_source




def struct_source(source, name):
    start = source.index("typedef struct " + name + " {")
    end = source.index("} " + name + ";", start) + len("} " + name + ";")
    return source[start:end] + "\n"


FIXTURE = r'''
static VxDeclaredTensor tensor(uint32_t rank, int64_t batch, int64_t sequence,
                               int64_t width) {
    VxDeclaredTensor t = {0};
    uint64_t elements = 1u;
    t.rank = rank;
    t.dtype = VX_DTYPE_F32;
    t.minimums[0] = t.maximums[0] = rank == 3u ? batch : sequence;
    t.minimums[rank - 2u] = t.maximums[rank - 2u] = sequence;
    t.minimums[rank - 1u] = t.maximums[rank - 1u] = width;
    for (uint32_t axis = 0; axis < rank; ++axis) {
        t.kinds[axis] = VX_DIMENSION_FIXED;
        t.multiples[axis] = 1;
        elements *= (uint64_t)t.maximums[axis];
    }
    t.maximum_byte_size = elements * 4u;
    return t;
}

int main(void) {
    VxNativeGpuDomainLimits limits = {0};
    cJSON* params = cJSON_Parse("{\"heads\":2,\"causal\":true}");
    cJSON* grouped = cJSON_Parse("{\"heads\":2,\"kv_heads\":1,\"causal\":false}");
    VxDeclaredTensor qkv = tensor(3u, 2, 9, 384);
    VxDeclaredTensor out = tensor(3u, 2, 9, 128);
    VxDeclaredTensor q = out, k = tensor(3u, 2, 13, 128), v = k;
    assert(params && grouped);
    limits.maximum_buffer_bytes = UINT32_MAX;
    limits.maximum_grid[0] = 7u;
    limits.maximum_grid[1] = 6u;
    limits.maximum_grid[2] = 4u;
    limits.maximum_block[0] = 128u;
    limits.maximum_block[1] = limits.maximum_block[2] = 1u;
    limits.maximum_threads_per_block = 128u;
    limits.maximum_shared_memory_bytes = 208u;
    assert(vx_native_gpu_cuda_sdpa_proven(&qkv, &out, NULL, params, &limits));
    assert(!vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, NULL,
        params, &limits, VX_BACKEND_KIND_CUDA));
    limits.maximum_shared_memory_bytes = 8716u;
    assert(!vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, NULL,
        params, &limits, VX_BACKEND_KIND_CUDA));
    limits.maximum_shared_memory_bytes = 8720u;
    assert(vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, NULL,
        params, &limits, VX_BACKEND_KIND_CUDA));
    limits.maximum_shared_memory_bytes = 207u;
    assert(!vx_native_gpu_cuda_sdpa_proven(&qkv, &out, NULL, params, &limits));
    assert(!vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, NULL,
        params, &limits, VX_BACKEND_KIND_CUDA));
    limits.maximum_shared_memory_bytes = 8720u;
    limits.maximum_grid[1] = 5u;
    assert(!vx_native_gpu_cuda_sdpa_proven(&qkv, &out, NULL, params, &limits));
    assert(!vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, NULL,
        params, &limits, VX_BACKEND_KIND_CUDA));
    limits.maximum_block[0] = limits.maximum_threads_per_block = 256u;
    assert(vx_native_gpu_one_d_launch_proven(36u, VX_BACKEND_KIND_CUDA, &limits));
    limits.maximum_grid[1] = 6u;
    limits.maximum_block[0] = limits.maximum_threads_per_block = 128u;
    for (unsigned int mode = 1u; mode <= 4u; ++mode) {
        VxDeclaredTensor mask = {0};
        mask.dtype = VX_DTYPE_I32;
        mask.rank = mode == 1u ? 1u : mode == 4u ? 3u : 2u;
        for (uint32_t axis = 0u; axis < mask.rank; ++axis) {
            mask.kinds[axis] = VX_DIMENSION_FIXED;
            mask.minimums[axis] = mask.maximums[axis] = 9;
        }
        if (mode == 2u || mode == 4u)
            mask.minimums[0] = mask.maximums[0] = 2;
        uint64_t elements;
        assert(vx_native_gpu_tensor_maximum_elements(&mask, &elements));
        mask.maximum_byte_size = elements * 4u;
        assert(vx_native_gpu_cuda_sdpa_proven(&qkv, &out, &mask, params, &limits));
        mask.dtype = VX_DTYPE_F32;
        assert(!vx_native_gpu_cuda_sdpa_proven(&qkv, &out, &mask, params, &limits));
    }
    qkv.kinds[1] = out.kinds[1] = VX_DIMENSION_SYMBOLIC;
    qkv.symbols[1] = out.symbols[1] = "S";
    qkv.minimums[1] = out.minimums[1] = 1;
    assert(vx_native_gpu_cuda_sdpa_proven(&qkv, &out, NULL, params, &limits));
    out.symbols[1] = "different";
    assert(!vx_native_gpu_cuda_sdpa_proven(&qkv, &out, NULL, params, &limits));
    q = tensor(3u, 2, 9, 256);
    out = q;
    assert(vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, NULL,
        grouped, &limits, VX_BACKEND_KIND_CUDA));
    k = v = tensor(3u, 2, 13, 256);
    assert(vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, NULL,
        params, &limits, VX_BACKEND_KIND_CUDA));
    qkv = tensor(2u, 1, 9, 390);
    out = tensor(2u, 1, 9, 130);
    assert(!vx_native_gpu_cuda_sdpa_proven(&qkv, &out, NULL, params, &limits));
    qkv = tensor(2u, 1, 9, 384);
    out = tensor(2u, 1, 9, 128);
    assert(vx_native_gpu_cuda_sdpa_proven(&qkv, &out, NULL, params, &limits));
    qkv.maximum_byte_size -= 4u;
    assert(!vx_native_gpu_cuda_sdpa_proven(&qkv, &out, NULL, params, &limits));
    /* Actual Qwen full-prefix Generate repeats its eight KV heads before
     * CrossSDPA. The resulting equal-head compact route omits kv_heads. */
    cJSON* qwen = cJSON_Parse(
        "{\"heads\":16,\"causal\":true,\"scale\":0.08838834764831843}");
    assert(qwen && !cJSON_GetObjectItemCaseSensitive(qwen, "kv_heads"));
    q = k = v = out = tensor(3u, 1, 128, 2048);
    VxDeclaredTensor keep = tensor(2u, 1, 1, 128);
    keep.dtype = VX_DTYPE_I32;
    for (uint32_t axis = 0u; axis < 2u; ++axis)
        keep.multiples[axis] = 1;
    limits.maximum_grid[0] = 2147483647u;
    limits.maximum_grid[1] = limits.maximum_grid[2] = 65535u;
    limits.maximum_block[0] = limits.maximum_block[1] = 1024u;
    limits.maximum_block[2] = 64u;
    limits.maximum_threads_per_block = 1024u;
    limits.maximum_shared_memory_bytes = 49152u;
    assert(vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, &keep,
        qwen, &limits, VX_BACKEND_KIND_CUDA));
    q.kinds[1] = k.kinds[1] = v.kinds[1] = out.kinds[1] =
        keep.kinds[1] = VX_DIMENSION_SYMBOLIC;
    q.symbols[1] = k.symbols[1] = v.symbols[1] = out.symbols[1] =
        keep.symbols[1] = "S";
    q.minimums[1] = k.minimums[1] = v.minimums[1] = out.minimums[1] =
        keep.minimums[1] = 1;
    assert(vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, &keep,
        qwen, &limits, VX_BACKEND_KIND_CUDA));
    keep.symbols[1] = "unrelated";
    assert(!vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, &keep,
        qwen, &limits, VX_BACKEND_KIND_CUDA));
    keep.symbols[1] = "S";
    limits.maximum_shared_memory_bytes = 207u;
    assert(!vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, &keep,
        qwen, &limits, VX_BACKEND_KIND_CUDA));
    limits.maximum_shared_memory_bytes = 8720u;
    limits.maximum_grid[0] = 2047u;
    limits.maximum_grid[1] = 1u;
    assert(!vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, &keep,
        qwen, &limits, VX_BACKEND_KIND_CUDA));
    limits.maximum_grid[0] = 2048u;
    assert(vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, &keep,
        qwen, &limits, VX_BACKEND_KIND_CUDA));
    limits.maximum_threads_per_block = 127u;
    assert(!vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, &keep,
        qwen, &limits, VX_BACKEND_KIND_CUDA));
    limits.maximum_threads_per_block = 128u;
    q = k = v = out = tensor(3u, 1, 128, 2064);
    assert(!vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, NULL,
        qwen, &limits, VX_BACKEND_KIND_CUDA));
    /* Packed SDPA still uses its distinct public 64-channel route. */
    qkv = tensor(3u, 1, 128, 6144);
    out = tensor(3u, 1, 128, 2048);
    assert(!vx_native_gpu_cuda_sdpa_proven(&qkv, &out, NULL, qwen, &limits));
    /* Compile admission must cover projected Q1 even when full Q is fixed
     * above one. Self attention reserves 208B; every cross entry reserves 8720B. */
    cJSON* decode = cJSON_Parse("{\"heads\":16,\"kv_heads\":8,\"causal\":false}");
    assert(decode);
    q = out = tensor(3u, 1, 31, 2048);
    k = v = tensor(3u, 1, 35, 1024);
    limits.maximum_grid[0] = UINT32_MAX;
    limits.maximum_grid[1] = 65535u;
    limits.maximum_shared_memory_bytes = 8720u;
    for (uint32_t z = 1u; z <= 4u; ++z) {
        limits.maximum_grid[2] = z;
        assert(vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, NULL,
            decode, &limits, VX_BACKEND_KIND_CUDA) == (z == 4u));
    }
    k = v = tensor(3u, 1, 49, 1024);
    limits.maximum_grid[2] = 1u;
    assert(vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, NULL,
        decode, &limits, VX_BACKEND_KIND_CUDA));
    limits.maximum_shared_memory_bytes = 8719u;
    assert(!vx_native_gpu_cuda_cross_sdpa_proven(&q, &k, &v, &out, NULL,
        decode, &limits, VX_BACKEND_KIND_CUDA));
    limits.maximum_shared_memory_bytes = 8720u;
    uint32_t widths[] = {1u, 31u, 32u, 33u, 64u, 127u, 128u, 129u};
    uint32_t keys[] = {35u, 48u, 49u, 2048u, 2049u};
    uint32_t queries[] = {1u, 2u, 31u};
    VxCudaAttentionLaunchPlan launch;
    limits.maximum_grid[2] = 4u;
    for (size_t qi = 0u; qi < sizeof queries / sizeof queries[0]; ++qi)
        for (size_t ki = 0u; ki < sizeof keys / sizeof keys[0]; ++ki)
            for (size_t wi = 0u; wi < sizeof widths / sizeof widths[0]; ++wi) {
                uint32_t tiles = queries[qi] == 1u && keys[ki] <= 48u && widths[wi] <= 128u
                    ? (widths[wi] + 31u) / 32u : 1u;
                assert(vx_cuda_cross_attention_launch_plan(16u * queries[qi],
                    queries[qi], keys[ki], widths[wi], limits.maximum_grid,
                    limits.maximum_block, 128u, 8720u, &launch));
                assert(launch.grid[2] == tiles && launch.static_shared_bytes == 8720u);
            }
    assert(!vx_cuda_cross_attention_launch_plan(UINT32_MAX / 4u + 1u, 1u, 48u,
        128u, limits.maximum_grid, limits.maximum_block, 128u, 8720u, &launch));
    assert(vx_cuda_cross_attention_launch_plan(UINT32_MAX, 31u, 48u,
        128u, limits.maximum_grid, limits.maximum_block, 128u, 8720u, &launch));
    assert(launch.grid[2] == 1u);
    cJSON_Delete(decode);
    cJSON_Delete(qwen);
    cJSON_Delete(params);
    cJSON_Delete(grouped);
    puts("PASS production CUDA packed/MHA/GQA admission predicates");
    return 0;
}
'''


class CudaAttentionLaunchDomainTests(unittest.TestCase):
    def test_production_attention_admission(self):
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        runtime = root / "native/src/runtime"
        source = (runtime / "public_api_native_gpu_domain.inc").read_text()
        symbolic = (runtime / "public_api_symbolic_dimensions.inc").read_text()
        translation = '''#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "volvoxai_types.h"
#include "generated/backend_vocabulary.h"
#include "cJSON.h"
#include "cuda/attention_launch_plan.h"
'''
        translation += struct_source((runtime / "public_api.c").read_text(), "VxDeclaredTensor")
        translation += struct_source(source, "VxNativeGpuDomainLimits")
        for name in ("vx_declared_dimension_fixed_value", "vx_declared_dimensions_provably_equal"):
            translation += function_source(symbolic, name)
        for name in ("vx_native_gpu_tensor_maximum_elements", "vx_native_gpu_tensor_maximum_product",
                     "vx_native_gpu_integer_parameter", "vx_native_gpu_tensor_shapes_equal",
                     "vx_native_gpu_tensor_domains_equal", "vx_native_gpu_one_d_launch_proven",
                     "vx_native_gpu_cuda_typed_span_proven", "vx_native_gpu_positive_f32_parameter",
                     "vx_native_gpu_cuda_attention_mask_proven", "vx_native_gpu_cuda_cross_sdpa_proven",
                     "vx_native_gpu_cuda_sdpa_proven"):
            translation += function_source(source, name)
        translation += FIXTURE
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-attention-domain-") as directory:
            fixture = Path(directory) / "domain.c"
            binary = Path(directory) / "domain"
            fixture.write_text(translation)
            built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I" + str(root / "native/include"), "-I" + str(root / "native/third_party"),
                "-I" + str(root / "native/src"),
                "-I" + str(root / "native/src/backends"), str(fixture),
                str(root / "native/third_party/cJSON.c"), "-lm", "-o", str(binary)],
                capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn("PASS production CUDA", checked.stdout)


if __name__ == "__main__":
    unittest.main()
