"""Exercise the private benchmark fixture's allocation/lifetime contract.

The operator's arithmetic oracle lives in native/tests/test_packed_linear_i8.c.
These tests cover dynamic storage, admission and failed initialization instead.
"""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = r'''
#ifdef malloc
#undef malloc
#endif
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int tile_bench_init(int,int,int,int,int);
int tile_bench_run(int);
uintptr_t tile_bench_output_ptr(void);
uint32_t tile_bench_output_bytes(void);
uint32_t tile_bench_hash(void);
uint32_t tile_bench_scratch_bytes(void);
int tile_bench_check_scratch(void);
uint32_t tile_bench_memory_limit(void);
uint32_t tile_bench_required_bytes(int,int,int,int,int);
void tile_bench_release(void);
int tile_bench_all_bits(void);

/* Only the fixture TU's malloc calls are redirected to this function. */
static int fail_after = -1;
static unsigned allocation_calls;
void* fixture_test_malloc(size_t bytes) {
    allocation_calls++;
    if (fail_after == 0) return NULL;
    if (fail_after > 0) fail_after--;
    return malloc(bytes);
}

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

static int empty_state(void) {
    CHECK(tile_bench_output_ptr() == 0);
    CHECK(tile_bench_output_bytes() == 0);
    CHECK(tile_bench_scratch_bytes() == 0);
    CHECK(tile_bench_run(1) != 0);
    return 0;
}

static int run_case(int rows, int k, int n, int bits, int group) {
    uint32_t needed = tile_bench_required_bytes(rows,k,n,bits,group);
    CHECK(needed > 0 && needed <= tile_bench_memory_limit());
    CHECK(needed >= 4u*((uint64_t)rows*k + (uint64_t)rows*n));
    CHECK(tile_bench_init(rows,k,n,bits,group) == 0);
    CHECK(tile_bench_output_ptr() != 0);
    CHECK(tile_bench_output_bytes() == 4u*(uint64_t)rows*n);
    CHECK(tile_bench_scratch_bytes() <= needed);
    CHECK(tile_bench_run(1) == 0);
    CHECK(tile_bench_check_scratch() == 0);
    float* out = (float*)tile_bench_output_ptr();
    for (uint64_t i=0;i<(uint64_t)rows*n;i++) CHECK(isfinite(out[i]));
    uint32_t first = tile_bench_hash();
    CHECK(tile_bench_run(2) == 0);
    CHECK(tile_bench_hash() == first && tile_bench_check_scratch() == 0);
    return 0;
}

static int shape_growth(void) {
    /* Exceed each former hard-coded dimension independently. */
    CHECK(run_case(1609,32,4,4,32) == 0);
    CHECK(run_case(2,1284,4,4,32) == 0);
    CHECK(run_case(2,1285,4,4,256) == 0);
    CHECK(run_case(2,32,1537,4,32) == 0);
    CHECK(run_case(1,1,1,4,32) == 0);
    tile_bench_release();
    return empty_state();
}

static int admission(void) {
    const int invalid[][5] = {
        {0,32,4,4,32}, {-1,32,4,4,32}, {2,0,4,4,32}, {2,-1,4,4,32},
        {2,32,0,4,32}, {2,32,-1,4,32}, {2,32,4,1,32}, {2,32,4,16,32},
        {2,32,4,4,0}, {2,32,4,4,16}, {2,32,4,4,33}, {2,32,4,4,512},
        {INT_MAX,1,1,4,32}, {1,INT_MAX,1,8,32}, {1,1,INT_MAX,4,32},
        {INT_MAX,INT_MAX,INT_MAX,8,256}
    };
    CHECK(tile_bench_memory_limit() == 128u*1024u*1024u);
    for (unsigned i=0;i<sizeof(invalid)/sizeof(invalid[0]);i++) {
        const int* q=invalid[i];
        CHECK(run_case(2,32,4,4,32) == 0);
        unsigned before=allocation_calls;
        uint32_t hash=tile_bench_hash();
        CHECK(tile_bench_required_bytes(q[0],q[1],q[2],q[3],q[4]) == 0);
        CHECK(allocation_calls == before); /* Admission itself allocates nothing. */
        CHECK(tile_bench_run(1) == 0 && tile_bench_hash() == hash);
        CHECK(tile_bench_init(q[0],q[1],q[2],q[3],q[4]) != 0);
        CHECK(allocation_calls == before); /* Invalid requests never reach malloc. */
        CHECK(empty_state() == 0);
    }
    return 0;
}

static int lifetime(void) {
    CHECK(tile_bench_all_bits() == 0);
    const int bits[]={2,3,4,8}, groups[]={32,64,128,256};
    for (unsigned i=0;i<sizeof(bits)/sizeof(bits[0]);i++)
        for (unsigned j=0;j<sizeof(groups)/sizeof(groups[0]);j++) {
            CHECK(run_case(6,257,8,bits[i],groups[j]) == 0);
            uint32_t hash=tile_bench_hash();
            CHECK(tile_bench_run(0) != 0 && tile_bench_run(-1) != 0 && tile_bench_run(10001) != 0);
            CHECK(tile_bench_hash() == hash);
            CHECK(tile_bench_init(6,257,8,bits[i],groups[j]) == 0);
            CHECK(tile_bench_run(1) == 0 && tile_bench_hash() == hash);
            CHECK(tile_bench_check_scratch() == 0);
            tile_bench_release();
            tile_bench_release();
            CHECK(empty_state() == 0);
        }
    return 0;
}

static int allocation_failure(void) {
    /* Fail each allocation, including failure after several successful ones.
     * ASan/LSan verifies that partially initialized storage is released. */
    unsigned before=allocation_calls;
    CHECK(tile_bench_init(10,257,8,4,64) == 0);
    unsigned allocations=allocation_calls-before;
    CHECK(allocations > 0 && allocations <= INT_MAX);
    tile_bench_release();
    for (int position=0;position<(int)allocations;position++) {
        CHECK(run_case(8,65,8,4,32) == 0);
        fail_after=position;
        CHECK(tile_bench_init(10,257,8,4,64) != 0);
        fail_after=-1;
        CHECK(empty_state() == 0);
        tile_bench_release();
    }
    CHECK(run_case(2,32,4,4,32) == 0);
    tile_bench_release();
    return empty_state();
}

int main(int argc,char** argv) {
    CHECK(argc == 2);
    CHECK(empty_state() == 0);
    if (!strcmp(argv[1],"shape-growth")) return shape_growth();
    if (!strcmp(argv[1],"admission")) return admission();
    if (!strcmp(argv[1],"lifetime")) return lifetime();
    if (!strcmp(argv[1],"allocation-failure")) return allocation_failure();
    return 2;
}
'''


class PackedLinearFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("clang")
        if compiler is None:
            raise unittest.SkipTest("clang is required for the fixture sanitizer checks")
        cls.temporary = tempfile.TemporaryDirectory(prefix="packed-linear-fixture-")
        cls.addClassCleanup(cls.temporary.cleanup)
        directory = Path(cls.temporary.name)
        driver = directory / "driver.c"
        driver.write_text(DRIVER)
        cls.executable = directory / "fixture-check"
        subprocess.run([
            compiler, "-std=c11", "-O1", "-g", "-ffp-contract=off",
            "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer", "-Dmalloc=fixture_test_malloc",
            "-DVX_PACKED_LINEAR_TILE_ROWS=64", "-DVX_PACKED_LINEAR_TILE2=1",
            "-DVX_PACKED_LINEAR_TILE_ALL_BITS=0", "-I" + str(ROOT / "native/src/kernels"),
            str(driver), str(ROOT / "examples/tiny_receipt_vqa/benchmarks/packed_row_tiles.c"),
            "-lm", "-o", str(cls.executable),
        ], check=True, capture_output=True, text=True)

    def check_scenario(self, name):
        environment = {**os.environ, "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1",
                       "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"}
        result = subprocess.run([str(self.executable), name], env=environment,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shapes_beyond_former_fixed_buffers(self):
        self.check_scenario("shape-growth")

    def test_invalid_and_oversized_requests_clear_state_without_allocation(self):
        self.check_scenario("admission")

    def test_reinitialization_and_release_preserve_canaries(self):
        self.check_scenario("lifetime")

    def test_partial_allocation_failures_release_every_buffer(self):
        self.check_scenario("allocation-failure")


if __name__ == "__main__":
    unittest.main()
