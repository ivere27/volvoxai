from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]

# Include the production serializer so the generated response cleanup is tested
# against partially populated messages, with nonzero allocator contents.
HARNESS = r'''
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "vx_api_training.c"

typedef struct Allocations {
    size_t calls;
    size_t fail_at;
    size_t live;
    void* blocks[8];
} Allocations;

static void* allocate(void* context, size_t size) {
    Allocations* allocations = (Allocations*)context;
    void* block;
    size_t index;
    if (++allocations->calls == allocations->fail_at) return NULL;
    block = malloc(size);
    assert(block);
    memset(block, 0xa5, size);
    for (index = 0; index < 8; index++) {
        if (!allocations->blocks[index]) {
            allocations->blocks[index] = block;
            allocations->live++;
            return block;
        }
    }
    abort();
}

static void deallocate(void* context, void* block) {
    Allocations* allocations = (Allocations*)context;
    size_t index;
    if (!block) return;
    for (index = 0; index < 8; index++) {
        if (allocations->blocks[index] == block) {
            allocations->blocks[index] = NULL;
            allocations->live--;
            free(block);
            return;
        }
    }
    abort(); /* An uninitialized or already freed pointer is never valid. */
}

/* Report conversion is independent of the metric ownership under test. */
int vx_api_report_attach(const SynurangLiteAllocator* allocator,
                         VolvoxaiV1OperationReport** slot,
                         const VxReport* report) {
    (void)allocator;
    (void)slot;
    (void)report;
    return 1;
}

int main(void) {
    size_t fail_at;
    for (fail_at = 0; fail_at <= 4; fail_at++) {
        Allocations allocations = {0};
        SynurangLiteAllocator allocator = {&allocations, allocate, deallocate};
        VolvoxaiV1TrainStepResult response;
        VxTrainStepResult result = VX_TRAIN_STEP_RESULT_INIT;
        VxReport report = VX_REPORT_INIT;
        int status;
        allocations.fail_at = fail_at;
        result.metric_count = 3;
        strcpy(result.metrics[0].name, "primary");
        strcpy(result.metrics[1].name, "auxiliary");
        strcpy(result.metrics[2].name, "regularization");
        result.metrics[1].loss = 0.25f;
        volvoxai_v1_train_step_result_init_with_allocator(&response, &allocator);
        status = vx_api_training_result(&response, &result, &report);
        assert(status == (fail_at ? -1 : 0));
        if (!fail_at) {
            assert(response.field_metrics.len == 3);
            assert(response.field_metrics.data[1].field_loss == 0.25f);
            assert(response.field_metrics.data[1].field_name.len == 9);
            assert(memcmp(response.field_metrics.data[1].field_name.data,
                          "auxiliary", 9) == 0);
        }
        volvoxai_v1_train_step_result_free(&response);
        assert(allocations.live == 0);
        /* Generated cleanup is also safe when called again. */
        volvoxai_v1_train_step_result_free(&response);
        assert(allocations.live == 0);
    }
    return 0;
}
'''


class TrainingResponseAllocationTests(unittest.TestCase):
    def test_metric_allocation_failures_are_safe_to_release(self) -> None:
        compiler = shlex.split(os.environ.get("CC", "cc"))
        if not compiler or not shutil.which(compiler[0]):
            self.fail("A C compiler is required for the allocation contract")
        includes = [
            "native/include", "native/src", "native/src/api",
            "native/src/runtime", "native/src/training", "native/src/backends",
            "native/src/kernels", "native/third_party",
            "native/third_party/synurang/include", "runtime/generated/c",
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "allocation.c"
            binary = Path(directory) / "allocation"
            source.write_text(HARNESS)
            subprocess.run(
                [
                    *compiler, "-std=c11", "-O1", "-ffunction-sections",
                    "-fdata-sections", "-DVOLVOXAI_ENABLE_TRAINING=1",
                    *(f"-I{ROOT / path}" for path in includes),
                    str(source), str(ROOT / "runtime/generated/c/volvoxai_lite.c"),
                    "-Wl,--gc-sections", "-o", str(binary),
                ],
                check=True,
            )
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
