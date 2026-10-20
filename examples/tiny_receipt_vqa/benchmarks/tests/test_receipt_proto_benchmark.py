"""Receipt timing must exclude output reads without dropping needed KV state."""
import ctypes
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import receipt_proto_benchmark as bench

pb = bench.pb


class ReceiptBenchmarkTests(unittest.TestCase):
    def native(self, *, fail_read=False):
        clock = SimpleNamespace(ns=0)
        calls = []
        report = pb.OperationReport()
        data = np.arange(4, dtype=np.float32)
        runner = bench.Native.__new__(bench.Native)
        runner.contracts = {"digit": {"inputs": {"image": "input"}, "outputs": {"slot_logits": "output"}}}
        runner.contexts, runner.output_buffers = {"digit": 1}, {"digit": {"output": np.empty(4, np.float32)}}
        runner.pending_reports, runner.pending_timings = [], []
        runner.decoder_started = False

        def execute(request):
            calls.append("execute")
            clock.ns += 20_000_000
            tensor = request.inputs.values[0]
            self.assertIsNotNone(tensor.borrowed)
            self.assertFalse(tensor.inline)
            actual = np.frombuffer(ctypes.string_at(tensor.borrowed.resource.handle, tensor.borrowed.length_bytes), np.float32)
            np.testing.assert_array_equal(actual, [0, 2, 4, 6])
            self.assertEqual(request.scope.entrypoint, "evaluate")
            self.assertEqual(request.outputs.storage, pb.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER)
            return pb.ExecuteResponse(report=report, metrics=pb.ExecutionMetrics(host_time_ns=7_000_000),
                outputs=[pb.Tensor(name="output", shape=[4], dtype=pb.DataType.DATA_TYPE_F32,
                    buffer=pb.BufferView(buffer_id=1, length_bytes=data.nbytes))])

        def copy_tensors(request):
            calls.append("copy_tensors")
            clock.ns += 100_000_000
            if fail_read:
                raise RuntimeError("read failed")
            self.assertEqual(len(request.sources), 1)
            destination = request.into[0]
            ctypes.memmove(destination.resource.handle, data.ctypes.data, data.nbytes)
            return pb.TensorBatch(report=report, outputs=[pb.Tensor(name="output", shape=[4],
                dtype=pb.DataType.DATA_TYPE_F32, borrowed=destination)])

        def release_buffers(request):
            calls.append("release_buffers")
            self.assertEqual(request.buffer_ids, [1])
            clock.ns += 1_000_000
            return report

        runner.client = SimpleNamespace(execute=execute)
        runner.buffers = SimpleNamespace(copy_tensors=copy_tensors, release_buffers=release_buffers)
        return runner, clock, calls

    def test_completed_execution_skips_poll_and_excludes_readback(self):
        runner, clock, calls = self.native()
        # A strided input must become a live contiguous BufferView, not bytes.
        with patch.object(bench.time, "perf_counter_ns", lambda: clock.ns):
            output, elapsed = runner.run("digit", {"image": np.arange(8, dtype=np.float32)[::2]})
        np.testing.assert_array_equal(output["slot_logits"], [0, 1, 2, 3])
        self.assertEqual(calls, ["execute", "copy_tensors", "release_buffers"])
        timing = bench.execution_metrics(runner)
        self.assertEqual((timing["execute_ms"], timing["copy_tensors_ms"], elapsed), (20, 100, 121))
        self.assertEqual(timing["runtime_execution_ms"], 7)

    def test_failed_read_still_releases_result(self):
        runner, clock, calls = self.native(fail_read=True)
        with patch.object(bench.time, "perf_counter_ns", lambda: clock.ns):
            with self.assertRaisesRegex(RuntimeError, "read failed"):
                runner.run("digit", {"image": np.arange(8, dtype=np.float32)[::2]})
        self.assertEqual(calls[-1], "release_buffers")
        self.assertFalse(runner.pending_timings)

    def test_last_step_and_eos_have_no_cache_consumer(self):
        for token, need_cache in [(2, True), (1, False), (1, True)]:
            with self.subTest(token=token, need_cache=need_cache):
                reads = []
                logits = np.eye(3, dtype=np.float32)[token]
                def read(name):
                    reads.append(name)
                    return logits if name == "logits" else np.ones(1, np.float32)
                output = bench.read_needed_outputs("decoder", read, need_cache=need_cache)
                if token == 2 or not need_cache:
                    self.assertEqual(reads, ["logits"])
                else:
                    self.assertEqual(len(reads), 10)
                    self.assertIn("present_padding_mask", output)
                    for layer in range(4):
                        self.assertIn(f"present_k_{layer}", output)
                        self.assertIn(f"present_v_{layer}", output)


if __name__ == "__main__":
    unittest.main()
