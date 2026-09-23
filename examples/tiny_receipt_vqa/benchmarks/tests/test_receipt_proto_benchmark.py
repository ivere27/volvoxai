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
    def native(self, *, pending=False, fail_read=False):
        clock = SimpleNamespace(ns=0)
        calls = []
        report = pb.OperationReport()
        data = np.arange(4, dtype=np.float32)
        runner = bench.Native.__new__(bench.Native)
        runner.contracts = {"digit": {"inputs": {"image": "input"}, "outputs": {"slot_logits": "output"}}}
        runner.contexts, runner.output_buffers = {"digit": 1}, {"digit": {"output": np.empty(4, np.float32)}}
        runner.pending_reports, runner.pending_timings = [], []

        def execute(request):
            calls.append("execute")
            clock.ns += 20_000_000
            tensor = request.inputs[0]
            self.assertIsNotNone(tensor.borrowed)
            self.assertFalse(tensor.inline)
            actual = np.frombuffer(ctypes.string_at(tensor.borrowed.resource.handle, tensor.borrowed.length_bytes), np.float32)
            np.testing.assert_array_equal(actual, [0, 2, 4, 6])
            return pb.ExecutionResultHandle(result_id=1, report=report, metrics=pb.ExecutionMetrics(host_time_ns=7_000_000),
                state=pb.ResultState.RESULT_STATE_PENDING if pending else pb.ResultState.RESULT_STATE_READY)

        def get_result(request):
            calls.append("get_result")
            clock.ns += 3_000_000
            return pb.ResultInfo(result_id=1, state=pb.ResultState.RESULT_STATE_READY, report=pb.OperationReport())

        def read_output(request):
            calls.append("read_output")
            clock.ns += 100_000_000
            if fail_read:
                raise RuntimeError("read failed")
            ctypes.memmove(request.into.resource.handle, data.ctypes.data, data.nbytes)
            return pb.ReadOutputResponse(required_bytes=data.nbytes, report=report,
                tensor=pb.Tensor(shape=[4], dtype=pb.DataType.DATA_TYPE_F32, borrowed=request.into))

        def release_result(request):
            calls.append("release_result")
            clock.ns += 1_000_000
            return report

        runner.client = SimpleNamespace(execute=execute, get_result=get_result,
            read_output=read_output, release_result=release_result)
        return runner, clock, calls

    def test_completed_execution_skips_poll_and_excludes_readback(self):
        runner, clock, calls = self.native()
        # A strided input must become a live contiguous BufferView, not bytes.
        with patch.object(bench.time, "perf_counter_ns", lambda: clock.ns):
            output, elapsed = runner.run("digit", {"image": np.arange(8, dtype=np.float32)[::2]})
        np.testing.assert_array_equal(output["slot_logits"], [0, 1, 2, 3])
        self.assertEqual(calls, ["execute", "read_output", "release_result"])
        timing = bench.execution_metrics(runner)
        self.assertEqual((timing["execute_ms"], timing["read_output_ms"], elapsed), (20, 100, 121))
        self.assertEqual(timing["runtime_execution_ms"], 7)

    def test_pending_execution_is_completed_once_before_reading(self):
        runner, clock, calls = self.native(pending=True)
        with patch.object(bench.time, "perf_counter_ns", lambda: clock.ns):
            runner.run("digit", {"image": np.arange(8, dtype=np.float32)[::2]})
        self.assertEqual(calls, ["execute", "get_result", "read_output", "release_result"])
        timing = bench.execution_metrics(runner)
        self.assertEqual(timing["execute_ms"], 23)
        # GetResult does not repeat Execute's timing evidence.
        self.assertEqual(timing["runtime_execution_ms"], 7)

    def test_failed_read_still_releases_result(self):
        runner, clock, calls = self.native(fail_read=True)
        with patch.object(bench.time, "perf_counter_ns", lambda: clock.ns):
            with self.assertRaisesRegex(RuntimeError, "read failed"):
                runner.run("digit", {"image": np.arange(8, dtype=np.float32)[::2]})
        self.assertEqual(calls[-1], "release_result")
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
