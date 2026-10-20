"""Prepared sensor freshness, native tensor/state ownership, and caller queue paced checks."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

STAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGE))
from tools.native_sensor_replay import Observation, latest_fresh, observations, policy, reference, replay, _copy_packet
from tools import native_sensor_paced as paced
import volvoxai as vx


class CallerFreshnessTest(unittest.TestCase):
    def test_clock_age_boundary_future_and_tie_order(self):
        records = [Observation(0, 20, {}), Observation(1, 40, {}),
                   Observation(2, 40, {}), Observation(3, 101, {})]
        self.assertIs(latest_fresh(records, now_ns=100, max_age_ns=60), records[2])
        self.assertIs(latest_fresh(records[:2], now_ns=100, max_age_ns=60), records[1])
        self.assertIsNone(latest_fresh(records, now_ns=103, max_age_ns=1))
        self.assertIsNone(latest_fresh(records[:1], now_ns=100, max_age_ns=60))

    def test_seed_reproduces_every_prepared_sensor_byte(self):
        spec = policy()
        first, second = observations(spec), observations(spec)
        self.assertEqual(len(first), 12)
        for expected, actual in zip(first, second):
            self.assertEqual(expected.source_ns, expected.index * 20_000_000)
            for name in expected.tensors:
                self.assertEqual(expected.tensors[name].tobytes(), actual.tensors[name].tobytes())
        self.assertFalse(np.array_equal(first[0].tensors["imu"], first[1].tensors["imu"]))

    def test_reference_accumulates_actual_features_and_observation_time(self):
        record = observations(policy())[3]
        baseline = reference(record)
        changed = Observation(record.index, record.source_ns,
                              {name: value.copy() for name, value in record.tensors.items()})
        changed.tensors["observation_time"] += 1
        np.testing.assert_allclose(reference(changed) - baseline, [[.125, -.25, .5]], atol=1e-7)
        state = np.array([[1, -2, .5]], dtype=np.float32)
        np.testing.assert_array_equal(reference(record, state), baseline + state)

    def test_owned_output_contract_rejects_inline_or_invalid_buffer_before_copy(self):
        pb = vx.pb
        values = np.array([[1, 2, 3]], dtype=np.float32)
        invalid = [
            pb.Tensor(name="state_next", shape=[1, 3], dtype=pb.DataType.DATA_TYPE_F32,
                      inline=values.tobytes()),
            pb.Tensor(name="state_next", shape=[1, 3], dtype=pb.DataType.DATA_TYPE_F32,
                      buffer=pb.BufferView(buffer_id=0, length_bytes=values.nbytes)),
            pb.Tensor(name="state_next", shape=[1, 3], dtype=pb.DataType.DATA_TYPE_F32,
                      buffer=pb.BufferView(buffer_id=7, length_bytes=values.nbytes - 4)),
        ]
        for tensor in invalid:
            with self.subTest(payload=tensor):
                buffers = Mock()
                with self.assertRaisesRegex(ValueError, "owned dense snapshot"):
                    _copy_packet(buffers, pb.ExecutionPacket(outputs=[tensor]))
                buffers.copy_tensors.assert_not_called()


class NativeReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            vx.find_library()
        except vx.VolvoxAIError as error:
            raise unittest.SkipTest(str(error))

    def test_all_feedback_latest_snapshots_and_context_close(self):
        with tempfile.TemporaryDirectory(prefix="volvox-sensor-replay-") as temporary:
            report = replay(Path(temporary))
            self.assertEqual(report["status"], "pass")
            self.assertEqual(len(report["stateful_all"]), 12)
            self.assertEqual([row["sample_index"] for row in report["stateless_latest"]], [2, 5, 8, 11])
            self.assertEqual(sum(row["superseded"] + row["completed_before_replacement"]
                                 for row in report["stateless_latest"]), 8)
            self.assertEqual(report["caller_stale_drops"], 4)
            self.assertFalse(report["physical_robot"])
            self.assertFalse(report["timing"])


class CallerPacedPolicyTest(unittest.TestCase):
    def record(self, index):
        return paced.capture.Observation(index, index * 20, {})

    def test_freshness_exact_age_boundary_and_future(self):
        self.assertIsNone(paced.freshness_decision(40, 100, 60))
        self.assertEqual(paced.freshness_decision(39, 100, 60), "caller_stale_drop")
        self.assertEqual(paced.freshness_decision(101, 100, 60), "caller_future_drop")

    def test_backlog_includes_processing_and_preserves_fifo_on_overflow(self):
        queue = paced.CallerQueue(2, 60)
        queue.emit(self.record(0), 100, 100)
        active = queue.pop(100, latest=False)
        self.assertEqual(active.record.index, 0)
        queue.emit(self.record(1), 120, 120)
        queue.emit(self.record(2), 130, 130)
        self.assertEqual(queue.rows[-1]["decision"], "caller_backlog_drop")
        queue.finish(135)
        self.assertEqual(queue.pop(135, latest=False).record.index, 1)
        self.assertEqual(max(row["total"] for row in queue.backlog), 2)

    def test_latest_filters_stale_future_and_resolves_timestamp_ties(self):
        queue = paced.CallerQueue(5, 60)
        for index, source in ((0, 39), (1, 40), (2, 80), (3, 80), (4, 101)):
            queue.emit(self.record(index), source, 100)
        selected = queue.pop(100, latest=True)
        self.assertEqual(selected.record.index, 3)
        self.assertEqual([row["decision"] for row in queue.rows],
                         ["caller_stale_drop", "caller_coalesced_drop", "caller_coalesced_drop",
                          "pending", "caller_future_drop"])
        selected.row.update(admitted=True, decision="completed")
        queue.finish(100)
        account = paced.accounting(queue.rows, 5)
        self.assertEqual((account["accepted"], account["caller_dropped"]), (1, 4))

    def test_cancellation_and_accepted_terminal_accounting_are_exhaustive(self):
        queue = paced.CallerQueue(3, 60)
        queue.emit(self.record(0), 100, 100)
        queue.emit(self.record(1), 100, 100)
        accepted = queue.pop(100, latest=False)
        accepted.row.update(admitted=True, decision="cancelled")
        queue.finish(100)
        queue.cancel_pending(101)
        account = paced.accounting(queue.rows, 4)
        self.assertEqual(account["not_released_after_cancel"], 2)
        self.assertEqual(account["accepted"] + account["caller_dropped"], 2)
        with self.assertRaises(AssertionError):
            paced.accounting([{**accepted.row, "admitted": False}], 1)
        with self.assertRaises(AssertionError):
            paced.accounting([accepted.row, accepted.row], 4)

    def test_linear_percentiles_and_no_samples_do_not_invent_zero_latency(self):
        actual = paced.percentiles([0, 10, 20, 30])
        np.testing.assert_allclose([actual["p50"], actual["p95"], actual["p99"]], [15, 28.5, 29.7])
        self.assertIsNone(paced.percentiles([])["p99"])
        self.assertEqual(paced.percentiles([])["count"], 0)

    def test_prospective_policy_rejects_inconsistent_rate_or_duration(self):
        original = paced.measurement_policy()
        for field, value in (("samples", 149), ("input_rate_hz", 51), ("caller_queue_limit", 0)):
            with tempfile.TemporaryDirectory(prefix="sensor-policy-") as temporary:
                path = Path(temporary) / "policy.json"
                path.write_text(json.dumps({**original, field: value}))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    paced.measurement_policy(path)

    def test_poll_preserves_busy_and_terminal_evidence_but_propagates_real_rpc_failures(self):
        pb = paced.pb
        ref = pb.RequestRef(request_id=7)
        cases = [(pb.RequestState.REQUEST_STATE_RUNNING, pb.NativeStatus.NATIVE_STATUS_BUSY,
                  pb.OperationCode.OPERATION_CODE_BUSY),
                 (pb.RequestState.REQUEST_STATE_QUEUED, pb.NativeStatus.NATIVE_STATUS_BUSY,
                  pb.OperationCode.OPERATION_CODE_BUSY),
                 (pb.RequestState.REQUEST_STATE_CANCELLED, pb.NativeStatus.NATIVE_STATUS_CANCELLED,
                  pb.OperationCode.OPERATION_CODE_CANCELLED),
                 (pb.RequestState.REQUEST_STATE_FAILED, pb.NativeStatus.NATIVE_STATUS_INTERNAL,
                  pb.OperationCode.OPERATION_CODE_INTERNAL)]
        for state, status, code in cases:
            with self.subTest(state=state):
                report = pb.OperationReport(status=status, code=code)
                info = pb.RequestInfo(request_id=7, state=state, status=status, report=report)
                error = paced.vx.VolvoxAIError("typed request status", status=status, report=report, response=info)
                scheduler = Mock(poll_request=Mock(side_effect=error))
                self.assertIs(paced.poll_request(scheduler, ref), info)
                info.request_id = 8
                with self.assertRaises(paced.vx.VolvoxAIError):
                    paced.poll_request(scheduler, ref)
        report = pb.OperationReport(status=pb.NativeStatus.NATIVE_STATUS_INTERNAL)
        info = pb.RequestInfo(request_id=7, state=pb.RequestState.REQUEST_STATE_UNSPECIFIED,
                              status=report.status, report=report)
        scheduler = Mock(poll_request=Mock(side_effect=paced.vx.VolvoxAIError(
            "invalid observer", report=report, response=info)))
        with self.assertRaises(paced.vx.VolvoxAIError):
            paced.poll_request(scheduler, ref)


@unittest.skipUnless(os.environ.get("VOLVOXAI_SENSOR_ARTIFACTS"),
    "set VOLVOXAI_SENSOR_ARTIFACTS and VOLVOXAI_LIBRARY for paced native qualification")
class NativePacedLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Opt-in qualification uses an exact accepted-artifact manifest. No hardware skips.
        cls.artifacts = Path(os.environ["VOLVOXAI_SENSOR_ARTIFACTS"])
        cls.library = Path(os.environ["VOLVOXAI_LIBRARY"])

    def run_short(self, mode):
        with tempfile.TemporaryDirectory(prefix="volvox-sensor-paced-") as temporary:
            directory = Path(temporary)
            policy = {**paced.measurement_policy(), "samples": 4, "duration_ns": 80_000_000,
                      "claim": "short lifecycle regression, not sustained-rate qualification"}
            policy_path = directory / "short-policy.json"
            policy_path.write_text(json.dumps(policy, indent=2) + "\n")
            report = paced.run_measurement(directory / "run", library=self.library, artifacts=self.artifacts,
                                           policy_path=policy_path, mode=mode)
            self.assertEqual(report["status"], "observed")
            self.assertTrue(report["accounting"]["balanced"])
            self.assertEqual(report["accounting"]["planned"], 4)
            self.assertEqual(report["accounting"]["accepted"],
                             report["correctness"]["deferred_numpy_comparisons"])
            self.assertGreater(report["accounting"]["accepted"], 0)
            self.assertTrue(report["cleanup"]["balanced"])
            self.assertTrue(report["cleanup"]["producer_joined"])
            self.assertTrue(report["correctness"]["owned_snapshot_checked_after_context_release"])
            self.assertFalse(report["comparison_performance_pass"])
            self.assertLessEqual(report["backlog"]["maximum_queued_plus_processing"], 4)
            final = report["memory"]["engine_inventory_samples"][-2]
            self.assertEqual(final["unconsumed_result_bytes"], 0)
            self.assertEqual(report["latency_ns"]["enqueue_to_completion_ns"]["count"],
                             report["accounting"]["accepted"])
            # Existing results are never overwritten on a second invocation.
            before = (directory / "run/result.json").read_bytes()
            with self.assertRaises(FileExistsError):
                paced.run_measurement(directory / "run", library=self.library, artifacts=self.artifacts,
                                       policy_path=policy_path, mode=mode)
            self.assertEqual((directory / "run/result.json").read_bytes(), before)

    def test_stateful_all_initializes_fresh_and_balances_owned_cleanup(self):
        self.run_short("stateful-all")

    def test_stateless_latest_uses_complete_fresh_contexts_and_balances_cleanup(self):
        self.run_short("stateless-latest")


if __name__ == "__main__":
    unittest.main()
