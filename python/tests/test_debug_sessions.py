"""Preserved-node Python debugging and portable captures over the real engine."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import volvoxai as vx
from volvoxai._diagnostics import _proto_json

pb = vx.pb


def graph(size=4):
    return {"format": "volvox-graph/v1", "dimensions": {},
            "inputs": {"x": {"dtype": "float32", "shape": [size]}}, "nodes": [
                {"id": "view", "opType": "Reshape", "inputs": {"input": "x"},
                 "outputs": {"out": {"tensor": "h", "dtype": "float32", "shape": [size]}},
                 "params": {"shape": [size]}},
                {"id": "double", "opType": "Add", "inputs": {"a": "h", "b": "h"},
                 "outputs": {"out": {"tensor": "y", "dtype": "float32", "shape": [size]}},
                 "params": {}}], "outputs": ["y"]}


def library_available():
    try:
        vx.find_library()
        return True
    except vx.VolvoxAIError:
        return False


@unittest.skipUnless(library_available(), "libvolvoxai is not built")
class DebugSessionTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="volvoxai-debug-session-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "graph.json"
        self.path.write_text(json.dumps(graph()), encoding="utf-8")
        self.x = np.array([-1, 2, -3, 4], dtype=np.float32)

    def test_preserved_debugging_keeps_the_ordinary_model_and_releases_its_model(self):
        with vx.InferenceSession(self.path, cpu_threads=1) as session:
            compiled_id = session.compiled_model_id
            before = session.run(self.x)["y"]
            resource_count = len(session._resources)
            with session.debug(self.x) as debug:
                self.assertTrue(debug.info.capabilities.preserves_node_boundaries)
                self.assertFalse(debug.plan.steps[0].skipped)
                debug.continue_()
                after = [event for event in debug.events()
                         if event.point == pb.DebugPoint.DEBUG_POINT_AFTER]
                view = after[0].snapshots[0]
                np.testing.assert_array_equal(debug.tensor(view), self.x)
                self.assertEqual(session.compiled_model_id, compiled_id)
                capture = debug.collect()
                debug_ref = pb.DebugSessionRef(debug_session_id=debug.info.debug_session_id)
            self.assertEqual(len(session._resources), resource_count)
            debug.close()
            with self.assertRaises(vx.VolvoxAIError):
                vx.VxDebugServiceClient(session._host).get_debug_session(debug_ref)
            np.testing.assert_array_equal(session.run(self.x)["y"], before)
        self.assertEqual(capture["format"], "volvoxai-debug/v1")
        self.assertEqual(capture["info"]["state"], "DEBUG_STATE_COMPLETED")
        json.dumps(capture, allow_nan=False)

    def test_optimized_schedule_and_metadata_only_capture_remain_available(self):
        with vx.InferenceSession(self.path) as session:
            with session.debug(self.x, preserve_node_boundaries=False) as debug:
                self.assertFalse(debug.info.capabilities.preserves_node_boundaries)
                self.assertTrue(debug.plan.steps[0].skipped)
                debug.continue_()
                artifact = debug.collect(values=False)
                snapshots = [snapshot for event in artifact["events"] for snapshot in event["snapshots"]]
                self.assertTrue(snapshots)
                self.assertTrue(all("valuesBase64" not in snapshot for snapshot in snapshots))
            np.testing.assert_array_equal(session.run(self.x)["y"], self.x * 2)

    def test_capture_does_not_run_a_paused_session(self):
        with vx.InferenceSession(self.path) as session, session.debug(self.x) as debug:
            artifact = debug.collect()
            self.assertEqual(artifact["events"], [])
            self.assertEqual(debug.state, pb.DebugState.DEBUG_STATE_PAUSED)
            self.assertEqual(debug.next_step, 0)

    def test_refused_creation_releases_the_separate_compiled_model(self):
        with vx.InferenceSession(self.path) as session:
            resource_count = len(session._resources)
            with self.assertRaises(vx.VolvoxAIError):
                session.debug(np.ones(2, dtype=np.float32))
            self.assertEqual(len(session._resources), resource_count)
            np.testing.assert_array_equal(session.run(self.x)["y"], self.x * 2)

    def test_preserved_compilation_keeps_required_backend_admission(self):
        with vx.InferenceSession(self.path) as session:
            # A required route must still be required by the extra compilation.
            session._backend_policy = pb.BackendPolicy(
                mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                backends=["missing-device"],
                operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)
            with self.assertRaises(vx.VolvoxAIError):
                session.debug(self.x)
            np.testing.assert_array_equal(session.run(self.x)["y"], self.x * 2)

    def test_parent_close_retires_open_debugging_resources(self):
        session = vx.InferenceSession(self.path)
        debug = session.debug(self.x)
        debug.continue_()
        artifact = debug.collect()
        session.close()
        debug.close()
        self.assertEqual(session._resources, [])
        self.assertTrue(artifact["events"])
        json.dumps(artifact, allow_nan=False)

    def test_capture_reads_more_than_one_tensor_chunk(self):
        self.path.write_text(json.dumps(graph(300001)), encoding="utf-8")
        data = np.arange(300001, dtype=np.float32)
        with vx.InferenceSession(self.path) as session, session.debug(data, max_bytes=16 << 20) as debug:
            debug.continue_()
            artifact = debug.collect()
        snapshot = next(snapshot for event in artifact["events"] for snapshot in event["snapshots"]
                        if "valuesBase64" in snapshot)
        stored = base64.b64decode(snapshot["valuesBase64"])
        self.assertGreater(len(stored), 1 << 20)
        np.testing.assert_array_equal(np.frombuffer(stored, dtype="<f4"), data)

    def test_capture_refuses_a_tensor_read_that_does_not_advance(self):
        with vx.InferenceSession(self.path) as session, session.debug(self.x) as debug:
            debug.continue_()
            chunk = pb.ReadDebugTensorResponse(status=pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_AVAILABLE,
                                              size_bytes=16, data=b"")
            with patch.object(debug._client, "read_debug_tensor", return_value=chunk):
                with self.assertRaisesRegex(ValueError, "did not advance"):
                    debug.collect()

    @unittest.skipUnless(shutil.which("node"), "Node is needed for JS/Python capture parity")
    def test_capture_matches_generated_javascript_protobuf_json(self):
        with vx.InferenceSession(self.path) as session, session.debug(self.x) as debug:
            debug.continue_()
            artifact = debug.collect()
            messages = [("DebugSessionInfo", debug.info), ("ExecutionPlan", debug.plan)]
            messages += [("DebugEvent", event) for event in debug.events()]
            payload = [[name, base64.b64encode(message.to_bytes()).decode("ascii")]
                       for name, message in messages]
            script = """
                import fs from 'node:fs';
                import * as p from './runtime/generated/typescript/volvoxai_lite.ts';
                const messages = JSON.parse(fs.readFileSync(0, 'utf8'));
                console.log(JSON.stringify(messages.map(([name, data]) =>
                  p[name].fromBinary(Buffer.from(data, 'base64')).toJson())));
            """
            result = subprocess.run([shutil.which("node"), "--import", "tsx", "--input-type=module", "-e", script],
                                    cwd=ROOT, input=json.dumps(payload), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            javascript = json.loads(result.stdout)
            self.assertEqual(artifact["info"], javascript[0])
            self.assertEqual(artifact["plan"], javascript[1])
            for captured, expected in zip(artifact["events"], javascript[2:]):
                for snapshot in captured["snapshots"]:
                    snapshot.pop("valuesBase64", None)
                self.assertEqual(captured, expected)

    @unittest.skipUnless(shutil.which("node"), "Node is needed for JS/Python report parity")
    def test_real_python_captures_produce_the_same_javascript_comparison(self):
        with vx.InferenceSession(self.path) as session:
            captures = []
            for data in (self.x, self.x + np.array([0, 0, 0, .5], dtype=np.float32)):
                with session.debug(data) as debug:
                    debug.continue_()
                    captures.append(debug.collect())
        report = vx.debug_compare(*captures)
        self.assertTrue(report["comparable"])
        self.assertEqual(report["firstMismatch"]["maxAbsError"], .5)
        script = """
            import fs from 'node:fs';
            import {compareDebugSessions} from './tools/debug_report.mjs';
            const [a, b] = JSON.parse(fs.readFileSync(0, 'utf8'));
            console.log(JSON.stringify(compareDebugSessions(a, b)));
        """
        result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", script, "python-capture-parity"],
                                cwd=ROOT, input=json.dumps(captures, allow_nan=False), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

        def compare(actual, expected):
            if isinstance(expected, dict):
                self.assertEqual(set(actual), set(expected))
                for name, value in expected.items():
                    compare(actual[name], value)
            elif isinstance(expected, list):
                self.assertEqual(len(actual), len(expected))
                for left, right in zip(actual, expected):
                    compare(left, right)
            elif isinstance(expected, float):
                self.assertAlmostEqual(actual, expected, places=12)
            else:
                self.assertEqual(actual, expected)

        compare(report, json.loads(result.stdout))


class ProtobufCaptureTest(unittest.TestCase):
    def test_optional_zero_and_nonfinite_statistics_have_standard_json_values(self):
        statistics = pb.DebugTensorStatistics(minimum=0, maximum=float("inf"), mean=float("nan"))
        snapshot = pb.DebugTensorSnapshot(tensor_id=0, statistics=statistics)
        value = _proto_json(snapshot)
        self.assertNotIn("tensorId", value)
        self.assertEqual(value["statistics"]["minimum"], 0)
        self.assertEqual(value["statistics"]["maximum"], "Infinity")
        self.assertEqual(value["statistics"]["mean"], "NaN")
        json.dumps(value, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
