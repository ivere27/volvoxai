"""Hermetic contract tests for the receipt backend benchmark driver."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

from examples.receipt_digit_reader.tools import benchmark_backends

_public_route_rows = benchmark_backends._public_route_rows
_requested_route_errors = benchmark_backends._requested_route_errors


class RequestedRouteFailureTests(unittest.TestCase):
    def test_reports_each_requested_route_failure(self):
        rows = [
            {"route": "onnxruntime", "phone": "0", "street": "999"},
            {"route": "native/cpu", "backend": "cpu", "error": "bad binary"},
            {"route": "js/wasm", "backend": "wasm", "error": "compile failed"},
        ]
        self.assertEqual(
            _requested_route_errors(rows),
            ["native/cpu: bad binary", "js/wasm: compile failed"],
        )

    def test_successful_rows_have_no_requested_route_failure(self):
        rows = [
            {"route": "native/cpu", "phone": "0", "street": "999"},
            {"route": "js/wasm", "phone": "0", "street": "999"},
        ]
        self.assertEqual(_requested_route_errors(rows), [])

    def test_public_rows_keep_agreement_without_receipt_values_or_stderr(self):
        rows = [
            {
                "route": "native/cpu", "phone": "private phone",
                "street": "private street", "median_ms": 1.0,
            },
            {
                "route": "js/wasm", "phone": "private phone",
                "street": "private street", "median_ms": 2.0,
            },
            {"route": "native/cuda", "error": "/private/path failed"},
        ]
        public = _public_route_rows(rows)
        self.assertTrue(public[0]["decoded_record_matches_reference"])
        self.assertTrue(public[1]["decoded_record_matches_reference"])
        self.assertEqual(public[2], {"route": "native/cuda", "status": "error"})
        serialized = repr(public)
        self.assertNotIn("private phone", serialized)
        self.assertNotIn("private street", serialized)
        self.assertNotIn("/private/path", serialized)

    def test_private_raw_input_is_removed_when_a_node_route_raises(self):
        staged_paths = []

        def fail_after_reading_raw(_package, raw_path, *_args):
            staged_paths.append(raw_path)
            self.assertTrue(raw_path.is_file())
            self.assertTrue(raw_path.parent.name.startswith(
                "volvoxai-receipt-benchmark-"
            ))
            raise RuntimeError("route failed after reading private tensor")

        with patch.object(benchmark_backends, "measure_node",
                          side_effect=fail_after_reading_raw):
            with self.assertRaisesRegex(RuntimeError, "route failed"):
                benchmark_backends._measure_node_routes(
                    Path("package"),
                    memoryview(b"private normalized receipt tensor"),
                    ["wasm"],
                    1,
                    0,
                    Path("api.js"),
                    Path("runtime.wasm"),
                    None,
                )

        self.assertEqual(len(staged_paths), 1)
        self.assertFalse(staged_paths[0].exists())
        self.assertFalse(staged_paths[0].parent.exists())


class AffinityAndArtifactTests(unittest.TestCase):
    def test_parent_affinity_is_restored_when_a_route_raises(self):
        with patch.object(
            benchmark_backends.os, "sched_getaffinity", return_value={0, 2}
        ), patch.object(benchmark_backends.os, "sched_setaffinity") as set_affinity:
            with self.assertRaisesRegex(RuntimeError, "route failed"):
                with benchmark_backends._pinned_process(2):
                    raise RuntimeError("route failed")

        self.assertEqual(
            set_affinity.call_args_list,
            [call(0, {2}), call(0, {0, 2})],
        )

    def test_main_pins_in_process_ort_and_child_routes_then_restores(self):
        affinity = {0, 2}

        def get_affinity(_pid):
            return set(affinity)

        def set_affinity(_pid, cpus):
            affinity.clear()
            affinity.update(cpus)

        def measured(route):
            self.assertEqual(affinity, {2})
            return {
                "route": route,
                "phone": "synthetic-phone",
                "street": "synthetic-street",
                "median_ms": 1.0,
            }

        with tempfile.TemporaryDirectory(prefix="receipt-affinity-test-") as temporary:
            root = Path(temporary)
            package = root / "package"
            package.mkdir()
            (package / "manifest.json").write_text(json.dumps({
                "decode": {"phone_slots": 1, "slots": 2},
                "preprocess": {"width": 1, "height": 1},
                "variant": "synthetic",
            }), encoding="utf-8")
            (package / "graph.json").write_bytes(b"graph")
            (package / "model.safetensors").write_bytes(b"weights")
            image = root / "synthetic.png"
            onnx = root / "model.onnx"
            native = root / "receipt_digit_reader"
            api = root / "volvoxai.js"
            for artifact in (image, onnx, native, api):
                artifact.write_bytes(b"fixture")
            native.chmod(0o700)

            with patch.object(
                benchmark_backends.os, "sched_getaffinity", side_effect=get_affinity
            ), patch.object(
                benchmark_backends.os, "sched_setaffinity", side_effect=set_affinity
            ), patch.object(
                benchmark_backends, "check_host_is_quiet", return_value=(True, 0.0)
            ), patch.object(
                benchmark_backends, "preprocess",
                return_value=memoryview(b"synthetic normalized tensor"),
            ), patch.object(
                benchmark_backends, "measure_onnxruntime",
                side_effect=lambda *_args: measured("onnxruntime"),
            ), patch.object(
                benchmark_backends, "measure_native",
                side_effect=lambda *_args: measured("native/cpu"),
            ), patch.object(
                benchmark_backends, "_measure_node_routes",
                side_effect=lambda *_args: [measured("js/wasm")],
            ):
                public_output = io.StringIO()
                with contextlib.redirect_stdout(public_output):
                    exit_code = benchmark_backends.main([
                        "--package", str(package),
                        "--image", str(image),
                        "--onnx", str(onnx),
                        "--native-binary", str(native),
                        "--api", str(api),
                        "--pin-cpu", "2",
                        "--repeat", "1",
                        "--warmup", "0",
                    ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(affinity, {0, 2})
        self.assertNotIn("synthetic-phone", public_output.getvalue())
        self.assertNotIn("synthetic-street", public_output.getvalue())
        self.assertNotIn(temporary, public_output.getvalue())

    def test_package_artifact_hashes_bind_payloads_without_paths(self):
        with tempfile.TemporaryDirectory(prefix="private-receipt-package-") as temporary:
            package = Path(temporary)
            payloads = {
                "manifest.json": b"manifest",
                "graph.json": b"graph",
                "model.safetensors": b"weights",
            }
            for filename, payload in payloads.items():
                (package / filename).write_bytes(payload)

            hashes = benchmark_backends._package_artifact_hashes(package)

        self.assertEqual(
            hashes,
            {
                filename: hashlib.sha256(payload).hexdigest()
                for filename, payload in payloads.items()
            },
        )
        self.assertNotIn(temporary, repr(hashes))


if __name__ == "__main__":
    unittest.main()
