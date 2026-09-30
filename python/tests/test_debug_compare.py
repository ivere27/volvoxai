"""Numerical comparison and JS/Python report parity, without a native engine."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
if not sys.flags.isolated and (ROOT / "python/volvoxai").is_dir():
    sys.path.insert(0, str(ROOT / "python"))

from volvoxai._debug_compare import debug_compare

AFTER, BEFORE = "DEBUG_POINT_AFTER", "DEBUG_POINT_BEFORE"
AVAILABLE = "DEBUG_TENSOR_STATUS_AVAILABLE"
DTYPES = {"F32": "f", "F64": "d", "F16": "e", "BF16": "H", "I64": "q", "U64": "Q",
          "I32": "i", "U32": "I", "I16": "h", "U16": "H", "I8": "b", "U8": "B", "BOOL": "B"}


def capture(values=(1, 2, 3), *, dtype="F32", name="activation", shape=None, nodes=("layer",),
            point=AFTER, quantization=None, status=AVAILABLE):
    snapshot = {"tensorId": 0, "snapshotId": "987654321", "status": status,
                "dtype": "DATA_TYPE_" + dtype, "shape": [str(n) for n in (shape or [len(values)])]}
    if dtype in DTYPES:
        snapshot["valuesBase64"] = base64.b64encode(struct.pack("<" + DTYPES[dtype] * len(values), *values)).decode()
    if quantization is not None:
        snapshot["quantization"] = quantization
    return {"format": "volvoxai-debug/v1", "plan": {
        "tensors": [{"name": name, "allocationId": "123"}],
        "steps": [{"sourceNodeIds": list(nodes), "outputs": [0]}]},
        "events": [{"step": 0, "point": point, "snapshots": [snapshot]}]}


def snapshot(artifact):
    return artifact["events"][0]["snapshots"][0]


class DebugComparisonTest(unittest.TestCase):
    def assert_js_parity(self, cases):
        """Run the actual JS comparator on the exact same artifacts/options."""
        if not shutil.which("node") or not (ROOT / "tools/debug_report.mjs").is_file():
            self.skipTest("JS parity needs Node and the repository debug_report.mjs")
        script = (f"import {{compareDebugSessions}} from {json.dumps((ROOT / 'tools/debug_report.mjs').as_uri())};"
                  "import {readFileSync} from 'node:fs';"
                  "const cases=JSON.parse(readFileSync(0,'utf8'));"
                  "process.stdout.write(JSON.stringify(cases.map(c=>compareDebugSessions(c.a,c.b,c.options))));")
        payload = []
        for a, b, options in cases:
            js_options = {"minSqnrDb" if key == "min_sqnr_db" else key: value for key, value in options.items()}
            payload.append({"a": a, "b": b, "options": js_options})
        result = subprocess.run(["node", "--input-type=module", "-e", script, "debug-comparison-parity"],
                                input=json.dumps(payload), text=True, capture_output=True, check=True)
        js_reports = json.loads(result.stdout)

        def same(actual, expected, path="report"):
            if isinstance(actual, dict):
                self.assertEqual(actual.keys(), expected.keys(), path)
                for key in actual:
                    same(actual[key], expected[key], path + "." + key)
            elif isinstance(actual, list):
                self.assertEqual(len(actual), len(expected), path)
                for index, (left, right) in enumerate(zip(actual, expected)):
                    same(left, right, f"{path}[{index}]")
            elif isinstance(actual, float) and isinstance(expected, (int, float)):
                self.assertTrue(math.isclose(actual, expected, rel_tol=2e-13, abs_tol=1e-15),
                                f"{path}: {actual} != {expected}")
            else:
                self.assertEqual(actual, expected, path)

        for (a, b, options), js in zip(cases, js_reports):
            python = debug_compare(a, b, **options)
            json.dumps(python, allow_nan=False)
            same(python, js)

    def test_exact_metrics_tolerances_and_file_inputs(self):
        a, b = capture((1, -2, 0)), capture((1.1, -1.8, .1))
        original = copy.deepcopy(a)
        report = debug_compare(a, b, atol=.11, rtol=.05)
        row = report["rows"][0]
        self.assertEqual(report["mode"], "exact")
        self.assertEqual(row["elements"], 3)
        self.assertAlmostEqual(row["maxAbsError"], .2, places=6)
        self.assertAlmostEqual(row["meanAbsError"], .4 / 3, places=6)
        self.assertAlmostEqual(row["rmse"], math.sqrt(.02), places=6)
        self.assertEqual(row["mismatches"], 0)
        self.assertIsNone(report["firstMismatch"])
        self.assertEqual(a, original, "comparison does not modify captures")
        strict = debug_compare(a, b, atol=0, rtol=0)
        self.assertEqual(strict["firstMismatch"]["mismatches"], 3)
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.json", Path(directory) / "b.json"
            first.write_text(json.dumps(a))
            second.write_text(json.dumps(b))
            self.assertEqual(debug_compare(str(first), second), debug_compare(a, b))
        self.assert_js_parity([(a, b, {"atol": .11, "rtol": .05}), (a, b, {"atol": 0, "rtol": 0}),
                               (a, a, {}), (a, b, {"mode": "quantization", "min_sqnr_db": 40})])

    def test_quantized_ptq_aliases_and_axis_dequantization(self):
        a = capture((.1, -.2, 1.3, 1.9), name="hidden", shape=[2, 2])
        digest = hashlib.sha256(b"volvox-typed-ptq/v1\0hidden\0activation").hexdigest()[:20]
        b = capture((1, -2, 6, 9), dtype="I8", name=f"__ptq__.{digest}.activation.2", shape=[2, 2],
                    quantization={"perAxis": {"axis": 0, "scales": [.1, .2], "zeroPoints": [0, -1]}})
        report = debug_compare(a, b, min_sqnr_db=40)
        row = report["rows"][0]
        self.assertTrue(report["comparable"])
        self.assertEqual(report["mode"], "quantization")
        self.assertEqual(report["aliases"]["ptqActivations"], 1)
        self.assertEqual(report["activationCoverage"], {"matched": 1, "total": 1, "ratio": 1})
        self.assertAlmostEqual(row["maxAbsError"], .1, places=6)
        self.assertAlmostEqual(row["rmseOverScale"], math.sqrt(.5 / 4), places=6)
        self.assertGreater(row["cosine"], .999)
        self.assertEqual(report["worst"][0]["tensor"], "hidden")
        self.assertTrue(row["mismatch"])
        provenance = copy.deepcopy(b)
        provenance["plan"]["tensors"][0].update(name="arbitrary-name", sourceTensorName="hidden")
        scalar = capture((1, -2, 13, 19), dtype="I8", name="hidden",
                         quantization={"perTensor": {"scale": .1}})
        self.assert_js_parity([(a, b, {"min_sqnr_db": 40}), (a, provenance, {}), (a, scalar, {}),
                               (a, b, {"mode": "exact", "atol": .01}), (a, b, {"mode": "quantization"})])

    def test_fused_boundary_matches_by_output_and_kind(self):
        a = capture(nodes=("conv", "bias"))
        b = capture(nodes=("bias",))
        fallback = capture(nodes=("renamed-producer",))
        self.assertEqual(debug_compare(a, b)["rows"][0]["node"], "bias")
        self.assertEqual(debug_compare(a, fallback)["rows"][0]["node"], "conv")
        before = capture(nodes=("renamed-producer",), point=BEFORE)
        self.assertFalse(debug_compare(capture(point=BEFORE), before)["comparable"])
        weights = capture(name="weight")
        weights["plan"]["steps"][0]["outputs"] = []
        weights["plan"]["allocations"] = [{"allocationId": "123", "role": "MEMORY_RESOURCE_ROLE_PACKED_WEIGHTS"}]
        self.assertEqual(debug_compare(weights, weights)["rows"][0]["kind"], "constant")
        external = copy.deepcopy(weights)
        external["plan"]["allocations"] = []
        self.assertEqual(debug_compare(external, external)["rows"][0]["kind"], "external")
        self.assert_js_parity([(a, b, {}), (a, fallback, {}), (capture(point=BEFORE), before, {}),
                               (weights, weights, {}), (external, external, {})])

    def test_worst_layers_are_ranked_limited_and_events_ordered(self):
        a, b = {"format": "volvoxai-debug/v1"}, {"format": "volvoxai-debug/v1"}
        for artifact in (a, b):
            artifact["plan"] = {"tensors": [{"name": f"layer-{i}"} for i in range(12)],
                                "steps": [{"sourceNodeIds": [f"node-{i}"], "outputs": [i]} for i in range(12)]}
            artifact["events"] = []
        for i in reversed(range(12)):
            for artifact, values in ((a, (1, 2)), (b, (1 + .01 * (i + 1), 2))):
                for point in (AFTER, BEFORE):
                    observation = snapshot(capture(values))
                    observation["tensorId"] = i
                    artifact["events"].append({"step": i, "point": point, "snapshots": [observation]})
        report = debug_compare(a, b, mode="quantization", min_sqnr_db=100)
        self.assertEqual(report["matched"], 24)
        self.assertEqual(report["candidateOnly"], 0)
        self.assertEqual(report["firstMismatch"]["point"], BEFORE)
        self.assertEqual([row["tensor"] for row in report["worst"]],
                         [f"layer-{i}" for i in range(11, 1, -1)])
        self.assert_js_parity([(a, b, {"mode": "quantization", "min_sqnr_db": 100})])

    def test_unmatched_unavailable_shape_and_unsupported_dtype(self):
        a = capture()
        unrelated = capture(name="other")
        missing = capture()
        del snapshot(missing)["valuesBase64"]
        unavailable = capture(status="DEBUG_TENSOR_STATUS_STATS_ONLY")
        packed = capture(dtype="QI4")
        different_length = capture((1, 2))
        cases = [(a, unrelated, {}), (a, missing, {}), (a, unavailable, {}), (a, packed, {}),
                 (a, different_length, {}), ({"format": "volvoxai-debug/v1"}, a, {})]
        for _, candidate, _ in cases[:5]:
            result = debug_compare(a, candidate)
            self.assertFalse(result["comparable"])
            self.assertIsNone(result["firstMismatch"])
        self.assertEqual(debug_compare(a, packed)["rows"][0]["comparison"], "unsupported dtype")
        self.assertEqual(debug_compare(a, different_length)["rows"][0]["comparison"], "shape differs")
        self.assertEqual(debug_compare(a, unrelated)["candidateOnly"], 1)
        # JS compares captured element counts; a reshaped view has the same values.
        reshaped = capture(shape=[1, 3])
        self.assertIsNone(debug_compare(a, reshaped)["firstMismatch"])
        self.assert_js_parity(cases + [(a, reshaped, {})])

    def test_nonfinite_statistics_and_zero_signals_are_json_safe(self):
        a = capture((math.nan, math.inf, -math.inf, 1))
        b = capture((math.nan, math.inf, -math.inf, 2))
        snapshot(a)["statistics"] = {"nanCount": "1", "positiveInfinityCount": "1", "negativeInfinityCount": "1"}
        snapshot(b)["realStatistics"] = {"nanCount": "2", "positiveInfinityCount": "1", "negativeInfinityCount": "1"}
        report = debug_compare(a, b)
        self.assertEqual(report["rows"][0]["finitePairs"], 1)
        self.assertEqual(report["firstMismatch"]["candidateNonfinite"], 4)
        only_nonfinite = capture((math.nan, math.inf))
        self.assertTrue(debug_compare(only_nonfinite, only_nonfinite)["comparable"])
        self.assertIsNone(debug_compare(only_nonfinite, only_nonfinite)["rows"][0]["rmse"])
        zero, error = capture((0, 0)), capture((1, 2))
        self.assertIsNone(debug_compare(zero, error, mode="quantization")["firstMismatch"])
        overflow_a, overflow_b = capture((1e308,), dtype="F64"), capture((.9e308,), dtype="F64")
        tiny = capture((1e-160,), dtype="F64")
        json.dumps(debug_compare(overflow_a, overflow_b), allow_nan=False)
        self.assert_js_parity([(a, b, {}), (a, b, {"mode": "quantization"}),
                               (only_nonfinite, only_nonfinite, {}), (zero, error, {"mode": "quantization"}),
                               (overflow_a, overflow_b, {}), (overflow_a, overflow_b, {"mode": "quantization"}),
                               (tiny, tiny, {})])

    def test_storage_dtypes_have_js_numeric_semantics(self):
        cases = []
        for dtype in DTYPES:
            values = (0x3f80, 0x4000) if dtype == "BF16" else (2 ** 63, 2 ** 63 + 1) if dtype == "U64" else (1, 2)
            source = capture(values, dtype=dtype)
            report = debug_compare(source, source)
            self.assertTrue(report["rows"][0]["identical"], dtype)
            cases.append((source, source, {}))
        boolean = capture((0, 8), dtype="BOOL")
        cases.append((capture((0, 1)), boolean, {}))
        absent_node = capture(nodes=())
        cases.append((absent_node, absent_node, {}))
        numeric_enums = capture(point=2, status=1)
        cases.append((numeric_enums, numeric_enums, {}))
        self.assert_js_parity(cases)

    def test_invalid_format_and_mode_are_rejected(self):
        for invalid in ({}, {"format": "other"}, []):
            with self.assertRaisesRegex(ValueError, "expected volvoxai-debug/v1"):
                debug_compare(invalid, capture())
        with self.assertRaisesRegex(ValueError, "unknown comparison mode"):
            debug_compare(capture(), capture(), mode="approximate")


if __name__ == "__main__":
    unittest.main()
