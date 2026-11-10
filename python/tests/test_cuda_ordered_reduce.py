"""Opt-in physical CUDA qualification of ordered ReduceSum/Mean and replay.

Run with VOLVOXAI_TEST_CUDA=1 and VOLVOXAI_LIBRARY selecting the built library.
VOLVOXAI_CUDA_RECEIPT_DIR optionally retains the public-route observations.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import volvoxai as vx


@unittest.skipUnless(os.environ.get("VOLVOXAI_TEST_CUDA") == "1", "requires physical CUDA")
class CudaOrderedReduceTests(unittest.TestCase):
    def test_shape_boundaries_changed_inputs_and_nonfinite_values(self):
        receipts = []
        shapes = ((127, 128), (128, 127), (128, 128), (128, 16384),
                  (128, 16385), (1024, 16384), (1025, 128))
        maximum = np.finfo(np.float32).max
        pattern = np.array([maximum, -maximum, 2.0**24, 1,
                            -(2.0**24), 1, -0.0, 0.0], np.float32)
        for operator in ("ReduceSum", "ReduceMean"):
            graph = {"format": "volvox-graph/v1",
                "dimensions": {"R": {"min": 1, "max": 1025},
                               "W": {"min": 1, "max": 16385}},
                "inputs": {"x": {"shape": ["R", "W"], "dtype": "float32"}},
                "outputs": ["y"], "nodes": [{"id": "reduce", "opType": operator,
                    "inputs": {"input": "x"}, "params": {"axis": -1, "keepdims": True},
                    "outputs": {"out": {"tensor": "y", "shape": ["R", 1], "dtype": "float32"}}}],
                "execution": {"stateless": {}, "entrypoints": [{"name": "evaluate",
                    "effect": 1, "allowed_ranges": [1], "inputs": ["x"], "outputs": ["y"]}]}}
            with tempfile.TemporaryDirectory(prefix="volvox-cuda-ordered-reduce-") as directory:
                path = Path(directory) / "model.graph.json"
                path.write_text(json.dumps(graph))
                with vx.InferenceSession(path, backend="cuda", cpu_threads=1) as session:
                    execute = session._engine.execute
                    routes = []

                    def observed(request, **kwargs):
                        response = execute(request, **kwargs)
                        route = response.report.route
                        self.assertEqual(response.report.backend, "cuda")
                        self.assertTrue(route.builtin and route.attested)
                        self.assertEqual((route.active_nodes, route.selected_nodes,
                                          route.fallback_nodes, route.missing_nodes), (1, 1, 0, 0))
                        routes.append({"active_nodes": route.active_nodes,
                            "selected_nodes": route.selected_nodes,
                            "fallback_nodes": route.fallback_nodes,
                            "missing_nodes": route.missing_nodes})
                        return response

                    session._engine.execute = observed
                    for rows, width in shapes:
                        values = np.tile(pattern, (rows, (width + 7) // 8))[:, :width].copy()
                        if (rows, width) == (128, 128):
                            values[0, 0] = np.nan
                            values[1, :2] = [np.inf, -np.inf]
                            values[2] = -0.0
                        for request in range(2):
                            with self.subTest(operator=operator, rows=rows, width=width, request=request), \
                                    np.errstate(all="ignore"):
                                # Keep the host allocation live and change its contents,
                                # so an eligible captured pass must upload current data.
                                if request:
                                    values *= np.float32(-1)
                                prefix = np.concatenate([np.zeros((rows, 1), np.float32), values], axis=1)
                                expected = np.add.accumulate(prefix, axis=1, dtype=np.float32)[:, -1:]
                                expected *= np.float32(1 / width if operator == "ReduceMean" else 1)
                                actual = session.run(values)["y"]
                                np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
                                defined = ~np.isnan(expected)
                                np.testing.assert_array_equal(actual.view(np.uint32)[defined],
                                                              expected.view(np.uint32)[defined])
                                receipts.append({"operator": operator, "rows": rows, "width": width,
                                    "request": request, "status": "PASS", "route": routes[-1],
                                    "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest()})
        destination = os.environ.get("VOLVOXAI_CUDA_RECEIPT_DIR")
        if destination:
            path = Path(destination) / "ordered-reduce-public.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"status": "PASS", "requests": receipts}, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
