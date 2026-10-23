"""Calibrated PTQ owns file/inline bytes independently of LoadModel's cap.

Exercises static and IMPORTANCE lifecycles around 64 MiB and with genuinely
consumed weights above that boundary, after the caller destroys its sources.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import volvoxai as vx

pb = vx.pb
INLINE_LIMIT = 64 * 1024 * 1024


def policy(importance=False):
    result = pb.QuantizationConfig(weights=pb.WeightQuantization(
        format=pb.WeightFormat(bits=4 if importance else 8,
                               group_size=32 if importance else 0),
        algorithm=(pb.WeightAlgorithm.WEIGHT_ALGORITHM_IMPORTANCE if importance
                   else pb.WeightAlgorithm.WEIGHT_ALGORITHM_RTN)))
    if not importance:
        result.static_activations.CopyFrom(pb.StaticActivations(
            dtype=pb.DataType.DATA_TYPE_I8,
            scheme=pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC))
    return result


def graph(width, rows):
    return json.dumps({
        "format": "volvox-graph/v1", "dimensions": {},
        "inputs": {"x": {"shape": [1, width], "dtype": "float32"}},
        "nodes": [{"id": "linear", "opType": "Linear",
                   "inputs": {"input": "x", "weight": "w"},
                   "outputs": {"out": {"tensor": "y", "shape": [1, rows],
                                       "dtype": "float32"}},
                   "params": {"weight_layout": "dout_din"}}],
        "outputs": ["y"],
        "execution": {"stateless": {}, "entrypoints": [{"name": "evaluate",
            "effect": 1, "allowed_ranges": [1], "inputs": ["x"], "outputs": ["y"]}]},
    }).encode()


class NativePtqFileSnapshotTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="volvoxai-ptq-file-snapshot-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.graph_path = self.root / "graph.json"
        self.weights_path = self.root / "model.safetensors"
        self.host = vx.open_library()
        self.addCleanup(self.host.close)
        self.quantization = vx.VxQuantizationServiceClient(self.host)

    def create(self, importance=False, *, inline=False):
        source = (dict(bytes=pb.ModelBytes(graph=self.graph_path.read_bytes(),
                      weight_shards=[self.weights_path.read_bytes()])) if inline else
                  dict(files=pb.ModelFiles(graph_path=str(self.graph_path),
                       weight_paths=[str(self.weights_path)])))
        request = pb.CreateQuantizationPlanRequest(
            **source, config=policy(importance), calibration=pb.CalibrationOptions(cpu_threads=1))
        plan = self.quantization.create_quantization_plan(request)
        if inline:
            # The owner must survive reuse/retirement of the caller's message.
            request.bytes.graph = b"not the accepted graph"
            request.bytes.weight_shards[0] = b"not the accepted weights"
            request.ClearField("bytes")
        return plan

    def export(self, plan, name):
        destination = self.root / name
        destination.mkdir()
        paths = pb.ModelFiles(graph_path=str(destination / "graph.json"),
                             weight_paths=[str(destination / "model.safetensors")])
        self.quantization.export_quantized_model(pb.ExportQuantizedModelRequest(
            quantization_plan_id=plan.quantization_plan_id, output_files=paths))
        return destination

    @staticmethod
    def digest(directory):
        return tuple(hashlib.sha256((directory / name).read_bytes()).digest()
                     for name in ("graph.json", "model.safetensors"))

    def lifecycle(self, plan, values, expected, importance, name):
        reference = pb.QuantizationPlanRef(quantization_plan_id=plan.quantization_plan_id)
        try:
            self.assertTrue(plan.info.requires_calibration)
            self.assertFalse(plan.info.ready)
            self.quantization.calibrate_quantization_plan(pb.CalibrateQuantizationPlanRequest(
                quantization_plan_id=plan.quantization_plan_id, sample_name="source-snapshot",
                sample_count=1, inputs=[pb.Tensor(name="x", dtype=pb.DataType.DATA_TYPE_F32,
                    shape=values.shape, inline=values.tobytes())]))
            state = self.quantization.inspect_quantization_plan(reference).info
            self.assertTrue(state.ready)
            self.assertEqual(state.calibration.samples, 1)
            destination = self.export(plan, name)
            first_digest = self.digest(destination)
            self.graph_path.unlink(missing_ok=True)
            self.weights_path.unlink(missing_ok=True)
            self.assertEqual(first_digest, self.digest(self.export(plan, name + "-again")))
            with vx.InferenceSession(destination, cpu_threads=1,
                    packed_arithmetic="dynamic-i8" if importance else "f32") as session:
                actual = session.run({"x": values})["y"]
            np.testing.assert_allclose(actual, expected, rtol=.006, atol=.02)
        finally:
            self.quantization.release_quantization_plan(reference)

    def test_ptq_file_and_inline_boundary_preserves_load_model_limit(self):
        values = np.ones((1, 32), np.float32)
        weights = np.full((2, 32), .125, np.float32)
        expected = np.full((1, 2), 4, np.float32)
        inference = vx.VxInferenceServiceClient(self.host)
        runtime = inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
        self.addCleanup(inference.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id))
        for delta in (-1, 0, 1):
            with self.subTest(package_bytes=INLINE_LIMIT + delta):
                save_file({"w": weights, "unused_padding": np.zeros(
                    (INLINE_LIMIT - 8192) // 4, np.float32)}, self.weights_path)
                document = graph(32, 2)
                required = INLINE_LIMIT + delta - self.weights_path.stat().st_size
                self.assertGreaterEqual(required, len(document))
                self.graph_path.write_bytes(document + b" " * (required - len(document)))
                self.assertEqual(self.graph_path.stat().st_size + self.weights_path.stat().st_size,
                                 INLINE_LIMIT + delta)
                request = pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                    package=pb.ModelPackage(graph_document=self.graph_path.read_bytes(),
                                            weight_shards=[self.weights_path.read_bytes()]))
                if delta > 0:
                    with self.assertRaises(vx.VolvoxAIError) as failure:
                        inference.load_model(request)
                    self.assertEqual(failure.exception.report.code,
                                     pb.OperationCode.OPERATION_CODE_PACKAGE_TOO_LARGE)
                else:
                    model = inference.load_model(request)
                    inference.release_model(pb.ModelRef(model_id=model.model_id))
                del request
                accepted_graph = self.graph_path.read_bytes()
                accepted_weights = self.weights_path.read_bytes()
                for inline in (False, True):
                    for importance in (False, True):
                        self.graph_path.write_bytes(accepted_graph)
                        self.weights_path.write_bytes(accepted_weights)
                        plan = self.create(importance, inline=inline)
                        # Both native source forms retain the accepted bytes even
                        # when the caller replaces/deletes every source file.
                        self.graph_path.write_text("not the accepted graph")
                        self.weights_path.write_bytes(b"not the accepted weights")
                        self.lifecycle(plan, values, expected, importance,
                            f"boundary-{delta}-{inline}-{importance}")

    def test_consumed_weights_over_72mib_static_and_importance(self):
        width, rows = 4096, 4609
        values = np.full((1, width), .25, np.float32)
        values[:, ::2] = 1
        weights = np.full((rows, width), .125, np.float32)
        weights[1::2] *= -1
        self.assertGreater(weights.nbytes, 72 * 1024 * 1024)
        expected = np.full((1, rows), 320, np.float32)
        expected[:, 1::2] *= -1
        self.graph_path.write_bytes(graph(width, rows))
        save_file({"w": weights}, self.weights_path)
        with vx.InferenceSession(self.graph_path, weights=[self.weights_path], cpu_threads=1) as session:
            np.testing.assert_array_equal(session.run({"x": values})["y"], expected)
        for inline in (False, True):
            for importance in (False, True):
                with self.subTest(inline=inline, importance=importance):
                    self.graph_path.write_bytes(graph(width, rows))
                    save_file({"w": weights}, self.weights_path)
                    plan = self.create(importance, inline=inline)
                    self.graph_path.unlink()
                    self.weights_path.unlink()
                    self.lifecycle(plan, values, expected, importance,
                                   f"consumed-{inline}-{importance}")


if __name__ == "__main__":
    unittest.main()
