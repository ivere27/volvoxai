"""Static quantization through the unified generated plan lifecycle."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import volvoxai as vx
import ptq_fixture as fixture
from execution_fixture import stateless_graph

pb = vx.pb


def library_available():
    try:
        vx.find_library()
        return True
    except vx.VolvoxAIError:
        return False


def config(dtype=pb.DataType.DATA_TYPE_I8, scheme=pb.AffineScheme.AFFINE_SCHEME_SYMMETRIC, rules=()):
    return pb.QuantizationConfig(weights=pb.WeightQuantization(format=pb.WeightFormat(bits=8)), rules=rules, static_activations=pb.StaticActivations(dtype=dtype, scheme=scheme))


@unittest.skipUnless(library_available(), "libvolvoxai.so is not built")
class QuantizationThroughSchemaTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="volvoxai-quantization-e2e-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.graph = self.directory / "graph.json"
        self.weights = self.directory / "model.safetensors"
        self.graph.write_text(json.dumps(stateless_graph(fixture.document())))
        save_file({name: np.ascontiguousarray(value) for name, value in fixture.tensors().items()}, self.weights)
        self.host = vx.open_library()
        self.addCleanup(self.host.close)
        self.q = vx.VxQuantizationServiceClient(self.host)
        self.values = np.linspace(-2, 2, 32, dtype=np.float32).reshape(1, 4, 8)

    def create(self, *, inline=False, profiles=(), policy=None):
        source = (dict(bytes=pb.ModelBytes(graph=self.graph.read_bytes(), weight_shards=[self.weights.read_bytes()])) if inline
                  else dict(files=pb.ModelFiles(graph_path=str(self.graph), weight_paths=[str(self.weights)])))
        plan = self.q.create_quantization_plan(pb.CreateQuantizationPlanRequest(**source, config=policy or config(), calibration=pb.CalibrationOptions(profile_names=profiles)))
        self.addCleanup(self.q.release_quantization_plan,
                        pb.QuantizationPlanRef(quantization_plan_id=plan.quantization_plan_id))
        return plan

    def calibrate(self, plan, *, values=None, profile="", sample="sample"):
        values = self.values if values is None else values
        return self.q.calibrate_quantization_plan(pb.CalibrateQuantizationPlanRequest(
            quantization_plan_id=plan.quantization_plan_id, profile_name=profile,
            sample_name=sample, sample_count=1, inputs=[pb.Tensor(name="hidden", shape=values.shape,
                dtype=pb.DataType.DATA_TYPE_F32, inline=values.tobytes())]))

    def export(self, plan):
        return self.q.export_quantized_model(pb.ExportQuantizedModelRequest(
            quantization_plan_id=plan.quantization_plan_id))

    def inspect(self, plan):
        return self.q.inspect_quantization_plan(pb.QuantizationPlanRef(
            quantization_plan_id=plan.quantization_plan_id)).info

    def test_prepare_owns_source_and_exposes_readiness_without_model_handles(self):
        plan = self.create()
        self.assertGreater(plan.quantization_plan_id, 0)
        self.assertTrue(plan.info.requires_calibration)
        self.assertFalse(plan.info.ready)
        self.assertTrue(plan.info.HasField("calibration"))
        self.assertEqual(sum(count.nodes for count in plan.info.quantized), 3)
        self.assertEqual(plan.info.float_nodes, 0)
        self.assertEqual([spec.name for spec in plan.info.calibration_inputs], ["hidden"])
        self.graph.unlink()
        self.weights.unlink()
        with self.assertRaises(vx.VolvoxAIError):
            self.export(plan)
        self.calibrate(plan)
        self.assertTrue(self.inspect(plan).ready)
        self.assertTrue(self.export(plan).bytes.graph)

    def test_export_has_integer_regions_and_unchanged_float_boundary(self):
        plan = self.create()
        self.calibrate(plan)
        package = self.export(plan)
        graph = json.loads(package.bytes.graph)
        operators = {node["opType"] for node in graph["nodes"]}
        self.assertTrue({"QLinear", "QuantizeLinear", "DequantizeLinear"} <= operators)
        self.assertNotIn("Linear", operators)
        self.assertEqual(graph["inputs"], fixture.document()["inputs"])
        self.assertEqual(graph["outputs"], fixture.document()["outputs"])
        observed = {tensor.tensor_name for tensor in package.plan.calibration.affines}
        self.assertEqual(observed, {"activated", "contracted", "expanded", "hidden", "normed", "output", "residual"})

    def test_explicit_activation_policy_and_float_selection_reach_authoring(self):
        plan = self.create(policy=config(pb.DataType.DATA_TYPE_U8, pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC, rules=[pb.QuantizationRule(operator_name=name, keep_float=pb.Empty()) for name in ["GELU"]]))
        self.assertEqual(sum(count.nodes for count in plan.info.quantized), 3)
        self.assertEqual(plan.info.float_nodes, 1)
        self.calibrate(plan)
        package = self.export(plan)
        graph = json.loads(package.bytes.graph)
        self.assertIn("GELU", {node["opType"] for node in graph["nodes"]})
        self.assertTrue(all(t.dtype == pb.DataType.DATA_TYPE_U8 for t in package.plan.calibration.affines))

    def test_inline_and_path_sources_produce_same_numerical_package(self):
        left, right = self.create(), self.create(inline=True)
        self.calibrate(left)
        self.calibrate(right)
        a, b = self.export(left), self.export(right)
        self.assertEqual(a.bytes.graph, b.bytes.graph)
        from safetensors.numpy import load
        a_weights, b_weights = load(a.bytes.weight_shards[0]), load(b.bytes.weight_shards[0])
        self.assertEqual(a_weights.keys(), b_weights.keys())
        for name in a_weights:
            np.testing.assert_array_equal(a_weights[name], b_weights[name])

    def test_profiles_require_successful_batches_and_failed_calibration_is_atomic(self):
        plan = self.create(profiles=["small", "large"])
        first = self.calibrate(plan, profile="small")
        self.assertFalse(first.ready)
        with self.assertRaises(vx.VolvoxAIError):
            self.export(plan)
        before = self.inspect(plan).calibration
        invalid = self.values.copy()
        invalid.flat[3] = np.nan
        with self.assertRaises(vx.VolvoxAIError):
            self.calibrate(plan, profile="large", values=invalid, sample="invalid")
        after = self.inspect(plan).calibration
        self.assertEqual(before, after)
        result = self.calibrate(plan, profile="large", sample="valid")
        self.assertEqual(result.batches, 2)
        self.assertTrue(result.ready)
        self.assertTrue(self.export(plan).plan.ready)

    def test_exported_package_runs_after_plan_release(self):
        plan = self.create()
        self.calibrate(plan)
        output = self.directory / "quantized"
        output.mkdir()
        result = self.q.export_quantized_model(pb.ExportQuantizedModelRequest(quantization_plan_id=plan.quantization_plan_id, output_files=pb.ModelFiles(graph_path=str(output / "graph.json"), weight_paths=[str(output / "model.safetensors")])))
        self.assertFalse(result.bytes.graph)
        ref = pb.QuantizationPlanRef(quantization_plan_id=plan.quantization_plan_id)
        self.q.release_quantization_plan(ref)
        self.q.release_quantization_plan(ref)
        with self.assertRaises(vx.VolvoxAIError):
            self.q.inspect_quantization_plan(ref).info
        with vx.InferenceSession(output) as session:
            values = session.run(self.values * .5)
        self.assertEqual(list(values), fixture.document()["outputs"])
        self.assertTrue(all(np.isfinite(value).all() for value in values.values()))

    def test_malformed_sources_and_static_i4_dense_selection_fail_before_handle(self):
        for source, weights in ((b"{broken", self.weights.read_bytes()),
                                (self.graph.read_bytes(), b"invalid")):
            with self.subTest(source=source[:12]), self.assertRaises(vx.VolvoxAIError):
                self.q.create_quantization_plan(pb.CreateQuantizationPlanRequest(config=config(), bytes=pb.ModelBytes(graph=source, weight_shards=[weights])))
        policy = config()
        policy.weights.format.bits = 4
        with self.assertRaises(vx.VolvoxAIError) as error:
            self.create(policy=policy)
        # Per-channel static I4 is valid for Conv2D. This source contains dense
        # weights, so refusal comes from source-node policy resolution.
        self.assertEqual(error.exception.report.code, pb.OperationCode.OPERATION_CODE_INVALID_PTQ_PLAN)
        self.assertIn("unsupported weight/activation combination", error.exception.report.message)


if __name__ == "__main__":
    unittest.main()
