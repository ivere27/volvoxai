"""Conv PTQ must pack HWIO sources into per-output-channel OHWI bytes.

Unequal channel/kernel dimensions, groups, asymmetric padding and optional
bias catch a relabelled weight as well as an author/plan/writer mismatch.
Every engine operation goes through the generated public dispatch.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
from safetensors.numpy import load, save

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution_fixture import stateless_graph, feedback_graph

import volvoxai as vx  # noqa: E402

pb = vx.pb


def library_available():
    try:
        vx.find_library()
        return True
    except vx.VolvoxAIError:
        return False


def fixture(layout: str, bias: bool, mixed: bool = False):
    weight = (np.arange(6 * 2 * 3 * 2, dtype=np.float32) - 30).reshape(6, 2, 3, 2) / 64
    stored = weight if layout == "OHWI" else weight.transpose(1, 2, 3, 0)
    tensors = {"weight": np.ascontiguousarray(stored)}
    inputs = {"input": "image", "weight": "weight"}
    if bias:
        tensors["bias"] = np.arange(6, dtype=np.float32) / 8
        inputs["bias"] = "bias"
    graph = {
        "format": "volvox-graph/v1", "dimensions": {},
        "inputs": {"image": {"shape": [1, 5, 6, 4], "dtype": "float32"}},
        "nodes": [{
            "id": "conv", "opType": "Conv2D", "inputs": inputs,
            "outputs": {"out": {"tensor": "features", "shape": [1, 5, 3, 6], "dtype": "float32"}},
            "params": {"data_layout": "NHWC", "weight_layout": layout,
                       "groups": 2, "stride": [1, 2], "dilation": [1, 1],
                       "pads": [1, 0, 0, 2]},
        }],
        "outputs": ["features"],
    }
    if mixed:
        graph["nodes"][0]["inputs"]["input"] = "reshaped"
        graph["nodes"].insert(0, {
            "id": "view", "opType": "Reshape", "inputs": {"input": "image"},
            "outputs": {"out": {"tensor": "reshaped", "shape": [1, 5, 6, 4], "dtype": "float32"}},
            "params": {"shape": [1, 5, 6, 4]},
        })
        for index in range(4):
            graph["nodes"].append({
                "id": f"float_activation_{index}", "opType": "SiLU",
                "inputs": {"input": "features" if index == 0 else f"activated_{index - 1}"},
                "outputs": {"out": {"tensor": "activated" if index == 3 else f"activated_{index}",
                                    "shape": [1, 5, 3, 6], "dtype": "float32"}},
                "params": {},
            })
        graph["outputs"] = ["activated"]
    return json.dumps(stateless_graph(graph)).encode(), save(tensors), weight


@unittest.skipUnless(library_available(), "libvolvoxai.so is not built")
class ConvPackingTest(unittest.TestCase):
    def pipeline(self, layout: str, bias: bool, mixed: bool = False):
        graph, weights, canonical = fixture(layout, bias, mixed)
        values = ((np.arange(120, dtype=np.float32) % 17 - 8) / 8).reshape(1, 5, 6, 4)
        sample = pb.Tensor(name="image", dtype=pb.DataType.DATA_TYPE_F32,
                           shape=list(values.shape), inline=values.tobytes())
        host = vx.open_library()
        try:
            inference = vx.VxInferenceServiceClient(host)
            quantization = vx.VxQuantizationServiceClient(host)
            runtime = inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
            source = inference.load_model(pb.LoadModelRequest(
                runtime_id=runtime.runtime_id,
                package=pb.ModelPackage(graph_document=graph, weight_shards=[weights])))
            plan = quantization.create_quantization_plan(pb.CreateQuantizationPlanRequest(config=pb.QuantizationConfig(weights=pb.WeightQuantization(format=pb.WeightFormat(bits=8)), static_activations=pb.StaticActivations(dtype=pb.DataType.DATA_TYPE_I8, scheme=pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC), rules=[pb.QuantizationRule(operator_name=name, keep_float=pb.Empty()) for name in (["SiLU"] if mixed else [])]), bytes=pb.ModelBytes(graph=graph, weight_shards=[weights]), calibration=pb.CalibrationOptions(profile_names=["default"])))
            quantization.calibrate_quantization_plan(pb.CalibrateQuantizationPlanRequest(
                quantization_plan_id=plan.quantization_plan_id, profile_name="default",
                sample_name="sample", sample_count=1, inputs=[sample]))
            observed = quantization.inspect_quantization_plan(pb.QuantizationPlanRef(quantization_plan_id=plan.quantization_plan_id)).info.calibration
            packed = quantization.export_quantized_model(pb.ExportQuantizedModelRequest(quantization_plan_id=plan.quantization_plan_id))
            repeated = quantization.export_quantized_model(pb.ExportQuantizedModelRequest(quantization_plan_id=plan.quantization_plan_id))
            self.assertEqual(packed.bytes.weight_shards[0], repeated.bytes.weight_shards[0])
            self.assertEqual(packed.bytes.graph, repeated.bytes.graph)
            storage = load(packed.bytes.weight_shards[0])
            document = json.loads(packed.bytes.graph)
            available = set(document["inputs"]) | set(storage)
            for node in document["nodes"]:
                self.assertTrue(set(node["inputs"].values()) <= available, node["id"])
                available.update(port["tensor"] for port in node["outputs"].values())
            converted = next(node for node in document["nodes"] if node["id"] == "conv")
            packed_weight_name = converted["inputs"]["weight"]
            packed_bias_name = converted["inputs"].get("bias")
            affine = document["quantization"]["tensors"][packed_weight_name]
            scale = storage[affine["scale_tensor"]]
            expected_scale = (np.abs(canonical).max(axis=(1, 2, 3)).astype(np.float64) / 127).astype(np.float32)
            np.testing.assert_array_equal(scale, expected_scale)
            expected_weight = np.clip(np.rint(canonical.astype(np.float64) /
                                              scale[:, None, None, None]), -127, 127).astype(np.int8)
            np.testing.assert_array_equal(storage[packed_weight_name], expected_weight)
            if not bias:
                np.testing.assert_array_equal(storage[packed_bias_name], np.zeros(6, np.int32))

            def run(model):
                compiled = inference.compile_model(pb.CompileModelRequest(
                    model_id=model.model_id, policy=pb.BackendPolicy(
                        mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends=["cpu"])))
                context = inference.create_execution_context(pb.CreateExecutionContextRequest(
                    compiled_model_id=compiled.compiled_model_id, graph=pb.GraphExecution()))
                try:
                    result = inference.execute(pb.ExecuteRequest(context_id=context.context_id,
                        scope=pb.ExecutionScope(entrypoint="evaluate", full=pb.Empty()),
                        inputs=pb.InputBindings(values=[sample])))
                    output = next(tensor for tensor in result.outputs
                                  if tensor.name == ("activated" if mixed else "features"))
                    return np.frombuffer(output.inline, dtype=np.float32).copy().reshape(output.shape)
                finally:
                    inference.release_execution_context(pb.ExecutionContextRef(context_id=context.context_id))
                    inference.release_compiled_model(pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))

            quantized = inference.load_model(pb.LoadModelRequest(
                runtime_id=runtime.runtime_id,
                package=pb.ModelPackage(graph_document=packed.bytes.graph, weight_shards=[packed.bytes.weight_shards[0]])))
            produced, reference = run(quantized), run(source)
            self.assertEqual(produced.shape, (1, 5, 3, 6))
            self.assertTrue(np.isfinite(produced).all())
            expected_float = np.zeros((1, 5, 3, 6), dtype=np.float32)
            padded = np.pad(values, ((0, 0), (1, 0), (0, 2), (0, 0)))
            for y in range(5):
                for x in range(3):
                    for channel in range(6):
                        group = channel // 3
                        patch = padded[0, y:y + 2, x * 2:x * 2 + 3, group * 2:group * 2 + 2]
                        expected_float[0, y, x, channel] = np.sum(patch * canonical[channel])
                        if bias:
                            expected_float[0, y, x, channel] += channel / 8
            ranges = {item.tensor_name: item for item in observed.affines}
            input_range = ranges["reshaped" if mixed else "image"]
            self.assertEqual(input_range.observed_min, float(values.min()))
            self.assertEqual(input_range.observed_max, float(values.max()))
            self.assertEqual(ranges["features"].observed_min, float(expected_float.min()))
            self.assertEqual(ranges["features"].observed_max, float(expected_float.max()))
            if mixed:
                for _ in range(4):
                    expected_float = expected_float / (1 + np.exp(-expected_float))
                np.testing.assert_allclose(reference, expected_float, atol=1e-6, rtol=1e-6)
            else:
                np.testing.assert_array_equal(reference, expected_float)
            np.testing.assert_allclose(produced, reference, atol=0.08, rtol=0.02)
            return storage, produced, reference
        finally:
            host.close()

    def test_grouped_conv_with_bias(self):
        self.pipeline("HWIO", bias=True)

    def test_grouped_conv_without_bias(self):
        self.pipeline("HWIO", bias=False)

    def test_conv_between_retained_float_nodes(self):
        self.pipeline("HWIO", bias=False, mixed=True)


if __name__ == "__main__":
    unittest.main()
