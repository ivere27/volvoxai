"""C PTQ packs either Linear source layout into per-output-channel rows."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from safetensors.numpy import load, save

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import volvoxai as vx  # noqa: E402

pb = vx.pb


class LinearPackingTest(unittest.TestCase):
    def pipeline(self, layout, with_bias):
        # Non-square and non-symmetric: relabelling without transposing fails.
        canonical = np.asarray([[.25, -.75, 1.5], [2, .5, -1]], np.float32)
        weights = {"weight": np.ascontiguousarray(canonical.T if layout == "din_dout" else canonical)}
        weights["bank"] = np.asarray([[0, 0], [.125, -.25]], np.float32)
        ports = {"input": "input", "weight": "weight"}
        bias = np.asarray([.125, -.25], np.float32) if with_bias else np.zeros(2, np.float32)
        if with_bias:
            weights["bias"], ports["bias"] = bias, "bias"
        dimensions = {"S": {"min": 1, "max": 4}, "adapters": {"min": 2, "max": 2},
                      "dense_source": {"min": weights["weight"].shape[0], "max": weights["weight"].shape[0]}}
        graph = json.dumps({"format": "volvox-graph/v1", "dimensions": dimensions,
            "banks": {"bank": "adapters", "weight": "dense_source"},
            "inputs": {"input": {"shape": [1, "S", 3], "dtype": "float32"},
                       "family": {"shape": [1], "dtype": "int32"}},
            "nodes": [{"id": "dense", "opType": "Linear", "inputs": ports,
                       "outputs": {"out": {"tensor": "linear", "shape": [1, "S", 2], "dtype": "float32"}},
                       "params": {"weight_layout": layout}},
                      {"id": "select", "opType": "Embedding", "inputs": {"input": "family", "weight": "bank"},
                       "outputs": {"out": {"tensor": "selected", "shape": [1, 2], "dtype": "float32"}}, "params": {}},
                      {"id": "combine", "opType": "Add", "inputs": {"a": "linear", "b": "selected"},
                       "outputs": {"out": {"tensor": "output", "shape": [1, "S", 2], "dtype": "float32"}}, "params": {}}],
            "outputs": ["output"]}).encode()
        shards = [save(weights)]
        host = vx.open_library()
        try:
            inference, quantization = vx.VxInferenceServiceClient(host), vx.VxQuantizationServiceClient(host)
            runtime = inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
            model = inference.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                package=pb.ModelPackage(graph_document=graph, weight_shards=shards)))
            authored = quantization.author_ptq_template(pb.AuthorPtqTemplateRequest(
                source_graph=graph, weight_shards=shards, config=pb.PtqAuthoringConfig(
                    activation_dtype=pb.DataType.DATA_TYPE_I8, activation_scheme=pb.PtqScheme.PTQ_SCHEME_ASYMMETRIC,
                    selected_nodes=["dense"])))
            template = json.loads(authored.template_graph)
            self.assertEqual(template["dimensions"], dimensions)
            for node in template["nodes"]:
                if node["id"] != "select":
                    self.assertEqual(node["outputs"]["out"]["shape"][1], "S")
            plan = quantization.create_ptq_plan(pb.CreatePtqPlanRequest(model_id=model.model_id,
                template_graph=authored.template_graph, observers=authored.observers,
                layers=authored.layers, profile_names=["dynamic"]))
            values = np.asarray([[[-1, .5, 1], [.25, -.75, .5], [0, 1, -1], [1, .25, -.5]]], np.float32)
            def tensor(count):
                return pb.Tensor(name="input", dtype=pb.DataType.DATA_TYPE_F32,
                    shape=[1, count, 3], inline=values[:, :count].tobytes())
            family = pb.Tensor(name="family", dtype=pb.DataType.DATA_TYPE_I32,
                shape=[1], inline=np.asarray([1], np.int32).tobytes())
            for index, count in enumerate((1, 4, 2)):
                quantization.calibrate_ptq_plan(pb.CalibratePtqPlanRequest(
                    ptq_plan_id=plan.ptq_plan_id, profile_name="dynamic", sample_name=f"sample-{index}",
                    sample_count=1, inputs=[tensor(count), family]))
            state = quantization.inspect_ptq_plan(pb.PtqPlanRef(ptq_plan_id=plan.ptq_plan_id))
            self.assertTrue(state.coverage.complete)
            self.assertEqual(state.calibration_samples, 3)
            packed = quantization.write_ptq_package(pb.WritePtqPackageRequest(ptq_plan_id=plan.ptq_plan_id))
            storage, document = load(packed.weights), json.loads(packed.graph)
            self.assertEqual(document["banks"], {"bank": "adapters", "weight": "dense_source"})
            np.testing.assert_array_equal(storage["bank"], weights["bank"])
            np.testing.assert_array_equal(storage["weight"], weights["weight"])
            layer = authored.layers[0]
            affine = document["quantization"]["tensors"][layer.packed_weight_name]
            scales = storage[affine["scale_tensor"]]
            np.testing.assert_array_equal(scales, (np.abs(canonical).max(1).astype(np.float64) / 127).astype(np.float32))
            expected = np.clip(np.rint(canonical.astype(np.float64) / scales[:, None]), -127, 127).astype(np.int8)
            np.testing.assert_array_equal(storage[layer.packed_weight_name], expected)
            if not with_bias:
                np.testing.assert_array_equal(storage[layer.packed_bias_name], np.zeros(2, np.int32))
            for graph_bytes, weight_shards, tolerance in ((graph, shards, 1e-6), (packed.graph, [packed.weights], .04)):
                loaded = inference.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                    package=pb.ModelPackage(graph_document=graph_bytes, weight_shards=weight_shards)))
                compiled = inference.compile_model(pb.CompileModelRequest(model_id=loaded.model_id,
                    policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends=["cpu"])))
                context = inference.create_execution_context(pb.CreateExecutionContextRequest(compiled_model_id=compiled.compiled_model_id))
                for count in (4, 1, 2):
                    result = inference.execute(pb.ExecuteRequest(context_id=context.context_id, inputs=[tensor(count), family]))
                    try:
                        output = inference.read_output(pb.ReadOutputRequest(result_id=result.result_id, name="output")).tensor
                        actual = np.frombuffer(output.inline, np.float32).reshape(output.shape)
                        np.testing.assert_allclose(actual, values[:, :count] @ canonical.T + bias + weights["bank"][1], atol=tolerance, rtol=0)
                    finally:
                        inference.release_result(pb.ResultRef(result_id=result.result_id))
            return storage[layer.packed_weight_name]
        finally:
            host.close()

    def test_both_layouts_and_optional_bias(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                np.testing.assert_array_equal(self.pipeline("din_dout", bias), self.pipeline("dout_din", bias))


if __name__ == "__main__":
    unittest.main()
