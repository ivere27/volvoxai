"""C PTQ packs either Linear source layout into per-output-channel rows."""
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
        graph = json.dumps(stateless_graph({"format": "volvox-graph/v1", "dimensions": dimensions,
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
            "outputs": ["output"]})).encode()
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
                context = inference.create_execution_context(pb.CreateExecutionContextRequest(compiled_model_id=compiled.compiled_model_id, graph=pb.GraphExecution()))
                for count in (4, 1, 2):
                    result = inference.execute(pb.ExecuteRequest(context_id=context.context_id, inputs=pb.InputBindings(values=[tensor(count), family]), scope=pb.ExecutionScope(entrypoint="evaluate", full=pb.Empty())))
                    output = next(tensor for tensor in result.outputs if tensor.name == "output")
                    actual = np.frombuffer(output.inline, np.float32).reshape(output.shape)
                    np.testing.assert_allclose(actual, values[:, :count] @ canonical.T + bias + weights["bank"][1], atol=tolerance, rtol=0)
            return storage[layer.packed_weight_name]
        finally:
            host.close()

    def test_both_layouts_and_optional_bias(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                np.testing.assert_array_equal(self.pipeline("din_dout", bias), self.pipeline("dout_din", bias))



class PackedPtqTest(unittest.TestCase):
    """Generated Author/Create/Calibrate/Write, including source-only BF16."""

    def setUp(self):
        self.host = vx.open_library()
        self.inference = vx.VxInferenceServiceClient(self.host)
        self.quantization = vx.VxQuantizationServiceClient(self.host)
        self.runtime = self.inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))

    def tearDown(self):
        self.host.close()

    @staticmethod
    def fixture(layout="dout_din", source_dtype="F32", canonical=None):
        if canonical is None:
            canonical = np.zeros((3, 137), np.float32)
            canonical[0, :8] = [7, -7, .5, 1.5, 2.5, -.5, -1.5, -2.5]
            canonical[0, 128:] = np.arange(-4, 5, dtype=np.float32)
            canonical[2, :8] = [127, -127, .5, 1.5, 2.5, -.5, -1.5, -2.5]
            canonical[2, 128:] = np.arange(-4, 5, dtype=np.float32) * np.float32(.125)
        weight = np.ascontiguousarray(canonical.T if layout == "din_dout" else canonical)
        bias = np.asarray([.125, -.25, .5], np.float32)
        graph = json.dumps(stateless_graph({"format": "volvox-graph/v1", "dimensions": {},
            "inputs": {"input": {"shape": [2, canonical.shape[1]], "dtype": "float32"}},
            "nodes": [{"id": "dense", "opType": "Linear",
                "inputs": {"input": "input", "weight": "weight", "bias": "bias"},
                "outputs": {"out": {"tensor": "output", "shape": [2, 3], "dtype": "float32"}},
                "params": {"weight_layout": layout}}], "outputs": ["output"]})).encode()
        if source_dtype == "BF16":
            raw = (weight.view(np.uint32) >> np.uint32(16)).astype(np.uint16).tobytes()
            header = json.dumps({"weight": {"dtype": "BF16", "shape": list(weight.shape),
                "data_offsets": [0, len(raw)]}, "bias": {"dtype": "F32", "shape": [3],
                "data_offsets": [len(raw), len(raw) + bias.nbytes]}}).encode()
            header += b" " * (-len(header) % 8)
            shard = len(header).to_bytes(8, "little") + header + raw + bias.tobytes()
        else:
            shard = save({"weight": weight, "bias": bias})
        return graph, shard, canonical, bias

    def author(self, mode, **fixture):
        graph, shard, canonical, bias = self.fixture(**fixture)
        model = self.inference.load_model(pb.LoadModelRequest(runtime_id=self.runtime.runtime_id,
            package=pb.ModelPackage(graph_document=graph, weight_shards=[shard])))
        authored = self.quantization.author_ptq_template(pb.AuthorPtqTemplateRequest(
            source_graph=graph, weight_shards=[shard], config=pb.PtqAuthoringConfig(mode=mode)))
        self.assertEqual(authored.layers[0].mode, mode)
        self.assertEqual(authored.layers[0].kind, pb.PtqLayerKind.PTQ_LAYER_KIND_PACKED_LINEAR)
        self.assertEqual(authored.layers[0].packed_bias_name, "")
        self.assertTrue(authored.layers[0].packed_scale_name)
        self.assertEqual(len(authored.observers), 2)
        self.assertTrue(all(not observer.quantized_tensor_name for observer in authored.observers))
        self.assertNotIn("quantization", json.loads(authored.template_graph))
        return model, authored, canonical, bias

    def plan(self, model, authored, template=None):
        return self.quantization.create_ptq_plan(pb.CreatePtqPlanRequest(model_id=model.model_id,
            template_graph=template or authored.template_graph, layers=authored.layers,
            observers=authored.observers, profile_names=["sample"]))

    def calibrate(self, plan, values=None):
        if values is None:
            values = np.linspace(-1, 1, 274, dtype=np.float32).reshape(2, 137)
        tensor = pb.Tensor(name="input", dtype=pb.DataType.DATA_TYPE_F32,
            shape=list(values.shape), inline=values.tobytes())
        self.quantization.calibrate_ptq_plan(pb.CalibratePtqPlanRequest(
            ptq_plan_id=plan.ptq_plan_id, profile_name="sample", sample_name="one",
            sample_count=1, inputs=[tensor]))
        return values, tensor

    def test_packed_storage_and_inference_f32_bf16_sources(self):
        modes = ((pb.PtqMode.PTQ_MODE_W4A8, 4, 8), (pb.PtqMode.PTQ_MODE_W4A16, 4, 16),
                 (pb.PtqMode.PTQ_MODE_W8A16, 8, 16))
        for mode, weight_bits, activation_bits in modes:
            for layout in ("dout_din", "din_dout"):
                for source_dtype in ("F32", "BF16"):
                    with self.subTest(mode=mode, layout=layout, source_dtype=source_dtype):
                        model, authored, canonical, bias = self.author(mode, layout=layout, source_dtype=source_dtype)
                        plan = self.plan(model, authored)
                        with self.assertRaises(vx.VolvoxAIError):
                            self.quantization.write_ptq_package(pb.WritePtqPackageRequest(ptq_plan_id=plan.ptq_plan_id))
                        values, tensor = self.calibrate(plan)
                        package = self.quantization.write_ptq_package(pb.WritePtqPackageRequest(ptq_plan_id=plan.ptq_plan_id))
                        storage, document = load(package.weights), json.loads(package.graph)
                        self.assertNotIn("weight", storage)
                        self.assertNotIn("quantization", document)
                        np.testing.assert_array_equal(storage["bias"], bias)
                        layer = authored.layers[0]
                        words, scales = storage[layer.packed_weight_name], storage[layer.packed_scale_name]
                        lanes, offset = 32 // weight_bits, 1 << (weight_bits - 1)
                        expected_q = np.zeros_like(canonical, dtype=np.int32)
                        expected_scale = np.ones((3, 2), np.float32)
                        for group, start in enumerate((0, 128)):
                            block = canonical[:, start:start + 128]
                            maximum = np.abs(block).max(axis=1)
                            expected_scale[:, group] = np.where(maximum == 0, np.float32(1), maximum / np.float32(offset - 1))
                            expected_q[:, start:start + 128] = np.clip(np.rint(block / expected_scale[:, group, None]), -offset, offset - 1).astype(np.int32)
                        np.testing.assert_array_equal(scales, expected_scale)
                        padded = np.full((3, words.shape[1] * lanes), offset, np.uint32)
                        padded[:, :137] = (expected_q + offset).astype(np.uint32)
                        expected_words = np.bitwise_or.reduce(padded.reshape(3, -1, lanes) <<
                            (np.arange(lanes, dtype=np.uint32) * np.uint32(weight_bits)), axis=2)
                        np.testing.assert_array_equal(words.view(np.uint32), expected_words)
                        if activation_bits == 16:
                            rounded_input = values.astype(np.float16).astype(np.float32)
                            rounded_weight = (expected_q.astype(np.float32) * scales[:, np.arange(137) // 128]).astype(np.float16).astype(np.float32)
                            reference = np.zeros((2, 3), np.float32)
                            for column in range(137):
                                reference += rounded_input[:, column, None] * rounded_weight[None, :, column]
                        else:
                            scale_a = np.abs(values).max(axis=1) / np.float32(127)
                            qa = np.rint(values / scale_a[:, None]).astype(np.int32)
                            reference = np.zeros((2, 3), np.float32)
                            for group, start in enumerate((0, 128)):
                                dot = qa[:, start:start + 128] @ expected_q[:, start:start + 128].T
                                reference += dot.astype(np.float32) * (scale_a[:, None] * scales[None, :, group])
                        reference += bias
                        loaded = self.inference.load_model(pb.LoadModelRequest(runtime_id=self.runtime.runtime_id,
                            package=pb.ModelPackage(graph_document=package.graph, weight_shards=[package.weights])))
                        compiled = self.inference.compile_model(pb.CompileModelRequest(model_id=loaded.model_id,
                            policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends=["cpu"])))
                        context = self.inference.create_execution_context(pb.CreateExecutionContextRequest(
                            compiled_model_id=compiled.compiled_model_id, graph=pb.GraphExecution()))
                        output = self.inference.execute(pb.ExecuteRequest(context_id=context.context_id,
                            inputs=pb.InputBindings(values=[tensor]), scope=pb.ExecutionScope(entrypoint="evaluate", full=pb.Empty()))).outputs[0]
                        np.testing.assert_array_equal(np.frombuffer(output.inline, np.float32).reshape(2, 3), reference)

    def test_writer_matches_independent_packer_for_extreme_groups(self):
        sys.path.insert(0, str(ROOT / "tools"))
        from exporter.packed_linear import pack_weight
        canonical = np.zeros((3, 131), np.float32)
        smallest = np.nextafter(np.float32(0), np.float32(1))
        canonical[0, :5] = np.asarray([1, -1, 2, -2, 3], np.float32) * smallest
        canonical[0, 128:] = [smallest, -smallest, smallest]
        canonical[1, :5] = np.asarray([1, -1, .5, -.5, .25], np.float32) * np.finfo(np.float32).max
        for mode, bits in ((pb.PtqMode.PTQ_MODE_W4A8, 4), (pb.PtqMode.PTQ_MODE_W4A16, 4),
                           (pb.PtqMode.PTQ_MODE_W8A16, 8)):
            with self.subTest(mode=mode):
                model, authored, _, _ = self.author(mode, canonical=canonical)
                plan = self.plan(model, authored)
                self.calibrate(plan, np.zeros((2, 131), np.float32))
                package = self.quantization.write_ptq_package(pb.WritePtqPackageRequest(ptq_plan_id=plan.ptq_plan_id))
                storage = load(package.weights)
                expected_words, expected_scales = pack_weight(canonical, weight_bits=bits)
                layer = authored.layers[0]
                np.testing.assert_array_equal(storage[layer.packed_weight_name], expected_words)
                np.testing.assert_array_equal(storage[layer.packed_scale_name], expected_scales)
                self.assertTrue(np.all(expected_scales > 0))

    def test_packed_modes_reject_ambiguous_configuration(self):
        graph, shard, _, _ = self.fixture()
        for fields in ({"activation_dtype": pb.DataType.DATA_TYPE_U8},
                       {"activation_scheme": pb.PtqScheme.PTQ_SCHEME_ASYMMETRIC},
                       {"reduce_range": True}, {"mode": 123}):
            with self.subTest(fields=fields), self.assertRaises(vx.VolvoxAIError):
                config = {"mode": pb.PtqMode.PTQ_MODE_W4A16, **fields}
                self.quantization.author_ptq_template(pb.AuthorPtqTemplateRequest(
                    source_graph=graph, weight_shards=[shard], config=pb.PtqAuthoringConfig(**config)))

    def test_authoring_name_storage_survives_large_decoder(self):
        graph, shard, _, _ = self.fixture()
        document = json.loads(graph)
        base = document["nodes"][0]
        document["nodes"] = []
        for index in range(80):
            node = json.loads(json.dumps(base))
            node["id"] = f"dense-{index}"
            node["outputs"]["out"]["tensor"] = f"output-{index}"
            document["nodes"].append(node)
        document["outputs"] = [node["outputs"]["out"]["tensor"] for node in document["nodes"]]
        graph = json.dumps(stateless_graph(document)).encode()
        for mode in (pb.PtqMode.PTQ_MODE_W8A8, pb.PtqMode.PTQ_MODE_W4A16):
            with self.subTest(mode=mode):
                request = pb.AuthorPtqTemplateRequest(source_graph=graph, weight_shards=[shard],
                    config=pb.PtqAuthoringConfig(mode=mode))
                first = self.quantization.author_ptq_template(request)
                second = self.quantization.author_ptq_template(request)
                self.assertEqual(first.template_graph, second.template_graph)
                self.assertEqual(len(first.layers), 80)
                self.assertEqual(len({layer.packed_weight_name for layer in first.layers}), 80)

    def test_writer_refuses_tampered_packed_contract(self):
        model, authored, _, _ = self.author(pb.PtqMode.PTQ_MODE_W4A16)
        mutations = (
            lambda n: n["params"].update(group_size=64),
            lambda n: n["params"].update(weight_bits=8),
            lambda n: n["params"].update(activation_bits=8),
            lambda n: n["params"].update(extra=1),
            lambda n: n["inputs"].update(scale="bias"),
            lambda n: n["outputs"]["out"].update(dtype="int32"),
            lambda n: n["outputs"]["out"].update(shape=[2, 4]),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(vx.VolvoxAIError):
                document = json.loads(authored.template_graph)
                mutate(document["nodes"][0])
                plan = self.plan(model, authored, json.dumps(document).encode())
                self.calibrate(plan)
                self.quantization.write_ptq_package(pb.WritePtqPackageRequest(ptq_plan_id=plan.ptq_plan_id))

if __name__ == "__main__":
    unittest.main()
