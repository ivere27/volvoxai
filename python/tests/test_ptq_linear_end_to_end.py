"""C PTQ packs either Linear source layout into per-output-channel rows."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import load, save

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution_fixture import stateless_graph, feedback_graph

import volvoxai as vx  # noqa: E402

pb = vx.pb


def grouped_config(weight_bits=4, rules=(), group_size=128):
    return pb.QuantizationConfig(weights=pb.WeightQuantization(
        format=pb.WeightFormat(bits=weight_bits, group_size=group_size)), rules=rules)


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
            plan = quantization.create_quantization_plan(pb.CreateQuantizationPlanRequest(config=pb.QuantizationConfig(weights=pb.WeightQuantization(format=pb.WeightFormat(bits=8)), static_activations=pb.StaticActivations(dtype=pb.DataType.DATA_TYPE_I8, scheme=pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC), rules=[pb.QuantizationRule(node_id=name, weights=pb.WeightQuantization(format=pb.WeightFormat(bits=8))) for name in ["dense"]] + [pb.QuantizationRule(node_glob="*", keep_float=pb.Empty())]), bytes=pb.ModelBytes(graph=graph, weight_shards=shards), calibration=pb.CalibrationOptions(profile_names=["dynamic"])))
            values = np.asarray([[[-1, .5, 1], [.25, -.75, .5], [0, 1, -1], [1, .25, -.5]]], np.float32)
            def tensor(count):
                return pb.Tensor(name="input", dtype=pb.DataType.DATA_TYPE_F32,
                    shape=[1, count, 3], inline=values[:, :count].tobytes())
            family = pb.Tensor(name="family", dtype=pb.DataType.DATA_TYPE_I32,
                shape=[1], inline=np.asarray([1], np.int32).tobytes())
            for index, count in enumerate((1, 4, 2)):
                quantization.calibrate_quantization_plan(pb.CalibrateQuantizationPlanRequest(
                    quantization_plan_id=plan.quantization_plan_id, profile_name="dynamic", sample_name=f"sample-{index}",
                    sample_count=1, inputs=[tensor(count), family]))
            state = quantization.inspect_quantization_plan(pb.QuantizationPlanRef(quantization_plan_id=plan.quantization_plan_id)).info
            self.assertTrue(state.ready)
            self.assertEqual(state.calibration.samples, 3)
            packed = quantization.export_quantized_model(pb.ExportQuantizedModelRequest(quantization_plan_id=plan.quantization_plan_id))
            storage, document = load(packed.bytes.weight_shards[0]), json.loads(packed.bytes.graph)
            self.assertEqual(document["banks"], {"bank": "adapters", "weight": "dense_source"})
            np.testing.assert_array_equal(storage["bank"], weights["bank"])
            np.testing.assert_array_equal(storage["weight"], weights["weight"])
            self.assertEqual(document["dimensions"], dimensions)
            converted = next(node for node in document["nodes"] if node["id"] == "dense")
            packed_weight_name, packed_bias_name = converted["inputs"]["weight"], converted["inputs"]["bias"]
            affine = document["quantization"]["tensors"][packed_weight_name]
            scales = storage[affine["scale_tensor"]]
            np.testing.assert_array_equal(scales, (np.abs(canonical).max(1).astype(np.float64) / 127).astype(np.float32))
            expected = np.clip(np.rint(canonical.astype(np.float64) / scales[:, None]), -127, 127).astype(np.int8)
            np.testing.assert_array_equal(storage[packed_weight_name], expected)
            if not with_bias:
                np.testing.assert_array_equal(storage[packed_bias_name], np.zeros(2, np.int32))
            for graph_bytes, weight_shards, tolerance in ((graph, shards, 1e-6), (packed.bytes.graph, [packed.bytes.weight_shards[0]], .04)):
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
            return storage[packed_weight_name]
        finally:
            host.close()

    def test_both_layouts_and_optional_bias(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                np.testing.assert_array_equal(self.pipeline("din_dout", bias), self.pipeline("dout_din", bias))



class PackedPtqTest(unittest.TestCase):
    """The public plan/export lifecycle preserves packed arithmetic across source layouts."""

    def setUp(self):
        self.host = vx.open_library()
        self.addCleanup(self.host.close)
        self.inference = vx.VxInferenceServiceClient(self.host)
        self.quantization = vx.VxQuantizationServiceClient(self.host)
        self.runtime = self.inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))

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

    def test_packed_storage_and_inference_f32_bf16_sources(self):
        for weight_bits, activation_bits in ((4, 8), (4, 16), (8, 8), (8, 16)):
            for layout in ("dout_din", "din_dout"):
                for source_dtype in ("F32", "BF16"):
                    with self.subTest(weight_bits=weight_bits, activation_bits=activation_bits,
                                      layout=layout, source_dtype=source_dtype):
                        graph, shard, canonical, bias = self.fixture(layout=layout, source_dtype=source_dtype)
                        config = grouped_config(weight_bits)
                        plan = self.quantization.create_quantization_plan(pb.CreateQuantizationPlanRequest(config=config, bytes=pb.ModelBytes(graph=graph, weight_shards=[shard])))
                        self.assertTrue(plan.info.ready)
                        self.assertFalse(plan.info.requires_calibration)
                        self.assertFalse(plan.info.HasField("calibration"))
                        package = self.quantization.export_quantized_model(pb.ExportQuantizedModelRequest(
                            quantization_plan_id=plan.quantization_plan_id))
                        self.quantization.release_quantization_plan(pb.QuantizationPlanRef(
                            quantization_plan_id=plan.quantization_plan_id))
                        values = np.linspace(-1, 1, 274, dtype=np.float32).reshape(2, 137)
                        tensor = pb.Tensor(name="input", dtype=pb.DataType.DATA_TYPE_F32,
                            shape=list(values.shape), inline=values.tobytes())
                        storage, document = load(package.bytes.weight_shards[0]), json.loads(package.bytes.graph)
                        self.assertNotIn("weight", storage)
                        self.assertNotIn("quantization", document)
                        np.testing.assert_array_equal(storage["bias"], bias)
                        node = document["nodes"][0]
                        words, scales = storage[node["inputs"]["weight"]], storage[node["inputs"]["scale"]]
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
                            package=pb.ModelPackage(graph_document=package.bytes.graph, weight_shards=[package.bytes.weight_shards[0]])))
                        compiled = self.inference.compile_model(pb.CompileModelRequest(model_id=loaded.model_id,
                            packed_arithmetic=(pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8 if activation_bits == 8 else pb.PackedArithmetic.PACKED_ARITHMETIC_F16),
                            policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends=["cpu"])))
                        context = self.inference.create_execution_context(pb.CreateExecutionContextRequest(
                            compiled_model_id=compiled.compiled_model_id, graph=pb.GraphExecution()))
                        output = self.inference.execute(pb.ExecuteRequest(context_id=context.context_id,
                            inputs=pb.InputBindings(values=[tensor]), scope=pb.ExecutionScope(entrypoint="evaluate", full=pb.Empty()))).outputs[0]
                        np.testing.assert_array_equal(np.frombuffer(output.inline, np.float32).reshape(2, 3), reference)

    def test_grouped_storage_matches_independent_packer_for_extreme_groups(self):
        sys.path.insert(0, str(ROOT / "tools"))
        from exporter.packed_linear import pack_weight
        canonical = np.zeros((3, 131), np.float32)
        smallest = np.nextafter(np.float32(0), np.float32(1))
        canonical[0, :5] = np.asarray([1, -1, 2, -2, 3], np.float32) * smallest
        canonical[0, 128:] = [smallest, -smallest, smallest]
        canonical[1, :5] = np.asarray([1, -1, .5, -.5, .25], np.float32) * np.finfo(np.float32).max
        graph, shard, _, _ = self.fixture(canonical=canonical)
        for bits, activation_bits in ((4, 8), (4, 16), (8, 8), (8, 16)):
            with self.subTest(bits=bits, activation_bits=activation_bits):
                plan = self.quantization.create_quantization_plan(pb.CreateQuantizationPlanRequest(config=grouped_config(bits), bytes=pb.ModelBytes(graph=graph, weight_shards=[shard])))
                try:
                    package = self.quantization.export_quantized_model(pb.ExportQuantizedModelRequest(
                        quantization_plan_id=plan.quantization_plan_id))
                finally:
                    self.quantization.release_quantization_plan(pb.QuantizationPlanRef(
                        quantization_plan_id=plan.quantization_plan_id))
                node = json.loads(package.bytes.graph)["nodes"][0]
                storage = load(package.bytes.weight_shards[0])
                words, scales = pack_weight(canonical, weight_bits=bits)
                np.testing.assert_array_equal(storage[node["inputs"]["weight"]], words)
                np.testing.assert_array_equal(storage[node["inputs"]["scale"]], scales)
                self.assertTrue(np.all(scales > 0))

    def test_group_sizes_export_and_execute_each_arithmetic_contract(self):
        from tools.exporter.packed_linear import pack_weight, packed_linear_reference
        width = 259
        canonical = np.random.default_rng(117).normal(0, .2, (3, width)).astype(np.float32)
        canonical[0, 33] = 3.5
        canonical[1] = 0
        graph, shard, _, bias = self.fixture(layout="din_dout", canonical=canonical)
        values = np.random.default_rng(118).normal(0, .1, (2, width)).astype(np.float32)
        tensor = pb.Tensor(name="input", dtype=pb.DataType.DATA_TYPE_F32,
            shape=values.shape, inline=values.tobytes())
        for group_size in (32, 64, 128, 256):
            for bits in (4, 8):
                with self.subTest(group_size=group_size, bits=bits):
                    config = grouped_config(bits, group_size=group_size)
                    plan = self.quantization.create_quantization_plan(pb.CreateQuantizationPlanRequest(
                        config=config, bytes=pb.ModelBytes(graph=graph, weight_shards=[shard])))
                    try:
                        self.assertEqual(plan.info.quantized[0].format.group_size, group_size)
                        package = self.quantization.export_quantized_model(pb.ExportQuantizedModelRequest(
                            quantization_plan_id=plan.quantization_plan_id))
                    finally:
                        self.quantization.release_quantization_plan(pb.QuantizationPlanRef(
                            quantization_plan_id=plan.quantization_plan_id))
                    node = json.loads(package.bytes.graph)["nodes"][0]
                    self.assertEqual(node["params"]["group_size"], group_size)
                    storage = load(package.bytes.weight_shards[0])
                    words, scales = pack_weight(canonical, weight_bits=bits, group_size=group_size)
                    np.testing.assert_array_equal(storage[node["inputs"]["weight"]], words)
                    np.testing.assert_array_equal(storage[node["inputs"]["scale"]], scales)
                    loaded = self.inference.load_model(pb.LoadModelRequest(runtime_id=self.runtime.runtime_id,
                        package=pb.ModelPackage(graph_document=package.bytes.graph,
                            weight_shards=[package.bytes.weight_shards[0]])))
                    for arithmetic, activation_bits in (
                        (pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8, 8),
                        (pb.PackedArithmetic.PACKED_ARITHMETIC_F16, 16),
                        (pb.PackedArithmetic.PACKED_ARITHMETIC_F32, 32),
                        (pb.PackedArithmetic.PACKED_ARITHMETIC_F32_FAST, 32)):
                        compiled = self.inference.compile_model(pb.CompileModelRequest(model_id=loaded.model_id,
                            packed_arithmetic=arithmetic, policy=pb.BackendPolicy(
                                mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends=["cpu"])))
                        self.assertEqual(compiled.packed_arithmetic, arithmetic)
                        context = self.inference.create_execution_context(pb.CreateExecutionContextRequest(
                            compiled_model_id=compiled.compiled_model_id, graph=pb.GraphExecution()))
                        output = self.inference.execute(pb.ExecuteRequest(context_id=context.context_id,
                            inputs=pb.InputBindings(values=[tensor]),
                            scope=pb.ExecutionScope(entrypoint="evaluate", full=pb.Empty()))).outputs[0]
                        expected = packed_linear_reference(values, words, scales, weight_bits=bits,
                            group_size=group_size, activation_bits=activation_bits, bias=bias)
                        np.testing.assert_allclose(np.frombuffer(output.inline, np.float32).reshape(2, 3),
                            expected, rtol=2e-6, atol=2e-7)

class GroupedQuantizationPlanTest(unittest.TestCase):
    """Grouped plans are ready immediately and never accept calibration."""

    def setUp(self):
        self.host = vx.open_library()
        self.addCleanup(self.host.close)
        self.quantization = vx.VxQuantizationServiceClient(self.host)
        self.graph, self.shard, _, _ = PackedPtqTest.fixture()

    def request(self, **changes):
        graph = changes.pop("source_graph", self.graph)
        shards = changes.pop("weight_shards", [self.shard])
        return pb.CreateQuantizationPlanRequest(bytes=pb.ModelBytes(graph=graph, weight_shards=shards),
            config=changes.pop("config", grouped_config()), **changes)

    def pack(self, request, **paths):
        plan = self.quantization.create_quantization_plan(request)
        self.assertTrue(plan.info.ready)
        self.assertFalse(plan.info.requires_calibration)
        self.assertFalse(plan.info.HasField("calibration"))
        reference = pb.QuantizationPlanRef(quantization_plan_id=plan.quantization_plan_id)
        try:
            with self.assertRaises(vx.VolvoxAIError):
                self.quantization.calibrate_quantization_plan(pb.CalibrateQuantizationPlanRequest(
                    quantization_plan_id=plan.quantization_plan_id, sample_count=1))
            return self.quantization.export_quantized_model(pb.ExportQuantizedModelRequest(
                quantization_plan_id=plan.quantization_plan_id,
                output_files=(pb.ModelFiles(graph_path=paths.get("output_graph_path", ""),
                    weight_paths=[paths["output_weights_path"]] if "output_weights_path" in paths else []) if paths else None)))
        finally:
            self.quantization.release_quantization_plan(reference)

    def test_explicit_valid_policies_are_required(self):
        invalid = [None, pb.QuantizationConfig(),
                   pb.QuantizationConfig(weights=pb.WeightQuantization(format=pb.WeightFormat(bits=5, group_size=128)), static_activations=None),
                   pb.QuantizationConfig(weights=grouped_config().weights, static_activations=pb.StaticActivations(dtype=pb.DataType.DATA_TYPE_F16))]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(vx.VolvoxAIError):
                self.quantization.create_quantization_plan(self.request(config=config))

    def test_invalid_selection_names_are_refused(self):
        for name in ("", "missing", "dense\0suffix"):
            with self.subTest(name=name), self.assertRaises(vx.VolvoxAIError):
                config = grouped_config(rules=[pb.QuantizationRule(node_id=name, keep_float=pb.Empty())])
                self.pack(self.request(config=config))

    def test_shard_count_and_nonfinite_weights_are_refused(self):
        for shards in ([], [self.shard, self.shard]):
            with self.subTest(shards=len(shards)), self.assertRaises(vx.VolvoxAIError):
                self.pack(self.request(weight_shards=shards))
        for value in (np.nan, np.inf, -np.inf):
            canonical = np.ones((3, 137), np.float32)
            canonical[1, 136] = value
            graph, shard, _, _ = PackedPtqTest.fixture(canonical=canonical)
            with self.subTest(weight=value), self.assertRaises(vx.VolvoxAIError):
                self.pack(self.request(source_graph=graph, weight_shards=[shard]))

    def test_selected_nodes_keep_float_consumers_and_omit_calibration_metadata(self):
        document = json.loads(self.graph)
        retained = json.loads(json.dumps(document["nodes"][0]))
        retained["id"] = "retained"
        retained["outputs"]["out"]["tensor"] = "float_output"
        document["nodes"].append(retained)
        document["outputs"].append("float_output")
        graph = json.dumps(stateless_graph(document)).encode()
        config = grouped_config(rules=[pb.QuantizationRule(node_id=name, weights=grouped_config().weights) for name in ["dense"]] + [pb.QuantizationRule(node_glob="*", keep_float=pb.Empty())])
        result = self.pack(self.request(source_graph=graph, config=config))
        repeated = self.pack(self.request(source_graph=graph, config=config))
        self.assertEqual((sum(count.nodes for count in result.plan.quantized), result.plan.float_nodes), (1, 1))
        self.assertEqual(result.bytes.graph, repeated.bytes.graph)
        self.assertEqual(result.bytes.weight_shards[0], repeated.bytes.weight_shards[0])
        authored = json.loads(result.bytes.graph)
        self.assertEqual([node["opType"] for node in authored["nodes"]], ["PackedLinear", "Linear"])
        self.assertEqual(authored["inputs"], document["inputs"])
        self.assertEqual(authored["outputs"], document["outputs"])
        self.assertNotIn("quantization", authored)
        storage = load(result.bytes.weight_shards[0])
        np.testing.assert_array_equal(storage["weight"], load(self.shard)["weight"])
        header_size = int.from_bytes(result.bytes.weight_shards[0][:8], "little")
        metadata = json.loads(result.bytes.weight_shards[0][8:8 + header_size]).get("__metadata__", {})
        for name in ("profile_coverage", "logical_fingerprint", "activation_sources"):
            self.assertNotIn(name, metadata)
        self.assertNotEqual(metadata.get("format"), "volvox.ptq.v1")

    def test_source_calibration_claims_are_removed_and_other_metadata_survives(self):
        shard = save(load(self.shard), metadata={
            "format": "volvox.ptq.v1", "logical_fingerprint": "old-model",
            "profile_coverage": '{"complete":true}', "activation_sources": "{}",
            "application": "keep this",
        })
        result = self.pack(self.request(weight_shards=[shard]))
        header_size = int.from_bytes(result.bytes.weight_shards[0][:8], "little")
        metadata = json.loads(result.bytes.weight_shards[0][8:8 + header_size]).get("__metadata__", {})
        self.assertEqual(metadata, {"application": "keep this"})

    def test_large_decoder_preserves_distinct_weights_and_reuses_tied_weights(self):
        for distinct in (False, True):
            with self.subTest(distinct=distinct):
                document = json.loads(self.graph)
                base = document["nodes"][0]
                document["nodes"] = []
                source_weights = load(self.shard)
                original_weight = source_weights[base["inputs"]["weight"]]
                for index in range(80):
                    node = json.loads(json.dumps(base))
                    node["id"] = f"dense-{index}"
                    if distinct:
                        node["inputs"]["weight"] = f"source-{index}.weight"
                        source_weights[node["inputs"]["weight"]] = original_weight.copy()
                    node["outputs"]["out"]["tensor"] = f"output-{index}"
                    document["nodes"].append(node)
                document["outputs"] = [node["outputs"]["out"]["tensor"] for node in document["nodes"]]
                graph = json.dumps(stateless_graph(document)).encode()
                request = self.request(source_graph=graph, weight_shards=[save(source_weights)])
                first, second = self.pack(request), self.pack(request)
                self.assertEqual(first.bytes.graph, second.bytes.graph)
                self.assertEqual(first.bytes.weight_shards[0], second.bytes.weight_shards[0])
                nodes = json.loads(first.bytes.graph)["nodes"]
                self.assertEqual(len(nodes), 80)
                self.assertEqual(len({node["inputs"]["weight"] for node in nodes}), 80 if distinct else 1)
                self.assertEqual(len({node["inputs"]["scale"] for node in nodes}), 80 if distinct else 1)

    def test_output_paths_are_paired_and_never_replace_existing_files(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-direct-pack-") as temporary:
            root = Path(temporary)
            output_graph, output_weights = root / "graph.json", root / "model.safetensors"
            paths = dict(output_graph_path=str(output_graph), output_weights_path=str(output_weights))
            invalid = (
                {"output_graph_path": str(output_graph)},
                {"output_weights_path": str(output_weights)},
                {**paths, "output_graph_path": str(root / "wrong.json")},
                {**paths, "output_weights_path": str(root / "nested" / "model.safetensors")},
            )
            for values in invalid:
                with self.subTest(paths=values), self.assertRaises(vx.VolvoxAIError):
                    self.pack(self.request(), **values)
                self.assertFalse(output_graph.exists())
                self.assertFalse(output_weights.exists())
            for existing in (output_graph, output_weights):
                existing.write_bytes(b"preserve existing file")
                with self.subTest(existing=existing), self.assertRaises(vx.VolvoxAIError):
                    self.pack(self.request(), **paths)
                self.assertEqual(existing.read_bytes(), b"preserve existing file")
                self.assertFalse((output_weights if existing == output_graph else output_graph).exists())
                existing.unlink()
            self.pack(self.request(), **paths)
            before_graph, before_weights = output_graph.read_bytes(), output_weights.read_bytes()
            with self.assertRaises(vx.VolvoxAIError):
                self.pack(self.request(), **paths)
            self.assertEqual(output_graph.read_bytes(), before_graph)
            self.assertEqual(output_weights.read_bytes(), before_weights)


if __name__ == "__main__":
    unittest.main()
