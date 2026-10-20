"""INT4/INT8 storage, precision boundaries and reusable HF lowering contracts."""
from __future__ import annotations

import copy
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import load_file, save_file

from tools.exporter.errors import ExporterError
from tools.exporter.hf_decoder import DecoderConfig, build_decoder_graph
from tools.exporter.packed_linear import pack_weight, pack_linear_weights, packed_linear_reference
from tools.llm_baseline.generate_modern_incremental_stress import tiny_dense_config


class PackedStorageTests(unittest.TestCase):
    def test_nibble_order_signed_words_tail_padding_and_zero_groups(self):
        weight = np.array([[-7, -6, -5, -4, 0, 1, 6, 7, 0], [0] * 9], np.float32)
        words, scales = pack_weight(weight)
        self.assertEqual(words.dtype, np.int32)
        np.testing.assert_array_equal(scales, [[1], [1]])
        np.testing.assert_array_equal(words.view(np.uint32),
            [[0xFE984321, 0x88888888], [0x88888888, 0x88888888]])
        self.assertLess(words[0, 0], 0)

    def test_groups_round_nearest_even_and_bound_quantization_error(self):
        source = np.zeros((257, 131), np.float32)
        source[:, :7] = [7, -.5, .5, -1.5, 1.5, -2.5, 2.5]
        source[:, 128:] = [14, -14, 3]
        words, scales = pack_weight(source)
        np.testing.assert_array_equal(scales, np.tile([1, 2], (257, 1)))
        first = int(words[0, 0]) & 0xFFFFFFFF
        self.assertEqual([(first >> (4 * i) & 15) - 8 for i in range(8)], [7, 0, 0, -2, 2, -2, 2, 0])
        decoded = np.array([[(int(words[row, k // 8]) >> (4 * (k % 8)) & 15) - 8
            for k in range(131)] for row in range(257)], np.float32)
        decoded *= scales[:, np.arange(131) // 128]
        self.assertTrue(np.all(np.abs(decoded - source) <= scales[:, np.arange(131) // 128] / 2))
        self.assertEqual((int(words[-1, -1]) & 0xFFFFFFFF) >> 12, 0x88888)

    def test_int8_words_signed_extrema_halfway_rounding_and_group_tail(self):
        source = np.zeros((257, 131), np.float32)
        source[:, :7] = [127, -127, -.5, .5, -1.5, 1.5, 2.5]
        source[:, 128:] = [254, -254, 3]
        words, scales = pack_weight(source, weight_bits=8)
        self.assertEqual(words.shape, (257, 33))
        np.testing.assert_array_equal(scales, np.tile([1, 2], (257, 1)))
        np.testing.assert_array_equal(words[:, 0].view(np.uint32), np.full(257, 0x808001FF, np.uint32))
        np.testing.assert_array_equal(words[:, -1].view(np.uint32), np.full(257, 0x808201FF, np.uint32))
        decoded = np.array([[(int(words[row, k // 4]) >> (8 * (k % 4)) & 255) - 128
            for k in range(131)] for row in range(257)], np.float32)
        decoded *= scales[:, np.arange(131) // 128]
        self.assertTrue(np.all(np.abs(decoded - source) <= scales[:, np.arange(131) // 128] / 2))
        # Include -128, which is accepted by the storage contract although the
        # symmetric max/127 exporter ordinarily emits the range [-127,127].
        extreme = np.array([[0x8080FF00]], np.uint32).view(np.int32)
        for activation_bits in (8, 16):
            actual = packed_linear_reference(np.ones((1, 2), np.float32), extreme,
                np.ones((1, 1), np.float32), weight_bits=8, activation_bits=activation_bits)
            np.testing.assert_array_equal(actual, [[-1]])

    def test_refuse_invalid_storage_inputs_and_keep_tiny_scale_positive(self):
        for source in (np.empty((0, 8), np.float32), np.zeros((8,), np.float32),
                       np.ones((1, 8), np.float64), np.array([[np.nan]], np.float32),
                       np.array([[np.inf]], np.float32)):
            with self.subTest(source=source), self.assertRaises(ExporterError):
                pack_weight(source)
        for bits in (4, 8):
            words, scales = pack_weight(np.array([[np.nextafter(np.float32(0), np.float32(1))]], np.float32),
                                        weight_bits=bits)
            self.assertGreater(scales[0, 0], 0)
            self.assertEqual((int(words[0, 0]) & ((1 << bits) - 1)) - (1 << (bits - 1)), 1)
        for bits in (0, 3, 16, True, 4.0):
            with self.subTest(weight_bits=bits), self.assertRaises(ExporterError):
                pack_weight(np.ones((1, 1), np.float32), weight_bits=bits)
        for bits in (4, 8):
            words, scales = pack_weight(np.ones((1, 1), np.float32), weight_bits=bits)
            for bad_scale in (0, -1, np.inf, np.nan):
                with self.subTest(weight_bits=bits, scale=bad_scale), self.assertRaises(ExporterError):
                    packed_linear_reference(np.ones((1, 1), np.float32), words,
                        np.array([[bad_scale]], np.float32), weight_bits=bits)
            words[0, 0] ^= 1 << bits
            with self.subTest(weight_bits=bits, padding=True), self.assertRaises(ExporterError):
                packed_linear_reference(np.ones((1, 1), np.float32), words, scales, weight_bits=bits)

    def test_activation_precision_group_accumulation_bias_and_nonfinite_rows(self):
        words = np.full((2, 17), 0x99999999, np.uint32).view(np.int32)
        words[:, -1] = np.uint32(0x88888889).view(np.int32)
        scales = np.array([[2, 3], [1, 5]], np.float32)
        inputs = np.ones((1, 3, 129), np.float32)
        inputs[0, 1] = 0
        inputs[0, 2, 7] = np.inf
        for bits in (8, 16):
            actual = packed_linear_reference(inputs, words, scales, activation_bits=bits,
                                             bias=np.array([1, -1], np.float32))
            np.testing.assert_allclose(actual[0, :2], [[260, 132], [1, -1]], atol=2e-5, rtol=0)
            self.assertTrue(np.isnan(actual[0, 2]).all())
        half_input = np.array([[1 + 2 ** -11, 1 + 3 * 2 ** -11]], np.float32)
        half_words, half_scales = pack_weight(np.array([[1, 1]], np.float32))
        actual = packed_linear_reference(half_input, half_words, half_scales)
        self.assertEqual(float(actual[0, 0]), 2 + 2 ** -9)
        overflow = packed_linear_reference(np.array([[65520, 1]], np.float32), half_words, half_scales)
        self.assertTrue(np.isposinf(overflow[0, 0]))


class PackedGraphTests(unittest.TestCase):
    @staticmethod
    def fixture():
        config = DecoderConfig.from_hf(tiny_dense_config())
        rng = np.random.default_rng(421)
        weights = {name: rng.normal(0, .02, shape).astype(np.float32)
                   for name, shape in config.weight_shapes().items()}
        graph, weights = build_decoder_graph(config, weights, compact_kv=True, cached=True)
        return config, graph, weights

    def test_shared_tied_embedding_ownership_and_dynamic_shape_proof(self):
        from tools.exporter.runtime_ir import import_runtime_package
        config, graph, weights = self.fixture()
        original = copy.deepcopy(graph)
        packed_graph, packed = pack_linear_weights(graph, weights)
        self.assertEqual(graph, original)
        self.assertIs(packed["model.embed_tokens.weight"], weights["model.embed_tokens.weight"])
        self.assertNotIn("model.layers.0.self_attn.q_proj.weight", packed)
        self.assertEqual(packed_graph["execution"], original["execution"])
        self.assertEqual(sum(node["opType"] == "PackedLinear" for node in packed_graph["nodes"]), 7 * config.layers + 1)
        ir = import_runtime_package(packed_graph, packed)
        for length in (1, config.max_context):
            bound = ir.bind_shape_profile({"S": length})
            self.assertEqual(bound.tensors["logits"].shape, (1, length, config.vocab_size))
        first = next(node for node in packed_graph["nodes"] if node["opType"] == "PackedLinear")
        self.assertEqual(first["params"], {"group_size": 128, "weight_bits": 4, "activation_bits": 16})

    def test_layout_and_shared_weight_are_packed_once(self):
        weight = np.array([[7, 0, -7], [0, 7, 0]], np.float32)
        graph = {"nodes": [{"id": name, "opType": "Linear", "inputs": {"input": "x", "weight": "w"},
                            "params": {"transB": False}} for name in ("a", "b")], "outputs": ["a", "b"]}
        converted, tensors = pack_linear_weights(graph, {"w": weight.T}, activation_bits=8)
        self.assertEqual(converted["nodes"][0]["inputs"], converted["nodes"][1]["inputs"])
        self.assertEqual(len(tensors), 2)
        words = tensors[converted["nodes"][0]["inputs"]["weight"]]
        np.testing.assert_array_equal(words, pack_weight(weight)[0])
        self.assertEqual(converted["nodes"][0]["params"]["activation_bits"], 8)

    def test_hf_cli_exports_packed_shard_without_intermediate_f32_package(self):
        from tools.exporter.cli import main
        from tools.export_safetensors import export_model
        _, _, weights = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text(json.dumps(tiny_dense_config()))
            save_file(weights, source / "model.safetensors")
            for precision in ("w4a16", "w4a8", "w8a16"):
                output = root / precision / "model.safetensors"
                code = main(export_model, ["--model", str(source), "--out", str(output),
                    "--weight-precision", precision, "--compact-kv", "--cached-decoder", "--target", "native-cpu"])
                self.assertEqual(code, 0)
                graph = json.loads((output.parent / "graph.json").read_text())
                packed = load_file(output)
                nodes = [node for node in graph["nodes"] if node["opType"] == "PackedLinear"]
                self.assertTrue(nodes)
                self.assertTrue(all(packed[node["inputs"]["weight"]].dtype == np.int32 for node in nodes))
                self.assertTrue(all(node["params"]["weight_bits"] == (8 if precision == "w8a16" else 4) for node in nodes))
                self.assertEqual(set(path.name for path in output.parent.iterdir()), {"graph.json", "model.safetensors"})


def _independent_packed_hf(activation_bits, weight_bits=4):
    """HF architecture with independently quantized Torch Linear arithmetic."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    class Linear(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            outputs, self.width = weight.shape
            padded = torch.nn.functional.pad(weight, (0, -self.width % 128))
            grouped = padded.reshape(outputs, -1, 128)
            maximum = grouped.abs().amax(-1)
            offset = 1 << (weight_bits - 1)
            self.scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / (offset - 1))
            self.integer = torch.clamp(torch.round(grouped / self.scale[..., None]), -offset, offset - 1)
            self.rounded = (self.integer * self.scale[..., None]).flatten(1)[:, :self.width].half().float()

        def forward(self, value):
            if activation_bits == 16:
                return torch.nn.functional.linear(value.half().float(), self.rounded)
            rows = value.reshape(-1, self.width)
            maximum = rows.abs().amax(-1)
            scale = torch.where(maximum == 0, torch.ones_like(maximum),
                                torch.clamp(maximum / 127, min=2**-126))
            integer = torch.clamp(torch.round(rows / scale[:, None]), -127, 127)
            integer = torch.nn.functional.pad(integer, (0, -self.width % 128)).reshape(len(rows), -1, 128)
            result = torch.zeros((len(rows), len(self.integer)), dtype=torch.float32)
            for group in range(integer.shape[1]):
                # Integer-valued FP32 products and sums are exact for a group:
                # 128*127*128 < 2**24. Scaling and group accumulation are F32.
                dot = integer[:, group] @ self.integer[:, group].T
                result += dot * (scale[:, None] * self.scale[:, group][None])
            return result.reshape(*value.shape[:-1], len(self.integer))

    torch.manual_seed(5521)
    torch.set_num_threads(1)
    raw = tiny_dense_config(hidden_size=64, intermediate_size=136, head_dim=16, max_position_embeddings=32,
        rope_theta=500000., rope_scaling={"rope_type": "llama3", "factor": 2.,
            "low_freq_factor": 1., "high_freq_factor": 4., "original_max_position_embeddings": 8})
    hf = LlamaConfig(**raw)
    hf._attn_implementation = "eager"
    reference = LlamaForCausalLM(hf).eval()
    source = {name: value.detach().numpy() for name, value in reference.state_dict().items()}
    config = DecoderConfig.from_hf(raw, max_context=32)
    graph, weights = build_decoder_graph(config, source, compact_kv=True, cached=True)
    graph, weights = pack_linear_weights(graph, weights, weight_bits=weight_bits, activation_bits=activation_bits)
    for name in [name for name, module in reference.named_modules() if isinstance(module, torch.nn.Linear)]:
        reference.set_submodule(name, Linear(reference.get_submodule(name).weight.detach()))
    return reference, config, graph, weights


class PackedHFGenerationTests(unittest.TestCase):
    def test_cached_rows_and_greedy_match_independent_hf_on_both_profiles(self):
        try:
            import torch
            import transformers
        except ImportError:
            self.skipTest("independent HF oracle requires torch and transformers")
        from safetensors.numpy import save
        from tools.llm_baseline.generate_modern_incremental_stress import initial_values, reference_rows
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "python"))
        import volvoxai as vx
        pb = vx.pb
        corpus = {"format": "volvox-modern-incremental/v1",
            "reference_policy": "independent-hf-torch-packed-linear",
            "precision": "group128 INT4/INT8; independent Torch FP16 rounding or exact INT8 group dots; F32 outputs",
            "torch": torch.__version__, "transformers": transformers.__version__, "cases": []}
        for weight_bits, bits in ((4, 16), (4, 8), (8, 16)):
            reference, config, graph, weights = _independent_packed_hf(bits, weight_bits)
            with tempfile.TemporaryDirectory(prefix="volvox-packed-hf-") as temporary:
                path = Path(temporary) / "model.safetensors"
                save_file(weights, path)
                for length in (3, 9):
                    prompt = [2 + (row * 7) % 30 for row in range(length)]
                    initial = initial_values(graph, config, prompt)
                    scores, ids = reference_rows(reference, prompt, 4, initial)
                    with torch.inference_mode():
                        prefill = reference(input_ids=torch.tensor([prompt]),
                            attention_mask=torch.ones(1, length, dtype=torch.long),
                            position_ids=torch.arange(length)[None], use_cache=False).logits.numpy().copy()
                    self.assertEqual(initial["token_ids"].shape, (1, 32))
                    corpus["cases"].append({"name": f"llama3-w{weight_bits}a{bits}-prompt{length}", "graph": graph,
                        "weight_shards": [base64.b64encode(save(weights)).decode()],
                        "inputs": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                                    "data": base64.b64encode(value.tobytes()).decode()} for name, value in initial.items()],
                        "expected_logits": [value.tolist() for value in scores], "expected_prefill_logits": prefill.tolist(),
                        "expected_ids": ids, "budgets": [1, 3]})
                    for library in ("libvolvoxai.so", "libvolvoxai-lite.so"):
                        with self.subTest(weight_bits=weight_bits, bits=bits, length=length, library=library), \
                                patch.dict(os.environ, {"VOLVOXAI_LIBRARY": str(root / "native" / library)}), \
                                vx.InferenceSession(graph_document=json.dumps(graph).encode(), weights=[path],
                                    cpu_threads=1, graph=pb.GraphExecution(slots=1,
                                        row_mode=pb.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,
                                        require_incremental=True)) as session:
                            values = {name: value.copy() for name, value in initial.items()}
                            previous = None
                            for step, expected in enumerate(scores):
                                row = length - 1 + step
                                if step:
                                    values["token_ids"][0, row] = ids[step - 1]
                                    values["attention_mask"][0, row] = 1
                                actual = session.run(values if not step else
                                    {name: value for name, value in values.items() if name != "active_lengths"},
                                    reuse_inputs=["active_lengths"] if step else (), output_names=["logits"],
                                    scope=pb.ExecutionScope(entrypoint="transition", slot_rows=pb.SlotRows(
                                        slots=[pb.SlotRow(position=row)])) if step else
                                        pb.ExecutionScope(entrypoint="initialize", full=pb.Empty()))["logits"]
                                np.testing.assert_allclose(actual[0, row], expected, atol=2e-4, rtol=2e-4)
                                if step == 0:
                                    np.testing.assert_allclose(actual[:, :length], prefill, atol=2e-4, rtol=2e-4)
                                if previous is not None:
                                    np.testing.assert_array_equal(actual[:, :row], previous[:, :row])
                                previous = actual
                            engine = session._engine
                            context = engine.create_execution_context(pb.CreateExecutionContextRequest(
                                compiled_model_id=session.compiled_model_id, autoregressive=pb.AutoregressiveExecution(
                                    sampling=pb.SamplingMode.SAMPLING_MODE_GREEDY, max_steps=4)))
                            try:
                                observed = []
                                for chunk, budget in enumerate((1, 3)):
                                    packet = engine.generate(pb.GenerateRequest(context_id=context.context_id,
                                        initial_inputs=pb.GenerationInputs(values=[pb.Tensor(name=name,
                                            shape=value.shape, dtype=pb.DataType.DATA_TYPE_I32, inline=value.tobytes())
                                            for name, value in initial.items()]) if chunk == 0 else None,
                                        budget=pb.GenerationBudget(max_steps=budget),
                                        outputs=pb.OutputOptions(storage=pb.OutputStorage.OUTPUT_STORAGE_INLINE)))
                                    output = next(value for value in packet.outputs if value.name == "generated_ids")
                                    observed.extend(np.frombuffer(output.inline, np.int32).tolist())
                                self.assertEqual(observed, ids)
                            finally:
                                engine.release_execution_context(pb.ExecutionContextRef(context_id=context.context_id))
        destination = os.environ.get("VOLVOXAI_PACKED_HF_CORPUS")
        if destination:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(corpus) + "\n")


if __name__ == "__main__":
    unittest.main()
