import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TOOLS = EXAMPLE_ROOT / "tools"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

try:
    import numpy as np
    import torch
    from safetensors.torch import load_file
except ImportError as error:  # pragma: no cover - environment dependency.
    raise unittest.SkipTest(f"TinyStories exporter dependencies are unavailable: {error}")

try:
    import export_gptneo_safetensors as exporter
    import export_tokenizer as tokenizer_exporter
    from tools.exporter.validate_runtime_package import validate_runtime_package
except ImportError as error:  # pragma: no cover - environment dependency.
    raise unittest.SkipTest(f"TinyStories exporter dependencies are unavailable: {error}")


def minimal_config(num_layers=0):
    return SimpleNamespace(
        hidden_size=2,
        num_layers=num_layers,
        num_heads=1,
        activation_function="gelu_new",
        layer_norm_epsilon=1e-5,
        vocab_size=5,
        eos_token_id=4,
    )


def minimal_state_dict():
    return {
        "transformer.wte.weight": torch.arange(10, dtype=torch.float32).reshape(5, 2),
        "transformer.wpe.weight": torch.zeros((exporter.MAX_SEQUENCE_LENGTH, 2)),
        "transformer.ln_f.weight": torch.ones(2),
        "transformer.ln_f.bias": torch.zeros(2),
    }


class TinyStoriesExporterTests(unittest.TestCase):
    def test_generic_exporter_has_no_model_family_fallback(self):
        generic_exporter_path = REPOSITORY_ROOT / "tools" / "export_safetensors.py"
        source = generic_exporter_path.read_text(encoding="utf-8")
        self.assertNotIn("AutoModelForCausalLM", source)
        self.assertNotIn("build_gptneo_graph", source)
        self.assertNotIn("GPT-Neo", source)
        self.assertNotIn("TinyStories", source)

        spec = importlib.util.spec_from_file_location(
            "volvoxai_generic_safetensors_exporter", generic_exporter_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        generic_exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generic_exporter)
        with self.assertRaisesRegex(ValueError, r"accepts only \.onnx and \.tflite"):
            generic_exporter.export_model(
                "roneneldan/TinyStories-1M", "unused/model.safetensors"
            )

    def test_graph_converts_family_checkpoint_layout(self):
        state_dict = minimal_state_dict()
        state_dict.update(
            {
                "transformer.h.0.ln_1.weight": torch.ones(2),
                "transformer.h.0.ln_1.bias": torch.zeros(2),
                "transformer.h.0.attn.attention.q_proj.weight": torch.eye(2),
                "transformer.h.0.attn.attention.k_proj.weight": torch.eye(2) * 2,
                "transformer.h.0.attn.attention.v_proj.weight": torch.eye(2) * 3,
                "transformer.h.0.attn.attention.out_proj.weight": torch.eye(2),
                "transformer.h.0.attn.attention.out_proj.bias": torch.zeros(2),
                "transformer.h.0.ln_2.weight": torch.ones(2),
                "transformer.h.0.ln_2.bias": torch.zeros(2),
                "transformer.h.0.mlp.c_fc.weight": torch.arange(
                    16, dtype=torch.float32
                ).reshape(8, 2),
                "transformer.h.0.mlp.c_fc.bias": torch.zeros(8),
                "transformer.h.0.mlp.c_proj.weight": torch.arange(
                    16, dtype=torch.float32
                ).reshape(2, 8),
                "transformer.h.0.mlp.c_proj.bias": torch.zeros(2),
            }
        )
        tensors = {}

        nodes = exporter.build_gptneo_graph(
            minimal_config(num_layers=1), state_dict, tensors
        )

        self.assertNotIn("h.0.attn.attention.q_proj.weight", tensors)
        self.assertEqual(tuple(tensors["h.0.attn.qkv_proj.weight"].shape), (2, 6))
        self.assertEqual(tuple(tensors["h.0.mlp.c_fc.weight"].shape), (2, 8))
        self.assertEqual(tuple(tensors["h.0.mlp.c_proj.weight"].shape), (8, 2))
        self.assertEqual(tuple(tensors["lm_head.weight"].shape), (2, 5))
        self.assertEqual(nodes[-1]["outputs"]["out"], {
            "tensor": "logits", "shape": [1, "S", 5], "dtype": "float32",
        })
        self.assertEqual(
            [node["id"] for node in nodes],
            [f"node_{index}" for index in range(len(nodes))],
        )
        self.assertTrue(all(
            set(node) == {"id", "opType", "inputs", "outputs", "params"}
            for node in nodes
        ))
        self.assertTrue(all(
            node["params"].get("weight_layout") == "din_dout"
            for node in nodes if node["opType"] == "MatMul"
        ))
        attention = next(node for node in nodes if node["opType"] == "SDPA")
        self.assertTrue(attention["params"]["causal"])
        self.assertEqual(attention["params"]["scale"], 1.0)
        activation = next(node for node in nodes if node["opType"] == "GELU")
        self.assertEqual(activation["params"], {"approximate": "tanh"})

    def test_graph_preserves_supported_checkpoint_gelu_semantics(self):
        state_dict = minimal_state_dict()
        state_dict.update(
            {
                "transformer.h.0.ln_1.weight": torch.ones(2),
                "transformer.h.0.ln_1.bias": torch.zeros(2),
                "transformer.h.0.attn.attention.q_proj.weight": torch.eye(2),
                "transformer.h.0.attn.attention.k_proj.weight": torch.eye(2),
                "transformer.h.0.attn.attention.v_proj.weight": torch.eye(2),
                "transformer.h.0.attn.attention.out_proj.weight": torch.eye(2),
                "transformer.h.0.attn.attention.out_proj.bias": torch.zeros(2),
                "transformer.h.0.ln_2.weight": torch.ones(2),
                "transformer.h.0.ln_2.bias": torch.zeros(2),
                "transformer.h.0.mlp.c_fc.weight": torch.zeros((8, 2)),
                "transformer.h.0.mlp.c_fc.bias": torch.zeros(8),
                "transformer.h.0.mlp.c_proj.weight": torch.zeros((2, 8)),
                "transformer.h.0.mlp.c_proj.bias": torch.zeros(2),
            }
        )

        exact_config = minimal_config(num_layers=1)
        exact_config.activation_function = "gelu"
        exact_nodes = exporter.build_gptneo_graph(exact_config, state_dict, {})
        exact_gelu = next(node for node in exact_nodes if node["opType"] == "GELU")
        self.assertEqual(exact_gelu["params"], {"approximate": "none"})

        unsupported_config = minimal_config()
        unsupported_config.activation_function = "relu"
        with self.assertRaisesRegex(ValueError, "supports only activation_function"):
            exporter.build_gptneo_graph(
                unsupported_config, minimal_state_dict(), {}
            )

    def test_export_preserves_example_package_filenames(self):
        model = SimpleNamespace(config=minimal_config(), state_dict=minimal_state_dict)
        with tempfile.TemporaryDirectory(
            prefix="volvoxai-tinystories-export-"
        ) as temporary:
            output_dir = Path(temporary)
            output_path = output_dir / "model.safetensors"
            with mock.patch.object(
                exporter.AutoModelForCausalLM,
                "from_pretrained",
                return_value=model,
            ):
                exporter.export_model("local-checkpoint", output_path)

            graph = json.loads((output_dir / "graph.json").read_text(encoding="utf-8"))
            validate_runtime_package(output_dir / "graph.json", [output_path])
            self.assertEqual(graph["format"], "volvox-graph/v1")
            self.assertEqual(graph["dimensions"], {
                "S": {"min": 1, "max": exporter.MAX_SEQUENCE_LENGTH},
            })
            self.assertEqual(
                set(graph),
                {"format", "dimensions", "inputs", "nodes", "outputs"},
            )
            tensors = load_file(str(output_path), device="cpu")
            tokens = np.fromfile(output_dir / "tokens.i32", dtype=np.int32)
            positions = np.fromfile(output_dir / "positions.i32", dtype=np.int32)

        self.assertEqual(graph["outputs"], ["logits"])
        self.assertEqual(graph["inputs"]["tokens"]["shape"], [1, "S"])
        self.assertEqual(graph["inputs"]["positions"]["shape"], [1, "S"])
        self.assertIn("lm_head.weight", tensors)
        self.assertEqual(tokens.shape, (exporter.MAX_SEQUENCE_LENGTH,))
        self.assertEqual(tokens[5], 4)
        np.testing.assert_array_equal(
            positions,
            np.arange(exporter.MAX_SEQUENCE_LENGTH, dtype=np.int32),
        )

    def test_tokenizer_export_writes_binary_vocabulary_and_merges(self):
        class FakeTokenizer:
            def get_vocab(self):
                return {"a": 0, "é": 2}

            def save_vocabulary(self, output_dir):
                vocab_path = Path(output_dir) / "vocab.json"
                merges_path = Path(output_dir) / "merges.txt"
                vocab_path.write_text("{}", encoding="utf-8")
                merges_path.write_text("#version: 0.2\n", encoding="utf-8")
                return str(vocab_path), str(merges_path)

        with tempfile.TemporaryDirectory(
            prefix="volvoxai-tinystories-tokenizer-"
        ) as temporary:
            output_dir = Path(temporary)
            with mock.patch.object(
                tokenizer_exporter.AutoTokenizer,
                "from_pretrained",
                return_value=FakeTokenizer(),
            ):
                tokenizer_exporter.export_tokenizer("local-tokenizer", output_dir)
            payload = (output_dir / "vocab.bin").read_bytes()

        size = struct.unpack_from("<i", payload, 0)[0]
        first_length = struct.unpack_from("<i", payload, 4)[0]
        self.assertEqual(size, 3)
        self.assertEqual(payload[8 : 8 + first_length], b"a")


if __name__ == "__main__":
    unittest.main()
