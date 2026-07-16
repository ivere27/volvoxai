import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TOOLS = EXAMPLE_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

try:
    import torch
    from safetensors.torch import load_file
except ImportError as error:  # pragma: no cover - environment dependency.
    raise unittest.SkipTest(f"TinyReceipt KIE exporter dependencies are unavailable: {error}")


exporter = importlib.import_module("export_kie_safetensors")


class TinyReceiptKieExporterTests(unittest.TestCase):
    def test_generic_exporter_has_no_kie_checkpoint_path(self):
        source = (REPOSITORY_ROOT / "tools" / "export_safetensors.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("build_kie_graph", source)
        self.assertNotIn("multihead_attn.in_proj_weight", source)

    def test_graph_keeps_cross_attention_weight_scale_name(self):
        graph = exporter.build_kie_graph({
            "d_model": 2,
            "heads": 1,
            "enc_layers": 1,
            "dec_layers": 1,
            "vocab_size": 5,
        })
        cross_projection = next(
            node for node in graph
            if node.get("outputs", {}).get("out") == "dec_0_cross_proj"
        )
        self.assertEqual(
            cross_projection["inputs"]["scale"],
            "decoder.layers.0.multihead_attn.out_proj.weight_scale",
        )

    def test_local_checkpoint_export_splits_cross_attention_and_writes_graph(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-tiny-receipt-kie-export-") as temporary:
            root = Path(temporary)
            checkpoint_path = root / "checkpoint.pt"
            output_path = root / "package" / "model.safetensors"
            torch.save({
                "config": {
                    "d_model": 2,
                    "heads": 1,
                    "enc_layers": 0,
                    "dec_layers": 0,
                    "vocab_size": 5,
                },
                "model": {
                    "decoder.layers.0.multihead_attn.in_proj_weight": torch.arange(
                        12, dtype=torch.float32
                    ).reshape(6, 2),
                    "decoder.layers.0.multihead_attn.in_proj_bias": torch.arange(
                        6, dtype=torch.float32
                    ),
                    "projection.weight": torch.tensor(
                        [[-2.0, 0.0], [1.0, 4.0]], dtype=torch.float32
                    ),
                    "norm.weight": torch.tensor([1.0, 1.0], dtype=torch.float32),
                },
            }, checkpoint_path)

            exporter.export_checkpoint(checkpoint_path, output_path)

            tensors = load_file(str(output_path), device="cpu")
            config = json.loads((output_path.parent / "config.json").read_text(encoding="utf-8"))

        prefix = "decoder.layers.0.multihead_attn"
        for projection in ("q", "k", "v"):
            self.assertIn(f"{prefix}.{projection}_proj_weight", tensors)
            self.assertIn(f"{prefix}.{projection}_proj_scale", tensors)
        self.assertNotIn(f"{prefix}.in_proj_weight", tensors)
        self.assertEqual(tensors["projection.weight"].dtype, torch.int8)
        self.assertEqual(tensors["norm.weight"].dtype, torch.float32)
        self.assertEqual(config["outputs"], {"logits": "logits"})
        self.assertEqual(config["nodes"][-1]["outputs"], {"out": "logits"})
        self.assertEqual(config["nodes"][-1]["outputs_shape"]["out"], [1, 192, 5])

    def test_model_flag_remains_an_alias_for_checkpoint(self):
        args = exporter.parse_args(["--model", "checkpoint.pt", "--out", "model.safetensors"])
        self.assertEqual(args.checkpoint, Path("checkpoint.pt"))
        self.assertEqual(args.out, Path("model.safetensors"))


if __name__ == "__main__":
    unittest.main()
