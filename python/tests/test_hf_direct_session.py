"""Local HF convenience uses public inline graphs and unchanged source shards."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))
import volvoxai as vx
from volvoxai import _model

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from safetensors.torch import save_file
    HAVE_HF = True
except ImportError:
    HAVE_HF = False


@unittest.skipUnless(HAVE_HF, "independent oracle requires torch and transformers")
class HFDirectSessionTests(unittest.TestCase):
    def fixture(self, directory, *, duplicate_head=False):
        torch.manual_seed(4421)
        torch.set_num_threads(1)
        config = LlamaConfig(hidden_size=64, intermediate_size=128, head_dim=16,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            vocab_size=32, max_position_embeddings=32, tie_word_embeddings=True,
            rope_theta=500000., rope_scaling={"rope_type": "llama3", "factor": 2.,
                "low_freq_factor": 1., "high_freq_factor": 4., "original_max_position_embeddings": 16})
        config._attn_implementation = "eager"
        reference = LlamaForCausalLM(config).eval().bfloat16()
        values = {name: tensor.detach().clone() for name, tensor in reference.state_dict().items()
                  if name != "lm_head.weight" or duplicate_head}
        names, mapping = sorted(values), {}
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(config.to_dict()))
        for index, selected in enumerate((names[::2], names[1::2])):
            filename = f"model-{index}.safetensors"
            # Preserve mixed source storage and original HF names/layouts.
            tensors = {name: values[name].float() if name.endswith("layernorm.weight") else values[name]
                       for name in selected}
            save_file(tensors, directory / filename)
            mapping.update({name: filename for name in selected})
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
        return reference.float().eval(), tuple(directory / name for name in sorted(set(mapping.values())))

    @staticmethod
    def inputs(length):
        return {"token_ids": (np.arange(length, dtype=np.int32)[None] + 2),
                "position_ids": np.arange(length, dtype=np.int32),
                "attention_mask": np.ones((1, length), np.int32)}

    @staticmethod
    def expected(reference, values, length=None):
        length = length or values["token_ids"].shape[1]
        with torch.inference_mode():
            return reference(input_ids=torch.tensor(values["token_ids"][:, :length]).long(),
                position_ids=torch.tensor(values["position_ids"][:length][None]).long(),
                attention_mask=torch.tensor(values["attention_mask"][:, :length]).long()).logits.numpy()

    def test_original_bf16_shards_inline_graph_and_snapshot_ownership(self):
        from tools.exporter import hf_graph
        original_load = vx.VxInferenceServiceClient.load_model
        for library in ("libvolvoxai.so", "libvolvoxai-lite.so"):
            with self.subTest(library=library), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "source"
                reference, paths = self.fixture(source)
                before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
                requests = []
                def load(client, request):
                    requests.append(vx.pb.LoadModelRequest.FromString(request.SerializeToString()))
                    return original_load(client, request)
                with patch.dict(os.environ, {"VOLVOXAI_LIBRARY": str(ROOT / "native" / library)}), \
                        patch.object(hf_graph, "read_checkpoint_storage", side_effect=AssertionError("no weight conversion")), \
                        patch.object(hf_graph, "write_hf_package", side_effect=AssertionError("no exported package")), \
                        patch.object(vx.VxInferenceServiceClient, "load_model", load):
                    session = vx.InferenceSession.from_pretrained(source, max_context=8, cpu_threads=1)
                with session:
                    self.assertIsNone(session.model_path)
                    self.assertEqual(requests[0].graph_path, "")
                    self.assertTrue(requests[0].graph_document)
                    self.assertEqual(tuple(Path(path) for path in requests[0].weight_paths[:-1]), paths)
                    derived = Path(requests[0].weight_paths[-1])
                    self.assertFalse(derived.exists())
                    self.assertFalse(derived.parent.exists())
                    self.assertEqual(before, {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()})
                    for path in paths:
                        path.unlink()
                    for length in (1, 7, 3):
                        values = self.inputs(length)
                        np.testing.assert_allclose(session.run(values)["logits"], self.expected(reference, values),
                                                   atol=1e-5, rtol=1e-5)
                    values = self.inputs(3)
                    retained = session.run_tensors(values)
                    session.close()
                    with retained:
                        np.testing.assert_allclose(retained["logits"].numpy(), self.expected(reference, values),
                                                   atol=1e-5, rtol=1e-5)

    def test_cached_required_rows_preserve_prefixes_against_hf(self):
        for library in ("libvolvoxai.so", "libvolvoxai-lite.so"):
            with self.subTest(library=library), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "source"
                reference, _ = self.fixture(source, duplicate_head=True)
                values = self.inputs(8)
                values["attention_mask"][:, 2:] = 0
                values["active_lengths"] = np.array([2], np.int32)
                with patch.dict(os.environ, {"VOLVOXAI_LIBRARY": str(ROOT / "native" / library)}), \
                        vx.InferenceSession.from_pretrained(source, max_context=8, cached=True, cpu_threads=1) as session:
                    actual = session.run(values, entrypoint="initialize", output_names=["logits"])["logits"]
                    np.testing.assert_allclose(actual[:, :2], self.expected(reference, values, 2), atol=1e-5, rtol=1e-5)
                    self.assertEqual(session._graph_execution.row_mode, vx.pb.DecodeRowMode.DECODE_ROW_MODE_REQUIRED)
                    for row in (2, 3, 4):
                        previous = actual.copy()
                        values["attention_mask"][0, row] = 1
                        actual = session.run({key: value for key, value in values.items() if key != "active_lengths"},
                            reuse_inputs=["active_lengths"], output_names=["logits"],
                            scope=vx.pb.ExecutionScope(entrypoint="transition",
                                slot_rows=vx.pb.SlotRows(slots=[vx.pb.SlotRow(position=row)])))["logits"]
                        np.testing.assert_array_equal(actual[:, :row], previous[:, :row])
                        np.testing.assert_allclose(actual[:, row], self.expected(reference, values, row + 1)[:, row],
                                                   atol=1e-5, rtol=1e-5)

    def test_source_refusals_and_failed_compile_remove_derived_files(self):
        original_load = vx.VxInferenceServiceClient.load_model
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            _, paths = self.fixture(source, duplicate_head=True)
            for options in ({}, {"max_context": 33}, {"max_context": 8, "cached": "yes"}):
                with self.subTest(options=options), patch("volvoxai._session.open_library") as opened:
                    with self.assertRaises((ValueError, TypeError)):
                        vx.InferenceSession.from_pretrained(source, **options)
                    opened.assert_not_called()
            derived = []
            def load(client, request):
                derived.extend(Path(path) for path in request.weight_paths if Path(path) not in paths)
                return original_load(client, request)
            with patch.object(vx.VxInferenceServiceClient, "load_model", load), self.assertRaises(vx.VolvoxAIError):
                vx.InferenceSession.from_pretrained(source, max_context=8, backend="unavailable-test-backend")
            self.assertEqual(len(derived), 1)
            self.assertFalse(derived[0].parent.exists())
            self.assertTrue(all(path.exists() for path in paths))
            from safetensors.torch import load_file
            index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
            head_path = source / index["lm_head.weight"]
            tensors = load_file(head_path)
            tensors["lm_head.weight"][0, 0] += 1
            save_file(tensors, head_path)
            with patch("volvoxai._session.open_library") as opened, self.assertRaisesRegex(ValueError, "tied lm_head.weight"):
                vx.InferenceSession.from_pretrained(source, max_context=8)
            opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
