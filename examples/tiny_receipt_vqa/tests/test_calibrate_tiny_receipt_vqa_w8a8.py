import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
TOOLS = EXAMPLE_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

calibration = importlib.import_module("calibrate_tiny_receipt_vqa_w8a8")


class TinyReceiptCalibrationPureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="volvoxai-tiny-receipt-calibration-")
        self.root = Path(self.temporary.name)
        self.annotations = self.root / "annotations"
        self.images = self.root / "images"
        self.annotations.mkdir()
        self.images.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def annotation(self, identifier, question, answer="value"):
        path = self.annotations / f"{identifier}.json"
        path.write_text(json.dumps({"id": identifier, "question": question, "answer": answer}), encoding="utf-8")
        (self.images / f"{identifier}.jpg").write_bytes(b"not decoded by discovery")

    def test_discovery_matches_evaluator_supported_questions(self):
        self.annotation("00002", "What is the phone number?")
        self.annotation("00001", "What is the store address?")
        self.annotation("00003", "What is the merchant name?")
        self.annotation("00004", "", "not-used")
        records = calibration.discover_heldout_records(self.annotations, self.images)
        self.assertEqual([(record.id, record.family, record.family_id) for record in records], [
            ("00001", "address", 1),
            ("00002", "phone", 0),
        ])

    def test_stratified_selection_is_deterministic_round_robin(self):
        records = [
            calibration.HeldoutRecord("p2", Path("p2.json"), Path("p2.jpg"), "phone", "", "phone", 0),
            calibration.HeldoutRecord("a2", Path("a2.json"), Path("a2.jpg"), "address", "", "address", 1),
            calibration.HeldoutRecord("p1", Path("p1.json"), Path("p1.jpg"), "phone", "", "phone", 0),
            calibration.HeldoutRecord("a1", Path("a1.json"), Path("a1.jpg"), "address", "", "address", 1),
        ]
        selected = calibration.select_stratified_records(records, 10)
        self.assertEqual([record.id for record in selected], ["p1", "a1", "p2", "a2"])

    def test_per_probe_scales_and_global_fallback_are_explicit(self):
        observations = {
            "encoder.layers.0.self_attn.q": {"observed_abs_max": 3.0},
            "encoder.layers.0.self_attn.k": {"observed_abs_max": 4.0},
            "encoder.layers.0.self_attn.v": {"observed_abs_max": 5.0},
        }
        scales = calibration.derive_probe_and_global_i8_scales(observations, 1.1)
        self.assertEqual(scales["scheme"], "symmetric_per_tensor_i8")
        self.assertEqual(scales["graph_edge_mapping"]["status"], "unbound_source_probe_names")
        self.assertEqual(set(scales["probe_boundaries"]), set(observations))
        global_fallback = scales["global_fallback"]
        self.assertGreater(global_fallback["threshold_abs"], 5.0 * 1.1 - 1e-12)
        self.assertEqual(global_fallback["zero_point"], 0)
        self.assertEqual(global_fallback["quantized_code_range"], [-127, 127])

    def test_zero_only_probe_reuses_safe_global_fallback_scale(self):
        observations = {
            "adapter.unused.up": {"observed_abs_max": 0.0},
            "encoder.layers.0.self_attn.q": {"observed_abs_max": 2.0},
        }
        scales = calibration.derive_probe_and_global_i8_scales(observations, 1.1)
        zero_probe = scales["probe_boundaries"]["adapter.unused.up"]
        self.assertTrue(zero_probe["observed_zero_only"])
        self.assertEqual(zero_probe["scale_source"], "global_fallback_for_zero_only_probe")
        self.assertEqual(zero_probe["scale"], scales["global_fallback"]["scale"])

    def test_contract_is_graph_unbound_but_binds_samples_and_qkv(self):
        observations = {
            "encoder.layers.0.self_attn.q": {
                "calls": 1, "elements": 2, "observed_min": -1.0,
                "observed_max": 1.0, "observed_abs_max": 1.0,
            },
            "encoder.layers.0.self_attn.k": {
                "calls": 1, "elements": 2, "observed_min": -2.0,
                "observed_max": 2.0, "observed_abs_max": 2.0,
            },
            "encoder.layers.0.self_attn.v": {
                "calls": 1, "elements": 2, "observed_min": -3.0,
                "observed_max": 3.0, "observed_abs_max": 3.0,
            },
        }
        samples = [{
            "id": "00001",
            "annotation_sha256": "a" * 64,
            "image_sha256": "b" * 64,
            "question": "What is the address?",
            "expected_annotation_family": "address",
            "expected_annotation_family_id": 1,
            "first_decode_step": {"next_token_id": 4},
            "router": {"argmax_family_id": 1},
        }]
        contract = calibration.build_calibration_contract(
            manifest={"config": {"d_model": 4}},
            hashes={
                "manifest_sha256": "c" * 64,
                "weights_sha256": "d" * 64,
                "model_definition_sha256": "e" * 64,
            },
            samples=samples,
            observations=observations,
            activation_scales=calibration.derive_probe_and_global_i8_scales(observations, 1.1),
        )
        self.assertTrue(contract["development_only"])
        self.assertIn("not a Volvox graph", contract["purpose"])
        self.assertEqual(contract["qkv_probe_count"], 3)
        self.assertEqual(contract["profile"]["activation_quantization"]["global_fallback"]["dtype"], "int8")
        self.assertEqual(
            contract["profile"]["activation_quantization"]["graph_edge_mapping"]["status"],
            "unbound_source_probe_names",
        )
        self.assertEqual(
            contract["sample_set_sha256"],
            calibration.canonical_json_sha256(contract["sample_identities"]),
        )
        self.assertFalse(contract["family_coverage"]["explicit_family"]["enabled"])
        self.assertEqual(contract["goldens"]["explicit_family_first_decode_steps"], [])

    def test_explicit_family_coverage_is_complete_and_keeps_auto_goldens_separate(self):
        observations = {
            "encoder.layers.0.self_attn.q": {
                "calls": 1, "elements": 2, "observed_min": -1.0,
                "observed_max": 1.0, "observed_abs_max": 1.0,
            },
            "encoder.layers.0.self_attn.k": {
                "calls": 1, "elements": 2, "observed_min": -2.0,
                "observed_max": 2.0, "observed_abs_max": 2.0,
            },
            "encoder.layers.0.self_attn.v": {
                "calls": 1, "elements": 2, "observed_min": -3.0,
                "observed_max": 3.0, "observed_abs_max": 3.0,
            },
        }
        auto_samples = [{
            "id": "00001",
            "annotation_sha256": "a" * 64,
            "image_sha256": "b" * 64,
            "question": "What is the address?",
            "expected_annotation_family": "address",
            "expected_annotation_family_id": 1,
            "first_decode_step": {"next_token_id": 4},
            "router": {"argmax_family_id": 1},
        }]
        explicit_steps = [
            {
                "family": family,
                "family_id": family_id,
                "next_token_id": 10 + family_id,
                "logits_f32_sha256": f"{family_id:x}" * 64,
                "top_token_ids": [10 + family_id],
            }
            for family_id, family in enumerate(calibration.FAMILY_ORDER)
        ]
        explicit_goldens = [{
            "id": "00001",
            "annotation_sha256": "a" * 64,
            "image_sha256": "b" * 64,
            "question": "What is the address?",
            "question_token_ids": [1, 2],
            "decoder_input_ids": [1],
            "steps": explicit_steps,
        }]
        adapter_coverage = {
            "all_adapter_module_families_covered": True,
            "collections": [{"name": "memory_adapters"}, {"name": "decoder_adapters"}],
        }
        contract = calibration.build_calibration_contract(
            manifest={"config": {"d_model": 4}},
            hashes={
                "manifest_sha256": "c" * 64,
                "weights_sha256": "d" * 64,
                "model_definition_sha256": "e" * 64,
            },
            samples=auto_samples,
            observations=observations,
            activation_scales=calibration.derive_probe_and_global_i8_scales(observations, 1.1),
            explicit_family_goldens=explicit_goldens,
            include_explicit_families=True,
            adapter_probe_coverage=adapter_coverage,
        )
        self.assertEqual(contract["goldens"]["first_decode_step_and_router"], auto_samples)
        self.assertEqual(contract["goldens"]["explicit_family_first_decode_steps"], explicit_goldens)
        explicit_coverage = contract["family_coverage"]["explicit_family"]
        self.assertTrue(explicit_coverage["enabled"])
        self.assertEqual(explicit_coverage["requested_family_ids"], list(range(8)))
        self.assertEqual(explicit_coverage["covered_family_ids"], list(range(8)))
        self.assertEqual(explicit_coverage["runs_per_family"], {
            family: 1 for family in calibration.FAMILY_ORDER
        })
        self.assertEqual(explicit_coverage["adapter_module_probe_coverage"], adapter_coverage)

    def test_cli_parses_opt_in_explicit_family_flag(self):
        with patch.dict(
            os.environ,
            {"RECEIPT_VQA_DATA_ROOT": "/path/to/receipt-vqa-data"},
        ):
            args = calibration.parse_args(["--out", "out.json", "--include-explicit-families"])
        self.assertTrue(args.include_explicit_families)
        self.assertEqual(
            args.release_dir,
            Path(
                "/path/to/receipt-vqa-data/temp/"
                "tiny-receipt-vqa-structured-qa-21m-lora-router-e100-v1"
            ),
        )
        self.assertEqual(
            args.annotations_dir,
            Path("/path/to/receipt-vqa-data/eval/heldout/annotations"),
        )
        self.assertEqual(
            args.images_dir,
            Path("/path/to/receipt-vqa-data/eval/heldout/images"),
        )

    def test_cli_accepts_explicit_paths_without_data_root_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            args = calibration.parse_args([
                "--release-dir", "/path/to/release",
                "--annotations-dir", "/path/to/annotations",
                "--images-dir", "/path/to/images",
                "--out", "out.json",
            ])
        self.assertEqual(args.release_dir, Path("/path/to/release"))
        self.assertEqual(args.annotations_dir, Path("/path/to/annotations"))
        self.assertEqual(args.images_dir, Path("/path/to/images"))


class TinyReceiptCalibrationTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest(f"PyTorch unavailable: {error}") from error
        cls.torch = torch

    def test_named_int8_state_dequantizes_only_for_reference_memory(self):
        torch = self.torch
        manifest = {
            "tensors": {
                "linear.weight": {"dtype": "int8_per_out", "shape": [2, 3]},
                "linear.bias": {"dtype": "float32", "shape": [2]},
            },
        }
        named = {
            "quantized.linear.weight": torch.tensor([[-2, 3, 4], [5, -6, 7]], dtype=torch.int8),
            "scale.linear.weight": torch.tensor([0.25, 0.5], dtype=torch.float32),
            "float32.linear.bias": torch.tensor([1.0, -2.0], dtype=torch.float32),
        }
        state = calibration.build_dequantized_reference_state(manifest, named, torch)
        self.assertTrue(torch.equal(
            state["linear.weight"],
            torch.tensor([[-0.5, 0.75, 1.0], [2.5, -3.0, 3.5]], dtype=torch.float32),
        ))
        self.assertTrue(torch.equal(state["linear.bias"], named["float32.linear.bias"]))

    def test_preprocessing_is_grayscale_bilinear_and_minus_one_to_one(self):
        try:
            from PIL import Image
        except ImportError as error:
            self.skipTest(f"Pillow unavailable: {error}")
        with tempfile.TemporaryDirectory(prefix="volvoxai-tiny-receipt-preprocess-") as temporary:
            path = Path(temporary) / "flat.png"
            Image.new("L", (1, 1), 128).save(path)
            value = calibration.preprocess_evaluator_image(path, self.torch)
        self.assertEqual(tuple(value.shape), (1, 1, calibration.IMAGE_HEIGHT, calibration.IMAGE_WIDTH))
        expected = (128.0 / 255.0 - 0.5) / 0.5
        self.assertTrue(self.torch.allclose(value, self.torch.full_like(value, expected), atol=1e-7, rtol=0.0))

    def test_explicit_step_uses_one_batch_with_every_fixed_family_id(self):
        torch = self.torch

        class FixedFamilyModel:
            def __init__(self):
                self.received_family_ids = None

            def __call__(self, image, question, decoder_input, family_ids=None):
                self.received_family_ids = family_ids.detach().cpu().tolist()
                batch = image.shape[0]
                logits = torch.zeros((batch, 1, 16), dtype=torch.float32)
                for family_id in range(batch):
                    logits[family_id, 0, family_id] = 1.0
                return logits, torch.zeros((batch, len(calibration.FAMILY_ORDER)), dtype=torch.float32)

        model = FixedFamilyModel()
        steps = calibration._explicit_family_first_decode_steps(
            model=model,
            image=torch.ones((1, 1, 2, 2), dtype=torch.float32),
            question=torch.tensor([[4, 5]], dtype=torch.long),
            decoder_input=torch.tensor([[1]], dtype=torch.long),
            torch=torch,
        )
        self.assertEqual(model.received_family_ids, list(range(len(calibration.FAMILY_ORDER))))
        self.assertEqual([step["family_id"] for step in steps], list(range(len(calibration.FAMILY_ORDER))))
        self.assertEqual([step["next_token_id"] for step in steps], list(range(len(calibration.FAMILY_ORDER))))

    def test_structural_probes_name_residual_and_transformer_norm_boundaries(self):
        torch = self.torch

        class ResBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Linear(4, 4)

            def forward(self, value):
                return torch.nn.functional.silu(value + self.net(value))

        class LoRALinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Linear(4, 4)
                self.lora_a = torch.nn.Linear(4, 2, bias=False)
                self.lora_b = torch.nn.Linear(2, 4, bias=False)

            def forward(self, value):
                return self.base(value) + self.lora_b(self.lora_a(value))

        class TaskAdapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.down = torch.nn.Linear(4, 2)
                self.up = torch.nn.Linear(2, 4)

            def forward(self, value):
                return value + self.up(torch.nn.functional.gelu(self.down(value)))

        class ProbeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.res = ResBlock()
                self.linear1 = LoRALinear()
                self.adapter = TaskAdapter()
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=4, nhead=2, dim_feedforward=8, batch_first=True, norm_first=True,
                )
                decoder_layer = torch.nn.TransformerDecoderLayer(
                    d_model=4, nhead=2, dim_feedforward=8, batch_first=True, norm_first=True,
                )
                self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
                self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=1)

            def forward(self, value):
                value = self.adapter(self.linear1(self.res(value)))
                memory = self.encoder(value)
                return self.decoder(value, memory)

        model = ProbeModel().eval()
        tracker = calibration.ActivationTracker(torch)
        handles = calibration.install_activation_probes(model, torch, tracker)
        try:
            with torch.inference_mode():
                model(torch.ones((1, 2, 4), dtype=torch.float32))
        finally:
            calibration._remove_handles(handles)
        observations = tracker.as_json()
        expected_names = {
            "res.residual_output",
            "res.residual_sum",
            "linear1.combined_output",
            "linear1.gelu_output",
            "adapter.residual_output",
            "adapter.gelu_output",
            "encoder.layers.0.output",
            "decoder.layers.0.output",
            "encoder.layers.0.norm1.input",
            "encoder.layers.0.norm2.input",
            "decoder.layers.0.norm1.input",
            "decoder.layers.0.norm2.input",
            "decoder.layers.0.norm3.input",
            "encoder.layers.0.self_attn.q",
            "decoder.layers.0.multihead_attn.k",
        }
        self.assertTrue(expected_names.issubset(observations))


if __name__ == "__main__":
    unittest.main()
