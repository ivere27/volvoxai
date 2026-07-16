import contextlib
import hashlib
import importlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
TOOLS = EXAMPLE_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

try:
    import numpy as np
    from safetensors.numpy import load_file, save_file
except ImportError as error:
    raise unittest.SkipTest(
        f"TinyReceiptVQA normalizer dependencies are unavailable: {error}"
    ) from error

normalizer = importlib.import_module("import_tiny_receipt_vqa_int8")


def release_manifest(tensors):
    return {
        "format": normalizer.SOURCE_FORMAT,
        "runtime": normalizer.SOURCE_RUNTIME,
        "files": {"model_safetensors": "model_int8.safetensors"},
        "safetensors": {
            "layout": normalizer.SOURCE_LAYOUT,
            "quantization": normalizer.SOURCE_QUANTIZATION,
        },
        "tensors": tensors,
    }


class TinyReceiptVqaInt8NormalizerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="volvoxai-tiny-receipt-normalizer-")
        self.root = Path(self.temporary.name)
        self.manifest_path = self.root / "manifest.json"
        self.weights_path = self.root / "model_int8.safetensors"
        self.output_path = self.root / "normalized.safetensors"
        self.fragment_path = self.root / "weights_quantization.json"

        self.conv = np.array([
            [[[-12, -11], [-10, -9]], [[-8, -7], [-6, -5]], [[-4, -3], [-2, -1]]],
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
        ], dtype=np.int8)
        self.linear = np.array([[-6, -5, -4], [3, 4, 5]], dtype=np.int8)
        self.bias = np.array([1.25, -2.5], dtype=np.float32)
        self.conv_scale = np.array([0.25, 0.5], dtype=np.float32)
        self.linear_scale = np.array([0.125, 0.75], dtype=np.float32)
        self.tensors = {
            "vision.conv.weight": {"shape": list(self.conv.shape), "dtype": "int8_per_out"},
            "decoder.linear.weight": {"shape": list(self.linear.shape), "dtype": "int8_per_out"},
            "decoder.linear.bias": {"shape": list(self.bias.shape), "dtype": "float32"},
        }

    def tearDown(self):
        self.temporary.cleanup()

    def write_fixture(self, *, manifest=None, tensors=None):
        if manifest is None:
            manifest = release_manifest(self.tensors)
        if tensors is None:
            tensors = {
                "quantized.vision.conv.weight": self.conv,
                "scale.vision.conv.weight": self.conv_scale,
                "quantized.decoder.linear.weight": self.linear,
                "scale.decoder.linear.weight": self.linear_scale,
                "float32.decoder.linear.bias": self.bias,
            }
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        save_file(tensors, str(self.weights_path))

    def normalize(self, *, conv_keys=("vision.conv.weight",), conversion_contract_path=None):
        return normalizer.normalize_release(
            self.manifest_path,
            self.output_path,
            self.fragment_path,
            conv_keys=conv_keys,
            conversion_contract_path=conversion_contract_path,
        )

    @staticmethod
    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def write_json(path, value):
        Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write_conversion_contract(
        self,
        *,
        graph_mutator=None,
        calibration_mutator=None,
        preprocessing_mutator=None,
        routing_mutator=None,
        goldens_mutator=None,
    ):
        self.write_fixture()
        contract_path = self.root / "conversion-contract.json"
        graph_path = self.root / "graph.json"
        calibration_path = self.root / "calibration.json"
        preprocessing_path = self.root / "preprocessing.json"
        goldens_path = self.root / "goldens.json"
        calibration_dataset_path = self.root / "calibration-inputs.bin"
        golden_input_path = self.root / "golden-input.bin"
        calibration_dataset_path.write_bytes(b"calibration sample set v1\n")
        golden_input_path.write_bytes(b"one deterministic receipt and question\n")

        manifest_sha256 = self.sha256(self.manifest_path)
        weights_sha256 = self.sha256(self.weights_path)
        graph = {
            "format": normalizer.GRAPH_SPEC_FORMAT,
            "id": "tiny-receipt-vqa-w8a8-explicit-family",
            "source_manifest_sha256": manifest_sha256,
            "source_weights_sha256": weights_sha256,
            "activation_edges": [
                {
                    "id": "vision.input.quantized",
                    "dtype": "uint8",
                    "shape": [1, 320, 672, 1],
                    "quantization": "per_tensor",
                },
                {
                    "id": "decoder.logits.quantized",
                    "dtype": "int8",
                    "shape": [1, 1, 17],
                    "quantization": "per_tensor",
                },
            ],
        }
        if graph_mutator is not None:
            graph_mutator(graph)
        self.write_json(graph_path, graph)
        graph_sha256 = self.sha256(graph_path)

        calibration = {
            "format": normalizer.ACTIVATION_CALIBRATION_FORMAT,
            "graph_sha256": graph_sha256,
            "dataset_file": calibration_dataset_path.name,
            "dataset_sha256": self.sha256(calibration_dataset_path),
            "sample_count": 3,
            "edges": [
                {
                    "id": "vision.input.quantized",
                    "dtype": "uint8",
                    "shape": [1, 320, 672, 1],
                    "scale": 1.0 / 255.0,
                    "zero_point": 0,
                    "observed_min": 0.0,
                    "observed_max": 1.0,
                },
                {
                    "id": "decoder.logits.quantized",
                    "dtype": "int8",
                    "shape": [1, 1, 17],
                    "scale": 0.125,
                    "zero_point": 0,
                    "observed_min": -8.0,
                    "observed_max": 7.0,
                },
            ],
        }
        if calibration_mutator is not None:
            calibration_mutator(calibration)
        self.write_json(calibration_path, calibration)
        calibration_sha256 = self.sha256(calibration_path)

        preprocessing = {
            "format": normalizer.PREPROCESSING_FORMAT,
            "input": {
                "color_space": "grayscale",
                "value_min": 0.0,
                "value_max": 255.0,
            },
            "resize": {
                "width": 672,
                "height": 320,
                "interpolation": "bilinear",
            },
            "output": {
                "dtype": "float32",
                "layout": "NHWC",
                "scale": 1.0 / 255.0,
                "offset": 0.0,
            },
        }
        if preprocessing_mutator is not None:
            preprocessing_mutator(preprocessing)
        self.write_json(preprocessing_path, preprocessing)
        preprocessing_sha256 = self.sha256(preprocessing_path)

        routing = {
            "mode": "explicit_family_v1",
            "family_order": list(normalizer.EXPLICIT_FAMILY_ORDER),
            "route_scope": "whole_execution",
            "batch_policy": "homogeneous",
        }
        if routing_mutator is not None:
            routing_mutator(routing)
        routing_sha256 = normalizer._canonical_json_sha256(routing)
        goldens = {
            "format": normalizer.GOLDENS_FORMAT,
            "source_manifest_sha256": manifest_sha256,
            "source_weights_sha256": weights_sha256,
            "graph_sha256": graph_sha256,
            "calibration_sha256": calibration_sha256,
            "preprocessing_sha256": preprocessing_sha256,
            "routing_sha256": routing_sha256,
            "cases": [
                {
                    "id": "phone-question",
                    "input_file": golden_input_path.name,
                    "input_sha256": self.sha256(golden_input_path),
                    "family_id": 0,
                    "expected_token_ids": [1, 12, 2],
                },
            ],
        }
        if goldens_mutator is not None:
            goldens_mutator(goldens)
        self.write_json(goldens_path, goldens)

        contract = {
            "format": normalizer.CONVERSION_CONTRACT_FORMAT,
            "source": {
                "manifest_sha256": manifest_sha256,
                "weights_sha256": weights_sha256,
            },
            "graph": {
                "file": graph_path.name,
                "sha256": graph_sha256,
            },
            "activation_calibration": {
                "file": calibration_path.name,
                "sha256": calibration_sha256,
            },
            "preprocessing": {
                "file": preprocessing_path.name,
                "sha256": preprocessing_sha256,
            },
            "routing": routing,
            "goldens": {
                "file": goldens_path.name,
                "sha256": self.sha256(goldens_path),
            },
        }
        self.write_json(contract_path, contract)
        return contract_path

    def test_normalizes_conv_byte_order_keeps_linear_bytes_and_preserves_float32(self):
        self.write_fixture()
        result = self.normalize()

        self.assertEqual(result.quantized_tensor_count, 2)
        self.assertEqual(result.float32_tensor_count, 1)
        output = load_file(str(self.output_path))
        conv_name = "tiny_receipt_vqa.int8.vision.conv.weight"
        linear_name = "tiny_receipt_vqa.int8.decoder.linear.weight"
        bias_name = "tiny_receipt_vqa.float32.decoder.linear.bias"
        self.assertEqual(set(output), {conv_name, linear_name, bias_name})
        self.assertEqual(output[conv_name].dtype, np.dtype(np.int8))
        expected_conv = np.ascontiguousarray(np.transpose(self.conv, (0, 2, 3, 1)))
        np.testing.assert_array_equal(output[conv_name], expected_conv)
        self.assertEqual(output[conv_name].tobytes(), expected_conv.tobytes())

        # Linear is rank 2, so its raw I8 order is not changed or dequantized.
        np.testing.assert_array_equal(output[linear_name], self.linear)
        self.assertEqual(output[linear_name].dtype, np.dtype(np.int8))
        self.assertEqual(output[linear_name].tobytes(), self.linear.tobytes())
        np.testing.assert_array_equal(output[bias_name], self.bias)
        self.assertEqual(output[bias_name].dtype, np.dtype(np.float32))

        fragment = json.loads(self.fragment_path.read_text(encoding="utf-8"))
        self.assertEqual(fragment["format"], normalizer.FRAGMENT_FORMAT)
        self.assertEqual(fragment["source_contract"]["format"], normalizer.SOURCE_FORMAT)
        self.assertEqual(fragment["source_contract"]["tensor_layout"], normalizer.SOURCE_LAYOUT)
        self.assertEqual(
            fragment["normalized_tensors"]["vision.conv.weight"]["transform"],
            "OIHW_to_OHWI",
        )
        self.assertEqual(
            fragment["normalized_tensors"]["decoder.linear.weight"]["transform"],
            "identity",
        )
        self.assertEqual(
            fragment["weights_quantization"][conv_name],
            {
                "scheme": "per_axis",
                "axis": 0,
                "scales": [0.25, 0.5],
                "zero_points": [0, 0],
            },
        )
        self.assertNotIn("nodes", fragment)
        self.assertNotIn("activation_calibration", fragment)

    def test_rejects_missing_named_scale_and_unselected_rank4_conv(self):
        tensors = {
            "quantized.vision.conv.weight": self.conv,
            "scale.vision.conv.weight": self.conv_scale,
            "quantized.decoder.linear.weight": self.linear,
            "float32.decoder.linear.bias": self.bias,
        }
        self.write_fixture(tensors=tensors)
        with self.assertRaisesRegex(
            normalizer.ManifestValidationError,
            "missing named entries.*scale.decoder.linear.weight",
        ):
            self.normalize()

        self.write_fixture()
        with self.assertRaisesRegex(normalizer.ManifestValidationError, "requires an explicit --conv-key"):
            self.normalize(conv_keys=())

    def test_rejects_actual_dtype_and_shape_mismatches(self):
        wrong_dtype = {
            "quantized.vision.conv.weight": self.conv.astype(np.uint8),
            "scale.vision.conv.weight": self.conv_scale,
            "quantized.decoder.linear.weight": self.linear,
            "scale.decoder.linear.weight": self.linear_scale,
            "float32.decoder.linear.bias": self.bias,
        }
        self.write_fixture(tensors=wrong_dtype)
        with self.assertRaisesRegex(normalizer.ManifestValidationError, "quantized.vision.conv.weight.*dtype"):
            self.normalize()

        wrong_shape = {
            "quantized.vision.conv.weight": self.conv[:, :, :, :1],
            "scale.vision.conv.weight": self.conv_scale,
            "quantized.decoder.linear.weight": self.linear,
            "scale.decoder.linear.weight": self.linear_scale,
            "float32.decoder.linear.bias": self.bias,
        }
        self.write_fixture(tensors=wrong_shape)
        with self.assertRaisesRegex(normalizer.ManifestValidationError, "quantized.vision.conv.weight.*shape"):
            self.normalize()

    def test_rejects_malformed_published_layout(self):
        manifest = release_manifest(self.tensors)
        manifest["safetensors"]["layout"] = {
            "int8_per_out": "legacy_offset_blob",
        }
        self.write_fixture(manifest=manifest)
        with self.assertRaisesRegex(normalizer.ManifestValidationError, "published TinyReceiptVQA SafeTensors layout"):
            self.normalize()

    def test_hash_bound_external_contract_is_attested_but_does_not_construct_a_graph(self):
        contract_path = self.write_conversion_contract()
        result = self.normalize(conversion_contract_path=contract_path)

        self.assertTrue(result.conversion_contract_validated)
        fragment = json.loads(self.fragment_path.read_text(encoding="utf-8"))
        validation = fragment["conversion_contract_validation"]
        self.assertEqual(validation["format"], normalizer.CONVERSION_CONTRACT_FORMAT)
        self.assertIn("did not construct or execute a graph", validation["attestation"])
        self.assertEqual(validation["graph"]["activation_edge_count"], 2)
        self.assertEqual(validation["activation_calibration"]["sample_count"], 3)
        self.assertEqual(validation["routing"], {
            "mode": "explicit_family_v1",
            "family_order": list(normalizer.EXPLICIT_FAMILY_ORDER),
            "route_scope": "whole_execution",
            "batch_policy": "homogeneous",
        })
        self.assertEqual(validation["goldens"]["case_count"], 1)
        self.assertNotIn("nodes", fragment)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = normalizer.main([
                "--manifest", str(self.manifest_path),
                "--out", str(self.output_path),
                "--quantization-out", str(self.fragment_path),
                "--conversion-contract", str(contract_path),
                "--conv-key", "vision.conv.weight",
            ])
        self.assertEqual(exit_code, 0)
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["conversion_contract_validated"])
        self.assertFalse(report["graph_constructed"])

    def test_rejects_contract_with_unbound_source_weights(self):
        contract_path = self.write_conversion_contract()
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["source"]["weights_sha256"] = "0" * 64
        self.write_json(contract_path, contract)

        with self.assertRaisesRegex(normalizer.ManifestValidationError, "source.weights_sha256.*SafeTensors"):
            self.normalize(conversion_contract_path=contract_path)

    def test_rejects_calibration_without_full_graph_descriptor_coverage(self):
        contract_path = self.write_conversion_contract(
            calibration_mutator=lambda calibration: calibration["edges"].pop(),
        )

        with self.assertRaisesRegex(normalizer.ManifestValidationError, "missing graph activation edge descriptors"):
            self.normalize(conversion_contract_path=contract_path)

    def test_rejects_unusable_preprocessing_or_automatic_routing(self):
        preprocessing_contract = self.write_conversion_contract(
            preprocessing_mutator=lambda preprocessing: preprocessing["output"].__setitem__("scale", 0.0),
        )
        with self.assertRaisesRegex(normalizer.ManifestValidationError, "output.scale must be non-zero"):
            self.normalize(conversion_contract_path=preprocessing_contract)

        routing_contract = self.write_conversion_contract(
            routing_mutator=lambda routing: routing.__setitem__("mode", "auto"),
        )
        with self.assertRaisesRegex(normalizer.ManifestValidationError, "routing contract.mode"):
            self.normalize(conversion_contract_path=routing_contract)

    def test_rejects_empty_goldens_and_mutated_hash_bound_artifact(self):
        empty_goldens_contract = self.write_conversion_contract(
            goldens_mutator=lambda goldens: goldens.__setitem__("cases", []),
        )
        with self.assertRaisesRegex(
            normalizer.ManifestValidationError,
            "goldens contract.cases must be a non-empty array",
        ):
            self.normalize(conversion_contract_path=empty_goldens_contract)

        artifact_contract = self.write_conversion_contract()
        graph_path = self.root / "graph.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        graph["id"] = "mutated-after-contract"
        self.write_json(graph_path, graph)
        with self.assertRaisesRegex(normalizer.ManifestValidationError, "conversion contract.graph SHA-256 mismatch"):
            self.normalize(conversion_contract_path=artifact_contract)

    def test_cli_smoke_writes_weights_and_fragment_only(self):
        self.write_fixture()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = normalizer.main([
                "--manifest", str(self.manifest_path),
                "--out", str(self.output_path),
                "--quantization-out", str(self.fragment_path),
                "--conv-key", "vision.conv.weight",
            ])
        self.assertEqual(exit_code, 0)
        report = json.loads(stdout.getvalue())
        self.assertTrue(self.output_path.is_file())
        self.assertTrue(self.fragment_path.is_file())
        self.assertFalse(report["graph_constructed"])
        self.assertFalse(report["activation_calibration_constructed"])


if __name__ == "__main__":
    unittest.main()
