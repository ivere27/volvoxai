import importlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.exporter.capabilities import classify_package


ROOT = Path(__file__).resolve().parents[3]
EXPORTER = ROOT / "tools" / "export_safetensors.py"
EFFICIENTDET_INT8 = (
    ROOT / "models" / "efficientdet_lite0_int8" / "efficientdet_lite0.tflite"
)
EFFICIENTDET_FP16 = (
    ROOT
    / "models"
    / "efficientdet_lite0_fp16"
    / "efficientdet_lite0_float16.tflite"
)
EFFICIENTDET_FP32 = (
    ROOT
    / "models"
    / "efficientdet_lite0_fp32"
    / "efficientdet_lite0_float32.tflite"
)
TASK_CLI_DEFAULT = ROOT / "examples" / "target" / "bin" / "volvoxai-tasks"
LABELS = ROOT / "examples" / "efficientdet_lite0" / "assets" / "coco_labels.txt"
FIXTURES = {
    "dog.jpg": {
        "index": 19011,
        "class": "17",
        "label": "dog",
        "scores": {"int8": 0.917969, "fp16": 0.910401, "fp32": 0.910444},
    },
    "cat.jpg": {
        "index": 19013,
        "class": "16",
        "label": "cat",
        "scores": {"int8": 0.808594, "fp16": 0.774408, "fp32": 0.775092},
    },
}
MODEL_VARIANTS = {
    "int8": {
        "source": EFFICIENTDET_INT8,
        "normalization": "raw-255",
        "weight_dtype": "auto",
    },
    "fp16": {
        "source": EFFICIENTDET_FP16,
        "normalization": "zero-one",
        "weight_dtype": "float16",
    },
    "fp32": {
        "source": EFFICIENTDET_FP32,
        "normalization": "zero-one",
        "weight_dtype": "float32",
    },
}


def _iter_keys(value):
    if isinstance(value, dict):
        yield from value.keys()
        for child in value.values():
            yield from _iter_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_keys(child)


class GenericExporterOutputNamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location(
            "volvoxai_generic_exporter", EXPORTER
        )
        if spec is None or spec.loader is None:
            raise AssertionError(f"cannot load exporter: {EXPORTER}")
        cls.exporter = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(cls.exporter)
        except ImportError as error:
            raise unittest.SkipTest(
                f"generic exporter dependencies are unavailable: {error}"
            ) from error

    def test_default_output_names_are_positional(self) -> None:
        self.assertEqual(
            self.exporter._resolve_output_names(3),
            ["output0", "output1", "output2"],
        )

    def test_explicit_output_names_are_validated(self) -> None:
        self.assertEqual(
            self.exporter._resolve_output_names(2, ["scores", "boxes"]),
            ["scores", "boxes"],
        )
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            self.exporter._resolve_output_names(2, ["only_one"])
        with self.assertRaisesRegex(ValueError, "unique"):
            self.exporter._resolve_output_names(2, ["same", "same"])

    def test_generic_exporter_has_no_detection_output_heuristic(self) -> None:
        source = EXPORTER.read_text(encoding="utf-8")
        self.assertNotIn('"boxes" if shape', source)
        self.assertNotIn('"scores" if shape', source)

    def test_image_normalization_is_rejected_as_application_policy(self) -> None:
        inputs = {
            "input0": {"shape": [1, 320, 320, 3], "dtype": "float32"},
            "input1": {"shape": [1], "dtype": "int32"},
        }
        before = json.loads(json.dumps(inputs))
        with self.assertRaisesRegex(ValueError, "application preprocessing"):
            self.exporter._apply_image_normalizations(inputs, ["input0=zero-one"])
        self.assertEqual(inputs, before)


class DirectTfliteW8A8ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The model exporter is deliberately a development-only Python tool.
        # Keep this smoke test out of minimal runtime environments. The source
        # model lives under the ignored models/ directory and is fetched by the
        # example's model target, so a clean checkout skips this opt-in test.
        try:
            for module in ("numpy", "flatbuffers", "safetensors", "torch"):
                importlib.import_module(module)
            cls.torch = importlib.import_module("torch")
            cls.safetensors_torch = importlib.import_module("safetensors.torch")
        except ImportError as error:
            raise unittest.SkipTest(
                f"direct TFLite exporter dependencies are unavailable: {error}"
            ) from error

        if not EXPORTER.is_file():
            raise AssertionError(f"missing exporter: {EXPORTER}")
        if not EFFICIENTDET_INT8.is_file():
            raise unittest.SkipTest(
                "EfficientDet INT8 source model has not been fetched: "
                f"{EFFICIENTDET_INT8}"
            )

    def assert_typed_quantized_output(self, graph, tensors, node) -> None:
        output_descriptor = node.get("outputs", {}).get("out", {})
        dtype = output_descriptor.get("dtype")
        self.assertIn(dtype, ("int8", "uint8"), node)
        output = output_descriptor.get("tensor")
        quantization = graph.get("quantization", {}).get("tensors", {}).get(output)
        self.assertIsInstance(quantization, dict, node)
        self.assertEqual(quantization.get("scheme"), "per_tensor", node)
        scale = tensors[quantization["scale_tensor"]]
        zero_point = tensors[quantization["zero_point_tensor"]]
        self.assertEqual(scale.dtype, self.torch.float32, node)
        self.assertEqual(scale.numel(), 1, node)
        self.assertTrue(math.isfinite(float(scale.item())), node)
        self.assertGreater(float(scale.item()), 0.0, node)
        self.assertEqual(zero_point.numel(), 1, node)
        zero_value = int(zero_point.item())
        if dtype == "int8":
            self.assertEqual(zero_point.dtype, self.torch.int8, node)
            self.assertGreaterEqual(zero_value, -128, node)
            self.assertLessEqual(zero_value, 127, node)
        else:
            self.assertEqual(zero_point.dtype, self.torch.uint8, node)
            self.assertGreaterEqual(zero_value, 0, node)
            self.assertLessEqual(zero_value, 255, node)

    def test_efficientdet_direct_export_is_canonical_hybrid_int8(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volvoxai-tflite-export-") as temporary:
            output_dir = Path(temporary)
            output_path = output_dir / "model.safetensors"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(EXPORTER),
                    "--model",
                    str(EFFICIENTDET_INT8),
                    "--out",
                    str(output_path),
                    "--output-name",
                    "scores",
                    "--output-name",
                    "boxes",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"exporter failed:\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )

            graph_path = output_dir / "graph.json"
            self.assertTrue(output_path.is_file())
            self.assertTrue(graph_path.is_file())
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            tensors = self.safetensors_torch.load_file(str(output_path), device="cpu")

        self.assertEqual(graph.get("format"), "volvox-graph/v1")
        self.assertNotIn("shape_system", graph)
        self.assertEqual(graph.get("dimensions"), {})
        self.assertNotIn("image_normalization", graph.get("inputs", {}).get("input0", {}))
        self.assertEqual(graph.get("outputs"), ["scores", "boxes"])
        self.assertEqual(classify_package(graph, tensors), "hybrid")
        self.assertNotIn("source", graph)
        self.assertGreater(len(tensors), 0)
        self.assertNotIn("input_scale", set(_iter_keys(graph)))
        self.assertNotIn("weight_scale", set(_iter_keys(graph)))

        nodes = graph.get("nodes", [])
        requantize_nodes = [node for node in nodes if node.get("opType") == "RequantizeLinear"]
        qconv_nodes = [node for node in nodes if node.get("opType") == "QConv2D"]
        qadd_nodes = [node for node in nodes if node.get("opType") == "QAdd"]
        sigmoid_nodes = [node for node in nodes if node.get("opType") == "Sigmoid"]
        self.assertGreater(len(requantize_nodes), 0)
        self.assertGreater(len(qconv_nodes), 0)
        self.assertGreater(len(qadd_nodes), 0)
        self.assertEqual(len(sigmoid_nodes), 1)

        affine_bridge_nodes = [
            node for node in nodes
            if node.get("opType") in {
                "QuantizeLinear", "DequantizeLinear", "RequantizeLinear",
            }
        ]
        self.assertGreater(len(affine_bridge_nodes), 0)
        for node in affine_bridge_nodes:
            self.assertEqual(
                node.get("params"), {},
                "affine conversion nodes must not carry retired layout metadata",
            )

        for node in requantize_nodes + qconv_nodes + qadd_nodes:
            self.assert_typed_quantized_output(graph, tensors, node)

        quantization = graph.get("quantization")
        self.assertEqual(quantization.get("format"), "volvox-affine-safetensors/v1")
        quantized_tensors = quantization.get("tensors")
        self.assertIsInstance(quantized_tensors, dict)
        self.assertGreater(len(quantized_tensors), 0)
        self.assertNotIn("weights_quantization", graph)
        self.assertNotIn("weights_quantization_storage", graph)

        for node in qconv_nodes:
            params = node.get("params", {})
            self.assertEqual(params.get("data_layout"), "NHWC", node)
            self.assertEqual(params.get("weight_layout"), "OHWI", node)
            inputs = node.get("inputs", {})
            self.assertIn("weight", inputs, node)
            self.assertIn("bias", inputs, node)

            weight = tensors[inputs["weight"]]
            bias = tensors[inputs["bias"]]
            self.assertIn(weight.dtype, (self.torch.int8, self.torch.uint8), node)
            self.assertEqual(bias.dtype, self.torch.int32, node)
            self.assertEqual(weight.ndim, 4, node)
            self.assertEqual(bias.numel(), weight.shape[0], node)

            descriptor = quantized_tensors.get(inputs["weight"])
            self.assertIsInstance(descriptor, dict, node)
            self.assertEqual(descriptor.get("scheme"), "per_axis", node)
            self.assertEqual(descriptor.get("axis"), 0, node)
            scales = tensors[descriptor["scale_tensor"]]
            zero_points = tensors[descriptor["zero_point_tensor"]]
            self.assertEqual(scales.dtype, self.torch.float32, node)
            self.assertEqual(scales.numel(), weight.shape[0], node)
            self.assertEqual(zero_points.dtype, weight.dtype, node)
            self.assertEqual(zero_points.numel(), weight.shape[0], node)


class EfficientDetNativeEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            for module in ("numpy", "flatbuffers", "safetensors", "torch"):
                importlib.import_module(module)
        except ImportError as error:
            raise unittest.SkipTest(
                f"direct TFLite exporter dependencies are unavailable: {error}"
            ) from error

        explicit_cli = os.environ.get("VOLVOXAI_TASK_CLI")
        cls.task_cli = Path(explicit_cli).expanduser() if explicit_cli else TASK_CLI_DEFAULT
        if not cls.task_cli.is_file():
            if explicit_cli:
                raise AssertionError(f"VOLVOXAI_TASK_CLI does not exist: {cls.task_cli}")
            raise unittest.SkipTest(
                "native task CLI has not been built; run make build_native_task_cli"
            )

        if not LABELS.is_file():
            raise AssertionError(f"missing COCO labels fixture: {LABELS}")
        for filename in FIXTURES:
            image = LABELS.parent / filename
            if not image.is_file():
                raise AssertionError(f"missing EfficientDet image fixture: {image}")

    def assert_fresh_export_detects_dog_and_cat(self, variant_name: str) -> None:
        variant = MODEL_VARIANTS[variant_name]
        source = variant["source"]
        if not source.is_file():
            self.skipTest(
                f"EfficientDet {variant_name} source model has not been fetched: {source}"
            )

        with tempfile.TemporaryDirectory(
            prefix=f"volvoxai-efficientdet-{variant_name}-e2e-"
        ) as temporary:
            model_dir = Path(temporary)
            output_path = model_dir / "model.safetensors"
            exported = subprocess.run(
                [
                    sys.executable,
                    str(EXPORTER),
                    "--model",
                    str(source),
                    "--out",
                    str(output_path),
                    "--weight-dtype",
                    variant["weight_dtype"],
                    "--output-name",
                    "scores",
                    "--output-name",
                    "boxes",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(
                exported.returncode,
                0,
                f"exporter failed:\nstdout:\n{exported.stdout}\nstderr:\n{exported.stderr}",
            )
            graph = json.loads((model_dir / "graph.json").read_text(encoding="utf-8"))
            self.assertNotIn(
                "image_normalization",
                graph.get("inputs", {}).get("input0", {}),
            )
            shutil.copyfile(LABELS, model_dir / "labels.txt")

            for filename, expected in FIXTURES.items():
                with self.subTest(image=filename):
                    detected = subprocess.run(
                        [
                            str(self.task_cli),
                            "detect",
                            str(model_dir),
                            "--image",
                            f"input0={LABELS.parent / filename}",
                            "--image-normalize",
                            variant["normalization"],
                            "--max-det",
                            "1",
                        ],
                        cwd=ROOT,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    self.assertEqual(
                        detected.returncode,
                        0,
                        "native detection failed:"
                        f"\nstdout:\n{detected.stdout}\nstderr:\n{detected.stderr}",
                    )

                    lines = detected.stdout.splitlines()
                    header = "rank\tindex\tscore\tscore_pct\tclass\tlabel\tx0\ty0\tx1\ty1"
                    self.assertIn(header, lines, detected.stdout)
                    row_index = lines.index(header) + 1
                    self.assertLess(row_index, len(lines), detected.stdout)
                    values = lines[row_index].split("\t")
                    self.assertEqual(len(values), len(header.split("\t")), detected.stdout)
                    detection = dict(zip(header.split("\t"), values))

                    self.assertEqual(int(detection["index"]), expected["index"])
                    self.assertEqual(detection["class"], expected["class"])
                    self.assertEqual(detection["label"], expected["label"])
                    self.assertAlmostEqual(
                        float(detection["score"]),
                        expected["scores"][variant_name],
                        delta=0.02,
                    )

    def test_fresh_int8_export_detects_dog_and_cat(self) -> None:
        self.assert_fresh_export_detects_dog_and_cat("int8")

    def test_fresh_fp16_export_detects_dog_and_cat(self) -> None:
        self.assert_fresh_export_detects_dog_and_cat("fp16")

    def test_fresh_fp32_export_detects_dog_and_cat(self) -> None:
        self.assert_fresh_export_detects_dog_and_cat("fp32")


if __name__ == "__main__":
    unittest.main()
