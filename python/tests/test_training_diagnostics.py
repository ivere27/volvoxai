"""TrainingSession numerical diagnostics over the native full library.

The same C engine code is covered for WASM by tests/training_diagnostics.test.mjs;
this test drives the native build through the Python adapter. Set
VOLVOXAI_TEST_NATIVE_GPU_BACKEND=cuda/vulkan/opengl to repeat it on a GPU Trainer.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

import numpy as np  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

import volvoxai as vx  # noqa: E402

D, H, C = 3, 4, 2
TRAINABLE = ["fc1.w", "fc1.b", "fc2.w", "fc2.b"]


def library_available() -> bool:
    try:
        vx.find_library()
    except vx.VolvoxAIError:
        return False
    return True


def write_package(directory: Path) -> Path:
    graph = {
        "format": "volvox-graph/v1", "dimensions": {}, "inputs": {"x": {"dtype": "float32", "shape": [2, D]}},
        "nodes": [
            {"id": "fc1", "opType": "Linear", "inputs": {"input": "x", "weight": "fc1.w", "bias": "fc1.b"},
             "outputs": {"out": {"tensor": "h", "dtype": "float32", "shape": [2, H]}},
             "params": {"weight_layout": "dout_din"}},
            {"id": "act", "opType": "GELU", "inputs": {"input": "h"},
             "outputs": {"out": {"tensor": "r", "dtype": "float32", "shape": [2, H]}}, "params": {}},
            {"id": "fc2", "opType": "Linear", "inputs": {"input": "r", "weight": "fc2.w", "bias": "fc2.b"},
             "outputs": {"out": {"tensor": "logits", "dtype": "float32", "shape": [2, C]}},
             "params": {"weight_layout": "dout_din"}},
        ],
        "outputs": ["logits"],
    }
    values = lambda count, offset: (np.cos(np.arange(count) * 1.3 + offset) * .5).astype(np.float32)  # noqa: E731
    package = directory / "package"
    package.mkdir()
    (package / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    save_file({"fc1.w": values(H * D, 0).reshape(H, D), "fc1.b": values(H, 1),
               "fc2.w": values(C * H, 2).reshape(C, H), "fc2.b": values(C, 3)},
              str(package / "model.safetensors"))
    return package


@unittest.skipUnless(library_available(), "libvolvoxai.so is not built")
class TrainingDiagnosticsTest(unittest.TestCase):
    backend = "cpu"

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp(prefix="volvoxai-training-diagnostics-"))
        cls.package = write_package(cls.directory)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.directory, ignore_errors=True)

    def session(self):
        return vx.TrainingSession(self.package, backend=self.backend,
                                  loss=vx.CrossEntropyLoss(output="logits", ignore_index=-1),
                                  optimizer=vx.SGD(lr=0.1), trainable_names=TRAINABLE)

    def test_statistics_report_norms_and_updates(self) -> None:
        x = np.array([[.3, -.2, .8], [-.5, .1, .4]], dtype=np.float32)
        with self.session() as trainer:
            plain = trainer.step(x, [0, 1])
            self.assertIsNone(plain.gradients)
            result = trainer.step(x, [0, 1], statistics=True)
        gradients = result.gradients
        self.assertEqual([item.name for item in gradients.parameters], TRAINABLE)
        norms = np.array([item.gradient_norm for item in gradients.parameters])
        self.assertAlmostEqual(float(np.sqrt((norms ** 2).sum())), gradients.global_norm, places=9)
        for item in gradients.parameters:
            self.assertEqual(item.nonfinite_count, 0)
            self.assertAlmostEqual(item.update_norm, 0.1 * item.gradient_norm, places=6)

    def test_nonfinite_gradient_names_parameter_and_node(self) -> None:
        # Row 1 is ignored by the loss; its NaN reaches the weight gradients as 0 * NaN.
        x = np.array([[.3, -.2, .8], [-.5, np.nan, .4]], dtype=np.float32)
        with self.session() as trainer:
            with self.assertRaises(vx.VolvoxAIError) as caught:
                trainer.step(x, [0, -1], locate_nonfinite=True)
            error = caught.exception
            self.assertEqual(error.code, vx.pb.OperationCode.OPERATION_CODE_TRAINING_GRADIENT_NONFINITE)
            self.assertEqual(error.response.gradients.first_nonfinite_parameter, "fc1.w")
            if self.backend == "cpu":  # Only a CPU Trainer locates the node.
                self.assertEqual(error.node, 2)  # fc2's backward runs first.
                self.assertIn("backward node 2", str(error))
            # The trainer restored its private state and keeps training.
            trainer.step(np.nan_to_num(x), [0, 1])

    def test_nonfinite_loss_names_the_loss(self) -> None:
        x = np.array([[.3, np.nan, .8], [-.5, .1, .4]], dtype=np.float32)
        with self.session() as trainer:
            with self.assertRaises(vx.VolvoxAIError) as caught:
                trainer.step(x, [0, 1], locate_nonfinite=True)
        error = caught.exception
        self.assertEqual(error.code, vx.pb.OperationCode.OPERATION_CODE_TRAINING_LOSS_NONFINITE)
        self.assertEqual(error.response.gradients.nonfinite_loss, "cross_entropy")
        if self.backend == "cpu":
            self.assertEqual(error.node, 0)


@unittest.skipUnless(os.environ.get("VOLVOXAI_TEST_NATIVE_GPU_BACKEND"), "set VOLVOXAI_TEST_NATIVE_GPU_BACKEND on physical hardware")
class GpuTrainingDiagnosticsTest(TrainingDiagnosticsTest):
    backend = os.environ.get("VOLVOXAI_TEST_NATIVE_GPU_BACKEND", "")

    def test_statistics_match_the_cpu_trainer(self) -> None:
        x = np.array([[.3, -.2, .8], [-.5, .1, .4]], dtype=np.float32)
        results = {}
        for backend in ("cpu", self.backend):
            with vx.TrainingSession(self.package, backend=backend,
                                    loss=vx.CrossEntropyLoss(output="logits", ignore_index=-1),
                                    optimizer=vx.SGD(lr=0.1), trainable_names=TRAINABLE) as trainer:
                result = trainer.step(x, [0, 1], statistics=True)
                self.assertEqual(result.backend, backend)
                results[backend] = result.gradients
        cpu, gpu = results["cpu"], results[self.backend]
        self.assertAlmostEqual(gpu.global_norm, cpu.global_norm, delta=1e-6 * cpu.global_norm)
        for expected, actual in zip(cpu.parameters, gpu.parameters, strict=True):
            self.assertEqual(actual.name, expected.name)
            for field in ("gradient_norm", "gradient_max_abs", "parameter_norm", "update_norm"):
                self.assertAlmostEqual(getattr(actual, field), getattr(expected, field),
                                       delta=1e-5 * getattr(expected, field), msg=f"{expected.name}.{field}")


if __name__ == "__main__":
    unittest.main()
