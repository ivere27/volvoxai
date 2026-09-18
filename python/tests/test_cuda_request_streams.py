"""Concurrent native owners: numerical, snapshot, reuse and teardown regressions.

Run with VOLVOXAI_TEST_CUDA=1 on a physical device. Stream identity and actual
GPU overlap are measured separately by benchmarks/cuda_request_streams.py.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
from safetensors.numpy import save_file
import volvoxai as vx


@unittest.skipUnless(os.environ.get("VOLVOXAI_TEST_CUDA") == "1", "requires physical CUDA")
class CudaRequestStreamsTest(unittest.TestCase):
    def setUp(self):
        import torch  # Initialize the optional integration before the native driver.
        self.assertTrue(torch.cuda.is_available())
        self.directory = tempfile.TemporaryDirectory(prefix="vx-request-streams-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.weight = np.random.default_rng(31).normal(0, .1, (16, 8)).astype(np.float32)
        self.x = np.random.default_rng(73).normal(0, .1, (8, 16)).astype(np.float32)
        self.labels = np.arange(8, dtype=np.int32)
        graph = {"format": "volvox-graph/v1", "dimensions": {},
            "inputs": {"x": {"shape": [8, 16], "dtype": "float32"}},
            "nodes": [{"id": "projection", "opType": "Linear",
                "inputs": {"input": "x", "weight": "weight"},
                "outputs": {"out": {"tensor": "logits", "shape": [8, 8], "dtype": "float32"}},
                "params": {"weight_layout": "din_dout"}}], "outputs": ["logits"]}
        (self.path / "graph.json").write_text(json.dumps(graph))
        save_file({"weight": self.weight}, self.path / "model.safetensors")

    def inference_pair(self):
        with ExitStack() as stack:
            sessions = [stack.enter_context(vx.InferenceSession(self.path, backend="cuda",
                cpu_threads=1)) for _ in range(2)]
            for session in sessions:
                for _ in range(4):
                    session.run_tensors(self.x).close()  # Capture and warm the per-context pool.
            barrier = threading.Barrier(2)
            def run(index):
                session = sessions[index]
                x = self.x + np.float32(index)
                barrier.wait(timeout=30)
                retained = []
                try:
                    # Keep many snapshots alive while the peer also allocates and retires.
                    for step in range(32):
                        result = session.run_tensors(x if step == 0 else {},
                            reuse_inputs=[] if step == 0 else ["x"])
                        retained.append(result)
                    session.close()
                    # Every snapshot survives its producer context and the peer's cleanup.
                    for result in retained:
                        np.testing.assert_allclose(result["logits"].numpy(), x @ self.weight,
                                                   atol=2e-6, rtol=2e-5)
                finally:
                    for result in retained:
                        result.close()
            with ThreadPoolExecutor(2) as executor:
                futures = [executor.submit(run, i) for i in range(2)]
                for future in futures:
                    future.result(timeout=90)

    def test_inference_contexts_and_retained_outputs(self):
        self.inference_pair()

    def test_two_trainers_and_inference_match_independent_sgd(self):
        with ExitStack() as stack:
            inference = stack.enter_context(vx.InferenceSession(self.path, backend="cuda",
                cpu_threads=1))
            trainers = [stack.enter_context(vx.TrainingSession(self.path, backend="cuda",
                cpu_threads=1, loss=vx.CrossEntropyLoss("logits"),
                optimizer=vx.SGD(lr=.05))) for _ in range(2)]
            barrier = threading.Barrier(3)
            def train(index):
                trainer = trainers[index]
                x = self.x + np.float32(index * .1)
                weight = self.weight.copy()
                old = trainer.parameters(["weight"])
                try:
                    barrier.wait(timeout=30)
                    for _ in range(12):
                        logits = x @ weight
                        shifted = logits - logits.max(axis=1, keepdims=True)
                        probabilities = np.exp(shifted)
                        probabilities /= probabilities.sum(axis=1, keepdims=True)
                        loss = -np.log(probabilities[np.arange(8), self.labels]).mean()
                        with trainer.step(x, self.labels) as result:
                            self.assertAlmostEqual(result.loss, float(loss), delta=2e-5)
                        probabilities[np.arange(8), self.labels] -= 1
                        weight -= .05 * (x.T @ probabilities) / 8
                    with trainer.parameters(["weight"]) as current:
                        np.testing.assert_allclose(current["weight"].numpy(), weight, atol=2e-6, rtol=2e-5)
                    trainer.close()
                    np.testing.assert_array_equal(old["weight"].numpy(), self.weight)
                finally:
                    old.close()
            def infer():
                barrier.wait(timeout=30)
                for _ in range(30):
                    with inference.run_tensors(self.x) as result:
                        np.testing.assert_allclose(result["logits"].numpy(), self.x @ self.weight,
                                                   atol=2e-6, rtol=2e-5)
            with ThreadPoolExecutor(3) as executor:
                futures = [executor.submit(train, i) for i in range(2)] + [executor.submit(infer)]
                for future in futures:
                    future.result(timeout=90)


if __name__ == "__main__":
    unittest.main()
