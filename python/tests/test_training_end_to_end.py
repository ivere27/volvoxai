"""Trainer rollback integration through the generated public schema client.

The deterministic fixture and independent SGD oracle are shared with the
native CLI training smoke.  This test deliberately uses only generated
VxInferenceService/VxTrainingService requests over Synurang's module call
transport; it does not reach into the native trainer lifecycle.
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import numpy as np  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

import test_native_gpu_training_smoke as fixture  # noqa: E402
import volvoxai  # noqa: E402


def full_library_available() -> bool:
    try:
        volvoxai.find_library("full")
    except volvoxai.VolvoxAIError:
        return False
    return True


@unittest.skipUnless(full_library_available(), "libvolvoxai-full.so is not built")
class TrainingRollbackThroughTheSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp(prefix="volvoxai-training-e2e-"))
        cls.paths = fixture.write_fixture(cls.directory)
        cls.host = volvoxai.open_library("full")
        cls.inference = volvoxai.VxInferenceServiceClient(cls.host)
        cls.training = volvoxai.VxTrainingServiceClient(cls.host)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.close()
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_train_then_rollback_restores_the_pinned_baseline(self) -> None:
        runtime = self.inference.create_runtime(volvoxai.pb.CreateRuntimeRequest())
        volvoxai.check(runtime.report, "CreateRuntime")
        model = trainer = None
        try:
            model = self.inference.load_model(volvoxai.pb.LoadModelRequest(
                runtime_id=runtime.runtime_id,
                graph_path=str(self.paths["graph"]),
                weight_paths=[str(self.paths["weights"])],
            ))
            volvoxai.check(model.report, "LoadModel")
            trainer = self.training.create_trainer(volvoxai.pb.CreateTrainerRequest(
                model_id=model.model_id,
                backend="cpu",
                rng_seed=0,
            ))
            volvoxai.check(trainer.report, "CreateTrainer")

            step = self.training.train_step(volvoxai.pb.TrainStepRequest(
                trainer_id=trainer.trainer_id,
                inputs=[volvoxai.pb.Tensor(
                    name="x",
                    shape=[1, 2],
                    dtype=volvoxai.pb.DataType.DATA_TYPE_F32,
                    inline=np.ascontiguousarray(fixture.INPUT).tobytes(),
                )],
                losses=[volvoxai.pb.CrossEntropyLoss(
                    name="cross_entropy",
                    logits_name="logits",
                    targets=[0],
                    ignore_index=-100,
                    row_index=-1,
                )],
                trainable_names=["parameter"],
                optimizer=volvoxai.pb.TrainerOptimizerOptions(
                    kind=volvoxai.pb.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD,
                    learning_rate=fixture.LEARNING_RATE,
                ),
                accumulation_steps=1,
                flush_accumulation=True,
            ))
            volvoxai.check(step.report, "TrainStep")

            expected_weight, _, expected_loss = fixture.expected_update()
            self.assertEqual(step.microbatch_id, 1)
            self.assertEqual(step.optimizer_step, 1)
            self.assertEqual(step.accumulated_microbatches, 1)
            self.assertTrue(step.update_applied)
            self.assertEqual(step.backend, "cpu")
            self.assertTrue(math.isfinite(step.loss))
            self.assertAlmostEqual(step.loss, expected_loss, delta=2e-5)

            dirty_path = self.directory / "trainer-dirty.safetensors"
            exported = self.training.export_trainer_weights(
                volvoxai.pb.ExportTrainerWeightsRequest(
                    trainer_id=trainer.trainer_id,
                    output_paths=[str(dirty_path)],
                ))
            volvoxai.check(exported.report, "ExportTrainerWeights(dirty)")
            dirty = load_file(dirty_path)
            self.assertEqual(set(dirty), {"parameter"})
            np.testing.assert_allclose(
                dirty["parameter"], expected_weight, rtol=2e-5, atol=2e-6)
            self.assertFalse(np.array_equal(
                dirty["parameter"], fixture.INITIAL_WEIGHT))

            rolled_back = self.training.rollback_trainer(
                volvoxai.pb.TrainerRef(trainer_id=trainer.trainer_id))
            volvoxai.check(rolled_back, "RollbackTrainer")

            restored_path = self.directory / "trainer-restored.safetensors"
            exported = self.training.export_trainer_weights(
                volvoxai.pb.ExportTrainerWeightsRequest(
                    trainer_id=trainer.trainer_id,
                    output_paths=[str(restored_path)],
                ))
            volvoxai.check(exported.report, "ExportTrainerWeights(restored)")
            restored = load_file(restored_path)
            self.assertEqual(set(restored), {"parameter"})
            np.testing.assert_array_equal(
                restored["parameter"], fixture.INITIAL_WEIGHT)
        finally:
            if trainer is not None:
                self.training.release_trainer(
                    volvoxai.pb.TrainerRef(trainer_id=trainer.trainer_id))
            if model is not None:
                self.inference.release_model(
                    volvoxai.pb.ModelRef(model_id=model.model_id))
            self.inference.release_runtime(
                volvoxai.pb.RuntimeRef(runtime_id=runtime.runtime_id))


if __name__ == "__main__":
    unittest.main()
