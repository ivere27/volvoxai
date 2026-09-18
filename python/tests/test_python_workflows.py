"""CPU workflow qualification, also runnable outside a source checkout."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import weakref

if not sys.flags.isolated:
    source = Path(__file__).resolve().parent.parent
    if (source / "volvoxai/__init__.py").is_file():
        sys.path.insert(0, str(source))

import numpy as np
from safetensors.numpy import load_file, save_file
import volvoxai as vx
from volvoxai import _arrays, _async_session

pb = vx.pb


class Fixture:
    def setUp(self):
        super().setUp()
        try:
            vx.find_library()
        except vx.VolvoxAIError as error:
            self.skipTest(str(error))
        temporary = tempfile.TemporaryDirectory(prefix="volvoxai-workflows-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "source"
        self.model.mkdir()
        self.weight = np.array([[.2, -.4], [.1, .3]], dtype=np.float32)
        self.x = np.array([[1., 0.]], dtype=np.float32)
        save_file({"parameter": self.weight}, self.model / "model.safetensors")
        self.graph = {"format": "volvox-graph/v1", "dimensions": {},
            "inputs": {"x": {"dtype": "float32", "shape": [1, 2]}},
            "nodes": [{"id": "projection", "opType": "Linear",
                "inputs": {"input": "x", "weight": "parameter"},
                "outputs": {"out": {"tensor": "logits", "dtype": "float32", "shape": [1, 2]}},
                "params": {"weight_layout": "din_dout"}}], "outputs": ["logits"]}
        self.write_graph()

    def write_graph(self):
        (self.model / "graph.json").write_text(json.dumps(self.graph))

    def expected_sgd(self):
        logits = (self.x @ self.weight).astype(np.float64)[0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        loss = -np.log(probabilities[0])
        gradient = probabilities.copy()
        gradient[0] -= 1
        return self.weight - .1 * np.outer(self.x[0], gradient), loss


class MetadataAndBuffers(Fixture, unittest.TestCase):
    def test_metadata_is_immutable_and_complete(self):
        with vx.InferenceSession(self.model) as session:
            spec = session.inputs[0]
            self.assertEqual((spec.name, spec.shape, spec.dtype), ("x", (1, 2), "float32"))
            with self.assertRaises(FrozenInstanceError):
                spec.name = "changed"
            with self.assertRaises(AttributeError):
                session.inputs = ()
            with self.assertRaises(TypeError):
                spec.constraints["N"] = vx.Dimension(1, 4, 1)
        raw = pb.TensorSpec(name="tokens", dtype=pb.DataType.DATA_TYPE_BF16, dimensions=[
            pb.DimensionConstraint(kind=pb.DimensionKind.DIMENSION_KIND_SYMBOLIC,
                                   symbol="S", min=2, max=16, multiple_of=2)])
        spec = vx.TensorSpec._from_proto(raw)
        raw.dimensions[0].max = 100
        self.assertEqual((spec.shape, spec.dtype), (("S",), "bfloat16"))
        self.assertEqual(spec.constraints["S"], vx.Dimension(2, 16, 2))
        with self.assertRaises(FrozenInstanceError):
            spec.constraints["S"].max = 10
        with self.assertRaisesRegex(TypeError, "no supported NumPy dtype"):
            _arrays.output_array(spec, pb.TensorInfo(shape=[2], dtype=pb.DataType.DATA_TYPE_BF16, byte_size=4))

    def test_host_buffers_skip_inline_payloads_and_static_shape_query(self):
        with vx.InferenceSession(self.model) as session:
            with patch.object(session._engine, "execute", wraps=session._engine.execute) as execute, \
                 patch.object(session._engine, "read_output", wraps=session._engine.read_output) as read, \
                 patch.object(session._engine, "get_result", wraps=session._engine.get_result) as get:
                actual = session.run(self.x)["logits"]
                request = execute.call_args.args[0]
                self.assertEqual(request.inputs[0].borrowed.resource.handle, self.x.ctypes.data)
                self.assertFalse(request.inputs[0].inline)
                self.assertEqual(read.call_args.args[0].into.resource.handle, actual.ctypes.data)
                get.assert_not_called()
                np.testing.assert_allclose(actual, self.x @ self.weight, atol=1e-7)

    def test_scalar_unaligned_and_snapshot_adaptation(self):
        for value in (np.array(3, dtype=np.int32), self.x[:, ::-1], self.x.astype(">f4"),
                      np.ndarray((1, 2), dtype=np.float32, buffer=bytearray(9), offset=1)):
            tensors, owners = _arrays.input_batch(value, ("x",), snapshot=True)
            self.assertEqual(tensors[0].shape, list(value.shape))
            self.assertTrue(owners[0].flags.c_contiguous and owners[0].flags.aligned)
            self.assertFalse(np.shares_memory(value, owners[0]))
            np.testing.assert_array_equal(owners[0], value)

    def test_read_transport_failure_retires_owner(self):
        with vx.InferenceSession(self.model) as session:
            with patch.object(session._engine, "read_output", side_effect=vx.FfiError("lost read")):
                with self.assertRaises(vx.FfiError):
                    session.run(self.x)
            with self.assertRaises(vx.PluginClosedError):
                session.run(self.x)

    def test_inference_import_does_not_import_training_or_exporter(self):
        script = ("import sys; import volvoxai; "
                  "assert 'volvoxai._training' not in sys.modules; "
                  "assert 'volvoxai._quantization' not in sys.modules; "
                  "assert 'onnx' not in sys.modules")
        # Source invocations need the source package path; installed tests use
        # isolated mode to prove there is no checkout fallback.
        if sys.flags.isolated:
            command = [sys.executable, "-I", "-c", script]
        else:
            script = f"import sys; sys.path.insert(0, {str(Path(vx.__file__).parent.parent)!r}); " + script
            command = [sys.executable, "-c", script]
        subprocess.run(command, check=True, capture_output=True)


class AsyncWorkflow(Fixture, unittest.IsolatedAsyncioTestCase):
    async def test_dynamic_shapes_return_only_selected_output(self):
        (self.model / "model.safetensors").unlink()
        shape = ["N", 2]
        self.graph = {"format": "volvox-graph/v1", "dimensions": {"N": {"min": 1, "max": 4}},
            "inputs": {name: {"shape": shape, "dtype": "float32"} for name in ("a", "b")},
            "nodes": [{"id": name, "opType": operator, "inputs": {"a": "a", "b": "b"},
                "outputs": {"out": {"tensor": name, "shape": shape, "dtype": "float32"}},
                "params": {}} for name, operator in (("sum", "Add"), ("product", "Mul"))],
            "outputs": ["sum", "product"]}
        self.write_graph()
        async with vx.AsyncInferenceSession(self.model) as session:
            self.assertEqual(session.inputs[0].shape, ("N", 2))
            self.assertEqual(session.inputs[0].constraints["N"].max, 4)
            for count in (1, 4, 2):
                a = np.arange(count * 2, dtype=np.float32).reshape(count, 2)
                with patch.object(session._engine, "read_output", wraps=session._engine.read_output) as read:
                    result = await session.run({"a": a, "b": a + 2}, output_names=["product"])
                    read.assert_awaited_once()
                    np.testing.assert_array_equal(result["product"], a * (a + 2))

    async def test_async_outputs_errors_and_serialization(self):
        async with vx.AsyncInferenceSession(self.model) as session:
            self.assertEqual(session.inputs[0].dtype, "float32")
            results = await asyncio.gather(*(session.run(self.x * value) for value in (1, 2, 3)))
            for value, outputs in zip((1, 2, 3), results):
                np.testing.assert_allclose(outputs["logits"], self.x * value @ self.weight, atol=1e-7)
            with self.assertRaises(vx.VolvoxAIError):
                await session.run(self.x.astype(np.float64))
            self.assertEqual(list(await session.run(self.x, output_names=["logits"])), ["logits"])
        self.assertTrue(results[0]["logits"].flags.owndata)
        with self.assertRaises(vx.PluginClosedError):
            await session.run(self.x)
        await session.close()

    async def test_cancel_drains_native_job_and_keeps_arrays_alive(self):
        async with vx.AsyncInferenceSession(self.model) as session:
            entered, finish = asyncio.Event(), asyncio.Event()
            real_execute, real_batch = session._engine.execute, _async_session.input_batch
            references = []
            def capture(*args, **kwargs):
                tensors, owners = real_batch(*args, **kwargs)
                references.extend(weakref.ref(owner) for owner in owners)
                return tensors, owners
            async def delayed(request):
                entered.set()
                await finish.wait()
                self.assertTrue(all(reference() is not None for reference in references))
                return await real_execute(request)
            with patch.object(_async_session, "input_batch", side_effect=capture), \
                 patch.object(session._engine, "execute", side_effect=delayed), \
                 patch.object(session._engine, "release_result", wraps=session._engine.release_result) as release:
                job = asyncio.create_task(session.run(self.x.copy()))
                await entered.wait()
                job.cancel()
                await asyncio.sleep(0)
                job.cancel()
                await asyncio.sleep(.005)
                self.assertFalse(job.done())
                finish.set()
                with self.assertRaises(asyncio.CancelledError):
                    await job
                release.assert_awaited_once()
            np.testing.assert_allclose((await session.run(self.x))["logits"], self.x @ self.weight, atol=1e-7)

    async def test_timeout_drains_inflight_work_and_loop_makes_progress(self):
        async with vx.AsyncInferenceSession(self.model) as session:
            real_execute = session._engine.execute
            ticks = []
            async def delayed(request):
                await asyncio.sleep(.035)
                return await real_execute(request)
            async def heartbeat():
                for _ in range(10):
                    await asyncio.sleep(.002)
                    ticks.append(1)
            with patch.object(session._engine, "execute", side_effect=delayed):
                beat = asyncio.create_task(heartbeat())
                started = time.monotonic()
                with self.assertRaises(asyncio.TimeoutError):
                    await session.run(self.x, timeout=.005)
                self.assertGreaterEqual(time.monotonic() - started, .03)
                await beat
                self.assertEqual(len(ticks), 10)
            await session.run(self.x)

    async def test_private_task_cancellation_during_loop_shutdown_closes_owner(self):
        async with vx.AsyncInferenceSession(self.model) as session:
            entered = asyncio.Event()
            private = []
            async def interrupted(request):
                private.append(asyncio.current_task())
                entered.set()
                await asyncio.Event().wait()
            with patch.object(session._engine, "execute", side_effect=interrupted):
                public = asyncio.create_task(session.run(self.x))
                await entered.wait()
                private[0].cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await public
            with self.assertRaises(vx.PluginClosedError):
                await session.run(self.x)

    async def test_queued_cancellation_does_not_submit_another_execution(self):
        async with vx.AsyncInferenceSession(self.model) as session:
            async with session._lock:
                with patch.object(session._engine, "execute", wraps=session._engine.execute) as execute:
                    with self.assertRaises(asyncio.TimeoutError):
                        await session.run(self.x, timeout=.005)
                    execute.assert_not_called()

    async def test_constructor_failure_and_transport_error_close(self):
        host = vx.open_library()
        (self.model / "graph.json").write_text("bad graph")
        with patch.object(_async_session, "open_library", return_value=host):
            with self.assertRaises(vx.VolvoxAIError):
                async with vx.AsyncInferenceSession(self.model):
                    pass
        with self.assertRaises(vx.PluginClosedError):
            vx.VxPlatformServiceClient(host).get_platform_info(pb.Empty())
        self.write_graph()
        async with vx.AsyncInferenceSession(self.model) as session:
            with patch.object(session._engine, "read_output", side_effect=vx.FfiError("lost response")):
                with self.assertRaises(vx.FfiError):
                    await session.run(self.x)
            with self.assertRaises(vx.PluginClosedError):
                await session.run(self.x)


class QuantizationWorkflow(Fixture, unittest.TestCase):
    def test_streaming_ptq_runs_quantized_model_and_preserves_source(self):
        before = (self.model / "model.safetensors").read_bytes()
        previous = None
        def batches():
            nonlocal previous
            for scale in np.linspace(-1, 1, 8, dtype=np.float32):
                # Only the previous completed batch may remain referenced.
                array = self.x * scale
                previous = weakref.ref(array)
                yield array
        samples_per_batch = 2**32 + 1  # The proto counter is uint64, independent of array shape.
        result = vx.quantize(self.model, calibration_data=batches(), output=self.root / "int8",
                             samples_per_batch=samples_per_batch)
        self.assertEqual(result.calibration_batches, 8)
        self.assertEqual(result.calibration_samples, 8 * samples_per_batch)
        self.assertGreater(result.quantized_nodes, 0)
        self.assertIsNone(previous())
        self.assertEqual((self.model / "model.safetensors").read_bytes(), before)
        with vx.InferenceSession(result) as session:
            np.testing.assert_allclose(session.run(self.x)["logits"], self.x @ self.weight, atol=.025)

    def test_empty_invalid_and_interrupted_calibration_publish_nothing(self):
        def interrupted():
            yield self.x
            raise RuntimeError("dataset read failed")
        cases = (([], ValueError), ([self.x.astype(np.float64)], vx.VolvoxAIError),
                 ([{"wrong": self.x}], vx.VolvoxAIError), (interrupted(), RuntimeError))
        for index, (data, error) in enumerate(cases):
            output = self.root / f"invalid-{index}"
            with self.subTest(index=index), self.assertRaises(error):
                vx.quantize(self.model, calibration_data=data, output=output)
            self.assertFalse((output / "graph.json").exists())
            self.assertFalse((output / "model.safetensors").exists())
            self.assertFalse(list(output.glob(".*.export-*")))

    def test_existing_package_is_preserved(self):
        result = vx.quantize(self.model, calibration_data=[self.x], output=self.root / "int8")
        before = result.weight_paths[0].read_bytes()
        with self.assertRaisesRegex(ValueError, "existing"):
            vx.quantize(self.model, calibration_data=[self.x], output=result.path)
        self.assertEqual(result.weight_paths[0].read_bytes(), before)


class TrainingWorkflow(Fixture, unittest.TestCase):
    def trainer(self, **options):
        return vx.TrainingSession(self.model, loss=vx.CrossEntropyLoss("logits"), **options)

    def test_sgd_matches_independent_gradient_and_saved_inference(self):
        before = (self.model / "model.safetensors").read_bytes()
        with self.trainer(optimizer=vx.SGD(lr=.1)) as trainer:
            result = trainer.step(self.x, np.array([0], dtype=np.int64))
            expected, loss = self.expected_sgd()
            self.assertAlmostEqual(result.loss, loss, delta=2e-6)
            self.assertEqual((result.optimizer_step, result.update_applied, result.backend), (1, True, "cpu"))
            self.assertEqual(result.metrics[0].examples, 1)
            package = trainer.save(self.root / "trained")
            np.testing.assert_allclose(load_file(package.weight_paths[0])["parameter"], expected, atol=2e-6)
            with vx.InferenceSession(package) as session:
                np.testing.assert_allclose(session.run(self.x)["logits"], self.x @ expected, atol=2e-6)
            trainer.rollback()
            restored = trainer.save(self.root / "restored")
            np.testing.assert_array_equal(load_file(restored.weight_paths[0])["parameter"], self.weight)
        self.assertEqual((self.model / "model.safetensors").read_bytes(), before)
        with self.assertRaises(vx.PluginClosedError):
            trainer.step(self.x, [0])

    def test_commit_changes_rollback_baseline_and_save_pins_graph(self):
        with self.trainer(optimizer=vx.SGD(lr=.1)) as trainer:
            trainer.step(self.x, [0])
            committed = trainer.save(self.root / "committed")
            trainer.commit()
            trainer.step(self.x, [1])
            trainer.rollback()
            (self.model / "graph.json").write_text("replaced during training")
            restored = trainer.save(self.root / "restored")
            np.testing.assert_array_equal(load_file(restored.weight_paths[0])["parameter"],
                                          load_file(committed.weight_paths[0])["parameter"])
            with vx.InferenceSession(restored) as session:
                session.run(self.x)

    def test_checkpoint_resumes_adamw_and_rollback_restores_imported_state(self):
        checkpoint = self.root / "step1.pb"
        with self.trainer(optimizer=vx.AdamW(lr=.015, beta1=.8, weight_decay=.02), rng_seed=17) as trainer:
            trainer.step(self.x, [0])
            trainer.save_checkpoint(checkpoint, metadata=b"dataset-position=1")
            first = trainer.save(self.root / "first")
            expected_result = trainer.step(self.x[:, ::-1], [1])
            expected = trainer.save(self.root / "continued")
        with self.trainer(checkpoint=checkpoint) as resumed:
            result = resumed.step(self.x[:, ::-1], [1])
            self.assertEqual(result.optimizer_step, expected_result.optimizer_step)
            self.assertAlmostEqual(result.loss, expected_result.loss, delta=1e-7)
            actual = resumed.save(self.root / "resumed")
            np.testing.assert_array_equal(load_file(actual.weight_paths[0])["parameter"],
                                          load_file(expected.weight_paths[0])["parameter"])
            resumed.rollback()
            baseline = resumed.save(self.root / "baseline")
            np.testing.assert_array_equal(load_file(baseline.weight_paths[0])["parameter"],
                                          load_file(first.weight_paths[0])["parameter"])
            saved = resumed.save_checkpoint(self.root / "resaved.pb")
            payload = pb.TrainerCheckpoint.from_bytes(saved.read_bytes())
            self.assertEqual(payload.metadata, b"dataset-position=1")
            self.assertEqual(payload.optimizer_step, 1)
            self.assertEqual(payload.rng_seed, 17)

    def test_accumulation_matches_full_update_and_rejects_unfinished_export(self):
        with vx.TrainingSession(self.model, loss=vx.CrossEntropyLoss("logits", normalizer=2),
                                optimizer=vx.SGD(lr=.1)) as trainer:
            first = trainer.step(self.x, [0], accumulation_steps=2)
            self.assertFalse(first.update_applied)
            with self.assertRaises(vx.VolvoxAIError):
                trainer.save(self.root / "unfinished")
            final = trainer.step(self.x, [0], accumulation_steps=2)
            self.assertTrue(final.update_applied)
            actual = trainer.save(self.root / "accumulated")
            np.testing.assert_allclose(load_file(actual.weight_paths[0])["parameter"], self.expected_sgd()[0], atol=2e-6)

    def test_rejected_targets_and_existing_exports_do_not_mutate_files(self):
        with self.trainer() as trainer:
            for targets in ([.5], [True], [2**32], []):
                with self.subTest(targets=targets), self.assertRaises((TypeError, ValueError)):
                    trainer.step(self.x, targets)
            package = trainer.save(self.root / "weights")
            checkpoint = trainer.save_checkpoint(self.root / "checkpoint.pb")
            before = checkpoint.read_bytes()
            with self.assertRaises(FileExistsError):
                trainer.save_checkpoint(checkpoint)
            with self.assertRaises(ValueError):
                trainer.save(package.path)
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertFalse(list(self.root.glob(".checkpoint-*")))

    def test_sharded_model_export_retains_order_and_selected_parameters(self):
        bias = np.array([.1, -.1], dtype=np.float32)
        save_file({"bias": bias}, self.model / "bias.safetensors")
        self.graph["nodes"][0]["inputs"]["bias"] = "bias"
        self.write_graph()
        weights = [self.model / "model.safetensors", self.model / "bias.safetensors"]
        with self.trainer(weights=weights, trainable_names=["bias"], optimizer=vx.SGD(lr=.1)) as trainer:
            trainer.step(self.x, [0])
            saved = trainer.save(self.root / "sharded")
            self.assertEqual(len(saved.weight_paths), 2)
            np.testing.assert_array_equal(load_file(saved.weight_paths[0])["parameter"], self.weight)
            trained_bias = load_file(saved.weight_paths[1])["bias"]
            self.assertFalse(np.array_equal(trained_bias, bias))
            with vx.InferenceSession(saved) as session:
                np.testing.assert_allclose(session.run(self.x)["logits"], self.x @ self.weight + trained_bias, atol=1e-7)

    def test_checkpoint_mismatch_and_unknown_parameter_raise_native_error(self):
        with self.trainer() as trainer:
            checkpoint = trainer.save_checkpoint(self.root / "original.pb")
        self.graph["nodes"][0]["id"] = "different-graph"
        self.write_graph()
        with self.assertRaises(vx.VolvoxAIError):
            self.trainer(checkpoint=checkpoint)
        with self.trainer(trainable_names=["missing"]) as trainer:
            with self.assertRaises(vx.VolvoxAIError):
                trainer.step(self.x, [0])

    def test_transport_failure_drains_before_reusing_trainer(self):
        for method in ("train_step", "export_trainer_weights", "commit_trainer"):
            with self.subTest(method=method), self.trainer() as trainer:
                with patch.object(trainer._engine, method, side_effect=vx.FfiError("lost training response")):
                    with self.assertRaises(vx.FfiError):
                        if method == "train_step":
                            trainer.step(self.x, [0])
                        elif method == "commit_trainer":
                            trainer.commit()
                        else:
                            trainer.save(self.root / "failed-export")
                with self.assertRaises(vx.PluginClosedError):
                    trainer.step(self.x, [0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
