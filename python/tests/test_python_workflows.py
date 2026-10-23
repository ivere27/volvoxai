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
from unittest.mock import Mock, patch
import weakref

if not sys.flags.isolated:
    source = Path(__file__).resolve().parent.parent
    if (source / "volvoxai/__init__.py").is_file():
        sys.path.insert(0, str(source))

import numpy as np
from safetensors.numpy import load_file, save_file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution_fixture import stateless_graph, feedback_graph

import volvoxai as vx
from volvoxai import _arrays, _async_session
from volvoxai._execution import array_destinations

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
        (self.model / "graph.json").write_text(json.dumps(stateless_graph(self.graph)))

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
            array_destinations([pb.Tensor(name="tokens", shape=[2], dtype=pb.DataType.DATA_TYPE_BF16,
                buffer=pb.BufferView(buffer_id=1, length_bytes=4))], ["tokens"])

    def test_host_buffers_skip_inline_payloads_and_static_shape_query(self):
        with vx.InferenceSession(self.model) as session:
            with patch.object(session._engine, "execute", wraps=session._engine.execute) as execute, \
                 patch.object(session._buffers, "copy_tensors", wraps=session._buffers.copy_tensors) as read:
                actual = session.run(self.x)["logits"]
                request = execute.call_args.args[0]
                self.assertEqual(request.inputs.values[0].borrowed.resource.handle, self.x.ctypes.data)
                self.assertFalse(request.inputs.values[0].inline)
                self.assertEqual(read.call_args.args[0].into[0].resource.handle, actual.ctypes.data)
                read.assert_called_once()
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
            with patch.object(session._buffers, "copy_tensors", side_effect=vx.FfiError("lost read")):
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


class ContextReplacement(Fixture, unittest.TestCase):
    def test_old_context_retires_before_replacement_and_outputs_survive(self):
        with vx.InferenceSession(self.model) as session:
            with session.run_tensors(self.x) as retained:
                old = pb.ExecutionContextRef(context_id=session._context.context_id)
                create = session._engine.create_execution_context

                def after_retirement(request):
                    with self.assertRaises(vx.VolvoxAIError):
                        session._engine.get_execution_context_info(old)
                    np.testing.assert_allclose(retained["logits"].numpy(), self.x @ self.weight, atol=1e-7)
                    return create(request)

                with patch.object(session._engine, "create_execution_context", side_effect=after_retirement) as replacement:
                    session.new_context()
                    replacement.assert_called_once()
                np.testing.assert_allclose(session.run(self.x)["logits"], self.x @ self.weight, atol=1e-7)

    def test_replacement_failure_closes_session_but_retained_outputs_survive(self):
        with vx.InferenceSession(self.model) as session:
            with session.run_tensors(self.x) as retained:
                with patch.object(session._engine, "create_execution_context", side_effect=RuntimeError("allocation refused")):
                    with self.assertRaisesRegex(RuntimeError, "allocation refused"):
                        session.new_context()
                with self.assertRaises(vx.PluginClosedError):
                    session.run(self.x)
                np.testing.assert_allclose(retained["logits"].numpy(), self.x @ self.weight, atol=1e-7)


class AsyncWorkflow(Fixture, unittest.IsolatedAsyncioTestCase):
    async def test_replacement_cancellation_drains_retirement_before_creation(self):
        async with vx.AsyncInferenceSession(self.model) as session:
            entered, finish = asyncio.Event(), asyncio.Event()
            old = pb.ExecutionContextRef(context_id=session._context.context_id)
            release, create = session._engine.release_execution_context, session._engine.create_execution_context

            async def delayed_release(request):
                await release(request)
                entered.set()
                await finish.wait()

            async def after_retirement(request):
                with self.assertRaises(vx.VolvoxAIError):
                    await session._engine.get_execution_context_info(old)
                return await create(request)

            with patch.object(session._engine, "release_execution_context", side_effect=delayed_release) as retire, \
                 patch.object(session._engine, "create_execution_context", side_effect=after_retirement) as replacement:
                job = asyncio.create_task(session.new_context())
                await entered.wait()
                replacement.assert_not_called()
                job.cancel()
                await asyncio.sleep(0)
                job.cancel()
                await asyncio.sleep(.005)
                self.assertFalse(job.done())
                finish.set()
                with self.assertRaises(asyncio.CancelledError):
                    await job
                retire.assert_awaited_once()
                replacement.assert_awaited_once()
            np.testing.assert_allclose((await session.run(self.x))["logits"], self.x @ self.weight, atol=1e-7)

    async def test_replacement_failure_closes_owner(self):
        async with vx.AsyncInferenceSession(self.model) as session:
            with patch.object(session._engine, "create_execution_context", side_effect=RuntimeError("allocation refused")):
                with self.assertRaisesRegex(RuntimeError, "allocation refused"):
                    await session.new_context()
            with self.assertRaises(vx.PluginClosedError):
                await session.run(self.x)

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
                with patch.object(session._buffers, "copy_tensors", wraps=session._buffers.copy_tensors) as read:
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
                 patch.object(session._buffers, "release_buffers", wraps=session._buffers.release_buffers) as release:
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
            with patch.object(session._buffers, "copy_tensors", side_effect=vx.FfiError("lost response")):
                with self.assertRaises(vx.FfiError):
                    await session.run(self.x)
            with self.assertRaises(vx.PluginClosedError):
                await session.run(self.x)


class QuantizationWorkflow(Fixture, unittest.TestCase):
    def test_static_reduce_range_matches_exporter_weight_codes_and_scales(self):
        from volvoxai._cli import _export_module
        oracle = _export_module()._exporter_module("typed_ptq")
        for reduce_range, bound in ((False, 127), (True, 64)):
            with self.subTest(reduce_range=reduce_range):
                result = vx.quantize(self.model, calibration_data=[self.x, -self.x], output=self.root / f"range-{bound}", group_size=0, static_activations=vx.pb.StaticActivations(dtype=vx.pb.DataType.DATA_TYPE_U8, scheme=vx.pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC, reduce_weight_range=reduce_range))
                graph = json.loads(result.graph_path.read_text())
                node = next(node for node in graph["nodes"] if node["id"] == "projection")
                name = node["inputs"]["weight"]
                stored = load_file(result.weight_paths[0])
                expected_codes, expected_scales, *_ = oracle._pack_weight(
                    np.ascontiguousarray(self.weight.T), bound=bound)
                np.testing.assert_array_equal(stored[name], expected_codes)
                scale_name = graph["quantization"]["tensors"][name]["scale_tensor"]
                np.testing.assert_array_equal(stored[scale_name], expected_scales)
                self.assertEqual(int(np.abs(stored[name]).max()), bound)

    def test_static_policy_requires_calibration(self):
        result = vx.quantize(self.model, calibration_data=[self.x, self.x * 2], output=self.root / "static", group_size=0, static_activations=vx.pb.StaticActivations(dtype=vx.pb.DataType.DATA_TYPE_I8, scheme=vx.pb.AffineScheme.AFFINE_SCHEME_SYMMETRIC, reduce_weight_range=False))
        self.assertTrue(result.requires_calibration)
        self.assertEqual(result.calibration_batches, 2)
        with self.assertRaisesRegex(ValueError, "required"):
            vx.quantize(self.model, output=self.root / "missing", group_size=0, static_activations=vx.pb.StaticActivations(dtype=vx.pb.DataType.DATA_TYPE_I8, scheme=vx.pb.AffineScheme.AFFINE_SCHEME_SYMMETRIC, reduce_weight_range=False))

    def test_grouped_weights_need_no_calibration_and_preserve_existing_output(self):
        from volvoxai import _quantization
        real = _quantization.VxQuantizationServiceClient
        clients = []
        def client(host):
            value = Mock(wraps=real(host))
            clients.append(value)
            return value
        options = dict(weight_bits=4, group_size=128)
        with patch.object(_quantization, "VxQuantizationServiceClient", client):
            result = vx.quantize(self.model, output=self.root / "packed", **options)
        clients[0].calibrate_quantization_plan.assert_not_called()
        clients[0].release_quantization_plan.assert_called_once()
        self.assertFalse(result.requires_calibration)
        self.assertEqual(result.calibration_batches, 0)
        self.assertEqual(result.calibration_backend, "")
        previous = result.weight_paths[0].read_bytes()
        with self.assertRaisesRegex(ValueError, "existing"):
            vx.quantize(self.model, output=result.path, **options)
        self.assertEqual(result.weight_paths[0].read_bytes(), previous)
        with self.assertRaisesRegex(ValueError, "omitted"):
            vx.quantize(self.model, output=self.root / "invalid", calibration_data=[self.x], **options)

    def test_weight_calibration_algorithms(self):
        with self.assertRaisesRegex(ValueError, "algorithm must be one of"):
            vx.quantize(self.model, output=self.root / "invalid_algo", weight_bits=4, group_size=128, algorithm="invalid")

        with self.assertRaises(vx.VolvoxAIError):
            vx.quantize(self.model, output=self.root / "invalid_static", calibration_data=[self.x], group_size=0, static_activations=vx.pb.StaticActivations(dtype=vx.pb.DataType.DATA_TYPE_I8, scheme=vx.pb.AffineScheme.AFFINE_SCHEME_SYMMETRIC, reduce_weight_range=False), algorithm="mse-clip")

        result_mse = vx.quantize(self.model, output=self.root / "packed_mse", weight_bits=4, group_size=128, algorithm="mse-clip")
        self.assertFalse(result_mse.requires_calibration)
        self.assertEqual(result_mse.calibration_batches, 0)
        with self.assertRaisesRegex(ValueError, "omitted"):
            vx.quantize(self.model, output=self.root / "invalid_mse", calibration_data=[self.x], weight_bits=4, group_size=128, algorithm="mse-clip")

        with self.assertRaisesRegex(ValueError, "required"):
            vx.quantize(self.model, output=self.root / "missing_calib_importance", weight_bits=4, group_size=128, algorithm="importance")

        result_imp_w4a16 = vx.quantize(self.model, output=self.root / "packed_imp_w4a16", weight_bits=4, group_size=128, calibration_data=[self.x, self.x * 2], algorithm="importance", rules=[vx.pb.QuantizationRule(node_id=name, weights=vx.pb.WeightQuantization(format=vx.pb.WeightFormat(bits=4, group_size=128), algorithm=vx.pb.WeightAlgorithm.WEIGHT_ALGORITHM_IMPORTANCE)) for name in ["projection"]] + [vx.pb.QuantizationRule(node_glob="*", keep_float=vx.pb.Empty())])
        self.assertTrue(result_imp_w4a16.requires_calibration)
        self.assertEqual(result_imp_w4a16.calibration_batches, 2)
        self.assertEqual(result_imp_w4a16.calibration_backend, "cpu")

        result_imp_w8a8_dyn = vx.quantize(self.model, output=self.root / "packed_imp_w8a8_dyn", calibration_data=[self.x], weight_bits=8, group_size=128, algorithm="importance", rules=[vx.pb.QuantizationRule(node_id=name, weights=vx.pb.WeightQuantization(format=vx.pb.WeightFormat(bits=8, group_size=128), algorithm=vx.pb.WeightAlgorithm.WEIGHT_ALGORITHM_IMPORTANCE)) for name in ["projection"]] + [vx.pb.QuantizationRule(node_glob="*", keep_float=vx.pb.Empty())])
        self.assertTrue(result_imp_w8a8_dyn.requires_calibration)
        self.assertEqual(result_imp_w8a8_dyn.calibration_batches, 1)

    def test_calibration_backend_does_not_claim_unused_cuda_execution(self):
        # RTN packing needs no executor, including on a host without a GPU.
        result = vx.quantize(self.model, output=self.root / "no-observers-cuda",
            weight_bits=3, group_size=32, algorithm="rtn", calibration_backend="cuda")
        self.assertFalse(result.requires_calibration)
        self.assertEqual(result.calibration_backend, "")
        for backend in ("not-a-backend", "cpu\0cuda"):
            with self.subTest(backend=backend), self.assertRaises(vx.VolvoxAIError):
                vx.quantize(self.model, output=self.root / "invalid-backend",
                    weight_bits=3, group_size=32, algorithm="rtn",
                    calibration_backend=backend)
        # Package staging may create an empty destination before native policy
        # validation; a rejected backend must publish no graph or weights.
        self.assertEqual(list((self.root / "invalid-backend").iterdir()), [])

    def test_grouped_policies_tail_bias_and_shared_float_weight(self):
        rng = np.random.default_rng(724)
        width, outputs = 131, 5
        weight = rng.normal(0, .2, (width, outputs)).astype(np.float32)
        weight[128:, 0] = 0  # Zero group and partial packed word.
        bias = np.linspace(-.2, .2, outputs, dtype=np.float32)
        x = rng.normal(0, .4, (3, width)).astype(np.float32)
        x[0] = 0
        self.graph["inputs"]["x"]["shape"] = [3, width]
        node = self.graph["nodes"][0]
        node["inputs"]["bias"] = "bias"
        node["outputs"]["out"]["shape"] = [3, outputs]
        # This retained node shares the weight; writing must preserve its F32
        # ownership alongside the packed copy selected for projection.
        retained = {**node, "id": "retained", "outputs": {"out": {
            "tensor": "float_logits", "dtype": "float32", "shape": [3, outputs]}}}
        self.graph["nodes"].append(retained)
        self.graph["outputs"].append("float_logits")
        save_file({"parameter": weight, "bias": bias}, self.model / "model.safetensors")
        self.write_graph()
        source = (self.model / "model.safetensors").read_bytes()
        for precision, weight_bits, activation_bits in (("w4a8", 4, 8), ("w4a16", 4, 16), ("w8a8", 8, 8), ("w8a16", 8, 16)):
            with self.subTest(precision=precision):
                result = vx.quantize(self.model, weight_bits=weight_bits, group_size=128, output=self.root / precision, rules=[vx.pb.QuantizationRule(node_id=name, weights=vx.pb.WeightQuantization(format=vx.pb.WeightFormat(bits=weight_bits, group_size=128), algorithm=vx.pb.WeightAlgorithm.WEIGHT_ALGORITHM_RTN)) for name in ["projection"]] + [vx.pb.QuantizationRule(node_glob="*", keep_float=vx.pb.Empty())])
                self.assertEqual((result.quantized_nodes, result.retained_float_nodes), (1, 1))
                graph = json.loads(result.graph_path.read_text())
                converted = next(item for item in graph["nodes"] if item["id"] == "projection")
                self.assertEqual(converted["opType"], "PackedLinear")
                self.assertEqual(converted["params"], {
                    "group_size": 128, "weight_bits": weight_bits})
                stored = load_file(result.weight_paths[0])
                np.testing.assert_array_equal(stored["parameter"], weight)
                np.testing.assert_array_equal(stored["bias"], bias)
                per_word, offset = 32 // weight_bits, 1 << (weight_bits - 1)
                packed = stored[converted["inputs"]["weight"]]
                scales = stored[converted["inputs"]["scale"]]
                self.assertEqual(packed.dtype, np.int32)
                self.assertEqual(packed.shape, (outputs, (width + per_word - 1) // per_word))
                self.assertEqual(scales.shape, (outputs, 2))
                q = np.empty((outputs, width), np.int32)
                expected = np.zeros((len(x), outputs), np.float32)
                for start in (0, 128):
                    block = weight.T[:, start:start + 128]
                    maximum = np.abs(block).max(axis=1)
                    scale = np.where(maximum == 0, np.float32(1), maximum / np.float32(offset - 1))
                    np.testing.assert_array_equal(scales[:, start // 128], scale)
                    q[:, start:start + 128] = np.clip(np.rint(block / scale[:, None]), -offset, offset - 1)
                for k in range(width):
                    decoded = ((packed[:, k // per_word].view(np.uint32) >> (weight_bits * (k % per_word)))
                               & (2 * offset - 1)).astype(np.int32) - offset
                    np.testing.assert_array_equal(decoded, q[:, k])
                for k in range(width, packed.shape[1] * per_word):
                    decoded = (packed[:, k // per_word].view(np.uint32) >> (weight_bits * (k % per_word))) & (2 * offset - 1)
                    np.testing.assert_array_equal(decoded, np.full(outputs, offset))
                if activation_bits == 16:
                    activation = x.astype(np.float16).astype(np.float32)
                    rounded = (q.astype(np.float32) * scales[:, np.arange(width) // 128]).astype(np.float16).astype(np.float32)
                    for k in range(width):
                        expected += activation[:, k, None] * rounded[None, :, k]
                else:
                    maximum = np.abs(x).max(axis=1)
                    scale = np.where(maximum == 0, np.float32(1), maximum / np.float32(127))
                    activation = np.clip(np.rint(x / scale[:, None]), -127, 127).astype(np.int32)
                    for start in (0, 128):
                        dot = activation[:, start:start + 128] @ q[:, start:start + 128].T
                        expected += dot.astype(np.float32) * (scale[:, None] * scales[None, :, start // 128])
                expected += bias
                with vx.InferenceSession(result, cpu_threads=1,
                    packed_arithmetic="dynamic-i8" if activation_bits == 8 else "f16") as session:
                    actual = session.run(x)
                np.testing.assert_allclose(actual["logits"], expected, atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(actual["float_logits"], x @ weight + bias, atol=1e-6, rtol=1e-6)
                self.assertEqual((self.model / "model.safetensors").read_bytes(), source)

    def test_calibration_requirement_comes_from_resolved_rules(self):
        result = vx.quantize(self.model, output=self.root / "resolved-rtn", algorithm="importance",
            rules=[vx.pb.QuantizationRule(node_id="projection", weights=vx.pb.WeightQuantization(
                format=vx.pb.WeightFormat(bits=8, group_size=128),
                algorithm=vx.pb.WeightAlgorithm.WEIGHT_ALGORITHM_RTN))])
        self.assertFalse(result.requires_calibration)
        self.assertEqual(result.calibration_batches, 0)

    def test_storage_package_reuses_every_execution_arithmetic(self):
        result = vx.quantize(self.model, weight_bits=4, output=self.root / "shared-storage")
        node = json.loads(result.graph_path.read_text())["nodes"][0]
        self.assertNotIn("activation_bits", node["params"])
        before = result.weight_paths[0].read_bytes()
        for arithmetic in ("dynamic-i8", "f16", "f32", "f32-fast"):
            with vx.InferenceSession(result, packed_arithmetic=arithmetic) as session:
                self.assertTrue(np.isfinite(session.run(self.x)["logits"]).all())
                self.assertEqual(session.packed_arithmetic,
                    vx.pb.PackedArithmetic.Value("PACKED_ARITHMETIC_" + arithmetic.upper().replace("-", "_")))
        self.assertEqual(before, result.weight_paths[0].read_bytes())

    def test_inference_session_packed_arithmetic(self):
        from volvoxai._session import _options, _resolve_packed_arithmetic
        self.assertIsNone(_options("cpu", None))
        for name in ("dynamic-i4", "dynamic-i8", "f16", "f32", "f32-fast"):
            self.assertEqual(_resolve_packed_arithmetic(name), vx.pb.PackedArithmetic.Value(
                "PACKED_ARITHMETIC_" + name.upper().replace("-", "_")))
        for invalid in ("float16", "w4a8", "auto", True, 99, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _resolve_packed_arithmetic(invalid)

    def test_reserved_dynamic_i4_is_refused_without_fallback(self):
        result = vx.quantize(self.model, weight_bits=4, output=self.root / "reserved-i4")
        before = result.weight_paths[0].read_bytes()
        for backend in ("cpu", "cuda"):
            with self.subTest(backend=backend), self.assertRaises(vx.VolvoxAIError) as raised:
                vx.InferenceSession(result, backend=backend, packed_arithmetic="dynamic-i4")
            self.assertEqual(raised.exception.status, pb.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED)
            self.assertEqual(raised.exception.code, pb.OperationCode.OPERATION_CODE_BACKEND_UNSUPPORTED)
            self.assertEqual(raised.exception.stage, pb.OperationStage.OPERATION_STAGE_COMPILE)
            self.assertEqual(raised.exception.response.compiled_model_id, 0)
        # Refusal leaves the same immutable model usable under implemented A8.
        with vx.InferenceSession(result, packed_arithmetic="dynamic-i8") as session:
            self.assertTrue(np.isfinite(session.run(self.x)["logits"]).all())
        self.assertEqual(before, result.weight_paths[0].read_bytes())

    def test_incompatible_policies_fail_before_reading_batches(self):
        def batches():
            self.fail("invalid policy consumed calibration")
            yield self.x
        cases = ({"weight_bits": 5}, {"group_size": 16}, {"algorithm": "unknown"},
                 {"static_activations": "i8"}, {"rules": ["dense"]})
        for index, options in enumerate(cases):
            output = self.root / f"invalid-precision-{index}"
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                vx.quantize(self.model, calibration_data=batches(), output=output, **options)
            self.assertFalse(output.exists())

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
        result = vx.quantize(self.model, calibration_data=batches(), output=self.root / "int8", samples_per_batch=samples_per_batch, group_size=0, static_activations=vx.pb.StaticActivations(dtype=vx.pb.DataType.DATA_TYPE_I8, scheme=vx.pb.AffineScheme.AFFINE_SCHEME_SYMMETRIC, reduce_weight_range=False))
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
                vx.quantize(self.model, calibration_data=data, output=output, group_size=0, static_activations=vx.pb.StaticActivations(dtype=vx.pb.DataType.DATA_TYPE_I8, scheme=vx.pb.AffineScheme.AFFINE_SCHEME_SYMMETRIC, reduce_weight_range=False))
            self.assertFalse((output / "graph.json").exists())
            self.assertFalse((output / "model.safetensors").exists())
            self.assertFalse(list(output.glob(".*.export-*")))

    def test_existing_package_is_preserved(self):
        result = vx.quantize(self.model, calibration_data=[self.x], output=self.root / "int8", group_size=0, static_activations=vx.pb.StaticActivations(dtype=vx.pb.DataType.DATA_TYPE_I8, scheme=vx.pb.AffineScheme.AFFINE_SCHEME_SYMMETRIC, reduce_weight_range=False))
        before = result.weight_paths[0].read_bytes()
        with self.assertRaisesRegex(ValueError, "existing"):
            vx.quantize(self.model, calibration_data=[self.x], output=result.path, group_size=0, static_activations=vx.pb.StaticActivations(dtype=vx.pb.DataType.DATA_TYPE_I8, scheme=vx.pb.AffineScheme.AFFINE_SCHEME_SYMMETRIC, reduce_weight_range=False))
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
            payload = pb.TrainerCheckpoint.FromString(saved.read_bytes())
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
