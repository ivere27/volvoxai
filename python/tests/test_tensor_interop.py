"""Native tensor ownership and DLPack numerics; also runs against installed wheels.

CPU tests need only the wheel. Set VOLVOXAI_TEST_CUDA=1 on a CUDA host with
PyTorch installed to run the GPU stream/lifetime/copy checks.
"""
from __future__ import annotations

import gc
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from google.protobuf.json_format import MessageToDict
import weakref
from unittest.mock import patch

if not sys.flags.isolated:
    source_python = Path(__file__).resolve().parent.parent
    if (source_python / "volvoxai/__init__.py").is_file():
        sys.path.insert(0, str(source_python))

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution_fixture import stateless_graph, feedback_graph

import volvoxai as vx
from volvoxai._tensor import _dlpack

# Initialize the optional profiler dependency before any native CUDA context.
# PyTorch 2.5.1/CUDA 12.1 with driver 535 can crash in CUPTI at process exit
# when cuInit precedes the PyTorch import, even without VolvoxAI loaded.
if os.environ.get("VOLVOXAI_TEST_CUDA") == "1":
    import torch


def model(directory, *, dtype="float32", shape=None):
    shape = [2, 4] if shape is None else shape
    dimensions = {"N": {"min": 1, "max": 4}} if "N" in shape else {}
    (Path(directory) / "graph.json").write_text(json.dumps(stateless_graph({
        "format": "volvox-graph/v1", "dimensions": dimensions,
        "inputs": {"x": {"shape": shape, "dtype": dtype}},
        "nodes": [{"id": "twice", "opType": "Add", "inputs": {"a": "x", "b": "x"},
                   "outputs": {"out": {"tensor": "y", "shape": shape, "dtype": dtype}}, "params": {}}],
        "outputs": ["y"]})))


def native_address(tensor):
    storage = tensor._get_storage()
    access = storage.engine.begin_buffer_access(vx.pb.BufferAccessRequest(view=storage.handle.buffer))
    try:
        return access.memory.resource.handle + access.memory.offset_bytes
    finally:
        storage.engine.end_buffer_access(vx.pb.EndBufferAccessRequest(access_ids=[access.access_id]))


class CapsuleProducer:
    def __init__(self, tensor, **options):
        self.tensor, self.options = tensor, options

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()

    def __dlpack__(self, **kwargs):
        return self.tensor.__dlpack__(**kwargs, **self.options)


class NativeTensorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="volvoxai-tensor-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        model(self.directory)
        self.x = np.arange(8, dtype=np.float32).reshape(2, 4)

    def test_shared_runtime_budget_is_enforced_until_owned_output_release(self):
        budget = vx.pb.RuntimeBudget(max_unconsumed_result_bytes=self.x.nbytes)
        with vx.Runtime(budget=budget) as runtime, \
                runtime.inference_session(self.directory) as session:
            with session.run_tensors(self.x) as retained:
                np.testing.assert_array_equal(retained["y"].numpy(), self.x * 2)
                with self.assertRaises(vx.VolvoxAIError) as denied:
                    session.run_tensors(self.x)
                self.assertEqual(denied.exception.report.status,
                                 vx.pb.NativeStatus.NATIVE_STATUS_OVERLOADED)
            # Releasing the final owned output returns the same runtime budget.
            np.testing.assert_array_equal(session.run(self.x)["y"], self.x * 2)

    def test_runtime_budget_requires_generated_message(self):
        with self.assertRaises(TypeError):
            vx.Runtime(budget={"max_unconsumed_result_bytes": 32})

    def test_dlpack_keeps_opaque_handle_separate_from_byte_offset(self):
        owner = CapsuleProducer(None)
        capsule = _dlpack().make_capsule(0x1000, 8, 0, 2, 32, 1, (2, 4), owner, 0, 32)
        descriptor, lease = _dlpack().consume(capsule)
        self.assertEqual(descriptor["device_type"], 8)
        self.assertEqual(descriptor["address"], 0x1000)
        self.assertEqual(descriptor["offset"], 32)
        self.assertEqual(descriptor["shape"], (2, 4))
        del lease

    def test_snapshots_and_shared_views_outlive_session(self):
        with vx.InferenceSession(self.directory) as session:
            tensor = session.run_tensors(self.x)["y"]
            view = tensor.numpy(copy=False)
            self.assertEqual(view.ctypes.data, native_address(tensor))
            self.assertEqual(tensor.shape, self.x.shape)
            self.assertEqual(tensor.dtype, np.dtype("float32"))
            self.assertEqual(tensor.device, "cpu")
            other = session.run_tensors(self.x + 100)["y"]
            np.testing.assert_array_equal(tensor.numpy(), self.x * 2)
        np.testing.assert_array_equal(tensor.numpy(), self.x * 2)
        np.testing.assert_array_equal(other.numpy(), (self.x + 100) * 2)
        tensor.close()
        other.close()
        gc.collect()
        np.testing.assert_array_equal(view, self.x * 2)
        with self.assertRaises(vx.PluginClosedError):
            tensor.numpy()
        with self.assertRaises(vx.PluginClosedError):
            session.run_tensors(self.x)
        del view

    def test_reuse_across_sessions_and_closed_producer_session(self):
        with vx.Runtime() as runtime:
            with runtime.inference_session(self.directory) as producer:
                tensor = producer.run_tensors(self.x)["y"]
            with runtime.inference_session(self.directory) as consumer:
                with consumer.run_tensors(tensor)["y"] as output:
                    np.testing.assert_array_equal(output.numpy(), self.x * 4)
            with vx.InferenceSession(self.directory) as foreign:
                with self.assertRaisesRegex(ValueError, "one Runtime"):
                    foreign.run_tensors(tensor)
                # Standard DLPack is an explicit, owner-independent import.
                for value in (CapsuleProducer(tensor), CapsuleProducer(tensor, max_version=(1, 2))):
                    with foreign.run_tensors(value)["y"] as output:
                        np.testing.assert_array_equal(output.numpy(), self.x * 4)
            tensor.close()


    def test_native_handle_is_retired_only_after_last_export(self):
        with vx.InferenceSession(self.directory) as session:
            tensor = session.run_tensors(self.x)["y"]
            reference = vx.pb.BufferHandle(buffer_id=tensor._get_storage().handle.buffer.buffer_id)
            view = np.from_dlpack(tensor)
            with self.assertRaises(vx.VolvoxAIError):
                tensor.__dlpack__()
            tensor.close()
            with self.assertRaises(vx.VolvoxAIError):
                session._buffers.get_buffer_info(reference)
            np.testing.assert_array_equal(view, self.x * 2)
            del view
            gc.collect()
            self.assertEqual(session._owner.references, 1)


    def test_capsule_is_consumed_once_and_unused_capsule_releases(self):
        with vx.InferenceSession(self.directory) as session:
            for version in (None, (1, 0), (1, 2)):
                tensor = session.run_tensors(self.x)["y"]
                storage = weakref.ref(tensor._get_storage())
                capsule = tensor.__dlpack__(max_version=version)
                tensor.close()
                descriptor, lease = _dlpack().consume(capsule)
                self.assertEqual(descriptor["shape"], self.x.shape)
                self.assertIsNone(descriptor["strides"])
                with self.assertRaisesRegex(ValueError, "consumed"):
                    _dlpack().consume(capsule)
                del descriptor, lease, capsule
                gc.collect()
                self.assertIsNone(storage())
            tensor = session.run_tensors(self.x)["y"]
            storage = weakref.ref(tensor._get_storage())
            capsule = tensor.__dlpack__()
            tensor.close()
            del capsule
            gc.collect()
            self.assertIsNone(storage())

    def test_host_copy_is_independent_and_output_selection_retires_unused_handles(self):
        with vx.InferenceSession(self.directory) as session:
            tensor = session.run_tensors(self.x)["y"]
            copy = tensor.numpy()
            copy[:] = -3
            np.testing.assert_array_equal(tensor.numpy(), self.x * 2)
            tensor.close()
            self.assertEqual(session.run_tensors(self.x, output_names=[]), {})
            self.assertEqual(session._owner.references, 1)

    def test_group_release_keeps_exports_and_retires_other_handles(self):
        graph = json.loads((self.directory / "graph.json").read_text())
        graph["nodes"].append({"id": "plus", "opType": "Add", "inputs": {"a": "x", "b": "y"},
            "outputs": {"out": {"tensor": "z", "shape": [2, 4], "dtype": "float32"}}, "params": {}})
        graph["outputs"].append("z")
        (self.directory / "graph.json").write_text(json.dumps(stateless_graph(graph)))
        with vx.InferenceSession(self.directory) as session:
            outputs = session.run_tensors(self.x)
            refs = {k: v._get_storage().handle.buffer.buffer_id for k, v in outputs.items()}
            view = np.from_dlpack(outputs["y"])
            outputs.close()
            outputs.close()
            np.testing.assert_array_equal(view, self.x * 2)
            for buffer_id in refs.values():
                with self.assertRaises(vx.VolvoxAIError):
                    session._buffers.get_buffer_info(vx.pb.BufferHandle(buffer_id=buffer_id))
            del view
            gc.collect()
            session._buffers.release_buffers(vx.pb.BufferRefs(buffer_ids=[refs["z"], refs["z"], 0]))
            session._buffers.release_buffers(vx.pb.BufferRefs())
            with session.run_tensors(self.x) as batch:
                np.testing.assert_array_equal(batch["z"].numpy(), self.x * 3)
            with self.assertRaises(vx.PluginClosedError):
                batch["z"].numpy()

    def test_dynamic_shapes_do_not_invalidate_prior_exports(self):
        model(self.directory, shape=["N", 4])
        views = []
        with vx.InferenceSession(self.directory) as session:
            for count in (1, 4, 2, 3):
                x = np.arange(count * 4, dtype=np.float32).reshape(count, 4)
                with session.run_tensors(x)["y"] as tensor:
                    views.append((np.from_dlpack(tensor), x * 2))
        for view, expected in views:
            np.testing.assert_array_equal(view, expected)

    def test_recycled_multioutput_buffers_do_not_overwrite_live_snapshots(self):
        backends = ["cpu"] + os.environ.get("VOLVOXAI_TEST_NATIVE_BUFFERS", "").split(",")
        graph = json.loads((self.directory / "graph.json").read_text())
        graph["inputs"]["z"] = graph["inputs"]["x"].copy()
        graph["nodes"].append({"id": "sum", "opType": "Add", "inputs": {"a": "x", "b": "z"},
            "outputs": {"out": {"tensor": "sum", "shape": [2, 4], "dtype": "float32"}}, "params": {}})
        graph["outputs"].append("sum")
        (self.directory / "graph.json").write_text(json.dumps(stateless_graph(graph)))
        for backend in filter(None, backends):
            with self.subTest(backend=backend):
                with vx.InferenceSession(self.directory, backend=backend) as session:
                    retained = session.run_tensors({"x": self.x, "z": self.x + 10})
                    for i in range(12):
                        source = session.run_tensors({"x": self.x + i, "z": self.x - i})
                        result = session.run_tensors({"x": source["y"], "z": source["sum"]})
                        source.close()
                        np.testing.assert_array_equal(result["sum"].numpy(), 4 * self.x + 2 * i)
                        result.close()
                np.testing.assert_array_equal(retained["y"].numpy(), 2 * self.x)
                np.testing.assert_array_equal(retained["sum"].numpy(), 2 * self.x + 10)
                retained.close()

    def test_invalid_batch_is_native_error_and_session_recovers(self):
        with vx.InferenceSession(self.directory) as session:
            for value in (self.x.astype(np.float64), self.x.ravel(), {"missing": self.x}, {}):
                with self.assertRaises(vx.VolvoxAIError):
                    session.run_tensors(value)
                with session.run_tensors(self.x)["y"] as result:
                    np.testing.assert_array_equal(result.numpy(), self.x * 2)

    def test_export_rejects_implicit_device_copy_and_invalid_stream(self):
        with vx.InferenceSession(self.directory) as session, session.run_tensors(self.x)["y"] as tensor:
            for arguments in ({"copy": True}, {"dl_device": (2, 0)}, {"stream": 1}):
                with self.assertRaises(BufferError):
                    tensor.__dlpack__(**arguments)
            with self.assertRaises(TypeError):
                tensor.__dlpack__(max_version="1.0")

    def test_foreign_thread_can_release_export(self):
        with vx.InferenceSession(self.directory) as session:
            tensor = session.run_tensors(self.x)["y"]
            storage = weakref.ref(tensor._get_storage())
            values = [np.from_dlpack(tensor)]
            tensor.close()
            thread = threading.Thread(target=values.clear)
            thread.start()
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
            gc.collect()
            self.assertIsNone(storage())

    def test_device_payload_cannot_be_misread_as_host(self):
        with vx.InferenceSession(self.directory) as session, session.run_tensors(self.x)["y"] as tensor:
            with self.assertRaises(vx.VolvoxAIError):
                session._buffers.copy_tensors(vx.pb.CopyTensorsRequest(
                    sources=[tensor._get_storage().handle], into=[vx.pb.BorrowedBuffer(
                        resource=vx.pb.NativeResource(kind=vx.pb.NativeResourceKind.NATIVE_RESOURCE_KIND_CUDA,
                            handle=self.x.ctypes.data, size_bytes=self.x.nbytes), length_bytes=self.x.nbytes)]))


    def test_int32_keeps_values_beyond_float32_precision(self):
        model(self.directory, dtype="int32")
        graph = json.loads((self.directory / "graph.json").read_text())
        graph["nodes"][0]["opType"] = "Identity"
        graph["nodes"][0]["inputs"] = {"input": "x"}
        (self.directory / "graph.json").write_text(json.dumps(stateless_graph(graph)))
        x = np.array([[-16_777_219, -3, 0, 16_777_217]] * 2, dtype=np.int32)
        with vx.InferenceSession(self.directory) as session, session.run_tensors(x)["y"] as tensor:
            self.assertEqual(tensor.dtype, np.dtype("int32"))
            np.testing.assert_array_equal(np.from_dlpack(tensor), x)


class ContextTensorStateTest(unittest.TestCase):
    setUp = NativeTensorTest.setUp

    def backends(self):
        return ["cpu", *filter(None, os.environ.get("VOLVOXAI_TEST_NATIVE_BUFFERS", "").split(","))]

    def test_selection_reserves_only_selected_result_bytes(self):
        graph = json.loads((self.directory / "graph.json").read_text())
        graph["nodes"].append({"id": "again", "opType": "Add", "inputs": {"a": "y", "b": "y"},
            "outputs": {"out": {"tensor": "z", "shape": [2, 4], "dtype": "float32"}}, "params": {}})
        graph["outputs"].append("z")
        with vx.open_library() as host:
            engine = vx.VxInferenceServiceClient(host)
            runtime = engine.create_runtime(vx.pb.CreateRuntimeRequest(
                budget=vx.pb.RuntimeBudget(max_unconsumed_result_bytes=self.x.nbytes)))
            loaded = engine.load_model(vx.pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                package=vx.pb.ModelPackage(graph_document=json.dumps(stateless_graph(graph)).encode())))
            compiled = engine.compile_model(vx.pb.CompileModelRequest(model_id=loaded.model_id))
            context = engine.create_execution_context(vx.pb.CreateExecutionContextRequest(compiled_model_id=compiled.compiled_model_id, graph=vx.pb.GraphExecution()))
            request = vx.pb.ExecuteRequest(context_id=context.context_id, inputs=vx.pb.InputBindings(values=[vx.pb.Tensor(
                name="x", shape=self.x.shape, dtype=vx.pb.DataType.DATA_TYPE_F32, inline=self.x.tobytes())]), scope=vx.pb.ExecutionScope(entrypoint="evaluate", full=vx.pb.Empty()), outputs=vx.pb.OutputOptions(storage=vx.pb.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER))
            with self.assertRaises(vx.VolvoxAIError):
                engine.execute(request)
            request.outputs.selection.CopyFrom(vx.pb.TensorOutputSelection(names=["z"]))
            batch = engine.execute(request)
            self.assertEqual([h.name for h in batch.outputs], ["z"])
            self.assertEqual(sum(output.buffer.length_bytes for output in batch.outputs), self.x.nbytes)
            vx.VxBufferServiceClient(host).release_buffers(vx.pb.BufferRefs(buffer_ids=[h.buffer.buffer_id for h in batch.outputs]))
            request.outputs.selection.CopyFrom(vx.pb.TensorOutputSelection())
            batch = engine.execute(request)
            self.assertEqual(batch.outputs, [])
            self.assertEqual(sum(output.buffer.length_bytes for output in batch.outputs), 0)

    def test_each_output_returns_its_byte_budget_after_its_last_lease(self):
        graph = json.loads((self.directory / "graph.json").read_text())
        graph["nodes"].append({"id": "again", "opType": "Add", "inputs": {"a": "y", "b": "y"},
            "outputs": {"out": {"tensor": "z", "shape": [2, 4], "dtype": "float32"}}, "params": {}})
        graph["outputs"].append("z")
        with vx.open_library() as host:
            engine = vx.VxInferenceServiceClient(host)
            buffers = vx.VxBufferServiceClient(host)
            runtime = engine.create_runtime(vx.pb.CreateRuntimeRequest(
                budget=vx.pb.RuntimeBudget(max_unconsumed_result_bytes=2 * self.x.nbytes)))
            loaded = engine.load_model(vx.pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                package=vx.pb.ModelPackage(graph_document=json.dumps(stateless_graph(graph)).encode())))
            compiled = engine.compile_model(vx.pb.CompileModelRequest(model_id=loaded.model_id))
            context = engine.create_execution_context(vx.pb.CreateExecutionContextRequest(compiled_model_id=compiled.compiled_model_id, graph=vx.pb.GraphExecution()))
            request = vx.pb.ExecuteRequest(context_id=context.context_id, inputs=vx.pb.InputBindings(values=[vx.pb.Tensor(
                name="x", shape=self.x.shape, dtype=vx.pb.DataType.DATA_TYPE_F32, inline=self.x.tobytes())]), scope=vx.pb.ExecutionScope(entrypoint="evaluate", full=vx.pb.Empty()), outputs=vx.pb.OutputOptions(storage=vx.pb.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER))
            first = engine.execute(request)
            y, z = first.outputs
            access = buffers.begin_buffer_access(vx.pb.BufferAccessRequest(view=y.buffer))
            buffers.release_buffers(vx.pb.BufferRefs(buffer_ids=[y.buffer.buffer_id]))
            request.outputs.selection.CopyFrom(vx.pb.TensorOutputSelection(names=["y"]))
            with self.assertRaises(vx.VolvoxAIError):
                engine.execute(request)
            buffers.end_buffer_access(vx.pb.EndBufferAccessRequest(access_ids=[access.access_id]))
            second = engine.execute(request)
            retained = buffers.copy_tensors(vx.pb.CopyTensorsRequest(sources=[z], inline_result=True))
            np.testing.assert_array_equal(np.frombuffer(retained.outputs[0].inline, dtype=np.float32),
                (self.x * 4).ravel())
            buffers.release_buffers(vx.pb.BufferRefs(buffer_ids=[z.buffer.buffer_id,
                second.outputs[0].buffer.buffer_id]))

    def test_feedback_selection_validation_and_external_isolation(self):
        graph = feedback_graph(json.loads((self.directory / "graph.json").read_text()), {"x": "y"})
        (self.directory / "graph.json").write_text(json.dumps(graph))
        position = lambda value: np.array([value], dtype=np.int32)
        for backend in self.backends():
            with self.subTest(backend=backend), vx.InferenceSession(self.directory, backend=backend) as session:
                with self.assertRaisesRegex(ValueError, "explicit entrypoint"):
                    session.run_tensors({"x": self.x, "position": position(0)})
                with self.assertRaises(vx.VolvoxAIError):
                    session.run_tensors({"position": position(1)}, entrypoint="transition", feedback={"x": "y"})
                with session.run_tensors({"x": self.x, "position": position(0)}, entrypoint="initialize") as first:
                    expected = self.x * 2
                    if backend == "cpu":
                        writable = first["y"].numpy(copy=False, writable=True)
                        writable[:] = 0
                        del writable
                    with self.assertRaises(vx.VolvoxAIError):
                        session.run_tensors({"position": position(1)}, entrypoint="transition", feedback={"x": "unknown"})
                    with self.assertRaises(vx.VolvoxAIError):
                        session.run_tensors({"x": self.x, "position": position(1)}, entrypoint="transition", feedback={"x": "y"})
                    for step in range(1, 4):
                        expected *= 2
                        self.assertEqual(session.run_tensors({"position": position(step)}, entrypoint="transition",
                            feedback={"x": "y"}, output_names=[]), {})
                    with session.run_tensors({"position": position(4)}, entrypoint="transition", feedback={"x": "y"}) as output:
                        np.testing.assert_array_equal(output["y"].numpy(), expected * 2)
                    compiled = session.compiled_model_id
                    session.new_context()
                    self.assertEqual(session.compiled_model_id, compiled)
                    np.testing.assert_array_equal(first["y"].numpy(),
                        np.zeros_like(self.x) if backend == "cpu" else self.x * 2)
                    with self.assertRaises(vx.VolvoxAIError):
                        session.run_tensors({"position": position(1)}, entrypoint="transition", feedback={"x": "y"})
                    np.testing.assert_array_equal(session.run({"x": self.x, "position": position(0)},
                        entrypoint="initialize")["y"], self.x * 2)

    def test_growing_feedback_and_unchanged_inputs(self):
        graph = {"format": "volvox-graph/v1", "dimensions": {"P": {"min": 1, "max": 6}, "R": {"min": 2, "max": 7}},
            "inputs": {"past": {"shape": ["P", 4], "dtype": "float32"},
                       "token": {"shape": [1, 4], "dtype": "float32"}},
            "nodes": [{"id": "append", "opType": "Concat", "inputs": {"input0": "past", "input1": "token"},
                "outputs": {"out": {"tensor": "present", "shape": ["R", 4], "dtype": "float32"}}, "params": {"axis": 0}}],
            "outputs": ["present"]}
        graph = feedback_graph(graph, {"past": "present"}, invariants=["token"])
        (self.directory / "graph.json").write_text(json.dumps(graph))
        for backend in self.backends():
            with self.subTest(backend=backend), vx.InferenceSession(self.directory, backend=backend) as session:
                token = self.x[:1].copy()
                expected = np.concatenate((self.x, token))
                session.run_tensors({"past": self.x, "token": token, "position": np.array([0], np.int32)},
                    entrypoint="initialize", output_names=[]).close()
                token[:] = -100
                for step in range(1, 4):
                    expected = np.concatenate((expected, self.x[:1]))
                    with session.run_tensors({"position": np.array([step], np.int32)}, entrypoint="transition",
                            reuse_inputs=["token"], feedback={"past": "present"}) as output:
                        np.testing.assert_array_equal(output["present"].numpy(), expected)

    def test_feedback_alias_cycle_is_simultaneous(self):
        descriptor = {"shape": [2, 4], "dtype": "float32"}
        graph = {"format": "volvox-graph/v1", "dimensions": {}, "inputs": {"a": descriptor, "b": descriptor},
            "nodes": [{"id": name, "opType": "Reshape", "inputs": {"input": name},
                "outputs": {"out": dict(descriptor, tensor=name + "_out")}, "params": {"shape": [2, 4]}} for name in ("a", "b")],
            "outputs": ["a_out", "b_out"]}
        graph = feedback_graph(graph, {"a": "b_out", "b": "a_out"})
        (self.directory / "graph.json").write_text(json.dumps(graph))
        for backend in self.backends():
            with self.subTest(backend=backend), vx.InferenceSession(self.directory, backend=backend) as session:
                session.run_tensors({"a": self.x, "b": self.x + 100, "position": np.array([0], np.int32)},
                    entrypoint="initialize", output_names=[]).close()
                for i in range(4):
                    with session.run_tensors({"position": np.array([i + 1], np.int32)}, entrypoint="transition",
                            feedback={"a": "b_out", "b": "a_out"}) as output:
                        np.testing.assert_array_equal(output["a_out"].numpy(), self.x + (100 if i % 2 == 0 else 0))
                        np.testing.assert_array_equal(output["b_out"].numpy(), self.x + (0 if i % 2 == 0 else 100))

    def test_feedback_views_keep_mixed_cycles_and_invariants_independent(self):
        descriptor = {"shape": [2, 4], "dtype": "float32"}
        for cycle_size in (2, 65):
            names = [f"value_{index}" for index in range(cycle_size)]
            graph = {"format": "volvox-graph/v1", "dimensions": {},
                "inputs": {name: descriptor for name in [*names, "fixed"]},
                "nodes": [{"id": name, "opType": "Reshape", "inputs": {"input": name},
                    "outputs": {"out": dict(descriptor, tensor=name + "_out")},
                    "params": {"shape": [2, 4]}} for name in [*names, "fixed"]],
                "outputs": [name + "_out" for name in [*names, "fixed"]]}
            graph = feedback_graph(graph,
                {name: names[(index + 1) % cycle_size] + "_out"
                    for index, name in enumerate(names)}, invariants=["fixed"])
            (self.directory / "graph.json").write_text(json.dumps(graph))
            for backend in self.backends():
                with self.subTest(cycle_size=cycle_size, backend=backend), \
                        vx.InferenceSession(self.directory, backend=backend) as session:
                    original = [self.x + 100 * index for index in range(cycle_size)]
                    fixed = self.x + 10000
                    inputs = dict(zip(names, original), fixed=fixed,
                        position=np.array([0], np.int32))
                    retained = []
                    try:
                        retained.append(session.run_tensors(inputs, entrypoint="initialize"))
                        fixed[:] = -100
                        for step in range(1, 4):
                            retained.append(session.run_tensors(
                                {"position": np.array([step], np.int32)}, entrypoint="transition",
                                reuse_inputs=["fixed"],
                                feedback={name: names[(index + 1) % cycle_size] + "_out"
                                    for index, name in enumerate(names)}))
                            for earlier, output in enumerate(retained):
                                for index, name in enumerate(names):
                                    np.testing.assert_array_equal(output[name + "_out"].numpy(),
                                        original[(index + earlier) % cycle_size])
                                np.testing.assert_array_equal(output["fixed_out"].numpy(),
                                    self.x + 10000)
                    finally:
                        for output in reversed(retained):
                            output.close()


class ContextRetirementTest(unittest.TestCase):
    """Fresh public tasks remain isolated when native preparation is retained."""

    setUp = NativeTensorTest.setUp
    backends = ContextTensorStateTest.backends

    def assert_fresh(self, session, context_id=None):
        info = session._engine.get_execution_context_info(vx.pb.ExecutionContextRef(
            context_id=session._context.context_id if context_id is None else context_id))
        self.assertEqual(info.state, vx.pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_READY)
        self.assertFalse(info.initialized)
        self.assertFalse(info.bindings_available)
        self.assertEqual(info.total_steps, 0)
        self.assertEqual(info.active_lengths, [0] * info.slots)
        self.assertFalse(info.HasField("cache"))

    def test_dynamic_new_context_requires_fresh_state_and_preserves_outputs(self):
        model(self.directory, shape=["N", 4])
        graph = feedback_graph(json.loads((self.directory / "graph.json").read_text()), {"x": "y"})
        (self.directory / "graph.json").write_text(json.dumps(graph))
        second = np.arange(16, dtype=np.float32).reshape(4, 4) + 100
        position = np.array([0], dtype=np.int32)
        for backend in self.backends():
            retained = []
            with self.subTest(backend=backend), vx.Runtime() as runtime:
                with runtime.inference_session(self.directory, backend=backend) as session:
                    compiled_id = session.compiled_model_id
                    for values in (self.x, second, self.x[:1] - 20):
                        self.assert_fresh(session)
                        for entrypoint, options in (("initialize", {"reuse_inputs": ["x"]}),
                                ("initialize", {"feedback": {"x": "y"}}),
                                ("transition", {"feedback": {"x": "y"}})):
                            with self.assertRaises(vx.VolvoxAIError):
                                session.run_tensors({"position": position}, entrypoint=entrypoint, **options)
                            self.assert_fresh(session)
                        output = session.run_tensors({"x": values, "position": position},
                                                     entrypoint="initialize")["y"]
                        retained.append((output, values * 2))
                        with session.run_tensors({"position": position + 1}, entrypoint="transition",
                                                 feedback={"x": "y"}) as transitioned:
                            np.testing.assert_array_equal(transitioned["y"].numpy(), values * 4)
                        old_id = session._context.context_id
                        session.new_context()
                        self.assertNotEqual(session._context.context_id, old_id)
                        self.assertEqual(session.compiled_model_id, compiled_id)
                        with self.assertRaises(vx.VolvoxAIError) as disposed:
                            session._engine.get_execution_context_info(vx.pb.ExecutionContextRef(context_id=old_id))
                        self.assertEqual(disposed.exception.status, vx.pb.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED)
                        for tensor, expected in retained:
                            np.testing.assert_array_equal(tensor.numpy(), expected)
            # Closing both public owners must leave each independently held snapshot alive.
            for tensor, expected in retained:
                np.testing.assert_array_equal(tensor.numpy(), expected)
                tensor.close()

    def test_same_compiled_contexts_keep_distinct_bindings(self):
        model(self.directory, shape=["N", 4])
        for backend in self.backends():
            with self.subTest(backend=backend), vx.InferenceSession(self.directory, backend=backend) as session:
                peers = []

                def execute(context_id, values=None):
                    inputs = vx.pb.InputBindings(reuse=["x"]) if values is None else vx.pb.InputBindings(
                        values=[vx.pb.Tensor(name="x", shape=values.shape,
                            dtype=vx.pb.DataType.DATA_TYPE_F32, inline=values.tobytes())])
                    packet = session._engine.execute(vx.pb.ExecuteRequest(context_id=context_id, inputs=inputs,
                        scope=vx.pb.ExecutionScope(entrypoint="evaluate", full=vx.pb.Empty())))
                    return np.frombuffer(packet.outputs[0].inline, dtype=np.float32).reshape(packet.outputs[0].shape)

                with session.run_tensors(self.x)["y"] as retained:
                    try:
                        for count in (4, 1):
                            peer = session._engine.create_execution_context(vx.pb.CreateExecutionContextRequest(
                                compiled_model_id=session.compiled_model_id, graph=vx.pb.GraphExecution()))
                            peers.append((peer.context_id, np.full((count, 4), 100 + count, np.float32)))
                            self.assert_fresh(session, peer.context_id)
                            with self.assertRaises(vx.VolvoxAIError):
                                execute(peer.context_id)
                            np.testing.assert_array_equal(execute(*peers[-1]), peers[-1][1] * 2)
                            # Retire a successful context while the peers and its output remain live.
                            session.new_context()
                            self.assert_fresh(session)
                            current = np.full((2, 4), -count, np.float32)
                            np.testing.assert_array_equal(session.run(current)["y"], current * 2)
                        self.assertEqual(len({session._context.context_id, *(p[0] for p in peers)}), 3)
                        for context_id, values in peers:
                            np.testing.assert_array_equal(execute(context_id), values * 2)
                        np.testing.assert_array_equal(retained.numpy(), self.x * 2)
                    finally:
                        for context_id, _ in peers:
                            session._engine.release_execution_context(vx.pb.ExecutionContextRef(context_id=context_id))

    def prefix_model(self, *, dynamic=True):
        model(self.directory, shape=[1, "N" if dynamic else 4, 4])
        graph = json.loads((self.directory / "graph.json").read_text())
        graph["execution"]["entrypoints"][0]["allowed_ranges"] = [1, 2]
        (self.directory / "graph.json").write_text(json.dumps(graph))

    def check_fresh_prefix(self, backend):
        old = np.full((1, 4, 4), 123, np.float32)
        fresh = np.arange(16, dtype=np.float32).reshape(1, 4, 4) + 1
        with vx.InferenceSession(self.directory, backend=backend) as session:
            for _ in range(2):
                with session.run_tensors(old)["y"] as retained:
                    session.new_context()
                    self.assert_fresh(session)
                    with session.run_tensors(fresh, scope=vx.pb.ExecutionScope(entrypoint="evaluate",
                            prefix_rows=vx.pb.PrefixRows(count=2))) as output:
                        expected = fresh * 2
                        expected[:, 2:] = 0
                        np.testing.assert_array_equal(output["y"].numpy(), expected)
                    np.testing.assert_array_equal(retained.numpy(), old * 2)

    def test_cpu_dynamic_prefix_starts_with_zero_suffix(self):
        self.prefix_model()
        self.check_fresh_prefix("cpu")

    @unittest.skipUnless("cuda" in os.environ.get("VOLVOXAI_TEST_NATIVE_BUFFERS", "").split(","),
                         "requires explicit CUDA hardware")
    def test_cuda_static_prefix_refusal_preserves_fresh_and_bound_state(self):
        self.check_cuda_prefix_refusal(dynamic=False)

    @unittest.skipUnless("cuda" in os.environ.get("VOLVOXAI_TEST_NATIVE_BUFFERS", "").split(","),
                         "requires explicit CUDA hardware")
    def test_cuda_dynamic_prefix_refusal_preserves_fresh_and_bound_state(self):
        self.check_cuda_prefix_refusal(dynamic=True)

    def check_cuda_prefix_refusal(self, *, dynamic):
        self.prefix_model(dynamic=dynamic)
        old = np.arange(16, dtype=np.float32).reshape(1, 4, 4) + 1
        scope = vx.pb.ExecutionScope(entrypoint="evaluate", prefix_rows=vx.pb.PrefixRows(count=2))
        storages = (vx.pb.OutputStorage.OUTPUT_STORAGE_INLINE,
                    vx.pb.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER)
        for storage in storages:
            with self.subTest(storage=storage), vx.InferenceSession(self.directory, backend="cuda") as session:
                def state():
                    info = MessageToDict(session._engine.get_execution_context_info(vx.pb.ExecutionContextRef(
                        context_id=session._context.context_id)))
                    info.pop("report", None)
                    return info

                def refuse(values):
                    before = state()
                    with self.assertRaises(vx.VolvoxAIError) as caught:
                        session._engine.execute(vx.pb.ExecuteRequest(context_id=session._context.context_id,
                            inputs=vx.pb.InputBindings(values=[vx.pb.Tensor(name="x", shape=values.shape,
                                dtype=vx.pb.DataType.DATA_TYPE_F32, inline=values.tobytes())]),
                            scope=scope, outputs=vx.pb.OutputOptions(storage=storage)))
                    error = caught.exception
                    self.assertEqual(error.status, vx.pb.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED)
                    self.assertEqual(error.code, vx.pb.OperationCode.OPERATION_CODE_BACKEND_UNSUPPORTED)
                    self.assertEqual(error.stage, vx.pb.OperationStage.OPERATION_STAGE_EXECUTE)
                    self.assertEqual(error.response.outputs, [])
                    self.assertEqual(state(), before)

                self.assert_fresh(session)
                refuse(old)
                self.assert_fresh(session)
                with session.run_tensors(old)["y"] as retained:
                    shapes = ((1, 4, 4), (1, 3, 4)) if dynamic else ((1, 4, 4),)
                    for shape in shapes:
                        refuse(np.full(shape, 99, np.float32))
                        with session.run_tensors({}, reuse_inputs=["x"]) as output:
                            np.testing.assert_array_equal(output["y"].numpy(), old * 2)
                        np.testing.assert_array_equal(retained.numpy(), old * 2)
                    session.new_context()
                    self.assert_fresh(session)
                    refuse(old)
                    np.testing.assert_array_equal(retained.numpy(), old * 2)

    @unittest.skipUnless(any(name in os.environ.get("VOLVOXAI_TEST_NATIVE_BUFFERS", "").split(",")
                            for name in ("vulkan", "opengl")),
                         "requires explicit Vulkan or OpenGL hardware")
    def test_vulkan_opengl_prefix_requires_explicit_cpu_operator_fallback(self):
        self.prefix_model()
        values = np.arange(16, dtype=np.float32).reshape(1, 4, 4) + 1
        for backend in (name for name in self.backends() if name in ("vulkan", "opengl")):
            for fallback in (vx.pb.OperatorFallback.OPERATOR_FALLBACK_ALLOW,
                             vx.pb.OperatorFallback.OPERATOR_FALLBACK_FORBID):
                with self.subTest(backend=backend, fallback=fallback), vx.open_library() as host:
                    engine = vx.VxInferenceServiceClient(host)
                    runtime = engine.create_runtime(vx.pb.CreateRuntimeRequest(cpu_threads=1))
                    loaded = engine.load_model(vx.pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                        graph_path=str(self.directory / "graph.json")))
                    compiled = engine.compile_model(vx.pb.CompileModelRequest(model_id=loaded.model_id,
                        policy=vx.pb.BackendPolicy(mode=vx.pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                            backends=[backend], operator_fallback=fallback)))
                    context = engine.create_execution_context(vx.pb.CreateExecutionContextRequest(
                        compiled_model_id=compiled.compiled_model_id, graph=vx.pb.GraphExecution()))
                    # Hybrid execution returns inline values; owned device buffers
                    # require FORBID and cannot stand in for the hybrid contract.
                    request = vx.pb.ExecuteRequest(context_id=context.context_id,
                        inputs=vx.pb.InputBindings(values=[vx.pb.Tensor(name="x", shape=values.shape,
                            dtype=vx.pb.DataType.DATA_TYPE_F32, inline=values.tobytes())]),
                        scope=vx.pb.ExecutionScope(entrypoint="evaluate", prefix_rows=vx.pb.PrefixRows(count=2)))
                    if fallback == vx.pb.OperatorFallback.OPERATOR_FALLBACK_FORBID:
                        with self.assertRaises(vx.VolvoxAIError) as caught:
                            engine.execute(request)
                        self.assertEqual(caught.exception.status,
                            vx.pb.NativeStatus.NATIVE_STATUS_OPERATOR_FALLBACK_FORBIDDEN)
                        self.assertEqual(caught.exception.code,
                            vx.pb.OperationCode.OPERATION_CODE_OPERATOR_FALLBACK_FORBIDDEN)
                        self.assertEqual(caught.exception.response.outputs, [])
                    else:
                        result = engine.execute(request)
                        self.assertTrue(result.report.fallback.operator_fallback_used)
                        actual = np.frombuffer(result.outputs[0].inline, np.float32).reshape(values.shape)
                        expected = values * 2
                        expected[:, 2:] = 0
                        np.testing.assert_array_equal(actual, expected)

    def test_memory_trace_retires_each_context_allocation_identity(self):
        model(self.directory, shape=["N", 4])
        for backend in self.backends():
            with self.subTest(backend=backend), vx.InferenceSession(self.directory, backend=backend) as session:
                with session.trace(memory=True) as prior:
                    with session.run_tensors(self.x) as output:
                        np.testing.assert_array_equal(output["y"].numpy(), self.x * 2)
                frozen_events = [MessageToDict(event) for event in prior.events]
                # Retirement after StopTrace must not recycle an observed engine
                # with its previous allocation observer or mutate the completed trace.
                session.new_context()
                context_ids = []
                with session.trace(detail="nodes", memory=True, execution_plans=True) as trace:
                    for count in (1, 4, 2):
                        context_ids.append(session._context.context_id)
                        values = np.arange(count * 4, dtype=np.float32).reshape(count, 4)
                        with session.run_tensors(values) as output:
                            np.testing.assert_array_equal(output["y"].numpy(), values * 2)
                        session.new_context()
                self.assertEqual(trace.info.state, vx.pb.TraceState.TRACE_STATE_READY)
                self.assertEqual(trace.info.events.dropped, 0)
                memory = [event for event in trace.events if event.HasField("memory")]
                introduced = [event.memory.allocation_id for event in memory
                    if event.memory.action != vx.pb.TraceMemoryAction.TRACE_MEMORY_ACTION_FREE]
                freed = [event.memory.allocation_id for event in memory
                    if event.memory.action == vx.pb.TraceMemoryAction.TRACE_MEMORY_ACTION_FREE]
                self.assertTrue(introduced)
                self.assertEqual(len(introduced), len(set(introduced)))
                self.assertCountEqual(freed, introduced)
                self.assertTrue(all(row.accounting_complete and row.live_bytes == 0 for row in trace.info.allocators))
                executed = {event.lineage.context_id for event in trace.events if event.name == "Execute"}
                self.assertEqual(executed, set(context_ids))
                # A stopped collector must not retain or contaminate the next ordinary task.
                session.new_context()
                np.testing.assert_array_equal(session.run(self.x)["y"], self.x * 2)
                self.assertEqual([MessageToDict(event) for event in prior.events], frozen_events)


@unittest.skipUnless(os.environ.get("VOLVOXAI_TEST_NATIVE_BUFFERS"), "requires an explicit native GPU test host")
class GpuNativeBufferTest(unittest.TestCase):
    setUp = NativeTensorTest.setUp

    def backends(self):
        return os.environ["VOLVOXAI_TEST_NATIVE_BUFFERS"].split(",")

    def test_closing_a_peer_owner_preserves_live_device_resources(self):
        for backend in self.backends():
            with self.subTest(backend=backend), vx.InferenceSession(self.directory,
                    backend=backend) as inference:
                with inference.run_tensors(self.x) as retained:
                    for _ in range(3):
                        with vx.InferenceSession(self.directory, backend=backend) as other:
                            with other.run_tensors(self.x) as output:
                                np.testing.assert_array_equal(output["y"].numpy(), self.x * 2)
                        np.testing.assert_array_equal(retained["y"].numpy(), self.x * 2)
                        with inference.run_tensors(self.x + 1) as output:
                            np.testing.assert_array_equal(output["y"].numpy(), (self.x + 1) * 2)

    def test_idle_session_survives_another_owner_closing(self):
        for backend in self.backends():
            with self.subTest(backend=backend), vx.InferenceSession(self.directory, backend=backend) as idle:
                with vx.InferenceSession(self.directory, backend=backend) as active:
                    with active.run_tensors(self.x)["y"] as tensor:
                        np.testing.assert_array_equal(tensor.numpy(), self.x * 2)
                with idle.run_tensors(self.x)["y"] as tensor:
                    np.testing.assert_array_equal(tensor.numpy(), self.x * 2)

    def test_cross_session_buffers_outlive_context_and_reject_wrong_device(self):
        for backend in self.backends():
            with self.subTest(backend=backend), vx.Runtime() as runtime:
                with runtime.inference_session(self.directory, backend=backend) as producer:
                    first = producer.run_tensors(self.x)["y"]
                with runtime.inference_session(self.directory, backend=backend) as consumer:
                    for _ in range(3):
                        with consumer.run_tensors(first)["y"] as second:
                            np.testing.assert_array_equal(second.numpy(), self.x * 4)
                    lease = consumer._buffers.begin_buffer_access(vx.pb.BufferAccessRequest(
                        view=first._get_storage().handle.buffer))
                    lease.memory.resource.device_id += 1
                    try:
                        with self.assertRaises(vx.VolvoxAIError):
                            consumer._engine.execute(vx.pb.ExecuteRequest(
                                context_id=consumer._context.context_id, inputs=vx.pb.InputBindings(values=[vx.pb.Tensor(
                                    name="x", shape=self.x.shape, dtype=vx.pb.DataType.DATA_TYPE_F32,
                                    borrowed=lease.memory)]), scope=vx.pb.ExecutionScope(entrypoint="evaluate", full=vx.pb.Empty()), outputs=vx.pb.OutputOptions(storage=vx.pb.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER)))
                    finally:
                        consumer._buffers.end_buffer_access(vx.pb.EndBufferAccessRequest(access_ids=[lease.access_id]))
                np.testing.assert_array_equal(first.numpy(), self.x * 2)
                if backend in ("vulkan", "opengl", "metal"):
                    with self.assertRaises(BufferError):
                        first.__dlpack__()
                first.close()


    def test_dynamic_buffer_reuse_preserves_every_retained_output(self):
        model(self.directory, shape=["N", 4])
        for backend in self.backends():
            retained = []
            with vx.Runtime() as runtime, \
                 runtime.inference_session(self.directory, backend=backend) as producer, \
                 runtime.inference_session(self.directory, backend=backend) as consumer:
                for count in (1, 4, 2, 3, 1):
                    x = np.arange(count * 4, dtype=np.float32).reshape(count, 4)
                    with producer.run_tensors(x)["y"] as first:
                        retained.append((consumer.run_tensors(first)["y"], x * 4))
            for tensor, expected in retained:
                np.testing.assert_array_equal(tensor.numpy(), expected)
                tensor.close()

    def test_int32_storage_preserves_large_values(self):
        model(self.directory, dtype="int32")
        graph = json.loads((self.directory / "graph.json").read_text())
        graph["nodes"][0].update(opType="Cast", inputs={"input": "x"}, params={"to": "int32"})
        (self.directory / "graph.json").write_text(json.dumps(stateless_graph(graph)))
        x = np.array([[-16_777_219, -3, 0, 16_777_217]] * 2, dtype=np.int32)
        for backend in self.backends():
            with self.subTest(backend=backend), vx.InferenceSession(self.directory, backend=backend) as session:
                with session.run_tensors(x)["y"] as first, session.run_tensors(first)["y"] as second:
                    np.testing.assert_array_equal(second.numpy(), x)

    def test_quantized_model_matches_cpu(self):
        from safetensors.numpy import save_file, load_file
        weight = np.arange(16, dtype=np.float32).reshape(4, 4) / 32 - .25
        save_file({"weight": weight}, self.directory / "model.safetensors")
        graph = json.loads((self.directory / "graph.json").read_text())
        graph["nodes"][0].update(opType="Linear", inputs={"input": "x", "weight": "weight"},
                                 params={"weight_layout": "din_dout"})
        (self.directory / "graph.json").write_text(json.dumps(stateless_graph(graph)))
        x = self.x / 8 - .5
        quantized = self.directory / "int8"
        vx.quantize(self.directory, calibration_data=[x, x + .1], output=quantized, group_size=0, static_activations=vx.pb.StaticActivations(dtype=vx.pb.DataType.DATA_TYPE_I8, scheme=vx.pb.AffineScheme.AFFINE_SCHEME_SYMMETRIC, reduce_weight_range=False))
        self.assertTrue(any(value.dtype == np.int8 for value in
                            load_file(quantized / "model.safetensors").values()))
        with vx.InferenceSession(quantized) as cpu:
            expected = cpu.run(x)["y"]
            expected_twice = cpu.run(expected)["y"]
        for backend in self.backends():
            with self.subTest(backend=backend), \
                 vx.InferenceSession(quantized, backend=backend) as session:
                with session.run_tensors(x)["y"] as first, session.run_tensors(first)["y"] as second:
                    np.testing.assert_allclose(first.numpy(), expected, rtol=1e-5, atol=1e-6)
                    np.testing.assert_allclose(second.numpy(), expected_twice, rtol=1e-5, atol=1e-6)

    def test_int8_payload_has_exact_logical_bytes(self):
        from safetensors.numpy import save_file
        model(self.directory, dtype="int8", shape=[1, 7])
        graph = json.loads((self.directory / "graph.json").read_text())
        graph["nodes"][0].update(opType="Transpose", inputs={"input": "x"}, params={"perm": [0, 1]})
        descriptor = {"scheme": "per_tensor", "scale_tensor": "scale", "zero_point_tensor": "zero"}
        graph["quantization"] = {"format": "volvox-affine-safetensors/v1",
                                  "tensors": {"x": descriptor, "y": descriptor}}
        save_file({"scale": np.array([.02], dtype=np.float32), "zero": np.array([0], dtype=np.int8)},
                  self.directory / "model.safetensors")
        (self.directory / "graph.json").write_text(json.dumps(stateless_graph(graph)))
        x = np.array([[-128, -127, -3, 0, 1, 126, 127]], dtype=np.int8)
        for backend in self.backends():
            if backend == "metal":
                continue  # The macOS blit adapter declares a four-byte span requirement.
            with self.subTest(backend=backend), vx.InferenceSession(self.directory, backend=backend) as session:
                with session.run_tensors(x)["y"] as first, session.run_tensors(first)["y"] as second:
                    np.testing.assert_array_equal(first.numpy(), x)
                    np.testing.assert_array_equal(second.numpy(), x)
                    self.assertEqual(second._get_storage().handle.buffer.length_bytes, 7)

    def test_released_pool_address_is_not_a_live_device_tensor(self):
        for backend in self.backends():
            with self.subTest(backend=backend), vx.InferenceSession(self.directory, backend=backend) as session:
                tensor = session.run_tensors(self.x)["y"]
                descriptor = tensor._get_storage().handle
                tensor.close()
                descriptor.name = "x"
                with self.assertRaises(vx.VolvoxAIError):
                    session._engine.execute(vx.pb.ExecuteRequest(
                        context_id=session._context.context_id, inputs=vx.pb.InputBindings(values=[descriptor]), scope=vx.pb.ExecutionScope(entrypoint="evaluate", full=vx.pb.Empty()), outputs=vx.pb.OutputOptions(storage=vx.pb.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER)))
                with session.run_tensors(self.x) as recovered:
                    np.testing.assert_array_equal(recovered["y"].numpy(), self.x * 2)

    def test_owned_subview_and_invalid_extent(self):
        model(self.directory, shape=["N", 4])
        for backend in self.backends():
            with self.subTest(backend=backend), vx.InferenceSession(self.directory, backend=backend) as session:
                with session.run_tensors(self.x)["y"] as first:
                    source = first._get_storage().handle
                    for offset, length in ((16, 16), (source.buffer.length_bytes, 16), (2, 16)):
                        view = vx.pb.BufferView(buffer_id=source.buffer.buffer_id,
                            offset_bytes=offset, length_bytes=length)
                        request = vx.pb.ExecuteRequest(context_id=session._context.context_id, inputs=vx.pb.InputBindings(values=[
                            vx.pb.Tensor(name="x", shape=[1, 4], dtype=source.dtype, buffer=view)]), scope=vx.pb.ExecutionScope(entrypoint="evaluate", full=vx.pb.Empty()), outputs=vx.pb.OutputOptions(storage=vx.pb.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER))
                        if offset != 16:
                            with self.assertRaises(vx.VolvoxAIError):
                                session._engine.execute(request)
                        else:
                            batch = session._engine.execute(request)
                            try:
                                read = session._buffers.copy_tensors(vx.pb.CopyTensorsRequest(sources=batch.outputs, inline_result=True))
                                actual = np.frombuffer(read.outputs[0].inline, dtype=np.float32).reshape(1, 4)
                                np.testing.assert_array_equal(actual, self.x[1:] * 4)
                            finally:
                                session._buffers.release_buffers(vx.pb.BufferRefs(buffer_ids=[item.buffer.buffer_id for item in batch.outputs]))



@unittest.skipUnless(os.environ.get("VOLVOXAI_TEST_CUDA") == "1", "requires an explicit CUDA test host")
class CudaTensorTest(unittest.TestCase):
    setUp = NativeTensorTest.setUp

    def test_dynamic_context_retirement_preserves_exported_dlpack(self):
        import torch
        model(self.directory, shape=["N", 4])
        with vx.Runtime() as runtime:
            with runtime.inference_session(self.directory, backend="cuda") as session:
                tensor = session.run_tensors(self.x)["y"]
                storage = weakref.ref(tensor._get_storage())
                owner = tensor._get_storage().owner
                exported = torch.from_dlpack(tensor)
                tensor.close()
                self.assertIsNone(storage())
                for count in (4, 1, 3):
                    session.new_context()
                    values = np.full((count, 4), count + 10, np.float32)
                    with session.run_tensors(values) as output:
                        np.testing.assert_array_equal(output["y"].numpy(), values * 2)
                    np.testing.assert_array_equal(exported.cpu().numpy(), self.x * 2)
        # The C export owns the allocation; its independent module pin outlives
        # the Python storage wrapper, context, and runtime handles.
        self.assertEqual(owner.references, 1)
        np.testing.assert_array_equal(exported.cpu().numpy(), self.x * 2)
        del exported
        gc.collect()
        self.assertEqual(owner.references, 0)

    def test_scoped_multistream_completion_is_local_to_the_allocation(self):
        import torch
        with vx.InferenceSession(
                self.directory, backend="cuda") as session, \
                session.run_tensors(self.x)["y"] as tensor, \
                session.run_tensors(self.x + 20)["y"] as unrelated:
            np.testing.assert_array_equal(unrelated.numpy(), (self.x + 20) * 2)
            first, second = torch.cuda.Stream(), torch.cuda.Stream()
            edge, done = torch.cuda.Event(), torch.cuda.Event()
            # Warm PyTorch's kernels before measuring pending GPU work.
            torch.ones(8, device="cuda").add_(3)
            torch.cuda.synchronize()
            with tensor.torch_access(streams=[first, second]) as view:
                with torch.cuda.stream(first):
                    torch.cuda._sleep(600_000_000)
                    view.add_(3)
                    edge.record()
                with torch.cuda.stream(second):
                    second.wait_event(edge)
                    view.add_(5)
                    done.record()
            self.assertFalse(done.query(), "EndBufferAccess must not wait on the host")
            # This catches eagerly queueing the wait on the shared engine
            # stream, which would stall unrelated VolvoxAI buffers too.
            np.testing.assert_array_equal(unrelated.numpy(), (self.x + 20) * 2)
            self.assertFalse(done.query(), "unrelated storage inherited the consumer dependency")
            with session.run_tensors(tensor)["y"] as result:
                np.testing.assert_array_equal(result.numpy(), (self.x * 2 + 8) * 2)
            del view  # Its permission already ended, even while it lived.

    def test_scoped_reader_keeps_pending_work_safe_during_pool_reuse(self):
        import torch
        with vx.InferenceSession(
                self.directory, backend="cuda") as session:
            stream = torch.cuda.Stream()
            copied = torch.empty(self.x.shape, dtype=torch.float32, device="cuda")
            for index in range(3):
                tensor = session.run_tensors(self.x + index)["y"]
                done = torch.cuda.Event()
                with tensor.torch_access(streams=[stream]) as view:
                    address = view.data_ptr()
                    with torch.cuda.stream(stream):
                        torch.cuda._sleep(300_000_000)
                        copied.copy_(view, non_blocking=True)
                        done.record()
                    tensor.close()
                    del view
                self.assertFalse(done.query())
                with session.run_tensors(self.x - 100)["y"] as replacement:
                    self.assertEqual(native_address(replacement), address)
                    np.testing.assert_array_equal(replacement.numpy(), (self.x - 100) * 2)
                done.synchronize()
                np.testing.assert_array_equal(copied.cpu().numpy(), (self.x + index) * 2)

    def test_scoped_default_stream_mutation_and_exception_cleanup(self):
        import torch
        with vx.InferenceSession(self.directory, backend="cuda") as session, \
                session.run_tensors(self.x)["y"] as tensor:
            with self.assertRaisesRegex(RuntimeError, "consumer error"):
                with tensor.torch_access() as view:
                    view.add_(7)
                    raise RuntimeError("consumer error")
            np.testing.assert_array_equal(tensor.numpy(), self.x * 2 + 7)
            with self.assertRaises(ValueError):
                with tensor.torch_access(streams=[]):
                    pass
            del view

    def test_invalid_cuda_evidence_keeps_the_registered_writer(self):
        import torch
        with vx.InferenceSession(self.directory, backend="cuda") as session, \
                session.run_tensors(self.x)["y"] as tensor:
            storage = tensor._get_storage()
            lease = storage.engine.begin_buffer_access(vx.pb.BufferAccessRequest(
                view=storage.handle.buffer, mode=vx.pb.BufferAccessMode.BUFFER_ACCESS_MODE_WRITE))
            try:
                for streams, device in (([], 0), ([0], 0), ([2], 0), ([1] * 65, 0), ([1], 1)):
                    with self.subTest(streams=streams, device=device), self.assertRaises(vx.VolvoxAIError):
                        storage.engine.end_buffer_access(vx.pb.EndBufferAccessRequest(
                            access_ids=[lease.access_id], cuda=vx.pb.CudaStreamCompletion(device_id=device, streams=streams)))
                    with self.assertRaises(vx.VolvoxAIError) as busy:
                        tensor.numpy()
                    self.assertEqual(busy.exception.report.status, vx.pb.NativeStatus.NATIVE_STATUS_BUSY)
            finally:
                storage.engine.end_buffer_access(vx.pb.EndBufferAccessRequest(
                    access_ids=[lease.access_id], cuda=vx.pb.CudaStreamCompletion(streams=[1])))
            np.testing.assert_array_equal(tensor.numpy(), self.x * 2)

    def test_scoped_import_outlives_owner_and_pins_foreign_deleter(self):
        import torch
        with vx.Runtime() as runtime, runtime.inference_session(self.directory, backend="cuda") as session:
            producer = torch.as_tensor(self.x, device="cuda")
            tensor = runtime.tensor(producer, copy=False)
            stream, done = torch.cuda.Stream(), torch.cuda.Event()
            destination = torch.empty_like(producer)
            session.close()
            runtime.close()
            with tensor.torch_access(streams=[stream]) as view:
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(300_000_000)
                    view.add_(9)
                    destination.copy_(view, non_blocking=True)
                    done.record()
                tensor.close()
                del producer
            # The escaped alias pins the imported storage and library.
            self.assertFalse(done.query())
            del view
            self.assertTrue(done.query(), "final foreign release must wait for recorded work")
            np.testing.assert_array_equal(destination.cpu().numpy(), self.x + 9)

    def test_capturing_stream_rejected_before_permission_retirement(self):
        import torch
        with vx.InferenceSession(self.directory, backend="cuda") as session, \
                session.run_tensors(self.x)["y"] as tensor:
            storage = tensor._get_storage()
            lease = storage.engine.begin_buffer_access(vx.pb.BufferAccessRequest(
                view=storage.handle.buffer, mode=vx.pb.BufferAccessMode.BUFFER_ACCESS_MODE_WRITE))
            stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
            temporary = torch.ones(8, device="cuda")
            torch.cuda.synchronize()
            try:
                with torch.cuda.graph(graph, stream=stream):
                    temporary.add_(1)
                    with self.assertRaises(vx.VolvoxAIError) as invalid:
                        storage.engine.end_buffer_access(vx.pb.EndBufferAccessRequest(
                            access_ids=[lease.access_id], cuda=vx.pb.CudaStreamCompletion(streams=[stream.cuda_stream])))
                    self.assertEqual(invalid.exception.report.status, vx.pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT)
                with self.assertRaises(vx.VolvoxAIError):
                    tensor.numpy()
            finally:
                storage.engine.end_buffer_access(vx.pb.EndBufferAccessRequest(
                    access_ids=[lease.access_id], cuda=vx.pb.CudaStreamCompletion(streams=[1])))
            np.testing.assert_array_equal(tensor.numpy(), self.x * 2)

    def test_host_transfers_cross_the_pinned_staging_capacity(self):
        shape = [2049, 1024]  # More than the bounded 8 MiB transfer slab.
        model(self.directory, shape=shape)
        x = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) % 97) / 8
        with vx.InferenceSession(
                self.directory, backend="cuda") as session:
            with session.run_tensors(x)["y"] as first:
                with session.run_tensors(x + 3)["y"] as second:
                    np.testing.assert_array_equal(first.numpy(), x * 2)
                    np.testing.assert_array_equal(second.numpy(), (x + 3) * 2)

    def test_imported_torch_storage_outlives_native_sessions_and_runtime(self):
        import torch
        with vx.Runtime() as runtime:
            with runtime.inference_session(self.directory, backend="cuda"):
                producer = torch.as_tensor(self.x, device="cuda")
                imported = runtime.tensor(producer, copy=False)
            del producer
            gc.collect()
            np.testing.assert_array_equal(imported.numpy(), self.x)
            runtime.close()
            np.testing.assert_array_equal(imported.numpy(), self.x)
            shared = torch.from_dlpack(imported)
            imported.close()
            torch.testing.assert_close(shared.cpu(), torch.from_numpy(self.x), rtol=0, atol=0)
            del shared
            gc.collect()

    def test_dynamic_cuda_reuse_across_sessions(self):
        import torch
        model(self.directory, shape=["N", 4])
        retained = []
        with vx.Runtime() as runtime, runtime.inference_session(self.directory, backend="cuda") as producer, \
             runtime.inference_session(self.directory, backend="cuda") as consumer:
            for count in (1, 4, 2, 3, 1):
                x = torch.arange(count * 4, device="cuda", dtype=torch.float32).reshape(count, 4)
                with producer.run_tensors(x)["y"] as first:
                    with consumer.run_tensors(first)["y"] as second:
                        retained.append((torch.from_dlpack(second), x * 4))
        for actual, expected in retained:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cuda_rejects_cpu_pointer_claiming_device_memory(self):
        with vx.InferenceSession(self.directory, backend="cuda") as session:
            with self.assertRaises(vx.VolvoxAIError):
                session._engine.execute(vx.pb.ExecuteRequest(
                    context_id=session._context.context_id, inputs=vx.pb.InputBindings(values=[vx.pb.Tensor(name="x", shape=self.x.shape,
                        dtype=vx.pb.DataType.DATA_TYPE_F32, borrowed=vx.pb.BorrowedBuffer(
                            resource=vx.pb.NativeResource(kind=vx.pb.NativeResourceKind.NATIVE_RESOURCE_KIND_CUDA,
                                handle=self.x.ctypes.data, size_bytes=self.x.nbytes), length_bytes=self.x.nbytes))]), scope=vx.pb.ExecutionScope(entrypoint="evaluate", full=vx.pb.Empty()), outputs=vx.pb.OutputOptions(storage=vx.pb.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER)))
            with session.run_tensors(self.x)["y"] as result:
                with self.assertRaisesRegex(BufferError, "no-synchronization"):
                    result.__dlpack__(stream=-1)
                np.testing.assert_array_equal(result.numpy(), self.x * 2)


    def test_cuda_int32_values_keep_exact_storage(self):
        import torch
        model(self.directory, dtype="int32")
        graph = json.loads((self.directory / "graph.json").read_text())
        graph["nodes"][0]["opType"] = "Cast"
        graph["nodes"][0]["inputs"] = {"input": "x"}
        graph["nodes"][0]["params"] = {"to": "int32"}
        (self.directory / "graph.json").write_text(json.dumps(stateless_graph(graph)))
        x = torch.tensor([[-16_777_219, -3, 0, 16_777_217]] * 2, dtype=torch.int32, device="cuda")
        with vx.InferenceSession(self.directory, backend="cuda") as session, \
             session.run_tensors(x)["y"] as tensor:
            shared = torch.from_dlpack(tensor)
            self.assertEqual(shared.dtype, torch.int32)
            torch.testing.assert_close(shared, x, rtol=0, atol=0)

    def test_quantized_package_uses_cuda_tensor_outputs(self):
        import torch
        from safetensors.numpy import save_file, load_file
        weight = np.arange(16, dtype=np.float32).reshape(4, 4) / 32 - .25
        save_file({"weight": weight}, self.directory / "model.safetensors")
        graph = json.loads((self.directory / "graph.json").read_text())
        graph["nodes"][0].update(opType="Linear", inputs={"input": "x", "weight": "weight"},
                                 params={"weight_layout": "din_dout"})
        (self.directory / "graph.json").write_text(json.dumps(stateless_graph(graph)))
        x = self.x / 8 - .5
        quantized = self.directory / "int8"
        vx.quantize(self.directory, calibration_data=[x, x + .1], output=quantized, group_size=0, static_activations=vx.pb.StaticActivations(dtype=vx.pb.DataType.DATA_TYPE_I8, scheme=vx.pb.AffineScheme.AFFINE_SCHEME_SYMMETRIC, reduce_weight_range=False))
        self.assertTrue(any(value.dtype == np.int8 for value in
                            load_file(quantized / "model.safetensors").values()))
        with vx.InferenceSession(quantized) as cpu:
            expected = cpu.run(x)["y"]
        with vx.InferenceSession(quantized, backend="cuda") as session, \
             session.run_tensors(torch.as_tensor(x, device="cuda"))["y"] as tensor:
            self.assertEqual(tensor.device, "cuda:0")
            np.testing.assert_allclose(tensor.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_torch_streams_snapshots_lifetime_and_pointer_identity(self):
        import torch
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), vx.InferenceSession(self.directory, backend="cuda") as session:
            x = torch.arange(8, device="cuda", dtype=torch.float32).reshape(2, 4) * 3 + 1
            tensor = session.run_tensors(x)["y"]
            address = native_address(tensor)
            with session.run_tensors(tensor)["y"] as other:
                np.testing.assert_array_equal(other.numpy(), (self.x * 3 + 1) * 4)
            shared = torch.from_dlpack(tensor)
            self.assertEqual(shared.data_ptr(), address)
            with self.assertRaises(vx.VolvoxAIError):
                session.run_tensors(tensor)
        consumer_stream = torch.cuda.Stream()
        with torch.cuda.stream(consumer_stream):
            second_view = shared.view_as(shared)
        tensor.close()
        consumer_stream.synchronize()
        torch.testing.assert_close(shared, x * 2, rtol=0, atol=0)
        del shared
        torch.testing.assert_close(second_view, x * 2, rtol=0, atol=0)
        del second_view


    def test_recycled_output_waits_for_external_stream_reader(self):
        import torch
        with vx.InferenceSession(self.directory, backend="cuda") as session:
            for i in range(4):
                tensor = session.run_tensors(self.x + i)["y"]
                address = native_address(tensor)
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    view = torch.from_dlpack(tensor)
                    copied = torch.empty_like(view)
                    # Keep the read queued while its external lease is
                    # released, so premature pool reuse corrupts the copy.
                    torch.cuda._sleep(300_000_000)
                    copied.copy_(view, non_blocking=True)
                tensor.close()
                del view
                self.assertTrue(stream.query(), "ending the external lease must drain unknown CUDA work")
                with session.run_tensors(self.x - 100)["y"] as replacement:
                    self.assertEqual(native_address(replacement), address)
                    np.testing.assert_array_equal(replacement.numpy(), (self.x - 100) * 2)
                stream.synchronize()
                np.testing.assert_array_equal(copied.cpu().numpy(), (self.x + i) * 2)

    def test_torch_input_on_nondefault_stream(self):
        import torch
        with vx.InferenceSession(self.directory, backend="cuda") as session:
            with torch.cuda.stream(torch.cuda.Stream()):
                x = torch.arange(8, device="cuda", dtype=torch.float32).reshape(2, 4) + 7
                tensor = session.run_tensors(x)["y"]
            np.testing.assert_array_equal(tensor.numpy(), (self.x + 7) * 2)
            tensor.close()

    def test_borrowed_alias_of_owned_storage_waits_for_its_producer(self):
        import torch
        with vx.InferenceSession(
                self.directory, backend="cuda") as session:
            with session.run_tensors(self.x)["y"] as owned:
                # Warm a second allocation so cold allocation cannot hide
                # a missing producer dependency through an implicit wait.
                session.run_tensors(self.x)["y"].close()
                with torch.cuda.stream(torch.cuda.Stream()):
                    alias = torch.from_dlpack(owned)
                    torch.cuda._sleep(100_000_000)
                    alias.add_(7)
                    with session.run_tensors(alias)["y"] as result:
                        np.testing.assert_array_equal(result.numpy(), (self.x * 2 + 7) * 2)
                del alias

    def test_cuda_rejects_foreign_device_and_noncontiguous_input(self):
        import torch
        with vx.InferenceSession(self.directory, backend="cuda") as session:
            x = torch.arange(16, device="cuda", dtype=torch.float32).reshape(2, 8)[:, ::2]
            with self.assertRaisesRegex(BufferError, "contiguous"):
                session.run_tensors(x)
            x = x.contiguous()
            with vx.InferenceSession(self.directory) as cpu:
                with self.assertRaisesRegex(BufferError, "same device"):
                    cpu.run_tensors(x)
            with session.run_tensors(x)["y"] as result:
                np.testing.assert_array_equal(result.numpy(), x.cpu().numpy() * 2)

    def test_warm_cuda_execution_has_only_device_to_device_transfers(self):
        import torch
        x = torch.arange(8, device="cuda", dtype=torch.float32).reshape(2, 4)
        evidence = {"case": "float32 Add, shape [2, 4], four warmup calls",
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda}
        with vx.InferenceSession(self.directory, backend="cuda") as session:
            for _ in range(4):
                session.run_tensors(x)["y"].close()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                tensor = session.run_tensors(x)["y"]
            trace = self.directory / "trace.json"
            profiler.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())["traceEvents"]
            copies = [event["name"] for event in events if "memcpy" in event.get("name", "").lower()
                      and event.get("cat") == "gpu_memcpy"]
            self.assertTrue(copies, "CUDA profiler did not record device transfers")
            self.assertTrue(all("DtoD" in name for name in copies), copies)
            evidence["gpu_copy_events"] = copies
            np.testing.assert_array_equal(tensor.numpy(), self.x * 2)
            tensor.close()
        if destination := os.environ.get("VOLVOXAI_TENSOR_TEST_EVIDENCE"):
            Path(destination).write_text(json.dumps(evidence, indent=2) + "\n")

    def test_subprocess_shutdown_with_live_torch_view(self):
        script = """import torch, volvoxai as vx
with vx.InferenceSession(MODEL, backend='cuda') as session:
    tensor = session.run_tensors(torch.ones((2,4), device='cuda'))['y']
    shared = torch.from_dlpack(tensor)
tensor.close()
assert shared.sum().item() == 16
""".replace("MODEL", repr(str(self.directory)))
        result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    faulthandler.dump_traceback_later(60, exit=True)
    try:
        unittest.main(verbosity=2)
    finally:
        faulthandler.cancel_dump_traceback_later()
