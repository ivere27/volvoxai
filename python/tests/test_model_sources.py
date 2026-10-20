"""Public LoadModel manifests: integrity, budgets and accepted snapshot lifetime."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import volvoxai as vx
from execution_fixture import stateless_graph


def storage_shard(entries):
    """Serialize declared storage directly, without a dtype-converting exporter."""
    header, payload = {}, bytearray()
    for name, dtype, shape, data in entries:
        begin = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [begin, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


class BF16ModelSourcesTest(unittest.TestCase):
    """Storage conversion preserves source ownership and logical F32 execution."""

    def test_every_bf16_pattern_and_context_lifetime(self):
        bits = np.arange(65536, dtype="<u2")
        expected_bits = bits.astype("<u4") << 16
        shard = storage_shard([("w", "BF16", [65536], bits.tobytes())])
        graph = stateless_graph({
            "format": "volvox-graph/v1", "dimensions": {},
            "inputs": {"x": {"shape": [1], "dtype": "float32"}},
            "nodes": [{"id": "identity", "opType": "Identity", "inputs": {"input": "x"},
                       "outputs": {"out": {"tensor": "y", "shape": [1], "dtype": "float32"}},
                       "params": {}}], "outputs": ["w", "y"],
        })
        for override in (False, True):
            # Later F32 storage must supersede an earlier BF16 source, even
            # though the final revision requires no BF16 conversion for w.
            expected = np.full(65536, 0x3e800000, dtype="<u4") if override else expected_bits
            shards = [shard]
            if override:
                shards.append(storage_shard([("w", "F32", [65536], expected.tobytes())]))
            for library in ("libvolvoxai-lite.so", "libvolvoxai.so"):
                with self.subTest(library=library, override=override), vx.open_library(
                        Path(__file__).resolve().parents[2] / "native" / library) as host:
                    api = vx.VxInferenceServiceClient(host)
                    p = vx.pb
                    runtime = api.create_runtime(p.CreateRuntimeRequest(cpu_threads=1))
                    model = api.load_model(p.LoadModelRequest(runtime_id=runtime.runtime_id,
                        package=p.ModelPackage(graph_document=json.dumps(graph).encode(),
                                               weight_shards=shards)))
                    compiled = api.compile_model(p.CompileModelRequest(model_id=model.model_id))
                    contexts = [api.create_execution_context(p.CreateExecutionContextRequest(
                        compiled_model_id=compiled.compiled_model_id, graph=p.GraphExecution()))
                        for _ in range(2)]
                    # Context leases retain the exact shared conversion after their parents retire.
                    api.release_compiled_model(p.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                    api.release_model(p.ModelRef(model_id=model.model_id))
                    for context in contexts:
                        result = api.execute(p.ExecuteRequest(context_id=context.context_id,
                            inputs=p.InputBindings(values=[p.Tensor(name="x", shape=[1],
                                dtype=p.DataType.DATA_TYPE_F32, inline=struct.pack("<f", 1.0))]),
                            scope=p.ExecutionScope(entrypoint="evaluate", full=p.Empty()),
                            outputs=p.OutputOptions(storage=p.OutputStorage.OUTPUT_STORAGE_INLINE)))
                        output = next(t for t in result.outputs if t.name == "w")
                        self.assertEqual(output.dtype, p.DataType.DATA_TYPE_F32)
                        np.testing.assert_array_equal(np.frombuffer(output.inline, dtype="<u4"), expected)
                        api.release_execution_context(p.ExecutionContextRef(context_id=context.context_id))
                    api.release_runtime(p.RuntimeRef(runtime_id=runtime.runtime_id))

    def test_original_bf16_shards_mixed_with_f16_linear_and_source_snapshot(self):
        weights = np.array([[1.5, -2., 0.25], [-0.5, 4., 2.]], dtype="<f4")
        weight_bits = (weights.view("<u4") >> 16).astype("<u2")
        bias = np.array([0.5, -1.], dtype="<f2")
        graph = stateless_graph({
            "format": "volvox-graph/v1", "dimensions": {"S": {"min": 1, "max": 3}},
            "inputs": {"x": {"shape": ["S", 3], "dtype": "float32"}},
            "nodes": [{"id": "linear", "opType": "Linear",
                       "inputs": {"input": "x", "weight": "weight", "bias": "bias"},
                       "outputs": {"out": {"tensor": "y", "shape": ["S", 2], "dtype": "float32"}},
                       "params": {"weight_layout": "dout_din"}}], "outputs": ["y"],
        })
        with tempfile.TemporaryDirectory(prefix="volvox-bf16-source-") as directory:
            root = Path(directory)
            (root / "graph.json").write_text(json.dumps(graph))
            first = root / "weight.safetensors"
            second = root / "bias.safetensors"
            first.write_bytes(storage_shard([("weight", "BF16", [2, 3], weight_bits.tobytes())]))
            second.write_bytes(storage_shard([("bias", "F16", [2], bias.tobytes())]))
            for library in ("libvolvoxai-lite.so", "libvolvoxai.so"):
                with self.subTest(library=library), mock.patch.dict(os.environ, {
                        "VOLVOXAI_LIBRARY": str(Path(__file__).resolve().parents[2] / "native" / library)}):
                    with vx.InferenceSession(root, weights=[first, second],
                                             backend="cpu", cpu_threads=1) as session:
                        # Accepted model storage owns the source snapshot, not these paths.
                        old_first, old_second = first.read_bytes(), second.read_bytes()
                        first.unlink()
                        second.unlink()
                        try:
                            for rows in (1, 3, 2, 1):
                                x = (np.arange(rows * 3, dtype=np.float32).reshape(rows, 3) - 2.) / 4.
                                np.testing.assert_array_equal(session.run({"x": x})["y"],
                                                              x @ weights.T + bias.astype(np.float32))
                        finally:
                            first.write_bytes(old_first)
                            second.write_bytes(old_second)


class ModelSourcesTest(unittest.TestCase):
    def setUp(self):
        self.host = vx.open_library()
        self.addCleanup(self.host.close)
        self.client = vx.VxInferenceServiceClient(self.host)
        runtime = self.client.create_runtime(vx.pb.CreateRuntimeRequest())
        self.runtime_id = runtime.runtime_id
        self.addCleanup(self.client.release_runtime,
                        vx.pb.RuntimeRef(runtime_id=self.runtime_id))
        self.directory = tempfile.TemporaryDirectory(prefix="volvox-source-contract-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "graph.json"
        self.graph = json.dumps(stateless_graph({
            "format": "volvox-graph/v1", "dimensions": {},
            "inputs": {"x": {"shape": [1, 2], "dtype": "float32"}},
            "nodes": [{"id": "relu", "opType": "ReLU", "inputs": {"input": "x"},
                       "outputs": {"out": {"tensor": "y", "shape": [1, 2],
                                            "dtype": "float32"}}, "params": {}}],
            "outputs": ["y"],
        })).encode()
        self.path.write_bytes(self.graph)

    def options(self, **changes):
        values = dict(max_total_bytes=len(self.graph), max_shard_bytes=len(self.graph),
                      graph=vx.pb.ModelSourceIntegrity(byte_length=len(self.graph),
                                                      sha256=hashlib.sha256(self.graph).digest()))
        values.update(changes)
        return vx.pb.ModelSourceOptions(**values)

    def request(self, options=None, **changes):
        values = dict(runtime_id=self.runtime_id, graph_path=str(self.path),
                      source_options=options or self.options())
        values.update(changes)
        return vx.pb.LoadModelRequest(**values)

    def refusal(self, request, code):
        with self.assertRaises(vx.VolvoxAIError) as failure:
            self.client.load_model(request)
        self.assertEqual(failure.exception.code, code)
        self.assertEqual(failure.exception.response.model_id, 0)

    def test_accepted_source_is_immutable(self):
        model = self.client.load_model(self.request())
        self.addCleanup(self.client.release_model, vx.pb.ModelRef(model_id=model.model_id))
        self.path.write_bytes(b"corrupted after accepted load")
        compiled = self.client.compile_model(vx.pb.CompileModelRequest(model_id=model.model_id))
        self.client.release_compiled_model(vx.pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))

    def test_inline_graph_accepts_original_path_weights_and_manifest(self):
        weight = storage_shard([("offset", "BF16", [1, 2], struct.pack("<HH", 0x3f80, 0xc000))])
        path = self.path.parent / "original.safetensors"
        path.write_bytes(weight)
        graph = stateless_graph({
            "format": "volvox-graph/v1", "dimensions": {},
            "inputs": {"x": {"shape": [1, 2], "dtype": "float32"}},
            "nodes": [{"id": "add", "opType": "Add", "inputs": {"a": "x", "b": "offset"},
                       "outputs": {"out": {"tensor": "y", "shape": [1, 2], "dtype": "float32"}},
                       "params": {}}], "outputs": ["y"],
        })
        document = json.dumps(graph).encode()
        options = vx.pb.ModelSourceOptions(max_total_bytes=len(document) + len(weight),
            max_shard_bytes=max(len(document), len(weight)),
            graph=vx.pb.ModelSourceIntegrity(byte_length=len(document), sha256=hashlib.sha256(document).digest()),
            weight_shards=[vx.pb.ModelSourceIntegrity(byte_length=len(weight), sha256=hashlib.sha256(weight).digest())])
        for manifest in (None, options):
            with self.subTest(manifest=manifest is not None):
                loaded = self.client.load_model(vx.pb.LoadModelRequest(runtime_id=self.runtime_id,
                    graph_document=document, weight_paths=[str(path)], source_options=manifest))
                # Source paths can retire before Compile; graph bytes never need a caller file.
                path.unlink()
                try:
                    compiled = self.client.compile_model(vx.pb.CompileModelRequest(model_id=loaded.model_id))
                    context = self.client.create_execution_context(vx.pb.CreateExecutionContextRequest(
                        compiled_model_id=compiled.compiled_model_id, graph=vx.pb.GraphExecution()))
                    self.client.release_model(vx.pb.ModelRef(model_id=loaded.model_id))
                    self.client.release_compiled_model(vx.pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                    result = self.client.execute(vx.pb.ExecuteRequest(context_id=context.context_id,
                        inputs=vx.pb.InputBindings(values=[vx.pb.Tensor(name="x", shape=[1, 2],
                            dtype=vx.pb.DataType.DATA_TYPE_F32, inline=struct.pack("<ff", 3., 5.))]),
                        scope=vx.pb.ExecutionScope(entrypoint="evaluate", full=vx.pb.Empty())))
                    self.assertEqual(struct.unpack("<ff", result.outputs[0].inline), (4., 3.))
                    self.client.release_execution_context(vx.pb.ExecutionContextRef(context_id=context.context_id))
                finally:
                    path.write_bytes(weight)

    def test_inline_graph_integrity_refuses_before_weight_file_io(self):
        # A missing path makes ordering observable: inline integrity must fail first.
        options = self.options(max_total_bytes=len(self.graph) + 1,
            weight_shards=[vx.pb.ModelSourceIntegrity(byte_length=1, sha256=bytes(32))],
            graph=vx.pb.ModelSourceIntegrity(byte_length=len(self.graph), sha256=bytes(32)))
        self.refusal(vx.pb.LoadModelRequest(runtime_id=self.runtime_id,
            graph_document=self.graph, weight_paths=[str(self.path.parent / "missing.safetensors")],
            source_options=options), vx.pb.OperationCode.OPERATION_CODE_MODEL_SOURCE_INTEGRITY_MISMATCH)

    def test_multiple_graph_sources_are_rejected(self):
        for sources in ({"graph_path": str(self.path), "graph_document": self.graph},
                        {"package": vx.pb.ModelPackage(graph_document=self.graph), "graph_document": self.graph},
                        {"package": vx.pb.ModelPackage(graph_document=self.graph), "weight_paths": ["missing.safetensors"]}):
            with self.subTest(sources=list(sources)), self.assertRaises(vx.VolvoxAIError) as failure:
                self.client.load_model(vx.pb.LoadModelRequest(runtime_id=self.runtime_id, **sources))
            self.assertEqual(failure.exception.response.model_id, 0)

    def test_graph_length_and_digest_are_atomic(self):
        for length, digest in ((len(self.graph)-1, hashlib.sha256(self.graph).digest()),
                               (len(self.graph)+1, hashlib.sha256(self.graph).digest()),
                               (len(self.graph), bytes(32))):
            limit = len(self.graph)+1
            options = self.options(max_total_bytes=limit, max_shard_bytes=limit,
                graph=vx.pb.ModelSourceIntegrity(byte_length=length, sha256=digest))
            self.refusal(self.request(options),
                         vx.pb.OperationCode.OPERATION_CODE_MODEL_SOURCE_INTEGRITY_MISMATCH)
        model = self.client.load_model(self.request())
        self.client.release_model(vx.pb.ModelRef(model_id=model.model_id))

    def test_manifest_admission_before_file_io(self):
        invalid = vx.pb.OperationCode.OPERATION_CODE_INVALID_MODEL_SOURCE
        oversized = vx.pb.OperationCode.OPERATION_CODE_PACKAGE_TOO_LARGE
        for options, code in (
            (self.options(max_total_bytes=0), invalid),
            (self.options(max_shard_bytes=0), invalid),
            (self.options(graph=None), invalid),
            (self.options(graph=vx.pb.ModelSourceIntegrity(byte_length=len(self.graph), sha256=b"hex")), invalid),
            (self.options(max_total_bytes=len(self.graph)-1, max_shard_bytes=len(self.graph)-1), oversized),
            (self.options(weight_shards=[vx.pb.ModelSourceIntegrity(byte_length=1, sha256=bytes(32))]), invalid),
        ):
            self.refusal(self.request(options, graph_path=str(self.path.parent/"missing.graph.json")), code)
        self.refusal(self.request(graph_path="", package=vx.pb.ModelPackage(graph_document=self.graph)), invalid)

    def test_ordered_shard_integrity(self):
        shards = [b"first source", b"second source"]
        paths = [self.path.parent / f"weights-{i}.safetensors" for i in range(2)]
        for path, data in zip(paths, shards):
            path.write_bytes(data)
        for broken in (0, 1):
            integrity = [vx.pb.ModelSourceIntegrity(byte_length=len(data),
                sha256=bytes(32) if i==broken else hashlib.sha256(data).digest())
                for i, data in enumerate(shards)]
            options = self.options(max_total_bytes=len(self.graph)+sum(map(len, shards)),
                                   max_shard_bytes=len(self.graph), weight_shards=integrity)
            self.refusal(self.request(options, weight_paths=list(map(str, paths))),
                         vx.pb.OperationCode.OPERATION_CODE_MODEL_SOURCE_INTEGRITY_MISMATCH)


if __name__ == "__main__":
    unittest.main()
