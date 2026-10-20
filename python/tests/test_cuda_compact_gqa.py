"""Physical CUDA F32 regressions through generated public dispatch.

Run with VOLVOXAI_TEST_NATIVE_GPU_BACKEND=cuda and an explicit
VOLVOXAI_LIBRARY. NumPy and the repository Python SDK are the only Python
dependencies. Missing CUDA is a failure after this opt-in. No timing gate is
measured. VOLVOXAI_CUDA_RECEIPT_DIR optionally preserves exclusive JSON receipts.
The full profile also runs the separate training refusal method; inference
callers should select the five numerical/admission methods explicitly.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import volvoxai as vx
from execution_fixture import stateless_graph

PB = vx.pb
ATOL = RTOL = 1e-5


def node(name, op, inputs, shape, params=None, dtype="float32"):
    return {"id": name, "opType": op, "inputs": inputs, "params": params or {},
        "outputs": {"out": {"tensor": name, "shape": shape, "dtype": dtype}}}


def graph(inputs, nodes, outputs, dimensions=None):
    return stateless_graph({"format": "volvox-graph/v1", "dimensions": dimensions or {},
        "inputs": inputs, "nodes": nodes, "outputs": outputs})


def tensor(name, value):
    value = np.ascontiguousarray(value)
    return PB.Tensor(name=name, shape=list(value.shape), inline=value.tobytes(),
        dtype=PB.DataType.DATA_TYPE_I32 if value.dtype == np.int32 else PB.DataType.DATA_TYPE_F32)


def array(packet, name):
    value = next(t for t in packet.outputs if t.name == name)
    assert value.dtype == PB.DataType.DATA_TYPE_F32 and not value.HasField("buffer")
    return np.frombuffer(value.inline, dtype=np.float32).copy().reshape(value.shape)


def attention_document(batch, width, kv_heads, mask_shape=None, causal=False, rank=3, explicit=True):
    shape = lambda sequence, channels: [batch, sequence, channels] if rank == 3 else [sequence, channels]
    inputs = {"q": {"shape": shape("Q", 4 * width), "dtype": "float32"},
        "k": {"shape": shape("K", kv_heads * width), "dtype": "float32"},
        "v": {"shape": shape("K", kv_heads * width), "dtype": "float32"}}
    ports = {name: name for name in inputs}
    if mask_shape is not None:
        inputs["keep"] = {"shape": mask_shape, "dtype": "int32"}
        ports["mask"] = "keep"
    params = {"heads": 4, "causal": causal}
    if explicit:
        params["kv_heads"] = kv_heads
    return graph(inputs, [node("y", "CrossSDPA", ports, shape("Q", 4 * width), params)], ["y"],
        {"Q": {"min": 1, "max": 5}, "K": {"min": 1, "max": 5}})


def attention_values(batch, queries, keys, width, kv_heads):
    def signal(count, phase):
        axis = np.arange(count, dtype=np.float64)
        return (.5 * np.sin(axis * .17 + phase) + .2 * np.cos(axis * .31)).astype(np.float32)
    return {"q": signal(batch * queries * 4 * width, .3).reshape(batch, queries, -1),
        "k": signal(batch * keys * kv_heads * width, .7).reshape(batch, keys, -1),
        "v": signal(batch * keys * kv_heads * width, 1.3).reshape(batch, keys, -1)}


def attention_oracle(values, width, kv_heads, causal):
    """Independent direct compact indexing agrees with vectorized repeat-KV."""
    batch, queries, _ = values["q"].shape
    keys = values["k"].shape[1]
    q = values["q"].astype(np.float64).reshape(batch, queries, 4, width)
    k = values["k"].astype(np.float64).reshape(batch, keys, kv_heads, width)
    v = values["v"].astype(np.float64).reshape(batch, keys, kv_heads, width)
    mask = values.get("keep")
    if mask is None:
        visible = np.ones((batch, queries, keys), dtype=bool)
    elif mask.ndim == 1:
        visible = np.broadcast_to(mask[None, None] != 0, (batch, queries, keys)).copy()
    elif mask.ndim == 2 and mask.shape[0] == batch:
        visible = np.broadcast_to(mask[:, None] != 0, (batch, queries, keys)).copy()
    elif mask.ndim == 2:
        visible = np.broadcast_to(mask[None] != 0, (batch, queries, keys)).copy()
    else:
        visible = mask != 0
    if causal:
        visible &= np.arange(keys)[None, None] <= np.arange(queries)[None, :, None]
    direct = np.zeros_like(q)
    for b in range(batch):
        for row in range(queries):
            indices = np.flatnonzero(visible[b, row])
            for head in range(4):
                if indices.size:
                    kv = head // (4 // kv_heads)
                    scores = k[b, indices, kv] @ q[b, row, head] / np.sqrt(width)
                    probability = np.exp(scores - scores.max())
                    direct[b, row, head] = probability @ v[b, indices, kv] / probability.sum()
    repeated_k, repeated_v = (np.repeat(x, 4 // kv_heads, axis=2) for x in (k, v))
    scores = np.einsum("bqhd,bkhd->bhqk", q, repeated_k) / np.sqrt(width)
    scores = np.where(visible[:, None], scores, -np.inf)
    maximum = np.max(scores, axis=-1, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0)
    probability = np.exp(scores - maximum)
    denominator = probability.sum(axis=-1, keepdims=True)
    probability = np.divide(probability, denominator, out=np.zeros_like(probability), where=denominator != 0)
    repeated = np.einsum("bhqk,bkhd->bqhd", probability, repeated_v)
    np.testing.assert_allclose(direct, repeated, atol=2e-14, rtol=2e-14)
    return direct.astype(np.float32).reshape(values["q"].shape)


def rope_document(batch, sequence, width, mode, rotary, theta, interleaved, rank=3, offset=13):
    shape = [batch, sequence, 4 * width] if rank == 3 else [sequence, 4 * width]
    inputs = {"x": {"shape": shape, "dtype": "float32"}}
    ports = {"input": "x"}
    if mode:
        inputs["positions"] = {"shape": [sequence] if mode == 1 else [batch, sequence], "dtype": "int32"}
        ports["position_ids"] = "positions"
    return graph(inputs, [node("y", "RoPE", ports, shape, {"heads": 4,
        "rotary_dim": rotary, "theta": theta, "interleaved": interleaved,
        "position_offset": offset})], ["y"])


def rope_oracle(x, positions, rotary, theta, interleaved, canonical_angle=False):
    """Pair rotations use NumPy float64 trig, with two explicit angle contracts.

    Ordinary small-position cases use the mathematical rotary frequencies.
    Large-I32 regression honors the CPU ABI's F32 exponent/powf/division angle;
    the platform powf prepares configuration metadata, never an engine output.
    """
    width = x.shape[-1] // 4
    batch, sequence = (x.shape[0], x.shape[1]) if x.ndim == 3 else (1, x.shape[0])
    source = x.astype(np.float64).reshape(batch, sequence, 4, width)
    result = source.copy()
    position = np.broadcast_to(positions, (batch, sequence))
    if canonical_angle:
        libm = ctypes.CDLL(ctypes.util.find_library("m") or None)
        libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
        libm.powf.restype = ctypes.c_float
        powers = np.array([libm.powf(theta, float(np.float32(np.float32(2 * pair) / np.float32(rotary))))
                           for pair in range(rotary // 2)], dtype=np.float32)
        angles = (position.astype(np.float32)[..., None] / powers).astype(np.float32).astype(np.float64)[:, :, None]
    else:
        angles = position[:, :, None, None] * theta ** (-np.arange(rotary // 2) * 2.0 / rotary)
    left = np.arange(rotary // 2) * (2 if interleaved else 1)
    right = left + (1 if interleaved else rotary // 2)
    a, b = source[..., left], source[..., right]
    result[..., left] = a * np.cos(angles) - b * np.sin(angles)
    result[..., right] = a * np.sin(angles) + b * np.cos(angles)
    return result.astype(np.float32).reshape(x.shape)


def weight_shard(weights):
    header, chunks, offset = {}, [], 0
    for name, data in weights.items():
        data = np.ascontiguousarray(data, dtype=np.float32)
        payload = data.tobytes()
        header[name] = {"dtype": "F32", "shape": list(data.shape), "data_offsets": [offset, offset + len(payload)]}
        chunks.append(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return len(encoded).to_bytes(8, "little") + encoded + b"".join(chunks)


@unittest.skipUnless(os.environ.get("VOLVOXAI_TEST_NATIVE_GPU_BACKEND") == "cuda",
                     "set VOLVOXAI_TEST_NATIVE_GPU_BACKEND=cuda on physical CUDA hardware")
class CudaCompactGQATest(unittest.TestCase):
    def setUp(self):
        self.host = vx.open_library()
        self.addCleanup(self.host.close)
        self.inference = vx.VxInferenceServiceClient(self.host)
        self.profile = vx.VxPlatformServiceClient(self.host).get_platform_info(PB.Empty()).build_profile
        self.records = []
        self.executions = 0
        self.refusals = 0

    def tearDown(self):
        destination = os.environ.get("VOLVOXAI_CUDA_RECEIPT_DIR")
        if destination:
            directory = Path(destination)
            directory.mkdir(parents=True, exist_ok=True)
            library = Path(os.environ["VOLVOXAI_LIBRARY"]).resolve(strict=True)
            receipt = {"format": "volvox-cuda-compact-public-fixture/v1", "test": self.id(),
                "profile": int(self.profile), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "artifact": {"path": str(library), "sha256": hashlib.sha256(library.read_bytes()).hexdigest()},
                "tolerances": {"atol": ATOL, "rtol": RTOL}, "executions": self.executions,
                "refusals": self.refusals, "route_records": self.records,
                "timing": False, "cache_qualified": False,
                "status": "observations only; unittest log determines pass/fail"}
            with (directory / (self._testMethodName + ".json")).open("x") as stream:
                json.dump(receipt, stream, indent=2)
                stream.write("\n")

    def route(self, report, label):
        self.assertEqual(report.status, PB.NativeStatus.NATIVE_STATUS_OK)
        self.assertEqual(report.backend, "cuda")
        route = report.route
        self.assertIsNotNone(route)
        self.assertTrue(route.attested)
        self.assertTrue(route.builtin)
        self.assertIn("cuda", route.provider)
        self.assertGreater(route.active_nodes, 0)
        self.assertEqual(route.selected_nodes, route.active_nodes)
        self.assertEqual((route.fallback_nodes, route.missing_nodes), (0, 0))
        self.assertFalse(report.fallback and report.fallback.operator_fallback_used)
        self.records.append({"label": label, "backend": report.backend, "device": report.device,
            "provider": route.provider, "attested": route.attested, "active_nodes": route.active_nodes,
            "selected_nodes": route.selected_nodes, "fallback_nodes": route.fallback_nodes,
            "missing_nodes": route.missing_nodes, "digest": int(route.digest)})

    @contextmanager
    def model(self, document, weights=None):
        with ExitStack() as cleanup:
            runtime = self.inference.create_runtime(PB.CreateRuntimeRequest(cpu_threads=1))
            cleanup.callback(self.inference.release_runtime, PB.RuntimeRef(runtime_id=runtime.runtime_id))
            package = PB.ModelPackage(graph_document=json.dumps(document).encode(),
                weight_shards=[weight_shard(weights)] if weights else [])
            model = self.inference.load_model(PB.LoadModelRequest(runtime_id=runtime.runtime_id, package=package))
            cleanup.callback(self.inference.release_model, PB.ModelRef(model_id=model.model_id))
            yield model.model_id, cleanup

    @contextmanager
    def context(self, document, weights=None):
        with self.model(document, weights) as (model_id, cleanup):
            compiled = self.inference.compile_model(PB.CompileModelRequest(model_id=model_id,
                policy=PB.BackendPolicy(mode=PB.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                    backends=["cuda"], operator_fallback=PB.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
            self.route(compiled.report, "compile")
            cleanup.callback(self.inference.release_compiled_model,
                PB.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
            context = self.inference.create_execution_context(PB.CreateExecutionContextRequest(
                compiled_model_id=compiled.compiled_model_id, graph=PB.GraphExecution()))
            cleanup.callback(self.inference.release_execution_context, PB.ExecutionContextRef(context_id=context.context_id))
            yield context.context_id

    def execute(self, context_id, values):
        packet = self.inference.execute(PB.ExecuteRequest(context_id=context_id,
            scope=PB.ExecutionScope(entrypoint="evaluate", full=PB.Empty()),
            inputs=PB.InputBindings(values=[tensor(name, value) for name, value in values.items()])))
        self.route(packet.report, "execute")
        self.executions += 1
        return packet

    def test_attention_compact_masks_unequal_lengths_and_rank2(self):
        for width in (8, 64, 128):
            for kv_heads in (4, 2, 1):
                for batch in (1, 3):
                    for mode, mask_shape in (("none", None), ("K", ["K"]), ("BK", [batch, "K"]),
                                             ("QK", ["Q", "K"]), ("BQK", [batch, "Q", "K"])):
                        for causal in (False, True):
                            with self.subTest(D=width, Hkv=kv_heads, B=batch, mask=mode, causal=causal):
                                with self.context(attention_document(batch, width, kv_heads, mask_shape, causal)) as context:
                                    for queries, keys in ((2, 5), (5, 2)):
                                        values = attention_values(batch, queries, keys, width, kv_heads)
                                        if mask_shape is not None:
                                            shape = [queries if d == "Q" else keys if d == "K" else d for d in mask_shape]
                                            keep = np.ones(shape, dtype=np.int32)
                                            keep[..., -1] = 0
                                            if mode in ("QK", "BQK"):
                                                keep[..., 0, :] = 0
                                            values["keep"] = keep
                                        actual = array(self.execute(context, values), "y")
                                        np.testing.assert_allclose(actual, attention_oracle(values, width, kv_heads, causal), atol=ATOL, rtol=RTOL)
                                        if mode in ("QK", "BQK"):
                                            np.testing.assert_array_equal(actual[:, 0], np.zeros_like(actual[:, 0]))
                                        if mask_shape is not None:
                                            values["keep"].fill(0)
                                            np.testing.assert_array_equal(array(self.execute(context, values), "y"), np.zeros_like(actual))
                values = attention_values(1, 2, 5, width, kv_heads)
                with self.context(attention_document(1, width, kv_heads, rank=2)) as context:
                    actual = array(self.execute(context, {name: data[0] for name, data in values.items()}), "y")
                    np.testing.assert_allclose(actual, attention_oracle(values, width, kv_heads, False)[0], atol=ATOL, rtol=RTOL)
        values = attention_values(1, 2, 5, 64, 4)
        outputs = []
        for explicit in (False, True):
            with self.context(attention_document(1, 64, 4, explicit=explicit)) as context:
                outputs.append(array(self.execute(context, values), "y"))
        # The legacy MHA kernel is retained; both domains must stay numerically equivalent.
        np.testing.assert_allclose(outputs[0], outputs[1], atol=ATOL, rtol=RTOL)

    def test_head_rope_modes_partial_odd_heads_and_nonzero_positions(self):
        for width in (64, 128, 65, 127):
            rotary = width - 2 if width % 2 == 0 else width - 1
            for rank, batch in ((2, 1), (3, 1), (3, 3)):
                for mode in (0, 1, 2):
                    if rank == 2 and mode == 2:
                        continue
                    for interleaved in (False, True):
                        for theta in (100000., 1000000.):
                            for sequence in (1, 5):
                                with self.subTest(D=width, B=batch, rank=rank, mode=mode, pairing=interleaved, theta=theta, S=sequence):
                                    doc = rope_document(batch, sequence, width, mode, rotary, theta, interleaved, rank)
                                    x = attention_values(batch, sequence, 1, width, 1)["q"]
                                    positions = (np.array([13, 31, 97], dtype=np.int32)[:batch, None]
                                                 + np.arange(sequence, dtype=np.int32)[None] * 3)
                                    if mode == 0:
                                        positions = np.arange(sequence, dtype=np.int32) + 13
                                    elif mode == 1:
                                        positions = np.array([13, 31, 97, 16, 7], dtype=np.int32)[:sequence]
                                    if rank == 2:
                                        x = x[0]
                                    values = {"x": x}
                                    if mode:
                                        values["positions"] = positions
                                    with self.context(doc) as context:
                                        actual = array(self.execute(context, values), "y")
                                        np.testing.assert_allclose(actual, rope_oracle(x, positions, rotary, theta, interleaved), atol=ATOL, rtol=RTOL)
                                        np.testing.assert_array_equal(actual.reshape(-1, 4, width)[..., rotary:], x.reshape(-1, 4, width)[..., rotary:])

    def test_rank4_last_d_rmsnorm(self):
        for batch in (1, 3):
            for sequence in (1, 5):
                for heads in (1, 4):
                    for width in (64, 128):
                        shape = [batch, sequence, heads, width]
                        doc = graph({"x": {"shape": shape, "dtype": "float32"},
                            "weight": {"shape": [width], "dtype": "float32"}},
                            [node("y", "RMSNorm", {"input": "x", "weight": "weight"}, shape, {"eps": 1e-6})], ["y"])
                        x = np.sin(np.arange(np.prod(shape), dtype=np.float64) * .17).astype(np.float32).reshape(shape)
                        weight = np.linspace(.7, 1.3, width, dtype=np.float32)
                        expected = x.astype(np.float64) / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + 1e-6) * weight
                        with self.subTest(B=batch, S=sequence, H=heads, D=width), self.context(doc) as context:
                            np.testing.assert_allclose(array(self.execute(context, {"x": x, "weight": weight}), "y"), expected, atol=ATOL, rtol=RTOL)

    def test_multiple_rope_metadata_tables_repeat_and_large_i32_positions(self):
        shape = [1, 4, 512]
        positions = np.array([[13, 31, 97, np.iinfo(np.int32).max]], dtype=np.int32)
        for ordering in (((1000000., 126), (100000., 64)), ((100000., 64), (1000000., 126))):
            nodes, source = [], "x"
            for index, (theta, rotary) in enumerate(ordering):
                name = "rotated" + str(index)
                nodes.append(node(name, "RoPE", {"input": source, "position_ids": "positions"}, shape,
                    {"heads": 4, "rotary_dim": rotary, "theta": theta, "interleaved": bool(index)}))
                source = name
            doc = graph({"x": {"shape": shape, "dtype": "float32"},
                "positions": {"shape": [1, 4], "dtype": "int32"}}, nodes, ["rotated0", "rotated1"])
            with self.subTest(ordering=ordering), self.context(doc) as context:
                retained = None
                for repeat in range(4):
                    x = np.sin(np.arange(np.prod(shape), dtype=np.float64) * .17 + repeat).astype(np.float32).reshape(shape)
                    packet = self.execute(context, {"x": x, "positions": positions})
                    expected = x
                    for index, (theta, rotary) in enumerate(ordering):
                        expected = rope_oracle(expected, positions, rotary, theta, bool(index), canonical_angle=True)
                        np.testing.assert_allclose(array(packet, "rotated" + str(index)), expected, atol=ATOL, rtol=RTOL)
                    if retained is None:
                        retained = packet
                        saved = array(packet, "rotated0").copy()
                    else:
                        np.testing.assert_array_equal(array(retained, "rotated0"), saved)

    def test_typed_subset_and_position_admission_before_new_outputs(self):
        rejected = [rope_document(1, 2, 64, 1, 64, .5, False),
                    attention_document(1, 129, 2)]
        authored = rope_document(1, 2, 64, 1, 64, 1000000., False)
        authored["nodes"].insert(0, node("gpu_positions", "Clip", {"input": "positions"}, [2],
                                        {"min": 0, "max": 127}, "int32"))
        authored["nodes"][1]["inputs"]["position_ids"] = "gpu_positions"
        rejected.append(authored)
        for doc in rejected:
            with self.model(doc) as (model_id, _):
                with self.assertRaises(vx.VolvoxAIError) as failure:
                    self.inference.compile_model(PB.CompileModelRequest(model_id=model_id,
                        policy=PB.BackendPolicy(mode=PB.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                            backends=["cuda"], operator_fallback=PB.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
                self.assertEqual(failure.exception.status, PB.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED)
                self.assertEqual(failure.exception.code, PB.OperationCode.OPERATION_CODE_BOUNDED_DOMAIN_UNSUPPORTED)
                self.assertFalse(failure.exception.response.compiled_model_id)
                self.refusals += 1
        doc = rope_document(3, 5, 64, 2, 62, 1000000., False)
        x = attention_values(3, 5, 1, 64, 1)["q"]
        positions = np.arange(15, dtype=np.int32).reshape(3, 5) + 7
        with self.context(doc) as context:
            initial = self.execute(context, {"x": x, "positions": positions})
            saved = array(initial, "y").copy()
            bad = positions.copy()
            bad[-1, -1] = -1
            with self.assertRaises(vx.VolvoxAIError) as failure:
                self.execute(context, {"x": x + 99, "positions": bad})
            self.assertEqual(failure.exception.status, PB.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT)
            self.assertEqual(failure.exception.code, PB.OperationCode.OPERATION_CODE_INVALID_INPUT_VALUES)
            self.assertFalse(failure.exception.response.outputs)
            self.refusals += 1
            np.testing.assert_array_equal(array(initial, "y"), saved)
            np.testing.assert_allclose(array(self.execute(context, {"x": x, "positions": positions}), "y"), saved, atol=ATOL, rtol=RTOL)

    def test_training_still_refuses_explicit_kv_heads_and_rope_before_state_mutation(self):
        if self.profile != PB.BuildProfile.BUILD_PROFILE_FULL:
            self.skipTest("select only the five CUDA numerical/admission methods for inference")
        training = vx.VxTrainingServiceClient(self.host)
        for kind in ("compact", "explicit_mha", "rope"):
            kv = 4 if kind == "explicit_mha" else 2
            doc = attention_document(1, 8, kv)
            doc["dimensions"] = {}
            for name in ("q", "k", "v"):
                doc["inputs"][name]["shape"][1] = 1 if name == "q" else 2
            doc["nodes"][0]["outputs"]["out"]["shape"][1] = 1
            source = "y"
            if kind == "rope":
                del doc["nodes"][0]["params"]["kv_heads"]
                for name in ("k", "v"):
                    doc["inputs"][name]["shape"][-1] = 32
                kv = 4
                doc["nodes"].append(node("rotated", "RoPE", {"input": "y"}, [1, 1, 32], {"heads": 4, "rotary_dim": 8}))
                source = "rotated"
            doc["nodes"].append(node("logits", "Add", {"a": source, "b": "parameter"}, [1, 1, 32]))
            doc["outputs"] = ["logits"]
            doc = stateless_graph({k: v for k, v in doc.items() if k != "execution"})
            with self.subTest(kind=kind), self.model(doc, {"parameter": np.zeros(32, dtype=np.float32)}) as (model_id, cleanup):
                trainer = training.create_trainer(PB.CreateTrainerRequest(model_id=model_id, backend="cpu"))
                tref = PB.TrainerRef(trainer_id=trainer.trainer_id)
                cleanup.callback(training.release_trainer, tref)
                before = training.get_trainer_state(tref)
                request = PB.TrainStepRequest(trainer_id=trainer.trainer_id,
                    inputs=[tensor(name, value) for name, value in attention_values(1, 1, 2, 8, kv).items()],
                    losses=[PB.CrossEntropyLoss(name="ce", logits_name="logits",
                        targets=tensor("", np.array([0], dtype=np.int32)), ignore_index=-100, row_index=-1)],
                    trainable_names=["parameter"], optimizer=PB.TrainerOptimizerOptions(
                        kind=PB.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD, learning_rate=.01),
                    accumulation_steps=1, flush_accumulation=True, reset_accumulation=True)
                with self.assertRaises(vx.VolvoxAIError) as failure:
                    training.train_step(request)
                self.assertEqual(failure.exception.status, PB.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED)
                self.assertEqual(failure.exception.code, PB.OperationCode.OPERATION_CODE_TRAINING_GRAPH_UNSUPPORTED)
                after = training.get_trainer_state(tref)
                for name in ("optimizer_step", "rng_seed", "accumulated_microbatches", "accumulation_steps",
                             "activation_signature", "optimizer_bytes", "gradient_bytes", "has_private_updates"):
                    self.assertEqual(getattr(before, name), getattr(after, name), name)
                for name in ("activation_grow_count", "activation_high_water_bytes", "plan_cache_entries"):
                    self.assertEqual(getattr(before.shape, name), getattr(after.shape, name), name)
                self.refusals += 1


if __name__ == "__main__":
    unittest.main()
