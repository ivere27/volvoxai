"""Independent compact attention, flat-head RoPE and training admission checks."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import volvoxai as vx
from execution_fixture import stateless_graph


def tensor(name, value):
    return vx.pb.Tensor(name=name, shape=list(value.shape),
        dtype=(vx.pb.DataType.DATA_TYPE_I32 if value.dtype == np.int32
               else vx.pb.DataType.DATA_TYPE_F32), inline=value.tobytes())


def attention_graph(batch, width, kv_heads, mask_shape=None, causal=False,
                    explicit=True, rank=3):
    def shape(sequence, features):
        return [batch, sequence, features] if rank == 3 else [sequence, features]
    qshape, kvshape = shape("Q", 4 * width), shape("K", kv_heads * width)
    inputs = {"q": {"shape": qshape, "dtype": "float32"},
              "k": {"shape": kvshape, "dtype": "float32"},
              "v": {"shape": kvshape, "dtype": "float32"}}
    ports = {"q": "q", "k": "k", "v": "v"}
    if mask_shape is not None:
        inputs["keep"] = {"shape": mask_shape, "dtype": "int32"}
        ports["mask"] = "keep"
    params = {"heads": 4, "causal": causal}
    if explicit:
        params["kv_heads"] = kv_heads
    return stateless_graph({"format": "volvox-graph/v1",
        "dimensions": {"Q": {"min": 1, "max": 5}, "K": {"min": 1, "max": 5}},
        "inputs": inputs, "nodes": [{"id": "attention", "opType": "CrossSDPA",
            "inputs": ports, "outputs": {"out": {"tensor": "y", "shape": qshape,
                "dtype": "float32"}}, "params": params}], "outputs": ["y"]})


def attention_values(batch, queries, keys, width, kv_heads):
    def values(count, phase):
        axis = np.arange(count, dtype=np.float64)
        return (.5 * np.sin(axis * .17 + phase) + .2 * np.cos(axis * .31)).astype(np.float32)
    return {"q": values(batch * queries * 4 * width, .3).reshape(batch, queries, -1),
            "k": values(batch * keys * kv_heads * width, .7).reshape(batch, keys, -1),
            "v": values(batch * keys * kv_heads * width, 1.3).reshape(batch, keys, -1)}


def visible_mask(mask, batch, queries, keys, causal):
    if mask is None:
        allowed = np.ones((batch, queries, keys), dtype=bool)
    elif mask.ndim == 1:
        allowed = np.broadcast_to(mask[None, None, :] != 0, (batch, queries, keys)).copy()
    elif mask.ndim == 2 and mask.shape[0] == batch:
        allowed = np.broadcast_to(mask[:, None, :] != 0, (batch, queries, keys)).copy()
    elif mask.ndim == 2:
        allowed = np.broadcast_to(mask[None, :, :] != 0, (batch, queries, keys)).copy()
    else:
        allowed = mask != 0
    if causal:
        allowed &= np.arange(keys)[None, None, :] <= np.arange(queries)[None, :, None]
    return allowed


def attention_reference(values, width, kv_heads, causal=False):
    """Direct compact indexing and a separately vectorized repeat-KV oracle."""
    batch, queries, _ = values["q"].shape
    keys = values["k"].shape[1]
    q = values["q"].astype(np.float64).reshape(batch, queries, 4, width)
    k = values["k"].astype(np.float64).reshape(batch, keys, kv_heads, width)
    v = values["v"].astype(np.float64).reshape(batch, keys, kv_heads, width)
    allowed = visible_mask(values.get("keep"), batch, queries, keys, causal)
    direct = np.zeros_like(q)
    for b in range(batch):
        for row in range(queries):
            indices = np.flatnonzero(allowed[b, row])
            if not indices.size:
                continue
            for head in range(4):
                kv = head // (4 // kv_heads)
                scores = k[b, indices, kv] @ q[b, row, head] / np.sqrt(width)
                probabilities = np.exp(scores - scores.max())
                probabilities /= probabilities.sum()
                direct[b, row, head] = probabilities @ v[b, indices, kv]
    repeated_k, repeated_v = (np.repeat(x, 4 // kv_heads, axis=2) for x in (k, v))
    scores = np.einsum("bqhd,bkhd->bhqk", q, repeated_k) / np.sqrt(width)
    visible = allowed[:, None, :, :]
    scores = np.where(visible, scores, -np.inf)
    maximum = np.max(scores, axis=-1, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0)
    probabilities = np.exp(scores - maximum)
    denominator = probabilities.sum(axis=-1, keepdims=True)
    probabilities = np.divide(probabilities, denominator,
        out=np.zeros_like(probabilities), where=denominator != 0)
    repeated = np.einsum("bhqk,bkhd->bqhd", probabilities, repeated_v)
    np.testing.assert_allclose(direct, repeated, atol=2e-14, rtol=2e-14)
    return direct.astype(np.float32).reshape(values["q"].shape)


def rope_reference(x, positions, heads, rotary, theta, interleaved):
    width = x.shape[-1] // heads
    batch, sequence = (x.shape[0], x.shape[1]) if x.ndim == 3 else (1, x.shape[0])
    source = x.astype(np.float64).reshape(batch, sequence, heads, width)
    result = source.copy()
    position = np.broadcast_to(positions, (batch, sequence))
    angles = position[:, :, None, None] * theta ** (-np.arange(rotary // 2) * 2.0 / rotary)
    left_index = np.arange(rotary // 2) * (2 if interleaved else 1)
    right_index = left_index + (1 if interleaved else rotary // 2)
    left, right = source[..., left_index], source[..., right_index]
    result[..., left_index] = left * np.cos(angles) - right * np.sin(angles)
    result[..., right_index] = right * np.cos(angles) + left * np.sin(angles)
    return result.astype(np.float32).reshape(x.shape)


class CompactGQATest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="volvox-compact-gqa-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def package(self, graph):
        path = self.directory / str(len(list(self.directory.iterdir())))
        path.mkdir()
        (path / "graph.json").write_text(json.dumps(graph))
        return path

    def test_compact_mha_gqa_mqa_d8_d64_d128_masks_and_unequal_lengths(self):
        for width in (8, 64, 128):
            for kv_heads in (4, 2, 1):
                for batch in (1, 3):
                    # Q!=B disambiguates the supported 2-D QK and BK masks.
                    for mode, maskshape in (("none", None), ("K", ["K"]),
                            ("BK", [batch, "K"]), ("QK", ["Q", "K"]),
                            ("BQK", [batch, "Q", "K"])):
                        for causal in (False, True):
                            graph = attention_graph(batch, width, kv_heads, maskshape, causal)
                            with self.subTest(D=width, Hkv=kv_heads, B=batch, mask=mode, causal=causal):
                                with vx.InferenceSession(self.package(graph), backend="cpu", cpu_threads=1) as session:
                                    for queries, keys in ((2, 5), (5, 2)):
                                        values = attention_values(batch, queries, keys, width, kv_heads)
                                        if maskshape is not None:
                                            shape = [queries if x == "Q" else keys if x == "K" else x for x in maskshape]
                                            keep = np.ones(shape, dtype=np.int32)
                                            keep[..., -1] = 0
                                            if mode in ("QK", "BQK"):
                                                keep[..., 0, :] = 0
                                            values["keep"] = keep
                                        expected = attention_reference(values, width, kv_heads, causal)
                                        actual = session.run(values)["y"]
                                        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
                                        if mode in ("QK", "BQK"):
                                            np.testing.assert_array_equal(actual[:, 0], np.zeros_like(actual[:, 0]))
                                        if maskshape is not None:
                                            values["keep"].fill(0)
                                            np.testing.assert_array_equal(session.run(values)["y"], np.zeros_like(actual))

    def test_default_mha_preserves_explicit_mha_bytes_and_rank2(self):
        values = attention_values(1, 2, 5, 64, 4)
        outputs = []
        for explicit in (False, True):
            with vx.InferenceSession(self.package(attention_graph(1, 64, 4, explicit=explicit)), backend="cpu") as session:
                outputs.append(session.run(values)["y"])
        self.assertEqual(outputs[0].tobytes(), outputs[1].tobytes())
        compact = attention_values(1, 2, 5, 128, 1)
        with vx.InferenceSession(self.package(attention_graph(1, 128, 1, rank=2)), backend="cpu") as session:
            actual = session.run({name: x[0] for name, x in compact.items()})["y"]
        np.testing.assert_allclose(actual, attention_reference(compact, 128, 1)[0], atol=2e-5, rtol=2e-5)

    def test_flat_head_rope_partial_tail_positions_and_pair_conventions(self):
        for width in (8, 64, 128):
            for rank, batch in ((2, 1), (3, 1), (3, 3)):
                for interleaved in (False, True):
                    shape = [batch, "S", 4 * width] if rank == 3 else ["S", 4 * width]
                    pshape = [batch, "S"] if rank == 3 else ["S"]
                    rotary = width - 2
                    graph = stateless_graph({"format": "volvox-graph/v1", "dimensions": {"S": {"min": 1, "max": 5}},
                        "inputs": {"x": {"shape": shape, "dtype": "float32"}, "positions": {"shape": pshape, "dtype": "int32"}},
                        "nodes": [{"id": "rope", "opType": "RoPE", "inputs": {"input": "x", "position_ids": "positions"},
                            "outputs": {"out": {"tensor": "y", "shape": shape, "dtype": "float32"}},
                            "params": {"heads": 4, "rotary_dim": rotary, "theta": 1000000, "interleaved": interleaved}}], "outputs": ["y"]})
                    with self.subTest(D=width, rank=rank, B=batch, interleaved=interleaved):
                        with vx.InferenceSession(self.package(graph), backend="cpu", cpu_threads=1) as session:
                            for sequence in (1, 5, 2):
                                x = attention_values(batch, sequence, 1, width, 1)["q"]
                                positions = (7 + np.arange(batch * sequence, dtype=np.int32) * 3).reshape(batch, sequence)
                                if rank == 2:
                                    x, positions = x[0], positions[0]
                                actual = session.run({"x": x, "positions": positions})["y"]
                                expected = rope_reference(x, positions, 4, rotary, 1000000, interleaved)
                                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
                                np.testing.assert_array_equal(actual.reshape(-1, 4, width)[..., rotary:],
                                                              x.reshape(-1, 4, width)[..., rotary:])
                            # Validate a late slot/row before publishing any new result.
                            positions.flat[-1] = -1
                            with self.assertRaises(vx.VolvoxAIError) as failure:
                                session.run({"x": x, "positions": positions})
                            self.assertEqual(failure.exception.status, vx.pb.NativeStatus.NATIVE_STATUS_EXECUTION_FAILED)

    def training_package(self, kind):
        kv_heads = 2 if kind == "gqa" else 4
        graph = attention_graph(1, 8, kv_heads, explicit=kind in ("gqa", "explicit_mha"))
        graph["dimensions"] = {}
        for name in ("q", "k", "v"):
            graph["inputs"][name]["shape"][1] = 1 if name == "q" else 2
        graph["nodes"][0]["outputs"]["out"]["shape"][1] = 1
        source = "y"
        if kind == "rope":
            graph["nodes"].append({"id": "rope", "opType": "RoPE", "inputs": {"input": source},
                "outputs": {"out": {"tensor": "rotated", "shape": [1, 1, 32], "dtype": "float32"}},
                "params": {"heads": 4, "rotary_dim": 8}})
            source = "rotated"
        if kind == "norm":
            graph["nodes"].extend([
                {"id": "heads", "opType": "Reshape", "inputs": {"input": source},
                 "outputs": {"out": {"tensor": "head_values", "shape": [1, 1, 4, 8], "dtype": "float32"}},
                 "params": {"shape": [1, 1, 4, 8]}},
                {"id": "norm", "opType": "RMSNorm", "inputs": {"input": "head_values", "weight": "norm_weight"},
                 "outputs": {"out": {"tensor": "head_normalized", "shape": [1, 1, 4, 8], "dtype": "float32"}},
                 "params": {"eps": 1e-6}},
                {"id": "flat", "opType": "Reshape", "inputs": {"input": "head_normalized"},
                 "outputs": {"out": {"tensor": "normalized", "shape": [1, 1, 32], "dtype": "float32"}},
                 "params": {"shape": [1, 1, 32]}},
            ])
            source = "normalized"
        graph["nodes"].append({"id": "logits", "opType": "Add", "inputs": {"a": source, "b": "parameter"},
            "outputs": {"out": {"tensor": "logits", "shape": [1, 1, 32], "dtype": "float32"}}, "params": {}})
        graph["outputs"] = ["logits"]
        graph = stateless_graph({k: v for k, v in graph.items() if k != "execution"})
        package = self.package(graph)
        weights = {"parameter": np.zeros(32, dtype=np.float32)}
        if kind == "norm":
            weights["norm_weight"] = np.linspace(.7, 1.3, 8, dtype=np.float32)
        save_file(weights, package / "model.safetensors")
        return package, kv_heads

    def test_training_refuses_inference_domains_before_binding_or_optimizer_mutation(self):
        with vx.open_library() as host:
            platform = vx.VxPlatformServiceClient(host).get_platform_info(vx.pb.Empty())
            if platform.build_profile != vx.pb.BuildProfile.BUILD_PROFILE_FULL:
                self.skipTest("training refusal requires the full profile")
            inference, training = vx.VxInferenceServiceClient(host), vx.VxTrainingServiceClient(host)
            for kind in ("gqa", "explicit_mha", "rope", "legacy_mha", "norm"):
                with self.subTest(kind=kind):
                    package, kv_heads = self.training_package(kind)
                    runtime = inference.create_runtime(vx.pb.CreateRuntimeRequest())
                    model = inference.load_model(vx.pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                        graph_path=str(package / "graph.json"), weight_paths=[str(package / "model.safetensors")]))
                    trainer = training.create_trainer(vx.pb.CreateTrainerRequest(model_id=model.model_id, backend="cpu"))
                    tref = vx.pb.TrainerRef(trainer_id=trainer.trainer_id)
                    try:
                        before = training.get_trainer_state(tref)
                        values = attention_values(1, 1, 2, 8, kv_heads)
                        request = vx.pb.TrainStepRequest(trainer_id=trainer.trainer_id,
                            inputs=[tensor(name, value) for name, value in values.items()],
                            losses=[vx.pb.CrossEntropyLoss(name="ce", logits_name="logits", targets=tensor("", np.array([0], dtype=np.int32)), ignore_index=-100, row_index=-1)],
                            trainable_names=["parameter"], optimizer=vx.pb.TrainerOptimizerOptions(kind=vx.pb.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD, learning_rate=.01),
                            accumulation_steps=1, flush_accumulation=True, reset_accumulation=True)
                        if kind in ("gqa", "explicit_mha", "rope"):
                            with self.assertRaises(vx.VolvoxAIError) as failure:
                                training.train_step(request)
                            self.assertEqual(failure.exception.status, vx.pb.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED)
                            self.assertEqual(failure.exception.code, vx.pb.OperationCode.OPERATION_CODE_TRAINING_GRAPH_UNSUPPORTED)
                            debug = vx.VxDebugServiceClient(host)
                            with self.assertRaises(vx.VolvoxAIError) as debug_failure:
                                debug.create_debug_session(vx.pb.CreateDebugSessionRequest(train_step=request))
                            self.assertEqual(debug_failure.exception.status, vx.pb.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED)
                            self.assertEqual(debug_failure.exception.code, vx.pb.OperationCode.OPERATION_CODE_TRAINING_GRAPH_UNSUPPORTED)
                            after = training.get_trainer_state(tref)
                            for field in ("optimizer_step", "rng_seed", "accumulated_microbatches", "accumulation_steps", "activation_signature", "optimizer_bytes", "gradient_bytes", "has_private_updates"):
                                self.assertEqual(getattr(before, field), getattr(after, field), field)
                            for field in ("signature", "activation_capacity_bytes", "activation_high_water_bytes", "activation_grow_count", "plan_cache_entries", "plan_cache_hits", "plan_cache_misses", "plan_cache_evictions"):
                                self.assertEqual(getattr(before.shape, field), getattr(after.shape, field), field)
                        else:
                            step = training.train_step(request)
                            self.assertTrue(step.update_applied)
                            self.assertEqual(step.optimizer_step, 1)
                        path = package / "after.safetensors"
                        training.export_trainer_weights(vx.pb.ExportTrainerWeightsRequest(trainer_id=trainer.trainer_id, output_paths=[str(path)]))
                        actual = load_file(path)["parameter"]
                        if kind in ("gqa", "explicit_mha", "rope"):
                            np.testing.assert_array_equal(actual, np.zeros_like(actual))
                        else:
                            logits = attention_reference(values, 8, 4).reshape(4, 8).astype(np.float64)
                            if kind == "norm":
                                logits = logits / np.sqrt(np.mean(logits ** 2, axis=1, keepdims=True) + 1e-6) * np.linspace(.7, 1.3, 8, dtype=np.float32)
                            logits = logits.ravel()
                            probabilities = np.exp(logits - logits.max()); probabilities /= probabilities.sum()
                            probabilities[0] -= 1
                            np.testing.assert_allclose(actual, -.01 * probabilities, atol=2e-6, rtol=2e-5)
                    finally:
                        training.release_trainer(tref)
                        inference.release_model(vx.pb.ModelRef(model_id=model.model_id))
                        inference.release_runtime(vx.pb.RuntimeRef(runtime_id=runtime.runtime_id))


if __name__ == "__main__":
    unittest.main()
