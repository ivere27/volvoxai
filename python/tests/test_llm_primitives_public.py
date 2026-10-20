"""Tiny independent modern-decoder oracles through generated public dispatch.

VOLVOXAI_TEST_LLM_BACKEND=cuda requires CUDA without CPU operator fallback.
The caller selects an explicit FULL/inference library with the normal SDK env.
These are numerical tests, not a performance or whole-model qualification.
"""
from __future__ import annotations

from contextlib import contextmanager
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution_fixture import stateless_graph
from test_modern_moe import save_weights
import volvoxai as vx


def softmax(x, axis=-1):
    x = np.asarray(x, np.float64)
    exp = np.exp(x - x.max(axis=axis, keepdims=True))
    return exp / exp.sum(axis=axis, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, np.float64)))


def rotary(x, cos, sin, width, offset=0, interleaved=False):
    result = x.astype(np.float64).copy()
    part = result[..., offset:offset + width].copy()
    if interleaved:
        turn = np.stack((-part[..., 1::2], part[..., ::2]), -1).reshape(part.shape)
        cos, sin = np.repeat(cos, 2, -1), np.repeat(sin, 2, -1)
    else:
        turn = np.concatenate((-part[..., width // 2:], part[..., :width // 2]), -1)
    result[..., offset:offset + width] = part * cos[..., None, :] + turn * sin[..., None, :]
    return result


class PublicModernDecoderTests(unittest.TestCase):
    backend = os.environ.get("VOLVOXAI_TEST_LLM_BACKEND", "cpu")
    observations = []
    wasm_cases = []

    def setUp(self):
        self.rng = np.random.default_rng(94)
        self._wasm_pending = None

    def random(self, shape):
        return self.rng.normal(0, .25, shape).astype(np.float32)

    @classmethod
    def tearDownClass(cls):
        target = os.environ.get("VOLVOXAI_TEST_LLM_OBSERVATIONS")
        if target:
            with Path(target).open("x") as stream:
                json.dump(cls.observations, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        target = os.environ.get("VOLVOXAI_TEST_LLM_WASM_CASES")
        if target:
            with Path(target).open("x") as stream:
                json.dump({"format": "volvox-modern-primitives-wasm-oracles/v1",
                    "cases": cls.wasm_cases, "count": len(cls.wasm_cases),
                    "tolerance": {"absolute": 1e-5, "relative": 1e-5},
                    "expected_source": "independent NumPy references, not actual native outputs"},
                    stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

    @staticmethod
    def tensor_payload(name, value, *, oracle=False):
        if oracle and value.dtype != np.int32:
            value = np.ascontiguousarray(value, dtype="<f8")
            dtype = "F64"
        else:
            dtype = "I32" if value.dtype == np.int32 else "F32"
            value = np.ascontiguousarray(value, dtype="<i4" if dtype == "I32" else "<f4")
        return {"name": name, "shape": list(value.shape), "dtype": dtype,
                "data": base64.b64encode(value.tobytes()).decode("ascii")}

    @contextmanager
    def session(self, operator, inputs, outputs, params=None, weights=None):
        weights = weights or {}
        with tempfile.TemporaryDirectory(prefix="volvox-public-llm-") as tmp:
            directory = Path(tmp)
            tensors = {**inputs, **weights}
            graph = stateless_graph({"format": "volvox-graph/v1", "dimensions": {},
                "inputs": {name: {"shape": list(value.shape),
                    "dtype": "int32" if value.dtype == np.int32 else "float32"}
                    for name, value in inputs.items()},
                "nodes": [{"id": "primitive", "opType": operator,
                    "inputs": {name: name for name in tensors},
                    "outputs": {port: {"tensor": port, "shape": list(shape),
                        "dtype": "int32" if (operator in ("NGramIds", "QSAIndexer", "SparseIndexer")
                            or (operator == "Gather" and tensors["input"].dtype == np.int32)) else "float32"}
                        for port, shape in outputs.items()}, "params": params or {}}],
                "outputs": list(outputs)})
            (directory / "graph.json").write_text(json.dumps(graph))
            save_weights(directory / "model.safetensors", weights)
            with vx.InferenceSession(tmp, backend=self.backend) as session:
                self.assertEqual(session.backend, self.backend)
                self.assertFalse(session.report.fallback.operator_fallback_used)
                if self.backend == "cuda":
                    self.assertEqual(session.report.compilation.policy_mode,
                        vx.pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE)
                    self.assertEqual(session.report.compilation.operator_fallback,
                        vx.pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)
                original = session._engine.execute

                def observed(request, **kwargs):
                    response = original(request, **kwargs)
                    report = response.report
                    self.assertEqual(report.backend, self.backend)
                    self.assertTrue(report.route.builtin and report.route.attested)
                    self.assertEqual(report.route.selected_nodes, report.route.active_nodes)
                    self.assertGreater(report.route.active_nodes, 0)
                    self.assertEqual(report.route.fallback_nodes, 0)
                    self.assertEqual(report.route.missing_nodes, 0)
                    self.assertFalse(report.fallback.operator_fallback_used)
                    self.observations.append({"test": self.id(), "operator": operator,
                        "backend": report.backend, "profile": session.build_profile,
                        "active": report.route.active_nodes, "selected": report.route.selected_nodes,
                        "fallback": report.route.fallback_nodes, "missing": report.route.missing_nodes,
                        "attested": report.route.attested})
                    return response

                session._engine.execute = observed
                if os.environ.get("VOLVOXAI_TEST_LLM_WASM_CASES"):
                    original_run = session.run
                    shard = base64.b64encode((directory / "model.safetensors").read_bytes()).decode("ascii")

                    def exported_run(values, **kwargs):
                        self._wasm_pending = {"name": f"{self.id()}:{len(self.wasm_cases)}",
                            "graph": graph, "weight_shards": [shard],
                            "inputs": [self.tensor_payload(name, value) for name, value in values.items()]}
                        return original_run(values, **kwargs)

                    session.run = exported_run
                yield session

    def compare(self, actual, expected):
        for name, value in expected.items():
            self.assertEqual(actual[name].dtype, np.int32 if value.dtype == np.int32 else np.float32)
            if value.dtype == np.int32:
                np.testing.assert_array_equal(actual[name], value)
            else:
                np.testing.assert_allclose(actual[name], value, atol=1e-5, rtol=1e-5)
        # Capture the explicit reference argument only after comparison passes.
        # Retained-result checks do not create extra cases without a new run.
        if self._wasm_pending is not None:
            self._wasm_pending["expected"] = [self.tensor_payload(name, value, oracle=True)
                for name, value in expected.items()]
            self.wasm_cases.append(self._wasm_pending)
            self._wasm_pending = None

    def test_packed_linear_a16_a8_tail_groups_and_rows(self):
        # Independently encode INT4/INT8 and calculate the specified arithmetic.
        # The same cases are exported to both WASM profiles and native GPUs.
        for weight_bits in (4, 8):
            per_word, offset = 32 // weight_bits, 1 << (weight_bits - 1)
            # The final shape also reaches CUDA matrix/shared-column launches;
            # small N alone would cover only the per-output warp fallback.
            for rows, n, k in ((1, 3, 7), (3, 7, 129), (17, 9, 257), (32, 16, 256), (32, 512, 33)):
                x = self.random((rows, k))
                x[0] = 0
                q = self.rng.integers(-offset, offset, (n, k), dtype=np.int32)
                zero_word = sum(offset << (i * weight_bits) for i in range(per_word))
                packed = np.full((n, (k + per_word - 1) // per_word), zero_word, np.uint32)
                for j in range(k):
                    shift = weight_bits * (j % per_word)
                    packed[:, j // per_word] &= np.uint32(0xffffffff ^ ((2 * offset - 1) << shift))
                    packed[:, j // per_word] |= (q[:, j] + offset).astype(np.uint32) << shift
                scales = self.rng.uniform(.001, .05, (n, (k + 127) // 128)).astype(np.float32)
                bias = self.random((n,))
                for bits in (8, 16):
                    expected = np.zeros((rows, n), np.float32)
                    if bits == 16:
                        activation = x.astype(np.float16).astype(np.float32)
                        weights = (q * np.repeat(scales, 128, axis=1)[:, :k]).astype(np.float32).astype(np.float16).astype(np.float32)
                        for j in range(k):
                            expected += activation[:, j, None] * weights[None, :, j]
                    else:
                        maximum = np.abs(x).max(axis=1)
                        sa = np.where(maximum == 0, np.float32(1), np.maximum(maximum / np.float32(127), np.float32(2**-126)))
                        activation = np.clip(np.rint(x / sa[:, None]), -127, 127).astype(np.int32)
                        for start in range(0, k, 128):
                            end = min(start + 128, k)
                            dot = activation[:, start:end] @ q[:, start:end].T
                            expected += dot.astype(np.float32) * (sa[:, None] * scales[None, :, start // 128])
                    expected += bias
                    with self.subTest(rows=rows, n=n, k=k, weight_bits=weight_bits, bits=bits), self.session(
                            "PackedLinear", {"input": x}, {"out": expected.shape},
                            {"group_size": 128, "weight_bits": weight_bits, "activation_bits": bits},
                            {"weight": packed.view(np.int32), "scale": scales, "bias": bias}) as session:
                        self.compare(session.run({"input": x}), {"out": expected})

    def test_packed_linear_rejects_bad_padding_and_group_scales(self):
        for weight_bits in (4, 8):
            per_word, offset, width = 32 // weight_bits, 1 << (weight_bits - 1), 5
            zero_word = sum(offset << (i * weight_bits) for i in range(per_word))
            words = np.full((1, (width + per_word - 1) // per_word), zero_word, np.uint32)
            invalid_padding = words.copy()
            invalid_padding[0, -1] ^= np.uint32(1 << (weight_bits * (width % per_word)))
            cases = [("padding", invalid_padding, np.float32(1))]
            cases.extend((str(scale), words, np.float32(scale)) for scale in (0, -1, np.inf, np.nan))
            for name, packed, scale in cases:
                with self.subTest(weight_bits=weight_bits, invalid=name), self.assertRaises(vx.VolvoxAIError):
                    with self.session("PackedLinear", {"input": np.ones((1, width), np.float32)},
                            {"out": (1, 1)}, {"group_size": 128, "weight_bits": weight_bits, "activation_bits": 16},
                            {"weight": packed.view(np.int32), "scale": np.array([[scale]], np.float32)}):
                        self.fail("invalid immutable packed storage was accepted")

    def test_gated_delta_grouped_heads_and_initial_state(self):
        b, s, hk, hv, dk, dv = 2, 4, 2, 4, 3, 5
        q, k = self.random((b, s, hk, dk)), self.random((b, s, hk, dk))
        v, a, beta = self.random((b, s, hv, dv)), self.random((b, s, hv)), self.random((b, s, hv))
        alog, dt = self.random((hv,)), self.random((hv,))
        for initial in (None, self.random((b, hv, dk, dv))):
            state = np.zeros((b, hv, dk, dv), np.float64) if initial is None else initial.astype(np.float64).copy()
            expected = np.empty(v.shape, np.float64)
            qn = q / np.sqrt((q.astype(np.float64) ** 2).sum(-1, keepdims=True) + 1e-6)
            kn = k / np.sqrt((k.astype(np.float64) ** 2).sum(-1, keepdims=True) + 1e-6)
            for t in range(s):
                for h in range(hv):
                    state[:, h] *= np.exp(-np.exp(alog[h]) * np.logaddexp(0, a[:, t, h] + dt[h]))[:, None, None]
                    key = kn[:, t, h // (hv // hk)]
                    prediction = np.einsum("bd,bdv->bv", key, state[:, h])
                    state[:, h] += key[:, :, None] * ((v[:, t, h] - prediction) * sigmoid(beta[:, t, h])[:, None])[:, None, :]
                    expected[:, t, h] = np.einsum("bd,bdv->bv", qn[:, t, h // (hv // hk)], state[:, h]) / np.sqrt(dk)
            values = {"q": q, "k": k, "v": v, "a": a, "b": beta}
            if initial is not None:
                values["initial_state"] = initial
            with self.subTest(initial=initial is not None), self.session("GatedDeltaNet", values,
                    {"out": v.shape, "state": state.shape}, {"normalize": True, "scale": 1 / np.sqrt(dk)},
                    {"a_log": alog, "dt_bias": dt}) as session:
                self.compare(session.run(values), {"out": expected, "state": state})

    def test_sparse_gqa_unequal_lengths_masks_sinks_and_all_masked(self):
        b, qs, ks, hq, hk, dk, dv = 2, 3, 5, 4, 2, 3, 5
        q, k, v = self.random((b, qs, hq, dk)), self.random((b, ks, hk, dk)), self.random((b, ks, hk, dv))
        keep = np.ones((b, qs, ks), np.int32)
        keep[0, 1, 0] = 0
        keep[1, 0] = 0
        for sinks in (None, self.random((hq,))):
            expected = np.zeros((b, qs, hq, dv), np.float64)
            for batch in range(b):
                for t in range(qs):
                    selected = np.flatnonzero((np.arange(ks) <= t) & keep[batch, t].astype(bool))
                    for h in range(hq):
                        logits = k[batch, selected, h // 2].astype(np.float64) @ q[batch, t, h] / np.sqrt(dk)
                        if sinks is not None:
                            probability = softmax(np.r_[logits, sinks[h]])[:-1]
                        elif not len(selected):
                            continue
                        else:
                            probability = softmax(logits)
                        expected[batch, t, h] = probability @ v[batch, selected, h // 2]
            values = {"q": q, "k": k, "v": v, "mask": keep}
            with self.subTest(sinks=sinks is not None), self.session("SparseAttention", values,
                    {"out": expected.shape}, {"causal": True}, {} if sinks is None else {"sinks": sinks}) as session:
                self.compare(session.run(values), {"out": expected})

    def test_sparse_compressed_index_ties_absolute_positions_and_refusal(self):
        b, s, h, d = 1, 8, 2, 3
        q, k, v = self.random((b, s, h, d)), self.random((b, s, 1, d)), self.random((b, s, 1, d))
        bank, iq, ik, iw = self.random((b, s, d)), np.zeros((b, s, 2, 2), np.float32), self.random((b, s, 2)), self.random((b, s, 2))
        positions = np.full((s,), 7, np.int32)
        expected = np.empty((b, s, h, d), np.float64)
        for t in range(s):
            local = np.arange(max(0, t - 1), t + 1)
            for head in range(h):
                keys = np.concatenate((k[0, local, 0], bank[0, :1]))
                vals = np.concatenate((v[0, local, 0], bank[0, :1]))
                expected[0, t, head] = softmax(keys.astype(np.float64) @ q[0, t, head]) @ vals
        values = {"q": q, "k": k, "v": v, "compressed_kv": bank, "index_q": iq,
                  "index_k": ik, "index_weights": iw, "position_ids": positions}
        with self.session("SparseAttention", values, {"out": expected.shape},
                {"causal": True, "scale": 1., "window_size": 2, "compress_rate": 4, "top_k": 1}) as session:
            first = session.run(values)
            self.compare(first, {"out": expected})
            bad = positions.copy(); bad[0] = -1
            with self.assertRaises(vx.VolvoxAIError):
                session.run({**values, "position_ids": bad})
            if self.backend == "cpu":
                # CPU value rejection occurs in the kernel after commit; that
                # context is terminal. CUDA rejects public values precommit.
                with self.session("SparseAttention", values, {"out": expected.shape},
                        {"causal": True, "scale": 1., "window_size": 2, "compress_rate": 4, "top_k": 1}) as restarted:
                    self.compare(restarted.run(values), {"out": expected})
            else:
                self.compare(session.run(values), {"out": expected})
            self.compare(first, {"out": expected})

    def test_compression_overlap_rotary_and_zero_tail(self):
        for groups in (1, 2):
            b, s, rate, d = 2, 7, 2, 4
            x, gates = self.random((b, s, groups * d)), self.random((b, s, groups * d))
            bias, weight = self.random((rate, groups * d)), self.random((d,))
            angles = np.arange(s)[:, None] * np.asarray([[.1, .3]])
            cos, sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
            expected = np.zeros((b, s, d), np.float64)
            for window in range(s // rate):
                kv = x[:, window * rate:(window + 1) * rate, -d:]
                scores = gates[:, window * rate:(window + 1) * rate, -d:] + bias[None, :, -d:]
                if groups == 2 and window:
                    kv = np.concatenate((x[:, (window - 1) * rate:window * rate, :d], kv), 1)
                    scores = np.concatenate((gates[:, (window - 1) * rate:window * rate, :d] + bias[None, :, :d], scores), 1)
                value = (softmax(scores, 1) * kv).sum(1)
                value = value / np.sqrt((value ** 2).mean(-1, keepdims=True) + 1e-6) * weight
                expected[:, window, ::2] = value[:, ::2] * cos[window] - value[:, 1::2] * sin[window]
                expected[:, window, 1::2] = value[:, 1::2] * cos[window] + value[:, ::2] * sin[window]
            values = {"input": x, "a": gates}
            with self.subTest(groups=groups), self.session("KVCompress", values, {"out": expected.shape},
                    {"num_groups": groups, "compress_rate": rate, "rotary_dim": d, "interleaved": True},
                    {"bias": bias, "weight": weight, "cos": cos, "sin": sin}) as session:
                self.compare(session.run(values), {"out": expected})

    def test_hyper_connection_residual_and_head(self):
        b, s, m, d = 1, 3, 3, 4
        x = self.random((b, s, m, d)); width = m * d; count = 2 * m + m * m
        weight, bias, scales = self.random((count, width)), self.random((count,)), self.random((3,))
        flat = x.reshape(b, s, width).astype(np.float64)
        normal = flat / np.sqrt((flat ** 2).mean(-1, keepdims=True) + 1e-6)
        raw = normal @ weight.T
        pre = sigmoid(raw[..., :m] * scales[0] + bias[:m]) + 1e-6
        post = 2 * sigmoid(raw[..., m:2 * m] * scales[1] + bias[m:2 * m])
        comb = softmax((raw[..., 2 * m:] * scales[2] + bias[2 * m:]).reshape(b, s, m, m)) + 1e-6
        for iteration in range(5):
            if iteration:
                comb /= comb.sum(-1, keepdims=True) + 1e-6
            comb /= comb.sum(-2, keepdims=True) + 1e-6
        with self.session("HyperConnection", {"input": x}, {"pre": pre.shape, "post": post.shape, "comb": comb.shape},
                {"iterations": 5}, {"weight": weight, "bias": bias, "scale": scales}) as session:
            self.compare(session.run({"input": x}), {"pre": pre, "post": post, "comb": comb})
        update = self.random((b, s, d))
        residual = post[..., None] * update[..., None, :] + np.einsum("bstu,bstd->bsud", comb, x)
        values = {"input": x, "update": update, "post": post.astype(np.float32), "comb": comb.astype(np.float32)}
        with self.session("HyperResidual", values, {"out": x.shape}) as session:
            self.compare(session.run(values), {"out": residual})
        hw, hb, hs = self.random((m, width)), self.random((m,)), self.random((1,))
        expected_head = ((sigmoid((normal @ hw.T) * hs[0] + hb) + 1e-6)[..., None] * x).sum(-2)
        with self.session("HyperHead", {"input": x}, {"out": expected_head.shape}, weights={"weight": hw, "bias": hb, "scale": hs}) as session:
            self.compare(session.run({"input": x}), {"out": expected_head})

    def test_rotary_partial_trailing_features_and_layouts(self):
        x = self.random((2, 3, 2, 512)); width, offset = 64, 448
        for interleaved in (False, True):
            angles = self.random((2, 3, width // 2))
            cos, sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
            if not interleaved:
                cos, sin = np.concatenate((cos, cos), -1), np.concatenate((sin, sin), -1)
            expected = rotary(x, cos, sin, width, offset, interleaved)
            with self.subTest(interleaved=interleaved), self.session("RotaryEmbedding", {"input": x}, {"out": x.shape},
                    {"rotary_dim": width, "position_offset": offset, "interleaved": interleaved}, {"cos": cos, "sin": sin}) as session:
                self.compare(session.run({"input": x}), {"out": expected})

    def test_ngram_wrapped_integer_hash_eos_and_atomic_refusal(self):
        tokens = np.asarray([[4, 8, 2, 6, 7, 1], [2, 3, 4, 2, 5, 8]], np.int32)
        multipliers = [2 ** 62 + 31, 2 ** 61 + 13, 1073741827, 4294967311]
        words = np.asarray([[x & 0xffffffff, x >> 32] for x in multipliers], np.uint32).view(np.int32)
        sizes = np.asarray([17, 19, 23, 29, 31, 37], np.int32)
        offsets = np.asarray([0, 17, 36, 59, 88, 119], np.int32)
        expected = np.empty((*tokens.shape, 6), np.int32)
        for batch in range(2):
            for t in range(6):
                for head in range(6):
                    mixed = int(tokens[batch, t]) * multipliers[0]
                    blocked = False
                    for shift in range(1, head // 2 + 2):
                        value = 2
                        if shift <= t and not blocked:
                            value = int(tokens[batch, t - shift]); blocked = value == 2
                        mixed ^= value * multipliers[shift]
                    mixed &= 2 ** 64 - 1
                    if mixed >= 2 ** 63:
                        mixed -= 2 ** 64
                    expected[batch, t, head] = mixed % int(sizes[head]) + offsets[head]
        with self.session("NGramIds", {"token_ids": tokens}, {"out": expected.shape},
                {"ngram_size": 4, "heads_per_ngram": 2, "eos_token_id": 2, "vocab_size": 9},
                {"multipliers": words, "vocab_sizes": sizes, "offsets": offsets}) as session:
            self.compare(session.run({"token_ids": tokens}), {"out": expected})
            bad = tokens.copy(); bad[0, 0] = 9
            with self.assertRaises(vx.VolvoxAIError):
                session.run({"token_ids": bad})
            if self.backend == "cpu":
                with self.session("NGramIds", {"token_ids": tokens}, {"out": expected.shape},
                        {"ngram_size": 4, "heads_per_ngram": 2, "eos_token_id": 2, "vocab_size": 9},
                        {"multipliers": words, "vocab_sizes": sizes, "offsets": offsets}) as restarted:
                    self.compare(restarted.run({"token_ids": tokens}), {"out": expected})
            else:
                self.compare(session.run({"token_ids": tokens}), {"out": expected})

    def test_ngram_padding_without_token_reset(self):
        tokens = np.array([[4, 2, 6, 2, 7]], np.int32)
        multipliers = [2 ** 61 + 3, 1073741827, 4294967311]
        words = np.asarray([[x & 0xffffffff, x >> 32] for x in multipliers], np.uint32).view(np.int32)
        sizes, offsets = np.array([17, 23], np.int32), np.array([0, 17], np.int32)
        expected = np.empty((1, 5, 2), np.int32)
        for t in range(5):
            for head in range(2):
                mixed = 0
                for shift in range(head + 2):
                    token = int(tokens[0, t-shift]) if shift <= t else 2
                    mixed ^= token * multipliers[shift]
                mixed = (mixed + 2**63) % 2**64 - 2**63
                expected[0, t, head] = mixed % int(sizes[head]) + offsets[head]
        with self.session("NGramIds", {"token_ids": tokens}, {"out": expected.shape},
                {"ngram_size": 3, "heads_per_ngram": 1, "eos_token_id": -1,
                 "pad_token_id": 2, "vocab_size": 9},
                {"multipliers": words, "vocab_sizes": sizes, "offsets": offsets}) as session:
            self.compare(session.run({"token_ids": tokens}), {"out": expected})

    def test_sparse_indexer_shared_candidate_source_consumer_and_attention(self):
        b,s,h,d,rate,topk=2,15,2,3,2,3
        q,k,w=self.random((b,s,h,d)),self.random((b,s,d)),self.random((b,s,h))
        q[0]=0  # Exact ties: source final top-k and newest candidate block differ.
        score=(np.maximum(np.einsum("bthd,bjd->bthj",q.astype(np.float64),k.astype(np.float64)),0)*w.astype(np.float64)[...,None]).sum(2)/np.sqrt(d*h)
        selected=np.zeros((b,s,s),np.int32);candidates=np.zeros_like(selected)
        for bi in range(b):
            for t in range(s):
                visible=(t+1)//rate
                order=np.lexsort((np.arange(visible),-score[bi,t,:visible]))
                selected[bi,t,order[:topk]]=1
                if visible: candidates[bi,t,((visible-1)//2)*2:min(((visible-1)//2+1)*2,s//rate)]=1
        values={"q":q,"k":k,"index_weights":w}
        with self.session("SparseIndexer",values,{"out":selected.shape,"candidates":candidates.shape},
                {"compress_rate":rate,"top_k":topk,"candidate_block_size":2,"candidate_top_k":1}) as session:
            self.compare(session.run(values),{"out":selected,"candidates":candidates})
        restricted=np.zeros_like(selected)
        for bi in range(b):
            for t in range(s):
                visible=(t+1)//rate
                scores=np.where(candidates[bi,t,:visible],score[bi,t,:visible],-np.inf)
                restricted[bi,t,np.lexsort((np.arange(visible),-scores))[:topk]]=1
        # Official top-k semantics can fill an undersized candidate set with
        # visible -inf entries. Those exact ties retain the low-index rule.
        self.assertEqual(restricted[0,-1,0],1)
        values["candidates"]=candidates
        with self.session("SparseIndexer",values,{"out":selected.shape},{"compress_rate":rate,"top_k":topk}) as session:
            self.compare(session.run(values),{"out":restricted})
        aq,ak,av,bank=self.random((b,s,h,d)),self.random((b,s,1,d)),self.random((b,s,1,d)),self.random((b,s,d))
        expected=np.zeros_like(aq,dtype=np.float64)
        for bi in range(b):
            for t in range(s):
                selected_rows=np.flatnonzero(restricted[bi,t,:((t+1)//rate)])
                keys=np.concatenate((ak[bi,:t+1,0],bank[bi,selected_rows])).astype(np.float64)
                values=np.concatenate((av[bi,:t+1,0],bank[bi,selected_rows])).astype(np.float64)
                for head in range(h):expected[bi,t,head]=softmax(keys@aq[bi,t,head].astype(np.float64))@values
        values={"q":aq,"k":ak,"v":av,"compressed_kv":bank,"compressed_mask":restricted}
        with self.session("SparseAttention",values,{"out":expected.shape},{"compress_rate":rate,"scale":1.,"causal":True}) as session:
            self.compare(session.run(values),{"out":expected})

    def test_qsa_visible_groups_rotary_tail_and_tie_policy(self):
        b, s, h, d, rate = 2, 7, 2, 4, 2
        q, k, weight = self.random((b, s, h, d)), self.random((b, s, d)), self.random((d,))
        angles = self.random((b, s, d // 2))
        cos = np.concatenate((np.cos(angles), np.cos(angles)), -1).astype(np.float32)
        sin = np.concatenate((np.sin(angles), np.sin(angles)), -1).astype(np.float32)
        keep = np.ones((b, s), np.int32); keep[0, 1] = 0; keep[1, :2] = 0
        for tied in (False, True):
            query_values = np.zeros_like(q) if tied else q
            expected = np.zeros((b, s, s), np.int32)
            for batch in range(b):
                for query in range(s):
                    visible = np.flatnonzero((np.arange(s) <= query) & keep[batch].astype(bool))
                    complete = len(visible) // rate
                    if complete:
                        groups = visible[:complete * rate].reshape(complete, rate)
                        bank = k[batch, groups].astype(np.float64).mean(1)
                        bank = bank / np.sqrt((bank ** 2).mean(-1, keepdims=True) + 1e-6) * weight
                        left, right = bank[:, :d // 2].copy(), bank[:, d // 2:].copy()
                        cc, ss = cos[batch, groups[:, 0]], sin[batch, groups[:, 0]]
                        bank[:, :d // 2] = left * cc[:, :d // 2] - right * ss[:, :d // 2]
                        bank[:, d // 2:] = right * cc[:, d // 2:] + left * ss[:, d // 2:]
                        scores = np.maximum(query_values[batch, query].astype(np.float64) @ bank.T, 0).sum(0) / np.sqrt(d)
                        winner = int(np.argmax(scores))
                        expected[batch, query, groups[winner]] = 1
                    expected[batch, query, visible[complete * rate:]] = 1
            values = {"q": query_values, "k": k, "mask": keep}
            with self.subTest(tied=tied), self.session("QSAIndexer", values, {"out": expected.shape},
                    {"compress_rate": rate, "top_k": 1, "rotary_dim": d}, {"weight": weight, "cos": cos, "sin": sin}) as session:
                self.compare(session.run(values), {"out": expected})

    def test_signed_sqrt_negative_zero_and_small_values(self):
        x = np.asarray([-9, -1e-9, -0., 0., 1e-9, 4], np.float32)
        expected = np.sign(x) * np.sqrt(np.maximum(np.abs(x), 1e-6))
        with self.session("SignedSqrt", {"input": x}, {"out": x.shape}) as session:
            actual = session.run({"input": x})
            self.compare(actual, {"out": expected})
            self.assertFalse(np.signbit(actual["out"][2]))

    def test_gather_preserves_i32_words_and_f32_storage(self):
        integers = np.asarray([-(2**31), 2**31 - 1, 2**24 + 1, -(2**24 + 1),
                               0, -1, 0x7fc00001, 0x01000001], np.int32).reshape(2, 2, 2)
        float_bits = np.asarray([0, 0x80000000, 1, 0x80000001,
                                 0x3f800000, 0xbf800000, 0x7f000001, 0x00800001], np.uint32)
        floats = float_bits.view(np.float32).reshape(2, 2, 2)
        indices = np.asarray([[0, -1], [1, -2]], np.int32)
        for data in (integers, floats):
            for axis in (0, 1, -1):
                expected = np.take(data, indices, axis=axis)
                values = {"input": data, "indices": indices}
                with self.subTest(dtype=str(data.dtype), axis=axis), self.session(
                        "Gather", values, {"out": expected.shape}, {"axis": axis}) as session:
                    actual = session.run(values)
                    self.compare(actual, {"out": expected})
                    np.testing.assert_array_equal(actual["out"].view(np.uint32),
                                                  expected.view(np.uint32))

    def test_embedding_shard_large_integer_ids_and_refusal(self):
        ids = np.asarray([[0, 16777219, 16777220, 300000000]], np.int32)
        bank = self.random((2, 4)); expected = np.zeros((1, 4, 4), np.float32); expected[0, 1:3] = bank
        with self.session("EmbeddingShard", {"input": ids}, {"out": expected.shape},
                {"vocab_offset": 16777219, "vocab_size": 300000001}, {"weight": bank}) as session:
            first = session.run({"input": ids})
            self.compare(first, {"out": expected})
            bad = ids.copy(); bad[0, 0] = 300000001
            with self.assertRaises(vx.VolvoxAIError):
                session.run({"input": bad})
            if self.backend == "cpu":
                with self.session("EmbeddingShard", {"input": ids}, {"out": expected.shape},
                        {"vocab_offset": 16777219, "vocab_size": 300000001}, {"weight": bank}) as restarted:
                    self.compare(restarted.run({"input": ids}), {"out": expected})
            else:
                self.compare(session.run({"input": ids}), {"out": expected})
            self.compare(first, {"out": expected})


PB = vx.pb


class PortableLLMWorkspaceBudgetTest(unittest.TestCase):
    def test_omitted_delta_state_is_charged_before_context_creation(self):
        width = 512
        values = {name: np.zeros((1, 1, 1, width), np.float32)
                  for name in ("q", "k", "v")}
        values.update({name: np.zeros((1, 1, 1), np.float32) for name in ("a", "b")})
        values.update({name: np.zeros((1,), np.float32) for name in ("a_log", "dt_bias")})
        graph = {"format": "volvox-graph/v1", "dimensions": {},
            "inputs": {name: {"dtype": "float32", "shape": list(value.shape)}
                       for name, value in values.items()},
            "nodes": [{"id": "delta", "opType": "GatedDeltaNet",
                "inputs": {name: name for name in values}, "params": {},
                "outputs": {"out": {"tensor": "y", "dtype": "float32", "shape": [1, 1, 1, width]}}}],
            "outputs": ["y"], "execution": {"stateless": {}, "entrypoints": [
                {"name": "evaluate", "effect": 1, "allowed_ranges": [1],
                 "inputs": list(values), "outputs": ["y"]}]}}
        host = vx.open_library()
        self.addCleanup(host.close)
        client = vx.VxInferenceServiceClient(host)
        runtime = client.create_runtime(PB.CreateRuntimeRequest(cpu_threads=1))
        self.addCleanup(client.release_runtime, PB.RuntimeRef(runtime_id=runtime.runtime_id))
        model = client.load_model(PB.LoadModelRequest(runtime_id=runtime.runtime_id,
            package=PB.ModelPackage(graph_document=json.dumps(graph).encode())))
        self.addCleanup(client.release_model, PB.ModelRef(model_id=model.model_id))
        compiled = client.compile_model(PB.CompileModelRequest(model_id=model.model_id,
            policy=PB.BackendPolicy(mode=PB.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                backends=["cpu"], operator_fallback=PB.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
        self.addCleanup(client.release_compiled_model,
            PB.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
        self.assertEqual(compiled.report.backend, "cpu")
        scratch_bytes = width * width * np.dtype(np.float32).itemsize
        bounds = {bound.budget_domain_id: bound for bound in compiled.memory_bounds.bounds}
        self.assertLess(bounds["maximum-tensor"].maximum_bytes.bytes, scratch_bytes)
        self.assertGreaterEqual(bounds["provider-resident"].maximum_bytes.bytes, scratch_bytes)
        with self.assertRaises(vx.VolvoxAIError) as refused:
            client.create_execution_context(PB.CreateExecutionContextRequest(
                compiled_model_id=compiled.compiled_model_id, graph=PB.GraphExecution(),
                limits=PB.ExecutionLimits(max_memory_bytes=scratch_bytes - 1)))
        self.assertEqual(refused.exception.status, PB.NativeStatus.NATIVE_STATUS_OUT_OF_MEMORY)
        self.assertEqual(refused.exception.response.context_id, 0)
        context = client.create_execution_context(PB.CreateExecutionContextRequest(
            compiled_model_id=compiled.compiled_model_id, graph=PB.GraphExecution(),
            limits=PB.ExecutionLimits(max_memory_bytes=64 << 20)))
        self.addCleanup(client.release_execution_context,
            PB.ExecutionContextRef(context_id=context.context_id))
        before = client.get_execution_context_info(PB.ExecutionContextRef(context_id=context.context_id))
        self.assertFalse(before.initialized)
        packet = client.execute(PB.ExecuteRequest(context_id=context.context_id,
            scope=PB.ExecutionScope(entrypoint="evaluate", full=PB.Empty()),
            inputs=PB.InputBindings(values=[PB.Tensor(name=name, shape=list(value.shape),
                dtype=PB.DataType.DATA_TYPE_F32, inline=value.tobytes()) for name, value in values.items()])))
        self.assertEqual(packet.report.backend, "cpu")
        self.assertEqual(len(packet.outputs), 1)
        self.assertEqual(packet.outputs[0].shape, [1, 1, 1, width])
        np.testing.assert_array_equal(np.frombuffer(packet.outputs[0].inline, np.float32),
                                      np.zeros(width, np.float32))


if __name__ == "__main__":
    unittest.main()

