#!/usr/bin/env python3
"""Run authoritative portable LLM shaders against independent NumPy oracles.

Requires optional development-only wgpu/numpy packages, a working WebGPU
adapter, and an oracle corpus captured by test_llm_primitives_public. This is
raw shader accuracy evidence; it does not qualify model admission, hardware
performance, giant weights, or an unexecuted native Metal driver.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import math
import struct
from pathlib import Path
import numpy as np
import wgpu

ROOT = Path(__file__).resolve().parents[2]
MODERN_NAMES = "batch sequence keys heads kv_heads key_dim value_dim streams groups rate window top_k iterations index_heads index_dim rotary_dim rotary_offset normalize causal interleaved mask_mode flags work pad0 eps hc_eps scale pad1 pad2 pad3 pad4 pad5".split()
MOE_NAMES = "rows d_in hidden d_out experts top_k normalize scoring groups selected_groups group_top_k has_bias has_correction has_hash vocabulary input_scale slot work pad0 pad1 temperature scale epsilon limit pad2 pad3 pad4 pad5 pad6 pad7 pad8 pad9".split()

def tensor(value):
    return np.frombuffer(base64.b64decode(value["data"]), dtype={"F32": "<f4", "float32": "<f4", "I32": "<i4", "int32": "<i4", "F64": "<f8"}[value["dtype"]]).reshape(value["shape"])

def tensors(fixture):
    result = {value["name"]: tensor(value) for value in fixture["inputs"]}
    for text in fixture["weight_shards"]:
        data = base64.b64decode(text)
        header_size = struct.unpack_from("<Q", data)[0]
        header = json.loads(data[8:8 + header_size])
        for name, value in header.items():
            if name == "__metadata__": continue
            first, end = value["data_offsets"]
            dtype = {"F32": "<f4", "I32": "<i4"}[value["dtype"]]
            result[name] = np.frombuffer(data[8 + header_size + first:8 + header_size + end], dtype=dtype).reshape(value["shape"])
    return result

class ShaderHarness:
    def __init__(self, power_preference="low-power"):
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
        if self.adapter is None: raise RuntimeError("A real WebGPU adapter is required")
        self.device = self.adapter.request_device_sync(required_limits={"max-storage-buffers-per-shader-stage": 8, "max-compute-workgroup-storage-size": 16384})
        self.pipelines = {}
        self.dispatches = 0
        self.owned = []

    def buffer(self, value):
        value = np.ascontiguousarray(value)
        if value.nbytes == 0: raise ValueError("zero storage")
        result = self.device.create_buffer_with_data(data=value, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        self.owned.append(result)
        return result

    def zeros(self, count, integer=False):
        return self.buffer(np.zeros(count, dtype=np.int32 if integer else np.float32))

    def read(self, buffer, integer=False):
        return np.frombuffer(self.device.queue.read_buffer(buffer), dtype=np.int32 if integer else np.float32).copy()

    def stage(self, name, buffers, uniform_binding, params, workgroups=False):
        if name not in self.pipelines:
            source = (ROOT / "shaders/inference" / (name + ".wgsl")).read_text()
            module = self.device.create_shader_module(code=source, label=name)
            self.pipelines[name] = self.device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"}, label=name)
        pipeline = self.pipelines[name]
        words = bytearray(128)
        is_moe = name.startswith("moe")
        names = MOE_NAMES if is_moe else MODERN_NAMES
        float_range = range(20, 24) if is_moe else range(24, 28)
        for index, key in enumerate(names):
            struct.pack_into("<f" if index in float_range else "<I", words, index * 4, params.get(key, 0))
        uniform = self.device.create_buffer_with_data(data=words, usage=wgpu.BufferUsage.UNIFORM)
        entries = [{"binding": i, "resource": {"buffer": value}} for i, value in enumerate(buffers)]
        entries.append({"binding": uniform_binding, "resource": {"buffer": uniform}})
        group = self.device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries)
        encoder = self.device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, group)
        work = params.get("rows", 0) if name == "moeRouterExtended" else params["work"]
        compute.dispatch_workgroups(work if workgroups else (work + 63) // 64)
        compute.end()
        self.device.queue.submit([encoder.finish()])
        self.dispatches += 1
        # Device commands retain the uniform until completion.
        self.owned.append(uniform)

    def cleanup_case(self):
        for resource in self.owned: resource.destroy()
        self.owned.clear()

    def run(self, fixture):
        node = fixture["graph"]["nodes"][0]
        assert len(fixture["graph"]["nodes"]) == 1 or [n["opType"] for n in fixture["graph"]["nodes"]] == ["MoERouter", "MoEGatedMLP"], "raw shader corpus must contain a primitive or router/gated pair"
        op = node["opType"]
        values = tensors(fixture)
        inputs = node["inputs"]
        params = node.get("params", {})
        buffers = {name: self.buffer(value) for name, value in values.items()}
        zero = self.zeros(1)
        def x(role): return buffers[inputs[role]] if role in inputs else zero
        def shape(role): return values[inputs[role]].shape
        outputs = {}
        for role, descriptor in {role: descriptor for n in fixture["graph"]["nodes"] for role, descriptor in n["outputs"].items()}.items():
            expected = next(value for value in fixture["expected"] if value["name"] == descriptor["tensor"])
            outputs[role] = self.zeros(math.prod(expected["shape"]), expected["dtype"] == "I32")
        out = outputs.get("out", outputs.get("pre", zero))
        p = dict(eps=params.get("eps", 1e-6), hc_eps=params.get("normalization_epsilon", 1e-6), scale=params.get("scale", 1.0), flags=0)
        if op not in ("NGramIds", "EmbeddingShard", "SignedSqrt", "MoERouter", "PackedLinear"):
            base = shape("q" if op in ("GatedDeltaNet", "SparseAttention", "QSAIndexer", "SparseIndexer") else "input")
            p.update(batch=base[0], sequence=base[1], key_dim=base[-1])
        if "mask" in inputs:
            ms = shape("mask")
            p["mask_mode"] = 1 if len(ms) == 1 else 4 if len(ms) == 3 else 2 if ms[0] == p["batch"] else 3
        else: p["mask_mode"] = 0
        if op == "GatedDeltaNet":
            p.update(heads=shape("v")[2], kv_heads=shape("k")[2], value_dim=shape("v")[3], normalize=int(params.get("normalize", True)))
            if "scale" not in params: p["scale"] = 1 / math.sqrt(p["key_dim"])
            p["flags"] = int("initial_state" in inputs) | (2 if "state" in outputs else 0)
            gates = self.zeros(4 * p["batch"] * p["sequence"] * p["heads"])
            self.stage("llmDeltaGates", [x("q"), x("k"), x("a"), x("b"), x("a_log"), x("dt_bias"), gates], 7, dict(p, work=p["batch"] * p["sequence"] * p["heads"]))
            self.stage("llmDeltaNet", [x("q"), x("k"), x("v"), gates, x("initial_state"), out, outputs.get("state", zero)], 7, dict(p, work=p["batch"] * p["heads"] * p["value_dim"]))
        elif op == "SparseAttention":
            p.update(keys=shape("k")[1], heads=shape("q")[2], kv_heads=shape("k")[2], value_dim=shape("v")[3], rate=params.get("compress_rate", 1), top_k=params.get("top_k", 0), window=params.get("window_size", 0), causal=int(params.get("causal", True)))
            if "scale" not in params: p["scale"] = 1 / math.sqrt(p["key_dim"])
            p["flags"] = (4 if "compressed_kv" in inputs else 0) | (8 if "sinks" in inputs else 0) | (16 if "index_q" in inputs else 0) | (32 if "position_ids" in inputs else 0) | (128 if "compressed_mask" in inputs else 0)
            if "compressed_mask" in inputs: p["top_k"] = p["keys"]
            selection = zero
            if "compressed_kv" in inputs:
                selection = self.zeros(p["batch"] * p["sequence"] * (p["top_k"] + 1), True)
                if "index_q" in inputs: p.update(index_heads=shape("index_q")[2], index_dim=shape("index_q")[3])
                self.stage("llmSparseSelect", [x("compressed_mask" if "compressed_mask" in inputs else "index_q"), x("index_k"), x("index_weights"), x("position_ids"), selection], 5, dict(p, work=p["batch"] * p["sequence"]), True)
            self.stage("llmSparseAttention", [x("q"), x("k"), x("v"), x("compressed_kv"), x("sinks"), x("mask"), selection, out], 8, dict(p, work=p["batch"] * p["sequence"] * p["heads"]))
        elif op == "KVCompress":
            p.update(key_dim=shape("weight")[0], groups=params.get("num_groups", 1), rate=params["compress_rate"], rotary_dim=params.get("rotary_dim", shape("weight")[0]), rotary_offset=params.get("position_offset", 0), interleaved=int(params.get("interleaved", False)), flags=64 if "cos" in inputs else 0)
            self.stage("llmKVCompress", [x("input"), x("a"), x("bias"), x("weight"), x("cos"), x("sin"), out], 7, dict(p, work=p["batch"] * p["sequence"]))
        elif op in ("HyperConnection", "HyperHead"):
            p.update(streams=shape("input")[2], iterations=params.get("iterations", 20))
            binds = [x("input"), x("weight"), x("bias"), x("scale"), out]
            if op == "HyperConnection": binds += [outputs["post"], outputs["comb"]]
            self.stage("llm" + op, binds, len(binds), dict(p, work=p["batch"] * p["sequence"]))
        elif op == "HyperResidual":
            p["streams"] = shape("input")[2]
            self.stage("llmHyperResidual", [x("input"), x("update"), x("post"), x("comb"), out], 5, dict(p, work=math.prod(shape("input"))))
        elif op == "RotaryEmbedding":
            p.update(heads=shape("input")[2], rotary_dim=params.get("rotary_dim", p["key_dim"]), rotary_offset=params.get("position_offset", 0), interleaved=int(params.get("interleaved", False)))
            self.stage("llmRotaryEmbedding", [x("input"), x("cos"), x("sin"), out], 4, dict(p, work=math.prod(shape("input"))))
        elif op == "SignedSqrt":
            self.stage("llmSignedSqrt", [x("input"), out], 2, dict(p, work=math.prod(shape("input"))))
        elif op == "PackedLinear":
            width=shape("input")[-1]; rows=math.prod(shape("input"))//width
            p.update(batch=1, sequence=rows, key_dim=width, value_dim=shape("weight")[0],
                     groups=params.get("weight_bits",4), rate=params.get("group_size",128),
                     normalize=params.get("activation_bits",16), flags=int("bias" in inputs))
            self.stage("packedLinear",[x("input"),x("weight"),x("scale"),x("bias"),out],5,
                       dict(p,work=rows*p["value_dim"]))
        elif op == "EmbeddingShard":
            p.update(key_dim=shape("weight")[1], value_dim=shape("weight")[0], rotary_offset=params["vocab_offset"])
            self.stage("llmEmbeddingShard", [x("input"), x("weight"), out], 3, dict(p, work=math.prod(shape("input")) * p["key_dim"]))
        elif op == "NGramIds":
            b, s = shape("token_ids")
            h = (params["ngram_size"] - 1) * params["heads_per_ngram"]
            p.update(batch=b, sequence=s, heads=h, groups=params["heads_per_ngram"], rotary_offset=params["eos_token_id"], normalize=params.get("pad_token_id", params["eos_token_id"]))
            self.stage("llmNGramIds", [x("token_ids"), x("multipliers"), x("vocab_sizes"), x("offsets"), out], 5, dict(p, work=b * s * h))
        elif op == "SparseIndexer":
            p.update(keys=shape("k")[1], heads=shape("q")[2], rate=params["compress_rate"], top_k=params["top_k"], window=params.get("candidate_block_size", 0), groups=params.get("candidate_top_k", 0), flags=int("candidates" in inputs) | (2 if "candidates" in outputs else 0))
            work = p["batch"] * p["sequence"] * p["keys"]
            scores = self.zeros(work)
            self.stage("llmSparseIndexerScores", [x("q"), x("k"), x("index_weights"), x("candidates"), scores], 5, dict(p, work=work))
            self.stage("llmSparseIndexerMask", [scores, out, outputs.get("candidates", zero)], 3, dict(p, work=work))
        elif op == "QSAIndexer":
            p.update(keys=shape("k")[1], heads=shape("q")[2], rate=params["compress_rate"], top_k=params["top_k"], rotary_dim=params.get("rotary_dim", p["key_dim"]), causal=int(params.get("causal", True)))
            capacity = (p["keys"] + p["rate"] - 1) // p["rate"]
            scores = self.zeros(p["batch"] * p["sequence"] * capacity)
            self.stage("llmQSAScores", [x("q"), x("k"), x("weight"), x("cos"), x("sin"), x("mask"), scores], 7, dict(p, work=p["batch"] * p["sequence"] * capacity))
            self.stage("llmQSAMask", [scores, x("mask"), out], 3, dict(p, work=p["batch"] * p["sequence"]))
        elif op == "MoERouter":
            width, experts = shape("weight")
            rows = math.prod(shape("input")) // width
            p = dict(rows=rows, d_in=width, experts=experts, top_k=params["top_k"], normalize=int(params.get("normalize", True)), scoring={"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2}[params.get("scoring_func", "softmax")], groups=params.get("num_groups", 1), selected_groups=params.get("topk_groups", params.get("num_groups", 1)), group_top_k=params.get("group_score_top_k", 1), has_bias=int("bias" in inputs), has_correction=int("correction_bias" in inputs), has_hash=int("hash_indices" in inputs), vocabulary=params.get("vocab_size", 0), temperature=params.get("temperature", 1.0), scale=params.get("routed_scaling_factor", 1.0), epsilon=params.get("normalization_epsilon", 0.0))
            self.stage("moeRouterExtended", [x("input"), x("weight"), x("bias"), x("correction_bias"), x("token_ids"), x("hash_indices"), outputs["indices"], outputs["weights"]], 8, p, True)
            if len(fixture["graph"]["nodes"]) == 2:
                gated = fixture["graph"]["nodes"][1]
                gi = gated["inputs"]; gp = gated.get("params", {})
                hidden = values[gi["gate_weight"]].shape[2]
                d_out = values[gi["down_weight"]].shape[2]
                hb = self.zeros(rows * hidden)
                mp = dict(rows=rows, d_in=width, hidden=hidden, d_out=d_out, experts=experts, top_k=p["top_k"], input_scale=int(gp.get("route_input_scale", False)), limit=gp.get("swiglu_limit", 0.0))
                for slot in range(p["top_k"]):
                    self.stage("moeGatedUp", [buffers[gi["input"]], buffers[gi["gate_weight"]], buffers[gi["up_weight"]], outputs["indices"], outputs["weights"], hb], 6, dict(mp, slot=slot, work=rows * hidden))
                    self.stage("moeGatedDown", [hb, buffers[gi["down_weight"]], outputs["indices"], outputs["weights"], outputs["out"]], 5, dict(mp, slot=slot, work=rows * d_out))
        else: raise AssertionError("unsupported raw shader fixture " + op)
        maximum = 0.0
        for role, descriptor in {role: descriptor for n in fixture["graph"]["nodes"] for role, descriptor in n["outputs"].items()}.items():
            expected = next(value for value in fixture["expected"] if value["name"] == descriptor["tensor"])
            oracle = tensor(expected).reshape(-1)
            actual = self.read(outputs[role], expected["dtype"] == "I32")
            if expected["dtype"] == "I32": np.testing.assert_array_equal(actual, oracle, err_msg=fixture["name"])
            else: np.testing.assert_allclose(actual, oracle, atol=1e-5, rtol=1e-5, err_msg=fixture["name"], equal_nan=False)
            maximum = max(maximum, float(np.max(np.abs(actual.astype(np.float64) - oracle))))
        record = dict(name=fixture["name"], operator=op, pass_=True, maximum_error=maximum)
        self.cleanup_case()
        return record

    def large_compressed_case(self):
        # This exceeds the indexed selection array bound, while the model uses
        # every causally complete compressed row and needs no selection array.
        # Expected logits come from an independent F64 softmax, not shader code.
        size, dim, rate = 1028, 2, 2
        rng = np.random.default_rng(6592)
        q, k, v, compressed = [rng.normal(size=(1, size, 1, dim)).astype(np.float32) * .2 for _ in range(4)]
        compressed = compressed.reshape(1, size, dim)
        oracle = np.zeros((1, size, 1, dim), np.float64)
        for query in range(size):
            first = max(0, query - 1)
            keys = np.concatenate((k[0, first:query + 1, 0], compressed[0, :(query + 1) // rate]))
            vals = np.concatenate((v[0, first:query + 1, 0], compressed[0, :(query + 1) // rate]))
            scores = keys.astype(np.float64) @ q[0, query, 0].astype(np.float64)
            probs = np.exp(scores - scores.max()); probs /= probs.sum()
            oracle[0, query, 0] = probs @ vals.astype(np.float64)
        arrays = dict(q=q, k=k, v=v, compressed_kv=compressed)
        encode = lambda name, a, dtype: dict(name=name, shape=list(a.shape), dtype=dtype, data=base64.b64encode(a.tobytes()).decode())
        node = dict(id="large", opType="SparseAttention", inputs={key: key for key in arrays},
                    outputs=dict(out=dict(tensor="out", dtype="float32", shape=list(oracle.shape))),
                    params=dict(causal=True, scale=1.0, window_size=2, compress_rate=rate, top_k=0))
        fixture = dict(name="SparseAttention/non-indexed-compressed-bank-514", graph=dict(nodes=[node]), weight_shards=[],
                       inputs=[encode(name, a, "F32") for name, a in arrays.items()], expected=[encode("out", oracle, "F64")])
        return self.run(fixture)

    def long_attention_cases(self, dim=64):
        # Exercise a long key reduction independently of any checkpoint. The
        # oracle evaluates the mathematical dot/softmax/AV in F64 from the exact
        # F32 inputs. Include GQA, a sink, absolute ROW offsets and empty masks.
        results = []
        rng = np.random.default_rng(40195)
        batch, queries, keys, heads = 1, 4, 1500, 6
        for name, kv_heads, flat, causal, window, sink in [
                ("flat-MHA", 6, True, False, 0, False),
                ("varying-MHA", 6, False, False, 0, False),
                ("GQA-row-mask-sink", 2, False, True, 128, True)]:
            q = rng.normal(size=(batch, queries, heads, dim)).astype(np.float32) * .3
            k = rng.normal(size=(batch, keys, kv_heads, dim)).astype(np.float32) * .3
            v = rng.normal(size=(batch, keys, kv_heads, dim)).astype(np.float32) * .3
            if flat:
                q.fill(0)
                v += np.float32(16.0)
            mask = np.ones((batch, queries, keys), np.int32)
            if causal:
                mask[:, :, ::7] = 0
                mask[:, -1, :] = 0
                # Masked storage must remain unread even when its values are
                # non-finite; multiplying it by a zero probability is unsafe.
                k[:, ::14] = np.nan
                v[:, ::14] = np.nan
                k[:, 7::14] = np.inf
                v[:, 7::14] = np.inf
            sinks = np.linspace(-.5, .5, heads, dtype=np.float32)
            offset, scale = keys - queries, 1 / math.sqrt(dim)
            oracle = np.zeros((batch, queries, heads, dim), np.float64)
            for b in range(batch):
                for t in range(queries):
                    visible = mask[b, t] != 0
                    if causal: visible &= np.arange(keys) <= t + offset
                    if window: visible &= np.arange(keys) > t + offset - window
                    for h in range(heads):
                        kh = h // (heads // kv_heads)
                        scores = k[b, visible, kh].astype(np.float64) @ q[b, t, h].astype(np.float64) * scale
                        if not len(scores) and not sink: continue
                        maximum = max(float(scores.max()) if len(scores) else -math.inf,
                                      float(sinks[h]) if sink else -math.inf)
                        probabilities = np.exp(scores - maximum)
                        denominator = float(probabilities.sum()) + (math.exp(float(sinks[h]) - maximum) if sink else 0)
                        oracle[b, t, h] = probabilities @ v[b, visible, kh].astype(np.float64) / denominator
            buffers = [self.buffer(x) for x in (q, k, v)]
            zero, out = self.zeros(1), self.zeros(oracle.size)
            sink_buffer = self.buffer(sinks) if sink else zero
            self.stage("llmSparseAttention", buffers + [zero, sink_buffer, self.buffer(mask), zero, out], 8,
                       dict(batch=batch, sequence=queries, keys=keys, heads=heads, kv_heads=kv_heads,
                            key_dim=dim, value_dim=dim, causal=int(causal), window=window,
                            mask_mode=4, flags=8 if sink else 0, scale=scale,
                            pad2=offset, work=batch * queries * heads))
            actual = self.read(out).reshape(oracle.shape)
            np.testing.assert_allclose(actual, oracle, atol=1e-5, rtol=1e-7,
                                       err_msg=name, equal_nan=False)
            for t in range(queries):
                row_out = self.zeros(heads * dim)
                self.stage("llmSparseAttention", [self.buffer(q[:, t:t + 1]), buffers[1], buffers[2],
                           zero, sink_buffer, self.buffer(mask[:, t:t + 1]), zero, row_out], 8,
                           dict(batch=batch, sequence=1, keys=keys, heads=heads, kv_heads=kv_heads,
                                key_dim=dim, value_dim=dim, causal=int(causal), window=window,
                                mask_mode=4, flags=8 if sink else 0, scale=scale,
                                pad2=offset + t, work=batch * heads))
                np.testing.assert_array_equal(self.read(row_out).reshape(batch, heads, dim),
                                              actual[:, t], err_msg=name + "/ROW")
            results.append(dict(name="SparseAttention/long-1500-" + name,
                                operator="SparseAttention", pass_=True,
                                row_equivalence="exact", paired_rows=queries,
                                maximum_error=float(np.max(np.abs(actual - oracle)))))
            self.cleanup_case()
        return results

    def attention_nonfinite_cases(self):
        # Flat finite logits give strictly positive probabilities. The IEEE
        # weighted average of +Inf/finite, -Inf/finite, or NaN/finite is known
        # without copying the shader's reduction. Check both sides of the long
        # path boundary and its first partial key tile.
        results = []
        for keys in (511, 512, 513):
            for name, first, masked in [("positive-inf", np.inf, False),
                                        ("negative-inf", -np.inf, False),
                                        ("nan", np.nan, False),
                                        ("opposite-infinities", np.inf, False),
                                        ("fully-masked-poison", np.nan, True)]:
                q, k = self.zeros(1), self.zeros(keys)
                values = np.full(keys, 3.0, dtype=np.float32)
                values[0] = first
                if name == "opposite-infinities": values[1] = -np.inf
                v, zero, out = self.buffer(values), self.zeros(1), self.zeros(1)
                mask = self.zeros(keys, True) if masked else zero
                self.stage("llmSparseAttention", [q, k, v, zero, zero, mask, zero, out], 8,
                           dict(batch=1, sequence=1, keys=keys, heads=1, kv_heads=1,
                                key_dim=1, value_dim=1, scale=1, mask_mode=1 if masked else 0,
                                work=1))
                expected = 0.0 if masked else np.nan if name == "opposite-infinities" else first
                np.testing.assert_array_equal(self.read(out), np.array([expected], np.float32),
                                              err_msg=f"{name}/keys={keys}")
                results.append(dict(name=f"SparseAttention/nonfinite-{name}-keys={keys}",
                                    operator="SparseAttention", pass_=True,
                                    expected_kind="zero" if masked else "nan" if np.isnan(expected)
                                    else "positive-inf" if expected > 0 else "negative-inf"))
                self.cleanup_case()
        return results

    def gated_cases(self):
        result = []
        rng = np.random.default_rng(1671)
        for input_scale, limit, hidden_width in [(0, 0.0, 5), (1, 0.0, 5), (0, 0.25, 5), (1, 0.25, 5), (0, 0.0, 7168)]:
            rows, d_in, d_out, experts, top_k = 2, 3, 4, 3, 2
            x = rng.normal(size=(rows, d_in)).astype(np.float32) * .2
            gate, up = [rng.normal(size=(experts, d_in, hidden_width)).astype(np.float32) * .2 for _ in range(2)]
            down = rng.normal(size=(experts, hidden_width, d_out)).astype(np.float32) * .2
            ids = np.array([[2, 0], [1, 2]], dtype=np.float32)
            routes = np.array([[.75, .25], [.4, .6]], dtype=np.float32)
            oracle = np.zeros((rows, d_out), np.float64)
            for row in range(rows):
                for slot in range(top_k):
                    e = int(ids[row, slot]); route = float(routes[row, slot]); value = (x[row] * (route if input_scale else 1.0)).astype(np.float32).astype(np.float64)
                    g = value @ gate[e].astype(np.float64); u = value @ up[e].astype(np.float64)
                    if limit: g = np.minimum(g, limit); u = np.clip(u, -limit, limit)
                    h = (g / (1 + np.exp(-g)) * u).astype(np.float32).astype(np.float64)
                    oracle[row] += (1 if input_scale else route) * (h @ down[e].astype(np.float64))
            xb, gb, ub, db, ib, rb = [self.buffer(a) for a in (x, gate, up, down, ids, routes)]
            hb = self.zeros(rows * hidden_width); out = self.zeros(rows * d_out)
            p = dict(rows=rows, d_in=d_in, hidden=hidden_width, d_out=d_out, experts=experts, top_k=top_k, input_scale=input_scale, limit=limit)
            for slot in range(top_k):
                self.stage("moeGatedUp", [xb, gb, ub, ib, rb, hb], 6, dict(p, slot=slot, work=rows * hidden_width))
                self.stage("moeGatedDown", [hb, db, ib, rb, out], 5, dict(p, slot=slot, work=rows * d_out))
            actual = self.read(out).reshape(oracle.shape)
            np.testing.assert_allclose(actual, oracle, rtol=1e-5, atol=1e-5, equal_nan=False)
            result.append(dict(name=f"MoEGatedMLP/input_scale={input_scale}/limit={limit}/hidden={hidden_width}", operator="MoEGatedMLP", pass_=True, maximum_error=float(np.max(np.abs(actual - oracle)))))
            self.cleanup_case()
        return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--power-preference", choices=("low-power", "high-performance"), default="low-power")
    args = parser.parse_args()
    document = json.loads(args.corpus.read_text())
    harness = ShaderHarness(args.power_preference)
    results = []
    status = "FAIL"
    error = None
    try:
        for fixture in document["cases"]:
            record = harness.run(fixture)
            results.append(record)
            print(record["operator"], "PASS", record["maximum_error"], flush=True)
        results.extend(harness.gated_cases())
        results.append(harness.large_compressed_case())
        results.extend(harness.long_attention_cases())
        results.extend(harness.attention_nonfinite_cases())
        status = "PASS"
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        paths = sorted((ROOT / "shaders/inference").glob("llm*.wgsl")) + [ROOT / "shaders/inference" / (name + ".wgsl") for name in ("moeRouterExtended", "moeGatedUp", "moeGatedDown")]
        receipt = dict(status=status, scope="raw WebGPU shaders, independent F64/I32 oracles; not public graph or native Metal qualification", adapter=dict(harness.adapter.info), requested_power_preference=args.power_preference, software_adapter=harness.adapter.info["adapter_type"] == "CPU", wgpu_version=wgpu.__version__, corpus=str(args.corpus), corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(), helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), shader_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, dispatches=harness.dispatches, cases=len(results), results=results, error=error)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
        harness.cleanup_case()
        harness.device.destroy()

if __name__ == "__main__": main()
