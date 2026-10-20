#!/usr/bin/env python3
"""Freeze independent F64 LayerNorm/CrossSDPA cases, then check raw WGSL.

This development-only corpus also runs in actual Chrome using the companion
verify_portable_numeric.mjs. It qualifies the named shader and adapter, not a
public model route, latency, or whole-device memory peak. Nonfinite cases are
reported separately from the required finite-input numerical gate.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TAIL = np.full(17, np.float32(123456.75), dtype="<f4")


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def encoded(value):
    value = np.ascontiguousarray(value)
    return dict(dtype=value.dtype.str, shape=list(value.shape),
                data=base64.b64encode(value.tobytes()).decode())


def decoded(value):
    return np.frombuffer(base64.b64decode(value["data"]), dtype=value["dtype"]).reshape(value["shape"])


def layernorm_reference(x, gamma, beta, epsilon):
    x = x.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        variance = (centered * centered).mean(axis=-1, keepdims=True)
        return centered / np.sqrt(variance + float(np.float32(epsilon))) * gamma.astype(np.float64) + beta.astype(np.float64)


def cross_reference(q, k, v, mask, params, query_rows):
    """Direct F64 math, independent of any shader accumulation order."""
    batch, queries, heads, width = q.shape
    keys, kv_heads = k.shape[1:3]
    result = np.zeros((batch, len(query_rows), heads, width), dtype=np.float64)
    for b in range(batch):
        for ri, row in enumerate(query_rows):
            allowed = np.ones(keys, dtype=bool)
            if params["causal"]:
                allowed &= np.arange(keys) <= row
            mode = params["mask_mode"]
            if mode:
                selected = mask if mode == 1 else mask[b] if mode == 2 else mask[row] if mode == 3 else mask[b, row]
                allowed &= selected != 0
            if not allowed.any():
                continue
            for head in range(heads):
                kv_head = head // (heads // kv_heads)
                scores = k[b, allowed, kv_head].astype(np.float64) @ q[b, row, head].astype(np.float64)
                scores *= float(np.float32(params["scale"]))
                weights = np.exp(scores - scores.max())
                weights /= weights.sum()
                result[b, ri, head] = weights @ v[b, allowed, kv_head].astype(np.float64)
    return result


def read_tensor_rows(path, name, rows=None):
    """Read only selected rows from the independent HF safetensors capture."""
    with Path(path).open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        entry = json.loads(stream.read(length))[name]
        if entry["dtype"] != "F32":
            raise ValueError("HF isolation source must be F32")
        start, end = entry["data_offsets"]
        shape = entry["shape"]
        if rows is None:
            stream.seek(8 + length + start)
            return np.frombuffer(stream.read(end - start), dtype="<f4").copy().reshape(shape)
        width = shape[-1]
        result = []
        for row in rows:
            if row < 0 or row >= math.prod(shape[:-1]):
                raise ValueError("HF isolation row outside capture")
            stream.seek(8 + length + start + row * width * 4)
            result.append(np.frombuffer(stream.read(width * 4), dtype="<f4").copy())
        return np.stack(result)


def layernorm_cases(hf_references=None, hf_weights=None):
    cases = []
    def add(name, x, gamma=None, beta=None, required=True, hf=None):
        x = np.asarray(x, dtype="<f4")
        width = x.shape[-1]
        gamma = np.ones(width, "<f4") if gamma is None else np.asarray(gamma, "<f4")
        beta = np.zeros(width, "<f4") if beta is None else np.asarray(beta, "<f4")
        eps = float(np.float32(1e-5))
        expected = layernorm_reference(x, gamma, beta, eps)
        case = dict(id=name, operator="layernorm", required=required,
                    inputs=[encoded(a) for a in (x, gamma, beta)],
                    uniform=base64.b64encode(struct.pack("<IIfI", x.shape[0], width, eps, 0)).decode(),
                    workgroups=[(x.shape[0] + 63) // 64, 1, 1], output_shape=list(x.shape),
                    expected=encoded(expected), sampled_query_rows=None,
                    atol=1e-5, rtol=1e-5, reference="independent F64 population statistics of exact F32 inputs and epsilon")
        if hf is not None:
            case["hf_expected"] = encoded(np.asarray(hf, "<f4"))
        cases.append(case)
    for width in (1, 3, 7, 383, 384, 385, 513, 1024):
        rng = np.random.default_rng(32000 + width)
        rows = 65 if width == 384 else 3
        add(f"normal-width{width}-rows{rows}", rng.normal(size=(rows, width)),
            rng.uniform(.5, 1.5, width), rng.uniform(-.1, .1, width))
    width = 384
    for mean in (16, 64):
        delta = np.spacing(np.float32(mean)) * np.float32(16)
        x = np.full((3, width), np.float32(mean), "<f4")
        x[:, :width // 2] += delta
        x[:, width // 2:] -= delta
        x[1] = x[1, ::-1]
        x[2, ::2], x[2, 1::2] = np.float32(mean) + delta, np.float32(mean) - delta
        add(f"nearconstant-balanced-width384-mean{mean}", x)
    add("constant-width384", np.full((3, width), 16, "<f4"),
        np.linspace(.5, 1.5, width, dtype="<f4"), np.linspace(-.1, .1, width, dtype="<f4"))
    cancellation = np.tile(np.asarray([1e4, 1, -1e4, -1], "<f4"), (3, width // 4))
    add("mixed-sign-cancellation-width384", cancellation)
    nonfinite = np.ones((3, width), "<f4")
    nonfinite[0, 0], nonfinite[1, 0], nonfinite[2, 0] = np.nan, np.inf, -np.inf
    add("nonfinite-input-classification-diagnostic", nonfinite, required=False)
    gamma = np.ones(width, "<f4"); beta = np.zeros(width, "<f4")
    gamma[0], beta[1], beta[2] = np.nan, np.inf, -np.inf
    add("nonfinite-affine-classification-diagnostic", np.tile(np.linspace(-1, 1, width, dtype="<f4"), (3, 1)),
        gamma, beta, required=False)
    if bool(hf_references) != bool(hf_weights):
        raise ValueError("Both HF references and weights are required")
    if hf_references:
        for layer, row in ((0, 2), (1, 2), (2, 834), (3, 999)):
            prefix = f"model.encoder.layers.{layer}.final_layer_norm"
            add(f"whisper-layer{layer}-finalnorm-row{row}",
                read_tensor_rows(hf_references, f"audio.encoder.{layer}.residual", [row]),
                read_tensor_rows(hf_weights, prefix + ".weight"),
                read_tensor_rows(hf_weights, prefix + ".bias"),
                hf=read_tensor_rows(hf_references, f"audio.encoder.{layer}.mlp.norm", [row]))
    return cases


def cross_cases():
    descriptors = [
        ("short", 1, 3, 31, 6, 6, 64, 0, False, "varying"),
        *[(f"boundary{keys}-{kind}", 1, 65, keys, 6, 6, 64, 0, False, kind)
          for keys in (511, 512) for kind in ("varying", "flat-positive")],
        ("whisper-long1500", 1, 3, 1500, 6, 6, 64, 0, False, "varying"),
        ("gqa-batch2", 2, 65, 33, 6, 3, 64, 0, False, "varying"),
        ("gqa-dim128", 2, 3, 513, 6, 3, 128, 0, False, "varying"),
        *[(f"maskmode{mode}", 2, 3, 512, 6, 3, 64, mode, False, "varying") for mode in range(1, 5)],
        *[(f"allmasked{keys}", 2, 3, keys, 6, 3, 64, 4, False, "allmasked") for keys in (511, 512)],
        ("causal-mask-gqa", 2, 65, 512, 6, 3, 7, 4, True, "varying"),
        ("finite-logits-below-minus1e38", 1, 3, 512, 2, 1, 64, 0, False, "very-negative"),
        *[(f"boundary{keys}-flat-positive", 1, 65, keys, 6, 6, 64, 0, False, "flat-positive")
          for keys in (255, 256)],
    ]
    cases = []
    for ci, (name, batch, queries, keys, heads, kvheads, width, mode, causal, kind) in enumerate(descriptors):
        rng = np.random.default_rng(65000 + ci)
        q = (rng.normal(size=(batch, queries, heads, width)).astype("<f4") * np.float32(.3))
        k, v = [(rng.normal(size=(batch, keys, kvheads, width)).astype("<f4") * np.float32(.3)) for _ in range(2)]
        scale = float(np.float32(1 / math.sqrt(width)))
        if kind == "flat-positive":
            q.fill(0); v += np.float32(16)
        elif kind == "very-negative":
            q.fill(0); k.fill(0)
            q[..., 0] = np.float32(-1e19); k[..., 0] = np.float32(2e19); scale = 1.0
        mask_shape = {0: (1,), 1: (keys,), 2: (batch, keys), 3: (queries, keys), 4: (batch, queries, keys)}[mode]
        mask = np.ones(mask_shape, "<i4")
        if mode:
            # Distinct patterns expose batch/query mask indexing mistakes.
            mask.flat[::3] = 0
            if mode in (2, 4):
                mask[1, ...] = 1 - mask[1, ...]
        if kind == "allmasked":
            mask.fill(0)
        params = dict(scale=scale, causal=causal, mask_mode=mode)
        rows = sorted(set((0, queries // 2, queries - 1)))
        expected = cross_reference(q, k, v, mask, params, rows)
        uniform = struct.pack("<IIIIIIfIIIII", queries, keys, heads * width, heads, width, batch,
                              scale, int(causal), mode, kvheads, 0, 0)
        cases.append(dict(id=name, operator="cross", required=True,
                          inputs=[encoded(a) for a in (q, k, v, mask)],
                          uniform=base64.b64encode(uniform).decode(), workgroups=[(queries + 63) // 64, heads, batch],
                          output_shape=list(q.shape), expected=encoded(expected), sampled_query_rows=rows,
                          atol=1e-5, rtol=1e-7, parameters=params,
                          reference="independent F64 masked softmax, GQA addressing, exact F32 inputs and scale"))
    return cases


def compare(actual, expected, atol, rtol):
    finite = np.isfinite(expected)
    classification = bool(np.array_equal(np.isnan(actual), np.isnan(expected)) and
                          np.array_equal(np.isposinf(actual), np.isposinf(expected)) and
                          np.array_equal(np.isneginf(actual), np.isneginf(expected)))
    errors = np.abs(actual.astype(np.float64)[finite] - expected[finite])
    finite_pass = bool(np.all(np.isfinite(actual[finite])) and
                       np.all(errors <= atol + rtol * np.abs(expected[finite])))
    return dict(pass_=classification and finite_pass, classification_pass=classification,
                finite_pass=finite_pass, finite_elements=int(finite.sum()),
                maximum_absolute_error=float(errors.max()) if errors.size and np.all(np.isfinite(errors)) else None)


def prepare(args):
    cases = layernorm_cases(args.hf_references, args.hf_weights) if args.operator == "layernorm" else cross_cases()
    corpus = dict(format="volvox-portable-numeric/v1", status="PREPARED", operator=args.operator,
                  helper=identity(__file__), oracle="frozen independent NumPy F64 mathematical reference",
                  physical_peak_qualified=False, model_accuracy_qualified=False, cases=cases,
                  output_tail=encoded(TAIL), nonfinite_policy="diagnostic classifications, excluded from finite-input qualification")
    if args.hf_references:
        corpus["independent_hf_sources"] = [identity(args.hf_references), identity(args.hf_weights)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(corpus, stream, indent=2, allow_nan=False); stream.write("\n")


def run(args):
    import wgpu
    corpus = json.loads(args.corpus.read_text())
    source = args.shader.read_text()
    receipt = dict(format="volvox-portable-numeric-observation/v1", status="FAIL", host=platform.node(),
                   runtime="wgpu-native", corpus=identity(args.corpus), shader=identity(args.shader),
                   helper=identity(__file__), cases=[], physical_peak_qualified=False, model_accuracy_qualified=False)
    device = None
    owned = []
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("No actual WebGPU adapter")
        receipt["adapter"] = dict(adapter.info)
        if args.hardware and adapter.info.get("adapter_type") == "CPU":
            raise RuntimeError("Hardware mode refuses software/CPU adapter")
        device = adapter.request_device_sync()
        module = device.create_shader_module(code=source)
        pipeline = device.create_compute_pipeline(layout="auto", compute=dict(module=module, entry_point="main"))
        for case in corpus["cases"]:
            usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
            def own(buffer):
                owned.append(buffer); return buffer
            buffers = [own(device.create_buffer_with_data(data=decoded(value), usage=usage)) for value in case["inputs"]]
            count = math.prod(case["output_shape"])
            initial = np.concatenate((np.full(count, np.float32(-98765.5)), TAIL))
            output = own(device.create_buffer_with_data(data=initial, usage=usage)); buffers.append(output)
            uniform = own(device.create_buffer_with_data(data=base64.b64decode(case["uniform"]), usage=wgpu.BufferUsage.UNIFORM))
            buffers.append(uniform)
            group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0),
                    entries=[dict(binding=i, resource=dict(buffer=b)) for i, b in enumerate(buffers)])
            encoder = device.create_command_encoder(); compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipeline); compute.set_bind_group(0, group)
            compute.dispatch_workgroups(*case["workgroups"]); compute.end()
            device.queue.submit([encoder.finish()]); device.queue.on_submitted_work_done_sync()
            complete = np.frombuffer(device.queue.read_buffer(output), dtype="<f4").copy()
            actual = complete[:count].reshape(case["output_shape"])
            if case["sampled_query_rows"] is not None:
                actual = actual[:, case["sampled_query_rows"]]
            check = compare(actual, decoded(case["expected"]), case["atol"], case["rtol"])
            check["pass"] = check.pop("pass_")
            guard = bool(np.array_equal(complete[count:].view("<u4"), TAIL.view("<u4")))
            check["pass"] &= guard
            entry = dict(id=case["id"], required=case["required"], guard_pass=guard,
                         output_sha256=hashlib.sha256(complete[:count].tobytes()).hexdigest(), **check)
            if "hf_expected" in case:
                entry["independent_hf"] = compare(actual, decoded(case["hf_expected"]), case["atol"], case["rtol"])
            receipt["cases"].append(entry)
            for buffer in owned:
                buffer.destroy()
            owned.clear()
        receipt["status"] = "PASS" if all(c["guard_pass"] and (not c["required"] or c["pass"])
                                         for c in receipt["cases"]) else "FAIL"
        receipt["nonfinite_diagnostics_pass"] = all(c["pass"] for c in receipt["cases"] if not c["required"])
    except Exception as error:
        receipt["error"] = repr(error)
    finally:
        for buffer in owned:
            buffer.destroy()
        if device:
            device.destroy()
        receipt["shader_after"] = identity(args.shader)
        if receipt["shader_after"] != receipt["shader"]:
            receipt["status"] = "FAIL"; receipt["error"] = "Shader source changed during worker"
        with args.out.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False); stream.write("\n")
    if receipt["status"] != "PASS":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--operator", choices=("layernorm", "cross"))
    parser.add_argument("--hf-references", type=Path); parser.add_argument("--hf-weights", type=Path)
    parser.add_argument("--corpus", type=Path); parser.add_argument("--shader", type=Path)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.prepare:
        if not args.operator:
            parser.error("Preparation requires --operator")
        prepare(args)
    else:
        if not args.corpus or not args.shader:
            parser.error("Execution requires --corpus and --shader")
        run(args)


if __name__ == "__main__":
    main()
