#!/usr/bin/env python3
"""Numerical and CUDA-event probe of the production private attention PTX.

The oracle uses quantized F32 input values, Python F64 arithmetic and libm;
there is no NumPy, engine fallback or downloaded model. Parallel mode uses
one 128-thread block per query/head; serial mode uses the preserved scalar
kernel's 256-thread launch. This qualifies private kernels, not public model
admission or end-to-end latency.
"""
from __future__ import annotations

import argparse
import array
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import time

from probe_cuda_sampling import Driver, digest, prepare_arguments

if sys.byteorder != "little":
    raise RuntimeError("this frozen fixture format requires a little-endian host")

SOURCE_FILES = (
    "native/src/backends/cuda/kernels/cuda_forward_argument_storage.inc",
    "native/src/backends/cuda/attention_launch_plan.h",
    "tools/llm_baseline/probe_cuda_sampling.py",
    "native/src/backends/cuda/kernels/cuda_forward_sdpa_kernels.inc",
    "native/src/backends/cuda/kernels/cuda_forward_cross_attention_kernels.inc",
    "native/src/backends/cuda/host/cuda_graph_conv_attention_host.inc",
    "native/src/backends/cuda/host/cuda_forward_function_registry_host.inc",
    "native/src/backends/cuda_kernels.cu",
)
KERNELS = {
    "mha": ("vx_cuda_cross_sdpa_f32", ["ptr"] * 5 + ["u32"] * 6 + ["f32", "u32", "u32", "ptr"]),
    "gqa": ("vx_cuda_cross_sdpa_gqa_f32", ["ptr"] * 5 + ["u32"] * 8 + ["f32", "u32", "u32", "ptr"]),
    "self": ("vx_cuda_sdpa_f32", ["ptr"] * 3 + ["u32"] * 5 + ["f32"] + ["u32"] * 4),
}
SENTINEL = -98765.5
GUARD = 8


def geometry(label, route, batch, queries, keys, heads, kv_heads, width,
             mask_mode=0, causal=False, all_masked=False, reverse=False,
             query_start=0, query_count=None, all_allowed=False, cancellation=False):
    return dict(label=label, route=route, batch=batch, queries=queries, keys=keys,
                heads=heads, kv_heads=kv_heads, width=width, mask_mode=mask_mode,
                causal=causal, all_masked=all_masked, reverse=reverse,
                query_start=query_start, query_count=queries if query_count is None else query_count,
                all_allowed=all_allowed, cancellation=cancellation,
                scale=struct.unpack("<f", struct.pack("<f", -.43 if cancellation else .73))[0])


def geometries():
    cases = [
        geometry("MHA-Whisper-long", "mha", 2, 2, 1500, 2, 2, 64),
        geometry("MHA-long-tail", "mha", 1, 3, 1537, 2, 2, 37, 1),
        geometry("MHA-batch-mask", "mha", 2, 2, 513, 2, 2, 40, 2),
        geometry("MHA-query-mask", "mha", 1, 3, 1537, 2, 2, 64, 3),
        geometry("GQA-batch-query-mask", "gqa", 2, 3, 1537, 4, 2, 80, 4),
        geometry("GQA-causal-tail", "gqa", 2, 7, 513, 4, 2, 37, 4, True),
        geometry("MHA-all-masked", "mha", 2, 2, 1500, 2, 2, 40, 1, all_masked=True),
        *[geometry(f"MHA-boundary-{key}", "mha", 2, 3, key, 2, 2, 37, 4) for key in (47, 48, 49)],
        geometry("GQA-wide-short", "gqa", 2, 3, 31, 4, 2, 128, 4),
        geometry("GQA-wide-long", "gqa", 1, 2, 97, 4, 1, 128, 1),
        geometry("GQA-one-query", "gqa", 2, 1, 49, 4, 1, 128, 2),
        geometry("Qwen-shaped-cached-prefill16", "gqa", 1, 16, 16, 16, 8, 128, 1, True, all_allowed=True),
        geometry("Qwen-shaped-cached-row16", "gqa", 1, 1, 16, 16, 8, 128, 1, all_allowed=True),
        geometry("SDPA-range-causal-row49", "self", 2, 65, 65, 2, 2, 37, 4, True, query_start=49, query_count=2),
        geometry("SDPA-all-masked", "self", 1, 49, 49, 2, 2, 37, 1, all_masked=True),
        geometry("SDPA-channel-stride257", "self", 2, 3, 3, 2, 2, 257, 4),
        *[geometry(f"Smol-shaped-row-{keys}", "gqa", 1, 1, keys, 9, 3, 64)
          for keys in (33, 34, 35)],
        *[geometry(f"Qwen-shaped-row-{keys}", "gqa", 1, 1, keys, 16, 8, 128)
          for keys in (33, 34, 35)],
        *[geometry(f"GQA-channel-boundary-{width}", "gqa", 2, 3, 48, 4, 2, width, 4)
          for width in (31, 32, 127, 128, 129)],
        geometry("GQA-short-all-masked", "gqa", 2, 1, 34, 4, 2, 128, 2, all_masked=True),
        geometry("MHA-short-cancellation", "mha", 2, 3, 35, 2, 2, 127, 4, cancellation=True),
        geometry("GQA-short-cancellation", "gqa", 2, 3, 48, 4, 2, 128, 4, cancellation=True),
        *[geometry(f"GQA-subgroup-channel-{width}", "gqa", 2, 3, 35, 4, 2, width, 4)
          for width in (7, 8, 9, 63, 64, 65)],
        *[geometry(f"GQA-key-group-{keys}", "gqa", 2, 3, keys, 4, 2, 64, 4)
          for keys in (15, 16, 17)],
        geometry("GQA-short64-all-masked", "gqa", 2, 1, 35, 4, 2, 64, 2, all_masked=True),
        geometry("GQA-causal-short-prefill36", "gqa", 1, 36, 256, 4, 2, 64, 4, True),
    ]
    for index, case in enumerate(cases):
        case["seed"] = 710913 + index * 7919
    # Reversing K/V and the corresponding mask preserves the noncausal
    # mathematical result and catches key addressing or reduction-order bugs.
    for index in (0, 4):
        case = dict(cases[index]); case["reverse"] = True
        case["label"] += "-reversed"
        cases.append(case)
    # Append after the reversals to preserve all existing fixture seeds.
    for label, keys, width, all_masked in (
            ("GQA-short-masked-value-poison", 34, 128, False),
            ("GQA-short-all-masked-value-poison", 48, 64, True)):
        case = geometry(label, "gqa", 2, 1, keys, 4, 2, width, 2,
                        all_masked=all_masked)
        case.update(seed=710913 + len(cases) * 7919, poison_masked_values=True)
        cases.append(case)
    case = geometry("GQA-narrow-cancellation", "gqa", 2, 3, 35, 4, 2, 64, 4,
                    cancellation=True)
    case["seed"] = 710913 + len(cases) * 7919
    cases.append(case)
    case = geometry("GQA-narrow-mask-poison", "gqa", 2, 1, 34, 4, 2, 64, 2)
    case.update(seed=710913 + len(cases) * 7919, poison_masked_values=True)
    cases.append(case)
    # Append bounded score-cache and streaming-fallback cases, preserving seeds.
    for keys in (47, 48, 49, 511, 512, 1536, 2048, 2049):
        for width in (32, 64, 128):
            case = geometry(f"cache-boundary-{keys}-d{width}", "gqa", 1, 1,
                            keys, 4, 1, width, 1, all_allowed=True)
            case["seed"] = 103603000 + len(cases) * 17
            cases.append(case)
    case = geometry("TinyLlama-prefill31", "gqa", 1, 31, 31, 32, 4, 64,
                    1, True, all_allowed=True)
    case["seed"] = 103603000 + len(cases) * 17
    cases.append(case)
    for keys in (512, 1536):
        case = geometry(f"TinyLlama-decode-{keys}", "gqa", 1, 1, keys,
                        32, 4, 64, 1, all_allowed=True)
        case["seed"] = 103603000 + len(cases) * 17
        cases.append(case)
    for all_masked in (False, True):
        case = geometry(f"cache-long-poison-{all_masked}", "gqa", 1, 1,
                        2048, 4, 1, 64, 2, all_masked=all_masked)
        case.update(seed=103603000 + len(cases) * 17, poison_masked_values=True)
        cases.append(case)
    for keys, width in ((49, 37), (512, 64), (1536, 64), (2048, 128), (2049, 64)):
        case = geometry(f"long-cancellation-{keys}-d{width}", "gqa", 1, 2,
                        keys, 4, 1, width, 4, cancellation=True)
        case["seed"] = 103604000 + len(cases) * 17
        cases.append(case)
    for keys in (512, 1536):
        case = geometry(f"TinyLlama-throughput8-{keys}", "gqa", 1, 8, keys,
                        32, 4, 64, 1, all_allowed=True)
        case["seed"] = 103604000 + len(cases) * 17
        cases.append(case)
    return cases


def fixture(g):
    state = g["seed"]
    def sample():
        nonlocal state
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        return ((state >> 8) / 16777216 - .5) * .3
    batches, queries, keys = g["batch"], g["queries"], g["keys"]
    heads, kv_heads, width = g["heads"], g["kv_heads"], g["width"]
    q = array.array("f", (sample() for _ in range(batches * queries * heads * width)))
    if g["cancellation"]:
        q[:] = array.array("f", ((-1 if index % width & 1 else 1) * (1 + value)
            for index, value in enumerate(q)))
    k, v = array.array("f"), array.array("f")
    for batch in range(batches):
        for key in range(keys):
            for head in range(kv_heads):
                for channel in range(width):
                    k.append((1 if g["cancellation"] else 0) + sample())
                    v.append((0 if g["cancellation"] else 100) + .25 * batch + .125 * head
                        + .01337 * (channel % 11) + .001 * (key % 7 - 3))
    mode = g["mask_mode"]
    mask_rows = batches if mode == 2 else queries if mode == 3 else batches * queries if mode == 4 else 1
    mask = array.array("i", (int(not g["all_masked"] and
        (g["all_allowed"] or ((mode <= 1 or row % queries != 1) and key % 7 != 0)))
        for row in range(mask_rows) for key in range(keys)))
    if g["reverse"]:
        columns = kv_heads * width
        for data in (k, v):
            data[:] = array.array("f", (data[(batch * keys + key) * columns + channel]
                for batch in range(batches) for key in range(keys - 1, -1, -1) for channel in range(columns)))
        mask[:] = array.array("i", (mask[row * keys + key] for row in range(mask_rows) for key in range(keys - 1, -1, -1)))
    if g.get("poison_masked_values", False):
        if g["queries"] != 1 or g["mask_mode"] != 2 or g["route"] != "gqa":
            raise ValueError("masked value poison requires one-query batch-masked GQA")
        poison = struct.unpack("<f", struct.pack("<I", 0x7fc01234))[0]
        columns = kv_heads * width
        for batch in range(batches):
            for key in range(keys):
                if not allowed(g, mask, batch, 0, key):
                    start = (batch * keys + key) * columns
                    v[start:start + columns] = array.array("f", [poison] * columns)
    buffers = {"query": q.tobytes(), "key": k.tobytes(), "value": v.tobytes(), "mask": mask.tobytes()}
    if g["route"] == "self":
        model_width = heads * width
        qkv = array.array("f")
        for batch in range(batches):
            for row in range(queries):
                start = (batch * queries + row) * model_width
                qkv.extend(q[start:start + model_width]); qkv.extend(k[start:start + model_width]); qkv.extend(v[start:start + model_width])
        buffers["qkv"] = qkv.tobytes()
    return q, k, v, mask, buffers


def allowed(g, mask, batch, query, key):
    if g["causal"] and key > query:
        return False
    mode, keys, queries = g["mask_mode"], g["keys"], g["queries"]
    if mode == 0:
        return True
    row = batch if mode == 2 else query if mode == 3 else batch * queries + query if mode == 4 else 0
    return bool(mask[row * keys + key])


def reference(g, q, k, v, mask):
    width, heads, kv_heads = g["width"], g["heads"], g["kv_heads"]
    query_width, kv_width = heads * width, kv_heads * width
    output = [SENTINEL] * len(q)
    start = g["query_start"] if g["route"] == "self" else 0
    count = g["query_count"] if g["route"] == "self" else g["queries"]
    for batch in range(g["batch"]):
        for query in range(start, start + count):
            active_keys = [key for key in range(g["keys"]) if allowed(g, mask, batch, query, key)]
            for head in range(heads):
                query_base = (batch * g["queries"] + query) * query_width + head * width
                kv_head = head // (heads // kv_heads)
                bases = [(batch * g["keys"] + key) * kv_width + kv_head * width for key in active_keys]
                if not bases:
                    output[query_base:query_base + width] = [0.] * width
                    continue
                scores = []
                for key_base in bases:
                    dot = 0.
                    for channel in range(width):
                        dot += float(q[query_base + channel]) * float(k[key_base + channel])
                    scores.append(dot * g["scale"])
                maximum = max(scores)
                weights = [math.exp(score - maximum) for score in scores]
                denominator = 0.
                for weight in weights:
                    denominator += weight
                for channel in range(width):
                    numerator = 0.
                    for key_base, weight in zip(bases, weights):
                        numerator += weight * float(v[key_base + channel])
                    output[query_base + channel] = numerator / denominator
    return output


def launch(g, mode):
    count = g["query_count"] if g["route"] == "self" else g["queries"]
    tasks = g["batch"] * count * g["heads"]
    if mode == "parallel":
        x = min(tasks, 65535)
        grid, block = [x, (tasks + x - 1) // x, 1], [128, 1, 1]
        # Mirror the production cross-attention output-channel tiling.
        if (g["route"] != "self" and g["queries"] == 1 and
                g["keys"] <= 48 and g["width"] <= 128):
            grid[2] = (g["width"] + 31) // 32
    else:
        grid, block = [(tasks + 255) // 256, 1, 1], [256, 1, 1]
    return {"grid": grid, "block": block, "shared_bytes": 0, "tasks": tasks}


def prepare(args):
    source_root = args.source_root.resolve()
    configs = []
    for g in geometries():
        q, k, v, mask, buffers = fixture(g)
        started = time.perf_counter_ns()
        oracle = reference(g, q, k, v, mask)
        configs.append({"geometry": g, "kernel": KERNELS[g["route"]][0],
            "kernel_abi": KERNELS[g["route"]][1], "launch": launch(g, args.mode),
            "input_bytes": {name: len(data) for name, data in buffers.items()},
            "input_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in buffers.items()},
            "reference_elements": len(oracle),
            "reference_f64_sha256": hashlib.sha256(array.array("d", oracle).tobytes()).hexdigest(),
            "reference_wall_ms": (time.perf_counter_ns() - started) / 1e6})
    return {"format": "volvoxai-private-cuda-attention-diagnostic-v1", "status": "REFERENCE_READY", "mode": args.mode,
        "provenance": {"ptx_path": str(args.ptx.resolve()), "ptx_sha256": digest(args.ptx),
            "library_path": str(args.library.resolve()), "library_sha256": digest(args.library),
            "helper_sha256": digest(__file__), "driver_helper_sha256": digest(Path(__file__).with_name("probe_cuda_sampling.py")),
            "sources_sha256": {name: digest(source_root / name) for name in SOURCE_FILES}},
        "configs": configs, "numerical_contract": {"max_abs": 1e-4, "normalized_rms": 1e-5,
            "normalization": "RMS(error)/max(RMS(reference), 1e-12), selected rows only", "exact_zero_when_fully_masked": True,
            "direct_stored_bit_identity": "parallel MHA/GQA; packed self and legacy serial use direct arguments",
            "repeat_bit_identity": "every warm and measured output equals the admitted direct output"},
        "timing_contract": "untimed numerical admission precedes one warm and three measured launches per fixture; CUDA events on a nonblocking stream; host uploads and output resets are context-fenced before each start event; copies/fence/JIT/allocation/validation excluded",
        "scope": "private exact production PTX attention kernels; synthetic geometry; no public admission or public Generate latency claim"}


def validate(g, actual, oracle):
    failures, errors, expected = [], [], []
    width = g["heads"] * g["width"]
    start = g["query_start"] if g["route"] == "self" else 0
    count = g["query_count"] if g["route"] == "self" else g["queries"]
    if any(value != SENTINEL for value in actual[:GUARD] + actual[-GUARD:]):
        failures.append("output guard changed")
    for index, value in enumerate(actual[GUARD:-GUARD]):
        row = (index // width) % g["queries"]
        if row < start or row >= start + count:
            if value != SENTINEL:
                failures.append(f"unselected row modified at {index}")
            continue
        reference_value = oracle[index]
        if not math.isfinite(value):
            failures.append(f"nonfinite output at {index}")
        if reference_value == 0. and value != 0.:
            failures.append(f"fully masked output not exact zero at {index}")
        errors.append(value - reference_value); expected.append(reference_value)
    max_abs = max(map(abs, errors), default=0.)
    denominator = max(math.sqrt(math.fsum(x * x for x in expected) / len(expected)), 1e-12)
    normalized_rms = math.sqrt(math.fsum(x * x for x in errors) / len(errors)) / denominator
    if max_abs > 1e-4 or normalized_rms > 1e-5:
        failures.append("numerical bound exceeded")
    return {"pass": not failures, "max_abs": max_abs, "normalized_rms": normalized_rms,
            "selected_elements": len(expected), "failures": failures[:8]}


def run(args, receipt):
    driver = Driver()
    context, stream, module, start, end = (C.c_void_p() for _ in range(5))
    allocations = []
    device, version = C.c_int(), C.c_int()
    driver.call("Init", 0); driver.call("DeviceGet", C.byref(device), 0)
    name = C.create_string_buffer(256)
    driver.call("DeviceGetName", name, len(name), device); driver.call("DriverGetVersion", C.byref(version))
    receipt["device"] = {"ordinal": 0, "name": name.value.decode(), "driver_api_version": version.value}
    try:
        driver.call("CtxCreate", C.byref(context), 0, device); driver.call("StreamCreate", C.byref(stream), 1)
        ptx = C.create_string_buffer(args.ptx.read_bytes())
        started = time.perf_counter_ns(); driver.call("ModuleLoadData", C.byref(module), ptx)
        receipt["excluded_module_load_jit_wall_ms"] = (time.perf_counter_ns() - started) / 1e6
        functions = {}
        for route, (symbol, _) in KERNELS.items():
            functions[route] = C.c_void_p()
            driver.call("ModuleGetFunction", C.byref(functions[route]), module, symbol.encode())
        driver.call("EventCreate", C.byref(start), 0); driver.call("EventCreate", C.byref(end), 0)
        for config in receipt["configs"]:
            g = config["geometry"]
            q, k, v, mask, buffers = fixture(g)
            oracle = reference(g, q, k, v, mask)
            if hashlib.sha256(array.array("d", oracle).tobytes()).hexdigest() != config["reference_f64_sha256"]:
                raise AssertionError("reference changed after preparation")
            pointers, fixture_allocations = {}, []
            upload_names = ("qkv", "mask") if g["route"] == "self" else ("query", "key", "value", "mask")
            for label in upload_names:
                data = buffers[label]
                if hashlib.sha256(data).hexdigest() != config["input_sha256"][label]:
                    raise AssertionError("input changed after preparation")
                pointer = C.c_uint64(); driver.call("MemAlloc", C.byref(pointer), len(data))
                allocations.append(pointer); fixture_allocations.append(pointer); pointers[label] = pointer
                packed = C.create_string_buffer(data)
                driver.call("MemcpyHtoD", pointer, packed, len(data))
            output_host = (C.c_float * (len(q) + 2 * GUARD))(*([SENTINEL] * (len(q) + 2 * GUARD)))
            output = C.c_uint64(); driver.call("MemAlloc", C.byref(output), C.sizeof(output_host))
            allocations.append(output); fixture_allocations.append(output)
            output_view = C.c_uint64(output.value + GUARD * 4)
            mask_pointer = pointers["mask"] if g["mask_mode"] else C.c_uint64(0)
            ints = lambda values: [C.c_uint(value) for value in values]
            if g["route"] == "self":
                scalars = [pointers["qkv"], mask_pointer, output_view] + ints(
                    [g["queries"], g["heads"] * g["width"], g["heads"], g["width"], g["batch"]]) + [C.c_float(g["scale"])] + ints(
                    [g["causal"], g["mask_mode"], g["query_start"], g["query_count"]])
            else:
                scalars = [pointers["query"], pointers["key"], pointers["value"], mask_pointer, output_view]
                dimensions = [g["queries"], g["keys"], g["heads"] * g["width"]]
                if g["route"] == "gqa":
                    dimensions += [g["kv_heads"] * g["width"], g["heads"], g["kv_heads"], g["width"], g["batch"]]
                else:
                    dimensions += [g["heads"], g["width"], g["batch"]]
                scalars += ints(dimensions) + [C.c_float(g["scale"])] + ints([g["causal"], g["mask_mode"]])
            owned_arguments, params = prepare_arguments(driver, scalars, allocations,
                supported=g["route"] != "self")
            argument_routes = [("direct", owned_arguments, params)]
            if args.mode == "parallel" and g["route"] != "self":
                allocation_start = len(allocations)
                stored_arguments, stored_params = prepare_arguments(driver, scalars, allocations, stored=True)
                fixture_allocations.extend(allocations[allocation_start:])
                argument_routes.append(("stored", stored_arguments, stored_params))
            admitted_hash = None
            config["argument_numerical_observations"] = []
            launch_config = config["launch"]
            for storage, owned, arguments in argument_routes:
                driver.call("MemcpyHtoD", output, output_host, C.sizeof(output_host))
                driver.call("CtxSynchronize")
                driver.call("LaunchKernel", functions[g["route"]], *launch_config["grid"], *launch_config["block"], 0, stream, arguments, None)
                driver.call("CtxSynchronize")
                actual = (C.c_float * len(output_host))(); driver.call("MemcpyDtoH", actual, output, C.sizeof(actual))
                numerical = validate(g, list(actual), oracle)
                output_hash = hashlib.sha256(bytes(actual)).hexdigest()
                config["argument_numerical_observations"].append({"argument_storage": storage,
                    "output_f32_sha256": output_hash, **numerical})
                if not numerical["pass"]:
                    raise AssertionError(f"attention mismatch {g['label']} {storage}: {numerical}")
                if admitted_hash is not None and output_hash != admitted_hash:
                    raise AssertionError(f"direct/stored output changed: {g['label']}")
                admitted_hash = output_hash
            observations = []
            for repetition in range(4):
                driver.call("MemcpyHtoD", output, output_host, C.sizeof(output_host))
                # Fence the output reset and fixture uploads before the
                # nonblocking stream records its start event; timing excludes it.
                driver.call("CtxSynchronize")
                driver.call("EventRecord", start, stream)
                driver.call("LaunchKernel", functions[g["route"]], *launch_config["grid"], *launch_config["block"], 0, stream, params, None)
                driver.call("EventRecord", end, stream); driver.call("EventSynchronize", end)
                elapsed = C.c_float(); driver.call("EventElapsedTime", C.byref(elapsed), start, end)
                actual = (C.c_float * len(output_host))(); driver.call("MemcpyDtoH", actual, output, C.sizeof(actual))
                numerical = validate(g, list(actual), oracle)
                observation = {"kind": "warm" if repetition == 0 else "measured", "repetition": repetition,
                    "gpu_event_ms": elapsed.value, "output_f32_sha256": hashlib.sha256(bytes(actual)).hexdigest(), **numerical}
                observations.append(observation); config["observations"] = observations
                if not numerical["pass"]:
                    raise AssertionError(f"attention mismatch {g['label']}: {numerical}")
                if observation["output_f32_sha256"] != admitted_hash:
                    raise AssertionError(f"warm/measured output changed: {g['label']}")
            config["measured_median_gpu_ms"] = statistics.median(item["gpu_event_ms"] for item in observations[1:])
            for pointer in reversed(fixture_allocations):
                driver.call("MemFree", pointer); allocations.remove(pointer)
        provenance = receipt["provenance"]
        if digest(args.ptx) != provenance["ptx_sha256"] or digest(args.library) != provenance["library_sha256"]:
            raise AssertionError("PTX or library changed during measurement")
        for name, before in provenance["sources_sha256"].items():
            if digest(args.source_root / name) != before:
                raise AssertionError(f"source changed during measurement: {name}")
        if digest(__file__) != provenance["helper_sha256"] or digest(Path(__file__).with_name("probe_cuda_sampling.py")) != provenance["driver_helper_sha256"]:
            raise AssertionError("probe changed during measurement")
        receipt["provenance_unchanged_after_run"] = True; receipt["status"] = "PASS"
    finally:
        for pointer in reversed(allocations): driver.call("MemFree", pointer)
        if end.value: driver.call("EventDestroy", end)
        if start.value: driver.call("EventDestroy", start)
        if module.value: driver.call("ModuleUnload", module)
        if stream.value: driver.call("StreamDestroy", stream)
        if context.value: driver.call("CtxDestroy", context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptx", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("parallel", "serial"), default="parallel")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.out.exists(): parser.error("refusing to overwrite a diagnostic receipt")
    receipt = prepare(args)
    if args.reference:
        before = json.loads(args.reference.read_text())
        for field in ("format", "mode", "numerical_contract", "timing_contract", "scope"):
            if before[field] != receipt[field]: raise ValueError(f"frozen {field} changed")
        for field in ("ptx_sha256", "library_sha256", "helper_sha256", "driver_helper_sha256", "sources_sha256"):
            if before["provenance"][field] != receipt["provenance"][field]: raise ValueError(f"frozen {field} changed")
        for old, current in zip(before["configs"], receipt["configs"], strict=True):
            old = dict(old); current = dict(current)
            old.pop("reference_wall_ms"); current.pop("reference_wall_ms")
            if old != current: raise ValueError("frozen attention fixture or reference changed")
        receipt["prepared_reference_sha256"] = digest(args.reference)
    try:
        if not args.prepare_only: run(args, receipt)
    except Exception as error:
        receipt["status"] = "FAIL"; receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"status": receipt["status"], "out": str(args.out), "configs": len(receipt["configs"]),
            "ptx_sha256": receipt["provenance"]["ptx_sha256"],
            "timings": [{"label": config["geometry"]["label"], "median_ms": config.get("measured_median_gpu_ms")}
                        for config in receipt["configs"]]}, indent=2))


if __name__ == "__main__":
    main()
