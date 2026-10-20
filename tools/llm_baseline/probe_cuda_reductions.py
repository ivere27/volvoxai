#!/usr/bin/env python3
"""Check exact production reduction PTX against F64 and strict ArgMax rules.

Uses only the CUDA driver and Python standard library. Synthetic fixtures,
guards, repeated output hashes and scalar/cooperative event samples are kept
in one receipt. Events exclude JIT, copies and readback; no model decode or
public API performance is inferred from these isolated kernels.
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
import sys
import time

from probe_cuda_sampling import Driver, digest, prepare_arguments

GUARD, SENTINEL, ID_SENTINEL = 8, -98765.5, -1234567
SOURCE_FILES = (
    "native/src/backends/cuda/kernels/cuda_forward_argument_storage.inc",
    "tools/llm_baseline/probe_cuda_sampling.py",
    "native/src/backends/cuda/kernels/cuda_forward_elementwise_norm_kernels.inc",
    "native/src/backends/cuda/kernels/cuda_forward_typed_graph_kernels.inc",
    "native/src/backends/cuda/host/cuda_graph_f32_basic_host.inc",
    "native/src/backends/cuda/host/cuda_graph_typed_graph_host.inc",
    "native/src/backends/cuda/host/cuda_forward_function_registry_host.inc",
    "native/src/backends/cuda_kernels.cu",
)


def geometries():
    cases = []
    widths = (1, 2, 15, 31, 32, 33, 64, 65, 127, 128, 129, 255, 256,
              257, 576, 1024, 1025, 4097)
    for width in widths:
        for inplace in (False, True):
            cases.append(dict(op="rmsnorm", rows=3, width=width, recipe="random", inplace=inplace))
    for recipe in ("dynamic", "signed-zero", "nan", "infinity"):
        for inplace in (False, True):
            cases.append(dict(op="rmsnorm", rows=2, width=576, recipe=recipe, inplace=inplace))
    for rows, width in ((1, 576), (1, 1024), (8, 128), (16, 128), (128, 576), (128, 1024)):
        cases.append(dict(op="rmsnorm", rows=rows, width=width, recipe="random", inplace=False))
    for axis in (*widths[:-2], 49152, 151936):
        for inner in (1, 3):
            cases.append(dict(op="argmax", outer=2, axis=axis, inner=inner, recipe="random"))
    for recipe in ("ties", "nan-first", "nan-tail", "negative-infinity", "infinity",
                   "signed-zero", "all-nan", "tail-maximum"):
        for inner in (1, 3):
            cases.append(dict(op="argmax", outer=2, axis=577, inner=inner, recipe=recipe))
    for axis in (49152, 151936):
        cases.append(dict(op="argmax", outer=1, axis=axis, inner=1, recipe="random"))
    for axis in (16383, 16384, 16385, 49153, 151937):
        for inner in (1, 3):
            cases.append(dict(op="argmax", outer=2, axis=axis, inner=inner, recipe="random"))
    for recipe in ("nonzero-ties", "part-ties"):
        for inner in (1, 3):
            cases.append(dict(op="argmax", outer=2, axis=577, inner=inner, recipe=recipe))
    for axis in (49152, 151936):
        for recipe in ("ties", "nan-first", "nan-tail", "negative-infinity", "infinity",
                       "signed-zero", "all-nan", "tail-maximum", "nonzero-ties", "part-ties"):
            for inner in (1, 3):
                cases.append(dict(op="argmax", outer=2, axis=axis, inner=inner, recipe=recipe))
    for index, geometry in enumerate(cases):
        geometry["label"] = f"{index:03d}-{geometry['op']}-{geometry['recipe']}"
    return cases


def fixture(g):
    state = 123987
    def sample():
        nonlocal state
        state = (state * 1664525 + 1013904223) & 0xffffffff
        return ((state >> 8) / 16777216 - .5) * 4.
    recipe = g["recipe"]
    if g["op"] == "rmsnorm":
        x = array.array("f", (sample() for _ in range(g["rows"] * g["width"])))
        if recipe == "dynamic":
            x = array.array("f", (math.ldexp(v, i % 41 - 20) for i, v in enumerate(x)))
        elif recipe == "signed-zero":
            x = array.array("f", (-0. if i & 1 else 0. for i in range(len(x))))
        elif recipe in ("nan", "infinity"):
            x[g["width"] // 2] = math.nan if recipe == "nan" else math.inf
        weight = array.array("f", (sample() for _ in range(g["width"])))
        epsilon = C.c_float(1e-12 if recipe == "signed-zero" else 1e-6).value
        oracle = array.array("d")
        for row in range(g["rows"]):
            values = x[row*g["width"]:(row+1)*g["width"]]
            squares = sum(float(v)*float(v) for v in values)
            inverse = 1. / math.sqrt(squares / g["width"] + epsilon)
            oracle.extend(float(v)*inverse*float(w) for v, w in zip(values, weight, strict=True))
        return x, weight, epsilon, oracle
    outer, axis, inner = g["outer"], g["axis"], g["inner"]
    x = array.array("f", (sample() for _ in range(outer*axis*inner)))
    for row in range(outer*inner):
        base = (row//inner)*axis*inner + row%inner
        if recipe == "ties":
            for index in (0, 33, axis-1):
                x[base+index*inner] = 9.
        elif recipe == "nan-first":
            x[base] = math.nan
        elif recipe == "nan-tail":
            x[base+(axis-1)*inner] = math.nan
        elif recipe in ("negative-infinity", "infinity", "signed-zero", "all-nan"):
            for index in range(axis):
                x[base+index*inner] = (-math.inf if recipe == "negative-infinity" else
                    math.inf if recipe == "infinity" else math.nan if recipe == "all-nan" else
                    -0. if index & 1 else 0.)
        elif recipe == "tail-maximum":
            x[base+(axis-1)*inner] = 10.
        elif recipe == "nonzero-ties":
            x[base+min(33, axis-1)*inner] = 9.
            x[base+(axis-1)*inner] = 9.
        elif recipe == "part-ties":
            x[base+min(257, axis-1)*inner] = 9.
            x[base+min(1025, axis-1)*inner] = 9.
    oracle = array.array("i")
    for row in range(outer*inner):
        base = (row//inner)*axis*inner + row%inner
        best, selected = x[base], 0
        for index in range(1, axis):
            value = x[base+index*inner]
            if value > best:
                best, selected = value, index
        oracle.append(selected)
    return x, None, None, oracle


def plan(g, mode):
    count = g["rows"] if g["op"] == "rmsnorm" else g["outer"]*g["inner"]
    width = g["width"] if g["op"] == "rmsnorm" else g["axis"]
    block = 32
    block_limit = 1024 if g["op"] == "argmax" and width >= 16384 else 256
    while block < width and block < block_limit:
        block *= 2
    cooperative = mode == "cooperative"
    symbol = ("vx_cuda_rmsnorm_cooperative_f32" if cooperative else "vx_cuda_rmsnorm_f32") \
        if g["op"] == "rmsnorm" else \
        ("vx_cuda_argmax_cooperative_f32_i32" if cooperative else "vx_cuda_argmax_f32_i32")
    return dict(kernel=symbol, grid=[count if cooperative else (count+255)//256, 1, 1],
                block=[block if cooperative else 256, 1, 1], dynamic_shared_bytes=0,
                static_shared_bytes=(32 if g["op"] == "rmsnorm" else 256) if cooperative else 0)


def provenance(args):
    return dict(ptx_sha256=digest(args.ptx), library_sha256=digest(args.library),
        helper_sha256=digest(__file__), driver_helper_sha256=digest(Path(__file__).with_name("probe_cuda_sampling.py")),
        sources_sha256={name: digest(args.source_root/name) for name in SOURCE_FILES})


def validate(actual, oracle, op):
    marker = SENTINEL if op == "rmsnorm" else ID_SENTINEL
    guards = all(v == marker for v in (*actual[:GUARD], *actual[-GUARD:]))
    values = actual[GUARD:-GUARD]
    if op == "argmax":
        mismatches = sum(a != b for a, b in zip(values, oracle, strict=True))
        return dict(guards_unchanged=guards, exact_index_mismatches=mismatches,
                    passed=guards and not mismatches)
    failures, nonfinite_mismatches, zero_sign_mismatches = 0, 0, 0
    errors, references = [], []
    for value, expected in zip(values, oracle, strict=True):
        if math.isnan(expected):
            nonfinite_mismatches += not math.isnan(value)
        elif math.isinf(expected):
            nonfinite_mismatches += value != expected
        else:
            if not math.isfinite(value):
                nonfinite_mismatches += 1
                continue
            error = abs(value-expected)
            failures += error > 2e-5 + 1e-5*abs(expected)
            errors.append(error); references.append(expected)
            if expected == 0.:
                zero_sign_mismatches += math.copysign(1., value) != math.copysign(1., expected)
    normalized = math.sqrt(sum(v*v for v in errors)/max(sum(v*v for v in references), 1e-24))
    return dict(guards_unchanged=guards, combined_tolerance_failures=failures,
                max_abs=max(errors, default=0.), normalized_rms=normalized,
                nonfinite_classification_mismatches=nonfinite_mismatches,
                signed_zero_mismatches=zero_sign_mismatches,
                passed=guards and not (failures or nonfinite_mismatches or zero_sign_mismatches) and normalized <= 1e-5)


def run(args, receipt):
    driver = Driver()
    context, stream, module, start, end = (C.c_void_p() for _ in range(5))
    allocations, functions = [], {}
    device, version = C.c_int(), C.c_int()
    driver.call("Init", 0); driver.call("DeviceGet", C.byref(device), 0)
    name = C.create_string_buffer(256)
    driver.call("DeviceGetName", name, len(name), device)
    driver.call("DriverGetVersion", C.byref(version))
    receipt["device"] = dict(ordinal=0, name=name.value.decode(), driver_api_version=version.value)
    try:
        driver.call("CtxCreate", C.byref(context), 0, device)
        driver.call("StreamCreate", C.byref(stream), 1)
        ptx = C.create_string_buffer(args.ptx.read_bytes())
        before = time.perf_counter_ns()
        driver.call("ModuleLoadData", C.byref(module), ptx)
        receipt["excluded_module_load_jit_ms"] = (time.perf_counter_ns()-before)/1e6
        driver.call("EventCreate", C.byref(start), 0); driver.call("EventCreate", C.byref(end), 0)
        for config in receipt["configs"]:
            g, launch = config["geometry"], config["launch"]
            x, weight, epsilon, oracle = fixture(g)
            if (hashlib.sha256(x.tobytes()).hexdigest() != config["input_sha256"] or
                hashlib.sha256(oracle.tobytes()).hexdigest() != config["reference_sha256"] or
                (weight is not None and hashlib.sha256(weight.tobytes()).hexdigest() != config["weight_sha256"])):
                raise AssertionError("fixture/reference changed after preparation")
            symbol = launch["kernel"]
            if symbol not in functions:
                functions[symbol] = C.c_void_p()
                driver.call("ModuleGetFunction", C.byref(functions[symbol]), module, symbol.encode())
            owned = []
            def allocate(host):
                pointer = C.c_uint64()
                driver.call("MemAlloc", C.byref(pointer), C.sizeof(host))
                allocations.append(pointer); owned.append(pointer)
                driver.call("MemcpyHtoD", pointer, host, C.sizeof(host))
                return pointer
            input_host = (C.c_float*(len(x)+2*GUARD))(*([SENTINEL]*GUARD), *x, *([SENTINEL]*GUARD))
            input_allocation = allocate(input_host)
            input_pointer = C.c_uint64(input_allocation.value+GUARD*4)
            output_type = C.c_float if g["op"] == "rmsnorm" else C.c_int32
            output_marker = SENTINEL if g["op"] == "rmsnorm" else ID_SENTINEL
            output_host = (output_type*(len(oracle)+2*GUARD))(*([output_marker]*(len(oracle)+2*GUARD)))
            output_allocation = input_allocation if g.get("inplace") else allocate(output_host)
            output_pointer = C.c_uint64(output_allocation.value+GUARD*4)
            if g["op"] == "rmsnorm":
                weight_host = (C.c_float*len(weight)).from_buffer(weight)
                weight_pointer = allocate(weight_host)
                scalars = [input_pointer, weight_pointer, output_pointer,
                           C.c_uint(g["rows"]), C.c_uint(g["width"]), C.c_float(epsilon)]
            else:
                scalars = [input_pointer, output_pointer, C.c_uint(g["outer"]), C.c_uint(g["axis"]), C.c_uint(g["inner"])]
            owned_arguments, params = prepare_arguments(driver, scalars, allocations,
                stored=args.argument_storage)
            observations = config["observations"] = []
            for repetition in range(args.warmups+args.samples):
                driver.call("MemcpyHtoD", input_allocation, input_host, C.sizeof(input_host))
                if not g.get("inplace"):
                    driver.call("MemcpyHtoD", output_allocation, output_host, C.sizeof(output_host))
                driver.call("CtxSynchronize")
                driver.call("EventRecord", start, stream)
                driver.call("LaunchKernel", functions[symbol], *launch["grid"], *launch["block"], 0, stream, params, None)
                driver.call("EventRecord", end, stream); driver.call("EventSynchronize", end)
                elapsed = C.c_float(); driver.call("EventElapsedTime", C.byref(elapsed), start, end)
                actual = (output_type*len(output_host))()
                driver.call("MemcpyDtoH", actual, output_allocation, C.sizeof(actual))
                check = validate(list(actual), oracle, g["op"])
                observed = dict(kind="warmup" if repetition<args.warmups else "measured",
                    repetition=repetition, gpu_event_ms=elapsed.value,
                    output_sha256=hashlib.sha256(bytes(actual)).hexdigest(), **check)
                observations.append(observed)
                if not check["passed"]:
                    raise AssertionError(f"{g['label']} {config['mode']}: {check}")
                if observed["output_sha256"] != observations[0]["output_sha256"]:
                    raise AssertionError(f"{g['label']}: nondeterministic output")
            config["median_gpu_event_ms"] = statistics.median(o["gpu_event_ms"] for o in observations if o["kind"] == "measured")
            config["output_bitwise_deterministic"] = True
            for pointer in reversed(owned):
                driver.call("MemFree", pointer); allocations.remove(pointer)
        if provenance(args) != receipt["provenance"]:
            raise AssertionError("probe/source/PTX/library changed during execution")
        receipt["provenance_unchanged_after_run"] = True
        receipt["status"] = "PASS"
    finally:
        failures = []
        for label, action in [("allocation", lambda ptr=ptr: driver.call("MemFree", ptr)) for ptr in reversed(allocations)] + [
            ("end_event", lambda: driver.call("EventDestroy", end) if end.value else None),
            ("start_event", lambda: driver.call("EventDestroy", start) if start.value else None),
            ("module", lambda: driver.call("ModuleUnload", module) if module.value else None),
            ("stream", lambda: driver.call("StreamDestroy", stream) if stream.value else None),
            ("context", lambda: driver.call("CtxDestroy", context) if context.value else None)]:
            try: action()
            except Exception as error: failures.append(f"{label}: {error}")
        receipt["cleanup_errors"] = failures
        if failures:
            receipt["status"] = "FAIL"
            raise RuntimeError("CUDA cleanup failed: " + "; ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("ptx", "library", "source-root", "out"):
        parser.add_argument("--"+option, type=Path, required=True)
    parser.add_argument("--mode", choices=("scalar", "cooperative", "both"), default="both")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--argument-storage", action="store_true",
                        help="load current device records with poisoned direct arguments")
    args = parser.parse_args()
    if args.out.exists(): parser.error("refusing to overwrite a diagnostic receipt")
    if args.warmups<1 or args.samples<3: parser.error("need warmups >= 1 and samples >= 3")
    if sys.byteorder != "little" or array.array("i").itemsize != 4:
        parser.error("fixture requires little endian and 32-bit array integers")
    receipt = dict(format="volvoxai-private-cuda-reductions-v1", status="REFERENCE_READY",
        provenance=provenance(args), configs=[], argument_storage=args.argument_storage,
        numerical_contract=dict(combined_atol=2e-5, combined_rtol=1e-5, normalized_rms_max=1e-5,
            normalization="sqrt(sum(error^2)/max(sum(reference^2),1e-24))",
            argmax_exact=True, nonfinite_classification_exact=True, signed_zero_exact=True,
            output_guards_exact=True, repeated_results_bitwise=True),
        timing_contract=dict(warmups=args.warmups, measured_samples=args.samples,
            clock="CUDA events on explicit nonblocking stream",
            excluded="allocation/JIT/copies/reset/fence/reference/readback/validation"),
        scope="exact private reduction PTX synthetic fixtures; no public model/decode latency claim")
    try:
        for geometry in geometries():
            x, weight, epsilon, oracle = fixture(geometry)
            for mode in (("scalar", "cooperative") if args.mode == "both" else (args.mode,)):
                receipt["configs"].append(dict(geometry=geometry, mode=mode, launch=plan(geometry, mode),
                    input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                    weight_sha256=hashlib.sha256(weight.tobytes()).hexdigest() if weight is not None else None,
                    epsilon_f32=epsilon, reference_sha256=hashlib.sha256(oracle.tobytes()).hexdigest()))
        if not args.prepare_only: run(args, receipt)
    except Exception as error:
        receipt["status"] = "FAIL"; receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2)+"\n")
        print(json.dumps(dict(status=receipt["status"], configs=len(receipt["configs"]), out=str(args.out))))


if __name__ == "__main__":
    main()
