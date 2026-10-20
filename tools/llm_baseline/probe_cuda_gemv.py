#!/usr/bin/env python3
"""Probe exact production CUDA Linear PTX against independent F64 arithmetic.

Both routes use the same F32 inputs, fixture seeds, output guards and CUDA
events. No models/downloads, NumPy dependency or persistent fixture files are
needed. Kernel timings exclude allocation, JIT, uploads, fences and readback;
they do not represent public API decode latency.
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

GUARD, SENTINEL = 8, -98765.5
SOURCE_FILES = (
    "native/src/backends/cuda/kernels/cuda_forward_argument_storage.inc",
    "tools/llm_baseline/probe_cuda_sampling.py",
    "native/src/backends/cuda/kernels/cuda_forward_linear_moe_kernels.inc",
    "native/src/backends/cuda/host/cuda_graph_dense_moe_host.inc",
    "native/src/backends/cuda/host/cuda_forward_function_registry_host.inc",
    "native/src/backends/cuda_kernels.cu",
)


def geometries(suite):
    if suite == "gemm":
        shapes = [(1, 1, 1), (15, 15, 7), (16, 16, 8), (17, 17, 9),
                  (31, 33, 17), (32, 65, 31), (33, 33, 33), (128, 63, 65),
                  (31, 17, 1537), (63, 33, 64), (65, 17, 65),
                  (33, 17, 1024), (65, 33, 1025), (256, 3, 1057),
                  (39, 5, 1027), (65, 17, 1057),
                  (16, 128, 64), (17, 129, 65), (31, 257, 129), (32, 128, 64)]
        cases = [dict(label=f"gemm-tail-{m}-{k}-{n}-{layout}-bias{int(bias)}",
                      rows=m, k=k, n=n, layout=layout, bias=bias, seed=98743)
                 for m, k, n in shapes for layout in ("IN_OUT", "OUT_IN")
                 for bias in (False, True)]
        # Match active/padded prefill row counts and input widths, while
        # bounding the independent Python F64 oracle to 64 output columns.
        # These strips are kernel fixtures, not full-model performance cases.
        for m, k in ((31, 576), (128, 576), (256, 576),
                     (31, 1024), (128, 1024), (256, 1024)):
            for layout in ("IN_OUT", "OUT_IN"):
                cases.append(dict(label=f"gemm-prefill-strip-{m}-{k}-64-{layout}",
                                  rows=m, k=k, n=64, layout=layout, bias=False,
                                  seed=98743))
        return cases
    shapes = [(1, 1), (15, 3), (16, 7), (17, 8), (31, 9),
              (32, 17), (33, 33), (257, 19)]
    cases = [dict(label=f"tail-{k}-{n}-{layout}-bias{int(bias)}", rows=1,
                  k=k, n=n, layout=layout, bias=bias)
             for k, n in shapes for layout in ("IN_OUT", "OUT_IN")
             for bias in (False, True)]
    if suite == "representative":
        for label, k, n in (("Smol-attention", 576, 576),
                            ("Smol-up", 576, 1536), ("Smol-down", 1536, 576),
                            ("Qwen-attention", 1024, 2048),
                            ("Qwen-up", 1024, 3072), ("Qwen-down", 3072, 1024)):
            # Both storage layouts are supported. The qualified HF model
            # exports use OUT_IN for projections as well as tied LM heads.
            for layout in ("IN_OUT", "OUT_IN"):
                cases.append(dict(label=f"{label}-{layout}", rows=1,
                                  k=k, n=n, layout=layout, bias=False))
        cases.extend([dict(label="Smol-LM-head", rows=1, k=576, n=49152,
                           layout="OUT_IN", bias=False),
                      dict(label="Qwen-LM-head", rows=1, k=1024, n=151936,
                           layout="OUT_IN", bias=False)])
    for case in cases:
        case["seed"] = 98743
    return cases


def fixture(g):
    state = g["seed"]
    def sample():
        nonlocal state
        state = (state * 1664525 + 1013904223) & 0xffffffff
        return ((state >> 8) / 16777216 - .5) * .3
    x = array.array("f", (sample() for _ in range(g["rows"] * g["k"])))
    weight = array.array("f", (sample() for _ in range(g["k"] * g["n"])))
    bias = array.array("f", (sample() for _ in range(g["n"])))
    return x, weight, bias


def reference(g, x, weight, bias):
    k, n, out_in = g["k"], g["n"], g["layout"] == "OUT_IN"
    result = array.array("d")
    for row in range(g["rows"]):
        for column in range(n):
            value = float(bias[column]) if g["bias"] else 0.
            if out_in:
                base = column * k
                for j in range(k):
                    value += float(x[row * k + j]) * float(weight[base + j])
            else:
                for j in range(k):
                    value += float(x[row * k + j]) * float(weight[j * n + column])
            result.append(value)
    return result


def launch(g, mode, tiled_block_x=8, tile_rows=16, tile_columns=16, tiled_shared_bytes=2176,
           adaptive=False, split_entries=False):
    if split_entries and mode == "tiled" and g["rows"] > 32 and g["n"] >= 1024:
        tile_rows, tile_columns, tiled_shared_bytes = 32, 64, 6336
    if adaptive and (g["rows"] <= 32 or g["n"] < 1024):
        tile_rows = tile_columns = 16
    tile = (4 if g["layout"] == "OUT_IN" else 8) if mode == "gemv" else tile_columns
    return dict(grid=[(g["n"] + tile - 1) // tile,
                      1 if mode == "gemv" else (g["rows"] + tile_rows - 1) // tile_rows, 1],
                block=[128, 1, 1] if mode == "gemv" else [tiled_block_x, 16, 1],
                static_shared_bytes=(576 if g["layout"] == "IN_OUT" else 0)
                    if mode == "gemv" else tiled_shared_bytes, dynamic_shared_bytes=0)


def tiled_geometry(source):
    if "void vx_cuda_matmul_register_f32(" in source:
        if not all(tile in source for tile in ("input_tile[32][17]", "weight_tile[16][65]")):
            raise ValueError("split GEMM register entry lacks its declared static tiles")
        if "input_tile[4][16][17]" in source and "weight_tile[4][16][17]" in source:
            return dict(tiled_block_x=8, tiled_shared_bytes=8704, split_entries=True)
        if "input_tile[16][17]" in source and "weight_tile[16][17]" in source:
            return dict(tiled_block_x=8, split_entries=True)
        raise ValueError("split GEMM compact entry lacks its declared static tiles")
    if "input_tile[32][17]" in source and "weight_tile[16][65]" in source:
        return dict(tiled_block_x=8, tile_rows=32, tile_columns=64, tiled_shared_bytes=6336,
                    adaptive="vx_cuda_matmul_compact_body" in source)
    if "input_tile[16][17]" in source and "weight_tile[16][17]" in source:
        return dict(tiled_block_x=8 if "second_column = column + 8u" in source else 16,
                    tile_rows=16, tile_columns=16, tiled_shared_bytes=2176)
    raise ValueError("unrecognized retained tiled kernel geometry")


def kernel(g, mode, split_entries=False):
    operation = "gemv" if mode == "gemv" else "matmul"
    if mode == "tiled" and split_entries and g["rows"] > 32 and g["n"] >= 1024:
        operation += "_register"
    return f"vx_cuda_{operation}{'_out_in' if g['layout'] == 'OUT_IN' else ''}_f32"


def kernel_argument_abi(g, mode):
    arguments = ["ptr"] * 4 + ["u32"] * 3
    if mode == "gemv" and g["layout"] == "OUT_IN":
        arguments += ["ptr", "ptr", "u32"]
    return arguments + ["argument_storage_ptr"]


def provenance(args):
    return {"ptx_sha256": digest(args.ptx), "library_sha256": digest(args.library),
            "helper_sha256": digest(__file__),
            "driver_helper_sha256": digest(Path(__file__).with_name("probe_cuda_sampling.py")),
            "sources_sha256": {name: digest(args.source_root / name) for name in SOURCE_FILES}}


def validate(actual, oracle):
    errors = [abs(float(value) - wanted) for value, wanted in zip(actual[GUARD:-GUARD], oracle, strict=True)]
    rms_error = math.sqrt(sum(value * value for value in errors) / len(errors))
    rms_reference = math.sqrt(sum(value * value for value in oracle) / len(oracle))
    normalized = rms_error / max(rms_reference, 1e-12)
    finite = all(math.isfinite(value) for value in actual)
    guards = all(value == SENTINEL for value in [*actual[:GUARD], *actual[-GUARD:]])
    failures = sum(error > 2e-5 + 1e-5 * abs(wanted) for error, wanted in zip(errors, oracle, strict=True))
    return {"max_abs": max(errors), "normalized_rms": normalized,
            "combined_tolerance_failures": failures, "finite": finite,
            "guards_unchanged": guards,
            "pass": finite and guards and not failures and normalized <= 1e-5}


def run(args, receipt):
    driver = Driver()
    context, stream, module, start, end = (C.c_void_p() for _ in range(5))
    allocations = []
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
        receipt["excluded_module_load_jit_ms"] = (time.perf_counter_ns() - before) / 1e6
        driver.call("EventCreate", C.byref(start), 0); driver.call("EventCreate", C.byref(end), 0)
        functions = {}
        for config in receipt["configs"]:
            g = config["geometry"]
            x, w, b = fixture(g)
            oracle = reference(g, x, w, b)
            inputs = {"input": x, "weight": w, "bias": b}
            if {label: hashlib.sha256(values.tobytes()).hexdigest() for label, values in inputs.items()} != config["input_sha256"]:
                raise AssertionError("fixture changed after preparation")
            if hashlib.sha256(oracle.tobytes()).hexdigest() != config["reference_f64_sha256"]:
                raise AssertionError("F64 reference changed after preparation")
            symbol = config["kernel"]
            if symbol not in functions:
                functions[symbol] = C.c_void_p()
                driver.call("ModuleGetFunction", C.byref(functions[symbol]), module, symbol.encode())
            pointers, owned = {}, []
            for label, values in inputs.items():
                if label == "bias" and not g["bias"]:
                    pointers[label] = C.c_uint64(0); continue
                pointer = C.c_uint64()
                driver.call("MemAlloc", C.byref(pointer), len(values) * 4)
                allocations.append(pointer); owned.append(pointer); pointers[label] = pointer
                host = (C.c_float * len(values)).from_buffer(values)
                driver.call("MemcpyHtoD", pointer, host, len(values) * 4)
            output_elements = g["rows"] * g["n"]
            output_host = (C.c_float * (output_elements + 2 * GUARD))(*([SENTINEL] * (output_elements + 2 * GUARD)))
            output = C.c_uint64(); driver.call("MemAlloc", C.byref(output), C.sizeof(output_host))
            allocations.append(output); owned.append(output)
            scalars = [pointers["input"], pointers["weight"], pointers["bias"],
                       C.c_uint64(output.value + 4 * GUARD),
                       C.c_uint(g["rows"]), C.c_uint(g["k"]), C.c_uint(g["n"])]
            if args.mode == "gemv" and g["layout"] == "OUT_IN":
                scalars += [C.c_uint64(0), C.c_uint64(0), C.c_uint(0)]
            owned_arguments, params = prepare_arguments(driver, scalars, allocations,
                stored=args.argument_storage)
            observations = config["observations"] = []
            for repetition in range(args.warmups + args.samples):
                driver.call("MemcpyHtoD", output, output_host, C.sizeof(output_host))
                driver.call("CtxSynchronize")
                driver.call("EventRecord", start, stream)
                plan = config["launch"]
                driver.call("LaunchKernel", functions[symbol], *plan["grid"], *plan["block"], 0, stream, params, None)
                driver.call("EventRecord", end, stream); driver.call("EventSynchronize", end)
                elapsed = C.c_float(); driver.call("EventElapsedTime", C.byref(elapsed), start, end)
                actual = (C.c_float * len(output_host))()
                driver.call("MemcpyDtoH", actual, output, C.sizeof(actual))
                check = validate(list(actual), oracle)
                observed = dict(kind="warmup" if repetition < args.warmups else "measured",
                                repetition=repetition, gpu_event_ms=elapsed.value,
                                output_f32_sha256=hashlib.sha256(bytes(actual)).hexdigest(), **check)
                observations.append(observed)
                if not check["pass"]:
                    raise AssertionError(f"{g['label']}: numerical mismatch {check}")
                if observed["output_f32_sha256"] != observations[0]["output_f32_sha256"]:
                    raise AssertionError(f"{g['label']}: repeated result is not bitwise deterministic")
            config["measured_median_gpu_ms"] = statistics.median(
                item["gpu_event_ms"] for item in observations if item["kind"] == "measured")
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
            raise RuntimeError("CUDA probe cleanup failed: " + "; ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("ptx", "library", "source-root", "out"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--mode", choices=("tiled", "gemv"), default="gemv")
    parser.add_argument("--suite", choices=("small", "representative", "gemm"), default="representative")
    parser.add_argument("--tiled-block-x", type=int, choices=(8, 16), default=8,
                        help="8 for register/compact PTX; 16 only for retained old scalar-tile PTX")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--argument-storage", action="store_true",
                        help="load current device records with poisoned direct arguments")
    args = parser.parse_args()
    if args.out.exists(): parser.error("refusing to overwrite a diagnostic receipt")
    if args.warmups < 1 or args.samples < 3: parser.error("need warmups >= 1 and samples >= 3")
    if args.suite == "gemm" and args.mode != "tiled": parser.error("multirow GEMM fixtures require --mode tiled")
    tile_geometry = {}
    if args.mode == "tiled":
        source = (args.source_root /
                  "native/src/backends/cuda/kernels/cuda_forward_linear_moe_kernels.inc").read_text()
        tile_geometry = tiled_geometry(source)
        if args.tiled_block_x != tile_geometry["tiled_block_x"]:
            parser.error("tiled block geometry disagrees with the retained kernel source")
    if sys.byteorder != "little": parser.error("fixture requires a little-endian host")
    receipt = dict(format="volvoxai-private-cuda-gemv-v1", status="REFERENCE_READY", mode=args.mode,
                   suite=args.suite, tiled_block_x=args.tiled_block_x,
                   provenance=provenance(args), configs=[],
                   numerical_contract=dict(combined_atol=2e-5, combined_rtol=1e-5,
                       normalized_rms_max=1e-5, normalization="RMS(error)/max(RMS(reference),1e-12)",
                       output_guard_exact=True, repeated_result_bitwise=True),
                   timing_contract=dict(warmups=args.warmups, measured_samples=args.samples,
                       clock="CUDA events on explicit nonblocking stream",
                       excluded="allocation/JIT/upload/output reset/fence/reference/readback/validation"),
                   scope="exact private production F32 Linear kernels; synthetic geometry; no public model/admission/decode latency claim")
    try:
        for g in geometries(args.suite):
            x, w, b = fixture(g); oracle = reference(g, x, w, b)
            receipt["configs"].append(dict(geometry=g, kernel=kernel(g, args.mode,
                split_entries=tile_geometry.get("split_entries", False)),
                kernel_abi=kernel_argument_abi(g, args.mode),
                argument_storage=args.argument_storage, launch=launch(g, args.mode, **tile_geometry),
                input_bytes=dict(input=len(x)*4, weight=len(w)*4, bias=len(b)*4),
                input_sha256={label: hashlib.sha256(values.tobytes()).hexdigest()
                              for label, values in (("input", x), ("weight", w), ("bias", b))},
                reference_f64_sha256=hashlib.sha256(oracle.tobytes()).hexdigest()))
        if not args.prepare_only: run(args, receipt)
    except Exception as error:
        receipt["status"] = "FAIL"; receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(dict(status=receipt["status"], configs=len(receipt["configs"]),
            out=str(args.out), timings=[dict(label=c["geometry"]["label"],
            median_ms=c.get("measured_median_gpu_ms")) for c in receipt["configs"]]), indent=2))


if __name__ == "__main__":
    main()
