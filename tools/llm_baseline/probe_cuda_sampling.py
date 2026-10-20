#!/usr/bin/env python3
"""Private, standalone sampler diagnostic; does not load or alter the engine.

The independent reference uses Python's stable sort and libm. Parallel mode
launches the exact production tile sort, pruned merges, weights and ordered
CDF chain. Serial mode supports the preserved pre-optimization PTX. CUDA
events exclude JIT, copies and allocation; this is not public Generate timing.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

KERNEL = "vx_cuda_generation_sample_f32"
PREPARATION_KERNELS = {
    "tiles": ("vx_cuda_generation_sort_tiles_f32", ["deviceptr", "deviceptr", "u32", "u32", "u32"]),
    "merge": ("vx_cuda_generation_merge_f32", ["deviceptr", "deviceptr", "u32", "u32", "u32"]),
    "weights": ("vx_cuda_generation_weights_f32", ["deviceptr", "deviceptr", "u32", "f64"]),
}
SOURCE_FILES = (
    "native/src/backends/cuda/kernels/cuda_forward_typed_graph_kernels.inc",
    "native/src/backends/cuda/host/cuda_graph_typed_graph_host.inc",
    "native/src/runtime/public_api_generation_policy.inc",
    "native/src/runtime/public_api_full_prefix_generate.inc",
    "native/src/backends/cuda/kernels/cuda_forward_rope_kernels.inc",
    "native/src/backends/cuda/host/cuda_graph_f32_basic_host.inc",
)
ABI = ["deviceptr", "deviceptr", "deviceptr", "deviceptr", "u32", "u32",
       "f64", "u32", "f64", "f64"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def uniform(seed):
    mask = (1 << 64) - 1
    value = (seed + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return (value >> 11) * 2.0 ** -53


def inputs(vocabulary, recipe="hashed"):
    # Exact binary fractions, deliberately repeated scores and two top ties.
    values = [(((((i + 1) * 2654435761) & 0xFFFFFFFF) >> 16) - 32768) / 8192.0
              for i in range(vocabulary)]
    if vocabulary > 83:
        values[17] = values[83] = 4.0
    if recipe == "zeros":
        values = [0.0] * vocabulary
    elif recipe == "infinities":
        values = [math.inf, -math.inf, 0.0, math.inf]
    elif recipe == "negative-infinity":
        values = [-math.inf] * vocabulary
    elif recipe == "nan-tail":
        values[3] = values[-1] = math.nan
    packed = (C.c_float * vocabulary)(*values)
    return values, packed


def reference(values, temperature, top_k, top_p, draw):
    started = time.perf_counter_ns()
    if any(math.isnan(value) for value in values) or max(values) == -math.inf:
        return {"selected_id": -1, "invalid_distribution": True,
                "reference_wall_ms": (time.perf_counter_ns() - started) / 1e6}
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    if top_k:
        order = order[:top_k]
    maximum = values[order[0]]
    weights = [float(values[i] == math.inf) if maximum == math.inf
               else math.exp((values[i] - maximum) / temperature) for i in order]
    total = 0.0
    for weight in weights:
        total += weight
    retained = 0.0
    admitted = 0
    while True:
        retained += weights[admitted]
        admitted += 1
        if admitted == len(order) or retained >= top_p * total:
            break
    target = draw * retained
    cumulative = 0.0
    selected = order[admitted - 1]
    previous = 0.0
    rank = admitted - 1
    for i in range(admitted):
        previous = cumulative
        cumulative += weights[i]
        if target < cumulative:
            selected, rank = order[i], i
            break
    return {
        "selected_id": selected, "selected_rank": rank,
        "top_ids": order[:12],
        "candidate_count": len(order), "top_p_admitted_count": admitted,
        "weight_total": total, "retained_weight": retained,
        "cdf_target": target, "selected_cdf_lower": previous,
        "selected_cdf_upper": cumulative,
        "cdf_min_distance": min(target - previous, cumulative - target),
        "reference_wall_ms": (time.perf_counter_ns() - started) / 1e6,
    }


def launch_schedule(vocabulary, top_k, mode):
    keep = min(top_k, vocabulary) if top_k else vocabulary
    schedule = []
    if mode == "parallel":
        schedule.append({"stage": "tiles", "kernel": PREPARATION_KERNELS["tiles"][0],
                         "grid": [(vocabulary + 255) // 256, 1, 1], "elements": vocabulary})
        width = 256
        while width < vocabulary:
            schedule.append({"stage": "merge", "kernel": PREPARATION_KERNELS["merge"][0],
                             "grid": [(vocabulary + 255) // 256, 1, 1],
                             "elements": vocabulary, "width": width, "keep": keep})
            width *= 2
        schedule.append({"stage": "weights", "kernel": PREPARATION_KERNELS["weights"][0],
                         "grid": [(keep + 255) // 256, 1, 1], "elements": keep})
    schedule.append({"stage": "cdf", "kernel": KERNEL, "grid": [1, 1, 1], "elements": 1})
    return schedule


def prepare(args):
    root = args.source_root.resolve()
    source = {name: digest(root / name) for name in SOURCE_FILES}
    if args.mode == "parallel":
        name = "native/src/backends/cuda/kernels/cuda_forward_generation_kernels.inc"
        source[name] = digest(root / name)
    configs = []
    for vocabulary in (49152, 151936):
        values, packed = inputs(vocabulary)
        input_sha = hashlib.sha256(bytes(packed)).hexdigest()
        for label, temperature, top_k, top_p, seed in (
            ("random-default", 1.0, 0, 1.0, 0),
            ("random-filtered", 0.7, 40, 0.85, 7),
        ):
            draw = uniform(seed)
            configs.append({
                "vocabulary": vocabulary, "row": 0, "policy": label,
                "temperature": temperature, "top_k": top_k, "top_p": top_p,
                "seed": seed, "uniform": draw, "input_sha256": input_sha,
                "input_recipe_name": "hashed",
                "input_recipe": "u32 multiplicative hash upper16 /8192 minus4; ids17+83=4",
                "independent_reference": reference(values, temperature, top_k, top_p, draw),
            })
    if args.edge_cases:
        for vocabulary, recipe, top_k, top_p, draws in (
            (1, "zeros", 0, 1.0, [0.0, math.nextafter(1.0, 0.0)]),
            (2, "zeros", 0, 1.0, [math.nextafter(.5, 0.0), .5, math.nextafter(.5, 1.0)]),
            (513, "zeros", 0, 256 / 513, [0.0, .5, math.nextafter(1.0, 0.0)]),
            (513, "hashed", 255, .85, [uniform(7)]),
            (513, "hashed", 256, .85, [uniform(7)]),
            (513, "hashed", 257, .85, [uniform(7)]),
            (4, "infinities", 0, 1.0, [0.0, .5, math.nextafter(1.0, 0.0)]),
            (513, "negative-infinity", 0, 1.0, [uniform(7)]),
            (513, "nan-tail", 1, 1.0, [uniform(7)]),
            (1048576, "hashed", 40, .85, [uniform(7)]),
        ):
            values, packed = inputs(vocabulary, recipe)
            for draw in draws:
                configs.append({"vocabulary": vocabulary, "row": 0, "policy": f"edge-{recipe}-k{top_k}",
                    "temperature": 1.0, "top_k": top_k, "top_p": top_p, "uniform": draw,
                    "input_sha256": hashlib.sha256(bytes(packed)).hexdigest(), "input_recipe_name": recipe,
                    "independent_reference": reference(values, 1.0, top_k, top_p, draw)})
    for config in configs:
        config["launch_schedule"] = launch_schedule(config["vocabulary"], config["top_k"], args.mode)
    return {
        "format": "volvoxai-private-cuda-sampler-diagnostic-v2", "mode": args.mode,
        "status": "REFERENCE_READY", "kernel": KERNEL, "kernel_abi": ABI,
        "launch": {"grid": [1, 1, 1], "block": [256, 1, 1], "shared_bytes": 0},
        "preparation_kernel_abi": {stage: {"kernel": name, "abi": abi}
            for stage, (name, abi) in PREPARATION_KERNELS.items()} if args.mode == "parallel" else {},
        "provenance": {"ptx_path": str(args.ptx.resolve()), "ptx_sha256": digest(args.ptx),
                       "library_path": str(args.library.resolve()), "library_sha256": digest(args.library),
                       "sources_sha256": source, "helper_sha256": digest(__file__)},
        "configs": configs,
        "scope": "complete private production sampler chain; not public Generate latency, graph, transfer or allocation",
        "timing_contract": "one warm chain and three measured chains per config; CUDA events on explicit nonblocking stream; host uploads and context fence complete before the first start event; allocation/JIT/upload/fence/readback excluded",
    }


class Driver:
    def __init__(self):
        self.lib = C.CDLL("libcuda.so.1")
        ptr = C.c_void_p
        u32 = C.c_uint
        i32 = C.c_int
        size = C.c_size_t
        deviceptr = C.c_uint64
        self.functions = {}
        signatures = {
            "Init": (("cuInit",), [u32]),
            "DeviceGet": (("cuDeviceGet",), [C.POINTER(i32), i32]),
            "DeviceGetName": (("cuDeviceGetName",), [ptr, i32, i32]),
            "DriverGetVersion": (("cuDriverGetVersion",), [C.POINTER(i32)]),
            "CtxCreate": (("cuCtxCreate_v2", "cuCtxCreate"), [C.POINTER(ptr), u32, i32]),
            "CtxSynchronize": (("cuCtxSynchronize",), []),
            "CtxDestroy": (("cuCtxDestroy_v2", "cuCtxDestroy"), [ptr]),
            "StreamCreate": (("cuStreamCreate",), [C.POINTER(ptr), u32]),
            "StreamDestroy": (("cuStreamDestroy_v2", "cuStreamDestroy"), [ptr]),
            "ModuleLoadData": (("cuModuleLoadData",), [C.POINTER(ptr), ptr]),
            "ModuleGetFunction": (("cuModuleGetFunction",), [C.POINTER(ptr), ptr, C.c_char_p]),
            "ModuleUnload": (("cuModuleUnload",), [ptr]),
            "MemAlloc": (("cuMemAlloc_v2", "cuMemAlloc"), [C.POINTER(deviceptr), size]),
            "MemFree": (("cuMemFree_v2", "cuMemFree"), [deviceptr]),
            "MemcpyHtoD": (("cuMemcpyHtoD_v2", "cuMemcpyHtoD"), [deviceptr, ptr, size]),
            "MemcpyDtoH": (("cuMemcpyDtoH_v2", "cuMemcpyDtoH"), [ptr, deviceptr, size]),
            "LaunchKernel": (("cuLaunchKernel",), [ptr, u32, u32, u32, u32, u32, u32, u32, ptr, C.POINTER(ptr), ptr]),
            "EventCreate": (("cuEventCreate",), [C.POINTER(ptr), u32]),
            "EventRecord": (("cuEventRecord",), [ptr, ptr]),
            "EventSynchronize": (("cuEventSynchronize",), [ptr]),
            "EventElapsedTime": (("cuEventElapsedTime",), [C.POINTER(C.c_float), ptr, ptr]),
            "EventDestroy": (("cuEventDestroy_v2", "cuEventDestroy"), [ptr]),
        }
        for name, (symbols, types) in signatures.items():
            fn = next((getattr(self.lib, symbol, None) for symbol in symbols
                       if getattr(self.lib, symbol, None) is not None), None)
            if fn is None:
                raise RuntimeError(f"missing CUDA driver entry {symbols}")
            fn.argtypes, fn.restype = types, C.c_int
            self.functions[name] = fn

    def call(self, name, *args):
        status = self.functions[name](*args)
        if status:
            raise RuntimeError(f"CUDA {name} failed with status {status}")


def prepare_arguments(driver, scalars, allocations, *, stored=False, supported=True):
    """Own parameter values until launch; stored calls poison direct values."""
    values = list(scalars)
    if supported:
        storage = C.c_uint64(0)
        if stored:
            image = (C.c_ubyte * (8 * len(values)))()
            for index, value in enumerate(values):
                if C.sizeof(value) > 8:
                    raise ValueError("device argument record accepts scalar slots only")
                C.memmove(C.addressof(image) + 8 * index, C.byref(value), C.sizeof(value))
            driver.call("MemAlloc", C.byref(storage), C.sizeof(image))
            allocations.append(storage)
            driver.call("MemcpyHtoD", storage, image, C.sizeof(image))
            values = [type(value)() for value in values]
        values.append(storage)
    params = (C.c_void_p * len(values))(*[C.cast(C.byref(value), C.c_void_p) for value in values])
    return values, params


def run(args, receipt):
    driver = Driver()
    ctx, stream, module, function = (C.c_void_p() for _ in range(4))
    start, end = C.c_void_p(), C.c_void_p()
    allocations = []
    device = C.c_int()
    driver.call("Init", 0)
    driver.call("DeviceGet", C.byref(device), 0)
    name = C.create_string_buffer(256)
    version = C.c_int()
    driver.call("DeviceGetName", name, len(name), device)
    driver.call("DriverGetVersion", C.byref(version))
    receipt["device"] = {"ordinal": 0, "name": name.value.decode(), "driver_api_version": version.value}
    try:
        driver.call("CtxCreate", C.byref(ctx), 0, device)
        driver.call("StreamCreate", C.byref(stream), 1)
        ptx = C.create_string_buffer(args.ptx.read_bytes())
        module_started = time.perf_counter_ns()
        driver.call("ModuleLoadData", C.byref(module), ptx)
        receipt["excluded_module_load_jit_wall_ms"] = (time.perf_counter_ns() - module_started) / 1e6
        driver.call("ModuleGetFunction", C.byref(function), module, KERNEL.encode())
        functions = {"cdf": function}
        if args.mode == "parallel":
            for stage, (name, _) in PREPARATION_KERNELS.items():
                functions[stage] = C.c_void_p()
                driver.call("ModuleGetFunction", C.byref(functions[stage]), module, name.encode())
        driver.call("EventCreate", C.byref(start), 0)
        driver.call("EventCreate", C.byref(end), 0)
        for config in receipt["configs"]:
            vocabulary = config["vocabulary"]
            values, packed = inputs(vocabulary, config["input_recipe_name"])
            pointers = []
            scratch_bytes = vocabulary * (32 if args.mode == "parallel" else 16)
            for amount in (vocabulary * 4, 4, scratch_bytes, 4):
                pointer = C.c_uint64()
                driver.call("MemAlloc", C.byref(pointer), amount)
                allocations.append(pointer)
                pointers.append(pointer)
            driver.call("MemcpyHtoD", pointers[0], packed, C.sizeof(packed))
            ids_marker = C.c_int(-555)
            driver.call("MemcpyHtoD", pointers[1], C.byref(ids_marker), 4)
            # Pageable host copies can return before destination DMA finishes.
            # Fence these uploads before the nonblocking stream's timed chain.
            driver.call("CtxSynchronize")
            config["scratch_bytes"] = scratch_bytes
            config["device_allocation_bytes"] = vocabulary * 4 + scratch_bytes + 8
            keep = min(config["top_k"], vocabulary) if config["top_k"] else vocabulary
            launches = []
            for repetition in range(4):
                # Only the kernel chain is between the timing events. Readback
                # and validation follow the completed end event.
                driver.call("EventRecord", start, stream)
                source = pointers[2]
                destination = C.c_uint64(source.value + vocabulary * 16)
                for stage in config["launch_schedule"]:
                    name = stage["stage"]
                    if name == "tiles":
                        scalars = [pointers[0], source, C.c_uint(vocabulary), C.c_uint(config["row"]), C.c_uint(keep)]
                    elif name == "merge":
                        scalars = [source, destination, C.c_uint(vocabulary), C.c_uint(stage["width"]), C.c_uint(keep)]
                    elif name == "weights":
                        scalars = [source, destination, C.c_uint(keep), C.c_double(config["temperature"])]
                    else:
                        weights = destination if args.mode == "parallel" else pointers[2]
                        scalars = [pointers[0], pointers[1], weights, pointers[3], C.c_uint(vocabulary),
                            C.c_uint(config["row"]), C.c_double(config["temperature"]), C.c_uint(config["top_k"]),
                            C.c_double(config["top_p"]), C.c_double(config["uniform"])]
                    params = (C.c_void_p * len(scalars))(*[C.cast(C.byref(v), C.c_void_p) for v in scalars])
                    driver.call("LaunchKernel", functions[name], *stage["grid"], 256, 1, 1, 0, stream, params, None)
                    if name == "merge":
                        source, destination = destination, source
                driver.call("EventRecord", end, stream)
                driver.call("EventSynchronize", end)
                elapsed = C.c_float()
                driver.call("EventElapsedTime", C.byref(elapsed), start, end)
                selected, ids = C.c_int(-1), C.c_int(-1)
                driver.call("MemcpyDtoH", C.byref(selected), pointers[3], 4)
                driver.call("MemcpyDtoH", C.byref(ids), pointers[1], 4)
                expected = config["independent_reference"]["selected_id"]
                expected_ids_row = -555 if expected < 0 else expected
                item = {"kind": "warm" if repetition == 0 else "measured", "repetition": repetition,
                        "gpu_event_ms": elapsed.value, "selected_id": selected.value, "ids_row": ids.value,
                        "expected_id": expected, "expected_ids_row": expected_ids_row,
                        "program_launches": len(config["launch_schedule"]),
                        "pass": selected.value == expected and ids.value == expected_ids_row}
                launches.append(item)
                config["launches"] = launches
                if not item["pass"]:
                    raise AssertionError(f"selected ID mismatch for V{vocabulary} {config['policy']}: {item}")
            config["measured_median_gpu_ms"] = statistics.median(x["gpu_event_ms"] for x in launches[1:])
            for pointer in reversed(pointers):
                driver.call("MemFree", pointer)
                allocations.remove(pointer)
        provenance = receipt["provenance"]
        if digest(args.ptx) != provenance["ptx_sha256"] or digest(args.library) != provenance["library_sha256"]:
            raise AssertionError("PTX or library changed during measurement")
        for name, before in provenance["sources_sha256"].items():
            if digest(args.source_root / name) != before:
                raise AssertionError(f"source changed during measurement: {name}")
        if digest(__file__) != provenance["helper_sha256"]:
            raise AssertionError("probe changed during measurement")
        receipt["provenance_unchanged_after_run"] = True
        receipt["status"] = "PASS"
    finally:
        for pointer in reversed(allocations):
            driver.call("MemFree", pointer)
        if end.value:
            driver.call("EventDestroy", end)
        if start.value:
            driver.call("EventDestroy", start)
        if module.value:
            driver.call("ModuleUnload", module)
        if stream.value:
            driver.call("StreamDestroy", stream)
        if ctx.value:
            driver.call("CtxDestroy", ctx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptx", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--mode", choices=("parallel", "serial"), default="parallel")
    parser.add_argument("--edge-cases", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite a diagnostic receipt")
    receipt = prepare(args)
    if args.reference:
        expected = json.loads(args.reference.read_text())
        for field in ("mode", "kernel", "kernel_abi", "launch", "preparation_kernel_abi"):
            if expected[field] != receipt[field]:
                raise ValueError(f"reference {field} changed")
        for field in ("ptx_sha256", "library_sha256", "helper_sha256"):
            if expected["provenance"][field] != receipt["provenance"][field]:
                raise ValueError(f"{field} differs from prepared reference")
        if expected["provenance"]["sources_sha256"] != receipt["provenance"]["sources_sha256"]:
            raise ValueError("source differs from prepared reference")
        for before, now in zip(expected["configs"], receipt["configs"], strict=True):
            before = dict(before)
            now = dict(now)
            before["independent_reference"] = dict(before["independent_reference"])
            now["independent_reference"] = dict(now["independent_reference"])
            before["independent_reference"].pop("reference_wall_ms")
            now["independent_reference"].pop("reference_wall_ms")
            if before != now:
                raise ValueError("independent reference configuration changed")
        receipt["prepared_reference_sha256"] = digest(args.reference)
    try:
        if not args.prepare_only:
            run(args, receipt)
    except Exception as error:
        receipt["status"] = "FAIL"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"status": receipt["status"], "out": str(args.out),
                          "ptx_sha256": receipt["provenance"]["ptx_sha256"],
                          "configs": [{"vocabulary": c["vocabulary"], "policy": c["policy"],
                                       "expected_id": c["independent_reference"]["selected_id"],
                                       "gpu_median_ms": c.get("measured_median_gpu_ms")}
                                      for c in receipt["configs"]]}, indent=2))


if __name__ == "__main__":
    main()
