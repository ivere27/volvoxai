#!/usr/bin/env python3
"""Benchmark prepared TTS requests with retained VolvoxAI and ONNX sessions.

The measured request starts with prepared host tensors and ends with host
outputs. Loading, compilation, phonemization, file I/O and assertions are
reported separately or excluded. No CLI subprocess is timed as inference.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.tts_common.artifacts import sha256


class ProviderUnavailable(RuntimeError):
    pass


def elapsed_ms(start):
    return (time.perf_counter_ns() - start) / 1e6


def summarize(values):
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("nonempty finite, nonnegative measurements required")
    ordered = sorted(values)
    position = .9 * (len(ordered) - 1)
    low = int(position)
    p90 = ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)
    return {"count": len(values), "median": statistics.median(values), "p90": p90,
            "minimum": min(values), "maximum": max(values), "samples": values}


def checked_cast(value, dtype):
    array, dtype = np.asarray(value), np.dtype(dtype)
    if array.dtype.kind in "iu" and dtype.kind in "iu" and array.size:
        bounds = np.iinfo(dtype)
        if int(array.min()) < bounds.min or int(array.max()) > bounds.max:
            raise ValueError(f"input values do not fit {dtype}")
    elif array.dtype != dtype and not np.can_cast(array.dtype, dtype, casting="safe"):
        # The exporter represents public floating tensors in F32. An F64
        # source endpoint would be a different contract, not an implicit cast.
        raise ValueError(f"unsafe input dtype conversion: {array.dtype} -> {dtype}")
    return np.array(array, dtype=dtype, order="C", copy=True)


def attest_route(report, backend):
    route = report.route
    if (report.backend != backend or route.provider != backend or not route.builtin or
            not route.attested or route.active_nodes <= 0 or
            route.active_nodes != route.selected_nodes or route.missing_nodes or route.fallback_nodes or
            report.fallback.operator_fallback_used or report.compilation.tier_fallback_used):
        raise RuntimeError(f"requested backend {backend!r} has no complete execution attestation")
    return {"backend": backend, "provider": route.provider, "device": report.device,
            "builtin": True, "attested": True,
            "active_nodes": route.active_nodes, "selected_nodes": route.selected_nodes,
            "fallback_nodes": 0, "missing_nodes": 0}


def compare_outputs(actual, expected, *, deterministic, audio_output, rtol, atol, waveform=None, report_waveform_differences=False):
    if not expected:
        raise ValueError("a nonempty prepared reference is required")
    if actual.keys() != expected.keys():
        raise ValueError("output names differ from the prepared reference")
    evidence = {}
    for name, reference in expected.items():
        reference = np.asarray(reference)
        if not reference.size or not np.isfinite(reference).all():
            raise ValueError(f"{name}: empty or nonfinite reference")
        value = np.asarray(actual[name])
        if value.dtype != reference.dtype:
            raise ValueError(f"{name}: dtype {value.dtype} != reference {reference.dtype}")
        if not value.size or not np.isfinite(value).all():
            raise ValueError(f"{name}: empty or nonfinite output")
        dynamic_audio = not deterministic and name == audio_output
        if value.shape != reference.shape and not (dynamic_audio and value.ndim == reference.ndim and
                                                    value.shape[:-1] == reference.shape[:-1]):
            raise ValueError(f"{name}: shape {value.shape} != reference {reference.shape}")
        item = {"shape": list(value.shape), "dtype": value.dtype.name, "finite": True}
        if deterministic:
            error = np.asarray(value, np.float64) - np.asarray(reference, np.float64)
            maximum, rmse = float(np.max(np.abs(error))), float(np.sqrt(np.mean(error * error)))
            strict_mismatches = int(np.count_nonzero(np.abs(error) > atol + rtol * np.abs(reference)))
            accepted = strict_mismatches == 0
            if reference.dtype.kind in "biu":
                np.testing.assert_array_equal(value, reference)
            elif waveform and name == audio_output:
                limits = [waveform["max_absolute_error"], waveform["max_rmse"]]
                if not all(isinstance(x, (float, int)) and math.isfinite(x) and x >= 0 for x in limits):
                    raise ValueError("invalid declared waveform criterion")
                accepted = maximum <= limits[0] and rmse <= limits[1]
                if not accepted and not report_waveform_differences:
                    raise AssertionError(f"{name}: waveform max {maximum} / RMSE {rmse} exceeds declared {limits}")
                power = float(np.mean(np.asarray(reference, np.float64) ** 2))
                item.update(criterion="declared_waveform_error", waveform_criterion=waveform,
                    strict_elementwise_pass=strict_mismatches == 0, strict_mismatches=strict_mismatches,
                    snr_db=10 * math.log10(power / (rmse * rmse)) if power > 0 and rmse > 0 else None)
            else:
                if not (report_waveform_differences and name == audio_output):
                    np.testing.assert_allclose(value, reference, rtol=rtol, atol=atol)
                item.update(criterion="elementwise", strict_elementwise_pass=strict_mismatches == 0,
                    strict_mismatches=strict_mismatches)
            item.update(max_absolute_error=maximum, rmse=rmse, numerical_parity=accepted)
        else:
            item["numerical_parity"] = "not_claimed_stochastic_model"
        evidence[name] = item
    return evidence


def audio_seconds(outputs, case, manifest):
    rate = case.get("sample_rate", manifest.get("sample_rate"))
    if not isinstance(rate, int) or rate <= 0:
        raise ValueError("a positive integer sample_rate is required")
    audio = np.asarray(outputs[case.get("audio_output", "audio")])
    if not audio.ndim or math.prod(audio.shape[:-1]) != 1 or audio.shape[-1] <= 0:
        raise ValueError("RTF requires one nonempty mono waveform in the last axis")
    return audio.shape[-1] / rate


class Native:
    """Only generated public service calls; one context survives all samples."""
    def __init__(self, root, contract, backend, threads, library, timeout):
        sys.path.insert(0, str(ROOT / "python"))
        import volvoxai as vx
        self.vx, self.pb, self.backend, self.timeout = vx, vx.pb, backend, timeout
        pb = self.pb
        self.resources, self.host, self.setup_ms, self.route = [], None, {}, {}
        self.contract = contract
        selected = Path(library).resolve() if library else vx.find_library().resolve()
        self.metadata = {"sdk_version": vx.__version__, "library_sha256": sha256(selected),
                         "cpu_threads": threads, "requested_backend": backend,
                         "vulkan_arena_mb": os.environ.get("VOLVOX_VULKAN_MB", "512 (default)"),
                         "input_storage": "borrowed_host", "output_storage": "owned_then_copy_to_host"}
        try:
            started = time.perf_counter_ns()
            self.host = vx.open_library(selected)
            self.setup_ms["library_open"] = elapsed_ms(started)
            self.client, self.buffers = vx.VxInferenceServiceClient(self.host), vx.VxBufferServiceClient(self.host)
            info = vx.VxPlatformServiceClient(self.host).get_platform_info(pb.Empty())
            self.metadata.update(library_version=info.library_version, api_version=info.api_version,
                                 build_profile=int(info.build_profile), compiled_backends=list(info.compiled_backends))
            def stage(name, operation, request):
                started = time.perf_counter_ns()
                result = operation(request, timeout=self.timeout)
                self.setup_ms[name] = elapsed_ms(started)
                return result
            runtime = stage("runtime_create", self.client.create_runtime, pb.CreateRuntimeRequest(cpu_threads=threads))
            self.resources.append((self.client.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id)))
            loaded = stage("model_load", self.client.load_model, pb.LoadModelRequest(
                runtime_id=runtime.runtime_id, graph_path=str((root / contract["graph"]).resolve()),
                weight_paths=[str((root / path).resolve()) for path in contract.get("weights", [])]))
            self.resources.append((self.client.release_model, pb.ModelRef(model_id=loaded.model_id)))
            compiled = stage("compile", self.client.compile_model, pb.CompileModelRequest(model_id=loaded.model_id,
                policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                    backends=[backend], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
            self.resources.append((self.client.release_compiled_model, pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id)))
            self.route["compile"] = attest_route(compiled.report, backend)
            context = stage("context_create", self.client.create_execution_context,
                pb.CreateExecutionContextRequest(compiled_model_id=compiled.compiled_model_id, graph=pb.GraphExecution()))
            self.context = context.context_id
            self.resources.append((self.client.release_execution_context, pb.ExecutionContextRef(context_id=self.context)))
            self.inputs = {spec.name: spec for spec in context.inputs}
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise RuntimeError(f"{type(error).__name__}: {error}; cleanup: {cleanup_error}") from error
            raise

    def prepare(self, inputs):
        pb = self.pb
        types = {pb.DataType.DATA_TYPE_F32: np.float32, pb.DataType.DATA_TYPE_I32: np.int32}
        mapping = self.contract["inputs"]
        if set(mapping) != set(inputs) or set(mapping.values()) != set(self.inputs):
            raise ValueError("native semantic input mapping does not match prepared inputs/model")
        self.arrays = {mapping[name]: checked_cast(value, types[self.inputs[mapping[name]].dtype])
                       for name, value in inputs.items()}
        tensors = [pb.Tensor(name=name, shape=list(array.shape), dtype=self.inputs[name].dtype,
            borrowed=pb.BorrowedBuffer(resource=pb.NativeResource(
                kind=pb.NativeResourceKind.NATIVE_RESOURCE_KIND_HOST, handle=array.ctypes.data,
                size_bytes=array.nbytes), length_bytes=array.nbytes)) for name, array in self.arrays.items()]
        self.request = pb.ExecuteRequest(context_id=self.context, inputs=pb.InputBindings(values=tensors),
            scope=pb.ExecutionScope(entrypoint=self.contract.get("entrypoint", "evaluate"), full=pb.Empty()),
            outputs=pb.OutputOptions(selection=pb.TensorOutputSelection(names=list(self.contract["outputs"].values())),
                storage=pb.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER))

    def run(self):
        pb = self.pb
        started = time.perf_counter_ns()
        packet = self.client.execute(self.request, timeout=self.timeout)
        execution_ms = elapsed_ms(started)
        buffers = [tensor.buffer.buffer_id for tensor in packet.outputs]
        timings = {"execute_ms": execution_ms}
        outputs = {}
        try:
            allocation = time.perf_counter_ns()
            types = {pb.DataType.DATA_TYPE_F32: np.float32, pb.DataType.DATA_TYPE_I32: np.int32}
            names = {name: semantic for semantic, name in self.contract["outputs"].items()}
            if set(names) != {tensor.name for tensor in packet.outputs}:
                raise ValueError("native output selection changed")
            into = []
            for tensor in packet.outputs:
                if tensor.WhichOneof("payload") != "buffer":
                    raise ValueError("owned native output required")
                array = np.empty(tuple(tensor.shape), dtype=types[tensor.dtype])
                outputs[names[tensor.name]] = array
                into.append(pb.BorrowedBuffer(resource=pb.NativeResource(
                    kind=pb.NativeResourceKind.NATIVE_RESOURCE_KIND_HOST, handle=array.ctypes.data,
                    size_bytes=array.nbytes), length_bytes=array.nbytes))
            timings["output_allocation_ms"] = elapsed_ms(allocation)
            copy = time.perf_counter_ns()
            self.buffers.copy_tensors(pb.CopyTensorsRequest(sources=packet.outputs, into=into), timeout=self.timeout)
            timings["copy_to_host_ms"] = elapsed_ms(copy)
        finally:
            release = time.perf_counter_ns()
            if buffers:
                self.buffers.release_buffers(pb.BufferRefs(buffer_ids=buffers), timeout=self.timeout)
            timings["release_ms"] = elapsed_ms(release)
        timings["request_ms"] = elapsed_ms(started)
        timings["runtime_host_ms"] = packet.metrics.host_time_ns / 1e6
        # Completion and copies precede this check; assertions are not timed.
        self.route["execute"] = attest_route(packet.report, self.backend)
        return outputs, timings

    def close(self):
        errors = []
        while self.resources:
            release, reference = self.resources.pop()
            try:
                release(reference, timeout=self.timeout)
            except Exception as error:
                errors.append(error)
        if self.host is not None:
            try:
                self.host.close()
            except Exception as error:
                errors.append(error)
            finally:
                self.host = None
        if errors:
            raise RuntimeError("Native cleanup failed: " + "; ".join(str(error) for error in errors)) from errors[0]


class NativePipeline:
    """Run every learned stage; only the declared duration scalar guides shape admission."""
    def __init__(self, root, contract, backend, threads, library, timeout):
        self.contract, self.stages, self.setup_ms, self.metadata, self.route = contract, [], {}, {}, {}
        entries = contract["pipeline"]
        if len(entries) < 2 or len({entry["name"] for entry in entries}) != len(entries):
            raise ValueError("invalid pipeline stage names")
        try:
            for entry in entries:
                path = root / entry["manifest"]
                stage = json.loads(path.read_text())
                if stage["volvox"].get("pipeline"):
                    raise ValueError("nested pipelines are unsupported")
                engine = Native(path.parent, stage["volvox"], backend, threads, library, timeout)
                self.stages.append((entry["name"], engine))
                self.setup_ms[entry["name"]] = engine.setup_ms
                self.metadata[entry["name"]] = engine.metadata
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise RuntimeError(f"{type(error).__name__}: {error}; cleanup: {cleanup_error}") from error
            raise

    def prepare(self, inputs):
        required, available = set(), set()
        for _, engine in self.stages:
            required.update(set(engine.contract["inputs"]) - available)
            available.update(engine.contract["outputs"])
        if required != set(inputs):
            raise ValueError("pipeline bindings must contain exactly the original request inputs")
        self.inputs = inputs

    def run(self):
        started = time.perf_counter_ns()
        values, timing = dict(self.inputs), {}
        for name, engine in self.stages:
            binding = time.perf_counter_ns()
            engine.prepare({key: values[key] for key in engine.contract["inputs"]})
            timing["input_binding_ms"] = timing.get("input_binding_ms", 0) + elapsed_ms(binding)
            outputs, stage_timing = engine.run()
            values.update(outputs)
            timing[name + "_request_ms"] = stage_timing["request_ms"]
            for key, value in stage_timing.items():
                if key != "request_ms": timing[key] = timing.get(key, 0) + value
            self.route[name] = engine.route
            feedback = self.contract.get("shape_feedback")
            if feedback and feedback["stage"] == name:
                count = outputs[feedback["output"]].reshape(-1)[feedback["index"]].item()
                if not math.isfinite(count) or int(count) != count or count <= 0 or count != feedback["expected"]:
                    raise ValueError(f"Duration {count} differs from prepared shape {feedback['expected']}; prepare this request again")
        timing["request_ms"] = elapsed_ms(started)
        return {"audio": values["audio"]}, timing

    def close(self):
        error = None
        while self.stages:
            _, engine = self.stages.pop()
            try: engine.close()
            except Exception as caught: error = error or caught
        if error is not None: raise error


class Ort:
    def __init__(self, root, contract, provider, threads, provider_options, temporary, *, allow_cpu_nodes=False):
        import onnxruntime as ort
        if provider not in ort.get_available_providers():
            raise ProviderUnavailable(f"ONNX Runtime provider unavailable: {provider}; available={ort.get_available_providers()}")
        self.contract, self.provider, self.profile_done = contract, provider, False
        self.allow_cpu_nodes = allow_cpu_nodes
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if provider != "CPUExecutionProvider" and not allow_cpu_nodes:
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(temporary) / "ort-placement")
        started = time.perf_counter_ns()
        self.session = ort.InferenceSession(str((root / contract["path"]).resolve()), options,
                                            providers=[(provider, provider_options)])
        self.setup_ms = {"session_load_and_compile": elapsed_ms(started)}
        self.session.disable_fallback()
        self.metadata = {"version": ort.__version__, "requested_provider": provider,
            "registered_providers": self.session.get_providers(),
            "provider_options": self.session.get_provider_options(),
            "intra_op_threads": threads, "inter_op_threads": threads,
            "execution_mode": "ORT_SEQUENTIAL", "graph_optimization": "ORT_ENABLE_ALL",
            "cpu_ep_fallback_disabled": provider != "CPUExecutionProvider" and not allow_cpu_nodes,
            "cpu_nodes_allowed": allow_cpu_nodes, "runtime_fallback_disabled": True}
        self.route = {}

    def prepare(self, inputs):
        types = {"tensor(float)": np.float32, "tensor(double)": np.float64,
                 "tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(bool)": np.bool_}
        expected = {spec.name: spec for spec in self.session.get_inputs()}
        mapping = self.contract["inputs"]
        if set(mapping) != set(inputs) or set(mapping.values()) != set(expected):
            raise ValueError("ONNX semantic input mapping does not match prepared inputs/model")
        self.feed = {mapping[name]: checked_cast(value, types[expected[mapping[name]].type]) for name, value in inputs.items()}

    def run(self):
        names = list(self.contract["outputs"].values())
        started = time.perf_counter_ns()
        values = self.session.run(names, self.feed)
        timings = {"request_ms": elapsed_ms(started)}
        # session.run returns CPU arrays after device completion/readback.
        outputs = dict(zip(self.contract["outputs"], values))
        if not self.profile_done:
            profile_path = Path(self.session.end_profiling())
            events = json.loads(profile_path.read_text())
            counts, operations = {}, {}
            for event in events:
                selected = event.get("args", {}).get("provider")
                if event.get("cat") == "Node" and selected:
                    counts[selected] = counts.get(selected, 0) + 1
                    op = event.get("args", {}).get("op_name", "unknown")
                    by_op = operations.setdefault(selected, {})
                    by_op[op] = by_op.get(op, 0) + 1
            profile_path.unlink()
            allowed = {self.provider}
            if self.allow_cpu_nodes:
                allowed.add("CPUExecutionProvider")
            self.route = {"node_provider_events": counts, "node_provider_operations": operations,
                "placement": "+".join(sorted(counts)), "profiling_ended_before_warmup": True}
            if self.provider not in counts or not set(counts).issubset(allowed):
                raise RuntimeError(f"ORT actual node providers differ from required provider: {counts}")
            self.profile_done = True
        return outputs, timings

    def close(self):
        self.session = None


def load_npz(path):
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name].copy() for name in stored.files}


def run_cases(runner, root, manifest, warmups, repeats, rtol, atol, report_waveform_differences=False):
    if not manifest.get("cases"):
        raise ValueError("at least one prepared request is required")
    rows, first_outputs = [], {}
    for case in manifest["cases"]:
        if case.get("validation", {}).get("reference_engine") != "onnxruntime":
            raise ValueError("ONNX comparison requires an explicit ONNX Runtime reference; prepare with validation enabled")
        inputs = load_npz(root / case["inputs"])
        expected = load_npz(root / case["reference"])
        deterministic = case.get("validation", {}).get("deterministic", True)
        prepared = time.perf_counter_ns()
        runner.prepare(inputs)
        input_prepare_ms = elapsed_ms(prepared)
        correctness = {}
        def verify(outputs):
            evidence = compare_outputs(outputs, expected, deterministic=deterministic,
                audio_output=case.get("audio_output", "audio"), rtol=rtol, atol=atol,
                waveform=case.get("validation", {}).get("waveform"),
                report_waveform_differences=report_waveform_differences)
            for name, item in evidence.items():
                prior = correctness.get(name)
                if prior is not None:
                    for key in ("max_absolute_error", "rmse", "strict_mismatches"):
                        if key in item: item[key] = max(item[key], prior.get(key, 0))
                    if prior.get("numerical_parity") is False: item["numerical_parity"] = False
                    if prior.get("strict_elementwise_pass") is False: item["strict_elementwise_pass"] = False
                    if item.get("snr_db") is not None and prior.get("snr_db") is not None:
                        item["snr_db"] = min(item["snr_db"], prior["snr_db"])
                correctness[name] = item
        output, first = runner.run()
        verify(output)
        first_outputs[case["id"]] = output
        warmup_samples = []
        for _ in range(warmups):
            output, timing = runner.run()
            verify(output)
            warmup_samples.append(timing["request_ms"])
        measurements = []
        for _ in range(repeats):
            output, timing = runner.run()
            verify(output)
            timing["audio_seconds"] = audio_seconds(output, case, manifest)
            timing["real_time_factor"] = timing["request_ms"] / (1000 * timing["audio_seconds"])
            measurements.append(timing)
        rows.append({"id": case["id"], "status": "pass" if all(item["numerical_parity"] is not False for item in correctness.values()) else "numerical_difference", "input_prepare_ms": input_prepare_ms,
            "first_request": first, "first_request_profiles_ort": isinstance(runner, Ort) and not rows,
            "warmups": warmups, "warmup_request_ms": warmup_samples, "repeats": repeats,
            "validation": {"deterministic": deterministic, "reason": case.get("validation", {}).get("reason"),
                           "outputs": correctness},
            "measurements": {key: summarize([item[key] for item in measurements]) for key in measurements[0]}})
        print(f"{rows[-1]['status'].upper()} {type(runner).__name__} {case['id']} median={rows[-1]['measurements']['request_ms']['median']:.3f} ms", flush=True)
    return rows, first_outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="Exit: 0 requested engines pass; 2 skipped/unsupported; 3 measured numerical differences; 1 failure.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", choices=("both", "volvox", "ort"), default="both")
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan", "opengl", "metal"), default="cpu")
    parser.add_argument("--ort-provider", choices=("CPUExecutionProvider", "CUDAExecutionProvider"), default="CPUExecutionProvider")
    parser.add_argument("--ort-provider-options", default="{}", help="JSON object of explicit ORT provider options")
    parser.add_argument("--ort-allow-cpu-nodes", action="store_true",
        help="Allow and report ORT CPU node placement alongside the requested provider; at least one requested-provider node remains mandatory")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--report-waveform-differences", action="store_true",
        help="Record latency and waveform errors without asserting numerical parity; shape/dtype/finite/route checks remain mandatory")
    args = parser.parse_args(argv)
    if (args.threads < 1 or args.warmups < 0 or args.repeats < 1 or args.timeout <= 0 or
            args.rtol < 0 or args.atol < 0 or not all(math.isfinite(value) for value in (args.timeout, args.rtol, args.atol))):
        parser.error("threads/repeats/timeout must be positive; warmups/tolerances nonnegative")
    root = args.manifest.resolve().parent
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("format") != "volvoxai-tts-example/v1" or not manifest.get("cases"):
        parser.error("a nonempty volvoxai-tts-example/v1 manifest is required")
    provider_options = json.loads(args.ort_provider_options)
    if not isinstance(provider_options, dict):
        parser.error("--ort-provider-options must be a JSON object")
    report = {"format": "volvoxai-tts-benchmark/v1", "status": "failed", "model": manifest.get("model"),
        "scope": manifest.get("scope", "scope not declared by prepared manifest"),
        "manifest_sha256": sha256(args.manifest), "caller_sha256": sha256(Path(__file__)),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "os": platform.system(), "machine": platform.machine(), "processor": platform.processor()},
        "timing_contract": "prepared host inputs through completed host outputs; loading, input preparation, file I/O, assertions excluded",
        "settings": {"threads": args.threads, "warmups": args.warmups, "repeats": args.repeats,
                     "rtol": args.rtol, "atol": args.atol, "report_waveform_differences": args.report_waveform_differences,
                     "ort_allow_cpu_nodes": args.ort_allow_cpu_nodes}, "artifacts": {}, "engines": {}, "comparison": {"status": "unavailable"}}
    files = {manifest["onnx"]["path"], *manifest["onnx"].get("external_data", [])}
    for case in manifest["cases"]:
        files.update((case["inputs"], case["reference"]))
    if args.engine in ("both", "volvox") and manifest.get("volvox", {}).get("status") == "ready":
        if manifest["volvox"].get("pipeline"):
            for entry in manifest["volvox"]["pipeline"]:
                files.add(entry["manifest"])
                stage_path = root / entry["manifest"]
                stage = json.loads(stage_path.read_text())
                for asset in (stage["volvox"]["graph"], *stage["volvox"].get("weights", [])):
                    files.add(os.path.relpath(stage_path.parent / asset, root))
        else:
            files.update((manifest["volvox"]["graph"], *manifest["volvox"].get("weights", [])))
    for name in sorted(files):
        report["artifacts"][name] = sha256(root / name)
    outputs = {}
    try:
        with tempfile.TemporaryDirectory(prefix="volvox-tts-benchmark-") as temporary:
            for engine in ("ort", "volvox"):
                if args.engine not in ("both", engine):
                    continue
                result = report["engines"][engine] = {"status": "failed"}
                runner = None
                if engine == "volvox" and manifest.get("volvox", {}).get("status") != "ready":
                    result.update(status="unsupported", reason=manifest.get("volvox", {}).get("reason", "no exported package"))
                    continue
                try:
                    if engine == "ort":
                        runner = Ort(root, manifest["onnx"], args.ort_provider, args.threads, provider_options, temporary,
                                     allow_cpu_nodes=args.ort_allow_cpu_nodes)
                    else:
                        runner_type = NativePipeline if manifest["volvox"].get("pipeline") else Native
                        runner = runner_type(root, manifest["volvox"], args.backend, args.threads, args.library, args.timeout)
                    result.update(runtime=runner.metadata, setup_ms=runner.setup_ms)
                    result["cases"], outputs[engine] = run_cases(runner, root, manifest,
                        args.warmups, args.repeats, args.rtol, args.atol, args.report_waveform_differences)
                    result.update(status="pass" if all(case["status"] == "pass" for case in result["cases"]) else "numerical_difference", route=runner.route)
                except (ImportError, ProviderUnavailable) as error:
                    result.update(status="skipped", reason=str(error))
                except Exception as error:
                    result["reason"] = f"{type(error).__name__}: {error}"
                    if getattr(error, "report", None) is not None:
                        from google.protobuf.json_format import MessageToDict
                        result["operation_report"] = MessageToDict(error.report, preserving_proto_field_name=True)
                        if "UNSUPPORTED" in result["operation_report"].get("status", ""):
                            result["status"] = "unsupported"
                finally:
                    if runner is not None:
                        if runner.route:
                            result["route"] = runner.route
                        runner.close()
            if set(outputs) == {"volvox", "ort"}:
                comparisons = []
                for case in manifest["cases"]:
                    deterministic = case.get("validation", {}).get("deterministic", True)
                    comparisons.append({"id": case["id"], "outputs": compare_outputs(
                        outputs["volvox"][case["id"]], outputs["ort"][case["id"]],
                        deterministic=deterministic, audio_output=case.get("audio_output", "audio"),
                        rtol=args.rtol, atol=args.atol, waveform=case.get("validation", {}).get("waveform"), report_waveform_differences=args.report_waveform_differences)})
                report["comparison"] = {"status": "pass" if all(c.get("validation", {}).get("deterministic", True)
                    for c in manifest["cases"]) else "not_claimed_stochastic_model", "cases": comparisons}
                if any(output["numerical_parity"] is False for case in comparisons for output in case["outputs"].values()):
                    report["comparison"]["status"] = "numerical_difference"
            statuses = [value["status"] for value in report["engines"].values()]
            if report["comparison"]["status"] == "numerical_difference":
                statuses.append("numerical_difference")
            report["status"] = "pass" if all(s == "pass" for s in statuses) else "failed" if "failed" in statuses else "numerical_difference" if "numerical_difference" in statuses else "partial"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["status"] = "failed"
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["status"] == "pass" else 3 if report["status"] == "numerical_difference" else 2 if report["status"] == "partial" else 1


if __name__ == "__main__":
    raise SystemExit(main())
