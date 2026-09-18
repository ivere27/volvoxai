#!/usr/bin/env python3
"""Retained-session receipt latency through generated proto clients or ORT.

Inputs and expected outputs are supplied by a private benchmark specification.
No model loading, preprocessing, correctness assertion or file I/O is included
in component timers. The request timer additionally includes explicit-KV greedy
decoding and input construction. This driver never interprets wall time from an
accuracy campaign as inference latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
import volvoxai as vx

pb = vx.pb
ENCODER_OUTPUTS = tuple(f"cross_{kind}_{layer}" for layer in range(4) for kind in ("k", "v")) + (
    "memory_padding_mask", "selected_family_ids")
CACHE_OUTPUTS = tuple(f"present_{kind}_{layer}" for layer in range(4) for kind in ("k", "v")) + (
    "present_padding_mask",)
NEEDED_OUTPUTS = {"digit": ("slot_logits",), "encoder": ENCODER_OUTPUTS, "decoder": ("logits", *CACHE_OUTPUTS)}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def signature(value):
    return digest(json.dumps(value, separators=(",", ":")).encode())


def route(report, backend):
    r = report.route
    assert report.backend == backend and r.provider == backend and r.attested
    assert r.active_nodes == r.selected_nodes and r.missing_nodes == r.fallback_nodes == 0
    assert not (report.fallback and report.fallback.operator_fallback_used)
    assert not (report.compilation and report.compilation.tier_fallback_used)
    return {"backend": backend, "attested": True, "active_nodes": r.active_nodes,
            "selected_nodes": r.selected_nodes, "fallback_nodes": 0, "missing_nodes": 0}


def summarize(values):
    ordered = sorted(values)
    def quantile(q):
        at = (len(ordered) - 1) * q
        lo = int(at)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (at - lo)
    return {"n": len(values), "median_ms": statistics.median(values), "p95_ms": quantile(.95),
            "mean_ms": statistics.mean(values), "min_ms": min(values), "max_ms": max(values)}


class Native:
    def __init__(self, spec, model, variant, backend, profile):
        self.contracts = spec["models"][model]["variants"][variant]
        self.backend, self.contexts, self.routes = backend, {}, {}
        self.native, self.pending_reports, self.pending_timings = True, [], []
        self.output_buffers = {}
        self.host = vx.open_library(profile)
        self.client = vx.VxInferenceServiceClient(self.host)
        try:
            runtime = self.client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
            for role, contract in self.contracts.items():
                graph = (ROOT / contract["graph"]).read_bytes()
                weights = (ROOT / contract["weights"]).read_bytes()
                assert digest(graph) == spec["file_hashes"][contract["graph"]]
                assert digest(weights) == spec["file_hashes"][contract["weights"]]
                loaded = self.client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                    package=pb.ModelPackage(graph_document=graph, weight_shards=[weights])))
                info = self.client.get_model_info(pb.ModelRef(model_id=loaded.model_id))
                types = {pb.DataType.DATA_TYPE_F32: np.float32, pb.DataType.DATA_TYPE_I32: np.int32}
                needed = {contract["outputs"][name] for name in NEEDED_OUTPUTS[role]}
                self.output_buffers[role] = {output.name: np.empty(
                    math.prod(axis.max for axis in output.dimensions), dtype=types[output.dtype])
                    for output in info.outputs if output.name in needed}
                assert self.output_buffers[role].keys() == needed
                compiled = self.client.compile_model(pb.CompileModelRequest(model_id=loaded.model_id,
                    policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                        backends=[backend], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
                self.routes[role] = {"compilation": route(compiled.report, backend)}
                self.contexts[role] = self.client.create_execution_context(pb.CreateExecutionContextRequest(compiled_model_id=compiled.compiled_model_id)).context_id
        except BaseException:
            self.close()
            raise

    def run(self, role, feed, *, need_cache=True):
        contract = self.contracts[role]
        # Keep borrowed arrays alive through completion. BorrowedBuffer avoids
        # encoding the same cross-attention memory into protobuf at every step.
        buffers = {name: np.ascontiguousarray(value) for name, value in feed.items()}
        inputs = [pb.Tensor(name=contract["inputs"][name], shape=list(value.shape),
                    dtype=pb.DataType.DATA_TYPE_F32 if value.dtype.kind == "f" else pb.DataType.DATA_TYPE_I32,
                    borrowed=pb.BorrowedBuffer(resource=pb.NativeResource(kind=pb.NativeResourceKind.NATIVE_RESOURCE_KIND_HOST,
                        handle=value.ctypes.data, size_bytes=value.nbytes), length_bytes=value.nbytes)) for name, value in buffers.items()]
        request = pb.ExecuteRequest(context_id=self.contexts[role], inputs=inputs)
        started = time.perf_counter_ns()
        execution = self.client.execute(request)
        get_result_calls, read_output_calls, read_output_ns = 0, 0, 0
        report = execution.report
        try:
            deadline = time.monotonic() + 120
            state = execution.state
            while state == pb.ResultState.RESULT_STATE_PENDING:
                if time.monotonic() >= deadline:
                    raise TimeoutError("GPU result timed out")
                info = self.client.get_result(pb.ResultRef(result_id=execution.result_id))
                get_result_calls += 1
                state, report = info.state, info.report
                if state == pb.ResultState.RESULT_STATE_PENDING:
                    time.sleep(.001)
            assert state == pb.ResultState.RESULT_STATE_READY
            execute_ms = (time.perf_counter_ns() - started) / 1e6

            def read(semantic):
                nonlocal read_output_calls, read_output_ns
                name = contract["outputs"][semantic]
                buffer = self.output_buffers[role][name]
                before = time.perf_counter_ns()
                response = self.client.read_output(pb.ReadOutputRequest(
                    result_id=execution.result_id, name=name,
                    into=pb.BorrowedBuffer(resource=pb.NativeResource(kind=pb.NativeResourceKind.NATIVE_RESOURCE_KIND_HOST,
                        handle=buffer.ctypes.data, size_bytes=buffer.nbytes), length_bytes=buffer.nbytes)))
                read_output_ns += time.perf_counter_ns() - before
                read_output_calls += 1
                # The next Execute consumes the old cache before these reusable
                # destinations are written. No caller retains a historical view.
                count = math.prod(response.tensor.shape)
                assert response.required_bytes == count * buffer.itemsize <= buffer.nbytes
                assert response.tensor.borrowed is not None and response.tensor.borrowed.resource.handle == buffer.ctypes.data
                return buffer[:count].reshape(response.tensor.shape)

            output = read_needed_outputs(role, read, need_cache=need_cache)
        finally:
            release_started = time.perf_counter_ns()
            self.client.release_result(pb.ResultRef(result_id=execution.result_id))
            release_ms = (time.perf_counter_ns() - release_started) / 1e6
        elapsed = (time.perf_counter_ns() - started) / 1e6
        self.pending_reports.append((role, report))
        self.pending_timings.append((role, {
            "execute_ms": execute_ms, "read_output_ms": read_output_ns / 1e6,
            "release_result_ms": release_ms,
            "runtime_execution_ms": execution.report.timings.execution_time_ms,
            "execute_calls": 1, "get_result_calls": get_result_calls,
            "read_output_calls": read_output_calls, "release_result_calls": 1}))
        return output, elapsed

    def validate(self):
        for role, report in self.pending_reports:
            self.routes[role]["execution"] = route(report, self.backend)
        self.pending_reports.clear()

    def close(self):
        self.host.close()


class Ort:
    def __init__(self, spec, model, variant):
        import onnxruntime as ort
        self.contracts = spec["models"][model]["variants"][variant]
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        for path in spec["models"][model]["onnx"][variant].values():
            assert digest((ROOT / path).read_bytes()) == spec["file_hashes"][path]
        self.sessions = {role: ort.InferenceSession(str(ROOT / path), options,
                         providers=["CPUExecutionProvider"])
                         for role, path in spec["models"][model]["onnx"][variant].items()}
        self.version, self.routes = ort.__version__, {}
        self.native = False

    def validate(self):
        pass

    def run(self, role, feed, *, need_cache=True):
        session = self.sessions[role]
        types = {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(bool)": np.bool_}
        names = {"image": session.get_inputs()[0].name} if role == "digit" else {k: k for k in feed}
        actual = {names[k]: value for k, value in feed.items()}
        inputs = {p.name: np.asarray(actual[p.name], dtype=types[p.type]) for p in session.get_inputs()}
        started = time.perf_counter_ns()
        values = session.run(None, inputs)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        output = {p.name: np.asarray(v, dtype=np.float32 if v.dtype.kind == "f" else np.int32)
                  for p, v in zip(session.get_outputs(), values)}
        if role == "digit":
            output = {"slot_logits": next(iter(output.values()))}
        return output, elapsed

    def close(self):
        self.sessions.clear()


def decoder_feed(encoded, native):
    feed = {k: v for k, v in encoded.items() if k.startswith("cross_") or k == "memory_padding_mask"}
    past = 1 if native else 0
    feed.update(decoder_input_ids=np.asarray([[1]], np.int32), position_ids=np.asarray([0], np.int32),
                family_ids=encoded["selected_family_ids"], past_padding_mask=np.ones((1, past), np.int32))
    for layer in range(4):
        for kind in ("k", "v"):
            feed[f"past_{kind}_{layer}"] = np.zeros((1, 8, past, 40), np.float32)
    return feed


def read_needed_outputs(role, read, *, need_cache=True):
    if role == "digit":
        return {"slot_logits": read("slot_logits")}
    if role == "encoder":
        return {name: read(name) for name in ENCODER_OUTPUTS}
    output = {"logits": read("logits")}
    # EOS and a benchmark's final capped step have no following cache consumer.
    if need_cache and int(output["logits"].reshape(-1).argmax()) != 2:
        output.update((name, read(name)) for name in CACHE_OUTPUTS)
    return output


def execution_metrics(runner):
    timings = getattr(runner, "pending_timings", [])
    output = {}
    for role, values in timings:
        for name, value in values.items():
            output[name] = output.get(name, 0) + value
            if role != "digit":
                key = f"{role}_{name}"
                output[key] = output.get(key, 0) + value
    timings.clear()
    return output


def request(runner, model, case, max_new_tokens=191, require_eos=True):
    if model == "digit":
        started = time.perf_counter_ns()
        output, elapsed = runner.run("digit", {"image": case["pixels"]})
        total = (time.perf_counter_ns() - started) / 1e6
        assert np.isfinite(output["slot_logits"]).all()
        return output["slot_logits"].reshape(16, 11).argmax(-1).tolist(), {
            "request_ms": total, "component_ms": elapsed, **execution_metrics(runner)}
    ids = np.asarray([case["question_ids"]], np.int32)
    feed = {"image": case["pixels"], "question_ids": ids,
            "question_position_ids": np.arange(ids.shape[1], dtype=np.int32)[None],
            "family_ids": np.asarray([case.get("family_id", -1)], np.int32)}
    started = time.perf_counter_ns()
    encoded, enc_ms = runner.run("encoder", feed)
    dec_feed = decoder_feed(encoded, runner.native or case.get("masked_zero_sentinel", False))
    tokens, decoder_ms, ttft = [1], [], None
    for position in range(max_new_tokens):
        output, elapsed = runner.run("decoder", dec_feed, need_cache=position + 1 < max_new_tokens)
        decoder_ms.append(elapsed)
        token = int(output["logits"].reshape(-1).argmax())
        tokens.append(token)
        if ttft is None:
            ttft = (time.perf_counter_ns() - started) / 1e6
        if token == 2 or position + 1 == max_new_tokens:
            break
        for layer in range(4):
            for kind in ("k", "v"):
                dec_feed[f"past_{kind}_{layer}"] = output[f"present_{kind}_{layer}"]
        dec_feed["past_padding_mask"] = output["present_padding_mask"]
        dec_feed["decoder_input_ids"] = np.asarray([[token]], np.int32)
        dec_feed["position_ids"] = np.asarray([position + 1], np.int32)
    total = (time.perf_counter_ns() - started) / 1e6
    assert (not require_eos or tokens[-1] == 2) and np.isfinite(output["logits"]).all()
    return [tokens, int(encoded["selected_family_ids"][0])], {
        "request_ms": total, "component_ms": enc_ms + sum(decoder_ms), "encoder_ms": enc_ms,
        "decoder_ms": sum(decoder_ms), "ttft_ms": ttft, "decoder_steps": len(decoder_ms),
        "decode_ms_per_token": sum(decoder_ms[1:]) / max(1, len(decoder_ms) - 1),
        **execution_metrics(runner)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--model", choices=["digit", "vqa"], required=True)
    parser.add_argument("--variant", choices=["fp32", "int8", "ptq", "ptq-wasm"], required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda", "vulkan", "opengl", "ort"], required=True)
    parser.add_argument("--profile", choices=["inference", "full"], default="inference")
    parser.add_argument("--warmup", type=int, default=3, help="untimed complete requests per input")
    parser.add_argument("--repeat", type=int, default=15, help="measured complete requests per input")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ort-reference", type=Path,
                        help="independent current-host VQA INT8 reference; retain the original 2000-case comparison")
    args = parser.parse_args()
    assert args.warmup >= 1 and args.repeat >= 1
    assert args.backend != "ort" or args.variant in ["fp32", "int8"]
    spec_bytes = args.spec.read_bytes()
    spec = json.loads(spec_bytes)
    prefix = spec.get("reference_kind") == "historical-four-token-prefix"
    generation = spec.get("generation", {"max_new_tokens": 191, "require_eos": True})
    assert generation == ({"max_new_tokens": 4, "require_eos": False} if prefix else
                          {"max_new_tokens": 191, "require_eos": True})
    assert not prefix or args.model == "vqa"
    cases = []
    for item in spec["cases"]:
        raw = (ROOT / item["pixels"]).read_bytes()
        assert digest(raw) == item["sha256"] and len(raw) == 320 * 672 * 4
        cases.append(dict(item, pixels=np.frombuffer(raw, dtype=np.float32).reshape(1, 1, 320, 672)))
    key = "/".join([args.model, args.backend, args.profile, args.variant])
    expected = spec["expected"].get(key)
    assert expected is not None, "the frozen 2000-case reference is required"
    original_expected, reference = expected, None
    if args.ort_reference:
        assert (args.model, args.backend, args.variant) == ("vqa", "ort", "int8")
        reference = json.loads(args.ort_reference.read_text())
        assert reference["schema"] == "volvoxai.receipt-ort-benchmark-reference/v1"
        assert reference["spec_sha256"] == digest(spec_bytes) and reference["repeated_outputs_identical"]
        assert reference["case_indices"] == [c["index"] for c in cases]
        assert reference["file_hashes"] == {path: spec["file_hashes"][path]
                                           for path in spec["models"]["vqa"]["onnx"]["int8"].values()}
        expected = reference["expected"]
    runner = Ort(spec, args.model, args.variant) if args.backend == "ort" else Native(
        spec, args.model, args.variant, args.backend, args.profile)
    samples, signatures = [], {}
    try:
        if reference:
            assert (reference["onnxruntime"], reference["numpy"], reference["python"]) == (
                runner.version, np.__version__, platform.python_version())
        for iteration in range(args.warmup + args.repeat):
            order = list(range(len(cases)))
            if iteration % 2:
                order.reverse()
            for index in order:
                actual, timing = request(runner, args.model, cases[index], **generation)
                runner.validate()
                sig = signature(actual)
                if expected is not None:
                    assert actual == expected[str(cases[index]["index"])], "output differs from frozen 2000-case reference"
                if index in signatures:
                    assert signatures[index] == sig, "output changed between benchmark requests"
                signatures[index] = sig
                if iteration >= args.warmup:
                    samples.append(dict(timing, case_index=cases[index]["index"]))
        metrics = {name: summarize([s[name] for s in samples]) for name in samples[0]
                   if name.endswith("_ms") or name == "decode_ms_per_token"}
        report = {"schema": "volvoxai.receipt-proto-prefix-latency/v1" if prefix else "volvoxai.receipt-proto-latency/v1", "status": "measured",
                  "host_quiet_verified": False,
                  "base_head": spec["base_head"], "model": args.model, "backend": args.backend,
                  "profile": args.profile, "variant": args.variant, "cpu_threads": 1,
                  "warmup_per_case": args.warmup, "repeat_per_case": args.repeat,
                  "case_indices": [c["index"] for c in cases], "order": "alternating-forward-reverse",
                  "spec_sha256": digest(spec_bytes), "harness_sha256": digest(Path(__file__).read_bytes()),
                  "python": platform.python_version(), "numpy": np.__version__,
                  "driver": "python-ort" if args.backend == "ort" else "python-proto-native-c",
                  "output_matches_2000_reference": not prefix and expected == original_expected,
                  "output_matches_benchmark_reference": True,
                  "repeated_outputs_identical": True, "output_sha256": signatures,
                  "metrics": metrics, "samples": samples, "routes": runner.routes,
                  "serial_requests_per_second": 1000 * len(samples) / sum(s["request_ms"] for s in samples)}
        if prefix:
            report.update(reference_kind=spec["reference_kind"], generation=generation,
                          historical_reference=spec["historical_reference"],
                          stopped_at_eos=False, complete_answer=False)
        if args.model == "vqa":
            report["serial_generated_tokens_per_second"] = 1000 * sum(s["decoder_steps"] for s in samples) / sum(s["request_ms"] for s in samples)
        if args.backend == "ort":
            report["onnxruntime"] = runner.version
            if reference:
                report["independent_ort_reference"] = {k: v for k, v in reference.items() if k != "expected"}
                report["independent_ort_reference"]["sha256"] = digest(args.ort_reference.read_bytes())
        else:
            report["library_sha256"] = digest(vx.find_library(args.profile).read_bytes())
            report["timing_contract"] = {
                "execute_ms": "Execute through READY, including any pending-result polling; excludes ReadOutput and ReleaseResult",
                "runtime_execution_ms": "OperationReport execution interval, including runtime binding/synchronization/output snapshots; not GPU kernel-only time",
                "read_output_ms": "needed ReadOutput calls into caller-owned native buffers",
                "component_ms": "execution, needed output reads/materialization/selection and result release; excludes input construction",
                "request_ms": "complete request including input construction, token selection and cache rebinding",
                "tensor_transport": "native BorrowedBuffer inputs and output destinations; no inline tensor payloads"}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"route": key, "measured_requests": len(samples), "request": metrics["request_ms"]}))
    finally:
        runner.close()


if __name__ == "__main__":
    main()
