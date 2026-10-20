#!/usr/bin/env python3
"""Replay retained HF greedy oracles on a required native GPU; collect no timings."""
from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack
import hashlib
import importlib
import json
import math
from pathlib import Path
import struct
import sys
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


CASE_NAMES = ["deepseek_v2", "deepseek_v3", "deepseek_v4", "llama4_text",
    "qwen3_5_text", "qwen3_5_moe_text", "qwen3_moe", "qwen3_next", "qwen4_exp_text"]
SOURCE_DIRECTORIES = ("native/src", "proto", "shaders/inference", "python/volvoxai",
    "runtime/generated/python")
SOURCE_SUFFIXES = {".c", ".h", ".inc", ".m", ".proto", ".py", ".wgsl"}

def source_snapshot(root):
    result = {}
    for relative in SOURCE_DIRECTORIES:
        directory = root / relative
        require(directory.is_dir(), f"source directory missing: {relative}")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                result[path.relative_to(root).as_posix()] = digest(path)
    for relative in ("native/CMakeLists.txt", "tools/release_profiles.mjs"):
        result[relative] = digest(root / relative)
    return result

def verify_snapshot(tracked, before, root, source_before, receipt):
    after = {label: digest(path) for label, path in tracked.items()}
    source_after = source_snapshot(root)
    receipt["sha256_after"] = after
    receipt["source_after"] = source_after
    require(after == before, "test input, SDK schema, bridge or library changed during replay")
    require(source_after == source_before, "source closure changed during replay")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_ids(tensor, expected_count, name, pb):
    require(tensor.name == name, "wrong generated output name")
    require(tensor.dtype == pb.DataType.DATA_TYPE_I32, "generated output is not I32")
    require(list(tensor.shape) == [1, expected_count], "wrong generated output shape")
    require(not tensor.HasField("buffer") and len(tensor.inline) == expected_count * 4,
            "inline generated output extent mismatch")
    return list(struct.unpack("<" + "i" * expected_count, tensor.inline))


def route(report, pb, backend):
    require(report.status == pb.NativeStatus.NATIVE_STATUS_OK, "operation status failed")
    require(report.backend == backend, "operation did not execute on the required backend")
    require(not report.fallback or not report.fallback.operator_fallback_used,
            "operator fallback was used")
    value = report.route
    require(value and value.builtin and value.attested, "GPU route is not built-in and attested")
    require(value.active_nodes > 0 and value.selected_nodes == value.active_nodes,
            "not every active node selected the required GPU")
    require(value.fallback_nodes == 0 and value.missing_nodes == 0, "incomplete GPU route")
    return {"backend": report.backend, "device": report.device,
            "provider": value.provider, "active_nodes": value.active_nodes,
            "selected_nodes": value.selected_nodes, "fallback_nodes": value.fallback_nodes,
            "missing_nodes": value.missing_nodes, "attested": value.attested,
            "digest": value.digest}


def run_case(case, vx, host, record, backend):
    pb = vx.pb
    engine = vx.VxInferenceServiceClient(host)
    buffers = vx.VxBufferServiceClient(host)
    retained = []
    try:
        with ExitStack() as owners:
            runtime = engine.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
            owners.callback(engine.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id))
            model = engine.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                package=pb.ModelPackage(graph_document=json.dumps(case["graph"], allow_nan=False).encode(),
                    weight_shards=[base64.b64decode(value, validate=True) for value in case["weight_shards"]])))
            owners.callback(engine.release_model, pb.ModelRef(model_id=model.model_id))
            compiled = engine.compile_model(pb.CompileModelRequest(model_id=model.model_id,
                policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                    backends=[backend], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
            owners.callback(engine.release_compiled_model,
                            pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
            record["compile_route"] = route(compiled.report, pb, backend)
            require(compiled.report.compilation.policy_mode == pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                    "compile policy is not REQUIRE")
            require(compiled.report.compilation.operator_fallback == pb.OperatorFallback.OPERATOR_FALLBACK_FORBID,
                    "compile fallback policy is not FORBID")
            initial = []
            for value in case["inputs"]:
                require(value["dtype"] == "int32", "fixture contains an unsupported initial dtype")
                data = base64.b64decode(value["data"], validate=True)
                require(len(data) == 4 * math.prod(value["shape"]), "initial tensor byte extent mismatch")
                initial.append(pb.Tensor(name=value["name"], shape=value["shape"],
                                         dtype=pb.DataType.DATA_TYPE_I32, inline=data))
            name = case["graph"]["execution"]["generation"]["output_name"]
            for storage_name, storage in (("inline", pb.OutputStorage.OUTPUT_STORAGE_INLINE),
                    ("owned", pb.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER)):
                with ExitStack() as context_owner:
                    context = engine.create_execution_context(pb.CreateExecutionContextRequest(
                        compiled_model_id=compiled.compiled_model_id,
                        autoregressive=pb.AutoregressiveExecution(max_steps=3,
                            sampling=pb.SamplingMode.SAMPLING_MODE_GREEDY)))
                    context_owner.callback(engine.release_execution_context,
                                           pb.ExecutionContextRef(context_id=context.context_id))
                    observed = []
                    mode = {"storage": storage_name, "chunks": []}
                    record["modes"].append(mode)
                    for index, count in enumerate(case["budgets"]):
                        packet = engine.generate(pb.GenerateRequest(context_id=context.context_id,
                            initial_inputs=pb.GenerationInputs(values=initial) if index == 0 else None,
                            budget=pb.GenerationBudget(max_steps=count),
                            outputs=pb.OutputOptions(storage=storage)))
                        require(len(packet.outputs) == 1, "unexpected generated output selection")
                        output = packet.outputs[0]
                        if storage_name == "owned":
                            require(output.HasField("buffer") and output.buffer.buffer_id != 0,
                                    "owned generation returned no buffer handle")
                            retained.append((output, count, name))
                        evidence = route(packet.report, pb, backend)
                        require(packet.progress.steps == count and packet.progress.total_steps == sum(case["budgets"][:index+1]),
                                "generation progress mismatch")
                        tensor = (buffers.copy_tensors(pb.CopyTensorsRequest(sources=[output], inline_result=True)).outputs[0]
                                  if storage_name == "owned" else output)
                        chunk = read_ids(tensor, count, name, pb)
                        observed.extend(chunk)
                        require(observed == case["expected_ids"][:len(observed)], "greedy token mismatch")
                        mode["chunks"].append({"count": count, "ids": chunk, "route": evidence})
                    require(observed == case["expected_ids"], "complete greedy sequence mismatch")
                    require(not packet.progress.can_continue and
                            packet.progress.stop_reason == pb.ExecutionStopReason.EXECUTION_STOP_REASON_STEP_LIMIT,
                            "context did not stop at its declared step limit")
                    mode["ids"] = observed
        # All execution/model/runtime owners are now retired. Buffer handles
        # retain their independent output storage until explicitly released.
        after_retirement = []
        for output, count, name in retained:
            copied = buffers.copy_tensors(pb.CopyTensorsRequest(sources=[output], inline_result=True))
            require(len(copied.outputs) == 1, "retained output copy returned a different selection")
            after_retirement.extend(read_ids(copied.outputs[0], count, name, pb))
        require(after_retirement == case["expected_ids"], "owned output changed after owner retirement")
        record["owned_after_context_compiled_model_runtime_release"] = after_retirement
        record["status"] = "passed"
    finally:
        if retained:
            buffers.release_buffers(pb.BufferRefs(buffer_ids=[output.buffer.buffer_id for output, _, _ in retained]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", choices=("vulkan", "opengl", "metal"), required=True)
    parser.add_argument("--pins", type=Path)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "refusing to overwrite an earlier result")
    root = args.source_root.resolve()
    tracked = {"helper": Path(__file__).resolve(), "cases": args.cases.resolve(),
        "library": args.library.resolve(), "proto": root / "proto/volvoxai.proto",
        "generated_schema": root / "runtime/generated/python/volvoxai_pb2.py"}
    bridge = args.bridge or args.library.parent / "libsynurang_module_host.so"
    if bridge.is_file():
        tracked["bridge"] = bridge.resolve()
    pins = json.loads(args.pins.read_text()) if args.pins else None
    receipt = {"format": "volvox-modern-native-gpu-generation-replay/v1", "status": "failed",
        "backend": args.backend, "latency_samples": 0, "service_actions": 0,
        "pins_sha256": digest(args.pins) if args.pins else None, "cases": []}
    before = {}
    source_before = {}
    try:
        before = {label: digest(path) for label, path in tracked.items()}
        source_before = source_snapshot(root)
        receipt["sha256_before"] = before
        receipt["source_before"] = source_before
        if pins:
            require(pins["format"] == "volvox-modern-native-gpu-generation-pins/v1", "unsupported pin format")
            require(pins.get("backend", args.backend) == args.backend, "pin backend mismatch")
            for label, expected in pins["sha256"].items():
                require(label in before and before[label] == expected, f"{label} pin mismatch")
        document = json.loads(args.cases.read_text())
        require(document["format"] == "volvox-hf-generation-fixtures/v1", "unexpected fixture format")
        cases = document["cases"]
        require(len(cases) == 9 and [case["name"] for case in cases] ==
                (pins["case_names"] if pins else CASE_NAMES),
                "fixture case set/order changed")
        for case in cases:
            require(case["budgets"] == [1, 2] and len(case["expected_ids"]) == 3,
                    "fixture generation budget changed")
        sys.path[:0] = [str(root / "python"), str(root / "runtime/generated/python")]
        vx = importlib.import_module("volvoxai")
        require(Path(vx.__file__).resolve().is_relative_to(root / "python"), "SDK resolved outside source root")
        require(digest(vx.pb.__file__) == before["generated_schema"], "loaded schema differs from pin")
        with vx.open_library(args.library.resolve()) as host:
            for case in cases:
                record = {"name": case["name"], "expected_ids": case["expected_ids"], "modes": [], "status": "failed"}
                receipt["cases"].append(record)
                run_case(case, vx, host, record, args.backend)
        verify_snapshot(tracked, before, root, source_before, receipt)
        receipt["counts"] = {"families": 9, "contexts": 18, "generate_calls": 36,
            "exact_greedy_ids": 54, "route_attestations": 45,
            "owned_tensors_after_all_owner_retirement": 18}
        receipt["status"] = "passed"
    except BaseException as error:
        receipt["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        if before and "sha256_after" not in receipt:
            try:
                verify_snapshot(tracked, before, root, source_before, receipt)
            except BaseException as verification_error:
                receipt["verification_error"] = {"type": type(verification_error).__name__,
                    "message": str(verification_error)}
                receipt["status"] = "failed"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": receipt["status"], "backend": args.backend,
            "counts": receipt.get("counts"), "result_sha256": digest(args.out)}, allow_nan=False))


if __name__ == "__main__":
    main()
