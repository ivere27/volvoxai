#!/usr/bin/env python3
"""Compare a native full-prefix decoder with a frozen independent HF capture.

This development harness uses generated engine calls through InferenceSession.
It neither implements model operators nor qualifies idle-host performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from tools.llm_reference import POLICY_PATH, validate_reference_policy


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def source_provenance(manifest_path: Path | None) -> dict:
    if manifest_path is None:
        return {"source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_dirty": subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()}
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "volvox-llama-source-freeze/v1":
        raise ValueError("unsupported source freeze manifest")
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe source manifest path")
        source = ROOT / relative
        if source.stat().st_size != entry["bytes"] or sha256_file(source) != entry["sha256"]:
            raise ValueError(f"source differs from its frozen manifest: {relative}")
    return {"source_manifest": identity(manifest_path),
            "source_base_commit": manifest["base_commit"],
            "source_files_verified": len(manifest["files"]),
            "git_metadata_available": (ROOT / ".git").exists()}


def compare(actual: np.ndarray, expected: np.ndarray, tolerance: dict) -> dict:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"shape/dtype mismatch: {actual.shape}/{actual.dtype} != "
                         f"{expected.shape}/{expected.dtype}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("comparison contains nonfinite values")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(delta)))
    rms = float(np.sqrt(np.mean(delta * delta)))
    passed = bool(np.allclose(actual, expected, atol=tolerance["absolute"],
                              rtol=tolerance["relative"]))
    if "rms_max" in tolerance:
        passed = passed and rms <= tolerance["rms_max"]
    return {"max_absolute": maximum, "rms": rms, "pass": passed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH,
                        help="pinned model and numerical policy used by the HF capture")
    parser.add_argument("--profile", choices=("inference", "full"), required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--case", help="select one recorded case for development")
    parser.add_argument("--source-manifest", type=Path,
                        help="verify exact source in a release snapshot without Git metadata")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.out.exists():
        parser.error("threads must be positive and output must be a new file")
    policy_path = args.policy
    manifest_path = args.reference / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    policy, pinned = validate_reference_policy(manifest, policy_path)
    graph_path, weight_path = args.package / "graph.json", args.package / "model.safetensors"
    graph = json.loads(graph_path.read_text())
    if graph["dimensions"]["S"] != {"min": 1, "max": policy["reference"]["qualification"]["max_context"]}:
        raise ValueError("package context bound differs from the fixed qualification scope")
    config = manifest["config"]
    layers, kv_heads, head_dim = config["num_hidden_layers"], config["num_key_value_heads"], pinned["head_dim"]
    required_outputs = ["logits"] + [name for layer in range(layers)
        for name in (f"layer.{layer}.k_rope", f"layer.{layer}.v")]
    if set(graph["outputs"]) != set(required_outputs):
        raise ValueError("comparison requires the complete logits and compact layer KV capture graph")
    expected_inputs = {"token_ids", "position_ids", "attention_mask"}
    if set(graph["inputs"]) != expected_inputs:
        raise ValueError("package input contract differs from the decoder capture contract")
    qualification = policy["reference"]["qualification"]
    expected_cases = {f"prompt{index}-prefix{prefix:03d}": (index, prefix, prompt)
        for index, prompt in enumerate(qualification["prompts"])
        for prefix in qualification["prefix_lengths"]}
    actual_ids = [case["id"] for case in manifest["cases"]]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_cases):
        raise ValueError("reference must contain exactly the fixed prompt/prefix matrix")
    for case in manifest["cases"]:
        metadata = (case["prompt_index"], case["prefix_length"], case["prompt"])
        if metadata != expected_cases[case["id"]] or len(case["input_token_ids"]) != case["prefix_length"]:
            raise ValueError("reference case metadata differs from the fixed matrix")
    selected = [case for case in manifest["cases"] if args.case is None or case["id"] == args.case]
    if not selected:
        raise ValueError("the selected case is absent from the fixed reference")
    for case in selected:
        path = args.reference / case["reference_path"]
        if sha256_file(path) != case["reference_sha256"] or path.stat().st_size != case["reference_bytes"]:
            raise ValueError(f"reference payload differs from its manifest: {case['id']}")
        input_path = args.reference / case["engine_input_ids"]["path"]
        if sha256_file(input_path) != case["engine_input_ids"]["sha256"]:
            raise ValueError("recorded input bytes differ from their manifest")

    import volvoxai as vx
    from volvoxai._clients import VxProfilingServiceClient
    from volvoxai._diagnostics import _proto_json
    library = ROOT / "native" / ("libvolvoxai-lite.so" if args.profile == "inference" else "libvolvoxai.so")
    # This is an explicit process-local profile selection, never a fallback.
    os.environ["VOLVOXAI_LIBRARY"] = str(library.resolve())
    tolerance = policy["reference"]["engine_tolerance"]
    # Admission reserves selected outputs over the declared maximum shape,
    # even when the first concrete request is one token. Include all compact
    # KV captures and full logits, with one outstanding result per session.
    result_budget_bytes = qualification["max_context"] * 4 * (
        config["vocab_size"] + 2 * layers * kv_heads * head_dim)
    report = {"format": "volvox-native-hf-comparison/v1", "status": "running",
        "model": manifest["model"], "revision": pinned["revision"], "backend": args.backend,
        "profile": args.profile, "cpu_threads": args.threads, "execution": "full-prefix",
        "kv_attention": "compact-direct" if any(node.get("params", {}).get("kv_heads")
            for node in graph["nodes"]) else "materialized-repeat-kv", "cached_decode_qualified": False,
        "engine_dtype": "float32", "source_dtype": pinned["source_dtype"],
        "reference": identity(manifest_path), "policy": identity(policy_path),
        "graph": identity(graph_path), "weights": identity(weight_path), "library": identity(library),
        "harness": identity(Path(__file__)), **source_provenance(args.source_manifest),
        "numpy_version": np.__version__, "command": sys.argv,
        "capture_enabled": True, "qualifies_performance": False, "tolerance": tolerance,
        "runtime_budget": {"max_unconsumed_result_bytes": result_budget_bytes},
        "cases": [], "resource_snapshots": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def persist() -> None:
        report["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["peak_process_rss_scope"] = "Linux whole process lifetime; includes imports, snapshots and validation"
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def check(value: dict, label: str) -> None:
        if not value["pass"]:
            raise ValueError(f"fixed numerical tolerance exceeded: {label}: {value}")

    try:
        persist()
        with vx.Runtime(cpu_threads=args.threads, budget=vx.pb.RuntimeBudget(
                max_unconsumed_result_bytes=result_budget_bytes)) as runtime:
            profiling = VxProfilingServiceClient(runtime._owner.host)

            def snapshot(phase: str) -> None:
                reply = profiling.get_resource_snapshot(vx.pb.GetResourceSnapshotRequest(
                    runtime_id=runtime._runtime.runtime_id, include_device=args.backend == "cuda"))
                report["resource_snapshots"].append({"phase": phase, "snapshot": _proto_json(reply.snapshot)})

            snapshot("before_model")
            with runtime.inference_session(args.package, backend=args.backend) as session:
                expected_profile = (vx.pb.BuildProfile.BUILD_PROFILE_INFERENCE if args.profile == "inference"
                                    else vx.pb.BuildProfile.BUILD_PROFILE_FULL)
                if session.build_profile != expected_profile:
                    raise ValueError("loaded engine profile differs from the selected profile")
                if session.backend != args.backend:
                    raise ValueError("compiled backend differs from the required backend")
                report["compiled_backend"] = session.backend
                report["compile_report"] = _proto_json(session.report)
                snapshot("loaded_compiled_context")
                for case in selected:
                    entry = {"id": case["id"], "prefix_length": case["prefix_length"], "steps": []}
                    report["cases"].append(entry)
                    with safe_open(args.reference / case["reference_path"], framework="numpy") as reference:
                        ids = reference.get_tensor("prefill.input_ids").astype(np.int32)
                        if hashlib.sha256(ids.tobytes()).hexdigest() != case["engine_input_ids"]["sha256"]:
                            raise ValueError("reference inputs disagree with their fixed engine bytes")
                        for step in range(policy["reference"]["qualification"]["decode_steps"] + 1):
                            phase = "prefill" if step == 0 else f"decode.{step - 1:03d}"
                            if step:
                                next_id = reference.get_tensor(f"decode.{step - 1:03d}.input_ids").astype(np.int32)
                                if int(outputs["logits"][0, -1].argmax()) != int(next_id[0, 0]):
                                    raise ValueError("native caller greedy loop differs from the fixed reference IDs")
                                ids = np.concatenate((ids, next_id), axis=1)
                            positions = np.arange(ids.shape[1], dtype=np.int32)
                            input_phase = "prefill" if step == 0 else f"prefix.{step - 1:03d}"
                            recorded_ids = reference.get_tensor(input_phase + ".input_ids")
                            recorded_positions = reference.get_tensor(input_phase + ".position_ids")[0]
                            mask = reference.get_tensor(input_phase + ".attention_mask")
                            if not np.array_equal(ids, recorded_ids) or not np.array_equal(
                                    positions, recorded_positions):
                                raise ValueError("full-prefix inputs differ from the captured tensors")
                            if mask.dtype != np.dtype("int64") or mask.shape != ids.shape or not np.isin(mask, (0, 1)).all():
                                raise ValueError("captured attention mask violates the fixed input contract")
                            inputs = {"token_ids": ids, "position_ids": positions,
                                "attention_mask": mask.astype(np.int32)}
                            outputs = session.run(inputs, output_names=required_outputs)
                            expected_logits = reference.get_tensor("prefill.logits" if step == 0
                                else f"prefix.{step - 1:03d}.logits")
                            actual_logits = outputs["logits"] if step == 0 else outputs["logits"][:, -1:, :]
                            values = compare(actual_logits, expected_logits, tolerance["logits"])
                            record = {"phase": phase, "inputs": {name: {
                                "shape": list(array.shape), "dtype": str(array.dtype),
                                "bytes": array.nbytes, "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
                                for name, array in inputs.items()}, "logits": values,
                                "greedy_exact": bool(np.array_equal(actual_logits.argmax(-1), expected_logits.argmax(-1))),
                                "layers": []}
                            entry["steps"].append(record)
                            check(values, case["id"] + ":" + phase + ":logits")
                            if not record["greedy_exact"]:
                                raise ValueError("native logits argmax differs from HF")
                            for layer in range(layers):
                                row = {"layer": layer}
                                for suffix, kind in (("k_rope", "key"), ("v", "value")):
                                    actual = outputs[f"layer.{layer}.{suffix}"]
                                    if actual.shape != (1, ids.shape[1], kv_heads * head_dim):
                                        raise ValueError("native compact KV output has the wrong logical shape")
                                    actual = actual.reshape(1, ids.shape[1], kv_heads, head_dim).transpose(0, 2, 1, 3)
                                    expected = reference.get_tensor(f"{phase}.layer.{layer:03d}.{kind}")
                                    row[kind] = compare(actual, expected, tolerance["layer_kv"])
                                    check(row[kind], f"{case['id']}:{phase}:layer.{layer}:{kind}")
                                record["layers"].append(row)
                    entry["status"] = "pass"
                    snapshot("after_" + case["id"])
                    persist()
                    print(f"PASS {manifest['model']} {args.profile} {case['id']}", flush=True)
                del outputs
            snapshot("after_session_close")
            baseline_memory = report["resource_snapshots"][0]["snapshot"]["memory"]
            final_memory = report["resource_snapshots"][-1]["snapshot"]["memory"]
            if ("unconsumedResultBytes" not in final_memory
                    or int(final_memory["unconsumedResultBytes"]) != 0):
                raise ValueError("session close did not return unconsumed result bytes")
            def known_allocations(value: dict) -> list:
                return sorted(value.get("allocations", []),
                    key=lambda allocation: json.dumps(allocation, sort_keys=True))
            if final_memory.get("truncated", False) or baseline_memory.get("truncated", False):
                raise ValueError("cleanup inventory is truncated")
            if known_allocations(final_memory) != known_allocations(baseline_memory):
                raise ValueError("session close did not restore its known allocation inventory")
            report["known_inventory_cleanup"] = {"pass": True,
                "unconsumed_result_bytes": 0,
                "allocation_count": len(known_allocations(final_memory)),
                "scope": "known runtime inventory only; excludes unobserved allocations and whole-process peaks"}
            capacities = ("retainedResultCapacityBytes", "idleResultCapacityBytes")
            if any(name not in final_memory for name in capacities):
                report["result_storage_after_close"] = {"coverage": "unknown",
                    "reason": "optional result capacities are absent in the runtime inventory after its contexts retire"}
            elif any(int(final_memory[name]) for name in capacities):
                raise ValueError("session close retained result storage")
            else:
                report["result_storage_after_close"] = {"coverage": "observed", "bytes": 0}
        report["status"] = "pass"
        report["qualification"] = "numerical full-prefix execution; memory inventory coverage remains explicit"
        report["complete_scope"] = args.case is None
        persist()
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        persist()
        raise
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]),
                      "report": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
