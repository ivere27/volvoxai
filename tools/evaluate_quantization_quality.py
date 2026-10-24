#!/usr/bin/env python3
"""Measure fixed teacher-forced language-model and multiple-choice quality.

Policies contain explicit token IDs; no prompt, sample, or mode is selected from
results. Prefill evaluates every scored causal row in one call. Incremental mode
is available for checking the same teacher-forcing alignment against decode.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
import time
import tempfile
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "python"))
sys.path.insert(0, str(root / "runtime/generated/python"))
import volvoxai as vx
from volvoxai._library import open_library
from volvoxai._session import _resolve_packed_arithmetic

pb = vx.pb


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def softmax_log_prob(logits):
    """Normalize finite logits in F64, independently for each final-axis row."""
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim not in (1, 2) or x.shape[-1] == 0 or not np.isfinite(x).all():
        raise ValueError("Logits must be a finite nonempty vector or matrix")
    shifted = x - np.max(x, axis=-1, keepdims=True)
    normalizer = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_p = shifted - normalizer
    return log_p, np.exp(log_p)


def kl_divergence(p, log_p, log_q):
    return float(np.sum(np.asarray(p) * (np.asarray(log_p) - np.asarray(log_q))))


def score_rows(logits, targets, reference=None):
    rows = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if rows.ndim != 2 or targets.shape != (len(rows),) or not len(rows):
        raise ValueError("One nonempty logits row is required per target token")
    if np.any(targets < 0) or np.any(targets >= rows.shape[1]):
        raise ValueError("A target token is outside the model vocabulary")
    log_q, _ = softmax_log_prob(rows)
    nll = -log_q[np.arange(len(rows)), targets]
    report = {
        "token_count": len(rows), "nll_sum": float(nll.sum()),
        "mean_nll": float(nll.mean()), "perplexity": float(np.exp(nll.mean())),
        "target_top1_correct": int(np.count_nonzero(rows.argmax(axis=1) == targets)),
        "target_top1_accuracy_pct": float(np.mean(rows.argmax(axis=1) == targets) * 100),
    }
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64)
        if ref.shape != rows.shape:
            raise ValueError("Reference and candidate logits shapes differ")
        log_p, p = softmax_log_prob(ref)
        reference_nll = -log_p[np.arange(len(ref)), targets]
        differences = rows - ref
        kls = np.sum(p * (log_p - log_q), axis=1)
        # Identical distributions can differ by a few ulps in the summed KL.
        if np.any(kls < -1e-10):
            raise ValueError("Invalid negative KL divergence")
        kls = np.maximum(kls, 0)
        nrms = np.sqrt(np.mean(differences ** 2, axis=1)) / np.maximum(
            np.sqrt(np.mean(ref ** 2, axis=1)), 1e-20)
        report.update({
            "delta_nll": float(np.mean(nll - reference_nll)),
            "kl_sum_nats": float(kls.sum()), "mean_kl_nats": float(kls.mean()),
            "top1_agreement_count": int(np.count_nonzero(rows.argmax(axis=1) == ref.argmax(axis=1))),
            "top1_agreement_pct": float(np.mean(rows.argmax(axis=1) == ref.argmax(axis=1)) * 100),
            "worst_max_abs": float(np.abs(differences).max()),
            "nrms_sum": float(nrms.sum()), "mean_nrms": float(nrms.mean()),
        })
    return report


def aggregate_language_metrics(case_scores, cases):
    """Aggregate tokens, never average case perplexities or score answer options."""
    selected = [case_scores[key] for key, case in cases.items() if not case.get("task")]
    if not selected:
        return None
    tokens = sum(s["token_count"] for s in selected)
    result = {"case_count": len(selected), "token_count": tokens,
              "mean_nll": sum(s["nll_sum"] for s in selected) / tokens,
              "target_top1_accuracy_pct": 100 * sum(s["target_top1_correct"] for s in selected) / tokens}
    result["perplexity"] = math.exp(result["mean_nll"])
    if "kl_sum_nats" in selected[0]:
        result.update({
            "delta_nll": sum(s["delta_nll"] * s["token_count"] for s in selected) / tokens,
            "mean_kl_nats": sum(s["kl_sum_nats"] for s in selected) / tokens,
            "top1_agreement_pct": 100 * sum(s["top1_agreement_count"] for s in selected) / tokens,
            "worst_max_abs": max(s["worst_max_abs"] for s in selected),
            "mean_nrms": sum(s["nrms_sum"] for s in selected) / tokens,
        })
    return result


def score_multiple_choice(case_scores, cases):
    grouped = {}
    for key, case in cases.items():
        task = case.get("task")
        if task:
            grouped.setdefault(task["id"], []).append((task, case_scores[key]))
    if not grouped:
        return None
    details = []
    for task_id, choices in grouped.items():
        choices.sort(key=lambda item: item[0]["choice_index"])
        expected = list(range(choices[0][0]["choice_count"]))
        if [task["choice_index"] for task, _ in choices] != expected:
            raise ValueError(f"Incomplete or duplicate choices for {task_id}")
        answers = {task["answer_index"] for task, _ in choices}
        if len(answers) != 1 or next(iter(answers)) not in expected:
            raise ValueError(f"Invalid answer for {task_id}")
        raw = [-score["nll_sum"] for _, score in choices]
        normalized = [value / task["normalization_chars"] for value, (task, _) in zip(raw, choices)]
        details.append({"id": task_id, "answer_index": next(iter(answers)),
                        "prediction": int(np.argmax(raw)), "normalized_prediction": int(np.argmax(normalized)),
                        "choice_log_likelihoods": raw, "choice_character_normalized_log_likelihoods": normalized})
    return {"question_count": len(details),
            "accuracy_pct": 100 * sum(d["prediction"] == d["answer_index"] for d in details) / len(details),
            "character_normalized_accuracy_pct": 100 * sum(d["normalized_prediction"] == d["answer_index"] for d in details) / len(details),
            "normalization": "sum log probability divided by answer text Unicode character count, excluding inserted leading space",
            "tie_break": "first choice in dataset order", "questions": details}


def validate_cases(cases, vocabulary, graph_capacity=None):
    if not cases:
        raise ValueError("The evaluation policy has no cases")
    for key, case in cases.items():
        initial, targets = case["initial_ids"], case["teacher_ids"]
        capacity = int(case["capacity"])
        mode = case.get("execution", "incremental")
        if not initial or not targets or mode not in ("prefill", "incremental"):
            raise ValueError(f"Invalid prompt, targets, or execution mode for {key}")
        if len(initial) + len(targets) - 1 > capacity:
            raise ValueError(f"Case {key} exceeds its declared capacity {capacity}")
        if graph_capacity is not None and capacity > graph_capacity:
            raise ValueError(f"Case {key} capacity {capacity} exceeds graph maximum {graph_capacity}")
        if any(type(token) is not int or not 0 <= token < vocabulary for token in initial + targets):
            raise ValueError(f"Invalid token ID in {key}")
        task = case.get("task")
        if task and (task.get("normalization_chars", 0) <= 0 or task.get("choice_count", 0) < 2):
            raise ValueError(f"Invalid multiple-choice metadata in {key}")


def graph_geometry(graph):
    generation = graph.get("execution", {}).get("generation", {})
    vocabulary = generation.get("id_domain_size")
    if not vocabulary:
        for node in graph["nodes"]:
            for output in node.get("outputs", {}).values():
                if output.get("tensor") == "logits":
                    vocabulary = output["shape"][-1]
    if type(vocabulary) is not int or vocabulary <= 0:
        raise ValueError("Cannot determine the logits vocabulary from the graph")
    sequence = graph["inputs"]["token_ids"]["shape"][1]
    capacity = sequence if type(sequence) is int else graph.get("dimensions", {}).get(sequence, {}).get("max")
    return vocabulary, capacity


def copy_logit_rows(buffers, output, position, count, vocabulary):
    if output.name != "logits" or output.dtype != pb.DataType.DATA_TYPE_F32:
        raise ValueError("Expected an F32 logits tensor")
    if len(output.shape) != 3 or output.shape[0] != 1 or output.shape[2] != vocabulary:
        raise ValueError(f"Unexpected logits shape: {output.shape}")
    if position < 0 or position + count > output.shape[1] or not output.HasField("buffer"):
        raise ValueError("The requested logits rows are unavailable")
    source = pb.Tensor(name="logits", shape=[count, vocabulary], dtype=output.dtype,
                       buffer=pb.BufferView(buffer_id=output.buffer.buffer_id,
                                            offset_bytes=output.buffer.offset_bytes + position * vocabulary * 4,
                                            length_bytes=count * vocabulary * 4))
    copied = buffers.copy_tensors(pb.CopyTensorsRequest(sources=[source], inline_result=True))
    if len(copied.outputs) != 1 or not copied.outputs[0].HasField("inline"):
        raise ValueError("Logits copy did not return one inline tensor")
    return np.frombuffer(copied.outputs[0].inline, dtype="<f4").reshape(count, vocabulary).copy()


def copy_logit_row(buffers, output, position, vocabulary=32000):
    return copy_logit_rows(buffers, output, position, 1, vocabulary)[0]


def evaluate_model_on_cases(graph_path, weights_path, cases, backend="cpu", cpu_threads=2,
                            packed_arithmetic="f32", on_case=None, reuse_prefill_context=True, execution_evidence=None):
    """Stream teacher-forced rows; stateless full evaluate reuses one context.

    Full evaluate never initializes incremental state, so each request replaces
    all inputs and may rebind the declared dynamic S. Initialize/transition
    checks use fresh contexts because initialize is a one-time state transition.
    """
    graph = json.loads(Path(graph_path).read_text())
    vocabulary, graph_capacity = graph_geometry(graph)
    validate_cases(cases, vocabulary, graph_capacity)
    # Result admission reserves the declared output maximum before binding a
    # smaller request. Cover that conservative bound plus copy/response overlap.
    max_capacity = max(graph_capacity or 0, max(c["capacity"] for c in cases.values()))
    results = {}
    with contextlib.ExitStack() as owners:
        host = open_library()
        owners.callback(host.close)
        client, buffers = vx.VxInferenceServiceClient(host), vx.VxBufferServiceClient(host)
        runtime = client.create_runtime(pb.CreateRuntimeRequest(
            cpu_threads=cpu_threads,
            budget=pb.RuntimeBudget(max_unconsumed_result_bytes=3 * max_capacity * vocabulary * 4)))
        owners.callback(client.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id))
        model = client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                                                      graph_path=str(graph_path), weight_paths=[str(weights_path)]))
        owners.callback(client.release_model, pb.ModelRef(model_id=model.model_id))
        requested_arithmetic = _resolve_packed_arithmetic(packed_arithmetic)
        compiled = client.compile_model(pb.CompileModelRequest(
            model_id=model.model_id, packed_arithmetic=requested_arithmetic,
            policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                                    backends=[backend], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
        owners.callback(client.release_compiled_model, pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
        if compiled.packed_arithmetic != requested_arithmetic:
            raise ValueError("Compilation did not attest the requested packed arithmetic")
        if compiled.report.backend != backend or compiled.report.compilation.tier_fallback_used:
            raise ValueError("Compilation did not select the required backend without fallback")
        if execution_evidence is not None:
            execution_evidence.update({"backend": compiled.report.backend, "device": compiled.report.device,
                                       "packed_arithmetic": int(compiled.packed_arithmetic),
                                       "operator_fallback": "forbid", "completed_requests": 0,
                                       "route_digests": []})
        shared_context = None
        shared_owner = contextlib.ExitStack()
        owners.callback(shared_owner.close)
        for case_id, case in cases.items():
            initial, teachers = case["initial_ids"], case["teacher_ids"]
            prefill = case.get("execution", "incremental") == "prefill"
            # A full model context can reserve half of a 24 GiB GPU even for a
            # short request. Never overlap a shared prefill and fresh decode
            # context; a later prefill group starts a new stateless context.
            if not prefill and shared_context is not None:
                shared_owner.close()
                shared_context = None
            if prefill and reuse_prefill_context and shared_context is None:
                shared_context = client.create_execution_context(pb.CreateExecutionContextRequest(
                    compiled_model_id=compiled.compiled_model_id,
                    graph=pb.GraphExecution(slots=1, row_mode=pb.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,
                                             require_incremental=True)))
                shared_owner.callback(client.release_execution_context,
                                      pb.ExecutionContextRef(context_id=shared_context.context_id))
            reuse_context = prefill and shared_context is not None
            context = shared_context if reuse_context else client.create_execution_context(pb.CreateExecutionContextRequest(
                compiled_model_id=compiled.compiled_model_id,
                graph=pb.GraphExecution(slots=1, row_mode=pb.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,
                                         require_incremental=True)))
            try:
                capacity = case["capacity"]
                prefix = initial + teachers[:-1] if prefill else initial
                values = {"token_ids": np.zeros((1, capacity), np.int32),
                          "position_ids": np.arange(capacity, dtype=np.int32),
                          "attention_mask": np.zeros((1, capacity), np.int32),
                          "active_lengths": np.array([len(prefix)], np.int32)}
                values["token_ids"][0, :len(prefix)] = prefix
                values["attention_mask"][0, :len(prefix)] = 1
                logits_steps = []
                for step in range(1 if prefill else len(teachers)):
                    position = len(initial) - 1 + step
                    if step:
                        values["token_ids"][0, position] = teachers[step - 1]
                        values["attention_mask"][0, position] = 1
                    first_entrypoint = "evaluate" if reuse_context else "initialize"
                    scope = pb.ExecutionScope(entrypoint=first_entrypoint, full=pb.Empty()) if not step else pb.ExecutionScope(
                        entrypoint="transition", slot_rows=pb.SlotRows(slots=[pb.SlotRow(position=position)]))
                    tensors = [pb.Tensor(name=name, dtype=pb.DataType.DATA_TYPE_I32, shape=value.shape,
                                         inline=value.tobytes()) for name, value in values.items()
                               if not step or name != "active_lengths"]
                    packet = client.execute(pb.ExecuteRequest(
                        context_id=context.context_id,
                        inputs=pb.InputBindings(values=tensors, reuse=["active_lengths"] if step else []), scope=scope,
                        outputs=pb.OutputOptions(selection=pb.TensorOutputSelection(names=["logits"]),
                                                storage=pb.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER)))
                    owned = [t.buffer.buffer_id for t in packet.outputs if t.HasField("buffer")]
                    try:
                        if backend == "cuda":
                            report, route = packet.report, packet.report.route
                            if (report.status != pb.NativeStatus.NATIVE_STATUS_OK or report.backend != "cuda"
                                    or not route.builtin or not route.attested or route.provider != "cuda"
                                    or route.active_nodes <= 0 or route.selected_nodes != route.active_nodes
                                    or route.fallback_nodes or route.missing_nodes or report.fallback.operator_fallback_used):
                                raise ValueError("Execute did not attest a complete CUDA route without fallback")
                            if execution_evidence is not None:
                                execution_evidence["route"] = {"provider": route.provider, "attested": route.attested,
                                    "active_nodes": route.active_nodes, "selected_nodes": route.selected_nodes,
                                    "fallback_nodes": route.fallback_nodes, "missing_nodes": route.missing_nodes}
                                if route.digest not in execution_evidence["route_digests"]:
                                    execution_evidence["route_digests"].append(route.digest)
                        if execution_evidence is not None:
                            execution_evidence["completed_requests"] += 1
                        if len(packet.outputs) != 1:
                            raise ValueError("Execute did not return exactly one logits tensor")
                        logits_steps.append(copy_logit_rows(buffers, packet.outputs[0], position,
                                                           len(teachers) if prefill else 1, vocabulary))
                    finally:
                        buffers.release_buffers(pb.BufferRefs(buffer_ids=owned))
                rows = np.concatenate(logits_steps)
                if on_case:
                    on_case(case_id, rows)
                else:
                    results[case_id] = rows
            finally:
                if not reuse_context:
                    client.release_execution_context(pb.ExecutionContextRef(context_id=context.context_id))
    return results


def packed_reuse_smoke(graph_path, weights_path, cases, backend, cpu_threads, arithmetic, evidence):
    """Compare identical arithmetic across fresh and reused 128→31→128 shapes."""
    source = next((case for case in cases.values() if not case.get("task")), next(iter(cases.values())))
    _, maximum = graph_geometry(json.loads(Path(graph_path).read_text()))
    stream = source["initial_ids"] + source["teacher_ids"]
    wide = min(128, len(stream) - 1, maximum or 128)
    short = min(31, max(1, wide // 2))
    def shape_case(capacity):
        scored = min(8, capacity)
        prompt = capacity - scored + 1
        return {"initial_ids": stream[:prompt], "teacher_ids": stream[prompt:prompt + scored],
                "capacity": capacity, "execution": "prefill"}
    smoke_cases = {"wide": shape_case(wide), "short": shape_case(short), "repeat": shape_case(wide)}
    evidence.update({"status": "RUNNING", "capacities": [wide, short, wide],
                     "packed_arithmetic": arithmetic, "reuse_execution": {}, "fresh_execution": {}})
    reused = evaluate_model_on_cases(graph_path, weights_path, smoke_cases, backend=backend,
                                    cpu_threads=cpu_threads, packed_arithmetic=arithmetic,
                                    execution_evidence=evidence["reuse_execution"])
    fresh = evaluate_model_on_cases(graph_path, weights_path,
                                   {"wide": smoke_cases["wide"], "short": smoke_cases["short"]},
                                   backend=backend, cpu_threads=cpu_threads, packed_arithmetic=arithmetic,
                                   reuse_prefill_context=False, execution_evidence=evidence["fresh_execution"])
    pairs = [(reused["wide"], fresh["wide"]), (reused["short"], fresh["short"]),
             (reused["repeat"], fresh["wide"])]
    differences = np.concatenate([a.astype(np.float64) - b.astype(np.float64) for a, b in pairs])
    passed = all(np.allclose(a, b, atol=.002, rtol=.0002) for a, b in pairs)
    evidence.update({"status": "PASSED" if passed else "FAILED", "max_abs": float(np.abs(differences).max()),
                     "rms": float(np.sqrt(np.mean(differences ** 2))), "atol": .002, "rtol": .0002,
                     "top1_agreement_pct": 100 * sum(np.count_nonzero(a.argmax(1) == b.argmax(1)) for a, b in pairs)
                         / sum(len(a) for a, _ in pairs)})
    return passed


def read_policy(path):
    policy = json.loads(Path(path).read_text())
    cases = policy["cases"]
    if isinstance(cases, list):
        ids = [case["id"] for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate evaluation case IDs")
        cases = {case["id"]: case for case in cases}
    for case in cases.values():
        case.setdefault("capacity", policy.get("context_capacity", 2048))
        case.setdefault("execution", policy.get("execution", "prefill"))
    return policy, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True,
                        help="NAME=package directory; repeat for every required model")
    parser.add_argument("--reference", default="f32")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--packed-arithmetic", nargs="+", choices=("f32", "f16", "dynamic-i8", "f32-fast"), default=["f32"])
    parser.add_argument("--verify-prefix-tokens", type=int, default=8,
                        help="Before evaluation, compare baseline prefill/decode on this many fixed teacher tokens; 0 disables")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    models = {}
    for item in args.model:
        name, separator, directory = item.partition("=")
        if not separator or not name.strip() or name != name.strip() or not directory.strip():
            parser.error("--model requires NAME=package directory with a nonempty name and path")
        if name in models:
            parser.error(f"Duplicate model name {name}")
        models[name] = Path(directory).expanduser()
    if args.reference not in models:
        parser.error(f"Required reference model {args.reference} was not provided")
    for name, path in models.items():
        for filename in ("graph.json", "model.safetensors"):
            if not (path / filename).is_file():
                parser.error(f"Model {name} is missing required file: {path / filename}")
    policy_path = args.policy
    policy, cases = read_policy(policy_path)
    metadata = {}
    for name, path in models.items():
        graph, weights = path / "graph.json", path / "model.safetensors"
        vocabulary, capacity = graph_geometry(json.loads(graph.read_text()))
        validate_cases(cases, vocabulary, capacity)
        metadata[name] = {"path": str(path), "graph_sha256": sha256_file(graph),
                          "weights_sha256": sha256_file(weights), "weights_bytes": weights.stat().st_size,
                          "vocabulary": vocabulary, "graph_capacity": capacity}
    report = {"schema": "volvoxai-quantization-quality-evaluation/v2", "status": "RUNNING",
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "backend": args.backend, "reference": args.reference,
              "policy_path": str(policy_path), "policy_sha256": sha256_file(policy_path),
              "max_concurrent_contexts": 1,
              "execution_strategy": "One stateless graph context per variant; full evaluate replaces all inputs and rebinds per-case dynamic capacity; incremental smoke uses fresh initialized contexts",
              "policy": {key: value for key, value in policy.items() if key != "cases"},
              "models": metadata, "variants": {}, "variant_reuse_smokes": {},
              "scope": "Fixed held-out subsets; exploratory paired comparisons, no quality-equivalence claim."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.verify_prefix_tokens < 0:
        parser.error("--verify-prefix-tokens must be nonnegative")
    cache = Path(tempfile.mkdtemp(prefix=args.out.stem + ".reference.", dir=args.out.parent))
    references = {key: cache / (hashlib.sha256(key.encode()).hexdigest() + ".npy") for key in cases}
    def save_report():
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.out)
    variants = [(args.reference, "f32")] + [(name, mode) for name in models if name != args.reference
                                                    for mode in dict.fromkeys(args.packed_arithmetic)]
    save_report()
    try:
        if args.verify_prefix_tokens:
            source = next((case for case in cases.values() if not case.get("task")), next(iter(cases.values())))
            initial = source["initial_ids"][:32]
            targets = source["teacher_ids"][:args.verify_prefix_tokens]
            smoke = {"initial_ids": initial, "teacher_ids": targets,
                     "capacity": len(initial) + len(targets) - 1}
            # A shorter, dissimilar full request followed by the original prefix
            # checks that dynamic rebinding and prior values do not leak across cases.
            other = {"initial_ids": list(reversed(targets[:2])) or initial,
                     "teacher_ids": list(reversed(targets[:max(1, len(targets) // 2)]))}
            other["capacity"] = len(other["initial_ids"]) + len(other["teacher_ids"]) - 1
            smoke_cases = {"prefill": dict(smoke, execution="prefill"),
                           "other": dict(other, execution="prefill"),
                           "repeated": dict(smoke, execution="prefill")}
            path = models[args.reference]
            smoke_rows = evaluate_model_on_cases(path / "graph.json", path / "model.safetensors", smoke_cases,
                                                 backend=args.backend, cpu_threads=args.cpu_threads)
            smoke_rows.update(evaluate_model_on_cases(path / "graph.json", path / "model.safetensors",
                                                       {"incremental": dict(smoke, execution="incremental")},
                                                       backend=args.backend, cpu_threads=args.cpu_threads))
            fresh_rows = evaluate_model_on_cases(path / "graph.json", path / "model.safetensors",
                                                 {"prefill": smoke_cases["prefill"], "other": smoke_cases["other"]},
                                                 backend=args.backend, cpu_threads=args.cpu_threads,
                                                 reuse_prefill_context=False)
            pairs = [(smoke_rows["prefill"], smoke_rows["incremental"]),
                     (smoke_rows["prefill"], fresh_rows["prefill"]),
                     (smoke_rows["other"], fresh_rows["other"]),
                     (smoke_rows["repeated"], fresh_rows["prefill"])]
            delta = np.concatenate([left.astype(np.float64) - right.astype(np.float64) for left, right in pairs])
            report["teacher_forcing_smoke"] = {
                "model": args.reference, "packed_arithmetic": "f32", "target_tokens": len(targets),
                "max_abs": float(np.abs(delta).max()), "rms": float(np.sqrt(np.mean(delta ** 2))),
                "top1_agreement_pct": float(np.mean(smoke_rows["prefill"].argmax(1) == smoke_rows["incremental"].argmax(1)) * 100),
                "atol": 0.002, "rtol": 0.0002,
                "checks": ["evaluate-vs-incremental", "evaluate-vs-fresh-initialize", "shorter-dissimilar-case-vs-fresh", "original-case-after-shape-change"],
                "passed": all(np.allclose(left, right, atol=.002, rtol=.0002) for left, right in pairs)}
            save_report()
            if not report["teacher_forcing_smoke"]["passed"]:
                raise ValueError("Baseline causal prefill rows disagree with incremental teacher forcing")
            print("Teacher-forcing prefill/decode smoke passed: " + json.dumps(report["teacher_forcing_smoke"]), flush=True)
        for name, mode in variants:
            key = f"{name}/{mode}"
            path = models[name]
            if name != args.reference:
                smoke_evidence = {}
                report["variant_reuse_smokes"][key] = smoke_evidence
                print(f"Verifying fresh/reused shape changes for {key}", flush=True)
                smoke_started = time.monotonic()
                try:
                    passed = packed_reuse_smoke(path / "graph.json", path / "model.safetensors", cases,
                                               args.backend, args.cpu_threads, mode, smoke_evidence)
                except BaseException as error:
                    smoke_evidence.update({"status": "FAILED", "error": str(error)})
                    raise
                finally:
                    smoke_evidence["elapsed_seconds"] = time.monotonic() - smoke_started
                    save_report()
                print(f"{key} context reuse: {smoke_evidence['status']}, max_abs={smoke_evidence['max_abs']:.6g}", flush=True)
                if not passed:
                    raise ValueError(f"Fresh/reused packed context mismatch for {key}")
            started = time.monotonic()
            scores = {}
            evidence = {}
            print(f"Evaluating {key}: {len(cases)} cases", flush=True)
            def receive(case_id, rows):
                if name == args.reference:
                    np.save(references[case_id], rows)
                    reference = None
                else:
                    reference = np.load(references[case_id], mmap_mode="r")
                scores[case_id] = score_rows(rows, cases[case_id]["teacher_ids"], reference)
                if len(scores) % 20 == 0 or len(scores) == len(cases):
                    print(f"{key}: {len(scores)}/{len(cases)} cases, {time.monotonic() - started:.1f}s", flush=True)
            path = models[name]
            evaluate_model_on_cases(path / "graph.json", path / "model.safetensors", cases,
                                    backend=args.backend, cpu_threads=args.cpu_threads,
                                    packed_arithmetic=mode, on_case=receive, execution_evidence=evidence)
            report["variants"][key] = {"model": name, "packed_arithmetic": mode,
                                       "elapsed_seconds": time.monotonic() - started, "execution_evidence": evidence,
                                       "language_model": aggregate_language_metrics(scores, cases),
                                       "multiple_choice": score_multiple_choice(scores, cases), "cases": scores}
            save_report()
        report["status"] = "COMPLETE"
        save_report()
    except BaseException as error:
        report["status"] = "FAILED"
        report["error"] = str(error)
        save_report()
        raise
    finally:
        # Large temporary baseline rows have served their purpose; reports retain
        # all per-case likelihoods and identities needed to audit the comparisons.
        for path in references.values():
            path.unlink(missing_ok=True)
        try:
            cache.rmdir()
        except OSError:
            pass
    print(f"Quality report saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
