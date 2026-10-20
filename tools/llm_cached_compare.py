#!/usr/bin/env python3
"""Qualify compact native decoder rows against frozen HF and fresh forwards.

The full profile observes live private KV with the existing debug service.
Inference observes row mode, storage accounting, logits and greedy IDs. This
capture harness does not measure performance or whole-system memory regression.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from tools.llm_compare import compare, identity, sha256_file, source_provenance
from tools.llm_reference import POLICY_PATH, read_policy, validate_reference_policy


def _retain_fresh_arrays(arrays, length, widths):
    """Own only logical prefix rows; views would retain padded capture buffers."""
    if length < 1 or set(arrays) != set(widths):
        raise ValueError("fresh capture names or prefix length differ from the contract")
    retained = {}
    for name, width in widths.items():
        value = arrays[name]
        if (value.dtype != np.float32 or value.ndim != 3 or value.shape[0] != 1
                or value.shape[1] < length or value.shape[2] != width):
            raise ValueError(f"fresh capture shape/dtype differs from the contract: {name}")
        retained[name] = value[:, :length, :].copy(order="C")
    return retained


def _native_snapshot_inventory():
    directory = Path(os.environ.get("TMPDIR") or "/tmp")
    return sorted([{"name": path.name, "inode": path.stat().st_ino,
                   "bytes": path.stat().st_size} for path in directory.glob("volvoxai-*")
                  if path.is_file()], key=lambda row: row["name"])


def _checked_native_gpu_route(evidence, backend, status_ok, label):
    if backend == "cpu":
        return None
    route = evidence.route
    # Keep CUDA's established provider test. Portable builtin providers use
    # the exact backend identifier rather than accepting a provider substring.
    provider_matches = backend in route.provider if backend == "cuda" else route.provider == backend
    if (evidence.status != status_ok or evidence.backend != backend or
            not route.attested or not route.builtin or not provider_matches or
            route.active_nodes <= 0 or route.selected_nodes != route.active_nodes or
            route.fallback_nodes or route.missing_nodes or
            evidence.fallback.operator_fallback_used):
        raise ValueError(f"{label}: {backend} execution route is not fully attested")
    return {"label": label, "backend": evidence.backend, "device": evidence.device,
            "provider": route.provider, "attested": route.attested,
            "active_nodes": route.active_nodes, "selected_nodes": route.selected_nodes,
            "fallback_nodes": route.fallback_nodes, "missing_nodes": route.missing_nodes,
            "digest": int(route.digest)}


def prepare(model: str, destination: Path, *, policy_path: Path = POLICY_PATH,
            weights_path: Path | None = None):
    """Reuse frozen F32 weight bytes; publish separate C execution/capture graphs."""
    from tools.exporter.hf_decoder import DecoderConfig, build_decoder_graph
    policy = read_policy(policy_path, model=model)
    config = DecoderConfig.from_hf(policy["models"][model]["config"],
        max_context=policy["reference"]["qualification"]["max_context"])
    source = (weights_path or ROOT / "build/llm/packages" / model / "model.safetensors").resolve()
    with safe_open(source, framework="numpy") as handle:
        weights = {name: handle.get_tensor(name) for name in config.weight_shapes()}
    destination.mkdir(parents=True, exist_ok=False)
    for name, options in (("execution", {"cached": True}), ("capture", {"capture_kv": True})):
        package = destination / name
        package.mkdir()
        graph, initializers = build_decoder_graph(config, weights, compact_kv=True, **options)
        derived = {key: value for key, value in initializers.items() if key not in config.weight_shapes()}
        with safe_open(source, framework="numpy") as handle:
            for key, expected in derived.items():
                if key not in handle.keys():
                    raise ValueError(f"shared F32 shard lacks derived graph initializer: {key}")
                actual = handle.get_tensor(key)
                if (actual.dtype != expected.dtype or actual.shape != expected.shape
                        or not np.array_equal(actual, expected)):
                    raise ValueError(f"shared F32 shard differs from derived graph initializer: {key}")
        (package / "graph.json").write_text(json.dumps(graph, indent=2, allow_nan=False) + "\n")
        (package / "model.safetensors").symlink_to(os.path.relpath(source, package))
    receipt = {"format": "volvox-compact-decoder-package/v1", "model": model,
        "policy": identity(policy_path),
        "source_weights": identity(source), "shared_weight_bytes": True,
        "unused_repeat_index_initializer": "Existing baseline shard may retain its small immutable index tensor; no graph consumes it.",
        "graphs": {name: identity(destination / name / "graph.json") for name in ("execution", "capture")}}
    (destination / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


def run(args):
    import volvoxai as vx
    from volvoxai._arrays import input_batch
    from volvoxai._clients import VxInferenceServiceClient, VxDebugServiceClient, VxProfilingServiceClient
    from volvoxai._diagnostics import _proto_json
    from volvoxai._execution import bindings, owned_outputs, buffer_refs, array_destinations, validate_copies
    p = vx.pb
    serial_models = getattr(args, "serial_models", False)
    policy_path = args.policy
    manifest_path = args.reference / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    policy, pinned = validate_reference_policy(manifest, policy_path)
    qualification = policy["reference"]["qualification"]
    expected_cases = {f"prompt{i}-prefix{n:03d}": (i, n, prompt)
        for i, prompt in enumerate(qualification["prompts"]) for n in qualification["prefix_lengths"]}
    if (len(manifest["cases"]) != len(expected_cases)
            or {case["id"] for case in manifest["cases"]} != set(expected_cases)):
        raise ValueError("HF capture does not contain the fixed case matrix")
    cases = [case for case in manifest["cases"] if args.case is None or case["id"] == args.case]
    if not cases:
        raise ValueError("selected case is absent")
    for case in cases:
        if (case["prompt_index"], case["prefix_length"], case["prompt"]) != expected_cases[case["id"]]:
            raise ValueError("HF case metadata differs from its policy")
        path = args.reference / case["reference_path"]
        if sha256_file(path) != case["reference_sha256"] or path.stat().st_size != case["reference_bytes"]:
            raise ValueError("HF tensor bytes differ from their manifest")
    library = ROOT / "native" / ("libvolvoxai-lite.so" if args.profile == "inference" else "libvolvoxai.so")
    os.environ["VOLVOXAI_LIBRARY"] = str(library.resolve())
    graph = json.loads((args.package / "graph.json").read_text())
    capture = json.loads((args.capture_package / "graph.json").read_text())
    capacity = qualification["max_context"]
    if (graph["dimensions"]["S"] != {"min": 1, "max": capacity}
            or graph["outputs"] != ["logits", "generated"]
            or set(graph["inputs"]) != {"token_ids", "position_ids", "attention_mask", "active_lengths"}
            or "incremental" not in graph["execution"] or "generation" not in graph["execution"]):
        raise ValueError("execution package does not declare the compact decoder contract")
    config = manifest["config"]
    layers, heads, width = config["num_hidden_layers"], config["num_key_value_heads"], pinned["head_dim"]
    cache_names = [name for layer in range(layers) for name in (f"layer.{layer}.k_rope", f"layer.{layer}.v")]
    if set(capture["outputs"]) != {"logits", *cache_names}:
        raise ValueError("fresh-forward companion must capture all compact KV")
    for document in (graph, capture):
        if any(node["opType"] in {"Gather", "Transpose"} for node in document["nodes"]):
            raise ValueError("compact package unexpectedly repeats or transposes KV")
        attention = [node for node in document["nodes"] if node["opType"] == "CrossSDPA"]
        if len(attention) != layers or any(node["params"].get("kv_heads") != heads for node in attention):
            raise ValueError("compact head contract differs from the model")
    budget = capacity * 4 * (config["vocab_size"] + 2 * layers * heads * width + 1)
    tolerance = policy["reference"]["engine_tolerance"]
    report = {"format": "volvox-native-compact-decoder-comparison/v1", "status": "RUNNING",
        "model": manifest["model"], "revision": pinned["revision"], "profile": args.profile,
        "backend": args.backend, "cpu_threads": args.threads, "execution": "required-row-paged-compact",
        "capture_enabled": True, "qualifies_performance": False, "tolerance": tolerance,
        "kv_observation": "live-private-cache-debug" if args.profile == "full" else "unavailable-in-inference-profile",
        "reference": identity(manifest_path), "policy": identity(policy_path), "library": identity(library),
        "graph": identity(args.package / "graph.json"), "capture_graph": identity(args.capture_package / "graph.json"),
        "weights": identity(args.package / "model.safetensors"), "harness": identity(Path(__file__)),
        "runtime_budget_bytes": budget, "cases": [], "command": sys.argv,
        "generation_requested": not args.skip_generate, "generation_qualified": False,
        **source_provenance(args.source_manifest)}
    if serial_models:
        report["serial_model_lifecycle"] = {"enabled": True, "phases": [],
            "maximum_simultaneously_loaded_models": 0,
            "fresh_array_bytes_bound": 4 * (config["vocab_size"] + 2 * layers * heads * width)
                * sum(case["prefix_length"] + step for case in cases
                      for step in range(qualification["decode_steps"] + 1)),
            "retained_array_scope": "Owned caller F32 prefix arrays; no process or whole-system peak claim"}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        generations = [result for case in report["cases"] for result in case.get("generate", [])]
        report["generation_qualified"] = (report.get("status") == "PASS" and
            report["generation_requested"] and len(generations) == 2 and
            all(result.get("pass") is True for result in generations))
        report["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["peak_process_rss_scope"] = ("Linux process lifetime, including validation and serial model instances"
            if serial_models else "Linux process lifetime, including validation and both model instances")
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def checked(actual, expected, category, label):
        result = compare(actual, expected, tolerance["layer_kv" if category == "kv" else category])
        if not result["pass"]:
            raise ValueError(f"fixed tolerance exceeded: {label}: {result}")
        return result

    try:
        persist()
        with vx.Runtime(cpu_threads=args.threads, budget=p.RuntimeBudget(max_unconsumed_result_bytes=budget)) as runtime:
            expected_profile = (p.BuildProfile.BUILD_PROFILE_FULL if args.profile == "full"
                                else p.BuildProfile.BUILD_PROFILE_INFERENCE)
            if runtime.build_profile != expected_profile:
                raise ValueError("loaded library profile differs from the selected profile")
            engine = VxInferenceServiceClient(runtime._owner.host)
            debug = VxDebugServiceClient(runtime._owner.host) if args.profile == "full" else None
            profiling = VxProfilingServiceClient(runtime._owner.host)
            baseline = profiling.get_resource_snapshot(p.GetResourceSnapshotRequest(
                runtime_id=runtime._runtime.runtime_id, include_device=args.backend != "cpu")).snapshot
            report["runtime_baseline"] = _proto_json(baseline)
            resources = []
            loaded_models = set()
            report["cuda_route_records"] = []
            report["native_gpu_route_records"] = []
            def check_route(evidence, label):
                record = _checked_native_gpu_route(evidence, args.backend, p.NativeStatus.NATIVE_STATUS_OK, label)
                if record is not None:
                    report["native_gpu_route_records"].append(record)
                    if args.backend == "cuda":
                        report["cuda_route_records"].append(record)
            def load(package):
                model = engine.load_model(p.LoadModelRequest(runtime_id=runtime._runtime.runtime_id,
                    graph_path=str(package / "graph.json"), weight_paths=[str(package / "model.safetensors")]))
                resources.append((engine.release_model, p.ModelRef(model_id=model.model_id)))
                if serial_models:
                    loaded_models.add(model.model_id)
                    report["serial_model_lifecycle"]["maximum_simultaneously_loaded_models"] = max(
                        report["serial_model_lifecycle"]["maximum_simultaneously_loaded_models"], len(loaded_models))
                    if len(loaded_models) != 1:
                        raise ValueError("serial qualification admitted overlapping model owners")
                compiled = engine.compile_model(p.CompileModelRequest(model_id=model.model_id,
                    policy=p.BackendPolicy(mode=p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                        backends=[args.backend], operator_fallback=p.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
                resources.append((engine.release_compiled_model, p.CompiledModelRef(compiled_model_id=compiled.compiled_model_id)))
                if compiled.report.backend != args.backend or compiled.report.compilation.tier_fallback_used:
                    raise ValueError("backend selection differs from the required route")
                check_route(compiled.report, "compile:" + package.name)
                return compiled
            def copy_packet(packet, names):
                refs = buffer_refs(packet.outputs)
                try:
                    arrays, request = array_destinations(packet.outputs, names)
                    validate_copies(runtime._buffers.copy_tensors(request), arrays)
                    return arrays
                finally:
                    runtime._buffers.release_buffers(refs)
            def release_serial_resources():
                first_error = None
                while resources:
                    release, ref = resources.pop()
                    try:
                        release(ref)
                        if isinstance(ref, p.ModelRef):
                            loaded_models.discard(ref.model_id)
                    except BaseException as error:
                        if first_error is None:
                            first_error = error
                if first_error is not None:
                    raise first_error
            def assert_baseline_inventory(snapshot):
                before, after = _proto_json(baseline.memory), _proto_json(snapshot.memory)
                ordered = lambda value: sorted(value.get("allocations", []),
                    key=lambda item: json.dumps(item, sort_keys=True))
                if (before.get("truncated", False) or after.get("truncated", False)
                        or snapshot.memory.unconsumed_result_bytes != 0 or ordered(before) != ordered(after)):
                    raise ValueError("serial leg cleanup did not restore the known runtime inventory")
            def capture_serial_fresh():
                saved = {}
                phase_record = {"phase": "fresh-forward", "status": "RUNNING", "cases": []}
                report["serial_model_lifecycle"]["phases"].append(phase_record)
                temp_before = _native_snapshot_inventory()
                widths = {"logits": config["vocab_size"], **{name: heads * width for name in cache_names}}
                try:
                    fresh_compiled = load(args.capture_package)
                    for case in cases:
                        context = engine.create_execution_context(p.CreateExecutionContextRequest(
                            compiled_model_id=fresh_compiled.compiled_model_id, graph=p.GraphExecution()))
                        entry = {"id": case["id"], "steps": []}
                        phase_record["cases"].append(entry)
                        try:
                            with safe_open(args.reference / case["reference_path"], framework="numpy") as reference:
                                initial = reference.get_tensor("prefill.input_ids").astype(np.int32)
                                if hashlib.sha256(initial.tobytes()).hexdigest() != case["engine_input_ids"]["sha256"]:
                                    raise ValueError("serial fresh input bytes differ from the fixed case")
                                tokens = np.zeros((1, capacity), dtype=np.int32)
                                tokens[:, :initial.shape[1]] = initial
                                positions = np.arange(capacity, dtype=np.int32)
                                for step in range(qualification["decode_steps"] + 1):
                                    phase = "prefill" if step == 0 else f"decode.{step-1:03d}"
                                    if step:
                                        tokens[:, case["prefix_length"] + step - 1] = reference.get_tensor(
                                            phase + ".input_ids").astype(np.int32).reshape(-1)
                                    length = case["prefix_length"] + step
                                    keep = np.zeros((1, capacity), dtype=np.int32)
                                    keep[:, :length] = 1
                                    values = {"token_ids": tokens, "position_ids": positions, "attention_mask": keep}
                                    tensors, owners = input_batch(values, values)
                                    packet = engine.execute(p.ExecuteRequest(context_id=context.context_id,
                                        inputs=bindings(tensors), scope=p.ExecutionScope(entrypoint="evaluate", full=p.Empty()),
                                        outputs=owned_outputs(capture["outputs"])))
                                    check_route(packet.report, case["id"] + ":fresh:" + phase)
                                    retained = _retain_fresh_arrays(copy_packet(packet, capture["outputs"]), length, widths)
                                    del owners
                                    selected = retained["logits"] if step == 0 else retained["logits"][:, -1:, :]
                                    expected = reference.get_tensor(phase + ".logits")
                                    metric = checked(selected, expected, "logits", phase + " serial fresh")
                                    greedy = int(np.argmax(retained["logits"][0, -1]))
                                    if greedy != int(np.argmax(expected[0, -1])):
                                        raise ValueError("serial fresh greedy ID differs from HF")
                                    kv_metrics = []
                                    for layer in range(layers):
                                        for suffix, role in (("k_rope", "key"), ("v", "value")):
                                            name = f"layer.{layer}.{suffix}"
                                            hf = reference.get_tensor(f"{phase}.layer.{layer:03d}.{role}").transpose(
                                                0, 2, 1, 3).reshape(1, length, heads * width)
                                            kv_metrics.append({"name": name, "hf": checked(retained[name], hf, "kv", name + " serial fresh")})
                                    saved[case["id"], phase] = retained
                                    entry["steps"].append({"phase": phase, "length": length,
                                        "greedy_id": greedy, "logits_hf": metric, "kv_hf": kv_metrics})
                                    persist()
                        finally:
                            engine.release_execution_context(p.ExecutionContextRef(context_id=context.context_id))
                except BaseException as error:
                    phase_record["failure"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    release_serial_resources()
                    snapshot = profiling.get_resource_snapshot(p.GetResourceSnapshotRequest(
                        runtime_id=runtime._runtime.runtime_id, include_device=args.backend != "cpu")).snapshot
                    phase_record["runtime_after_release"] = _proto_json(snapshot)
                    assert_baseline_inventory(snapshot)
                    temp_after = _native_snapshot_inventory()
                    phase_record["native_snapshots_before"] = temp_before
                    phase_record["native_snapshots_after_release"] = temp_after
                    if temp_after != temp_before or loaded_models:
                        raise ValueError("serial fresh leg retained model snapshots or owners")
                    phase_record["cleanup"] = "PASS"
                    phase_record["status"] = "FAIL_CLOSED" if "failure" in phase_record else "PASS_CLOSED"
                    persist()
                retained_bytes = sum(array.nbytes for tensors in saved.values() for array in tensors.values())
                report["serial_model_lifecycle"]["retained_fresh_array_bytes"] = retained_bytes
                if retained_bytes != report["serial_model_lifecycle"]["fresh_array_bytes_bound"]:
                    raise ValueError("retained fresh arrays differ from their logical prefix byte bound")
                return saved
            try:
                saved_fresh = capture_serial_fresh() if serial_models else None
                if serial_models:
                    report["serial_model_lifecycle"]["phases"].append({"phase": "cached-row", "status": "RUNNING",
                        "native_snapshots_before": _native_snapshot_inventory()})
                    compiled, fresh_compiled = load(args.package), None
                else:
                    compiled, fresh_compiled = load(args.package), load(args.capture_package)
                report["compilation"] = _proto_json(compiled.report)
                for case in cases:
                    cached = engine.create_execution_context(p.CreateExecutionContextRequest(
                        compiled_model_id=compiled.compiled_model_id,
                        graph=p.GraphExecution(slots=1, row_mode=p.DecodeRowMode.DECODE_ROW_MODE_REQUIRED, require_incremental=True),
                        limits=p.ExecutionLimits(cache=p.ContextCacheOptions(tensors=cache_names, page_tokens=2,
                            max_pages=capacity // 2, clear_on_recycle=True))))
                    fresh = None if serial_models else engine.create_execution_context(p.CreateExecutionContextRequest(
                        compiled_model_id=fresh_compiled.compiled_model_id, graph=p.GraphExecution()))
                    entry = {"id": case["id"], "steps": []}
                    report["cases"].append(entry)
                    previous_kv = {}
                    try:
                        with safe_open(args.reference / case["reference_path"], framework="numpy") as reference:
                            initial = reference.get_tensor("prefill.input_ids").astype(np.int32)
                            if hashlib.sha256(initial.tobytes()).hexdigest() != case["engine_input_ids"]["sha256"]:
                                raise ValueError("input bytes differ from the fixed case")
                            tokens = np.zeros((1, capacity), dtype=np.int32)
                            tokens[:, :initial.shape[1]] = initial
                            positions = np.arange(capacity, dtype=np.int32)
                            for step in range(qualification["decode_steps"] + 1):
                                phase = "prefill" if step == 0 else f"decode.{step-1:03d}"
                                if step:
                                    next_token = reference.get_tensor(phase + ".input_ids").astype(np.int32)
                                    tokens[:, case["prefix_length"] + step - 1] = next_token.reshape(-1)
                                length = case["prefix_length"] + step
                                keep = np.zeros((1, capacity), dtype=np.int32)
                                keep[:, :length] = 1
                                values = {"token_ids": tokens, "position_ids": positions, "attention_mask": keep}
                                if serial_models:
                                    ordinary = saved_fresh[case["id"], phase]
                                else:
                                    fresh_tensors, fresh_owners = input_batch(values, values)
                                    fresh_packet = engine.execute(p.ExecuteRequest(context_id=fresh.context_id,
                                        inputs=bindings(fresh_tensors), scope=p.ExecutionScope(entrypoint="evaluate", full=p.Empty()),
                                        outputs=owned_outputs(capture["outputs"])))
                                    check_route(fresh_packet.report, case["id"] + ":fresh:" + phase)
                                    ordinary = copy_packet(fresh_packet, capture["outputs"])
                                    del fresh_owners
                                supplied = dict(values)
                                if step == 0:
                                    supplied["active_lengths"] = np.array([length], dtype=np.int32)
                                tensors, owners = input_batch(supplied, graph["inputs"])
                                scope = (p.ExecutionScope(entrypoint="initialize", full=p.Empty()) if step == 0 else
                                    p.ExecutionScope(entrypoint="transition", slot_rows=p.SlotRows(slots=[p.SlotRow(position=length-1)])))
                                request = p.ExecuteRequest(context_id=cached.context_id,
                                    inputs=bindings(tensors, () if step == 0 else ("active_lengths",)), scope=scope,
                                    outputs=owned_outputs(["logits", "generated"]))
                                debug_info = None
                                try:
                                    if debug:
                                        debug_info = debug.create_debug_session(p.CreateDebugSessionRequest(execute=request,
                                            capture=p.DebugCapture(tensor_names=["generated"], inputs=False, outputs=False, values=False, constants=False)))
                                        debug_info = debug.continue_debug_session(p.ContinueDebugSessionRequest(
                                            debug_session_id=debug_info.debug_session_id, expected_revision=debug_info.revision))
                                        if debug_info.state != p.DebugState.DEBUG_STATE_COMPLETED:
                                            raise ValueError("debugged cache operation did not complete")
                                        debug_ref = p.DebugSessionRef(debug_session_id=debug_info.debug_session_id)
                                        packet = debug.take_debug_result(debug_ref)
                                    else:
                                        packet = engine.execute(request)
                                    check_route(packet.report, case["id"] + ":cached:" + phase)
                                    actual = copy_packet(packet, ["logits", "generated"])
                                    state = engine.get_execution_context_info(p.ExecutionContextRef(context_id=cached.context_id))
                                    if state.mode != p.DecodeMode.DECODE_MODE_ROW or list(state.active_lengths) != [length]:
                                        raise ValueError("required row execution did not publish its expected state")
                                    expected_pool = capacity * 2 * layers * heads * width * 4
                                    if state.cache.pool_bytes != expected_pool:
                                        raise ValueError("cache pool does not use compact Hkv*D rows")
                                    selected = actual["logits"][:, :length, :] if step == 0 else actual["logits"][:, length-1:length, :]
                                    expected = reference.get_tensor(phase + ".logits")
                                    record = {"phase": phase, "length": length, "mode": int(state.mode),
                                        "cache": _proto_json(state.cache), "logits_hf": checked(selected, expected, "logits", phase),
                                        "logits_fresh": checked(selected, ordinary["logits"][:, :length, :] if step == 0 else
                                            ordinary["logits"][:, length-1:length, :], "logits", phase + " fresh"), "kv": []}
                                    greedy = int(np.argmax(actual["logits"][0, length-1]))
                                    if greedy != int(np.argmax(expected[0, -1])) or greedy != int(actual["generated"][0, length-1]):
                                        raise ValueError("greedy ID differs from HF/engine ArgMax")
                                    record["greedy_id"] = greedy
                                    caches = {}
                                    if debug:
                                        plan = debug.get_debug_plan(debug_ref).plan
                                        decode = debug.get_debug_decode_state(debug_ref)
                                        for cache in decode.caches:
                                            name = plan.tensors[cache.tensor_id].name
                                            if name not in cache_names or cache.token_shape != [heads * width] or cache.page_tokens != 2:
                                                raise ValueError(f"live KV descriptor differs from compact paged storage: name={name!r}, expected_width={heads*width}, descriptor={_proto_json(cache)}")
                                            raw = debug.read_debug_kv_cache(p.ReadDebugKVCacheRequest(
                                                debug_session_id=debug_info.debug_session_id, cache_id=cache.cache_id,
                                                slot=0, token_limit=capacity))
                                            if raw.dtype != p.DataType.DATA_TYPE_F32 or list(raw.shape) != [length, heads * width]:
                                                raise ValueError("live KV read differs from its logical shape")
                                            caches[name] = np.frombuffer(raw.data, dtype="<f4").copy().reshape(1, length, heads * width)
                                        if set(caches) != set(cache_names):
                                            raise ValueError("live cache observation is incomplete")
                                    for layer in range(layers):
                                        for suffix, role in (("k_rope", "key"), ("v", "value")):
                                            name = f"layer.{layer}.{suffix}"
                                            hf = reference.get_tensor(f"{phase}.layer.{layer:03d}.{role}").transpose(0, 2, 1, 3).reshape(1, length, heads * width)
                                            observed = caches.get(name, ordinary[name][:, :length])
                                            result = {"name": name, "fresh_hf": checked(ordinary[name][:, :length], hf, "kv", name)}
                                            if debug:
                                                result["live_hf"] = checked(observed, hf, "kv", name + " live")
                                                result["live_fresh"] = checked(observed, ordinary[name][:, :length], "kv", name + " fresh")
                                                if step and not np.array_equal(observed[:, :length-1], previous_kv[name]):
                                                    raise ValueError("row transition changed prior KV")
                                                result["prior_rows_unchanged"] = bool(step)
                                            previous_kv[name] = observed.copy()
                                            record["kv"].append(result)
                                    entry["steps"].append(record)
                                    persist()
                                finally:
                                    if debug_info:
                                        debug.release_debug_session(p.DebugSessionRef(debug_session_id=debug_info.debug_session_id))
                                    del owners
                            if case is cases[0] and not args.skip_generate:
                                expected_ids = reference.get_tensor("continuation.greedy_ids").astype(np.int32)
                                generation = []
                                for chunks in ((qualification["decode_steps"],), (1, 2, 1)):
                                    if sum(chunks) != qualification["decode_steps"]:
                                        raise ValueError("greedy chunk fixture differs from its fixed decode scope")
                                    context = engine.create_execution_context(p.CreateExecutionContextRequest(
                                        compiled_model_id=compiled.compiled_model_id,
                                        autoregressive=p.AutoregressiveExecution(sampling=p.SamplingMode.SAMPLING_MODE_GREEDY,
                                            max_steps=qualification["decode_steps"]),
                                        limits=p.ExecutionLimits(cache=p.ContextCacheOptions(tensors=cache_names,
                                            page_tokens=2, max_pages=capacity // 2, clear_on_recycle=True))))
                                    try:
                                        conditioning = np.zeros((1, capacity), dtype=np.int32)
                                        conditioning[:, :initial.shape[1]] = initial
                                        keep = np.zeros_like(conditioning)
                                        keep[:, :initial.shape[1]] = 1
                                        tensors, owners = input_batch({"token_ids": conditioning,
                                            "attention_mask": keep, "position_ids": positions,
                                            "active_lengths": np.array([initial.shape[1]], dtype=np.int32)}, graph["inputs"])
                                        chosen = []
                                        for index, count in enumerate(chunks):
                                            response = engine.generate(p.GenerateRequest(context_id=context.context_id,
                                                initial_inputs=p.GenerationInputs(values=tensors) if index == 0 else None,
                                                budget=p.GenerationBudget(max_steps=count), outputs=owned_outputs(["generated_ids"])))
                                            check_route(response.report, case["id"] + ":generate:" + str(chunks) + ":" + str(index))
                                            chosen.append(copy_packet(response, ["generated_ids"])["generated_ids"])
                                        ids = np.concatenate(chosen, axis=1)
                                        if not np.array_equal(ids, expected_ids):
                                            raise ValueError("engine Generate IDs differ from frozen HF continuation")
                                        state = engine.get_execution_context_info(p.ExecutionContextRef(context_id=context.context_id))
                                        if state.mode != p.DecodeMode.DECODE_MODE_ROW or state.total_steps != qualification["decode_steps"]:
                                            raise ValueError("Generate did not use the compact row state")
                                        generation.append({"chunks": list(chunks), "ids": ids.tolist(),
                                            "mode": int(state.mode), "total_steps": state.total_steps,
                                            "pool_bytes": state.cache.pool_bytes, "pass": True})
                                    finally:
                                        engine.release_execution_context(p.ExecutionContextRef(context_id=context.context_id))
                                entry["generate"] = generation
                                persist()
                    finally:
                        engine.release_execution_context(p.ExecutionContextRef(context_id=cached.context_id))
                        if fresh is not None:
                            engine.release_execution_context(p.ExecutionContextRef(context_id=fresh.context_id))
            except BaseException as error:
                if serial_models and report["serial_model_lifecycle"]["phases"][-1]["phase"] == "cached-row":
                    report["serial_model_lifecycle"]["phases"][-1]["failure"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                if serial_models:
                    release_serial_resources()
                    phase_record = report["serial_model_lifecycle"]["phases"][-1]
                    if phase_record["phase"] == "cached-row":
                        snapshot = profiling.get_resource_snapshot(p.GetResourceSnapshotRequest(
                            runtime_id=runtime._runtime.runtime_id, include_device=args.backend != "cpu")).snapshot
                        phase_record["runtime_after_release"] = _proto_json(snapshot)
                        assert_baseline_inventory(snapshot)
                        phase_record["native_snapshots_after_release"] = _native_snapshot_inventory()
                        if loaded_models or phase_record["native_snapshots_after_release"] != phase_record["native_snapshots_before"]:
                            raise ValueError("serial cached leg retained model snapshots or owners")
                        phase_record["cleanup"] = "PASS"
                        phase_record["status"] = "FAIL_CLOSED" if "failure" in phase_record else "PASS_CLOSED"
                        persist()
                else:
                    for release, ref in reversed(resources):
                        release(ref)
            if serial_models:
                saved_fresh.clear()
                ordinary = None
                report["serial_model_lifecycle"]["retained_fresh_arrays_after_release"] = 0
            final = profiling.get_resource_snapshot(
                p.GetResourceSnapshotRequest(runtime_id=runtime._runtime.runtime_id,
                    include_device=args.backend != "cpu"))
            report["runtime_end"] = _proto_json(final.snapshot)
            if final.snapshot.memory.unconsumed_result_bytes != 0:
                raise ValueError("completed output buffers retained the result budget")
            before_memory, after_memory = _proto_json(baseline.memory), _proto_json(final.snapshot.memory)
            def allocations(value):
                return sorted(value.get("allocations", []), key=lambda item: json.dumps(item, sort_keys=True))
            if (before_memory.get("truncated", False) or after_memory.get("truncated", False)
                    or allocations(before_memory) != allocations(after_memory)):
                raise ValueError("cleanup did not restore the known runtime allocation inventory")
            report["known_inventory_cleanup"] = {"pass": True, "unconsumed_result_bytes": 0,
                "allocation_count": len(allocations(after_memory)), "scope": "known runtime inventory; whole-process memory regression unqualified"}
            if serial_models:
                report["serial_model_lifecycle"]["phases"][-1].update({"status": "PASS_CLOSED", "cleanup": "PASS",
                    "runtime_after_release": _proto_json(final.snapshot)})
        report["status"] = "PASS"
        persist()
    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        persist()
        raise
    print(json.dumps({"status": report["status"], "model": report["model"], "profile": args.profile,
        "executions": sum(len(case["steps"]) for case in report["cases"]), "out": str(args.out)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", help="model alias in the selected policy")
    parser.add_argument("--policy", type=Path, default=POLICY_PATH,
                        help="pinned model and numerical policy used by the HF capture")
    parser.add_argument("--weights", type=Path,
                        help="existing F32 initializer shard reused by --prepare")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--capture-package", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--profile", choices=("inference", "full"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan", "opengl"), default="cpu")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--skip-generate", action="store_true",
                        help="qualify Execute steps separately from device ID feedback")
    parser.add_argument("--serial-models", action="store_true",
                        help="release the fresh companion before loading the cached model; retain only logical F32 prefixes")
    parser.add_argument("--case")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.threads < 1:
        parser.error("output must be new and threads positive")
    if args.prepare:
        if args.serial_models:
            parser.error("serial-models is only used with comparison")
        if args.prepare not in read_policy(args.policy)["models"]:
            parser.error("prepare model is absent from the selected policy")
        prepare(args.prepare, args.out, policy_path=args.policy, weights_path=args.weights)
    elif args.weights is not None:
        parser.error("weights is only used with prepare")
    elif None in (args.package, args.capture_package, args.reference, args.profile):
        parser.error("comparison requires package, capture-package, reference and profile")
    else:
        run(args)


if __name__ == "__main__":
    main()
