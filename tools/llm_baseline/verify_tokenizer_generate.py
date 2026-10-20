#!/usr/bin/env python3
"""Freeze HF observations, then replay real tokenizer → public Generate calls.

These are correctness runs, never idle-host performance measurements. Weights
are local and revision-pinned. Capture has no engine dependency; replay never
loads PyTorch or computes model logits. Native full/lite and required CUDA use
the same generated services and explicit FORBID-fallback backend policy.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import resource
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from tools.llm_reference import read_policy, sha256_file, verify_snapshot, _provenance
from tools.llm_baseline.tokenizer_generate_reference import ByteInverse, categorical, utf8_stream

PROMPTS = ["한글 한글 한글 한글", "🙂🚀 🙂🚀 🙂🚀 🙂🚀"]
TOKENIZER_TEXTS = ["한국어와 emoji 🙂🚀 123\n\tA e\u0301", "  hello\n\nworld 42\t!", "", "한글 🙂"]
POLICIES = [
    {"name": "greedy", "sampling": "GREEDY"},
    {"name": "random-default", "sampling": "RANDOM"},
    {"name": "random-filtered", "sampling": "RANDOM", "seed": 7,
     "temperature": .7, "top_k": 40, "top_p": .85},
    {"name": "random-top1", "sampling": "RANDOM", "seed": 2**64-1, "top_k": 1},
]
STEPS = 4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def prepare_graph(source, config, destination):
    """Keep existing operator lowering and weights; attach canonical Generate metadata."""
    from tools.exporter.execution_contract import full_prefix_decoder_execution
    require(destination.name == "graph.json" or destination.name.endswith(".graph.json"),
            "prepared graph must be graph.json or a named *.graph.json document")
    graph = copy.deepcopy(json.loads(source.read_text()))
    require(set(graph["inputs"]) == {"token_ids", "position_ids", "attention_mask"},
            "prepare needs the ordinary full-prefix HF decoder graph")
    require("logits" in graph["outputs"], "HF logits output missing")
    require(not any(node["id"] == "generated" for node in graph["nodes"]), "ID node already exists")
    graph["outputs"] = ["logits", "generated"]
    graph["nodes"].append({"id": "generated", "opType": "ArgMax", "inputs": {"input": "logits"},
        "outputs": {"out": {"tensor": "generated", "dtype": "int32", "shape": [1, "S"]}},
        "params": {"axis": -1, "keepdims": False}})
    graph["execution"] = full_prefix_decoder_execution(graph["inputs"], graph["outputs"],
        token_input="token_ids", token_output="generated", keep_input="attention_mask",
        position_input="position_ids", logits_output="logits", id_domain_size=config["vocab_size"])
    write(destination, graph)


def capture(args):
    policy = read_policy(args.policy, model=args.model)
    pinned = policy["models"][args.model]
    require(not args.out.exists(), "reference output must be new")
    hashes = verify_snapshot(args.snapshot, pinned)
    provenance = _provenance(policy, args.policy, model=args.model)
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.numpy import save_file
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True,
        trust_remote_code=False, use_fast=True)
    rust = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
    definition = json.loads((args.snapshot / "tokenizer.json").read_text())
    byte_level = definition["decoder"]["type"] == "ByteLevel"
    inverse = ByteInverse(definition) if byte_level else None
    report = {"format": "volvox-tokenizer-generate-hf/v1", "status": "running", "model": args.model,
        "model_kind": "real trained checkpoint", "repo_id": pinned["repo_id"], "revision": pinned["revision"],
        "policy": identity(args.policy), "snapshot_files": hashes, "reference_provenance": provenance,
        "harness": identity(Path(__file__)), "reference_helper": identity(Path(__file__).with_name("tokenizer_generate_reference.py")),
        "engine_dependency": False, "dtype": "float32", "attention": "eager", "execution": "full-prefix",
        "tokenizer_profile": "HF_BYTE_LEVEL_BPE" if byte_level else "HF_SENTENCEPIECE_BPE",
        "config": pinned["config"], "qualifies_performance": False, "steps": STEPS,
        "tokenizer_cases": [], "byte_cases": [], "generation_cases": []}
    args.out.mkdir(parents=True)
    # The admitted unmodified HF JSON is the product contract. Transformers
    # 5.13's legacy Llama wrapper rewrites leading-space processing; record its
    # distinct behavior rather than changing the JSON or calling it our oracle.
    report["tokenizer_authority"] = "Rust tokenizers.Tokenizer.from_file of the unmodified pinned HF JSON"
    special = [entry["id"] for entry in definition.get("added_tokens", []) if entry["special"]]
    texts = TOKENIZER_TEXTS + [entry["content"] + " 한국어🙂" for entry in
        definition.get("added_tokens", []) if entry["special"]][:2]
    for text in texts:
        for add_special in (False, True):
            ids = rust.encode(text, add_special_tokens=add_special).ids
            wrapper_ids = tokenizer.encode(text, add_special_tokens=add_special)
            if byte_level:
                require(ids == wrapper_ids, "ByteLevel AutoTokenizer wrapper differs from pinned HF JSON")
            row = {"text": text, "add_special_tokens": add_special, "ids": ids, "decode": {},
                "transformers_wrapper_ids": wrapper_ids, "transformers_wrapper_agrees": wrapper_ids == ids}
            for skip in (False, True):
                decoded = rust.decode(ids, skip_special_tokens=skip)
                if byte_level:
                    require(decoded == tokenizer.decode(ids, skip_special_tokens=skip,
                        clean_up_tokenization_spaces=False), "ByteLevel HF tokenizer decode disagreement")
                row["decode"][str(skip)] = decoded
                if inverse:
                    raw = inverse.bytes(ids, skip)
                    require(raw.decode("utf-8", "replace") == decoded, "independent inverse/HF decode disagreement")
            report["tokenizer_cases"].append(row)
    if inverse:
        raw_cases = {"korean": "한".encode(), "emoji": "🙂🚀".encode(),
            "incomplete-korean-eof": b"\xed\x95", "incomplete-emoji-eof": b"\xf0\x9f\x99",
            "invalid-utf8": b"\xed\xa0\x80\xff"}
        for name, raw in raw_cases.items():
            ids = [inverse.byte_ids[value] for value in raw]
            if None in ids:
                report["byte_cases"].append({"name": name, "available": False,
                    "reason": "this actual vocabulary omits a required standalone byte ID", "raw_hex": raw.hex()})
                continue
            report["byte_cases"].append({"name": name, "available": True, "ids": ids,
                "raw_hex": raw.hex(), "text": tokenizer.decode(ids, clean_up_tokenization_spaces=False),
                "finalize": utf8_stream([bytes([value]) for value in raw]),
                "discard": utf8_stream([bytes([value]) for value in raw], "discard")})
        if special:
            ids = [special[0]] + rust.encode("한🙂", add_special_tokens=False).ids + [special[-1]]
            for skip in (False, True):
                raw = inverse.bytes(ids, skip)
                report["byte_cases"].append({"name": "special-keep" if not skip else "special-skip",
                    "available": True, "ids": ids, "skip_special_tokens": skip, "raw_hex": raw.hex(),
                    "text": tokenizer.decode(ids, skip_special_tokens=skip, clean_up_tokenization_spaces=False)})
    write(args.out / "manifest.json", report)
    model = AutoModelForCausalLM.from_pretrained(args.snapshot, local_files_only=True,
        trust_remote_code=False, use_safetensors=True, dtype=torch.float32,
        attn_implementation="eager").cpu().eval()
    require(all(not value.is_floating_point() or value.dtype == torch.float32
                for value in model.parameters()), "HF reference must be entirely F32")
    for prompt_index, prompt in enumerate(PROMPTS):
        initial = rust.encode(prompt, add_special_tokens=True).ids
        for selection in POLICIES:
            name = f"prompt{prompt_index}-{selection['name']}"
            ids, state, observed, chosen, witnesses = initial[:], selection.get("seed", 0), [], [], []
            with torch.inference_mode():
                for _ in range(STEPS):
                    input_ids = torch.tensor([ids], dtype=torch.int64)
                    output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                        position_ids=torch.arange(len(ids)).unsqueeze(0), use_cache=False)
                    logits = output.logits[0, -1].cpu().numpy().astype(np.float32).copy()
                    token, state, witness = categorical(logits, selection, state)
                    observed.append(logits)
                    chosen.append(token)
                    witnesses.append(witness)
                    ids.append(token)
                    del output
            payload = args.out / (name + ".safetensors")
            save_file({"hf.logits": np.stack(observed), "hf.initial_ids": np.array(initial, dtype=np.int32),
                "hf.selected_ids": np.array(chosen, dtype=np.int32)}, payload)
            row = {"name": name, "prompt": prompt, "add_special_tokens": True,
                "input_ids": initial, "selection": selection, "ids": chosen, "witnesses": witnesses,
                "payload": identity(payload), "decode": {}, "whole_decode": {}}
            for skip in (False, True):
                row["decode"][str(skip)] = rust.decode(chosen, skip_special_tokens=skip)
                row["whole_decode"][str(skip)] = rust.decode(initial + chosen, skip_special_tokens=skip)
                if inverse:
                    row.setdefault("raw_hex", {})[str(skip)] = inverse.bytes(chosen, skip).hex()
                    row.setdefault("whole_raw_hex", {})[str(skip)] = inverse.bytes(initial + chosen, skip).hex()
            report["generation_cases"].append(row)
            write(args.out / "manifest.json", report)
            print(f"HF {args.model} {name} {chosen} {row['decode']['True']!r}", flush=True)
    report["status"] = "pass"
    report["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write(args.out / "manifest.json", report)


def replay(args):
    import volvoxai as vx
    from volvoxai._diagnostics import _proto_json
    pb = vx.pb
    reference_path = args.reference / "manifest.json"
    reference = json.loads(reference_path.read_text())
    policy = read_policy(args.policy, model=args.model)
    pinned = policy["models"][args.model]
    require(reference["status"] == "pass" and reference["model"] == args.model, "complete matching HF reference required")
    require(reference["revision"] == pinned["revision"] and reference["config"] == pinned["config"], "pinned HF identity mismatch")
    require(reference["policy"]["sha256"] == sha256_file(args.policy), "reference policy changed")
    require(not args.out.exists(), "qualification output must be new")
    for case in reference["generation_cases"]:
        payload = args.reference / Path(case["payload"]["path"]).name
        require(sha256_file(payload) == case["payload"]["sha256"], "independent HF payload changed")
    tokenizer_path = args.snapshot / "tokenizer.json"
    require(sha256_file(tokenizer_path) == pinned["assets"]["tokenizer.json"]["sha256"], "tokenizer revision mismatch")
    report = {"format": "volvox-tokenizer-generate-native/v1", "status": "running", "model": args.model,
        "repo_id": pinned["repo_id"], "revision": pinned["revision"], "model_kind": "real trained checkpoint",
        "profile": args.profile, "backend": args.backend, "cpu_threads": args.threads,
        "execution": "full-prefix", "cached_decode_qualified": False, "qualifies_performance": False,
        "reference": identity(reference_path), "library": identity(args.library), "weights": identity(args.weights),
        "graph": identity(args.graph), "tokenizer": identity(tokenizer_path), "harness": identity(Path(__file__)),
        "reference_helper": identity(Path(__file__).with_name("tokenizer_generate_reference.py")),
        "command": sys.argv, "tokenizer_checks": [], "byte_checks": [], "generation_checks": [], "resources": []}
    def persist():
        report["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        write(args.out, report)
    def route(operation):
        require(operation.backend == args.backend, "required backend changed")
        require(not operation.fallback or not operation.fallback.operator_fallback_used, "operator fallback used")
        if args.backend == "cuda":
            require(operation.route and operation.route.attested and operation.route.builtin,
                    "physical device route not attested")
            require(operation.route.selected_nodes == operation.route.active_nodes and operation.route.active_nodes > 0,
                    "device route omitted an active graph node")
            require(operation.route.missing_nodes == 0 and operation.route.fallback_nodes == 0, "device graph route incomplete")
    try:
        persist()
        with vx.open_library(args.library, loader=args.loader) as owner:
            text = vx.VxTextServiceClient(owner)
            inference = vx.VxInferenceServiceClient(owner)
            profiling = vx.VxProfilingServiceClient(owner)
            platform = vx.VxPlatformServiceClient(owner)
            # A full library cannot silently stand in for the inference profile.
            caps = platform.get_platform_info(pb.Empty())
            expected_profile = pb.BuildProfile.BUILD_PROFILE_FULL if args.profile == "full" else pb.BuildProfile.BUILD_PROFILE_INFERENCE
            require(caps.build_profile == expected_profile, "selected native build profile mismatch")
            tokenizer = text.create_tokenizer(pb.CreateTokenizerRequest(hugging_face_tokenizer_json=tokenizer_path.read_bytes()))
            tokenizer_id = tokenizer.tokenizer_id
            def decode(ids, skip=False):
                return text.decode_tokens(pb.DecodeTokensRequest(tokenizer_id=tokenizer_id,
                    tokens=ids, skip_special_tokens=skip)).text
            def raw(ids, skip=False):
                value = text.decode_token_bytes(pb.DecodeTokenBytesRequest(tokenizer_id=tokenizer_id,
                    tokens=ids, skip_special_tokens=skip))
                require(value.complete, "raw-byte operation is incomplete")
                return value.value
            byte_level = reference["tokenizer_profile"] == "HF_BYTE_LEVEL_BPE"
            for case in reference["tokenizer_cases"]:
                encoded = text.encode_text(pb.EncodeTextRequest(tokenizer_id=tokenizer_id,
                    text=case["text"], add_special_tokens=case["add_special_tokens"]))
                require(encoded.tokens == case["ids"], "engine EncodeText differs from pinned HF")
                for skip in (False, True):
                    require(decode(encoded.tokens, skip) == case["decode"][str(skip)], "engine DecodeTokens differs from HF")
                    if byte_level:
                        # Token-by-token caller streaming includes every prompt
                        # Korean/emoji boundary even if the continuation is ASCII.
                        stream = utf8_stream([raw([token], skip) for token in encoded.tokens])
                        require(stream["text"] == case["decode"][str(skip)], "prompt token bytes/UTF-8 differ from HF")
                report["tokenizer_checks"].append({"text": case["text"], "add_special_tokens": case["add_special_tokens"],
                    "ids": list(encoded.tokens), "status": "pass"})
            for case in reference["byte_cases"]:
                if not case["available"]:
                    report["byte_checks"].append({"name": case["name"], "status": "unavailable", "reason": case["reason"]})
                    continue
                ids, skip = case["ids"], case.get("skip_special_tokens", False)
                parts = [raw([token], skip) for token in ids]
                require(b"".join(parts).hex() == case["raw_hex"], "engine lossless bytes differ from independent inverse")
                require(decode(ids, skip) == case["text"], "raw corpus whole decode differs from HF")
                row = {"name": case["name"], "status": "pass", "raw_hex": case["raw_hex"]}
                for termination in ("finalize", "discard"):
                    result = utf8_stream(parts, termination)
                    if termination in case:
                        require(result == case[termination], "UTF-8 end/cancellation policy changed")
                    row[termination] = result
                report["byte_checks"].append(row)
            if not byte_level:
                try:
                    raw([pinned["config"]["bos_token_id"]])
                except vx.VolvoxAIError as error:
                    report["sentencepiece_raw_refusal"] = {"status": "pass", "error": str(error)}
                else:
                    raise ValueError("contextual SentencePiece decoder unexpectedly returned independent token bytes")
                require(decode(reference["tokenizer_cases"][0]["ids"]) == reference["tokenizer_cases"][0]["decode"]["False"],
                        "SentencePiece refusal corrupted tokenizer state")
            runtime = inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=args.threads,
                budget=pb.RuntimeBudget(max_unconsumed_result_bytes=4096)))
            def snapshot(phase):
                value = profiling.get_resource_snapshot(pb.GetResourceSnapshotRequest(runtime_id=runtime.runtime_id,
                    include_device=args.backend == "cuda"))
                report["resources"].append({"phase": phase, "snapshot": _proto_json(value.snapshot)})
            snapshot("before_model")
            model = inference.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                graph_path=str(args.graph.resolve()), weight_paths=[str(args.weights.resolve())]))
            compiled = inference.compile_model(pb.CompileModelRequest(model_id=model.model_id,
                policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                    backends=[args.backend], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
            route(compiled.report)
            report["compile_report"] = _proto_json(compiled.report)
            snapshot("loaded_compiled")
            def initial(case):
                encoded = text.encode_text(pb.EncodeTextRequest(tokenizer_id=tokenizer_id,
                    text=case["prompt"], add_special_tokens=case["add_special_tokens"]))
                require(encoded.tokens == case["input_ids"], "Generate prompt differs from independent HF encoding")
                values = {"token_ids": np.array([encoded.tokens], dtype=np.int32),
                    "position_ids": np.arange(len(encoded.tokens), dtype=np.int32),
                    "attention_mask": np.ones((1, len(encoded.tokens)), dtype=np.int32)}
                return pb.GenerationInputs(values=[pb.Tensor(name=name, dtype=pb.DataType.DATA_TYPE_I32,
                    shape=list(value.shape), inline=value.tobytes()) for name, value in values.items()])
            def run(case, label, chunks, *, stops=None, cancel=False):
                policy = case["selection"]
                options = {key: value for key, value in policy.items() if key not in ("sampling", "name")}
                options.update(stops or {})
                if "stop_sequences" in options:
                    options["stop_sequences"] = [pb.IdStopSequence(ids=value) for value in options["stop_sequences"]]
                context = inference.create_execution_context(pb.CreateExecutionContextRequest(
                    compiled_model_id=compiled.compiled_model_id, autoregressive=pb.AutoregressiveExecution(
                        sampling=pb.SamplingMode.SAMPLING_MODE_GREEDY if policy["sampling"] == "GREEDY"
                        else pb.SamplingMode.SAMPLING_MODE_RANDOM, max_steps=STEPS, **options)))
                packets, selected, raw_parts, row = [], [], [], {"case": case["name"], "variant": label,
                    "chunks": chunks, "packets": [], "selection": policy}
                try:
                    if label == "one-call":
                        # A positive budget smaller than one I32 ID cannot
                        # commit conditioning or consume a RANDOM draw.
                        before = inference.get_execution_context_info(pb.ExecutionContextRef(context_id=context.context_id))
                        try:
                            inference.generate(pb.GenerateRequest(context_id=context.context_id,
                                initial_inputs=initial(case), budget=pb.GenerationBudget(max_steps=4, max_output_bytes=3)))
                        except vx.VolvoxAIError as error:
                            require(error.status == pb.NativeStatus.NATIVE_STATUS_OVERLOADED,
                                    "sub-ID output budget failed with the wrong status")
                            row["sub_id_budget_refusal"] = _proto_json(error.report)
                        else:
                            raise ValueError("sub-ID output budget unexpectedly committed")
                        after = inference.get_execution_context_info(pb.ExecutionContextRef(context_id=context.context_id))
                        require(not after.initialized and after.total_steps == 0 and after.state == before.state
                                and after.bindings_available == before.bindings_available,
                                "output-budget refusal mutated initial context")
                    for index, chunk in enumerate(chunks):
                        budget = pb.GenerationBudget(**chunk)
                        packet = inference.generate(pb.GenerateRequest(context_id=context.context_id,
                            initial_inputs=initial(case) if not index else None, budget=budget))
                        route(packet.report)
                        require(len(packet.outputs) == 1, "Generate output count changed")
                        output = packet.outputs[0]
                        require(output.name == "generated_ids" and output.dtype == pb.DataType.DATA_TYPE_I32,
                                "Generate output contract changed")
                        require(not output.HasField("buffer"), "default output must be an immutable INLINE snapshot")
                        ids = np.frombuffer(output.inline, dtype=np.int32).tolist()
                        require(output.shape == [1, len(ids)] and len(ids) == packet.progress.steps, "ID output shape/count mismatch")
                        require(len(output.inline) <= chunk.get("max_output_bytes", 2**63), "output byte budget exceeded")
                        require(ids == case["ids"][len(selected):len(selected)+len(ids)], "Generate selected IDs differ from independent HF policy")
                        packets.append((packet, bytes(output.inline)))
                        selected.extend(ids)
                        require(packet.progress.total_steps == len(selected), "chunk progress lost prior IDs")
                        row["packets"].append({"ids": ids, "bytes": len(output.inline), "progress": _proto_json(packet.progress),
                                               "report": _proto_json(packet.report)})
                        if byte_level:
                            raw_parts.append(raw(ids, True))
                        if cancel:
                            inference.cancel_execution_context(pb.ExecutionContextRef(context_id=context.context_id))
                            before = inference.get_execution_context_info(pb.ExecutionContextRef(context_id=context.context_id))
                            require(before.total_steps == len(selected) and before.stop_reason == pb.ExecutionStopReason.EXECUTION_STOP_REASON_CANCELLED,
                                    "cancellation did not terminally retain completed progress")
                            try:
                                inference.generate(pb.GenerateRequest(context_id=context.context_id,
                                    budget=pb.GenerationBudget(max_steps=1)))
                            except vx.VolvoxAIError:
                                pass
                            else:
                                raise ValueError("cancelled context generated another ID")
                            after = inference.get_execution_context_info(pb.ExecutionContextRef(context_id=context.context_id))
                            require(after.total_steps == before.total_steps, "retry mutated cancelled progress")
                            row["cancel_info"] = _proto_json(after)
                            break
                        if not packet.progress.can_continue:
                            break
                    expected = case["ids"][:len(selected)]
                    require(selected == expected and selected, "empty or mismatched generation")
                    if not cancel:
                        if stops:
                            expected_count = 2 if "stop_sequences" in stops else 1
                            reason = pb.ExecutionStopReason.EXECUTION_STOP_REASON_EOS if "eos_ids" in stops else pb.ExecutionStopReason.EXECUTION_STOP_REASON_ID_STOP
                            require(len(selected) == expected_count and packets[-1][0].progress.stop_reason == reason,
                                    "generated-ID stop/EOS precedence differs from public contract")
                        else:
                            require(len(selected) == STEPS and packets[-1][0].progress.stop_reason == pb.ExecutionStopReason.EXECUTION_STOP_REASON_STEP_LIMIT,
                                    "fixed generation budget did not finish exactly four IDs")
                    for skip in (False, True):
                        expected_text = case["decode"][str(skip)] if len(selected) == STEPS else None
                        if expected_text is not None:
                            require(decode(selected, skip) == expected_text, "generated whole text differs from HF")
                            require(decode(case["input_ids"] + selected, skip) == case["whole_decode"][str(skip)],
                                    "contextual prompt+generation whole decode differs from HF")
                        if byte_level:
                            parts = [raw(np.frombuffer(packet.outputs[0].inline, dtype=np.int32).tolist(), skip) for packet, _ in packets]
                            stream = utf8_stream(parts)
                            require(stream["text"] == decode(selected, skip), "Generate packet UTF-8 differs from whole decode")
                            if len(selected) == STEPS:
                                require(b"".join(parts).hex() == case["raw_hex"][str(skip)], "Generate raw bytes differ from independent HF inverse")
                                prompt_parts = [raw([token], skip) for token in case["input_ids"]]
                                whole = utf8_stream(prompt_parts + parts)
                                require(whole["text"] == case["whole_decode"][str(skip)], "prompt+Generate text changed at packet boundaries")
                                require(b"".join(prompt_parts + parts).hex() == case["whole_raw_hex"][str(skip)], "prompt+Generate raw bytes changed")
                            row.setdefault("stream", {})[str(skip)] = stream
                    if cancel and byte_level:
                        row["caller_cancel_finalize"] = utf8_stream(raw_parts, "finalize")
                        row["caller_cancel_discard"] = utf8_stream(raw_parts, "discard")
                    row["ids"], row["status"] = selected, "pass"
                finally:
                    inference.release_execution_context(pb.ExecutionContextRef(context_id=context.context_id))
                require(all(bytes(packet.outputs[0].inline) == frozen for packet, frozen in packets),
                        "retained INLINE output changed after owner retirement")
                row["retained_inline_after_owner_retirement"] = True
                report["generation_checks"].append(row)
                persist()
                print(f"PASS {args.model} {args.profile}/{args.backend} {case['name']} {label} {selected}", flush=True)
            for case in reference["generation_cases"]:
                run(case, "one-call", [{"max_steps": 4}])
                run(case, "step-chunks", [{"max_steps": 1}, {"max_steps": 2}, {"max_steps": 1}])
                run(case, "output-byte-chunks", [{"max_steps": 4, "max_output_bytes": 8},
                    {"max_steps": 4, "max_output_bytes": 4}, {"max_steps": 4, "max_output_bytes": 4}])
                if case["selection"]["sampling"] == "GREEDY":
                    first = case["ids"][0]
                    run(case, "eos-precedence", [{"max_steps": 4}], stops={"eos_ids": [first], "stop_ids": [first]})
                    run(case, "id-stop", [{"max_steps": 4}], stops={"stop_ids": [first]})
                    run(case, "sequence-across-calls", [{"max_steps": 1}, {"max_steps": 3}],
                        stops={"stop_sequences": [case["ids"][:2]]})
                # Smol's real Korean/emoji greedy continuations end this
                # two-ID chunk with incomplete UTF-8. RANDOM variants also
                # prove cancellation and a fresh context's seed/state isolation.
                run(case, "cancel-after-first-chunk", [{"max_steps": 2}], cancel=True)
                run(case, "fresh-after-cancel", [{"max_steps": 4}])
            snapshot("before_model_retirement")
            inference.release_compiled_model(pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
            inference.release_model(pb.ModelRef(model_id=model.model_id))
            snapshot("after_model_retirement")
            final_memory = report["resources"][-1]["snapshot"]["memory"]
            require(int(final_memory.get("unconsumedResultBytes", -1)) == 0, "unconsumed results remained after retirement")
            require(not final_memory.get("truncated", False), "cleanup inventory is truncated")
            baseline = report["resources"][0]["snapshot"]["memory"]
            allocation_key = lambda value: sorted(value.get("allocations", []), key=lambda row: json.dumps(row, sort_keys=True))
            require(allocation_key(final_memory) == allocation_key(baseline), "known allocation inventory did not return to baseline")
            report["known_inventory_cleanup"] = {"status": "pass", "unconsumed_result_bytes": 0,
                "scope": "known runtime allocation inventory; whole-process RSS remains separate"}
            inference.release_runtime(pb.RuntimeRef(runtime_id=runtime.runtime_id))
            text.release_tokenizer(pb.TokenizerRef(tokenizer_id=tokenizer_id))
        report["status"] = "pass"
        report["generation_case_count"] = len(report["generation_checks"])
        report["actual_generate_calls"] = sum(len(row["packets"]) for row in report["generation_checks"])
        report["sub_id_budget_refusals"] = sum("sub_id_budget_refusal" in row for row in report["generation_checks"])
        report["actual_selected_ids"] = sum(len(row["ids"]) for row in report["generation_checks"])
        report["sampling_comparison_scope"] = "selected HF F32 CDF witnesses; same-backend seed/chunk identity, no universal cross-backend RANDOM equality"
        persist()
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        persist()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("capture", "replay"):
        p = sub.add_parser(command)
        p.add_argument("--model", required=True)
        p.add_argument("--policy", type=Path, default=ROOT / "tools/llm_baseline/models.json")
        p.add_argument("--snapshot", type=Path, required=True)
        p.add_argument("--threads", type=int, default=2)
        p.add_argument("--out", type=Path, required=True)
    p = sub.choices["replay"]
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--loader", type=Path)
    p.add_argument("--profile", choices=("full", "inference"), required=True)
    p.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        require(not args.out.exists(), "prepared graph must be a new file")
        prepare_graph(args.graph, json.loads(args.config.read_text()), args.out)
    else:
        require(args.threads > 0, "positive CPU thread count required")
        (capture if args.command == "capture" else replay)(args)


if __name__ == "__main__":
    main()
