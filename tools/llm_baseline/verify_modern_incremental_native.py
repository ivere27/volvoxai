#!/usr/bin/env python3
"""Check REQUIRED cached public contexts against retained independent HF oracles.

Generate the corpus with generate_modern_incremental_stress.py, which shares
independent HF fixtures with the incremental numerical tests.
This runner needs NumPy and the public SDK, but no reference model or weights
download. Replay checks preserved rows, continued generation and context isolation.
Policies check capacity refusal/retry and same-backend seeded sampling across
chunks and owners; they do not claim independent-HF or cross-backend random equality.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import volvoxai as vx
from verify_modern_native_gpu import route, source_snapshot


def tensor(name, array):
    return vx.pb.Tensor(name=name, shape=list(array.shape), inline=array.tobytes(),
        dtype=vx.pb.DataType.DATA_TYPE_I32 if array.dtype == np.int32 else vx.pb.DataType.DATA_TYPE_F32)


def replay(case, host, backend):
    pb, client = vx.pb, vx.VxInferenceServiceClient(host)
    values = {item["name"]:np.frombuffer(base64.b64decode(item["data"], validate=True),
        np.int32 if item["dtype"] in ("int32", "I32") else np.float32).reshape(item["shape"]).copy()
        for item in case["inputs"]}
    prompt_length = int(values["active_lengths"][0])
    maximum_error = 0.0
    with ExitStack() as owners:
        runtime = client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
        owners.callback(client.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id))
        model = client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
            package=pb.ModelPackage(graph_document=json.dumps(case["graph"], allow_nan=False).encode(),
                weight_shards=[base64.b64decode(data, validate=True) for data in case["weight_shards"]])))
        owners.callback(client.release_model, pb.ModelRef(model_id=model.model_id))
        info = client.get_model_info(pb.ModelRef(model_id=model.model_id))
        assert info.execution.generation.incremental_controls.position_input == "position_ids"
        compiled = client.compile_model(pb.CompileModelRequest(model_id=model.model_id,
            policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                backends=[backend], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
        owners.callback(client.release_compiled_model, pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
        witness = route(compiled.report, pb, backend)
        context = client.create_execution_context(pb.CreateExecutionContextRequest(
            compiled_model_id=compiled.compiled_model_id,
            graph=pb.GraphExecution(slots=1, row_mode=pb.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,
                require_incremental=True)))
        owners.callback(client.release_execution_context, pb.ExecutionContextRef(context_id=context.context_id))
        bindings = {name:value.copy() for name,value in values.items()}
        invariants = case["graph"]["execution"]["invariant_inputs"]
        previous = None
        for step, expected in enumerate(case["expected_logits"]):
            row = prompt_length - 1 + step
            if step:
                bindings["token_ids"][0, row] = case["expected_ids"][step - 1]
                bindings["attention_mask"][0, row] = 1
                for name in ("image_feature_ids", "image_token_mask"):
                    if name in bindings:
                        bindings[name].reshape(-1)[row] = 0
            packet = client.execute(pb.ExecuteRequest(context_id=context.context_id,
                inputs=pb.InputBindings(values=[tensor(name,value) for name,value in bindings.items()
                    if step == 0 or name not in invariants], reuse=invariants if step else []),
                scope=pb.ExecutionScope(entrypoint="initialize", full=pb.Empty()) if step == 0 else
                    pb.ExecutionScope(entrypoint="transition", slot_rows=pb.SlotRows(slots=[pb.SlotRow(position=row)])),
                outputs=pb.OutputOptions(selection=pb.TensorOutputSelection(names=["logits"]),
                    storage=pb.OutputStorage.OUTPUT_STORAGE_INLINE)))
            route(packet.report, pb, backend)
            output = next(item for item in packet.outputs if item.name == "logits")
            actual = np.frombuffer(output.inline, np.float32).reshape(output.shape).copy()
            np.testing.assert_allclose(actual[0,row], expected, atol=2e-4, rtol=2e-4,
                err_msg=f"{case['name']}/{backend}/row{row}")
            maximum_error = max(maximum_error, float(np.max(np.abs(actual[0,row] - expected))))
            if previous is not None:
                np.testing.assert_array_equal(actual[0,:row], previous[0,:row])
            previous = actual
        contexts, observed = [], [[], []]
        for _ in range(2):
            context = client.create_execution_context(pb.CreateExecutionContextRequest(
                compiled_model_id=compiled.compiled_model_id,
                autoregressive=pb.AutoregressiveExecution(sampling=pb.SamplingMode.SAMPLING_MODE_GREEDY,
                    max_steps=len(case["expected_ids"]))))
            owners.callback(client.release_execution_context, pb.ExecutionContextRef(context_id=context.context_id))
            contexts.append(context.context_id)
        for chunk, count in enumerate(case["budgets"]):
            for owner, context_id in enumerate(contexts):
                packet = client.generate(pb.GenerateRequest(context_id=context_id,
                    initial_inputs=pb.GenerationInputs(values=[tensor(name,value) for name,value in values.items()]) if chunk == 0 else None,
                    budget=pb.GenerationBudget(max_steps=count),
                    outputs=pb.OutputOptions(storage=pb.OutputStorage.OUTPUT_STORAGE_INLINE)))
                route(packet.report, pb, backend)
                output = next(item for item in packet.outputs if item.name == "generated_ids")
                observed[owner].extend(np.frombuffer(output.inline, np.int32).tolist())
        assert observed == [case["expected_ids"], case["expected_ids"]], observed
    return {"name":case["name"], "pass":True, "maximum_error":maximum_error,
        "route":witness, "rows":len(case["expected_logits"]), "generation_ids":observed}


def values(items):
    return {item["name"]:np.frombuffer(base64.b64decode(item["data"], validate=True),
        np.int32 if item["dtype"] in ("int32", "I32") else np.float32).reshape(item["shape"]).copy()
        for item in items}


def check_policies(case, host, backend, *, boundary, random):
    pb, client = vx.pb, vx.VxInferenceServiceClient(host)
    with ExitStack() as owners:
        runtime = client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
        owners.callback(client.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id))
        model = client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
            package=pb.ModelPackage(graph_document=json.dumps(case["graph"], allow_nan=False).encode(),
                weight_shards=[base64.b64decode(data, validate=True) for data in case["weight_shards"]])))
        owners.callback(client.release_model, pb.ModelRef(model_id=model.model_id))
        compiled = client.compile_model(pb.CompileModelRequest(model_id=model.model_id,
            policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                backends=[backend], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
        owners.callback(client.release_compiled_model, pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
        witness = route(compiled.report, pb, backend)
        record = {"name":case["name"], "pass":False, "route":witness}

        def context(maximum, sampling, **options):
            packet = client.create_execution_context(pb.CreateExecutionContextRequest(
                compiled_model_id=compiled.compiled_model_id,
                autoregressive=pb.AutoregressiveExecution(max_steps=maximum, sampling=sampling, **options)))
            owners.callback(client.release_execution_context, pb.ExecutionContextRef(context_id=packet.context_id))
            return packet.context_id

        def generate(context_id, count, initial=None):
            packet = client.generate(pb.GenerateRequest(context_id=context_id,
                initial_inputs=pb.GenerationInputs(values=[tensor(name,value) for name,value in initial.items()])
                    if initial is not None else None,
                budget=pb.GenerationBudget(max_steps=count),
                outputs=pb.OutputOptions(storage=pb.OutputStorage.OUTPUT_STORAGE_INLINE)))
            route(packet.report, pb, backend)
            assert len(packet.outputs) == 1
            output = packet.outputs[0]
            assert output.dtype == pb.DataType.DATA_TYPE_I32 and not output.HasField("buffer")
            assert list(output.shape) == [1,count] and len(output.inline) == 4 * count
            return np.frombuffer(output.inline, np.int32).tolist()

        def state(context_id):
            info = client.get_execution_context_info(pb.ExecutionContextRef(context_id=context_id))
            return {"initialized":info.initialized, "total_steps":info.total_steps, "state":int(info.state)}

        def refuse(context_id, count, initial=None):
            before = state(context_id)
            try:
                generate(context_id, count, initial)
            except vx.VolvoxAIError as error:
                assert error.status == pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT, error.report
            else:
                raise AssertionError("capacity overflow unexpectedly accepted")
            assert state(context_id) == before, "capacity rejection mutated generation progress"
            return {"requested_steps":count, "state_before_after":before, "status":"INVALID_ARGUMENT"}

        if boundary:
            expected = case["boundary"]["expected_ids"]
            assert len(expected) == 2
            initial = values(case["boundary"]["inputs"])
            capacity = int(initial["token_ids"].shape[1])
            assert int(initial["active_lengths"][0]) == capacity - 1
            current = context(4, pb.SamplingMode.SAMPLING_MODE_GREEDY)
            refused = [refuse(current, 3, initial)]
            observed = generate(current, 1, initial)
            assert observed == expected[:1], "boundary first ID differs from independent HF"
            refused.append(refuse(current, 2))
            observed.extend(generate(current, 1))
            assert observed == expected, "boundary continuation ID differs from independent HF"
            refused.append(refuse(current, 1))
            record["boundary"] = {"capacity":capacity, "prompt_length":capacity - 1,
                "generated_ids":observed, "independent_hf_ids":expected,
                "refusals":refused, "retry_after_initial_and_continuation_refusal":True}

        if random:
            initial = values(case["inputs"])
            steps = len(case["expected_ids"])
            options = ({"seed":7, "temperature":0.7, "top_k":5, "top_p":0.8},
                {"seed":2**64-1, "temperature":1.5, "top_k":0, "top_p":1.0})
            baselines = []
            for option in options:
                owner = context(steps, pb.SamplingMode.SAMPLING_MODE_RANDOM, **option)
                baselines.append(generate(owner, steps, initial))
            split = [context(steps, pb.SamplingMode.SAMPLING_MODE_RANDOM, **option)
                for option in (options[0], options[1], options[0])]
            observed = [[],[],[]]
            for chunk, count in enumerate(case["budgets"]):
                for owner, current in enumerate(split):
                    observed[owner].extend(generate(current, count, initial if chunk == 0 else None))
            assert observed == [baselines[0],baselines[1],baselines[0]], "RNG depends on chunks or another context"
            vocabulary = len(case["expected_logits"][0])
            assert all(0 <= token < vocabulary for row in observed for token in row)
            record["random"] = {"sampling_options":list(options), "single_chunk_ids":baselines,
                "interleaved_chunk_ids":observed, "budgets":case["budgets"],
                "same_backend_seed_and_chunk_invariance":True,
                "independent_random_oracle":False, "cross_backend_equality_claim":False}
        record["pass"] = True
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "policies"), default="replay")
    parser.add_argument("--skip-random", action="store_true", help="policies: check capacity only")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan", "opengl", "metal"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    if args.skip_random and args.mode != "policies":
        parser.error("--skip-random requires --mode policies")
    if args.mode == "policies":
        if args.backend == "metal":
            parser.error("policies are qualified only on CPU/CUDA/Vulkan/OpenGL")
        if args.backend not in ("cpu", "cuda") and not args.skip_random:
            parser.error("RANDOM is qualified only on CPU/CUDA; use --skip-random")
    tracked = {"fixtures":args.fixtures, "library":args.library, "runner":Path(__file__)}
    hashes = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in tracked.items()}
    sources = source_snapshot(ROOT)
    receipt = {"status":"FAIL", "mode":args.mode, "backend":args.backend, "sha256":hashes, "source_before":sources,
        "results":[], "timing_claim":False, "physical_memory_peak_claim":False}
    try:
        document = json.loads(args.fixtures.read_text())
        assert document["format"] == "volvox-modern-incremental/v1"
        receipt["reference_policy"] = document.get("reference_policy", "raw-hf")
        cases = [case for case in document["cases"] if not args.only or args.only in case["name"]]
        assert cases
        seen = set()
        with vx.open_library(args.library.resolve()) as host:
            for case in cases:
                if args.mode == "replay":
                    result = replay(case, host, args.backend)
                else:
                    family = case["name"].rsplit("-prompt", 1)[0]
                    result = check_policies(case, host, args.backend,
                        boundary=family not in seen, random=not args.skip_random)
                    seen.add(family)
                result["reference_policy"] = case.get("reference_policy", "raw-hf")
                receipt["results"].append(result)
                print("PASS", args.backend, args.mode, case["name"], flush=True)
        assert hashes == {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in tracked.items()}
        receipt["source_after"] = source_snapshot(ROOT)
        assert receipt["source_after"] == sources, "source changed during replay"
        receipt["status"] = "PASS"
    except Exception:
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
