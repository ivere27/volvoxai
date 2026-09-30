#!/usr/bin/env python3
"""Check debug stepping against ordinary inference on a required native backend."""
import argparse
import array
import json
from pathlib import Path
import sys
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
VERSION = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))["version"]
sys.path.insert(0, str(ROOT / "python"))
import volvoxai as vx  # noqa: E402

p = vx.pb


def qualify(backend, library, output):
    width, depth = 256, 12
    graph = {"format": "volvox-graph/v1", "dimensions": {},
             "inputs": {"x": {"dtype": "float32", "shape": [width // 2, 2]}}, "nodes": [], "outputs": [f"t{depth - 1}"]}
    for index in range(depth):
        graph["nodes"].append({"id": f"node{index}", "opType": "Linear",
            "inputs": {"input": "x" if index == 0 else f"t{index - 1}", "weight": "weight", "bias": "bias"},
            "outputs": {"out": {"tensor": f"t{index}", "dtype": "float32", "shape": [width // 2, 2]}},
            "params": {"weight_layout": "dout_din"}})
    header = json.dumps({"weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
                         "bias": {"dtype": "F32", "shape": [2], "data_offsets": [16, 24]}}).encode()
    header += b" " * (-len(header) % 8)
    weights = struct.pack("<Q", len(header)) + header + struct.pack("<6f", 1, 2, -3, 4, .25, -.5)
    reports = []
    with vx.open_library(ROOT / library) as host:
        inference, debug = vx.VxInferenceServiceClient(host), vx.VxDebugServiceClient(host)
        runtime = inference.create_runtime(p.CreateRuntimeRequest(cpu_threads=1))
        model = inference.load_model(p.LoadModelRequest(runtime_id=runtime.runtime_id,
            package=p.ModelPackage(graph_document=json.dumps(graph).encode(), weight_shards=[weights])))
        for preserve in [False, True]:
            compiled = inference.compile_model(p.CompileModelRequest(model_id=model.model_id, preserve_node_boundaries=preserve,
                policy=p.BackendPolicy(mode=p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends=[backend],
                    operator_fallback=p.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
            for iteration in range(10):
                values = [(i % 17 - 8 + iteration) / 4 for i in range(width)]
                tensor = p.Tensor(name="x", dtype=p.DataType.DATA_TYPE_F32, shape=[width // 2, 2], inline=array.array("f", values).tobytes())
                started = time.perf_counter()
                info = debug.create_debug_session(p.CreateDebugSessionRequest(
                    forward=p.DebugForward(compiled_model_id=compiled.compiled_model_id, inputs=[tensor]),
                    capture=p.DebugCapture(values=True)))
                ref = p.DebugSessionRef(debug_session_id=info.debug_session_id)
                assert info.state == p.DebugState.DEBUG_STATE_PAUSED and info.next_step == 0
                assert info.stop_reason == p.DebugStopReason.DEBUG_STOP_REASON_ENTRY
                plan = debug.get_debug_plan(ref).plan
                assert plan.backend == backend and len(plan.steps) == depth
                assert info.capabilities.preserves_node_boundaries == preserve
                info = debug.step_debug_session(p.StepDebugSessionRequest(debug_session_id=info.debug_session_id, expected_revision=info.revision))
                assert info.next_step == 1 and info.events.count == 2
                early = debug.read_debug_tensor(p.ReadDebugTensorRequest(debug_session_id=info.debug_session_id, snapshot_id=4)).data
                first = [value for i in range(0, width, 2) for value in
                         (values[i] + 2 * values[i + 1] + .25, -3 * values[i] + 4 * values[i + 1] - .5)]
                assert array.array("f", first).tobytes() == early
                # An ordinary execution while the debugger is paused must use independent storage.
                normal = inference.run(p.RunRequest(compiled_model_id=compiled.compiled_model_id, inputs=[tensor]))
                while inference.get_result(p.ResultRef(result_id=normal.result_id)).state == p.ResultState.RESULT_STATE_PENDING:
                    time.sleep(.001)
                expected = inference.read_output(p.ReadOutputRequest(result_id=normal.result_id, name=f"t{depth - 1}")).tensor.inline
                inference.release_result(p.ResultRef(result_id=normal.result_id))
                info = debug.continue_debug_session(p.ContinueDebugSessionRequest(debug_session_id=info.debug_session_id,
                    expected_revision=info.revision, break_before_nodes=[plan.steps[5].source_node_ids[0]]))
                assert info.next_step == 5 and info.state == p.DebugState.DEBUG_STATE_PAUSED
                assert info.stop_reason == p.DebugStopReason.DEBUG_STOP_REASON_BREAKPOINT
                info = debug.continue_debug_session(p.ContinueDebugSessionRequest(debug_session_id=info.debug_session_id, expected_revision=info.revision))
                assert info.state == p.DebugState.DEBUG_STATE_COMPLETED and info.capture_complete
                assert info.peak_bytes <= info.max_bytes and info.snapshots.dropped == 0
                events = debug.list_debug_events(p.ListDebugEventsRequest(debug_session_id=info.debug_session_id, page_size=4096)).events
                final = debug.read_debug_tensor(p.ReadDebugTensorRequest(debug_session_id=info.debug_session_id,
                    snapshot_id=events[-1].snapshots[-1].snapshot_id)).data
                assert final == expected, (backend, "ordinary output differs")
                assert debug.read_debug_tensor(p.ReadDebugTensorRequest(debug_session_id=info.debug_session_id, snapshot_id=4)).data == early
                debug.release_debug_session(ref)
                reports.append({"source_boundaries": preserve, "iteration": iteration, "steps": info.next_step,
                    "snapshots": info.snapshots.count, "peak_bytes": info.peak_bytes,
                    "elapsed_ms": (time.perf_counter() - started) * 1000, "output_exact": True})
            inference.release_compiled_model(p.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
    result = {"backend": backend, "library": library, "sessions": reports}
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{Path(library).name}-{backend}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def decoder(slots, sequence, width, quantized, layers=2):
    """Causal attention layers over q/k/v projections: F32 Linear + CrossSDPA,
    or W8A8 QLinear + QSDPA with affine int8 activations."""
    nodes, header, blobs, quantization = [], {}, [], {}
    dtype = "int8" if quantized else "float32"

    def add(name, shape, kind, values):
        offset = sum(len(blob) for blob in blobs)
        data = array.array({"F32": "f", "I8": "b", "I32": "i"}[kind], values).tobytes()
        header[name] = {"dtype": kind, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)

    def affine(name, per_axis):
        count = width if per_axis else 1
        add(f"{name}.scale", [count], "F32", [0.05] * count)
        add(f"{name}.zero", [count], "I8", [0] * count)
        quantization[name] = {"scheme": "per_axis" if per_axis else "per_tensor", **({"axis": 0} if per_axis else {}),
                              "scale_tensor": f"{name}.scale", "zero_point_tensor": f"{name}.zero"}

    if quantized:
        affine("x", False)
    source = "x"
    for layer in range(layers):
        for index, port in enumerate("qkv"):
            name = f"{port}{layer}"
            if quantized:
                add(f"{name}.weight", [width, width], "I8",
                    [19 if i % width == i // width else (i + index) % 5 - 2 for i in range(width * width)])
                add(f"{name}.bias", [width], "I32", [0] * width)
                affine(f"{name}.weight", True)
                affine(name, False)
            else:
                add(f"{name}.weight", [width, width], "F32",
                    [0.8 if i % width == i // width else ((i * 7 + index + layer) % 11 - 5) * 0.01 for i in range(width * width)])
                add(f"{name}.bias", [width], "F32", [0.0] * width)
            nodes.append({"id": name, "opType": "QLinear" if quantized else "Linear",
                          "inputs": {"input": source, "weight": f"{name}.weight", "bias": f"{name}.bias"},
                          "outputs": {"out": {"tensor": name, "shape": [slots, sequence, width], "dtype": dtype}}, "params": {}})
        name = "out" if layer == layers - 1 else f"attention{layer}"
        nodes.append({"id": name, "opType": "QSDPA" if quantized else "CrossSDPA",
                      "inputs": {"q": f"q{layer}", "k": f"k{layer}", "v": f"v{layer}", "mask": "keep"},
                      "outputs": {"out": {"tensor": name, "shape": [slots, sequence, width], "dtype": dtype}},
                      "params": {"heads": 2, "causal": True}})
        if quantized:
            affine(name, False)
        source = name
    graph = {"format": "volvox-graph/v1", "dimensions": {}, "nodes": nodes, "outputs": ["out"],
             "inputs": {"x": {"shape": [slots, sequence, width], "dtype": dtype}, "keep": {"shape": [slots, sequence], "dtype": "int32"}}}
    if quantized:
        graph["quantization"] = {"format": "volvox-affine-safetensors/v1", "tensors": quantization}
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    return graph, struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs)


def qualify_decode(backend, library, output):
    """A debugged DecodeStep must equal the ordinary one and expose its KV writes."""
    sequence, width, reports = 8, 64, []
    # Native GPUs qualify only the W8A8 attention; the CPU runs both.
    variants = [(quantized, slots) for quantized in ([False, True] if backend == "cpu" else [True]) for slots in (2, 1)]
    with vx.open_library(ROOT / library) as host:
        inference, debug = vx.VxInferenceServiceClient(host), vx.VxDebugServiceClient(host)
        runtime = inference.create_runtime(p.CreateRuntimeRequest(cpu_threads=1))
        done = set()
        for quantized, slots in variants:
            if quantized in done:
                continue
            element = 1 if quantized else 4
            graph, weights = decoder(slots, sequence, width, quantized)
            model = inference.load_model(p.LoadModelRequest(runtime_id=runtime.runtime_id,
                package=p.ModelPackage(graph_document=json.dumps(graph).encode(), weight_shards=[weights])))
            compiled = inference.compile_model(p.CompileModelRequest(model_id=model.model_id,
                policy=p.BackendPolicy(mode=p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends=[backend],
                    operator_fallback=p.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
            try:
                contexts = [inference.create_execution_context(p.CreateExecutionContextRequest(
                    compiled_model_id=compiled.compiled_model_id, decode_row_mode=p.DecodeRowMode.DECODE_ROW_MODE_AUTO,
                    decode_slots=slots, decode_inputs=["x", "keep"])).context_id for _ in range(2)]
            except vx.VolvoxAIError:
                if slots == 1:
                    raise
                continue  # No multi-slot row path on this backend; qualify one slot.
            lengths = [4, 2][:slots]

            def feed(counts, seed):
                x = array.array("b" if quantized else "f",
                    [((i * 13 + seed * 7) % 23 - 11) * (1 if quantized else 1 / 8) for i in range(slots * sequence * width)])
                keep = array.array("i", [1 if i % sequence < counts[i // sequence] else 0 for i in range(slots * sequence)])
                return [p.Tensor(name="x", dtype=p.DataType.DATA_TYPE_I8 if quantized else p.DataType.DATA_TYPE_F32,
                                 shape=[slots, sequence, width], inline=x.tobytes()),
                        p.Tensor(name="keep", dtype=p.DataType.DATA_TYPE_I32, shape=[slots, sequence], inline=keep.tobytes())]

            def ready(result):
                while inference.get_result(p.ResultRef(result_id=result.result_id)).state == p.ResultState.RESULT_STATE_PENDING:
                    time.sleep(.001)
                data = inference.read_output(p.ReadOutputRequest(result_id=result.result_id, name="out")).tensor.inline
                inference.release_result(p.ResultRef(result_id=result.result_id))
                return data

            for context in contexts:
                ready(inference.decode_prefill(p.DecodePrefillRequest(context_id=context, inputs=feed(lengths, 0),
                    slot_positions=p.DecodeSlotPositions(positions=[n - 1 for n in lengths]))))
            actions = p.DecodeSlotActions(slots=[p.DecodeSlotAction(position=n) for n in lengths])
            step = lambda context: p.DecodeStepRequest(context_id=context, slot_actions=actions, inputs=feed([n + 1 for n in lengths], 1))
            started = time.perf_counter()
            # The ordinary step decides the expectation: its output, or its refusal.
            try:
                expected, refusal = ready(inference.decode_step(step(contexts[1]))), None
            except vx.VolvoxAIError as error:
                expected, refusal = None, error.report
            if backend != "cpu" and refusal is None:
                # Comparing the debugger with the same device alone cannot see a
                # device that is wrong both ways; the CPU is the reference.
                cpu = inference.compile_model(p.CompileModelRequest(model_id=model.model_id,
                    policy=p.BackendPolicy(mode=p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends=["cpu"])))
                oracle = inference.create_execution_context(p.CreateExecutionContextRequest(
                    compiled_model_id=cpu.compiled_model_id, decode_row_mode=p.DecodeRowMode.DECODE_ROW_MODE_AUTO,
                    decode_slots=slots, decode_inputs=["x", "keep"])).context_id
                ready(inference.decode_prefill(p.DecodePrefillRequest(context_id=oracle, inputs=feed(lengths, 0),
                    slot_positions=p.DecodeSlotPositions(positions=[n - 1 for n in lengths]))))
                assert ready(inference.decode_step(step(oracle))) == expected, (backend, "ordinary step differs from CPU")
                inference.release_execution_context(p.ExecutionContextRef(context_id=oracle))
                inference.release_compiled_model(p.CompiledModelRef(compiled_model_id=cpu.compiled_model_id))
            info = debug.create_debug_session(p.CreateDebugSessionRequest(decode_step=step(contexts[0])))
            ref = p.DebugSessionRef(debug_session_id=info.debug_session_id)
            assert info.target == p.DebugTarget.DEBUG_TARGET_DECODE_STEP
            assert info.state == p.DebugState.DEBUG_STATE_PAUSED and info.stop_reason == p.DebugStopReason.DEBUG_STOP_REASON_ENTRY
            try:
                inference.decode_step(step(contexts[0]))
            except vx.VolvoxAIError as error:
                assert error.report.status == p.NativeStatus.NATIVE_STATUS_BUSY
            else:
                raise AssertionError("an attached context must refuse other operations")
            state = debug.get_debug_decode_state(ref)
            key = state.caches[0]
            assert len(state.caches) == 4 and key.role == p.DebugKVRole.DEBUG_KV_ROLE_KEY and not key.written

            if refusal is not None:
                # A refused step fails the session the same way and resets the context.
                info = debug.continue_debug_session(p.ContinueDebugSessionRequest(debug_session_id=ref.debug_session_id, expected_revision=info.revision))
                assert info.state == p.DebugState.DEBUG_STATE_FAILED and not info.result_id, (backend, info)
                assert (info.failure.message, info.failure.offending_node) == (refusal.message, refusal.offending_node), (info.failure, refusal)
                debug.release_debug_session(ref)
                assert not inference.get_decode_state(p.ExecutionContextRef(context_id=contexts[0])).prefilled
                done.add(quantized)
                reports.append({"dtype": "I8" if quantized else "F32", "slots": slots, "refused": refusal.message,
                                "failure_parity": True, "elapsed_ms": (time.perf_counter() - started) * 1000})
                inference.release_compiled_model(p.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                continue

            def rows(cache):
                return debug.read_debug_kv_cache(p.ReadDebugKVCacheRequest(debug_session_id=ref.debug_session_id,
                    cache_id=cache.cache_id, slot=0, token_limit=4096)).data

            before = rows(key)
            while info.next_step <= key.writer_step:
                info = debug.step_debug_session(p.StepDebugSessionRequest(debug_session_id=ref.debug_session_id, expected_revision=info.revision))
                assert info.state == p.DebugState.DEBUG_STATE_PAUSED, (backend, info.failure)
            after, row = rows(key), lengths[0] * width * element
            assert debug.get_debug_decode_state(ref).caches[0].written
            assert after[:row] == before[:row] and after[row:] != before[row:], (backend, "KV write outside its row")
            info = debug.continue_debug_session(p.ContinueDebugSessionRequest(debug_session_id=ref.debug_session_id, expected_revision=info.revision))
            assert info.state == p.DebugState.DEBUG_STATE_COMPLETED and info.result_id
            debugged = ready(p.ExecutionResultHandle(result_id=info.result_id))
            debug.release_debug_session(ref)
            assert debugged == expected, (backend, "debugged step differs")

            def committed(context):
                """Every cache row the finished step left, read at the next step's ENTRY stop."""
                following = p.DecodeStepRequest(context_id=context, inputs=feed([n + 2 for n in lengths], 2),
                    slot_actions=p.DecodeSlotActions(slots=[p.DecodeSlotAction(position=n + 1) for n in lengths]))
                session = debug.create_debug_session(p.CreateDebugSessionRequest(decode_step=following))
                current = p.DebugSessionRef(debug_session_id=session.debug_session_id)
                data = [debug.read_debug_kv_cache(p.ReadDebugKVCacheRequest(debug_session_id=current.debug_session_id,
                    cache_id=cache.cache_id, slot=slot, token_limit=4096)).data[:(lengths[slot] + 1) * width * element]
                    for cache in debug.get_debug_decode_state(current).caches for slot in range(slots)]
                debug.release_debug_session(current)
                return data

            assert committed(contexts[0]) == committed(contexts[1]), (backend, "debugged KV differs from the ordinary step")
            done.add(quantized)
            reports.append({"dtype": "I8" if quantized else "F32", "slots": slots, "steps_visited": info.events.count // 2, "step_count": info.step_count,
                            "elapsed_ms": (time.perf_counter() - started) * 1000, "output_exact": True, "kv_exact": True,
                            "cpu_exact": backend != "cpu"})
            inference.release_compiled_model(p.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
    result = {"backend": backend, "library": library, "decode": reports}
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{Path(library).name}-{backend}-decode.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def qualify_decode_sequence(backend, library, steps=3):
    """Consecutive ordinary GPU decode steps must equal the CPU byte for byte.

    Stale per-step device state -- a row list, a mask -- is invisible to one
    step and to comparing a device with itself; it shows from the second step.
    Two widths change the host allocation pattern those slots are keyed by."""
    slots, sequence, results = 2, 8, []
    for width in (32, 64):
        graph, weights = decoder(slots, sequence, width, True)
        outputs = {}
        for target in ("cpu", backend):
            with vx.open_library(ROOT / library) as host:
                inference = vx.VxInferenceServiceClient(host)
                runtime = inference.create_runtime(p.CreateRuntimeRequest(cpu_threads=1))
                model = inference.load_model(p.LoadModelRequest(runtime_id=runtime.runtime_id,
                    package=p.ModelPackage(graph_document=json.dumps(graph).encode(), weight_shards=[weights])))
                compiled = inference.compile_model(p.CompileModelRequest(model_id=model.model_id,
                    policy=p.BackendPolicy(mode=p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends=[target],
                        operator_fallback=p.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
                context = inference.create_execution_context(p.CreateExecutionContextRequest(
                    compiled_model_id=compiled.compiled_model_id, decode_row_mode=p.DecodeRowMode.DECODE_ROW_MODE_AUTO,
                    decode_slots=slots, decode_inputs=["x", "keep"])).context_id

                def feed(counts, seed):
                    x = array.array("b", [(i * 13 + seed * 7) % 23 - 11 for i in range(slots * sequence * width)])
                    keep = array.array("i", [1 if i % sequence < counts[i // sequence] else 0 for i in range(slots * sequence)])
                    return [p.Tensor(name="x", dtype=p.DataType.DATA_TYPE_I8, shape=[slots, sequence, width], inline=x.tobytes()),
                            p.Tensor(name="keep", dtype=p.DataType.DATA_TYPE_I32, shape=[slots, sequence], inline=keep.tobytes())]

                def read(result):
                    data = inference.read_output(p.ReadOutputRequest(result_id=result.result_id, name="out")).tensor.inline
                    inference.release_result(p.ResultRef(result_id=result.result_id))
                    return data

                sequence_outputs = [read(inference.decode_prefill(p.DecodePrefillRequest(context_id=context,
                    inputs=feed([4, 2], 0), slot_positions=p.DecodeSlotPositions(positions=[3, 1]))))]
                for step in range(steps):
                    sequence_outputs.append(read(inference.decode_step(p.DecodeStepRequest(context_id=context,
                        inputs=feed([5 + step, 3 + step], 1 + step), slot_actions=p.DecodeSlotActions(
                            slots=[p.DecodeSlotAction(position=4 + step), p.DecodeSlotAction(position=2 + step)])))))
                outputs[target] = sequence_outputs
        for index, (actual, expected) in enumerate(zip(outputs[backend], outputs["cpu"])):
            assert actual == expected, (backend, width, "prefill" if index == 0 else f"step {index}", "differs from CPU")
        results.append({"width": width, "steps": steps, "cpu_exact": True})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "cuda", "opengl", "vulkan"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "build/debugging")
    args = parser.parse_args()
    # Node debugging is full-profile only; the inference library must refuse it.
    if args.backend == "cpu":
        with vx.open_library(ROOT / f"native/libvolvoxai-lite.so.{VERSION}") as lite:
            try:
                vx.VxDebugServiceClient(lite).get_debug_session(p.DebugSessionRef(debug_session_id=1))
            except Exception:
                pass
            else:
                raise AssertionError("the inference library must not serve VxDebugService")
    for library in [f"native/libvolvoxai.so.{VERSION}"]:
        result = qualify(args.backend, library, args.output)
        print(json.dumps({"backend": args.backend, "library": library, "sessions": len(result["sessions"]), "output_exact": True}), flush=True)
        decode = qualify_decode(args.backend, library, args.output)
        if args.backend != "cpu":
            decode["sequence"] = qualify_decode_sequence(args.backend, library)
        print(json.dumps(decode), flush=True)
