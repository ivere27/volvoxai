"""Native required-row compact KV, head norm/RoPE and paging regression."""
from __future__ import annotations

import json
from contextlib import contextmanager, ExitStack
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import volvoxai as vx
from test_compact_gqa import attention_reference, rope_reference, tensor

B, S, HQ, LAYERS = 3, 8, 4, 2


def decoder(width, kv_heads):
    qwidth, kwidth = HQ * width, kv_heads * width
    nodes, weights, paged = [], {}, []

    def add(name, operator, inputs, shape, params=None):
        nodes.append({"id": name, "opType": operator, "inputs": inputs,
            "outputs": {"out": {"tensor": name, "shape": shape, "dtype": "float32"}},
            "params": params or {}})

    source = "x"
    for layer in range(LAYERS):
        for phase, port in enumerate(("q", "k", "v")):
            heads = HQ if port == "q" else kv_heads
            features, name = heads * width, f"{port}{layer}"
            index = np.arange(features * qwidth, dtype=np.float64).reshape(features, qwidth)
            weight = np.sin(index * .03 + phase + layer) * .01
            weight[np.arange(features), np.arange(features)] = .8
            weights[name + ".weight"] = weight.astype(np.float32)
            projected = name if port == "v" else name + "_projection"
            add(projected, "Linear", {"input": source, "weight": name + ".weight"}, [B, S, features])
            if port != "v":
                weights[name + ".norm"] = np.linspace(.7, 1.3, width, dtype=np.float32)
                add(name + "_heads", "Reshape", {"input": projected}, [B, S, heads, width],
                    {"shape": [B, S, heads, width]})
                add(name + "_normalized", "RMSNorm", {"input": name + "_heads", "weight": name + ".norm"},
                    [B, S, heads, width], {"eps": 1e-6})
                add(name + "_flat", "Reshape", {"input": name + "_normalized"}, [B, S, features],
                    {"shape": [B, S, features]})
                add(name, "RoPE", {"input": name + "_flat", "position_ids": "positions"},
                    [B, S, features], {"heads": heads, "rotary_dim": width, "theta": 1000000})
            if port != "q":
                paged.append(name)
        output = "out" if layer == LAYERS - 1 else f"attention{layer}"
        add(output, "CrossSDPA", {"q": f"q{layer}", "k": f"k{layer}", "v": f"v{layer}", "mask": "keep"},
            [B, S, qwidth], {"heads": HQ, "kv_heads": kv_heads, "causal": True})
        source = output
    inputs = {"x": {"shape": [B, S, qwidth], "dtype": "float32"},
              "keep": {"shape": [B, S], "dtype": "int32"},
              "positions": {"shape": [B, S], "dtype": "int32"},
              "active_lengths": {"shape": [B], "dtype": "int32"}}
    graph = {"format": "volvox-graph/v1", "dimensions": {}, "inputs": inputs,
        "nodes": nodes, "outputs": ["out"], "execution": {
            "incremental": {"active_lengths_input": "active_lengths", "sequence_axis": 1,
                "update_inputs": ["x", "keep", "positions"]},
            "invariant_inputs": ["active_lengths"], "entrypoints": [
                {"name": "evaluate", "effect": 1, "allowed_ranges": [1], "inputs": list(inputs), "outputs": ["out"]},
                {"name": "initialize", "effect": 2, "allowed_ranges": [1], "inputs": list(inputs), "outputs": ["out"]},
                {"name": "transition", "effect": 3, "allowed_ranges": [3], "inputs": list(inputs), "outputs": ["out"]}]}}
    return graph, weights, paged


def reference(values, weights, width, kv_heads):
    """NumPy-only full decoder; no runtime graph or row helper is used."""
    source, captures = values["x"], {}
    for layer in range(LAYERS):
        projected = {}
        for port in ("q", "k", "v"):
            name = f"{port}{layer}"
            result = (source.astype(np.float64) @ weights[name + ".weight"].astype(np.float64).T).astype(np.float32)
            if port != "v":
                heads = HQ if port == "q" else kv_heads
                groups = result.astype(np.float64).reshape(B, S, heads, width)
                groups /= np.sqrt(np.mean(groups ** 2, axis=-1, keepdims=True) + 1e-6)
                groups *= weights[name + ".norm"].astype(np.float64)
                result = rope_reference(groups.astype(np.float32).reshape(B, S, -1),
                    values["positions"], heads, width, 1000000, False)
            projected[port] = result
            captures[name] = result
        source = attention_reference({**projected, "keep": values["keep"]}, width, kv_heads, causal=True)
    captures["out"] = source
    return captures


def request(context_id, values, lengths, actions=None):
    pb = vx.pb
    bindings = [tensor(name, value) for name, value in values.items()]
    if actions is None:
        bindings.append(tensor("active_lengths", np.asarray(lengths, dtype=np.int32)))
        scope = pb.ExecutionScope(entrypoint="initialize", full=pb.Empty())
        reuse = []
    else:
        rows = [pb.SlotRow(empty=pb.Empty()) if action is None else
                pb.SlotRow(recompute=pb.Empty()) if action == "recompute" else
                pb.SlotRow(position=action) for action in actions]
        scope = pb.ExecutionScope(entrypoint="transition", slot_rows=pb.SlotRows(slots=rows))
        reuse = ["active_lengths"]
    return pb.ExecuteRequest(context_id=context_id, inputs=pb.InputBindings(values=bindings, reuse=reuse),
        scope=scope, outputs=pb.OutputOptions(storage=pb.OutputStorage.OUTPUT_STORAGE_INLINE))


def inline(packet):
    value = next(value for value in packet.outputs if value.name == "out")
    return np.frombuffer(value.inline, dtype=np.float32).copy().reshape(value.shape)


class CompactDecodeCacheTest(unittest.TestCase):
    def test_required_rows_paged_compact_kv_norm_rope_prefix_cow_and_retirement(self):
        pb = vx.pb
        with tempfile.TemporaryDirectory(prefix="volvox-compact-cache-") as temp, vx.open_library() as host:
            inference = vx.VxInferenceServiceClient(host)
            full = vx.VxPlatformServiceClient(host).get_platform_info(pb.Empty()).build_profile == pb.BuildProfile.BUILD_PROFILE_FULL
            debug = vx.VxDebugServiceClient(host) if full else None
            for width in (64, 128):
                for kv_heads in (2, 1):
                    with self.subTest(D=width, Hkv=kv_heads, profile="full" if full else "inference"):
                        graph, weights, paged = decoder(width, kv_heads)
                        directory = Path(temp) / f"{width}-{kv_heads}"
                        directory.mkdir(); (directory / "graph.json").write_text(json.dumps(graph))
                        save_file(weights, directory / "model.safetensors")
                        runtime = inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
                        model = inference.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                            graph_path=str(directory / "graph.json"), weight_paths=[str(directory / "model.safetensors")]))
                        compiled = inference.compile_model(pb.CompileModelRequest(model_id=model.model_id,
                            policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                                backends=["cpu"], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
                        create = lambda: inference.create_execution_context(pb.CreateExecutionContextRequest(
                            compiled_model_id=compiled.compiled_model_id,
                            graph=pb.GraphExecution(slots=B, row_mode=pb.DecodeRowMode.DECODE_ROW_MODE_REQUIRED, require_incremental=True),
                            limits=pb.ExecutionLimits(cache=pb.ContextCacheOptions(tensors=paged, page_tokens=2, max_pages=8, clear_on_recycle=True))))
                        context = create(); context_id = context.context_id
                        ref = pb.ExecutionContextRef(context_id=context_id)
                        qwidth, kwidth = HQ * width, kv_heads * width
                        lengths = [4, 2, 6]
                        values = {"x": np.sin(np.arange(B * S * qwidth, dtype=np.float64) * .71).astype(np.float32).reshape(B, S, qwidth),
                            "keep": np.array([[int(row < lengths[slot]) for row in range(S)] for slot in range(B)], dtype=np.int32),
                            "positions": np.array([[7 + slot * 19 + row for row in range(S)] for slot in range(B)], dtype=np.int32)}
                        state = lambda: inference.get_execution_context_info(ref)

                        def execute(actions=None):
                            expected = reference(values, weights, width, kv_heads)
                            operation = request(context_id, values, lengths, actions)
                            if debug is None:
                                return inference.execute(operation), expected
                            info = debug.create_debug_session(pb.CreateDebugSessionRequest(execute=operation))
                            dref = pb.DebugSessionRef(debug_session_id=info.debug_session_id)
                            try:
                                prior = debug.get_debug_decode_state(dref)
                                plan = debug.get_debug_plan(dref).plan
                                before = {}
                                for cache in prior.caches:
                                    if actions is not None:
                                        for slot in range(B):
                                            if lengths[slot]:
                                                observed = debug.read_debug_kv_cache(pb.ReadDebugKVCacheRequest(debug_session_id=info.debug_session_id,
                                                    cache_id=cache.cache_id, slot=slot, token_limit=S))
                                                before[cache.cache_id, slot] = np.frombuffer(observed.data, dtype=np.float32).copy()
                                info = debug.continue_debug_session(pb.ContinueDebugSessionRequest(debug_session_id=info.debug_session_id, expected_revision=info.revision))
                                self.assertEqual(info.state, pb.DebugState.DEBUG_STATE_COMPLETED, str(info.failure))
                                after = debug.get_debug_decode_state(dref)
                                self.assertEqual(len(after.caches), LAYERS * 2)
                                for cache in after.caches:
                                    name = plan.tensors[cache.tensor_id].name
                                    self.assertIn(name, paged)
                                    self.assertEqual(cache.page_tokens, 2)
                                    self.assertEqual(cache.token_shape, [kwidth])
                                    for slot in range(B):
                                        action = None if actions is None else actions[slot]
                                        end = lengths[slot] if actions is None or action is None or action == "recompute" else action + 1
                                        if not end:
                                            continue
                                        observed = debug.read_debug_kv_cache(pb.ReadDebugKVCacheRequest(debug_session_id=info.debug_session_id,
                                            cache_id=cache.cache_id, slot=slot, token_limit=S))
                                        self.assertEqual(observed.token_count, end)
                                        actual = np.frombuffer(observed.data, dtype=np.float32).copy().reshape(end, kwidth)
                                        np.testing.assert_allclose(actual, expected[name][slot, :end], atol=5e-5, rtol=5e-5)
                                        if actions is not None:
                                            preserved = lengths[slot] if action is None else lengths[slot] - 1 if action == "recompute" else action
                                            if preserved:
                                                np.testing.assert_array_equal(actual[:preserved].ravel(), before[cache.cache_id, slot][:preserved * kwidth])
                                return debug.take_debug_result(dref), expected
                            finally:
                                debug.release_debug_session(dref)

                        try:
                            packet, expected = execute()
                            initial = inline(packet); previous = initial.copy()
                            for slot in range(B):
                                np.testing.assert_allclose(initial[slot, :lengths[slot]], expected["out"][slot, :lengths[slot]], atol=5e-5, rtol=5e-5)
                            cache = state().cache
                            self.assertEqual(cache.pool_bytes, B * S * kwidth * 2 * LAYERS * 4)
                            self.assertEqual((cache.resident_pages, cache.free_pages), (6, 2))

                            def step(actions):
                                nonlocal previous
                                packet, expected = execute(actions)
                                actual = inline(packet)
                                for slot, action in enumerate(actions):
                                    row = lengths[slot] - 1 if action == "recompute" else action
                                    for position in range(S):
                                        if row is not None and row == position:
                                            np.testing.assert_allclose(actual[slot, position], expected["out"][slot, position], atol=5e-5, rtol=5e-5)
                                        else:
                                            np.testing.assert_array_equal(actual[slot, position], previous[slot, position])
                                    if row is not None:
                                        lengths[slot] = row + 1
                                previous = actual
                                info = state()
                                self.assertEqual(info.mode, pb.DecodeMode.DECODE_MODE_ROW)
                                self.assertEqual(info.active_lengths, lengths)
                                self.assertEqual(info.cache.pool_bytes, B * S * kwidth * 2 * LAYERS * 4)

                            values["keep"][0, 4] = values["keep"][1, 2] = 1
                            step([4, 2, None]); prior = state()
                            self.assertEqual(prior.cache.free_pages, 0)
                            with self.assertRaises(vx.VolvoxAIError):
                                inference.execute(request(context_id, values, lengths, [5, 3, 6]))
                            after = state()
                            self.assertEqual(after.active_lengths, prior.active_lengths)
                            self.assertEqual(after.cache.cache_generation, prior.cache.cache_generation)
                            self.assertEqual([slot.pages for slot in after.cache.slots], [slot.pages for slot in prior.cache.slots])
                            self.assertEqual(after.cache.reserved_pages, 0)
                            inference.release_context_slot(pb.ContextSlotRef(context_id=context_id, slot=2)); lengths[2] = 0
                            saved = {name: value[0, :4].copy() for name, value in values.items()}
                            saved_out = previous[0, :4].copy()
                            key = "compact-prefix/" + "한" * 180
                            cache = inference.publish_context_prefix(pb.PublishContextPrefixRequest(context_id=context_id, key=key, slot=0, tokens=4))
                            self.assertGreater(cache.prefix_snapshot_bytes, 0)
                            inference.release_context_slot(pb.ContextSlotRef(context_id=context_id, slot=1)); lengths[1] = 0
                            cache = inference.reuse_context_prefix(pb.ReuseContextPrefixRequest(context_id=context_id, key=key, slot=1)); lengths[1] = 4
                            for name, value in saved.items():
                                values[name][1, :4] = value
                            previous[1, :4] = saved_out
                            self.assertEqual(cache.slots[1].pages[:2], cache.slots[0].pages[:2])
                            values["x"][1, 3, 0] += .35; values["keep"][0, 5] = 1
                            step([5, "recompute", None]); cache = state().cache
                            self.assertEqual(cache.copy_on_writes, 1)
                            self.assertNotEqual(cache.slots[1].pages[1], cache.slots[0].pages[1])
                            self.assertNotEqual(cache.slots[1].pages[:2], [0, 1], "actual logical KV reads must follow the changed physical page order")
                            inference.release_context_slot(pb.ContextSlotRef(context_id=context_id, slot=1)); lengths[1] = 0
                            inference.reuse_context_prefix(pb.ReuseContextPrefixRequest(context_id=context_id, key=key, slot=1)); lengths[1] = 4
                            for name, value in saved.items():
                                values[name][1, :4] = value
                            previous[1, :4] = saved_out; values["keep"][1, 4] = 1
                            step([None, 4, None])
                            values["x"][2] = np.cos(np.arange(S * qwidth).reshape(S, qwidth) * .31).astype(np.float32) * .7
                            values["keep"][2].fill(0); values["keep"][2, 0] = 1
                            step([None, None, 0]); values["keep"][2, 1] = 1; step([None, None, 1])
                            np.testing.assert_array_equal(inline(packet), initial)
                            for slot in range(B):
                                inference.release_context_slot(pb.ContextSlotRef(context_id=context_id, slot=slot))
                            cache = inference.evict_context_prefixes(pb.EvictContextPrefixesRequest(context_id=context_id, free_pages=8))
                            self.assertEqual((cache.free_pages, cache.resident_pages, cache.prefix_snapshot_bytes), (8, 0, 0))
                            fresh = create()
                            try:
                                info = inference.get_execution_context_info(pb.ExecutionContextRef(context_id=fresh.context_id))
                                self.assertFalse(info.cache and info.cache.configured)
                            finally:
                                inference.release_execution_context(pb.ExecutionContextRef(context_id=fresh.context_id))
                        finally:
                            inference.release_execution_context(ref)
                            inference.release_compiled_model(pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                            inference.release_model(pb.ModelRef(model_id=model.model_id))
                            inference.release_runtime(pb.RuntimeRef(runtime_id=runtime.runtime_id))


def incremental(inputs, nodes, outputs, batch):
    inputs = {**inputs, "active_lengths": {"shape": [batch], "dtype": "int32"}}
    names = list(inputs)
    return {"format": "volvox-graph/v1", "dimensions": {}, "inputs": inputs,
        "nodes": nodes, "outputs": outputs, "execution": {
            "incremental": {"active_lengths_input": "active_lengths", "sequence_axis": 1,
                "update_inputs": [name for name in names if name != "active_lengths"]},
            "invariant_inputs": ["active_lengths"], "entrypoints": [
                {"name": "evaluate", "effect": 1, "allowed_ranges": [1],
                    "inputs": names, "outputs": outputs},
                {"name": "initialize", "effect": 2, "allowed_ranges": [1],
                    "inputs": names, "outputs": outputs},
                {"name": "transition", "effect": 3, "allowed_ranges": [3],
                    "inputs": names, "outputs": outputs}]}}


def node(name, operator, ports, shape, params=None, dtype="float32"):
    return {"id": name, "opType": operator, "inputs": ports, "params": params or {},
        "outputs": {"out": {"tensor": name, "shape": shape, "dtype": dtype}}}


def output(packet, name, dtype):
    value = next(value for value in packet.outputs if value.name == name)
    expected = vx.pb.DataType.DATA_TYPE_I32 if dtype == np.int32 else vx.pb.DataType.DATA_TYPE_F32
    if value.dtype != expected:
        raise AssertionError(f"{name}: wrong native output dtype {value.dtype}")
    return np.frombuffer(value.inline, dtype=dtype).copy().reshape(value.shape)


class DecodeProjectionRegressionTest(unittest.TestCase):
    @contextmanager
    def context(self, host, directory, graph, weights=None):
        pb = vx.pb
        inference = vx.VxInferenceServiceClient(host)
        directory.mkdir()
        (directory / "graph.json").write_text(json.dumps(graph))
        paths = []
        if weights:
            path = directory / "model.safetensors"
            save_file(weights, path)
            paths.append(str(path))
        with ExitStack() as cleanup:
            runtime = inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
            cleanup.callback(inference.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id))
            model = inference.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                graph_path=str(directory / "graph.json"), weight_paths=paths))
            cleanup.callback(inference.release_model, pb.ModelRef(model_id=model.model_id))
            compiled = inference.compile_model(pb.CompileModelRequest(model_id=model.model_id,
                policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                    backends=["cpu"], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
            cleanup.callback(inference.release_compiled_model,
                pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
            context = inference.create_execution_context(pb.CreateExecutionContextRequest(
                compiled_model_id=compiled.compiled_model_id,
                graph=pb.GraphExecution(slots=graph["inputs"]["active_lengths"]["shape"][0],
                    row_mode=pb.DecodeRowMode.DECODE_ROW_MODE_REQUIRED, require_incremental=True)))
            cleanup.callback(inference.release_execution_context, pb.ExecutionContextRef(context_id=context.context_id))
            yield inference, context.context_id

    def assert_row(self, inference, context_id, lengths):
        info = inference.get_execution_context_info(vx.pb.ExecutionContextRef(context_id=context_id))
        self.assertEqual(info.mode, vx.pb.DecodeMode.DECODE_MODE_ROW)
        self.assertEqual(info.active_lengths, lengths)

    def test_argmax_rank3_to_i32_rank2_keeps_batch_and_sequence_token_axes(self):
        with tempfile.TemporaryDirectory(prefix="volvox-token-axis-") as temp, vx.open_library() as host:
            for batch, sequence in ((1, 6), (1, 1), (3, 1), (3, 6)):
                with self.subTest(B=batch, S=sequence):
                    graph = incremental({"logits": {"shape": [batch, sequence, 7], "dtype": "float32"}},
                        [node("ids", "ArgMax", {"input": "logits"}, [batch, sequence],
                            {"axis": -1, "keepdims": False}, "int32")], ["ids"], batch)
                    with self.context(host, Path(temp) / f"{batch}-{sequence}", graph) as (inference, context_id):
                        values = {"logits": np.arange(batch * sequence * 7, dtype=np.float32).reshape(batch, sequence, 7)}
                        lengths = [1] * batch
                        initial = inference.execute(request(context_id, values, lengths))
                        saved = output(initial, "ids", np.int32)
                        self.assertEqual(saved.shape, (batch, sequence))
                        np.testing.assert_array_equal(saved[:, 0], np.full(batch, 6, dtype=np.int32))
                        previous = saved.copy()
                        actions = [1 if sequence > 1 else "recompute"] * batch
                        if batch > 1:
                            actions[1] = None
                        for iteration in range(2):
                            values["logits"].fill(-20 - iteration)
                            for slot in range(batch):
                                values["logits"][slot, :, (slot + iteration + 1) % 7] = 30 + slot
                            packet = inference.execute(request(context_id, values, lengths, actions))
                            actual = output(packet, "ids", np.int32)
                            self.assertEqual(actual.shape, (batch, sequence))
                            for slot, action in enumerate(actions):
                                row = lengths[slot] - 1 if action == "recompute" else action
                                for index in range(sequence):
                                    self.assertEqual(actual[slot, index],
                                        (slot + iteration + 1) % 7 if index == row else previous[slot, index])
                                if row is not None:
                                    lengths[slot] = row + 1
                            self.assert_row(inference, context_id, lengths)
                            previous = actual
                            actions = ["recompute" if action is not None else None for action in actions]
                        np.testing.assert_array_equal(output(initial, "ids", np.int32), saved)

    def test_singleton_sequence_ordered_head_norm_rope_and_i32_argmax_rows(self):
        heads, epsilon = 4, 1e-6
        with tempfile.TemporaryDirectory(prefix="volvox-single-head-") as temp, vx.open_library() as host:
            for batch in (1, 3):
                for width in (64, 128):
                    with self.subTest(B=batch, S=1, H=heads, D=width):
                        features = heads * width
                        graph = incremental({"x": {"shape": [batch, 1, features], "dtype": "float32"},
                            "positions": {"shape": [batch, 1], "dtype": "int32"}}, [
                            node("heads", "Reshape", {"input": "x"}, [batch, 1, heads, width],
                                {"shape": [batch, 1, heads, width]}),
                            node("norm", "RMSNorm", {"input": "heads", "weight": "norm.weight"},
                                [batch, 1, heads, width], {"eps": epsilon}),
                            node("flat", "Reshape", {"input": "norm"}, [batch, 1, features],
                                {"shape": [batch, 1, features]}),
                            node("rope", "RoPE", {"input": "flat", "position_ids": "positions"},
                                [batch, 1, features], {"heads": heads, "rotary_dim": width - 2,
                                    "theta": 1000000, "interleaved": False}),
                            node("ids", "ArgMax", {"input": "rope"}, [batch, 1],
                                {"axis": -1, "keepdims": False}, "int32")], ["norm", "rope", "ids"], batch)
                        weight = np.linspace(.65, 1.4, width, dtype=np.float32)
                        with self.context(host, Path(temp) / f"{batch}-{width}",
                                graph, {"norm.weight": weight}) as (inference, context_id):
                            indices = np.arange(batch * features).reshape(batch, 1, features)
                            values = {"x": (.4 * np.sin(indices * .071) + .1 * np.cos(indices * .19)).astype(np.float32),
                                "positions": (7 + 19 * np.arange(batch, dtype=np.int32)).reshape(batch, 1)}
                            lengths = [1] * batch

                            def oracle():
                                grouped = values["x"].astype(np.float64).reshape(batch, 1, heads, width)
                                norm = grouped / np.sqrt(np.mean(grouped * grouped, axis=-1, keepdims=True) + epsilon) * weight
                                norm = norm.astype(np.float32)
                                rope = rope_reference(norm.reshape(batch, 1, features), values["positions"],
                                    heads, width - 2, 1000000, False)
                                return {"norm": norm, "rope": rope, "ids": np.argmax(rope, axis=-1).astype(np.int32)}

                            initial = inference.execute(request(context_id, values, lengths))
                            expected = oracle()
                            previous = {}
                            for name, data in expected.items():
                                actual = output(initial, name, data.dtype.type)
                                self.assertEqual(actual.shape, data.shape)
                                np.testing.assert_allclose(actual, data, atol=2e-5, rtol=2e-5)
                                previous[name] = actual
                            original = {name: data.copy() for name, data in previous.items()}
                            actions = ["recompute"] * batch
                            if batch > 1:
                                actions[1] = None
                            for iteration in range(2):
                                values["x"] += np.asarray(.12 + iteration * .05, dtype=np.float32)
                                values["positions"] += 3
                                packet = inference.execute(request(context_id, values, lengths, actions))
                                expected = oracle()
                                for name, data in expected.items():
                                    actual = output(packet, name, data.dtype.type)
                                    self.assertEqual(actual.shape, data.shape)
                                    for slot, action in enumerate(actions):
                                        if action is None:
                                            np.testing.assert_array_equal(actual[slot], previous[name][slot])
                                        elif name == "ids":
                                            np.testing.assert_array_equal(actual[slot], data[slot])
                                        else:
                                            np.testing.assert_allclose(actual[slot], data[slot], atol=2e-5, rtol=2e-5)
                                    previous[name] = actual
                                self.assert_row(inference, context_id, lengths)
                            for name, data in original.items():
                                np.testing.assert_array_equal(output(initial, name, data.dtype.type), data)


if __name__ == "__main__":
    unittest.main()
