"""Physical CUDA required-row decoder regression through public dispatch.

NumPy and the repository SDK are the only Python dependencies. Select a
physical CUDA library explicitly with VOLVOXAI_LIBRARY and opt in with
VOLVOXAI_TEST_NATIVE_GPU_BACKEND=cuda. Contiguous and paged methods qualify
separately; neither establishes prefix sharing/COW or performance support.
VOLVOXAI_CUDA_RECEIPT_DIR preserves exclusive per-method observations.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest

import numpy as np

import test_cuda_compact_gqa as _f32
from test_cuda_compact_gqa import (ATOL, RTOL, PB, attention_oracle, array, node, tensor, vx)

B, S, HQ, LAYERS = 3, 8, 4, 2
OBSERVED = ["out"] + [f"{p}{l}{suffix}" for l in range(LAYERS)
    for p, suffix in (("q", "_normalized"), ("k", "_normalized"), ("q", ""))]


def decoder(width, kv_heads):
    """Two layers deliberately alternate RoPE theta and partial rotary tails."""
    qwidth = HQ * width
    nodes, weights, caches = [], {}, []
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
            nodes.append(node(projected, "Linear", {"input": source, "weight": name + ".weight"}, [B, S, features]))
            if port != "v":
                weights[name + ".norm"] = np.linspace(.7, 1.3, width, dtype=np.float32)
                nodes.append(node(name + "_heads", "Reshape", {"input": projected}, [B, S, heads, width],
                    {"shape": [B, S, heads, width]}))
                nodes.append(node(name + "_normalized", "RMSNorm", {"input": name + "_heads", "weight": name + ".norm"},
                    [B, S, heads, width], {"eps": 1e-6}))
                nodes.append(node(name + "_flat", "Reshape", {"input": name + "_normalized"}, [B, S, features],
                    {"shape": [B, S, features]}))
                theta = 1000000 if (port == "q") == (layer == 0) else 100000
                rotary = width if port == "q" else width - 2
                nodes.append(node(name, "RoPE", {"input": name + "_flat", "position_ids": "positions"},
                    [B, S, features], {"heads": heads, "rotary_dim": rotary, "theta": theta,
                        "interleaved": bool(layer)}))
            if port != "q":
                caches.append(name)
        attention = f"attention{layer}"
        nodes.append(node(attention, "CrossSDPA", {"q": f"q{layer}", "k": f"k{layer}", "v": f"v{layer}", "mask": "keep"},
            [B, S, qwidth], {"heads": HQ, "kv_heads": kv_heads, "causal": True}))
        summed = f"residual_sum{layer}"
        output = "out" if layer == LAYERS - 1 else f"residual{layer}"
        nodes.append(node(summed, "Add", {"a": attention, "b": source}, [B, S, qwidth]))
        nodes.append(node(output, "Add", {"a": summed, "b": "residual"}, [B, S, qwidth]))
        source = output
    inputs = {"x": {"shape": [B, S, qwidth], "dtype": "float32"},
        "keep": {"shape": [B, S], "dtype": "int32"},
        "positions": {"shape": [B, S], "dtype": "int32"},
        "residual": {"shape": [B, S, qwidth], "dtype": "float32"},
        "active_lengths": {"shape": [B], "dtype": "int32"}}
    graph = {"format": "volvox-graph/v1", "dimensions": {}, "inputs": inputs, "nodes": nodes,
        "outputs": OBSERVED, "execution": {
            "incremental": {"active_lengths_input": "active_lengths", "sequence_axis": 1,
                "update_inputs": ["x", "keep", "positions", "residual"]},
            "invariant_inputs": ["active_lengths"], "entrypoints": [
                {"name": "evaluate", "effect": 1, "allowed_ranges": [1], "inputs": list(inputs), "outputs": OBSERVED},
                {"name": "initialize", "effect": 2, "allowed_ranges": [1], "inputs": list(inputs), "outputs": OBSERVED},
                {"name": "transition", "effect": 3, "allowed_ranges": [3], "inputs": list(inputs), "outputs": OBSERVED}]}}
    return graph, weights, caches


def rotate(x, positions, heads, rotary, theta, interleaved):
    """Independent float64 pair rotations, preserving the unrotated tail."""
    width = x.shape[-1] // heads
    batch, sequence = x.shape[:2]
    source = x.astype(np.float64).reshape(batch, sequence, heads, width)
    result = source.copy()
    angle = positions[:, :, None, None] * theta ** (-np.arange(rotary // 2) * 2.0 / rotary)
    left = np.arange(rotary // 2) * (2 if interleaved else 1)
    right = left + (1 if interleaved else rotary // 2)
    a, b = source[..., left], source[..., right]
    result[..., left] = a * np.cos(angle) - b * np.sin(angle)
    result[..., right] = a * np.sin(angle) + b * np.cos(angle)
    return result.astype(np.float32).reshape(x.shape)


def reference(values, weights, width, kv_heads):
    """Full NumPy decoder with compact/direct versus repeated-KV oracle."""
    source, captures = values["x"], {}
    batch, sequence = source.shape[:2]
    for layer in range(LAYERS):
        projected = {}
        for port in ("q", "k", "v"):
            name = f"{port}{layer}"
            result = (source.astype(np.float64) @ weights[name + ".weight"].astype(np.float64).T).astype(np.float32)
            if port != "v":
                heads = HQ if port == "q" else kv_heads
                groups = result.astype(np.float64).reshape(batch, sequence, heads, width)
                groups /= np.sqrt(np.mean(groups ** 2, axis=-1, keepdims=True) + 1e-6)
                groups *= weights[name + ".norm"].astype(np.float64)
                normalized = groups.astype(np.float32)
                captures[name + "_normalized"] = normalized
                theta = 1000000 if (port == "q") == (layer == 0) else 100000
                result = rotate(normalized.reshape(batch, sequence, -1), values["positions"], heads,
                    width if port == "q" else width - 2, theta, bool(layer))
            projected[port] = result
            captures[name] = result
        attended = attention_oracle({**projected, "keep": values["keep"]}, width, kv_heads, True)
        # Two explicit F32 sums match the declared Add/Add chain; CUDA's
        # ordinary optimization may fuse this into one Add3 invocation.
        source = (attended + source).astype(np.float32)
        source = (source + values["residual"]).astype(np.float32)
    captures["out"] = source
    return captures


def generation_decoder(width, kv_heads):
    graph, weights, caches = decoder(width, kv_heads)
    graph = copy.deepcopy(graph)
    for descriptor in graph["inputs"].values():
        descriptor["shape"][0] = 1
    for work in graph["nodes"]:
        work["outputs"]["out"]["shape"][0] = 1
        if "shape" in work["params"]:
            work["params"]["shape"][0] = 1
        for port, source in work["inputs"].items():
            if source == "x":
                work["inputs"][port] = "embedding"
    vocabulary, channels = 13, HQ * width
    index = np.arange(vocabulary * channels, dtype=np.float64).reshape(vocabulary, channels)
    weights["embedding.weight"] = (.6 * np.sin(index * .073) + .2 * np.cos(index * .13)).astype(np.float32)
    weights["logits.weight"] = (.11 * np.cos(index * .031 + .7)).astype(np.float32)
    graph["inputs"].pop("x")
    graph["inputs"]["token_ids"] = {"shape": [1, S], "dtype": "int32"}
    graph["nodes"].insert(0, node("embedding", "Embedding", {"input": "token_ids", "weight": "embedding.weight"}, [1, S, channels]))
    graph["nodes"].append(node("logits", "Linear", {"input": "out", "weight": "logits.weight"}, [1, S, vocabulary]))
    graph["nodes"].append(node("generated", "ArgMax", {"input": "logits"}, [1, S],
        {"axis": -1, "keepdims": False}, "int32"))
    graph["outputs"] = ["logits", "generated"]
    execution = graph["execution"]
    execution["incremental"]["update_inputs"] = ["token_ids", "keep", "positions", "residual"]
    execution["incremental"]["position_input"] = "positions"
    for entry in execution["entrypoints"]:
        entry["inputs"] = list(graph["inputs"])
        entry["outputs"] = graph["outputs"]
    execution["generation"] = {"initialize_entrypoint": "initialize", "transition_entrypoint": "transition",
        "token_input": "token_ids", "token_output": "generated", "keep_input": "keep",
        "logits_output": "logits", "logits_axis": 2, "id_domain_size": vocabulary,
        "output_name": "generated_ids", "sequence_axis": 1}
    return graph, weights, caches


def greedy_reference(values, weights, width, kv_heads, steps):
    """Caller feedback in NumPy; no engine-produced IDs enter the reference."""
    current = {name: value.copy() for name, value in values.items()}
    length = int(current.pop("active_lengths")[0])
    selected = []
    for _ in range(steps):
        x = weights["embedding.weight"][current["token_ids"]]
        output = reference({"x": x, "keep": current["keep"], "positions": current["positions"],
            "residual": current["residual"]}, weights, width, kv_heads)["out"]
        logits = output[0, length - 1].astype(np.float64) @ weights["logits.weight"].astype(np.float64).T
        chosen = int(logits.argmax())
        selected.append(chosen)
        current["token_ids"][0, length] = chosen
        current["keep"][0, length] = 1
        length += 1
    return np.array([selected], dtype=np.int32)


def dynamic_generation_decoder(width, kv_heads, batch=1, generation=False):
    """Reuse the two-layer decoder with a genuine minimum-one sequence symbol."""
    graph, weights, caches = generation_decoder(width, kv_heads)
    graph["dimensions"] = {"S": {"min": 1, "max": S}}
    for name, descriptor in graph["inputs"].items():
        descriptor["shape"][0] = batch
        if name != "active_lengths":
            descriptor["shape"][1] = "S"
    for work in graph["nodes"]:
        work["outputs"]["out"]["shape"][0:2] = [batch, "S"]
        if "shape" in work["params"]:
            work["params"]["shape"][0:2] = [batch, "S"]
    if not generation:
        graph["execution"].pop("generation")
        graph["outputs"] = ["logits", "generated"] + OBSERVED
        for entry in graph["execution"]["entrypoints"]:
            entry["outputs"] = graph["outputs"]
    return graph, weights, caches


def dynamic_token_values(batch, sequence, width):
    """Each prefix stays byte-identical when the concrete shape grows."""
    slots = np.arange(batch)[:, None]
    rows = np.arange(sequence)[None, :]
    channels = np.arange(HQ * width)[None, None, :]
    return {"token_ids": ((2 + slots * 3 + rows * 5) % 13).astype(np.int32),
        "keep": np.ones((batch, sequence), dtype=np.int32),
        "positions": (13 + slots * 19 + rows * 3).astype(np.int32),
        "residual": (.02 * np.cos(((slots * S + rows)[:, :, None] * HQ * width + channels) * .11)).astype(np.float32)}


def token_reference(values, weights, width, kv_heads):
    captures = reference({"x": weights["embedding.weight"][values["token_ids"]],
        "keep": values["keep"], "positions": values["positions"], "residual": values["residual"]},
        weights, width, kv_heads)
    logits = (captures["out"].astype(np.float64) @ weights["logits.weight"].astype(np.float64).T).astype(np.float32)
    return {**captures, "logits": logits, "generated": logits.argmax(-1).astype(np.int32)}


def request(context_id, values, lengths, actions=None):
    bindings = [tensor(name, value) for name, value in values.items()]
    if actions is None:
        bindings.append(tensor("active_lengths", np.asarray(lengths, dtype=np.int32)))
        scope = PB.ExecutionScope(entrypoint="initialize", full=PB.Empty())
        reuse = []
    else:
        rows = [PB.SlotRow(empty=PB.Empty()) if action is None else
            PB.SlotRow(recompute=PB.Empty()) if action == "recompute" else
            PB.SlotRow(position=action) for action in actions]
        scope = PB.ExecutionScope(entrypoint="transition", slot_rows=PB.SlotRows(slots=rows))
        reuse = ["active_lengths"]
    return PB.ExecuteRequest(context_id=context_id, inputs=PB.InputBindings(values=bindings, reuse=reuse),
        scope=scope, outputs=PB.OutputOptions(storage=PB.OutputStorage.OUTPUT_STORAGE_INLINE))


@unittest.skipUnless(os.environ.get("VOLVOXAI_TEST_NATIVE_GPU_BACKEND") == "cuda",
    "set VOLVOXAI_TEST_NATIVE_GPU_BACKEND=cuda on physical CUDA hardware")
class CudaCompactDecodeTest(unittest.TestCase):
    route = _f32.CudaCompactGQATest.route
    model = _f32.CudaCompactGQATest.model

    def setUp(self):
        _f32.CudaCompactGQATest.setUp(self)
        self.variants = []

    def tearDown(self):
        destination = os.environ.get("VOLVOXAI_CUDA_RECEIPT_DIR")
        if destination:
            directory = Path(destination)
            directory.mkdir(parents=True, exist_ok=True)
            library = Path(os.environ["VOLVOXAI_LIBRARY"]).resolve(strict=True)
            receipt = {"format": "volvox-cuda-compact-decode-fixture/v1", "test": self.id(),
                "profile": int(self.profile), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "oracle_helper_sha256": hashlib.sha256(Path(__file__).with_name("test_cuda_compact_gqa.py").read_bytes()).hexdigest(),
                "artifact": {"path": str(library), "sha256": hashlib.sha256(library.read_bytes()).hexdigest()},
                "tolerances": {"atol": ATOL, "rtol": RTOL}, "executions": self.executions,
                "refusals": self.refusals, "route_records": self.records, "variants": self.variants,
                "timing": False, "prefix_sharing_qualified": False,
                "status": "observations only; unittest log determines pass/fail"}
            with (directory / (self._testMethodName + ".json")).open("x") as stream:
                json.dump(receipt, stream, indent=2)
                stream.write("\n")

    @contextmanager
    def context(self, document, weights, caches, paged):
        with self.model(document, weights) as (model_id, cleanup):
            compiled = self.inference.compile_model(PB.CompileModelRequest(model_id=model_id,
                policy=PB.BackendPolicy(mode=PB.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                    backends=["cuda"], operator_fallback=PB.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
            self.route(compiled.report, "compile")
            cleanup.callback(self.inference.release_compiled_model, PB.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
            context = self.inference.create_execution_context(PB.CreateExecutionContextRequest(
                compiled_model_id=compiled.compiled_model_id,
                graph=PB.GraphExecution(slots=B, row_mode=PB.DecodeRowMode.DECODE_ROW_MODE_REQUIRED, require_incremental=True),
                limits=PB.ExecutionLimits(cache=PB.ContextCacheOptions(tensors=caches, page_tokens=2,
                    max_pages=8 if paged else 12, clear_on_recycle=True,
                    policy=PB.DecodeCachePolicy.DECODE_CACHE_POLICY_PAGED if paged else PB.DecodeCachePolicy.DECODE_CACHE_POLICY_CONTIGUOUS))))
            cleanup.callback(self.inference.release_execution_context, PB.ExecutionContextRef(context_id=context.context_id))
            yield context.context_id

    def run_matrix(self, paged):
        self.variants = []
        debug = vx.VxDebugServiceClient(self.host) if self.profile == PB.BuildProfile.BUILD_PROFILE_FULL else None
        for width in (64, 128):
            for kv_heads in (2, 1):
                with self.subTest(D=width, Hkv=kv_heads, paged=paged):
                    graph, weights, caches = decoder(width, kv_heads)
                    with self.context(graph, weights, caches, paged) as context:
                        ref = PB.ExecutionContextRef(context_id=context)
                        lengths = [4, 2, 6]
                        values = {"x": np.sin(np.arange(B * S * HQ * width, dtype=np.float64) * .71).astype(np.float32).reshape(B, S, -1),
                            "residual": (.02 * np.cos(np.arange(B * S * HQ * width, dtype=np.float64) * .11)).astype(np.float32).reshape(B, S, -1),
                            "keep": np.array([[int(row < lengths[slot]) for row in range(S)] for slot in range(B)], dtype=np.int32),
                            "positions": np.array([[7 + slot * 19 + row * 3 for row in range(S)] for slot in range(B)], dtype=np.int32)}
                        previous, retained, before_kv = {}, None, {}
                        kv_reads, row_steps = 0, 0

                        def execute(actions=None):
                            nonlocal before_kv, kv_reads
                            expected = reference(values, weights, width, kv_heads)
                            operation = request(context, values, lengths, actions)
                            if debug is None:
                                packet = self.inference.execute(operation)
                            else:
                                session = debug.create_debug_session(PB.CreateDebugSessionRequest(execute=operation))
                                dref = PB.DebugSessionRef(debug_session_id=session.debug_session_id)
                                try:
                                    plan = debug.get_debug_plan(dref).plan
                                    for layer in range(LAYERS):
                                        pair = {f"residual_sum{layer}", "out" if layer == LAYERS - 1 else f"residual{layer}"}
                                        self.assertTrue(any(work.fused and pair.issubset(set(work.source_node_ids))
                                            for work in plan.steps), "default CUDA Add3 optimization must actually be present")
                                    session = debug.continue_debug_session(PB.ContinueDebugSessionRequest(
                                        debug_session_id=session.debug_session_id, expected_revision=session.revision))
                                    self.assertEqual(session.state, PB.DebugState.DEBUG_STATE_COMPLETED, str(session.failure))
                                    state = debug.get_debug_decode_state(dref)
                                    self.assertEqual(len(state.caches), len(caches))
                                    next_kv = {}
                                    for cache in state.caches:
                                        name = plan.tensors[cache.tensor_id].name
                                        self.assertIn(name, caches)
                                        self.assertEqual(cache.token_shape, [kv_heads * width])
                                        self.assertEqual(cache.page_tokens, 2)
                                        for slot in range(B):
                                            action = None if actions is None else actions[slot]
                                            end = lengths[slot] if actions is None or action is None or action == "recompute" else action + 1
                                            if not end:
                                                continue
                                            data = debug.read_debug_kv_cache(PB.ReadDebugKVCacheRequest(
                                                debug_session_id=session.debug_session_id, cache_id=cache.cache_id, slot=slot, token_limit=S))
                                            self.assertEqual(data.token_count, end)
                                            actual = np.frombuffer(data.data, dtype=np.float32).copy().reshape(end, kv_heads * width)
                                            np.testing.assert_allclose(actual, expected[name][slot, :end], atol=ATOL, rtol=RTOL)
                                            key = name, slot
                                            preserved = lengths[slot] if action is None else lengths[slot] - 1 if action == "recompute" else action
                                            if actions is not None and preserved:
                                                np.testing.assert_array_equal(actual[:preserved], before_kv[key][:preserved])
                                            next_kv[key] = actual
                                            kv_reads += 1
                                    before_kv = next_kv
                                    packet = debug.take_debug_result(dref)
                                finally:
                                    debug.release_debug_session(dref)
                            self.route(packet.report, "initialize" if actions is None else "required_row")
                            self.executions += 1
                            actuals = {name: array(packet, name) for name in OBSERVED}
                            return packet, actuals, expected

                        packet, initial, expected = execute()
                        retained = packet
                        for name in OBSERVED:
                            for slot in range(B):
                                np.testing.assert_allclose(initial[name][slot, :lengths[slot]], expected[name][slot, :lengths[slot]], atol=ATOL, rtol=RTOL)
                        previous = {name: value.copy() for name, value in initial.items()}
                        info = self.inference.get_execution_context_info(ref)
                        self.assertEqual(info.cache.pool_bytes, B * S * kv_heads * width * 2 * LAYERS * 4)

                        def step(actions):
                            nonlocal previous, row_steps
                            _, actuals, expected = execute(actions)
                            for name in OBSERVED:
                                for slot, action in enumerate(actions):
                                    row = lengths[slot] - 1 if action == "recompute" else action
                                    for position in range(S):
                                        if row is not None and position == row:
                                            np.testing.assert_allclose(actuals[name][slot, position], expected[name][slot, position], atol=ATOL, rtol=RTOL)
                                        else:
                                            np.testing.assert_array_equal(actuals[name][slot, position], previous[name][slot, position])
                            for slot, action in enumerate(actions):
                                if action is not None and action != "recompute":
                                    lengths[slot] = action + 1
                            previous = actuals
                            info = self.inference.get_execution_context_info(ref)
                            self.assertEqual(info.mode, PB.DecodeMode.DECODE_MODE_ROW)
                            self.assertEqual(info.active_lengths, lengths)
                            row_steps += 1

                        values["keep"][0, 4] = values["keep"][1, 2] = 1
                        step([4, 2, None])
                        saved_state = self.inference.get_execution_context_info(ref)
                        bad = values["positions"].copy(); bad[2, -1] = -1
                        invalid = {**values, "positions": bad}
                        with self.assertRaises(vx.VolvoxAIError) as failure:
                            self.inference.execute(request(context, invalid, lengths, [None, "recompute", None]))
                        self.assertEqual(failure.exception.status, PB.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT)
                        self.assertEqual(failure.exception.code, PB.OperationCode.OPERATION_CODE_INVALID_INPUT_VALUES)
                        self.assertFalse(failure.exception.response.outputs)
                        self.refusals += 1
                        after = self.inference.get_execution_context_info(ref)
                        self.assertEqual(after.active_lengths, saved_state.active_lengths)
                        self.assertEqual(after.cache.cache_generation, saved_state.cache.cache_generation)
                        for operation in (
                            lambda: self.inference.publish_context_prefix(PB.PublishContextPrefixRequest(context_id=context, key="unqualified-cuda-prefix", slot=0, tokens=4)),
                            lambda: self.inference.reuse_context_prefix(PB.ReuseContextPrefixRequest(context_id=context, key="unqualified-cuda-prefix", slot=1))):
                            with self.assertRaises(vx.VolvoxAIError) as failure:
                                operation()
                            self.assertEqual(failure.exception.status, PB.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED)
                            self.refusals += 1
                        with self.assertRaises(vx.VolvoxAIError) as failure:
                            self.inference.execute(request(context, values, lengths, [None, None, None]))
                        self.assertEqual(failure.exception.status, PB.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED)
                        self.assertEqual(failure.exception.code, PB.OperationCode.OPERATION_CODE_DECODE_ROW_DOMAIN_UNSUPPORTED)
                        self.assertFalse(failure.exception.response.outputs)
                        self.refusals += 1
                        after = self.inference.get_execution_context_info(ref)
                        self.assertEqual(after.active_lengths, saved_state.active_lengths)
                        self.assertEqual(after.cache.cache_generation, saved_state.cache.cache_generation)
                        self.inference.release_context_slot(PB.ContextSlotRef(context_id=context, slot=2)); lengths[2] = 0
                        before_kv = {key: value for key, value in before_kv.items() if key[1] != 2}
                        values["x"][1, 2, 0] += .35
                        values["keep"][0, 5] = 1
                        step([5, "recompute", None])
                        values["x"][2] = np.cos(np.arange(S * HQ * width).reshape(S, -1) * .31).astype(np.float32) * .7
                        values["keep"][2].fill(0); values["keep"][2, 0] = 1
                        step([None, None, 0])
                        values["keep"][2, 1] = 1
                        step([None, None, 1])
                        values["x"][0, 5, 0] -= .2
                        step(["recompute", None, None])
                        for name in OBSERVED:
                            np.testing.assert_array_equal(array(retained, name), initial[name])
                        info = self.inference.get_execution_context_info(ref)
                        if paged:
                            self.assertTrue(any(list(slot.pages) != list(range(index * (S // 2), index * (S // 2) + len(slot.pages)))
                                for index, slot in enumerate(info.cache.slots)), "actual paged KV must use noncontiguous physical order")
                        self.variants.append({"D": width, "Hkv": kv_heads, "B": B, "layers": LAYERS,
                            "policy": "paged" if paged else "contiguous", "required_row_steps": row_steps,
                            "private_kv_reads": kv_reads, "pool_bytes": int(info.cache.pool_bytes),
                            "pages": [list(slot.pages) for slot in info.cache.slots], "inactive_and_prior_outputs_preserved": True})
                    # Context and model teardown must not invalidate retained inline packets.
                    for name in OBSERVED:
                        np.testing.assert_array_equal(array(retained, name), initial[name])

    def test_contiguous_required_rows_compact_kv_headnorm_rope_and_retirement(self):
        self.run_matrix(False)

    def test_paged_required_rows_compact_kv_headnorm_rope_and_retirement(self):
        self.run_matrix(True)

    def generate_matrix(self, paged):
        for width, kv_heads in ((64, 2), (128, 1)):
            graph, weights, caches = generation_decoder(width, kv_heads)
            values = {"token_ids": np.array([[2, 5, 0, 0, 0, 0, 0, 0]], dtype=np.int32),
                "residual": (.02 * np.cos(np.arange(S * HQ * width, dtype=np.float64) * .11)).astype(np.float32).reshape(1, S, -1),
                "keep": np.array([[1, 1, 0, 0, 0, 0, 0, 0]], dtype=np.int32),
                "positions": np.array([[13 + row * 3 for row in range(S)]], dtype=np.int32),
                "active_lengths": np.array([2], dtype=np.int32)}
            expected = greedy_reference(values, weights, width, kv_heads, 4)
            all_ids = []
            for chunks in ((4,), (1, 2, 1)):
                with self.subTest(D=width, Hkv=kv_heads, paged=paged, chunks=chunks), self.model(graph, weights) as (model_id, cleanup):
                    compiled = self.inference.compile_model(PB.CompileModelRequest(model_id=model_id,
                        policy=PB.BackendPolicy(mode=PB.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                            backends=["cuda"], operator_fallback=PB.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
                    self.route(compiled.report, "generate_compile")
                    cleanup.callback(self.inference.release_compiled_model, PB.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                    context = self.inference.create_execution_context(PB.CreateExecutionContextRequest(
                        compiled_model_id=compiled.compiled_model_id,
                        autoregressive=PB.AutoregressiveExecution(sampling=PB.SamplingMode.SAMPLING_MODE_GREEDY, max_steps=4),
                        limits=PB.ExecutionLimits(cache=PB.ContextCacheOptions(tensors=caches, page_tokens=2, max_pages=4,
                            clear_on_recycle=True, policy=PB.DecodeCachePolicy.DECODE_CACHE_POLICY_PAGED if paged else PB.DecodeCachePolicy.DECODE_CACHE_POLICY_CONTIGUOUS))))
                    ref = PB.ExecutionContextRef(context_id=context.context_id)
                    cleanup.callback(self.inference.release_execution_context, ref)
                    initial = PB.GenerationInputs(values=[tensor(name, value) for name, value in values.items()])
                    with self.assertRaises(vx.VolvoxAIError) as failure:
                        self.inference.generate(PB.GenerateRequest(context_id=context.context_id, initial_inputs=initial,
                            budget=PB.GenerationBudget(max_steps=0)))
                    self.assertEqual(failure.exception.status, PB.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT)
                    self.assertFalse(failure.exception.response.outputs)
                    self.refusals += 1
                    chosen, retained = [], None
                    completed = 0
                    for index, count in enumerate(chunks):
                        packet = self.inference.generate(PB.GenerateRequest(context_id=context.context_id,
                            initial_inputs=initial if index == 0 else None, budget=PB.GenerationBudget(max_steps=count)))
                        self.route(packet.report, "generate_chunk")
                        self.executions += 1
                        output = packet.outputs[0]
                        self.assertEqual(output.name, "generated_ids")
                        self.assertEqual(output.dtype, PB.DataType.DATA_TYPE_I32)
                        self.assertEqual(output.shape, [1, count])
                        self.assertFalse(output.HasField("buffer"))
                        ids = np.frombuffer(output.inline, dtype=np.int32).copy().reshape(1, count)
                        np.testing.assert_array_equal(ids, expected[:, completed:completed + count])
                        completed += count
                        self.assertEqual(packet.progress.steps, count)
                        self.assertEqual(packet.progress.total_steps, completed)
                        if retained:
                            np.testing.assert_array_equal(np.frombuffer(retained[0].outputs[0].inline, dtype=np.int32), retained[1])
                        else:
                            retained = packet, ids.ravel().copy()
                        chosen.append(ids)
                    selected = np.concatenate(chosen, axis=1)
                    np.testing.assert_array_equal(selected, expected)
                    state = self.inference.get_execution_context_info(ref)
                    self.assertEqual(state.mode, PB.DecodeMode.DECODE_MODE_ROW)
                    self.assertEqual(state.total_steps, 4)
                    self.assertEqual(state.active_lengths, [5])
                    self.assertEqual(state.cache.pool_bytes, S * kv_heads * width * 2 * LAYERS * 4)
                    self.assertEqual(state.state, PB.ExecutionContextState.EXECUTION_CONTEXT_STATE_COMPLETED)
                    with self.assertRaises(vx.VolvoxAIError) as failure:
                        self.inference.generate(PB.GenerateRequest(context_id=context.context_id, budget=PB.GenerationBudget(max_steps=1)))
                    self.assertFalse(failure.exception.response.outputs)
                    self.refusals += 1
                    self.variants.append({"D": width, "Hkv": kv_heads, "B": 1, "layers": LAYERS,
                        "policy": "paged" if paged else "contiguous", "chunks": list(chunks),
                        "ids": selected.tolist(), "exact_independent_numpy_ids": True,
                        "mode": int(state.mode), "total_steps": int(state.total_steps)})
                    all_ids.append(selected)
                np.testing.assert_array_equal(np.frombuffer(retained[0].outputs[0].inline, dtype=np.int32), retained[1])
            np.testing.assert_array_equal(all_ids[0], all_ids[1])

    def test_contiguous_generate_device_feedback_chunks_and_exact_numpy_ids(self):
        self.generate_matrix(False)

    def test_paged_generate_device_feedback_chunks_and_exact_numpy_ids(self):
        self.generate_matrix(True)

    def test_dynamic_minimum_one_context_prefill_and_required_row(self):
        """The min-S=1 context accepts a prompt-one, fixed-capacity ROW stream."""
        if self.profile != PB.BuildProfile.BUILD_PROFILE_FULL:
            self.skipTest("dynamic minimum-one regression is explicitly FULL-only")
        debug = vx.VxDebugServiceClient(self.host)
        for batch in (1, 3):
            for width, kv_heads in ((64, 2), (128, 1)):
                for paged in (False, True):
                    with self.subTest(B=batch, D=width, Hkv=kv_heads, paged=paged):
                        graph, weights, caches = dynamic_generation_decoder(width, kv_heads, batch)
                        observed = graph["outputs"]
                        with self.model(graph, weights) as (model_id, cleanup):
                            compiled = self.inference.compile_model(PB.CompileModelRequest(model_id=model_id,
                                policy=PB.BackendPolicy(mode=PB.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                                    backends=["cuda"], operator_fallback=PB.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
                            self.route(compiled.report, "minimum_one_compile")
                            cleanup.callback(self.inference.release_compiled_model, PB.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                            context = self.inference.create_execution_context(PB.CreateExecutionContextRequest(
                                compiled_model_id=compiled.compiled_model_id,
                                graph=PB.GraphExecution(slots=batch, row_mode=PB.DecodeRowMode.DECODE_ROW_MODE_REQUIRED, require_incremental=True),
                                limits=PB.ExecutionLimits(cache=PB.ContextCacheOptions(tensors=caches, page_tokens=2,
                                    max_pages=batch, clear_on_recycle=True,
                                    policy=PB.DecodeCachePolicy.DECODE_CACHE_POLICY_PAGED if paged else PB.DecodeCachePolicy.DECODE_CACHE_POLICY_CONTIGUOUS))))
                            ref = PB.ExecutionContextRef(context_id=context.context_id)
                            cleanup.callback(self.inference.release_execution_context, ref)
                            created = self.inference.get_execution_context_info(ref)
                            self.assertEqual(created.mode, PB.DecodeMode.DECODE_MODE_ROW)
                            self.assertFalse(created.initialized)
                            # Context capability probes its declared minimum
                            # before bindings. Cached replay then keeps one
                            # concrete capacity, just like the real LLM caller.
                            first_values = dynamic_token_values(batch, 2, width)
                            first_values["keep"][:, 1] = 0
                            previous_kv = {}
                            private_reads = 0

                            def execute(values, actions=None):
                                nonlocal previous_kv, private_reads
                                expected = token_reference(values, weights, width, kv_heads)
                                operation = request(context.context_id, values, [1] * batch, actions)
                                session = debug.create_debug_session(PB.CreateDebugSessionRequest(execute=operation))
                                dref = PB.DebugSessionRef(debug_session_id=session.debug_session_id)
                                try:
                                    plan = debug.get_debug_plan(dref).plan
                                    session = debug.continue_debug_session(PB.ContinueDebugSessionRequest(
                                        debug_session_id=session.debug_session_id, expected_revision=session.revision))
                                    self.assertEqual(session.state, PB.DebugState.DEBUG_STATE_COMPLETED, str(session.failure))
                                    state = debug.get_debug_decode_state(dref)
                                    self.assertEqual(len(state.caches), len(caches))
                                    current_kv = {}
                                    for cache in state.caches:
                                        name = plan.tensors[cache.tensor_id].name
                                        self.assertIn(name, caches)
                                        self.assertEqual(cache.token_shape, [kv_heads * width])
                                        for slot in range(batch):
                                            limit = 2 if actions is not None and slot == 0 else 1
                                            data = debug.read_debug_kv_cache(PB.ReadDebugKVCacheRequest(
                                                debug_session_id=session.debug_session_id, cache_id=cache.cache_id, slot=slot, token_limit=S))
                                            self.assertEqual(data.token_count, limit)
                                            actual = np.frombuffer(data.data, dtype=np.float32).copy().reshape(limit, kv_heads * width)
                                            np.testing.assert_allclose(actual, expected[name][slot, :limit], atol=ATOL, rtol=RTOL)
                                            if actions is not None:
                                                np.testing.assert_array_equal(actual[:1], previous_kv[name, slot])
                                            current_kv[name, slot] = actual
                                            private_reads += 1
                                    previous_kv = current_kv
                                    packet = debug.take_debug_result(dref)
                                finally:
                                    debug.release_debug_session(dref)
                                self.route(packet.report, "minimum_one_prefill" if actions is None else "minimum_one_required_row")
                                self.executions += 1
                                actuals = {name: array(packet, name) for name in observed if name != "generated"}
                                ids = next(t for t in packet.outputs if t.name == "generated")
                                self.assertEqual(ids.dtype, PB.DataType.DATA_TYPE_I32)
                                self.assertEqual(ids.shape, [batch, values["token_ids"].shape[1]])
                                self.assertFalse(ids.HasField("buffer"))
                                actuals["generated"] = np.frombuffer(ids.inline, dtype=np.int32).copy().reshape(ids.shape)
                                return packet, actuals, expected

                            retained, first, expected = execute(first_values)
                            for name in observed:
                                if name == "generated":
                                    np.testing.assert_array_equal(first[name], expected[name])
                                else:
                                    np.testing.assert_allclose(first[name], expected[name], atol=ATOL, rtol=RTOL)
                            continued_values = {name: value.copy() for name, value in first_values.items()}
                            continued_values["keep"][0, 1] = 1
                            continued_values["token_ids"][0, 1] = 11
                            continued_values["positions"][0, 1] += 4
                            continued_values["residual"][0, 1, 0] += .03
                            _, continued, expected = execute(continued_values, [1] + [None] * (batch - 1))
                            for name in observed:
                                self.assertEqual(continued[name].shape[:2], (batch, 2))
                                np.testing.assert_array_equal(continued[name][:, :1], first[name][:, :1])
                                if name == "generated":
                                    np.testing.assert_array_equal(continued[name][0, 1], expected[name][0, 1])
                                else:
                                    np.testing.assert_allclose(continued[name][0, 1], expected[name][0, 1], atol=ATOL, rtol=RTOL)
                                if batch > 1:
                                    np.testing.assert_array_equal(continued[name][1:], first[name][1:])
                            info = self.inference.get_execution_context_info(ref)
                            self.assertEqual(info.mode, PB.DecodeMode.DECODE_MODE_ROW)
                            self.assertEqual(info.active_lengths, [2] + [1] * (batch - 1))
                            self.assertEqual(info.cache.pool_bytes, batch * 2 * kv_heads * width * 2 * LAYERS * 4)
                            self.variants.append({"D": width, "Hkv": kv_heads, "B": batch, "layers": LAYERS,
                                "policy": "paged" if paged else "contiguous", "dynamic_sequence": {"minimum": 1, "maximum": S},
                                "actual_sequence_shapes": [2, 2], "prompt_length": 1, "private_kv_reads": private_reads,
                                "required_row_steps": 1, "prior_rows_byte_exact": True,
                                "inactive_rows_byte_exact": batch > 1, "cached_shape_growth_qualified": False})
                        for name in observed:
                            if name == "generated":
                                data = next(t for t in retained.outputs if t.name == name)
                                np.testing.assert_array_equal(np.frombuffer(data.inline, dtype=np.int32).reshape(data.shape), first[name])
                            else:
                                np.testing.assert_array_equal(array(retained, name), first[name])

    def test_dynamic_minimum_one_context_generate_prompt_one_with_reserved_capacity(self):
        """Generate has fixed conditioning capacity, while context starts at min S=1."""
        if self.profile != PB.BuildProfile.BUILD_PROFILE_FULL:
            self.skipTest("dynamic minimum-one regression is explicitly FULL-only")
        for width, kv_heads in ((64, 2), (128, 1)):
            for paged in (False, True):
                with self.subTest(D=width, Hkv=kv_heads, paged=paged):
                    graph, weights, caches = dynamic_generation_decoder(width, kv_heads, generation=True)
                    values = dynamic_token_values(1, 2, width)
                    values["keep"][0, 1] = 0
                    values["active_lengths"] = np.array([1], dtype=np.int32)
                    # The existing caller oracle writes the last chosen token
                    # into spare storage. Its capacity is independent of the
                    # actual two-token input capacity submitted to Generate.
                    oracle_values = dynamic_token_values(1, 3, width)
                    oracle_values["keep"][0, 1:] = 0
                    oracle_values["active_lengths"] = np.array([1], dtype=np.int32)
                    expected = greedy_reference(oracle_values, weights, width, kv_heads, 2)
                    with self.model(graph, weights) as (model_id, cleanup):
                        compiled = self.inference.compile_model(PB.CompileModelRequest(model_id=model_id,
                            policy=PB.BackendPolicy(mode=PB.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                                backends=["cuda"], operator_fallback=PB.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
                        self.route(compiled.report, "minimum_one_generate_compile")
                        cleanup.callback(self.inference.release_compiled_model, PB.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                        context = self.inference.create_execution_context(PB.CreateExecutionContextRequest(
                            compiled_model_id=compiled.compiled_model_id,
                            autoregressive=PB.AutoregressiveExecution(sampling=PB.SamplingMode.SAMPLING_MODE_GREEDY, max_steps=2),
                            limits=PB.ExecutionLimits(cache=PB.ContextCacheOptions(tensors=caches, page_tokens=2,
                                max_pages=1, clear_on_recycle=True,
                                policy=PB.DecodeCachePolicy.DECODE_CACHE_POLICY_PAGED if paged else PB.DecodeCachePolicy.DECODE_CACHE_POLICY_CONTIGUOUS))))
                        ref = PB.ExecutionContextRef(context_id=context.context_id)
                        cleanup.callback(self.inference.release_execution_context, ref)
                        created = self.inference.get_execution_context_info(ref)
                        self.assertEqual(created.mode, PB.DecodeMode.DECODE_MODE_ROW)
                        self.assertFalse(created.initialized)
                        packets, chunks = [], []
                        for step in range(2):
                            packet = self.inference.generate(PB.GenerateRequest(context_id=context.context_id,
                                initial_inputs=PB.GenerationInputs(values=[tensor(name, value) for name, value in values.items()]) if step == 0 else None,
                                budget=PB.GenerationBudget(max_steps=1)))
                            self.route(packet.report, "minimum_one_generate_prefill" if step == 0 else "minimum_one_generate_row")
                            self.executions += 1
                            self.assertEqual(len(packet.outputs), 1)
                            output = packet.outputs[0]
                            self.assertEqual(output.name, "generated_ids")
                            self.assertEqual(output.dtype, PB.DataType.DATA_TYPE_I32)
                            self.assertEqual(output.shape, [1, 1])
                            self.assertFalse(output.HasField("buffer"))
                            ids = np.frombuffer(output.inline, dtype=np.int32).copy().reshape(1, 1)
                            np.testing.assert_array_equal(ids, expected[:, step:step + 1])
                            self.assertEqual(packet.progress.steps, 1)
                            self.assertEqual(packet.progress.total_steps, step + 1)
                            packets.append(packet)
                            chunks.append(ids)
                        np.testing.assert_array_equal(np.concatenate(chunks, axis=1), expected)
                        info = self.inference.get_execution_context_info(ref)
                        self.assertEqual(info.mode, PB.DecodeMode.DECODE_MODE_ROW)
                        self.assertEqual(info.active_lengths, [2])
                        self.assertEqual(info.total_steps, 2)
                        self.assertEqual(info.state, PB.ExecutionContextState.EXECUTION_CONTEXT_STATE_COMPLETED)
                        self.variants.append({"D": width, "Hkv": kv_heads, "B": 1, "layers": LAYERS,
                            "policy": "paged" if paged else "contiguous", "dynamic_sequence": {"minimum": 1, "maximum": S},
                            "prompt_length": 1, "conditioning_capacity": 2, "chunks": [1, 1],
                            "exact_independent_numpy_ids": True, "ids": expected.tolist(),
                            "actual_sequence_shape_growth": False})
                    for step, packet in enumerate(packets):
                        np.testing.assert_array_equal(np.frombuffer(packet.outputs[0].inline, dtype=np.int32).reshape(1, 1), expected[:, step:step + 1])


if __name__ == "__main__":
    unittest.main()
