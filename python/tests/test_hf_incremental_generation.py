"""Modern cached rows and C-owned generation against independent HF prefixes.

The reference always reevaluates the exact active prefix. The tested engine
uses a fixed-capacity graph and retained rows across calls and contexts.
"""
from __future__ import annotations

from contextlib import ExitStack
import base64
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))
import volvoxai as vx
from tools.exporter.hf_extended import ExtendedDecoderConfig, build_extended_decoder_graph
from tools.llm_baseline.generate_modern_incremental_stress import (FAMILIES, tiny_config, tiny_deepseek_config,
    tiny_qwen4, qwen4_reference, incremental_fixture as fixture, reference_rows, initial_values)

try:
    import torch
    import transformers
    HAVE_HF = True
except ImportError:
    HAVE_HF = False

CAPACITY = 16


def tensor(name, value):
    return vx.pb.Tensor(name=name, shape=list(value.shape), inline=value.tobytes(),
        dtype=vx.pb.DataType.DATA_TYPE_I32 if value.dtype == np.int32 else vx.pb.DataType.DATA_TYPE_F32)


def retain_fixture(family, graph, weights, values, scores, ids):
    destination = os.environ.get("VOLVOXAI_INCREMENTAL_DUMP")
    if not destination:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = json.loads(path.read_text()) if path.exists() else {"format":"volvox-modern-incremental/v1", "cases":[]}
    name = family + "-prompt" + str(int(values["active_lengths"][0]))
    document["cases"] = [case for case in document["cases"] if case["name"] != name]
    document["cases"].append({"name":name, "graph":graph,
        "weight_shards":[base64.b64encode(save(weights, metadata={"format":"pt"})).decode("ascii")],
        "inputs":[{"name":key, "shape":list(value.shape), "dtype":str(value.dtype),
            "data":base64.b64encode(value.tobytes()).decode("ascii")} for key,value in values.items()],
        "expected_logits":[value.tolist() for value in scores], "expected_ids":ids, "budgets":[1,1,2]})
    path.write_text(json.dumps(document, allow_nan=False) + "\n")


@unittest.skipUnless(HAVE_HF, "independent oracle requires torch and transformers")
class ModernIncrementalTests(unittest.TestCase):
    def test_shared_reference_factory_keeps_capacity_and_seed_explicit(self):
        # The longer-corpus CLI used to patch only the extended config factory,
        # leaving DeepSeek's rotary domain at sixteen positions.
        with ExitStack() as resources:
            reference, config, graph, weights = fixture("deepseek_v3", resources,
                capacity=128, seed=3801)
            self.assertEqual((reference.config.max_position_embeddings, config.max_context), (128, 128))
            self.assertEqual(graph["dimensions"]["S"]["max"], 128)
            prompt = [2 + row % 20 for row in range(63)]
            inputs = initial_values(graph, config, prompt)
            self.assertEqual(inputs["token_ids"].shape, (1, 128))
            self.assertTrue(np.all(inputs["token_ids"][0, 63:] == 7))
            scores, ids = reference_rows(reference, prompt, 2, inputs)
            self.assertEqual(len(ids), 2)
            self.assertTrue(all(np.isfinite(value).all() for value in scores))
            repeated, _, _, repeated_weights = fixture("deepseek_v3", resources,
                capacity=128, seed=3801)
            repeated_scores, repeated_ids = reference_rows(repeated, prompt, 2, inputs)
            self.assertEqual(ids, repeated_ids)
            for name in weights:
                np.testing.assert_array_equal(weights[name], repeated_weights[name])
            for expected, actual in zip(scores, repeated_scores):
                np.testing.assert_array_equal(expected, actual)
            _, small_config, small_graph, changed = fixture("deepseek_v3", resources,
                capacity=16, seed=3802)
            self.assertEqual(small_config.max_context, 16)
            self.assertEqual(initial_values(small_graph, small_config, [2,3,4])["token_ids"].shape, (1, 16))
            self.assertTrue(any(not np.array_equal(weights[name], changed[name]) for name in weights))

    def test_first_chunk_control_overflow_is_retryable(self):
        pb = vx.pb
        with ExitStack() as resources:
            reference, config, graph, weights = fixture("dense_controls", resources)
            graph["inputs"]["rotary_control"] = {"shape":["S"], "dtype":"int32"}
            for entrypoint in graph["execution"]["entrypoints"]:
                entrypoint["inputs"].append("rotary_control")
            incremental = graph["execution"]["incremental"]
            incremental["position_input"] = "rotary_control"
            incremental["update_inputs"].remove("position_ids")
            incremental["update_inputs"].append("rotary_control")
            graph["execution"]["invariant_inputs"].append("position_ids")
            graph["execution"]["generation"]["incremental_controls"]["position_input"] = "rotary_control"
            values = initial_values(graph, config, [2,3,4])
            values["rotary_control"] = np.full(CAPACITY, np.iinfo(np.int32).max - 1, np.int32)
            _, expected = reference_rows(reference, [2,3,4], 3)
            for profile in ("libvolvoxai.so", "libvolvoxai-lite.so"):
                with self.subTest(profile=profile), ExitStack() as owners:
                    host = owners.enter_context(vx.open_library(ROOT / "native" / profile))
                    client = vx.VxInferenceServiceClient(host)
                    runtime = client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
                    owners.callback(client.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id))
                    model = client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                        package=pb.ModelPackage(graph_document=json.dumps(graph).encode(),
                            weight_shards=[save(weights, metadata={"format":"pt"})])))
                    owners.callback(client.release_model, pb.ModelRef(model_id=model.model_id))
                    compiled = client.compile_model(pb.CompileModelRequest(model_id=model.model_id,
                        policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                            backends=["cpu"], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
                    owners.callback(client.release_compiled_model, pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                    context = client.create_execution_context(pb.CreateExecutionContextRequest(
                        compiled_model_id=compiled.compiled_model_id,
                        autoregressive=pb.AutoregressiveExecution(max_steps=3,
                            sampling=pb.SamplingMode.SAMPLING_MODE_GREEDY)))
                    owners.callback(client.release_execution_context, pb.ExecutionContextRef(context_id=context.context_id))
                    def generate():
                        return client.generate(pb.GenerateRequest(context_id=context.context_id,
                            initial_inputs=pb.GenerationInputs(values=[tensor(name,value) for name,value in values.items()]),
                            budget=pb.GenerationBudget(max_steps=3)))
                    with self.assertRaises(vx.VolvoxAIError):
                        generate()
                    values["rotary_control"] = np.arange(CAPACITY, dtype=np.int32)
                    packet = generate()
                    self.assertEqual(np.frombuffer(packet.outputs[0].inline, np.int32).tolist(), expected)
                    values["rotary_control"].fill(np.iinfo(np.int32).max - 1)

    def test_generation_metadata_roundtrip_and_malformed_controls(self):
        from tools.exporter.hf_extended import build_extended_decoder_graph
        pb = vx.pb
        with ExitStack() as resources:
            planning_host = resources.enter_context(vx.open_library(ROOT / "native/libvolvoxai.so"))
            planning = vx.VxPlanningServiceClient(planning_host)
            reference, config, cached, weights = fixture("qwen3_next", resources)
            source = {name:value.detach().numpy() for name,value in reference.state_dict().items()}
            prefix, prefix_weights = build_extended_decoder_graph(config, source, generate=True)
            for profile in ("libvolvoxai.so", "libvolvoxai-lite.so"):
                with self.subTest(profile=profile), ExitStack() as owner:
                    host = owner.enter_context(vx.open_library(ROOT / "native" / profile))
                    client = vx.VxInferenceServiceClient(host)
                    runtime = client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
                    owner.callback(client.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id))
                    def load(graph, tensors=weights):
                        return client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                            package=pb.ModelPackage(graph_document=json.dumps(graph).encode(),
                                weight_shards=[save(tensors, metadata={"format":"pt"})])))
                    for graph, tensors, field in ((cached,weights,"incremental_controls"),
                            (prefix,prefix_weights,"full_prefix")):
                        model = load(graph, tensors)
                        owner.callback(client.release_model, pb.ModelRef(model_id=model.model_id))
                        info = client.get_model_info(pb.ModelRef(model_id=model.model_id))
                        controls = getattr(info.execution.generation, field)
                        self.assertEqual(controls.position_input, "position_ids")
                        document = json.loads(planning.serialize_graph(pb.GraphDefinition(execution=info.execution)).data)
                        self.assertEqual(document["execution"]["generation"][field]["position_input"], "position_ids")
                        controls.chunk_size = 2
                        malformed = json.loads(planning.serialize_graph(pb.GraphDefinition(execution=info.execution)).data)
                        self.assertEqual(malformed["execution"]["generation"][field]["chunk_size"], 2)
                        complete = copy.deepcopy(graph)
                        complete["execution"] = malformed["execution"]
                        with self.assertRaises(vx.VolvoxAIError):
                            load(complete, tensors)
                    for name in ("token_input", "keep_input"):
                        for value in ("", None, 5):
                            malformed = copy.deepcopy(cached)
                            malformed["execution"]["generation"][name] = value
                            with self.subTest(field=name, value=value), self.assertRaises(vx.VolvoxAIError):
                                load(malformed)
                        malformed = copy.deepcopy(cached)
                        del malformed["execution"]["generation"][name]
                        with self.subTest(field=name, value="missing"), self.assertRaises(vx.VolvoxAIError):
                            load(malformed)

    def test_graph_only_rejects_past_edits_without_mutating_cache(self):
        pb = vx.pb
        for family in ("qwen3_next", "deepseek_v4", "dense_controls"):
            with self.subTest(family=family), ExitStack() as resources:
                reference, config, graph, weights = fixture(family, resources)
                graph = copy.deepcopy(graph)
                generated_name = graph["execution"]["generation"]["token_output"]
                if family != "dense_controls":
                    del graph["execution"]["generation"]
                prompt = [2,3,4]
                values = initial_values(graph, config, prompt)
                expected, ids = reference_rows(reference, prompt, 2)
                host = resources.enter_context(vx.open_library(ROOT / "native/libvolvoxai-lite.so"))
                client = vx.VxInferenceServiceClient(host)
                runtime = client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
                resources.callback(client.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id))
                model = client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                    package=pb.ModelPackage(graph_document=json.dumps(graph).encode(),
                        weight_shards=[save(weights, metadata={"format":"pt"})])))
                resources.callback(client.release_model, pb.ModelRef(model_id=model.model_id))
                compiled = client.compile_model(pb.CompileModelRequest(model_id=model.model_id,
                    policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                        backends=["cpu"], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
                resources.callback(client.release_compiled_model, pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                context = client.create_execution_context(pb.CreateExecutionContextRequest(
                    compiled_model_id=compiled.compiled_model_id,
                    graph=pb.GraphExecution(slots=1, row_mode=pb.DecodeRowMode.DECODE_ROW_MODE_AUTO,
                        require_incremental=True)))
                resources.callback(client.release_execution_context, pb.ExecutionContextRef(context_id=context.context_id))
                def execute(bindings, initialize=False):
                    return client.execute(pb.ExecuteRequest(context_id=context.context_id,
                        inputs=pb.InputBindings(values=[tensor(name,value) for name,value in bindings.items()
                            if initialize or name != "active_lengths"], reuse=[] if initialize else ["active_lengths"]),
                        scope=pb.ExecutionScope(entrypoint="initialize", full=pb.Empty()) if initialize else
                            pb.ExecutionScope(entrypoint="transition", slot_rows=pb.SlotRows(slots=[pb.SlotRow(position=3)])),
                        outputs=pb.OutputOptions(selection=pb.TensorOutputSelection(names=["logits"]),
                            storage=pb.OutputStorage.OUTPUT_STORAGE_INLINE)))
                if family == "deepseek_v4":
                    shifted = {name:value.copy() for name,value in values.items()}
                    shifted["position_ids"] += 4
                    with self.assertRaises(vx.VolvoxAIError):
                        execute(shifted, initialize=True)
                client.execute(pb.ExecuteRequest(context_id=context.context_id,
                    inputs=pb.InputBindings(values=[tensor(name,value) for name,value in values.items()]),
                    scope=pb.ExecutionScope(entrypoint="evaluate", full=pb.Empty()),
                    outputs=pb.OutputOptions(selection=pb.TensorOutputSelection(names=[generated_name]),
                        storage=pb.OutputStorage.OUTPUT_STORAGE_INLINE)))
                with self.assertRaises(vx.VolvoxAIError) as refused:
                    client.execute(pb.ExecuteRequest(context_id=context.context_id,
                        inputs=pb.InputBindings(values=[tensor(name,value) for name,value in values.items() if name != "token_ids"],
                            feedback=[pb.TensorFeedback(input_name="token_ids", output_name=generated_name)]),
                        scope=pb.ExecutionScope(entrypoint="initialize", full=pb.Empty())))
                self.assertEqual(refused.exception.status, pb.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED)
                first = execute(values, initialize=True)
                with self.assertRaises(vx.VolvoxAIError):
                    client.execute(pb.ExecuteRequest(context_id=context.context_id,
                        inputs=pb.InputBindings(values=[tensor(name,value) for name,value in values.items()]),
                        scope=pb.ExecutionScope(entrypoint="evaluate", full=pb.Empty())))
                next_values = {name:value.copy() for name,value in values.items()}
                next_values["token_ids"][0,3] = ids[0]
                next_values["attention_mask"][0,3] = 1
                changed = [tensor(name,value) for name,value in next_values.items() if name != "active_lengths"]
                with self.assertRaises(vx.VolvoxAIError) as refused:
                    client.execute(pb.ExecuteRequest(context_id=context.context_id,
                        inputs=pb.InputBindings(values=changed, reuse=["active_lengths"]),
                        scope=pb.ExecutionScope(entrypoint="transition",
                            dependencies=pb.DependencyClosure(roots=["token_ids", "attention_mask", "position_ids"]))))
                self.assertEqual(refused.exception.status, pb.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED)
                with self.assertRaises(vx.VolvoxAIError) as refused:
                    client.execute(pb.ExecuteRequest(context_id=context.context_id,
                        inputs=pb.InputBindings(values=[tensor(name,value) for name,value in next_values.items()
                            if name not in ("token_ids", "active_lengths")], reuse=["active_lengths"],
                            feedback=[pb.TensorFeedback(input_name="token_ids", output_name=generated_name)]),
                        scope=pb.ExecutionScope(entrypoint="transition", slot_rows=pb.SlotRows(slots=[pb.SlotRow(position=3)]))))
                self.assertEqual(refused.exception.status, pb.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED)
                for name in ("token_ids", "position_ids"):
                    corrupted = {key:value.copy() for key,value in next_values.items()}
                    corrupted[name].reshape(-1)[0] += 1
                    with self.assertRaises(vx.VolvoxAIError):
                        execute(corrupted)
                resumed = execute(next_values)
                actual = np.frombuffer(resumed.outputs[0].inline, np.float32).reshape(resumed.outputs[0].shape)
                initial = np.frombuffer(first.outputs[0].inline, np.float32).reshape(first.outputs[0].shape)
                np.testing.assert_allclose(actual[0,3], expected[1], atol=2e-4, rtol=2e-4)
                np.testing.assert_array_equal(actual[0,:3], initial[0,:3])
                context = client.create_execution_context(pb.CreateExecutionContextRequest(
                    compiled_model_id=compiled.compiled_model_id,
                    graph=pb.GraphExecution(slots=1, row_mode=pb.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,
                        require_incremental=True)))
                resources.callback(client.release_execution_context, pb.ExecutionContextRef(context_id=context.context_id))
                reinitialized = execute(values, initialize=True)
                reset = np.frombuffer(reinitialized.outputs[0].inline, np.float32).reshape(reinitialized.outputs[0].shape)
                np.testing.assert_allclose(reset[0,2], expected[0], atol=2e-4, rtol=2e-4)

    def test_modern_rows_generation_continuation_and_context_isolation(self):
        pb = vx.pb
        selected = os.environ.get("VOLVOXAI_INCREMENTAL_FAMILIES")
        families = selected.split(",") if selected else FAMILIES
        backend = os.environ.get("VOLVOXAI_INCREMENTAL_BACKEND", "cpu")
        profiles = os.environ.get("VOLVOXAI_INCREMENTAL_PROFILES", "libvolvoxai.so,libvolvoxai-lite.so").split(",")
        for family in families:
            with self.subTest(family=family), ExitStack() as fixture_resources:
                reference, config, graph, weights = fixture(family, fixture_resources)
                self.assertIn("incremental", graph["execution"])
                self.assertNotIn("full_prefix", graph["execution"]["generation"])
                with tempfile.TemporaryDirectory(prefix="volvox-modern-cache-") as temporary:
                    directory = Path(temporary)
                    (directory / "graph.json").write_text(json.dumps(graph))
                    save_file(weights, str(directory / "model.safetensors"))
                    for prompt in ([2,3,4], [2,3,4,5,6]):
                        values = initial_values(graph, config, prompt)
                        scores, expected = reference_rows(reference, prompt, 4, values)
                        retain_fixture(family, graph, weights, values, scores, expected)
                        for profile in profiles:
                            with self.subTest(profile=profile, prompt_length=len(prompt)), ExitStack() as resources:
                                host = resources.enter_context(vx.open_library(ROOT / "native" / profile))
                                client = vx.VxInferenceServiceClient(host)
                                runtime = client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
                                resources.callback(client.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id))
                                model = client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                                    graph_path=str(directory / "graph.json"), weight_paths=[str(directory / "model.safetensors")]))
                                resources.callback(client.release_model, pb.ModelRef(model_id=model.model_id))
                                info = client.get_model_info(pb.ModelRef(model_id=model.model_id))
                                self.assertEqual(info.execution.generation.incremental_controls.position_input, "position_ids")
                                compiled = client.compile_model(pb.CompileModelRequest(model_id=model.model_id,
                                    policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                                        backends=[backend], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
                                resources.callback(client.release_compiled_model, pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                                graph_context = client.create_execution_context(pb.CreateExecutionContextRequest(
                                    compiled_model_id=compiled.compiled_model_id,
                                    graph=pb.GraphExecution(slots=1, row_mode=pb.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,
                                        require_incremental=True)))
                                resources.callback(client.release_execution_context, pb.ExecutionContextRef(context_id=graph_context.context_id))
                                bindings = {name:value.copy() for name,value in values.items()}
                                invariants = graph["execution"]["invariant_inputs"]
                                previous_logits = None
                                for step in range(4):
                                    row = len(prompt) - 1 + step
                                    if step:
                                        bindings["token_ids"][0, row] = expected[step - 1]
                                        bindings["attention_mask"][0, row] = 1
                                        for name in ("image_feature_ids", "image_token_mask"):
                                            if name in bindings:
                                                bindings[name].reshape(-1)[row] = 0
                                    scope = (pb.ExecutionScope(entrypoint="initialize", full=pb.Empty()) if step == 0 else
                                        pb.ExecutionScope(entrypoint="transition", slot_rows=pb.SlotRows(slots=[pb.SlotRow(position=row)])))
                                    packet = client.execute(pb.ExecuteRequest(context_id=graph_context.context_id, scope=scope,
                                        inputs=pb.InputBindings(values=[tensor(name,value) for name,value in bindings.items()
                                            if name not in invariants or step == 0], reuse=invariants if step else []),
                                        outputs=pb.OutputOptions(selection=pb.TensorOutputSelection(names=["logits"]),
                                            storage=pb.OutputStorage.OUTPUT_STORAGE_INLINE)))
                                    output = next(value for value in packet.outputs if value.name == "logits")
                                    logits = np.frombuffer(output.inline, np.float32).reshape(output.shape).copy()
                                    np.testing.assert_allclose(logits[0, row], scores[step], atol=2e-4, rtol=2e-4,
                                        err_msg=f"{family}/{profile}/row{row}")
                                    if previous_logits is not None:
                                        np.testing.assert_array_equal(logits[0, :row], previous_logits[0, :row],
                                            err_msg="transition recomputed or changed a retained row")
                                    previous_logits = logits
                                # Two independent owners alternate calls to catch shared recurrent state.
                                contexts = []
                                for _ in range(2):
                                    context = client.create_execution_context(pb.CreateExecutionContextRequest(
                                        compiled_model_id=compiled.compiled_model_id,
                                        autoregressive=pb.AutoregressiveExecution(sampling=pb.SamplingMode.SAMPLING_MODE_GREEDY, max_steps=4)))
                                    resources.callback(client.release_execution_context, pb.ExecutionContextRef(context_id=context.context_id))
                                    contexts.append(context.context_id)
                                observed = [[], []]
                                for chunk, count in enumerate((1,1,2)):
                                    for owner, context_id in enumerate(contexts):
                                        packet = client.generate(pb.GenerateRequest(context_id=context_id,
                                            initial_inputs=pb.GenerationInputs(values=[tensor(name,value) for name,value in values.items()]) if chunk == 0 else None,
                                            budget=pb.GenerationBudget(max_steps=count)))
                                        self.assertEqual(packet.report.backend, backend)
                                        self.assertFalse(packet.report.fallback and packet.report.fallback.operator_fallback_used)
                                        output = next(value for value in packet.outputs if value.name == "generated_ids")
                                        observed[owner].extend(np.frombuffer(output.inline, np.int32).tolist())
                                self.assertEqual(observed, [expected, expected])


def dump_wasm_case(name,graph,weights,inputs,expected):
    """Optionally retain this exact HF oracle for another public host binding."""
    destination=os.environ.get("VOLVOXAI_HF_GENERATION_DUMP")
    if not destination:return
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(path.read_text()) if path.exists() else {"format":"volvox-hf-generation-fixtures/v1","cases":[]}
    case={"name":name,"graph":graph,"weight_shards":[base64.b64encode(save(weights,metadata={"format":"pt"})).decode("ascii")],
        "inputs":[{"name":key,"shape":list(value.shape),"dtype":"int32",
            "data":base64.b64encode(value.tobytes()).decode("ascii")} for key,value in inputs.items()],
        "expected_ids":expected,"budgets":[1,2]}
    manifest["cases"]=[value for value in manifest["cases"] if value["name"]!=name]+[case]
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(manifest,allow_nan=False)+"\n");temporary.replace(path)


@unittest.skipUnless(HAVE_HF, "independent oracle needs torch and transformers")
class ExtendedGenerateReferenceTests(unittest.TestCase):
    def compare(self, family, config_cls, model_cls, *, raw=None, validator=None, builder=None, seed=3801):
        import torch
        torch.manual_seed(seed); torch.set_num_threads(1)
        hf = config_cls(**(raw or tiny_config(family))); hf._attn_implementation = "eager"
        reference = model_cls(hf).float().eval()
        config = (validator or ExtendedDecoderConfig.from_hf)(hf.to_dict(), max_context=16)
        source = {name:value.detach().numpy() for name,value in reference.state_dict().items()}
        graph, weights = (builder(config,source) if builder else build_extended_decoder_graph(config, source, generate=True))
        prefix = [2,3,4]
        expected = []
        with torch.inference_mode():
            for _ in range(3):
                output = reference(input_ids=torch.tensor([prefix]),
                    attention_mask=torch.ones(1,len(prefix),dtype=torch.long),
                    position_ids=torch.arange(len(prefix))[None],use_cache=False).logits
                token = int(output[0,-1].argmax())
                expected.append(token); prefix.append(token)
        inputs = {"token_ids":np.array([[2,3,4]],np.int32),"attention_mask":np.ones((1,3),np.int32),
            "position_ids":np.arange(3,dtype=np.int32)}
        if "sequence_ids" in graph["inputs"]:inputs["sequence_ids"] = np.arange(3,dtype=np.int32)
        if "chunk_attention_mask" in graph["inputs"]:
            positions=np.arange(3)//config.chunk_size
            inputs["chunk_attention_mask"] = (positions[:,None]==positions[None,:])[None].astype(np.int32)
        dump_wasm_case(family,graph,weights,inputs,expected)
        pb = vx.pb
        initial = pb.GenerationInputs(values=[pb.Tensor(name=name, shape=list(value.shape), inline=value.tobytes(),
            dtype=pb.DataType.DATA_TYPE_I32) for name,value in inputs.items()])
        with tempfile.TemporaryDirectory(prefix="volvox-hf-generate-") as temporary:
            directory=Path(temporary)
            (directory/"graph.json").write_text(json.dumps(graph))
            save_file(weights,str(directory/"model.safetensors"))
            for profile in ("libvolvoxai.so","libvolvoxai-lite.so"):
                with self.subTest(family=family,profile=profile),ExitStack() as resources:
                    host=resources.enter_context(vx.open_library(ROOT/"native"/profile))
                    client=vx.VxInferenceServiceClient(host)
                    runtime=client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
                    resources.callback(client.release_runtime,pb.RuntimeRef(runtime_id=runtime.runtime_id))
                    model=client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                        graph_path=str(directory/"graph.json"),weight_paths=[str(directory/"model.safetensors")]))
                    resources.callback(client.release_model,pb.ModelRef(model_id=model.model_id))
                    compiled=client.compile_model(pb.CompileModelRequest(model_id=model.model_id,
                        policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                            backends=["cpu"],operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
                    resources.callback(client.release_compiled_model,pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                    context=client.create_execution_context(pb.CreateExecutionContextRequest(compiled_model_id=compiled.compiled_model_id,
                        autoregressive=pb.AutoregressiveExecution(sampling=pb.SamplingMode.SAMPLING_MODE_GREEDY,max_steps=3)))
                    resources.callback(client.release_execution_context,pb.ExecutionContextRef(context_id=context.context_id))
                    observed=[]
                    for index,count in enumerate((1,2)):
                        packet=client.generate(pb.GenerateRequest(context_id=context.context_id,
                            initial_inputs=initial if index==0 else None,budget=pb.GenerationBudget(max_steps=count)))
                        self.assertEqual(packet.report.backend,"cpu")
                        self.assertFalse(packet.report.fallback and packet.report.fallback.operator_fallback_used)
                        output=next(value for value in packet.outputs if value.name=="generated_ids")
                        self.assertEqual(output.dtype,pb.DataType.DATA_TYPE_I32)
                        observed.extend(np.frombuffer(output.inline,dtype=np.int32).tolist())
                    self.assertEqual(observed,expected)

    def test_qwen3_next_recurrent_and_routed_prefix_feedback(self):
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
        self.compare("qwen3_next",Qwen3NextConfig,Qwen3NextForCausalLM)

    def test_qwen3_5_dense_prefix_feedback(self):
        from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
        self.compare("qwen3_5_text",Qwen3_5TextConfig,Qwen3_5ForCausalLM)

    def test_qwen3_5_moe_prefix_feedback(self):
        from transformers import Qwen3_5MoeTextConfig,Qwen3_5MoeForCausalLM
        self.compare("qwen3_5_moe_text",Qwen3_5MoeTextConfig,Qwen3_5MoeForCausalLM)

    def test_qwen3_moe_prefix_feedback(self):
        from transformers import Qwen3MoeConfig,Qwen3MoeForCausalLM
        self.compare("qwen3_moe",Qwen3MoeConfig,Qwen3MoeForCausalLM)

    def test_qwen4_hyperconnection_qsa_ple_prefix_feedback(self):
        reference_scope = ExitStack()
        self.addCleanup(reference_scope.close)
        config_cls,model_cls=reference_scope.enter_context(qwen4_reference())
        self.compare("qwen4_exp_text",config_cls,model_cls,raw=tiny_qwen4(),seed=4819)

    def test_llama4_chunk_and_sequence_prefix_feedback(self):
        from transformers import Llama4TextConfig,Llama4ForCausalLM
        self.compare("llama4_text",Llama4TextConfig,Llama4ForCausalLM)

    def deepseek(self,family,config_cls,model_cls,seed=3801):
        from tools.exporter.hf_deepseek import DeepSeekDecoderConfig,build_deepseek_decoder_graph
        self.compare(family,config_cls,model_cls,raw=tiny_deepseek_config(family),seed=seed,
            validator=DeepSeekDecoderConfig.from_hf,
            builder=lambda config,source:build_deepseek_decoder_graph(config,source,generate=True))

    def test_deepseek_v2_mla_prefix_feedback(self):
        from transformers import DeepseekV2Config,DeepseekV2ForCausalLM
        self.deepseek("deepseek_v2",DeepseekV2Config,DeepseekV2ForCausalLM)

    def test_deepseek_v3_routed_prefix_feedback(self):
        from transformers import DeepseekV3Config,DeepseekV3ForCausalLM
        self.deepseek("deepseek_v3",DeepseekV3Config,DeepseekV3ForCausalLM)

    def test_deepseek_v4_mhc_compressed_prefix_feedback(self):
        from transformers import DeepseekV4Config,DeepseekV4ForCausalLM
        self.deepseek("deepseek_v4",DeepseekV4Config,DeepseekV4ForCausalLM,seed=746)


if __name__ == "__main__":
    unittest.main()

