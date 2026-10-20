"""Pinned official V4.1 architecture, explicitly widened F32 (no QAT).

This qualification adapter replaces only the official TileLang primitives with
independent torch operations. It is not the released FP8/FP4 execution oracle.
"""
import importlib.util
import ast
import inspect
import textwrap
import base64
import copy
import hashlib
import json
import os
import tempfile
import unittest
import sys
import types
from contextlib import contextmanager, ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
try:
    import torch
except ImportError:
    torch = None

REVISION = "5254448103b6254fc646dacbca1cbbf20e9216c1"
REPOSITORY = Path(__file__).resolve().parents[2]
REFERENCE_DIR = Path(os.environ.get("VOLVOXAI_DEEPSEEK_V41_REFERENCE_DIR",
    REPOSITORY / "build/llm/deepseek-support-001/official/DeepSeek-V4.1-Flash" / REVISION / "inference"))
REFERENCE_PINS = {
    'model.py': 'e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d',
    'engram.py': '11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897',
    'vision.py': '5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c',
    'kernel.py': '1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455',
    'image_processor.py': '482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272',
}
QUALIFICATION = {
    "revision": REVISION,
    "arithmetic": "F32 architecture reference; FP8/FP4 activation and KV QAT disabled",
    "selection": "raw official torch topk; ties have no portable identity promise",
    "engram": "explicit identity token map; already decoded F32 embedding",
    "checkpoint_download": False,
}


@contextmanager
def reference_policy(module, selection="raw-hf"):
    """Alter only tie selection in the two pinned official selector functions.

    The default executes the original source. Canonical mode is an explicit
    architecture qualification policy, not a claim of raw HF parity.
    """
    if selection == "raw-hf":
        yield
        return
    if selection != "canonical-lowest-index":
        raise ValueError(f"unknown V4.1 reference selection policy: {selection}")
    verify_reference()
    targets = ((module.Indexer, "forward"), (module, "select_candidate_blocks"))
    originals = [(owner, name, getattr(owner, name)) for owner, name in targets]
    helper = "_volvox_canonical_topk"
    if helper in module.__dict__:
        raise AssertionError("V4.1 reference policy is already active")

    def stable_topk(values, k, dim=-1, largest=True, sorted=True):
        if dim != -1 or not largest:
            raise AssertionError("unexpected pinned V4.1 topk contract")
        indices = torch.argsort(values, dim=-1, descending=True, stable=True)[..., :k]
        return types.SimpleNamespace(indices=indices, values=values.gather(-1, indices))

    class StableCalls(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "topk":
                self.count += 1
                return ast.copy_location(ast.Call(func=ast.Name(id=helper, ctx=ast.Load()),
                    args=[node.func.value, *node.args], keywords=node.keywords), node)
            return node

    module.__dict__[helper] = stable_topk
    try:
        for owner, name, original in originals:
            tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
            change = StableCalls()
            tree = change.visit(tree)
            if change.count != 1:
                raise AssertionError("pinned V4.1 selector must contain exactly one topk call")
            scope = {}
            exec(compile(ast.fix_missing_locations(tree), inspect.getfile(original), "exec"),
                 module.__dict__, scope)
            setattr(owner, name, scope[name])
        yield
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)
        module.__dict__.pop(helper, None)


def qualification(selection):
    if selection not in ("raw-hf", "canonical-lowest-index"):
        raise ValueError(f"unknown V4.1 reference selection policy: {selection}")
    return {**QUALIFICATION, "selection_policy": selection,
        "selection": QUALIFICATION["selection"] if selection == "raw-hf" else
        "explicit lowest-index tie policy in official Indexer.forward and select_candidate_blocks only"}


def verify_reference(directory=REFERENCE_DIR):
    for name, digest in REFERENCE_PINS.items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise AssertionError(f"official V4.1 reference identity mismatch: {path}")



def _hc(mixes, scale, base, count, iterations, eps):
    pre = (mixes[..., :count] * scale[0] + base[:count]).sigmoid() + eps
    post = 2 * (mixes[..., count:2*count] * scale[1] + base[count:2*count]).sigmoid()
    comb = (mixes[..., 2*count:] * scale[2] + base[2*count:]).unflatten(-1, (count, count))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def _attention(q, kv, sinks, indices, scale):
    b, s, heads, d = q.shape
    picked = kv[torch.arange(b)[:, None, None], indices.clamp_min(0).long()]
    scores = torch.einsum("bshd,bskd->bshk", q, picked) * scale
    scores = scores.masked_fill(indices[:, :, None, :] < 0, -torch.inf)
    sink = sinks[None, None, :, None].expand(b, s, heads, 1)
    probabilities = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", probabilities, picked)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def official_reference():
    verify_reference()
    kernel = types.ModuleType("kernel")
    kernel.act_quant = lambda x, *a, **k: x
    kernel.fp4_act_quant = lambda x, *a, **k: x
    kernel.fp4_gemm = kernel.fp8_gemm = lambda *a, **k: (_ for _ in ()).throw(AssertionError("quantized GEMM is outside F32 qualification"))
    kernel.hc_split_sinkhorn = _hc
    kernel.sparse_attn = _attention
    image = types.ModuleType("image_processor")
    for i, key in enumerate(("IMAGE", "IMAGE_END", "IMAGE_NEW_LINE", "IMAGE_START")):
        setattr(image, key, i)
    # Restore only our aliases. Restoring the entire sys.modules dictionary
    # would discard lazy torch imports while leaving their C registrations.
    names = ("kernel", "image_processor", "engram", "vision", "_volvox_official_deepseek_v41")
    missing = object()
    before = {name: sys.modules.get(name, missing) for name in names}
    try:
        sys.modules.update(kernel=kernel, image_processor=image)
        _load("engram", REFERENCE_DIR / "engram.py")
        _load("vision", REFERENCE_DIR / "vision.py")
        model = _load("_volvox_official_deepseek_v41", REFERENCE_DIR / "model.py")
        yield model
    finally:
        for name, value in before.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def tiny_args(module, *, engram=False, capacity=32, top_k=2):
    return module.ModelArgs(max_batch_size=1, max_seq_len=capacity, temperature=0,
        dtype="bf16", expert_dtype=None, vocab_size=32, dim=16, moe_inter_dim=12,
        n_layers=5, n_mtp_layers=0, n_heads=4, n_routed_experts=4,
        n_shared_experts=1, n_activated_experts=2, route_scale=1.5,
        swiglu_limit=0.12, q_lora_rank=8, head_dim=8, rope_head_dim=4,
        norm_eps=1e-6, o_groups=2, o_lora_rank=4, window_size=3,
        compress_ratios=(0,2,2,1,1), kv_source_layers=(1,3),
        index_source_layers=(1,3,4), candidate_source_layer=3,
        candidate_topk_blocks=2, candidate_block_size=2,
        index_n_heads=2, index_head_dim=4, index_topk=top_k,
        hc_mult=2, hc_sinkhorn_iters=5, engram_layer_ids=(1,) if engram else (),
        engram_max_ngram_size=3 if engram else 1,
        engram_n_heads=2 if engram else 0, engram_head_dim=4 if engram else 0,
        engram_vocab_size=5 if engram else 0, engram_num_embeddings=(36,) if engram else (),
        engram_compressed_vocab_size=32 if engram else 0, engram_pad_id=2)


def hf_config(args):
    aliases = dict(dim="hidden_size", moe_inter_dim="moe_intermediate_size", n_layers="num_hidden_layers",
        n_heads="num_attention_heads", n_activated_experts="num_experts_per_tok", score_func="scoring_func",
        route_scale="routed_scaling_factor", rope_head_dim="qk_rope_head_dim", norm_eps="rms_norm_eps",
        window_size="sliding_window", kv_source_layers="kv_source_layer_ids", index_source_layers="index_source_layer_ids",
        candidate_source_layer="candidate_source_layer_id", engram_pad_id="engram_pad_token_id")
    text = {aliases.get(k,k): list(v) if isinstance(v,tuple) else v for k,v in vars(args).items()}
    for name in ("dtype", "expert_dtype", "max_batch_size", "max_seq_len", "temperature", "gate_temp", "n_mtp_layers", "original_seq_len", "rope_factor", "beta_fast", "beta_slow"):
        text.pop(name, None)
    text.update(model_type="deepseek_v41_text", num_key_value_heads=1,
        max_position_embeddings=args.max_seq_len, hidden_act="silu", topk_method="noaux_tc",
        tie_word_embeddings=False, attention_bias=False, num_nextn_predict_layers=0)
    return dict(model_type="deepseek_v41", architectures=["DeepseekV41ForCausalLM"], text_config=text)


def initialized_model(module, args):
    torch.manual_seed(260910)
    torch.set_num_threads(1)
    if args.engram_layer_ids:
        # The released embedding's packed weight+scale reader is storage logic.
        # F32 qualification supplies already widened embedding rows instead.
        module.ParallelEngramEmbedding = torch.nn.Embedding
        # Toy tokenizer identity map is explicit and independent of graph lowering.
        with patch("engram.build_compressed_token_map", return_value=(list(range(args.vocab_size)), args.vocab_size)):
            model = module.Transformer(args, tokenizer=object()).float().eval()
    else:
        model = module.Transformer(args).float().eval()
    with torch.no_grad():
        for name, weight in model.named_parameters():
            weight.copy_(torch.randn_like(weight) * 0.08)
            if name.endswith("norm.weight") or name.endswith(("q_weight", "k_weight")):
                weight.add_(1)
    return model


def prefix_tokens(length):
    tokens = [2 + (i * 7 + i // 5) % 30 for i in range(length)]
    # In V4.1 a real token equal to the left-padding ID does not reset history.
    for i in (0, 1, 3, 5, 6):
        if i < length:
            tokens[i] = 2
    return np.array([tokens], np.int32)


@unittest.skipIf(torch is None, "official F32 reference requires torch")
class DeepSeekLatestReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not REFERENCE_DIR.is_dir() and "VOLVOXAI_DEEPSEEK_V41_REFERENCE_DIR" not in os.environ:
            raise unittest.SkipTest("supply the pinned official source directory; no automatic downloads")
        verify_reference()

    def test_source_pins_refuse_modification_and_imports_are_scoped(self):
        names = ("kernel", "image_processor", "engram", "vision", "_volvox_official_deepseek_v41")
        before = {name: sys.modules.get(name) for name in names}
        with official_reference() as module:
            self.assertEqual(module.__name__, "_volvox_official_deepseek_v41")
        self.assertEqual(before, {name: sys.modules.get(name) for name in names})
        with self.assertRaisesRegex(RuntimeError, "scope failure"):
            with official_reference():
                raise RuntimeError("scope failure")
        self.assertEqual(before, {name: sys.modules.get(name) for name in names})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in REFERENCE_PINS:
                (root / name).write_bytes((REFERENCE_DIR / name).read_bytes())
            verify_reference(root)
            with (root / "model.py").open("ab") as stream:
                stream.write(b"\n# changed\n")
            with self.assertRaisesRegex(AssertionError, "identity mismatch"):
                verify_reference(root)

    def test_config_namespace_and_token_map_refusals(self):
        from tools.exporter.errors import ExporterError
        from tools.exporter.hf_deepseek_v41 import DeepSeekV41Config, build_deepseek_v41_graph
        with official_reference() as module:
            args = tiny_args(module, engram=True)
            raw = hf_config(args)
            changes = [
                {"kv_source_layer_ids": [1, 1, 3]}, {"kv_source_layer_ids": [3, 1]},
                {"kv_source_layer_ids": [1, 5]}, {"index_source_layer_ids": [3, 4]},
                {"index_source_layer_ids": [0, 1, 3, 4]},
                {"compress_ratios": [0, 2, 1, 1, 1]},
                {"candidate_source_layer_id": 2}, {"candidate_source_layer_id": True},
                {"candidate_source_layer_id": 1},  # Later sources change the bank domain to rate 1.
                {"candidate_topk_blocks": 0}, {"candidate_block_size": 0},
                {"index_topk": 0}, {"engram_num_embeddings": [35]},
                {"engram_layer_ids": [1, 1]}, {"engram_compressed_vocab_size": 33},
                {"engram_pad_token_id": 32}, {"qk_rope_head_dim": 3},
            ]
            for changeset in changes:
                altered = copy.deepcopy(raw)
                altered["text_config"].update(changeset)
                with self.subTest(config=changeset), self.assertRaises(ExporterError):
                    DeepSeekV41Config.from_hf(altered, max_context=32)
            config = DeepSeekV41Config.from_hf(raw, max_context=32)
            model = initialized_model(module, args)
            source = {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}
            self.assertEqual(set(source), set(config.weight_shapes()))
            valid_map = np.arange(args.vocab_size, dtype=np.int32)
            name = "layers.0.attn.wq_a.weight"
            variants = [{**source, "unknown.weight": np.ones(1, np.float32)},
                {key: value for key, value in source.items() if key != name},
                {**source, name: source[name].reshape(-1)},
                {**source, name: np.full_like(source[name], np.nan)},
                {**source, name: source[name].astype(np.int32)},
                {**source, "model." + name: source[name]},
                {**source, "mtp.0.unknown.weight": np.ones(1, np.float32)}]
            for i, weights in enumerate(variants):
                with self.subTest(weight_case=i), self.assertRaises(ExporterError):
                    build_deepseek_v41_graph(config, weights, token_map=valid_map)
            for token_map in (None, valid_map[:-1], valid_map.astype(np.float32),
                              np.zeros_like(valid_map), valid_map - 1, valid_map + 1):
                with self.subTest(token_map=token_map), self.assertRaises(ExporterError):
                    build_deepseek_v41_graph(config, source, token_map=token_map)
            with self.assertRaisesRegex(ExporterError, "caching is not qualified"):
                build_deepseek_v41_graph(config, source, token_map=valid_map, cached=True)

    def test_canonical_selection_is_explicit_scoped_and_preserves_ranked_rows(self):
        with official_reference() as module:
            original = module.Indexer.forward, module.select_candidate_blocks
            scores = torch.tensor([[[0., 0., 0., 0., 0., 0., -torch.inf]]])
            ranked = torch.tensor([[[3., 2., 1., -1., -2., -3., -torch.inf]]])
            visible = torch.tensor([[6]])
            raw = module.select_candidate_blocks(scores, visible, 2, 2)
            ranked_raw = module.select_candidate_blocks(ranked, visible, 2, 2)
            with reference_policy(module, "raw-hf"):
                self.assertEqual(original, (module.Indexer.forward, module.select_candidate_blocks))
            with self.assertRaisesRegex(RuntimeError, "restore"):
                with reference_policy(module, "canonical-lowest-index"):
                    canonical = module.select_candidate_blocks(scores, visible, 2, 2)
                    torch.testing.assert_close(canonical, torch.tensor([[[1,1,0,0,1,1,0]]], dtype=torch.bool))
                    torch.testing.assert_close(module.select_candidate_blocks(ranked, visible, 2, 2), ranked_raw)
                    topk = module._volvox_canonical_topk
                    self.assertEqual(topk(torch.tensor([[0., 0., -1.]]), 1).indices.tolist(), [[0]])
                    self.assertEqual(topk(torch.tensor([[1., 2., 3.]]), 2).indices.tolist(), [[2, 1]])
                    raise RuntimeError("restore")
            self.assertEqual(original, (module.Indexer.forward, module.select_candidate_blocks))
            self.assertFalse(hasattr(module, "_volvox_canonical_topk"))
            torch.testing.assert_close(module.select_candidate_blocks(scores, visible, 2, 2), raw)
            with self.assertRaisesRegex(ValueError, "unknown"):
                with reference_policy(module, "unspecified"):
                    pass

    def test_official_f32_feature_domain_and_full_prefix_logits(self):
        for engram in (False, True):
            for top_k in (1, 2, 3):
                with self.subTest(engram=engram, top_k=top_k), official_reference() as module:
                    args = tiny_args(module, engram=engram, top_k=top_k)
                    model = initialized_model(module, args)
                    self.assertIsNotNone(model.layers[1].attn.compressor)
                    self.assertIsNone(model.layers[2].attn.compressor)
                    self.assertIsNone(model.layers[2].attn.indexer)
                    self.assertTrue(model.layers[3].attn.indexer.is_candidate_source)
                    self.assertTrue(model.layers[4].attn.indexer.uses_candidates)
                    self.assertFalse(model.layers[4].attn.indexer.owns_k)
                    for length in (1, 3, 7, 11, 31):
                        tokens = torch.from_numpy(prefix_tokens(length)).long()
                        logits, final = full_prefix_logits(model, tokens)
                        self.assertEqual(tuple(logits.shape), (1, length, 32))
                        self.assertTrue(torch.isfinite(logits).all())
                        # The official full-head and last-row branches may use
                        # different BLAS shapes, so this is a numerical check.
                        np.testing.assert_allclose(logits[:, -1].numpy(), final.numpy(), atol=1e-7, rtol=1e-6)

    def test_native_logits_match_pinned_f32_canonical_architecture(self):
        from tools.exporter.native_execution import NativeGraphRunner
        from tools.exporter.runtime_ir import import_runtime_package

        for name, graph, weights, inputs, expected in reference_cases(selection="canonical-lowest-index"):
            ir = import_runtime_package(graph, weights).bind_shape_profile({"S": inputs["token_ids"].shape[1]})
            for profile in ("volvoxai", "volvoxai-lite"):
                with self.subTest(case=name, profile=profile), \
                        patch.dict(os.environ, {"VOLVOXAI_NATIVE_RUNNER": str(REPOSITORY / "native" / profile)}):
                    runner = NativeGraphRunner(ir, weights)
                    actual = runner.run(runner.bind(inputs)).outputs["logits"]
                    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
                    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))

    def test_public_full_prefix_generation_chunks_and_input_ownership(self):
        from safetensors.numpy import load, save
        with patch.object(sys, "path", [str(REPOSITORY / "python"), *sys.path]):
            import volvoxai as vx
        pb = vx.pb

        def tensor(name, value):
            return pb.Tensor(name=name, dtype=pb.DataType.DATA_TYPE_I32,
                             shape=list(value.shape), inline=value.tobytes())

        # Exercise the transported corpus, not just the pre-export graph.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation.json"
            export_corpus(path, selection="canonical-lowest-index", generation=True)
            cases = json.loads(path.read_text())["cases"]
        for case in cases:
            name, graph = case["name"], case["graph"]
            weights = load(base64.b64decode(case["weight_shards"][0], validate=True))
            inputs = {item["name"]: np.frombuffer(base64.b64decode(item["data"], validate=True),
                dtype="<i4").reshape(item["shape"]).copy() for item in case["inputs"]}
            item = case["expected"][0]
            expected = np.frombuffer(base64.b64decode(item["data"], validate=True),
                dtype="<f8").reshape(item["shape"])
            for library in ("libvolvoxai.so", "libvolvoxai-lite.so"):
                with self.subTest(case=name, library=library), \
                        vx.open_library(REPOSITORY / "native" / library) as host, ExitStack() as owners:
                    client = vx.VxInferenceServiceClient(host)
                    runtime = client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
                    owners.callback(client.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id))
                    model = client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                        package=pb.ModelPackage(graph_document=json.dumps(graph).encode(), weight_shards=[save(weights)])))
                    owners.callback(client.release_model, pb.ModelRef(model_id=model.model_id))
                    compiled = client.compile_model(pb.CompileModelRequest(model_id=model.model_id,
                        policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                            backends=["cpu"], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
                    owners.callback(client.release_compiled_model, pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                    initial = pb.GenerationInputs(values=[tensor(key, value) for key, value in inputs.items()])
                    initial_bytes = initial.SerializeToString()
                    arrays_before = {key: value.copy() for key, value in inputs.items()}
                    expected_ids = expected.argmax(-1).reshape(-1).tolist()
                    contexts = []
                    for _ in range(2):
                        context = client.create_execution_context(pb.CreateExecutionContextRequest(
                            compiled_model_id=compiled.compiled_model_id,
                            autoregressive=pb.AutoregressiveExecution(max_steps=3,
                                sampling=pb.SamplingMode.SAMPLING_MODE_GREEDY)))
                        owners.callback(client.release_execution_context, pb.ExecutionContextRef(context_id=context.context_id))
                        contexts.append(context.context_id)

                    def generate(context, count, first):
                        packet = client.generate(pb.GenerateRequest(context_id=context,
                            initial_inputs=initial if first else None, budget=pb.GenerationBudget(max_steps=count),
                            outputs=pb.OutputOptions(storage=pb.OutputStorage.OUTPUT_STORAGE_INLINE)))
                        self.assertEqual(packet.report.status, pb.NativeStatus.NATIVE_STATUS_OK)
                        self.assertEqual(packet.report.backend, "cpu")
                        self.assertFalse(packet.report.fallback.operator_fallback_used)
                        output = next(item for item in packet.outputs if item.name == "generated_ids")
                        self.assertFalse(output.HasField("buffer"))
                        self.assertEqual(list(output.shape), [1, count])
                        return packet, np.frombuffer(output.inline, np.int32).tolist()

                    first, first_ids = generate(contexts[0], 1, True)
                    retained = first.SerializeToString()
                    _, whole = generate(contexts[1], 3, True)
                    _, remaining = generate(contexts[0], 2, False)
                    self.assertEqual(first_ids + remaining, expected_ids)
                    self.assertEqual(whole, expected_ids)
                    self.assertEqual(first.SerializeToString(), retained)
                    self.assertEqual(initial.SerializeToString(), initial_bytes)
                    for key, value in inputs.items():
                        np.testing.assert_array_equal(value, arrays_before[key])

    def test_generation_corpus_preserves_authored_execution_contract(self):
        fixtures = list(generation_cases(selection="canonical-lowest-index"))
        with tempfile.TemporaryDirectory() as directory, \
                patch(__name__ + ".generation_cases", return_value=iter(fixtures)):
            path = Path(directory) / "generation.json"
            self.assertEqual(export_corpus(path, selection="canonical-lowest-index", generation=True), 2)
            document = json.loads(path.read_text())
        for case, (_, graph, _, _, _) in zip(document["cases"], fixtures):
            self.assertEqual(case["graph"]["execution"], graph["execution"])
            self.assertIn("full_prefix", case["graph"]["execution"]["generation"])
            self.assertEqual(case["graph"]["dimensions"], graph["dimensions"])
            self.assertEqual(case["budgets"], [1, 2])
            self.assertEqual(case["generation_steps"], 3)


def reference_cases(*, selection="raw-hf"):
    """The same independently computed cases feed CPU tests and remote runs."""
    from tools.exporter.hf_deepseek_v41 import DeepSeekV41Config, build_deepseek_v41_graph
    for engram in (False, True):
        for top_k in (1, 2, 3):
            with official_reference() as module, reference_policy(module, selection):
                args = tiny_args(module, engram=engram, top_k=top_k)
                model = initialized_model(module, args)
                source = {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}
                config = DeepSeekV41Config.from_hf(hf_config(args), max_context=32)
                graph, weights = build_deepseek_v41_graph(config, source,
                    token_map=np.arange(args.vocab_size, dtype=np.int32))
                for length in (1, 3, 7, 11, 31):
                    tokens = prefix_tokens(length)
                    expected, _ = full_prefix_logits(model, torch.from_numpy(tokens).long())
                    inputs = {"token_ids": tokens, "position_ids": np.arange(length, dtype=np.int32),
                              "attention_mask": np.ones((1, length), np.int32)}
                    name = f"v41-engram{int(engram)}-topk{top_k}-prefix{length}"
                    yield name, graph, weights, inputs, expected.numpy().copy()


def generation_cases(*, selection="raw-hf"):
    """Three full-prefix reference steps; two independent public contexts reuse these inputs."""
    from tools.exporter.hf_deepseek_v41 import DeepSeekV41Config, build_deepseek_v41_graph
    for engram in (False, True):
        with official_reference() as module, reference_policy(module, selection):
            args = tiny_args(module, engram=engram, top_k=2)
            model = initialized_model(module, args)
            source = {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}
            config = DeepSeekV41Config.from_hf(hf_config(args), max_context=32)
            graph, weights = build_deepseek_v41_graph(config, source,
                token_map=np.arange(args.vocab_size, dtype=np.int32), generate=True)
            tokens = prefix_tokens(7)
            prefix, expected = tokens.copy(), []
            for _ in range(3):
                logits, _ = full_prefix_logits(model, torch.from_numpy(prefix).long())
                last = logits[:, -1].numpy().copy()
                expected.append(last)
                prefix = np.concatenate((prefix, last.argmax(-1).astype(np.int32)[:, None]), axis=1)
            inputs = {"token_ids": tokens, "position_ids": np.arange(7, dtype=np.int32),
                      "attention_mask": np.ones((1, 7), np.int32)}
            yield f"v41-engram{int(engram)}-topk2-generate3", graph, weights, inputs, np.stack(expected, axis=1)


def export_corpus(path, *, selection="raw-hf", generation=False):
    """Write portable public-execution fixtures; consumers need neither torch nor source downloads."""
    from safetensors.numpy import save
    from tools.exporter.runtime_ir import import_runtime_package, export_runtime_package

    def tensor(name, value, oracle=False):
        dtype = "F64" if oracle else "I32"
        value = np.ascontiguousarray(value, dtype="<f8" if oracle else "<i4")
        return {"name": name, "shape": list(value.shape), "dtype": dtype,
                "data": base64.b64encode(value.tobytes()).decode("ascii")}

    cases = []
    fixtures = generation_cases if generation else reference_cases
    for name, graph, weights, inputs, expected in fixtures(selection=selection):
        ir = import_runtime_package(graph, weights)
        fixed, pruned = export_runtime_package(ir, weights,
            shape_profile=None if generation else {"S": inputs["token_ids"].shape[1]})
        if generation:
            # Generic complete-evaluation publication regenerates its stateless
            # contract. This corpus retains the authored full-prefix controls.
            if "generation" not in graph["execution"]:
                raise AssertionError("generation fixture requires an authored generation contract")
            fixed["execution"] = copy.deepcopy(graph["execution"])
        cases.append({"name": name, "selection_policy": selection, "graph": fixed,
            "weight_shards": [base64.b64encode(save(pruned, metadata={"format": "pt"})).decode("ascii")],
            "inputs": [tensor(key, value) for key, value in inputs.items()],
            "expected": [tensor("logits", expected, oracle=True)],
            "expected_ids": expected.argmax(-1).tolist(),
            **({"budgets": [1, 2], "generation_steps": 3} if generation else {})})
    document = {"format": "volvox-modern-full-prefix-generation-oracles/v1" if generation else
            "volvox-modern-primitives-wasm-oracles/v1", "count": len(cases),
        "tolerance": {"absolute": 2e-5, "relative": 2e-5}, "qualification": qualification(selection),
        "expected_source": "pinned official F32 architecture; F32 logits widened exactly to F64 transport",
        "reference_sources": REFERENCE_PINS,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": cases}
    with Path(path).open("x") as stream:
        json.dump(document, stream, allow_nan=False)
        stream.write("\n")
    return len(cases)


def full_prefix_logits(model, tokens):
    """Use the official head's full_logits branch without changing the model."""
    hidden = []
    handle = model.head.register_forward_pre_hook(lambda module, args: hidden.append(args[0]))
    try:
        with torch.inference_mode():
            final = model(tokens)[1]
            return model.head.forward(hidden[-1], full_logits=True), final
    finally:
        handle.remove()


if __name__ == "__main__":
    unittest.main()
