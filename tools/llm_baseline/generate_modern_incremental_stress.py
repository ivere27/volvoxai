#!/usr/bin/env python3
"""Independent small HF fixtures and fixed-capacity prefix-oracle corpora.

Numerical importer tests use the same import-safe fixture factories as this CLI.
Capacity and seed are explicit inputs; official Qwen4 sources are hash-pinned
and their compatibility bindings are scoped to the reference owner's context.
Torch is imported only when constructing or executing an oracle. No checkpoint
is downloaded, and these constructed models do not qualify pretrained quality
or an architecture's advertised maximum context.
"""
from __future__ import annotations

import argparse
import ast
import base64
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import sys
import traceback
import types
from unittest import SkipTest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save

ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(ROOT))

FAMILIES = ("qwen3_moe", "llama4_text", "qwen3_next", "qwen3_5_text",
    "qwen3_5_moe_text", "qwen4_exp_text", "deepseek_v2", "deepseek_v3", "deepseek_v4",
    "llama4_text-image", "qwen3_5_text-image")
REFERENCE_POLICIES = ("raw-hf", "canonical-lowest-index")


@contextmanager
def deepseek_v4_reference_policy(reference, policy="raw-hf"):
    """Optionally refine HF's unspecified indexer ties; never call this raw HF.

    The scorer, projections, compression and attention remain the HF bodies.
    Only selection uses a stable descending sort of the actual scorer output.
    No epsilon perturbation or tolerance is used to classify equal scores.
    """
    if policy not in REFERENCE_POLICIES:
        raise ValueError("unknown reference policy: " + policy)
    if policy == "raw-hf":
        yield
        return
    import torch

    indexers = [module for module in reference.modules()
                if type(module).__name__ == "DeepseekV4Indexer"]
    if not indexers:
        raise ValueError("canonical-lowest-index requires a DeepSeek V4 indexer")
    with ExitStack() as hooks:
        for indexer in indexers:
            scores = []

            def capture(module, arguments, output, scores=scores):
                scores.append(output)

            def select(module, arguments, keywords, output, scores=scores):
                actual = scores.pop()
                positions = arguments[2] if len(arguments) > 2 else keywords["position_ids"]
                ready = (positions + 1) // module.compress_rate
                future = torch.arange(actual.shape[-1], device=actual.device) >= ready.unsqueeze(-1)
                masked = actual.masked_fill(future, float("-inf"))
                indices = masked.argsort(dim=-1, descending=True, stable=True)[..., :output.shape[-1]]
                return torch.where(indices < ready.unsqueeze(-1), indices, -1)

            hooks.callback(indexer.scorer.register_forward_hook(capture).remove)
            hooks.callback(indexer.register_forward_hook(select, with_kwargs=True).remove)
        yield


def tiny_dense_config(**changes):
    result = {"model_type": "llama", "vocab_size": 32, "hidden_size": 16,
              "intermediate_size": 24, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2,
              "max_position_embeddings": 16, "rms_norm_eps": 1e-6,
              "hidden_act": "silu", "tie_word_embeddings": True}
    result.update(changes)
    return result


def tiny_config(family="qwen3_moe", **changes):
    if family.startswith("deepseek_"):
        return tiny_deepseek_config(family, **changes)
    raw = dict(model_type=family, vocab_size=24, hidden_size=16,
        intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
        rms_norm_eps=1e-6, tie_word_embeddings=False, hidden_act="silu",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.})
    if family == "llama4_text":
        raw.update(intermediate_size_mlp=32, num_local_experts=3,
            num_experts_per_tok=2, moe_layers=[1], no_rope_layers=[1, 0],
            attention_chunk_size=2, floor_scale=2)
    else:
        if family != "qwen3_5_text":
            raw.update(num_experts=3, num_experts_per_tok=2, moe_intermediate_size=12,
                norm_topk_prob=True, shared_expert_intermediate_size=12)
        if family != "qwen3_moe":
            raw.update(layer_types=["linear_attention", "full_attention"],
                linear_num_key_heads=1, linear_num_value_heads=2,
                linear_key_head_dim=4, linear_value_head_dim=4, linear_conv_kernel_dim=3)
            raw["rope_parameters"]["partial_rotary_factor"] = .5
            if family.startswith("qwen3_5"):
                raw["rope_parameters"].update(mrope_section=[1, 1, 0], mrope_interleaved=True)
    raw.update(changes)
    return raw


def tiny_deepseek_config(family="deepseek_v3", **changes):
    result = dict(model_type=family, vocab_size=32, hidden_size=16,
        intermediate_size=24, moe_intermediate_size=12, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=16,
        q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4,
        v_head_dim=4, n_routed_experts=4, num_experts_per_tok=2,
        n_shared_experts=1, n_group=2, topk_group=1, first_k_dense_replace=1,
        rope_theta=10000, rms_norm_eps=1e-6)
    if family == "deepseek_v4":
        result.update(num_key_value_heads=1, head_dim=8, num_hidden_layers=3,
            hc_mult=2, hc_sinkhorn_iters=5, o_groups=2, o_lora_rank=4,
            index_n_heads=2, index_head_dim=4, index_topk=1, sliding_window=3,
            compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
            layer_types=["sliding_attention", "heavily_compressed_attention", "compressed_sparse_attention"],
            mlp_layer_types=["hash_moe", "moe", "moe"], swiglu_limit=0.12)
    result.update(changes)
    return result


def tiny_qwen4():
    return dict(model_type="qwen4_exp_text",vocab_size=24,hidden_size=16,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=32,
        num_experts=3,num_experts_per_tok=2,moe_intermediate_size=12,shared_expert_intermediate_size=12,
        hc_count=2,hc_lowrank=4,linear_num_key_heads=1,linear_num_value_heads=2,
        linear_key_head_dim=4,linear_value_head_dim=4,linear_conv_kernel_dim=3,
        layer_types=["linear_attention","indexed_attention"],indexer_n_heads=2,indexer_kv_heads=1,
        indexer_head_dim=8,indexer_budget=2,indexer_compress_ratio=2,ple_layer_ids=[1],ple_embed_dim=8,
        ngram_size=3,heads_per_ngram=2,ngram_vocab_size_base=5,make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,eos_token_id=23,ple_conv_kernel_size=3,output_gate_type="sigmoid",
        rope_parameters={"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":.5,
            "mrope_section":[1,1,0],"mrope_interleaved":True})


def tiny_vision_config():
    return dict(model_type="qwen3_5_vision", hidden_size=16, intermediate_size=24,
        num_heads=2, depth=2, in_channels=3, patch_size=2, temporal_patch_size=2,
        spatial_merge_size=2, num_position_embeddings=16, out_hidden_size=16,
        hidden_act="gelu_pytorch_tanh")


def execute_native(profile,graph,weights,inputs):
    from tools.exporter.runtime_ir import import_runtime_package
    from tools.exporter.native_execution import NativeGraphRunner
    root=ROOT
    ir=import_runtime_package(graph,weights)
    if graph["dimensions"]:ir=ir.bind_shape_profile({"S":inputs["token_ids"].shape[1]})
    with patch.dict(os.environ,{"VOLVOXAI_NATIVE_RUNNER":str(root/"native"/profile)}):
        runner=NativeGraphRunner(ir,weights)
        return runner.run(runner.bind(inputs)).outputs


@contextmanager
def _mapping_entries(mapping, values):
    """Restore only the names this compatibility scope owns."""
    absent = object()
    previous = {key: mapping.get(key, absent) for key in values}
    mapping.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is absent:
                mapping.pop(key, None)
            else:
                mapping[key] = value


@contextmanager
def qwen4_reference(reference_dir=None):
    """Load the pinned official oracle and restore compatibility bindings."""
    import torch
    from transformers import AutoModel
    import transformers.configuration_utils as configuration
    import transformers.integrations as integrations
    import transformers.modeling_utils as modeling_utils
    import transformers.utils.generic as generic
    import transformers.vision_utils as vision

    path = Path(reference_dir or os.environ.get(
        "VOLVOXAI_QWEN4_REFERENCE_DIR", "/tmp/volvoxai-architecture-reference"))
    expected = {
        "configuration_qwen4_exp": "066ad3f5fccb26a8a263a5073f00bef4a2a116be45b872c093d05148ecc7014c",
        "modeling_qwen4_exp": "c98846c8f2faa2ecbd8dc19977c8e38f30da60b1b9661af04bfd2e54d85ab4f9",
        "qwen4_vision_utils": "bcecd5a92b3266b9926272a549d2b1a0f1fe7646c698c0fa19bd96f976085356",
        "qwen4_generic": "222da9314771386b9666d15b242074ffc22dd440f74ce864137b793e1c2cfcda",
    }
    for name, digest in expected.items():
        file = path / (name + ".py")
        if not file.is_file():
            raise SkipTest("set VOLVOXAI_QWEN4_REFERENCE_DIR to the pinned official HF reference files")
        if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise AssertionError("reference source hash mismatch: " + str(file))
    helpers = {"torch": torch, "F": torch.nn.functional, "re": re}
    for file, names in (("qwen4_generic", {"is_flash_attention_requested", "get_max_seqlen"}),
            ("qwen4_vision_utils", {"get_vision_cu_seqlens", "get_vision_attention_seqlens",
                "get_vision_position_ids", "_interpolation_axis_taps_weights",
                "get_vision_interpolation_indices_and_weights"})):
        tree = ast.parse((path / (file + ".py")).read_text())
        body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        body.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names)
        if len(body) != len(names) + 1:
            raise AssertionError("pinned helper declaration missing")
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(path / (file + ".py")), "exec"), helpers)
    with ExitStack() as scope:
        # Transformers 5.13 lacks these API associations. The loaded numerical
        # function bodies are literal official source, not exporter math.
        if not hasattr(integrations, "use_kernel_func_from_hub_with_fallback"):
            scope.enter_context(patch.object(integrations, "use_kernel_func_from_hub_with_fallback",
                lambda *args, **kwargs: lambda function: function, create=True))
        scope.enter_context(patch.object(generic, "get_max_seqlen", helpers["get_max_seqlen"], create=True))
        for name in ("get_vision_attention_seqlens", "get_vision_position_ids",
                "get_vision_interpolation_indices_and_weights"):
            scope.enter_context(patch.object(vision, name, helpers[name], create=True))
        scope.enter_context(patch.object(configuration, "ALLOWED_LAYER_TYPES",
            configuration.ALLOWED_LAYER_TYPES + ("indexed_attention",)))
        package = "transformers.models.qwen4_exp"
        pkg = types.ModuleType(package)
        pkg.__path__ = [str(path)]
        scope.enter_context(_mapping_entries(sys.modules, {package: pkg}))
        modules = {}
        for name in ("configuration_qwen4_exp", "modeling_qwen4_exp"):
            full = package + "." + name
            spec = importlib.util.spec_from_file_location(full, path / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            scope.enter_context(_mapping_entries(sys.modules, {full: module}))
            spec.loader.exec_module(module)
            modules[name] = module
        configs, modeling = modules["configuration_qwen4_exp"], modules["modeling_qwen4_exp"]
        causal_mask = modeling.create_causal_mask
        if "allow_is_causal_skip" not in inspect.signature(causal_mask).parameters:
            def eager_mask(**kwargs):
                kwargs.pop("allow_is_causal_skip", None)
                result = causal_mask(**kwargs)
                if result is None:
                    raise AssertionError("eager causal mask unexpectedly elided")
                return result
            modeling.create_causal_mask = eager_mask
        scope.enter_context(_mapping_entries(AutoModel._model_mapping._extra_content, {
            configs.Qwen4ExpVisionConfig: modeling.Qwen4ExpVisionModel,
            configs.Qwen4ExpTextConfig: modeling.Qwen4ExpTextModel}))
        if "axial" not in modeling_utils.ROPE_INIT_FUNCTIONS:
            scope.enter_context(_mapping_entries(modeling_utils.ROPE_INIT_FUNCTIONS, {
                "axial": modeling.Qwen4ExpVisionRotaryEmbedding.compute_axial_rope_parameters}))
        yield configs.Qwen4ExpTextConfig, modeling.Qwen4ExpForCausalLM


def incremental_fixture(family, resources, *, capacity=16, seed=None, reference_dir=None):
    import torch
    import transformers as hf
    image_conditioned = family.endswith("-image")
    family = family.removesuffix("-image")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("capacity must be a positive integer")
    if family not in (*FAMILIES, "dense_controls"):
        raise ValueError("unknown fixture family: " + family)
    torch.manual_seed((746 if family == "deepseek_v4" else 3801) if seed is None else seed)
    torch.set_num_threads(1)
    if family == "dense_controls":
        from functools import partial
        from tools.exporter.hf_decoder import DecoderConfig, build_decoder_graph
        raw, validator = tiny_dense_config(), DecoderConfig.from_hf
        config_class, model_class = hf.LlamaConfig, hf.LlamaForCausalLM
        builder = partial(build_decoder_graph, compact_kv=True)
    elif family.startswith("deepseek_"):
        from tools.exporter.hf_deepseek import DeepSeekDecoderConfig, build_deepseek_decoder_graph
        config_class = getattr(hf, {"deepseek_v2":"DeepseekV2Config", "deepseek_v3":"DeepseekV3Config", "deepseek_v4":"DeepseekV4Config"}[family])
        model_class = getattr(hf, {"deepseek_v2":"DeepseekV2ForCausalLM", "deepseek_v3":"DeepseekV3ForCausalLM", "deepseek_v4":"DeepseekV4ForCausalLM"}[family])
        raw, validator, builder = tiny_deepseek_config(family), DeepSeekDecoderConfig.from_hf, build_deepseek_decoder_graph
    elif family == "qwen4_exp_text":
        from tools.exporter.hf_qwen4 import Qwen4DecoderConfig, build_qwen4_decoder_graph
        config_class, model_class = resources.enter_context(qwen4_reference(reference_dir))
        raw, validator, builder = tiny_qwen4(), Qwen4DecoderConfig.from_hf, build_qwen4_decoder_graph
    else:
        from tools.exporter.hf_extended import ExtendedDecoderConfig, build_extended_decoder_graph
        names = {"qwen3_moe":("Qwen3MoeConfig", "Qwen3MoeForCausalLM"),
            "llama4_text":("Llama4TextConfig", "Llama4ForCausalLM"),
            "qwen3_next":("Qwen3NextConfig", "Qwen3NextForCausalLM"),
            "qwen3_5_text":("Qwen3_5TextConfig", "Qwen3_5ForCausalLM"),
            "qwen3_5_moe_text":("Qwen3_5MoeTextConfig", "Qwen3_5MoeForCausalLM")}
        config_class, model_class = (getattr(hf, name) for name in names[family])
        raw, validator, builder = tiny_config(family), ExtendedDecoderConfig.from_hf, build_extended_decoder_graph
    if image_conditioned and family == "qwen3_5_text":
        raw["head_dim"] = 16
        raw["rope_parameters"]["mrope_section"] = [1,1,2]
    raw["max_position_embeddings"] = max(raw["max_position_embeddings"], capacity)
    config = config_class(**raw)
    config._attn_implementation = "eager"
    reference = model_class(config).float().eval()
    source = {name:value.detach().numpy() for name,value in reference.state_dict().items()}
    admitted = validator(config.to_dict(), max_context=capacity)
    graph, weights = builder(admitted, source, cached=True,
        **({"image_features_count":2} if image_conditioned else {}))
    if family == "dense_controls":
        graph["execution"]["generation"]["incremental_controls"] = {"position_input":"position_ids"}
    return reference, admitted, graph, weights


def reference_rows(reference, prompt, count, initial=None):
    import torch
    prefix, scores, ids = list(prompt), [], []
    with torch.inference_mode():
        for _ in range(count):
            arguments = {"input_ids":torch.tensor([prefix]),
                "attention_mask":torch.ones(1, len(prefix), dtype=torch.long),
                "position_ids":torch.arange(len(prefix))[None], "use_cache":False}
            if initial is not None and "image_features" in initial:
                embeddings = reference.get_input_embeddings()(arguments.pop("input_ids"))
                for row in range(len(prompt)):
                    if initial["image_token_mask"][0,row]:
                        embeddings[0,row] = torch.from_numpy(initial["image_features"][initial["image_feature_ids"][row]])
                arguments["inputs_embeds"] = embeddings
                if "mrope_position_ids" in initial:
                    axes = initial["mrope_position_ids"][:,:len(prompt)].copy()
                    for _ in range(len(prefix) - len(prompt)):
                        axes = np.concatenate([axes, np.full((3,1), int(axes.max()) + 1, np.int32)], axis=1)
                    arguments["position_ids"] = torch.from_numpy(axes[:,None]).long()
            logits = reference(**arguments).logits
            scores.append(logits[0, -1].numpy().copy())
            token = int(logits[0, -1].argmax())
            ids.append(token)
            prefix.append(token)
    return scores, ids


def initial_values(graph, config, prompt):
    capacity = config.max_context
    if not 0 < len(prompt) <= capacity:
        raise ValueError("prompt must fit the configured capacity")
    # Poison inactive tokens to catch recurrence seeded beyond active_lengths.
    values = {"token_ids":np.full((1, capacity), 7, np.int32),
        "attention_mask":np.zeros((1, capacity), np.int32),
        "position_ids":np.arange(capacity, dtype=np.int32),
        "active_lengths":np.asarray([len(prompt)], np.int32)}
    values["token_ids"][0, :len(prompt)] = prompt
    values["attention_mask"][0, :len(prompt)] = 1
    if "sequence_ids" in graph["inputs"]:
        values["sequence_ids"] = np.arange(capacity, dtype=np.int32)
    if "chunk_attention_mask" in graph["inputs"]:
        chunks = np.arange(capacity) // config.chunk_size
        values["chunk_attention_mask"] = (chunks[:, None] == chunks[None, :])[None].astype(np.int32)
    if "image_features" in graph["inputs"]:
        values["image_features"] = np.random.default_rng(151).standard_normal(
            graph["inputs"]["image_features"]["shape"]).astype(np.float32) * .2
        values["image_feature_ids"] = np.ones(capacity, np.int32)
        values["image_feature_ids"][:len(prompt)] = 0
        values["image_feature_ids"][2] = 1
        values["image_token_mask"] = np.ones((1,capacity), np.int32)
        values["image_token_mask"][0,:len(prompt)] = 0
        values["image_token_mask"][0,1:3] = 1
    if "mrope_position_ids" in graph["inputs"]:
        values["mrope_position_ids"] = np.tile(np.arange(capacity, dtype=np.int32), (3,1))
        values["mrope_position_ids"][:2,2] = 1
    return values


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--prompts", default="31,63")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--budgets", default="1,3,5,7")
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--with-boundary", action="store_true")
    parser.add_argument("--deepseek-v4-seed", type=int, default=746)
    parser.add_argument("--reference-policy", choices=REFERENCE_POLICIES, default="raw-hf",
        help="canonical-lowest-index explicitly refines only DeepSeek V4 indexer ties; it is not raw HF parity")
    parser.add_argument("--require-index-gap", type=float, default=0.0)
    args = parser.parse_args()
    if args.out.exists() or args.receipt.exists() or args.out.resolve() == args.receipt.resolve():
        parser.error("corpus and receipt require distinct, new paths")
    try:
        prompts = [int(value) for value in args.prompts.split(",")]
        budgets = [int(value) for value in args.budgets.split(",")]
    except ValueError:
        parser.error("prompts and budgets require comma-separated integers")
    families = args.families.split(",")
    if args.steps <= 0 or sum(budgets) != args.steps or min(budgets) <= 0:
        parser.error("positive budgets must sum to the positive step count")
    if min(prompts) < 3 or max(prompts) + args.steps - 1 > args.capacity:
        parser.error("prompts and generated rows must fit the declared capacity")
    if any(family not in FAMILIES for family in families):
        parser.error("unknown fixture family")
    if len(families) != len(set(families)) or len(prompts) != len(set(prompts)):
        parser.error("families and prompt lengths must not repeat")
    import torch
    import transformers
    receipt = {"status":"FAIL", "capacity":args.capacity, "prompt_lengths":prompts,
        "reference_policy":args.reference_policy,
        "decode_steps":args.steps, "budgets":budgets,
        "reference":{"torch":torch.__version__, "transformers":transformers.__version__,
            "dtype":"float32", "attention_implementation":"eager", "use_cache":False,
            "checkpoint_download":False, "deepseek_v4_seed":args.deepseek_v4_seed,
            "deepseek_v4_minimum_indexer_top1_gap":args.require_index_gap}, "cases":[], "timing_claim":False,
        "qualification":"constructed small models at the selected fixed capacity; not actual checkpoints or maximum context"}
    document = {"format":"volvox-modern-incremental/v1", "reference_policy":args.reference_policy, "cases":[]}
    reference_sources = {}
    try:
        for family in families:
            with ExitStack() as resources:
                reference, config, graph, weights = incremental_fixture(family, resources,
                    capacity=args.capacity,
                    seed=args.deepseek_v4_seed if family == "deepseek_v4" else None)
                policy = args.reference_policy if family == "deepseek_v4" else "raw-hf"
                if family == "deepseek_v4":
                    resources.enter_context(deepseek_v4_reference_policy(reference, policy))
                index_gaps = []
                if family == "deepseek_v4":
                    for name, module in reference.named_modules():
                        if type(module).__name__ == "DeepseekV4IndexerScorer":
                            def observe(module, arguments, output):
                                if output.shape[-1] < 2:
                                    return  # No top-two gap exists for a single candidate.
                                scores = output[0,-1].detach().topk(2).values
                                gap = float(scores[0] - scores[1])
                                assert gap >= args.require_index_gap, "HF top-k boundary tie/near-tie at observed output row"
                                index_gaps.append(gap)
                            handle = module.register_forward_hook(observe)
                            resources.callback(handle.remove)
                source = Path(inspect.getsourcefile(type(reference)))
                reference_sources[str(source)] = digest(source)
                vocabulary = int(reference.config.vocab_size)
                boundary = None
                if args.with_boundary:
                    length = args.capacity - 1
                    prompt = [2 + (row * 7 + row // 5) % (vocabulary - 2) for row in range(length)]
                    values = initial_values(graph, config, prompt)
                    scores, ids = reference_rows(reference, prompt, 2, values)
                    boundary = {"inputs":[{"name":name, "shape":list(value.shape), "dtype":str(value.dtype),
                        "data":base64.b64encode(value.tobytes()).decode("ascii")} for name,value in values.items()],
                        "expected_logits":[value.tolist() for value in scores], "expected_ids":ids}
                for length in prompts:
                    prompt = [2 + (row * 7 + row // 5) % (vocabulary - 2) for row in range(length)]
                    values = initial_values(graph, config, prompt)
                    scores, ids = reference_rows(reference, prompt, args.steps, values)
                    case = {"name":family + "-prompt" + str(length), "reference_policy":policy, "graph":graph,
                        "weight_shards":[base64.b64encode(save(weights, metadata={"format":"pt"})).decode("ascii")],
                        "inputs":[{"name":name, "shape":list(value.shape), "dtype":str(value.dtype),
                            "data":base64.b64encode(value.tobytes()).decode("ascii")} for name,value in values.items()],
                        "expected_logits":[value.tolist() for value in scores], "expected_ids":ids,
                        "budgets":budgets}
                    if boundary is not None:
                        case["boundary"] = boundary
                    document["cases"].append(case)
                    receipt["cases"].append({"name":case["name"], "reference_policy":policy, "rows":len(scores),
                        "generated_ids":ids, "vocabulary":vocabulary})
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps(document, allow_nan=False) + "\n")
                    print("ORACLE", case["name"], len(scores), flush=True)
                if index_gaps:
                    receipt["deepseek_v4_indexer_gaps"] = {"minimum":min(index_gaps),
                        "count":len(index_gaps), "values":index_gaps}
        receipt["corpus"] = {"path":str(args.out), "bytes":args.out.stat().st_size,
            "sha256":digest(args.out)}
        receipt["generator_sha256"] = digest(Path(__file__))
        receipt["fixture_sources"] = {str(Path(__file__).relative_to(ROOT)):
            digest(Path(__file__))}
        receipt["reference_sources"] = reference_sources
        receipt["status"] = "PASS"
    except Exception:
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
