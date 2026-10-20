#!/usr/bin/env python3
"""Pinned, development-only HF oracle for native decoder qualification.

Downloads and captures are explicit commands. The runtime never imports this
tool or Transformers. Capture loads verified local SafeTensors offline, uses
CPU F32 eager attention, and writes independently owned logits/KV snapshots.
It deliberately does not collect performance timings.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from typing import Any
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "tools/llm_baseline/models.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_policy(path: Path = POLICY_PATH, *, model: str | None = None) -> dict[str, Any]:
    """Resolve one model's reference settings from the shared checkpoint catalog.

    Overrides replace whole reference fields; no implicit recursive merge or
    numerical-budget widening is performed. File-byte provenance still binds
    the complete catalog, including every checkpoint and override.
    """
    policy = json.loads(path.read_text())
    if policy.get("format") != "volvox-hf-baseline/v1":
        raise ValueError("unsupported baseline policy")
    if model is not None:
        if model not in policy["models"]:
            raise ValueError(f"unknown model alias: {model}")
        overrides = policy["models"][model].get("reference_overrides", {})
        if not isinstance(overrides, dict) or overrides.keys() - policy["reference"].keys():
            raise ValueError("unknown model reference override")
        policy["reference"].update(overrides)
    return policy


def validate_reference_policy(manifest: dict[str, Any], policy_path: Path = POLICY_PATH
                              ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind a numerical capture to the selected policy's actual file bytes."""
    policy = read_policy(policy_path, model=manifest.get("model"))
    pinned = policy["models"].get(manifest.get("model"))
    if (pinned is None or manifest.get("status") != "complete"
            or manifest.get("repo_id") != pinned["repo_id"]
            or manifest.get("revision") != pinned["revision"]
            or manifest.get("config") != pinned["config"]
            or manifest.get("reference_policy") != policy["reference"]
            or manifest.get("execution_dtype") != "float32"
            or manifest.get("provenance", {}).get("policy_sha256") != sha256_file(policy_path)):
        raise ValueError("HF capture differs from its fixed model/numerical policy")
    return policy, pinned


def verify_snapshot(snapshot: Path, model: dict[str, Any]) -> dict[str, Any]:
    """Verify actual bytes, rather than treating an upstream LFS ID as proof."""
    result = {}
    for name, expected in {**model["assets"], **model["weights"]}.items():
        path = snapshot / name
        if not path.is_file():
            raise ValueError(f"snapshot is missing {name}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"snapshot hash/size mismatch: {name}")
        result[name] = {"bytes": size, "sha256": digest}
    if json.loads((snapshot / "config.json").read_text()) != model["config"]:
        raise ValueError("snapshot config differs from the pinned model")
    return result


def download_snapshot(directory: Path, model: dict[str, Any], *, metadata_only: bool) -> None:
    """Keep every downloaded byte in the caller's explicitly selected directory."""
    directory.mkdir(parents=True, exist_ok=True)
    if metadata_only:
        for name, expected in model["assets"].items():
            path = directory / name
            if path.exists() and sha256_file(path) == expected["sha256"]:
                continue
            temporary = directory / (name + ".part")
            try:
                with urllib.request.urlopen(expected["url"], timeout=120) as response:
                    with temporary.open("wb") as target:
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            target.write(block)
                if (temporary.stat().st_size != expected["bytes"] or
                        sha256_file(temporary) != expected["sha256"]):
                    raise ValueError(f"download hash/size mismatch: {name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return
    # local_dir is the snapshot itself; Hub's local-dir metadata avoids another
    # weight copy in its regular cache. Xet and Hub state also stay under it.
    os.environ["HF_HOME"] = str(directory / ".hf-home")
    os.environ["HF_XET_CACHE"] = str(directory / ".hf-home/xet")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=model["repo_id"], revision=model["revision"],
                      local_dir=str(directory), cache_dir=str(directory / ".hf-home/hub"),
                      allow_patterns=list(model["assets"]) + list(model["weights"]))
    verify_snapshot(directory, model)


def _cache_layers(cache: Any) -> list[tuple[Any, Any]]:
    if cache is None:
        raise ValueError("HF model did not return a KV cache")
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    layers = list(cache)
    if not layers or any(len(layer) < 2 for layer in layers):
        raise ValueError("unsupported HF KV cache representation")
    return [(layer[0], layer[1]) for layer in layers]


def _snapshot_tensor(tensor: Any) -> Any:
    # HF DynamicCache mutates in place. Clone before the next transition, even
    # for contiguous CPU tensors, so each saved prefix remains independent.
    return tensor.detach().cpu().contiguous().clone()


def _comparison(actual: Any, expected: Any, tolerance: dict[str, float]) -> dict[str, Any]:
    import torch
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("reference comparison shape/dtype mismatch")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("nonfinite numerical reference")
    delta = (actual - expected).double()
    result = {"max_absolute": float(delta.abs().max()),
              "rms": float(delta.square().mean().sqrt()),
              "allclose": bool(torch.allclose(actual, expected,
                  atol=tolerance["absolute"], rtol=tolerance["relative"]))}
    if not result["allclose"]:
        raise ValueError(f"HF cached/full parity exceeds fixed tolerance: {result}")
    return result


def capture_sequence(model: Any, input_ids: Any, *, decode_steps: int,
                     tolerance: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture one fixed prefix and cached continuation, with a full-run oracle."""
    import torch
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("reference requires a nonempty batch-one prefix")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("reference token IDs must be I32 or I64")
    if decode_steps < 1:
        raise ValueError("decode_steps must be positive")
    tensors: dict[str, Any] = {}
    checks = []
    ids = input_ids.clone().to(dtype=torch.int64, device="cpu")

    def save(name: str, tensor: Any) -> None:
        saved = _snapshot_tensor(tensor)
        if saved.is_floating_point() and not torch.isfinite(saved).all():
            raise ValueError(f"nonfinite reference tensor: {name}")
        tensors[name] = saved

    def save_cache(name: str, cache: Any) -> None:
        for layer, (key, value) in enumerate(_cache_layers(cache)):
            save(f"{name}.layer.{layer:03d}.key", key)
            save(f"{name}.layer.{layer:03d}.value", value)

    with torch.inference_mode():
        positions = torch.arange(ids.shape[1], dtype=torch.int64).unsqueeze(0)
        mask = torch.ones_like(ids)
        save("prefill.input_ids", ids)
        save("prefill.position_ids", positions)
        save("prefill.attention_mask", mask)
        output = model(input_ids=ids, position_ids=positions, attention_mask=mask,
                       use_cache=True, return_dict=True)
        save("prefill.logits", output.logits)
        save_cache("prefill", output.past_key_values)
        cache = output.past_key_values
        greedy_ids = []
        for step in range(decode_steps):
            next_id = output.logits[:, -1, :].argmax(dim=-1).reshape(1, 1)
            greedy_ids.append(int(next_id.item()))
            position = torch.tensor([[ids.shape[1]]], dtype=torch.int64)
            ids = torch.cat((ids, next_id), dim=1)
            mask = torch.ones_like(ids)
            name = f"decode.{step:03d}"
            save(name + ".input_ids", next_id)
            save(name + ".position_ids", position)
            save(name + ".attention_mask", mask)
            output = model(input_ids=next_id, position_ids=position, attention_mask=mask,
                           past_key_values=cache, use_cache=True, return_dict=True)
            cache = output.past_key_values
            save(name + ".logits", output.logits)
            save_cache(name, cache)
            # This is a separately allocated fresh cache over the entire prefix.
            full_positions = torch.arange(ids.shape[1]).unsqueeze(0)
            full = model(input_ids=ids, position_ids=full_positions,
                         attention_mask=mask, use_cache=True, return_dict=True)
            save(f"prefix.{step:03d}.input_ids", ids)
            save(f"prefix.{step:03d}.position_ids", full_positions)
            save(f"prefix.{step:03d}.attention_mask", mask)
            # Prefill retains every row. Later full-prefix observations retain
            # the selected final row, avoiding repeated large vocabulary copies.
            save(f"prefix.{step:03d}.logits", full.logits[:, -1:, :])
            check = {"step": step, "logits": _comparison(output.logits[:, -1, :],
                      full.logits[:, -1, :], tolerance), "layers": []}
            if not torch.equal(output.logits[:, -1, :].argmax(-1),
                               full.logits[:, -1, :].argmax(-1)):
                raise ValueError("HF cached/full greedy ID mismatch")
            cached_layers, full_layers = _cache_layers(cache), _cache_layers(full.past_key_values)
            if len(cached_layers) != len(full_layers):
                raise ValueError("HF cached/full layer count mismatch")
            for layer, ((key, value), (full_key, full_value)) in enumerate(zip(cached_layers, full_layers)):
                check["layers"].append({"layer": layer, "key": _comparison(key, full_key, tolerance),
                                        "value": _comparison(value, full_value, tolerance)})
            checks.append(check)
            del full
        save("continuation.greedy_ids", torch.tensor([greedy_ids], dtype=torch.int32))
    return tensors, {"greedy_ids": greedy_ids, "cached_full_parity": checks,
                     "prefix_logits_selection": "final sequence row; prefill retains all rows",
                     "kv_layout": "batch, kv_head, sequence, head_dim; key after RoPE"}


def _provenance(policy: dict[str, Any], policy_path: Path = POLICY_PATH,
                *, model: str | None = None) -> dict[str, Any]:
    if read_policy(policy_path, model=model) != policy:
        raise ValueError("reference policy differs from its provenance file")
    import transformers
    reference = policy["reference"]
    if transformers.__version__ != reference["transformers_version"]:
        raise ValueError("Transformers version differs from the fixed reference policy")
    package_root = Path(transformers.__file__).parent.parent
    sources = {}
    for name, expected in reference["transformers_source_sha256"].items():
        actual = sha256_file(package_root / name)
        if actual != expected:
            raise ValueError(f"Transformers source differs from its pinned release: {name}")
        sources[name] = actual
    versions = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy", "huggingface_hub"):
        module = importlib.import_module(name)
        versions[name] = module.__version__
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--",
            "tools/llm_reference.py", "tools/llm_baseline"], cwd=ROOT, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    import torch
    return {"versions": versions, "torch_git_commit": torch.version.git_version,
            "python": sys.version, "platform": platform.platform(),
            "transformers_release_commit": reference["transformers_release_commit"],
            "transformers_source_sha256": sources, "harness_git_commit": commit,
            "harness_git_status": dirty, "harness_sha256": sha256_file(Path(__file__)),
            "policy_sha256": sha256_file(policy_path), "command": sys.argv}


def capture_snapshot(snapshot: Path, output_directory: Path, alias: str,
                     policy: dict[str, Any], *, policy_path: Path = POLICY_PATH) -> dict[str, Any]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError("capture output directory must be new or empty")
    pinned = policy["models"][alias]
    snapshot_hashes = verify_snapshot(snapshot, pinned)
    provenance = _provenance(policy, policy_path, model=alias)
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    reference = policy["reference"]
    torch.manual_seed(reference["seed"])
    torch.set_num_threads(reference["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
        trust_remote_code=False, use_safetensors=True, dtype=torch.float32,
        attn_implementation="eager").cpu().eval()
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32
           for parameter in model.parameters()):
        raise ValueError("reference model is not entirely F32")
    output_directory.mkdir(parents=True, exist_ok=True)
    qualification = reference["qualification"]
    cases = []
    for prompt_index, prompt in enumerate(qualification["prompts"]):
        token_ids = tokenizer.encode(prompt, add_special_tokens=qualification["add_special_tokens"])
        if len(token_ids) < max(qualification["prefix_lengths"]):
            raise ValueError("fixed prompt does not cover every required prefix length")
        for prefix in qualification["prefix_lengths"]:
            if prefix + qualification["decode_steps"] > qualification["max_context"]:
                raise ValueError("capture exceeds the qualified context bound")
            case = f"prompt{prompt_index}-prefix{prefix:03d}"
            tensors, observation = capture_sequence(model, torch.tensor([token_ids[:prefix]]),
                decode_steps=qualification["decode_steps"], tolerance=reference["cached_full_parity"])
            path = output_directory / (case + ".safetensors")
            save_file(tensors, str(path), metadata={"format": "volvox-hf-reference/v1", "case": case})
            tensor_inventory = {}
            for name, tensor in tensors.items():
                raw = tensor.numpy().tobytes(order="C")
                tensor_inventory[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                    "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "byte_order": sys.byteorder}
            # Engine token inputs are I32; HF accepts I64. Record both byte identities.
            engine_ids = tensors["prefill.input_ids"].to(torch.int32).numpy().tobytes(order="C")
            engine_path = output_directory / (case + ".input_ids.i32.bin")
            engine_path.write_bytes(engine_ids)
            cases.append({"id": case, "prompt_index": prompt_index, "prompt": prompt,
                "prompt_utf8_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "input_token_ids": token_ids[:prefix], "prefix_length": prefix,
                "engine_input_ids": {"path": engine_path.name, "dtype": "int32", "shape": [1, prefix],
                    "bytes": len(engine_ids), "sha256": hashlib.sha256(engine_ids).hexdigest(),
                    "byte_order": sys.byteorder}, "reference_path": path.name,
                "reference_sha256": sha256_file(path), "reference_bytes": path.stat().st_size,
                "tensors": tensor_inventory, **observation})
            print(f"captured {alias} {case}", flush=True)
            del tensors
    manifest = {"format": "volvox-hf-reference/v1", "status": "complete", "model": alias,
        "repo_id": pinned["repo_id"], "revision": pinned["revision"], "config": pinned["config"],
        "snapshot_hashes": snapshot_hashes, "source_dtype": pinned["source_dtype"],
        "execution_dtype": "float32", "reference_policy": reference, "provenance": provenance,
        "cases": cases, "capture_enabled": True, "timing": False,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_process_rss_scope": "whole process lifetime, including imports, load and capture; Linux KiB"}
    (output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("metadata", "download", "verify", "capture"))
    parser.add_argument("--policy", type=Path, default=POLICY_PATH,
                        help="pinned checkpoint catalog and numerical reference settings")
    parser.add_argument("--model", required=True, help="model alias in the selected policy")
    parser.add_argument("--snapshot", required=True, type=Path,
                        help="explicit snapshot/cache location; capture and verify are offline")
    parser.add_argument("--output", type=Path, help="new/empty numerical capture directory")
    args = parser.parse_args()
    policy = read_policy(args.policy)
    if args.model not in policy["models"]:
        parser.error("model must be one of: " + ", ".join(policy["models"]))
    policy = read_policy(args.policy, model=args.model)
    if args.operation in ("metadata", "download"):
        download_snapshot(args.snapshot, policy["models"][args.model], metadata_only=args.operation == "metadata")
    elif args.operation == "verify":
        print(json.dumps(verify_snapshot(args.snapshot, policy["models"][args.model]), indent=2))
    else:
        if args.output is None:
            parser.error("capture requires --output")
        result = capture_snapshot(args.snapshot, args.output, args.model, policy,
                                  policy_path=args.policy)
        print(json.dumps({"model": args.model, "cases": len(result["cases"]),
                          "manifest": str(args.output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
