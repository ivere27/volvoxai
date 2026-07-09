#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


TASK_ADAPTER_FAMILIES = [
    "phone",
    "address",
    "store",
    "item_row",
    "item_math",
    "item_lookup",
    "math",
    "other",
]


def dumps_with_compact_lists(obj, indent=2):
    unit = " " * indent

    def render(value, level=0):
        if isinstance(value, dict):
            if not value:
                return "{}"
            items = list(value.items())
            lines = ["{"]
            for i, (k, v) in enumerate(items):
                comma = "," if i + 1 < len(items) else ""
                lines.append(f"{unit * (level + 1)}{json.dumps(k)}: {render(v, level + 1)}{comma}")
            lines.append(f"{unit * level}}}")
            return "\n".join(lines)
        if isinstance(value, list):
            if not value:
                return "[]"
            if all(not isinstance(x, (dict, list)) for x in value):
                return "[" + ", ".join(json.dumps(x, ensure_ascii=False) for x in value) + "]"
            lines = ["["]
            for i, item in enumerate(value):
                comma = "," if i + 1 < len(value) else ""
                lines.append(f"{unit * (level + 1)}{render(item, level + 1)}{comma}")
            lines.append(f"{unit * level}]")
            return "\n".join(lines)
        return json.dumps(value, ensure_ascii=False)

    return render(obj) + "\n"


_DROP = object()


def sanitize_metadata_value(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, child in value.items():
            if key in {"args", "examples", "source_checkpoint"}:
                continue
            child = sanitize_metadata_value(child)
            if child is not _DROP:
                cleaned[key] = child
        return cleaned
    if isinstance(value, list):
        return _DROP
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _DROP


def torch_load_dict(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"unsupported checkpoint/component file: {path}")
    return data


def component_state(data: dict) -> dict:
    if "state" in data and isinstance(data["state"], dict):
        return data["state"]
    return data


def adapter_family_id(name: str) -> int:
    name = name.strip()
    if name not in TASK_ADAPTER_FAMILIES:
        raise ValueError(f"unknown adapter family: {name}")
    return TASK_ADAPTER_FAMILIES.index(name)


def load_extra_components(state: dict, lora_paths: list[Path], adapter_specs: list[str]) -> list[str]:
    report = []
    for path in lora_paths:
        data = torch_load_dict(path)
        if data.get("kind") not in (None, "lora", "raw_state"):
            raise ValueError(f"not a LoRA component: {path}")
        loaded = 0
        for key, value in component_state(data).items():
            state[key] = value
            loaded += 1
        report.append(f"lora:{loaded}")
    for spec in adapter_specs:
        if "=" not in spec:
            raise ValueError(f"adapter spec must be FAMILY=PATH, got: {spec}")
        family, raw_path = spec.split("=", 1)
        family = family.strip()
        adapter_id = adapter_family_id(family)
        data = torch_load_dict(Path(raw_path))
        if data.get("kind") not in (None, "task_adapter", "raw_state"):
            raise ValueError(f"not an adapter component: {raw_path}")
        loaded = 0
        for key, value in component_state(data).items():
            if key.startswith("memory."):
                target = f"memory_adapters.{adapter_id}." + key[len("memory."):]
            elif key.startswith("decoder."):
                target = f"decoder_adapters.{adapter_id}." + key[len("decoder."):]
            else:
                target = key
            state[target] = value
            loaded += 1
        report.append(f"adapter:{family}:{loaded}")
    return report


def lora_scaling(cfg: dict, lora_paths: list[Path], default_rank: int) -> float:
    config = dict(cfg)
    for path in lora_paths:
        data = torch_load_dict(path)
        if isinstance(data.get("config"), dict):
            for key, value in data["config"].items():
                config.setdefault(key, value)
    rank = int(config.get("lora_r", default_rank) or default_rank or 1)
    alpha = float(config.get("lora_alpha", 16.0))
    return alpha / float(rank)


def fold_lora_tensors(state: dict, cfg: dict, lora_paths: list[Path]) -> tuple[dict, int]:
    merged = {}
    folded = 0
    skip = set()
    prefixes = sorted({
        key[:-len(".lora_a.weight")]
        for key in state
        if key.endswith(".lora_a.weight")
    })
    for prefix in prefixes:
        a_key = prefix + ".lora_a.weight"
        b_key = prefix + ".lora_b.weight"
        if b_key not in state:
            continue
        a = state[a_key].detach().cpu().float()
        b = state[b_key].detach().cpu().float()
        base_key = prefix + ".base.weight"
        plain_key = prefix + ".weight"
        if base_key in state:
            base = state[base_key].detach().cpu().float()
        elif plain_key in state:
            base = state[plain_key].detach().cpu().float()
        else:
            continue
        scale = lora_scaling(cfg, lora_paths, int(a.shape[0]))
        merged[plain_key] = (base + torch.matmul(b, a) * scale).contiguous()
        base_bias = prefix + ".base.bias"
        plain_bias = prefix + ".bias"
        if base_bias in state:
            merged[plain_bias] = state[base_bias].detach().cpu().float().contiguous()
            skip.add(base_bias)
        skip.update({a_key, b_key, base_key, plain_key})
        folded += 1

    tensors = {}
    for key, value in state.items():
        if key in skip or ".lora_" in key:
            continue
        if key.endswith(".base.weight"):
            key = key[:-len(".base.weight")] + ".weight"
        elif key.endswith(".base.bias"):
            key = key[:-len(".base.bias")] + ".bias"
        tensors[key] = value.detach().cpu().float().contiguous().clone()
    tensors.update(merged)
    return tensors, folded


def export(checkpoint: Path, out_dir: Path, lora_paths: list[Path], adapter_specs: list[str]) -> None:
    ckpt = torch_load_dict(checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_state = dict(ckpt["model"])
    components = load_extra_components(raw_state, lora_paths, adapter_specs)
    cfg = dict(ckpt["config"])
    tensors, folded_lora = fold_lora_tensors(raw_state, cfg, lora_paths)
    tensors = {
        name: tensor.detach().cpu().float().contiguous().clone()
        for name, tensor in tensors.items()
    }
    save_file(tensors, out_dir / "model.safetensors")

    if folded_lora:
        cfg["lora_r"] = 0
        cfg["lora_dropout"] = 0.0
        cfg["lora_targets"] = "merged"
    vocab = ckpt["vocab"]
    config = {
        "format": "volvoxai-tiny-receipt-kie-v1",
        "model_type": "tiny_receipt_kie",
        "interface": {
            "type": "tiny_receipt_kie",
            "command": "chat",
            "image": {
                "target": "image",
                "shape": [1, 1, 320, 672],
                "normalize": "minus-one-one",
                "formats": ["png", "jpeg"]
            },
            "prompt": {
                "tokenizer": "char",
                "add_eos": True,
                "max_len": cfg.get("max_q_len", 192)
            },
            "decoder": {
                "type": "greedy",
                "max_new": cfg.get("max_out_len", 192),
                "bos": 1,
                "eos": 2,
                "pad": 0
            }
        },
        "model_config": cfg,
        "vocab": vocab,
        "metadata": {
            "metrics": sanitize_metadata_value(ckpt.get("metrics", {})),
            "params": int(sum(t.numel() for t in tensors.values())),
            "folded_lora_modules": folded_lora,
            "loaded_components": components,
        }
    }
    (out_dir / "config.json").write_text(dumps_with_compact_lists(config), encoding="utf-8")
    print(json.dumps({
        "out": str(out_dir),
        "weights": str(out_dir / "model.safetensors"),
        "config": str(out_dir / "config.json"),
        "params": config["metadata"]["params"],
        "vocab": len(vocab.get("itos", [])),
        "format": config["format"]
    }, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", action="append", default=[],
                    help="optional LoRA component to fold into exported weights; repeatable")
    ap.add_argument("--adapter", action="append", default=[],
                    help="optional adapter component, FAMILY=PATH; repeatable")
    args = ap.parse_args()
    export(Path(args.checkpoint), Path(args.out), [Path(p) for p in args.lora], args.adapter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
