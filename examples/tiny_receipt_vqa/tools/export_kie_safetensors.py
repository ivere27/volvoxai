#!/usr/bin/env python3
"""Export a TinyReceipt/KIE PyTorch checkpoint as a VolvoxAI graph package.

This is the model-specific local-checkpoint path. Generic HuggingFace, ONNX,
and TFLite export remains in the repository-level ``tools/export_safetensors.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import save_file


def quantize_to_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not tensor.is_floating_point():
        return tensor, None
    maximum = tensor.abs().max()
    scale = maximum / 127.0
    if scale == 0:
        return tensor.to(torch.int8), torch.tensor([1.0], dtype=torch.float32)
    return torch.round(tensor / scale).to(torch.int8), scale.view(1).float()


def dumps_with_compact_lists(value: Any, indent: int = 2) -> str:
    indent_unit = " " * indent

    def render(item: Any, level: int = 0) -> str:
        if isinstance(item, dict):
            if not item:
                return "{}"
            entries = list(item.items())
            lines = ["{"]
            for index, (key, child) in enumerate(entries):
                comma = "," if index < len(entries) - 1 else ""
                lines.append(
                    f"{indent_unit * (level + 1)}{json.dumps(key)}: "
                    f"{render(child, level + 1)}{comma}"
                )
            lines.append(f"{indent_unit * level}}}")
            return "\n".join(lines)
        if isinstance(item, (list, tuple)):
            if not item:
                return "[]"
            if all(not isinstance(child, (dict, list, tuple)) for child in item):
                return "[" + ", ".join(json.dumps(child) for child in item) + "]"
            lines = ["["]
            for index, child in enumerate(item):
                comma = "," if index < len(item) - 1 else ""
                lines.append(f"{indent_unit * (level + 1)}{render(child, level + 1)}{comma}")
            lines.append(f"{indent_unit * level}]")
            return "\n".join(lines)
        return json.dumps(item)

    return render(value) + "\n"


def build_kie_graph(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    d_model = config.get("d_model", 320)
    for index in range(4):
        input_name = "images" if index == 0 else f"stem_{index - 1}_out"
        nodes.append({
            "op": "Conv2D",
            "inputs": {"input": input_name, "weight": f"stem.{index}.net.0.weight"},
            "outputs": {"out": f"stem_{index}_out"},
            "outputs_shape": {"out": [1, 32, 112, 112]},
            "params": {"stride": 2 if index < 2 else 1, "padding": 1},
        })

    last_output = "stem_3_out"
    for index in range(config.get("enc_layers", 6)):
        prefix = f"encoder.layers.{index}"
        nodes.append({"op": "LayerNorm", "inputs": {"input": last_output, "weight": f"{prefix}.norm1.weight", "bias": f"{prefix}.norm1.bias"}, "outputs": {"out": f"enc_{index}_n1"}, "outputs_shape": {"out": [1, 196, d_model]}, "params": {"d_model": d_model}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"enc_{index}_n1", "weight": f"{prefix}.self_attn.in_proj_weight", "scale": f"{prefix}.self_attn.in_proj_weight_scale", "bias": f"{prefix}.self_attn.in_proj_bias"}, "outputs": {"out": f"enc_{index}_qkv"}, "outputs_shape": {"out": [1, 196, d_model * 3]}})
        nodes.append({"op": "SDPA", "inputs": {"qkv": f"enc_{index}_qkv"}, "outputs": {"out": f"enc_{index}_attn"}, "outputs_shape": {"out": [1, 196, d_model]}, "params": {"heads": config.get("heads", 8)}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"enc_{index}_attn", "weight": f"{prefix}.self_attn.out_proj.weight", "scale": f"{prefix}.self_attn.out_proj.weight_scale", "bias": f"{prefix}.self_attn.out_proj.bias"}, "outputs": {"out": f"enc_{index}_attn_proj"}, "outputs_shape": {"out": [1, 196, d_model]}})
        nodes.append({"op": "Add", "inputs": {"a": last_output, "b": f"enc_{index}_attn_proj"}, "outputs": {"out": f"enc_{index}_add1"}, "outputs_shape": {"out": [1, 196, d_model]}})
        nodes.append({"op": "LayerNorm", "inputs": {"input": f"enc_{index}_add1", "weight": f"{prefix}.norm2.weight", "bias": f"{prefix}.norm2.bias"}, "outputs": {"out": f"enc_{index}_n2"}, "outputs_shape": {"out": [1, 196, d_model]}, "params": {"d_model": d_model}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"enc_{index}_n2", "weight": f"{prefix}.linear1.weight", "scale": f"{prefix}.linear1.weight_scale", "bias": f"{prefix}.linear1.bias"}, "outputs": {"out": f"enc_{index}_ff1"}, "outputs_shape": {"out": [1, 196, d_model * 4]}})
        nodes.append({"op": "GELU", "inputs": {"input": f"enc_{index}_ff1"}, "outputs": {"out": f"enc_{index}_gelu"}, "outputs_shape": {"out": [1, 196, d_model * 4]}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"enc_{index}_gelu", "weight": f"{prefix}.linear2.weight", "scale": f"{prefix}.linear2.weight_scale", "bias": f"{prefix}.linear2.bias"}, "outputs": {"out": f"enc_{index}_ff2"}, "outputs_shape": {"out": [1, 196, d_model]}})
        nodes.append({"op": "Add", "inputs": {"a": f"enc_{index}_add1", "b": f"enc_{index}_ff2"}, "outputs": {"out": f"enc_{index}_out"}, "outputs_shape": {"out": [1, 196, d_model]}})
        last_output = f"enc_{index}_out"

    nodes.append({"op": "Embedding", "inputs": {"input": "q_tokens", "weight": "q_pos"}, "outputs": {"out": "dec_in"}, "outputs_shape": {"out": [1, 192, d_model]}})
    decoder_output = "dec_in"
    for index in range(config.get("dec_layers", 4)):
        prefix = f"decoder.layers.{index}"
        nodes.append({"op": "LayerNorm", "inputs": {"input": decoder_output, "weight": f"{prefix}.norm1.weight", "bias": f"{prefix}.norm1.bias"}, "outputs": {"out": f"dec_{index}_n1"}, "outputs_shape": {"out": [1, 192, d_model]}, "params": {"d_model": d_model}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"dec_{index}_n1", "weight": f"{prefix}.self_attn.in_proj_weight", "scale": f"{prefix}.self_attn.in_proj_weight_scale", "bias": f"{prefix}.self_attn.in_proj_bias"}, "outputs": {"out": f"dec_{index}_qkv"}, "outputs_shape": {"out": [1, 192, d_model * 3]}})
        nodes.append({"op": "SDPA", "inputs": {"qkv": f"dec_{index}_qkv"}, "outputs": {"out": f"dec_{index}_attn"}, "outputs_shape": {"out": [1, 192, d_model]}, "params": {"heads": config.get("heads", 8)}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"dec_{index}_attn", "weight": f"{prefix}.self_attn.out_proj.weight", "scale": f"{prefix}.self_attn.out_proj.weight_scale", "bias": f"{prefix}.self_attn.out_proj.bias"}, "outputs": {"out": f"dec_{index}_attn_proj"}, "outputs_shape": {"out": [1, 192, d_model]}})
        nodes.append({"op": "Add", "inputs": {"a": decoder_output, "b": f"dec_{index}_attn_proj"}, "outputs": {"out": f"dec_{index}_add1"}, "outputs_shape": {"out": [1, 192, d_model]}})
        nodes.append({"op": "LayerNorm", "inputs": {"input": f"dec_{index}_add1", "weight": f"{prefix}.norm2.weight", "bias": f"{prefix}.norm2.bias"}, "outputs": {"out": f"dec_{index}_n2"}, "outputs_shape": {"out": [1, 192, d_model]}, "params": {"d_model": d_model}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"dec_{index}_n2", "weight": f"{prefix}.multihead_attn.q_proj_weight", "scale": f"{prefix}.multihead_attn.q_proj_scale", "bias": f"{prefix}.multihead_attn.q_proj_bias"}, "outputs": {"out": f"dec_{index}_q"}, "outputs_shape": {"out": [1, 192, d_model]}})
        nodes.append({"op": "MatMul", "inputs": {"input": last_output, "weight": f"{prefix}.multihead_attn.k_proj_weight", "scale": f"{prefix}.multihead_attn.k_proj_scale", "bias": f"{prefix}.multihead_attn.k_proj_bias"}, "outputs": {"out": f"dec_{index}_k"}, "outputs_shape": {"out": [1, 196, d_model]}})
        nodes.append({"op": "MatMul", "inputs": {"input": last_output, "weight": f"{prefix}.multihead_attn.v_proj_weight", "scale": f"{prefix}.multihead_attn.v_proj_scale", "bias": f"{prefix}.multihead_attn.v_proj_bias"}, "outputs": {"out": f"dec_{index}_v"}, "outputs_shape": {"out": [1, 196, d_model]}})
        nodes.append({"op": "CrossSDPA", "inputs": {"q": f"dec_{index}_q", "k": f"dec_{index}_k", "v": f"dec_{index}_v"}, "outputs": {"out": f"dec_{index}_cross"}, "outputs_shape": {"out": [1, 192, d_model]}, "params": {"heads": config.get("heads", 8)}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"dec_{index}_cross", "weight": f"{prefix}.multihead_attn.out_proj.weight", "scale": f"{prefix}.multihead_attn.out_proj.weight_scale", "bias": f"{prefix}.multihead_attn.out_proj.bias"}, "outputs": {"out": f"dec_{index}_cross_proj"}, "outputs_shape": {"out": [1, 192, d_model]}})
        nodes.append({"op": "Add", "inputs": {"a": f"dec_{index}_add1", "b": f"dec_{index}_cross_proj"}, "outputs": {"out": f"dec_{index}_add2"}, "outputs_shape": {"out": [1, 192, d_model]}})
        nodes.append({"op": "LayerNorm", "inputs": {"input": f"dec_{index}_add2", "weight": f"{prefix}.norm3.weight", "bias": f"{prefix}.norm3.bias"}, "outputs": {"out": f"dec_{index}_n3"}, "outputs_shape": {"out": [1, 192, d_model]}, "params": {"d_model": d_model}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"dec_{index}_n3", "weight": f"{prefix}.linear1.weight", "scale": f"{prefix}.linear1.weight_scale", "bias": f"{prefix}.linear1.bias"}, "outputs": {"out": f"dec_{index}_ff1"}, "outputs_shape": {"out": [1, 192, d_model * 4]}})
        nodes.append({"op": "GELU", "inputs": {"input": f"dec_{index}_ff1"}, "outputs": {"out": f"dec_{index}_gelu"}, "outputs_shape": {"out": [1, 192, d_model * 4]}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"dec_{index}_gelu", "weight": f"{prefix}.linear2.weight", "scale": f"{prefix}.linear2.weight_scale", "bias": f"{prefix}.linear2.bias"}, "outputs": {"out": f"dec_{index}_ff2"}, "outputs_shape": {"out": [1, 192, d_model]}})
        nodes.append({"op": "Add", "inputs": {"a": f"dec_{index}_add2", "b": f"dec_{index}_ff2"}, "outputs": {"out": f"dec_{index}_out"}, "outputs_shape": {"out": [1, 192, d_model]}})
        decoder_output = f"dec_{index}_out"

    nodes.append({"op": "LayerNorm", "inputs": {"input": decoder_output, "weight": "encoder.layers.0.norm1.weight", "bias": "encoder.layers.0.norm1.bias"}, "outputs": {"out": "final_norm"}, "outputs_shape": {"out": [1, 192, d_model]}, "params": {"d_model": d_model}})
    nodes.append({"op": "MatMul", "inputs": {"input": "final_norm", "weight": "encoder.layers.0.self_attn.out_proj.weight", "scale": "encoder.layers.0.self_attn.out_proj.weight_scale", "bias": "out_bias"}, "outputs": {"out": "logits"}, "outputs_shape": {"out": [1, 192, config.get("vocab_size", 530)]}})
    return nodes


def export_checkpoint(checkpoint_path: Path, output_path: Path) -> None:
    print(
        "[Export] Loading TinyReceipt/KIE PyTorch checkpoint. "
        "Using Vision-Encoder-Decoder Graph Builder."
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    config = checkpoint.get("config", {})
    output_tensors: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        if "multihead_attn.in_proj_weight" in key:
            q_weight, k_weight, v_weight = tensor.chunk(3, dim=0)
            prefix = key.replace(".in_proj_weight", "")
            for name, weight in zip(
                ("q_proj_weight", "k_proj_weight", "v_proj_weight"),
                (q_weight, k_weight, v_weight),
            ):
                quantized, scale = quantize_to_int8(weight)
                output_tensors[f"{prefix}.{name}"] = quantized
                output_tensors[f"{prefix}.{name.replace('weight', 'scale')}"] = scale.view(1)
        elif "multihead_attn.in_proj_bias" in key:
            q_bias, k_bias, v_bias = tensor.chunk(3, dim=0)
            prefix = key.replace(".in_proj_bias", "")
            output_tensors[f"{prefix}.q_proj_bias"] = q_bias.float()
            output_tensors[f"{prefix}.k_proj_bias"] = k_bias.float()
            output_tensors[f"{prefix}.v_proj_bias"] = v_bias.float()
        elif tensor.dim() == 2 and "weight" in key and "norm" not in key:
            quantized, scale = quantize_to_int8(tensor)
            output_tensors[key] = quantized
            output_tensors[f"{key}_scale"] = scale.view(1)
        else:
            output_tensors[key] = tensor.float()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Export] Saving standard Safetensors to {output_path}...")
    save_file(
        {name: tensor.clone().contiguous() for name, tensor in output_tensors.items()},
        str(output_path),
    )

    graph = {
        "inputs": {
            "images": {"shape": [1, 3, 224, 224], "dtype": "float32"},
            "q_tokens": {"shape": [1, 192], "dtype": "int32"},
        },
        "nodes": build_kie_graph(config),
        "outputs": {"logits": "logits"},
    }
    config_path = output_path.parent / "config.json"
    print(f"[Export] Saving graph topology to {config_path}...")
    config_path.write_text(dumps_with_compact_lists(graph, indent=2), encoding="utf-8")
    print(
        f"[Export] Success! Safetensors size: "
        f"{output_path.stat().st_size / (1024 * 1024):.2f} MB"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a local TinyReceipt/KIE PyTorch checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        "--model",
        dest="checkpoint",
        required=True,
        type=Path,
        help="local TinyReceipt/KIE PyTorch checkpoint",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="output SafeTensors path (config.json is written beside it)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    export_checkpoint(args.checkpoint, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
