#!/usr/bin/env python3
"""Export a GPT-Neo checkpoint as the TinyStories VolvoxAI example package.

The exporter owns the family-specific checkpoint names, graph topology, bounded
sequence capacity, and prefill inputs. Generic ONNX/TFLite lowering remains in the
repository-level ``tools/export_safetensors.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


MAX_SEQUENCE_LENGTH = 256
SEQUENCE_SYMBOL = "S"
DEFAULT_EOS_TOKEN_ID = 50256


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


def build_gptneo_graph(
    config: Any,
    state_dict: Mapping[str, torch.Tensor],
    output_tensors: MutableMapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    d_model = config.hidden_size
    hidden_shape = [1, SEQUENCE_SYMBOL, d_model]
    activation_function = getattr(config, "activation_function", None)
    if activation_function == "gelu_new":
        gelu_params = {"approximate": "tanh"}
    elif activation_function == "gelu":
        gelu_params = {"approximate": "none"}
    else:
        raise ValueError(
            "TinyStories GPT-Neo export supports only activation_function "
            f"'gelu_new' or 'gelu', got {activation_function!r}"
        )

    def add_node(
        op: str,
        inputs: Mapping[str, str],
        output: str,
        shape: Sequence[int | str],
        params: Mapping[str, Any] | None = None,
    ) -> None:
        node: dict[str, Any] = {
            "id": f"node_{len(nodes)}",
            "opType": op,
            "inputs": dict(inputs),
            "outputs": {
                "out": {
                    "tensor": output,
                    "shape": list(shape),
                    "dtype": "float32",
                }
            },
            "params": dict(params or {}),
        }
        nodes.append(node)

    for name, tensor in state_dict.items():
        output_tensors[name.removeprefix("transformer.")] = tensor.float().contiguous()

    add_node("Embedding", {"input": "tokens", "weight": "wte.weight"}, "emb_tok", hidden_shape)
    add_node(
        "Embedding",
        {"input": "positions", "weight": "wpe.weight"},
        "emb_pos",
        hidden_shape,
    )
    add_node("Add", {"a": "emb_tok", "b": "emb_pos"}, "hidden_0", hidden_shape)

    last_hidden = "hidden_0"
    for index in range(config.num_layers):
        prefix = f"h.{index}"
        add_node(
            "LayerNorm",
            {
                "input": last_hidden,
                "weight": f"{prefix}.ln_1.weight",
                "bias": f"{prefix}.ln_1.bias",
            },
            f"ln1_{index}",
            hidden_shape,
            {"eps": config.layer_norm_epsilon, "d_model": d_model},
        )

        q_weight = output_tensors.pop(f"{prefix}.attn.attention.q_proj.weight").t()
        k_weight = output_tensors.pop(f"{prefix}.attn.attention.k_proj.weight").t()
        v_weight = output_tensors.pop(f"{prefix}.attn.attention.v_proj.weight").t()
        output_tensors[f"{prefix}.attn.qkv_proj.weight"] = torch.cat(
            [q_weight, k_weight, v_weight], dim=1
        ).contiguous()

        add_node(
            "MatMul",
            {
                "input": f"ln1_{index}",
                "weight": f"{prefix}.attn.qkv_proj.weight",
            },
            f"qkv_{index}",
            [1, SEQUENCE_SYMBOL, d_model * 3],
            {"weight_layout": "din_dout"},
        )
        add_node(
            "SDPA",
            {"qkv": f"qkv_{index}"},
            f"attn_{index}",
            hidden_shape,
            {
                "heads": config.num_heads,
                # GPT-Neo's source attention computes QK^T without the usual
                # inverse-square-root head scaling. Keep the explicit value so
                # the runtime SDPA default cannot change checkpoint semantics.
                "scale": 1.0,
                "causal": True,
            },
        )

        output_tensors[f"{prefix}.attn.out_proj.weight"] = output_tensors.pop(
            f"{prefix}.attn.attention.out_proj.weight"
        ).t().contiguous()
        output_tensors[f"{prefix}.attn.out_proj.bias"] = output_tensors.pop(
            f"{prefix}.attn.attention.out_proj.bias"
        ).contiguous()

        add_node(
            "MatMul",
            {
                "input": f"attn_{index}",
                "weight": f"{prefix}.attn.out_proj.weight",
                "bias": f"{prefix}.attn.out_proj.bias",
            },
            f"attn_proj_{index}",
            hidden_shape,
            {"weight_layout": "din_dout"},
        )
        add_node(
            "Add",
            {"a": last_hidden, "b": f"attn_proj_{index}"},
            f"add1_{index}",
            hidden_shape,
        )
        add_node(
            "LayerNorm",
            {
                "input": f"add1_{index}",
                "weight": f"{prefix}.ln_2.weight",
                "bias": f"{prefix}.ln_2.bias",
            },
            f"ln2_{index}",
            hidden_shape,
            {"eps": config.layer_norm_epsilon, "d_model": d_model},
        )

        output_tensors[f"{prefix}.mlp.c_fc.weight"] = output_tensors.pop(
            f"{prefix}.mlp.c_fc.weight"
        ).t().contiguous()
        output_tensors[f"{prefix}.mlp.c_fc.bias"] = output_tensors.pop(
            f"{prefix}.mlp.c_fc.bias"
        ).contiguous()
        feed_forward_shape = [1, SEQUENCE_SYMBOL, d_model * 4]
        add_node(
            "MatMul",
            {
                "input": f"ln2_{index}",
                "weight": f"{prefix}.mlp.c_fc.weight",
                "bias": f"{prefix}.mlp.c_fc.bias",
            },
            f"mlp1_{index}",
            feed_forward_shape,
            {"weight_layout": "din_dout"},
        )
        add_node(
            "GELU",
            {"input": f"mlp1_{index}"},
            f"mlp_act_{index}",
            feed_forward_shape,
            gelu_params,
        )

        output_tensors[f"{prefix}.mlp.c_proj.weight"] = output_tensors.pop(
            f"{prefix}.mlp.c_proj.weight"
        ).t().contiguous()
        output_tensors[f"{prefix}.mlp.c_proj.bias"] = output_tensors.pop(
            f"{prefix}.mlp.c_proj.bias"
        ).contiguous()
        add_node(
            "MatMul",
            {
                "input": f"mlp_act_{index}",
                "weight": f"{prefix}.mlp.c_proj.weight",
                "bias": f"{prefix}.mlp.c_proj.bias",
            },
            f"mlp2_{index}",
            hidden_shape,
            {"weight_layout": "din_dout"},
        )
        add_node(
            "Add",
            {"a": f"add1_{index}", "b": f"mlp2_{index}"},
            f"hidden_{index + 1}",
            hidden_shape,
        )
        last_hidden = f"hidden_{index + 1}"

    add_node(
        "LayerNorm",
        {"input": last_hidden, "weight": "ln_f.weight", "bias": "ln_f.bias"},
        "final_norm",
        hidden_shape,
        {"eps": config.layer_norm_epsilon, "d_model": d_model},
    )
    if "lm_head.weight" in output_tensors:
        output_tensors["lm_head.weight"] = output_tensors["lm_head.weight"].t().contiguous()
    else:
        output_tensors["lm_head.weight"] = output_tensors["wte.weight"].t().contiguous()
    add_node(
        "MatMul",
        {"input": "final_norm", "weight": "lm_head.weight"},
        "logits",
        [1, SEQUENCE_SYMBOL, config.vocab_size],
        {"weight_layout": "din_dout"},
    )
    return nodes


def export_model(model_id_or_path: str, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Export] Loading GPT-Neo checkpoint {model_id_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_id_or_path)
    config = model.config
    output_tensors: dict[str, torch.Tensor] = {}
    nodes = build_gptneo_graph(config, model.state_dict(), output_tensors)

    eos_token_id = getattr(config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = DEFAULT_EOS_TOKEN_ID
    token_input = np.full((1, MAX_SEQUENCE_LENGTH), eos_token_id, dtype=np.int32)
    token_input[0, :5] = [123, 456, 789, 1011, 1213]
    position_input = np.arange(MAX_SEQUENCE_LENGTH, dtype=np.int32).reshape(
        1, MAX_SEQUENCE_LENGTH
    )
    (output_path.parent / "tokens.i32").write_bytes(token_input.tobytes())
    (output_path.parent / "positions.i32").write_bytes(position_input.tobytes())

    final_tensors = {
        name: tensor.clone().contiguous() for name, tensor in output_tensors.items()
    }
    save_file(final_tensors, str(output_path))

    graph = {
        "format": "volvox-graph/v1",
        "dimensions": {
            SEQUENCE_SYMBOL: {"min": 1, "max": MAX_SEQUENCE_LENGTH},
        },
        "inputs": {
            "tokens": {"shape": [1, SEQUENCE_SYMBOL], "dtype": "int32"},
            "positions": {"shape": [1, SEQUENCE_SYMBOL], "dtype": "int32"},
        },
        "nodes": nodes,
        "outputs": ["logits"],
    }
    graph_path = output_path.parent / "graph.json"
    graph_path.write_text(dumps_with_compact_lists(graph), encoding="utf-8")
    print(f"[Export] Wrote {output_path} and {graph_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a GPT-Neo checkpoint as the TinyStories example package."
    )
    parser.add_argument(
        "--model", required=True, help="Hugging Face model ID or directory"
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Output model.safetensors path"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    export_model(args.model, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
