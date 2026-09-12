#!/usr/bin/env python3
"""
Cut one node out of an exported model into a graph the engine still accepts.

Hand-written quantized graphs kept being refused -- a missing parameter, a port
spelled differently, a relation between scales that is not obvious from the
shape contract. A node taken from a model the engine already runs is valid by
construction, which makes it the right shape for a device test of an operator
whose graph is hard to author.

The node's own inputs become graph inputs, its output becomes the graph
output, and only the weights and affine descriptors it touches are copied.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

DTYPE_BYTES = {"F32": 4, "I32": 4, "I8": 1, "U8": 1, "F16": 2}
GRAPH_DTYPE = {"F32": "float32", "I32": "int32", "I8": "int8", "U8": "uint8"}


def read_safetensors(path: Path) -> tuple[dict, memoryview]:
    raw = path.read_bytes()
    (header_length,) = struct.unpack_from("<Q", raw, 0)
    header = json.loads(raw[8 : 8 + header_length])
    header.pop("__metadata__", None)
    return header, memoryview(raw)[8 + header_length :]


def write_safetensors(path: Path, entries: dict) -> None:
    header = {}
    blocks = []
    offset = 0
    for name, (dtype, shape, data) in entries.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blocks.append(data)
        offset += len(data)
    text = json.dumps(header).encode()
    padding = (-len(text)) % 8
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(text) + padding))
        handle.write(text)
        handle.write(b" " * padding)
        for block in blocks:
            handle.write(block)


def declared_shapes(graph: dict) -> dict:
    """Every tensor the graph declares, by name."""
    shapes = {}
    for name, descriptor in graph.get("inputs", {}).items():
        shapes[name] = (descriptor["shape"], descriptor["dtype"])
    for node in graph["nodes"]:
        for output in node["outputs"].values():
            shapes[output["tensor"]] = (output["shape"], output["dtype"])
    return shapes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="directory holding graph.json")
    parser.add_argument("--op", required=True, help="operator to extract")
    parser.add_argument("--out", required=True, help="directory to write into")
    parser.add_argument("--name", default=None, help="basename, default the operator")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="only consider nodes whose parameter KEY prints as VALUE; "
        "repeatable. A grouped convolution and a dense one take different "
        "paths through the same shader, so each needs its own case.",
    )
    options = parser.parse_args()

    model = Path(options.model)
    graph = json.loads((model / "graph.json").read_text())
    header, blob = read_safetensors(model / "model.safetensors")
    shapes = declared_shapes(graph)
    affine = graph.get("quantization", {}).get("tensors", {})

    wanted = dict(clause.split("=", 1) for clause in options.param)

    def matches(node) -> bool:
        params = node.get("params", {})
        return all(str(params.get(key)) == value for key, value in wanted.items())

    candidates = [
        node
        for node in graph["nodes"]
        if node["opType"] == options.op and matches(node)
    ]
    if not candidates:
        raise SystemExit(f"{options.op} does not appear in {model}")

    def output_size(node):
        total = 1
        for extent in node["outputs"]["out"]["shape"]:
            total *= extent if isinstance(extent, int) else 1
        return total

    node = min(candidates, key=output_size)

    inputs = {}
    weights = {}
    quantization = {}

    def take_weight(name: str) -> None:
        entry = header.get(name)
        if entry is None:
            raise SystemExit(f"{name} is not in the model's weights")
        start, end = entry["data_offsets"]
        weights[name] = (entry["dtype"], entry["shape"], bytes(blob[start:end]))

    def take_affine(name: str) -> None:
        descriptor = affine.get(name)
        if descriptor is None:
            return
        quantization[name] = descriptor
        take_weight(descriptor["scale_tensor"])
        take_weight(descriptor["zero_point_tensor"])

    for port, name in node["inputs"].items():
        if name in header:
            take_weight(name)
        else:
            shape, dtype = shapes[name]
            inputs[name] = {"shape": shape, "dtype": dtype}
        take_affine(name)
    take_affine(node["outputs"]["out"]["tensor"])

    minimal = {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": inputs,
        "outputs": [node["outputs"]["out"]["tensor"]],
        "nodes": [node],
    }
    if quantization:
        minimal["quantization"] = {
            "format": "volvox-affine-safetensors/v1",
            "tensors": quantization,
        }

    out = Path(options.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = options.name or options.op
    (out / f"{stem}.graph.json").write_text(json.dumps(minimal, indent=1))
    write_safetensors(out / f"{stem}.safetensors", weights)

    plan = {
        name: {
            "shape": descriptor["shape"],
            "dtype": descriptor["dtype"],
            "elements": _elements(descriptor["shape"]),
        }
        for name, descriptor in inputs.items()
    }
    (out / f"{stem}.inputs.json").write_text(json.dumps(plan, indent=1))
    print(
        f"{options.op}: node {node['id']} -> {out}/{stem}.graph.json "
        f"({len(inputs)} inputs, {len(weights)} weights)"
    )
    return 0


def _elements(shape) -> int:
    total = 1
    for extent in shape:
        total *= extent if isinstance(extent, int) else 1
    return total


if __name__ == "__main__":
    raise SystemExit(main())
