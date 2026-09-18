"""Model discovery shared by the Python workflow adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelPackage:
    """Published graph and ordered weight paths; accepted by session adapters."""

    path: Path
    graph_path: Path
    weight_paths: tuple[Path, ...]

    def __fspath__(self) -> str:
        return os.fspath(self.graph_path)


def _graphs(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and (path.name == "graph.json"
                                         or path.name.endswith(".graph.json")))


def _one_graph(candidates: list[Path], location: Path) -> Path:
    if not candidates:
        raise FileNotFoundError(
            f"No VolvoxAI graph found in {location}. Pass a graph.json or model "
            "directory; convert ONNX with 'volvoxai export' first.")
    if len(candidates) != 1:
        names = ", ".join(str(path) for path in candidates)
        raise ValueError(f"Multiple models found: {names}. Pass the model path explicitly.")
    return candidates[0]


def _model_paths(model, weights) -> tuple[Path, tuple[Path, ...]]:
    if isinstance(model, ModelPackage) and weights is None:
        weights = model.weight_paths
    location = Path.cwd() if model is None else Path(model).expanduser().resolve()
    if location.is_dir():
        candidates = _graphs(location)
        # One level finds a single exported package without a recursive search
        # through model caches or the application's entire directory tree.
        for child in sorted(location.iterdir()):
            if child.is_dir() and not child.name.startswith(".") and not child.is_symlink():
                candidates.extend(_graphs(child))
        graph = _one_graph(candidates, location)
    elif location.is_file():
        if location.suffix == ".safetensors":
            if weights is not None:
                raise ValueError("Pass either a SafeTensors model path or weights=, not both.")
            weights = [location]
            paired = location.with_suffix(".graph.json")
            graph = paired if paired.is_file() else _one_graph(_graphs(location.parent), location.parent)
        elif location.suffix == ".json":
            graph = location
        else:
            raise ValueError("Pass a VolvoxAI model directory, graph JSON or SafeTensors file. "
                             "Convert ONNX with 'volvoxai export' first.")
    else:
        raise FileNotFoundError(f"Model path does not exist: {location}")

    if weights is None:
        paired = graph.with_name(graph.name.removesuffix(".graph.json") + ".safetensors")
        if graph.name != "graph.json" and paired.is_file():
            selected = [paired]
        else:
            selected = sorted(graph.parent.glob("*.safetensors"))
            if len(selected) > 1:
                raise ValueError(
                    f"Multiple SafeTensors files found beside {graph}. "
                    "Pass weights=[...] to select the model's weight files or shards.")
    else:
        if isinstance(weights, (str, os.PathLike)):
            raise TypeError("weights must be a sequence of paths, for example weights=['model.safetensors']")
        selected = [Path(path).expanduser().resolve() for path in weights]
    for path in selected:
        if not path.is_file():
            raise FileNotFoundError(f"Weight file does not exist: {path}")
    return graph, tuple(selected)
