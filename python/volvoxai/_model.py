"""Model discovery shared by the Python workflow adapters."""

from __future__ import annotations

import os
from contextlib import contextmanager
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
        selected = weights
    return graph, _weight_paths(selected)


def _weight_paths(weights) -> tuple[Path, ...]:
    if isinstance(weights, (str, os.PathLike)):
        raise TypeError("weights must be a sequence of paths, for example weights=['model.safetensors']")
    selected = tuple(Path(path).expanduser().resolve() for path in (() if weights is None else weights))
    for path in selected:
        if not path.is_file():
            raise FileNotFoundError(f"Weight file does not exist: {path}")
    return selected


@contextmanager
def _hf_model_source(directory, *, max_context, cached):
    """Prepare a local HF graph while keeping original weight files unchanged."""
    import json
    import tempfile
    from safetensors.numpy import save_file
    from ._cli import _export_module

    if not isinstance(cached, bool):
        raise TypeError("cached must be a boolean.")
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Local Hugging Face directory does not exist: {directory}")
    exporter = _export_module()
    decoder = exporter._exporter_module("hf_decoder")
    storage = exporter._exporter_module("hf_graph")
    error_type = exporter._exporter_module("errors").ExporterError
    try:
        source = storage.read_config(directory / "config.json")
        config = decoder.DecoderConfig.from_hf(source, max_context=max_context)
        if config.table_rope and max_context is None:
            raise ValueError("Scaled RoPE requires an explicit max_context for its bounded position tables.")
        _, inventory = storage.read_checkpoint_metadata(directory, source_config=source)
        expected = config.weight_shapes()
        optional = {"lm_head.weight"} if config.tied_embeddings else set()
        missing, extra = set(expected) - set(inventory), set(inventory) - set(expected) - optional
        if missing or extra:
            raise ValueError(f"HF weight inventory mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}")
        for name, (_, dtype, shape, _) in inventory.items():
            wanted = expected.get(name, expected["model.embed_tokens.weight"])
            if dtype not in ("F32", "F16", "BF16") or shape != wanted:
                raise ValueError(f"HF tensor {name} requires floating shape {wanted}; received {dtype} {shape}")
        if config.tied_embeddings and "lm_head.weight" in inventory:
            decoder.check_tied_checkpoint_weights(inventory)
        document, derived = decoder.build_decoder_structure(config, compact_kv=True, cached=cached)
    except error_type as error:
        raise ValueError(str(error)) from error
    paths = tuple(sorted({descriptor[0] for descriptor in inventory.values()}))
    encoded = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    if not derived:
        yield encoded, paths
    else:
        with tempfile.TemporaryDirectory(prefix="volvoxai-hf-derived-") as temporary:
            path = Path(temporary) / "derived.safetensors"
            save_file(derived, path)
            yield encoded, (*paths, path)
