"""Validate and refresh model-neutral multi-graph package manifests.

This module knows only the common split-package storage contract: each named
graph owns a RuntimeIR document, a safetensors store, explicit public input and
output mappings, and optional exporter evidence. Model-family tokenizers,
routing policy, calibration coverage, and task qualification belong to the
application that authors the manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import stat
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path, PurePosixPath
from typing import Any, Callable

if __package__ in {None, ""}:  # Support direct validation-tool imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "exporter"

from .optimizer.safetensors_io import read_safetensors
from .quantization_storage import validate_external_quantization
from .runtime_ir import (
    import_runtime_package,
    load_runtime_document,
    prove_dynamic_quantized_runtime_domain,
)


_MAX_JSON_SAFE_INTEGER = (1 << 53) - 1
_SPLIT_ASSETS = {
    "graph": "graph.json",
    "weights": "model.safetensors",
}
_EXPORT_REPORT_FILENAME = "export_report.json"


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    digest = _nonempty_string(value, label)
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    if value > _MAX_JSON_SAFE_INTEGER:
        raise ValueError(f"{label} must be a JSON-safe integer")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} must contain exactly {', '.join(sorted(expected))}"
        )


def _selected_graph_kinds(
    graphs: Mapping[str, Any],
    graph_kinds: tuple[str, ...] | None,
) -> tuple[str, ...]:
    selected = tuple(graphs) if graph_kinds is None else graph_kinds
    if (
        not isinstance(selected, tuple)
        or not selected
        or any(not isinstance(kind, str) or not kind for kind in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("split package graph kinds must be unique non-empty strings")
    missing = [kind for kind in selected if kind not in graphs]
    if missing:
        raise ValueError(f"split package manifest is missing graphs {missing!r}")
    return selected


def graph_manifest_inputs(
    document: Mapping[str, Any],
    declared: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Validate one semantic-name to graph-tensor public-input mapping."""

    if not isinstance(document, Mapping):
        raise ValueError("graph document must be an object")
    inputs = document.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("graph inputs must be an object")

    tensor_names: list[str] = []
    for tensor_name, descriptor in inputs.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("graph input names must be non-empty strings")
        if not isinstance(descriptor, Mapping):
            raise ValueError(
                f"graph input {tensor_name!r} descriptor must be an object"
            )
        tensor_names.append(tensor_name)

    if declared is None:
        return {name: name for name in tensor_names}
    if not isinstance(declared, Mapping):
        raise ValueError("split package graph inputs must be an object")

    result: dict[str, str] = {}
    owners: dict[str, str] = {}
    known = set(tensor_names)
    for semantic_name, tensor_name in declared.items():
        if not isinstance(semantic_name, str) or not semantic_name:
            raise ValueError("manifest semantic input names must be non-empty strings")
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError(
                f"manifest input {semantic_name!r} must name a non-empty graph tensor"
            )
        if tensor_name not in known:
            raise ValueError(
                f"manifest input {semantic_name!r} names unknown graph tensor "
                f"{tensor_name!r}"
            )
        previous = owners.get(tensor_name)
        if previous is not None:
            raise ValueError(
                f"manifest inputs {previous!r} and {semantic_name!r} share graph "
                f"tensor {tensor_name!r}"
            )
        owners[tensor_name] = semantic_name
        result[semantic_name] = tensor_name
    missing = [name for name in tensor_names if name not in owners]
    if missing:
        raise ValueError(f"manifest graph input mapping is missing tensors {missing!r}")
    return result


def refresh_graph_manifest_inputs(
    manifest: MutableMapping[str, Any],
    kind: str,
    document: Mapping[str, Any],
) -> dict[str, str]:
    """Replace one graph's manifest inputs from its final graph document."""

    if not isinstance(manifest, MutableMapping):
        raise ValueError("split package manifest must be an object")
    if not isinstance(kind, str) or not kind:
        raise ValueError("split package graph kind must be a non-empty string")
    graphs = manifest.get("graphs")
    if not isinstance(graphs, MutableMapping):
        raise ValueError("split package manifest graphs must be an object")
    entry = graphs.get(kind)
    if not isinstance(entry, MutableMapping):
        raise ValueError(f"split package manifest graph {kind!r} must be an object")
    existing = entry.get("inputs")
    if not isinstance(existing, Mapping):
        raise ValueError(
            f"split package manifest graph {kind!r} inputs must be an object"
        )

    graph_inputs = graph_manifest_inputs(document)
    graph_names = set(graph_inputs)
    projected: dict[str, str] = {}
    claimed: set[str] = set()
    for semantic_name, tensor_name in existing.items():
        if (
            isinstance(semantic_name, str)
            and semantic_name
            and isinstance(tensor_name, str)
            and tensor_name in graph_names
            and tensor_name not in claimed
        ):
            projected[semantic_name] = tensor_name
            claimed.add(tensor_name)
    for tensor_name in graph_inputs:
        if tensor_name in claimed:
            continue
        if tensor_name in projected:
            raise ValueError(
                f"new graph input {tensor_name!r} conflicts with an existing "
                "manifest semantic name"
            )
        projected[tensor_name] = tensor_name
        claimed.add(tensor_name)
    entry["inputs"] = projected
    return projected


def refresh_split_package_identities(
    manifest: MutableMapping[str, Any],
    root: Path,
    *,
    graph_kinds: tuple[str, ...] | None = None,
) -> None:
    """Refresh graph and weight identities from staged package bytes."""

    graphs = manifest.get("graphs") if isinstance(manifest, MutableMapping) else None
    if not isinstance(graphs, MutableMapping):
        raise ValueError("split package manifest graphs must be an object")
    for kind in _selected_graph_kinds(graphs, graph_kinds):
        graph_entry = graphs[kind]
        if not isinstance(graph_entry, MutableMapping):
            raise ValueError(f"split package manifest graph {kind!r} must be an object")
        for asset, filename in _SPLIT_ASSETS.items():
            asset_entry = graph_entry.get(asset)
            if not isinstance(asset_entry, MutableMapping):
                raise ValueError(
                    f"split package manifest {kind}.{asset} must be an object"
                )
            payload = (Path(root) / kind / filename).read_bytes()
            asset_entry["sha256"] = hashlib.sha256(payload).hexdigest()
            asset_entry["bytes"] = len(payload)


def _load_manifest_object(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"{label} contains forbidden non-finite JSON number {token!r}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")

    def verify_numbers(item: Any, path_label: str) -> None:
        if item is None or isinstance(item, (str, bool)):
            return
        if isinstance(item, int):
            if abs(item) > _MAX_JSON_SAFE_INTEGER:
                raise ValueError(
                    f"{path_label} contains an integer outside JSON's safe range"
                )
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{path_label} contains a non-finite JSON number")
            if item.is_integer() and abs(item) > _MAX_JSON_SAFE_INTEGER:
                raise ValueError(
                    f"{path_label} contains an integer outside JSON's safe range"
                )
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                verify_numbers(child, f"{path_label}[{index}]")
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                verify_numbers(child, f"{path_label}.{key}")

    verify_numbers(value, label)
    return value


def _required_asset_path(
    root: Path,
    kind: str,
    asset: str,
    entry: Mapping[str, Any],
) -> Path:
    label = f"split package manifest {kind}.{asset}"
    raw_path = _nonempty_string(entry.get("path"), f"{label}.path")
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ValueError(f"{label}.path must be a normalized package-relative path")
    path = root.joinpath(*relative.parts)
    expected = root / kind / _SPLIT_ASSETS[asset]
    if path.resolve(strict=False) != expected.resolve(strict=False):
        raise ValueError(f"{label}.path must be {kind}/{_SPLIT_ASSETS[asset]}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required split package artifact is missing: {path}")
    return path


def validated_split_package_export_report_path(
    root: Path,
    kind: str,
    graph_entry: Mapping[str, Any],
) -> Path | None:
    """Return a declared export report after proving its path and identity."""

    entry_source = _object(graph_entry, f"split package manifest graph {kind}")
    if "export_report" not in entry_source:
        return None
    label = f"split package manifest {kind}.export_report"
    report_entry = _object(entry_source["export_report"], label)
    _exact_keys(report_entry, {"path", "bytes", "sha256"}, label)
    expected_relative = f"{kind}/{_EXPORT_REPORT_FILENAME}"
    raw_path = _nonempty_string(report_entry.get("path"), f"{label}.path")
    if raw_path != expected_relative:
        raise ValueError(f"{label}.path must be {expected_relative}")
    expected_bytes = _integer(report_entry.get("bytes"), f"{label}.bytes")
    expected_sha = _sha256(report_entry.get("sha256"), f"{label}.sha256")

    path = Path(root) / kind / _EXPORT_REPORT_FILENAME
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"declared split package export report is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(
            "declared split package export report must be a regular "
            f"non-symlink file: {path}"
        )
    if metadata.st_size != expected_bytes:
        raise ValueError(
            f"split package {kind}.export_report size mismatch: "
            f"expected {expected_bytes}, got {metadata.st_size}"
        )
    payload = path.read_bytes()
    if len(payload) != expected_bytes:
        raise ValueError(
            f"split package {kind}.export_report size mismatch: "
            f"expected {expected_bytes}, got {len(payload)}"
        )
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            f"split package {kind}.export_report digest mismatch: "
            f"expected {expected_sha}, got {actual_sha}"
        )
    return path


def validate_split_package(
    root: Path,
    *,
    graph_kinds: tuple[str, ...] | None = None,
    graph_validator: (
        Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None
    ) = None,
) -> None:
    """Validate every selected graph and asset in one staged package."""

    package_root = Path(root).resolve()
    manifest_path = package_root / "package_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("staged split package is missing package_manifest.json")
    manifest = _load_manifest_object(manifest_path, "split package manifest")
    _nonempty_string(manifest.get("format"), "split package manifest format")
    graphs = _object(manifest.get("graphs"), "split package manifest graphs")

    for kind in _selected_graph_kinds(graphs, graph_kinds):
        graph_entry = _object(
            graphs[kind], f"split package manifest graph {kind}"
        )
        validated_split_package_export_report_path(
            package_root, kind, graph_entry,
        )
        paths: dict[str, Path] = {}
        for asset in _SPLIT_ASSETS:
            asset_entry = _object(
                graph_entry.get(asset),
                f"split package manifest {kind}.{asset}",
            )
            path = _required_asset_path(package_root, kind, asset, asset_entry)
            payload = path.read_bytes()
            expected_bytes = _integer(
                asset_entry.get("bytes"),
                f"split package manifest {kind}.{asset}.bytes",
            )
            expected_sha = _sha256(
                asset_entry.get("sha256"),
                f"split package manifest {kind}.{asset}.sha256",
            )
            if len(payload) != expected_bytes:
                raise ValueError(
                    f"split package {kind}.{asset} size mismatch: "
                    f"expected {expected_bytes}, got {len(payload)}"
                )
            actual_sha = hashlib.sha256(payload).hexdigest()
            if actual_sha != expected_sha:
                raise ValueError(
                    f"split package {kind}.{asset} digest mismatch: "
                    f"expected {expected_sha}, got {actual_sha}"
                )
            paths[asset] = path

        document = load_runtime_document(paths["graph"])
        weights = read_safetensors(paths["weights"])
        validate_external_quantization(document, weights)
        try:
            graph_manifest_inputs(document, graph_entry.get("inputs"))
        except ValueError as error:
            raise ValueError(
                f"split package manifest {kind}.inputs does not match graph inputs: "
                f"{error}"
            ) from error

        outputs = document.get("outputs")
        manifest_outputs = graph_entry.get("outputs")
        if (
            not isinstance(outputs, list)
            or not isinstance(manifest_outputs, Mapping)
            or any(not isinstance(name, str) or not name for name in manifest_outputs)
            or any(
                not isinstance(name, str) or not name
                for name in manifest_outputs.values()
            )
            or len(manifest_outputs) != len(outputs)
            or len(set(manifest_outputs.values())) != len(outputs)
            or set(manifest_outputs.values()) != set(outputs)
        ):
            raise ValueError(
                f"split package manifest {kind}.outputs does not match graph outputs"
            )

        if graph_validator is None:
            bounded_domain_proof = prove_dynamic_quantized_runtime_domain(
                document, weights,
            )
            import_runtime_package(
                document,
                weights,
                source_name=str(paths["graph"]),
                bounded_domain_proof=bounded_domain_proof,
            )
        else:
            graph_validator(document, weights)


__all__ = [
    "graph_manifest_inputs",
    "refresh_graph_manifest_inputs",
    "refresh_split_package_identities",
    "validate_split_package",
    "validated_split_package_export_report_path",
]
