"""Validate persisted runtime packages through the canonical typed importer.

This module is the semantic companion to ``tools/validate_model_packages.mjs``.
The JavaScript validator owns package discovery and file-name rules; this
module re-opens each discovered graph and its already-scoped safetensors files
with the strict Python readers, merges the tensor inventory without aliases,
proves every canonical operator over the exact bounded domain for all four
portable members, and runs the same ``import_runtime_package`` gate used by
the optimizer.  Warm profiles are not accepted as qualification evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .generated.kernel_registry import PROFILE_MEMBERS
from .optimizer.safetensors_io import read_safetensors
from .portable_domain import prove_portable_graph_domain
from .runtime_ir import import_runtime_package, load_runtime_document


_SPEC_FIELDS = frozenset({"graph", "weights"})


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"package specification contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_number(token: str) -> None:
    raise ValueError(
        f"package specification contains non-finite JSON number {token!r}"
    )


def _parse_spec(raw: str) -> tuple[Path, tuple[Path, ...]]:
    value = json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_number,
    )
    if not isinstance(value, Mapping) or set(value) != _SPEC_FIELDS:
        raise ValueError(
            "package specification must contain exactly 'graph' and 'weights'"
        )
    graph = value.get("graph")
    weights = value.get("weights")
    if not isinstance(graph, str) or not graph:
        raise ValueError("package specification graph must be a non-empty path")
    if (
        not isinstance(weights, list)
        or any(not isinstance(item, str) or not item for item in weights)
        or len(weights) != len(set(weights))
    ):
        raise ValueError(
            "package specification weights must be unique non-empty paths"
        )
    return Path(graph), tuple(Path(item) for item in weights)


def validate_runtime_package(
    graph_path: str | Path,
    weight_paths: Sequence[str | Path],
) -> None:
    """Strictly load and semantically validate one current runtime package."""

    graph = Path(graph_path)
    tensors: dict[str, Any] = {}
    owners: dict[str, Path] = {}
    for raw_weight_path in weight_paths:
        weight_path = Path(raw_weight_path)
        for name, value in read_safetensors(weight_path).items():
            previous = owners.get(name)
            if previous is not None:
                raise ValueError(
                    f"tensor {name!r} occurs in multiple safetensors files: "
                    f"{previous} and {weight_path}"
                )
            owners[name] = weight_path
            tensors[name] = value

    document = load_runtime_document(graph)
    portable_members = tuple(PROFILE_MEMBERS["portable"])
    bounded_domain_proof = prove_portable_graph_domain(
        document,
        tensors,
        portable_members,
    )
    import_runtime_package(
        document,
        tensors,
        source_name=str(graph),
        bounded_domain_proof=bounded_domain_proof,
    )


def validate_runtime_packages(
    packages: Sequence[tuple[str | Path, Sequence[str | Path]]],
) -> None:
    """Validate a batch in one process, preserving each graph's weight scope."""

    for graph_path, weight_paths in packages:
        try:
            validate_runtime_package(graph_path, weight_paths)
        except Exception as error:
            raise ValueError(f"{Path(graph_path)}: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate current volvox-graph/v1 packages through strict "
            "safetensors loading and the canonical typed RuntimeIR importer."
        ),
    )
    parser.add_argument(
        "--package",
        action="append",
        required=True,
        metavar="JSON",
        help="JSON object with an absolute graph path and its scoped weights array",
    )
    args = parser.parse_args(argv)
    try:
        packages = [_parse_spec(raw) for raw in args.package]
        validate_runtime_packages(packages)
    except Exception as error:
        print(f"runtime package semantic validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
