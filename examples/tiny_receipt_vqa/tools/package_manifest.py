"""Project verified optimizer metadata into a split-package manifest.

The package manifest is an audit surface, not storage for calibration ranges.
Only the identity and the explicitly selected, non-affine provenance fields of
an activation-calibration report may be published here.  Numeric range maps
remain in the calibration input, while affine parameters remain tensors in the
package's safetensors files.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, MutableMapping
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from tools.exporter.quantization_storage import validate_external_quantization
from tools.exporter.runtime_ir import import_runtime_package, load_runtime_document
from tools.exporter.optimizer.safetensors_io import read_safetensors


CALIBRATION_PROVENANCE_FORMAT = "volvox-calibration-provenance/v2"
CALIBRATION_ROUTING_QUALIFICATION_FORMAT = (
    "volvox-tiny-receipt-routing-qualification/v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_JSON_SAFE_INTEGER = (1 << 53) - 1

_SPLIT_ASSETS = {
    "graph": "graph.json",
    "weights": "model.safetensors",
}


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
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _identity(value: Any, label: str) -> dict[str, Any]:
    source = _object(value, label)
    result: dict[str, Any] = {
        "sha256": _sha256(source.get("sha256"), f"{label}.sha256"),
    }
    if "bytes" in source:
        result["bytes"] = _integer(source["bytes"], f"{label}.bytes")
    return result


def _string_list(value: Any, label: str, *, allow_empty: bool) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        name = _nonempty_string(item, f"{label}[{index}]")
        if name in seen:
            raise ValueError(f"{label} must not contain duplicate {name!r}")
        seen.add(name)
        result.append(name)
    if not allow_empty and not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _family_count_map(value: Any, label: str) -> dict[str, int] | None:
    if value is None:
        return None
    source = _object(value, label)
    if not source:
        raise ValueError(f"{label} must not be empty")
    result: dict[str, int] = {}
    for family, count in sorted(source.items()):
        name = _nonempty_string(family, f"{label} family name")
        result[name] = _integer(count, f"{label}.{name}")
    return result


def _selected_records(value: Any) -> dict[str, Any]:
    source = _object(value, "calibration provenance selected_records")
    count = _integer(
        source.get("count"), "calibration provenance selected_records.count"
    )
    identifiers = _string_list(
        source.get("ids"),
        "calibration provenance selected_records.ids",
        allow_empty=True,
    )
    if count != len(identifiers):
        raise ValueError(
            "calibration provenance selected_records.count must equal ids length"
        )
    return {"count": count, "ids": identifiers}


def _family_counts(
    value: Any,
    *,
    selected_count: int | None,
) -> dict[str, dict[str, int] | None]:
    source = _object(value, "calibration provenance family_counts")
    allowed = ("required", "record")
    if set(source) != set(allowed):
        raise ValueError(
            "calibration provenance family_counts requires exactly required and record"
        )
    result = {
        name: _family_count_map(
            source[name], f"calibration provenance family_counts.{name}"
        )
        for name in allowed
    }
    if selected_count is not None:
        counts = result["record"]
        if counts is not None and sum(counts.values()) != selected_count:
            raise ValueError(
                "calibration provenance family_counts.record total must equal "
                "selected record count"
            )
    return result


def _route_executions(value: Any) -> dict[str, Any]:
    """Validate numeric executions independently of representative records."""

    source = _object(value, "calibration provenance route_executions")
    expected = {"mode", "count", "family_counts"}
    if set(source) != expected:
        raise ValueError(
            "calibration provenance route_executions requires exactly mode, count, "
            "and family_counts"
        )
    mode = _nonempty_string(
        source.get("mode"), "calibration provenance route_executions.mode"
    )
    if mode not in {"all-public", "balanced-public", "graph-routing"}:
        raise ValueError(
            "calibration provenance route_executions.mode must be all-public, "
            "balanced-public, or graph-routing"
        )
    count = _integer(
        source.get("count"),
        "calibration provenance route_executions.count",
        positive=True,
    )
    counts = _family_count_map(
        source.get("family_counts"),
        "calibration provenance route_executions.family_counts",
    )
    if counts is not None and sum(counts.values()) != count:
        raise ValueError(
            "calibration provenance route_executions.family_counts total must equal count"
        )
    return {"mode": mode, "count": count, "family_counts": counts}


def _settings(value: Any, *, selected_count: int | None) -> dict[str, Any]:
    source = _object(value, "calibration provenance settings")
    routing_mode = _nonempty_string(
        source.get("routing_mode"),
        "calibration provenance settings.routing_mode",
    )
    if routing_mode not in {"auto", "explicit", "specialized", "fixture"}:
        raise ValueError(
            "calibration provenance settings.routing_mode must be auto, explicit, "
            "specialized, or fixture"
        )
    result = {
        "samples": _integer(
            source.get("samples"), "calibration provenance settings.samples", positive=True
        ),
        "prefixes_per_record": _integer(
            source.get("prefixes_per_record"),
            "calibration provenance settings.prefixes_per_record",
        ),
        "backend": _nonempty_string(
            source.get("backend"), "calibration provenance settings.backend"
        ),
        "batch": _integer(
            source.get("batch"), "calibration provenance settings.batch", positive=True
        ),
        "graphs": _string_list(
            source.get("graphs"),
            "calibration provenance settings.graphs",
            allow_empty=False,
        ),
        "routing_mode": routing_mode,
    }
    if selected_count and result["samples"] != selected_count:
        raise ValueError(
            "calibration provenance settings.samples must equal selected record count"
        )
    return result


def verify_calibration_source_package(
    provenance: Mapping[str, Any],
    source: Path,
    *,
    graph_kinds: tuple[str, ...] = ("encoder", "decoder"),
) -> None:
    """Prove a calibration report was measured on the exact PTQ input bytes.

    The calibration artifact is external authoring data and may be copied or
    accidentally reused across specialized family packages.  A valid digest in
    its metadata is not enough: each declared graph and weight digest must equal
    the files that this quantization invocation is about to transform.
    """

    projected = _object(provenance, "calibration provenance projection")
    package = _object(
        projected.get("package"), "calibration provenance source package"
    )
    graphs = _object(
        package.get("graphs"), "calibration provenance source package graphs"
    )
    expected_kinds = set(graph_kinds)
    if set(graphs) != expected_kinds:
        raise ValueError(
            "calibration provenance source package must identify exactly "
            + ", ".join(sorted(expected_kinds))
        )

    root = Path(source)
    for kind in graph_kinds:
        entry = _object(
            graphs[kind], f"calibration provenance source package {kind}"
        )
        for asset, filename in (
            ("graph", "graph.json"),
            ("weights", "model.safetensors"),
        ):
            identity = _object(
                entry.get(asset),
                f"calibration provenance source package {kind} {asset}",
            )
            expected_sha = _sha256(
                identity.get("sha256"),
                f"calibration provenance source package {kind} {asset}.sha256",
            )
            path = root / kind / filename
            try:
                payload = path.read_bytes()
            except OSError as error:
                raise ValueError(
                    f"cannot verify calibration source {kind} {asset}: {path}"
                ) from error
            actual_sha = hashlib.sha256(payload).hexdigest()
            if actual_sha != expected_sha:
                raise ValueError(
                    f"calibration source identity mismatch for {kind} {asset}: "
                    f"expected {expected_sha}, got {actual_sha}"
                )
            if "bytes" in identity:
                expected_bytes = _integer(
                    identity["bytes"],
                    f"calibration provenance source package {kind} {asset}.bytes",
                )
                if len(payload) != expected_bytes:
                    raise ValueError(
                        f"calibration source size mismatch for {kind} {asset}: "
                        f"expected {expected_bytes}, got {len(payload)}"
                    )


def _source_package(value: Any) -> dict[str, Any]:
    source = _object(value, "calibration provenance package")
    graphs = _object(
        source.get("graphs"), "calibration provenance package.graphs"
    )
    if not graphs:
        raise ValueError("calibration provenance package.graphs must not be empty")
    projected: dict[str, Any] = {}
    for kind, raw_entry in sorted(graphs.items()):
        graph_kind = _nonempty_string(
            kind, "calibration provenance package graph name"
        )
        entry = _object(
            raw_entry, f"calibration provenance package.graphs.{graph_kind}"
        )
        projected[graph_kind] = {
            "graph": _identity(
                entry.get("graph"),
                f"calibration provenance package.graphs.{graph_kind}.graph",
            ),
            "weights": _identity(
                entry.get("weights"),
                f"calibration provenance package.graphs.{graph_kind}.weights",
            ),
        }
    return {"graphs": projected}


def _affine_exclusions(value: Any) -> dict[str, Any]:
    source = _object(value, "calibration provenance affine_exclusions")
    result: dict[str, Any] = {}
    for kind, raw_entry in sorted(source.items()):
        graph_kind = _nonempty_string(
            kind, "calibration provenance affine_exclusions graph name"
        )
        label = f"calibration provenance affine_exclusions.{graph_kind}"
        entry = _object(raw_entry, label)
        tensors = _string_list(
            entry.get("tensors"), f"{label}.tensors", allow_empty=True
        )
        count = _integer(entry.get("count"), f"{label}.count")
        if count != len(tensors):
            raise ValueError(f"{label}.count must equal tensors length")
        reasons_source = _object(entry.get("reasons"), f"{label}.reasons")
        if set(reasons_source) != set(tensors):
            raise ValueError(f"{label}.reasons must describe exactly its tensors")
        reasons = {
            tensor: _nonempty_string(reasons_source[tensor], f"{label}.reasons.{tensor}")
            for tensor in tensors
        }
        result[graph_kind] = {
            "semantic_domain": _nonempty_string(
                entry.get("semantic_domain"), f"{label}.semantic_domain"
            ),
            "count": count,
            "tensors": tensors,
            "reasons": reasons,
        }
    return result


def calibration_provenance(
    document: Mapping[str, Any], calibration_bytes: bytes
) -> dict[str, Any]:
    """Return the safe, reproducible identity of a calibration report.

    The projection is deliberately a whitelist.  Absolute paths, activation
    ``min``/``max`` maps, and any future unreviewed producer fields cannot leak
    into a package manifest or export report.  Present whitelisted fields are
    validated as one coherent contract rather than copied best-effort.
    """

    source = _object(document, "calibration document")
    if not isinstance(calibration_bytes, bytes):
        raise ValueError("calibration bytes must be bytes")
    result: dict[str, Any] = {
        "format": CALIBRATION_PROVENANCE_FORMAT,
        "artifact": {
            "sha256": hashlib.sha256(calibration_bytes).hexdigest(),
            "bytes": len(calibration_bytes),
        },
    }
    if "provenance" not in source:
        return result
    provenance = _object(source["provenance"], "calibration provenance")

    selected_count: int | None = None
    if "selected_records" in provenance:
        selected = _selected_records(provenance["selected_records"])
        selected_count = selected["count"]
        result["selected_records"] = selected
    if "calibration_manifest" in provenance and provenance["calibration_manifest"] is not None:
        result["calibration_manifest"] = _identity(
            provenance["calibration_manifest"],
            "calibration provenance calibration_manifest",
        )
    if "family_counts" in provenance:
        result["family_counts"] = _family_counts(
            provenance["family_counts"], selected_count=selected_count
        )
    if "route_executions" in provenance:
        result["route_executions"] = _route_executions(
            provenance["route_executions"]
        )
    if "settings" in provenance:
        result["settings"] = _settings(
            provenance["settings"], selected_count=selected_count
        )
    if "package" in provenance:
        result["package"] = _source_package(provenance["package"])
    if "calibrator_source" in provenance:
        result["calibrator_source"] = _identity(
            provenance["calibrator_source"],
            "calibration provenance calibrator_source",
        )
    if "affine_exclusions" in provenance:
        result["affine_exclusions"] = _affine_exclusions(
            provenance["affine_exclusions"]
        )
    return result


def graph_manifest_inputs(document: Mapping[str, Any]) -> dict[str, str]:
    """Return the exact semantic-name -> graph-tensor input mapping.

    Exported graphs commonly retain positional tensor names such as ``input0``
    and put the authored name in ``source_name``.  Optimizer-created inputs do
    not necessarily have source provenance, so their exact graph name is their
    public semantic name.  A duplicate semantic name is ambiguous to package
    consumers and therefore invalid rather than last-write-wins.
    """

    if not isinstance(document, Mapping):
        raise ValueError("graph document must be an object")
    inputs = document.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("graph inputs must be an object")

    result: dict[str, str] = {}
    owners: dict[str, str] = {}
    for tensor_name, descriptor in inputs.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("graph input names must be non-empty strings")
        if not isinstance(descriptor, Mapping):
            raise ValueError(
                f"graph input {tensor_name!r} descriptor must be an object"
            )
        semantic_name = descriptor.get("source_name", tensor_name)
        if not isinstance(semantic_name, str) or not semantic_name:
            raise ValueError(
                f"graph input {tensor_name!r} source_name must be a non-empty string"
            )
        previous = owners.get(semantic_name)
        if previous is not None:
            raise ValueError(
                f"graph inputs {previous!r} and {tensor_name!r} share semantic "
                f"name {semantic_name!r}"
            )
        owners[semantic_name] = tensor_name
        result[semantic_name] = tensor_name
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
    projected = graph_manifest_inputs(document)
    graphs = manifest.get("graphs")
    if not isinstance(graphs, MutableMapping):
        raise ValueError("split package manifest graphs must be an object")
    entry = graphs.get(kind)
    if not isinstance(entry, MutableMapping):
        raise ValueError(f"split package manifest graph {kind!r} must be an object")
    entry["inputs"] = projected
    return projected


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} must contain exactly {', '.join(sorted(expected))}"
        )


def split_package_routing(
    manifest: Mapping[str, Any],
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    """Validate and project the split package's public family-routing ABI.

    Runtime-selectable packages name the physical ``family_ids`` input of both
    graphs. Specialized packages instead record the exact family ID baked into
    their tensor stores and must expose no public family input. Keeping these
    two forms disjoint prevents an eight-family catalog from disguising a
    single-family graph.
    """

    source = _object(manifest, "split package manifest")
    if "routing" not in source:
        if required:
            raise ValueError("split package manifest requires explicit routing")
        return None
    routing = _object(source["routing"], "split package manifest routing")
    mode = _nonempty_string(
        routing.get("mode"), "split package manifest routing.mode"
    )
    graphs = _object(source.get("graphs"), "split package manifest graphs")
    graph_inputs = {
        kind: _object(
            _object(
                graphs.get(kind), f"split package manifest graph {kind}"
            ).get("inputs"),
            f"split package manifest graph {kind}.inputs",
        )
        for kind in ("encoder", "decoder")
    }

    if mode == "runtime":
        _exact_keys(
            routing, {"mode", "family_inputs"}, "split package manifest routing"
        )
        family_inputs = _object(
            routing.get("family_inputs"),
            "split package manifest routing.family_inputs",
        )
        _exact_keys(
            family_inputs,
            {"encoder", "decoder"},
            "split package manifest routing.family_inputs",
        )
        projected: dict[str, str] = {}
        for kind in ("encoder", "decoder"):
            declared = _nonempty_string(
                family_inputs.get(kind),
                f"split package manifest routing.family_inputs.{kind}",
            )
            actual = graph_inputs[kind].get("family_ids")
            if actual != declared:
                raise ValueError(
                    f"split package runtime routing {kind} family input does not "
                    "match graph inputs"
                )
            projected[kind] = declared
        return {"mode": "runtime", "family_inputs": projected}

    if mode == "specialized":
        _exact_keys(
            routing, {"mode", "family_id"}, "split package manifest routing"
        )
        family_id = _integer(
            routing.get("family_id"), "split package manifest routing.family_id"
        )
        for kind, inputs in graph_inputs.items():
            if "family_ids" in inputs:
                raise ValueError(
                    f"split package specialized routing forbids {kind} family_ids input"
                )
        families = source.get("families")
        if isinstance(families, Mapping):
            ordered = families.get("ordered_names")
            if isinstance(ordered, list) and family_id >= len(ordered):
                raise ValueError(
                    "split package manifest routing.family_id is outside the family catalog"
                )
        return {"mode": "specialized", "family_id": family_id}

    raise ValueError(
        "split package manifest routing.mode must be runtime or specialized"
    )


def refresh_split_package_routing(
    manifest: MutableMapping[str, Any],
    *,
    specialized_family_id: int | None = None,
) -> dict[str, Any]:
    """Publish routing metadata from the final encoder/decoder input ABI."""

    graphs = _object(manifest.get("graphs"), "split package manifest graphs")
    inputs = {
        kind: _object(
            _object(
                graphs.get(kind), f"split package manifest graph {kind}"
            ).get("inputs"),
            f"split package manifest graph {kind}.inputs",
        )
        for kind in ("encoder", "decoder")
    }
    live = {kind: values.get("family_ids") for kind, values in inputs.items()}
    if all(isinstance(value, str) and value for value in live.values()):
        if specialized_family_id is not None:
            raise ValueError(
                "split package cannot publish a specialized family while family_ids "
                "inputs remain live"
            )
        manifest["routing"] = {
            "mode": "runtime",
            "family_inputs": {
                "encoder": live["encoder"],
                "decoder": live["decoder"],
            },
        }
    elif all(value is None for value in live.values()):
        family_id = specialized_family_id
        if family_id is None:
            previous = split_package_routing(manifest, required=True)
            if previous is None or previous["mode"] != "specialized":
                raise ValueError(
                    "split package without family_ids inputs requires an exact "
                    "specialized family_id"
                )
            family_id = previous["family_id"]
        manifest["routing"] = {
            "mode": "specialized",
            "family_id": _integer(
                family_id, "split package manifest routing.family_id"
            ),
        }
    else:
        raise ValueError(
            "split package encoder and decoder must both expose family_ids or both "
            "be specialized"
        )
    projected = split_package_routing(manifest, required=True)
    assert projected is not None
    return projected


def refresh_split_package_identities(
    manifest: MutableMapping[str, Any],
    root: Path,
    *,
    graph_kinds: tuple[str, ...] = ("encoder", "decoder"),
) -> None:
    """Refresh required graph/weight identities from staged package bytes."""

    graphs = manifest.get("graphs") if isinstance(manifest, MutableMapping) else None
    if not isinstance(graphs, MutableMapping):
        raise ValueError("split package manifest graphs must be an object")
    for kind in graph_kinds:
        graph_entry = graphs.get(kind)
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
        raise ValueError(
            f"{label} contains forbidden non-finite JSON number {token!r}"
        )

    value = json.loads(path.read_text(encoding="utf-8"),
                       object_pairs_hook=reject_duplicates,
                       parse_constant=reject_nonfinite)
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


def calibration_routing_qualification(
    provenance: Mapping[str, Any],
    source: Path,
) -> dict[str, Any]:
    """Describe route coverage without granting it PTQ legality authority.

    RuntimeIR PTQ is model-neutral: an I32 routing input is an ordinary public
    input and calibration does not specialize it.  This TinyReceipt-only
    projection retains route evidence for later quality qualification, including
    explicit zero counts for every unobserved catalog family.  Its result must
    never be used to add/remove graph inputs or refuse otherwise complete tensor
    calibration.
    """

    projected = _object(provenance, "calibration provenance projection")
    if projected.get("format") != CALIBRATION_PROVENANCE_FORMAT:
        raise ValueError("calibration provenance projection has the wrong format")

    manifest_path = Path(source) / "package_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("calibration source package is missing package_manifest.json")
    manifest = _load_manifest_object(
        manifest_path,
        "calibration source package manifest",
    )
    routing = split_package_routing(manifest, required=True)
    assert routing is not None

    families = _object(
        manifest.get("families"),
        "calibration source package manifest families",
    )
    ordered_names = _string_list(
        families.get("ordered_names"),
        "calibration source package manifest families.ordered_names",
        allow_empty=False,
    )
    settings = projected.get("settings")
    calibration_routing_mode = (
        settings.get("routing_mode") if isinstance(settings, Mapping) else None
    )
    route_executions = projected.get("route_executions")
    routed_value = (
        route_executions.get("family_counts")
        if isinstance(route_executions, Mapping) else None
    )
    routed = routed_value if isinstance(routed_value, Mapping) else {}
    normalized_routed = {
        _nonempty_string(family, "calibration routed family name"): _integer(
            count, f"calibration routed family {family!r} count"
        )
        for family, count in sorted(routed.items())
    }
    catalog_counts = {
        family: normalized_routed.get(family, 0) for family in ordered_names
    }
    unknown_counts = {
        family: count for family, count in normalized_routed.items()
        if family not in catalog_counts
    }

    if routing["mode"] == "runtime":
        required_routes = list(ordered_names)
    else:
        family_id = routing["family_id"]
        if family_id >= len(ordered_names):
            raise ValueError(
                "specialized routing family_id is outside families.ordered_names"
            )
        required_routes = [ordered_names[family_id]]
    unobserved = [
        family for family in required_routes if catalog_counts[family] == 0
    ]
    observed_outside_required = [
        family for family in ordered_names
        if family not in required_routes and catalog_counts[family] > 0
    ]
    return {
        "format": CALIBRATION_ROUTING_QUALIFICATION_FORMAT,
        "package_routing": routing,
        "calibration_routing_mode": calibration_routing_mode,
        "execution_mode": (
            route_executions.get("mode")
            if isinstance(route_executions, Mapping) else None
        ),
        "execution_count": (
            route_executions.get("count")
            if isinstance(route_executions, Mapping) else 0
        ),
        "required_route_families": required_routes,
        "catalog_family_counts": catalog_counts,
        "unobserved_required_families": unobserved,
        "observed_outside_required_families": observed_outside_required,
        "unknown_family_counts": unknown_counts,
        "coverage_complete": not (
            unobserved or observed_outside_required or unknown_counts
        ),
    }


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
        raise ValueError(
            f"{label}.path must be {kind}/{_SPLIT_ASSETS[asset]}"
        )
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required split package artifact is missing: {path}")
    return path


def validate_split_package(
    root: Path,
    *,
    graph_kinds: tuple[str, ...] = ("encoder", "decoder"),
    graph_validator: (
        Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None
    ) = None,
    require_routing: bool = False,
) -> None:
    """Validate a complete staged split package before its one rename.

    Required artifact paths are canonical and package-relative, graph and
    safetensors bytes parse under the current v1 contract, manifest ABI maps
    match those graphs, and every recorded size/digest covers the staged bytes.
    Runnable packages use the full typed RuntimeIR verifier by default. An
    explicitly supplied validator exists only for named offline compiler
    artifacts such as the fuse-before-PTQ intermediate; it receives the
    strict-loaded document and only that graph's safetensors inventory.
    """

    package_root = Path(root).resolve()
    manifest_path = package_root / "package_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("staged split package is missing package_manifest.json")
    manifest = _load_manifest_object(manifest_path, "split package manifest")
    _nonempty_string(manifest.get("format"), "split package manifest format")
    graphs = _object(manifest.get("graphs"), "split package manifest graphs")
    split_package_routing(manifest, required=require_routing)

    for kind in graph_kinds:
        graph_entry = _object(
            graphs.get(kind), f"split package manifest graph {kind}"
        )
        paths: dict[str, Path] = {}
        for asset in _SPLIT_ASSETS:
            asset_entry = _object(
                graph_entry.get(asset),
                f"split package manifest {kind}.{asset}",
            )
            path = _required_asset_path(
                package_root, kind, asset, asset_entry
            )
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
        if graph_manifest_inputs(document) != graph_entry.get("inputs"):
            raise ValueError(
                f"split package manifest {kind}.inputs does not match graph inputs"
            )
        outputs = document.get("outputs")
        manifest_outputs = graph_entry.get("outputs")
        if (
            not isinstance(outputs, list)
            or not isinstance(manifest_outputs, Mapping)
            or list(manifest_outputs.values()) != outputs
        ):
            raise ValueError(
                f"split package manifest {kind}.outputs does not match graph outputs"
            )
        if graph_validator is None:
            import_runtime_package(
                document,
                weights,
                source_name=str(paths["graph"]),
            )
        else:
            graph_validator(document, weights)
