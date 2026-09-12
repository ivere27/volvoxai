#!/usr/bin/env python3
"""Inventory one oracle strategy and separately count executable ORT evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.exporter.generated.kernel_registry import OPS_BY_TARGET, PROFILE_MEMBERS


MANIFEST_PATH = REPOSITORY_ROOT / "tests" / "contracts" / "oracle_coverage.json"
CASE_MANIFEST_PATH = (
    REPOSITORY_ROOT / "tests" / "parity" / "external" / "cases.json"
)
SOURCE_REGISTRY = "tools/exporter/generated/kernel_registry.py::OPS_BY_TARGET"
CASE_MANIFEST = "tests/parity/external/cases.json"
CLASSIFICATIONS = (
    "onnx-direct",
    "onnx-decomposition",
    "oracle-unsupported",
    "local-invariant",
)


class ManifestError(ValueError):
    """The coverage manifest is stale or malformed."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _portable_operators(profile: str, members: tuple[str, ...]) -> frozenset[str]:
    generated_members = PROFILE_MEMBERS.get(profile)
    if generated_members is None:
        raise ManifestError(f"generated registry has no profile {profile!r}")
    if members != tuple(generated_members):
        raise ManifestError(
            "profile_members disagree with the generated portable profile: "
            f"manifest={members!r}, generated={tuple(generated_members)!r}"
        )
    missing_targets = [member for member in members if member not in OPS_BY_TARGET]
    if missing_targets:
        raise ManifestError(
            f"generated registry has no target inventory for {missing_targets!r}"
        )
    return frozenset.intersection(
        *(frozenset(OPS_BY_TARGET[member]) for member in members)
    )


def _load_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except ManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot load {path}: {error}") from error


def _validate_case_evidence(
    document: object,
    expected: frozenset[str],
    owner: dict[str, str],
) -> tuple[int, int]:
    if not isinstance(document, dict):
        raise ManifestError("ONNX case manifest root must be an object")
    if (
        document.get("schema") != "volvoxai.onnx-oracle-cases"
        or document.get("version") != 1
    ):
        raise ManifestError("ONNX case manifest has an unsupported schema/version")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ManifestError("ONNX case manifest must contain executable cases")
    case_ids: set[str] = set()
    exercised: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ManifestError("ONNX case manifest contains a non-object case")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ManifestError(f"invalid or duplicate ONNX case id {case_id!r}")
        case_ids.add(case_id)
        if case.get("oracle_class") not in {
            "onnx-direct",
            "onnx-decomposition",
        }:
            raise ManifestError(f"ONNX case {case_id!r} has no admitted oracle class")
        operators = case.get("expected_lowered_ops")
        if (
            not isinstance(operators, list)
            or not operators
            or any(not isinstance(operator, str) or not operator for operator in operators)
        ):
            raise ManifestError(f"ONNX case {case_id!r} has no lowered-op evidence")
        for operator in operators:
            if operator not in expected:
                raise ManifestError(
                    f"ONNX case {case_id!r} exercises non-portable operator {operator!r}"
                )
            classification = owner.get(operator)
            if classification not in {"onnx-direct", "onnx-decomposition"}:
                raise ManifestError(
                    f"ONNX case {case_id!r} claims ORT evidence for {operator!r}, "
                    f"classified as {classification!r}"
                )
            exercised.add(operator)
    return len(case_ids), len(exercised)


def validate_document(
    document: object,
    case_document: object | None = None,
) -> dict[str, int]:
    if not isinstance(document, dict):
        raise ManifestError("manifest root must be an object")
    required_root = {
        "schema_version",
        "source_registry",
        "case_manifest",
        "profile",
        "profile_members",
        "classifications",
    }
    if set(document) != required_root:
        raise ManifestError(
            "manifest root fields must be exactly " + ", ".join(sorted(required_root))
        )
    if document["schema_version"] != 1:
        raise ManifestError("unsupported oracle coverage schema_version")
    if document["source_registry"] != SOURCE_REGISTRY:
        raise ManifestError("source_registry does not name the generated registry")
    if document["case_manifest"] != CASE_MANIFEST:
        raise ManifestError("case_manifest does not name the required ONNX case manifest")
    profile = document["profile"]
    members_value = document["profile_members"]
    if profile != "portable":
        raise ManifestError("oracle strategy inventory must use the portable profile")
    if not isinstance(members_value, list) or any(
        not isinstance(member, str) for member in members_value
    ):
        raise ManifestError("profile and profile_members are malformed")
    members = tuple(members_value)
    expected = _portable_operators(profile, members)

    classifications = document["classifications"]
    if not isinstance(classifications, dict) or set(classifications) != set(
        CLASSIFICATIONS
    ):
        raise ManifestError(
            "classifications must be exactly " + ", ".join(CLASSIFICATIONS)
        )

    owner: dict[str, str] = {}
    counts: dict[str, int] = {}
    for classification in CLASSIFICATIONS:
        entry = classifications[classification]
        if not isinstance(entry, dict) or set(entry) != {"description", "operators"}:
            raise ManifestError(
                f"classification {classification!r} must contain description and operators"
            )
        description = entry["description"]
        operators = entry["operators"]
        if (
            not isinstance(description, str)
            or not description
            or description != description.strip()
        ):
            raise ManifestError(
                f"classification {classification!r} has an invalid description"
            )
        if not isinstance(operators, list) or any(
            not isinstance(operator, str)
            or not operator
            or operator != operator.strip()
            for operator in operators
        ):
            raise ManifestError(
                f"classification {classification!r} has an invalid operator list"
            )
        if operators != sorted(operators):
            raise ManifestError(
                f"classification {classification!r} operators must be sorted"
            )
        if len(operators) != len(set(operators)):
            raise ManifestError(
                f"classification {classification!r} contains duplicate operators"
            )
        for operator in operators:
            previous = owner.get(operator)
            if previous is not None:
                raise ManifestError(
                    f"operator {operator!r} is classified more than once: "
                    f"{previous!r} and {classification!r}"
                )
            owner[operator] = classification
        counts[classification] = len(operators)

    classified = frozenset(owner)
    missing = sorted(expected - classified)
    extra = sorted(classified - expected)
    if missing:
        raise ManifestError("missing portable operators: " + ", ".join(missing))
    if extra:
        raise ManifestError("non-portable operators in manifest: " + ", ".join(extra))
    counts["total"] = len(classified)
    if case_document is None:
        case_document = _load_json(CASE_MANIFEST_PATH)
    case_count, operator_count = _validate_case_evidence(
        case_document, expected, owner
    )
    counts["executable_cases"] = case_count
    counts["executable_operators"] = operator_count
    return counts


def validate_manifest(path: Path = MANIFEST_PATH) -> dict[str, int]:
    document = _load_json(path)
    return validate_document(document)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    arguments = parser.parse_args(argv)
    try:
        counts = validate_manifest(arguments.manifest)
    except ManifestError as error:
        print(f"oracle coverage: FAIL: {error}", file=sys.stderr)
        return 1
    summary = ", ".join(
        f"{classification}={counts[classification]}"
        for classification in CLASSIFICATIONS
    )
    print(
        f"oracle strategy inventory: {counts['total']} portable operators "
        f"({summary}); executable ORT evidence="
        f"{counts['executable_operators']} operators/"
        f"{counts['executable_cases']} cases"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
