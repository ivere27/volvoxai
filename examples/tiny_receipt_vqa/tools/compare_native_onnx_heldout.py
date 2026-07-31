#!/usr/bin/env python3
"""Compare chunked Volvox native TinyReceipt reports with ONNX heldout evidence.

The record oracle supplies the common IDs, questions, answers, and structured
targets.  Every native package label is explicitly mapped to one precision in
one ONNX report.  A mapping with record-level ONNX rows reports exact output,
answer, and selected-family agreement.  A summary-only mapping reports only
aggregate score deltas.

The emitted JSON contains content identities and model artifact identities,
but never copies input or output filesystem paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


NATIVE_REPORT_FORMAT = "volvoxai.tiny-receipt-native-heldout-profile/v1"
ONNX_REPORT_FORMAT = "tiny_receipt_vqa_onnx_heldout_eval_v1"
ONNX_MANIFEST_FORMAT = "tiny_receipt_vqa_split_onnx_v1"
REPORT_FORMAT = "volvoxai.tiny-receipt-native-onnx-comparison/v1"
DEFAULT_OUTPUT = Path("build/tiny-receipt-native-onnx-comparison.json")
CURRENT_EXPECTED_COUNT = 2000
PRECISIONS = frozenset(("fp32", "int8"))
METRICS = ("answer_exact", "recomputed_answer_exact", "target_exact")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ComparisonError(RuntimeError):
    """The native or ONNX evidence cannot be compared safely."""


class _DuplicateJsonKey(ValueError):
    pass


@dataclass(frozen=True)
class ReportSource:
    document: Mapping[str, Any]
    identity: Mapping[str, object]


@dataclass(frozen=True)
class ComparisonSpec:
    label: str
    precision: str
    report: ReportSource


@dataclass(frozen=True)
class ScoredPrediction:
    prediction: str
    predicted_answer: str
    recomputed_answer: str
    answer_exact: bool
    recomputed_answer_exact: bool
    target_exact: bool


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_source(path: Path, label: str) -> ReportSource:
    try:
        if not path.is_file():
            raise OSError("not a regular file")
        payload = path.read_bytes()
    except OSError as error:
        raise ComparisonError(f"could not read {label}: {error}") from error
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ComparisonError(f"could not parse {label}: {error}") from error
    if not isinstance(document, dict):
        raise ComparisonError(f"{label} must contain a JSON object")
    return ReportSource(
        document=document,
        identity={
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    )


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ComparisonError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ComparisonError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ComparisonError(f"{label} must be a string")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ComparisonError(f"{label} must be a positive integer")
    return value


def _non_negative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ComparisonError(f"{label} must be a non-negative integer")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ComparisonError(f"{label} must be a boolean")
    return value


def _nullable_string(value: Any, label: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ComparisonError(f"{label} must be a string or null")
    return value


def _rate(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"{label} must be a finite rate")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ComparisonError(f"{label} must be in [0, 1]")
    return result


def _content_identity(value: Any, label: str) -> dict[str, object]:
    source = _object(value, label)
    size = source.get("bytes")
    digest = source.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ComparisonError(f"{label}.bytes must be a non-negative integer")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ComparisonError(f"{label}.sha256 must be a lowercase SHA-256 digest")
    return {"bytes": size, "sha256": digest}


def _artifact_identity(value: Any, label: str) -> dict[str, object]:
    source = _object(value, label)
    filename = _string(source.get("filename"), f"{label}.filename")
    path = PurePosixPath(filename)
    if (
        "\\" in filename
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name != filename
    ):
        raise ComparisonError(f"{label}.filename must be a plain filename")
    return {"filename": filename, **_content_identity(source, label)}


def _native_asset_identity(
    value: Any, label: str, expected_path: str
) -> dict[str, object]:
    source = _object(value, label)
    path_value = _string(source.get("path"), f"{label}.path")
    path = PurePosixPath(path_value)
    if (
        "\\" in path_value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != expected_path
    ):
        raise ComparisonError(f"{label}.path must equal {expected_path!r}")
    return {"path": path_value, **_content_identity(source, label)}


def clean_text(value: object) -> str:
    """Match the producer evaluator's NFC and whitespace normalization."""

    return unicodedata.normalize(
        "NFC",
        re.sub(
            r"\s+",
            " ",
            str(value if value is not None else "").replace("\n", " "),
        ).strip(),
    )


def extract_answer(text: str) -> str:
    """Match the ONNX evaluator, including its no-tag fallback."""

    match = re.search(r"<answer>(.*?)</answer>", text)
    return clean_text(match.group(1)) if match else clean_text(text)


def _extract_native_answer(text: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if len(matches) != 1 or "<" in matches[0]:
        return None
    return matches[0].strip()


def recompute_answer(question: str, generated: str) -> str:
    """Recompute phone/address answers exactly like eval_onnx_heldout.py."""

    values: dict[str, str] = {}
    for name in ("field", "value", "op"):
        match = re.search(fr"<{name}>(.*?)</{name}>", generated)
        values[name] = clean_text(match.group(1)) if match else ""
    field = values["field"]
    value = values["value"]
    operation = values["op"]
    digits = re.sub(r"\D+", "", value)
    normalized_question = clean_text(question).lower()
    if "phone number" in normalized_question or "전화번호" in normalized_question:
        if field and field != "phone":
            return ""
        match = re.match(r"(front|back)_(\d+)", operation)
        if not match or not digits:
            return ""
        index = int(match.group(2))
        if index < 1 or index > len(digits):
            return ""
        position = index - 1 if match.group(1) == "front" else len(digits) - index
        return digits[position] if 0 <= position < len(digits) else ""
    if (
        "location" in normalized_question
        or "address" in normalized_question
        or "가게 위치" in normalized_question
        or "빈 칸" in normalized_question
        or "blank" in normalized_question
    ):
        if field and field != "addr":
            return ""
        numbers = re.findall(r"\d+", value)
        return (numbers[-1].lstrip("0") or "0") if numbers else ""
    return ""


def score_prediction(
    *, question: str, truth: str, target: str, generated_text: str
) -> ScoredPrediction:
    prediction = clean_text(generated_text)
    answer = clean_text(truth)
    canonical_target = clean_text(target)
    predicted_answer = extract_answer(prediction)
    recomputed = recompute_answer(question, prediction)
    return ScoredPrediction(
        prediction=prediction,
        predicted_answer=predicted_answer,
        recomputed_answer=recomputed,
        answer_exact=clean_text(predicted_answer) == answer,
        recomputed_answer_exact=clean_text(recomputed) == answer,
        target_exact=prediction == canonical_target,
    )


def _metric_summary(scores: Sequence[ScoredPrediction]) -> dict[str, float | int]:
    if not scores:
        raise ComparisonError("cannot summarize an empty score set")
    count = len(scores)
    return {
        "n": count,
        "answer_exact": sum(score.answer_exact for score in scores) / count,
        "recomputed_n": count,
        "recomputed_answer_exact": sum(
            score.recomputed_answer_exact for score in scores
        )
        / count,
        "target_exact": sum(score.target_exact for score in scores) / count,
    }


def _declared_summary(document: Mapping[str, Any], precision: str) -> dict[str, Any]:
    summary = _object(document.get("summary"), "ONNX report summary")
    precision_summary = _object(
        summary.get(precision), f"ONNX report summary.{precision}"
    )
    overall = _object(
        precision_summary.get("overall"),
        f"ONNX report summary.{precision}.overall",
    )
    count = _positive_integer(overall.get("n"), f"summary.{precision}.overall.n")
    recomputed_count = _positive_integer(
        overall.get("recomputed_n"),
        f"summary.{precision}.overall.recomputed_n",
    )
    if recomputed_count != count:
        raise ComparisonError(
            f"summary.{precision}.overall recomputed_n must equal n"
        )
    return {
        "n": count,
        "answer_exact": _rate(
            overall.get("answer_exact"), f"summary.{precision}.overall.answer_exact"
        ),
        "recomputed_n": recomputed_count,
        "recomputed_answer_exact": _rate(
            overall.get("recomputed_answer_exact"),
            f"summary.{precision}.overall.recomputed_answer_exact",
        ),
        "target_exact": _rate(
            overall.get("target_exact"), f"summary.{precision}.overall.target_exact"
        ),
    }


def _validate_onnx_document(
    document: Mapping[str, Any], label: str, expected_count: int
) -> int:
    if document.get("format") != ONNX_REPORT_FORMAT:
        raise ComparisonError(f"{label} has an unsupported format")
    if document.get("manifest_format") != ONNX_MANIFEST_FORMAT:
        raise ComparisonError(f"{label} has an unsupported manifest_format")
    count = _positive_integer(document.get("records_total"), f"{label}.records_total")
    if count != expected_count:
        raise ComparisonError(
            f"{label}.records_total must equal the current contract count "
            f"{expected_count}"
        )
    if document.get("skipped") != {}:
        raise ComparisonError(f"{label}.skipped must be an empty object")
    if document.get("routing") != "auto":
        raise ComparisonError(f"{label}.routing must be 'auto'")
    if document.get("target_mode") != "rationale":
        raise ComparisonError(f"{label}.target_mode must be 'rationale'")
    if document.get("max_length") != 192:
        raise ComparisonError(f"{label}.max_length must equal 192")
    return count


def _score_onnx_row(row: Mapping[str, Any], label: str) -> ScoredPrediction:
    question = _string(row.get("question"), f"{label}.question")
    truth = _string(row.get("answer"), f"{label}.answer")
    target = _string(row.get("target"), f"{label}.target")
    prediction = _string(row.get("pred"), f"{label}.pred")
    score = score_prediction(
        question=question,
        truth=truth,
        target=target,
        generated_text=prediction,
    )
    expected_values = {
        "pred_answer": score.predicted_answer,
        "recomputed_answer": score.recomputed_answer,
        "answer_exact": score.answer_exact,
        "recomputed_answer_exact": score.recomputed_answer_exact,
        "target_exact": score.target_exact,
    }
    for key, expected in expected_values.items():
        if row.get(key) != expected:
            raise ComparisonError(f"{label}.{key} disagrees with canonical scoring")
    return score


def _record_rows(
    document: Mapping[str, Any], precision: str, label: str
) -> tuple[dict[str, Mapping[str, Any]], dict[str, ScoredPrediction]] | None:
    records = document.get("records")
    if records is None:
        return None
    records_object = _object(records, f"{label}.records")
    values = records_object.get(precision)
    if values is None:
        return None
    rows = _array(values, f"{label}.records.{precision}")
    expected_count = _positive_integer(
        document.get("records_total"), f"{label}.records_total"
    )
    if len(rows) != expected_count:
        raise ComparisonError(
            f"{label}.records.{precision} length does not match records_total"
        )
    by_id: dict[str, Mapping[str, Any]] = {}
    scores: dict[str, ScoredPrediction] = {}
    for index, raw_row in enumerate(rows):
        row_label = f"{label}.records.{precision}[{index}]"
        row = _object(raw_row, row_label)
        case_id = _string(row.get("id"), f"{row_label}.id")
        if not _ID_RE.fullmatch(case_id):
            raise ComparisonError(f"{row_label}.id is invalid")
        if case_id in by_id:
            raise ComparisonError(f"{label} contains duplicate ONNX ID {case_id}")
        _string(row.get("selected_family"), f"{row_label}.selected_family")
        by_id[case_id] = row
        scores[case_id] = _score_onnx_row(row, row_label)
    return by_id, scores


def _validate_declared_against_rows(
    declared: Mapping[str, Any], scores: Mapping[str, ScoredPrediction], label: str
) -> None:
    computed = _metric_summary(list(scores.values()))
    for key in ("n", "recomputed_n"):
        if declared[key] != computed[key]:
            raise ComparisonError(f"{label} {key} disagrees with record rows")
    for key in METRICS:
        if not math.isclose(
            float(declared[key]), float(computed[key]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ComparisonError(f"{label} {key} disagrees with record rows")


def _selected_artifacts(
    document: Mapping[str, Any], precision: str, label: str
) -> dict[str, object]:
    artifacts = _object(document.get("artifacts"), f"{label}.artifacts")
    selected = _object(artifacts.get(precision), f"{label}.artifacts.{precision}")
    return {
        graph: _artifact_identity(
            selected.get(graph), f"{label}.artifacts.{precision}.{graph}"
        )
        for graph in ("encoder", "decoder")
    }


def _native_settings(document: Mapping[str, Any], label: str) -> dict[str, Any]:
    settings = dict(_object(document.get("settings"), f"{label}.settings"))
    case_count = _positive_integer(settings.get("case_count"), f"{label}.case_count")
    _positive_integer(settings.get("repeat"), f"{label}.repeat")
    required = {
        "backend": "cpu",
        "decode": "incremental-row-required",
        "application_timing": True,
        "runtime_node_trace": False,
        "max_new": 191,
        "accuracy_mode": "full-generation",
    }
    for key, expected in required.items():
        if settings.get(key) != expected:
            raise ComparisonError(
                f"{label}.settings.{key} must equal {expected!r}"
            )
    settings.pop("case_count", None)
    settings.pop("case_selection", None)
    settings["_validated_case_count"] = case_count
    return settings


def _native_identity(document: Mapping[str, Any], label: str) -> tuple[Any, Any]:
    provenance = _object(document.get("provenance"), f"{label}.provenance")
    executable = _object(provenance.get("executable"), f"{label}.executable")
    packages = _array(provenance.get("packages"), f"{label}.packages")
    return executable, packages


def _validate_native_artifact(
    value: Any,
    *,
    truth: str,
    repeat_count: int,
    label: str,
) -> tuple[str, str | None, str, bool]:
    artifact = _object(value, label)
    generated_text = _string(
        artifact.get("generated_text"), f"{label}.generated_text"
    )
    if "answer" not in artifact:
        raise ComparisonError(f"{label}.answer is required")
    answer = _nullable_string(artifact.get("answer"), f"{label}.answer")
    if answer != _extract_native_answer(generated_text):
        raise ComparisonError(f"{label}.answer disagrees with generated_text")
    selected_family = _string(
        artifact.get("selected_family"), f"{label}.selected_family"
    )
    exact_match = _boolean(artifact.get("exact_match"), f"{label}.exact_match")
    if exact_match != (answer == truth):
        raise ComparisonError(f"{label}.exact_match disagrees with answer and truth")

    runs = _array(artifact.get("runs"), f"{label}.runs")
    if len(runs) != repeat_count:
        raise ComparisonError(f"{label}.runs length does not match settings.repeat")
    router_families: set[str] = set()
    for run_index, raw_run in enumerate(runs):
        run_label = f"{label}.runs[{run_index}]"
        run = _object(raw_run, run_label)
        run_repeat = _positive_integer(run.get("repeat"), f"{run_label}.repeat")
        if run_repeat != run_index + 1:
            raise ComparisonError(
                f"{run_label}.repeat must equal its one-based run index"
            )
        if _string(run.get("generated_text"), f"{run_label}.generated_text") != generated_text:
            raise ComparisonError(
                f"{run_label}.generated_text disagrees with the artifact result"
            )
        if "answer" not in run:
            raise ComparisonError(f"{run_label}.answer is required")
        if _nullable_string(run.get("answer"), f"{run_label}.answer") != answer:
            raise ComparisonError(f"{run_label}.answer disagrees with the artifact result")
        if (
            _string(run.get("selected_family"), f"{run_label}.selected_family")
            != selected_family
        ):
            raise ComparisonError(
                f"{run_label}.selected_family disagrees with the artifact result"
            )
        router_families.add(
            _string(run.get("router_family"), f"{run_label}.router_family")
        )
        if _boolean(run.get("limit_exhausted"), f"{run_label}.limit_exhausted"):
            raise ComparisonError(
                f"{run_label}.limit_exhausted is invalid for full-generation evidence"
            )
    if len(router_families) != 1:
        raise ComparisonError(f"{label}.runs disagree on router_family")
    return generated_text, answer, selected_family, exact_match


def _validate_native_stored_summary(
    document: Mapping[str, Any],
    *,
    package_order: Sequence[str],
    package_correct: Mapping[str, int],
    answer_agree: int,
    structured_agree: int,
    total: int,
    label: str,
) -> None:
    summary = _object(document.get("summary"), f"{label}.summary")
    packages = _object(summary.get("packages"), f"{label}.summary.packages")
    if set(packages) != set(package_order):
        raise ComparisonError(
            f"{label}.summary.packages do not match package_order"
        )
    for package_label in package_order:
        package = _object(
            packages.get(package_label),
            f"{label}.summary.packages.{package_label}",
        )
        exact = _object(
            package.get("exact_match"),
            f"{label}.summary.packages.{package_label}.exact_match",
        )
        prefix = f"{label}.summary.packages.{package_label}.exact_match"
        correct = _non_negative_integer(exact.get("correct"), f"{prefix}.correct")
        declared_total = _positive_integer(exact.get("total"), f"{prefix}.total")
        rate = _rate(exact.get("rate"), f"{prefix}.rate")
        if correct != package_correct[package_label] or declared_total != total:
            raise ComparisonError(f"{prefix} counts disagree with native cases")
        if not math.isclose(rate, correct / total, rel_tol=0.0, abs_tol=1e-12):
            raise ComparisonError(f"{prefix}.rate disagrees with native cases")

    agreement = _object(
        summary.get("cross_artifact_agreement"),
        f"{label}.summary.cross_artifact_agreement",
    )
    for name, expected in (
        ("answer", answer_agree),
        ("structured_text", structured_agree),
    ):
        counts = _object(
            agreement.get(name),
            f"{label}.summary.cross_artifact_agreement.{name}",
        )
        prefix = f"{label}.summary.cross_artifact_agreement.{name}"
        agree = _non_negative_integer(counts.get("agree"), f"{prefix}.agree")
        declared_total = _positive_integer(counts.get("total"), f"{prefix}.total")
        if agree != expected or declared_total != total:
            raise ComparisonError(f"{prefix} counts disagree with native cases")


def _merge_native_reports(
    reports: Sequence[ReportSource],
) -> tuple[
    list[Mapping[str, Any]],
    list[str],
    Mapping[str, Any],
    Sequence[Any],
    Mapping[str, Any],
]:
    if not reports:
        raise ComparisonError("at least one native report is required")
    merged: dict[str, Mapping[str, Any]] = {}
    reference_settings: dict[str, Any] | None = None
    reference_executable: Mapping[str, Any] | None = None
    reference_packages: Sequence[Any] | None = None
    public_settings: Mapping[str, Any] | None = None
    package_order: list[str] | None = None
    for report_index, report in enumerate(reports):
        document = report.document
        label = f"native report {report_index + 1}"
        if document.get("format") != NATIVE_REPORT_FORMAT:
            raise ComparisonError(f"{label} has an unsupported format")
        settings = _native_settings(document, label)
        declared_count = settings.pop("_validated_case_count")
        raw_public_settings = _object(document.get("settings"), f"{label}.settings")
        repeat_count = _positive_integer(
            raw_public_settings.get("repeat"), f"{label}.settings.repeat"
        )
        order = _array(raw_public_settings.get("package_order"), f"{label}.package_order")
        current_order = [_string(value, f"{label}.package_order") for value in order]
        if not current_order or len(set(current_order)) != len(current_order):
            raise ComparisonError(f"{label}.package_order must be non-empty and unique")
        executable, packages = _native_identity(document, label)
        if reference_settings is None:
            reference_settings = settings
            reference_executable = executable
            reference_packages = packages
            public_settings = {
                key: value
                for key, value in raw_public_settings.items()
                if key not in ("case_count", "case_selection")
            }
            package_order = current_order
        else:
            if settings != reference_settings:
                raise ComparisonError(
                    f"{label} settings differ from the first native report"
                )
            if executable != reference_executable:
                raise ComparisonError(
                    f"{label} executable identity differs from the first native report"
                )
            if packages != reference_packages:
                raise ComparisonError(
                    f"{label} package identities differ from the first native report"
                )
            if current_order != package_order:
                raise ComparisonError(
                    f"{label} package order differs from the first native report"
                )
        cases = _array(document.get("cases"), f"{label}.cases")
        if len(cases) != declared_count:
            raise ComparisonError(f"{label} case_count does not match cases length")
        provenance = _object(document.get("provenance"), f"{label}.provenance")
        heldout_cases = _array(
            provenance.get("heldout_cases"), f"{label}.provenance.heldout_cases"
        )
        if len(heldout_cases) != declared_count:
            raise ComparisonError(
                f"{label} heldout_cases length does not match case_count"
            )
        heldout_by_id: dict[str, Mapping[str, Any]] = {}
        for heldout_index, raw_heldout in enumerate(heldout_cases):
            heldout_label = f"{label}.provenance.heldout_cases[{heldout_index}]"
            heldout = _object(raw_heldout, heldout_label)
            heldout_id = _string(heldout.get("id"), f"{heldout_label}.id")
            if not _ID_RE.fullmatch(heldout_id):
                raise ComparisonError(f"{heldout_label}.id is invalid")
            if heldout_id in heldout_by_id:
                raise ComparisonError(
                    f"{label} contains duplicate heldout provenance ID {heldout_id}"
                )
            heldout_by_id[heldout_id] = heldout
        package_correct = {package_label: 0 for package_label in current_order}
        answer_agree_count = 0
        structured_agree_count = 0
        for case_index, raw_case in enumerate(cases):
            case_label = f"{label}.cases[{case_index}]"
            case = _object(raw_case, case_label)
            case_id = _string(case.get("id"), f"{case_label}.id")
            if not _ID_RE.fullmatch(case_id):
                raise ComparisonError(f"{case_label}.id is invalid")
            if case_id in merged:
                raise ComparisonError(f"duplicate native case ID {case_id}")
            _string(case.get("question"), f"{case_label}.question")
            _string(case.get("truth"), f"{case_label}.truth")
            artifacts = _object(case.get("artifacts"), f"{case_label}.artifacts")
            if set(artifacts) != set(current_order):
                raise ComparisonError(
                    f"{case_label}.artifacts do not match package_order"
                )
            heldout = heldout_by_id.get(case_id)
            if heldout is None:
                raise ComparisonError(
                    f"{case_label} is missing matching heldout provenance"
                )
            data = _object(case.get("data"), f"{case_label}.data")
            for asset in ("annotation", "image"):
                if data.get(asset) != heldout.get(asset):
                    raise ComparisonError(
                        f"{case_label}.{asset} differs from heldout provenance"
                    )
                suffix = "json" if asset == "annotation" else "jpg"
                _native_asset_identity(
                    data.get(asset),
                    f"{case_label}.data.{asset}",
                    f"{'annotations' if asset == 'annotation' else 'images'}/"
                    f"{case_id}.{suffix}",
                )

            generated_values: list[str] = []
            answer_values: list[str | None] = []
            for package_label in current_order:
                generated, answer, _family, exact_match = _validate_native_artifact(
                    artifacts.get(package_label),
                    truth=str(case["truth"]),
                    repeat_count=repeat_count,
                    label=f"{case_label}.artifacts.{package_label}",
                )
                generated_values.append(generated)
                answer_values.append(answer)
                package_correct[package_label] += int(exact_match)
            expected_answer_agreement = answer_values[0] is not None and all(
                answer == answer_values[0] for answer in answer_values[1:]
            )
            expected_structured_agreement = all(
                generated == generated_values[0] for generated in generated_values[1:]
            )
            stored_agreement = _object(
                case.get("cross_artifact_agreement"),
                f"{case_label}.cross_artifact_agreement",
            )
            stored_answer_agreement = _boolean(
                stored_agreement.get("answer"),
                f"{case_label}.cross_artifact_agreement.answer",
            )
            stored_structured_agreement = _boolean(
                stored_agreement.get("structured_text"),
                f"{case_label}.cross_artifact_agreement.structured_text",
            )
            if stored_answer_agreement != expected_answer_agreement:
                raise ComparisonError(
                    f"{case_label}.cross_artifact_agreement.answer disagrees "
                    "with artifacts"
                )
            if stored_structured_agreement != expected_structured_agreement:
                raise ComparisonError(
                    f"{case_label}.cross_artifact_agreement.structured_text "
                    "disagrees with artifacts"
                )
            answer_agree_count += int(expected_answer_agreement)
            structured_agree_count += int(expected_structured_agreement)
            merged[case_id] = case
        if set(heldout_by_id) != {str(case["id"]) for case in cases}:
            raise ComparisonError(
                f"{label} heldout provenance IDs do not match native case IDs"
            )
        _validate_native_stored_summary(
            document,
            package_order=current_order,
            package_correct=package_correct,
            answer_agree=answer_agree_count,
            structured_agree=structured_agree_count,
            total=declared_count,
            label=label,
        )
    assert package_order is not None
    assert reference_executable is not None
    assert reference_packages is not None
    assert public_settings is not None
    return (
        [merged[case_id] for case_id in sorted(merged)],
        package_order,
        reference_executable,
        reference_packages,
        public_settings,
    )


def _public_native_execution(
    executable: Mapping[str, Any], packages: Sequence[Any], package_order: Sequence[str]
) -> dict[str, object]:
    package_by_label: dict[str, Mapping[str, Any]] = {}
    for index, raw_package in enumerate(packages):
        package = _object(raw_package, f"native package identity {index}")
        label = _string(package.get("label"), f"native package identity {index}.label")
        if label in package_by_label:
            raise ComparisonError(f"duplicate native package identity label {label}")
        package_by_label[label] = package
    if set(package_by_label) != set(package_order):
        raise ComparisonError("native package identities do not match package_order")
    result_packages = []
    for label in package_order:
        package = package_by_label[label]
        result_packages.append(
            {
                "label": label,
                "manifest": _content_identity(
                    package.get("manifest"), f"native package {label}.manifest"
                ),
            }
        )
    return {
        "executable": _content_identity(executable, "native executable"),
        "packages": result_packages,
    }


def _metric_delta(
    native: Mapping[str, Any], onnx: Mapping[str, Any]
) -> dict[str, float]:
    return {key: float(native[key]) - float(onnx[key]) for key in METRICS}


def _agreement_summary(values: Sequence[bool]) -> dict[str, float | int]:
    total = len(values)
    agree = sum(values)
    return {"agree": agree, "total": total, "rate": agree / total}


def _has_unsafe_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            _has_unsafe_path(key) or _has_unsafe_path(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_unsafe_path(item) for item in value)
    if isinstance(value, str):
        stripped = value.lstrip()
        return (
            stripped.startswith("/")
            or "\\" in value
            or stripped.lower().startswith("file:")
            or bool(re.match(r"^[A-Za-z]:[\\/]", stripped))
        )
    return False


def _verify_eval_assets(
    cases: Sequence[Mapping[str, Any]], eval_root: Path
) -> int:
    try:
        resolved_root = eval_root.resolve(strict=True)
    except OSError as error:
        raise ComparisonError(f"could not resolve --eval-root: {error}") from error
    if not resolved_root.is_dir():
        raise ComparisonError("--eval-root must be a directory")
    for case in cases:
        case_id = str(case["id"])
        data = _object(case.get("data"), f"native case {case_id}.data")
        for asset, relative_path in (
            ("annotation", f"annotations/{case_id}.json"),
            ("image", f"images/{case_id}.jpg"),
        ):
            reported = _native_asset_identity(
                data.get(asset),
                f"native case {case_id}.data.{asset}",
                relative_path,
            )
            try:
                candidate = (resolved_root / relative_path).resolve(strict=True)
                candidate.relative_to(resolved_root)
                if not candidate.is_file():
                    raise OSError("not a regular file")
                payload = candidate.read_bytes()
            except (OSError, ValueError) as error:
                raise ComparisonError(
                    f"could not verify {asset} for native case {case_id}: {error}"
                ) from error
            actual = {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            if actual != {key: reported[key] for key in ("bytes", "sha256")}:
                raise ComparisonError(
                    f"{asset} content differs from the native report for case {case_id}"
                )
    return len(cases)


def build_report(
    *,
    native_reports: Sequence[ReportSource],
    record_oracle: ReportSource,
    record_precision: str,
    comparisons: Sequence[ComparisonSpec],
    allow_partial: bool = False,
    expected_count: int = CURRENT_EXPECTED_COUNT,
    eval_root: Path | None = None,
) -> Mapping[str, object]:
    expected_count = _positive_integer(expected_count, "expected_count")
    if record_precision not in PRECISIONS:
        raise ComparisonError(f"unsupported record-oracle precision {record_precision!r}")
    if not comparisons:
        raise ComparisonError("at least one comparison mapping is required")
    labels = [comparison.label for comparison in comparisons]
    if any(not _LABEL_RE.fullmatch(label) for label in labels):
        raise ComparisonError("comparison labels must be simple package labels")
    if len(set(labels)) != len(labels):
        raise ComparisonError("comparison labels must be unique")
    if any(comparison.precision not in PRECISIONS for comparison in comparisons):
        raise ComparisonError("comparison precision must be fp32 or int8")

    cases, package_order, executable, packages, native_settings = _merge_native_reports(
        native_reports
    )
    if set(labels) != set(package_order):
        missing = sorted(set(package_order) - set(labels))
        extra = sorted(set(labels) - set(package_order))
        raise ComparisonError(
            f"comparison mappings do not match native labels; missing={missing}, extra={extra}"
        )

    oracle_count = _validate_onnx_document(
        record_oracle.document, "record oracle", expected_count
    )
    oracle_declared = _declared_summary(record_oracle.document, record_precision)
    oracle_rows_result = _record_rows(
        record_oracle.document, record_precision, "record oracle"
    )
    if oracle_rows_result is None:
        raise ComparisonError("record oracle precision has no record-level rows")
    oracle_rows, oracle_scores = oracle_rows_result
    if len(oracle_rows) != oracle_count or oracle_declared["n"] != oracle_count:
        raise ComparisonError("record oracle count is inconsistent")
    _validate_declared_against_rows(
        oracle_declared, oracle_scores, "record oracle summary"
    )
    oracle_summary = _object(record_oracle.document.get("summary"), "record oracle summary")
    oracle_fp32_summary = _object(
        oracle_summary.get("fp32"), "record oracle summary.fp32"
    )
    oracle_fp32_declared = _declared_summary(record_oracle.document, "fp32")
    if oracle_fp32_declared["n"] != oracle_count:
        raise ComparisonError("record oracle FP32 summary count is inconsistent")
    oracle_artifacts = _object(
        record_oracle.document.get("artifacts"), "record oracle artifacts"
    )
    oracle_fp32_artifacts_raw = _object(
        oracle_artifacts.get("fp32"), "record oracle artifacts.fp32"
    )
    oracle_fp32_artifacts = _selected_artifacts(
        record_oracle.document, "fp32", "record oracle"
    )
    native_ids = {str(case["id"]) for case in cases}
    oracle_ids = set(oracle_rows)
    extra_ids = sorted(native_ids - oracle_ids)
    missing_ids = sorted(oracle_ids - native_ids)
    if extra_ids:
        raise ComparisonError(f"native reports contain IDs absent from record oracle: {extra_ids}")
    if missing_ids and not allow_partial:
        raise ComparisonError(f"native reports are missing record-oracle IDs: {missing_ids}")

    for case in cases:
        case_id = str(case["id"])
        oracle = oracle_rows[case_id]
        if case["question"] != oracle["question"]:
            raise ComparisonError(f"native question differs from record oracle for ID {case_id}")
        if case["truth"] != oracle["answer"]:
            raise ComparisonError(f"native truth differs from record oracle for ID {case_id}")

    verified_eval_cases = 0
    if eval_root is not None:
        verified_eval_cases = _verify_eval_assets(cases, eval_root)

    comparison_data: dict[str, dict[str, Any]] = {}
    provenance_comparisons: list[dict[str, Any]] = []
    for comparison in comparisons:
        label = comparison.label
        precision = comparison.precision
        document = comparison.report.document
        report_label = f"comparison {label}"
        report_count = _validate_onnx_document(
            document, report_label, expected_count
        )
        declared = _declared_summary(document, precision)
        if report_count != oracle_count or declared["n"] != oracle_count:
            raise ComparisonError(
                f"{report_label} count does not match the record oracle"
            )
        row_result = _record_rows(document, precision, report_label)
        rows: dict[str, Mapping[str, Any]] | None = None
        scores: dict[str, ScoredPrediction] | None = None
        mode = "aggregate-only"
        if row_result is not None:
            rows, scores = row_result
            if set(rows) != oracle_ids:
                raise ComparisonError(f"{report_label} record IDs differ from record oracle")
            for case_id, row in rows.items():
                oracle = oracle_rows[case_id]
                for key in ("question", "answer", "target"):
                    if row[key] != oracle[key]:
                        raise ComparisonError(
                            f"{report_label} {key} differs from record oracle for ID {case_id}"
                        )
            _validate_declared_against_rows(
                declared, scores, f"{report_label} summary"
            )
            mode = "record-level"
        else:
            if document.get("records") is not None:
                raise ComparisonError(
                    f"{report_label} records omit selected precision {precision}"
                )
            privacy = _object(document.get("privacy"), f"{report_label}.privacy")
            if privacy.get("summary_only") is not True:
                raise ComparisonError(
                    f"{report_label}.privacy.summary_only must be true when "
                    "record rows are absent"
                )
            report_summary = _object(
                document.get("summary"), f"{report_label}.summary"
            )
            report_fp32_summary = _object(
                report_summary.get("fp32"), f"{report_label}.summary.fp32"
            )
            report_fp32_declared = _declared_summary(document, "fp32")
            if report_fp32_declared["n"] != oracle_count:
                raise ComparisonError(
                    f"{report_label} FP32 summary count does not match the oracle"
                )
            if report_fp32_summary != oracle_fp32_summary:
                raise ComparisonError(
                    f"{report_label} FP32 summary does not exactly match the "
                    "record oracle"
                )
            report_artifacts = _object(
                document.get("artifacts"), f"{report_label}.artifacts"
            )
            report_fp32_artifacts_raw = _object(
                report_artifacts.get("fp32"), f"{report_label}.artifacts.fp32"
            )
            report_fp32_artifacts = _selected_artifacts(
                document, "fp32", report_label
            )
            if (
                report_fp32_artifacts_raw != oracle_fp32_artifacts_raw
                or report_fp32_artifacts != oracle_fp32_artifacts
            ):
                raise ComparisonError(
                    f"{report_label} FP32 artifacts do not match the record oracle"
                )
        comparison_data[label] = {
            "precision": precision,
            "declared": declared,
            "rows": rows,
            "scores": scores,
            "mode": mode,
        }
        provenance_comparisons.append(
            {
                "label": label,
                "precision": precision,
                "mode": mode,
                "report": _content_identity(
                    comparison.report.identity, f"{report_label} report identity"
                ),
                "artifacts": _selected_artifacts(document, precision, report_label),
            }
        )

    per_label_scores: dict[str, list[ScoredPrediction]] = {
        label: [] for label in labels
    }
    per_label_agreements: dict[str, dict[str, list[bool]]] = {
        label: {"prediction": [], "answer": [], "selected_family": []}
        for label in labels
    }
    case_results: list[dict[str, object]] = []
    for case in cases:
        case_id = str(case["id"])
        oracle = oracle_rows[case_id]
        artifacts = _object(case["artifacts"], f"native case {case_id}.artifacts")
        label_results: dict[str, object] = {}
        for label in labels:
            artifact = _object(artifacts.get(label), f"native case {case_id}.{label}")
            generated = _string(
                artifact.get("generated_text"),
                f"native case {case_id}.{label}.generated_text",
            )
            selected_family = _string(
                artifact.get("selected_family"),
                f"native case {case_id}.{label}.selected_family",
            )
            score = score_prediction(
                question=str(case["question"]),
                truth=str(case["truth"]),
                target=str(oracle["target"]),
                generated_text=generated,
            )
            per_label_scores[label].append(score)
            metrics = {key: bool(getattr(score, key)) for key in METRICS}
            agreement: dict[str, bool] | None = None
            comparison_rows = comparison_data[label]["rows"]
            comparison_scores = comparison_data[label]["scores"]
            if comparison_rows is not None and comparison_scores is not None:
                expected_row = comparison_rows[case_id]
                expected_score = comparison_scores[case_id]
                agreement = {
                    "prediction": score.prediction == expected_score.prediction,
                    "answer": score.predicted_answer
                    == expected_score.predicted_answer,
                    "selected_family": selected_family
                    == expected_row["selected_family"],
                }
                for key, value in agreement.items():
                    per_label_agreements[label][key].append(value)
            label_results[label] = {
                "metrics": metrics,
                "onnx_agreement": agreement,
            }
        case_results.append({"id": case_id, "labels": label_results})

    summary_labels: dict[str, object] = {}
    complete = not missing_ids
    selected_ids = [str(case["id"]) for case in cases]
    for label in labels:
        native_metrics = _metric_summary(per_label_scores[label])
        data = comparison_data[label]
        rows = data["rows"]
        scores = data["scores"]
        if rows is not None and scores is not None:
            onnx_metrics: Mapping[str, Any] = _metric_summary(
                [scores[case_id] for case_id in selected_ids]
            )
            delta: Mapping[str, float] | None = _metric_delta(
                native_metrics, onnx_metrics
            )
            agreement = {
                key: _agreement_summary(values)
                for key, values in per_label_agreements[label].items()
            }
            onnx_scope = "all-records" if complete else "selected-native-records"
        else:
            onnx_metrics = data["declared"]
            delta = _metric_delta(native_metrics, onnx_metrics) if complete else None
            agreement = None
            onnx_scope = "all-records"
        summary_labels[label] = {
            "mode": data["mode"],
            "precision": data["precision"],
            "native_metrics": native_metrics,
            "onnx_metrics": dict(onnx_metrics),
            "onnx_metric_scope": onnx_scope,
            "score_delta": delta,
            "agreement": agreement,
        }

    report: dict[str, object] = {
        "format": REPORT_FORMAT,
        "settings": {
            "native_report_count": len(native_reports),
            "case_count": len(cases),
            "allow_partial": allow_partial,
            "native_execution": dict(native_settings),
            "mappings": [
                {
                    "label": item["label"],
                    "precision": item["precision"],
                    "mode": item["mode"],
                }
                for item in provenance_comparisons
            ],
        },
        "coverage": {
            "expected": len(oracle_ids),
            "observed": len(native_ids),
            "complete": complete,
            "missing_ids": missing_ids,
        },
        "validation_scope": {
            "onnx_record_fields": ["id", "question", "answer", "target"],
            "onnx_annotation_image_identities": "unavailable-in-producer-report",
            "native_asset_relative_layout": "validated",
            "native_annotation_image_content": {
                "status": (
                    "verified-against-eval-root"
                    if eval_root is not None
                    else "reported-identities-only"
                ),
                "verified_cases": verified_eval_cases,
            },
        },
        "provenance": {
            "native_reports": [
                _content_identity(report.identity, f"native report {index + 1} identity")
                for index, report in enumerate(native_reports)
            ],
            "native_execution": _public_native_execution(
                executable, packages, package_order
            ),
            "record_oracle": {
                "precision": record_precision,
                "report": _content_identity(
                    record_oracle.identity, "record oracle report identity"
                ),
                "artifacts": _selected_artifacts(
                    record_oracle.document, record_precision, "record oracle"
                ),
            },
            "comparisons": provenance_comparisons,
        },
        "summary": {"labels": summary_labels},
        "cases": case_results,
    }
    if _has_unsafe_path(report):
        raise ComparisonError("comparison report would expose an unsafe path")
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-report",
        action="append",
        required=True,
        type=Path,
        help="native heldout report chunk; repeat for disjoint chunks",
    )
    parser.add_argument(
        "--record-oracle",
        required=True,
        nargs=2,
        metavar=("PRECISION", "ONNX_REPORT"),
        help="record-level ONNX precision and report supplying IDs/truth/targets",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        required=True,
        nargs=3,
        metavar=("NATIVE_LABEL", "ONNX_PRECISION", "ONNX_REPORT"),
        help="explicit native-label to ONNX precision/report mapping; repeat per label",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="emit detected missing oracle IDs; summary-only score deltas become null",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        help=(
            "optionally verify every native annotation and image identity against "
            "the heldout root"
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _print_summary(report: Mapping[str, object]) -> None:
    labels = report["summary"]["labels"]  # type: ignore[index]
    for label, value in labels.items():  # type: ignore[union-attr]
        metrics = value["native_metrics"]
        delta = value["score_delta"]
        delta_text = "unaligned partial summary"
        if delta is not None:
            delta_text = ", ".join(
                f"{name} {float(delta[name]):+.6f}" for name in METRICS
            )
        print(
            f"{label}: n={metrics['n']} answer_exact={metrics['answer_exact']:.6f}; "
            f"delta: {delta_text}",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    cache: dict[Path, ReportSource] = {}

    def load(path_value: str | Path, label: str) -> ReportSource:
        path = Path(path_value)
        key = path.resolve()
        if key not in cache:
            cache[key] = _read_source(path, label)
        return cache[key]

    try:
        input_paths = [
            *arguments.native_report,
            Path(arguments.record_oracle[1]),
            *(Path(item[2]) for item in arguments.comparison),
        ]
        output_target = arguments.out.resolve()
        output_exists = arguments.out.exists()
        if any(
            output_target == path.resolve()
            or (
                output_exists
                and path.exists()
                and arguments.out.samefile(path)
            )
            for path in input_paths
        ):
            raise ComparisonError(
                "--out must not resolve to an input report or its symlink target"
            )
        if arguments.eval_root is not None:
            eval_target = arguments.eval_root.resolve(strict=True)
            try:
                output_target.relative_to(eval_target)
            except ValueError:
                pass
            else:
                raise ComparisonError("--out must stay outside --eval-root")
            if output_exists:
                eval_assets = (
                    *eval_target.glob("annotations/*.json"),
                    *eval_target.glob("images/*.jpg"),
                )
                if any(arguments.out.samefile(path) for path in eval_assets):
                    raise ComparisonError(
                        "--out must not be a hardlink to an --eval-root asset"
                    )
        native_reports = [
            load(path, f"native report {index + 1}")
            for index, path in enumerate(arguments.native_report)
        ]
        record_precision, record_path = arguments.record_oracle
        record_oracle = load(record_path, "record oracle")
        comparisons = [
            ComparisonSpec(
                label=label,
                precision=precision,
                report=load(path, f"comparison {label}"),
            )
            for label, precision, path in arguments.comparison
        ]
        report = build_report(
            native_reports=native_reports,
            record_oracle=record_oracle,
            record_precision=record_precision,
            comparisons=comparisons,
            allow_partial=arguments.allow_partial,
            eval_root=arguments.eval_root,
        )
        encoded = json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(encoded, encoding="utf-8")
        _print_summary(report)
    except (ComparisonError, OSError, UnicodeError) as error:
        print(f"[tinyreceipt-native-onnx-compare] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
