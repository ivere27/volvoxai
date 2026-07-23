from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path


_TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "compare_native_onnx_heldout.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "compare_native_onnx_heldout", _TOOL_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
comparison = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = comparison
_SPEC.loader.exec_module(comparison)


def _digest(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _identity(seed: str, *, path: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {"bytes": len(seed), "sha256": _digest(seed)}
    if path is not None:
        result["path"] = path
    return result


def _artifact(seed: str, filename: str) -> dict[str, object]:
    return {"filename": filename, **_identity(seed)}


def _source(document: dict[str, object]) -> comparison.ReportSource:
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True).encode()
    return comparison.ReportSource(
        document=document,
        identity={"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
    )


def _summary(scores: list[comparison.ScoredPrediction]) -> dict[str, object]:
    return {"overall": comparison._metric_summary(scores)}


def _clone(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _onnx_contract(record_count: int) -> dict[str, object]:
    return {
        "format": comparison.ONNX_REPORT_FORMAT,
        "manifest_format": comparison.ONNX_MANIFEST_FORMAT,
        "records_total": record_count,
        "skipped": {},
        "routing": "auto",
        "target_mode": "rationale",
        "max_length": 192,
    }


class NativeOnnxComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = {
            "00001": {
                "id": "00001",
                "question": "What is the first number of the store's phone number?",
                "answer": "2",
                "target": (
                    "<field>phone</field><value>2586238</value>"
                    "<op>front_1</op><answer>2</answer>"
                ),
                "family": "phone",
                "kind": "phone_number",
            },
            "00002": {
                "id": "00002",
                "question": "The location of the store is Main St. [?]. (Fill the blank)",
                "answer": "10",
                "target": (
                    "<field>addr</field><value>Seoul Main St. 10</value>"
                    "<op>street_no</op><answer>10</answer>"
                ),
                "family": "address",
                "kind": "address",
            },
        }
        self.changed_address = (
            "<field>addr</field><value>Seoul Main St. 11</value>"
            "<op>street_no</op><answer>10</answer>"
        )
        self.record_document = self._record_onnx_report()
        self.summary_document = self._summary_only_onnx_report()

    def test_canonical_text_answer_and_recompute_semantics(self) -> None:
        self.assertEqual(comparison.clean_text("  e\u0301\n  value  "), "é value")
        self.assertEqual(comparison.extract_answer("  fallback answer  "), "fallback answer")
        self.assertEqual(
            comparison.recompute_answer(
                "The store address has a blank.",
                "<field>addr</field><value>Main St. 0010</value><op>street_no</op>",
            ),
            "10",
        )

    def _onnx_row(self, case_id: str) -> dict[str, object]:
        case = self.cases[case_id]
        score = comparison.score_prediction(
            question=str(case["question"]),
            truth=str(case["answer"]),
            target=str(case["target"]),
            generated_text=str(case["target"]),
        )
        return {
            "id": case_id,
            "kind": case["kind"],
            "family": case["family"],
            "selected_family": case["family"],
            "question": case["question"],
            "answer": case["answer"],
            "pred_answer": score.predicted_answer,
            "recomputed_answer": score.recomputed_answer,
            "answer_exact": score.answer_exact,
            "recomputed_answer_exact": score.recomputed_answer_exact,
            "target_exact": score.target_exact,
            "pred": score.prediction,
            "target": case["target"],
            "router_logits": [0.0] * 8,
        }

    def _record_onnx_report(self) -> dict[str, object]:
        rows = [self._onnx_row(case_id) for case_id in sorted(self.cases)]
        scores = [
            comparison.score_prediction(
                question=str(row["question"]),
                truth=str(row["answer"]),
                target=str(row["target"]),
                generated_text=str(row["pred"]),
            )
            for row in rows
        ]
        return {
            **_onnx_contract(len(rows)),
            "artifacts": {
                "fp32": {
                    "encoder": _artifact("encoder-fp32", "encoder_model.onnx"),
                    "decoder": _artifact("decoder-fp32", "decoder_model.onnx"),
                }
            },
            "summary": {"fp32": _summary(scores)},
            "records": {"fp32": rows},
        }

    def _summary_only_onnx_report(self) -> dict[str, object]:
        return {
            **_onnx_contract(2),
            "artifacts": {
                "fp32": _clone(self.record_document["artifacts"]["fp32"]),
                "int8": {
                    "encoder": _artifact("encoder-int8", "encoder_model_int8.onnx"),
                    "decoder": _artifact("decoder-int8", "decoder_model_int8.onnx"),
                }
            },
            "summary": {
                "fp32": _clone(self.record_document["summary"]["fp32"]),
                "int8": {
                    "overall": {
                        "n": 2,
                        "answer_exact": 0.5,
                        "recomputed_n": 2,
                        "recomputed_answer_exact": 0.5,
                        "target_exact": 0.5,
                    }
                }
            },
            "privacy": {"summary_only": True},
        }

    def _native_report(
        self,
        case_ids: list[str],
        *,
        settings_update: dict[str, object] | None = None,
        executable_seed: str = "native-executable",
        package_seed: str = "packages",
    ) -> dict[str, object]:
        labels = ["f32", "w8a8"]
        settings: dict[str, object] = {
            "backend": "cpu",
            "decode": "incremental-row-required",
            "case_selection": "explicit-ids",
            "case_count": len(case_ids),
            "repeat": 1,
            "max_new": 191,
            "accuracy_mode": "full-generation",
            "application_timing": True,
            "runtime_node_trace": False,
            "package_order": labels,
        }
        if settings_update:
            settings.update(settings_update)
        repeat_count = int(settings["repeat"])
        cases = []
        heldout = []
        for case_id in case_ids:
            case = self.cases[case_id]
            annotation = _identity(
                f"annotation-{case_id}", path=f"annotations/{case_id}.json"
            )
            image = _identity(f"image-{case_id}", path=f"images/{case_id}.jpg")
            generated = (
                self.changed_address if case_id == "00002" else str(case["target"])
            )
            artifacts = {
                label: {
                    "generated_text": generated,
                    "answer": case["answer"],
                    "exact_match": True,
                    "selected_family": case["family"],
                    "runs": [
                        {
                            "generated_text": generated,
                            "answer": case["answer"],
                            "router_family": case["family"],
                            "selected_family": case["family"],
                            "limit_exhausted": False,
                            "repeat": repeat_index + 1,
                        }
                        for repeat_index in range(repeat_count)
                    ],
                }
                for label in labels
            }
            cases.append(
                {
                    "id": case_id,
                    "question": case["question"],
                    "truth": case["answer"],
                    "data": {"annotation": annotation, "image": image},
                    "artifacts": artifacts,
                    "cross_artifact_agreement": {
                        "answer": True,
                        "structured_text": True,
                    },
                }
            )
            heldout.append(
                {
                    "id": case_id,
                    "annotation": annotation,
                    "image": image,
                }
            )
        return {
            "format": comparison.NATIVE_REPORT_FORMAT,
            "settings": settings,
            "provenance": {
                "executable": {
                    "name": "/private/build/native-binary",
                    **_identity(executable_seed),
                },
                "packages": [
                    {
                        "label": label,
                        "manifest": {
                            "path": f"/private/packages/{label}/package_manifest.json",
                            **_identity(f"{package_seed}-{label}"),
                        },
                    }
                    for label in labels
                ],
                "heldout_cases": heldout,
            },
            "summary": {
                "packages": {
                    label: {
                        "exact_match": {
                            "correct": len(cases),
                            "total": len(cases),
                            "rate": 1.0,
                        }
                    }
                    for label in labels
                },
                "cross_artifact_agreement": {
                    "answer": {"agree": len(cases), "total": len(cases)},
                    "structured_text": {
                        "agree": len(cases),
                        "total": len(cases),
                    },
                },
            },
            "cases": cases,
        }

    def _build(
        self,
        native_documents: list[dict[str, object]],
        *,
        allow_partial: bool = False,
        eval_root: Path | None = None,
        record_document: dict[str, object] | None = None,
        summary_document: dict[str, object] | None = None,
    ) -> dict[str, object]:
        record_document = self.record_document if record_document is None else record_document
        summary_document = (
            self.summary_document if summary_document is None else summary_document
        )
        return comparison.build_report(
            native_reports=[_source(document) for document in native_documents],
            record_oracle=_source(record_document),
            record_precision="fp32",
            comparisons=[
                comparison.ComparisonSpec(
                    label="f32",
                    precision="fp32",
                    report=_source(record_document),
                ),
                comparison.ComparisonSpec(
                    label="w8a8",
                    precision="int8",
                    report=_source(summary_document),
                ),
            ],
            allow_partial=allow_partial,
            expected_count=2,
            eval_root=eval_root,
        )

    def test_merges_chunks_scores_canonically_and_handles_summary_only(self) -> None:
        report = self._build(
            [self._native_report(["00002"]), self._native_report(["00001"])]
        )

        self.assertEqual([case["id"] for case in report["cases"]], ["00001", "00002"])
        self.assertEqual(
            report["coverage"],
            {"expected": 2, "observed": 2, "complete": True, "missing_ids": []},
        )
        self.assertEqual(
            report["validation_scope"],
            {
                "onnx_record_fields": ["id", "question", "answer", "target"],
                "onnx_annotation_image_identities": (
                    "unavailable-in-producer-report"
                ),
                "native_asset_relative_layout": "validated",
                "native_annotation_image_content": {
                    "status": "reported-identities-only",
                    "verified_cases": 0,
                },
            },
        )
        self.assertFalse(comparison._has_unsafe_path(report))
        self.assertEqual(
            report["settings"]["mappings"],
            [
                {"label": "f32", "precision": "fp32", "mode": "record-level"},
                {"label": "w8a8", "precision": "int8", "mode": "aggregate-only"},
            ],
        )
        f32 = report["summary"]["labels"]["f32"]
        self.assertEqual(f32["mode"], "record-level")
        self.assertEqual(f32["native_metrics"]["answer_exact"], 1.0)
        self.assertEqual(f32["native_metrics"]["recomputed_answer_exact"], 0.5)
        self.assertEqual(f32["native_metrics"]["target_exact"], 0.5)
        self.assertEqual(
            f32["score_delta"],
            {
                "answer_exact": 0.0,
                "recomputed_answer_exact": -0.5,
                "target_exact": -0.5,
            },
        )
        self.assertEqual(
            f32["agreement"],
            {
                "prediction": {"agree": 1, "total": 2, "rate": 0.5},
                "answer": {"agree": 2, "total": 2, "rate": 1.0},
                "selected_family": {"agree": 2, "total": 2, "rate": 1.0},
            },
        )
        w8a8 = report["summary"]["labels"]["w8a8"]
        self.assertEqual(w8a8["mode"], "aggregate-only")
        self.assertIsNone(w8a8["agreement"])
        self.assertEqual(
            w8a8["score_delta"],
            {
                "answer_exact": 0.5,
                "recomputed_answer_exact": 0.0,
                "target_exact": 0.0,
            },
        )
        second = report["cases"][1]["labels"]["f32"]
        self.assertEqual(
            second["metrics"],
            {
                "answer_exact": True,
                "recomputed_answer_exact": False,
                "target_exact": False,
            },
        )
        self.assertEqual(
            second["onnx_agreement"],
            {"prediction": False, "answer": True, "selected_family": True},
        )

    def test_rejects_duplicate_and_missing_chunk_ids(self) -> None:
        with self.assertRaisesRegex(comparison.ComparisonError, "duplicate native case ID"):
            self._build(
                [self._native_report(["00001"]), self._native_report(["00001"])]
            )
        with self.assertRaisesRegex(comparison.ComparisonError, "missing record-oracle IDs"):
            self._build([self._native_report(["00001"])])

        partial = self._build(
            [self._native_report(["00001"])], allow_partial=True
        )
        self.assertEqual(partial["coverage"]["missing_ids"], ["00002"])
        self.assertIsNotNone(
            partial["summary"]["labels"]["f32"]["score_delta"]
        )
        self.assertIsNone(
            partial["summary"]["labels"]["w8a8"]["score_delta"]
        )

    def test_rejects_mismatched_shard_identity_settings_and_heldout_provenance(self) -> None:
        first = self._native_report(["00001"])
        variants = [
            (
                self._native_report(["00002"], settings_update={"repeat": 2}),
                "settings differ",
            ),
            (
                self._native_report(["00002"], executable_seed="other-executable"),
                "executable identity differs",
            ),
            (
                self._native_report(["00002"], package_seed="other-packages"),
                "package identities differ",
            ),
        ]
        for second, message in variants:
            with self.subTest(message=message):
                with self.assertRaisesRegex(comparison.ComparisonError, message):
                    self._build([first, second])

        bad_provenance = self._native_report(["00002"])
        bad_provenance["provenance"]["heldout_cases"][0]["image"] = _identity(
            "wrong-image", path="images/00002.jpg"
        )
        with self.assertRaisesRegex(comparison.ComparisonError, "heldout provenance"):
            self._build([first, bad_provenance])

        fixed_contract_variants = (
            ("backend", "cuda"),
            ("decode", "ordinary"),
            ("max_new", 1),
            ("application_timing", False),
            ("runtime_node_trace", True),
            ("accuracy_mode", "latency-only"),
        )
        for key, value in fixed_contract_variants:
            with self.subTest(fixed_contract=key):
                report = self._native_report(
                    ["00001", "00002"], settings_update={key: value}
                )
                with self.assertRaisesRegex(
                    comparison.ComparisonError, rf"settings\.{key}"
                ):
                    self._build([report])

    def test_rejects_native_question_or_truth_mismatch(self) -> None:
        for field, message in (
            ("question", "question differs"),
            ("truth", "truth differs"),
        ):
            first = self._native_report(["00001"])
            first["cases"][0][field] = "changed"
            if field == "truth":
                for artifact in first["cases"][0]["artifacts"].values():
                    artifact["exact_match"] = False
                for package in first["summary"]["packages"].values():
                    package["exact_match"] = {
                        "correct": 0,
                        "total": 1,
                        "rate": 0.0,
                    }
            with self.subTest(field=field):
                with self.assertRaisesRegex(comparison.ComparisonError, message):
                    self._build([first, self._native_report(["00002"])])

    def test_rejects_onnx_current_contract_drift(self) -> None:
        variants = (
            ("manifest_format", "old-format", "manifest_format"),
            ("records_total", 3, "current contract count 2"),
            ("skipped", {"00001": "failed"}, "skipped"),
            ("routing", "manual", "routing"),
            ("target_mode", "answer", "target_mode"),
            ("max_length", 191, "max_length"),
        )
        for field, value, message in variants:
            record = _clone(self.record_document)
            assert isinstance(record, dict)
            record[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(comparison.ComparisonError, message):
                    self._build(
                        [self._native_report(["00001", "00002"])],
                        record_document=record,
                    )

    def test_summary_only_report_must_match_fp32_oracle_anchor(self) -> None:
        variants: list[tuple[dict[str, object], str]] = []

        missing_privacy = _clone(self.summary_document)
        assert isinstance(missing_privacy, dict)
        missing_privacy["privacy"] = {}
        variants.append((missing_privacy, "privacy.summary_only"))

        changed_summary = _clone(self.summary_document)
        assert isinstance(changed_summary, dict)
        changed_summary["summary"]["fp32"]["overall"]["target_exact"] = 0.25
        variants.append((changed_summary, "FP32 summary does not exactly match"))

        changed_artifact = _clone(self.summary_document)
        assert isinstance(changed_artifact, dict)
        changed_artifact["artifacts"]["fp32"]["encoder"]["sha256"] = _digest(
            "different-fp32-encoder"
        )
        variants.append((changed_artifact, "FP32 artifacts do not match"))

        selected_rows_missing = _clone(self.summary_document)
        assert isinstance(selected_rows_missing, dict)
        selected_rows_missing["records"] = {"fp32": self.record_document["records"]["fp32"]}
        variants.append((selected_rows_missing, "records omit selected precision"))

        for document, message in variants:
            with self.subTest(message=message):
                with self.assertRaisesRegex(comparison.ComparisonError, message):
                    self._build(
                        [self._native_report(["00001", "00002"])],
                        summary_document=document,
                    )

    def test_rejects_inconsistent_native_artifacts_runs_and_summary(self) -> None:
        variants: list[tuple[dict[str, object], str]] = []

        bad_exact = self._native_report(["00001", "00002"])
        bad_exact["cases"][0]["artifacts"]["f32"]["exact_match"] = False
        variants.append((bad_exact, "exact_match disagrees"))

        bad_run_count = self._native_report(["00001", "00002"])
        bad_run_count["cases"][0]["artifacts"]["f32"]["runs"] = []
        variants.append((bad_run_count, "runs length"))

        bad_run_index = self._native_report(["00001", "00002"])
        bad_run_index["cases"][0]["artifacts"]["f32"]["runs"][0]["repeat"] = 2
        variants.append((bad_run_index, "one-based run index"))

        bad_run_text = self._native_report(["00001", "00002"])
        bad_run_text["cases"][0]["artifacts"]["f32"]["runs"][0][
            "generated_text"
        ] = "changed"
        variants.append((bad_run_text, "generated_text disagrees"))

        bad_answer = self._native_report(["00001", "00002"])
        bad_answer["cases"][0]["artifacts"]["f32"]["answer"] = "wrong"
        bad_answer["cases"][0]["artifacts"]["f32"]["exact_match"] = False
        bad_answer["cases"][0]["artifacts"]["f32"]["runs"][0][
            "answer"
        ] = "wrong"
        variants.append((bad_answer, "answer disagrees with generated_text"))

        bad_run_family = self._native_report(["00001", "00002"])
        bad_run_family["cases"][0]["artifacts"]["f32"]["runs"][0][
            "selected_family"
        ] = "other"
        variants.append((bad_run_family, "selected_family disagrees"))

        bad_summary = self._native_report(["00001", "00002"])
        bad_summary["summary"]["packages"]["f32"]["exact_match"]["correct"] = 1
        variants.append((bad_summary, "counts disagree"))

        bad_agreement_summary = self._native_report(["00001", "00002"])
        bad_agreement_summary["summary"]["cross_artifact_agreement"]["answer"][
            "agree"
        ] = 1
        variants.append((bad_agreement_summary, "counts disagree"))

        for document, message in variants:
            with self.subTest(message=message):
                with self.assertRaisesRegex(comparison.ComparisonError, message):
                    self._build([document])

        unstable_router = self._native_report(
            ["00001", "00002"], settings_update={"repeat": 2}
        )
        unstable_router["cases"][0]["artifacts"]["f32"]["runs"][1][
            "router_family"
        ] = "other"
        with self.assertRaisesRegex(comparison.ComparisonError, "router_family"):
            self._build([unstable_router])

    def test_eval_root_verifies_native_annotation_and_image_bytes(self) -> None:
        native = self._native_report(["00001", "00002"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "annotations").mkdir()
            (root / "images").mkdir()
            for case_id in self.cases:
                (root / "annotations" / f"{case_id}.json").write_bytes(
                    f"annotation-{case_id}".encode()
                )
                (root / "images" / f"{case_id}.jpg").write_bytes(
                    f"image-{case_id}".encode()
                )

            report = self._build([native], eval_root=root)
            self.assertEqual(
                report["validation_scope"]["native_annotation_image_content"],
                {"status": "verified-against-eval-root", "verified_cases": 2},
            )
            self.assertNotIn(str(root), json.dumps(report, ensure_ascii=False))

            (root / "images" / "00002.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(comparison.ComparisonError, "content differs"):
                self._build([native], eval_root=root)

    def test_path_guards_reject_backslashes_unc_file_uri_and_bad_filename(self) -> None:
        for value in (
            r"folder\file",
            r"\\server\share",
            r"C:\private\file",
            "file:///private/file",
        ):
            with self.subTest(value=value):
                self.assertTrue(comparison._has_unsafe_path({"value": value}))

        summary = _clone(self.summary_document)
        assert isinstance(summary, dict)
        summary["artifacts"]["int8"]["encoder"]["filename"] = r"dir\model.onnx"
        with self.assertRaisesRegex(comparison.ComparisonError, "plain filename"):
            self._build(
                [self._native_report(["00001", "00002"])],
                summary_document=summary,
            )

    def test_cli_enforces_current_count_and_rejects_output_symlink_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_one = root / "native-1.json"
            native_two = root / "native-2.json"
            record = root / "record.json"
            summary = root / "summary.json"
            output = root / "comparison.json"
            for path, document in (
                (native_one, self._native_report(["00001"])),
                (native_two, self._native_report(["00002"])),
                (record, self.record_document),
                (summary, self.summary_document),
            ):
                path.write_text(json.dumps(document), encoding="utf-8")

            arguments = [
                "--native-report",
                str(native_one),
                "--native-report",
                str(native_two),
                "--record-oracle",
                "fp32",
                str(record),
                "--comparison",
                "f32",
                "fp32",
                str(record),
                "--comparison",
                "w8a8",
                "int8",
                str(summary),
                "--out",
                str(output),
            ]
            current_contract_error = io.StringIO()
            with redirect_stderr(current_contract_error):
                self.assertEqual(comparison.main(arguments), 2)
            self.assertIn(
                "current contract count 2000", current_contract_error.getvalue()
            )
            self.assertFalse(output.exists())

            original_record = record.read_bytes()
            output.symlink_to(record)
            collision_error = io.StringIO()
            with redirect_stderr(collision_error):
                self.assertEqual(comparison.main(arguments), 2)
            self.assertIn(
                "--out must not resolve to an input report",
                collision_error.getvalue(),
            )
            self.assertEqual(record.read_bytes(), original_record)
            self.assertTrue(output.is_symlink())

            output.unlink()
            os.link(record, output)
            hardlink_error = io.StringIO()
            with redirect_stderr(hardlink_error):
                self.assertEqual(comparison.main(arguments), 2)
            self.assertIn(
                "--out must not resolve to an input report",
                hardlink_error.getvalue(),
            )
            self.assertEqual(record.read_bytes(), original_record)
            self.assertTrue(output.samefile(record))

            output.unlink()
            eval_root = root / "heldout"
            (eval_root / "annotations").mkdir(parents=True)
            (eval_root / "images").mkdir()
            eval_asset = eval_root / "images" / "00001.jpg"
            eval_asset.write_bytes(b"do-not-overwrite")
            eval_arguments = [
                *arguments[:-1],
                str(eval_asset),
                "--eval-root",
                str(eval_root),
            ]
            eval_collision_error = io.StringIO()
            with redirect_stderr(eval_collision_error):
                self.assertEqual(comparison.main(eval_arguments), 2)
            self.assertIn("--out must stay outside --eval-root", eval_collision_error.getvalue())
            self.assertEqual(eval_asset.read_bytes(), b"do-not-overwrite")


if __name__ == "__main__":
    unittest.main()
