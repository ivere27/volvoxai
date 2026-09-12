from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.tests.check_oracle_coverage import (
    CASE_MANIFEST_PATH,
    MANIFEST_PATH,
    ManifestError,
    validate_document,
    validate_manifest,
)
from tools.exporter.generated.kernel_registry import OPS_BY_TARGET, PROFILE_MEMBERS


class OracleCoverageManifestTests(unittest.TestCase):
    def test_checked_in_manifest_covers_generated_portable_intersection(self) -> None:
        counts = validate_manifest()
        expected = set.intersection(
            *(set(OPS_BY_TARGET[member]) for member in PROFILE_MEMBERS["portable"])
        )
        self.assertEqual(counts["total"], len(expected))
        cases = json.loads(CASE_MANIFEST_PATH.read_text(encoding="utf-8"))
        exercised = {
            operator
            for case in cases["cases"]
            for operator in case["expected_lowered_ops"]
        }
        self.assertEqual(counts["executable_cases"], len(cases["cases"]))
        self.assertEqual(counts["executable_operators"], len(exercised))

    def test_missing_operator_fails_closed(self) -> None:
        document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        document["classifications"]["onnx-direct"]["operators"].remove("Add")
        with self.assertRaisesRegex(ManifestError, "missing portable operators: Add"):
            validate_document(document)

    def test_duplicate_classification_fails_closed(self) -> None:
        document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        duplicate = copy.deepcopy(document)
        duplicate["classifications"]["local-invariant"]["operators"].append(
            "QMaskedMean"
        )
        duplicate["classifications"]["local-invariant"]["operators"].sort()
        with self.assertRaisesRegex(ManifestError, "classified more than once"):
            validate_document(duplicate)

    def test_executable_case_cannot_claim_an_unsupported_route(self) -> None:
        document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        document["classifications"]["onnx-direct"]["operators"].remove(
            "Gather"
        )
        document["classifications"]["oracle-unsupported"]["operators"].append(
            "Gather"
        )
        document["classifications"]["oracle-unsupported"]["operators"].sort()
        with self.assertRaisesRegex(
            ManifestError, "claims ORT evidence for 'Gather'"
        ):
            validate_document(document)

    def test_non_portable_profile_fails_closed(self) -> None:
        document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        document["profile"] = "browser"
        document["profile_members"] = list(PROFILE_MEMBERS["browser"])
        with self.assertRaisesRegex(ManifestError, "must use the portable profile"):
            validate_document(document)


if __name__ == "__main__":
    unittest.main()
