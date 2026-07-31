from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


_TOOL_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "benchmark_native_heldout.py"
)
_SPEC = importlib.util.spec_from_file_location("benchmark_native_heldout", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
native_profile = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = native_profile
_SPEC.loader.exec_module(native_profile)


def _debug_output(
    *,
    tokens: int = 42,
    steady_steps: int = 42,
    family: str = "phone",
) -> bytes:
    return (
        "[debug] tinyreceipt split ABI routing=runtime decoder_output=token_ids "
        "keep_input=hoisted argmax=graph\n"
        f"[debug] tinyreceipt split router={family} selected={family} "
        "encoder=12.500 ms\n"
        f"[debug] tinyreceipt split family={family} tokens={tokens} "
        "total=84.000 ms tok/s=500.00 "
        "(incremental retained execution context)\n"
        "[debug] tinyreceipt split timing encoder=12.500 ms first=3.000 ms "
        f"steady_steps={steady_steps} steady_mean=1.500 ms "
        "steady_tok/s=666.67\n"
    ).encode()


class NativeHeldoutProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.eval_root = self.root / "heldout"
        (self.eval_root / "annotations").mkdir(parents=True)
        (self.eval_root / "images").mkdir()
        self._make_case("00001", "What is the phone number?", "2586238")
        self._make_case("00002", "What is the store?", "Example Mart")
        self.package_a = self._make_package("package-a", "f32")
        self.package_b = self._make_package("package-b", "int8")
        self.binary = self.root / "tiny_receipt_split_w8a8"
        self.binary.write_bytes(b"fixture native executable\n")
        self.binary.chmod(
            self.binary.stat().st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_case(self, case_id: str, question: str, answer: str) -> None:
        annotation = {
            "id": case_id,
            "question": question,
            "answer": answer,
        }
        (self.eval_root / "annotations" / f"{case_id}.json").write_text(
            json.dumps(annotation), encoding="utf-8"
        )
        (self.eval_root / "images" / f"{case_id}.jpg").write_bytes(
            f"fixture image {case_id}".encode()
        )

    @staticmethod
    def _asset(root: Path, relative: str, data: bytes) -> dict[str, object]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {
            "path": relative,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def _make_package(self, name: str, variant: str) -> Path:
        root = self.root / name
        root.mkdir()
        config = self._asset(root, "config.json", b'{"fixture":true}\n')
        vocab = self._asset(root, "vocab.json", b'{"itos":[]}\n')
        encoder_graph = self._asset(root, "encoder/graph.json", b"encoder graph\n")
        encoder_weights = self._asset(
            root, "encoder/model.safetensors", b"encoder weights\n"
        )
        encoder_report = self._asset(
            root, "encoder/export_report.json", b'{"fixture":true}\n'
        )
        decoder_graph = self._asset(root, "decoder/graph.json", b"decoder graph\n")
        decoder_weights = self._asset(
            root, "decoder/model.safetensors", b"decoder weights\n"
        )
        decoder_report = self._asset(
            root, "decoder/export_report.json", b'{"fixture":true}\n'
        )
        manifest = {
            "format": native_profile.PACKAGE_FORMAT,
            "assets": {"config": config, "vocab": vocab},
            "graphs": {
                "encoder": {
                    "graph": encoder_graph,
                    "weights": encoder_weights,
                    "export_report": encoder_report,
                },
                "decoder": {
                    "graph": decoder_graph,
                    "weights": decoder_weights,
                    "export_report": decoder_report,
                },
            },
            "routing": {"mode": "runtime"},
            "generation": {"decoder_output": "token_ids"},
            "variant": {"requested": variant},
        }
        (root / "package_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        return root

    @staticmethod
    def _stdout(answer: str = "2586238") -> bytes:
        text = (
            "<field>phone</field><value>2586238</value>"
            f"<answer>{answer}</answer>"
        )
        return (
            "VolvoxAI Native Runtime\nBackend policy: cpu\n" + text + "\n"
        ).encode()

    def test_profiles_identical_cases_with_strict_native_flags(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
            calls.append(command)
            self.assertFalse(options["check"])
            self.assertEqual(options["stdout"], subprocess.PIPE)
            self.assertEqual(options["stderr"], subprocess.PIPE)
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=self._stdout(),
                stderr=_debug_output(),
            )

        clock_values = iter(index * 100_000_000 for index in range(8))
        report = native_profile.build_report(
            eval_root=self.eval_root,
            package_arguments=[
                f"f32={self.package_a}",
                f"direct-int8={self.package_b}",
            ],
            binary=self.binary,
            count=None,
            requested_ids=["00001"],
            repeat=2,
            runner=runner,
            clock_ns=lambda: next(clock_values),
        )

        self.assertEqual(len(calls), 4)
        for command in calls:
            self.assertEqual(command[0], str(self.binary))
            self.assertIn("--incremental", command)
            self.assertIn("--cpu", command)
            self.assertEqual(command[command.index("--threads") + 1], "1")
            self.assertIn("--require-row", command)
            self.assertIn("--timing", command)
            self.assertNotIn("--debug", command)
            self.assertEqual(command[command.index("--family") + 1], "auto")
            self.assertEqual(command[command.index("--max-new") + 1], "191")

        self.assertEqual(report["format"], native_profile.REPORT_FORMAT)
        self.assertEqual(report["settings"]["cpu_threads"], 1)
        summary = report["summary"]
        self.assertEqual(summary["packages"]["f32"]["exact_match"]["correct"], 1)
        self.assertEqual(
            summary["packages"]["direct-int8"]["exact_match"]["correct"], 1
        )
        self.assertEqual(
            summary["cross_artifact_agreement"]["answer"],
            {"agree": 1, "total": 1},
        )
        self.assertEqual(
            summary["cross_artifact_agreement"]["structured_text"],
            {"agree": 1, "total": 1},
        )
        self.assertEqual(
            report["cases"][0]["artifacts"]["f32"]["runs"][0]["wall_ms"],
            100.0,
        )
        encoded = json.dumps(report, allow_nan=False, sort_keys=True)
        self.assertNotIn(str(self.root), encoded)

    def test_accuracy_rejects_short_or_limit_exhausted_generation(self) -> None:
        with self.assertRaisesRegex(native_profile.ProfileError, "requires --max-new 191"):
            native_profile.validate_generation_settings(96, False)
        native_profile.validate_generation_settings(96, True)

        with self.assertRaisesRegex(native_profile.ProfileError, "accuracy would be truncated"):
            native_profile.parse_native_output(
                self._stdout(),
                _debug_output(tokens=191, steady_steps=190),
                191,
                False,
            )

    def test_latency_only_reports_no_accuracy_for_short_runs(self) -> None:
        def runner(command: list[str], **_options: object) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    b"VolvoxAI Native Runtime\nBackend policy: cpu\n"
                    b"<field>phone</field>\n"
                ),
                stderr=_debug_output(tokens=8, steady_steps=7),
            )

        ticks = iter((0, 5_000_000))
        report = native_profile.build_report(
            eval_root=self.eval_root,
            package_arguments=[f"probe={self.package_a}"],
            binary=self.binary,
            count=None,
            requested_ids=["00001"],
            max_new=8,
            latency_only=True,
            runner=runner,
            clock_ns=lambda: next(ticks),
        )
        self.assertIsNone(report["summary"]["packages"]["probe"]["exact_match"])
        self.assertTrue(
            report["cases"][0]["artifacts"]["probe"]["runs"][0][
                "limit_exhausted"
            ]
        )

    def test_rejects_missing_or_duplicate_debug_timing(self) -> None:
        stderr = _debug_output()
        with self.assertRaisesRegex(native_profile.ProfileError, "exactly one"):
            native_profile.parse_native_output(self._stdout(), b"", 191, False)
        with self.assertRaisesRegex(native_profile.ProfileError, "exactly one"):
            native_profile.parse_native_output(
                self._stdout(), stderr + stderr, 191, False
            )

    def test_package_asset_identity_is_verified(self) -> None:
        (self.package_a / "decoder" / "graph.json").write_bytes(b"changed")
        with self.assertRaisesRegex(native_profile.ProfileError, "manifest identity"):
            native_profile.build_report(
                eval_root=self.eval_root,
                package_arguments=[f"broken={self.package_a}"],
                binary=self.binary,
                count=1,
            )

    def test_export_report_identity_is_verified(self) -> None:
        (self.package_a / "encoder" / "export_report.json").write_bytes(b"changed")
        with self.assertRaisesRegex(native_profile.ProfileError, "manifest identity"):
            native_profile.build_report(
                eval_root=self.eval_root,
                package_arguments=[f"broken={self.package_a}"],
                binary=self.binary,
                count=1,
            )


if __name__ == "__main__":
    unittest.main()
