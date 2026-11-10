from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import uuid

from examples.tts_common import run_remote as controller


TOKEN = "a" * 32
OWNED = "/tmp/volvox-tts-examples-" + TOKEN


def report_archive(corrupt=False):
    report = b'{"status":"PASS"}\n'
    manifest = json.dumps({"format": "volvoxai-remote-reports/v1", "files": [
        {"path": "reports/matrix.json", "bytes": len(report), "sha256": hashlib.sha256(report).hexdigest()}
    ]}).encode()
    sealed = {"manifest_sha256": hashlib.sha256(manifest).hexdigest(), "manifest_bytes": len(manifest),
              "file_count": 1, "total_bytes": len(report)}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, value in [(controller.MANIFEST, manifest), ("reports/matrix.json", b"X" * len(report) if corrupt else report)]:
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
    return stream.getvalue(), sealed


class RemoteCleanupTests(unittest.TestCase):
    def run_mock(self, payloads, *, survivors=()):
        archive, sealed = report_archive()
        operations = []
        def python(code, **kwargs):
            operation = code.splitlines()[0].split(":")[-1]
            operations.append(operation)
            result = {"stop": {"task_absent": False, "survivors": list(survivors)},
                      "seal": sealed, "cleanup": {"task_removed": True, "recovered_bytes": 4096},
                      "prune": {"task_removed": False, "reports_preserved": True, "recovered_bytes": 4096}}[operation]
            return subprocess.CompletedProcess([], 0, stdout=json.dumps(result))
        downloads = iter(payloads)
        def remote(command, **kwargs):
            operations.append("download")
            value = next(downloads)
            if isinstance(value, Exception):
                raise value
            kwargs["stdout"].write(value)
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory)
            result = controller.finalize_task(remote, python, local, OWNED, TOKEN, {"exit_code": 0}, delay=lambda _: None)
            evidence = {name: json.loads((local / name).read_text()) for name in ("cleanup.json", "collection.json") if (local / name).exists()}
            retained = (local / "reports/matrix.json").read_bytes() if (local / "reports/matrix.json").exists() else None
            self.assertFalse(list(local.glob(".reports-*.tar")))
        return result, operations, evidence, retained

    def test_corrupt_download_retries_before_deletion(self):
        valid, _ = report_archive()
        corrupt, _ = report_archive(corrupt=True)
        ok, operations, evidence, report = self.run_mock([corrupt, valid])
        self.assertTrue(ok)
        self.assertEqual(operations, ["stop", "seal", "download", "seal", "download", "cleanup"])
        self.assertEqual(report, b'{"status":"PASS"}\n')
        self.assertTrue(evidence["collection.json"]["verified"])
        self.assertEqual(len(evidence["collection.json"]["attempts"]), 2)

    def test_failed_collection_preserves_only_remote_reports(self):
        failure = subprocess.CalledProcessError(255, ["ssh"])
        ok, operations, evidence, report = self.run_mock([failure, failure, failure])
        self.assertFalse(ok)
        self.assertNotIn("cleanup", operations)
        self.assertEqual(operations[-1], "prune")
        self.assertTrue(evidence["cleanup.json"]["reports_preserved"])
        self.assertFalse(evidence["collection.json"]["verified"])
        self.assertIsNone(report)

    def test_live_processes_forbid_collection_and_cleanup(self):
        ok, operations, evidence, _ = self.run_mock([], survivors=[123])
        self.assertFalse(ok)
        self.assertEqual(operations, ["stop"])
        self.assertFalse(evidence["cleanup.json"]["task_removed"])

    def test_task_identity_rejects_unrelated_paths(self):
        with self.assertRaises(ValueError):
            controller.cleanup_code("/tmp/shared-models", TOKEN, preserve_reports=True)
        with self.assertRaises(ValueError):
            controller.safe_report_path("reports/../../elsewhere")

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "Linux process ownership check")
    def test_stop_escalates_only_the_owned_cwd(self):
        token = uuid.uuid4().hex
        owned = Path("/tmp") / ("volvox-tts-examples-" + token)
        owned.mkdir(mode=0o700)
        (owned / ".owner-token").write_text(token)
        children = []
        try:
            program = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(30)"
            with tempfile.TemporaryDirectory() as unrelated:
                for cwd in (owned, unrelated):
                    child = subprocess.Popen([sys.executable, "-c", program], cwd=cwd, stdout=subprocess.PIPE, text=True)
                    children.append(child)
                    self.assertEqual(child.stdout.readline().strip(), "ready")
                result = subprocess.run([sys.executable, "-c", controller.stop_code(str(owned), token)],
                                        capture_output=True, text=True, timeout=15, check=True)
                evidence = json.loads(result.stdout)
                self.assertEqual(evidence["survivors"], [])
                self.assertIn(children[0].pid, evidence["kill_pids"])
                self.assertEqual(children[0].wait(timeout=2), -9)
                self.assertIsNone(children[1].poll())
                children[1].kill()
                children[1].wait(timeout=2)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=2)
                child.stdout.close()
            shutil.rmtree(owned)


if __name__ == "__main__":
    unittest.main()
