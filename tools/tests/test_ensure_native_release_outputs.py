from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "ensure_native_release_outputs.py"
FORMAT = "volvoxai-native-release-finalization/v1"


def _fingerprint(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "rawBytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


class NativeReleaseOutputGuardTests(unittest.TestCase):
    def test_matching_bundle_is_stable_and_map_corruption_forces_relink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "build" / "native" / "release-link" / "volvoxai"
            debug = artifact.parent / ".debug" / "volvoxai.debug"
            evidence = debug.with_suffix(".debug.json")
            link_map = root / "build" / "size-maps" / "volvoxai.map"
            published = root / "native" / "volvoxai"
            published_debug = root / "native" / ".debug" / "volvoxai.debug"
            published_evidence = published_debug.with_suffix(".debug.json")
            stamp = root / "build" / "native" / "release-state" / "volvoxai.stamp"
            link_object = root / "build" / "native" / "CMakeFiles" / "engine.c.o"
            for path in (
                artifact,
                debug,
                link_map,
                link_object,
                published,
                published_debug,
                stamp,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"release")
            debug.write_bytes(b"debug")
            link_map.write_bytes(b"map")
            link_object.write_bytes(b"object")
            published.write_bytes(artifact.read_bytes())
            published_debug.write_bytes(debug.read_bytes())
            document = {
                "format": FORMAT,
                "build": {
                    "directory": "build",
                    "artifact": _fingerprint(artifact),
                    "linkMap": _fingerprint(link_map),
                    "link": {
                        "objects": [
                            {
                                "path": "build/native/CMakeFiles/engine.c.o",
                                **_fingerprint(link_object),
                            }
                        ]
                    },
                },
                "release": _fingerprint(artifact),
                "debug": _fingerprint(debug),
            }
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text(json.dumps(document), encoding="utf-8")
            published_evidence.write_bytes(evidence.read_bytes())
            stamp.touch()
            original_mtime = 1_000_000_000
            stamp.touch()
            stamp.chmod(0o600)
            stamp_time = stamp.stat().st_atime_ns
            # Give the healthy check an unmistakable timestamp to preserve.
            import os

            os.utime(stamp, ns=(stamp_time, original_mtime))

            command = [
                sys.executable,
                str(SCRIPT),
                "--artifact",
                str(artifact),
                "--debug",
                str(debug),
                "--evidence",
                str(evidence),
                "--map",
                str(link_map),
                "--published-artifact",
                str(published),
                "--published-debug",
                str(published_debug),
                "--published-evidence",
                str(published_evidence),
                "--build-directory",
                "build",
                "--repository-root",
                str(root),
                "--stamp",
                str(stamp),
            ]
            healthy = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(healthy.returncode, 0, healthy.stderr)
            self.assertEqual(stamp.stat().st_mtime_ns, original_mtime)

            link_map.write_bytes(b"corrupt-map")
            corrupt = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(corrupt.returncode, 0, corrupt.stderr)
            self.assertIn("require relink", corrupt.stdout)
            self.assertGreater(stamp.stat().st_mtime_ns, artifact.stat().st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
