"""Unit tests for local GitHub-Release-backed npm and PyPI publishing."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "publish_release_local", ROOT / "tools/publish_release_local.py"
)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def _make_npm_tarball(destination: Path, pkg_name: str, version: str, dist_files: tuple[str, ...]) -> tuple[bytes, str, str]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        entries = {
            "package/package.json": json.dumps({"name": pkg_name, "version": version}).encode("utf-8"),
            "package/README.md": f"# {pkg_name}\n".encode("utf-8"),
            "package/LICENSE": b"MIT\n",
            **{f"package/dist/{version}/{name}": f"fixture {name}".encode("utf-8") for name in dist_files},
        }
        for name, data in sorted(entries.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()
    destination.write_bytes(raw)
    sha256 = hashlib.sha256(raw).hexdigest()
    sri = "sha512-" + base64.b64encode(hashlib.sha512(raw).digest()).decode("ascii")
    return raw, sha256, sri


def _make_wheel(destination: Path, version: str) -> tuple[bytes, str]:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"volvoxai-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: volvoxai\nVersion: {version}\n\nFixture wheel\n",
        )
        archive.writestr("volvoxai/libvolvoxai.so", b"elf-so")
        archive.writestr("volvoxai/libsynurang_module_host.so", b"elf-host")
        archive.writestr("volvoxai/bin/volvoxai", b"elf-bin")
        archive.writestr("volvoxai/BUILD-INFO.json", json.dumps({"version": version}))
    raw = buf.getvalue()
    destination.write_bytes(raw)
    return raw, hashlib.sha256(raw).hexdigest()


STUB_TOOL = r'''
import json
import os
from pathlib import Path
import shutil
import sys

tool = Path(sys.argv[0]).name
args = sys.argv[1:]
cfg = json.loads(Path(os.environ["LOCAL_PUB_CONFIG"]).read_text(encoding="utf-8"))
with Path(os.environ["LOCAL_PUB_LOG"]).open("a", encoding="utf-8") as fh:
    fh.write(json.dumps({"tool": tool, "args": args}) + "\n")

if tool == "gh":
    if args[0] == "api":
        # Draft binary download: -H Accept: application/octet-stream .../releases/assets/<id>
        if "-H" in args and "Accept: application/octet-stream" in args:
            asset_id = int(args[-1].rsplit("/", 1)[1])
            src = Path(cfg["asset_paths_by_id"][str(asset_id)])
            sys.stdout.buffer.write(src.read_bytes())
            raise SystemExit(0)
        endpoint = next(a for a in args if a.startswith("repos/"))
        if endpoint.endswith("/releases"):
            for rel in cfg.get("releases", []):
                print(json.dumps(rel))
            raise SystemExit(0)
        if "/releases/" in endpoint and endpoint.endswith("/assets"):
            rel_id = int(endpoint.split("/releases/")[1].split("/")[0])
            for asset in cfg.get("assets_by_release", {}).get(str(rel_id), []):
                print(json.dumps(asset))
            raise SystemExit(0)
        raise SystemExit(f"unexpected gh api endpoint: {endpoint}")
    if args[:2] == ["release", "download"]:
        dir_idx = args.index("--dir")
        out_dir = Path(args[dir_idx + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        patterns = [args[i + 1] for i, a in enumerate(args) if a == "-p"]
        for name in patterns:
            src = Path(cfg["asset_paths"][name])
            shutil.copy2(src, out_dir / name)
        raise SystemExit(0)
    raise SystemExit(f"unexpected gh command: {args}")

if tool == "npm":
    if args[0] == "view":
        if len(args) >= 3 and args[2] == "versions":
            pkg = args[1]
            versions = cfg.get("npm_versions", {}).get(pkg)
            if versions is None:
                print(json.dumps({"error": {"code": "E404"}}))
                raise SystemExit(1)
            print(json.dumps(versions))
            raise SystemExit(0)
        spec = args[1]
        entry = cfg.get("npm_view", {}).get(spec, "__MISSING__")
        if entry == "__MISSING__":
            print(json.dumps({"error": {"code": "E404"}}))
            raise SystemExit(1)
        if entry == "__EMPTY__":
            raise SystemExit(0)
        print(json.dumps(entry))
        raise SystemExit(0)
    if args[0] == "whoami":
        if cfg.get("npm_unauth"):
            raise SystemExit(1)
        print("fixture-maintainer")
        raise SystemExit(0)
    if args[0] == "publish":
        if cfg.get("fail_npm_publish") == Path(args[1]).name:
            raise SystemExit(7)
        raise SystemExit(0)
    raise SystemExit(f"unexpected npm command: {args}")

if tool == "twine":
    if args[0] == "check":
        if cfg.get("fail_twine_check"):
            print("metadata error", file=sys.stderr)
            raise SystemExit(1)
        print("PASSED")
        raise SystemExit(0)
    if args[0] == "upload":
        if cfg.get("fail_twine_upload"):
            raise SystemExit(9)
        raise SystemExit(0)
    raise SystemExit(f"unexpected twine command: {args}")

raise SystemExit(f"unexpected tool: {tool}")
'''


class PublishReleaseLocalTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="volvoxai-local-pub-test-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.bin_dir = self.tmp / "bin"
        self.bin_dir.mkdir()
        for tool in ("gh", "npm", "twine"):
            exe = self.bin_dir / tool
            exe.write_text(f"#!{sys.executable}\n{STUB_TOOL}", encoding="utf-8")
            exe.chmod(0o755)
        self.assets_dir = self.tmp / "remote-assets"
        self.assets_dir.mkdir()
        self.config_path = self.tmp / "config.json"
        self.log_path = self.tmp / "calls.jsonl"
        self.orig_env = {
            k: os.environ.get(k)
            for k in ("PATH", "LOCAL_PUB_CONFIG", "LOCAL_PUB_LOG", "TWINE_REPOSITORY", "TWINE_REPOSITORY_URL")
        }
        os.environ["PATH"] = f"{self.bin_dir}{os.pathsep}{os.environ['PATH']}"
        os.environ["LOCAL_PUB_CONFIG"] = str(self.config_path)
        os.environ["LOCAL_PUB_LOG"] = str(self.log_path)
        os.environ.pop("TWINE_REPOSITORY", None)
        os.environ.pop("TWINE_REPOSITORY_URL", None)

    def tearDown(self):
        for key, value in self.orig_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _calls(self) -> list[dict]:
        if not self.log_path.is_file():
            return []
        return [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _stage_release(
        self,
        version: str = "1.2.3",
        *,
        draft: bool = False,
        prerelease: bool = False,
        corrupt_sums: bool = False,
        extra_config: dict | None = None,
    ) -> dict:
        self.log_path.unlink(missing_ok=True)
        asset_paths: dict[str, str] = {}
        asset_paths_by_id: dict[str, str] = {}
        assets_meta: list[dict] = []
        sri_by_pkg: dict[str, str] = {}
        sha_by_name: dict[str, str] = {}
        next_id = 100

        for pkg_name, prefix, dist_files in MOD.NPM_PACKAGES:
            fname = f"{prefix}-{version}.tgz"
            path = self.assets_dir / fname
            raw, sha, sri = _make_npm_tarball(path, pkg_name, version, dist_files)
            sri_by_pkg[pkg_name] = sri
            sha_by_name[fname] = sha
            asset_paths[fname] = str(path)
            asset_paths_by_id[str(next_id)] = str(path)
            assets_meta.append({
                "id": next_id,
                "name": fname,
                "state": "uploaded",
                "size": len(raw),
                "digest": f"sha256:{sha}",
            })
            next_id += 1

        wheel_name = MOD.expected_wheel_filename(version)
        wheel_path = self.assets_dir / wheel_name
        wheel_raw, wheel_sha = _make_wheel(wheel_path, version)
        sha_by_name[wheel_name] = wheel_sha
        asset_paths[wheel_name] = str(wheel_path)
        asset_paths_by_id[str(next_id)] = str(wheel_path)
        assets_meta.append({
            "id": next_id,
            "name": wheel_name,
            "state": "uploaded",
            "size": len(wheel_raw),
            "digest": f"sha256:{wheel_sha}",
        })
        next_id += 1

        sums_lines = []
        for name, sha in sorted(sha_by_name.items()):
            digest = ("0" * 64) if corrupt_sums else sha
            sums_lines.append(f"{digest}  {name}\n")
        sums_path = self.assets_dir / "SHA256SUMS"
        sums_raw = "".join(sums_lines).encode("utf-8")
        sums_path.write_bytes(sums_raw)
        sums_sha = hashlib.sha256(sums_raw).hexdigest()
        asset_paths["SHA256SUMS"] = str(sums_path)
        asset_paths_by_id[str(next_id)] = str(sums_path)
        assets_meta.append({
            "id": next_id,
            "name": "SHA256SUMS",
            "state": "uploaded",
            "size": len(sums_raw),
            "digest": f"sha256:{sums_sha}",
        })

        config = {
            "releases": [{
                "id": 42,
                "tag_name": f"v{version}",
                "draft": draft,
                "prerelease": prerelease,
            }],
            "assets_by_release": {"42": assets_meta},
            "asset_paths": asset_paths,
            "asset_paths_by_id": asset_paths_by_id,
            **(extra_config or {}),
        }
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        return {"sri": sri_by_pkg, "sha": sha_by_name, "wheel_name": wheel_name}

    def test_npm_dry_run_downloads_verifies_and_skips_registry(self):
        self._stage_release("1.2.3")
        out = MOD.publish_npm_local(tag="v1.2.3", repo="ivere27/volvoxai", dry_run=True, root=self.tmp)
        self.assertEqual(out, ["volvoxai@1.2.3", "volvoxai-lite@1.2.3"])
        self.assertFalse(any(c["tool"] == "npm" for c in self._calls()))

    def test_npm_publish_handles_empty_view_and_forwards_tag_and_otp(self):
        self._stage_release(
            "1.2.3",
            extra_config={
                "npm_view": {"volvoxai@1.2.3": "__EMPTY__"},
                "npm_versions": {"volvoxai": ["1.2.2"]},
            },
        )
        out = MOD.publish_npm_local(tag="v1.2.3", repo="ivere27/volvoxai", otp="654321", root=self.tmp)
        self.assertEqual(out, ["volvoxai@1.2.3", "volvoxai-lite@1.2.3"])
        publishes = [c["args"] for c in self._calls() if c["tool"] == "npm" and c["args"][0] == "publish"]
        self.assertEqual(len(publishes), 2)
        for args in publishes:
            self.assertIn("--tag", args)
            self.assertEqual(args[args.index("--tag") + 1], "latest")
            self.assertIn("--otp", args)
            self.assertEqual(args[args.index("--otp") + 1], "654321")

    def test_npm_publish_skips_identical_and_rejects_conflicting_integrity(self):
        info = self._stage_release("1.2.3")
        self._stage_release(
            "1.2.3",
            extra_config={
                "npm_view": {
                    "volvoxai@1.2.3": {"version": "1.2.3", "dist": {"integrity": info["sri"]["volvoxai"]}},
                }
            },
        )
        out = MOD.publish_npm_local(tag="v1.2.3", repo="ivere27/volvoxai", root=self.tmp)
        self.assertEqual(out, ["volvoxai-lite@1.2.3"])

        self._stage_release(
            "1.2.3",
            extra_config={
                "npm_view": {
                    "volvoxai-lite@1.2.3": {"version": "1.2.3", "dist": {"integrity": "sha512-conflict"}},
                }
            },
        )
        with self.assertRaisesRegex(MOD.ReleaseError, "different integrity"):
            MOD.publish_npm_local(tag="v1.2.3", repo="ivere27/volvoxai", root=self.tmp)

    def test_rejects_draft_prerelease_and_corrupt_sha256sums(self):
        self._stage_release("1.2.3", draft=True)
        with self.assertRaisesRegex(MOD.ReleaseError, "is a draft"):
            MOD.publish_npm_local(tag="v1.2.3", repo="ivere27/volvoxai", dry_run=True, root=self.tmp)
        # --allow-draft downloads via asset binary API stream and succeeds.
        out = MOD.publish_npm_local(
            tag="v1.2.3", repo="ivere27/volvoxai", allow_draft=True, dry_run=True, root=self.tmp
        )
        self.assertEqual(out, ["volvoxai@1.2.3", "volvoxai-lite@1.2.3"])

        self._stage_release("1.2.3-rc.1", prerelease=True)
        with self.assertRaisesRegex(MOD.ReleaseError, "Refusing to publish prerelease"):
            MOD.publish_npm_local(tag="v1.2.3-rc.1", repo="ivere27/volvoxai", dry_run=True, root=self.tmp)
        with self.assertRaisesRegex(MOD.ReleaseError, "dist-tag 'latest'"):
            MOD.publish_npm_local(
                tag="v1.2.3-rc.1",
                repo="ivere27/volvoxai",
                npm_tag="latest",
                allow_prerelease=True,
                dry_run=True,
                root=self.tmp,
            )
        out = MOD.publish_npm_local(
            tag="v1.2.3-rc.1", repo="ivere27/volvoxai", allow_prerelease=True, root=self.tmp
        )
        self.assertEqual(out, ["volvoxai@1.2.3-rc.1", "volvoxai-lite@1.2.3-rc.1"])
        publishes = [c["args"] for c in self._calls() if c["tool"] == "npm" and c["args"][0] == "publish"]
        self.assertTrue(all(args[args.index("--tag") + 1] == "rc" for args in publishes))

        self._stage_release("1.2.3", corrupt_sums=True)
        with self.assertRaisesRegex(MOD.ReleaseError, "SHA256SUMS mismatch"):
            MOD.publish_npm_local(tag="v1.2.3", repo="ivere27/volvoxai", dry_run=True, root=self.tmp)

    def test_pypi_dry_run_and_upload_from_github_release(self):
        info = self._stage_release("1.2.3")
        out = MOD.publish_pypi_local(tag="v1.2.3", repo="ivere27/volvoxai", dry_run=True, root=self.tmp)
        self.assertEqual(out, [info["wheel_name"]])
        twine_calls = [c["args"] for c in self._calls() if c["tool"] == "twine"]
        self.assertEqual([args[0] for args in twine_calls], ["check"])

        self._stage_release("1.2.3")
        with mock.patch.object(MOD, "_check_pypi_existing", return_value=False):
            out = MOD.publish_pypi_local(tag="v1.2.3", repo="ivere27/volvoxai", root=self.tmp)
        self.assertEqual(out, [info["wheel_name"]])
        twine_calls = [c["args"] for c in self._calls() if c["tool"] == "twine"]
        self.assertEqual([args[0] for args in twine_calls], ["check", "upload"])

        self._stage_release("1.2.3")
        with mock.patch.object(MOD, "_check_pypi_existing", return_value=True):
            out = MOD.publish_pypi_local(tag="v1.2.3", repo="ivere27/volvoxai", root=self.tmp)
        self.assertEqual(out, [])
        twine_calls = [c["args"] for c in self._calls() if c["tool"] == "twine"]
        self.assertEqual([args[0] for args in twine_calls], ["check"])

    def test_makefile_publish_release_targets_require_tag_and_run_dry_run(self):
        self._stage_release("1.2.3")
        missing = subprocess.run(
            ["make", "--no-print-directory", "publish_release"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("TAG is required", missing.stdout + missing.stderr)

        dry = subprocess.run(
            ["make", "--no-print-directory", "publish_release", "TAG=v1.2.3", "REPO=ivere27/volvoxai", "DRY_RUN=1"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(dry.returncode, 0, dry.stdout + dry.stderr)
        self.assertIn("Dry run (npm -> latest): volvoxai@1.2.3", dry.stdout)
        self.assertIn("Dry run (PyPI): volvoxai-1.2.3-cp310-abi3-manylinux_2_28_x86_64.whl", dry.stdout)


if __name__ == "__main__":
    unittest.main()
