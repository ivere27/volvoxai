#!/usr/bin/env python3
"""Download verified artifacts from a GitHub Release and publish locally.

GitHub Actions builds, tests, and attaches verified release artifacts to
GitHub Releases without holding external registry tokens. Maintainers run
this tool locally (typically via ``make publish_npm_release``,
``make publish_pypi_release``, or ``make publish_release``) to fetch the
released artifacts, verify GitHub asset digests and ``SHA256SUMS``, validate
package manifests, and publish to npm or PyPI using local credentials.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "ivere27/volvoxai"
GITHUB_DIGEST_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z.-]+))?(?:\+([0-9A-Za-z.-]+))?$"
)
SHA256SUM_LINE_RE = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+)$")

NPM_PACKAGES = (
    ("volvoxai", "volvoxai", (
        "volvoxai.lite.js",
        "volvoxai.lite.min.js",
        "volvoxai.js",
        "volvoxai.min.js",
        "volvoxai.lite.wasm",
        "volvoxai.wasm",
    )),
    ("volvoxai-lite", "volvoxai-lite", (
        "volvoxai.lite.js",
        "volvoxai.lite.min.js",
        "volvoxai.lite.wasm",
    )),
)


class ReleaseError(RuntimeError):
    """Raised when release verification or local publishing cannot proceed."""


@dataclass(frozen=True)
class SemVer:
    major: int
    minor: int
    patch: int
    prerelease: tuple[str, ...]
    build: tuple[str, ...]

    @property
    def is_prerelease(self) -> bool:
        return bool(self.prerelease)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def parse_semver(value: str) -> SemVer:
    match = SEMVER_RE.fullmatch(value)
    require(match is not None, f"Invalid semantic version: {value!r}")
    assert match is not None
    prerelease: tuple[str, ...] = ()
    if match.group(4) is not None:
        parts = tuple(match.group(4).split("."))
        for part in parts:
            require(
                bool(part) and re.fullmatch(r"[0-9A-Za-z-]+", part) is not None,
                f"Invalid prerelease identifier in version {value!r}: {part!r}",
            )
            require(
                not (part.isdigit() and len(part) > 1 and part.startswith("0")),
                f"Numeric prerelease identifier has leading zero in {value!r}: {part!r}",
            )
        prerelease = parts
    build: tuple[str, ...] = ()
    if match.group(5) is not None:
        parts = tuple(match.group(5).split("."))
        for part in parts:
            require(
                bool(part) and re.fullmatch(r"[0-9A-Za-z-]+", part) is not None,
                f"Invalid build identifier in version {value!r}: {part!r}",
            )
        build = parts
    return SemVer(
        major=int(match.group(1)),
        minor=int(match.group(2)),
        patch=int(match.group(3)),
        prerelease=prerelease,
        build=build,
    )


def expected_npm_filenames(version: str) -> set[str]:
    return {f"{prefix}-{version}.tgz" for _, prefix, _ in NPM_PACKAGES}


def expected_wheel_filename(version: str) -> str:
    return f"volvoxai-{version}-cp310-abi3-manylinux_2_28_x86_64.whl"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha512_sri_file(path: Path) -> str:
    digest = hashlib.sha512()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha512-" + base64.b64encode(digest.digest()).decode("ascii")


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _resolve_gh_host() -> str:
    return (os.environ.get("GH_HOST") or "github.com").strip() or "github.com"


def resolve_repository(explicit_repo: str = "", *, root: Path = ROOT) -> str:
    repo = explicit_repo.strip() or (os.environ.get("GH_REPO") or "").strip()
    if repo:
        require(
            re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) is not None,
            f"Invalid GitHub repository slug: {repo!r}",
        )
        return repo
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        url = result.stdout.strip()
        match = re.search(r"[:/]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?$", url)
        if match:
            return match.group(1)
    return DEFAULT_REPO


def _gh_api_ndjson(endpoint: str, *, root: Path) -> list[dict]:
    result = subprocess.run(
        ["gh", "api", "--hostname", _resolve_gh_host(), "--paginate", endpoint, "--jq", ".[]"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit {result.returncode}"
        raise ReleaseError(f"gh api {endpoint} failed: {detail}")
    items: list[dict] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        parsed = json.loads(text)
        require(isinstance(parsed, dict), f"Unexpected JSON item from {endpoint}: {parsed!r}")
        items.append(parsed)
    return items


def _parse_sha256sums(path: Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        match = SHA256SUM_LINE_RE.fullmatch(line)
        require(match is not None, f"Invalid SHA256SUMS line {line_no}: {raw_line!r}")
        assert match is not None
        digest_hex = match.group(1).lower()
        name = Path(match.group(2).strip()).name
        require(name not in sums, f"Duplicate SHA256SUMS entry for {name!r}")
        sums[name] = digest_hex
    require(bool(sums), "SHA256SUMS is empty")
    return sums


def download_github_release_assets(
    *,
    tag: str,
    required_assets: set[str],
    repo: str = "",
    download_dir: Path,
    allow_prerelease: bool = False,
    allow_draft: bool = False,
    require_sums: bool = True,
    root: Path = ROOT,
) -> tuple[dict, dict[str, str]]:
    require(shutil.which("gh") is not None, "gh CLI is required to download GitHub release assets")
    target_repo = resolve_repository(repo, root=root)
    releases = _gh_api_ndjson(f"repos/{target_repo}/releases", root=root)
    matches = [rel for rel in releases if rel.get("tag_name") == tag]
    require(
        len(matches) == 1,
        f"Expected one GitHub release for tag {tag!r} on {target_repo}, found {len(matches)}",
    )
    release = matches[0]
    is_draft = release.get("draft") is not False
    if is_draft and not allow_draft:
        raise ReleaseError(f"GitHub release {tag!r} is a draft (pass --allow-draft to override)")
    if release.get("prerelease") is True and not allow_prerelease:
        raise ReleaseError(
            f"GitHub release {tag!r} is a prerelease (pass --allow-prerelease to override)"
        )

    release_id = release.get("id")
    if _is_positive_int(release_id):
        actual_assets = _gh_api_ndjson(
            f"repos/{target_repo}/releases/{release_id}/assets", root=root
        )
    else:
        actual_assets = release.get("assets") or []
    require(isinstance(actual_assets, list), f"Invalid asset payload on release {tag!r}")

    asset_by_name = {
        item.get("name"): item
        for item in actual_assets
        if isinstance(item, dict) and item.get("name")
    }
    to_fetch = set(required_assets)
    if require_sums:
        to_fetch.add("SHA256SUMS")

    missing_remote = sorted(to_fetch - set(asset_by_name))
    require(
        not missing_remote,
        f"GitHub release {tag!r} is missing expected asset(s): {', '.join(missing_remote)}",
    )
    for name in sorted(to_fetch):
        meta = asset_by_name[name]
        if meta.get("state") != "uploaded" or not _is_positive_int(meta.get("size")):
            raise ReleaseError(
                f"GitHub release asset {name!r} on {tag!r} is incomplete "
                f"(state={meta.get('state')!r}, size={meta.get('size')!r})"
            )
        if is_draft and not _is_positive_int(meta.get("id")):
            raise ReleaseError(f"GitHub release asset {name!r} on draft {tag!r} lacks a valid asset id")
        digest = meta.get("digest")
        if not isinstance(digest, str) or not GITHUB_DIGEST_RE.fullmatch(digest):
            raise ReleaseError(
                f"GitHub release asset {name!r} on {tag!r} lacks a valid SHA-256 digest: {digest!r}"
            )

    download_dir.mkdir(parents=True, exist_ok=True)
    gh_host = _resolve_gh_host()
    if is_draft:
        for name in sorted(to_fetch):
            asset_id = asset_by_name[name]["id"]
            local_path = download_dir / name
            cmd = [
                "gh",
                "api",
                "--hostname",
                gh_host,
                "-H",
                "Accept: application/octet-stream",
                f"repos/{target_repo}/releases/assets/{asset_id}",
            ]
            with local_path.open("wb") as handle:
                res = subprocess.run(cmd, cwd=root, stdout=handle, stderr=subprocess.PIPE, check=False)
            if res.returncode != 0:
                local_path.unlink(missing_ok=True)
                detail = res.stderr.decode("utf-8", errors="replace").strip() or f"exit {res.returncode}"
                raise ReleaseError(f"gh api failed downloading draft asset {name!r} (id={asset_id}): {detail}")
    else:
        dl_cmd = [
            "gh",
            "release",
            "download",
            tag,
            "--repo",
            f"{gh_host}/{target_repo}",
            "--dir",
            str(download_dir),
            "--clobber",
        ]
        for name in sorted(to_fetch):
            dl_cmd.extend(["-p", name])
        res = subprocess.run(dl_cmd, cwd=root, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            detail = (res.stderr or res.stdout).strip() or f"exit {res.returncode}"
            raise ReleaseError(f"gh release download failed for {tag!r}: {detail}")

    local_sha256_by_name: dict[str, str] = {}
    for name in sorted(to_fetch):
        local_path = download_dir / name
        require(local_path.is_file() and not local_path.is_symlink(), f"Downloaded asset missing: {local_path}")
        actual_size = local_path.stat().st_size
        meta = asset_by_name[name]
        require(
            actual_size == meta["size"],
            f"Downloaded asset {name!r} size mismatch: local={actual_size} github={meta['size']}",
        )
        actual_hex = _sha256_file(local_path)
        actual_digest = "sha256:" + actual_hex
        require(
            actual_digest == meta["digest"],
            f"Downloaded asset {name!r} digest mismatch: local={actual_digest} github={meta['digest']}",
        )
        local_sha256_by_name[name] = actual_hex

    if require_sums:
        parsed_sums = _parse_sha256sums(download_dir / "SHA256SUMS")
        for name in sorted(required_assets):
            require(name in parsed_sums, f"SHA256SUMS on release {tag!r} is missing entry for {name}")
            actual_hex = local_sha256_by_name[name]
            require(
                parsed_sums[name] == actual_hex,
                f"SHA256SUMS mismatch for {name}: manifest={parsed_sums[name]} actual={actual_hex}",
            )

    return release, local_sha256_by_name


def verify_npm_tarballs(npm_dir: Path, version: str) -> list[tuple[str, Path, str]]:
    items: list[tuple[str, Path, str]] = []
    for pkg_name, prefix, dist_files in NPM_PACKAGES:
        tgz_path = npm_dir / f"{prefix}-{version}.tgz"
        require(tgz_path.is_file(), f"Missing npm tarball: {tgz_path}")
        with tarfile.open(tgz_path, "r:gz") as archive:
            names = {member.name.lstrip("./") for member in archive.getmembers() if member.isfile()}
            require("package/package.json" in names, f"{tgz_path.name} is missing package/package.json")
            pkg_member = archive.extractfile("package/package.json")
            require(pkg_member is not None, f"Cannot read package/package.json from {tgz_path.name}")
            assert pkg_member is not None
            manifest = json.loads(pkg_member.read().decode("utf-8"))
            require(
                manifest.get("name") == pkg_name,
                f"{tgz_path.name} package.json name mismatch: {manifest.get('name')!r} != {pkg_name!r}",
            )
            require(
                manifest.get("version") == version,
                f"{tgz_path.name} package.json version mismatch: {manifest.get('version')!r} != {version!r}",
            )
            expected_members = {
                "package/package.json",
                "package/README.md",
                "package/LICENSE",
                *(f"package/dist/{version}/{filename}" for filename in dist_files),
            }
            missing = sorted(expected_members - names)
            require(
                not missing,
                f"{tgz_path.name} is missing expected packed file(s): {', '.join(missing)}",
            )
        items.append((pkg_name, tgz_path, _sha512_sri_file(tgz_path)))
    return items


def _parse_npm_json(output: str):
    text = output.strip()
    if not text:
        raise ReleaseError("npm returned empty output")
    starts = [0] + [m.start() + 1 for m in re.finditer(r"\n(?=[\[{])", text)]
    for idx in reversed(starts):
        try:
            return json.loads(text[idx:])
        except json.JSONDecodeError:
            continue
    raise ReleaseError(f"npm did not return valid JSON: {text[:200]!r}")


def publish_npm_local(
    *,
    tag: str,
    repo: str = "",
    npm_tag: str = "",
    otp: str = "",
    allow_prerelease: bool = False,
    allow_draft: bool = False,
    dry_run: bool = False,
    root: Path = ROOT,
) -> list[str]:
    require(shutil.which("npm") is not None, "npm is required to publish npm packages")
    require(tag.startswith("v"), f"Release tag must start with 'v': {tag!r}")
    ver = tag[1:]
    parsed_ver = parse_semver(ver)
    if parsed_ver.is_prerelease and not allow_prerelease:
        raise ReleaseError(
            f"Refusing to publish prerelease version {ver!r} to npm (pass --allow-prerelease to override)"
        )
    default_prerelease_tag = (
        parsed_ver.prerelease[0]
        if (parsed_ver.is_prerelease and any(ch.isalpha() for ch in parsed_ver.prerelease[0]))
        else "next"
    )
    resolved_npm_tag = npm_tag.strip() or (default_prerelease_tag if parsed_ver.is_prerelease else "latest")
    if parsed_ver.is_prerelease and resolved_npm_tag == "latest":
        raise ReleaseError(
            f"Refusing to publish prerelease version {ver!r} under npm dist-tag 'latest' "
            f"(use --npm-tag {default_prerelease_tag} or another non-latest tag)"
        )

    with tempfile.TemporaryDirectory(prefix="volvoxai-npm-local-") as work_tmp:
        npm_dir = Path(work_tmp) / "downloaded-npm"
        download_github_release_assets(
            tag=tag,
            required_assets=expected_npm_filenames(ver),
            repo=repo,
            download_dir=npm_dir,
            allow_prerelease=allow_prerelease,
            allow_draft=allow_draft,
            require_sums=True,
            root=root,
        )
        items = verify_npm_tarballs(npm_dir, ver)

        if dry_run:
            for pkg_name, tgz_path, sri in items:
                print(
                    f"Dry run (npm -> {resolved_npm_tag}): {pkg_name}@{ver} "
                    f"({tgz_path.name}, {sri[:24]}...)",
                    flush=True,
                )
            return [f"{pkg_name}@{ver}" for pkg_name, _, _ in items]

        pending: list[tuple[str, Path, str]] = []
        for pkg_name, tgz_path, sri in items:
            spec = f"{pkg_name}@{ver}"
            view_res = subprocess.run(
                ["npm", "view", spec, "--json"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            if view_res.returncode != 0:
                payload = None
                try:
                    payload = _parse_npm_json(view_res.stdout or view_res.stderr)
                except ReleaseError:
                    pass
                err_code = payload.get("error", {}).get("code") if isinstance(payload, dict) else ""
                if err_code == "E404" or "E404" in (view_res.stderr + view_res.stdout):
                    pending.append((pkg_name, tgz_path, sri))
                    continue
                raise ReleaseError(
                    f"Cannot query npm registry for {spec}: {(view_res.stderr or view_res.stdout).strip()}"
                )

            raw_view = view_res.stdout.strip()
            payload = _parse_npm_json(raw_view) if raw_view else None
            if payload is None or payload == []:
                ver_res = subprocess.run(
                    ["npm", "view", pkg_name, "versions", "--json"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if ver_res.returncode != 0:
                    ver_payload = None
                    try:
                        ver_payload = _parse_npm_json(ver_res.stdout or ver_res.stderr)
                    except ReleaseError:
                        pass
                    ver_code = ver_payload.get("error", {}).get("code") if isinstance(ver_payload, dict) else ""
                    if ver_code == "E404" or "E404" in (ver_res.stderr + ver_res.stdout):
                        pending.append((pkg_name, tgz_path, sri))
                        continue
                    raise ReleaseError(
                        f"Cannot query npm registry versions for {pkg_name}: "
                        f"{(ver_res.stderr or ver_res.stdout).strip()}"
                    )
                versions_payload = _parse_npm_json(ver_res.stdout)
                if isinstance(versions_payload, str):
                    published_versions = {versions_payload}
                elif isinstance(versions_payload, list) and all(isinstance(v, str) for v in versions_payload):
                    published_versions = set(versions_payload)
                else:
                    raise ReleaseError(f"Unexpected npm versions payload for {pkg_name}: {versions_payload!r}")
                if ver not in published_versions:
                    pending.append((pkg_name, tgz_path, sri))
                    continue
                raise ReleaseError(
                    f"{spec} appears in npm versions list, but npm view returned empty metadata; refusing to overwrite."
                )

            if isinstance(payload, list):
                require(
                    len(payload) == 1 and isinstance(payload[0], dict),
                    f"Unexpected npm view payload for {spec}: {payload!r}",
                )
                payload = payload[0]
            require(isinstance(payload, dict), f"Unexpected npm view payload for {spec}: {payload!r}")
            if "version" in payload:
                require(
                    payload.get("version") == ver,
                    f"npm view returned unexpected version metadata for {spec}: {payload.get('version')!r}",
                )
            dist = payload.get("dist")
            published_sri = dist.get("integrity") if isinstance(dist, dict) else None
            require(
                isinstance(published_sri, str) and bool(published_sri.strip()),
                f"{spec} already exists on npm without dist.integrity; refusing to overwrite.",
            )
            if published_sri == sri:
                print(f"{spec} is already published on npm with identical bytes; skipping.", flush=True)
            else:
                raise ReleaseError(
                    f"{spec} already exists on npm with different integrity "
                    f"(registry={published_sri!r}, local={sri!r}); refusing to overwrite."
                )

        if not pending:
            print(f"All {len(items)} npm packages for {ver} are already published with matching integrity.", flush=True)
            return []

        whoami = subprocess.run(["npm", "whoami"], cwd=root, capture_output=True, text=True, check=False)
        if whoami.returncode != 0:
            raise ReleaseError("npm is not authenticated. Run 'npm login' locally before publishing.")

        published: list[str] = []
        for pkg_name, tgz_path, sri in pending:
            require(_sha512_sri_file(tgz_path) == sri, f"{tgz_path.name} changed before upload.")
            cmd = ["npm", "publish", str(tgz_path), "--access", "public", "--tag", resolved_npm_tag]
            if otp.strip():
                cmd.extend(["--otp", otp.strip()])
            res = subprocess.run(cmd, cwd=root, check=False)
            if res.returncode != 0:
                raise ReleaseError(
                    f"Publishing {pkg_name}@{ver} failed (exit {res.returncode}). "
                    f"Published in this run: {', '.join(published) or 'none'}."
                )
            published.append(f"{pkg_name}@{ver}")
        print(f"npm publish complete ({resolved_npm_tag}): {', '.join(published)}.", flush=True)
        return published


def _twine_command() -> list[str]:
    if shutil.which("twine") is not None:
        return ["twine"]
    check = subprocess.run(
        [sys.executable, "-m", "twine", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode == 0:
        return [sys.executable, "-m", "twine"]
    raise ReleaseError(
        "twine is required to check and upload Python wheels locally (install via 'pip install twine')."
    )


def verify_wheel_archive(wheel_path: Path, version: str) -> str:
    require(wheel_path.is_file(), f"Missing Python wheel: {wheel_path}")
    with zipfile.ZipFile(wheel_path, "r") as archive:
        names = set(archive.namelist())
        metadata_candidates = [n for n in names if n.endswith(".dist-info/METADATA")]
        require(
            len(metadata_candidates) == 1,
            f"{wheel_path.name} must contain one .dist-info/METADATA file; found {metadata_candidates}",
        )
        metadata = archive.read(metadata_candidates[0]).decode("utf-8")
        headers: dict[str, str] = {}
        for line in metadata.splitlines():
            if not line.strip():
                break
            if ":" in line:
                key, value = line.split(":", 1)
                headers.setdefault(key.strip(), value.strip())
        require(
            headers.get("Name") == "volvoxai",
            f"{wheel_path.name} METADATA Name mismatch: {headers.get('Name')!r}",
        )
        require(
            headers.get("Version") == version,
            f"{wheel_path.name} METADATA Version mismatch: {headers.get('Version')!r} != {version!r}",
        )
        required_payloads = {
            "volvoxai/libvolvoxai.so",
            "volvoxai/libsynurang_module_host.so",
            "volvoxai/bin/volvoxai",
            "volvoxai/BUILD-INFO.json",
        }
        missing = sorted(required_payloads - names)
        require(
            not missing,
            f"{wheel_path.name} is missing expected wheel payload(s): {', '.join(missing)}",
        )
    return _sha256_file(wheel_path)


def _check_pypi_existing(version: str, wheel_name: str, local_sha256: str) -> bool:
    """Return True if wheel already exists on PyPI with identical SHA-256."""
    url = f"https://pypi.org/pypi/volvoxai/{version}/json"
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "volvoxai-release-local"})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise ReleaseError(f"Cannot query PyPI for volvoxai=={version}: HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise ReleaseError(f"Cannot reach PyPI to inspect volvoxai=={version}: {error.reason}") from error

    for item in payload.get("urls") or []:
        if item.get("filename") != wheel_name:
            continue
        remote_sha = (item.get("digests") or {}).get("sha256")
        if remote_sha == local_sha256:
            return True
        raise ReleaseError(
            f"PyPI already has {wheel_name} with different SHA-256 "
            f"(remote={remote_sha!r}, local={local_sha256!r}); refusing to overwrite."
        )
    return False


def publish_pypi_local(
    *,
    tag: str,
    repo: str = "",
    repository: str = "",
    repository_url: str = "",
    allow_prerelease: bool = False,
    allow_draft: bool = False,
    dry_run: bool = False,
    root: Path = ROOT,
) -> list[str]:
    require(tag.startswith("v"), f"Release tag must start with 'v': {tag!r}")
    ver = tag[1:]
    parsed_ver = parse_semver(ver)
    if parsed_ver.is_prerelease and not allow_prerelease:
        raise ReleaseError(
            f"Refusing to publish prerelease version {ver!r} to PyPI (pass --allow-prerelease to override)"
        )
    twine = _twine_command()
    wheel_name = expected_wheel_filename(ver)

    with tempfile.TemporaryDirectory(prefix="volvoxai-pypi-local-") as work_tmp:
        wheel_dir = Path(work_tmp) / "downloaded-wheel"
        download_github_release_assets(
            tag=tag,
            required_assets={wheel_name},
            repo=repo,
            download_dir=wheel_dir,
            allow_prerelease=allow_prerelease,
            allow_draft=allow_draft,
            require_sums=True,
            root=root,
        )
        wheel_path = wheel_dir / wheel_name
        local_sha = verify_wheel_archive(wheel_path, ver)

        check_res = subprocess.run(
            [*twine, "check", "--strict", str(wheel_path)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if check_res.returncode != 0:
            detail = (check_res.stderr or check_res.stdout).strip() or f"exit {check_res.returncode}"
            raise ReleaseError(f"twine check --strict failed for {wheel_name}: {detail}")

        if dry_run:
            print(f"Dry run (PyPI): {wheel_name} (sha256:{local_sha[:16]}...)", flush=True)
            return [wheel_name]

        custom_repo = (
            repository.strip()
            or repository_url.strip()
            or (os.environ.get("TWINE_REPOSITORY") or "").strip()
            or (os.environ.get("TWINE_REPOSITORY_URL") or "").strip()
        )
        if not custom_repo and _check_pypi_existing(ver, wheel_name, local_sha):
            print(f"{wheel_name} is already published on PyPI with identical SHA-256; skipping.", flush=True)
            return []

        require(_sha256_file(wheel_path) == local_sha, f"{wheel_name} changed before upload.")
        upload_cmd = [*twine, "upload"]
        if repository.strip():
            upload_cmd.extend(["--repository", repository.strip()])
        if repository_url.strip():
            upload_cmd.extend(["--repository-url", repository_url.strip()])
        upload_cmd.append(str(wheel_path))
        upload_res = subprocess.run(upload_cmd, cwd=root, check=False)
        if upload_res.returncode != 0:
            raise ReleaseError(f"twine upload failed for {wheel_name} (exit {upload_res.returncode}).")
        print(f"PyPI publish complete: {wheel_name}.", flush=True)
        return [wheel_name]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    npm_parser = subparsers.add_parser(
        "publish-npm-local",
        help="Download verified npm tarballs from a GitHub Release and publish locally.",
    )
    npm_parser.add_argument("--tag", required=True, help="GitHub Release tag (for example v0.9.2)")
    npm_parser.add_argument("--repo", default="", help="GitHub repository slug (default: origin remote)")
    npm_parser.add_argument("--npm-tag", default="", help="Explicit npm dist-tag")
    npm_parser.add_argument("--otp", default="", help="npm 2FA one-time password")
    npm_parser.add_argument("--allow-prerelease", action="store_true")
    npm_parser.add_argument("--allow-draft", action="store_true")
    npm_parser.add_argument("--dry-run", action="store_true")

    pypi_parser = subparsers.add_parser(
        "publish-pypi-local",
        help="Download verified Python wheel from a GitHub Release and publish locally via twine.",
    )
    pypi_parser.add_argument("--tag", required=True, help="GitHub Release tag (for example v0.9.2)")
    pypi_parser.add_argument("--repo", default="", help="GitHub repository slug (default: origin remote)")
    pypi_parser.add_argument("--repository", default="", help="twine --repository name (for example testpypi)")
    pypi_parser.add_argument("--repository-url", default="", help="twine --repository-url")
    pypi_parser.add_argument("--allow-prerelease", action="store_true")
    pypi_parser.add_argument("--allow-draft", action="store_true")
    pypi_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "publish-npm-local":
            publish_npm_local(
                tag=args.tag,
                repo=args.repo,
                npm_tag=args.npm_tag,
                otp=args.otp,
                allow_prerelease=args.allow_prerelease,
                allow_draft=args.allow_draft,
                dry_run=args.dry_run,
            )
        elif args.command == "publish-pypi-local":
            publish_pypi_local(
                tag=args.tag,
                repo=args.repo,
                repository=args.repository,
                repository_url=args.repository_url,
                allow_prerelease=args.allow_prerelease,
                allow_draft=args.allow_draft,
                dry_run=args.dry_run,
            )
        return 0
    except ReleaseError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
