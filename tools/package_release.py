#!/usr/bin/env python3
"""Create a deterministic inference or full runtime package."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode()


def package_files(root, profile="inference"):
    if profile not in ("inference", "full"):
        raise ValueError("profile must be inference or full")
    version = json.loads((root / "package.json").read_text())["version"]
    stem = "volvoxai.full" if profile == "full" else "volvoxai"
    native = "volvoxai-full" if profile == "full" else "volvoxai"
    names = [f"dist/{version}/{stem}{suffix}" for suffix in (".js", ".min.js", ".wasm")]
    names.append(f"native/{native}")
    files = {name: (root / name).read_bytes() for name in names}
    entries = {name: {"bytes": len(data), "sha256": digest(data)}
               for name, data in sorted(files.items())}
    contract = {
        "proto_sha256": digest((root / "proto/volvoxai.proto").read_bytes()),
        "operator_registry_sha256": digest((root / "proto/kernel_registry.proto").read_bytes()),
        "wasm_abi_sha256": re.search(r'"manifestSha256": "([0-9a-f]+)"',
                                     (root / "tools/generated/wasmInternalAbi.mjs").read_text())[1],
    }
    if profile == "full":
        contract.update({
            "gpu_bridge_abi_sha256": re.search(r'GPU_BRIDGE_ABI_HASH = "([0-9a-f]+)"',
                                               (root / "ts/generated/gpuBridge.ts").read_text())[1],
            "shader_catalogue_sha256": re.search(r'Shader source SHA-256: ([0-9a-f]+)',
                                                 (root / "ts/generated/shaderCatalog.ts").read_text())[1],
        })
    manifest = {"format": "volvoxai-package/v1", "version": version,
                "profile": profile, "files": entries,
                "content_sha256": digest(canonical(entries)), "contract": contract}
    files["manifest.json"] = canonical(manifest)
    return files, manifest


def write_archive(destination, files):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name.startswith("native/") else 0o100644) << 16
            archive.writestr(info, data, compresslevel=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("inference", "full"), default="inference")
    args = parser.parse_args()
    protected = [*(ROOT / "dist").rglob("*"), ROOT / "native/volvoxai", ROOT / "native/volvoxai-full"]
    if args.out.resolve() in {source.resolve() for source in protected}:
        parser.error("--out must not overwrite a release artifact")
    files, manifest = package_files(ROOT, args.profile)
    write_archive(args.out, files)
    print(json.dumps({"archive": str(args.out), "sha256": digest(args.out.read_bytes()),
                      "content_sha256": manifest["content_sha256"], "profile": args.profile}))


if __name__ == "__main__":
    main()
