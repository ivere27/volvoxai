#!/usr/bin/env python3
"""Download and verify the pinned, unquantized Kokoro model and one voice."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.request

REPOSITORY = "onnx-community/Kokoro-82M-v1.0-ONNX"
REVISION = "1939ad2a8e416c0acfeecc08a694d14ef25f2231"
FILES = {
    "config.json": "df34b4f930b23447cd4dc410fabfb42eb3f24e803e6c3f97d618fb359380a36f",
    "tokenizer.json": "77a02c8e164413299b4b4c403b14f8e0e1c1b727db4d46a09d6327b861060a34",
    "tokenizer_config.json": "be1cb066d6ef6b074b3f15e6a6dd21ac88ff3cdaedf325f0aaed686c70f75d20",
    "voices/af_heart.bin": "d583ccff3cdca2f7fae535cb998ac07e9fcb90f09737b9a41fa2734ec44a8f0b",
    "onnx/model.onnx": "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fetch(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = root / name
        if path.is_file():
            if digest(path) != expected:
                raise RuntimeError(f"Existing asset has the wrong SHA256: {path}")
            print(f"Verified {path}", flush=True)
            continue
        if shutil.disk_usage(root).free < 500_000_000:
            raise RuntimeError("Kokoro download needs at least 500 MB of free disk space")
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                with urllib.request.urlopen(url, timeout=120) as response:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            if digest(temporary) != expected:
                raise RuntimeError(f"Downloaded asset has the wrong SHA256: {name}")
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"Downloaded {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("build/tts-examples/assets/kokoro"))
    fetch(parser.parse_args().assets)
