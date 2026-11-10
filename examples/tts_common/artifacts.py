"""Write one prepared TTS request for native, ORT, and browser callers."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Mapping

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_case(root: Path, case_id: str, inputs: Mapping[str, np.ndarray],
               outputs: Mapping[str, np.ndarray], *, audio_output: str = "audio",
               sample_rate: int | None = None) -> dict:
    """Preserve semantic names/dtypes and emit little-endian browser inputs.

    Returned paths are relative to ``root``, which also holds manifest.json.
    Each case owns its directory; an existing case is never overwritten.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id):
        raise ValueError("case_id must be a simple filename component")
    if not inputs or audio_output not in outputs:
        raise ValueError("inputs and the declared audio output are required")
    if sample_rate is not None and (not isinstance(sample_rate, int) or sample_rate <= 0):
        raise ValueError("sample_rate must be a positive integer")
    root = Path(root)
    directory = root / "cases" / case_id
    directory.mkdir(parents=True, exist_ok=False)

    def write_tensors(values, prefix):
        arrays, descriptors = {}, []
        for index, (name, value) in enumerate(values.items()):
            if not isinstance(name, str) or not name:
                raise ValueError("tensor semantic names must be nonempty strings")
            array = np.asarray(value)
            if array.dtype.name not in {"float32", "float64", "int32", "int64", "bool"}:
                raise ValueError(f"unsupported prepared tensor dtype: {array.dtype}")
            # np.ascontiguousarray promotes scalars to rank one. Preserve rank.
            array = np.array(array, dtype=array.dtype.newbyteorder("<"), order="C", copy=True)
            arrays[name] = array
            path = directory / f"{prefix}-{index}.bin"
            path.write_bytes(array.tobytes(order="C"))
            descriptors.append({"name": name, "dtype": array.dtype.name,
                "shape": list(array.shape), "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path)})
        path = directory / ("inputs.npz" if prefix == "input" else "reference.npz")
        np.savez(path, **arrays)
        return path.relative_to(root).as_posix(), descriptors

    input_path, input_files = write_tensors(inputs, "input")
    output_path, output_files = write_tensors(outputs, "output")
    result = {"id": case_id, "inputs": input_path, "reference": output_path,
        "input_files": input_files, "output_files": output_files,
        "audio_output": audio_output}
    if sample_rate is not None:
        result["sample_rate"] = sample_rate
    return result
