"""Minimal safetensors reader/writer for the PTQ materialization step.

safetensors is a flat container: an 8-byte little-endian header length, a JSON
header mapping tensor name -> ``{dtype, shape, data_offsets}`` (offsets relative
to the start of the data section), then the concatenated little-endian row-major
tensor bytes. This is the I/O half the PTQ orchestrator needs to read float32
weights and write their int8 replacements; it deliberately supports only the
dtypes the runtime executes.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from ..quantization_storage import reject_legacy_safetensors_metadata
from ..runtime_names import is_runtime_graph_name

_DTYPE_TO_NUMPY = {
    "F32": np.float32,
    "F16": np.float16,
    "I8": np.int8,
    "U8": np.uint8,
    "I32": np.int32,
}
_NUMPY_TO_DTYPE = {np.dtype(value): key for key, value in _DTYPE_TO_NUMPY.items()}
_MAX_SAFE_INTEGER = (1 << 53) - 1
_TENSOR_FIELDS = frozenset({"dtype", "shape", "data_offsets"})


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"safetensors header contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(
        f"safetensors header contains non-standard JSON constant {value!r}"
    )


def _is_safe_nonnegative_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= _MAX_SAFE_INTEGER
    )


def _validate_metadata(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(
            "safetensors __metadata__ must be an object mapping strings to strings"
        )
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(
                "safetensors __metadata__ must be an object mapping strings "
                "to strings"
            )
    reject_legacy_safetensors_metadata(value)
    return value


def _decode_header(raw: bytes) -> tuple[dict[str, Any], int]:
    if len(raw) < 8:
        raise ValueError("safetensors file is too small to hold a header length")
    header_length = struct.unpack("<Q", raw[:8])[0]
    if (
        header_length == 0
        or header_length > _MAX_SAFE_INTEGER
        or header_length > len(raw) - 8
    ):
        raise ValueError("safetensors header length is invalid")
    header_end = 8 + header_length
    try:
        header_text = raw[8:header_end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("safetensors header must be valid UTF-8") from error
    try:
        header = json.loads(
            header_text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("safetensors header must contain valid JSON") from error
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    return header, header_end


def read_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    raw = Path(path).read_bytes()
    header, header_end = _decode_header(raw)
    if "__metadata__" in header:
        _validate_metadata(header["__metadata__"])
    body = raw[header_end:]
    records: list[tuple[str, np.dtype[Any], list[int], int, int]] = []
    spans: list[tuple[int, str, int]] = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if (
            not is_runtime_graph_name(name)
            or not isinstance(meta, dict)
            or set(meta) != _TENSOR_FIELDS
        ):
            raise ValueError(
                f"safetensors tensor {name!r} has an invalid header record"
            )
        dtype_name = meta["dtype"]
        dtype = _DTYPE_TO_NUMPY.get(dtype_name) if isinstance(dtype_name, str) else None
        if dtype is None:
            raise ValueError(f"unsupported safetensors dtype {dtype_name!r}")
        shape = meta["shape"]
        if not isinstance(shape, list) or any(
            not _is_safe_nonnegative_integer(dimension) for dimension in shape
        ):
            raise ValueError(f"safetensors tensor {name!r} has an invalid shape")
        element_count = 1
        for dimension in shape:
            element_count *= dimension
            if element_count > _MAX_SAFE_INTEGER:
                raise ValueError(f"safetensors tensor {name!r} shape is too large")
        expected_bytes = element_count * np.dtype(dtype).itemsize
        if expected_bytes > _MAX_SAFE_INTEGER:
            raise ValueError(f"safetensors tensor {name!r} shape is too large")
        offsets = meta["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not _is_safe_nonnegative_integer(offset) for offset in offsets)
        ):
            raise ValueError(
                f"safetensors tensor {name!r} has invalid data offsets"
            )
        start, end = offsets
        if (
            end < start
            or end > len(body)
            or end - start != expected_bytes
        ):
            raise ValueError(
                f"safetensors tensor {name!r} data span is invalid"
            )
        records.append((name, np.dtype(dtype), shape, start, end))
        spans.append((start, name, end))

    spans.sort()
    cursor = 0
    for start, name, end in spans:
        if start != cursor:
            raise ValueError(
                "safetensors data has a gap, overlap, or aliased span near "
                f"{name!r}"
            )
        cursor = end
    if cursor != len(body):
        raise ValueError(
            "safetensors tensor spans do not account for the full data payload"
        )

    tensors: dict[str, np.ndarray] = {}
    for name, dtype, shape, start, end in records:
        array = np.frombuffer(body[start:end], dtype=dtype)
        tensors[name] = array.reshape(shape)
    return tensors


def write_safetensors(
    path: str | Path,
    tensors: dict[str, np.ndarray],
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    stored_metadata = _validate_metadata(metadata) if metadata is not None else None
    for name in tensors:
        if not is_runtime_graph_name(name):
            raise ValueError(
                "safetensors tensor names must satisfy runtime Graph.validName"
            )
    header: dict[str, Any] = {}
    body = bytearray()
    for name in sorted(tensors):
        array = tensors[name]
        array = np.ascontiguousarray(array)
        dtype = _NUMPY_TO_DTYPE.get(array.dtype)
        if dtype is None:
            raise ValueError(f"unsupported numpy dtype {array.dtype!r}")
        payload = array.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [len(body), len(body) + len(payload)],
        }
        body += payload
    if stored_metadata is not None:
        header["__metadata__"] = stored_metadata
    header_bytes = json.dumps(
        header, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    temporary = Path(path).with_name(f".{Path(path).name}.tmp")
    with open(temporary, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(body)
    temporary.replace(path)
