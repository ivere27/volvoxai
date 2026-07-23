from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from tools.exporter.optimizer.safetensors_io import (
    read_safetensors,
    write_safetensors,
)


def _raw_safetensors(header: str | Any, body: bytes = b"") -> bytes:
    if not isinstance(header, str):
        header = json.dumps(header, separators=(",", ":"))
    encoded = header.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + body


class SafetensorsReaderContractTest(unittest.TestCase):
    def _read(self, payload: bytes) -> dict[str, np.ndarray]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            path.write_bytes(payload)
            return read_safetensors(path)

    def test_accepts_valid_empty_container_without_payload(self):
        self.assertEqual(self._read(_raw_safetensors({})), {})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            write_safetensors(path, {})
            self.assertEqual(read_safetensors(path), {})

    def test_round_trips_only_current_persisted_dtypes(self):
        tensors = {
            "f16": np.asarray([1.5], dtype=np.float16),
            "f32": np.asarray([-2.0], dtype=np.float32),
            "i32": np.asarray([1234], dtype=np.int32),
            "i8": np.asarray([-4], dtype=np.int8),
            "u8": np.asarray([250], dtype=np.uint8),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            write_safetensors(path, tensors, metadata={"profile": "runtime-v1"})
            restored = read_safetensors(path)
        self.assertEqual(set(restored), set(tensors))
        for name, expected in tensors.items():
            self.assertEqual(restored[name].dtype, expected.dtype)
            self.assertTrue(np.array_equal(restored[name], expected))

    def test_rejects_duplicate_json_keys_at_every_object_level(self):
        cases = (
            (
                '{"w":{"dtype":"I8","shape":[1],"data_offsets":[0,1]},'
                '"w":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}',
                b"\0",
                "w",
            ),
            ('{"__metadata__":{},"__metadata__":{}}', b"", "__metadata__"),
            (
                '{"__metadata__":{"profile":"a","profile":"b"}}',
                b"",
                "profile",
            ),
            (
                '{"w":{"dtype":"I8","dtype":"U8","shape":[1],'
                '"data_offsets":[0,1]}}',
                b"\0",
                "dtype",
            ),
        )
        for header, body, key in cases:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, f"duplicate JSON key '{key}'"):
                    self._read(_raw_safetensors(header, body))

    def test_rejects_invalid_header_envelope_and_json(self):
        cases = (
            (b"", "too small"),
            (struct.pack("<Q", 0), "header length is invalid"),
            (struct.pack("<Q", 2) + b"{", "header length is invalid"),
            (struct.pack("<Q", 1) + b"\xff", "valid UTF-8"),
            (_raw_safetensors("{"), "valid JSON"),
            (_raw_safetensors("[]"), "header must be an object"),
            (_raw_safetensors('{"x":NaN}'), "non-standard JSON constant"),
        )
        for payload, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._read(payload)

    def test_requires_exact_tensor_records_and_string_metadata(self):
        tensor = {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]}
        cases = (
            ({" ": tensor}, b"\0", "invalid header record"),
            ({"w": []}, b"", "invalid header record"),
            ({"w": {"dtype": "I8", "shape": [1]}}, b"\0", "invalid header record"),
            ({"w": {**tensor, "extra": True}}, b"\0", "invalid header record"),
            ({"__metadata__": None}, b"", "mapping strings to strings"),
            ({"__metadata__": []}, b"", "mapping strings to strings"),
            ({"__metadata__": {"profile": 1}}, b"", "mapping strings to strings"),
        )
        for header, body, message in cases:
            with self.subTest(header=header):
                with self.assertRaisesRegex(ValueError, message):
                    self._read(_raw_safetensors(header, body))

    def test_rejects_unsupported_dtype_and_invalid_shape(self):
        tensor = lambda dtype, shape, offsets: {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": offsets,
        }
        cases = (
            ({"w": tensor("F64", [1], [0, 8])}, bytes(8), "unsupported"),
            ({"w": tensor("I64", [1], [0, 8])}, bytes(8), "unsupported"),
            ({"w": tensor([], [1], [0, 1])}, b"\0", "unsupported"),
            ({"w": tensor("I8", True, [0, 1])}, b"\0", "invalid shape"),
            ({"w": tensor("I8", [True], [0, 1])}, b"\0", "invalid shape"),
            ({"w": tensor("I8", [-1], [0, 0])}, b"", "invalid shape"),
            ({"w": tensor("I8", [1.0], [0, 1])}, b"\0", "invalid shape"),
            (
                {"w": tensor("I8", [2**53], [0, 0])},
                b"",
                "invalid shape",
            ),
            (
                {"w": tensor("I8", [2**52, 4], [0, 0])},
                b"",
                "shape is too large",
            ),
        )
        for header, body, message in cases:
            with self.subTest(header=header):
                with self.assertRaisesRegex(ValueError, message):
                    self._read(_raw_safetensors(header, body))

    def test_requires_safe_exact_in_bounds_offsets(self):
        tensor = lambda shape, offsets: {
            "dtype": "I8",
            "shape": shape,
            "data_offsets": offsets,
        }
        cases = (
            ({"w": tensor([1], None)}, b"\0", "invalid data offsets"),
            ({"w": tensor([1], [0])}, b"\0", "invalid data offsets"),
            ({"w": tensor([1], [False, 1])}, b"\0", "invalid data offsets"),
            ({"w": tensor([1], [-1, 0])}, b"\0", "invalid data offsets"),
            ({"w": tensor([1], [0, 1.0])}, b"\0", "invalid data offsets"),
            ({"w": tensor([0], [0, 2**53])}, b"", "invalid data offsets"),
            ({"w": tensor([1], [1, 0])}, b"\0", "data span is invalid"),
            ({"w": tensor([1], [0, 2])}, b"\0", "data span is invalid"),
            ({"w": tensor([2], [0, 1])}, b"\0", "data span is invalid"),
        )
        for header, body, message in cases:
            with self.subTest(header=header):
                with self.assertRaisesRegex(ValueError, message):
                    self._read(_raw_safetensors(header, body))

    def test_rejects_alias_overlap_gap_and_uncovered_payload(self):
        tensor = lambda shape, offsets: {
            "dtype": "I8",
            "shape": shape,
            "data_offsets": offsets,
        }
        cases = (
            (
                {"a": tensor([1], [0, 1]), "b": tensor([1], [0, 1])},
                bytes(1),
                "gap, overlap, or aliased span",
            ),
            (
                {"a": tensor([2], [0, 2]), "b": tensor([2], [1, 3])},
                bytes(3),
                "gap, overlap, or aliased span",
            ),
            (
                {"a": tensor([1], [0, 1]), "b": tensor([1], [2, 3])},
                bytes(3),
                "gap, overlap, or aliased span",
            ),
            (
                {"a": tensor([1], [0, 1])},
                bytes(2),
                "full data payload",
            ),
            ({}, b"stowaway", "full data payload"),
        )
        for header, body, message in cases:
            with self.subTest(header=header):
                with self.assertRaisesRegex(ValueError, message):
                    self._read(_raw_safetensors(header, body))

    def test_writer_rejects_noncurrent_dtype_names_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            for array in (
                np.asarray([1], dtype=np.float64),
                np.asarray([1], dtype=np.int64),
                np.asarray([True], dtype=np.bool_),
            ):
                with self.subTest(dtype=array.dtype):
                    with self.assertRaisesRegex(ValueError, "unsupported numpy dtype"):
                        write_safetensors(path, {"w": array})
            with self.assertRaisesRegex(ValueError, "mapping strings to strings"):
                write_safetensors(
                    path,
                    {},
                    metadata={"profile": 1},  # type: ignore[dict-item]
                )
            with self.assertRaisesRegex(ValueError, "tensor names"):
                write_safetensors(path, {"__metadata__": np.asarray([1], np.int8)})


if __name__ == "__main__":
    unittest.main()
