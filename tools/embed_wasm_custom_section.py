#!/usr/bin/env python3
"""Deterministically replace one WebAssembly custom section with file bytes."""

from __future__ import annotations

import argparse
from pathlib import Path


WASM_HEADER = b"\x00asm\x01\x00\x00\x00"


def decode_u32(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    start = offset
    while offset < len(data) and shift < 35:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            if value > 0xFFFFFFFF:
                raise ValueError(f"varuint32 overflow at byte {start}")
            return value, offset
        shift += 7
    raise ValueError(f"unterminated varuint32 at byte {start}")


def encode_u32(value: int) -> bytes:
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"value {value} is outside varuint32")
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        encoded.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(encoded)


def custom_section_name(payload: bytes) -> bytes:
    name_size, name_offset = decode_u32(payload, 0)
    name_end = name_offset + name_size
    if name_end > len(payload):
        raise ValueError("custom section name extends beyond its payload")
    return payload[name_offset:name_end]


def replace_custom_section(module: bytes, name: str, contents: bytes) -> bytes:
    if not module.startswith(WASM_HEADER):
        raise ValueError("input is not a version-1 WebAssembly module")
    encoded_name = name.encode("utf-8")
    retained: list[bytes] = []
    offset = len(WASM_HEADER)
    while offset < len(module):
        section_start = offset
        section_id = module[offset]
        offset += 1
        section_size, offset = decode_u32(module, offset)
        section_end = offset + section_size
        if section_end > len(module):
            raise ValueError(f"section at byte {section_start} exceeds the module")
        payload = module[offset:section_end]
        if section_id != 0 or custom_section_name(payload) != encoded_name:
            retained.append(module[section_start:section_end])
        offset = section_end
    payload = encode_u32(len(encoded_name)) + encoded_name + contents
    section = b"\x00" + encode_u32(len(payload)) + payload
    return WASM_HEADER + b"".join(retained) + section


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="baseline parent .wasm module")
    parser.add_argument("--payload", required=True, type=Path,
                        help="file embedded as the custom-section contents")
    parser.add_argument("--section-name", required=True,
                        help="UTF-8 custom-section name to replace")
    parser.add_argument("--output", required=True, type=Path,
                        help="output .wasm; may be the same path as --input")
    args = parser.parse_args()

    module = args.input.read_bytes()
    contents = args.payload.read_bytes()
    result = replace_custom_section(module, args.section_name, contents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result)


if __name__ == "__main__":
    main()
