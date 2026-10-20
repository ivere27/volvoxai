#!/usr/bin/env python3
"""Remove compiler line comments while preserving PTX tokens and line numbers.

PTX comments follow C/C++ delimiters, excluding quoted strings and character
constants: https://docs.nvidia.com/cuda/parallel-thread-execution/#comments
Block comments and all non-comment bytes remain verbatim. The compiler output
is a separate input artifact, never modified by this step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def strip_line_comments(source: bytes) -> tuple[bytes, int]:
    if not source or b"\0" in source or len(source) >= 64 * 1024 * 1024:
        raise ValueError("invalid or oversized PTX source")
    output = bytearray()
    offset = 0
    comments = 0
    while offset < len(source):
        byte = source[offset]
        if byte in (ord('"'), ord("'")):
            start = offset
            quote = byte
            offset += 1
            while offset < len(source):
                byte = source[offset]
                offset += 1
                if byte == ord("\\"):
                    if offset == len(source):
                        raise ValueError("unterminated PTX quoted escape")
                    offset += 1
                elif byte == quote:
                    break
            else:
                raise ValueError("unterminated PTX quoted literal")
            output.extend(source[start:offset])
        elif source[offset:offset + 2] == b"/*":
            end = source.find(b"*/", offset + 2)
            if end < 0:
                raise ValueError("unterminated PTX block comment")
            output.extend(source[offset:end + 2])
            offset = end + 2
        elif source[offset:offset + 2] == b"//":
            comments += 1
            offset += 2
            # Preserve CRLF as well as LF. No following token can join the
            # previous token: a line comment ends at a newline or EOF.
            end = source.find(b"\n", offset)
            if end < 0:
                end = len(source)
            if end > offset and source[end - 1] == 13:
                output.append(13)
            offset = end
        else:
            output.append(byte)
            offset += 1
    result = bytes(output)
    if result.count(b"\n") != source.count(b"\n"):
        raise AssertionError("PTX line numbers changed")
    return result, comments


def prepare_ptx(input_path: Path, output_path: Path, manifest_path: Path) -> None:
    paths = [p.resolve() for p in (input_path, output_path, manifest_path)]
    if len(set(paths)) != 3:
        raise ValueError("compiler PTX, prepared PTX and provenance must be separate files")
    for index, path in enumerate(paths):
        for other in paths[index + 1:]:
            if path.exists() and other.exists() and path.samefile(other):
                raise ValueError("PTX preparation files must not share an inode")
    source = input_path.read_bytes()
    prepared, comments = strip_line_comments(source)
    digest = lambda data: hashlib.sha256(data).hexdigest()
    report = {
        "schema": "volvoxai.cuda-ptx-preparation.v1",
        "transformation": "remove line comments outside quoted literals and block comments",
        "compiler_ptx": {"file": input_path.name, "bytes": len(source), "sha256": digest(source)},
        "prepared_ptx": {"file": output_path.name, "bytes": len(prepared), "sha256": digest(prepared)},
        "line_comments_removed": comments,
        "newline_count": source.count(b"\n"),
        "block_comments": "preserved verbatim",
        "quoted_literals": "preserved verbatim including escapes",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(prepared)
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    prepare_ptx(args.input, args.output, args.manifest)


if __name__ == "__main__":
    main()
