#!/usr/bin/env python3
"""Generate or check the deterministic TinyReceipt split-ONNX E2E oracle.

The output is intentionally path-free.  It binds an independently executed
ONNX Runtime result to the hashes of one exported VolvoxAI package, so a
runtime candidate cannot be compared with a reference from another model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import onnxruntime as ort

from import_hf_split_onnx import (
    ImportFailure,
    PACKAGE_FORMAT,
    SOURCE_FORMAT,
    validate_source,
)
from tiny_receipt_tokenizer import TinyReceiptTokenizer, TokenizerContractError


REFERENCE_SCHEMA = "volvoxai.tiny-receipt-split-e2e-reference/v1"
WORKLOAD_ID = "synthetic-exact-f32-v1"
PROMPT = "phone number last one"
FAMILY_ORDER = (
    "phone",
    "address",
    "store",
    "item_row",
    "item_math",
    "item_lookup",
    "math",
    "other",
)
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 672
SEQUENCE_LENGTH = 192
MAX_NEW_TOKENS = 4
MINIMUM_DECODER_STEPS = 2
IMAGE_GENERATOR = "q=(17*x+29*y+7*(x^y))&255; f32=(q-128)/128"


class OracleFailure(RuntimeError):
    """A source, package, execution, or reference contract violation."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleFailure(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise OracleFailure(f"{label} root must be an object")
    return value


def _relative_asset(root: Path, record: Any, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
        raise OracleFailure(f"{label} must be an exact package asset record")
    relative = record.get("path")
    byte_count = record.get("bytes")
    digest = record.get("sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or any(part in {"", ".", ".."} for part in Path(relative).parts)
        or not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise OracleFailure(f"{label} is not a safe, complete package asset record")
    path = root.joinpath(*Path(relative).parts)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise OracleFailure(f"{label} is unavailable: {error}") from error
    if path.is_symlink() or not path.is_file():
        raise OracleFailure(f"{label} must be a regular non-symlink file")
    if metadata.st_size != byte_count or _sha256_file(path) != digest:
        raise OracleFailure(f"{label} does not match its declared bytes and SHA-256")
    return path


def _package_identity(
    package_root: Path,
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], TinyReceiptTokenizer]:
    manifest = _read_json(package_root / "package_manifest.json", "package manifest")
    if manifest.get("format") != PACKAGE_FORMAT:
        raise OracleFailure(f"package format must be {PACKAGE_FORMAT!r}")
    package_source = manifest.get("source")
    if (
        not isinstance(package_source, Mapping)
        or package_source.get("format") != SOURCE_FORMAT
        or package_source.get("variant") != "int8-w8a8"
    ):
        raise OracleFailure("package must identify the static INT8 split-ONNX source")
    selected_keys = source["selected_file_keys"]
    expected_source_hashes = {
        "manifest": source["hashes"]["manifest"],
        "encoder_onnx": source["hashes"][selected_keys["encoder"]],
        "decoder_onnx": source["hashes"][selected_keys["decoder"]],
    }
    for name, expected_digest in expected_source_hashes.items():
        record = package_source.get(name)
        if not isinstance(record, Mapping) or record.get("sha256") != expected_digest:
            raise OracleFailure(f"package source.{name} does not match the validated source")

    graphs = manifest.get("graphs")
    assets = manifest.get("assets")
    if not isinstance(graphs, Mapping) or not isinstance(assets, Mapping):
        raise OracleFailure("package manifest requires graphs and assets")
    encoder = graphs.get("encoder")
    decoder = graphs.get("decoder")
    if not isinstance(encoder, Mapping) or not isinstance(decoder, Mapping):
        raise OracleFailure("package manifest requires encoder and decoder graphs")
    records = {
        "encoderGraph": encoder.get("graph"),
        "encoderWeights": encoder.get("weights"),
        "decoderGraph": decoder.get("graph"),
        "decoderWeights": decoder.get("weights"),
    }
    for name, record in records.items():
        _relative_asset(package_root, record, f"package {name}")
    vocabulary_path = _relative_asset(
        package_root,
        assets.get("vocab"),
        "package vocabulary",
    )
    vocabulary = _read_json(vocabulary_path, "package vocabulary")
    if vocabulary != source["vocab"]:
        raise OracleFailure("package vocabulary differs from the validated source")
    try:
        tokenizer = TinyReceiptTokenizer.from_documents(
            manifest.get("tokenizer"),
            vocabulary,
        )
    except TokenizerContractError as error:
        raise OracleFailure(f"package tokenizer is invalid: {error}") from error
    if tokenizer.vocab_size != source["vocab_size"]:
        raise OracleFailure("package tokenizer size differs from the validated source")
    return (
        {
            "format": PACKAGE_FORMAT,
            "assets": {
                name: records[name]["sha256"]
                for name in (
                    "encoderGraph",
                    "encoderWeights",
                    "decoderGraph",
                    "decoderWeights",
                )
            },
            "vocabularySha256": tokenizer.package_vocabulary_sha256,
        },
        tokenizer,
    )


def _workload(
    tokenizer: TinyReceiptTokenizer,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    question_token_ids = tokenizer.encode(
        PROMPT,
        add_eos=True,
        max_len=SEQUENCE_LENGTH,
    )
    question = np.zeros((1, SEQUENCE_LENGTH), dtype=np.int64)
    question[0, : len(question_token_ids)] = question_token_ids

    y = np.arange(IMAGE_HEIGHT, dtype=np.int64)[:, None]
    x = np.arange(IMAGE_WIDTH, dtype=np.int64)[None, :]
    quantized = (17 * x + 29 * y + 7 * np.bitwise_xor(x, y)) & 255
    image = ((quantized - 128) / 128).astype(np.float32)
    return image.reshape(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH), question_token_ids, question


def _session(path: Path) -> ort.InferenceSession:
    try:
        session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
    except Exception as error:
        raise OracleFailure(f"ONNX Runtime could not open the validated model: {error}") from error
    if session.get_providers()[0] != "CPUExecutionProvider":
        raise OracleFailure("ONNX Runtime did not select CPUExecutionProvider")
    return session


def _named_outputs(
    session: ort.InferenceSession,
    feeds: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    try:
        values = session.run(None, dict(feeds))
    except Exception as error:
        raise OracleFailure(f"ONNX Runtime execution failed: {error}") from error
    names = [output.name for output in session.get_outputs()]
    if len(names) != len(values) or len(names) != len(set(names)):
        raise OracleFailure("ONNX Runtime returned an invalid output signature")
    return dict(zip(names, values, strict=True))


def _numeric_summary(values: np.ndarray) -> dict[str, float | int]:
    flat = values.reshape(-1)
    if flat.size == 0 or not np.all(np.isfinite(flat)):
        raise OracleFailure("router logits must be non-empty and finite")
    minimum = float("inf")
    maximum = float("-inf")
    total = 0.0
    absolute_total = 0.0
    sum_squares = 0.0
    for raw in flat:
        value = float(raw)
        minimum = min(minimum, value)
        maximum = max(maximum, value)
        total += value
        absolute_total += abs(value)
        sum_squares += value * value
    return {
        "count": int(flat.size),
        "min": minimum,
        "max": maximum,
        "sum": total,
        "absSum": absolute_total,
        "sumSquares": sum_squares,
    }


def generate_reference(
    source_directory: Path,
    package_directory: Path,
    *,
    router_atol: float = 1e-4,
    router_rtol: float = 1e-5,
) -> dict[str, Any]:
    if not np.isfinite(router_atol) or router_atol < 0:
        raise OracleFailure("router atol must be finite and non-negative")
    if not np.isfinite(router_rtol) or router_rtol < 0:
        raise OracleFailure("router rtol must be finite and non-negative")
    try:
        source = validate_source(
            source_directory.resolve(strict=True),
            variant="int8-w8a8",
        )
    except (ImportFailure, OSError, ValueError) as error:
        raise OracleFailure(f"source validation failed: {error}") from error
    package_root = package_directory.resolve(strict=True)
    if not package_root.is_dir() or package_root.is_symlink():
        raise OracleFailure("package must be a real local directory")
    package, tokenizer = _package_identity(package_root, source)
    image, question_token_ids, question = _workload(tokenizer)
    family = np.asarray([-1], dtype=np.int64)

    encoder = _session(source["selected_paths"]["encoder"])
    encoded = _named_outputs(
        encoder,
        {
            "image": image,
            "question_ids": question,
            "family_ids": family,
        },
    )
    expected_encoder_names = {
        "memory",
        "memory_padding_mask",
        "router_logits",
        "selected_family_ids",
    }
    if set(encoded) != expected_encoder_names:
        raise OracleFailure("encoder output names differ from the split package contract")
    memory = encoded["memory"]
    memory_padding_mask = encoded["memory_padding_mask"]
    router_logits = encoded["router_logits"]
    selected_family_ids = encoded["selected_family_ids"]
    if (
        memory.shape != (1, 402, 320)
        or memory.dtype != np.float32
        or memory_padding_mask.shape != (1, 402)
        or memory_padding_mask.dtype != np.bool_
        or router_logits.shape != (1, 8)
        or router_logits.dtype != np.float32
        or selected_family_ids.shape != (1,)
        or selected_family_ids.dtype != np.int64
    ):
        raise OracleFailure("encoder output dtypes/shapes differ from the split contract")
    selected_family_id = int(selected_family_ids[0])
    if selected_family_id < 0 or selected_family_id >= len(FAMILY_ORDER):
        raise OracleFailure("encoder selected an invalid family")

    decoder = _session(source["selected_paths"]["decoder"])
    decoder_input_ids = np.zeros((1, SEQUENCE_LENGTH), dtype=np.int64)
    decoder_input_ids[0, 0] = 1
    token_ids: list[int] = []
    for step in range(MAX_NEW_TOKENS):
        decoded = _named_outputs(
            decoder,
            {
                "decoder_input_ids": decoder_input_ids,
                "memory": memory,
                "memory_padding_mask": memory_padding_mask,
                "family_ids": selected_family_ids,
            },
        )
        logits = decoded.get("logits")
        if (
            set(decoded) != {"logits"}
            or not isinstance(logits, np.ndarray)
            or logits.shape != (1, SEQUENCE_LENGTH, tokenizer.vocab_size)
            or logits.dtype != np.float32
            or not np.all(np.isfinite(logits[0, step]))
        ):
            raise OracleFailure("decoder logits differ from the split package contract")
        token = int(np.argmax(logits[0, step]))
        token_ids.append(token)
        if token == 2:
            break
        decoder_input_ids[0, step + 1] = token
    if len(token_ids) < MINIMUM_DECODER_STEPS:
        raise OracleFailure("oracle did not execute the required multiple decoder steps")

    text = tokenizer.decode(token_ids)
    image_bytes = image.astype("<f4", copy=False).tobytes(order="C")
    question_bytes = np.asarray(question_token_ids, dtype="<i4").tobytes(order="C")
    token_bytes = np.asarray(token_ids, dtype="<i4").tobytes(order="C")
    router = router_logits.reshape(-1).astype("<f4", copy=False)
    router_bytes = router.tobytes(order="C")
    return {
        "schema": REFERENCE_SCHEMA,
        "provenance": {
            "kind": "onnx-runtime-oracle",
            "sourceFormat": SOURCE_FORMAT,
            "sourceVariant": "int8-w8a8",
            "provider": "ONNX Runtime CPUExecutionProvider",
            "description": (
                "Independent fixed-workload encoder plus four-step decoder "
                "oracle from the producer INT8 ONNX pair."
            ),
        },
        "package": package,
        "workload": {
            "id": WORKLOAD_ID,
            "prompt": PROMPT,
            "family": "auto",
            "maxNewTokens": MAX_NEW_TOKENS,
            "minimumDecoderSteps": MINIMUM_DECODER_STEPS,
            "image": {
                "generator": IMAGE_GENERATOR,
                "dtype": "float32",
                "shape": [1, 1, IMAGE_HEIGHT, IMAGE_WIDTH],
                "byteOrder": "little",
                "sha256": _sha256_bytes(image_bytes),
            },
        },
        "expected": {
            "family": FAMILY_ORDER[selected_family_id],
            "familyId": selected_family_id,
            "requestedFamily": "auto",
            "requestedFamilyId": -1,
            "questionTokenIds": question_token_ids,
            "questionTokenIdsSha256": _sha256_bytes(question_bytes),
            "tokenIds": token_ids,
            "tokenIdsSha256": _sha256_bytes(token_bytes),
            "textUtf8Sha256": _sha256_bytes(text.encode("utf-8")),
            "stoppedAtEos": bool(token_ids and token_ids[-1] == 2),
            "routerLogits": {
                "values": [float(value) for value in router],
                "sha256": _sha256_bytes(router_bytes),
                "summary": _numeric_summary(router),
                "atol": router_atol,
                "rtol": router_rtol,
            },
        },
    }


def _real_output_parent(path: Path) -> Path:
    parent = Path(os.path.abspath(os.fspath(path.parent)))
    current = Path(parent.anchor)
    for part in parent.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir()
            except FileExistsError:
                pass
            metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise OracleFailure("output parent may not traverse a symbolic link")
        if not stat.S_ISDIR(metadata.st_mode):
            raise OracleFailure("output parent components must be real directories")
    return parent


def _atomic_write_new(path: Path, document: Mapping[str, Any]) -> None:
    parent = _real_output_parent(path)
    destination = parent / path.name
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(
                document,
                output,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise OracleFailure(
                "output already exists; use --check to verify it"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or check the path-free TinyReceipt INT8 ONNX E2E oracle.",
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--router-atol", default=1e-4, type=float)
    parser.add_argument("--router-rtol", default=1e-5, type=float)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the generated document with --out instead of writing it.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        generated = generate_reference(
            args.source,
            args.package,
            router_atol=args.router_atol,
            router_rtol=args.router_rtol,
        )
        if args.check:
            existing = _read_json(args.out, "reference")
            if existing != generated:
                raise OracleFailure("reference differs from the current source and package")
            print("TinyReceipt split INT8 E2E reference: verified")
        else:
            _atomic_write_new(args.out, generated)
            print("TinyReceipt split INT8 E2E reference: generated")
    except (OracleFailure, OSError, ValueError) as error:
        print(f"[TinyReceipt split ONNX E2E oracle] {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
