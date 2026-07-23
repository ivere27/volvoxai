#!/usr/bin/env python3
"""Focused integration checks for the opt-in native task CLI."""

from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path


def run(binary: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(binary), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def require(
    condition: bool,
    message: str,
    result: subprocess.CompletedProcess[bytes] | None = None,
) -> None:
    if condition:
        return
    if result is not None:
        message += (
            f"\nexit={result.returncode}"
            f"\nstdout={result.stdout.decode(errors='replace')}"
            f"\nstderr={result.stderr.decode(errors='replace')}"
        )
    raise AssertionError(message)


def write_rgb_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    require(len(pixels) == width * height * 3, "Invalid RGB fixture size")

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    rows = b"".join(
        b"\x00" + pixels[row * width * 3 : (row + 1) * width * 3]
        for row in range(height)
    )
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def write_image_identity_graph(
    path: Path,
    dtype: str,
    *,
    scale: float | None = None,
    zero_point: int = 0,
    image_normalization: str | None = None,
) -> Path | None:
    input_descriptor: dict[str, object] = {
        "shape": [1, 1, 2, 3],
        "dtype": dtype,
    }
    graph_quantization: dict[str, object] | None = None
    weights_path: Path | None = None
    if scale is not None:
        require(dtype in {"int8", "uint8"}, "Quantized image dtype must be I8/U8")
        scale_name = "__quant__.image.scale"
        zero_name = "__quant__.image.zero_point"
        descriptor = {
            "scheme": "per_tensor",
            "scale_tensor": scale_name,
            "zero_point_tensor": zero_name,
        }
        graph_quantization = {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {"image": descriptor, "pixels": descriptor},
        }
        scale_bytes = struct.pack("<f", scale)
        zero_bytes = struct.pack("b" if dtype == "int8" else "B", zero_point)
        tensors = {
            scale_name: {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, len(scale_bytes)],
            },
            zero_name: {
                "dtype": "I8" if dtype == "int8" else "U8",
                "shape": [1],
                "data_offsets": [len(scale_bytes), len(scale_bytes) + len(zero_bytes)],
            },
        }
        header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
        header += b" " * ((-len(header)) % 8)
        weights_path = path.with_suffix(".safetensors")
        weights_path.write_bytes(
            struct.pack("<Q", len(header)) + header + scale_bytes + zero_bytes
        )
    if image_normalization is not None:
        input_descriptor["image_normalization"] = image_normalization
    path.write_text(
        json.dumps(
            {
                "format": "volvox-graph/v1",
                **({"quantization": graph_quantization} if graph_quantization else {}),
                "inputs": {"image": input_descriptor},
                "nodes": [
                    {
                        "opType": "Identity",
                        "inputs": {"input": "image"},
                        "outputs": {"out": "pixels"},
                        "outputs_shape": {"out": [1, 1, 2, 3]},
                        "outputs_dtype": {"out": dtype},
                    }
                ],
                "outputs": ["pixels"],
            }
        ),
        encoding="utf-8",
    )
    return weights_path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <volvoxai-tasks-binary>", file=sys.stderr)
        return 2
    binary = Path(sys.argv[1]).resolve()
    require(binary.is_file(), f"Native task CLI not found: {binary}")

    help_result = run(binary, "--help")
    require(help_result.returncode == 0, "Task CLI help failed", help_result)
    help_text = help_result.stdout.decode(errors="replace")
    for command in ("run", "classify", "detect", "decode"):
        require(f"  {command} " in help_text, f"Missing task CLI command: {command}")
    for undeclared in ("generate", "ctc", "seq2seq", "chat", "train"):
        require(
            f"  {undeclared} " not in help_text,
            f"Undeclared task command is exposed: {undeclared}",
        )

    detect_help = run(binary, "detect", "--help")
    require(detect_help.returncode == 0, "Detect help failed", detect_help)
    require(
        b"--labels <file>" in detect_help.stdout,
        "Detect help does not describe label files",
        detect_help,
    )
    require(
        b"--include_transfers" in detect_help.stdout,
        "Detect help does not describe transfer-inclusive timing",
        detect_help,
    )

    with tempfile.TemporaryDirectory(prefix="volvoxai-task-cli-") as temporary:
        root = Path(temporary)
        graph = root / "graph.json"
        x_path = root / "x.f32"
        ids_path = root / "ids.i32"
        y_output = root / "y.f32"
        ids_output = root / "ids-out.i32"

        graph.write_text(
            json.dumps(
                {
                    "format": "volvox-graph/v1",
                    "inputs": {
                        "x": {"shape": [2], "dtype": "float32"},
                        "ids": {"shape": [2], "dtype": "int32"},
                    },
                    "nodes": [
                        {
                            "opType": "Sin",
                            "inputs": {"input": "x"},
                            "outputs": {"out": "y"},
                            "outputs_shape": {"out": [2]},
                        }
                    ],
                    "outputs": ["y", "ids"],
                }
            ),
            encoding="utf-8",
        )
        x_path.write_bytes(struct.pack("=2f", 0.0, 1.0))
        ids_path.write_bytes(struct.pack("=2i", 7, 11))

        result = run(
            binary,
            "run",
            str(root),
            "--input",
            f"x={x_path}",
            "--input",
            f"ids={ids_path}",
            "--output",
            f"y={y_output}",
            "--output",
            f"ids={ids_output}",
        )
        require(result.returncode == 0, "Weightless typed task-CLI run failed", result)
        values = struct.unpack("=2f", y_output.read_bytes())
        require(
            abs(values[0]) < 1.0e-6 and abs(values[1] - math.sin(1.0)) < 1.0e-6,
            f"Unexpected F32 output: {values}",
        )
        require(
            struct.unpack("=2i", ids_output.read_bytes()) == (7, 11),
            "I32 output was not written as exact raw data",
        )

        image_path = root / "pixels.png"
        image_pixels = bytes((0, 64, 127, 128, 200, 255))
        write_rgb_png(image_path, 2, 1, image_pixels)

        for dtype, suffix, scale, zero_point, expected in (
            ("uint8", ".u8", 0.0078125, 127, image_pixels),
            ("int8", ".i8", 0.03125, -3, bytes((128, 192, 255, 0, 72, 127))),
        ):
            image_graph = root / f"image-{dtype}-raw.graph.json"
            image_output = root / f"image-{dtype}-raw{suffix}"
            image_weights = write_image_identity_graph(
                image_graph,
                dtype,
                scale=scale,
                zero_point=zero_point,
                image_normalization="raw-255",
            )
            result = run(
                binary,
                "run",
                str(image_graph),
                "--weights",
                str(image_weights),
                "--image",
                f"image={image_path}",
                "--output",
                f"pixels={image_output}",
            )
            require(result.returncode == 0, f"{dtype} image binding failed", result)
            require(
                image_output.read_bytes() == expected,
                f"{dtype} raw image binding incorrectly applied quantization metadata",
                result,
            )

        normalized_cases = (
            ("uint8", ".u8", "zero-one", 1.0 / 255.0, 0, image_pixels),
            (
                "int8",
                ".i8",
                "minus-one-one",
                0.0078125,
                0,
                bytes((128, 192, 255, 1, 73, 127)),
            ),
        )
        for dtype, suffix, normalize, scale, zero_point, expected in normalized_cases:
            image_graph = root / f"image-{dtype}-{normalize}.graph.json"
            image_output = root / f"image-{dtype}-{normalize}{suffix}"
            image_weights = write_image_identity_graph(
                image_graph,
                dtype,
                scale=scale,
                zero_point=zero_point,
                image_normalization=normalize,
            )
            result = run(
                binary,
                "run",
                str(image_graph),
                "--weights",
                str(image_weights),
                "--image",
                f"image={image_path}",
                "--output",
                f"pixels={image_output}",
            )
            require(result.returncode == 0, f"{dtype} normalized image binding failed", result)
            require(
                image_output.read_bytes() == expected,
                f"{dtype} normalized image binding ignored its quantization metadata",
                result,
            )

        f32_image_graph = root / "image-f32.graph.json"
        f32_image_output = root / "image-f32.f32"
        write_image_identity_graph(
            f32_image_graph, "float32", image_normalization="zero-one"
        )
        result = run(
            binary,
            "run",
            str(f32_image_graph),
            "--image",
            f"image={image_path}",
            "--output",
            f"pixels={f32_image_output}",
        )
        require(result.returncode == 0, "F32 image binding failed", result)
        f32_pixels = struct.unpack("=6f", f32_image_output.read_bytes())
        require(
            all(abs(actual - expected / 255.0) < 1.0e-6
                for actual, expected in zip(f32_pixels, image_pixels)),
            f"F32 zero-one normalization is incorrect: {f32_pixels}",
            result,
        )

        f32_override_output = root / "image-f32-override.f32"
        result = run(
            binary,
            "run",
            str(f32_image_graph),
            "--image",
            f"image={image_path}",
            "--image-normalize",
            "raw-255",
            "--output",
            f"pixels={f32_override_output}",
        )
        require(result.returncode == 0, "Explicit image normalization override failed", result)
        f32_override_pixels = struct.unpack("=6f", f32_override_output.read_bytes())
        require(
            all(abs(actual - expected) < 1.0e-6
                for actual, expected in zip(f32_override_pixels, image_pixels)),
            f"Explicit image normalization did not override package metadata: {f32_override_pixels}",
            result,
        )

        f32_unannotated_graph = root / "image-f32-unannotated.graph.json"
        f32_unannotated_output = root / "image-f32-unannotated.f32"
        write_image_identity_graph(f32_unannotated_graph, "float32")
        result = run(
            binary,
            "run",
            str(f32_unannotated_graph),
            "--image",
            f"image={image_path}",
            "--output",
            f"pixels={f32_unannotated_output}",
        )
        require(
            result.returncode != 0
            and b"Input image does not declare image_normalization; add package metadata or pass --image-normalize explicitly."
            in result.stderr,
            "Unannotated image input did not fail closed",
            result,
        )

        result = run(
            binary,
            "run",
            str(f32_unannotated_graph),
            "--image",
            f"image={image_path}",
            "--image-normalize",
            "zero-one",
            "--output",
            f"pixels={f32_unannotated_output}",
        )
        require(
            result.returncode == 0,
            "Explicit normalization did not support an unannotated package",
            result,
        )

        f32_invalid_graph = root / "image-f32-invalid.graph.json"
        write_image_identity_graph(
            f32_invalid_graph, "float32", image_normalization="automatic"
        )
        result = run(
            binary,
            "run",
            str(f32_invalid_graph),
            "--image",
            f"image={image_path}",
        )
        require(
            result.returncode != 0
            and b"Input image has unsupported image_normalization 'automatic'" in result.stderr,
            "Invalid package image normalization was not rejected",
            result,
        )

        wrong_output = root / "ids-out.f32"
        result = run(
            binary,
            "run",
            str(graph),
            "--input",
            f"x={x_path}",
            "--input",
            f"ids={ids_path}",
            "--output",
            f"ids={wrong_output}",
        )
        require(
            result.returncode != 0 and b"requires a .i32 file" in result.stderr,
            "Output dtype/suffix mismatch was not rejected",
            result,
        )

        detector = root / "detector"
        detector.mkdir()
        detector_graph = detector / "graph.json"
        boxes_input = detector / "boxes.f32"
        scores_input = detector / "scores.f32"
        labels = detector / "labels.txt"
        override_labels = root / "override-labels.txt"

        detector_graph.write_text(
            json.dumps(
                {
                    "format": "volvox-graph/v1",
                    "inputs": {
                        "box_in": {"shape": [2, 4], "dtype": "float32"},
                        "score_in": {"shape": [2, 3], "dtype": "float32"},
                    },
                    "nodes": [
                        {
                            "opType": "Identity",
                            "inputs": {"input": "box_in"},
                            "outputs": {"out": "boxes"},
                            "outputs_shape": {"out": [2, 4]},
                        },
                        {
                            "opType": "Identity",
                            "inputs": {"input": "score_in"},
                            "outputs": {"out": "scores"},
                            "outputs_shape": {"out": [2, 3]},
                        },
                    ],
                    "outputs": ["boxes", "scores"],
                }
            ),
            encoding="utf-8",
        )
        boxes_input.write_bytes(
            struct.pack("=8f", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
        )
        scores_input.write_bytes(struct.pack("=6f", 0.1, 0.2, 0.9, 0.8, 0.1, 0.2))
        labels.write_text("person\ncat\ndog\n", encoding="utf-8")
        override_labels.write_text("person\ncat\nhound\n", encoding="utf-8")

        detect_args = (
            "detect",
            str(detector),
            "--input",
            f"box_in={boxes_input}",
            "--input",
            f"score_in={scores_input}",
            "--max-det",
            "2",
        )
        result = run(binary, *detect_args)
        require(result.returncode == 0, "Default detection outputs failed", result)
        output = result.stdout.decode(errors="replace")
        require(
            "rank\tindex\tscore\tscore_pct\tclass\tlabel\tx0\ty0\tx1\ty1"
            in output,
            "Labeled detection header is missing",
            result,
        )
        require(
            "1\t0\t0.9\t90.00%\t2\tdog\t0.1\t0.2\t0.3\t0.4" in output,
            "Auto-discovered label or first box coordinates are incorrect",
            result,
        )
        require(
            "2\t1\t0.8\t80.00%\t0\tperson\t0.5\t0.6\t0.7\t0.8" in output,
            "Second labeled detection is incorrect",
            result,
        )

        result = run(
            binary,
            *detect_args,
            "--warmup_runs",
            "1",
            "--num_runs",
            "2",
            "--include_transfers",
        )
        require(result.returncode == 0, "Transfer-inclusive detection failed", result)
        require(
            b"[bench] detect:" in result.stderr
            and b"warmup=1, runs=2, transfers=on" in result.stderr,
            "Transfer-inclusive detection did not report its timed scope",
            result,
        )
        require(
            b"1\t0\t0.9\t90.00%\t2\tdog\t0.1\t0.2\t0.3\t0.4" in result.stdout,
            "Transfer-inclusive timing changed detection output",
            result,
        )

        unlabeled_args = list(detect_args)
        unlabeled_args[1] = str(detector_graph)
        result = run(binary, *unlabeled_args)
        require(result.returncode == 0, "Unlabeled detection failed", result)
        require(
            b"rank\tindex\tscore\tscore_pct\tclass\tx0\ty0\tx1\ty1"
            in result.stdout
            and b"\tlabel\t" not in result.stdout,
            "Detection without a label file changed its numeric-only table",
            result,
        )

        result = run(binary, *detect_args, "--labels", str(override_labels))
        require(result.returncode == 0, "Explicit detection labels failed", result)
        require(
            b"\t2\thound\t" in result.stdout and b"\t2\tdog\t" not in result.stdout,
            "Explicit --labels did not override model-directory labels.txt",
            result,
        )

        result = run(binary, *detect_args, "--labels", str(root / "missing.txt"))
        require(
            result.returncode != 0 and b"Cannot read labels file" in result.stderr,
            "An explicit missing detection label file was not rejected",
            result,
        )

        custom_detector_graph = detector / "custom-outputs.graph.json"
        custom_detector_graph.write_text(
            detector_graph.read_text(encoding="utf-8")
            .replace('"boxes"', '"custom_boxes"')
            .replace('"scores"', '"custom_scores"'),
            encoding="utf-8",
        )
        custom_args = list(detect_args)
        custom_args[1] = str(custom_detector_graph)
        result = run(binary, *custom_args)
        require(
            result.returncode != 0
            and b"Unknown output tensor: boxes" in result.stderr,
            "Missing default detection outputs did not fail clearly",
            result,
        )

        result = run(
            binary,
            *custom_args,
            "--boxes",
            "custom_boxes",
            "--scores",
            "custom_scores",
        )
        require(result.returncode == 0, "Detection output overrides failed", result)
        require(
            b"1\t0\t0.9\t90.00%\t2\t0.1\t0.2\t0.3\t0.4" in result.stdout,
            "Detection output overrides selected the wrong tensors",
            result,
        )

    print("Native task CLI tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
