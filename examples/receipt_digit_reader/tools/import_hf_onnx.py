#!/usr/bin/env python3
"""Import the tiny-receipt-reader digit-slot release into a VolvoxAI package.

The importer is deliberately local and model-specific. It accepts only the
`receipt_digit_reader_onnx_v1` producer contract, verifies the release
checksums, derives the per-request ABI and optional bounded batch domain from
the producer manifest, and then delegates model-neutral ONNX conversion to the
repository's generic exporter. It never loads a PyTorch checkpoint and never
downloads model data.

Three variants share one source directory:

    fp32   producer model.onnx
    int8   producer model_int8.onnx, the upstream ORT static QDQ graph
    ptq    fp32 lowered to VolvoxAI's own byte domain from a calibration
           profile produced by tools/calibrate.mjs

`ptq` is not a re-encoding of `int8`. The producer quantizes convolution only,
so its graph keeps float GroupNorm and float SiLU that VolvoxAI would have
quantized natively. See docs/typed-ptq.md for why the two paths cannot
converge numerically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.exporter.capabilities import (  # noqa: E402
    classify_package,
    expand_targets,
    normalize_targets,
)

SOURCE_FORMAT = "receipt_digit_reader_onnx_v1"
PACKAGE_FORMAT = "volvoxai-receipt-digit-reader-onnx-package-v1"
INPUT_TENSOR = "input0"
OUTPUT_TENSOR = "slot_logits"
REQUIRED_OPSET = 18
# Convolution-only coverage, matching what the producer's own INT8 graph
# quantizes. Keep the readout and normalization in F32 by default. BENCHMARK.md
# records the latest accuracy audit; `--ptq-float-op` replaces this list of
# float operators, and changed coverage needs its own accuracy evaluation.
DEFAULT_PTQ_FLOAT_OPS = (
    "BatchMatMul", "Linear", "GroupNorm", "SiLU", "LayerNorm", "Add",
)


class ImportFailure(RuntimeError):
    """One explicit, actionable importer contract violation."""


def _fail(detail: str) -> "ImportFailure":
    return ImportFailure(detail)


def _read_json(path: Path, label: str) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise _fail(f"{label} is missing at {path}") from error
    except json.JSONDecodeError as error:
        raise _fail(f"{label} at {path} is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise _fail(f"{label} at {path} must be a JSON object")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release(source: Path, *, checked: Sequence[str]) -> dict[str, str]:
    """Verify the producer SHA256SUMS entries for every file we consume."""

    sums_path = source / "SHA256SUMS"
    if not sums_path.is_file():
        raise _fail(f"release checksum list is missing at {sums_path}")
    expected: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise _fail(f"malformed SHA256SUMS entry {line!r}")
        expected[parts[1].strip()] = parts[0].strip()
    verified: dict[str, str] = {}
    for name in checked:
        want = expected.get(name)
        if want is None:
            raise _fail(f"SHA256SUMS does not cover required artifact {name!r}")
        actual = _sha256(source / name)
        if actual != want:
            raise _fail(
                f"{name} SHA-256 {actual} does not match the release value {want}"
            )
        verified[name] = actual
    return verified


def _static_input_shape(config: Mapping[str, Any]) -> list[int]:
    spec = config.get("input")
    if not isinstance(spec, Mapping):
        raise _fail("config.json has no input specification")
    try:
        channels = int(spec["channels"])
        height = int(spec["height"])
        width = int(spec["width"])
    except (KeyError, TypeError, ValueError) as error:
        raise _fail("config.json input needs integer channels/height/width") from error
    if min(channels, height, width) <= 0:
        raise _fail("config.json input extents must be positive")
    return [1, channels, height, width]


def _validated_max_batch_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) \
            or value < 1:
        raise _fail("max batch size must be a positive integer")
    return value


def _producer_batch_symbol(model_path: Path, manifest: Mapping[str, Any]) -> str:
    """Return the producer-proved leading public batch symbol.

    Dynamic import is opt-in. It is safe only when the producer manifest and
    ONNX graph agree that the same named leading axis appears on the sole image
    input and every public output. A coincidental dynamic input axis is not
    enough to authorize output slicing by request.
    """

    import onnx

    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(outputs, list) \
            or not outputs:
        raise _fail(
            "dynamic batch import requires one manifest input and at least one output"
        )
    input_entry = inputs[0]
    if not isinstance(input_entry, Mapping) or not isinstance(input_entry.get("shape"), list) \
            or not input_entry["shape"] or not isinstance(input_entry["shape"][0], str) \
            or not input_entry["shape"][0]:
        raise _fail("producer manifest input has no named leading batch symbol")
    symbol = input_entry["shape"][0]
    input_name = input_entry.get("name")
    if not isinstance(input_name, str) or not input_name:
        raise _fail("producer manifest input needs a non-empty name")

    model = onnx.load(str(model_path), load_external_data=False)
    graph_inputs = {value.name: value for value in model.graph.input}
    graph_outputs = {value.name: value for value in model.graph.output}

    def leading_symbol(value: Any, label: str) -> str:
        axes = value.type.tensor_type.shape.dim
        if not axes or not axes[0].HasField("dim_param") or not axes[0].dim_param:
            raise _fail(f"{label} has no named leading batch symbol")
        return axes[0].dim_param

    graph_input = graph_inputs.get(input_name)
    if graph_input is None:
        raise _fail(f"producer ONNX graph has no manifest input {input_name!r}")
    if leading_symbol(graph_input, f"input {input_name!r}") != symbol:
        raise _fail("producer manifest and ONNX input batch symbols disagree")

    for entry in outputs:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("shape"), list) \
                or not entry["shape"] or entry["shape"][0] != symbol:
            raise _fail(
                "every producer manifest output must share the leading batch symbol"
            )
        name = entry.get("name")
        if not isinstance(name, str) or name not in graph_outputs:
            raise _fail(f"producer ONNX graph has no manifest output {name!r}")
        if leading_symbol(graph_outputs[name], f"output {name!r}") != symbol:
            raise _fail("producer manifest and ONNX output batch symbols disagree")
    return symbol


def _symbol_bindings(model_path: Path, manifest: Mapping[str, Any]) -> dict[str, int]:
    """Bind producer output symbols the manifest already proves constant.

    torch.onnx.export leaves a stale `dim_param` on the slot axis even though
    the manifest declares a fixed slot count. The exporter refuses to silently
    reinterpret a declared symbol, so bind it here from the producer's own
    manifest rather than weakening that contract.
    """

    import onnx

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise _fail("manifest.json declares no outputs")
    declared: dict[str, list[Any]] = {}
    for entry in outputs:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("shape"), list):
            raise _fail("manifest.json outputs need a name and shape")
        declared[str(entry.get("name"))] = list(entry["shape"])

    model = onnx.load(str(model_path), load_external_data=False)
    bindings: dict[str, int] = {}
    for value in model.graph.output:
        manifest_shape = declared.get(value.name)
        if manifest_shape is None:
            raise _fail(f"manifest.json does not declare graph output {value.name!r}")
        axes = value.type.tensor_type.shape.dim
        if len(axes) != len(manifest_shape):
            raise _fail(
                f"graph output {value.name!r} has rank {len(axes)}, but the "
                f"manifest declares rank {len(manifest_shape)}"
            )
        for axis, (dimension, expected) in enumerate(zip(axes, manifest_shape)):
            symbol = dimension.dim_param
            if not symbol or not isinstance(expected, int):
                continue
            if expected <= 0:
                raise _fail(
                    f"manifest extent for {value.name!r} axis {axis} must be positive"
                )
            previous = bindings.get(symbol)
            if previous is not None and previous != expected:
                raise _fail(
                    f"output symbol {symbol!r} is bound to both {previous} and {expected}"
                )
            bindings[symbol] = expected
    return bindings


def _run(command: Sequence[str], *, label: str) -> str:
    completed = subprocess.run(
        list(command),
        cwd=REPOSITORY_ROOT,
        env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise _fail(f"{label} failed: {detail}")
    return completed.stdout


def export_source_graph(
    model_path: Path,
    out_dir: Path,
    *,
    input_shape: Sequence[int] | None,
    dimension_bounds: Mapping[str, Mapping[str, int]],
    targets: Sequence[str],
    report_path: Path | None,
) -> None:
    """Convert one producer ONNX graph with the generic exporter."""

    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "tools/export_safetensors.py",
        "--model", str(model_path),
        "--out", str(out_dir / "model.safetensors"),
        "--output-name", OUTPUT_TENSOR,
    ]
    if input_shape is not None:
        command += [
            "--input-shape", "image=" + "x".join(str(value) for value in input_shape),
        ]
    for symbol, bounds in sorted(dimension_bounds.items()):
        minimum = int(bounds["min"])
        maximum = int(bounds["max"])
        encoded = f"{symbol}={minimum}:{maximum}"
        if "multiple_of" in bounds:
            encoded += f":{int(bounds['multiple_of'])}"
        command += [
            "--dimension-bound", encoded,
        ]
    for target in targets:
        command += ["--target", target]
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        command += ["--report", str(report_path), "--report-format", "json"]
    _run(command, label="generic ONNX exporter")


def optimize_package(
    package: Path,
    out_dir: Path,
    *,
    prepare_for_ptq: bool = False,
    report_path: Path | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "tools.exporter.optimizer",
        str(package / "graph.json"),
        "--weights", str(package / "model.safetensors"),
        "--out", str(out_dir / "graph.json"),
        "--out-weights", str(out_dir / "model.safetensors"),
    ]
    if prepare_for_ptq:
        command.append("--prepare-fp32-for-ptq")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        command += ["--report", str(report_path)]
    _run(command, label="typed graph optimizer")


def author_ptq_package(
    prepared: Path,
    out_dir: Path,
    *,
    calibration_path: Path,
    activation_dtype: str,
    activation_scheme: str,
    float_ops: Sequence[str],
) -> dict[str, Any]:
    """Run the registry PTQ recipe against an exact calibration profile."""

    from safetensors.numpy import load_file, save_file

    from tools.exporter.optimizer.target import TargetEnvironment
    from tools.exporter.optimizer.typed_pipeline import author_runtime_ptq_package
    from tools.exporter.runtime_ir import import_runtime_package
    from tools.exporter.typed_ptq import (
        PTQConfig,
        calibration_profile_from_ranges,
        required_ptq_observations,
    )

    calibration = _read_json(calibration_path, "calibration profile")
    ranges = calibration.get("ranges")
    sample_count = calibration.get("samples")
    sample_digest = calibration.get("digest")
    if not isinstance(ranges, Mapping) or not isinstance(sample_count, int) \
            or not isinstance(sample_digest, str):
        raise _fail(
            "calibration profile needs {ranges, samples, digest}; regenerate it "
            "with tools/calibrate.mjs"
        )

    document = _read_json(prepared / "graph.json", "prepared graph")
    tensors = dict(load_file(str(prepared / "model.safetensors")))
    graph = import_runtime_package(document, dict(tensors))

    # An operand that is an immutable weight is not an activation, so its node
    # has no calibrated affine and must stay in F32 whatever the op policy is.
    weight_operand_nodes = sorted({
        node["id"]
        for node in document["nodes"]
        if node["opType"] == "BatchMatMul"
        and any(name in tensors for name in node["inputs"].values())
    })
    config = PTQConfig(
        activation_dtype=activation_dtype,
        activation_scheme=activation_scheme,
        float_ops=frozenset(float_ops),
        float_nodes=frozenset(weight_operand_nodes),
    )
    demanded = required_ptq_observations(graph, None, config=config)
    missing = sorted(name for name in demanded if name not in ranges)
    if missing:
        raise _fail(
            f"calibration profile is missing {len(missing)} demanded tensor(s): "
            f"{', '.join(missing[:6])}"
        )
    profile = calibration_profile_from_ranges(
        graph, ranges,
        sample_count=sample_count,
        sample_digest=sample_digest,
        config=config,
    )
    authored_document, authored_tensors, report = author_runtime_ptq_package(
        document, tensors, profile,
        shape_profile={},
        config=config,
        target_environment=TargetEnvironment(
            backend_profile="portable",
            compile_backend="wasm",
            tune_backend="native-cpu",
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph.json").write_text(
        json.dumps(authored_document, indent=1), encoding="utf-8",
    )
    save_file(
        {name: np.ascontiguousarray(value) for name, value in authored_tensors.items()},
        str(out_dir / "model.safetensors"),
    )
    counts: dict[str, int] = {}
    for node in authored_document["nodes"]:
        counts[node["opType"]] = counts.get(node["opType"], 0) + 1
    return {
        "retained_float_nodes": weight_operand_nodes,
        "float_ops": sorted(float_ops),
        "activation_dtype": activation_dtype,
        "activation_scheme": activation_scheme,
        "calibration_samples": sample_count,
        "calibration_digest": sample_digest,
        "operators": dict(sorted(counts.items())),
        "passes": len(getattr(report, "entries", ()) or ()),
    }


def _validate_published_graph_abi(
    graph: Mapping[str, Any],
    *,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    batch_symbol: str | None,
    max_batch_size: int,
) -> None:
    """Require the final exporter result to preserve the digit-reader ABI.

    Producer inspection authorizes a contract; it does not prove that a later
    exporter/optimizer result still has it. Validate the final graph immediately
    before manifest publication so a malformed result is never labelled as a
    usable receipt package.
    """

    inputs = graph.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {INPUT_TENSOR}:
        raise _fail(
            f"published graph inputs must be exactly [{INPUT_TENSOR!r}]"
        )
    input_descriptor = inputs[INPUT_TENSOR]
    if not isinstance(input_descriptor, Mapping):
        raise _fail(f"published graph input {INPUT_TENSOR!r} needs a descriptor")

    outputs = graph.get("outputs")
    if outputs != [OUTPUT_TENSOR]:
        raise _fail(
            f"published graph outputs must be exactly [{OUTPUT_TENSOR!r}]"
        )

    output_descriptors: list[Mapping[str, Any]] = []
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise _fail("published graph nodes must be an array")
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        ports = node.get("outputs")
        if not isinstance(ports, Mapping):
            continue
        for descriptor in ports.values():
            if isinstance(descriptor, Mapping) \
                    and descriptor.get("tensor") == OUTPUT_TENSOR:
                output_descriptors.append(descriptor)
    if len(output_descriptors) != 1:
        raise _fail(
            f"published graph output {OUTPUT_TENSOR!r} must have exactly one producer"
        )
    output_descriptor = output_descriptors[0]

    if input_descriptor.get("dtype") != "float32" \
            or output_descriptor.get("dtype") != "float32":
        raise _fail("published graph public input and output dtypes must be float32")
    actual_input_shape = input_descriptor.get("shape")
    actual_output_shape = output_descriptor.get("shape")
    if not isinstance(actual_input_shape, list) \
            or len(actual_input_shape) != len(input_shape) \
            or actual_input_shape[1:] != list(input_shape[1:]):
        raise _fail(
            f"published graph input tail must be exactly {list(input_shape[1:])}"
        )
    if not isinstance(actual_output_shape, list) \
            or len(actual_output_shape) != len(output_shape) \
            or actual_output_shape[1:] != list(output_shape[1:]):
        raise _fail(
            f"published graph output tail must be exactly {list(output_shape[1:])}"
        )

    dimensions = graph.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise _fail("published graph dimensions must be an object")
    if batch_symbol is None:
        if max_batch_size != 1:
            raise _fail("a static published graph requires max batch size 1")
        if actual_input_shape[0] != 1 or actual_output_shape[0] != 1:
            raise _fail(
                "a static published graph requires literal leading batch extent 1"
            )
        return

    if max_batch_size <= 1:
        raise _fail("a dynamic published graph requires max batch size greater than 1")
    if actual_input_shape[0] != batch_symbol \
            or actual_output_shape[0] != batch_symbol:
        raise _fail(
            f"published graph public leading axes must be batch symbol {batch_symbol!r}"
        )
    domain = dimensions.get(batch_symbol)
    expected_domain = {
        "min": 1,
        "max": max_batch_size,
        "multiple_of": 1,
    }
    if not isinstance(domain, Mapping) or dict(domain) != expected_domain:
        raise _fail(
            f"published graph dimension {batch_symbol!r} must be exactly "
            f"{expected_domain} using 'multiple_of'"
        )


def write_manifest(
    out_dir: Path,
    *,
    variant: str,
    source: Path,
    checksums: Mapping[str, str],
    config: Mapping[str, Any],
    input_shape: Sequence[int],
    dimension_bounds: Mapping[str, int],
    batch_symbol: str | None,
    max_batch_size: int,
    targets: Sequence[str],
    resolved_targets: Sequence[str],
    ptq: Mapping[str, Any] | None,
) -> dict:
    graph = _read_json(out_dir / "graph.json", "published graph")
    _validate_published_graph_abi(
        graph,
        input_shape=input_shape,
        output_shape=(1, int(config["slots"]), int(config["num_classes"])),
        batch_symbol=batch_symbol,
        max_batch_size=max_batch_size,
    )
    operators: dict[str, int] = {}
    for node in graph.get("nodes", ()):
        operators[node["opType"]] = operators.get(node["opType"], 0) + 1
    manifest = {
        "format": PACKAGE_FORMAT,
        "variant": variant,
        "source_format": SOURCE_FORMAT,
        "package_class": classify_package(graph),
        "source": {"directory": source.name, "sha256": dict(sorted(checksums.items()))},
        "abi": {
            "input": {"name": INPUT_TENSOR, "shape": list(input_shape)},
            "output": OUTPUT_TENSOR,
            "dimension_bounds": dict(sorted(dimension_bounds.items())),
            "batch": {
                "per_request": 1,
                "symbol": batch_symbol,
                "min": 1,
                "max": max_batch_size,
                "multiple_of": 1,
            },
        },
        "decode": {
            "slots": int(config["slots"]),
            "phone_slots": int(config["phone_slots"]),
            "street_slots": int(config["street_slots"]),
            "blank_class": int(config["blank_class"]),
            "num_classes": int(config["num_classes"]),
        },
        "preprocess": dict(config["input"]),
        "targets": {
            "requested": list(targets),
            "resolved": list(resolved_targets),
        },
        "operators": dict(sorted(operators.items())),
    }
    if ptq is not None:
        manifest["ptq"] = dict(ptq)
    destination = out_dir / "manifest.json"
    temporary = out_dir / f".manifest.json.tmp-{os.getpid()}"
    try:
        temporary.write_text(
            json.dumps(manifest, indent=1, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return manifest


def import_release(
    source: Path,
    out_dir: Path,
    *,
    variant: str,
    targets: Sequence[str],
    calibration_path: Path | None,
    activation_dtype: str,
    activation_scheme: str,
    float_ops: Sequence[str],
    max_batch_size: int = 1,
    workspace: Path | None = None,
) -> dict:
    if variant not in {"fp32", "int8", "ptq"}:
        raise _fail(f"unknown variant {variant!r}; expected fp32, int8, or ptq")
    max_batch_size = _validated_max_batch_size(max_batch_size)
    # Argument contracts are settled before anything reads a multi-megabyte
    # graph, so a missing profile reports in milliseconds rather than after a
    # full parse and export.
    if variant == "ptq" and calibration_path is None:
        raise _fail(
            "the ptq variant requires --calibration; produce it with "
            "node examples/receipt_digit_reader/tools/calibrate.mjs"
        )
    config = _read_json(source / "config.json", "config.json")
    manifest = _read_json(source / "manifest.json", "manifest.json")
    if config.get("format") != SOURCE_FORMAT or manifest.get("format") != SOURCE_FORMAT:
        raise _fail(
            f"source is not {SOURCE_FORMAT}; found config "
            f"{config.get('format')!r} and manifest {manifest.get('format')!r}"
        )
    if int(manifest.get("opset", 0)) != REQUIRED_OPSET:
        raise _fail(
            f"source declares opset {manifest.get('opset')!r}, expected {REQUIRED_OPSET}"
        )

    model_name = "model_int8.onnx" if variant == "int8" else "model.onnx"
    checksums = verify_release(
        source, checked=("config.json", "manifest.json", model_name),
    )
    input_shape = _static_input_shape(config)
    bindings = _symbol_bindings(source / model_name, manifest)
    batch_symbol = None
    if max_batch_size > 1:
        batch_symbol = _producer_batch_symbol(source / model_name, manifest)
    export_bounds = {
        symbol: {"min": extent, "max": extent}
        for symbol, extent in bindings.items()
    }
    if batch_symbol is not None:
        export_bounds[batch_symbol] = {
            "min": 1, "max": max_batch_size, "multiple_of": 1,
        }
    requested = normalize_targets(list(targets) or None)
    resolved = expand_targets(requested)

    out_dir.mkdir(parents=True, exist_ok=True)
    staging = workspace or (out_dir.parent / f".{out_dir.name}.import")
    ptq_report: dict[str, Any] | None = None

    if variant in {"fp32", "int8"}:
        export_source_graph(
            source / model_name, out_dir,
            input_shape=input_shape if batch_symbol is None else None,
            dimension_bounds=export_bounds,
            targets=requested,
            report_path=out_dir / "export-report.json",
        )
    else:
        exported = staging / "fp32"
        prepared = staging / "prepared"
        export_source_graph(
            source / model_name, exported,
            input_shape=input_shape if batch_symbol is None else None,
            dimension_bounds=export_bounds,
            targets=requested,
            report_path=None,
        )
        optimize_package(exported, prepared, prepare_for_ptq=True)
        authored = staging / "authored"
        ptq_report = author_ptq_package(
            prepared, authored,
            calibration_path=calibration_path,
            activation_dtype=activation_dtype,
            activation_scheme=activation_scheme,
            float_ops=float_ops,
        )
        optimize_package(
            authored, out_dir, report_path=out_dir / "optimizer-report.json",
        )

    return write_manifest(
        out_dir,
        variant=variant,
        source=source,
        checksums=checksums,
        config=config,
        input_shape=input_shape,
        dimension_bounds=bindings,
        batch_symbol=batch_symbol,
        max_batch_size=max_batch_size,
        targets=requested,
        resolved_targets=resolved,
        ptq=ptq_report,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, type=Path,
                        help="tiny-receipt-reader release directory")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="destination package directory")
    parser.add_argument("--variant", default="fp32", choices=("fp32", "int8", "ptq"))
    parser.add_argument("--target", action="append", dest="targets", default=[],
                        help="required target profile; repeat for the intersection")
    parser.add_argument(
        "--max-batch-size", type=int, default=1,
        help=("retain the producer batch symbol with domain 1..N; "
              "backend compilation proves the requested bound (default: 1)"),
    )
    parser.add_argument("--calibration", type=Path,
                        help="calibration profile JSON for --variant ptq")
    parser.add_argument("--activation-dtype", default="int8", choices=("int8", "uint8"))
    parser.add_argument("--activation-scheme", default="asymmetric",
                        choices=("symmetric", "asymmetric"))
    parser.add_argument("--ptq-float-op", action="append", dest="float_ops",
                        default=None,
                        help="runtime opType retained in F32; repeat. "
                             f"Default: {', '.join(DEFAULT_PTQ_FLOAT_OPS)}")
    parser.add_argument("--ptq-quantize-attention-scores", action="store_true",
                        help="also quantize the readout score BatchMatMul. This "
                             "collapses phone accuracy on the released weights.")
    args = parser.parse_args(argv)

    float_ops = args.float_ops
    if float_ops is None:
        float_ops = [] if args.ptq_quantize_attention_scores else list(DEFAULT_PTQ_FLOAT_OPS)
    elif args.ptq_quantize_attention_scores:
        raise SystemExit(
            "--ptq-quantize-attention-scores conflicts with an explicit --ptq-float-op"
        )

    try:
        manifest = import_release(
            args.source.resolve(),
            args.out_dir.resolve(),
            variant=args.variant,
            targets=args.targets,
            calibration_path=args.calibration.resolve() if args.calibration else None,
            activation_dtype=args.activation_dtype,
            activation_scheme=args.activation_scheme,
            float_ops=float_ops,
            max_batch_size=args.max_batch_size,
        )
    except ImportFailure as error:
        print(f"[receipt_digit_reader] {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "variant": manifest["variant"],
        "package_class": manifest["package_class"],
        "operators": manifest["operators"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
