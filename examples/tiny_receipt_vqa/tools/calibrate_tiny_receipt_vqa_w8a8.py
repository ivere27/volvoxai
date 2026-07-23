#!/usr/bin/env python3
"""Create a development-only W8A8 calibration/reference-golden record.

This tool deliberately has a narrow job.  It loads the published
TinyReceiptVQA *INT8 SafeTensors* file by its named entries, dequantizes those
weights only while constructing an in-memory PyTorch reference model, and
observes a deterministic subset of the local heldout annotations/images.

The output is useful evidence for an eventually authored Volvox graph:

* it binds the exact release, reference model definition, annotations, and
  source images with SHA-256;
* it records the evaluator preprocessing contract (grayscale, BILINEAR,
  ``(pixel / 255 - .5) / .5``);
* it proposes one conservative, symmetric I8 *global* activation scale from
  observed reference ranges; and
* it records first-decode-step router and next-token reference goldens.

``--include-explicit-families`` additionally runs each selected heldout
question as a batch replicated across every adapter family ``[0..7]``.  That
does not change the auto-router golden path; it supplies distinct first-token
coverage evidence and makes reference hooks observe all task-adapter modules.

It is not a model converter or the source-bound per-edge profile consumed by
the strict W8A8 materializer. A Volvox graph still needs explicit
activation-edge descriptors, graph lowering, and backend execution validation.
Keeping this boundary explicit prevents a reference observation file from
being mistaken for an executable W8A8 model.

The reference model is loaded from ``modeling_tiny_receipt_vqa.py`` packaged
beside the release.  Treat that code as part of the trusted development
release, just as the release's own ``inference.py`` does.  CPU execution with
one PyTorch thread is the default so the JSON is reproducible on a given
PyTorch/Pillow release.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence


FORMAT = "volvoxai-tiny-receipt-vqa-w8a8-development-calibration-v1"
PROFILE_ID = "tiny-receipt-vqa-w8a8-global-i8-development-v1"
SOURCE_FORMAT = "tiny_receipt_vqa_int8_safetensors_v1"
SOURCE_RUNTIME = "pytorch-reference-dequantized-safetensors-v1"
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
IMAGE_WIDTH = 672
IMAGE_HEIGHT = 320


class CalibrationError(ValueError):
    """The supplied release or heldout inputs cannot produce a safe record."""


@dataclass(frozen=True)
class HeldoutRecord:
    """One evaluator-compatible supported heldout question."""

    id: str
    annotation_path: Path
    image_path: Path
    question: str
    answer: str
    family: str
    family_id: int


@dataclass
class RangeObservation:
    """Aggregate finite range evidence for one reference activation probe."""

    minimum: float = math.inf
    maximum: float = -math.inf
    maximum_absolute: float = 0.0
    calls: int = 0
    elements: int = 0

    def update(self, *, minimum: float, maximum: float, elements: int) -> None:
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise CalibrationError("reference activation probe produced a non-finite value")
        if elements < 1:
            return
        self.minimum = min(self.minimum, minimum)
        self.maximum = max(self.maximum, maximum)
        self.maximum_absolute = max(self.maximum_absolute, abs(minimum), abs(maximum))
        self.calls += 1
        self.elements += elements

    def as_json(self) -> dict[str, Any]:
        if self.calls < 1:
            raise CalibrationError("cannot serialize an activation probe without observations")
        return {
            "calls": self.calls,
            "elements": self.elements,
            "observed_max": self.maximum,
            "observed_min": self.minimum,
            "observed_abs_max": self.maximum_absolute,
        }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular source file."""

    path = Path(path)
    if not path.is_file():
        raise CalibrationError(f"required file is missing or not regular: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON without host-specific whitespace or non-finite numbers."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"could not canonicalize JSON value: {error}") from error


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_deterministic_json(path: Path, value: Any) -> None:
    """Write a stable human-readable JSON document atomically."""

    path = Path(path)
    if path.exists() and path.is_dir():
        raise CalibrationError(f"output path is a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"could not encode calibration JSON: {error}") from error
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def clean_text(value: object) -> str:
    return " ".join(str(value if value is not None else "").replace("\n", " ").split())


def heldout_family(question: object) -> tuple[str, int] | None:
    """Match the published heldout evaluator's supported-question routing."""

    normalized = clean_text(question)
    lower = normalized.lower()
    if "phone number" in lower or "전화번호" in normalized:
        return "phone", FAMILY_ORDER.index("phone")
    if (
        "location" in lower
        or "address" in lower
        or "가게 위치" in normalized
        or "빈 칸" in normalized
        or "blank" in lower
    ):
        return "address", FAMILY_ORDER.index("address")
    return None


def discover_heldout_records(annotations_dir: Path, images_dir: Path) -> list[HeldoutRecord]:
    """Read exactly the supported records accepted by the local evaluator."""

    annotations_dir = Path(annotations_dir)
    images_dir = Path(images_dir)
    if not annotations_dir.is_dir():
        raise CalibrationError(f"annotation directory does not exist: {annotations_dir}")
    if not images_dir.is_dir():
        raise CalibrationError(f"image directory does not exist: {images_dir}")

    records: list[HeldoutRecord] = []
    for annotation_path in sorted(annotations_dir.glob("*.json")):
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CalibrationError(f"could not read heldout annotation {annotation_path}: {error}") from error
        if not isinstance(annotation, dict):
            raise CalibrationError(f"heldout annotation is not an object: {annotation_path}")
        question = clean_text(annotation.get("question"))
        answer = clean_text(annotation.get("answer"))
        family = heldout_family(question)
        image_path = images_dir / f"{annotation_path.stem}.jpg"
        # These are exactly the evaluator's eligibility checks: an image,
        # question, answer, and a supported phone/address question.
        if not image_path.is_file() or not question or not answer or family is None:
            continue
        family_name, family_id = family
        raw_id = clean_text(annotation.get("id")) or annotation_path.stem
        records.append(
            HeldoutRecord(
                id=raw_id,
                annotation_path=annotation_path,
                image_path=image_path,
                question=question,
                answer=answer,
                family=family_name,
                family_id=family_id,
            )
        )
    if not records:
        raise CalibrationError("no evaluator-compatible heldout records were found")
    return records


def select_stratified_records(records: Sequence[HeldoutRecord], sample_count: int) -> list[HeldoutRecord]:
    """Choose a deterministic family round-robin subset without a RNG."""

    if sample_count < 1:
        raise CalibrationError("sample_count must be positive")
    buckets: dict[int, list[HeldoutRecord]] = defaultdict(list)
    for record in records:
        buckets[record.family_id].append(record)
    for bucket in buckets.values():
        bucket.sort(key=lambda record: (record.id, record.annotation_path.name))

    result: list[HeldoutRecord] = []
    offsets: dict[int, int] = defaultdict(int)
    ordered_ids = [family_id for family_id in range(len(FAMILY_ORDER)) if buckets.get(family_id)]
    while len(result) < sample_count:
        made_progress = False
        for family_id in ordered_ids:
            offset = offsets[family_id]
            bucket = buckets[family_id]
            if offset >= len(bucket):
                continue
            result.append(bucket[offset])
            offsets[family_id] += 1
            made_progress = True
            if len(result) == sample_count:
                break
        if not made_progress:
            break
    if not result:
        raise CalibrationError("deterministic sample selection selected no records")
    return result


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationError(f"{label} must be a JSON object")
    return value


def load_release_manifest(release_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    """Validate the minimal named-SafeTensors release boundary used here."""

    release_dir = Path(release_dir)
    manifest_path = release_dir / "int8" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(f"could not read release manifest {manifest_path}: {error}") from error
    manifest = _require_object(manifest, "release manifest")
    if manifest.get("format") != SOURCE_FORMAT:
        raise CalibrationError(
            f"unsupported release manifest format {manifest.get('format')!r}; expected {SOURCE_FORMAT!r}"
        )
    if manifest.get("runtime") != SOURCE_RUNTIME:
        raise CalibrationError(
            f"unsupported release runtime {manifest.get('runtime')!r}; expected {SOURCE_RUNTIME!r}"
        )
    files = _require_object(manifest.get("files"), "release manifest.files")
    weight_filename = files.get("model_safetensors")
    if not isinstance(weight_filename, str) or not weight_filename:
        raise CalibrationError("release manifest.files.model_safetensors must be a non-empty filename")
    if "/" in weight_filename or "\\" in weight_filename or Path(weight_filename).suffix != ".safetensors":
        raise CalibrationError("release manifest.files.model_safetensors must be a local .safetensors filename")
    tensors = _require_object(manifest.get("tensors"), "release manifest.tensors")
    if not tensors:
        raise CalibrationError("release manifest.tensors must not be empty")
    config = _require_object(manifest.get("config"), "release manifest.config")
    vocab = _require_object(manifest.get("vocab"), "release manifest.vocab")
    if not isinstance(vocab.get("itos"), list) or not vocab["itos"]:
        raise CalibrationError("release manifest.vocab.itos must be a non-empty list")
    weights_path = manifest_path.parent / weight_filename
    if not weights_path.is_file():
        raise CalibrationError(f"release SafeTensors file is missing: {weights_path}")
    model_definition_path = release_dir / "modeling_tiny_receipt_vqa.py"
    if not model_definition_path.is_file():
        raise CalibrationError(
            f"release model definition is missing: {model_definition_path}; expected packaged reference source"
        )
    return manifest, manifest_path, model_definition_path


def load_reference_module(path: Path) -> ModuleType:
    """Import the packaged PyTorch reference under an isolated module name."""

    path = Path(path)
    module_name = "_volvoxai_tiny_receipt_reference_" + sha256_file(path)[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise CalibrationError(f"could not import packaged reference definition: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules during
    # execution, so register before exec_module just as normal import does.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # pragma: no cover - error depends on external release/deps.
        sys.modules.pop(module_name, None)
        raise CalibrationError(f"could not execute packaged reference definition {path}: {error}") from error
    return module


def _shape_from_manifest(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise CalibrationError(f"{label}.shape must be a non-empty array")
    shape: list[int] = []
    for index, dimension in enumerate(value):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise CalibrationError(f"{label}.shape[{index}] must be a positive integer")
        shape.append(dimension)
    return tuple(shape)


def build_dequantized_reference_state(
    manifest: Mapping[str, Any],
    named_tensors: Mapping[str, Any],
    torch: Any,
) -> dict[str, Any]:
    """Dequantize named I8 weights only for the in-memory PyTorch reference.

    This function intentionally never writes a dequantized file.  It also
    rejects a missing/extra named payload rather than relying on manifest byte
    offsets, because the SafeTensors names and shapes are the release contract.
    """

    tensors = _require_object(manifest.get("tensors"), "release manifest.tensors")
    expected_names: set[str] = set()
    state: dict[str, Any] = {}
    for name, raw_info in tensors.items():
        if not isinstance(name, str) or not name:
            raise CalibrationError("release manifest tensor names must be non-empty strings")
        info = _require_object(raw_info, f"release manifest.tensors[{name!r}]")
        dtype = info.get("dtype")
        shape = _shape_from_manifest(info.get("shape"), f"release manifest.tensors[{name!r}]")
        if dtype == "int8_per_out":
            quantized_name = f"quantized.{name}"
            scale_name = f"scale.{name}"
            expected_names.update((quantized_name, scale_name))
            try:
                quantized = named_tensors[quantized_name]
                scale = named_tensors[scale_name]
            except KeyError as error:
                raise CalibrationError(f"SafeTensors payload is missing {error.args[0]!r}") from error
            if tuple(quantized.shape) != shape:
                raise CalibrationError(
                    f"SafeTensors {quantized_name!r} has shape {tuple(quantized.shape)!r}, expected {shape!r}"
                )
            if getattr(quantized, "dtype", None) != torch.int8:
                raise CalibrationError(f"SafeTensors {quantized_name!r} must have dtype torch.int8")
            if getattr(scale, "dtype", None) != torch.float32:
                raise CalibrationError(f"SafeTensors {scale_name!r} must have dtype torch.float32")
            if tuple(scale.shape) != (shape[0],):
                raise CalibrationError(
                    f"SafeTensors {scale_name!r} has shape {tuple(scale.shape)!r}, expected {(shape[0],)!r}"
                )
            if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
                raise CalibrationError(f"SafeTensors {scale_name!r} must contain finite positive scales")
            view_shape = (shape[0],) + (1,) * (len(shape) - 1)
            # This is the only deliberate dequantization in this program.  It
            # is immediately consumed by load_state_dict below and is never
            # emitted as an artifact.
            state[name] = (quantized.to(dtype=torch.float32) * scale.reshape(view_shape)).contiguous()
        elif dtype == "float32":
            source_name = f"float32.{name}"
            expected_names.add(source_name)
            try:
                value = named_tensors[source_name]
            except KeyError as error:
                raise CalibrationError(f"SafeTensors payload is missing {source_name!r}") from error
            if tuple(value.shape) != shape:
                raise CalibrationError(
                    f"SafeTensors {source_name!r} has shape {tuple(value.shape)!r}, expected {shape!r}"
                )
            if getattr(value, "dtype", None) != torch.float32:
                raise CalibrationError(f"SafeTensors {source_name!r} must have dtype torch.float32")
            if not bool(torch.isfinite(value).all()):
                raise CalibrationError(f"SafeTensors {source_name!r} must contain finite values")
            state[name] = value.contiguous()
        else:
            raise CalibrationError(f"unsupported release tensor dtype {dtype!r} for {name!r}")
    actual_names = set(named_tensors)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        pieces: list[str] = []
        if missing:
            pieces.append(f"missing={missing!r}")
        if unexpected:
            pieces.append(f"unexpected={unexpected!r}")
        raise CalibrationError("SafeTensors named payload does not exactly match manifest: " + "; ".join(pieces))
    return state


def load_reference_model(release_dir: Path, torch: Any) -> tuple[Any, Any, dict[str, Any], dict[str, Path]]:
    """Construct a CPU PyTorch model from only release INT8 SafeTensors names."""

    manifest, manifest_path, definition_path = load_release_manifest(release_dir)
    weights_path = manifest_path.parent / str(manifest["files"]["model_safetensors"])
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - install-time dependency failure.
        raise CalibrationError("safetensors.torch is required for reference calibration") from error
    module = load_reference_module(definition_path)
    for required in ("CharVocab", "ModelConfig", "TinyReceiptVQA"):
        if not hasattr(module, required):
            raise CalibrationError(f"packaged reference definition is missing {required}")
    named_tensors = load_file(str(weights_path), device="cpu")
    state = build_dequantized_reference_state(manifest, named_tensors, torch)
    try:
        config = module.ModelConfig(**manifest["config"])
        vocab = module.CharVocab.from_json(manifest["vocab"])
        model = module.TinyReceiptVQA(config)
        model.load_state_dict(state, strict=True)
        model.to("cpu")
        model.eval()
    except Exception as error:  # pragma: no cover - error depends on external release/model.
        raise CalibrationError(f"could not initialize the packaged PyTorch reference model: {error}") from error
    return model, vocab, manifest, {
        "manifest": manifest_path,
        "weights": weights_path,
        "model_definition": definition_path,
        "config": Path(release_dir) / "model" / "config.json",
        "vocab": Path(release_dir) / "model" / "vocab.json",
    }


def preprocess_evaluator_image(path: Path, torch: Any) -> Any:
    """Use the heldout evaluator's grayscale+BILINEAR ``[-1, 1]`` transform."""

    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency failure is environment-specific.
        raise CalibrationError("numpy and Pillow are required for evaluator-compatible preprocessing") from error
    path = Path(path)
    if not path.is_file():
        raise CalibrationError(f"heldout image is missing: {path}")
    try:
        with Image.open(path) as source:
            image = source.convert("L").resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    except OSError as error:
        raise CalibrationError(f"could not preprocess heldout image {path}: {error}") from error
    array = (array - np.float32(0.5)) / np.float32(0.5)
    # [B, C, H, W], matching ReceiptVQADataset._image plus collation.
    return torch.from_numpy(np.ascontiguousarray(array))[None, None, :, :]


def _tensor_bytes_sha256(value: Any) -> str:
    """Hash contiguous CPU F32 reference bytes in a host-independent order."""

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover
        raise CalibrationError("numpy is required for tensor hashing") from error
    array = value.detach().to(device="cpu", dtype=value.dtype).contiguous().numpy()
    if array.dtype == np.float32:
        canonical = np.ascontiguousarray(array.astype("<f4", copy=False))
    elif array.dtype == np.int64:
        canonical = np.ascontiguousarray(array.astype("<i8", copy=False))
    else:
        raise CalibrationError(f"unsupported tensor hash dtype {array.dtype}")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def f32_hex_words(value: Any) -> list[str]:
    """Return exact F32 golden values as portable little-endian hex words."""

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover
        raise CalibrationError("numpy is required for F32 golden encoding") from error
    array = value.detach().to(device="cpu", dtype=value.dtype).contiguous().numpy()
    if array.dtype != np.float32:
        raise CalibrationError(f"F32 golden requested for non-F32 tensor {array.dtype}")
    words = np.ascontiguousarray(array.astype("<f4", copy=False)).view("<u4").reshape(-1)
    return [f"0x{int(word):08x}" for word in words]


class ActivationTracker:
    """Range-only hooks; no activation tensor is retained between calls."""

    def __init__(self, torch: Any):
        self._torch = torch
        self._observations: dict[str, RangeObservation] = {}

    def observe(self, name: str, value: Any) -> None:
        if not isinstance(value, self._torch.Tensor):
            return
        if not value.is_floating_point():
            return
        if value.numel() < 1:
            return
        detached = value.detach()
        minimum = float(detached.amin().item())
        maximum = float(detached.amax().item())
        self._observations.setdefault(name, RangeObservation()).update(
            minimum=minimum,
            maximum=maximum,
            elements=int(detached.numel()),
        )

    def observe_tree(self, name: str, value: Any) -> None:
        if isinstance(value, self._torch.Tensor):
            self.observe(name, value)
        elif isinstance(value, (tuple, list)):
            for index, member in enumerate(value):
                self.observe_tree(f"{name}[{index}]", member)
        elif isinstance(value, dict):
            for key in sorted(value):
                self.observe_tree(f"{name}.{key}", value[key])

    def as_json(self) -> dict[str, dict[str, Any]]:
        return {name: self._observations[name].as_json() for name in sorted(self._observations)}

    @property
    def maximum_absolute(self) -> float:
        if not self._observations:
            raise CalibrationError("no activation probes were observed")
        return max(value.maximum_absolute for value in self._observations.values())

    @property
    def count(self) -> int:
        return len(self._observations)


def _mha_qkv_pre_hook(name: str, tracker: ActivationTracker, torch: Any) -> Callable[..., None]:
    """Probe q/k/v before attention, which torch does not expose as modules."""

    def hook(module: Any, args: tuple[Any, ...]) -> None:
        if len(args) < 3:
            raise CalibrationError(f"{name} MultiheadAttention hook received too few positional arguments")
        query, key, value = args[:3]
        if not all(isinstance(item, torch.Tensor) for item in (query, key, value)):
            raise CalibrationError(f"{name} MultiheadAttention hook received non-tensor q/k/v")
        weight = getattr(module, "in_proj_weight", None)
        bias = getattr(module, "in_proj_bias", None)
        dimension = int(getattr(module, "embed_dim", 0))
        if weight is None or dimension < 1 or tuple(weight.shape) != (3 * dimension, dimension):
            # The supplied TinyReceipt model uses the bundled self-attention
            # projection.  Failing closed keeps a changed reference architecture
            # from silently losing Q/K/V coverage.
            raise CalibrationError(f"{name} does not expose the expected packed in_proj_weight")
        if query.shape[-1] != dimension or key.shape[-1] != dimension or value.shape[-1] != dimension:
            raise CalibrationError(f"{name} q/k/v dimensions do not match packed projection")
        functional = torch.nn.functional
        tracker.observe(
            f"{name}.q",
            functional.linear(query, weight[:dimension], None if bias is None else bias[:dimension]),
        )
        tracker.observe(
            f"{name}.k",
            functional.linear(
                key,
                weight[dimension:2 * dimension],
                None if bias is None else bias[dimension:2 * dimension],
            ),
        )
        tracker.observe(
            f"{name}.v",
            functional.linear(
                value,
                weight[2 * dimension:3 * dimension],
                None if bias is None else bias[2 * dimension:3 * dimension],
            ),
        )

    return hook


def install_activation_probes(model: Any, torch: Any, tracker: ActivationTracker) -> list[Any]:
    """Install leaf, residual-boundary, structural, and Q/K/V range probes.

    Source-aligned structural names intentionally describe the value after a
    compound source operation.  They are evidence for a later materializer to
    bind QAdd/QSiLU and transformer residual boundaries without inferring them
    from the broad global fallback scale.
    """

    handles: list[Any] = []
    named_modules = dict(model.named_modules())
    for name, module in named_modules.items():
        if not name:
            continue
        # Module containers duplicate their children and can obscure which
        # intermediate was observed.  MultiheadAttention is the intentional
        # exception: it owns a packed Q/K/V projection while exposing an
        # out_proj child, so treating it as a plain container would lose the
        # required Q/K/V probes.
        is_mha = isinstance(module, torch.nn.MultiheadAttention)
        is_encoder_layer = isinstance(module, torch.nn.TransformerEncoderLayer)
        is_decoder_layer = isinstance(module, torch.nn.TransformerDecoderLayer)
        source_class_name = module.__class__.__name__
        is_res_block = source_class_name == "ResBlock"
        is_lora_linear = source_class_name == "LoRALinear"
        is_task_adapter = source_class_name == "TaskAdapter"
        structural_output = (
            is_mha
            or is_encoder_layer
            or is_decoder_layer
            or is_res_block
            or is_lora_linear
            or is_task_adapter
        )
        if any(True for _ in module.children()) and not structural_output:
            continue
        if is_res_block or is_task_adapter:
            output_probe_name = f"{name}.residual_output"
        elif is_lora_linear:
            output_probe_name = f"{name}.combined_output"
        elif is_encoder_layer or is_decoder_layer:
            output_probe_name = f"{name}.output"
        else:
            output_probe_name = name
        handles.append(
            module.register_forward_hook(
                lambda _module, _args, output, probe_name=output_probe_name: tracker.observe_tree(probe_name, output)
            )
        )
        if is_mha:
            handles.append(module.register_forward_pre_hook(_mha_qkv_pre_hook(name, tracker, torch)))
        if is_res_block:
            net = getattr(module, "net", None)
            if net is None:
                raise CalibrationError(f"{name} ResBlock is missing expected net module")

            # ResBlock.forward applies F.silu(x + self.net(x)).  The final
            # module hook above observes only the post-SiLU result, which is
            # not a safe range bound for the raw residual sum (large negative
            # values can be compressed by SiLU).  The inner Sequential hook
            # has both x and self.net(x), so record the exact QAdd boundary
            # without retaining either activation between invocations.
            def residual_sum_hook(
                _net: Any,
                args: tuple[Any, ...],
                output: Any,
                *,
                probe_name: str = f"{name}.residual_sum",
            ) -> None:
                if len(args) != 1 or not isinstance(args[0], torch.Tensor) or not isinstance(output, torch.Tensor):
                    raise CalibrationError(f"{probe_name} hook received an unexpected ResBlock net value")
                if tuple(args[0].shape) != tuple(output.shape):
                    raise CalibrationError(f"{probe_name} ResBlock input/net output shapes differ")
                tracker.observe(probe_name, args[0] + output)

            handles.append(net.register_forward_hook(residual_sum_hook))
        if is_lora_linear and (name == "linear1" or name.endswith(".linear1")):
            # TransformerEncoder/DecoderLayer applies its functional GELU to
            # the combined base+LoRA linear1 output.  The functional call has
            # no child module to hook, so calculate the same F.gelu value from
            # the source-aligned LoRALinear output for a tight QGELU scale.
            handles.append(
                module.register_forward_hook(
                    lambda _module, _args, output, probe_name=f"{name}.gelu_output": tracker.observe(
                        probe_name,
                        torch.nn.functional.gelu(output),
                    )
                )
            )
        if is_task_adapter:
            down = getattr(module, "down", None)
            if down is None:
                raise CalibrationError(f"{name} TaskAdapter is missing expected down projection")
            # TaskAdapter.forward applies F.gelu directly after down(), so
            # record the otherwise module-less QGELU output boundary.
            handles.append(
                down.register_forward_hook(
                    lambda _module, _args, output, probe_name=f"{name}.gelu_output": tracker.observe(
                        probe_name,
                        torch.nn.functional.gelu(output),
                    )
                )
            )
        if is_encoder_layer or is_decoder_layer:
            # The source uses norm_first=True.  norm1.input is the residual
            # stream before self-attention; encoder norm2.input is after the
            # self-attention QAdd and before FFN; decoder norm2.input is after
            # self-attention and before cross-attention; decoder norm3.input
            # is after cross-attention and before FFN.  Capture each directly.
            norm_names = ("norm1", "norm2") if is_encoder_layer else ("norm1", "norm2", "norm3")
            for norm_name in norm_names:
                norm = getattr(module, norm_name, None)
                if norm is None:
                    raise CalibrationError(f"{name} is missing expected {norm_name}")
                handles.append(
                    norm.register_forward_pre_hook(
                        lambda _module, args, probe_name=f"{name}.{norm_name}.input": (
                            tracker.observe(probe_name, args[0]) if args else None
                        )
                    )
                )

    for name, input_positions in (
        ("router.q_tokens", (0,)),
        ("encoder.input", (0,)),
        ("decoder.target_input", (0,)),
        ("decoder.memory_input", (1,)),
    ):
        module = named_modules.get(name.split(".", 1)[0]) if name == "router.q_tokens" else None
        if name == "encoder.input":
            module = getattr(model, "encoder", None)
        elif name.startswith("decoder."):
            module = getattr(model, "decoder", None)
        if module is None:
            continue

        def structural_hook(_module: Any, args: tuple[Any, ...], *, probe_name: str = name,
                            positions: tuple[int, ...] = input_positions) -> None:
            for position in positions:
                if len(args) > position:
                    tracker.observe(probe_name, args[position])

        handles.append(module.register_forward_pre_hook(structural_hook))
    return handles


def _remove_handles(handles: Iterable[Any]) -> None:
    for handle in handles:
        handle.remove()


def _topk_ids(logits: Any, count: int) -> list[int]:
    width = int(logits.numel())
    values, indices = logits.topk(min(count, width))
    # Values are intentionally not used as portable golden values: raw F32
    # words and a digest below bind the entire logits vector exactly.
    del values
    return [int(value) for value in indices.detach().to("cpu").tolist()]


def _validate_explicit_adapter_probe_coverage(
    model: Any,
    observations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove that a fixed-family sweep executed every source adapter module."""

    if not getattr(model.cfg, "use_adapters", False):
        raise CalibrationError("--include-explicit-families requires a release with task adapters")
    collections: list[dict[str, Any]] = []
    for collection_name in ("memory_adapters", "decoder_adapters"):
        adapters = getattr(model, collection_name, None)
        if adapters is None or len(adapters) != len(FAMILY_ORDER):
            size = 0 if adapters is None else len(adapters)
            raise CalibrationError(
                f"{collection_name} has {size} adapters; expected exactly {len(FAMILY_ORDER)}"
            )
        families: list[dict[str, Any]] = []
        for family_id, family in enumerate(FAMILY_ORDER):
            prefix = f"{collection_name}.{family_id}."
            probe_names = sorted(name for name in observations if name.startswith(prefix))
            if not probe_names:
                raise CalibrationError(
                    f"explicit-family reference sweep did not observe {collection_name} family {family_id}"
                )
            families.append({
                "family": family,
                "family_id": family_id,
                "probe_names": probe_names,
            })
        collections.append({
            "families": families,
            "name": collection_name,
        })
    return {
        "all_adapter_module_families_covered": True,
        "collections": collections,
    }


def _explicit_family_first_decode_steps(
    *,
    model: Any,
    image: Any,
    question: Any,
    decoder_input: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    """Run one sample as B=8 with fixed family_ids ``[0..7]``."""

    family_count = len(FAMILY_ORDER)
    explicit_image = image.repeat((family_count, 1, 1, 1))
    explicit_question = question.repeat((family_count, 1))
    explicit_decoder_input = decoder_input.repeat((family_count, 1))
    family_ids = torch.arange(family_count, dtype=torch.long)
    output = model(explicit_image, explicit_question, explicit_decoder_input, family_ids=family_ids)
    if not isinstance(output, tuple) or len(output) != 2:
        raise CalibrationError("explicit-family reference model did not return (logits, router_logits)")
    logits, router_logits = output
    if tuple(logits.shape[:2]) != (family_count, 1):
        raise CalibrationError(
            "explicit-family first-step logits have unexpected shape "
            f"{tuple(logits.shape)!r}; expected leading shape {(family_count, 1)!r}"
        )
    if tuple(router_logits.shape) != (family_count, family_count):
        raise CalibrationError(
            "explicit-family router logits have unexpected shape "
            f"{tuple(router_logits.shape)!r}; expected {(family_count, family_count)!r}"
        )
    steps: list[dict[str, Any]] = []
    for family_id, family in enumerate(FAMILY_ORDER):
        next_logits = logits[family_id, -1].to(dtype=torch.float32)
        if not bool(torch.isfinite(next_logits).all()):
            raise CalibrationError(f"explicit-family {family_id} first-step logits contain non-finite values")
        steps.append({
            "family": family,
            "family_id": family_id,
            "logits_f32_sha256": _tensor_bytes_sha256(next_logits),
            "next_token_id": int(next_logits.argmax().item()),
            "top_token_ids": _topk_ids(next_logits, 8),
        })
    return steps


def run_reference_samples(
    model: Any,
    vocab: Any,
    records: Sequence[HeldoutRecord],
    torch: Any,
    tracker: ActivationTracker,
    *,
    include_explicit_families: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run auto goldens and optionally fixed-family first-step coverage.

    The first tuple item is deliberately the pre-existing auto-router golden
    shape.  Explicit-family data is returned separately so clients can compare
    it without treating an auto-route golden as a fixed-route reference.
    """

    if not getattr(model.cfg, "use_router", False) or getattr(model, "router", None) is None:
        raise CalibrationError("the supplied release does not expose its required learned router")
    if not hasattr(vocab, "encode") or not hasattr(vocab, "bos"):
        raise CalibrationError("the packaged reference vocabulary lacks encode/bos")
    handles = install_activation_probes(model, torch, tracker)
    goldens: list[dict[str, Any]] = []
    explicit_family_goldens: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for record in records:
                image = preprocess_evaluator_image(record.image_path, torch)
                question_ids = vocab.encode(
                    record.question,
                    add_eos=True,
                    max_len=int(model.cfg.max_q_len),
                )
                if not question_ids:
                    raise CalibrationError(f"reference vocabulary encoded no ids for question {record.id!r}")
                question = torch.tensor(question_ids, dtype=torch.long)[None, :]
                decoder_input = torch.tensor([[int(vocab.bos)]], dtype=torch.long)
                tracker.observe("input.image_preprocessed", image)
                output = model(image, question, decoder_input, family_ids=None)
                if not isinstance(output, tuple) or len(output) != 2:
                    raise CalibrationError("reference model did not return (logits, router_logits)")
                logits, router_logits = output
                if router_logits.ndim != 2 or tuple(router_logits.shape) != (1, len(FAMILY_ORDER)):
                    raise CalibrationError(
                        "reference router has unexpected shape "
                        f"{tuple(router_logits.shape)!r}; expected {(1, len(FAMILY_ORDER))!r}"
                    )
                if logits.ndim != 3 or tuple(logits.shape[:2]) != (1, 1):
                    raise CalibrationError(f"reference first-step logits have unexpected shape {tuple(logits.shape)!r}")
                router_vector = router_logits[0].to(dtype=torch.float32)
                next_logits = logits[0, -1].to(dtype=torch.float32)
                if not bool(torch.isfinite(router_vector).all()) or not bool(torch.isfinite(next_logits).all()):
                    raise CalibrationError("reference router/logits goldens contain non-finite values")
                router_id = int(router_vector.argmax().item())
                token_id = int(next_logits.argmax().item())
                if router_id < 0 or router_id >= len(FAMILY_ORDER):
                    raise CalibrationError("reference router produced an out-of-range family id")
                goldens.append(
                    {
                        "annotation_sha256": sha256_file(record.annotation_path),
                        "answer": record.answer,
                        "expected_annotation_family": record.family,
                        "expected_annotation_family_id": record.family_id,
                        "first_decode_step": {
                            "decoder_input_ids": [int(vocab.bos)],
                            "logits_f32_hex_le": f32_hex_words(next_logits),
                            "logits_f32_sha256": _tensor_bytes_sha256(next_logits),
                            "next_token_id": token_id,
                            "top_token_ids": _topk_ids(next_logits, 8),
                        },
                        "id": record.id,
                        "image_sha256": sha256_file(record.image_path),
                        "preprocessed_image_f32_sha256": _tensor_bytes_sha256(image[0, 0]),
                        "question": record.question,
                        "question_token_ids": [int(value) for value in question_ids],
                        "router": {
                            "argmax_family": FAMILY_ORDER[router_id],
                            "argmax_family_id": router_id,
                            "logits_f32_hex_le": f32_hex_words(router_vector),
                            "logits_f32_sha256": _tensor_bytes_sha256(router_vector),
                            "shape": [len(FAMILY_ORDER)],
                        },
                    }
                )
                if include_explicit_families:
                    explicit_family_goldens.append({
                        "annotation_sha256": sha256_file(record.annotation_path),
                        "decoder_input_ids": [int(vocab.bos)],
                        "id": record.id,
                        "image_sha256": sha256_file(record.image_path),
                        "question": record.question,
                        "question_token_ids": [int(value) for value in question_ids],
                        "steps": _explicit_family_first_decode_steps(
                            model=model,
                            image=image,
                            question=question,
                            decoder_input=decoder_input,
                            torch=torch,
                        ),
                    })
    finally:
        _remove_handles(handles)
    return goldens, explicit_family_goldens


def conservative_symmetric_i8_scale(observed_abs_max: float, headroom: float) -> dict[str, Any]:
    """Derive an I8 scale with explicit slack above all observed magnitudes."""

    if not math.isfinite(observed_abs_max) or observed_abs_max <= 0.0:
        raise CalibrationError("observed activation absolute maximum must be finite and positive")
    if not math.isfinite(headroom) or headroom <= 1.0:
        raise CalibrationError("headroom must be finite and greater than 1.0")
    requested_threshold = observed_abs_max * headroom
    if not math.isfinite(requested_threshold):
        raise CalibrationError("headroom produces a non-finite activation threshold")
    # nextafter preserves strict headroom even after a downstream decimal JSON
    # round trip, rather than landing exactly on a 127-code boundary.
    scale = math.nextafter(requested_threshold / 127.0, math.inf)
    threshold = scale * 127.0
    return {
        "dtype": "int8",
        "observed_abs_max": observed_abs_max,
        "headroom_factor": headroom,
        "quantized_code_range": [-127, 127],
        "scale": scale,
        "symmetric": True,
        "threshold_abs": threshold,
        "zero_point": 0,
    }


def derive_probe_and_global_i8_scales(
    observations: Mapping[str, Mapping[str, Any]],
    headroom: float,
) -> dict[str, Any]:
    """Emit source-probe scales plus a graph-agnostic conservative fallback.

    Probe names are deliberately the PyTorch module/probe names, rather than
    invented Volvox edge names.  A later graph materializer must bind a graph
    edge to a source probe explicitly.  This gives it useful per-boundary
    calibration evidence now without quietly asserting a lowering that has
    not been authored or verified yet.
    """

    if not observations:
        raise CalibrationError("cannot derive activation scales without observations")
    observed_maxima: dict[str, float] = {}
    for name in sorted(observations):
        observation = observations[name]
        try:
            maximum = float(observation["observed_abs_max"])
        except (KeyError, TypeError, ValueError) as error:
            raise CalibrationError(f"activation observation {name!r} lacks observed_abs_max") from error
        if not math.isfinite(maximum) or maximum < 0.0:
            raise CalibrationError(f"activation observation {name!r} has an invalid observed_abs_max")
        observed_maxima[name] = maximum
    nonzero_maxima = [maximum for maximum in observed_maxima.values() if maximum > 0.0]
    if not nonzero_maxima:
        raise CalibrationError("all activation probes were zero; no positive global I8 scale can be derived")
    global_fallback = conservative_symmetric_i8_scale(max(nonzero_maxima), headroom)
    probe_boundaries: dict[str, dict[str, Any]] = {}
    for name in sorted(observed_maxima):
        maximum = observed_maxima[name]
        if maximum == 0.0:
            # Some unselected/zero-initialized adapter branches can be exactly
            # zero on a finite calibration subset.  Zero is exactly
            # representable at any positive symmetric scale, so reuse the
            # observed global fallback instead of inventing an arbitrary tiny
            # scale that would be unsafe when that branch becomes nonzero.
            scale = dict(global_fallback)
            scale["observed_abs_max"] = 0.0
            scale["observed_zero_only"] = True
            scale["scale_source"] = "global_fallback_for_zero_only_probe"
        else:
            scale = conservative_symmetric_i8_scale(maximum, headroom)
            scale["observed_zero_only"] = False
            scale["scale_source"] = "probe_observation"
        probe_boundaries[name] = scale
    return {
        "graph_edge_mapping": {
            "status": "unbound_source_probe_names",
            "requirement": (
                "A graph materializer must explicitly map each Volvox activation edge to one or more "
                "source probe boundaries before using a per-probe scale."
            ),
        },
        "global_fallback": global_fallback,
        "probe_boundaries": probe_boundaries,
        "scheme": "symmetric_per_tensor_i8",
    }


def source_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    """Return hashes for the release sources that affect reference outputs."""

    output: dict[str, str] = {}
    for name in ("manifest", "weights", "model_definition", "config", "vocab"):
        path = paths.get(name)
        if path is not None and path.is_file():
            output[f"{name}_sha256"] = sha256_file(path)
    required = {"manifest_sha256", "weights_sha256", "model_definition_sha256"}
    missing = sorted(required - set(output))
    if missing:
        raise CalibrationError(f"could not bind required release sources: {missing!r}")
    return output


def build_family_coverage(
    samples: Sequence[Mapping[str, Any]],
    explicit_family_goldens: Sequence[Mapping[str, Any]],
    *,
    include_explicit_families: bool,
    adapter_probe_coverage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Summarize actual auto and fixed-family evidence without graph coupling."""

    auto_ids: set[int] = set()
    for sample in samples:
        try:
            family_id = int(sample["router"]["argmax_family_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise CalibrationError("auto-router golden lacks router.argmax_family_id") from error
        if family_id < 0 or family_id >= len(FAMILY_ORDER):
            raise CalibrationError("auto-router golden has an out-of-range family id")
        auto_ids.add(family_id)

    explicit_requested = list(range(len(FAMILY_ORDER))) if include_explicit_families else []
    explicit_covered: set[int] = set()
    run_counts = {family: 0 for family in FAMILY_ORDER}
    if include_explicit_families:
        if adapter_probe_coverage is None:
            raise CalibrationError("explicit-family coverage is missing adapter-module probe evidence")
        if len(explicit_family_goldens) != len(samples):
            raise CalibrationError("explicit-family goldens must contain one fixed-family batch per auto sample")
        for auto_sample, explicit_sample in zip(samples, explicit_family_goldens):
            if (
                explicit_sample.get("id") != auto_sample.get("id")
                or explicit_sample.get("annotation_sha256") != auto_sample.get("annotation_sha256")
                or explicit_sample.get("image_sha256") != auto_sample.get("image_sha256")
            ):
                raise CalibrationError("explicit-family golden does not bind the corresponding auto sample")
            steps = explicit_sample.get("steps")
            if not isinstance(steps, list) or len(steps) != len(FAMILY_ORDER):
                raise CalibrationError("each explicit-family golden must contain exactly all eight family steps")
            observed_in_sample: set[int] = set()
            for step in steps:
                if not isinstance(step, Mapping):
                    raise CalibrationError("explicit-family step must be an object")
                try:
                    family_id = int(step["family_id"])
                    family = str(step["family"])
                    int(step["next_token_id"])
                except (KeyError, TypeError, ValueError) as error:
                    raise CalibrationError("explicit-family step lacks family_id/family/next_token_id") from error
                if family_id < 0 or family_id >= len(FAMILY_ORDER) or family != FAMILY_ORDER[family_id]:
                    raise CalibrationError("explicit-family step has an invalid family mapping")
                if family_id in observed_in_sample:
                    raise CalibrationError("explicit-family batch repeats a family id")
                observed_in_sample.add(family_id)
                explicit_covered.add(family_id)
                run_counts[family] += 1
            if observed_in_sample != set(range(len(FAMILY_ORDER))):
                raise CalibrationError("explicit-family batch did not cover every family id")
        if explicit_covered != set(explicit_requested):
            raise CalibrationError("explicit-family calibration did not cover every requested family id")
    elif explicit_family_goldens:
        raise CalibrationError("explicit-family goldens are present although explicit-family coverage is disabled")

    return {
        "auto_router": {
            "observed_argmax_family_ids": sorted(auto_ids),
            "observed_argmax_families": [FAMILY_ORDER[family_id] for family_id in sorted(auto_ids)],
            "sample_count": len(samples),
        },
        "explicit_family": {
            "adapter_module_probe_coverage": dict(adapter_probe_coverage or {}),
            "covered_family_ids": sorted(explicit_covered),
            "covered_families": [FAMILY_ORDER[family_id] for family_id in sorted(explicit_covered)],
            "enabled": include_explicit_families,
            "family_order": list(FAMILY_ORDER),
            "requested_family_ids": explicit_requested,
            "runs_per_family": run_counts,
            "total_runs": sum(run_counts.values()),
        },
    }


def build_calibration_contract(
    *,
    manifest: Mapping[str, Any],
    hashes: Mapping[str, str],
    samples: Sequence[Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
    activation_scales: Mapping[str, Any],
    explicit_family_goldens: Sequence[Mapping[str, Any]] = (),
    include_explicit_families: bool = False,
    adapter_probe_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable development-only JSON record from verified evidence."""

    if not samples:
        raise CalibrationError("a calibration contract requires at least one golden sample")
    if not observations:
        raise CalibrationError("a calibration contract requires activation observations")
    qkv_names = [name for name in observations if name.endswith((".q", ".k", ".v"))]
    if not qkv_names:
        raise CalibrationError("calibration contract lacks required Q/K/V activation probes")
    sample_identities = [
        {
            "annotation_sha256": str(sample["annotation_sha256"]),
            "family": str(sample["expected_annotation_family"]),
            "family_id": int(sample["expected_annotation_family_id"]),
            "id": str(sample["id"]),
            "image_sha256": str(sample["image_sha256"]),
            "question": str(sample["question"]),
        }
        for sample in samples
    ]
    sample_set_sha256 = canonical_json_sha256(sample_identities)
    config = _require_object(manifest.get("config"), "release manifest.config")
    family_coverage = build_family_coverage(
        samples,
        explicit_family_goldens,
        include_explicit_families=include_explicit_families,
        adapter_probe_coverage=adapter_probe_coverage,
    )
    return {
        "activation_observations": {name: observations[name] for name in sorted(observations)},
        "development_only": True,
        "family_coverage": family_coverage,
        "format": FORMAT,
        "goldens": {
            "explicit_family_first_decode_steps": list(explicit_family_goldens),
            "first_decode_step_and_router": list(samples),
        },
        "profile": {
            "activation_quantization": dict(activation_scales),
            "id": PROFILE_ID,
            "model_config": dict(config),
            "preprocessing": {
                "color_space": "grayscale",
                "input_value_range": [0, 255],
                "layout": "NCHW",
                "normalization": "(pixel / 255.0 - 0.5) / 0.5",
                "output_dtype": "float32",
                "output_value_range": [-1.0, 1.0],
                "resize": {
                    "height": IMAGE_HEIGHT,
                    "interpolation": "bilinear",
                    "width": IMAGE_WIDTH,
                },
            },
            "reference_execution": {
                "decoder": "one BOS-only first step",
                "device": "cpu",
                "explicit_family_sweep": include_explicit_families,
                "mode": "torch.inference_mode+model.eval",
                "router": "learned_auto_router",
                "threads": 1,
                "weight_loading": "named SafeTensors INT8 per-output dequantized only in PyTorch reference memory",
            },
        },
        "purpose": (
            "development reference observations and first-step goldens; not a Volvox graph, "
            "not an executable W8A8 conversion contract"
        ),
        "qkv_probe_count": len(qkv_names),
        "sample_identities": sample_identities,
        "sample_set_sha256": sample_set_sha256,
        "source": dict(hashes),
    }


def generate_calibration_contract(
    *,
    release_dir: Path,
    annotations_dir: Path,
    images_dir: Path,
    sample_count: int,
    headroom: float,
    include_explicit_families: bool = False,
    torch: Any | None = None,
) -> dict[str, Any]:
    """Run the complete deterministic reference-calibration workflow."""

    if torch is None:
        try:
            import torch as imported_torch
        except ImportError as error:  # pragma: no cover - environment dependency.
            raise CalibrationError("PyTorch is required for TinyReceipt reference calibration") from error
        torch = imported_torch
    if sample_count < 1:
        raise CalibrationError("sample_count must be positive")
    if not math.isfinite(headroom) or headroom <= 1.0:
        raise CalibrationError("headroom must be finite and greater than 1.0")
    # CPU/one-thread prevents GPU kernels and host-thread reduction ordering
    # from making a supposedly deterministic golden file vary by machine.
    torch.set_num_threads(1)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    model, vocab, manifest, paths = load_reference_model(Path(release_dir), torch)
    records = discover_heldout_records(Path(annotations_dir), Path(images_dir))
    selected = select_stratified_records(records, sample_count)
    tracker = ActivationTracker(torch)
    samples, explicit_family_goldens = run_reference_samples(
        model,
        vocab,
        selected,
        torch,
        tracker,
        include_explicit_families=include_explicit_families,
    )
    observations = tracker.as_json()
    adapter_probe_coverage = (
        _validate_explicit_adapter_probe_coverage(model, observations)
        if include_explicit_families
        else None
    )
    activation_scales = derive_probe_and_global_i8_scales(observations, headroom)
    return build_calibration_contract(
        manifest=manifest,
        hashes=source_hashes(paths),
        samples=samples,
        observations=observations,
        activation_scales=activation_scales,
        explicit_family_goldens=explicit_family_goldens,
        include_explicit_families=include_explicit_families,
        adapter_probe_coverage=adapter_probe_coverage,
    )


def _receipt_data_path(*parts: str) -> Path | None:
    root = os.environ.get("RECEIPT_VQA_DATA_ROOT")
    if not root:
        return None
    return Path(root).expanduser().joinpath(*parts)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    release_dir = _receipt_data_path(
        "temp", "tiny-receipt-vqa-structured-qa-21m-lora-router-e100-v1"
    )
    annotations_dir = _receipt_data_path("eval", "heldout", "annotations")
    images_dir = _receipt_data_path("eval", "heldout", "images")
    parser = argparse.ArgumentParser(
        description="Generate development-only TinyReceiptVQA W8A8 reference calibration/goldens.",
        epilog=(
            "Set RECEIPT_VQA_DATA_ROOT=/path/to/receipt-vqa-data to use the "
            "standard release and heldout-data locations, or pass all three "
            "path options explicitly."
        ),
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=release_dir,
        required=release_dir is None,
        help=(
            "release root containing int8/manifest.json and modeling_tiny_receipt_vqa.py "
            "(default: standard path under $RECEIPT_VQA_DATA_ROOT)"
        ),
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=annotations_dir,
        required=annotations_dir is None,
        help=(
            "local heldout annotations directory "
            "(default: $RECEIPT_VQA_DATA_ROOT/eval/heldout/annotations)"
        ),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=images_dir,
        required=images_dir is None,
        help=(
            "local heldout images directory "
            "(default: $RECEIPT_VQA_DATA_ROOT/eval/heldout/images)"
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="destination JSON path (development output; not a graph conversion contract)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=8,
        help="number of deterministic heldout records to observe (default: 8)",
    )
    parser.add_argument(
        "--headroom",
        type=float,
        default=1.10,
        help="strict multiplicative slack over observed abs max for global I8 scale (default: 1.10)",
    )
    parser.add_argument(
        "--include-explicit-families",
        action="store_true",
        help=(
            "for every selected heldout sample, run one replicated B=8 first decode step with "
            "family_ids=[0,1,2,3,4,5,6,7] and record fixed-family coverage"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        contract = generate_calibration_contract(
            release_dir=args.release_dir,
            annotations_dir=args.annotations_dir,
            images_dir=args.images_dir,
            sample_count=args.sample_count,
            headroom=args.headroom,
            include_explicit_families=args.include_explicit_families,
        )
        write_deterministic_json(args.out, contract)
    except CalibrationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "format": FORMAT,
        "output": str(args.out),
        "qkv_probe_count": contract["qkv_probe_count"],
        "sample_count": len(contract["sample_identities"]),
        "explicit_family_coverage": contract["family_coverage"]["explicit_family"]["covered_family_ids"],
        "global_fallback_scale": contract["profile"]["activation_quantization"]["global_fallback"]["scale"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
