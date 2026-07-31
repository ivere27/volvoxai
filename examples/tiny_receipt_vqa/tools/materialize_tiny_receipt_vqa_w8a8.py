#!/usr/bin/env python3
"""Materialize a TinyReceiptVQA INT8 release as Volvox inference graph documents.

This is a *development-only* converter for either the published
``tiny_receipt_vqa_int8_safetensors_v1`` release or the separately identified
``tiny_receipt_vqa_volvox_trained_int8_safetensors_v1`` output of the example's
JS PTQ mapper. Unlike the adjacent weight-only normalizer, it deliberately
writes runnable Volvox graph documents:

* one B=1 hard-router graph, and
* one B=1 complete encoder/decoder graph for each of the eight explicit task
  adapter families, using W8A8 by default or weight-only W8A32 when requested.

The W8A8 router always requires activation calibration.  A caller must
therefore provide exactly one profile source: a positive symmetric
``--activation-scale`` fallback, an explicit logical-edge profile, or the
repository's ``--development-calibration`` source-probe record.  The latter is
copied into the output package and bound edge-by-edge; this materializer never
runs calibration or reference/golden validation itself.  Immutable F32
position/type constants are instead quantized with their own recorded,
tightly-derived symmetric scales.  The emitted package manifest distinguishes
source-probe bindings, intentional fallback edges, static constants, and
execution limitations instead of implying unsupported source-model fidelity.

The optional W8A32 family mode keeps the learned router in W8A8 so AUTO remains
comparable, while explicit-family activations, biases, embeddings, and constants
are F32.  It reuses the source I8 weights and exact per-output scale tensors.

Each graph owns one central reference-only affine table.  Its finite positive
F32 scales and typed symmetric zero points live in the shared SafeTensors file;
neither numeric values nor a second descriptor map are stored in JSON or
SafeTensors metadata.

The tool intentionally does not implement a host autoregressive session,
preprocessing, calibration, golden-output verification, or automatic router
to adapter-graph dispatch.  The router graph emits a family ID; its caller
must select the corresponding explicit-family B=1 graph document for the complete
generation run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from safetensors.numpy import load_file, save_file

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from tools.exporter.errors import ExporterError
from tools.exporter.quantization_storage import (
    QUANTIZATION_FORMAT,
    validate_external_quantization,
)


SOURCE_FORMAT = "tiny_receipt_vqa_int8_safetensors_v1"
SOURCE_RUNTIME = "pytorch-reference-dequantized-safetensors-v1"
VOLVOX_TRAINED_SOURCE_FORMAT = "tiny_receipt_vqa_volvox_trained_int8_safetensors_v1"
VOLVOX_TRAINED_SOURCE_RUNTIME = "volvoxai-js-ptq-trained-f32-v1"
SOURCE_QUANTIZATION = "symmetric per-output-channel INT8"
SOURCE_LAYOUT = {
    "int8_per_out": "quantized.<state_dict_key>",
    "scale_fp32": "scale.<state_dict_key>",
    "float32": "float32.<state_dict_key>",
}
VOLVOX_TRAINED_SOURCE_LAYOUT = {
    "int8_per_out": "quantized.<state_dict_key>",
    "scale_fp32": "scale.<state_dict_key>",
    "zero_point_i8": "quantized.<state_dict_key>.zero_point",
    "float32": "float32.<state_dict_key>",
}

PACKAGE_FORMAT = "volvoxai-tiny-receipt-vqa-w8a8-materialized-package-v1"
WEIGHTS_FORMAT = "volvoxai-tiny-receipt-vqa-w8a8-materialized-weights-v2"
W8A32_WEIGHTS_FORMAT = "volvoxai-tiny-receipt-vqa-w8a32-family-weights-v2"
GRAPH_FORMAT = "volvox-graph/v1"
W8A8_GRAPH_PROFILE = "volvoxai-graph-w8a8-v2"
W8A32_GRAPH_PROFILE = "volvoxai-graph-w8a32-v2"
CALIBRATION_FORMAT = "volvoxai-tiny-receipt-vqa-w8a8-activation-scales-v1"
SCOPED_WEIGHTS_DIR = "scoped_weights"
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
IMAGE_CHANNELS = 1
IMAGE_TOKENS = 210
ACTIVATION_DTYPE = "int8"
ACTIVATION_ZERO_POINT = 0
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


class MaterializationError(ValueError):
    """The source release or requested materialization contract is invalid."""


@dataclass(frozen=True)
class ModelDimensions:
    vocab_size: int
    d_model: int
    heads: int
    enc_layers: int
    dec_layers: int
    ff_mult: int
    max_q_len: int
    max_out_len: int
    img_tokens: int
    adapter_bottleneck: int
    lora_r: int
    lora_alpha: float
    stem_channels: tuple[int, int, int, int, int]

    @property
    def ff_dim(self) -> int:
        return self.d_model * self.ff_mult

    @property
    def head_dim(self) -> int:
        return self.d_model // self.heads

    @property
    def lora_scale(self) -> np.float32:
        return np.float32(np.float32(self.lora_alpha) / np.float32(self.lora_r))


@dataclass(frozen=True)
class SourceRelease:
    manifest_path: Path
    weights_path: Path
    manifest: Mapping[str, Any]
    dimensions: ModelDimensions
    tensors: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class QuantizedWeight:
    name: str
    shape: tuple[int, ...]
    scales: np.ndarray


@dataclass(frozen=True)
class MaterializationResult:
    output_dir: Path
    weights_path: Path
    router_graph_path: Path
    explicit_graph_paths: tuple[Path, ...]
    package_manifest_path: Path
    constant_saturation_count: int
    family_execution: str


class ActivationProfile:
    """Resolve deterministic per-edge I8 descriptors for one materialization.

    A one-scale command line invocation is intentionally marked as a fallback
    profile.  An external profile may provide a default plus stable logical
    edge overrides, or may omit the default and enumerate every consumed
    logical edge.  The latter is useful for a fully calibrated package.
    """

    def __init__(
        self,
        *,
        profile_id: str,
        default_scale: np.float32 | None,
        overrides: Mapping[str, np.float32],
        source: str,
        calibration_path: Path | None = None,
        strict_source_probes: bool = False,
        zero_only_source_probes: Iterable[str] = (),
    ):
        self.profile_id = profile_id
        self.default_scale = default_scale
        self.overrides = dict(overrides)
        self.source = source
        self.calibration_path = calibration_path
        self.strict_source_probes = strict_source_probes
        self.zero_only_source_probes = set(zero_only_source_probes)
        self.used: dict[str, dict[str, Any]] = {}

    @classmethod
    def fallback(cls, scale: Any) -> "ActivationProfile":
        return cls(
            profile_id="caller-supplied-single-scale",
            default_scale=_require_finite_positive_f32(scale, "--activation-scale"),
            overrides={},
            source="single_scale_fallback",
        )

    @classmethod
    def from_development_calibration(cls, path: Path) -> "ActivationProfile":
        value = _json_object(path, "activation calibration profile")
        if value.get("format") != "volvoxai-tiny-receipt-vqa-w8a8-development-calibration-v1":
            raise MaterializationError("--development-calibration requires the repository development calibration format")
        profile = value.get("profile")
        if not isinstance(profile, dict):
            raise MaterializationError("development calibration profile must contain profile object")
        activation = profile.get("activation_quantization")
        if not isinstance(activation, dict):
            raise MaterializationError("development calibration lacks activation_quantization")
        fallback = activation.get("global_fallback")
        probes = activation.get("probe_boundaries")
        if not isinstance(fallback, dict) or not isinstance(probes, dict):
            raise MaterializationError("development calibration lacks global_fallback or probe_boundaries")
        default_scale = _require_finite_positive_f32(
            fallback.get("scale"), "development calibration global fallback scale"
        )
        overrides: dict[str, np.float32] = {}
        zero_only: set[str] = set()
        for key, record in probes.items():
            if not isinstance(key, str) or not key or not isinstance(record, dict):
                raise MaterializationError("development calibration probe_boundaries must map names to objects")
            if record.get("dtype") != "int8" or record.get("symmetric") is not True or record.get("zero_point") != 0:
                raise MaterializationError(f"development calibration probe {key!r} is not symmetric I8 zero-point-zero")
            overrides[key] = _require_finite_positive_f32(
                record.get("scale"), f"development calibration probe {key!r} scale"
            )
            if record.get("observed_zero_only") is True or record.get("scale_source") == "global_fallback_for_zero_only_probe":
                zero_only.add(key)
        return cls(
            profile_id=_require_string(profile.get("id"), "development calibration profile.id"),
            default_scale=default_scale,
            overrides=overrides,
            source="development_calibration_probe_boundaries",
            calibration_path=Path(path),
            strict_source_probes=True,
            zero_only_source_probes=zero_only,
        )

    @classmethod
    def from_json(cls, path: Path) -> "ActivationProfile":
        value = _json_object(path, "activation calibration profile")
        if value.get("format") == "volvoxai-tiny-receipt-vqa-w8a8-development-calibration-v1":
            raise MaterializationError(
                "use --development-calibration for a source-probe calibration record; "
                "it enforces explicit source-probe bindings"
            )
        allowed = {"format", "profile_id", "default_scale", "scales"}
        unknown = sorted(set(value) - allowed)
        missing = sorted({"format", "profile_id", "scales"} - set(value))
        if missing or unknown:
            raise MaterializationError(
                "activation calibration profile must contain format, profile_id, scales, and optional default_scale"
            )
        if value.get("format") != CALIBRATION_FORMAT:
            raise MaterializationError(f"activation calibration profile.format must be {CALIBRATION_FORMAT!r}")
        profile_id = _require_string(value.get("profile_id"), "activation calibration profile.profile_id")
        raw_scales = value.get("scales")
        if not isinstance(raw_scales, dict):
            raise MaterializationError("activation calibration profile.scales must be an object")
        overrides: dict[str, np.float32] = {}
        for key, scale in raw_scales.items():
            if not isinstance(key, str) or not key or key == "default":
                raise MaterializationError("activation calibration profile scale keys must be non-empty logical edge IDs")
            overrides[key] = _require_finite_positive_f32(scale, f"activation calibration profile.scales[{key!r}]")
        default_raw = value.get("default_scale")
        default_scale = None if default_raw is None else _require_finite_positive_f32(
            default_raw, "activation calibration profile.default_scale"
        )
        if default_scale is None and not overrides:
            raise MaterializationError("activation calibration profile needs default_scale or at least one edge scale")
        return cls(
            profile_id=profile_id,
            default_scale=default_scale,
            overrides=overrides,
            source="external_calibration_profile",
            calibration_path=Path(path),
        )

    def scale_for(
        self,
        logical_edge: str,
        *,
        source_probe: str | None = None,
        multiplier: float = 1.0,
    ) -> np.float32:
        multiplier_f32 = _require_finite_positive_f32(multiplier, f"calibration multiplier for {logical_edge}")
        if logical_edge in self.overrides:
            source_scale = self.overrides[logical_edge]
            origin = "logical_edge_override"
        elif source_probe is not None and source_probe in self.overrides:
            source_scale = self.overrides[source_probe]
            origin = (
                "zero_only_probe_global_fallback"
                if source_probe in self.zero_only_source_probes else "source_probe_override"
            )
        elif source_probe is not None and self.strict_source_probes:
            raise MaterializationError(
                f"development calibration is missing required source probe {source_probe!r} for logical edge {logical_edge!r}"
            )
        elif self.default_scale is not None:
            source_scale = self.default_scale
            origin = "default"
        else:
            raise MaterializationError(
                f"activation calibration profile does not define logical edge {logical_edge!r} "
                f"or source probe {source_probe!r}"
            )
        scale = np.float32(source_scale * multiplier_f32)
        if not math.isfinite(float(scale)) or scale <= 0:
            raise MaterializationError(f"calibrated scale for {logical_edge!r} is not positive finite F32")
        self.used[logical_edge] = {
            "scale": float(scale),
            "origin": origin,
            "source_probe": source_probe,
            "source_scale": float(source_scale),
            "source_multiplier": float(multiplier_f32),
        }
        return scale

    def descriptor(self, logical_edge: str, *, source_probe: str | None = None) -> dict[str, Any]:
        return {
            "scheme": "per_tensor",
            "scale": float(self.scale_for(logical_edge, source_probe=source_probe)),
            "zero_point": ACTIVATION_ZERO_POINT,
        }

    def manifest_record(self) -> dict[str, Any]:
        consumed_override_names = set(self.used)
        consumed_override_names.update(
            binding["source_probe"] for binding in self.used.values() if binding["source_probe"] is not None
        )
        unused = sorted(set(self.overrides) - consumed_override_names)
        unqualified_fallbacks = sorted(
            key for key, binding in self.used.items()
            if binding["origin"] == "default" and binding["source_probe"] is None
        )
        source_probe_fallbacks = sorted(
            key for key, binding in self.used.items()
            if binding["origin"] == "default" and binding["source_probe"] is not None
        )
        bound = sorted(
            key for key, binding in self.used.items()
            if binding["origin"] == "source_probe_override"
        )
        zero_only_reused = sorted(
            key for key, binding in self.used.items()
            if binding["origin"] == "zero_only_probe_global_fallback"
        )
        record: dict[str, Any] = {
            "format": CALIBRATION_FORMAT,
            "profile_id": self.profile_id,
            "source": self.source,
            "default_scale": None if self.default_scale is None else float(self.default_scale),
            "used_logical_edges": dict(sorted(self.used.items())),
            "unused_override_edges": unused,
            "fully_enumerated": self.default_scale is None,
            "strict_source_probes": self.strict_source_probes,
            "source_probe_bound_logical_edges": bound,
            "unqualified_global_fallback_logical_edges": unqualified_fallbacks,
            "source_probe_global_fallback_logical_edges": source_probe_fallbacks,
            "zero_only_probe_global_fallback_logical_edges": zero_only_reused,
            "qualified_per_edge_calibration": bool(self.source == "development_calibration_probe_boundaries" and
                                                    not unqualified_fallbacks and not source_probe_fallbacks and
                                                    not zero_only_reused),
        }
        if self.calibration_path is not None:
            # The source path is intentionally not emitted: package consumers
            # must not be induced to escape their package directory.
            record["source_file_basename"] = self.calibration_path.name
            record["sha256"] = _sha256(self.calibration_path)
        return record


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MaterializationError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise MaterializationError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise MaterializationError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MaterializationError(f"{label} must be a non-empty string")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MaterializationError(f"{label} must be a positive integer")
    return value


def _require_finite_positive_f32(value: Any, label: str) -> np.float32:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise MaterializationError(f"{label} must be a positive finite number")
    converted = np.float32(value)
    if not math.isfinite(float(value)) or not math.isfinite(float(converted)) or converted <= 0:
        raise MaterializationError(f"{label} must be representable as a positive finite F32")
    return converted


def _array(value: Any, *, name: str, dtype: np.dtype[Any], shape: Sequence[int]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise MaterializationError(f"SafeTensors entry {name!r} is not an array")
    expected_shape = tuple(int(item) for item in shape)
    if value.dtype != dtype:
        raise MaterializationError(
            f"SafeTensors entry {name!r} has dtype {value.dtype}, expected {dtype}"
        )
    if tuple(value.shape) != expected_shape:
        raise MaterializationError(
            f"SafeTensors entry {name!r} has shape {list(value.shape)}, expected {list(expected_shape)}"
        )
    return value


def _groups(channels: int) -> int:
    for candidate in (32, 16, 8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1


def _validate_dimensions(config: Any) -> ModelDimensions:
    if not isinstance(config, dict):
        raise MaterializationError("manifest.config must be an object")
    d_model = _require_positive_int(config.get("d_model"), "manifest.config.d_model")
    raw_stem_channels = config.get("stem_channels", [48, 96, 192, d_model, d_model])
    if not isinstance(raw_stem_channels, list) or len(raw_stem_channels) != 5:
        raise MaterializationError("manifest.config.stem_channels must contain five channel counts")
    stem_channels = tuple(
        _require_positive_int(value, f"manifest.config.stem_channels[{index}]")
        for index, value in enumerate(raw_stem_channels)
    )
    if stem_channels[3:] != (d_model, d_model):
        raise MaterializationError("manifest.config.stem_channels final two entries must equal d_model")
    dimensions = ModelDimensions(
        vocab_size=_require_positive_int(config.get("vocab_size"), "manifest.config.vocab_size"),
        d_model=d_model,
        heads=_require_positive_int(config.get("heads"), "manifest.config.heads"),
        enc_layers=_require_positive_int(config.get("enc_layers"), "manifest.config.enc_layers"),
        dec_layers=_require_positive_int(config.get("dec_layers"), "manifest.config.dec_layers"),
        ff_mult=_require_positive_int(config.get("ff_mult"), "manifest.config.ff_mult"),
        max_q_len=_require_positive_int(config.get("max_q_len"), "manifest.config.max_q_len"),
        max_out_len=_require_positive_int(config.get("max_out_len"), "manifest.config.max_out_len"),
        img_tokens=_require_positive_int(config.get("img_tokens"), "manifest.config.img_tokens"),
        adapter_bottleneck=_require_positive_int(
            config.get("adapter_bottleneck"), "manifest.config.adapter_bottleneck"
        ),
        lora_r=_require_positive_int(config.get("lora_r"), "manifest.config.lora_r"),
        lora_alpha=float(_require_finite_positive_f32(config.get("lora_alpha"), "manifest.config.lora_alpha")),
        stem_channels=stem_channels,
    )
    if dimensions.d_model % 4 != 0 or dimensions.d_model % dimensions.heads != 0:
        raise MaterializationError("manifest.config requires d_model divisible by 4 and heads")
    if dimensions.head_dim % 4 != 0 or dimensions.head_dim > 64:
        raise MaterializationError("manifest.config head dimension must be divisible by 4 and <= 64")
    if dimensions.img_tokens != IMAGE_TOKENS:
        raise MaterializationError(
            f"manifest.config.img_tokens must be {IMAGE_TOKENS} for the fixed {IMAGE_HEIGHT}x{IMAGE_WIDTH} stem"
        )
    if config.get("use_adapters") is not True or config.get("use_router") is not True:
        raise MaterializationError("materialization requires the published router and task-adapter architecture")
    if config.get("adapter_families") != len(FAMILY_ORDER):
        raise MaterializationError(
            f"manifest.config.adapter_families must be {len(FAMILY_ORDER)}"
        )
    if config.get("lora_targets") != "transformer":
        raise MaterializationError("materialization supports the published transformer-only LoRA target set")
    return dimensions


def _stem_specs(dimensions: ModelDimensions) -> tuple[dict[str, Any], ...]:
    first, second, third, fourth, fifth = dimensions.stem_channels
    return (
        {"kind": "conv", "index": 0, "cin": 1, "cout": first, "stride": 2, "conv": 0, "norm": 1},
        {"kind": "conv", "index": 1, "cin": first, "cout": second, "stride": 2, "conv": 0, "norm": 1},
        {"kind": "res", "index": 2, "channels": second},
        {"kind": "conv", "index": 3, "cin": second, "cout": third, "stride": 2, "conv": 0, "norm": 1},
        {"kind": "res", "index": 4, "channels": third},
        {"kind": "conv", "index": 5, "cin": third, "cout": fourth, "stride": 2, "conv": 0, "norm": 1},
        {"kind": "res", "index": 6, "channels": fourth},
        {"kind": "conv", "index": 7, "cin": fourth, "cout": fifth, "stride": 2, "conv": 0, "norm": 1},
        {"kind": "res", "index": 8, "channels": fifth},
    )


def _expected_source_shapes(dimensions: ModelDimensions) -> dict[str, tuple[np.dtype[Any], tuple[int, ...]]]:
    """Exact state-dict contract for the release this tool understands."""
    expected: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {}

    def q(name: str, shape: Sequence[int]) -> None:
        expected[name] = (np.dtype(np.int8), tuple(shape))

    def f(name: str, shape: Sequence[int]) -> None:
        expected[name] = (np.dtype(np.float32), tuple(shape))

    d = dimensions.d_model
    f("img_pos", (1, dimensions.img_tokens, d))
    f("q_pos", (1, dimensions.max_q_len, d))
    f("y_pos", (1, dimensions.max_out_len, d))
    f("type_img", (1, 1, d))
    f("type_q", (1, 1, d))
    f("out_bias", (dimensions.vocab_size,))

    for spec in _stem_specs(dimensions):
        index = spec["index"]
        if spec["kind"] == "conv":
            q(f"stem.{index}.net.{spec['conv']}.weight", (spec["cout"], spec["cin"], 3, 3))
            f(f"stem.{index}.net.{spec['norm']}.weight", (spec["cout"],))
            f(f"stem.{index}.net.{spec['norm']}.bias", (spec["cout"],))
        else:
            channels = spec["channels"]
            q(f"stem.{index}.net.0.weight", (channels, channels, 3, 3))
            f(f"stem.{index}.net.1.weight", (channels,))
            f(f"stem.{index}.net.1.bias", (channels,))
            q(f"stem.{index}.net.3.weight", (channels, channels, 3, 3))
            f(f"stem.{index}.net.4.weight", (channels,))
            f(f"stem.{index}.net.4.bias", (channels,))

    q("tok.weight", (dimensions.vocab_size, d))
    for prefix, count, decoder in (
        ("encoder.layers", dimensions.enc_layers, False),
        ("decoder.layers", dimensions.dec_layers, True),
    ):
        for layer in range(count):
            root = f"{prefix}.{layer}"
            attention_names = ["self_attn"] + (["multihead_attn"] if decoder else [])
            for attention in attention_names:
                q(f"{root}.{attention}.in_proj_weight", (3 * d, d))
                f(f"{root}.{attention}.in_proj_bias", (3 * d,))
                q(f"{root}.{attention}.out_proj.weight", (d, d))
                f(f"{root}.{attention}.out_proj.bias", (d,))
            for linear, dout, din in (("linear1", dimensions.ff_dim, d), ("linear2", d, dimensions.ff_dim)):
                q(f"{root}.{linear}.base.weight", (dout, din))
                f(f"{root}.{linear}.base.bias", (dout,))
                q(f"{root}.{linear}.lora_a.weight", (dimensions.lora_r, din))
                q(f"{root}.{linear}.lora_b.weight", (dout, dimensions.lora_r))
            norm_count = 3 if decoder else 2
            for norm in range(1, norm_count + 1):
                f(f"{root}.norm{norm}.weight", (d,))
                f(f"{root}.norm{norm}.bias", (d,))

    q("router.net.0.weight", (d // 2, d))
    f("router.net.0.bias", (d // 2,))
    q("router.net.2.weight", (len(FAMILY_ORDER), d // 2))
    f("router.net.2.bias", (len(FAMILY_ORDER),))
    for group in ("memory_adapters", "decoder_adapters"):
        for family in range(len(FAMILY_ORDER)):
            root = f"{group}.{family}"
            q(f"{root}.down.weight", (dimensions.adapter_bottleneck, d))
            f(f"{root}.down.bias", (dimensions.adapter_bottleneck,))
            q(f"{root}.up.weight", (d, dimensions.adapter_bottleneck))
            f(f"{root}.up.bias", (d,))
    f("norm.weight", (d,))
    f("norm.bias", (d,))
    q("head.weight", (dimensions.vocab_size, d))
    return expected


def _validate_manifest_tensors(
    manifest: Mapping[str, Any], expected: Mapping[str, tuple[np.dtype[Any], tuple[int, ...]]]
) -> None:
    raw = manifest.get("tensors")
    if not isinstance(raw, dict):
        raise MaterializationError("manifest.tensors must be an object")
    actual_names = set(raw)
    expected_names = set(expected)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        pieces: list[str] = []
        if missing:
            pieces.append(f"missing state-dict entries {missing!r}")
        if unexpected:
            pieces.append(f"unexpected state-dict entries {unexpected!r}")
        raise MaterializationError("manifest.tensors does not match the TinyReceipt architecture; " + "; ".join(pieces))
    for key, (dtype, shape) in expected.items():
        record = raw[key]
        if not isinstance(record, dict):
            raise MaterializationError(f"manifest.tensors[{key!r}] must be an object")
        expected_kind = "int8_per_out" if dtype == np.dtype(np.int8) else "float32"
        if record.get("dtype") != expected_kind or record.get("shape") != list(shape):
            raise MaterializationError(
                f"manifest.tensors[{key!r}] must declare {expected_kind} shape {list(shape)}"
            )


def load_release(manifest_path: Path) -> SourceRelease:
    """Load and strictly validate the named TinyReceipt source tensors."""
    manifest_path = Path(manifest_path)
    manifest = _json_object(manifest_path, "source manifest")
    source_contracts = {
        SOURCE_FORMAT: SOURCE_RUNTIME,
        VOLVOX_TRAINED_SOURCE_FORMAT: VOLVOX_TRAINED_SOURCE_RUNTIME,
    }
    source_format = manifest.get("format")
    source_runtime = manifest.get("runtime")
    if source_format not in source_contracts:
        raise MaterializationError(
            f"manifest.format must be one of {sorted(source_contracts)!r}"
        )
    if source_runtime != source_contracts[source_format]:
        raise MaterializationError(
            f"manifest.runtime must be {source_contracts[source_format]!r} for {source_format!r}"
        )
    safetensors = manifest.get("safetensors")
    expected_layout = (VOLVOX_TRAINED_SOURCE_LAYOUT
                       if source_format == VOLVOX_TRAINED_SOURCE_FORMAT else SOURCE_LAYOUT)
    if not isinstance(safetensors, dict) or safetensors.get("layout") != expected_layout or \
            safetensors.get("quantization") != SOURCE_QUANTIZATION:
        raise MaterializationError("manifest.safetensors is not the supported named INT8 layout")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise MaterializationError("manifest.files must be an object")
    filename = _require_string(files.get("model_safetensors"), "manifest.files.model_safetensors")
    if Path(filename).name != filename or not filename.endswith(".safetensors"):
        raise MaterializationError("manifest.files.model_safetensors must be a local .safetensors filename")
    dimensions = _validate_dimensions(manifest.get("config"))
    expected = _expected_source_shapes(dimensions)
    _validate_manifest_tensors(manifest, expected)
    weights_path = manifest_path.parent / filename
    if not weights_path.is_file():
        raise MaterializationError(f"source SafeTensors payload is missing: {weights_path}")
    try:
        tensors = load_file(str(weights_path))
    except Exception as error:  # safetensors exposes a Rust exception type.
        raise MaterializationError(f"could not read source SafeTensors payload {weights_path}: {error}") from error

    expected_entries: set[str] = set()
    for key, (dtype, shape) in expected.items():
        if dtype == np.dtype(np.int8):
            expected_entries.add(f"quantized.{key}")
            expected_entries.add(f"scale.{key}")
            if source_format == VOLVOX_TRAINED_SOURCE_FORMAT:
                expected_entries.add(f"quantized.{key}.zero_point")
        else:
            expected_entries.add(f"float32.{key}")
    actual_entries = set(tensors)
    if actual_entries != expected_entries:
        missing = sorted(expected_entries - actual_entries)
        unexpected = sorted(actual_entries - expected_entries)
        pieces: list[str] = []
        if missing:
            pieces.append(f"missing named SafeTensors entries {missing!r}")
        if unexpected:
            pieces.append(f"unexpected named SafeTensors entries {unexpected!r}")
        raise MaterializationError("source SafeTensors entry set does not match manifest; " + "; ".join(pieces))

    for key, (dtype, shape) in expected.items():
        if dtype == np.dtype(np.int8):
            _array(tensors[f"quantized.{key}"], name=f"quantized.{key}", dtype=np.dtype(np.int8), shape=shape)
            scale = _array(tensors[f"scale.{key}"], name=f"scale.{key}", dtype=np.dtype(np.float32), shape=(shape[0],))
            if not np.all(np.isfinite(scale)) or not np.all(scale > 0):
                raise MaterializationError(f"SafeTensors entry scale.{key!r} must be finite and positive")
            if source_format == VOLVOX_TRAINED_SOURCE_FORMAT:
                zero_point = _array(
                    tensors[f"quantized.{key}.zero_point"],
                    name=f"quantized.{key}.zero_point",
                    dtype=np.dtype(np.int8),
                    shape=(shape[0],),
                )
                if np.any(zero_point != 0):
                    raise MaterializationError(
                        f"SafeTensors entry quantized.{key!r}.zero_point must be exactly zero"
                    )
        else:
            value = _array(tensors[f"float32.{key}"], name=f"float32.{key}", dtype=np.dtype(np.float32), shape=shape)
            if not np.all(np.isfinite(value)):
                raise MaterializationError(f"SafeTensors entry float32.{key!r} must be finite")

    token_values = tensors["quantized.tok.weight"]
    head_values = tensors["quantized.head.weight"]
    token_scales = tensors["scale.tok.weight"]
    head_scales = tensors["scale.head.weight"]
    if not np.array_equal(token_values, head_values) or not np.array_equal(token_scales, head_scales):
        raise MaterializationError(
            "source tied tok.weight/head.weight tensors or per-row scales do not match exactly"
        )
    if source_format == VOLVOX_TRAINED_SOURCE_FORMAT and not np.array_equal(
            tensors["quantized.tok.weight.zero_point"],
            tensors["quantized.head.weight.zero_point"]):
        raise MaterializationError(
            "source tied tok.weight/head.weight zero points do not match exactly"
        )
    return SourceRelease(
        manifest_path=manifest_path,
        weights_path=weights_path,
        manifest=manifest,
        dimensions=dimensions,
        tensors=tensors,
    )


def round_ties_to_even_i32(values: np.ndarray, label: str) -> np.ndarray:
    """Round finite F32-domain values with IEEE nearest-even semantics into I32."""
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise MaterializationError(f"{label} contains a non-finite value before I32 rounding")
    # NumPy rint is nearest-even.  Keep the range check before astype so a
    # NumPy cast can never wrap an out-of-range accumulator bias.
    rounded = np.rint(values)
    if np.any(rounded < INT32_MIN) or np.any(rounded > INT32_MAX):
        raise MaterializationError(f"{label} cannot be represented as I32 accumulator bias")
    return np.ascontiguousarray(rounded.astype(np.int32))


def quantize_symmetric_i8(values: np.ndarray, activation_scale: np.float32) -> tuple[np.ndarray, int]:
    """Quantize F32 constants with ties-to-even rounding and report saturation."""
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise MaterializationError("F32 constant contains a non-finite value")
    scale = _require_finite_positive_f32(float(activation_scale), "activation scale")
    scaled = np.asarray(values / scale, dtype=np.float32)
    rounded = np.rint(scaled)
    saturation_count = int(np.count_nonzero((rounded < -128) | (rounded > 127)))
    return np.ascontiguousarray(np.clip(rounded, -128, 127).astype(np.int8)), saturation_count


def derive_static_constant_i8_scale(values: np.ndarray, *, headroom: float = 1.01) -> tuple[np.float32, float]:
    """Derive a deterministic tight symmetric scale for an immutable F32 constant."""
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise MaterializationError("F32 constant contains a non-finite value")
    if not math.isfinite(headroom) or headroom <= 1.0:
        raise MaterializationError("static constant headroom must be finite and greater than one")
    maximum = float(np.max(np.abs(values))) if values.size else 0.0
    if maximum == 0.0:
        return np.float32(1.0), maximum
    raw = np.float32(np.float32(maximum) * np.float32(headroom) / np.float32(127.0))
    # Preserve the requested strict headroom after F32 serialization and JSON
    # round trips.  This is a parameter constant, not activation calibration.
    scale = np.nextafter(raw, np.float32(math.inf), dtype=np.float32)
    return _require_finite_positive_f32(scale, "derived static constant scale"), maximum


def transpose_conv_oihw_to_ohwi(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.dtype(np.int8) or values.ndim != 4:
        raise MaterializationError("Conv transform requires rank-4 I8 OIHW source storage")
    return np.ascontiguousarray(np.transpose(values, (0, 2, 3, 1)))


def split_mha_in_proj(
    weight: np.ndarray, bias: np.ndarray, scales: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split PyTorch [Q;K;V] packed MHA rows without changing raw bytes."""
    if weight.dtype != np.dtype(np.int8) or weight.ndim != 2 or weight.shape[0] % 3 != 0:
        raise MaterializationError("MHA in_proj_weight must be I8 [3*d_model,d_model]")
    d_model = weight.shape[0] // 3
    if bias.dtype != np.dtype(np.float32) or tuple(bias.shape) != (3 * d_model,):
        raise MaterializationError("MHA in_proj_bias must be F32 [3*d_model]")
    if scales.dtype != np.dtype(np.float32) or tuple(scales.shape) != (3 * d_model,):
        raise MaterializationError("MHA in_proj scales must be F32 [3*d_model]")
    return tuple(
        (
            np.ascontiguousarray(weight[offset:offset + d_model]),
            np.ascontiguousarray(bias[offset:offset + d_model]),
            np.ascontiguousarray(scales[offset:offset + d_model]),
        )
        for offset in (0, d_model, 2 * d_model)
    )  # type: ignore[return-value]


class WeightBuilder:
    """Build transformed package weights and their Volvox quantization metadata."""

    def __init__(self, release: SourceRelease):
        self.release = release
        self.values: dict[str, np.ndarray] = {}
        self.quantization: dict[str, dict[str, Any]] = {}
        self.constant_saturation: dict[str, int] = {}
        self.constant_quantization: dict[str, dict[str, Any]] = {}
        self._quantization_parameters: dict[str, str] = {}

    def _insert(self, name: str, value: np.ndarray) -> None:
        if name in self.values:
            parameter_owner = self._quantization_parameters.get(name)
            if parameter_owner is not None:
                raise MaterializationError(
                    f"materialized tensor {name!r} collides with a quantization parameter for {parameter_owner!r}"
                )
            raise AssertionError(f"duplicate materialized weight {name!r}")
        self.values[name] = np.ascontiguousarray(value)

    def _insert_quantization_parameter(
        self, name: str, kind: str, values: np.ndarray
    ) -> str:
        parameter = f"{name}_{kind}"
        if parameter in self.values:
            raise MaterializationError(
                f"quantization parameter {parameter!r} for {name!r} collides with a materialized tensor"
            )
        self.values[parameter] = np.ascontiguousarray(values)
        self._quantization_parameters[parameter] = name
        return parameter

    def _register_per_axis_quantization(self, name: str, scales: np.ndarray) -> None:
        if name in self.quantization:
            raise AssertionError(f"duplicate quantization descriptor {name!r}")
        array = np.ascontiguousarray(np.asarray(scales, dtype=np.float32))
        if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)) or not np.all(array > 0):
            raise MaterializationError(f"materialized scales for {name!r} are not finite positive rank-1 F32")
        weight = self.values.get(name)
        if (
            weight is None or weight.dtype != np.dtype(np.int8) or weight.ndim == 0 or
            int(array.size) != int(weight.shape[0])
        ):
            raise MaterializationError(
                f"materialized scales for {name!r} do not match an I8 tensor's axis-0 size"
            )
        scale_tensor = self._insert_quantization_parameter(name, "scale", array)
        zero_point_tensor = self._insert_quantization_parameter(
            name, "zero_point", np.zeros(array.shape, dtype=np.int8)
        )
        self.quantization[name] = {
            "scheme": "per_axis",
            "axis": 0,
            "scale_tensor": scale_tensor,
            "zero_point_tensor": zero_point_tensor,
        }

    def _register_per_tensor_quantization(self, name: str, scale: np.float32 | float) -> None:
        if name in self.quantization:
            raise AssertionError(f"duplicate quantization descriptor {name!r}")
        weight = self.values.get(name)
        if weight is None or weight.dtype != np.dtype(np.int8):
            raise MaterializationError(f"materialized per-tensor scale for {name!r} requires an I8 tensor")
        value = _require_finite_positive_f32(scale, f"materialized scale for {name}")
        scale_tensor = self._insert_quantization_parameter(
            name, "scale", np.asarray([value], dtype=np.float32)
        )
        zero_point_tensor = self._insert_quantization_parameter(
            name, "zero_point", np.zeros((1,), dtype=np.int8)
        )
        self.quantization[name] = {
            "scheme": "per_tensor",
            "scale_tensor": scale_tensor,
            "zero_point_tensor": zero_point_tensor,
        }

    def weight_scale_tensor(self, name: str) -> str:
        descriptor = self.quantization.get(name)
        scale_tensor = descriptor.get("scale_tensor") if descriptor else None
        if not isinstance(scale_tensor, str):
            raise AssertionError(f"quantized tensor {name!r} has no registered scale tensor")
        return scale_tensor

    def add_activation_quantization(
        self,
        namespace: str,
        name: str,
        scale: np.float32 | float,
        *,
        scale_tensor: str | None = None,
    ) -> dict[str, Any]:
        value = _require_finite_positive_f32(
            scale, f"activation scale for {namespace}:{name}"
        )
        expected = np.asarray([value], dtype=np.float32)
        if scale_tensor is not None:
            stored = self.values.get(scale_tensor)
            if stored is None or not np.array_equal(stored, expected):
                raise MaterializationError(
                    f"explicit scale tensor {scale_tensor!r} disagrees with {name!r}"
                )
        else:
            digest = hashlib.sha256(
                f"{namespace}\0{name}".encode("utf-8")
            ).hexdigest()[:24]
            base = f"__quant__.{digest}"
            scale_tensor = self._insert_quantization_parameter(
                base, "scale", expected
            )
        digest = hashlib.sha256(
            f"{namespace}\0{name}\0zero_point".encode("utf-8")
        ).hexdigest()[:24]
        zero_point_tensor = self._insert_quantization_parameter(
            f"__quant__.{digest}", "zero_point", np.zeros((1,), dtype=np.int8)
        )
        return {
            "scheme": "per_tensor",
            "scale_tensor": scale_tensor,
            "zero_point_tensor": zero_point_tensor,
        }

    def source_q(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        values = self.release.tensors[f"quantized.{key}"]
        scales = self.release.tensors[f"scale.{key}"]
        return values, scales

    def source_f32(self, key: str) -> np.ndarray:
        return self.release.tensors[f"float32.{key}"]

    def add_f32(self, name: str, key: str) -> str:
        self._insert(name, np.array(self.source_f32(key), dtype=np.float32, copy=True, order="C"))
        return name

    def add_literal_f32(self, name: str, values: np.ndarray | Sequence[float]) -> str:
        array = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
        if not np.all(np.isfinite(array)):
            raise MaterializationError(f"F32 literal {name!r} contains a non-finite value")
        self._insert(name, array)
        return name

    def add_shared_literal_f32(self, name: str, values: np.ndarray | Sequence[float]) -> str:
        """Insert one reusable F32 tensor, requiring exact agreement on reuse."""
        array = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
        if not np.all(np.isfinite(array)):
            raise MaterializationError(f"F32 literal {name!r} contains a non-finite value")
        existing = self.values.get(name)
        if existing is not None:
            if existing.dtype != np.dtype(np.float32) or not np.array_equal(existing, array):
                raise MaterializationError(f"shared F32 tensor {name!r} was requested with incompatible values")
            return name
        self._insert(name, array)
        return name

    def add_weight_scale(self, weight: QuantizedWeight) -> str:
        """Publish the source per-output scale as an explicit W8A32 input."""
        scale_tensor = self.weight_scale_tensor(weight.name)
        if not np.array_equal(self.values[scale_tensor], np.asarray(weight.scales, dtype=np.float32)):
            raise MaterializationError(f"scale tensor for {weight.name!r} disagrees with its W8A32 scales")
        return scale_tensor

    def add_dequantized_embedding(self, name: str, weight: QuantizedWeight) -> str:
        """Materialize the tied token table for the F32 Embedding operator."""
        values = self.values.get(weight.name)
        if values is None or values.dtype != np.dtype(np.int8) or values.ndim != 2:
            raise MaterializationError(f"embedding source {weight.name!r} is not a rank-2 I8 tensor")
        if weight.scales.shape != (values.shape[0],):
            raise MaterializationError(f"embedding source {weight.name!r} has incompatible per-row scales")
        dequantized = np.asarray(values, dtype=np.float32) * weight.scales[:, np.newaxis]
        return self.add_shared_literal_f32(name, dequantized)

    def add_exact_f32_bias(
        self, name: str, source: np.ndarray | None, weight: QuantizedWeight
    ) -> str:
        """Publish an exact source bias for a weight-only W8A32 operator."""
        output_channels = weight.shape[0]
        values = (
            np.zeros((output_channels,), dtype=np.float32)
            if source is None else np.asarray(source, dtype=np.float32)
        )
        if tuple(values.shape) != (output_channels,):
            raise MaterializationError(
                f"source bias for {name!r} has shape {list(values.shape)}, "
                f"expected [{output_channels}]"
            )
        return self.add_shared_literal_f32(name, values)

    def add_q(
        self,
        name: str,
        key: str,
        *,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        rows: slice | None = None,
        scale_multiplier: np.float32 | float = 1.0,
    ) -> QuantizedWeight:
        values, scales = self.source_q(key)
        if rows is not None:
            values = np.ascontiguousarray(values[rows])
            scales = np.ascontiguousarray(scales[rows])
        if transform is not None:
            values = transform(values)
        else:
            values = np.array(values, dtype=np.int8, copy=True, order="C")
        multiplier = _require_finite_positive_f32(float(scale_multiplier), f"scale multiplier for {name}")
        output_scales = np.asarray(scales * multiplier, dtype=np.float32)
        if not np.all(np.isfinite(output_scales)) or not np.all(output_scales > 0):
            raise MaterializationError(f"materialized scales for {name!r} are not finite positive F32")
        self._insert(name, values)
        self._register_per_axis_quantization(name, output_scales)
        return QuantizedWeight(name=name, shape=tuple(values.shape), scales=output_scales)

    def add_split_mha(self, prefix: str, source_prefix: str) -> tuple[tuple[QuantizedWeight, str], ...]:
        raw_weight, raw_scales = self.source_q(f"{source_prefix}.in_proj_weight")
        raw_bias = self.source_f32(f"{source_prefix}.in_proj_bias")
        qkv = split_mha_in_proj(raw_weight, raw_bias, raw_scales)
        result: list[tuple[QuantizedWeight, str]] = []
        for label, (weight, bias, scales) in zip(("q", "k", "v"), qkv):
            weight_name = f"w.{prefix}.{label}.weight"
            self._insert(weight_name, weight)
            self._register_per_axis_quantization(weight_name, scales)
            weight_spec = QuantizedWeight(weight_name, tuple(weight.shape), scales)
            # The caller creates the input-scale-specific I32 bias after it
            # has assigned this projection's logical activation edge.
            bias_name = f"b.{prefix}.{label}.i32"
            result.append((weight_spec, bias_name))
        return tuple(result)

    def add_accumulator_bias(
        self,
        name: str,
        source: np.ndarray | None,
        weight: QuantizedWeight,
        input_scale: np.float32 | float,
    ) -> str:
        output_channels = weight.shape[0]
        if source is None:
            values = np.zeros((output_channels,), dtype=np.float32)
        else:
            values = np.asarray(source, dtype=np.float32)
            if tuple(values.shape) != (output_channels,):
                raise MaterializationError(
                    f"source bias for {name!r} has shape {list(values.shape)}, expected [{output_channels}]"
                )
        activation_scale = _require_finite_positive_f32(input_scale, f"input activation scale for {name}")
        denominator = np.asarray(activation_scale * weight.scales, dtype=np.float32)
        if not np.all(np.isfinite(denominator)) or not np.all(denominator > 0):
            raise MaterializationError(f"I32 accumulator scale for {name!r} is not finite positive F32")
        bias = round_ties_to_even_i32(np.asarray(values / denominator, dtype=np.float32), name)
        existing = self.values.get(name)
        if existing is not None:
            if existing.dtype != np.dtype(np.int32) or not np.array_equal(existing, bias):
                raise MaterializationError(
                    f"materialized I32 bias {name!r} was requested with incompatible input-scale semantics"
                )
            return name
        self._insert(name, bias)
        self._assert_i32_accumulator_safe(weight, bias, name)
        return name

    def _assert_i32_accumulator_safe(self, weight: QuantizedWeight, bias: np.ndarray, label: str) -> None:
        # All W8A8 graph activations use signed zero-point-zero I8.  Use the
        # full representable domain, matching backend preflight rather than
        # trusting only the current raw source values.
        if len(weight.shape) == 2:
            terms = weight.shape[1]
        elif len(weight.shape) == 4:
            terms = weight.shape[1] * weight.shape[2] * weight.shape[3]
        else:
            raise MaterializationError(f"cannot establish accumulator bound for {label!r}")
        bound = 128 * 128 * int(terms)
        if np.any(np.abs(bias.astype(np.int64)) + bound > INT32_MAX):
            raise MaterializationError(
                f"{label} may overflow the canonical I32 accumulator for its selected input activation scale"
            )

    def add_quantized_constant(self, name: str, values: np.ndarray) -> str:
        scale, maximum = derive_static_constant_i8_scale(values)
        quantized, saturated = quantize_symmetric_i8(values, scale)
        self._insert(name, quantized)
        self._register_per_tensor_quantization(name, scale)
        self.constant_saturation[name] = saturated
        self.constant_quantization[name] = {
            "origin": "static_constant_derived",
            "source_abs_max": maximum,
            "headroom_factor": 1.01,
            "scale": float(scale),
            "zero_point": 0,
            "saturation_count": saturated,
        }
        return name


def _activation_quantization(scale: np.float32) -> dict[str, Any]:
    return {"scheme": "per_tensor", "scale": float(scale), "zero_point": ACTIVATION_ZERO_POINT}


class GraphDocumentBuilder:
    """Small declarative helper for strict typed Volvox graph document nodes."""

    def __init__(self, activation_profile: ActivationProfile, *, bias_scope: str):
        self.activation_profile = activation_profile
        self.bias_scope = bias_scope
        self.nodes: list[dict[str, Any]] = []
        self._serial = 0
        self.tensor_scales: dict[str, np.float32] = {}
        self.tensor_logical_edges: dict[str, str] = {}
        self.edge_bindings: list[dict[str, Any]] = []

    def _id(self, prefix: str) -> str:
        identifier = f"{self._serial:04d}_{prefix}"
        self._serial += 1
        return identifier

    def typed(
        self,
        op_type: str,
        prefix: str,
        inputs: Mapping[str, str],
        output: str,
        shape: Sequence[int],
        params: Mapping[str, Any] | None = None,
        *,
        scale_key: str | None = None,
        source_probe: str | None = None,
        source_multiplier: float = 1.0,
    ) -> str:
        logical_edge = scale_key or prefix
        output_scale = self.activation_profile.scale_for(
            logical_edge, source_probe=source_probe, multiplier=source_multiplier
        )
        self.tensor_scales[output] = output_scale
        self.tensor_logical_edges[output] = logical_edge
        self.edge_bindings.append({
            "tensor": output,
            "logical_edge": logical_edge,
            "source_probe": source_probe,
            "source_multiplier": float(_require_finite_positive_f32(source_multiplier, f"binding multiplier for {logical_edge}")),
            "kind": "operator_output",
        })
        self.nodes.append({
            "id": self._id(prefix),
            "opType": op_type,
            "inputs": dict(inputs),
            "outputs": {"out": output},
            "outputs_shape": {"out": list(shape)},
            "outputs_dtype": {"out": ACTIVATION_DTYPE},
            "params": dict(params or {}),
        })
        return output

    def register_input(
        self, name: str, logical_edge: str, *, source_probe: str | None = None
    ) -> dict[str, Any]:
        scale = self.activation_profile.scale_for(logical_edge, source_probe=source_probe)
        self.tensor_scales[name] = scale
        self.tensor_logical_edges[name] = logical_edge
        self.edge_bindings.append({
            "tensor": name,
            "logical_edge": logical_edge,
            "source_probe": source_probe,
            "source_multiplier": 1.0,
            "kind": "graph_input",
        })
        return _activation_quantization(scale)

    def register_static(
        self, name: str, logical_edge: str, *, source_probe: str | None = None
    ) -> np.float32:
        scale = self.activation_profile.scale_for(logical_edge, source_probe=source_probe)
        self.tensor_scales[name] = scale
        self.tensor_logical_edges[name] = logical_edge
        self.edge_bindings.append({
            "tensor": name,
            "logical_edge": logical_edge,
            "source_probe": source_probe,
            "source_multiplier": 1.0,
            "kind": "static_constant",
        })
        return scale

    def register_derived_static(self, name: str, logical_edge: str, scale: np.float32 | float) -> np.float32:
        """Register a pre-quantized immutable parameter without calibration lookup."""
        resolved = _require_finite_positive_f32(scale, f"derived static scale for {name}")
        self.tensor_scales[name] = resolved
        self.tensor_logical_edges[name] = logical_edge
        self.edge_bindings.append({
            "tensor": name,
            "logical_edge": logical_edge,
            "source_probe": None,
            "source_multiplier": 1.0,
            "kind": "static_constant_derived",
        })
        return resolved

    def scale_of(self, name: str) -> np.float32:
        try:
            return self.tensor_scales[name]
        except KeyError as error:
            raise MaterializationError(f"internal materializer missing activation scale for tensor {name!r}") from error

    def shape_copy(
        self,
        op_type: str,
        prefix: str,
        input_name: str,
        output: str,
        shape: Sequence[int],
    ) -> str:
        """Emit a representation-preserving typed shape operation.

        Shape operations must preserve the exact descriptor, so they do not
        ask the profile for a fresh edge scale.
        """
        scale = self.scale_of(input_name)
        self.tensor_scales[output] = scale
        logical_edge = self.tensor_logical_edges[input_name]
        self.tensor_logical_edges[output] = logical_edge
        self.edge_bindings.append({
            "tensor": output,
            "logical_edge": logical_edge,
            "source_probe": None,
            "source_multiplier": 1.0,
            "kind": "descriptor_preserving_shape_copy",
            "source_tensor": input_name,
        })
        self.nodes.append({
            "id": self._id(prefix),
            "opType": op_type,
            "inputs": {"input": input_name},
            "outputs": {"out": output},
            "outputs_shape": {"out": list(shape)},
            "outputs_dtype": {"out": ACTIVATION_DTYPE},
            "params": {},
        })
        return output

    def concat(
        self,
        prefix: str,
        inputs: Sequence[str],
        output: str,
        shape: Sequence[int],
        axis: int,
    ) -> str:
        if not inputs:
            raise AssertionError("Concat needs at least one input")
        scale = self.scale_of(inputs[0])
        if any(self.scale_of(name) != scale for name in inputs[1:]):
            raise MaterializationError(
                f"typed Concat {prefix!r} requires matching calibrated scales for its inputs; "
                "use one logical edge scale for the concat domain"
            )
        self.tensor_scales[output] = scale
        logical_edge = self.tensor_logical_edges[inputs[0]]
        self.tensor_logical_edges[output] = logical_edge
        self.edge_bindings.append({
            "tensor": output,
            "logical_edge": logical_edge,
            "source_probe": None,
            "source_multiplier": 1.0,
            "kind": "descriptor_preserving_concat",
            "source_tensors": list(inputs),
        })
        self.nodes.append({
            "id": self._id(prefix),
            "opType": "Concat",
            "inputs": {f"input{index}": name for index, name in enumerate(inputs)},
            "outputs": {"out": output},
            "outputs_shape": {"out": list(shape)},
            "outputs_dtype": {"out": ACTIVATION_DTYPE},
            "params": {"axis": axis},
        })
        return output

    def annotate_binding(self, tensor: str, **details: Any) -> None:
        for binding in reversed(self.edge_bindings):
            if binding.get("tensor") == tensor:
                binding.update(details)
                return
        raise AssertionError(f"missing activation binding for tensor {tensor!r}")

    def i32(
        self,
        op_type: str,
        prefix: str,
        inputs: Mapping[str, str],
        output: str,
        shape: Sequence[int],
        params: Mapping[str, Any] | None = None,
    ) -> str:
        self.nodes.append({
            "id": self._id(prefix),
            "opType": op_type,
            "inputs": dict(inputs),
            "outputs": {"out": output},
            "outputs_shape": {"out": list(shape)},
            "outputs_dtype": {"out": "int32"},
            "params": dict(params or {}),
        })
        return output

    def qlinear(
        self,
        prefix: str,
        input_name: str,
        input_shape: Sequence[int],
        weight: QuantizedWeight,
        bias: str,
        *,
        scale_key: str | None = None,
        source_probe: str | None = None,
        source_multiplier: float = 1.0,
    ) -> str:
        if len(input_shape) < 1 or input_shape[-1] != weight.shape[1]:
            raise AssertionError(f"bad QLinear shape for {prefix}")
        output = f"a.{prefix}"
        return self.typed(
            "QLinear", prefix, {"input": input_name, "weight": weight.name, "bias": bias}, output,
            [*input_shape[:-1], weight.shape[0]], {}, scale_key=scale_key,
            source_probe=source_probe, source_multiplier=source_multiplier,
        )

    def qadd(
        self,
        prefix: str,
        left: str,
        right: str,
        shape: Sequence[int],
        *,
        scale_key: str | None = None,
        source_probe: str | None = None,
    ) -> str:
        return self.typed(
            "QAdd", prefix, {"a": left, "b": right}, f"a.{prefix}", shape, {"relu": 0}, scale_key=scale_key,
            source_probe=source_probe,
        )

    def qgelu(
        self,
        prefix: str,
        input_name: str,
        shape: Sequence[int],
        *,
        scale_key: str | None = None,
        source_probe: str | None = None,
    ) -> str:
        return self.typed(
            "QGELU", prefix, {"input": input_name}, f"a.{prefix}", shape, {"approximate": "none"}, scale_key=scale_key,
            source_probe=source_probe,
        )

    def qlayernorm(
        self,
        prefix: str,
        input_name: str,
        shape: Sequence[int],
        weight: str,
        bias: str,
        d_model: int,
        *,
        scale_key: str | None = None,
        source_probe: str | None = None,
    ) -> str:
        return self.typed(
            "QLayerNorm", prefix, {"input": input_name, "weight": weight, "bias": bias}, f"a.{prefix}", shape,
            {"eps": 1e-5, "d_model": d_model}, scale_key=scale_key, source_probe=source_probe,
        )

    def qsdpa(
        self,
        prefix: str,
        q: str,
        k: str,
        v: str,
        shape: Sequence[int],
        mask: str,
        heads: int,
        causal: bool,
        head_dim: int,
        *,
        scale_key: str | None = None,
        source_probe: str | None = None,
        cross: bool = False,
    ) -> str:
        del cross
        return self.typed(
            "QSDPA", prefix, {"q": q, "k": k, "v": v, "mask": mask}, f"a.{prefix}", shape,
            {"heads": heads, "causal": causal, "scale": float(np.float32(1.0 / math.sqrt(head_dim)))},
            scale_key=scale_key, source_probe=source_probe,
        )


class W8A32GraphDocumentBuilder:
    """Emit F32-activation family graphs backed by shared I8 weights.

    The router deliberately remains a calibrated W8A8 graph.  This builder is
    used only for explicit-family graphs and therefore has no activation-scale
    bindings.  Every dense/conv node receives the source per-output weight
    scale as a concrete tensor input, and every bias is retained as exact F32.
    """

    family_execution = "w8a32"

    def __init__(self, weights: WeightBuilder, *, bias_scope: str):
        self.weights = weights
        self.bias_scope = bias_scope
        self.nodes: list[dict[str, Any]] = []
        self.edge_bindings: list[dict[str, Any]] = []
        self._serial = 0
        self._aliases: dict[str, str] = {}
        self._embedding_aliases: dict[str, str] = {}

    def _id(self, prefix: str) -> str:
        identifier = f"{self._serial:04d}_{prefix}"
        self._serial += 1
        return identifier

    def _resolve(self, name: str) -> str:
        return self._aliases.get(name, name)

    def alias_weight(self, source: str, target: str) -> None:
        existing = self._aliases.get(source)
        if existing is not None and existing != target:
            raise MaterializationError(
                f"W8A32 tensor alias {source!r} conflicts: {existing!r} versus {target!r}"
            )
        self._aliases[source] = target

    def alias_embedding(self, source: str, target: str) -> None:
        self._embedding_aliases[source] = target

    def typed(
        self,
        op_type: str,
        prefix: str,
        inputs: Mapping[str, str],
        output: str,
        shape: Sequence[int],
        params: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> str:
        mapped = {
            "QAdd": "Add",
            "QEmbedding": "Embedding",
            "QGELU": "GELU",
            "QGroupNorm": "GroupNorm",
            "QLayerNorm": "LayerNorm",
            "QSiLU": "SiLU",
        }.get(op_type, op_type)
        resolved_inputs = {key: self._resolve(name) for key, name in inputs.items()}
        if op_type == "QEmbedding" and "weight" in inputs:
            resolved_inputs["weight"] = self._embedding_aliases.get(
                inputs["weight"], resolved_inputs["weight"]
            )
        self.nodes.append({
            "id": self._id(prefix),
            "opType": mapped,
            "inputs": resolved_inputs,
            "outputs": {"out": output},
            "outputs_shape": {"out": list(shape)},
            "outputs_dtype": {"out": "float32"},
            "params": dict(params or {}),
        })
        return output

    def shape_copy(
        self,
        op_type: str,
        prefix: str,
        input_name: str,
        output: str,
        shape: Sequence[int],
    ) -> str:
        return self.typed(op_type, prefix, {"input": input_name}, output, shape)

    def concat(
        self,
        prefix: str,
        inputs: Sequence[str],
        output: str,
        shape: Sequence[int],
        axis: int,
    ) -> str:
        if not inputs:
            raise AssertionError("Concat needs at least one input")
        return self.typed(
            "Concat", prefix, {f"input{index}": name for index, name in enumerate(inputs)},
            output, shape, {"axis": axis},
        )

    def annotate_binding(self, tensor: str, **details: Any) -> None:
        del tensor, details

    def i32(
        self,
        op_type: str,
        prefix: str,
        inputs: Mapping[str, str],
        output: str,
        shape: Sequence[int],
        params: Mapping[str, Any] | None = None,
    ) -> str:
        self.nodes.append({
            "id": self._id(prefix),
            "opType": op_type,
            "inputs": {key: self._resolve(name) for key, name in inputs.items()},
            "outputs": {"out": output},
            "outputs_shape": {"out": list(shape)},
            "outputs_dtype": {"out": "int32"},
            "params": dict(params or {}),
        })
        return output

    def qlinear(
        self,
        prefix: str,
        input_name: str,
        input_shape: Sequence[int],
        weight: QuantizedWeight,
        bias: str,
        **_: Any,
    ) -> str:
        if len(input_shape) < 1 or input_shape[-1] != weight.shape[1]:
            raise AssertionError(f"bad W8A32 Linear shape for {prefix}")
        return self.typed(
            "Linear", prefix,
            {
                "input": input_name,
                "weight": weight.name,
                "weight_scale": self.weights.add_weight_scale(weight),
                "bias": bias,
            },
            f"a.{prefix}", [*input_shape[:-1], weight.shape[0]],
            {"weight_layout": "OUT_IN"},
        )

    def qadd(self, prefix: str, left: str, right: str, shape: Sequence[int], **_: Any) -> str:
        return self.typed("Add", prefix, {"a": left, "b": right}, f"a.{prefix}", shape, {"relu": 0})

    def qgelu(self, prefix: str, input_name: str, shape: Sequence[int], **_: Any) -> str:
        return self.typed(
            "GELU", prefix, {"input": input_name}, f"a.{prefix}", shape,
            {"approximate": "none"},
        )

    def qlayernorm(
        self,
        prefix: str,
        input_name: str,
        shape: Sequence[int],
        weight: str,
        bias: str,
        d_model: int,
        **_: Any,
    ) -> str:
        return self.typed(
            "LayerNorm", prefix,
            {"input": input_name, "weight": weight, "bias": bias},
            f"a.{prefix}", shape, {"eps": 1e-5, "d_model": d_model},
        )

    def qsdpa(
        self,
        prefix: str,
        q: str,
        k: str,
        v: str,
        shape: Sequence[int],
        mask: str,
        heads: int,
        causal: bool,
        head_dim: int,
        *,
        cross: bool = False,
        **_: Any,
    ) -> str:
        params = {
            "heads": heads,
            "causal": causal,
            "scale": float(np.float32(1.0 / math.sqrt(head_dim))),
        }
        # Keep separate projections for both self and cross attention.  Native
        # incremental SDPA currently rejects a mask in active-row mode, while
        # CrossSDPA supports the TinyReceipt [B,K] keep mask for causal rows.
        del cross
        return self.typed(
            "CrossSDPA", prefix, {"q": q, "k": k, "v": v, "mask": mask},
            f"a.{prefix}", shape, params,
        )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError) as error:
        raise MaterializationError(f"could not write {path}: {error}") from error


class TinyReceiptMaterializer:
    """Translate the fixed source architecture into reusable typed graph documents."""

    def __init__(self, release: SourceRelease, activation_profile: ActivationProfile):
        self.release = release
        self.dimensions = release.dimensions
        self.profile = activation_profile
        self.weights = WeightBuilder(release)
        self.qweights: dict[str, QuantizedWeight] = {}
        self.fweights: dict[str, str] = {}
        self.mha: dict[tuple[str, str], tuple[QuantizedWeight, np.ndarray]] = {}

    def _q(self, source_key: str) -> QuantizedWeight:
        try:
            return self.qweights[source_key]
        except KeyError as error:
            raise AssertionError(f"materializer omitted quantized source weight {source_key!r}") from error

    def _f(self, source_key: str) -> str:
        try:
            return self.fweights[source_key]
        except KeyError as error:
            raise AssertionError(f"materializer omitted F32 source weight {source_key!r}") from error

    def _source_bias(self, source_key: str) -> np.ndarray:
        return self.weights.source_f32(source_key)

    def _add_q_source(
        self,
        source_key: str,
        *,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        scale_multiplier: np.float32 | float = 1.0,
    ) -> QuantizedWeight:
        name = f"w.{source_key}"
        result = self.weights.add_q(
            name, source_key, transform=transform, scale_multiplier=scale_multiplier
        )
        self.qweights[source_key] = result
        return result

    def _add_f_source(self, source_key: str) -> str:
        name = self.weights.add_f32(f"f.{source_key}", source_key)
        self.fweights[source_key] = name
        return name

    def _prepare_mha(self, source_root: str) -> None:
        raw_weight, raw_scales = self.weights.source_q(f"{source_root}.in_proj_weight")
        raw_bias = self.weights.source_f32(f"{source_root}.in_proj_bias")
        for label, (weight, bias, scales) in zip(("q", "k", "v"), split_mha_in_proj(raw_weight, raw_bias, raw_scales)):
            name = f"w.{source_root}.{label}.weight"
            self.weights._insert(name, weight)
            self.weights._register_per_axis_quantization(name, scales)
            self.mha[(source_root, label)] = (
                QuantizedWeight(name=name, shape=tuple(weight.shape), scales=scales),
                bias,
            )

    def prepare_weights(self) -> None:
        """Copy/repack every source parameter once; no graph activations here."""
        dimensions = self.dimensions
        for spec in _stem_specs(dimensions):
            index = spec["index"]
            if spec["kind"] == "conv":
                self._add_q_source(
                    f"stem.{index}.net.{spec['conv']}.weight", transform=transpose_conv_oihw_to_ohwi
                )
                self._add_f_source(f"stem.{index}.net.{spec['norm']}.weight")
                self._add_f_source(f"stem.{index}.net.{spec['norm']}.bias")
            else:
                self._add_q_source(f"stem.{index}.net.0.weight", transform=transpose_conv_oihw_to_ohwi)
                self._add_f_source(f"stem.{index}.net.1.weight")
                self._add_f_source(f"stem.{index}.net.1.bias")
                self._add_q_source(f"stem.{index}.net.3.weight", transform=transpose_conv_oihw_to_ohwi)
                self._add_f_source(f"stem.{index}.net.4.weight")
                self._add_f_source(f"stem.{index}.net.4.bias")

        self._add_q_source("tok.weight")
        # head.weight is deliberately not copied: source validation proved it
        # tied byte-for-byte and scale-for-scale to tok.weight, and every
        # explicit graph references w.tok.weight for the output projection.
        for prefix, count, decoder in (
            ("encoder.layers", dimensions.enc_layers, False),
            ("decoder.layers", dimensions.dec_layers, True),
        ):
            for layer in range(count):
                root = f"{prefix}.{layer}"
                for attention in ["self_attn"] + (["multihead_attn"] if decoder else []):
                    attention_root = f"{root}.{attention}"
                    self._prepare_mha(attention_root)
                    self._add_q_source(f"{attention_root}.out_proj.weight")
                for linear in ("linear1", "linear2"):
                    self._add_q_source(f"{root}.{linear}.base.weight")
                    self._add_q_source(f"{root}.{linear}.lora_a.weight")
                    # Folding alpha/r into B's per-output-channel scales keeps
                    # the LoRA delta in byte storage without a scalar F32 op.
                    self._add_q_source(
                        f"{root}.{linear}.lora_b.weight", scale_multiplier=dimensions.lora_scale
                    )
                norm_count = 3 if decoder else 2
                for norm in range(1, norm_count + 1):
                    self._add_f_source(f"{root}.norm{norm}.weight")
                    self._add_f_source(f"{root}.norm{norm}.bias")

        self._add_q_source("router.net.0.weight")
        self._add_q_source("router.net.2.weight")
        for group in ("memory_adapters", "decoder_adapters"):
            for family in range(len(FAMILY_ORDER)):
                root = f"{group}.{family}"
                self._add_q_source(f"{root}.down.weight")
                self._add_q_source(f"{root}.up.weight")
        self._add_f_source("norm.weight")
        self._add_f_source("norm.bias")

        self.weights.add_quantized_constant(
            "c.img_pos", self.weights.source_f32("img_pos")
        )
        self.weights.add_quantized_constant(
            "c.img_type", np.repeat(self.weights.source_f32("type_img"), dimensions.img_tokens, axis=1),
        )
        self.weights.add_quantized_constant(
            "c.q_pos", self.weights.source_f32("q_pos")
        )
        self.weights.add_quantized_constant(
            "c.q_type", np.repeat(self.weights.source_f32("type_q"), dimensions.max_q_len, axis=1),
        )
        self.weights.add_quantized_constant(
            "c.y_pos", self.weights.source_f32("y_pos")
        )
        self.weights.add_literal_f32(
            "c.input_image_scale",
            np.asarray([self.profile.scale_for("input.image", source_probe="input.image_preprocessed")], dtype=np.float32),
        )

    def prepare_w8a32_family_weights(self) -> None:
        """Add exact F32 constants required only by W8A32 family graphs."""
        dimensions = self.dimensions
        self.weights.add_dequantized_embedding("f.tok.embedding", self._q("tok.weight"))
        self.weights.add_shared_literal_f32("c.img_pos.f32", self.weights.source_f32("img_pos"))
        self.weights.add_shared_literal_f32(
            "c.img_type.f32",
            np.repeat(self.weights.source_f32("type_img"), dimensions.img_tokens, axis=1),
        )
        self.weights.add_shared_literal_f32("c.q_pos.f32", self.weights.source_f32("q_pos"))
        self.weights.add_shared_literal_f32(
            "c.q_type.f32",
            np.repeat(self.weights.source_f32("type_q"), dimensions.max_q_len, axis=1),
        )
        self.weights.add_shared_literal_f32("c.y_pos.f32", self.weights.source_f32("y_pos"))

    def _linear(
        self,
        graph: GraphDocumentBuilder,
        prefix: str,
        input_name: str,
        input_shape: Sequence[int],
        weight: QuantizedWeight,
        source_bias: np.ndarray | None,
        *,
        scale_key: str | None = None,
        source_probe: str | None = None,
        source_multiplier: float = 1.0,
    ) -> str:
        if isinstance(graph, W8A32GraphDocumentBuilder):
            bias_name = self.weights.add_exact_f32_bias(
                f"b.f32.{weight.name.removeprefix('w.')}", source_bias, weight
            )
            return graph.qlinear(prefix, input_name, input_shape, weight, bias_name)
        bias_name = self.weights.add_accumulator_bias(
            f"b.{graph.bias_scope}.{prefix}.i32", source_bias, weight, graph.scale_of(input_name)
        )
        return graph.qlinear(
            prefix, input_name, input_shape, weight, bias_name, scale_key=scale_key,
            source_probe=source_probe, source_multiplier=source_multiplier,
        )

    def _mha_linear(
        self,
        graph: GraphDocumentBuilder,
        prefix: str,
        source_root: str,
        projection: str,
        input_name: str,
        input_shape: Sequence[int],
        *,
        scale_key: str | None = None,
    ) -> str:
        weight, bias = self.mha[(source_root, projection)]
        return self._linear(
            graph, prefix, input_name, input_shape, weight, bias, scale_key=scale_key,
            source_probe=f"{source_root}.{projection}",
        )

    def _conv(
        self,
        graph: GraphDocumentBuilder,
        prefix: str,
        input_name: str,
        input_shape: Sequence[int],
        source_weight: str,
        stride: int,
        *,
        scale_key: str | None = None,
        source_probe: str | None = None,
    ) -> tuple[str, list[int]]:
        weight = self._q(source_weight)
        batch, height, width, channels = input_shape
        out_channels, kernel_h, kernel_w, in_channels = weight.shape
        if channels != in_channels:
            raise AssertionError(f"bad Conv source dimensions at {prefix}")
        out_h = (height + 2 - kernel_h) // stride + 1
        out_w = (width + 2 - kernel_w) // stride + 1
        output_shape = [batch, out_h, out_w, out_channels]
        if isinstance(graph, W8A32GraphDocumentBuilder):
            bias = self.weights.add_exact_f32_bias(
                f"b.f32.{weight.name.removeprefix('w.')}", None, weight
            )
            output = graph.typed(
                "QConv2D", prefix,
                {
                    "input": input_name,
                    "weight": weight.name,
                    "weight_scale": self.weights.add_weight_scale(weight),
                    "bias": bias,
                },
                f"a.{prefix}", output_shape,
                {
                    "stride": [stride, stride],
                    "padding": [1, 1],
                    "pads": [1, 1, 1, 1],
                    "data_layout": "NHWC",
                    "weight_layout": "OHWI",
                    "weight_only": True,
                },
            )
            return output, output_shape
        bias = self.weights.add_accumulator_bias(
            f"b.{graph.bias_scope}.{prefix}.i32", None, weight, graph.scale_of(input_name)
        )
        output = graph.typed(
            "QConv2D", prefix,
            {"input": input_name, "weight": weight.name, "bias": bias}, f"a.{prefix}", output_shape,
            {
                "stride": [stride, stride],
                "padding": [1, 1],
                "pads": [1, 1, 1, 1],
                "data_layout": "NHWC",
                "weight_layout": "OHWI",
            },
            scale_key=scale_key, source_probe=source_probe,
        )
        return output, output_shape

    def _group_norm(
        self,
        graph: GraphDocumentBuilder,
        prefix: str,
        input_name: str,
        shape: Sequence[int],
        source_root: str,
        *,
        scale_key: str | None = None,
        source_probe: str | None = None,
    ) -> str:
        channels = int(shape[-1])
        return graph.typed(
            "QGroupNorm", prefix,
            {
                "input": input_name,
                "weight": self._f(f"{source_root}.weight"),
                "bias": self._f(f"{source_root}.bias"),
            },
            f"a.{prefix}", shape,
            {"num_groups": _groups(channels), "eps": 1e-5, "data_layout": "NHWC"},
            scale_key=scale_key, source_probe=source_probe,
        )

    def _qsilu(
        self,
        graph: GraphDocumentBuilder,
        prefix: str,
        input_name: str,
        shape: Sequence[int],
        *,
        scale_key: str | None = None,
        source_probe: str | None = None,
    ) -> str:
        return graph.typed(
            "QSiLU", prefix, {"input": input_name}, f"a.{prefix}", shape, {}, scale_key=scale_key,
            source_probe=source_probe,
        )

    def _ffn(
        self,
        graph: GraphDocumentBuilder,
        prefix: str,
        input_name: str,
        shape: Sequence[int],
        source_root: str,
        norm_index: int,
    ) -> str:
        d = self.dimensions.d_model
        normalized = graph.qlayernorm(
            f"{prefix}.norm{norm_index}", input_name, shape,
            self._f(f"{source_root}.norm{norm_index}.weight"),
            self._f(f"{source_root}.norm{norm_index}.bias"), d,
            source_probe=f"{source_root}.norm{norm_index}",
        )
        hidden_shape = [*shape[:-1], self.dimensions.ff_dim]
        base1 = self._linear(
            graph, f"{prefix}.linear1.base", normalized, shape,
            self._q(f"{source_root}.linear1.base.weight"),
            self._source_bias(f"{source_root}.linear1.base.bias"),
            source_probe=f"{source_root}.linear1.base",
        )
        low1 = self._linear(
            graph, f"{prefix}.linear1.lora_a", normalized, shape,
            self._q(f"{source_root}.linear1.lora_a.weight"), None,
            source_probe=f"{source_root}.linear1.lora_a",
        )
        delta1 = self._linear(
            graph, f"{prefix}.linear1.lora_b", low1, [*shape[:-1], self.dimensions.lora_r],
            self._q(f"{source_root}.linear1.lora_b.weight"), None,
            source_probe=f"{source_root}.linear1.lora_b", source_multiplier=float(self.dimensions.lora_scale),
        )
        # Source LoRALinear applies base + B(A(x))*alpha/r before the
        # Transformer activation.  B's scales were multiplied by alpha/r in
        # prepare_weights(), so this QAdd is the exact graph placement.
        combined1 = graph.qadd(
            f"{prefix}.linear1.sum", base1, delta1, hidden_shape,
            source_probe=f"{source_root}.linear1.combined_output"
        )
        activated = graph.qgelu(
            f"{prefix}.gelu", combined1, hidden_shape,
            source_probe=f"{source_root}.linear1.gelu_output"
        )
        base2 = self._linear(
            graph, f"{prefix}.linear2.base", activated, hidden_shape,
            self._q(f"{source_root}.linear2.base.weight"),
            self._source_bias(f"{source_root}.linear2.base.bias"),
            source_probe=f"{source_root}.linear2.base",
        )
        low2 = self._linear(
            graph, f"{prefix}.linear2.lora_a", activated, hidden_shape,
            self._q(f"{source_root}.linear2.lora_a.weight"), None,
            source_probe=f"{source_root}.linear2.lora_a",
        )
        delta2 = self._linear(
            graph, f"{prefix}.linear2.lora_b", low2, [*shape[:-1], self.dimensions.lora_r],
            self._q(f"{source_root}.linear2.lora_b.weight"), None,
            source_probe=f"{source_root}.linear2.lora_b", source_multiplier=float(self.dimensions.lora_scale),
        )
        combined2 = graph.qadd(
            f"{prefix}.linear2.sum", base2, delta2, shape,
            source_probe=f"{source_root}.linear2.combined_output"
        )
        return graph.qadd(f"{prefix}.residual", input_name, combined2, shape, source_probe=f"{source_root}.output")

    def _self_attention(
        self,
        graph: GraphDocumentBuilder,
        prefix: str,
        input_name: str,
        shape: Sequence[int],
        source_root: str,
        norm_index: int,
        mask: str,
        *,
        causal: bool,
    ) -> str:
        normalized = graph.qlayernorm(
            f"{prefix}.norm{norm_index}", input_name, shape,
            self._f(f"{source_root}.norm{norm_index}.weight"),
            self._f(f"{source_root}.norm{norm_index}.bias"), self.dimensions.d_model,
            source_probe=f"{source_root}.norm{norm_index}",
        )
        attention_root = f"{source_root}.self_attn"
        q = self._mha_linear(graph, f"{prefix}.q", attention_root, "q", normalized, shape)
        k = self._mha_linear(graph, f"{prefix}.k", attention_root, "k", normalized, shape)
        v = self._mha_linear(graph, f"{prefix}.v", attention_root, "v", normalized, shape)
        attended = graph.qsdpa(
            f"{prefix}.attention", q, k, v, shape, mask, self.dimensions.heads, causal, self.dimensions.head_dim,
            # Attention is a convex combination of V rows.  The MHA [0]
            # probe is after out_proj, so V is the conservative source-bound
            # for this pre-projection typed QSDPA output.
            source_probe=f"{attention_root}.v",
        )
        projected = self._linear(
            graph, f"{prefix}.out", attended, shape, self._q(f"{attention_root}.out_proj.weight"),
            self._source_bias(f"{attention_root}.out_proj.bias"),
            source_probe=f"{attention_root}[0]",
        )
        pre_ffn_probe = f"{source_root}.norm2.input" if source_root.startswith("encoder.") else f"{source_root}.norm2.input"
        return graph.qadd(f"{prefix}.residual", input_name, projected, shape, source_probe=pre_ffn_probe)

    def _cross_attention(
        self,
        graph: GraphDocumentBuilder,
        prefix: str,
        input_name: str,
        input_shape: Sequence[int],
        memory_name: str,
        memory_shape: Sequence[int],
        source_root: str,
        mask: str,
    ) -> str:
        normalized = graph.qlayernorm(
            f"{prefix}.norm2", input_name, input_shape,
            self._f(f"{source_root}.norm2.weight"),
            self._f(f"{source_root}.norm2.bias"), self.dimensions.d_model,
            source_probe=f"{source_root}.norm2",
        )
        attention_root = f"{source_root}.multihead_attn"
        q = self._mha_linear(graph, f"{prefix}.q", attention_root, "q", normalized, input_shape)
        k = self._mha_linear(graph, f"{prefix}.k", attention_root, "k", memory_name, memory_shape)
        v = self._mha_linear(graph, f"{prefix}.v", attention_root, "v", memory_name, memory_shape)
        attended = graph.qsdpa(
            f"{prefix}.attention", q, k, v, input_shape, mask, self.dimensions.heads, False, self.dimensions.head_dim,
            source_probe=f"{attention_root}.v", cross=True,
        )
        projected = self._linear(
            graph, f"{prefix}.out", attended, input_shape, self._q(f"{attention_root}.out_proj.weight"),
            self._source_bias(f"{attention_root}.out_proj.bias"),
            source_probe=f"{attention_root}[0]",
        )
        return graph.qadd(
            f"{prefix}.residual", input_name, projected, input_shape,
            source_probe=f"{source_root}.norm3.input",
        )

    def _adapter(
        self,
        graph: GraphDocumentBuilder,
        prefix: str,
        input_name: str,
        shape: Sequence[int],
        source_root: str,
        *,
        output_probe: str | None = None,
    ) -> str:
        down_shape = [*shape[:-1], self.dimensions.adapter_bottleneck]
        down = self._linear(
            graph, f"{prefix}.down", input_name, shape, self._q(f"{source_root}.down.weight"),
            self._source_bias(f"{source_root}.down.bias"),
            source_probe=f"{source_root}.down",
        )
        active = graph.qgelu(f"{prefix}.gelu", down, down_shape, source_probe=f"{source_root}.gelu_output")
        raw_up, _ = self.weights.source_q(f"{source_root}.up.weight")
        raw_up_bias = self._source_bias(f"{source_root}.up.bias")
        exact_zero_up = bool(np.all(raw_up == 0) and np.all(raw_up_bias == 0))
        up = self._linear(
            graph, f"{prefix}.up", active, down_shape, self._q(f"{source_root}.up.weight"),
            raw_up_bias,
            # An all-zero quantized weight plus all-zero F32 bias is exactly
            # zero under canonical QLinear.  The source calibration correctly
            # labels its leaf probe zero-only; use the nonzero residual-output
            # boundary as a descriptor alias instead of calling that fallback
            # a calibrated adapter-up range.
            source_probe=f"{source_root}.residual_output" if exact_zero_up else f"{source_root}.up",
        )
        if exact_zero_up:
            graph.annotate_binding(
                up,
                qualification="semantic_exact_zero_qlinear_output_alias",
                aliased_from_source_probe=f"{source_root}.residual_output",
            )
        return graph.qadd(
            f"{prefix}.residual", input_name, up, shape,
            source_probe=output_probe or f"{source_root}.residual_output",
        )

    def _register_static(self, graph: GraphDocumentBuilder, name: str, logical_edge: str) -> None:
        if isinstance(graph, W8A32GraphDocumentBuilder):
            graph.alias_weight(name, f"{name}.f32")
            return
        descriptor = self.weights.quantization.get(name)
        if descriptor is None or descriptor.get("scheme") != "per_tensor":
            raise AssertionError(f"static typed constant {name!r} has invalid descriptor")
        scale_tensor = descriptor.get("scale_tensor")
        zero_point_tensor = descriptor.get("zero_point_tensor")
        if not isinstance(scale_tensor, str) or not isinstance(zero_point_tensor, str):
            raise AssertionError(f"static typed constant {name!r} has incomplete references")
        stored_scale = self.weights.values[scale_tensor]
        stored_zero_point = self.weights.values[zero_point_tensor]
        if stored_scale.dtype != np.dtype(np.float32) or stored_scale.shape != (1,):
            raise AssertionError(f"static typed constant {name!r} has an invalid scale tensor")
        if (
            stored_zero_point.dtype != np.dtype(np.int8)
            or stored_zero_point.shape != (1,)
            or int(stored_zero_point[0]) != 0
        ):
            raise AssertionError(f"static typed constant {name!r} has an invalid zero point")
        scale = graph.register_derived_static(name, logical_edge, stored_scale[0])
        if stored_scale[0] != scale:
            raise AssertionError(f"static typed constant {name!r} has mismatched graph descriptor")

    def _router_inputs(self) -> dict[str, dict[str, Any]]:
        d = self.dimensions
        return {
            "q_ids": {"shape": [1, d.max_q_len], "dtype": "int32"},
            "router_keep": {"shape": [1, d.max_q_len], "dtype": "int32"},
        }

    def _explicit_inputs(self, graph: GraphDocumentBuilder) -> dict[str, dict[str, Any]]:
        d = self.dimensions
        return {
            "image": {
                "shape": [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
                "dtype": "float32",
            },
            "q_ids": {"shape": [1, d.max_q_len], "dtype": "int32"},
            "router_keep": {"shape": [1, d.max_q_len], "dtype": "int32"},
            "memory_keep": {"shape": [1, d.img_tokens + d.max_q_len], "dtype": "int32"},
            "y_ids": {"shape": [1, d.max_out_len], "dtype": "int32"},
            "y_keep": {"shape": [1, d.max_out_len], "dtype": "int32"},
        }

    def _config(
        self,
        *,
        kind: str,
        graph: GraphDocumentBuilder,
        inputs: Mapping[str, Any],
        output_name: str,
        family: str | None = None,
    ) -> dict[str, Any]:
        d = self.dimensions
        w8a32 = isinstance(graph, W8A32GraphDocumentBuilder)
        namespace = f"{kind}:{family or graph.bias_scope}"
        referenced = {
            name
            for node in graph.nodes
            for name in (node.get("inputs") or {}).values()
            if isinstance(name, str)
        }
        quantization_table = {
            name: dict(self.weights.quantization[name])
            for name in sorted(referenced.intersection(self.weights.quantization))
        }
        if not w8a32:
            explicit_scales = {
                target: inputs_map.get("scale")
                for node in graph.nodes
                if node.get("opType") == "QuantizeLinear"
                for inputs_map in [node.get("inputs") or {}]
                for target in (node.get("outputs") or {}).values()
                if isinstance(target, str) and isinstance(inputs_map.get("scale"), str)
            }
            for name, scale in sorted(graph.tensor_scales.items()):
                if name in self.weights.quantization:
                    quantization_table[name] = dict(self.weights.quantization[name])
                else:
                    quantization_table[name] = self.weights.add_activation_quantization(
                        namespace,
                        name,
                        scale,
                        scale_tensor=explicit_scales.get(name),
                    )
            for node in graph.nodes:
                if node.get("opType") != "QuantizeLinear":
                    continue
                output = next(iter((node.get("outputs") or {}).values()), None)
                descriptor = quantization_table.get(output)
                if descriptor is not None:
                    node["inputs"]["scale"] = descriptor["scale_tensor"]
                    node["inputs"]["zero_point"] = descriptor["zero_point_tensor"]
        result: dict[str, Any] = {
            "format": GRAPH_FORMAT,
            "graph_profile": W8A32_GRAPH_PROFILE if w8a32 else W8A8_GRAPH_PROFILE,
            "model_type": (
                "tiny_receipt_vqa_w8a32_weight_only_materialized"
                if w8a32 else "tiny_receipt_vqa_w8a8_materialized"
            ),
            "development_only": True,
            "package_kind": kind,
            "execution": {
                "weights": "int8_per_output_channel",
                "activations": "float32" if w8a32 else "int8_symmetric",
                "bias": "float32_exact_source" if w8a32 else "int32_accumulator",
            },
            "fixed_batch_size": 1,
            "weights_file": "model.safetensors",
            "inputs": dict(inputs),
            "nodes": graph.nodes,
            "outputs": [output_name],
            "output_contract": {
                "name": output_name,
                "dtype": "int32",
            },
            "model_dimensions": {
                "vocab_size": d.vocab_size,
                "d_model": d.d_model,
                "heads": d.heads,
                "head_dim": d.head_dim,
                "encoder_layers": d.enc_layers,
                "decoder_layers": d.dec_layers,
                "max_q_len": d.max_q_len,
                "max_out_len": d.max_out_len,
                "img_tokens": d.img_tokens,
                "stem_channels": list(d.stem_channels),
            },
        }
        if quantization_table:
            result["quantization"] = {
                "format": QUANTIZATION_FORMAT,
                "tensors": quantization_table,
            }
        if w8a32:
            result["activation_edge_bindings"] = []
        else:
            result["activation_edge_bindings"] = graph.edge_bindings
            result["activation_scale_profile"] = {
                "format": CALIBRATION_FORMAT,
                "profile_id": self.profile.profile_id,
                "source": self.profile.source,
                "mapping_record": "package_manifest.json#activation_scale_profile",
            }
        if family is not None:
            result["explicit_family"] = {
                "id": FAMILY_ORDER.index(family),
                "name": family,
                "route_scope": "whole_execution",
                "batch_policy": "homogeneous_fixed_b1",
            }
        try:
            validate_external_quantization(result, self.weights.values)
        except ExporterError as error:
            raise MaterializationError(
                f"invalid materialized quantization contract: {error.diagnostic.message}"
            ) from error
        return result

    def build_router_graph(self) -> dict[str, Any]:
        d = self.dimensions
        graph = GraphDocumentBuilder(self.profile, bias_scope="router")
        self._register_static(graph, "c.q_pos", "constant.q_pos")
        self._register_static(graph, "c.q_type", "constant.q_type")
        inputs = self._router_inputs()
        question_shape = [1, d.max_q_len, d.d_model]
        embedded = graph.typed(
            "QEmbedding", "router.embed", {"input": "q_ids", "weight": self._q("tok.weight").name},
            "a.router.embed", question_shape, {}, source_probe="tok",
        )
        positioned = graph.qadd(
            "router.position", embedded, "c.q_pos", question_shape, source_probe="router.q_tokens"
        )
        tokens = graph.qadd(
            "router.tokens", positioned, "c.q_type", question_shape, source_probe="router.q_tokens"
        )
        pooled = graph.typed(
            "QMaskedMean", "router.pool", {"input": tokens, "mask": "router_keep"}, "a.router.pool", [1, d.d_model], {},
            source_probe="router.q_tokens",
        )
        hidden = self._linear(
            graph, "router.net0", pooled, [1, d.d_model], self._q("router.net.0.weight"),
            self._source_bias("router.net.0.bias"), source_probe="router.net.0",
        )
        hidden = graph.qgelu("router.gelu", hidden, [1, d.d_model // 2], source_probe="router.net.1")
        logits = self._linear(
            graph, "router.net2", hidden, [1, d.d_model // 2], self._q("router.net.2.weight"),
            self._source_bias("router.net.2.bias"), source_probe="router.net.2",
        )
        graph.i32("QArgMax", "router.argmax", {"input": logits}, "router_family", [1], {"axis": -1})
        return self._config(kind="router", graph=graph, inputs=inputs, output_name="router_family")

    def build_explicit_family_config(
        self, family: str, *, family_execution: str = "w8a8"
    ) -> dict[str, Any]:
        if family not in FAMILY_ORDER:
            raise MaterializationError(f"unknown explicit task family {family!r}")
        if family_execution not in {"w8a8", "w8a32"}:
            raise MaterializationError("family_execution must be 'w8a8' or 'w8a32'")
        family_id = FAMILY_ORDER.index(family)
        d = self.dimensions
        graph: GraphDocumentBuilder | W8A32GraphDocumentBuilder
        if family_execution == "w8a32":
            graph = W8A32GraphDocumentBuilder(self.weights, bias_scope=f"family_{family}")
            graph.alias_embedding(self._q("tok.weight").name, "f.tok.embedding")
        else:
            graph = GraphDocumentBuilder(self.profile, bias_scope=f"family_{family}")
        for name, key in (
            ("c.img_pos", "constant.img_pos"),
            ("c.img_type", "constant.img_type"),
            ("c.q_pos", "constant.q_pos"),
            ("c.q_type", "constant.q_type"),
            ("c.y_pos", "constant.y_pos"),
        ):
            self._register_static(graph, name, key)
        inputs = self._explicit_inputs(graph)

        image_shape = [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
        if family_execution == "w8a32":
            x = "image"
        else:
            x = graph.typed(
                "QuantizeLinear", "input.quantize_image", {"input": "image", "scale": "c.input_image_scale"},
                "a.input.image_i8", image_shape, {}, scale_key="input.image", source_probe="input.image_preprocessed",
            )
        shape = image_shape
        for spec in _stem_specs(d):
            index = spec["index"]
            if spec["kind"] == "conv":
                root = f"stem.{index}.net"
                x, shape = self._conv(
                    graph, f"stem.{index}.conv", x, shape, f"{root}.{spec['conv']}.weight", int(spec["stride"]),
                    source_probe=f"{root}.{spec['conv']}",
                )
                x = self._group_norm(
                    graph, f"stem.{index}.norm", x, shape, f"{root}.{spec['norm']}",
                    source_probe=f"{root}.{spec['norm']}",
                )
                x = self._qsilu(graph, f"stem.{index}.silu", x, shape, source_probe=f"{root}.2")
            else:
                root = f"stem.{index}.net"
                shortcut = x
                first, _ = self._conv(
                    graph, f"stem.{index}.conv1", x, shape, f"{root}.0.weight", 1, source_probe=f"{root}.0"
                )
                first = self._group_norm(graph, f"stem.{index}.norm1", first, shape, f"{root}.1", source_probe=f"{root}.1")
                first = self._qsilu(graph, f"stem.{index}.silu1", first, shape, source_probe=f"{root}.2")
                second, _ = self._conv(
                    graph, f"stem.{index}.conv2", first, shape, f"{root}.3.weight", 1, source_probe=f"{root}.3"
                )
                second = self._group_norm(graph, f"stem.{index}.norm2", second, shape, f"{root}.4", source_probe=f"{root}.4")
                summed = graph.qadd(
                    f"stem.{index}.residual_sum", shortcut, second, shape,
                    source_probe=f"stem.{index}.residual_sum",
                )
                x = self._qsilu(
                    graph, f"stem.{index}.residual_silu", summed, shape,
                    source_probe=f"stem.{index}.residual_output",
                )

        image_tokens = graph.shape_copy("Reshape", "stem.reshape", x, "a.stem.tokens", [1, d.img_tokens, d.d_model])
        image_tokens = graph.qadd(
            "memory.image.position", image_tokens, "c.img_pos", [1, d.img_tokens, d.d_model],
            source_probe="encoder.input",
        )
        graph.annotate_binding(
            image_tokens,
            qualification="conservative_downstream_encoder_input_bound_for_position_add",
        )
        image_tokens = graph.qadd(
            "memory.image.type", image_tokens, "c.img_type", [1, d.img_tokens, d.d_model],
            scale_key="memory.tokens", source_probe="encoder.input",
        )
        question_shape = [1, d.max_q_len, d.d_model]
        question_tokens = graph.typed(
            "QEmbedding", "question.embed", {"input": "q_ids", "weight": self._q("tok.weight").name},
            "a.question.embed", question_shape, {}, source_probe="tok",
        )
        question_tokens = graph.qadd(
            "question.position", question_tokens, "c.q_pos", question_shape, source_probe="encoder.input"
        )
        graph.annotate_binding(
            question_tokens,
            qualification="conservative_downstream_encoder_input_bound_for_position_add",
        )
        question_tokens = graph.qadd(
            "question.type", question_tokens, "c.q_type", question_shape, scale_key="memory.tokens", source_probe="encoder.input",
        )
        memory_shape = [1, d.img_tokens + d.max_q_len, d.d_model]
        memory = graph.concat("memory.concat", [image_tokens, question_tokens], "a.memory", memory_shape, 1)

        for layer in range(d.enc_layers):
            root = f"encoder.layers.{layer}"
            memory = self._self_attention(
                graph, f"encoder.l{layer}.self", memory, memory_shape, root, 1, "memory_keep", causal=False
            )
            memory = self._ffn(graph, f"encoder.l{layer}.ffn", memory, memory_shape, root, 2)
        memory_adapter_alias = (
            f"encoder.layers.{d.enc_layers - 1}.output" if family in {"math", "other"} else None
        )
        memory = self._adapter(
            graph, f"encoder.adapter.{family}", memory, memory_shape, f"memory_adapters.{family_id}",
            output_probe=memory_adapter_alias,
        )
        if memory_adapter_alias is not None:
            graph.annotate_binding(
                memory,
                qualification="semantic_zero_delta_identity_alias_for_zero_only_adapter_probe",
                aliased_from_source_probe=memory_adapter_alias,
            )

        decoder_shape = [1, d.max_out_len, d.d_model]
        decoder = graph.typed(
            "QEmbedding", "decoder.embed", {"input": "y_ids", "weight": self._q("tok.weight").name},
            "a.decoder.embed", decoder_shape, {}, source_probe="tok",
        )
        decoder = graph.qadd(
            "decoder.position", decoder, "c.y_pos", decoder_shape, scale_key="decoder.input", source_probe="decoder.target_input"
        )
        for layer in range(d.dec_layers):
            root = f"decoder.layers.{layer}"
            decoder = self._self_attention(
                graph, f"decoder.l{layer}.self", decoder, decoder_shape, root, 1, "y_keep", causal=True
            )
            decoder = self._cross_attention(
                graph, f"decoder.l{layer}.cross", decoder, decoder_shape, memory, memory_shape, root, "memory_keep"
            )
            decoder = self._ffn(graph, f"decoder.l{layer}.ffn", decoder, decoder_shape, root, 3)
        decoder_adapter_alias = (
            f"decoder.layers.{d.dec_layers - 1}.output" if family in {"math", "other"} else None
        )
        decoder = self._adapter(
            graph, f"decoder.adapter.{family}", decoder, decoder_shape, f"decoder_adapters.{family_id}",
            output_probe=decoder_adapter_alias,
        )
        if decoder_adapter_alias is not None:
            graph.annotate_binding(
                decoder,
                qualification="semantic_zero_delta_identity_alias_for_zero_only_adapter_probe",
                aliased_from_source_probe=decoder_adapter_alias,
            )
        normalized = graph.qlayernorm(
            "decoder.final_norm", decoder, decoder_shape, self._f("norm.weight"), self._f("norm.bias"), d.d_model,
            source_probe="norm",
        )
        logits = self._linear(
            graph, "decoder.head", normalized, decoder_shape, self._q("tok.weight"), self._source_bias("out_bias"),
            source_probe="head",
        )
        graph.i32(
            "ArgMax" if family_execution == "w8a32" else "QArgMax",
            "decoder.argmax", {"input": logits}, "token_ids", [1, d.max_out_len],
            {"axis": -1, "keepdims": 0} if family_execution == "w8a32" else {"axis": -1},
        )
        return self._config(
            kind="explicit_family", graph=graph, inputs=inputs, output_name="token_ids", family=family
        )


def _validated_vocab(manifest: Mapping[str, Any], dimensions: ModelDimensions) -> tuple[dict[str, Any], dict[str, int]]:
    vocab = manifest.get("vocab")
    if not isinstance(vocab, dict):
        raise MaterializationError("manifest.vocab must be an object")
    itos = vocab.get("itos")
    if not isinstance(itos, list) or len(itos) != dimensions.vocab_size or any(not isinstance(token, str) for token in itos):
        raise MaterializationError("manifest.vocab.itos must be a string array matching vocab_size")
    expected = ("<pad>", "<bos>", "<eos>", "<unk>")
    if tuple(itos[:4]) != expected:
        raise MaterializationError(f"manifest vocabulary must begin with {list(expected)!r}")
    return {"itos": list(itos)}, {"pad": 0, "bos": 1, "eos": 2, "unk": 3}


def _graph_weight_names(
    graph: Mapping[str, Any], weights: Mapping[str, np.ndarray]
) -> frozenset[str]:
    """Return the exact persisted-tensor closure named by one graph.

    Runtime tensor references live in node inputs, direct graph outputs, and
    the central affine descriptor table.  Intersecting those names with the
    already-validated materialized inventory keeps graph inputs and node
    outputs out of the shard without inferring meaning from tensor names.
    """
    referenced: set[str] = set()

    def include(value: Any) -> None:
        if isinstance(value, str) and value in weights:
            referenced.add(value)

    for output in graph.get("outputs", ()):  # A graph output may be a weight.
        include(output)
    for node in graph.get("nodes", ()):
        if not isinstance(node, Mapping):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, Mapping):
            for value in inputs.values():
                include(value)
    quantization = graph.get("quantization")
    descriptors = quantization.get("tensors") if isinstance(quantization, Mapping) else None
    if isinstance(descriptors, Mapping):
        for target, descriptor in descriptors.items():
            include(target)
            if isinstance(descriptor, Mapping):
                include(descriptor.get("scale_tensor"))
                include(descriptor.get("zero_point_tensor"))
    return frozenset(referenced)


def _save_scoped_weight_file(
    destination: Path,
    relative_path: str,
    names: Iterable[str],
    weights: Mapping[str, np.ndarray],
    metadata: Mapping[str, str],
) -> str:
    selected_names = sorted(set(names))
    if not selected_names:
        raise MaterializationError(f"scoped weight file {relative_path} would be empty")
    path = destination / relative_path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {name: weights[name] for name in selected_names},
            str(path),
            metadata=dict(metadata),
        )
    except (KeyError, OSError, ValueError) as error:
        raise MaterializationError(f"could not write scoped weights {path}: {error}") from error
    return relative_path


def _ensure_new_or_empty_output_dir(output_dir: Path, source_dir: Path) -> None:
    output_dir = Path(output_dir)
    if output_dir.resolve() == source_dir.resolve():
        raise MaterializationError("--out-dir must not overwrite the source release directory")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise MaterializationError(f"--out-dir is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise MaterializationError(f"--out-dir must be new or empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)


def materialize_release(
    manifest_path: Path,
    output_dir: Path,
    *,
    activation_scale: float | None = None,
    activation_calibration: Path | None = None,
    development_calibration: Path | None = None,
    family_execution: str = "w8a8",
) -> MaterializationResult:
    """Write a self-contained development W8A8 or hybrid package.

    Exactly one profile source is required.  ``activation_scale`` deliberately
    remains available as a transparent fallback for early development.  A
    generic activation profile can supply logical-edge overrides, while a
    repository development calibration is imported only through the strict
    source-probe binding path.
    """
    if family_execution not in {"w8a8", "w8a32"}:
        raise MaterializationError("family_execution must be 'w8a8' or 'w8a32'")
    selected = sum(value is not None for value in (activation_scale, activation_calibration, development_calibration))
    if selected != 1:
        raise MaterializationError(
            "provide exactly one of --activation-scale, --activation-calibration, or --development-calibration"
        )
    if activation_scale is not None:
        profile = ActivationProfile.fallback(activation_scale)
    elif activation_calibration is not None:
        profile = ActivationProfile.from_json(Path(activation_calibration))
    else:
        assert development_calibration is not None
        profile = ActivationProfile.from_development_calibration(Path(development_calibration))

    release = load_release(Path(manifest_path))
    destination = Path(output_dir)
    _ensure_new_or_empty_output_dir(destination, release.manifest_path.parent)
    converter = TinyReceiptMaterializer(release, profile)
    try:
        converter.prepare_weights()
        if family_execution == "w8a32":
            converter.prepare_w8a32_family_weights()
        router_graph = converter.build_router_graph()
        explicit_graphs = {
            family: converter.build_explicit_family_config(
                family, family_execution=family_execution
            )
            for family in FAMILY_ORDER
        }
        vocab, token_ids = _validated_vocab(release.manifest, release.dimensions)

        weights_format = W8A32_WEIGHTS_FORMAT if family_execution == "w8a32" else WEIGHTS_FORMAT
        weights_metadata = {
            "format": weights_format,
            "source_format": str(release.manifest["format"]),
            "source_manifest_sha256": _sha256(release.manifest_path),
        }
        weights_path = destination / "model.safetensors"
        try:
            save_file(
                converter.weights.values,
                str(weights_path),
                metadata=weights_metadata,
            )
        except (OSError, ValueError) as error:
            raise MaterializationError(f"could not write materialized weights {weights_path}: {error}") from error

        router_weight_names = _graph_weight_names(
            router_graph, converter.weights.values
        )
        family_weight_names = {
            family: _graph_weight_names(graph, converter.weights.values)
            for family, graph in explicit_graphs.items()
        }
        if not router_weight_names or any(
            not names for names in family_weight_names.values()
        ):
            raise MaterializationError(
                "every materialized graph must reference persisted weights"
            )
        common_family_weights = set.intersection(
            *(set(family_weight_names[family]) for family in FAMILY_ORDER)
        )
        router_weight_files = [
            _save_scoped_weight_file(
                destination,
                f"{SCOPED_WEIGHTS_DIR}/router.safetensors",
                router_weight_names,
                converter.weights.values,
                {**weights_metadata, "scope": "router"},
            )
        ]
        common_weight_file = (
            _save_scoped_weight_file(
                destination,
                f"{SCOPED_WEIGHTS_DIR}/common.safetensors",
                common_family_weights,
                converter.weights.values,
                {**weights_metadata, "scope": "explicit-family-common"},
            )
            if common_family_weights else None
        )
        family_weight_files: dict[str, list[str]] = {}
        for family in FAMILY_ORDER:
            paths = [common_weight_file] if common_weight_file is not None else []
            family_only = set(family_weight_names[family]) - common_family_weights
            if family_only:
                paths.append(_save_scoped_weight_file(
                    destination,
                    f"{SCOPED_WEIGHTS_DIR}/explicit_family_{family}.safetensors",
                    family_only,
                    converter.weights.values,
                    {**weights_metadata, "scope": f"explicit-family:{family}"},
                ))
            if not paths:
                raise MaterializationError(
                    f"explicit family {family!r} has no scoped weight files"
                )
            family_weight_files[family] = paths

        router_path = destination / "router.graph.json"
        _write_json(router_path, router_graph)
        family_paths: dict[str, Path] = {}
        for family, graph in explicit_graphs.items():
            path = destination / f"explicit_family_{family}.graph.json"
            _write_json(path, graph)
            family_paths[family] = path
        vocab_path = destination / "vocab.json"
        _write_json(vocab_path, vocab)

        calibration_copy_path: Path | None = None
        if profile.calibration_path is not None:
            calibration_copy_path = destination / "activation_calibration.json"
            try:
                shutil.copyfile(profile.calibration_path, calibration_copy_path)
            except OSError as error:
                raise MaterializationError(f"could not copy activation calibration into package: {error}") from error

        router_input_aliases = {"q_ids": "q_ids", "router_keep": "router_keep"}
        explicit_input_aliases = {
            "image": "image",
            "q_ids": "q_ids",
            "router_keep": "router_keep",
            "memory_keep": "memory_keep",
            "y_ids": "y_ids",
            "y_keep": "y_keep",
        }
        explicit_interfaces = {
            family: {
                "graph": path.name,
                "weight_files": family_weight_files[family],
                "interface": {
                    "inputs": dict(explicit_input_aliases),
                    "output_name": "token_ids",
                },
            }
            for family, path in family_paths.items()
        }
        profile_record = profile.manifest_record()
        if calibration_copy_path is not None:
            profile_record["package_file"] = calibration_copy_path.name
        profile_record["static_constant_quantization"] = dict(sorted(converter.weights.constant_quantization.items()))
        profile_record["binding_files"] = {
            "router": router_path.name,
            "explicit_families": (
                {family: path.name for family, path in family_paths.items()}
                if family_execution == "w8a8" else {}
            ),
        }
        package_manifest: dict[str, Any] = {
            "format": PACKAGE_FORMAT,
            "development_only": True,
            "execution_variant": {
                "router": "w8a8",
                "explicit_families": family_execution,
                "hybrid": family_execution == "w8a32",
                "selection_time": "package_materialization",
            },
            "weights": {"file": weights_path.name, "sha256": _sha256(weights_path)},
            "router": {
                "graph": router_path.name,
                "weight_files": router_weight_files,
                "output_name": "router_family",
                "inputs": router_input_aliases,
            },
            "family_order": list(FAMILY_ORDER),
            "explicit_families": explicit_interfaces,
            "vocab": {"file": vocab_path.name, "token_ids": token_ids},
            "preprocessing": {
                "color_space": "grayscale",
                "resize": {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "resample": "bilinear"},
                "normalization": "minus-one-one",
                "formula": "(pixel / 255.0 - 0.5) / 0.5",
                "model_input_layout": "NHWC",
                "model_input_dtype": "float32",
            },
            "mask_contract": {
                "router_keep": "I32 [1,max_q_len], nonzero exactly where q_ids is not pad (0)",
                "memory_keep": "I32 [1,img_tokens+max_q_len], nonzero for all image tokens and non-pad q_ids",
                "y_keep": "I32 [1,max_out_len], nonzero exactly where y_ids is not pad (0)",
                "attention_mask_semantics": "nonzero means allowed/kept",
            },
            "generation_contract": {
                "batch_size": 1,
                "decoder": "host supplies fixed max_out_len y_ids/y_keep on each autoregressive step",
                "router": "run router.graph.json, then select the returned explicit family graph for the whole execution",
                "execution_variant": (
                    "hybrid W8A8 router plus W8A32 explicit family"
                    if family_execution == "w8a32" else "W8A8 router and explicit family"
                ),
            },
            "source": {
                "manifest_file": release.manifest_path.name,
                "manifest_sha256": _sha256(release.manifest_path),
                "weights_file": release.weights_path.name,
                "weights_sha256": _sha256(release.weights_path),
                "format": release.manifest["format"],
                "runtime": release.manifest["runtime"],
            },
            "activation_scale_profile": profile_record,
            "limitations": [
                "Development materialization only: no executable preprocessing, calibration corpus, or golden-output verification is performed here.",
                "The package has no automatic router-to-family session dispatcher; callers must run the router once and select one fixed-B=1 explicit-family graph.",
                (
                    "QGELU uses Volvox's portable approximation; quantized output fidelity still needs goldens."
                    if release.manifest["format"] == VOLVOX_TRAINED_SOURCE_FORMAT else
                    "QGELU uses Volvox's portable F32 erf approximation, while the source PyTorch Transformer requested GELU; output fidelity needs goldens."
                ),
                "The source model is evaluated with dropout disabled; this graph document contains no dropout operator.",
                "A fallback activation profile or any listed unqualified fallback edge is not a claim of production-quality per-edge calibration.",
            ],
        }
        if family_execution == "w8a32":
            package_manifest["activation_scale_profile"]["scope"] = "router_only"
            package_manifest["limitations"].append(
                "This hybrid evaluation package keeps its learned router in W8A8; only the selected explicit-family graph uses W8A32."
            )
        package_path = destination / "package_manifest.json"
        _write_json(package_path, package_manifest)
    except Exception:
        # Keep a deliberate fail-closed contract: a partial package is not an
        # executable artifact.  The directory was proven empty at entry, so
        # removing only files produced in this invocation is safe.
        for child in list(destination.iterdir()):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        raise
    return MaterializationResult(
        output_dir=destination,
        weights_path=weights_path,
        router_graph_path=router_path,
        explicit_graph_paths=tuple(family_paths[family] for family in FAMILY_ORDER),
        package_manifest_path=package_path,
        constant_saturation_count=sum(converter.weights.constant_saturation.values()),
        family_execution=family_execution,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only TinyReceiptVQA INT8 -> Volvox W8A8/hybrid materializer. "
            "It creates a W8A8 router plus eight fixed-B=1 W8A8 or W8A32 family graphs."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path, help="source int8/manifest.json")
    parser.add_argument("--out-dir", required=True, type=Path, help="new or empty destination package directory")
    parser.add_argument(
        "--family-execution", choices=("w8a8", "w8a32"), default="w8a8",
        help="explicit-family graph mode; w8a32 retains the W8A8 router for AUTO evaluation",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--activation-scale", type=float,
        help="explicit positive symmetric I8 fallback scale for all live activation edges",
    )
    source.add_argument(
        "--activation-calibration", type=Path,
        help=f"external {CALIBRATION_FORMAT} logical-edge scale profile",
    )
    source.add_argument(
        "--development-calibration", type=Path,
        help="repository development calibration record; strict source-probe bindings are required",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = materialize_release(
            args.manifest,
            args.out_dir,
            activation_scale=args.activation_scale,
            activation_calibration=args.activation_calibration,
            development_calibration=args.development_calibration,
            family_execution=args.family_execution,
        )
    except MaterializationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output_dir": str(result.output_dir),
        "weights": str(result.weights_path),
        "router_graph": str(result.router_graph_path),
        "explicit_family_graphs": [str(path) for path in result.explicit_graph_paths],
        "package_manifest": str(result.package_manifest_path),
        "constant_saturation_count": result.constant_saturation_count,
        "family_execution": result.family_execution,
        "development_only": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
