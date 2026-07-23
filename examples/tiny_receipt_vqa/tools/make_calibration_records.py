#!/usr/bin/env python3
"""Build a provenance-complete TinyReceipt PTQ calibration manifest.

The release training code partitions real annotations by sorting their paths,
shuffling with ``random.Random(seed)``, and reserving the leading validation
fraction.  This tool intentionally reproduces that rule without importing the
training package (and therefore without pulling PyTorch into an export tool).

Only the training partition is eligible.  Representative records are selected
deterministically and evenly per requested routed family, and every selected
annotation/image is content-addressed in the emitted manifest.

For the structured-QA release, ``--synth-root`` replays the exact producer
selection through a lazily loaded ``--producer-root`` training module.  That
offline, model-owned mode never enters the generic exporter or an inference
entry.  It preserves the producer's six semantic record families; public-route
execution coverage is a later, separate calibration dimension.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


FORMAT = "volvox-calibration-records/v1"
FAMILIES = ("phone", "address")
STRUCTURED_QA_FAMILIES = (
    "phone",
    "address",
    "store",
    "item_row",
    "item_math",
    "item_lookup",
    "math",
    "other",
)


def clean_text(value: object) -> str:
    text = str(value if value is not None else "").replace("\n", " ")
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", text).strip())


def digits_only(value: object) -> str:
    return "".join(re.findall(r"\d", str(value if value is not None else "")))


def normalize_address_text(value: object) -> str:
    text = clean_text(value)
    return re.sub(
        r"(\d+(?:\s+\d+)+)$",
        lambda match: re.sub(r"\s+", "", match.group(1)),
        text,
    )


def phone_op_from_question(question: str) -> str:
    text = clean_text(question).lower()
    operation = "phone_digit"
    ordinal = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
    }
    match = re.search(r"(first|second|third|fourth|fifth|sixth|seventh|eighth)", text)
    if match:
        operation = f"front_{ordinal[match.group(1)]}"
    match = re.search(r"(?:front of|from the front|digit)\D*(\d+)", text)
    if match:
        operation = f"front_{match.group(1)}"
    match = re.search(r"(?:from the end|from the back|from the right|last)\D*(\d+)", text)
    if match:
        operation = f"back_{match.group(1)}"
    match = re.search(r"앞에서\s*(\d+)번째", text)
    if match:
        operation = f"front_{match.group(1)}"
    match = re.search(r"(?:앞|앞자리)\s*(\d+)(?:번째|번)?", text)
    if match:
        operation = f"front_{match.group(1)}"
    match = re.search(r"뒤에서\s*(\d+)번째", text)
    if match:
        operation = f"back_{match.group(1)}"
    match = re.search(r"(?:뒤|뒷자리|끝자리)\s*(\d+)(?:번째|번)?", text)
    if match:
        operation = f"back_{match.group(1)}"
    if "from the back" in text or "from the end" in text or "from last" in text:
        match = re.search(
            r"(first|second|third|fourth|fifth|sixth|seventh|eighth)", text
        )
        if match:
            operation = f"back_{ordinal[match.group(1)]}"
    return operation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_record_id(value: Mapping[str, Any]) -> str:
    receipt_id = clean_text(value.get("receipt_id"))
    if not receipt_id:
        raise ValueError("synthetic calibration record has no receipt_id")
    identity = {
        "receipt_id": receipt_id,
        "family": clean_text(value.get("family")),
        "task": clean_text(value.get("task")),
        "op": clean_text(value.get("op")),
        "question": clean_text(value.get("question")),
        "target": clean_text(value.get("target")),
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"synth-{receipt_id}-{digest[:16]}"


def _load_producer_training_module(producer_root: Path) -> ModuleType:
    """Load the producer's task generator only in the offline example CLI."""

    root = producer_root.resolve()
    expected = root / "tiny_receipt_vqa" / "train.py"
    if not expected.is_file():
        raise ValueError(f"producer training module is missing: {expected}")
    package_name = "tiny_receipt_vqa"
    module_name = f"{package_name}.train"
    existing = sys.modules.get(module_name)
    if existing is not None:
        origin = Path(getattr(existing, "__file__", "")).resolve()
        if origin != expected:
            raise ValueError(
                f"{module_name} was already loaded from {origin}, expected {expected}"
            )
        return existing
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module(module_name)
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:  # pragma: no cover - defensive against import hooks.
            pass
    origin = Path(getattr(module, "__file__", "")).resolve()
    if origin != expected:
        raise ValueError(
            f"loaded {module_name} from {origin}, expected {expected}"
        )
    return module


def _require_training_api(module: ModuleType) -> None:
    for name in (
        "TASK_PROFILE_STRUCTURED_QA",
        "synth_records",
        "stratified_sample",
        "task_family",
    ):
        if not hasattr(module, name):
            raise ValueError(f"producer training module is missing {name}")
    for name in ("synth_records", "stratified_sample", "task_family"):
        if not callable(getattr(module, name)):
            raise ValueError(f"producer training module {name} must be callable")


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _synthetic_record(
    value: Mapping[str, Any],
    *,
    module: ModuleType,
    annotation_dir: Path,
    image_dir: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("producer-selected calibration records must be objects")
    family = clean_text(module.task_family(dict(value)))
    if family not in STRUCTURED_QA_FAMILIES:
        raise ValueError(f"synthetic record has unknown family {family!r}")
    question = clean_text(value.get("question"))
    target = clean_text(value.get("target"))
    receipt_id = clean_text(value.get("receipt_id"))
    if not question or not target or not receipt_id:
        raise ValueError(
            "synthetic calibration record requires receipt_id, question, and target"
        )
    annotation_path = (annotation_dir / f"{receipt_id}.json").resolve()
    image_path = Path(str(value.get("image", ""))).expanduser().resolve()
    if not _within(annotation_path, annotation_dir) or not annotation_path.is_file():
        raise ValueError(f"synthetic annotation is missing: {annotation_path}")
    if not _within(image_path, image_dir) or not image_path.is_file():
        raise ValueError(
            f"synthetic image must be a file under the training image root: {image_path}"
        )
    normalized = {
        "receipt_id": receipt_id,
        "family": family,
        "task": clean_text(value.get("task")),
        "op": clean_text(value.get("op")),
        "question": question,
        "target": target,
    }
    return {
        "id": _canonical_record_id(normalized),
        "image": str(image_path),
        "question": question,
        "target": target,
        "family": family,
        "split": "train",
        "receipt_id": receipt_id,
        "task": normalized["task"],
        "op": normalized["op"],
        "annotation_sha256": _sha256(annotation_path),
        "image_sha256": _sha256(image_path),
    }


def build_synthetic_manifest(
    synth_root: Path,
    producer_root: Path,
    *,
    seed: int = 71,
    receipt_limit: int = 1024,
    sample_count: int = 256,
    tasks_per_receipt: int = 6,
    training_module: ModuleType | None = None,
) -> dict[str, Any]:
    """Replay the release producer's structured-QA calibration selection."""

    if receipt_limit < 1 or sample_count < 1 or tasks_per_receipt < 1:
        raise ValueError(
            "receipt limit, sample count, and tasks per receipt must be positive"
        )
    synth_root = synth_root.resolve()
    producer_root = producer_root.resolve()
    annotation_dir = synth_root / "train" / "ann"
    image_dir = synth_root / "train" / "img"
    if not annotation_dir.is_dir() or not image_dir.is_dir():
        raise ValueError(
            f"synthetic training split requires {annotation_dir} and {image_dir}"
        )
    module = training_module or _load_producer_training_module(producer_root)
    _require_training_api(module)
    generator_path = Path(getattr(module, "__file__", "")).resolve()
    if not generator_path.is_file():
        raise ValueError("producer training module must have a readable source file")

    candidates = module.synth_records(
        synth_root,
        "train",
        "rationale",
        receipt_limit,
        tasks_per_receipt,
        seed,
        module.TASK_PROFILE_STRUCTURED_QA,
    )
    if not isinstance(candidates, list):
        raise ValueError("producer synth_records must return a list")
    selected = module.stratified_sample(
        candidates,
        sample_count,
        random.Random(seed),
    )
    if not isinstance(selected, list) or len(selected) != sample_count:
        count = len(selected) if isinstance(selected, list) else "non-list"
        raise ValueError(
            f"producer selection returned {count} records; expected {sample_count}"
        )
    records = [
        _synthetic_record(
            value,
            module=module,
            annotation_dir=annotation_dir,
            image_dir=image_dir,
        )
        for value in selected
    ]
    identifiers = [record["id"] for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("synthetic calibration selection contains duplicate records")
    counts = Counter(record["family"] for record in records)
    required = [family for family in STRUCTURED_QA_FAMILIES if counts[family]]
    eligible_annotations = sorted(annotation_dir.glob("*.json"))[:receipt_limit]
    if not eligible_annotations:
        raise ValueError(f"no synthetic annotations found under {annotation_dir}")
    return {
        "format": FORMAT,
        "required_families": required,
        "source": {
            "kind": "tiny-receipt-synthetic-training-split",
            "producer_root": str(producer_root),
            "synth_root": str(synth_root),
            "split": "train",
            "heldout_used": False,
            "generator": {
                "path": str(generator_path),
                "sha256": _sha256(generator_path),
            },
            "eligible_annotation_inventory_sha256": _inventory_sha256(
                eligible_annotations
            ),
            "eligible_annotation_count": len(eligible_annotations),
            "candidate_count": len(candidates),
            "selection": {
                "algorithm": "producer-synth-records-stratified-sample/v1",
                "task_profile": str(module.TASK_PROFILE_STRUCTURED_QA),
                "mode": "rationale",
                "seed": seed,
                "receipt_limit": receipt_limit,
                "tasks_per_receipt": tasks_per_receipt,
                "sample_count": sample_count,
                "family_counts": {
                    family: counts[family] for family in required
                },
            },
        },
        "records": records,
    }


def split_annotation_paths(
    annotation_dir: Path, *, seed: int, validation_fraction: float
) -> tuple[list[Path], list[Path], list[Path]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation fraction must be strictly between zero and one")
    inventory = sorted(annotation_dir.glob("*.json"))
    if not inventory:
        raise ValueError(f"no JSON annotations found under {annotation_dir}")
    shuffled = list(inventory)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(len(shuffled) * validation_fraction))
    return inventory, shuffled[validation_count:], shuffled[:validation_count]


def _family_and_target(annotation: dict[str, Any]) -> tuple[str, str]:
    question = clean_text(annotation.get("question"))
    answer = clean_text(annotation.get("answer"))
    if not question or not answer:
        raise ValueError("question and answer must be non-empty")
    receipt = annotation.get("receipt")
    store = receipt.get("store") if isinstance(receipt, dict) else None
    if not isinstance(store, dict):
        raise ValueError("annotation receipt.store must be an object")
    if "phone number" in question or "전화번호" in question:
        family = "phone"
        field = "phone"
        value = digits_only(store.get("phone"))
        operation = phone_op_from_question(question)
    elif any(marker in question for marker in ("location", "address", "가게 위치")):
        family = "address"
        field = "addr"
        value = normalize_address_text(store.get("address"))
        operation = "street_no"
    else:
        raise ValueError("annotation is not a supported phone/address release task")
    if not value:
        raise ValueError(f"{family} annotation has an empty source value")
    target = (
        f"<field>{field}</field><value>{value}</value>"
        f"<op>{operation}</op><answer>{answer}</answer>"
    )
    return family, target


def _record(annotation_path: Path, image_dir: Path) -> dict[str, Any]:
    image_path = (image_dir / f"{annotation_path.stem}.jpg").resolve()
    if not image_path.is_file():
        raise ValueError(f"missing calibration image {image_path}")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(annotation, dict):
        raise ValueError(f"annotation {annotation_path} must contain an object")
    family, target = _family_and_target(annotation)
    return {
        "id": annotation_path.stem,
        "image": str(image_path),
        "question": clean_text(annotation.get("question")),
        "target": target,
        "family": family,
        "split": "train",
        "annotation_sha256": _sha256(annotation_path),
        "image_sha256": _sha256(image_path),
    }


def build_manifest(
    annotation_dir: Path,
    image_dir: Path,
    *,
    seed: int = 71,
    validation_fraction: float = 0.1,
    per_family: int = 8,
    families: tuple[str, ...] = FAMILIES,
) -> dict[str, Any]:
    if per_family < 1:
        raise ValueError("per-family count must be positive")
    if not families or len(set(families)) != len(families):
        raise ValueError("families must contain unique supported family names")
    unsupported = sorted(set(families) - set(FAMILIES))
    if unsupported:
        raise ValueError(f"unsupported calibration families: {unsupported}")
    annotation_dir = annotation_dir.resolve()
    image_dir = image_dir.resolve()
    inventory, training, validation = split_annotation_paths(
        annotation_dir, seed=seed, validation_fraction=validation_fraction
    )
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in FAMILIES}
    skipped = 0
    for path in training:
        try:
            record = _record(path, image_dir)
        except ValueError as error:
            if "not a supported phone/address release task" in str(error):
                skipped += 1
                continue
            raise
        buckets[record["family"]].append(record)
    selected: list[dict[str, Any]] = []
    for family in families:
        candidates = sorted(buckets[family], key=lambda record: record["id"])
        if len(candidates) < per_family:
            raise ValueError(
                f"training split has only {len(candidates)} {family} records; "
                f"need {per_family}"
            )
        indices = (
            [0]
            if per_family == 1
            else [
                (index * (len(candidates) - 1)) // (per_family - 1)
                for index in range(per_family)
            ]
        )
        selected.extend(candidates[index] for index in indices)
    validation_ids = {path.stem for path in validation}
    if any(record["id"] in validation_ids for record in selected):
        raise AssertionError("selected calibration record leaked from validation")
    return {
        "format": FORMAT,
        "required_families": list(families),
        "source": {
            "kind": "tiny-receipt-real-training-split",
            "annotation_dir": str(annotation_dir),
            "image_dir": str(image_dir),
            "annotation_inventory_sha256": _inventory_sha256(inventory),
            "annotation_count": len(inventory),
            "training_count": len(training),
            "validation_count": len(validation),
            "unsupported_training_records": skipped,
            "split": {
                "algorithm": "sorted-paths-python-random-shuffle-prefix/v1",
                "seed": seed,
                "validation_fraction": validation_fraction,
            },
            "selection": {
                "algorithm": "per-family-sorted-even-coverage/v1",
                "per_family": per_family,
                "families": list(families),
            },
        },
        "records": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotations", type=Path)
    source.add_argument("--synth-root", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument(
        "--producer-root",
        type=Path,
        help="producer source root containing tiny_receipt_vqa/train.py",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--per-family", type=int, default=8)
    parser.add_argument("--receipt-limit", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--tasks-per-receipt", type=int, default=6)
    parser.add_argument(
        "--family", action="append", choices=FAMILIES, dest="families",
        help="emit only this routed specialization profile; repeatable",
    )
    args = parser.parse_args()
    if args.synth_root is not None:
        if args.producer_root is None:
            parser.error("--synth-root requires --producer-root")
        if args.images is not None or args.families:
            parser.error("--images/--family apply only to --annotations")
        manifest = build_synthetic_manifest(
            args.synth_root,
            args.producer_root,
            seed=args.seed,
            receipt_limit=args.receipt_limit,
            sample_count=args.samples,
            tasks_per_receipt=args.tasks_per_receipt,
        )
    else:
        if args.images is None:
            parser.error("--annotations requires --images")
        if args.producer_root is not None:
            parser.error("--producer-root applies only to --synth-root")
        manifest = build_manifest(
            args.annotations,
            args.images,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            per_family=args.per_family,
            families=tuple(args.families or FAMILIES),
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts = {
        family: sum(record["family"] == family for record in manifest["records"])
        for family in manifest["required_families"]
    }
    print(
        f"wrote {args.out} ({len(manifest['records'])} records; "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
        + ")"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
