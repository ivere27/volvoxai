import importlib.util
import json
import random
import types
import unittest
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory


TOOL = Path(__file__).parents[1] / "tools" / "make_calibration_records.py"
SPEC = importlib.util.spec_from_file_location("make_calibration_records", TOOL)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_record(root: Path, name: str, family: str, number: int) -> None:
    annotations = root / "annotations"
    images = root / "images"
    annotations.mkdir(exist_ok=True)
    images.mkdir(exist_ok=True)
    if family == "phone":
        question = "가게 전화번호의 앞에서 1번째 숫자는 무엇입니까?"
        answer = str(number)[0]
    else:
        question = f"영수증의 가게 위치는 시험길 [?] 입니다. (빈 칸을 채워주세요)"
        answer = str(number)
    document = {
        "question": question,
        "answer": answer,
        "receipt": {"store": {
            "phone": f"010-{number:04d}-1234",
            "address": f"서울시 시험길 {number}",
        }},
    }
    (annotations / f"{name}.json").write_text(
        json.dumps(document, ensure_ascii=False), encoding="utf-8"
    )
    (images / f"{name}.jpg").write_bytes(f"image-{name}".encode())


def _stub_structured_qa_generator(root: Path, record_count: int = 300):
    source = root / "producer" / "tiny_receipt_vqa" / "train.py"
    source.parent.mkdir(parents=True)
    source.write_text("# deterministic test producer\n", encoding="utf-8")
    synth_root = root / "synth" / "data"
    annotations = synth_root / "train" / "ann"
    images = synth_root / "train" / "img"
    annotations.mkdir(parents=True)
    images.mkdir(parents=True)
    families = (
        "phone", "address", "store", "item_row", "item_math", "item_lookup",
    )
    candidates = []
    for index in range(record_count):
        receipt_id = f"{index:06d}"
        family = families[index % len(families)]
        annotation = annotations / f"{receipt_id}.json"
        image = images / f"{receipt_id}.jpg"
        annotation.write_text(
            json.dumps({"answers": {"store": f"store-{index}"}}),
            encoding="utf-8",
        )
        image.write_bytes(f"image-{index}".encode())
        candidates.append({
            "receipt_id": receipt_id,
            "source": "synth",
            "image": str(image.resolve()),
            "question": f"question-{family}-{index}",
            "target": f"<answer>{index}</answer>",
            "family": family,
            "task": f"{family}_task",
            "op": "identity",
        })

    module = types.ModuleType("stub_producer_train")
    module.__file__ = str(source)
    module.TASK_PROFILE_STRUCTURED_QA = "structured_qa"
    module.calls = []

    def synth_records(
        requested_root, split, mode, receipt_limit, tasks_per_receipt, seed,
        task_profile,
    ):
        module.calls.append((
            Path(requested_root), split, mode, receipt_limit,
            tasks_per_receipt, seed, task_profile,
        ))
        return list(candidates)

    def stratified_sample(records, limit, rng: random.Random):
        buckets = defaultdict(list)
        for record in records:
            buckets[record["family"]].append(record)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        selected = []
        while len(selected) < limit:
            for family in families:
                if len(selected) == limit:
                    break
                selected.append(buckets[family].pop())
        return selected

    module.synth_records = synth_records
    module.stratified_sample = stratified_sample
    module.task_family = lambda record: record["family"]
    return module, synth_root, root / "producer"


class CalibrationRecordBuilderTests(unittest.TestCase):
    def test_replays_structured_qa_synthetic_release_selection_without_relabeling(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            module, synth_root, producer_root = _stub_structured_qa_generator(root)
            first = MODULE.build_synthetic_manifest(
                synth_root,
                producer_root,
                training_module=module,
            )
            second = MODULE.build_synthetic_manifest(
                synth_root,
                producer_root,
                training_module=module,
            )

            self.assertEqual(first, second)
            self.assertEqual(len(first["records"]), 256)
            self.assertEqual(first["required_families"], [
                "phone", "address", "store", "item_row", "item_math", "item_lookup",
            ])
            self.assertEqual(
                first["source"]["selection"]["family_counts"],
                {
                    "phone": 43,
                    "address": 43,
                    "store": 43,
                    "item_row": 43,
                    "item_math": 42,
                    "item_lookup": 42,
                },
            )
            self.assertNotIn("math", first["source"]["selection"]["family_counts"])
            self.assertNotIn("other", first["source"]["selection"]["family_counts"])
            self.assertFalse(first["source"]["heldout_used"])
            self.assertEqual(
                first["source"]["producer_root"], str(producer_root.resolve())
            )
            self.assertEqual(
                first["source"]["selection"]["algorithm"],
                "producer-synth-records-stratified-sample/v1",
            )
            self.assertEqual(module.calls[0][1:], (
                "train", "rationale", 1024, 6, 71, "structured_qa",
            ))
            self.assertEqual(
                len({record["id"] for record in first["records"]}), 256,
            )
            self.assertTrue(all(record["split"] == "train" for record in first["records"]))
            self.assertTrue(all(
                Path(record["image"]).is_relative_to(synth_root / "train" / "img")
                for record in first["records"]
            ))
            self.assertTrue(all(
                len(record["annotation_sha256"]) == 64
                and len(record["image_sha256"]) == 64
                for record in first["records"]
            ))

    def test_replays_release_split_and_selects_only_balanced_training(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(30):
                _write_record(root, f"p{index:02d}", "phone", 1000 + index)
                _write_record(root, f"a{index:02d}", "address", 2000 + index)
            manifest = MODULE.build_manifest(
                root / "annotations",
                root / "images",
                seed=71,
                validation_fraction=0.1,
                per_family=4,
            )
            inventory, training, validation = MODULE.split_annotation_paths(
                root / "annotations", seed=71, validation_fraction=0.1
            )
            selected = {record["id"] for record in manifest["records"]}
            self.assertEqual(manifest["format"], "volvox-calibration-records/v1")
            families = [record["family"] for record in manifest["records"]]
            self.assertEqual(families.count("phone"), 4)
            self.assertEqual(families.count("address"), 4)
            self.assertLessEqual(selected, {path.stem for path in training})
            self.assertTrue(selected.isdisjoint(path.stem for path in validation))
            self.assertEqual(manifest["source"]["annotation_count"], len(inventory))
            self.assertTrue(all(record["split"] == "train" for record in manifest["records"]))
            self.assertTrue(all(
                len(record["annotation_sha256"]) == 64
                for record in manifest["records"]
            ))
            self.assertTrue(all(
                len(record["image_sha256"]) == 64
                for record in manifest["records"]
            ))

    def test_is_deterministic_and_rejects_invalid_fraction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(8):
                _write_record(root, f"p{index:02d}", "phone", 3000 + index)
                _write_record(root, f"a{index:02d}", "address", 4000 + index)
            first = MODULE.build_manifest(
                root / "annotations", root / "images", per_family=2
            )
            second = MODULE.build_manifest(
                root / "annotations", root / "images", per_family=2
            )
            self.assertEqual(first, second)
            with self.assertRaisesRegex(ValueError, "strictly between"):
                MODULE.split_annotation_paths(
                    root / "annotations", seed=71, validation_fraction=0
                )

    def test_emits_one_explicit_specialization_profile_without_cross_family_records(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(12):
                _write_record(root, f"p{index:02d}", "phone", 5000 + index)
                _write_record(root, f"a{index:02d}", "address", 6000 + index)
            manifest = MODULE.build_manifest(
                root / "annotations",
                root / "images",
                per_family=4,
                families=("address",),
            )
            self.assertEqual(manifest["required_families"], ["address"])
            self.assertEqual(manifest["source"]["selection"]["families"], ["address"])
            self.assertEqual(len(manifest["records"]), 4)
            self.assertTrue(all(
                record["family"] == "address" for record in manifest["records"]
            ))
            with self.assertRaisesRegex(ValueError, "unique supported"):
                MODULE.build_manifest(
                    root / "annotations", root / "images", families=(),
                )
            with self.assertRaisesRegex(ValueError, "unsupported"):
                MODULE.build_manifest(
                    root / "annotations", root / "images", families=("other",),
                )


if __name__ == "__main__":
    unittest.main()
