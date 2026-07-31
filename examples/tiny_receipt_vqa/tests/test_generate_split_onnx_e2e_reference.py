from __future__ import annotations

import importlib
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
TOOLS = EXAMPLE_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

try:
    oracle = importlib.import_module("generate_split_onnx_e2e_reference")
except ImportError as error:  # pragma: no cover - environment dependency.
    raise unittest.SkipTest(
        f"TinyReceipt split oracle dependencies unavailable: {error}"
    )


class AtomicReferencePublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.document = {
            "schema": "volvoxai.test-reference/v1",
            "source": "references/source.json",
            "values": [1, 2, 3],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_publishes_complete_document_and_creates_real_parents(self) -> None:
        destination = self.root / "new" / "nested" / "reference.json"

        oracle._atomic_write_new(destination, self.document)

        self.assertEqual(
            json.loads(destination.read_text(encoding="utf-8")),
            self.document,
        )
        self.assertTrue(destination.read_bytes().endswith(b"\n"))
        self.assertEqual(
            list(destination.parent.glob(f".{destination.name}.*.tmp")),
            [],
        )

    def test_existing_target_is_never_overwritten(self) -> None:
        destination = self.root / "reference.json"
        original = b"existing-reference\n"
        destination.write_bytes(original)

        with self.assertRaisesRegex(oracle.OracleFailure, "output already exists"):
            oracle._atomic_write_new(destination, self.document)

        self.assertEqual(destination.read_bytes(), original)
        self.assertEqual(
            list(destination.parent.glob(f".{destination.name}.*.tmp")),
            [],
        )

    def test_rejects_symlinked_output_parent_component(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        linked_parent = self.root / "linked-parent"
        try:
            linked_parent.symlink_to(outside, target_is_directory=True)
        except OSError as error:  # pragma: no cover - platform permission.
            self.skipTest(f"symbolic links unavailable: {error}")
        destination = linked_parent / "nested" / "reference.json"

        with self.assertRaisesRegex(oracle.OracleFailure, "symbolic link"):
            oracle._atomic_write_new(destination, self.document)

        self.assertFalse((outside / "nested").exists())

    def test_workload_uses_validated_tokenizer_instead_of_character_indices(self) -> None:
        tokenizer = mock.Mock()
        tokenizer.encode.return_value = [1038, 54, 1124, 2]

        image, token_ids, question = oracle._workload(tokenizer)

        tokenizer.encode.assert_called_once_with(
            oracle.PROMPT,
            add_eos=True,
            max_len=oracle.SEQUENCE_LENGTH,
        )
        self.assertEqual(token_ids, [1038, 54, 1124, 2])
        self.assertEqual(question.shape, (1, oracle.SEQUENCE_LENGTH))
        self.assertEqual(question[0, :4].tolist(), token_ids)
        self.assertEqual(image.shape, (1, 1, oracle.IMAGE_HEIGHT, oracle.IMAGE_WIDTH))

    def test_package_identity_uses_validated_tokenizer_digest(self) -> None:
        package_root = self.root / "package"
        package_root.mkdir()

        def asset(relative: str, contents: bytes) -> dict[str, object]:
            path = package_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
            return {
                "path": relative,
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }

        records = {
            "encoderGraph": asset("encoder/graph.json", b"encoder-graph"),
            "encoderWeights": asset("encoder/model.safetensors", b"encoder-weights"),
            "decoderGraph": asset("decoder/graph.json", b"decoder-graph"),
            "decoderWeights": asset("decoder/model.safetensors", b"decoder-weights"),
        }
        vocabulary = {"fixture": "validated-by-tokenizer"}
        vocabulary_record = asset(
            "vocab.json",
            json.dumps(vocabulary, separators=(",", ":")).encode("utf-8"),
        )
        source_hashes = {
            "manifest": "1" * 64,
            "encoder": "2" * 64,
            "decoder": "3" * 64,
        }
        manifest = {
            "format": oracle.PACKAGE_FORMAT,
            "source": {
                "format": oracle.SOURCE_FORMAT,
                "variant": "int8-w8a8",
                "manifest": {"sha256": source_hashes["manifest"]},
                "encoder_onnx": {"sha256": source_hashes["encoder"]},
                "decoder_onnx": {"sha256": source_hashes["decoder"]},
            },
            "graphs": {
                "encoder": {
                    "graph": records["encoderGraph"],
                    "weights": records["encoderWeights"],
                },
                "decoder": {
                    "graph": records["decoderGraph"],
                    "weights": records["decoderWeights"],
                },
            },
            "assets": {"vocab": vocabulary_record},
            "tokenizer": {"type": "byte_fallback_bpe"},
        }
        (package_root / "package_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        source = {
            "selected_file_keys": {"encoder": "encoder", "decoder": "decoder"},
            "hashes": source_hashes,
            "vocab": vocabulary,
            "vocab_size": 1536,
        }
        tokenizer = mock.Mock(
            vocab_size=1536,
            package_vocabulary_sha256="a" * 64,
        )

        with mock.patch.object(
            oracle.TinyReceiptTokenizer,
            "from_documents",
            return_value=tokenizer,
        ):
            identity, returned_tokenizer = oracle._package_identity(
                package_root,
                source,
            )

        self.assertIs(returned_tokenizer, tokenizer)
        self.assertEqual(identity["vocabularySha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
