from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock


_TOOL_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "run_native_split_e2e.py"
)
_SPEC = importlib.util.spec_from_file_location("run_native_split_e2e", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
native_e2e = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = native_e2e
_SPEC.loader.exec_module(native_e2e)

from tiny_receipt_tokenizer import (  # noqa: E402
    ATOMIC_TOKENS,
    BYTE_TOKENS,
    SPECIAL_TOKENS,
    tokenizer_fingerprint,
)


_ROUTER = [
    11.88134765625,
    -1.9163464307785034,
    5.749039173126221,
    -18.39692497253418,
    1.9163464307785034,
    -8.815193176269531,
    -17.630386352539062,
    -17.630386352539062,
]
_TOKENS = [29, 66, 69, 65]


_FAKE_CLI = r"""#!/usr/bin/env python3
import array
import json
import os
import sys
from pathlib import Path


def parse():
    values = sys.argv[1:]
    if len(values) < 2 or values[0] != "run":
        raise SystemExit(2)
    graph_path = Path(values[1])
    options = {"input": [], "output": []}
    backend = None
    index = 2
    while index < len(values):
        name = values[index]
        if name in ("--cpu", "--cuda"):
            backend = name[2:]
            index += 1
            continue
        if index + 1 >= len(values):
            raise SystemExit(3)
        value = values[index + 1]
        key = name[2:]
        if key in ("input", "output"):
            options[key].append(value)
        else:
            options[key] = value
        index += 2
    if backend is None:
        raise SystemExit(4)
    return graph_path, options, backend


def bindings(values):
    result = {}
    for value in values:
        name, path = value.split("=", 1)
        result[name] = Path(path)
    return result


def write_f32(path, values):
    array.array("f", values).tofile(path.open("wb"))


def write_i32(path, values):
    array.array("i", values).tofile(path.open("wb"))


def evidence(backend, graph, kind):
    shapes = {
        "encoder": [
            ([1, 402, 320], "float32"),
            ([1, 402], "int32"),
            ([1, 8], "float32"),
            ([1], "int32"),
        ],
        "decoder": [
            ([1, 192], "int32")
            if decoder_token_ids_output
            else ([1, 192, vocab_size], "float32")
        ],
    }[kind]
    outputs = []
    for name, (shape, dtype) in zip(graph["outputs"], shapes):
        elements = 1
        for dimension in shape:
            elements *= dimension
        outputs.append({
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "location": "host",
            "byteLength": elements * 4,
        })
    fallback = os.environ.get("FAKE_NATIVE_FALLBACK") == "1"
    route = {
        "tierFallback": False,
        "operator": {
            "attestation": "reported",
            "used": fallback,
            "offendingNode": "fixture-node" if fallback else None,
        },
    }
    revisions = {
        "topologyRevision": "1",
        "weightRevision": "1",
        "weightRevisionId": "native-weight-1",
        "adapterRevisionId": None,
        "adapterRevisionIds": [],
    }
    execution_revisions = dict(revisions)
    execution_revisions["adapterRevisionIds"] = [None]
    if os.environ.get("FAKE_NATIVE_NULL_DEVICE") == "1":
        device = None
    else:
        device_name = "fixture-device"
        if (
            os.environ.get("FAKE_NATIVE_DEVICE_BY_GRAPH") == "1"
            and kind == "decoder"
        ):
            device_name = "fixture-decoder-device"
        device = {"backend": backend, "device": device_name}
    return {
        "schema": "volvoxai.runtime-evidence",
        "version": 1,
        "compilation": {
            "compilationId": "native-compiled-1",
            "policy": {
                "mode": "require",
                "backend": backend,
                "operatorFallback": "forbid",
            },
            "selectedBackend": backend,
            "selectedDevice": device,
            "definitionId": "native-graph-1",
            "revisions": revisions,
            "route": route,
        },
        "execution": {
            "executionId": "native-execution-1",
            "contextId": "native-context-1",
            "backend": backend,
            "device": device,
            "outcome": "success",
            "revisions": execution_revisions,
            "route": route,
            "decodeState": {
                "operation": "execute",
                "mode": None,
                "cacheState": "not-applicable",
                "cacheGeneration": None,
                "position": None,
            },
        },
        "stableResult": {
            "outputs": outputs,
            "freshCallerOwnedReads": True,
            "readableAfterContextClose": True,
            "contextClosedBeforeResult": True,
            "resultClosedAfterVerification": True,
        },
    }


graph_path, options, backend = parse()
graph = json.loads(graph_path.read_text(encoding="utf-8"))
kind = graph_path.parent.name
inputs = bindings(options["input"])
outputs = bindings(options["output"])
vocabulary = json.loads(
    (graph_path.parents[1] / "vocab.json").read_text(encoding="utf-8")
)["itos"]
stoi = {token: index for index, token in enumerate(vocabulary)}
vocab_size = len(vocabulary)
decoder_token_ids_output = any(path.suffix == ".i32" for path in outputs.values())

if backend == "cuda":
    cuda_mode = os.environ.get("FAKE_CUDA_BANNER", "hardware")
    if cuda_mode != "missing":
        if cuda_mode == "malformed":
            print("[CUDA] device 0: NVIDIA GPU (compute eight-six)", file=sys.stderr)
            cuda_mode = "malformed-only"
        cuda_index = 0
        cuda_name = "NVIDIA GeForce RTX 3090"
        cuda_compute = "8.6"
        if cuda_mode == "software":
            cuda_name = "Google SwiftShader CPU"
        elif cuda_mode == "mismatch" and kind == "decoder":
            cuda_index = 1
            cuda_name = "NVIDIA A100-SXM4-40GB"
            cuda_compute = "8.0"
        elif cuda_mode == "path":
            cuda_name = "/private/device"
        if cuda_mode != "malformed-only":
            banner = (
                f"[CUDA] device {cuda_index}: {cuda_name} "
                f"(compute {cuda_compute})"
            )
            print(banner, file=sys.stderr)
            if cuda_mode == "duplicate":
                print(banner, file=sys.stderr)
            elif cuda_mode == "differing-duplicate":
                print(
                    "[CUDA] device 1: NVIDIA A100-SXM4-40GB (compute 8.0)",
                    file=sys.stderr,
                )

if kind == "encoder":
    if set(inputs) != {"enc_image_0", "enc_question_0", "enc_family_0"}:
        raise SystemExit(5)
    if set(outputs) != {
        "enc_memory_0", "enc_mask_0", "enc_router_0", "enc_selected_0",
    }:
        raise SystemExit(6)
    write_f32(outputs["enc_memory_0"], [0.0] * (402 * 320))
    write_i32(outputs["enc_mask_0"], [0] * 402)
    write_f32(outputs["enc_router_0"], [
        11.88134765625,
        -1.9163464307785034,
        5.749039173126221,
        -18.39692497253418,
        1.9163464307785034,
        -8.815193176269531,
        -17.630386352539062,
        -17.630386352539062,
    ])
    write_i32(outputs["enc_selected_0"], [0])
elif kind == "decoder":
    expected_inputs = {"dec_ids_0", "dec_memory_0", "dec_mask_0", "dec_family_0"}
    keep_name = "@runtime/fixture.keep"
    if keep_name in graph["inputs"]:
        expected_inputs.add(keep_name)
    if set(inputs) != expected_inputs:
        raise SystemExit(7)
    if set(outputs) not in ({"dec_logits_0"}, {"dec_token_ids_0"}):
        raise SystemExit(8)
    decoder_ids = array.array("i")
    decoder_ids.fromfile(inputs["dec_ids_0"].open("rb"), 192)
    prefix = 1
    while prefix < 192 and decoder_ids[prefix] != 0:
        prefix += 1
    step = prefix - 1
    if vocab_size == 760:
        tokens = [29, 66, 69, 65]
    else:
        tokens = [stoi["<field>"], stoi["phone"], stoi[" "], stoi["number"]]
    if keep_name in inputs:
        keep = array.array("i")
        keep.fromfile(inputs[keep_name].open("rb"), 192)
        if list(keep[:prefix]) != [1] * prefix or any(keep[prefix:]):
            raise SystemExit(10)
    if "dec_token_ids_0" in outputs:
        selected = [0] * 192
        selected[prefix - 1] = tokens[step]
        write_i32(outputs["dec_token_ids_0"], selected)
    else:
        logits = [0.0] * (192 * vocab_size)
        logits[(prefix - 1) * vocab_size + tokens[step]] = 10.0
        write_f32(outputs["dec_logits_0"], logits)
else:
    raise SystemExit(9)

Path(options["report-json"]).write_text(
    json.dumps(evidence(backend, graph, kind)),
    encoding="utf-8",
)
"""


def _make_bpe_vocabulary_document() -> dict[str, object]:
    vocabulary = list(SPECIAL_TOKENS + ATOMIC_TOKENS + BYTE_TOKENS)
    lexical_tokens = [
        " ",
        "p",
        "h",
        "o",
        "n",
        "e",
        "u",
        "m",
        "b",
        "r",
        "l",
        "a",
        "s",
        "t",
        "c",
        "f",
        "é",
        "ph",
        "pho",
        "phon",
        "phone",
        "nu",
        "num",
        "numb",
        "numbe",
        "number",
        "la",
        "las",
        "last",
        "on",
        "one",
        "ca",
        "caf",
        "café",
    ]
    vocabulary.extend(lexical_tokens)
    merges = [
        ["p", "h"],
        ["ph", "o"],
        ["pho", "n"],
        ["phon", "e"],
        ["n", "u"],
        ["nu", "m"],
        ["num", "b"],
        ["numb", "e"],
        ["numbe", "r"],
        ["l", "a"],
        ["la", "s"],
        ["las", "t"],
        ["o", "n"],
        ["on", "e"],
        ["c", "a"],
        ["ca", "f"],
        ["caf", "é"],
    ]
    unused_tokens = [
        f"<unused_{index:04d}>"
        for index in range(1536 - len(vocabulary))
    ]
    vocabulary.extend(unused_tokens)
    document: dict[str, object] = {
        "type": "byte_fallback_bpe",
        "version": 1,
        "vocab_size": 1536,
        "itos": vocabulary,
        "merges": merges,
        "normalization": "NFC",
        "atomic_tokens": list(ATOMIC_TOKENS),
        "byte_tokens": list(BYTE_TOKENS),
        "unused_tokens": unused_tokens,
        "special_tokens": {
            "pad": "<pad>",
            "bos": "<bos>",
            "eos": "<eos>",
            "unk": "<unk>",
        },
    }
    document["tokenizer_hash"] = tokenizer_fingerprint(document)
    return document


class NativeSplitE2ETests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.package = self.root / "package"
        self.package.mkdir()
        self.reference = self.root / "reference.json"
        self.binary = self.root / "fake-volvoxai"
        self.binary.write_text(_FAKE_CLI, encoding="utf-8")
        self.binary.chmod(
            self.binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )
        self._make_package_and_reference()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_asset(self, relative: str, value: bytes) -> dict[str, object]:
        path = self.package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return {
            "path": relative,
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }

    def _make_vocabulary(self) -> list[str]:
        vocabulary = [f"<token-{index}>" for index in range(760)]
        vocabulary[:4] = ["<pad>", "<bos>", "<eos>", "<unk>"]
        for token, index in {
            " ": 4,
            "<": 29,
            "a": 61,
            "b": 62,
            "e": 65,
            "f": 66,
            "h": 68,
            "i": 69,
            "l": 72,
            "m": 73,
            "n": 74,
            "o": 75,
            "p": 76,
            "r": 78,
            "s": 79,
            "t": 80,
            "u": 81,
        }.items():
            vocabulary[index] = token
        return vocabulary

    def _make_package_and_reference(self) -> None:
        vocabulary = self._make_vocabulary()
        config = self._write_asset(
            "config.json",
            json.dumps({"fixture": True}, separators=(",", ":")).encode(),
        )
        vocab = self._write_asset(
            "vocab.json",
            json.dumps(
                {"itos": vocabulary},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode(),
        )
        encoder_graph_document = {
            "format": "volvox-graph/v1",
            "inputs": {
                "enc_image_0": {},
                "enc_question_0": {},
                "enc_family_0": {},
            },
            "outputs": [
                "enc_memory_0",
                "enc_mask_0",
                "enc_router_0",
                "enc_selected_0",
            ],
            "nodes": [],
        }
        decoder_graph_document = {
            "format": "volvox-graph/v1",
            "inputs": {
                "dec_ids_0": {},
                "dec_memory_0": {},
                "dec_mask_0": {},
                "dec_family_0": {},
            },
            "outputs": ["dec_logits_0"],
            "nodes": [],
        }
        encoder_graph = self._write_asset(
            "encoder/graph.json",
            json.dumps(encoder_graph_document, separators=(",", ":")).encode(),
        )
        encoder_weights = self._write_asset(
            "encoder/model.safetensors", b"fixture-encoder-weights"
        )
        encoder_report = self._write_asset(
            "encoder/export_report.json", b'{"fixture":true}'
        )
        decoder_graph = self._write_asset(
            "decoder/graph.json",
            json.dumps(decoder_graph_document, separators=(",", ":")).encode(),
        )
        decoder_weights = self._write_asset(
            "decoder/model.safetensors", b"fixture-decoder-weights"
        )
        decoder_report = self._write_asset(
            "decoder/export_report.json", b'{"fixture":true}'
        )
        manifest = {
            "format": native_e2e.PACKAGE_FORMAT,
            "variant": {"requested": "int8-w8a8"},
            "assets": {"config": config, "vocab": vocab},
            "tokenizer": {
                "type": "char-vocab",
                "version": 1,
                "itos_key": "itos",
                "token_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3},
            },
            "preprocessing": {
                "layout": "NCHW",
                "shape": [1, 1, 320, 672],
                "color": "grayscale",
            },
            "families": {
                "auto_id": -1,
                "ordered_names": list(native_e2e.FAMILY_ORDER),
                "name_to_id": {
                    name: index
                    for index, name in enumerate(native_e2e.FAMILY_ORDER)
                },
            },
            "generation": {
                "strategy": "greedy-autoregressive",
                "decoder_input_length": 192,
                "maximum_new_tokens": 191,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
                "logits_row": "prefix_length_minus_one",
                "tie_policy": "first-index",
            },
            "graphs": {
                "encoder": {
                    "graph": encoder_graph,
                    "weights": encoder_weights,
                    "export_report": encoder_report,
                    "inputs": {
                        "image": "enc_image_0",
                        "question_ids": "enc_question_0",
                        "family_ids": "enc_family_0",
                    },
                    "outputs": {
                        "memory": "enc_memory_0",
                        "memory_padding_mask": "enc_mask_0",
                        "router_logits": "enc_router_0",
                        "selected_family_ids": "enc_selected_0",
                    },
                },
                "decoder": {
                    "graph": decoder_graph,
                    "weights": decoder_weights,
                    "export_report": decoder_report,
                    "inputs": {
                        "decoder_input_ids": "dec_ids_0",
                        "memory": "dec_memory_0",
                        "memory_padding_mask": "dec_mask_0",
                        "family_ids": "dec_family_0",
                    },
                    "outputs": {"logits": "dec_logits_0"},
                },
            },
            "mask_semantics": {
                "memory_padding_mask": "nonzero_means_blocked"
            },
        }
        (self.package / "package_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        package_identity = {
            "format": native_e2e.PACKAGE_FORMAT,
            "assets": {
                "encoderGraph": encoder_graph["sha256"],
                "encoderWeights": encoder_weights["sha256"],
                "decoderGraph": decoder_graph["sha256"],
                "decoderWeights": decoder_weights["sha256"],
            },
            "vocabularySha256": hashlib.sha256(
                native_e2e._canonical_json(vocabulary).encode("utf-8")
            ).hexdigest(),
        }
        text = "<fie"
        reference = {
            "schema": native_e2e.REFERENCE_SCHEMA,
            "provenance": {
                "kind": "onnx-runtime-oracle",
                "sourceFormat": "tiny_receipt_vqa_split_onnx_v1",
                "sourceVariant": "int8-w8a8",
                "provider": "fixture ONNX Runtime CPUExecutionProvider",
                "description": "Focused fake-CLI orchestration fixture.",
            },
            "package": package_identity,
            "workload": {
                "id": native_e2e.WORKLOAD_ID,
                "prompt": native_e2e.PROMPT,
                "family": "auto",
                "maxNewTokens": 4,
                "minimumDecoderSteps": 2,
                "image": {
                    "generator": "q=(17*x+29*y+7*(x^y))&255; f32=(q-128)/128",
                    "dtype": "float32",
                    "shape": [1, 1, 320, 672],
                    "byteOrder": "little",
                    "sha256": native_e2e.IMAGE_SHA256,
                },
            },
            "expected": {
                "family": "phone",
                "familyId": 0,
                "requestedFamily": "auto",
                "requestedFamilyId": -1,
                "questionTokenIds": [
                    76, 68, 75, 74, 65, 4, 74, 81, 73, 62, 65,
                    78, 4, 72, 61, 79, 80, 4, 75, 74, 65, 2,
                ],
                "questionTokenIdsSha256": native_e2e.QUESTION_TOKEN_IDS_SHA256,
                "tokenIds": _TOKENS,
                "tokenIdsSha256": hashlib.sha256(
                    native_e2e._int32_bytes(_TOKENS)
                ).hexdigest(),
                "textUtf8Sha256": hashlib.sha256(text.encode()).hexdigest(),
                "stoppedAtEos": False,
                "routerLogits": {
                    "values": _ROUTER,
                    "sha256": hashlib.sha256(
                        native_e2e._float32_bytes(_ROUTER)
                    ).hexdigest(),
                    "summary": native_e2e._numeric_summary(
                        _ROUTER, "fixture router"
                    ),
                    "atol": 0.0001,
                    "rtol": 0.00001,
                },
            },
        }
        self.reference.write_text(json.dumps(reference), encoding="utf-8")

    def _reference_document(self) -> dict[str, object]:
        return json.loads(self.reference.read_text(encoding="utf-8"))

    def _convert_fixture_to_bpe(self) -> native_e2e.Package:
        vocabulary_document = _make_bpe_vocabulary_document()
        vocab_bytes = json.dumps(
            vocabulary_document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        vocab_path = self.package / "vocab.json"
        vocab_path.write_bytes(vocab_bytes)

        manifest_path = self.package / "package_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["assets"]["vocab"] = {
            "path": "vocab.json",
            "bytes": len(vocab_bytes),
            "sha256": hashlib.sha256(vocab_bytes).hexdigest(),
        }
        manifest["tokenizer"] = {
            "type": "byte_fallback_bpe",
            "version": 1,
            "vocab_size": 1536,
            "normalization": "NFC",
            "tokenizer_hash": vocabulary_document["tokenizer_hash"],
            "itos_key": "itos",
            "merges_key": "merges",
            "token_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3},
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        package = native_e2e.load_package(self.package)
        reference = self._reference_document()
        reference["package"]["vocabularySha256"] = package.vocabulary_sha256
        question_ids = package.tokenizer.encode(
            native_e2e.PROMPT,
            add_eos=True,
            max_len=native_e2e.QUESTION_LENGTH,
        )
        generated = [
            package.tokenizer.stoi["<field>"],
            package.tokenizer.stoi["phone"],
            package.tokenizer.stoi[" "],
            package.tokenizer.stoi["number"],
        ]
        text = package.tokenizer.decode(generated)
        reference["expected"]["questionTokenIds"] = question_ids
        reference["expected"]["questionTokenIdsSha256"] = hashlib.sha256(
            native_e2e._int32_bytes(question_ids)
        ).hexdigest()
        reference["expected"]["tokenIds"] = generated
        reference["expected"]["tokenIdsSha256"] = hashlib.sha256(
            native_e2e._int32_bytes(generated)
        ).hexdigest()
        reference["expected"]["textUtf8Sha256"] = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()
        self.reference.write_text(json.dumps(reference), encoding="utf-8")
        return package

    def _convert_fixture_to_token_ids_abi(self) -> native_e2e.Package:
        graph_path = self.package / "decoder" / "graph.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        graph["inputs"]["@runtime/fixture.keep"] = {}
        graph["outputs"] = ["dec_token_ids_0"]
        graph_bytes = json.dumps(graph, separators=(",", ":")).encode("utf-8")
        graph_path.write_bytes(graph_bytes)

        manifest_path = self.package / "package_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["generation"] = {
            "strategy": "greedy-autoregressive",
            "decoder_input_length": 192,
            "maximum_new_tokens": 191,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "decoder_output": "token_ids",
            "token_ids_row": "prefix_length_minus_one",
            "tie_policy": "first-index",
        }
        manifest["graphs"]["decoder"]["graph"] = {
            "path": "decoder/graph.json",
            "bytes": len(graph_bytes),
            "sha256": hashlib.sha256(graph_bytes).hexdigest(),
        }
        manifest["graphs"]["decoder"]["inputs"]["v4_keep"] = (
            "@runtime/fixture.keep"
        )
        manifest["graphs"]["decoder"]["outputs"] = {
            "token_ids": "dec_token_ids_0"
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        package = native_e2e.load_package(self.package)
        reference = self._reference_document()
        reference["package"]["assets"]["decoderGraph"] = (
            package.decoder.graph.sha256
        )
        self.reference.write_text(json.dumps(reference), encoding="utf-8")
        return package

    def _run(
        self,
        *,
        backend: str,
        artifact_name: str,
        require_cuda_device: bool = False,
    ) -> dict[str, object]:
        artifacts = self.root / artifact_name
        artifacts.mkdir()
        return native_e2e.run_native_split_e2e(
            package_directory=self.package,
            binary=self.binary,
            backend=backend,
            reference_path=self.reference,
            artifact_directory=artifacts,
            timeout_seconds=10,
            require_cuda_device=require_cuda_device,
        )

    def test_runs_encoder_and_four_decoder_cli_forwards_path_safely(self) -> None:
        artifacts = self.root / "artifacts"
        artifacts.mkdir()
        result = native_e2e.run_native_split_e2e(
            package_directory=self.package,
            binary=self.binary,
            backend="cpu",
            reference_path=self.reference,
            artifact_directory=artifacts,
            timeout_seconds=10,
        )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["output"]["tokenIds"], _TOKENS)
        self.assertEqual(result["output"]["text"], "<fie")
        self.assertEqual(result["provider"]["encoderExecutions"], 1)
        self.assertEqual(result["provider"]["decoderExecutions"], 4)
        self.assertEqual(result["provider"]["invocationCount"], 5)
        self.assertEqual(
            result["provider"]["devices"],
            [{"backend": "cpu", "device": "fixture-device"}],
        )
        self.assertEqual(
            [(item["stage"], item["step"]) for item in result["provider"]["evidence"]],
            [
                ("encoder", None),
                ("decoder", 0),
                ("decoder", 1),
                ("decoder", 2),
                ("decoder", 3),
            ],
        )
        self.assertTrue(result["reference"]["exactTokenIds"])
        self.assertTrue(result["reference"]["routerSummaryWithinTolerance"])
        serialized = json.dumps(result)
        self.assertNotIn(str(self.root), serialized)
        self.assertEqual(result["lifecycle"]["temporaryArtifacts"], "retained")
        self.assertTrue((artifacts / "decoder_03_evidence.json").is_file())
        package = native_e2e.load_package(self.package)
        self.assertEqual(
            package.vocabulary_sha256,
            hashlib.sha256(
                native_e2e._canonical_json(list(package.vocabulary)).encode("utf-8")
            ).hexdigest(),
        )

    def test_runs_bpe1536_package_with_dynamic_logits_stride(self) -> None:
        package = self._convert_fixture_to_bpe()
        result = self._run(backend="cpu", artifact_name="bpe-artifacts")

        self.assertEqual(package.tokenizer.kind, "byte_fallback_bpe")
        self.assertEqual(len(package.vocabulary), 1536)
        self.assertEqual(
            package.vocabulary_sha256,
            package.tokenizer.tokenizer_hash,
        )
        self.assertEqual(
            result["output"]["questionTokenIds"],
            [
                package.tokenizer.stoi["phone"],
                package.tokenizer.stoi[" "],
                package.tokenizer.stoi["number"],
                package.tokenizer.stoi[" "],
                package.tokenizer.stoi["last"],
                package.tokenizer.stoi[" "],
                package.tokenizer.stoi["one"],
                package.token_ids["eos"],
            ],
        )
        self.assertEqual(result["output"]["text"], "<field>phone number")
        self.assertEqual(
            (self.root / "bpe-artifacts" / "logits.f32").stat().st_size,
            192 * 1536 * 4,
        )

    def test_bpe1536_exact_boundaries_nfc_and_utf8_fallback(self) -> None:
        package = self._convert_fixture_to_bpe()
        tokenizer = package.tokenizer
        encoded = tokenizer.encode("cafe\u0301 7<field>🙂")

        self.assertEqual(
            encoded,
            [
                tokenizer.stoi["café"],
                tokenizer.stoi[" "],
                tokenizer.stoi["7"],
                tokenizer.stoi["<field>"],
                tokenizer.stoi["<0xF0>"],
                tokenizer.stoi["<0x9F>"],
                tokenizer.stoi["<0x99>"],
                tokenizer.stoi["<0x82>"],
            ],
        )
        self.assertEqual(tokenizer.decode(encoded), "café 7<field>🙂")

    def test_runs_canonical_token_ids_abi_with_aliased_v4_keep(self) -> None:
        self._convert_fixture_to_bpe()
        package = self._convert_fixture_to_token_ids_abi()
        result = self._run(
            backend="cpu",
            artifact_name="token-ids-artifacts",
        )

        self.assertEqual(package.decoder_output, "token_ids")
        self.assertEqual(
            package.decoder.inputs["v4_keep"],
            "@runtime/fixture.keep",
        )
        self.assertEqual(result["output"]["text"], "<field>phone number")
        artifacts = self.root / "token-ids-artifacts"
        self.assertEqual((artifacts / "token_ids.i32").stat().st_size, 192 * 4)
        self.assertFalse((artifacts / "logits.f32").exists())

    def test_bpe1536_rejects_mismatched_tokenizer_hash(self) -> None:
        self._convert_fixture_to_bpe()
        manifest_path = self.package / "package_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["tokenizer"]["tokenizer_hash"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(native_e2e.E2EError, "tokenizer_hash"):
            native_e2e.load_package(self.package)

    def test_reference_recomputes_all_internal_hashes_and_router_summary(self) -> None:
        mutations = (
            (
                "image",
                "reference.workload.image.sha256",
                lambda value: value["workload"]["image"].__setitem__(
                    "sha256", "0" * 64
                ),
            ),
            (
                "question IDs",
                "questionTokenIdsSha256",
                lambda value: value["expected"]["questionTokenIds"].__setitem__(
                    0, 75
                ),
            ),
            (
                "generated token IDs",
                "tokenIdsSha256",
                lambda value: value["expected"]["tokenIds"].__setitem__(0, 30),
            ),
            (
                "router values",
                "router SHA-256",
                lambda value: value["expected"]["routerLogits"]["values"].__setitem__(
                    0, 10.0
                ),
            ),
            (
                "router summary",
                "router summary sum",
                lambda value: value["expected"]["routerLogits"]["summary"].__setitem__(
                    "sum",
                    value["expected"]["routerLogits"]["summary"]["sum"] + 1.0,
                ),
            ),
        )
        for label, message, mutate in mutations:
            with self.subTest(label=label):
                reference = self._reference_document()
                mutate(reference)
                with self.assertRaisesRegex(native_e2e.E2EError, message):
                    native_e2e._validate_reference(reference)

    def test_committed_reference_passes_internal_integrity_recomputation(self) -> None:
        reference = native_e2e._read_json(
            native_e2e.DEFAULT_REFERENCE,
            "committed E2E reference",
        )
        self.assertIs(native_e2e._validate_reference(reference), reference)

    def test_rejects_null_or_changing_runtime_device_evidence(self) -> None:
        cases = (
            ("FAKE_NATIVE_NULL_DEVICE", "non-empty string map"),
            ("FAKE_NATIVE_DEVICE_BY_GRAPH", "changed between CLI invocations"),
        )
        for environment, message in cases:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, {environment: "1"}):
                    with self.assertRaisesRegex(native_e2e.E2EError, message):
                        self._run(
                            backend="cpu",
                            artifact_name=f"runtime-device-{environment}",
                        )

    def test_required_cuda_device_banner_is_attested_path_safely(self) -> None:
        for mode in ("hardware", "duplicate"):
            with self.subTest(mode=mode):
                with mock.patch.dict(
                    os.environ,
                    {"FAKE_CUDA_BANNER": mode},
                ):
                    result = self._run(
                        backend="cuda",
                        artifact_name=f"cuda-device-{mode}",
                        require_cuda_device=True,
                    )
                self.assertEqual(
                    result["provider"]["cudaDevice"],
                    {
                        "name": "NVIDIA GeForce RTX 3090",
                        "computeCapability": "8.6",
                    },
                )
                self.assertNotIn("index", result["provider"]["cudaDevice"])

    def test_required_cuda_device_rejects_bad_or_changing_banners(self) -> None:
        cases = (
            ("missing", "banner is missing"),
            ("malformed", "malformed CUDA"),
            ("differing-duplicate", "banners disagree"),
            ("software", "software device"),
            ("path", "non-path-safe"),
            ("mismatch", "changed between CLI invocations"),
        )
        for mode, message in cases:
            with self.subTest(mode=mode):
                with mock.patch.dict(
                    os.environ,
                    {"FAKE_CUDA_BANNER": mode},
                ):
                    with self.assertRaisesRegex(native_e2e.E2EError, message):
                        self._run(
                            backend="cuda",
                            artifact_name=f"cuda-device-{mode}",
                            require_cuda_device=True,
                        )

    def test_require_cuda_device_rejects_cpu_backend(self) -> None:
        with self.assertRaisesRegex(
            native_e2e.E2EError,
            "requires the CUDA backend",
        ):
            self._run(
                backend="cpu",
                artifact_name="cuda-device-on-cpu",
                require_cuda_device=True,
            )

    def test_json_output_is_atomic_and_new_only_under_contention(self) -> None:
        output_directory = self.root / "json-output"
        output_directory.mkdir()
        destination = output_directory / "result.json"
        barrier = threading.Barrier(2)

        def publish(identifier: int) -> tuple[int, bool]:
            barrier.wait()
            try:
                native_e2e._write_result(
                    {"schema": "fixture", "identifier": identifier},
                    str(destination),
                )
            except native_e2e.E2EError:
                return identifier, False
            return identifier, True

        with ThreadPoolExecutor(max_workers=2) as executor:
            attempts = list(executor.map(publish, (1, 2)))
        winners = [identifier for identifier, succeeded in attempts if succeeded]
        self.assertEqual(len(winners), 1)
        self.assertEqual(
            json.loads(destination.read_text(encoding="utf-8")),
            {"schema": "fixture", "identifier": winners[0]},
        )
        self.assertEqual(
            [path.name for path in output_directory.iterdir()],
            ["result.json"],
        )

    def test_json_output_rejects_parent_component_symlink(self) -> None:
        real_parent = self.root / "real-output"
        real_parent.mkdir()
        linked_parent = self.root / "linked-output"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaisesRegex(native_e2e.E2EError, "symbolic links"):
            native_e2e._write_result(
                {"schema": "fixture"},
                str(linked_parent / "result.json"),
            )
        self.assertFalse((real_parent / "result.json").exists())

    def test_rejects_cli_operator_fallback_evidence(self) -> None:
        artifacts = self.root / "fallback-artifacts"
        artifacts.mkdir()
        with mock.patch.dict(os.environ, {"FAKE_NATIVE_FALLBACK": "1"}):
            with self.assertRaisesRegex(
                native_e2e.E2EError, "exact no-fallback route"
            ):
                native_e2e.run_native_split_e2e(
                    package_directory=self.package,
                    binary=self.binary,
                    backend="cuda",
                    reference_path=self.reference,
                    artifact_directory=artifacts,
                    timeout_seconds=10,
                )

    def test_rejects_ambiguous_duplicate_vocabulary_before_cli(self) -> None:
        vocab_path = self.package / "vocab.json"
        vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
        vocabulary["itos"][100] = vocabulary["itos"][101]
        vocab_bytes = json.dumps(
            vocabulary,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        vocab_path.write_bytes(vocab_bytes)
        manifest_path = self.package / "package_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["assets"]["vocab"]["bytes"] = len(vocab_bytes)
        manifest["assets"]["vocab"]["sha256"] = hashlib.sha256(vocab_bytes).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(native_e2e.E2EError, "760 unique strings"):
            native_e2e.load_package(self.package)


if __name__ == "__main__":
    unittest.main()
