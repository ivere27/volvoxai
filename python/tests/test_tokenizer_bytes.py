"""Generated text RPCs: immutable ByteLevel bytes and one UTF-8 message boundary."""
from __future__ import annotations

import codecs
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import volvoxai as vx

PB = vx.pb


def alphabet():
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    missing = [byte for byte in range(256) if byte not in visible]
    return dict(zip(visible + missing, map(chr, visible + list(range(256, 256 + len(missing))))))


ALPHABET = alphabet()


def byte_vocabulary(definition):
    """Preserve encoded asset strings. The native C core performs the inverse."""
    if definition["model"]["type"] != "BPE" or definition["decoder"].get("type") != "ByteLevel":
        raise ValueError("only plain ByteLevel BPE token-domain decoding is declared")
    entries = {index: PB.ByteLevelVocabularyToken(id=index, token=token)
               for token, index in definition["model"]["vocab"].items()}
    if len(entries) != len(definition["model"]["vocab"]):
        raise ValueError("duplicate ID in source vocabulary")
    seen = set()
    for added in definition.get("added_tokens", []):
        index = added["id"]
        if index in seen or added.get("lstrip") or added.get("rstrip"):
            raise ValueError("duplicate or contextual added token")
        seen.add(index)
        if index in entries and entries[index].token != added["content"]:
            raise ValueError("added token disagrees with base vocabulary")
        entries[index] = PB.ByteLevelVocabularyToken(id=index, token=added["content"],
            added=True, special=added["special"])
    return PB.ByteLevelTokenizerVocabulary(tokens=list(entries.values()))


class TokenizerBytesTest(unittest.TestCase):
    def setUp(self):
        self.host = vx.open_library()
        self.addCleanup(self.host.close)
        self.text = vx.VxTextServiceClient(self.host)

    def create(self, entries):
        handle = self.text.create_tokenizer(PB.CreateTokenizerRequest(
            byte_level=PB.ByteLevelTokenizerVocabulary(tokens=entries)))
        self.addCleanup(self.text.release_tokenizer, PB.TokenizerRef(tokenizer_id=handle.tokenizer_id))
        self.assertEqual(handle.profile, PB.TokenizerProfile.TOKENIZER_PROFILE_BYTE_LEVEL_DECODE_ONLY)
        self.assertFalse(handle.supports_encode)
        self.assertTrue(handle.supports_decode_token_bytes)
        self.assertTrue(handle.token_byte_table_complete)
        self.assertEqual(handle.merge_count, 0)
        return handle

    def decode(self, handle, ids, skip=False):
        return self.text.decode_token_bytes(PB.DecodeTokenBytesRequest(
            tokenizer_id=handle.tokenizer_id, tokens=ids, skip_special_tokens=skip))

    def test_all_bytes_sparse_ids_and_snapshot(self):
        entries = [PB.ByteLevelVocabularyToken(id=b * 7 + 1, token=ALPHABET[b]) for b in range(256)]
        handle = self.create(entries)
        self.assertTrue(handle.byte_alphabet_complete)
        for value in entries:
            value.token = "corrupted after create"
        ids = [b * 7 + 1 for b in range(256)]
        result = self.decode(handle, ids)
        self.assertTrue(result.complete)
        self.assertEqual(result.value, bytes(range(256)))
        self.assertEqual([(x.token_id, x.byte_offset, x.byte_length) for x in result.tokens],
                         [(index, offset, 1) for offset, index in enumerate(ids)])
        decoded = self.text.decode_tokens(PB.DecodeTokensRequest(tokenizer_id=handle.tokenizer_id, tokens=ids))
        self.assertEqual(decoded.text, bytes(range(256)).decode("utf-8", "replace"))

    def test_special_added_and_empty_offsets(self):
        handle = self.create([PB.ByteLevelVocabularyToken(id=1, token="A"),
            PB.ByteLevelVocabularyToken(id=3, token="<eos>", added=True, special=True),
            PB.ByteLevelVocabularyToken(id=5, token="<think>한", added=True),
            PB.ByteLevelVocabularyToken(id=9, token="")])
        self.assertFalse(handle.byte_alphabet_complete)
        ids = [1, 3, 5, 9, 3, 1]
        keep = self.decode(handle, ids)
        self.assertEqual(keep.value, b"A<eos>" + "<think>한".encode() + b"<eos>A")
        skip = self.decode(handle, ids, True)
        self.assertEqual(skip.value, "A<think>한A".encode())
        self.assertEqual([x.skipped for x in skip.tokens], [False, True, False, False, True, False])
        self.assertEqual([x.byte_offset for x in skip.tokens], [0, 1, 1, 11, 11, 11])
        self.assertEqual([x.byte_length for x in skip.tokens], [1, 0, 10, 0, 0, 1])
        for mode in (False, True):
            raw = self.decode(handle, ids, mode)
            self.assertEqual(self.text.decode_tokens(PB.DecodeTokensRequest(tokenizer_id=handle.tokenizer_id,
                tokens=ids, skip_special_tokens=mode)).text, raw.value.decode("utf-8", "replace"))

    def test_unknown_atomic_and_decode_only_refusal(self):
        handle = self.create([PB.ByteLevelVocabularyToken(id=7, token="A")])
        for method, message in ((self.text.decode_token_bytes, PB.DecodeTokenBytesRequest),
                                (self.text.decode_tokens, PB.DecodeTokensRequest)):
            for skip in (False, True):
                with self.assertRaises(vx.VolvoxAIError) as failure:
                    method(message(tokenizer_id=handle.tokenizer_id, tokens=[7, 8], skip_special_tokens=skip))
                self.assertEqual(failure.exception.code, PB.OperationCode.OPERATION_CODE_TOKEN_ID_UNKNOWN)
                response = failure.exception.response
                if isinstance(response, PB.DecodedTokenBytes):
                    self.assertFalse(response.complete); self.assertEqual(response.value, b""); self.assertEqual(response.tokens, [])
                else:
                    self.assertEqual(response.text, "")
        with self.assertRaises(vx.VolvoxAIError) as failure:
            self.text.encode_text(PB.EncodeTextRequest(tokenizer_id=handle.tokenizer_id, text="x"))
        self.assertEqual(failure.exception.code, PB.OperationCode.OPERATION_CODE_TOKENIZER_OPERATION_UNSUPPORTED)
        self.assertEqual(failure.exception.response.tokens, [])
        self.assertEqual(self.decode(handle, [7]).value, b"A")

    def test_invalid_definitions_and_lifetime(self):
        invalid = [[PB.ByteLevelVocabularyToken(id=0, token="한")],
            [PB.ByteLevelVocabularyToken(id=0, token="A", special=True)],
            [PB.ByteLevelVocabularyToken(id=0, token="A"), PB.ByteLevelVocabularyToken(id=0, token="B")]]
        for entries in invalid:
            with self.assertRaises(vx.VolvoxAIError) as failure:
                self.text.create_tokenizer(PB.CreateTokenizerRequest(byte_level=PB.ByteLevelTokenizerVocabulary(tokens=entries)))
            self.assertEqual(failure.exception.response.tokenizer_id, 0)
        with self.assertRaises(vx.VolvoxAIError):
            self.text.create_tokenizer(PB.CreateTokenizerRequest(byte_level=PB.ByteLevelTokenizerVocabulary(), merges=b"a b"))
        handle = self.create([])
        self.assertTrue(self.decode(handle, []).complete)
        self.text.release_tokenizer(PB.TokenizerRef(tokenizer_id=handle.tokenizer_id))
        with self.assertRaises(vx.VolvoxAIError) as failure:
            self.decode(handle, [])
        self.assertEqual(failure.exception.code, PB.OperationCode.OPERATION_CODE_HANDLE_DISPOSED)

    def test_whole_message_utf8_legacy_bom_and_placeholders(self):
        values = [b"\xe2", b"\x82\xac", b"\xef\xbb\xbfX", b"x\xef\xbb\xbfy",
                  b"\xef", b"\xbb\xbf", b"\xc4", b"\xa0", b"\xff"]
        handle = self.text.create_tokenizer(PB.CreateTokenizerRequest(tokens=PB.TokenizerVocabulary(
            tokens=[PB.VocabularyToken(id=index, value=raw) for index, raw in enumerate(values)])))
        self.addCleanup(self.text.release_tokenizer, PB.TokenizerRef(tokenizer_id=handle.tokenizer_id))
        self.assertTrue(handle.supports_encode)
        self.assertFalse(handle.supports_decode_token_bytes)
        result = self.text.decode_tokens(PB.DecodeTokensRequest(tokenizer_id=handle.tokenizer_id,
            tokens=[0, 1, 2, 3, 999, 4, 5, 6, 7, 8]))
        self.assertEqual(result.text, "€Xx\ufeffy\ufeff �")
        with self.assertRaises(vx.VolvoxAIError) as failure:
            self.decode(handle, [0])
        self.assertEqual(failure.exception.code, PB.OperationCode.OPERATION_CODE_TOKENIZER_OPERATION_UNSUPPORTED)

    def test_actual_generate_chunks_use_bytes_and_one_incremental_utf8_task(self):
        """A deterministic native decoder emits the three bytes of Korean 한."""
        import numpy as np
        from safetensors.numpy import save_file
        handle = self.create([PB.ByteLevelVocabularyToken(id=index, token=ALPHABET[byte])
            for index, byte in enumerate("한".encode())] +
            [PB.ByteLevelVocabularyToken(id=3, token="<eos>", added=True, special=True)])
        channels, sequence = 4, 8
        def node(name, operator, inputs, shape, params=None, dtype="float32"):
            return {"id": name, "opType": operator, "inputs": inputs, "params": params or {},
                "outputs": {"out": {"tensor": name, "shape": shape, "dtype": dtype}}}
        inputs = {"tokens": {"shape": [1, sequence], "dtype": "int32"},
                  "keep": {"shape": [1, sequence], "dtype": "int32"},
                  "active_lengths": {"shape": [1], "dtype": "int32"}}
        graph = {"format": "volvox-graph/v1", "dimensions": {}, "inputs": inputs,
            "nodes": [node("embedding", "Embedding", {"input": "tokens", "weight": "embedding.weight"}, [1, sequence, channels]),
                node("key", "Linear", {"input": "embedding", "weight": "key.weight"}, [1, sequence, channels]),
                node("value", "Linear", {"input": "embedding", "weight": "value.weight"}, [1, sequence, channels]),
                node("attention", "CrossSDPA", {"q": "embedding", "k": "key", "v": "value", "mask": "keep"},
                    [1, sequence, channels], {"heads": 1, "causal": True}),
                node("conditioned", "Add", {"a": "embedding", "b": "attention"}, [1, sequence, channels]),
                node("logits", "Linear", {"input": "conditioned", "weight": "logits.weight"}, [1, sequence, channels]),
                node("chosen", "ArgMax", {"input": "logits"}, [1, sequence], {"axis": -1, "keepdims": False}, "int32")],
            "outputs": ["chosen", "logits"], "execution": {
                "incremental": {"active_lengths_input": "active_lengths", "sequence_axis": 1,
                    "update_inputs": ["tokens", "keep"]}, "invariant_inputs": ["active_lengths"],
                "entrypoints": [{"name": name, "effect": effect, "allowed_ranges": ranges,
                    "inputs": list(inputs), "outputs": ["chosen", "logits"]}
                    for name, effect, ranges in (("evaluate", 1, [1]), ("initialize", 2, [1]), ("transition", 3, [3]))],
                "generation": {"initialize_entrypoint": "initialize", "transition_entrypoint": "transition",
                    "token_input": "tokens", "token_output": "chosen", "keep_input": "keep", "logits_output": "logits",
                    "logits_axis": 2, "id_domain_size": channels, "output_name": "generated_ids", "sequence_axis": 1}}}
        weights = {"embedding.weight": np.eye(channels, dtype=np.float32),
            "key.weight": np.eye(channels, dtype=np.float32), "value.weight": np.zeros((channels, channels), dtype=np.float32),
            "logits.weight": np.roll(np.eye(channels, dtype=np.float32), 1, axis=0)}
        inference = vx.VxInferenceServiceClient(self.host)
        with tempfile.TemporaryDirectory() as temp, ExitStack() as cleanup:
            directory = Path(temp)
            (directory / "graph.json").write_text(json.dumps(graph))
            save_file(weights, directory / "model.safetensors")
            runtime = inference.create_runtime(PB.CreateRuntimeRequest(cpu_threads=1))
            cleanup.callback(inference.release_runtime, PB.RuntimeRef(runtime_id=runtime.runtime_id))
            model = inference.load_model(PB.LoadModelRequest(runtime_id=runtime.runtime_id,
                graph_path=str(directory / "graph.json"), weight_paths=[str(directory / "model.safetensors")]))
            cleanup.callback(inference.release_model, PB.ModelRef(model_id=model.model_id))
            compiled = inference.compile_model(PB.CompileModelRequest(model_id=model.model_id,
                policy=PB.BackendPolicy(mode=PB.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                    backends=["cpu"], operator_fallback=PB.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
            cleanup.callback(inference.release_compiled_model, PB.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
            for skip in (False, True):
                for chunks in ((4,), (1, 2, 1)):
                    with self.subTest(skip=skip, chunks=chunks):
                        context = inference.create_execution_context(PB.CreateExecutionContextRequest(
                            compiled_model_id=compiled.compiled_model_id,
                            autoregressive=PB.AutoregressiveExecution(sampling=PB.SamplingMode.SAMPLING_MODE_GREEDY, max_steps=4)))
                        ref = PB.ExecutionContextRef(context_id=context.context_id)
                        try:
                            values = {"tokens": np.array([[3] + [0] * 7], dtype=np.int32),
                                "keep": np.array([[1] + [0] * 7], dtype=np.int32), "active_lengths": np.array([1], dtype=np.int32)}
                            initial = PB.GenerationInputs(values=[PB.Tensor(name=name, dtype=PB.DataType.DATA_TYPE_I32,
                                shape=list(value.shape), inline=value.tobytes()) for name, value in values.items()])
                            decoder = codecs.getincrementaldecoder("utf-8")("replace")
                            deltas, joined, ids = [], bytearray(), []
                            for index, count in enumerate(chunks):
                                packet = inference.generate(PB.GenerateRequest(context_id=context.context_id,
                                    initial_inputs=initial if index == 0 else None, budget=PB.GenerationBudget(max_steps=count)))
                                output = packet.outputs[0]
                                self.assertEqual((output.name, output.dtype, output.shape),
                                    ("generated_ids", PB.DataType.DATA_TYPE_I32, [1, count]))
                                self.assertFalse(output.HasField("buffer"))
                                current = np.frombuffer(output.inline, dtype=np.int32).tolist()
                                self.assertEqual(current, [0, 1, 2, 3][len(ids):len(ids) + count])
                                ids += current
                                raw = self.decode(handle, current, skip)
                                joined.extend(raw.value)
                                deltas.append(decoder.decode(raw.value, final=False))
                                if chunks == (1, 2, 1) and index == 0:
                                    self.assertEqual(deltas[-1], "")
                                    self.assertEqual(decoder.getstate()[0], b"\xed")
                            deltas.append(decoder.decode(b"", final=True))
                            expected = "한" + ("" if skip else "<eos>")
                            self.assertEqual("".join(deltas), expected)
                            self.assertEqual(bytes(joined), expected.encode())
                            self.assertEqual(self.text.decode_tokens(PB.DecodeTokensRequest(tokenizer_id=handle.tokenizer_id,
                                tokens=ids, skip_special_tokens=skip)).text, expected)
                        finally:
                            inference.release_execution_context(ref)


@unittest.skipUnless(os.environ.get("VOLVOXAI_TOKENIZER_REFERENCE_ROOT"), "offline HF byte reference assets are opt-in")
class PinnedTokenizerBytesTest(unittest.TestCase):
    setUp = TokenizerBytesTest.setUp
    decode = TokenizerBytesTest.decode
    def test_pinned_hf_decode_and_stream_cases(self):
        repo = Path(os.environ["VOLVOXAI_TOKENIZER_REFERENCE_ROOT"]).resolve()
        bundle = repo / "build/llm/e-tokenizer-reference"
        sys.path.insert(0, str(bundle))
        from capture import check_stream
        policy = json.loads((bundle / "policy.json").read_text())
        self.assertEqual(hashlib.sha256((bundle / "policy.json").read_bytes()).hexdigest(),
            "2563ae4fc14d6a01b07d4366bf557430c8278dcf50e43eb45c10266bab4c8e55")
        receipt = json.loads((bundle / "capture-v3/receipt.json").read_text())
        self.assertEqual(receipt["status"], "PASS")
        decode_checks = stream_checks = unavailable = independent_stream_checks = 0
        for model in receipt["models"]:
            asset = model["assets"]["tokenizer.json"]
            path = repo / asset["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), asset["sha256"])
            vocabulary = byte_vocabulary(json.loads(path.read_text()))
            handle = self.text.create_tokenizer(PB.CreateTokenizerRequest(byte_level=vocabulary))
            self.addCleanup(self.text.release_tokenizer, PB.TokenizerRef(tokenizer_id=handle.tokenizer_id))
            self.assertEqual(handle.vocabulary_size, len(vocabulary.tokens))
            self.assertEqual(handle.byte_alphabet_complete, not model["missing_single_byte_vocabulary"])
            for case in model["cases"]:
                for encoded in case["encodings"]:
                    for expected in encoded["decodes"]:
                        ids, skip = encoded["ids"], expected["skip_special_tokens"]
                        raw = self.decode(handle, ids, skip)
                        self.assertEqual(raw.value.hex(), expected["raw_bytes_hex"])
                        parts = [raw.value[x.byte_offset:x.byte_offset + x.byte_length] for x in raw.tokens]
                        self.assertEqual([x.hex() for x in parts], expected["token_bytes_hex"])
                        decoded = self.text.decode_tokens(PB.DecodeTokensRequest(tokenizer_id=handle.tokenizer_id,
                            tokens=ids, skip_special_tokens=skip))
                        self.assertEqual(decoded.text, expected["hf_text_cleanup_false"])
                        result = check_stream(raw.value, parts, policy["stream"])
                        self.assertEqual(result, expected["stream"])
                        stream_checks += result["exact_checks"]; decode_checks += 1
            for case in model["raw_byte_cases"]:
                if not case["hf_ids_available"]:
                    unavailable += 1
                    # Sparse selected asset: independent UTF8 bytes remain
                    # useful, but no fabricated model token ID is submitted.
                    result = check_stream(bytes.fromhex(case["hex"]),
                        [bytes([b]) for b in bytes.fromhex(case["hex"])], policy["stream"])
                    self.assertEqual(result, case["independent_utf8_stream"])
                    independent_stream_checks += result["exact_checks"]
                else:
                    raw = self.decode(handle, case["ids"])
                    self.assertEqual(raw.value.hex(), case["hex"])
                    self.assertEqual(self.text.decode_tokens(PB.DecodeTokensRequest(
                        tokenizer_id=handle.tokenizer_id, tokens=case["ids"])).text, case["hf_text_cleanup_false"])
                    parts = [raw.value[x.byte_offset:x.byte_offset + x.byte_length] for x in raw.tokens]
                    result = check_stream(raw.value, parts, policy["stream"])
                    self.assertEqual(result, case["stream"]); decode_checks += 1
                stream_checks += result["exact_checks"]
            for case in model["added_token_cases"]:
                for expected in case["modes"]:
                    skip = expected["skip_special_tokens"]
                    self.assertEqual(self.decode(handle, [case["id"]], skip).value.hex(), expected["raw_bytes_hex"])
                    self.assertEqual(self.text.decode_tokens(PB.DecodeTokensRequest(tokenizer_id=handle.tokenizer_id,
                        tokens=[case["id"]], skip_special_tokens=skip)).text, expected["hf_text"])
                    decode_checks += 1
        self.assertEqual((decode_checks, stream_checks, unavailable), (230, 5966, 2))
        self.assertEqual(independent_stream_checks, 284)
        print(json.dumps({"qualification": "pinned ByteLevel generated RPC", "hf_decode_checks": decode_checks,
            "caller_utf8_stream_checks": stream_checks, "engine_backed_stream_checks": stream_checks - independent_stream_checks,
            "independent_unavailable_id_stream_checks": independent_stream_checks,
            "unavailable_raw_id_cases": unavailable, "native_HF_encode_claim": False}))


if __name__ == "__main__":
    unittest.main()
