"""Small independent Rust-tokenizers comparisons for the C HF JSON profiles.

No model weights or network access are used. Optional local tokenizer assets
can be checked with VOLVOXAI_HF_TOKENIZER_ASSETS (os.pathsep separated paths).
"""
from __future__ import annotations

import copy
import ctypes as ct
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

import numpy as np

from tools.llm_baseline.tokenizer_generate_reference import ByteInverse, categorical, utf8_stream

ROOT = Path(__file__).resolve().parents[2]
QWEN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
LLAMA = QWEN.replace(r"\p{N}|", r"\p{N}{1,3}|")


class Source(ct.Structure):
    _fields_ = [("tokens", ct.c_void_p), ("token_count", ct.c_size_t),
                ("json", ct.c_void_p), ("json_bytes", ct.c_size_t),
                ("binary", ct.c_void_p), ("binary_bytes", ct.c_size_t),
                ("merges", ct.c_void_p), ("merge_bytes", ct.c_size_t),
                ("format", ct.c_int)]


class NativeTokenizer:
    def __init__(self, library, definition):
        self.library = library
        raw = definition if isinstance(definition, bytes) else json.dumps(definition, ensure_ascii=False).encode()
        payload = ct.create_string_buffer(raw)
        source = Source(json=ct.addressof(payload), json_bytes=len(raw), format=6)
        self.owner = ct.c_void_p()
        self.status = library.vx_tokenizer_create(ct.byref(source), ct.byref(self.owner))

    def close(self):
        if self.owner:
            self.library.vx_tokenizer_release(self.owner)
            self.owner = ct.c_void_p()

    def encode(self, text, special=False, limit=65536, mode=0):
        raw = text.encode() if isinstance(text, str) else text
        payload = ct.create_string_buffer(raw)
        ids, count = ct.c_void_p(), ct.c_size_t()
        status = self.library.vx_tokenizer_encode_options(self.owner, payload, len(raw), limit,
                                                         mode, special, ct.byref(ids), ct.byref(count))
        value = list(ct.cast(ids, ct.POINTER(ct.c_uint32 * count.value)).contents) if count.value else []
        LIBC.free(ids)
        return status, value

    def decode(self, ids, skip=False):
        values = (ct.c_uint32 * len(ids))(*ids)
        output, count, code = ct.c_void_p(), ct.c_size_t(), ct.c_int()
        status = self.library.vx_tokenizer_decode(self.owner, values, len(ids), skip,
                                                 ct.byref(output), ct.byref(count), ct.byref(code))
        value = ct.string_at(output, count.value).decode() if count.value else ""
        LIBC.free(output)
        return status, value, code.value


LIBC = ct.CDLL(None)
LIBC.free.argtypes = [ct.c_void_p]


def byte_definition(alphabet, profile="qwen", nfc=True):
    vocab = {character: index for index, character in enumerate(sorted(alphabet))}
    merges = []
    for a, b in [("a", "b"), ("ab", "c"), ("1", "2"), ("12", "3"), ("123", "4"),
                 ("Ġ", "a"), ("Ġa", "b"), ("a", "a"), ("aa", "a")]:
        vocab.setdefault(a + b, len(vocab))
        merges.append([a, b])
    byte = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False,
            "use_regex": profile in ("gpt2", "digits")}
    if profile in ("qwen", "llama"):
        pre = {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": QWEN if profile == "qwen" else LLAMA},
             "behavior": "Isolated", "invert": False}, byte]}
    elif profile == "digits":
        pre = {"type": "Sequence", "pretokenizers": [{"type": "Digits", "individual_digits": True}, byte]}
    else:
        pre = byte
    return {"version": "1.0", "normalizer": {"type": "NFC"} if nfc else None,
            "pre_tokenizer": pre, "post_processor": None, "decoder": {
                "type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
            "added_tokens": [], "model": {"type": "BPE", "dropout": None,
            "unk_token": None, "fuse_unk": False, "byte_fallback": False,
            "ignore_merges": profile == "llama", "vocab": vocab, "merges": merges}}


def sentencepiece_definition():
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "▁": 3, "a": 4, "b": 5,
             "ab": 6, "▁ab": 7}
    vocab.update({f"<0x{byte:02X}>": 8 + byte for byte in range(256)})
    return {"version": "1.0", "model": {"type": "BPE", "vocab": vocab,
            "merges": [["a", "b"], ["▁", "ab"]], "unk_token": "<unk>",
            "fuse_unk": True, "byte_fallback": True}, "pre_tokenizer": None,
            "normalizer": {"type": "Sequence", "normalizers": [
                {"type": "Prepend", "prepend": "▁"},
                {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]},
            "decoder": {"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"}, {"type": "Fuse"},
                {"type": "Strip", "content": " ", "start": 1, "stop": 0}]},
            "added_tokens": [{"id": i, "content": content, "normalized": False, "special": True,
                              "single_word": False, "lstrip": False, "rstrip": False}
                             for i, content in enumerate(["<unk>", "<s>", "</s>"])],
            "post_processor": {"type": "TemplateProcessing", "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}}], "pair": [],
                "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}}}}


class HFTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from tokenizers import Tokenizer, pre_tokenizers
        except ImportError as error:
            raise unittest.SkipTest("independent tokenizers package is required") from error
        cls.reference_type = Tokenizer
        cls.alphabet = pre_tokenizers.ByteLevel.alphabet()
        cls.temp = tempfile.TemporaryDirectory(prefix="volvox-hf-tokenizer-")
        cls.libraries = []
        for profile in [0, 1]:
            output = Path(cls.temp.name) / f"tokenizer-{profile}.so"
            subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-O2", "-Wall", "-Wextra",
                            "-shared", "-fPIC", f"-DVOLVOXAI_ENABLE_TRAINING={profile}",
                            "-I", str(ROOT / "native/include"), "-I", str(ROOT / "native/src"),
                            "-I", str(ROOT / "native/src/runtime"), "-I", str(ROOT / "native/third_party"),
                            str(ROOT / "native/src/runtime/tokenizer.c"),
                            str(ROOT / "native/third_party/cJSON.c"), "-o", str(output)],
                           check=True, capture_output=True)
            library = ct.CDLL(str(output))
            library.vx_tokenizer_create.argtypes = [ct.POINTER(Source), ct.POINTER(ct.c_void_p)]
            library.vx_tokenizer_release.argtypes = [ct.c_void_p]
            library.vx_tokenizer_encode_options.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t,
                ct.c_uint32, ct.c_int, ct.c_int, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_size_t)]
            library.vx_tokenizer_decode.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int,
                ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_int)]
            cls.libraries.append(library)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, definition, prompts):
        raw = definition if isinstance(definition, bytes) else json.dumps(definition, ensure_ascii=False).encode()
        reference = self.reference_type.from_str(raw.decode())
        for library in self.libraries:
            native = NativeTokenizer(library, raw)
            self.assertEqual(native.status, 0)
            try:
                for text in prompts:
                    for special in [False, True]:
                        expected = reference.encode(text, add_special_tokens=special).ids
                        self.assertEqual(native.encode(text, special), (0, expected), repr(text))
                        self.assertEqual(native.encode(text, special, limit=2), (0, expected[:2]), repr(text))
                        for skip in [False, True]:
                            self.assertEqual(native.decode(expected, skip)[:2],
                                             (0, reference.decode(expected, skip_special_tokens=skip)), repr(text))
            finally:
                native.close()

    def test_regex_profiles_nfc_and_byte_bpe(self):
        prompts = ["abc 1234567", "abaaa", "안녕🙂 café cafe\u0301", "한 Å A\u030a\u0323",
                   "\t\r\n \n  ab\t!\n", "\u0085\ufeff\u00a0\u2028ab", "１２３ Ⅸ ²", "A\x00B", "", "   "]
        for profile in ["gpt2", "digits", "qwen", "llama"]:
            with self.subTest(profile=profile):
                self.compare(byte_definition(self.alphabet, profile), prompts)

    def test_added_tokens_normalization_longest_word_and_strip(self):
        definition = byte_definition(self.alphabet)
        next_id = max(definition["model"]["vocab"].values()) + 1
        for content, flags in [
                ("<s>", {"special": True, "normalized": False}),
                ("<xy>", {"lstrip": True, "rstrip": True}),
                ("<xy>long", {}), ("only", {"single_word": True}),
                ("é", {"normalized": True}), ("한", {"normalized": False}),
                ("Ġ", {"normalized": False})]:
            # Existing model entries must retain their original ID.
            id_value = definition["model"]["vocab"].get(content)
            if id_value is None:
                id_value = next_id
                next_id += 1
            token = {"id": id_value, "content": content, "special": False, "normalized": False,
                     "single_word": False, "lstrip": False, "rstrip": False}
            token.update(flags)
            definition["added_tokens"].append(token)
        self.compare(definition, ["<s>ab<xy> abc", "ab\t<xy>\u00a0ab", "<xy>long <xy>",
                                  "only notonly onlyx ◌̰only only!", "cafe\u0301é", "한한", "Ġab"])

    def test_template_and_prefix_space(self):
        definition = byte_definition(self.alphabet, "gpt2", nfc=False)
        definition["pre_tokenizer"]["add_prefix_space"] = True
        next_id = max(definition["model"]["vocab"].values()) + 1
        definition["added_tokens"] = [
            {"id": next_id + i, "content": text, "special": True, "normalized": False,
             "single_word": False, "lstrip": False, "rstrip": False}
            for i, text in enumerate(["<s>", "</s>"])]
        definition["post_processor"] = {"type": "TemplateProcessing", "single": [
            {"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "</s>", "type_id": 0}}], "pair": [], "special_tokens": {
            text: {"id": text, "ids": [next_id + i], "tokens": [text]}
            for i, text in enumerate(["<s>", "</s>"])}}
        self.compare(definition, ["abc", " abc", "<s>abc</s>", "", "\x00", "한🙂"])

    def test_added_single_word_unicode_boundary_properties(self):
        # HF AddedVocabulary uses Rust regex Unicode \\w. Other_Number such as
        # superscripts/fractions is not a word class; Letter_Number, marks,
        # alphabetic symbols, connector punctuation and join controls are.
        edges = ["²", "⅔", "፩", "Ⅷ", "\u0345", "\u0301", "\u200c", "\u200d", "_", "\u203f", "Ⓐ", "🄰", "🅐", "🅰"]
        for normalized in [False, True]:
            definition = byte_definition(self.alphabet)
            token_id = max(definition["model"]["vocab"].values()) + 1
            definition["added_tokens"] = [{"id": token_id, "content": "only", "normalized": normalized,
                "special": False, "single_word": True, "lstrip": False, "rstrip": False}]
            prompts = [text for edge in edges for text in [edge + "only", "only" + edge, edge + "only" + edge]]
            self.compare(definition, prompts)

    def test_added_ids_start_at_model_vocabulary_size(self):
        definition = byte_definition(self.alphabet)
        vocabulary = definition["model"]["vocab"]
        vocabulary[max(vocabulary, key=vocabulary.get)] = 500
        definition["added_tokens"] = [{"id": len(vocabulary), "content": "<special>",
            "normalized": False, "special": True, "single_word": False, "lstrip": False, "rstrip": False}]
        self.compare(definition, ["abc<special> ab", "<special>", ""])

    def test_added_match_selection_precedes_flags(self):
        for normalized in [False, True]:
            for tokens, prompts in [
                ([("only", {}), ("onlyx", {"single_word": True})], ["onlyxx", "onlyxa only onlyx!", "notonlyxx only"]),
                ([("a", {"rstrip": True}), (" ", {})], ["a  b", "a a", "a   "]),
            ]:
                definition = byte_definition(self.alphabet)
                next_id = max(definition["model"]["vocab"].values()) + 1
                for content, flags in tokens:
                    token_id = definition["model"]["vocab"].get(content)
                    if token_id is None:
                        token_id = next_id
                        next_id += 1
                    token = {"id": token_id, "content": content, "normalized": normalized,
                        "special": False, "single_word": False, "lstrip": False, "rstrip": False}
                    token.update(flags)
                    definition["added_tokens"].append(token)
                self.compare(definition, prompts)

    def test_sentencepiece_bpe_and_contextual_byte_fallback(self):
        definition = sentencepiece_definition()
        self.compare(definition, ["ab ab", "안녕🙂", " ab  ", "<s>ab</s>", "\n\tab", "\x00", ""])
        reference = self.reference_type.from_str(json.dumps(definition, ensure_ascii=False))
        cases = [[8 + 0xe5, 8 + 0x8f], [8 + 0xe5, 8 + 0x8f, 8 + 0xab],
                 [3, 8 + 0x61, 8 + 0xff], [3, 3, 4], [1, 3, 4, 2]]
        for library in self.libraries:
            native = NativeTokenizer(library, definition)
            try:
                for ids in cases:
                    for skip in [False, True]:
                        self.assertEqual(native.decode(ids, skip)[:2], (0, reference.decode(ids, skip_special_tokens=skip)))
            finally:
                native.close()

    def test_unsupported_definition_and_atomic_request_refusals(self):
        base = byte_definition(self.alphabet)
        definitions = []
        for field, value in [("normalizer", {"type": "NFKC"}), ("padding", {}), ("truncation", {}),
                             ("unknown_field", True)]:
            definition = copy.deepcopy(base)
            definition[field] = value
            definitions.append(definition)
        for field, value in [("type", "WordPiece"), ("dropout", .2), ("byte_fallback", True),
                             ("fuse_unk", "false"), ("continuing_subword_prefix", "##")]:
            definition = copy.deepcopy(base)
            definition["model"][field] = value
            definitions.append(definition)
        definition = copy.deepcopy(base)
        definition["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = ".*"
        definitions.append(definition)
        definition = copy.deepcopy(base)
        definition["model"]["merges"].append(definition["model"]["merges"][0])
        definitions.append(definition)
        definition = copy.deepcopy(base)
        definition["added_tokens"] = [{"id": 10000, "content": "<special>",
            "special": True, "normalized": False, "single_word": False,
            "lstrip": False, "rstrip": False}]
        definitions.append(definition)
        definitions.extend([b'{"model":{},"model":{}}', b'{"x":"\\ud800"}', b'{"x":"\\u0000"}', b'{} trailing'])
        for library in self.libraries:
            for definition in definitions:
                native = NativeTokenizer(library, definition)
                self.assertNotEqual(native.status, 0, definition)
                self.assertFalse(native.owner)
            native = NativeTokenizer(library, base)
            try:
                self.assertEqual(native.encode("abc", mode=1)[1], [])
                self.assertNotEqual(native.encode("abc", mode=1)[0], 0)
                self.assertEqual(native.encode(b"\xed\xa0\x80")[1], [])
                self.assertNotEqual(native.encode(b"\xed\xa0\x80")[0], 0)
                self.assertEqual(native.encode("abc", special=True, limit=0), (0, []))
                self.assertNotEqual(native.decode([0, 0xffffffff])[0], 0)
                self.assertEqual(native.decode([0, 0xffffffff])[1], "")
            finally:
                native.close()

    def test_seeded_unicode_edge_strings(self):
        rng = random.Random(4816)
        alphabet = list(" abCDEf123456\t\r\n!?!한글🙂é\u0301\u0323\ufeff\u0085\u00a0\u2028\u0662Ⅷ²四")
        prompts = ["".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 64))) for _ in range(40)]
        for profile in ["gpt2", "digits", "qwen", "llama"]:
            self.compare(byte_definition(self.alphabet, profile), prompts)

    def test_optional_local_tokenizer_assets(self):
        paths = os.environ.get("VOLVOXAI_HF_TOKENIZER_ASSETS", "")
        if not paths:
            return  # Optional assets do not replace the mandatory small fixtures.
        for path in paths.split(os.pathsep):
            self.compare(Path(path).read_bytes(), ["Hello world!", "안녕하세요🙂 cafe\u0301 한", "123456789", "\t\r\n abc", "A\x00B", ""])


class TokenizerGenerateReferenceTest(unittest.TestCase):
    def test_uniform_categories_have_analytical_seed_witnesses(self):
        # Published SplitMix64 seed-zero words, expressed as uniforms. Four
        # equal categories map each draw to floor(4*u), in ascending ID order.
        uniforms = [.8833108082136426, .43152799704850997, .026433771592597743, .9708819781538285]
        state = 0
        ids = []
        for expected in uniforms:
            token, state, witness = categorical(np.zeros(4, dtype=np.float32), {"sampling": "RANDOM"}, state)
            self.assertEqual(witness["uniform"], expected)
            self.assertEqual(token, int(4 * expected))
            ids.append(token)
        self.assertEqual(ids, [3, 1, 0, 3])

    def test_temperature_and_minimal_top_p_prefix(self):
        # log(4),log(2),log(1) give masses ~4:2:1. top-k=2 gives 2:1;
        # top-p=.6 retains only ID 0, regardless of the random draw.
        scores = np.log(np.array([4, 2, 1], dtype=np.float32))
        token, _, witness = categorical(scores, {"sampling": "RANDOM", "top_k": 2, "top_p": .6}, 7)
        self.assertEqual(token, 0)
        self.assertEqual(witness["retained_categories"], 1)
        token, _, witness = categorical([0, 1, 1], {"sampling": "RANDOM", "top_k": 1}, 0)
        self.assertEqual(token, 1)  # Lowest ID wins an equal-score tie.
        self.assertEqual(witness["retained_categories"], 1)

    def test_unicode_chunks_are_not_eof(self):
        raw = "한🙂🚀".encode()
        for size in range(1, len(raw) + 1):
            parts = [raw[index:index+size] for index in range(0, len(raw), size)]
            self.assertEqual(utf8_stream(parts)["text"], "한🙂🚀")
        self.assertEqual(utf8_stream([b"\xed\x95", b"\x9c"])["deltas"], ["", "한"])

    def test_terminal_finalize_and_cancel_discard_are_distinct(self):
        parts = [b"A", b"\xf0\x9f\x99"]
        finalized, discarded = utf8_stream(parts), utf8_stream(parts, "discard")
        self.assertEqual(finalized["text"], "A\ufffd")
        self.assertEqual(discarded["text"], "A")
        self.assertEqual(discarded["discarded_pending_bytes"], 3)
        with self.assertRaises(ValueError):
            utf8_stream(parts, "implicit")

    def test_raw_inverse_preserves_invalid_bytes_and_special_policy(self):
        # HF byte alphabet maps bytes ED,95,9C to í,ķ,ľ. Added tokens have
        # literal contents and only special tokens disappear under skip.
        inverse = ByteInverse({"model": {"vocab": {"í": 0, "ķ": 1, "ľ": 2}},
            "decoder": {"type": "ByteLevel"}, "added_tokens": [
                {"id": 3, "content": "<eos>", "special": True},
                {"id": 4, "content": "<literal>", "special": False}]})
        self.assertEqual(inverse.bytes([0, 1]), b"\xed\x95")
        self.assertEqual(inverse.bytes([0, 1, 2]), "한".encode())
        self.assertEqual(inverse.bytes([3, 4], True), b"<literal>")
        self.assertEqual(inverse.bytes([3], False), b"<eos>")


if __name__ == "__main__":
    unittest.main()

