"""Validated TinyReceipt byte-fallback BPE contract for Python example tools.

The explicit-KV package-v2 contract has one tokenizer: the 1536-entry NFC
byte-fallback BPE vocabulary.  This module intentionally has no training or
third-party tokenizer dependency.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


BPE_VOCAB_SIZE = 1536
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
SPECIAL_TOKEN_IDS = {"pad": 0, "bos": 1, "eos": 2, "unk": 3}
STRUCTURAL_TOKENS = (
    "<field>",
    "</field>",
    "<value>",
    "</value>",
    "<op>",
    "</op>",
    "<answer>",
    "</answer>",
)
DIGIT_TOKENS = tuple("0123456789")
ATOMIC_TOKENS = STRUCTURAL_TOKENS + DIGIT_TOKENS
BYTE_TOKENS = tuple(f"<0x{value:02X}>" for value in range(256))
_WHITESPACE_CODEPOINTS = frozenset(
    (
        0x0009,
        0x000A,
        0x000B,
        0x000C,
        0x000D,
        0x0020,
        0x0085,
        0x00A0,
        0x1680,
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
    )
    + tuple(range(0x2000, 0x200B))
)
_BPE_VOCAB_KEYS = frozenset(
    {
        "type",
        "version",
        "vocab_size",
        "itos",
        "merges",
        "normalization",
        "atomic_tokens",
        "byte_tokens",
        "unused_tokens",
        "special_tokens",
        "tokenizer_hash",
    }
)
_BPE_MANIFEST_KEYS = frozenset(
    {
        "type",
        "version",
        "vocab_size",
        "normalization",
        "tokenizer_hash",
        "itos_key",
        "merges_key",
        "token_ids",
    }
)


class TokenizerContractError(ValueError):
    """A package tokenizer or vocabulary violates the release contract."""


def tokenizer_fingerprint(vocabulary: Mapping[str, Any]) -> str:
    """Return the canonical byte-fallback BPE tokenizer fingerprint."""

    unhashed = dict(vocabulary)
    unhashed.pop("tokenizer_hash", None)
    encoded = json.dumps(
        unhashed,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_whitespace(character: str) -> bool:
    return ord(character) in _WHITESPACE_CODEPOINTS


def _exact_integer(value: object) -> bool:
    return type(value) is int


def _validate_token_ids(value: object) -> dict[str, int]:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(SPECIAL_TOKEN_IDS)
        or any(not _exact_integer(value.get(name)) for name in SPECIAL_TOKEN_IDS)
        or dict(value) != SPECIAL_TOKEN_IDS
    ):
        raise TokenizerContractError(
            "token IDs must be pad=0, bos=1, eos=2, and unk=3"
        )
    return dict(SPECIAL_TOKEN_IDS)


class TinyReceiptTokenizer:
    """Inference-only encoder/decoder for the TinyReceipt BPE tokenizer."""

    def __init__(
        self,
        *,
        kind: str,
        vocabulary: Sequence[str],
        token_ids: Mapping[str, int],
        merges: Sequence[tuple[str, str]] = (),
        atomic_tokens: Sequence[str] = (),
        byte_tokens: Sequence[str] = (),
        unused_tokens: Sequence[str] = (),
        tokenizer_hash: str | None = None,
    ) -> None:
        if kind != "byte_fallback_bpe":
            raise TokenizerContractError(
                "TinyReceiptTokenizer supports only byte_fallback_bpe v1"
            )
        self.kind = kind
        self.vocabulary = tuple(vocabulary)
        self.stoi = {token: index for index, token in enumerate(self.vocabulary)}
        self.token_ids = dict(token_ids)
        self.merges = tuple(merges)
        self.atomic_tokens = tuple(atomic_tokens)
        self.byte_tokens = tuple(byte_tokens)
        self.unused_tokens = tuple(unused_tokens)
        self.tokenizer_hash = tokenizer_hash
        self._atomic_set = frozenset(self.atomic_tokens)
        self._byte_set = frozenset(self.byte_tokens)
        self._unused_set = frozenset(self.unused_tokens)
        self._tags_longest_first = tuple(
            sorted(
                (token for token in self.atomic_tokens if len(token) > 1),
                key=len,
                reverse=True,
            )
        )
        self._merge_ranks = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @property
    def package_vocabulary_sha256(self) -> str:
        """Return the package identity digest for this tokenizer contract."""

        if self.tokenizer_hash is None:  # Defensive: construction is internal.
            raise TokenizerContractError(
                "byte_fallback_bpe tokenizer identity is unavailable"
            )
        return self.tokenizer_hash

    @classmethod
    def from_documents(
        cls,
        manifest: Mapping[str, Any],
        vocabulary_document: Mapping[str, Any],
    ) -> "TinyReceiptTokenizer":
        if not isinstance(manifest, Mapping):
            raise TokenizerContractError("package tokenizer must be an object")
        if not isinstance(vocabulary_document, Mapping):
            raise TokenizerContractError("vocab.json must contain an object")
        return cls._from_bpe_documents(manifest, vocabulary_document)

    @classmethod
    def _from_bpe_documents(
        cls,
        manifest: Mapping[str, Any],
        vocabulary_document: Mapping[str, Any],
    ) -> "TinyReceiptTokenizer":
        if set(manifest) != _BPE_MANIFEST_KEYS:
            raise TokenizerContractError(
                "byte_fallback_bpe package tokenizer fields are incomplete"
            )
        token_ids = _validate_token_ids(manifest.get("token_ids"))
        if (
            manifest.get("type") != "byte_fallback_bpe"
            or not _exact_integer(manifest.get("version"))
            or manifest.get("version") != 1
            or not _exact_integer(manifest.get("vocab_size"))
            or manifest.get("vocab_size") != BPE_VOCAB_SIZE
            or manifest.get("normalization") != "NFC"
            or manifest.get("itos_key") != "itos"
            or manifest.get("merges_key") != "merges"
        ):
            raise TokenizerContractError(
                "package tokenizer must declare byte_fallback_bpe v1, NFC, and 1536 tokens"
            )
        if set(vocabulary_document) != _BPE_VOCAB_KEYS:
            raise TokenizerContractError(
                "byte_fallback_bpe vocab.json fields are incomplete"
            )
        if (
            vocabulary_document.get("type") != "byte_fallback_bpe"
            or not _exact_integer(vocabulary_document.get("version"))
            or vocabulary_document.get("version") != 1
            or not _exact_integer(vocabulary_document.get("vocab_size"))
            or vocabulary_document.get("vocab_size") != BPE_VOCAB_SIZE
            or vocabulary_document.get("normalization") != "NFC"
        ):
            raise TokenizerContractError(
                "vocab.json must declare byte_fallback_bpe v1, NFC, and 1536 tokens"
            )

        vocabulary = vocabulary_document.get("itos")
        if (
            not isinstance(vocabulary, list)
            or len(vocabulary) != BPE_VOCAB_SIZE
            or any(not isinstance(token, str) or not token for token in vocabulary)
            or len(set(vocabulary)) != len(vocabulary)
            or vocabulary[: len(SPECIAL_TOKENS)] != list(SPECIAL_TOKENS)
        ):
            raise TokenizerContractError(
                "vocab.json itos must contain exactly 1536 unique non-empty strings"
            )
        vocabulary_set = frozenset(vocabulary)

        atomic_tokens = vocabulary_document.get("atomic_tokens")
        byte_tokens = vocabulary_document.get("byte_tokens")
        unused_tokens = vocabulary_document.get("unused_tokens")
        if atomic_tokens != list(ATOMIC_TOKENS):
            raise TokenizerContractError(
                "byte_fallback_bpe atomic_tokens are not the structured OCR contract"
            )
        if byte_tokens != list(BYTE_TOKENS):
            raise TokenizerContractError(
                "byte_tokens must contain <0x00> through <0xFF>"
            )
        if vocabulary_document.get("special_tokens") != {
            name: token for name, token in zip(SPECIAL_TOKEN_IDS, SPECIAL_TOKENS)
        }:
            raise TokenizerContractError("byte_fallback_bpe special_tokens are invalid")
        if (
            not isinstance(unused_tokens, list)
            or any(not isinstance(token, str) or not token for token in unused_tokens)
            or len(set(unused_tokens)) != len(unused_tokens)
            or not set(unused_tokens).issubset(vocabulary_set)
        ):
            raise TokenizerContractError("byte_fallback_bpe unused_tokens are invalid")
        required_tokens = frozenset(
            (*SPECIAL_TOKENS, *ATOMIC_TOKENS, *BYTE_TOKENS)
        )
        if not required_tokens.issubset(vocabulary_set):
            raise TokenizerContractError(
                "byte_fallback_bpe vocabulary omits a required token"
            )

        raw_merges = vocabulary_document.get("merges")
        if not isinstance(raw_merges, list):
            raise TokenizerContractError("byte_fallback_bpe merges must be an array")
        merges: list[tuple[str, str]] = []
        for index, raw_pair in enumerate(raw_merges):
            if (
                not isinstance(raw_pair, list)
                or len(raw_pair) != 2
                or any(not isinstance(token, str) or not token for token in raw_pair)
            ):
                raise TokenizerContractError(
                    f"byte_fallback_bpe merges[{index}] must contain two strings"
                )
            left, right = raw_pair
            merged = left + right
            if (
                left not in vocabulary_set
                or right not in vocabulary_set
                or merged not in vocabulary_set
            ):
                raise TokenizerContractError(
                    f"byte_fallback_bpe merges[{index}] references an absent token"
                )
            if left in required_tokens or right in required_tokens:
                raise TokenizerContractError(
                    f"byte_fallback_bpe merges[{index}] crosses an atomic boundary"
                )
            if any(character in DIGIT_TOKENS for character in merged) or any(
                _is_whitespace(character) for character in merged
            ):
                raise TokenizerContractError(
                    f"byte_fallback_bpe merges[{index}] crosses a hard boundary"
                )
            merges.append((left, right))
        if len(set(merges)) != len(merges):
            raise TokenizerContractError("byte_fallback_bpe merge pairs must be unique")

        fingerprint = vocabulary_document.get("tokenizer_hash")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
            or fingerprint != tokenizer_fingerprint(vocabulary_document)
            or manifest.get("tokenizer_hash") != fingerprint
        ):
            raise TokenizerContractError(
                "byte_fallback_bpe tokenizer_hash does not match vocab.json"
            )
        return cls(
            kind="byte_fallback_bpe",
            vocabulary=vocabulary,
            token_ids=token_ids,
            merges=merges,
            atomic_tokens=atomic_tokens,
            byte_tokens=byte_tokens,
            unused_tokens=unused_tokens,
            tokenizer_hash=fingerprint,
        )

    def _fallback_ids(self, character: str) -> list[int]:
        return [
            self.stoi[f"<0x{byte:02X}>"]
            for byte in character.encode("utf-8", errors="strict")
        ]

    def _apply_bpe(self, initial: list[str]) -> list[str]:
        tokens = initial
        while len(tokens) > 1:
            best_pair: tuple[str, str] | None = None
            best_rank: int | None = None
            for index in range(len(tokens) - 1):
                pair = (tokens[index], tokens[index + 1])
                rank = self._merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:
                break
            merged: list[str] = []
            index = 0
            while index < len(tokens):
                if (
                    index + 1 < len(tokens)
                    and (tokens[index], tokens[index + 1]) == best_pair
                ):
                    merged.append(tokens[index] + tokens[index + 1])
                    index += 2
                else:
                    merged.append(tokens[index])
                    index += 1
            tokens = merged
        return tokens

    def _encode_mergeable_span(self, span: str) -> list[int]:
        initial: list[str] = []
        for character in span:
            if (
                character in self.stoi
                and character not in self._byte_set
                and character not in self._unused_set
            ):
                initial.append(character)
            else:
                initial.extend(
                    f"<0x{byte:02X}>"
                    for byte in character.encode("utf-8", errors="strict")
                )
        return [self.stoi[token] for token in self._apply_bpe(initial)]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: int = 0,
    ) -> list[int]:
        normalized = unicodedata.normalize("NFC", str(text))
        ids = []
        span_start = 0
        cursor = 0
        while cursor < len(normalized):
            tag = next(
                (
                    token
                    for token in self._tags_longest_first
                    if normalized.startswith(token, cursor)
                ),
                None,
            )
            is_digit = normalized[cursor] in DIGIT_TOKENS
            is_space = _is_whitespace(normalized[cursor])
            if tag is None and not is_digit and not is_space:
                cursor += 1
                continue
            if span_start < cursor:
                ids.extend(
                    self._encode_mergeable_span(normalized[span_start:cursor])
                )
            if tag is not None:
                ids.append(self.stoi[tag])
                cursor += len(tag)
            elif is_digit:
                ids.append(self.stoi[normalized[cursor]])
                cursor += 1
            else:
                character = normalized[cursor]
                if character in self.stoi and character not in self._byte_set:
                    ids.append(self.stoi[character])
                else:
                    ids.extend(self._fallback_ids(character))
                cursor += 1
            span_start = cursor
        if span_start < len(normalized):
            ids.extend(self._encode_mergeable_span(normalized[span_start:]))
        if add_bos:
            ids.insert(0, self.token_ids["bos"])
        if add_eos:
            ids.append(self.token_ids["eos"])
        if max_len:
            ids = ids[:max_len]
            if add_eos and ids[-1] != self.token_ids["eos"]:
                ids[-1] = self.token_ids["eos"]
        return ids

    def decode(self, token_ids: Iterable[int], *, errors: str = "replace") -> str:
        output = bytearray()
        for raw_token_id in token_ids:
            token_id = int(raw_token_id)
            if token_id == self.token_ids["eos"]:
                break
            if token_id in (self.token_ids["pad"], self.token_ids["bos"]):
                continue
            if not 0 <= token_id < len(self.vocabulary):
                continue
            token = self.vocabulary[token_id]
            if token in self._unused_set:
                continue
            if token in self._byte_set:
                output.append(int(token[3:5], 16))
            else:
                output.extend(token.encode("utf-8", errors="strict"))
        return output.decode("utf-8", errors=errors)
