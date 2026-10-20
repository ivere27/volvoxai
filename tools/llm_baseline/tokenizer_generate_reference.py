"""Independent HF observations and caller byte handling for the E workflow.

No engine imports or numerical operators live here. RANDOM uses the public
SplitMix64/inverse-CDF contract on independently observed HF F32 logits; it is
not an assertion that PyTorch's unrelated RNG produces the same samples.
"""
from __future__ import annotations

import codecs
import math

import numpy as np


def categorical(logits, policy, state):
    scores = np.asarray(logits, dtype=np.float32).astype(np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("finite categorical F32 row required")
    order = np.lexsort((np.arange(scores.size), -scores))
    if policy["sampling"] == "GREEDY":
        return int(order[0]), state, {"greedy_margin": float(scores[order[0]] - scores[order[1]])}
    if policy["sampling"] != "RANDOM":
        raise ValueError("explicit selection policy required")
    # Protobuf's temperature/top_p fields are float32, including explicit .7/.85.
    temperature = float(np.float32(policy.get("temperature", 1)))
    top_p = float(np.float32(policy.get("top_p", 1)))
    top_k = policy.get("top_k", 0)
    if temperature <= 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("invalid categorical policy")
    order = order[:top_k] if top_k else order
    weights = [math.exp((scores[index] - scores[order[0]]) / temperature) for index in order]
    total = math.fsum(weights)
    if top_p < 1:
        cumulative = 0.0
        retain = 0
        for weight in weights:
            retain += 1
            cumulative += weight
            if cumulative >= top_p * total:
                break
        order, weights = order[:retain], weights[:retain]
        total = math.fsum(weights)
    mask = (1 << 64) - 1
    state = (state + 0x9E3779B97F4A7C15) & mask
    word = state
    word = ((word ^ (word >> 30)) * 0xBF58476D1CE4E5B9) & mask
    word = ((word ^ (word >> 27)) * 0x94D049BB133111EB) & mask
    word ^= word >> 31
    uniform = (word >> 11) / (1 << 53)
    target, cumulative, lower = uniform * total, 0.0, 0.0
    selected = len(order) - 1
    for i, weight in enumerate(weights):
        lower = cumulative
        cumulative += weight
        if target < cumulative:
            selected = i
            break
    return int(order[selected]), state, {"uniform": uniform, "retained_categories": len(order),
        "cdf_lower": lower / total, "cdf_upper": cumulative / total,
        "distance_to_cdf_boundary": min(target - lower, cumulative - target) / total}


class ByteInverse:
    """Invert HF's published ByteLevel alphabet, independently of the engine."""
    def __init__(self, definition):
        if definition["decoder"]["type"] != "ByteLevel":
            raise ValueError("plain ByteLevel decoder required")
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        alphabet = {value: chr(value) for value in visible}
        extra = 256
        for value in range(256):
            if value not in alphabet:
                alphabet[value] = chr(extra)
                extra += 1
        self.inverse = {character: value for value, character in alphabet.items()}
        self.vocab = definition["model"]["vocab"]
        self.tokens = {index: token for token, index in self.vocab.items()}
        self.added = {entry["id"]: entry for entry in definition.get("added_tokens", [])}
        self.tokens.update({index: entry["content"] for index, entry in self.added.items()})
        self.byte_ids = {byte: self.vocab.get(character) for byte, character in alphabet.items()}

    def parts(self, ids, skip_special_tokens=False):
        result = []
        for index in ids:
            if index not in self.tokens:
                raise ValueError("unknown reference vocabulary ID")
            entry = self.added.get(index)
            if entry:
                result.append(b"" if skip_special_tokens and entry["special"]
                              else entry["content"].encode("utf-8"))
            else:
                result.append(bytes(self.inverse[character] for character in self.tokens[index]))
        return result

    def bytes(self, ids, skip_special_tokens=False):
        return b"".join(self.parts(ids, skip_special_tokens))


def utf8_stream(parts, termination="finalize"):
    if termination not in ("finalize", "discard"):
        raise ValueError("explicit cancellation policy required")
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    deltas, boundaries = [], []
    for part in parts:
        deltas.append(decoder.decode(part, final=False))
        boundaries.append(decoder.getstate()[0].hex())
    pending = decoder.getstate()[0]
    final = decoder.decode(b"", final=True) if termination == "finalize" else ""
    decoder.reset()
    return {"text": "".join(deltas) + final, "deltas": deltas,
        "pending_at_boundaries_hex": boundaries, "pending_before_finish_hex": pending.hex(),
        "final_delta": final, "discarded_pending_bytes": len(pending) if termination == "discard" else 0}
