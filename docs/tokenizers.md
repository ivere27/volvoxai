# Tokenize text and decode generated IDs

A generated token ID identifies a vocabulary entry. Its bytes may end in the
middle of a UTF-8 character, so decode a complete message after joining token
bytes. During generation, keep one incremental UTF-8 decoder for each task and
feed it the raw bytes from successive ID packets.

## Load a complete Hugging Face tokenizer

Use the local `tokenizer.json` from the same checkpoint revision as the weights.
The engine copies and parses the definition; encoding and decoding then work
offline without a tokenizer library in the deployed runtime.

```python
from pathlib import Path
import volvoxai as vx

pb = vx.pb
with vx.open_library() as owner:
    text = vx.VxTextServiceClient(owner)
    handle = text.create_tokenizer(pb.CreateTokenizerRequest(
        hugging_face_tokenizer_json=Path("checkpoint/tokenizer.json").read_bytes()))
    try:
        prompt = text.encode_text(pb.EncodeTextRequest(
            tokenizer_id=handle.tokenizer_id, text="한국어와 emoji 🙂",
            add_special_tokens=True))
        print(prompt.tokens)
        decoded = text.decode_tokens(pb.DecodeTokensRequest(
            tokenizer_id=handle.tokenizer_id, tokens=prompt.tokens,
            skip_special_tokens=True))
        print(decoded.text)
    finally:
        text.release_tokenizer(pb.TokenizerRef(tokenizer_id=handle.tokenizer_id))
```

`AUTO` encoding uses the definition's BPE model. `add_special_tokens` defaults
to true for these profiles and applies the admitted single-sequence BOS/EOS
postprocessor. The caller still applies the model's chat template and chooses
generation settings. There is no automatic conversation formatting.

`HF_BYTE_LEVEL_BPE` includes null/NFC normalization, the supported GPT-2, Qwen
and Llama 3 regex splits, byte mapping, ordered merges, `ignore_merges`, added
token matching and special-token policy. It offers both `DecodeTokens` and
lossless `DecodeTokenBytes`; use the incremental workflow below for generated
ID packets. NFC changes the token-domain spelling of decomposed input.

`HF_SENTENCEPIECE_BPE` accepts the converted SentencePiece-style BPE definition
used by Llama 1/2 families, including the space marker and byte fallback. Its
decoder acts on the entire ID sequence: space replacement and initial-space
stripping depend on context. Accumulate IDs and call `DecodeTokens` on the
whole sequence when updating text; independently decoded packet strings cannot
be concatenated. This profile refuses `DecodeTokenBytes`.

Definitions are limited to 32 MiB. Arbitrary regex, Unigram, raw binary `.model`,
Precompiled/NFKC normalization, tokenizer padding/truncation and unsupported
postprocessors are refused before publishing a handle. Use an external
tokenizer for those definitions. Supported asset families are exercised by the
small implementation regressions; this is not a claim about every tokenizer
published under a Llama model name.

## Use the smaller decode-only definition

The `BYTE_LEVEL_DECODE_ONLY` tokenizer profile restores the fixed ByteLevel byte
alphabet used by the selected SmolLM2 and Qwen3 tokenizer assets. It accepts the
encoded vocabulary strings and added-token flags; the engine performs the byte
inverse and copies an immutable table before publishing the handle. The profile
supports byte decoding and whole-message text decoding. `EncodeText` is refused:
normalization, pretokenization, BPE encoding, chat templates and sampling have
separate contracts and are not implemented by this profile.

Use the existing external tokenizer workflow in [HF decoder](hf-decoder.md) to
prepare prompt IDs. A byte inverse restores the token-domain bytes after any
normalization that the external tokenizer applied. It cannot recover the
original spelling of text normalized before tokenization.

## Restore bytes and decode a message

This small vocabulary splits the Korean character 한 over three IDs. Its encoded
ByteLevel strings are valid UTF-8; their inverse bytes are ED, 95 and 9C.

```python
import codecs
import volvoxai as vx

pb = vx.pb
with vx.open_library() as owner:
    text = vx.VxTextServiceClient(owner)
    handle = text.create_tokenizer(pb.CreateTokenizerRequest(
        byte_level=pb.ByteLevelTokenizerVocabulary(tokens=[
            pb.ByteLevelVocabularyToken(id=0, token="í"),
            pb.ByteLevelVocabularyToken(id=1, token="ķ"),
            pb.ByteLevelVocabularyToken(id=2, token="ľ"),
            pb.ByteLevelVocabularyToken(id=3, token="<eos>",
                                        added=True, special=True),
        ])))
    try:
        decoded = text.decode_tokens(pb.DecodeTokensRequest(
            tokenizer_id=handle.tokenizer_id, tokens=[0, 1, 2]))
        assert decoded.text == "한"

        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        deltas = []
        for generated_ids in ([0], [1, 2], [3]):
            raw = text.decode_token_bytes(pb.DecodeTokenBytesRequest(
                tokenizer_id=handle.tokenizer_id, tokens=generated_ids,
                skip_special_tokens=True))
            assert raw.complete
            deltas.append(decoder.decode(raw.value, final=False))
        deltas.append(decoder.decode(b"", final=True))
        assert deltas == ["", "한", "", ""]
    finally:
        text.release_tokenizer(pb.TokenizerRef(tokenizer_id=handle.tokenizer_id))
```

Pass each actual `Generate` ID packet to the same byte operation. The UTF-8
decoder belongs to the caller: ID packets and API calls are not text boundaries.
At normal completion, finalize once to replace any remaining incomplete UTF-8
sequence. On cancellation, choose explicitly whether to finalize or discard the
pending incomplete sequence; preserve text already emitted and create a fresh
decoder for the next task. Raw byte decoding never inserts replacement bytes.

`DecodedTokenBytes.tokens` follows request order, including repeated IDs and
skipped entries. Offsets and lengths refer to the joined `value`; a skipped
special entry has zero length. `complete` is true only when every requested ID
was resolved. Unknown IDs fail the entire request and publish no partial bytes
or metadata, even when special-token skipping is enabled.

## Import a ByteLevel vocabulary definition

Read the pinned `tokenizer.json` and preserve its model vocabulary strings.
For each added token, replace the corresponding vocabulary entry with its
literal `content` and its `special` flag, setting `added=True`. A special entry
must also be added. Added entries contribute literal UTF-8 rather than encoded
alphabet characters. Ordinary added tokens, such as Qwen's thinking or tool
markers, remain when `skip_special_tokens=True`.

Set `TOKENIZER_JSON` to the local asset from the same pinned revision as the
weights. This example checks the literal added-token profile used by the
selected models and passes encoded strings to the engine:

```python
import json
import os
from pathlib import Path
import volvoxai as vx

pb = vx.pb
definition = json.loads(Path(os.environ["TOKENIZER_JSON"]).read_text())
if (definition["model"]["type"] != "BPE" or
        definition["decoder"].get("type") != "ByteLevel"):
    raise ValueError("Choose a model with a ByteLevel BPE decoder.")
entries = {index: pb.ByteLevelVocabularyToken(id=index, token=token)
           for token, index in definition["model"]["vocab"].items()}
if len(entries) != len(definition["model"]["vocab"]):
    raise ValueError("Vocabulary IDs must be unique.")
seen = set()
for added in definition.get("added_tokens", []):
    index = added["id"]
    if index in seen or added.get("lstrip") or added.get("rstrip"):
        raise ValueError("Use the validated literal added-token asset profile.")
    seen.add(index)
    if index in entries and entries[index].token != added["content"]:
        raise ValueError("Added token content must agree with the vocabulary.")
    entries[index] = pb.ByteLevelVocabularyToken(
        id=index, token=added["content"], added=True, special=added["special"])

with vx.open_library() as owner:
    text = vx.VxTextServiceClient(owner)
    handle = text.create_tokenizer(pb.CreateTokenizerRequest(
        byte_level=pb.ByteLevelTokenizerVocabulary(tokens=list(entries.values()))))
    try:
        assert handle.token_byte_table_complete
        assert handle.supports_decode_token_bytes and not handle.supports_encode
        # Decode generated IDs here while the owner and tokenizer stay alive.
    finally:
        text.release_tokenizer(pb.TokenizerRef(tokenizer_id=handle.tokenizer_id))
```

The default policy keeps every entry. The `skip_special_tokens` option removes
only entries explicitly marked special. Nonempty merges are refused for the
decode-only profile. It does not infer a normalizer or encoder from merge data.

The returned handle separates two facts:

- `token_byte_table_complete` means every registered ID was converted into an
  immutable byte value. Sparse ID domains and empty entries are valid.
- `byte_alphabet_complete` means all 256 ordinary single-byte tokens exist.
  SmolLM2's selected vocabulary lacks some standalone bytes and remains valid
  for its registered token IDs. A false value does not permit invented UNK IDs.

`supports_encode` and `supports_decode_token_bytes` identify implemented
operations. This definition is sufficient for complete decoding of the supplied
IDs; it is not a complete Hugging Face tokenizer configuration. SentencePiece and
Llama3 tokenizer asset families are not covered by this profile.

## Legacy dictionaries

Existing explicit token entries, JSON dictionaries and binary dictionaries keep
their greedy/BPE encoding behavior. Legacy decoding removes a complete entry's
leading UTF-8 BOM, joins all remaining bytes, applies UTF-8 replacement once and
maps U+0120 to a space. Unknown IDs remain empty placeholders. This intentionally
repairs split-character boundaries from the former per-token decoder while
preserving complete-token values. ByteLevel preserves BOM and performs no space
marker cleanup after its inverse. Legacy raw-byte RPC and special-token skipping
are refused because those policies are not declared by legacy dictionaries.

All profiles use immutable CPU-resident tokenizers and acquire no GPU device.
Release each handle after its final use. The exhaustive message and error
contracts are in [volvoxai.proto](../proto/volvoxai.proto) and the
[generated API reference](generated/api-contract.inference.md).
