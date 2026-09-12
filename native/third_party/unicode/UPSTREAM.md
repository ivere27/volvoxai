Unicode 17.0.0 character categories used by C BPE pretokenization.

- Source: https://www.unicode.org/Public/17.0.0/ucd/UnicodeData.txt
- SHA-256: `2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c`
- License: `LICENSE.txt`, https://www.unicode.org/license.txt
- `tools/generate_tokenizer_unicode.py` downloads the pinned source into memory
  on every generation or `--check` invocation, then verifies the digest before
  projecting only the Letter and Number intervals. No source archive or download
  cache is stored in the repository.
- Generation and verification require network access to the source URL. A
  download failure or digest mismatch fails the command.
- The generated `native/src/generated/tokenizer_unicode.h` and
  `tokenizer_unicode_license.inc` remain committed. Product execution uses these
  embedded projections without network access, host locale or JavaScript regex
  tables.
