# XZ Embedded upstream

- Project: XZ Embedded
- Repository: https://github.com/tukaani-project/xz-embedded
- Commit: `ae63ae3a36ed01724674e8f3d750dc47bf125410`
- Commit date: 2024-12-30
- License: 0BSD; see `COPYING`

The files in this directory are the minimal userspace decoder set listed by
the upstream README. VolvoxAI does not vendor the optional BCJ, CRC64, SHA-256,
or concatenated-stream decoders.

`xz_config.h` has one local configuration change: `XZ_DEC_SINGLE` is defined
because shader blocks are complete in-memory XZ streams. All other vendored
source and header files are byte-for-byte copies from the commit above.
