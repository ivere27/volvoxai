/* Immutable text semantics shared by native and WASM. Private API only. */
#ifndef VOLVOXAI_TOKENIZER_H
#define VOLVOXAI_TOKENIZER_H
#include "vx_lifecycle.h"

typedef struct VxTokenizer VxTokenizer;
typedef struct VxVocabularyToken {
    uint32_t id;
    const uint8_t* bytes;
    size_t length;
    int added, special; /* Used by encoded ByteLevel vocabulary. */
} VxVocabularyToken;

typedef struct VxTokenizerSource {
    const VxVocabularyToken* tokens;
    size_t token_count;
    const uint8_t* json;
    size_t json_bytes;
    const uint8_t* binary;
    size_t binary_bytes;
    const uint8_t* merges;
    size_t merge_bytes;
    int format; /* Public vocabulary oneof tag; 6 is complete HF tokenizer JSON. */
} VxTokenizerSource;

typedef struct VxTokenByteMetadata {
    uint32_t id;
    size_t offset, length;
    int added, special, skipped;
} VxTokenByteMetadata;

VxStatus vx_tokenizer_create(const VxTokenizerSource*, VxTokenizer**);
void vx_tokenizer_retain(void*);
void vx_tokenizer_release(void*);
uint32_t vx_tokenizer_size(const VxTokenizer*);
uint32_t vx_tokenizer_merges(const VxTokenizer*);
VxTokenizerProfile vx_tokenizer_profile(const VxTokenizer*);
int vx_tokenizer_byte_alphabet_complete(const VxTokenizer*);
/* The returned arrays belong to the caller and are freed with free(). */
VxStatus vx_tokenizer_encode(const VxTokenizer*, const uint8_t*, size_t,
                              uint32_t limit, int mode,
                              uint32_t** ids, size_t* count);
VxStatus vx_tokenizer_encode_options(const VxTokenizer*, const uint8_t*, size_t,
                              uint32_t limit, int mode, int add_special_tokens,
                              uint32_t** ids, size_t* count);
VxStatus vx_tokenizer_decode(const VxTokenizer*, const uint32_t*, size_t,
                              int skip_special, uint8_t** text, size_t* bytes,
                              VxOperationCode* code);
/* Complete, atomic lossless bytes for the ByteLevel decode-only profile. */
VxStatus vx_tokenizer_decode_bytes(const VxTokenizer*, const uint32_t*, size_t,
                              int skip_special, uint8_t** value, size_t* bytes,
                              VxTokenByteMetadata** metadata,
                              VxOperationCode* code);
#endif
