/* Independent byte identities and whole-message UTF-8 boundary regressions. */
#include "tokenizer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t encode_scalar(unsigned cp, uint8_t* out) {
    if (cp < 128) { out[0] = (uint8_t)cp; return 1; }
    out[0] = (uint8_t)(0xc0 | (cp >> 6));
    out[1] = (uint8_t)(0x80 | (cp & 63)); return 2;
}

static void test_alphabet_and_flags(void) {
    uint8_t strings[256][2]; VxVocabularyToken entries[259];
    unsigned extra = 256;
    for (unsigned b = 0; b < 256; ++b) {
        int visible = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
        unsigned cp = visible ? b : extra++;
        entries[b] = (VxVocabularyToken){b * 7u + 10u, strings[b], encode_scalar(cp, strings[b]), 0, 0};
    }
    const uint8_t literal[] = "<think>\xed\x95\x9c";
    const uint8_t special[] = "<eos>";
    entries[256] = (VxVocabularyToken){9000, literal, sizeof(literal) - 1, 1, 0};
    entries[257] = (VxVocabularyToken){9001, special, sizeof(special) - 1, 1, 1};
    entries[258] = (VxVocabularyToken){9002, NULL, 0, 0, 0};
    VxTokenizerSource source = {.tokens = entries, .token_count = 259, .format = 5};
    VxTokenizer* t = NULL; assert(vx_tokenizer_create(&source, &t) == VX_STATUS_OK);
    assert(vx_tokenizer_profile(t) == VX_TOKENIZER_PROFILE_BYTE_LEVEL_DECODE_ONLY);
    assert(vx_tokenizer_byte_alphabet_complete(t));
    /* Creation snapshots both raw bytes and flags. */
    memset(strings, 0, sizeof(strings)); entries[256].special = 1;
    uint32_t ids[259]; uint8_t expected[256];
    for (unsigned i = 0; i < 256; ++i) { expected[i] = (uint8_t)(i * 53u); ids[i] = expected[i] * 7u + 10u; }
    ids[256] = 9001; ids[257] = 9000; ids[258] = 9002;
    uint8_t* bytes = NULL; size_t count = 0; VxTokenByteMetadata* metadata = NULL; VxOperationCode code;
    assert(vx_tokenizer_decode_bytes(t, ids, 259, 1, &bytes, &count, &metadata, &code) == VX_STATUS_OK);
    assert(count == 256 + sizeof(literal) - 1 && !memcmp(bytes, expected, 256));
    assert(!memcmp(bytes + 256, literal, sizeof(literal) - 1));
    for (unsigned i = 0; i < 256; ++i) {
        assert(metadata[i].id == ids[i] && metadata[i].offset == i && metadata[i].length == 1);
        assert(!metadata[i].added && !metadata[i].special && !metadata[i].skipped);
    }
    assert(metadata[256].offset == 256 && !metadata[256].length && metadata[256].skipped);
    assert(metadata[257].offset == 256 && !metadata[257].special && !metadata[257].skipped);
    assert(metadata[258].offset == count && !metadata[258].length && !metadata[258].skipped);
    free(bytes); free(metadata);
    ids[258] = 8; /* unknown follows valid IDs: no partial result may escape. */
    assert(vx_tokenizer_decode_bytes(t, ids, 259, 1, &bytes, &count, &metadata, &code) == VX_STATUS_INVALID_ARGUMENT);
    assert(code == VX_CODE_TOKEN_ID_UNKNOWN && !bytes && !count && !metadata);
    assert(vx_tokenizer_decode(t, ids, 259, 0, &bytes, &count, &code) == VX_STATUS_INVALID_ARGUMENT);
    assert(code == VX_CODE_TOKEN_ID_UNKNOWN && !bytes && !count);
    uint32_t* encoded = NULL; size_t encoded_count;
    assert(vx_tokenizer_encode(t, (const uint8_t*)"x", 1, 256, 0, &encoded, &encoded_count) == VX_STATUS_INVALID_ARGUMENT);
    assert(!encoded && !encoded_count);
    vx_tokenizer_release(t);
}

static void test_bytelevel_utf8_and_invalid_definitions(void) {
    /* Encoded ByteLevel strings for EF BB BF, E2 82 AC, FF and 00. */
    const uint8_t* strings[] = {(const uint8_t*)"\xc3\xaf", (const uint8_t*)"\xc2\xbb",
        (const uint8_t*)"\xc2\xbf", (const uint8_t*)"\xc3\xa2", (const uint8_t*)"\xc4\xa4",
        (const uint8_t*)"\xc2\xac", (const uint8_t*)"\xc3\xbf", (const uint8_t*)"\xc4\x80"};
    VxVocabularyToken entries[8];
    for (unsigned i = 0; i < 8; ++i) entries[i] = (VxVocabularyToken){i, strings[i], 2, 0, 0};
    VxTokenizerSource source = {.tokens = entries, .token_count = 8, .format = 5};
    VxTokenizer* t = NULL; assert(vx_tokenizer_create(&source, &t) == VX_STATUS_OK);
    assert(!vx_tokenizer_byte_alphabet_complete(t));
    uint32_t ids[] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint8_t expected[] = {0xef, 0xbb, 0xbf, 0xe2, 0x82, 0xac, 0xef, 0xbf, 0xbd, 0};
    uint8_t* text; size_t bytes; VxOperationCode code;
    assert(vx_tokenizer_decode(t, ids, 8, 0, &text, &bytes, &code) == VX_STATUS_OK);
    assert(bytes == sizeof(expected) && !memcmp(text, expected, bytes)); free(text);
    assert(vx_tokenizer_decode(t, ids + 3, 2, 0, &text, &bytes, &code) == VX_STATUS_OK);
    assert(bytes == 3 && !memcmp(text, "\xef\xbf\xbd", 3)); free(text);
    vx_tokenizer_release(t);
    VxVocabularyToken invalid[] = {{0, (const uint8_t*)"\xed\x95\x9c", 3, 0, 0},
        {0, (const uint8_t*)"a", 1, 0, 1}, {0, (const uint8_t*)"\xff", 1, 1, 0}};
    for (unsigned i = 0; i < 3; ++i) {
        source.tokens = &invalid[i]; source.token_count = 1;
        assert(vx_tokenizer_create(&source, &t) == VX_STATUS_INVALID_ARGUMENT && !t);
    }
    source.tokens = entries; source.token_count = 8; source.merges = (const uint8_t*)"a b"; source.merge_bytes = 3;
    assert(vx_tokenizer_create(&source, &t) == VX_STATUS_INVALID_ARGUMENT && !t);
    source.merge_bytes = 0; entries[1].id = 0;
    assert(vx_tokenizer_create(&source, &t) == VX_STATUS_INVALID_ARGUMENT && !t);
}

static void test_legacy_join_and_bom(void) {
    const uint8_t* raw[] = {(const uint8_t*)"\xe2", (const uint8_t*)"\x82\xac",
        (const uint8_t*)"\xef\xbb\xbfX", (const uint8_t*)"x\xef\xbb\xbfy",
        (const uint8_t*)"\xef", (const uint8_t*)"\xbb\xbf", (const uint8_t*)"\xc4", (const uint8_t*)"\xa0"};
    size_t lengths[] = {1, 2, 4, 5, 1, 2, 1, 1}; VxVocabularyToken entries[8];
    for (unsigned i = 0; i < 8; ++i) entries[i] = (VxVocabularyToken){i, raw[i], lengths[i], 0, 0};
    VxTokenizerSource source = {.tokens = entries, .token_count = 8, .format = 1};
    VxTokenizer* t; assert(vx_tokenizer_create(&source, &t) == VX_STATUS_OK);
    uint32_t ids[] = {0, 1, 2, 3, 999, 4, 5, 6, 7};
    const uint8_t expected[] = "\xe2\x82\xacXx\xef\xbb\xbfy\xef\xbb\xbf ";
    uint8_t* text; size_t bytes; VxOperationCode code;
    assert(vx_tokenizer_decode(t, ids, 9, 0, &text, &bytes, &code) == VX_STATUS_OK);
    assert(bytes == sizeof(expected) - 1 && !memcmp(text, expected, bytes)); free(text);
    assert(vx_tokenizer_decode(t, ids, 9, 1, &text, &bytes, &code) == VX_STATUS_INVALID_ARGUMENT);
    assert(code == VX_CODE_TOKENIZER_OPERATION_UNSUPPORTED && !text && !bytes);
    VxTokenByteMetadata* metadata;
    assert(vx_tokenizer_decode_bytes(t, ids, 9, 0, &text, &bytes, &metadata, &code) == VX_STATUS_INVALID_ARGUMENT);
    assert(code == VX_CODE_TOKENIZER_OPERATION_UNSUPPORTED && !text && !metadata && !bytes);
    vx_tokenizer_release(t);
}

int main(void) {
    test_alphabet_and_flags(); test_bytelevel_utf8_and_invalid_definitions(); test_legacy_join_and_bom();
    puts("test_tokenizer_bytes: PASS"); return 0;
}
