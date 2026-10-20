/* Functional text semantics, independent of model weights and GPU availability. */
#include "tokenizer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char fixture[] =
    "{\"version\":\"1.0\",\"normalizer\":null,"
    "\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":true},"
    "\"decoder\":{\"type\":\"ByteLevel\"},"
    "\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"b\":1,\"ab\":2,\"Ġ\":3},\"merges\":[[\"a\",\"b\"]]},"
    "\"added_tokens\":[{\"id\":4,\"content\":\"<s>\",\"special\":true,\"normalized\":false},"
    "{\"id\":5,\"content\":\"[x]\",\"special\":false,\"normalized\":false,\"lstrip\":true,\"rstrip\":true}],"
    "\"post_processor\":{\"type\":\"TemplateProcessing\",\"single\":[{\"SpecialToken\":{\"id\":\"<s>\",\"type_id\":0}},"
    "{\"Sequence\":{\"id\":\"A\",\"type_id\":0}}],\"special_tokens\":{\"<s>\":{\"id\":\"<s>\",\"ids\":[4],\"tokens\":[\"<s>\"]}}}}";

static void byte_level(void) {
    VxTokenizerSource source = {.json = (const uint8_t*)fixture, .json_bytes = sizeof(fixture) - 1, .format = 6};
    VxTokenizer* tokenizer = NULL;
    assert(vx_tokenizer_create(&source, &tokenizer) == VX_STATUS_OK);
    assert(vx_tokenizer_profile(tokenizer) == VX_TOKENIZER_PROFILE_HF_BYTE_LEVEL_BPE);
    assert(vx_tokenizer_merges(tokenizer) == 1);
    uint32_t* ids = NULL; size_t count = 0;
    const char input[] = "ab [x] ab";
    const uint32_t expected[] = {4, 2, 5, 2};
    assert(vx_tokenizer_encode_options(tokenizer, (const uint8_t*)input, sizeof(input) - 1,
        32, 0, 1, &ids, &count) == VX_STATUS_OK);
    assert(count == 4 && !memcmp(ids, expected, sizeof(expected)));
    free(ids);
    assert(vx_tokenizer_encode_options(tokenizer, (const uint8_t*)input, sizeof(input) - 1,
        2, 0, 0, &ids, &count) == VX_STATUS_OK);
    assert(count == 2 && ids[0] == 2 && ids[1] == 5);
    free(ids);
    assert(vx_tokenizer_encode_options(tokenizer, (const uint8_t*)input, sizeof(input) - 1,
        32, 1, 1, &ids, &count) == VX_STATUS_INVALID_ARGUMENT && !ids && !count);
    assert(vx_tokenizer_encode_options(tokenizer, (const uint8_t*)input, sizeof(input) - 1,
        0, 0, 1, &ids, &count) == VX_STATUS_OK && !ids && !count);
    uint8_t* text = NULL; size_t length = 0; VxOperationCode code;
    assert(vx_tokenizer_decode(tokenizer, expected, 4, 1, &text, &length, &code) == VX_STATUS_OK);
    assert(length == 7 && !memcmp(text, "ab[x]ab", 7)); free(text);
    VxTokenByteMetadata* metadata = NULL;
    assert(vx_tokenizer_decode_bytes(tokenizer, expected, 4, 1, &text, &length, &metadata, &code) == VX_STATUS_OK);
    assert(length == 7 && metadata[0].skipped && metadata[2].added && !metadata[2].special);
    free(text); free(metadata);
    const uint32_t unknown[] = {2, 999, 4};
    assert(vx_tokenizer_decode_bytes(tokenizer, unknown, 3, 1, &text, &length, &metadata, &code) == VX_STATUS_INVALID_ARGUMENT);
    assert(code == VX_CODE_TOKEN_ID_UNKNOWN && !text && !length && !metadata);
    const uint8_t malformed[] = {0xed, 0xa0, 0x80};
    assert(vx_tokenizer_encode_options(tokenizer, malformed, 3, 32, 0, 1, &ids, &count) == VX_STATUS_INVALID_ARGUMENT && !ids && !count);
    vx_tokenizer_release(tokenizer);
    source.merges = (const uint8_t*)"a b"; source.merge_bytes = 3;
    assert(vx_tokenizer_create(&source, &tokenizer) == VX_STATUS_INVALID_ARGUMENT && !tokenizer);
}

static void strict_json(void) {
    const char* invalid[] = {
        "{\"model\":{\"type\":\"WordPiece\"}}",
        "{\"model\":{},\"model\":{}}",
        "{\"normalizer\":\"\\ud800\"}",
        "{\"model\":{\"type\\u0000hidden\":\"BPE\"}}",
        "{} trailing"
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        VxTokenizerSource source = {.json = (const uint8_t*)invalid[i], .json_bytes = strlen(invalid[i]), .format = 6};
        VxTokenizer* tokenizer = NULL;
        assert(vx_tokenizer_create(&source, &tokenizer) == VX_STATUS_INVALID_ARGUMENT && !tokenizer);
    }
}

int main(void) {
    byte_level(); strict_json();
    puts("test_tokenizer_hf: PASS");
    return 0;
}
