#include "volvoxai_tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

static int write_bytes(const char* path, const void* data, size_t size) {
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    int failed = fwrite(data, 1, size, file) != size;
    failed |= fclose(file) != 0;
    return failed ? -1 : 0;
}

static int write_valid_vocab(const char* path) {
    static const char* tokens[] = {"a", "b", "ab"};
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    int32_t count = 3;
    int failed = fwrite(&count, sizeof(count), 1, file) != 1;
    for (int index = 0; !failed && index < count; index++) {
        int32_t length = (int32_t)strlen(tokens[index]);
        failed |= fwrite(&length, sizeof(length), 1, file) != 1;
        failed |= fwrite(tokens[index], 1, (size_t)length, file) != (size_t)length;
    }
    failed |= fclose(file) != 0;
    return failed ? -1 : 0;
}

static int expect_invalid(const char* path, const void* data, size_t size) {
    if (write_bytes(path, data, size) != 0) return -1;
    VolvoxAITokenizer* tokenizer = volvoxai_tokenizer_init(path, NULL);
    volvoxai_tokenizer_free(tokenizer);
    return tokenizer ? -1 : 0;
}

int main(void) {
    const char* vocab_path = "/tmp/volvoxai-tokenizer-vocab.bin";
    const char* merges_path = "/tmp/volvoxai-tokenizer-merges.txt";
    const char* invalid_path = "/tmp/volvoxai-tokenizer-invalid.bin";
    CHECK(volvoxai_tokenizer_init(NULL, NULL) == NULL);
    CHECK(volvoxai_tokenizer_init("", NULL) == NULL);

    CHECK(write_valid_vocab(vocab_path) == 0);
    CHECK(write_bytes(merges_path, "a b\n", 4) == 0);
    VolvoxAITokenizer* tokenizer = volvoxai_tokenizer_init(vocab_path, merges_path);
    CHECK(tokenizer != NULL);
    int encoded[4] = {-1, -1, -1, -1};
    CHECK(volvoxai_tokenizer_encode(tokenizer, "ab", encoded, 4) == 1);
    CHECK(encoded[0] == 2);
    CHECK(strcmp(volvoxai_tokenizer_decode(tokenizer, 2), "ab") == 0);
    CHECK(strcmp(volvoxai_tokenizer_decode(tokenizer, -1), "") == 0);
    volvoxai_tokenizer_free(tokenizer);

    const unsigned char short_header[] = {1, 0};
    CHECK(expect_invalid(invalid_path, short_header, sizeof(short_header)) == 0);
    const int32_t zero_count = 0;
    CHECK(expect_invalid(invalid_path, &zero_count, sizeof(zero_count)) == 0);
    const int32_t negative_count = -1;
    CHECK(expect_invalid(invalid_path, &negative_count, sizeof(negative_count)) == 0);
    const int32_t huge_count = INT32_MAX;
    CHECK(expect_invalid(invalid_path, &huge_count, sizeof(huge_count)) == 0);

    const int32_t negative_length[] = {1, -1};
    CHECK(expect_invalid(invalid_path, negative_length, sizeof(negative_length)) == 0);
    const int32_t truncated_token[] = {1, 8};
    CHECK(expect_invalid(invalid_path, truncated_token, sizeof(truncated_token)) == 0);
    const unsigned char embedded_nul[] = {
        1, 0, 0, 0,
        2, 0, 0, 0,
        'a', 0,
    };
    CHECK(write_bytes(invalid_path, embedded_nul, sizeof(embedded_nul)) == 0);
    tokenizer = volvoxai_tokenizer_init(invalid_path, NULL);
    CHECK(tokenizer != NULL);
    /* vocab.bin is byte-oriented like the browser loader. The C-string decode
     * surface exposes the prefix of a token containing NUL, but the declared
     * byte length must still load safely and without consuming adjacent data. */
    CHECK(strcmp(volvoxai_tokenizer_decode(tokenizer, 0), "a") == 0);
    volvoxai_tokenizer_free(tokenizer);
    const unsigned char trailing_data[] = {
        1, 0, 0, 0,
        1, 0, 0, 0,
        'a', 'x',
    };
    CHECK(expect_invalid(invalid_path, trailing_data, sizeof(trailing_data)) == 0);

    remove(vocab_path);
    remove(merges_path);
    remove(invalid_path);
    puts("native tokenizer tests passed");
    return 0;
}
