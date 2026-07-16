#ifndef VOLVOXAI_TOKENIZER_H
#define VOLVOXAI_TOKENIZER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VolvoxAITokenizer VolvoxAITokenizer;

// Load the vocab.bin file
VolvoxAITokenizer* volvoxai_tokenizer_init(const char* vocab_path, const char* merges_path);

// Free the tokenizer memory
void volvoxai_tokenizer_free(VolvoxAITokenizer* tokenizer);

// Decode a single token ID to its string representation
const char* volvoxai_tokenizer_decode(VolvoxAITokenizer* tokenizer, int token_id);

// Encode a string into an array of token IDs using byte-level BPE merges.
// Returns the number of tokens generated.
int volvoxai_tokenizer_encode(VolvoxAITokenizer* tokenizer, const char* text,
                              int* tokens, int max_tokens);

#ifdef __cplusplus
}
#endif

#endif
