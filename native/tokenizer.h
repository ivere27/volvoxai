#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>

typedef struct {
    char** vocab;
    int vocab_size;
    int* merge_left;
    int* merge_right;
    int* merge_rank;
    int merge_count;
    int vocab_hash_cap;
    int* vocab_hash_ids;
    int merge_hash_cap;
    uint64_t* merge_hash_keys;
    int* merge_hash_values;
} Tokenizer;

// Load the vocab.bin file
Tokenizer* tokenizer_init(const char* vocab_path, const char* merges_path);

// Free the tokenizer memory
void tokenizer_free(Tokenizer* t);

// Decode a single token ID to its string representation
const char* tokenizer_decode(Tokenizer* t, int token_id);

// Encode a string into an array of token IDs using byte-level BPE merges.
// Returns the number of tokens generated.
int tokenizer_encode(Tokenizer* t, const char* text, int* tokens, int max_tokens);

#endif // TOKENIZER_H
