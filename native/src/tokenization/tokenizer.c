#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "volvoxai_tokenizer.h"

#define VOLVOXAI_TOKENIZER_MAX_VOCAB (1 << 22)
#define VOLVOXAI_TOKENIZER_MAX_TOKEN_BYTES (1 << 24)
#define VOLVOXAI_TOKENIZER_MAX_MERGES (1 << 24)

struct VolvoxAITokenizer {
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
};

static int tokenizer_encode_bpe_word(VolvoxAITokenizer* t, const char* word, int* tokens, int max_tokens);
static int find_token(VolvoxAITokenizer* t, const char* str);

static void tokenizer_destroy(VolvoxAITokenizer* t) {
    if (!t) return;
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    }
    free(t->merge_left);
    free(t->merge_right);
    free(t->merge_rank);
    free(t->vocab_hash_ids);
    free(t->merge_hash_keys);
    free(t->merge_hash_values);
    free(t->vocab);
    free(t);
}

static uint64_t hash_str(const char* str) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char* p = (const unsigned char*)str;
    while (*p) {
        h ^= (uint64_t)*p++;
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t hash_u64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static int next_pow2(int n) {
    int cap = 1;
    while (cap < n && cap < (1 << 30)) cap <<= 1;
    return cap;
}

static uint64_t pair_key(int left, int right) {
    return ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
}

static int build_vocab_hash(VolvoxAITokenizer* t) {
    int cap = next_pow2(t->vocab_size * 2 + 1);
    int* ids = (int*)malloc((size_t)cap * sizeof(int));
    if (!ids) return 0;
    for (int i = 0; i < cap; i++) ids[i] = -1;

    for (int i = 0; i < t->vocab_size; i++) {
        int slot = (int)(hash_str(t->vocab[i]) & (uint64_t)(cap - 1));
        while (ids[slot] != -1) slot = (slot + 1) & (cap - 1);
        ids[slot] = i;
    }

    t->vocab_hash_cap = cap;
    t->vocab_hash_ids = ids;
    return 1;
}

static int build_merge_hash(VolvoxAITokenizer* t) {
    if (t->merge_count <= 0) return 1;
    int cap = next_pow2(t->merge_count * 2 + 1);
    uint64_t* keys = (uint64_t*)malloc((size_t)cap * sizeof(uint64_t));
    int* values = (int*)malloc((size_t)cap * sizeof(int));
    if (!keys || !values) {
        free(keys);
        free(values);
        return 0;
    }
    for (int i = 0; i < cap; i++) values[i] = -1;

    for (int i = 0; i < t->merge_count; i++) {
        uint64_t key = pair_key(t->merge_left[i], t->merge_right[i]);
        int slot = (int)(hash_u64(key) & (uint64_t)(cap - 1));
        while (values[slot] != -1) {
            if (keys[slot] == key) break;
            slot = (slot + 1) & (cap - 1);
        }
        if (values[slot] == -1) {
            keys[slot] = key;
            values[slot] = i;
        }
    }

    t->merge_hash_cap = cap;
    t->merge_hash_keys = keys;
    t->merge_hash_values = values;
    return 1;
}

static int safe_strncpy(char* dst, const char* src, size_t cap) {
    if (!dst || !src || cap == 0) return 0;
    size_t n = strlen(src);
    if (n + 1 > cap) return 0;
    memcpy(dst, src, n + 1);
    return 1;
}

// Some byte-level BPE vocabularies use "Ġ" to represent a leading space.
static int normalize_merge_token(const char* token, char* out, size_t out_cap) {
    const char* lead = "\xC4\xA0";
    if (!token || !out || out_cap == 0) return 0;
    if (strncmp(token, lead, 2) == 0) {
        size_t n = strlen(token);
        if (n < 2) return 0;
        if (n > out_cap) return 0;
        out[0] = ' ';
        if (!safe_strncpy(out + 1, token + 2, out_cap - 1)) return 0;
        return 1;
    }
    return safe_strncpy(out, token, out_cap);
}

VolvoxAITokenizer* volvoxai_tokenizer_init(const char* vocab_path, const char* merges_path) {
    if (!vocab_path || !vocab_path[0]) return NULL;
    FILE* f = fopen(vocab_path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", vocab_path);
        return NULL;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long file_size = ftell(f);
    if (file_size < (long)sizeof(int32_t) || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }

    int32_t vocab_size = 0;
    if (fread(&vocab_size, sizeof(vocab_size), 1, f) != 1 || vocab_size <= 0 ||
        vocab_size > VOLVOXAI_TOKENIZER_MAX_VOCAB ||
        (long)vocab_size > (file_size - (long)sizeof(vocab_size)) /
            (long)sizeof(int32_t)) {
        fclose(f);
        return NULL;
    }

    VolvoxAITokenizer* t = (VolvoxAITokenizer*)calloc(1, sizeof(VolvoxAITokenizer));
    if (!t) {
        fclose(f);
        return NULL;
    }
    t->vocab_size = (int)vocab_size;
    t->vocab = (char**)calloc((size_t)t->vocab_size, sizeof(char*));
    if (!t->vocab) goto vocab_fail;

    long remaining = file_size - (long)sizeof(vocab_size);

    for (int i = 0; i < t->vocab_size; i++) {
        int32_t len = 0;
        if (remaining < (long)sizeof(len) || fread(&len, sizeof(len), 1, f) != 1) {
            fprintf(stderr, "Failed to read vocab string length at index %d\n", i);
            goto vocab_fail;
        }
        remaining -= (long)sizeof(len);
        if (len < 0 || len > VOLVOXAI_TOKENIZER_MAX_TOKEN_BYTES ||
            (long)len > remaining) {
            fprintf(stderr, "Invalid vocab string length at index %d\n", i);
            goto vocab_fail;
        }

        t->vocab[i] = (char*)malloc((size_t)len + 1);
        if (!t->vocab[i]) {
            fprintf(stderr, "Failed to allocate vocab string for index %d\n", i);
            goto vocab_fail;
        }

        if (len > 0 && fread(t->vocab[i], 1, (size_t)len, f) != (size_t)len) {
            fprintf(stderr, "Failed to read vocab string at index %d\n", i);
            goto vocab_fail;
        }
        t->vocab[i][len] = '\0';
        remaining -= (long)len;
    }

    if (remaining != 0) {
        fprintf(stderr, "Vocab file has trailing data\n");
        goto vocab_fail;
    }
    fclose(f);
    f = NULL;
    if (!build_vocab_hash(t)) {
        tokenizer_destroy(t);
        return NULL;
    }

    if (merges_path && merges_path[0]) {
        FILE* mf = fopen(merges_path, "r");
        if (!mf) {
            fprintf(stderr, "Failed to open merges file %s (fallback to non-merges mode)\n", merges_path);
            return t;
        }

        char line[1024];
        char left_tok[1024];
        char right_tok[1024];
        char left_norm[1024];
        char right_norm[1024];
        int cap = 0;
        int count = 0;
        while (fgets(line, sizeof(line), mf)) {
            if (line[0] == '\0' || line[0] == '\n' || line[0] == '#') {
                continue;
            }

            int len = (int)strlen(line);
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
                line[--len] = '\0';
            }
            if (len == 0) continue;

            if (sscanf(line, "%1023s %1023s", left_tok, right_tok) != 2) {
                continue;
            }

            if (!normalize_merge_token(left_tok, left_norm, sizeof(left_norm))) continue;
            if (!normalize_merge_token(right_tok, right_norm, sizeof(right_norm))) continue;

            int left_id = find_token(t, left_norm);
            int right_id = find_token(t, right_norm);
            if (left_id < 0) left_id = find_token(t, left_tok);
            if (right_id < 0) right_id = find_token(t, right_tok);

            if (left_id < 0 || right_id < 0) {
                continue;
            }

            if (count == cap) {
                if (count >= VOLVOXAI_TOKENIZER_MAX_MERGES) {
                    fclose(mf);
                    tokenizer_destroy(t);
                    return NULL;
                }
                int new_cap = cap ? cap * 2 : 128;
                if (new_cap > VOLVOXAI_TOKENIZER_MAX_MERGES) {
                    new_cap = VOLVOXAI_TOKENIZER_MAX_MERGES;
                }
                int* grow_left = (int*)malloc((size_t)new_cap * sizeof(int));
                int* grow_right = (int*)malloc((size_t)new_cap * sizeof(int));
                int* grow_rank = (int*)malloc((size_t)new_cap * sizeof(int));
                if (!grow_left || !grow_right || !grow_rank) {
                    free(grow_left);
                    free(grow_right);
                    free(grow_rank);
                    fclose(mf);
                    tokenizer_destroy(t);
                    return NULL;
                }
                if (count > 0) {
                    memcpy(grow_left, t->merge_left, (size_t)count * sizeof(int));
                    memcpy(grow_right, t->merge_right, (size_t)count * sizeof(int));
                    memcpy(grow_rank, t->merge_rank, (size_t)count * sizeof(int));
                }
                free(t->merge_left);
                free(t->merge_right);
                free(t->merge_rank);
                t->merge_left = grow_left;
                t->merge_right = grow_right;
                t->merge_rank = grow_rank;
                cap = new_cap;
            }

            t->merge_left[count] = left_id;
            t->merge_right[count] = right_id;
            t->merge_rank[count] = count;
            count++;
        }
        fclose(mf);
        t->merge_count = count;
        if (!build_merge_hash(t)) {
            tokenizer_destroy(t);
            return NULL;
        }
    }

    return t;

vocab_fail:
    fclose(f);
    tokenizer_destroy(t);
    return NULL;
}

void volvoxai_tokenizer_free(VolvoxAITokenizer* t) {
    tokenizer_destroy(t);
}

const char* volvoxai_tokenizer_decode(VolvoxAITokenizer* t, int token_id) {
    if (!t || token_id < 0 || token_id >= t->vocab_size) return "";
    return t->vocab[token_id];
}

static int find_token(VolvoxAITokenizer* t, const char* str) {
    if (t->vocab_hash_ids && t->vocab_hash_cap > 0) {
        int cap = t->vocab_hash_cap;
        int slot = (int)(hash_str(str) & (uint64_t)(cap - 1));
        for (int probe = 0; probe < cap; probe++) {
            int id = t->vocab_hash_ids[slot];
            if (id < 0) return -1;
            if (strcmp(t->vocab[id], str) == 0) return id;
            slot = (slot + 1) & (cap - 1);
        }
        return -1;
    }
    for (int i = 0; i < t->vocab_size; i++) {
        if (strcmp(t->vocab[i], str) == 0) return i;
    }
    return -1;
}

static int utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

static int is_space_char(unsigned char c) {
    return c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '\f' || c == '\v';
}

static int is_ascii_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

static int is_ascii_digit(unsigned char c) {
    return (c >= '0' && c <= '9');
}

static int match_ascii_contraction(const char* text, int len, int pos, int* matched_len) {
    if (pos >= len || text[pos] != '\'') return 0;

    if (pos + 1 < len && text[pos + 1] == 's') {
        *matched_len = 2;
        return 1;
    }
    if (pos + 1 < len && text[pos + 1] == 't') {
        *matched_len = 2;
        return 1;
    }
    if (pos + 2 < len && text[pos + 1] == 'r' && text[pos + 2] == 'e') {
        *matched_len = 3;
        return 1;
    }
    if (pos + 2 < len && text[pos + 1] == 'v' && text[pos + 2] == 'e') {
        *matched_len = 3;
        return 1;
    }
    if (pos + 1 < len && text[pos + 1] == 'm') {
        *matched_len = 2;
        return 1;
    }
    if (pos + 2 < len && text[pos + 1] == 'l' && text[pos + 2] == 'l') {
        *matched_len = 3;
        return 1;
    }
    if (pos + 1 < len && text[pos + 1] == 'd') {
        *matched_len = 2;
        return 1;
    }
    return 0;
}

static int tokenizer_encode_chunk(VolvoxAITokenizer* t, const char* text, int start, int end, int add_leading_space, int* tokens, int max_tokens) {
    if (!tokens || max_tokens <= 0 || end <= start) return 0;

    int chunk_len = end - start;
    int total_len = chunk_len + (add_leading_space ? 1 : 0);
    char* chunk = (char*)malloc((size_t)total_len + 1);
    if (!chunk) return 0;

    if (add_leading_space) chunk[0] = ' ';
    if (chunk_len > 0) {
        memcpy(chunk + (add_leading_space ? 1 : 0), text + start, (size_t)chunk_len);
    }
    chunk[total_len] = '\0';

    int out = tokenizer_encode_bpe_word(t, chunk, tokens, max_tokens);
    free(chunk);
    return out;
}

static int pair_rank(const VolvoxAITokenizer* t, int left, int right) {
    if (t->merge_hash_values && t->merge_hash_cap > 0) {
        uint64_t key = pair_key(left, right);
        int cap = t->merge_hash_cap;
        int slot = (int)(hash_u64(key) & (uint64_t)(cap - 1));
        for (int probe = 0; probe < cap; probe++) {
            int idx = t->merge_hash_values[slot];
            if (idx < 0) return INT_MAX;
            if (t->merge_hash_keys[slot] == key) return t->merge_rank[idx];
            slot = (slot + 1) & (cap - 1);
        }
        return INT_MAX;
    }
    for (int i = 0; i < t->merge_count; i++) {
        if (t->merge_left[i] == left && t->merge_right[i] == right) {
            return t->merge_rank[i];
        }
    }
    return INT_MAX;
}

static int concat_and_lookup(const VolvoxAITokenizer* t, int left_id, int right_id, int* merged_id_out) {
    const char* left = t->vocab[left_id];
    const char* right = t->vocab[right_id];
    int left_len = (int)strlen(left);
    int right_len = (int)strlen(right);

    char* merged = (char*)malloc((size_t)left_len + (size_t)right_len + 1);
    if (!merged) return 0;

    memcpy(merged, left, (size_t)left_len);
    memcpy(merged + left_len, right, (size_t)right_len);
    merged[left_len + right_len] = '\0';

    int merged_id = find_token((VolvoxAITokenizer*)t, merged);
    free(merged);
    if (merged_id < 0) return 0;

    *merged_id_out = merged_id;
    return 1;
}

static int tokenizer_encode_greedy(VolvoxAITokenizer* t, const char* text, int* tokens, int max_tokens) {
    int num_tokens = 0;
    int len = (int)strlen(text);
    int pos = 0;

    while (pos < len && num_tokens < max_tokens) {
        int best_id = -1;
        int best_len = 0;

        for (int i = 0; i < t->vocab_size; i++) {
            int vlen = (int)strlen(t->vocab[i]);
            if (vlen > best_len && vlen <= len - pos) {
                if (strncmp(text + pos, t->vocab[i], vlen) == 0) {
                    best_id = i;
                    best_len = vlen;
                }
            }
        }

        if (best_id == -1) {
            pos++;
            continue;
        }

        tokens[num_tokens++] = best_id;
        pos += best_len;
    }
    return num_tokens;
}

static int tokenizer_encode_bpe_word(VolvoxAITokenizer* t, const char* word, int* tokens, int max_tokens) {
    int word_len = (int)strlen(word);
    if (word_len == 0 || max_tokens <= 0) return 0;

    int* work = (int*)malloc((size_t)word_len * sizeof(int));
    if (!work) return 0;
    int work_len = 0;
    int pos = 0;

    while (pos < word_len) {
        int clen = utf8_len((unsigned char)word[pos]);
        if (pos + clen > word_len) clen = word_len - pos;

        char symbol[8] = {0};
        memcpy(symbol, word + pos, (size_t)clen);
        int id = find_token(t, symbol);
        if (id != -1) {
            work[work_len++] = id;
        } else {
            for (int i = 0; i < clen; i++) {
                char byte_buf[2] = {word[pos + i], '\0'};
                int byte_id = find_token(t, byte_buf);
                if (byte_id != -1) work[work_len++] = byte_id;
            }
        }
        pos += clen;
    }

    while (1) {
        int best_idx = -1;
        int best_rank = INT_MAX;
        int best_id = -1;

        for (int i = 0; i < work_len - 1; i++) {
            int r = pair_rank(t, work[i], work[i + 1]);
            if (r >= best_rank) continue;

            int merged_id = -1;
            if (!concat_and_lookup(t, work[i], work[i + 1], &merged_id)) continue;

            best_rank = r;
            best_idx = i;
            best_id = merged_id;
        }

        if (best_idx == -1) break;
        work[best_idx] = best_id;
        for (int i = best_idx + 1; i < work_len - 1; i++) {
            work[i] = work[i + 1];
        }
        work_len--;
    }

    int num_tokens = work_len < max_tokens ? work_len : max_tokens;
    for (int i = 0; i < num_tokens; i++) {
        tokens[i] = work[i];
    }
    free(work);
    return num_tokens;
}

int volvoxai_tokenizer_encode(VolvoxAITokenizer* t, const char* text, int* tokens, int max_tokens) {
    if (!t || !text || !tokens || max_tokens <= 0) return 0;
    size_t text_size = strlen(text);
    if (text_size > INT_MAX) return 0;
    if (t->merge_left == NULL || t->merge_count == 0) {
        return tokenizer_encode_greedy(t, text, tokens, max_tokens);
    }

    int text_len = (int)text_size;
    if (text_len == 0) return 0;

    int num_tokens = 0;
    int pos = 0;
    int add_leading_space = 0;
    int remain;
    int match_len;
    int start;
    int end;

    while (pos < text_len && num_tokens < max_tokens) {
        unsigned char c = (unsigned char)text[pos];

        if (is_space_char(c)) {
            int ws = pos;
            while (pos < text_len && is_space_char((unsigned char)text[pos])) pos++;
            int ws_len = pos - ws;

            if (pos >= text_len) {
                remain = max_tokens - num_tokens;
                if (remain <= 0) break;
                num_tokens += tokenizer_encode_chunk(t, text, ws, pos, 0, tokens + num_tokens, remain);
                break;
            }

            if (ws_len == 1) {
                add_leading_space = 1;
                continue;
            }

            remain = max_tokens - num_tokens;
            if (remain <= 0) break;
            num_tokens += tokenizer_encode_chunk(t, text, ws, pos, 0, tokens + num_tokens, remain);
            add_leading_space = 0;
            continue;
        }

        if (match_ascii_contraction(text, text_len, pos, &match_len)) {
            remain = max_tokens - num_tokens;
            if (remain <= 0) break;
            num_tokens += tokenizer_encode_chunk(t, text, pos, pos + match_len, 0, tokens + num_tokens, remain);
            pos += match_len;
            add_leading_space = 0;
            continue;
        }

        start = pos;
        if (is_ascii_letter((unsigned char)text[pos])) {
            pos++;
            while (pos < text_len && is_ascii_letter((unsigned char)text[pos])) pos++;
            end = pos;
        } else if (is_ascii_digit((unsigned char)text[pos])) {
            pos++;
            while (pos < text_len && is_ascii_digit((unsigned char)text[pos])) pos++;
            end = pos;
        } else {
            while (pos < text_len &&
                   !is_space_char((unsigned char)text[pos]) &&
                   !is_ascii_letter((unsigned char)text[pos]) &&
                   !is_ascii_digit((unsigned char)text[pos])) {
                pos++;
            }
            end = pos;
            if (end == start) {
                pos++;
                end = pos;
            }
        }

        if (add_leading_space) {
            remain = max_tokens - num_tokens;
            if (remain <= 0) break;
            num_tokens += tokenizer_encode_chunk(t, text, start, end, 1, tokens + num_tokens, remain);
        } else {
            remain = max_tokens - num_tokens;
            if (remain <= 0) break;
            num_tokens += tokenizer_encode_chunk(t, text, start, end, 0, tokens + num_tokens, remain);
        }
        add_leading_space = 0;
    }

    return num_tokens;
}
