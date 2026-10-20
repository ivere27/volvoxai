/* Token dictionaries, UTF-8, pretokenization and ranked BPE live in C.
 * Instances are immutable. Each encode/decode owns its temporary storage. */
#include "tokenizer.h"
#include "generated/tokenizer_unicode.h"
#include "generated/tokenizer_unicode_license.inc"
#include "json_validation.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct VxTextBuffer {
    uint8_t* data;
    size_t length, capacity;
} VxTextBuffer;
typedef struct VxToken {
    uint32_t id;
    VxTextBuffer raw, text;
    size_t next; /* Raw-byte first-character bucket, index + 1. */
    int added, special;
    int single_word, lstrip, rstrip, normalized;
    VxTextBuffer match;
} VxToken;
typedef struct VxTokenMap { size_t* slots; size_t capacity; } VxTokenMap;
typedef struct VxMerge {
    uint32_t left, right, merged, rank;
    int occupied;
} VxMerge;
struct VxTokenizer {
    atomic_uint references;
    VxToken* tokens;
    size_t count, capacity;
    VxTokenMap raw, text;
    size_t first[256];
    VxMerge* merges;
    size_t merge_capacity;
    uint32_t merge_count;
    VxTokenizerProfile profile;
    int byte_alphabet_complete;
    int hf_pretokenizer, hf_normalizer, hf_prefix_space, hf_ignore_merges;
    int hf_has_unk, hf_fuse_unk;
    uint32_t hf_unk, hf_fallback[256];
    size_t* added;
    size_t added_count;
    VxTextBuffer prefix, suffix;
};

static int vx_text_append(VxTextBuffer* buffer, const void* data, size_t bytes) {
    if (bytes > SIZE_MAX - buffer->length || (bytes && !data)) return 0;
    size_t need = buffer->length + bytes;
    if (need > buffer->capacity) {
        size_t capacity = buffer->capacity ? buffer->capacity : 32u;
        while (capacity < need) {
            if (capacity > SIZE_MAX / 2u) { capacity = need; break; }
            capacity *= 2u;
        }
        void* next = realloc(buffer->data, capacity);
        if (!next) return 0;
        buffer->data = next;
        buffer->capacity = capacity;
    }
    if (bytes) memcpy(buffer->data + buffer->length, data, bytes);
    buffer->length = need;
    return 1;
}

/* WHATWG UTF-8 replacement consumes a valid prefix of an incomplete sequence,
 * leaving the first offending byte for the next character. */
static size_t vx_text_character(const uint8_t* p, size_t n,
                                uint32_t* point, int* valid) {
    *valid = 1;
    if (!n) return 0;
    if (p[0] < 0x80) { *point = p[0]; return 1; }
    size_t width;
    uint32_t cp;
    if (p[0] >= 0xc2 && p[0] <= 0xdf) { width = 2; cp = p[0] & 31u; }
    else if (p[0] >= 0xe0 && p[0] <= 0xef) { width = 3; cp = p[0] & 15u; }
    else if (p[0] >= 0xf0 && p[0] <= 0xf4) { width = 4; cp = p[0] & 7u; }
    else { *point = 0xfffd; *valid = 0; return 1; }
    for (size_t i = 1; i < width; ++i) {
        if (i >= n || p[i] < 0x80 || p[i] > 0xbf ||
            (i == 1 && ((p[0] == 0xe0 && p[i] < 0xa0) ||
                        (p[0] == 0xed && p[i] > 0x9f) ||
                        (p[0] == 0xf0 && p[i] < 0x90) ||
                        (p[0] == 0xf4 && p[i] > 0x8f)))) {
            *point = 0xfffd; *valid = 0; return i;
        }
        cp = (cp << 6) | (p[i] & 63u);
    }
    *point = cp;
    return width;
}

static int vx_text_emit(VxTextBuffer* output, uint32_t cp) {
    uint8_t p[4]; size_t n;
    if (cp < 0x80) { p[0] = (uint8_t)cp; n = 1; }
    else if (cp < 0x800) {
        p[0] = 0xc0 | (cp >> 6); p[1] = 0x80 | (cp & 63); n = 2;
    } else if (cp < 0x10000) {
        p[0] = 0xe0 | (cp >> 12); p[1] = 0x80 | ((cp >> 6) & 63);
        p[2] = 0x80 | (cp & 63); n = 3;
    } else {
        p[0] = 0xf0 | (cp >> 18); p[1] = 0x80 | ((cp >> 12) & 63);
        p[2] = 0x80 | ((cp >> 6) & 63); p[3] = 0x80 | (cp & 63); n = 4;
    }
    return vx_text_append(output, p, n);
}

static int vx_text_utf8(VxTextBuffer* output, const uint8_t* p, size_t n,
                        int strip_bom, int spaces) {
    if (n && !p) return 0;
    for (size_t offset = 0; offset < n;) {
        uint32_t cp; int valid;
        size_t width = vx_text_character(p + offset, n - offset, &cp, &valid);
        if (!(strip_bom && offset == 0 && cp == 0xfeff) &&
            !vx_text_emit(output, spaces && cp == 0x120 ? 32u : cp)) return 0;
        offset += width;
    }
    return 1;
}

static int vx_text_valid(const uint8_t* p, size_t n) {
    if (n && !p) return 0;
    for (size_t offset = 0; offset < n;) {
        uint32_t cp; int valid;
        offset += vx_text_character(p + offset, n - offset, &cp, &valid);
        if (!valid) return 0;
    }
    return 1;
}

static int vx_text_space(uint32_t cp) {
    return (cp >= 9 && cp <= 13) || cp == 32 || cp == 0xa0 || cp == 0x1680 ||
        (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 ||
        cp == 0x202f || cp == 0x205f || cp == 0x3000 || cp == 0xfeff;
}

static int vx_text_category(uint32_t cp) {
    size_t low = 0, high = sizeof(vx_unicode_categories) / sizeof(vx_unicode_categories[0]);
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        if (cp < vx_unicode_categories[mid].low) high = mid;
        else if (cp > vx_unicode_categories[mid].high) low = mid + 1;
        else return (int)vx_unicode_categories[mid].kind;
    }
    return 0;
}

/* HF regex whitespace follows Unicode White_Space, rather than the legacy
 * ECMAScript set above (notably U+0085 and U+FEFF differ). */
static int vx_hf_space(uint32_t cp) {
    return (cp >= 9 && cp <= 13) || cp == 32 || cp == 0x85 || cp == 0xa0 ||
        cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 ||
        cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

static int vx_hf_word(uint32_t cp) {
    size_t lo = 0, hi = sizeof(vx_unicode_word) / sizeof(vx_unicode_word[0]);
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < vx_unicode_word[mid].low) hi = mid;
        else if (cp > vx_unicode_word[mid].high) lo = mid + 1;
        else return 1;
    }
    return 0;
}

static uint32_t vx_hf_combining(uint32_t cp) {
    size_t lo = 0, hi = sizeof(vx_unicode_combining) / sizeof(vx_unicode_combining[0]);
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < vx_unicode_combining[mid].point) hi = mid;
        else if (cp > vx_unicode_combining[mid].point) lo = mid + 1;
        else return vx_unicode_combining[mid].kind;
    }
    return 0;
}

static int vx_hf_decompose(VxTextBuffer* out, uint32_t cp, unsigned depth) {
    if (depth > 32) return 0; /* Generated canonical mappings are acyclic. */
    if (cp >= 0xac00 && cp < 0xac00 + 11172) {
        uint32_t offset = cp - 0xac00;
        uint32_t parts[] = {0x1100 + offset / 588, 0x1161 + offset % 588 / 28,
            0x11a7 + offset % 28};
        return vx_text_append(out, parts, (parts[2] == 0x11a7 ? 2u : 3u) * sizeof(uint32_t));
    }
    size_t lo = 0, hi = sizeof(vx_unicode_decomposition) / sizeof(vx_unicode_decomposition[0]);
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < vx_unicode_decomposition[mid].point) hi = mid;
        else if (cp > vx_unicode_decomposition[mid].point) lo = mid + 1;
        else return vx_hf_decompose(out, vx_unicode_decomposition[mid].first, depth + 1) &&
            (!vx_unicode_decomposition[mid].second ||
             vx_hf_decompose(out, vx_unicode_decomposition[mid].second, depth + 1));
    }
    return vx_text_append(out, &cp, sizeof(cp));
}

static uint32_t vx_hf_compose(uint32_t first, uint32_t second) {
    if (first >= 0x1100 && first < 0x1100 + 19 && second >= 0x1161 && second < 0x1161 + 21)
        return 0xac00 + ((first - 0x1100) * 21 + second - 0x1161) * 28;
    if (first >= 0xac00 && first < 0xac00 + 11172 && (first - 0xac00) % 28 == 0 &&
        second > 0x11a7 && second < 0x11a7 + 28) return first + second - 0x11a7;
    size_t lo = 0, hi = sizeof(vx_unicode_composition) / sizeof(vx_unicode_composition[0]);
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        uint32_t a = vx_unicode_composition[mid].first, b = vx_unicode_composition[mid].second;
        if (first < a || (first == a && second < b)) hi = mid;
        else if (first > a || (first == a && second > b)) lo = mid + 1;
        else return vx_unicode_composition[mid].point;
    }
    return 0;
}

static int vx_hf_nfc(VxTextBuffer* out, const uint8_t* p, size_t n) {
    VxTextBuffer decomposed = {0};
    for (size_t at = 0; at < n;) {
        uint32_t cp; int valid;
        at += vx_text_character(p + at, n - at, &cp, &valid);
        if (!valid || !vx_hf_decompose(&decomposed, cp, 0)) { free(decomposed.data); return 0; }
    }
    uint32_t* points = (uint32_t*)decomposed.data;
    size_t count = decomposed.length / sizeof(uint32_t);
    /* Stable canonical ordering never moves a nonstarter across a starter. */
    for (size_t i = 1; i < count; ++i) {
        uint32_t cls = vx_hf_combining(points[i]), cp = points[i];
        if (!cls) continue;
        size_t j = i;
        while (j && vx_hf_combining(points[j - 1]) > cls) { points[j] = points[j - 1]; --j; }
        points[j] = cp;
    }
    size_t used = 0, starter = SIZE_MAX;
    uint32_t previous_class = 0;
    for (size_t i = 0; i < count; ++i) {
        uint32_t cp = points[i], cls = vx_hf_combining(cp);
        uint32_t composite = starter == SIZE_MAX ? 0 : vx_hf_compose(points[starter], cp);
        if (composite && (previous_class == 0 || previous_class < cls)) points[starter] = composite;
        else {
            if (!cls) starter = used;
            points[used++] = cp; previous_class = cls;
        }
    }
    for (size_t i = 0; i < used; ++i) if (!vx_text_emit(out, points[i])) { free(decomposed.data); return 0; }
    free(decomposed.data); return 1;
}

static int vx_hf_normalize(const VxTokenizer* t, VxTextBuffer* out,
                           const uint8_t* p, size_t n) {
    if (t->hf_normalizer == 1) return vx_hf_nfc(out, p, n);
    if (t->hf_normalizer == 2) {
        if (!n) return 1; /* HF Prepend leaves an empty split empty. */
        if (!vx_text_emit(out, 0x2581)) return 0;
        for (size_t at = 0; at < n;) {
            uint32_t cp; int valid;
            at += vx_text_character(p + at, n - at, &cp, &valid);
            if (!vx_text_emit(out, cp == 32 ? 0x2581 : cp)) return 0;
        }
        return 1;
    }
    return vx_text_append(out, p, n);
}

static uint32_t vx_text_hash(const uint8_t* p, size_t n) {
    uint32_t hash = UINT32_C(2166136261);
    for (size_t i = 0; i < n; ++i) hash = (hash ^ p[i]) * UINT32_C(16777619);
    return hash;
}

static int vx_token_equal(const VxTextBuffer* a, const uint8_t* p, size_t n) {
    return a->length == n && (!n || !memcmp(a->data, p, n));
}

static size_t vx_token_lookup(const VxTokenizer* t, const VxTokenMap* map,
                              const uint8_t* p, size_t n, int decoded) {
    size_t slot = vx_text_hash(p, n) & (map->capacity - 1);
    while (map->slots[slot]) {
        size_t index = map->slots[slot] - 1;
        const VxTextBuffer* key = decoded ? &t->tokens[index].text : &t->tokens[index].raw;
        if (vx_token_equal(key, p, n)) return index + 1;
        slot = (slot + 1) & (map->capacity - 1);
    }
    return 0;
}

static void vx_token_index(const VxTokenizer* t, VxTokenMap* map,
                            size_t index, int decoded) {
    const VxTextBuffer* key = decoded ? &t->tokens[index].text : &t->tokens[index].raw;
    size_t slot = vx_text_hash(key->data, key->length) & (map->capacity - 1);
    while (map->slots[slot]) {
        size_t other = map->slots[slot] - 1;
        const VxTextBuffer* value = decoded ? &t->tokens[other].text : &t->tokens[other].raw;
        if (vx_token_equal(value, key->data, key->length)) break;
        slot = (slot + 1) & (map->capacity - 1);
    }
    map->slots[slot] = index + 1; /* Decoded/raw lookup keeps the last token. */
}

static VxStatus vx_token_add(VxTokenizer* t, uint32_t id, const uint8_t* p,
                              size_t n, int json) {
    if ((n && !p) || t->count == UINT32_MAX) return VX_STATUS_INVALID_ARGUMENT;
    if (t->count == t->capacity) {
        size_t capacity = t->capacity ? t->capacity * 2 : 32;
        if (capacity < t->capacity || capacity > SIZE_MAX / sizeof(VxToken))
            return VX_STATUS_OUT_OF_MEMORY;
        void* next = realloc(t->tokens, capacity * sizeof(VxToken));
        if (!next) return VX_STATUS_OUT_OF_MEMORY;
        t->tokens = next; t->capacity = capacity;
    }
    VxToken* token = &t->tokens[t->count++];
    memset(token, 0, sizeof(*token)); token->id = id;
    if (!vx_text_append(&token->raw, p, n)) return VX_STATUS_OUT_OF_MEMORY;
    /* Decode-only ByteLevel never indexes or reads a normalized text copy. */
    if (t->profile == VX_TOKENIZER_PROFILE_LEGACY &&
        !(json ? vx_text_append(&token->text, p, n) : vx_text_utf8(&token->text, p, n, 1, 0)))
        return VX_STATUS_OUT_OF_MEMORY;
    return VX_STATUS_OK;
}

/* Canonical ByteLevel alphabet: visible bytes map to themselves. The remaining
 * bytes 0..32, 127..160 and 173 map, in that order, to U+0100..U+0143. */
static int vx_byte_level_inverse(uint32_t cp, uint8_t* byte) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) ||
        (cp >= 174 && cp <= 255)) *byte = (uint8_t)cp;
    else if (cp >= 256 && cp <= 288) *byte = (uint8_t)(cp - 256);
    else if (cp >= 289 && cp <= 322) *byte = (uint8_t)(cp - 289 + 127);
    else if (cp == 323) *byte = 173;
    else return 0;
    return 1;
}

static VxStatus vx_token_add_byte_level(VxTokenizer* t, const VxVocabularyToken* source) {
    if ((source->added != 0 && source->added != 1) ||
        (source->special != 0 && source->special != 1) ||
        (source->special && !source->added) ||
        !vx_text_valid(source->bytes, source->length)) return VX_STATUS_INVALID_ARGUMENT;
    VxTextBuffer raw = {0};
    VxStatus status = VX_STATUS_OK;
    if (source->added) {
        if (!vx_text_append(&raw, source->bytes, source->length)) status = VX_STATUS_OUT_OF_MEMORY;
    } else for (size_t at = 0; at < source->length;) {
        uint32_t cp; int valid; uint8_t byte;
        at += vx_text_character(source->bytes + at, source->length - at, &cp, &valid);
        if (!valid || !vx_byte_level_inverse(cp, &byte)) { status = VX_STATUS_INVALID_ARGUMENT; break; }
        if (!vx_text_append(&raw, &byte, 1)) { status = VX_STATUS_OUT_OF_MEMORY; break; }
    }
    if (status == VX_STATUS_OK) {
        /* Decoding owns raw bytes; this profile never creates encode indexes. */
        status = vx_token_add(t, source->id, raw.data, raw.length, 1);
        if (status == VX_STATUS_OK) {
            t->tokens[t->count - 1].added = source->added;
            t->tokens[t->count - 1].special = source->special;
        }
    }
    free(raw.data);
    return status;
}

static int vx_token_compare(const void* a, const void* b) {
    uint32_t left = ((const VxToken*)a)->id, right = ((const VxToken*)b)->id;
    return (left > right) - (left < right);
}

static const VxToken* vx_token_id(const VxTokenizer* t, uint32_t id) {
    size_t low = 0, high = t->count;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        if (id < t->tokens[mid].id) high = mid;
        else if (id > t->tokens[mid].id) low = mid + 1;
        else return &t->tokens[mid];
    }
    return NULL;
}

/* A length-aware JSON dictionary reader: cJSON object keys cannot represent
 * embedded NUL, which is a valid vocabulary byte. IDs are exact uint32s. */
typedef struct VxVocabJson { const uint8_t* p; size_t n, at; } VxVocabJson;
static void vx_vocab_json_space(VxVocabJson* j) {
    while (j->at < j->n && (j->p[j->at] == 32 || j->p[j->at] == 9 ||
           j->p[j->at] == 10 || j->p[j->at] == 13)) ++j->at;
}
static int vx_vocab_hex(VxVocabJson* j, uint32_t* cp) {
    if (j->n - j->at < 4) return 0;
    *cp = 0;
    for (int i = 0; i < 4; ++i) {
        uint8_t c = j->p[j->at++]; uint32_t v;
        if (c >= '0' && c <= '9') v = c - '0';
        else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
        else return 0;
        *cp = (*cp << 4) | v;
    }
    return 1;
}
static VxStatus vx_vocab_json_string(VxVocabJson* j, VxTextBuffer* out) {
    if (j->at == j->n || j->p[j->at++] != '"') return VX_STATUS_INVALID_ARGUMENT;
    while (j->at < j->n) {
        uint32_t cp = j->p[j->at++];
        if (cp == '"') return VX_STATUS_OK;
        if (cp < 32) return VX_STATUS_INVALID_ARGUMENT;
        if (cp == '\\') {
            if (j->at == j->n) return VX_STATUS_INVALID_ARGUMENT;
            cp = j->p[j->at++];
            if (cp == 'u') {
                if (!vx_vocab_hex(j, &cp)) return VX_STATUS_INVALID_ARGUMENT;
                if (cp >= 0xd800 && cp <= 0xdbff) {
                    uint32_t tail;
                    if (j->n - j->at < 6 || j->p[j->at] != '\\' || j->p[j->at + 1] != 'u')
                        return VX_STATUS_INVALID_ARGUMENT;
                    j->at += 2;
                    if (!vx_vocab_hex(j, &tail) || tail < 0xdc00 || tail > 0xdfff)
                        return VX_STATUS_INVALID_ARGUMENT;
                    cp = 0x10000 + ((cp - 0xd800) << 10) + tail - 0xdc00;
                } else if (cp >= 0xdc00 && cp <= 0xdfff) return VX_STATUS_INVALID_ARGUMENT;
            } else if (cp == 'b') cp = 8;
            else if (cp == 'f') cp = 12;
            else if (cp == 'n') cp = 10;
            else if (cp == 'r') cp = 13;
            else if (cp == 't') cp = 9;
            else if (cp != '"' && cp != '\\' && cp != '/') return VX_STATUS_INVALID_ARGUMENT;
        } else if (cp >= 0x80) {
            int valid;
            --j->at;
            j->at += vx_text_character(j->p + j->at, j->n - j->at, &cp, &valid);
            if (!valid) return VX_STATUS_INVALID_ARGUMENT;
        }
        if (!vx_text_emit(out, cp)) return VX_STATUS_OUT_OF_MEMORY;
    }
    return VX_STATUS_INVALID_ARGUMENT;
}

static VxStatus vx_vocab_json(VxTokenizer* t, const uint8_t* p, size_t n) {
    if (!p || !n) return VX_STATUS_INVALID_ARGUMENT;
    VxVocabJson j = {p, n, 0};
    if (n >= 3 && !memcmp(p, "\xef\xbb\xbf", 3)) j.at = 3;
    vx_vocab_json_space(&j);
    if (j.at == n || p[j.at++] != '{') return VX_STATUS_INVALID_ARGUMENT;
    vx_vocab_json_space(&j);
    if (j.at < n && p[j.at] != '}') for (;;) {
        VxTextBuffer key = {0};
        VxStatus status = vx_vocab_json_string(&j, &key);
        if (status != VX_STATUS_OK) { free(key.data); return status; }
        vx_vocab_json_space(&j);
        if (j.at == n || p[j.at++] != ':') { free(key.data); return VX_STATUS_INVALID_ARGUMENT; }
        vx_vocab_json_space(&j);
        uint64_t id = 0; size_t start = j.at;
        while (j.at < n && p[j.at] >= '0' && p[j.at] <= '9') {
            id = id * 10 + p[j.at++] - '0';
            if (id > UINT32_MAX) { free(key.data); return VX_STATUS_INVALID_ARGUMENT; }
        }
        if (j.at == start || (p[start] == '0' && j.at - start > 1)) {
            free(key.data); return VX_STATUS_INVALID_ARGUMENT;
        }
        status = vx_token_add(t, (uint32_t)id, key.data, key.length, 1);
        free(key.data);
        if (status != VX_STATUS_OK) return status;
        vx_vocab_json_space(&j);
        if (j.at == n || p[j.at] != ',') break;
        ++j.at; vx_vocab_json_space(&j);
    }
    if (j.at == n || p[j.at++] != '}') return VX_STATUS_INVALID_ARGUMENT;
    vx_vocab_json_space(&j);
    return j.at == n ? VX_STATUS_OK : VX_STATUS_INVALID_ARGUMENT;
}

static uint32_t vx_vocab_u32(const uint8_t* p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static VxStatus vx_vocab_binary(VxTokenizer* t, const uint8_t* p, size_t n) {
    if (!p || n < 4) return VX_STATUS_INVALID_ARGUMENT;
    uint32_t count = vx_vocab_u32(p); size_t offset = 4;
    if (count > INT32_MAX || count > (n - 4) / 4) return VX_STATUS_INVALID_ARGUMENT;
    for (uint32_t i = 0; i < count; ++i) {
        if (n - offset < 4) return VX_STATUS_INVALID_ARGUMENT;
        uint32_t length = vx_vocab_u32(p + offset); offset += 4;
        if (length > INT32_MAX || length > n - offset) return VX_STATUS_INVALID_ARGUMENT;
        VxStatus status = vx_token_add(t, i, p + offset, length, 0);
        if (status != VX_STATUS_OK) return status;
        offset += length;
    }
    return offset == n ? VX_STATUS_OK : VX_STATUS_INVALID_ARGUMENT;
}

static size_t vx_merge_hash(uint32_t left, uint32_t right) {
    uint64_t key = ((uint64_t)left << 32) | right;
    key ^= key >> 33; key *= UINT64_C(0xff51afd7ed558ccd); key ^= key >> 33;
    return (size_t)key;
}
static VxMerge* vx_merge_slot(const VxTokenizer* t, uint32_t left, uint32_t right) {
    size_t at = vx_merge_hash(left, right) & (t->merge_capacity - 1);
    while (t->merges[at].occupied &&
           (t->merges[at].left != left || t->merges[at].right != right))
        at = (at + 1) & (t->merge_capacity - 1);
    return &t->merges[at];
}
static int vx_merge_add(VxTokenizer* t, uint32_t left, uint32_t right, uint32_t merged) {
    if (!t->merge_capacity || (size_t)t->merge_count >= t->merge_capacity / 2) {
        size_t capacity = t->merge_capacity ? t->merge_capacity * 2 : 32;
        if (capacity < t->merge_capacity || capacity > SIZE_MAX / sizeof(VxMerge)) return 0;
        VxMerge* next = calloc(capacity, sizeof(VxMerge));
        if (!next) return 0;
        VxMerge* old = t->merges; size_t old_capacity = t->merge_capacity;
        t->merges = next; t->merge_capacity = capacity;
        for (size_t i = 0; i < old_capacity; ++i) if (old[i].occupied)
            *vx_merge_slot(t, old[i].left, old[i].right) = old[i];
        free(old);
    }
    VxMerge* slot = vx_merge_slot(t, left, right);
    if (!slot->occupied) {
        if (t->merge_count == UINT32_MAX) return 0;
        *slot = (VxMerge){left, right, merged, t->merge_count++, 1};
    }
    return 1;
}

/* Text lookup uses decoded keys first, then raw Latin-1 keys, matching the
 * vocabulary import semantics. Bytes lookup decodes with replacement first. */
static VxStatus vx_token_find_text(const VxTokenizer* t, const uint8_t* p,
                                    size_t n, size_t* index) {
    *index = vx_token_lookup(t, &t->text, p, n, 1);
    if (*index) return VX_STATUS_OK;
    VxTextBuffer raw = {0};
    for (size_t at = 0; at < n;) {
        uint32_t cp; int valid;
        at += vx_text_character(p + at, n - at, &cp, &valid);
        if (cp > 255) { free(raw.data); return VX_STATUS_OK; }
        uint8_t byte = (uint8_t)cp;
        if (!vx_text_append(&raw, &byte, 1)) { free(raw.data); return VX_STATUS_OUT_OF_MEMORY; }
    }
    *index = vx_token_lookup(t, &t->raw, raw.data, raw.length, 0);
    free(raw.data); return VX_STATUS_OK;
}
static VxStatus vx_token_find_bytes(const VxTokenizer* t, const uint8_t* p,
                                     size_t n, size_t* index) {
    VxTextBuffer text = {0};
    if (!vx_text_utf8(&text, p, n, 1, 0)) { free(text.data); return VX_STATUS_OUT_OF_MEMORY; }
    *index = vx_token_lookup(t, &t->text, text.data, text.length, 1);
    free(text.data);
    if (!*index) *index = vx_token_lookup(t, &t->raw, p, n, 0);
    return VX_STATUS_OK;
}
static size_t vx_merge_word(const uint8_t* p, size_t n, size_t at, size_t* start) {
    uint32_t cp; int valid;
    while (at < n) {
        size_t width = vx_text_character(p + at, n - at, &cp, &valid);
        if (!vx_text_space(cp)) break;
        at += width;
    }
    *start = at;
    while (at < n) {
        size_t width = vx_text_character(p + at, n - at, &cp, &valid);
        if (vx_text_space(cp)) break;
        at += width;
    }
    return at;
}
static VxStatus vx_merge_token(const VxTokenizer* t, const uint8_t* p, size_t n, size_t* index) {
    if (n >= 2 && p[0] == 0xc4 && p[1] == 0xa0) {
        VxTextBuffer normalized = {0}; uint8_t space = 32;
        if (!vx_text_append(&normalized, &space, 1) || !vx_text_append(&normalized, p + 2, n - 2)) {
            free(normalized.data); return VX_STATUS_OUT_OF_MEMORY;
        }
        VxStatus status = vx_token_find_text(t, normalized.data, normalized.length, index);
        free(normalized.data); return status;
    }
    return vx_token_find_text(t, p, n, index);
}
static VxStatus vx_token_load_merges(VxTokenizer* t, const uint8_t* bytes, size_t n) {
    VxTextBuffer text = {0}; VxStatus status = VX_STATUS_OK;
    if (!vx_text_utf8(&text, bytes, n, 1, 0)) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    for (size_t line = 0; line < text.length;) {
        size_t end = line;
        while (end < text.length && text.data[end] != 10) ++end;
        const uint8_t* p = text.data + line; size_t length = end - line;
        size_t first, second;
        size_t first_end = vx_merge_word(p, length, 0, &first);
        size_t second_end = vx_merge_word(p, length, first_end, &second);
        line = end + (end < text.length);
        if (first == first_end || p[first] == '#' || second == second_end) continue;
        size_t left, right;
        status = vx_merge_token(t, p + first, first_end - first, &left);
        if (status != VX_STATUS_OK) goto done;
        status = vx_merge_token(t, p + second, second_end - second, &right);
        if (status != VX_STATUS_OK) goto done;
        if (!left || !right) continue;
        VxTextBuffer merged = {0};
        const VxToken* a = &t->tokens[left - 1]; const VxToken* b = &t->tokens[right - 1];
        if (!vx_text_append(&merged, a->text.data, a->text.length) ||
            !vx_text_append(&merged, b->text.data, b->text.length)) {
            free(merged.data); status = VX_STATUS_OUT_OF_MEMORY; goto done;
        }
        size_t result = vx_token_lookup(t, &t->text, merged.data, merged.length, 1);
        free(merged.data);
        if (result && !vx_merge_add(t, a->id, b->id, t->tokens[result - 1].id)) {
            status = VX_STATUS_OUT_OF_MEMORY; goto done;
        }
    }
done:
    free(text.data); return status;
}

/* Complete tokenizer.json admission is deliberately explicit. Arbitrary regex,
 * training/dropout and other tokenizer families never degrade to legacy BPE. */
static int vx_hf_keys(const cJSON* object, const char* const* keys, size_t count) {
    if (!cJSON_IsObject(object)) return 0;
    for (const cJSON* item = object->child; item; item = item->next) {
        size_t i = 0;
        while (i < count && strcmp(item->string, keys[i])) ++i;
        if (i == count) return 0;
    }
    return 1;
}
#define VX_HF_KEYS(object, ...) vx_hf_keys(object, (const char* const[]){__VA_ARGS__}, \
    sizeof((const char* const[]){__VA_ARGS__}) / sizeof(const char*))
static const cJSON* vx_hf_get(const cJSON* object, const char* key) {
    return cJSON_GetObjectItemCaseSensitive(object, key);
}
static int vx_hf_is(const cJSON* value, const char* string) {
    return cJSON_IsString(value) && !strcmp(value->valuestring, string);
}
static int vx_hf_null(const cJSON* value) { return !value || cJSON_IsNull(value); }
static int vx_hf_bool(const cJSON* object, const char* key, int fallback, int* out) {
    const cJSON* value = vx_hf_get(object, key);
    if (!value) { *out = fallback; return 1; }
    if (!cJSON_IsBool(value)) return 0;
    *out = cJSON_IsTrue(value); return 1;
}
static int vx_hf_id(const cJSON* value, uint32_t* out) {
    if (!cJSON_IsNumber(value) || !(value->valuedouble >= 0) || value->valuedouble > UINT32_MAX)
        return 0;
    *out = (uint32_t)value->valuedouble;
    return value->valuedouble == (double)*out;
}

static int vx_hf_byte_config(const cJSON* value, int* prefix, int* regex) {
    int trim;
    return VX_HF_KEYS(value, "type", "add_prefix_space", "trim_offsets", "use_regex") &&
        vx_hf_is(vx_hf_get(value, "type"), "ByteLevel") &&
        vx_hf_bool(value, "add_prefix_space", 1, prefix) &&
        vx_hf_bool(value, "trim_offsets", 1, &trim) &&
        vx_hf_bool(value, "use_regex", 1, regex);
}

static int vx_hf_sp_definition(VxTokenizer* t, const cJSON* root) {
    const cJSON* normalizer = vx_hf_get(root, "normalizer");
    if (!VX_HF_KEYS(normalizer, "type", "normalizers") ||
        !vx_hf_is(vx_hf_get(normalizer, "type"), "Sequence")) return 0;
    const cJSON* sequence = vx_hf_get(normalizer, "normalizers");
    if (!cJSON_IsArray(sequence) || cJSON_GetArraySize(sequence) != 2) return 0;
    const cJSON* first = cJSON_GetArrayItem(sequence, 0), *second = cJSON_GetArrayItem(sequence, 1);
    const cJSON* pattern = vx_hf_get(second, "pattern");
    if (!VX_HF_KEYS(first, "type", "prepend") || !vx_hf_is(vx_hf_get(first, "type"), "Prepend") ||
        !vx_hf_is(vx_hf_get(first, "prepend"), "\xe2\x96\x81") ||
        !VX_HF_KEYS(second, "type", "pattern", "content") ||
        !vx_hf_is(vx_hf_get(second, "type"), "Replace") ||
        !VX_HF_KEYS(pattern, "String") || !vx_hf_is(vx_hf_get(pattern, "String"), " ") ||
        !vx_hf_is(vx_hf_get(second, "content"), "\xe2\x96\x81") ||
        !vx_hf_null(vx_hf_get(root, "pre_tokenizer"))) return 0;
    const cJSON* decoder = vx_hf_get(root, "decoder");
    if (!VX_HF_KEYS(decoder, "type", "decoders") ||
        !vx_hf_is(vx_hf_get(decoder, "type"), "Sequence")) return 0;
    sequence = vx_hf_get(decoder, "decoders");
    if (!cJSON_IsArray(sequence) || cJSON_GetArraySize(sequence) != 4) return 0;
    first = cJSON_GetArrayItem(sequence, 0); pattern = vx_hf_get(first, "pattern");
    second = cJSON_GetArrayItem(sequence, 3);
    uint32_t start, stop;
    if (!VX_HF_KEYS(first, "type", "pattern", "content") ||
        !vx_hf_is(vx_hf_get(first, "type"), "Replace") ||
        !VX_HF_KEYS(pattern, "String") || !vx_hf_is(vx_hf_get(pattern, "String"), "\xe2\x96\x81") ||
        !vx_hf_is(vx_hf_get(first, "content"), " ") ||
        !VX_HF_KEYS(cJSON_GetArrayItem(sequence, 1), "type") ||
        !vx_hf_is(vx_hf_get(cJSON_GetArrayItem(sequence, 1), "type"), "ByteFallback") ||
        !VX_HF_KEYS(cJSON_GetArrayItem(sequence, 2), "type") ||
        !vx_hf_is(vx_hf_get(cJSON_GetArrayItem(sequence, 2), "type"), "Fuse") ||
        !VX_HF_KEYS(second, "type", "content", "start", "stop") ||
        !vx_hf_is(vx_hf_get(second, "type"), "Strip") || !vx_hf_is(vx_hf_get(second, "content"), " ") ||
        !vx_hf_id(vx_hf_get(second, "start"), &start) || start != 1 ||
        !vx_hf_id(vx_hf_get(second, "stop"), &stop) || stop != 0) return 0;
    t->profile = VX_TOKENIZER_PROFILE_HF_SENTENCEPIECE_BPE; t->hf_normalizer = 2;
    return 1;
}

static int vx_hf_byte_definition(VxTokenizer* t, const cJSON* root) {
    const cJSON* normalizer = vx_hf_get(root, "normalizer");
    if (!vx_hf_null(normalizer)) {
        if (!VX_HF_KEYS(normalizer, "type") || !vx_hf_is(vx_hf_get(normalizer, "type"), "NFC")) return 0;
        t->hf_normalizer = 1;
    }
    const cJSON* pre = vx_hf_get(root, "pre_tokenizer");
    int regex = 0;
    if (vx_hf_byte_config(pre, &t->hf_prefix_space, &regex)) t->hf_pretokenizer = regex ? 1 : 0;
    else {
        if (!VX_HF_KEYS(pre, "type", "pretokenizers") || !vx_hf_is(vx_hf_get(pre, "type"), "Sequence")) return 0;
        const cJSON* seq = vx_hf_get(pre, "pretokenizers");
        if (!cJSON_IsArray(seq) || cJSON_GetArraySize(seq) != 2) return 0;
        const cJSON* first = cJSON_GetArrayItem(seq, 0);
        if (!vx_hf_byte_config(cJSON_GetArrayItem(seq, 1), &t->hf_prefix_space, &regex) || t->hf_prefix_space) return 0;
        int individual;
        if (VX_HF_KEYS(first, "type", "individual_digits") &&
            vx_hf_is(vx_hf_get(first, "type"), "Digits") &&
            vx_hf_bool(first, "individual_digits", 0, &individual) && individual && regex)
            t->hf_pretokenizer = 2;
        else {
            const cJSON* pattern = vx_hf_get(first, "pattern");
            int invert;
            if (!VX_HF_KEYS(first, "type", "pattern", "behavior", "invert") ||
                !vx_hf_is(vx_hf_get(first, "type"), "Split") ||
                !vx_hf_is(vx_hf_get(first, "behavior"), "Isolated") ||
                !vx_hf_bool(first, "invert", 0, &invert) || invert || regex ||
                !VX_HF_KEYS(pattern, "Regex")) return 0;
            static const char qwen[] = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
            static const char llama[] = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
            if (vx_hf_is(vx_hf_get(pattern, "Regex"), qwen)) t->hf_pretokenizer = 3;
            else if (vx_hf_is(vx_hf_get(pattern, "Regex"), llama)) t->hf_pretokenizer = 4;
            else return 0;
        }
    }
    int prefix;
    if (!vx_hf_byte_config(vx_hf_get(root, "decoder"), &prefix, &regex)) return 0;
    t->profile = VX_TOKENIZER_PROFILE_HF_BYTE_LEVEL_BPE; return 1;
}

static VxStatus vx_hf_token_add(VxTokenizer* t, uint32_t id, const char* encoded) {
    VxTextBuffer raw = {0}; size_t n = strlen(encoded);
    if (t->profile == VX_TOKENIZER_PROFILE_HF_BYTE_LEVEL_BPE) {
        for (size_t at = 0; at < n;) {
            uint32_t cp; int valid; uint8_t byte;
            at += vx_text_character((const uint8_t*)encoded + at, n - at, &cp, &valid);
            if (!valid || !vx_byte_level_inverse(cp, &byte)) { free(raw.data); return VX_STATUS_INVALID_ARGUMENT; }
            if (!vx_text_append(&raw, &byte, 1)) { free(raw.data); return VX_STATUS_OUT_OF_MEMORY; }
        }
    } else if (!vx_text_append(&raw, encoded, n)) return VX_STATUS_OUT_OF_MEMORY;
    VxStatus status = vx_token_add(t, id, raw.data, raw.length, 1);
    free(raw.data);
    if (status == VX_STATUS_OK && !vx_text_append(&t->tokens[t->count - 1].text, encoded, n))
        status = VX_STATUS_OUT_OF_MEMORY;
    return status;
}

static VxStatus vx_hf_load(VxTokenizer* t, const uint8_t* p, size_t n, cJSON** out) {
    *out = NULL;
    if (!p || !n || n > 64u * 1024u * 1024u || !vx_text_valid(p, n) ||
        vx_json_text_contains_decoded_nul(p, n)) return VX_STATUS_INVALID_ARGUMENT;
    /* Validate JSON escapes length-aware before cJSON may discard their length. */
    VxVocabJson scan = {p, n, 0};
    while (scan.at < n) {
        if (p[scan.at] == '"') {
            VxTextBuffer string = {0}; VxStatus s = vx_vocab_json_string(&scan, &string);
            free(string.data); if (s != VX_STATUS_OK) return s;
        } else ++scan.at;
    }
    const char* end = NULL;
    cJSON* root = cJSON_ParseWithLengthOpts((const char*)p, n, &end, 0);
    if (!root) return VX_STATUS_INVALID_ARGUMENT;
    size_t used = (size_t)(end - (const char*)p);
    while (used < n && (p[used] == 32 || p[used] == 9 || p[used] == 10 || p[used] == 13)) ++used;
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    if (used != n || !vx_json_object_keys_unique_recursive(root) ||
        !VX_HF_KEYS(root, "version", "truncation", "padding", "added_tokens", "normalizer", "pre_tokenizer", "post_processor", "decoder", "model") ||
        (vx_hf_get(root, "version") && !vx_hf_is(vx_hf_get(root, "version"), "1.0")) ||
        !vx_hf_null(vx_hf_get(root, "truncation")) || !vx_hf_null(vx_hf_get(root, "padding"))) goto fail;
    const cJSON* model = vx_hf_get(root, "model");
    if (!VX_HF_KEYS(model, "type", "dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix", "fuse_unk", "byte_fallback", "ignore_merges", "vocab", "merges") ||
        !vx_hf_is(vx_hf_get(model, "type"), "BPE") ||
        !vx_hf_null(vx_hf_get(model, "dropout"))) goto fail;
    const cJSON* value = vx_hf_get(model, "continuing_subword_prefix");
    if (!vx_hf_null(value) && !vx_hf_is(value, "")) goto fail;
    value = vx_hf_get(model, "end_of_word_suffix");
    if (!vx_hf_null(value) && !vx_hf_is(value, "")) goto fail;
    int fallback;
    if (!vx_hf_bool(model, "byte_fallback", 0, &fallback) ||
        !vx_hf_bool(model, "ignore_merges", 0, &t->hf_ignore_merges) ||
        !vx_hf_bool(model, "fuse_unk", 0, &t->hf_fuse_unk)) goto fail;
    if (fallback ? !vx_hf_sp_definition(t, root) : !vx_hf_byte_definition(t, root)) goto fail;
    const cJSON* vocab = vx_hf_get(model, "vocab");
    if (!cJSON_IsObject(vocab) || !cJSON_IsArray(vx_hf_get(model, "merges"))) goto fail;
    for (const cJSON* token = vocab->child; token; token = token->next) {
        uint32_t id;
        if (!vx_hf_id(token, &id) || !token->string || !*token->string) goto fail;
        status = vx_hf_token_add(t, id, token->string);
        if (status != VX_STATUS_OK) goto fail;
    }
    if (!t->count) { status = VX_STATUS_INVALID_ARGUMENT; goto fail; }
    qsort(t->tokens, t->count, sizeof(VxToken), vx_token_compare);
    for (size_t i = 1; i < t->count; ++i) if (t->tokens[i].id == t->tokens[i - 1].id) {
        status = VX_STATUS_INVALID_ARGUMENT; goto fail;
    }
    uint64_t next_added_id = t->count;
    const cJSON* added = vx_hf_get(root, "added_tokens");
    if (added && !cJSON_IsArray(added)) { status = VX_STATUS_INVALID_ARGUMENT; goto fail; }
    for (const cJSON* entry = added ? added->child : NULL; entry; entry = entry->next) {
        uint32_t id; int special, normalized, single, left, right;
        const cJSON* content = vx_hf_get(entry, "content");
        if (!VX_HF_KEYS(entry, "id", "content", "single_word", "lstrip", "rstrip", "normalized", "special") ||
            !vx_hf_id(vx_hf_get(entry, "id"), &id) || !cJSON_IsString(content) || !*content->valuestring ||
            !vx_hf_bool(entry, "special", 0, &special) ||
            !vx_hf_bool(entry, "normalized", !special, &normalized) ||
            !vx_hf_bool(entry, "single_word", 0, &single) ||
            !vx_hf_bool(entry, "lstrip", 0, &left) || !vx_hf_bool(entry, "rstrip", 0, &right)) {
            status = VX_STATUS_INVALID_ARGUMENT; goto fail;
        }
        /* A normalized SentencePiece added token would introduce a second
         * dummy-prefix policy; it is not part of this admitted Llama profile. */
        if (t->hf_normalizer == 2 && normalized) { status = VX_STATUS_INVALID_ARGUMENT; goto fail; }
        VxToken* token = NULL;
        for (size_t i = 0; i < t->count; ++i) if (t->tokens[i].id == id) { token = &t->tokens[i]; break; }
        size_t length = strlen(content->valuestring);
        if (token) {
            if (token->added || !vx_token_equal(&token->text, (const uint8_t*)content->valuestring, length)) {
                status = VX_STATUS_INVALID_ARGUMENT; goto fail;
            }
        } else {
            /* The HF added-vocabulary loader assigns new IDs consecutively
             * from the model's vocabulary size. Refuse contradictory serialized
             * IDs instead of accepting a tokenizer with a different ID map. */
            if (id != next_added_id) { status = VX_STATUS_INVALID_ARGUMENT; goto fail; }
            ++next_added_id;
            status = vx_token_add(t, id, (const uint8_t*)content->valuestring, length, 1);
            if (status != VX_STATUS_OK) goto fail;
            token = &t->tokens[t->count - 1];
            if (!vx_text_append(&token->text, content->valuestring, length)) { status = VX_STATUS_OUT_OF_MEMORY; goto fail; }
        }
        token->added = 1; token->special = special; token->single_word = single;
        token->normalized = normalized; token->lstrip = left; token->rstrip = right;
        if (!(normalized ? vx_hf_normalize(t, &token->match, (const uint8_t*)content->valuestring, length) :
            vx_text_append(&token->match, content->valuestring, length))) { status = VX_STATUS_OUT_OF_MEMORY; goto fail; }
        if (t->profile == VX_TOKENIZER_PROFILE_HF_BYTE_LEVEL_BPE) {
            /* HF's ByteLevel decoder tries its inverse on every token and
             * falls back to literal UTF-8 if any character is outside it. */
            VxTextBuffer raw = {0}; int inverse = 1;
            for (size_t at = 0; at < length;) {
                uint32_t cp; int valid; uint8_t byte;
                at += vx_text_character((const uint8_t*)content->valuestring + at, length - at, &cp, &valid);
                if (!vx_byte_level_inverse(cp, &byte)) { inverse = 0; break; }
                if (!vx_text_append(&raw, &byte, 1)) { free(raw.data); status = VX_STATUS_OUT_OF_MEMORY; goto fail; }
            }
            if (!inverse) {
                raw.length = 0;
                if (!vx_text_append(&raw, content->valuestring, length)) { free(raw.data); status = VX_STATUS_OUT_OF_MEMORY; goto fail; }
            }
            free(token->raw.data); token->raw = raw;
        }
    }
    *out = root; return VX_STATUS_OK;
fail:
    cJSON_Delete(root); return status == VX_STATUS_OK ? VX_STATUS_INVALID_ARGUMENT : status;
}

static int vx_hf_processor(VxTokenizer* t, const cJSON* processor, int depth) {
    if (vx_hf_null(processor)) return 1;
    if (depth > 4) return 0;
    int prefix, regex;
    if (vx_hf_byte_config(processor, &prefix, &regex)) return 1; /* Offsets only. */
    if (VX_HF_KEYS(processor, "type", "processors") && vx_hf_is(vx_hf_get(processor, "type"), "Sequence")) {
        const cJSON* list = vx_hf_get(processor, "processors");
        if (!cJSON_IsArray(list)) return 0;
        for (const cJSON* p = list->child; p; p = p->next) if (!vx_hf_processor(t, p, depth + 1)) return 0;
        return 1;
    }
    if (!VX_HF_KEYS(processor, "type", "single", "pair", "special_tokens") ||
        !vx_hf_is(vx_hf_get(processor, "type"), "TemplateProcessing") || t->prefix.length || t->suffix.length) return 0;
    const cJSON* single = vx_hf_get(processor, "single"), *specials = vx_hf_get(processor, "special_tokens");
    if (!cJSON_IsArray(single) || !cJSON_IsObject(specials)) return 0;
    int seen = 0;
    for (const cJSON* entry = single->child; entry; entry = entry->next) {
        const cJSON* sequence = vx_hf_get(entry, "Sequence");
        if (sequence) {
            uint32_t type;
            if (!VX_HF_KEYS(entry, "Sequence") || !VX_HF_KEYS(sequence, "id", "type_id") ||
                !vx_hf_is(vx_hf_get(sequence, "id"), "A") ||
                !vx_hf_id(vx_hf_get(sequence, "type_id"), &type) || type || seen) return 0;
            seen = 1; continue;
        }
        const cJSON* token = vx_hf_get(entry, "SpecialToken"); uint32_t type;
        if (!VX_HF_KEYS(entry, "SpecialToken") || !VX_HF_KEYS(token, "id", "type_id") ||
            !cJSON_IsString(vx_hf_get(token, "id")) || !vx_hf_id(vx_hf_get(token, "type_id"), &type) || type) return 0;
        const char* name = vx_hf_get(token, "id")->valuestring;
        const cJSON* definition = vx_hf_get(specials, name);
        if (!VX_HF_KEYS(definition, "id", "ids", "tokens") || !vx_hf_is(vx_hf_get(definition, "id"), name)) return 0;
        const cJSON* ids = vx_hf_get(definition, "ids"), *strings = vx_hf_get(definition, "tokens");
        if (!cJSON_IsArray(ids) || !cJSON_IsArray(strings) || cJSON_GetArraySize(ids) != cJSON_GetArraySize(strings) ||
            !cJSON_GetArraySize(ids)) return 0;
        const cJSON* text = strings->child;
        for (const cJSON* id = ids->child; id; id = id->next, text = text->next) {
            uint32_t value;
            if (!vx_hf_id(id, &value) || !cJSON_IsString(text)) return 0;
            const VxToken* source = vx_token_id(t, value);
            if (!source || !source->special || !vx_token_equal(&source->text,
                (const uint8_t*)text->valuestring, strlen(text->valuestring))) return 0;
            if (!vx_text_append(seen ? &t->suffix : &t->prefix, &value, sizeof(value))) return 0;
        }
    }
    return seen;
}

static VxStatus vx_hf_finish(VxTokenizer* t, const cJSON* root) {
    const cJSON* model = vx_hf_get(root, "model");
    const cJSON* unk = vx_hf_get(model, "unk_token");
    if (!vx_hf_null(unk)) {
        if (!cJSON_IsString(unk)) return VX_STATUS_INVALID_ARGUMENT;
        size_t index = vx_token_lookup(t, &t->text, (const uint8_t*)unk->valuestring, strlen(unk->valuestring), 1);
        if (!index) return VX_STATUS_INVALID_ARGUMENT;
        t->hf_has_unk = 1; t->hf_unk = t->tokens[index - 1].id;
    }
    if (t->profile == VX_TOKENIZER_PROFILE_HF_SENTENCEPIECE_BPE) {
        static const char hex[] = "0123456789ABCDEF";
        for (size_t i = 0; i < 256; ++i) {
            char name[] = "<0x00>"; name[3] = hex[i >> 4]; name[4] = hex[i & 15];
            size_t index = vx_token_lookup(t, &t->text, (const uint8_t*)name, 6, 1);
            if (!index || t->tokens[index - 1].added) return VX_STATUS_INVALID_ARGUMENT;
            t->hf_fallback[i] = t->tokens[index - 1].id;
        }
    }
    for (size_t i = 0; i < t->count; ++i) if (t->tokens[i].added) ++t->added_count;
    if (t->added_count) {
        if (t->added_count > SIZE_MAX / sizeof(size_t)) return VX_STATUS_OUT_OF_MEMORY;
        t->added = malloc(t->added_count * sizeof(size_t));
        if (!t->added) return VX_STATUS_OUT_OF_MEMORY;
        size_t at = 0;
        for (size_t i = 0; i < t->count; ++i) if (t->tokens[i].added) t->added[at++] = i;
        for (size_t i = 0; i < t->added_count; ++i) for (size_t j = 0; j < i; ++j)
            if (vx_token_equal(&t->tokens[t->added[i]].match, t->tokens[t->added[j]].match.data,
                t->tokens[t->added[j]].match.length)) return VX_STATUS_INVALID_ARGUMENT;
    }
    const cJSON* merges = vx_hf_get(model, "merges");
    for (const cJSON* pair = merges->child; pair; pair = pair->next) {
        const uint8_t* a; const uint8_t* b; size_t an, bn;
        if (cJSON_IsString(pair)) {
            const char* separator = strchr(pair->valuestring, ' ');
            if (!separator || separator == pair->valuestring || !separator[1] || strchr(separator + 1, ' '))
                return VX_STATUS_INVALID_ARGUMENT;
            a = (const uint8_t*)pair->valuestring; an = (size_t)(separator - pair->valuestring);
            b = (const uint8_t*)separator + 1; bn = strlen((const char*)b);
        } else if (cJSON_IsArray(pair) && cJSON_GetArraySize(pair) == 2 &&
            cJSON_IsString(pair->child) && cJSON_IsString(pair->child->next)) {
            a = (const uint8_t*)pair->child->valuestring; an = strlen((const char*)a);
            b = (const uint8_t*)pair->child->next->valuestring; bn = strlen((const char*)b);
        } else return VX_STATUS_INVALID_ARGUMENT;
        size_t left = vx_token_lookup(t, &t->text, a, an, 1), right = vx_token_lookup(t, &t->text, b, bn, 1);
        if (!left || !right || !an || !bn) return VX_STATUS_INVALID_ARGUMENT;
        VxTextBuffer joined = {0};
        if (!vx_text_append(&joined, a, an) || !vx_text_append(&joined, b, bn)) {
            free(joined.data); return VX_STATUS_OUT_OF_MEMORY;
        }
        size_t merged = vx_token_lookup(t, &t->text, joined.data, joined.length, 1); free(joined.data);
        if (!merged) return VX_STATUS_INVALID_ARGUMENT;
        if (t->merge_capacity && vx_merge_slot(t, t->tokens[left - 1].id, t->tokens[right - 1].id)->occupied)
            return VX_STATUS_INVALID_ARGUMENT;
        if (!vx_merge_add(t, t->tokens[left - 1].id, t->tokens[right - 1].id, t->tokens[merged - 1].id))
            return VX_STATUS_OUT_OF_MEMORY;
    }
    return vx_hf_processor(t, vx_hf_get(root, "post_processor"), 0) ? VX_STATUS_OK : VX_STATUS_INVALID_ARGUMENT;
}
#undef VX_HF_KEYS

void vx_tokenizer_retain(void* pointer) {
    if (pointer) atomic_fetch_add_explicit(&((VxTokenizer*)pointer)->references, 1, memory_order_relaxed);
}
void vx_tokenizer_release(void* pointer) {
    VxTokenizer* t = pointer;
    if (!t || atomic_fetch_sub_explicit(&t->references, 1, memory_order_acq_rel) != 1) return;
    for (size_t i = 0; i < t->count; ++i) {
        free(t->tokens[i].raw.data); free(t->tokens[i].text.data); free(t->tokens[i].match.data);
    }
    free(t->tokens); free(t->raw.slots); free(t->text.slots); free(t->merges);
    free(t->added); free(t->prefix.data); free(t->suffix.data); free(t);
}
uint32_t vx_tokenizer_size(const VxTokenizer* t) { return (uint32_t)t->count; }
uint32_t vx_tokenizer_merges(const VxTokenizer* t) { return t->merge_count; }
VxTokenizerProfile vx_tokenizer_profile(const VxTokenizer* t) { return t->profile; }
int vx_tokenizer_byte_alphabet_complete(const VxTokenizer* t) { return t->byte_alphabet_complete; }

VxStatus vx_tokenizer_create(const VxTokenizerSource* source, VxTokenizer** output) {
#if !defined(__wasm__)
    /* Keep the data notice in native artifacts even with section GC/LTO. */
    (void)vx_unicode_license[0];
#endif
    if (!output) return VX_STATUS_INVALID_ARGUMENT;
    *output = NULL;
    if (!source || (source->merge_bytes && !source->merges)) return VX_STATUS_INVALID_ARGUMENT;
    VxTokenizer* t = calloc(1, sizeof(*t));
    if (!t) return VX_STATUS_OUT_OF_MEMORY;
    atomic_init(&t->references, 1);
    VxStatus status = VX_STATUS_OK; cJSON* hf = NULL;
    if (source->format == 1 || source->format == 5) {
        if (source->format == 5) {
            t->profile = VX_TOKENIZER_PROFILE_BYTE_LEVEL_DECODE_ONLY;
            if (source->merge_bytes) status = VX_STATUS_INVALID_ARGUMENT;
        }
        if ((source->token_count && !source->tokens) || source->token_count > UINT32_MAX)
            status = VX_STATUS_INVALID_ARGUMENT;
        for (size_t i = 0; status == VX_STATUS_OK && i < source->token_count; ++i)
            status = source->format == 5 ? vx_token_add_byte_level(t, &source->tokens[i]) :
                vx_token_add(t, source->tokens[i].id, source->tokens[i].bytes, source->tokens[i].length, 0);
    } else if (source->format == 2) status = vx_vocab_json(t, source->json, source->json_bytes);
    else if (source->format == 3) status = vx_vocab_binary(t, source->binary, source->binary_bytes);
    else if (source->format == 6) {
        status = source->merge_bytes ? VX_STATUS_INVALID_ARGUMENT :
            vx_hf_load(t, source->json, source->json_bytes, &hf);
    }
    else status = VX_STATUS_INVALID_ARGUMENT;
    if (status != VX_STATUS_OK) goto fail;
    if (t->count) qsort(t->tokens, t->count, sizeof(VxToken), vx_token_compare);
    for (size_t i = 1; i < t->count; ++i) if (t->tokens[i].id == t->tokens[i - 1].id) {
        status = VX_STATUS_INVALID_ARGUMENT; goto fail;
    }
    if (t->profile == VX_TOKENIZER_PROFILE_BYTE_LEVEL_DECODE_ONLY ||
        t->profile == VX_TOKENIZER_PROFILE_HF_BYTE_LEVEL_BPE) {
        uint8_t present[256] = {0};
        for (size_t i = 0; i < t->count; ++i)
            if (!t->tokens[i].added && t->tokens[i].raw.length == 1)
                present[t->tokens[i].raw.data[0]] = 1;
        t->byte_alphabet_complete = 1;
        for (size_t i = 0; i < 256; ++i) if (!present[i]) t->byte_alphabet_complete = 0;
        if (t->profile == VX_TOKENIZER_PROFILE_BYTE_LEVEL_DECODE_ONLY) {
            *output = t; return VX_STATUS_OK;
        }
    }
    size_t capacity = 2;
    while (t->count >= capacity / 2) {
        if (capacity > SIZE_MAX / 2 / sizeof(size_t)) { status = VX_STATUS_OUT_OF_MEMORY; goto fail; }
        capacity *= 2;
    }
    t->raw.capacity = t->text.capacity = capacity;
    t->raw.slots = calloc(capacity, sizeof(size_t)); t->text.slots = calloc(capacity, sizeof(size_t));
    if (!t->raw.slots || !t->text.slots) { status = VX_STATUS_OUT_OF_MEMORY; goto fail; }
    for (size_t i = 0; i < t->count; ++i) {
        if (hf && vx_token_lookup(t, &t->text, t->tokens[i].text.data, t->tokens[i].text.length, 1)) {
            status = VX_STATUS_INVALID_ARGUMENT; goto fail;
        }
        vx_token_index(t, &t->raw, i, 0); vx_token_index(t, &t->text, i, 1);
    }
    for (size_t i = t->count; i-- > 0;) if (t->tokens[i].raw.length) {
        uint8_t first = t->tokens[i].raw.data[0];
        t->tokens[i].next = t->first[first]; t->first[first] = i + 1;
    }
    status = hf ? vx_hf_finish(t, hf) : vx_token_load_merges(t, source->merges, source->merge_bytes);
    if (status != VX_STATUS_OK) goto fail;
    cJSON_Delete(hf);
    *output = t; return VX_STATUS_OK;
fail:
    cJSON_Delete(hf);
    vx_tokenizer_release(t); return status;
}

/* Exact equivalent of the ordered Unicode pretokenization alternatives:
 * contractions, optional-space letters, numbers, punctuation, trailing space.
 * Only an ASCII space can prefix a non-whitespace chunk. */
static size_t vx_token_chunk(const uint8_t* p, size_t n) {
    static const char* const suffixes[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    if (p[0] == '\'') for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); ++i) {
        size_t width = strlen(suffixes[i]);
        if (width <= n && !memcmp(p, suffixes[i], width)) return width;
    }
    size_t at = p[0] == 32 && n > 1 ? 1 : 0;
    uint32_t cp; int valid;
    vx_text_character(p + at, n - at, &cp, &valid);
    if (!vx_text_space(cp)) {
        int category = vx_text_category(cp);
        do {
            at += vx_text_character(p + at, n - at, &cp, &valid);
            if (at == n) break;
            vx_text_character(p + at, n - at, &cp, &valid);
        } while (!vx_text_space(cp) && vx_text_category(cp) == category);
        return at;
    }
    at = 0; size_t last = 0;
    while (at < n) {
        size_t width = vx_text_character(p + at, n - at, &cp, &valid);
        if (!vx_text_space(cp)) break;
        last = at; at += width;
    }
    return at < n && last ? last : at;
}

static VxStatus vx_token_bpe(const VxTokenizer* t, const uint8_t* p, size_t n,
                              uint32_t limit, VxTextBuffer* result) {
    VxTextBuffer storage = {0}; VxStatus status = VX_STATUS_OK;
    for (size_t at = 0; at < n;) {
        uint32_t cp; int valid; size_t index;
        size_t width = vx_text_character(p + at, n - at, &cp, &valid);
        status = vx_token_find_bytes(t, p + at, width, &index);
        if (status != VX_STATUS_OK) goto done;
        if (index) {
            uint32_t id = t->tokens[index - 1].id;
            if (!vx_text_append(&storage, &id, sizeof(id))) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
        } else for (size_t i = 0; i < width; ++i) {
            status = vx_token_find_bytes(t, p + at + i, 1, &index);
            if (status != VX_STATUS_OK) goto done;
            if (index) {
                uint32_t id = t->tokens[index - 1].id;
                if (!vx_text_append(&storage, &id, sizeof(id))) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
            }
        }
        at += width;
    }
    uint32_t* symbols = (uint32_t*)storage.data; size_t count = storage.length / sizeof(uint32_t);
    while (t->merge_count && count > 1) {
        size_t best = SIZE_MAX; uint32_t rank = UINT32_MAX, merged = 0;
        for (size_t i = 0; i + 1 < count; ++i) {
            VxMerge* candidate = vx_merge_slot(t, symbols[i], symbols[i + 1]);
            if (candidate->occupied && candidate->rank < rank) {
                best = i; rank = candidate->rank; merged = candidate->merged;
            }
        }
        if (best == SIZE_MAX) break;
        symbols[best] = merged;
        memmove(symbols + best + 1, symbols + best + 2, (count - best - 2) * sizeof(uint32_t));
        --count;
    }
    if (count > limit) count = limit;
    if (!vx_text_append(result, symbols, count * sizeof(uint32_t))) status = VX_STATUS_OUT_OF_MEMORY;
done:
    free(storage.data); return status;
}

/* Ordered fixed regex profiles. These are the serialized GPT-2, Qwen and
 * Llama-3 expressions admitted above, implemented without locale or regex ABI. */
static size_t vx_hf_chunk(const uint8_t* p, size_t n, int profile) {
    if (!n || !profile) return n;
    static const char* const suffixes[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    if (p[0] == '\'') for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); ++i) {
        size_t width = strlen(suffixes[i]), j = 0;
        while (j < width && j < n) {
            uint8_t byte = p[j];
            if (profile >= 3 && byte >= 'A' && byte <= 'Z') byte += 'a' - 'A';
            if (byte != (uint8_t)suffixes[i][j]) break;
            ++j;
        }
        if (j == width) return width;
    }
    uint32_t cp; int valid;
    size_t first = vx_text_character(p, n, &cp, &valid), at = 0;
    int category = vx_text_category(cp);
    if (profile >= 3) {
        if (category != 1 && category != 2 && cp != 13 && cp != 10 && first < n) {
            uint32_t next;
            vx_text_character(p + first, n - first, &next, &valid);
            if (vx_text_category(next) == 1) at = first;
        }
        if (category == 1 || at) {
            do { at += vx_text_character(p + at, n - at, &cp, &valid);
                if (at == n) break;
                vx_text_character(p + at, n - at, &cp, &valid);
            } while (vx_text_category(cp) == 1);
            return at;
        }
        if (category == 2) {
            unsigned digits = 0, maximum = profile == 4 ? 3u : 1u;
            do { at += vx_text_character(p + at, n - at, &cp, &valid);
                if (++digits == maximum || at == n) break;
                vx_text_character(p + at, n - at, &cp, &valid);
            } while (vx_text_category(cp) == 2);
            return at;
        }
    } else {
        at = p[0] == 32 && n > 1 ? 1u : 0u;
        vx_text_character(p + at, n - at, &cp, &valid);
        category = vx_text_category(cp);
        if (category) {
            do { at += vx_text_character(p + at, n - at, &cp, &valid);
                if (at == n) break;
                vx_text_character(p + at, n - at, &cp, &valid);
            } while (vx_text_category(cp) == category);
            return at;
        }
    }
    at = p[0] == 32 && n > 1 ? 1u : 0u;
    vx_text_character(p + at, n - at, &cp, &valid);
    if (!vx_hf_space(cp) && !vx_text_category(cp)) {
        do { at += vx_text_character(p + at, n - at, &cp, &valid);
            if (at == n) break;
            vx_text_character(p + at, n - at, &cp, &valid);
        } while (!vx_hf_space(cp) && !vx_text_category(cp));
        if (profile >= 3) while (at < n && (p[at] == 13 || p[at] == 10)) ++at;
        return at;
    }
    at = 0; size_t previous = 0, newline = 0;
    while (at < n) {
        size_t width = vx_text_character(p + at, n - at, &cp, &valid);
        if (!vx_hf_space(cp)) break;
        previous = at; at += width;
        if (cp == 13 || cp == 10) newline = at;
    }
    if (profile >= 3 && newline) return newline;
    return at < n && previous ? previous : at;
}

static uint32_t vx_byte_level_forward(uint8_t byte) {
    if ((byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174) return byte;
    if (byte <= 32) return 256u + byte;
    if (byte <= 160) return 289u + byte - 127u;
    return 323;
}

static VxStatus vx_hf_bpe(const VxTokenizer* t, const uint8_t* p, size_t n,
                         VxTextBuffer* result) {
    VxTextBuffer storage = {0}; VxStatus status = VX_STATUS_OK;
    int byte_level = t->profile == VX_TOKENIZER_PROFILE_HF_BYTE_LEVEL_BPE;
    if (t->hf_ignore_merges) {
        VxTextBuffer encoded = {0};
        if (byte_level) {
            for (size_t i = 0; i < n; ++i) if (!vx_text_emit(&encoded, vx_byte_level_forward(p[i]))) {
                free(encoded.data); return VX_STATUS_OUT_OF_MEMORY;
            }
        } else if (!vx_text_append(&encoded, p, n)) return VX_STATUS_OUT_OF_MEMORY;
        size_t index = vx_token_lookup(t, &t->text, encoded.data, encoded.length, 1); free(encoded.data);
        if (index) return vx_text_append(result, &t->tokens[index - 1].id, sizeof(uint32_t)) ?
            VX_STATUS_OK : VX_STATUS_OUT_OF_MEMORY;
    }
    for (size_t at = 0; at < n;) {
        size_t index, width; VxTextBuffer character = {0};
        if (byte_level) {
            width = 1;
            if (!vx_text_emit(&character, vx_byte_level_forward(p[at]))) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
            index = vx_token_lookup(t, &t->text, character.data, character.length, 1); free(character.data);
        } else {
            uint32_t cp; int valid;
            width = vx_text_character(p + at, n - at, &cp, &valid);
            index = vx_token_lookup(t, &t->text, p + at, width, 1);
        }
        if (index) {
            uint32_t id = t->tokens[index - 1].id;
            if (!vx_text_append(&storage, &id, sizeof(id))) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
        } else if (!byte_level) {
            for (size_t i = 0; i < width; ++i) if (!vx_text_append(&storage, &t->hf_fallback[p[at + i]], sizeof(uint32_t))) {
                status = VX_STATUS_OUT_OF_MEMORY; goto done;
            }
        } else if (t->hf_has_unk) {
            if (!t->hf_fuse_unk || !storage.length ||
                ((uint32_t*)storage.data)[storage.length / sizeof(uint32_t) - 1] != t->hf_unk) {
                if (!vx_text_append(&storage, &t->hf_unk, sizeof(uint32_t))) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
            }
        } /* HF BPE with no unknown token omits an unregistered initial symbol. */
        at += width;
    }
    uint32_t* symbols = (uint32_t*)storage.data; size_t count = storage.length / sizeof(uint32_t);
    while (t->merge_count && count > 1) {
        size_t best = SIZE_MAX; uint32_t rank = UINT32_MAX, merged = 0;
        for (size_t i = 0; i + 1 < count; ++i) {
            VxMerge* candidate = vx_merge_slot(t, symbols[i], symbols[i + 1]);
            if (candidate->occupied && candidate->rank < rank) {
                best = i; rank = candidate->rank; merged = candidate->merged;
            }
        }
        if (best == SIZE_MAX) break;
        symbols[best] = merged;
        memmove(symbols + best + 1, symbols + best + 2, (count - best - 2) * sizeof(uint32_t)); --count;
    }
    if (!vx_text_append(result, symbols, count * sizeof(uint32_t))) status = VX_STATUS_OUT_OF_MEMORY;
done:
    free(storage.data); return status;
}

static VxStatus vx_hf_pretokenize(const VxTokenizer* t, const uint8_t* p, size_t n,
                                 VxTextBuffer* result) {
    if (!n) return VX_STATUS_OK;
    if (t->profile == VX_TOKENIZER_PROFILE_HF_SENTENCEPIECE_BPE) return vx_hf_bpe(t, p, n, result);
    VxTextBuffer prefixed = {0};
    if (t->hf_prefix_space && p[0] != 32) {
        uint8_t space = 32;
        if (!vx_text_append(&prefixed, &space, 1) || !vx_text_append(&prefixed, p, n)) {
            free(prefixed.data); return VX_STATUS_OUT_OF_MEMORY;
        }
        p = prefixed.data; n = prefixed.length;
    }
    VxStatus status = VX_STATUS_OK;
    for (size_t at = 0; at < n;) {
        size_t end = n;
        if (t->hf_pretokenizer == 2) {
            uint32_t cp; int valid;
            size_t width = vx_text_character(p + at, n - at, &cp, &valid);
            end = at + width;
            if (vx_text_category(cp) != 2) while (end < n) {
                width = vx_text_character(p + end, n - end, &cp, &valid);
                if (vx_text_category(cp) == 2) break;
                end += width;
            }
        }
        while (at < end) {
            size_t width = vx_hf_chunk(p + at, end - at, t->hf_pretokenizer == 2 ? 1 : t->hf_pretokenizer);
            if (!width) { status = VX_STATUS_INVALID_ARGUMENT; goto done; }
            status = vx_hf_bpe(t, p + at, width, result);
            if (status != VX_STATUS_OK) goto done;
            at += width;
        }
    }
done:
    free(prefixed.data); return status;
}

static size_t vx_hf_previous(const uint8_t* p, size_t at) {
    if (!at) return 0;
    --at;
    while (at && (p[at] & 0xc0) == 0x80) --at;
    return at;
}

static VxStatus vx_hf_extract(const VxTokenizer* t, const uint8_t* p, size_t n,
                             int normalized, VxTextBuffer* result) {
    size_t cursor = 0, search = 0;
    while (cursor < n || search < n) {
        size_t start = n, stop = n; const VxToken* found = NULL;
        for (size_t at = search; at < n;) {
            for (size_t i = 0; i < t->added_count; ++i) {
                const VxToken* token = &t->tokens[t->added[i]];
                if (token->normalized != normalized || token->match.length > n - at ||
                    memcmp(token->match.data, p + at, token->match.length)) continue;
                size_t end = at + token->match.length;
                /* HF selects its unfiltered trie match before single_word
                 * filtering. Colliding normalized patterns are refused when
                 * loading the definition. */
                if (!found || token->match.length > found->match.length) {
                    found = token; start = at; stop = end;
                }
            }
            if (found) break;
            uint32_t cp; int valid;
            at += vx_text_character(p + at, n - at, &cp, &valid);
        }
        if (found) {
            uint32_t cp; int valid;
            /* Trie iteration advances past the original match, even when
             * word filtering rejects it or rstrip later extends its span. */
            search = stop;
            if (found->single_word) {
                int rejected = 0;
                if (start) {
                    size_t previous = vx_hf_previous(p, start);
                    vx_text_character(p + previous, n - previous, &cp, &valid);
                    rejected = vx_hf_word(cp);
                }
                if (!rejected && stop < n) {
                    vx_text_character(p + stop, n - stop, &cp, &valid);
                    rejected = vx_hf_word(cp);
                }
                if (rejected) continue;
            }
            if (found->lstrip) while (start > cursor) {
                size_t previous = vx_hf_previous(p, start);
                vx_text_character(p + previous, n - previous, &cp, &valid);
                if (!vx_hf_space(cp)) break;
                start = previous;
            }
            if (found->lstrip && start < cursor) start = cursor;
            if (found->rstrip) while (stop < n) {
                size_t width = vx_text_character(p + stop, n - stop, &cp, &valid);
                if (!vx_hf_space(cp)) break;
                stop += width;
            }
            if (start > stop) return VX_STATUS_INVALID_ARGUMENT;
        }
        VxStatus status;
        size_t ordinary = start > cursor ? start - cursor : 0;
        if (normalized) status = vx_hf_pretokenize(t, p + cursor, ordinary, result);
        else {
            VxTextBuffer value = {0};
            if (!vx_hf_normalize(t, &value, p + cursor, ordinary)) return VX_STATUS_OUT_OF_MEMORY;
            status = vx_hf_extract(t, value.data, value.length, 1, result); free(value.data);
        }
        if (status != VX_STATUS_OK) return status;
        if (!found) break;
        if (!vx_text_append(result, &found->id, sizeof(uint32_t))) return VX_STATUS_OUT_OF_MEMORY;
        cursor = stop;
    }
    return VX_STATUS_OK;
}

VxStatus vx_tokenizer_encode_options(const VxTokenizer* t, const uint8_t* p, size_t n,
                              uint32_t limit, int mode, int add_special_tokens,
                              uint32_t** ids, size_t* count) {
    if (!ids || !count) return VX_STATUS_INVALID_ARGUMENT;
    *ids = NULL; *count = 0;
    if (!t || !vx_text_valid(p, n) || mode < 0 || mode > 2) return VX_STATUS_INVALID_ARGUMENT;
    if (t->profile == VX_TOKENIZER_PROFILE_LEGACY) return vx_tokenizer_encode(t, p, n, limit, mode, ids, count);
    if (t->profile == VX_TOKENIZER_PROFILE_BYTE_LEVEL_DECODE_ONLY || mode != 0)
        return VX_STATUS_INVALID_ARGUMENT;
    if (!limit) return VX_STATUS_OK;
    VxTextBuffer result = {0}; VxStatus status = VX_STATUS_OK;
    if (add_special_tokens && !vx_text_append(&result, t->prefix.data, t->prefix.length)) status = VX_STATUS_OUT_OF_MEMORY;
    if (status == VX_STATUS_OK) status = vx_hf_extract(t, p, n, 0, &result);
    if (status == VX_STATUS_OK && add_special_tokens && !vx_text_append(&result, t->suffix.data, t->suffix.length))
        status = VX_STATUS_OUT_OF_MEMORY;
    if (status != VX_STATUS_OK) { free(result.data); return status; }
    *ids = (uint32_t*)result.data;
    *count = result.length / sizeof(uint32_t);
    if (*count > limit) *count = limit;
    return VX_STATUS_OK;
}

VxStatus vx_tokenizer_encode(const VxTokenizer* t, const uint8_t* p, size_t n,
                              uint32_t limit, int mode, uint32_t** ids, size_t* count) {
    if (!ids || !count) return VX_STATUS_INVALID_ARGUMENT;
    *ids = NULL; *count = 0;
    if (!t || mode < 0 || mode > 2 || !vx_text_valid(p, n)) return VX_STATUS_INVALID_ARGUMENT;
    if (t->profile == VX_TOKENIZER_PROFILE_HF_BYTE_LEVEL_BPE ||
        t->profile == VX_TOKENIZER_PROFILE_HF_SENTENCEPIECE_BPE)
        return vx_tokenizer_encode_options(t, p, n, limit, mode, 1, ids, count);
    if (t->profile != VX_TOKENIZER_PROFILE_LEGACY) return VX_STATUS_INVALID_ARGUMENT;
    VxTextBuffer result = {0}; VxStatus status = VX_STATUS_OK;
    for (size_t at = 0; at < n && result.length / sizeof(uint32_t) < limit;) {
        if (mode == 1 || (!mode && !t->merge_count)) {
            const VxToken* best = NULL;
            for (size_t index = t->first[p[at]]; index; index = t->tokens[index - 1].next) {
                const VxToken* token = &t->tokens[index - 1];
                if ((!best || token->raw.length > best->raw.length) && token->raw.length <= n - at &&
                    !memcmp(token->raw.data, p + at, token->raw.length)) best = token;
            }
            if (!best) { ++at; continue; }
            if (!vx_text_append(&result, &best->id, sizeof(uint32_t))) { status = VX_STATUS_OUT_OF_MEMORY; break; }
            at += best->raw.length;
        } else {
            size_t width = mode == 2 ? n : vx_token_chunk(p + at, n - at);
            status = vx_token_bpe(t, p + at, width, limit - (uint32_t)(result.length / sizeof(uint32_t)), &result);
            if (status != VX_STATUS_OK) break;
            at += width;
        }
    }
    if (status != VX_STATUS_OK) { free(result.data); return status; }
    *ids = (uint32_t*)result.data; *count = result.length / sizeof(uint32_t);
    return VX_STATUS_OK;
}

VxStatus vx_tokenizer_decode_bytes(const VxTokenizer* t, const uint32_t* ids, size_t count,
                              int skip_special, uint8_t** value, size_t* bytes,
                              VxTokenByteMetadata** metadata, VxOperationCode* code) {
    if (!value || !bytes || !metadata || !code) return VX_STATUS_INVALID_ARGUMENT;
    *value = NULL; *bytes = 0; *metadata = NULL; *code = VX_CODE_INVALID_ARGUMENT;
    if (!t || (count && !ids)) return VX_STATUS_INVALID_ARGUMENT;
    if (t->profile != VX_TOKENIZER_PROFILE_BYTE_LEVEL_DECODE_ONLY &&
        t->profile != VX_TOKENIZER_PROFILE_HF_BYTE_LEVEL_BPE) {
        *code = VX_CODE_TOKENIZER_OPERATION_UNSUPPORTED; return VX_STATUS_INVALID_ARGUMENT;
    }
    /* No partial bytes, metadata or allocation before every ID is resolved. */
    for (size_t i = 0; i < count; ++i) if (!vx_token_id(t, ids[i])) {
        *code = VX_CODE_TOKEN_ID_UNKNOWN; return VX_STATUS_INVALID_ARGUMENT;
    }
    if (count > SIZE_MAX / sizeof(VxTokenByteMetadata)) return VX_STATUS_OUT_OF_MEMORY;
    VxTokenByteMetadata* info = count ? calloc(count, sizeof(*info)) : NULL;
    if (count && !info) return VX_STATUS_OUT_OF_MEMORY;
    VxTextBuffer output = {0};
    for (size_t i = 0; i < count; ++i) {
        const VxToken* token = vx_token_id(t, ids[i]);
        int skipped = skip_special && token->special;
        size_t length = skipped ? 0 : token->raw.length;
        info[i] = (VxTokenByteMetadata){ids[i], output.length, length,
            token->added, token->special, skipped};
        if (!vx_text_append(&output, token->raw.data, length)) {
            free(output.data); free(info); return VX_STATUS_OUT_OF_MEMORY;
        }
    }
    *value = output.data; *bytes = output.length; *metadata = info; *code = VX_CODE_NONE;
    return VX_STATUS_OK;
}

static int vx_hf_fallback_flush(VxTextBuffer* output, VxTextBuffer* run) {
    int valid = vx_text_valid(run->data, run->length);
    if (valid) {
        if (!vx_text_append(output, run->data, run->length)) return 0;
    } else for (size_t i = 0; i < run->length; ++i) if (!vx_text_emit(output, 0xfffd)) return 0;
    run->length = 0; return 1;
}

static int vx_hf_hex_digit(uint8_t value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

static VxStatus vx_hf_sp_decode(const VxTokenizer* t, const uint32_t* ids, size_t count,
                               int skip_special, uint8_t** text, size_t* bytes,
                               VxOperationCode* code) {
    for (size_t i = 0; i < count; ++i) if (!vx_token_id(t, ids[i])) {
        *code = VX_CODE_TOKEN_ID_UNKNOWN; return VX_STATUS_INVALID_ARGUMENT;
    }
    VxTextBuffer output = {0}, run = {0};
    for (size_t i = 0; i < count; ++i) {
        const VxToken* token = vx_token_id(t, ids[i]);
        if (skip_special && token->special) continue;
        const uint8_t* p = token->text.data; size_t n = token->text.length;
        int a = n == 6 ? vx_hf_hex_digit(p[3]) : -1, b = n == 6 ? vx_hf_hex_digit(p[4]) : -1;
        if (n == 6 && !memcmp(p, "<0x", 3) && p[5] == '>' && a >= 0 && b >= 0) {
            uint8_t byte = (uint8_t)(a * 16 + b);
            if (!vx_text_append(&run, &byte, 1)) goto oom;
            continue;
        }
        if (!vx_hf_fallback_flush(&output, &run)) goto oom;
        for (size_t at = 0; at < n;) {
            uint32_t cp; int valid;
            at += vx_text_character(p + at, n - at, &cp, &valid);
            if (!vx_text_emit(&output, cp == 0x2581 ? 32 : cp)) goto oom;
        }
    }
    if (!vx_hf_fallback_flush(&output, &run)) goto oom;
    free(run.data);
    if (output.length && output.data[0] == 32) {
        memmove(output.data, output.data + 1, --output.length);
    }
    *text = output.data; *bytes = output.length; *code = VX_CODE_NONE;
    return VX_STATUS_OK;
oom:
    free(run.data); free(output.data); return VX_STATUS_OUT_OF_MEMORY;
}

VxStatus vx_tokenizer_decode(const VxTokenizer* t, const uint32_t* ids, size_t count,
                              int skip_special, uint8_t** text, size_t* bytes,
                              VxOperationCode* code) {
    if (!text || !bytes || !code) return VX_STATUS_INVALID_ARGUMENT;
    *text = NULL; *bytes = 0; *code = VX_CODE_INVALID_ARGUMENT;
    if (!t || (count && !ids)) return VX_STATUS_INVALID_ARGUMENT;
    if (t->profile == VX_TOKENIZER_PROFILE_HF_SENTENCEPIECE_BPE)
        return vx_hf_sp_decode(t, ids, count, skip_special, text, bytes, code);
    if (t->profile == VX_TOKENIZER_PROFILE_LEGACY && skip_special) {
        *code = VX_CODE_TOKENIZER_OPERATION_UNSUPPORTED; return VX_STATUS_INVALID_ARGUMENT;
    }
    VxTextBuffer raw = {0}, output = {0};
    if (t->profile == VX_TOKENIZER_PROFILE_BYTE_LEVEL_DECODE_ONLY ||
        t->profile == VX_TOKENIZER_PROFILE_HF_BYTE_LEVEL_BPE) {
        VxTokenByteMetadata* metadata = NULL;
        VxStatus status = vx_tokenizer_decode_bytes(t, ids, count, skip_special,
            &raw.data, &raw.length, &metadata, code);
        free(metadata);
        if (status != VX_STATUS_OK) return status;
    } else for (size_t i = 0; i < count; ++i) {
        const VxToken* token = vx_token_id(t, ids[i]);
        if (token) {
            /* Preserve old complete-entry BOM stripping without decoding each
             * entry independently. Incomplete multi-byte pieces may now join. */
            size_t bom = token->raw.length >= 3 && token->raw.data[0] == 0xef &&
                token->raw.data[1] == 0xbb && token->raw.data[2] == 0xbf ? 3 : 0;
            if (!vx_text_append(&raw, token->raw.data ? token->raw.data + bom : NULL, token->raw.length - bom)) {
                free(raw.data); return VX_STATUS_OUT_OF_MEMORY;
            }
        }
    }
    int okay = vx_text_utf8(&output, raw.data, raw.length, 0,
        t->profile == VX_TOKENIZER_PROFILE_LEGACY);
    free(raw.data);
    if (!okay) { free(output.data); return VX_STATUS_OUT_OF_MEMORY; }
    *text = output.data; *bytes = output.length; *code = VX_CODE_NONE; return VX_STATUS_OK;
}
