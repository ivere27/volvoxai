/* Token dictionaries, UTF-8, pretokenization and ranked BPE live in C.
 * Instances are immutable. Each encode/decode owns its temporary storage. */
#include "tokenizer.h"
#include "generated/tokenizer_unicode.h"
#include "generated/tokenizer_unicode_license.inc"
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
    if (!vx_text_append(&token->raw, p, n) ||
        !(json ? vx_text_append(&token->text, p, n) : vx_text_utf8(&token->text, p, n, 1, 0)))
        return VX_STATUS_OUT_OF_MEMORY;
    return VX_STATUS_OK;
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

void vx_tokenizer_retain(void* pointer) {
    if (pointer) atomic_fetch_add_explicit(&((VxTokenizer*)pointer)->references, 1, memory_order_relaxed);
}
void vx_tokenizer_release(void* pointer) {
    VxTokenizer* t = pointer;
    if (!t || atomic_fetch_sub_explicit(&t->references, 1, memory_order_acq_rel) != 1) return;
    for (size_t i = 0; i < t->count; ++i) { free(t->tokens[i].raw.data); free(t->tokens[i].text.data); }
    free(t->tokens); free(t->raw.slots); free(t->text.slots); free(t->merges); free(t);
}
uint32_t vx_tokenizer_size(const VxTokenizer* t) { return (uint32_t)t->count; }
uint32_t vx_tokenizer_merges(const VxTokenizer* t) { return t->merge_count; }

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
    VxStatus status = VX_STATUS_OK;
    if (source->format == 1) {
        if ((source->token_count && !source->tokens) || source->token_count > UINT32_MAX)
            status = VX_STATUS_INVALID_ARGUMENT;
        for (size_t i = 0; status == VX_STATUS_OK && i < source->token_count; ++i)
            status = vx_token_add(t, source->tokens[i].id, source->tokens[i].bytes, source->tokens[i].length, 0);
    } else if (source->format == 2) status = vx_vocab_json(t, source->json, source->json_bytes);
    else if (source->format == 3) status = vx_vocab_binary(t, source->binary, source->binary_bytes);
    else status = VX_STATUS_INVALID_ARGUMENT;
    if (status != VX_STATUS_OK) goto fail;
    if (t->count) qsort(t->tokens, t->count, sizeof(VxToken), vx_token_compare);
    for (size_t i = 1; i < t->count; ++i) if (t->tokens[i].id == t->tokens[i - 1].id) {
        status = VX_STATUS_INVALID_ARGUMENT; goto fail;
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
        vx_token_index(t, &t->raw, i, 0); vx_token_index(t, &t->text, i, 1);
    }
    for (size_t i = t->count; i-- > 0;) if (t->tokens[i].raw.length) {
        uint8_t first = t->tokens[i].raw.data[0];
        t->tokens[i].next = t->first[first]; t->first[first] = i + 1;
    }
    status = vx_token_load_merges(t, source->merges, source->merge_bytes);
    if (status != VX_STATUS_OK) goto fail;
    *output = t; return VX_STATUS_OK;
fail:
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

VxStatus vx_tokenizer_encode(const VxTokenizer* t, const uint8_t* p, size_t n,
                              uint32_t limit, int mode, uint32_t** ids, size_t* count) {
    if (!ids || !count) return VX_STATUS_INVALID_ARGUMENT;
    *ids = NULL; *count = 0;
    if (!t || mode < 0 || mode > 2 || !vx_text_valid(p, n)) return VX_STATUS_INVALID_ARGUMENT;
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

VxStatus vx_tokenizer_decode(const VxTokenizer* t, const uint32_t* ids, size_t count,
                              uint8_t** text, size_t* bytes) {
    if (!text || !bytes) return VX_STATUS_INVALID_ARGUMENT;
    *text = NULL; *bytes = 0;
    if (!t || (count && !ids)) return VX_STATUS_INVALID_ARGUMENT;
    VxTextBuffer output = {0};
    for (size_t i = 0; i < count; ++i) {
        const VxToken* token = vx_token_id(t, ids[i]);
        if (token && !vx_text_utf8(&output, token->raw.data, token->raw.length, 1, 1)) {
            free(output.data); return VX_STATUS_OUT_OF_MEMORY;
        }
    }
    *text = output.data; *bytes = output.length; return VX_STATUS_OK;
}
