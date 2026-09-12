#include "ptq_names.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- SHA-256 (FIPS 180-4) --------------------------------------------------
 *
 * Written out here rather than pulled in because the engine links no crypto
 * library and this is the only digest it needs. Authoring uses it for two
 * things — deriving tensor names and fingerprinting a graph — and both are
 * identity, not secrecy. */

static const uint32_t VX_SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

static uint32_t vx_rotr(uint32_t value, unsigned bits) {
    return (value >> bits) | (value << (32u - bits));
}

static void vx_sha256_block(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h;
    unsigned index;

    for (index = 0; index < 16u; index++) {
        w[index] = ((uint32_t)block[index * 4u] << 24) |
                   ((uint32_t)block[index * 4u + 1u] << 16) |
                   ((uint32_t)block[index * 4u + 2u] << 8) |
                   (uint32_t)block[index * 4u + 3u];
    }
    for (index = 16u; index < 64u; index++) {
        uint32_t s0 = vx_rotr(w[index - 15u], 7) ^ vx_rotr(w[index - 15u], 18) ^
                      (w[index - 15u] >> 3);
        uint32_t s1 = vx_rotr(w[index - 2u], 17) ^ vx_rotr(w[index - 2u], 19) ^
                      (w[index - 2u] >> 10);
        w[index] = w[index - 16u] + s0 + w[index - 7u] + s1;
    }

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    for (index = 0; index < 64u; index++) {
        uint32_t s1 = vx_rotr(e, 6) ^ vx_rotr(e, 11) ^ vx_rotr(e, 25);
        uint32_t choose = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + s1 + choose + VX_SHA256_K[index] + w[index];
        uint32_t s0 = vx_rotr(a, 2) ^ vx_rotr(a, 13) ^ vx_rotr(a, 22);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + majority;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

void vx_ptq_sha256_init(VxPtqSha256* context) {
    if (!context) return;
    context->state[0] = 0x6a09e667u;
    context->state[1] = 0xbb67ae85u;
    context->state[2] = 0x3c6ef372u;
    context->state[3] = 0xa54ff53au;
    context->state[4] = 0x510e527fu;
    context->state[5] = 0x9b05688cu;
    context->state[6] = 0x1f83d9abu;
    context->state[7] = 0x5be0cd19u;
    context->length = 0u;
    context->pending = 0u;
    memset(context->buffer, 0, sizeof(context->buffer));
}

void vx_ptq_sha256_update(VxPtqSha256* context, const void* data,
                          size_t length) {
    const uint8_t* cursor = (const uint8_t*)data;
    if (!context || (!data && length)) return;
    context->length += (uint64_t)length;
    while (length) {
        size_t room = 64u - context->pending;
        size_t take = length < room ? length : room;
        memcpy(context->buffer + context->pending, cursor, take);
        context->pending += take;
        cursor += take;
        length -= take;
        if (context->pending == 64u) {
            vx_sha256_block(context->state, context->buffer);
            context->pending = 0u;
        }
    }
}

void vx_ptq_sha256_final(VxPtqSha256* context, uint8_t digest[32]) {
    uint64_t bits;
    unsigned index;
    if (!context || !digest) return;
    bits = context->length * 8u;
    context->buffer[context->pending++] = 0x80u;
    if (context->pending > 56u) {
        memset(context->buffer + context->pending, 0, 64u - context->pending);
        vx_sha256_block(context->state, context->buffer);
        context->pending = 0u;
    }
    memset(context->buffer + context->pending, 0, 56u - context->pending);
    for (index = 0; index < 8u; index++) {
        context->buffer[56u + index] =
            (uint8_t)((bits >> (56u - 8u * index)) & 0xffu);
    }
    vx_sha256_block(context->state, context->buffer);
    for (index = 0; index < 8u; index++) {
        digest[index * 4u] = (uint8_t)(context->state[index] >> 24);
        digest[index * 4u + 1u] = (uint8_t)(context->state[index] >> 16);
        digest[index * 4u + 2u] = (uint8_t)(context->state[index] >> 8);
        digest[index * 4u + 3u] = (uint8_t)context->state[index];
    }
}

void vx_ptq_sha256(const void* data, size_t length, uint8_t digest[32]) {
    VxPtqSha256 context;
    vx_ptq_sha256_init(&context);
    vx_ptq_sha256_update(&context, data, length);
    vx_ptq_sha256_final(&context, digest);
}

void vx_ptq_sha256_hex(const void* data, size_t length, char hex[65]) {
    static const char DIGITS[] = "0123456789abcdef";
    uint8_t digest[32];
    unsigned index;
    if (!hex) return;
    vx_ptq_sha256(data, length, digest);
    for (index = 0; index < 32u; index++) {
        hex[index * 2u] = DIGITS[digest[index] >> 4];
        hex[index * 2u + 1u] = DIGITS[digest[index] & 0x0fu];
    }
    hex[64] = '\0';
}

/* --- occupied-name set ----------------------------------------------------
 *
 * Linear scan. A graph large enough for this to matter has thousands of
 * tensors, and authoring touches each name a handful of times; the hash table
 * this would become is not yet worth the code. */

void vx_ptq_name_set_init(VxPtqNameSet* set) {
    if (!set) return;
    set->names = NULL;
    set->count = 0u;
    set->capacity = 0u;
}

void vx_ptq_name_set_free(VxPtqNameSet* set) {
    if (!set) return;
    free(set->names);
    vx_ptq_name_set_init(set);
}

int vx_ptq_name_set_contains(const VxPtqNameSet* set, const char* name) {
    size_t index;
    if (!set || !name) return 0;
    for (index = 0; index < set->count; index++) {
        if (set->names[index] && strcmp(set->names[index], name) == 0) return 1;
    }
    return 0;
}

int vx_ptq_name_set_add(VxPtqNameSet* set, const char* name) {
    if (!set || !name) return -1;
    if (vx_ptq_name_set_contains(set, name)) return 0;
    if (set->count == set->capacity) {
        size_t capacity = set->capacity ? set->capacity * 2u : 32u;
        const char** grown =
            (const char**)realloc((void*)set->names, capacity * sizeof(*grown));
        if (!grown) return -1;
        set->names = grown;
        set->capacity = capacity;
    }
    set->names[set->count++] = name;
    return 0;
}

/* --- derivation ----------------------------------------------------------- */

int vx_ptq_allocate_name(VxPtqNameSet* occupied,
                         const char* key,
                         const char* role,
                         char output[VX_PTQ_ALLOCATED_NAME_CAPACITY]) {
    static const char DIGITS[] = "0123456789abcdef";
    VxPtqSha256 context;
    uint8_t digest[32];
    char hex[21];
    unsigned index;
    unsigned suffix;
    int written;

    if (!key || !key[0] || !role || !role[0] || !output) return -1;

    /* The NUL bytes are separators, not terminators: without them the pair
     * ("ab", "c") and ("a", "bc") would hash the same. */
    vx_ptq_sha256_init(&context);
    vx_ptq_sha256_update(&context, VX_PTQ_PLAN_FORMAT,
                         sizeof(VX_PTQ_PLAN_FORMAT) - 1u);
    vx_ptq_sha256_update(&context, "\0", 1u);
    vx_ptq_sha256_update(&context, key, strlen(key));
    vx_ptq_sha256_update(&context, "\0", 1u);
    vx_ptq_sha256_update(&context, role, strlen(role));
    vx_ptq_sha256_final(&context, digest);

    /* 20 hex characters — the first 10 bytes of the digest. */
    for (index = 0; index < 10u; index++) {
        hex[index * 2u] = DIGITS[digest[index] >> 4];
        hex[index * 2u + 1u] = DIGITS[digest[index] & 0x0fu];
    }
    hex[20] = '\0';

    written = snprintf(output, VX_PTQ_ALLOCATED_NAME_CAPACITY,
                       "__ptq__.%s.%s", hex, role);
    if (written <= 0 || (size_t)written >= VX_PTQ_ALLOCATED_NAME_CAPACITY) {
        return -1;
    }
    if (!occupied) return 0;

    for (suffix = 0u; vx_ptq_name_set_contains(occupied, output); suffix++) {
        if (suffix >= 1024u) return -1;
        written = snprintf(output, VX_PTQ_ALLOCATED_NAME_CAPACITY,
                           "__ptq__.%s.%s.%u", hex, role, suffix + 1u);
        if (written <= 0 ||
            (size_t)written >= VX_PTQ_ALLOCATED_NAME_CAPACITY) {
            return -1;
        }
    }
    /* The set borrows, so the caller's copy is what must live on. Adding
     * `output` here would dangle the moment the caller reused the buffer. */
    return 0;
}
