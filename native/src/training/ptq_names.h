/* Deriving the names PTQ authoring introduces.
 *
 * Authoring adds tensors the source graph never had: packed weights, per-
 * channel scales, activation affines. Their names have to be stable, because
 * a template written by one implementation is consumed by another — the C
 * writer, the TypeScript host, a caller who saved it and came back — and a
 * name that depends on iteration order or a pointer value would make those
 * disagree about the same graph.
 *
 * So a name is a function of what it describes and nothing else:
 *
 *     __ptq__.<sha256("volvox-typed-ptq/v1\0<key>\0<role>")[:20]>.<role>
 *
 * The key is the source tensor for an activation affine, and the node id for
 * anything belonging to a node's weights. The role is the suffix, repeated
 * inside the digest so two roles of one key never collide.
 *
 * This is the same derivation `tools/exporter/typed_ptq.py` uses. That is not
 * a coincidence to be tidied away later: it is what lets the C and Python
 * implementations produce the same template for the same input, and it is
 * checked by test_ptq_names.
 *
 * A collision against a name the graph already declares is resolved by
 * appending `.1`, `.2`, ... — deterministic, and in practice never reached.
 */
#ifndef VOLVOXAI_PTQ_NAMES_H
#define VOLVOXAI_PTQ_NAMES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The domain separator inside every digest. Changing it renames every tensor
 * authoring produces, so it is a package format version. */
#define VX_PTQ_PLAN_FORMAT "volvox-typed-ptq/v1"

/* `__ptq__.` + 20 hex + `.` + the longest role + a disambiguating suffix. */
#define VX_PTQ_ALLOCATED_NAME_CAPACITY 64u

/* Names already spoken for: the source graph's tensors, plus everything
 * allocated so far. Borrowed, never copied — the caller owns the strings and
 * must keep them alive for the allocator's lifetime. */
typedef struct VxPtqNameSet {
    const char** names;
    size_t count;
    size_t capacity;
} VxPtqNameSet;

void vx_ptq_name_set_init(VxPtqNameSet* set);
void vx_ptq_name_set_free(VxPtqNameSet* set);

/* Returns 0, or -1 when out of memory. Adding a name twice is not an error;
 * the set is a membership question, not a log. */
int vx_ptq_name_set_add(VxPtqNameSet* set, const char* name);
int vx_ptq_name_set_contains(const VxPtqNameSet* set, const char* name);

/* Writes the derived name for (key, role) into `output` and records it in
 * `occupied`. Returns 0, or -1 on a bad argument or allocation failure.
 *
 * `occupied` may be NULL to derive a name without reserving it, which is what
 * a reader validating an existing template wants. */
int vx_ptq_allocate_name(VxPtqNameSet* occupied,
                         const char* key,
                         const char* role,
                         char output[VX_PTQ_ALLOCATED_NAME_CAPACITY]);

/* SHA-256, exposed because authoring also fingerprints graphs and weights.
 * `digest` receives 32 bytes; `hex` receives 64 characters plus a NUL. */
void vx_ptq_sha256(const void* data, size_t length, uint8_t digest[32]);
void vx_ptq_sha256_hex(const void* data, size_t length, char hex[65]);

typedef struct VxPtqSha256 {
    uint32_t state[8];
    uint64_t length;
    size_t pending;
    uint8_t buffer[64];
} VxPtqSha256;

void vx_ptq_sha256_init(VxPtqSha256* context);
void vx_ptq_sha256_update(VxPtqSha256* context, const void* data, size_t length);
void vx_ptq_sha256_final(VxPtqSha256* context, uint8_t digest[32]);

#ifdef __cplusplus
}
#endif

#endif /* VOLVOXAI_PTQ_NAMES_H */
