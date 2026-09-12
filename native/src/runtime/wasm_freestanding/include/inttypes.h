#ifndef VOLVOXAI_WASM_FREESTANDING_INTTYPES_H
#define VOLVOXAI_WASM_FREESTANDING_INTTYPES_H

/* The engine formats diagnostics with these width macros. wasm32 has 32-bit
 * long, so the 64-bit forms use the long long spellings. */

#include <stdint.h>

#define PRId8 "d"
#define PRIu8 "u"
#define PRId16 "d"
#define PRIu16 "u"
#define PRId32 "d"
#define PRIu32 "u"
#define PRIx32 "x"
#define PRId64 "lld"
#define PRIu64 "llu"
#define PRIx64 "llx"
#define PRIdPTR "d"
#define PRIuPTR "u"

#endif
