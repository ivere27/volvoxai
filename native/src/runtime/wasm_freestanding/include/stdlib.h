#ifndef VOLVOXAI_WASM_FREESTANDING_STDLIB_H
#define VOLVOXAI_WASM_FREESTANDING_STDLIB_H

#include <stddef.h>

#ifndef VOLVOXAI_WASM_FREESTANDING_SCOPE
#define VOLVOXAI_WASM_FREESTANDING_SCOPE extern
#endif

VOLVOXAI_WASM_FREESTANDING_SCOPE void* malloc(size_t size);
VOLVOXAI_WASM_FREESTANDING_SCOPE void* calloc(size_t count, size_t size);
VOLVOXAI_WASM_FREESTANDING_SCOPE void free(void* pointer);

#if defined(VOLVOXAI_WASM_FREESTANDING_EXTENDED_LIBC)
VOLVOXAI_WASM_FREESTANDING_SCOPE void* realloc(void* pointer, size_t size);

/* JSON numbers only. See wasm_libc.c for what this deliberately does not
 * parse. */
VOLVOXAI_WASM_FREESTANDING_SCOPE double strtod(const char* text, char** end);

/* Duplicate-key detection and canonical member ordering sort small arrays of
 * pointers. Not a general-purpose sort: see wasm_libc.c. */
VOLVOXAI_WASM_FREESTANDING_SCOPE void qsort(
    void* base, size_t count, size_t size,
    int (*compare)(const void*, const void*));
VOLVOXAI_WASM_FREESTANDING_SCOPE char* getenv(const char* name);
VOLVOXAI_WASM_FREESTANDING_SCOPE long strtol(const char* nptr, char** endptr, int base);
VOLVOXAI_WASM_FREESTANDING_SCOPE unsigned long long strtoull(const char* nptr, char** endptr, int base);
#endif

#endif
