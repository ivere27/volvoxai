#ifdef __wasm__
#include <wasm_simd128.h>
#else
#include "wasm_simd128_polyfill.h"
#endif
#include <stdint.h>

#ifdef __wasm__
#define WASM_EXPORT(name) __attribute__((export_name(name)))
#else
#define WASM_EXPORT(name)
#endif

#ifdef __wasm__
extern unsigned char __heap_base;
static unsigned char* heap = &__heap_base;
#else
#include <stdlib.h>
static unsigned char* heap_mem = NULL;
static unsigned char* heap = NULL;
#endif

unsigned char* alloc_bytes(int n) {
    n = (n + 15) & ~15;
#ifndef __wasm__
    if (!heap_mem) {
        heap_mem = malloc(1024 * 1024 * 512); // 512MB
        heap = heap_mem;
    }
#endif
    unsigned char* ptr = heap;
    heap += n;
    return ptr;
}

void reset_heap() {
#ifdef __wasm__
    heap = &__heap_base;
#else
    heap = heap_mem;
#endif
}

#ifdef __wasm__
void* memset(void* dest, int c, unsigned long count) {
    unsigned char* p = (unsigned char*)dest;
    while (count--) { *p++ = (unsigned char)c; }
    return dest;
}

/* Clang may recognize fixed-width panel-copy loops as memcpy even under
 * -nostdlib.  Keep the freestanding module self-contained; volatile accesses
 * prevent this definition from being folded back into an unresolved import. */
void* memcpy(void* dest, const void* src, unsigned long count) {
    volatile unsigned char* out = (volatile unsigned char*)dest;
    const volatile unsigned char* in = (const volatile unsigned char*)src;
    while (count--) { *out++ = *in++; }
    return dest;
}
#endif
