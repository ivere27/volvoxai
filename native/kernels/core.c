#include <wasm_simd128.h>
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
#endif
