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

#define VOLVOXAI_WASM_RUNTIME_ABI_VERSION 1u

WASM_EXPORT("wasm_runtime_abi_version")
uint32_t wasm_runtime_abi_version(void) {
    return VOLVOXAI_WASM_RUNTIME_ABI_VERSION;
}

#ifdef __wasm__
extern unsigned char __heap_base;
static unsigned char* heap = &__heap_base;
#define HEAP_BASE (&__heap_base)
#else
#include <stdlib.h>
static unsigned char* heap_mem = NULL;
static unsigned char* heap = NULL;

static unsigned char* ensure_heap_base(void) {
    if (!heap_mem) {
        heap_mem = malloc(1024 * 1024 * 512); /* 512 MiB test/native kernel heap. */
        heap = heap_mem;
    }
    return heap_mem;
}

#define HEAP_BASE (ensure_heap_base())
#endif

unsigned char* alloc_bytes(int n) {
    n = (n + 15) & ~15;
#ifndef __wasm__
    (void)ensure_heap_base();
#endif
    unsigned char* ptr = heap;
    heap += n;
    return ptr;
}

void reset_heap() {
    heap = HEAP_BASE;
}

/*
 * Context-local dynamic-shape execution needs to rebuild variant metadata and
 * scratch without resetting invariant tensors or packed weights.  A mark is an
 * opaque address in the current module instance; rewind accepts only an
 * aligned address in the already-allocated prefix, so a caller cannot advance
 * the allocator or escape its heap.  The JavaScript owner pre-grows linear
 * memory before rewinding, making subsequent allocations deterministic and
 * non-failing through the commit section.
 */
uintptr_t heap_mark(void) {
    (void)HEAP_BASE;
    return (uintptr_t)heap;
}

int heap_rewind(uintptr_t mark) {
    unsigned char* base = HEAP_BASE;
    uintptr_t base_address = (uintptr_t)base;
    uintptr_t current_address = (uintptr_t)heap;
    if (mark < base_address || mark > current_address ||
        ((mark - base_address) & 15u) != 0u) {
        return 0;
    }
    heap = (unsigned char*)mark;
    return 1;
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
