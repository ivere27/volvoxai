#ifdef __wasm__
#include <wasm_simd128.h>
#else
#include "wasm_simd128_polyfill.h"
#endif
#include <stddef.h>
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
#define VX_WASM_PAGE_BYTES 65536u

/* The module owns its linear memory, so it also owns how far that memory
 * reaches. Growing here rather than from JavaScript keeps capacity a property
 * of the engine instead of a contract its caller has to honour before every
 * entry. */
static int arena_reserve(unsigned char* end) {
    size_t addressable = (size_t)__builtin_wasm_memory_size(0) * VX_WASM_PAGE_BYTES;
    size_t needed = (size_t)end;
    size_t pages;
    if (needed <= addressable) return 1;
    pages = (needed - addressable + VX_WASM_PAGE_BYTES - 1u) / VX_WASM_PAGE_BYTES;
    if (__builtin_wasm_memory_grow(0, pages) == (size_t)-1) return 0;
    return 1;
}
#else
#include <stdlib.h>
static unsigned char* heap_mem = NULL;
static unsigned char* heap = NULL;

static unsigned char* ensure_heap_base(void) {
    if (!heap_mem) {
        heap_mem = malloc(1024 * 1024 * 512); /* 512 MiB native kernel heap. */
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
#else
    /* A wasm32 module cannot report an allocation failure through this ABI --
     * address zero is ordinary linear memory, so returning it would silently
     * corrupt the low addresses instead of failing. Trap instead: the caller
     * sees a WebAssembly RuntimeError, which is a failure it can observe. */
    if (n < 0 || !arena_reserve(heap + n)) __builtin_trap();
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
/* A parent module without wasm_libc.c has no libc at all, so these two exist
 * to keep it self-contained. They are weak: a link that does include the
 * freestanding libc gets that one implementation instead of two. */
__attribute__((weak))
void* memset(void* dest, int c, unsigned long count) {
    unsigned char* p = (unsigned char*)dest;
    while (count--) { *p++ = (unsigned char)c; }
    return dest;
}

/* Clang may recognize fixed-width panel-copy loops as memcpy even under
 * -nostdlib.  Keep the freestanding module self-contained; volatile accesses
 * prevent this definition from being folded back into an unresolved import. */
__attribute__((weak))
void* memcpy(void* dest, const void* src, unsigned long count) {
    volatile unsigned char* out = (volatile unsigned char*)dest;
    const volatile unsigned char* in = (const volatile unsigned char*)src;
    while (count--) { *out++ = *in++; }
    return dest;
}
#endif
