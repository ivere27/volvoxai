/* Model the raw kernel arena interleaved with the freestanding C heap. */
#include <stdint.h>
#include <stddef.h>
extern unsigned char __heap_base;
static uintptr_t cursor;
uintptr_t heap_mark(void) {
    if (!cursor) cursor = ((uintptr_t)&__heap_base + 15u) & ~(uintptr_t)15u;
    return cursor;
}
unsigned char* alloc_bytes(int bytes) {
    uintptr_t start = heap_mark(), size = ((uintptr_t)bytes + 15u) & ~(uintptr_t)15u;
    if (bytes < 0 || size > UINT32_MAX - start) return NULL;
    uintptr_t end = start + size;
    size_t available = __builtin_wasm_memory_size(0) * 65536u;
    if (end > available && __builtin_wasm_memory_grow(0,
            (end - available + 65535u) / 65536u) == (size_t)-1) return NULL;
    cursor = end;
    return (unsigned char*)start;
}
