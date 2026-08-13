#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

unsigned char* alloc_bytes(int bytes);
void reset_heap(void);
uintptr_t heap_mark(void);
int heap_rewind(uintptr_t mark);
uint32_t wasm_runtime_abi_version(void);

int main(void) {
    reset_heap();
    CHECK(wasm_runtime_abi_version() == 1u);

    unsigned char* first = alloc_bytes(1);
    uintptr_t reusable_mark = heap_mark();
    unsigned char* candidate = alloc_bytes(17);
    uintptr_t candidate_end = heap_mark();

    CHECK(first != NULL);
    CHECK(candidate == first + 16);
    CHECK(candidate_end == reusable_mark + 32);

    CHECK(heap_rewind(reusable_mark) == 1);
    CHECK(alloc_bytes(17) == candidate);

    /* A mark may only select an aligned address in the allocated prefix. */
    CHECK(heap_rewind(reusable_mark + 1) == 0);
    CHECK(heap_rewind(candidate_end + 16) == 0);
    return 0;
}
