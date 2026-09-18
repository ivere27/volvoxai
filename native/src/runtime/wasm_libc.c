/* Freestanding C library for the persistent WASM runtime.
 * The owner is single-threaded. Its coalescing allocator reuses storage across
 * RPCs; VFS entries pin immutable model snapshots while readers retain them.
 * JSON and SafeTensors use C-locale decimal parsing and formatting throughout.
 */

#include <float.h>
#include <limits.h>
#include <math.h>
#include <errno.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* wasm-ld places this after the static data; everything above it is ours. */
extern unsigned char __heap_base;

#define VX_WASM_PAGE_SIZE 65536u
#define VX_WASM_ALIGNMENT 16u

/* One block, free or in use. `size` is the payload; the header sits directly
 * before it, so free() finds it by subtracting. */
typedef struct VxWasmBlock {
    size_t size;
    struct VxWasmBlock* next;
    int in_use;
    struct VxWasmBlock* previous;
} VxWasmBlock;

/* Free blocks use their otherwise unused payload for the search list. Keeping
 * live model storage out of that list avoids scanning every model allocation
 * for each short-lived protobuf or shape-validation allocation. */
typedef struct VxWasmFreeLinks {
    VxWasmBlock* previous;
    VxWasmBlock* next;
} VxWasmFreeLinks;

_Static_assert(sizeof(VxWasmBlock) % VX_WASM_ALIGNMENT == 0,
               "allocator headers preserve payload alignment");
_Static_assert(sizeof(VxWasmFreeLinks) <= VX_WASM_ALIGNMENT,
               "free links fit the minimum payload");

static VxWasmBlock* vx_wasm_free_blocks = NULL;
static VxWasmBlock* vx_wasm_tail = NULL;
static unsigned char* vx_wasm_break = NULL;
static unsigned char* vx_wasm_limit = NULL;

static size_t vx_wasm_align(size_t value) {
    return (value + (VX_WASM_ALIGNMENT - 1u)) & ~(size_t)(VX_WASM_ALIGNMENT - 1u);
}

static VxWasmFreeLinks* vx_wasm_free_links(VxWasmBlock* block) {
    return (VxWasmFreeLinks*)((unsigned char*)block + sizeof(*block));
}

static void vx_wasm_free_insert(VxWasmBlock* block) {
    VxWasmFreeLinks* links = vx_wasm_free_links(block);
    links->previous = NULL;
    links->next = vx_wasm_free_blocks;
    if (links->next) vx_wasm_free_links(links->next)->previous = block;
    vx_wasm_free_blocks = block;
}

static void vx_wasm_free_remove(VxWasmBlock* block) {
    VxWasmFreeLinks* links = vx_wasm_free_links(block);
    if (links->previous) vx_wasm_free_links(links->previous)->next = links->next;
    else vx_wasm_free_blocks = links->next;
    if (links->next) vx_wasm_free_links(links->next)->previous = links->previous;
}

static void vx_wasm_split_block(VxWasmBlock* block, size_t size) {
    VxWasmBlock* remainder;
    size_t remainder_bytes;
    if (block->size < size) return;
    remainder_bytes = block->size - size;
    if (remainder_bytes < sizeof(VxWasmBlock) + VX_WASM_ALIGNMENT) return;

    remainder = (VxWasmBlock*)((unsigned char*)block +
                               sizeof(VxWasmBlock) + size);
    remainder->size = remainder_bytes - sizeof(VxWasmBlock);
    remainder->next = block->next;
    remainder->in_use = 0;
    remainder->previous = block;
    if (remainder->next) remainder->next->previous = remainder;
    block->size = size;
    block->next = remainder;
    if (vx_wasm_tail == block) vx_wasm_tail = remainder;
    vx_wasm_free_insert(remainder);
}

static void vx_wasm_merge_next(VxWasmBlock* block) {
    VxWasmBlock* next = block ? block->next : NULL;
    if (!next || next->in_use) return;
    if ((unsigned char*)block + sizeof(VxWasmBlock) + block->size != (unsigned char*)next) return;
    vx_wasm_free_remove(next);
    block->size += sizeof(VxWasmBlock) + next->size;
    block->next = next->next;
    if (block->next) block->next->previous = block;
    if (vx_wasm_tail == next) vx_wasm_tail = block;
}

#if defined(VOLVOXAI_PTQ_AUTHORING_NO_FILES) || !defined(__wasm__)
static unsigned char* vx_wasm_sbrk(size_t bytes) {
    unsigned char* result;
    if (!vx_wasm_break) {
        vx_wasm_break = &__heap_base;
        vx_wasm_limit = (unsigned char*)((size_t)__builtin_wasm_memory_size(0) *
                                         VX_WASM_PAGE_SIZE);
        if (vx_wasm_limit < vx_wasm_break) vx_wasm_limit = vx_wasm_break;
    }
    bytes = vx_wasm_align(bytes);
    if (bytes > (size_t)(SIZE_MAX - (size_t)vx_wasm_break)) return NULL;
    if (vx_wasm_break + bytes > vx_wasm_limit) {
        size_t needed = (size_t)(vx_wasm_break + bytes - vx_wasm_limit);
        size_t pages = (needed + VX_WASM_PAGE_SIZE - 1u) / VX_WASM_PAGE_SIZE;
        if (__builtin_wasm_memory_grow(0, pages) == (size_t)-1) return NULL;
        vx_wasm_limit += pages * VX_WASM_PAGE_SIZE;
    }
    result = vx_wasm_break;
    vx_wasm_break += bytes;
    return result;
}
#else
extern unsigned char* alloc_bytes(int n);
extern uintptr_t heap_mark(void);

static unsigned char* vx_wasm_sbrk(size_t bytes) {
    uintptr_t mark;
    bytes = vx_wasm_align(bytes);
    /* alloc_bytes takes a signed i32, adds its own alignment padding and
     * exposes its resulting mark through the same signed numeric ABI. */
    if (bytes > 0x7ffffff0u) return NULL;
    mark = heap_mark();
    if (mark > 0x7ffffff0u || bytes > 0x7ffffff0u - mark) return NULL;
    return alloc_bytes((int)bytes);
}
#endif

void* malloc(size_t size) {
    VxWasmBlock* block;
    VxWasmBlock* best = NULL;
    unsigned char* raw;

    if (!size) size = 1u;
    if (size > SIZE_MAX - (VX_WASM_ALIGNMENT - 1u)) return NULL;
    size = vx_wasm_align(size);
    if (size > SIZE_MAX - sizeof(VxWasmBlock)) return NULL;

    /* Best fit prevents a small input from consuming the large document block
     * retained by the previous call. Splitting and coalescing keep changing
     * graph shapes from turning that choice into monotonic memory growth. */
    for (block = vx_wasm_free_blocks; block; block = vx_wasm_free_links(block)->next) {
        if (block->size >= size &&
            (!best || block->size < best->size)) {
            best = block;
            if (block->size == size) break;
        }
    }
    if (best) {
        vx_wasm_free_remove(best);
        vx_wasm_split_block(best, size);
        best->in_use = 1;
        return (unsigned char*)best + sizeof(VxWasmBlock);
    }

    raw = vx_wasm_sbrk(sizeof(VxWasmBlock) + size);
    if (!raw) return NULL;
    block = (VxWasmBlock*)raw;
    block->size = size;
    block->in_use = 1;
    block->previous = vx_wasm_tail;
    block->next = NULL;
    if (vx_wasm_tail) {
        vx_wasm_tail->next = block;
    }
    vx_wasm_tail = block;
    return raw + sizeof(VxWasmBlock);
}

void free(void* pointer) {
    VxWasmBlock* block;
    VxWasmBlock* previous;
    if (!pointer) return;
    block = (VxWasmBlock*)((unsigned char*)pointer - sizeof(VxWasmBlock));
    if (!block->in_use) return;
    block->in_use = 0;
    previous = block->previous;
    vx_wasm_free_insert(block);
    vx_wasm_merge_next(block);
    if (previous && !previous->in_use) {
        vx_wasm_merge_next(previous);
    }
}

void* calloc(size_t count, size_t size) {
    size_t total;
    void* pointer;
    if (count && size > SIZE_MAX / count) return NULL;
    total = count * size;
    pointer = malloc(total ? total : 1u);
    if (pointer) memset(pointer, 0, total);
    return pointer;
}

void* realloc(void* pointer, size_t size) {
    VxWasmBlock* block;
    void* grown;
    size_t aligned;
    if (!pointer) return malloc(size);
    if (!size) { free(pointer); return NULL; }
    if (size > SIZE_MAX - (VX_WASM_ALIGNMENT - 1u)) return NULL;
    aligned = vx_wasm_align(size);
    block = (VxWasmBlock*)((unsigned char*)pointer - sizeof(VxWasmBlock));
    if (block->size >= aligned) return pointer;
    if (block->next && !block->next->in_use &&
        (unsigned char*)block + sizeof(*block) + block->size ==
            (unsigned char*)block->next &&
        block->size + sizeof(VxWasmBlock) + block->next->size >= aligned) {
        vx_wasm_merge_next(block);
        vx_wasm_split_block(block, aligned);
        return pointer;
    }
    grown = malloc(size);
    if (!grown) return NULL;
    memcpy(grown, pointer, block->size);
    free(pointer);
    return grown;
}

/* --- sort ------------------------------------------------------------------ */

static void vx_wasm_insertion_sort(
        unsigned char* items,
        size_t count,
        size_t size,
        int (*compare)(const void*, const void*)) {
    size_t sorted;
    for (sorted = 1u; sorted < count; sorted++) {
        size_t position = sorted;
        while (position > 0u &&
               compare(items + (position - 1u) * size,
                       items + position * size) > 0) {
            unsigned char* left = items + (position - 1u) * size;
            unsigned char* right = items + position * size;
            size_t byte;
            for (byte = 0u; byte < size; byte++) {
                unsigned char swap = left[byte];
                left[byte] = right[byte];
                right[byte] = swap;
            }
            position--;
        }
    }
}

static void vx_wasm_merge_runs(
        const unsigned char* source,
        unsigned char* destination,
        size_t left,
        size_t middle,
        size_t right,
        size_t size,
        int (*compare)(const void*, const void*)) {
    size_t lhs = left;
    size_t rhs = middle;
    size_t output = left;
    while (lhs < middle && rhs < right) {
        /* Equal elements come from the left run first. Besides making the
         * implementation deterministic, this preserves the stable ordering
         * used by canonical graph and weight projections. */
        if (compare(source + lhs * size, source + rhs * size) <= 0) {
            memcpy(destination + output * size, source + lhs * size, size);
            lhs++;
        } else {
            memcpy(destination + output * size, source + rhs * size, size);
            rhs++;
        }
        output++;
    }
    while (lhs < middle) {
        memcpy(destination + output * size, source + lhs * size, size);
        lhs++;
        output++;
    }
    while (rhs < right) {
        memcpy(destination + output * size, source + rhs * size, size);
        rhs++;
        output++;
    }
}

/* Stable bottom-up merge sort. Weight-name and compiled-weight collections
 * scale with model tensor count, so the normal allocation path must avoid
 * insertion sort's O(n^2) growth. Short inputs use a bounded insertion sort;
 * allocation failure retains the old stable algorithm because qsort cannot
 * report OOM without silently changing its contract. */
void qsort(void* base, size_t count, size_t size,
           int (*compare)(const void*, const void*)) {
    unsigned char* items = (unsigned char*)base;
    unsigned char* scratch;
    unsigned char* source;
    unsigned char* destination;
    size_t total;
    size_t width;
    if (!items || !compare || size == 0u || count < 2u) return;
    if (count <= 16u) {
        vx_wasm_insertion_sort(items, count, size, compare);
        return;
    }
    if (count > SIZE_MAX / size) return;
    total = count * size;
    scratch = (unsigned char*)malloc(total);
    if (!scratch) {
        vx_wasm_insertion_sort(items, count, size, compare);
        return;
    }
    source = items;
    destination = scratch;
    for (width = 1u; width < count;) {
        size_t left = 0u;
        while (left < count) {
            size_t remaining = count - left;
            size_t middle = left + (remaining < width ? remaining : width);
            size_t after_middle = count - middle;
            size_t right = middle +
                (after_middle < width ? after_middle : width);
            vx_wasm_merge_runs(source, destination, left, middle, right,
                               size, compare);
            left = right;
        }
        {
            unsigned char* swap = source;
            source = destination;
            destination = swap;
        }
        if (width > count / 2u) width = count;
        else width *= 2u;
    }
    if (source != items) memcpy(items, source, total);
    free(scratch);
}

/* --- string ---------------------------------------------------------------- */

int memcmp(const void* left, const void* right, size_t count) {
    const unsigned char* a = (const unsigned char*)left;
    const unsigned char* b = (const unsigned char*)right;
    size_t index;
    for (index = 0; index < count; index++) {
        if (a[index] != b[index]) return (int)a[index] - (int)b[index];
    }
    return 0;
}

void* memmove(void* destination, const void* source, size_t count) {
    unsigned char* to = (unsigned char*)destination;
    const unsigned char* from = (const unsigned char*)source;
    size_t index;
    if (to == from || !count) return destination;
    /* Backwards when the ranges overlap the wrong way, which memcpy is
     * allowed to get wrong and this is not. */
    if (to < from) {
        for (index = 0; index < count; index++) to[index] = from[index];
    } else {
        for (index = count; index > 0; index--) to[index - 1u] = from[index - 1u];
    }
    return destination;
}

char* strcpy(char* destination, const char* source) {
    char* cursor = destination;
    while ((*cursor++ = *source++) != '\0') { }
    return destination;
}

char* strncpy(char* destination, const char* source, size_t count) {
    size_t index = 0u;
    for (; index < count && source[index]; index++) {
        destination[index] = source[index];
    }
    for (; index < count; index++) destination[index] = '\0';
    return destination;
}

char* strchr(const char* text, int character) {
    for (; *text; text++) {
        if (*text == (char)character) return (char*)text;
    }
    return character == '\0' ? (char*)text : NULL;
}

char* strrchr(const char* text, int character) {
    const char* last = NULL;
    for (; *text; text++) {
        if (*text == (char)character) last = text;
    }
    if (character == '\0') return (char*)text;
    return (char*)last;
}

char* strstr(const char* haystack, const char* needle) {
    if (!*needle) return (char*)haystack;
    size_t needle_len = strlen(needle);
    for (; *haystack; haystack++) {
        if (*haystack == *needle && strncmp(haystack, needle, needle_len) == 0)
            return (char*)haystack;
    }
    return NULL;
}

/* JSON lexical scanning and correctly rounded binary64 decimal conversion. */
#include "wasm_decimal.c"

double strtod(const char* text, char** end) {
    const char* cursor = text;
    const char* start;
    int digits = 0;

    while (*cursor == ' ' || (*cursor >= '\t' && *cursor <= '\r')) cursor++;
    start = cursor;
    if (*cursor == '+' || *cursor == '-') cursor++;
    for (; *cursor >= '0' && *cursor <= '9'; cursor++) digits = 1;
    if (*cursor == '.') {
        cursor++;
        for (; *cursor >= '0' && *cursor <= '9'; cursor++) digits = 1;
    }
    if (!digits) {
        /* Nothing consumed. Reporting the original pointer is what tells the
         * caller no number was there. */
        if (end) *end = (char*)text;
        return 0.0;
    }
    if (*cursor == 'e' || *cursor == 'E') {
        const char* mark = cursor;
        int written = 0;
        cursor++;
        if (*cursor == '+' || *cursor == '-') cursor++;
        for (; *cursor >= '0' && *cursor <= '9'; cursor++) written = 1;
        if (!written) cursor = mark; /* A bare 'e' is not part of the number. */
    }
    if (end) *end = (char*)cursor;
    return vx_decimal_parse(start, (size_t)(cursor - start));
}

/* --- the two stdio entry points cJSON reaches for -------------------------- */

/* cJSON's uses are a version string and a number, both far below this. An
 * unbounded sprintf has no place in a build that cannot report a crash, so
 * the bound is real rather than nominal. */
#define VX_WASM_SPRINTF_CAPACITY 512u

int sprintf(char* destination, const char* format, ...) {
    va_list arguments;
    int written;
    va_start(arguments, format);
    written = vsnprintf(destination, VX_WASM_SPRINTF_CAPACITY, format,
                        arguments);
    va_end(arguments);
    return written;
}

/* One conversion: `%lg`, which cJSON uses to check that a printed number
 * reads back as the double it came from.
 *
 * Anything else returns zero conversions rather than guessing. A partial
 * scanf that silently accepted a format it did not implement would make that
 * round-trip check pass without checking anything, and the symptom would be
 * numbers that differ in their last digit between this build and the native
 * one — which is exactly the difference the check exists to catch. */
int sscanf(const char* text, const char* format, ...) {
    va_list arguments;
    double* target;
    char* end = NULL;
    double value;

    if (!text || !format) return 0;
    if (strcmp(format, "%lg") != 0 && strcmp(format, "%lf") != 0) return 0;

    value = strtod(text, &end);
    if (!end || end == text) return 0;
    va_start(arguments, format);
    target = va_arg(arguments, double*);
    va_end(arguments);
    if (!target) return 0;
    *target = value;
    return 1;
}


/* --- the rest of string.h -------------------------------------------------- */
/*
 * clang lowers some of these to memory.copy/memory.fill intrinsics and calls
 * the rest, so both the definitions and the builtins have to agree. Defining
 * them here rather than importing them from JavaScript keeps the crossing out
 * of an inner loop, and — for the formatting below — keeps C semantics.
 */

void* memcpy(void* destination, const void* source, size_t count) {
    unsigned char* to = (unsigned char*)destination;
    const unsigned char* from = (const unsigned char*)source;
    size_t index;
    for (index = 0; index < count; index++) to[index] = from[index];
    return destination;
}

void* memset(void* destination, int value, size_t count) {
    unsigned char* to = (unsigned char*)destination;
    size_t index;
    for (index = 0; index < count; index++) to[index] = (unsigned char)value;
    return destination;
}

size_t strlen(const char* value) {
    const char* cursor = value;
    while (*cursor) cursor++;
    return (size_t)(cursor - value);
}

int strcmp(const char* left, const char* right) {
    while (*left && *left == *right) { left++; right++; }
    return (int)(unsigned char)*left - (int)(unsigned char)*right;
}

int strncmp(const char* left, const char* right, size_t count) {
    size_t index;
    for (index = 0; index < count; index++) {
        unsigned char a = (unsigned char)left[index];
        unsigned char b = (unsigned char)right[index];
        if (a != b) return (int)a - (int)b;
        if (!a) break;
    }
    return 0;
}

/* --- formatting ------------------------------------------------------------
 *
 * Only the conversions this build reaches: %s, %c, %%, the signed and
 * unsigned integers, lowercase %x, and floating %g/%f/%e (also uppercase).
 * Precision and zero padding are supported. General flag handling and
 * positional arguments are outside the runtime's formatting contract.
 *
 * %g matters more than the rest put together. cJSON tries %1.15g, parses it
 * back, and uses %1.17g only when the first result lies outside its epsilon
 * comparison. The C decimal formatter expands the exact binary rational and
 * rounds its decimal digits with ties-to-even at the requested precision.
 */

typedef struct VxWasmSink {
    char* buffer;
    size_t capacity;
    size_t written;
    int failed;
} VxWasmSink;

static void vx_wasm_put(VxWasmSink* sink, char character) {
    if (sink->written + 1u < sink->capacity) {
        sink->buffer[sink->written] = character;
    }
    sink->written++;
}

static void vx_wasm_put_text(VxWasmSink* sink, const char* text) {
    while (*text) vx_wasm_put(sink, *text++);
}

static void vx_wasm_put_unsigned_base(VxWasmSink* sink,
                                      unsigned long long value,
                                      unsigned base,
                                      int minimum_width,
                                      char padding) {
    static const char alphabet[] = "0123456789abcdef";
    char digits[64];
    int count = 0;
    do {
        digits[count++] = alphabet[value % base];
        value /= base;
    } while (value && count < (int)sizeof(digits));
    while (count < minimum_width) {
        vx_wasm_put(sink, padding);
        minimum_width--;
    }
    while (count) vx_wasm_put(sink, digits[--count]);
}

static void vx_wasm_put_unsigned(VxWasmSink* sink, unsigned long long value) {
    vx_wasm_put_unsigned_base(sink, value, 10u, 0, ' ');
}

static void vx_wasm_put_signed(VxWasmSink* sink, long long value) {
    if (value < 0) {
        vx_wasm_put(sink, '-');
        vx_wasm_put_unsigned(sink, (unsigned long long)(-(value + 1)) + 1ull);
    } else {
        vx_wasm_put_unsigned(sink, (unsigned long long)value);
    }
}

int vsnprintf(char* destination, size_t capacity, const char* format,
              va_list arguments) {
    VxWasmSink sink;
    sink.buffer = destination;
    sink.capacity = capacity;
    sink.written = 0u;
    sink.failed = 0;

    for (; *format; format++) {
        int precision = -1;
        int is_long = 0;
        int is_size = 0;
        int minimum_width = 0;
        int zero_pad = 0;
        if (*format != '%') { vx_wasm_put(&sink, *format); continue; }
        format++;
        if (*format == '%') { vx_wasm_put(&sink, '%'); continue; }
        if (*format == '0') {
            zero_pad = 1;
            format++;
        }
        while (*format >= '0' && *format <= '9') {
            if (minimum_width < 1024) {
                minimum_width = minimum_width * 10 + (*format - '0');
            }
            format++;
        }
        if (*format == '.') {
            format++;
            if (*format == '*') {
                precision = va_arg(arguments, int);
                format++;
            } else {
                precision = 0;
                while (*format >= '0' && *format <= '9') {
                    precision = precision * 10 + (*format++ - '0');
                }
            }
        }
        while (*format == 'l' || *format == 'z') {
            if (*format == 'l') is_long++;
            else is_size = 1;
            format++;
        }
        switch (*format) {
        case 's': {
            const char* text = va_arg(arguments, const char*);
            if (!text) text = "(null)";
            if (precision < 0) vx_wasm_put_text(&sink, text);
            else for (int i = 0; text[i] && i < precision; i++) vx_wasm_put(&sink, text[i]);
            break;
        }
        case 'c':
            vx_wasm_put(&sink, (char)va_arg(arguments, int));
            break;
        case 'd':
        case 'i':
            if (is_size) {
                vx_wasm_put_signed(&sink, va_arg(arguments, ptrdiff_t));
            } else if (is_long >= 2) {
                vx_wasm_put_signed(&sink, va_arg(arguments, long long));
            } else if (is_long) {
                vx_wasm_put_signed(&sink, va_arg(arguments, long));
            } else {
                vx_wasm_put_signed(&sink, va_arg(arguments, int));
            }
            break;
        case 'u':
            if (is_size) {
                vx_wasm_put_unsigned_base(&sink, va_arg(arguments, size_t), 10u,
                    minimum_width, zero_pad ? '0' : ' ');
            } else if (is_long >= 2) {
                vx_wasm_put_unsigned_base(
                    &sink, va_arg(arguments, unsigned long long), 10u,
                    minimum_width, zero_pad ? '0' : ' ');
            } else if (is_long) {
                vx_wasm_put_unsigned_base(&sink, va_arg(arguments, unsigned long), 10u,
                    minimum_width, zero_pad ? '0' : ' ');
            } else {
                vx_wasm_put_unsigned_base(
                    &sink, va_arg(arguments, unsigned int), 10u,
                    minimum_width, zero_pad ? '0' : ' ');
            }
            break;
        case 'x':
            if (is_size) {
                vx_wasm_put_unsigned_base(&sink, va_arg(arguments, size_t), 16u,
                    minimum_width, zero_pad ? '0' : ' ');
            } else if (is_long >= 2) {
                vx_wasm_put_unsigned_base(
                    &sink, va_arg(arguments, unsigned long long), 16u,
                    minimum_width, zero_pad ? '0' : ' ');
            } else if (is_long) {
                vx_wasm_put_unsigned_base(&sink, va_arg(arguments, unsigned long), 16u,
                    minimum_width, zero_pad ? '0' : ' ');
            } else {
                vx_wasm_put_unsigned_base(
                    &sink, va_arg(arguments, unsigned int), 16u,
                    minimum_width, zero_pad ? '0' : ' ');
            }
            break;
        case 'g': case 'G': case 'f': case 'F': case 'e': case 'E': {
            size_t available = sink.written < sink.capacity ? sink.capacity - sink.written : 0;
            int written = vx_decimal_format(available ? sink.buffer + sink.written : NULL,
                available, va_arg(arguments, double), precision, minimum_width, zero_pad, *format);
            if (written < 0 || sink.written > (size_t)INT_MAX - (size_t)written) sink.failed = 1;
            else sink.written += (size_t)written;
            break;
        }
        default:
            /* Unreached by this build. Emitting the specifier verbatim keeps
             * a mistake visible in the output instead of consuming an
             * argument that was never there. */
            vx_wasm_put(&sink, '%');
            if (*format) vx_wasm_put(&sink, *format);
            break;
        }
        if (!*format) break;
    }
    if (sink.failed) {
        if (capacity) sink.buffer[0] = '\0';
        return -1;
    }
    if (capacity) {
        sink.buffer[sink.written < capacity ? sink.written : capacity - 1u] = '\0';
    }
    return (int)sink.written;
}

int snprintf(char* destination, size_t capacity, const char* format, ...) {
    va_list arguments;
    int written;
    va_start(arguments, format);
    written = vsnprintf(destination, capacity, format, arguments);
    va_end(arguments);
    return written;
}

#if defined(__wasm__)
typedef struct VxVFile {
    char path[256];
    unsigned char* data;
    size_t size;
    size_t capacity;
    size_t map_count;
    int is_dynamic;
    int unlink_pending;
    int in_use;
} VxVFile;

/* File objects have stable addresses while the index grows. This admits
 * multiple retained model revisions and trainer baselines without a second
 * fixed file-count budget; allocation failures remain ordinary C failures. */
static VxVFile** g_vx_vfiles;
static int g_vx_vfile_count;
#define VX_MAX_VFILES g_vx_vfile_count

static int vx_wasm_grow_file_slots(void) {
    if (g_vx_vfile_count == INT32_MAX ||
        (size_t)g_vx_vfile_count >= SIZE_MAX / sizeof(*g_vx_vfiles) - 1u) return -1;
    VxVFile* file = (VxVFile*)calloc(1, sizeof(*file));
    if (!file) return -1;
    VxVFile** next = (VxVFile**)realloc(g_vx_vfiles,
        ((size_t)g_vx_vfile_count + 1u) * sizeof(*next));
    if (!next) { free(file); return -1; }
    g_vx_vfiles = next;
    int index = g_vx_vfile_count++;
    next[index] = file;
    return index;
}

struct FILE {
    VxVFile* file;
    size_t position;
    int mode;
};

#define VX_MAX_STREAMS 32
static FILE g_vx_streams[VX_MAX_STREAMS];

/* Reserve an exact backing-store size for a writable VFS stream. Model
 * snapshots know their final length before copying, so this avoids the
 * geometric realloc peak that would otherwise hold both the old and new
 * 30--40 MiB buffers at once. The helper stays private to the amalgamated
 * browser runtime; native stdio keeps using the host filesystem. */
static int vx_wasm_file_reserve(FILE* stream, size_t capacity) {
    VxVFile* vfile;
    unsigned char* data;
    if (!stream || !stream->file || !stream->mode) return -1;
    vfile = stream->file;
    if (vfile->map_count) return -1;
    if (capacity <= vfile->capacity) return 0;
    data = (unsigned char*)malloc(capacity);
    if (!data) return -1;
    if (vfile->data && vfile->size != 0u) {
        memcpy(data, vfile->data, vfile->size);
    }
    if (vfile->is_dynamic && vfile->data) free(vfile->data);
    vfile->data = data;
    vfile->capacity = capacity;
    vfile->is_dynamic = 1;
    return 0;
}

/* One synchronous browser host owns one upload transaction at a time. Keep
 * its bytes outside g_vx_vfiles until finish so fopen can never observe a
 * prefix after a failed or interrupted transfer. */
typedef struct VxVFileUpload {
    char path[256];
    unsigned char* data;
    size_t size;
    size_t written;
    int active;
} VxVFileUpload;

static VxVFileUpload g_vx_vfile_upload;

static void vx_wasm_clear_file_upload(int release_data) {
    if (release_data && g_vx_vfile_upload.data) {
        free(g_vx_vfile_upload.data);
    }
    g_vx_vfile_upload.path[0] = '\0';
    g_vx_vfile_upload.data = NULL;
    g_vx_vfile_upload.size = 0u;
    g_vx_vfile_upload.written = 0u;
    g_vx_vfile_upload.active = 0;
}

static int vx_wasm_file_slot_available(const char* path) {
    int free_slot = 0;
    int index;
    for (index = 0; index < VX_MAX_VFILES; index++) {
        if (g_vx_vfiles[index]->in_use) {
            if (strcmp(g_vx_vfiles[index]->path, path) == 0) return 1;
        } else {
            free_slot = 1;
        }
    }
    return free_slot || vx_wasm_grow_file_slots() >= 0;
}

/* SafeTensors needs an immutable view of a model-scale file, not another
 * model-scale allocation. Browser VFS snapshots are already immutable while
 * readers hold them, so pin their backing store and let the parser borrow it
 * just as native builds borrow an mmap. These helpers remain private to the
 * portable-control amalgamation and do not add a browser ABI export. */
static const unsigned char* vx_wasm_file_map(const char* path,
                                             size_t* out_size) {
    int index;
    if (!path || !out_size) return NULL;
    for (index = 0; index < VX_MAX_VFILES; index++) {
        VxVFile* vfile = g_vx_vfiles[index];
        int stream_index;
        if (!vfile->in_use || strcmp(vfile->path, path) != 0) continue;
        if (vfile->unlink_pending || !vfile->data ||
            vfile->map_count == SIZE_MAX)
            return NULL;
        for (stream_index = 0; stream_index < VX_MAX_STREAMS;
             stream_index++)
            if (g_vx_streams[stream_index].file == vfile &&
                g_vx_streams[stream_index].mode)
                return NULL;
        vfile->map_count++;
        *out_size = vfile->size;
        return vfile->data;
    }
    return NULL;
}

static int vx_wasm_file_unmap(const void* data) {
    int index;
    if (!data) return -1;
    for (index = 0; index < VX_MAX_VFILES; index++) {
        VxVFile* vfile = g_vx_vfiles[index];
        if (!vfile->in_use || vfile->data != data || !vfile->map_count)
            continue;
        vfile->map_count--;
        if (!vfile->map_count && vfile->unlink_pending) {
            if (vfile->is_dynamic && vfile->data) free(vfile->data);
            memset(vfile, 0, sizeof(*vfile));
        }
        return 0;
    }
    return -1;
}

int vx_wasm_mount_file(const char* path, const unsigned char* data, size_t size) {
    if (!path || !data || !vx_wasm_file_slot_available(path)) return -1;
    for (int i = 0; i < VX_MAX_VFILES; i++) {
        if (g_vx_vfiles[i]->in_use && strcmp(g_vx_vfiles[i]->path, path) == 0) {
            if (g_vx_vfiles[i]->map_count) return -1;
            if (g_vx_vfiles[i]->is_dynamic && g_vx_vfiles[i]->data) free(g_vx_vfiles[i]->data);
            g_vx_vfiles[i]->data = (unsigned char*)data;
            g_vx_vfiles[i]->size = size;
            g_vx_vfiles[i]->capacity = size;
            g_vx_vfiles[i]->map_count = 0u;
            g_vx_vfiles[i]->is_dynamic = 0;
            g_vx_vfiles[i]->unlink_pending = 0;
            return 0;
        }
    }
    for (int i = 0; i < VX_MAX_VFILES; i++) {
        if (!g_vx_vfiles[i]->in_use) {
            size_t len = strlen(path);
            if (len >= sizeof(g_vx_vfiles[i]->path)) return -1;
            memcpy(g_vx_vfiles[i]->path, path, len + 1);
            g_vx_vfiles[i]->data = (unsigned char*)data;
            g_vx_vfiles[i]->size = size;
            g_vx_vfiles[i]->capacity = size;
            g_vx_vfiles[i]->map_count = 0u;
            g_vx_vfiles[i]->is_dynamic = 0;
            g_vx_vfiles[i]->unlink_pending = 0;
            g_vx_vfiles[i]->in_use = 1;
            return 0;
        }
    }
    return -1;
}

/* Begin a copy-owning mount without publishing a partially uploaded file.
 * The size is a wasm32 size_t; allocation failure remains an observable
 * nonzero status (or a WebAssembly trap when linear memory cannot grow). */
int vx_wasm_mount_file_begin(const char* path, size_t size) {
    size_t path_length;
    if (!path || g_vx_vfile_upload.active) return -1;
    path_length = strlen(path);
    if (path_length == 0u || path_length >= sizeof(g_vx_vfile_upload.path) ||
        !vx_wasm_file_slot_available(path)) {
        return -1;
    }
    memcpy(g_vx_vfile_upload.path, path, path_length + 1u);
    g_vx_vfile_upload.data = NULL;
    g_vx_vfile_upload.size = size;
    g_vx_vfile_upload.written = 0u;
    g_vx_vfile_upload.active = 1;
    g_vx_vfile_upload.data = (unsigned char*)malloc(size == 0u ? 1u : size);
    if (!g_vx_vfile_upload.data) {
        vx_wasm_clear_file_upload(0);
        return -1;
    }
    return 0;
}

/* Append exactly one sequential host-mailbox chunk. Offsets live only in C,
 * so JavaScript never has to project a model-scale destination pointer. */
int vx_wasm_mount_file_write(const unsigned char* data, size_t size) {
    size_t remaining;
    if (!g_vx_vfile_upload.active || (!data && size != 0u)) return -1;
    if (g_vx_vfile_upload.written > g_vx_vfile_upload.size) return -1;
    remaining = g_vx_vfile_upload.size - g_vx_vfile_upload.written;
    if (size > remaining) return -1;
    if (size != 0u) {
        memcpy(g_vx_vfile_upload.data + g_vx_vfile_upload.written, data, size);
        g_vx_vfile_upload.written += size;
    }
    return 0;
}

/* Publish atomically only after the declared byte count arrived. */
int vx_wasm_mount_file_finish(void) {
    int target = -1;
    int free_slot = -1;
    int index;
    if (!g_vx_vfile_upload.active ||
        g_vx_vfile_upload.written != g_vx_vfile_upload.size) {
        return -1;
    }
    for (index = 0; index < VX_MAX_VFILES; index++) {
        if (g_vx_vfiles[index]->in_use &&
            strcmp(g_vx_vfiles[index]->path, g_vx_vfile_upload.path) == 0) {
            target = index;
            break;
        }
        if (!g_vx_vfiles[index]->in_use && free_slot < 0) free_slot = index;
    }
    if (target < 0) target = free_slot;
    if (target < 0) target = vx_wasm_grow_file_slots();
    if (target < 0) return -1;
    if (g_vx_vfiles[target]->in_use && g_vx_vfiles[target]->map_count)
        return -1;
    if (g_vx_vfiles[target]->in_use && g_vx_vfiles[target]->is_dynamic &&
        g_vx_vfiles[target]->data) {
        free(g_vx_vfiles[target]->data);
    }
    memcpy(g_vx_vfiles[target]->path, g_vx_vfile_upload.path,
           strlen(g_vx_vfile_upload.path) + 1u);
    g_vx_vfiles[target]->data = g_vx_vfile_upload.data;
    g_vx_vfiles[target]->size = g_vx_vfile_upload.size;
    g_vx_vfiles[target]->capacity = g_vx_vfile_upload.size;
    g_vx_vfiles[target]->map_count = 0u;
    g_vx_vfiles[target]->is_dynamic = 1;
    g_vx_vfiles[target]->unlink_pending = 0;
    g_vx_vfiles[target]->in_use = 1;
    vx_wasm_clear_file_upload(0);
    return 0;
}

/* Idempotent cleanup for every failed/trapped host-side transaction. */
int vx_wasm_mount_file_abort(void) {
    vx_wasm_clear_file_upload(1);
    return 0;
}

int vx_wasm_unmount_file(const char* path) {
    int stream_index;
    if (!path) return -1;
    for (stream_index = 0; stream_index < VX_MAX_STREAMS; stream_index++) {
        if (g_vx_streams[stream_index].file &&
            strcmp(g_vx_streams[stream_index].file->path, path) == 0) {
            return -1;
        }
    }
    return remove(path);
}

FILE* fopen(const char* path, const char* mode) {
    if (!path || !mode) return NULL;
    int is_write = (strchr(mode, 'w') != NULL || strchr(mode, 'a') != NULL);
    VxVFile* vfile = NULL;

    for (int i = 0; i < VX_MAX_VFILES; i++) {
        if (g_vx_vfiles[i]->in_use && strcmp(g_vx_vfiles[i]->path, path) == 0) {
            vfile = g_vx_vfiles[i];
            break;
        }
    }

    if (!vfile) {
        if (!is_write || !vx_wasm_file_slot_available(path)) return NULL;
        for (int i = 0; i < VX_MAX_VFILES; i++) {
            if (!g_vx_vfiles[i]->in_use) {
                vfile = g_vx_vfiles[i];
                size_t len = strlen(path);
                if (len >= sizeof(vfile->path)) return NULL;
                memcpy(vfile->path, path, len + 1);
                vfile->data = NULL;
                vfile->size = 0;
                vfile->capacity = 0;
                vfile->is_dynamic = 1;
                vfile->in_use = 1;
                break;
            }
        }
        if (!vfile) return NULL;
    } else if (is_write) {
        if (vfile->map_count) return NULL;
        if (strchr(mode, 'w') != NULL) {
            if (vfile->is_dynamic && vfile->data) {
                free(vfile->data);
                vfile->data = NULL;
            }
            vfile->size = 0;
            vfile->capacity = 0;
            vfile->is_dynamic = 1;
        }
    }

    for (int i = 0; i < VX_MAX_STREAMS; i++) {
        if (!g_vx_streams[i].file) {
            g_vx_streams[i].file = vfile;
            g_vx_streams[i].position = (strchr(mode, 'a') != NULL) ? vfile->size : 0;
            g_vx_streams[i].mode = is_write ? 1 : 0;
            return &g_vx_streams[i];
        }
    }
    return NULL;
}

int fclose(FILE* stream) {
    if (!stream || !stream->file) return -1;
    stream->file = NULL;
    stream->position = 0;
    return 0;
}

size_t fread(void* ptr, size_t size, size_t count, FILE* stream) {
    if (!stream || !stream->file || !ptr || !size || !count) return 0;
    if (size > SIZE_MAX / count) return 0;
    size_t total = size * count;
    VxVFile* vfile = stream->file;
    if (stream->position >= vfile->size) return 0;
    size_t avail = vfile->size - stream->position;
    size_t to_read = (total < avail) ? total : avail;
    memcpy(ptr, vfile->data + stream->position, to_read);
    stream->position += to_read;
    return to_read / size;
}

size_t fwrite(const void* ptr, size_t size, size_t count, FILE* stream) {
    if (!stream || !stream->file || !ptr || !size || !count) return 0;
    if (size > SIZE_MAX / count) return 0;
    size_t total = size * count;
    VxVFile* vfile = stream->file;
    if (stream->position > SIZE_MAX - total) return 0;
    size_t end_pos = stream->position + total;
    if (end_pos > vfile->capacity) {
        size_t new_cap = end_pos < 1024u ? 1024u : end_pos;
        if (end_pos <= SIZE_MAX / 2u) new_cap = end_pos * 2u;
        unsigned char* new_data = (unsigned char*)malloc(new_cap);
        if (!new_data) return 0;
        if (vfile->data && vfile->size > 0) {
            memcpy(new_data, vfile->data, vfile->size);
        }
        if (vfile->is_dynamic && vfile->data) {
            free(vfile->data);
        }
        vfile->data = new_data;
        vfile->capacity = new_cap;
        vfile->is_dynamic = 1;
    }
    memcpy(vfile->data + stream->position, ptr, total);
    stream->position = end_pos;
    if (end_pos > vfile->size) vfile->size = end_pos;
    return count;
}

int fseek(FILE* stream, long offset, int origin) {
    if (!stream || !stream->file) return -1;
    size_t new_pos = 0;
    if (origin == SEEK_SET) {
        if (offset < 0) return -1;
        new_pos = (size_t)offset;
    } else if (origin == SEEK_CUR) {
        long target = (long)stream->position + offset;
        if (target < 0) return -1;
        new_pos = (size_t)target;
    } else if (origin == SEEK_END) {
        long target = (long)stream->file->size + offset;
        if (target < 0) return -1;
        new_pos = (size_t)target;
    } else {
        return -1;
    }
    stream->position = new_pos;
    return 0;
}

long ftell(FILE* stream) {
    if (!stream || !stream->file) return -1;
    return (long)stream->position;
}

int ferror(FILE* stream) {
    (void)stream;
    return 0;
}

int fputc(int character, FILE* stream) {
    unsigned char byte = (unsigned char)character;
    return fwrite(&byte, 1, 1, stream) == 1 ? byte : EOF;
}

int fflush(FILE* stream) {
    (void)stream;
    return 0;
}

char* fgets(char* str, int num, FILE* stream) {
    if (!str || num <= 1 || !stream || !stream->file) return NULL;
    VxVFile* vfile = stream->file;
    if (stream->position >= vfile->size) return NULL;
    int i = 0;
    while (i < num - 1 && stream->position < vfile->size) {
        char c = (char)vfile->data[stream->position++];
        str[i++] = c;
        if (c == '\n') break;
    }
    str[i] = '\0';
    return str;
}

int remove(const char* path) {
    if (!path) return -1;
    for (int i = 0; i < VX_MAX_VFILES; i++) {
        if (g_vx_vfiles[i]->in_use && strcmp(g_vx_vfiles[i]->path, path) == 0) {
            if (g_vx_vfiles[i]->map_count) {
                g_vx_vfiles[i]->path[0] = '\0';
                g_vx_vfiles[i]->unlink_pending = 1;
                return 0;
            }
            if (g_vx_vfiles[i]->is_dynamic && g_vx_vfiles[i]->data) {
                free(g_vx_vfiles[i]->data);
            }
            memset(g_vx_vfiles[i], 0, sizeof(*g_vx_vfiles[i]));
            return 0;
        }
    }
    return -1;
}

int rename(const char* oldpath, const char* newpath) {
    if (!oldpath || !newpath) return -1;
    for (int i = 0; i < VX_MAX_VFILES; i++) {
        if (g_vx_vfiles[i]->in_use && strcmp(g_vx_vfiles[i]->path, oldpath) == 0) {
            size_t len = strlen(newpath);
            if (len >= sizeof(g_vx_vfiles[i]->path)) return -1;
            if (!strcmp(oldpath, newpath)) return 0;
            /* Atomic writers replace a pre-created temporary destination.
             * Keep mapped readers of the old destination alive via unlink. */
            (void)remove(newpath);
            memcpy(g_vx_vfiles[i]->path, newpath, len + 1);
            return 0;
        }
    }
    return -1;
}

extern void vx_wasm_host_log(const char* msg, int len);

int fprintf(FILE* stream, const char* format, ...) {
    (void)stream;
    char buf[512];
    va_list args;
    va_start(args, format);
    int n = vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);
    if (n > 0) {
        vx_wasm_host_log(buf, n);
    }
    return n;
}

int printf(const char* format, ...) {
    char buf[512];
    va_list args;
    va_start(args, format);
    int n = vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);
    if (n > 0) {
        vx_wasm_host_log(buf, n);
    }
    return n;
}

char* getenv(const char* name) {
    (void)name;
    return NULL;
}

void* memchr(const void* s, int c, size_t n) {
    const unsigned char* p = (const unsigned char*)s;
    unsigned char uc = (unsigned char)c;
    while (n--) {
        if (*p == uc) return (void*)p;
        p++;
    }
    return NULL;
}

unsigned long long strtoull(const char* nptr, char** endptr, int base) {
    const char* s = nptr;
    unsigned long long acc = 0;
    int c;
    unsigned long long cutoff;
    int neg = 0, any = 0, cutlim;

    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
    if (*s == '-') {
        neg = 1;
        s++;
    } else if (*s == '+') {
        s++;
    }
    if ((base == 0 || base == 16) && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        s += 2;
        base = 16;
    }
    if (base == 0) base = (s[0] == '0') ? 8 : 10;

    cutoff = (unsigned long long)-1 / (unsigned long long)base;
    cutlim = (int)((unsigned long long)-1 % (unsigned long long)base);
    for (; (c = *s); s++) {
        if (c >= '0' && c <= '9') c -= '0';
        else if (c >= 'A' && c <= 'Z') c -= 'A' - 10;
        else if (c >= 'a' && c <= 'z') c -= 'a' - 10;
        else break;
        if (c >= base) break;
        if (any < 0 || acc > cutoff || (acc == cutoff && c > cutlim)) {
            any = -1;
        } else {
            any = 1;
            acc *= base;
            acc += c;
        }
    }
    if (any < 0) {
        acc = (unsigned long long)-1;
    } else if (neg) {
        acc = -acc;
    }
    if (endptr != 0) *endptr = (char*)(any ? s : nptr);
    return acc;
}

long strtol(const char* nptr, char** endptr, int base) {
    return (long)strtoull(nptr, endptr, base);
}

/* Private host transport, inventoried in wasm_internal_abi_manifest.mjs.
 * Platform samples and scheduler policy read exactly this clock domain. */
__attribute__((import_module("host"), import_name("vx_host_monotonic_micros_v1")))
extern double vx_host_monotonic_micros_v1(void);

int clock_gettime(int clock_id, struct timespec* out) {
    double micros;
    uint64_t value;
    if (!out || clock_id != CLOCK_MONOTONIC) return -1;
    micros = vx_host_monotonic_micros_v1();
    /* Ordered comparisons also reject NaN. time_t is signed wasm32 long. */
    if (!(micros >= 0.0 && micros < 2147483648000000.0)) return -1;
    value = (uint64_t)micros;
    out->tv_sec = (time_t)(value / UINT64_C(1000000));
    out->tv_nsec = (long)((value % UINT64_C(1000000)) * UINT64_C(1000));
    return 0;
}

__attribute__((weak)) void vx_wasm_host_log(const char* msg, int len) {
    (void)msg;
    (void)len;
}

#undef sqrtf
#undef fabsf
#undef floorf
#undef sqrt
#undef fabs
#undef floor
#undef lrintf
#undef lrint
float sqrtf(float x) { return __builtin_sqrtf(x); }
float fabsf(float x) { return __builtin_fabsf(x); }
float floorf(float x) { return __builtin_floorf(x); }
double sqrt(double x) { return __builtin_sqrt(x); }
double fabs(double x) { return __builtin_fabs(x); }
double floor(double x) { return __builtin_floor(x); }
long lrintf(float x) { return (long)__builtin_lrintf(x); }
long lrint(double x) { return (long)__builtin_lrint(x); }

typedef unsigned int vx_uint128_t __attribute__((mode(TI)));
typedef int vx_int128_t __attribute__((mode(TI)));

vx_int128_t __multi3(vx_int128_t a, vx_int128_t b) {
    uint64_t a_lo = (uint64_t)a, a_hi = (uint64_t)(a >> 64);
    uint64_t b_lo = (uint64_t)b, b_hi = (uint64_t)(b >> 64);

    uint64_t a0 = (uint32_t)a_lo;
    uint64_t a1 = a_lo >> 32;
    uint64_t b0 = (uint32_t)b_lo;
    uint64_t b1 = b_lo >> 32;

    uint64_t p00 = a0 * b0;
    uint64_t p01 = a0 * b1;
    uint64_t p10 = a1 * b0;
    uint64_t p11 = a1 * b1;

    uint64_t mid = p01 + (p00 >> 32);
    mid += p10;
    if (mid < p10) {
        p11 += ((uint64_t)1 << 32);
    }
    uint64_t low = (p00 & 0xffffffffULL) | (mid << 32);
    uint64_t high = p11 + (mid >> 32);

    high += a_lo * b_hi + a_hi * b_lo;

    return ((vx_int128_t)high << 64) | low;
}
#endif /* __wasm__ */
