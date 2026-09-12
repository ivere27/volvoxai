/* The freestanding C library, checked against a real one.
 *
 * `native/src/runtime/wasm_libc.c` supplies what the WebAssembly build has
 * none of. Most of it is memcpy and friends. The part worth defending is
 * number printing, because cJSON renders every shape, channel count and group
 * count in a template through it — and a shape printed as 7.9999999 is not a
 * shape.
 *
 * Decimal parsing and formatting use the project's exact-integer conversion.
 * The host library supplies an independent round-trip and formatting oracle.
 *
 * The file is compiled for the host with its public names renamed out of the
 * way, so the two implementations can be run side by side.
 */
#include <stdint.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Renamed so the file under test can define malloc and memcpy without
 * colliding with the host's. */
#define malloc vx_wasm_test_malloc
#define calloc vx_wasm_test_calloc
#define realloc vx_wasm_test_realloc
#define free vx_wasm_test_free
#define memcpy vx_wasm_test_memcpy
#define memset vx_wasm_test_memset
#define memcmp vx_wasm_test_memcmp
#define memmove vx_wasm_test_memmove
#define strlen vx_wasm_test_strlen
#define strcmp vx_wasm_test_strcmp
#define strncmp vx_wasm_test_strncmp
#define strcpy vx_wasm_test_strcpy
#define strncpy vx_wasm_test_strncpy
#define strchr vx_wasm_test_strchr
#define strtod vx_wasm_test_strtod
#define qsort vx_wasm_test_qsort
#define snprintf vx_wasm_test_snprintf
#define vsnprintf vx_wasm_test_vsnprintf
#define sprintf vx_wasm_test_sprintf
#define sscanf vx_wasm_test_sscanf
/* The linker normally places this; the host build needs somewhere to point. */
#define __builtin_wasm_memory_size(index) ((size_t)0)
#define __builtin_wasm_memory_grow(index, pages) ((size_t)-1)
unsigned char __heap_base;

/* The file under test calls these through the renaming macros above, and the
 * freestanding headers that would declare them are not on this include path. */
#include <stdarg.h>
void* vx_wasm_test_malloc(size_t);
void* vx_wasm_test_calloc(size_t, size_t);
void* vx_wasm_test_realloc(void*, size_t);
void vx_wasm_test_free(void*);
void* vx_wasm_test_memcpy(void*, const void*, size_t);
void* vx_wasm_test_memset(void*, int, size_t);
int vx_wasm_test_memcmp(const void*, const void*, size_t);
void* vx_wasm_test_memmove(void*, const void*, size_t);
size_t vx_wasm_test_strlen(const char*);
int vx_wasm_test_strcmp(const char*, const char*);
int vx_wasm_test_strncmp(const char*, const char*, size_t);
char* vx_wasm_test_strcpy(char*, const char*);
char* vx_wasm_test_strncpy(char*, const char*, size_t);
char* vx_wasm_test_strchr(const char*, int);
double vx_wasm_test_strtod(const char*, char**);
void vx_wasm_test_qsort(void*, size_t, size_t,
                        int (*)(const void*, const void*));
int vx_wasm_test_snprintf(char*, size_t, const char*, ...);
int vx_wasm_test_vsnprintf(char*, size_t, const char*, va_list);
int vx_wasm_test_sprintf(char*, const char*, ...);
int vx_wasm_test_sscanf(const char*, const char*, ...);

#include "../src/runtime/wasm_libc.c"
#undef snprintf
#undef strtod
#undef strcmp
#undef strlen
#undef qsort

typedef struct VxSortRecord {
    int key;
    int source_order;
} VxSortRecord;

static size_t g_sort_comparisons = 0u;

static int compare_sort_record(const void* left, const void* right) {
    const VxSortRecord* lhs = (const VxSortRecord*)left;
    const VxSortRecord* rhs = (const VxSortRecord*)right;
    g_sort_comparisons++;
    return (lhs->key > rhs->key) - (lhs->key < rhs->key);
}

static int sort_records_are_stable(const VxSortRecord* records,
                                   size_t count) {
    size_t index;
    for (index = 1u; index < count; index++) {
        if (records[index - 1u].key > records[index].key ||
            (records[index - 1u].key == records[index].key &&
             records[index - 1u].source_order >
                 records[index].source_order))
            return 0;
    }
    return 1;
}

static int check_decimal_parse(const char* text) {
    char *actual_end, *expected_end;
    double actual = vx_wasm_test_strtod(text, &actual_end);
    double expected = strtod(text, &expected_end);
    if ((!(isnan(actual) && isnan(expected)) && memcmp(&actual, &expected, sizeof(actual))) ||
        actual_end != expected_end) {
        fprintf(stderr, "decimal parse failed: %.80s => %a, expected %a\n", text, actual, expected);
        return 0;
    }
    return 1;
}

static int check_decimal_boundaries(void) {
    static const char* texts[] = {
        "0", "-0", "-0e999999999999999999999", "1e999999999999999999999",
        "1e-999999999999999999999", "  -12.75tail", "1e+", "1e-", ".", "-",
        ".5", "1.", "1.e3",
        "9007199254740993", "2.2250738585072012e-308", "4.9406564584124654e-324",
        "2.4703282292062327e-324", "2.4703282292062328e-324", "1.7976931348623159e308",
    };
    for (unsigned i = 0; i < sizeof(texts) / sizeof(texts[0]); i++)
        if (!check_decimal_parse(texts[i])) return 0;

#if LDBL_MANT_DIG > DBL_MANT_DIG && LDBL_MIN_EXP < DBL_MIN_EXP
    /* Exact midpoints, then one decimal unit above/below at the 1500th
     * fractional digit. This tests ties-to-even and discarded nonzero tails
     * beyond the parser's retained prefix, including zero/subnormal ties. */
    static const double neighbors[] = {0, 0x1p-1074, DBL_MIN, 1, 1.5, 2, DBL_MAX};
    for (unsigned i = 0; i < sizeof(neighbors) / sizeof(neighbors[0]); i++) {
        long double lower = neighbors[i];
        long double upper = neighbors[i] == DBL_MAX ? scalbnl(1, 1024) : nextafter(neighbors[i], INFINITY);
        long double midpoint = (lower + upper) / 2;
        for (int sign = -1; sign <= 1; sign += 2) {
            char text[1900];
            int count = snprintf(text, sizeof(text), "%.1500Lf", midpoint * sign);
            if (count <= 0 || count >= (int)sizeof(text) || !check_decimal_parse(text)) return 0;
            text[count - 1] = '1';
            if (!check_decimal_parse(text)) return 0;
            text[count - 1] = '0';
            int digit = count - 1;
            while (text[digit] == '0' || text[digit] == '.') {
                if (text[digit] == '0') text[digit] = '9';
                digit--;
            }
            text[digit]--;
            if (!check_decimal_parse(text)) return 0;
        }
    }
#endif
    char long_text[4000];
    long_text[0] = '1';
    memset(long_text + 1, '0', 3000);
    strcpy(long_text + 3001, "e-3000");
    if (!check_decimal_parse(long_text)) return 0;
    long_text[3000] = '1';
    return check_decimal_parse(long_text);
}

static int check_decimal_formats(void) {
    static const char formats[] = "gGeEfF";
    static const double special[] = {
        0, -0.0, INFINITY, -INFINITY, NAN, -NAN, 2.5, 3.5,
        9.999999, 0.0000999999, 0.00000999999, DBL_MAX, DBL_MIN, 0x1p-1074,
    };
    uint64_t bits = UINT64_C(0x9b85c725ac037e16);
    for (int trial = 0; trial < 1024; trial++) {
        double value;
        bits ^= bits << 13; bits ^= bits >> 7; bits ^= bits << 17;
        memcpy(&value, &bits, sizeof(value));
        if (trial < (int)(sizeof(special) / sizeof(special[0]))) value = special[trial];
        for (unsigned specifier = 0; specifier < sizeof(formats) - 1; specifier++) {
            for (int precision = 0; precision <= 20; precision++) {
                char format[32], actual[384], expected[384];
                snprintf(format, sizeof(format), trial & 1 ? "%%030.%d%c" : "%%7.%d%c", precision, formats[specifier]);
                size_t capacity = trial % 4 == 0 ? 0 : trial % 4 == 1 ? 1 : trial % 4 == 2 ? 9 : sizeof(actual);
                int mine = vx_wasm_test_snprintf(actual, capacity, format, value);
                int theirs = snprintf(expected, capacity, format, value);
                if (mine != theirs || (capacity && strcmp(actual, expected))) {
                    fprintf(stderr, "format %s of %a failed with capacity %zu: %d versus %d bytes\n",
                            format, value, capacity, mine, theirs);
                    return 0;
                }
            }
        }
    }
    char actual[16], expected[16];
    int mine = vx_wasm_test_snprintf(actual, sizeof(actual), "%.2000e", DBL_MIN);
    volatile int large_precision = 2000;
    int theirs = snprintf(expected, sizeof(expected), "%.*e", large_precision, DBL_MIN);
    return mine == theirs && !strcmp(actual, expected);
}

int main(void) {
    long integer_mismatch = 0;
    long text_mismatch = 0;
    long trial;
    size_t merge_sort_comparisons = 0u;
    size_t insertion_sort_comparisons = 0u;

    if (!check_decimal_boundaries() || !check_decimal_formats()) return 1;

    /* Exercise the allocating merge-sort path with a real backing region.
     * The source's host-test sbrk normally has no linear memory to grow. */
    {
        enum { RECORD_COUNT = 4096, TEST_HEAP_BYTES = 256 * 1024 };
        static union {
            max_align_t alignment;
            unsigned char bytes[TEST_HEAP_BYTES];
        } heap;
        VxSortRecord records[RECORD_COUNT];
        VxSortRecord insertion_records[RECORD_COUNT];
        size_t index;
        vx_wasm_blocks = NULL;
        vx_wasm_tail = NULL;
        vx_wasm_break = heap.bytes;
        vx_wasm_limit = heap.bytes + sizeof(heap.bytes);
        for (index = 0u; index < RECORD_COUNT; index++) {
            records[index].key =
                (int)((RECORD_COUNT - 1u - index) / 4u);
            records[index].source_order = (int)index;
            insertion_records[index] = records[index];
        }
        g_sort_comparisons = 0u;
        vx_wasm_insertion_sort(
            (unsigned char*)insertion_records, RECORD_COUNT,
            sizeof(insertion_records[0]), compare_sort_record);
        insertion_sort_comparisons = g_sort_comparisons;
        g_sort_comparisons = 0u;
        vx_wasm_test_qsort(records, RECORD_COUNT, sizeof(records[0]),
                           compare_sort_record);
        merge_sort_comparisons = g_sort_comparisons;
        if (!sort_records_are_stable(records, RECORD_COUNT)) {
            fputs("stable qsort order failed for model-scale input\n", stderr);
            return 1;
        }
        if (merge_sort_comparisons > (size_t)RECORD_COUNT * 16u ||
            insertion_sort_comparisons < merge_sort_comparisons * 100u) {
            fprintf(stderr,
                    "qsort used %zu comparisons versus insertion sort's %zu "
                    "for %d records; expected O(n log n)\n",
                    merge_sort_comparisons, insertion_sort_comparisons,
                    RECORD_COUNT);
            return 1;
        }
    }

    /* Non-power-of-two counts exercise truncated merge runs on both sides of
     * the short-input cutoff. */
    {
        static const size_t counts[] = {17u, 31u, 33u};
        VxSortRecord records[33];
        size_t trial_index;
        for (trial_index = 0u;
             trial_index < sizeof(counts) / sizeof(counts[0]);
             trial_index++) {
            size_t count = counts[trial_index];
            size_t index;
            for (index = 0u; index < count; index++) {
                records[index].key = (int)((count - 1u - index) / 3u);
                records[index].source_order = (int)index;
            }
            vx_wasm_test_qsort(records, count, sizeof(records[0]),
                               compare_sort_record);
            if (!sort_records_are_stable(records, count)) {
                fprintf(stderr,
                        "stable qsort tail-run order failed for %zu records\n",
                        count);
                return 1;
            }
        }
    }

    /* qsort cannot report scratch OOM. Force that branch and pin its
     * correctness-preserving stable insertion fallback. */
    {
        static union {
            max_align_t alignment;
            unsigned char bytes[1024];
        } exhausted_heap;
        VxSortRecord records[33];
        size_t index;
        vx_wasm_blocks = NULL;
        vx_wasm_tail = NULL;
        vx_wasm_break = exhausted_heap.bytes;
        vx_wasm_limit = exhausted_heap.bytes + 1u;
        for (index = 0u; index < 33u; index++) {
            records[index].key = (int)((32u - index) / 3u);
            records[index].source_order = (int)index;
        }
        vx_wasm_test_qsort(records, 33u, sizeof(records[0]),
                           compare_sort_record);
        if (!sort_records_are_stable(records, 33u)) {
            fputs("stable qsort scratch-OOM fallback failed\n", stderr);
            return 1;
        }
    }

    /* Integer shapes and full-range floating parameters share the formatter. */
    {
        uint64_t bits = UINT64_C(0x123456789abcdef0);
        for (int index = 0; index < 4096; index++) {
            double value, parsed;
            char printed[64], expected[64];
            bits ^= bits << 13; bits ^= bits >> 7; bits ^= bits << 17;
            memcpy(&value, &bits, sizeof(value));
            if (!isfinite(value)) continue;
            vx_wasm_test_snprintf(printed, sizeof(printed), "%.17g", value);
            snprintf(expected, sizeof(expected), "%.17g", value);
            parsed = vx_wasm_test_strtod(printed, NULL);
            if (strcmp(printed, expected) || memcmp(&value, &parsed, sizeof(value))) {
                fprintf(stderr, "decimal round trip failed: %s versus %s\n", printed, expected);
                return 1;
            }
        }
    }

    for (trial = -200000; trial < 200000; trial++) {
        char mine[64];
        char theirs[64];
        double value = (double)trial;
        vx_wasm_test_snprintf(mine, sizeof(mine), "%1.15g", value);
        snprintf(theirs, sizeof(theirs), "%1.15g", value);
        if (strcmp(mine, theirs)) {
            if (integer_mismatch < 4) {
                printf("FAIL %.17g printed as %s, expected %s\n", value, mine,
                       theirs);
            }
            integer_mismatch++;
        }
    }

    /* The conversions authoring itself performs: names, error messages, and
     * the version string cJSON builds. */
    {
        struct { const char* format; const char* expected; long argument; }
        cases[] = {
            { "%d", "0", 0 },
            { "%d", "-1", -1 },
            { "%d", "2147483647", 2147483647L },
            { "%u", "4294967295", 4294967295L },
            { "%d nodes", "85 nodes", 85 },
        };
        unsigned index;
        for (index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
            char produced[64];
            vx_wasm_test_snprintf(produced, sizeof(produced), cases[index].format,
                                  (int)cases[index].argument);
            if (strcmp(produced, cases[index].expected)) {
                printf("FAIL %s of %ld produced %s, expected %s\n",
                       cases[index].format, cases[index].argument, produced,
                       cases[index].expected);
                text_mismatch++;
            }
        }
        {
            char produced[64];
            vx_wasm_test_snprintf(produced, sizeof(produced), "__ptq__.%s.%s",
                                  "0123456789abcdef0123", "weight");
            if (strcmp(produced, "__ptq__.0123456789abcdef0123.weight")) {
                printf("FAIL name formatting produced %s\n", produced);
                text_mismatch++;
            }
        }
        {
            char produced[16];
            vx_wasm_test_snprintf(produced, sizeof(produced), "u%04x",
                                  (unsigned int)0x1f);
            if (strcmp(produced, "u001f")) {
                printf("FAIL JSON control escape produced %s, expected u001f\n",
                       produced);
                text_mismatch++;
            }
        }
        {
            /* Truncation reports what it would have written, which is how a
             * caller detects a name that does not fit. */
            char produced[8];
            int written = vx_wasm_test_snprintf(produced, sizeof(produced),
                                                "%s", "0123456789");
            if (written != 10 || strcmp(produced, "0123456")) {
                printf("FAIL truncation wrote %d bytes as %s\n", written,
                       produced);
                text_mismatch++;
            }
        }
    }

    if (integer_mismatch || text_mismatch) {
        fprintf(stderr, "%ld integer and %ld text mismatches\n",
                integer_mismatch, text_mismatch);
        return 1;
    }
    printf("stable qsort: %zu comparisons versus insertion sort's %zu for "
           "4096 records; 400000 integers and every text conversion match "
           "the host C library\n",
           merge_sort_comparisons, insertion_sort_comparisons);
    return 0;
}
