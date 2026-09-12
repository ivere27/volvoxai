/* The names C authoring gives the tensors it introduces.
 *
 * These are not arbitrary. `tools/exporter/typed_ptq.py` derives them the same
 * way, and the expectations below were taken from its output for the fixture
 * in python/tests/ptq_fixture.py. If the two implementations disagree here,
 * they produce different templates for the same graph and nothing downstream
 * can be compared.
 *
 * The SHA-256 vectors come first, because a wrong digest would make every name
 * wrong in a way that looks like a naming bug.
 */
#include "ptq_names.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

static void expect_hex(const char* label, const char* input,
                       const char* expected) {
    char hex[65];
    vx_ptq_sha256_hex(input, strlen(input), hex);
    if (strcmp(hex, expected) != 0) {
        fprintf(stderr, "FAIL %s\n  want %s\n  got  %s\n", label, expected, hex);
        failures++;
    }
}

static void expect_name(const char* key, const char* role,
                        const char* expected) {
    char name[VX_PTQ_ALLOCATED_NAME_CAPACITY];
    if (vx_ptq_allocate_name(NULL, key, role, name) != 0) {
        fprintf(stderr, "FAIL allocate(%s, %s) returned -1\n", key, role);
        failures++;
        return;
    }
    if (strcmp(name, expected) != 0) {
        fprintf(stderr, "FAIL allocate(%s, %s)\n  want %s\n  got  %s\n",
                key, role, expected, name);
        failures++;
    }
}

int main(void) {
    VxPtqNameSet occupied;
    char first[VX_PTQ_ALLOCATED_NAME_CAPACITY];
    char second[VX_PTQ_ALLOCATED_NAME_CAPACITY];
    char reserved[VX_PTQ_ALLOCATED_NAME_CAPACITY];

    /* FIPS 180-4 vectors, plus the empty string, which is where a length or
     * padding mistake shows up first. */
    expect_hex("sha256(\"\")", "",
               "e3b0c44298fc1c149afbf4c8996fb924"
               "27ae41e4649b934ca495991b7852b855");
    expect_hex("sha256(\"abc\")", "abc",
               "ba7816bf8f01cfea414140de5dae2223"
               "b00361a396177a9cb410ff61f20015ad");
    expect_hex("sha256(56 chars)",
               "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
               "248d6a61d20638b8e5c026930c3e6039"
               "a33ce45964ff2167f6ecedd419db06c1");
    /* 119 bytes: the length lands in the second block's padding, which is the
     * case a single-block implementation gets wrong. */
    expect_hex("sha256(119 chars)",
               "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
               "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
               "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
               "31eba51c313a5c08226adf18d4a359cf"
               "dfd8d2e816b13f4af952f7ea6584dcfb");

    /* Activation affines are keyed by the tensor they quantize. */
    expect_name("hidden", "activation",
                "__ptq__.393d04789d3785411bbf.activation");
    expect_name("hidden", "scale", "__ptq__.117e5e51b42efebcaf10.scale");
    expect_name("hidden", "zero_point",
                "__ptq__.405f2c85b5891108297e.zero_point");
    expect_name("hidden", "quantize", "__ptq__.4c074ba6b69e65fed84e.quantize");
    expect_name("output", "activation",
                "__ptq__.d1a50ae25891668024fc.activation");

    /* Weight payloads are keyed by the node that reads them, not by the
     * source weight — two nodes may share one weight and still need distinct
     * packed tensors. */
    expect_name("node_1", "weight", "__ptq__.3b25914eb0448b4626d4.weight");
    expect_name("node_1", "bias", "__ptq__.7ba80c83c7e4b48d093d.bias");
    expect_name("node_1", "weight_scale",
                "__ptq__.286e28d6a2218668c018.weight_scale");
    expect_name("node_1", "weight_zero_point",
                "__ptq__.1c4040c13faacd00ddf3.weight_zero_point");
    expect_name("node_3", "weight", "__ptq__.11f49ad633f4da2ab3f2.weight");
    expect_name("node_5", "weight", "__ptq__.b9b87ffdf1663abffaf3.weight");
    expect_name("node_5", "dequantize",
                "__ptq__.0046188248b7fd4f8a8b.dequantize");

    /* The separator matters: without it these two would collide. */
    vx_ptq_allocate_name(NULL, "ab", "c", first);
    vx_ptq_allocate_name(NULL, "a", "bc", second);
    if (strcmp(first, second) == 0) {
        fprintf(stderr, "FAIL (\"ab\",\"c\") and (\"a\",\"bc\") collide\n");
        failures++;
    }

    /* A name the graph already declares is stepped over, not overwritten. */
    vx_ptq_name_set_init(&occupied);
    vx_ptq_allocate_name(NULL, "hidden", "activation", reserved);
    vx_ptq_name_set_add(&occupied, reserved);
    if (vx_ptq_allocate_name(&occupied, "hidden", "activation", first) != 0) {
        fprintf(stderr, "FAIL collision path returned -1\n");
        failures++;
    } else if (strcmp(first, "__ptq__.393d04789d3785411bbf.activation.1") != 0) {
        fprintf(stderr, "FAIL collision suffix\n  got  %s\n", first);
        failures++;
    }
    vx_ptq_name_set_free(&occupied);

    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
