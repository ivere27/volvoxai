/* C PTQ authoring, driven from files.
 *
 * Run with no arguments it checks its own error handling — the cases where
 * authoring must refuse rather than produce a graph nobody can use.
 *
 * Run with paths it authors one template:
 *
 *   test_ptq_authoring <graph.json> <weights.safetensors> <template.json>
 *                      [--activation i8|u8] [--scheme symmetric|asymmetric]
 *
 * and prints the derived plan to stdout, one field per line, so a caller in
 * another language can compare it against its own. python/tests/
 * test_ptq_authoring_parity.py is that caller: it writes the same fixture
 * `tools/exporter/typed_ptq.py` authors, runs this, and diffs the two
 * templates. That comparison is the point — two implementations of the same
 * rewrite are only worth having if they agree.
 */
#include "ptq_authoring.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

static void expect_refusal(const char* label, int result,
                           const VxPtqAuthored* authored) {
    if (result == 0) {
        fprintf(stderr, "FAIL %s: authoring succeeded when it should refuse\n",
                label);
        failures++;
        return;
    }
    if (!authored->message[0]) {
        fprintf(stderr, "FAIL %s: refused without saying why\n", label);
        failures++;
    }
}

static int self_check(void) {
    VxPtqAuthoringConfig config = VX_PTQ_AUTHORING_CONFIG_INIT;
    VxPtqAuthored authored;
    char long_name[129];
    char long_name_graph[2048];
    char* template_json = NULL;
    size_t template_length = 0u;
    int result;

    result = vx_ptq_author_template(NULL, NULL, 0u, "out.json", &config,
                                    &authored);
    expect_refusal("missing source path", result, &authored);
    vx_ptq_authored_free(&authored);

    result = vx_ptq_author_template("no-such-graph.json", NULL, 0u, "out.json",
                                    &config, &authored);
    expect_refusal("unreadable source", result, &authored);
    vx_ptq_authored_free(&authored);

    /* A caller built against an older header passes a smaller struct. Taking
     * it would read fields that are not there. */
    config.struct_size = sizeof(config) - 1u;
    result = vx_ptq_author_template("graph.json", NULL, 0u, "out.json", &config,
                                    &authored);
    expect_refusal("stale config size", result, &authored);
    vx_ptq_authored_free(&authored);

    /* Weights are packed symmetrically. A zero point on a weight survives
     * into the accumulator, where it costs a correction on every output. */
    config.struct_size = sizeof(config);
    config.weight_storage = VX_PTQ_STORAGE_U8;
    result = vx_ptq_author_template("graph.json", NULL, 0u, "out.json", &config,
                                    &authored);
    expect_refusal("asymmetric weights", result, &authored);
    vx_ptq_authored_free(&authored);

    /* A structural name limit is an invalid graph, not an allocation failure.
     * Keeping those statuses distinct is part of the generated API contract. */
    memset(long_name, 'x', sizeof(long_name) - 1u);
    long_name[sizeof(long_name) - 1u] = '\0';
    snprintf(long_name_graph, sizeof(long_name_graph),
             "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},"
             "\"inputs\":{\"%s\":{\"shape\":[1],\"dtype\":\"float32\"}},"
             "\"nodes\":[{\"id\":\"gelu\",\"opType\":\"GELU\","
             "\"inputs\":{\"input\":\"%s\"},\"outputs\":{\"out\":{"
             "\"tensor\":\"output\",\"shape\":[1],\"dtype\":\"float32\"}},"
             "\"params\":{}}],\"outputs\":[\"output\"]}",
             long_name, long_name);
    config.weight_storage = VX_PTQ_STORAGE_I8;
    result = vx_ptq_author_graph(
        long_name_graph, strlen(long_name_graph), NULL, 0u, &config,
        &template_json, &template_length, &authored);
    expect_refusal("overlong tensor name", result, &authored);
    if (authored.status != VX_STATUS_INVALID_GRAPH) {
        fprintf(stderr,
                "FAIL overlong tensor name: status %d instead of INVALID_GRAPH\n",
                (int)authored.status);
        failures++;
    }
    vx_ptq_authored_release_template(template_json);
    vx_ptq_authored_free(&authored);

    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}

int main(int argc, char** argv) {
    VxPtqAuthoringConfig config = VX_PTQ_AUTHORING_CONFIG_INIT;
    VxPtqAuthored authored;
    const char* weights;
    size_t index;
    int argument;

    if (argc < 4) return self_check();

    for (argument = 4; argument + 1 < argc; argument += 2) {
        if (strcmp(argv[argument], "--activation") == 0) {
            config.activation_storage = strcmp(argv[argument + 1], "u8") == 0
                                            ? VX_PTQ_STORAGE_U8
                                            : VX_PTQ_STORAGE_I8;
        } else if (strcmp(argv[argument], "--scheme") == 0) {
            config.activation_scheme =
                strcmp(argv[argument + 1], "asymmetric") == 0
                    ? VX_PTQ_SCHEME_ASYMMETRIC
                    : VX_PTQ_SCHEME_SYMMETRIC;
        } else {
            fprintf(stderr, "unknown option %s\n", argv[argument]);
            return 2;
        }
    }

    weights = argv[2];
    if (vx_ptq_author_template(argv[1], &weights, 1u, argv[3], &config,
                               &authored) != 0) {
        fprintf(stderr, "%s\n", authored.message);
        vx_ptq_authored_free(&authored);
        return 1;
    }

    printf("quantized_nodes %llu\n",
           (unsigned long long)authored.quantized_nodes);
    printf("retained_float_nodes %llu\n",
           (unsigned long long)authored.retained_float_nodes);
    for (index = 0; index < authored.observer_count; index++) {
        const VxPtqAuthoredObserver* observer = &authored.observers[index];
        printf("observer %s %s %s\n", observer->tensor_name,
               observer->storage == VX_PTQ_STORAGE_U8 ? "u8" : "i8",
               observer->scheme == VX_PTQ_SCHEME_ASYMMETRIC ? "asymmetric"
                                                            : "symmetric");
    }
    for (index = 0; index < authored.layer_count; index++) {
        const VxPtqAuthoredLayer* layer = &authored.layers[index];
        printf("layer %d axis=%d input=%s output=%s weight=%s->%s bias=%s->%s\n",
               (int)layer->node_index, (int)layer->weight_axis,
               layer->input_tensor_name, layer->output_tensor_name,
               layer->source_weight_name, layer->packed_weight_name,
               layer->source_bias_name[0] ? layer->source_bias_name : "-",
               layer->packed_bias_name[0] ? layer->packed_bias_name : "-");
    }
    vx_ptq_authored_free(&authored);
    return 0;
}
