/* Public native-CPU integration check. The companion .mjs runner replaces
 * the CLI main object in each CMake link.txt and adds
 * --wrap=vx_call_sequence_policy_validate_response_v1. */
#include "vx_lifecycle.h"
#include "call_sequence_policy.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    TEST_VALUE_COUNT = 8,
    TEST_NODE_COUNT = 3,
    TEST_CHANGED_CAPACITY = 2
};

typedef struct FixedExpectation {
    const char* label;
    uint32_t changed_count;
    uint32_t changed[TEST_CHANGED_CAPACITY];
    uint32_t selected_count;
    uint32_t selected[TEST_NODE_COUNT];
    unsigned validations_before;
    int reject_after_replay;
    int armed;
} FixedExpectation;

static FixedExpectation g_fixed_expectation;
static unsigned g_fixed_validation_count;
static unsigned g_wrapper_failures;

int32_t __real_vx_call_sequence_policy_validate_response_v1(
    const uint8_t* request,
    uint32_t request_bytes,
    const uint8_t* response,
    uint32_t response_bytes,
    uint8_t* replay_response,
    uint32_t replay_response_bytes,
    uint8_t* replay_scratch,
    uint32_t replay_scratch_bytes);

static uint32_t read_u32(const uint8_t* bytes) {
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8u) |
           ((uint32_t)bytes[2] << 16u) |
           ((uint32_t)bytes[3] << 24u);
}

static void wrapper_failure(const char* format, ...) {
    va_list arguments;
    ++g_wrapper_failures;
    fprintf(stderr, "validator wrapper failure");
    if (g_fixed_expectation.label)
        fprintf(stderr, " (%s)", g_fixed_expectation.label);
    fputs(": ", stderr);
    va_start(arguments, format);
    vfprintf(stderr, format, arguments);
    va_end(arguments);
    fputc('\n', stderr);
}

static int expected_node_selected(uint32_t node_index) {
    for (uint32_t index = 0u;
         index < g_fixed_expectation.selected_count; ++index) {
        if (g_fixed_expectation.selected[index] == node_index) return 1;
    }
    return 0;
}

int32_t __wrap_vx_call_sequence_policy_validate_response_v1(
        const uint8_t* request,
        uint32_t request_bytes,
        const uint8_t* response,
        uint32_t response_bytes,
        uint8_t* replay_response,
        uint32_t replay_response_bytes,
        uint8_t* replay_scratch,
        uint32_t replay_scratch_bytes) {
    int32_t replay_valid =
        __real_vx_call_sequence_policy_validate_response_v1(
            request, request_bytes, response, response_bytes,
            replay_response, replay_response_bytes,
            replay_scratch, replay_scratch_bytes);
    uint32_t policy_kind;
    uint32_t changed_count;
    uint32_t changed_offset;
    uint32_t node_count;
    uint32_t node_offset;
    uint32_t selected_count;
    int malformed = 0;

    if (!replay_valid || !request ||
        request_bytes < sizeof(VxCallSequencePolicyRequestV1))
        return replay_valid;
    policy_kind = read_u32(
        request + offsetof(VxCallSequencePolicyRequestV1, policy_kind));
    if (policy_kind != VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY)
        return replay_valid;

    ++g_fixed_validation_count;
    if (!g_fixed_expectation.armed) {
        wrapper_failure("unexpected FIXED replay validation");
        return 0;
    }
    changed_count = read_u32(
        request + offsetof(VxCallSequencePolicyRequestV1, changed_input_count));
    changed_offset = read_u32(
        request + offsetof(VxCallSequencePolicyRequestV1, changed_input_offset));
    if (changed_count != g_fixed_expectation.changed_count) {
        wrapper_failure("changed count is %u, expected %u", changed_count,
                        g_fixed_expectation.changed_count);
        malformed = 1;
    }
    if ((changed_count == 0u && changed_offset != 0u) ||
        (changed_count != 0u &&
         ((uint64_t)changed_offset +
              (uint64_t)changed_count *
                  sizeof(VxCallSequenceChangedInputV1) > request_bytes))) {
        wrapper_failure("changed-input section is out of bounds");
        malformed = 1;
    } else {
        for (uint32_t index = 0u;
             index < changed_count &&
                 index < g_fixed_expectation.changed_count; ++index) {
            const uint32_t actual = read_u32(
                request + changed_offset +
                    (size_t)index * sizeof(VxCallSequenceChangedInputV1));
            if (actual != g_fixed_expectation.changed[index]) {
                wrapper_failure(
                    "changed index %u is %u, expected %u", index, actual,
                    g_fixed_expectation.changed[index]);
                malformed = 1;
            }
        }
    }

    if (!response ||
        response_bytes < sizeof(VxCallSequencePolicyResponseV1)) {
        wrapper_failure("successful replay returned a short response");
        malformed = 1;
        node_count = 0u;
        node_offset = 0u;
        selected_count = 0u;
    } else {
        node_count = read_u32(
            response + offsetof(VxCallSequencePolicyResponseV1, node_count));
        node_offset = read_u32(
            response + offsetof(VxCallSequencePolicyResponseV1, node_offset));
        selected_count = read_u32(
            response + offsetof(
                VxCallSequencePolicyResponseV1, selected_node_count));
        if (read_u32(response + offsetof(
                VxCallSequencePolicyResponseV1, selection_kind)) !=
                VX_CALL_SEQUENCE_SELECTION_EXPLICIT ||
            read_u32(response + offsetof(
                VxCallSequencePolicyResponseV1, changed_input_count)) !=
                g_fixed_expectation.changed_count ||
            node_count != TEST_NODE_COUNT ||
            selected_count != g_fixed_expectation.selected_count ||
            (uint64_t)node_offset +
                    (uint64_t)node_count * sizeof(VxCallSequenceNodeV1) >
                response_bytes) {
            wrapper_failure(
                "response summary is not the expected explicit 3-node closure");
            malformed = 1;
        } else {
            for (uint32_t node = 0u; node < node_count; ++node) {
                const uint8_t* record = response + node_offset +
                    (size_t)node * sizeof(VxCallSequenceNodeV1);
                const uint32_t actual_index = read_u32(record);
                const uint32_t flags = read_u32(record + 4u);
                const uint32_t expected_flags = expected_node_selected(node)
                    ? VX_CALL_SEQUENCE_NODE_SELECTED : 0u;
                if (actual_index != node || flags != expected_flags) {
                    wrapper_failure(
                        "node %u record is index=%u flags=%u, expected flags=%u",
                        node, actual_index, flags, expected_flags);
                    malformed = 1;
                }
            }
        }
    }

    g_fixed_expectation.armed = 0;
    if (malformed || g_fixed_expectation.reject_after_replay) return 0;
    return replay_valid;
}

static int test_failure(const char* format, ...) {
    va_list arguments;
    fputs("native CPU selection test failure: ", stderr);
    va_start(arguments, format);
    vfprintf(stderr, format, arguments);
    va_end(arguments);
    fputc('\n', stderr);
    return 0;
}

static int status_is(const char* label, VxStatus actual, VxStatus expected,
                     const VxReport* report) {
    if (actual == expected) return 1;
    return test_failure(
        "%s returned %s (%d), expected %s (%d); code=%d message=%s",
        label, vx_status_string(actual), (int)actual,
        vx_status_string(expected), (int)expected,
        report ? report->code : VX_CODE_NONE,
        report && report->message[0] ? report->message : "none");
}

static int contains(const char* label, const char* value,
                    const char* expected) {
    if (value && strstr(value, expected)) return 1;
    return test_failure("%s '%s' does not contain '%s'", label,
                        value ? value : "(null)", expected);
}

static VxTensorBinding binding(const char* name, const float* values) {
    VxTensorBinding result = VX_TENSOR_BINDING_INIT;
    result.name = name;
    result.dtype = VX_DTYPE_F32;
    result.rank = 3u;
    result.shape[0] = 1;
    result.shape[1] = 4;
    result.shape[2] = 2;
    result.data = values;
    result.byte_size = TEST_VALUE_COUNT * sizeof(float);
    result.location = VX_MEMORY_HOST;
    return result;
}

static void add_values(const float* left, const float* right,
                       float* output) {
    for (size_t index = 0u; index < TEST_VALUE_COUNT; ++index)
        output[index] = left[index] + right[index];
}

static int result_is(const char* label, VxResult* result,
                     const float expected[TEST_VALUE_COUNT]) {
    float actual[TEST_VALUE_COUNT] = {0};
    size_t required = 0u;
    VxReport report = VX_REPORT_INIT;
    VxStatus status;
    if (!result) return test_failure("%s returned a NULL result", label);
    if (vx_result_output_count(result) != 1u)
        return test_failure("%s returned an unexpected output count", label);
    status = vx_result_read(result, "sum", actual, sizeof(actual),
                            &required, &report);
    if (!status_is(label, status, VX_STATUS_OK, &report)) return 0;
    if (required != sizeof(actual))
        return test_failure("%s reported %zu output bytes, expected %zu",
                            label, required, sizeof(actual));
    for (size_t index = 0u; index < TEST_VALUE_COUNT; ++index) {
        if (actual[index] != expected[index])
            return test_failure(
                "%s output[%zu]=%.9g, expected %.9g", label, index,
                actual[index], expected[index]);
    }
    return 1;
}

static int arm_fixed(const char* label, const uint32_t* changed,
                     uint32_t changed_count, const uint32_t* selected,
                     uint32_t selected_count, int reject_after_replay) {
    if (g_fixed_expectation.armed)
        return test_failure("cannot arm %s while %s remains armed", label,
                            g_fixed_expectation.label);
    if (changed_count > TEST_CHANGED_CAPACITY ||
        selected_count > TEST_NODE_COUNT)
        return test_failure("%s expectation exceeds fixture capacity", label);
    memset(&g_fixed_expectation, 0, sizeof(g_fixed_expectation));
    g_fixed_expectation.label = label;
    g_fixed_expectation.changed_count = changed_count;
    g_fixed_expectation.selected_count = selected_count;
    g_fixed_expectation.validations_before = g_fixed_validation_count;
    g_fixed_expectation.reject_after_replay = reject_after_replay;
    g_fixed_expectation.armed = 1;
    if (changed_count)
        memcpy(g_fixed_expectation.changed, changed,
               changed_count * sizeof(*changed));
    if (selected_count)
        memcpy(g_fixed_expectation.selected, selected,
               selected_count * sizeof(*selected));
    return 1;
}

static int finish_fixed(const char* label) {
    if (g_fixed_expectation.armed)
        return test_failure("%s did not reach FIXED replay validation", label);
    if (g_fixed_validation_count !=
            g_fixed_expectation.validations_before + 1u)
        return test_failure("%s performed %u FIXED replay validations", label,
                            g_fixed_validation_count -
                                g_fixed_expectation.validations_before);
    if (g_wrapper_failures)
        return test_failure("%s observed %u validator wrapper failures", label,
                            g_wrapper_failures);
    return 1;
}

static int fixed_count_unchanged(const char* label, unsigned before) {
    if (g_fixed_validation_count == before && !g_fixed_expectation.armed)
        return 1;
    return test_failure("%s unexpectedly reached FIXED replay validation",
                        label);
}

static int create_context(VxCompiledModel* compiled, VxDecodeRowMode row_mode,
                          VxExecutionContext** out_context, VxReport* report) {
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxStatus status;
    options.decode_row_mode = row_mode;
    options.require_incremental = 1;
    status = vx_compiled_model_create_context(
        compiled, &options, out_context, report);
    return status_is("vx_compiled_model_create_context", status,
                     VX_STATUS_OK, report);
}

int main(int argc, char** argv) {
    static const uint32_t changed_alpha[] = {0u};
    static const uint32_t changed_zeta[] = {1u};
    static const uint32_t changed_both[] = {0u, 1u};
    static const uint32_t selected_alpha[] = {1u, 2u};
    static const uint32_t selected_zeta[] = {0u, 2u};
    static const uint32_t selected_both[] = {0u, 1u, 2u};
    static const char* const cpu_backends[] = {"cpu"};
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxResult* result = NULL;
    VxStatus status;
    unsigned validations_before;
    int return_code = 1;
    float expected[TEST_VALUE_COUNT];
    float alpha[TEST_VALUE_COUNT] = {1, 2, 3, 4, 5, 6, 7, 8};
    float zeta[TEST_VALUE_COUNT] = {10, 20, 30, 40, 50, 60, 70, 80};
    float bad_alpha[TEST_VALUE_COUNT] = {
        1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007
    };
    VxTensorBinding inputs[2];

    if (argc != 3) {
        fprintf(stderr, "usage: %s <graph.json> <profile>\n",
                argc > 0 ? argv[0] : "native_cpu_selection_test");
        return 2;
    }

    runtime_options.cpu_threads = 1;
    runtime_options.execution_mode = VX_EXECUTION_MODE_DIRECT;
    source.graph_path = argv[1];
    policy.mode = VX_BACKEND_REQUIRE;
    policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    policy.backends = cpu_backends;
    policy.backend_count = 1u;

    status = vx_runtime_create(&runtime_options, &runtime, &report);
    if (!status_is("vx_runtime_create", status, VX_STATUS_OK, &report))
        goto cleanup;
    status = vx_runtime_load_model(runtime, &source, &model, &report);
    if (!status_is("vx_runtime_load_model", status, VX_STATUS_OK, &report))
        goto cleanup;
    status = vx_model_compile(model, &policy, &compiled, &report);
    if (!status_is("vx_model_compile", status, VX_STATUS_OK, &report))
        goto cleanup;
    if (!create_context(compiled, VX_DECODE_ROW_AUTO, &context, &report))
        goto cleanup;

    inputs[0] = binding("zeta", zeta);
    inputs[1] = binding("alpha", alpha);
    validations_before = g_fixed_validation_count;
    status = vx_execution_context_decode_prefill(
        context, 0, inputs, 2u, &result, &report);
    if (!status_is("AUTO prefill", status, VX_STATUS_OK, &report) ||
        !fixed_count_unchanged("AUTO prefill", validations_before))
        goto cleanup;
    add_values(zeta, alpha, expected);
    if (!result_is("AUTO prefill", result, expected) ||
        !contains("AUTO prefill decode state", report.decode_state,
                  "prefilled=1") ||
        !contains("AUTO prefill decode state", report.decode_state,
                  "mode=row") ||
        !contains("AUTO prefill decode state", report.decode_state,
                  "last=dependency") ||
        strstr(report.decode_state, "position=") != NULL)
        goto cleanup;
    vx_result_release(result);
    result = NULL;

    zeta[2] = 300;
    zeta[3] = 400;
    inputs[0] = binding("zeta", zeta);
    if (!arm_fixed("zeta row", changed_zeta, 1u,
                   selected_zeta, 2u, 0))
        goto cleanup;
    status = vx_execution_context_decode_step(
        context, 1, inputs, 1u, &result, &report);
    if (!status_is("zeta row", status, VX_STATUS_OK, &report) ||
        !finish_fixed("zeta row"))
        goto cleanup;
    add_values(zeta, alpha, expected);
    if (!result_is("zeta row", result, expected) ||
        !contains("zeta row decode state", report.decode_state, "last=row") ||
        !contains("zeta row decode position", report.decode_state,
                  "position=1"))
        goto cleanup;
    vx_result_release(result);
    result = NULL;

    alpha[4] = 500;
    alpha[5] = 600;
    inputs[0] = binding("alpha", alpha);
    if (!arm_fixed("alpha row", changed_alpha, 1u,
                   selected_alpha, 2u, 0))
        goto cleanup;
    status = vx_execution_context_decode_step(
        context, 2, inputs, 1u, &result, &report);
    if (!status_is("alpha row", status, VX_STATUS_OK, &report) ||
        !finish_fixed("alpha row"))
        goto cleanup;
    add_values(zeta, alpha, expected);
    if (!result_is("alpha row", result, expected) ||
        !contains("alpha row decode state", report.decode_state, "last=row"))
        goto cleanup;
    vx_result_release(result);
    result = NULL;

    {
        const float next_alpha[TEST_VALUE_COUNT] = {2, 3, 4, 5, 6, 7, 8, 9};
        const float next_zeta[TEST_VALUE_COUNT] = {
            20, 30, 40, 50, 60, 70, 80, 90
        };
        memcpy(alpha, next_alpha, sizeof(alpha));
        memcpy(zeta, next_zeta, sizeof(zeta));
    }
    /* Deliberately reverse declaration order. The wire must still contain
     * canonical alpha=0,zeta=1 records. */
    inputs[0] = binding("zeta", zeta);
    inputs[1] = binding("alpha", alpha);
    if (!arm_fixed("both dependency", changed_both, 2u,
                   selected_both, 3u, 0))
        goto cleanup;
    status = vx_execution_context_decode_step(
        context, -1, inputs, 2u, &result, &report);
    if (!status_is("both dependency", status, VX_STATUS_OK, &report) ||
        !finish_fixed("both dependency"))
        goto cleanup;
    add_values(zeta, alpha, expected);
    if (!result_is("both dependency", result, expected) ||
        !contains("dependency decode state", report.decode_state,
                  "last=dependency"))
        goto cleanup;
    vx_result_release(result);
    result = NULL;

    if (!arm_fixed("empty dependency", NULL, 0u, NULL, 0u, 0))
        goto cleanup;
    status = vx_execution_context_decode_step(
        context, -1, NULL, 0u, &result, &report);
    if (!status_is("empty dependency", status, VX_STATUS_OK, &report) ||
        !finish_fixed("empty dependency") ||
        !result_is("empty dependency", result, expected))
        goto cleanup;
    vx_result_release(result);
    result = NULL;

    inputs[0] = binding("alpha", bad_alpha);
    if (!arm_fixed("replay rejection", changed_alpha, 1u,
                   selected_alpha, 2u, 1))
        goto cleanup;
    status = vx_execution_context_decode_step(
        context, -1, inputs, 1u, &result, &report);
    if (!status_is("replay rejection", status, VX_STATUS_INTERNAL, &report) ||
        !finish_fixed("replay rejection") || result != NULL ||
        report.code != VX_CODE_CALL_SEQUENCE_POLICY_INVALID ||
        !contains("replay rejection decode state", report.decode_state,
                  "prefilled=1"))
        goto cleanup;

    for (size_t index = 0u; index < TEST_VALUE_COUNT; ++index)
        zeta[index] += 1.0f;
    inputs[0] = binding("zeta", zeta);
    if (!arm_fixed("post-rejection dependency", changed_zeta, 1u,
                   selected_zeta, 2u, 0))
        goto cleanup;
    status = vx_execution_context_decode_step(
        context, -1, inputs, 1u, &result, &report);
    if (!status_is("post-rejection dependency", status,
                   VX_STATUS_OK, &report) ||
        !finish_fixed("post-rejection dependency"))
        goto cleanup;
    add_values(zeta, alpha, expected);
    if (!result_is("post-rejection dependency", result, expected))
        goto cleanup;
    vx_result_release(result);
    result = NULL;

    {
        const float refresh_alpha[TEST_VALUE_COUNT] = {
            101, 102, 103, 104, 105, 106, 107, 108
        };
        const float refresh_zeta[TEST_VALUE_COUNT] = {
            201, 202, 203, 204, 205, 206, 207, 208
        };
        memcpy(alpha, refresh_alpha, sizeof(alpha));
        memcpy(zeta, refresh_zeta, sizeof(zeta));
    }
    inputs[0] = binding("zeta", zeta);
    inputs[1] = binding("alpha", alpha);
    validations_before = g_fixed_validation_count;
    status = vx_execution_context_decode_prefill(
        context, 0, inputs, 2u, &result, &report);
    if (!status_is("replacement prefill", status, VX_STATUS_OK, &report) ||
        !fixed_count_unchanged("replacement prefill", validations_before))
        goto cleanup;
    add_values(zeta, alpha, expected);
    if (!result_is("replacement prefill", result, expected)) goto cleanup;
    vx_result_release(result);
    result = NULL;

    validations_before = g_fixed_validation_count;
    status = vx_execution_context_decode_reset(context, &report);
    if (!status_is("decode reset", status, VX_STATUS_OK, &report) ||
        !fixed_count_unchanged("decode reset", validations_before) ||
        !contains("reset decode state", report.decode_state, "prefilled=0"))
        goto cleanup;
    inputs[0] = binding("alpha", bad_alpha);
    status = vx_execution_context_decode_step(
        context, -1, inputs, 1u, &result, &report);
    if (!status_is("post-reset step", status,
                   VX_STATUS_INVALID_ARGUMENT, &report) || result != NULL ||
        !fixed_count_unchanged("post-reset step", validations_before) ||
        !contains("post-reset decode state", report.decode_state,
                  "prefilled=0"))
        goto cleanup;

    inputs[0] = binding("zeta", zeta);
    inputs[1] = binding("alpha", alpha);
    status = vx_execution_context_decode_prefill(
        context, 0, inputs, 2u, &result, &report);
    if (!status_is("reset recovery prefill", status,
                   VX_STATUS_OK, &report) ||
        !result_is("reset recovery prefill", result, expected))
        goto cleanup;
    vx_result_release(result);
    result = NULL;

    {
        const float ordinary_alpha[TEST_VALUE_COUNT] = {
            11, 12, 13, 14, 15, 16, 17, 18
        };
        const float ordinary_zeta[TEST_VALUE_COUNT] = {
            31, 32, 33, 34, 35, 36, 37, 38
        };
        memcpy(alpha, ordinary_alpha, sizeof(alpha));
        memcpy(zeta, ordinary_zeta, sizeof(zeta));
    }
    inputs[0] = binding("zeta", zeta);
    inputs[1] = binding("alpha", alpha);
    validations_before = g_fixed_validation_count;
    status = vx_execution_context_execute(
        context, inputs, 2u, &result, &report);
    if (!status_is("ordinary execution", status, VX_STATUS_OK, &report) ||
        !fixed_count_unchanged("ordinary execution", validations_before))
        goto cleanup;
    add_values(zeta, alpha, expected);
    if (!result_is("ordinary execution", result, expected)) goto cleanup;
    vx_result_release(result);
    result = NULL;

    zeta[2] += 1000.0f;
    zeta[3] += 1000.0f;
    inputs[0] = binding("zeta", zeta);
    status = vx_execution_context_decode_step(
        context, 1, inputs, 1u, &result, &report);
    if (!status_is("stale-prefill step", status,
                   VX_STATUS_INVALID_ARGUMENT, &report) || result != NULL ||
        !fixed_count_unchanged("stale-prefill step", validations_before) ||
        !contains("stale-prefill decode state", report.decode_state,
                  "prefilled=0"))
        goto cleanup;

    inputs[0] = binding("zeta", zeta);
    inputs[1] = binding("alpha", alpha);
    add_values(zeta, alpha, expected);
    status = vx_execution_context_decode_prefill(
        context, 0, inputs, 2u, &result, &report);
    if (!status_is("stale-prefill recovery", status,
                   VX_STATUS_OK, &report) ||
        !result_is("stale-prefill recovery", result, expected))
        goto cleanup;
    vx_result_release(result);
    result = NULL;

    (void)vx_execution_context_close(context, NULL);
    vx_execution_context_release(context);
    context = NULL;

    if (!create_context(compiled, VX_DECODE_ROW_REQUIRED,
                        &context, &report))
        goto cleanup;
    {
        const float required_alpha[TEST_VALUE_COUNT] = {
            41, 42, 43, 44, 45, 46, 47, 48
        };
        const float required_zeta[TEST_VALUE_COUNT] = {
            51, 52, 53, 54, 55, 56, 57, 58
        };
        memcpy(alpha, required_alpha, sizeof(alpha));
        memcpy(zeta, required_zeta, sizeof(zeta));
    }
    inputs[0] = binding("zeta", zeta);
    inputs[1] = binding("alpha", alpha);
    status = vx_execution_context_decode_prefill(
        context, 0, inputs, 2u, &result, &report);
    add_values(zeta, alpha, expected);
    if (!status_is("REQUIRED prefill", status, VX_STATUS_OK, &report) ||
        !result_is("REQUIRED prefill", result, expected) ||
        !contains("REQUIRED prefill state", report.decode_state, "mode=row"))
        goto cleanup;
    vx_result_release(result);
    result = NULL;

    validations_before = g_fixed_validation_count;
    inputs[0] = binding("alpha", bad_alpha);
    status = vx_execution_context_decode_step(
        context, -1, inputs, 1u, &result, &report);
    if (!status_is("REQUIRED dependency rejection", status,
                   VX_STATUS_INVALID_ARGUMENT, &report) || result != NULL ||
        !fixed_count_unchanged("REQUIRED dependency rejection",
                               validations_before))
        goto cleanup;
    status = vx_execution_context_decode_step(
        context, 0, inputs, 1u, &result, &report);
    if (!status_is("REQUIRED row-zero rejection", status,
                   VX_STATUS_INVALID_ARGUMENT, &report) || result != NULL ||
        !fixed_count_unchanged("REQUIRED row-zero rejection",
                               validations_before))
        goto cleanup;

    zeta[2] = 501;
    zeta[3] = 502;
    inputs[0] = binding("zeta", zeta);
    if (!arm_fixed("REQUIRED row", changed_zeta, 1u,
                   selected_zeta, 2u, 0))
        goto cleanup;
    status = vx_execution_context_decode_step(
        context, 1, inputs, 1u, &result, &report);
    if (!status_is("REQUIRED row", status, VX_STATUS_OK, &report) ||
        !finish_fixed("REQUIRED row"))
        goto cleanup;
    add_values(zeta, alpha, expected);
    if (!result_is("REQUIRED row", result, expected) ||
        !contains("REQUIRED row state", report.decode_state, "last=row"))
        goto cleanup;
    vx_result_release(result);
    result = NULL;

    if (g_fixed_expectation.armed || g_wrapper_failures)
        goto cleanup;
    printf("Verified native CPU portable-C selection (%s): "
           "canonical mapping, explicit closures, row/dependency/empty, "
           "replay precommit, refresh/reset/stale-prefill, REQUIRED precommit.\n",
           argv[2]);
    return_code = 0;

cleanup:
    vx_result_release(result);
    if (context) (void)vx_execution_context_close(context, NULL);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    if (runtime) (void)vx_runtime_close(runtime, NULL);
    vx_runtime_release(runtime);
    return return_code;
}
