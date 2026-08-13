#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runtime_state.h"
#include "vulkan_engine.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "vulkan training check failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int close_enough(float a, float b) {
    return fabsf(a - b) <= 1.0e-4f;
}

static int probe_equal(const VkContextStateProbe* left,
                       const VkContextStateProbe* right) {
    return left->slot_count == right->slot_count &&
        left->arena_cursor == right->arena_cursor &&
        left->dispatch_cursor == right->dispatch_cursor &&
        left->touched_count == right->touched_count &&
        left->is_training == right->is_training;
}

static int test_arena_size_configuration(void) {
    const size_t mib = (size_t)1024u * 1024u;
    const size_t maximum_mib = SIZE_MAX / mib;
    char maximum_text[64];
    char overflow_text[65];
    size_t bytes = 0u;
    int length;

    CHECK(vk_test_parse_arena_mebibytes("130", 256u, &bytes) == 0 &&
          bytes == (size_t)130u * mib);
    if (maximum_mib >= (size_t)6144u) {
        CHECK(vk_test_parse_arena_mebibytes("4096", 256u, &bytes) == 0 &&
              bytes == (size_t)4096u * mib);
        CHECK(vk_test_parse_arena_mebibytes("4097", 256u, &bytes) == 0 &&
              bytes == (size_t)4097u * mib);
        CHECK(vk_test_parse_arena_mebibytes("6144", 256u, &bytes) == 0 &&
              bytes == (size_t)6144u * mib);
    } else {
        CHECK(vk_test_parse_arena_mebibytes("4096", 256u, &bytes) == -1);
        CHECK(vk_test_parse_arena_mebibytes("4097", 256u, &bytes) == -1);
        CHECK(vk_test_parse_arena_mebibytes("6144", 256u, &bytes) == -1);
    }

    length = snprintf(maximum_text, sizeof(maximum_text), "%zu", maximum_mib);
    CHECK(length > 0 && (size_t)length < sizeof(maximum_text));
    CHECK(vk_test_parse_arena_mebibytes(
              maximum_text, 256u, &bytes) == 0 &&
          bytes == maximum_mib * mib);
    CHECK((size_t)length + 1u < sizeof(overflow_text));
    memcpy(overflow_text, maximum_text, (size_t)length);
    overflow_text[length] = '0';
    overflow_text[length + 1] = '\0';

    CHECK(vk_test_parse_arena_mebibytes("129", 256u, &bytes) == -1);
    CHECK(vk_test_parse_arena_mebibytes("", 256u, &bytes) == -1);
    CHECK(vk_test_parse_arena_mebibytes(NULL, 256u, &bytes) == -1);
    CHECK(vk_test_parse_arena_mebibytes(" 6144", 256u, &bytes) == -1);
    CHECK(vk_test_parse_arena_mebibytes("+6144", 256u, &bytes) == -1);
    CHECK(vk_test_parse_arena_mebibytes("-6144", 256u, &bytes) == -1);
    CHECK(vk_test_parse_arena_mebibytes("6144MiB", 256u, &bytes) == -1);
    CHECK(vk_test_parse_arena_mebibytes(overflow_text, 256u, &bytes) == -1);
    CHECK(vk_test_parse_arena_mebibytes("6144", 0u, &bytes) == -1);
    CHECK(vk_test_parse_arena_mebibytes("6144", 256u, NULL) == -1);
    return 0;
}

static int test_device_local_staging_path(void) {
    VkMemoryPlacementProbe before = {0};
    VkMemoryPlacementProbe after = {0};
    char telemetry[512] = {0};
    unsigned char* input = NULL;
    unsigned char* guarded_output = NULL;
    size_t bytes;
    size_t range_offset = 0u;
    size_t range_bytes = 0u;
    size_t range_end;
    CHECK(vk_test_memory_placement_read(&before) == 0 &&
          before.compute_device_local && before.arena_bytes > 0u &&
          before.arena_allocation_bytes >= before.arena_bytes &&
          before.staging_host_visible &&
          before.staging_bytes == (uint64_t)32u * 1024u * 1024u &&
          before.staging_allocation_bytes >= before.staging_bytes &&
          before.noncoherent_atom_bytes > 0u &&
          before.staging_bytes <= SIZE_MAX - 5u);
    bytes = (size_t)before.staging_bytes + 5u;
    CHECK(bytes <= SIZE_MAX - 2u);
    input = (unsigned char*)malloc(bytes);
    guarded_output = (unsigned char*)malloc(bytes + 2u);
    CHECK(input && guarded_output);
    for (size_t index = 0u; index < bytes; index++)
        input[index] = (unsigned char)(index * 131u + 17u);
    memset(guarded_output, 0xa5, bytes + 2u);
    vk_graph_begin_forward();
    CHECK(vk_test_staging_round_trip(
              input, guarded_output + 1u, bytes) == 0);
    CHECK(guarded_output[0] == 0xa5 &&
          guarded_output[bytes + 1u] == 0xa5 &&
          memcmp(input, guarded_output + 1u, bytes) == 0);
    CHECK(vk_test_memory_placement_read(&after) == 0 &&
          after.upload_count == 1u && after.upload_bytes == bytes &&
          after.download_count == 1u && after.download_bytes == bytes &&
          after.submit_count >= 3u);
    CHECK(vk_test_staging_mapped_range(
              (size_t)after.staging_bytes - 3u, 3u,
              &range_offset, &range_bytes) == 0 &&
          range_offset <= (size_t)after.staging_bytes - 3u &&
          range_offset % (size_t)after.noncoherent_atom_bytes == 0u &&
          range_offset <= SIZE_MAX - range_bytes);
    range_end = range_offset + range_bytes;
    CHECK(range_end >= (size_t)after.staging_bytes &&
          range_end <= (size_t)after.staging_allocation_bytes &&
          (range_bytes % (size_t)after.noncoherent_atom_bytes == 0u ||
           range_end == (size_t)after.staging_allocation_bytes));
    CHECK(vk_graph_execution_evidence(
              telemetry, sizeof(telemetry)) == 0 &&
          strstr(telemetry, ";vk_mem=device-local;") &&
          strstr(telemetry, ";vk_stage=33554432;") &&
          strstr(telemetry, ";vk_up=1;") &&
          strstr(telemetry, ";vk_down=1"));
    free(guarded_output);
    free(input);
    vk_graph_reset();
    return 0;
}

static int test_context_capsule_isolation(void) {
    VxEngineState* first = (VxEngineState*)calloc(1, sizeof(*first));
    VxEngineState* second = (VxEngineState*)calloc(1, sizeof(*second));
    VkContextStateProbe first_value = {7, 1441792u, 23, 11, 1};
    VkContextStateProbe second_value = {3, 1703936u, 5, 2, 0};
    VkContextStateProbe observed = {0};
    CHECK(first && second);
    CHECK(vx_engine_state_init(first) == 0);
    CHECK(vx_engine_state_init(second) == 0);

    VxEngineStateScope first_scope = vx_engine_state_scope_enter(first);
    CHECK(vk_test_context_state_write(&first_value) == 0);
    CHECK(vk_test_context_state_read(&observed) == 0);
    CHECK(probe_equal(&observed, &first_value));
    vx_engine_state_scope_leave(first_scope);

    VxEngineStateScope second_scope = vx_engine_state_scope_enter(second);
    CHECK(vk_test_context_state_read(&observed) == -1);
    CHECK(vk_test_context_state_write(&second_value) == 0);
    CHECK(vk_test_context_state_read(&observed) == 0);
    CHECK(probe_equal(&observed, &second_value));
    vx_engine_state_scope_leave(second_scope);

    first_scope = vx_engine_state_scope_enter(first);
    CHECK(vk_test_context_state_read(&observed) == 0);
    CHECK(probe_equal(&observed, &first_value));
    vx_engine_state_scope_leave(first_scope);

    second_scope = vx_engine_state_scope_enter(second);
    CHECK(vk_test_context_state_read(&observed) == 0);
    CHECK(probe_equal(&observed, &second_value));
    vx_engine_state_scope_leave(second_scope);

    vx_engine_state_deinit(second);
    vx_engine_state_deinit(first);
    free(second);
    free(first);
    return 0;
}

static int test_physical_context_isolation(VxEngineState* first) {
    VxEngineState* second = (VxEngineState*)calloc(1, sizeof(*second));
    VxEngineStateScope second_scope;
    VkContextStateProbe first_after = {0};
    VkContextStateProbe first_observed = {0};
    VkContextStateProbe second_before = {0};
    VkPipelineCacheProbe first_pipeline = {0};
    VkPipelineCacheProbe second_pipeline = {0};
    float first_input[2] = {2.0f, -3.0f};
    float first_output[2] = {0.0f, 0.0f};
    float second_input[2] = {7.0f, 11.0f};
    float second_output[2] = {0.0f, 0.0f};
    if (!first || !second || vx_engine_state_init(second) != 0) goto fail;

    vk_graph_reset();
    vk_graph_begin_forward();
    if (!vk_graph_copy_f32(first_input, first_output, 2) ||
        vk_graph_end_forward() != 0 ||
        vk_test_context_state_read(&first_after) != 0 ||
        vk_test_pipeline_cache_read(&first_pipeline) != 0 ||
        first_pipeline.prepared_kernel_count == 0u ||
        first_pipeline.prepared_pipeline_creates == 0u ||
        vk_training_begin() != 0) goto fail;

    second_scope = vx_engine_state_scope_enter(second);
    if (vk_init() != 0 ||
        vk_test_context_state_read(&second_before) != 0 ||
        second_before.slot_count != 0 || second_before.dispatch_cursor != 0 ||
        second_before.is_training != 0 ||
        vk_training_begin() != 0) {
        vx_engine_state_scope_leave(second_scope);
        goto fail_first_training;
    }
    vk_training_end();
    vk_graph_reset();
    vk_graph_begin_forward();
    if (!vk_graph_copy_f32(second_input, second_output, 2) ||
        vk_graph_end_forward() != 0 ||
        vk_test_pipeline_cache_read(&second_pipeline) != 0 ||
        second_pipeline.prepared_kernel_count == 0u ||
        second_pipeline.prepared_pipeline_creates <=
            first_pipeline.prepared_pipeline_creates ||
        second_pipeline.pipeline_cache_available !=
            first_pipeline.pipeline_cache_available ||
        !vk_graph_sync_host(second_output, sizeof(second_output), 0) ||
        !close_enough(second_output[0], second_input[0]) ||
        !close_enough(second_output[1], second_input[1])) {
        vk_cleanup();
        vx_engine_state_scope_leave(second_scope);
        goto fail_first_training;
    }
    vk_cleanup();
    vx_engine_state_scope_leave(second_scope);

    if (vk_test_context_state_read(&first_observed) != 0 ||
        first_observed.slot_count != first_after.slot_count ||
        first_observed.dispatch_cursor != first_after.dispatch_cursor ||
        first_observed.is_training != 1 ||
        !vk_graph_sync_host(first_output, sizeof(first_output), 0) ||
        !close_enough(first_output[0], first_input[0]) ||
        !close_enough(first_output[1], first_input[1])) goto fail_first_training;
    vk_training_end();
    vx_engine_state_deinit(second);
    free(second);
    return 0;

fail_first_training:
    vk_training_end();
fail:
    if (second) {
        if (second->vulkan_context_state) {
            second_scope = vx_engine_state_scope_enter(second);
            vk_cleanup();
            vx_engine_state_scope_leave(second_scope);
        }
        if (second->kernel_thread_pool) vx_engine_state_deinit(second);
        free(second);
    }
    return 1;
}

static void fill_f32_matrix(float* values, size_t count, int multiplier, int modulus,
                            int center, float scale) {
    for (size_t i = 0; i < count; i++) {
        values[i] = (float)(((int)(i % (size_t)modulus) * multiplier) % modulus - center) * scale;
    }
}

static int check_one_shot_matmul_case(int rows, int d_in, int d_out,
                                      float* input, float* weight, float* bias,
                                      float* output) {
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < d_out; column++) {
            float expected = bias[column];
            for (int k = 0; k < d_in; k++) {
                expected += input[row * d_in + k] * weight[k * d_out + column];
            }
            output[row * d_out + column] = expected;
        }
    }
    float expected[17 * 23];
    memcpy(expected, output, (size_t)rows * d_out * sizeof(float));
    memset(output, 0, (size_t)rows * d_out * sizeof(float));
    CHECK(vk_matmul(input, weight, bias, output, rows, d_in, d_out) == 1);
    for (int i = 0; i < rows * d_out; i++) CHECK(close_enough(output[i], expected[i]));
    return 0;
}

static int test_one_shot_matmul_tails_and_fallback(void) {
    float tiled_input[17 * 19];
    float tiled_weight[19 * 23];
    float tiled_bias[23];
    float tiled_output[17 * 23];
    float scalar_input[3 * 7];
    float scalar_weight[7 * 5];
    float scalar_bias[5];
    float scalar_output[3 * 5];
    fill_f32_matrix(tiled_input, 17u * 19u, 7, 17, 8, 0.0625f);
    fill_f32_matrix(tiled_weight, 19u * 23u, 11, 13, 6, 0.125f);
    fill_f32_matrix(tiled_bias, 23u, 3, 5, 2, 0.25f);
    fill_f32_matrix(scalar_input, 3u * 7u, 5, 11, 5, 0.125f);
    fill_f32_matrix(scalar_weight, 7u * 5u, 3, 7, 3, 0.25f);
    fill_f32_matrix(scalar_bias, 5u, 2, 5, 2, 0.5f);
    vk_free_weight_cache();
    CHECK(check_one_shot_matmul_case(17, 19, 23, tiled_input, tiled_weight,
                                     tiled_bias, tiled_output) == 0);
    CHECK(check_one_shot_matmul_case(3, 7, 5, scalar_input, scalar_weight,
                                     scalar_bias, scalar_output) == 0);

    /* A host allocation may be reused with a different logical matrix shape.
       The resident transpose cache must include the shape, not just its address. */
    float reshaped_weight[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float first_input[2] = {2.0f, -1.0f};
    float first_bias[3] = {0.25f, -0.5f, 1.0f};
    float first_output[3];
    float second_input[3] = {1.0f, 2.0f, 3.0f};
    float second_bias[2] = {-1.0f, 0.5f};
    float second_output[2];
    CHECK(check_one_shot_matmul_case(1, 2, 3, first_input, reshaped_weight,
                                     first_bias, first_output) == 0);
    CHECK(check_one_shot_matmul_case(1, 3, 2, second_input, reshaped_weight,
                                     second_bias, second_output) == 0);

    /* 4097x4096 F32 is one page larger than the 64 MiB resident-weight arena.
       It must decline before reading or writing through these one-float pointers. */
    float sentinel = 0.0f;
    CHECK(vk_matmul(&sentinel, &sentinel, NULL, &sentinel, 1, 4097, 4096) == 0);
    vk_free_weight_cache();
    return 0;
}

static void reference_graph_linear(const float* input, const float* weight,
                                   const float* bias, float* output, int rows,
                                   int d_in, int d_out,
                                   int output_major_weight) {
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < d_out; column++) {
            float sum = bias ? bias[column] : 0.0f;
            for (int k = 0; k < d_in; k++) {
                size_t weight_index = output_major_weight
                    ? (size_t)column * (size_t)d_in + (size_t)k
                    : (size_t)k * (size_t)d_out + (size_t)column;
                sum += input[(size_t)row * (size_t)d_in + (size_t)k] *
                    weight[weight_index];
            }
            output[(size_t)row * (size_t)d_out + (size_t)column] = sum;
        }
    }
}

static int run_reserved_graph_linear(
        float* input, const float* weight, float* output, float* expected,
        int rows, int d_in, int d_out, const char* signature) {
    size_t input_bytes = (size_t)rows * (size_t)d_in * sizeof(float);
    size_t output_bytes = (size_t)rows * (size_t)d_out * sizeof(float);
    fill_f32_matrix(input, (size_t)rows * (size_t)d_in,
                    7 + rows, 29, 14, 0.03125f);
    reference_graph_linear(input, weight, NULL, expected, rows, d_in, d_out, 1);
    CHECK(vk_graph_bind_shape_domain(signature, (VolvoxAIEnginePhysicalSpan[2]){
              {input, (size_t)17 * 19 * sizeof(float)},
              {output, (size_t)17 * 23 * sizeof(float)},
          }, 2u, 0u, 0u) == 0);
    vk_graph_mark_host(input, input_bytes, 0);
    vk_graph_begin_forward();
    CHECK(vk_graph_linear_f32(input, weight, NULL, output,
                              rows, d_in, d_out, 1) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output, output_bytes, 0) == 1);
    for (int index = 0; index < rows * d_out; index++)
        CHECK(close_enough(output[index], expected[index]));
    return 0;
}

static int test_graph_linear_dynamic_domain(void) {
    enum { MAX_ROWS = 17, D_IN = 19, D_OUT = 23 };
    float arena[MAX_ROWS * D_IN + MAX_ROWS * D_OUT];
    float* input = arena;
    float* output = arena + MAX_ROWS * D_IN;
    float weight[D_OUT * D_IN];
    float input_major_weight[D_IN * D_OUT];
    float bias[D_OUT];
    float narrower_weight[17 * D_IN];
    float narrower_output[17];
    float expected[MAX_ROWS * D_OUT];
    VkGraphDynamicStateProbe reserved = {0};
    VkGraphDynamicStateProbe maximum = {0};
    VkGraphDynamicStateProbe rebound = {0};
    VkGraphDynamicStateProbe rejected = {0};
    const VolvoxAIEnginePhysicalSpan spans[2] = {
        {input, sizeof(float) * MAX_ROWS * D_IN},
        {output, sizeof(float) * MAX_ROWS * D_OUT},
    };
    fill_f32_matrix(weight, D_OUT * D_IN, 11, 31, 15, 0.015625f);
    fill_f32_matrix(narrower_weight, 17 * D_IN,
                    9, 23, 11, 0.015625f);
    for (int k = 0; k < D_IN; k++)
        for (int column = 0; column < D_OUT; column++)
            input_major_weight[k * D_OUT + column] =
                weight[column * D_IN + k];
    fill_f32_matrix(bias, D_OUT, 5, 13, 6, 0.0625f);
    fill_f32_matrix(input, D_IN, 3, 17, 8, 0.0625f);

    vk_graph_reset();
    CHECK(vk_graph_bind_shape("linear:input-major-tiled") == 0);
    fill_f32_matrix(input, MAX_ROWS * D_IN, 13, 29, 14, 0.03125f);
    reference_graph_linear(input, input_major_weight, bias, expected,
                           MAX_ROWS, D_IN, D_OUT, 0);
    vk_graph_begin_forward();
    CHECK(vk_graph_linear_f32(input, input_major_weight, bias, output,
                              MAX_ROWS, D_IN, D_OUT, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output, sizeof(expected), 0) == 1);
    for (int index = 0; index < MAX_ROWS * D_OUT; index++)
        CHECK(close_enough(output[index], expected[index]));
    vk_graph_reset();

    CHECK(vk_graph_bind_shape("linear:bootstrap") == 0);
    vk_graph_begin_forward();
    CHECK(vk_graph_linear_f32(input, weight, NULL, output,
                              1, D_IN, D_OUT, 1) == 1);
    /* A smaller later null-bias node must not narrow the retained zero slot
     * used by the wider Linear above. */
    CHECK(vk_graph_linear_f32(input, narrower_weight, NULL,
                              narrower_output, 1, D_IN, 17, 1) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output, D_OUT * sizeof(float), 0) == 1);
    CHECK(vk_graph_bind_shape_domain(
              "linear:q1", spans, 2u, 0u, 0u) == 0);
    CHECK(vk_test_graph_dynamic_state(&reserved) == 0);
    CHECK(reserved.domain_enforced && reserved.domain_span_count == 2u &&
          reserved.slot_count == 7);

    CHECK(run_reserved_graph_linear(input, weight, output, expected,
                                    1, D_IN, D_OUT, "linear:q1") == 0);
    CHECK(run_reserved_graph_linear(input, weight, output, expected,
                                    MAX_ROWS, D_IN, D_OUT,
                                    "linear:q17") == 0);
    CHECK(vk_test_graph_dynamic_state(&maximum) == 0);
    CHECK(maximum.capacity_generation == reserved.capacity_generation &&
          maximum.active_capacity_bytes == reserved.active_capacity_bytes);
    CHECK(run_reserved_graph_linear(input, weight, output, expected,
                                    1, D_IN, D_OUT, "linear:q1") == 0);
    CHECK(vk_test_graph_dynamic_state(&rebound) == 0);
    CHECK(rebound.capacity_generation == maximum.capacity_generation &&
          rebound.active_capacity_bytes == maximum.active_capacity_bytes);

    vk_graph_begin_forward();
    CHECK(vk_graph_linear_f32(input, weight, NULL, input,
                              1, D_IN, D_OUT, 1) == 0);
    CHECK(vk_graph_linear_f32(input, weight, NULL, input + 1,
                              1, D_IN, D_OUT, 1) == 0);
    CHECK(vk_graph_linear_f32(input, weight, NULL, output,
                              0, D_IN, D_OUT, 1) == 0);
    CHECK(vk_graph_linear_f32(input, weight, NULL, output,
                              1, D_IN, D_OUT, 2) == 0);
    CHECK(vk_graph_linear_f32(input, weight, NULL, output,
                              MAX_ROWS + 1, D_IN, D_OUT, 1) == 0);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_test_graph_dynamic_state(&rejected) == 0);
    CHECK(rejected.capacity_generation == rebound.capacity_generation &&
          rejected.active_capacity_bytes == rebound.active_capacity_bytes &&
          rejected.slot_count == rebound.slot_count);
    CHECK(run_reserved_graph_linear(input, weight, output, expected,
                                    1, D_IN, D_OUT, "linear:q1") == 0);
    vk_graph_reset();
    return 0;
}

static int test_tiled_qlinear_i8u8_tails(void) {
    enum { ROWS = 9, D_IN = 19, D_OUT = 36 };
    int8_t input[ROWS * D_IN];
    int8_t weight[D_OUT * D_IN];
    float scales[D_OUT];
    int32_t zero_points[D_OUT];
    int32_t bias[D_OUT];
    int8_t output[ROWS * D_OUT];
    int8_t expected[ROWS * D_OUT];
    for (int i = 0; i < ROWS * D_IN; i++) input[i] = (int8_t)((i * 5) % 7 - 3);
    for (int i = 0; i < D_OUT * D_IN; i++) weight[i] = (int8_t)((i * 3) % 5 - 2);
    for (int column = 0; column < D_OUT; column++) {
        scales[column] = 1.0f;
        zero_points[column] = 0;
        bias[column] = column % 5 - 2;
    }
    for (int row = 0; row < ROWS; row++) for (int column = 0; column < D_OUT; column++) {
        int accumulator = bias[column];
        for (int k = 0; k < D_IN; k++) {
            accumulator += input[row * D_IN + k] * weight[column * D_IN + k];
        }
        if (accumulator < -128) accumulator = -128;
        if (accumulator > 127) accumulator = 127;
        expected[row * D_OUT + column] = (int8_t)accumulator;
    }
    memset(output, 0, sizeof(output));
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qlinear_i8u8(input, weight, scales, zero_points, bias, output,
                                ROWS, D_IN, D_OUT, 1.0f, 0, 1.0f, 0,
                                VX_DTYPE_I8, VX_DTYPE_I8,
                                VX_DTYPE_I8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    vk_graph_reset();
    return 0;
}

static int test_scalar_qlinear_i8u8_fused_word(void) {
    enum { ROWS = 1, D_IN = 12, D_OUT = 6 };
    uint8_t input[D_IN];
    int8_t weight[D_OUT * D_IN];
    float scales[D_OUT];
    int32_t zero_points[D_OUT];
    int32_t bias[D_OUT];
    uint8_t output[D_OUT];
    uint8_t expected[D_OUT];
    const int32_t input_zero_point = 127;
    const int32_t output_zero_point = 119;
    for (int k = 0; k < D_IN; k++) input[k] = (uint8_t)(123 + (k * 7) % 11);
    for (int column = 0; column < D_OUT; column++) {
        scales[column] = 1.0f;
        zero_points[column] = column - 3;
        bias[column] = column * 5 - 9;
        for (int k = 0; k < D_IN; k++)
            weight[column * D_IN + k] =
                (int8_t)(zero_points[column] + (k * 3 + column) % 7 - 3);
        int accumulator = bias[column];
        for (int k = 0; k < D_IN; k++)
            accumulator += ((int)input[k] - input_zero_point) *
                ((int)weight[column * D_IN + k] - zero_points[column]);
        int quantized = accumulator + output_zero_point;
        if (quantized < 0) quantized = 0;
        if (quantized > 255) quantized = 255;
        expected[column] = (uint8_t)quantized;
    }
    memset(output, 0, sizeof(output));
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qlinear_i8u8(
        input, weight, scales, zero_points, bias, output,
        ROWS, D_IN, D_OUT, 1.0f, input_zero_point, 1.0f,
        output_zero_point, VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    vk_graph_reset();
    return 0;
}

static int test_tiled_qlinear_staged_rounding(void) {
    enum { ROWS = 2, D_IN = 16, D_OUT = 32 };
    const uint8_t input[ROWS * D_IN] = {0};
    const int8_t weight[D_OUT * D_IN] = {0};
    float weight_scales[D_OUT];
    const int32_t weight_zero_points[D_OUT] = {0};
    int32_t bias[D_OUT];
    uint8_t output[ROWS * D_OUT] = {0};
    const float input_scale = 0.028062894940376282f;
    const float weight_scale = 0.0006811817875131965f;
    const float output_scale = 0.02981325425207615f;
    const int32_t output_zero_point = 127;

    for (int column = 0; column < D_OUT; column++) {
        weight_scales[column] = weight_scale;
        bias[column] = -3899;
    }

    /* Force the canonical f32 schedule on the host as an independent oracle:
       round scale multiplication, division, accumulator multiplication, and
       zero-point addition separately. A contracted multiply-add produces the
       adjacent value 125 for this vector. */
    volatile float scale_product = input_scale * weight_scale;
    volatile float multiplier = scale_product / output_scale;
    volatile float scaled = (float)bias[0] * multiplier;
    volatile float shifted = scaled + (float)output_zero_point;
    CHECK(shifted == 124.5f);
    CHECK(((int)floorf(shifted) & 1) == 0);

    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qlinear_i8u8(
        input, weight, weight_scales, weight_zero_points, bias, output,
        ROWS, D_IN, D_OUT, input_scale, 0, output_scale,
        output_zero_point, VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    for (int index = 0; index < ROWS * D_OUT; index++)
        CHECK(output[index] == 124u);
    vk_graph_reset();
    return 0;
}

static int test_qbatch_matmul_i8u8_arbitrary_k(void) {
    enum {
        BATCH = 2, HEADS = 4, ROWS = 3, K = 5, COLUMNS = 7,
        A_ELEMENTS = BATCH * ROWS * K,
        B_ELEMENTS = HEADS * K * COLUMNS,
        OUTPUT_ELEMENTS = BATCH * HEADS * ROWS * COLUMNS,
    };
    const int a_shape[4] = {BATCH, 1, ROWS, K};
    const int b_shape[4] = {1, HEADS, K, COLUMNS};
    const int output_shape[4] = {BATCH, HEADS, ROWS, COLUMNS};
    const int32_t a_zero_point = 173;
    const int32_t b_zero_point = -37;
    const int32_t output_zero_point = -11;
    uint8_t a[A_ELEMENTS];
    int8_t b[B_ELEMENTS];
    int8_t output[OUTPUT_ELEMENTS];
    int8_t expected[OUTPUT_ELEMENTS];

    for (int index = 0; index < A_ELEMENTS; index++)
        a[index] = (uint8_t)(a_zero_point + (index * 7) % 5 - 2);
    for (int index = 0; index < B_ELEMENTS; index++)
        b[index] = (int8_t)(b_zero_point + (index * 11) % 5 - 2);
    for (int batch = 0; batch < BATCH; batch++) {
        for (int head = 0; head < HEADS; head++) {
            for (int row = 0; row < ROWS; row++) {
                for (int column = 0; column < COLUMNS; column++) {
                    int accumulator = 0;
                    for (int inner = 0; inner < K; inner++) {
                        int a_value = a[(batch * ROWS + row) * K + inner];
                        int b_value = b[(head * K + inner) * COLUMNS + column];
                        accumulator += (a_value - a_zero_point) *
                            (b_value - b_zero_point);
                    }
                    expected[((batch * HEADS + head) * ROWS + row) *
                        COLUMNS + column] =
                            (int8_t)(accumulator + output_zero_point);
                }
            }
        }
    }

    memset(output, 0, sizeof(output));
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qbatch_matmul_i8u8(
        a, a_shape, 4, 1.0f, a_zero_point, VX_DTYPE_U8,
        b, b_shape, 4, 1.0f, b_zero_point, VX_DTYPE_I8,
        output, output_shape, 4, 1.0f, output_zero_point,
        VX_DTYPE_I8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    vk_graph_reset();
    return 0;
}

static float gelu_reference(float x, int approximate_tanh) {
    return approximate_tanh
        ? 0.5f * x * (1.0f + tanhf(0.7978845608028654f *
            (x + 0.044715f * x * x * x)))
        : 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

static float gelu_derivative_reference(float x, int approximate_tanh) {
    if (!approximate_tanh) {
        return 0.5f * (1.0f + erff(x * 0.7071067811865475f)) +
            x * 0.3989422804014327f * expf(-0.5f * x * x);
    }
    float x2 = x * x;
    float u = 0.7978845608028654f * (x + 0.044715f * x * x2);
    float t = tanhf(u);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) *
        0.7978845608028654f * (1.0f + 0.134145f * x2);
}

static float layernorm_expected(const float* input, const float* weight,
                                const float* bias, int index, int d_model,
                                float epsilon) {
    int offset = (index / d_model) * d_model;
    float sum = 0.0f, square_sum = 0.0f;
    for (int i = 0; i < d_model; i++) {
        float value = input[offset + i];
        sum += value;
        square_sum += value * value;
    }
    float mean = sum / (float)d_model;
    float variance = square_sum / (float)d_model - mean * mean;
    int channel = index % d_model;
    return (input[index] - mean) / sqrtf(variance + epsilon) * weight[channel] +
        bias[channel];
}

static int test_layernorm_epsilon(void) {
    enum { ROWS = 2, D_MODEL = 4, ELEMENTS = ROWS * D_MODEL };
    const float input[ELEMENTS] = {
        -0.03f, 0.01f, 0.02f, 0.0f,
        1.0f, 1.002f, 0.998f, 1.001f
    };
    const float weight[D_MODEL] = {1.0f, 0.5f, 1.5f, 0.75f};
    const float bias[D_MODEL] = {0.1f, -0.2f, 0.3f, -0.4f};
    const float epsilons[2] = {0.25f, 1.0e-4f};
    float output[2][ELEMENTS] = {{0}};

    for (int pass = 0; pass < 2; pass++) {
        vk_graph_reset();
        vk_graph_begin_forward();
        CHECK(vk_graph_layernorm_f32(input, weight, bias, output[pass],
                                     ROWS, D_MODEL, epsilons[pass]) == 1);
        CHECK(vk_graph_end_forward() == 0);
        CHECK(vk_graph_sync_host(output[pass], sizeof(output[pass]), 0) == 1);
        for (int i = 0; i < ELEMENTS; i++) {
            float expected = layernorm_expected(input, weight, bias, i, D_MODEL,
                                                epsilons[pass]);
            CHECK(isfinite(output[pass][i]));
            CHECK(fabsf(output[pass][i] - expected) < 5.0e-4f);
        }
    }
    CHECK(fabsf(output[0][0] - output[1][0]) > 0.5f);
    CHECK(vk_graph_layernorm_f32(input, weight, bias, output[0],
                                 ROWS, D_MODEL, 0.0f) == 0);
    CHECK(vk_graph_layernorm_f32(input, weight, bias, output[0],
                                 ROWS, D_MODEL, NAN) == 0);
    vk_graph_reset();
    return 0;
}

static uint32_t dropout_bits(uint32_t seed, uint32_t counter, uint32_t index) {
    uint32_t value = seed ^ index ^ counter * 0x9e3779b9u;
    value = (value ^ (value >> 16u)) * 0x7feb352du;
    value = (value ^ (value >> 15u)) * 0x846ca68bu;
    return value ^ (value >> 16u);
}

static int test_activation_backward(void) {
    float input[2] = {-1.0f, 2.0f};
    float output[2] = {0.0f, 2.0f};
    float grad_output[2] = {3.0f, 4.0f};
    float grad_input[2] = {0.0f, 0.0f};
    struct {
        uint32_t length;
        uint32_t kind;
        float alpha;
        float beta;
    } params = {2u, 0u, 0.0f, 0.0f};
    void* hosts[5] = {input, output, grad_output, grad_input, &params};
    size_t bytes[5] = {sizeof(input), sizeof(output), sizeof(grad_output),
                       sizeof(grad_input), sizeof(params)};
    unsigned char access[5] = {
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    unsigned char weights[5] = {0, 0, 0, 0, 0};
    CHECK(vk_training_dispatch("activationBackward", "main", hosts, bytes,
                               access, weights, 5, 1, 1, 1) == 0);
    CHECK(vk_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(close_enough(grad_input[0], 0.0f));
    CHECK(close_enough(grad_input[1], 4.0f));
    return 0;
}

static int test_gelu_backward_modes(void) {
    float input[7] = {-3.0f, -1.0f, -0.25f, 0.0f, 0.5f, 2.0f, 4.0f};
    float exact_output[7], tanh_output[7], grad_output[7];
    float exact_grad[7] = {0}, tanh_grad[7] = {0};
    for (int i = 0; i < 7; i++) {
        exact_output[i] = gelu_reference(input[i], 0);
        tanh_output[i] = gelu_reference(input[i], 1);
        grad_output[i] = 0.25f * (float)(i + 1);
    }
    uint32_t exact_params[4] = {7u, 1u, 0u, 0u};
    uint32_t tanh_params[4] = {7u, 9u, 0u, 0u};
    void* exact_hosts[5] = {input, exact_output, grad_output, exact_grad, exact_params};
    void* tanh_hosts[5] = {input, tanh_output, grad_output, tanh_grad, tanh_params};
    size_t exact_bytes[5] = {sizeof(input), sizeof(exact_output), sizeof(grad_output),
                             sizeof(exact_grad), sizeof(exact_params)};
    size_t tanh_bytes[5] = {sizeof(input), sizeof(tanh_output), sizeof(grad_output),
                            sizeof(tanh_grad), sizeof(tanh_params)};
    unsigned char access[5] = {
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    CHECK(vk_training_dispatch("activationBackward", "main", exact_hosts, exact_bytes,
                               access, NULL, 5, 1, 1, 1) == 0);
    CHECK(vk_training_dispatch("activationBackward", "main", tanh_hosts, tanh_bytes,
                               access, NULL, 5, 1, 1, 1) == 0);
    CHECK(vk_training_sync(exact_grad, sizeof(exact_grad)) == 0);
    CHECK(vk_training_sync(tanh_grad, sizeof(tanh_grad)) == 0);
    for (int i = 0; i < 7; i++) {
        CHECK(fabsf(exact_grad[i] - grad_output[i] * gelu_derivative_reference(input[i], 0)) < 2.0e-4f);
        CHECK(fabsf(tanh_grad[i] - grad_output[i] * gelu_derivative_reference(input[i], 1)) < 2.0e-4f);
    }
    return 0;
}

static int test_gelu_forward_modes(void) {
    float input[7] = {-3.0f, -1.0f, -0.25f, 0.0f, 0.5f, 2.0f, 4.0f};
    float exact_output[7] = {0}, tanh_output[7] = {0};
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_gelu_f32(input, exact_output, 7, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(exact_output, sizeof(exact_output), 0) == 1);
    vk_graph_begin_forward();
    CHECK(vk_graph_gelu_f32(input, tanh_output, 7, 1) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(tanh_output, sizeof(tanh_output), 0) == 1);
    for (int i = 0; i < 7; i++) {
        CHECK(fabsf(exact_output[i] - gelu_reference(input[i], 0)) < 2.0e-4f);
        CHECK(fabsf(tanh_output[i] - gelu_reference(input[i], 1)) < 2.0e-4f);
    }
    CHECK(fabsf(exact_output[0] - tanh_output[0]) > 1.0e-5f);
    vk_graph_reset();
    return 0;
}

static int test_qlinear_i8u8_packed_chain(void) {
    /* Every tensor here has a non-word-aligned logical byte length.  The
       second dispatch consumes `hidden` while it is still device-resident. */
    const int8_t input[3] = {2, -2, 4};
    const int8_t first_weight[6] = {1, 2, -1, -2, 1, 3};
    const float first_scales[2] = {0.25f, 0.5f};
    const int32_t first_zero_points[2] = {0, 0};
    const int32_t first_bias[2] = {0, 0};
    uint8_t hidden[2] = {0, 0};
    const int8_t second_weight[2] = {2, -2};
    const float second_scales[1] = {0.5f};
    const int32_t second_zero_points[1] = {0};
    const int32_t second_bias[1] = {0};
    int8_t output[1] = {0};

    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qlinear_i8u8(input, first_weight, first_scales,
                                first_zero_points, first_bias, hidden,
                                1u, 3u, 2u, 0.5f, 0, 0.25f, 128,
                                VX_DTYPE_I8, VX_DTYPE_I8,
                                VX_DTYPE_U8) == 1);
    CHECK(vk_graph_qlinear_i8u8(hidden, second_weight, second_scales,
                                second_zero_points, second_bias, output,
                                1u, 2u, 1u, 0.25f, 128, 0.25f, -3,
                                VX_DTYPE_U8, VX_DTYPE_I8,
                                VX_DTYPE_I8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(hidden, sizeof(hidden), 0) == 1);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    CHECK(hidden[0] == 125u && hidden[1] == 134u);
    CHECK(output[0] == -12);
    vk_graph_reset();
    return 0;
}

/* The portable qLinear shader fuses the first four channels of each packed
 * word. D_OUT=5 makes the second word cross a row and use qlinear_one, while
 * K=5 exercises the non-word-aligned reduction with asymmetric U8/I8
 * zero-points. Disable optional dot kernels so both paths are deterministic. */
static int test_qlinear_i8u8_scalar_fused_tail(void) {
    enum { ROWS = 2, D_IN = 5, D_OUT = 5 };
    const int32_t input_zero_point = 131;
    const int32_t output_zero_point = -9;
    uint8_t input[ROWS * D_IN];
    int8_t weight[D_OUT * D_IN];
    float scales[D_OUT];
    int32_t zero_points[D_OUT];
    int32_t bias[D_OUT];
    int8_t output[ROWS * D_OUT];
    int8_t expected[ROWS * D_OUT];
    char telemetry[128] = {0};

    for (int index = 0; index < ROWS * D_IN; index++)
        input[index] = (uint8_t)(input_zero_point + (index * 3) % 7 - 3);
    for (int channel = 0; channel < D_OUT; channel++) {
        scales[channel] = 1.0f;
        zero_points[channel] = -5 + channel;
        bias[channel] = channel - 2;
        for (int k = 0; k < D_IN; k++)
            weight[channel * D_IN + k] = (int8_t)(
                zero_points[channel] + (channel * 2 + k * 3) % 7 - 3);
    }
    for (int row = 0; row < ROWS; row++) {
        for (int channel = 0; channel < D_OUT; channel++) {
            int32_t accumulator = bias[channel];
            for (int k = 0; k < D_IN; k++) {
                accumulator +=
                    ((int32_t)input[row * D_IN + k] - input_zero_point) *
                    ((int32_t)weight[channel * D_IN + k] -
                     zero_points[channel]);
            }
            accumulator += output_zero_point;
            if (accumulator < -128) accumulator = -128;
            if (accumulator > 127) accumulator = 127;
            expected[row * D_OUT + channel] = (int8_t)accumulator;
        }
    }

    vk_graph_reset();
    const int previous_dot_disabled = vk_test_set_packed_dot_disabled(1);
    CHECK(previous_dot_disabled >= 0);
    vk_graph_begin_forward();
    CHECK(vk_graph_qlinear_i8u8(
              input, weight, scales, zero_points, bias, output,
              ROWS, D_IN, D_OUT, 0.5f, input_zero_point, 0.5f,
              output_zero_point, VX_DTYPE_U8, VX_DTYPE_I8,
              VX_DTYPE_I8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    CHECK(vk_graph_append_dynamic_telemetry(
              telemetry, sizeof(telemetry)) == 0);
    CHECK(strstr(telemetry, ";vk_qls=1") != NULL &&
          strstr(telemetry, "vk_qld=") == NULL &&
          strstr(telemetry, "vk_qlt=") == NULL);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    CHECK(vk_test_set_packed_dot_disabled(previous_dot_disabled) == 1);
    vk_graph_reset();
    return 0;
}

static int test_qembedding_i8u8_packed_gather(void) {
    /* Nine logical output bytes exercise the packed-word tail while both
       table dtypes and output dtypes use the canonical row descriptor ABI. */
    const int32_t ids[3] = {1, 0, 2};
    const int32_t invalid_ids[3] = {1, 3, 0};
    const float scales[3] = {0.5f, 0.25f, 1.0f};
    const int32_t i8_zero_points[3] = {0, 1, -1};
    const int32_t u8_zero_points[3] = {128, 129, 127};
    const int8_t i8_table[9] = {0, 1, -1, 2, -2, 4, 99, -101, 4};
    const uint8_t u8_table[9] = {128, 129, 127, 130, 126, 132, 227, 27, 132};
    const uint8_t expected_u8[9] = {129, 125, 131, 128, 130, 126, 255, 0, 148};
    const int8_t expected_i8[9] = {-2, -6, 0, -3, -1, -5, 127, -128, 17};
    uint8_t output_u8[9] = {0};
    int8_t output_i8[9] = {0};

    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qembedding_i8u8(ids, i8_table, scales, i8_zero_points, output_u8,
                                   3u, 3u, 3u, 0.25f, 128,
                                   VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output_u8, sizeof(output_u8), 0) == 1);
    for (int index = 0; index < 9; index++) CHECK(output_u8[index] == expected_u8[index]);

    vk_graph_begin_forward();
    CHECK(vk_graph_qembedding_i8u8(ids, u8_table, scales, u8_zero_points, output_i8,
                                   3u, 3u, 3u, 0.25f, -3,
                                   VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output_i8, sizeof(output_i8), 0) == 1);
    for (int index = 0; index < 9; index++) CHECK(output_i8[index] == expected_i8[index]);

    memset(output_u8, 0x5a, sizeof(output_u8));
    CHECK(vk_graph_qembedding_i8u8(invalid_ids, i8_table, scales, i8_zero_points,
                                   output_u8, 3u, 3u, 3u, 0.25f, 128,
                                   VX_DTYPE_I8, VX_DTYPE_U8) == 0);
    for (int index = 0; index < 9; index++) CHECK(output_u8[index] == 0x5a);
    vk_graph_reset();
    return 0;
}

static int test_qadd_requantize_i8u8_packed_chain(void) {
    /* Five logical bytes force a padded second u32 word.  `sum` remains on
       the device while RequantizeLinear changes its U8 output domain. */
    const int8_t a[5] = {-2, 1, 13, 0, 30};
    const uint8_t b[5] = {124, 128, 140, 129, 128};
    int8_t sum[5] = {0, 0, 0, 0, 0};
    uint8_t output[5] = {0, 0, 0, 0, 0};

    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qadd_i8u8(a, 5u, b, 5u, sum, 5u,
                              0.5f, 0, 0.25f, 128,
                              0.5f, -2, VX_DTYPE_I8, VX_DTYPE_U8,
                              VX_DTYPE_I8, 2u) == 1);
    CHECK(vk_graph_requantize_linear_i8u8(sum, 5u, output, 5u,
                                          0.5f, -2, 1.0f, 100,
                                          VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(sum, sizeof(sum), 0) == 1);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    const int8_t expected_sum[5] = {-2, -1, 10, -2, 10};
    const uint8_t expected_output[5] = {100, 100, 106, 100, 106};
    for (int i = 0; i < 5; i++) {
        CHECK(sum[i] == expected_sum[i]);
        CHECK(output[i] == expected_output[i]);
    }
    /* Exact-shape and relu ABI validation must fail before a dispatch. */
    CHECK(vk_graph_qadd_i8u8(a, 5u, b, 4u, sum, 5u,
                              0.5f, 0, 0.25f, 128,
                              0.5f, -2, VX_DTYPE_I8, VX_DTYPE_U8,
                              VX_DTYPE_I8, 2u) == 0);
    CHECK(vk_graph_qadd_i8u8(a, 5u, b, 5u, sum, 5u,
                              0.5f, 0, 0.25f, 128,
                              0.5f, -2, VX_DTYPE_I8, VX_DTYPE_U8,
                              VX_DTYPE_I8, 3u) == 0);
    CHECK(vk_graph_requantize_linear_i8u8(sum, 5u, output, 4u,
                                          0.5f, -2, 1.0f, 100,
                                          VX_DTYPE_I8, VX_DTYPE_U8) == 0);
    vk_graph_reset();
    return 0;
}

/* Five elements require a padded packed-u32 tail. The four direct dispatches
 * cover every I8/U8 boundary pair, and the final pair proves QSiLU can consume
 * its predecessor while both byte tensors remain device-resident. */
static int test_qsilu_i8u8_packed_chain(void) {
    const int8_t input_i8[5] = {-8, -2, 0, 2, 8};
    const uint8_t input_u8[5] = {120, 126, 128, 130, 136};
    const int8_t expected_i8[5] = {-3, -4, -3, 0, 13};
    const uint8_t expected_u8[5] = {128, 127, 128, 131, 144};
    int8_t from_i8_i8[5] = {0};
    int8_t from_u8_i8[5] = {0};
    uint8_t from_i8_u8[5] = {0};
    uint8_t from_u8_u8[5] = {0};
    int8_t chained[5] = {0};
    int8_t alias[5] = {-8, -2, 0, 2, 8};

    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qsilu_i8u8(input_i8, from_i8_i8, 5u,
                               0.5f, 0, 0.25f, -3,
                               VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_qsilu_i8u8(input_i8, from_i8_u8, 5u,
                               0.5f, 0, 0.25f, 128,
                               VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(vk_graph_qsilu_i8u8(input_u8, from_u8_i8, 5u,
                               0.5f, 128, 0.25f, -3,
                               VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_qsilu_i8u8(input_u8, from_u8_u8, 5u,
                               0.5f, 128, 0.25f, 128,
                               VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(vk_graph_qsilu_i8u8(from_i8_u8, chained, 5u,
                               0.25f, 128, 0.125f, -4,
                               VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(from_i8_i8, sizeof(from_i8_i8), 0) == 1);
    CHECK(vk_graph_sync_host(from_i8_u8, sizeof(from_i8_u8), 0) == 1);
    CHECK(vk_graph_sync_host(from_u8_i8, sizeof(from_u8_i8), 0) == 1);
    CHECK(vk_graph_sync_host(from_u8_u8, sizeof(from_u8_u8), 0) == 1);
    CHECK(vk_graph_sync_host(chained, sizeof(chained), 0) == 1);
    CHECK(memcmp(from_i8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_i8_u8, expected_u8, sizeof(expected_u8)) == 0);
    CHECK(memcmp(from_u8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_u8_u8, expected_u8, sizeof(expected_u8)) == 0);
    {
        const int8_t expected_chained[5] = {-4, -5, -4, 0, 27};
        CHECK(memcmp(chained, expected_chained, sizeof(expected_chained)) == 0);
    }
    CHECK(vk_graph_qsilu_i8u8(input_i8, from_i8_i8, 0u,
                               0.5f, 0, 0.25f, -3,
                               VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(vk_graph_qsilu_i8u8(alias, alias, 5u,
                               0.5f, 0, 0.25f, -3,
                               VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    vk_graph_reset();
    return 0;
}

/* QGELU uses the fixed Abramowitz-Stegun erf route.  Keep the same
 * non-word-aligned shape and all I8/U8 boundaries as QSiLU, then consume the
 * U8 result while it is still packed in Vulkan graph storage. */
static int test_qgelu_i8u8_packed_chain(void) {
    const int8_t input_i8[5] = {-8, -2, 0, 2, 8};
    const uint8_t input_u8[5] = {120, 126, 128, 130, 136};
    const int8_t expected_i8[5] = {-3, -4, -3, 4, 29};
    const uint8_t expected_u8[5] = {128, 127, 128, 135, 160};
    int8_t from_i8_i8[5] = {0};
    int8_t from_u8_i8[5] = {0};
    uint8_t from_i8_u8[5] = {0};
    uint8_t from_u8_u8[5] = {0};
    int8_t chained[5] = {0};
    int8_t alias[5] = {-8, -2, 0, 2, 8};

    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qgelu_i8u8(input_i8, from_i8_i8, 5u,
                               0.5f, 0, 0.125f, -3,
                               VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_qgelu_i8u8(input_i8, from_i8_u8, 5u,
                               0.5f, 0, 0.125f, 128,
                               VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(vk_graph_qgelu_i8u8(input_u8, from_u8_i8, 5u,
                               0.5f, 128, 0.125f, -3,
                               VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_qgelu_i8u8(input_u8, from_u8_u8, 5u,
                               0.5f, 128, 0.125f, 128,
                               VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(vk_graph_qgelu_i8u8(from_i8_u8, chained, 5u,
                               0.125f, 128, 0.125f, -4,
                               VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(from_i8_i8, sizeof(from_i8_i8), 0) == 1);
    CHECK(vk_graph_sync_host(from_i8_u8, sizeof(from_i8_u8), 0) == 1);
    CHECK(vk_graph_sync_host(from_u8_i8, sizeof(from_u8_i8), 0) == 1);
    CHECK(vk_graph_sync_host(from_u8_u8, sizeof(from_u8_u8), 0) == 1);
    CHECK(vk_graph_sync_host(chained, sizeof(chained), 0) == 1);
    CHECK(memcmp(from_i8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_i8_u8, expected_u8, sizeof(expected_u8)) == 0);
    CHECK(memcmp(from_u8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_u8_u8, expected_u8, sizeof(expected_u8)) == 0);
    {
        const int8_t expected_chained[5] = {-4, -4, -4, 2, 28};
        CHECK(memcmp(chained, expected_chained, sizeof(expected_chained)) == 0);
    }
    CHECK(vk_graph_qgelu_i8u8(input_i8, from_i8_i8, 0u,
                               0.5f, 0, 0.125f, -3,
                               VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(vk_graph_qgelu_i8u8(alias, alias, 5u,
                               0.5f, 0, 0.125f, -3,
                               VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    vk_graph_reset();
    return 0;
}

/* QGroupNorm's stats/apply split must preserve raw bytes across every typed
 * boundary, including C=3 tails and C/G=3 boundaries inside packed words. */
static int test_qgroupnorm_i8u8_packed_chain(void) {
    const int8_t input_i8[3] = {-5, 0, 4};
    const uint8_t input_u8[3] = {123, 128, 132};
    const float gamma[3] = {1.0f, 0.5f, -0.75f};
    const float beta[3] = {0.25f, -0.5f, 0.75f};
    const float zero_gamma[3] = {0.0f, 0.0f, 0.0f};
    const int8_t expected_i8[3] = {-11, -7, -4};
    const uint8_t expected_u8[3] = {120, 124, 127};
    int8_t from_i8_i8[3] = {0};
    int8_t from_u8_i8[3] = {0};
    uint8_t from_i8_u8[3] = {0};
    uint8_t from_u8_u8[3] = {0};
    int8_t chained[3] = {0};
    int8_t alias[3] = {-5, 0, 4};
    const int8_t boundary_input[18] = {
        -8, -3, 4, 7, 5, -1, 2, -6, 0, 8, -4, 3, -7, 6, 1, -2, 9, -5,
    };
    const int8_t boundary_expected[18] = {
        -10, -6, 8, 5, -1, -12, 7, 1, 5, 7, 6, -9, -11, -14, 4, -9, -5, -11,
    };
    const float boundary_gamma[6] = {1.0f, -0.75f, 0.5f, 1.25f, -0.5f, 0.25f};
    const float boundary_beta[6] = {0.25f, -0.5f, 0.75f, -0.25f, 0.5f, -0.75f};
    int8_t boundary_output[18] = {0};
    enum { high_elements = 3 * 57 * 6 };
    uint8_t high_input[high_elements];
    uint8_t high_output[high_elements] = {0};
    const float high_gamma[6] = {1, 1, 1, 1, 1, 1};
    const float high_beta[6] = {0, 0, 0, 0, 0, 0};

    for (int index = 0; index < high_elements; index++)
        high_input[index] = (uint8_t)(index & 1 ? 241 : 240);
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qgroupnorm_i8u8(input_i8, gamma, beta, from_i8_i8,
                                    1u, 1u, 1u, 3u, 1u, 0.5f, -1,
                                    0.125f, -3, 1.0e-5f,
                                    VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_qgroupnorm_i8u8(input_i8, gamma, beta, from_i8_u8,
                                    1u, 1u, 1u, 3u, 1u, 0.5f, -1,
                                    0.125f, 128, 1.0e-5f,
                                    VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(vk_graph_qgroupnorm_i8u8(input_u8, gamma, beta, from_u8_i8,
                                    1u, 1u, 1u, 3u, 1u, 0.5f, 127,
                                    0.125f, -3, 1.0e-5f,
                                    VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_qgroupnorm_i8u8(input_u8, gamma, beta, from_u8_u8,
                                    1u, 1u, 1u, 3u, 1u, 0.5f, 127,
                                    0.125f, 128, 1.0e-5f,
                                    VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    /* The second pass consumes packed U8 output without a host sync. */
    CHECK(vk_graph_qgroupnorm_i8u8(from_i8_u8, zero_gamma, beta, chained,
                                    1u, 1u, 1u, 3u, 1u, 0.125f, 128,
                                    0.125f, -4, 1.0e-5f,
                                    VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_qgroupnorm_i8u8(boundary_input, boundary_gamma, boundary_beta,
                                    boundary_output, 3u, 1u, 1u, 6u, 2u,
                                    0.25f, -1, 0.125f, -3, 1.0e-5f,
                                    VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_qgroupnorm_i8u8(high_input, high_gamma, high_beta, high_output,
                                    3u, 57u, 1u, 6u, 2u, 0.5f, 17,
                                    0.25f, 128, 1.0e-5f,
                                    VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(from_i8_i8, sizeof(from_i8_i8), 0) == 1);
    CHECK(vk_graph_sync_host(from_i8_u8, sizeof(from_i8_u8), 0) == 1);
    CHECK(vk_graph_sync_host(from_u8_i8, sizeof(from_u8_i8), 0) == 1);
    CHECK(vk_graph_sync_host(from_u8_u8, sizeof(from_u8_u8), 0) == 1);
    CHECK(vk_graph_sync_host(chained, sizeof(chained), 0) == 1);
    CHECK(vk_graph_sync_host(boundary_output, sizeof(boundary_output), 0) == 1);
    CHECK(vk_graph_sync_host(high_output, sizeof(high_output), 0) == 1);
    CHECK(memcmp(from_i8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_i8_u8, expected_u8, sizeof(expected_u8)) == 0);
    CHECK(memcmp(from_u8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_u8_u8, expected_u8, sizeof(expected_u8)) == 0);
    {
        const int8_t expected_chained[3] = {-2, -8, 2};
        CHECK(memcmp(chained, expected_chained, sizeof(expected_chained)) == 0);
    }
    CHECK(memcmp(boundary_output, boundary_expected, sizeof(boundary_expected)) == 0);
    for (int index = 0; index < high_elements; index += 6) {
        const uint8_t expected[6] = {125, 134, 125, 131, 122, 131};
        CHECK(memcmp(high_output + index, expected, sizeof(expected)) == 0);
    }
    CHECK(vk_graph_qgroupnorm_i8u8(input_i8, gamma, beta, from_i8_i8,
                                    1u, 1u, 1u, 3u, 0u, 0.5f, -1,
                                    0.125f, -3, 1.0e-5f,
                                    VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(vk_graph_qgroupnorm_i8u8(alias, gamma, beta, alias,
                                    1u, 1u, 1u, 3u, 1u, 0.5f, -1,
                                    0.125f, -3, 1.0e-5f,
                                    VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    vk_graph_reset();
    return 0;
}

/* D=3 deliberately splits two final-axis rows across a packed word. The
 * stats/apply split must still give every row its own centered byte variance. */
static int test_qlayernorm_i8u8_packed_chain(void) {
    const int8_t input_i8[6] = {-5, 0, 4, 6, -2, 1};
    const uint8_t input_u8[6] = {123, 128, 132, 134, 126, 129};
    const float gamma[3] = {1.0f, 0.5f, -0.75f};
    const float beta[3] = {0.25f, -0.5f, 0.75f};
    const float zero_gamma[3] = {0.0f, 0.0f, 0.0f};
    const int8_t expected_i8[6] = {-11, -7, -4, 10, -11, 4};
    const uint8_t expected_u8[6] = {120, 124, 127, 141, 120, 135};
    int8_t from_i8_i8[6] = {0};
    int8_t from_u8_i8[6] = {0};
    uint8_t from_i8_u8[6] = {0};
    uint8_t from_u8_u8[6] = {0};
    int8_t chained[6] = {0};
    int8_t alias[6] = {-5, 0, 4, 6, -2, 1};
    enum { high_d_model = 1024, high_elements = 2 * high_d_model };
    uint8_t high_input[high_elements];
    uint8_t high_output[high_elements] = {0};
    float high_gamma[high_d_model];
    float high_beta[high_d_model];

    for (int index = 0; index < high_elements; index++)
        high_input[index] = (uint8_t)(index & 1 ? 241 : 240);
    for (int channel = 0; channel < high_d_model; channel++) {
        high_gamma[channel] = 1.0f;
        high_beta[channel] = 0.0f;
    }
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qlayernorm_i8u8(input_i8, gamma, beta, from_i8_i8,
                                    2u, 3u, 0.5f, -1, 0.125f, -3,
                                    1.0e-5f, VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_qlayernorm_i8u8(input_i8, gamma, beta, from_i8_u8,
                                    2u, 3u, 0.5f, -1, 0.125f, 128,
                                    1.0e-5f, VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(vk_graph_qlayernorm_i8u8(input_u8, gamma, beta, from_u8_i8,
                                    2u, 3u, 0.5f, 127, 0.125f, -3,
                                    1.0e-5f, VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_qlayernorm_i8u8(input_u8, gamma, beta, from_u8_u8,
                                    2u, 3u, 0.5f, 127, 0.125f, 128,
                                    1.0e-5f, VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(vk_graph_qlayernorm_i8u8(from_i8_u8, zero_gamma, beta, chained,
                                    2u, 3u, 0.125f, 128, 0.125f, -4,
                                    1.0e-5f, VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_qlayernorm_i8u8(high_input, high_gamma, high_beta, high_output,
                                    2u, high_d_model, 0.5f, 17, 0.25f, 128,
                                    1.0e-5f, VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(from_i8_i8, sizeof(from_i8_i8), 0) == 1);
    CHECK(vk_graph_sync_host(from_i8_u8, sizeof(from_i8_u8), 0) == 1);
    CHECK(vk_graph_sync_host(from_u8_i8, sizeof(from_u8_i8), 0) == 1);
    CHECK(vk_graph_sync_host(from_u8_u8, sizeof(from_u8_u8), 0) == 1);
    CHECK(vk_graph_sync_host(chained, sizeof(chained), 0) == 1);
    CHECK(vk_graph_sync_host(high_output, sizeof(high_output), 0) == 1);
    CHECK(memcmp(from_i8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_i8_u8, expected_u8, sizeof(expected_u8)) == 0);
    CHECK(memcmp(from_u8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_u8_u8, expected_u8, sizeof(expected_u8)) == 0);
    {
        const int8_t expected_chained[6] = {-2, -8, 2, -2, -8, 2};
        CHECK(memcmp(chained, expected_chained, sizeof(expected_chained)) == 0);
    }
    for (int index = 0; index < high_elements; index++)
        CHECK(high_output[index] == (uint8_t)(index & 1 ? 132 : 124));
    CHECK(vk_graph_qlayernorm_i8u8(input_i8, gamma, beta, from_i8_i8,
                                    2u, 0u, 0.5f, -1, 0.125f, -3,
                                    1.0e-5f, VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(vk_graph_qlayernorm_i8u8(alias, gamma, beta, alias,
                                    2u, 3u, 0.5f, -1, 0.125f, -3,
                                    1.0e-5f, VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    vk_graph_reset();
    return 0;
}

/* qSDPAInt8 has one workgroup per [query, head, batch]. This verifies an
 * I8->U8->I8 device-resident chain, ordinary/no-mask dummy binding, and the
 * all-masked U8 zero-point result without allocating a score buffer. */
static int test_qsdpa_i8u8_packed_chain(void) {
    const int8_t q_i8[8] = {0, -1, -2, 1, -2, 1, 0, -1};
    const int8_t k_i8[8] = {0, 0, -1, -1, -1, 0, 0, -2};
    const int8_t v_i8[8] = {3, -3, 0, -1, -5, 1, 2, -2};
    const uint8_t q_u8[4] = {129, 126, 131, 128};
    const uint8_t k_u8[8] = {121, 120, 119, 122, 119, 123, 120, 118};
    const uint8_t v_u8[8] = {134, 126, 132, 130, 128, 132, 136, 128};
    const int32_t mask_none[2] = {0, 0};
    const uint8_t expected_first[8] = {128, 128, 130, 128, 128, 128, 130, 127};
    const int8_t expected_second[8] = {0, 0, 2, -1, 0, 0, 2, -1};
    const uint8_t expected_zero[4] = {127, 127, 127, 127};
    uint8_t first[8] = {0};
    int8_t second[8] = {0};
    uint8_t all_masked[4] = {0};
    int8_t alias[8] = {0, -1, -2, 1, -2, 1, 0, -1};
    int8_t alias_before[8];
    enum { head_dim = 64 };
    int8_t q64[head_dim] = {0};
    int8_t k64[head_dim] = {0};
    int8_t v64[head_dim];
    int8_t out64[head_dim] = {0};

    for (int channel = 0; channel < head_dim; channel++)
        v64[channel] = (int8_t)((channel % 15) - 7);
    memcpy(alias_before, alias, sizeof(alias));
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qsdpa_i8u8(q_i8, k_i8, v_i8, NULL, first, 1u, 2u, 2u,
                               4u, 1u, 0.25f, -1, 0.25f, -1, 0.25f, -1,
                               0.25f, 128, 0.5f,
                               VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                               VX_DTYPE_U8, 0u, 0u) == 1);
    CHECK(vk_graph_qsdpa_i8u8(first, k_i8, v_i8, NULL, second, 1u, 2u, 2u,
                               4u, 1u, 0.25f, 128, 0.25f, -1, 0.25f, -1,
                               0.25f, 0, 0.5f,
                               VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8,
                               VX_DTYPE_I8, 0u, 0u) == 1);
    CHECK(vk_graph_qsdpa_i8u8(q_u8, k_u8, v_u8, mask_none, all_masked, 1u,
                               1u, 2u, 4u, 1u, 0.25f, 128, 0.5f, 120,
                               0.25f, 130, 0.25f, 127, 0.5f,
                               VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8,
                               VX_DTYPE_U8, 0u, 1u) == 1);
    CHECK(vk_graph_qsdpa_i8u8(q64, k64, v64, NULL, out64, 1u, 1u, 1u,
                               head_dim, 1u, 0.25f, 0, 0.25f, 0, 0.25f, 0,
                               0.25f, 0, 1.0f,
                               VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                               VX_DTYPE_I8, 0u, 0u) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(first, sizeof(first), 0) == 1);
    CHECK(vk_graph_sync_host(second, sizeof(second), 0) == 1);
    CHECK(vk_graph_sync_host(all_masked, sizeof(all_masked), 0) == 1);
    CHECK(vk_graph_sync_host(out64, sizeof(out64), 0) == 1);
    CHECK(memcmp(first, expected_first, sizeof(first)) == 0);
    CHECK(memcmp(second, expected_second, sizeof(second)) == 0);
    CHECK(memcmp(all_masked, expected_zero, sizeof(all_masked)) == 0);
    CHECK(memcmp(out64, v64, sizeof(out64)) == 0);
    CHECK(vk_graph_qsdpa_i8u8(alias, k_i8, v_i8, NULL, alias, 1u, 2u, 2u,
                               4u, 1u, 0.25f, -1, 0.25f, -1, 0.25f, -1,
                               0.25f, 0, 0.5f,
                               VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                               VX_DTYPE_I8, 0u, 0u) == 0);
    CHECK(memcmp(alias, alias_before, sizeof(alias)) == 0);
    vk_graph_reset();
    return 0;
}

/* qArgMaxInt8 reads packed byte storage directly and emits conventional I32
 * indices. This includes the non-word-aligned I8 input tail, signed negative
 * ordering, U8 order with an asymmetric zero point, and first-tie behavior. */
static int test_qargmax_i8u8_raw(void) {
    const int8_t input_i8[12] = {
        -5, 4, -1, 4, -1, 3,
        -128, 0, -127, 1, -126, 1,
    };
    const int32_t expected_i8[4] = {1, 0, 2, 1};
    const uint8_t input_u8[6] = {130, 3, 129, 255, 130, 254};
    const int32_t expected_u8[2] = {0, 1};
    int32_t output_i8[4] = {0};
    int32_t output_u8[2] = {0};
    int32_t sentinel[2] = {17, 23};
    const int32_t sentinel_before[2] = {17, 23};
    union {
        int32_t i32[4];
        uint8_t bytes[16];
    } alias;
    uint8_t alias_before[sizeof(alias.bytes)];

    memset(alias.bytes, 0x5a, sizeof(alias.bytes));
    memcpy(alias.bytes, input_i8, 6u);
    memcpy(alias_before, alias.bytes, sizeof(alias_before));
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qargmax_i8u8(input_i8, output_i8, 2u, 3u, 2u,
                                VX_DTYPE_I8) == 1);
    CHECK(vk_graph_qargmax_i8u8(input_u8, output_u8, 1u, 3u, 2u,
                                VX_DTYPE_U8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output_i8, sizeof(output_i8), 0) == 1);
    CHECK(vk_graph_sync_host(output_u8, sizeof(output_u8), 0) == 1);
    CHECK(memcmp(output_i8, expected_i8, sizeof(output_i8)) == 0);
    CHECK(memcmp(output_u8, expected_u8, sizeof(output_u8)) == 0);
    CHECK(vk_graph_qargmax_i8u8(alias.bytes, alias.i32, 1u, 3u, 2u,
                                VX_DTYPE_I8) == 0);
    CHECK(memcmp(alias.bytes, alias_before, sizeof(alias.bytes)) == 0);
    CHECK(vk_graph_qargmax_i8u8(input_i8, sentinel, 1u, 0u, 2u,
                                VX_DTYPE_I8) == 0);
    CHECK(memcmp(sentinel, sentinel_before, sizeof(sentinel)) == 0);
    vk_graph_reset();
    return 0;
}

/* qMaskedMeanInt8 owns packed output words (including the U8 tail here),
 * averages centered raw bytes only for nonzero I32 mask entries, and keeps
 * the all-masked row at the declared output zero point. */
static int test_qmaskedmean_i8u8_packed(void) {
    const int8_t input_i8[24] = {
        -3, -1,  1,  1,
         1, -5, -3, -1,
        99, 98, 97, 96,
        40, 41, 42, 43,
        44, 45, 46, 47,
        48, 49, 50, 51,
    };
    const int32_t mask_i8[6] = {1, 1, 0, 0, 0, 0};
    const int8_t expected_i8[8] = {6, 5, 6, 6, 5, 5, 5, 5};
    const uint8_t input_u8[6] = {130, 134, 20, 20, 131, 129};
    const int32_t mask_u8[3] = {1, 0, 1};
    const uint8_t expected_u8[2] = {132, 134};
    int8_t output_i8[8] = {0};
    uint8_t output_u8[2] = {0};
    int8_t staged_input[127];
    int32_t staged_mask[127];
    int8_t staged_output = 0;
    union {
        int8_t input[24];
        uint8_t bytes[24];
    } alias;
    union {
        int32_t mask[6];
        uint8_t bytes[24];
    } mask_alias;
    uint8_t alias_before[sizeof(alias.bytes)];
    uint8_t mask_before[sizeof(mask_alias.bytes)];

    memcpy(alias.input, input_i8, sizeof(input_i8));
    memcpy(alias_before, alias.bytes, sizeof(alias_before));
    memcpy(mask_alias.mask, mask_i8, sizeof(mask_i8));
    memcpy(mask_before, mask_alias.bytes, sizeof(mask_before));
    for (uint32_t index = 0; index < 127u; index++) {
        staged_input[index] = -128;
        staged_mask[index] = 1;
    }
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qmaskedmean_i8u8(input_i8, mask_i8, output_i8,
                                     2u, 3u, 4u, 0.25f, -3, 0.5f, 5,
                                     VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_qmaskedmean_i8u8(input_u8, mask_u8, output_u8,
                                     1u, 3u, 2u, 0.25f, 128, 0.25f, 130,
                                     VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(vk_graph_qmaskedmean_i8u8(staged_input, staged_mask, &staged_output,
                                     1u, 127u, 1u, 0.001f, 127, 0.01f, -37,
                                     VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output_i8, sizeof(output_i8), 0) == 1);
    CHECK(vk_graph_sync_host(output_u8, sizeof(output_u8), 0) == 1);
    CHECK(vk_graph_sync_host(&staged_output, sizeof(staged_output), 0) == 1);
    CHECK(memcmp(output_i8, expected_i8, sizeof(output_i8)) == 0);
    CHECK(memcmp(output_u8, expected_u8, sizeof(output_u8)) == 0);
    CHECK(staged_output == -62);
    CHECK(vk_graph_qmaskedmean_i8u8(alias.input, mask_i8, alias.bytes,
                                     2u, 3u, 4u, 0.25f, -3, 0.5f, 5,
                                     VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(memcmp(alias.bytes, alias_before, sizeof(alias.bytes)) == 0);
    CHECK(vk_graph_qmaskedmean_i8u8(input_i8, mask_alias.mask, mask_alias.bytes,
                                     2u, 3u, 4u, 0.25f, -3, 0.5f, 5,
                                     VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(memcmp(mask_alias.bytes, mask_before, sizeof(mask_alias.bytes)) == 0);
    vk_graph_reset();
    return 0;
}

static int test_qconv2d_i8u8_packed_chain(void) {
    /* Grouped NHWC/OHWI convolution with full nonzero padding, stride,
       dilation, ReLU6, no bias, and a non-word-aligned output tail. */
    const uint8_t input[36] = {
        1, 2, 3, 1, 2, 3, 1, 2, 3,
        1, 2, 3, 1, 2, 3, 1, 2, 3,
        1, 2, 3, 1, 2, 3, 1, 2, 3,
        1, 2, 3, 1, 2, 3, 1, 2, 3,
    };
    const int8_t weight[12] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    };
    const float scales[3] = {1.0f, 1.0f, 1.0f};
    const int32_t zero_points[3] = {0, 0, 0};
    uint8_t hidden[27] = {0};
    int8_t output[27] = {0};
    const uint8_t expected_hidden[27] = {
        1, 2, 3, 2, 4, 6, 1, 2, 3,
        2, 4, 6, 4, 6, 6, 2, 4, 6,
        1, 2, 3, 2, 4, 6, 1, 2, 3,
    };
    const int8_t expected_output[27] = {
        0, 2, 4, 2, 6, 10, 0, 2, 4,
        2, 6, 10, 6, 10, 10, 2, 6, 10,
        0, 2, 4, 2, 6, 10, 0, 2, 4,
    };

    VkQConvTacticProbe grouped_tactic = {0};
    vk_graph_reset();
    vk_test_qconv_tactic_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qconv2d_i8u8(input, weight, scales, zero_points, NULL, hidden,
                                 1u, 3u, 4u, 3u, 3u, 3u, 3u, 2u, 2u, 1u,
                                 1u, 2u, 2u, 1u, 1u, 1u, 1u, 1u, 3u, 2u,
                                 1.0f, 0, 1.0f, 0,
                                 VX_DTYPE_U8, VX_DTYPE_I8,
                                 VX_DTYPE_U8) == 1);
    CHECK(vk_test_qconv_tactic_read(&grouped_tactic) == 0);
    CHECK(grouped_tactic.dot_tiled_dispatches == 0u &&
          grouped_tactic.tiled_dispatches == 0u &&
          grouped_tactic.scalar_dispatches == 1u);
    /* `hidden` is deliberately not synced before this byte-domain bridge. */
    CHECK(vk_graph_requantize_linear_i8u8(hidden, 27u, output, 27u,
                                          1.0f, 0, 0.5f, -2,
                                          VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(hidden, sizeof(hidden), 0) == 1);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    for (int i = 0; i < 27; i++) {
        CHECK(hidden[i] == expected_hidden[i]);
        CHECK(output[i] == expected_output[i]);
    }
    vk_graph_reset();

    /* Optional I32 bias and ordinary ReLU use the same canonical ABI. */
    const int8_t bias_input[1] = {-2};
    const int8_t bias_weight[1] = {3};
    const float bias_scale[1] = {0.25f};
    const int32_t bias_zero_point[1] = {0};
    const int32_t bias[1] = {4};
    int8_t bias_output[1] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_qconv2d_i8u8(bias_input, bias_weight, bias_scale,
                                 bias_zero_point, bias, bias_output,
                                 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                                 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 1u, 1u,
                                 0.5f, 0, 0.25f, -3,
                                 VX_DTYPE_I8, VX_DTYPE_I8,
                                 VX_DTYPE_I8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(bias_output, sizeof(bias_output), 0) == 1);
    CHECK(bias_output[0] == -3);
    CHECK(vk_graph_qconv2d_i8u8(input, weight, scales, zero_points, NULL, hidden,
                                 1u, 3u, 4u, 3u, 2u, 3u, 3u, 2u, 2u, 1u,
                                 1u, 2u, 2u, 1u, 1u, 1u, 1u, 1u, 3u, 2u,
                                 1.0f, 0, 1.0f, 0,
                                 VX_DTYPE_U8, VX_DTYPE_I8,
                                 VX_DTYPE_U8) == 0);
    vk_graph_reset();
    return 0;
}

/* The scalar QConv shader fuses channels 0..3 into one packed word. Five
 * output channels force the next word to cross a pixel and take qconv_one;
 * five input channels cover the odd-K byte tail with asymmetric U8/I8 ZPs. */
static int test_qconv2d_i8u8_scalar_fused_tail(void) {
    enum { INPUT_CHANNELS = 5, OUTPUT_CHANNELS = 5 };
    const int32_t input_zero_point = 129;
    const int32_t output_zero_point = -7;
    uint8_t input[2 * INPUT_CHANNELS];
    int8_t weight[OUTPUT_CHANNELS * INPUT_CHANNELS];
    float scales[OUTPUT_CHANNELS];
    int32_t zero_points[OUTPUT_CHANNELS];
    int32_t bias[OUTPUT_CHANNELS];
    int8_t output[2 * OUTPUT_CHANNELS];
    int8_t expected[2 * OUTPUT_CHANNELS];
    VkQConvTacticProbe tactic = {0};

    for (int index = 0; index < 2 * INPUT_CHANNELS; index++)
        input[index] = (uint8_t)(input_zero_point + (index * 5) % 7 - 3);
    for (int channel = 0; channel < OUTPUT_CHANNELS; channel++) {
        scales[channel] = 1.0f;
        zero_points[channel] = -4 + channel;
        bias[channel] = channel - 2;
        for (int k = 0; k < INPUT_CHANNELS; k++)
            weight[channel * INPUT_CHANNELS + k] = (int8_t)(
                zero_points[channel] + (channel * 3 + k * 2) % 7 - 3);
    }
    for (int pixel = 0; pixel < 2; pixel++) {
        for (int channel = 0; channel < OUTPUT_CHANNELS; channel++) {
            int32_t accumulator = bias[channel];
            for (int k = 0; k < INPUT_CHANNELS; k++) {
                accumulator +=
                    ((int32_t)input[pixel * INPUT_CHANNELS + k] -
                     input_zero_point) *
                    ((int32_t)weight[channel * INPUT_CHANNELS + k] -
                     zero_points[channel]);
            }
            accumulator += output_zero_point;
            if (accumulator < -128) accumulator = -128;
            if (accumulator > 127) accumulator = 127;
            expected[pixel * OUTPUT_CHANNELS + channel] =
                (int8_t)accumulator;
        }
    }

    vk_graph_reset();
    const int previous_dot_disabled = vk_test_set_packed_dot_disabled(1);
    CHECK(previous_dot_disabled >= 0);
    vk_test_qconv_tactic_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qconv2d_i8u8(
              input, weight, scales, zero_points, bias, output,
              1u, 1u, 2u, INPUT_CHANNELS,
              1u, 2u, OUTPUT_CHANNELS,
              1u, 1u, INPUT_CHANNELS,
              1u, 1u, 1u, 1u,
              0u, 0u, 0u, 0u,
              1u, 0u,
              0.5f, input_zero_point, 0.5f, output_zero_point,
              VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    CHECK(vk_test_qconv_tactic_read(&tactic) == 0);
    CHECK(tactic.dot_tiled_dispatches == 0u &&
          tactic.tiled_dispatches == 0u &&
          tactic.scalar_dispatches == 1u);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    CHECK(vk_test_set_packed_dot_disabled(previous_dot_disabled) == 1);
    vk_graph_reset();
    return 0;
}

static int test_conv2d_f32_regular_out16(void) {
    enum {
        INPUT_HEIGHT = 3,
        INPUT_WIDTH = 4,
        INPUT_CHANNELS = 5,
        OUTPUT_HEIGHT = 3,
        OUTPUT_WIDTH = 3,
        OUTPUT_CHANNELS = 16,
        KERNEL_HEIGHT = 2,
        KERNEL_WIDTH = 2,
    };
    float input[INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS];
    float weight[KERNEL_HEIGHT * KERNEL_WIDTH * INPUT_CHANNELS *
                 OUTPUT_CHANNELS];
    float bias[OUTPUT_CHANNELS];
    float output[OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS];
    float expected[OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS];
    VkConvTacticProbe tactic = {0};
    char telemetry[128] = {0};
    for (size_t index = 0; index < sizeof(input) / sizeof(input[0]); index++)
        input[index] = (float)((int)(index * 3u + 1u) % 11 - 5) * 0.0625f;
    for (size_t index = 0; index < sizeof(weight) / sizeof(weight[0]); index++)
        weight[index] = (float)((int)(index * 5u + 2u) % 13 - 6) * 0.03125f;
    for (int channel = 0; channel < OUTPUT_CHANNELS; channel++)
        bias[channel] = (float)(channel % 5 - 2) * 0.125f;
    for (int output_y = 0; output_y < OUTPUT_HEIGHT; output_y++) {
        for (int output_x = 0; output_x < OUTPUT_WIDTH; output_x++) {
            for (int output_channel = 0; output_channel < OUTPUT_CHANNELS;
                 output_channel++) {
                float sum = 0.0f;
                for (int input_channel = 0; input_channel < INPUT_CHANNELS;
                     input_channel++) {
                    for (int kernel_y = 0; kernel_y < KERNEL_HEIGHT; kernel_y++) {
                        const int input_y = output_y + kernel_y * 2 - 1;
                        if (input_y < 0 || input_y >= INPUT_HEIGHT) continue;
                        for (int kernel_x = 0; kernel_x < KERNEL_WIDTH; kernel_x++) {
                            const int input_x = output_x * 2 + kernel_x - 1;
                            if (input_x < 0 || input_x >= INPUT_WIDTH) continue;
                            const size_t input_index =
                                ((size_t)input_y * INPUT_WIDTH + (size_t)input_x) *
                                    INPUT_CHANNELS + (size_t)input_channel;
                            const size_t weight_index =
                                (((size_t)kernel_y * KERNEL_WIDTH +
                                  (size_t)kernel_x) * INPUT_CHANNELS +
                                 (size_t)input_channel) * OUTPUT_CHANNELS +
                                    (size_t)output_channel;
                            sum += input[input_index] * weight[weight_index];
                        }
                    }
                }
                float value = sum + bias[output_channel];
                if (value < 0.0f) value = 0.0f;
                if (value > 6.0f) value = 6.0f;
                expected[((size_t)output_y * OUTPUT_WIDTH +
                          (size_t)output_x) * OUTPUT_CHANNELS +
                         (size_t)output_channel] = value;
            }
        }
    }

    vk_graph_reset();
    vk_test_conv_tactic_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_conv2d_f32(
              input, output, weight, bias,
              1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS,
              KERNEL_HEIGHT, KERNEL_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH,
              1, 2, 1, 1, 1, 2, 2, 1) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    CHECK(vk_test_conv_tactic_read(&tactic) == 0);
    CHECK(tactic.out16_dispatches == 1u && tactic.scalar_dispatches == 0u);
    CHECK(vk_graph_append_dynamic_telemetry(
              telemetry, sizeof(telemetry)) == 0);
    CHECK(strstr(telemetry, ";vk_c16=1") != NULL &&
          strstr(telemetry, "vk_cs=") == NULL);
    for (size_t index = 0; index < sizeof(output) / sizeof(output[0]); index++)
        CHECK(close_enough(output[index], expected[index]));
    vk_graph_reset();
    return 0;
}

/* A 1x1 pointwise convolution also satisfies the regular out_c%16 shape.
 * Keep the pointwise selector ahead of the generic regular-out16 tactic. */
static int test_conv2d_f32_pointwise_precedes_regular_out16(void) {
    enum {
        INPUT_HEIGHT = 2,
        INPUT_WIDTH = 3,
        INPUT_CHANNELS = 5,
        OUTPUT_CHANNELS = 16,
    };
    float input[INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS];
    float weight[INPUT_CHANNELS * OUTPUT_CHANNELS];
    float bias[OUTPUT_CHANNELS];
    float output[INPUT_HEIGHT * INPUT_WIDTH * OUTPUT_CHANNELS];
    float expected[INPUT_HEIGHT * INPUT_WIDTH * OUTPUT_CHANNELS];
    VkConvTacticProbe tactic = {0};

    for (size_t index = 0; index < sizeof(input) / sizeof(input[0]); index++)
        input[index] = (float)((int)(index * 7u + 3u) % 13 - 6) * 0.0625f;
    for (size_t index = 0; index < sizeof(weight) / sizeof(weight[0]); index++)
        weight[index] = (float)((int)(index * 5u + 1u) % 11 - 5) * 0.03125f;
    for (int output_channel = 0; output_channel < OUTPUT_CHANNELS;
         output_channel++) {
        bias[output_channel] = (float)(output_channel % 7 - 3) * 0.0625f;
    }
    for (int position = 0; position < INPUT_HEIGHT * INPUT_WIDTH; position++) {
        for (int output_channel = 0; output_channel < OUTPUT_CHANNELS;
             output_channel++) {
            float sum = bias[output_channel];
            for (int input_channel = 0; input_channel < INPUT_CHANNELS;
                 input_channel++) {
                sum += input[position * INPUT_CHANNELS + input_channel] *
                    weight[input_channel * OUTPUT_CHANNELS + output_channel];
            }
            expected[position * OUTPUT_CHANNELS + output_channel] = sum;
        }
    }

    vk_graph_reset();
    vk_test_conv_tactic_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_conv2d_f32(
              input, output, weight, bias,
              1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS,
              1, 1, INPUT_HEIGHT, INPUT_WIDTH,
              1, 1, 0, 0, 1, 0, 1, 1) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    CHECK(vk_test_conv_tactic_read(&tactic) == 0);
    CHECK(tactic.pointwise_selections == 1u);
    CHECK(tactic.out16_dispatches == 0u);
    for (size_t index = 0; index < sizeof(output) / sizeof(output[0]); index++)
        CHECK(close_enough(output[index], expected[index]));
    vk_graph_reset();
    return 0;
}

/* The feature-independent 8x4 Vulkan QConv tactic must cover an asymmetric
 * U8/I8 boundary, padded output positions, and a 17-byte reduction tail.
 * Force packed dot off so this remains deterministic on dot-capable hosts;
 * groups != 1 is covered by the scalar assertion above. */
static int test_qconv2d_i8u8_scalar_tiled_tail(void) {
    enum {
        INPUT_HEIGHT = 2,
        INPUT_WIDTH = 3,
        INPUT_CHANNELS = 17,
        OUTPUT_HEIGHT = 4,
        OUTPUT_WIDTH = 5,
        OUTPUT_CHANNELS = 36,
    };
    uint8_t input[INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS];
    int8_t weight[OUTPUT_CHANNELS * INPUT_CHANNELS];
    float scales[OUTPUT_CHANNELS];
    int32_t zero_points[OUTPUT_CHANNELS];
    int32_t bias[OUTPUT_CHANNELS];
    int8_t output[OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS];
    int8_t expected[OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS];
    VkQConvTacticProbe tactic = {0};
    char telemetry[128] = {0};
    const int32_t input_zero_point = 131;
    const int32_t output_zero_point = -7;

    for (size_t index = 0; index < sizeof(input); index++)
        input[index] = (uint8_t)(input_zero_point +
            ((int32_t)(index * 3u + 1u) % 5) - 2);
    for (int channel = 0; channel < OUTPUT_CHANNELS; channel++) {
        zero_points[channel] = -4 + channel % 5;
        scales[channel] = 0.5f;
        bias[channel] = channel % 7 - 3;
        for (int k = 0; k < INPUT_CHANNELS; k++) {
            weight[channel * INPUT_CHANNELS + k] = (int8_t)(
                zero_points[channel] + (channel * 3 + k * 7 + 1) % 5 - 2);
        }
    }
    for (int output_y = 0; output_y < OUTPUT_HEIGHT; output_y++) {
        for (int output_x = 0; output_x < OUTPUT_WIDTH; output_x++) {
            const int input_y = output_y - 1;
            const int input_x = output_x - 1;
            for (int channel = 0; channel < OUTPUT_CHANNELS; channel++) {
                int32_t accumulator = bias[channel];
                if (input_y >= 0 && input_y < INPUT_HEIGHT &&
                    input_x >= 0 && input_x < INPUT_WIDTH) {
                    for (int k = 0; k < INPUT_CHANNELS; k++) {
                        const size_t input_index =
                            ((size_t)input_y * INPUT_WIDTH + (size_t)input_x) *
                                INPUT_CHANNELS + (size_t)k;
                        const size_t weight_index =
                            (size_t)channel * INPUT_CHANNELS + (size_t)k;
                        accumulator +=
                            ((int32_t)input[input_index] - input_zero_point) *
                            ((int32_t)weight[weight_index] - zero_points[channel]);
                    }
                }
                int32_t quantized = accumulator + output_zero_point;
                if (quantized < -128) quantized = -128;
                if (quantized > 127) quantized = 127;
                expected[((size_t)output_y * OUTPUT_WIDTH + (size_t)output_x) *
                    OUTPUT_CHANNELS + (size_t)channel] = (int8_t)quantized;
            }
        }
    }

    vk_graph_reset();
    const int previous_dot_disabled = vk_test_set_packed_dot_disabled(1);
    CHECK(previous_dot_disabled >= 0);
    vk_test_qconv_tactic_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qconv2d_i8u8(
              input, weight, scales, zero_points, bias, output,
              1u, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS,
              OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_CHANNELS,
              1u, 1u, INPUT_CHANNELS,
              1u, 1u, 1u, 1u,
              1u, 1u, 1u, 1u,
              1u, 0u,
              0.25f, input_zero_point, 0.125f, output_zero_point,
              VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    CHECK(vk_test_qconv_tactic_read(&tactic) == 0);
    CHECK(tactic.dot_tiled_dispatches == 0u &&
          tactic.tiled_dispatches == 1u &&
          tactic.scalar_dispatches == 0u);
    CHECK(vk_graph_append_dynamic_telemetry(
              telemetry, sizeof(telemetry)) == 0);
    CHECK(strstr(telemetry, ";vk_qct=1") != NULL &&
          strstr(telemetry, "vk_qcs=") == NULL);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    CHECK(vk_test_set_packed_dot_disabled(previous_dot_disabled) == 1);
    vk_graph_reset();
    return 0;
}

static int test_typed_i8u8_shape_qdq_chain(void) {
    /* The entire chain stays byte-packed on the device until the final
       DequantizeLinear.  Concat deliberately has unaligned channel lanes,
       and Resize has a two-byte tail, covering the packed-word boundaries. */
    const float source_a[4] = {2.0f, 4.0f, 1.0f, 3.0f};
    const float source_b[4] = {0.0f, 5.0f, 6.0f, 1.0f};
    int8_t quantized_a[4] = {0};
    int8_t copied_a[4] = {0};
    int8_t quantized_b[4] = {0};
    int8_t concatenated[8] = {0};
    int8_t pooled[8] = {0};
    int8_t resized[18] = {0};
    float output[18] = {0};
    const void* inputs[2] = {copied_a, quantized_b};
    const uint32_t input_elements[2] = {4u, 4u};
    const uint32_t input_axes[2] = {1u, 1u};
    const float input_scales[2] = {1.0f, 1.0f};
    const int32_t input_zero_points[2] = {0, 0};
    const uint32_t input_dtypes[2] = {VX_DTYPE_I8, VX_DTYPE_I8};
    const int8_t expected[18] = {
        4, 6, 4, 6, 4, 5,
        4, 6, 4, 6, 4, 5,
        3, 6, 3, 6, 3, 1,
    };

    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_quantize_typed_f32_i8u8(source_a, 4u, quantized_a,
                                            1.0f, 0, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_copy_i8u8(quantized_a, 4u, copied_a, 4u,
                             1.0f, 0, 1.0f, 0,
                             VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_quantize_typed_f32_i8u8(source_b, 4u, quantized_b,
                                            1.0f, 0, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_concat_i8u8(inputs, input_elements, input_axes,
                               input_scales, input_zero_points, input_dtypes,
                               2u, concatenated, 8u, 2u, 1u,
                               1.0f, 0, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_maxpool2d_i8u8(concatenated, pooled,
                                  1u, 2u, 2u, 2u, 2u, 2u,
                                  2u, 2u, 1u, 1u, 0u, 0u, 1u, 1u,
                                  1.0f, 0, 1.0f, 0,
                                  VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_resize_nearest_i8u8(pooled, resized,
                                       1u, 2u, 2u, 2u, 3u, 3u,
                                       1.0f, 0, 1.0f, 0,
                                       VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(vk_graph_dequantize_typed_i8u8_f32(resized, 18u,
                                              1.0f, 0, VX_DTYPE_I8,
                                              output) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    for (int index = 0; index < 18; index++) {
        CHECK(close_enough(output[index], (float)expected[index]));
    }

    /* U8 uses the same packed tail path but has a distinct zero-point domain. */
    const float unsigned_source[3] = {-1.0f, 0.0f, 5.0f};
    uint8_t unsigned_quantized[3] = {0};
    float unsigned_output[3] = {0};
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_quantize_typed_f32_i8u8(unsigned_source, 3u,
                                            unsigned_quantized, 1.0f, 128,
                                            VX_DTYPE_U8) == 1);
    CHECK(vk_graph_dequantize_typed_i8u8_f32(unsigned_quantized, 3u,
                                              1.0f, 128, VX_DTYPE_U8,
                                              unsigned_output) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(unsigned_quantized, sizeof(unsigned_quantized), 0) == 1);
    CHECK(vk_graph_sync_host(unsigned_output, sizeof(unsigned_output), 0) == 1);
    CHECK(unsigned_quantized[0] == 127u && unsigned_quantized[1] == 128u &&
          unsigned_quantized[2] == 133u);
    CHECK(close_enough(unsigned_output[0], -1.0f));
    CHECK(close_enough(unsigned_output[1], 0.0f));
    CHECK(close_enough(unsigned_output[2], 5.0f));

    /* The byte-preserving shape path cannot silently change quantization. */
    CHECK(vk_graph_copy_i8u8(quantized_a, 4u, copied_a, 4u,
                             1.0f, 0, 0.5f, 0,
                             VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    vk_graph_reset();
    return 0;
}

static int test_multi_entry_matmul_backward(void) {
    float input[2] = {1.0f, 2.0f};
    /* Native IN_OUT layout; the shader separately controls weight reads and
       the destination layout of grad_weight. */
    float weight[4] = {3.0f, 5.0f, 4.0f, 6.0f};
    float grad_output[2] = {7.0f, 8.0f};
    float grad_input[2] = {0.0f, 0.0f};
    float grad_weight[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float grad_bias[2] = {0.0f, 0.0f};
    uint32_t params[8] = {1u, 2u, 2u, 1u, 1u, 1u, 0u, 0u};
    void* hosts[7] = {input, weight, grad_output, grad_input, grad_weight,
                      grad_bias, params};
    size_t bytes[7] = {sizeof(input), sizeof(weight), sizeof(grad_output),
                       sizeof(grad_input), sizeof(grad_weight), sizeof(grad_bias),
                       sizeof(params)};
    unsigned char access[7] = {
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ
    };
    unsigned char weights[7] = {0, 1, 0, 0, 0, 0, 0};
    CHECK(vk_training_dispatch("matMulBackward", "input_main", hosts, bytes,
                               access, weights, 7, 1, 1, 1) == 0);
    CHECK(vk_training_dispatch("matMulBackward", "weight_main", hosts, bytes,
                               access, weights, 7, 1, 1, 1) == 0);
    CHECK(vk_training_dispatch("matMulBackward", "bias_main", hosts, bytes,
                               access, weights, 7, 1, 1, 1) == 0);
    CHECK(vk_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(vk_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    CHECK(vk_training_sync(grad_bias, sizeof(grad_bias)) == 0);
    CHECK(close_enough(grad_input[0], 61.0f));
    CHECK(close_enough(grad_input[1], 76.0f));
    CHECK(close_enough(grad_weight[0], 7.0f));
    CHECK(close_enough(grad_weight[1], 8.0f));
    CHECK(close_enough(grad_weight[2], 14.0f));
    CHECK(close_enough(grad_weight[3], 16.0f));
    CHECK(close_enough(grad_bias[0], 7.0f));
    CHECK(close_enough(grad_bias[1], 8.0f));
    return 0;
}

static int test_batched_attention_forward(void) {
    /* seq=1 makes attention output exactly V, which isolates batch offsets. */
    float qkv[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f};
    float sdpa_output[2] = {0.0f, 0.0f};
    vk_graph_begin_forward();
    CHECK(vk_graph_sdpa_f32(qkv, NULL, 0, sdpa_output,
                            1, 1, 1, 1, 2, 1.0f, 1, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(sdpa_output, sizeof(sdpa_output), 0) == 1);
    CHECK(close_enough(sdpa_output[0], 3.0f));
    CHECK(close_enough(sdpa_output[1], 7.0f));

    float q[2] = {1.0f, 4.0f};
    float k[2] = {2.0f, 5.0f};
    float v[2] = {11.0f, 13.0f};
    float cross_output[2] = {0.0f, 0.0f};
    vk_graph_begin_forward();
    CHECK(vk_graph_cross_sdpa_f32(q, k, v, NULL, 0, cross_output,
                                  1, 1, 1, 1, 1, 2, 1.0f, 0, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(cross_output, sizeof(cross_output), 0) == 1);
    CHECK(close_enough(cross_output[0], 11.0f));
    CHECK(close_enough(cross_output[1], 13.0f));

    float masked_qkv[6] = {0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 6.0f};
    int32_t key_mask[2] = {1, 0};
    float masked_output[2] = {0.0f, 0.0f};
    vk_graph_begin_forward();
    CHECK(vk_graph_sdpa_f32(masked_qkv, key_mask, 2, masked_output,
                            2, 1, 1, 1, 1, 1.0f, 0, 1) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(masked_output, sizeof(masked_output), 0) == 1);
    CHECK(close_enough(masked_output[0], 2.0f));
    CHECK(close_enough(masked_output[1], 2.0f));

    float cross_q[1] = {0.0f};
    float cross_k[2] = {0.0f, 0.0f};
    float cross_v[2] = {3.0f, 9.0f};
    int32_t cross_mask[2] = {0, 1};
    float masked_cross_output[1] = {0.0f};
    vk_graph_begin_forward();
    CHECK(vk_graph_cross_sdpa_f32(cross_q, cross_k, cross_v, cross_mask, 2,
                                  masked_cross_output, 1, 2, 1, 1, 1, 1,
                                  1.0f, 0, 1) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(masked_cross_output, sizeof(masked_cross_output), 0) == 1);
    CHECK(close_enough(masked_cross_output[0], 9.0f));

    const float parity_qkv[12] = {
        0.4f, -0.2f, 0.1f, 0.5f, 0.7f, -0.3f,
        -0.1f, 0.6f, 0.3f, -0.4f, 0.2f, 0.8f
    };
    const float parity_q[4] = {0.4f, -0.2f, -0.1f, 0.6f};
    const float parity_k[4] = {0.1f, 0.5f, 0.3f, -0.4f};
    const float parity_v[4] = {0.7f, -0.3f, 0.2f, 0.8f};
    const float expected_inference[4] = {
        0.4270835221f, 0.3004162014f, 0.4988606870f, 0.1425064802f
    };
    const float expected_training[4] = {
        0.6358339190f, -0.2725002468f, 0.1609114408f, 0.6436457634f
    };
    float parity_inference[4] = {0};
    float parity_training[4] = {0};
    float parity_cross_training[4] = {0};
    const float attention_scale = 0.7071067811865475f;
    vk_graph_begin_forward();
    CHECK(vk_graph_sdpa_f32(parity_qkv, NULL, 0, parity_inference,
                            2, 2, 1, 2, 1, attention_scale, 0, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(parity_inference, sizeof(parity_inference), 0) == 1);
    vk_graph_begin_forward();
    CHECK(vk_graph_sdpa_training_f32(parity_qkv, NULL, 0, parity_training,
                                     2, 2, 1, 2, 1, attention_scale, 0, 0,
                                     0x80000000u, 0x9e3779abu, 7u, 2.0f) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(parity_training, sizeof(parity_training), 0) == 1);
    vk_graph_begin_forward();
    CHECK(vk_graph_cross_sdpa_training_f32(parity_q, parity_k, parity_v, NULL, 0,
                                           parity_cross_training, 2, 2, 2, 1, 2, 1,
                                           attention_scale, 0, 0, 0x80000000u,
                                           0x9e3779abu, 7u, 2.0f) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(parity_cross_training, sizeof(parity_cross_training), 0) == 1);
    for (int i = 0; i < 4; i++) {
        CHECK(close_enough(parity_inference[i], expected_inference[i]));
        CHECK(close_enough(parity_training[i], expected_training[i]));
        CHECK(close_enough(parity_cross_training[i], expected_training[i]));
    }
    CHECK(!close_enough(parity_inference[0], parity_training[0]));

    int32_t tokens[2] = {1, 0};
    float embedding_weight[2] = {4.0f, 7.0f};
    float embedding_output[2] = {0.0f, 0.0f};
    vk_graph_begin_forward();
    CHECK(vk_graph_embedding_f32(tokens, embedding_weight, embedding_output, 2, 1, 2) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(embedding_output, sizeof(embedding_output), 0) == 1);
    CHECK(close_enough(embedding_output[0], 7.0f));
    CHECK(close_enough(embedding_output[1], 4.0f));

    float pool_input[8] = {1.0f, 4.0f, 2.0f, 3.0f, -1.0f, -2.0f, 8.0f, 5.0f};
    float pool_output[2] = {0.0f, 0.0f};
    vk_graph_begin_forward();
    CHECK(vk_graph_maxpool2d_f32(pool_input, pool_output, 2, 2, 2, 1,
                                 1, 1, 2, 2, 2, 2, 0, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(pool_output, sizeof(pool_output), 0) == 1);
    CHECK(close_enough(pool_output[0], 4.0f));
    CHECK(close_enough(pool_output[1], 8.0f));

    float conv_input[6] = {1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f};
    float conv_weight[1] = {2.0f};
    float conv_bias[1] = {1.0f};
    float conv_output[6] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_conv1d_f32(conv_input, conv_weight, conv_bias, conv_output,
                              2, 1, 3, 1, 3, 1, 1, 0, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(conv_output, sizeof(conv_output), 0) == 1);
    const float expected_conv[6] = {3.0f, 5.0f, 7.0f, 21.0f, 41.0f, 61.0f};
    for (int i = 0; i < 6; i++) CHECK(close_enough(conv_output[i], expected_conv[i]));

    float projected_q[2] = {1.0f, -1.0f};
    float projected_kv[2] = {3.0f, 4.0f};
    float projected_weight[3] = {1.0f, 1.0f, 2.0f};
    float projected_output[2] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_cross_attention_f32(projected_q, projected_kv, projected_weight,
                                       NULL, NULL, projected_output,
                                       1, 1, 1, 1, 1, 2, 0, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(projected_output, sizeof(projected_output), 0) == 1);
    CHECK(close_enough(projected_output[0], 6.0f));
    CHECK(close_enough(projected_output[1], 8.0f));

    float profile_input[4] = {1.0f, 3.0f, 10.0f, 14.0f};
    float profile_output[4] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_profile_x_f32(profile_input, profile_output, 2, 2, 1, 1) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(profile_output, sizeof(profile_output), 0) == 1);
    const float expected_profile[4] = {3.0f, 2.0f, 14.0f, 12.0f};
    for (int i = 0; i < 4; i++) CHECK(close_enough(profile_output[i], expected_profile[i]));

    float concat_a[2] = {1.0f, 10.0f};
    float concat_b[4] = {2.0f, 3.0f, 20.0f, 30.0f};
    const float* concat_inputs[2] = {concat_a, concat_b};
    long concat_sizes[2] = {2, 4};
    int concat_axes[2] = {1, 2};
    float concat_output[6] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_concat_f32(concat_inputs, concat_sizes, concat_axes, 2,
                              concat_output, 3, 1, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(concat_output, sizeof(concat_output), 0) == 1);
    const float expected_concat[6] = {1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f};
    for (int i = 0; i < 6; i++) CHECK(close_enough(concat_output[i], expected_concat[i]));
    vk_graph_begin_forward();
    CHECK(vk_graph_concat_f32(concat_inputs, concat_sizes, concat_axes, 2,
                              concat_output, 3, 1, 1) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(concat_output, sizeof(concat_output), 0) == 1);
    for (int i = 0; i < 6; i++) {
        CHECK(close_enough(concat_output[i], 1.0f / (1.0f + expf(-expected_concat[i]))));
    }
    return 0;
}

static int test_general_gather_i32(void) {
    const float input[12] = {
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
    };
    const int32_t indices[5] = {2, 0, -1, -4, 3};
    const float expected[20] = {
        5, 6, 1, 2, 5, 6, -1, -1, -1, -1,
        11, 12, 7, 8, 11, 12, -1, -1, -1, -1,
    };
    float output[20] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_gather_i32_f32(input, indices, output,
                                  2, 3, 2, 5, 20) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    for (int index = 0; index < 20; index++)
        CHECK(close_enough(output[index], expected[index]));
    vk_graph_reset();
    return 0;
}

static int test_batched_attention_backward(void) {
    float qkv[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f};
    float grad_output[2] = {2.0f, 3.0f};
    float grad_qkv[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    struct {
        uint32_t seq_len, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, threshold, seed, counter;
        float dropout_scale;
    } params = {1u, 1u, 1u, 1u, 2u, 1.0f, 1u, 0u, 0u, 0u, 0u, 1.0f};
    void* hosts[5] = {qkv, qkv, grad_output, grad_qkv, &params};
    size_t bytes[5] = {sizeof(qkv), sizeof(qkv), sizeof(grad_output), sizeof(grad_qkv),
                       sizeof(params)};
    unsigned char access[5] = {
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    CHECK(vk_training_dispatch("sdpaBackward", "main", hosts, bytes, access,
                               NULL, 5, 1, 1, 6) == 0);
    CHECK(vk_training_sync(grad_qkv, sizeof(grad_qkv)) == 0);
    CHECK(close_enough(grad_qkv[0], 0.0f));
    CHECK(close_enough(grad_qkv[1], 0.0f));
    CHECK(close_enough(grad_qkv[2], 2.0f));
    CHECK(close_enough(grad_qkv[3], 0.0f));
    CHECK(close_enough(grad_qkv[4], 0.0f));
    CHECK(close_enough(grad_qkv[5], 3.0f));

    float masked_qkv[6] = {0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 6.0f};
    int32_t mask[2] = {1, 0};
    float masked_grad_output[2] = {1.0f, 1.0f};
    float masked_grad_qkv[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    struct {
        uint32_t seq_len, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, threshold, seed, counter;
        float dropout_scale;
    } masked_params = {2u, 1u, 1u, 1u, 1u, 1.0f, 0u, 1u, 0u, 0u, 0u, 1.0f};
    void* masked_hosts[5] = {masked_qkv, mask, masked_grad_output, masked_grad_qkv,
                             &masked_params};
    size_t masked_bytes[5] = {sizeof(masked_qkv), sizeof(mask), sizeof(masked_grad_output),
                              sizeof(masked_grad_qkv), sizeof(masked_params)};
    CHECK(vk_training_dispatch("sdpaBackward", "main", masked_hosts, masked_bytes,
                               access, NULL, 5, 2, 1, 3) == 0);
    CHECK(vk_training_sync(masked_grad_qkv, sizeof(masked_grad_qkv)) == 0);
    CHECK(close_enough(masked_grad_qkv[0], 0.0f));
    CHECK(close_enough(masked_grad_qkv[1], 0.0f));
    CHECK(close_enough(masked_grad_qkv[2], 2.0f));
    CHECK(close_enough(masked_grad_qkv[3], 0.0f));
    CHECK(close_enough(masked_grad_qkv[4], 0.0f));
    CHECK(close_enough(masked_grad_qkv[5], 0.0f));
    return 0;
}

static int test_batched_cross_attention_backward(void) {
    float q[2] = {1.0f, 4.0f};
    float k[2] = {2.0f, 5.0f};
    float v[2] = {11.0f, 13.0f};
    float grad_output[2] = {2.0f, 3.0f};
    float grad_q[2] = {0.0f, 0.0f};
    float grad_k[2] = {0.0f, 0.0f};
    float grad_v[2] = {0.0f, 0.0f};
    struct {
        uint32_t seq_q, seq_kv, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, threshold, seed, counter;
        float dropout_scale;
        uint32_t pad[3];
    } params = {1u, 1u, 1u, 1u, 1u, 2u, 1.0f, 0u, 0u,
                0u, 0u, 0u, 1.0f, {0u, 0u, 0u}};
    void* hosts[9] = {q, k, v, q, grad_output, grad_q, grad_k, grad_v, &params};
    size_t bytes[9] = {
        sizeof(q), sizeof(k), sizeof(v), sizeof(q), sizeof(grad_output), sizeof(grad_q),
        sizeof(grad_k), sizeof(grad_v), sizeof(params)
    };
    unsigned char access[9] = {
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ
    };
    const char* entries[3] = {"q_main", "k_main", "v_main"};
    for (int i = 0; i < 3; i++) {
        CHECK(vk_training_dispatch("crossSdpaBackward", entries[i], hosts, bytes,
                                   access, NULL, 9, 1, 1, 2) == 0);
    }
    CHECK(vk_training_sync(grad_q, sizeof(grad_q)) == 0);
    CHECK(vk_training_sync(grad_k, sizeof(grad_k)) == 0);
    CHECK(vk_training_sync(grad_v, sizeof(grad_v)) == 0);
    CHECK(close_enough(grad_q[0], 0.0f) && close_enough(grad_q[1], 0.0f));
    CHECK(close_enough(grad_k[0], 0.0f) && close_enough(grad_k[1], 0.0f));
    CHECK(close_enough(grad_v[0], 2.0f) && close_enough(grad_v[1], 3.0f));
    return 0;
}

static int test_moe_backward_bindings(void) {
    float input = 2.0f;
    float expert_weight = 3.0f;
    float expert_bias = 4.0f;
    float route_index = 2.0f;
    float route_weight = 0.5f;
    float grad_output = 7.0f;
    float grad_input = 0.0f;
    float grad_expert_weight = 0.0f;
    float grad_expert_bias = 0.0f;
    float grad_route_weight = 0.0f;
    uint32_t slot_rows[3] = {UINT32_MAX, UINT32_MAX, 0u};
    uint32_t row_slots[1] = {2u};
    uint32_t params[8] = {1u, 1u, 1u, 1u, 1u, 1u, 3u, 0u};
    void* hosts[13] = {
        &input, &expert_weight, &expert_bias, &route_index, &route_weight,
        &grad_output, &grad_input, &grad_expert_weight, &grad_expert_bias,
        &grad_route_weight, slot_rows, row_slots, params
    };
    size_t bytes[13] = {
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float),
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float),
        sizeof(slot_rows), sizeof(row_slots), sizeof(params)
    };
    unsigned char access[13] = {
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ
    };
    unsigned char weights[13] = {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const char* entries[4] = {"input_main", "weight_main", "bias_main", "route_main"};
    for (int i = 0; i < 4; i++) {
        CHECK(vk_training_dispatch("moeLinearBackward", entries[i], hosts, bytes,
                                   access, weights, 13, 1, 1, 1) == 0);
    }
    CHECK(vk_training_sync(&grad_input, sizeof(grad_input)) == 0);
    CHECK(vk_training_sync(&grad_expert_weight, sizeof(grad_expert_weight)) == 0);
    CHECK(vk_training_sync(&grad_expert_bias, sizeof(grad_expert_bias)) == 0);
    CHECK(vk_training_sync(&grad_route_weight, sizeof(grad_route_weight)) == 0);
    CHECK(close_enough(grad_input, 10.5f));
    CHECK(close_enough(grad_expert_weight, 7.0f));
    CHECK(close_enough(grad_expert_bias, 3.5f));
    CHECK(close_enough(grad_route_weight, 70.0f));
    return 0;
}

static int test_prelu_logsoftmax_split_backward(void) {
    float input[4] = {-1.0f, 2.0f, -0.5f, 1.5f};
    float weight[2] = {0.2f, 0.4f};
    float grad_output[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float grad_input[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float grad_weight[2] = {0.0f, 0.0f};
    uint32_t prelu_params[4] = {4u, 2u, 2u, 0u};
    void* prelu_hosts[6] = {input, weight, grad_output, grad_input, grad_weight, prelu_params};
    size_t prelu_bytes[6] = {sizeof(input), sizeof(weight), sizeof(grad_output),
                             sizeof(grad_input), sizeof(grad_weight), sizeof(prelu_params)};
    unsigned char prelu_access[6] = {
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    CHECK(vk_training_dispatch("preluBackward", "input_main", prelu_hosts, prelu_bytes,
                               prelu_access, NULL, 6, 1, 1, 1) == 0);
    CHECK(vk_training_dispatch("preluBackward", "weight_main", prelu_hosts, prelu_bytes,
                               prelu_access, NULL, 6, 1, 1, 1) == 0);
    CHECK(vk_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(vk_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    CHECK(close_enough(grad_input[0], 0.2f) && close_enough(grad_input[1], 2.0f));
    CHECK(close_enough(grad_input[2], 0.6f) && close_enough(grad_input[3], 4.0f));
    CHECK(close_enough(grad_weight[0], -2.5f) && close_enough(grad_weight[1], 0.0f));

    float log_output[2] = {-1.38629436112f, -0.28768207245f};
    float log_grad_output[2] = {2.0f, -1.0f};
    float log_grad_input[2] = {0.0f, 0.0f};
    uint32_t softmax_params[4] = {1u, 2u, 1u, 0u};
    void* softmax_hosts[4] = {log_output, log_grad_output, log_grad_input, softmax_params};
    size_t softmax_bytes[4] = {sizeof(log_output), sizeof(log_grad_output),
                               sizeof(log_grad_input), sizeof(softmax_params)};
    unsigned char softmax_access[4] = {
        VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    CHECK(vk_training_dispatch("softmaxBackward", "main", softmax_hosts, softmax_bytes,
                               softmax_access, NULL, 4, 1, 1, 1) == 0);
    CHECK(vk_training_sync(log_grad_input, sizeof(log_grad_input)) == 0);
    CHECK(close_enough(log_grad_input[0], 1.75f));
    CHECK(close_enough(log_grad_input[1], -1.75f));

    float left_grad[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float right_grad[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float split_grad[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t left_params[5] = {4u, 1u, 2u, 4u, 0u};
    uint32_t right_params[5] = {4u, 1u, 2u, 4u, 2u};
    void* split_hosts[3] = {left_grad, split_grad, left_params};
    size_t split_bytes[3] = {sizeof(left_grad), sizeof(split_grad), sizeof(left_params)};
    unsigned char split_access[3] = {
        VK_TRAINING_READ, VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    CHECK(vk_training_dispatch("splitBackward", "main", split_hosts, split_bytes,
                               split_access, NULL, 3, 1, 1, 1) == 0);
    split_hosts[0] = right_grad;
    split_hosts[2] = right_params;
    CHECK(vk_training_dispatch("splitBackward", "main", split_hosts, split_bytes,
                               split_access, NULL, 3, 1, 1, 1) == 0);
    CHECK(vk_training_sync(split_grad, sizeof(split_grad)) == 0);
    const float expected_split[8] = {1.0f, 2.0f, 5.0f, 6.0f,
                                     3.0f, 4.0f, 7.0f, 8.0f};
    for (int i = 0; i < 8; i++) CHECK(close_enough(split_grad[i], expected_split[i]));
    return 0;
}

static int test_groupnorm_dropout_reduce_backward(void) {
    float input[8] = {1.0f, 3.0f, 2.0f, 6.0f, 4.0f, 0.0f, 5.0f, 1.0f};
    float weight[4] = {1.0f, 0.5f, 1.5f, 0.75f};
    float grad_output[8] = {1.0f, 2.0f, -1.0f, 3.0f, 4.0f, -2.0f, 2.0f, 1.0f};
    float grad_input[8] = {0};
    float grad_weight[4] = {0};
    float grad_bias[4] = {0};
    struct {
        uint32_t batch, height, width, channels, groups, has_bias;
        float epsilon;
        uint32_t pad;
    } params = {1u, 1u, 2u, 4u, 2u, 1u, 1.0e-4f, 0u};
    void* hosts[7] = {input, weight, grad_output, grad_input, grad_weight,
                      grad_bias, &params};
    size_t bytes[7] = {sizeof(input), sizeof(weight), sizeof(grad_output),
                       sizeof(grad_input), sizeof(grad_weight), sizeof(grad_bias),
                       sizeof(params)};
    unsigned char access[7] = {
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    CHECK(vk_training_dispatch("groupNormBackward", "input_main", hosts, bytes,
                               access, NULL, 7, 1, 1, 1) == 0);
    CHECK(vk_training_dispatch("groupNormBackward", "param_main", hosts, bytes,
                               access, NULL, 7, 1, 1, 1) == 0);
    CHECK(vk_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(vk_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    CHECK(vk_training_sync(grad_bias, sizeof(grad_bias)) == 0);
    float expected_input[8] = {0}, expected_weight[4] = {0}, expected_bias[4] = {0};
    for (int group = 0; group < 2; group++) {
        float mean = 0.0f, square_mean = 0.0f;
        for (int spatial = 0; spatial < 2; spatial++) for (int local = 0; local < 2; local++) {
            float value = input[spatial * 4 + group * 2 + local];
            mean += value * 0.25f;
            square_mean += value * value * 0.25f;
        }
        float inverse = 1.0f / sqrtf(square_mean - mean * mean + params.epsilon);
        float sum = 0.0f, sum_xhat = 0.0f;
        for (int spatial = 0; spatial < 2; spatial++) for (int local = 0; local < 2; local++) {
            int channel = group * 2 + local;
            int index = spatial * 4 + channel;
            float xhat = (input[index] - mean) * inverse;
            float scaled = grad_output[index] * weight[channel];
            sum += scaled;
            sum_xhat += scaled * xhat;
            expected_weight[channel] += grad_output[index] * xhat;
            expected_bias[channel] += grad_output[index];
        }
        for (int spatial = 0; spatial < 2; spatial++) for (int local = 0; local < 2; local++) {
            int channel = group * 2 + local;
            int index = spatial * 4 + channel;
            float xhat = (input[index] - mean) * inverse;
            float scaled = grad_output[index] * weight[channel];
            expected_input[index] = inverse * 0.25f * (4.0f * scaled - sum - xhat * sum_xhat);
        }
    }
    for (int i = 0; i < 8; i++) CHECK(close_enough(grad_input[i], expected_input[i]));
    for (int i = 0; i < 4; i++) {
        CHECK(close_enough(grad_weight[i], expected_weight[i]));
        CHECK(close_enough(grad_bias[i], expected_bias[i]));
    }

    float dropout_go[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float dropout_gi[8] = {0};
    struct { uint32_t length, threshold, seed, counter; float scale; uint32_t pad[3]; }
        dropout_params = {8u, 0x80000000u, 123u, 7u, 2.0f, {0u, 0u, 0u}};
    void* dropout_hosts[3] = {dropout_go, dropout_gi, &dropout_params};
    size_t dropout_bytes[3] = {sizeof(dropout_go), sizeof(dropout_gi), sizeof(dropout_params)};
    unsigned char dropout_access[3] = {
        VK_TRAINING_READ, VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    CHECK(vk_training_dispatch("dropoutBackward", "main", dropout_hosts, dropout_bytes,
                               dropout_access, NULL, 3, 1, 1, 1) == 0);
    CHECK(vk_training_sync(dropout_gi, sizeof(dropout_gi)) == 0);
    for (int i = 0; i < 8; i++) {
        float expected = dropout_bits(123u, 7u, (uint32_t)i) >= 0x80000000u
            ? dropout_go[i] * 2.0f : 0.0f;
        CHECK(close_enough(dropout_gi[i], expected));
    }

    float reduce_go[2] = {2.0f, 4.0f};
    float reduce_gi[6] = {0};
    struct { uint32_t rows, width; float scale; uint32_t pad; }
        reduce_params = {6u, 3u, 1.0f / 3.0f, 0u};
    void* reduce_hosts[3] = {reduce_go, reduce_gi, &reduce_params};
    size_t reduce_bytes[3] = {sizeof(reduce_go), sizeof(reduce_gi), sizeof(reduce_params)};
    CHECK(vk_training_dispatch("reduceBackward", "main", reduce_hosts, reduce_bytes,
                               dropout_access, NULL, 3, 1, 1, 1) == 0);
    CHECK(vk_training_sync(reduce_gi, sizeof(reduce_gi)) == 0);
    for (int i = 0; i < 3; i++) CHECK(close_enough(reduce_gi[i], 2.0f / 3.0f));
    for (int i = 3; i < 6; i++) CHECK(close_enough(reduce_gi[i], 4.0f / 3.0f));

    float a[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float b[4] = {2, 3, 5, 7};
    float binary_go[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    float grad_a[12] = {0};
    float grad_b[4] = {0};
    uint32_t binary_params[32] = {3u, 12u, 12u, 4u, 1u, 0u, 0u, 0u};
    const uint32_t output_strides[3] = {6u, 2u, 1u};
    const uint32_t a_strides[3] = {6u, 2u, 1u};
    const uint32_t b_strides[3] = {2u, 0u, 1u};
    for (int i = 0; i < 3; i++) {
        binary_params[8 + i] = output_strides[i];
        binary_params[16 + i] = a_strides[i];
        binary_params[24 + i] = b_strides[i];
    }
    void* binary_hosts[6] = {a, b, binary_go, grad_a, grad_b, binary_params};
    size_t binary_bytes[6] = {sizeof(a), sizeof(b), sizeof(binary_go), sizeof(grad_a),
                              sizeof(grad_b), sizeof(binary_params)};
    unsigned char binary_access[6] = {
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    CHECK(vk_training_dispatch("basicBackward", "a_main", binary_hosts, binary_bytes,
                               binary_access, NULL, 6, 1, 1, 1) == 0);
    CHECK(vk_training_dispatch("basicBackward", "b_main", binary_hosts, binary_bytes,
                               binary_access, NULL, 6, 1, 1, 1) == 0);
    CHECK(vk_training_sync(grad_a, sizeof(grad_a)) == 0);
    CHECK(vk_training_sync(grad_b, sizeof(grad_b)) == 0);
    for (int batch = 0; batch < 2; batch++) for (int row = 0; row < 3; row++)
        for (int column = 0; column < 2; column++) {
            int index = (batch * 3 + row) * 2 + column;
            CHECK(close_enough(grad_a[index], b[batch * 2 + column]));
        }
    CHECK(close_enough(grad_b[0], 9.0f) && close_enough(grad_b[1], 12.0f));
    CHECK(close_enough(grad_b[2], 27.0f) && close_enough(grad_b[3], 30.0f));
    return 0;
}

static int test_groupnorm_dropout_reduce_broadcast_forward(void) {
    vk_graph_reset();
    float input[8] = {1.0f, 3.0f, 2.0f, 6.0f, 4.0f, 0.0f, 5.0f, 1.0f};
    float weight[4] = {1.0f, 0.5f, 1.5f, 0.75f};
    float bias[4] = {0.1f, -0.2f, 0.3f, 0.4f};
    float normalized[8] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_groupnorm_f32(input, weight, bias, normalized,
                                 1, 1, 2, 4, 2, 1.0e-4f) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(normalized, sizeof(normalized), 0) == 1);
    for (int group = 0; group < 2; group++) {
        float mean = 0.0f, square_mean = 0.0f;
        for (int spatial = 0; spatial < 2; spatial++) for (int local = 0; local < 2; local++) {
            float value = input[spatial * 4 + group * 2 + local];
            mean += value * 0.25f;
            square_mean += value * value * 0.25f;
        }
        float inverse = 1.0f / sqrtf(square_mean - mean * mean + 1.0e-4f);
        for (int spatial = 0; spatial < 2; spatial++) for (int local = 0; local < 2; local++) {
            int channel = group * 2 + local;
            int index = spatial * 4 + channel;
            float expected = (input[index] - mean) * inverse * weight[channel] + bias[channel];
            CHECK(close_enough(normalized[index], expected));
        }
    }

    float dropout_output[8] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_dropout_f32(input, dropout_output, 8, 0x80000000u,
                               123u, 7u, 2.0f) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(dropout_output, sizeof(dropout_output), 0) == 1);
    for (int i = 0; i < 8; i++) {
        float expected = dropout_bits(123u, 7u, (uint32_t)i) >= 0x80000000u
            ? input[i] * 2.0f : 0.0f;
        CHECK(close_enough(dropout_output[i], expected));
    }

    float a[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float b[4] = {2, 3, 5, 7};
    float product[12] = {0};
    uint32_t output_strides[8] = {6, 2, 1, 0, 0, 0, 0, 0};
    uint32_t a_strides[8] = {6, 2, 1, 0, 0, 0, 0, 0};
    uint32_t b_strides[8] = {2, 0, 1, 0, 0, 0, 0, 0};
    vk_graph_begin_forward();
    CHECK(vk_graph_binary_f32(a, 12, b, 4, product, 12, output_strides,
                              a_strides, b_strides, 3, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(product, sizeof(product), 0) == 1);
    for (int batch = 0; batch < 2; batch++) for (int row = 0; row < 3; row++)
        for (int column = 0; column < 2; column++) {
            int index = (batch * 3 + row) * 2 + column;
            CHECK(close_enough(product[index], a[index] * b[batch * 2 + column]));
        }
    float reduced[2] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_reduce_f32(a, reduced, 2, 6, 1.0f / 6.0f) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(reduced, sizeof(reduced), 0) == 1);
    CHECK(close_enough(reduced[0], 3.5f) && close_enough(reduced[1], 9.5f));
    return 0;
}

static int test_typed_control_graph_ops(void) {
    const int32_t compare_a[2] = {1, 4};
    const int32_t compare_b[3] = {1, 3, 4};
    const uint32_t output_strides[2] = {3u, 1u};
    const uint32_t a_strides[2] = {1u, 0u};
    const uint32_t b_strides[2] = {0u, 1u};
    const int32_t expected_equal[6] = {1, 0, 0, 0, 0, 1};
    const int32_t expected_ge[6] = {1, 0, 0, 1, 1, 1};
    const int32_t expected_not[6] = {0, 1, 1, 1, 1, 0};
    int32_t equal_output[6] = {0};
    int32_t ge_output[6] = {0};
    int32_t not_output[6] = {0};
    const int32_t clip_input[4] = {-4, 0, 3, 9};
    const int32_t expected_clip[4] = {0, 0, 3, 7};
    int32_t clip_output[4] = {0};
    const int32_t cast_i32[4] = {-2, 0, 7, 10};
    const float expected_cast_f32[4] = {-2.0f, 0.0f, 7.0f, 10.0f};
    float cast_f32[4] = {0};
    const float cast_float_input[13] = {
        -2.9f, 0.0f, 7.75f,
        2147483648.0f, 2147483904.0f,
        4294967296.0f, 4294967808.0f, 6442450944.0f,
        -2147483904.0f, -4294967808.0f,
        NAN, INFINITY, -INFINITY,
    };
    const int32_t expected_cast_i32[13] = {
        -2, 0, 7,
        INT32_MIN, INT32_MIN + 256,
        0, 512, INT32_MIN,
        INT32_MAX - 255, -512,
        0, 0, 0,
    };
    int32_t cast_i32_output[13] = {0};
    const uint32_t copy_input[4] = {
        0x7fc01234u, 0x80000000u, 0xffffffffu, 0x12345678u,
    };
    uint32_t copy_output[4] = {0};
    const int32_t where_condition[4] = {0, 1, -1, 0};
    const uint32_t where_a[4] = {
        0x7fc01234u, 0x80000000u, 0x11111111u, 0x22222222u,
    };
    const uint32_t where_b[4] = {
        0x33333333u, 0x44444444u, 0xffffffffu, 0x7fa00001u,
    };
    const uint32_t expected_where[4] = {
        0x33333333u, 0x80000000u, 0x11111111u, 0x7fa00001u,
    };
    uint32_t where_output[4] = {0};
    const float argmax_input[12] = {
        1.0f, 5.0f, 3.0f, 5.0f, 3.0f, 4.0f,
        -1.0f, -2.0f, -1.0f, 7.0f, -3.0f, 7.0f,
    };
    const int32_t expected_argmax[4] = {1, 0, 0, 1};
    int32_t argmax_output[4] = {0};
    const uint32_t concat_a[4] = {
        0x7fc00011u, 0x80000000u, 0x11111111u, 0x22222222u,
    };
    const uint32_t concat_b[2] = {0xffffffffu, 0x33333333u};
    const void* concat_inputs[2] = {concat_a, concat_b};
    const long concat_sizes[2] = {4, 2};
    const int concat_axes[2] = {2, 1};
    const uint32_t expected_concat[6] = {
        0x7fc00011u, 0x80000000u, 0x11111111u,
        0x22222222u, 0xffffffffu, 0x33333333u,
    };
    uint32_t concat_output[6] = {0};

    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_compare_i32(
        compare_a, 2, compare_b, 3, equal_output, 6,
        output_strides, a_strides, b_strides, 2, 0) == 1);
    CHECK(vk_graph_compare_i32(
        compare_a, 2, compare_b, 3, ge_output, 6,
        output_strides, a_strides, b_strides, 2, 1) == 1);
    CHECK(vk_graph_not_i32(equal_output, not_output, 6) == 1);
    CHECK(vk_graph_clip_i32(
        clip_input, clip_output, 4, 0, 7) == 1);
    CHECK(vk_graph_cast_typed(
        cast_i32, VX_DTYPE_I32, cast_f32, VX_DTYPE_F32, 4) == 1);
    CHECK(vk_graph_cast_typed(
        cast_float_input, VX_DTYPE_F32,
        cast_i32_output, VX_DTYPE_I32, 13) == 1);
    CHECK(vk_graph_copy_32(copy_input, copy_output, 4) == 1);
    CHECK(vk_graph_where_32(
        where_condition, where_a, where_b, where_output, 4) == 1);
    CHECK(vk_graph_argmax_f32(
        argmax_input, argmax_output, 2u, 3u, 2u) == 1);
    CHECK(vk_graph_concat_32(
        concat_inputs, concat_sizes, concat_axes, 2,
        concat_output, 3, 2) == 1);
    CHECK(vk_graph_end_forward() == 0);

    CHECK(vk_graph_sync_host(
        equal_output, sizeof(equal_output), 0) == 1);
    CHECK(vk_graph_sync_host(ge_output, sizeof(ge_output), 0) == 1);
    CHECK(vk_graph_sync_host(not_output, sizeof(not_output), 0) == 1);
    CHECK(vk_graph_sync_host(clip_output, sizeof(clip_output), 0) == 1);
    CHECK(vk_graph_sync_host(cast_f32, sizeof(cast_f32), 0) == 1);
    CHECK(vk_graph_sync_host(
        cast_i32_output, sizeof(cast_i32_output), 0) == 1);
    CHECK(vk_graph_sync_host(copy_output, sizeof(copy_output), 0) == 1);
    CHECK(vk_graph_sync_host(where_output, sizeof(where_output), 0) == 1);
    CHECK(vk_graph_sync_host(
        argmax_output, sizeof(argmax_output), 0) == 1);
    CHECK(vk_graph_sync_host(
        concat_output, sizeof(concat_output), 0) == 1);
    CHECK(memcmp(equal_output, expected_equal, sizeof(equal_output)) == 0);
    CHECK(memcmp(ge_output, expected_ge, sizeof(ge_output)) == 0);
    CHECK(memcmp(not_output, expected_not, sizeof(not_output)) == 0);
    CHECK(memcmp(clip_output, expected_clip, sizeof(clip_output)) == 0);
    CHECK(memcmp(cast_f32, expected_cast_f32, sizeof(cast_f32)) == 0);
    CHECK(memcmp(
        cast_i32_output, expected_cast_i32,
        sizeof(cast_i32_output)) == 0);
    CHECK(memcmp(copy_output, copy_input, sizeof(copy_output)) == 0);
    CHECK(memcmp(where_output, expected_where, sizeof(where_output)) == 0);
    CHECK(memcmp(
        argmax_output, expected_argmax, sizeof(argmax_output)) == 0);
    CHECK(memcmp(
        concat_output, expected_concat, sizeof(concat_output)) == 0);

    CHECK(vk_graph_not_i32(equal_output, equal_output, 6) == 0);
    CHECK(vk_graph_clip_i32(
        clip_input, clip_output, 4, 8, 7) == 0);
    CHECK(vk_graph_cast_typed(
        cast_i32, VX_DTYPE_I32,
        cast_i32_output, VX_DTYPE_U8, 4) == 0);
    CHECK(vk_graph_argmax_f32(
        argmax_input, argmax_output, 2u, 0u, 2u) == 0);
    vk_graph_reset();
    return 0;
}

static int test_expand_f32_uniform_abi(void) {
    union {
        uint32_t bits[4];
        float values[4];
    } input = {
        .bits = {0xff800000u, 0x7fc12345u, 0x80000000u, 0x3f800000u},
    };
    union {
        uint32_t bits[48];
        float values[48];
    } output;
    const int input_shape[5] = {1, 2, 1, 1, 2};
    const int output_shape[6] = {3, 1, 2, 1, 4, 2};
    const int incompatible_output_shape[6] = {3, 1, 3, 1, 4, 2};
    const int smaller_output_shape[4] = {1, 2, 1, 2};
    const int invalid_input_shape[5] = {1, 2, 0, 1, 2};
    const int scalar_shape[1] = {1};
    const int overflow_shape[2] = {INT32_MAX, 2};
    float overlap_storage[49] = {0};
    uint32_t expected[48];
    size_t index = 0;

    for (size_t leading = 0; leading < 3; leading++) {
        for (size_t row = 0; row < 2; row++) {
            for (size_t broadcast = 0; broadcast < 4; broadcast++) {
                for (size_t lane = 0; lane < 2; lane++) {
                    expected[index++] = input.bits[row * 2 + lane];
                }
            }
        }
    }
    for (index = 0; index < 48; index++) output.bits[index] = 0xdeadbeefu;

    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_expand_f32(
        input.values, output.values, input_shape, 5, output_shape, 6) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output.values, sizeof(output.values), 0) == 1);
    CHECK(memcmp(output.bits, expected, sizeof(expected)) == 0);
    vk_graph_reset();

    CHECK(vk_graph_expand_f32(
        input.values, output.values, input_shape, 5,
        incompatible_output_shape, 6) == 0);
    CHECK(vk_graph_expand_f32(
        input.values, output.values, input_shape, 5,
        smaller_output_shape, 4) == 0);
    CHECK(vk_graph_expand_f32(
        input.values, output.values, invalid_input_shape, 5,
        output_shape, 6) == 0);
    CHECK(vk_graph_expand_f32(
        input.values, output.values, scalar_shape, 1,
        overflow_shape, 2) == 0);
    CHECK(vk_graph_expand_f32(
        overlap_storage, overlap_storage, input_shape, 5,
        output_shape, 6) == 0);
    CHECK(vk_graph_expand_f32(
        overlap_storage, overlap_storage + 1, input_shape, 5,
        output_shape, 6) == 0);
    return 0;
}

static int test_dynamic_shape_capacity_lifecycle(void) {
    float small_input_a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float small_output_a[4] = {0};
    float small_input_b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float small_output_b[4] = {0};
    float large_input[128];
    float large_output[128] = {0};
    VkGraphDynamicStateProbe bound = {0};
    VkGraphDynamicStateProbe active = {0};
    VkGraphDynamicStateProbe pooled = {0};
    VkGraphDynamicStateProbe reused = {0};
    VkGraphDynamicStateProbe grown = {0};
    for (int index = 0; index < 128; index++)
        large_input[index] = (float)(index - 17);

    vk_graph_reset();
    CHECK(vk_graph_bind_shape("copy:f32:[4]->[4]") == 0);
    CHECK(vk_test_graph_dynamic_state(&bound) == 0);
    CHECK(bound.shape_generation != 0);
    CHECK(vk_graph_bind_shape("copy:f32:[4]->[4]") == 0);
    CHECK(vk_test_graph_dynamic_state(&active) == 0);
    CHECK(active.shape_generation == bound.shape_generation);
    CHECK(vk_graph_bind_shape("") == -1);
    CHECK(vk_test_graph_dynamic_state(&active) == 0);
    CHECK(active.shape_generation == bound.shape_generation);

    vk_graph_begin_forward();
    CHECK(vk_graph_copy_f32(small_input_a, small_output_a, 4) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(small_output_a, sizeof(small_output_a), 0) == 1);
    CHECK(memcmp(small_input_a, small_output_a, sizeof(small_input_a)) == 0);
    CHECK(vk_test_graph_dynamic_state(&active) == 0);
    CHECK(active.active_capacity_bytes > 0 && active.pooled_capacity_bytes == 0 &&
          active.slot_count == 2);

    CHECK(vk_graph_bind_shape("copy:f32:[1,4]->[1,4]") == 0);
    CHECK(vk_test_graph_dynamic_state(&pooled) == 0);
    CHECK(pooled.shape_generation != active.shape_generation &&
          pooled.capacity_generation == active.capacity_generation &&
          pooled.active_capacity_bytes == 0 &&
          pooled.pooled_capacity_bytes == active.active_capacity_bytes &&
          pooled.slot_count == active.slot_count);
    vk_graph_begin_forward();
    CHECK(vk_graph_copy_f32(small_input_b, small_output_b, 4) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(small_output_b, sizeof(small_output_b), 0) == 1);
    CHECK(memcmp(small_input_b, small_output_b, sizeof(small_input_b)) == 0);
    CHECK(vk_test_graph_dynamic_state(&reused) == 0);
    CHECK(reused.capacity_generation == pooled.capacity_generation &&
          reused.slot_count == pooled.slot_count &&
          reused.active_capacity_bytes == active.active_capacity_bytes);

    CHECK(vk_graph_bind_shape("copy:f32:[128]->[128]") == 0);
    vk_graph_begin_forward();
    CHECK(vk_graph_copy_f32(large_input, large_output, 128) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(large_output, sizeof(large_output), 0) == 1);
    CHECK(memcmp(large_input, large_output, sizeof(large_input)) == 0);
    CHECK(vk_test_graph_dynamic_state(&grown) == 0);
    CHECK(grown.capacity_generation > reused.capacity_generation &&
          grown.active_capacity_bytes >= sizeof(large_input) * 2u &&
          grown.slot_count == reused.slot_count);
    vk_graph_reset();
    return 0;
}

static int test_dynamic_domain_reservation(void) {
    const size_t qgroupnorm_stats_bytes = 64u;
    const size_t qlayernorm_stats_bytes = 80u;
    float arena[64] = {0};
    float bootstrap_input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float invariant[4] = {0.5f, 1.5f, 2.5f, 3.5f};
    float bootstrap_output[4] = {0};
    float missing_input[4] = {0};
    float missing_output[4] = {0};
    const VolvoxAIEnginePhysicalSpan spans[2] = {
        {arena, 32u * sizeof(float)},
        {arena + 32, 32u * sizeof(float)},
    };
    const VolvoxAIEnginePhysicalSpan mismatch[2] = {
        {arena, 31u * sizeof(float)},
        {arena + 32, 32u * sizeof(float)},
    };
    VulkanDomainLimits limits = {0};
    VkGraphDynamicStateProbe before = {0};
    VkGraphDynamicStateProbe rolled_back = {0};
    VkGraphDynamicStateProbe reserved = {0};
    VkGraphDynamicStateProbe same = {0};
    VkGraphDynamicStateProbe rebound = {0};
    VkGraphDynamicStateProbe executed = {0};

    CHECK(vk_query_domain_limits(&limits) == 0);
    CHECK(limits.maximum_storage_buffer_bytes >= spans[0].capacity_bytes &&
          limits.maximum_uniform_buffer_bytes > 0u &&
          limits.maximum_total_span_bytes >= spans[0].capacity_bytes * 2u &&
          limits.compute_arena_allocation_bytes >
              limits.maximum_total_span_bytes &&
          limits.staging_allocation_bytes >=
              (uint64_t)32u * 1024u * 1024u &&
          limits.maximum_scratch_bytes > 0u &&
          limits.current_graph_scratch_bytes <=
              limits.maximum_scratch_bytes &&
          limits.storage_alignment > 0u &&
          limits.maximum_workgroups[0] > 0u &&
          limits.maximum_workgroup_size[0] > 0u &&
          limits.maximum_workgroup_invocations > 0u &&
          limits.maximum_storage_bindings > 0u &&
          limits.maximum_uniform_bindings > 0u &&
          limits.maximum_tensor_slots >= 2u &&
          limits.maximum_dispatches > 0u);

    vk_graph_reset();
    CHECK(vk_graph_bind_shape("domain:published") == 0);
    vk_graph_begin_forward();
    CHECK(vk_graph_add_f32(
              bootstrap_input, invariant, bootstrap_output, 4) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_query_domain_limits(&limits) == 0 &&
          limits.current_graph_scratch_bytes > 0u &&
          limits.current_graph_scratch_bytes % limits.storage_alignment == 0u &&
          limits.current_graph_scratch_bytes <=
              limits.maximum_scratch_bytes);
    CHECK(vk_graph_sync_host(
              bootstrap_output, sizeof(bootstrap_output), 0) == 1);
    /* Simulate a public tensor consumed through a weight/affine port during
     * bootstrap, then apply the logical-preload completion classification. */
    vk_graph_retain_weight(bootstrap_input, sizeof(bootstrap_input));
    vk_graph_demote_weight(bootstrap_input, sizeof(bootstrap_input));
    vk_graph_retain_weight(invariant, sizeof(invariant));
    CHECK(vk_test_graph_dynamic_state(&before) == 0);
    CHECK(vk_graph_bind_shape_domain(
              "domain:q4", spans, 2u, 1024u * 1024u, 1u) == -1);
    CHECK(vk_test_graph_dynamic_state(&rolled_back) == 0);
    CHECK(rolled_back.shape_generation == before.shape_generation &&
          rolled_back.capacity_generation == before.capacity_generation &&
          rolled_back.active_capacity_bytes == before.active_capacity_bytes &&
          rolled_back.pooled_capacity_bytes == before.pooled_capacity_bytes &&
          rolled_back.domain_scratch_capacity_bytes ==
              before.domain_scratch_capacity_bytes &&
          rolled_back.domain_span_count == 0u &&
          rolled_back.slot_count == before.slot_count &&
          !rolled_back.domain_enforced);
    CHECK(vk_test_fail_domain_allocation_after(1u) == 0);
    CHECK(vk_graph_bind_shape_domain(
              "domain:q4", spans, 2u, qgroupnorm_stats_bytes,
              qlayernorm_stats_bytes) == -2);
    CHECK(vk_test_graph_dynamic_state(&rolled_back) == 0);
    CHECK(rolled_back.shape_generation == before.shape_generation &&
          rolled_back.capacity_generation == before.capacity_generation &&
          rolled_back.active_capacity_bytes == before.active_capacity_bytes &&
          rolled_back.pooled_capacity_bytes == before.pooled_capacity_bytes &&
          rolled_back.domain_scratch_capacity_bytes ==
              before.domain_scratch_capacity_bytes &&
          rolled_back.domain_span_count == 0u &&
          rolled_back.slot_count == before.slot_count &&
          !rolled_back.domain_enforced);

    CHECK(vk_graph_bind_shape_domain(
              "domain:q4", spans, 2u, qgroupnorm_stats_bytes,
              qlayernorm_stats_bytes) == 0);
    CHECK(vk_test_graph_dynamic_state(&reserved) == 0);
    CHECK(reserved.shape_generation != before.shape_generation &&
          reserved.capacity_generation != before.capacity_generation &&
          reserved.active_capacity_bytes >=
              spans[0].capacity_bytes * 2u + sizeof(invariant) &&
          reserved.pooled_capacity_bytes == 0u &&
          reserved.domain_scratch_capacity_bytes ==
              qgroupnorm_stats_bytes + qlayernorm_stats_bytes &&
          reserved.domain_span_count == 2u && reserved.slot_count == 3 &&
          reserved.domain_enforced);

    CHECK(vk_graph_bind_shape_domain(
              "domain:q4", spans, 2u, qgroupnorm_stats_bytes,
              qlayernorm_stats_bytes) == 0);
    CHECK(vk_test_graph_dynamic_state(&same) == 0);
    CHECK(same.shape_generation == reserved.shape_generation &&
          same.capacity_generation == reserved.capacity_generation &&
          same.active_capacity_bytes == reserved.active_capacity_bytes &&
          same.domain_scratch_capacity_bytes ==
              reserved.domain_scratch_capacity_bytes &&
          same.slot_count == reserved.slot_count);

    CHECK(vk_graph_bind_shape_domain(
              "domain:q8", spans, 2u, qgroupnorm_stats_bytes,
              qlayernorm_stats_bytes) == 0);
    CHECK(vk_test_graph_dynamic_state(&rebound) == 0);
    CHECK(rebound.shape_generation != same.shape_generation &&
          rebound.capacity_generation == same.capacity_generation &&
          rebound.active_capacity_bytes == same.active_capacity_bytes &&
          rebound.domain_scratch_capacity_bytes ==
              same.domain_scratch_capacity_bytes &&
          rebound.slot_count == same.slot_count);
    CHECK(vk_graph_bind_shape_domain(
              "domain:q8", spans, 2u, qgroupnorm_stats_bytes + 1u,
              qlayernorm_stats_bytes) == -1);
    CHECK(vk_graph_bind_shape_domain(
              "domain:q8", mismatch, 2u, qgroupnorm_stats_bytes,
              qlayernorm_stats_bytes) == -1);
    CHECK(vk_test_graph_dynamic_state(&executed) == 0);
    CHECK(executed.shape_generation == rebound.shape_generation &&
          executed.capacity_generation == rebound.capacity_generation &&
          executed.active_capacity_bytes == rebound.active_capacity_bytes &&
          executed.domain_scratch_capacity_bytes ==
              rebound.domain_scratch_capacity_bytes &&
          executed.slot_count == rebound.slot_count);
    CHECK(vk_graph_bind_shape("domain:legacy") == -1);

    for (int index = 0; index < 4; index++) arena[index] = (float)(index + 3);
    vk_graph_begin_forward();
    CHECK(vk_graph_add_f32(arena, invariant, arena + 32, 4) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(arena + 32, 4u * sizeof(float), 0) == 1);
    for (int index = 0; index < 4; index++)
        CHECK(arena[32 + index] == arena[index] + invariant[index]);
    vk_graph_begin_forward();
    CHECK(vk_graph_alias_f32(arena, arena + 32, 4) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(arena + 32, 4u * sizeof(float), 0) == 1);
    CHECK(memcmp(arena, arena + 32, 4u * sizeof(float)) == 0);
    vk_graph_begin_forward();
    CHECK(vk_graph_copy_f32(missing_input, missing_output, 4) == 0);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_test_graph_dynamic_state(&executed) == 0);
    CHECK(executed.capacity_generation == rebound.capacity_generation &&
          executed.active_capacity_bytes == rebound.active_capacity_bytes &&
          executed.slot_count == rebound.slot_count);
    vk_graph_reset();
    return 0;
}

/* A storage-view node (Reshape/Flatten/Squeeze/Unsqueeze/Identity/Dropout)
 * whose input and output have distinct host storage must receive independent
 * device storage. Device slots are keyed by host pointer and the runtime pools
 * many disjoint-lifetime activations into one host span, so sharing a device
 * range across two spans does not extend one tensor's lifetime -- it merges
 * the spans, and every later tensor planned into either one writes through
 * both. The regression this pins: SiLU -> Reshape -> Transpose returned a
 * different wrong answer on every run. */
static int test_storage_view_alias_isolation(void) {
    float source[8];
    float view[8];
    float successor[8];
    vk_graph_reset();
    for (int index = 0; index < 8; index++) {
        source[index] = (float)(index + 1);
        view[index] = 0.0f;
        successor[index] = -1.0f - (float)index;
    }
    vk_graph_mark_host(source, sizeof(source), 0);
    vk_graph_mark_host(successor, sizeof(successor), 0);
    vk_graph_begin_forward();
    CHECK(vk_graph_alias_f32(source, view, 8) == 1);
    /* The planner reuses the source span once the view's producer is dead. */
    CHECK(vk_graph_copy_f32(successor, source, 8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(view, sizeof(view), 0) == 1);
    for (int index = 0; index < 8; index++)
        CHECK(view[index] == (float)(index + 1));
    /* A view the runtime already aliased on the host stays a free view. */
    vk_graph_begin_forward();
    CHECK(vk_graph_alias_f32(source, source, 8) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(source, sizeof(source), 0) == 1);
    for (int index = 0; index < 8; index++)
        CHECK(source[index] == -1.0f - (float)index);
    vk_graph_reset();
    return 0;
}

int main(void) {
    VxEngineState* state;
    VxEngineStateScope state_scope;
    CHECK(test_arena_size_configuration() == 0);
    CHECK(test_context_capsule_isolation() == 0);
    state = (VxEngineState*)calloc(1, sizeof(*state));
    CHECK(state && vx_engine_state_init(state) == 0);
    state_scope = vx_engine_state_scope_enter(state);
    if (vk_init() != 0) {
        vx_engine_state_scope_leave(state_scope);
        vx_engine_state_deinit(state);
        free(state);
        puts("vulkan training tests skipped: no Vulkan compute device");
        return 77;
    }
    CHECK(vk_training_available() == 1);
    CHECK(test_device_local_staging_path() == 0);
    CHECK(test_physical_context_isolation(state) == 0);
    float host_current = 1.0f;
    vk_graph_reset();
    CHECK(vk_graph_sync_host(&host_current, sizeof(host_current), 1) == 1);
    vk_graph_mark_host(&host_current, sizeof(host_current), 1);
    CHECK(vk_graph_sync_host(&host_current, sizeof(host_current), 1) == 1);
    vk_graph_reset();
    size_t copy_bytes[3] = {sizeof(float), sizeof(float), sizeof(uint32_t)};
    size_t copy_params[1] = {sizeof(uint32_t)};
    float plan_input = 0.0f, plan_output = 0.0f;
    VkTrainingTensorRequirement copy_tensors[2] = {
        {&plan_input, sizeof(plan_input)}, {&plan_output, sizeof(plan_output)}
    };
    CHECK(vk_training_supports("copyBackward", "main", copy_bytes, 3, 1, 1, 1) == 1);
    CHECK(vk_training_supports("copyBackward", "main", copy_bytes, 2, 1, 1, 1) == 0);
    CHECK(vk_training_supports("copyBackward", "main", copy_bytes, 3,
                               UINT32_MAX, 1, 1) == 0);
    CHECK(vk_training_supports("notARealShader", "main", copy_bytes, 3, 1, 1, 1) == 0);
    CHECK(vk_training_plan_supported(1, copy_params, copy_tensors, 2) == 1);
    size_t oversized_params[1] = {2u * 1024u * 1024u};
    VkTrainingTensorRequirement oversized_tensor = {&plan_input, SIZE_MAX};
    CHECK(vk_training_plan_supported(1, oversized_params, copy_tensors, 2) == 0);
    CHECK(vk_training_plan_supported(1, copy_params, &oversized_tensor, 1) == 0);
    CHECK(vk_training_plan_supported(4097, NULL, NULL, 0) == 0);
    CHECK(test_one_shot_matmul_tails_and_fallback() == 0);
    CHECK(test_graph_linear_dynamic_domain() == 0);
    CHECK(test_tiled_qlinear_i8u8_tails() == 0);
    CHECK(test_scalar_qlinear_i8u8_fused_word() == 0);
    CHECK(test_tiled_qlinear_staged_rounding() == 0);
    CHECK(test_qbatch_matmul_i8u8_arbitrary_k() == 0);
    CHECK(test_layernorm_epsilon() == 0);
    CHECK(test_qlinear_i8u8_packed_chain() == 0);
    CHECK(test_qlinear_i8u8_scalar_fused_tail() == 0);
    CHECK(test_qembedding_i8u8_packed_gather() == 0);
    CHECK(test_qadd_requantize_i8u8_packed_chain() == 0);
    CHECK(test_qsilu_i8u8_packed_chain() == 0);
    CHECK(test_qgelu_i8u8_packed_chain() == 0);
    CHECK(test_qgroupnorm_i8u8_packed_chain() == 0);
    CHECK(test_qlayernorm_i8u8_packed_chain() == 0);
    CHECK(test_qsdpa_i8u8_packed_chain() == 0);
    CHECK(test_qargmax_i8u8_raw() == 0);
    CHECK(test_qmaskedmean_i8u8_packed() == 0);
    CHECK(test_conv2d_f32_regular_out16() == 0);
    CHECK(test_conv2d_f32_pointwise_precedes_regular_out16() == 0);
    CHECK(test_qconv2d_i8u8_packed_chain() == 0);
    CHECK(test_qconv2d_i8u8_scalar_fused_tail() == 0);
    CHECK(test_qconv2d_i8u8_scalar_tiled_tail() == 0);
    CHECK(test_typed_i8u8_shape_qdq_chain() == 0);
    CHECK(test_general_gather_i32() == 0);
    CHECK(test_typed_control_graph_ops() == 0);
    CHECK(test_expand_f32_uniform_abi() == 0);
    CHECK(test_dynamic_shape_capacity_lifecycle() == 0);
    CHECK(test_dynamic_domain_reservation() == 0);
    CHECK(test_storage_view_alias_isolation() == 0);
    CHECK(vk_training_begin() == 0);

    float dummy = 0.0f;
    void* hosts[1] = {&dummy};
    size_t bytes[1] = {sizeof(dummy)};
    unsigned char access[1] = {VK_TRAINING_READ};
    CHECK(vk_training_dispatch("unknown", "main", hosts, bytes, access, NULL,
                               1, 1, 1, 1) == -1);
    CHECK(test_activation_backward() == 0);
    CHECK(test_gelu_backward_modes() == 0);
    vk_training_end();
    CHECK(vk_training_begin() == 0);
    CHECK(test_multi_entry_matmul_backward() == 0);
    vk_training_end();
    CHECK(vk_training_begin() == 0);
    CHECK(test_batched_attention_backward() == 0);
    vk_training_end();
    CHECK(vk_training_begin() == 0);
    CHECK(test_batched_cross_attention_backward() == 0);
    vk_training_end();
    CHECK(vk_training_begin() == 0);
    CHECK(test_moe_backward_bindings() == 0);
    vk_training_end();
    CHECK(vk_training_begin() == 0);
    CHECK(test_prelu_logsoftmax_split_backward() == 0);
    vk_training_end();
    CHECK(vk_training_begin() == 0);
    CHECK(test_groupnorm_dropout_reduce_backward() == 0);
    vk_training_end();
    CHECK(test_batched_attention_forward() == 0);
    CHECK(test_gelu_forward_modes() == 0);
    CHECK(test_groupnorm_dropout_reduce_broadcast_forward() == 0);
    vk_cleanup();
    CHECK(vk_training_available() == 0);
    CHECK(vk_init() == 0);
    CHECK(vk_training_available() == 1);
    vk_cleanup();
    vx_engine_state_scope_leave(state_scope);
    vx_engine_state_deinit(state);
    free(state);
    puts("vulkan training tests passed");
    return 0;
}
