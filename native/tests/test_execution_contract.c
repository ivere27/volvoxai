#include "public_api_internal.h"
#include "cJSON.h"
#include "profiling.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static VxReport report = VX_REPORT_INIT;
#define CHECK(value) do { if (!(value)) { fprintf(stderr, "%s:%d: %s; status=%d code=%d %s\n", \
    __FILE__, __LINE__, #value, report.status, report.code, report.message); \
    fprintf(stderr, "%s\n%s\n", report.route_evidence, report.fallback_evidence); exit(1); } } while (0)

static void model_snapshot_size_test(VxRuntime* runtime) {
    const VxModelSource source = VX_MODEL_SOURCE_INIT;
    const char byte = 0;
    VxSourceBytes shard = {&byte, 64u * 1024u * 1024u - 2u};
    VxModelPackageSource package = {{&byte, 1}, &shard, 1};
    VxModel* model = NULL;
    /* Preflight must validate sizes without reading the dummy byte ranges. */
    CHECK(vx_runtime_internal_load_model_package(runtime, &source, &package,
        NULL, &report, 1) == VX_STATUS_OK);
    shard.size++;
    CHECK(vx_runtime_internal_load_model_package(runtime, &source, &package,
        NULL, &report, 1) == VX_STATUS_OK);
    shard.size++;
    CHECK(vx_runtime_internal_load_model_package(runtime, &source, &package,
        NULL, &report, 1) == VX_STATUS_OUT_OF_MEMORY);
    CHECK(report.code == VX_CODE_PACKAGE_TOO_LARGE);
    /* The private native route has no transport cap, but cannot wrap the
     * aggregate count and must reject before snapshotting either source. */
    shard.size = SIZE_MAX;
    CHECK(vx_runtime_internal_load_model_snapshot(runtime, &package,
        &model, &report) == VX_STATUS_OUT_OF_MEMORY);
    CHECK(!model && report.code == VX_CODE_PACKAGE_TOO_LARGE);
    shard.size = 0;
    CHECK(vx_runtime_internal_load_model_snapshot(runtime, &package,
        &model, &report) == VX_STATUS_INVALID_ARGUMENT);
    CHECK(!model && report.code == VX_CODE_INVALID_MODEL_SOURCE);
}

static VxTensorBinding tensor(const char* name, const float* values) {
    VxTensorBinding value = VX_TENSOR_BINDING_INIT;
    value.name = name; value.dtype = VX_DTYPE_F32; value.rank = 1;
    value.shape[0] = 2; value.data = values; value.byte_size = 2 * sizeof(float);
    value.location = VX_MEMORY_HOST;
    return value;
}

static VxStatus capture(const VxExecutionInfo* info, void* user) {
    *(VxExecutionInfo*)user = *info;
    return VX_STATUS_OK;
}

static VxExecutionInfo inspect(VxExecutionContext* context) {
    VxExecutionInfo info;
    CHECK(vx_execution_context_inspect_execution(context, capture, &info, &report) == VX_STATUS_OK);
    return info;
}

static void output_is(VxResult* result, const char* name, const float* expected) {
    float values[2]; size_t bytes = 0;
    CHECK(vx_result_read(result, name, values, sizeof(values), &bytes, &report) == VX_STATUS_OK);
    CHECK(bytes == sizeof(values) && !memcmp(values, expected, sizeof(values)));
}

static VxTensorBinding shaped(const char* name, VxDataType dtype, const void* data,
    int rank, int first, int second) {
    VxTensorBinding value = VX_TENSOR_BINDING_INIT;
    value.name = name; value.dtype = dtype; value.data = data; value.rank = (uint32_t)rank;
    value.shape[0] = first; if (rank == 2) value.shape[1] = second;
    value.byte_size = (size_t)first * (rank == 2 ? (size_t)second : 1) * 4;
    return value;
}

#include "test_full_prefix_generation.inc"
#include "test_packed_arithmetic.inc"

static void optional_state_binding_test(VxRuntime* runtime, const char* path) {
    const char* graph = "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},\"inputs\":{"
        "\"x\":{\"dtype\":\"float32\",\"shape\":[1,2]},"
        "\"lengths\":{\"dtype\":\"int32\",\"shape\":[1]},\"position\":{\"dtype\":\"int32\",\"shape\":[1]}},"
        "\"nodes\":[{\"id\":\"relu\",\"opType\":\"ReLU\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"dtype\":\"float32\",\"shape\":[1,2]}},\"params\":{}}],"
        "\"outputs\":[\"y\"],\"execution\":{\"entrypoints\":["
        "{\"name\":\"initialize\",\"effect\":2,\"allowed_ranges\":[1],\"inputs\":[\"x\",\"lengths\",\"position\"],\"outputs\":[\"y\"]},"
        "{\"name\":\"transition\",\"effect\":3,\"allowed_ranges\":[1],\"inputs\":[\"x\",\"lengths\",\"position\"],\"outputs\":[\"y\"]}]}}";
    const char* const values[] = {NULL, "\"\"", "\"control\"", "7", "true", "null", "[]", "{}"};
    VxModelSource source = VX_MODEL_SOURCE_INIT; source.graph_path = path;
    for (int binding = 0; binding < 3; binding++) {
        const char* field = binding == 1 ? "active_lengths_input" : "position_input";
        const char* control = binding == 1 ? "lengths" : "position";
        const char* state_name = binding == 2 ? "incremental" : "tensor_feedback";
        for (size_t variant = 0; variant < sizeof(values) / sizeof(values[0]); variant++) {
            cJSON* document = cJSON_Parse(graph); CHECK(document);
            cJSON* execution = cJSON_GetObjectItemCaseSensitive(document, "execution");
            cJSON* state = cJSON_Parse(binding == 2
                ? "{\"active_lengths_input\":\"lengths\",\"sequence_axis\":1,\"update_inputs\":[\"x\"]}"
                : "{\"bindings\":[{\"input_name\":\"x\",\"output_name\":\"y\"}]}");
            CHECK(state && cJSON_AddItemToObject(execution, state_name, state));
            if (binding < 2) CHECK(cJSON_AddStringToObject(state,
                binding ? "position_input" : "active_lengths_input", binding ? "position" : "lengths"));
            if (values[variant]) CHECK(cJSON_AddItemToObject(state, field,
                variant == 2 ? cJSON_CreateString(control) : cJSON_Parse(values[variant])));
            char* changed = cJSON_PrintUnformatted(document); CHECK(changed);
            FILE* fixture = fopen(path, "wb"); CHECK(fixture);
            CHECK(fwrite(changed, 1, strlen(changed), fixture) == strlen(changed)); CHECK(!fclose(fixture));
            cJSON_free(changed); cJSON_Delete(document);
            VxModel* model = NULL;
            VxStatus status = vx_runtime_load_model(runtime, &source, &model, &report);
            if (variant < 3) {
                CHECK(status == VX_STATUS_OK && model);
                vx_model_release(model);
            } else {
                /* Malformed optional controls must be refused at load, before
                 * their later typed GetModelInfo projection could fail. */
                CHECK(status == VX_STATUS_INVALID_GRAPH && !model);
                CHECK(report.code == VX_CODE_INVALID_GRAPH_CONTRACT && report.stage == VX_STAGE_MODEL_LOAD);
            }
        }
    }
    for (unsigned empty = 0; empty < 4; empty++) {
        cJSON* document = cJSON_Parse(graph); CHECK(document);
        cJSON* execution = cJSON_GetObjectItemCaseSensitive(document, "execution");
        cJSON* state = cJSON_Parse("{\"bindings\":[{\"input_name\":\"x\",\"output_name\":\"y\"}]}");
        CHECK(state && cJSON_AddItemToObject(execution, "tensor_feedback", state));
        if (empty & 1) CHECK(cJSON_AddStringToObject(state, "position_input", ""));
        if (empty & 2) CHECK(cJSON_AddStringToObject(state, "active_lengths_input", ""));
        char* changed = cJSON_PrintUnformatted(document); CHECK(changed);
        FILE* fixture = fopen(path, "wb"); CHECK(fixture);
        CHECK(fwrite(changed, 1, strlen(changed), fixture) == strlen(changed)); CHECK(!fclose(fixture));
        cJSON_free(changed); cJSON_Delete(document);
        VxModel* model = NULL;
        CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_INVALID_GRAPH && !model);
        CHECK(report.code == VX_CODE_INVALID_GRAPH_CONTRACT && report.stage == VX_STAGE_MODEL_LOAD);
    }
}

static void generation_test(VxRuntime* runtime, const char* path, const VxBackendPolicy* policy) {
    const char* graph = "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},\"inputs\":{"
        "\"tokens\":{\"dtype\":\"int32\",\"shape\":[1,6]},\"keep\":{\"dtype\":\"int32\",\"shape\":[1,6]},"
        "\"lengths\":{\"dtype\":\"int32\",\"shape\":[1]},\"embedding\":{\"dtype\":\"float32\",\"shape\":[2,6]}},"
        "\"nodes\":[{\"id\":\"embedding\",\"opType\":\"Embedding\",\"inputs\":{\"input\":\"tokens\",\"weight\":\"embedding\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"qkv\",\"shape\":[1,6,6],\"dtype\":\"float32\"}},\"params\":{}},"
        "{\"id\":\"attention\",\"opType\":\"SDPA\",\"inputs\":{\"qkv\":\"qkv\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"logits\",\"shape\":[1,6,2],\"dtype\":\"float32\"}},"
        "\"params\":{\"heads\":1,\"causal\":true}},"
        "{\"id\":\"select\",\"opType\":\"ArgMax\",\"inputs\":{\"input\":\"logits\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"ids\",\"shape\":[1,6],\"dtype\":\"int32\"}},\"params\":{\"axis\":2,\"keepdims\":false,\"select_last_index\":0}}],"
        "\"outputs\":[\"ids\",\"logits\"],\"execution\":{\"invariant_inputs\":[\"embedding\",\"lengths\"],"
        "\"incremental\":{\"active_lengths_input\":\"lengths\",\"sequence_axis\":1,\"update_inputs\":[\"tokens\",\"keep\"]},"
        "\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],"
        "\"inputs\":[\"tokens\",\"keep\",\"lengths\",\"embedding\"],\"outputs\":[\"ids\",\"logits\"]},"
        "{\"name\":\"initialize\",\"effect\":2,\"allowed_ranges\":[1,2],"
        "\"inputs\":[\"tokens\",\"keep\",\"lengths\",\"embedding\"],\"outputs\":[\"ids\",\"logits\"]},"
        "{\"name\":\"transition\",\"effect\":3,\"allowed_ranges\":[3],"
        "\"inputs\":[\"tokens\",\"keep\"],\"outputs\":[\"ids\",\"logits\"]}],"
        "\"generation\":{\"initialize_entrypoint\":\"initialize\",\"transition_entrypoint\":\"transition\","
        "\"token_input\":\"tokens\",\"keep_input\":\"keep\",\"token_output\":\"ids\",\"logits_output\":\"logits\","
        "\"logits_axis\":2,\"id_domain_size\":2,\"output_name\":\"generated\",\"sequence_axis\":1}}}";
    FILE* fixture = fopen(path, "wb"); CHECK(fixture);
    CHECK(fwrite(graph, 1, strlen(graph), fixture) == strlen(graph)); CHECK(!fclose(fixture));
    VxModelSource source = VX_MODEL_SOURCE_INIT; source.graph_path = path;
    VxModel* model = NULL; VxCompiledModel* compiled = NULL; VxExecutionContext* reference = NULL;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK);
    CHECK(vx_model_compile_options_revision(model, policy, 0, NULL, VX_PACKED_ARITHMETIC_F32, &compiled, &report) == VX_STATUS_OK);
    VxExecutionConfig config = {.context = VX_CONTEXT_OPTIONS_INIT};
    config.context.decode_row_mode = VX_DECODE_ROW_REQUIRED;
    config.context.require_incremental = 1;
    CHECK(vx_compiled_model_create_execution_context(compiled, &config, &reference, &report) == VX_STATUS_OK);
    int32_t tokens[6] = {0, 1}, keep[6] = {1, 1}, length = 2;
    float embedding[12] = {0,0,0,0,0,3, 0,0,0,0,2,0};
    VxTensorBinding inputs[] = {shaped("tokens", VX_DTYPE_I32, tokens, 2, 1, 6),
        shaped("keep", VX_DTYPE_I32, keep, 2, 1, 6), shaped("lengths", VX_DTYPE_I32, &length, 1, 1, 0),
        shaped("embedding", VX_DTYPE_F32, embedding, 2, 2, 6)};
    VxGraphExecutionOptions manual = {.entrypoint = "initialize", .range = VX_EXECUTION_RANGE_FULL};
    VxResult* result = NULL;
    manual.range = VX_EXECUTION_RANGE_PREFIX_ROWS; manual.prefix_rows = 2;
    CHECK(vx_execution_context_execute_graph(reference, inputs, 4, &manual, &result, &report) == VX_STATUS_BACKEND_UNSUPPORTED);
    CHECK(!result && !inspect(reference).initialized && inspect(reference).active_lengths[0] == 0);
    manual.range = VX_EXECUTION_RANGE_FULL;
    CHECK(vx_execution_context_execute_graph(reference, inputs, 4, &manual, &result, &report) == VX_STATUS_OK);
    int32_t expected[5], all_ids[6]; size_t bytes = 0;
    CHECK(vx_result_read(result, "ids", all_ids, sizeof(all_ids), &bytes, &report) == VX_STATUS_OK);
    expected[0] = all_ids[1]; vx_result_release(result); result = NULL;
    VxGraphExecutionOptions readonly = {.entrypoint = "evaluate", .range = VX_EXECUTION_RANGE_FULL};
    CHECK(vx_execution_context_execute_graph(reference, inputs, 4, &readonly, &result, &report) == VX_STATUS_BACKEND_UNSUPPORTED);
    CHECK(result && inspect(reference).initialized && inspect(reference).active_lengths[0] == 2);
    vx_result_release(result); result = NULL;
    VxTensorReference reuse[] = {{"lengths", NULL}, {"embedding", NULL}};
    manual.entrypoint = "transition"; manual.range = VX_EXECUTION_RANGE_SLOT_ROWS;
    manual.slot_count = 1; manual.tensors.references = reuse; manual.tensors.reference_count = 2;
    for (int32_t step = 1; step < 5; step++) {
        int32_t position = step + 1; tokens[position] = expected[step - 1]; keep[position] = 1;
        manual.slot_positions = &position;
        CHECK(vx_execution_context_execute_graph(reference, inputs, 2, &manual, &result, &report) == VX_STATUS_OK);
        CHECK(vx_result_read(result, "ids", all_ids, sizeof(all_ids), &bytes, &report) == VX_STATUS_OK);
        expected[step] = all_ids[position]; vx_result_release(result); result = NULL;
    }
    vx_execution_context_release(reference);
    VxExecutionConfig full_config = {.context = VX_CONTEXT_OPTIONS_INIT};
    CHECK(vx_compiled_model_create_execution_context(compiled, &full_config, &reference, &report) == VX_STATUS_OK);
    VxGraphExecutionOptions full = {.entrypoint = "evaluate", .range = VX_EXECUTION_RANGE_FULL};
    CHECK(vx_execution_context_execute_graph(reference, inputs, 4, &full, &result, &report) == VX_STATUS_OK);
    CHECK(vx_result_read(result, "ids", all_ids, sizeof(all_ids), &bytes, &report) == VX_STATUS_OK);
    CHECK(!memcmp(expected, all_ids + 1, sizeof(expected)));
    vx_result_release(result); result = NULL; vx_execution_context_release(reference);
    for (int owned = 0; owned < 2; owned++) for (int chunk = 1; chunk <= 3; chunk += 2) {
        VxTrace* trace = vx_trace_create(1024 * 1024, VX_TRACE_DETAIL_BASIC, 0, 1, 0, 0, 0);
        CHECK(trace && vx_runtime_trace_start(runtime, trace) == VX_STATUS_OK);
        VxExecutionContext* context = NULL;
        config.autoregressive = 1; config.max_steps = 5; config.sampling = VX_SAMPLING_GREEDY;
        CHECK(vx_compiled_model_create_execution_context(compiled, &config, &context, &report) == VX_STATUS_OK);
        memset(tokens, 0, sizeof(tokens)); tokens[1] = 1;
        memset(keep, 0, sizeof(keep)); keep[0] = keep[1] = 1;
        int emitted = 0;
        while (emitted < 5) {
            VxGenerationOptions options = {.initial_inputs_present = emitted == 0, .max_steps = (uint32_t)chunk};
            options.tensors.owned_output = owned;
            CHECK(vx_execution_context_generate(context, emitted ? NULL : inputs, emitted ? 0 : 4,
                &options, &result, &report) == VX_STATUS_OK);
            int32_t actual[3] = {0};
            CHECK(vx_result_read(result, "generated", actual, sizeof(actual), &bytes, &report) == VX_STATUS_OK);
            int count = (int)(bytes / sizeof(int32_t)); CHECK(count > 0 && count <= chunk);
            CHECK(!memcmp(actual, expected + emitted, bytes)); emitted += count;
            VxExecutionInfo info; vx_result_execution_info(result, &info);
            CHECK(info.call_steps == (uint64_t)count && info.total_steps == (uint64_t)emitted && info.initialized);
            vx_result_release(result); result = NULL;
        }
        CHECK(inspect(context).state == VX_EXECUTION_CONTEXT_STATE_COMPLETED);
        CHECK(inspect(context).active_lengths[0] == 6);
        vx_execution_context_release(context);
        vx_runtime_trace_stop(runtime, trace);
        VxTraceView view; vx_trace_view(trace, &view);
        const VxAllocatorMemory* memory = &view.allocators[VX_MEMORY_RESULT_HOST];
        CHECK(view.state == VX_TRACE_STATE_READY && !view.dropped && !memory->dropped);
        CHECK(memory->observed && memory->complete && memory->allocated > 0);
        CHECK(memory->allocated == memory->freed && !memory->existing && !memory->live);
        if (!owned) CHECK(memory->allocated == sizeof(expected));
        vx_trace_release(trace);
    }
    /* The causal zero-Q/K fixture has analytically uniform attention. Its
     * logits are [2 * count(1) / length, 3 * count(0) / length]. These fixed
     * draws and expected IDs were computed independently of the C sampler. */
    const int32_t random_expected[][5] = {{0,1,1,1,0}, {1,0,0,1,1}, {1,0,0,0,1}};
    for (int variant = 0; variant < 5; variant++) for (int chunk = 1; chunk <= 5; chunk += 4) {
        VxExecutionContext* context = NULL;
        VxExecutionConfig sampled = {.context = VX_CONTEXT_OPTIONS_INIT, .autoregressive = 1,
            .max_steps = 5, .sampling = VX_SAMPLING_RANDOM, .has_seed = variant != 0, .seed = 7};
        if (variant == 2) { sampled.has_temperature = 1; sampled.temperature = 1000; }
        if (variant == 3) { sampled.has_top_k = 1; sampled.top_k = 1; }
        if (variant == 4) { sampled.has_top_p = 1; sampled.top_p = 0.4f; }
        CHECK(vx_compiled_model_create_execution_context(compiled, &sampled, &context, &report) == VX_STATUS_OK);
        memset(tokens, 0, sizeof(tokens)); tokens[1] = 1;
        memset(keep, 0, sizeof(keep)); keep[0] = keep[1] = 1;
        VxGenerationOptions oversized = {.initial_inputs_present = 1, .max_steps = 5};
        int32_t invalid_length = 0; inputs[2].data = &invalid_length;
        CHECK(vx_execution_context_generate(context, inputs, 4, &oversized, &result, &report) == VX_STATUS_INVALID_ARGUMENT);
        CHECK(!inspect(context).initialized && inspect(context).state == VX_EXECUTION_CONTEXT_STATE_READY);
        vx_result_release(result); result = NULL; inputs[2].data = &length;
        int emitted = 0;
        while (emitted < 5) {
            VxGenerationOptions options = {.initial_inputs_present = emitted == 0, .max_steps = (uint32_t)chunk};
            CHECK(vx_execution_context_generate(context, emitted ? NULL : inputs, emitted ? 0 : 4,
                &options, &result, &report) == VX_STATUS_OK);
            int32_t actual[5];
            CHECK(vx_result_read(result, "generated", actual, sizeof(actual), &bytes, &report) == VX_STATUS_OK);
            const int32_t* expected_ids = variant < 3 ? random_expected[variant] : expected;
            CHECK(bytes && !memcmp(actual, expected_ids + emitted, bytes)); emitted += (int)(bytes / sizeof(int32_t));
            vx_result_release(result); result = NULL;
        }
        CHECK(inspect(context).total_steps == 5 && inspect(context).stop_reason == VX_EXECUTION_STOP_STEP_LIMIT);
        vx_execution_context_release(context);
    }
    /* Interleaved contexts own independent RNG streams, including a uint64
     * seed with all bits set; neither another context nor a refused binding
     * can consume a draw in this context. */
    VxExecutionContext* independent[2] = {NULL, NULL};
    const int32_t independent_expected[][5] = {{1,0,0,1,1}, {0,0,1,1,0}};
    for (int lane = 0; lane < 2; lane++) {
        VxExecutionConfig sampled = {.context = VX_CONTEXT_OPTIONS_INIT, .autoregressive = 1,
            .max_steps = 5, .sampling = VX_SAMPLING_RANDOM, .has_seed = 1, .seed = lane ? UINT64_MAX : 7};
        CHECK(vx_compiled_model_create_execution_context(compiled, &sampled, &independent[lane], &report) == VX_STATUS_OK);
    }
    memset(tokens, 0, sizeof(tokens)); tokens[1] = 1;
    memset(keep, 0, sizeof(keep)); keep[0] = keep[1] = 1;
    for (int step = 0; step < 5; step++) for (int lane = 0; lane < 2; lane++) {
        VxGenerationOptions options = {.initial_inputs_present = step == 0, .max_steps = 1};
        CHECK(vx_execution_context_generate(independent[lane], step ? NULL : inputs, step ? 0 : 4,
            &options, &result, &report) == VX_STATUS_OK);
        int32_t actual;
        CHECK(vx_result_read(result, "generated", &actual, sizeof(actual), &bytes, &report) == VX_STATUS_OK);
        CHECK(bytes == sizeof(actual) && actual == independent_expected[lane][step]);
        vx_result_release(result); result = NULL;
    }
    for (int lane = 0; lane < 2; lane++) vx_execution_context_release(independent[lane]);
    for (int stop = 0; stop < 4; stop++) {
        int32_t stop_id = 1, sequence_ids[] = {0,1};
        VxGenerationIdSequence sequence = {sequence_ids, 2};
        VxExecutionConfig stopped = {.context = VX_CONTEXT_OPTIONS_INIT, .autoregressive = 1,
            .max_steps = 5, .sampling = VX_SAMPLING_GREEDY};
        if (stop == 0) { stopped.stop_ids = &stop_id; stopped.stop_id_count = 1; }
        if (stop == 1) { stopped.stop_sequences = &sequence; stopped.stop_sequence_count = 1; }
        if (stop >= 2) { stopped.eos_ids = &stop_id; stopped.eos_id_count = 1; }
        if (stop == 3) { stopped.stop_ids = &stop_id; stopped.stop_id_count = 1; }
        VxExecutionContext* context = NULL;
        CHECK(vx_compiled_model_create_execution_context(compiled, &stopped, &context, &report) == VX_STATUS_OK);
        /* Caller-owned policies are copied. The prompt itself ends in [0,1],
         * but only newly generated IDs participate in the sequence match. */
        stop_id = 0; sequence_ids[0] = sequence_ids[1] = 1;
        memset(tokens, 0, sizeof(tokens)); tokens[1] = 1;
        memset(keep, 0, sizeof(keep)); keep[0] = keep[1] = 1;
        int emitted = 0, terminal_count = stop == 1 ? 3 : 1;
        while (emitted < terminal_count) {
            VxGenerationOptions options = {.initial_inputs_present = emitted == 0, .max_steps = stop == 1 ? 1 : 5};
            CHECK(vx_execution_context_generate(context, emitted ? NULL : inputs, emitted ? 0 : 4,
                &options, &result, &report) == VX_STATUS_OK);
            int32_t actual[5];
            CHECK(vx_result_read(result, "generated", actual, sizeof(actual), &bytes, &report) == VX_STATUS_OK);
            CHECK(bytes && !memcmp(actual, expected + emitted, bytes)); emitted += (int)(bytes / sizeof(int32_t));
            VxExecutionInfo progress; vx_result_execution_info(result, &progress);
            CHECK(progress.total_steps == (uint64_t)emitted);
            vx_result_release(result); result = NULL;
        }
        CHECK(emitted == terminal_count && inspect(context).state == VX_EXECUTION_CONTEXT_STATE_COMPLETED);
        CHECK(inspect(context).stop_reason == (stop >= 2 ? VX_EXECUTION_STOP_EOS : VX_EXECUTION_STOP_ID_STOP));
        VxGenerationOptions again = {.max_steps = 1};
        CHECK(vx_execution_context_generate(context, NULL, 0, &again, &result, &report) == VX_STATUS_INVALID_ARGUMENT);
        CHECK(!result && inspect(context).total_steps == (uint64_t)terminal_count);
        vx_execution_context_release(context);
    }
    for (int invalid = 0; invalid < 10; invalid++) {
        int32_t invalid_id = 2;
        VxGenerationIdSequence empty = {NULL, 0};
        VxExecutionConfig refused = {.context = VX_CONTEXT_OPTIONS_INIT, .autoregressive = 1,
            .max_steps = 5, .sampling = VX_SAMPLING_RANDOM};
        if (invalid < 3) { refused.has_temperature = 1; refused.temperature = invalid == 0 ? 0 : invalid == 1 ? NAN : INFINITY; }
        if (invalid >= 3 && invalid < 6) { refused.has_top_p = 1; refused.top_p = invalid == 3 ? 0 : invalid == 4 ? NAN : 1.1f; }
        if (invalid == 6) { refused.has_top_k = 1; refused.top_k = 3; }
        if (invalid == 7) { refused.stop_ids = &invalid_id; refused.stop_id_count = 1; }
        if (invalid == 8) { refused.stop_sequences = &empty; refused.stop_sequence_count = 1; }
        if (invalid == 9) { refused.sampling = VX_SAMPLING_GREEDY; refused.has_seed = 1; }
        VxExecutionContext* context = NULL;
        CHECK(vx_compiled_model_create_execution_context(compiled, &refused, &context, &report) == VX_STATUS_INVALID_ARGUMENT);
        CHECK(!context && report.stage == VX_STAGE_CONTEXT_CREATE);
    }
    vx_compiled_model_release(compiled); vx_model_release(model);
    for (int invalid = 0; invalid < 9; invalid++) {
        cJSON* document = cJSON_Parse(graph); CHECK(document);
        cJSON* execution = cJSON_GetObjectItemCaseSensitive(document, "execution");
        cJSON* entries = cJSON_GetObjectItemCaseSensitive(execution, "entrypoints");
        cJSON* incremental = cJSON_GetObjectItemCaseSensitive(execution, "incremental");
        if (!invalid) {
            cJSON* node = cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(document, "nodes"), 2);
            CHECK(cJSON_ReplaceItemInObjectCaseSensitive(node, "opType", cJSON_CreateString("Identity")));
            CHECK(cJSON_ReplaceItemInObjectCaseSensitive(cJSON_GetObjectItemCaseSensitive(node, "inputs"),
                "input", cJSON_CreateString("tokens")));
            CHECK(cJSON_ReplaceItemInObjectCaseSensitive(node, "params", cJSON_CreateObject()));
        } else if (invalid == 1) {
            cJSON* embedding = cJSON_GetObjectItemCaseSensitive(
                cJSON_GetObjectItemCaseSensitive(document, "inputs"), "embedding");
            CHECK(cJSON_ReplaceItemInArray(cJSON_GetObjectItemCaseSensitive(embedding, "shape"), 0, cJSON_CreateNumber(3)));
        } else if (invalid <= 3) {
            CHECK(cJSON_ReplaceItemInObjectCaseSensitive(incremental, "update_inputs", cJSON_Parse("[\"keep\"]")));
            if (invalid == 3) CHECK(cJSON_AddItemToArray(
                cJSON_GetObjectItemCaseSensitive(execution, "invariant_inputs"), cJSON_CreateString("tokens")));
        } else if (invalid == 4) {
            CHECK(cJSON_ReplaceItemInObjectCaseSensitive(cJSON_GetArrayItem(entries, 1), "inputs",
                cJSON_Parse("[\"tokens\",\"keep\",\"lengths\"]")));
        } else if (invalid == 5) {
            CHECK(cJSON_ReplaceItemInObjectCaseSensitive(cJSON_GetArrayItem(entries, 2), "inputs", cJSON_Parse("[\"keep\"]")));
        } else if (invalid < 8) {
            CHECK(cJSON_ReplaceItemInObjectCaseSensitive(cJSON_GetArrayItem(entries, invalid == 6 ? 1 : 2),
                "allowed_ranges", cJSON_Parse(invalid == 6 ? "[2]" : "[1]")));
        } else {
            CHECK(cJSON_ReplaceItemInObjectCaseSensitive(document, "dimensions", cJSON_Parse("{\"S\":{\"min\":1,\"max\":6}}")));
            cJSON* model_inputs = cJSON_GetObjectItemCaseSensitive(document, "inputs");
            for (int i = 0; i < 2; i++) CHECK(cJSON_ReplaceItemInArray(cJSON_GetObjectItemCaseSensitive(
                cJSON_GetObjectItemCaseSensitive(model_inputs, i ? "keep" : "tokens"), "shape"), 1, cJSON_CreateString("S")));
            cJSON* nodes = cJSON_GetObjectItemCaseSensitive(document, "nodes");
            for (int i = 0; i < 3; i++) CHECK(cJSON_ReplaceItemInArray(cJSON_GetObjectItemCaseSensitive(
                cJSON_GetObjectItemCaseSensitive(cJSON_GetObjectItemCaseSensitive(cJSON_GetArrayItem(nodes, i),
                    "outputs"), "out"), "shape"), 1, cJSON_CreateString("S")));
        }
        char* changed = cJSON_PrintUnformatted(document); CHECK(changed);
        fixture = fopen(path, "wb"); CHECK(fixture);
        CHECK(fwrite(changed, 1, strlen(changed), fixture) == strlen(changed)); CHECK(!fclose(fixture));
        cJSON_free(changed); cJSON_Delete(document);
        model = NULL; compiled = NULL; reference = NULL;
        CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK);
        CHECK(vx_model_compile_options_revision(model, policy, 0, NULL, VX_PACKED_ARITHMETIC_F32, &compiled, &report) == VX_STATUS_OK);
        VxStatus status = vx_compiled_model_create_execution_context(compiled, &config, &reference, &report);
        if (invalid < 8) CHECK(status == VX_STATUS_BACKEND_UNSUPPORTED && !reference && report.stage == VX_STAGE_CONTEXT_CREATE);
        else {
            CHECK(status == VX_STATUS_OK && reference && !inspect(reference).initialized);
            VxGenerationOptions options = {.initial_inputs_present = 1, .max_steps = 1};
            CHECK(vx_execution_context_generate(reference, inputs, 4, &options, &result, &report) == VX_STATUS_OK);
            int32_t actual;
            CHECK(vx_result_read(result, "generated", &actual, sizeof(actual), &bytes, &report) == VX_STATUS_OK);
            CHECK(bytes == sizeof(actual) && actual == expected[0]);
            vx_result_release(result); result = NULL;
            vx_execution_context_release(reference);
        }
        vx_compiled_model_release(compiled); vx_model_release(model);
    }
}

static uint64_t prepared_engine_key(VxCompiledModel* compiled, VxExecutionContext* context) {
    VxMemoryAllocation allocations[256];
    VxMemoryInventory inventory = {.items = allocations, .capacity = 256};
    VxMemoryView view;
    if (context) CHECK(vx_context_memory_inventory(context, &inventory, &view) == VX_STATUS_OK);
    else vx_compiled_memory_inventory(compiled, &inventory);
    CHECK(!inventory.truncated);
    uint64_t key = 0;
    for (size_t i = 0; i < inventory.count; ++i)
        if (!strcmp(allocations[i].name, "engine state")) {
            CHECK(!key);
            key = allocations[i].key;
        }
    return key;
}

static void cuda_prefix_refusal_case(VxRuntime* runtime, const char* path, int dynamic) {
    char graph[1024];
    int length = snprintf(graph, sizeof(graph), "{\"format\":\"volvox-graph/v1\",\"dimensions\":%s,"
        "\"inputs\":{\"x\":{\"dtype\":\"float32\",\"shape\":[1,%s,2]}},"
        "\"nodes\":[{\"id\":\"twice\",\"opType\":\"Add\",\"inputs\":{\"a\":\"x\",\"b\":\"x\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"dtype\":\"float32\",\"shape\":[1,%s,2]}},\"params\":{}}],"
        "\"outputs\":[\"y\"],\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\","
        "\"effect\":1,\"allowed_ranges\":[1,2],\"inputs\":[\"x\"],\"outputs\":[\"y\"]}]}}",
        dynamic ? "{\"S\":{\"min\":1,\"max\":4}}" : "{}",
        dynamic ? "\"S\"" : "4", dynamic ? "\"S\"" : "4");
    CHECK(length > 0 && (size_t)length < sizeof(graph));
    FILE* fixture = fopen(path, "wb"); CHECK(fixture);
    CHECK(fwrite(graph, 1, strlen(graph), fixture) == strlen(graph)); CHECK(!fclose(fixture));
    VxModelSource source = VX_MODEL_SOURCE_INIT; source.graph_path = path;
    const char* backend = "cuda";
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    policy.mode = VX_BACKEND_REQUIRE; policy.backends = &backend; policy.backend_count = 1;
    policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    VxModel* model = NULL; VxCompiledModel* compiled = NULL;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK);
    CHECK(vx_model_compile_options_revision(model, &policy, 0, NULL, VX_PACKED_ARITHMETIC_F32, &compiled, &report) == VX_STATUS_OK);
    VxExecutionConfig config = {.context = VX_CONTEXT_OPTIONS_INIT};
    const float values[8] = {1, 2, 3, 4, 5, 6, 7, 8}, expected[8] = {2, 4, 6, 8, 10, 12, 14, 16};
    const float changed[8] = {20, 21, 22, 23, 24, 25, 26, 27};
    for (int owned = 0; owned < 2; ++owned) {
        VxExecutionContext* context = NULL;
        VxResult *result = NULL, *retained = NULL;
        CHECK(vx_compiled_model_create_execution_context(compiled, &config, &context, &report) == VX_STATUS_OK);
        VxTensorBinding input = shaped("x", VX_DTYPE_F32, values, 1, 8, 0);
        input.rank = 3; input.shape[0] = 1; input.shape[1] = 4; input.shape[2] = 2;
        VxGraphExecutionOptions options = {.entrypoint = "evaluate", .range = VX_EXECUTION_RANGE_PREFIX_ROWS,
            .prefix_rows = 2, .tensors = {.owned_output = owned}};
        VxExecutionInfo before = inspect(context);
        CHECK(vx_execution_context_execute_graph(context, &input, 1, &options, &result, &report) == VX_STATUS_BACKEND_UNSUPPORTED);
        CHECK(!result && report.stage == VX_STAGE_EXECUTE && report.code == VX_CODE_BACKEND_UNSUPPORTED &&
            strstr(report.message, "full-window preservation"));
        VxExecutionInfo after = inspect(context);
        CHECK(!memcmp(&before, &after, sizeof(before)) && !after.bindings_available);
        options.range = VX_EXECUTION_RANGE_FULL;
        CHECK(vx_execution_context_execute_graph(context, &input, 1, &options, &retained, &report) == VX_STATUS_OK);
        before = inspect(context);
        CHECK(before.bindings_available);
        float actual[8]; size_t bytes;
        for (int variant = 0; variant < (dynamic ? 2 : 1); ++variant) {
            input.data = changed;
            input.shape[1] = variant ? 2 : 4;
            input.byte_size = (size_t)input.shape[1] * 2u * sizeof(float);
            options.range = VX_EXECUTION_RANGE_PREFIX_ROWS; options.prefix_rows = 1;
            options.tensors.references = NULL; options.tensors.reference_count = 0;
            CHECK(vx_execution_context_execute_graph(context, &input, 1, &options, &result, &report) == VX_STATUS_BACKEND_UNSUPPORTED);
            CHECK(!result && report.stage == VX_STAGE_EXECUTE && report.code == VX_CODE_BACKEND_UNSUPPORTED);
            after = inspect(context); CHECK(!memcmp(&before, &after, sizeof(before)));
            CHECK(vx_result_read(retained, "y", actual, sizeof(actual), &bytes, &report) == VX_STATUS_OK);
            CHECK(bytes == sizeof(expected) && !memcmp(actual, expected, sizeof(expected)));
            VxTensorReference reuse = {"x", NULL};
            options.range = VX_EXECUTION_RANGE_FULL;
            options.tensors.references = &reuse; options.tensors.reference_count = 1;
            CHECK(vx_execution_context_execute_graph(context, NULL, 0, &options, &result, &report) == VX_STATUS_OK);
            CHECK(vx_result_read(result, "y", actual, sizeof(actual), &bytes, &report) == VX_STATUS_OK);
            CHECK(bytes == sizeof(expected) && !memcmp(actual, expected, sizeof(expected)));
            vx_result_release(result); result = NULL;
            before = inspect(context);
        }
        vx_execution_context_release(context);
        CHECK(!prepared_engine_key(compiled, NULL)); /* PREFIX-capable owners never cache an engine. */
        CHECK(vx_result_read(retained, "y", actual, sizeof(actual), &bytes, &report) == VX_STATUS_OK);
        CHECK(bytes == sizeof(expected) && !memcmp(actual, expected, sizeof(expected)));
        vx_result_release(retained);
    }
    vx_compiled_model_release(compiled); vx_model_release(model);
}

static void cuda_prefix_refusal_test(VxRuntime* runtime, const char* path) {
    cuda_prefix_refusal_case(runtime, path, 0);
    cuda_prefix_refusal_case(runtime, path, 1);
    puts("CUDA prefix refusal: passed");
}

/* Run explicitly on a supported device; the normal inference/full CPU gate
 * remains hardware independent. Inventory proves exclusive engine transfer,
 * while retained values and first-binding refusal prove public isolation. */
static void prepared_native_engine_test(VxRuntime* runtime, const char* path) {
    const char* backend = getenv("VOLVOXAI_TEST_PREPARED_BACKEND");
    if (!backend || !backend[0]) return;
    const char* graph = "{\"format\":\"volvox-graph/v1\",\"dimensions\":{\"S\":{\"min\":1,\"max\":4}},"
        "\"inputs\":{\"x\":{\"dtype\":\"float32\",\"shape\":[\"S\"]}},"
        "\"nodes\":[{\"id\":\"twice\",\"opType\":\"Add\",\"inputs\":{\"a\":\"x\",\"b\":\"x\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"dtype\":\"float32\",\"shape\":[\"S\"]}},\"params\":{}}],"
        "\"outputs\":[\"y\"],\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\","
        "\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\"],\"outputs\":[\"y\"]}]}}";
    FILE* fixture = fopen(path, "wb"); CHECK(fixture);
    CHECK(fwrite(graph, 1, strlen(graph), fixture) == strlen(graph)); CHECK(!fclose(fixture));
    VxModelSource source = VX_MODEL_SOURCE_INIT; source.graph_path = path;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    policy.mode = VX_BACKEND_REQUIRE; policy.backends = &backend; policy.backend_count = 1;
    policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    VxModel* model = NULL; VxCompiledModel* compiled = NULL;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK);
    CHECK(vx_model_compile_options_revision(model, &policy, 0, NULL, VX_PACKED_ARITHMETIC_F32, &compiled, &report) == VX_STATUS_OK);
    VxExecutionConfig config = {.context = VX_CONTEXT_OPTIONS_INIT};
    VxGraphExecutionOptions options = {.entrypoint = "evaluate", .range = VX_EXECUTION_RANGE_FULL};
    VxExecutionContext *first = NULL, *second = NULL, *parallel = NULL;
    VxResult *retained = NULL, *result = NULL;
    CHECK(vx_compiled_model_create_execution_context(compiled, &config, &first, &report) == VX_STATUS_OK);
    uint64_t first_id = report.context_id;
    uint64_t first_engine = prepared_engine_key(compiled, first); CHECK(first_engine);
    float a[2] = {1, 2}, b[4] = {3, 4, 5, 6}, actual[4]; size_t bytes;
    const float expected_a[2] = {2, 4}, expected_b[4] = {6, 8, 10, 12};
    VxTensorBinding input = shaped("x", VX_DTYPE_F32, a, 1, 2, 0);
    CHECK(vx_execution_context_execute_graph(first, &input, 1, &options, &retained, &report) == VX_STATUS_OK);
    vx_execution_context_release(first);
    CHECK(prepared_engine_key(compiled, NULL) == first_engine);
    CHECK(vx_compiled_model_create_execution_context(compiled, &config, &second, &report) == VX_STATUS_OK);
    CHECK(report.context_id != first_id && !inspect(second).initialized && !inspect(second).bindings_available);
    CHECK(!prepared_engine_key(compiled, NULL));
    CHECK(prepared_engine_key(compiled, second) == first_engine);
    VxTensorReference reuse = {"x", NULL};
    options.tensors.references = &reuse; options.tensors.reference_count = 1;
    CHECK(vx_execution_context_execute_graph(second, NULL, 0, &options, &result, &report) != VX_STATUS_OK);
    vx_result_release(result); result = NULL;
    options.tensors = (VxTensorRunOptions){0};
    input = shaped("x", VX_DTYPE_F32, b, 1, 4, 0);
    CHECK(vx_execution_context_execute_graph(second, &input, 1, &options, &result, &report) == VX_STATUS_OK);
    CHECK(vx_result_read(result, "y", actual, sizeof(actual), &bytes, &report) == VX_STATUS_OK);
    CHECK(bytes == sizeof(expected_b) && !memcmp(actual, expected_b, sizeof(expected_b)));
    vx_result_release(result); result = NULL;
    output_is(retained, "y", expected_a);
    CHECK(vx_compiled_model_create_execution_context(compiled, &config, &parallel, &report) == VX_STATUS_OK);
    CHECK(prepared_engine_key(compiled, parallel) != first_engine);
    CHECK(vx_execution_context_execute_graph(parallel, &input, 1, &options, &result, &report) == VX_STATUS_OK);
    vx_result_release(result); result = NULL;
    vx_execution_context_release(second);
    vx_execution_context_release(parallel); /* Only the first idle engine fits. */
    CHECK(prepared_engine_key(compiled, NULL) == first_engine);
    CHECK(vx_compiled_model_create_execution_context(compiled, &config, &second, &report) == VX_STATUS_OK);
    CHECK(prepared_engine_key(compiled, second) == first_engine);
    CHECK(vx_execution_context_execute_graph(second, &input, 1, &options, &result, &report) == VX_STATUS_OK);
    vx_result_release(result); result = NULL;
    CHECK(vx_execution_context_cancel_execution(second, &report) == VX_STATUS_OK);
    vx_execution_context_release(second);
    CHECK(!prepared_engine_key(compiled, NULL));
    /* Profiling attached after checkout must observe actual retirement, not
     * transfer traced allocation identities into another public context. */
    CHECK(vx_compiled_model_create_execution_context(compiled, &config, &second, &report) == VX_STATUS_OK);
    CHECK(vx_execution_context_execute_graph(second, &input, 1, &options, &result, &report) == VX_STATUS_OK);
    vx_result_release(result); result = NULL; vx_execution_context_release(second);
    CHECK(prepared_engine_key(compiled, NULL));
    CHECK(vx_compiled_model_create_execution_context(compiled, &config, &second, &report) == VX_STATUS_OK);
    VxTrace* trace = vx_trace_create(1024 * 1024, VX_TRACE_DETAIL_BASIC, 0, 1, 0, 0, 0);
    CHECK(trace && vx_runtime_trace_start(runtime, trace) == VX_STATUS_OK);
    CHECK(vx_execution_context_execute_graph(second, &input, 1, &options, &result, &report) == VX_STATUS_OK);
    vx_result_release(result); result = NULL; vx_execution_context_release(second);
    CHECK(!prepared_engine_key(compiled, NULL));
    vx_runtime_trace_stop(runtime, trace);
    VxTraceView view; vx_trace_view(trace, &view);
    CHECK(view.state == VX_TRACE_STATE_READY && !view.dropped);
    for (int i = 0; i < VX_MEMORY_ALLOCATOR_COUNT; ++i) {
        const VxAllocatorMemory* memory = &view.allocators[i];
        CHECK(!memory->observed || (memory->complete && !memory->dropped && !memory->live &&
            memory->existing + memory->allocated == memory->freed));
    }
    vx_trace_release(trace);
    /* Last-owner retirement destroys its idle engine, while an old result
     * remains independently readable after model and compiled release. */
    CHECK(vx_compiled_model_create_execution_context(compiled, &config, &second, &report) == VX_STATUS_OK);
    CHECK(vx_execution_context_execute_graph(second, &input, 1, &options, &result, &report) == VX_STATUS_OK);
    vx_result_release(result); vx_execution_context_release(second);
    CHECK(prepared_engine_key(compiled, NULL));
    vx_compiled_model_release(compiled); vx_model_release(model);
    output_is(retained, "y", expected_a); vx_result_release(retained);
    printf("prepared native engine (%s): passed\n", backend);
    if (!strcmp(backend, "cuda")) cuda_prefix_refusal_test(runtime, path);
}

/* Public behavior regressions for immutable logical definitions. These tests
 * neither inspect a cache flag nor depend on allocation implementation. */
static const char logical_valid_graph[] =
    "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},\"inputs\":{\"x\":{"
    "\"dtype\":\"float32\",\"shape\":[2]}},\"nodes\":[{\"id\":\"twice\","
    "\"opType\":\"Add\",\"inputs\":{\"a\":\"x\",\"b\":\"x\"},\"outputs\":{\"out\":{"
    "\"tensor\":\"twice\",\"dtype\":\"float32\",\"shape\":[2]}},\"params\":{}},{"
    "\"id\":\"relu\",\"opType\":\"ReLU\",\"inputs\":{\"input\":\"twice\"},\"outputs\":{"
    "\"out\":{\"tensor\":\"y\",\"dtype\":\"float32\",\"shape\":[2]}},\"params\":{}}],"
    "\"outputs\":[\"y\"],\"execution\":{\"stateless\":{},\"entrypoints\":[{"
    "\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\"],"
    "\"outputs\":[\"y\"]}]}}";
static const struct { const char* name; const char* graph; } logical_invalid_cases[] = {
    {"nested-input-dtype-key",
        "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},\"inputs\":{\"x\":{"
        "\"dtype\":\"float32\",\"dtype\":\"float32\",\"shape\":[2]}},\"nodes\":[{"
        "\"id\":\"twice\",\"opType\":\"Add\",\"inputs\":{\"a\":\"x\",\"b\":\"x\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"twice\",\"dtype\":\"float32\",\"shape\":[2]"
        "}},\"params\":{}},{\"id\":\"relu\",\"opType\":\"ReLU\",\"inputs\":{"
        "\"input\":\"twice\"},\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[2]}},\"params\":{}}],\"outputs\":[\"y\"],"
        "\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\","
        "\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\"],\"outputs\":[\"y\"]}]}}"},
    {"nested-node-input-port-key",
        "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},\"inputs\":{\"x\":{"
        "\"dtype\":\"float32\",\"shape\":[2]}},\"nodes\":[{\"id\":\"twice\","
        "\"opType\":\"Add\",\"inputs\":{\"a\":\"x\",\"a\":\"x\",\"b\":\"x\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"twice\",\"dtype\":\"float32\",\"shape\":[2]"
        "}},\"params\":{}},{\"id\":\"relu\",\"opType\":\"ReLU\",\"inputs\":{"
        "\"input\":\"twice\"},\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[2]}},\"params\":{}}],\"outputs\":[\"y\"],"
        "\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\","
        "\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\"],\"outputs\":[\"y\"]}]}}"},
    {"nested-node-output-tensor-key",
        "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},\"inputs\":{\"x\":{"
        "\"dtype\":\"float32\",\"shape\":[2]}},\"nodes\":[{\"id\":\"twice\","
        "\"opType\":\"Add\",\"inputs\":{\"a\":\"x\",\"b\":\"x\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"twice\",\"tensor\":\"twice\",\"dtype\":\"float32\",\"shape\":[2]}"
        "},\"params\":{}},{\"id\":\"relu\",\"opType\":\"ReLU\",\"inputs\":{"
        "\"input\":\"twice\"},\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[2]}},\"params\":{}}],\"outputs\":[\"y\"],"
        "\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\","
        "\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\"],\"outputs\":[\"y\"]}]}}"},
    {"input-output-logical-name-collision",
        "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},\"inputs\":{\"x\":{"
        "\"dtype\":\"float32\",\"shape\":[2]}},\"nodes\":[{\"id\":\"twice\","
        "\"opType\":\"Add\",\"inputs\":{\"a\":\"x\",\"b\":\"x\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"x\",\"dtype\":\"float32\",\"shape\":[2]}},\"params\":{}},{"
        "\"id\":\"relu\",\"opType\":\"ReLU\",\"inputs\":{\"input\":\"twice\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"dtype\":\"float32\",\"shape\":[2]}},"
        "\"params\":{}}],\"outputs\":[\"y\"],\"execution\":{\"stateless\":{},"
        "\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],"
        "\"inputs\":[\"x\"],\"outputs\":[\"y\"]}]}}"},
    {"output-output-logical-name-collision",
        "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},\"inputs\":{\"x\":{"
        "\"dtype\":\"float32\",\"shape\":[2]}},\"nodes\":[{\"id\":\"twice\","
        "\"opType\":\"Add\",\"inputs\":{\"a\":\"x\",\"b\":\"x\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"twice\",\"dtype\":\"float32\",\"shape\":[2]}},\"params\":{}},{"
        "\"id\":\"relu\",\"opType\":\"ReLU\",\"inputs\":{\"input\":\"twice\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"twice\",\"dtype\":\"float32\",\"shape\":[2]"
        "}},\"params\":{}}],\"outputs\":[\"y\"],\"execution\":{\"stateless\":{},"
        "\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],"
        "\"inputs\":[\"x\"],\"outputs\":[\"y\"]}]}}"}
};

static void logical_write_graph(const char* path, const char* graph) {
    FILE* file = fopen(path, "wb"); CHECK(file);
    CHECK(fwrite(graph, 1, strlen(graph), file) == strlen(graph));
    CHECK(fclose(file) == 0);
}

static void logical_report(const char* name, VxStatus status) {
    printf("logical-load-observation name=%s status=%d report_status=%d stage=%d code=%d message=%s\n",
        name, status, report.status, report.stage, report.code, report.message);
    fflush(stdout);
}

static void logical_context_output(VxCompiledModel* compiled, float first, float second) {
    VxExecutionContext* context = NULL; VxResult* result = NULL;
    VxExecutionConfig config = {.context = VX_CONTEXT_OPTIONS_INIT};
    VxGraphExecutionOptions options = {.entrypoint = "evaluate", .range = VX_EXECUTION_RANGE_FULL};
    float input_values[2] = {first, second};
    const float expected[2] = {first > 0 ? 2 * first : 0, second > 0 ? 2 * second : 0};
    VxTensorBinding input = tensor("x", input_values);
    CHECK(vx_compiled_model_create_execution_context(compiled, &config, &context, &report) == VX_STATUS_OK);
    CHECK(!inspect(context).initialized && !inspect(context).bindings_available);
    CHECK(vx_execution_context_execute_graph(context, &input, 1, &options, &result, &report) == VX_STATUS_OK);
    CHECK(vx_result_output_count(result) == 1);
    output_is(result, "y", expected);
    vx_execution_context_release(context);
    output_is(result, "y", expected);
    vx_result_release(result);
}

static void logical_valid_recovery(VxRuntime* runtime, const char* path, const VxBackendPolicy* policy) {
    VxModelSource source = VX_MODEL_SOURCE_INIT; source.graph_path = path;
    VxModel* model = NULL; VxCompiledModel* compiled = NULL;
    logical_write_graph(path, logical_valid_graph);
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK && model);
    CHECK(vx_model_compile_options_revision(model, policy, 0, NULL, VX_PACKED_ARITHMETIC_F32, &compiled, &report) == VX_STATUS_OK && compiled);
    logical_context_output(compiled, -2, 3);
    vx_compiled_model_release(compiled); vx_model_release(model);
}

static void logical_load_rejection_test(VxRuntime* runtime, const char* path, const VxBackendPolicy* policy) {
    VxModelSource source = VX_MODEL_SOURCE_INIT; source.graph_path = path;
    for (size_t index = 0; index < sizeof(logical_invalid_cases) / sizeof(logical_invalid_cases[0]); ++index) {
        VxModel* model = NULL;
        logical_write_graph(path, logical_invalid_cases[index].graph);
        VxStatus status = vx_runtime_load_model(runtime, &source, &model, &report);
        logical_report(logical_invalid_cases[index].name, status);
        CHECK(status == VX_STATUS_INVALID_GRAPH && !model);
        CHECK(report.status == status && report.stage == VX_STAGE_MODEL_LOAD &&
            report.code == VX_CODE_INVALID_GRAPH_CONTRACT);
        CHECK(!strcmp(report.message, index < 3 ?
            "graph.json must use the volvox-graph/v1 contract" :
            "every graph output must have one canonical execution descriptor"));
        logical_valid_recovery(runtime, path, policy);
    }
    puts("logical rejection followed by numerical recovery: passed");
}

static void logical_source_snapshot_test(VxRuntime* runtime, const char* path, const VxBackendPolicy* policy) {
    VxModelSource source = VX_MODEL_SOURCE_INIT; source.graph_path = path;
    VxModel *model = NULL, *fresh = NULL; VxCompiledModel* compiled = NULL;
    VxExecutionContext* last = NULL; VxResult* retained[3] = {NULL, NULL, NULL};
    const float inputs[3][2] = {{-1, 3}, {4, -5}, {6, 7}};
    const float expected[3][2] = {{0, 6}, {8, 0}, {12, 14}};
    VxExecutionConfig config = {.context = VX_CONTEXT_OPTIONS_INIT};
    VxGraphExecutionOptions options = {.entrypoint = "evaluate", .range = VX_EXECUTION_RANGE_FULL};
    logical_write_graph(path, logical_valid_graph);
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK && model);
    CHECK(vx_model_compile_options_revision(model, policy, 0, NULL, VX_PACKED_ARITHMETIC_F32, &compiled, &report) == VX_STATUS_OK && compiled);
    logical_context_output(compiled, -2, 3);
    logical_write_graph(path, "{broken graph");
    VxStatus status = vx_runtime_load_model(runtime, &source, &fresh, &report);
    logical_report("fresh-malformed-source", status);
    CHECK(status == VX_STATUS_INVALID_GRAPH && !fresh);
    CHECK(report.status == status && report.stage == VX_STAGE_MODEL_LOAD &&
        report.code == VX_CODE_INVALID_GRAPH_CONTRACT);
    CHECK(!strcmp(report.message, "graph.json must use the volvox-graph/v1 contract"));
    for (int index = 0; index < 3; ++index) {
        if (index == 1) { vx_model_release(model); model = NULL; }
        if (index == 2) {
            CHECK(remove(path) == 0);
            status = vx_runtime_load_model(runtime, &source, &fresh, &report);
            logical_report("fresh-deleted-source", status);
            CHECK(status == VX_STATUS_IO_ERROR && !fresh);
            CHECK(report.status == status && report.stage == VX_STAGE_MODEL_LOAD &&
                report.code == VX_CODE_GRAPH_NOT_READABLE);
            CHECK(!strcmp(report.message, "graph package file could not be snapshotted"));
        }
        VxExecutionContext* context = NULL;
        CHECK(vx_compiled_model_create_execution_context(compiled, &config, &context, &report) == VX_STATUS_OK && context);
        CHECK(!inspect(context).initialized && !inspect(context).bindings_available);
        VxTensorBinding input = tensor("x", inputs[index]);
        options.tensors.owned_output = index != 0;
        if (index == 2) { vx_compiled_model_release(compiled); compiled = NULL; }
        CHECK(vx_execution_context_execute_graph(context, &input, 1, &options, &retained[index], &report) == VX_STATUS_OK);
        CHECK(vx_result_output_count(retained[index]) == 1);
        output_is(retained[index], "y", expected[index]);
        for (int earlier = 0; earlier < index; ++earlier) output_is(retained[earlier], "y", expected[earlier]);
        if (index == 2) last = context;
        else vx_execution_context_release(context);
    }
    vx_execution_context_release(last);
    for (int index = 0; index < 3; ++index) {
        output_is(retained[index], "y", expected[index]);
        vx_result_release(retained[index]);
    }
    /* Original source I/O refusals must not poison subsequent valid loading. */
    logical_valid_recovery(runtime, path, policy);
    puts("logical source snapshot survives replacement/deletion and parent retirement: passed");
}

static const char logical_symbolic_graph[] =
    "{\"format\":\"volvox-graph/v1\",\"dimensions\":{\"S\":{\"min\":1,\"max\":4}},"
    "\"inputs\":{\"x\":{\"dtype\":\"float32\",\"shape\":[\"S\"]},"
    "\"z\":{\"dtype\":\"float32\",\"shape\":[\"S\"]}},\"nodes\":[{\"id\":\"sum\","
    "\"opType\":\"Add\",\"inputs\":{\"a\":\"x\",\"b\":\"z\"},\"outputs\":{\"out\":{"
    "\"tensor\":\"sum\",\"dtype\":\"float32\",\"shape\":[\"S\"]}},\"params\":{}},{"
    "\"id\":\"relu\",\"opType\":\"ReLU\",\"inputs\":{\"input\":\"sum\"},\"outputs\":{"
    "\"out\":{\"tensor\":\"y\",\"dtype\":\"float32\",\"shape\":[\"S\"]}},\"params\":{}}],"
    "\"outputs\":[\"y\"],\"execution\":{\"stateless\":{},\"entrypoints\":[{"
    "\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\",\"z\"],"
    "\"outputs\":[\"y\"]}]}}";

typedef struct {
    VxMemoryAllocation items[64];
    size_t count;
    uint64_t bytes;
} LogicalInventory;

static int logical_inventory_item(const VxMemoryAllocation* item) {
    return !strcmp(item->name, "logical tensor declarations") ||
        !strcmp(item->name, "logical tensor name index") ||
        !strcmp(item->name, "logical tensor name") ||
        !strcmp(item->name, "logical tensor dimension symbol");
}

static LogicalInventory logical_model_inventory(VxModel* model) {
    VxMemoryAllocation allocations[256];
    VxMemoryInventory inventory = {.items = allocations, .capacity = 256};
    LogicalInventory logical = {0};
    size_t declarations = 0, indexes = 0, names = 0, symbols = 0;
    vx_model_memory_inventory(model, &inventory);
    CHECK(!inventory.truncated);
    for (size_t i = 0; i < inventory.count; ++i) {
        const VxMemoryAllocation* item = &allocations[i];
        if (!logical_inventory_item(item)) continue;
        CHECK(item->owner_kind == VX_MEMORY_OWNER_MODEL && item->allocator == VX_MEMORY_HOST_METADATA);
        CHECK(item->key && item->capacity && logical.count < 64);
        declarations += !strcmp(item->name, "logical tensor declarations");
        indexes += !strcmp(item->name, "logical tensor name index");
        names += !strcmp(item->name, "logical tensor name");
        symbols += !strcmp(item->name, "logical tensor dimension symbol");
        logical.items[logical.count++] = *item;
        CHECK(logical.bytes <= UINT64_MAX - item->capacity);
        logical.bytes += item->capacity;
    }
    CHECK(declarations == 1 && indexes == 1 && names && symbols);
    return logical;
}

/* Resource keys are reported allocation identities; no private cache pointers
 * are inspected. Accumulating sibling inventories must charge Model storage
 * once, with the same owner and capacity as before either context existed. */
static void logical_inventory_is_shared(const LogicalInventory* expected,
    VxCompiledModel* compiled, VxExecutionContext* first, VxExecutionContext* second) {
    VxMemoryAllocation allocations[256];
    VxMemoryInventory inventory = {.items = allocations, .capacity = 256};
    VxMemoryView view;
    size_t count = 0;
    uint64_t bytes = 0;
    if (compiled) vx_compiled_memory_inventory(compiled, &inventory);
    if (first) CHECK(vx_context_memory_inventory(first, &inventory, &view) == VX_STATUS_OK);
    if (second) CHECK(vx_context_memory_inventory(second, &inventory, &view) == VX_STATUS_OK);
    CHECK(!inventory.truncated);
    for (size_t i = 0; i < inventory.count; ++i) {
        const VxMemoryAllocation* item = &allocations[i];
        if (!logical_inventory_item(item)) continue;
        size_t matches = 0;
        for (size_t j = 0; j < expected->count; ++j) {
            const VxMemoryAllocation* original = &expected->items[j];
            if (item->key != original->key) continue;
            CHECK(item->owner_kind == VX_MEMORY_OWNER_MODEL && item->owner_id == original->owner_id &&
                item->allocator == original->allocator && item->space == original->space &&
                item->role == original->role && item->capacity == original->capacity &&
                !strcmp(item->name, original->name));
            ++matches;
        }
        CHECK(matches == 1 && bytes <= UINT64_MAX - item->capacity);
        bytes += item->capacity; ++count;
    }
    CHECK(count == expected->count && bytes == expected->bytes);
}

static void logical_shaped_output_is(VxResult* result, const float* expected, size_t count) {
    VxTensorInfo info = VX_TENSOR_INFO_INIT;
    float values[4]; size_t bytes = 0;
    CHECK(count && count <= 4 && vx_result_output_count(result) == 1);
    CHECK(vx_result_output_info(result, 0, &info, &report) == VX_STATUS_OK);
    CHECK(!strcmp(info.name, "y") && info.dtype == VX_DTYPE_F32 && info.rank == 1 &&
        info.shape[0] == (int64_t)count && info.byte_size == count * sizeof(float));
    CHECK(vx_result_read(result, "y", values, sizeof(values), &bytes, &report) == VX_STATUS_OK);
    CHECK(bytes == count * sizeof(float) && !memcmp(values, expected, bytes));
}

static void logical_symbolic_siblings_test(VxRuntime* runtime, const char* path,
    const VxBackendPolicy* policy) {
    const float small_x[] = {-2, 3}, small_z[] = {1, 4}, small_y[] = {0, 7};
    const float large_x[] = {4, -5, 6, -7}, large_z[] = {3, 2, 1, 10}, large_y[] = {7, 0, 7, 3};
    const float changed_x[] = {7, -1, 2}, changed_z[] = {-2, 6, -4}, changed_y[] = {5, 5, 0};
    const float reset_x[] = {1.5f}, reset_z[] = {-0.5f}, reset_y[] = {1};
    const float final_x[] = {-3, 8}, final_z[] = {9, -1}, final_y[] = {6, 7};
    for (int owned = 0; owned < 2; ++owned) {
        VxModelSource source = VX_MODEL_SOURCE_INIT; source.graph_path = path;
        VxModel* model = NULL; VxCompiledModel* compiled = NULL;
        VxExecutionContext *first = NULL, *second = NULL;
        VxResult* retained[5] = {NULL, NULL, NULL, NULL, NULL}; VxResult* result = NULL;
        VxExecutionConfig config = {.context = VX_CONTEXT_OPTIONS_INIT};
        VxGraphExecutionOptions options = {.entrypoint = "evaluate", .range = VX_EXECUTION_RANGE_FULL};
        options.tensors.owned_output = owned;
        logical_write_graph(path, logical_symbolic_graph);
        CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK && model);
        LogicalInventory shared = logical_model_inventory(model);
        CHECK(vx_model_compile_options_revision(model, policy, 0, NULL, VX_PACKED_ARITHMETIC_F32, &compiled, &report) == VX_STATUS_OK && compiled);
        CHECK(vx_compiled_model_create_execution_context(compiled, &config, &first, &report) == VX_STATUS_OK);
        CHECK(vx_compiled_model_create_execution_context(compiled, &config, &second, &report) == VX_STATUS_OK);
        CHECK(!inspect(first).initialized && !inspect(first).bindings_available);
        CHECK(!inspect(second).initialized && !inspect(second).bindings_available);
        logical_inventory_is_shared(&shared, compiled, first, second);
        VxTensorBinding small[] = {shaped("x", VX_DTYPE_F32, small_x, 1, 2, 0),
            shaped("z", VX_DTYPE_F32, small_z, 1, 2, 0)};
        VxTensorBinding large[] = {shaped("x", VX_DTYPE_F32, large_x, 1, 4, 0),
            shaped("z", VX_DTYPE_F32, large_z, 1, 4, 0)};
        CHECK(vx_execution_context_execute_graph(first, small, 2, &options, &retained[0], &report) == VX_STATUS_OK);
        CHECK(vx_execution_context_execute_graph(second, large, 2, &options, &retained[1], &report) == VX_STATUS_OK);
        logical_shaped_output_is(retained[0], small_y, 2);
        logical_shaped_output_is(retained[1], large_y, 4);
        VxExecutionInfo before = inspect(second);
        large[1] = small[1];
        CHECK(vx_execution_context_execute_graph(second, large, 2, &options, &result, &report) == VX_STATUS_INVALID_ARGUMENT);
        CHECK(report.input_issue.code == VX_INPUT_SYMBOL_MISMATCH && !strcmp(report.input_issue.name, "z"));
        vx_result_release(result); result = NULL;
        VxExecutionInfo after = inspect(second); CHECK(!memcmp(&before, &after, sizeof(before)));
        const float outside_values[5] = {1, 2, 3, 4, 5};
        VxTensorBinding outside[] = {shaped("x", VX_DTYPE_F32, outside_values, 1, 5, 0),
            shaped("z", VX_DTYPE_F32, outside_values, 1, 5, 0)};
        CHECK(vx_execution_context_execute_graph(second, outside, 2, &options, &result, &report) == VX_STATUS_INVALID_ARGUMENT);
        CHECK(report.input_issue.code == VX_INPUT_DIMENSION_OUT_OF_RANGE && !strcmp(report.input_issue.name, "x"));
        vx_result_release(result); result = NULL;
        after = inspect(second); CHECK(!memcmp(&before, &after, sizeof(before)));
        large[1] = shaped("z", VX_DTYPE_F32, large_z, 1, 4, 0);
        VxTensorBinding changed[] = {shaped("x", VX_DTYPE_F32, changed_x, 1, 3, 0),
            shaped("z", VX_DTYPE_F32, changed_z, 1, 3, 0)};
        CHECK(vx_execution_context_execute_graph(first, changed, 2, &options, &retained[2], &report) == VX_STATUS_OK);
        logical_shaped_output_is(retained[2], changed_y, 3);
        after = inspect(second); CHECK(!memcmp(&before, &after, sizeof(before)));
        /* Stateless contexts do not admit a decode reset. Its typed refusal
         * must preserve this context's bindings and its sibling's state. */
        VxExecutionInfo first_before_reset = inspect(first);
        CHECK(vx_execution_context_decode_reset(first, &report) == VX_STATUS_BACKEND_UNSUPPORTED);
        CHECK(report.code == VX_CODE_DECODE_NOT_ENABLED);
        VxExecutionInfo first_after_reset = inspect(first);
        CHECK(!memcmp(&first_before_reset, &first_after_reset, sizeof(first_before_reset)));
        after = inspect(second); CHECK(!memcmp(&before, &after, sizeof(before)));
        logical_inventory_is_shared(&shared, compiled, first, second);
        vx_model_release(model); model = NULL;
        vx_compiled_model_release(compiled); compiled = NULL;
        VxTensorBinding reset[] = {shaped("x", VX_DTYPE_F32, reset_x, 1, 1, 0),
            shaped("z", VX_DTYPE_F32, reset_z, 1, 1, 0)};
        CHECK(vx_execution_context_execute_graph(first, reset, 2, &options, &retained[3], &report) == VX_STATUS_OK);
        logical_shaped_output_is(retained[3], reset_y, 1);
        VxTensorReference reuse[] = {{"x", NULL}, {"z", NULL}};
        options.tensors.references = reuse; options.tensors.reference_count = 2;
        CHECK(vx_execution_context_execute_graph(second, NULL, 0, &options, &result, &report) == VX_STATUS_OK);
        logical_shaped_output_is(result, large_y, 4); vx_result_release(result); result = NULL;
        logical_inventory_is_shared(&shared, NULL, first, second);
        vx_execution_context_release(first); first = NULL;
        logical_inventory_is_shared(&shared, NULL, NULL, second);
        options.tensors.references = NULL; options.tensors.reference_count = 0;
        VxTensorBinding final[] = {shaped("x", VX_DTYPE_F32, final_x, 1, 2, 0),
            shaped("z", VX_DTYPE_F32, final_z, 1, 2, 0)};
        CHECK(vx_execution_context_execute_graph(second, final, 2, &options, &retained[4], &report) == VX_STATUS_OK);
        vx_execution_context_release(second);
        logical_shaped_output_is(retained[0], small_y, 2);
        logical_shaped_output_is(retained[1], large_y, 4);
        logical_shaped_output_is(retained[2], changed_y, 3);
        logical_shaped_output_is(retained[3], reset_y, 1);
        logical_shaped_output_is(retained[4], final_y, 2);
        for (size_t i = 0; i < 5; ++i) vx_result_release(retained[i]);
    }
    puts("shared Model logical declarations preserve sibling shapes, snapshots and accounting: passed");
}

/* Each graph has a defined final y and all references resolve. The sole
 * defect is one repeated logical tensor name, without a missing output. */
static const struct { const char* name; const char* graph; } sole_collision_cases[] = {
    {"sole-input-output-name-collision",
        "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},\"inputs\":{\"x\":{"
        "\"dtype\":\"float32\",\"shape\":[2]}},\"nodes\":[{\"id\":\"twice\","
        "\"opType\":\"Add\",\"inputs\":{\"a\":\"x\",\"b\":\"x\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"x\",\"dtype\":\"float32\",\"shape\":[2]}},\"params\":{}},{"
        "\"id\":\"relu\",\"opType\":\"ReLU\",\"inputs\":{\"input\":\"x\"},\"outputs\":{"
        "\"out\":{\"tensor\":\"y\",\"dtype\":\"float32\",\"shape\":[2]}},\"params\":{}}"
        "],\"outputs\":[\"y\"],\"execution\":{\"stateless\":{},\"entrypoints\":[{"
        "\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\"],"
        "\"outputs\":[\"y\"]}]}}"},
    {"sole-output-output-name-collision",
        "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},\"inputs\":{\"x\":{"
        "\"dtype\":\"float32\",\"shape\":[2]}},\"nodes\":[{\"id\":\"twice\","
        "\"opType\":\"Add\",\"inputs\":{\"a\":\"x\",\"b\":\"x\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"twice\",\"dtype\":\"float32\",\"shape\":[2]}},\"params\":{}},{"
        "\"id\":\"second-twice\",\"opType\":\"Add\",\"inputs\":{\"a\":\"x\","
        "\"b\":\"x\"},\"outputs\":{\"out\":{\"tensor\":\"twice\",\"dtype\":\"float32\","
        "\"shape\":[2]}},\"params\":{}},{\"id\":\"relu\",\"opType\":\"ReLU\","
        "\"inputs\":{\"input\":\"twice\"},\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[2]}},\"params\":{}}],\"outputs\":[\"y\"],"
        "\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\","
        "\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\"],\"outputs\":[\"y\"]}]}}"}
};

static void logical_sole_collision_test(VxRuntime* runtime, const char* path, const VxBackendPolicy* policy) {
    VxModelSource source = VX_MODEL_SOURCE_INIT; source.graph_path = path;
    for (size_t index = 0; index < sizeof(sole_collision_cases) / sizeof(sole_collision_cases[0]); ++index) {
        VxModel* model = NULL;
        logical_write_graph(path, sole_collision_cases[index].graph);
        VxStatus status = vx_runtime_load_model(runtime, &source, &model, &report);
        logical_report(sole_collision_cases[index].name, status);
        CHECK(status == VX_STATUS_INVALID_GRAPH && !model);
        CHECK(report.status == status && report.stage == VX_STAGE_MODEL_LOAD &&
            report.code == VX_CODE_INVALID_GRAPH_CONTRACT);
        CHECK(!strcmp(report.message, "every graph output must have one canonical execution descriptor"));
        logical_valid_recovery(runtime, path, policy);
    }
    puts("sole logical-name collisions followed by numerical recovery: passed");
}

int main(int argc, char** argv) {
    CHECK(argc == 2);
    FILE* fixture = fopen(argv[1], "wb");
    CHECK(fixture);
    const char* graph = "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},"
        "\"inputs\":{\"x\":{\"dtype\":\"float32\",\"shape\":[2]},\"y\":{\"dtype\":\"float32\",\"shape\":[2]}},"
        "\"nodes\":[],\"outputs\":[\"x\",\"y\"],\"execution\":{\"stateless\":{},\"entrypoints\":[{"
        "\"name\":\"evaluate\",\"effect\":\"EXECUTION_EFFECT_EVALUATE\",\"allowed_ranges\":[\"EXECUTION_RANGE_KIND_FULL\"],"
        "\"inputs\":[\"x\",\"y\"],\"outputs\":[\"x\",\"y\"]}]}}";
    CHECK(fwrite(graph, 1, strlen(graph), fixture) == strlen(graph)); CHECK(!fclose(fixture));
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    runtime_options.cpu_threads = 1;
    runtime_options.execution_mode = VX_EXECUTION_MODE_DIRECT;
    VxRuntime* runtime = NULL; VxModel* model = NULL; VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL; VxResult* result = NULL;
    CHECK(vx_runtime_create(&runtime_options, &runtime, &report) == VX_STATUS_OK);
    model_snapshot_size_test(runtime);
    VxModelSource source = VX_MODEL_SOURCE_INIT; source.graph_path = argv[1];
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK);
    const char* backends[] = {"cpu"};
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    policy.mode = VX_BACKEND_REQUIRE; policy.backends = backends; policy.backend_count = 1;
    policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    CHECK(vx_model_compile_options_revision(model, &policy, 0, NULL, VX_PACKED_ARITHMETIC_F32, &compiled, &report) == VX_STATUS_OK);
    VxExecutionConfig config = {.context = VX_CONTEXT_OPTIONS_INIT};
    config.max_memory_bytes = 1;
    CHECK(vx_compiled_model_create_execution_context(compiled, &config, &context, &report) == VX_STATUS_OUT_OF_MEMORY);
    CHECK(!context);
    config.max_memory_bytes = UINT64_C(1) << 30;
    CHECK(vx_compiled_model_create_execution_context(compiled, &config, &context, &report) == VX_STATUS_OK);
    CHECK(inspect(context).state == VX_EXECUTION_CONTEXT_STATE_READY);
    float x[] = {1, 2}, y[] = {10, 20};
    VxTensorBinding values[] = {tensor("x", x), tensor("y", y)};
    VxGraphExecutionOptions options = {.entrypoint = "evaluate", .range = VX_EXECUTION_RANGE_FULL};
    values[0].name = "unknown";
    CHECK(vx_execution_context_execute_graph(context, values, 2, &options, &result, &report) == VX_STATUS_INVALID_ARGUMENT);
    CHECK(report.input_issue.code == VX_INPUT_UNKNOWN_INPUT && report.input_issue.has_actual &&
        report.input_issue.has_input_index && !strcmp(report.input_issue.name, "unknown"));
    vx_result_release(result); result = NULL; values[0].name = "x";
    CHECK(vx_execution_context_execute_graph(context, values, 2, &options, &result, &report) == VX_STATUS_OK);
    CHECK(vx_result_output_count(result) == 2);
    output_is(result, "x", x); output_is(result, "y", y);
    VxResult* retained = result; result = NULL;
    VxTensorReference swap[] = {{"x", "y"}, {"y", "x"}};
    options.tensors.references = swap; options.tensors.reference_count = 2;
    CHECK(vx_execution_context_execute_graph(context, NULL, 0, &options, &result, &report) == VX_STATUS_OK);
    output_is(result, "x", y); output_is(result, "y", x);
    output_is(retained, "x", x); output_is(retained, "y", y);
    vx_result_release(result); vx_result_release(retained); result = NULL;
    options.tensors = (VxTensorRunOptions){.select_outputs = 1};
    CHECK(vx_execution_context_execute_graph(context, values, 2, &options, &result, &report) == VX_STATUS_OK);
    CHECK(vx_result_output_count(result) == 0 && inspect(context).bindings_available);
    vx_result_release(result); result = NULL;
    options.tensors = (VxTensorRunOptions){0}; options.max_output_bytes = 4;
    CHECK(vx_execution_context_execute_graph(context, values, 2, &options, &result, &report) != VX_STATUS_OK);
    CHECK(!result && inspect(context).state == VX_EXECUTION_CONTEXT_STATE_READY);
    options.max_output_bytes = 0;
    CHECK(vx_execution_context_execute_graph(context, values, 1, &options, &result, &report) != VX_STATUS_OK);
    CHECK(result && inspect(context).state == VX_EXECUTION_CONTEXT_STATE_READY);
    VxExecutionInfo info; vx_result_execution_info(result, &info);
    CHECK(info.state == VX_EXECUTION_CONTEXT_STATE_READY);
    vx_result_release(result); result = NULL;
    uint64_t token = 0;
    CHECK(vx_execution_context_reserve_graph(context, values, 2, &options, &token, &report) == VX_STATUS_OK && token);
    CHECK(inspect(context).state == VX_EXECUTION_CONTEXT_STATE_QUEUED);
    CHECK(vx_execution_context_execute_graph(context, values, 2, &options, &result, &report) == VX_STATUS_BUSY);
    vx_execution_context_unreserve(context, token, 0);
    CHECK(inspect(context).state == VX_EXECUTION_CONTEXT_STATE_READY);
    CHECK(vx_execution_context_run_reserved_graph(context, token, values, 2, &options, &result, &report) != VX_STATUS_OK);
    CHECK(!result);
    CHECK(vx_execution_context_reserve_graph(context, values, 2, &options, &token, &report) == VX_STATUS_OK);
    CHECK(vx_execution_context_run_reserved_graph(context, token, values, 2, &options, &result, &report) == VX_STATUS_OK);
    retained = result; result = NULL;
    CHECK(vx_execution_context_execute_graph(context, values, 2, &options, &result, &report) == VX_STATUS_BUSY);
    vx_execution_context_finish_reserved(context, token, 1);
    CHECK(!inspect(context).bindings_available);
    CHECK(vx_execution_context_execute_graph(context, values, 2, &options, &result, &report) == VX_STATUS_OK);
    vx_execution_context_cancel_reserved(context, token);
    CHECK(inspect(context).state == VX_EXECUTION_CONTEXT_STATE_READY);
    vx_result_release(result); vx_result_release(retained); result = NULL;
    CHECK(vx_execution_context_cancel_execution(context, &report) == VX_STATUS_OK);
    CHECK(inspect(context).state == VX_EXECUTION_CONTEXT_STATE_CANCELLED);
    CHECK(vx_execution_context_execute_graph(context, values, 2, &options, &result, &report) != VX_STATUS_OK);
    vx_result_release(result); vx_execution_context_release(context);
    vx_compiled_model_release(compiled); vx_model_release(model);
    optional_state_binding_test(runtime, argv[1]);
    packed_arithmetic_test(runtime, argv[1], &policy);
    generation_test(runtime, argv[1], &policy);
    full_prefix_generation_test(runtime, argv[1], &policy);
    llm_workspace_budget_test(runtime, argv[1], &policy);
    prepared_native_engine_test(runtime, argv[1]);
    logical_sole_collision_test(runtime, argv[1], &policy);
    logical_load_rejection_test(runtime, argv[1], &policy);
    logical_source_snapshot_test(runtime, argv[1], &policy);
    logical_symbolic_siblings_test(runtime, argv[1], &policy);
    vx_runtime_release(runtime);
    remove(argv[1]);
    puts("execution contract: passed");
    return 0;
}
