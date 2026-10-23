/* One immutable weight package must execute independently under every compiled
 * arithmetic, including generation's cached row path and CPU worker dispatch. */
#include "safetensors.h"
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
#include <pthread.h>
typedef struct {
    VxExecutionContext* context;
    const VxTensorBinding* inputs;
    VxResult* result;
    VxStatus status;
    VxReport report;
} PackedArithmeticWorker;

static void* packed_arithmetic_execute_worker(void* opaque) {
    PackedArithmeticWorker* worker = opaque;
    VxGraphExecutionOptions options = {.entrypoint = "evaluate", .range = VX_EXECUTION_RANGE_FULL};
    worker->status = vx_execution_context_execute_graph(worker->context, worker->inputs,
        3, &options, &worker->result, &worker->report);
    return NULL;
}
#endif

static void packed_arithmetic_test(VxRuntime* runtime, const char* path,
                                   const VxBackendPolicy* policy) {
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},\"inputs\":{"
        "\"tokens\":{\"dtype\":\"int32\",\"shape\":[1,4]},"
        "\"keep\":{\"dtype\":\"int32\",\"shape\":[1,4]},"
        "\"lengths\":{\"dtype\":\"int32\",\"shape\":[1]}},\"nodes\":["
        "{\"id\":\"embed\",\"opType\":\"Embedding\",\"inputs\":{\"input\":\"tokens\",\"weight\":\"embedding\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"qkv\",\"dtype\":\"float32\",\"shape\":[1,4,390]}},\"params\":{}},"
        "{\"id\":\"attention\",\"opType\":\"SDPA\",\"inputs\":{\"qkv\":\"qkv\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"x\",\"dtype\":\"float32\",\"shape\":[1,4,130]}},"
        "\"params\":{\"heads\":1,\"causal\":true}},"
        "{\"id\":\"packed\",\"opType\":\"PackedLinear\",\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"scale\":\"s\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"logits\",\"dtype\":\"float32\",\"shape\":[1,4,2]}},"
        "\"params\":{\"group_size\":128,\"weight_bits\":4}},"
        "{\"id\":\"select\",\"opType\":\"ArgMax\",\"inputs\":{\"input\":\"logits\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"ids\",\"dtype\":\"int32\",\"shape\":[1,4]}},"
        "\"params\":{\"axis\":2,\"keepdims\":false,\"select_last_index\":0}}],"
        "\"outputs\":[\"ids\",\"logits\"],\"execution\":{\"invariant_inputs\":[\"lengths\"],"
        "\"incremental\":{\"active_lengths_input\":\"lengths\",\"sequence_axis\":1,\"update_inputs\":[\"tokens\",\"keep\"]},"
        "\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],"
        "\"inputs\":[\"tokens\",\"keep\",\"lengths\"],\"outputs\":[\"ids\",\"logits\"]},"
        "{\"name\":\"initialize\",\"effect\":2,\"allowed_ranges\":[1,2],"
        "\"inputs\":[\"tokens\",\"keep\",\"lengths\"],\"outputs\":[\"ids\",\"logits\"]},"
        "{\"name\":\"transition\",\"effect\":3,\"allowed_ranges\":[3],"
        "\"inputs\":[\"tokens\",\"keep\"],\"outputs\":[\"ids\",\"logits\"]}],"
        "\"generation\":{\"initialize_entrypoint\":\"initialize\",\"transition_entrypoint\":\"transition\","
        "\"token_input\":\"tokens\",\"keep_input\":\"keep\",\"token_output\":\"ids\",\"logits_output\":\"logits\","
        "\"logits_axis\":2,\"id_domain_size\":2,\"output_name\":\"generated\",\"sequence_axis\":1}}}";
    const VxPackedArithmetic modes[] = {VX_PACKED_ARITHMETIC_DYNAMIC_I8,
        VX_PACKED_ARITHMETIC_F16, VX_PACKED_ARITHMETIC_F32,
        VX_PACKED_ARITHMETIC_F32_FAST, VX_PACKED_ARITHMETIC_UNSPECIFIED};
    enum { MODE_COUNT = sizeof(modes) / sizeof(modes[0]) };
    char weight_path[4096];
    CHECK(snprintf(weight_path, sizeof(weight_path), "%s.packed.safetensors", path) > 0);
    float embedding[780] = {0}, scales[4] = {1, 1, 1, 1}, bias[2] = {-0.0002f, 0};
    /* Zero Q/K and identical V rows keep causal attention's output constant,
     * while exercising the actual supported incremental generation topology. */
    embedding[260] = embedding[650] = 1.0003f;
    embedding[261] = embedding[651] = 1.0f;
    const int embedding_shape[] = {2, 390}, scale_shape[] = {2, 2}, bias_shape[] = {2};
    for (int weight_bits = 4; weight_bits <= 8; weight_bits += 4) {
        const int words = weight_bits == 4 ? 17 : 33;
        uint32_t codes[66];
        for (int i = 0; i < words * 2; ++i)
            codes[i] = weight_bits == 4 ? UINT32_C(0x88888888) : UINT32_C(0x80808080);
        codes[0] += 1;
        codes[words] += 1u << weight_bits;
        const int code_shape[] = {2, words};
        SafetensorsFile weights = {0};
        CHECK(!safetensors_init_empty(&weights, SAFETENSORS_OPEN_READ_WRITE));
        CHECK(!safetensors_add_tensor(&weights, "embedding", SAFETENSORS_DTYPE_F32,
            embedding_shape, 2, embedding, sizeof(embedding)));
        CHECK(!safetensors_add_tensor(&weights, "w", SAFETENSORS_DTYPE_I32,
            code_shape, 2, codes, (size_t)words * 2u * sizeof(uint32_t)));
        CHECK(!safetensors_add_tensor(&weights, "s", SAFETENSORS_DTYPE_F32,
            scale_shape, 2, scales, sizeof(scales)));
        CHECK(!safetensors_add_tensor(&weights, "b", SAFETENSORS_DTYPE_F32,
            bias_shape, 1, bias, sizeof(bias)));
        CHECK(!safetensors_save(weight_path, &weights));
        safetensors_free(&weights);
        cJSON* document = cJSON_Parse(graph); CHECK(document);
        cJSON* packed = cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(document, "nodes"), 2);
        cJSON_SetNumberValue(cJSON_GetObjectItemCaseSensitive(
            cJSON_GetObjectItemCaseSensitive(packed, "params"), "weight_bits"), weight_bits);
        char* serialized = cJSON_PrintUnformatted(document); CHECK(serialized);
        FILE* fixture = fopen(path, "wb"); CHECK(fixture);
        CHECK(fwrite(serialized, 1, strlen(serialized), fixture) == strlen(serialized));
        CHECK(!fclose(fixture)); free(serialized); cJSON_Delete(document);
        const char* paths[] = {weight_path};
        VxModelSource source = VX_MODEL_SOURCE_INIT;
        source.graph_path = path; source.weight_paths = paths; source.weight_path_count = 1;
        VxModel* model = NULL;
        CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK);
        VxCompiledModel* compiled[MODE_COUNT] = {0};
        VxExecutionContext* contexts[MODE_COUNT] = {0};
        for (int m = 0; m < MODE_COUNT; ++m) {
            CHECK(vx_model_compile_options_revision(model, policy, 0, NULL, modes[m],
                &compiled[m], &report) == VX_STATUS_OK);
            CHECK(vx_compiled_model_packed_arithmetic(compiled[m]) ==
                (modes[m] ? modes[m] : VX_PACKED_ARITHMETIC_F32));
            VxExecutionConfig config = {.context = VX_CONTEXT_OPTIONS_INIT};
            CHECK(vx_compiled_model_create_execution_context(compiled[m], &config,
                &contexts[m], &report) == VX_STATUS_OK);
        }
        VxCompiledModel* rejected = NULL;
        CHECK(vx_model_compile_options_revision(model, policy, 0, NULL, (VxPackedArithmetic)99,
            &rejected, &report) == VX_STATUS_INVALID_ARGUMENT && !rejected);
        /* A known future option must fail before allocating an executable,
         * including policies which otherwise permit another backend. */
        const char* future_backends[] = {"cuda", "cpu"};
        VxBackendPolicy future_policy = VX_BACKEND_POLICY_INIT;
        future_policy.backends = future_backends;
        future_policy.backend_count = 2;
        CHECK(vx_model_compile_options_revision(model, &future_policy, 0, NULL,
            VX_PACKED_ARITHMETIC_DYNAMIC_I4, &rejected, &report) ==
            VX_STATUS_BACKEND_UNSUPPORTED && !rejected);
        CHECK(report.code == VX_CODE_BACKEND_UNSUPPORTED && report.stage == VX_STAGE_COMPILE);
        CHECK(vx_model_compile_options_revision(model, policy, 0, NULL,
            VX_PACKED_ARITHMETIC_DYNAMIC_I4, &rejected, &report) ==
            VX_STATUS_BACKEND_UNSUPPORTED && !rejected);
        int32_t tokens[4] = {0}, keep[4] = {1, 0, 0, 0}, length = 1;
        VxTensorBinding inputs[] = {shaped("tokens", VX_DTYPE_I32, tokens, 2, 1, 4),
            shaped("keep", VX_DTYPE_I32, keep, 2, 1, 4),
            shaped("lengths", VX_DTYPE_I32, &length, 1, 1, 0)};
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
        PackedArithmeticWorker workers[MODE_COUNT];
        pthread_t threads[MODE_COUNT];
        for (int m = 0; m < MODE_COUNT; ++m) {
            workers[m] = (PackedArithmeticWorker){.context = contexts[m], .inputs = inputs,
                .report = VX_REPORT_INIT};
            CHECK(!pthread_create(&threads[m], NULL, packed_arithmetic_execute_worker, &workers[m]));
        }
        for (int m = MODE_COUNT - 1; m >= 0; --m) {
            CHECK(!pthread_join(threads[m], NULL));
            report = workers[m].report;
            CHECK(workers[m].status == VX_STATUS_OK);
            float logits[8]; size_t bytes = 0;
            CHECK(vx_result_read(workers[m].result, "logits", logits, sizeof(logits), &bytes, &report) == VX_STATUS_OK);
            CHECK(bytes == sizeof(logits));
            CHECK(fabsf(logits[0] - ((modes[m] == VX_PACKED_ARITHMETIC_F16 ? 1.0f : 1.0003f) + bias[0])) < 1e-6f);
            CHECK(fabsf(logits[1] - (modes[m] == VX_PACKED_ARITHMETIC_DYNAMIC_I8 ? 1.0003f : 1.0f)) < 1e-6f);
            vx_result_release(workers[m].result);
        }
#endif
        /* Revisit older contexts after newer arithmetic has executed. */
        for (int pass = 0; pass < 2; ++pass) for (int ordinal = 0; ordinal < MODE_COUNT; ++ordinal) {
            const int m = pass ? MODE_COUNT - 1 - ordinal : ordinal;
            VxResult* result = NULL;
            VxGraphExecutionOptions options = {.entrypoint = "evaluate", .range = VX_EXECUTION_RANGE_FULL};
            CHECK(vx_execution_context_execute_graph(contexts[m], inputs, 3, &options,
                &result, &report) == VX_STATUS_OK);
            float logits[8]; size_t bytes = 0;
            CHECK(vx_result_read(result, "logits", logits, sizeof(logits), &bytes, &report) == VX_STATUS_OK);
            CHECK(bytes == sizeof(logits));
            const float first = modes[m] == VX_PACKED_ARITHMETIC_F16 ? 1.0f : 1.0003f;
            const float second = modes[m] == VX_PACKED_ARITHMETIC_DYNAMIC_I8 ? 1.0003f : 1.0f;
            for (int row = 0; row < 4; ++row) {
                CHECK(fabsf(logits[row * 2] - (first + bias[0])) < 1e-6f);
                CHECK(fabsf(logits[row * 2 + 1] - second) < 1e-6f);
            }
            vx_result_release(result);
        }
        /* Generation must keep the same arithmetic through prefill and each
         * cached row. The near-tie intentionally changes the emitted token. */
        for (int m = MODE_COUNT - 1; m >= 0; --m) {
            VxExecutionConfig config = {.context = VX_CONTEXT_OPTIONS_INIT,
                .autoregressive = 1, .max_steps = 2, .sampling = VX_SAMPLING_GREEDY};
            config.context.decode_row_mode = VX_DECODE_ROW_REQUIRED;
            config.context.require_incremental = 1;
            VxExecutionContext* context = NULL;
            CHECK(vx_compiled_model_create_execution_context(compiled[m], &config,
                &context, &report) == VX_STATUS_OK);
            for (int step = 0; step < 2; ++step) {
                VxGenerationOptions options = {.initial_inputs_present = step == 0, .max_steps = 1};
                VxResult* result = NULL;
                CHECK(vx_execution_context_generate(context, step ? NULL : inputs, step ? 0 : 3,
                    &options, &result, &report) == VX_STATUS_OK);
                int32_t generated = -1; size_t bytes = 0;
                CHECK(vx_result_read(result, "generated", &generated, sizeof(generated),
                    &bytes, &report) == VX_STATUS_OK && bytes == sizeof(generated));
                CHECK(generated == (modes[m] == VX_PACKED_ARITHMETIC_DYNAMIC_I8 ||
                    modes[m] == VX_PACKED_ARITHMETIC_F16 ? 1 : 0));
                vx_result_release(result);
            }
            vx_execution_context_release(context);
            vx_execution_context_release(contexts[m]);
            vx_compiled_model_release(compiled[m]);
        }
        vx_model_release(model);
    }
    CHECK(!remove(weight_path));
}
