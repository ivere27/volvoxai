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
    /* Multi-slot (B=4) row decode must gather active slot rows into a batch,
     * execute the multi-row PackedLinear kernel across W2/W3/W4/W8 and all
     * arithmetic modes, and scatter only non-empty slots. */
    const char* batch_graph =
        "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},\"inputs\":{"
        "\"tokens\":{\"dtype\":\"int32\",\"shape\":[4,4]},"
        "\"keep\":{\"dtype\":\"int32\",\"shape\":[4,4]},"
        "\"lengths\":{\"dtype\":\"int32\",\"shape\":[4]}},\"nodes\":["
        "{\"id\":\"embed\",\"opType\":\"Embedding\",\"inputs\":{\"input\":\"tokens\",\"weight\":\"embedding\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"x\",\"dtype\":\"float32\",\"shape\":[4,4,130]}},\"params\":{}},"
        "{\"id\":\"packed\",\"opType\":\"PackedLinear\",\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"scale\":\"s\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"logits\",\"dtype\":\"float32\",\"shape\":[4,4,4]}},"
        "\"params\":{\"group_size\":128,\"weight_bits\":4}},"
        "{\"id\":\"select\",\"opType\":\"ArgMax\",\"inputs\":{\"input\":\"logits\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"ids\",\"dtype\":\"int32\",\"shape\":[4,4]}},"
        "\"params\":{\"axis\":2,\"keepdims\":false,\"select_last_index\":0}}],"
        "\"outputs\":[\"ids\",\"logits\"],\"execution\":{\"invariant_inputs\":[\"lengths\"],"
        "\"incremental\":{\"active_lengths_input\":\"lengths\",\"sequence_axis\":1,\"update_inputs\":[\"tokens\",\"keep\"]},"
        "\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],"
        "\"inputs\":[\"tokens\",\"keep\",\"lengths\"],\"outputs\":[\"ids\",\"logits\"]},"
        "{\"name\":\"initialize\",\"effect\":2,\"allowed_ranges\":[1,2],"
        "\"inputs\":[\"tokens\",\"keep\",\"lengths\"],\"outputs\":[\"ids\",\"logits\"]},"
        "{\"name\":\"transition\",\"effect\":3,\"allowed_ranges\":[3],"
        "\"inputs\":[\"tokens\",\"keep\"],\"outputs\":[\"ids\",\"logits\"]}]}}";
    const int bit_widths[] = {2, 3, 4, 8};
    float batch_embedding[4 * 130] = {0};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 130; ++c)
            batch_embedding[r * 130 + c] = (float)(((r + 1) * (c + 3)) % 11 - 5) * 0.25f;
    const float batch_scales[4 * 2] = {0.5f, 0.25f, 0.75f, 0.5f, 1.0f, 0.5f, 0.25f, 0.75f};
    const float batch_bias[4] = {-0.125f, 0.25f, -0.5f, 0.125f};
    const int batch_embed_shape[] = {4, 130}, batch_scale_shape[] = {4, 2}, batch_bias_shape[] = {4};
    for (size_t b_idx = 0; b_idx < sizeof(bit_widths) / sizeof(bit_widths[0]); ++b_idx) {
        const int weight_bits = bit_widths[b_idx];
        const int words = 4 * weight_bits + 1;
        const uint32_t tail_mask = UINT32_MAX << (uint32_t)((130u * (uint32_t)weight_bits) % 32u);
        const uint32_t tail_pad = weight_bits == 3 ? 0u : weight_bits == 2 ? 0xaaaaaaaau :
            weight_bits == 4 ? 0x88888888u : 0x80808080u;
        uint32_t codes[4 * 33] = {0};
        for (int n = 0; n < 4; ++n) {
            for (int k = 0; k < 130; ++k) {
                const uint32_t q = (uint32_t)((n * 7 + k * 3) % (1 << weight_bits));
                const int group = k / 32, lane = k % 32;
                const uint32_t bit = (uint32_t)(lane * weight_bits);
                const int word = group * weight_bits + (int)(bit / 32u);
                const uint32_t shift = bit % 32u;
                codes[n * words + word] |= q << shift;
                if (shift + (uint32_t)weight_bits > 32u)
                    codes[n * words + word + 1] |= q >> (32u - shift);
            }
            codes[n * words + words - 1] |= tail_pad & tail_mask;
        }
        const int code_shape[] = {4, words};
        SafetensorsFile weights = {0};
        CHECK(!safetensors_init_empty(&weights, SAFETENSORS_OPEN_READ_WRITE));
        CHECK(!safetensors_add_tensor(&weights, "embedding", SAFETENSORS_DTYPE_F32,
            batch_embed_shape, 2, batch_embedding, sizeof(batch_embedding)));
        CHECK(!safetensors_add_tensor(&weights, "w", SAFETENSORS_DTYPE_I32,
            code_shape, 2, codes, (size_t)words * 4u * sizeof(uint32_t)));
        CHECK(!safetensors_add_tensor(&weights, "s", SAFETENSORS_DTYPE_F32,
            batch_scale_shape, 2, batch_scales, sizeof(batch_scales)));
        CHECK(!safetensors_add_tensor(&weights, "b", SAFETENSORS_DTYPE_F32,
            batch_bias_shape, 1, batch_bias, sizeof(batch_bias)));
        CHECK(!safetensors_save(weight_path, &weights));
        safetensors_free(&weights);
        cJSON* document = cJSON_Parse(batch_graph); CHECK(document);
        cJSON* packed = cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(document, "nodes"), 1);
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
        for (int m = 0; m < MODE_COUNT; ++m) {
            VxCompiledModel* compiled = NULL;
            CHECK(vx_model_compile_options_revision(model, policy, 0, NULL, modes[m],
                &compiled, &report) == VX_STATUS_OK);
            VxExecutionConfig row_cfg = {.context = VX_CONTEXT_OPTIONS_INIT};
            row_cfg.context.decode_row_mode = VX_DECODE_ROW_REQUIRED;
            row_cfg.context.require_incremental = 1;
            row_cfg.context.decode_slots = 4;
            VxExecutionContext *decode_ctx = NULL, *ref_ctx = NULL;
            CHECK(vx_compiled_model_create_execution_context(compiled, &row_cfg,
                &decode_ctx, &report) == VX_STATUS_OK);
            VxExecutionConfig full_cfg = {.context = VX_CONTEXT_OPTIONS_INIT};
            CHECK(vx_compiled_model_create_execution_context(compiled, &full_cfg,
                &ref_ctx, &report) == VX_STATUS_OK);
            int32_t batch_tokens[16] = {
                0, 0, 0, 0,
                1, 2, 0, 0,
                2, 0, 0, 0,
                3, 1, 0, 0
            };
            int32_t batch_keep[16] = {
                1, 0, 0, 0,
                1, 1, 0, 0,
                1, 0, 0, 0,
                1, 1, 0, 0
            };
            int32_t batch_lengths[4] = {1, 2, 1, 2};
            VxTensorBinding batch_inputs[] = {
                shaped("tokens", VX_DTYPE_I32, batch_tokens, 2, 4, 4),
                shaped("keep", VX_DTYPE_I32, batch_keep, 2, 4, 4),
                shaped("lengths", VX_DTYPE_I32, batch_lengths, 1, 4, 0)
            };
            VxGraphExecutionOptions init_opts = {.entrypoint = "initialize", .range = VX_EXECUTION_RANGE_FULL};
            VxResult *decode_res = NULL, *ref_res = NULL;
            CHECK(vx_execution_context_execute_graph(decode_ctx, batch_inputs, 3, &init_opts,
                &decode_res, &report) == VX_STATUS_OK);
            vx_result_release(decode_res); decode_res = NULL;
            const int32_t step_positions[2][4] = {{1, 2, -1, 2}, {2, 3, 1, 3}};
            const int32_t step_tokens[2][4] = {{3, 0, 0, 2}, {1, 3, 0, 1}};
            VxTensorReference reuse = {"lengths", NULL};
            VxGraphExecutionOptions step_opts = {
                .entrypoint = "transition",
                .range = VX_EXECUTION_RANGE_SLOT_ROWS,
                .slot_count = 4,
                .tensors = {.references = &reuse, .reference_count = 1}
            };
            VxGraphExecutionOptions eval_opts = {.entrypoint = "evaluate", .range = VX_EXECUTION_RANGE_FULL};
            for (int step = 0; step < 2; ++step) {
                for (int slot = 0; slot < 4; ++slot) {
                    const int32_t pos = step_positions[step][slot];
                    if (pos >= 0) {
                        batch_tokens[slot * 4 + pos] = step_tokens[step][slot];
                        batch_keep[slot * 4 + pos] = 1;
                    }
                }
                step_opts.slot_positions = step_positions[step];
                CHECK(vx_execution_context_execute_graph(decode_ctx, batch_inputs, 2, &step_opts,
                    &decode_res, &report) == VX_STATUS_OK);
                CHECK(inspect(decode_ctx).mode == VX_DECODE_MODE_ROW);
                CHECK(vx_execution_context_execute_graph(ref_ctx, batch_inputs, 3, &eval_opts,
                    &ref_res, &report) == VX_STATUS_OK);
                float got_logits[64] = {0}, ref_logits[64] = {0};
                int32_t got_ids[16] = {0}, ref_ids[16] = {0};
                size_t bytes = 0;
                CHECK(vx_result_read(decode_res, "logits", got_logits, sizeof(got_logits), &bytes, &report) == VX_STATUS_OK);
                CHECK(bytes == sizeof(got_logits));
                CHECK(vx_result_read(ref_res, "logits", ref_logits, sizeof(ref_logits), &bytes, &report) == VX_STATUS_OK);
                CHECK(bytes == sizeof(ref_logits));
                CHECK(vx_result_read(decode_res, "ids", got_ids, sizeof(got_ids), &bytes, &report) == VX_STATUS_OK);
                CHECK(bytes == sizeof(got_ids));
                CHECK(vx_result_read(ref_res, "ids", ref_ids, sizeof(ref_ids), &bytes, &report) == VX_STATUS_OK);
                CHECK(bytes == sizeof(ref_ids));
                for (int slot = 0; slot < 4; ++slot) {
                    const int active_len = (int)inspect(decode_ctx).active_lengths[slot];
                    for (int pos = 0; pos < active_len; ++pos) {
                        const int row = slot * 4 + pos;
                        CHECK(got_ids[row] == ref_ids[row]);
                        for (int col = 0; col < 4; ++col)
                            CHECK(fabsf(got_logits[row * 4 + col] - ref_logits[row * 4 + col]) <= 1e-6f);
                    }
                }
                vx_result_release(decode_res); decode_res = NULL;
                vx_result_release(ref_res); ref_res = NULL;
            }
            vx_execution_context_release(decode_ctx);
            vx_execution_context_release(ref_ctx);
            vx_compiled_model_release(compiled);
        }
        vx_model_release(model);
    }
    CHECK(!remove(weight_path));
}
