/* Compact source weights must preserve the F32 graph contract without a
 * model-sized expansion. Shared unsupported consumers must still materialize. */
#include "safetensors.h"
#include "compiled_weight_resources.h"

static const char lookup_storage_graph[] =
    "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},"
    "\"inputs\":{\"ids\":{\"dtype\":\"int32\",\"shape\":[2]},"
    "\"columns\":{\"dtype\":\"int32\",\"shape\":[2]}},\"nodes\":["
    "{\"id\":\"embedding\",\"opType\":\"Embedding\",\"inputs\":{\"input\":\"ids\",\"weight\":\"w\"},"
    "\"outputs\":{\"out\":{\"tensor\":\"e\",\"dtype\":\"float32\",\"shape\":[2,3]}},\"params\":{}},"
    "{\"id\":\"rows\",\"opType\":\"Gather\",\"inputs\":{\"input\":\"w\",\"indices\":\"ids\"},"
    "\"outputs\":{\"out\":{\"tensor\":\"g\",\"dtype\":\"float32\",\"shape\":[2,3]}},\"params\":{\"axis\":0}},"
    "{\"id\":\"columns\",\"opType\":\"Gather\",\"inputs\":{\"input\":\"w\",\"indices\":\"columns\"},"
    "\"outputs\":{\"out\":{\"tensor\":\"h\",\"dtype\":\"float32\",\"shape\":[4,2]}},\"params\":{\"axis\":-1}}],"
    "\"outputs\":[\"e\",\"g\",\"h\"],\"execution\":{\"stateless\":{},\"entrypoints\":[{"
    "\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],"
    "\"inputs\":[\"ids\",\"columns\"],\"outputs\":[\"e\",\"g\",\"h\"]}]}}";

static void lookup_storage_output(VxResult* result, const char* name,
                                  const float* expected, size_t count) {
    float actual[12]; size_t bytes = 0;
    CHECK(count <= sizeof(actual) / sizeof(actual[0]));
    CHECK(vx_result_read(result, name, actual, sizeof(actual), &bytes, &report) == VX_STATUS_OK);
    CHECK(bytes == count * sizeof(float));
    for (size_t i = 0; i < count; i++) {
        if (isnan(expected[i])) CHECK(isnan(actual[i]));
        else CHECK(!memcmp(&actual[i], &expected[i], sizeof(float)));
    }
}

static uint64_t lookup_storage_converted_bytes(const VxMemoryInventory* inventory) {
    uint64_t bytes = 0;
    CHECK(!inventory->truncated);
    for (size_t i = 0; i < inventory->count; i++)
        if (inventory->items[i].allocator == VX_MEMORY_HOST_PACKED_WEIGHTS &&
            !strcmp(inventory->items[i].name, "w"))
            bytes += inventory->items[i].capacity;
    return bytes;
}

static void lookup_storage_test(VxRuntime* runtime, const VxBackendPolicy* policy) {
    const VxDataType types[] = {VX_DTYPE_F16, VX_DTYPE_BF16};
    const uint16_t bits[][12] = {
        {0x3c00, 0xc000, 0x0001, 0x7bff, 0x8000, 0x3555,
         0x7c00, 0xfc00, 0x7e00, 0x0400, 0x3c01, 0xbc00},
        {0x3f80, 0xc000, 0x0001, 0x7f7f, 0x8000, 0x3eab,
         0x7f80, 0xff80, 0x7fc0, 0x0080, 0x3f81, 0xbf80}
    };
    const float values[][12] = {
        {1, -2, 0x1p-24f, 65504, -0.0f, 0x1.554p-2f,
         INFINITY, -INFINITY, NAN, 0x1p-14f, 0x1.004p0f, -1},
        {1, -2, 0x1p-133f, 0x1.fep127f, -0.0f, 0x1.56p-2f,
         INFINITY, -INFINITY, NAN, 0x1p-126f, 0x1.02p0f, -1}
    };
    const int shape[] = {4, 3};
    for (size_t dtype = 0; dtype < 2; dtype++) {
        SafetensorsFile weights = {0};
        CHECK(!safetensors_init_empty(&weights, SAFETENSORS_OPEN_READ_WRITE));
        CHECK(!safetensors_add_tensor(&weights, "w", types[dtype], shape, 2,
            bits[dtype], sizeof(bits[dtype])));
        unsigned char* shard = NULL; size_t shard_bytes = 0;
        CHECK(!safetensors_serialize(&weights, &shard, &shard_bytes));
        for (int variant = 0; variant < 4; variant++) {
            int expanded = variant == 1 || variant == 2;
            cJSON* graph = cJSON_Parse(lookup_storage_graph); CHECK(graph);
            cJSON* outputs = cJSON_GetObjectItemCaseSensitive(graph, "outputs");
            if (variant == 1) {
                cJSON* identity = cJSON_Parse(
                    "{\"id\":\"copy\",\"opType\":\"Identity\",\"inputs\":{\"input\":\"w\"},"
                    "\"outputs\":{\"out\":{\"tensor\":\"extra\",\"dtype\":\"float32\",\"shape\":[4,3]}},\"params\":{}}");
                CHECK(identity && cJSON_AddItemToArray(cJSON_GetObjectItemCaseSensitive(graph, "nodes"), identity));
                CHECK(cJSON_AddItemToArray(outputs, cJSON_CreateString("extra")));
            } else if (variant == 2) {
                CHECK(cJSON_AddItemToArray(outputs, cJSON_CreateString("w")));
            } else if (variant == 3) {
                cJSON* nodes = cJSON_GetObjectItemCaseSensitive(graph, "nodes");
                cJSON_DeleteItemFromArray(nodes, 2);
                cJSON_DeleteItemFromArray(nodes, 0);
                CHECK(cJSON_ReplaceItemInObjectCaseSensitive(graph, "outputs", cJSON_Parse("[\"g\"]")));
                outputs = cJSON_GetObjectItemCaseSensitive(graph, "outputs");
                CHECK(cJSON_AddItemToObject(graph, "banks", cJSON_Parse("{\"w\":\"E\"}")));
                CHECK(cJSON_AddItemToObject(cJSON_GetObjectItemCaseSensitive(graph, "dimensions"),
                    "E", cJSON_Parse("{\"min\":1,\"max\":4}")));
            }
            cJSON* entrypoint = cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(
                cJSON_GetObjectItemCaseSensitive(graph, "execution"), "entrypoints"), 0);
            CHECK(cJSON_ReplaceItemInObjectCaseSensitive(entrypoint, "outputs", cJSON_Duplicate(outputs, 1)));
            VxCompiledWeightConversions* conversions = NULL;
            CHECK(!vx_compiled_weight_conversions_create(&weights, 1, NULL, 0,
                VX_PORTABLE_BACKEND_KIND, graph, &conversions));
            CHECK(conversions && conversions->materialized_bytes == (expanded ? 12u * sizeof(float) : 0u));
            vx_compiled_weight_conversions_destroy(conversions);
            /* A device backend retains its existing full BF16 conversion. */
            CHECK(!vx_compiled_weight_conversions_create(&weights, 1, NULL, 0,
                VX_BACKEND_KIND_VULKAN, graph, &conversions));
            CHECK(conversions && conversions->materialized_bytes == (dtype ? 12u * sizeof(float) : 0u));
            vx_compiled_weight_conversions_destroy(conversions);
            char* document = cJSON_PrintUnformatted(graph); CHECK(document);
            cJSON_Delete(graph);
            VxSourceBytes source_bytes = {shard, shard_bytes};
            VxModelPackageSource package = {{document, strlen(document)}, &source_bytes, 1};
            VxModelSource source = VX_MODEL_SOURCE_INIT;
            VxModel* model = NULL; VxCompiledModel* compiled = NULL;
            CHECK(vx_runtime_internal_load_model_package(runtime, &source, &package,
                &model, &report, 0) == VX_STATUS_OK && model);
            cJSON_free(document);
            CHECK(vx_model_compile_options_revision(model, policy, 0, NULL,
                VX_PACKED_ARITHMETIC_F32, &compiled, &report) == VX_STATUS_OK && compiled);
            VxMemoryAllocation allocations[256];
            VxMemoryInventory inventory = {.items = allocations, .capacity = 256};
            vx_compiled_memory_inventory(compiled, &inventory);
            CHECK(lookup_storage_converted_bytes(&inventory) == (expanded ? 12u * sizeof(float) : 0u));
            VxExecutionConfig config = {.context = VX_CONTEXT_OPTIONS_INIT};
            VxExecutionContext* contexts[2] = {NULL, NULL};
            for (size_t i = 0; i < 2; i++)
                CHECK(vx_compiled_model_create_execution_context(compiled, &config,
                    &contexts[i], &report) == VX_STATUS_OK && contexts[i]);
            vx_model_release(model); vx_compiled_model_release(compiled);
            const int32_t ids[] = {3, 1}, columns[] = {-1, 0};
            VxTensorBinding inputs[] = {
                shaped("ids", VX_DTYPE_I32, ids, 1, 2, 0),
                shaped("columns", VX_DTYPE_I32, columns, 1, 2, 0)
            };
            float rows[6], gathered_columns[8];
            memcpy(rows, values[dtype] + 9, 3 * sizeof(float));
            memcpy(rows + 3, values[dtype] + 3, 3 * sizeof(float));
            for (size_t row = 0; row < 4; row++) {
                gathered_columns[row * 2] = values[dtype][row * 3 + 2];
                gathered_columns[row * 2 + 1] = values[dtype][row * 3];
            }
            VxGraphExecutionOptions options = {.entrypoint = "evaluate", .range = VX_EXECUTION_RANGE_FULL};
            for (size_t i = 0; i < 2; i++) {
                VxResult* result = NULL;
                CHECK(vx_execution_context_execute_graph(contexts[i], inputs, 2,
                    &options, &result, &report) == VX_STATUS_OK && result);
                if (variant != 3) lookup_storage_output(result, "e", rows, 6);
                lookup_storage_output(result, "g", rows, 6);
                if (variant != 3) lookup_storage_output(result, "h", gathered_columns, 8);
                if (expanded) lookup_storage_output(result, variant == 1 ? "extra" : "w", values[dtype], 12);
                VxMemoryView view;
                CHECK(vx_context_memory_inventory(contexts[i], &inventory, &view) == VX_STATUS_OK);
                CHECK(lookup_storage_converted_bytes(&inventory) == (expanded ? 12u * sizeof(float) : 0u));
                vx_execution_context_release(contexts[i]);
                lookup_storage_output(result, variant == 3 ? "g" : "e", rows, 6);
                vx_result_release(result);
            }
        }
        free(shard); safetensors_free(&weights);
    }
    puts("compact lookup storage: F16/BF16 numerics, shared consumers and ownership passed");
}
