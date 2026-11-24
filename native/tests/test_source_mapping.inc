/* Source storage must preserve the accepted numerical revision and its owner
 * lifetime. These tests deliberately change file names before compilation,
 * when reopening caller paths would read a different model or fail. */
#if defined(__unix__) || defined(__APPLE__)
#include "source_sha256.h"
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#if VOLVOXAI_ENABLE_TRAINING
#include "runtime_state.h"
#endif

static const char mapping_graph[] =
    "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},"
    "\"inputs\":{\"x\":{\"dtype\":\"float32\",\"shape\":[2]}},\"nodes\":["
    "{\"id\":\"first\",\"opType\":\"Add\",\"inputs\":{\"a\":\"x\",\"b\":\"w1\"},"
    "\"outputs\":{\"out\":{\"tensor\":\"sum\",\"dtype\":\"float32\",\"shape\":[2]}},\"params\":{}},"
    "{\"id\":\"second\",\"opType\":\"Add\",\"inputs\":{\"a\":\"sum\",\"b\":\"w2\"},"
    "\"outputs\":{\"out\":{\"tensor\":\"y\",\"dtype\":\"float32\",\"shape\":[2]}},\"params\":{}}],"
    "\"outputs\":[\"y\"],\"execution\":{\"stateless\":{},\"entrypoints\":[{"
    "\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\"],\"outputs\":[\"y\"]}]}}";

static void mapping_write(const char* path, const void* bytes, size_t size) {
    FILE* file = fopen(path, "wb"); CHECK(file);
    CHECK(fwrite(bytes, 1, size, file) == size && !fclose(file));
}

static void mapping_file_is(const char* path, const void* bytes, size_t size) {
    FILE* file = fopen(path, "rb"); CHECK(file);
    unsigned char* actual = malloc(size); CHECK(actual);
    CHECK(fread(actual, 1, size, file) == size && fgetc(file) == EOF);
    CHECK(!ferror(file) && !fclose(file) && !memcmp(actual, bytes, size));
    free(actual);
}

static void mapping_typed_shard(const char* name, VxDataType dtype,
                                const void* values, size_t value_bytes,
                                unsigned char** bytes, size_t* size) {
    const int shape[] = {2};
    SafetensorsFile file = {0};
    CHECK(!safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE));
    CHECK(!safetensors_add_tensor(&file, name, dtype,
        shape, 1, values, value_bytes));
    CHECK(!safetensors_serialize(&file, bytes, size));
    safetensors_free(&file);
}

static void mapping_shard(const char* name, const float values[2],
                          unsigned char** bytes, size_t* size) {
    mapping_typed_shard(name, VX_DTYPE_F32, values, 2 * sizeof(float), bytes, size);
}

static size_t mapping_entries(const char* path, const char* prefix) {
    DIR* directory = opendir(path); CHECK(directory);
    size_t count = 0;
    struct dirent* entry;
    while ((entry = readdir(directory)) != NULL) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        if (!prefix || !strncmp(entry->d_name, prefix, strlen(prefix))) count++;
    }
    CHECK(!closedir(directory));
    return count;
}

static size_t mapping_live_maps(const char* path) {
#if defined(__linux__)
    FILE* file = fopen("/proc/self/maps", "r"); CHECK(file);
    char line[8192]; size_t count = 0;
    while (fgets(line, sizeof(line), file)) if (strstr(line, path)) count++;
    CHECK(!ferror(file) && !fclose(file));
    return count;
#else
    (void)path; return 0;
#endif
}

static size_t mapping_live_fds(void) {
#if defined(__linux__)
    return mapping_entries("/proc/self/fd", NULL);
#else
    return 0;
#endif
}

static VxModelSourceIntegrity mapping_integrity(const void* bytes, size_t size,
                                                unsigned char digest[32]) {
    VxSourceSha256 hash;
    vx_source_sha256_init(&hash);
    vx_source_sha256_update(&hash, bytes, size);
    vx_source_sha256_finish(&hash, digest);
    return (VxModelSourceIntegrity){size, digest, 32};
}

static size_t mapping_inventory_weights(const VxMemoryInventory* inventory,
                                       uint64_t* bytes) {
    size_t count = 0; *bytes = 0;
    CHECK(!inventory->truncated);
    for (size_t i = 0; i < inventory->count; i++) {
        const VxMemoryAllocation* item = &inventory->items[i];
        if (item->space != VX_MEMORY_SPACE_MAPPED_FILE) continue;
        CHECK(item->allocator == VX_MEMORY_HOST_WEIGHTS && item->key && item->capacity);
        CHECK(item->owner_kind == VX_MEMORY_OWNER_MODEL);
        *bytes += item->capacity; count++;
    }
    return count;
}

static void mapping_model_bytes(VxModel* model, size_t expected_count, uint64_t expected_bytes) {
    VxMemoryAllocation allocations[256];
    VxMemoryInventory inventory = {.items = allocations, .capacity = 256};
    uint64_t bytes = 0;
    vx_model_memory_inventory(model, &inventory);
    CHECK(mapping_inventory_weights(&inventory, &bytes) == expected_count && bytes == expected_bytes);
}

static void mapping_sparse_load(VxRuntime* runtime, const char* graph,
    const char* weights, const char* temp, const char* directory, size_t baseline_fds) {
    const char* enabled = getenv("VOLVOXAI_TEST_MMAP_LARGE");
    if (!enabled || strcmp(enabled, "1") || sizeof(off_t) < 8 || sizeof(size_t) < 8) return;
    static const char passthrough[] =
        "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},"
        "\"inputs\":{\"x\":{\"dtype\":\"float32\",\"shape\":[2]}},\"nodes\":[],"
        "\"outputs\":[\"x\"],\"execution\":{\"stateless\":{},\"entrypoints\":[{"
        "\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\"],\"outputs\":[\"x\"]}]}}";
    static const char header[] =
        "{\"unused\":{\"dtype\":\"F32\",\"shape\":[1073741825],\"data_offsets\":[0,4294967300]}}";
    const size_t header_bytes = (sizeof(header) - 1u + 7u) & ~(size_t)7u;
    const uint64_t file_bytes = UINT64_C(4294967300) + 8u + header_bytes;
    unsigned char prefix[8];
    for (size_t i = 0; i < 8; i++) prefix[i] = (unsigned char)((uint64_t)header_bytes >> (i * 8));
    FILE* file = fopen(weights, "wb"); CHECK(file);
    CHECK(fwrite(prefix, 1, sizeof(prefix), file) == sizeof(prefix));
    CHECK(fwrite(header, 1, sizeof(header) - 1u, file) == sizeof(header) - 1u);
    for (size_t i = sizeof(header) - 1u; i < header_bytes; i++) CHECK(fputc(' ', file) != EOF);
    CHECK(!fflush(file) && !ftruncate(fileno(file), (off_t)file_bytes) && !fclose(file));
    mapping_write(graph, passthrough, sizeof(passthrough) - 1u);
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    source.graph_path = graph; source.weight_paths = &weights; source.weight_path_count = 1;
    VxModel* model = NULL;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK && model);
    CHECK(mapping_entries(temp, NULL) == 1);
#if defined(__linux__)
    CHECK(mapping_live_maps(weights));
#endif
    mapping_model_bytes(model, 1, file_bytes);
    vx_model_release(model);
    CHECK(!mapping_live_maps(directory) && !mapping_entries(temp, NULL));
    CHECK(mapping_live_fds() == baseline_fds);
    puts("native source mapping: optional sparse >4 GiB LoadModel passed");
}

static void mapping_execute_expected(VxModel* model, const VxBackendPolicy* policy,
                                     const float expected[2]) {
    VxCompiledModel* compiled = NULL; VxExecutionContext* context = NULL;
    VxExecutionConfig config = {.context = VX_CONTEXT_OPTIONS_INIT};
    VxMemoryAllocation allocations[256];
    VxMemoryInventory inventory = {.items = allocations, .capacity = 256};
    uint64_t model_bytes = 0, combined_bytes = 0;
    vx_model_memory_inventory(model, &inventory);
    size_t mapped_count = mapping_inventory_weights(&inventory, &model_bytes);
    CHECK(vx_model_compile_options_revision(model, policy, 0, NULL,
        VX_PACKED_ARITHMETIC_F32, &compiled, &report) == VX_STATUS_OK && compiled);
    if (mapped_count) {
        vx_compiled_memory_inventory(compiled, &inventory);
        CHECK(mapping_inventory_weights(&inventory, &combined_bytes) == mapped_count);
        CHECK(combined_bytes == model_bytes);
    }
    vx_model_release(model);
    CHECK(vx_compiled_model_create_execution_context(compiled, &config,
        &context, &report) == VX_STATUS_OK && context);
    if (mapped_count) {
        VxMemoryView view;
        CHECK(vx_context_memory_inventory(context, &inventory, &view) == VX_STATUS_OK);
        CHECK(mapping_inventory_weights(&inventory, &combined_bytes) == mapped_count);
        CHECK(combined_bytes == model_bytes);
    }
    vx_compiled_model_release(compiled);
    const float input[] = {10, 20};
    VxTensorBinding binding = tensor("x", input);
    VxGraphExecutionOptions options = {.entrypoint = "evaluate", .range = VX_EXECUTION_RANGE_FULL};
    VxResult* retained = NULL;
    for (int i = 0; i < 2; i++) {
        VxResult* result = NULL;
        CHECK(vx_execution_context_execute_graph(context, &binding, 1,
            &options, &result, &report) == VX_STATUS_OK);
        output_is(result, "y", expected);
        if (retained) vx_result_release(retained);
        retained = result;
    }
    vx_execution_context_release(context);
    output_is(retained, "y", expected);
    vx_result_release(retained);
}

static void mapping_execute(VxModel* model, const VxBackendPolicy* policy) {
    const float expected[] = {11, 23};
    mapping_execute_expected(model, policy, expected);
}

#if VOLVOXAI_ENABLE_TRAINING
static void mapping_authoring_copy_expected(VxModel* model, const float expected[2]) {
    VxWeightRevisionRecord* revision = NULL;
    VxEngineState* state = NULL;
    CHECK(vx_model_internal_accept_full_owner(model, &revision, &report) == VX_STATUS_OK);
    CHECK(vx_model_internal_create_authoring_engine(model, revision,
        VX_BACKEND_KIND_NATIVE_CPU, NULL, &state, &report) == VX_STATUS_OK && state);
    CHECK(state->weight_file_count == 2 && !state->weight_files_borrowed);
    for (int i = 0; i < state->weight_file_count; i++)
        CHECK(!state->weight_files[i].is_mmap && !state->weight_files[i].borrows_storage);
    VxEngineStateScope scope = vx_engine_state_scope_enter(state);
    const float input[] = {10, 20}, changed[] = {100, 200};
    float output[2];
    CHECK(!volvoxai_engine_set_input_f32("x", input, 2));
    CHECK(!volvoxai_engine_forward());
    CHECK(!volvoxai_engine_copy_tensor_f32("y", output, 2));
    CHECK(output[0] == expected[0] && output[1] == expected[1]);
    int weight_dtype = 0;
    CHECK(!volvoxai_engine_tensor_info_ex("w1", NULL, NULL, NULL, &weight_dtype, NULL));
    if (weight_dtype == T_F16) {
        const uint16_t changed_f16[] = {0x5640, 0x5a40}; /* 100, 200 */
        CHECK(!volvoxai_engine_set_tensor_raw("w1", T_F16, changed_f16, sizeof(changed_f16)));
    } else {
        CHECK(weight_dtype == T_F32);
        CHECK(!volvoxai_engine_set_tensor_f32("w1", changed, 2));
    }
    CHECK(!volvoxai_engine_forward());
    CHECK(!volvoxai_engine_copy_tensor_f32("y", output, 2));
    CHECK(output[0] == 113 && output[1] == 219);
    vx_engine_state_scope_leave(scope);
    vx_model_internal_destroy_authoring_engine(state);
    vx_model_internal_release_weight_revision(revision);
}

static void mapping_authoring_copy(VxModel* model) {
    const float expected[] = {11, 23};
    mapping_authoring_copy_expected(model, expected);
}

static void mapping_context_is(VxExecutionContext* context, const float expected[2]) {
    const float input[] = {10, 20};
    VxTensorBinding binding = tensor("x", input);
    VxGraphExecutionOptions options = {.entrypoint = "evaluate", .range = VX_EXECUTION_RANGE_FULL};
    VxResult* result = NULL;
    CHECK(vx_execution_context_execute_graph(context, &binding, 1,
        &options, &result, &report) == VX_STATUS_OK);
    output_is(result, "y", expected);
    vx_result_release(result);
}
#endif

static void source_mapping_test(VxRuntime* runtime, const char* fixture_path,
                                const VxBackendPolicy* policy) {
    char directory[4096], temp[4096], graph[4096], first[4096], second[4096], retired[4096];
    int length = snprintf(directory, sizeof(directory), "%s.mapping-XXXXXX", fixture_path);
    CHECK(length > 0 && (size_t)length < sizeof(directory) && mkdtemp(directory));
#define MAPPING_PATH(target, suffix) do { \
    length = snprintf(target, sizeof(target), "%s/%s", directory, suffix); \
    CHECK(length > 0 && (size_t)length < sizeof(target)); \
} while (0)
    MAPPING_PATH(temp, "snapshots"); MAPPING_PATH(graph, "graph.json");
    MAPPING_PATH(first, "first.safetensors"); MAPPING_PATH(second, "second.safetensors");
    MAPPING_PATH(retired, "retired.safetensors");
#undef MAPPING_PATH
    CHECK(!mkdir(temp, 0700));
    const char* old_temp = getenv("TMPDIR");
    char* saved_temp = old_temp ? strdup(old_temp) : NULL;
    CHECK(!old_temp || saved_temp);
    CHECK(!setenv("TMPDIR", temp, 1));
    mapping_write(graph, mapping_graph, sizeof(mapping_graph) - 1u);
    const float w1[] = {-2, 4}, w2[] = {3, -1}, changed[] = {100, 200};
    unsigned char *data[2] = {NULL, NULL}, *replacement = NULL;
    size_t sizes[2] = {0}, replacement_size = 0;
    mapping_shard("w1", w1, &data[0], &sizes[0]);
    mapping_shard("w2", w2, &data[1], &sizes[1]);
    mapping_shard("w1", changed, &replacement, &replacement_size);
    const char* paths[] = {first, second};
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    source.graph_path = graph; source.weight_paths = paths; source.weight_path_count = 2;
    const size_t baseline_fds = mapping_live_fds();

    /* Native paths keep the exact accepted mapped file object, even when the
     * old names disappear before CompileModel. */
    {
        mapping_write(first, data[0], sizes[0]); mapping_write(second, data[1], sizes[1]);
        VxModel* model = NULL;
        CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK && model);
        CHECK(mapping_entries(temp, NULL) == 1); /* graph only */
        mapping_model_bytes(model, 2, sizes[0] + sizes[1]);
#if defined(__linux__)
        CHECK(mapping_live_maps(first) && mapping_live_maps(second));
#endif
        CHECK(!rename(first, retired));
        mapping_write(first, replacement, replacement_size);
        CHECK(!remove(retired) && !remove(second));
        VxGraphPlan* plan = NULL;
        CHECK(vx_graph_plan_internal_create_model(model, &plan, &report) == VX_STATUS_OK && plan);
        vx_graph_plan_internal_release(plan);
        mapping_execute(model, policy);
        CHECK(!mapping_live_maps(directory));
        CHECK(mapping_live_fds() == baseline_fds && !mapping_entries(temp, NULL));
        /* Releasing a direct source must not unlink a new file at its old name. */
        mapping_file_is(first, replacement, replacement_size);
    }

#if VOLVOXAI_ENABLE_TRAINING
    /* Full authoring receives independent writable storage from the accepted
     * mapped objects, even after both original names have disappeared. */
    mapping_write(first, data[0], sizes[0]); mapping_write(second, data[1], sizes[1]);
    VxModel* authoring_model = NULL;
    CHECK(vx_runtime_load_model(runtime, &source, &authoring_model, &report) == VX_STATUS_OK);
    CHECK(!rename(first, retired) && !remove(second));
    mapping_write(first, replacement, replacement_size);
    mapping_authoring_copy(authoring_model);
    mapping_file_is(retired, data[0], sizes[0]);
    mapping_file_is(first, replacement, replacement_size);
    mapping_execute(authoring_model, policy);
    CHECK(!remove(retired));
    CHECK(!mapping_live_maps(directory) && !mapping_entries(temp, NULL));
    CHECK(mapping_live_fds() == baseline_fds);

    /* Trainer publication retains newly mapped export files after their
     * temporary names disappear; existing compiled owners keep the old bytes. */
    mapping_write(first, data[0], sizes[0]); mapping_write(second, data[1], sizes[1]);
    VxModel* revised_model = NULL;
    CHECK(vx_runtime_load_model(runtime, &source, &revised_model, &report) == VX_STATUS_OK);
    VxWeightRevisionRecord *base_revision = NULL, *successor = NULL;
    CHECK(vx_model_internal_accept_full_owner(revised_model, &base_revision, &report) == VX_STATUS_OK);
    VxCompiledModel* old_compiled = NULL;
    VxExecutionContext* old_context = NULL;
    VxExecutionConfig old_config = {.context = VX_CONTEXT_OPTIONS_INIT};
    CHECK(vx_model_compile_options_revision(revised_model, policy, 0, NULL,
        VX_PACKED_ARITHMETIC_F32, &old_compiled, &report) == VX_STATUS_OK);
    CHECK(vx_compiled_model_create_execution_context(old_compiled, &old_config,
        &old_context, &report) == VX_STATUS_OK);
    char successor_paths[2][4096];
    const char* successor_files[] = {successor_paths[0], successor_paths[1]};
    for (int i = 0; i < 2; i++) {
        length = snprintf(successor_paths[i], sizeof(successor_paths[i]),
            "%s/successor-%d.safetensors", directory, i);
        CHECK(length > 0 && (size_t)length < sizeof(successor_paths[i]));
    }
    const float next_w1[] = {7, 9}, old_expected[] = {11, 23}, next_expected[] = {20, 28};
    unsigned char* next_bytes = NULL; size_t next_size = 0;
    mapping_shard("w1", next_w1, &next_bytes, &next_size);
    mapping_write(successor_files[0], next_bytes, next_size);
    mapping_write(successor_files[1], data[1], sizes[1]);
    free(next_bytes);
    CHECK(vx_model_internal_prepare_weight_revision(revised_model, successor_files,
        2, &successor, &report) == VX_STATUS_OK && successor);
    CHECK(mapping_entries(temp, NULL) == 1); /* graph only */
#if defined(__linux__)
    CHECK(mapping_live_maps(successor_files[0]) && mapping_live_maps(successor_files[1]));
#endif
    CHECK(!remove(successor_files[0]) && !remove(successor_files[1]));
    CHECK(vx_model_internal_publish_weight_revision(revised_model, base_revision,
        successor, &report) == VX_STATUS_OK);
    vx_model_internal_release_weight_revision(base_revision);
    vx_model_internal_release_weight_revision(successor);
    mapping_model_bytes(revised_model, 2, next_size + sizes[1]);
    mapping_context_is(old_context, old_expected);
    mapping_authoring_copy_expected(revised_model, next_expected);
    CHECK(mapping_entries(temp, NULL) == 1);
    mapping_execute_expected(revised_model, policy, next_expected);
    mapping_context_is(old_context, old_expected);
    vx_compiled_model_release(old_compiled);
    vx_execution_context_release(old_context);
    CHECK(!mapping_live_maps(directory) && !mapping_entries(temp, NULL));
    CHECK(mapping_live_fds() == baseline_fds);
#endif

    /* Inline packages own accepted bytes independently of the caller's byte
     * buffers; their transport storage is not a native-path loading choice. */
    VxSourceBytes bytes[] = {{data[0], sizes[0]}, {data[1], sizes[1]}};
    VxModelPackageSource package = {{mapping_graph, sizeof(mapping_graph) - 1u}, bytes, 2};
    VxModelSource inline_source = VX_MODEL_SOURCE_INIT;
    VxModel* model = NULL;
    CHECK(vx_runtime_internal_load_model_package(runtime, &inline_source, &package,
        &model, &report, 0) == VX_STATUS_OK && model);
    CHECK(mapping_entries(temp, NULL) == 3);
    CHECK(!mapping_live_maps(first) && !mapping_live_maps(second));
    CHECK(sizes[0] == replacement_size);
    memcpy(data[0], replacement, replacement_size);
    mapping_execute(model, policy);
    free(data[0]);
    mapping_shard("w1", w1, &data[0], &sizes[0]);
    CHECK(!mapping_entries(temp, NULL) && mapping_live_fds() == baseline_fds);

    /* An observed direct source mutation is rejected rather than compiling a
     * mixture of accepted metadata and newly written payload. */
    mapping_write(first, data[0], sizes[0]); mapping_write(second, data[1], sizes[1]);
    model = NULL;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK && model);
    struct stat original_stat; CHECK(!stat(first, &original_stat));
    mapping_write(first, replacement, replacement_size);
    const struct timespec changed_time[] = {{0, UTIME_OMIT}, {original_stat.st_mtime + 1, 0}};
    CHECK(!utimensat(AT_FDCWD, first, changed_time, 0));
    VxCompiledModel* compiled = NULL;
    CHECK(vx_model_compile_options_revision(model, policy, 0, NULL,
        VX_PACKED_ARITHMETIC_F32, &compiled, &report) == VX_STATUS_REVISION_CONFLICT && !compiled);
    CHECK(report.code == VX_CODE_REVISION_CONFLICT);
    const char* mutation_backends[] = {"cpu", "missing-provider"};
    VxBackendPolicy mutation_policy = VX_BACKEND_POLICY_INIT;
    mutation_policy.backends = mutation_backends; mutation_policy.backend_count = 2;
    /* A later unavailable route must not replace the source-mutation error. */
    CHECK(vx_model_compile_options_revision(model, &mutation_policy, 0, NULL,
        VX_PACKED_ARITHMETIC_F32, &compiled, &report) == VX_STATUS_REVISION_CONFLICT && !compiled);
    CHECK(report.code == VX_CODE_REVISION_CONFLICT);
    VxGraphPlan* mutated_plan = NULL;
    CHECK(vx_graph_plan_internal_create_model(model, &mutated_plan, &report) == VX_STATUS_REVISION_CONFLICT && !mutated_plan);
    CHECK(report.code == VX_CODE_REVISION_CONFLICT);
    vx_model_release(model);
    CHECK(!mapping_live_maps(directory) && !mapping_entries(temp, NULL));

    /* Reusing a prepared engine must still recheck its retained source before
     * publishing a new context. */
    mapping_write(first, data[0], sizes[0]); model = NULL; compiled = NULL;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK && model);
    CHECK(vx_model_compile_options_revision(model, policy, 0, NULL,
        VX_PACKED_ARITHMETIC_F32, &compiled, &report) == VX_STATUS_OK && compiled);
    CHECK(!stat(first, &original_stat));
    mapping_write(first, replacement, replacement_size);
    const struct timespec after_compile_time[] = {{0, UTIME_OMIT}, {original_stat.st_mtime + 1, 0}};
    CHECK(!utimensat(AT_FDCWD, first, after_compile_time, 0));
    VxExecutionContext* refused_context = NULL;
    VxExecutionConfig context_config = {.context = VX_CONTEXT_OPTIONS_INIT};
    CHECK(vx_compiled_model_create_execution_context(compiled, &context_config,
        &refused_context, &report) == VX_STATUS_REVISION_CONFLICT && !refused_context);
    CHECK(report.code == VX_CODE_REVISION_CONFLICT);
    vx_compiled_model_release(compiled); vx_model_release(model);
    CHECK(!mapping_live_maps(directory) && !mapping_entries(temp, NULL));
    CHECK(mapping_live_fds() == baseline_fds);

    /* Both compact source formats keep their numbers under mmap ownership,
     * including the full profile's private writable conversion. */
    const VxDataType narrow_dtypes[] = {VX_DTYPE_F16, VX_DTYPE_BF16};
    const uint16_t narrow_values[][2][2] = {
        {{0xc000, 0x4400}, {0x4200, 0xbc00}}, /* F16: {-2,4}, {3,-1} */
        {{0xc000, 0x4080}, {0x4040, 0xbf80}}  /* BF16: same values */
    };
    for (size_t dtype = 0; dtype < 2; dtype++) {
        unsigned char* narrow[2] = {NULL, NULL}; size_t narrow_sizes[2] = {0, 0};
        mapping_typed_shard("w1", narrow_dtypes[dtype], narrow_values[dtype][0],
            sizeof(narrow_values[0][0]), &narrow[0], &narrow_sizes[0]);
        mapping_typed_shard("w2", narrow_dtypes[dtype], narrow_values[dtype][1],
            sizeof(narrow_values[0][1]), &narrow[1], &narrow_sizes[1]);
        mapping_write(first, narrow[0], narrow_sizes[0]);
        mapping_write(second, narrow[1], narrow_sizes[1]);
        model = NULL;
        CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK && model);
        mapping_model_bytes(model, 2, narrow_sizes[0] + narrow_sizes[1]);
        CHECK(!rename(first, retired) && !remove(second));
#if VOLVOXAI_ENABLE_TRAINING
        mapping_authoring_copy(model);
#endif
        mapping_file_is(retired, narrow[0], narrow_sizes[0]);
        mapping_execute(model, policy);
        CHECK(!remove(retired));
        CHECK(!mapping_live_maps(directory) && !mapping_entries(temp, NULL));
        CHECK(mapping_live_fds() == baseline_fds);
        free(narrow[0]); free(narrow[1]);
    }

    /* A manifest still validates all bytes in direct mode. A failed later
     * shard must release earlier mappings, graph snapshots and file handles. */
    mapping_write(first, data[0], sizes[0]); mapping_write(second, data[1], sizes[1]);
    unsigned char digests[3][32];
    VxModelSourceIntegrity graph_integrity = mapping_integrity(mapping_graph,
        sizeof(mapping_graph) - 1u, digests[0]);
    VxModelSourceIntegrity shards[] = {mapping_integrity(data[0], sizes[0], digests[1]),
        mapping_integrity(data[1], sizes[1], digests[2])};
    VxModelSourceOptions manifest = {.max_total_bytes = 4096, .max_shard_bytes = 4096,
        .graph = &graph_integrity, .weights = shards, .weight_count = 2};
    model = NULL;
    CHECK(vx_runtime_internal_load_model_sources(runtime, &source, NULL, &manifest,
        &model, &report, 0) == VX_STATUS_OK && model);
    mapping_execute(model, policy);
    digests[2][0] ^= 1u;
    for (int repeat = 0; repeat < 8; repeat++) {
        model = NULL;
        CHECK(vx_runtime_internal_load_model_sources(runtime, &source, NULL, &manifest,
            &model, &report, 0) != VX_STATUS_OK && !model);
        CHECK(report.code == VX_CODE_MODEL_SOURCE_INTEGRITY_MISMATCH);
        CHECK(!mapping_live_maps(directory) && !mapping_entries(temp, NULL));
        CHECK(mapping_live_fds() == baseline_fds);
    }
    digests[2][0] ^= 1u;
    /* Ordered raw-shard integrity mismatches must fail before safetensors
     * header parsing, including when an earlier non-safetensors shard matches
     * its manifest digest and a later shard mismatches. */
    const char raw_first[] = "first source";
    const char raw_second[] = "second source";
    mapping_write(first, raw_first, sizeof(raw_first) - 1u);
    mapping_write(second, raw_second, sizeof(raw_second) - 1u);
    unsigned char raw_digests[2][32];
    VxModelSourceIntegrity raw_shards[] = {
        mapping_integrity(raw_first, sizeof(raw_first) - 1u, raw_digests[0]),
        mapping_integrity(raw_second, sizeof(raw_second) - 1u, raw_digests[1]),
    };
    VxModelSourceOptions raw_manifest = {.max_total_bytes = 4096, .max_shard_bytes = 4096,
        .graph = &graph_integrity, .weights = raw_shards, .weight_count = 2};
    for (int broken = 0; broken < 2; broken++) {
        raw_digests[broken][0] ^= 1u;
        model = NULL;
        CHECK(vx_runtime_internal_load_model_sources(runtime, &source, NULL, &raw_manifest,
            &model, &report, 0) != VX_STATUS_OK && !model);
        CHECK(report.code == VX_CODE_MODEL_SOURCE_INTEGRITY_MISMATCH);
        CHECK(!mapping_live_maps(directory) && !mapping_entries(temp, NULL));
        CHECK(mapping_live_fds() == baseline_fds);
        raw_digests[broken][0] ^= 1u;
    }
    mapping_write(first, "", 0);
    model = NULL;
    CHECK(vx_runtime_internal_load_model_sources(runtime, &source, NULL, &raw_manifest,
        &model, &report, 0) != VX_STATUS_OK && !model);
    CHECK(report.code == VX_CODE_MODEL_SOURCE_INTEGRITY_MISMATCH);
    CHECK(!mapping_live_maps(directory) && !mapping_entries(temp, NULL));
    CHECK(mapping_live_fds() == baseline_fds);
    mapping_write(first, data[0], sizes[0]); mapping_write(second, data[1], sizes[1]);
    manifest.weight_count = 1;
    CHECK(vx_runtime_internal_load_model_sources(runtime, &source, NULL, &manifest,
        &model, &report, 0) == VX_STATUS_INVALID_ARGUMENT && !model);
    CHECK(!remove(second));
    for (int repeat = 0; repeat < 8; repeat++) {
        CHECK(vx_runtime_load_model(runtime, &source, &model, &report) != VX_STATUS_OK && !model);
        CHECK(!mapping_live_maps(directory) && !mapping_entries(temp, NULL));
        CHECK(mapping_live_fds() == baseline_fds);
    }

    mapping_sparse_load(runtime, graph, first, temp, directory, baseline_fds);
    CHECK(!remove(first) && !remove(graph) && !rmdir(temp) && !rmdir(directory));
    if (saved_temp) CHECK(!setenv("TMPDIR", saved_temp, 1));
    else CHECK(!unsetenv("TMPDIR"));
    free(saved_temp); free(data[0]); free(data[1]); free(replacement);
    puts("native source mapping: lifetime, package isolation, F16/BF16, inventory, manifests and cleanup passed");
}
#else
static void source_mapping_test(VxRuntime* runtime, const char* fixture_path,
                                const VxBackendPolicy* policy) {
    (void)runtime; (void)fixture_path; (void)policy;
}
#endif
