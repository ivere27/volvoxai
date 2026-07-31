#if VOLVOXAI_ENABLE_TRAINING
static int cuda_profile_path_seen(const char* path) {
    for (CudaProfilePathHistory* entry = cuda_profile_path_history;
         entry; entry = entry->next) {
        if (!strcmp(entry->path, path)) return 1;
    }
    return 0;
}

static void cuda_profile_register_function(CUfunction function,
                                           const char* name) {
    if (!cuda_profile.enabled || !function || !name) return;
    for (size_t index = 0; index < cuda_profile.function_count; index++) {
        if (cuda_profile.functions[index].function == function) return;
    }
    if (cuda_profile.function_count >= CUDA_PROFILE_MAX_FUNCTIONS) {
        cuda_profile.incomplete = 1;
        return;
    }
    cuda_profile.functions[cuda_profile.function_count++] =
        (CudaProfileFunction){function, name};
}

static const char* cuda_profile_function_name(CUfunction function) {
    for (size_t index = 0; index < cuda_profile.function_count; index++) {
        if (cuda_profile.functions[index].function == function)
            return cuda_profile.functions[index].name;
    }
    cuda_profile.incomplete = 1;
    return "unmapped";
}

static void cuda_profile_destroy_event(CUevent event) {
    if (!event) return;
    if (!p_cuEventDestroy || p_cuEventDestroy(event) != CUDA_SUCCESS)
        cuda_profile.incomplete = 1;
}

static void cuda_profile_discard_records(void) {
    for (size_t index = 0; index < cuda_profile.record_count; index++) {
        cuda_profile_destroy_event(cuda_profile.records[index].start);
        cuda_profile_destroy_event(cuda_profile.records[index].end);
    }
    cuda_profile.record_count = 0;
}

static void cuda_profile_begin_scope(int loaded_model_trainstep) {
    if (!cuda_profile.enabled) return;
    cuda_profile_discard_records();
    cuda_profile.incomplete = !cuda_profile.api_available;
    cuda_profile.scope_open = 1;
    cuda_profile.loaded_model_trainstep = loaded_model_trainstep != 0;
    cuda_profile.non_cuda_route = 0;
}

static void cuda_profile_configure(void) {
    const char* path = getenv("VOLVOXAI_CUDA_PROFILE_PATH");
    CudaProfilePathHistory* new_path = NULL;
    int seen_path;
    const char* mode;
    long output_size = 0;
    size_t path_length;
    if (!path || !path[0]) return;
    path_length = strlen(path);
    if (path_length >= CUDA_PROFILE_PATH_CAPACITY) {
        fprintf(stderr,
            "[CUDA] profile path is too long; profiling is disabled\n");
        return;
    }
    seen_path = cuda_profile_path_seen(path);
    mode = seen_path ? "a" : "w";
    if (!seen_path) {
        if (path_length > SIZE_MAX - sizeof(*new_path) - 1u) {
            fprintf(stderr,
                "[CUDA] profile path allocation overflow; profiling is disabled\n");
            return;
        }
        new_path = (CudaProfilePathHistory*)malloc(
            sizeof(*new_path) + path_length + 1u);
        if (!new_path) {
            fprintf(stderr,
                "[CUDA] profile path history allocation failed; profiling is disabled\n");
            return;
        }
        memcpy(new_path->path, path, path_length + 1u);
    }
    memset(&cuda_profile, 0, sizeof(cuda_profile));
    cuda_profile.enabled = 1;
    cuda_profile.api_available = p_cuEventCreate && p_cuEventRecord &&
        p_cuEventElapsedTime && p_cuEventDestroy;
    cuda_profile.incomplete = !cuda_profile.api_available;
    cuda_profile.output = fopen(path, mode);
    cuda_profile.owns_output = cuda_profile.output != NULL;
    if (cuda_profile.output &&
        setvbuf(cuda_profile.output, NULL, _IONBF, 0) != 0) {
        (void)fclose(cuda_profile.output);
        cuda_profile.output = NULL;
        cuda_profile.owns_output = 0;
    }
    if (cuda_profile.output && new_path) {
        new_path->next = cuda_profile_path_history;
        cuda_profile_path_history = new_path;
        new_path = NULL;
    }
    if (cuda_profile.output && seen_path &&
        fseek(cuda_profile.output, 0, SEEK_END) == 0)
        output_size = ftell(cuda_profile.output);
    cuda_profile.header_written = output_size > 0;
    if (!cuda_profile.output) {
        free(new_path);
        cuda_profile.output = stderr;
        cuda_profile.owns_output = 0;
        fprintf(stderr,
            "[CUDA] could not open profile path '%s'; using stderr\n",
            path);
    }
    if (!cuda_profile.api_available) {
        fprintf(cuda_profile.output,
            "# CUDA event timing is unavailable; profile is incomplete\n");
        fflush(cuda_profile.output);
    }
}

static void cuda_profile_shutdown(void) {
    cuda_profile_discard_records();
    free(cuda_profile.records);
    cuda_profile.records = NULL;
    cuda_profile.record_capacity = 0;
    if (cuda_profile.owns_output && cuda_profile.output &&
        fclose(cuda_profile.output) != 0)
        fprintf(stderr,
            "[CUDA] profile output close failed; final scope may be incomplete\n");
    memset(&cuda_profile, 0, sizeof(cuda_profile));
}

static void cuda_profile_launch_begin(CudaProfilePendingLaunch* pending,
                                      const CudaLaunchSignature* signature) {
    CudaProfileRecord* records;
    size_t capacity;
    if (!pending) return;
    memset(pending, 0, sizeof(*pending));
    if (!cuda_profile.enabled || !cuda_profile.scope_open ||
        !cuda_profile.api_available || !signature) return;
    if (cuda_profile.record_count >= cuda_profile.record_capacity) {
        capacity = cuda_profile.record_capacity
            ? cuda_profile.record_capacity * 2u : 1024u;
        if (capacity < cuda_profile.record_count + 1u ||
            capacity > SIZE_MAX / sizeof(*records)) {
            cuda_profile.incomplete = 1;
            return;
        }
        records = (CudaProfileRecord*)realloc(
            cuda_profile.records, capacity * sizeof(*records));
        if (!records) {
            cuda_profile.incomplete = 1;
            return;
        }
        cuda_profile.records = records;
        cuda_profile.record_capacity = capacity;
    }
    pending->signature = *signature;
    pending->name = cuda_profile_function_name(signature->function);
    if (p_cuEventCreate(&pending->start, 0u) != CUDA_SUCCESS ||
        p_cuEventCreate(&pending->end, 0u) != CUDA_SUCCESS ||
        p_cuEventRecord(pending->start, cuda_stream) != CUDA_SUCCESS) {
        cuda_profile_destroy_event(pending->start);
        cuda_profile_destroy_event(pending->end);
        memset(pending, 0, sizeof(*pending));
        cuda_profile.incomplete = 1;
        return;
    }
    pending->active = 1;
}

static void cuda_profile_launch_end(CudaProfilePendingLaunch* pending,
                                    int launched) {
    CudaProfileRecord* record;
    if (!pending || !pending->active) return;
    if (!launched || p_cuEventRecord(pending->end, cuda_stream) !=
            CUDA_SUCCESS) {
        cuda_profile_destroy_event(pending->start);
        cuda_profile_destroy_event(pending->end);
        cuda_profile.incomplete = 1;
        memset(pending, 0, sizeof(*pending));
        return;
    }
    record = &cuda_profile.records[cuda_profile.record_count++];
    record->start = pending->start;
    record->end = pending->end;
    record->signature = pending->signature;
    record->name = pending->name;
    memset(pending, 0, sizeof(*pending));
}

static int cuda_profile_signature_equal(const CudaLaunchSignature* a,
                                        const CudaLaunchSignature* b) {
    return a && b && a->function == b->function &&
        a->grid_x == b->grid_x && a->grid_y == b->grid_y &&
        a->grid_z == b->grid_z && a->block_x == b->block_x &&
        a->block_y == b->block_y && a->block_z == b->block_z &&
        a->shared_bytes == b->shared_bytes;
}

static int cuda_profile_aggregate_compare(const void* left,
                                          const void* right) {
    const CudaProfileAggregate* a = (const CudaProfileAggregate*)left;
    const CudaProfileAggregate* b = (const CudaProfileAggregate*)right;
    int name_order;
    if (a->total_ms > b->total_ms) return -1;
    if (a->total_ms < b->total_ms) return 1;
    name_order = strcmp(a->name, b->name);
    if (name_order) return name_order;
#define CUDA_PROFILE_COMPARE_FIELD(field) \
    do { \
        if (a->signature.field < b->signature.field) return -1; \
        if (a->signature.field > b->signature.field) return 1; \
    } while (0)
    CUDA_PROFILE_COMPARE_FIELD(grid_x);
    CUDA_PROFILE_COMPARE_FIELD(grid_y);
    CUDA_PROFILE_COMPARE_FIELD(grid_z);
    CUDA_PROFILE_COMPARE_FIELD(block_x);
    CUDA_PROFILE_COMPARE_FIELD(block_y);
    CUDA_PROFILE_COMPARE_FIELD(block_z);
    CUDA_PROFILE_COMPARE_FIELD(shared_bytes);
#undef CUDA_PROFILE_COMPARE_FIELD
    return 0;
}

/* snprintf follows LC_NUMERIC, while the profile format always uses commas as
 * field separators. Normalize the locale's (possibly multibyte) decimal point
 * locally instead of mutating the process-wide locale. */
static int cuda_profile_format_decimal(char* buffer, size_t capacity,
                                       double value) {
    const struct lconv* numeric;
    const char* decimal_point;
    char* point;
    size_t decimal_length;
    int length;
    if (!buffer || !capacity || !isfinite(value)) return 0;
    length = snprintf(buffer, capacity, "%.6f", value);
    if (length < 0 || (size_t)length >= capacity) return 0;
    numeric = localeconv();
    decimal_point = numeric ? numeric->decimal_point : NULL;
    if (!decimal_point || !decimal_point[0] || !strcmp(decimal_point, "."))
        return 1;
    point = strstr(buffer, decimal_point);
    if (!point) return 0;
    decimal_length = strlen(decimal_point);
    if (decimal_length != 1u)
        memmove(point + 1u, point + decimal_length,
                strlen(point + decimal_length) + 1u);
    point[0] = '.';
    return 1;
}

static int cuda_profile_emit_aggregate(const char* kind, int complete,
                                       const CudaProfileAggregate* value) {
    double mean;
    char total_text[64];
    char mean_text[64];
    char max_text[64];
    int formatted = 1;
    int written;
    if (!cuda_profile.output || !kind || !value) return 0;
    mean = value->count ? value->total_ms / (double)value->count : 0.0;
    if (!cuda_profile_format_decimal(
            total_text, sizeof(total_text), value->total_ms) ||
        !cuda_profile_format_decimal(mean_text, sizeof(mean_text), mean) ||
        !cuda_profile_format_decimal(
            max_text, sizeof(max_text), (double)value->max_ms)) {
        cuda_profile.incomplete = 1;
        complete = 0;
        formatted = 0;
        memcpy(total_text, "0.000000", sizeof("0.000000"));
        memcpy(mean_text, "0.000000", sizeof("0.000000"));
        memcpy(max_text, "0.000000", sizeof("0.000000"));
    }
    written = fprintf(cuda_profile.output,
        "%s,%llu,%d,%s,%u,%u,%u,%u,%u,%u,%u,%llu,%s,%s,%s\n",
        kind, (unsigned long long)cuda_profile.scope, complete,
        value->name, value->signature.grid_x, value->signature.grid_y,
        value->signature.grid_z, value->signature.block_x,
        value->signature.block_y, value->signature.block_z,
        value->signature.shared_bytes, (unsigned long long)value->count,
        total_text, mean_text, max_text);
    if (written < 0 || ferror(cuda_profile.output)) {
        cuda_profile.incomplete = 1;
        return 0;
    }
    return formatted;
}

static void cuda_profile_report_training_scope(int synchronized,
                                               int schedule_succeeded) {
    CudaProfileAggregate* entries = NULL;
    CudaProfileAggregate* signatures = NULL;
    size_t entry_count = 0;
    size_t signature_count = 0;
    uint64_t measured_count = 0;
    double total_ms = 0.0;
    float max_ms = 0.0f;
    int complete;
    int write_ok = 1;
    if (!cuda_profile.enabled || !cuda_profile.scope_open) return;
    if (cuda_profile_scope_serial != UINT64_MAX)
        cuda_profile_scope_serial++;
    else
        cuda_profile.incomplete = 1;
    cuda_profile.scope = cuda_profile_scope_serial;
    if (!synchronized) cuda_profile.incomplete = 1;
    if (cuda_profile.record_count) {
        entries = (CudaProfileAggregate*)calloc(
            cuda_profile.record_count, sizeof(*entries));
        signatures = (CudaProfileAggregate*)calloc(
            cuda_profile.record_count, sizeof(*signatures));
        if (!entries || !signatures) cuda_profile.incomplete = 1;
    }
    for (size_t index = 0; index < cuda_profile.record_count; index++) {
        CudaProfileRecord* record = &cuda_profile.records[index];
        float milliseconds = 0.0f;
        if (!synchronized || !entries || !signatures ||
            p_cuEventElapsedTime(&milliseconds, record->start, record->end) !=
                CUDA_SUCCESS || !isfinite(milliseconds) || milliseconds < 0.0f) {
            cuda_profile.incomplete = 1;
        } else {
            size_t entry_index;
            size_t signature_index;
            for (entry_index = 0; entry_index < entry_count; entry_index++) {
                if (strcmp(entries[entry_index].name, record->name) == 0)
                    break;
            }
            if (entry_index == entry_count) {
                entries[entry_count].name = record->name;
                entry_count++;
            }
            entries[entry_index].count++;
            entries[entry_index].total_ms += milliseconds;
            if (milliseconds > entries[entry_index].max_ms)
                entries[entry_index].max_ms = milliseconds;
            for (signature_index = 0; signature_index < signature_count;
                 signature_index++) {
                if (strcmp(signatures[signature_index].name,
                           record->name) == 0 &&
                    cuda_profile_signature_equal(
                        &signatures[signature_index].signature,
                        &record->signature)) break;
            }
            if (signature_index == signature_count) {
                signatures[signature_count].name = record->name;
                signatures[signature_count].signature = record->signature;
                signature_count++;
            }
            signatures[signature_index].count++;
            signatures[signature_index].total_ms += milliseconds;
            if (milliseconds > signatures[signature_index].max_ms)
                signatures[signature_index].max_ms = milliseconds;
            measured_count++;
            total_ms += milliseconds;
            if (milliseconds > max_ms) max_ms = milliseconds;
        }
        cuda_profile_destroy_event(record->start);
        cuda_profile_destroy_event(record->end);
    }
    cuda_profile.record_count = 0;
    complete = synchronized && schedule_succeeded && !cuda_profile.incomplete &&
        cuda_profile.loaded_model_trainstep && !cuda_profile.non_cuda_route &&
        measured_count > 0;
    if (cuda_profile.output) {
        CudaProfileAggregate all = {
            "all", {0}, measured_count, total_ms, max_ms,
        };
        if (!cuda_profile.header_written) {
            if (fputs(
                    "record,scope,complete,entry,grid_x,grid_y,grid_z,block_x,"
                    "block_y,block_z,shared_bytes,count,total_ms,mean_ms,max_ms\n",
                    cuda_profile.output) == EOF || ferror(cuda_profile.output)) {
                cuda_profile.incomplete = 1;
                write_ok = 0;
            } else {
                cuda_profile.header_written = 1;
            }
        }
        if (entry_count > 1u)
            qsort(entries, entry_count, sizeof(*entries),
                  cuda_profile_aggregate_compare);
        if (signature_count > 1u)
            qsort(signatures, signature_count, sizeof(*signatures),
                  cuda_profile_aggregate_compare);
        for (size_t index = 0; write_ok && index < entry_count; index++)
            write_ok = cuda_profile_emit_aggregate(
                "entry", complete, &entries[index]);
        for (size_t index = 0; write_ok && index < signature_count; index++)
            write_ok = cuda_profile_emit_aggregate(
                "signature", complete, &signatures[index]);
        if (write_ok && fflush(cuda_profile.output) != 0) {
            cuda_profile.incomplete = 1;
            write_ok = 0;
        }
        complete = complete && write_ok && !cuda_profile.incomplete;
        if (!write_ok) clearerr(cuda_profile.output);
        if (!cuda_profile_emit_aggregate("scope", complete, &all) ||
            fflush(cuda_profile.output) != 0) {
            cuda_profile.incomplete = 1;
            if (cuda_profile.output != stderr)
                fprintf(stderr,
                    "[CUDA] profile output failed; scope evidence is incomplete\n");
        }
    }
    free(entries);
    free(signatures);
    cuda_profile.scope_open = 0;
    cuda_profile.loaded_model_trainstep = 0;
    cuda_profile.non_cuda_route = 0;
    cuda_profile.incomplete = !cuda_profile.api_available;
}
#endif
