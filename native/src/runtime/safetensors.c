#include "safetensors.h"
#include "cJSON.h"
#include "json_validation.h"
#include <limits.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <io.h>
#include <process.h>
#define VX_PROCESS_ID() ((long)_getpid())
#define VX_SYNC_STREAM(stream) (_commit(_fileno(stream)))
#define VX_HAVE_STREAM_SYNC 1
#elif defined(__unix__) || defined(__APPLE__) || defined(__ANDROID__)
#include <fcntl.h>
#include <unistd.h>
#define VX_PROCESS_ID() ((long)getpid())
#define VX_SYNC_STREAM(stream) (fsync(fileno(stream)))
#define VX_HAVE_STREAM_SYNC 1
#define VX_HAVE_DIRECTORY_SYNC 1
#else
/* Freestanding targets still get collision-free in-process temp names and atomic rename. */
#define VX_PROCESS_ID() 0L
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define SAFETENSORS_MAX_HEADER_SIZE 100000000L
static atomic_ulong g_safetensors_temp_counter = 1;

static int metadata_values_are_strings(const cJSON* metadata) {
    if (!cJSON_IsObject(metadata)) return 0;
    for (const cJSON* item = metadata->child; item; item = item->next) {
        if (!item->string || !cJSON_IsString(item) || !item->valuestring) return 0;
    }
    return 1;
}

static uint64_t read_u64_le(const unsigned char* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

static void write_u64_le(unsigned char* p, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        p[i] = (unsigned char)(v & 0xffu);
        v >>= 8;
    }
}

static char* read_file_bytes(const char* path, long* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open safetensors file: %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < 0) {
        fclose(f);
        return NULL;
    }
    char* data = (char*)malloc((size_t)size + 1);
    if (!data) {
        fclose(f);
        return NULL;
    }
    if (size > 0 && fread(data, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "Failed to read safetensors file: %s\n", path);
        free(data);
        fclose(f);
        return NULL;
    }
    data[size] = 0;
    fclose(f);
    if (out_size) *out_size = size;
    return data;
}

VxDataType safetensors_dtype_from_name(const char* dtype) {
    if (!dtype) return SAFETENSORS_DTYPE_UNKNOWN;
    if (strcmp(dtype, "BOOL") == 0) return SAFETENSORS_DTYPE_BOOL;
    if (strcmp(dtype, "F4") == 0) return SAFETENSORS_DTYPE_F4;
    if (strcmp(dtype, "F6_E2M3") == 0) return SAFETENSORS_DTYPE_F6_E2M3;
    if (strcmp(dtype, "F6_E3M2") == 0) return SAFETENSORS_DTYPE_F6_E3M2;
    if (strcmp(dtype, "U8") == 0) return SAFETENSORS_DTYPE_U8;
    if (strcmp(dtype, "I8") == 0) return SAFETENSORS_DTYPE_I8;
    if (strcmp(dtype, "F8_E5M2") == 0) return SAFETENSORS_DTYPE_F8_E5M2;
    if (strcmp(dtype, "F8_E4M3") == 0) return SAFETENSORS_DTYPE_F8_E4M3;
    if (strcmp(dtype, "F8_E8M0") == 0) return SAFETENSORS_DTYPE_F8_E8M0;
    if (strcmp(dtype, "F8_E4M3FNUZ") == 0) return SAFETENSORS_DTYPE_F8_E4M3FNUZ;
    if (strcmp(dtype, "F8_E5M2FNUZ") == 0) return SAFETENSORS_DTYPE_F8_E5M2FNUZ;
    if (strcmp(dtype, "I16") == 0) return SAFETENSORS_DTYPE_I16;
    if (strcmp(dtype, "U16") == 0) return SAFETENSORS_DTYPE_U16;
    if (strcmp(dtype, "F16") == 0) return SAFETENSORS_DTYPE_F16;
    if (strcmp(dtype, "BF16") == 0) return SAFETENSORS_DTYPE_BF16;
    if (strcmp(dtype, "I32") == 0) return SAFETENSORS_DTYPE_I32;
    if (strcmp(dtype, "U32") == 0) return SAFETENSORS_DTYPE_U32;
    if (strcmp(dtype, "F32") == 0) return SAFETENSORS_DTYPE_F32;
    if (strcmp(dtype, "C64") == 0) return SAFETENSORS_DTYPE_C64;
    if (strcmp(dtype, "F64") == 0) return SAFETENSORS_DTYPE_F64;
    if (strcmp(dtype, "I64") == 0) return SAFETENSORS_DTYPE_I64;
    if (strcmp(dtype, "U64") == 0) return SAFETENSORS_DTYPE_U64;
    return SAFETENSORS_DTYPE_UNKNOWN;
}

const char* safetensors_dtype_name(VxDataType dtype) {
    switch (dtype) {
        case SAFETENSORS_DTYPE_BOOL: return "BOOL";
        case SAFETENSORS_DTYPE_F4: return "F4";
        case SAFETENSORS_DTYPE_F6_E2M3: return "F6_E2M3";
        case SAFETENSORS_DTYPE_F6_E3M2: return "F6_E3M2";
        case SAFETENSORS_DTYPE_U8: return "U8";
        case SAFETENSORS_DTYPE_I8: return "I8";
        case SAFETENSORS_DTYPE_F8_E5M2: return "F8_E5M2";
        case SAFETENSORS_DTYPE_F8_E4M3: return "F8_E4M3";
        case SAFETENSORS_DTYPE_F8_E8M0: return "F8_E8M0";
        case SAFETENSORS_DTYPE_F8_E4M3FNUZ: return "F8_E4M3FNUZ";
        case SAFETENSORS_DTYPE_F8_E5M2FNUZ: return "F8_E5M2FNUZ";
        case SAFETENSORS_DTYPE_I16: return "I16";
        case SAFETENSORS_DTYPE_U16: return "U16";
        case SAFETENSORS_DTYPE_F16: return "F16";
        case SAFETENSORS_DTYPE_BF16: return "BF16";
        case SAFETENSORS_DTYPE_I32: return "I32";
        case SAFETENSORS_DTYPE_U32: return "U32";
        case SAFETENSORS_DTYPE_F32: return "F32";
        case SAFETENSORS_DTYPE_C64: return "C64";
        case SAFETENSORS_DTYPE_F64: return "F64";
        case SAFETENSORS_DTYPE_I64: return "I64";
        case SAFETENSORS_DTYPE_U64: return "U64";
        case SAFETENSORS_DTYPE_UNKNOWN:
        default: return "UNKNOWN";
    }
}

size_t safetensors_dtype_bit_width(VxDataType dtype) {
    switch (dtype) {
        case SAFETENSORS_DTYPE_F4:
            return 4;
        case SAFETENSORS_DTYPE_F6_E2M3:
        case SAFETENSORS_DTYPE_F6_E3M2:
            return 6;
        case SAFETENSORS_DTYPE_BOOL:
        case SAFETENSORS_DTYPE_U8:
        case SAFETENSORS_DTYPE_I8:
        case SAFETENSORS_DTYPE_F8_E5M2:
        case SAFETENSORS_DTYPE_F8_E4M3:
        case SAFETENSORS_DTYPE_F8_E8M0:
        case SAFETENSORS_DTYPE_F8_E4M3FNUZ:
        case SAFETENSORS_DTYPE_F8_E5M2FNUZ:
            return 8;
        case SAFETENSORS_DTYPE_I16:
        case SAFETENSORS_DTYPE_U16:
        case SAFETENSORS_DTYPE_F16:
        case SAFETENSORS_DTYPE_BF16:
            return 16;
        case SAFETENSORS_DTYPE_I32:
        case SAFETENSORS_DTYPE_U32:
        case SAFETENSORS_DTYPE_F32:
            return 32;
        case SAFETENSORS_DTYPE_C64:
        case SAFETENSORS_DTYPE_F64:
        case SAFETENSORS_DTYPE_I64:
        case SAFETENSORS_DTYPE_U64:
            return 64;
        case SAFETENSORS_DTYPE_UNKNOWN:
        default:
            return 0;
    }
}

size_t safetensors_dtype_byte_width(VxDataType dtype) {
    size_t bits = safetensors_dtype_bit_width(dtype);
    return bits && bits % 8 == 0 ? bits / 8 : 0;
}

void safetensors_free(SafetensorsFile* file) {
    if (!file) return;
    if (file->tensors) {
        for (int i = 0; i < file->tensor_count; i++) {
            if ((file->tensors[i].flags & SAFETENSORS_TENSOR_OWNED) && file->tensors[i].data) {
                free(file->tensors[i].data);
            }
        }
    }
    free(file->blob);
    free(file->metadata_json);
    free(file->tensors);
    memset(file, 0, sizeof(*file));
}

static int tensor_nbytes(VxDataType dtype, const int* shape, int ndim, size_t* out_nbytes) {
    size_t bits = safetensors_dtype_bit_width(dtype);
    if (bits == 0) return -1;
    size_t elems = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 0 || (elems != 0 && (size_t)shape[i] > SIZE_MAX / elems)) return -1;
        elems *= (size_t)shape[i];
    }
    if (elems != 0 && bits > SIZE_MAX / elems) return -1;
    size_t total_bits = elems * bits;
    if (total_bits % 8 != 0) return -1;
    *out_nbytes = total_bits / 8;
    return 0;
}

int safetensors_tensor_nbytes(VxDataType dtype, const int* shape, int ndim, size_t* out_nbytes) {
    return tensor_nbytes(dtype, shape, ndim, out_nbytes);
}

static int parse_tensor_entry(cJSON* node, long data_base, long file_size, SafetensorsTensor* out) {
    cJSON* dtype_node = cJSON_GetObjectItem(node, "dtype");
    cJSON* shape_node = cJSON_GetObjectItem(node, "shape");
    cJSON* offsets_node = cJSON_GetObjectItem(node, "data_offsets");
    if (!node->string || !cJSON_IsString(dtype_node) || !cJSON_IsArray(shape_node) || !cJSON_IsArray(offsets_node)) {
        return -1;
    }

    VxDataType dtype = safetensors_dtype_from_name(dtype_node->valuestring);
    if (dtype == SAFETENSORS_DTYPE_UNKNOWN) return -1;

    int ndim = cJSON_GetArraySize(shape_node);
    if (ndim < 0 || ndim > 8 || cJSON_GetArraySize(offsets_node) < 2) return -1;

    memset(out, 0, sizeof(*out));
    strncpy(out->name, node->string, sizeof(out->name) - 1);
    out->dtype = dtype;
    out->ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        cJSON* dim = cJSON_GetArrayItem(shape_node, i);
        if (!cJSON_IsNumber(dim) || dim->valueint < 0) return -1;
        out->shape[i] = dim->valueint;
    }

    long start = (long)cJSON_GetArrayItem(offsets_node, 0)->valuedouble;
    long end = (long)cJSON_GetArrayItem(offsets_node, 1)->valuedouble;
    if (start < 0 || end < start || data_base + end > file_size) return -1;
    size_t expected_nbytes = 0;
    if (tensor_nbytes(dtype, out->shape, ndim, &expected_nbytes) != 0) return -1;
    if ((size_t)(end - start) != expected_nbytes) return -1;
    out->data_start = start;
    out->data_end = end;
    out->nbytes = (size_t)(end - start);
    return 0;
}

static int compare_tensor_offsets(const void* a, const void* b) {
    const SafetensorsTensor* ta = (const SafetensorsTensor*)a;
    const SafetensorsTensor* tb = (const SafetensorsTensor*)b;
    if (ta->data_start < tb->data_start) return -1;
    if (ta->data_start > tb->data_start) return 1;
    return strcmp(ta->name, tb->name);
}

static int compare_name_ptrs(const void* a, const void* b) {
    const char* const* left = (const char* const*)a;
    const char* const* right = (const char* const*)b;
    return strcmp(*left, *right);
}

int safetensors_load(const char* file_path, SafetensorsFile* out) {
    SafetensorsLoadOptions options = { SAFETENSORS_OPEN_READ_ONLY };
    return safetensors_load_with_options(file_path, &options, out);
}

int safetensors_init_empty(SafetensorsFile* out, unsigned flags) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    out->flags = flags | SAFETENSORS_OPEN_READ_WRITE;
    out->data_base = 8;
    return 0;
}

int safetensors_load_with_options(const char* file_path, const SafetensorsLoadOptions* options, SafetensorsFile* out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    unsigned flags = options ? options->flags : SAFETENSORS_OPEN_READ_ONLY;

    long file_size = 0;
    char* blob = read_file_bytes(file_path, &file_size);
    if (!blob) return -1;
    if (file_size < 8) {
        fprintf(stderr, "Invalid safetensors file: %s\n", file_path);
        free(blob);
        return -1;
    }

    uint64_t header_size_u64 = read_u64_le((const unsigned char*)blob);
    if (header_size_u64 > (uint64_t)(file_size - 8) ||
        header_size_u64 > (uint64_t)SAFETENSORS_MAX_HEADER_SIZE) {
        fprintf(stderr, "Invalid safetensors header size in %s\n", file_path);
        free(blob);
        return -1;
    }

    long header_size = (long)header_size_u64;
    long data_base = 8 + header_size;
    char* json = (char*)malloc((size_t)header_size + 1);
    if (!json) {
        free(blob);
        return -1;
    }
    memcpy(json, blob + 8, (size_t)header_size);
    json[header_size] = 0;

    cJSON* root = cJSON_Parse(json);
    free(json);
    if (!root || !cJSON_IsObject(root) ||
        !vx_json_object_keys_unique_recursive(root)) {
        fprintf(stderr, "Failed to parse safetensors JSON header: %s\n", file_path);
        cJSON_Delete(root);
        free(blob);
        return -1;
    }

    int count = 0;
    char* metadata_json = NULL;
    int has_metadata = 0;
    for (cJSON* it = root->child; it; it = it->next) {
        if (it->string && strcmp(it->string, "__metadata__") == 0) {
            if (!metadata_values_are_strings(it)) {
                fprintf(stderr, "Invalid safetensors __metadata__ in %s\n", file_path);
                cJSON_Delete(root);
                free(blob);
                return -1;
            }
            has_metadata = 1;
            metadata_json = cJSON_PrintUnformatted(it);
            if (!metadata_json) {
                cJSON_Delete(root);
                free(blob);
                return -1;
            }
            continue;
        }
        count++;
    }

    SafetensorsTensor* tensors = count > 0 ? (SafetensorsTensor*)calloc((size_t)count, sizeof(*tensors)) : NULL;
    if (count > 0 && !tensors) {
        cJSON_Delete(root);
        free(metadata_json);
        free(blob);
        return -1;
    }

    int idx = 0;
    for (cJSON* it = root->child; it; it = it->next) {
        if (it->string && strcmp(it->string, "__metadata__") == 0) continue;
        if (parse_tensor_entry(it, data_base, file_size, &tensors[idx]) != 0) {
            fprintf(stderr, "Invalid safetensors tensor metadata: %s\n", it->string ? it->string : "<unnamed>");
            free(tensors);
            cJSON_Delete(root);
            free(metadata_json);
            free(blob);
            return -1;
        }
        tensors[idx].data = blob + data_base + tensors[idx].data_start;
        tensors[idx].flags = SAFETENSORS_TENSOR_READABLE |
                             ((flags & SAFETENSORS_OPEN_READ_WRITE) ? SAFETENSORS_TENSOR_WRITABLE : 0u);
        idx++;
    }

    const char** names = count > 0 ? (const char**)malloc((size_t)count * sizeof(*names)) : NULL;
    if (count > 0 && !names) {
        free(tensors); cJSON_Delete(root); free(metadata_json); free(blob); return -1;
    }
    for (int i = 0; i < count; i++) names[i] = tensors[i].name;
    if (count > 1) qsort(names, (size_t)count, sizeof(*names), compare_name_ptrs);
    for (int i = 1; i < count; i++) {
        if (!strcmp(names[i - 1], names[i])) {
            fprintf(stderr, "Duplicate safetensors tensor name: %s\n", names[i]);
            free(names); free(tensors); cJSON_Delete(root); free(metadata_json); free(blob); return -1;
        }
    }
    free(names);
    if (count > 1) qsort(tensors, (size_t)count, sizeof(*tensors), compare_tensor_offsets);
    long cursor = 0;
    for (int i = 0; i < count; i++) {
        if (tensors[i].data_start != cursor) {
            fprintf(stderr, "Invalid safetensors tensor offsets near %s\n", tensors[i].name);
            free(tensors);
            cJSON_Delete(root);
            free(metadata_json);
            free(blob);
            return -1;
        }
        cursor = tensors[i].data_end;
    }
    if (data_base + cursor != file_size) {
        fprintf(stderr, "Safetensors metadata does not cover full file: %s\n", file_path);
        free(tensors);
        cJSON_Delete(root);
        free(metadata_json);
        free(blob);
        return -1;
    }

    cJSON_Delete(root);
    out->blob = blob;
    out->size = file_size;
    out->data_base = data_base;
    out->metadata_json = metadata_json;
    out->has_metadata = has_metadata;
    out->tensors = tensors;
    out->tensor_count = count;
    out->flags = flags;
    return 0;
}

static int write_all_file_atomic(const char* file_path, const void* data, size_t n) {
    char tmp_path[PATH_MAX];
    unsigned long sequence = atomic_fetch_add_explicit(&g_safetensors_temp_counter, 1, memory_order_relaxed);
    int path_len = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%ld.%lu", file_path, VX_PROCESS_ID(), sequence);
    if (path_len <= 0 || (size_t)path_len >= sizeof(tmp_path)) return -1;
    FILE* f = fopen(tmp_path, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open safetensors output file: %s\n", file_path);
        return -1;
    }
    if (n > 0 && fwrite(data, 1, n, f) != n) {
        fprintf(stderr, "Failed to write safetensors output file: %s\n", file_path);
        fclose(f);
        remove(tmp_path);
        return -1;
    }
    if (fflush(f) != 0) {
        fclose(f);
        remove(tmp_path);
        return -1;
    }
#if defined(VX_HAVE_STREAM_SYNC)
    if (VX_SYNC_STREAM(f) != 0) {
        fclose(f);
        remove(tmp_path);
        return -1;
    }
#endif
    if (fclose(f) != 0) {
        remove(tmp_path);
        return -1;
    }
    if (rename(tmp_path, file_path) != 0) {
        remove(tmp_path);
        return -1;
    }
#if defined(VX_HAVE_DIRECTORY_SYNC)
    char parent[PATH_MAX];
    size_t filepath_len = strlen(file_path);
    if (filepath_len < sizeof(parent)) {
        memcpy(parent, file_path, filepath_len + 1);
        char* slash = strrchr(parent, '/');
        if (slash) {
            if (slash == parent) slash[1] = 0;
            else *slash = 0;
        } else {
            strcpy(parent, ".");
        }
        int flags = O_RDONLY;
#ifdef O_DIRECTORY
        flags |= O_DIRECTORY;
#endif
        int dir_fd = open(parent, flags);
        if (dir_fd >= 0) {
            int sync_rc = fsync(dir_fd);
            close(dir_fd);
            if (sync_rc != 0) return -1;
        }
    }
#endif
    return 0;
}

static int build_serialized_blob(const SafetensorsFile* file, char** out_blob, long* out_size, long* out_data_base) {
    cJSON* root = cJSON_CreateObject();
    if (!root) return -1;
    if (file->has_metadata) {
        cJSON* md = file->metadata_json ? cJSON_Parse(file->metadata_json) : cJSON_CreateObject();
        if (!md || !vx_json_object_keys_unique_recursive(md) ||
            !metadata_values_are_strings(md)) {
            cJSON_Delete(md);
            cJSON_Delete(root);
            return -1;
        }
        if (!cJSON_AddItemToObject(root, "__metadata__", md)) {
            cJSON_Delete(md);
            cJSON_Delete(root);
            return -1;
        }
    }

    size_t data_bytes = 0;
    for (int i = 0; i < file->tensor_count; i++) {
        const SafetensorsTensor* t = &file->tensors[i];
        cJSON* entry = cJSON_CreateObject();
        cJSON* shape = cJSON_CreateArray();
        cJSON* offsets = cJSON_CreateArray();
        if (!entry || !shape || !offsets) {
            cJSON_Delete(entry);
            cJSON_Delete(shape);
            cJSON_Delete(offsets);
            cJSON_Delete(root);
            return -1;
        }
        cJSON_AddStringToObject(entry, "dtype", safetensors_dtype_name(t->dtype));
        for (int d = 0; d < t->ndim; d++) cJSON_AddItemToArray(shape, cJSON_CreateNumber(t->shape[d]));
        cJSON_AddItemToArray(offsets, cJSON_CreateNumber((double)data_bytes));
        cJSON_AddItemToArray(offsets, cJSON_CreateNumber((double)(data_bytes + t->nbytes)));
        cJSON_AddItemToObject(entry, "shape", shape);
        cJSON_AddItemToObject(entry, "data_offsets", offsets);
        cJSON_AddItemToObject(root, t->name, entry);
        data_bytes += t->nbytes;
    }

    char* json = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (!json) return -1;
    size_t header_len = strlen(json);
    size_t aligned_header_len = (header_len + 7u) & ~(size_t)7u;
    if (aligned_header_len > (size_t)SAFETENSORS_MAX_HEADER_SIZE ||
        data_bytes > SIZE_MAX - aligned_header_len - 8u) {
        free(json);
        return -1;
    }

    size_t total = 8u + aligned_header_len + data_bytes;
    char* blob = (char*)malloc(total);
    if (!blob) {
        free(json);
        return -1;
    }
    write_u64_le((unsigned char*)blob, (uint64_t)aligned_header_len);
    memcpy(blob + 8, json, header_len);
    memset(blob + 8 + header_len, ' ', aligned_header_len - header_len);
    free(json);

    size_t cursor = 0;
    for (int i = 0; i < file->tensor_count; i++) {
        const SafetensorsTensor* t = &file->tensors[i];
        if (t->nbytes > 0 && !t->data) {
            free(blob);
            return -1;
        }
        if (t->nbytes > 0) memcpy(blob + 8 + aligned_header_len + cursor, t->data, t->nbytes);
        cursor += t->nbytes;
    }
    *out_blob = blob;
    *out_size = (long)total;
    *out_data_base = (long)(8u + aligned_header_len);
    return 0;
}

int safetensors_save(const char* file_path, SafetensorsFile* file) {
    if (!file_path || !file) return -1;
    char* blob = NULL;
    long size = 0;
    long data_base = 0;
    if (build_serialized_blob(file, &blob, &size, &data_base) != 0) return -1;
    int rc = write_all_file_atomic(file_path, blob, (size_t)size);
    /* Saving must not invalidate tensor pointers or external engine aliases. */
    free(blob);
    return rc;
}

const SafetensorsTensor* safetensors_find_tensor(const SafetensorsFile* file, const char* name) {
    if (!file || !name) return NULL;
    for (int i = 0; i < file->tensor_count; i++) {
        if (strcmp(file->tensors[i].name, name) == 0) return &file->tensors[i];
    }
    return NULL;
}

SafetensorsTensor* safetensors_find_tensor_mutable(SafetensorsFile* file, const char* name) {
    if (!file || !name) return NULL;
    for (int i = 0; i < file->tensor_count; i++) {
        if (strcmp(file->tensors[i].name, name) == 0) return &file->tensors[i];
    }
    return NULL;
}

int safetensors_set_metadata_json(SafetensorsFile* file, const char* metadata_json) {
    if (!file || !(file->flags & SAFETENSORS_OPEN_READ_WRITE)) return -1;
    char* copy = NULL;
    if (metadata_json && metadata_json[0]) {
        cJSON* parsed = cJSON_Parse(metadata_json);
        if (!parsed || !vx_json_object_keys_unique_recursive(parsed) ||
            !metadata_values_are_strings(parsed)) {
            cJSON_Delete(parsed);
            return -1;
        }
        copy = cJSON_PrintUnformatted(parsed);
        cJSON_Delete(parsed);
        if (!copy) return -1;
    }
    free(file->metadata_json);
    file->metadata_json = copy;
    file->has_metadata = copy != NULL;
    return 0;
}

int safetensors_add_tensor(SafetensorsFile* file, const char* name, VxDataType dtype,
                           const int* shape, int ndim, const void* data, size_t nbytes) {
    if (!file || !name || !name[0] || !shape || ndim < 0 || ndim > 8 ||
        !(file->flags & SAFETENSORS_OPEN_READ_WRITE)) return -1;
    if (safetensors_find_tensor(file, name)) return -1;
    size_t expected = 0;
    if (tensor_nbytes(dtype, shape, ndim, &expected) != 0) return -1;
    if (!data && nbytes == 0) nbytes = expected;
    if (nbytes != expected) return -1;

    SafetensorsTensor* next = (SafetensorsTensor*)realloc(
        file->tensors,
        (size_t)(file->tensor_count + 1) * sizeof(*file->tensors));
    if (!next) return -1;
    file->tensors = next;

    SafetensorsTensor* t = &file->tensors[file->tensor_count];
    memset(t, 0, sizeof(*t));
    strncpy(t->name, name, sizeof(t->name) - 1);
    t->dtype = dtype;
    t->ndim = ndim;
    for (int i = 0; i < ndim && i < 8; i++) t->shape[i] = shape[i];
    t->nbytes = expected;
    t->data_start = 0;
    t->data_end = (long)expected;
    t->flags = SAFETENSORS_TENSOR_READABLE | SAFETENSORS_TENSOR_WRITABLE | SAFETENSORS_TENSOR_OWNED;
    if (expected > 0) {
        t->data = data ? malloc(expected) : calloc(1, expected);
        if (!t->data) {
            memset(t, 0, sizeof(*t));
            return -1;
        }
        if (data) memcpy(t->data, data, expected);
    }
    file->tensor_count++;
    return 0;
}

int safetensors_remove_tensor(SafetensorsFile* file, const char* name) {
    if (!file || !name || !(file->flags & SAFETENSORS_OPEN_READ_WRITE)) return -1;
    for (int i = 0; i < file->tensor_count; i++) {
        SafetensorsTensor* t = &file->tensors[i];
        if (strcmp(t->name, name) != 0) continue;
        if ((t->flags & SAFETENSORS_TENSOR_OWNED) && t->data) free(t->data);
        if (i + 1 < file->tensor_count) {
            memmove(&file->tensors[i], &file->tensors[i + 1],
                    (size_t)(file->tensor_count - i - 1) * sizeof(*file->tensors));
        }
        file->tensor_count--;
        return 0;
    }
    return -1;
}

void* safetensors_tensor_mutable_data(SafetensorsTensor* tensor) {
    if (!tensor || !(tensor->flags & SAFETENSORS_TENSOR_WRITABLE)) return NULL;
    return tensor->data;
}

int safetensors_set_tensor_data(SafetensorsFile* file, const char* name, const void* data, size_t nbytes) {
    if (!file || !(file->flags & SAFETENSORS_OPEN_READ_WRITE) || !data) return -1;
    SafetensorsTensor* tensor = safetensors_find_tensor_mutable(file, name);
    if (!tensor || !(tensor->flags & SAFETENSORS_TENSOR_WRITABLE) || nbytes != tensor->nbytes) return -1;
    if (tensor->data != data && nbytes > 0) memcpy(tensor->data, data, nbytes);
    return 0;
}
