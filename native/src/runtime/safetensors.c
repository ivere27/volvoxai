#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "safetensors.h"
#include "vx_platform.h"
#include "cJSON.h"
#include "json_validation.h"
#include <limits.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <process.h>
#define VX_PROCESS_ID() ((long)_getpid())
#define VX_SYNC_STREAM(stream) (_commit(_fileno(stream)))
#define VX_HAVE_STREAM_SYNC 1
#elif defined(__unix__) || defined(__APPLE__) || defined(__ANDROID__)
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
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

#define SAFETENSORS_MAX_HEADER_SIZE INT64_C(100000000)
#define SAFETENSORS_MAX_EXACT_JSON_INTEGER INT64_C(9007199254740991)
static atomic_ulong g_safetensors_temp_counter = 1;

static int metadata_values_are_strings(const cJSON* metadata) {
    if (!cJSON_IsObject(metadata)) return 0;
    for (const cJSON* item = metadata->child; item; item = item->next) {
        if (!item->string || !cJSON_IsString(item) || !item->valuestring) return 0;
    }
    return 1;
}

static void write_u64_le(unsigned char* p, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        p[i] = (unsigned char)(v & 0xffu);
        v >>= 8;
    }
}

/* Native targets resolve ordinary paths. The freestanding WASM composition
 * resolves the same path API through wasm_libc's module-local VFS. */
static int stream_size(FILE* stream, int64_t* out_size) {
    if (!stream || !out_size) return -1;
#if defined(_WIN32)
    if (_fseeki64(stream, 0, SEEK_END) != 0) return -1;
    __int64 size = _ftelli64(stream);
    if (size < 0 || _fseeki64(stream, 0, SEEK_SET) != 0) return -1;
    *out_size = (int64_t)size;
#elif defined(__unix__) || defined(__APPLE__) || defined(__ANDROID__)
    if (fseeko(stream, 0, SEEK_END) != 0) return -1;
    off_t size = ftello(stream);
    if (size < 0 || (uintmax_t)size > (uintmax_t)INT64_MAX ||
        fseeko(stream, 0, SEEK_SET) != 0) return -1;
    *out_size = (int64_t)size;
#else
    if (fseek(stream, 0, SEEK_END) != 0) return -1;
    long size = ftell(stream);
    if (size < 0 || fseek(stream, 0, SEEK_SET) != 0) return -1;
    *out_size = (int64_t)size;
#endif
    return 0;
}

static char* read_file_bytes(const char* path, int64_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open safetensors file: %s\n", path);
        return NULL;
    }
    int64_t size = 0;
    if (stream_size(f, &size) != 0 || (uint64_t)size >= (uint64_t)SIZE_MAX) {
        fclose(f);
        return NULL;
    }
    size_t addressable_size = (size_t)size;
    char* data = (char*)malloc(addressable_size + 1u);
    if (!data) {
        fclose(f);
        return NULL;
    }
    size_t cursor = 0;
    while (cursor < addressable_size) {
        size_t amount = fread(data + cursor, 1, addressable_size - cursor, f);
        if (amount == 0) {
            fprintf(stderr, "Failed to read safetensors file: %s\n", path);
            free(data);
            fclose(f);
            return NULL;
        }
        cursor += amount;
    }
    data[addressable_size] = 0;
    fclose(f);
    if (out_size) *out_size = size;
    return data;
}

#if defined(__unix__) || defined(__APPLE__) || defined(__ANDROID__)
static char* mmap_file_bytes(const char* path, int64_t* out_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
        close(fd);
        return NULL;
    }
    if (st.st_size == 0) {
        close(fd);
        return NULL;
    }
    if ((uintmax_t)st.st_size > (uintmax_t)INT64_MAX ||
        (uintmax_t)st.st_size > (uintmax_t)SIZE_MAX) {
        close(fd);
        return NULL;
    }
    int64_t size = (int64_t)st.st_size;
    void* mapped = mmap(NULL, (size_t)size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) return NULL;
    if (out_size) *out_size = size;
    return (char*)mapped;
}

static void unmap_file_bytes(void* addr, int64_t size) {
    if (addr && size > 0) {
        int unmapped = munmap(addr, (size_t)size) == 0;
        (void)unmapped;
    }
}
#elif defined(_WIN32)
static char* mmap_file_bytes(const char* path, int64_t* out_size) {
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) return NULL;
    LARGE_INTEGER size_li;
    if (!GetFileSizeEx(file, &size_li) || size_li.QuadPart <= 0 ||
        (uint64_t)size_li.QuadPart > (uint64_t)SIZE_MAX) {
        CloseHandle(file);
        return NULL;
    }
    int64_t size = (int64_t)size_li.QuadPart;
    HANDLE mapping = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(file);
    if (!mapping) return NULL;
    void* mapped = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping);
    if (!mapped) return NULL;
    if (out_size) *out_size = size;
    return (char*)mapped;
}

static void unmap_file_bytes(void* addr, int64_t size) {
    (void)size;
    if (addr) {
        int unmapped = UnmapViewOfFile(addr) != 0;
        (void)unmapped;
    }
}
#elif defined(__wasm__)
static char* mmap_file_bytes(const char* path, int64_t* out_size) {
    size_t size = 0u;
    const unsigned char* bytes = vx_wasm_file_map(path, &size);
    if (!bytes || size > (size_t)INT64_MAX) {
        if (bytes) (void)vx_wasm_file_unmap(bytes);
        return NULL;
    }
    if (out_size) *out_size = (int64_t)size;
    return (char*)bytes;
}

static void unmap_file_bytes(void* addr, int64_t size) {
    (void)size;
    if (addr) (void)vx_wasm_file_unmap(addr);
}
#else
static char* mmap_file_bytes(const char* path, int64_t* out_size) {
    (void)path; (void)out_size;
    return NULL;
}
static void unmap_file_bytes(void* addr, int64_t size) {
    (void)addr; (void)size;
}
#endif

static void free_blob_storage(char* blob, int64_t size, int is_mmap) {
    if (!blob) return;
    if (is_mmap) {
        unmap_file_bytes(blob, size);
        return;
    }
    free(blob);
}


void safetensors_free(SafetensorsFile* file) {
    if (!file) return;
    if (!file->borrows_storage && file->tensors) {
        for (int i = 0; i < file->tensor_count; i++) {
            if ((file->tensors[i].flags & SAFETENSORS_TENSOR_OWNED) && file->tensors[i].data) {
                free(file->tensors[i].data);
            }
        }
    }
    if (!file->borrows_storage) {
        free_blob_storage(file->blob, file->size, file->is_mmap);
        free(file->metadata_json);
    }
    free(file->tensors);
    memset(file, 0, sizeof(*file));
}

int safetensors_load(const char* file_path, SafetensorsFile* out) {
    SafetensorsLoadOptions options = { SAFETENSORS_OPEN_READ_ONLY };
    return safetensors_load_with_options(file_path, &options, out);
}

int safetensors_borrow_immutable(const SafetensorsFile* source,
                                 SafetensorsFile* out) {
    if (!source || !out || source == out || source->borrows_storage ||
        (source->flags & SAFETENSORS_OPEN_READ_WRITE) ||
        source->tensor_count < 0 ||
        (source->tensor_count > 0 && !source->tensors))
        return -1;
    for (int index = 0; index < source->tensor_count; index++) {
        if (source->tensors[index].flags &
            (SAFETENSORS_TENSOR_WRITABLE | SAFETENSORS_TENSOR_OWNED))
            return -1;
    }
    SafetensorsTensor* descriptors = NULL;
    if (source->tensor_count > 0) {
        descriptors = (SafetensorsTensor*)malloc(
            (size_t)source->tensor_count * sizeof(*descriptors));
        if (!descriptors) return -1;
        memcpy(descriptors, source->tensors,
               (size_t)source->tensor_count * sizeof(*descriptors));
        for (int index = 0; index < source->tensor_count; index++)
            descriptors[index].flags &=
                ~(SAFETENSORS_TENSOR_WRITABLE | SAFETENSORS_TENSOR_OWNED);
    }
    *out = *source;
    out->tensors = descriptors;
    out->borrows_storage = 1;
    return 0;
}

int safetensors_init_empty(SafetensorsFile* out, unsigned flags) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    out->flags = flags | SAFETENSORS_OPEN_READ_WRITE;
    out->data_base = 8;
    return 0;
}

int safetensors_load_with_options(const char* file_path, const SafetensorsLoadOptions* options, SafetensorsFile* out) {
    SafetensorsBorrowedBytes parsed = {0};
    unsigned flags;
    int64_t file_size = 0;
    int is_mmap = 0;
    char* blob = NULL;
    int index;
    if (!file_path || !out) return -1;
    memset(out, 0, sizeof(*out));
    flags = options ? options->flags : SAFETENSORS_OPEN_READ_ONLY;

    if (flags == SAFETENSORS_OPEN_READ_ONLY) {
        blob = mmap_file_bytes(file_path, &file_size);
        if (blob) is_mmap = 1;
    }
    if (!blob) {
        blob = read_file_bytes(file_path, &file_size);
        is_mmap = 0;
    }
    if (!blob) return -1;
    if (file_size < 0 ||
        safetensors_parse_borrowed_bytes(blob, (size_t)file_size,
                                         &parsed) != 0) {
        fprintf(stderr, "Invalid safetensors file: %s\n", file_path);
        free_blob_storage(blob, file_size, is_mmap);
        return -1;
    }
    for (index = 0; index < parsed.tensor_count; index++) {
        if (flags & SAFETENSORS_OPEN_READ_WRITE)
            parsed.tensors[index].flags |= SAFETENSORS_TENSOR_WRITABLE;
    }
    out->blob = blob;
    out->size = file_size;
    out->data_base = (int64_t)parsed.data_base;
    out->metadata_json = parsed.metadata_json;
    out->has_metadata = parsed.has_metadata;
    out->tensors = parsed.tensors;
    out->tensor_count = parsed.tensor_count;
    out->flags = flags;
    out->is_mmap = is_mmap;
    parsed.metadata_json = NULL;
    parsed.tensors = NULL;
    safetensors_borrowed_bytes_free(&parsed);
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

static int build_serialized_blob(const SafetensorsFile* file, char** out_blob,
                                 size_t* out_size, size_t* out_data_base) {
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
        if (t->nbytes > SIZE_MAX - data_bytes ||
            (uint64_t)data_bytes >
                (uint64_t)SAFETENSORS_MAX_EXACT_JSON_INTEGER ||
            (uint64_t)t->nbytes >
                (uint64_t)SAFETENSORS_MAX_EXACT_JSON_INTEGER -
                    (uint64_t)data_bytes) {
            cJSON_Delete(root);
            return -1;
        }
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
    *out_size = total;
    *out_data_base = 8u + aligned_header_len;
    return 0;
}

int safetensors_serialize(const SafetensorsFile* file, unsigned char** bytes,
                          size_t* size) {
    char* blob = NULL;
    size_t data_base = 0;
    if (!file || !bytes || !size) return -1;
    *bytes = NULL;
    *size = 0;
    if (build_serialized_blob(file, &blob, size, &data_base) != 0) return -1;
    *bytes = (unsigned char*)blob;
    return 0;
}

int safetensors_save(const char* file_path, SafetensorsFile* file) {
    if (!file_path || !file) return -1;
    char* blob = NULL;
    size_t size = 0;
    size_t data_base = 0;
    if (build_serialized_blob(file, &blob, &size, &data_base) != 0) return -1;
    int rc = write_all_file_atomic(file_path, blob, size);
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
    if (safetensors_tensor_nbytes(dtype, shape, ndim, &expected) != 0)
        return -1;
    if ((uint64_t)expected > (uint64_t)INT64_MAX) return -1;
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
    t->data_end = (int64_t)expected;
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
