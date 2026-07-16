#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stddef.h>

typedef enum {
    SAFETENSORS_DTYPE_UNKNOWN = 0,
    SAFETENSORS_DTYPE_BOOL = 1,
    SAFETENSORS_DTYPE_F4 = 2,
    SAFETENSORS_DTYPE_F6_E2M3 = 3,
    SAFETENSORS_DTYPE_F6_E3M2 = 4,
    SAFETENSORS_DTYPE_U8 = 5,
    SAFETENSORS_DTYPE_I8 = 6,
    SAFETENSORS_DTYPE_F8_E5M2 = 7,
    SAFETENSORS_DTYPE_F8_E4M3 = 8,
    SAFETENSORS_DTYPE_F8_E8M0 = 9,
    SAFETENSORS_DTYPE_F8_E4M3FNUZ = 10,
    SAFETENSORS_DTYPE_F8_E5M2FNUZ = 11,
    SAFETENSORS_DTYPE_I16 = 12,
    SAFETENSORS_DTYPE_U16 = 13,
    SAFETENSORS_DTYPE_F16 = 14,
    SAFETENSORS_DTYPE_BF16 = 15,
    SAFETENSORS_DTYPE_I32 = 16,
    SAFETENSORS_DTYPE_U32 = 17,
    SAFETENSORS_DTYPE_F32 = 18,
    SAFETENSORS_DTYPE_C64 = 19,
    SAFETENSORS_DTYPE_F64 = 20,
    SAFETENSORS_DTYPE_I64 = 21,
    SAFETENSORS_DTYPE_U64 = 22
} SafetensorsDType;

enum {
    SAFETENSORS_TENSOR_READABLE = 1u << 0,
    SAFETENSORS_TENSOR_WRITABLE = 1u << 1,
    SAFETENSORS_TENSOR_OWNED = 1u << 2
};

enum {
    SAFETENSORS_OPEN_READ_ONLY = 0u,
    SAFETENSORS_OPEN_READ_WRITE = 1u << 0
};

typedef struct {
    unsigned flags;
} SafetensorsLoadOptions;

typedef struct {
    char name[128];
    SafetensorsDType dtype;
    int shape[8];
    int ndim;
    long data_start;
    long data_end;
    size_t nbytes;
    void* data;
    unsigned flags;
} SafetensorsTensor;

typedef struct {
    char* blob;
    long size;
    long data_base;
    char* metadata_json;
    int has_metadata;
    SafetensorsTensor* tensors;
    int tensor_count;
    unsigned flags;
} SafetensorsFile;

int safetensors_init_empty(SafetensorsFile* out, unsigned flags);
int safetensors_load(const char* file_path, SafetensorsFile* out);
int safetensors_load_with_options(const char* file_path, const SafetensorsLoadOptions* options, SafetensorsFile* out);
int safetensors_save(const char* file_path, SafetensorsFile* file);
void safetensors_free(SafetensorsFile* file);
const SafetensorsTensor* safetensors_find_tensor(const SafetensorsFile* file, const char* name);
SafetensorsTensor* safetensors_find_tensor_mutable(SafetensorsFile* file, const char* name);
void* safetensors_tensor_mutable_data(SafetensorsTensor* tensor);
int safetensors_set_metadata_json(SafetensorsFile* file, const char* metadata_json);
int safetensors_add_tensor(SafetensorsFile* file, const char* name, SafetensorsDType dtype,
                           const int* shape, int ndim, const void* data, size_t nbytes);
int safetensors_remove_tensor(SafetensorsFile* file, const char* name);
int safetensors_set_tensor_data(SafetensorsFile* file, const char* name, const void* data, size_t nbytes);
SafetensorsDType safetensors_dtype_from_name(const char* dtype);
const char* safetensors_dtype_name(SafetensorsDType dtype);
size_t safetensors_dtype_bit_width(SafetensorsDType dtype);
size_t safetensors_dtype_byte_width(SafetensorsDType dtype);
int safetensors_tensor_nbytes(SafetensorsDType dtype, const int* shape, int ndim, size_t* out_nbytes);

#endif
