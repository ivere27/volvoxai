#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include "volvoxai_enums.h"

#include <stddef.h>

/* Safetensors and graph tensors share the protobuf-generated dtype contract.
 * Keep the existing private spellings as source-compatible aliases only. */
enum {
    SAFETENSORS_DTYPE_UNSPECIFIED = VX_DTYPE_UNSPECIFIED,
    SAFETENSORS_DTYPE_UNKNOWN = SAFETENSORS_DTYPE_UNSPECIFIED,
    SAFETENSORS_DTYPE_BOOL = VX_DTYPE_BOOL,
    SAFETENSORS_DTYPE_F4 = VX_DTYPE_F4,
    SAFETENSORS_DTYPE_F6_E2M3 = VX_DTYPE_F6_E2M3,
    SAFETENSORS_DTYPE_F6_E3M2 = VX_DTYPE_F6_E3M2,
    SAFETENSORS_DTYPE_U8 = VX_DTYPE_U8,
    SAFETENSORS_DTYPE_I8 = VX_DTYPE_I8,
    SAFETENSORS_DTYPE_F8_E5M2 = VX_DTYPE_F8_E5M2,
    SAFETENSORS_DTYPE_F8_E4M3 = VX_DTYPE_F8_E4M3,
    SAFETENSORS_DTYPE_F8_E8M0 = VX_DTYPE_F8_E8M0,
    SAFETENSORS_DTYPE_F8_E4M3FNUZ = VX_DTYPE_F8_E4M3FNUZ,
    SAFETENSORS_DTYPE_F8_E5M2FNUZ = VX_DTYPE_F8_E5M2FNUZ,
    SAFETENSORS_DTYPE_I16 = VX_DTYPE_I16,
    SAFETENSORS_DTYPE_U16 = VX_DTYPE_U16,
    SAFETENSORS_DTYPE_F16 = VX_DTYPE_F16,
    SAFETENSORS_DTYPE_BF16 = VX_DTYPE_BF16,
    SAFETENSORS_DTYPE_I32 = VX_DTYPE_I32,
    SAFETENSORS_DTYPE_U32 = VX_DTYPE_U32,
    SAFETENSORS_DTYPE_F32 = VX_DTYPE_F32,
    SAFETENSORS_DTYPE_C64 = VX_DTYPE_C64,
    SAFETENSORS_DTYPE_F64 = VX_DTYPE_F64,
    SAFETENSORS_DTYPE_I64 = VX_DTYPE_I64,
    SAFETENSORS_DTYPE_U64 = VX_DTYPE_U64
};

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
    VxDataType dtype;
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
int safetensors_add_tensor(SafetensorsFile* file, const char* name, VxDataType dtype,
                           const int* shape, int ndim, const void* data, size_t nbytes);
int safetensors_remove_tensor(SafetensorsFile* file, const char* name);
int safetensors_set_tensor_data(SafetensorsFile* file, const char* name, const void* data, size_t nbytes);
VxDataType safetensors_dtype_from_name(const char* dtype);
const char* safetensors_dtype_name(VxDataType dtype);
size_t safetensors_dtype_bit_width(VxDataType dtype);
size_t safetensors_dtype_byte_width(VxDataType dtype);
int safetensors_tensor_nbytes(VxDataType dtype, const int* shape, int ndim, size_t* out_nbytes);

#endif
