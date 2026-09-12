#ifndef VOLVOXAI_RUNTIME_SAFETENSORS_HEADER_H
#define VOLVOXAI_RUNTIME_SAFETENSORS_HEADER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef VX_SAFETENSORS_HEADER_API
#define VX_SAFETENSORS_HEADER_API
#define VX_SAFETENSORS_HEADER_API_DEFINED_HERE 1
#endif

/*
 * Path-independent SafeTensors header authority.
 *
 * `header` contains only the eight-byte SafeTensors length prefix followed by
 * exactly that many UTF-8 JSON bytes. One leading UTF-8 BOM is accepted for
 * compatibility with the legacy browser TextDecoder path; repeated or
 * interior BOMs remain invalid JSON. Tensor payload bytes never enter this
 * API. `file_bytes` and `data_bytes` are logical lengths and must agree
 * exactly with the prefix and supplied header range.
 *
 * All ranges are caller-owned. The response is a fixed-width little-endian
 * wire record with relative offsets and no host pointers. The implementation
 * is reentrant and uses no allocator, cJSON, files, locks, locale, or mutable
 * global state. Scratch is used only for the object-member prepass records,
 * bounded duplicate-key sort records, and the tensor coverage index sort. It
 * is never retained and is cleared before return.
 */
#define VOLVOXAI_SAFETENSORS_HEADER_ABI_VERSION UINT32_C(1)
#define VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_MAGIC UINT32_C(0x31535856) /* VXS1 */
#define VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_BYTES UINT32_C(128)
#define VOLVOXAI_SAFETENSORS_HEADER_TENSOR_BYTES UINT32_C(56)
#define VOLVOXAI_SAFETENSORS_HEADER_METADATA_BYTES UINT32_C(24)
#define VOLVOXAI_SAFETENSORS_MAX_EXACT_INTEGER UINT64_C(9007199254740991)

typedef int32_t VxSafetensorsHeaderStatusV1;
enum {
    VX_SAFETENSORS_HEADER_STATUS_OK = 0,
    VX_SAFETENSORS_HEADER_STATUS_INVALID_ARGUMENT = -1,
    VX_SAFETENSORS_HEADER_STATUS_INVALID_HEADER = -2,
    VX_SAFETENSORS_HEADER_STATUS_RESPONSE_TOO_SMALL = -3,
    VX_SAFETENSORS_HEADER_STATUS_SCRATCH_TOO_SMALL = -4,
    VX_SAFETENSORS_HEADER_STATUS_INTERNAL = -5
};

typedef int32_t VxSafetensorsHeaderErrorCodeV1;
enum {
    VX_SAFETENSORS_HEADER_ERROR_NONE = 0,
    VX_SAFETENSORS_HEADER_ERROR_INVALID_PREFIX = 1,
    VX_SAFETENSORS_HEADER_ERROR_LENGTH_MISMATCH = 2,
    VX_SAFETENSORS_HEADER_ERROR_INVALID_UTF8 = 3,
    VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON = 4,
    VX_SAFETENSORS_HEADER_ERROR_DUPLICATE_KEY = 5,
    VX_SAFETENSORS_HEADER_ERROR_ROOT_NOT_OBJECT = 6,
    VX_SAFETENSORS_HEADER_ERROR_INVALID_METADATA = 7,
    VX_SAFETENSORS_HEADER_ERROR_UNKNOWN_DTYPE = 8,
    VX_SAFETENSORS_HEADER_ERROR_INVALID_TENSOR = 9,
    VX_SAFETENSORS_HEADER_ERROR_INVALID_SHAPE = 10,
    VX_SAFETENSORS_HEADER_ERROR_NON_BYTE_ALIGNED = 11,
    VX_SAFETENSORS_HEADER_ERROR_INVALID_OFFSETS = 12,
    VX_SAFETENSORS_HEADER_ERROR_BYTE_LENGTH_MISMATCH = 13,
    VX_SAFETENSORS_HEADER_ERROR_INVALID_COVERAGE = 14,
    VX_SAFETENSORS_HEADER_ERROR_INTEGER_OVERFLOW = 15,
    VX_SAFETENSORS_HEADER_ERROR_DEPTH_LIMIT = 16,
    VX_SAFETENSORS_HEADER_ERROR_RESERVED_NONZERO = 17,
    VX_SAFETENSORS_HEADER_ERROR_INTERNAL = 18
};

typedef uint32_t VxSafetensorsHeaderErrorSectionV1;
enum {
    VX_SAFETENSORS_HEADER_SECTION_NONE = 0,
    VX_SAFETENSORS_HEADER_SECTION_PREFIX = 1,
    VX_SAFETENSORS_HEADER_SECTION_JSON = 2,
    VX_SAFETENSORS_HEADER_SECTION_ROOT = 3,
    VX_SAFETENSORS_HEADER_SECTION_METADATA = 4,
    VX_SAFETENSORS_HEADER_SECTION_TENSOR = 5,
    VX_SAFETENSORS_HEADER_SECTION_COVERAGE = 6
};

/* Stable wire vocabulary; native adapters map this to generated VxDataType. */
typedef uint32_t VxSafetensorsHeaderDTypeV1;
enum {
    VX_SAFETENSORS_HEADER_DTYPE_BOOL = 1,
    VX_SAFETENSORS_HEADER_DTYPE_F4 = 2,
    VX_SAFETENSORS_HEADER_DTYPE_F6_E2M3 = 3,
    VX_SAFETENSORS_HEADER_DTYPE_F6_E3M2 = 4,
    VX_SAFETENSORS_HEADER_DTYPE_U8 = 5,
    VX_SAFETENSORS_HEADER_DTYPE_I8 = 6,
    VX_SAFETENSORS_HEADER_DTYPE_F8_E5M2 = 7,
    VX_SAFETENSORS_HEADER_DTYPE_F8_E4M3 = 8,
    VX_SAFETENSORS_HEADER_DTYPE_F8_E8M0 = 9,
    VX_SAFETENSORS_HEADER_DTYPE_F8_E4M3FNUZ = 10,
    VX_SAFETENSORS_HEADER_DTYPE_F8_E5M2FNUZ = 11,
    VX_SAFETENSORS_HEADER_DTYPE_I16 = 12,
    VX_SAFETENSORS_HEADER_DTYPE_U16 = 13,
    VX_SAFETENSORS_HEADER_DTYPE_F16 = 14,
    VX_SAFETENSORS_HEADER_DTYPE_BF16 = 15,
    VX_SAFETENSORS_HEADER_DTYPE_I32 = 16,
    VX_SAFETENSORS_HEADER_DTYPE_U32 = 17,
    VX_SAFETENSORS_HEADER_DTYPE_F32 = 18,
    VX_SAFETENSORS_HEADER_DTYPE_C64 = 19,
    VX_SAFETENSORS_HEADER_DTYPE_F64 = 20,
    VX_SAFETENSORS_HEADER_DTYPE_I64 = 21,
    VX_SAFETENSORS_HEADER_DTYPE_U64 = 22
};

enum {
    VX_SAFETENSORS_HEADER_FLAG_HAS_METADATA = 1u,
    /* error_index identifies a real member even when its value is zero. */
    VX_SAFETENSORS_HEADER_FLAG_ERROR_INDEX_PRESENT = 2u
};

#if defined(__clang__) || defined(__GNUC__)
#define VX_SAFETENSORS_HEADER_PACKED __attribute__((packed, aligned(4)))
#else
#error "The VolvoxAI SafeTensors header ABI requires packed-record support."
#endif

typedef struct VX_SAFETENSORS_HEADER_PACKED VxSafetensorsHeaderResponseV1 {
    uint32_t magic;
    uint32_t abi_version;
    VxSafetensorsHeaderStatusV1 status;
    VxSafetensorsHeaderErrorCodeV1 error_code;
    VxSafetensorsHeaderErrorSectionV1 error_section;
    uint32_t error_index;
    uint32_t error_subindex;
    uint32_t error_byte_offset;
    uint32_t written_bytes;
    uint32_t required_response_bytes;
    uint32_t required_scratch_bytes;
    uint32_t flags;
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t metadata_offset;
    uint32_t metadata_count;
    uint32_t shape_offset;
    uint32_t shape_count;
    uint32_t string_offset;
    uint32_t string_bytes;
    uint32_t header_bytes;
    uint32_t reserved0;
    uint64_t data_base;
    uint64_t data_bytes;
    uint64_t file_bytes;
    uint64_t reserved1;
    uint64_t reserved2;
} VxSafetensorsHeaderResponseV1;

typedef struct VX_SAFETENSORS_HEADER_PACKED VxSafetensorsHeaderTensorV1 {
    uint32_t name_offset;
    uint32_t name_length;
    VxSafetensorsHeaderDTypeV1 dtype;
    uint32_t rank;
    uint32_t shape_first;
    uint32_t member_index;
    uint64_t data_start;
    uint64_t data_end;
    uint64_t byte_length;
    uint32_t reserved0;
    uint32_t reserved1;
} VxSafetensorsHeaderTensorV1;

typedef struct VX_SAFETENSORS_HEADER_PACKED VxSafetensorsHeaderMetadataV1 {
    uint32_t key_offset;
    uint32_t key_length;
    uint32_t value_offset;
    uint32_t value_length;
    uint32_t member_index;
    uint32_t reserved;
} VxSafetensorsHeaderMetadataV1;

#if defined(__cplusplus)
static_assert(sizeof(VxSafetensorsHeaderResponseV1) == 128,
              "safetensors header response ABI drift");
static_assert(sizeof(VxSafetensorsHeaderTensorV1) == 56,
              "safetensors header tensor ABI drift");
static_assert(sizeof(VxSafetensorsHeaderMetadataV1) == 24,
              "safetensors header metadata ABI drift");
#else
_Static_assert(sizeof(VxSafetensorsHeaderResponseV1) == 128,
               "safetensors header response ABI drift");
_Static_assert(sizeof(VxSafetensorsHeaderTensorV1) == 56,
               "safetensors header tensor ABI drift");
_Static_assert(sizeof(VxSafetensorsHeaderMetadataV1) == 24,
               "safetensors header metadata ABI drift");
#endif

VX_SAFETENSORS_HEADER_API uint32_t vx_safetensors_header_abi_version(void);

/*
 * `response` is 8-byte aligned and contains at least the 128-byte header.
 * `scratch` is either NULL/zero or a 16-byte aligned range. Ranges must be
 * mutually disjoint. SCRATCH_TOO_SMALL reports a strictly larger lower bound;
 * callers retry geometrically until parsing reaches an exact response size.
 */
VX_SAFETENSORS_HEADER_API int32_t vx_safetensors_header_parse_v1(
    const uint8_t* header,
    uint32_t header_bytes,
    uint64_t file_bytes,
    uint64_t data_bytes,
    uint8_t* response,
    uint32_t response_bytes,
    uint8_t* scratch,
    uint32_t scratch_bytes);

#undef VX_SAFETENSORS_HEADER_PACKED

#ifdef VX_SAFETENSORS_HEADER_API_DEFINED_HERE
#undef VX_SAFETENSORS_HEADER_API_DEFINED_HERE
#undef VX_SAFETENSORS_HEADER_API
#endif

#ifdef __cplusplus
}
#endif

#endif
