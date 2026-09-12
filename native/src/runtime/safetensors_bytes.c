#include "safetensors.h"
#include "safetensors_header.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAFETENSORS_MAX_HEADER_SIZE INT64_C(100000000)

static uint64_t safetensors_read_u64_le(const unsigned char* bytes) {
    uint64_t value = 0;
    int index;
    for (index = 7; index >= 0; index--)
        value = (value << 8) | bytes[index];
    return value;
}

static void safetensors_string_pointer_sift_down(const char** values,
                                                 size_t root,
                                                 size_t count) {
    while (count >= 2u && root <= (count - 2u) / 2u) {
        size_t child = root * 2u + 1u;
        const char* swap_value;
        if (child + 1u < count &&
            strcmp(values[child], values[child + 1u]) < 0)
            child++;
        if (strcmp(values[root], values[child]) >= 0) return;
        swap_value = values[root];
        values[root] = values[child];
        values[child] = swap_value;
        root = child;
    }
}

static void safetensors_sort_string_pointers(const char** values,
                                             size_t count) {
    size_t start;
    size_t end;
    if (count < 2u) return;
    for (start = count / 2u; start > 0u; start--)
        safetensors_string_pointer_sift_down(values, start - 1u, count);
    for (end = count; end > 1u; end--) {
        const char* swap_value = values[0];
        values[0] = values[end - 1u];
        values[end - 1u] = swap_value;
        safetensors_string_pointer_sift_down(values, 0u, end - 1u);
    }
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
    if (strcmp(dtype, "F8_E4M3FNUZ") == 0)
        return SAFETENSORS_DTYPE_F8_E4M3FNUZ;
    if (strcmp(dtype, "F8_E5M2FNUZ") == 0)
        return SAFETENSORS_DTYPE_F8_E5M2FNUZ;
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

int safetensors_tensor_nbytes(VxDataType dtype, const int* shape, int ndim,
                              size_t* out_nbytes) {
    size_t bits;
    size_t elements = 1;
    size_t whole_octets;
    size_t tail_elements;
    size_t tail_bits;
    size_t tail_octets;
    int index;
    if (!out_nbytes || ndim < 0 || ndim > 8 || (ndim > 0 && !shape))
        return -1;
    bits = safetensors_dtype_bit_width(dtype);
    if (bits == 0) return -1;
    for (index = 0; index < ndim; index++) {
        if (shape[index] < 0 ||
            (elements != 0 && (size_t)shape[index] > SIZE_MAX / elements))
            return -1;
        elements *= (size_t)shape[index];
    }
    /* Compute packed bytes without first materializing `elements * bits`.
     * That intermediate is eight times wider than the result and used to
     * reject otherwise representable tensors on wasm32. Eight elements of
     * any supported dtype occupy exactly `bits` bytes; only the tail needs a
     * byte-alignment check. */
    if (elements / 8u > SIZE_MAX / bits) return -1;
    whole_octets = (elements / 8u) * bits;
    tail_elements = elements % 8u;
    tail_bits = tail_elements * bits;
    if (tail_bits % 8u != 0u) return -1;
    tail_octets = tail_bits / 8u;
    if (whole_octets > SIZE_MAX - tail_octets) return -1;
    *out_nbytes = whole_octets + tail_octets;
    return 0;
}

static int safetensors_tensor_offset_compare(const SafetensorsTensor* left,
                                             const SafetensorsTensor* right) {
    if (left->data_start < right->data_start) return -1;
    if (left->data_start > right->data_start) return 1;
    return strcmp(left->name, right->name);
}

static void safetensors_tensor_sift_down(SafetensorsTensor* tensors,
                                        size_t root,
                                        size_t count) {
    while (count >= 2u && root <= (count - 2u) / 2u) {
        size_t child = root * 2u + 1u;
        SafetensorsTensor swap_value;
        if (child + 1u < count &&
            safetensors_tensor_offset_compare(&tensors[child],
                                              &tensors[child + 1u]) < 0)
            child++;
        if (safetensors_tensor_offset_compare(&tensors[root],
                                              &tensors[child]) >= 0)
            return;
        swap_value = tensors[root];
        tensors[root] = tensors[child];
        tensors[child] = swap_value;
        root = child;
    }
}

static void safetensors_sort_tensors_by_offset(SafetensorsTensor* tensors,
                                               int tensor_count) {
    size_t count = tensor_count > 0 ? (size_t)tensor_count : 0u;
    size_t start;
    size_t end;
    if (count < 2u) return;
    for (start = count / 2u; start > 0u; start--)
        safetensors_tensor_sift_down(tensors, start - 1u, count);
    for (end = count; end > 1u; end--) {
        SafetensorsTensor swap_value = tensors[0];
        tensors[0] = tensors[end - 1u];
        tensors[end - 1u] = swap_value;
        safetensors_tensor_sift_down(tensors, 0u, end - 1u);
    }
}

static int safetensors_tensor_names_unique(const SafetensorsTensor* tensors,
                                           int tensor_count) {
    const char** names;
    size_t count = tensor_count > 0 ? (size_t)tensor_count : 0u;
    size_t index;
    if (count > SIZE_MAX / sizeof(*names)) return 0;
    names = count ? (const char**)malloc(count * sizeof(*names)) : NULL;
    if (count && !names) return 0;
    for (index = 0u; index < count; index++) names[index] = tensors[index].name;
    safetensors_sort_string_pointers(names, count);
    for (index = 1u; index < count; index++) {
        if (strcmp(names[index - 1u], names[index]) == 0) {
            free(names);
            return 0;
        }
    }
    free(names);
    return 1;
}

void safetensors_borrowed_bytes_free(SafetensorsBorrowedBytes* view) {
    if (!view) return;
    free(view->metadata_json);
    free(view->tensors);
    memset(view, 0, sizeof(*view));
}

static uint32_t safetensors_header_u32(const unsigned char* bytes) {
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8u) |
           ((uint32_t)bytes[2] << 16u) |
           ((uint32_t)bytes[3] << 24u);
}

static uint64_t safetensors_header_u64(const unsigned char* bytes) {
    return (uint64_t)safetensors_header_u32(bytes) |
           ((uint64_t)safetensors_header_u32(bytes + 4u) << 32u);
}

static void* safetensors_aligned_allocate(size_t bytes, size_t alignment,
                                          void** raw) {
    uintptr_t address;
    void* allocation;
    if (!raw || alignment == 0u || (alignment & (alignment - 1u)) != 0u ||
        bytes > SIZE_MAX - (alignment - 1u))
        return NULL;
    allocation = malloc(bytes + alignment - 1u);
    if (!allocation) return NULL;
    address = ((uintptr_t)allocation + alignment - 1u) &
              ~(uintptr_t)(alignment - 1u);
    *raw = allocation;
    return (void*)address;
}

static VxDataType safetensors_header_dtype(uint32_t dtype) {
    switch (dtype) {
        case VX_SAFETENSORS_HEADER_DTYPE_BOOL:
            return SAFETENSORS_DTYPE_BOOL;
        case VX_SAFETENSORS_HEADER_DTYPE_F4:
            return SAFETENSORS_DTYPE_F4;
        case VX_SAFETENSORS_HEADER_DTYPE_F6_E2M3:
            return SAFETENSORS_DTYPE_F6_E2M3;
        case VX_SAFETENSORS_HEADER_DTYPE_F6_E3M2:
            return SAFETENSORS_DTYPE_F6_E3M2;
        case VX_SAFETENSORS_HEADER_DTYPE_U8:
            return SAFETENSORS_DTYPE_U8;
        case VX_SAFETENSORS_HEADER_DTYPE_I8:
            return SAFETENSORS_DTYPE_I8;
        case VX_SAFETENSORS_HEADER_DTYPE_F8_E5M2:
            return SAFETENSORS_DTYPE_F8_E5M2;
        case VX_SAFETENSORS_HEADER_DTYPE_F8_E4M3:
            return SAFETENSORS_DTYPE_F8_E4M3;
        case VX_SAFETENSORS_HEADER_DTYPE_F8_E8M0:
            return SAFETENSORS_DTYPE_F8_E8M0;
        case VX_SAFETENSORS_HEADER_DTYPE_F8_E4M3FNUZ:
            return SAFETENSORS_DTYPE_F8_E4M3FNUZ;
        case VX_SAFETENSORS_HEADER_DTYPE_F8_E5M2FNUZ:
            return SAFETENSORS_DTYPE_F8_E5M2FNUZ;
        case VX_SAFETENSORS_HEADER_DTYPE_I16:
            return SAFETENSORS_DTYPE_I16;
        case VX_SAFETENSORS_HEADER_DTYPE_U16:
            return SAFETENSORS_DTYPE_U16;
        case VX_SAFETENSORS_HEADER_DTYPE_F16:
            return SAFETENSORS_DTYPE_F16;
        case VX_SAFETENSORS_HEADER_DTYPE_BF16:
            return SAFETENSORS_DTYPE_BF16;
        case VX_SAFETENSORS_HEADER_DTYPE_I32:
            return SAFETENSORS_DTYPE_I32;
        case VX_SAFETENSORS_HEADER_DTYPE_U32:
            return SAFETENSORS_DTYPE_U32;
        case VX_SAFETENSORS_HEADER_DTYPE_F32:
            return SAFETENSORS_DTYPE_F32;
        case VX_SAFETENSORS_HEADER_DTYPE_C64:
            return SAFETENSORS_DTYPE_C64;
        case VX_SAFETENSORS_HEADER_DTYPE_F64:
            return SAFETENSORS_DTYPE_F64;
        case VX_SAFETENSORS_HEADER_DTYPE_I64:
            return SAFETENSORS_DTYPE_I64;
        case VX_SAFETENSORS_HEADER_DTYPE_U64:
            return SAFETENSORS_DTYPE_U64;
        default:
            return SAFETENSORS_DTYPE_UNKNOWN;
    }
}

static int safetensors_wire_slice(const unsigned char* response,
                                  uint32_t written,
                                  uint32_t string_offset,
                                  uint32_t string_bytes,
                                  uint32_t offset,
                                  uint32_t bytes,
                                  const unsigned char** out) {
    if (!response || !out || offset > string_bytes ||
        bytes > string_bytes - offset || string_offset > written ||
        string_bytes > written - string_offset)
        return 0;
    *out = response + string_offset + offset;
    return 1;
}

static int safetensors_json_escaped_size(const unsigned char* value,
                                         uint32_t bytes,
                                         size_t* total) {
    uint32_t index;
    size_t size = 0u;
    if (!total || (bytes && !value)) return 0;
    for (index = 0u; index < bytes; index++) {
        size_t add = 1u;
        if (value[index] == (unsigned char)'"' ||
            value[index] == (unsigned char)'\\' ||
            value[index] == 0x08u || value[index] == 0x0cu ||
            value[index] == 0x0au || value[index] == 0x0du ||
            value[index] == 0x09u)
            add = 2u;
        else if (value[index] < 0x20u)
            add = 6u;
        if (size > SIZE_MAX - add) return 0;
        size += add;
    }
    *total = size;
    return 1;
}

static char* safetensors_json_escape_write(char* output,
                                           const unsigned char* value,
                                           uint32_t bytes) {
    static const char hex[] = "0123456789abcdef";
    uint32_t index;
    for (index = 0u; index < bytes; index++) {
        unsigned char byte = value[index];
        if (byte == (unsigned char)'"' || byte == (unsigned char)'\\') {
            *output++ = '\\';
            *output++ = (char)byte;
        } else if (byte == 0x08u || byte == 0x0cu || byte == 0x0au ||
                   byte == 0x0du || byte == 0x09u) {
            *output++ = '\\';
            *output++ = byte == 0x08u ? 'b' : byte == 0x0cu ? 'f' :
                        byte == 0x0au ? 'n' : byte == 0x0du ? 'r' : 't';
        } else if (byte < 0x20u) {
            *output++ = '\\';
            *output++ = 'u';
            *output++ = '0';
            *output++ = '0';
            *output++ = hex[byte >> 4u];
            *output++ = hex[byte & 0x0fu];
        } else {
            *output++ = (char)byte;
        }
    }
    return output;
}

static char* safetensors_header_metadata_json(
    const unsigned char* response,
    uint32_t written,
    uint32_t metadata_offset,
    uint32_t metadata_count,
    uint32_t string_offset,
    uint32_t string_bytes) {
    size_t output_bytes = 2u;
    uint32_t index;
    char* output;
    char* cursor;
    for (index = 0u; index < metadata_count; index++) {
        const unsigned char* record = response + metadata_offset +
            index * VOLVOXAI_SAFETENSORS_HEADER_METADATA_BYTES;
        const unsigned char* key;
        const unsigned char* value;
        size_t key_bytes;
        size_t value_bytes;
        if (safetensors_header_u32(record + 16u) != index ||
            safetensors_header_u32(record + 20u) != 0u ||
            !safetensors_wire_slice(
                response, written, string_offset, string_bytes,
                safetensors_header_u32(record),
                safetensors_header_u32(record + 4u), &key) ||
            !safetensors_wire_slice(
                response, written, string_offset, string_bytes,
                safetensors_header_u32(record + 8u),
                safetensors_header_u32(record + 12u), &value) ||
            !safetensors_json_escaped_size(
                key, safetensors_header_u32(record + 4u), &key_bytes) ||
            !safetensors_json_escaped_size(
                value, safetensors_header_u32(record + 12u), &value_bytes))
            return NULL;
        if (index != 0u) {
            if (output_bytes == SIZE_MAX) return NULL;
            output_bytes++;
        }
        if (key_bytes > SIZE_MAX - output_bytes) return NULL;
        output_bytes += key_bytes;
        if (value_bytes > SIZE_MAX - output_bytes) return NULL;
        output_bytes += value_bytes;
        if (output_bytes > SIZE_MAX - 5u) return NULL;
        output_bytes += 5u;
    }
    if (output_bytes == SIZE_MAX) return NULL;
    output = (char*)malloc(output_bytes + 1u);
    if (!output) return NULL;
    cursor = output;
    *cursor++ = '{';
    for (index = 0u; index < metadata_count; index++) {
        const unsigned char* record = response + metadata_offset +
            index * VOLVOXAI_SAFETENSORS_HEADER_METADATA_BYTES;
        const unsigned char* key;
        const unsigned char* value;
        if (!safetensors_wire_slice(
                response, written, string_offset, string_bytes,
                safetensors_header_u32(record),
                safetensors_header_u32(record + 4u), &key) ||
            !safetensors_wire_slice(
                response, written, string_offset, string_bytes,
                safetensors_header_u32(record + 8u),
                safetensors_header_u32(record + 12u), &value)) {
            free(output);
            return NULL;
        }
        if (index != 0u) *cursor++ = ',';
        *cursor++ = '"';
        cursor = safetensors_json_escape_write(
            cursor, key, safetensors_header_u32(record + 4u));
        *cursor++ = '"';
        *cursor++ = ':';
        *cursor++ = '"';
        cursor = safetensors_json_escape_write(
            cursor, value, safetensors_header_u32(record + 12u));
        *cursor++ = '"';
    }
    *cursor++ = '}';
    *cursor = '\0';
    return output;
}

int safetensors_parse_borrowed_bytes(const void* bytes_value,
                                     size_t byte_count,
                                     SafetensorsBorrowedBytes* out) {
    const unsigned char* bytes = (const unsigned char*)bytes_value;
    uint64_t header_size_u64;
    size_t header_size;
    size_t data_base;
    union {
        uint64_t alignment;
        unsigned char bytes[VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_BYTES];
    } query;
    unsigned char* response = query.bytes;
    void* response_raw = NULL;
    uint32_t response_bytes = VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_BYTES;
    unsigned char* scratch = NULL;
    void* scratch_raw = NULL;
    uint32_t scratch_bytes = 0u;
    int32_t status = VX_SAFETENSORS_HEADER_STATUS_INVALID_HEADER;
    uint32_t written;
    uint32_t tensor_offset;
    uint32_t tensor_count_u32;
    uint32_t metadata_offset;
    uint32_t metadata_count;
    uint32_t shape_offset;
    uint32_t shape_count;
    uint32_t string_offset;
    uint32_t string_bytes;
    SafetensorsTensor* tensors = NULL;
    char* metadata_json = NULL;
    int has_metadata = 0;
    int tensor_count = 0;
    uint32_t index;
    uint32_t previous_member_index = 0u;
    int has_previous_member_index = 0;
    unsigned attempt;
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!bytes || byte_count < 8u || byte_count > (size_t)INT64_MAX)
        return -1;
    header_size_u64 = safetensors_read_u64_le(bytes);
    if (header_size_u64 > (uint64_t)(byte_count - 8u) ||
        header_size_u64 > (uint64_t)SAFETENSORS_MAX_HEADER_SIZE ||
        header_size_u64 > (uint64_t)SIZE_MAX ||
        header_size_u64 > UINT32_MAX - 8u)
        return -1;
    header_size = (size_t)header_size_u64;
    data_base = 8u + header_size;
    if (header_size == 0u) return -1;
    memset(query.bytes, 0, sizeof(query.bytes));
    for (attempt = 0u; attempt < 32u; attempt++) {
        uint32_t required_response;
        uint32_t required_scratch;
        status = vx_safetensors_header_parse_v1(
            bytes, (uint32_t)data_base, (uint64_t)byte_count,
            (uint64_t)(byte_count - data_base), response, response_bytes,
            scratch, scratch_bytes);
        required_response = safetensors_header_u32(response + 36u);
        required_scratch = safetensors_header_u32(response + 40u);
        if (status == VX_SAFETENSORS_HEADER_STATUS_SCRATCH_TOO_SMALL) {
            void* next_raw = NULL;
            unsigned char* next;
            uint32_t next_bytes = scratch_bytes && scratch_bytes <= UINT32_MAX / 2u
                ? scratch_bytes * 2u : required_scratch;
            if (next_bytes < required_scratch) next_bytes = required_scratch;
            if (next_bytes <= scratch_bytes) goto fail;
            next = (unsigned char*)safetensors_aligned_allocate(
                next_bytes, 16u, &next_raw);
            if (!next) goto fail;
            free(scratch_raw);
            scratch = next;
            scratch_raw = next_raw;
            scratch_bytes = next_bytes;
            continue;
        }
        if (status == VX_SAFETENSORS_HEADER_STATUS_RESPONSE_TOO_SMALL) {
            void* next_raw = NULL;
            unsigned char* next;
            if (required_response <= response_bytes) goto fail;
            next = (unsigned char*)safetensors_aligned_allocate(
                required_response, 8u, &next_raw);
            if (!next) goto fail;
            free(response_raw);
            response = next;
            response_raw = next_raw;
            response_bytes = required_response;
            continue;
        }
        break;
    }
    if (status != VX_SAFETENSORS_HEADER_STATUS_OK || attempt == 32u ||
        safetensors_header_u32(response) !=
            VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_MAGIC ||
        safetensors_header_u32(response + 4u) !=
            VOLVOXAI_SAFETENSORS_HEADER_ABI_VERSION ||
        safetensors_header_u32(response + 12u) !=
            VX_SAFETENSORS_HEADER_ERROR_NONE ||
        safetensors_header_u32(response + 16u) !=
            VX_SAFETENSORS_HEADER_SECTION_NONE ||
        safetensors_header_u32(response + 80u) != (uint32_t)data_base ||
        safetensors_header_u32(response + 84u) != 0u ||
        safetensors_header_u64(response + 88u) != (uint64_t)data_base ||
        safetensors_header_u64(response + 96u) !=
            (uint64_t)(byte_count - data_base) ||
        safetensors_header_u64(response + 104u) != (uint64_t)byte_count ||
        safetensors_header_u64(response + 112u) != 0u ||
        safetensors_header_u64(response + 120u) != 0u)
        goto fail;
    written = safetensors_header_u32(response + 32u);
    if (written != safetensors_header_u32(response + 36u) ||
        written > response_bytes) goto fail;
    tensor_offset = safetensors_header_u32(response + 48u);
    tensor_count_u32 = safetensors_header_u32(response + 52u);
    metadata_offset = safetensors_header_u32(response + 56u);
    metadata_count = safetensors_header_u32(response + 60u);
    shape_offset = safetensors_header_u32(response + 64u);
    shape_count = safetensors_header_u32(response + 68u);
    string_offset = safetensors_header_u32(response + 72u);
    string_bytes = safetensors_header_u32(response + 76u);
    if (tensor_count_u32 > INT_MAX ||
        tensor_offset > written ||
        tensor_count_u32 >
            (written - tensor_offset) /
                VOLVOXAI_SAFETENSORS_HEADER_TENSOR_BYTES ||
        metadata_offset > written ||
        metadata_count >
            (written - metadata_offset) /
                VOLVOXAI_SAFETENSORS_HEADER_METADATA_BYTES ||
        shape_offset > written || shape_count > (written - shape_offset) / 8u ||
        string_offset > written || string_bytes > written - string_offset)
        goto fail;
    has_metadata =
        (safetensors_header_u32(response + 44u) &
         VX_SAFETENSORS_HEADER_FLAG_HAS_METADATA) != 0u;
    if (safetensors_header_u32(response + 44u) &
        ~(uint32_t)VX_SAFETENSORS_HEADER_FLAG_HAS_METADATA)
        goto fail;
    if (!has_metadata && metadata_count != 0u) goto fail;
    if (has_metadata) {
        metadata_json = safetensors_header_metadata_json(
            response, written, metadata_offset, metadata_count,
            string_offset, string_bytes);
        if (!metadata_json) goto fail;
    }
    tensor_count = (int)tensor_count_u32;
    if (tensor_count > 0) {
        if ((size_t)tensor_count > SIZE_MAX / sizeof(*tensors)) goto fail;
        tensors = (SafetensorsTensor*)calloc((size_t)tensor_count,
                                             sizeof(*tensors));
        if (!tensors) goto fail;
    }
    for (index = 0u; index < tensor_count_u32; index++) {
        const unsigned char* record = response + tensor_offset +
            index * VOLVOXAI_SAFETENSORS_HEADER_TENSOR_BYTES;
        const unsigned char* name;
        uint32_t name_bytes = safetensors_header_u32(record + 4u);
        uint32_t rank = safetensors_header_u32(record + 12u);
        uint32_t shape_first = safetensors_header_u32(record + 16u);
        uint32_t member_index = safetensors_header_u32(record + 20u);
        uint64_t start = safetensors_header_u64(record + 24u);
        uint64_t end = safetensors_header_u64(record + 32u);
        uint64_t nbytes = safetensors_header_u64(record + 40u);
        size_t name_index;
        uint32_t dimension_index;
        VxDataType dtype = safetensors_header_dtype(
            safetensors_header_u32(record + 8u));
        if (dtype == SAFETENSORS_DTYPE_UNKNOWN || rank > 8u ||
            name_bytes >= sizeof(tensors[index].name) ||
            member_index > tensor_count_u32 - (has_metadata ? 0u : 1u) ||
            (has_previous_member_index &&
             member_index <= previous_member_index) ||
            shape_first > shape_count || rank > shape_count - shape_first ||
            start > end || end > (uint64_t)(byte_count - data_base) ||
            end - start != nbytes || nbytes > SIZE_MAX ||
            start > INT64_MAX || end > INT64_MAX ||
            start > (uint64_t)(SIZE_MAX - data_base) ||
            safetensors_header_u32(record + 48u) != 0u ||
            safetensors_header_u32(record + 52u) != 0u ||
            !safetensors_wire_slice(
                response, written, string_offset, string_bytes,
                safetensors_header_u32(record), name_bytes, &name))
            goto fail;
        previous_member_index = member_index;
        has_previous_member_index = 1;
        for (name_index = 0u; name_index < name_bytes; name_index++) {
            if (name[name_index] == 0u) goto fail;
            tensors[index].name[name_index] = (char)name[name_index];
        }
        tensors[index].name[name_index] = '\0';
        tensors[index].dtype = dtype;
        tensors[index].ndim = (int)rank;
        for (dimension_index = 0u; dimension_index < rank;
             dimension_index++) {
            uint64_t dimension = safetensors_header_u64(
                response + shape_offset +
                (shape_first + dimension_index) * 8u);
            if (dimension > INT_MAX) goto fail;
            tensors[index].shape[dimension_index] = (int)dimension;
        }
        tensors[index].data_start = (int64_t)start;
        tensors[index].data_end = (int64_t)end;
        tensors[index].nbytes = (size_t)nbytes;
        tensors[index].data = (void*)(bytes + data_base + (size_t)start);
        tensors[index].flags = SAFETENSORS_TENSOR_READABLE;
    }
    if (!safetensors_tensor_names_unique(tensors, tensor_count)) goto fail;
    safetensors_sort_tensors_by_offset(tensors, tensor_count);
    free(response_raw);
    free(scratch_raw);
    out->bytes = bytes;
    out->byte_count = byte_count;
    out->data_base = data_base;
    out->metadata_json = metadata_json;
    out->has_metadata = has_metadata;
    out->tensors = tensors;
    out->tensor_count = tensor_count;
    return 0;

fail:
    free(response_raw);
    free(scratch_raw);
    free(metadata_json);
    free(tensors);
    return -1;
}
