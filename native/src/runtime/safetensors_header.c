#ifndef VX_SAFETENSORS_HEADER_API
#define VX_SAFETENSORS_HEADER_API
#define VX_SAFETENSORS_HEADER_API_LOCAL_EMPTY 1
#endif

#include "safetensors_header.h"

#include <stddef.h>
#include <stdint.h>

#define VX_ST_JSON_MAX_DEPTH UINT32_C(512)
#define VX_ST_KEY_RECORD_BYTES UINT32_C(32)
#define VX_ST_OBJECT_PLAN_BYTES UINT32_C(8)
#define VX_ST_FNV_OFFSET UINT64_C(1469598103934665603)
#define VX_ST_FNV_PRIME UINT64_C(1099511628211)

typedef struct VxStStringSpan {
    uint32_t start;
    uint32_t end;
    uint32_t decoded_bytes;
    uint64_t hash;
} VxStStringSpan;

typedef struct VxStNumberSpan {
    uint32_t start;
    uint32_t end;
} VxStNumberSpan;

typedef struct VxStScratch {
    uint8_t* bytes;
    uint32_t capacity;
    uint32_t used;
    uint32_t required;
} VxStScratch;

typedef struct VxStBuild {
    uint8_t* response;
    uint32_t response_bytes;
    uint32_t write;
    uint32_t tensor_offset;
    uint32_t metadata_offset;
    uint32_t shape_offset;
    uint32_t string_offset;
    uint32_t tensor_count;
    uint32_t metadata_count;
    uint32_t shape_count;
    uint32_t string_bytes;
    uint64_t payload_sum;
    uint32_t has_metadata;
} VxStBuild;

typedef struct VxStObjectPlan {
    uint8_t* counts;
    uint32_t capacity;
    uint32_t object_count;
    uint32_t cursor;
    uint32_t count_bytes;
} VxStObjectPlan;

typedef struct VxStParser {
    const uint8_t* bytes;
    uint32_t length;
    uint32_t cursor;
    VxStScratch* scratch;
    VxStBuild* build;
    VxStObjectPlan* object_plan;
    VxSafetensorsHeaderErrorCodeV1 error_code;
    VxSafetensorsHeaderErrorSectionV1 error_section;
    uint32_t error_index;
    uint32_t error_index_present;
    uint32_t error_subindex;
    uint32_t error_byte_offset;
} VxStParser;

typedef struct VxStStringCursor {
    const uint8_t* bytes;
    uint32_t cursor;
    uint32_t end;
    uint8_t pending[4];
    uint32_t pending_cursor;
    uint32_t pending_bytes;
} VxStStringCursor;

typedef struct VxStTensorTemp {
    uint32_t dtype;
    uint32_t dtype_bits;
    uint32_t shape_first;
    uint32_t rank;
    uint64_t elements;
    uint64_t start;
    uint64_t end;
    uint64_t expected_bytes;
    uint32_t has_dtype;
    uint32_t has_shape;
    uint32_t has_offsets;
} VxStTensorTemp;

static void vx_st_zero(uint8_t* bytes, uint32_t count) {
    uint32_t index;
    if (!bytes) return;
    for (index = 0u; index < count; index++) bytes[index] = 0u;
}

static uint32_t vx_st_get_u32(const uint8_t* bytes) {
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8u) |
           ((uint32_t)bytes[2] << 16u) |
           ((uint32_t)bytes[3] << 24u);
}

static uint64_t vx_st_get_u64(const uint8_t* bytes) {
    return (uint64_t)vx_st_get_u32(bytes) |
           ((uint64_t)vx_st_get_u32(bytes + 4u) << 32u);
}

static void vx_st_put_u32(uint8_t* bytes, uint32_t value) {
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8u);
    bytes[2] = (uint8_t)(value >> 16u);
    bytes[3] = (uint8_t)(value >> 24u);
}

static void vx_st_put_i32(uint8_t* bytes, int32_t value) {
    vx_st_put_u32(bytes, (uint32_t)value);
}

static void vx_st_put_u64(uint8_t* bytes, uint64_t value) {
    vx_st_put_u32(bytes, (uint32_t)value);
    vx_st_put_u32(bytes + 4u, (uint32_t)(value >> 32u));
}

static int vx_st_add_u32(uint32_t left, uint32_t right, uint32_t* out) {
    if (!out || left > UINT32_MAX - right) return 0;
    *out = left + right;
    return 1;
}

static int vx_st_mul_u32(uint32_t left, uint32_t right, uint32_t* out) {
    if (!out || (left != 0u && right > UINT32_MAX / left)) return 0;
    *out = left * right;
    return 1;
}

static int vx_st_align_u32(uint32_t value, uint32_t alignment,
                           uint32_t* out) {
    uint32_t remainder;
    if (!alignment || !out) return 0;
    remainder = value % alignment;
    if (remainder == 0u) {
        *out = value;
        return 1;
    }
    return vx_st_add_u32(value, alignment - remainder, out);
}

static int vx_st_ranges_overlap(const void* left_value, uint32_t left_bytes,
                                const void* right_value,
                                uint32_t right_bytes) {
    uintptr_t left = (uintptr_t)left_value;
    uintptr_t right = (uintptr_t)right_value;
    uintptr_t left_end;
    uintptr_t right_end;
    if (!left_bytes || !right_bytes) return 0;
    if (left > UINTPTR_MAX - left_bytes || right > UINTPTR_MAX - right_bytes)
        return 1;
    left_end = left + left_bytes;
    right_end = right + right_bytes;
    return left < right_end && right < left_end;
}

static int vx_st_fail(VxStParser* parser,
                      VxSafetensorsHeaderErrorCodeV1 code,
                      VxSafetensorsHeaderErrorSectionV1 section,
                      uint32_t byte_offset) {
    if (parser && parser->error_code == VX_SAFETENSORS_HEADER_ERROR_NONE) {
        parser->error_code = code;
        parser->error_section = section;
        parser->error_byte_offset = byte_offset;
        if (section != VX_SAFETENSORS_HEADER_SECTION_ROOT &&
            section != VX_SAFETENSORS_HEADER_SECTION_METADATA &&
            section != VX_SAFETENSORS_HEADER_SECTION_TENSOR &&
            section != VX_SAFETENSORS_HEADER_SECTION_COVERAGE)
            parser->error_index_present = 0u;
    }
    return 0;
}

static int vx_st_is_space(uint8_t value) {
    return value == 0x20u || value == 0x09u || value == 0x0au ||
           value == 0x0du;
}

static void vx_st_skip_space(VxStParser* parser) {
    while (parser->cursor < parser->length &&
           vx_st_is_space(parser->bytes[parser->cursor]))
        parser->cursor++;
}

static int vx_st_hex(uint8_t value, uint32_t* out) {
    if (value >= (uint8_t)'0' && value <= (uint8_t)'9') {
        *out = (uint32_t)(value - (uint8_t)'0');
        return 1;
    }
    if (value >= (uint8_t)'a' && value <= (uint8_t)'f') {
        *out = (uint32_t)(value - (uint8_t)'a') + 10u;
        return 1;
    }
    if (value >= (uint8_t)'A' && value <= (uint8_t)'F') {
        *out = (uint32_t)(value - (uint8_t)'A') + 10u;
        return 1;
    }
    return 0;
}

static int vx_st_u16_escape(const uint8_t* bytes, uint32_t length,
                            uint32_t offset, uint32_t* out) {
    uint32_t value = 0u;
    uint32_t index;
    if (!out || offset > length || length - offset < 4u) return 0;
    for (index = 0u; index < 4u; index++) {
        uint32_t digit;
        if (!vx_st_hex(bytes[offset + index], &digit)) return 0;
        value = (value << 4u) | digit;
    }
    *out = value;
    return 1;
}

static int vx_st_utf8_scalar(const uint8_t* bytes, uint32_t length,
                             uint32_t* cursor, uint32_t* out) {
    uint32_t at;
    uint32_t scalar;
    uint8_t first;
    uint32_t count;
    uint32_t index;
    if (!bytes || !cursor || !out || *cursor >= length) return 0;
    at = *cursor;
    first = bytes[at];
    if (first < 0x80u) {
        *out = first;
        *cursor = at + 1u;
        return 1;
    }
    if (first >= 0xc2u && first <= 0xdfu) {
        count = 2u;
        scalar = first & 0x1fu;
    } else if (first >= 0xe0u && first <= 0xefu) {
        count = 3u;
        scalar = first & 0x0fu;
    } else if (first >= 0xf0u && first <= 0xf4u) {
        count = 4u;
        scalar = first & 0x07u;
    } else {
        return 0;
    }
    if (at > length || length - at < count) return 0;
    for (index = 1u; index < count; index++) {
        uint8_t next = bytes[at + index];
        if ((next & 0xc0u) != 0x80u) return 0;
        scalar = (scalar << 6u) | (uint32_t)(next & 0x3fu);
    }
    if ((count == 3u && scalar < 0x800u) ||
        (count == 4u && scalar < 0x10000u) ||
        (scalar >= 0xd800u && scalar <= 0xdfffu) || scalar > 0x10ffffu)
        return 0;
    *cursor = at + count;
    *out = scalar;
    return 1;
}

static int vx_st_validate_utf8(VxStParser* parser) {
    uint32_t cursor = 0u;
    while (cursor < parser->length) {
        uint32_t scalar;
        uint32_t at = cursor;
        if (parser->bytes[cursor] < 0x80u) {
            cursor++;
        } else if (!vx_st_utf8_scalar(parser->bytes, parser->length,
                                      &cursor, &scalar)) {
            return vx_st_fail(parser,
                              VX_SAFETENSORS_HEADER_ERROR_INVALID_UTF8,
                              VX_SAFETENSORS_HEADER_SECTION_JSON,
                              at + 8u);
        }
    }
    return 1;
}

static uint32_t vx_st_encode_utf8(uint32_t scalar, uint8_t output[4]) {
    if (scalar <= 0x7fu) {
        output[0] = (uint8_t)scalar;
        return 1u;
    }
    if (scalar <= 0x7ffu) {
        output[0] = (uint8_t)(0xc0u | (scalar >> 6u));
        output[1] = (uint8_t)(0x80u | (scalar & 0x3fu));
        return 2u;
    }
    if (scalar <= 0xffffu) {
        output[0] = (uint8_t)(0xe0u | (scalar >> 12u));
        output[1] = (uint8_t)(0x80u | ((scalar >> 6u) & 0x3fu));
        output[2] = (uint8_t)(0x80u | (scalar & 0x3fu));
        return 3u;
    }
    output[0] = (uint8_t)(0xf0u | (scalar >> 18u));
    output[1] = (uint8_t)(0x80u | ((scalar >> 12u) & 0x3fu));
    output[2] = (uint8_t)(0x80u | ((scalar >> 6u) & 0x3fu));
    output[3] = (uint8_t)(0x80u | (scalar & 0x3fu));
    return 4u;
}

static int vx_st_string_next(VxStStringCursor* cursor, uint8_t* out) {
    uint8_t value;
    if (cursor->pending_cursor < cursor->pending_bytes) {
        *out = cursor->pending[cursor->pending_cursor++];
        return 1;
    }
    if (cursor->cursor >= cursor->end) return 0;
    value = cursor->bytes[cursor->cursor++];
    if (value != (uint8_t)'\\') {
        *out = value;
        return 1;
    }
    if (cursor->cursor >= cursor->end) return -1;
    value = cursor->bytes[cursor->cursor++];
    if (value == (uint8_t)'"' || value == (uint8_t)'\\' ||
        value == (uint8_t)'/') {
        *out = value;
        return 1;
    }
    if (value == (uint8_t)'b') value = 0x08u;
    else if (value == (uint8_t)'f') value = 0x0cu;
    else if (value == (uint8_t)'n') value = 0x0au;
    else if (value == (uint8_t)'r') value = 0x0du;
    else if (value == (uint8_t)'t') value = 0x09u;
    else if (value == (uint8_t)'u') {
        uint32_t first;
        uint32_t scalar;
        if (!vx_st_u16_escape(cursor->bytes, cursor->end, cursor->cursor,
                              &first))
            return -1;
        cursor->cursor += 4u;
        scalar = first;
        if (first >= 0xd800u && first <= 0xdbffu) {
            uint32_t second;
            if (cursor->cursor > cursor->end ||
                cursor->end - cursor->cursor < 6u ||
                cursor->bytes[cursor->cursor] != (uint8_t)'\\' ||
                cursor->bytes[cursor->cursor + 1u] != (uint8_t)'u' ||
                !vx_st_u16_escape(cursor->bytes, cursor->end,
                                  cursor->cursor + 2u, &second) ||
                second < 0xdc00u || second > 0xdfffu)
                return -1;
            cursor->cursor += 6u;
            scalar = 0x10000u + ((first - 0xd800u) << 10u) +
                     (second - 0xdc00u);
        } else if (first >= 0xdc00u && first <= 0xdfffu) {
            return -1;
        }
        cursor->pending_bytes = vx_st_encode_utf8(scalar, cursor->pending);
        cursor->pending_cursor = 1u;
        *out = cursor->pending[0];
        return 1;
    } else {
        return -1;
    }
    *out = value;
    return 1;
}

static int vx_st_string_metrics(const uint8_t* bytes,
                                const VxStStringSpan* span,
                                uint32_t* decoded_bytes,
                                uint64_t* hash) {
    VxStStringCursor cursor;
    uint32_t count = 0u;
    uint64_t value = VX_ST_FNV_OFFSET;
    uint8_t byte = 0u;
    int status;
    cursor.bytes = bytes;
    cursor.cursor = span->start;
    cursor.end = span->end;
    cursor.pending_cursor = 0u;
    cursor.pending_bytes = 0u;
    while ((status = vx_st_string_next(&cursor, &byte)) > 0) {
        if (count == UINT32_MAX) return 0;
        count++;
        value ^= byte;
        value *= VX_ST_FNV_PRIME;
    }
    if (status < 0) return 0;
    if (value == 0u) value = UINT64_C(1);
    *decoded_bytes = count;
    *hash = value;
    return 1;
}

static int vx_st_parse_string(VxStParser* parser, VxStStringSpan* out) {
    uint32_t start;
    out->start = out->end = out->decoded_bytes = 0u;
    out->hash = 0u;
    if (parser->cursor >= parser->length ||
        parser->bytes[parser->cursor] != (uint8_t)'"')
        return vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                          VX_SAFETENSORS_HEADER_SECTION_JSON,
                          parser->cursor + 8u);
    parser->cursor++;
    start = parser->cursor;
    while (parser->cursor < parser->length) {
        uint8_t value = parser->bytes[parser->cursor];
        if (value == (uint8_t)'"') {
            out->start = start;
            out->end = parser->cursor;
            parser->cursor++;
            if (!vx_st_string_metrics(parser->bytes, out,
                                      &out->decoded_bytes, &out->hash))
                return vx_st_fail(
                    parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_UTF8,
                    VX_SAFETENSORS_HEADER_SECTION_JSON, start + 8u);
            return 1;
        }
        if (value < 0x20u)
            return vx_st_fail(parser,
                              VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                              VX_SAFETENSORS_HEADER_SECTION_JSON,
                              parser->cursor + 8u);
        if (value == (uint8_t)'\\') {
            uint32_t escape_at = parser->cursor;
            parser->cursor++;
            if (parser->cursor >= parser->length)
                return vx_st_fail(
                    parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                    VX_SAFETENSORS_HEADER_SECTION_JSON, escape_at + 8u);
            value = parser->bytes[parser->cursor++];
            if (value == (uint8_t)'"' || value == (uint8_t)'\\' ||
                value == (uint8_t)'/' || value == (uint8_t)'b' ||
                value == (uint8_t)'f' || value == (uint8_t)'n' ||
                value == (uint8_t)'r' || value == (uint8_t)'t')
                continue;
            if (value == (uint8_t)'u') {
                uint32_t first;
                if (!vx_st_u16_escape(parser->bytes, parser->length,
                                      parser->cursor, &first))
                    return vx_st_fail(
                        parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                        VX_SAFETENSORS_HEADER_SECTION_JSON, escape_at + 8u);
                parser->cursor += 4u;
                if (first >= 0xd800u && first <= 0xdbffu) {
                    uint32_t second;
                    if (parser->cursor > parser->length ||
                        parser->length - parser->cursor < 6u ||
                        parser->bytes[parser->cursor] != (uint8_t)'\\' ||
                        parser->bytes[parser->cursor + 1u] != (uint8_t)'u' ||
                        !vx_st_u16_escape(parser->bytes, parser->length,
                                          parser->cursor + 2u, &second) ||
                        second < 0xdc00u || second > 0xdfffu)
                        return vx_st_fail(
                            parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_UTF8,
                            VX_SAFETENSORS_HEADER_SECTION_JSON,
                            escape_at + 8u);
                    parser->cursor += 6u;
                } else if (first >= 0xdc00u && first <= 0xdfffu) {
                    return vx_st_fail(
                        parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_UTF8,
                        VX_SAFETENSORS_HEADER_SECTION_JSON, escape_at + 8u);
                }
                continue;
            }
            return vx_st_fail(parser,
                              VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                              VX_SAFETENSORS_HEADER_SECTION_JSON,
                              escape_at + 8u);
        }
        if (value >= 0x80u) {
            uint32_t scalar;
            uint32_t at = parser->cursor;
            if (!vx_st_utf8_scalar(parser->bytes, parser->length,
                                   &parser->cursor, &scalar))
                return vx_st_fail(
                    parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_UTF8,
                    VX_SAFETENSORS_HEADER_SECTION_JSON, at + 8u);
        } else {
            parser->cursor++;
        }
    }
    return vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                      VX_SAFETENSORS_HEADER_SECTION_JSON, start + 8u);
}

static int vx_st_string_equal(const uint8_t* bytes,
                              const VxStStringSpan* left,
                              const VxStStringSpan* right) {
    VxStStringCursor a;
    VxStStringCursor b;
    int left_status;
    int right_status;
    uint8_t left_byte = 0u;
    uint8_t right_byte = 0u;
    if (left->decoded_bytes != right->decoded_bytes || left->hash != right->hash)
        return 0;
    a.bytes = bytes;
    a.cursor = left->start;
    a.end = left->end;
    a.pending_cursor = a.pending_bytes = 0u;
    b.bytes = bytes;
    b.cursor = right->start;
    b.end = right->end;
    b.pending_cursor = b.pending_bytes = 0u;
    do {
        left_status = vx_st_string_next(&a, &left_byte);
        right_status = vx_st_string_next(&b, &right_byte);
        if (left_status != right_status ||
            (left_status > 0 && left_byte != right_byte))
            return 0;
    } while (left_status > 0);
    return left_status == 0;
}

static int vx_st_string_ascii(const uint8_t* bytes,
                              const VxStStringSpan* span,
                              const char* ascii, uint32_t ascii_bytes) {
    VxStStringCursor cursor;
    uint32_t index = 0u;
    uint8_t byte = 0u;
    int status;
    if (span->decoded_bytes != ascii_bytes) return 0;
    cursor.bytes = bytes;
    cursor.cursor = span->start;
    cursor.end = span->end;
    cursor.pending_cursor = cursor.pending_bytes = 0u;
    while ((status = vx_st_string_next(&cursor, &byte)) > 0) {
        if (index >= ascii_bytes || byte != (uint8_t)ascii[index]) return 0;
        index++;
    }
    return status == 0 && index == ascii_bytes;
}

static int vx_st_copy_string(const uint8_t* bytes,
                             const VxStStringSpan* span,
                             uint8_t* output) {
    VxStStringCursor cursor;
    uint32_t index = 0u;
    uint8_t byte = 0u;
    int status;
    cursor.bytes = bytes;
    cursor.cursor = span->start;
    cursor.end = span->end;
    cursor.pending_cursor = cursor.pending_bytes = 0u;
    while ((status = vx_st_string_next(&cursor, &byte)) > 0)
        output[index++] = byte;
    return status == 0 && index == span->decoded_bytes;
}

static int vx_st_parse_number(VxStParser* parser, VxStNumberSpan* out) {
    uint32_t start = parser->cursor;
    out->start = out->end = start;
    if (parser->cursor < parser->length &&
        parser->bytes[parser->cursor] == (uint8_t)'-')
        parser->cursor++;
    if (parser->cursor >= parser->length)
        goto invalid;
    if (parser->bytes[parser->cursor] == (uint8_t)'0') {
        parser->cursor++;
        if (parser->cursor < parser->length &&
            parser->bytes[parser->cursor] >= (uint8_t)'0' &&
            parser->bytes[parser->cursor] <= (uint8_t)'9')
            goto invalid;
    } else if (parser->bytes[parser->cursor] >= (uint8_t)'1' &&
               parser->bytes[parser->cursor] <= (uint8_t)'9') {
        do {
            parser->cursor++;
        } while (parser->cursor < parser->length &&
                 parser->bytes[parser->cursor] >= (uint8_t)'0' &&
                 parser->bytes[parser->cursor] <= (uint8_t)'9');
    } else {
        goto invalid;
    }
    if (parser->cursor < parser->length &&
        parser->bytes[parser->cursor] == (uint8_t)'.') {
        parser->cursor++;
        if (parser->cursor >= parser->length ||
            parser->bytes[parser->cursor] < (uint8_t)'0' ||
            parser->bytes[parser->cursor] > (uint8_t)'9')
            goto invalid;
        do {
            parser->cursor++;
        } while (parser->cursor < parser->length &&
                 parser->bytes[parser->cursor] >= (uint8_t)'0' &&
                 parser->bytes[parser->cursor] <= (uint8_t)'9');
    }
    if (parser->cursor < parser->length &&
        (parser->bytes[parser->cursor] == (uint8_t)'e' ||
         parser->bytes[parser->cursor] == (uint8_t)'E')) {
        parser->cursor++;
        if (parser->cursor < parser->length &&
            (parser->bytes[parser->cursor] == (uint8_t)'+' ||
             parser->bytes[parser->cursor] == (uint8_t)'-'))
            parser->cursor++;
        if (parser->cursor >= parser->length ||
            parser->bytes[parser->cursor] < (uint8_t)'0' ||
            parser->bytes[parser->cursor] > (uint8_t)'9')
            goto invalid;
        do {
            parser->cursor++;
        } while (parser->cursor < parser->length &&
                 parser->bytes[parser->cursor] >= (uint8_t)'0' &&
                 parser->bytes[parser->cursor] <= (uint8_t)'9');
    }
    out->start = start;
    out->end = parser->cursor;
    return 1;
invalid:
    return vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                      VX_SAFETENSORS_HEADER_SECTION_JSON, start + 8u);
}

static uint32_t vx_st_number_digit_count(const uint8_t* bytes,
                                         const VxStNumberSpan* span,
                                         uint32_t* fraction_digits,
                                         int64_t* exponent,
                                         uint32_t* first_digit,
                                         uint32_t* negative) {
    uint32_t cursor = span->start;
    uint32_t count = 0u;
    uint32_t fraction = 0u;
    uint32_t after_dot = 0u;
    int64_t exp = 0;
    uint32_t exp_negative = 0u;
    *negative = 0u;
    if (bytes[cursor] == (uint8_t)'-') {
        *negative = 1u;
        cursor++;
    }
    *first_digit = cursor;
    while (cursor < span->end && bytes[cursor] != (uint8_t)'e' &&
           bytes[cursor] != (uint8_t)'E') {
        if (bytes[cursor] == (uint8_t)'.') {
            after_dot = 1u;
        } else {
            count++;
            if (after_dot) fraction++;
        }
        cursor++;
    }
    if (cursor < span->end) {
        cursor++;
        if (cursor < span->end &&
            (bytes[cursor] == (uint8_t)'+' || bytes[cursor] == (uint8_t)'-')) {
            exp_negative = bytes[cursor] == (uint8_t)'-';
            cursor++;
        }
        while (cursor < span->end) {
            uint32_t digit = (uint32_t)(bytes[cursor++] - (uint8_t)'0');
            if (exp < INT64_C(1000000000)) exp = exp * 10 + (int64_t)digit;
            if (exp > INT64_C(1000000000)) exp = INT64_C(1000000000);
        }
    }
    *fraction_digits = fraction;
    *exponent = exp_negative ? -exp : exp;
    return count;
}

static uint8_t vx_st_number_digit_at(const uint8_t* bytes,
                                     const VxStNumberSpan* span,
                                     uint32_t wanted) {
    uint32_t cursor = span->start;
    uint32_t index = 0u;
    if (bytes[cursor] == (uint8_t)'-') cursor++;
    while (cursor < span->end && bytes[cursor] != (uint8_t)'e' &&
           bytes[cursor] != (uint8_t)'E') {
        uint8_t value = bytes[cursor++];
        if (value == (uint8_t)'.') continue;
        if (index++ == wanted) return (uint8_t)(value - (uint8_t)'0');
    }
    return 0u;
}

static int vx_st_number_safe(const uint8_t* bytes,
                             const VxStNumberSpan* span) {
    const uint8_t safe_limit[16] = {
        9, 0, 0, 7, 1, 9, 9, 2, 5, 4, 7, 4, 0, 9, 9, 1
    };
    const uint8_t finite_prefix[17] = {
        1, 7, 9, 7, 6, 9, 3, 1, 3, 4, 8, 6, 2, 3, 1, 5, 7
    };
    uint32_t fraction;
    int64_t exponent;
    uint32_t first_digit;
    uint32_t negative;
    uint32_t digits = vx_st_number_digit_count(bytes, span, &fraction,
                                                &exponent, &first_digit,
                                                &negative);
    uint32_t leading = 0u;
    int64_t scale;
    int64_t order;
    uint32_t index;
    (void)first_digit;
    (void)negative;
    while (leading < digits &&
           vx_st_number_digit_at(bytes, span, leading) == 0u)
        leading++;
    if (leading == digits) return 1;
    scale = exponent - (int64_t)fraction;
    order = (int64_t)(digits - leading) + scale;
    if (order > 309) return 0;
    if (order == 309) {
        int equal = 1;
        for (index = 0u; index < 17u; index++) {
            uint8_t digit = leading + index < digits
                ? vx_st_number_digit_at(bytes, span, leading + index)
                : 0u;
            if (digit < finite_prefix[index]) {
                equal = 0;
                break;
            }
            if (digit > finite_prefix[index]) return 0;
        }
        if (equal) {
            for (index = leading + 17u; index < digits; index++)
                if (vx_st_number_digit_at(bytes, span, index) != 0u)
                    return 0;
        }
    }
    if (order > 16) return 0;
    if (order < 16) return 1;
    for (index = 0u; index < 16u; index++) {
        int64_t coefficient_index = (int64_t)leading + (int64_t)index;
        uint8_t digit = coefficient_index < (int64_t)digits
            ? vx_st_number_digit_at(bytes, span, (uint32_t)coefficient_index)
            : 0u;
        if (digit < safe_limit[index]) return 1;
        if (digit > safe_limit[index]) return 0;
    }
    /* At 2^53-1 the binary64 ULP is one. Exact decimal remainders below .5
     * round back to the safe maximum; .5 is a ties-to-even step to 2^53 and
     * is therefore unsafe, as is every larger remainder. `order == 16`
     * places the next significant decimal digit exactly at the tenths place. */
    if (leading + 16u < digits &&
        vx_st_number_digit_at(bytes, span, leading + 16u) >= 5u)
        return 0;
    return 1;
}

static int vx_st_number_u64(const uint8_t* bytes,
                            const VxStNumberSpan* span, uint64_t maximum,
                            uint64_t* out) {
    uint32_t fraction;
    int64_t exponent;
    uint32_t first_digit;
    uint32_t negative;
    uint32_t digits = vx_st_number_digit_count(bytes, span, &fraction,
                                                &exponent, &first_digit,
                                                &negative);
    int64_t scale = exponent - (int64_t)fraction;
    uint32_t kept = digits;
    uint32_t index;
    uint64_t value = 0u;
    (void)first_digit;
    if (negative) return 0;
    if (scale < 0) {
        uint64_t remove = (uint64_t)(-scale);
        if (remove > digits) {
            for (index = 0u; index < digits; index++)
                if (vx_st_number_digit_at(bytes, span, index) != 0u) return 0;
            *out = 0u;
            return 1;
        }
        kept = digits - (uint32_t)remove;
        for (index = kept; index < digits; index++)
            if (vx_st_number_digit_at(bytes, span, index) != 0u) return 0;
        scale = 0;
    }
    for (index = 0u; index < kept; index++) {
        uint32_t digit = vx_st_number_digit_at(bytes, span, index);
        if (value > (maximum - digit) / 10u) return 0;
        value = value * 10u + digit;
    }
    if (value == 0u) {
        *out = 0u;
        return 1;
    }
    if (scale > 32) return 0;
    while (scale-- > 0) {
        if (value > maximum / 10u) return 0;
        value *= 10u;
    }
    *out = value;
    return 1;
}

static int vx_st_match_literal(VxStParser* parser, const char* value,
                               uint32_t bytes) {
    uint32_t index;
    if (parser->cursor > parser->length ||
        parser->length - parser->cursor < bytes)
        return 0;
    for (index = 0u; index < bytes; index++)
        if (parser->bytes[parser->cursor + index] != (uint8_t)value[index])
            return 0;
    parser->cursor += bytes;
    return 1;
}

static int vx_st_key_records_layout(uint32_t members,
                                    uint32_t* record_bytes) {
    return record_bytes &&
           vx_st_mul_u32(members, VX_ST_KEY_RECORD_BYTES, record_bytes);
}

static int vx_st_plan_value(VxStParser* parser, uint32_t depth,
                            uint32_t* table_peak_bytes);

static int vx_st_plan_store_object(VxStParser* parser,
                                   uint32_t object_index,
                                   uint32_t object_start,
                                   uint32_t members) {
    uint32_t offset;
    uint32_t end;
    if (!parser->object_plan ||
        !vx_st_mul_u32(object_index, VX_ST_OBJECT_PLAN_BYTES, &offset) ||
        !vx_st_add_u32(offset, VX_ST_OBJECT_PLAN_BYTES, &end))
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_INTEGER_OVERFLOW,
                          VX_SAFETENSORS_HEADER_SECTION_JSON,
                          parser->cursor + 8u);
    if (parser->object_plan->counts &&
        end <= parser->object_plan->capacity) {
        vx_st_put_u32(parser->object_plan->counts + offset, object_start);
        vx_st_put_u32(parser->object_plan->counts + offset + 4u, members);
    }
    return 1;
}

static int vx_st_plan_object(VxStParser* parser, uint32_t depth,
                             uint32_t* table_peak_bytes) {
    VxStStringSpan key;
    uint32_t object_index;
    uint32_t object_start = parser->cursor;
    uint32_t members = 0u;
    uint32_t child_peak = 0u;
    uint32_t record_bytes;
    if (!parser->object_plan ||
        parser->object_plan->object_count == UINT32_MAX)
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_INTEGER_OVERFLOW,
                          VX_SAFETENSORS_HEADER_SECTION_JSON,
                          parser->cursor + 8u);
    object_index = parser->object_plan->object_count++;
    parser->cursor++;
    vx_st_skip_space(parser);
    if (parser->cursor < parser->length &&
        parser->bytes[parser->cursor] == (uint8_t)'}') {
        parser->cursor++;
        goto complete;
    }
    for (;;) {
        uint32_t value_peak = 0u;
        if (!vx_st_parse_string(parser, &key)) return 0;
        vx_st_skip_space(parser);
        if (parser->cursor >= parser->length ||
            parser->bytes[parser->cursor++] != (uint8_t)':')
            return vx_st_fail(parser,
                              VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                              VX_SAFETENSORS_HEADER_SECTION_JSON,
                              parser->cursor + 8u);
        if (!vx_st_plan_value(parser, depth + 1u, &value_peak)) return 0;
        if (members == UINT32_MAX)
            return vx_st_fail(
                parser, VX_SAFETENSORS_HEADER_ERROR_INTEGER_OVERFLOW,
                VX_SAFETENSORS_HEADER_SECTION_JSON,
                parser->cursor + 8u);
        members++;
        if (value_peak > child_peak) child_peak = value_peak;
        vx_st_skip_space(parser);
        if (parser->cursor >= parser->length) break;
        if (parser->bytes[parser->cursor] == (uint8_t)',') {
            parser->cursor++;
            vx_st_skip_space(parser);
            continue;
        }
        if (parser->bytes[parser->cursor] == (uint8_t)'}') {
            parser->cursor++;
            goto complete;
        }
        break;
    }
    return vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                      VX_SAFETENSORS_HEADER_SECTION_JSON,
                      parser->cursor + 8u);
complete:
    if (!vx_st_key_records_layout(members, &record_bytes) ||
        !vx_st_add_u32(record_bytes, child_peak, table_peak_bytes))
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_INTEGER_OVERFLOW,
                          VX_SAFETENSORS_HEADER_SECTION_JSON,
                          parser->cursor + 8u);
    return vx_st_plan_store_object(parser, object_index, object_start,
                                   members);
}

static int vx_st_plan_array(VxStParser* parser, uint32_t depth,
                            uint32_t* table_peak_bytes) {
    uint32_t peak = 0u;
    parser->cursor++;
    vx_st_skip_space(parser);
    if (parser->cursor < parser->length &&
        parser->bytes[parser->cursor] == (uint8_t)']') {
        parser->cursor++;
        *table_peak_bytes = 0u;
        return 1;
    }
    for (;;) {
        uint32_t child_peak = 0u;
        if (!vx_st_plan_value(parser, depth + 1u, &child_peak)) return 0;
        if (child_peak > peak) peak = child_peak;
        vx_st_skip_space(parser);
        if (parser->cursor >= parser->length) break;
        if (parser->bytes[parser->cursor] == (uint8_t)',') {
            parser->cursor++;
            vx_st_skip_space(parser);
            continue;
        }
        if (parser->bytes[parser->cursor] == (uint8_t)']') {
            parser->cursor++;
            *table_peak_bytes = peak;
            return 1;
        }
        break;
    }
    return vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                      VX_SAFETENSORS_HEADER_SECTION_JSON,
                      parser->cursor + 8u);
}

static int vx_st_plan_value(VxStParser* parser, uint32_t depth,
                            uint32_t* table_peak_bytes) {
    VxStStringSpan string;
    VxStNumberSpan number;
    if (!table_peak_bytes)
        return vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INTERNAL,
                          VX_SAFETENSORS_HEADER_SECTION_JSON,
                          parser->cursor + 8u);
    *table_peak_bytes = 0u;
    if (depth > VX_ST_JSON_MAX_DEPTH)
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_DEPTH_LIMIT,
                          VX_SAFETENSORS_HEADER_SECTION_JSON,
                          parser->cursor + 8u);
    vx_st_skip_space(parser);
    if (parser->cursor >= parser->length)
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                          VX_SAFETENSORS_HEADER_SECTION_JSON,
                          parser->cursor + 8u);
    switch (parser->bytes[parser->cursor]) {
        case (uint8_t)'{':
            return vx_st_plan_object(parser, depth, table_peak_bytes);
        case (uint8_t)'[':
            return vx_st_plan_array(parser, depth, table_peak_bytes);
        case (uint8_t)'"':
            return vx_st_parse_string(parser, &string);
        case (uint8_t)'t':
            if (vx_st_match_literal(parser, "true", 4u)) return 1;
            break;
        case (uint8_t)'f':
            if (vx_st_match_literal(parser, "false", 5u)) return 1;
            break;
        case (uint8_t)'n':
            if (vx_st_match_literal(parser, "null", 4u)) return 1;
            break;
        default:
            if (vx_st_parse_number(parser, &number)) {
                if (!vx_st_number_safe(parser->bytes, &number))
                    return vx_st_fail(
                        parser,
                        VX_SAFETENSORS_HEADER_ERROR_INTEGER_OVERFLOW,
                        VX_SAFETENSORS_HEADER_SECTION_JSON,
                        number.start + 8u);
                return 1;
            }
            return 0;
    }
    return vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                      VX_SAFETENSORS_HEADER_SECTION_JSON,
                      parser->cursor + 8u);
}

static int vx_st_plan_next_object(VxStParser* parser,
                                  uint32_t object_start,
                                  uint32_t* members) {
    uint32_t offset;
    if (!parser->object_plan || !members ||
        parser->object_plan->cursor >= parser->object_plan->object_count ||
        !vx_st_mul_u32(parser->object_plan->cursor,
                       VX_ST_OBJECT_PLAN_BYTES, &offset) ||
        offset > parser->object_plan->count_bytes ||
        parser->object_plan->count_bytes - offset <
            VX_ST_OBJECT_PLAN_BYTES ||
        vx_st_get_u32(parser->object_plan->counts + offset) != object_start)
        return vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INTERNAL,
                          VX_SAFETENSORS_HEADER_SECTION_JSON,
                          parser->cursor + 8u);
    *members = vx_st_get_u32(parser->object_plan->counts + offset + 4u);
    parser->object_plan->cursor++;
    return 1;
}

static int vx_st_scratch_key_records(VxStParser* parser, uint32_t members,
                                     uint32_t* mark,
                                     uint32_t* records_offset) {
    uint32_t bytes;
    uint32_t aligned;
    uint32_t end;
    *mark = parser->scratch->used;
    if (!vx_st_key_records_layout(members, &bytes) ||
        !vx_st_align_u32(parser->scratch->used, 16u, &aligned) ||
        !vx_st_add_u32(aligned, bytes, &end))
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_INTEGER_OVERFLOW,
                          VX_SAFETENSORS_HEADER_SECTION_JSON,
                          parser->cursor + 8u);
    if (end > parser->scratch->required) parser->scratch->required = end;
    if (end > parser->scratch->capacity) return 0;
    vx_st_zero(parser->scratch->bytes + aligned, bytes);
    parser->scratch->used = end;
    *records_offset = aligned;
    return 1;
}

static void vx_st_scratch_rewind(VxStParser* parser, uint32_t mark) {
    if (parser->scratch->used > mark) {
        vx_st_zero(parser->scratch->bytes + mark,
                   parser->scratch->used - mark);
        parser->scratch->used = mark;
    }
}

static uint8_t* vx_st_key_record(VxStParser* parser,
                                 uint32_t records_offset,
                                 uint32_t member_index) {
    return parser->scratch->bytes + records_offset +
           member_index * VX_ST_KEY_RECORD_BYTES;
}

static int vx_st_key_record_append(VxStParser* parser,
                                   uint32_t records_offset,
                                   uint32_t members,
                                   uint32_t member_index,
                                   const VxStStringSpan* key) {
    uint32_t relative;
    uint32_t offset;
    uint32_t end;
    uint8_t* record;
    if (!key || member_index >= members ||
        !vx_st_mul_u32(member_index, VX_ST_KEY_RECORD_BYTES, &relative) ||
        !vx_st_add_u32(records_offset, relative, &offset) ||
        !vx_st_add_u32(offset, VX_ST_KEY_RECORD_BYTES, &end) ||
        end > parser->scratch->used || end > parser->scratch->capacity)
        return vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INTERNAL,
                          VX_SAFETENSORS_HEADER_SECTION_JSON,
                          key ? key->start + 8u : parser->cursor + 8u);
    record = parser->scratch->bytes + offset;
    vx_st_put_u64(record, key->hash);
    vx_st_put_u32(record + 8u, key->start);
    vx_st_put_u32(record + 12u, key->end);
    vx_st_put_u32(record + 16u, key->decoded_bytes);
    vx_st_put_u32(record + 20u, member_index);
    vx_st_put_u32(record + 24u, 0u);
    vx_st_put_u32(record + 28u, 0u);
    return 1;
}

static void vx_st_key_record_span(const uint8_t* record,
                                  VxStStringSpan* span) {
    span->hash = vx_st_get_u64(record);
    span->start = vx_st_get_u32(record + 8u);
    span->end = vx_st_get_u32(record + 12u);
    span->decoded_bytes = vx_st_get_u32(record + 16u);
}

static int vx_st_string_compare(const uint8_t* bytes,
                                const VxStStringSpan* left,
                                const VxStStringSpan* right) {
    VxStStringCursor a;
    VxStStringCursor b;
    uint8_t left_byte = 0u;
    uint8_t right_byte = 0u;
    int left_status;
    int right_status;
    a.bytes = bytes;
    a.cursor = left->start;
    a.end = left->end;
    a.pending_cursor = a.pending_bytes = 0u;
    b.bytes = bytes;
    b.cursor = right->start;
    b.end = right->end;
    b.pending_cursor = b.pending_bytes = 0u;
    for (;;) {
        left_status = vx_st_string_next(&a, &left_byte);
        right_status = vx_st_string_next(&b, &right_byte);
        if (left_status <= 0 || right_status <= 0) {
            if (left_status == right_status) return 0;
            return left_status < right_status ? -1 : 1;
        }
        if (left_byte != right_byte)
            return left_byte < right_byte ? -1 : 1;
    }
}

static int vx_st_key_record_compare(VxStParser* parser,
                                    const uint8_t* left,
                                    const uint8_t* right) {
    uint64_t left_hash = vx_st_get_u64(left);
    uint64_t right_hash = vx_st_get_u64(right);
    uint32_t left_bytes;
    uint32_t right_bytes;
    uint32_t left_index;
    uint32_t right_index;
    VxStStringSpan left_span;
    VxStStringSpan right_span;
    int decoded_order;
    if (left_hash != right_hash) return left_hash < right_hash ? -1 : 1;
    left_bytes = vx_st_get_u32(left + 16u);
    right_bytes = vx_st_get_u32(right + 16u);
    if (left_bytes != right_bytes) return left_bytes < right_bytes ? -1 : 1;
    vx_st_key_record_span(left, &left_span);
    vx_st_key_record_span(right, &right_span);
    decoded_order = vx_st_string_compare(parser->bytes, &left_span,
                                         &right_span);
    if (decoded_order != 0) return decoded_order;
    left_index = vx_st_get_u32(left + 20u);
    right_index = vx_st_get_u32(right + 20u);
    if (left_index == right_index) return 0;
    return left_index < right_index ? -1 : 1;
}

static void vx_st_key_record_swap(uint8_t* left, uint8_t* right) {
    uint32_t index;
    for (index = 0u; index < VX_ST_KEY_RECORD_BYTES; index++) {
        uint8_t temporary = left[index];
        left[index] = right[index];
        right[index] = temporary;
    }
}

static void vx_st_key_record_sift_down(VxStParser* parser,
                                       uint32_t records_offset,
                                       uint32_t root,
                                       uint32_t count) {
    for (;;) {
        uint32_t child;
        uint32_t selected;
        if (root > (UINT32_MAX - 1u) / 2u) return;
        child = root * 2u + 1u;
        if (child >= count) return;
        selected = child;
        if (child + 1u < count &&
            vx_st_key_record_compare(
                parser,
                vx_st_key_record(parser, records_offset, child),
                vx_st_key_record(parser, records_offset, child + 1u)) < 0)
            selected = child + 1u;
        if (vx_st_key_record_compare(
                parser, vx_st_key_record(parser, records_offset, root),
                vx_st_key_record(parser, records_offset, selected)) >= 0)
            return;
        vx_st_key_record_swap(
            vx_st_key_record(parser, records_offset, root),
            vx_st_key_record(parser, records_offset, selected));
        root = selected;
    }
}

static void vx_st_key_records_sort(VxStParser* parser,
                                   uint32_t records_offset,
                                   uint32_t members) {
    uint32_t root = members / 2u;
    uint32_t end = members;
    while (root > 0u) {
        root--;
        vx_st_key_record_sift_down(parser, records_offset, root, members);
    }
    while (end > 1u) {
        end--;
        vx_st_key_record_swap(
            vx_st_key_record(parser, records_offset, 0u),
            vx_st_key_record(parser, records_offset, end));
        vx_st_key_record_sift_down(parser, records_offset, 0u, end);
    }
}

static int vx_st_key_records_finish(
    VxStParser* parser, uint32_t mark, uint32_t records_offset,
    uint32_t members, uint32_t actual_members,
    VxSafetensorsHeaderErrorSectionV1 section,
    uint32_t member_is_error_index) {
    uint32_t index;
    if (actual_members != members) {
        vx_st_scratch_rewind(parser, mark);
        return vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INTERNAL,
                          VX_SAFETENSORS_HEADER_SECTION_JSON,
                          parser->cursor + 8u);
    }
    vx_st_key_records_sort(parser, records_offset, members);
    for (index = 1u; index < members; index++) {
        uint8_t* previous =
            vx_st_key_record(parser, records_offset, index - 1u);
        uint8_t* current =
            vx_st_key_record(parser, records_offset, index);
        VxStStringSpan previous_span;
        VxStStringSpan current_span;
        vx_st_key_record_span(previous, &previous_span);
        vx_st_key_record_span(current, &current_span);
        if (vx_st_string_equal(parser->bytes, &previous_span,
                               &current_span)) {
            uint32_t duplicate_index = vx_st_get_u32(current + 20u);
            uint32_t duplicate_offset = current_span.start + 8u;
            if (member_is_error_index) {
                parser->error_index = duplicate_index;
                parser->error_index_present = 1u;
            }
            vx_st_scratch_rewind(parser, mark);
            return vx_st_fail(
                parser, VX_SAFETENSORS_HEADER_ERROR_DUPLICATE_KEY,
                section == VX_SAFETENSORS_HEADER_SECTION_NONE
                    ? (VxSafetensorsHeaderErrorSectionV1)
                          VX_SAFETENSORS_HEADER_SECTION_JSON
                    : section,
                duplicate_offset);
        }
    }
    vx_st_scratch_rewind(parser, mark);
    return 1;
}

static int vx_st_builder_string(VxStParser* parser,
                                const VxStStringSpan* span,
                                uint32_t* offset) {
    VxStBuild* build = parser->build;
    uint32_t next;
    *offset = build->string_bytes;
    if (!vx_st_add_u32(build->string_bytes, span->decoded_bytes, &next))
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_INTEGER_OVERFLOW,
                          parser->error_section, span->start + 8u);
    if (build->write) {
        if (build->string_offset > build->response_bytes ||
            next > build->response_bytes - build->string_offset ||
            !vx_st_copy_string(parser->bytes, span,
                               build->response + build->string_offset +
                               build->string_bytes))
            return vx_st_fail(parser,
                              VX_SAFETENSORS_HEADER_ERROR_INTERNAL,
                              parser->error_section, span->start + 8u);
    }
    build->string_bytes = next;
    return 1;
}

static int vx_st_parse_generic(VxStParser* parser, uint32_t depth);

static int vx_st_parse_generic_object(VxStParser* parser, uint32_t depth) {
    uint32_t members = 0u;
    uint32_t mark = 0u;
    uint32_t records_offset = 0u;
    uint32_t member_index = 0u;
    VxSafetensorsHeaderErrorSectionV1 object_section =
        parser->error_section;
    VxStStringSpan key;
    parser->cursor++;
    if (!vx_st_plan_next_object(parser, parser->cursor - 1u, &members) ||
        !vx_st_scratch_key_records(parser, members, &mark, &records_offset))
        return 0;
    vx_st_skip_space(parser);
    if (parser->cursor < parser->length &&
        parser->bytes[parser->cursor] == (uint8_t)'}') {
        parser->cursor++;
        return vx_st_key_records_finish(parser, mark, records_offset,
                                        members, member_index,
                                        object_section, 0u);
    }
    for (;;) {
        if (!vx_st_parse_string(parser, &key) ||
            !vx_st_key_record_append(parser, records_offset, members,
                                     member_index, &key))
            goto fail;
        member_index++;
        vx_st_skip_space(parser);
        if (parser->cursor >= parser->length ||
            parser->bytes[parser->cursor++] != (uint8_t)':')
            goto invalid;
        if (!vx_st_parse_generic(parser, depth + 1u)) goto fail;
        vx_st_skip_space(parser);
        if (parser->cursor >= parser->length) goto invalid;
        if (parser->bytes[parser->cursor] == (uint8_t)',') {
            parser->cursor++;
            vx_st_skip_space(parser);
            continue;
        }
        if (parser->bytes[parser->cursor] == (uint8_t)'}') {
            parser->cursor++;
            return vx_st_key_records_finish(parser, mark, records_offset,
                                            members, member_index,
                                            object_section, 0u);
        }
        goto invalid;
    }
invalid:
    vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
               VX_SAFETENSORS_HEADER_SECTION_JSON, parser->cursor + 8u);
fail:
    vx_st_scratch_rewind(parser, mark);
    return 0;
}

static int vx_st_parse_generic_array(VxStParser* parser, uint32_t depth) {
    parser->cursor++;
    vx_st_skip_space(parser);
    if (parser->cursor < parser->length &&
        parser->bytes[parser->cursor] == (uint8_t)']') {
        parser->cursor++;
        return 1;
    }
    for (;;) {
        if (!vx_st_parse_generic(parser, depth + 1u)) return 0;
        vx_st_skip_space(parser);
        if (parser->cursor >= parser->length) break;
        if (parser->bytes[parser->cursor] == (uint8_t)',') {
            parser->cursor++;
            vx_st_skip_space(parser);
            continue;
        }
        if (parser->bytes[parser->cursor] == (uint8_t)']') {
            parser->cursor++;
            return 1;
        }
        break;
    }
    return vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                      VX_SAFETENSORS_HEADER_SECTION_JSON,
                      parser->cursor + 8u);
}

static int vx_st_parse_generic(VxStParser* parser, uint32_t depth) {
    VxStStringSpan string;
    VxStNumberSpan number;
    if (depth > VX_ST_JSON_MAX_DEPTH)
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_DEPTH_LIMIT,
                          VX_SAFETENSORS_HEADER_SECTION_JSON,
                          parser->cursor + 8u);
    vx_st_skip_space(parser);
    if (parser->cursor >= parser->length) goto invalid;
    switch (parser->bytes[parser->cursor]) {
        case (uint8_t)'{': return vx_st_parse_generic_object(parser, depth);
        case (uint8_t)'[': return vx_st_parse_generic_array(parser, depth);
        case (uint8_t)'"': return vx_st_parse_string(parser, &string);
        case (uint8_t)'t':
            if (vx_st_match_literal(parser, "true", 4u)) return 1;
            break;
        case (uint8_t)'f':
            if (vx_st_match_literal(parser, "false", 5u)) return 1;
            break;
        case (uint8_t)'n':
            if (vx_st_match_literal(parser, "null", 4u)) return 1;
            break;
        default:
            if (!vx_st_parse_number(parser, &number)) return 0;
            if (!vx_st_number_safe(parser->bytes, &number))
                return vx_st_fail(
                    parser, VX_SAFETENSORS_HEADER_ERROR_INTEGER_OVERFLOW,
                    VX_SAFETENSORS_HEADER_SECTION_JSON,
                    number.start + 8u);
            return 1;
    }
invalid:
    return vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                      VX_SAFETENSORS_HEADER_SECTION_JSON,
                      parser->cursor + 8u);
}

static uint32_t vx_st_dtype(const uint8_t* bytes,
                            const VxStStringSpan* value,
                            uint32_t* bits) {
#define VX_ST_DTYPE(name, id, width) \
    if (vx_st_string_ascii(bytes, value, name, (uint32_t)(sizeof(name) - 1u))) { \
        *bits = width; \
        return id; \
    }
    VX_ST_DTYPE("BOOL", VX_SAFETENSORS_HEADER_DTYPE_BOOL, 8u)
    VX_ST_DTYPE("F4", VX_SAFETENSORS_HEADER_DTYPE_F4, 4u)
    VX_ST_DTYPE("F6_E2M3", VX_SAFETENSORS_HEADER_DTYPE_F6_E2M3, 6u)
    VX_ST_DTYPE("F6_E3M2", VX_SAFETENSORS_HEADER_DTYPE_F6_E3M2, 6u)
    VX_ST_DTYPE("U8", VX_SAFETENSORS_HEADER_DTYPE_U8, 8u)
    VX_ST_DTYPE("I8", VX_SAFETENSORS_HEADER_DTYPE_I8, 8u)
    VX_ST_DTYPE("F8_E5M2", VX_SAFETENSORS_HEADER_DTYPE_F8_E5M2, 8u)
    VX_ST_DTYPE("F8_E4M3", VX_SAFETENSORS_HEADER_DTYPE_F8_E4M3, 8u)
    VX_ST_DTYPE("F8_E8M0", VX_SAFETENSORS_HEADER_DTYPE_F8_E8M0, 8u)
    VX_ST_DTYPE("F8_E4M3FNUZ", VX_SAFETENSORS_HEADER_DTYPE_F8_E4M3FNUZ, 8u)
    VX_ST_DTYPE("F8_E5M2FNUZ", VX_SAFETENSORS_HEADER_DTYPE_F8_E5M2FNUZ, 8u)
    VX_ST_DTYPE("I16", VX_SAFETENSORS_HEADER_DTYPE_I16, 16u)
    VX_ST_DTYPE("U16", VX_SAFETENSORS_HEADER_DTYPE_U16, 16u)
    VX_ST_DTYPE("F16", VX_SAFETENSORS_HEADER_DTYPE_F16, 16u)
    VX_ST_DTYPE("BF16", VX_SAFETENSORS_HEADER_DTYPE_BF16, 16u)
    VX_ST_DTYPE("I32", VX_SAFETENSORS_HEADER_DTYPE_I32, 32u)
    VX_ST_DTYPE("U32", VX_SAFETENSORS_HEADER_DTYPE_U32, 32u)
    VX_ST_DTYPE("F32", VX_SAFETENSORS_HEADER_DTYPE_F32, 32u)
    VX_ST_DTYPE("C64", VX_SAFETENSORS_HEADER_DTYPE_C64, 64u)
    VX_ST_DTYPE("F64", VX_SAFETENSORS_HEADER_DTYPE_F64, 64u)
    VX_ST_DTYPE("I64", VX_SAFETENSORS_HEADER_DTYPE_I64, 64u)
    VX_ST_DTYPE("U64", VX_SAFETENSORS_HEADER_DTYPE_U64, 64u)
#undef VX_ST_DTYPE
    *bits = 0u;
    return 0u;
}

static int vx_st_parse_shape(VxStParser* parser, VxStTensorTemp* tensor) {
    uint64_t elements = 1u;
    uint32_t rank = 0u;
    if (parser->cursor >= parser->length ||
        parser->bytes[parser->cursor] != (uint8_t)'[')
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_INVALID_SHAPE,
                          VX_SAFETENSORS_HEADER_SECTION_TENSOR,
                          parser->cursor + 8u);
    parser->cursor++;
    tensor->shape_first = parser->build->shape_count;
    vx_st_skip_space(parser);
    if (parser->cursor < parser->length &&
        parser->bytes[parser->cursor] == (uint8_t)']') {
        parser->cursor++;
        tensor->rank = 0u;
        tensor->elements = 1u;
        return 1;
    }
    for (;;) {
        VxStNumberSpan number;
        uint64_t dimension;
        uint32_t next;
        if (!vx_st_parse_number(parser, &number)) return 0;
        if (!vx_st_number_u64(parser->bytes, &number,
                              VOLVOXAI_SAFETENSORS_MAX_EXACT_INTEGER,
                              &dimension))
            return vx_st_fail(
                parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_SHAPE,
                VX_SAFETENSORS_HEADER_SECTION_TENSOR,
                number.start + 8u);
        if (elements != 0u &&
            dimension > VOLVOXAI_SAFETENSORS_MAX_EXACT_INTEGER / elements)
            return vx_st_fail(
                parser, VX_SAFETENSORS_HEADER_ERROR_INTEGER_OVERFLOW,
                VX_SAFETENSORS_HEADER_SECTION_TENSOR,
                number.start + 8u);
        elements *= dimension;
        if (!vx_st_add_u32(parser->build->shape_count, 1u, &next))
            return vx_st_fail(
                parser, VX_SAFETENSORS_HEADER_ERROR_INTEGER_OVERFLOW,
                VX_SAFETENSORS_HEADER_SECTION_TENSOR,
                number.start + 8u);
        if (parser->build->write)
            vx_st_put_u64(parser->build->response +
                              parser->build->shape_offset +
                              parser->build->shape_count * 8u,
                          dimension);
        parser->build->shape_count = next;
        rank++;
        vx_st_skip_space(parser);
        if (parser->cursor >= parser->length) break;
        if (parser->bytes[parser->cursor] == (uint8_t)',') {
            parser->cursor++;
            vx_st_skip_space(parser);
            continue;
        }
        if (parser->bytes[parser->cursor] == (uint8_t)']') {
            parser->cursor++;
            tensor->rank = rank;
            tensor->elements = elements;
            return 1;
        }
        break;
    }
    return vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_SHAPE,
                      VX_SAFETENSORS_HEADER_SECTION_TENSOR,
                      parser->cursor + 8u);
}

static int vx_st_parse_offsets(VxStParser* parser, VxStTensorTemp* tensor,
                               uint64_t data_bytes) {
    uint32_t count = 0u;
    if (parser->cursor >= parser->length ||
        parser->bytes[parser->cursor] != (uint8_t)'[')
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_INVALID_OFFSETS,
                          VX_SAFETENSORS_HEADER_SECTION_TENSOR,
                          parser->cursor + 8u);
    parser->cursor++;
    vx_st_skip_space(parser);
    if (parser->cursor < parser->length &&
        parser->bytes[parser->cursor] == (uint8_t)']') {
        parser->cursor++;
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_INVALID_OFFSETS,
                          VX_SAFETENSORS_HEADER_SECTION_TENSOR,
                          parser->cursor + 8u);
    }
    for (;;) {
        if (count < 2u) {
            VxStNumberSpan number;
            uint64_t value;
            if (!vx_st_parse_number(parser, &number)) return 0;
            if (!vx_st_number_u64(parser->bytes, &number,
                                  VOLVOXAI_SAFETENSORS_MAX_EXACT_INTEGER,
                                  &value))
                return vx_st_fail(
                    parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_OFFSETS,
                    VX_SAFETENSORS_HEADER_SECTION_TENSOR,
                    number.start + 8u);
            if (count == 0u) tensor->start = value;
            else tensor->end = value;
        } else if (!vx_st_parse_generic(parser, 1u)) {
            return 0;
        }
        count++;
        vx_st_skip_space(parser);
        if (parser->cursor >= parser->length) break;
        if (parser->bytes[parser->cursor] == (uint8_t)',') {
            parser->cursor++;
            vx_st_skip_space(parser);
            continue;
        }
        if (parser->bytes[parser->cursor] == (uint8_t)']') {
            parser->cursor++;
            if (count < 2u || tensor->end < tensor->start ||
                tensor->end > data_bytes)
                return vx_st_fail(
                    parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_OFFSETS,
                    VX_SAFETENSORS_HEADER_SECTION_TENSOR,
                    parser->cursor + 8u);
            return 1;
        }
        break;
    }
    return vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_OFFSETS,
                      VX_SAFETENSORS_HEADER_SECTION_TENSOR,
                      parser->cursor + 8u);
}

static int vx_st_parse_tensor(VxStParser* parser,
                              const VxStStringSpan* name,
                              uint32_t member_index, uint64_t data_bytes) {
    VxStTensorTemp tensor;
    uint32_t members = 0u;
    uint32_t mark = 0u;
    uint32_t records_offset = 0u;
    uint32_t field_member = 0u;
    uint32_t tensor_index = parser->build->tensor_count;
    VxStStringSpan key;
    uint32_t name_offset;
    tensor.dtype = tensor.dtype_bits = 0u;
    tensor.shape_first = tensor.rank = 0u;
    tensor.elements = 0u;
    tensor.start = tensor.end = tensor.expected_bytes = 0u;
    tensor.has_dtype = tensor.has_shape = tensor.has_offsets = 0u;
    parser->error_section = VX_SAFETENSORS_HEADER_SECTION_TENSOR;
    parser->error_index = tensor_index;
    parser->error_index_present = 1u;
    if (parser->cursor >= parser->length ||
        parser->bytes[parser->cursor] != (uint8_t)'{')
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_INVALID_TENSOR,
                          VX_SAFETENSORS_HEADER_SECTION_TENSOR,
                          parser->cursor + 8u);
    parser->cursor++;
    if (!vx_st_plan_next_object(parser, parser->cursor - 1u, &members) ||
        !vx_st_scratch_key_records(parser, members, &mark, &records_offset))
        return 0;
    vx_st_skip_space(parser);
    if (parser->cursor < parser->length &&
        parser->bytes[parser->cursor] == (uint8_t)'}') {
        parser->cursor++;
        goto missing;
    }
    for (;;) {
        if (!vx_st_parse_string(parser, &key) ||
            !vx_st_key_record_append(parser, records_offset, members,
                                     field_member, &key))
            goto fail;
        field_member++;
        vx_st_skip_space(parser);
        if (parser->cursor >= parser->length ||
            parser->bytes[parser->cursor++] != (uint8_t)':')
            goto invalid;
        vx_st_skip_space(parser);
        if (vx_st_string_ascii(parser->bytes, &key, "dtype", 5u)) {
            VxStStringSpan value;
            if (!vx_st_parse_string(parser, &value)) goto fail;
            tensor.dtype = vx_st_dtype(parser->bytes, &value,
                                       &tensor.dtype_bits);
            if (!tensor.dtype) {
                vx_st_fail(parser,
                           VX_SAFETENSORS_HEADER_ERROR_UNKNOWN_DTYPE,
                           VX_SAFETENSORS_HEADER_SECTION_TENSOR,
                           value.start + 8u);
                goto fail;
            }
            tensor.has_dtype = 1u;
        } else if (vx_st_string_ascii(parser->bytes, &key, "shape", 5u)) {
            if (!vx_st_parse_shape(parser, &tensor)) goto fail;
            tensor.has_shape = 1u;
        } else if (vx_st_string_ascii(parser->bytes, &key,
                                      "data_offsets", 12u)) {
            if (!vx_st_parse_offsets(parser, &tensor, data_bytes)) goto fail;
            tensor.has_offsets = 1u;
        } else if (!vx_st_parse_generic(parser, 2u)) {
            goto fail;
        }
        vx_st_skip_space(parser);
        if (parser->cursor >= parser->length) goto invalid;
        if (parser->bytes[parser->cursor] == (uint8_t)',') {
            parser->cursor++;
            vx_st_skip_space(parser);
            continue;
        }
        if (parser->bytes[parser->cursor] == (uint8_t)'}') {
            parser->cursor++;
            break;
        }
        goto invalid;
    }
    if (!vx_st_key_records_finish(
            parser, mark, records_offset, members, field_member,
            VX_SAFETENSORS_HEADER_SECTION_TENSOR, 0u))
        return 0;
    if (!tensor.has_dtype || !tensor.has_shape || !tensor.has_offsets)
        goto missing;
    if (tensor.elements != 0u &&
        tensor.dtype_bits >
            VOLVOXAI_SAFETENSORS_MAX_EXACT_INTEGER / tensor.elements) {
        vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INTEGER_OVERFLOW,
                   VX_SAFETENSORS_HEADER_SECTION_TENSOR, name->start + 8u);
        goto fail;
    }
    tensor.expected_bytes = tensor.elements * tensor.dtype_bits;
    if (tensor.expected_bytes % 8u != 0u) {
        vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_NON_BYTE_ALIGNED,
                   VX_SAFETENSORS_HEADER_SECTION_TENSOR, name->start + 8u);
        goto fail;
    }
    tensor.expected_bytes /= 8u;
    if (tensor.end - tensor.start != tensor.expected_bytes) {
        vx_st_fail(parser,
                   VX_SAFETENSORS_HEADER_ERROR_BYTE_LENGTH_MISMATCH,
                   VX_SAFETENSORS_HEADER_SECTION_TENSOR, name->start + 8u);
        goto fail;
    }
    if (UINT64_MAX - parser->build->payload_sum < tensor.expected_bytes) {
        vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INTEGER_OVERFLOW,
                   VX_SAFETENSORS_HEADER_SECTION_TENSOR, name->start + 8u);
        goto fail;
    }
    parser->build->payload_sum += tensor.expected_bytes;
    if (!vx_st_builder_string(parser, name, &name_offset)) goto fail;
    if (parser->build->write) {
        uint8_t* record = parser->build->response +
                          parser->build->tensor_offset +
                          tensor_index *
                              VOLVOXAI_SAFETENSORS_HEADER_TENSOR_BYTES;
        vx_st_put_u32(record, name_offset);
        vx_st_put_u32(record + 4u, name->decoded_bytes);
        vx_st_put_u32(record + 8u, tensor.dtype);
        vx_st_put_u32(record + 12u, tensor.rank);
        vx_st_put_u32(record + 16u, tensor.shape_first);
        vx_st_put_u32(record + 20u, member_index);
        vx_st_put_u64(record + 24u, tensor.start);
        vx_st_put_u64(record + 32u, tensor.end);
        vx_st_put_u64(record + 40u, tensor.expected_bytes);
        vx_st_put_u32(record + 48u, 0u);
        vx_st_put_u32(record + 52u, 0u);
    }
    parser->build->tensor_count++;
    return 1;
missing:
    vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_TENSOR,
               VX_SAFETENSORS_HEADER_SECTION_TENSOR, name->start + 8u);
    goto fail;
invalid:
    vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
               VX_SAFETENSORS_HEADER_SECTION_TENSOR,
               parser->cursor + 8u);
fail:
    vx_st_scratch_rewind(parser, mark);
    return 0;
}

static int vx_st_parse_metadata(VxStParser* parser) {
    uint32_t members = 0u;
    uint32_t mark = 0u;
    uint32_t records_offset = 0u;
    uint32_t member_index = 0u;
    parser->error_section = VX_SAFETENSORS_HEADER_SECTION_METADATA;
    if (parser->cursor >= parser->length ||
        parser->bytes[parser->cursor] != (uint8_t)'{')
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_INVALID_METADATA,
                          VX_SAFETENSORS_HEADER_SECTION_METADATA,
                          parser->cursor + 8u);
    parser->cursor++;
    if (!vx_st_plan_next_object(parser, parser->cursor - 1u, &members) ||
        !vx_st_scratch_key_records(parser, members, &mark, &records_offset))
        return 0;
    vx_st_skip_space(parser);
    if (parser->cursor < parser->length &&
        parser->bytes[parser->cursor] == (uint8_t)'}') {
        parser->cursor++;
        return vx_st_key_records_finish(
            parser, mark, records_offset, members, member_index,
            VX_SAFETENSORS_HEADER_SECTION_METADATA, 1u);
    }
    for (;;) {
        VxStStringSpan key;
        VxStStringSpan value;
        uint32_t key_offset;
        uint32_t value_offset;
        uint32_t metadata_index = parser->build->metadata_count;
        parser->error_index = metadata_index;
        parser->error_index_present = 1u;
        if (!vx_st_parse_string(parser, &key) ||
            !vx_st_key_record_append(parser, records_offset, members,
                                     member_index, &key))
            goto fail;
        vx_st_skip_space(parser);
        if (parser->cursor >= parser->length ||
            parser->bytes[parser->cursor++] != (uint8_t)':')
            goto invalid;
        vx_st_skip_space(parser);
        if (!vx_st_parse_string(parser, &value)) {
            if (parser->error_code == VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON)
                parser->error_code =
                    VX_SAFETENSORS_HEADER_ERROR_INVALID_METADATA;
            goto fail;
        }
        if (!vx_st_builder_string(parser, &key, &key_offset) ||
            !vx_st_builder_string(parser, &value, &value_offset))
            goto fail;
        if (parser->build->write) {
            uint8_t* record = parser->build->response +
                              parser->build->metadata_offset +
                              metadata_index *
                                  VOLVOXAI_SAFETENSORS_HEADER_METADATA_BYTES;
            vx_st_put_u32(record, key_offset);
            vx_st_put_u32(record + 4u, key.decoded_bytes);
            vx_st_put_u32(record + 8u, value_offset);
            vx_st_put_u32(record + 12u, value.decoded_bytes);
            vx_st_put_u32(record + 16u, member_index);
            vx_st_put_u32(record + 20u, 0u);
        }
        parser->build->metadata_count++;
        member_index++;
        vx_st_skip_space(parser);
        if (parser->cursor >= parser->length) goto invalid;
        if (parser->bytes[parser->cursor] == (uint8_t)',') {
            parser->cursor++;
            vx_st_skip_space(parser);
            continue;
        }
        if (parser->bytes[parser->cursor] == (uint8_t)'}') {
            parser->cursor++;
            return vx_st_key_records_finish(
                parser, mark, records_offset, members, member_index,
                VX_SAFETENSORS_HEADER_SECTION_METADATA, 1u);
        }
        goto invalid;
    }
invalid:
    vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_METADATA,
               VX_SAFETENSORS_HEADER_SECTION_METADATA,
               parser->cursor + 8u);
fail:
    vx_st_scratch_rewind(parser, mark);
    return 0;
}

static int vx_st_parse_root(VxStParser* parser, uint64_t data_bytes) {
    uint32_t members = 0u;
    uint32_t mark = 0u;
    uint32_t records_offset = 0u;
    uint32_t root_member = 0u;
    parser->error_section = VX_SAFETENSORS_HEADER_SECTION_ROOT;
    vx_st_skip_space(parser);
    if (parser->cursor >= parser->length ||
        parser->bytes[parser->cursor] != (uint8_t)'{')
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_ROOT_NOT_OBJECT,
                          VX_SAFETENSORS_HEADER_SECTION_ROOT,
                          parser->cursor + 8u);
    parser->cursor++;
    if (!vx_st_plan_next_object(parser, parser->cursor - 1u, &members) ||
        !vx_st_scratch_key_records(parser, members, &mark, &records_offset))
        return 0;
    vx_st_skip_space(parser);
    if (parser->cursor < parser->length &&
        parser->bytes[parser->cursor] == (uint8_t)'}') {
        parser->cursor++;
        if (!vx_st_key_records_finish(
                parser, mark, records_offset, members, root_member,
                VX_SAFETENSORS_HEADER_SECTION_ROOT, 1u))
            return 0;
        vx_st_skip_space(parser);
        if (parser->cursor != parser->length)
            return vx_st_fail(
                parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                VX_SAFETENSORS_HEADER_SECTION_JSON,
                parser->cursor + 8u);
        return 1;
    }
    for (;;) {
        VxStStringSpan key;
        parser->error_section = VX_SAFETENSORS_HEADER_SECTION_ROOT;
        parser->error_index = root_member;
        parser->error_index_present = 1u;
        parser->error_subindex = 0u;
        if (!vx_st_parse_string(parser, &key) ||
            !vx_st_key_record_append(parser, records_offset, members,
                                     root_member, &key))
            goto fail;
        vx_st_skip_space(parser);
        if (parser->cursor >= parser->length ||
            parser->bytes[parser->cursor++] != (uint8_t)':')
            goto invalid;
        vx_st_skip_space(parser);
        if (vx_st_string_ascii(parser->bytes, &key, "__metadata__", 12u)) {
            parser->build->has_metadata = 1u;
            if (!vx_st_parse_metadata(parser)) goto fail;
        } else if (!vx_st_parse_tensor(parser, &key, root_member,
                                       data_bytes)) {
            goto fail;
        }
        root_member++;
        vx_st_skip_space(parser);
        if (parser->cursor >= parser->length) goto invalid;
        if (parser->bytes[parser->cursor] == (uint8_t)',') {
            parser->cursor++;
            vx_st_skip_space(parser);
            continue;
        }
        if (parser->bytes[parser->cursor] == (uint8_t)'}') {
            parser->cursor++;
            if (!vx_st_key_records_finish(
                    parser, mark, records_offset, members, root_member,
                    VX_SAFETENSORS_HEADER_SECTION_ROOT, 1u))
                return 0;
            vx_st_skip_space(parser);
            if (parser->cursor != parser->length)
                return vx_st_fail(
                    parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                    VX_SAFETENSORS_HEADER_SECTION_JSON,
                    parser->cursor + 8u);
            return 1;
        }
        goto invalid;
    }
invalid:
    vx_st_fail(parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
               VX_SAFETENSORS_HEADER_SECTION_ROOT,
               parser->cursor + 8u);
fail:
    vx_st_scratch_rewind(parser, mark);
    return 0;
}

static int vx_st_layout(VxStBuild* build, uint32_t* required) {
    uint32_t tensor_bytes;
    uint32_t metadata_bytes;
    uint32_t shape_bytes;
    uint32_t cursor = VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_BYTES;
    if (!vx_st_mul_u32(build->tensor_count,
                       VOLVOXAI_SAFETENSORS_HEADER_TENSOR_BYTES,
                       &tensor_bytes) ||
        !vx_st_mul_u32(build->metadata_count,
                       VOLVOXAI_SAFETENSORS_HEADER_METADATA_BYTES,
                       &metadata_bytes) ||
        !vx_st_mul_u32(build->shape_count, 8u, &shape_bytes) ||
        !vx_st_align_u32(cursor, 8u, &cursor))
        return 0;
    build->tensor_offset = cursor;
    if (!vx_st_add_u32(cursor, tensor_bytes, &cursor) ||
        !vx_st_align_u32(cursor, 8u, &cursor))
        return 0;
    build->metadata_offset = cursor;
    if (!vx_st_add_u32(cursor, metadata_bytes, &cursor) ||
        !vx_st_align_u32(cursor, 8u, &cursor))
        return 0;
    build->shape_offset = cursor;
    if (!vx_st_add_u32(cursor, shape_bytes, &cursor) ||
        !vx_st_align_u32(cursor, 8u, &cursor))
        return 0;
    build->string_offset = cursor;
    if (!vx_st_add_u32(cursor, build->string_bytes, &cursor)) return 0;
    *required = cursor;
    return 1;
}

static const uint8_t* vx_st_tensor_record(const VxStBuild* build,
                                          uint32_t index) {
    return build->response + build->tensor_offset +
           index * VOLVOXAI_SAFETENSORS_HEADER_TENSOR_BYTES;
}

static int vx_st_coverage_index_compare(const VxStBuild* build,
                                        uint32_t left, uint32_t right) {
    const uint8_t* a = vx_st_tensor_record(build, left);
    const uint8_t* b = vx_st_tensor_record(build, right);
    uint64_t a_start = vx_st_get_u64(a + 24u);
    uint64_t b_start = vx_st_get_u64(b + 24u);
    uint64_t a_end;
    uint64_t b_end;
    if (a_start < b_start) return -1;
    if (a_start > b_start) return 1;
    a_end = vx_st_get_u64(a + 32u);
    b_end = vx_st_get_u64(b + 32u);
    if (a_end < b_end) return -1;
    if (a_end > b_end) return 1;
    return left < right ? -1 : left > right ? 1 : 0;
}

static void vx_st_coverage_sift_down(const VxStBuild* build,
                                     uint8_t* indices, uint32_t root,
                                     uint32_t count) {
    while (count >= 2u && root <= (count - 2u) / 2u) {
        uint32_t child = root * 2u + 1u;
        uint32_t root_index = vx_st_get_u32(indices + root * 4u);
        uint32_t child_index = vx_st_get_u32(indices + child * 4u);
        if (child + 1u < count) {
            uint32_t sibling = vx_st_get_u32(indices + (child + 1u) * 4u);
            if (vx_st_coverage_index_compare(build, child_index, sibling) < 0) {
                child++;
                child_index = sibling;
            }
        }
        if (vx_st_coverage_index_compare(build, root_index, child_index) >= 0)
            return;
        vx_st_put_u32(indices + root * 4u, child_index);
        vx_st_put_u32(indices + child * 4u, root_index);
        root = child;
    }
}

static int vx_st_validate_coverage(VxStParser* parser, uint64_t data_bytes) {
    VxStBuild* build = parser->build;
    uint32_t mark = parser->scratch->used;
    uint32_t aligned;
    uint32_t index_bytes;
    uint32_t end;
    uint8_t* indices;
    uint32_t index;
    uint64_t cursor = 0u;
    if (build->payload_sum != data_bytes)
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_INVALID_COVERAGE,
                          VX_SAFETENSORS_HEADER_SECTION_COVERAGE, 0u);
    if (!vx_st_align_u32(mark, 16u, &aligned) ||
        !vx_st_mul_u32(build->tensor_count, 4u, &index_bytes) ||
        !vx_st_add_u32(aligned, index_bytes, &end))
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_INTEGER_OVERFLOW,
                          VX_SAFETENSORS_HEADER_SECTION_COVERAGE, 0u);
    if (end > parser->scratch->required) parser->scratch->required = end;
    if (end > parser->scratch->capacity) return 0;
    parser->scratch->used = end;
    indices = parser->scratch->bytes + aligned;
    for (index = 0u; index < build->tensor_count; index++)
        vx_st_put_u32(indices + index * 4u, index);
    if (build->tensor_count >= 2u) {
        uint32_t start;
        uint32_t count;
        for (start = build->tensor_count / 2u; start > 0u; start--)
            vx_st_coverage_sift_down(build, indices, start - 1u,
                                     build->tensor_count);
        for (count = build->tensor_count; count > 1u; count--) {
            uint32_t first = vx_st_get_u32(indices);
            uint32_t last = vx_st_get_u32(indices + (count - 1u) * 4u);
            vx_st_put_u32(indices, last);
            vx_st_put_u32(indices + (count - 1u) * 4u, first);
            vx_st_coverage_sift_down(build, indices, 0u, count - 1u);
        }
    }
    for (index = 0u; index < build->tensor_count; index++) {
        uint32_t tensor_index = vx_st_get_u32(indices + index * 4u);
        const uint8_t* tensor = vx_st_tensor_record(build, tensor_index);
        uint64_t start = vx_st_get_u64(tensor + 24u);
        uint64_t finish = vx_st_get_u64(tensor + 32u);
        if (start > finish || finish > data_bytes) {
            parser->error_index = tensor_index;
            parser->error_index_present = 1u;
            vx_st_scratch_rewind(parser, mark);
            return vx_st_fail(
                parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_COVERAGE,
                VX_SAFETENSORS_HEADER_SECTION_COVERAGE, 0u);
        }
        if (start != cursor) {
            parser->error_index = tensor_index;
            parser->error_index_present = 1u;
            vx_st_scratch_rewind(parser, mark);
            return vx_st_fail(
                parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_COVERAGE,
                VX_SAFETENSORS_HEADER_SECTION_COVERAGE, 0u);
        }
        /* Empty tensors consume no byte but, like the legacy TS walk, must
         * occur at the current exact coverage boundary. Sorting by end places
         * [x,x] before a nonempty [x,y], so both remain representable. */
        if (start != finish) cursor = finish;
    }
    vx_st_scratch_rewind(parser, mark);
    if (cursor != data_bytes)
        return vx_st_fail(parser,
                          VX_SAFETENSORS_HEADER_ERROR_INVALID_COVERAGE,
                          VX_SAFETENSORS_HEADER_SECTION_COVERAGE, 0u);
    return 1;
}

static void vx_st_response_header(uint8_t* response,
                                  VxSafetensorsHeaderStatusV1 status,
                                  const VxStParser* parser,
                                  const VxStBuild* build,
                                  uint32_t written,
                                  uint32_t required_response,
                                  uint32_t required_scratch,
                                  uint32_t header_bytes,
                                  uint64_t data_base,
                                  uint64_t data_bytes,
                                  uint64_t file_bytes) {
    const int success = status == VX_SAFETENSORS_HEADER_STATUS_OK;
    const int has_error = !success && parser &&
        parser->error_code != VX_SAFETENSORS_HEADER_ERROR_NONE;
    vx_st_zero(response, VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_BYTES);
    vx_st_put_u32(response, VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_MAGIC);
    vx_st_put_u32(response + 4u,
                  VOLVOXAI_SAFETENSORS_HEADER_ABI_VERSION);
    vx_st_put_i32(response + 8u, status);
    vx_st_put_i32(response + 12u,
                  has_error ? parser->error_code
                            : VX_SAFETENSORS_HEADER_ERROR_NONE);
    vx_st_put_u32(response + 16u,
                  has_error ? parser->error_section
                            : VX_SAFETENSORS_HEADER_SECTION_NONE);
    vx_st_put_u32(response + 20u,
                  has_error ? parser->error_index : 0u);
    vx_st_put_u32(response + 24u,
                  has_error ? parser->error_subindex : 0u);
    vx_st_put_u32(response + 28u,
                  has_error ? parser->error_byte_offset : 0u);
    vx_st_put_u32(response + 32u, written);
    vx_st_put_u32(response + 36u, required_response);
    vx_st_put_u32(response + 40u, required_scratch);
    vx_st_put_u32(
        response + 44u,
        (build && build->has_metadata
             ? VX_SAFETENSORS_HEADER_FLAG_HAS_METADATA : 0u) |
        (has_error && parser->error_index_present
             ? VX_SAFETENSORS_HEADER_FLAG_ERROR_INDEX_PRESENT : 0u));
    if (build) {
        vx_st_put_u32(response + 48u, build->tensor_offset);
        vx_st_put_u32(response + 52u, build->tensor_count);
        vx_st_put_u32(response + 56u, build->metadata_offset);
        vx_st_put_u32(response + 60u, build->metadata_count);
        vx_st_put_u32(response + 64u, build->shape_offset);
        vx_st_put_u32(response + 68u, build->shape_count);
        vx_st_put_u32(response + 72u, build->string_offset);
        vx_st_put_u32(response + 76u, build->string_bytes);
    }
    vx_st_put_u32(response + 80u, header_bytes);
    vx_st_put_u64(response + 88u, data_base);
    vx_st_put_u64(response + 96u, data_bytes);
    vx_st_put_u64(response + 104u, file_bytes);
}

static void vx_st_copy_build(VxStBuild* destination,
                             const VxStBuild* source) {
    destination->response = source->response;
    destination->response_bytes = source->response_bytes;
    destination->write = source->write;
    destination->tensor_offset = source->tensor_offset;
    destination->metadata_offset = source->metadata_offset;
    destination->shape_offset = source->shape_offset;
    destination->string_offset = source->string_offset;
    destination->tensor_count = source->tensor_count;
    destination->metadata_count = source->metadata_count;
    destination->shape_count = source->shape_count;
    destination->string_bytes = source->string_bytes;
    destination->payload_sum = source->payload_sum;
    destination->has_metadata = source->has_metadata;
}

VX_SAFETENSORS_HEADER_API uint32_t vx_safetensors_header_abi_version(void) {
    return VOLVOXAI_SAFETENSORS_HEADER_ABI_VERSION;
}

VX_SAFETENSORS_HEADER_API int32_t vx_safetensors_header_parse_v1(
                                       const uint8_t* header,
                                       uint32_t header_bytes,
                                       uint64_t file_bytes,
                                       uint64_t data_bytes,
                                       uint8_t* response,
                                       uint32_t response_bytes,
                                       uint8_t* scratch,
                                       uint32_t scratch_bytes) {
    VxStParser parser;
    VxStScratch scratch_state;
    VxStObjectPlan object_plan;
    VxStBuild count_build;
    VxStBuild write_build;
    uint64_t declared_json_bytes;
    uint64_t data_base;
    uint32_t json_start = 0u;
    uint32_t plan_table_peak = 0u;
    uint32_t plan_count_bytes = 0u;
    uint32_t plan_table_base = 0u;
    uint32_t plan_required = 0u;
    uint32_t required_response = VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_BYTES;
    int32_t status = VX_SAFETENSORS_HEADER_STATUS_INVALID_HEADER;
    if (!header || !response ||
        response_bytes < VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_BYTES ||
        ((uintptr_t)response & 7u) != 0u ||
        (scratch_bytes != 0u &&
         (!scratch || ((uintptr_t)scratch & 15u) != 0u)) ||
        (scratch_bytes == 0u && scratch != NULL) ||
        vx_st_ranges_overlap(header, header_bytes, response,
                             response_bytes) ||
        vx_st_ranges_overlap(header, header_bytes, scratch, scratch_bytes) ||
        vx_st_ranges_overlap(response, response_bytes, scratch,
                             scratch_bytes))
        return VX_SAFETENSORS_HEADER_STATUS_INVALID_ARGUMENT;
    vx_st_zero(response, VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_BYTES);
    if (scratch_bytes) vx_st_zero(scratch, scratch_bytes);
    parser.bytes = header_bytes >= 8u ? header + 8u : header;
    parser.length = header_bytes >= 8u ? header_bytes - 8u : 0u;
    parser.cursor = 0u;
    parser.scratch = &scratch_state;
    parser.build = &count_build;
    parser.object_plan = &object_plan;
    parser.error_code = VX_SAFETENSORS_HEADER_ERROR_NONE;
    parser.error_section = VX_SAFETENSORS_HEADER_SECTION_NONE;
    parser.error_index = parser.error_subindex = parser.error_byte_offset = 0u;
    parser.error_index_present = 0u;
    scratch_state.bytes = scratch;
    scratch_state.capacity = scratch_bytes;
    scratch_state.used = 0u;
    scratch_state.required = 0u;
    object_plan.counts = scratch;
    object_plan.capacity = scratch_bytes;
    object_plan.object_count = 0u;
    object_plan.cursor = 0u;
    object_plan.count_bytes = 0u;
    count_build.response = response;
    count_build.response_bytes = response_bytes;
    count_build.write = 0u;
    count_build.tensor_offset = count_build.metadata_offset = 0u;
    count_build.shape_offset = count_build.string_offset = 0u;
    count_build.tensor_count = count_build.metadata_count = 0u;
    count_build.shape_count = count_build.string_bytes = 0u;
    count_build.payload_sum = 0u;
    count_build.has_metadata = 0u;
    if (header_bytes < 8u) {
        vx_st_fail(&parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_PREFIX,
                   VX_SAFETENSORS_HEADER_SECTION_PREFIX, 0u);
        goto done;
    }
    declared_json_bytes = vx_st_get_u64(header);
    data_base = (uint64_t)header_bytes;
    if (declared_json_bytes != (uint64_t)(header_bytes - 8u)) {
        vx_st_fail(&parser, VX_SAFETENSORS_HEADER_ERROR_LENGTH_MISMATCH,
                   VX_SAFETENSORS_HEADER_SECTION_PREFIX, 0u);
        goto done;
    }
    if (file_bytes < data_base || file_bytes - data_base != data_bytes ||
        data_base > UINT64_MAX - data_bytes ||
        data_base + data_bytes != file_bytes) {
        vx_st_fail(&parser, VX_SAFETENSORS_HEADER_ERROR_LENGTH_MISMATCH,
                   VX_SAFETENSORS_HEADER_SECTION_PREFIX, 0u);
        goto done;
    }
    if (declared_json_bytes == 0u) {
        vx_st_fail(&parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                   VX_SAFETENSORS_HEADER_SECTION_JSON, 8u);
        goto done;
    }
    if (!vx_st_validate_utf8(&parser)) goto done;
    /* TextDecoder, which owned the legacy browser path, removes one leading
     * UTF-8 BOM. Preserve that valid-file behavior without treating a repeated
     * or interior BOM as JSON whitespace. */
    if (parser.length >= 3u && parser.bytes[0] == 0xefu &&
        parser.bytes[1] == 0xbbu && parser.bytes[2] == 0xbfu)
        json_start = 3u;
    parser.cursor = json_start;
    if (!vx_st_plan_value(&parser, 0u, &plan_table_peak)) goto done;
    vx_st_skip_space(&parser);
    if (parser.cursor != parser.length) {
        vx_st_fail(&parser, VX_SAFETENSORS_HEADER_ERROR_INVALID_JSON,
                   VX_SAFETENSORS_HEADER_SECTION_JSON,
                   parser.cursor + 8u);
        goto done;
    }
    if (!vx_st_mul_u32(object_plan.object_count,
                       VX_ST_OBJECT_PLAN_BYTES, &plan_count_bytes) ||
        !vx_st_align_u32(plan_count_bytes, 16u, &plan_table_base) ||
        !vx_st_add_u32(plan_table_base, plan_table_peak,
                       &plan_required)) {
        vx_st_fail(&parser, VX_SAFETENSORS_HEADER_ERROR_INTEGER_OVERFLOW,
                   VX_SAFETENSORS_HEADER_SECTION_JSON, 0u);
        goto done;
    }
    object_plan.count_bytes = plan_count_bytes;
    scratch_state.required = plan_required;
    if (plan_required > scratch_bytes) {
        status = VX_SAFETENSORS_HEADER_STATUS_SCRATCH_TOO_SMALL;
        goto done;
    }
    object_plan.cursor = 0u;
    scratch_state.used = plan_table_base;
    parser.cursor = json_start;
    if (!vx_st_parse_root(&parser, data_bytes)) {
        if (parser.error_code == VX_SAFETENSORS_HEADER_ERROR_NONE &&
            scratch_state.required > scratch_bytes) {
            status = VX_SAFETENSORS_HEADER_STATUS_SCRATCH_TOO_SMALL;
        }
        goto done;
    }
    /* Subsequent layout failures are file-wide, not attributed to the last
     * successfully parsed root member. */
    parser.error_index_present = 0u;
    if (object_plan.cursor != object_plan.object_count ||
        scratch_state.used != plan_table_base) {
        vx_st_fail(&parser, VX_SAFETENSORS_HEADER_ERROR_INTERNAL,
                   VX_SAFETENSORS_HEADER_SECTION_JSON, parser.cursor + 8u);
        status = VX_SAFETENSORS_HEADER_STATUS_INTERNAL;
        goto done;
    }
    if (!vx_st_layout(&count_build, &required_response)) {
        vx_st_fail(&parser, VX_SAFETENSORS_HEADER_ERROR_INTEGER_OVERFLOW,
                   VX_SAFETENSORS_HEADER_SECTION_ROOT, 0u);
        goto done;
    }
    if (response_bytes < required_response) {
        status = VX_SAFETENSORS_HEADER_STATUS_RESPONSE_TOO_SMALL;
        goto done;
    }
    vx_st_copy_build(&write_build, &count_build);
    write_build.response = response;
    write_build.response_bytes = response_bytes;
    write_build.write = 1u;
    write_build.tensor_count = write_build.metadata_count = 0u;
    write_build.shape_count = write_build.string_bytes = 0u;
    write_build.payload_sum = 0u;
    write_build.has_metadata = 0u;
    parser.cursor = json_start;
    parser.build = &write_build;
    parser.error_code = VX_SAFETENSORS_HEADER_ERROR_NONE;
    parser.error_section = VX_SAFETENSORS_HEADER_SECTION_NONE;
    parser.error_index = parser.error_subindex = parser.error_byte_offset = 0u;
    parser.error_index_present = 0u;
    object_plan.cursor = 0u;
    scratch_state.used = plan_table_base;
    vx_st_zero(response + VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_BYTES,
               required_response -
                   VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_BYTES);
    if (!vx_st_parse_root(&parser, data_bytes)) {
        if (parser.error_code == VX_SAFETENSORS_HEADER_ERROR_NONE &&
            scratch_state.required > scratch_bytes)
            status = VX_SAFETENSORS_HEADER_STATUS_SCRATCH_TOO_SMALL;
        goto done_write;
    }
    /* Coverage establishes its own tensor coordinate only on the branches
     * that identify an offending declaration. */
    parser.error_index_present = 0u;
    if (object_plan.cursor != object_plan.object_count ||
        scratch_state.used != plan_table_base) {
        vx_st_fail(&parser, VX_SAFETENSORS_HEADER_ERROR_INTERNAL,
                   VX_SAFETENSORS_HEADER_SECTION_JSON, parser.cursor + 8u);
        status = VX_SAFETENSORS_HEADER_STATUS_INTERNAL;
        goto done_write;
    }
    if (write_build.tensor_count != count_build.tensor_count ||
        write_build.metadata_count != count_build.metadata_count ||
        write_build.shape_count != count_build.shape_count ||
        write_build.string_bytes != count_build.string_bytes ||
        write_build.has_metadata != count_build.has_metadata) {
        vx_st_fail(&parser, VX_SAFETENSORS_HEADER_ERROR_INTERNAL,
                   VX_SAFETENSORS_HEADER_SECTION_ROOT, 0u);
        status = VX_SAFETENSORS_HEADER_STATUS_INTERNAL;
        goto done_write;
    }
    scratch_state.used = 0u;
    if (!vx_st_validate_coverage(&parser, data_bytes)) goto done_write;
    status = VX_SAFETENSORS_HEADER_STATUS_OK;
done_write:
    vx_st_copy_build(&count_build, &write_build);
done:
    if (parser.error_code == VX_SAFETENSORS_HEADER_ERROR_NONE &&
        scratch_state.required > scratch_bytes)
        status = VX_SAFETENSORS_HEADER_STATUS_SCRATCH_TOO_SMALL;
    vx_st_response_header(
        response, status, &parser, &count_build,
        status == VX_SAFETENSORS_HEADER_STATUS_OK ? required_response
                                                  : VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_BYTES,
        required_response, scratch_state.required, header_bytes,
        header_bytes, data_bytes, file_bytes);
    if (scratch_bytes) vx_st_zero(scratch, scratch_bytes);
    return status;
}

#undef VX_ST_FNV_PRIME
#undef VX_ST_FNV_OFFSET
#undef VX_ST_OBJECT_PLAN_BYTES
#undef VX_ST_KEY_RECORD_BYTES
#undef VX_ST_JSON_MAX_DEPTH

#ifdef VX_SAFETENSORS_HEADER_API_LOCAL_EMPTY
#undef VX_SAFETENSORS_HEADER_API_LOCAL_EMPTY
#undef VX_SAFETENSORS_HEADER_API
#endif
