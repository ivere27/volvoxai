/* Where and integer Clip: elementwise selection over typed tensors. */
#include "mathcompat.h"
#include "../../include/volvoxai_enums.h"
#include "w8a8_affine.h"
#if !defined(__wasm__)
#include "quant_cpu_isa.h"
#endif
#include <stdint.h>

static double vx_typed_number(const void *values, int type, int index) {
    if (type == VX_DTYPE_F32) return (double)((const float *)values)[index];
    if (type == VX_DTYPE_I32) return (double)((const int32_t *)values)[index];
    if (type == VX_DTYPE_I8) return (double)((const int8_t *)values)[index];
    return (double)((const uint8_t *)values)[index];
}

static int vx_typed_number_dtype(int dtype) {
    return dtype == VX_DTYPE_F32 || dtype == VX_DTYPE_I32 ||
        dtype == VX_DTYPE_I8 || dtype == VX_DTYPE_U8;
}

void where_f32(const float* cond, const float* a, const float* b, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = cond[i] != 0.0f ? a[i] : b[i];
}

/* Canonical portable Where/Mask: F32 data/output, exact-shape F32 or I32
   condition. Keeping the condition pointer typed avoids reinterpreting an
   Int32.MIN_VALUE bit pattern as an F32 value. */
int where_typed_f32(const void *condition, int condition_dtype,
        const float *a, const float *b, float *output, uint32_t elements) {
    if (!condition || !a || !b || !output ||
        (condition_dtype != VX_DTYPE_F32 &&
         condition_dtype != VX_DTYPE_I32)) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        int selected = condition_dtype == VX_DTYPE_I32
            ? ((const int32_t *)condition)[index] != 0
            : ((const float *)condition)[index] != 0.0f;
        output[index] = selected ? a[index] : b[index];
    }
    return 1;
}

/* Bit-preserving Where for either F32 or I32 branch/output storage. */
WASM_EXPORT("where_typed_32")
int where_typed_32(const void *condition, int condition_dtype,
        const uint32_t *a, const uint32_t *b, uint32_t *output,
        uint32_t elements, uint32_t data_dtype) {
    if (!condition || !a || !b || !output ||
        (condition_dtype != VX_DTYPE_F32 &&
         condition_dtype != VX_DTYPE_I32) ||
        (data_dtype != VX_DTYPE_F32 && data_dtype != VX_DTYPE_I32)) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        int selected = condition_dtype == VX_DTYPE_I32
            ? ((const int32_t *)condition)[index] != 0
            : ((const float *)condition)[index] != 0.0f;
        output[index] = selected ? a[index] : b[index];
    }
    return 1;
}

WASM_EXPORT("clip_i32")
int clip_i32(const int32_t *input, int32_t *output, uint32_t elements,
        int32_t minimum, int32_t maximum) {
    if (!input || !output || !elements || minimum > maximum) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        int32_t value = input[index];
        if (value < minimum) value = minimum;
        if (value > maximum) value = maximum;
        output[index] = value;
    }
    return 1;
}

