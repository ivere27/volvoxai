// --- Missing Deep Math & Shape Primitives (Batch 4) ---
#include <stddef.h>
#include <stdint.h>
#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif
void shape_f32(float* output, int n0, int n1, int n2, int n3, int ndims) {
    if (ndims > 0) output[0] = (float)n0;
    if (ndims > 1) output[1] = (float)n1;
    if (ndims > 2) output[2] = (float)n2;
    if (ndims > 3) output[3] = (float)n3;
}

void expand_4d_f32(const float* input, float* output, 
                   int in_b, int in_h, int in_w, int in_c,
                   int out_b, int out_h, int out_w, int out_c) {
    for (int ob = 0; ob < out_b; ob++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                for (int oc = 0; oc < out_c; oc++) {
                    int ib = ob % in_b;
                    int ih = oh % in_h;
                    int iw = ow % in_w;
                    int ic = oc % in_c;
                    output[((long)ob * out_h * out_w + (long)oh * out_w + ow) * out_c + oc] =
                        input[((long)ib * in_h * in_w + (long)ih * in_w + iw) * in_c + ic];
                }
            }
        }
    }
}

void pad_2d_f32(const float* input, float* output, float pad_val,
                int b, int in_h, int in_w, int c,
                int pt, int pb, int pl, int pr) {
    int out_h = in_h + pt + pb;
    int out_w = in_w + pl + pr;
    for (int i_b = 0; i_b < b; i_b++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                for (int i_c = 0; i_c < c; i_c++) {
                    float val = pad_val;
                    if (y >= pt && y < pt + in_h && x >= pl && x < pl + in_w) {
                        val = input[((long)i_b * in_h * in_w + (long)(y - pt) * in_w + (x - pl)) * c + i_c];
                    }
                    output[((long)i_b * out_h * out_w + (long)y * out_w + x) * c + i_c] = val;
                }
            }
        }
    }
}

/* Canonical rank-1..8 positive-step Slice. Shape and parameter arrays are
   supplied by the JS WASM planner, which normalizes negative axes/starts. */
static int vx_slice_nd_validate(const uint32_t *input_shape,
        const uint32_t *output_shape, const uint32_t *starts,
        const uint32_t *steps, uint32_t rank, uint32_t output_elements,
        size_t *input_strides, size_t *input_elements) {
    if (!input_shape || !output_shape || !starts || !steps || !input_strides ||
        !input_elements ||
        rank == 0 || rank > 8) return 0;
    size_t input_stride = 1, output_product = 1;
    for (uint32_t reverse = rank; reverse-- > 0;) {
        uint32_t input_size = input_shape[reverse];
        uint32_t output_size = output_shape[reverse];
        uint32_t start = starts[reverse];
        uint32_t step = steps[reverse];
        if (!input_size || !output_size || !step || start >= input_size ||
            (size_t)(output_size - 1u) > (size_t)(input_size - 1u - start) / step ||
            input_stride > (size_t)-1 / input_size ||
            output_product > (size_t)-1 / output_size) return 0;
        input_strides[reverse] = input_stride;
        input_stride *= input_size;
        output_product *= output_size;
    }
    *input_elements = input_stride;
    return output_product == output_elements;
}

/* A Slice that leaves a trailing set of axes whole is a sequence of contiguous
 * blocks.  Copy those blocks directly instead of recovering every output
 * coordinate with rank-many integer divisions.  The disjoint-storage check
 * keeps the exported kernel's existing ordered behavior for unusual callers
 * that overlap input and output; graph arenas use distinct live ranges. */
static int vx_slice_nd_copy_contiguous_blocks(const void *input, void *output,
        const uint32_t *input_shape, const uint32_t *output_shape,
        const uint32_t *starts, const uint32_t *steps, uint32_t rank,
        uint32_t output_elements, const size_t *input_strides,
        size_t input_elements) {
    const uintptr_t input_address = (uintptr_t)input;
    const uintptr_t output_address = (uintptr_t)output;
    size_t contiguous_elements = 1u;
    uint32_t prefix_rank = rank;
    size_t input_bytes;
    size_t output_bytes;
    size_t block_bytes;
    size_t blocks;
    if (input_elements > SIZE_MAX / sizeof(uint32_t) ||
        (size_t)output_elements > SIZE_MAX / sizeof(uint32_t)) return 0;
    input_bytes = input_elements * sizeof(uint32_t);
    output_bytes = (size_t)output_elements * sizeof(uint32_t);
    if (input_bytes > UINTPTR_MAX - input_address ||
        output_bytes > UINTPTR_MAX - output_address ||
        !((input_address + input_bytes <= output_address) ||
          (output_address + output_bytes <= input_address))) return 0;
    while (prefix_rank > 0u) {
        const uint32_t axis = prefix_rank - 1u;
        if (starts[axis] != 0u || steps[axis] != 1u ||
            output_shape[axis] != input_shape[axis]) break;
        if (contiguous_elements > SIZE_MAX / output_shape[axis]) return 0;
        contiguous_elements *= output_shape[axis];
        prefix_rank--;
    }
    if (contiguous_elements < 4u ||
        (size_t)output_elements % contiguous_elements != 0u ||
        contiguous_elements > SIZE_MAX / sizeof(uint32_t)) return 0;
    block_bytes = contiguous_elements * sizeof(uint32_t);
    blocks = (size_t)output_elements / contiguous_elements;
    for (size_t block = 0u; block < blocks; block++) {
        size_t remaining = block;
        size_t input_index = 0u;
        const uint8_t *source;
        uint8_t *destination;
        size_t byte = 0u;
        for (uint32_t reverse = prefix_rank; reverse-- > 0u;) {
            const uint32_t coordinate =
                (uint32_t)(remaining % output_shape[reverse]);
            remaining /= output_shape[reverse];
            input_index +=
                (size_t)(starts[reverse] + coordinate * steps[reverse]) *
                input_strides[reverse];
        }
        source = (const uint8_t *)input + input_index * sizeof(uint32_t);
        destination = (uint8_t *)output + block * block_bytes;
#if defined(__wasm_simd128__)
        for (; block_bytes - byte >= 16u; byte += 16u) {
            wasm_v128_store(destination + byte, wasm_v128_load(source + byte));
        }
#endif
        for (; byte < block_bytes; byte++) destination[byte] = source[byte];
    }
    return 1;
}

int slice_nd_f32(const float *input, float *output, const uint32_t *input_shape,
        const uint32_t *output_shape, const uint32_t *starts, const uint32_t *steps,
        uint32_t rank, uint32_t output_elements) {
    size_t input_strides[8];
    size_t input_elements;
    if (!input || !output || !vx_slice_nd_validate(input_shape, output_shape,
        starts, steps, rank, output_elements, input_strides,
        &input_elements)) return 0;
    if (vx_slice_nd_copy_contiguous_blocks(input, output, input_shape,
        output_shape, starts, steps, rank, output_elements, input_strides,
        input_elements)) return 1;
    for (uint32_t output_index = 0; output_index < output_elements; output_index++) {
        size_t remaining = output_index, input_index = 0;
        for (uint32_t reverse = rank; reverse-- > 0;) {
            uint32_t coordinate = (uint32_t)(remaining % output_shape[reverse]);
            remaining /= output_shape[reverse];
            input_index += (size_t)(starts[reverse] + coordinate * steps[reverse]) *
                input_strides[reverse];
        }
        output[output_index] = input[input_index];
    }
    return 1;
}

WASM_EXPORT("slice_nd_u32")
int slice_nd_u32(const uint32_t *input, uint32_t *output,
        const uint32_t *input_shape, const uint32_t *output_shape,
        const uint32_t *starts, const uint32_t *steps, uint32_t rank,
        uint32_t output_elements) {
    size_t input_strides[8];
    size_t input_elements;
    if (!input || !output || !vx_slice_nd_validate(input_shape, output_shape,
        starts, steps, rank, output_elements, input_strides,
        &input_elements)) return 0;
    if (vx_slice_nd_copy_contiguous_blocks(input, output, input_shape,
        output_shape, starts, steps, rank, output_elements, input_strides,
        input_elements)) return 1;
    for (uint32_t output_index = 0; output_index < output_elements; output_index++) {
        size_t remaining = output_index, input_index = 0;
        for (uint32_t reverse = rank; reverse-- > 0;) {
            uint32_t coordinate = (uint32_t)(remaining % output_shape[reverse]);
            remaining /= output_shape[reverse];
            input_index += (size_t)(starts[reverse] + coordinate * steps[reverse]) *
                input_strides[reverse];
        }
        output[output_index] = input[input_index];
    }
    return 1;
}

static int vx_gather_i32_validate(uint32_t outer, uint32_t axis_size,
        uint32_t inner, uint32_t indices_elements, uint32_t output_elements) {
    if (!outer || !axis_size || !inner || !indices_elements) return 0;
    if ((size_t)outer > (size_t)-1 / indices_elements) return 0;
    size_t product = (size_t)outer * indices_elements;
    if (product > (size_t)-1 / inner) return 0;
    return product * inner == output_elements;
}

/* Canonical ONNX I32 Gather. Negative indices wrap once by the selected axis
   size. Validate every value before the first output write so an invalid
   request fails atomically instead of manufacturing sentinel data. */
int gather_i32_f32(const float *input, const int32_t *indices, float *output,
        uint32_t outer, uint32_t axis_size, uint32_t inner,
        uint32_t indices_elements, uint32_t output_elements) {
    if (!input || !indices || !output || !vx_gather_i32_validate(outer, axis_size,
        inner, indices_elements, output_elements)) return 0;
    for (uint32_t index_position = 0; index_position < indices_elements; index_position++) {
        int64_t selected = indices[index_position];
        if (selected < 0) selected += (int64_t)axis_size;
        if (selected < 0 || selected >= (int64_t)axis_size) return 0;
    }
    for (uint32_t outer_index = 0; outer_index < outer; outer_index++) {
        for (uint32_t index_position = 0; index_position < indices_elements; index_position++) {
            int64_t selected = indices[index_position];
            size_t output_base = ((size_t)outer_index * indices_elements + index_position) * inner;
            if (selected < 0) selected += (int64_t)axis_size;
            size_t input_base = ((size_t)outer_index * axis_size + (size_t)selected) * inner;
            for (uint32_t inner_index = 0; inner_index < inner; inner_index++) {
                output[output_base + inner_index] = input[input_base + inner_index];
            }
        }
    }
    return 1;
}

static int vx_gather_elements_validate(const uint32_t *input_shape,
        const uint32_t *indices_shape, uint32_t rank, uint32_t axis,
        uint32_t elements, size_t *input_strides) {
    if (!input_shape || !indices_shape || !input_strides || rank == 0 || rank > 8 ||
        axis >= rank) return 0;
    size_t input_stride = 1, output_product = 1;
    for (uint32_t reverse = rank; reverse-- > 0;) {
        uint32_t input_size = input_shape[reverse];
        uint32_t index_size = indices_shape[reverse];
        if (!input_size || !index_size ||
            (reverse != axis && index_size > input_size) ||
            input_stride > (size_t)-1 / input_size ||
            output_product > (size_t)-1 / index_size) return 0;
        input_strides[reverse] = input_stride;
        input_stride *= input_size;
        output_product *= index_size;
    }
    return output_product == elements;
}

/* ONNX GatherElements accepts negative axis selections. Invalid values follow
   the portable inference sentinel contract and yield -1. */
int gather_elements_i32_f32(const float *input, const int32_t *indices,
        float *output, const uint32_t *input_shape, const uint32_t *indices_shape,
        uint32_t rank, uint32_t axis, uint32_t elements) {
    size_t input_strides[8];
    if (!input || !indices || !output || !vx_gather_elements_validate(input_shape,
        indices_shape, rank, axis, elements, input_strides)) return 0;
    for (uint32_t output_index = 0; output_index < elements; output_index++) {
        size_t remaining = output_index, input_index = 0;
        int valid = 1;
        for (uint32_t reverse = rank; reverse-- > 0;) {
            uint32_t coordinate = (uint32_t)(remaining % indices_shape[reverse]);
            remaining /= indices_shape[reverse];
            if (reverse == axis) {
                int64_t selected = indices[output_index];
                if (selected < 0) selected += (int64_t)input_shape[reverse];
                if (selected < 0 || selected >= (int64_t)input_shape[reverse]) {
                    valid = 0;
                    break;
                }
                input_index += (size_t)selected * input_strides[reverse];
            } else {
                input_index += (size_t)coordinate * input_strides[reverse];
            }
        }
        output[output_index] = valid ? input[input_index] : -1.0f;
    }
    return 1;
}
