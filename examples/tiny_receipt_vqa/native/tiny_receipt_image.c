#include "tiny_receipt_image.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static void tiny_receipt_set_error(char* err, size_t err_size, const char* message) {
    if (err && err_size) snprintf(err, err_size, "%s", message);
}

static float tiny_receipt_normalize(float value) {
    const float unit = value / 255.0f;
    const float doubled = unit * 2.0f;
    return doubled - 1.0f;
}

/* Match the source evaluator's semantic order:
 *
 *   Image.open(...).convert("L").resize(..., BILINEAR)
 *
 * Pillow's RGB-to-L conversion is an 8-bit, rounded 299/587/114 transform.
 * Resize that resulting grayscale plane with a triangle filter whose support
 * expands while downsampling.  The latter avoids treating a downscale as one
 * point sample. */
static unsigned char grayscale_luma_u8(const unsigned char* rgb) {
    return (unsigned char)(((unsigned int)rgb[0] * 299u +
                            (unsigned int)rgb[1] * 587u +
                            (unsigned int)rgb[2] * 114u + 500u) / 1000u);
}

static int resize_axis_bounds(int source_size, int target_size, int destination,
                              int* first, int* last, float* center, float* filter_scale) {
    float scale;
    float support;
    int lower;
    int upper;
    if (source_size <= 0 || target_size <= 0 || destination < 0 || destination >= target_size ||
        !first || !last || !center || !filter_scale) return -1;
    scale = (float)source_size / (float)target_size;
    *filter_scale = scale > 1.0f ? scale : 1.0f;
    support = *filter_scale;
    *center = ((float)destination + 0.5f) * scale;
    lower = (int)ceilf(*center - support - 0.5f);
    upper = (int)floorf(*center + support - 0.5f);
    if (lower < 0) lower = 0;
    if (upper >= source_size) upper = source_size - 1;
    if (lower > upper) {
        int nearest = (int)floorf(*center);
        if (nearest < 0) nearest = 0;
        if (nearest >= source_size) nearest = source_size - 1;
        lower = nearest;
        upper = nearest;
    }
    *first = lower;
    *last = upper;
    return 0;
}

int tiny_receipt_rgb_to_grayscale_bilinear(const unsigned char* rgb, int source_width,
                                            int source_height, float* dst, int target_width,
                                            int target_height) {
    unsigned char* grayscale = NULL;
    float* horizontal = NULL;
    size_t source_pixels;
    size_t horizontal_count;
    if (!rgb || !dst || source_width <= 0 || source_height <= 0 ||
        target_width <= 0 || target_height <= 0 ||
        (size_t)source_width > SIZE_MAX / (size_t)source_height ||
        (size_t)source_height > SIZE_MAX / (size_t)target_width) return -1;
    source_pixels = (size_t)source_width * (size_t)source_height;
    horizontal_count = (size_t)source_height * (size_t)target_width;
    if (source_pixels > SIZE_MAX / 3u || source_pixels > SIZE_MAX / sizeof(*grayscale) ||
        horizontal_count > SIZE_MAX / sizeof(*horizontal)) return -1;
    grayscale = (unsigned char*)malloc(source_pixels * sizeof(*grayscale));
    horizontal = (float*)malloc(horizontal_count * sizeof(*horizontal));
    if (!grayscale || !horizontal) {
        free(horizontal);
        free(grayscale);
        return -1;
    }
    for (size_t index = 0; index < source_pixels; index++) {
        grayscale[index] = grayscale_luma_u8(rgb + index * 3u);
    }
    for (int y = 0; y < source_height; y++) {
        for (int x = 0; x < target_width; x++) {
            int first;
            int last;
            float center;
            float filter_scale;
            float sum = 0.0f;
            float total = 0.0f;
            if (resize_axis_bounds(source_width, target_width, x, &first, &last,
                                   &center, &filter_scale) != 0) goto fail;
            for (int source_x = first; source_x <= last; source_x++) {
                float distance = fabsf(((float)source_x + 0.5f - center) / filter_scale);
                float weight = distance < 1.0f ? 1.0f - distance : 0.0f;
                sum += (float)grayscale[(size_t)y * (size_t)source_width + (size_t)source_x] * weight;
                total += weight;
            }
            if (total <= 0.0f) goto fail;
            horizontal[(size_t)y * (size_t)target_width + (size_t)x] = sum / total;
        }
    }
    for (int y = 0; y < target_height; y++) {
        int first;
        int last;
        float center;
        float filter_scale;
        if (resize_axis_bounds(source_height, target_height, y, &first, &last,
                               &center, &filter_scale) != 0) goto fail;
        for (int x = 0; x < target_width; x++) {
            float sum = 0.0f;
            float total = 0.0f;
            int rounded;
            for (int source_y = first; source_y <= last; source_y++) {
                float distance = fabsf(((float)source_y + 0.5f - center) / filter_scale);
                float weight = distance < 1.0f ? 1.0f - distance : 0.0f;
                sum += horizontal[(size_t)source_y * (size_t)target_width + (size_t)x] * weight;
                total += weight;
            }
            if (total <= 0.0f) goto fail;
            rounded = (int)floorf(sum / total + 0.5f);
            if (rounded < 0) rounded = 0;
            if (rounded > 255) rounded = 255;
            dst[(size_t)y * (size_t)target_width + (size_t)x] =
                tiny_receipt_normalize((float)rounded);
        }
    }
    free(horizontal);
    free(grayscale);
    return 0;

fail:
    free(horizontal);
    free(grayscale);
    return -1;
}

int tiny_receipt_load_image_to_tensor(const char* path, float* dst, const int* shape,
                                      int ndim, char* err, size_t err_size) {
    int width;
    int height;
    int components;
    unsigned char* rgb;
    int result;

    if (!dst || !shape || ndim != 4 || shape[0] != 1 ||
        shape[1] <= 0 || shape[2] <= 0 || shape[3] != 1) {
        tiny_receipt_set_error(err, err_size,
                               "TinyReceipt image input must be [1,H,W,1]");
        return -1;
    }
    rgb = stbi_load(path, &width, &height, &components, 3);
    if (!rgb) {
        const char* reason = stbi_failure_reason();
        tiny_receipt_set_error(err, err_size,
                               reason ? reason : "image decode failed");
        return -1;
    }
    result = tiny_receipt_rgb_to_grayscale_bilinear(
        rgb, width, height, dst, shape[2], shape[1]);
    stbi_image_free(rgb);
    if (result != 0) {
        tiny_receipt_set_error(err, err_size,
                               "could not grayscale/resize TinyReceipt image");
    }
    return result;
}
