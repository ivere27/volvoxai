#include "image_io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

typedef struct {
    int w;
    int h;
    unsigned char* rgb;
} ImageRGB;

static void set_err(char* err, size_t err_size, const char* msg) {
    if (err && err_size) {
        snprintf(err, err_size, "%s", msg);
    }
}

static int load_image_rgb(const char* path, ImageRGB* img, char* err, size_t err_size) {
    memset(img, 0, sizeof(*img));
    int comp = 0;
    img->rgb = stbi_load(path, &img->w, &img->h, &comp, 3);
    if (!img->rgb) {
        const char* reason = stbi_failure_reason();
        set_err(err, err_size, reason ? reason : "image decode failed");
        return -1;
    }
    return 0;
}

static void free_image_rgb(ImageRGB* img) {
    if (img && img->rgb) stbi_image_free(img->rgb);
}

static float sample_channel(const ImageRGB* img, float x, float y, int c) {
    if (x < 0.0f) x = 0.0f;
    if (y < 0.0f) y = 0.0f;
    if (x > (float)(img->w - 1)) x = (float)(img->w - 1);
    if (y > (float)(img->h - 1)) y = (float)(img->h - 1);
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    int x1 = x0 + 1 < img->w ? x0 + 1 : x0;
    int y1 = y0 + 1 < img->h ? y0 + 1 : y0;
    float wx = x - (float)x0;
    float wy = y - (float)y0;
    const unsigned char* p00 = img->rgb + ((size_t)y0 * img->w + x0) * 3;
    const unsigned char* p10 = img->rgb + ((size_t)y0 * img->w + x1) * 3;
    const unsigned char* p01 = img->rgb + ((size_t)y1 * img->w + x0) * 3;
    const unsigned char* p11 = img->rgb + ((size_t)y1 * img->w + x1) * 3;
    float a = (1.0f - wx) * p00[c] + wx * p10[c];
    float b = (1.0f - wx) * p01[c] + wx * p11[c];
    return (1.0f - wy) * a + wy * b;
}

static float normalize_value(float v, int normalize) {
    if (normalize == VOLVOX_IMAGE_RAW_255) return v;
    v /= 255.0f;
    if (normalize == VOLVOX_IMAGE_MINUS_ONE_ONE) return v * 2.0f - 1.0f;
    return v;
}

int volvox_load_image_to_tensor(const char* path, float* dst, const int* shape, int ndim,
                                int normalize, char* err, size_t err_size) {
    if (!dst || !shape) { set_err(err, err_size, "missing output tensor"); return -1; }
    int c = 0, h = 0, w = 0;
    if (ndim == 4) {
        if (shape[0] != 1) { set_err(err, err_size, "image inputs require batch size 1"); return -1; }
        if (shape[3] == 1 || shape[3] == 3) {
            h = shape[1]; w = shape[2]; c = shape[3];
        } else {
            set_err(err, err_size, "image input tensor must be [1,H,W,C]");
            return -1;
        }
    } else if (ndim == 3) {
        h = shape[0]; w = shape[1]; c = shape[2];
    } else {
        set_err(err, err_size, "image input tensor must be [1,H,W,C] or [H,W,C]");
        return -1;
    }
    if (!((c == 1 || c == 3) && h > 0 && w > 0)) {
        set_err(err, err_size, "image tensor channel count must be 1 or 3");
        return -1;
    }

    ImageRGB img;
    if (load_image_rgb(path, &img, err, err_size) != 0) return -1;

    for (int oy = 0; oy < h; oy++) {
        float sy = ((float)oy + 0.5f) * (float)img.h / (float)h - 0.5f;
        for (int ox = 0; ox < w; ox++) {
            float sx = ((float)ox + 0.5f) * (float)img.w / (float)w - 0.5f;
            if (c == 1) {
                float r = sample_channel(&img, sx, sy, 0);
                float g = sample_channel(&img, sx, sy, 1);
                float b = sample_channel(&img, sx, sy, 2);
                float y = 0.299f * r + 0.587f * g + 0.114f * b;
                dst[(size_t)oy * w + ox] = normalize_value(y, normalize);
            } else {
                for (int oc = 0; oc < 3; oc++) {
                    float v = normalize_value(sample_channel(&img, sx, sy, oc), normalize);
                    dst[((size_t)oy * w + ox) * 3 + oc] = v;
                }
            }
        }
    }

    free_image_rgb(&img);
    return 0;
}
